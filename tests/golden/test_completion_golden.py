"""Golden tests (DuckDB): the pipeline must recover the percentiles derivable
from the generator parameters, honor the as-of watermark and NULL buckets,
refuse censored percentiles, apply the negative-lag policy, and abort on caps.
Assertion style: accumulate ALL failing cases into one message."""

import dataclasses
import functools

import pandas as pd
import pytest
from tests.support import probe, table_config
from tests.synth import generator as g

from metricprobe.extract.canonical import ProbeAborted
from metricprobe.metrics.completion import assess_completion
from metricprobe.status import ReasonCode, Severity

TRICKLE = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=12,
    rows_per_month=20_000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8),  # p50 ~5d, p95 ~19d, p99 ~32d
    seed=31,
)

BATCHY = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=12,
    rows_per_month=20_000,
    lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
    seed=32,
)

AS_OF = "2026-07-01"  # all 2024 months long mature by then

# Tolerances per percentile, in days. At N=20k/month the day-grain quantile
# std-error is ~0.2d at p95 and ~0.6d at p99 (tail count ~200), so +-1 covers
# >3 sigma everywhere except p99, which gets +-2.
TOLERANCE_DAYS = {50: 1, 90: 1, 95: 1, 99: 2}


@functools.cache
def trickle_df() -> pd.DataFrame:
    return g.generate(TRICKLE)


def _assert_golden_percentiles(spec: g.TableSpec, config, as_of=AS_OF):
    canonical = probe(g.generate(spec), config, as_of)
    assessment = assess_completion(canonical, config, pd.Timestamp(as_of))
    expected = g.expected_days_to_percentiles(spec)  # day grain, the SQL semantics
    failures = []
    for month, per_pct in assessment.percentiles.items():
        for pct, got in per_pct.items():
            want = expected[str(month)][pct]
            if got.over_cap or abs(got.value - want) > TOLERANCE_DAYS[pct]:
                failures.append(f"{month} p{pct}: expected {want}, got {got}")
    assert not failures, "; ".join(failures)
    return assessment


def test_trickle_percentiles_recovered():
    assessment = _assert_golden_percentiles(TRICKLE, table_config())
    # p95 ~19d over 12 near-identical months => wait = ceil(mean + 2*pstdev) ~ 19-22
    assert assessment.recommended_wait is not None
    assert 17 <= assessment.recommended_wait <= 24


def test_batchy_percentiles_recovered():
    assessment = _assert_golden_percentiles(BATCHY, table_config())
    assert assessment.recommended_wait is not None


def test_null_buckets_survive_the_as_of_predicate():
    df = trickle_df()
    df = g.inject_null_event_time(df, 0.02, seed=1)
    df = g.inject_null_load_time(df, 0.03, seed=2)
    n_null_event = int(df["event_time"].isna().sum())
    # null-event rows keep their load_time; null-load only counts event-present rows
    n_null_load_only = int((df["event_time"].notna() & df["load_time"].isna()).sum())
    canonical = probe(df, table_config(), AS_OF)
    row = canonical.global_row
    assert int(row["n_null_event_time"]) == n_null_event
    assert int(row["n_null_load_time_only"]) == n_null_load_only
    assert int(row["row_count"]) == len(df)


def test_as_of_watermark_excludes_later_loads():
    df = trickle_df()
    as_of = pd.Timestamp("2024-07-15")
    admitted = int(((df["load_time"] <= as_of) | df["load_time"].isna()).sum())
    canonical = probe(df, table_config(), as_of)
    assert int(canonical.global_row["row_count"]) == admitted


def test_censoring_twins_overflow_mass_vs_percentile():
    # UNHEALTHY twin: cap 15d; lognormal(1.6, 0.8) has ~8% mass beyond 15d,
    # more than (1 - 0.95) => p95 must report "> cap" and the wait be REFUSED.
    tight = table_config(analysis={"lag_cap_days": 15})
    canonical = probe(trickle_df(), tight, AS_OF)
    assessment = assess_completion(canonical, tight, pd.Timestamp(AS_OF))
    month = next(iter(assessment.percentiles))
    assert assessment.percentiles[month][95].over_cap
    assert assessment.percentiles[month][95].value is None
    assert assessment.learned_wait is None
    assert assessment.recommended_wait is None
    assert any(s.reason is ReasonCode.PERCENTILE_OVER_CAP for s in assessment.statuses)

    # HEALTHY twin: cap 60d; mass beyond 60d ~0.1% < 1% => p99 still a number.
    wide = table_config(analysis={"lag_cap_days": 60, "training_cutoff_days": 365})
    canonical = probe(trickle_df(), wide, AS_OF)
    assessment = assess_completion(canonical, wide, pd.Timestamp(AS_OF))
    for pct in (50, 90, 95, 99):
        assert not assessment.percentiles[month][pct].over_cap
    assert assessment.recommended_wait is not None
    assert not any(s.reason is ReasonCode.PERCENTILE_OVER_CAP for s in assessment.statuses)


def test_negative_lag_within_tolerance_is_clipped():
    df = g.inject_negative_lags(trickle_df(), fraction=0.05, skew_days=1.0, seed=3)
    config = table_config()  # clock_skew_tolerance_days default 1.0
    canonical = probe(df, config, AS_OF)
    row = canonical.global_row
    injected = round(0.05 * len(trickle_df()))
    assert int(row["n_negative_clipped"]) == injected
    assert int(row["n_negative_lag_excluded"]) == 0
    # clipped rows stay curve-eligible at lag 0
    assert int(row["n_curve_eligible"]) == len(df)
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    assert not any(s.reason is ReasonCode.NEGATIVE_LAG_EXCESS for s in assessment.statuses)


def test_negative_lag_beyond_tolerance_is_excluded_and_red():
    df = g.inject_negative_lags(trickle_df(), fraction=0.05, skew_days=3.0, seed=3)
    config = table_config()
    canonical = probe(df, config, AS_OF)
    row = canonical.global_row
    injected = round(0.05 * len(df))
    assert int(row["n_negative_lag_excluded"]) == injected
    assert int(row["n_curve_eligible"]) == len(df) - injected
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    assert assessment.negative_lag_excess_fraction == pytest.approx(0.05, abs=0.001)
    assert any(
        s.severity is Severity.RED and s.reason is ReasonCode.NEGATIVE_LAG_EXCESS
        for s in assessment.statuses
    )


def test_trap_corrupt_negatives_must_not_improve_percentiles():
    # If corrupt rows were CLIPPED instead of excluded they would pile up at lag
    # 0 and pull every percentile DOWN. Excluded, the remaining distribution is
    # unchanged, so percentiles must not improve beyond sampling jitter.
    config = table_config()
    clean = assess_completion(
        probe(trickle_df(), config, AS_OF), config, pd.Timestamp(AS_OF)
    )
    corrupt_df = g.inject_negative_lags(trickle_df(), fraction=0.10, skew_days=5.0, seed=4)
    corrupt = assess_completion(probe(corrupt_df, config, AS_OF), config, pd.Timestamp(AS_OF))
    failures = []
    for month, per_pct in clean.percentiles.items():
        for pct, clean_value in per_pct.items():
            corrupt_value = corrupt.percentiles[month][pct]
            if corrupt_value.value < clean_value.value - 1:
                failures.append(
                    f"{month} p{pct}: corrupt {corrupt_value.value} improved on "
                    f"clean {clean_value.value}"
                )
    assert not failures, "; ".join(failures)


def test_result_cell_cap_aborts_the_probe():
    config = table_config(analysis={"result_cell_cap": 10})
    with pytest.raises(ProbeAborted) as excinfo:
        probe(trickle_df(), config, AS_OF)
    assert excinfo.value.reason is ReasonCode.RESULT_CELL_CAP_EXCEEDED


# ------------------------------------------------------------------- via joins


def _via_frames(spec=TRICKLE):
    """Base table carries only keys+load; the lookup owns the event time."""
    df = g.generate(spec)
    base = pd.DataFrame(
        {"referral_id": df["row_id"], "site_code": df["row_id"] % 7, "load_time": df["load_time"]}
    )
    lookup = pd.DataFrame(
        {"id": df["row_id"], "site": df["row_id"] % 7, "referral_date": df["event_time"]}
    )
    return df, base, lookup


def _via_config(join_on, **overrides):
    return table_config(
        event_time=None,
        event_time_via={
            "join_table": "memory.main.referrals",
            "on": join_on,
            "column": "referral_date",
        },
        **overrides,
    )


def _probe_via(base, lookup, config, as_of=AS_OF):
    from tests.support import engine_with

    from metricprobe.extract.canonical import run_canonical

    engine = engine_with(base, "events")
    g.load_via_sqlalchemy(lookup, engine, "referrals")
    try:
        return run_canonical(engine, config, pd.Timestamp(as_of))
    finally:
        engine.dispose()


def test_via_join_matches_direct_probe():
    # borrowing the event time through a differently-named key must reproduce
    # the direct probe's month/lag cells exactly
    df, base, lookup = _via_frames()
    direct = probe(df, table_config(), AS_OF).rows_for("month_lag")
    via = _probe_via(base, lookup, _via_config([{"base_col": "referral_id", "lookup_col": "id"}]))
    cells = via.rows_for("month_lag")
    key = ["event_month", "lag_day"]
    pd.testing.assert_frame_equal(
        direct[key + ["row_count"]].sort_values(key).reset_index(drop=True),
        cells[key + ["row_count"]].sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


def test_via_composite_key_matches_direct_probe():
    df, base, lookup = _via_frames()
    via = _probe_via(
        base,
        lookup,
        _via_config(
            [
                {"base_col": "referral_id", "lookup_col": "id"},
                {"base_col": "site_code", "lookup_col": "site"},
            ]
        ),
    )
    direct = probe(df, table_config(), AS_OF)
    assert int(via.global_row["n_curve_eligible"]) == int(direct.global_row["n_curve_eligible"])
    assert int(via.global_row["n_join_unmatched"]) == 0


def test_via_unmatched_rows_are_counted_and_reconciled():
    _, base, lookup = _via_frames()
    lookup_missing = lookup.iloc[100:]  # 100 base rows lose their lookup match
    nulled = lookup_missing.copy()
    nulled.iloc[:50, nulled.columns.get_loc("referral_date")] = pd.NaT  # matched, column NULL
    via = _probe_via(
        base, nulled, _via_config([{"base_col": "referral_id", "lookup_col": "id"}])
    )
    row = via.global_row
    assert int(row["n_join_unmatched"]) == 100
    assert int(row["n_null_event_time"]) == 50
    # pre/post reconciliation: unique lookup => post-join count == base count
    assert int(row["row_count"]) == len(base)
    assert int(row["n_base_rows"]) == len(base)
    assert int(row["n_ambiguous_base_rows"]) == 0


def test_via_non_unique_lookup_fails_loudly():
    _, base, lookup = _via_frames()
    duplicated = pd.concat([lookup, lookup.iloc[:5]], ignore_index=True)
    with pytest.raises(ProbeAborted) as excinfo:
        _probe_via(base, duplicated, _via_config([{"base_col": "referral_id", "lookup_col": "id"}]))
    assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE
    # ambiguous base rows are counted and reported in the abort detail
    assert "5 base rows are ambiguous" in excinfo.value.detail


def test_via_lookup_duplicates_abort_even_when_unreferenced():
    """The uniqueness contract covers the LOOKUP TABLE, not just the keys a
    base row happens to reference: duplicate lookup keys that join to no
    current base row must still abort the probe."""
    _, base, lookup = _via_frames()
    stranger = pd.DataFrame(
        {
            "id": [-1, -1],  # a duplicated key NO base row references
            "site": [0, 0],
            "referral_date": [pd.Timestamp("2024-01-01")] * 2,
        }
    )
    duplicated = pd.concat([lookup, stranger], ignore_index=True)
    with pytest.raises(ProbeAborted) as excinfo:
        _probe_via(
            base, duplicated, _via_config([{"base_col": "referral_id", "lookup_col": "id"}])
        )
    assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE
    assert "0 base rows are ambiguous" in excinfo.value.detail


def test_via_lookup_duplicates_survive_the_watermark():
    """Duplicate lookup keys whose ONLY base matches are loaded AFTER as_of
    must still abort: the watermark applies to the base side BEFORE the join,
    so filtered-out matches leave the lookup rows as guard-only artifacts
    instead of deleting them."""
    _, base, lookup = _via_frames()
    future = base.head(1).copy()
    future["referral_id"] = 987_654
    future["site_code"] = 0
    future["load_time"] = pd.Timestamp("2099-01-01")  # beyond the as_of watermark
    duplicated = pd.DataFrame(
        {
            "id": [987_654, 987_654],
            "site": [0, 0],
            "referral_date": [pd.Timestamp("2024-01-01")] * 2,
        }
    )
    with pytest.raises(ProbeAborted) as excinfo:
        _probe_via(
            pd.concat([base, future], ignore_index=True),
            pd.concat([lookup, duplicated], ignore_index=True),
            _via_config([{"base_col": "referral_id", "lookup_col": "id"}]),
        )
    assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE


def test_via_lookup_duplicates_abort_with_zero_matched_base_rows():
    """The degenerate corner: EVERY admitted base row is unmatched (the
    duplicated lookup keys join to nothing at all) — the FULL OUTER guard
    still stages the lookup rows and the probe still refuses."""
    _, base, lookup = _via_frames()
    disjoint = pd.DataFrame(
        {
            "id": [99_999_999, 99_999_999],  # duplicated, matches NO base row
            "site": [0, 0],
            "referral_date": [pd.Timestamp("2024-01-01")] * 2,
        }
    )
    with pytest.raises(ProbeAborted) as excinfo:
        _probe_via(
            base, disjoint, _via_config([{"base_col": "referral_id", "lookup_col": "id"}])
        )
    assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE


def test_via_percentiles_match_generator_expectations():
    df, base, lookup = _via_frames()
    config = _via_config([{"base_col": "referral_id", "lookup_col": "id"}])
    canonical = _probe_via(base, lookup, config)
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    expected = g.expected_days_to_percentiles(TRICKLE)
    failures = []
    for month, per_pct in assessment.percentiles.items():
        for pct, got in per_pct.items():
            want = expected[str(month)][pct]
            if got.over_cap or abs(got.value - want) > TOLERANCE_DAYS[pct]:
                failures.append(f"{month} p{pct}: expected {want}, got {got}")
    assert not failures, "; ".join(failures)


def test_insufficient_history_refuses_the_mature_summary():
    """Below min_mature_months, the mature percentile summary is refused
    (all None) exactly like recommended_wait — a mean over 2 months must
    never surface as a confident published number."""
    spec = g.TableSpec(
        name="events", start_month="2024-01", n_months=15, rows_per_month=1000,
        lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=88,
    )
    config = table_config(analysis={"lag_cap_days": 300, "training_cutoff_days": 365})
    # as_of leaves only ~2 mature months (< the default minimum of 6)
    assessment = assess_completion(
        probe(g.generate(spec), config, "2025-04-10"), config, pd.Timestamp("2025-04-10")
    )
    assert 0 < len(assessment.mature_months) < 6
    assert assessment.recommended_wait is None
    assert all(v is None for v in assessment.mature_percentile_summary.values())
    assert any(
        s.reason is ReasonCode.INSUFFICIENT_MATURE_MONTHS for s in assessment.statuses
    )
    # ...and with enough mature months the summary IS published
    grown = dataclasses.replace(spec, n_months=30)
    sufficient = assess_completion(
        probe(g.generate(grown), config, "2026-07-01"), config, pd.Timestamp("2026-07-01")
    )
    assert sufficient.mature_percentile_summary[95] is not None


def test_lookup_only_guard_rows_create_no_alt_cells():
    """Unreferenced lookup rows exist ONLY for the uniqueness guard: they
    must not surface as a bogus NULL-alt cell nor count against
    result_cell_cap. A minimal world where EVERY real grouping set has
    exactly two cells, probed under cap 2: one unreferenced lookup row must
    change nothing."""
    base = pd.DataFrame(
        {
            "referral_id": [1, 2, 3, 4],
            "region": ["north", "south", "north", "south"],
            "load_time": pd.to_datetime(
                ["2024-01-05", "2024-01-05", "2024-01-06", "2024-01-06"]
            ),
        }
    )
    lookup = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 77_777_777],  # the last row matches NO base row
            "referral_date": pd.to_datetime(["2024-01-04"] * 4 + ["2024-01-01"]),
        }
    )
    config = _via_config(
        [{"base_col": "referral_id", "lookup_col": "id"}],
        group_by_alt="region",
        analysis={"result_cell_cap": 2},  # exactly the per-set real cell count
    )
    result = _probe_via(base, lookup, config)  # no false cap abort
    alt_cells = result.rows_for("alt")
    assert sorted(alt_cells["alt_value"]) == ["north", "south"]  # no NULL ghost
    assert (alt_cells["row_count"] > 0).all()
    # the raw frame carries no guard-artifact alt cell either
    raw_alt = result.frame[result.frame["grouping_id"] == 30]
    assert not raw_alt["alt_value"].isna().any()
