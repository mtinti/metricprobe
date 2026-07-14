"""Step 5 goldens: dual lag (metric 3), batch metrics (metric 4), the
compare_event_time side-stat, and the "complete back to" headline date."""

import dataclasses
import math
import statistics

import pandas as pd
import pytest
from tests.support import probe, probe_dual, table_config
from tests.synth import generator as g
from tests.synth.scenarios import catalog

from metricprobe.metrics.batch import assess_batch
from metricprobe.metrics.completion import (
    assess_completion,
    compare_mismatch_by_month,
    complete_back_to,
)
from metricprobe.metrics.dual_lag import assess_dual_lag
from metricprobe.status import ReasonCode, Severity

AS_OF = "2026-07-01"

DUAL = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=12,
    rows_per_month=20_000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8),
    dual_offset_days=2.0,  # the known source->load offset
    seed=81,
)

TOLERANCE_DAYS = {50: 1, 90: 1, 95: 1, 99: 2}  # same rationale as completion goldens


def _dual_config(**overrides):
    return table_config(source_insert_time="source_insert_time", **overrides)


# ----------------------------------------------------------------------- dual


def test_dual_source_and_load_percentiles_recovered_from_known_offset():
    # "the same curves computed on each": BOTH sides of the dual table must
    # recover their generator-derived percentiles for the same scenario
    config = _dual_config()
    df = g.generate(DUAL)
    dual = probe_dual(df, config, AS_OF)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    canonical = probe(df, config, AS_OF)
    load_side = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    expected_source = g.expected_days_to_percentiles(DUAL, timestamp="source")
    expected_load = g.expected_days_to_percentiles(DUAL, timestamp="load")
    failures = []
    for label, actual, expected in (
        ("source", assessment.source_percentiles, expected_source),
        ("load", load_side.percentiles, expected_load),
    ):
        for month, per_pct in actual.items():
            for pct, got in per_pct.items():
                want = expected[str(month)][pct]
                if got.over_cap or abs(got.value - want) > TOLERANCE_DAYS[pct]:
                    failures.append(f"{label} {month} p{pct}: expected {want}, got {got}")
    assert not failures, "; ".join(failures)


def test_dual_clip_cap_and_negative_policy():
    config = _dual_config()
    df = g.generate(DUAL)
    # corrupt 1% of source stamps to 5 days BEFORE the event: beyond the 1-day
    # tolerance, they must be EXCLUDED (never clipped into the curves) and trip
    # the RED excess; within-tolerance skew (-1d) must clip to lag 0
    rng_rows = df.sample(frac=0.01, random_state=3).index
    corrupted = df.copy()
    corrupted.loc[rng_rows, "source_insert_time"] = corrupted.loc[
        rng_rows, "event_time"
    ] - pd.Timedelta(days=5)
    clipped_rows = df.sample(frac=0.02, random_state=4).index.difference(rng_rows)
    corrupted.loc[clipped_rows, "source_insert_time"] = corrupted.loc[
        clipped_rows, "event_time"
    ] - pd.Timedelta(days=1)
    dual = probe_dual(corrupted, config, AS_OF)
    row = dual.global_row
    assert int(row["n_negative_lag_excluded"]) == len(rng_rows)
    assert int(row["n_negative_clipped"]) == len(clipped_rows)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    assert any(
        s.severity is Severity.RED and s.reason is ReasonCode.NEGATIVE_LAG_EXCESS
        for s in assessment.statuses
    )
    # over-cap censoring on the SOURCE curve: a 15-day cap censors p95, the
    # value is reported ONLY as "> cap", and the check must not read GREEN
    tight = _dual_config(analysis={"lag_cap_days": 15})
    censored = assess_dual_lag(probe_dual(df, tight, AS_OF), tight, pd.Timestamp(AS_OF))
    month = next(iter(censored.source_percentiles))
    assert censored.source_percentiles[month][95].over_cap
    assert int(probe_dual(df, tight, AS_OF).global_row["n_overflow"]) > 0
    over_cap = [s for s in censored.statuses if s.reason is ReasonCode.PERCENTILE_OVER_CAP]
    assert over_cap and over_cap[0].severity is Severity.INSUFFICIENT_HISTORY
    assert not any(s.severity is Severity.GREEN for s in censored.statuses)


def test_dual_supports_event_time_via():
    # the borrowed-event dual pass: base carries keys + source + load, the
    # lookup owns the event time — source percentiles must match the direct run
    config = _dual_config()
    df = g.generate(DUAL)
    base = pd.DataFrame(
        {
            "referral_id": df["row_id"],
            "load_time": df["load_time"],
            "source_insert_time": df["source_insert_time"],
        }
    )
    lookup = pd.DataFrame({"id": df["row_id"], "referral_date": df["event_time"]})
    via_config = table_config(
        event_time=None,
        source_insert_time="source_insert_time",
        event_time_via={
            "join_table": "memory.main.referrals",
            "on": [{"base_col": "referral_id", "lookup_col": "id"}],
            "column": "referral_date",
        },
    )
    import sqlalchemy as sa

    from metricprobe.extract.dual import run_dual_lag

    engine = sa.create_engine("duckdb:///:memory:")
    try:
        g.load_via_sqlalchemy(base, engine, "events")
        g.load_via_sqlalchemy(lookup, engine, "referrals")
        via_dual = run_dual_lag(engine, via_config, pd.Timestamp(AS_OF))
    finally:
        engine.dispose()
    direct = assess_dual_lag(probe_dual(df, config, AS_OF), config, pd.Timestamp(AS_OF))
    borrowed = assess_dual_lag(via_dual, via_config, pd.Timestamp(AS_OF))
    assert borrowed.source_percentiles == direct.source_percentiles
    assert borrowed.delta_histogram.equals(direct.delta_histogram)


def test_dual_via_asserts_lookup_uniqueness_itself():
    # the dual pass runs on its OWN connection: the main pass's guard cannot
    # protect it against lookup mutation in between, so a duplicated lookup
    # key must abort the dual pass, never silently multiply source rows
    import sqlalchemy as sa

    from metricprobe.extract.canonical import ProbeAborted
    from metricprobe.extract.dual import run_dual_lag

    df = g.generate(DUAL)
    base = pd.DataFrame(
        {
            "referral_id": df["row_id"],
            "load_time": df["load_time"],
            "source_insert_time": df["source_insert_time"],
        }
    )
    lookup = pd.DataFrame({"id": df["row_id"], "referral_date": df["event_time"]})
    duplicated = pd.concat([lookup, lookup.head(5)], ignore_index=True)
    via_config = table_config(
        event_time=None,
        source_insert_time="source_insert_time",
        event_time_via={
            "join_table": "memory.main.referrals",
            "on": [{"base_col": "referral_id", "lookup_col": "id"}],
            "column": "referral_date",
        },
    )
    engine = sa.create_engine("duckdb:///:memory:")
    try:
        g.load_via_sqlalchemy(base, engine, "events")
        g.load_via_sqlalchemy(duplicated, engine, "referrals")
        with pytest.raises(ProbeAborted) as excinfo:
            run_dual_lag(engine, via_config, pd.Timestamp(AS_OF))
    finally:
        engine.dispose()
    assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE
    # the dual abort MEASURES its fanout exactly like the main pass
    assert excinfo.value.staged_rows is not None
    assert excinfo.value.staged_rows > len(base)
    # and the degenerate corner: duplicated lookup keys that match NO base
    # row still abort (the FULL OUTER guard stages every lookup row)
    disjoint = pd.DataFrame(
        {"id": [-77, -77], "referral_date": [pd.Timestamp("2024-01-01")] * 2}
    )
    engine = sa.create_engine("duckdb:///:memory:")
    try:
        g.load_via_sqlalchemy(base, engine, "events")
        g.load_via_sqlalchemy(disjoint, engine, "referrals")
        with pytest.raises(ProbeAborted) as excinfo:
            run_dual_lag(engine, via_config, pd.Timestamp(AS_OF))
    finally:
        engine.dispose()
    assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE
    assert excinfo.value.staged_rows is not None  # measured even when disjoint


def test_dual_delta_histogram_is_the_exact_offset():
    config = _dual_config()
    df = g.generate(DUAL)
    dual = probe_dual(df, config, AS_OF)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    # load = source + exactly 2.0 days: every delta-eligible row sits at day 2
    assert list(assessment.delta_histogram.index) == [2]
    assert int(assessment.delta_histogram.loc[2]) == len(df)
    assert assessment.n_delta_rows == len(df)


def test_dual_null_source_rows_form_their_own_bucket():
    config = _dual_config()
    df = g.inject_null_source_insert(g.generate(DUAL), fraction=0.03, seed=1)
    injected = round(0.03 * len(g.generate(DUAL)))
    dual = probe_dual(df, config, AS_OF)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    assert assessment.n_null_source_only == injected
    # dual reconciliation: total = eligible + null_event + null_source + excluded
    row = dual.global_row
    assert int(row["row_count"]) == (
        int(row["n_source_eligible"])
        + int(row["n_null_event_time"])
        + int(row["n_null_source_only"])
        + int(row["n_negative_lag_excluded"])
    )
    assert not any(s.severity is Severity.RED for s in assessment.statuses)


# ---------------------------------------------------------------------- batch


def test_batch_completion_hand_calculated_from_the_schedule():
    # schedule (3d, 60%), (10d, 30%), (20d, 10%) after month end:
    # cumulative .6 / .9 / 1.0 -> p50=3, p90=10, p95=20, p99=20 for EVERY month
    config = table_config(load_batch_col="batch_id")
    canonical = probe(catalog()["straggler_batch"].healthy(), config, AS_OF)
    assessment = assess_batch(canonical, config)
    failures = []  # accumulate ALL failing months, never die on the first
    for month_row in assessment.months:
        # p90 sits EXACTLY at the .6+.3 cumulative boundary: multinomial
        # assignment makes it genuinely bistable between the 10d and 20d batch
        checks = {
            "runs": month_row.runs == 3,
            "p50": month_row.days_to[50] == 3,
            "p90": month_row.days_to[90] in (10, 20),
            "p95": month_row.days_to[95] == 20,
            "p99": month_row.days_to[99] == 20,
            # rows per run are weighted by actual batch sizes (~fractions x 2000)
            "largest_run": abs(max(month_row.rows_per_run.values()) - 1200) <= 180,
            "total": sum(month_row.rows_per_run.values()) == 2000,
        }
        for name, ok in checks.items():
            if not ok:
                failures.append(f"{month_row.month} {name}: {month_row}")
    assert not failures, "; ".join(failures)


def test_straggler_batch_shifts_batch_level_completion():
    config = table_config(load_batch_col="batch_id")
    canonical = probe(catalog()["straggler_batch"].unhealthy(), config, AS_OF)
    assessment = assess_batch(canonical, config)
    by_month = {m.month: m for m in assessment.months}
    hit = by_month[pd.Period("2024-11", freq="M")]
    # 15% of the month rides the 45-day straggler: cum ~.51/.765/.85/1.0
    assert hit.runs == 4
    assert hit.days_to[50] in (3, 10)  # cum at 3d is ~.51: within 1 sigma of .5
    assert hit.days_to[90] == 45 and hit.days_to[95] == 45
    clean = by_month[pd.Period("2024-10", freq="M")]
    assert clean.days_to[50] == 3 and clean.days_to[95] == 20
    assert clean.days_to[90] in (10, 20)  # the exact-boundary bistability again


def test_null_batch_ids_are_counted_and_hold_the_curve_back():
    # NULL out the batch id on ~8% of one month's rows: they must stay in the
    # completion denominator (silently dropping them would overstate
    # completion), be reported AMBER, and make unreachable percentiles None
    df = catalog()["straggler_batch"].healthy()
    month_rows = df["event_time"].dt.to_period("M") == "2024-06"
    nulled = df.copy()
    chosen = df[month_rows].sample(frac=0.08, random_state=6).index
    nulled.loc[chosen, "batch_id"] = None
    config = table_config(load_batch_col="batch_id")
    assessment = assess_batch(probe(nulled, config, AS_OF), config)
    by_month = {m.month: m for m in assessment.months}
    june = by_month[pd.Period("2024-06", freq="M")]
    assert june.null_batch_rows == len(chosen)
    assert sum(june.rows_per_run.values()) + june.null_batch_rows == 2000
    # ~92% attributed: p95/p99 are unreachable, p50 still fine
    assert june.days_to[50] == 3
    assert june.days_to[95] is None and june.days_to[99] is None
    amber = [s for s in assessment.statuses if s.reason is ReasonCode.NULL_BATCH_IDS]
    assert amber and amber[0].severity is Severity.AMBER
    assert str(len(chosen)) in amber[0].detail


# ----------------------------------------------------- compare_event_time stat


def test_compare_event_time_side_stat():
    pair = catalog()["raw_vs_corrected"]
    config = table_config(compare_event_time="event_time_raw")
    healthy = compare_mismatch_by_month(probe(pair.healthy(), config, AS_OF))
    assert sum(healthy.values()) == 0
    unhealthy_df = pair.unhealthy()
    unhealthy = compare_mismatch_by_month(probe(unhealthy_df, config, AS_OF))
    # day-grain mismatches among eligible rows, by month, summing to the total
    expected_total = int(
        (
            unhealthy_df["event_time_raw"].dt.normalize()
            != unhealthy_df["event_time"].dt.normalize()
        ).sum()
    )
    assert sum(unhealthy.values()) == expected_total
    assert expected_total == round(0.08 * 30 * 2000)


# ------------------------------------------------------------------- headline


def test_complete_back_to_ties_to_the_completion_percentiles():
    spec = dataclasses.replace(DUAL, dual_offset_days=None, seed=82)
    config = table_config()
    canonical = probe(g.generate(spec), config, AS_OF)
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    assert assessment.recommended_wait is not None
    # the headline formula, recomputed independently from the percentiles
    p95s = [assessment.percentiles[m][95].value for m in assessment.mature_months]
    expected_wait = math.ceil(statistics.fmean(p95s) + 2 * statistics.pstdev(p95s))
    assert assessment.recommended_wait == expected_wait
    date = complete_back_to(assessment, pd.Timestamp(AS_OF))
    assert date == pd.Timestamp(AS_OF) - pd.Timedelta(days=expected_wait)
    # and it is deliberately DISTINCT from the (stricter) maturity horizon
    assert assessment.horizon >= expected_wait


def test_complete_back_to_is_refused_when_the_wait_is():
    config = table_config(analysis={"lag_cap_days": 15})  # censors p95
    spec = dataclasses.replace(DUAL, dual_offset_days=None, seed=82)
    canonical = probe(g.generate(spec), config, AS_OF)
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    assert assessment.recommended_wait is None
    assert complete_back_to(assessment, pd.Timestamp(AS_OF)) is None
    assert any(s.reason is ReasonCode.PERCENTILE_OVER_CAP for s in assessment.statuses)


def test_month_with_only_null_batch_ids_still_appears():
    # NULL out EVERY batch id in one month: the month must not vanish — it
    # reports zero runs, its full NULL count, and unreachable percentiles
    df = catalog()["straggler_batch"].healthy()
    month_rows = df["event_time"].dt.to_period("M") == "2024-06"
    nulled = df.copy()
    nulled.loc[month_rows, "batch_id"] = None
    config = table_config(load_batch_col="batch_id")
    assessment = assess_batch(probe(nulled, config, AS_OF), config)
    by_month = {m.month: m for m in assessment.months}
    june = by_month[pd.Period("2024-06", freq="M")]
    assert june.runs == 0 and june.rows_per_run == {}
    assert june.null_batch_rows == int(month_rows.sum()) == 2000
    assert all(value is None for value in june.days_to.values())
    # the neighbouring months are untouched
    assert by_month[pd.Period("2024-05", freq="M")].runs == 3
    amber = [s for s in assessment.statuses if s.reason is ReasonCode.NULL_BATCH_IDS]
    assert amber and amber[0].severity is Severity.AMBER


def test_batch_canonical_timestamp_includes_null_event_arrivals():
    """A batch's canonical timestamp is MIN(load_time) over ALL its arrivals:
    when its earliest wave carries corrupt (NULL) event times, that wave still
    dates the batch — dropping it would shift every day count derived from it."""
    df = catalog()["straggler_batch"].healthy()
    config = table_config(load_batch_col="batch_id")
    baseline = assess_batch(probe(df, config, AS_OF), config)
    june_before = {m.month: m for m in baseline.months}[pd.Period("2024-06", freq="M")]
    assert june_before.days_to[50] == 3  # the day-3 batch dates the median

    june = df[df["event_time"].dt.to_period("M") == "2024-06"]
    first_batch = june.loc[june["load_time"].idxmin(), "batch_id"]
    corrupt = june[june["batch_id"] == first_batch].head(5).copy()
    corrupt["event_time"] = pd.NaT
    corrupt["load_time"] = pd.Timestamp("2024-07-02")  # day 1 after month end
    with_early_wave = pd.concat([df, corrupt], ignore_index=True)
    assessment = assess_batch(probe(with_early_wave, config, AS_OF), config)
    june_after = {m.month: m for m in assessment.months}[pd.Period("2024-06", freq="M")]
    # the same batch now dates from its EARLIER (null-event) arrival wave
    assert june_after.days_to[50] == 1


def test_dual_empty_table_still_yields_the_global_row():
    """The dual () row must exist over an EMPTY population (it carries
    max_lookup_dup and the zero buckets). SQL Server returns NO rows for
    GROUPING SETS including () over empty input, so the global row is an
    ungrouped UNION ALL branch (v9) — this pins the duckdb half; container
    equivalence pins the mssql half."""
    empty = pd.DataFrame(
        {
            "event_time": pd.Series([], dtype="datetime64[ns]"),
            "load_time": pd.Series([], dtype="datetime64[ns]"),
            "source_insert_time": pd.Series([], dtype="datetime64[ns]"),
        }
    )
    result = probe_dual(empty, _dual_config(), "2026-07-01")
    row = result.global_row
    assert int(row["row_count"]) == 0
    assert int(row["n_staged_rows"]) == 0
