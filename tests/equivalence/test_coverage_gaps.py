"""Step 7 gap closure: the equivalence cases the coverage audit found missing —
pathology scenarios, censoring/overflow, the hour bucket, NULL key hashing,
empty tables, cell-cap aborts, parity verdict variants and the dual via pass.
The explicit coverage matrix lives in tests/unit/test_equivalence_coverage.py."""

import dataclasses

import pandas as pd
import pytest
from tests.equivalence.test_canonical_equivalence import _config
from tests.synth import generator as g

from metricprobe.extract.canonical import ProbeAborted, run_canonical
from metricprobe.extract.dual import run_dual_lag
from metricprobe.metrics.batch import assess_batch
from metricprobe.metrics.completion import assess_completion
from metricprobe.metrics.dual_lag import assess_dual_lag
from metricprobe.metrics.freshness import assess_freshness
from metricprobe.metrics.parity import ParitySide, assess_parity
from metricprobe.metrics.volume import assess_volume
from metricprobe.status import ReasonCode, Severity

TRICKLE_12 = g.TableSpec(
    name="events", start_month="2024-01", n_months=12, rows_per_month=2000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=201,
)

ENGINES = (("duckdb", "memory", "main"), ("mssql", "tempdb", "dbo"))


def _both(duckdb_engine, mssql_engine):
    return ((duckdb_engine, "memory", "main"), (mssql_engine, "tempdb", "dbo"))


def _assess_all(engine, config, as_of):
    canonical = run_canonical(engine, config, as_of)
    completion = assess_completion(canonical, config, as_of)
    volume = assess_volume(canonical, config, as_of, completion)
    freshness = assess_freshness(canonical, config, as_of)
    return canonical, completion, volume, freshness


def test_volume_pathology_scenarios_match(duckdb_engine, mssql_engine):
    """Spike, drop, missing month, within-tolerance clipping and a stale feed —
    identical typed verdicts from both engines, in BOTH twin directions: every
    check fires on the unhealthy twin AND stays silent on the healthy twin."""
    spec = g.volume_spike(TRICKLE_12, 3, factor=6.0)  # 2024-04
    spec = g.volume_drop(spec, 5, factor=0.1)  # 2024-06
    spec = g.missing_month(spec, 7)  # 2024-08
    df = g.inject_negative_lags(g.generate(spec), fraction=0.02, skew_days=1.0, seed=4)
    healthy_df = g.generate(TRICKLE_12)  # the healthy twin: same parameters
    as_of = pd.Timestamp("2026-07-01")  # all mature; the feed is long stale
    results = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(database, schema, load_batch_col=None, key_cols=None)
        g.load_via_sqlalchemy(df, engine, "events")
        canonical, completion, volume, freshness = _assess_all(engine, config, as_of)
        healthy_config = _config(
            database, schema, table="events_healthy", load_batch_col=None, key_cols=None
        )
        g.load_via_sqlalchemy(healthy_df, engine, "events_healthy")
        _, _, healthy_volume, _ = _assess_all(engine, healthy_config, as_of)
        results.append((canonical, volume, freshness, healthy_volume))
    (
        (duck_canonical, duck_volume, duck_fresh, duck_healthy),
        (ms_canonical, ms_volume, ms_fresh, ms_healthy),
    ) = results
    assert duck_volume.statuses == ms_volume.statuses
    assert duck_volume.months == ms_volume.months
    assert duck_volume.gaps == ms_volume.gaps
    assert duck_fresh == ms_fresh
    # the healthy twin stays silent on the volume check, identically
    assert duck_healthy.statuses == ms_healthy.statuses
    assert not [s for s in ms_healthy.statuses if s.severity is Severity.RED]
    assert not [s for s in ms_healthy.statuses if s.reason is ReasonCode.VOLUME_OUTLIER]
    # the spike and the drop are EACH detected (not either/or)
    outlier_details = "; ".join(
        s.detail for s in ms_volume.statuses if s.reason is ReasonCode.VOLUME_OUTLIER
    )
    assert "2024-04" in outlier_details  # the spike month
    assert "2024-06" in outlier_details  # the drop month
    gap_details = "; ".join(
        s.detail for s in ms_volume.statuses if s.reason is ReasonCode.VOLUME_GAP
    )
    assert "2024-08" in gap_details  # the missing month
    assert any(
        s.reason is ReasonCode.STALE_FEED and s.severity is Severity.RED
        for s in ms_fresh.statuses
    )
    # within-tolerance negatives CLIP identically (never excluded)
    assert int(duck_canonical.global_row["n_negative_clipped"]) == int(
        ms_canonical.global_row["n_negative_clipped"]
    )
    assert int(ms_canonical.global_row["n_negative_clipped"]) > 0
    assert int(ms_canonical.global_row["n_negative_lag_excluded"]) == 0


def test_censoring_overflow_and_hour_bucket_match(duckdb_engine, mssql_engine):
    """A 15-day lag cap censors p95 (overflow bucket populated) and the HOUR
    freshness bucket exercises the dialect-specific hour floor."""
    df = g.generate(dataclasses.replace(TRICKLE_12, n_months=8, seed=202))
    as_of = pd.Timestamp("2026-07-01")
    results = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(
            database,
            schema,
            load_batch_col=None,
            key_cols=None,
            analysis={"lag_cap_days": 15, "freshness_bucket": "hour"},
        )
        g.load_via_sqlalchemy(df, engine, "events")
        canonical, completion, _, freshness = _assess_all(engine, config, as_of)
        results.append((canonical, completion, freshness))
    (duck_canonical, duck_completion, duck_fresh), (ms_canonical, ms_completion, ms_fresh) = (
        results
    )
    assert duck_completion.percentiles == ms_completion.percentiles
    month = next(iter(ms_completion.percentiles))
    assert ms_completion.percentiles[month][95].over_cap  # censored, not a number
    assert ms_completion.recommended_wait is None
    assert int(ms_canonical.global_row["n_overflow"]) > 0
    assert int(duck_canonical.global_row["n_overflow"]) == int(
        ms_canonical.global_row["n_overflow"]
    )
    # hour-bucket epochs agree exactly — KEYS AND COUNTS, not just cardinality
    # (the mssql DATEADD/DATEDIFF hour floor vs duckdb date_trunc)
    def hour_cells(canonical):
        cells = canonical.rows_for("epoch")[["load_epoch_day", "row_count"]].copy()
        cells["load_epoch_day"] = pd.to_datetime(cells["load_epoch_day"])
        return cells.sort_values("load_epoch_day").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        hour_cells(duck_canonical), hour_cells(ms_canonical), check_dtype=False
    )
    assert duck_fresh == ms_fresh


def test_straggler_batch_scenario_matches(duckdb_engine, mssql_engine):
    spec = dataclasses.replace(
        TRICKLE_12,
        n_months=8,
        lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
        seed=203,
    )
    healthy_df = g.generate(spec)  # the healthy twin
    df = g.inject_straggler_batch(
        healthy_df, month="2024-04", late_day=45.0, fraction=0.15, seed=5
    )
    as_of = pd.Timestamp("2026-07-01")
    assessments = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(database, schema, key_cols=None)
        g.load_via_sqlalchemy(df, engine, "events")
        assessments.append(assess_batch(run_canonical(engine, config, as_of), config))
        healthy_config = _config(database, schema, table="events_healthy", key_cols=None)
        g.load_via_sqlalchemy(healthy_df, engine, "events_healthy")
        assessments.append(
            assess_batch(run_canonical(engine, healthy_config, as_of), healthy_config)
        )
    duck, duck_healthy, mssql, mssql_healthy = assessments
    assert duck.months == mssql.months
    assert duck.rows_per_run == mssql.rows_per_run
    hit = {m.month: m for m in mssql.months}[pd.Period("2024-04", freq="M")]
    assert hit.days_to[95] == 45  # the straggler moved batch-level completion
    # the healthy twin stays at the schedule's 20 days, identically
    assert duck_healthy.months == mssql_healthy.months
    clean = {m.month: m for m in mssql_healthy.months}[pd.Period("2024-04", freq="M")]
    assert clean.days_to[95] == 20


def test_empty_table_and_null_keys_match(duckdb_engine, mssql_engine):
    """Zero rows must reduce to the hard RED (COALESCE over an empty grouping)
    and NULL key values must hash through the distinct NULL sentinel."""
    empty = g.generate(dataclasses.replace(TRICKLE_12, n_months=1, rows_per_month=0))
    with_null_keys = g.generate(dataclasses.replace(TRICKLE_12, n_months=6, seed=204))
    with_null_keys["k"] = with_null_keys["row_id"].astype("Int64")
    with_null_keys.loc[with_null_keys.sample(frac=0.05, random_state=6).index, "k"] = pd.NA
    with_null_keys = pd.concat(
        [with_null_keys, with_null_keys.iloc[:25]], ignore_index=True
    )  # 25 true duplicates on top of the NULL-key rows
    as_of = pd.Timestamp("2026-07-01")
    outcomes = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        g.load_via_sqlalchemy(empty, engine, "events_empty")
        g.load_via_sqlalchemy(with_null_keys, engine, "events_nullkey")
        empty_config = _config(
            database, schema, table="events_empty", load_batch_col=None, key_cols=None
        )
        _, _, empty_volume, _ = _assess_all(engine, empty_config, as_of)
        key_config = _config(
            database, schema, table="events_nullkey", load_batch_col=None, key_cols=["k"]
        )
        canonical = run_canonical(engine, key_config, as_of)
        outcomes.append((empty_volume, int(canonical.global_row["distinct_keys"])))
    (duck_empty, duck_distinct), (ms_empty, ms_distinct) = outcomes
    assert duck_empty.statuses == ms_empty.statuses
    assert any(s.reason is ReasonCode.ZERO_ROW_MONTH for s in ms_empty.statuses)
    # 12000 unique keys - 600 NULLed + exactly ONE distinct NULL-sentinel value:
    # a hash that returned SQL NULL for NULL keys would count 11400, not 11401
    assert duck_distinct == ms_distinct == 11_401


def test_cell_cap_aborts_on_both_dialects(duckdb_engine, mssql_engine):
    df = g.generate(
        dataclasses.replace(TRICKLE_12, n_months=6, dual_offset_days=2.0, seed=205)
    )
    as_of = pd.Timestamp("2026-07-01")
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(
            database, schema, load_batch_col=None, key_cols=None,
            source_insert_time="source_insert_time",
            analysis={"result_cell_cap": 10},
        )
        g.load_via_sqlalchemy(df, engine, "events")
        with pytest.raises(ProbeAborted) as excinfo:
            run_canonical(engine, config, as_of)
        assert excinfo.value.reason is ReasonCode.RESULT_CELL_CAP_EXCEEDED
        # the DUAL pass has its own dialect-sensitive limit + cap loop
        with pytest.raises(ProbeAborted) as excinfo:
            run_dual_lag(engine, config, as_of)
        assert excinfo.value.reason is ReasonCode.RESULT_CELL_CAP_EXCEEDED


def test_dual_via_matches_between_dialects(duckdb_engine, mssql_engine):
    spec = dataclasses.replace(TRICKLE_12, n_months=6, dual_offset_days=2.0, seed=206)
    df = g.generate(spec)
    base = pd.DataFrame(
        {
            "referral_id": df["row_id"],
            "load_time": df["load_time"],
            "source_insert_time": df["source_insert_time"],
        }
    )
    lookup = pd.DataFrame({"id": df["row_id"], "referral_date": df["event_time"]})
    as_of = pd.Timestamp("2026-07-01")
    assessments = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(
            database,
            schema,
            event_time=None,
            load_batch_col=None,
            key_cols=None,
            source_insert_time="source_insert_time",
            event_time_via={
                "join_table": f"{database}.{schema}.referrals",
                "on": [{"base_col": "referral_id", "lookup_col": "id"}],
                "column": "referral_date",
            },
        )
        g.load_via_sqlalchemy(base, engine, "events")
        g.load_via_sqlalchemy(lookup, engine, "referrals")
        dual = run_dual_lag(engine, config, as_of)
        assessments.append(assess_dual_lag(dual, config, as_of))
    duck, mssql = assessments
    assert duck.source_percentiles == mssql.source_percentiles
    assert duck.delta_histogram.equals(mssql.delta_histogram)


def test_parity_mismatch_and_prereq_verdicts_match(duckdb_engine, mssql_engine):
    df = g.generate(dataclasses.replace(TRICKLE_12, n_months=10, seed=207))
    march = df["event_time"].dt.to_period("M") == "2024-03"
    short = df.drop(df[march].index[:50]).reset_index(drop=True)
    duplicated = g.inject_duplicate_keys(df, fraction=0.01, seed=7)
    as_of = pd.Timestamp("2026-07-01")

    def side(engine, database, schema, table, frame, **overrides):
        config = _config(
            database, schema, table=table, probe_name=f"{table}_probe",
            load_batch_col=None, key_cols=["row_id"], **overrides,
        )
        g.load_via_sqlalchemy(frame, engine, table)
        canonical = run_canonical(engine, config, as_of)
        completion = assess_completion(canonical, config, as_of)
        volume = assess_volume(canonical, config, as_of, completion)
        return ParitySide(config, canonical, completion, volume)

    backdated = g.inject_negative_lags(df, fraction=0.005, skew_days=5.0, seed=8)
    outcomes = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        left = side(engine, database, schema, "events_a", df, parity_with="events_b_probe")
        mismatch = assess_parity(left, side(engine, database, schema, "events_b", short), as_of)
        dup_ind = assess_parity(
            left, side(engine, database, schema, "events_b", duplicated), as_of
        )
        # read_uncommitted on one side: unverifiable watermark
        dirty_left = side(
            engine, database, schema, "events_a", df,
            parity_with="events_b_probe", read_uncommitted=True,
        )
        dirty_ind = assess_parity(
            dirty_left, side(engine, database, schema, "events_b", df), as_of
        )
        # negative-lag excess beyond threshold: the backdating proxy
        backdate_ind = assess_parity(
            left, side(engine, database, schema, "events_b", backdated), as_of
        )
        outcomes.append((mismatch, dup_ind, dirty_ind, backdate_ind))
    (
        (duck_mismatch, duck_dup, duck_dirty, duck_backdate),
        (ms_mismatch, ms_dup, ms_dirty, ms_backdate),
    ) = outcomes
    assert duck_mismatch.rows == ms_mismatch.rows
    assert duck_mismatch.statuses == ms_mismatch.statuses
    assert any(s.reason is ReasonCode.PARITY_MISMATCH for s in ms_mismatch.statuses)
    assert duck_dup.statuses == ms_dup.statuses
    assert [s.reason for s in ms_dup.statuses] == [ReasonCode.PARITY_PREREQ_UNIQUENESS]
    assert duck_dirty.statuses == ms_dirty.statuses
    assert [s.reason for s in ms_dirty.statuses] == [
        ReasonCode.PARITY_PREREQ_READ_UNCOMMITTED
    ]
    assert duck_backdate.statuses == ms_backdate.statuses
    assert [s.reason for s in ms_backdate.statuses] == [
        ReasonCode.PARITY_PREREQ_NEGATIVE_LAG
    ]


def test_scan_budget_refusals_through_the_production_runner(mssql_engine, monkeypatch):
    """SCAN_BUDGET_EXCEEDED and SCAN_BUDGET_UNVERIFIABLE produced through
    run_canonical itself on the real server, not just the pure check."""
    import metricprobe.extract.canonical as canonical_module

    df = g.generate(dataclasses.replace(TRICKLE_12, n_months=3, seed=208))
    g.load_via_sqlalchemy(df, mssql_engine, "events")
    config = _config("tempdb", "dbo", load_batch_col=None, key_cols=None)
    as_of = pd.Timestamp("2026-07-01")

    # a zero-page target makes ANY read exceed 3x pages -> EXCEEDED
    monkeypatch.setattr(canonical_module, "_mssql_target_pages", lambda conn, table: 0)
    with pytest.raises(ProbeAborted) as excinfo:
        run_canonical(mssql_engine, config, as_of)
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_EXCEEDED
    monkeypatch.undo()

    # capture disabled -> no STATISTICS IO -> fail CLOSED, never unbudgeted
    monkeypatch.setattr(
        canonical_module, "_install_message_capture", lambda conn, messages: False
    )
    with pytest.raises(ProbeAborted) as excinfo:
        run_canonical(mssql_engine, config, as_of)
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_UNVERIFIABLE
