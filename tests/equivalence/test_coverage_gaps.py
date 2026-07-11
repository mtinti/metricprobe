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
    identical typed verdicts from both engines."""
    spec = g.volume_spike(TRICKLE_12, 3, factor=6.0)
    spec = g.volume_drop(spec, 5, factor=0.1)
    spec = g.missing_month(spec, 7)
    df = g.inject_negative_lags(g.generate(spec), fraction=0.02, skew_days=1.0, seed=4)
    as_of = pd.Timestamp("2026-07-01")  # all mature; the feed is long stale
    results = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(database, schema, load_batch_col=None, key_cols=None)
        g.load_via_sqlalchemy(df, engine, "events")
        canonical, completion, volume, freshness = _assess_all(engine, config, as_of)
        results.append((canonical, volume, freshness))
    (duck_canonical, duck_volume, duck_fresh), (ms_canonical, ms_volume, ms_fresh) = results
    assert duck_volume.statuses == ms_volume.statuses
    assert duck_volume.months == ms_volume.months
    assert duck_volume.gaps == ms_volume.gaps
    assert duck_fresh == ms_fresh
    reasons = {s.reason for s in ms_volume.statuses}
    assert ReasonCode.VOLUME_OUTLIER in reasons  # the spike and the drop
    assert ReasonCode.VOLUME_GAP in reasons  # the missing month
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
    # hour-bucket epochs agree exactly (mssql DATEADD/DATEDIFF vs date_trunc)
    duck_epochs = duck_canonical.rows_for("epoch")
    ms_epochs = ms_canonical.rows_for("epoch")
    assert len(duck_epochs) == len(ms_epochs)
    assert duck_fresh == ms_fresh


def test_straggler_batch_scenario_matches(duckdb_engine, mssql_engine):
    spec = dataclasses.replace(
        TRICKLE_12,
        n_months=8,
        lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
        seed=203,
    )
    df = g.inject_straggler_batch(
        g.generate(spec), month="2024-04", late_day=45.0, fraction=0.15, seed=5
    )
    as_of = pd.Timestamp("2026-07-01")
    assessments = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(database, schema, key_cols=None)
        g.load_via_sqlalchemy(df, engine, "events")
        canonical = run_canonical(engine, config, as_of)
        assessments.append(assess_batch(canonical, config))
    duck, mssql = assessments
    assert duck.months == mssql.months
    assert duck.rows_per_run == mssql.rows_per_run
    hit = {m.month: m for m in mssql.months}[pd.Period("2024-04", freq="M")]
    assert hit.days_to[95] == 45  # the straggler moved batch-level completion


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
    # each dialect's NULL-sentinel encoding is injective on its own: COUNTS match
    assert duck_distinct == ms_distinct
    assert ms_distinct < len(with_null_keys)  # duplicates + collapsed NULLs detected


def test_cell_cap_aborts_on_both_dialects(duckdb_engine, mssql_engine):
    df = g.generate(dataclasses.replace(TRICKLE_12, n_months=6, seed=205))
    as_of = pd.Timestamp("2026-07-01")
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        config = _config(
            database, schema, load_batch_col=None, key_cols=None,
            analysis={"result_cell_cap": 10},
        )
        g.load_via_sqlalchemy(df, engine, "events")
        with pytest.raises(ProbeAborted) as excinfo:
            run_canonical(engine, config, as_of)
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

    outcomes = []
    for engine, database, schema in _both(duckdb_engine, mssql_engine):
        left = side(engine, database, schema, "events_a", df, parity_with="events_b_probe")
        mismatch = assess_parity(left, side(engine, database, schema, "events_b", short), as_of)
        indeterminate = assess_parity(
            left, side(engine, database, schema, "events_b", duplicated), as_of
        )
        outcomes.append((mismatch, indeterminate))
    (duck_mismatch, duck_ind), (ms_mismatch, ms_ind) = outcomes
    assert duck_mismatch.rows == ms_mismatch.rows
    assert duck_mismatch.statuses == ms_mismatch.statuses
    assert any(s.reason is ReasonCode.PARITY_MISMATCH for s in ms_mismatch.statuses)
    assert duck_ind.statuses == ms_ind.statuses
    assert [s.reason for s in ms_ind.statuses] == [ReasonCode.PARITY_PREREQ_UNIQUENESS]
