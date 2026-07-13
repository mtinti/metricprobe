"""Step 3 equivalence: the identical synthetic dataset loaded into DuckDB and a
real SQL Server must yield the same canonical aggregation and the same
completion percentiles — including via-joins, group_by_alt, and the scan
budget enforced through the production runner."""

import dataclasses
import os

import pandas as pd
import pytest
import sqlalchemy as sa
from tests.synth import generator as g

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import ProbeAborted, run_canonical
from metricprobe.metrics.completion import assess_completion
from metricprobe.status import ReasonCode

AS_OF = pd.Timestamp("2026-07-01")

SPEC = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=6,
    rows_per_month=4000,
    lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
    seed=61,
)

INT_COLUMNS = (
    "lag_day",
    "row_count",
    "n_curve_eligible",
    "n_null_event_time",
    "n_null_load_time_only",
    "n_negative_clipped",
    "n_negative_lag_excluded",
    "n_overflow",
    "n_join_unmatched",
    "n_other_exclusions",
    "n_base_rows",
    "n_ambiguous_base_rows",
    "n_compare_mismatch",
    "distinct_keys",
)


def _dataset() -> pd.DataFrame:
    # exercise every bucket: nulls, in-tolerance and beyond-tolerance negatives
    df = g.generate(SPEC)
    df = g.inject_null_event_time(df, 0.02, seed=1)
    df = g.inject_null_load_time(df, 0.02, seed=2)
    df = g.inject_negative_lags(df, 0.02, skew_days=5.0, seed=3)
    return df


def _config(database: str, schema: str, **overrides) -> TableConfig:
    data = {
        "probe_name": "events_probe",
        "database": database,
        "schema": schema,
        "table": "events",
        "event_time": "event_time",
        "load_time": "load_time",
        "load_batch_col": "batch_id",
        "key_cols": ["row_id", "batch_id"],
    } | overrides
    if "resolution" not in data:
        # synthetic fixtures carry full timestamps: declare every configured
        # time role as datetime unless a test overrides
        via = data.get("event_time_via") or {}
        columns = [
            column
            for column in (
                data.get("event_time"),
                via.get("column") if isinstance(via, dict) else None,
                data.get("load_time"),
                data.get("source_insert_time"),
                data.get("compare_event_time"),
            )
            if column
        ]
        data["resolution"] = dict.fromkeys(columns, "datetime")
    return TableConfig.model_validate(data)


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("event_month", "load_epoch_day", "min_load_time"):
        out[col] = pd.to_datetime(out[col])
    for col in INT_COLUMNS:
        out[col] = out[col].astype("Int64")
    keys = ["grouping_id", "event_month", "lag_day", "load_epoch_day", "batch_id", "alt_value"]
    return out.drop(columns=["max_lookup_dup"]).sort_values(keys).reset_index(drop=True)


def test_canonical_pass_matches_between_duckdb_and_mssql(duckdb_engine, mssql_engine):
    df = _dataset()
    g.load_via_sqlalchemy(df, duckdb_engine, "events")
    g.load_via_sqlalchemy(df, mssql_engine, "events")

    duck = run_canonical(duckdb_engine, _config("memory", "main"), AS_OF)
    mssql = run_canonical(mssql_engine, _config("tempdb", "dbo"), AS_OF)
    pd.testing.assert_frame_equal(
        _normalized(duck.frame), _normalized(mssql.frame), check_dtype=False
    )

    duck_result = assess_completion(duck, _config("memory", "main"), AS_OF)
    mssql_result = assess_completion(mssql, _config("tempdb", "dbo"), AS_OF)
    assert duck_result.percentiles == mssql_result.percentiles
    assert duck_result.recommended_wait == mssql_result.recommended_wait
    assert duck_result.learned_wait == mssql_result.learned_wait
    assert duck_result.mature_percentile_summary == mssql_result.mature_percentile_summary


def _via_dataset():
    df = g.generate(dataclasses.replace(SPEC, seed=63))
    base = pd.DataFrame(
        {
            "referral_id": df["row_id"],
            "site_code": df["row_id"] % 5,
            "region": (df["row_id"] % 3).map({0: "north", 1: "south", 2: "west"}),
            "load_time": df["load_time"],
            "batch_id": df["batch_id"],
        }
    )
    lookup = pd.DataFrame(
        {"id": df["row_id"], "site": df["row_id"] % 5, "referral_date": df["event_time"]}
    )
    lookup = lookup.iloc[200:].copy()  # 200 unmatched base rows
    lookup.iloc[:80, lookup.columns.get_loc("referral_date")] = pd.NaT  # matched, NULL column
    return base, lookup


def _via_config(database: str, schema: str, **overrides) -> TableConfig:
    data = {
        "probe_name": "episodes_via",
        "database": database,
        "schema": schema,
        "table": "events",
        "load_time": "load_time",
        "group_by_alt": "region",
        "event_time_via": {
            "join_table": f"{database}.{schema}.referrals",
            "on": [
                {"base_col": "referral_id", "lookup_col": "id"},
                {"base_col": "site_code", "lookup_col": "site"},
            ],
            "column": "referral_date",
        },
        "resolution": {"referral_date": "datetime", "load_time": "datetime"},
    } | overrides
    return TableConfig.model_validate(data)


def test_via_join_and_alt_grouping_match_between_dialects(duckdb_engine, mssql_engine):
    # composite differently-named keys + unmatched rows + NULL borrowed column
    # + group_by_alt: all dialect-sensitive paths execute on the REAL server
    base, lookup = _via_dataset()
    for engine in (duckdb_engine, mssql_engine):
        g.load_via_sqlalchemy(base, engine, "events")
        g.load_via_sqlalchemy(lookup, engine, "referrals")
    duck = run_canonical(duckdb_engine, _via_config("memory", "main"), AS_OF)
    mssql = run_canonical(mssql_engine, _via_config("tempdb", "dbo"), AS_OF)
    pd.testing.assert_frame_equal(
        _normalized(duck.frame), _normalized(mssql.frame), check_dtype=False
    )
    row = mssql.global_row
    assert int(row["n_join_unmatched"]) == 200
    assert int(row["n_base_rows"]) == len(base)


def test_via_non_unique_lookup_aborts_on_both_dialects(duckdb_engine, mssql_engine):
    from metricprobe.extract.dual import run_dual_lag

    base, lookup = _via_dataset()
    duplicated = pd.concat([lookup, lookup.iloc[:7]], ignore_index=True)
    # the dual pass runs on its own connection and must refuse INDEPENDENTLY
    dual_base = base.assign(source_insert_time=base["load_time"] - pd.Timedelta(days=2))
    dual_overrides = {
        "source_insert_time": "source_insert_time",
        "resolution": {
            "referral_date": "datetime",
            "load_time": "datetime",
            "source_insert_time": "datetime",
        },
    }
    for engine, config, dual_config in (
        (
            duckdb_engine,
            _via_config("memory", "main"),
            _via_config("memory", "main", **dual_overrides),
        ),
        (
            mssql_engine,
            _via_config("tempdb", "dbo"),
            _via_config("tempdb", "dbo", **dual_overrides),
        ),
    ):
        g.load_via_sqlalchemy(dual_base, engine, "events")
        g.load_via_sqlalchemy(duplicated, engine, "referrals")
        with pytest.raises(ProbeAborted) as excinfo:
            run_canonical(engine, config, AS_OF)
        assert excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE
        assert "7 base rows are ambiguous" in excinfo.value.detail
        with pytest.raises(ProbeAborted) as dual_excinfo:
            run_dual_lag(engine, dual_config, AS_OF)
        assert dual_excinfo.value.reason is ReasonCode.JOIN_NOT_UNIQUE


def test_volume_assessment_matches_between_dialects(duckdb_engine, mssql_engine):
    """Step 4 equivalence: an ONGOING sustained collapse + injected duplicate
    keys, probed MID-DATA so still-filling is live — identical volume verdicts,
    baselines, expected bands, nowcasts, deficits, open-month rows and
    freshness from both engines."""
    from metricprobe.metrics.freshness import assess_freshness
    from metricprobe.metrics.volume import assess_volume

    # a TRICKLE collapse: loads arrive within the month, so the OPEN month has
    # rows (a batchy feed only loads after month end and can never show one)
    trickle = g.TableSpec(
        name="events", start_month="2023-01", n_months=30, rows_per_month=2000,
        lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=77,
    )
    healthy_df = g.generate(trickle)  # the healthy twin
    df = g.inject_duplicate_keys(
        g.generate(g.sustained_collapse(trickle, last_k=15, factor=0.1)),
        fraction=0.01,
        seed=9,
    )
    as_of = pd.Timestamp("2025-06-15")  # inside the last event month
    assessments = []
    healthy_assessments = []
    for engine, database, schema in (
        (duckdb_engine, "memory", "main"),
        (mssql_engine, "tempdb", "dbo"),
    ):
        config = _config(database, schema, key_cols=["row_id"], load_batch_col=None)
        g.load_via_sqlalchemy(df, engine, "events")
        canonical = run_canonical(engine, config, as_of)
        completion = assess_completion(canonical, config, as_of)
        volume = assess_volume(canonical, config, as_of, completion)
        fresh = assess_freshness(canonical, config, as_of)
        assessments.append((volume, fresh))
        clean_config = _config(
            database, schema, table="events_healthy", key_cols=["row_id"],
            load_batch_col=None,
        )
        g.load_via_sqlalchemy(healthy_df, engine, "events_healthy")
        clean_canonical = run_canonical(engine, clean_config, as_of)
        clean_completion = assess_completion(clean_canonical, clean_config, as_of)
        healthy_assessments.append(
            assess_volume(clean_canonical, clean_config, as_of, clean_completion)
        )
    (duck, duck_fresh), (mssql, mssql_fresh) = assessments
    duck_healthy, mssql_healthy = healthy_assessments
    # the healthy twin is silent in the same way on both engines
    assert duck_healthy.statuses == mssql_healthy.statuses
    healthy_reasons = {s.reason for s in mssql_healthy.statuses}
    assert ReasonCode.VOLUME_COLLAPSE not in healthy_reasons
    assert ReasonCode.DUPLICATE_KEYS not in healthy_reasons
    assert duck.statuses == mssql.statuses
    assert duck.baseline_median == mssql.baseline_median
    assert duck.baseline_sigma == mssql.baseline_sigma
    assert duck.duplicate_rows == mssql.duplicate_rows
    assert duck.months == mssql.months  # incl. expected bands, nowcasts, states
    assert duck_fresh == mssql_fresh
    reasons = {s.reason for s in mssql.statuses}
    assert ReasonCode.VOLUME_COLLAPSE in reasons  # the mature part of the collapse
    assert ReasonCode.ARRIVAL_DEFICIT in reasons  # the immature part
    assert ReasonCode.DUPLICATE_KEYS in reasons
    states = {m.state for m in mssql.months}
    assert "open" in states and "immature" in states and "mature" in states
    assert any(m.deficit for m in mssql.months)


def test_scan_budget_enforced_through_the_production_runner(mssql_engine):
    """run_canonical itself measures target-table logical reads (STATISTICS IO
    via the driver) against 3x one full scan and records both on the result.
    The staging design keeps the target at ~1 scan, so a healthy probe passes
    with plenty of headroom; check_scan_budget's abort path is unit-tested."""
    df = g.generate(dataclasses.replace(SPEC, rows_per_month=30_000, seed=62))
    g.load_via_sqlalchemy(df, mssql_engine, "events")
    # exercise the configured isolation: read_uncommitted maps to the mssql
    # READ UNCOMMITTED isolation level on the extraction engine
    from metricprobe.cli import _engine_for

    config = _config("tempdb", "dbo", read_uncommitted=True)
    engine = _engine_for(os.environ["METRICPROBE_MSSQL_URL"], True)
    with engine.connect() as conn:
        level = conn.execute(
            sa.text(
                "SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
                "WHERE session_id = @@SPID"
            )
        ).scalar_one()
        assert level == 1  # 1 = READ UNCOMMITTED: the configured flag is honored
    try:
        result = run_canonical(engine, config, AS_OF)
    finally:
        engine.dispose()
    assert result.target_logical_reads is not None, "budget measurement did not run"
    assert result.scan_budget_reads is not None
    # the SCRATCH ledger (aggregation + distinct-count guard) is measured and
    # enforced too — the guard's reads are counted, fail-closed
    assert result.scratch_logical_reads is not None and result.scratch_logical_reads > 0
    assert result.scratch_logical_reads <= result.scratch_budget_reads
    # the baseline is measured INDEPENDENTLY: the logical reads of an actual
    # forced full scan of the base data, straight from STATISTICS IO — the
    # budget must be 3x ONE SCAN, not 3x whatever the DMV happens to sum
    with mssql_engine.connect() as conn:
        conn.exec_driver_sql("SET STATISTICS IO ON")
        messages: list[str] = []
        from metricprobe.extract.canonical import (
            _install_message_capture,
            _target_reads_from,
        )

        _install_message_capture(conn, messages)
        conn.exec_driver_sql(
            "SELECT COUNT(*) FROM dbo.events WITH (INDEX(0))"  # force base scan
        )
        conn.exec_driver_sql("SELECT 1")  # flush pymssql's pending messages
        one_scan = _target_reads_from(messages, {"events"})
    assert one_scan is not None and one_scan > 0
    # the enforced budget brackets 3x that measured single scan (page counts
    # and read counts differ by small constants: IAM/allocation pages)
    assert result.scan_budget_reads <= 3 * one_scan * 1.2
    assert result.scan_budget_reads >= 3 * one_scan * 0.8
    assert result.target_logical_reads <= result.scan_budget_reads

    # a fat NONCLUSTERED index must not inflate the baseline: the budget is
    # the scan path (heap/clustered), not the sum of every index
    with mssql_engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE NONCLUSTERED INDEX ix_events_fat ON dbo.events (row_id) "
            "INCLUDE (event_time, load_time, batch_id)"
        )
    try:
        engine = _engine_for(os.environ["METRICPROBE_MSSQL_URL"], True)
        try:
            with_index = run_canonical(engine, config, AS_OF)
        finally:
            engine.dispose()
        assert with_index.scan_budget_reads <= result.scan_budget_reads * 1.1, (
            "a nonclustered index inflated the scan-budget baseline"
        )
    finally:
        with mssql_engine.begin() as conn:
            conn.exec_driver_sql("DROP INDEX ix_events_fat ON dbo.events")


def test_step5_metrics_match_between_dialects(duckdb_engine, mssql_engine):
    """Dual lag, batch metrics, freshness and the compare side-stat computed
    from the same batchy dual-timestamp dataset on both engines."""
    from metricprobe.extract.dual import run_dual_lag
    from metricprobe.metrics.batch import assess_batch
    from metricprobe.metrics.completion import compare_mismatch_by_month
    from metricprobe.metrics.dual_lag import assess_dual_lag
    from metricprobe.metrics.freshness import assess_freshness

    spec = dataclasses.replace(SPEC, dual_offset_days=2.0, seed=64)
    df = g.inject_raw_vs_corrected(g.generate(spec), fraction=0.05, shift_days=-35, seed=1)
    df = g.inject_null_source_insert(df, fraction=0.02, seed=2)
    # negative SOURCE lags beyond tolerance: the dual clip/cap policy on mssql
    corrupt = df.sample(frac=0.01, random_state=3).index
    df.loc[corrupt, "source_insert_time"] = df.loc[corrupt, "event_time"] - pd.Timedelta(days=5)
    results = []
    for engine, database, schema in (
        (duckdb_engine, "memory", "main"),
        (mssql_engine, "tempdb", "dbo"),
    ):
        config = _config(
            database,
            schema,
            source_insert_time="source_insert_time",
            compare_event_time="event_time_raw",
        )
        g.load_via_sqlalchemy(df, engine, "events")
        canonical = run_canonical(engine, config, AS_OF)
        # the <=3x target budget is CUMULATIVE across main + dual
        dual = run_dual_lag(
            engine, config, AS_OF, prior_target_reads=canonical.target_logical_reads
        )
        if engine.dialect.name == "mssql":
            assert dual.target_logical_reads > canonical.target_logical_reads
            assert dual.target_logical_reads <= dual.scan_budget_reads
            assert dual.scratch_logical_reads <= dual.scratch_budget_reads
        results.append(
            {
                "batch": assess_batch(canonical, config),
                "dual": assess_dual_lag(dual, config, AS_OF),
                "fresh": assess_freshness(canonical, config, AS_OF),
                "compare": compare_mismatch_by_month(canonical),
            }
        )
    healthy_df = g.inject_raw_vs_corrected(
        g.generate(spec), fraction=0.0, shift_days=-35, seed=1
    )  # the healthy twin: the raw column exists but never differs
    healthy_counts = []
    for engine, database, schema in (
        (duckdb_engine, "memory", "main"),
        (mssql_engine, "tempdb", "dbo"),
    ):
        clean_config = _config(
            database, schema, table="events_healthy",
            source_insert_time="source_insert_time", compare_event_time="event_time_raw",
        )
        g.load_via_sqlalchemy(healthy_df, engine, "events_healthy")
        clean = run_canonical(engine, clean_config, AS_OF)
        healthy_counts.append(compare_mismatch_by_month(clean))
    assert healthy_counts[0] == healthy_counts[1]
    assert sum(healthy_counts[1].values()) == 0  # silent on the healthy twin
    duck, mssql = results
    assert duck["batch"].months == mssql["batch"].months
    assert duck["batch"].rows_per_run == mssql["batch"].rows_per_run
    assert duck["dual"].source_percentiles == mssql["dual"].source_percentiles
    assert duck["dual"].delta_histogram.equals(mssql["dual"].delta_histogram)
    assert duck["dual"].n_null_source_only == mssql["dual"].n_null_source_only
    assert duck["dual"].negative_lag_excess_fraction == mssql["dual"].negative_lag_excess_fraction
    assert duck["dual"].negative_lag_excess_fraction > 0  # the corrupt rows registered
    assert duck["fresh"] == mssql["fresh"]
    assert duck["compare"] == mssql["compare"]
    assert sum(duck["compare"].values()) > 0  # the side-stat actually fired


def test_parity_matches_between_dialects(duckdb_engine, mssql_engine):
    from metricprobe.metrics.parity import ParitySide, assess_parity
    from metricprobe.metrics.volume import assess_volume

    df = g.generate(dataclasses.replace(SPEC, seed=65))
    right_df = df[df["event_time"].dt.to_period("M") != "2024-02"].reset_index(drop=True)
    outcomes = []
    for engine, database, schema in (
        (duckdb_engine, "memory", "main"),
        (mssql_engine, "tempdb", "dbo"),
    ):
        sides = []
        for name, frame in (("events_a", df), ("events_b", right_df)):
            config = _config(
                database,
                schema,
                table=name,
                probe_name=f"{name}_probe",
                parity_with="events_b_probe" if name == "events_a" else None,
            )
            g.load_via_sqlalchemy(frame, engine, name)
            canonical = run_canonical(engine, config, AS_OF)
            completion = assess_completion(canonical, config, AS_OF)
            volume = assess_volume(canonical, config, AS_OF, completion)
            sides.append(ParitySide(config, canonical, completion, volume))
        outcomes.append(assess_parity(*sides, AS_OF))
    duck, mssql = outcomes
    assert duck.rows == mssql.rows
    assert duck.statuses == mssql.statuses
    assert any(s.reason is ReasonCode.PARITY_ONE_SIDED_MONTH for s in mssql.statuses)


def test_mssql_store_shares_the_run_contract(mssql_engine):
    """Step 6: the config-flagged mssql writer honors the same lifecycle —
    staged rows are invisible until the manifest INSERT commits the run, a
    concurrent writer can never claim the same run_id, an abort never touches
    a committed run, and retention pruning works. Rerunnable: unique ids plus
    prune-based cleanup."""
    import dataclasses
    import os
    import uuid

    from metricprobe.store import MssqlStore, RunMeta, stamp

    def make_meta(run_id, run_at="2026-07-01T06:00:00"):
        return RunMeta(
            run_id=run_id,
            run_at=run_at,
            as_of="2026-07-01T00:00:00",
            git_sha="deadbeef",
            tool_version="0.1.0.dev0",
            config_digest="abc",
            schema_version=1,
            window_start="2024-07-01T00:00:00",
            window_end="2026-07-01T00:00:00",
        )

    def manifest_for(meta):
        return {**dataclasses.asdict(meta), "stages": {"analysis": {}}}

    url = os.environ["METRICPROBE_MSSQL_URL"]
    store = MssqlStore(url, schema="dbo")
    run_id = f"eqv-{uuid.uuid4().hex[:12]}"
    meta = make_meta(run_id)
    frame = stamp(pd.DataFrame({"probe": ["a", "b"], "volume": [1, 2]}), meta)
    store.begin_run(meta)
    # a concurrent writer cannot claim the same run_id even BEFORE the commit
    rival = MssqlStore(url, schema="dbo")
    with pytest.raises(FileExistsError):
        rival.begin_run(meta)
    store.write_table(run_id, "month_volumes", frame)
    assert not any(m["run_id"] == run_id for m in store.list_runs())  # staged: invisible
    store.commit_run(run_id, manifest_for(meta))
    assert any(m["run_id"] == run_id for m in store.list_runs())
    # the run reports which logical tables it holds (presentation reads
    # exactly these and lets read failures on present ones propagate)
    assert store.table_names(run_id) == ["month_volumes"]
    with pytest.raises(FileNotFoundError):
        store.table_names("never-committed")
    # post-commit lifecycle stages update the committed manifest in place
    store.record_stage(run_id, "render", {"completed_at": "2026-07-01T06:05:00"})
    # the publish record is TWO-PHASE: prepare front-loads the fallible work
    # before any push; nothing is visible until the finalize runs
    finalize = store.prepare_stage(run_id, "publish", {"remotes": ["origin"]})
    staged = next(m for m in store.list_runs() if m["run_id"] == run_id)
    assert "publish" not in staged["stages"]
    finalize()
    recorded = next(m for m in store.list_runs() if m["run_id"] == run_id)
    assert recorded["stages"]["render"]["completed_at"] == "2026-07-01T06:05:00"
    assert recorded["stages"]["publish"]["remotes"] == ["origin"]
    with pytest.raises(FileNotFoundError):
        store.record_stage("never-committed", "render", {})
    with pytest.raises(FileNotFoundError):
        store.prepare_stage("never-committed", "publish", {})
    # the rival's abort of the SAME id must not touch the committed rows
    rival._staged[run_id] = {"claim": "someone-else", "names": ["month_volumes"]}
    rival.abort_run(run_id)
    read_back = store.read_table(run_id, "month_volumes")
    assert read_back["probe"].tolist() == ["a", "b"]
    assert read_back["git_sha"].tolist() == ["deadbeef", "deadbeef"]
    with pytest.raises(FileExistsError):
        store.begin_run(meta)
    # retention pruning shares the parquet contract — and cleans this test up
    newer_id = f"eqv-{uuid.uuid4().hex[:12]}"
    newer = make_meta(newer_id, run_at="2026-07-01T07:00:00")
    store.begin_run(newer)
    store.write_table(newer_id, "month_volumes", stamp(pd.DataFrame({"probe": ["c"]}), newer))
    store.commit_run(newer_id, manifest_for(newer))
    store.prune(keep=0)
    assert store.list_runs() == []
    with pytest.raises(FileNotFoundError):
        store.read_table(run_id, "month_volumes")


def test_mssql_store_sweeps_are_isolated_and_types_are_frozen(mssql_engine):
    """The sweeping deletes (abort without in-memory state, prune) must never
    touch a FOREIGN table that matches 'mp_%' only through LIKE's underscore
    wildcard; a writer whose staging claim was taken over must refuse to
    commit; and the frozen snapshot dtypes survive a None-first frame."""
    import dataclasses
    import os
    import uuid

    from metricprobe.cli import _typed
    from metricprobe.store import MssqlStore, RunMeta, stamp

    url = os.environ["METRICPROBE_MSSQL_URL"]
    store = MssqlStore(url, schema="dbo")

    def make_meta(run_id, run_at="2026-07-01T06:00:00"):
        return RunMeta(
            run_id=run_id, run_at=run_at, as_of="2026-07-01T00:00:00",
            git_sha="deadbeef", tool_version="0.1.0.dev0", config_digest="abc",
            schema_version=1, window_start="2024-07-01T00:00:00",
            window_end="2026-07-01T00:00:00",
        )

    run_id = f"eqv-{uuid.uuid4().hex[:12]}"
    # ---- foreign tables: neither a LIKE-wildcard cousin (mpx_...) nor a
    # table LITERALLY named mp_something is metricprobe-owned — only tables
    # recorded in the ownership catalog may ever be swept
    for decoy in ("mpx_audit_foreign", "mp_foreign_business"):
        with store.engine.begin() as conn:
            conn.exec_driver_sql(
                f"IF OBJECT_ID('dbo.{decoy}') IS NOT NULL DROP TABLE dbo.{decoy}"
            )
            conn.exec_driver_sql(
                f"CREATE TABLE dbo.{decoy} (run_id varchar(64), payload int)"
            )
            conn.exec_driver_sql(f"INSERT INTO dbo.{decoy} VALUES ('{run_id}', 42)")
        assert decoy not in store._data_tables()
    store.abort_run(run_id)  # unknown names -> sweeps all CATALOGED tables
    store.prune(keep=0)
    with store.engine.connect() as conn:
        for decoy in ("mpx_audit_foreign", "mp_foreign_business"):
            survivors = conn.exec_driver_sql(
                f"SELECT COUNT(*) FROM dbo.{decoy}"
            ).scalar()
            assert survivors == 1, decoy  # the foreign row is untouched
    with store.engine.begin() as conn:
        for decoy in ("mpx_audit_foreign", "mp_foreign_business"):
            conn.exec_driver_sql(f"DROP TABLE dbo.{decoy}")

    # ---- C2: a takeover invalidates the original writer's commit
    meta = make_meta(run_id)
    store.begin_run(meta)
    frame = stamp(pd.DataFrame({"probe": ["a"], "volume": [1]}), meta)
    store.write_table(run_id, "month_volumes", frame)
    usurper = MssqlStore(url, schema="dbo")
    assert usurper.staging_claim(run_id) is not None  # visible server-side
    usurper.abort_run(run_id)  # deletes rows AND the claim
    with pytest.raises(RuntimeError, match="staging claim lost"):
        store.commit_run(run_id, {**dataclasses.asdict(meta), "stages": {}})
    # nothing became visible: no manifest, no phantom rows
    assert not any(m["run_id"] == run_id for m in store.list_runs())

    # ---- H3: None-first frames do not freeze varchar for numeric columns
    first_id = f"eqv-{uuid.uuid4().hex[:12]}"
    first = make_meta(first_id)
    store.begin_run(first)
    none_frame = _typed(
        "freshness",
        pd.DataFrame({"probe": ["p"], "epoch_count": [1], "cadence_median_days": [None]}),
    )
    store.write_table(first_id, "freshness", stamp(none_frame, first))
    store.commit_run(first_id, {**dataclasses.asdict(first), "stages": {}})
    second_id = f"eqv-{uuid.uuid4().hex[:12]}"
    second = make_meta(second_id, run_at="2026-07-01T07:00:00")
    store.begin_run(second)
    value_frame = _typed(
        "freshness",
        pd.DataFrame({"probe": ["p"], "epoch_count": [9], "cadence_median_days": [7.5]}),
    )
    store.write_table(second_id, "freshness", stamp(value_frame, second))
    store.commit_run(second_id, {**dataclasses.asdict(second), "stages": {}})
    read_back = store.read_table(second_id, "freshness")
    assert float(read_back["cadence_median_days"].iloc[0]) == 7.5
    assert not isinstance(read_back["cadence_median_days"].iloc[0], str)
    assert int(read_back["epoch_count"].iloc[0]) == 9

    # numeric-looking batch ids first, textual later: both survive as text
    third_id = f"eqv-{uuid.uuid4().hex[:12]}"
    third = make_meta(third_id, run_at="2026-07-01T08:00:00")
    store.begin_run(third)
    numeric_ids = _typed(
        "batch_runs", pd.DataFrame({"probe": ["p"], "batch_id": [20240101], "rows": [5]})
    )
    store.write_table(third_id, "batch_runs", stamp(numeric_ids, third))
    store.commit_run(third_id, {**dataclasses.asdict(third), "stages": {}})
    fourth_id = f"eqv-{uuid.uuid4().hex[:12]}"
    fourth = make_meta(fourth_id, run_at="2026-07-01T09:00:00")
    store.begin_run(fourth)
    text_ids = _typed(
        "batch_runs",
        pd.DataFrame({"probe": ["p"], "batch_id": ["2024-02-run1"], "rows": [7]}),
    )
    store.write_table(fourth_id, "batch_runs", stamp(text_ids, fourth))
    store.commit_run(fourth_id, {**dataclasses.asdict(fourth), "stages": {}})
    assert store.read_table(fourth_id, "batch_runs")["batch_id"].iloc[0] == "2024-02-run1"

    # ---- the physical-schema version marker refuses cross-version appends
    with store.engine.begin() as conn:
        conn.exec_driver_sql(
            f"UPDATE dbo.{MssqlStore.META_TABLE} SET meta_value = '2' "
            "WHERE meta_key = 'snapshot_schema_version'"
        )
    with pytest.raises(RuntimeError, match="snapshot schema v2"):
        MssqlStore(url, schema="dbo")
    from metricprobe.store import SNAPSHOT_SCHEMA_VERSION

    with store.engine.begin() as conn:
        conn.execute(
            sa.text(
                f"UPDATE dbo.{MssqlStore.META_TABLE} SET meta_value = :v "
                "WHERE meta_key = 'snapshot_schema_version'"
            ),
            {"v": str(SNAPSHOT_SCHEMA_VERSION)},
        )
    store.prune(keep=0)  # clean up


def test_legacy_datetime_columns_accept_a_microsecond_as_of(duckdb_engine, mssql_engine):
    """Type-precision AND locale regression (found in production, B3): a
    microsecond as_of used to be inlined as a six-fractional-digit literal,
    which legacy DATETIME (3 digits max) and SMALLDATETIME reject with Msg
    241 — while datetime2 parses it, so a datetime2-only harness never sees
    the bug. The session ALSO runs under SET DATEFORMAT dmy with a day > 12
    in the cutoff: a space-separated literal raises Msg 242 there (or
    silently swaps month and day); only the T-separated ISO form is
    locale-independent. Numbers must still match DuckDB exactly."""
    import sqlalchemy as sa_events
    from sqlalchemy.dialects import mssql as mssql_types

    df = _dataset()
    # whole-minute timestamps: SMALLDATETIME rounds to the minute, and the
    # comparison is exact only when rounding cannot move any row
    for column in ("event_time", "load_time"):
        df[column] = df[column].dt.floor("min")
    g.load_via_sqlalchemy(df, duckdb_engine, "events")
    g.load_via_sqlalchemy(
        df,
        mssql_engine,
        "events",
        dtype={
            "event_time": mssql_types.DATETIME(),      # legacy, 3 fractional digits
            "load_time": mssql_types.SMALLDATETIME(),  # minute precision
        },
    )

    # the hostile locale applies to the PROBE's connections (a real login's
    # DATEFORMAT is set at query time). It is flipped only after the load:
    # pymssql itself sends datetime parameters as locale-sensitive strings,
    # so a dmy session would break the seeding INSERTs, not the probe.
    @sa_events.event.listens_for(mssql_engine, "connect")
    def _hostile_locale(dbapi_connection, record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET DATEFORMAT dmy")
        cursor.close()

    mssql_engine.dispose()  # drop pooled connections: probes reconnect as dmy

    microsecond_as_of = pd.Timestamp("2026-07-13 07:49:08.085711")  # day > 12
    duck = run_canonical(duckdb_engine, _config("memory", "main"), microsecond_as_of)
    mssql = run_canonical(mssql_engine, _config("tempdb", "dbo"), microsecond_as_of)
    pd.testing.assert_frame_equal(
        _normalized(duck.frame), _normalized(mssql.frame), check_dtype=False
    )
    # the tempdb sizing observable agrees across engines and counts all rows
    assert duck.staged_row_count == mssql.staged_row_count == len(df)


def test_smalldatetime_cannot_admit_rows_after_the_cutoff(duckdb_engine, mssql_engine):
    """Precision-degradation boundary: comparing a SMALLDATETIME column
    against a bare string converts the CUTOFF to smalldatetime, which ROUNDS
    a latter-half-minute cutoff up — '07:49:45' becomes 07:50:00 and admits
    a row loaded 15 seconds AFTER the recorded as_of. The DATETIME2(0) cast
    keeps the comparison at the cutoff's own precision: exactly one of the
    three rows is admitted, identically on both engines."""
    from sqlalchemy.dialects import mssql as mssql_types

    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "event_time": pd.to_datetime(["2026-06-01 12:00:00"] * 3),
            "load_time": pd.to_datetime(
                # before the cutoff / 15s after (the rounding trap) / far after
                ["2026-06-02 07:49:00", "2026-06-02 07:50:00", "2026-06-02 08:10:00"]
            ),
            "batch_id": ["b1", "b1", "b1"],
        }
    )
    g.load_via_sqlalchemy(df, duckdb_engine, "events")
    g.load_via_sqlalchemy(
        df, mssql_engine, "events", dtype={"load_time": mssql_types.SMALLDATETIME()}
    )

    latter_half_minute = pd.Timestamp("2026-06-02 07:49:45.5")
    duck = run_canonical(
        duckdb_engine, _config("memory", "main", key_cols=["row_id"]), latter_half_minute
    )
    mssql = run_canonical(
        mssql_engine, _config("tempdb", "dbo", key_cols=["row_id"]), latter_half_minute
    )
    pd.testing.assert_frame_equal(
        _normalized(duck.frame), _normalized(mssql.frame), check_dtype=False
    )
    global_row = mssql.rows_for("global").iloc[0]
    assert int(global_row["row_count"]) == 1  # ONLY the pre-cutoff row
