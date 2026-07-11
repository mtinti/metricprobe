"""Step 3 equivalence: the identical synthetic dataset loaded into DuckDB and a
real SQL Server must yield the same canonical aggregation and the same
completion percentiles — plus the scan budget measured in LOGICAL READS."""

import dataclasses
import re

import pandas as pd
import sqlalchemy as sa
from tests.synth import generator as g

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import (
    build_aggregation_query,
    build_uniqueness_query,
    drop_staging_sql,
    run_canonical,
    staging_sql,
)
from metricprobe.metrics.completion import assess_completion

AS_OF = pd.Timestamp("2026-07-01")

SPEC = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=6,
    rows_per_month=4000,
    lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
    seed=61,
)


def _dataset() -> pd.DataFrame:
    # exercise every bucket: nulls, in-tolerance and beyond-tolerance negatives
    df = g.generate(SPEC)
    df = g.inject_null_event_time(df, 0.02, seed=1)
    df = g.inject_null_load_time(df, 0.02, seed=2)
    df = g.inject_negative_lags(df, 0.02, skew_days=5.0, seed=3)
    return df


def _config(database: str, schema: str) -> TableConfig:
    return TableConfig.model_validate(
        {
            "probe_name": "events_probe",
            "database": database,
            "schema": schema,
            "table": "events",
            "event_time": "event_time",
            "load_time": "load_time",
            "load_batch_col": "batch_id",
            "key_cols": ["row_id", "batch_id"],
        }
    )


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["event_month"] = pd.to_datetime(out["event_month"])
    out["load_epoch_day"] = pd.to_datetime(out["load_epoch_day"])
    out["min_load_time"] = pd.to_datetime(out["min_load_time"])
    for col in ("lag_day", "row_count", "n_curve_eligible", "n_null_event_time",
                "n_null_load_time_only", "n_negative_clipped", "n_negative_lag_excluded",
                "n_overflow", "n_join_unmatched", "distinct_keys"):
        out[col] = out[col].astype("Int64")
    keys = ["grouping_id", "event_month", "lag_day", "load_epoch_day", "batch_id"]
    return (
        out.drop(columns=["max_lookup_dup"])
        .sort_values(keys)
        .reset_index(drop=True)
    )


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


STATS_IO_LINE = re.compile(r"Table '([^']+)'\. Scan count \d+, logical reads (\d+)")


def _target_table_reads(engine, sql_statements) -> dict[str, int]:
    """Execute raw SQL statements with SET STATISTICS IO ON and sum per-table
    logical reads from the captured info messages (pymssql message handler)."""
    reads: dict[str, int] = {}
    messages: list[str] = []

    def handler(msgstate, severity, srvname, procname, line, text):
        messages.append(text.decode() if isinstance(text, bytes) else str(text))

    raw = engine.raw_connection()
    try:
        raw.driver_connection._conn.set_msghandler(handler)
        cursor = raw.cursor()
        cursor.execute("SET STATISTICS IO ON")
        for sql in sql_statements:
            cursor.execute(sql)
            if cursor.description is not None:
                cursor.fetchall()
            while cursor.nextset():
                pass
    finally:
        raw.close()
    for message in messages:
        for table, count in STATS_IO_LINE.findall(message):
            reads[table] = reads.get(table, 0) + int(count)
    return reads


def test_scan_budget_in_logical_reads(mssql_engine):
    """The probe (canonical pass + distinct-count guard, which is counted) must
    cost <= 3x one full scan of the TARGET TABLE, in SQL Server logical reads.
    Measured per table via STATISTICS IO: tempdb worktable spool pages are the
    aggregation's own scratch space, not pressure on the production table."""
    df = g.generate(dataclasses.replace(SPEC, rows_per_month=30_000, seed=62))
    g.load_via_sqlalchemy(df, mssql_engine, "events_budget")
    config = TableConfig.model_validate(
        {
            "probe_name": "events_budget_probe",
            "database": "tempdb",
            "schema": "dbo",
            "table": "events_budget",
            "event_time": "event_time",
            "load_time": "load_time",
            "load_batch_col": "batch_id",
            "key_cols": ["row_id", "batch_id"],
        }
    )
    with mssql_engine.connect() as conn:
        pages = conn.execute(
            sa.text(
                "SELECT SUM(used_page_count) FROM sys.dm_db_partition_stats "
                "WHERE object_id = OBJECT_ID('dbo.events_budget')"
            )
        ).scalar_one()
    def literal_sql(query) -> str:
        return str(
            query.compile(
                dialect=mssql_engine.dialect, compile_kwargs={"literal_binds": True}
            )
        )

    sql_statements = [
        staging_sql(config, "mssql", as_of=AS_OF),
        literal_sql(build_aggregation_query(config, "mssql")),
        literal_sql(build_uniqueness_query(config, "mssql")),
        drop_staging_sql("mssql"),
    ]
    reads = _target_table_reads(mssql_engine, sql_statements)
    target_reads = reads.get("events_budget", 0)
    assert pages > 0 and target_reads > 0
    assert target_reads <= 3 * pages, (
        f"probe cost {target_reads} logical reads on the target table "
        f"({pages} pages; budget 3x one scan). All reads: {reads}"
    )
