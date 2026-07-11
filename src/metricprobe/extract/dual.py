"""The dual-lag pass (ALGORITHMS.md section 14) — the ONE additional pass the
budget allows (main + dual = 2 target scans <= 3x).

Same execution shape as the canonical pass: stage one scan of narrow derived
columns into a session temp table, aggregate with GROUPING SETS, drop. The
scan budget is verified fail-closed on mssql exactly like the main pass.

FROZEN DUAL SCHEMA (v1). Grouping columns and GROUPING() weights:
    event_month (4), lag_day (2), delta_day (1)

    1 = (event_month, lag_day)   source-side completion curves, where lag_day
                                 = DATEDIFF(day, event_time, source_insert_time)
                                 under the SAME clip/cap/overflow policy
    6 = (delta_day)              DATEDIFF(day, source_insert_time, load_time):
                                 the upstream-vs-local split (delta histogram)
    7 = ()                       global bucket scalars

Dual reconciliation (mutually exclusive, over the as-of-admitted rows):
    total = source_eligible + null_event_time + null_source_only
          + negative_lag_excluded
Rows with NULL source_insert_time (event present) form their own reported
bucket `n_null_source_only`. delta_day is defined only where BOTH
source_insert_time and load_time are present (`n_delta_rows`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import (
    DateDiffDay,
    GroupingSetsClause,
    MonthFloor,
    ProbeAborted,
    _dialect_instance,
    _harvest_cursor_messages,
    _install_message_capture,
    _mssql_target_pages,
    _table_clause,
    _target_reads_from,
    verify_scan_budget,
)
from metricprobe.status import ReasonCode

DUAL_SCHEMA_VERSION = 1

DUAL_GROUPING_WEIGHTS = {"event_month": 4, "lag_day": 2, "delta_day": 1}

DUAL_GROUPING_SET_IDS = {"month_src_lag": 1, "delta": 6, "global": 7}

DUAL_STAGING_COLUMNS = (
    "event_month",
    "lag_day",  # SOURCE lag (event -> source_insert), reusing curve machinery
    "delta_day",
    "is_source_eligible",
    "is_null_event_time",
    "is_null_source_only",
    "is_negative_clipped",
    "is_negative_lag_excluded",
    "is_overflow",
    "is_delta_row",
)

DUAL_RESULT_COLUMNS = (
    "grouping_id",
    "event_month",
    "lag_day",
    "delta_day",
    "row_count",
    "n_source_eligible",
    "n_null_event_time",
    "n_null_source_only",
    "n_negative_clipped",
    "n_negative_lag_excluded",
    "n_overflow",
    "n_delta_rows",
)


def dual_staging_table_name(dialect: str) -> str:
    return "#mp_dual" if dialect == "mssql" else "mp_dual"


def _dual_staging_clause(dialect: str):
    return sa.table(
        dual_staging_table_name(dialect), *[sa.column(c) for c in DUAL_STAGING_COLUMNS]
    )


def build_dual_staging_select(table: TableConfig, dialect: str) -> sa.Select:
    """The one target-table scan of the dual pass."""
    if not table.source_insert_time:
        raise ValueError(f"probe {table.probe_name!r} has no source_insert_time")
    if not table.event_time:
        raise ValueError(
            f"probe {table.probe_name!r}: dual lag with event_time_via is not "
            "supported in v1 (the borrowed-event pass carries no source column)"
        )
    analysis = table.analysis
    cap = analysis.lag_cap_days
    tolerance = analysis.clock_skew_tolerance_days

    columns = sorted({table.event_time, table.load_time, table.source_insert_time})
    base = _table_clause(table.database, table.table_schema, table.table, columns, dialect)
    event = base.c[table.event_time]
    load = base.c[table.load_time]
    source = base.c[table.source_insert_time]

    lag = DateDiffDay(event, source)
    both = sa.and_(event.is_not(None), source.is_not(None))
    is_null_event = event.is_(None)
    is_null_source_only = sa.and_(event.is_not(None), source.is_(None))
    is_negative_excluded = sa.and_(both, lag < -tolerance)
    is_eligible = sa.and_(both, lag >= -tolerance)
    is_clipped = sa.and_(both, lag >= -tolerance, lag < 0)
    is_overflow = sa.and_(is_eligible, lag > cap)
    is_delta_row = sa.and_(source.is_not(None), load.is_not(None))

    def flag(condition):
        return sa.case((condition, 1), else_=0)

    month_key = sa.case((is_eligible, MonthFloor(event)), else_=sa.null()).label("event_month")
    lag_key = sa.case(
        (sa.and_(is_eligible, lag > cap), cap + 1),  # frozen overflow sentinel
        (sa.and_(is_eligible, lag < 0), 0),  # clock-skew clip
        (is_eligible, lag),
        else_=sa.null(),
    ).label("lag_day")
    delta_key = sa.case((is_delta_row, DateDiffDay(source, load)), else_=sa.null()).label(
        "delta_day"
    )

    return (
        sa.select(
            month_key,
            lag_key,
            delta_key,
            flag(is_eligible).label("is_source_eligible"),
            flag(is_null_event).label("is_null_event_time"),
            flag(is_null_source_only).label("is_null_source_only"),
            flag(is_clipped).label("is_negative_clipped"),
            flag(is_negative_excluded).label("is_negative_lag_excluded"),
            flag(is_overflow).label("is_overflow"),
            flag(is_delta_row).label("is_delta_row"),
        )
        .select_from(base)
        # the bare <= would silently delete the NULL-load bucket
        .where(sa.or_(load <= sa.bindparam("as_of", type_=sa.DateTime()), load.is_(None)))
    )


def dual_staging_sql(table: TableConfig, dialect: str, as_of=None) -> str:
    select = build_dual_staging_select(table, dialect)
    if as_of is not None:
        select = select.params(as_of=as_of)
        compiled = str(
            select.compile(
                dialect=_dialect_instance(dialect), compile_kwargs={"literal_binds": True}
            )
        )
    else:
        compiled = str(select.compile(dialect=_dialect_instance(dialect)))
    if dialect == "mssql":
        return compiled.replace(
            "\nFROM ", f" INTO {dual_staging_table_name(dialect)} \nFROM ", 1
        )
    return (
        f"CREATE OR REPLACE TEMPORARY TABLE {dual_staging_table_name(dialect)} AS {compiled}"
    )


def build_dual_aggregation_query(table: TableConfig, dialect: str) -> sa.Select:
    staging = _dual_staging_clause(dialect)
    month_key = staging.c.event_month
    lag_key = staging.c.lag_day
    delta_key = staging.c.delta_day
    sets = [[month_key, lag_key], [delta_key], []]
    grouping_id = (
        sa.func.grouping(month_key) * DUAL_GROUPING_WEIGHTS["event_month"]
        + sa.func.grouping(lag_key) * DUAL_GROUPING_WEIGHTS["lag_day"]
        + sa.func.grouping(delta_key) * DUAL_GROUPING_WEIGHTS["delta_day"]
    )

    def total(col):
        return sa.func.coalesce(sa.func.sum(col), 0)

    return (
        sa.select(
            grouping_id.label("grouping_id"),
            month_key.label("event_month"),
            lag_key.label("lag_day"),
            delta_key.label("delta_day"),
            sa.func.count().label("row_count"),
            total(staging.c.is_source_eligible).label("n_source_eligible"),
            total(staging.c.is_null_event_time).label("n_null_event_time"),
            total(staging.c.is_null_source_only).label("n_null_source_only"),
            total(staging.c.is_negative_clipped).label("n_negative_clipped"),
            total(staging.c.is_negative_lag_excluded).label("n_negative_lag_excluded"),
            total(staging.c.is_overflow).label("n_overflow"),
            total(staging.c.is_delta_row).label("n_delta_rows"),
        )
        .select_from(staging)
        .group_by(GroupingSetsClause(sets))
        .limit(table.analysis.result_cell_cap * len(sets) + 1)
    )


def drop_dual_staging_sql(dialect: str) -> str:
    return f"DROP TABLE {dual_staging_table_name(dialect)}"


@dataclass
class DualLagResult:
    frame: pd.DataFrame
    target_logical_reads: int | None = None
    scan_budget_reads: int | None = None

    def rows_for(self, set_name: str) -> pd.DataFrame:
        gid = DUAL_GROUPING_SET_IDS[set_name]
        rows = self.frame[self.frame["grouping_id"] == gid]
        if set_name == "month_src_lag":
            rows = rows[rows["event_month"].notna()]
        elif set_name == "delta":
            rows = rows[rows["delta_day"].notna()]
        return rows.reset_index(drop=True)

    @property
    def global_row(self) -> pd.Series:
        rows = self.frame[self.frame["grouping_id"] == DUAL_GROUPING_SET_IDS["global"]]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one dual global row, got {len(rows)}")
        return rows.iloc[0]


def run_dual_lag(engine, table: TableConfig, as_of) -> DualLagResult:
    """Stage -> aggregate -> drop for the dual pass, with the same fail-closed
    scan-budget verification as the main pass."""
    dialect = engine.dialect.name
    cap = table.analysis.result_cell_cap
    counts: dict[int, int] = {}
    rows = []
    target_reads = budget = None
    with engine.connect() as conn:
        messages: list[str] = []
        pages = None
        if dialect == "mssql":
            pages = _mssql_target_pages(conn, table)
            _install_message_capture(conn, messages)
            conn.exec_driver_sql("SET STATISTICS IO ON")
        staging_result = conn.exec_driver_sql(
            dual_staging_sql(table, dialect, as_of=pd.Timestamp(as_of))
        )
        _harvest_cursor_messages(staging_result, messages)
        try:
            result = conn.execute(build_dual_aggregation_query(table, dialect))
            keys = result.keys()
            for row in result:
                gid = row.grouping_id
                counts[gid] = counts.get(gid, 0) + 1
                if counts[gid] > cap:
                    raise ProbeAborted(
                        ReasonCode.RESULT_CELL_CAP_EXCEEDED,
                        f"probe {table.probe_name!r}: dual grouping set {gid} exceeded "
                        f"result_cell_cap={cap}",
                    )
                rows.append(tuple(row))
            _harvest_cursor_messages(result, messages)
            if dialect == "mssql":
                target_reads, budget = verify_scan_budget(
                    _target_reads_from(messages, {table.table}), pages, table.probe_name
                )
        finally:
            conn.exec_driver_sql(drop_dual_staging_sql(dialect))
    frame = pd.DataFrame(rows, columns=list(keys))
    frame["event_month"] = pd.to_datetime(frame["event_month"])
    return DualLagResult(
        frame=frame, target_logical_reads=target_reads, scan_budget_reads=budget
    )
