"""The ONE canonical per-table aggregation pass (PLAN Step 3).

Execution shape (per probe, one connection):

    1. STAGING  — the ONLY statement that touches the target table: a single
       scan projecting narrow derived columns (event_month, lag_day, epoch,
       batch, alt, classification flags, key hash) into a session temp table.
    2. AGGREGATION — GROUP BY GROUPING SETS over the staging table.
    3. UNIQUENESS GUARD — COUNT(DISTINCT key_hash) over the staging table
       (only when key_cols are configured); patched into the global row.
    4. DROP the staging table.

Why staging: measured on SQL Server, GROUPING SETS over the base table re-scans
it once per dimension (3-5 scans), and COUNT(DISTINCT HASHBYTES(...)) inside
that plan forces a per-row spool (~150x one scan). Staging keeps target-table
pressure at EXACTLY ONE scan — comfortably under the <=3x logical-reads budget
(the guard is counted) — and moves the aggregation onto a projection that is
dramatically narrower than a production row. The budget test measures this
per-table via STATISTICS IO.

FROZEN RESULT SCHEMA (v1) — pinned by tests before any SQL was written.

Grouping columns, in this exact order with these GROUPING() weights:
    event_month (16), lag_day (8), load_epoch_day (4), batch_id (2), alt_value (1)

`grouping_id` = SUM(GROUPING(col) * weight); a column that is not configured
(batch/alt) contributes its weight as a constant (always aggregated). The id
therefore identifies the grouping set with stable values:

    7  = (event_month, lag_day)      completion + volume, curve-eligible rows
    27 = (load_epoch_day)            freshness epochs (rows with load_time)
    13 = (event_month, batch_id)     when load_batch_col is configured
    30 = (alt_value)                 when group_by_alt is configured
    31 = ()                          global scalars

Aggregate columns present in EVERY result row (the global row carries the
table-level truth):
    row_count, n_curve_eligible, n_null_event_time, n_null_load_time_only,
    n_negative_clipped, n_negative_lag_excluded, n_overflow, n_join_unmatched,
    max_lookup_dup, distinct_keys, min_load_time

Row classification (mutually exclusive, ALGORITHMS.md section 7):
    join_unmatched > null_event_time > null_load_time_only >
    negative_lag_excluded > curve_eligible
Within curve-eligible: lag in [-tolerance, 0) clips to lag 0
(n_negative_clipped); lag > lag_cap_days lands at lag_cap_days + 1 — the
frozen overflow bucket sentinel (n_overflow).

The staging query carries `(load_time <= :as_of OR load_time IS NULL)` — the
bare `<=` would silently delete the NULL-load bucket reconciliation requires.

The via-join uniqueness guard piggybacks on the staging pass: the lookup side
is wrapped with COUNT(*) OVER (PARTITION BY join keys); MAX(lookup_dup) > 1
aborts the probe (JOIN_NOT_UNIQUE) before any metric is computed from inflated
rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ClauseElement, FunctionElement

from metricprobe.config import TableConfig
from metricprobe.status import ReasonCode

CANONICAL_SCHEMA_VERSION = 1

GROUPING_WEIGHTS = {
    "event_month": 16,
    "lag_day": 8,
    "load_epoch_day": 4,
    "batch_id": 2,
    "alt_value": 1,
}

GROUPING_SET_IDS = {
    "month_lag": 7,
    "epoch": 27,
    "month_batch": 13,
    "alt": 30,
    "global": 31,
}

# frozen columns of the staging projection (always all present; unconfigured
# roles are staged as NULL so the aggregation SQL is uniform)
STAGING_COLUMNS = (
    "event_month",
    "lag_day",
    "load_epoch_day",
    "batch_id",
    "alt_value",
    "is_eligible",
    "is_null_event_time",
    "is_null_load_time_only",
    "is_negative_clipped",
    "is_negative_lag_excluded",
    "is_overflow",
    "is_join_unmatched",
    "lookup_dup",
    "key_hash",
    "load_time",
)

RESULT_COLUMNS = (
    "grouping_id",
    "event_month",
    "lag_day",
    "load_epoch_day",
    "batch_id",
    "alt_value",
    "row_count",
    "n_curve_eligible",
    "n_null_event_time",
    "n_null_load_time_only",
    "n_negative_clipped",
    "n_negative_lag_excluded",
    "n_overflow",
    "n_join_unmatched",
    "max_lookup_dup",
    "distinct_keys",
    "min_load_time",
)


class ProbeAborted(Exception):
    """The probe refuses to produce results (budget/cap/join violations)."""

    def __init__(self, reason: ReasonCode, detail: str):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason.value}: {detail}")


# ------------------------------------------------------- dialect-aware pieces


class DateDiffDay(FunctionElement):
    """Integer calendar-day boundaries crossed (T-SQL DATEDIFF semantics)."""

    name = "datediff_day"
    inherit_cache = True


@compiles(DateDiffDay, "mssql")
def _datediff_mssql(element, compiler, **kw):
    start, end = list(element.clauses)
    return f"DATEDIFF(day, {compiler.process(start, **kw)}, {compiler.process(end, **kw)})"


@compiles(DateDiffDay)
def _datediff_default(element, compiler, **kw):
    start, end = list(element.clauses)
    return f"date_diff('day', {compiler.process(start, **kw)}, {compiler.process(end, **kw)})"


class MonthFloor(FunctionElement):
    name = "month_floor"
    inherit_cache = True


@compiles(MonthFloor, "mssql")
def _month_floor_mssql(element, compiler, **kw):
    col = compiler.process(list(element.clauses)[0], **kw)
    return f"DATEFROMPARTS(YEAR({col}), MONTH({col}), 1)"


@compiles(MonthFloor)
def _month_floor_default(element, compiler, **kw):
    col = compiler.process(list(element.clauses)[0], **kw)
    return f"CAST(date_trunc('month', {col}) AS DATE)"


class TimeBucket(FunctionElement):
    """Floor a timestamp to the freshness bucket (day or hour)."""

    name = "time_bucket"
    inherit_cache = True

    def __init__(self, col, unit: str):
        self.unit = unit
        super().__init__(col)


@compiles(TimeBucket, "mssql")
def _time_bucket_mssql(element, compiler, **kw):
    col = compiler.process(list(element.clauses)[0], **kw)
    if element.unit == "day":
        return f"CAST({col} AS DATE)"
    return f"DATEADD(hour, DATEDIFF(hour, 0, {col}), 0)"


@compiles(TimeBucket)
def _time_bucket_default(element, compiler, **kw):
    col = compiler.process(list(element.clauses)[0], **kw)
    if element.unit == "day":
        return f"CAST({col} AS DATE)"
    return f"date_trunc('hour', {col})"


class KeyHash(FunctionElement):
    """SHA-256 over a type-tagged, length-prefixed encoding with a distinct
    NULL sentinel (ALGORITHMS.md section 9). Each dialect's encoding is
    injective on its own; equivalence compares COUNTS, not hash bytes.
    Never 32-bit CHECKSUM."""

    name = "key_hash"
    inherit_cache = True


@compiles(KeyHash, "mssql")
def _key_hash_mssql(element, compiler, **kw):
    parts = []
    for clause in element.clauses:
        col = compiler.process(clause, **kw)
        binary = f"CAST({col} AS varbinary(max))"
        parts.append(
            f"CASE WHEN {col} IS NULL THEN 0x00 ELSE CAST(0x01 AS varbinary(max)) "
            f"+ CAST(CAST(DATALENGTH({binary}) AS int) AS binary(4)) + {binary} END"
        )
    return f"HASHBYTES('SHA2_256', {' + '.join(parts)})"


@compiles(KeyHash)
def _key_hash_default(element, compiler, **kw):
    parts = []
    for clause in element.clauses:
        col = compiler.process(clause, **kw)
        text = f"CAST({col} AS VARCHAR)"
        parts.append(
            f"CASE WHEN {col} IS NULL THEN chr(0) ELSE '1' || "
            f"lpad(CAST(length({text}) AS VARCHAR), 10, '0') || {text} END"
        )
    return f"sha256({' || '.join(parts)})"


class GroupingSetsClause(ClauseElement):
    """GROUP BY GROUPING SETS ((...), (...), ()) — built manually so the
    compiled shape is identical and reviewable on both dialects."""

    inherit_cache = False

    def __init__(self, sets):
        self.sets = sets  # list of lists of ColumnElements; [] = the () set

    def _compiler_dispatch(self, compiler, **kw):
        rendered = []
        for group in self.sets:
            inner = ", ".join(compiler.process(col, **kw) for col in group)
            rendered.append(f"({inner})")
        return "GROUPING SETS (" + ", ".join(rendered) + ")"


# --------------------------------------------------------------- SQL builders


def _table_clause(database: str, schema: str, name: str, columns, dialect: str):
    if dialect == "mssql":
        # the mssql dialect natively splits two-part "database.owner" schemas
        schema_arg = f"{database}.{schema}"
    else:
        # duckdb: emit the dotted path unquoted (config names are identifiers)
        schema_arg = sa.sql.quoted_name(f"{database}.{schema}", quote=False)
    return sa.table(name, *[sa.column(col) for col in columns], schema=schema_arg)


def staging_table_name(dialect: str) -> str:
    return "#mp_probe" if dialect == "mssql" else "mp_probe"


def _staging_clause(dialect: str):
    return sa.table(staging_table_name(dialect), *[sa.column(c) for c in STAGING_COLUMNS])


def build_staging_select(table: TableConfig, dialect: str) -> sa.Select:
    """The ONLY statement that reads the target table: one scan projecting the
    narrow derived columns, with bindparam `as_of`."""
    analysis = table.analysis
    cap = analysis.lag_cap_days
    tolerance = analysis.clock_skew_tolerance_days

    base_columns = {table.load_time}
    if table.event_time:
        base_columns.add(table.event_time)
    if table.load_batch_col:
        base_columns.add(table.load_batch_col)
    if table.group_by_alt:
        base_columns.add(table.group_by_alt)
    for key in table.key_cols or ():
        base_columns.add(key)
    via = table.event_time_via
    if via:
        for pair in via.on:
            base_columns.add(pair.base_col)
    base = _table_clause(
        table.database, table.table_schema, table.table, sorted(base_columns), dialect
    )
    load = base.c[table.load_time]

    lk = None
    if via:
        lookup_cols = {pair.lookup_col for pair in via.on} | {via.column}
        lookup = _table_clause(
            via.database, via.table_schema, via.table, sorted(lookup_cols), dialect
        )
        lk = (
            sa.select(
                *[lookup.c[col].label(f"lk_{col}") for col in sorted(lookup_cols)],
                sa.literal(1).label("lk_matched"),
                sa.func.count()
                .over(partition_by=[lookup.c[pair.lookup_col] for pair in via.on])
                .label("lk_dup"),
            ).subquery("lk")
        )
        event = lk.c[f"lk_{via.column}"]
        matched = lk.c.lk_matched.is_not(None)
        source = base.outerjoin(
            lk,
            sa.and_(
                *[base.c[pair.base_col] == lk.c[f"lk_{pair.lookup_col}"] for pair in via.on]
            ),
        )
    else:
        event = base.c[table.event_time]
        matched = sa.true()
        source = base

    lag = DateDiffDay(event, load)
    both = sa.and_(matched, event.is_not(None), load.is_not(None))
    is_null_event = sa.and_(matched, event.is_(None))
    is_null_load_only = sa.and_(matched, event.is_not(None), load.is_(None))
    is_negative_excluded = sa.and_(both, lag < -tolerance)
    is_eligible = sa.and_(both, lag >= -tolerance)
    is_clipped = sa.and_(both, lag >= -tolerance, lag < 0)
    is_overflow = sa.and_(is_eligible, lag > cap)

    def flag(condition) -> sa.ColumnElement:
        return sa.case((condition, 1), else_=0)

    # grouping keys are NULL outside their population; the NULL-key artifact
    # rows are dropped by the runner — their mass lives in the flag counts
    month_key = sa.case((is_eligible, MonthFloor(event)), else_=sa.null()).label("event_month")
    lag_key = sa.case(
        (sa.and_(is_eligible, lag > cap), cap + 1),  # frozen overflow sentinel
        (sa.and_(is_eligible, lag < 0), 0),  # clock-skew clip
        (is_eligible, lag),
        else_=sa.null(),
    ).label("lag_day")
    epoch_key = TimeBucket(load, analysis.freshness_bucket).label("load_epoch_day")
    if table.load_batch_col:
        batch_key = sa.case(
            (is_eligible, base.c[table.load_batch_col]), else_=sa.null()
        ).label("batch_id")
    else:
        batch_key = sa.cast(sa.null(), sa.String()).label("batch_id")
    if table.group_by_alt:
        alt_key = base.c[table.group_by_alt].label("alt_value")
    else:
        alt_key = sa.cast(sa.null(), sa.String()).label("alt_value")
    if table.key_cols:
        key_hash = KeyHash(*[base.c[key] for key in table.key_cols]).label("key_hash")
    else:
        key_hash = sa.cast(sa.null(), sa.String()).label("key_hash")
    if via:
        unmatched_flag = flag(lk.c.lk_matched.is_(None))
        lookup_dup = lk.c.lk_dup
    else:
        unmatched_flag = sa.cast(sa.literal(0), sa.Integer())
        lookup_dup = sa.cast(sa.null(), sa.Integer())

    return (
        sa.select(
            month_key,
            lag_key,
            epoch_key,
            batch_key,
            alt_key,
            flag(is_eligible).label("is_eligible"),
            flag(is_null_event).label("is_null_event_time"),
            flag(is_null_load_only).label("is_null_load_time_only"),
            flag(is_clipped).label("is_negative_clipped"),
            flag(is_negative_excluded).label("is_negative_lag_excluded"),
            flag(is_overflow).label("is_overflow"),
            unmatched_flag.label("is_join_unmatched"),
            lookup_dup.label("lookup_dup"),
            key_hash,
            load.label("load_time"),
        )
        .select_from(source)
        # the bare <= would silently delete the NULL-load bucket
        .where(sa.or_(load <= sa.bindparam("as_of", type_=sa.DateTime()), load.is_(None)))
    )


def staging_sql(table: TableConfig, dialect: str, as_of=None) -> str:
    """Wrap the staging select as temp-table creation DDL. When as_of is given
    it is rendered as a literal (the runner's execution form); without it the
    bindparam form is kept (the snapshot form)."""
    select = build_staging_select(table, dialect)
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
        # SELECT ... INTO #mp_probe FROM ... (types inferred); the first FROM
        # in the compiled text is the top-level one (no subqueries precede it)
        return compiled.replace("\nFROM ", f" INTO {staging_table_name(dialect)} \nFROM ", 1)
    return f"CREATE OR REPLACE TEMPORARY TABLE {staging_table_name(dialect)} AS {compiled}"


def _dialect_instance(dialect: str):
    if dialect == "mssql":
        from sqlalchemy.dialects import mssql as mssql_dialect

        return mssql_dialect.dialect()
    return sa.create_engine("duckdb:///:memory:").dialect


def build_aggregation_query(table: TableConfig, dialect: str) -> sa.Select:
    """GROUPING SETS over the staging table — touches the target table ZERO
    times."""
    staging = _staging_clause(dialect)
    month_key = staging.c.event_month
    lag_key = staging.c.lag_day
    epoch_key = staging.c.load_epoch_day

    sets = [[month_key, lag_key], [epoch_key]]
    grouping_id = (
        sa.func.grouping(month_key) * GROUPING_WEIGHTS["event_month"]
        + sa.func.grouping(lag_key) * GROUPING_WEIGHTS["lag_day"]
        + sa.func.grouping(epoch_key) * GROUPING_WEIGHTS["load_epoch_day"]
    )
    if table.load_batch_col:
        batch_key = staging.c.batch_id
        sets.append([month_key, batch_key])
        grouping_id = grouping_id + sa.func.grouping(batch_key) * GROUPING_WEIGHTS["batch_id"]
    else:
        batch_key = staging.c.batch_id
        grouping_id = grouping_id + GROUPING_WEIGHTS["batch_id"]
    if table.group_by_alt:
        alt_key = staging.c.alt_value
        sets.append([alt_key])
        grouping_id = grouping_id + sa.func.grouping(alt_key) * GROUPING_WEIGHTS["alt_value"]
    else:
        alt_key = staging.c.alt_value
        grouping_id = grouping_id + GROUPING_WEIGHTS["alt_value"]
    sets.append([])  # the () set carrying the global scalars

    select_batch = batch_key if table.load_batch_col else sa.func.max(batch_key)
    select_alt = alt_key if table.group_by_alt else sa.func.max(alt_key)
    return (
        sa.select(
            grouping_id.label("grouping_id"),
            month_key.label("event_month"),
            lag_key.label("lag_day"),
            epoch_key.label("load_epoch_day"),
            select_batch.label("batch_id"),
            select_alt.label("alt_value"),
            sa.func.count().label("row_count"),
            sa.func.sum(staging.c.is_eligible).label("n_curve_eligible"),
            sa.func.sum(staging.c.is_null_event_time).label("n_null_event_time"),
            sa.func.sum(staging.c.is_null_load_time_only).label("n_null_load_time_only"),
            sa.func.sum(staging.c.is_negative_clipped).label("n_negative_clipped"),
            sa.func.sum(staging.c.is_negative_lag_excluded).label("n_negative_lag_excluded"),
            sa.func.sum(staging.c.is_overflow).label("n_overflow"),
            sa.func.sum(staging.c.is_join_unmatched).label("n_join_unmatched"),
            sa.func.max(staging.c.lookup_dup).label("max_lookup_dup"),
            sa.cast(sa.null(), sa.BigInteger()).label("distinct_keys"),
            sa.func.min(staging.c.load_time).label("min_load_time"),
        )
        .select_from(staging)
        .group_by(GroupingSetsClause(sets))
    )


def build_uniqueness_query(table: TableConfig, dialect: str) -> sa.Select:
    """The distinct-count uniqueness guard over the STAGED key hashes — the
    target table is not touched again."""
    if not table.key_cols:
        raise ValueError(f"probe {table.probe_name!r} has no key_cols configured")
    staging = _staging_clause(dialect)
    return sa.select(
        sa.func.count().label("total_rows"),
        sa.func.count(sa.distinct(staging.c.key_hash)).label("distinct_keys"),
    ).select_from(staging)


def drop_staging_sql(dialect: str) -> str:
    return f"DROP TABLE {staging_table_name(dialect)}"


# ---------------------------------------------------------------------- runner


@dataclass
class CanonicalResult:
    """Typed access to the canonical aggregation, split by grouping set."""

    frame: pd.DataFrame

    def rows_for(self, set_name: str) -> pd.DataFrame:
        gid = GROUPING_SET_IDS[set_name]
        rows = self.frame[self.frame["grouping_id"] == gid]
        # drop NULL-key artifact rows: their mass is carried by the buckets
        if set_name in ("month_lag", "month_batch"):
            rows = rows[rows["event_month"].notna()]
        elif set_name == "epoch":
            rows = rows[rows["load_epoch_day"].notna()]
        return rows.reset_index(drop=True)

    @property
    def global_row(self) -> pd.Series:
        rows = self.frame[self.frame["grouping_id"] == GROUPING_SET_IDS["global"]]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one global row, got {len(rows)}")
        return rows.iloc[0]


def run_canonical(engine, table: TableConfig, as_of) -> CanonicalResult:
    """Execute stage -> aggregate -> guard -> drop on one connection; abort on
    cap or join-uniqueness violations."""
    dialect = engine.dialect.name
    cap = table.analysis.result_cell_cap
    counts: dict[int, int] = {}
    rows = []
    guard_distinct = None
    with engine.connect() as conn:
        conn.exec_driver_sql(staging_sql(table, dialect, as_of=pd.Timestamp(as_of)))
        try:
            result = conn.execute(build_aggregation_query(table, dialect))
            keys = result.keys()
            for row in result:
                gid = row.grouping_id
                counts[gid] = counts.get(gid, 0) + 1
                if counts[gid] > cap:
                    raise ProbeAborted(
                        ReasonCode.RESULT_CELL_CAP_EXCEEDED,
                        f"probe {table.probe_name!r}: grouping set {gid} exceeded "
                        f"result_cell_cap={cap}",
                    )
                rows.append(tuple(row))
            if table.key_cols:
                guard_row = conn.execute(build_uniqueness_query(table, dialect)).one()
                guard_distinct = int(guard_row.distinct_keys)
        finally:
            conn.exec_driver_sql(drop_staging_sql(dialect))
    frame = pd.DataFrame(rows, columns=list(keys))
    frame["event_month"] = pd.to_datetime(frame["event_month"])
    if guard_distinct is not None:
        global_mask = frame["grouping_id"] == GROUPING_SET_IDS["global"]
        frame.loc[global_mask, "distinct_keys"] = guard_distinct
    result = CanonicalResult(frame=frame)
    if table.event_time_via is not None:
        max_dup = result.global_row["max_lookup_dup"]
        if pd.notna(max_dup) and int(max_dup) > 1:
            raise ProbeAborted(
                ReasonCode.JOIN_NOT_UNIQUE,
                f"probe {table.probe_name!r}: lookup side of event_time_via is not "
                f"unique on the join key (a key matches {int(max_dup)} rows)",
            )
    return result
