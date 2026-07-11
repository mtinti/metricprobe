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
is wrapped with COUNT(*) OVER (PARTITION BY join keys) and a second layer
carries MAX(dup) OVER () — the GLOBAL lookup-side worst key — on every joined
row, so MAX(lookup_dup) > 1 aborts the probe (JOIN_NOT_UNIQUE) even when the
duplicated key is not referenced by any base row (the uniqueness contract
covers the lookup table itself).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ClauseElement, FunctionElement

from metricprobe.config import TableConfig
from metricprobe.status import ReasonCode

# v2: the scan-budget accounting formulas changed (row-linear scratch bound,
# enforced spool bound) — ALGORITHMS.md section 15 records the history.
# v3: the staged lookup_dup carries the GLOBAL lookup-side max duplication
# (a second window layer), so JOIN_NOT_UNIQUE fires on duplicate lookup keys
# even when no current base row references them
CANONICAL_SCHEMA_VERSION = 3

# frozen lag_day sentinel for negative-excluded rows: they carry their event
# month (parity's watermarked population) but never enter curves or volumes
NEGATIVE_LAG_SENTINEL = -1

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
    "is_base_row",
    "is_ambiguous_base",
    "is_compare_mismatch",
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
    "n_other_exclusions",  # reserved, 0 in v1 — part of the reconciliation contract
    "n_base_rows",  # pre-join base count (via probes); == row_count otherwise
    "n_ambiguous_base_rows",  # base rows matching >1 lookup rows (via probes)
    "n_compare_mismatch",  # raw-vs-corrected side-stat (compare_event_time)
    "max_lookup_dup",
    "distinct_keys",
    "min_load_time",
)

# A result CELL is one entry of a grouping set's matrix (e.g. one event-month x
# lag-day combination). result_cell_cap bounds cells per set; the aggregation
# query also carries a server-side row limit of cap x set-count + 1 so the
# database never returns unbounded rows even before the client counts them.


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
    NULL sentinel (ALGORITHMS.md section 9). The type tag is each column's
    DECLARED type, fetched once from INFORMATION_SCHEMA (a per-row function
    like sql_variant would reject varchar(max) and cost a call per row). Each
    dialect's encoding is injective on its own; equivalence compares COUNTS,
    not hash bytes. Never 32-bit CHECKSUM."""

    name = "key_hash"
    inherit_cache = False

    def __init__(self, type_tags: list[str], *cols):
        if len(type_tags) != len(cols):
            raise ValueError("one type tag per key column required")
        self.type_tags = type_tags
        super().__init__(*cols)


def _sql_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


@compiles(KeyHash, "mssql")
def _key_hash_mssql(element, compiler, **kw):
    # per column: 0x00 for NULL, else 0x01 + len(tag) + tag + len(value) + value
    parts = []
    for tag, clause in zip(element.type_tags, element.clauses, strict=True):
        col = compiler.process(clause, **kw)
        tag_binary = f"CAST({_sql_literal(tag)} AS varbinary(64))"
        binary = f"CAST({col} AS varbinary(max))"
        parts.append(
            f"CASE WHEN {col} IS NULL THEN 0x00 ELSE CAST(0x01 AS varbinary(max)) "
            f"+ CAST(CAST(DATALENGTH({tag_binary}) AS int) AS binary(4)) + {tag_binary} "
            f"+ CAST(CAST(DATALENGTH({binary}) AS int) AS binary(4)) + {binary} END"
        )
    return f"HASHBYTES('SHA2_256', {' + '.join(parts)})"


@compiles(KeyHash)
def _key_hash_default(element, compiler, **kw):
    # same shape on duckdb: NULL sentinel, then length-prefixed type tag and
    # length-prefixed value
    parts = []
    for tag, clause in zip(element.type_tags, element.clauses, strict=True):
        col = compiler.process(clause, **kw)
        tag_literal = _sql_literal(tag)
        text = f"CAST({col} AS VARCHAR)"
        parts.append(
            f"CASE WHEN {col} IS NULL THEN chr(0) ELSE '1' || "
            f"lpad(CAST(length({tag_literal}) AS VARCHAR), 10, '0') || {tag_literal} || "
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


def build_staging_select(
    table: TableConfig, dialect: str, key_types: dict[str, str] | None = None
) -> sa.Select:
    """The ONLY statement that reads the target table: one scan projecting the
    narrow derived columns, with bindparam `as_of`. `key_types` maps each key
    column to its declared SQL type (the hash's type tag); required when
    key_cols are configured — the runner fetches them from INFORMATION_SCHEMA."""
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
    if table.compare_event_time:
        base_columns.add(table.compare_event_time)
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
        partition = [lookup.c[pair.lookup_col] for pair in via.on]
        lk_inner = (
            sa.select(
                *[lookup.c[col].label(f"lk_{col}") for col in sorted(lookup_cols)],
                sa.literal(1).label("lk_matched"),
                sa.func.count().over(partition_by=partition).label("lk_dup"),
                # numbers a base row's matches so it is counted ONCE in the
                # pre-join base count even when the lookup is ambiguous
                sa.func.row_number()
                .over(partition_by=partition, order_by=partition)
                .label("lk_rn"),
            ).subquery("lk_inner")
        )
        # the GLOBAL max duplication over the whole lookup side, carried on
        # every joined row: the uniqueness contract covers the lookup table
        # itself, so duplicate keys that no base row references must still
        # surface (window nesting is illegal, hence the second layer)
        lk = sa.select(
            *lk_inner.c,
            sa.func.max(lk_inner.c.lk_dup).over().label("lk_maxdup"),
        ).subquery("lk")
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
    # rows are dropped by the runner — their mass lives in the flag counts.
    # The month key covers eligible AND negative-excluded rows so parity can
    # compare the full WATERMARKED per-month population; excluded rows land at
    # the frozen lag sentinel -1 and are filtered out of curves/volumes.
    month_key = sa.case(
        (sa.or_(is_eligible, is_negative_excluded), MonthFloor(event)), else_=sa.null()
    ).label("event_month")
    lag_key = sa.case(
        (is_negative_excluded, NEGATIVE_LAG_SENTINEL),
        (sa.and_(is_eligible, lag > cap), cap + 1),  # frozen overflow sentinel
        (sa.and_(is_eligible, lag < 0), 0),  # clock-skew clip
        (is_eligible, lag),
        else_=sa.null(),
    ).label("lag_day")
    epoch_key = TimeBucket(load, analysis.freshness_bucket).label("load_epoch_day")
    if table.load_batch_col:
        # keyed by LOAD presence, not eligibility: a batch whose rows have
        # corrupt/null event times is still a real arrival epoch
        batch_key = sa.case(
            (load.is_not(None), base.c[table.load_batch_col]), else_=sa.null()
        ).label("batch_id")
    else:
        batch_key = sa.cast(sa.null(), sa.String()).label("batch_id")
    if table.group_by_alt:
        alt_key = base.c[table.group_by_alt].label("alt_value")
    else:
        alt_key = sa.cast(sa.null(), sa.String()).label("alt_value")
    if table.key_cols:
        missing = [key for key in table.key_cols if key not in (key_types or {})]
        if missing:
            raise ValueError(
                f"probe {table.probe_name!r}: declared types required for key "
                f"column(s) {missing} (the hash type tag)"
            )
        key_hash = KeyHash(
            [key_types[key] for key in table.key_cols],
            *[base.c[key] for key in table.key_cols],
        ).label("key_hash")
    else:
        key_hash = sa.cast(sa.null(), sa.String()).label("key_hash")
    if via:
        unmatched_flag = flag(lk.c.lk_matched.is_(None))
        # staged as the GLOBAL lookup-side max: MAX(lookup_dup) in the result
        # is then the whole lookup table's worst key, joined-or-not (v3)
        lookup_dup = lk.c.lk_maxdup
        # pre-join base count: unmatched rows plus the FIRST match per base row
        base_row_flag = flag(sa.or_(lk.c.lk_matched.is_(None), lk.c.lk_rn == 1))
        ambiguous_flag = flag(sa.and_(lk.c.lk_rn == 1, lk.c.lk_dup > 1))
    else:
        unmatched_flag = sa.cast(sa.literal(0), sa.Integer())
        lookup_dup = sa.cast(sa.null(), sa.Integer())
        base_row_flag = sa.cast(sa.literal(1), sa.Integer())
        ambiguous_flag = sa.cast(sa.literal(0), sa.Integer())

    if table.compare_event_time:
        compare = base.c[table.compare_event_time]
        # among curve-eligible rows: dates differ at day grain, or compare NULL
        compare_mismatch = flag(
            sa.and_(
                is_eligible,
                sa.or_(compare.is_(None), DateDiffDay(event, compare) != 0),
            )
        )
    else:
        compare_mismatch = sa.cast(sa.literal(0), sa.Integer())
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
            base_row_flag.label("is_base_row"),
            ambiguous_flag.label("is_ambiguous_base"),
            compare_mismatch.label("is_compare_mismatch"),
            lookup_dup.label("lookup_dup"),
            key_hash,
            load.label("load_time"),
        )
        .select_from(source)
        # the bare <= would silently delete the NULL-load bucket
        .where(sa.or_(load <= sa.bindparam("as_of", type_=sa.DateTime()), load.is_(None)))
    )


def staging_sql(
    table: TableConfig,
    dialect: str,
    as_of=None,
    key_types: dict[str, str] | None = None,
) -> str:
    """Wrap the staging select as temp-table creation DDL. When as_of is given
    it is rendered as a literal (the runner's execution form); without it the
    bindparam form is kept (the snapshot form)."""
    select = build_staging_select(table, dialect, key_types=key_types)
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


def _bucket_sums(staging):
    """The aggregate columns shared by the grouped rows and the () branch.
    COALESCE(SUM(..), 0): a SUM over zero rows is NULL, and reconciliation
    arithmetic must survive an empty table."""

    def total(col):
        return sa.func.coalesce(sa.func.sum(col), 0)

    return [
        sa.func.count().label("row_count"),
        total(staging.c.is_eligible).label("n_curve_eligible"),
        total(staging.c.is_null_event_time).label("n_null_event_time"),
        total(staging.c.is_null_load_time_only).label("n_null_load_time_only"),
        total(staging.c.is_negative_clipped).label("n_negative_clipped"),
        total(staging.c.is_negative_lag_excluded).label("n_negative_lag_excluded"),
        total(staging.c.is_overflow).label("n_overflow"),
        total(staging.c.is_join_unmatched).label("n_join_unmatched"),
        sa.cast(sa.literal(0), sa.BigInteger()).label("n_other_exclusions"),
        total(staging.c.is_base_row).label("n_base_rows"),
        total(staging.c.is_ambiguous_base).label("n_ambiguous_base_rows"),
        total(staging.c.is_compare_mismatch).label("n_compare_mismatch"),
        sa.func.max(staging.c.lookup_dup).label("max_lookup_dup"),
    ]


def build_aggregation_query(table: TableConfig, dialect: str):
    """ONE statement over the staging table (the target table is touched ZERO
    times): the dimensional grouping sets UNION ALL the () global row, which
    carries COUNT(*) vs COUNT(DISTINCT key_hash) — the canonical result's
    empty grouping set holds the uniqueness scalars, per the frozen contract.
    (Computing the DISTINCT inside the GROUPING SETS plan would spool per row;
    the union branch is a plain single-pass aggregate.)"""
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
    batch_key = staging.c.batch_id
    if table.load_batch_col:
        sets.append([month_key, batch_key])
        grouping_id = grouping_id + sa.func.grouping(batch_key) * GROUPING_WEIGHTS["batch_id"]
    else:
        grouping_id = grouping_id + GROUPING_WEIGHTS["batch_id"]
    alt_key = staging.c.alt_value
    if table.group_by_alt:
        sets.append([alt_key])
        grouping_id = grouping_id + sa.func.grouping(alt_key) * GROUPING_WEIGHTS["alt_value"]
    else:
        grouping_id = grouping_id + GROUPING_WEIGHTS["alt_value"]

    select_batch = batch_key if table.load_batch_col else sa.func.max(batch_key)
    select_alt = alt_key if table.group_by_alt else sa.func.max(alt_key)
    grouped = (
        sa.select(
            grouping_id.label("grouping_id"),
            month_key.label("event_month"),
            lag_key.label("lag_day"),
            epoch_key.label("load_epoch_day"),
            select_batch.label("batch_id"),
            select_alt.label("alt_value"),
            *_bucket_sums(staging),
            sa.cast(sa.null(), sa.BigInteger()).label("distinct_keys"),
            sa.func.min(staging.c.load_time).label("min_load_time"),
        )
        .select_from(staging)
        .group_by(GroupingSetsClause(sets))
    )
    if table.key_cols:
        distinct_keys = sa.func.count(sa.distinct(staging.c.key_hash))
    else:
        distinct_keys = sa.cast(sa.null(), sa.BigInteger())
    global_row = sa.select(
        sa.literal(GROUPING_SET_IDS["global"]).label("grouping_id"),
        sa.cast(sa.null(), sa.Date()).label("event_month"),
        sa.cast(sa.null(), sa.Integer()).label("lag_day"),
        sa.cast(sa.null(), sa.Date()).label("load_epoch_day"),
        sa.cast(sa.null(), sa.String()).label("batch_id"),
        sa.cast(sa.null(), sa.String()).label("alt_value"),
        *_bucket_sums(staging),
        distinct_keys.label("distinct_keys"),
        sa.func.min(staging.c.load_time).label("min_load_time"),
    ).select_from(staging)
    # the union is wrapped so the row bound applies to a PLAIN select: mssql
    # silently drops .limit() on a compound select (no TOP/FETCH rendered)
    compound = grouped.union_all(global_row).subquery("canonical")
    return (
        sa.select(*[compound.c[name] for name in RESULT_COLUMNS])
        .order_by(compound.c.grouping_id)
        # server-side bound: the database never returns unbounded rows; the
        # runner still enforces the per-set cell cap while streaming
        .limit(table.analysis.result_cell_cap * (len(sets) + 1) + 1)
    )


def drop_staging_sql(dialect: str) -> str:
    return f"DROP TABLE {staging_table_name(dialect)}"


# ------------------------------------------------------ scan-budget machinery

_STATS_IO_LINE = re.compile(r"Table '([^']+)'\. Scan count \d+, logical reads (\d+)")


def check_scan_budget(target_reads: int, budget_reads: int, probe_name: str) -> None:
    """The frozen numeric budget: logical reads on the TARGET table(s) <= 3x
    one full scan per probe. Exceeding it ABORTS with SCAN_BUDGET_EXCEEDED."""
    if target_reads > budget_reads:
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_EXCEEDED,
            f"probe {probe_name!r}: {target_reads} logical reads on the target "
            f"table(s) exceed the budget of {budget_reads} (3x one full scan)",
        )


def _bracket(name: str) -> str:
    return "[" + name.replace("]", "]]") + "]"


def _mssql_target_pages(conn, table: TableConfig) -> int | None:
    """used_page_count of the probed table (+ lookup for via probes); None when
    the DMV is not readable (budget then unenforced, by design)."""
    locators = [(table.database, table.table_schema, table.table)]
    if table.event_time_via is not None:
        via = table.event_time_via
        locators.append((via.database, via.table_schema, via.table))
    total = 0
    for database, schema, name in locators:
        try:
            pages = conn.execute(
                sa.text(
                    f"SELECT SUM(used_page_count) FROM {_bracket(database)}.sys."
                    "dm_db_partition_stats WHERE object_id = OBJECT_ID(:full_name)"
                ),
                {"full_name": f"{database}.{schema}.{name}"},
            ).scalar()
        except sa.exc.DBAPIError:
            return None
        if pages is None:
            return None
        total += int(pages)
    return total


def _key_column_types(conn, table: TableConfig, dialect: str) -> dict[str, str]:
    """Declared types of the key columns from INFORMATION_SCHEMA — a metadata
    lookup, not a table scan. Used as the hash encoding's type tags."""
    if dialect == "mssql":
        sql = (
            f"SELECT COLUMN_NAME, DATA_TYPE FROM {_bracket(table.database)}."
            "INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t"
        )
        params = {"s": table.table_schema, "t": table.table}
    else:
        sql = (
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_catalog = :c AND table_schema = :s AND table_name = :t"
        )
        params = {"c": table.database, "s": table.table_schema, "t": table.table}
    types = {name: data_type for name, data_type in conn.execute(sa.text(sql), params)}
    missing = [key for key in table.key_cols or () if key not in types]
    if missing:
        raise ProbeAborted(
            ReasonCode.MISSING_TABLE,
            f"probe {table.probe_name!r}: key column(s) {missing} not found in "
            f"{table.database}.{table.table_schema}.{table.table}",
        )
    return types


def _install_message_capture(conn, messages: list[str]) -> bool:
    """Route server info messages (STATISTICS IO) into `messages`. Supported
    for pymssql (message handler); best-effort otherwise."""
    driver = getattr(conn.connection, "driver_connection", None)
    inner = getattr(driver, "_conn", None)
    if hasattr(inner, "set_msghandler"):

        def handler(msgstate, severity, srvname, procname, line, text):
            messages.append(text.decode() if isinstance(text, bytes) else str(text))

        inner.set_msghandler(handler)
        return True
    return False


def _harvest_cursor_messages(result, messages: list[str]) -> None:
    """pyodbc exposes info messages on cursor.messages instead."""
    cursor_messages = getattr(getattr(result, "cursor", None), "messages", None)
    for entry in cursor_messages or []:
        messages.append(str(entry[-1] if isinstance(entry, tuple) else entry))


def _reads_by_table(messages: list[str]) -> dict[str, int]:
    reads: dict[str, int] = {}
    for message in messages:
        for name, count in _STATS_IO_LINE.findall(message):
            reads[name] = reads.get(name, 0) + int(count)
    return reads


def _target_reads_from(messages: list[str], target_tables: set[str]) -> int | None:
    # SQL Server identifiers are case-insensitive under the usual collations:
    # a config naming the table in different case still executes, so the
    # STATISTICS IO attribution must fold case too, or a legitimate probe
    # would abort as unverifiable
    reads = _reads_by_table(messages)
    targets = {name.casefold() for name in target_tables}
    matched = {
        name: count for name, count in reads.items() if name.casefold() in targets
    }
    return sum(matched.values()) if matched else None


def _scratch_reads_from(messages: list[str], staging_prefix: str) -> int | None:
    """Everything that is not a target table: the staging temp (its tempdb
    name is the prefix padded with underscores) plus worktables/workfiles."""
    reads = _reads_by_table(messages)
    if not reads:
        return None
    prefix = staging_prefix.casefold()
    return sum(
        count
        for name, count in reads.items()
        if name.casefold().startswith(prefix) or name in ("Worktable", "Workfile")
    )


def _mssql_staging_pages(conn, staging_name: str) -> int | None:
    try:
        pages = conn.execute(
            sa.text(
                "SELECT SUM(used_page_count) FROM tempdb.sys.dm_db_partition_stats "
                "WHERE object_id = OBJECT_ID(:name)"
            ),
            {"name": f"tempdb..{staging_name}"},
        ).scalar()
    except sa.exc.DBAPIError:
        return None
    return int(pages) if pages is not None else None


# ---------------------------------------------------------------------- runner


def aggregation_branch_count(table: TableConfig) -> int:
    """Scans of the staging table by construction: one per grouping set in the
    grouped branch (month_lag, epoch [, month_batch][, alt]) plus the () branch
    carrying the global scalars and the distinct-count guard."""
    return 3 + (1 if table.load_batch_col else 0) + (1 if table.group_by_alt else 0)


@dataclass
class CanonicalResult:
    """Typed access to the canonical aggregation, split by grouping set."""

    frame: pd.DataFrame
    target_logical_reads: int | None = None  # measured on mssql when supported
    scan_budget_reads: int | None = None  # 3x one full scan of the target(s)
    scratch_logical_reads: int | None = None  # aggregation reads incl. the guard
    scratch_budget_reads: int | None = None  # (branches + 1) x staging pages
    staging_spool_reads: int | None = None  # staging stmt worktables (via window spool)

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


def verify_scan_budget(
    target_reads: int | None, pages: int | None, probe_name: str
) -> tuple[int, int]:
    """Fail-CLOSED budget verification for mssql: if either the measured reads
    (STATISTICS IO capture) or the table size (dm_db_partition_stats) is
    unavailable, the probe ABORTS as unverifiable rather than silently running
    unbudgeted. Exceeding the budget aborts with its own reason code."""
    if pages is None:  # 0 is valid: an empty table has a 0-read, 0-budget scan
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_UNVERIFIABLE,
            f"probe {probe_name!r}: cannot read the target table's page count "
            "(sys.dm_db_partition_stats — the login may need VIEW DATABASE STATE); "
            "the scan budget cannot be verified, refusing to run unbudgeted",
        )
    if target_reads is None:
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_UNVERIFIABLE,
            f"probe {probe_name!r}: no STATISTICS IO output captured from the "
            "driver; the scan budget cannot be verified, refusing to run unbudgeted",
        )
    budget = 3 * pages
    check_scan_budget(target_reads, budget, probe_name)
    return target_reads, budget


SCRATCH_ROW_ALLOWANCE = 6  # sort/spool worktable reads per staged row (measured <=2.6)
SPOOL_ROW_ALLOWANCE = 10  # via-join window spool reads per staged row (measured ~6)


def verify_scratch_budget(
    scratch_reads: int | None,
    staging_pages: int | None,
    branches: int,
    staged_rows: int,
    probe_name: str,
) -> tuple[int, int]:
    """The SCRATCH ledger (ALGORITHMS.md section 15): the aggregation and the
    distinct-count guard read the probe's own staging materialization, never
    the target. Their reads are COUNTED and enforced against the
    by-construction bound: one scan per aggregation branch (+1 spare) PLUS
    sort/spool worktable activity, which is linear in the staged rows (the
    Step 7 audit measured up to ~2.6 reads/row on sort-heavy plans such as
    hour-bucket epochs; 6/row is the tripwire). Fail-closed."""
    if staging_pages is None or scratch_reads is None:
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_UNVERIFIABLE,
            f"probe {probe_name!r}: the scratch-read ledger (staging pages or "
            "STATISTICS IO) is unmeasurable; refusing to run unbudgeted",
        )
    budget = (branches + 1) * max(staging_pages, 1) + SCRATCH_ROW_ALLOWANCE * max(staged_rows, 0)
    if scratch_reads > budget:
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_EXCEEDED,
            f"probe {probe_name!r}: {scratch_reads} scratch logical reads exceed "
            f"the bound of {budget} ({branches} branches + 1 spare, x "
            f"{staging_pages} pages, + {SCRATCH_ROW_ALLOWANCE}/row x {staged_rows} rows)",
        )
    return scratch_reads, budget


def verify_spool_budget(
    spool_reads: int | None, staging_pages: int | None, staged_rows: int, probe_name: str
) -> tuple[int, int]:
    """The staging statement's own worktable spool (the via-join uniqueness
    window functions) is row-proportional; it is ENFORCED fail-closed against
    a row-linear bound, not merely reported (ALGORITHMS.md section 15)."""
    if staging_pages is None or spool_reads is None:
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_UNVERIFIABLE,
            f"probe {probe_name!r}: the staging-spool ledger is unmeasurable; "
            "refusing to run unbudgeted",
        )
    budget = max(staging_pages, 1) + SPOOL_ROW_ALLOWANCE * max(staged_rows, 0)
    if spool_reads > budget:
        raise ProbeAborted(
            ReasonCode.SCAN_BUDGET_EXCEEDED,
            f"probe {probe_name!r}: {spool_reads} staging-spool logical reads exceed "
            f"the bound of {budget} ({staging_pages} pages + "
            f"{SPOOL_ROW_ALLOWANCE}/row x {staged_rows} rows)",
        )
    return spool_reads, budget


def _staged_row_count(rows: list[tuple], keys, global_id: int) -> int:
    """Total staged rows, read from the () grouping-set row already fetched."""
    key_list = list(keys)
    gid_index = key_list.index("grouping_id")
    count_index = key_list.index("row_count")
    for row in rows:
        if row[gid_index] == global_id:
            return int(row[count_index])
    return 0


def run_canonical(engine, table: TableConfig, as_of) -> CanonicalResult:
    """Execute stage -> aggregate (incl. the () uniqueness row) -> drop on one
    connection; abort on scan-budget, cell-cap or join-uniqueness violations."""
    dialect = engine.dialect.name
    cap = table.analysis.result_cell_cap
    counts: dict[int, int] = {}
    rows = []
    target_reads = budget = scratch_reads = scratch_budget = None
    with engine.connect() as conn:
        key_types = _key_column_types(conn, table, dialect) if table.key_cols else None
        messages: list[str] = []
        pages = staging_pages = None
        if dialect == "mssql":
            pages = _mssql_target_pages(conn, table)
            _install_message_capture(conn, messages)
            conn.exec_driver_sql("SET STATISTICS IO ON")
        staging_result = conn.exec_driver_sql(
            staging_sql(table, dialect, as_of=pd.Timestamp(as_of), key_types=key_types)
        )
        _harvest_cursor_messages(staging_result, messages)
        staging_message_count = len(messages)
        spool_reads = None
        try:
            if dialect == "mssql":
                staging_pages = _mssql_staging_pages(conn, staging_table_name(dialect))
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
            _harvest_cursor_messages(result, messages)
        finally:
            # pymssql flushes a statement's STATISTICS IO tokens only when the
            # NEXT statement runs, so the drop both cleans up AND completes the
            # aggregation's message stream — ledgers are computed after it
            conn.exec_driver_sql(drop_staging_sql(dialect))
        if dialect == "mssql":
            # TWO fail-closed ledgers (ALGORITHMS.md section 15): target
            # pressure <= 3x one full scan over ALL statements, and the
            # aggregation statement's scratch reads (its branches INCLUDING
            # the distinct-count guard) <= their by-construction bound
            target_tables = {table.table}
            if table.event_time_via is not None:
                target_tables.add(table.event_time_via.table)
            target_reads, budget = verify_scan_budget(
                _target_reads_from(messages, target_tables), pages, table.probe_name
            )
            scratch_reads, scratch_budget = verify_scratch_budget(
                _scratch_reads_from(
                    messages[staging_message_count:], staging_table_name(dialect)
                ),
                staging_pages,
                aggregation_branch_count(table),
                _staged_row_count(rows, keys, GROUPING_SET_IDS["global"]),
                table.probe_name,
            )
            # the staging statement's own worktable spool (the via uniqueness
            # window functions) is row-proportional: measured AND enforced
            spool_reads, _ = verify_spool_budget(
                _scratch_reads_from(
                    messages[:staging_message_count], staging_table_name(dialect)
                ),
                staging_pages,
                _staged_row_count(rows, keys, GROUPING_SET_IDS["global"]),
                table.probe_name,
            )
    frame = pd.DataFrame(rows, columns=list(keys))
    frame["event_month"] = pd.to_datetime(frame["event_month"])
    result = CanonicalResult(
        frame=frame,
        target_logical_reads=target_reads,
        scan_budget_reads=budget,
        scratch_logical_reads=scratch_reads,
        scratch_budget_reads=scratch_budget,
        staging_spool_reads=spool_reads,
    )
    if table.event_time_via is not None:
        max_dup = result.global_row["max_lookup_dup"]
        if pd.notna(max_dup) and int(max_dup) > 1:
            ambiguous = int(result.global_row["n_ambiguous_base_rows"])
            raise ProbeAborted(
                ReasonCode.JOIN_NOT_UNIQUE,
                f"probe {table.probe_name!r}: lookup side of event_time_via is not "
                f"unique on the join key (worst key matches {int(max_dup)} rows; "
                f"{ambiguous} base rows are ambiguous)",
            )
    return result
