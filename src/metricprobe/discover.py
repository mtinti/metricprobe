"""INFORMATION_SCHEMA scanner: date/time column inventory, role-candidate
matching, draft YAML emission (`metricprobe discover`).

Candidate lists are SHIPPED DEFAULTS and overridable per call. The draft is a
starting point, not an answer: required roles with no candidate are emitted
empty with a FIXME comment (the config loader then fails loudly until a human
chooses), and the optional roles (source_insert_time, load_batch_col) are
emitted as commented suggestions — a wrong guess there would silently change
what gets probed.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa

from metricprobe.config import CONFIG_SCHEMA_VERSION

# column data types that can carry an event/load timestamp (mssql + duckdb
# spellings; bare "time" — time-of-day — is deliberately not one of them)
DATETIME_TYPES = {
    "date",
    "datetime",
    "datetime2",
    "smalldatetime",
    "datetimeoffset",
    "timestamp",
    "timestamp with time zone",
    "timestamp without time zone",
    "timestamp_ns",
    "timestamptz",
}

# shipped default role candidates: case-insensitive substrings, strongest
# first. Override any subset via the `candidates` argument.
DEFAULT_ROLE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "event_time": ("event", "occurred", "recorded", "effective", "activity", "date_of"),
    "load_time": ("load", "ingest", "etl", "landed", "imported", "arrival"),
    "source_insert_time": ("insert", "created", "source"),
    "load_batch_col": ("batch", "run", "job", "file"),
}

TIME_ROLES = ("event_time", "load_time", "source_insert_time")


@dataclass(frozen=True)
class ColumnInfo:
    table_schema: str
    table: str
    column: str
    data_type: str
    ordinal: int

    @property
    def is_datetime(self) -> bool:
        lowered = self.data_type.lower()
        return lowered in DATETIME_TYPES or lowered.startswith(("timestamp", "datetime"))


def scan_columns(engine, database: str, schema: str | None = None) -> list[ColumnInfo]:
    """Every column in the database (optionally one schema), via
    INFORMATION_SCHEMA — a metadata read, never a table scan."""
    if engine.dialect.name == "mssql":
        sql = (
            "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION "
            "FROM [{db}].INFORMATION_SCHEMA.COLUMNS"
        ).format(db=database.replace("]", "]]"))
        params: dict[str, str] = {}
        if schema:
            sql += " WHERE TABLE_SCHEMA = :s"
            params["s"] = schema
    else:
        sql = (
            "SELECT table_schema, table_name, column_name, data_type, ordinal_position "
            "FROM information_schema.columns WHERE table_catalog = :c"
        )
        params = {"c": database}
        if schema:
            sql += " AND table_schema = :s"
            params["s"] = schema
    sql += " ORDER BY 1, 2, 5"
    with engine.connect() as conn:
        rows = conn.execute(sa.text(sql), params).all()
    return [
        ColumnInfo(
            table_schema=row[0], table=row[1], column=row[2], data_type=row[3], ordinal=row[4]
        )
        for row in rows
    ]


def datetime_inventory(columns: list[ColumnInfo]) -> dict[tuple[str, str], list[ColumnInfo]]:
    """(schema, table) -> its date/time columns, for tables that have any."""
    inventory: dict[tuple[str, str], list[ColumnInfo]] = {}
    for column in columns:
        if column.is_datetime:
            inventory.setdefault((column.table_schema, column.table), []).append(column)
    return inventory


def match_roles(
    table_columns: list[ColumnInfo],
    candidates: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, list[str]]:
    """Role -> ranked candidate column names for ONE table. Time roles match
    date/time columns only; load_batch_col matches any column. Ranked by
    (pattern strength, column position); a column matching both event and
    load goes to load_time (the more distinctive, required role)."""
    patterns = DEFAULT_ROLE_CANDIDATES | (candidates or {})
    ranked: dict[str, list[str]] = {}
    for role, needles in patterns.items():
        pool = [
            column
            for column in table_columns
            if (column.is_datetime if role in TIME_ROLES else True)
        ]
        scored = []
        for column in pool:
            lowered = column.column.lower()
            for strength, needle in enumerate(needles):
                if needle in lowered:
                    scored.append((strength, column.ordinal, column.column))
                    break
        ranked[role] = [name for _, _, name in sorted(scored)]
    for shared in [c for c in ranked["event_time"] if c in ranked["load_time"]]:
        ranked["event_time"].remove(shared)
    return ranked


def draft_config(
    engine,
    database: str,
    connection_url: str,
    schema: str | None = None,
    candidates: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """A draft YAML config for every table carrying date/time columns."""
    columns = scan_columns(engine, database, schema=schema)
    by_table: dict[tuple[str, str], list[ColumnInfo]] = {}
    for column in columns:
        by_table.setdefault((column.table_schema, column.table), []).append(column)
    inventory = datetime_inventory(columns)

    lines = [
        "# metricprobe draft config — generated by `metricprobe discover`.",
        "# Review every FIXME and candidate comment before running: role guesses",
        "# come from column NAMES and must be confirmed by a human.",
        f"schema_version: {CONFIG_SCHEMA_VERSION}",
        f'connection_url: "{connection_url}"',
        "store:",
        "  path: ./metricprobe_store",
        "tables:",
    ]
    for (table_schema, table), datetime_columns in sorted(inventory.items()):
        roles = match_roles(by_table[(table_schema, table)], candidates=candidates)
        described = ", ".join(f"{c.column} ({c.data_type})" for c in datetime_columns)
        lines.append(f"  # {database}.{table_schema}.{table} — date/time columns: {described}")
        lines.append(f"  - probe_name: {table}_main")
        lines.append(f"    database: {database}")
        lines.append(f"    schema: {table_schema}")
        lines.append(f"    table: {table}")
        for role in ("event_time", "load_time"):
            picks = roles[role]
            if picks:
                others = f"  # other candidates: {', '.join(picks[1:])}" if picks[1:] else ""
                lines.append(f"    {role}: {picks[0]}{others}")
            else:
                pool = ", ".join(c.column for c in datetime_columns)
                lines.append(f"    {role}:  # FIXME — choose one of: {pool}")
        for role in ("source_insert_time", "load_batch_col"):
            picks = [p for p in roles[role] if p not in (roles["load_time"][:1] or [])]
            if picks:
                lines.append(
                    f"    # {role}: {picks[0]}  # candidate — enable only if correct"
                )
    return "\n".join(lines) + "\n"
