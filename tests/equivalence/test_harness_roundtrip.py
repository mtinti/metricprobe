"""Trivial round-trip proving the equivalence harness end to end (Step 0).

The same small dataset is loaded into DuckDB and SQL Server through SQLAlchemy
Core and an identical aggregate must come back from both. Every later metric
adds its real equivalence cases in the step it lands.
"""

import datetime

import sqlalchemy as sa

ROWS = [
    {"id": 1, "category": "a", "event_time": datetime.datetime(2026, 1, 5, 10, 30)},
    {"id": 2, "category": "a", "event_time": datetime.datetime(2026, 1, 7, 0, 0)},
    {"id": 3, "category": "b", "event_time": datetime.datetime(2026, 2, 1, 23, 59)},
    {"id": 4, "category": "b", "event_time": datetime.datetime(2026, 2, 14, 8, 15)},
    {"id": 5, "category": "b", "event_time": datetime.datetime(2026, 3, 31, 12, 0)},
]


def _load_and_aggregate(engine):
    metadata = sa.MetaData()
    table = sa.Table(
        "mp_equivalence_roundtrip",
        metadata,
        # autoincrement=False: duckdb-engine compiles autoincrement PKs to SERIAL,
        # which DuckDB does not support; ids are always supplied explicitly here.
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
        sa.Column("category", sa.String(10), nullable=False),
        sa.Column("event_time", sa.DateTime, nullable=False),
    )
    with engine.begin() as conn:
        metadata.drop_all(conn)
        metadata.create_all(conn)
        conn.execute(table.insert(), ROWS)
        stmt = (
            sa.select(
                table.c.category,
                sa.func.count().label("n"),
                sa.func.min(table.c.event_time).label("first_event"),
                sa.func.max(table.c.event_time).label("last_event"),
            )
            .group_by(table.c.category)
            .order_by(table.c.category)
        )
        result = [tuple(row) for row in conn.execute(stmt)]
        metadata.drop_all(conn)
    return result


def test_same_aggregate_from_duckdb_and_mssql(duckdb_engine, mssql_engine):
    assert _load_and_aggregate(duckdb_engine) == _load_and_aggregate(mssql_engine)
