"""The canonical-aggregation result schema is FROZEN before any SQL: grouping
set ids, result columns, and the compiled SQL itself (snapshot per dialect —
a T-SQL change appears as a readable diff in review)."""

import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mssql

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import (
    GROUPING_SET_IDS,
    GROUPING_WEIGHTS,
    RESULT_COLUMNS,
    STAGING_COLUMNS,
    build_aggregation_query,
    build_uniqueness_query,
    staging_sql,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def test_grouping_ids_are_frozen():
    assert GROUPING_WEIGHTS == {
        "event_month": 16,
        "lag_day": 8,
        "load_epoch_day": 4,
        "batch_id": 2,
        "alt_value": 1,
    }
    assert GROUPING_SET_IDS == {
        "month_lag": 7,
        "epoch": 27,
        "month_batch": 13,
        "alt": 30,
        "global": 31,
    }


def test_result_columns_are_frozen():
    assert RESULT_COLUMNS == (
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
    assert STAGING_COLUMNS == (
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


def _config_basic() -> TableConfig:
    return TableConfig.model_validate(
        {
            "probe_name": "orders_main",
            "database": "demo_retail",
            "schema": "dbo",
            "table": "orders",
            "event_time": "order_date",
            "load_time": "loaded_at",
        }
    )


def _config_batch_alt_keys() -> TableConfig:
    return TableConfig.model_validate(
        {
            "probe_name": "settlements_main",
            "database": "demo_finance",
            "schema": "dbo",
            "table": "settlements",
            "event_time": "settled_on",
            "load_time": "loaded_at",
            "load_batch_col": "batch_run",
            "group_by_alt": "region",
            "key_cols": ["settlement_id", "leg"],
        }
    )


def _config_via_composite() -> TableConfig:
    return TableConfig.model_validate(
        {
            "probe_name": "episodes_via",
            "database": "demo_health",
            "schema": "dbo",
            "table": "episodes",
            "load_time": "loaded_at",
            "event_time_via": {
                "join_table": "demo_health.dbo.referrals",
                "on": [
                    {"base_col": "referral_id", "lookup_col": "id"},
                    {"base_col": "site_code", "lookup_col": "site"},
                ],
                "column": "referral_date",
            },
        }
    )


CASES = {
    "canonical_basic": _config_basic,
    "canonical_batch_alt_keys": _config_batch_alt_keys,
    "canonical_via_composite": _config_via_composite,
}


def _dialect(dialect_name: str):
    if dialect_name == "mssql":
        return mssql.dialect()
    return sa.create_engine("duckdb:///:memory:").dialect


def _statements(config: TableConfig, dialect_name: str) -> dict[str, str]:
    out = {
        "staging": staging_sql(config, dialect_name),
        "aggregation": str(
            build_aggregation_query(config, dialect_name).compile(dialect=_dialect(dialect_name))
        ),
    }
    if config.key_cols:
        out["uniqueness"] = str(
            build_uniqueness_query(config, dialect_name).compile(dialect=_dialect(dialect_name))
        )
    return out


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("dialect", ["duckdb", "mssql"])
def test_compiled_sql_matches_snapshots(case, dialect):
    for kind, compiled in _statements(CASES[case](), dialect).items():
        path = SNAPSHOT_DIR / f"{case}_{kind}_{dialect}.sql"
        if os.environ.get("UPDATE_SNAPSHOTS"):
            SNAPSHOT_DIR.mkdir(exist_ok=True)
            path.write_text(compiled + "\n")
        assert path.exists(), f"snapshot missing: run UPDATE_SNAPSHOTS=1 pytest {__name__}"
        assert compiled + "\n" == path.read_text(), (
            f"compiled {kind} SQL for {case} ({dialect}) changed; if intentional, "
            f"regenerate with UPDATE_SNAPSHOTS=1 and review the diff"
        )


def test_only_staging_touches_the_target_table():
    # the scan budget is enforced by construction: ONE scan of the target table
    # (the staging statement); aggregation and guard read only the temp table
    for dialect in ("duckdb", "mssql"):
        statements = _statements(_config_batch_alt_keys(), dialect)
        assert statements["staging"].count("FROM demo_finance.dbo.settlements") == 1
        assert "demo_finance" not in statements["aggregation"]
        assert "demo_finance" not in statements["uniqueness"]
        via = _statements(_config_via_composite(), dialect)
        assert via["staging"].count("FROM demo_health.dbo.episodes") == 1
        assert via["staging"].count("FROM demo_health.dbo.referrals") == 1
        assert "demo_health" not in via["aggregation"]


def test_as_of_predicate_keeps_null_loads():
    # the WHERE clause must be (load <= :as_of OR load IS NULL) — the bare <=
    # would silently delete the NULL-load bucket reconciliation requires
    for case in CASES.values():
        for dialect in ("duckdb", "mssql"):
            sql = _statements(case(), dialect)["staging"]
            assert "IS NULL" in sql.split("WHERE", 1)[1]
