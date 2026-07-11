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
    staging_sql,
)

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def test_schema_version_is_pinned():
    # v2: the scan-budget accounting formulas changed (ALGORITHMS section 15);
    # changing any frozen formula must bump this deliberately
    from metricprobe.extract.canonical import CANONICAL_SCHEMA_VERSION

    assert CANONICAL_SCHEMA_VERSION == 2


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
        "n_other_exclusions",
        "n_base_rows",
        "n_ambiguous_base_rows",
        "n_compare_mismatch",
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
        "is_base_row",
        "is_ambiguous_base",
        "is_compare_mismatch",
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
            "resolution": {"order_date": "date", "loaded_at": "datetime"},
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
            "compare_event_time": "settled_on_raw",
            "resolution": {"settled_on": "date", "loaded_at": "datetime"},
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
            "resolution": {"referral_date": "date", "loaded_at": "datetime"},
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


# declared key-column types for the snapshot form (the runner fetches these
# from INFORMATION_SCHEMA at probe time)
KEY_TYPES = {"settlement_id": "bigint", "leg": "varchar"}


def _statements(config: TableConfig, dialect_name: str) -> dict[str, str]:
    return {
        "staging": staging_sql(
            config, dialect_name, key_types=KEY_TYPES if config.key_cols else None
        ),
        "aggregation": str(
            build_aggregation_query(config, dialect_name).compile(dialect=_dialect(dialect_name))
        ),
    }


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
    # (the staging statement); the aggregation reads only the temp table
    for dialect in ("duckdb", "mssql"):
        statements = _statements(_config_batch_alt_keys(), dialect)
        assert statements["staging"].count("FROM demo_finance.dbo.settlements") == 1
        assert "demo_finance" not in statements["aggregation"]
        via = _statements(_config_via_composite(), dialect)
        assert via["staging"].count("FROM demo_health.dbo.episodes") == 1
        assert via["staging"].count("FROM demo_health.dbo.referrals") == 1
        assert "demo_health" not in via["aggregation"]


def test_empty_grouping_set_row_carries_the_uniqueness_scalars():
    # the frozen contract: the () row of the ONE canonical statement carries
    # COUNT(*) vs COUNT(DISTINCT key_hash) — no separate patched-in query
    for dialect in ("duckdb", "mssql"):
        sql = _statements(_config_batch_alt_keys(), dialect)["aggregation"]
        assert "UNION ALL" in sql
        assert "count(DISTINCT" in sql or "count(distinct" in sql.lower()


def test_scan_budget_verification_fails_closed():
    from metricprobe.extract.canonical import ProbeAborted, verify_scan_budget
    from metricprobe.status import ReasonCode

    assert verify_scan_budget(3000, 1100, "p") == (3000, 3300)
    with pytest.raises(ProbeAborted) as excinfo:
        verify_scan_budget(3301, 1100, "p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_EXCEEDED
    # unmeasurable is an ABORT, never a silent unbudgeted run
    with pytest.raises(ProbeAborted) as excinfo:
        verify_scan_budget(None, 1100, "p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_UNVERIFIABLE
    with pytest.raises(ProbeAborted) as excinfo:
        verify_scan_budget(3000, None, "p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_UNVERIFIABLE


def test_scratch_budget_counts_the_guard_and_fails_closed():
    from metricprobe.extract.canonical import ProbeAborted, verify_scratch_budget
    from metricprobe.status import ReasonCode

    # 4 branches (month_lag, epoch, month_batch, global/distinct) + 1 spare,
    # x 2800 pages, plus the 6/row sort-spool allowance over 1000 staged rows
    assert verify_scratch_budget(11_300, 2_800, 4, 1000, "p") == (11_300, 20_000)
    with pytest.raises(ProbeAborted) as excinfo:
        verify_scratch_budget(20_001, 2_800, 4, 1000, "p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_EXCEEDED
    with pytest.raises(ProbeAborted) as excinfo:
        verify_scratch_budget(None, 2_800, 4, 1000, "p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_UNVERIFIABLE
    with pytest.raises(ProbeAborted) as excinfo:
        verify_scratch_budget(11_300, None, 4, 1000, "p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_UNVERIFIABLE


def test_as_of_predicate_keeps_null_loads():
    # the WHERE clause must be (load <= :as_of OR load IS NULL) — the bare <=
    # would silently delete the NULL-load bucket reconciliation requires
    for case in CASES.values():
        for dialect in ("duckdb", "mssql"):
            sql = _statements(case(), dialect)["staging"]
            assert "IS NULL" in sql.split("WHERE", 1)[1]


def test_key_hash_encoding_is_type_tagged():
    # the hard rule: type-tagged, length-prefixed, distinct NULL sentinel —
    # tags are the DECLARED column types, embedded as literals
    mssql_sql = _statements(_config_batch_alt_keys(), "mssql")["staging"]
    assert "CAST('bigint' AS varbinary(64))" in mssql_sql
    assert "CAST('varchar' AS varbinary(64))" in mssql_sql
    assert "HASHBYTES('SHA2_256'" in mssql_sql and "CHECKSUM" not in mssql_sql
    duckdb_sql = _statements(_config_batch_alt_keys(), "duckdb")["staging"]
    assert "'bigint'" in duckdb_sql and "sha256(" in duckdb_sql
    # missing declared types fail loudly at build time
    with pytest.raises(ValueError, match="declared types required"):
        staging_sql(_config_batch_alt_keys(), "duckdb")


def test_aggregation_carries_a_server_side_row_bound():
    # the database must never return unbounded rows, independent of the
    # client-side per-set cell counting
    mssql_sql = _statements(_config_batch_alt_keys(), "mssql")["aggregation"]
    assert "TOP" in mssql_sql
    duckdb_sql = _statements(_config_batch_alt_keys(), "duckdb")["aggregation"]
    assert "LIMIT" in duckdb_sql


def test_scan_budget_check_aborts_with_reason_code():
    from metricprobe.extract.canonical import ProbeAborted, check_scan_budget
    from metricprobe.status import ReasonCode

    check_scan_budget(target_reads=3000, budget_reads=3300, probe_name="p")  # within
    with pytest.raises(ProbeAborted) as excinfo:
        check_scan_budget(target_reads=3301, budget_reads=3300, probe_name="p")
    assert excinfo.value.reason is ReasonCode.SCAN_BUDGET_EXCEEDED

# ------------------------------------------------------------- dual-lag pass


def test_dual_schema_is_frozen():
    from metricprobe.extract.dual import (
        DUAL_GROUPING_SET_IDS,
        DUAL_GROUPING_WEIGHTS,
        DUAL_RESULT_COLUMNS,
    )

    assert DUAL_GROUPING_WEIGHTS == {"event_month": 4, "lag_day": 2, "delta_day": 1}
    assert DUAL_GROUPING_SET_IDS == {"month_src_lag": 1, "delta": 6, "global": 7}
    assert DUAL_RESULT_COLUMNS == (
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
        "max_lookup_dup",
    )


def _config_dual() -> TableConfig:
    return TableConfig.model_validate(
        {
            "probe_name": "orders_dual",
            "database": "demo_retail",
            "schema": "dbo",
            "table": "orders",
            "event_time": "order_date",
            "load_time": "loaded_at",
            "source_insert_time": "src_inserted_at",
            "resolution": {
                "order_date": "date",
                "loaded_at": "datetime",
                "src_inserted_at": "datetime",
            },
        }
    )


@pytest.mark.parametrize("dialect", ["duckdb", "mssql"])
def test_dual_sql_matches_snapshots(dialect):
    from metricprobe.extract.dual import build_dual_aggregation_query, dual_staging_sql

    statements = {
        "staging": dual_staging_sql(_config_dual(), dialect),
        "aggregation": str(
            build_dual_aggregation_query(_config_dual(), dialect).compile(
                dialect=_dialect(dialect)
            )
        ),
    }
    for kind, compiled in statements.items():
        path = SNAPSHOT_DIR / f"dual_{kind}_{dialect}.sql"
        if os.environ.get("UPDATE_SNAPSHOTS"):
            path.write_text(compiled + "\n")
        assert compiled + "\n" == path.read_text()
    # the dual pass reads the target exactly once, in its staging statement
    assert statements["staging"].count("FROM demo_retail.dbo.orders") == 1
    assert "demo_retail" not in statements["aggregation"]


def test_dual_requires_source_and_direct_event_time():
    from metricprobe.extract.dual import build_dual_staging_select

    with pytest.raises(ValueError, match="source_insert_time"):
        build_dual_staging_select(_config_basic(), "duckdb")
