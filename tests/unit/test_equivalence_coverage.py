"""Step 7: the EXPLICIT equivalence coverage matrix, enforced.

DuckDB is the fast vehicle; the SQL Server container job is the truth-teller.
Every status channel, every named pathology scenario, every dialect-sensitive
compiled construct, every probe refusal and every store backend must have an
equivalence case. The required list is DERIVED from the code (enums, the
scenario catalog, the @compiles classes), so adding a metric, a scenario or a
construct WITHOUT an equivalence case fails this fast-suite test — review
cannot miss it. COVERAGE maps each item to the equivalence test that proves it
on the real server; mapped names are verified to exist."""

import inspect

import tests.equivalence.test_canonical_equivalence as eqv_core
import tests.equivalence.test_coverage_gaps as eqv_gaps
import tests.equivalence.test_harness_roundtrip as eqv_harness
from sqlalchemy.sql.expression import ClauseElement, FunctionElement
from tests.synth.scenarios import catalog

from metricprobe.extract import canonical, dual
from metricprobe.status import Check

COVERAGE: dict[str, str] = {
    # ---- status channels (one per Check member)
    "check:probe": "test_cell_cap_aborts_on_both_dialects",
    "check:volume": "test_volume_assessment_matches_between_dialects",
    "check:completion": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "check:freshness": "test_volume_pathology_scenarios_match",
    "check:uniqueness": "test_volume_assessment_matches_between_dialects",
    "check:parity": "test_parity_matches_between_dialects",
    "check:dual_lag": "test_step5_metrics_match_between_dialects",
    "check:batch": "test_step5_metrics_match_between_dialects",
    "check:reconciliation": "test_canonical_pass_matches_between_duckdb_and_mssql",
    # ---- named pathology scenarios (the twin catalog)
    "scenario:volume_spike": "test_volume_pathology_scenarios_match",
    "scenario:volume_drop": "test_volume_pathology_scenarios_match",
    "scenario:missing_month": "test_volume_pathology_scenarios_match",
    "scenario:duplicate_keys": "test_volume_assessment_matches_between_dialects",
    "scenario:straggler_batch": "test_straggler_batch_scenario_matches",
    "scenario:raw_vs_corrected": "test_step5_metrics_match_between_dialects",
    "scenario:sustained_collapse": "test_volume_assessment_matches_between_dialects",
    # ---- dialect-sensitive compiled constructs (@compiles classes)
    "construct:DateDiffDay": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "construct:MonthFloor": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "construct:TimeBucket": "test_censoring_overflow_and_hour_bucket_match",
    "construct:KeyHash": "test_empty_table_and_null_keys_match",
    "construct:GroupingSetsClause": "test_canonical_pass_matches_between_duckdb_and_mssql",
    # ---- dialect-sensitive behaviors beyond single constructs
    "behavior:via_join_composite_unmatched":
        "test_via_join_and_alt_grouping_match_between_dialects",
    "behavior:dual_pass": "test_step5_metrics_match_between_dialects",
    "behavior:dual_via": "test_dual_via_matches_between_dialects",
    "behavior:censoring_over_cap": "test_censoring_overflow_and_hour_bucket_match",
    "behavior:negative_lag_clip": "test_volume_pathology_scenarios_match",
    "behavior:negative_lag_exclude": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "behavior:empty_table": "test_empty_table_and_null_keys_match",
    "behavior:still_filling": "test_volume_assessment_matches_between_dialects",
    "behavior:stale_feed_red": "test_volume_pathology_scenarios_match",
    "behavior:parity_one_sided": "test_parity_matches_between_dialects",
    "behavior:parity_mismatch_and_prereqs": "test_parity_mismatch_and_prereq_verdicts_match",
    "behavior:read_uncommitted_isolation":
        "test_scan_budget_enforced_through_the_production_runner",
    # ---- probe refusals (fail-closed paths)
    "refusal:join_not_unique": "test_via_non_unique_lookup_aborts_on_both_dialects",
    "refusal:result_cell_cap": "test_cell_cap_aborts_on_both_dialects",
    "refusal:scan_budget": "test_scan_budget_enforced_through_the_production_runner",
    # ---- storage + harness
    "store:mssql": "test_mssql_store_shares_the_run_contract",
    "harness:roundtrip": "test_same_aggregate_from_duckdb_and_mssql",
}


def _compiled_constructs(module) -> set[str]:
    return {
        name
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and issubclass(obj, (FunctionElement, ClauseElement))
        and obj.__module__ == module.__name__
    }


def required_items() -> set[str]:
    items = {f"check:{check.value}" for check in Check}
    items |= {f"scenario:{name}" for name in catalog()}
    for module in (canonical, dual):
        items |= {f"construct:{name}" for name in _compiled_constructs(module)}
    return items


def test_every_derived_item_is_covered():
    # new Check members, scenarios or compiled constructs REQUIRE a mapping
    missing = sorted(required_items() - set(COVERAGE))
    assert not missing, (
        "equivalence coverage gaps — add an equivalence case and map it here: "
        f"{missing}"
    )


def test_every_mapped_test_exists():
    available = set()
    for module in (eqv_core, eqv_gaps, eqv_harness):
        available |= {name for name in dir(module) if name.startswith("test_")}
    dangling = {item: name for item, name in COVERAGE.items() if name not in available}
    assert not dangling, f"coverage map points at nonexistent tests: {dangling}"


def test_coverage_map_is_all_equivalence_marked():
    # everything referenced must live in tests/equivalence (auto-marked there)
    for module in (eqv_core, eqv_gaps, eqv_harness):
        assert module.__name__.startswith("tests.equivalence.")
