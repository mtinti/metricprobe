"""Step 7: the EXPLICIT equivalence coverage matrix, enforced.

DuckDB is the fast vehicle; the SQL Server container job is the truth-teller.
The required list is DERIVED from the code so growth forces coverage:

  * metric:<module>      — every metricprobe.metrics module exposing an
                           assess_* entry point (a NEW METRIC lands here
                           automatically and demands an equivalence case)
  * scenario:<name>      — every named twin pair in the pathology catalog;
                           the mapped test must exercise BOTH directions
                           (its source must reference the scenario and the
                           word "healthy")
  * construct:<class>    — every @compiles construct in the extract modules
  * dialect_module:<mod> — every metricprobe module whose source branches on
                           "mssql" (new dialect-specific code demands a case)

Mapped test names must exist in tests/equivalence, and for metric/scenario
items the mapped test's SOURCE must actually reference the subject (the
assess_* entry point / the scenario name), so an unrelated mapping cannot
satisfy the gate. Explicit `behavior:`/`refusal:`/`store:` entries pin
cross-construct behaviors and the fail-closed refusal paths."""

import importlib
import inspect
import pkgutil

import tests.equivalence.test_canonical_equivalence as eqv_core
import tests.equivalence.test_coverage_gaps as eqv_gaps
import tests.equivalence.test_harness_roundtrip as eqv_harness
from sqlalchemy.sql.expression import ClauseElement, FunctionElement
from tests.synth.scenarios import catalog

import metricprobe.metrics as metrics_package
from metricprobe import cli, config, store
from metricprobe.extract import canonical, dual
from metricprobe.status import Check

EQUIVALENCE_MODULES = (eqv_core, eqv_gaps, eqv_harness)

COVERAGE: dict[str, str] = {
    # ---- metrics (derived from assess_* entry points)
    "metric:volume": "test_volume_assessment_matches_between_dialects",
    "metric:completion": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "metric:freshness": "test_volume_pathology_scenarios_match",
    "metric:batch": "test_step5_metrics_match_between_dialects",
    "metric:dual_lag": "test_step5_metrics_match_between_dialects",
    "metric:parity": "test_parity_mismatch_and_prereq_verdicts_match",
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
    # ---- named pathology twins (both directions enforced via source check)
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
    # ---- modules with dialect branches (source contains "mssql")
    "dialect_module:canonical": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_module:dual": "test_step5_metrics_match_between_dialects",
    "dialect_module:store": "test_mssql_store_shares_the_run_contract",
    "dialect_module:cli": "test_scan_budget_enforced_through_the_production_runner",
    "dialect_module:config": "test_mssql_store_shares_the_run_contract",
    # ---- behaviors beyond single constructs
    "behavior:via_join_composite_unmatched":
        "test_via_join_and_alt_grouping_match_between_dialects",
    "behavior:dual_via": "test_dual_via_matches_between_dialects",
    "behavior:censoring_over_cap": "test_censoring_overflow_and_hour_bucket_match",
    "behavior:negative_lag_clip": "test_volume_pathology_scenarios_match",
    "behavior:negative_lag_exclude": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "behavior:empty_table": "test_empty_table_and_null_keys_match",
    "behavior:null_key_sentinel": "test_empty_table_and_null_keys_match",
    "behavior:still_filling": "test_volume_assessment_matches_between_dialects",
    "behavior:stale_feed_red": "test_volume_pathology_scenarios_match",
    "behavior:parity_one_sided": "test_parity_matches_between_dialects",
    "behavior:read_uncommitted_isolation":
        "test_scan_budget_enforced_through_the_production_runner",
    # ---- probe refusals (fail-closed paths), produced through the runners
    "refusal:join_not_unique": "test_via_non_unique_lookup_aborts_on_both_dialects",
    "refusal:result_cell_cap": "test_cell_cap_aborts_on_both_dialects",
    "refusal:scan_budget": "test_scan_budget_refusals_through_the_production_runner",
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


def _metric_modules() -> dict[str, set[str]]:
    """metric module name -> its assess_* entry points."""
    modules = {}
    for info in pkgutil.iter_modules(metrics_package.__path__):
        module = importlib.import_module(f"metricprobe.metrics.{info.name}")
        entry_points = {name for name in vars(module) if name.startswith("assess_")}
        if entry_points:
            modules[info.name] = entry_points
    return modules


def _dialect_modules() -> set[str]:
    names = set()
    for module in (canonical, dual, store, cli, config):
        if '"mssql"' in inspect.getsource(module):
            names.add(module.__name__.rsplit(".", 1)[-1])
    return names


def required_items() -> set[str]:
    items = {f"metric:{name}" for name in _metric_modules()}
    items |= {f"check:{check.value}" for check in Check}
    items |= {f"scenario:{name}" for name in catalog()}
    for module in (canonical, dual):
        items |= {f"construct:{name}" for name in _compiled_constructs(module)}
    items |= {f"dialect_module:{name}" for name in _dialect_modules()}
    return items


def _test_sources() -> dict[str, str]:
    """Test name -> its source PLUS the source of same-module helpers it
    references (a test may exercise a metric through a shared helper)."""
    sources = {}
    for module in EQUIVALENCE_MODULES:
        helpers = {
            name: inspect.getsource(obj)
            for name, obj in vars(module).items()
            if inspect.isfunction(obj)
            and not name.startswith("test_")
            and obj.__module__ == module.__name__
        }
        for name, obj in vars(module).items():
            if name.startswith("test_") and inspect.isfunction(obj):
                source = inspect.getsource(obj)
                expanded = source + "".join(
                    body for helper, body in helpers.items() if helper in source
                )
                sources[name] = expanded
    return sources


def test_every_derived_item_is_covered():
    # new metrics, Check members, scenarios, compiled constructs or dialect
    # branches REQUIRE a mapped equivalence case
    missing = sorted(required_items() - set(COVERAGE))
    assert not missing, (
        "equivalence coverage gaps — add an equivalence case and map it here: "
        f"{missing}"
    )


def test_every_mapped_test_exists():
    available = set(_test_sources())
    dangling = {item: name for item, name in COVERAGE.items() if name not in available}
    assert not dangling, f"coverage map points at nonexistent tests: {dangling}"


def test_mappings_are_meaningful_not_arbitrary():
    """A mapping must point at a test that actually exercises its subject."""
    sources = _test_sources()
    metrics = _metric_modules()
    problems = []
    for item, test_name in COVERAGE.items():
        source = sources.get(test_name, "")
        kind, _, subject = item.partition(":")
        if kind == "metric":
            if not any(entry in source for entry in metrics[subject]):
                problems.append(f"{item} -> {test_name} never calls assess_{subject}")
        elif kind == "scenario":
            # the injector/scenario name must appear, and BOTH twin directions
            # must be exercised (the healthy twin stays silent)
            if subject not in source:
                problems.append(f"{item} -> {test_name} never references {subject}")
            elif "healthy" not in source:
                problems.append(f"{item} -> {test_name} does not run the healthy twin")
    assert not problems, "; ".join(problems)
