"""Step 7: the EXPLICIT equivalence coverage matrix, enforced.

DuckDB is the fast vehicle; the SQL Server container job is the truth-teller.
The required list is DERIVED from the code so growth forces coverage:

  * metric:<module>.<fn> — every assess_* ENTRY POINT in every
                           metricprobe.metrics module (a NEW METRIC lands
                           here automatically and demands an equivalence
                           case — even when added to an EXISTING module)
  * scenario:<name>      — every named twin pair in the pathology catalog;
                           the mapped test must exercise BOTH directions
                           (its source must reference the scenario and the
                           word "healthy")
  * construct:<class>    — every @compiles construct in the extract modules
  * dialect_branch:<module>.<function> — every FUNCTION (or method) whose
                           source branches on "mssql" (a new dialect branch
                           inside an existing module demands a case)

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
from metricprobe.extract import canonical, dual
from metricprobe.status import Check

EQUIVALENCE_MODULES = (eqv_core, eqv_gaps, eqv_harness)

COVERAGE: dict[str, str] = {
    # ---- metrics (derived per assess_* ENTRY POINT)
    "metric:volume.assess_volume": "test_volume_assessment_matches_between_dialects",
    "metric:completion.assess_completion":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "metric:freshness.assess_freshness": "test_volume_pathology_scenarios_match",
    "metric:batch.assess_batch": "test_step5_metrics_match_between_dialects",
    "metric:dual_lag.assess_dual_lag": "test_step5_metrics_match_between_dialects",
    "metric:parity.assess_parity": "test_parity_mismatch_and_prereq_verdicts_match",
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
    "scenario:sustained_collapse_short": "test_short_collapse_scenario_matches",
    # ---- dialect-sensitive compiled constructs (@compiles classes)
    "construct:DateDiffDay": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "construct:MonthFloor": "test_canonical_pass_matches_between_duckdb_and_mssql",
    "construct:TimeBucket": "test_censoring_overflow_and_hour_bucket_match",
    "construct:KeyHash": "test_empty_table_and_null_keys_match",
    "construct:GroupingSetsClause": "test_canonical_pass_matches_between_duckdb_and_mssql",
    # ---- dialect code (functions mentioning mssql; ALL methods of
    # classes mentioning mssql — derived by walking the whole package)
    "dialect_branch:extract.canonical._datediff_mssql":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical._month_floor_mssql":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical._time_bucket_mssql":
        "test_censoring_overflow_and_hour_bucket_match",
    "dialect_branch:extract.canonical._key_hash_mssql":
        "test_empty_table_and_null_keys_match",
    "dialect_branch:extract.canonical._key_column_types":
        "test_empty_table_and_null_keys_match",
    "dialect_branch:extract.canonical._table_clause":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical.staging_table_name":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical.staging_sql":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical._dialect_instance":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical.build_aggregation_query":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical.run_canonical":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical.CanonicalResult.rows_for":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical.CanonicalResult.global_row":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:extract.canonical._install_message_capture":
        "test_scan_budget_enforced_through_the_production_runner",
    "dialect_branch:extract.canonical._mssql_target_pages":
        "test_scan_budget_enforced_through_the_production_runner",
    "dialect_branch:extract.canonical._mssql_staging_pages":
        "test_scan_budget_enforced_through_the_production_runner",
    "dialect_branch:extract.canonical.verify_scan_budget":
        "test_scan_budget_refusals_through_the_production_runner",
    "dialect_branch:extract.dual.dual_staging_table_name":
        "test_step5_metrics_match_between_dialects",
    "dialect_branch:extract.dual.dual_staging_sql":
        "test_step5_metrics_match_between_dialects",
    "dialect_branch:extract.dual.run_dual_lag":
        "test_dual_via_matches_between_dialects",
    "dialect_branch:store.open_store": "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.__init__":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.register_run":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.registration":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.staging_claim":
        "test_mssql_store_sweeps_are_isolated_and_types_are_frozen",
    "dialect_branch:store.MssqlStore.begin_run":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.write_table":
        "test_mssql_store_sweeps_are_isolated_and_types_are_frozen",
    "dialect_branch:store.MssqlStore.commit_run":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:extract.canonical._exec_with_messages":
        "test_pyodbc_captures_statistics_io_and_verifies_the_budget",
    "dialect_branch:extract.canonical.AsOfLiteral.literal_processor":
        "test_legacy_datetime_columns_accept_a_microsecond_as_of",
    "dialect_branch:extract.canonical.CanonicalResult.staged_row_count":
        "test_legacy_datetime_columns_accept_a_microsecond_as_of",
    "dialect_branch:store.MssqlStore.record_stage":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore._migrate_v4_to_v5":
        "test_mssql_store_migrates_a_v4_marker_in_place",
    "dialect_branch:store.MssqlStore.prepare_stage":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.table_names":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.abort_run":
        "test_mssql_store_sweeps_are_isolated_and_types_are_frozen",
    "dialect_branch:store.MssqlStore.list_runs":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.read_table":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:store.MssqlStore.prune":
        "test_mssql_store_sweeps_are_isolated_and_types_are_frozen",
    "dialect_branch:store.MssqlStore._data_tables":
        "test_mssql_store_sweeps_are_isolated_and_types_are_frozen",
    "dialect_branch:store.MssqlStore._verify_physical_schema_version":
        "test_mssql_store_sweeps_are_isolated_and_types_are_frozen",
    "dialect_branch:cli._engine_for":
        "test_scan_budget_enforced_through_the_production_runner",
    "dialect_branch:cli._table_exists":
        "test_missing_table_detection_matches_between_dialects",
    "dialect_branch:config.StoreConfig._mssql_needs_url":
        "test_mssql_store_shares_the_run_contract",
    "dialect_branch:config.ProbeConfig._cross_checks":
        "test_canonical_pass_matches_between_duckdb_and_mssql",
    "dialect_branch:discover.scan_columns": "test_discover_matches_between_dialects",
    "dialect_branch:discover.ColumnInfo.is_datetime":
        "test_discover_matches_between_dialects",
    "dialect_branch:discover.ColumnInfo.resolution":
        "test_discover_matches_between_dialects",
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
    "refusal:dual_join_not_unique": "test_via_non_unique_lookup_aborts_on_both_dialects",
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


def _metric_entry_points() -> dict[str, set[str]]:
    """metric module name -> its assess_* entry points. Every entry point is
    its own required item, so a new metric added to an EXISTING module still
    demands an equivalence case."""
    modules = {}
    for info in pkgutil.iter_modules(metrics_package.__path__):
        module = importlib.import_module(f"metricprobe.metrics.{info.name}")
        entry_points = {name for name in vars(module) if name.startswith("assess_")}
        if entry_points:
            modules[info.name] = entry_points
    return modules


def _all_package_modules():
    """EVERY module in the metricprobe package (walked, never hard-coded):
    new dialect-touching code cannot land outside the gate's sight."""
    import metricprobe

    return [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(metricprobe.__path__, prefix="metricprobe.")
        # __main__ executes the CLI on import
        if not info.name.endswith("__main__")
    ]


def _dialect_branch_functions() -> set[str]:
    """<module>.<qualname> of every function whose source mentions mssql
    (case-insensitive), plus EVERY method of a class whose source mentions
    mssql (e.g. MssqlStore: its methods run dialect-specific SQL without
    spelling the word) — a new dialect branch or store method anywhere
    demands an equivalence case."""

    def _source(obj) -> str:
        try:
            return inspect.getsource(obj)
        except (OSError, TypeError):  # generated code (pydantic internals etc.)
            return ""

    items = set()
    for module in _all_package_modules():
        short = module.__name__.removeprefix("metricprobe.")
        for name, obj in vars(module).items():
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            if inspect.isfunction(obj):
                if "mssql" in _source(obj).lower():
                    items.add(f"{short}.{name}")
            elif inspect.isclass(obj):
                class_is_dialect = "mssql" in _source(obj).lower()
                for method_name, member in vars(obj).items():
                    if isinstance(member, property) and member.fget:
                        method = member.fget
                    elif inspect.isfunction(member):
                        method = member
                    else:
                        continue
                    if getattr(method, "__module__", None) != module.__name__:
                        continue  # pydantic/base-class machinery
                    src = _source(method)  # "" for generated code (dataclass)
                    if src and (class_is_dialect or "mssql" in src.lower()):
                        items.add(f"{short}.{name}.{method_name}")
    return items


def required_items() -> set[str]:
    items = {
        f"metric:{module}.{entry}"
        for module, entries in _metric_entry_points().items()
        for entry in entries
    }
    items |= {f"check:{check.value}" for check in Check}
    items |= {f"scenario:{name}" for name in catalog()}
    for module in (canonical, dual):
        items |= {f"construct:{name}" for name in _compiled_constructs(module)}
    items |= {f"dialect_branch:{name}" for name in _dialect_branch_functions()}
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


# every refusal mapping must point at a test whose source shows the refusal
# actually happening: the reason-code token AND (for pass-specific refusals)
# the runner that must raise it
REFUSAL_REQUIRED_TOKENS = {
    "join_not_unique": ("JOIN_NOT_UNIQUE", "run_canonical"),
    "dual_join_not_unique": ("JOIN_NOT_UNIQUE", "run_dual_lag"),
    "result_cell_cap": ("RESULT_CELL_CAP_EXCEEDED",),
    "scan_budget": ("SCAN_BUDGET",),
}


def test_mappings_are_meaningful_not_arbitrary():
    """A mapping must point at a test that actually exercises its subject."""
    sources = _test_sources()
    problems = []
    for item, test_name in COVERAGE.items():
        source = sources.get(test_name, "")
        kind, _, subject = item.partition(":")
        if kind == "metric":
            entry = subject.split(".", 1)[1]
            if entry not in source:
                problems.append(f"{item} -> {test_name} never calls {entry}")
        elif kind == "refusal":
            for token in REFUSAL_REQUIRED_TOKENS[subject]:
                if token not in source:
                    problems.append(
                        f"{item} -> {test_name} never demonstrates {token}"
                    )
        elif kind == "scenario":
            # the injector/scenario name must appear, and BOTH twin directions
            # must be exercised (the healthy twin stays silent)
            if subject not in source:
                problems.append(f"{item} -> {test_name} never references {subject}")
            elif "healthy" not in source:
                problems.append(f"{item} -> {test_name} does not run the healthy twin")
    assert not problems, "; ".join(problems)
