"""CLI contract (Step 6): exit codes are relative to the CONFIGURED stages —
here the only configured terminal state is ANALYSIS_COMMITTED. 0/2 = analysis
committed without/with RED (outputs committed FIRST); 1 = stage failure with
nothing partial. Verified both in-process and through `python -m metricprobe`
(the exact invocation an Actions-style runner uses)."""

import subprocess
import sys

import duckdb
import pytest
import yaml
from tests.synth import generator as g

from metricprobe.cli import main
from metricprobe.config import CONFIG_SCHEMA_VERSION
from metricprobe.store import STAMP_COLUMNS, ParquetStore

# just after the last synthetic load: freshness stays GREEN while the
# first 18 months are mature under the 365d horizon
AS_OF = "2025-07-02"

HEALTHY = g.TableSpec(
    name="events", start_month="2023-01", n_months=30, rows_per_month=1000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=101,
)


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    """One duckdb database file (catalog name = file stem 'demo') holding a
    healthy table, an unhealthy (dropped-month outlier) table, and a pair."""
    root = tmp_path_factory.mktemp("db")
    path = root / "demo.duckdb"
    con = duckdb.connect(str(path))
    healthy_df = g.generate(HEALTHY)
    g.load_into_duckdb(healthy_df, con, "events")
    import dataclasses

    g.load_into_duckdb(
        g.generate(dataclasses.replace(HEALTHY, dual_offset_days=2.0)), con, "events_dual"
    )
    # the dropped month must be MATURE at AS_OF: index 10 = 2023-11
    g.load_into_duckdb(g.generate(g.volume_drop(HEALTHY, 10, factor=0.1)), con, "events_bad")
    g.load_into_duckdb(healthy_df, con, "events_copy")
    con.close()
    return path


def table_entry(table: str, probe: str, **overrides) -> dict:
    return {
        "probe_name": probe,
        "database": "demo",
        "schema": "main",
        "table": table,
        "event_time": "event_time",
        "load_time": "load_time",
        "resolution": {"event_time": "datetime", "load_time": "datetime"},
    } | overrides


def write_config(tmp_path, demo_db, tables: list[dict], **top) -> str:
    config = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "connection_url": f"duckdb:///{demo_db}",
        "store": {"path": str(tmp_path / "store")},
        "tables": tables,
    } | top
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def run_cli(*argv) -> int:
    return main(["run", *argv])


def test_exit_0_healthy_run_commits_analysis(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", config, "--as-of", AS_OF) == 0
    store = ParquetStore(tmp_path / "store")
    runs = store.list_runs()
    assert len(runs) == 1
    manifest = runs[0]
    # the manifest carries the full provenance + per-probe wall clocks
    assert manifest["as_of"].startswith(AS_OF)
    assert manifest["config_digest"] and manifest["git_sha"]
    (probe,) = manifest["probes"]
    assert probe["probe"] == "orders_main"
    assert probe["duration_seconds"] >= 0
    assert probe["extraction_started"] <= probe["extraction_finished"]
    assert manifest["stages"]["analysis"]["completed_at"]
    # every stored row is stamped
    statuses = store.read_table(manifest["run_id"], "statuses")
    for column in STAMP_COLUMNS:
        assert column in statuses.columns
    assert set(store.table_names(manifest["run_id"])) >= {
        "statuses", "probe_runs", "month_volumes", "completion_summary",
        "completion_percentiles", "month_lag_cells", "epoch_cells", "freshness",
        "population_buckets", "volume_summary",
    }
    # the manifest names the exact contract versions that produced the run,
    # and every status row carries the status model version
    from metricprobe.cli import COMPONENT_VERSIONS

    assert manifest["component_versions"] == COMPONENT_VERSIONS
    assert set(statuses["status_schema_version"]) == {COMPONENT_VERSIONS["status"]}
    # the declared resolution labels the completion grain in the snapshot
    percentiles = store.read_table(manifest["run_id"], "completion_percentiles")
    assert set(percentiles["lag_resolution"]) == {"datetime"}
    # the reconciliation buckets are reproducible from the snapshot alone
    buckets = store.read_table(manifest["run_id"], "population_buckets").iloc[0]
    assert int(buckets["row_count"]) > 0
    assert int(buckets["row_count"]) == (
        int(buckets["n_curve_eligible"])
        + int(buckets["n_null_event_time"])
        + int(buckets["n_null_load_time_only"])
        + int(buckets["n_negative_lag_excluded"])
        + int(buckets["n_join_unmatched"])
        + int(buckets["n_other_exclusions"])
    )


def test_exit_2_red_run_still_commits_first(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events_bad", "orders_bad")])
    assert run_cli("--config", config, "--as-of", AS_OF) == 2
    runs = ParquetStore(tmp_path / "store").list_runs()
    assert len(runs) == 1  # committed BEFORE exiting red
    assert any(s["severity"] == "red" for s in runs[0]["statuses"])


def test_missing_optional_table_is_skipped_exit_0(tmp_path, demo_db):
    config = write_config(
        tmp_path,
        demo_db,
        [
            table_entry("events", "orders_main"),
            table_entry("absent", "ghost_probe", optional=True),
        ],
    )
    assert run_cli("--config", config, "--as-of", AS_OF) == 0
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    ghost = [s for s in manifest["statuses"] if s["probe"] == "ghost_probe"]
    assert ghost == [
        {"probe": "ghost_probe", "check": "probe", "severity": "skipped",
         "reason": "optional_table_absent"}
    ]


def test_missing_required_table_is_red_exit_2(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("absent", "ghost_probe")])
    assert run_cli("--config", config, "--as-of", AS_OF) == 2
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    assert manifest["statuses"][0]["reason"] == "missing_table"


def test_parity_indeterminate_reduces_to_exit_0(tmp_path, demo_db):
    config = write_config(
        tmp_path,
        demo_db,
        [
            table_entry(
                "events", "orders_main",
                parity_with="orders_copy", read_uncommitted=True, key_cols=["row_id"],
            ),
            table_entry("events_copy", "orders_copy", key_cols=["row_id"]),
        ],
    )
    assert run_cli("--config", config, "--as-of", AS_OF) == 0
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    parity = [s for s in manifest["statuses"] if s["check"] == "parity"]
    assert parity and parity[0]["severity"] == "indeterminate"
    assert parity[0]["reason"] == "parity_prereq_read_uncommitted"


def test_dry_run_touches_nothing(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", config, "--as-of", AS_OF, "--dry-run") == 0
    assert not (tmp_path / "store" / "runs").exists() or not any(
        (tmp_path / "store" / "runs").iterdir()
    )


def test_injected_failure_leaves_nothing_partial(tmp_path, demo_db, monkeypatch):
    import metricprobe.cli as cli

    def explode(*args, **kwargs):
        raise RuntimeError("injected analysis failure")

    monkeypatch.setattr(cli, "assess_volume", explode)
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", config, "--as-of", AS_OF) == 1
    store_root = tmp_path / "store"
    assert ParquetStore(store_root).list_runs() == []
    assert list((store_root / "staging").iterdir()) == []  # staging cleaned


def test_config_error_is_exit_1(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 99\n")
    assert run_cli("--config", str(bad)) == 1


def test_resume_is_idempotent_and_digest_guarded(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", "r-fixed") == 0
    # idempotent retry: analysis already committed, nothing redone, same exit
    assert (
        run_cli("--config", config, "--as-of", AS_OF,
                "--resume-from", "analysis", "--run-id", "r-fixed") == 0
    )
    assert len(ParquetStore(tmp_path / "store").list_runs()) == 1
    # a changed config means a different digest: refuse the resume
    changed = write_config(
        tmp_path, demo_db,
        [table_entry("events", "orders_main", analysis={"lag_cap_days": 200,
                                                        "training_cutoff_days": 200})],
    )
    assert (
        run_cli("--config", changed, "--as-of", AS_OF,
                "--resume-from", "analysis", "--run-id", "r-fixed") == 1
    )
    # resume without a run id is refused
    assert run_cli("--config", config, "--resume-from", "analysis") == 1
    # an unknown run id has neither a committed manifest nor a registration
    # to verify the digest against: refused
    assert (
        run_cli("--config", config, "--as-of", AS_OF,
                "--resume-from", "analysis", "--run-id", "never-ran") == 1
    )


def test_failed_analysis_is_resumable_under_the_same_run_id(tmp_path, demo_db, monkeypatch):
    import metricprobe.cli as cli

    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])

    def explode(*args, **kwargs):
        raise RuntimeError("injected analysis failure")

    monkeypatch.setattr(cli, "assess_volume", explode)
    assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", "r-retry") == 1
    store = ParquetStore(tmp_path / "store")
    assert store.list_runs() == []  # the failed stage left nothing partial...
    registration = store.registration("r-retry")
    assert registration is not None  # ...but the durable registration survives
    # a changed config cannot resume the failed run (digest guard applies to
    # registered runs exactly as to committed ones)
    changed_dir = tmp_path / "changed"
    changed_dir.mkdir()
    changed = write_config(
        changed_dir, demo_db,
        [table_entry("events", "orders_main", analysis={"lag_cap_days": 200,
                                                        "training_cutoff_days": 200})],
        store={"path": str(tmp_path / "store")},  # SAME store, different digest
    )
    assert (
        run_cli("--config", changed, "--as-of", AS_OF,
                "--resume-from", "analysis", "--run-id", "r-retry") == 1
    )
    monkeypatch.undo()
    # with the failure gone, the retry re-runs the stage under the SAME run_id
    assert (
        run_cli("--config", config, "--as-of", AS_OF,
                "--resume-from", "analysis", "--run-id", "r-retry") == 0
    )
    (manifest,) = store.list_runs()
    assert manifest["run_id"] == "r-retry"
    # the registration is consumed by the commit; a second resume is the
    # idempotent already-committed no-op
    assert store.registration("r-retry") is None
    assert (
        run_cli("--config", config, "--as-of", AS_OF,
                "--resume-from", "analysis", "--run-id", "r-retry") == 0
    )
    assert len(store.list_runs()) == 1


def test_probe_runs_schema_is_stable_across_outcomes(tmp_path, demo_db):
    """A skipped/missing/aborted probe writes the SAME probe_runs columns as a
    healthy one — the first frame a store sees must never fix a narrower
    physical schema than later appends need."""
    from metricprobe.cli import READ_COLUMNS

    missing = write_config(
        tmp_path, demo_db, [table_entry("no_such_table", "ghost_main")]
    )
    assert run_cli("--config", missing, "--as-of", AS_OF, "--run-id", "r-ghost") == 2
    store = ParquetStore(tmp_path / "store")
    ghost_runs = store.read_table("r-ghost", "probe_runs")
    healthy_dir = tmp_path / "h"
    healthy_dir.mkdir()
    healthy = write_config(healthy_dir, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", healthy, "--as-of", AS_OF, "--run-id", "r-live") == 0
    live_runs = ParquetStore(healthy_dir / "store").read_table("r-live", "probe_runs")
    # identical columns in identical order, whatever the probe outcome; the
    # read columns are None off-mssql (STATISTICS IO does not exist) and for
    # skipped/aborted probes — present either way
    assert list(ghost_runs.columns) == list(live_runs.columns)
    for column in READ_COLUMNS:
        assert column in ghost_runs.columns
        assert ghost_runs[column].isna().all()


def test_retention_prunes_after_commit(tmp_path, demo_db):
    config = write_config(
        tmp_path, demo_db, [table_entry("events", "orders_main")],
        store={"path": str(tmp_path / "store"), "retention_runs": 2},
    )
    for suffix in ("a", "b", "c"):
        assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", f"r-{suffix}") == 0
    assert len(ParquetStore(tmp_path / "store").list_runs()) == 2


def test_unimplemented_commands_say_so():
    for command in ("report", "publish", "serve"):
        assert main([command]) == 1


def test_exit_codes_through_a_real_process(tmp_path, demo_db):
    """The Actions-style contract: ONE invocation of `python -m metricprobe`,
    exit code read by the workflow. (Re-verified on Gitea in the private
    bootstrap.)"""
    healthy = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    red_dir = tmp_path / "red"
    red_dir.mkdir()
    red = write_config(red_dir, demo_db, [table_entry("events_bad", "orders_bad")])

    def invoke(*argv):
        return subprocess.run(
            [sys.executable, "-m", "metricprobe", "run", *argv],
            capture_output=True,
            text=True,
        )

    ok = invoke("--config", healthy, "--as-of", AS_OF)
    assert ok.returncode == 0, ok.stderr
    bad = invoke("--config", red, "--as-of", AS_OF)
    assert bad.returncode == 2, bad.stderr
    broken = invoke("--config", str(tmp_path / "missing.yaml"))
    assert broken.returncode == 1
    assert "metricprobe:" in broken.stderr


def test_malformed_invocations_never_return_the_data_health_code():
    # argparse would exit 2 on usage errors, but 2 means "committed with RED"
    assert main(["run"]) == 1  # missing --config
    assert main(["run", "--bogus-flag"]) == 1
    assert main(["bogus-command"]) == 1
    assert main(["--version"]) == 0


def test_year_window_bounds_reported_completion_results(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", config, "--as-of", AS_OF, "--year", "2023") == 0
    store = ParquetStore(tmp_path / "store")
    (manifest,) = store.list_runs()
    months = store.read_table(manifest["run_id"], "completion_percentiles")["month"]
    assert months.str.startswith("2023").all()  # the probe window bounds results
    # volume history stays FULL HISTORY by contract, stamped with the window
    volumes = store.read_table(manifest["run_id"], "month_volumes")
    assert volumes["month"].str.startswith("2025").any()
    assert manifest["window_start"].startswith("2023-01-01")


def test_frozen_clock_makes_run_metadata_deterministic(tmp_path, demo_db, monkeypatch):
    monkeypatch.setenv("METRICPROBE_RUN_AT", "2025-07-02T06:00:00")
    monkeypatch.setenv("METRICPROBE_GIT_SHA", "cafe1234")
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", "frozen-run") == 0
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    # the frozen clock is normalized to timezone-AWARE UTC (a naive local
    # run_at would be misread by schedule math and monotonic comparisons)
    assert manifest["run_at"] == "2025-07-02T06:00:00+00:00"
    assert manifest["git_sha"] == "cafe1234"
    (probe,) = manifest["probes"]
    assert probe["duration_seconds"] == 0.0  # every timestamp is the frozen clock
    assert probe["extraction_started"] == probe["extraction_finished"]
    assert manifest["stages"]["analysis"]["completed_at"] == "2025-07-02T06:00:00+00:00"


def test_probe_abort_commits_no_partial_frames(tmp_path, demo_db, monkeypatch):
    import metricprobe.cli as cli
    from metricprobe.extract.canonical import ProbeAborted
    from metricprobe.status import ReasonCode

    def abort_dual(*args, **kwargs):
        raise ProbeAborted(ReasonCode.SCAN_BUDGET_EXCEEDED, "injected dual abort")

    monkeypatch.setattr(cli, "run_dual_lag", abort_dual)
    config = write_config(
        tmp_path,
        demo_db,
        [
            table_entry("events", "orders_main"),
            table_entry(
                "events_dual", "dual_probe", source_insert_time="source_insert_time",
                resolution={"event_time": "datetime", "load_time": "datetime",
                            "source_insert_time": "datetime"},
            ),
        ],
    )
    assert run_cli("--config", config, "--as-of", AS_OF) == 2  # typed RED, committed
    store = ParquetStore(tmp_path / "store")
    (manifest,) = store.list_runs()
    aborted = [s for s in manifest["statuses"] if s["probe"] == "dual_probe"]
    assert aborted == [
        {"probe": "dual_probe", "check": "probe", "severity": "red",
         "reason": "scan_budget_exceeded"}
    ]
    # NO partial metric output from the aborted probe reached the snapshot
    for table in ("completion_summary", "month_volumes", "month_lag_cells"):
        frame = store.read_table(manifest["run_id"], table)
        assert "dual_probe" not in set(frame["probe"])


def test_failure_at_the_commit_boundary_leaves_nothing_partial(tmp_path, demo_db, monkeypatch):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])

    def boom(*args, **kwargs):
        raise OSError("injected commit failure")

    monkeypatch.setattr(ParquetStore, "commit_run", boom)
    assert run_cli("--config", config, "--as-of", AS_OF) == 1
    store_root = tmp_path / "store"
    monkeypatch.undo()
    assert ParquetStore(store_root).list_runs() == []
    assert list((store_root / "staging").iterdir()) == []

    monkeypatch.setattr(ParquetStore, "write_table", boom)
    assert run_cli("--config", config, "--as-of", AS_OF) == 1
    monkeypatch.undo()
    assert ParquetStore(store_root).list_runs() == []
    assert list((store_root / "staging").iterdir()) == []


def test_multi_config_campaign_composes_and_validates(tmp_path, demo_db):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir(), right_dir.mkdir()
    shared_store = {"path": str(tmp_path / "store")}
    left = write_config(
        left_dir, demo_db,
        [table_entry("events", "orders_main", parity_with="orders_copy",
                     key_cols=["row_id"])],
        store=shared_store,
    )
    right = write_config(
        right_dir, demo_db,
        [table_entry("events_copy", "orders_copy", key_cols=["row_id"])],
        store=shared_store,
    )
    # cross-file parity composes and runs green
    assert run_cli("--config", left, "--config", right, "--as-of", AS_OF) == 0
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    parity = [s for s in manifest["statuses"] if s["check"] == "parity"]
    assert parity and parity[0]["severity"] == "green"
    # campaign-wide duplicate names are refused before anything runs
    dup = write_config(right_dir, demo_db, [table_entry("events", "orders_main")])
    assert run_cli("--config", left, "--config", dup, "--as-of", AS_OF) == 1
    # invalid run ids are refused loudly
    assert run_cli("--config", left, "--as-of", AS_OF, "--run-id", "../evil") == 1


def test_resume_preserves_the_failed_runs_identity(tmp_path, demo_db, monkeypatch):
    """A resumed run keeps its registered as_of/window (the run id keeps
    meaning ONE analysis); conflicting explicit flags refuse; plain --run-id
    reuse of a registered id refuses without --resume-from."""
    import metricprobe.cli as cli

    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])

    def explode(*args, **kwargs):
        raise RuntimeError("injected analysis failure")

    monkeypatch.setattr(cli, "assess_volume", explode)
    assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", "r-id") == 1
    monkeypatch.undo()
    # plain reuse of the registered id (digest guard bypass) is refused
    assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", "r-id") == 1
    # resuming under a DIFFERENT as_of or window is a different analysis
    assert (
        run_cli("--config", config, "--as-of", "2025-01-01",
                "--resume-from", "analysis", "--run-id", "r-id") == 1
    )
    assert (
        run_cli("--config", config, "--window", "6m",
                "--resume-from", "analysis", "--run-id", "r-id") == 1
    )
    # with no conflicting flags the registration's identity is adopted
    assert run_cli("--config", config, "--resume-from", "analysis", "--run-id", "r-id") == 0
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    assert manifest["as_of"].startswith(AS_OF)


def test_resume_refuses_while_a_staging_claim_exists(tmp_path, demo_db):
    """A present staging claim means another writer may still be working the
    run id: resuming must refuse rather than abort its rows (a crashed writer
    needs explicit cleanup, never a silent takeover)."""
    from metricprobe.store import SNAPSHOT_SCHEMA_VERSION, RunMeta

    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    # first, learn the digest of this config by dry-running a real run
    assert run_cli("--config", config, "--as-of", AS_OF, "--run-id", "r-probe") == 0
    store = ParquetStore(tmp_path / "store")
    digest = store.list_runs()[0]["config_digest"]
    # simulate a writer that crashed (or is still alive) mid-stage: claim held
    meta = RunMeta(
        run_id="r-crashed", run_at="2025-07-02T00:00:00", as_of=f"{AS_OF}T00:00:00",
        git_sha="x", tool_version="0", config_digest=digest,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        window_start="2023-07-02T00:00:00", window_end=f"{AS_OF}T00:00:00",
    )
    store.register_run(meta)
    store.begin_run(meta)  # the claim: staging dir exists, never aborted
    assert store.staging_claim("r-crashed") is not None
    assert (
        run_cli("--config", config, "--resume-from", "analysis", "--run-id", "r-crashed")
        == 1
    )
    # the DOCUMENTED cleanup path is the abort command, not an internal API:
    # it releases the claim so the resume can proceed
    assert main(["abort", "--config", config, "--run-id", "r-crashed"]) == 0
    assert store.staging_claim("r-crashed") is None
    assert (
        run_cli("--config", config, "--resume-from", "analysis", "--run-id", "r-crashed")
        == 0
    )
    # abort never touches a committed run
    assert main(["abort", "--config", config, "--run-id", "r-crashed"]) == 1
    assert len(store.list_runs()) >= 1


def test_window_and_as_of_usage_errors_are_exit_1(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    # zero/negative windows never analyse anything: refused
    assert run_cli("--config", config, "--as-of", AS_OF, "--window", "0m") == 1
    assert run_cli("--config", config, "--as-of", AS_OF, "--window", "-3m") == 1
    # --window and --year are alternatives, never silently ranked
    assert run_cli("--config", config, "--as-of", AS_OF,
                   "--window", "12m", "--year", "2024") == 1
    # a malformed --as-of is a usage error, not a traceback
    assert run_cli("--config", config, "--as-of", "not-a-date") == 1


def test_timezone_qualified_as_of_is_normalized(tmp_path, demo_db):
    config = write_config(tmp_path, demo_db, [table_entry("events", "orders_main")])
    # a +02:00-qualified cutoff analyses (its UTC instant), never exits 1
    assert run_cli("--config", config, "--as-of", "2025-07-02T02:00:00+02:00",
                   "--run-id", "r-tz") == 0
    (manifest,) = ParquetStore(tmp_path / "store").list_runs()
    assert manifest["as_of"] == "2025-07-02T00:00:00"  # the naive UTC instant


def test_manual_runs_can_be_forbidden(tmp_path, demo_db, monkeypatch):
    config = write_config(
        tmp_path, demo_db, [table_entry("events", "orders_main")],
        campaign={"manual_run_behavior": "forbid"},
    )
    monkeypatch.delenv("METRICPROBE_SCHEDULED", raising=False)
    assert run_cli("--config", config, "--as-of", AS_OF) == 1  # manual: refused
    monkeypatch.setenv("METRICPROBE_SCHEDULED", "1")
    assert run_cli("--config", config, "--as-of", AS_OF) == 0  # scheduled: runs
