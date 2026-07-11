"""Step 9 lifecycle: analyse -> ANALYSIS_COMMITTED -> ARTIFACTS_RENDERED ->
PUBLISHED in ONE process, exit codes relative to the configured stages,
per-stage resume, and the monotonic-publication guard — against a real local
git remote (file://), the way the scheduled workflow runs it."""

from __future__ import annotations

import json
import subprocess

import duckdb
import pytest
import yaml
from tests.synth import generator as g

from metricprobe.cli import main
from metricprobe.config import CONFIG_SCHEMA_VERSION
from metricprobe.publish import PUBLISHED_MARKER
from metricprobe.store import ParquetStore

AS_OF = "2025-07-02"

HEALTHY = g.TableSpec(
    name="events", start_month="2023-01", n_months=30, rows_per_month=500,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=606,
)


def _git(cwd, *args) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        cwd=cwd, capture_output=True, text=True, check=True,
    )
    return result.stdout


@pytest.fixture()
def campaign(tmp_path):
    """(config path, store root, bare remote path) with delivery configured."""
    db = tmp_path / "demo.duckdb"
    con = duckdb.connect(str(db))
    g.load_into_duckdb(g.generate(HEALTHY), con, "events")
    con.close()
    bare = tmp_path / "dashboard.git"
    bare.mkdir()
    _git(bare, "init", "--bare", "--initial-branch", "main")
    config_path = tmp_path / "probe.yaml"
    config_path.write_text(yaml.safe_dump({
        "schema_version": CONFIG_SCHEMA_VERSION,
        "connection_url": f"duckdb:///{db}",
        "store": {"path": str(tmp_path / "store")},
        "campaign": {"schedule": "0 6 * * 1"},
        "delivery": {
            "remotes": [{"name": "origin", "url": f"file://{bare}", "ref": "main"}],
            "worktree": str(tmp_path / "worktree"),
        },
        "tables": [{
            "probe_name": "orders_main", "database": "demo", "schema": "main",
            "table": "events", "event_time": "event_time", "load_time": "load_time",
            "resolution": {"event_time": "datetime", "load_time": "datetime"},
        }],
    }))
    return config_path, tmp_path / "store", bare


def _pushed_files(bare) -> dict[str, str]:
    listing = _git(bare, "ls-tree", "-r", "--name-only", "main").splitlines()
    return {
        name: _git(bare, "show", f"main:{name}") for name in listing if "img/" not in name
    } | {name: "" for name in listing if "img/" in name}


def test_full_lifecycle_publishes_in_one_process(campaign):
    config, store_root, bare = campaign
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-pub"]) == 0
    # the dashboard IS the repo front page of the remote
    files = _pushed_files(bare)
    assert "README.md" in files and "report.html" in files
    assert any(name.startswith("img/") and name.endswith(".svg") for name in files)
    assert "**Next update expected by:**" in files["README.md"]
    marker = json.loads(files[PUBLISHED_MARKER])
    assert marker["run_id"] == "r-pub"
    # ALL THREE stages are recorded on the committed manifest
    (manifest,) = ParquetStore(store_root).list_runs()
    assert set(manifest["stages"]) == {"analysis", "render", "publish"}
    assert manifest["stages"]["publish"]["remotes"] == ["origin"]
    # the idempotent retry re-runs render+publish and stays green
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "analysis", "--run-id", "r-pub"]) == 0


def test_publish_failure_is_exit_1_with_earlier_stages_committed(campaign, tmp_path):
    config, store_root, bare = campaign
    # break the remote (same config, same digest): analysis+render succeed,
    # the push cannot
    aside = tmp_path / "aside.git"
    bare.rename(aside)
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-fail"]) == 1
    aside.rename(bare)
    # analysis + render remain committed and honestly recorded; publish absent
    (manifest,) = ParquetStore(store_root).list_runs()
    assert manifest["run_id"] == "r-fail"
    assert "analysis" in manifest["stages"] and "render" in manifest["stages"]
    assert "publish" not in manifest["stages"]
    # PUBLISHED was never claimed: the remote holds nothing
    assert _git(bare, "ls-remote", "--heads", str(bare), "main") == ""
    # fixing the config and resuming the PUBLISH stage alone completes the run
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "publish", "--run-id", "r-fail"]) == 0
    assert "README.md" in _pushed_files(bare)


def test_monotonic_guard_refuses_older_run(campaign, tmp_path):
    config, store_root, bare = campaign
    # publish a run stamped in the FUTURE directly to the remote
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "--initial-branch", "main")
    (seed / PUBLISHED_MARKER).write_text(json.dumps(
        {"run_id": "r-newer", "run_at": "2099-01-01T00:00:00"}
    ))
    (seed / "README.md").write_text("# newer dashboard\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-m", "newer")
    _git(seed, "push", f"file://{bare}", "HEAD:refs/heads/main")
    # the campaign's publish stage must refuse to clobber it
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-older"]) == 1
    files = _pushed_files(bare)
    assert json.loads(files[PUBLISHED_MARKER])["run_id"] == "r-newer"  # untouched
    # analysis and render are still committed and reported
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "render" in manifest["stages"] and "publish" not in manifest["stages"]


def test_resume_publish_without_artifacts_is_refused(campaign, tmp_path):
    config, store_root, bare = campaign
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-a"]) == 0
    # wipe the rendered artifacts; publish-only resume must point at render
    import shutil

    shutil.rmtree(tmp_path / "worktree" / "artifacts")
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "publish", "--run-id", "r-a"]) == 1
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "render", "--run-id", "r-a"]) == 0


def test_stage_names_are_relative_to_configuration(campaign, tmp_path):
    config, store_root, bare = campaign
    # without delivery, render/publish are NOT configured stages
    text = yaml.safe_load(config.read_text())
    del text["delivery"]
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump(text))
    assert main(["run", "--config", str(plain), "--as-of", AS_OF,
                 "--run-id", "r-x"]) == 0
    assert main(["run", "--config", str(plain), "--as-of", AS_OF,
                 "--resume-from", "render", "--run-id", "r-x"]) == 1


def test_standalone_report_and_publish_commands(campaign, tmp_path):
    config, store_root, bare = campaign
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-cmd"]) == 0
    out = tmp_path / "standalone"
    assert main(["report", "--config", str(config), "--out", str(out / "rep")]) == 0
    assert (out / "rep" / "report.html").exists()
    assert main(["publish", "--config", str(config), "--out", str(out / "dash"),
                 "--run-id", "r-cmd"]) == 0
    assert (out / "dash" / "README.md").exists()
    # standalone re-DELIVERY of the same run is idempotent
    assert main(["publish", "--config", str(config), "--out", str(out / "dash"),
                 "--run-id", "r-cmd", "--deliver"]) == 0
