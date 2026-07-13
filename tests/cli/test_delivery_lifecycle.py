"""Step 9 lifecycle: analyse -> ANALYSIS_COMMITTED -> ARTIFACTS_RENDERED ->
PUBLISHED in ONE process, exit codes relative to the configured stages,
per-stage resume, and the monotonic-publication guard — against a real local
git remote (file://), the way the scheduled workflow runs it."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

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


@pytest.fixture(autouse=True)
def stub_static_export():
    """Lifecycle tests exercise the orchestration state machine (stages,
    atomicity, rollback) — not the renderer, which the render smoke tests
    own with a REAL Chrome. Stubbing image export here removes ~40 browser
    launches from this module: files are still written (existence and
    delivery are asserted), only the bytes are fake.

    Uses a PRIVATE MonkeyPatch instance: several tests call their own
    monkeypatch.undo(), which would strip a shared fixture's patches too
    and relaunch real Chrome mid-test from an ungrouped worker."""
    import plotly.io as pio

    fake_svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>'
    fake_png = b"\x89PNG\r\n\x1a\n" + b"0" * 16

    def fake_write_images(fig, file, format=None, **kwargs):
        payload = fake_svg if format == "svg" else fake_png
        for path in file:
            Path(path).write_bytes(payload)

    own_patch = pytest.MonkeyPatch()
    own_patch.setattr(pio, "write_images", fake_write_images)
    own_patch.setattr(pio, "to_image", lambda *args, **kwargs: fake_png)
    yield
    own_patch.undo()


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
    # --deliver re-runs the SAME lifecycle stages (render recorded before
    # publish — never an analysis -> publish manifest) and is idempotent
    assert main(["publish", "--config", str(config), "--run-id", "r-cmd",
                 "--deliver"]) == 0
    (manifest,) = ParquetStore(store_root).list_runs()
    assert set(manifest["stages"]) == {"analysis", "render", "publish"}
    # --out and --deliver together are ambiguous: refused
    assert main(["publish", "--config", str(config), "--run-id", "r-cmd",
                 "--out", str(out / "x"), "--deliver"]) == 1


def test_delivery_is_bound_to_the_requested_run(campaign, tmp_path):
    """Artifacts are per-run and the marker is verified: one run's files can
    never be published under another run's name (which would defeat the
    monotonic guarantee)."""
    from metricprobe.config import load_config
    from metricprobe.publish import deliver

    config, store_root, bare = campaign
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-one"]) == 0
    loaded = load_config(config)
    artifacts_root = tmp_path / "worktree" / "artifacts"
    assert (artifacts_root / "r-one" / PUBLISHED_MARKER).exists()  # per-run dir
    # delivering r-one's artifacts under a different run id is refused
    import pytest as _pytest

    with _pytest.raises(RuntimeError, match="belong to run"):
        deliver(artifacts_root / "r-one", loaded.delivery, "r-two",
                "2099-01-01T00:00:00")


def test_multi_remote_prepare_failure_touches_no_remote(campaign, tmp_path):
    """Two remotes, the second broken: phase-1 preparation fails BEFORE any
    push, so the healthy first remote is untouched (no partial publication)."""
    config, store_root, bare = campaign
    text = yaml.safe_load(config.read_text())
    text["delivery"]["remotes"].append(
        {"name": "mirror", "url": f"file://{tmp_path}/missing.git", "ref": "main"}
    )
    two_remotes = tmp_path / "two.yaml"
    two_remotes.write_text(yaml.safe_dump(text))
    assert main(["run", "--config", str(two_remotes), "--as-of", AS_OF,
                 "--run-id", "r-two-remotes"]) == 1  # publish stage failed
    # the FIRST remote received nothing: prepare-all happens before push-any
    assert _git(bare, "ls-remote", "--heads", str(bare), "main") == ""
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "render" in manifest["stages"] and "publish" not in manifest["stages"]
    # with both remotes reachable (same config, same digest: the missing
    # repo comes into existence at its configured URL), BOTH publish at once
    second = tmp_path / "missing.git"
    second.mkdir()
    _git(second, "init", "--bare", "--initial-branch", "main")
    assert main(["run", "--config", str(two_remotes), "--as-of", AS_OF,
                 "--resume-from", "render", "--run-id", "r-two-remotes"]) == 0
    (manifest,) = ParquetStore(store_root).list_runs()
    assert manifest["stages"]["publish"]["remotes"] == ["origin", "mirror"]
    for remote in (bare, second):
        assert "README.md" in _pushed_files(remote)


def test_corrupt_snapshot_fails_render_not_publishes_incomplete(campaign, tmp_path):
    """A present-but-unreadable snapshot table must FAIL the render stage —
    never quietly publish a dashboard missing that data."""
    config, store_root, bare = campaign
    # analysis only (strip delivery so the run commits without rendering)
    text = yaml.safe_load(config.read_text())
    delivery = text.pop("delivery")
    plain = tmp_path / "plain.yaml"
    plain.write_text(yaml.safe_dump(text))
    assert main(["run", "--config", str(plain), "--as-of", AS_OF,
                 "--run-id", "r-corrupt"]) == 0
    # corrupt a PRESENT table
    statuses = store_root / "runs" / "r-corrupt" / "statuses.parquet"
    statuses.write_bytes(b"not parquet at all")
    text["delivery"] = delivery
    withd = tmp_path / "withd.yaml"
    withd.write_text(yaml.safe_dump(text))
    assert main(["run", "--config", str(withd), "--as-of", AS_OF,
                 "--resume-from", "render", "--run-id", "r-corrupt"]) == 1
    assert _git(bare, "ls-remote", "--heads", str(bare), "main") == ""


def test_multi_config_campaign_renders_every_probe(campaign, tmp_path):
    """A repeatable-config campaign renders ALL configs' probes: the second
    file's probe must appear in the published README with its own settings."""
    config, store_root, bare = campaign
    base = yaml.safe_load(config.read_text())
    second = dict(base)
    second["tables"] = [dict(base["tables"][0])
                        | {"probe_name": "orders_variant", "proxy": True}]
    second_path = tmp_path / "second.yaml"
    second_path.write_text(yaml.safe_dump(second))
    assert main(["run", "--config", str(config), "--config", str(second_path),
                 "--as-of", AS_OF, "--run-id", "r-multi"]) == 0
    files = _pushed_files(bare)
    assert "orders_main" in files["README.md"]
    assert "orders_variant" in files["README.md"]  # the second file's probe
    # its presentation settings applied too: proxy labelling in its figures
    assert any("orders_variant" in name for name in files if name.startswith("img/"))


def test_standalone_render_requires_the_matching_config(campaign, tmp_path):
    """report/publish render UNDER the supplied config: rendering a run with
    a config it was not created with (e.g. suppression toggled off) refuses."""
    config, store_root, bare = campaign
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-guard"]) == 0
    text = yaml.safe_load(config.read_text())
    text["tables"][0]["suppress_small_counts"] = False  # a DIFFERENT config
    text["tables"][0]["proxy"] = True
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(text))
    out = tmp_path / "out"
    assert main(["report", "--config", str(other), "--out", str(out)]) == 1
    assert main(["publish", "--config", str(other), "--out", str(out)]) == 1
    assert main(["report", "--config", str(config), "--out", str(out)]) == 0


def test_render_record_failure_rolls_the_swap_back(campaign, tmp_path, monkeypatch):
    """The render stage is only real once RECORDED: when recording fails, the
    previously rendered artifacts are restored, and a publish-only resume is
    refused until a successful render records itself."""
    from metricprobe.store import ParquetStore as PS

    config, store_root, bare = campaign
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-roll"]) == 0
    artifacts = tmp_path / "worktree" / "artifacts" / "r-roll"
    before = (artifacts / "README.md").read_bytes()

    real_record = PS.record_stage

    def failing_record(self, run_id, stage, info):
        if stage == "render":
            raise RuntimeError("injected record failure")
        return real_record(self, run_id, stage, info)

    monkeypatch.setattr(PS, "record_stage", failing_record)
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "render", "--run-id", "r-roll"]) == 1
    # the previous complete artifacts are back in place
    assert (artifacts / "README.md").read_bytes() == before
    monkeypatch.undo()
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "render", "--run-id", "r-roll"]) == 0


def test_partial_push_rolls_earlier_remotes_back(campaign, tmp_path, monkeypatch):
    """When a later push fails after an earlier one landed, the compensating
    rollback restores the earlier remote to its prior state — the failed
    stage leaves nothing partial — and the retry then publishes both."""
    import metricprobe.publish as publish_module

    config, store_root, bare = campaign
    second = tmp_path / "mirror.git"
    second.mkdir()
    _git(second, "init", "--bare", "--initial-branch", "main")
    text = yaml.safe_load(config.read_text())
    text["delivery"]["remotes"].append(
        {"name": "mirror", "url": f"file://{second}", "ref": "main"}
    )
    two = tmp_path / "two.yaml"
    two.write_text(yaml.safe_dump(text))

    real_git = publish_module._git
    state = {"fail_mirror_push": True}

    def flaky_git(clone, *args, **kwargs):
        if (
            state["fail_mirror_push"]
            and args[0] == "push"
            and "mirror.git" in " ".join(str(a) for a in args)
        ):
            raise RuntimeError("injected network failure")
        return real_git(clone, *args, **kwargs)

    monkeypatch.setattr(publish_module, "_git", flaky_git)
    assert main(["run", "--config", str(two), "--as-of", AS_OF,
                 "--run-id", "r-partial"]) == 1
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "publish" not in manifest["stages"]  # PUBLISHED never claimed
    # the first remote was pushed, then ROLLED BACK to an unpublished state
    # (an empty commit: deleting a bare remote's default branch is commonly
    # prohibited) — no dashboard, no marker survives the failed stage
    assert "publish_partial" not in manifest["stages"]
    assert _pushed_files(bare) == {}
    # the retry converges to a full publication of BOTH remotes
    state["fail_mirror_push"] = False
    assert main(["run", "--config", str(two), "--as-of", AS_OF,
                 "--resume-from", "publish", "--run-id", "r-partial"]) == 0
    (manifest,) = ParquetStore(store_root).list_runs()
    assert manifest["stages"]["publish"]["remotes"] == ["origin", "mirror"]
    for remote in (bare, second):
        assert "README.md" in _pushed_files(remote)


def test_failed_rollback_is_recorded_as_partial(campaign, tmp_path, monkeypatch):
    """Only when the compensating rollback ITSELF fails does the manifest
    record publish_partial — the honest fallback, never a silent lie."""
    import metricprobe.publish as publish_module

    config, store_root, bare = campaign
    second = tmp_path / "mirror.git"
    second.mkdir()
    _git(second, "init", "--bare", "--initial-branch", "main")
    text = yaml.safe_load(config.read_text())
    text["delivery"]["remotes"].append(
        {"name": "mirror", "url": f"file://{second}", "ref": "main"}
    )
    two = tmp_path / "two.yaml"
    two.write_text(yaml.safe_dump(text))

    real_git = publish_module._git

    def hostile_git(clone, *args, **kwargs):
        args_text = " ".join(str(a) for a in args)
        if args[0] == "push" and "mirror.git" in args_text:
            raise RuntimeError("injected network failure")
        if args[0] == "push" and "force-with-lease" in args_text:
            raise RuntimeError("rollback also failed")
        return real_git(clone, *args, **kwargs)

    monkeypatch.setattr(publish_module, "_git", hostile_git)
    assert main(["run", "--config", str(two), "--as-of", AS_OF,
                 "--run-id", "r-stuck"]) == 1
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "publish" not in manifest["stages"]
    assert manifest["stages"]["publish_partial"]["remotes"] == ["origin"]
    assert "README.md" in _pushed_files(bare)  # origin genuinely still holds it


def test_publish_record_failure_aborts_before_any_push(campaign, monkeypatch):
    """The publish record is STAGED before anything is pushed: a failing
    record aborts the stage while the remote is still untouched — no push,
    no rollback needed, nothing recorded — and the retry publishes."""
    from metricprobe.store import ParquetStore as PS

    config, store_root, bare = campaign

    real_prepare = PS.prepare_stage

    def failing_prepare(self, run_id, stage, info):
        if stage == "publish":
            raise RuntimeError("injected publish-record failure")
        return real_prepare(self, run_id, stage, info)

    monkeypatch.setattr(PS, "prepare_stage", failing_prepare)
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-rec"]) == 1
    # nothing was ever pushed: the record failed before delivery started,
    # so the remote does not even have the branch
    assert _git(bare, "ls-remote", "--heads", str(bare), "main") == ""
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "publish" not in manifest["stages"]
    assert "publish_partial" not in manifest["stages"]
    monkeypatch.undo()
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "publish", "--run-id", "r-rec"]) == 0
    assert "README.md" in _pushed_files(bare)


def test_record_finalize_failure_rolls_the_remote_back(campaign, monkeypatch):
    """The residual window after staging: the finalize (one atomic rename)
    fails AFTER the push landed. The compensating rollback restores the
    single remote — atomicity holds — and the retry publishes cleanly."""
    from metricprobe.store import ParquetStore as PS

    config, store_root, bare = campaign

    real_prepare = PS.prepare_stage

    def sabotaged_prepare(self, run_id, stage, info):
        finalize = real_prepare(self, run_id, stage, info)
        if stage != "publish":
            return finalize

        def failing_finalize():
            raise RuntimeError("injected finalize failure")

        return failing_finalize

    monkeypatch.setattr(PS, "prepare_stage", sabotaged_prepare)
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-fin"]) == 1
    # the push was compensated: the remote holds no dashboard
    assert _pushed_files(bare) == {}
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "publish" not in manifest["stages"]
    assert "publish_partial" not in manifest["stages"]
    monkeypatch.undo()
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "publish", "--run-id", "r-fin"]) == 0
    assert "README.md" in _pushed_files(bare)


def test_finalize_and_rollback_both_failing_is_honest_partial(
    campaign, monkeypatch
):
    """The unavoidable residue: finalize fails after the push AND the
    compensation cannot reach the remote. The manifest records
    publish_partial — never PUBLISHED — and the retry converges."""
    import metricprobe.publish as publish_module
    from metricprobe.store import ParquetStore as PS

    config, store_root, bare = campaign

    real_prepare = PS.prepare_stage

    def sabotaged_prepare(self, run_id, stage, info):
        finalize = real_prepare(self, run_id, stage, info)
        if stage != "publish":
            return finalize

        def failing_finalize():
            raise RuntimeError("injected finalize failure")

        return failing_finalize

    real_git = publish_module._git

    def hostile_git(clone, *args, **kwargs):
        if args[0] == "push" and any("force-with-lease" in str(a) for a in args):
            raise RuntimeError("rollback also failed")
        return real_git(clone, *args, **kwargs)

    monkeypatch.setattr(PS, "prepare_stage", sabotaged_prepare)
    monkeypatch.setattr(publish_module, "_git", hostile_git)
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--run-id", "r-res"]) == 1
    (manifest,) = ParquetStore(store_root).list_runs()
    assert "publish" not in manifest["stages"]  # PUBLISHED never claimed
    assert manifest["stages"]["publish_partial"]["remotes"] == ["origin"]
    assert "README.md" in _pushed_files(bare)  # honestly still holds the run
    monkeypatch.undo()
    assert main(["run", "--config", str(config), "--as-of", AS_OF,
                 "--resume-from", "publish", "--run-id", "r-res"]) == 0
    assert "README.md" in _pushed_files(bare)


def test_export_stub_survives_test_level_undo(monkeypatch):
    """Several tests in this module call monkeypatch.undo(); the autouse
    export stub must live on its OWN MonkeyPatch so an undo cannot strip it
    and relaunch real Chrome from an ungrouped worker mid-test."""
    import plotly.io as pio

    monkeypatch.setattr("os.environ", dict(**__import__("os").environ))
    monkeypatch.undo()
    assert pio.to_image(None).startswith(b"\x89PNG")  # still the stub
