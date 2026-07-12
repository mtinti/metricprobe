"""Store contract: begin -> staging -> ATOMIC manifest commit (or abort);
readers only ever see manifest-committed runs; every row is stamped."""

import pandas as pd
import pytest

from metricprobe.publish import check_monotonic_publication
from metricprobe.store import (
    STAMP_COLUMNS,
    ParquetStore,
    RunMeta,
    stamp,
    validate_run_id,
)


def meta(run_id: str, run_at: str = "2026-07-01T06:00:00") -> RunMeta:
    return RunMeta(
        run_id=run_id,
        run_at=run_at,
        as_of="2026-07-01T00:00:00",
        git_sha="deadbeef",
        tool_version="0.1.0.dev0",
        config_digest="abc123",
        schema_version=1,
        window_start="2024-07-01T00:00:00",
        window_end="2026-07-01T00:00:00",
    )


def manifest_for(run_meta: RunMeta) -> dict:
    import dataclasses

    return {**dataclasses.asdict(run_meta), "stages": {"analysis": {}}}


def frame(run_meta: RunMeta, **columns) -> "pd.DataFrame":
    return stamp(pd.DataFrame(columns or {"probe": ["a"]}), run_meta)


def test_readers_see_only_committed_runs(tmp_path):
    store = ParquetStore(tmp_path)
    store.begin_run(meta("r1"))
    store.write_table("r1", "statuses", frame(meta("r1")))
    assert store.list_runs() == []  # staged but NOT committed: invisible
    with pytest.raises(FileNotFoundError):
        store.read_table("r1", "statuses")
    store.commit_run("r1", manifest_for(meta("r1")))
    assert [m["run_id"] for m in store.list_runs()] == ["r1"]
    assert store.read_table("r1", "statuses")["probe"].tolist() == ["a"]


def test_abort_leaves_nothing_partial(tmp_path):
    store = ParquetStore(tmp_path)
    store.begin_run(meta("r1"))
    store.write_table("r1", "statuses", frame(meta("r1")))
    store.abort_run("r1")
    assert store.list_runs() == []
    assert list((tmp_path / "staging").iterdir()) == []
    # the run id is reusable after an abort (idempotent retry)
    store.begin_run(meta("r1"))


def test_duplicate_run_ids_are_rejected(tmp_path):
    store = ParquetStore(tmp_path)
    store.begin_run(meta("r1"))
    with pytest.raises(FileExistsError):
        store.begin_run(meta("r1"))


def test_every_row_is_stamped(tmp_path):
    stamped = stamp(pd.DataFrame({"x": [1, 2]}), meta("r1"))
    for column in STAMP_COLUMNS:
        assert column in stamped.columns
    assert stamped["run_id"].tolist() == ["r1", "r1"]
    assert stamped["git_sha"].tolist() == ["deadbeef", "deadbeef"]


def test_retention_prunes_oldest_committed_runs(tmp_path):
    store = ParquetStore(tmp_path)
    for i, hour in enumerate(("01", "02", "03")):
        run_meta = meta(f"r{i}", run_at=f"2026-07-01T{hour}:00:00")
        store.begin_run(run_meta)
        store.write_table(f"r{i}", "statuses", frame(run_meta))
        store.commit_run(f"r{i}", manifest_for(run_meta))
    dropped = store.prune(keep=2)
    assert dropped == ["r0"]
    assert [m["run_id"] for m in store.list_runs()] == ["r1", "r2"]


def test_monotonic_publication_guard():
    # an older failed run may never overwrite a newer published dashboard
    check_monotonic_publication("2026-07-02T00:00:00", ["2026-07-01T00:00:00"])
    with pytest.raises(RuntimeError, match="newer run"):
        check_monotonic_publication(
            "2026-07-01T00:00:00", ["2026-06-30T00:00:00", "2026-07-02T00:00:00"]
        )
    # timestamps are compared as INSTANTS, never lexicographically: the
    # candidate string sorts later but is 19h OLDER in UTC than the published
    with pytest.raises(RuntimeError, match="newer run"):
        check_monotonic_publication(
            "2026-07-02T00:00:00+10:00",  # = 2026-07-01T14:00Z
            ["2026-07-01T20:00:00+00:00"],
        )


def test_run_ids_cannot_traverse_the_store(tmp_path):
    store = ParquetStore(tmp_path)
    for evil in ("../evil", "/abs/path", "a/b", "..", ".hidden", "x" * 65):
        with pytest.raises(ValueError, match="invalid run_id"):
            store.begin_run(meta(evil))
    validate_run_id("20260701T060000-abc123")  # the normal shape passes


def test_store_enforces_the_frozen_snapshot_schema(tmp_path):
    store = ParquetStore(tmp_path)
    store.begin_run(meta("r1"))
    with pytest.raises(ValueError, match="stamp column"):
        store.write_table("r1", "statuses", pd.DataFrame({"probe": ["a"]}))
    store.write_table("r1", "statuses", frame(meta("r1")))
    with pytest.raises(ValueError, match="manifest is missing"):
        store.commit_run("r1", {"run_id": "r1", "run_at": "2026-07-01T06:00:00"})
    store.commit_run("r1", manifest_for(meta("r1")))


def test_prune_protects_the_completing_run(tmp_path):
    import pandas as pd

    from metricprobe.store import ParquetStore, RunMeta, stamp

    store = ParquetStore(tmp_path)
    for index, run_id in enumerate(["r-old", "r-mid", "r-new"]):
        meta = RunMeta(
            run_id=run_id, run_at=f"2026-07-0{index + 1}T00:00:00+00:00",
            as_of="2026-07-04T00:00:00", git_sha="x", tool_version="0",
            config_digest="d", schema_version=1,
            window_start="2024-07-04T00:00:00", window_end="2026-07-04T00:00:00",
        )
        store.begin_run(meta)
        store.write_table(run_id, "statuses", stamp(pd.DataFrame({"probe": ["p"]}), meta))
        store.commit_run(run_id, {
            "run_id": run_id, "run_at": meta.run_at, "as_of": meta.as_of,
            "git_sha": "x", "tool_version": "0", "config_digest": "d",
            "schema_version": 1, "window_start": meta.window_start,
            "window_end": meta.window_end, "stages": {},
        })
    # keep 1: the oldest run survives because the CALLER is still completing
    # it (a resumed old run must not be pruned before its render stage)
    dropped = store.prune(keep=1, protect="r-old")
    assert dropped == ["r-mid"]
    remaining = {m["run_id"] for m in store.list_runs()}
    assert remaining == {"r-old", "r-new"}
