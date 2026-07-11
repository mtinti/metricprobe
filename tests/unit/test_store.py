"""Store contract: begin -> staging -> ATOMIC manifest commit (or abort);
readers only ever see manifest-committed runs; every row is stamped."""

import pandas as pd
import pytest

from metricprobe.cli import check_monotonic_publication
from metricprobe.store import STAMP_COLUMNS, ParquetStore, RunMeta, stamp


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


def test_readers_see_only_committed_runs(tmp_path):
    store = ParquetStore(tmp_path)
    store.begin_run(meta("r1"))
    store.write_table("r1", "statuses", pd.DataFrame({"probe": ["a"]}))
    assert store.list_runs() == []  # staged but NOT committed: invisible
    with pytest.raises(FileNotFoundError):
        store.read_table("r1", "statuses")
    store.commit_run("r1", {"run_id": "r1", "run_at": "2026-07-01T06:00:00"})
    assert [m["run_id"] for m in store.list_runs()] == ["r1"]
    assert store.read_table("r1", "statuses")["probe"].tolist() == ["a"]


def test_abort_leaves_nothing_partial(tmp_path):
    store = ParquetStore(tmp_path)
    store.begin_run(meta("r1"))
    store.write_table("r1", "statuses", pd.DataFrame({"probe": ["a"]}))
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
        store.write_table(f"r{i}", "statuses", pd.DataFrame({"probe": ["a"]}))
        store.commit_run(f"r{i}", {"run_id": f"r{i}", "run_at": run_meta.run_at})
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
