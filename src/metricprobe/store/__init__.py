"""Snapshot store: transactional runs with an ATOMIC manifest commit.

Lifecycle: begin_run -> write_table (staging) -> commit_run (atomic) | abort_run.
Readers (report/publish/app) only ever see manifest-committed runs.

Backends share one interface:
  * ParquetStore (default, config backends "duckdb"/"parquet"): a directory of
    parquet files per run; the commit is an atomic same-filesystem rename of
    the fully-written staging directory into runs/. DuckDB reads the parquet
    files directly.
  * MssqlStore (config-flagged): frames appended into tables in the configured
    schema; rows are invisible to readers until the single-row manifest INSERT
    (the transactional commit point).

Every stored row is stamped (STAMP_COLUMNS): run_id, run_at, as_of, git_sha,
tool_version, config_digest, schema_version, window_start, window_end.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

from metricprobe.config import StoreConfig, expand_env

SNAPSHOT_SCHEMA_VERSION = 1

STAMP_COLUMNS = (
    "run_id",
    "run_at",
    "as_of",
    "git_sha",
    "tool_version",
    "config_digest",
    "schema_version",
    "window_start",
    "window_end",
)


# run ids become filenames and SQL values: a conservative shape, no traversal
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

MANIFEST_REQUIRED_KEYS = (
    "run_id",
    "run_at",
    "as_of",
    "git_sha",
    "tool_version",
    "config_digest",
    "schema_version",
    "window_start",
    "window_end",
    "stages",
)


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID.match(run_id) or ".." in run_id:
        raise ValueError(
            f"invalid run_id {run_id!r}: letters, digits, '.', '_', '-' only "
            "(max 64 chars, no traversal)"
        )
    return run_id


def _require_stamped(frame: pd.DataFrame) -> None:
    missing = [column for column in STAMP_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing stamp column(s) {missing}; use stamp()")


def _require_manifest(manifest: dict) -> None:
    missing = [key for key in MANIFEST_REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"manifest is missing required key(s) {missing}")


@dataclass(frozen=True)
class RunMeta:
    run_id: str
    run_at: str  # ISO timestamps: identical in parquet, JSON and mssql
    as_of: str
    git_sha: str
    tool_version: str
    config_digest: str
    schema_version: int
    window_start: str
    window_end: str


def stamp(frame: pd.DataFrame, meta: RunMeta) -> pd.DataFrame:
    """Every output row carries the full run provenance, from the first release."""
    out = frame.copy()
    for column, value in asdict(meta).items():
        out[column] = value
    return out


class ParquetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "staging").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(parents=True, exist_ok=True)

    def _staging(self, run_id: str) -> Path:
        return self.root / "staging" / validate_run_id(run_id)

    def _committed(self, run_id: str) -> Path:
        return self.root / "runs" / validate_run_id(run_id)

    def begin_run(self, meta: RunMeta) -> None:
        staging = self._staging(meta.run_id)
        if staging.exists() or self._committed(meta.run_id).exists():
            raise FileExistsError(f"run {meta.run_id!r} already exists")
        staging.mkdir(parents=True)

    def write_table(self, run_id: str, name: str, frame: pd.DataFrame) -> None:
        staging = self._staging(run_id)
        if not staging.exists():
            raise FileNotFoundError(f"run {run_id!r} has no open staging area")
        _require_stamped(frame)
        frame.to_parquet(staging / f"{name}.parquet", index=False)

    def commit_run(self, run_id: str, manifest: dict) -> None:
        """The atomic commit: manifest lands in staging first, then ONE rename
        publishes the fully-formed run directory."""
        staging = self._staging(run_id)
        if not staging.exists():
            raise FileNotFoundError(f"run {run_id!r} has no open staging area")
        _require_manifest(manifest)
        tmp = staging / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(staging / "manifest.json")
        staging.rename(self._committed(run_id))

    def abort_run(self, run_id: str) -> None:
        staging = self._staging(run_id)
        if staging.exists():
            shutil.rmtree(staging)

    def list_runs(self) -> list[dict]:
        """Committed runs only (manifest present), oldest first by run_at."""
        manifests = []
        for path in sorted((self.root / "runs").glob("*/manifest.json")):
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        return sorted(manifests, key=lambda m: m["run_at"])

    def read_table(self, run_id: str, name: str) -> pd.DataFrame:
        path = self._committed(run_id) / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"no committed table {name!r} for run {run_id!r}")
        return pd.read_parquet(path)

    def table_names(self, run_id: str) -> list[str]:
        return sorted(p.stem for p in self._committed(run_id).glob("*.parquet"))

    def prune(self, keep: int) -> list[str]:
        """Drop the oldest committed runs beyond `keep`; returns dropped ids."""
        runs = self.list_runs()
        dropped = []
        for manifest in runs[: max(0, len(runs) - keep)]:
            shutil.rmtree(self._committed(manifest["run_id"]))
            dropped.append(manifest["run_id"])
        return dropped


class MssqlStore:
    """Same interface over SQL Server — data rows are staged into per-table
    physical tables (invisible without a manifest row); the single manifest
    INSERT is the atomic commit point."""

    MANIFEST_TABLE = "mp_run_manifest"
    STAGING_TABLE = "mp_run_staging"

    def __init__(self, url: str, schema: str):
        self.engine = sa.create_engine(url)
        self.schema = schema
        with self.engine.begin() as conn:
            conn.exec_driver_sql(
                f"IF OBJECT_ID('{self.schema}.{self.MANIFEST_TABLE}') IS NULL "
                f"CREATE TABLE {self.schema}.{self.MANIFEST_TABLE} ("
                "run_id varchar(64) NOT NULL PRIMARY KEY, "
                "run_at varchar(40) NOT NULL, manifest nvarchar(max) NOT NULL)"
            )
            conn.exec_driver_sql(
                f"IF OBJECT_ID('{self.schema}.{self.STAGING_TABLE}') IS NULL "
                f"CREATE TABLE {self.schema}.{self.STAGING_TABLE} ("
                "run_id varchar(64) NOT NULL PRIMARY KEY, "
                "claimed_at varchar(40) NOT NULL)"
            )
        self._staged: dict[str, list[str]] = {}

    def begin_run(self, meta: RunMeta) -> None:
        """The staging claim is a SERVER-SIDE primary-key insert: two processes
        can never stage the same run_id, so an abort can never touch another
        writer's rows."""
        validate_run_id(meta.run_id)
        if any(m["run_id"] == meta.run_id for m in self.list_runs()):
            raise FileExistsError(f"run {meta.run_id!r} already exists")
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    sa.text(
                        f"INSERT INTO {self.schema}.{self.STAGING_TABLE} "
                        "(run_id, claimed_at) VALUES (:run_id, :claimed_at)"
                    ),
                    {"run_id": meta.run_id, "claimed_at": meta.run_at},
                )
        except sa.exc.IntegrityError as exc:
            raise FileExistsError(
                f"run {meta.run_id!r} is already staged by another writer"
            ) from exc
        self._staged[meta.run_id] = []

    def write_table(self, run_id: str, name: str, frame: pd.DataFrame) -> None:
        if run_id not in self._staged:
            raise FileNotFoundError(f"run {run_id!r} has no open staging area")
        _require_stamped(frame)
        frame.to_sql(
            f"mp_{name}", self.engine, schema=self.schema, if_exists="append", index=False
        )
        self._staged[run_id].append(name)

    def commit_run(self, run_id: str, manifest: dict) -> None:
        if run_id not in self._staged:
            raise FileNotFoundError(f"run {run_id!r} has no open staging area")
        _require_manifest(manifest)
        with self.engine.begin() as conn:  # ONE transaction: manifest + claim release
            conn.execute(
                sa.text(
                    f"INSERT INTO {self.schema}.{self.MANIFEST_TABLE} "
                    "(run_id, run_at, manifest) VALUES (:run_id, :run_at, :manifest)"
                ),
                {
                    "run_id": run_id,
                    "run_at": manifest["run_at"],
                    "manifest": json.dumps(manifest, sort_keys=True),
                },
            )
            conn.execute(
                sa.text(
                    f"DELETE FROM {self.schema}.{self.STAGING_TABLE} WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )
        del self._staged[run_id]

    def abort_run(self, run_id: str) -> None:
        """Deletes data rows ONLY when no manifest exists for the id (a
        committed run is never touched) and releases the staging claim."""
        names = self._staged.pop(run_id, [])
        with self.engine.begin() as conn:
            committed = conn.execute(
                sa.text(
                    f"SELECT COUNT(*) FROM {self.schema}.{self.MANIFEST_TABLE} "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            ).scalar_one()
            if not committed:
                for name in set(names):
                    conn.execute(
                        sa.text(
                            f"DELETE FROM {self.schema}.mp_{name} WHERE run_id = :run_id"
                        ),
                        {"run_id": run_id},
                    )
            conn.execute(
                sa.text(
                    f"DELETE FROM {self.schema}.{self.STAGING_TABLE} WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )

    def list_runs(self) -> list[dict]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.text(
                    f"SELECT manifest FROM {self.schema}.{self.MANIFEST_TABLE} ORDER BY run_at"
                )
            ).scalars()
            return [json.loads(row) for row in rows]

    def read_table(self, run_id: str, name: str) -> pd.DataFrame:
        if not any(m["run_id"] == run_id for m in self.list_runs()):
            raise FileNotFoundError(f"run {run_id!r} is not committed")
        with self.engine.connect() as conn:
            return pd.read_sql(
                sa.text(f"SELECT * FROM {self.schema}.mp_{name} WHERE run_id = :run_id"),
                conn,
                params={"run_id": run_id},
            )

    def _data_tables(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                sa.text(
                    "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                    "WHERE TABLE_SCHEMA = :s AND TABLE_NAME LIKE 'mp_%'"
                ),
                {"s": self.schema},
            ).scalars()
            return [
                name
                for name in rows
                if name not in (self.MANIFEST_TABLE, self.STAGING_TABLE)
            ]

    def prune(self, keep: int) -> list[str]:
        """Same retention contract as the parquet store."""
        runs = self.list_runs()
        dropped = []
        for manifest in runs[: max(0, len(runs) - keep)]:
            run_id = manifest["run_id"]
            with self.engine.begin() as conn:
                for name in self._data_tables():
                    conn.execute(
                        sa.text(f"DELETE FROM {self.schema}.{name} WHERE run_id = :run_id"),
                        {"run_id": run_id},
                    )
                conn.execute(
                    sa.text(
                        f"DELETE FROM {self.schema}.{self.MANIFEST_TABLE} "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": run_id},
                )
            dropped.append(run_id)
        return dropped


def open_store(config: StoreConfig):
    if config.backend == "mssql":
        # the schema comes from config, like every database locator
        return MssqlStore(expand_env(config.mssql_url), schema=config.mssql_schema)
    return ParquetStore(config.path)
