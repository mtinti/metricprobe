"""Shared fixture for the render suites (viz smoke / report / publish): ONE
production CLI run over EVERY pathology twin plus dual, parity and a
suppressed probe, committed to a parquet store the way production writes it."""

from __future__ import annotations

import dataclasses

import duckdb
import pytest
import yaml
from tests.synth import generator as g
from tests.synth.scenarios import DUAL_BASE, TRICKLE_BASE, catalog

from metricprobe.cli import main
from metricprobe.config import CONFIG_SCHEMA_VERSION, load_config
from metricprobe.store import ParquetStore

# 3 days after the newest batch load (sustained_collapse's 2025-06 +20d batch):
# batchy feeds read fresh, trickle feeds read stale — both verdicts appear
AS_OF = "2025-07-24"
RUN_ID = "render-fixture"

TINY = g.TableSpec(
    name="tiny", start_month="2024-01", n_months=18, rows_per_month=3,
    lag_model=g.LognormalLag(mu=1.0, sigma=0.5), seed=404,
)


def _entry(table: str, probe: str, **overrides) -> dict:
    time_columns = {"event_time": "datetime", "load_time": "datetime"}
    for role in ("source_insert_time", "compare_event_time"):
        if overrides.get(role):
            time_columns[overrides[role]] = "datetime"
    return {
        "probe_name": probe,
        "database": "demo",
        "schema": "main",
        "table": table,
        "event_time": "event_time",
        "load_time": "load_time",
        "resolution": time_columns,
    } | overrides


@pytest.fixture(scope="session")
def dashboard_run(tmp_path_factory):
    """(store, run_id, config) after one full production run."""
    root = tmp_path_factory.mktemp("render")
    db_path = root / "demo.duckdb"
    con = duckdb.connect(str(db_path))

    entries: list[dict] = []
    pairs = catalog()
    per_scenario = {
        "duplicate_keys": {"key_cols": ["row_id"]},
        "straggler_batch": {"load_batch_col": "batch_id", "expect_batchy": True},
        "sustained_collapse": {"load_batch_col": "batch_id"},
        "sustained_collapse_short": {
            "load_batch_col": "batch_id",
            "analysis": {"lag_cap_days": 52, "training_cutoff_days": 54},
        },
        "raw_vs_corrected": {
            "compare_event_time": "event_time_raw",
        },
    }
    for name, pair in pairs.items():
        for twin, frame in (("ok", pair.healthy()), ("bad", pair.unhealthy())):
            table = f"{name}_{twin}"
            g.load_into_duckdb(frame, con, table)
            entries.append(_entry(table, f"{table}_probe", **per_scenario.get(name, {})))

    dual_df = g.generate(DUAL_BASE)
    g.load_into_duckdb(dual_df, con, "dual_registry")
    entries.append(
        _entry(
            "dual_registry",
            "dual_registry_probe",
            source_insert_time="source_insert_time",
            proxy=True,
        )
    )

    parity_df = g.generate(dataclasses.replace(TRICKLE_BASE, seed=515))
    g.load_into_duckdb(parity_df, con, "parity_a")
    dropped = parity_df[parity_df["event_time"].dt.to_period("M") != "2024-03"]
    g.load_into_duckdb(dropped.reset_index(drop=True), con, "parity_b")
    entries.append(
        _entry("parity_a", "parity_a_probe", key_cols=["row_id"], parity_with="parity_b_probe")
    )
    entries.append(_entry("parity_b", "parity_b_probe", key_cols=["row_id"]))

    g.load_into_duckdb(g.generate(TINY), con, "tiny_counts")
    entries.append(_entry("tiny_counts", "tiny_probe", suppress_small_counts=True))
    con.close()

    config_path = root / "probe.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "connection_url": f"duckdb:///{db_path}",
                "store": {"path": str(root / "store")},
                "campaign": {"schedule": "0 6 * * 1", "timezone": "UTC"},
                "tables": entries,
            }
        )
    )
    code = main(["run", "--config", str(config_path), "--as-of", AS_OF,
                 "--run-id", RUN_ID])
    assert code in (0, 2)  # pathology twins make REDs; both mean COMMITTED
    store = ParquetStore(root / "store")
    assert store.list_runs()
    return store, RUN_ID, load_config(config_path)
