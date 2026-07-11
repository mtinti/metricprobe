"""Shared helpers for golden/property/equivalence tests: build TableConfigs for
the synthetic tables and run the canonical pass on a DuckDB engine."""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa
from tests.synth import generator as g

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult, run_canonical


def table_config(**overrides) -> TableConfig:
    data = {
        "probe_name": "events_probe",
        "database": "memory",  # DuckDB's in-memory catalog name
        "schema": "main",
        "table": "events",
        "event_time": "event_time",
        "load_time": "load_time",
    }
    analysis = overrides.pop("analysis", None)
    data.update(overrides)
    if analysis:
        data["analysis"] = analysis
    if "resolution" not in data:
        # synthetic fixtures carry full timestamps: default every configured
        # time role to datetime unless a test declares otherwise
        via = data.get("event_time_via") or {}
        columns = [
            column
            for column in (
                data.get("event_time"),
                via.get("column") if isinstance(via, dict) else None,
                data.get("load_time"),
                data.get("source_insert_time"),
            )
            if column
        ]
        data["resolution"] = dict.fromkeys(columns, "datetime")
    return TableConfig.model_validate(data)


def engine_with(df: pd.DataFrame, table: str = "events") -> sa.Engine:
    engine = sa.create_engine("duckdb:///:memory:")
    g.load_via_sqlalchemy(df, engine, table)
    return engine


def probe(df: pd.DataFrame, config: TableConfig, as_of: str | pd.Timestamp) -> CanonicalResult:
    engine = engine_with(df, config.table)
    try:
        return run_canonical(engine, config, pd.Timestamp(as_of))
    finally:
        engine.dispose()


def probe_dual(df: pd.DataFrame, config: TableConfig, as_of: str | pd.Timestamp):
    from metricprobe.extract.dual import run_dual_lag

    engine = engine_with(df, config.table)
    try:
        return run_dual_lag(engine, config, pd.Timestamp(as_of))
    finally:
        engine.dispose()
