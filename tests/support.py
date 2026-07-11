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
