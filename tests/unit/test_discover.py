"""Step 8: the INFORMATION_SCHEMA scanner — inventory, role matching with
shipped-but-overridable candidates, and draft YAML emission (DuckDB here; the
mssql path runs in the equivalence job)."""

import duckdb
import pytest
import sqlalchemy as sa
import yaml

from metricprobe.cli import main
from metricprobe.config import ConfigError, load_config
from metricprobe.discover import (
    datetime_inventory,
    draft_config,
    match_roles,
    scan_columns,
)

DDL = [
    """CREATE TABLE orders (
        order_id BIGINT, event_time DATE, load_time TIMESTAMP,
        batch_id VARCHAR, amount DOUBLE)""",
    """CREATE TABLE telemetry (
        device VARCHAR, recorded_at TIMESTAMP, ingested_at TIMESTAMP,
        created_at TIMESTAMP)""",
    """CREATE TABLE ambiguous (only_stamp TIMESTAMP, note VARCHAR)""",
    """CREATE TABLE reference_codes (code INTEGER, label VARCHAR)""",  # no datetimes
]


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("disc") / "demo.duckdb"
    con = duckdb.connect(str(path))
    for statement in DDL:
        con.execute(statement)
    con.close()
    return path


@pytest.fixture(scope="module")
def engine(demo_db):
    return sa.create_engine(f"duckdb:///{demo_db}")


def test_scan_inventories_datetime_columns(engine):
    columns = scan_columns(engine, "demo", schema="main")
    inventory = datetime_inventory(columns)
    assert set(inventory) == {("main", "orders"), ("main", "telemetry"), ("main", "ambiguous")}
    assert [c.column for c in inventory[("main", "orders")]] == ["event_time", "load_time"]
    # TIME-of-day columns and plain scalars never enter the inventory
    assert ("main", "reference_codes") not in inventory


def test_role_matching_with_shipped_defaults(engine):
    columns = scan_columns(engine, "demo", schema="main")
    orders = [c for c in columns if c.table == "orders"]
    roles = match_roles(orders)
    assert roles["event_time"] == ["event_time"]
    assert roles["load_time"] == ["load_time"]
    assert roles["load_batch_col"] == ["batch_id"]
    telemetry = [c for c in columns if c.table == "telemetry"]
    roles = match_roles(telemetry)
    assert roles["event_time"] == ["recorded_at"]
    assert roles["load_time"] == ["ingested_at"]
    assert roles["source_insert_time"] == ["created_at"]


def test_candidates_are_overridable(engine):
    columns = [c for c in scan_columns(engine, "demo", schema="main") if c.table == "ambiguous"]
    assert match_roles(columns)["event_time"] == []  # defaults find nothing
    custom = match_roles(columns, candidates={"event_time": ("only_stamp",)})
    assert custom["event_time"] == ["only_stamp"]


def test_draft_yaml_parses_and_complete_tables_validate(engine, demo_db, tmp_path):
    url = f"duckdb:///{demo_db}"
    draft = draft_config(engine, "demo", url, schema="main")
    parsed = yaml.safe_load(draft)  # comments and all, it IS valid YAML
    entries = {entry["probe_name"]: entry for entry in parsed["tables"]}
    assert set(entries) == {"orders_main", "telemetry_main", "ambiguous_main"}
    assert entries["orders_main"]["event_time"] == "event_time"
    assert entries["orders_main"]["load_time"] == "load_time"
    # optional roles are COMMENTED suggestions, never silently enabled
    assert entries["orders_main"].get("load_batch_col") is None
    assert "# load_batch_col: batch_id" in draft
    # a table with no candidates stays loudly incomplete: the empty role plus
    # a FIXME comment, and the loader refuses it until a human decides
    assert entries["ambiguous_main"]["event_time"] is None
    assert "FIXME — choose one of: only_stamp" in draft
    draft_path = tmp_path / "draft.yaml"
    draft_path.write_text(draft)
    with pytest.raises(ConfigError, match="load_time"):
        load_config(draft_path)
    # with the FIXME table removed, the draft loads as a valid config
    parsed["tables"] = [e for e in parsed["tables"] if e["probe_name"] != "ambiguous_main"]
    trimmed = tmp_path / "trimmed.yaml"
    trimmed.write_text(yaml.safe_dump(parsed))
    config = load_config(trimmed)
    assert [t.probe_name for t in config.tables] == ["orders_main", "telemetry_main"]


def test_cli_discover_writes_a_draft(demo_db, tmp_path, capsys):
    url = f"duckdb:///{demo_db}"
    assert main(["discover", "--url", url, "--database", "demo", "--schema", "main"]) == 0
    printed = capsys.readouterr().out
    assert "orders_main" in printed and "FIXME" in printed
    out = tmp_path / "draft.yaml"
    assert (
        main(["discover", "--url", url, "--database", "demo", "--out", str(out)]) == 0
    )
    assert "orders_main" in out.read_text()
    # a bad URL is an execution error (1), never the data-health code
    assert main(["discover", "--url", "duckdb:///nope/nope.duckdb", "--database", "x"]) == 1
