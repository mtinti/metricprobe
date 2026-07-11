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


def test_yaml_hostile_identifiers_survive(tmp_path):
    """Identifiers named `yes` or containing ':' must round-trip as STRINGS."""
    path = tmp_path / "hostile.duckdb"
    con = duckdb.connect(str(path))
    con.execute('CREATE TABLE "yes" ("event: time" TIMESTAMP, "load" TIMESTAMP)')
    con.close()
    engine = sa.create_engine(f"duckdb:///{path}")
    try:
        draft = draft_config(engine, "hostile", f"duckdb:///{path}", schema="main")
    finally:
        engine.dispose()
    parsed = yaml.safe_load(draft)
    (entry,) = parsed["tables"]
    assert entry["table"] == "yes"  # a string, not boolean True
    assert entry["probe_name"] == "yes_main"
    assert entry["event_time"] == "event: time"
    assert entry["resolution"] == {"event: time": "datetime", "load": "datetime"}


def test_duplicate_table_names_across_schemas_get_unique_probe_names(tmp_path):
    path = tmp_path / "multi.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA alpha")
    con.execute("CREATE SCHEMA beta")
    for schema in ("alpha", "beta"):
        con.execute(
            f"CREATE TABLE {schema}.orders (event_time DATE, load_time TIMESTAMP)"
        )
    con.close()
    engine = sa.create_engine(f"duckdb:///{path}")
    try:
        draft = draft_config(engine, "multi", f"duckdb:///{path}")
    finally:
        engine.dispose()
    parsed = yaml.safe_load(draft)
    names = sorted(entry["probe_name"] for entry in parsed["tables"])
    assert names == ["alpha_orders_main", "beta_orders_main"]
    # the deduplicated draft VALIDATES (campaign-wide probe-name uniqueness)
    draft_path = tmp_path / "multi.yaml"
    draft_path.write_text(draft)
    config = load_config(draft_path)
    assert len(config.tables) == 2


def test_probe_name_dedupe_is_globally_collision_safe(tmp_path):
    # schema-qualification alone is not enough: alpha.orders qualifies to
    # alpha_orders_main, which COLLIDES with main.alpha_orders' unqualified
    # name — the numeric-suffix fallback must keep the draft valid
    path = tmp_path / "clash.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA alpha")
    con.execute("CREATE SCHEMA beta")
    con.execute("CREATE TABLE alpha.orders (event_time DATE, load_time TIMESTAMP)")
    con.execute("CREATE TABLE beta.orders (event_time DATE, load_time TIMESTAMP)")
    con.execute("CREATE TABLE main.alpha_orders (event_time DATE, load_time TIMESTAMP)")
    con.close()
    engine = sa.create_engine(f"duckdb:///{path}")
    try:
        draft = draft_config(engine, "clash", f"duckdb:///{path}")
    finally:
        engine.dispose()
    parsed = yaml.safe_load(draft)
    names = [entry["probe_name"] for entry in parsed["tables"]]
    assert len(names) == len(set(names)) == 3
    assert sorted(names) == ["alpha_orders_main", "alpha_orders_main_2", "beta_orders_main"]
    draft_path = tmp_path / "clash.yaml"
    draft_path.write_text(draft)
    assert len(load_config(draft_path).tables) == 3


def test_draft_declares_per_column_resolution(engine):
    draft = draft_config(engine, "demo", "duckdb:///demo", schema="main")
    parsed = yaml.safe_load(draft)
    entries = {entry["probe_name"]: entry for entry in parsed["tables"]}
    # orders: event_time is DATE, load_time is TIMESTAMP
    assert entries["orders_main"]["resolution"] == {
        "event_time": "date",
        "load_time": "datetime",
    }
    assert entries["telemetry_main"]["resolution"] == {
        "recorded_at": "datetime",
        "ingested_at": "datetime",
    }


def test_cli_candidate_overrides_and_guarded_output(demo_db, tmp_path, capsys):
    url = f"duckdb:///{demo_db}"
    # the ambiguous table's lone stamp becomes the event via a CLI override
    assert (
        main([
            "discover", "--url", url, "--database", "demo", "--schema", "main",
            "--candidates", "event_time=only_stamp",
        ])
        == 0
    )
    printed = capsys.readouterr().out
    parsed = yaml.safe_load(printed)
    entries = {entry["probe_name"]: entry for entry in parsed["tables"]}
    assert entries["ambiguous_main"]["event_time"] == "only_stamp"
    # a malformed override is an execution error
    assert (
        main(["discover", "--url", url, "--database", "demo",
              "--candidates", "bogus_role=x"])
        == 1
    )
    # an unwritable output path is an execution error, not a traceback
    assert (
        main(["discover", "--url", url, "--database", "demo",
              "--out", str(tmp_path / "missing_dir" / "draft.yaml")])
        == 1
    )


def test_candidate_overrides_are_case_insensitive(engine):
    columns = [c for c in scan_columns(engine, "demo", schema="main") if c.table == "ambiguous"]
    upper = match_roles(columns, candidates={"event_time": ("ONLY_STAMP",)})
    assert upper["event_time"] == ["only_stamp"]


def test_newline_identifiers_do_not_break_draft_comments(tmp_path):
    # a legal (if hostile) newline-bearing table name must not inject raw
    # newlines into a YAML comment and split the document mid-line
    path = tmp_path / "newline.duckdb"
    con = duckdb.connect(str(path))
    con.execute('CREATE TABLE "bad\ntable" (event_time DATE, load_time TIMESTAMP)')
    con.close()
    engine = sa.create_engine(f"duckdb:///{path}")
    try:
        draft = draft_config(engine, "newline", f"duckdb:///{path}", schema="main")
    finally:
        engine.dispose()
    parsed = yaml.safe_load(draft)  # the draft still parses
    (entry,) = parsed["tables"]
    assert entry["table"] == "bad\ntable"  # the VALUE round-trips exactly
