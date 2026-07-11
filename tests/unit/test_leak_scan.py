"""Tests for the private-material CI guard (scripts/scan_private_material.py)."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "scan_private_material", REPO_ROOT / "scripts" / "scan_private_material.py"
)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)


def test_private_file_names_are_flagged():
    for path in ["CLAUDE.private.md", "docs/CLAUDE.private.md", "notes.private.yaml"]:
        assert scan.path_violation(path) is not None, path


def test_ordinary_file_names_pass():
    for path in ["CLAUDE.md", "src/metricprobe/config.py", "tests/unit/test_leak_scan.py"]:
        assert scan.path_violation(path) is None, path


# Offending samples are BUILT AT RUNTIME (never as contiguous literals) so the
# repo self-scan below does not flag this very test file.
def _mssql_url(host: str) -> str:
    return "mssql+pyodbc:" + f"//@{host}/SomeDb?driver=ODBC+Driver+18+for+SQL+Server"


def test_real_connection_host_is_flagged():
    text = f'url = "{_mssql_url("SQLPROD01")}"'
    assert scan.content_violations("examples/x.yaml", text, [])


def test_placeholder_and_loopback_hosts_pass():
    text = (
        "mssql+pyodbc://@SERVER/DB?Trusted_Connection=yes\n"
        "mssql+pymssql://sa:pw@localhost:1433/tempdb\n"
        "duckdb:///:memory:\n"
    )
    assert scan.content_violations("README.md", text, []) == []


def test_unc_path_is_flagged():
    backslash = chr(92)
    unc = backslash * 2 + "fileserver" + backslash + "share"
    assert scan.content_violations("doc.md", f"data lives on {unc}", [])


def test_environment_shaped_fields_need_no_marker_list():
    # A plain server/database/schema field with a real-looking value must fail
    # even though CI has no private marker list to compare against.
    for key in ["server", "database", "schema", "hostname", "password", "Data Source"]:
        field = key + ": realhost42"
        assert scan.content_violations("cfg.yaml", field, []), field
        assert scan.content_violations("cfg.yaml", field.replace(": ", "="), []), field


def test_environment_shaped_fields_with_placeholders_pass():
    for value in ["localhost", "localhost:1433", "${PROD_SERVER}", "$PROD_SERVER",
                  "<your-server>", "%SERVER%", "demo_retail", "tempdb", "dbo",
                  "Metricprobe1!", "example.com", "str"]:  # "str": type annotation in code
        field = "ser" + f"ver: {value}"
        assert scan.content_violations("cfg.yaml", field, []) == [], field


def test_environment_shaped_field_with_empty_value_is_flagged():
    assert scan.content_violations("cfg.yaml", "ser" + "ver:", [])


def test_prefixed_keys_are_detected():
    field = "DB_HO" + "ST=prod01"
    assert scan.content_violations(".env.example", field, [])


def test_literal_ip_addresses_are_flagged():
    ip = ".".join(["10", "11", "12", "13"])
    assert scan.content_violations("doc.md", f"connect to {ip}", [])
    assert scan.content_violations("doc.md", "bind to 127.0.0.1 or 0.0.0.0", []) == []


def test_env_markers_extend_the_scan_case_insensitively():
    text = "see the wiki page about realhost42 for details"
    assert scan.content_violations("doc.md", text, []) == []
    assert scan.content_violations("doc.md", text, ["RealHost42"])


def test_markers_parsed_from_env():
    env = {"METRICPROBE_LEAK_MARKERS": "alpha, beta ,,"}
    assert scan.markers_from_env(env) == ["alpha", "beta"]
    assert scan.markers_from_env({}) == []


def test_this_repo_is_clean():
    assert scan.scan_repo(REPO_ROOT, scan.markers_from_env()) == []
