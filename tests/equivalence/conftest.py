"""Equivalence harness fixtures (built in Step 0, used by every metric after).

DuckDB is the fast development vehicle; the SQL Server container job is the
truth-teller for the T-SQL dialect production actually runs. Everything in this
directory is auto-marked `equivalence` and skips without METRICPROBE_MSSQL_URL.
"""

import os
from pathlib import Path

import pytest
import sqlalchemy

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if _HERE in Path(item.fspath).parents or Path(item.fspath).parent == _HERE:
            item.add_marker(pytest.mark.equivalence)


@pytest.fixture()
def duckdb_engine():
    engine = sqlalchemy.create_engine("duckdb:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture()
def mssql_engine():
    url = os.environ.get("METRICPROBE_MSSQL_URL")
    if not url:
        pytest.skip("METRICPROBE_MSSQL_URL not set; SQL Server equivalence runs in CI")
    engine = sqlalchemy.create_engine(url)
    yield engine
    engine.dispose()
