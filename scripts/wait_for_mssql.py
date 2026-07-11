"""Block until the SQL Server container accepts connections, then exit 0.

The official mssql image has no Docker HEALTHCHECK, so CI cannot rely on the
service being ready when steps start. Polls METRICPROBE_MSSQL_URL with SELECT 1
until success or METRICPROBE_MSSQL_WAIT_SECONDS (default 180) elapses.
"""

from __future__ import annotations

import os
import sys
import time

import sqlalchemy


def main() -> int:
    url = os.environ["METRICPROBE_MSSQL_URL"]
    deadline = time.monotonic() + float(os.environ.get("METRICPROBE_MSSQL_WAIT_SECONDS", "180"))
    engine = sqlalchemy.create_engine(url)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            print("SQL Server is ready.")
            return 0
        except Exception as exc:  # driver raises various OperationalError subtypes
            last_error = exc
            time.sleep(2)
    print(f"SQL Server never became ready: {last_error}", file=sys.stderr)
    print(
        "hint: if this runner is not Linux x86-64, the SQL Server container "
        "cannot run (see scripts/assert_runner_arch.py)",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
