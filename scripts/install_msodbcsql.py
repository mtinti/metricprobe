"""Install Microsoft ODBC Driver 18 on the CI runner (Ubuntu).

The equivalence job tests BOTH mssql drivers: pymssql (TDS, no system
driver) and pyodbc, which needs msodbcsql18 — the driver production
actually uses (a pymssql-only harness let a pyodbc-specific STATISTICS IO
loss reach production). Kept as a Python script per the house rule: CI
steps invoke Python with arguments; logic lives in files.

Run with sudo. Idempotent: exits 0 immediately if the driver is present.
"""

from __future__ import annotations

import pathlib
import subprocess


def sh(*argv: str, env: dict | None = None) -> None:
    subprocess.run(argv, check=True, env=env)


def main() -> int:
    if list(pathlib.Path("/opt/microsoft").glob("msodbcsql18/lib64/*.so*")):
        print("msodbcsql18 already installed")
        return 0
    import os
    import urllib.request

    codename = "noble"  # ubuntu-24.04 (the pinned CI image)
    urllib.request.urlretrieve(
        "https://packages.microsoft.com/keys/microsoft.asc", "/tmp/microsoft.asc"
    )
    sh("gpg", "--yes", "--dearmor", "-o",
       "/usr/share/keyrings/microsoft-prod.gpg", "/tmp/microsoft.asc")
    pathlib.Path("/etc/apt/sources.list.d/mssql-release.list").write_text(
        "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] "
        f"https://packages.microsoft.com/ubuntu/24.04/prod {codename} main\n"
    )
    env = dict(os.environ, ACCEPT_EULA="Y", DEBIAN_FRONTEND="noninteractive")
    sh("apt-get", "update", "-q", env=env)
    sh("apt-get", "install", "-yq", "msodbcsql18", "unixodbc-dev", env=env)
    print("msodbcsql18 installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
