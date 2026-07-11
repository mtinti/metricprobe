"""Fail fast when the equivalence runner is not Linux x86-64.

The official SQL Server container images exist for Linux x86-64 only (the sole
platform Microsoft supports). GitHub-hosted `ubuntu-24.04` resolves to a Linux
x86-64 VM, but on Gitea `runs-on` labels are administrator-defined, so the
platform must be asserted executably here — a mislabeled runner fails with this
clear message instead of an obscure container error.
"""

from __future__ import annotations

import platform
import sys

SUPPORTED_MACHINES = {"x86_64", "amd64"}


def check(system: str, machine: str) -> str | None:
    """Return an error message when (system, machine) cannot run the official
    SQL Server container, else None."""
    if system.lower() != "linux":
        return f"the equivalence job requires Linux, this runner is {system}"
    if machine.lower() not in SUPPORTED_MACHINES:
        return f"SQL Server containers support only x86-64, this runner is {machine}"
    return None


def main() -> int:
    error = check(platform.system(), platform.machine())
    if error:
        print(f"Unsupported equivalence runner: {error}", file=sys.stderr)
        return 1
    print(f"Equivalence runner OK: {platform.system()} {platform.machine()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
