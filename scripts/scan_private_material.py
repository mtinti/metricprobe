"""CI guard: fail when environment-specific or private material is tracked by git.

This repo is public and must contain zero environment-specific details. .gitignore
alone is insufficient (a file can be force-added), so this scan walks every
git-TRACKED file and fails on:

  * forbidden file names (CLAUDE.private.md and any *.private.* file);
  * connection-URL hosts that are not obvious placeholders (localhost, SERVER, ...);
  * UNC paths (double-backslash Windows shares), which are always environment-specific;
  * extra literal markers supplied via METRICPROBE_LEAK_MARKERS (comma-separated).
    The marker list itself would leak the private names it protects, so it is never
    committed here — a private mirror injects it through the environment.

Exit 0 = clean, exit 1 = violations (each one printed as path:line: reason).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"(^|/)CLAUDE\.private\.md$", re.IGNORECASE),
    re.compile(r"\.private\.", re.IGNORECASE),
]

# Hosts that are clearly placeholders or loopback, allowed inside connection URLs.
PLACEHOLDER_HOSTS = {
    "server",
    "your-server",
    "your_server",
    "example",
    "example.com",
    "host",
    "hostname",
    "placeholder",
    "localhost",
    "127.0.0.1",
}

# host portion of an SQLAlchemy-style URL: scheme+driver://[user[:pw]@]HOST[:port]/...
_URL_HOST = re.compile(
    r"\b(?:mssql|duckdb|postgresql|mysql)(?:\+\w+)?://(?:[^@/\s]*@)?([A-Za-z0-9_.-]+)"
)
_UNC_PATH = re.compile(r"\\\\[A-Za-z0-9_.$-]{2,}\\")


def path_violation(path: str) -> str | None:
    """Return a reason string when the tracked path itself is forbidden."""
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern.search(path):
            return f"forbidden private file name (pattern {pattern.pattern!r})"
    return None


def content_violations(path: str, text: str, extra_markers: list[str]) -> list[str]:
    """Return 'path:line: reason' strings for environment-shaped content."""
    violations: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _URL_HOST.finditer(line):
            host = match.group(1)
            if host.lower() not in PLACEHOLDER_HOSTS:
                violations.append(
                    f"{path}:{lineno}: connection URL host {host!r} is not a known placeholder"
                )
        if _UNC_PATH.search(line):
            violations.append(f"{path}:{lineno}: UNC path (environment-specific)")
        for marker in extra_markers:
            if marker and marker in line:
                violations.append(f"{path}:{lineno}: private marker present")
    return violations


def markers_from_env(environ: dict[str, str] | None = None) -> list[str]:
    raw = (environ if environ is not None else dict(os.environ)).get(
        "METRICPROBE_LEAK_MARKERS", ""
    )
    return [m.strip() for m in raw.split(",") if m.strip()]


def tracked_files(repo_root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def scan_repo(repo_root: Path, extra_markers: list[str]) -> list[str]:
    violations: list[str] = []
    for rel_path in tracked_files(repo_root):
        reason = path_violation(rel_path)
        if reason is not None:
            violations.append(f"{rel_path}: {reason}")
        full = repo_root / rel_path
        try:
            text = full.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary or deleted-in-worktree; nothing textual to scan
        violations.extend(content_violations(rel_path, text, extra_markers))
    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    violations = scan_repo(repo_root, markers_from_env())
    if violations:
        print("Private-material scan FAILED:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print("Private-material scan passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
