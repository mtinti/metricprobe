"""CI guard: fail when environment-specific or private material is tracked by git.

This repo is public and must contain zero environment-specific details. .gitignore
alone is insufficient (a file can be force-added), so this scan walks every
git-TRACKED file and fails on:

  * forbidden file names (CLAUDE.private.md and any *.private.* file);
  * environment-shaped FIELDS regardless of value source — a line assigning any of
    the keys in _ENV_FIELD (server / hostname / database / schema / password ...)
    must carry a placeholder-shaped value or an env-var reference, else it fails;
    an empty inline value also fails (it cannot be verified as a placeholder);
  * connection-URL hosts that are not obvious placeholders (localhost, SERVER, ...);
  * literal IPv4 addresses other than loopback/unspecified;
  * UNC paths (double-backslash Windows shares), which are always environment-specific;
  * extra literal markers (case-insensitive) via METRICPROBE_LEAK_MARKERS
    (comma-separated). The marker list itself would leak the private names it
    protects, so it is never committed here — CI injects it from a repo secret
    (see .github/workflows/ci.yml) and a private mirror sets it in its environment.

The scan cannot prove the absence of private names it has never been told about;
the structural checks exist so that anything environment-SHAPED fails closed even
when no marker list is supplied.

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

# Environment-shaped field assignment: `<key>: value` / `<key>=value` in YAML, INI,
# code or connection strings. The lookbehind permits prefixed keys (DB_HOST=...)
# but not path segments (a Docker image ref like mssql/server:tag is not a field).
_ENV_FIELD = re.compile(
    r"""(?<![A-Za-z0-9/])
        (server|hostname|host|address|data[ _-]?source|database|dbname|db_name
         |catalog|initial[ _]catalog|schema|password|passwd|pwd)
        \s*[:=]\s*["']?([^\s;,"']*)""",
    re.IGNORECASE | re.VERBOSE,
)

# Values allowed for environment-shaped fields: env-var references, template/angle
# placeholders, loopback, generic role words, and clearly-synthetic stems.
_PLACEHOLDER_VALUE = re.compile(
    r"""^(?:
        \$.*                                  # $VAR, ${VAR}, ${{ secrets.X }}
      | <[^>]*>?                              # <your-server>
      | %[^%]+%                               # %SERVER%
      | \{.*                                  # {templated}
      | localhost(?::\d+)?(?:/.*)?
      | 127\.0\.0\.1(?::\d+)?(?:/.*)?
      | :memory:
      | (?:your|example|demo|sample|synth|fake|mock|dummy|placeholder|metricprobe|test)[\w.!-]*
      | server | host | hostname | database | db | dbname | catalog
      | dbo | main | public | master | tempdb | information_schema
      | none | null | true | false | yes | no
      | str | int | float | bool | bytes | date | datetime   # type annotations
    )$""",
    re.IGNORECASE | re.VERBOSE,
)

_IPV4 = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b")
_ALLOWED_IPS = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}


def path_violation(path: str) -> str | None:
    """Return a reason string when the tracked path itself is forbidden."""
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if pattern.search(path):
            return f"forbidden private file name (pattern {pattern.pattern!r})"
    return None


def content_violations(path: str, text: str, extra_markers: list[str]) -> list[str]:
    """Return 'path:line: reason' strings for environment-shaped content."""
    violations: list[str] = []
    lowered_markers = [m.lower() for m in extra_markers if m]
    for lineno, line in enumerate(text.splitlines(), start=1):
        for match in _URL_HOST.finditer(line):
            host = match.group(1)
            if host.lower() not in PLACEHOLDER_HOSTS:
                violations.append(
                    f"{path}:{lineno}: connection URL host {host!r} is not a known placeholder"
                )
        for match in _ENV_FIELD.finditer(line):
            key, value = match.group(1), match.group(2)
            if "(" in value or ")" in value or "[" in value:
                continue  # code expression / annotation, not a literal value
            if not value:
                violations.append(
                    f"{path}:{lineno}: environment-shaped field {key!r} has no inline value"
                    " (use a placeholder or env-var reference)"
                )
            elif not _PLACEHOLDER_VALUE.match(value):
                violations.append(
                    f"{path}:{lineno}: environment-shaped field {key!r} carries"
                    f" non-placeholder value {value!r}"
                )
        for match in _IPV4.finditer(line):
            ip = match.group(1)
            if ip not in _ALLOWED_IPS and all(int(octet) <= 255 for octet in ip.split(".")):
                violations.append(f"{path}:{lineno}: literal IPv4 address {ip}")
        if _UNC_PATH.search(line):
            violations.append(f"{path}:{lineno}: UNC path (environment-specific)")
        lowered_line = line.lower()
        for marker in lowered_markers:
            if marker in lowered_line:
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
