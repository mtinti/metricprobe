"""CI guard: the committed demo dashboard (reports/) must match what
examples/demo.py generates TODAY.

Regenerates the dashboard in place and fails when `git diff` shows drift —
byte-stability holds because the demo's clock/run id/git metadata are frozen,
the data seeds are fixed, SVG uids are canonicalized, the plotly/kaleido
versions are pinned exactly, and the renderer is kaleido's pinned Chrome
build with the Liberation Sans font (fonts-liberation), so regeneration on
the pinned Linux CI image is reproducible.

The committed artifacts are themselves BUILT on that same image family
(see reports/README: regenerate via examples/demo.py inside linux/amd64).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    build = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "demo.py"), "--out",
         str(REPO_ROOT / "reports")],
        cwd=REPO_ROOT,
    )
    if build.returncode not in (0, 2):  # 2 = data-health RED: the demo SHOWS reds
        print(f"demo build failed with exit {build.returncode}", file=sys.stderr)
        return 1
    diff = subprocess.run(
        ["git", "diff", "--exit-code", "--stat", "--", "reports"], cwd=REPO_ROOT
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "reports"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.strip()
    if diff.returncode != 0 or untracked:
        if untracked:
            print(f"untracked demo outputs:\n{untracked}", file=sys.stderr)
        print(
            "\nthe committed demo dashboard is stale: regenerate it with\n"
            "  python examples/demo.py --out reports\n"
            "on linux/amd64 (the committed bytes are produced there) and "
            "commit the result",
            file=sys.stderr,
        )
        return 1
    print("committed demo dashboard is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
