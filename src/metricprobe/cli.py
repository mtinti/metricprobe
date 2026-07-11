"""Command-line interface: discover | run | report | publish | serve.

Exit codes separate code failure from data failure: 0 = ran with no RED,
2 = ran with at least one data-health RED (outputs committed first),
1 = execution error (nothing partial becomes visible).
"""

import sys

from metricprobe import __version__


def main() -> int:
    """Placeholder entry point; subcommands land with their features (Step 6)."""
    print(f"metricprobe {__version__} — no commands implemented yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
