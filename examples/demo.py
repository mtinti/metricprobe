"""The public demo: a fixed-seed synthetic world -> probe -> dashboard.

Builds FOUR fake databases in different domains (deliberately not
single-industry — the tool is generic) with healthy AND unhealthy twins, runs
the full metricprobe campaign over them, and publishes the markdown dashboard
that is committed under reports/ in this repository:

    demo_retail    orders feed (lognormal trickle, dual timestamps: order
                   placed vs warehouse-loaded); returns — orders' TWIN (same
                   generating parameters + seed) with a missing month; a
                   shipments pair whose parity prerequisite fails
                   (read_uncommitted -> indeterminate); and an optional
                   decommissioned feed that no longer exists (skipped)
    demo_sensors   IoT telemetry (near-real-time with straggler devices);
                   device_pings — telemetry's twin that silently STOPPED
                   (freshness red); gateway_logs — telemetry's twin with one
                   mature month mildly spiked (amber); and a feed too young
                   to classify (insufficient history)
    demo_finance   card settlements (monthly step batches with a batch-run
                   id, healthy) and card_disputes — settlements' twin in
                   sustained volume collapse (updating green / volume red)
    demo_health    episode records (slow trickle over weeks with a long
                   tail, dual timestamps) and a registry-like table whose
                   UPSTREAM lag dominates the local ingestion lag

Unhealthy tables are parameter-matched TWINS of their healthy siblings (same
generating parameters and seed, differing only in the injected pathology), so
every verdict fires against a like-for-like control. The dashboard exercises
the FULL vocabulary: green, amber, red (missing month / stale feed /
collapse), indeterminate, insufficient history, skipped, and the censored
p95 ("> cap", via a lag-capped variant probe on the episodes feed).

`serve` (the Streamlit app) is deliberately NOT part of this demo: Step 10
was skipped by the plan owner; the self-contained report.html is the
interactive artifact.

The build is BYTE-DETERMINISTIC: data comes from fixed seeds, the clock /
run id / git metadata are injected, SVG output is canonicalized, and the
renderer is kaleido's pinned Chrome build (downloaded on first use), so a CI
job regenerates the dashboard and diffs it against the committed one.
Synthetic data ONLY — nothing environment-specific exists in this world.

Usage:
    python examples/demo.py --out reports [--work /tmp/mp-demo]

Run from a repository checkout (the synthetic generator lives in tests/).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import tempfile
from pathlib import Path

import duckdb
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.synth import generator as g  # noqa: E402

from metricprobe.cli import main as metricprobe_main  # noqa: E402
from metricprobe.config import CONFIG_SCHEMA_VERSION  # noqa: E402

# ---- the frozen identity of the demo run (byte-stability contract)
AS_OF = "2025-07-24"
RUN_AT = "2025-07-24T06:00:00+00:00"
RUN_ID = "demo-0001"
GIT_SHA = "demo0000demo0000"

# months are calendar-aligned: histories start 2023-01; feeds that should
# read FRESH at the as_of include the open month 2025-07
FULL = 31  # 2023-01 .. 2025-07
BATCHY = 30  # 2023-01 .. 2025-06 (June's last batch loads 2025-07-21)

ORDERS = g.TableSpec(
    name="orders", start_month="2023-01", n_months=FULL, rows_per_month=1500,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8), dual_offset_days=1.0, seed=1101,
)
SHIPMENTS = g.TableSpec(
    name="shipments", start_month="2023-01", n_months=FULL, rows_per_month=1000,
    lag_model=g.LognormalLag(mu=1.4, sigma=0.7), seed=1103,
)
TELEMETRY = g.TableSpec(
    name="telemetry", start_month="2023-01", n_months=FULL, rows_per_month=3000,
    # near-real-time with straggler devices: hours-scale median, wide sigma
    # gives the long device tail
    lag_model=g.LognormalLag(mu=-1.0, sigma=1.2), seed=1201,
)
SETTLEMENTS = g.TableSpec(
    name="settlements", start_month="2023-01", n_months=BATCHY, rows_per_month=2000,
    lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
    seed=1301,
)

WORLD: dict[str, list[tuple[g.TableSpec, dict]]] = {
    "demo_retail": [
        (ORDERS, {"source_insert_time": "source_insert_time", "key_cols": ["row_id"]}),
        (
            # orders' parameter-matched TWIN: identical spec and seed, one
            # mature month (2023-10) simply absent
            g.missing_month(dataclasses.replace(ORDERS, name="returns"), 9),
            {"source_insert_time": "source_insert_time"},
        ),
        (SHIPMENTS, {"key_cols": ["row_id"],
                     "parity_with": "shipments_replica",
                     "read_uncommitted": True}),
        (dataclasses.replace(SHIPMENTS, name="shipments_replica"),
         {"key_cols": ["row_id"]}),
    ],
    "demo_sensors": [
        (TELEMETRY, {}),
        (
            # telemetry's twin that silently STOPPED five months before as_of
            dataclasses.replace(TELEMETRY, name="device_pings", n_months=26),
            {},
        ),
        (
            # telemetry's twin with ONE mature month mildly spiked: the
            # constant baseline's sigma floor is 5% of the median, so a 12%
            # spike sits between 2 and 3 robust deviations -> AMBER
            g.volume_spike(
                dataclasses.replace(TELEMETRY, name="gateway_logs"), 12, factor=1.12
            ),
            {},
        ),
        (
            g.TableSpec(
                name="new_feed", start_month="2025-03", n_months=5,
                rows_per_month=800, lag_model=g.LognormalLag(mu=-0.5, sigma=0.8),
                seed=1203,  # too young for any maturity classification
            ),
            {},
        ),
    ],
    "demo_finance": [
        (SETTLEMENTS, {"load_batch_col": "batch_id", "expect_batchy": True,
                       "key_cols": ["row_id"]}),
        (
            # settlements' twin: batches keep their cadence, rows collapse ~10x
            g.sustained_collapse(
                dataclasses.replace(SETTLEMENTS, name="card_disputes"),
                last_k=15, factor=0.1,
            ),
            {"load_batch_col": "batch_id"},
        ),
    ],
    "demo_health": [
        (
            g.TableSpec(
                name="episodes", start_month="2023-01", n_months=FULL,
                rows_per_month=1200,
                # slow trickle over weeks with a long tail
                lag_model=g.LognormalLag(mu=2.2, sigma=0.9),
                dual_offset_days=1.0, seed=1401,
            ),
            {"source_insert_time": "source_insert_time"},
        ),
        (
            g.TableSpec(
                name="registry", start_month="2023-01", n_months=FULL,
                rows_per_month=900,
                # the registry-like case: UPSTREAM lag dominates (median ~12
                # days to the source system, half a day source -> local)
                lag_model=g.LognormalLag(mu=2.5, sigma=0.7),
                dual_offset_days=0.5, seed=1402,
            ),
            {"source_insert_time": "source_insert_time"},
        ),
    ],
}

# extra VARIANT probes on existing tables (variants are first-class): a
# lag-capped view of the episodes feed whose true p95 (~40 days for
# mu=2.2/sigma=0.9) exceeds the 15-day cap — overflow mass ~29% > 5%, so the
# dashboard's p95 columns render the censored "> cap" state, never a precise
# number computed from a truncated curve
VARIANTS: dict[str, list[dict]] = {
    "demo_health": [
        {
            "probe_name": "episodes_capped",
            "database": "demo_health",
            "schema": "main",
            "table": "episodes",
            "event_time": "event_time",
            "load_time": "load_time",
            "resolution": {"event_time": "datetime", "load_time": "datetime"},
            "analysis": {"lag_cap_days": 15, "training_cutoff_days": 365},
        }
    ],
}

# probes whose TABLE deliberately does not exist (optional -> skipped ➖)
ABSENT: dict[str, list[dict]] = {
    "demo_retail": [
        {
            "probe_name": "decommissioned_feed",
            "database": "demo_retail",
            "schema": "main",
            "table": "decommissioned_feed",
            "event_time": "event_time",
            "load_time": "load_time",
            "resolution": {"event_time": "datetime", "load_time": "datetime"},
            "optional": True,
        }
    ],
}


def _entry(database: str, spec: g.TableSpec, overrides: dict) -> dict:
    resolution = {"event_time": "datetime", "load_time": "datetime"}
    if overrides.get("source_insert_time"):
        resolution[overrides["source_insert_time"]] = "datetime"
    return {
        "probe_name": f"{spec.name}",
        "database": database,
        "schema": "main",
        "table": spec.name,
        "event_time": "event_time",
        "load_time": "load_time",
        "resolution": resolution,
    } | overrides


def build_world(work: Path) -> list[str]:
    """Generate the four databases and one config file per database
    (metricprobe composes repeatable --config files into ONE campaign)."""
    config_paths = []
    for database, tables in WORLD.items():
        db_path = work / f"{database}.duckdb"
        if db_path.exists():
            db_path.unlink()
        con = duckdb.connect(str(db_path))
        entries = []
        for spec, overrides in tables:
            g.load_into_duckdb(g.generate(spec), con, spec.name)
            entries.append(_entry(database, spec, overrides))
        entries.extend(VARIANTS.get(database, ()))
        entries.extend(ABSENT.get(database, ()))
        con.close()
        config_path = work / f"{database}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": CONFIG_SCHEMA_VERSION,
                    "connection_url": f"duckdb:///{db_path}",
                    "store": {"path": str(work / "store")},
                    "campaign": {"schedule": "0 6 * * 1", "timezone": "UTC"},
                    "tables": entries,
                }
            )
        )
        config_paths.append(str(config_path))
    return config_paths


def pin_renderer() -> None:
    """Render with kaleido's PINNED Chrome build (downloaded to the user
    profile on first use): choreographer prefers it over any system browser,
    so the SVG bytes do not depend on whatever Chrome the machine ships."""
    import kaleido

    kaleido.get_chrome_sync()


def run_demo(out_dir: Path, work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    pin_renderer()
    configs = build_world(work)
    os.environ["METRICPROBE_RUN_AT"] = RUN_AT
    os.environ["METRICPROBE_GIT_SHA"] = GIT_SHA
    config_args: list[str] = []
    for path in configs:
        config_args += ["--config", path]
    code = metricprobe_main(
        ["run", *config_args, "--as-of", AS_OF, "--run-id", RUN_ID]
    )
    if code == 1:
        return code  # 2 = data-health red: expected, the demo SHOWS pathologies
    return metricprobe_main(
        ["publish", *config_args, "--run-id", RUN_ID, "--out", str(out_dir)]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports", help="dashboard output directory")
    parser.add_argument(
        "--work", default=None,
        help="working directory for the synthetic databases and snapshot "
        "store (default: a temporary directory)",
    )
    args = parser.parse_args()
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="mp-demo-"))
    return run_demo(Path(args.out), work)


if __name__ == "__main__":
    raise SystemExit(main())
