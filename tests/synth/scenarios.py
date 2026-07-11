"""Named pathology scenarios in healthy/unhealthy twin pairs (PLAN Step 1).

Each pair shares the same generating parameters except the injected pathology,
so every check is tested in both directions: it must fire on the unhealthy twin
AND stay silent on the healthy twin. `expected_detection` names the verdict the
pipeline is required to produce once the corresponding metric exists.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from tests.synth import generator as g

TRICKLE_BASE = g.TableSpec(
    name="trickle_orders",
    start_month="2023-01",
    n_months=30,
    rows_per_month=2000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8),  # median ~5 days, p99 ~32 days
    seed=101,
)

BATCHY_BASE = g.TableSpec(
    name="batchy_settlements",
    start_month="2023-01",
    n_months=30,
    rows_per_month=2000,
    lag_model=g.StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
    seed=202,
)

DUAL_BASE = dataclasses.replace(
    TRICKLE_BASE, name="dual_registry", dual_offset_days=2.0, seed=303
)


@dataclass(frozen=True)
class ScenarioPair:
    name: str
    description: str
    expected_detection: str
    healthy: Callable[[], pd.DataFrame]
    unhealthy: Callable[[], pd.DataFrame]


def catalog() -> dict[str, ScenarioPair]:
    """All named pathology pairs. Month indexes target MATURE months (well before
    the end of history) unless the pathology is specifically about recency."""
    spike_month = 18  # 2024-07: mature under default horizons
    drop_month = 15  # 2024-04
    gap_month = 12  # 2024-01
    pairs = [
        ScenarioPair(
            name="volume_spike",
            description="one mature month at 6x its normal volume",
            expected_detection="volume: RED/AMBER on the spiked month (median±MAD outlier)",
            healthy=lambda: g.generate(TRICKLE_BASE),
            unhealthy=lambda: g.generate(g.volume_spike(TRICKLE_BASE, spike_month, factor=6.0)),
        ),
        ScenarioPair(
            name="volume_drop",
            description="one mature month at ~10% of its normal volume",
            expected_detection="volume: RED/AMBER on the dropped month (median±MAD outlier)",
            healthy=lambda: g.generate(TRICKLE_BASE),
            unhealthy=lambda: g.generate(g.volume_drop(TRICKLE_BASE, drop_month, factor=0.1)),
        ),
        ScenarioPair(
            name="missing_month",
            description="one interior month entirely absent",
            expected_detection="volume: RED, month rendered as an explicit gap",
            healthy=lambda: g.generate(TRICKLE_BASE),
            unhealthy=lambda: g.generate(g.missing_month(TRICKLE_BASE, gap_month)),
        ),
        ScenarioPair(
            name="duplicate_keys",
            description="2% of rows re-loaded with the same key an hour later",
            expected_detection="uniqueness check RED when key_cols configured",
            healthy=lambda: g.generate(TRICKLE_BASE),
            unhealthy=lambda: g.inject_duplicate_keys(
                g.generate(TRICKLE_BASE), fraction=0.02, seed=7
            ),
        ),
        ScenarioPair(
            name="straggler_batch",
            description="15% of one month's rows arrive in an extra batch 45 days late",
            expected_detection="completion p95/p99 shifted; batch metrics show the extra run",
            healthy=lambda: g.generate(BATCHY_BASE),
            unhealthy=lambda: g.inject_straggler_batch(
                g.generate(BATCHY_BASE), month="2024-11", late_day=45.0, fraction=0.15, seed=7
            ),
        ),
        ScenarioPair(
            name="raw_vs_corrected",
            description="8% of rows carry a raw event date 35 days off the corrected one",
            expected_detection="compare_event_time side-stat counts the mismatching rows",
            healthy=lambda: g.inject_raw_vs_corrected(
                g.generate(TRICKLE_BASE), fraction=0.0, shift_days=-35, seed=7
            ),
            unhealthy=lambda: g.inject_raw_vs_corrected(
                g.generate(TRICKLE_BASE), fraction=0.08, shift_days=-35, seed=7
            ),
        ),
        ScenarioPair(
            name="sustained_collapse",
            description="batches keep their cadence but the last 3 months carry ~10x fewer rows",
            expected_detection="volume: RED on mature collapsed months with freshness: GREEN",
            healthy=lambda: g.generate(BATCHY_BASE),
            unhealthy=lambda: g.generate(
                g.sustained_collapse(BATCHY_BASE, last_k=3, factor=0.1)
            ),
        ),
    ]
    return {pair.name: pair for pair in pairs}
