"""Freshness: learned cadence from DISTINCT ARRIVAL EPOCHS, never per-row gaps
(row-level timestamps measure row frequency, not feed cadence) — ALGORITHMS.md
section 11.

Epochs are distinct batch IDs when load_batch_col is configured (each batch's
canonical timestamp = MIN(load_time) within the batch, the minimum over its
month cells), else distinct load-time buckets. The staleness core lands here in
Step 4 because the sustained-collapse acceptance case requires volume=RED with
freshness=GREEN simultaneously; the full freshness golden suite (stopped feed,
bulk loads, minimum epochs, zero-MAD) is Step 5.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult
from metricprobe.metrics.robust import robust_sigma
from metricprobe.status import Check, ReasonCode, Severity, Status

# v1 algorithm constants, mirroring the volume amber/red defaults
FRESHNESS_AMBER_MADS = 2.0
FRESHNESS_RED_MADS = 3.0


@dataclass
class FreshnessAssessment:
    epoch_count: int
    last_epoch: pd.Timestamp | None
    cadence_median_days: float | None
    cadence_sigma_days: float | None
    days_since_last: float | None
    statuses: list[Status] = field(default_factory=list)


def epoch_timestamps(canonical: CanonicalResult, table: TableConfig) -> list[pd.Timestamp]:
    """Distinct arrival epochs: batch canonical timestamps when configured,
    else load-time day/hour buckets."""
    if table.load_batch_col:
        cells = canonical.rows_for("month_batch")
        # a batch can span cohorts: its canonical timestamp is the MIN over
        # its per-month cells' MIN(load_time)
        epochs = cells.groupby("batch_id")["min_load_time"].min()
        return sorted(pd.to_datetime(stamp) for stamp in epochs)
    cells = canonical.rows_for("epoch")
    return sorted(pd.to_datetime(stamp) for stamp in cells["load_epoch_day"])


def assess_freshness(
    canonical: CanonicalResult, table: TableConfig, as_of: pd.Timestamp
) -> FreshnessAssessment:
    analysis = table.analysis
    stamps = epoch_timestamps(canonical, table)
    if len(stamps) < analysis.freshness_min_epochs:
        return FreshnessAssessment(
            epoch_count=len(stamps),
            last_epoch=stamps[-1] if stamps else None,
            cadence_median_days=None,
            cadence_sigma_days=None,
            days_since_last=None,
            statuses=[
                Status(
                    check=Check.FRESHNESS,
                    severity=Severity.INSUFFICIENT_HISTORY,
                    reason=ReasonCode.INSUFFICIENT_EPOCHS,
                    detail=f"{len(stamps)} arrival epochs < freshness_min_epochs="
                    f"{analysis.freshness_min_epochs}",
                )
            ],
        )
    gaps = [
        (later - earlier) / pd.Timedelta(days=1)
        for earlier, later in zip(stamps, stamps[1:], strict=False)
    ]
    cadence = float(statistics.median(gaps))
    # zero-MAD fallback: perfectly regular feeds use the configured fixed tolerance
    sigma = max(robust_sigma(gaps), analysis.freshness_zero_mad_tolerance_days)
    days_since = float((as_of - stamps[-1]) / pd.Timedelta(days=1))
    if days_since > cadence + FRESHNESS_RED_MADS * sigma:
        severity = Severity.RED
    elif days_since > cadence + FRESHNESS_AMBER_MADS * sigma:
        severity = Severity.AMBER
    else:
        severity = Severity.GREEN
    if severity is Severity.GREEN:
        statuses = [Status(check=Check.FRESHNESS, severity=Severity.GREEN)]
    else:
        statuses = [
            Status(
                check=Check.FRESHNESS,
                severity=severity,
                reason=ReasonCode.STALE_FEED,
                detail=f"{days_since:.1f} days since the last arrival epoch vs "
                f"learned cadence {cadence:.1f}d (robust sigma {sigma:.1f}d)",
            )
        ]
    return FreshnessAssessment(
        epoch_count=len(stamps),
        last_epoch=stamps[-1],
        cadence_median_days=cadence,
        cadence_sigma_days=sigma,
        days_since_last=days_since,
        statuses=statuses,
    )
