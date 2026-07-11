"""Dual lag (ALGORITHMS.md section 14): source-side completion curves —
reusing the section-2 formulas verbatim — plus the per-row delta histogram
separating upstream provider lag from local ingestion lag.

Rows with NULL source_insert_time form their own reported bucket; the dual
reconciliation equation must hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.dual import DualLagResult
from metricprobe.metrics.completion import (
    PERCENTILES,
    Percentile,
    curves_from_cells,
    days_to_percentile,
)
from metricprobe.status import Check, ReasonCode, Severity, Status


@dataclass
class DualLagAssessment:
    source_percentiles: dict[pd.Period, dict[int, Percentile]]
    delta_histogram: pd.Series  # index: delta_day -> row count
    n_null_source_only: int
    n_delta_rows: int
    negative_lag_excess_fraction: float
    statuses: list[Status] = field(default_factory=list)


def assess_dual_lag(
    dual: DualLagResult, table: TableConfig, as_of: pd.Timestamp
) -> DualLagAssessment:
    analysis = table.analysis
    curves = curves_from_cells(dual.rows_for("month_src_lag"), analysis.lag_cap_days)
    percentiles = {
        month: {pct: days_to_percentile(curve, pct) for pct in PERCENTILES}
        for month, curve in sorted(curves.items())
    }
    delta_cells = dual.rows_for("delta")
    histogram = (
        delta_cells.set_index(delta_cells["delta_day"].astype(int))["row_count"].sort_index()
        if len(delta_cells)
        else pd.Series(dtype=int)
    )
    statuses: list[Status] = []

    g = dual.global_row
    bucket_sum = int(
        g["n_source_eligible"]
        + g["n_null_event_time"]
        + g["n_null_source_only"]
        + g["n_negative_lag_excluded"]
    )
    if bucket_sum != int(g["row_count"]):
        statuses.append(
            Status(
                check=Check.RECONCILIATION,
                severity=Severity.RED,
                reason=ReasonCode.RECONCILIATION_MISMATCH,
                detail=f"dual pass: total_rows={int(g['row_count'])} != bucket "
                f"sum={bucket_sum}",
            )
        )
    rows_with_both = int(g["row_count"] - g["n_null_event_time"] - g["n_null_source_only"])
    excess = int(g["n_negative_lag_excluded"]) / rows_with_both if rows_with_both else 0.0
    if excess > analysis.negative_lag_red_fraction:
        statuses.append(
            Status(
                check=Check.DUAL_LAG,
                severity=Severity.RED,
                reason=ReasonCode.NEGATIVE_LAG_EXCESS,
                detail=f"source-lag negative excess {excess:.4%} exceeds "
                f"{analysis.negative_lag_red_fraction:.4%}",
            )
        )
    if not statuses:
        statuses.append(Status(check=Check.DUAL_LAG, severity=Severity.GREEN))
    return DualLagAssessment(
        source_percentiles=percentiles,
        delta_histogram=histogram,
        n_null_source_only=int(g["n_null_source_only"]),
        n_delta_rows=int(g["n_delta_rows"]),
        negative_lag_excess_fraction=excess,
        statuses=statuses,
    )
