"""Batch metrics (ALGORITHMS.md section 12): rows per run, runs per month, and
batch-level completion — cumulative fractions WEIGHTED BY BATCH ROW COUNTS,
with days measured from month end at the batch's canonical timestamp
(MIN(load_time) within the batch, the minimum over its month cells)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult
from metricprobe.metrics.completion import PERCENTILES, month_end
from metricprobe.status import Check, Severity, Status


@dataclass
class BatchMonth:
    month: pd.Period
    runs: int
    rows_per_run: dict[str, int]  # batch_id -> rows in this month
    days_to: dict[int, int | None]  # percentile -> days from month end


@dataclass
class BatchAssessment:
    months: list[BatchMonth]
    rows_per_run: dict[str, int]  # batch_id -> total rows across months
    statuses: list[Status] = field(default_factory=list)


def assess_batch(canonical: CanonicalResult, table: TableConfig) -> BatchAssessment:
    if not table.load_batch_col:
        raise ValueError(f"probe {table.probe_name!r} has no load_batch_col configured")
    cells = canonical.rows_for("month_batch")
    # canonical batch timestamp: MIN over the batch's month cells (spans cohorts)
    batch_ts = {
        batch: pd.to_datetime(stamp)
        for batch, stamp in cells.groupby("batch_id")["min_load_time"].min().items()
    }
    rows_per_run = {
        batch: int(rows) for batch, rows in cells.groupby("batch_id")["row_count"].sum().items()
    }
    months: list[BatchMonth] = []
    for month_ts, group in sorted(cells.groupby("event_month"), key=lambda item: item[0]):
        month = pd.Period(month_ts, freq="M")
        per_batch = {
            batch: int(rows) for batch, rows in group.groupby("batch_id")["row_count"].sum().items()
        }
        total = sum(per_batch.values())
        # order by canonical timestamp; cumulative fraction weighted by rows
        ordered = sorted(per_batch, key=lambda batch: batch_ts[batch])
        days_to: dict[int, int | None] = {}
        for pct in PERCENTILES:
            target = pct / 100
            cumulative = 0
            days_to[pct] = None
            for batch in ordered:
                cumulative += per_batch[batch]
                if cumulative / total >= target:
                    days_to[pct] = int(
                        (batch_ts[batch].normalize() - month_end(month).normalize())
                        / pd.Timedelta(days=1)
                    )
                    break
        months.append(
            BatchMonth(month=month, runs=len(per_batch), rows_per_run=per_batch, days_to=days_to)
        )
    return BatchAssessment(
        months=months,
        rows_per_run=rows_per_run,
        statuses=[Status(check=Check.BATCH, severity=Severity.GREEN)],
    )
