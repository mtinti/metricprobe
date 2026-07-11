"""Batch metrics (ALGORITHMS.md section 12): rows per run, runs per month, and
batch-level completion — cumulative fractions WEIGHTED BY BATCH ROW COUNTS
against the month's FULL curve-eligible denominator, with days measured from
month end at the batch's canonical timestamp (MIN(load_time) within the batch,
the minimum over its month cells).

Rows with NULL batch IDs are counted, reported (AMBER NULL_BATCH_IDS) and kept
in the denominator: a curve over only the attributed rows would overstate
completion. Percentiles the attributed mass cannot reach stay None."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult
from metricprobe.metrics.completion import PERCENTILES, month_end
from metricprobe.status import Check, ReasonCode, Severity, Status


@dataclass
class BatchMonth:
    month: pd.Period
    runs: int
    rows_per_run: dict[str, int]  # batch_id -> eligible rows in this month
    null_batch_rows: int  # eligible rows with a NULL batch id
    days_to: dict[int, int | None]  # percentile -> days from month end


@dataclass
class BatchAssessment:
    months: list[BatchMonth]
    rows_per_run: dict[str, int]  # batch_id -> total eligible rows across months
    statuses: list[Status] = field(default_factory=list)


def assess_batch(canonical: CanonicalResult, table: TableConfig) -> BatchAssessment:
    if not table.load_batch_col:
        raise ValueError(f"probe {table.probe_name!r} has no load_batch_col configured")
    cells = canonical.rows_for("month_batch")
    valid = cells[cells["batch_id"].notna()]
    orphaned = cells[cells["batch_id"].isna()]
    # canonical batch timestamp: MIN over the batch's month cells (spans cohorts)
    batch_ts = {
        batch: pd.to_datetime(stamp)
        for batch, stamp in valid.groupby("batch_id")["min_load_time"].min().items()
    }
    rows_per_run = {
        batch: int(rows)
        for batch, rows in valid.groupby("batch_id")["n_curve_eligible"].sum().items()
    }
    # the month denominator: FULL curve-eligible count from the month/lag cells
    month_cells = canonical.rows_for("month_lag")
    month_cells = month_cells[month_cells["lag_day"] >= 0]
    finals = {
        pd.Period(ts, freq="M"): int(count)
        for ts, count in month_cells.groupby("event_month")["row_count"].sum().items()
    }
    orphaned_by_month = {
        pd.Period(ts, freq="M"): int(count)
        for ts, count in orphaned.groupby("event_month")["n_curve_eligible"].sum().items()
    }
    # iterate the UNION of months seen with and without batch ids: a month
    # whose eligible rows ALL carry NULL batch ids must still appear — with
    # zero runs, its NULL count, and every percentile unreachable — never
    # silently vanish from the results
    valid_by_month = {
        pd.Period(ts, freq="M"): group for ts, group in valid.groupby("event_month")
    }
    months: list[BatchMonth] = []
    for month in sorted(set(valid_by_month) | set(orphaned_by_month)):
        group = valid_by_month.get(month)
        per_batch = (
            {
                batch: int(rows)
                for batch, rows in group.groupby("batch_id")["n_curve_eligible"].sum().items()
            }
            if group is not None
            else {}
        )
        total = finals.get(month, sum(per_batch.values()))
        ordered = sorted(per_batch, key=lambda batch: batch_ts[batch])
        days_to: dict[int, int | None] = {}
        for pct in PERCENTILES:
            target = pct / 100
            cumulative = 0
            days_to[pct] = None  # unreachable when unattributed rows hold it back
            for batch in ordered:
                cumulative += per_batch[batch]
                if total and cumulative / total >= target:
                    days_to[pct] = int(
                        (batch_ts[batch].normalize() - month_end(month).normalize())
                        / pd.Timedelta(days=1)
                    )
                    break
        months.append(
            BatchMonth(
                month=month,
                runs=len(per_batch),
                rows_per_run=per_batch,
                null_batch_rows=orphaned_by_month.get(month, 0),
                days_to=days_to,
            )
        )
    total_orphaned = sum(orphaned_by_month.values())
    if total_orphaned > 0:
        statuses = [
            Status(
                check=Check.BATCH,
                severity=Severity.AMBER,
                reason=ReasonCode.NULL_BATCH_IDS,
                detail=f"{total_orphaned} curve-eligible rows carry a NULL "
                f"{table.load_batch_col!r}; they stay in the completion "
                "denominator but cannot be attributed to a run",
            )
        ]
    else:
        statuses = [Status(check=Check.BATCH, severity=Severity.GREEN)]
    return BatchAssessment(months=months, rows_per_run=rows_per_run, statuses=statuses)
