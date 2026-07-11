"""Completion curves, censoring-aware day-grain percentiles, and single-pass
exposure-based maturity — formulas per docs/ALGORITHMS.md sections 2-7.

Consumes the canonical aggregation (extract.canonical). Maturity is classified
by EXPOSURE only: a self-normalized curve always reaches 100% at its largest
observed lag, so curve shape carries no censoring signal.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult
from metricprobe.metrics.robust import median_of_curves
from metricprobe.metrics.robust import recommended_wait as _wait_formula
from metricprobe.status import Check, ReasonCode, Severity, Status

# v1 algorithm constants for the lag-support backtest (ALGORITHMS.md section 5)
BACKTEST_MIN_COHORT = 6
BACKTEST_FLOOR_DAYS = 7
BACKTEST_RATIO = 0.25

PERCENTILES = (50, 90, 95, 99)


@dataclass(frozen=True)
class Percentile:
    """A day-grain percentile; value is None when censored past the lag cap."""

    value: int | None
    over_cap: bool

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "> cap" if self.over_cap else str(self.value)


@dataclass
class MonthCurve:
    month: pd.Period
    counts: pd.Series  # index: lag_day (int, cap+1 = overflow bucket) -> count
    lag_cap: int

    @property
    def final_count(self) -> int:
        return int(self.counts.sum())

    @property
    def overflow_count(self) -> int:
        return int(self.counts.get(self.lag_cap + 1, 0))

    def fraction_at(self, day: int) -> float:
        """Cumulative fraction of the month's final count at lag <= day."""
        return float(self.counts[self.counts.index <= day].sum()) / self.final_count

    def cumulative(self, grid: np.ndarray) -> np.ndarray:
        ordered = self.counts.sort_index()
        cum = ordered.cumsum() / self.final_count
        return np.array([float(cum[cum.index <= d].iloc[-1]) if (cum.index <= d).any() else 0.0
                         for d in grid])


def build_month_curves(canonical: CanonicalResult, lag_cap: int) -> dict[pd.Period, MonthCurve]:
    rows = canonical.rows_for("month_lag")
    curves: dict[pd.Period, MonthCurve] = {}
    for month_ts, group in rows.groupby("event_month"):
        month = pd.Period(month_ts, freq="M")
        counts = group.set_index(group["lag_day"].astype(int))["row_count"].sort_index()
        curves[month] = MonthCurve(month=month, counts=counts, lag_cap=lag_cap)
    return curves


def days_to_percentile(curve: MonthCurve, pct: float) -> Percentile:
    """Smallest integer day d with F(d) >= pct/100; censoring-aware: if the
    curve cannot reach the percentile at the cap (overflow mass > 1-p), the
    answer is '> cap', never a number (ALGORITHMS.md section 2)."""
    target = pct / 100
    if curve.fraction_at(curve.lag_cap) < target:
        return Percentile(value=None, over_cap=True)
    ordered = curve.counts.sort_index()
    cum = ordered.cumsum() / curve.final_count
    for day, fraction in cum.items():
        if fraction >= target:
            return Percentile(value=int(day), over_cap=False)
    raise AssertionError("unreachable: curve reaches 1.0 by construction")


def month_end(month: pd.Period) -> pd.Timestamp:
    return (month + 1).start_time


def _wait_over(months: list[pd.Period], p95s: dict[pd.Period, Percentile]) -> int | None:
    """recommended_wait over the given months; None when refused (over-cap)."""
    if not months or any(p95s[m].over_cap for m in months):
        return None
    return _wait_formula([p95s[m].value for m in months])


def percentile_summary(
    percentiles: dict, months: list
) -> dict[int, tuple[float, float] | None]:
    """Per-percentile (mean, population std) across the given months — the
    headline 'days-to-pXX, mean/std across mature months' (CLAUDE.md metric 2).
    None for a percentile when there are no months or any month is over-cap."""
    summary: dict[int, tuple[float, float] | None] = {}
    for pct in PERCENTILES:
        values = [percentiles[m][pct] for m in months]
        if not values or any(p.over_cap for p in values):
            summary[pct] = None
        else:
            numbers = [p.value for p in values]
            summary[pct] = (statistics.fmean(numbers), statistics.pstdev(numbers))
    return summary


@dataclass
class CompletionAssessment:
    percentiles: dict[pd.Period, dict[int, Percentile]]
    training_months: list[pd.Period]
    learned_wait: int | None
    horizon: int | None
    mature_months: list[pd.Period]
    recommended_wait: int | None
    # per-percentile (mean, population std) across MATURE months; None entries
    # when there are no mature months or a percentile is censored
    mature_percentile_summary: dict[int, tuple[float, float] | None]
    f_mature: pd.Series | None  # index: lag day grid, values: median fraction
    negative_lag_excess_fraction: float
    statuses: list[Status] = field(default_factory=list)


def assess_completion(
    canonical: CanonicalResult, table: TableConfig, as_of: pd.Timestamp
) -> CompletionAssessment:
    analysis = table.analysis
    curves = build_month_curves(canonical, analysis.lag_cap_days)
    months = sorted(curves)
    percentiles = {
        month: {pct: days_to_percentile(curves[month], pct) for pct in PERCENTILES}
        for month in months
    }
    p95s = {month: percentiles[month][95] for month in months}
    statuses: list[Status] = []

    # ---- reconciliation (ALGORITHMS.md section 7) — from the ONE aggregation
    g = canonical.global_row
    bucket_sum = int(
        g["n_curve_eligible"]
        + g["n_null_event_time"]
        + g["n_null_load_time_only"]
        + g["n_negative_lag_excluded"]
        + g["n_join_unmatched"]
        + g["n_other_exclusions"]
    )
    if bucket_sum != int(g["row_count"]):
        statuses.append(
            Status(
                check=Check.RECONCILIATION,
                severity=Severity.RED,
                reason=ReasonCode.RECONCILIATION_MISMATCH,
                detail=f"total_rows={int(g['row_count'])} != bucket sum={bucket_sum}",
            )
        )
    if table.event_time_via is not None and int(g["n_base_rows"]) != int(g["row_count"]):
        # pre/post join reconciliation: with a unique lookup these are equal
        statuses.append(
            Status(
                check=Check.RECONCILIATION,
                severity=Severity.RED,
                reason=ReasonCode.RECONCILIATION_MISMATCH,
                detail=f"pre-join base rows {int(g['n_base_rows'])} != post-join "
                f"rows {int(g['row_count'])}",
            )
        )

    # ---- negative-lag policy (ALGORITHMS.md section 6)
    rows_with_both = int(
        g["row_count"] - g["n_null_event_time"] - g["n_null_load_time_only"]
        - g["n_join_unmatched"]
    )
    excess = (
        int(g["n_negative_lag_excluded"]) / rows_with_both if rows_with_both else 0.0
    )
    if excess > analysis.negative_lag_red_fraction:
        statuses.append(
            Status(
                check=Check.COMPLETION,
                severity=Severity.RED,
                reason=ReasonCode.NEGATIVE_LAG_EXCESS,
                detail=f"negative-lag excess fraction {excess:.4%} exceeds "
                f"{analysis.negative_lag_red_fraction:.4%}",
            )
        )

    # ---- learned wait from the FIXED training cohort (no iteration)
    cutoff_edge = as_of - pd.Timedelta(days=analysis.training_cutoff_days)
    training = [m for m in months if month_end(m) <= cutoff_edge]
    learned_wait = _wait_over(training, p95s)
    if not training:
        # an empty cohort is insufficient history, never a silent GREEN
        statuses.append(
            Status(
                check=Check.COMPLETION,
                severity=Severity.INSUFFICIENT_HISTORY,
                reason=ReasonCode.INSUFFICIENT_MATURE_MONTHS,
                detail="training cohort is empty: no month ends on or before "
                f"as_of - training_cutoff_days ({cutoff_edge.date()})",
            )
        )
    if training and learned_wait is None:
        statuses.append(
            Status(
                check=Check.COMPLETION,
                severity=Severity.INSUFFICIENT_HISTORY,
                reason=ReasonCode.PERCENTILE_OVER_CAP,
                detail="a training-cohort month is censored past lag_cap_days; "
                "recommended_wait refused",
            )
        )

    # ---- lag-support backtest (heuristic; ALGORITHMS.md section 5). Runs even
    # when the cohort wait was already refused: strata evidence is independent.
    if len(training) >= BACKTEST_MIN_COHORT:
        half = len(training) // 2
        older, younger = training[:half], training[half:]
        wait_older = _wait_over(older, p95s)
        wait_younger = _wait_over(younger, p95s)
        disagreement = None
        if (wait_older is None) != (wait_younger is None):
            # one stratum censored past the cap while the other is not IS a
            # disagreement about the lag support
            disagreement = (
                f"one cohort stratum is censored past lag_cap_days while the other "
                f"is not (older wait: {wait_older}, younger: {wait_younger})"
            )
        elif wait_older is not None and wait_younger is not None:
            gap = abs(wait_older - wait_younger)
            threshold = max(BACKTEST_FLOOR_DAYS, BACKTEST_RATIO * max(wait_older, wait_younger))
            if gap > threshold:
                disagreement = (
                    f"cohort strata disagree: older wait {wait_older}d vs "
                    f"younger {wait_younger}d (threshold {threshold:.1f}d)"
                )
        if disagreement:
            statuses.append(
                Status(
                    check=Check.COMPLETION,
                    severity=Severity.INSUFFICIENT_HISTORY,
                    reason=ReasonCode.BACKTEST_DISAGREEMENT,
                    detail=disagreement
                    + "; the modeled lag support may exceed training_cutoff_days",
                )
            )
            learned_wait = None

    # ---- maturity: exposure-based, single pass
    horizon = None
    mature: list[pd.Period] = []
    rec_wait = None
    if learned_wait is not None:
        horizon = max(analysis.training_cutoff_days, learned_wait)
        horizon_edge = as_of - pd.Timedelta(days=horizon)
        mature = [m for m in months if month_end(m) <= horizon_edge]
        if len(mature) < analysis.min_mature_months:
            statuses.append(
                Status(
                    check=Check.COMPLETION,
                    severity=Severity.INSUFFICIENT_HISTORY,
                    reason=ReasonCode.INSUFFICIENT_MATURE_MONTHS,
                    detail=f"{len(mature)} mature months < min_mature_months="
                    f"{analysis.min_mature_months}",
                )
            )
        else:
            rec_wait = _wait_over(mature, p95s)
            if rec_wait is None:
                statuses.append(
                    Status(
                        check=Check.COMPLETION,
                        severity=Severity.INSUFFICIENT_HISTORY,
                        reason=ReasonCode.PERCENTILE_OVER_CAP,
                        detail="a mature month is censored past lag_cap_days; "
                        "recommended_wait refused",
                    )
                )

    # ---- F_mature: pointwise median-of-curves (ALGORITHMS.md section 8)
    f_mature = None
    if mature:
        grid = np.arange(0, analysis.lag_cap_days + 2)
        stacked = [list(curves[m].cumulative(grid)) for m in mature]
        f_mature = pd.Series(median_of_curves(stacked), index=grid)

    if not statuses:
        statuses.append(Status(check=Check.COMPLETION, severity=Severity.GREEN))

    return CompletionAssessment(
        percentiles=percentiles,
        training_months=training,
        learned_wait=learned_wait,
        horizon=horizon,
        mature_months=mature,
        recommended_wait=rec_wait,
        mature_percentile_summary=percentile_summary(percentiles, mature),
        f_mature=f_mature,
        negative_lag_excess_fraction=excess,
        statuses=statuses,
    )
