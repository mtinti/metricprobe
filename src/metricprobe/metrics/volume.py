"""Volume history, robust baselines, and still-filling expectations —
formulas per docs/ALGORITHMS.md sections 8 and 10.

All counts come from the ONE canonical aggregation (the month/lag cells);
no separate query. Verdicts:

  * gaps (interior missing months)      -> RED VOLUME_GAP
  * mature month beyond amber/red MADs  -> AMBER/RED VOLUME_OUTLIER
  * >=2 consecutive below-red MATURE months ending at the last mature month
                                        -> RED VOLUME_COLLAPSE (mature-only)
  * immature CLOSED month below its expected-fill band
                                        -> AMBER ARRIVAL_DEFICIT
                                           ("cause unresolved" — late arrival
                                           and low source volume are not
                                           identifiable before maturity)
  * duplicate keys (key_cols configured) -> RED DUPLICATE_KEYS
  * the OPEN month (containing as_of) is excluded from every check

The still-filling expectation is expected(t) = forecast x F_mature(t) at
month-end age t, with the forecast the baseline median of mature volumes —
NEVER the immature month's own count. The inverted nowcast observed/F(t) is
REPORTED but never fed back (tautology guard).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult
from metricprobe.metrics.completion import CompletionAssessment, month_end
from metricprobe.metrics.robust import expected_fill_band, robust_sigma_floor
from metricprobe.status import Check, ReasonCode, Severity, Status

# a sustained collapse is at least this many consecutive below-red mature
# months ending at the most recent mature month (v1 algorithm constant)
COLLAPSE_MIN_RUN = 2


@dataclass
class MonthVolume:
    month: pd.Period
    volume: int
    state: str  # "mature" | "immature" | "open"
    expected_low: float | None = None  # still-filling band (immature months)
    expected_high: float | None = None
    nowcast: float | None = None  # observed / F_mature(age): reported, never used
    deficit: bool = False


@dataclass
class VolumeAssessment:
    months: list[MonthVolume]
    gaps: list[pd.Period]
    baseline_months: list[pd.Period]
    baseline_median: float | None
    baseline_sigma: float | None
    forecast: float | None
    duplicate_rows: int | None  # None when no key_cols configured
    statuses: list[Status] = field(default_factory=list)


def _volumes_by_month(canonical: CanonicalResult) -> dict[pd.Period, int]:
    """Curve-eligible rows per month (the -1 negative-excluded sentinel cells
    belong to parity's watermarked population, not to volume history)."""
    cells = canonical.rows_for("month_lag")
    cells = cells[cells["lag_day"] >= 0]
    grouped = cells.groupby("event_month")["row_count"].sum()
    return {pd.Period(ts, freq="M"): int(count) for ts, count in grouped.items()}


def assess_volume(
    canonical: CanonicalResult,
    table: TableConfig,
    as_of: pd.Timestamp,
    completion: CompletionAssessment,
) -> VolumeAssessment:
    analysis = table.analysis
    volumes = _volumes_by_month(canonical)
    months = sorted(volumes)
    statuses: list[Status] = []

    if not months:
        # zero rows admitted: a hard failure, never a crash or a silent pass
        statuses.append(
            Status(
                check=Check.VOLUME,
                severity=Severity.RED,
                reason=ReasonCode.ZERO_ROW_MONTH,
                detail="no curve-eligible rows: the table is empty (or every row "
                "was excluded) under the as_of watermark",
            )
        )
        return VolumeAssessment(
            months=[],
            gaps=[],
            baseline_months=[],
            baseline_median=None,
            baseline_sigma=None,
            forecast=None,
            duplicate_rows=None,
            statuses=statuses,
        )

    # ---- explicit gaps: interior months entirely absent
    gaps: list[pd.Period] = []
    if months:
        gaps = [
            month
            for month in pd.period_range(months[0], months[-1], freq="M")
            if month not in volumes
        ]
    if gaps:
        statuses.append(
            Status(
                check=Check.VOLUME,
                severity=Severity.RED,
                reason=ReasonCode.VOLUME_GAP,
                detail="month(s) with zero rows: " + ", ".join(str(m) for m in gaps),
            )
        )

    # ---- uniqueness (from the distinct-count guard, same probe)
    duplicate_rows = None
    if table.key_cols:
        g = canonical.global_row
        duplicate_rows = int(g["row_count"]) - int(g["distinct_keys"])
        if duplicate_rows > 0:
            statuses.append(
                Status(
                    check=Check.UNIQUENESS,
                    severity=Severity.RED,
                    reason=ReasonCode.DUPLICATE_KEYS,
                    detail=f"{duplicate_rows} duplicate rows over key {list(table.key_cols)}",
                )
            )

    # ---- classify months: open (contains as_of or later) / mature / immature
    mature_set = set(completion.mature_months)
    month_states: dict[pd.Period, str] = {}
    for month in months:
        # open = the month has not ended by as_of (its events are still occurring)
        if month_end(month) > as_of:
            month_states[month] = "open"
        elif month in mature_set:
            month_states[month] = "mature"
        else:
            month_states[month] = "immature"

    # ---- baseline: mature history EXCLUDING the evaluation window (the last
    # observed months), so a sustained degradation cannot normalize itself
    evaluation_window = set(months[-analysis.evaluation_window_months :])
    baseline = [
        m for m in months if month_states[m] == "mature" and m not in evaluation_window
    ]
    baseline_median = baseline_sigma = forecast = None
    maturity_known = completion.horizon is not None
    if not maturity_known:
        statuses.append(
            Status(
                check=Check.VOLUME,
                severity=Severity.INSUFFICIENT_HISTORY,
                reason=ReasonCode.INSUFFICIENT_MATURE_MONTHS,
                detail="volume baselines need maturity classification, which is "
                "unavailable (see completion statuses)",
            )
        )
    elif len(baseline) < analysis.min_mature_months:
        statuses.append(
            Status(
                check=Check.VOLUME,
                severity=Severity.INSUFFICIENT_HISTORY,
                reason=ReasonCode.INSUFFICIENT_MATURE_MONTHS,
                detail=f"{len(baseline)} baseline months (mature minus evaluation "
                f"window) < min_mature_months={analysis.min_mature_months}",
            )
        )
        baseline = []
    if baseline:
        baseline_volumes = [volumes[m] for m in baseline]
        baseline_median = float(statistics.median(baseline_volumes))
        baseline_sigma = robust_sigma_floor(baseline_volumes)  # zero-MAD floor
        forecast = baseline_median  # independent volume forecast (non-seasonal v1)

    # ---- mature-month verdicts: sustained collapse first, then outliers
    collapse_run: list[pd.Period] = []
    if baseline_median is not None:
        red_low = baseline_median - analysis.volume_red_mads * baseline_sigma
        mature_months = [m for m in months if month_states[m] == "mature"]
        run: list[pd.Period] = []
        for month in mature_months:
            if run and month != run[-1] + 1:
                run = []  # a calendar gap breaks the run: "consecutive" months only
            if volumes[month] < red_low:
                run.append(month)
            else:
                run = []
        if len(run) >= COLLAPSE_MIN_RUN:  # ends at the most recent mature month
            collapse_run = run
            statuses.append(
                Status(
                    check=Check.VOLUME,
                    severity=Severity.RED,
                    reason=ReasonCode.VOLUME_COLLAPSE,
                    detail="sustained volume collapse on mature month(s) "
                    + ", ".join(str(m) for m in collapse_run)
                    + f" (each below {red_low:.0f})",
                )
            )
        for month in mature_months:
            if month in collapse_run:
                continue
            deviation = abs(volumes[month] - baseline_median)
            if deviation > analysis.volume_red_mads * baseline_sigma:
                severity = Severity.RED
            elif deviation > analysis.volume_amber_mads * baseline_sigma:
                severity = Severity.AMBER
            else:
                continue
            statuses.append(
                Status(
                    check=Check.VOLUME,
                    severity=severity,
                    reason=ReasonCode.VOLUME_OUTLIER,
                    detail=f"{month}: {volumes[month]} rows vs baseline median "
                    f"{baseline_median:.0f} (robust sigma {baseline_sigma:.0f})",
                )
            )

    # ---- still-filling: immature CLOSED months against forecast x F_mature(t)
    month_rows: list[MonthVolume] = []
    deficits: list[pd.Period] = []
    f_mature = completion.f_mature
    # the frozen band dispersion population: MATURE final volumes (ALGORITHMS
    # section 8). Whenever immature months exist the evaluation window is
    # immature too (maturity is monotone in time), so this equals the baseline.
    mature_volumes = [volumes[m] for m in months if month_states[m] == "mature"]
    for month in months:
        row = MonthVolume(month=month, volume=volumes[month], state=month_states[month])
        if (
            row.state == "immature"
            and forecast is not None
            and f_mature is not None
        ):
            age_days = int((as_of - month_end(month)) / pd.Timedelta(days=1))
            grid_max = int(f_mature.index.max())
            fill = float(f_mature.loc[min(max(age_days, 0), grid_max)])
            row.expected_low, row.expected_high = expected_fill_band(
                forecast=forecast,
                fill_fraction=fill,
                mature_volumes=mature_volumes,
                band_mads=analysis.expected_fill_band_mads,
            )
            row.nowcast = volumes[month] / fill if fill > 0 else None
            if volumes[month] < row.expected_low:
                row.deficit = True
                deficits.append(month)
        month_rows.append(row)
    if deficits:
        statuses.append(
            Status(
                check=Check.VOLUME,
                severity=Severity.AMBER,
                reason=ReasonCode.ARRIVAL_DEFICIT,
                detail="arrival deficit — cause unresolved (late arrival vs low "
                "source volume are not identifiable before maturity) on "
                + ", ".join(str(m) for m in deficits),
            )
        )

    if not statuses:
        statuses.append(Status(check=Check.VOLUME, severity=Severity.GREEN))

    return VolumeAssessment(
        months=month_rows,
        gaps=gaps,
        baseline_months=baseline,
        baseline_median=baseline_median,
        baseline_sigma=baseline_sigma,
        forecast=forecast,
        duplicate_rows=duplicate_rows,
        statuses=statuses,
    )
