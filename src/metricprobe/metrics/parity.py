"""Parity between two NAMED probes (ALGORITHMS.md section 13).

Exact parity compares the watermarked per-month population (curve-eligible
counts) over the union of both sides' observed MATURE months via a full outer
join: a month present on one side only is an explicit RED diff, never silently
dropped. Zero-tolerance parity is sound only under VERIFIED prerequisites;
any failure or unverifiability yields INDETERMINATE with the failing
prerequisite as the reason code — never a false mismatch. NULL-load rows are
query-time counts, reported informationally, never inside the exact diff."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from metricprobe.config import TableConfig
from metricprobe.extract.canonical import CanonicalResult
from metricprobe.metrics.completion import CompletionAssessment, month_end
from metricprobe.metrics.volume import VolumeAssessment
from metricprobe.status import Check, ReasonCode, Severity, Status


@dataclass
class ParitySide:
    config: TableConfig
    canonical: CanonicalResult
    completion: CompletionAssessment
    volume: VolumeAssessment

    def watermarked_counts(self) -> dict[pd.Period, int]:
        """The population exact parity compares: rows with non-NULL load_time
        <= as_of and a defined event month — curve-eligible rows PLUS the
        negative-lag-excluded rows kept at the frozen -1 sentinel (an allowed
        below-threshold excess must not create false diffs)."""
        cells = self.canonical.rows_for("month_lag")
        grouped = cells.groupby("event_month")["row_count"].sum()
        return {pd.Period(ts, freq="M"): int(count) for ts, count in grouped.items()}


@dataclass
class ParityMonth:
    """The parity storage schema — one row per compared month."""

    month: pd.Period
    left_count: int | None  # None = month absent on this side
    right_count: int | None
    diff: int | None  # None when one-sided
    verdict: str  # "match" | "mismatch" | "left_only" | "right_only"


@dataclass
class ParityAssessment:
    rows: list[ParityMonth]
    null_load_left: int  # informational: query-time counts, never in the diff
    null_load_right: int
    statuses: list[Status] = field(default_factory=list)


def _failed_prerequisite(left: ParitySide, right: ParitySide) -> tuple[ReasonCode, str] | None:
    for name, side in (("left", left), ("right", right)):
        if side.config.read_uncommitted:
            return (
                ReasonCode.PARITY_PREREQ_READ_UNCOMMITTED,
                f"read_uncommitted is enabled on the {name} side "
                f"({side.config.probe_name!r})",
            )
    for name, side in (("left", left), ("right", right)):
        if not side.config.key_cols:
            return (
                ReasonCode.PARITY_PREREQ_UNIQUENESS,
                f"uniqueness is not configured (no key_cols) on the {name} side "
                f"({side.config.probe_name!r}) — exact parity is unverifiable",
            )
        if side.volume.duplicate_rows is None or side.volume.duplicate_rows > 0:
            return (
                ReasonCode.PARITY_PREREQ_UNIQUENESS,
                f"{side.volume.duplicate_rows} duplicate keys on the {name} side "
                f"({side.config.probe_name!r})",
            )
    for name, side in (("left", left), ("right", right)):
        threshold = side.config.analysis.negative_lag_red_fraction
        # the contract requires the excess to be BELOW the threshold: exactly
        # AT it is not "below", so the prerequisite fails (>=, not >)
        if side.completion.negative_lag_excess_fraction >= threshold:
            return (
                ReasonCode.PARITY_PREREQ_NEGATIVE_LAG,
                f"negative-lag excess (the backdating proxy) is not below the "
                f"threshold on the {name} side ({side.config.probe_name!r})",
            )
    return None


def assess_parity(left: ParitySide, right: ParitySide, as_of: pd.Timestamp) -> ParityAssessment:
    # parity_with is the declared relationship: refuse arbitrary pairings
    if left.config.parity_with != right.config.probe_name:
        raise ValueError(
            f"probe {left.config.probe_name!r} declares parity_with="
            f"{left.config.parity_with!r}, not {right.config.probe_name!r}"
        )
    null_load_left = int(left.canonical.global_row["n_null_load_time_only"])
    null_load_right = int(right.canonical.global_row["n_null_load_time_only"])

    failed = _failed_prerequisite(left, right)
    if failed is not None:
        reason, detail = failed
        return ParityAssessment(
            rows=[],
            null_load_left=null_load_left,
            null_load_right=null_load_right,
            statuses=[
                Status(
                    check=Check.PARITY,
                    severity=Severity.INDETERMINATE,
                    reason=reason,
                    detail=detail + "; parity is indeterminate, not a mismatch",
                )
            ],
        )

    if left.completion.horizon is None or right.completion.horizon is None:
        return ParityAssessment(
            rows=[],
            null_load_left=null_load_left,
            null_load_right=null_load_right,
            statuses=[
                Status(
                    check=Check.PARITY,
                    severity=Severity.INSUFFICIENT_HISTORY,
                    reason=ReasonCode.INSUFFICIENT_MATURE_MONTHS,
                    detail="maturity is unavailable on at least one side; the "
                    "common mature population cannot be established",
                )
            ],
        )
    tolerance = left.config.analysis.parity_tolerance
    # the COMMON mature population is defined by TIME under the STRICTER of
    # the two horizons — different learned horizons on identical data must
    # never manufacture one-sided months
    common_edge = as_of - pd.Timedelta(
        days=max(left.completion.horizon, right.completion.horizon)
    )
    left_counts = {
        m: c for m, c in left.watermarked_counts().items() if month_end(m) <= common_edge
    }
    right_counts = {
        m: c for m, c in right.watermarked_counts().items() if month_end(m) <= common_edge
    }
    rows: list[ParityMonth] = []
    statuses: list[Status] = []
    # full outer join over the UNION of both sides' observed common-mature months
    for month in sorted(set(left_counts) | set(right_counts)):
        left_count = left_counts.get(month)
        right_count = right_counts.get(month)
        if left_count is None or right_count is None:
            verdict = "left_only" if right_count is None else "right_only"
            rows.append(ParityMonth(month, left_count, right_count, None, verdict))
            statuses.append(
                Status(
                    check=Check.PARITY,
                    severity=Severity.RED,
                    reason=ReasonCode.PARITY_ONE_SIDED_MONTH,
                    detail=f"{month} is present only on the "
                    f"{'left' if right_count is None else 'right'} side",
                )
            )
            continue
        diff = left_count - right_count
        if abs(diff) > tolerance:
            rows.append(ParityMonth(month, left_count, right_count, diff, "mismatch"))
            statuses.append(
                Status(
                    check=Check.PARITY,
                    severity=Severity.RED,
                    reason=ReasonCode.PARITY_MISMATCH,
                    detail=f"{month}: left {left_count} vs right {right_count} "
                    f"(diff {diff:+d}, tolerance {tolerance})",
                )
            )
        else:
            rows.append(ParityMonth(month, left_count, right_count, diff, "match"))
    if not statuses:
        statuses.append(Status(check=Check.PARITY, severity=Severity.GREEN))
    return ParityAssessment(
        rows=rows,
        null_load_left=null_load_left,
        null_load_right=null_load_right,
        statuses=statuses,
    )
