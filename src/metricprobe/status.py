"""Typed result/status model, frozen and versioned before any metric exists.

Statuses are TYPED: each carries the check it belongs to, a severity, and a
reason code (mandatory for anything that is not GREEN). Severity precedence is
frozen as RED > AMBER > INDETERMINATE > INSUFFICIENT_HISTORY > SKIPPED > GREEN.

CLI reduction (exit codes separate code failure from data failure):
  0 = ran, no RED — amber/indeterminate/insufficient-history/skipped all
      reduce to 0 and stay visible in the statuses themselves
  2 = ran successfully, at least one data-health RED
  1 = execution error — never derived from statuses; it is the CLI's
      exception path (Step 6), where nothing partial becomes visible
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

STATUS_SCHEMA_VERSION = 1


class Severity(StrEnum):
    RED = "red"
    AMBER = "amber"
    INDETERMINATE = "indeterminate"
    INSUFFICIENT_HISTORY = "insufficient_history"
    SKIPPED = "skipped"
    GREEN = "green"


# worst first; frozen — tests pin this exact order
SEVERITY_PRECEDENCE: tuple[Severity, ...] = (
    Severity.RED,
    Severity.AMBER,
    Severity.INDETERMINATE,
    Severity.INSUFFICIENT_HISTORY,
    Severity.SKIPPED,
    Severity.GREEN,
)


class Check(StrEnum):
    VOLUME = "volume"
    COMPLETION = "completion"
    FRESHNESS = "freshness"
    UNIQUENESS = "uniqueness"
    PARITY = "parity"
    DUAL_LAG = "dual_lag"
    BATCH = "batch"
    RECONCILIATION = "reconciliation"


class ReasonCode(StrEnum):
    MISSING_TABLE = "missing_table"
    ZERO_ROW_MONTH = "zero_row_month"
    VOLUME_GAP = "volume_gap"
    VOLUME_OUTLIER = "volume_outlier"
    VOLUME_COLLAPSE = "volume_collapse"
    ARRIVAL_DEFICIT = "arrival_deficit"  # cause unresolved until maturity
    DUPLICATE_KEYS = "duplicate_keys"
    NEGATIVE_LAG_EXCESS = "negative_lag_excess"
    STALE_FEED = "stale_feed"
    PARITY_MISMATCH = "parity_mismatch"
    PARITY_ONE_SIDED_MONTH = "parity_one_sided_month"
    PARITY_PREREQUISITE_FAILED = "parity_prerequisite_failed"
    INSUFFICIENT_MATURE_MONTHS = "insufficient_mature_months"
    INSUFFICIENT_EPOCHS = "insufficient_epochs"
    BACKTEST_DISAGREEMENT = "backtest_disagreement"
    PERCENTILE_OVER_CAP = "percentile_over_cap"
    SCAN_BUDGET_EXCEEDED = "scan_budget_exceeded"
    RESULT_CELL_CAP_EXCEEDED = "result_cell_cap_exceeded"
    OPTIONAL_TABLE_ABSENT = "optional_table_absent"
    JOIN_NOT_UNIQUE = "join_not_unique"
    JOIN_UNMATCHED_ROWS = "join_unmatched_rows"
    RECONCILIATION_MISMATCH = "reconciliation_mismatch"


class Status(BaseModel):
    """One check's verdict. Serialized into snapshots, so it must round-trip."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: Check
    severity: Severity
    reason: ReasonCode | None = None
    detail: str = ""

    @model_validator(mode="after")
    def _reason_required_unless_green(self) -> Status:
        if self.severity is not Severity.GREEN and self.reason is None:
            raise ValueError(f"a {self.severity.value} status requires a reason code")
        return self


def worst_severity(statuses: Iterable[Status]) -> Severity:
    """The worst severity present, by frozen precedence; GREEN when empty."""
    present = {status.severity for status in statuses}
    for severity in SEVERITY_PRECEDENCE:
        if severity in present:
            return severity
    return Severity.GREEN


def exit_code_for(statuses: Iterable[Status]) -> int:
    """Reduce statuses to the CLI exit code: 2 on any RED, else 0. Exit 1 is
    the execution-error path and is never derived from statuses."""
    return 2 if worst_severity(statuses) is Severity.RED else 0
