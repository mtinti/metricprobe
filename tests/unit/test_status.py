"""Step 2 tests: the typed result/status model — severities with frozen
precedence, reason codes, and the CLI reduction to exit codes — defined and
tested before any metric exists."""

import pytest
from pydantic import ValidationError

from metricprobe.status import (
    SEVERITY_PRECEDENCE,
    STATUS_SCHEMA_VERSION,
    Check,
    ReasonCode,
    Severity,
    Status,
    exit_code_for,
    worst_severity,
)


def s(severity: Severity, reason: ReasonCode | None = None) -> Status:
    if severity is not Severity.GREEN and reason is None:
        reason = ReasonCode.VOLUME_OUTLIER
    return Status(check=Check.VOLUME, severity=severity, reason=reason)


def test_precedence_is_frozen():
    assert SEVERITY_PRECEDENCE == (
        Severity.RED,
        Severity.AMBER,
        Severity.INDETERMINATE,
        Severity.INSUFFICIENT_HISTORY,
        Severity.SKIPPED,
        Severity.GREEN,
    )


def test_worst_severity_follows_precedence():
    # dropping the current worst must surface the next one, all the way down
    remaining = list(SEVERITY_PRECEDENCE)
    while remaining:
        statuses = [s(severity) for severity in remaining]
        assert worst_severity(statuses) is remaining[0]
        remaining.pop(0)
    assert worst_severity([]) is Severity.GREEN


def test_exit_codes():
    # Exit 0 = ran, no RED: green/amber/indeterminate/insufficient/skipped all 0.
    assert exit_code_for([]) == 0
    assert exit_code_for([s(Severity.GREEN)]) == 0
    non_red = [
        s(Severity.AMBER),
        s(Severity.INDETERMINATE),
        s(Severity.INSUFFICIENT_HISTORY),
        s(Severity.SKIPPED),
        s(Severity.GREEN),
    ]
    assert exit_code_for(non_red) == 0
    # Exit 2 = ran successfully, at least one data-health RED.
    assert exit_code_for(non_red + [s(Severity.RED)]) == 2


def test_non_green_requires_reason_code():
    with pytest.raises(ValidationError, match="reason"):
        Status(check=Check.VOLUME, severity=Severity.RED)
    assert Status(check=Check.VOLUME, severity=Severity.GREEN).reason is None


def test_status_round_trips_for_snapshots():
    status = Status(
        check=Check.PARITY,
        severity=Severity.INDETERMINATE,
        reason=ReasonCode.PARITY_PREREQUISITE_FAILED,
        detail="read_uncommitted enabled on one side",
    )
    assert Status.model_validate(status.model_dump()) == status


def test_reason_codes_are_frozen_v1():
    assert STATUS_SCHEMA_VERSION == 1
    assert {code.name for code in ReasonCode} == {
        "MISSING_TABLE",
        "ZERO_ROW_MONTH",
        "VOLUME_GAP",
        "VOLUME_OUTLIER",
        "VOLUME_COLLAPSE",
        "ARRIVAL_DEFICIT",
        "DUPLICATE_KEYS",
        "NEGATIVE_LAG_EXCESS",
        "STALE_FEED",
        "PARITY_MISMATCH",
        "PARITY_ONE_SIDED_MONTH",
        "PARITY_PREREQUISITE_FAILED",
        "INSUFFICIENT_MATURE_MONTHS",
        "INSUFFICIENT_EPOCHS",
        "BACKTEST_DISAGREEMENT",
        "PERCENTILE_OVER_CAP",
        "SCAN_BUDGET_EXCEEDED",
        "RESULT_CELL_CAP_EXCEEDED",
        "OPTIONAL_TABLE_ABSENT",
        "JOIN_NOT_UNIQUE",
        "JOIN_UNMATCHED_ROWS",
        "RECONCILIATION_MISMATCH",
    }


def test_checks_are_frozen_v1():
    assert {check.name for check in Check} == {
        "VOLUME",
        "COMPLETION",
        "FRESHNESS",
        "UNIQUENESS",
        "PARITY",
        "DUAL_LAG",
        "BATCH",
        "RECONCILIATION",
    }
