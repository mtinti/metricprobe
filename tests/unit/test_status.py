"""Step 2 tests: the typed result/status model — severities with frozen
precedence, reason codes, and the full CLI reduction to exit codes 0/1/2 —
defined and tested before any metric exists."""

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


def test_exit_codes_full_reduction():
    non_red = [
        s(Severity.AMBER),
        s(Severity.INDETERMINATE),
        s(Severity.INSUFFICIENT_HISTORY),
        s(Severity.SKIPPED),
        s(Severity.GREEN),
    ]
    # Exit 0 = ran, no RED: green/amber/indeterminate/insufficient/skipped all 0.
    assert exit_code_for([]) == 0
    assert exit_code_for([s(Severity.GREEN)]) == 0
    assert exit_code_for(non_red) == 0
    # Exit 2 = ran successfully, at least one data-health RED.
    assert exit_code_for(non_red + [s(Severity.RED)]) == 2
    # Exit 1 = execution error; it dominates everything, because nothing
    # partial becomes visible — even REDs found before the failure.
    assert exit_code_for(non_red, execution_error=True) == 1
    assert exit_code_for(non_red + [s(Severity.RED)], execution_error=True) == 1
    assert exit_code_for([], execution_error=True) == 1


def test_non_green_requires_reason_code():
    with pytest.raises(ValidationError, match="reason"):
        Status(check=Check.VOLUME, severity=Severity.RED)


def test_green_must_not_carry_a_reason_code():
    with pytest.raises(ValidationError, match="green"):
        Status(check=Check.VOLUME, severity=Severity.GREEN, reason=ReasonCode.MISSING_TABLE)
    assert Status(check=Check.VOLUME, severity=Severity.GREEN).reason is None


def test_status_round_trips_over_the_wire():
    status = Status(
        check=Check.PARITY,
        severity=Severity.INDETERMINATE,
        reason=ReasonCode.PARITY_PREREQ_READ_UNCOMMITTED,
        detail="read_uncommitted enabled on one side",
    )
    # snapshots serialize to JSON, so the JSON-mode dump is the wire format
    assert Status.model_validate(status.model_dump(mode="json")) == status


def test_wire_values_are_frozen_v2():
    # These are the SERIALIZED values stored in snapshots; changing any of them
    # is a schema change and must bump STATUS_SCHEMA_VERSION (v2 = the value
    # sets below, including Check.PROBE and the Step 3-7 reason codes).
    assert STATUS_SCHEMA_VERSION == 2
    assert {severity.value for severity in Severity} == {
        "red",
        "amber",
        "indeterminate",
        "insufficient_history",
        "skipped",
        "green",
    }
    assert {check.value for check in Check} == {
        "probe",
        "volume",
        "completion",
        "freshness",
        "uniqueness",
        "parity",
        "dual_lag",
        "batch",
        "reconciliation",
    }
    assert {code.value for code in ReasonCode} == {
        "missing_table",
        "zero_row_month",
        "volume_gap",
        "volume_outlier",
        "volume_collapse",
        "arrival_deficit",
        "duplicate_keys",
        "negative_lag_excess",
        "stale_feed",
        "null_batch_ids",
        "parity_mismatch",
        "parity_one_sided_month",
        # parity prerequisites are SPECIFIC codes: the contract requires the
        # failing prerequisite as the reason, never a generic bucket
        "parity_prereq_uniqueness",
        "parity_prereq_read_uncommitted",
        "parity_prereq_negative_lag",
        "insufficient_mature_months",
        "insufficient_epochs",
        "backtest_disagreement",
        "percentile_over_cap",
        "scan_budget_exceeded",
        "scan_budget_unverifiable",
        "result_cell_cap_exceeded",
        "optional_table_absent",
        "join_not_unique",
        "join_unmatched_rows",
        "reconciliation_mismatch",
    }


def test_component_versions_are_pinned():
    """Changing ANY contract (config schema, status model, canonical or dual
    result schema / budget formulas, snapshot shape) must arrive as a visible
    version bump here — schema versions are wired to what they identify."""
    from metricprobe.cli import COMPONENT_VERSIONS

    assert COMPONENT_VERSIONS == {
        "config": 3,  # v3: compare resolution, dialect whitelist, finite params
        "status": 2,  # v2: PROBE check + Step 3-7 reason codes joined the wire
        "canonical": 6,  # v6: guard-artifact-only cells dropped (HAVING)
        "dual": 6,  # v6: same, with the () global row exempt
        "snapshot": 4,  # v4: mature summary refused below min_mature_months
    }
