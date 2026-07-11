"""Step 4 golden tests: volume history + baselines + still-filling.

Every check is tested in both directions via the healthy/unhealthy twins: it
must fire on the pathology AND stay silent on the healthy twin. The freshness
VERDICT for the collapse pair (updating: GREEN) lands with the freshness
metric in Step 5; here the data-level premise (batches keep their cadence) is
asserted directly."""

import pandas as pd
import pytest
from tests.support import probe, table_config
from tests.synth import generator as g
from tests.synth.scenarios import BATCHY_BASE, catalog

from metricprobe.metrics.completion import assess_completion
from metricprobe.metrics.volume import assess_volume
from metricprobe.status import Check, ReasonCode, Severity, Status

AS_OF_MATURE = "2026-07-01"  # every scenario month is long mature by then


def _assess(df, config, as_of):
    canonical = probe(df, config, as_of)
    completion = assess_completion(canonical, config, pd.Timestamp(as_of))
    return canonical, assess_volume(canonical, config, pd.Timestamp(as_of), completion)


def _reasons(assessment) -> set[ReasonCode]:
    return {s.reason for s in assessment.statuses if s.reason is not None}


def _months_in_detail(assessment, reason: ReasonCode) -> str:
    return "; ".join(s.detail for s in assessment.statuses if s.reason is reason)


# ------------------------------------------------------------- outliers + gaps


def test_volume_spike_pair():
    pair = catalog()["volume_spike"]
    _, unhealthy = _assess(pair.unhealthy(), table_config(), AS_OF_MATURE)
    spikes = [s for s in unhealthy.statuses if s.reason is ReasonCode.VOLUME_OUTLIER]
    assert spikes and all(s.severity is Severity.RED for s in spikes)
    assert "2024-07" in spikes[0].detail  # the injected month
    _, healthy = _assess(pair.healthy(), table_config(), AS_OF_MATURE)
    assert _reasons(healthy) == set()  # silent on the healthy twin


def test_volume_drop_pair():
    pair = catalog()["volume_drop"]
    _, unhealthy = _assess(pair.unhealthy(), table_config(), AS_OF_MATURE)
    drops = [s for s in unhealthy.statuses if s.reason is ReasonCode.VOLUME_OUTLIER]
    assert drops and "2024-04" in drops[0].detail
    _, healthy = _assess(pair.healthy(), table_config(), AS_OF_MATURE)
    assert _reasons(healthy) == set()


def test_missing_month_is_an_explicit_gap():
    pair = catalog()["missing_month"]
    _, unhealthy = _assess(pair.unhealthy(), table_config(), AS_OF_MATURE)
    assert pd.Period("2024-01", freq="M") in unhealthy.gaps
    gap_statuses = [s for s in unhealthy.statuses if s.reason is ReasonCode.VOLUME_GAP]
    assert gap_statuses and gap_statuses[0].severity is Severity.RED
    assert "2024-01" in gap_statuses[0].detail
    _, healthy = _assess(pair.healthy(), table_config(), AS_OF_MATURE)
    assert healthy.gaps == []


def test_duplicate_keys_trip_the_uniqueness_check():
    pair = catalog()["duplicate_keys"]
    config = table_config(key_cols=["row_id"])
    _, unhealthy = _assess(pair.unhealthy(), config, AS_OF_MATURE)
    dup = [s for s in unhealthy.statuses if s.reason is ReasonCode.DUPLICATE_KEYS]
    assert dup and dup[0].severity is Severity.RED and dup[0].check is Check.UNIQUENESS
    expected_duplicates = round(0.02 * 30 * 2000)
    assert unhealthy.duplicate_rows == expected_duplicates
    _, healthy = _assess(pair.healthy(), config, AS_OF_MATURE)
    assert healthy.duplicate_rows == 0
    assert _reasons(healthy) == set()


def test_no_uniqueness_check_without_key_cols():
    _, assessment = _assess(catalog()["duplicate_keys"].unhealthy(), table_config(), AS_OF_MATURE)
    assert assessment.duplicate_rows is None
    assert ReasonCode.DUPLICATE_KEYS not in _reasons(assessment)


# --------------------------------------------------------- sustained collapse


def test_sustained_collapse_pair():
    pair = catalog()["sustained_collapse"]
    as_of = "2026-08-01"  # the collapsed months (2025-04..06) are mature
    config = table_config(load_batch_col="batch_id")
    canonical, unhealthy = _assess(pair.unhealthy(), config, as_of)

    # (b) volume=RED with the specific collapse verdict, on MATURE months only
    collapse = [s for s in unhealthy.statuses if s.reason is ReasonCode.VOLUME_COLLAPSE]
    assert len(collapse) == 1 and collapse[0].severity is Severity.RED
    for month in ("2025-04", "2025-05", "2025-06"):
        assert month in collapse[0].detail

    # (a) baseline computed EXCLUDING the evaluation window: the collapse must
    # not normalize itself — the baseline median stays at the healthy level
    assert unhealthy.baseline_median == pytest.approx(2000, rel=0.05)

    # the freshness premise: loads kept arriving on their normal cadence
    # (batch epochs exist for every collapsed month; the freshness VERDICT
    # test lands with the metric in Step 5)
    batches = canonical.rows_for("month_batch")
    collapsed = batches[batches["event_month"] >= "2025-04-01"]
    assert collapsed["batch_id"].nunique() == 9  # 3 batches x 3 collapsed months

    _, healthy = _assess(pair.healthy(), config, as_of)
    assert _reasons(healthy) == set()  # all-green twin


def test_mad_zero_fallback_on_perfectly_regular_volumes():
    # synthetic volumes are exactly constant, so MAD would be 0; the frozen
    # relative floor (5% of median) must apply instead of a zero band
    _, assessment = _assess(catalog()["sustained_collapse"].healthy(), table_config(), "2026-08-01")
    assert assessment.baseline_sigma == pytest.approx(0.05 * 2000)


def test_minimum_history_for_volume_baselines():
    spec = g.TableSpec(
        name="events", start_month="2025-01", n_months=4, rows_per_month=2000,
        lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=71,
    )
    _, assessment = _assess(g.generate(spec), table_config(), "2025-06-01")
    insufficient = [
        s for s in assessment.statuses
        if s.check is Check.VOLUME and s.severity is Severity.INSUFFICIENT_HISTORY
    ]
    assert insufficient
    assert ReasonCode.VOLUME_OUTLIER not in _reasons(assessment)


# ---------------------------------------------------------------- still-filling

TRICKLE_30 = g.TableSpec(
    name="events", start_month="2023-01", n_months=30, rows_per_month=2000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=72,
)


def test_open_month_is_open_never_a_deficit():
    # events through 2025-06 plus an as_of INSIDE 2025-06: the open month must
    # be excluded entirely — reported as "open", never as a deficit
    _, assessment = _assess(g.generate(TRICKLE_30), table_config(), "2025-06-15")
    by_month = {m.month: m for m in assessment.months}
    open_month = by_month[pd.Period("2025-06", freq="M")]
    assert open_month.state == "open"
    assert not open_month.deficit
    deficit_details = _months_in_detail(assessment, ReasonCode.ARRIVAL_DEFICIT)
    assert "2025-06" not in deficit_details


def test_tautology_guard_immature_month_at_half_expectation_is_flagged():
    # The estimator must be able to FAIL: expected = independent mature-volume
    # forecast x F_mature(age) — NEVER the month's own count. A month whose
    # true volume halved is at ~50% of expectation and must be flagged even
    # though its own inverted nowcast looks self-consistent.
    dropped = g.volume_drop(TRICKLE_30, 29, factor=0.5)  # 2025-06, closed but immature
    as_of = "2025-07-20"  # month-end age 19d; F_mature(19) ~ 0.95
    _, unhealthy = _assess(g.generate(dropped), table_config(), as_of)
    by_month = {m.month: m for m in unhealthy.months}
    june = by_month[pd.Period("2025-06", freq="M")]
    assert june.state == "immature"
    assert june.deficit
    # expected band is centered on forecast x F (~1900), proving the month's
    # own count (~950) was never fed back into its own expectation
    assert june.expected_low > 1200
    # the inverted nowcast IS reported (observed / F ~ the true dropped volume)
    assert june.nowcast == pytest.approx(1000, rel=0.15)
    # ... and the label is "arrival deficit — cause unresolved", never collapse
    deficits = [s for s in unhealthy.statuses if s.reason is ReasonCode.ARRIVAL_DEFICIT]
    assert deficits and deficits[0].severity is Severity.AMBER
    assert "cause unresolved" in deficits[0].detail
    assert ReasonCode.VOLUME_COLLAPSE not in _reasons(unhealthy)

    _, healthy = _assess(g.generate(TRICKLE_30), table_config(), as_of)
    assert ReasonCode.ARRIVAL_DEFICIT not in _reasons(healthy)


def test_verdict_upgrades_when_the_deficit_month_matures():
    # golden: the same collapsed data seen young reads "arrival deficit";
    # seen after maturity it reads "volume collapse" — the verdict upgrades
    collapsed = g.sustained_collapse(BATCHY_BASE, last_k=3, factor=0.1)
    df = g.generate(collapsed)
    config = table_config()

    _, young = _assess(df, config, "2025-07-25")  # collapsed months immature
    assert ReasonCode.ARRIVAL_DEFICIT in _reasons(young)
    assert ReasonCode.VOLUME_COLLAPSE not in _reasons(young)
    deficit_details = _months_in_detail(young, ReasonCode.ARRIVAL_DEFICIT)
    assert "2025-04" in deficit_details and "2025-05" in deficit_details

    _, matured = _assess(df, config, "2026-08-01")  # now mature
    assert ReasonCode.VOLUME_COLLAPSE in _reasons(matured)
    assert ReasonCode.ARRIVAL_DEFICIT not in _reasons(matured)


def test_volume_verdicts_are_never_assigned_to_immature_months():
    collapsed = g.sustained_collapse(BATCHY_BASE, last_k=3, factor=0.1)
    _, young = _assess(g.generate(collapsed), table_config(), "2025-07-25")
    for status in young.statuses:
        if status.reason in (ReasonCode.VOLUME_OUTLIER, ReasonCode.VOLUME_COLLAPSE):
            raise AssertionError(f"volume verdict on immature data: {status}")


def test_statuses_feed_the_frozen_typed_model():
    _, assessment = _assess(
        catalog()["sustained_collapse"].unhealthy(), table_config(), "2026-08-01"
    )
    for status in assessment.statuses:
        assert isinstance(status, Status)
        assert Status.model_validate(status.model_dump(mode="json")) == status
