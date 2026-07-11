"""Step 4 golden tests: volume history + baselines + still-filling.

Every check is tested in both directions via the healthy/unhealthy twins: it
must fire on the pathology AND stay silent on the healthy twin. The
sustained-collapse acceptance case asserts volume=RED AND freshness=GREEN
simultaneously (the freshness staleness core lands here for exactly that;
its full golden suite is Step 5)."""

import statistics

import pandas as pd
import pytest
from tests.support import probe, table_config
from tests.synth import generator as g
from tests.synth.scenarios import BATCHY_BASE, catalog

from metricprobe.metrics.completion import assess_completion
from metricprobe.metrics.freshness import assess_freshness
from metricprobe.metrics.robust import robust_sigma_floor
from metricprobe.metrics.volume import assess_volume
from metricprobe.status import Check, ReasonCode, Severity, Status, worst_severity

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


def test_sustained_collapse_pair_volume_red_and_freshness_green():
    # THE acceptance case: volume=RED on mature collapsed months WHILE
    # freshness=GREEN. Only satisfiable when the collapse is OLDER than the
    # maturity horizon and loads continue to arrive: the catalog collapse
    # spans the last 15 months, and as_of sits 3 days after the latest batch.
    pair = catalog()["sustained_collapse"]
    as_of = "2025-07-24"  # last batch (2025-06's day-20 batch) is 2025-07-21
    config = table_config(load_batch_col="batch_id")
    canonical, unhealthy = _assess(pair.unhealthy(), config, as_of)

    # (b) volume=RED with the specific collapse verdict, on the MATURE part
    # of the collapse (2024-04..06 under the 365d horizon)
    collapse = [s for s in unhealthy.statuses if s.reason is ReasonCode.VOLUME_COLLAPSE]
    assert len(collapse) == 1 and collapse[0].severity is Severity.RED
    for month in ("2024-04", "2024-05", "2024-06"):
        assert month in collapse[0].detail

    # freshness=GREEN: the feed is updating on its learned cadence
    fresh = assess_freshness(canonical, config, pd.Timestamp(as_of))
    assert worst_severity(fresh.statuses) is Severity.GREEN
    assert fresh.days_since_last == pytest.approx(3, abs=1)

    # (a) baseline computed EXCLUDING degraded recent history: the median
    # stays at the healthy level (the 3 collapsed mature months out of 18
    # cannot move a median)
    assert unhealthy.baseline_median == pytest.approx(2000, rel=0.05)

    _, healthy = _assess(pair.healthy(), config, as_of)
    assert _reasons(healthy) == set()  # all-green twin (volume side)
    healthy_canonical = probe(pair.healthy(), config, as_of)
    healthy_fresh = assess_freshness(healthy_canonical, config, pd.Timestamp(as_of))
    assert worst_severity(healthy_fresh.statuses) is Severity.GREEN


def test_evaluation_window_exclusion_is_load_bearing():
    # A case where INCLUDING the evaluation window would flip the verdict:
    # 5 healthy + 5 collapsed months with a 5-month window. Excluded, the
    # baseline median is 2000 and the collapse is RED; included, the median
    # would sink to 1100 with a huge MAD and the collapse would vanish.
    spec = g.TableSpec(
        name="events", start_month="2024-01", n_months=10, rows_per_month=2000,
        lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=74,
    )
    spec = g.sustained_collapse(spec, last_k=5, factor=0.1)
    config = table_config(analysis={"evaluation_window_months": 5, "min_mature_months": 5})
    _, assessment = _assess(g.generate(spec), config, "2025-12-01")  # all mature
    assert [str(m) for m in assessment.baseline_months] == [
        "2024-01", "2024-02", "2024-03", "2024-04", "2024-05",
    ]
    assert assessment.baseline_median == 2000
    assert ReasonCode.VOLUME_COLLAPSE in _reasons(assessment)
    # the counterfactual: an all-months median would be non-discriminating
    all_volumes = [m.volume for m in assessment.months]
    assert statistics.median(all_volumes) == pytest.approx(1100, rel=0.01)


def test_mad_zero_fallback_on_perfectly_regular_volumes():
    # synthetic volumes are exactly constant, so MAD would be 0; the frozen
    # relative floor (5% of median) must apply instead of a zero band
    _, assessment = _assess(catalog()["sustained_collapse"].healthy(), table_config(), "2025-07-24")
    assert assessment.baseline_sigma == pytest.approx(0.05 * 2000)


def test_minimum_history_hits_the_baseline_size_branch():
    # maturity IS classifiable (8 mature months >= min_mature_months for
    # completion), but the volume baseline — mature minus the 3-month
    # evaluation window — has only 5 members < 6: the Step 4 branch.
    spec = g.TableSpec(
        name="events", start_month="2024-01", n_months=8, rows_per_month=2000,
        lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=71,
    )
    canonical = probe(g.generate(spec), table_config(), "2025-11-01")
    completion = assess_completion(canonical, table_config(), pd.Timestamp("2025-11-01"))
    assert completion.recommended_wait is not None  # maturity available
    assessment = assess_volume(
        canonical, table_config(), pd.Timestamp("2025-11-01"), completion
    )
    insufficient = [
        s for s in assessment.statuses
        if s.check is Check.VOLUME and s.severity is Severity.INSUFFICIENT_HISTORY
    ]
    assert insufficient and "evaluation window" in insufficient[0].detail
    assert ReasonCode.VOLUME_OUTLIER not in _reasons(assessment)


def test_empty_table_is_a_hard_red_not_a_crash():
    empty = g.TableSpec(
        name="events", start_month="2025-01", n_months=1, rows_per_month=0,
        lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=75,
    )
    canonical, assessment = _assess(g.generate(empty), table_config(), "2025-06-01")
    zero = [s for s in assessment.statuses if s.reason is ReasonCode.ZERO_ROW_MONTH]
    assert zero and zero[0].severity is Severity.RED
    assert assessment.months == []
    assert int(canonical.global_row["row_count"]) == 0


# ------------------------------------------------- hand-calculated boundaries
# baseline volumes are exactly 2000/month, MAD = 0, so sigma floors at
# 0.05 * 2000 = 100: amber edge = |dev| > 200, red edge = |dev| > 300.

BOUNDARY = g.TableSpec(
    name="events", start_month="2023-01", n_months=30, rows_per_month=2000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8), seed=73,
)


def test_amber_red_boundaries_hand_calculated():
    # idx 18 (2024-07): 2000*1.10  = 2200, dev exactly 200 -> NOT flagged
    # idx 19 (2024-08): 2000*1.105 = 2210, dev 210 -> AMBER
    # idx 20 (2024-09): 2000*1.155 = 2310, dev 310 -> RED
    spec = BOUNDARY
    spec = g.volume_spike(spec, 18, factor=1.10)
    spec = g.volume_spike(spec, 19, factor=1.105)
    spec = g.volume_spike(spec, 20, factor=1.155)
    _, assessment = _assess(g.generate(spec), table_config(), AS_OF_MATURE)
    outliers = {
        s.detail.split(":")[0]: s.severity
        for s in assessment.statuses
        if s.reason is ReasonCode.VOLUME_OUTLIER
    }
    assert "2024-07" not in outliers  # exactly at the boundary is inside
    assert outliers["2024-08"] is Severity.AMBER
    assert outliers["2024-09"] is Severity.RED


def test_one_low_month_is_an_outlier_two_are_a_collapse():
    single = g.volume_drop(BOUNDARY, 29, factor=0.1)  # last mature month only
    _, assessment = _assess(g.generate(single), table_config(), AS_OF_MATURE)
    assert ReasonCode.VOLUME_OUTLIER in _reasons(assessment)
    assert ReasonCode.VOLUME_COLLAPSE not in _reasons(assessment)

    double = g.volume_drop(g.volume_drop(BOUNDARY, 28, factor=0.1), 29, factor=0.1)
    _, assessment = _assess(g.generate(double), table_config(), AS_OF_MATURE)
    collapse = [s for s in assessment.statuses if s.reason is ReasonCode.VOLUME_COLLAPSE]
    assert collapse and "2025-05" in collapse[0].detail and "2025-06" in collapse[0].detail
    # collapse months are not double-reported as outliers
    assert ReasonCode.VOLUME_OUTLIER not in _reasons(assessment)


def test_collapse_run_must_end_at_the_latest_mature_month():
    # two consecutive low months followed by healthy ones: a PAST dip is
    # outliers, not an ongoing collapse
    spec = g.volume_drop(g.volume_drop(BOUNDARY, 24, factor=0.1), 25, factor=0.1)
    _, assessment = _assess(g.generate(spec), table_config(), AS_OF_MATURE)
    assert ReasonCode.VOLUME_COLLAPSE not in _reasons(assessment)
    outliers = [s for s in assessment.statuses if s.reason is ReasonCode.VOLUME_OUTLIER]
    assert len(outliers) == 2


def test_a_calendar_gap_breaks_the_collapse_run():
    # low April, MISSING May, low June: nonconsecutive low months must never
    # read as a sustained collapse — they are a gap plus two outliers
    spec = g.volume_drop(g.volume_drop(BOUNDARY, 27, factor=0.05), 29, factor=0.05)
    spec = g.missing_month(spec, 28)
    _, assessment = _assess(g.generate(spec), table_config(), "2026-08-01")
    reasons = _reasons(assessment)
    assert ReasonCode.VOLUME_GAP in reasons  # 2025-05
    assert ReasonCode.VOLUME_COLLAPSE not in reasons
    outliers = [s for s in assessment.statuses if s.reason is ReasonCode.VOLUME_OUTLIER]
    assert len(outliers) == 2


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
    # band dispersion uses the MATURE final volumes (ALGORITHMS section 8):
    # all mature volumes are exactly 2000 -> sigma floors at 100, so the band
    # width is 2 * band_mads * F * 100 = 400 * F
    fill = ((june.expected_low + june.expected_high) / 2) / unhealthy.forecast
    expected_width = 2 * 2.0 * fill * robust_sigma_floor([2000] * 10)
    assert june.expected_high - june.expected_low == pytest.approx(expected_width)
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


def test_uniqueness_check_runs_even_without_eligible_months():
    # a populated table whose rows are ALL excluded from the curves (every
    # event_time NULL) still carries duplicate keys: the global distinct-key
    # guard must be read BEFORE the empty-months early exit
    df = catalog()["duplicate_keys"].unhealthy()
    df = df.copy()
    df["event_time"] = pd.NaT
    config = table_config(key_cols=["row_id"])
    _, assessment = _assess(df, config, AS_OF_MATURE)
    assert assessment.duplicate_rows == round(0.02 * 30 * 2000)
    assert {ReasonCode.DUPLICATE_KEYS, ReasonCode.ZERO_ROW_MONTH} <= _reasons(assessment)


def test_short_three_month_collapse_with_a_configured_horizon():
    # The Step 4 spec's SHORT case: only the last 3 months are degraded. The
    # default 365d horizon can never call any of them mature, so the pair is
    # probed with lag_cap 52 / cutoff 54 (covering the ~50d batch lag support):
    # 2024-10 and 2024-11 are mature -> RED collapse; 2024-12 is immature ->
    # arrival deficit, never collapse; the feed stays freshness=GREEN.
    pair = catalog()["sustained_collapse_short"]
    as_of = "2025-01-25"  # 4 days after the last batch (2024-12's +20d batch)
    config = table_config(
        load_batch_col="batch_id",
        analysis={"lag_cap_days": 52, "training_cutoff_days": 54},
    )
    canonical, unhealthy = _assess(pair.unhealthy(), config, as_of)
    collapse = [s for s in unhealthy.statuses if s.reason is ReasonCode.VOLUME_COLLAPSE]
    assert len(collapse) == 1 and collapse[0].severity is Severity.RED
    for month in ("2024-10", "2024-11"):
        assert month in collapse[0].detail
    assert "2024-12" not in collapse[0].detail
    deficit = [s for s in unhealthy.statuses if s.reason is ReasonCode.ARRIVAL_DEFICIT]
    assert deficit and "2024-12" in deficit[0].detail
    # baseline excludes the evaluation window: the median holds at 2000
    assert unhealthy.baseline_median == pytest.approx(2000, rel=0.05)
    fresh = assess_freshness(canonical, config, pd.Timestamp(as_of))
    assert worst_severity(fresh.statuses) is Severity.GREEN

    _, healthy = _assess(pair.healthy(), config, as_of)
    assert _reasons(healthy) == set()
    healthy_fresh = assess_freshness(
        probe(pair.healthy(), config, as_of), config, pd.Timestamp(as_of)
    )
    assert worst_severity(healthy_fresh.statuses) is Severity.GREEN


def test_key_column_metadata_match_is_case_insensitive():
    # identifiers resolve case-insensitively in SQL, so a column created as
    # OrderRef configured as orderref must probe, not abort as missing
    df = catalog()["duplicate_keys"].healthy().rename(columns={"row_id": "OrderRef"})
    config = table_config(key_cols=["orderref"])
    _, assessment = _assess(df, config, AS_OF_MATURE)
    assert assessment.duplicate_rows == 0
