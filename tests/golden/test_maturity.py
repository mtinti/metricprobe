"""Maturity is single-pass and EXPOSURE-based; the lag-support backtest is a
heuristic that detects strata disagreement — and its blind spot is documented
by a test that proves it CANNOT detect a shared late tail."""


import pandas as pd
from tests.support import probe, table_config
from tests.synth import generator as g

from metricprobe.metrics.completion import assess_completion, build_month_curves
from metricprobe.status import ReasonCode, Severity

SHORT_LAG = g.LognormalLag(mu=1.6, sigma=0.8)  # p99 ~32d, far inside every cutoff


def _assess(df, config, as_of):
    canonical = probe(df, config, as_of)
    return canonical, assess_completion(canonical, config, pd.Timestamp(as_of))


def test_exposure_classifies_maturity_never_curve_shape():
    # PLAN: a month with only ~10% of its rows arrived, whose self-normalized
    # curve nevertheless ALREADY reads 100% at its largest observed lag, must
    # still be classified immature — classification is exposure-based only.
    spec = g.TableSpec(
        name="events", start_month="2023-01", n_months=30, rows_per_month=2000,
        lag_model=SHORT_LAG, seed=41,
    )
    as_of = "2025-06-07"  # 2025-06 events + lognormal lags: ~10% arrived by now
    df = g.generate(spec)
    config = table_config()
    june = df["event_time"].dt.to_period("M") == "2025-06"
    arrived_fraction = float(
        (june & (df["load_time"] <= pd.Timestamp(as_of))).sum() / june.sum()
    )
    assert 0.03 <= arrived_fraction <= 0.20, f"fixture drifted: {arrived_fraction:.1%}"
    canonical, assessment = _assess(df, config, as_of)
    young = pd.Period("2025-06", freq="M")
    curves = build_month_curves(canonical, config.analysis.lag_cap_days)
    observed_max_lag = int(curves[young].counts.index.max())
    assert curves[young].fraction_at(observed_max_lag) == 1.0  # "its own curve shows 100%"
    assert young not in assessment.mature_months  # ... and it is still immature
    # while the horizon itself is monotone: max(cutoff, learned_wait)
    assert assessment.horizon >= config.analysis.training_cutoff_days


def test_empty_training_cohort_is_insufficient_history_not_green():
    # 3 months of history, none ending before as_of - 365d: the cohort is
    # EMPTY. This must surface as insufficient history — never a silent GREEN.
    spec = g.TableSpec(
        name="events", start_month="2025-03", n_months=3, rows_per_month=2000,
        lag_model=SHORT_LAG, seed=48,
    )
    config = table_config()
    _, assessment = _assess(g.generate(spec), config, "2025-07-01")
    assert assessment.training_months == []
    assert assessment.learned_wait is None
    assert assessment.recommended_wait is None
    assert any(
        s.severity is Severity.INSUFFICIENT_HISTORY
        and s.reason is ReasonCode.INSUFFICIENT_MATURE_MONTHS
        for s in assessment.statuses
    )
    assert not any(s.severity is Severity.GREEN for s in assessment.statuses)


def test_too_few_mature_months_is_insufficient_history():
    spec = g.TableSpec(
        name="events", start_month="2024-06", n_months=8, rows_per_month=2000,
        lag_model=SHORT_LAG, seed=42,
    )
    config = table_config()
    _, assessment = _assess(g.generate(spec), config, "2025-08-01")
    # only ~2 months end before as_of - 365d: cohort exists but maturity is thin
    assert len(assessment.mature_months) < config.analysis.min_mature_months
    assert any(
        s.severity is Severity.INSUFFICIENT_HISTORY
        and s.reason is ReasonCode.INSUFFICIENT_MATURE_MONTHS
        for s in assessment.statuses
    )
    assert assessment.recommended_wait is None


def test_latency_drift_trips_the_strata_backtest():
    # Older history arrived fast (p95 ~7d); recent history arrives slow
    # (p95 ~90d). The two cohort strata learn waits differing far beyond
    # max(7d, 25%), so the backtest must yield insufficient history rather
    # than a confidently wrong wait.
    fast = g.TableSpec(
        name="events", start_month="2022-01", n_months=12, rows_per_month=2000,
        lag_model=g.LognormalLag(mu=1.0, sigma=0.6), seed=43,
    )
    slow = g.TableSpec(
        name="events", start_month="2023-01", n_months=12, rows_per_month=2000,
        lag_model=g.LognormalLag(mu=3.2, sigma=0.8), seed=44,
    )
    df = pd.concat([g.generate(fast), g.generate(slow)], ignore_index=True)
    df["row_id"] = range(len(df))  # re-key: the two specs overlap row_ids
    config = table_config()
    _, assessment = _assess(df, config, "2024-06-01")
    assert any(s.reason is ReasonCode.BACKTEST_DISAGREEMENT for s in assessment.statuses)
    assert assessment.learned_wait is None
    assert assessment.recommended_wait is None


def test_true_latency_beyond_cutoff_yields_insufficient_history():
    # PLAN scenario: real lag support exceeds training_cutoff_days (= lag_cap).
    # 8% of rows arrive 565 days after month end with cutoff = cap = 365. The
    # 13-month cohort's OLDER half (month ages >= 580d) has received that mass
    # as overflow: p95 censored past the cap. The YOUNGER half (ages <= 550d)
    # has not received it at all: p95 defined (~37d). One stratum censored
    # while the other is not IS a strata disagreement — never a confident
    # underestimated wait.
    late_tail = g.TableSpec(
        name="events", start_month="2023-01", n_months=13, rows_per_month=2000,
        lag_model=g.StepBatches(schedule=((3.0, 0.60), (10.0, 0.32), (565.0, 0.08))),
        seed=45,
    )
    config = table_config()
    _, assessment = _assess(g.generate(late_tail), config, "2025-02-01")
    reasons = {s.reason for s in assessment.statuses}
    assert ReasonCode.BACKTEST_DISAGREEMENT in reasons or ReasonCode.PERCENTILE_OVER_CAP in reasons
    assert assessment.recommended_wait is None
    # the disagreement path must actually be exercised by this scenario
    assert ReasonCode.BACKTEST_DISAGREEMENT in reasons


def test_shared_late_tail_is_the_backtests_documented_blind_spot():
    # BLIND SPOT (documented in ALGORITHMS.md section 5): 3% of every month's
    # rows arrive at day ~1200 — beyond the age of even the OLDEST cohort month
    # (~900d), so the mass is invisible to every stratum equally. Both halves
    # agree, the backtest stays silent, and the wait is confidently wrong.
    # This test EXPECTS the non-detection: proving completeness would need an
    # external finality signal the data cannot provide.
    # 10% of every month rides the invisible tail, so the TRUE p95 sits at
    # day ~1200+ (visible mass reaches only 90% < 95%). The observed, self-
    # normalized curves complete at ~41d and yield a ~39-day wait — a genuinely
    # UNDERESTIMATED p95, which the backtest cannot see because every stratum
    # is censored identically.
    shared_tail = g.TableSpec(
        name="events", start_month="2023-01", n_months=30, rows_per_month=2000,
        lag_model=g.StepBatches(schedule=((3.0, 0.60), (10.0, 0.30), (1200.0, 0.10))),
        seed=46,
    )
    config = table_config()
    _, assessment = _assess(g.generate(shared_tail), config, "2026-01-01")
    reasons = {s.reason for s in assessment.statuses}
    assert ReasonCode.BACKTEST_DISAGREEMENT not in reasons
    assert ReasonCode.PERCENTILE_OVER_CAP not in reasons
    # true p95 (with the tail visible) would be ~1200d; the confident answer
    # is under 60d — wrong by a factor of ~20, and undetectably so
    assert assessment.recommended_wait is not None
    assert assessment.recommended_wait < 60


def test_healthy_twin_passes_the_backtest():
    healthy = g.TableSpec(
        name="events", start_month="2022-01", n_months=30, rows_per_month=2000,
        lag_model=SHORT_LAG, seed=47,
    )
    config = table_config()
    _, assessment = _assess(g.generate(healthy), config, "2025-06-01")
    assert not any(s.reason is ReasonCode.BACKTEST_DISAGREEMENT for s in assessment.statuses)
    assert assessment.learned_wait is not None
    assert assessment.recommended_wait is not None
    assert assessment.horizon == max(
        config.analysis.training_cutoff_days, assessment.learned_wait
    )
