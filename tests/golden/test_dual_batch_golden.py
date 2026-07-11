"""Step 5 goldens: dual lag (metric 3), batch metrics (metric 4), the
compare_event_time side-stat, and the "complete back to" headline date."""

import dataclasses
import math
import statistics

import pandas as pd
import pytest
from tests.support import probe, probe_dual, table_config
from tests.synth import generator as g
from tests.synth.scenarios import catalog

from metricprobe.metrics.batch import assess_batch
from metricprobe.metrics.completion import (
    assess_completion,
    compare_mismatch_by_month,
    complete_back_to,
)
from metricprobe.metrics.dual_lag import assess_dual_lag
from metricprobe.status import ReasonCode, Severity

AS_OF = "2026-07-01"

DUAL = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=12,
    rows_per_month=20_000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8),
    dual_offset_days=2.0,  # the known source->load offset
    seed=81,
)

TOLERANCE_DAYS = {50: 1, 90: 1, 95: 1, 99: 2}  # same rationale as completion goldens


def _dual_config(**overrides):
    return table_config(source_insert_time="source_insert_time", **overrides)


# ----------------------------------------------------------------------- dual


def test_dual_source_percentiles_recovered_from_known_offset():
    config = _dual_config()
    dual = probe_dual(g.generate(DUAL), config, AS_OF)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    expected = g.expected_days_to_percentiles(DUAL, timestamp="source")
    failures = []
    for month, per_pct in assessment.source_percentiles.items():
        for pct, got in per_pct.items():
            want = expected[str(month)][pct]
            if got.over_cap or abs(got.value - want) > TOLERANCE_DAYS[pct]:
                failures.append(f"{month} p{pct}: expected {want}, got {got}")
    assert not failures, "; ".join(failures)


def test_dual_delta_histogram_is_the_exact_offset():
    config = _dual_config()
    df = g.generate(DUAL)
    dual = probe_dual(df, config, AS_OF)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    # load = source + exactly 2.0 days: every delta-eligible row sits at day 2
    assert list(assessment.delta_histogram.index) == [2]
    assert int(assessment.delta_histogram.loc[2]) == len(df)
    assert assessment.n_delta_rows == len(df)


def test_dual_null_source_rows_form_their_own_bucket():
    config = _dual_config()
    df = g.inject_null_source_insert(g.generate(DUAL), fraction=0.03, seed=1)
    injected = round(0.03 * len(g.generate(DUAL)))
    dual = probe_dual(df, config, AS_OF)
    assessment = assess_dual_lag(dual, config, pd.Timestamp(AS_OF))
    assert assessment.n_null_source_only == injected
    # dual reconciliation: total = eligible + null_event + null_source + excluded
    row = dual.global_row
    assert int(row["row_count"]) == (
        int(row["n_source_eligible"])
        + int(row["n_null_event_time"])
        + int(row["n_null_source_only"])
        + int(row["n_negative_lag_excluded"])
    )
    assert not any(s.severity is Severity.RED for s in assessment.statuses)


# ---------------------------------------------------------------------- batch


def test_batch_completion_hand_calculated_from_the_schedule():
    # schedule (3d, 60%), (10d, 30%), (20d, 10%) after month end:
    # cumulative .6 / .9 / 1.0 -> p50=3, p90=10, p95=20, p99=20 for EVERY month
    config = table_config(load_batch_col="batch_id")
    canonical = probe(catalog()["straggler_batch"].healthy(), config, AS_OF)
    assessment = assess_batch(canonical, config)
    for month_row in assessment.months:
        assert month_row.runs == 3
        assert month_row.days_to[50] == 3
        assert month_row.days_to[95] == 20 and month_row.days_to[99] == 20
        # p90 sits EXACTLY at the .6+.3 cumulative boundary: multinomial
        # assignment makes it genuinely bistable between the 10d and 20d batch
        assert month_row.days_to[90] in (10, 20)
        # rows per run are weighted by actual batch sizes (~fractions x 2000)
        sizes = sorted(month_row.rows_per_run.values(), reverse=True)
        assert sizes[0] == pytest.approx(1200, rel=0.15)
        assert sum(sizes) == 2000


def test_straggler_batch_shifts_batch_level_completion():
    config = table_config(load_batch_col="batch_id")
    canonical = probe(catalog()["straggler_batch"].unhealthy(), config, AS_OF)
    assessment = assess_batch(canonical, config)
    by_month = {m.month: m for m in assessment.months}
    hit = by_month[pd.Period("2024-11", freq="M")]
    # 15% of the month rides the 45-day straggler: cum ~.51/.765/.85/1.0
    assert hit.runs == 4
    assert hit.days_to[50] in (3, 10)  # cum at 3d is ~.51: within 1 sigma of .5
    assert hit.days_to[90] == 45 and hit.days_to[95] == 45
    clean = by_month[pd.Period("2024-10", freq="M")]
    assert clean.days_to[50] == 3 and clean.days_to[95] == 20
    assert clean.days_to[90] in (10, 20)  # the exact-boundary bistability again


# ----------------------------------------------------- compare_event_time stat


def test_compare_event_time_side_stat():
    pair = catalog()["raw_vs_corrected"]
    config = table_config(compare_event_time="event_time_raw")
    healthy = compare_mismatch_by_month(probe(pair.healthy(), config, AS_OF))
    assert sum(healthy.values()) == 0
    unhealthy_df = pair.unhealthy()
    unhealthy = compare_mismatch_by_month(probe(unhealthy_df, config, AS_OF))
    # day-grain mismatches among eligible rows, by month, summing to the total
    expected_total = int(
        (
            unhealthy_df["event_time_raw"].dt.normalize()
            != unhealthy_df["event_time"].dt.normalize()
        ).sum()
    )
    assert sum(unhealthy.values()) == expected_total
    assert expected_total == round(0.08 * 30 * 2000)


# ------------------------------------------------------------------- headline


def test_complete_back_to_ties_to_the_completion_percentiles():
    spec = dataclasses.replace(DUAL, dual_offset_days=None, seed=82)
    config = table_config()
    canonical = probe(g.generate(spec), config, AS_OF)
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    assert assessment.recommended_wait is not None
    # the headline formula, recomputed independently from the percentiles
    p95s = [assessment.percentiles[m][95].value for m in assessment.mature_months]
    expected_wait = math.ceil(statistics.fmean(p95s) + 2 * statistics.pstdev(p95s))
    assert assessment.recommended_wait == expected_wait
    date = complete_back_to(assessment, pd.Timestamp(AS_OF))
    assert date == pd.Timestamp(AS_OF) - pd.Timedelta(days=expected_wait)
    # and it is deliberately DISTINCT from the (stricter) maturity horizon
    assert assessment.horizon >= expected_wait


def test_complete_back_to_is_refused_when_the_wait_is():
    config = table_config(analysis={"lag_cap_days": 15})  # censors p95
    spec = dataclasses.replace(DUAL, dual_offset_days=None, seed=82)
    canonical = probe(g.generate(spec), config, AS_OF)
    assessment = assess_completion(canonical, config, pd.Timestamp(AS_OF))
    assert assessment.recommended_wait is None
    assert complete_back_to(assessment, pd.Timestamp(AS_OF)) is None
    assert any(s.reason is ReasonCode.PERCENTILE_OVER_CAP for s in assessment.statuses)
