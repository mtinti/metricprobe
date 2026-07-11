"""Hand-calculated boundary tests for every formula in docs/ALGORITHMS.md.

Each expected value below is worked out by hand in a comment — these tests pin
the formulas themselves, independent of any pipeline code."""

import pytest

from metricprobe.metrics.completion import Percentile, percentile_summary
from metricprobe.metrics.robust import (
    expected_fill_band,
    mad,
    median_of_curves,
    recommended_wait,
    robust_sigma,
    robust_sigma_floor,
)


def test_mad_hand_calculated():
    # median([1,2,3,4,5]) = 3; |x-3| = [2,1,0,1,2]; median = 1
    assert mad([1, 2, 3, 4, 5]) == 1.0
    # median([1,1,1,9]) = 1; |x-1| = [0,0,0,8]; median of [0,0,0,8] = 0
    assert mad([1, 1, 1, 9]) == 0.0


def test_robust_sigma_scaling():
    # robust_sigma = 1.4826 * MAD; MAD([1..5]) = 1
    assert robust_sigma([1, 2, 3, 4, 5]) == pytest.approx(1.4826)


def test_zero_mad_relative_floor():
    # the frozen formula is max(scaled MAD, rel_tol * median) — a FLOOR:
    # exactly-zero MAD: max(0, 0.05*7) = 0.35
    assert robust_sigma_floor([7, 7, 7, 7], rel_tol=0.05) == pytest.approx(0.35)
    # SMALL NONZERO MAD also gets the relative floor: sorted [9.8,10,10,10.1,10.2],
    # median 10, deviations sorted [0,0,.1,.2,.2] -> MAD .1 -> sigma .14826;
    # floor 0.05*10 = 0.5 wins
    assert robust_sigma_floor([10, 10, 10.2, 9.8, 10.1], rel_tol=0.05) == pytest.approx(0.5)
    # a healthy spread beats the floor: MAD([1..5])=1 -> 1.4826 > 0.05*3
    assert robust_sigma_floor([1, 2, 3, 4, 5], rel_tol=0.05) == pytest.approx(1.4826)


def test_recommended_wait_hand_calculated():
    # d95 = [10, 12, 14]: mean = 12; pstdev = sqrt((4+0+4)/3) = 1.63299...
    # wait = ceil(12 + 2*1.63299) = ceil(15.2660) = 16
    assert recommended_wait([10, 12, 14]) == 16
    # single month: pstdev = 0 -> wait = that month's p95
    assert recommended_wait([10]) == 10
    # exact integer boundary: [10, 14]: mean 12, pstdev 2 -> ceil(16.0) = 16
    assert recommended_wait([10, 14]) == 16


def test_recommended_wait_rejects_empty():
    with pytest.raises(ValueError):
        recommended_wait([])


def test_median_of_curves_hand_calculated():
    # pointwise median, NOT pooled rows: at each grid point take the median of
    # the three curves: [0.2,0.4,0.1] -> 0.2; [0.6,0.8,0.9] -> 0.8; [1,1,1] -> 1
    curves = [
        [0.2, 0.6, 1.0],
        [0.4, 0.8, 1.0],
        [0.1, 0.9, 1.0],
    ]
    assert median_of_curves(curves) == [0.2, 0.8, 1.0]
    with pytest.raises(ValueError):
        median_of_curves([])
    with pytest.raises(ValueError):
        median_of_curves([[0.1], [0.1, 0.2]])


def test_expected_fill_band_hand_calculated():
    # forecast=1000, F(t)=0.5 -> expected 500
    # sigma = robust_sigma_floor([900,1000,1100], 0.05):
    #   MAD=100 -> 148.26; floor 0.05*1000=50 -> 148.26 wins
    # sigma_band = 0.5*148.26 = 74.13; k=2 -> half-width 148.26
    low, high = expected_fill_band(
        forecast=1000,
        fill_fraction=0.5,
        mature_volumes=[900, 1000, 1100],
        band_mads=2.0,
        rel_tol=0.05,
    )
    assert low == pytest.approx(500 - 148.26)
    assert high == pytest.approx(500 + 148.26)


def test_expected_fill_band_uses_the_relative_floor():
    # constant volumes: MAD=0 -> sigma floors at 0.05*1000 = 50
    # sigma_band = 0.5*50 = 25; k=2 -> band 500 +- 50
    low, high = expected_fill_band(
        forecast=1000,
        fill_fraction=0.5,
        mature_volumes=[1000, 1000, 1000],
        band_mads=2.0,
        rel_tol=0.05,
    )
    assert (low, high) == (450.0, 550.0)


def test_mature_percentile_summary_hand_calculated():
    # p95 values [10, 12, 14] over three mature months:
    # mean 12, pstdev sqrt(8/3) = 1.63299
    months = ["a", "b", "c"]
    percentiles = {
        month: {pct: Percentile(value=value, over_cap=False) for pct in (50, 90, 95, 99)}
        for month, value in zip(months, [10, 12, 14], strict=True)
    }
    summary = percentile_summary(percentiles, months)
    mean, spread = summary[95]
    assert mean == pytest.approx(12.0)
    assert spread == pytest.approx(1.63299, abs=1e-5)
    # any over-cap month makes the summary undefined for that percentile
    percentiles["c"] = {pct: Percentile(value=None, over_cap=True) for pct in (50, 90, 95, 99)}
    assert percentile_summary(percentiles, months)[95] is None
    # no months -> undefined
    assert percentile_summary(percentiles, [])[95] is None
