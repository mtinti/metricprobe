"""Hand-calculated boundary tests for every formula in docs/ALGORITHMS.md.

Each expected value below is worked out by hand in a comment — these tests pin
the formulas themselves, independent of any pipeline code."""

import pytest

from metricprobe.metrics.robust import (
    expected_fill_band,
    mad,
    recommended_wait,
    robust_sigma,
    robust_sigma_or,
)


def test_mad_hand_calculated():
    # median([1,2,3,4,5]) = 3; |x-3| = [2,1,0,1,2]; median = 1
    assert mad([1, 2, 3, 4, 5]) == 1.0
    # median([1,1,1,9]) = 1; |x-1| = [0,0,0,8]; median of [0,0,0,8] = 0
    assert mad([1, 1, 1, 9]) == 0.0


def test_robust_sigma_scaling():
    # robust_sigma = 1.4826 * MAD; MAD([1..5]) = 1
    assert robust_sigma([1, 2, 3, 4, 5]) == pytest.approx(1.4826)


def test_zero_mad_fallback():
    # perfectly regular values: MAD = 0 -> the explicit fallback, never 0
    assert robust_sigma_or([7, 7, 7, 7], fallback=1.5) == 1.5
    # non-degenerate values ignore the fallback
    assert robust_sigma_or([1, 2, 3, 4, 5], fallback=99.0) == pytest.approx(1.4826)


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


def test_expected_fill_band_hand_calculated():
    # forecast=1000, F(t)=0.5 -> expected 500
    # sigma_band = F * robust_sigma(volumes); volumes [900,1000,1100]:
    #   median=1000, |x-1000|=[100,0,100], MAD=100, sigma=148.26
    #   sigma_band = 0.5 * 148.26 = 74.13; k=2 -> half-width 148.26
    low, high = expected_fill_band(
        forecast=1000,
        fill_fraction=0.5,
        mature_volumes=[900, 1000, 1100],
        band_mads=2.0,
        zero_mad_fallback=10.0,
    )
    assert low == pytest.approx(500 - 148.26)
    assert high == pytest.approx(500 + 148.26)


def test_expected_fill_band_uses_zero_mad_fallback():
    # constant volumes: MAD=0 -> sigma falls back to 10.0
    # sigma_band = 0.5 * 10 = 5; k=2 -> band 500 +- 10
    low, high = expected_fill_band(
        forecast=1000,
        fill_fraction=0.5,
        mature_volumes=[1000, 1000, 1000],
        band_mads=2.0,
        zero_mad_fallback=10.0,
    )
    assert (low, high) == (490.0, 510.0)
