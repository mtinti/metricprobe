"""Robust statistics and shared formulas, exactly as pinned in
docs/ALGORITHMS.md sections 1, 3 and 8. Hand-calculated boundary tests in
tests/unit/test_algorithms.py enforce every formula here."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

MAD_TO_SIGMA = 1.4826  # normal-consistency scaling (ALGORITHMS.md section 1)


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation: median(|x - median(x)|)."""
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def robust_sigma(values: Sequence[float]) -> float:
    """1.4826 * MAD — comparable to a standard deviation under normality."""
    return MAD_TO_SIGMA * mad(values)


def robust_sigma_or(values: Sequence[float], fallback: float) -> float:
    """Zero-MAD fallback: perfectly regular values get the explicit fallback
    tolerance instead of a zero band (ALGORITHMS.md section 1)."""
    sigma = robust_sigma(values)
    return sigma if sigma > 0 else fallback


def recommended_wait(days_to_p95: Sequence[float]) -> int:
    """ceil(mean + 2 * population stdev) over per-month days-to-p95 values
    (ALGORITHMS.md section 3). Callers must refuse the wait (never call this)
    when any month's p95 is over-cap."""
    if not days_to_p95:
        raise ValueError("recommended_wait requires at least one month")
    mean = statistics.fmean(days_to_p95)
    spread = statistics.pstdev(days_to_p95)
    return math.ceil(mean + 2 * spread)


def expected_fill_band(
    forecast: float,
    fill_fraction: float,
    mature_volumes: Sequence[float],
    band_mads: float,
    zero_mad_fallback: float,
) -> tuple[float, float]:
    """Expected-count band for a month at fill fraction F(t), including the
    forecast dispersion scaled by the fill fraction (ALGORITHMS.md section 8):
        expected = forecast * F(t)
        sigma_band = F(t) * robust_sigma_or(mature final volumes, fallback)
        band = expected +- band_mads * sigma_band
    """
    expected = forecast * fill_fraction
    sigma_band = fill_fraction * robust_sigma_or(mature_volumes, zero_mad_fallback)
    half_width = band_mads * sigma_band
    return expected - half_width, expected + half_width
