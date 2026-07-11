"""Robust statistics and shared formulas, exactly as pinned in
docs/ALGORITHMS.md sections 1, 3 and 8. Hand-calculated boundary tests in
tests/unit/test_algorithms.py enforce every formula here."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

MAD_TO_SIGMA = 1.4826  # normal-consistency scaling (ALGORITHMS.md section 1)
DEFAULT_REL_TOL = 0.05  # v1 relative floor where no explicit tolerance is configured


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation: median(|x - median(x)|)."""
    center = statistics.median(values)
    return statistics.median([abs(value - center) for value in values])


def robust_sigma(values: Sequence[float]) -> float:
    """1.4826 * MAD — comparable to a standard deviation under normality."""
    return MAD_TO_SIGMA * mad(values)


def robust_sigma_floor(values: Sequence[float], rel_tol: float = DEFAULT_REL_TOL) -> float:
    """The frozen zero-MAD fallback (CLAUDE.md / ALGORITHMS.md section 1):
        max(scaled MAD, rel_tol * median)
    A relative FLOOR, not an exact-zero special case: perfectly regular AND
    nearly regular values both get at least rel_tol of their median as spread.
    """
    return max(robust_sigma(values), rel_tol * abs(statistics.median(values)))


def recommended_wait(days_to_p95: Sequence[float]) -> int:
    """ceil(mean + 2 * population stdev) over per-month days-to-p95 values
    (ALGORITHMS.md section 3). Callers must refuse the wait (never call this)
    when any month's p95 is over-cap."""
    if not days_to_p95:
        raise ValueError("recommended_wait requires at least one month")
    mean = statistics.fmean(days_to_p95)
    spread = statistics.pstdev(days_to_p95)
    return math.ceil(mean + 2 * spread)


def median_of_curves(curves: Sequence[Sequence[float]]) -> list[float]:
    """Pointwise median across per-month curves on a shared grid — the
    median-of-curves F_mature (ALGORITHMS.md section 8): robust to one weird
    month, never pooled rows."""
    if not curves:
        raise ValueError("median_of_curves requires at least one curve")
    length = len(curves[0])
    if any(len(curve) != length for curve in curves):
        raise ValueError("all curves must share the same grid")
    return [statistics.median([curve[i] for curve in curves]) for i in range(length)]


def expected_fill_band(
    forecast: float,
    fill_fraction: float,
    mature_volumes: Sequence[float],
    band_mads: float,
    rel_tol: float = DEFAULT_REL_TOL,
) -> tuple[float, float]:
    """Expected-count band for a month at fill fraction F(t), including the
    forecast dispersion scaled by the fill fraction (ALGORITHMS.md section 8):
        expected = forecast * F(t)
        sigma_band = F(t) * robust_sigma_floor(mature final volumes, rel_tol)
        band = expected +- band_mads * sigma_band
    """
    expected = forecast * fill_fraction
    sigma_band = fill_fraction * robust_sigma_floor(mature_volumes, rel_tol)
    half_width = band_mads * sigma_band
    return expected - half_width, expected + half_width
