"""The shared presentation-data transformation (CLAUDE.md hard rule).

Small-count suppression is applied HERE, before any serialization: a
suppressed value must not survive into Plotly JSON, hover payloads, markdown
tables, or SVG internals. Figures receive count columns with values below the
threshold already blanked to NA (a gap in the chart, nothing in the hover);
tables render them through display_count() as "<5".

Everything the report and the dashboard show comes through load_run_frames /
frames_for_probe, so there is exactly ONE place where raw counts can leak —
and the render test suites inspect the generated HTML/SVG/markdown for it.
"""

from __future__ import annotations

import pandas as pd

SUPPRESSION_THRESHOLD = 5

# count-valued columns per snapshot table — INCLUDING count-SCALE model
# estimates (expected fill, forecast, baselines): a 2.7-row expected band
# discloses the same small magnitude the raw count would. Percentile days,
# fractions and dates are not counts.
SUPPRESSIBLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "month_volumes": ("volume", "expected_low", "expected_high", "nowcast"),
    "month_lag_cells": ("row_count",),
    "epoch_cells": ("row_count",),
    "dual_lag_cells": ("row_count",),
    "dual_delta": ("row_count",),
    "batch_runs": ("rows",),
    "batch_months": ("null_batch_rows",),
    "parity_months": ("left_count", "right_count", "diff"),
    "compare_mismatch": ("mismatches",),
    "volume_summary": ("duplicate_rows", "baseline_median", "baseline_sigma", "forecast"),
    "dual_summary": ("n_null_source_only", "n_delta_rows"),
    "population_buckets": (
        "row_count",
        "n_curve_eligible",
        "n_null_event_time",
        "n_null_load_time_only",
        "n_negative_clipped",
        "n_negative_lag_excluded",
        "n_overflow",
        "n_join_unmatched",
        "n_other_exclusions",
        "n_base_rows",
        "n_ambiguous_base_rows",
        "n_compare_mismatch",
        "distinct_keys",
    ),
}

# every snapshot table a run may hold (probe-scoped ones carry a `probe`
# column); readers tolerate absent tables — a probe without batches simply
# has no batch frames
RUN_TABLES = (
    "statuses",
    "probe_runs",
    "population_buckets",
    "completion_summary",
    "volume_summary",
    "completion_percentiles",
    "month_volumes",
    "month_lag_cells",
    "epoch_cells",
    "freshness",
    "batch_runs",
    "batch_months",
    "dual_summary",
    "dual_lag_cells",
    "dual_percentiles",
    "dual_delta",
    "compare_mismatch",
    "parity_months",
)


def suppress_small_counts(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Blank every suppressible count below the threshold to NA. Applied
    BEFORE any figure or table is built, so the raw value cannot reach any
    serialized artifact."""
    out = frame.copy()
    for column in SUPPRESSIBLE_COLUMNS.get(name, ()):
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            out[column] = values.mask(values < SUPPRESSION_THRESHOLD)
    return out


def display_count(value, suppressed: bool) -> str:
    """Table rendering of a count: '<5' when suppression blanked it (NA under
    an enabled flag), the plain integer otherwise, '—' for genuinely absent."""
    if value is None or pd.isna(value):
        return f"<{SUPPRESSION_THRESHOLD}" if suppressed else "—"
    return str(int(value))


def load_run_frames(store, run_id: str) -> dict[str, pd.DataFrame]:
    """Every snapshot table of a committed run. The store SAYS which tables
    the run holds (table_names); an absent table is skipped, but a READ
    FAILURE on a present one propagates — a corrupt file or a database outage
    must fail the render, never publish an incomplete dashboard."""
    present = set(store.table_names(run_id))
    frames: dict[str, pd.DataFrame] = {}
    for name in RUN_TABLES:
        if name in present:
            frames[name] = store.read_table(run_id, name)
    return frames


def probes_in(frames: dict[str, pd.DataFrame]) -> list[str]:
    runs = frames.get("probe_runs")
    if runs is None or runs.empty:
        return []
    return list(runs["probe"])


def frames_for_probe(
    frames: dict[str, pd.DataFrame], probe: str, suppress: bool
) -> dict[str, pd.DataFrame]:
    """One probe's slice of every table, suppression applied per its flag."""
    out: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        if "probe" not in frame.columns:
            continue
        subset = frame[frame["probe"] == probe].reset_index(drop=True)
        if subset.empty:
            continue
        out[name] = suppress_small_counts(name, subset) if suppress else subset
    return out
