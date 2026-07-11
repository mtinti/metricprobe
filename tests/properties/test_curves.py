"""Property tests: invariants that must hold for EVERY scenario — curve
monotonicity (and 100% at the max observed lag, BY CONSTRUCTION — which is why
curve shape is never a censoring signal), row-order invariance, and the
population reconciliation equation from the ONE canonical aggregation."""

import numpy as np
import pandas as pd
import pytest
from tests.support import probe, table_config
from tests.synth import generator as g
from tests.synth.scenarios import catalog

from metricprobe.metrics.completion import assess_completion, build_month_curves

AS_OF = "2026-07-01"

SCENARIO_FRAMES = {
    "trickle_healthy": lambda: catalog()["volume_spike"].healthy(),
    "batchy_straggler": lambda: catalog()["straggler_batch"].unhealthy(),
    "sustained_collapse": lambda: catalog()["sustained_collapse"].unhealthy(),
    "missing_month": lambda: catalog()["missing_month"].unhealthy(),
    "with_nulls_and_negatives": lambda: g.inject_null_load_time(
        g.inject_null_event_time(
            g.inject_negative_lags(catalog()["volume_spike"].healthy(), 0.02, 5.0, seed=9),
            0.02,
            seed=9,
        ),
        0.02,
        seed=9,
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIO_FRAMES))
def test_curves_monotone_and_complete_at_max_lag(name):
    config = table_config()
    canonical = probe(SCENARIO_FRAMES[name](), config, AS_OF)
    curves = build_month_curves(canonical, config.analysis.lag_cap_days)
    assert curves, name
    for month, curve in curves.items():
        ordered = curve.counts.sort_index()
        cum = (ordered.cumsum() / curve.final_count).to_numpy()
        assert np.all(np.diff(cum) >= 0), f"{name} {month}: curve not monotone"
        assert cum[-1] == pytest.approx(1.0), f"{name} {month}: does not reach 100%"


@pytest.mark.parametrize("name", sorted(SCENARIO_FRAMES))
def test_reconciliation_equation_holds(name):
    config = table_config()
    row = probe(SCENARIO_FRAMES[name](), config, AS_OF).global_row
    buckets = (
        int(row["n_curve_eligible"])
        + int(row["n_null_event_time"])
        + int(row["n_null_load_time_only"])
        + int(row["n_negative_lag_excluded"])
        + int(row["n_join_unmatched"])
        + int(row["n_other_exclusions"])
    )
    assert int(row["row_count"]) == buckets, name
    assert int(row["n_base_rows"]) == int(row["row_count"]), name  # pre/post join


def test_row_order_invariance():
    config = table_config()
    df = catalog()["straggler_batch"].unhealthy()
    shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
    sort_keys = ["grouping_id", "event_month", "lag_day", "load_epoch_day", "batch_id"]

    def canonical_sorted(frame):
        out = probe(frame, config, AS_OF).frame
        return out.sort_values(sort_keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(canonical_sorted(df), canonical_sorted(shuffled))
    original = assess_completion(probe(df, config, AS_OF), config, pd.Timestamp(AS_OF))
    reordered = assess_completion(probe(shuffled, config, AS_OF), config, pd.Timestamp(AS_OF))
    assert original.percentiles == reordered.percentiles
    assert original.recommended_wait == reordered.recommended_wait
