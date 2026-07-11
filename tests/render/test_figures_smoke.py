"""Viz smoke (PLAN Step 9): every figure builds for EVERY scenario — batchy
and missing-month included — and suppression keeps small counts out of the
serialized figure payloads."""

from __future__ import annotations

import json

from metricprobe.viz.figures import FIGURE_ORDER, figures_for_probe
from metricprobe.viz.presentation import (
    display_count,
    frames_for_probe,
    load_run_frames,
    probes_in,
)


def _config_by_probe(config):
    return {table.probe_name: table for table in config.tables}


def test_every_figure_builds_for_every_scenario(dashboard_run):
    store, run_id, config = dashboard_run
    frames = load_run_frames(store, run_id)
    tables = _config_by_probe(config)
    failures = []
    built: dict[str, set[str]] = {}
    for probe in probes_in(frames):
        table = tables[probe]
        probe_frames = frames_for_probe(frames, probe, table.suppress_small_counts)
        try:
            figures = figures_for_probe(
                probe_frames, probe, proxy=table.proxy, expect_batchy=table.expect_batchy
            )
            for key, figure in figures.items():
                payload = figure.to_dict()  # serializes exactly like the report
                assert payload["data"], f"{probe}/{key} produced an empty figure"
            built[probe] = set(figures)
        except Exception as error:  # accumulate ALL failures, never die first
            failures.append(f"{probe}: {type(error).__name__}: {error}")
    assert not failures, "; ".join(failures)

    # the applicable figure set matches each probe's configuration
    assert {"volume", "completion_curves", "completion_heatmap", "percentiles"} <= built[
        "volume_spike_ok_probe"
    ]
    assert "batch" in built["straggler_batch_bad_probe"]
    assert "batch" in built["sustained_collapse_ok_probe"]
    assert {"dual_overlay", "dual_delta"} <= built["dual_registry_probe"]
    assert "parity" in built["parity_a_probe"]
    assert "volume" in built["missing_month_bad_probe"]
    for key in FIGURE_ORDER:
        assert any(key in keys for keys in built.values()), f"{key} never built"


def test_proxy_flag_labels_every_figure(dashboard_run):
    store, run_id, config = dashboard_run
    frames = load_run_frames(store, run_id)
    probe_frames = frames_for_probe(frames, "dual_registry_probe", False)
    figures = figures_for_probe(probe_frames, "dual_registry_probe", proxy=True)
    for key, figure in figures.items():
        assert "PROXY" in figure.layout.title.text, key


def test_suppression_blanks_small_counts_in_figure_payloads(dashboard_run):
    """The tiny probe's real counts (1..4 rows per cell) must not survive
    into any trace of any figure — suppression happens BEFORE serialization."""
    store, run_id, config = dashboard_run
    frames = load_run_frames(store, run_id)
    probe_frames = frames_for_probe(frames, "tiny_probe", suppress=True)
    figures = figures_for_probe(probe_frames, "tiny_probe")
    assert figures  # the probe still renders — with gaps, not values
    offenders = []
    for key, figure in figures.items():
        for trace in json.loads(figure.to_json())["data"]:
            for value in trace.get("y") or []:
                if value is not None and 0 < float(value) < 5 and key == "volume":
                    offenders.append(f"{key}: {value}")
    assert not offenders, offenders
    # and the shared table renderer spells the suppressed value
    assert display_count(None, suppressed=True) == "<5"
    assert display_count(7, suppressed=True) == "7"
    assert display_count(None, suppressed=False) == "—"
