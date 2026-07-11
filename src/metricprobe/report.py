"""Static report: ONE self-contained HTML file (Plotly.js embedded, ZERO
external URLs — it must open on an offline machine) plus per-figure PNGs.

Static image export uses Plotly's kaleido, which with current Plotly requires
an installed Chrome/Chromium. ensure_static_export_available() verifies the
whole chain at startup by actually exporting a tiny figure, and fails with an
actionable error instead of a deep traceback halfway through a render. On
locked-down Windows machines without admin rights, `kaleido_get_chrome`
downloads a private Chromium into the user profile — no system install needed.
"""

from __future__ import annotations

import html
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from metricprobe.config import ProbeConfig
from metricprobe.viz.figures import figures_for_probe
from metricprobe.viz.presentation import frames_for_probe, load_run_frames, probes_in

EXPORT_INSTALL_HELP = (
    "static image export needs the kaleido package AND an installed "
    "Chrome/Chromium.\n"
    '  1. pip install "metricprobe[export]"   (installs kaleido)\n'
    "  2. ensure Chrome or Chromium is installed. On locked-down Windows "
    "machines without admin rights, run `kaleido_get_chrome` — it downloads "
    "a private Chromium into your user profile, no system install required."
)


def ensure_static_export_available() -> None:
    """Verify kaleido + Chrome by exporting a 1-trace figure. Called at
    report/publish startup so the failure is immediate and actionable."""
    try:
        import kaleido  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"metricprobe report/publish: kaleido is not installed.\n{EXPORT_INSTALL_HELP}"
        ) from exc
    try:
        pio.to_image(
            go.Figure(go.Bar(x=[0], y=[1])), format="png", width=32, height=32
        )
    except Exception as exc:
        raise RuntimeError(
            f"metricprobe report/publish: static export failed ({exc}).\n"
            f"{EXPORT_INSTALL_HELP}"
        ) from exc


def build_all_figures(
    store, run_id: str, config: ProbeConfig
) -> dict[str, dict[str, go.Figure]]:
    """probe -> {figure key -> Figure} for every probe of a committed run,
    through the shared presentation transform (suppression happens there)."""
    frames = load_run_frames(store, run_id)
    tables = {table.probe_name: table for table in config.tables}
    all_figures: dict[str, dict[str, go.Figure]] = {}
    for probe in probes_in(frames):
        table = tables.get(probe)
        suppress = bool(table and table.suppress_small_counts)
        probe_frames = frames_for_probe(frames, probe, suppress)
        all_figures[probe] = figures_for_probe(
            probe_frames,
            probe,
            proxy=bool(table and table.proxy),
            expect_batchy=bool(table and table.expect_batchy),
        )
    return all_figures


def _manifest_for(store, run_id: str) -> dict:
    for manifest in store.list_runs():
        if manifest["run_id"] == run_id:
            return manifest
    raise FileNotFoundError(f"run {run_id!r} is not committed")


def generate_report(
    store,
    run_id: str,
    config: ProbeConfig,
    out_dir: str | Path,
    png: bool = True,
) -> Path:
    """Write report.html (self-contained) and img/<probe>_<figure>.png files.
    Returns the report path. Static-export availability must have been
    verified by the caller (ensure_static_export_available)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_for(store, run_id)
    all_figures = build_all_figures(store, run_id, config)

    parts: list[str] = []
    parts.append(
        "<h1>metricprobe report</h1>\n"
        f"<p><b>Run:</b> {html.escape(manifest['run_id'])} · "
        f"<b>Generated at:</b> {html.escape(manifest['run_at'])} · "
        f"<b>as-of:</b> {html.escape(manifest['as_of'])} · "
        f"<b>git:</b> {html.escape(manifest['git_sha'])}<br>"
        f"<b>Analysed window:</b> {html.escape(manifest['window_start'])} → "
        f"{html.escape(manifest['window_end'])}</p>"
    )
    include_js: bool | str = True  # first figure embeds the bundled plotly.js
    exports: list[tuple[object, Path]] = []
    if png:
        (out / "img").mkdir(exist_ok=True)
    for probe, figures in all_figures.items():
        parts.append(f"<h2>{html.escape(probe)}</h2>")
        if not figures:
            parts.append("<p>no figures (probe skipped or aborted)</p>")
        for key, figure in figures.items():
            parts.append(
                pio.to_html(
                    figure,
                    full_html=False,
                    include_plotlyjs=include_js,
                    default_width="100%",
                    default_height="450px",
                )
            )
            include_js = False
            if png:
                exports.append((figure, out / "img" / f"{probe}_{key}.png"))
    if exports:
        # ONE batched kaleido session (a per-figure Chrome roundtrip is ~7x slower)
        pio.write_images(
            fig=[figure for figure, _ in exports],
            file=[path for _, path in exports],
            format="png",
            width=1000,
            height=500,
        )
    body = "\n".join(parts)
    document = (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        "<title>metricprobe report</title></head>\n"
        f"<body>\n{body}\n</body></html>\n"
    )
    report_path = out / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path
