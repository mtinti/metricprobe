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
    store, run_id: str, configs: list[ProbeConfig]
) -> dict[str, dict[str, go.Figure]]:
    """probe -> {figure key -> Figure} for every probe of a committed run —
    across EVERY campaign config file — through the shared presentation
    transform (suppression happens there)."""
    frames = load_run_frames(store, run_id)
    tables = {
        table.probe_name: table for config in configs for table in config.tables
    }
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
            analysis=table.analysis if table else None,
        )
    return all_figures


def _manifest_for(store, run_id: str) -> dict:
    for manifest in store.list_runs():
        if manifest["run_id"] == run_id:
            return manifest
    raise FileNotFoundError(f"run {run_id!r} is not committed")


# tab chrome: tiny, inline, offline. Revealing a pane fires a resize so the
# plotly figure inside (rendered while hidden) lays itself out.
_TABS_STYLE_AND_SCRIPT = (
    "<style>.mp-tab{padding:4px 14px;border:1px solid #999;background:#eee;"
    "cursor:pointer}.mp-tab.mp-active{background:#2f6fb2;color:#fff}</style>\n"
    "<script>function mpShow(index, key, button){\n"
    "  for (const pane of ['curves','heatmap']) {\n"
    "    document.getElementById('mp-'+index+'-'+pane).style.display =\n"
    "      (pane === key) ? '' : 'none';\n"
    "  }\n"
    "  for (const sibling of button.parentNode.children)"
    " sibling.classList.remove('mp-active');\n"
    "  button.classList.add('mp-active');\n"
    "  window.dispatchEvent(new Event('resize'));\n"
    "}</script>"
)


def generate_report(
    store,
    run_id: str,
    configs: list[ProbeConfig],
    out_dir: str | Path,
    png: bool = True,
) -> Path:
    """Write report.html (self-contained) and img/<probe>_<figure>.png files.
    Returns the report path. Static-export availability must have been
    verified by the caller (ensure_static_export_available)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = _manifest_for(store, run_id)
    all_figures = build_all_figures(store, run_id, configs)

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
    parts.append(_TABS_STYLE_AND_SCRIPT)
    include_js: bool | str = True  # first figure embeds the bundled plotly.js
    exports: list[tuple[object, Path]] = []
    if png:
        (out / "img").mkdir(exist_ok=True)
    for index, (probe, figures) in enumerate(all_figures.items()):
        parts.append(f"<h2>{html.escape(probe)}</h2>")
        if not figures:
            parts.append("<p>no figures (probe skipped or aborted)</p>")

        def _fragment(figure, key: str, index: int = index) -> str:
            nonlocal include_js
            fragment = pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=include_js,
                default_width="100%",
                default_height="450px",
                # deterministic div ids: to_html would mint a random uuid per
                # render, breaking the committed demo's byte-stability
                div_id=f"mp-fig-{index}-{key}",
            )
            include_js = False
            return fragment

        tabbed = {"completion_curves", "completion_heatmap"} & set(figures)
        for key, figure in figures.items():
            if png:  # every figure exports, tabbed or not
                exports.append((figure, out / "img" / f"{probe}_{key}.png"))
            if key == "completion_curves" and len(tabbed) == 2:
                # the completion views are TABS: curves by default, the
                # event-month x lag-week heatmap behind a toggle
                heatmap = figures["completion_heatmap"]
                parts.append(
                    f'<div class="mp-tabs">'
                    f'<button class="mp-tab mp-active" '
                    f"onclick=\"mpShow({index},'curves',this)\">Curves</button>"
                    f'<button class="mp-tab" '
                    f"onclick=\"mpShow({index},'heatmap',this)\">Heatmap</button>"
                    f"</div>"
                    f'<div id="mp-{index}-curves" class="mp-pane">'
                    f"{_fragment(figure, key)}</div>"
                    f'<div id="mp-{index}-heatmap" class="mp-pane" '
                    f'style="display:none">{_fragment(heatmap, "completion_heatmap")}</div>'
                )
            elif key == "completion_heatmap" and len(tabbed) == 2:
                continue  # rendered inside the tab container above
            else:
                parts.append(_fragment(figure, key))
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
