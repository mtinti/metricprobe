"""Static-report contract (PLAN Step 9): self-contained offline HTML, PNGs
per figure, prerequisite failure is actionable, suppression survives
end-to-end into the serialized artifact."""

from __future__ import annotations

import re

import pytest

from metricprobe.report import (
    ensure_static_export_available,
    generate_report,
)

# the OFFLINE guarantee: no tag that would FETCH an external resource when
# the page opens (script/link/img/iframe/source). The vendored plotly.js
# bundle contains https:// string literals (map attribution links) that are
# never fetched — those are library source, not resource loads.
_EXTERNAL_RESOURCE = re.compile(
    r"""<(?:script|link|img|iframe|source)\b[^>]*(?:src|href)\s*=\s*["']https?://""",
    re.IGNORECASE,
)
_CDN_SCRIPT = re.compile(r"""<script[^>]*(?:cdn\.plot\.ly|unpkg\.com|cdnjs)""", re.IGNORECASE)


@pytest.fixture(scope="module")
def report_dir(dashboard_run, tmp_path_factory):
    store, run_id, config = dashboard_run
    ensure_static_export_available()
    out = tmp_path_factory.mktemp("report")
    generate_report(store, run_id, [config], out)
    return out


def test_report_is_self_contained_offline_html(report_dir, dashboard_run):
    html_text = (report_dir / "report.html").read_text(encoding="utf-8")
    # embedded plotly.js, not a CDN script tag
    assert "<script" in html_text and "plotly" in html_text.lower()
    assert not _EXTERNAL_RESOURCE.search(html_text), "external resource load found"
    assert not _CDN_SCRIPT.search(html_text), "CDN script tag found"
    # provenance header
    _, run_id, _ = dashboard_run
    assert run_id in html_text
    assert "Analysed window" in html_text
    # every probe has a section
    _, _, config = dashboard_run
    for table in config.tables:
        assert table.probe_name in html_text


def test_completion_views_are_tabs(report_dir):
    """Curves are the DEFAULT view; the heatmap sits behind a tab toggle."""
    html_text = (report_dir / "report.html").read_text(encoding="utf-8")
    assert "mpShow(" in html_text and 'class="mp-tab' in html_text
    # every heatmap pane starts hidden, every curves pane starts visible
    hidden = re.findall(r'id="mp-\d+-heatmap" class="mp-pane" style="display:none"',
                        html_text)
    assert hidden, "no hidden heatmap panes"
    assert 'id="mp-0-curves" class="mp-pane">' in html_text


def test_pngs_exist_per_figure(report_dir):
    pngs = sorted(p.name for p in (report_dir / "img").glob("*.png"))
    assert pngs, "no PNGs were exported"
    assert any(name.endswith("_volume.png") for name in pngs)
    assert any(name.endswith("_completion_curves.png") for name in pngs)
    assert any(name.endswith("_completion_heatmap.png") for name in pngs)  # tabbed too
    assert any(name.startswith("dual_registry_probe_dual_overlay") for name in pngs)
    for png in (report_dir / "img").glob("*.png"):
        assert png.stat().st_size > 1000  # a real raster, not an error stub


def test_suppressed_counts_never_reach_the_html(report_dir):
    """The tiny probe's raw per-month count (3) must not appear as a y-value
    anywhere in its serialized figures."""
    html_text = (report_dir / "report.html").read_text(encoding="utf-8")
    # locate the tiny probe's volume trace payloads
    tiny_sections = re.findall(r'"name":\s*"(?:mature|immature|open)"[^}]*?"y":\s*(\[[^\]]*\])',
                               html_text)
    assert tiny_sections  # volume traces exist in the document
    # the tiny table has 3 rows in EVERY month: with suppression on, a literal
    # small count in any tiny-probe trace would be a leak. Other probes carry
    # thousands of rows, so a bare 3/4 y-value can only come from the leak.
    for payload in tiny_sections:
        values = [v.strip() for v in payload.strip("[]").split(",") if v.strip()]
        for value in values:
            if value in {"1", "2", "3", "4", "1.0", "2.0", "3.0", "4.0"}:
                raise AssertionError(f"suppressed count leaked into HTML: {value}")


def test_missing_kaleido_gives_an_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_kaleido(name, *args, **kwargs):
        if name == "kaleido":
            raise ImportError("No module named 'kaleido'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_kaleido)
    with pytest.raises(RuntimeError) as excinfo:
        ensure_static_export_available()
    message = str(excinfo.value)
    assert "metricprobe[export]" in message
    assert "kaleido_get_chrome" in message  # the locked-down-Windows path


def test_broken_chrome_gives_an_actionable_error(monkeypatch):
    import plotly.io as pio

    def explode(*args, **kwargs):
        raise RuntimeError("Chrome executable not found")

    monkeypatch.setattr(pio, "to_image", explode)
    with pytest.raises(RuntimeError) as excinfo:
        ensure_static_export_available()
    assert "kaleido_get_chrome" in str(excinfo.value)
