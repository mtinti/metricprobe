"""Publish contract (PLAN Step 9): the README renders for every scenario with
the full status vocabulary, relative image links that exist, the
next-expected date, suppression honored in markdown AND SVG, and byte-stable
output under a frozen clock."""

from __future__ import annotations

import json
import re

import pandas as pd
import pytest

from metricprobe.config import CampaignConfig
from metricprobe.publish import (
    PUBLISHED_MARKER,
    canonicalize_svg,
    emit_dashboard,
    next_cron_fire,
    next_expected_by,
)


@pytest.fixture(scope="module")
def dashboard_dir(dashboard_run, tmp_path_factory):
    store, run_id, config = dashboard_run
    out = tmp_path_factory.mktemp("dashboard")
    emit_dashboard(store, run_id, [config], out)
    return out


def test_readme_status_block_and_table(dashboard_dir, dashboard_run):
    _, run_id, config = dashboard_run
    text = (dashboard_dir / "README.md").read_text(encoding="utf-8")
    assert "**Generated at:**" in text
    assert f"`{run_id}`" in text
    assert "**Git:**" in text
    assert "**Analysed window:**" in text
    assert "**Next update expected by:**" in text
    # the campaign schedule (Mondays 06:00 UTC + 6h grace) resolves to a date
    assert re.search(r"\*\*Next update expected by:\*\* \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC", text)
    # every probe appears as a table row answering the three questions
    for table in config.tables:
        assert table.probe_name in text
    header = (
        "| Table | Probe | Healthy? | Updating? | Complete back to | "
        "p95 (days) | p95 (months) |"
    )
    assert header in text
    # the pathology mix exercises the FULL vocabulary
    for badge in ("✅", "⚠️", "🔴", "⏳"):
        assert badge in text, f"badge {badge} never rendered"
    assert "❓" in text  # parity_b holds duplicate-free keys; parity a vs b mismatch is RED,
    # while read_uncommitted/prereq INDETERMINATE appears via the legend at minimum
    # a healthy probe promises a concrete completeness date AND its p95
    # latency in both units (mean ± std days, fractional months)
    healthy_row = next(
        line for line in text.splitlines() if "volume_spike_ok_probe" in line
    )
    assert re.search(r"\d{4}-\d{2}-\d{2}", healthy_row), healthy_row
    assert re.search(r"\| \d+ ± \d+ d \| \d+\.\d mo \|", healthy_row), healthy_row
    # a probe with nothing classifiable renders em-dashes, never a number
    # (the tiny probe is deliberately too young: INSUFFICIENT_HISTORY)
    tiny_row = next(line for line in text.splitlines() if "| tiny_probe |" in line)
    assert tiny_row.rstrip().endswith("| — | — |"), tiny_row
    assert "⏳" in tiny_row
    assert "p95 = mean ± std days" in text  # the legend explains the columns
    # a probe whose training cohort is CENSORED past its lag cap renders the
    # "> cap" cells — via the production status path (a censored cohort has
    # no learned wait, hence no mature months to inspect)
    censored_row = next(
        line for line in text.splitlines() if "| censored_probe |" in line
    )
    assert "| > 15 d | > 0.5 mo |" in censored_row, censored_row


def test_image_links_are_relative_and_exist(dashboard_dir):
    text = (dashboard_dir / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    assert links, "no figures embedded"
    for link in links:
        assert not link.startswith(("http://", "https://", "/")), link
        assert (dashboard_dir / link).exists(), f"broken image link {link}"
        assert link.endswith(".svg")
    # the interactive report is committed alongside
    assert (dashboard_dir / "report.html").exists()
    assert json.loads((dashboard_dir / PUBLISHED_MARKER).read_text())["run_id"]


def test_svgs_are_canonical_and_deterministic(dashboard_dir, dashboard_run, tmp_path):
    svgs = sorted((dashboard_dir / "img").glob("*.svg"))
    assert svgs
    for svg in svgs[:5]:
        content = svg.read_text(encoding="utf-8")
        assert "defs-000000" in content  # the random document uid was canonicalized
    # a second emission is BYTE-IDENTICAL (fixed-seed CI diff check relies on it)
    store, run_id, config = dashboard_run
    again = tmp_path / "again"
    emit_dashboard(store, run_id, [config], again)
    assert (again / "README.md").read_bytes() == (
        dashboard_dir / "README.md"
    ).read_bytes()
    for svg in svgs:
        assert (again / "img" / svg.name).read_bytes() == svg.read_bytes(), svg.name


def test_suppression_holds_in_markdown_and_svg(dashboard_dir):
    """The tiny probe's raw COUNTS (3 rows/month) must not surface in its SVG
    volume figure: every count was suppressed before serialization, so the
    chart has NO drawn bars at all. (Percentile DAYS may legitimately be
    small numbers — durations are not counts.)"""
    volume_svg = dashboard_dir / "img" / "tiny_probe_volume.svg"
    assert volume_svg.exists()
    content = volume_svg.read_text(encoding="utf-8")
    # plotly draws a None bar as the degenerate path M0,0Z: EVERY bar of the
    # suppressed chart must be degenerate — no rectangle encodes the count
    bar_paths = re.findall(r'<g class="point"><path d="([^"]*)"', content)
    real_bars = [d for d in bar_paths if d != "M0,0Z"]
    assert not real_bars, f"suppressed volume bars were drawn: {real_bars[:2]}"
    # a healthy probe's volume figure DOES draw its bars (the check can fail)
    control = (dashboard_dir / "img" / "volume_spike_ok_probe_volume.svg").read_text(
        encoding="utf-8"
    )
    control_paths = re.findall(r'<g class="point"><path d="([^"]*)"', control)
    assert any(d != "M0,0Z" for d in control_paths)
    text = (dashboard_dir / "README.md").read_text(encoding="utf-8")
    assert "tiny_probe" in text


def test_canonicalize_svg_is_idempotent():
    sample = (
        '<defs id="defs-a1b2c3"><clipPath id="clipa1b2c3xy">'
        '<g class="trace scatter trace9f8e7d">'
    )
    once = canonicalize_svg(sample)
    assert "a1b2c3" not in once and "9f8e7d" not in once
    assert 'defs-000000' in once and "trace000001" in once
    assert canonicalize_svg(once) == once


def test_next_cron_fire_and_expected_by():
    # Mondays 06:00: from a Thursday, the next fire is the coming Monday
    fire = next_cron_fire("0 6 * * 1", pd.Timestamp("2026-07-09 12:00"))
    assert fire == pd.Timestamp("2026-07-13 06:00")
    # strictly after: AT the fire minute, the next one is a week later
    fire = next_cron_fire("0 6 * * 1", pd.Timestamp("2026-07-13 06:00"))
    assert fire == pd.Timestamp("2026-07-20 06:00")
    # day-of-month schedule
    fire = next_cron_fire("30 2 1 * *", pd.Timestamp("2026-07-09 12:00"))
    assert fire == pd.Timestamp("2026-08-01 02:30")
    # step minutes
    fire = next_cron_fire("*/15 * * * *", pd.Timestamp("2026-07-09 12:07"))
    assert fire == pd.Timestamp("2026-07-09 12:15")
    # the grace period lands in the printed promise
    campaign = CampaignConfig(schedule="0 6 * * 1", timezone="UTC", grace_period_hours=6)
    assert next_expected_by(campaign, "2026-07-09T12:00:00") == "2026-07-13 12:00 UTC"
    assert next_expected_by(CampaignConfig(), "2026-07-09T12:00:00") == "manual runs only"
    # timezone-aware: the schedule fires in the campaign timezone
    london = CampaignConfig(
        schedule="0 6 * * 1", timezone="Europe/London", grace_period_hours=0
    )
    assert next_expected_by(london, "2026-07-09T12:00:00") == (
        "2026-07-13 06:00 Europe/London"
    )


def test_cron_handles_leap_days_and_dst():
    # a Feb-29 schedule resolves years ahead, never "manual runs only"
    fire = next_cron_fire("0 6 29 2 *", pd.Timestamp("2026-07-09 12:00"))
    assert fire == pd.Timestamp("2028-02-29 06:00")
    # DST spring-forward: 01:30 does not exist on 2026-03-29 in London
    # (clocks jump 01:00 -> 02:00) — the fire lands on the NEXT valid day,
    # at the requested wall time
    fire = next_cron_fire(
        "30 1 * * *",
        pd.Timestamp("2026-03-28 12:00", tz="Europe/London"),
    )
    assert fire == pd.Timestamp("2026-03-30 01:30", tz="Europe/London")
    # daily schedules keep local wall time ACROSS the DST boundary
    fire = next_cron_fire(
        "0 6 * * *",
        pd.Timestamp("2026-03-28 12:00", tz="Europe/London"),
    )
    assert fire == pd.Timestamp("2026-03-29 06:00", tz="Europe/London")
    assert str(fire.tz) == "Europe/London" and fire.hour == 6


def test_p95_cells_render_every_edge(dashboard_run):
    """The p95 columns are honest at the edges: censored mature months say
    '> cap', unclassifiable probes say '—', healthy ones carry mean ± std."""
    import pandas as pd

    from metricprobe.publish import _p95_cells

    _, _, config = dashboard_run
    table = config.tables[0]
    healthy = {
        "completion_summary": pd.DataFrame(
            {"probe": ["p"], "p95_mean": [12.4], "p95_std": [2.6]}
        )
    }
    assert _p95_cells(healthy, "p", table) == ("12 ± 3 d", "0.4 mo")
    # censoring is read from the persisted PERCENTILE_OVER_CAP status — the
    # production signal (a censored training cohort produces no mature months
    # at all, so inspecting mature months could never fire)
    censored = {
        "completion_summary": pd.DataFrame(
            {"probe": ["p"], "p95_mean": [None], "p95_std": [None]}
        ),
        "statuses": pd.DataFrame(
            {"probe": ["p"], "check": ["completion"],
             "severity": ["insufficient_history"],
             "reason": ["percentile_over_cap"]}
        ),
    }
    days, months = _p95_cells(censored, "p", table)
    assert days == f"> {table.analysis.lag_cap_days} d"
    assert months.startswith("> ") and months.endswith(" mo")
    # insufficient WITHOUT censoring stays an em-dash
    insufficient = {
        "completion_summary": pd.DataFrame(
            {"probe": ["p"], "p95_mean": [None], "p95_std": [None]}
        ),
        "statuses": pd.DataFrame(
            {"probe": ["p"], "check": ["completion"],
             "severity": ["insufficient_history"],
             "reason": ["insufficient_mature_months"]}
        ),
    }
    assert _p95_cells(insufficient, "p", table) == ("—", "—")
    assert _p95_cells({}, "p", table) == ("—", "—")
