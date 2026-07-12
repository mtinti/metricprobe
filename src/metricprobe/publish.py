"""Markdown dashboard emitter: a forge-renderable README (git forges serve
raw HTML as plain text, so markdown IS the in-repo dashboard format) plus
committed SVG figures — small, text-based, git-delta-friendly — and the
self-contained HTML report alongside for download.

The README opens with a status block: Generated-at, run id, git SHA, analysed
window, and "Next update expected by: <date>" computed from the campaign
schedule cadence + grace period — a silently-dead pipeline is self-evident to
any reader of the repo front page.

SVG output is CANONICALIZED (plotly's per-render random uid is rewritten to a
stable token) so a fixed-seed demo build is byte-stable and CI can diff it.

A machine-readable `.metricprobe-published.json` marker is emitted beside the
README; delivery reads the currently-published marker to enforce the
monotonic-publication guard (an older run may never overwrite a newer
published dashboard).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import plotly.io as pio

from metricprobe.config import CampaignConfig, ProbeConfig
from metricprobe.report import build_all_figures, generate_report
from metricprobe.status import Severity
from metricprobe.viz.presentation import load_run_frames

SEVERITY_BADGES = {
    Severity.RED: "🔴",
    Severity.AMBER: "⚠️",
    Severity.INDETERMINATE: "❓",
    Severity.INSUFFICIENT_HISTORY: "⏳",
    Severity.SKIPPED: "➖",
    Severity.GREEN: "✅",
}

PUBLISHED_MARKER = ".metricprobe-published.json"

# plotly stamps every SVG render with random 6-hex uids: one per document
# (defs-xxxxxx / clipxxxxxx...) and one per TRACE (class="... tracexxxxxx").
# Rewrite them to sequential stable tokens (first-appearance order) so a
# fixed-seed build is byte-stable and CI can diff committed SVGs.
# every random uid family plotly stamps into an SVG render: the document uid
# (defs-/topdefs-/clip...), per-trace and legend uids, colorbar tokens (cb...)
# and gradient ids (g<hex>-..., referenced as url(#g...) or url('#g...)).
# No leading \b on defs-: "topdefs-<uid>" must match through its suffix.
# cb tokens embed mid-word (class="ycb<uid>tick"), so no boundaries there
_PREFIXED_UID = re.compile(r"(defs-|\btrace|\blegend|cb)([0-9a-f]{6})")
_GRADIENT_UID = re.compile(r"""(id="|url\('?\#)g([0-9a-f]{6})""")


def canonicalize_svg(svg: str) -> str:
    mapping: dict[str, str] = {}

    def stable(uid: str) -> str:
        if uid not in mapping:
            mapping[uid] = f"{len(mapping):06x}"
        return mapping[uid]

    svg = _PREFIXED_UID.sub(
        lambda match: match.group(1) + stable(match.group(2)), svg
    )
    for uid, replacement in list(mapping.items()):  # clip ids share the defs uid
        svg = svg.replace(f"clip{uid}", f"clip{replacement}")
    return _GRADIENT_UID.sub(
        lambda match: match.group(1) + "g" + stable(match.group(2)), svg
    )  # the -suffix of a gradient id is a cb token, already canonical


# ------------------------------------------------------------- next expected


def _cron_part_matches(part: str, value: int, low: int, high: int) -> bool:
    body, _, step_text = part.partition("/")
    step = int(step_text) if step_text else 1
    if body == "*":
        return (value - low) % step == 0
    if "-" in body:
        start, end = (int(piece) for piece in body.split("-"))
        return start <= value <= end and (value - start) % step == 0
    start = int(body)
    if step_text:  # "a/step" runs from a to the field maximum
        return start <= value <= high and (value - start) % step == 0
    return value == start


def _cron_field_matches(field_text: str, value: int, low: int, high: int) -> bool:
    return any(
        _cron_part_matches(part, value, low, high) for part in field_text.split(",")
    )


# scan horizon: 8 years covers the sparsest valid schedule (Feb 29 recurs
# within any 8-year window, including the 2100-style century gap-free range)
_CRON_SCAN_DAYS = 366 * 8


def next_cron_fire(schedule: str, after: pd.Timestamp) -> pd.Timestamp | None:
    """First schedule fire STRICTLY after `after` (same timezone as `after`).
    Day-of-month and day-of-week follow standard cron OR semantics when both
    are restricted; 0 and 7 both mean Sunday. Timezone-aware inputs handle
    DST: days are enumerated by local midnight, and a candidate whose local
    wall time was shifted (nonexistent hour on a spring-forward day) is
    skipped, matching common cron behavior."""
    minute_f, hour_f, dom_f, month_f, dow_f = schedule.split()
    start = (after + pd.Timedelta(minutes=1)).floor("min")
    for day_offset in range(_CRON_SCAN_DAYS):
        # 24h steps then normalize(): local midnight of every day, duplicates
        # possible around fall-back (harmless), no day ever skipped
        day = (start + pd.Timedelta(days=day_offset)).normalize()
        if not _cron_field_matches(month_f, day.month, 1, 12):
            continue
        cron_dow = (day.weekday() + 1) % 7  # cron: Sunday = 0 (and 7)
        dom_ok = _cron_field_matches(dom_f, day.day, 1, 31)
        dow_ok = _cron_field_matches(dow_f, cron_dow, 0, 7) or (
            cron_dow == 0 and _cron_field_matches(dow_f, 7, 0, 7)
        )
        if dom_f != "*" and dow_f != "*":
            if not (dom_ok or dow_ok):
                continue
        elif not (dom_ok and dow_ok):
            continue
        for hour in range(24):
            if not _cron_field_matches(hour_f, hour, 0, 23):
                continue
            for minute in range(60):
                if not _cron_field_matches(minute_f, minute, 0, 59):
                    continue
                if day.tz is None:
                    candidate = day + pd.Timedelta(hours=hour, minutes=minute)
                else:
                    # construct the LOCAL wall time (Timedelta arithmetic
                    # counts elapsed hours and drifts across DST changes);
                    # a nonexistent local time (spring forward) is skipped,
                    # an ambiguous one (fall back) fires on its first pass
                    naive = pd.Timestamp(day.year, day.month, day.day, hour, minute)
                    try:
                        candidate = naive.tz_localize(day.tz, ambiguous=True)
                    except Exception:  # noqa: BLE001 — nonexistent wall time
                        continue
                if candidate >= start:
                    return candidate
    return None


def next_expected_by(campaign: CampaignConfig, run_at: str) -> str:
    """The self-diagnosing staleness promise: next scheduled fire after this
    run, plus the grace period. 'manual runs only' when unscheduled.
    run_at is timezone-aware UTC from the runner; a naive value (older
    snapshots, frozen demo clocks) is DEFINED as UTC."""
    if campaign.schedule is None:
        return "manual runs only"
    stamp = pd.Timestamp(run_at)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    local = stamp.tz_convert(campaign.timezone)
    fire = next_cron_fire(campaign.schedule, local)
    if fire is None:
        return "manual runs only"
    expected = fire + pd.Timedelta(hours=campaign.grace_period_hours)
    return f"{expected:%Y-%m-%d %H:%M} {campaign.timezone}"


# ------------------------------------------------------------------ delivery


def _as_utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def check_monotonic_publication(candidate_run_at: str, published_run_ats: list[str]) -> None:
    """An older run may never overwrite a dashboard published from a newer
    run. Timestamps are compared as INSTANTS (offset-aware), never strings."""
    candidate = _as_utc(candidate_run_at)
    newer = [stamp for stamp in published_run_ats if _as_utc(stamp) > candidate]
    if newer:
        raise RuntimeError(
            f"refusing to publish run from {candidate_run_at}: a newer run "
            f"({max(newer, key=_as_utc)}) is already published"
        )


def _git(clone: Path, *args: str) -> str:
    """Run git in `clone` with a fixed identity; logic stays in Python (the
    CI workflow is one invocation, never a chain of shell one-liners)."""
    result = subprocess.run(
        ["git", "-c", "user.name=metricprobe", "-c", "user.email=metricprobe@localhost",
         *args],
        cwd=clone,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args[:2])} failed in {clone}: {result.stderr.strip()}"
        )
    return result.stdout


def _push_url(remote) -> str:
    """The remote URL with the token from token_env injected for http(s).
    Config holds env var NAMES only; the value is read here, at push time,
    and never written to any file."""
    url = remote.url  # ${VAR} references were expanded by the config loader
    if remote.token_env is None:
        return url
    token = os.environ.get(remote.token_env)
    if not token:
        raise RuntimeError(
            f"delivery remote {remote.name!r}: environment variable "
            f"{remote.token_env} is not set (it must hold the push token)"
        )
    scheme, separator, rest = url.partition("://")
    if separator and scheme in ("http", "https") and "@" not in rest:
        return f"{scheme}://x-access-token:{token}@{rest}"
    return url


def _remote_has_ref(clone: Path, url: str, ref: str) -> bool:
    out = subprocess.run(
        ["git", "ls-remote", "--heads", url, ref],
        cwd=clone, capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git ls-remote failed: {out.stderr.strip()}")
    return bool(out.stdout.strip())


def deliver(artifacts_dir: str | Path, delivery, run_id: str, run_at: str) -> list[str]:
    """Push the rendered dashboard to EVERY configured remote, in TWO phases:
    first PREPARE every remote (sync its clone, enforce the
    monotonic-publication guard against what it already publishes, stage this
    run's files, commit locally) — any failure here leaves every remote
    untouched — then push all. The artifacts must carry THIS run's marker:
    delivery can never publish one run's files under another run's name."""
    artifacts = Path(artifacts_dir)
    marker = artifacts / PUBLISHED_MARKER
    if not marker.exists():
        raise RuntimeError(f"no rendered dashboard at {artifacts} (marker missing)")
    marked = json.loads(marker.read_text(encoding="utf-8"))
    if marked["run_id"] != run_id:
        raise RuntimeError(
            f"rendered artifacts at {artifacts} belong to run {marked['run_id']!r}, "
            f"not {run_id!r}; refusing to deliver them under the wrong name"
        )
    worktree = Path(delivery.worktree)

    # ---- phase 1: prepare every remote (no pushes yet)
    prepared: list[tuple[object, str, Path]] = []
    for remote in delivery.remotes:
        url = _push_url(remote)
        clone = worktree / "remotes" / remote.name
        clone.mkdir(parents=True, exist_ok=True)
        if not (clone / ".git").exists():
            _git(clone, "init", "--initial-branch", remote.ref)
        if _remote_has_ref(clone, url, remote.ref):
            _git(clone, "fetch", url, f"+refs/heads/{remote.ref}:refs/remotes/delivery/head")
            _git(clone, "checkout", "-B", "mp-delivery", "refs/remotes/delivery/head")
            published = clone / PUBLISHED_MARKER
            if published.exists():
                previous = json.loads(published.read_text(encoding="utf-8"))
                check_monotonic_publication(run_at, [previous["run_at"]])
        # replace the dashboard files with this run's artifacts
        for name in ("README.md", "report.html", PUBLISHED_MARKER):
            source = artifacts / name
            if source.exists():
                shutil.copy2(source, clone / name)
        if (clone / "img").exists():
            shutil.rmtree(clone / "img")
        if (artifacts / "img").exists():
            shutil.copytree(artifacts / "img", clone / "img")
        _git(clone, "add", "-A")
        if _git(clone, "status", "--porcelain").strip():
            _git(clone, "commit", "-m", f"metricprobe dashboard {run_id}")
        prepared.append((remote, url, clone))

    # ---- phase 2: push (each push is idempotent, so a retry after a partial
    # network failure re-pushes no-change commits harmlessly)
    pushed = []
    for remote, url, clone in prepared:
        _git(clone, "push", url, f"HEAD:refs/heads/{remote.ref}")
        pushed.append(remote.name)
    return pushed


# ---------------------------------------------------------------- the README


def _worst(severities: list[str]) -> Severity | None:
    present = {Severity(value) for value in severities}
    for severity in SEVERITY_BADGES:  # iteration order IS the frozen precedence
        if severity in present:
            return severity
    return None


def _badge(severity: Severity | None) -> str:
    return SEVERITY_BADGES.get(severity, "➖")


def _status_rows(frames: dict[str, pd.DataFrame], configs: list[ProbeConfig]) -> list[dict]:
    statuses = frames.get("statuses", pd.DataFrame())
    summaries = frames.get("completion_summary", pd.DataFrame())
    rows = []
    for table in (table for config in configs for table in config.tables):
        probe = table.probe_name
        mine = statuses[statuses["probe"] == probe] if not statuses.empty else statuses
        healthy = _worst(list(mine["severity"])) if not mine.empty else None
        fresh = mine[mine["check"] == "freshness"] if not mine.empty else mine
        updating = _worst(list(fresh["severity"])) if not fresh.empty else None
        back_to = "—"
        if not summaries.empty:
            summary = summaries[summaries["probe"] == probe]
            if not summary.empty:
                value = summary.iloc[0].get("complete_back_to")
                if value is not None and not pd.isna(value):
                    back_to = str(value)
        rows.append(
            {
                "database": table.database,
                "table": f"{table.table_schema}.{table.table}",
                "probe": probe,
                "healthy": _badge(healthy),
                "updating": _badge(updating),
                "back_to": back_to,
            }
        )
    return rows


def emit_dashboard(
    store,
    run_id: str,
    configs: list[ProbeConfig],
    out_dir: str | Path,
) -> Path:
    """Write README.md + img/<probe>_<figure>.svg + report.html + the
    published marker into out_dir, over EVERY campaign config. Returns the
    README path."""
    out = Path(out_dir)
    (out / "img").mkdir(parents=True, exist_ok=True)
    manifest = next(m for m in store.list_runs() if m["run_id"] == run_id)
    frames = load_run_frames(store, run_id)
    all_figures = build_all_figures(store, run_id, configs)

    # ---- SVG figures: ONE batched export, then canonicalized in place
    exports = [
        (figure, out / "img" / f"{probe}_{key}.svg")
        for probe, figures in all_figures.items()
        for key, figure in figures.items()
    ]
    if exports:
        pio.write_images(
            fig=[figure for figure, _ in exports],
            file=[path for _, path in exports],
            format="svg",
            width=1000,
            height=450,
        )
        for _, path in exports:
            path.write_text(
                canonicalize_svg(path.read_text(encoding="utf-8")), encoding="utf-8"
            )

    # ---- README
    lines: list[str] = []
    lines.append("# metricprobe dashboard")
    lines.append("")
    lines.append(
        f"**Generated at:** {manifest['run_at']} · **Run:** `{manifest['run_id']}` · "
        f"**Git:** `{manifest['git_sha'][:12]}` · **Tool:** {manifest['tool_version']}"
    )
    lines.append("")
    lines.append(
        f"**Analysed window:** {manifest['window_start'][:10]} → "
        f"{manifest['window_end'][:10]} · **as-of:** {manifest['as_of'][:10]}"
    )
    lines.append("")
    lines.append(
        "**Next update expected by:** "
        f"{next_expected_by(configs[0].campaign, manifest['run_at'])}"
    )
    lines.append("")
    lines.append(
        "Legend: ✅ green · ⚠️ amber · 🔴 red · ❓ indeterminate · "
        "⏳ insufficient history · ➖ skipped"
    )
    lines.append("")
    rows = _status_rows(frames, configs)
    for database in sorted({row["database"] for row in rows}):
        lines.append(f"## {database}")
        lines.append("")
        lines.append("| Table | Probe | Healthy? | Updating? | Complete back to |")
        lines.append("| --- | --- | :---: | :---: | --- |")
        for row in rows:
            if row["database"] != database:
                continue
            lines.append(
                f"| {row['table']} | {row['probe']} | "
                f"{row['healthy']} | {row['updating']} | {row['back_to']} |"
            )
        lines.append("")
    lines.append("Full interactive report: [report.html](report.html) (download to open).")
    lines.append("")
    for probe, figures in all_figures.items():
        lines.append(f"### {probe}")
        lines.append("")
        for key in figures:
            lines.append(f"![{probe} {key}](img/{probe}_{key}.svg)")
        lines.append("")

    readme = out / "README.md"
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- self-contained HTML alongside (no extra PNGs: SVGs are committed)
    generate_report(store, run_id, configs, out, png=False)

    # ---- the machine-readable publication marker (monotonic guard input)
    (out / PUBLISHED_MARKER).write_text(
        json.dumps(
            {"run_id": manifest["run_id"], "run_at": manifest["run_at"]},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return readme
