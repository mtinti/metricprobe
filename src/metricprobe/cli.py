"""Command-line interface and the ONE orchestration state machine.

The campaign command (`metricprobe run`) owns the whole lifecycle in one
process and is the SOLE delivery owner: analyse -> ANALYSIS_COMMITTED (atomic
manifest) -> ARTIFACTS_RENDERED (dashboard emitted into the delivery
worktree) -> PUBLISHED (pushed to every configured remote; claimed only after
the actual delivery succeeded). Render + publish are CONFIGURED exactly when
`delivery:` is; without it the only configured stage is analysis. Exit codes
are relative to the CONFIGURED stages:

    0 = all configured stages completed, no data-health RED
    2 = all configured stages completed, at least one RED (committed FIRST)
    1 = a configured stage failed OR the invocation itself was invalid:
        everything before the failure remains committed and honestly
        reported; the failed stage leaves NOTHING partial. Exit 2 is
        reserved for data health and is NEVER returned for usage errors.

Windowing: --window/--year bound the REPORTED completion results (per-month
percentiles, dual percentiles, compare side-stat) to the probe window. Volume
history is full-history BY CONTRACT (CLAUDE.md metric 1), and the learned
wait's training cohort always uses full history — both are stamped with the
window so readers know the analysis frame.

Probes execute SEQUENTIALLY, never in parallel (assume every probe is a full
scan of a possibly unindexed production table); each probe's wall-clock
duration and its DATABASE extraction interval (bracketing the canonical and
dual passes only) land in the manifest. Clock, run_id and git metadata are
INJECTABLE (--as-of, --run-id, METRICPROBE_RUN_AT, METRICPROBE_GIT_SHA); with
a frozen clock every recorded timestamp is deterministic.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import subprocess
import sys
import uuid

import pandas as pd
import sqlalchemy as sa

import metricprobe
from metricprobe.config import (
    CONFIG_SCHEMA_VERSION,
    ConfigError,
    ProbeConfig,
    TableConfig,
    compose_campaign,
    config_digest,
    expand_env,
    load_config,
)
from metricprobe.discover import DEFAULT_ROLE_CANDIDATES, draft_config
from metricprobe.extract.canonical import (
    CANONICAL_SCHEMA_VERSION,
    ProbeAborted,
    run_canonical,
)
from metricprobe.extract.dual import DUAL_SCHEMA_VERSION, run_dual_lag
from metricprobe.metrics.batch import assess_batch
from metricprobe.metrics.completion import (
    assess_completion,
    compare_mismatch_by_month,
    complete_back_to,
)
from metricprobe.metrics.dual_lag import assess_dual_lag
from metricprobe.metrics.freshness import assess_freshness
from metricprobe.metrics.parity import ParitySide, assess_parity
from metricprobe.metrics.volume import assess_volume
from metricprobe.status import (
    STATUS_SCHEMA_VERSION,
    Check,
    ReasonCode,
    Severity,
    Status,
    exit_code_for,
)
from metricprobe.store import (
    SNAPSHOT_SCHEMA_VERSION,
    RunMeta,
    open_store,
    stamp,
    validate_run_id,
)

# the full lifecycle; a campaign without delivery: configures only "analysis"
STAGES = ("analysis", "render", "publish")


def _configured_stages(config: ProbeConfig) -> tuple[str, ...]:
    """Exit codes are relative to the CONFIGURED stages: render + publish are
    configured exactly when delivery is (the campaign command is the sole
    delivery owner; PUBLISHED is claimed only after the actual push)."""
    return STAGES if config.delivery is not None else ("analysis",)

BUCKET_COLUMNS = (
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
)


def _wall() -> pd.Timestamp:
    """The injectable clock: METRICPROBE_RUN_AT freezes EVERY recorded
    timestamp (durations become 0), making demo builds byte-stable.
    Always timezone-AWARE UTC — a naive local run_at would be misread as UTC
    by schedule math and cross-machine monotonic comparisons."""
    frozen = os.environ.get("METRICPROBE_RUN_AT")
    stamp = pd.Timestamp(frozen) if frozen else pd.Timestamp.now(tz="UTC")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _git_sha() -> str:
    injected = os.environ.get("METRICPROBE_GIT_SHA")
    if injected:
        return injected
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _combined_digest(configs: list[ProbeConfig]) -> str:
    if len(configs) == 1:
        return config_digest(configs[0])
    joined = "".join(config_digest(config) for config in configs)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _parse_window(args, as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if args.year and args.window:
        raise ConfigError("pass either --window or --year, not both")
    if args.year:
        start = pd.Timestamp(year=args.year, month=1, day=1)
        return start, start + pd.DateOffset(years=1)
    window = args.window or "24m"
    if not window.endswith("m"):
        raise ConfigError(f"--window must look like '24m', got {window!r}")
    months = int(window[:-1])
    if months < 1:
        raise ConfigError(f"--window must cover at least one month, got {window!r}")
    return as_of - pd.DateOffset(months=months), as_of


def _in_window(month: pd.Period, window: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    return window[0] <= month.start_time < window[1]


def _engine_for(url: str, read_uncommitted: bool):
    """Per-connection isolation: the configured read_uncommitted flag is
    honored on mssql (elsewhere it has no equivalent and is ignored)."""
    if read_uncommitted and sa.engine.make_url(url).get_backend_name() == "mssql":
        return sa.create_engine(url, isolation_level="READ UNCOMMITTED")
    return sa.create_engine(url)


def _table_exists(engine, table: TableConfig) -> bool:
    if engine.dialect.name == "mssql":
        sql = (
            "SELECT COUNT(*) FROM [{db}].INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_SCHEMA = :s AND TABLE_NAME = :t"
        ).format(db=table.database.replace("]", "]]"))
        params = {"s": table.table_schema, "t": table.table}
    else:
        sql = (
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_catalog = :c "
            "AND table_schema = :s AND table_name = :t"
        )
        params = {"c": table.database, "s": table.table_schema, "t": table.table}
    with engine.connect() as conn:
        return bool(conn.execute(sa.text(sql), params).scalar())


class _Frames:
    """Accumulates the tidy snapshot tables, one row-list per table."""

    def __init__(self):
        self.rows: dict[str, list[dict]] = {}

    def add(self, table: str, row: dict) -> None:
        self.rows.setdefault(table, []).append(row)

    def merge(self, other: _Frames) -> None:
        for table, rows in other.rows.items():
            self.rows.setdefault(table, []).extend(rows)

    def frames(self) -> dict[str, pd.DataFrame]:
        return {name: pd.DataFrame(rows) for name, rows in self.rows.items() if rows}


def _probe_one(
    engine,
    table: TableConfig,
    as_of: pd.Timestamp,
    window: tuple[pd.Timestamp, pd.Timestamp],
):
    """One probe. Returns (statuses, parity side, its OWN frames, extraction
    interval): frames only reach the run when the whole probe succeeded, so an
    abort mid-probe can never commit partial metric output."""
    frames = _Frames()
    probe = table.probe_name

    # ---- extraction: the only phase that touches the database
    extraction_started = _wall()
    canonical = run_canonical(engine, table, as_of)
    dual = None
    if table.source_insert_time:
        dual = run_dual_lag(
            engine, table, as_of, prior_target_reads=canonical.target_logical_reads
        )
    extraction_finished = _wall()

    # ---- assessments (no database access from here on)
    completion = assess_completion(canonical, table, as_of)
    volume = assess_volume(canonical, table, as_of, completion)
    freshness = assess_freshness(canonical, table, as_of)
    statuses = completion.statuses + volume.statuses + freshness.statuses

    global_row = canonical.global_row
    frames.add(
        "population_buckets",
        {
            "probe": probe,
            **{column: _int_or_none(global_row[column]) for column in BUCKET_COLUMNS},
        },
    )
    frames.add(
        "completion_summary",
        {
            "probe": probe,
            "learned_wait": completion.learned_wait,
            "horizon": completion.horizon,
            "recommended_wait": completion.recommended_wait,
            "complete_back_to": str(back.date())
            if (back := complete_back_to(completion, as_of)) is not None
            else None,
            "negative_lag_excess_fraction": completion.negative_lag_excess_fraction,
            **{
                f"p{pct}_{stat}": (None if summary is None else summary[i])
                for pct, summary in completion.mature_percentile_summary.items()
                for i, stat in enumerate(("mean", "std"))
            },
        },
    )
    frames.add(
        "volume_summary",
        {
            "probe": probe,
            "baseline_median": volume.baseline_median,
            "baseline_sigma": volume.baseline_sigma,
            "forecast": volume.forecast,
            "duplicate_rows": volume.duplicate_rows,
            "gaps": ", ".join(str(m) for m in volume.gaps),
        },
    )
    for month, per_pct in completion.percentiles.items():
        if not _in_window(month, window):
            continue  # the probe window bounds REPORTED completion results
        for pct, value in per_pct.items():
            frames.add(
                "completion_percentiles",
                {
                    "probe": probe,
                    "month": str(month),
                    "pct": pct,
                    "days": value.value,
                    "over_cap": value.over_cap,
                    # the declared grain basis: "date" flags that sub-day
                    # arrival detail does not exist for this probe
                    "lag_resolution": table.lag_resolution,
                },
            )
    for row in volume.months:
        frames.add(
            "month_volumes",
            {
                "probe": probe,
                "month": str(row.month),
                "volume": row.volume,
                "state": row.state,
                "expected_low": row.expected_low,
                "expected_high": row.expected_high,
                "nowcast": row.nowcast,
                "deficit": row.deficit,
            },
        )
    cells = canonical.rows_for("month_lag")
    for record in cells[["event_month", "lag_day", "row_count"]].to_dict("records"):
        frames.add("month_lag_cells", {"probe": probe, **record})
    epochs = canonical.rows_for("epoch")
    for record in epochs[["load_epoch_day", "row_count"]].to_dict("records"):
        frames.add("epoch_cells", {"probe": probe, **record})
    frames.add(
        "freshness",
        {
            "probe": probe,
            "epoch_count": freshness.epoch_count,
            "last_epoch": str(freshness.last_epoch),
            "cadence_median_days": freshness.cadence_median_days,
            "cadence_sigma_days": freshness.cadence_sigma_days,
            "days_since_last": freshness.days_since_last,
        },
    )
    if table.load_batch_col:
        batch = assess_batch(canonical, table)
        statuses += batch.statuses
        for batch_id, rows in batch.rows_per_run.items():
            frames.add(
                "batch_runs", {"probe": probe, "batch_id": batch_id, "rows": rows}
            )
        for month_row in batch.months:
            frames.add(
                "batch_months",
                {
                    "probe": probe,
                    "month": str(month_row.month),
                    "runs": month_row.runs,
                    "null_batch_rows": month_row.null_batch_rows,
                    **{f"days_to_p{pct}": d for pct, d in month_row.days_to.items()},
                },
            )
    if dual is not None:
        dual_assessment = assess_dual_lag(dual, table, as_of)
        statuses += dual_assessment.statuses
        frames.add(
            "dual_summary",
            {
                "probe": probe,
                "n_null_source_only": dual_assessment.n_null_source_only,
                "n_delta_rows": dual_assessment.n_delta_rows,
                "negative_lag_excess_fraction": dual_assessment.negative_lag_excess_fraction,
            },
        )
        dual_cells = dual.rows_for("month_src_lag")
        for record in dual_cells[["event_month", "lag_day", "row_count"]].to_dict("records"):
            frames.add("dual_lag_cells", {"probe": probe, **record})
        for month, per_pct in dual_assessment.source_percentiles.items():
            if not _in_window(month, window):
                continue
            for pct, value in per_pct.items():
                frames.add(
                    "dual_percentiles",
                    {
                        "probe": probe,
                        "month": str(month),
                        "pct": pct,
                        "days": value.value,
                        "over_cap": value.over_cap,
                        "lag_resolution": table.dual_lag_resolution,
                    },
                )
        for delta_day, count in dual_assessment.delta_histogram.items():
            frames.add(
                "dual_delta",
                {"probe": probe, "delta_day": int(delta_day), "row_count": int(count)},
            )
    if table.compare_event_time:
        for month, count in compare_mismatch_by_month(canonical).items():
            if _in_window(month, window):
                frames.add(
                    "compare_mismatch",
                    {"probe": probe, "month": str(month), "mismatches": count},
                )
    side = ParitySide(config=table, canonical=canonical, completion=completion, volume=volume)
    reads = {
        "target_logical_reads": canonical.target_logical_reads,
        "scan_budget_reads": canonical.scan_budget_reads,
        "scratch_logical_reads": canonical.scratch_logical_reads,
    }
    return statuses, side, frames, extraction_started, extraction_finished, reads


def _int_or_none(value):
    return None if pd.isna(value) else int(value)


# probe_runs read-accounting columns: ALWAYS present (None when the probe was
# skipped or aborted) so the physical table schema a first to_sql() fixes is
# identical no matter which probe outcome writes first
READ_COLUMNS = ("target_logical_reads", "scan_budget_reads", "scratch_logical_reads")

# the FROZEN snapshot column types (nullable pandas dtypes). Applied to every
# frame before it is written: to_sql() infers physical column types from the
# FIRST frame it sees, so a None-first numeric column (e.g. an insufficient
# run's freshness cadence) would otherwise freeze a varchar that silently
# turns later numbers into strings. Unlisted columns are strings.
TYPED_COLUMNS: dict[str, dict[str, str]] = {
    "statuses": {
        "status_schema_version": "Int64",
        # None-able / free-form text: an all-None or numeric-looking first
        # frame must not freeze a non-text physical column
        "reason": "string",
        "detail": "string",
    },
    "probe_runs": {
        "duration_seconds": "Float64",
        "extraction_started": "string",
        "extraction_finished": "string",
        **dict.fromkeys(READ_COLUMNS, "Int64"),
    },
    "population_buckets": {
        "row_count": "Int64",
        "n_curve_eligible": "Int64",
        "n_null_event_time": "Int64",
        "n_null_load_time_only": "Int64",
        "n_negative_clipped": "Int64",
        "n_negative_lag_excluded": "Int64",
        "n_overflow": "Int64",
        "n_join_unmatched": "Int64",
        "n_other_exclusions": "Int64",
        "n_base_rows": "Int64",
        "n_ambiguous_base_rows": "Int64",
        "n_compare_mismatch": "Int64",
        "distinct_keys": "Int64",
    },
    "completion_summary": {
        "complete_back_to": "string",
        "learned_wait": "Int64",
        "horizon": "Int64",
        "recommended_wait": "Int64",
        "negative_lag_excess_fraction": "Float64",
        "p50_mean": "Float64", "p50_std": "Float64",
        "p90_mean": "Float64", "p90_std": "Float64",
        "p95_mean": "Float64", "p95_std": "Float64",
        "p99_mean": "Float64", "p99_std": "Float64",
    },
    "volume_summary": {
        "gaps": "string",
        "baseline_median": "Float64",
        "baseline_sigma": "Float64",
        "forecast": "Float64",
        "duplicate_rows": "Int64",
    },
    "completion_percentiles": {"pct": "Int64", "days": "Int64", "over_cap": "boolean"},
    "month_volumes": {
        "volume": "Int64",
        "expected_low": "Float64",
        "expected_high": "Float64",
        "nowcast": "Float64",
        "deficit": "boolean",
    },
    "month_lag_cells": {"lag_day": "Int64", "row_count": "Int64"},
    "epoch_cells": {"row_count": "Int64"},
    "freshness": {
        "last_epoch": "string",
        "epoch_count": "Int64",
        "cadence_median_days": "Float64",
        "cadence_sigma_days": "Float64",
        "days_since_last": "Float64",
    },
    # batch ids are USER DATA: numeric-looking ids must never freeze a
    # numeric physical column that later textual ids cannot enter
    "batch_runs": {"rows": "Int64", "batch_id": "string"},
    "batch_months": {
        "runs": "Int64",
        "null_batch_rows": "Int64",
        "days_to_p50": "Int64",
        "days_to_p90": "Int64",
        "days_to_p95": "Int64",
        "days_to_p99": "Int64",
    },
    "dual_summary": {
        "n_null_source_only": "Int64",
        "n_delta_rows": "Int64",
        "negative_lag_excess_fraction": "Float64",
    },
    "dual_lag_cells": {"lag_day": "Int64", "row_count": "Int64"},
    "dual_percentiles": {"pct": "Int64", "days": "Int64", "over_cap": "boolean"},
    "dual_delta": {"delta_day": "Int64", "row_count": "Int64"},
    "compare_mismatch": {"mismatches": "Int64"},
    "parity_months": {"left_count": "Int64", "right_count": "Int64", "diff": "Int64",
                      "verdict": "string"},
}


def _typed(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    for column, dtype in TYPED_COLUMNS.get(name, {}).items():
        if column in frame.columns:
            frame[column] = frame[column].astype(dtype)
    return frame


# the contract versions this build writes under — serialized into every run
# manifest so a stored run names the exact contracts that produced it
COMPONENT_VERSIONS = {
    "config": CONFIG_SCHEMA_VERSION,
    "status": STATUS_SCHEMA_VERSION,
    "canonical": CANONICAL_SCHEMA_VERSION,
    "dual": DUAL_SCHEMA_VERSION,
    "snapshot": SNAPSHOT_SCHEMA_VERSION,
}


def _collect_statuses(frames: _Frames, probe: str, statuses: list[Status]) -> None:
    for status in statuses:
        frames.add(
            "statuses",
            {
                "probe": probe,
                **status.model_dump(mode="json"),
                "status_schema_version": STATUS_SCHEMA_VERSION,
            },
        )


def _stage_analysis(
    configs: list[ProbeConfig],
    meta: RunMeta,
    store,
    as_of: pd.Timestamp,
    window: tuple[pd.Timestamp, pd.Timestamp],
) -> list[Status]:
    frames = _Frames()
    all_statuses: list[Status] = []
    sides: dict[str, ParitySide] = {}
    probe_records = []
    store.begin_run(meta)
    try:
        for config in configs:
            engines: dict[bool, sa.Engine] = {}
            try:
                for table in config.tables:
                    if table.read_uncommitted not in engines:
                        engines[table.read_uncommitted] = _engine_for(
                            config.connection_url, table.read_uncommitted
                        )
                    engine = engines[table.read_uncommitted]
                    started = _wall()
                    statuses: list[Status] = []
                    extraction = (None, None)
                    reads = dict.fromkeys(READ_COLUMNS)
                    if not _table_exists(engine, table):
                        severity = Severity.SKIPPED if table.optional else Severity.RED
                        reason = (
                            ReasonCode.OPTIONAL_TABLE_ABSENT
                            if table.optional
                            else ReasonCode.MISSING_TABLE
                        )
                        statuses.append(
                            Status(
                                check=Check.PROBE,
                                severity=severity,
                                reason=reason,
                                detail=f"{table.database}.{table.table_schema}."
                                f"{table.table} does not exist",
                            )
                        )
                    else:
                        try:
                            (
                                statuses,
                                side,
                                probe_frames,
                                ext_start,
                                ext_end,
                                reads,
                            ) = _probe_one(engine, table, as_of, window)
                            frames.merge(probe_frames)  # only on full success
                            sides[table.probe_name] = side
                            extraction = (ext_start, ext_end)
                        except ProbeAborted as abort:
                            statuses = [
                                Status(
                                    check=Check.PROBE,
                                    severity=Severity.RED,
                                    reason=abort.reason,
                                    detail=abort.detail,
                                )
                            ]
                    finished = _wall()
                    probe_records.append(
                        {
                            "probe": table.probe_name,
                            "extraction_started": extraction[0].isoformat()
                            if extraction[0] is not None
                            else None,
                            "extraction_finished": extraction[1].isoformat()
                            if extraction[1] is not None
                            else None,
                            "duration_seconds": round((finished - started).total_seconds(), 3),
                            **reads,
                        }
                    )
                    _collect_statuses(frames, table.probe_name, statuses)
                    all_statuses += statuses
            finally:
                for engine in engines.values():
                    engine.dispose()
        # parity runs after every probe, over the declared pairs
        for config in configs:
            for table in config.tables:
                if table.parity_with is None:
                    continue
                left, right = sides.get(table.probe_name), sides.get(table.parity_with)
                if left is None or right is None:
                    missing = table.probe_name if left is None else table.parity_with
                    statuses = [
                        Status(
                            check=Check.PARITY,
                            severity=Severity.INDETERMINATE,
                            reason=ReasonCode.PARITY_PREREQ_UNIQUENESS,
                            detail=f"probe {missing!r} produced no results this run; "
                            "parity is unverifiable",
                        )
                    ]
                else:
                    parity = assess_parity(left, right, as_of)
                    statuses = parity.statuses
                    for row in parity.rows:
                        frames.add(
                            "parity_months",
                            {
                                "probe": table.probe_name,
                                "partner": table.parity_with,
                                "month": str(row.month),
                                "left_count": row.left_count,
                                "right_count": row.right_count,
                                "diff": row.diff,
                                "verdict": row.verdict,
                            },
                        )
                _collect_statuses(frames, table.probe_name, statuses)
                all_statuses += statuses
        for name, frame in frames.frames().items():
            store.write_table(meta.run_id, name, stamp(_typed(name, frame), meta))
        store.write_table(
            meta.run_id,
            "probe_runs",
            stamp(_typed("probe_runs", pd.DataFrame(probe_records)), meta),
        )
        manifest = {
            **dataclasses.asdict(meta),
            "probes": probe_records,
            "statuses": [
                {"probe": row["probe"], **{k: row[k] for k in ("check", "severity", "reason")}}
                for row in frames.rows.get("statuses", [])
            ],
            "stages": {"analysis": {"completed_at": _wall().isoformat()}},
            "component_versions": COMPONENT_VERSIONS,
        }
        store.commit_run(meta.run_id, manifest)
    except BaseException:
        store.abort_run(meta.run_id)  # the failed stage leaves NOTHING partial
        raise
    return all_statuses




def _statuses_from_manifest(manifest: dict) -> list[Status]:
    return [
        Status(
            check=Check(row["check"]),
            severity=Severity(row["severity"]),
            reason=ReasonCode(row["reason"]) if row["reason"] else None,
        )
        for row in manifest["statuses"]
    ]


def _artifacts_dir(config: ProbeConfig, run_id: str):
    """PER-RUN artifact directory: overlapping runs can never publish each
    other's artifacts (delivery verifies the marker against run_id too)."""
    from pathlib import Path

    return Path(config.delivery.worktree) / "artifacts" / validate_run_id(run_id)


def _stage_render(configs: list[ProbeConfig], store, run_id: str):
    """ARTIFACTS_RENDERED: emit the dashboard into the run's OWN worktree
    directory. The swap keeps the previous complete artifacts as a backup
    until the new directory is in place — no window where neither exists."""
    import shutil

    from metricprobe.publish import emit_dashboard

    final = _artifacts_dir(configs[0], run_id)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"{final.name}.tmp")
    previous = final.with_name(f"{final.name}.prev")
    for leftover in (tmp, previous):
        if leftover.exists():
            shutil.rmtree(leftover)
    emit_dashboard(store, run_id, configs, tmp)
    if final.exists():
        final.rename(previous)
    tmp.rename(final)
    if previous.exists():
        shutil.rmtree(previous)
    store.record_stage(
        run_id, "render", {"completed_at": _wall().isoformat(), "artifacts": str(final)}
    )
    return final


def _stage_publish(configs: list[ProbeConfig], store, run_id: str, run_at: str) -> list[str]:
    """PUBLISHED: push the rendered dashboard to every configured remote.
    deliver() verifies the artifact marker BELONGS to run_id and prepares
    every remote before pushing any; the stage is recorded only after EVERY
    push succeeded."""
    from metricprobe.publish import deliver

    pushed = deliver(
        _artifacts_dir(configs[0], run_id), configs[0].delivery, run_id, run_at
    )
    store.record_stage(
        run_id, "publish", {"completed_at": _wall().isoformat(), "remotes": pushed}
    )
    return pushed


def _run_finishing_stages(
    configs: list[ProbeConfig], store, run_id: str, run_at: str, start_stage: str
) -> int | None:
    """Render + publish for a committed run. Returns None on success, 1 on a
    stage failure (everything before it stays committed and recorded)."""
    if start_stage in ("analysis", "render"):
        try:
            _stage_render(configs, store, run_id)
            print(f"run {run_id}: artifacts rendered")
        except Exception as error:
            print(f"metricprobe: render stage failed: {error}", file=sys.stderr)
            return 1
    else:
        # resuming publish alone: the rendered artifacts must exist AND belong
        # to this run — anything else needs --resume-from render
        import json as _json

        from metricprobe.publish import PUBLISHED_MARKER

        marker = _artifacts_dir(configs[0], run_id) / PUBLISHED_MARKER
        if not marker.exists() or _json.loads(marker.read_text())["run_id"] != run_id:
            print(
                f"metricprobe: no rendered artifacts for run {run_id!r}; "
                "resume from the render stage instead",
                file=sys.stderr,
            )
            return 1
    try:
        pushed = _stage_publish(configs, store, run_id, run_at)
        print(f"run {run_id}: published to {', '.join(pushed)}")
    except Exception as error:
        print(f"metricprobe: publish stage failed: {error}", file=sys.stderr)
        return 1
    return None


def cmd_run(args) -> int:
    try:
        configs = [load_config(path) for path in args.config]
        compose_campaign(configs)
        if args.run_id:
            validate_run_id(args.run_id)
        if configs[0].delivery is not None:
            # fail BEFORE a long analysis, with the actionable install hint
            from metricprobe.report import ensure_static_export_available

            ensure_static_export_available()
    except (ConfigError, ValueError, RuntimeError) as error:
        print(f"metricprobe: {error}", file=sys.stderr)
        return 1
    run_at = _wall()
    try:
        # pd raises DateParseError (a ValueError) on malformed --as-of: a
        # usage error must exit 1, never escape as a traceback. as_of stays
        # NAIVE (it compares against naive event months throughout the
        # metrics); run_at is the aware wall clock.
        as_of = pd.Timestamp(args.as_of) if args.as_of else run_at.tz_localize(None)
        window_start, window_end = _parse_window(args, as_of)
    except (ConfigError, ValueError) as error:
        print(f"metricprobe: {error}", file=sys.stderr)
        return 1
    digest = _combined_digest(configs)
    probes = [table.probe_name for config in configs for table in config.tables]
    if args.dry_run:
        print(f"dry run: {len(probes)} probe(s): {', '.join(probes)}")
        print(f"as_of={as_of.isoformat()} window={window_start.date()}..{window_end.date()}")
        print(f"config digest {digest}")
        return 0
    run_id = args.run_id or f"{as_of:%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
    meta = RunMeta(
        run_id=run_id,
        run_at=run_at.isoformat(),
        as_of=as_of.isoformat(),
        git_sha=_git_sha(),
        tool_version=metricprobe.__version__,
        config_digest=digest,
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
    )
    configured = _configured_stages(configs[0])
    try:
        store = open_store(configs[0].store)
        if args.resume_from:
            if not args.run_id:
                print("metricprobe: --resume-from requires --run-id", file=sys.stderr)
                return 1
            if args.resume_from not in configured:
                print(
                    f"metricprobe: unknown stage {args.resume_from!r}; configured "
                    f"stages: {', '.join(configured)}",
                    file=sys.stderr,
                )
                return 1
            committed = {m["run_id"]: m for m in store.list_runs()}
            if args.run_id in committed:
                manifest = committed[args.run_id]
                if manifest["config_digest"] != digest:
                    print(
                        "metricprobe: config digest mismatch — the config changed "
                        f"since run {args.run_id!r} was created; refusing to resume",
                        file=sys.stderr,
                    )
                    return 1
                # analysis is committed: the idempotent retry redoes only the
                # configured stages AFTER it
                statuses = _statuses_from_manifest(manifest)
                if len(configured) == 1:
                    print(
                        f"run {args.run_id!r}: analysis already committed; "
                        "nothing to redo"
                    )
                    return exit_code_for(statuses)
                failure = _run_finishing_stages(
                    configs, store, args.run_id, manifest["run_at"], args.resume_from
                )
                return failure if failure is not None else exit_code_for(statuses)
            if args.resume_from != "analysis":
                print(
                    f"metricprobe: run {args.run_id!r} has no committed analysis; "
                    f"the {args.resume_from} stage needs one — resume from "
                    "analysis instead",
                    file=sys.stderr,
                )
                return 1
            registration = store.registration(args.run_id)
            if registration is None:
                # neither committed nor registered: no durable record exists to
                # verify the digest against — refusing is the only way to
                # honor the matching-digest rule
                print(
                    f"metricprobe: run {args.run_id!r} has no committed or "
                    "registered record to resume from; start a fresh run "
                    "without --resume-from",
                    file=sys.stderr,
                )
                return 1
            if registration["config_digest"] != digest:
                print(
                    "metricprobe: config digest mismatch — the config changed "
                    f"since run {args.run_id!r} was registered; refusing to resume",
                    file=sys.stderr,
                )
                return 1
            claim = store.staging_claim(args.run_id)
            if claim is not None:
                # a live staging claim means ANOTHER writer may still be
                # working this run id: aborting it here would delete its rows
                # while it can still commit — refuse instead. (A clean stage
                # failure always releases the claim; a present claim is either
                # an active writer or a crashed one needing explicit cleanup.)
                print(
                    f"metricprobe: run {args.run_id!r} still holds a staging "
                    f"claim ({claim}); another writer may be active. If that "
                    "writer is dead, run `metricprobe abort --config ... "
                    f"--run-id {args.run_id}` and retry",
                    file=sys.stderr,
                )
                return 1
            # a resumed run KEEPS its identity: as_of, window and run_at come
            # from the registration (the run id keeps meaning ONE analysis);
            # explicitly passed flags must agree or the resume is refused.
            # git_sha/tool_version are refreshed: they describe the execution.
            conflicts = []
            if args.as_of and pd.Timestamp(args.as_of) != pd.Timestamp(
                registration["as_of"]
            ):
                conflicts.append(f"--as-of {args.as_of} != {registration['as_of']}")
            if args.year or args.window:
                registered_as_of = pd.Timestamp(registration["as_of"])
                want_start, want_end = _parse_window(args, registered_as_of)
                if want_start != pd.Timestamp(
                    registration["window_start"]
                ) or want_end != pd.Timestamp(registration["window_end"]):
                    conflicts.append(
                        f"window {want_start.date()}..{want_end.date()} != "
                        f"{registration['window_start']}..{registration['window_end']}"
                    )
            if conflicts:
                print(
                    f"metricprobe: refusing to resume run {args.run_id!r} under a "
                    "different identity: " + "; ".join(conflicts),
                    file=sys.stderr,
                )
                return 1
            meta = RunMeta(
                **{
                    **registration,
                    "git_sha": _git_sha(),
                    "tool_version": metricprobe.__version__,
                }
            )
            as_of = pd.Timestamp(meta.as_of)
            window_start = pd.Timestamp(meta.window_start)
            window_end = pd.Timestamp(meta.window_end)
            run_id = meta.run_id
            # no claim exists, so nothing is staged on a claim-checking store;
            # sweep any orphan rows left by a writer that died mid-statement
            store.abort_run(args.run_id)
            print(f"run {args.run_id!r}: resuming failed analysis stage")
        elif args.run_id:
            # a fresh run must not silently reuse an id: the digest guard only
            # exists if the id keeps pointing at one durable record
            if store.registration(args.run_id) is not None or any(
                m["run_id"] == args.run_id for m in store.list_runs()
            ):
                print(
                    f"metricprobe: run id {args.run_id!r} is already registered or "
                    "committed; pass --resume-from analysis to retry it, or choose "
                    "a new run id",
                    file=sys.stderr,
                )
                return 1
        store.register_run(meta)
        statuses = _stage_analysis(configs, meta, store, as_of, (window_start, window_end))
    except Exception as error:  # a configured stage failed -> exit 1
        print(f"metricprobe: analysis stage failed: {error}", file=sys.stderr)
        return 1
    if configs[0].store.retention_runs:
        try:
            store.prune(configs[0].store.retention_runs)
        except Exception as error:  # retention is best-effort AFTER the commit
            print(f"metricprobe: warning: retention pruning failed: {error}", file=sys.stderr)
    reds = [s for s in statuses if s.severity is Severity.RED]
    print(f"run {run_id}: analysis committed — {len(statuses)} statuses, {len(reds)} red")
    if len(configured) > 1:
        failure = _run_finishing_stages(
            configs, store, run_id, meta.run_at, "analysis"
        )
        if failure is not None:
            return failure
    return exit_code_for(statuses)


def _parse_candidate_overrides(pairs: list[str] | None) -> dict[str, tuple[str, ...]]:
    """--candidates role=pattern1,pattern2 (repeatable): override any subset of
    the shipped role-candidate lists."""
    overrides: dict[str, tuple[str, ...]] = {}
    for pair in pairs or []:
        role, separator, patterns = pair.partition("=")
        if not separator or role not in DEFAULT_ROLE_CANDIDATES:
            raise ConfigError(
                f"--candidates must be role=patterns with role one of "
                f"{sorted(DEFAULT_ROLE_CANDIDATES)}; got {pair!r}"
            )
        needles = tuple(needle.strip() for needle in patterns.split(",") if needle.strip())
        if not needles:
            raise ConfigError(f"--candidates {pair!r} lists no patterns")
        overrides[role] = needles
    return overrides


def cmd_discover(args) -> int:
    try:
        candidates = _parse_candidate_overrides(args.candidates)
        engine = sa.create_engine(expand_env(args.url))
        try:
            draft = draft_config(
                engine, args.database, args.url, schema=args.schema, candidates=candidates
            )
        finally:
            engine.dispose()
        if args.out:
            from pathlib import Path

            Path(args.out).write_text(draft, encoding="utf-8")
            print(f"draft config written to {args.out}")
        else:
            print(draft, end="")
    except Exception as error:
        print(f"metricprobe: discover failed: {error}", file=sys.stderr)
        return 1
    return 0


def _latest_run_id(store, run_id: str | None) -> str:
    if run_id:
        return run_id
    runs = store.list_runs()
    if not runs:
        raise ConfigError("the store holds no committed runs")
    return runs[-1]["run_id"]


def cmd_report(args) -> int:
    """Standalone re-render of a committed run's interactive report."""
    from metricprobe.report import ensure_static_export_available, generate_report

    try:
        configs = [load_config(path) for path in args.config]
        compose_campaign(configs)
        ensure_static_export_available()
        store = open_store(configs[0].store)
        run_id = _latest_run_id(store, args.run_id)
        path = generate_report(store, run_id, configs, args.out)
    except (ConfigError, ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"metricprobe: {error}", file=sys.stderr)
        return 1
    print(f"run {run_id}: report written to {path}")
    return 0


def cmd_publish(args) -> int:
    """Standalone dashboard emission (--out DIR), or — with --deliver — the
    SAME finishing stages the campaign runs (render recorded, then publish):
    the campaign command stays the sole delivery code path, and the supplied
    config must still match the committed run's digest."""
    from metricprobe.publish import emit_dashboard
    from metricprobe.report import ensure_static_export_available

    try:
        configs = [load_config(path) for path in args.config]
        compose_campaign(configs)
        ensure_static_export_available()
        store = open_store(configs[0].store)
        run_id = _latest_run_id(store, args.run_id)
        if args.deliver:
            if args.out:
                print(
                    "metricprobe: --deliver renders into the delivery worktree; "
                    "pass either --out or --deliver, not both",
                    file=sys.stderr,
                )
                return 1
            if configs[0].delivery is None:
                print("metricprobe: --deliver requires a delivery config", file=sys.stderr)
                return 1
            manifest = next(m for m in store.list_runs() if m["run_id"] == run_id)
            if manifest["config_digest"] != _combined_digest(configs):
                print(
                    "metricprobe: config digest mismatch — the config changed "
                    f"since run {run_id!r} was created; refusing to deliver",
                    file=sys.stderr,
                )
                return 1
            failure = _run_finishing_stages(
                configs, store, run_id, manifest["run_at"], "render"
            )
            return 1 if failure is not None else 0
        if not args.out:
            print("metricprobe: pass --out DIR (or --deliver)", file=sys.stderr)
            return 1
        readme = emit_dashboard(store, run_id, configs, args.out)
        print(f"run {run_id}: dashboard written to {readme.parent}")
    except (ConfigError, ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"metricprobe: {error}", file=sys.stderr)
        return 1
    return 0


def cmd_abort(args) -> int:
    """Explicit cleanup for a crashed writer: releases the staging claim and
    sweeps the run's partial rows so `--resume-from` can proceed. Deliberately
    a SEPARATE, human-invoked command — the resume path itself must never
    silently take over a claim that may belong to a live writer."""
    try:
        configs = [load_config(path) for path in args.config]
        compose_campaign(configs)
        validate_run_id(args.run_id)
        store = open_store(configs[0].store)
        if any(m["run_id"] == args.run_id for m in store.list_runs()):
            print(
                f"metricprobe: run {args.run_id!r} is committed; abort never "
                "touches committed runs",
                file=sys.stderr,
            )
            return 1
        claim = store.staging_claim(args.run_id)
        store.abort_run(args.run_id)
        if claim is None:
            print(f"run {args.run_id!r}: no staging claim was held; nothing to release")
        else:
            print(f"run {args.run_id!r}: staging claim released ({claim})")
    except (ConfigError, ValueError) as error:
        print(f"metricprobe: {error}", file=sys.stderr)
        return 1
    return 0


def _unimplemented(command: str, step: str):
    def handler(args) -> int:
        print(f"metricprobe {command}: lands in {step}", file=sys.stderr)
        return 1

    return handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="metricprobe")
    parser.add_argument("--version", action="version", version=metricprobe.__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="execute all configured probes under one run")
    run.add_argument("--config", action="append", required=True, help="YAML config (repeatable)")
    # default applied in _parse_window: None distinguishes "not passed" so a
    # resume can tell an explicit window (must match the registration) from
    # the default (adopts the registration)
    run.add_argument("--window", default=None, help="rolling window, e.g. 24m (default)")
    run.add_argument("--year", type=int, help="fixed calendar year instead of --window")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume-from", help=f"stage to resume ({', '.join(STAGES)})")
    run.add_argument("--run-id", help="required with --resume-from; injectable for demos")
    run.add_argument("--as-of", help="freeze the analysis cutoff (injectable clock)")
    run.set_defaults(handler=cmd_run)

    discover = commands.add_parser(
        "discover", help="scan INFORMATION_SCHEMA and emit a draft config"
    )
    discover.add_argument("--url", required=True, help="SQLAlchemy URL (env-expandable)")
    discover.add_argument("--database", required=True)
    discover.add_argument("--schema", help="restrict the scan to one schema")
    discover.add_argument("--out", help="write the draft here instead of stdout")
    discover.add_argument(
        "--candidates",
        action="append",
        help="override a role's candidate patterns, e.g. event_time=admitted,seen "
        "(repeatable)",
    )
    discover.set_defaults(handler=cmd_discover)

    report = commands.add_parser(
        "report", help="render a committed run's self-contained HTML report"
    )
    report.add_argument("--config", action="append", required=True, help="YAML config")
    report.add_argument("--run-id", help="default: the latest committed run")
    report.add_argument("--out", required=True, help="output directory")
    report.set_defaults(handler=cmd_report)

    publish = commands.add_parser(
        "publish", help="emit (and optionally deliver) the markdown dashboard"
    )
    publish.add_argument("--config", action="append", required=True, help="YAML config")
    publish.add_argument("--run-id", help="default: the latest committed run")
    publish.add_argument("--out", help="output directory (standalone emission)")
    publish.add_argument(
        "--deliver", action="store_true",
        help="re-run the render+publish lifecycle stages for the run "
        "(renders into the delivery worktree; monotonic guard enforced)",
    )
    publish.set_defaults(handler=cmd_publish)

    abort = commands.add_parser(
        "abort",
        help="release a crashed run's staging claim and delete its partial "
        "state (committed runs are never touched)",
    )
    abort.add_argument("--config", action="append", required=True, help="YAML config")
    abort.add_argument("--run-id", required=True)
    abort.set_defaults(handler=cmd_abort)

    for command, step in (("serve", "Step 10"),):
        stub = commands.add_parser(command)
        stub.set_defaults(handler=_unimplemented(command, step))

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on usage errors, but 2 is RESERVED for data-health
        # RED; malformed invocations are execution errors (1)
        return 0 if exc.code == 0 else 1
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
