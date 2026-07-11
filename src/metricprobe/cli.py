"""Command-line interface and the ONE orchestration state machine.

The campaign command (`metricprobe run`) owns the whole lifecycle in one
process: analyse -> ANALYSIS_COMMITTED (atomic manifest) [-> ARTIFACTS_RENDERED
-> PUBLISHED once the emitters exist, Step 9]. Exit codes are relative to the
CONFIGURED stages — at this step the only configured terminal state is
ANALYSIS_COMMITTED (no renderer exists yet; claiming more would be a lie):

    0 = all configured stages completed, no data-health RED
    2 = all configured stages completed, at least one RED (committed FIRST)
    1 = a configured stage failed: everything before it remains committed and
        honestly reported; the failed stage leaves NOTHING partial

Probes execute SEQUENTIALLY, never in parallel (assume every probe is a full
scan of a possibly unindexed production table); each probe's wall-clock
duration and extraction start/end land in the manifest. Clock, run_id and git
metadata are INJECTABLE (--as-of, --run-id, METRICPROBE_RUN_AT,
METRICPROBE_GIT_SHA) so demo builds are byte-stable.
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
from metricprobe.config import ConfigError, ProbeConfig, TableConfig, config_digest, load_config
from metricprobe.extract.canonical import ProbeAborted, run_canonical
from metricprobe.extract.dual import run_dual_lag
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
from metricprobe.status import Check, ReasonCode, Severity, Status, exit_code_for
from metricprobe.store import SNAPSHOT_SCHEMA_VERSION, RunMeta, open_store, stamp

STAGES = ("analysis",)  # render + delivery join in Step 9 with the emitters


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
    if args.year:
        start = pd.Timestamp(year=args.year, month=1, day=1)
        return start, start + pd.DateOffset(years=1)
    if not args.window.endswith("m"):
        raise ConfigError(f"--window must look like '24m', got {args.window!r}")
    months = int(args.window[:-1])
    return as_of - pd.DateOffset(months=months), as_of


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

    def frames(self) -> dict[str, pd.DataFrame]:
        return {name: pd.DataFrame(rows) for name, rows in self.rows.items() if rows}


def _collect_statuses(frames: _Frames, probe: str, statuses: list[Status]) -> None:
    for status in statuses:
        frames.add("statuses", {"probe": probe, **status.model_dump(mode="json")})


def _probe_one(
    engine, table: TableConfig, as_of: pd.Timestamp, frames: _Frames
) -> tuple[list[Status], ParitySide | None]:
    """One probe: canonical pass + every configured metric. Data-safety refusals
    (ProbeAborted) become typed RED statuses; the run continues."""
    canonical = run_canonical(engine, table, as_of)
    completion = assess_completion(canonical, table, as_of)
    volume = assess_volume(canonical, table, as_of, completion)
    freshness = assess_freshness(canonical, table, as_of)
    statuses = completion.statuses + volume.statuses + freshness.statuses
    probe = table.probe_name

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
    for month, per_pct in completion.percentiles.items():
        for pct, value in per_pct.items():
            frames.add(
                "completion_percentiles",
                {
                    "probe": probe,
                    "month": str(month),
                    "pct": pct,
                    "days": value.value,
                    "over_cap": value.over_cap,
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
    if table.source_insert_time:
        dual = run_dual_lag(
            engine, table, as_of, prior_target_reads=canonical.target_logical_reads
        )
        dual_assessment = assess_dual_lag(dual, table, as_of)
        statuses += dual_assessment.statuses
        for month, per_pct in dual_assessment.source_percentiles.items():
            for pct, value in per_pct.items():
                frames.add(
                    "dual_percentiles",
                    {
                        "probe": probe,
                        "month": str(month),
                        "pct": pct,
                        "days": value.value,
                        "over_cap": value.over_cap,
                    },
                )
        for delta_day, count in dual_assessment.delta_histogram.items():
            frames.add(
                "dual_delta",
                {"probe": probe, "delta_day": int(delta_day), "row_count": int(count)},
            )
    if table.compare_event_time:
        for month, count in compare_mismatch_by_month(canonical).items():
            frames.add(
                "compare_mismatch", {"probe": probe, "month": str(month), "mismatches": count}
            )
    side = ParitySide(config=table, canonical=canonical, completion=completion, volume=volume)
    return statuses, side


def _stage_analysis(
    configs: list[ProbeConfig], meta: RunMeta, store, as_of: pd.Timestamp
) -> list[Status]:
    frames = _Frames()
    all_statuses: list[Status] = []
    sides: dict[str, ParitySide] = {}
    probe_records = []
    store.begin_run(meta)
    try:
        for config in configs:
            engine = sa.create_engine(config.connection_url)
            try:
                for table in config.tables:
                    started = pd.Timestamp.now()
                    statuses: list[Status] = []
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
                            statuses, side = _probe_one(engine, table, as_of, frames)
                            sides[table.probe_name] = side
                        except ProbeAborted as abort:
                            statuses = [
                                Status(
                                    check=Check.PROBE,
                                    severity=Severity.RED,
                                    reason=abort.reason,
                                    detail=abort.detail,
                                )
                            ]
                    finished = pd.Timestamp.now()
                    probe_records.append(
                        {
                            "probe": table.probe_name,
                            "extraction_started": started.isoformat(),
                            "extraction_finished": finished.isoformat(),
                            "duration_seconds": round((finished - started).total_seconds(), 3),
                        }
                    )
                    _collect_statuses(frames, table.probe_name, statuses)
                    all_statuses += statuses
            finally:
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
            store.write_table(meta.run_id, name, stamp(frame, meta))
        store.write_table(
            meta.run_id, "probe_runs", stamp(pd.DataFrame(probe_records), meta)
        )
        manifest = {
            **dataclasses.asdict(meta),
            "probes": probe_records,
            "statuses": [
                {"probe": row["probe"], **{k: row[k] for k in ("check", "severity", "reason")}}
                for row in frames.rows.get("statuses", [])
            ],
            "stages": {"analysis": {"completed_at": pd.Timestamp.now().isoformat()}},
        }
        store.commit_run(meta.run_id, manifest)
    except BaseException:
        store.abort_run(meta.run_id)  # the failed stage leaves NOTHING partial
        raise
    return all_statuses


def check_monotonic_publication(candidate_run_at: str, published_run_ats: list[str]) -> None:
    """Delivery guard (wired to real delivery in Step 9): an older run may never
    overwrite a dashboard published from a newer run."""
    newer = [stamp for stamp in published_run_ats if stamp > candidate_run_at]
    if newer:
        raise RuntimeError(
            f"refusing to publish run from {candidate_run_at}: a newer run "
            f"({max(newer)}) is already published"
        )


def cmd_run(args) -> int:
    try:
        configs = [load_config(path) for path in args.config]
    except ConfigError as error:
        print(f"metricprobe: {error}", file=sys.stderr)
        return 1
    run_at = pd.Timestamp(os.environ.get("METRICPROBE_RUN_AT") or pd.Timestamp.now())
    as_of = pd.Timestamp(args.as_of) if args.as_of else run_at
    try:
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
    try:
        store = open_store(configs[0].store)
        if args.resume_from:
            if not args.run_id:
                print("metricprobe: --resume-from requires --run-id", file=sys.stderr)
                return 1
            if args.resume_from not in STAGES:
                print(
                    f"metricprobe: unknown stage {args.resume_from!r}; configured "
                    f"stages: {', '.join(STAGES)}",
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
                # analysis already committed: idempotent retry has nothing to do
                statuses = [
                    Status(
                        check=Check(row["check"]),
                        severity=Severity(row["severity"]),
                        reason=ReasonCode(row["reason"]) if row["reason"] else None,
                    )
                    for row in manifest["statuses"]
                ]
                print(f"run {args.run_id!r}: analysis already committed; nothing to redo")
                return exit_code_for(statuses)
        statuses = _stage_analysis(configs, meta, store, as_of)
    except Exception as error:  # a configured stage failed -> exit 1
        print(f"metricprobe: analysis stage failed: {error}", file=sys.stderr)
        return 1
    if configs[0].store.retention_runs:
        store.prune(configs[0].store.retention_runs)
    reds = [s for s in statuses if s.severity is Severity.RED]
    print(
        f"run {run_id}: analysis committed — {len(statuses)} statuses, {len(reds)} red"
    )
    return exit_code_for(statuses)


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
    run.add_argument("--window", default="24m", help="rolling window, e.g. 24m (default)")
    run.add_argument("--year", type=int, help="fixed calendar year instead of --window")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--resume-from", help=f"stage to resume ({', '.join(STAGES)})")
    run.add_argument("--run-id", help="required with --resume-from; injectable for demos")
    run.add_argument("--as-of", help="freeze the analysis cutoff (injectable clock)")
    run.set_defaults(handler=cmd_run)

    for command, step in (
        ("report", "Step 9"),
        ("publish", "Step 9"),
        ("serve", "Step 10"),
        ("discover", "Step 8"),
    ):
        stub = commands.add_parser(command)
        stub.set_defaults(handler=_unimplemented(command, step))

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
