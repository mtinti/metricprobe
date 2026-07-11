# PLAN.md — metricprobe build plan

Read CLAUDE.md first: it defines the metrics, architecture, hard rules, and the
test-first method. This file is the ordered execution plan. Work strictly in order;
each step ends green before the next begins.

## Step 0 — Repo skeleton
- `pyproject.toml` (name: metricprobe, `requires-python = ">=3.12"`), src layout per
  CLAUDE.md architecture, ruff config, pytest config, MIT LICENSE file.
- CI workflow: two parallel jobs (fast suite / mssql container equivalence) on PR+main,
  GitHub/Gitea-compatible subset only (checkout, upload-artifact, plain `run:`).
  Pin the equivalence job's runner to Linux x86-64 (only platform Microsoft supports
  for the SQL Server container). Wire the equivalence HARNESS now with a trivial
  round-trip test — every later metric adds its equivalence cases in its own step.
- CI guard against private-material leaks: a tracked-content scan failing on
  environment-specific markers (real server/host names, private-file names) —
  .gitignore alone is insufficient. CLAUDE.private.md must NEVER enter this repo.
- Empty package modules with docstrings; CI green on a trivial test.

## Step 1 — Synthetic arrival generator (tests first means the generator IS first)
- `tests/synth/generator.py`: per event month, N rows, lag models:
  `lognormal(mu, sigma)` trickle; `step_batches([(day, fraction), ...])` with batch IDs;
  optional dual timestamps with known source→load offset.
- Deterministic seeding. Emits pandas DataFrame + loaders into DuckDB and (later) mssql.
- Pathology injectors: volume spike/drop, missing month, duplicate keys, straggler
  batch, raw-vs-corrected date deltas, and **sustained volume collapse** (loads keep
  arriving on their normal cadence, but the last K months carry ~10× fewer rows than
  the historical norm).
- Scenarios come in **healthy/unhealthy twin pairs**: same generating parameters
  except the injected pathology, so every check is tested in both directions —
  it must fire on the unhealthy twin AND stay silent on the healthy twin
  (false-positive tests are as mandatory as detection tests).
- Helper that computes EXPECTED days-to-p50/p90/p95/p99 from generator parameters.
- Unit-test the generator itself (row counts, seed determinism, expected-value helper
  against brute-force empirical percentiles).

## Step 2 — Config
- `ProbeConfig` / `TableConfig` (pydantic v2): required probe_name/load_time
  + full table locator (database.schema.table) + EXACTLY ONE of event_time XOR
  event_time_via (structured join spec w/ {base_col, lookup_col} pairs);
  optional source_insert_time, load_batch_col, group_by_alt, key_cols,
  compare_event_time, parity_with,
  optional flag, proxy flag, expect_batchy, resolution map, suppress_small_counts,
  read_uncommitted, and ALL analysis parameters (training_cutoff_days
  [validated >= lag_cap_days, both default 365], lag_cap_days,
  clock_skew_tolerance_days, batch/alt-grouping result-cell caps,
  min_mature_months, freshness bucket + min_epochs, evaluation window,
  amber/red thresholds, expected-fill band, parity_tolerance) with versioned
  defaults. Unknown fields REJECTED. **Freeze and version the complete schema HERE —
  no metric work starts against an incomplete contract.**
- **Three typed config layers**: probe/table config PLUS CampaignConfig (schedule,
  timezone, grace period, manual-run behavior), StoreConfig (store path, retention),
  DeliveryConfig (remotes, refs, worktree, token env-var NAMES). Config digest over
  the secret-redacted canonical form.
- **Typed result/status model frozen here too**: statuses with reason codes,
  precedence RED > AMBER > INDETERMINATE > INSUFFICIENT_HISTORY > SKIPPED > GREEN,
  and the CLI reduction to exit codes 0/1/2 — defined and tested before any
  metric exists.
- YAML loader with env-var expansion; validation failures are loud and specific.
- Failing tests first: valid configs load; each invalid shape produces a named error.

## Step 3 — Canonical aggregation + completion percentiles
- ONE canonical per-table pass with GROUPING SETS: (event_month, **lag_day** —
  computed per row as datediff(day, event_time, load_time)), (load_epoch_day) for
  freshness, (event_month, batch_id) when configured — batch completion needs the
  month dimension, (group_by_alt) when configured, AND the EMPTY () grouping set
  carrying the global scalars COUNT(*) vs COUNT(DISTINCT HASHBYTES-SHA256
  canonical key encoding) — never CHECKSUM; without () these compute per-set,
  not globally. The exact result schema incl. GROUPING_ID interpretation is
  FROZEN before any SQL is written. Numeric budgets: logical reads <= 3x a single
  full scan per probe (any distinct-count guard is counted); result cells per
  grouping set capped (config, default 100k) — exceeding either ABORTS the probe
  with a reason code, never returns unbounded rows.
  Every query carries `(load_time <= :as_of OR load_time IS NULL)` — test that the
  NULL bucket survives the predicate. lag_day capped with overflow bucket;
  **censoring-aware percentile test: overflow mass > (1−p) ⇒ percentile reported
  as "> cap" and recommended_wait refused** (twin: overflow below threshold ⇒
  normal). Scan budget measured in LOGICAL READS via a SQL Server performance test.
- Failing golden tests: trickle scenario and batchy scenario → expected percentiles
  within commented tolerance; accumulate-all-failures assertion style.
- Maturity tests (single-pass, per CLAUDE.md): learned_wait from the FIXED training
  cohort (month_end <= as_of − training_cutoff_days, default 365); classification
  horizon = max(cutoff, learned_wait); min_mature_months re-checked; NO iteration.
  Config validation test: training_cutoff_days < lag_cap_days is REJECTED.
  **Lag-support backtest (heuristic)**: a scenario whose true latency exceeds the
  cutoff must trip the strata-disagreement backtest and yield "insufficient
  history", never a confident underestimated wait; a shared-late-tail scenario
  documents the backtest's known blind spot (cannot prove completeness). Exposure test: a 10%-arrived
  immature month whose own curve shows "100%" is still classified immature.
- Negative-lag tests: within clock-skew tolerance → clipped; beyond → excluded and
  bucketed; excess fraction → RED; trap test: corrupt negative timestamps must NOT
  improve percentiles.
- Reconciliation tests: total_rows == sum of mutually exclusive buckets, all from
  the ONE aggregation; buckets include null_load and negative_lag_excluded.
- Join tests (event_time_via): {base_col, lookup_col} pairs incl. differently named
  keys; non-unique lookup fails loudly; unmatched counted; pre/post reconciliation;
  composite key case; piggybacks on the borrowed-event-time pass.
- Property tests: monotonic curves; row-order invariance.
- Algorithm formulas per docs/ALGORITHMS.md with HAND-CALCULATED boundary tests
  (recommended_wait, MAD scaling, zero-MAD fallback, F_mature median-of-curves,
  band incl. forecast dispersion).
- NOTE: "still filling" is NOT implemented here — it needs the mature-volume
  baseline from Step 4 (deliberate ordering).
- Compiled-SQL snapshot tests for duckdb and mssql dialects; equivalence cases NOW.

## Step 4 — Metric 1: volume history + baselines + still-filling
- Failing tests: spike scenario flagged by MAD outlier check (baseline from the same
  canonical aggregation — no separate query); missing month rendered as explicit
  gap; duplicate-key scenario trips the uniqueness check (when key_cols configured).
- **Sustained-collapse pair**: unhealthy twin (uploads continue on normal cadence,
  last 3 months at ~10% of typical volume) must yield volume=RED on MATURE collapsed
  months while freshness=GREEN; the healthy twin yields all-green. Requirements:
  (a) baseline computed from mature history EXCLUDING the evaluation window;
  (b) volume RED is assigned to MATURE months only; (c) partial current month
  excluded; (d) minimum history and MAD-zero fallback defined and tested.
- **Still-filling (needs this step's volume baseline — why it lives here):**
  age measured from MONTH_END (conservative — can only under-alarm); the open
  current month is EXCLUDED entirely (test: it appears as "open", never as a
  deficit). Expected count at month-end age t = independent mature-volume forecast
  × F_mature(t). Tautology guard test: the estimator must be able to FAIL — an
  immature month at 50% of expectation must be flagged; the inverted nowcast is
  reported but never fed back. Deficits are "arrival deficit — cause unresolved",
  never "volume collapse" (test both labels). Golden test: an immature month later
  matures into a confirmed collapse and the verdict upgrades.
- Statuses feed the frozen typed-status model (Step 2).
- Equivalence cases added.

## Step 5 — Metrics 3 (dual lag), 4 (batch), 5 (parity), 6 (freshness)
- Dual-lag golden test from the known source→load offset scenario; delta histogram
  data; rows with NULL source_insert_time form their own reported bucket (tested).
- Batch metrics from the step-batch scenario (rows per run, runs per month),
  weighted by batch row counts per ALGORITHMS.md.
- **Parity (metric 5)**: `parity_with` references two NAMED probes; FULL OUTER JOIN
  over event months (a one-sided month = explicit RED diff, tested); COMMON MATURE
  population; `parity_tolerance` (default 0) is valid only under the append-only +
  trustworthy-load_time watermark — a scenario violating it (injected duplicates)
  must yield INDETERMINATE, not a false mismatch. Status semantics, storage schema,
  golden twin test with injected missing-month divergence, plus equivalence cases.
- Freshness: cadence from DISTINCT ARRIVAL EPOCHS — batch IDs (epoch timestamp =
  MIN(load_time) within batch) when present, else load-time day buckets; never
  per-row gaps. Golden tests: healthy weekly feed passes; feed-stopped-3-cadences
  trips staleness; duplicate-timestamp bulk load does NOT corrupt cadence;
  minimum-epochs and zero-MAD fallback scenarios.
- Batch-level completion: implemented AND golden-tested from the step-batch scenario
  (not just promised).
- Horizon-date derivation ("complete back to: YYYY-MM-DD" = as_of − recommended
  wait) with a golden test tying it to the completion percentiles.
- Optional raw-vs-corrected side-stat (`compare_event_time`): count of rows where the
  two event-date columns differ, by month.
- Each metric adds its equivalence cases to the container job as it lands.

## Step 6 — Storage + CLI (lifecycle orchestration)
- **The campaign command owns the WHOLE lifecycle including DELIVERY in one
  process**: analyse → ANALYSIS_COMMITTED (atomic manifest) → ARTIFACTS_RENDERED →
  deliver (configured git commit + pushes for the dashboard) → PUBLISHED → exit
  with final status. PUBLISHED is claimed only after actual delivery succeeded.
  The CI workflow is ONE invocation plus an alert reader — never a chain of
  publish steps. Per-stage atomicity with idempotent retry: `--resume-from
  <stage> --run-id <id>` requires the run_id AND a matching config digest;
  delivery performs a monotonic-publication check (an older failed run can never
  overwrite a newer published dashboard — tested). Injected-failure tests at each
  stage boundary. Exit outcomes 0/1/2 are integration-tested under a real
  Actions-style runner (verified again on Gitea in the private bootstrap).
- Snapshot schema with run_id, run_at, as_of, extraction start/end per probe,
  git_sha, tool version, config digest, schema_version, window; parquet/duckdb
  writer (default) and mssql writer (config-flagged) sharing one interface.
- CLI: `run` executes ALL configured probes sequentially under ONE
  run_id/as_of/manifest. Flags: --window 24m default | --year YYYY | --config
  repeatable | --dry-run | --resume-from + --run-id. Also `report`, `serve`,
  `discover`, `publish` (standalone re-render/deliver of a committed run; the
  campaign calls the same code — emitter lands in Step 9, until then the
  campaign's render and delivery stages are simply NOT CONFIGURED).
- CLI contract tests HERE test exits relative to CONFIGURED stages, and at Step 6
  the ONLY configured terminal state is ANALYSIS_COMMITTED (no renderer exists yet;
  claiming ARTIFACTS_RENDERED would be a lie): exit 0/2 mean analysis committed,
  no RED / with RED. ARTIFACTS_RENDERED and PUBLISHED semantics + tests land in
  Step 9 with the emitters.
  Exit 1 = configured-stage failure with earlier stages honestly reported;
  missing optional table ⇒ SKIPPED status; parity INDETERMINATE ⇒ exit 0,
  grey-questioned on outputs.

## Step 7 — Equivalence coverage audit
- The harness exists since Step 0 and each metric added its cases as it landed;
  this step is the AUDIT: verify every metric, every pathology scenario, and every
  dialect-sensitive SQL construct has an equivalence case; close any gaps; make the
  coverage list explicit in the test suite so a new metric without equivalence
  cases fails review.

## Step 8 — discover
- Productize the INFORMATION_SCHEMA scanner: date/time column inventory, role-candidate
  matching (candidate lists are shipped defaults, overridable), draft YAML emission.
- Test against DuckDB information_schema; mssql path covered in equivalence job.

## Step 9 — Static report
- Prerequisite check: static export needs kaleido + an installed Chrome/Chromium
  (current Plotly requirement) — verify at startup with a clear actionable error;
  document install for locked-down Windows.
- Self-contained HTML (embedded Plotly.js) + per-figure PNGs. Figures: volume bars w/
  outliers+gaps; completion tabs (curves default w/ median + p10–p90 band; heatmap
  event-month × lag-week); percentile summary dot-lines w/ mean±std band; dual-lag
  overlay + delta histogram; batch charts; parity diverging bars.
- Smoke tests: every figure builds for every scenario; zero external URLs in HTML;
  suppression flag honored end-to-end.
- `metricprobe publish`: markdown dashboard emitter — a forge-renderable README
  (status table per database/table answering healthy? / updating? / complete back
  to: DATE, with the FULL status vocabulary — ✅ green, ⚠️ amber, 🔴 red,
  ❓ grey indeterminate, ⏳ insufficient history, ➖ skipped — and embedded
  committed **SVG** figures — small,
  text-based, git-delta-friendly, rendered natively by Gitea/GitHub) + the
  self-contained HTML committed alongside for download. The README opens with a
  status block: Generated-at, run number, git SHA, analysed window, and
  "**Next update expected by:** <date>" (computed from the schedule cadence) so a
  silently-dead pipeline is self-evident to any reader. Designed to be
  auto-committed by a scheduled workflow so the repo front page IS the dashboard.
  Tests: renders for every scenario; image links are relative paths that exist;
  next-expected date present; suppression honored.

## Step 10 — Streamlit app
- Pages: Overview (traffic lights + "complete back to: DATE" headline), per-table detail
  (period/lag selectors, curve/heatmap tabs), Runs (cross-run p95 trend, run table
  with git_sha + status).
- Snapshots-first; per-table "refresh now" behind config flag; snapshot timestamp
  always visible; telemetry disabled.

## Step 11 — Examples, docs, release
- `examples/demo.py`: generate synthetic data → probe → report → serve (the public demo).
- **Committed demo dashboard: `reports/README.md` + `reports/img/*.svg`** in this
  public repo, generated by `metricprobe publish` from a fixed-seed synthetic world
  of FOUR fake databases spanning different domains — deliberately not
  single-industry, to show the tool is generic:
  `demo_retail` (orders feed, lognormal trickle arrivals, dual timestamps:
  order placed vs. warehouse-loaded), `demo_sensors` (IoT telemetry, near-real-time
  with straggler devices), `demo_finance` (card settlements, monthly step batches
  with a batch-run ID), and `demo_health` (episode records: slow trickle over weeks
  with a long tail, dual timestamps, one registry-like table whose upstream lag
  dominates — the healthcare-flavoured case, kept generic in naming). Across them, a mix of healthy and unhealthy twin tables —
  at minimum: an all-green table, a stale one (freshness 🔴), a sustained-collapse
  one (updating ✅ / volume 🔴), a missing-month one, and a batchy-but-healthy one —
  so every verdict combination is visible.
  Each row answers: healthy? / updating? / complete back to: DATE.
  Regenerated by a CI job with a diff check — byte-stable because clock, run_id,
  and git metadata are INJECTED (frozen for the demo build) and SVG output is
  canonicalized; fixed seed covers the data itself.
- README links to `reports/README.md` as "what the output looks like", with
  synthetic-data screenshots.
- Generic workflow template in examples/.
- Tag v0.1.0. Until the PyPI job exists (phase 2), the supported install path is
  the git tag: `pip install git+https://github.com/<user>/metricprobe@v0.1.0`.
  Downstream deployments pin this exact reference.

## Later / phase 2 backlog
- Parity metric generalized beyond two-copy case.
- PyPI publishing job on tag.
- Additional dialects (postgres) once a second real user exists.
