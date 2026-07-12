# CLAUDE.md — metricprobe (public repo)

Repo description (use verbatim on GitHub):
"When can you trust that a month of data is complete? Measures arrival latency and
completeness for SQL tables — completion curves, freshness and volume checks, and a
git-friendly status dashboard."
README tagline: "Data arrival latency & completeness probes for database tables."

## What this project is

`metricprobe` is a Python package that measures **data arrival latency and completeness**
for database tables. Given a table with an *event time* column (when the fact happened)
and a *load time* column (when the row landed in the warehouse), it answers:

1. **Volume history** — rows per event month over full history, with outlier detection
   (robust, median ± k·MAD) and sanity checks (gaps, zero months, duplicate keys).
2. **Completion curves** — for each event month in a probe window, the cumulative % of
   the month's *final* row count (final = count of rows with `load_time <= :as_of`)
   as a function of **lag_day = datediff(day, event_time, load_time), computed
   per row in SQL** — the aggregation grain is event_month × lag_day (event_month ×
   load_day CANNOT recover lag; two same-month events loaded the same day have
   different lags). `as_of` is a **common analysis cutoff**, frozen at campaign
   start and applied on EVERY query as `(load_time <= :as_of OR load_time IS NULL)`
   — the bare `<=` predicate would silently delete the NULL-load bucket that
   reconciliation requires; NULL-load counts are acknowledged as query-time counts,
   not reproducible as-of counts. It makes results reproducible against a growing
   append-only table but is NOT a database snapshot; per-probe extraction start/end
   times are recorded.
   From each curve: days-to-p50/p90/p95/p99, then mean/std across mature months.
   **Maturity (single-pass, no iteration):** learned_wait is computed from the FIXED
   training cohort (months with month_end <= as_of − training_cutoff_days, default
   365 — days, like every duration in this system); the classification horizon =
   max(training_cutoff_days, learned_wait) (monotone by construction); mature months
   = month_end <= as_of − horizon; re-check `min_mature_months` against THAT set,
   else "insufficient history". **The cutoff must cover the modeled lag support,
   enforced by config validation: training_cutoff_days >= lag_cap_days** (defaults:
   both 365). A stability backtest across older cohort strata (e.g. 12–18 vs 18–24
   months old) is a HEURISTIC completeness check — it detects disagreement but
   cannot prove completeness, since all strata can share the same very late tail;
   tails beyond the oldest observed cohorts remain unknowable without an external
   finality signal, and the docs say so. Backtest disagreement ⇒ "insufficient
   history", never a confidently wrong wait.
   **"Still filling" (immature months):** age is measured conservatively from
   MONTH_END (rows within a month have differing exposure; month-end age
   under-states fill expectation, so it can only under-alarm, never false-alarm);
   the current open month is EXCLUDED entirely (its events haven't all occurred).
   Expected count at month-end age t = independent mature-volume forecast ×
   F_mature(t); the inverted nowcast is reported but never fed back. Deficits are
   "arrival deficit — cause unresolved"; the collapse verdict is mature-only.
   **Negative lags** (load before event): clipped to 0 only within a configured
   clock-skew tolerance (default 1 day); beyond it EXCLUDED from percentiles and
   counted in their own bucket, with a RED threshold on the excess fraction —
   clipping everything would let timestamp corruption IMPROVE the percentiles.
   **Populations (reconciliation contract):** mutually exclusive buckets from the
   canonical aggregation: total_rows = curve_eligible + null_event_time +
   null_load_time_only + negative_lag_excluded + other_exclusions, each reported.
3. **Dual lag** — when a table carries both a source-side insert timestamp and a local
   load timestamp, the same curves computed on each, plus the per-row delta
   (separates upstream provider lag from local ingestion lag).
4. **Batch metrics** (optional, when a `load_batch_col` exists) — each batch gets a
   canonical timestamp (MIN(load_time) within the batch); rows per run, runs per
   month, batch-level completion (tested, not just promised).
5. **Parity** (optional) — references two NAMED probes (`parity_with`), compares the
   COMMON MATURE population by event month via FULL OUTER JOIN (a month present on
   one side only is an explicit RED diff, not silently dropped), with
   `parity_tolerance` (default 0). **Exact parity compares only rows with
   load_time <= :as_of (non-NULL)** — NULL-load rows are query-time counts, not
   watermarked, so they are compared separately and informationally, never inside
   the exact diff. Zero-tolerance parity on the watermarked population is sound
   ONLY under VERIFIED prerequisites, checked automatically per run: the
   uniqueness check is configured (key_cols present) AND green on BOTH sides,
   read_uncommitted is DISABLED on both connections, and the negative-lag excess
   bucket (the backdating proxy) is below threshold on both sides. If any
   prerequisite is unverifiable or fails, parity reports INDETERMINATE with the
   failing prerequisite as its reason code — never a false mismatch.
6. **Freshness** — cadence is derived from **distinct arrival epochs**, never per-row
   gaps (row-level timestamps measure row frequency, not feed cadence): epochs =
   distinct batch IDs when `load_batch_col` exists, else distinct load-time days
   (configurable bucket). Learned cadence = median ± MAD of inter-epoch gaps, with a
   minimum-epochs requirement (default 5, else "insufficient history") and a zero-MAD
   fallback (perfectly regular feeds use a small fixed tolerance). Staleness flag:
   time since last epoch vs learned cadence. Answers "is this table still updated?".

Headline outputs per table, phrased as answers: **healthy?** (traffic light),
**updating?** (freshness flag), and **complete back to: YYYY-MM-DD** = `as_of` −
recommended_wait, explicitly meaning "months ending on or before this date are
expected ≥95% complete" (recommended_wait is the p95+2σ formula). This is
deliberately DISTINCT from the maturity horizon max(training_cutoff, learned_wait),
which is a stricter classification threshold, not a user-facing promise.

Outputs are tiny aggregate tables (never raw rows), stored as snapshots, and rendered
as Plotly figures via a static HTML/PNG report and a Streamlit app.

## Hard rules

- **This repo is public. It must contain ZERO environment-specific details.** No real
  server names, database names, schema/column mappings, row counts, or screenshots of
  real data. Anything environment-shaped enters only via user config. Demo material
  uses the synthetic generator exclusively.
- **Only bounded aggregate results leave the database, under a per-table scan
  budget measured in SQL Server LOGICAL READS, not statement count.** The canonical
  pass uses GROUPING SETS over one scan where the engine allows:
  (event_month, lag_day) for completion/volume; (load_epoch_day) for freshness;
  (event_month, batch_id) when `load_batch_col` is configured — batch-level
  completion needs the month dimension since a batch can span cohorts;
  (group_by_alt value) when configured; and the EMPTY () grouping set carrying the
  global scalars COUNT(*) vs COUNT(DISTINCT canonical key hash) for uniqueness
  (without () they would compute per-set, not globally). The () set is REALIZED
  as a UNION ALL branch of the same statement: measured on SQL Server, the
  distinct-count inside the GROUPING SETS plan spools per row (~150x one scan),
  which would violate the read budget this same rule mandates; a golden test
  proves the branch is row-for-row identical to the literal () grouping set —
  the hash is HASHBYTES SHA-256 over a **type-tagged, length-prefixed binary
  encoding with a distinct NULL sentinel** (delimiter concatenation is ambiguous:
  'a|b','c' and 'a','b|c' collide before hashing), and the check is documented as
  probabilistic (SHA-256 collision odds negligible; encoding ambiguity was the real
  risk). NEVER 32-bit CHECKSUM. Numeric budgets: logical reads on the PROBED
  tables <= 3x one full scan per probe, cumulative across passes; the
  aggregation/distinct-count guard and any join-validation spool run against the
  probe's own tempdb staging and carry their own enforced fail-closed bounds
  (ALGORITHMS.md section 15 — a single combined 3x ledger is unsatisfiable for
  tables narrower than their staging projection); result cells per grouping set
  capped (config, default 100k) — exceeding ANY of these ABORTS the probe with a
  reason code. The exact result schema incl. GROUPING_ID interpretation is frozen before
  SQL is written. lag_day is capped with an overflow bucket, and **percentiles are censoring-aware: if the overflow mass exceeds
  (1 − p), the percentile is reported only as "> cap" and NO precise
  recommended_wait is computed from it** (status: insufficient/censored). Dual-lag
  adds at most one more pass; join validation piggybacks on the borrowed-event-time
  pass. Budget verified by a realistic SQL Server performance test reading
  STATISTICS IO. Predicates must be sargable. Never pull raw rows into pandas.
- **No hardcoded database targeting in SQL.** Database/schema/table always come from
  config; never embed a `USE` statement or a literal database name in generated SQL.
- **One orchestration state machine, executable in CI — and the campaign command is
  the SOLE delivery owner.** It runs the whole lifecycle in one process — analyse →
  commit analysis → render artifacts → deliver (including any configured git
  commit/push of the dashboard) — and only THEN returns the final status; the CI
  workflow is one invocation plus an alert reader, never a chain of publish steps
  (a nonzero exit mid-chain would make Actions skip them and hide RED).
  **PUBLISHED is claimed only after the actual delivery succeeded**, not after
  rendering. Staged states with per-stage atomicity and idempotent retry:
  ANALYSIS_COMMITTED → ARTIFACTS_RENDERED → PUBLISHED. Retry (`--resume-from`)
  requires an explicit run_id and a matching config digest, and delivery performs a
  monotonic-publication check (an old failed run may never overwrite a newer
  published dashboard). **Exit codes are relative to the CONFIGURED stages**:
  exit 0 = all configured stages completed, no RED; exit 2 = all configured stages
  completed, ≥1 data-health RED; exit 1 = a configured stage failed (everything
  before it remains committed and honestly reported; the failed stage leaves
  nothing partial). PUBLISHED semantics apply only once delivery is configured
  and implemented. Statuses are TYPED with reason codes and frozen precedence
  (RED > AMBER > INDETERMINATE > INSUFFICIENT_HISTORY > SKIPPED > GREEN;
  INDETERMINATE renders as grey-questioned on dashboards and reduces to exit 0),
  frozen alongside the config schema before any metric work.
- **Analysis commits are atomic.** begin_run → staging → atomic run-manifest commit
  (or abort). Readers (report/publish/app) only read manifest-committed runs.
- **Reports must work offline.** Static HTML embeds Plotly.js (no CDN) and
  carries a Content-Security-Policy that FORBIDS network fetches (browser-
  enforced, not just linted). Streamlit telemetry (`gatherUsageStats`) is
  disabled in shipped config. A test enforces zero external RESOURCE LOADS and
  the CSP's presence in report output (the vendored Plotly.js source contains
  inert URL string literals — attribution links and exporter constants that are
  never fetched, and that the CSP would block if they ever were).
- **Every output row is stamped** with `run_id`, `run_at`, `as_of`, `git_sha`,
  tool version, config digest, `schema_version`, and the analysed window, from the
  very first release.
- **Small-count suppression** (`suppress_small_counts: true` → values < 5 render "<5")
  is applied in a shared presentation-data transformation BEFORE any serialization —
  a suppressed value must not survive in Plotly JSON, hover payloads, markdown
  tables, or SVG internals. Tests inspect generated HTML/SVG/markdown content for
  leaked raw values. Off by default.

## Architecture

```
metricprobe/
  config.py        # ProbeConfig / TableConfig (pydantic), YAML loader, validation
  extract/         # SQL builders (SQLAlchemy Core), dialect-aware: mssql + duckdb
  metrics/         # volume.py, completion.py, dual_lag.py, batch.py, parity.py
  store/           # writers: parquet/duckdb (default), mssql schema (flagged)
  viz/             # plotly figure builders (single source for app + report)
  report.py        # static self-contained HTML + per-figure PNGs
  publish.py       # markdown dashboard emitter (forge-renderable README + committed
                   # SVG figures; git forges serve raw HTML as plain text, so
                   # markdown is the in-repo dashboard format). The README opens
                   # with a status block: Generated-at, run number, git SHA,
                   # analysed window, and "Next update expected by: <date>" so
                   # staleness is self-diagnosing on a static page.
  app.py           # streamlit: Overview / per-table detail / Runs pages
  cli.py           # discover | run | report | publish | serve
  discover.py      # INFORMATION_SCHEMA scan -> draft YAML with role candidates
tests/
  synth/           # parametric synthetic arrival generator (see below)
  unit/            # config validation, SQL-builder snapshot tests per dialect
  golden/          # generator -> DuckDB -> metrics -> expected values w/ tolerance
  properties/      # invariants (see below)
  equivalence/     # same dataset on DuckDB AND SQL Server container -> same numbers
examples/          # demo_config.yaml (placeholder server), generic workflow.yml, demo.py
reports/           # committed DEMO dashboard (README.md + img/*.svg) generated from
                   # fixed-seed synthetic scenarios — healthy AND unhealthy twin
                   # tables across four fake databases in DIFFERENT domains
                   # (retail orders / IoT telemetry / finance settlements /
                   # health episode records) so the demo reads as general-purpose,
                   # with healthcare as one domain among several.
                   # Regenerated + diff-checked in CI. Synthetic data ONLY.
```

Key config semantics:
- Every probe entry has a required unique `probe_name` (a table can be probed multiple
  times with different event_time or grouping — variants are first-class, named in
  snapshots, dashboard rows, and figures).
- `event_time` XOR `event_time_via` (exactly one, validated); `load_time` required;
  `source_insert_time`, `load_batch_col`, `group_by_alt`, `key_cols` (uniqueness
  check), `compare_event_time` (raw-vs-corrected side-stat), `parity_with`
  (references another probe by its CAMPAIGN-WIDE probe_name — never a raw db.table
  pair) optional. **Config is three typed layers: CampaignConfig (schedule,
  timezone, grace period, manual-run behavior — "Next update expected by" is a
  campaign property, never per-probe), StoreConfig (snapshot store location,
  retention), DeliveryConfig (remotes, refs, worktree, token ENV VAR NAMES — never
  token values).** Config digests are computed over the secret-redacted canonical
  form.
  Per-column `resolution` (date | datetime) declared; curves computed at daily lag
  grain, coarser fallback labelled in output. Analysis parameters are ALL explicit
  config with versioned defaults: training_cutoff_days, min_mature_months,
  freshness bucket + min_epochs, evaluation window length, amber/red thresholds,
  expected-fill band width, parity_tolerance. Unknown fields are REJECTED (typo
  safety). The full schema is FROZEN AND VERSIONED before any metric work begins.
- `event_time_via: {join_table, on: [{base_col, lookup_col}, ...], column}` borrows
  event time from a related table — explicit column PAIRS, so differently named join
  keys map correctly; composite keys supported. Join contract: the lookup side MUST
  be unique on the join key (asserted at probe time); unmatched and ambiguous base
  rows are counted and reported; base-row count must reconcile pre/post join.
- **Algorithms are frozen, not just thresholds**: exact formulas live in a versioned
  spec (docs/ALGORITHMS.md) with hand-calculated boundary tests — recommended_wait =
  ceil(mean(p95_days) + 2·std(p95_days)); MAD scaled ×1.4826; zero-MAD fallback =
  max(scaled MAD, rel_tol·median); default baseline non-seasonal (seasonal
  month-of-year median requires ≥24 mature months); F_mature = median of per-month
  fill fractions at each lag_day (median-of-curves, not pooled rows — robust to one
  weird month); dual-lag populations: rows with NULL source_insert_time form their
  own reported bucket; batch-level completion weighted by batch row counts; band =
  ±2 scaled MAD widened by forecast dispersion (month-end aging biases toward
  under-alarming, but forecast error and distribution shift CAN still false-alarm —
  the band, not the aging, carries that risk). Rounding rules stated. Changing any
  formula bumps schema_version.
- `optional: true` tables are skipped with a warning if absent, never a failure.
- Proxy timestamps flagged `proxy: true` are labelled honestly in figures.
- `expect_batchy: true` annotates step-function feeds (affects report wording,
  not metric math).
- Connection: SQLAlchemy URL; support both trusted connections
  (`mssql+pyodbc://@SERVER/DB?driver=ODBC+Driver+18+for+SQL+Server&Trusted_Connection=yes&TrustServerCertificate=yes`)
  and SQL logins via full-URL override. URLs are env-var expandable.

## Development method: TEST FIRST, always

Rhythm per feature: **generator scenario → failing golden test → implement → green on
DuckDB → PR proves T-SQL parity via the container job.** Never write metric code before
its expected values exist in a test.

### Synthetic generator (tests/synth)
Parametric simulation of the arrival process — not hand-made fixtures. Per event month:
draw N rows, assign each a load lag from a chosen model:
- lognormal trickle (records arriving gradually over days/weeks, e.g. orders,
  claims, registrations)
- step batches with a batch ID (periodic bulk loads, e.g. monthly settlement cycles)
- dual timestamps with a known source→load offset
Expected percentiles are *derivable from the generating parameters*; tests assert the
pipeline recovers them within an explicit, commented tolerance.
Named pathology scenarios, each with a named expected detection: volume spike, missing
month, duplicate keys, late straggler batch, raw-vs-corrected date mismatches,
sustained volume collapse (loads keep arriving on cadence but recent months carry
~10× fewer rows — must yield volume=RED with freshness=GREEN). Scenarios come in
healthy/unhealthy twin pairs: every check must fire on the unhealthy twin AND stay
silent on the healthy twin.

### Test tiers
1. **unit** — config validation fails loudly on bad mappings; compiled-SQL snapshot
   tests per dialect (a T-SQL change appears as a readable diff in review).
2. **golden (DuckDB)** — every metric vs derived expected values. Assertion style:
   accumulate ALL failing cases into one message, never die on the first.
3. **properties** — completion curves monotonic non-decreasing and reaching 100% at
   their max observed lag BY CONSTRUCTION (therefore curve shape is never used for
   censoring — maturity classification is exposure-based and tested as such);
   row-order invariance; the population
   reconciliation equation holds (total_rows == curve_eligible + each reported
   exclusion bucket, tying completion to volume counts); suppression on ⇒ no
   rendered value < 5 anywhere.
4. **equivalence** — identical synthetic dataset loaded into DuckDB and a real
   SQL Server (official mssql container — pin the CI runner to Linux x86-64, the
   only platform Microsoft supports for it); pipeline numbers must match. The
   harness is built in Step 0 and EVERY metric adds its equivalence cases in the
   same step it lands — parity per feature, not deferred. DuckDB is the fast
   vehicle; the container job is the truth-teller for the T-SQL dialect production
   actually runs.
5. **viz/report smoke** — every figure builds for every scenario (incl. batchy and
   missing-month); HTML report loads zero external resources and carries the
   no-network CSP.
6. **CLI contract** — red status ⇒ non-zero exit code, tested like any feature.

### CI (GitHub Actions, also Gitea-compatible syntax)
Two parallel jobs on every PR and on main: (a) lint + unit + DuckDB golden/property
suite (seconds); (b) SQL Server container equivalence (~2–3 min). Merge requires both
green. Releases are git tags; downstream deployments pin exact versions.

## Conventions

- Python ≥ 3.12 (`requires-python = ">=3.12"`); modern typing, no compat shims.
- SQLAlchemy Core for all SQL construction (compiles to mssql and duckdb dialects).
- Plotly for all figures; figures built once in `viz/`, consumed by report and app.
- Streamlit app: snapshots-first (instant load, snapshot timestamp always visible);
  per-table "refresh now" behind a config flag (disabled ⇒ app holds no DB credentials).
- Rolling probe window default (`--window 24m`); fixed `--year YYYY` for ad hoc.
- Status traffic lights are statistical and self-calibrating against the table's own
  historical months within the same probe run: amber ≈ 2 robust deviations,
  red ≈ 3 or any hard failure (missing table, zero-row month, parity mismatch).
  Baselines are computed from history EXCLUDING the recent evaluation window, so a
  sustained degradation cannot normalize itself. Freshness and volume are independent
  verdicts: a table can be "updating: OK" and "volume: RED" simultaneously.
- Probes execute **sequentially, never in parallel** — target tables are large and
  possibly unindexed on the probed columns, so assume each probe is a full scan and
  be a polite tenant. Each probe's wall-clock duration is recorded in the snapshot.
  Optional per-connection `read_uncommitted: true` (off by default) for busy servers.
- Static image export (PNG/SVG) uses Plotly's kaleido, which with current Plotly
  requires an installed Chrome/Chromium — declare it as an explicit optional
  dependency, verify at `report`/`publish` startup with a clear error, and document
  the install for locked-down Windows machines.
- Deterministic outputs for the committed demo: clock, run_id, and git metadata are
  INJECTABLE; SVG output is canonicalized (stable element ids). The demo build
  freezes all of them so the fixed-seed CI diff check is byte-stable.
- License: MIT.
- Keep CI workflow steps to "invoke Python with arguments"; logic lives in
  Python/SQL files in the repo, never in shell one-liners.
