# ALGORITHMS.md — v1 formula reference

Every formula the pipeline uses, pinned here BEFORE implementation and enforced
by hand-calculated boundary tests (`tests/unit/test_algorithms.py`). Changing a
formula is a schema-level act: update this file, the tests, and bump the
relevant schema version.

All durations are days. All percentiles are day-grain: the production lag is
`lag_day = DATEDIFF(day, event_time, load_time)` — calendar-day boundaries
crossed, an integer.

## 1. Robust statistics

- `median(x)`: standard median (mean of middle two for even n).
- `MAD(x) = median(|x_i - median(x)|)` — median absolute deviation.
- `robust_sigma(x) = 1.4826 * MAD(x)` — the normal-consistency scaling, so a
  Gaussian sample yields sigma ~= standard deviation.
- **Zero-MAD fallback** (frozen in CLAUDE.md): a RELATIVE FLOOR, not an
  exact-zero special case —
  `robust_sigma_floor(x, rel_tol) = max(robust_sigma(x), rel_tol * median(x))`.
  Perfectly regular AND nearly regular values both get at least rel_tol of
  their median as spread. v1 default `rel_tol = 0.05` where no explicit
  tolerance is configured (freshness uses its configured absolute tolerance).

## 2. Completion curves and day-grain percentiles

For each event month m, from the canonical aggregation's (event_month, lag_day)
cells over CURVE-ELIGIBLE rows only:

- `final_m` = the month's total curve-eligible count as of the frozen `as_of`.
- `F_m(d)` = (rows of month m with lag_day <= d) / final_m — cumulative,
  self-normalized. By construction F_m is monotone non-decreasing and reaches
  1.0 at the month's largest observed lag — THEREFORE curve shape carries no
  censoring signal and is never used for maturity classification.
- `days_to_p(m, p)` = smallest integer d with `F_m(d) >= p`.

**Frozen lag sentinels**: rows beyond the cap land in the overflow bucket at
`lag_cap_days + 1`; negative-excluded rows (section 6) keep their event month
at lag sentinel `-1` — they belong to parity's watermarked population but
NEVER enter curves or volume history.

**Censoring-aware cap rule**: lag_day is capped at `lag_cap_days`; rows beyond
it land in the overflow bucket (recorded at lag_cap_days + 1). If
`F_m(lag_cap_days) < p` — equivalently overflow mass > (1 - p) — the percentile
is reported as `> lag_cap_days` (over_cap), NOT a number, and any wait derived
from it is REFUSED (reason PERCENTILE_OVER_CAP): the missing mass could lie
anywhere above the cap.

## 3. recommended_wait

Over a set of months M with defined (not over-cap) `days_to_p95`:

```
recommended_wait(M) = ceil( mean(d95) + 2 * pstdev(d95) )
```

with `pstdev` the population standard deviation (ddof = 0; a single month gives
pstdev 0, so the wait equals that month's p95). If ANY month in M is over-cap,
the wait is refused (PERCENTILE_OVER_CAP).

- `learned_wait` = recommended_wait over the TRAINING cohort (fixed: months
  with `month_end <= as_of - training_cutoff_days`). An EMPTY cohort is
  insufficient history, never a silent green.
- The user-facing `recommended_wait` = the same formula over MATURE months.
- Headline percentile summary: for each of p50/p90/p95/p99, the (mean,
  population std) across mature months — undefined when any mature month's
  percentile is over-cap.
- "complete back to" date = `as_of - recommended_wait` (Step 5 wiring).

## 4. Maturity (single-pass, exposure-based)

```
horizon        = max(training_cutoff_days, learned_wait)      # monotone
mature(m)      = month_end(m) <= as_of - horizon
```

`month_end(m)` is the first instant of the next month. Classification uses
EXPOSURE only — never curve shape (see section 2). After classification,
`min_mature_months` is re-checked against the mature set; fewer ⇒
INSUFFICIENT_MATURE_MONTHS. No iteration.

## 5. Lag-support backtest (heuristic)

Split the training cohort, ordered by month_end, into an OLDER half and a
YOUNGER half (only when the cohort has >= 6 months; otherwise the backtest is
skipped — too few months to compare strata). Compute learned_wait per half.

```
disagreement  iff  |wait_older - wait_younger| > max(7, 0.25 * max(waits))
              or   exactly one stratum's wait is refused (censored past the
                   lag cap) while the other's is defined
```

The backtest runs even when the whole-cohort wait was already refused — the
strata comparison is independent evidence.

Disagreement ⇒ INSUFFICIENT_HISTORY (reason BACKTEST_DISAGREEMENT), never a
confidently wrong wait. The 7-day floor and 25% ratio are v1 algorithm
constants (not config).

**Known blind spot (documented, tested)**: the backtest detects strata that
DISAGREE; it cannot prove completeness. If all strata share the same very late
tail (mass beyond the oldest cohort's age), both halves are equally censored,
agree, and the backtest stays silent. Tails beyond the oldest observed cohorts
are unknowable without an external finality signal.

## 6. Negative lags (clock skew vs corruption)

For rows with both timestamps, `lag_day < 0`:

- `-clock_skew_tolerance_days <= lag_day < 0` ⇒ CLIPPED to lag 0, stays
  curve-eligible, counted in `n_negative_clipped`.
- `lag_day < -clock_skew_tolerance_days` ⇒ EXCLUDED from curves, counted in
  `n_negative_lag_excluded`. Clipping everything would let timestamp
  corruption IMPROVE the percentiles.
- `excess_fraction = n_negative_lag_excluded / n_rows_with_both_timestamps`;
  RED iff `excess_fraction > negative_lag_red_fraction`.

## 7. Population reconciliation (mutually exclusive buckets)

Every row admitted by the as-of predicate `(load_time <= :as_of OR load_time
IS NULL)` falls in exactly one bucket:

```
total_rows = curve_eligible + null_event_time + null_load_time_only
           + negative_lag_excluded + join_unmatched + other_exclusions
```

- `null_event_time`: event time NULL (for via-joins: matched but the borrowed
  column is NULL).
- `null_load_time_only`: event time present, load_time NULL (query-time count,
  not watermarked by as_of).
- `join_unmatched` (via-joins only): no lookup row matched.
- `other_exclusions`: reserved, 0 in v1 — carried explicitly in the result
  schema so the equation is complete.

Via-joins additionally report `n_base_rows` (pre-join base count: unmatched
rows plus the FIRST match per base row) and `n_ambiguous_base_rows` (base rows
matching more than one lookup row). The pre/post reconciliation
`n_base_rows == total_rows` must hold; ambiguity aborts the probe.

The equation must hold exactly; violation is RECONCILIATION_MISMATCH (RED) —
it ties completion to volume counts from the ONE canonical aggregation.

## 8. F_mature and the expected-fill band (consumed by Step 4)

- `F_mature(d)` = pointwise MEDIAN over mature months' curves F_m(d), each
  forward-filled onto the common integer lag grid (median-of-curves).
- Expected count for an immature month at month-end age t:
  `expected(t) = volume_forecast * F_mature(t)`, with volume_forecast the
  median of mature months' final volumes — NEVER the immature month's own
  count (tautology guard).
- Band half-width includes the forecast dispersion, scaled by the fill
  fraction: `sigma_band(t) = F_mature(t) * robust_sigma_floor(MATURE final
  volumes)` (section 1). Note: whenever immature months exist, the evaluation
  window (section 10) is itself immature — maturity is monotone in time — so
  the mature set and the section-10 baseline coincide here. Giving

```
band(t) = expected(t) ± expected_fill_band_mads * sigma_band(t)
```

## 9. Uniqueness (probabilistic distinct-key guard)

`duplicate_rows = total_rows - COUNT(DISTINCT key_hash)` where key_hash is
SHA-256 over a type-tagged, length-prefixed binary encoding with a distinct
NULL sentinel. Per column: `0x00` for NULL, else `0x01` + length-prefixed TYPE
TAG (the column's DECLARED type, fetched once from INFORMATION_SCHEMA and
embedded as a literal — a per-row type function would reject varchar(max) and
cost a call per row) + length-prefixed value bytes. Delimiter concatenation is
ambiguous; the check is documented as probabilistic: SHA-256 collision odds
are negligible; encoding ambiguity was the real risk. Never 32-bit CHECKSUM.
Duplicates present iff duplicate_rows > 0.

## 10. Volume baselines and verdicts

- Volumes are the per-month curve-eligible counts from the canonical
  aggregation's (event_month, lag_day) cells — no separate query.
- **Baseline** = MATURE months EXCLUDING the evaluation window (the last
  `evaluation_window_months` OBSERVED months), so a sustained degradation can
  never normalize itself. Requires `min_mature_months` baseline months, else
  insufficient history. `baseline_sigma = robust_sigma_floor(baseline
  volumes)` (section 1); the volume forecast for section 8 is the baseline
  median (non-seasonal v1; a seasonal month-of-year median requires >= 24
  mature months and is deferred).
- **Outliers** (MATURE months only): |volume - baseline_median| >
  `volume_red_mads * sigma` => RED, > `volume_amber_mads * sigma` => AMBER
  (reason VOLUME_OUTLIER).
- **Sustained collapse** (MATURE months only): a run of >= 2 consecutive
  below-red months ending at the most recent mature month => one RED
  VOLUME_COLLAPSE verdict (those months are not double-reported as outliers).
- **Gaps**: interior months with zero rows => RED VOLUME_GAP, rendered
  explicitly.
- **Still-filling** (immature CLOSED months): age from MONTH_END
  (conservative — can only under-alarm); the OPEN month (not yet ended at
  as_of) is excluded from every check and reported as "open". Observed count
  below the section-8 band => AMBER ARRIVAL_DEFICIT, phrased "arrival deficit
  — cause unresolved"; the collapse verdict is mature-only. The inverted
  nowcast `observed / F_mature(age)` is reported but NEVER fed back into the
  expectation (tautology guard: a month at 50% of expectation must flag even
  though its own nowcast is self-consistent).

## 11. Freshness (staleness core; full metric in Step 5)

- Arrival epochs = distinct batch IDs when `load_batch_col` is configured
  (canonical batch timestamp = MIN(load_time) within the batch, i.e. the
  minimum over its per-month cells), else distinct load-time buckets. NEVER
  per-row gaps: row timestamps measure row frequency, not feed cadence.
- Requires `freshness_min_epochs` epochs, else INSUFFICIENT_EPOCHS.
- Learned cadence = median of inter-epoch gaps; spread =
  `max(robust_sigma(gaps), freshness_zero_mad_tolerance_days)` (perfectly
  regular feeds use the configured fixed tolerance).
- Staleness: `days_since_last_epoch > cadence + 3 * spread` => RED STALE_FEED,
  `> cadence + 2 * spread` => AMBER (v1 algorithm constants mirroring the
  volume defaults).
- Freshness and volume are independent verdicts: a table can be
  "updating: GREEN" and "volume: RED" simultaneously — the sustained-collapse
  acceptance case requires exactly that, which is only satisfiable when the
  collapse is OLDER than the maturity horizon while loads continue to arrive.

## 12. Batch metrics (when load_batch_col is configured)

From the (event_month, batch_id) cells; a batch's canonical timestamp is
MIN(load_time) within the batch — the minimum over its per-month cells, since
a batch can span cohorts.

- Batch cells are keyed by LOAD presence, not curve eligibility: a batch
  whose rows have corrupt/NULL event times is still a real arrival epoch for
  freshness. Batch COMPLETION weights use the cells' curve-eligible counts.
- `rows_per_run(b)` = the batch's curve-eligible rows summed over its cells.
- `runs_per_month(m)` = distinct batches touching event month m.
- Curve-eligible rows with a NULL batch id are counted and reported (AMBER
  NULL_BATCH_IDS) and REMAIN in the completion denominator — attributing only
  labelled rows would overstate completion; unreachable percentiles stay None.
- **Batch-level completion** for month m: order its batches by canonical
  timestamp T_b; `F_m(T) = sum(rows of batches with T_b <= T) / final_m` —
  cumulative and WEIGHTED BY BATCH ROW COUNTS (a bulk run moves the curve by
  its size, a straggler by its size, never one-run-one-vote).
  `batch_days_to_p(m, p)` = the smallest integer day d (measured from
  MONTH END, day-grain; negative for in-month batches) such that a batch at
  month_end + d brings F_m to >= p.

## 13. Parity (two NAMED probes)

Exact parity compares the WATERMARKED per-month population: rows with
non-NULL load_time <= as_of and a defined event month — the curve-eligible
rows PLUS the negative-lag-excluded rows kept at the frozen -1 sentinel (an
allowed below-threshold excess must not create false diffs). NULL-load rows
are query-time counts, reported separately and informationally, never inside
the exact diff. The pairing itself is declared: the left probe's parity_with
must name the right probe.

- The COMMON MATURE population is defined by TIME under the STRICTER of the
  two sides' horizons (month_end <= as_of - max(horizon_left, horizon_right)):
  different learned horizons on identical data must never manufacture
  one-sided months. Candidate months = the union of both sides' OBSERVED
  months within that common range: present on both sides -> compare counts;
  present on exactly one side -> RED PARITY_ONE_SIDED_MONTH (an explicit
  diff, never silently dropped). |left - right| > parity_tolerance -> RED
  PARITY_MISMATCH.
- Zero-tolerance parity is sound ONLY under VERIFIED prerequisites, checked
  per run; any failure or unverifiability yields INDETERMINATE with the
  failing prerequisite as the reason code, never a false mismatch:
    * uniqueness configured (key_cols) AND zero duplicates on BOTH sides
      (PARITY_PREREQ_UNIQUENESS),
    * read_uncommitted disabled on both configs
      (PARITY_PREREQ_READ_UNCOMMITTED),
    * negative-lag excess below threshold on both sides
      (PARITY_PREREQ_NEGATIVE_LAG — the backdating proxy).

## 14. Dual lag and the raw-vs-corrected side-stat

- Dual lag runs as ONE additional pass (per the budget: main + dual = 2 scans
  <= 3x): the same staging/aggregation shape over (event_month, src_lag_day)
  — src_lag_day = DATEDIFF(day, event_time, source_insert_time), with the
  SAME clip/cap/overflow policy as section 6 — plus (delta_day) with
  delta_day = DATEDIFF(day, source_insert_time, load_time), the per-row
  upstream-vs-local split, and the () global buckets. Rows with NULL
  source_insert_time (event present) form their own reported bucket
  `n_null_source_only`; the dual reconciliation
  `total = source_eligible + null_event + null_source_only + negative_excluded`
  must hold.
- Source-side curves/percentiles reuse the section 2 formulas verbatim.
- `compare_event_time` side-stat: among curve-eligible rows, count per event
  month where DATEDIFF(day, event_time, compare_event_time) != 0 or the
  compare column is NULL (`n_compare_mismatch`, aggregated in the main pass —
  no extra scan).

## 15. Scan-budget accounting (two fail-closed ledgers)

The probe's I/O is measured per table via STATISTICS IO and split into two
ledgers, BOTH enforced fail-closed on SQL Server (unmeasurable => abort with
SCAN_BUDGET_UNVERIFIABLE; exceeded => SCAN_BUDGET_EXCEEDED):

1. **Target ledger** — reads on the probed table(s) (base + via lookup):
   `target_reads <= 3 x one full scan` (3 x used_page_count). This budget is
   per PROBE and CUMULATIVE: a dual-lag probe verifies main-pass + dual-pass
   target reads together against the same 3x bound (by construction each pass
   scans the target once, so main + dual = 2 <= 3).
2. **Scratch ledger** — everything else the probe reads: its own staging
   materialization, worktables and workfiles. This is where the aggregation
   branches and the COUNT(DISTINCT key_hash) uniqueness guard run, so the
   guard's reads are COUNTED and enforced here:
   `scratch_reads <= (branches + 1) x staging_pages + 6 x staged_rows`,
   where branches is the by-construction number of staging scans (one per
   grouping set plus the () global/distinct branch), +1 is margin, and the
   row-linear term covers sort/spool WORKTABLE activity, which scales with
   rows, not pages (the Step 7 audit measured up to ~2.6 reads/row on
   sort-heavy plans such as hour-bucket epochs; a pure pages bound
   false-aborted legitimate probes). The bound is a tripwire against
   pathological plans, not decoration.

3. **Spool ledger** — the staging statement's own worktable spool (the
   via-join uniqueness window functions) is row-proportional and ENFORCED
   fail-closed: `spool_reads <= staging_pages + 10 x staged_rows` (measured
   ~6 reads/row). It is tempdb-local and the price of piggybacking join
   validation instead of a second lookup scan.

Formula history: CANONICAL_SCHEMA_VERSION 2 introduced the row-linear scratch
term and the enforced spool ledger (v1 used a pages-only scratch bound, which
the Step 7 audit showed false-aborts sort-heavy plans).

Rationale: the hard rule's "3x one full scan" bounds pressure on the
PRODUCTION table; the scratch work is tempdb-local, bounded by construction,
and now measured and enforced rather than assumed. A single combined
"3x target pages" ledger cannot honestly include scratch: for tables narrower
than their own staging projection it is unsatisfiable by ANY architecture
that materializes derived columns.
