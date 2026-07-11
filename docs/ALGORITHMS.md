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
- **Zero-MAD fallback**: when `MAD == 0` (perfectly regular values), callers
  use an explicit fallback tolerance instead of a zero band:
  `robust_sigma_or(x, fallback) = robust_sigma(x) if MAD(x) > 0 else fallback`.

## 2. Completion curves and day-grain percentiles

For each event month m, from the canonical aggregation's (event_month, lag_day)
cells over CURVE-ELIGIBLE rows only:

- `final_m` = the month's total curve-eligible count as of the frozen `as_of`.
- `F_m(d)` = (rows of month m with lag_day <= d) / final_m — cumulative,
  self-normalized. By construction F_m is monotone non-decreasing and reaches
  1.0 at the month's largest observed lag — THEREFORE curve shape carries no
  censoring signal and is never used for maturity classification.
- `days_to_p(m, p)` = smallest integer d with `F_m(d) >= p`.

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
  with `month_end <= as_of - training_cutoff_days`).
- The user-facing `recommended_wait` = the same formula over MATURE months.
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
- `other_exclusions`: reserved, 0 in v1.

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
  fraction: `sigma_band(t) = F_mature(t) * robust_sigma(mature final volumes)`
  (zero-MAD fallback per section 1), giving

```
band(t) = expected(t) ± expected_fill_band_mads * sigma_band(t)
```

## 9. Uniqueness (probabilistic distinct-key guard)

`duplicate_rows = total_rows - COUNT(DISTINCT key_hash)` where key_hash is
SHA-256 over a type-tagged, length-prefixed binary encoding with a distinct
NULL sentinel (delimiter concatenation is ambiguous). The check is documented
as probabilistic: SHA-256 collision odds are negligible; encoding ambiguity was
the real risk. Never 32-bit CHECKSUM. Duplicates present iff duplicate_rows > 0.
