"""Parametric synthetic arrival generator (PLAN Step 1).

Simulates the arrival process itself — not hand-made fixtures. Per event month it
draws N rows with event times uniform over the month and assigns each row a load
lag from a chosen model, so expected completion percentiles are DERIVABLE from
the generating parameters (see expected_days_to_percentiles).

Lag models
  LognormalLag(mu, sigma)  records trickle in gradually over days/weeks
  StepBatches(schedule)    periodic bulk loads: each (days_after_month_end,
                           fraction) is one physical batch per month with its own
                           batch_id; all its rows share the batch timestamp

Dual timestamps: with TableSpec.dual_offset_days set, the lag model produces
source_insert_time and load_time follows it by exactly that offset.

Determinism: every month uses its own RNG streams seeded by (seed, month_index),
so healthy/unhealthy twins share byte-identical rows in unaffected months.
Within a month, event times and the arrival process draw from SEPARATE streams:
changing a month's row count only truncates/extends the sequence (NumPy fills
sequentially), so a volume twin's retained rows keep identical event times AND
lags — the pathology is purely "fewer/more rows", never resampled arrivals.

All timestamps are floored to whole seconds so values survive SQL Server
DATETIME's 3.33 ms rounding unchanged in equivalence tests.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd


def _require_finite(value: float, what: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value}")


@dataclass(frozen=True)
class LognormalLag:
    """Lag-days ~ Lognormal(mu, sigma): gradual trickle arrivals."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        _require_finite(self.mu, "LognormalLag mu")
        _require_finite(self.sigma, "LognormalLag sigma")
        if self.sigma <= 0:
            raise ValueError("LognormalLag sigma must be > 0")

    def quantile(self, q: float) -> float:
        return float(np.exp(self.mu + self.sigma * NormalDist().inv_cdf(q)))


@dataclass(frozen=True)
class StepBatches:
    """Periodic bulk loads: each schedule entry (days_after_month_end, fraction)
    is one batch per month loading that fraction of the month's rows."""

    schedule: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.schedule:
            raise ValueError("StepBatches schedule must not be empty")
        for day, fraction in self.schedule:
            _require_finite(day, "StepBatches day")
            _require_finite(fraction, "StepBatches fraction")
        if any(day < 0 for day, _ in self.schedule):
            raise ValueError("StepBatches days must be >= 0 (measured from month end)")
        for _, fraction in self.schedule:
            # zero would silently produce no physical batch for that entry
            if not 0 < fraction <= 1:
                raise ValueError(f"each StepBatches fraction must be in (0, 1], got {fraction}")
        total = sum(fraction for _, fraction in self.schedule)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"StepBatches fractions must sum to 1, got {total}")


# per-month row-id block size: ids are month_index * stride + offset
_ROW_ID_STRIDE = 10_000_000


@dataclass(frozen=True)
class TableSpec:
    """Full generating parameters for one synthetic table."""

    name: str
    start_month: str  # "YYYY-MM"
    n_months: int
    rows_per_month: int
    lag_model: LognormalLag | StepBatches
    dual_offset_days: float | None = None
    volume_overrides: Mapping[int, float] = field(default_factory=dict)  # index -> multiplier
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dual_offset_days is not None:
            _require_finite(self.dual_offset_days, "dual_offset_days")
        for index, factor in self.volume_overrides.items():
            _require_finite(factor, f"volume override for month {index}")
        if self.n_months < 1:
            raise ValueError("n_months must be >= 1")
        if self.rows_per_month < 0:
            raise ValueError("rows_per_month must be >= 0")
        for index, factor in self.volume_overrides.items():
            if not 0 <= index < self.n_months:
                raise ValueError(
                    f"volume override for month {index} outside 0..{self.n_months - 1}"
                )
            if factor < 0:
                raise ValueError("volume multiplier must be >= 0")


def generate(spec: TableSpec) -> pd.DataFrame:
    """Generate the table as a DataFrame: row_id, event_time, load_time, plus
    source_insert_time (dual) and batch_id (step batches) when configured."""
    periods = pd.period_range(spec.start_month, periods=spec.n_months, freq="M")
    frames = [_month_frame(spec, index, period) for index, period in enumerate(periods)]
    non_empty = [frame for frame in frames if len(frame)] or frames[:1]
    return pd.concat(non_empty, ignore_index=True)


def _month_frame(spec: TableSpec, index: int, period: pd.Period) -> pd.DataFrame:
    multiplier = spec.volume_overrides.get(index, 1.0)
    n = round(spec.rows_per_month * multiplier)
    # BEFORE any array allocation: a month bigger than the row-id stride
    # would collide with the next month's ids (ids 0..n-1 fit exactly when
    # n == stride, so only n > stride overflows)
    if n > _ROW_ID_STRIDE:
        raise ValueError(
            f"month {period} draws {n} rows > the {_ROW_ID_STRIDE} row-id "
            "stride; row ids would collide across months"
        )
    # separate per-purpose streams: a different n only truncates/extends each
    # sequence, so volume twins retain byte-identical rows (prefix property)
    rng_event = np.random.default_rng([spec.seed, index, 0])
    rng_arrival = np.random.default_rng([spec.seed, index, 1])
    start = period.start_time
    month_end = (period + 1).start_time
    length_days = (month_end - start) / pd.Timedelta(days=1)

    event = (start + pd.to_timedelta(rng_event.uniform(0, length_days, n), unit="D")).floor("s")
    batch_id: list[str] | None = None
    if isinstance(spec.lag_model, LognormalLag):
        lag_days = rng_arrival.lognormal(spec.lag_model.mu, spec.lag_model.sigma, n)
        arrival = (event + pd.to_timedelta(lag_days, unit="D")).floor("s")
    else:
        days = np.array([day for day, _ in spec.lag_model.schedule])
        fractions = np.array([fraction for _, fraction in spec.lag_model.schedule])
        choice = rng_arrival.choice(len(days), size=n, p=fractions / fractions.sum())
        batch_times = (month_end + pd.to_timedelta(days, unit="D")).floor("s")
        arrival = pd.DatetimeIndex(batch_times[choice])
        batch_id = [f"{period}-run{c}" for c in choice]

    frame = pd.DataFrame({"row_id": index * _ROW_ID_STRIDE + np.arange(n), "event_time": event})
    if spec.dual_offset_days is not None:
        frame["source_insert_time"] = arrival
        frame["load_time"] = arrival + pd.Timedelta(days=spec.dual_offset_days)
    else:
        frame["load_time"] = arrival
    if batch_id is not None:
        frame["batch_id"] = batch_id
    return frame


# --------------------------------------------------------------- expected values


def expected_days_to_percentiles(
    spec: TableSpec,
    percentiles: tuple[int, ...] = (50, 90, 95, 99),
    timestamp: str = "load",
    grain: str = "day",
) -> dict[str, dict[int, float]]:
    """Days-to-pXX per event month, derived from the generating parameters alone.

    grain="day" (the default) matches the production metric's DATEDIFF(day)
    semantics — lag_day = calendar-day boundaries crossed between event_time and
    load_time — and returns the smallest INTEGER day whose day-grain CDF reaches
    the percentile. This is the oracle golden tests must consume. With events
    uniform in time, the within-day position u~U(0,1) is independent of the
    continuous lag, so P(lag_day <= d) = integral of F_continuous over [d, d+1]
    (trickle); for batches the day lag is exactly month_days + floor(day_b) -
    event_day_index, giving the discrete closed form in _day_quantile.

    grain="continuous" returns the continuous elapsed-days quantile: lognormal
    closed form, or the StepBatches mixture CDF
        F(t) = sum_b fraction_b * clamp((t - day_b) / month_len, 0, 1)
    inverted numerically — still parameter-derived, never fitted to data.

    timestamp="load" is the local-arrival curve (includes the dual offset when
    configured); "source" is the source-side curve and requires dual timestamps.
    """
    if timestamp not in ("load", "source"):
        raise ValueError("timestamp must be 'load' or 'source'")
    if grain not in ("day", "continuous"):
        raise ValueError("grain must be 'day' or 'continuous'")
    if timestamp == "source" and spec.dual_offset_days is None:
        raise ValueError("source percentiles require dual_offset_days")
    offset = spec.dual_offset_days or 0.0 if timestamp == "load" else 0.0

    out: dict[str, dict[int, float]] = {}
    for period in pd.period_range(spec.start_month, periods=spec.n_months, freq="M"):
        length_days = ((period + 1).start_time - period.start_time) / pd.Timedelta(days=1)
        values: dict[int, float] = {}
        for pct in percentiles:
            if not 0 < pct < 100:
                raise ValueError(f"percentile {pct} outside (0, 100)")
            q = pct / 100
            if grain == "day":
                values[pct] = _day_quantile(spec.lag_model, length_days, offset, q)
            elif isinstance(spec.lag_model, LognormalLag):
                values[pct] = spec.lag_model.quantile(q) + offset
            else:
                values[pct] = _step_quantile(spec.lag_model.schedule, length_days, q) + offset
        out[str(period)] = values
    return out


def _step_cdf(schedule: tuple[tuple[float, float], ...], length_days: float, t: float) -> float:
    return sum(
        fraction * min(max((t - day) / length_days, 0.0), 1.0) for day, fraction in schedule
    )


def _step_quantile(
    schedule: tuple[tuple[float, float], ...], length_days: float, q: float
) -> float:
    lo, hi = 0.0, max(day for day, _ in schedule) + length_days
    for _ in range(200):
        mid = (lo + hi) / 2
        if _step_cdf(schedule, length_days, mid) >= q:
            hi = mid
        else:
            lo = mid
    return hi


def _day_quantile(
    model: LognormalLag | StepBatches, length_days: float, offset: float, q: float
) -> int:
    """Smallest integer day d with P(lag_day <= d) >= q under DATEDIFF semantics."""
    if isinstance(model, StepBatches):
        # batch date = date(month_end + day_b + offset); event day index k uniform
        # over {0..D-1}; lag_day = D + floor(day_b + offset) - k, hence
        # P(lag_day <= d) = clamp((d - floor(day_b + offset)) / D, 0, 1) per batch
        floors = [(math.floor(day + offset), fraction) for day, fraction in model.schedule]

        def day_cdf(d: int) -> float:
            return sum(
                fraction * min(max((d - fd) / length_days, 0.0), 1.0) for fd, fraction in floors
            )

        upper = max(fd for fd, _ in floors) + math.ceil(length_days)
    else:

        def day_cdf(d: int) -> float:
            return _lognormal_day_cdf(model, offset, d)

        # F_day(d) >= F_continuous(d - offset), so the shifted continuous quantile
        # is an upper bound for the day-grain one
        upper = math.ceil(model.quantile(q) + offset) + 1

    for d in range(upper + 1):
        if day_cdf(d) >= q:
            return d
    return upper  # unreachable by construction; keeps the type checker honest


def _lognormal_day_cdf(model: LognormalLag, offset: float, d: int) -> float:
    """P(lag_day <= d) = integral of the continuous lognormal CDF (shifted by the
    dual offset) over [d, d+1], via trapezoid on 128 panels (error << 1e-4)."""
    lo, hi = d - offset, d + 1 - offset
    lo = max(lo, 0.0)  # the CDF is 0 at t <= 0, contributing nothing
    if hi <= 0:
        return 0.0
    normal = NormalDist()
    xs = [lo + (hi - lo) * i / 128 for i in range(129)]
    ys = [0.0 if x <= 0 else normal.cdf((math.log(x) - model.mu) / model.sigma) for x in xs]
    return (sum(ys) - 0.5 * (ys[0] + ys[-1])) * (hi - lo) / 128


# ------------------------------------------------------- volume-plan injectors
# These return a NEW spec (frozen dataclasses), leaving the healthy twin intact.


def volume_spike(spec: TableSpec, month_index: int, factor: float = 6.0) -> TableSpec:
    """One month at `factor` times its normal volume."""
    if factor <= 1:
        raise ValueError("a volume spike factor must be > 1")
    return _with_override(spec, {month_index: factor})


def volume_drop(spec: TableSpec, month_index: int, factor: float = 0.1) -> TableSpec:
    """One month at `factor` of its normal volume (a one-month drop, as opposed
    to the multi-month sustained_collapse)."""
    if not 0 < factor < 1:
        raise ValueError("a volume drop factor must be in (0, 1); use missing_month for 0")
    return _with_override(spec, {month_index: factor})


def missing_month(spec: TableSpec, month_index: int) -> TableSpec:
    """One month entirely absent from the data."""
    return _with_override(spec, {month_index: 0.0})


def sustained_collapse(spec: TableSpec, last_k: int = 3, factor: float = 0.1) -> TableSpec:
    """Loads keep arriving on their normal cadence, but the last `last_k` months
    carry ~`factor` of the historical volume."""
    if not 0 < factor < 1:
        raise ValueError("a collapse factor must be in (0, 1); use missing_month for 0")
    return _with_override(spec, {spec.n_months - i: factor for i in range(1, last_k + 1)})


def _with_override(spec: TableSpec, overrides: dict[int, float]) -> TableSpec:
    return dataclasses.replace(spec, volume_overrides={**spec.volume_overrides, **overrides})


# -------------------------------------------------------- row-level injectors
# Each returns a modified COPY, deterministically sampled from (seed, salt).


def _sample(df: pd.DataFrame, fraction: float, seed: int, salt: int) -> np.ndarray:
    rng = np.random.default_rng([seed, salt])
    return rng.choice(len(df), size=round(fraction * len(df)), replace=False)


def inject_duplicate_keys(df: pd.DataFrame, fraction: float, seed: int = 0) -> pd.DataFrame:
    """Re-append a sample of rows with the same row_id, loaded an hour later."""
    duplicates = df.iloc[_sample(df, fraction, seed, 101)].copy()
    duplicates["load_time"] = duplicates["load_time"] + pd.Timedelta(hours=1)
    return pd.concat([df, duplicates], ignore_index=True)


def inject_straggler_batch(
    df: pd.DataFrame, month: str, late_day: float, fraction: float, seed: int = 0
) -> pd.DataFrame:
    """Move a fraction of one month's rows into a very late extra batch."""
    out = df.copy()
    period = pd.Period(month, freq="M")
    positions = np.flatnonzero(out["event_time"].dt.to_period("M") == period)
    rng = np.random.default_rng([seed, 202])
    chosen = out.index[rng.choice(positions, size=round(fraction * len(positions)), replace=False)]
    batch_time = ((period + 1).start_time + pd.Timedelta(days=late_day)).floor("s")
    out.loc[chosen, "load_time"] = batch_time
    if "batch_id" in out.columns:
        out.loc[chosen, "batch_id"] = f"{period}-straggler"
    return out


def inject_raw_vs_corrected(
    df: pd.DataFrame, fraction: float, shift_days: float, seed: int = 0
) -> pd.DataFrame:
    """Add event_time_raw, differing from event_time by shift_days on a sample."""
    out = df.copy()
    out["event_time_raw"] = out["event_time"]
    chosen = out.index[_sample(df, fraction, seed, 303)]
    out.loc[chosen, "event_time_raw"] = out.loc[chosen, "event_time"] + pd.Timedelta(
        days=shift_days
    )
    return out


def inject_negative_lags(
    df: pd.DataFrame, fraction: float, skew_days: float, seed: int = 0
) -> pd.DataFrame:
    """Corrupt a sample so load_time precedes event_time by skew_days."""
    out = df.copy()
    chosen = out.index[_sample(df, fraction, seed, 404)]
    out.loc[chosen, "load_time"] = out.loc[chosen, "event_time"] - pd.Timedelta(days=skew_days)
    return out


def inject_null_event_time(df: pd.DataFrame, fraction: float, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    out.loc[out.index[_sample(df, fraction, seed, 505)], "event_time"] = pd.NaT
    return out


def inject_null_load_time(df: pd.DataFrame, fraction: float, seed: int = 0) -> pd.DataFrame:
    out = df.copy()
    out.loc[out.index[_sample(df, fraction, seed, 606)], "load_time"] = pd.NaT
    return out


def inject_null_source_insert(df: pd.DataFrame, fraction: float, seed: int = 0) -> pd.DataFrame:
    """NULL out source_insert_time on a sample (dual-timestamp tables only)."""
    out = df.copy()
    out.loc[out.index[_sample(df, fraction, seed, 707)], "source_insert_time"] = pd.NaT
    return out


# ------------------------------------------------------------------------ loaders


def load_into_duckdb(df: pd.DataFrame, con, table: str = "events") -> None:
    """Create/replace `table` in an open DuckDB connection from the DataFrame."""
    con.register("__synth_df", df)
    con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM __synth_df')
    con.unregister("__synth_df")


def load_via_sqlalchemy(df: pd.DataFrame, engine, table: str = "events") -> None:
    """Load through any SQLAlchemy engine — the mssql path for equivalence tests
    (mssql+pymssql) uses this same loader from Step 3 on."""
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5_000)
