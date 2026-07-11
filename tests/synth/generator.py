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

Determinism: every month uses its own RNG stream seeded by (seed, month_index),
so healthy/unhealthy twins share byte-identical rows in unaffected months.

All timestamps are floored to whole seconds so values survive SQL Server
DATETIME's 3.33 ms rounding unchanged in equivalence tests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LognormalLag:
    """Lag-days ~ Lognormal(mu, sigma): gradual trickle arrivals."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
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
        if any(day < 0 for day, _ in self.schedule):
            raise ValueError("StepBatches days must be >= 0 (measured from month end)")
        total = sum(fraction for _, fraction in self.schedule)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"StepBatches fractions must sum to 1, got {total}")


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
    rng = np.random.default_rng([spec.seed, index])  # independent per-month stream
    start = period.start_time
    month_end = (period + 1).start_time
    length_days = (month_end - start) / pd.Timedelta(days=1)

    event = (start + pd.to_timedelta(rng.uniform(0, length_days, n), unit="D")).floor("s")
    batch_id: list[str] | None = None
    if isinstance(spec.lag_model, LognormalLag):
        lag_days = rng.lognormal(spec.lag_model.mu, spec.lag_model.sigma, n)
        arrival = (event + pd.to_timedelta(lag_days, unit="D")).floor("s")
    else:
        days = np.array([day for day, _ in spec.lag_model.schedule])
        fractions = np.array([fraction for _, fraction in spec.lag_model.schedule])
        choice = rng.choice(len(days), size=n, p=fractions / fractions.sum())
        batch_times = (month_end + pd.to_timedelta(days, unit="D")).floor("s")
        arrival = pd.DatetimeIndex(batch_times[choice])
        batch_id = [f"{period}-run{c}" for c in choice]

    frame = pd.DataFrame({"row_id": index * 10_000_000 + np.arange(n), "event_time": event})
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
) -> dict[str, dict[int, float]]:
    """Days-to-pXX per event month, derived from the generating parameters alone.

    timestamp="load" is the local-arrival curve (includes the dual offset when
    configured); "source" is the source-side curve and requires dual timestamps.
    For StepBatches the lag CDF is the closed form
        F(t) = sum_b fraction_b * clamp((t - day_b) / month_len, 0, 1)
    (events uniform over the month, batch at month_end + day_b), inverted
    numerically — still parameter-derived, never fitted to generated data.
    """
    if timestamp not in ("load", "source"):
        raise ValueError("timestamp must be 'load' or 'source'")
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
            if isinstance(spec.lag_model, LognormalLag):
                base = spec.lag_model.quantile(q)
            else:
                base = _step_quantile(spec.lag_model.schedule, length_days, q)
            values[pct] = base + offset
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


# ------------------------------------------------------- volume-plan injectors
# These return a NEW spec (frozen dataclasses), leaving the healthy twin intact.


def volume_spike(spec: TableSpec, month_index: int, factor: float = 6.0) -> TableSpec:
    """One month at `factor` times its normal volume."""
    return _with_override(spec, {month_index: factor})


def missing_month(spec: TableSpec, month_index: int) -> TableSpec:
    """One month entirely absent from the data."""
    return _with_override(spec, {month_index: 0.0})


def sustained_collapse(spec: TableSpec, last_k: int = 3, factor: float = 0.1) -> TableSpec:
    """Loads keep arriving on their normal cadence, but the last `last_k` months
    carry ~`factor` of the historical volume."""
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
