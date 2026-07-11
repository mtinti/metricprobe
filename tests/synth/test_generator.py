"""Unit tests for the synthetic arrival generator itself (PLAN Step 1).

The generator is the foundation of the test-first method: every later metric is
validated against data whose expected values are DERIVABLE from these parameters,
so the generator's own behavior (counts, determinism, expected-value helper) must
be nailed down first.
"""

import dataclasses

import duckdb
import numpy as np
import pandas as pd
import pytest
import sqlalchemy
from tests.synth.generator import (
    LognormalLag,
    StepBatches,
    TableSpec,
    expected_days_to_percentiles,
    generate,
    inject_duplicate_keys,
    inject_negative_lags,
    inject_null_event_time,
    inject_null_load_time,
    inject_raw_vs_corrected,
    inject_straggler_batch,
    load_into_duckdb,
    load_via_sqlalchemy,
    missing_month,
    sustained_collapse,
    volume_drop,
    volume_spike,
)
from tests.synth.scenarios import catalog

TRICKLE = TableSpec(
    name="orders",
    start_month="2024-01",
    n_months=3,
    rows_per_month=1000,
    lag_model=LognormalLag(mu=1.6, sigma=0.8),
    seed=7,
)

BATCHY = TableSpec(
    name="settlements",
    start_month="2024-01",
    n_months=3,
    rows_per_month=1200,
    lag_model=StepBatches(schedule=((3.0, 0.6), (10.0, 0.3), (20.0, 0.1))),
    seed=7,
)


def month_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df["event_time"].dt.to_period("M")).size()


# ---------------------------------------------------------------- core generation


def test_row_counts_and_schema():
    df = generate(TRICKLE)
    assert len(df) == 3000
    assert list(month_counts(df)) == [1000, 1000, 1000]
    assert df["row_id"].is_unique
    assert {"row_id", "event_time", "load_time"} <= set(df.columns)
    # events stay inside their month, loads never precede events
    assert (df["load_time"] >= df["event_time"]).all()


def test_seed_determinism():
    pd.testing.assert_frame_equal(generate(TRICKLE), generate(TRICKLE))
    other = generate(dataclasses.replace(TRICKLE, seed=8))
    assert not other["event_time"].equals(generate(TRICKLE)["event_time"])


def test_volume_overrides():
    spiked = generate(volume_spike(TRICKLE, 1, factor=6.0))
    assert list(month_counts(spiked)) == [1000, 6000, 1000]
    dropped = generate(volume_drop(TRICKLE, 1, factor=0.1))
    assert list(month_counts(dropped)) == [1000, 100, 1000]
    gap = generate(missing_month(TRICKLE, 1))
    assert list(month_counts(gap)) == [1000, 1000]
    collapsed = generate(sustained_collapse(TRICKLE, last_k=2, factor=0.1))
    assert list(month_counts(collapsed)) == [1000, 100, 100]


def test_override_factors_validated():
    with pytest.raises(ValueError):
        volume_spike(TRICKLE, 1, factor=0.5)  # a spike must increase volume
    with pytest.raises(ValueError):
        volume_drop(TRICKLE, 1, factor=1.5)  # a drop must decrease it
    with pytest.raises(ValueError):
        sustained_collapse(TRICKLE, last_k=2, factor=0.0)  # zero months are missing_month


def test_volume_twin_rows_are_a_prefix_of_the_healthy_month():
    # Volume twins must be isolated to the volume pathology: the retained rows of
    # a dropped month are byte-identical to the healthy twin's first k rows (event
    # times AND lags), not a resample. Guaranteed by separate per-purpose RNG
    # streams plus NumPy's sequential-fill prefix property.
    healthy = generate(TRICKLE)
    dropped = generate(volume_drop(TRICKLE, 1, factor=0.1))
    dropped_feb = dropped[dropped["row_id"] // 10_000_000 == 1].reset_index(drop=True)
    healthy_feb = healthy[healthy["row_id"].isin(dropped_feb["row_id"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(dropped_feb, healthy_feb)
    # same guarantee for the batchy model (batch assignment draws)
    healthy_b = generate(BATCHY)
    dropped_b = generate(volume_drop(BATCHY, 1, factor=0.1))
    dropped_feb_b = dropped_b[dropped_b["row_id"] // 10_000_000 == 1].reset_index(drop=True)
    healthy_feb_b = healthy_b[healthy_b["row_id"].isin(dropped_feb_b["row_id"])].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(dropped_feb_b, healthy_feb_b)


def test_overrides_do_not_perturb_other_months():
    # Twin-pair guarantee: per-month RNG streams are independent, so injecting a
    # pathology into one month leaves every other month's rows byte-identical.
    healthy = generate(TRICKLE)
    collapsed = generate(sustained_collapse(TRICKLE, last_k=1, factor=0.1))
    jan_healthy = healthy[healthy["event_time"] < "2024-02-01"].reset_index(drop=True)
    jan_collapsed = collapsed[collapsed["event_time"] < "2024-02-01"].reset_index(drop=True)
    pd.testing.assert_frame_equal(jan_healthy, jan_collapsed)


def test_out_of_range_override_rejected():
    with pytest.raises(ValueError):
        volume_spike(TRICKLE, 99)


# ------------------------------------------------------- expected-value helper


def day_lags(df: pd.DataFrame) -> np.ndarray:
    """Per-row lag with the production metric's DATEDIFF(day) semantics:
    calendar-day boundaries crossed between event_time and load_time."""
    return (df["load_time"].dt.normalize() - df["event_time"].dt.normalize()).dt.days.to_numpy()


def _check_both_grains(spec, df, month):
    """Compare the parameter-derived oracle against brute-force empirical
    percentiles at BOTH grains; accumulate all failures into one message."""
    continuous = expected_days_to_percentiles(spec, grain="continuous")[month]
    day = expected_days_to_percentiles(spec)[month]  # grain="day" is the default
    lag_continuous = (df["load_time"] - df["event_time"]).dt.total_seconds() / 86_400
    lag_day = day_lags(df)
    failures = []
    for pct in (50, 90, 95, 99):
        emp_c = float(np.quantile(lag_continuous, pct / 100))
        # Tolerance: at N>=150k the p99 sampling std-error is ~0.2 days on a
        # ~32-day value (<1%); 5% relative absorbs that plus the 1-second floor.
        if abs(emp_c - continuous[pct]) / continuous[pct] > 0.05:
            failures.append(f"continuous p{pct}: oracle {continuous[pct]:.2f}d, got {emp_c:.2f}d")
        emp_d = int(np.quantile(lag_day, pct / 100, method="inverted_cdf"))
        # Tolerance +-1 day: when the day-grain CDF sits within sampling error of
        # the target at the boundary day, the empirical order statistic can land
        # one day either side (e.g. batchy p99: F(48)=0.9903 vs q=0.99).
        if abs(emp_d - day[pct]) > 1:
            failures.append(f"day p{pct}: oracle {day[pct]}, got {emp_d}")
    assert not failures, "; ".join(failures)


def test_lognormal_expected_percentiles_match_brute_force():
    spec = dataclasses.replace(TRICKLE, n_months=1, rows_per_month=200_000, seed=42)
    _check_both_grains(spec, generate(spec), "2024-01")


def test_step_expected_percentiles_match_brute_force():
    spec = dataclasses.replace(BATCHY, n_months=1, rows_per_month=150_000, seed=42)
    _check_both_grains(spec, generate(spec), "2024-01")


def test_step_oracle_matches_hand_calculation():
    # Hand-calculated for schedule (3d,60%),(10d,30%),(20d,10%) in January (31d):
    # continuous: F(t)=.6+( .4t-5)/31 on t in (34,41); F(t)=.95 => t=39.625.
    # day grain (DATEDIFF semantics): F(39)=.6+.3*29/31+.1*19/31=.9419 < .95 and
    # F(40)=.6+.3*30/31+.1*20/31=.9548 >= .95 => p95 is integer day 40, NOT 39.625.
    spec = dataclasses.replace(BATCHY, n_months=1)
    continuous = expected_days_to_percentiles(spec, grain="continuous")["2024-01"]
    assert continuous[95] == pytest.approx(39.625, abs=1e-6)
    day = expected_days_to_percentiles(spec)["2024-01"]
    assert day[95] == 40
    assert isinstance(day[95], int)


def test_expected_percentiles_are_monotone():
    for spec in (TRICKLE, BATCHY):
        for month, vals in expected_days_to_percentiles(spec).items():
            ordered = [vals[p] for p in (50, 90, 95, 99)]
            assert ordered == sorted(ordered), (spec.name, month)


def test_source_percentiles_require_dual():
    with pytest.raises(ValueError):
        expected_days_to_percentiles(TRICKLE, timestamp="source")


# ------------------------------------------------------------------ step batches


def test_step_batches_structure():
    df = generate(BATCHY)
    assert df["batch_id"].notna().all()
    # exactly one physical batch per schedule entry per month, loading at
    # month_end + day; every load_time is one of those batch timestamps
    for period, group in df.groupby(df["event_time"].dt.to_period("M")):
        month_end = (period + 1).start_time
        expected_times = {month_end + pd.Timedelta(days=d) for d, _ in BATCHY.lag_model.schedule}
        assert set(group["load_time"]) == expected_times
        assert group["batch_id"].nunique() == 3
        fractions = group.groupby("batch_id").size().sort_index() / len(group)
        for got, (_, want) in zip(fractions, BATCHY.lag_model.schedule, strict=True):
            assert abs(got - want) < 0.05


def test_step_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        StepBatches(schedule=((3.0, 0.5), (10.0, 0.3)))


def test_step_individual_fractions_validated():
    # sums to 1 but individual fractions are invalid
    with pytest.raises(ValueError):
        StepBatches(schedule=((3.0, 1.2), (10.0, -0.2)))
    # a zero fraction would silently produce no physical batch for that entry
    with pytest.raises(ValueError):
        StepBatches(schedule=((3.0, 1.0), (10.0, 0.0)))


# ------------------------------------------------------------- dual timestamps


def test_dual_timestamps_offset_is_exact():
    spec = dataclasses.replace(TRICKLE, dual_offset_days=2.0)
    df = generate(spec)
    assert (df["source_insert_time"] >= df["event_time"]).all()
    deltas = df["load_time"] - df["source_insert_time"]
    assert (deltas == pd.Timedelta(days=2)).all()
    # load percentiles = source percentiles + offset
    exp = expected_days_to_percentiles(spec)["2024-01"]
    exp_src = expected_days_to_percentiles(spec, timestamp="source")["2024-01"]
    assert all(abs(exp[p] - exp_src[p] - 2.0) < 1e-9 for p in exp)


# ---------------------------------------------------------------------- injectors


def test_inject_duplicate_keys():
    df = generate(TRICKLE)
    dup = inject_duplicate_keys(df, fraction=0.02, seed=1)
    assert len(dup) == len(df) + round(0.02 * len(df))
    assert dup["row_id"].duplicated().sum() == round(0.02 * len(df))
    assert not df["row_id"].duplicated().any()


def test_inject_straggler_batch():
    df = generate(BATCHY)
    month_rows = (df["event_time"].dt.to_period("M") == "2024-02").sum()
    late = inject_straggler_batch(df, month="2024-02", late_day=45.0, fraction=0.15, seed=1)
    stragglers = late[late["batch_id"] == "2024-02-straggler"]
    assert len(stragglers) == round(0.15 * month_rows)
    assert (stragglers["load_time"] == pd.Timestamp("2024-03-01") + pd.Timedelta(days=45)).all()
    assert len(late) == len(df)  # rows moved, not added: final counts unchanged


def test_inject_raw_vs_corrected():
    df = generate(TRICKLE)
    out = inject_raw_vs_corrected(df, fraction=0.1, shift_days=-35, seed=1)
    mismatches = (out["event_time_raw"] != out["event_time"]).sum()
    assert mismatches == round(0.1 * len(df))
    clean = inject_raw_vs_corrected(df, fraction=0.0, shift_days=-35, seed=1)
    assert (clean["event_time_raw"] == clean["event_time"]).all()


def test_inject_negative_lags():
    df = generate(TRICKLE)
    out = inject_negative_lags(df, fraction=0.05, skew_days=3.0, seed=1)
    negative = out["load_time"] < out["event_time"]
    assert negative.sum() == round(0.05 * len(df))
    lags = (out.loc[negative, "event_time"] - out.loc[negative, "load_time"]).dt.total_seconds()
    assert (lags == 3 * 86_400).all()


def test_inject_null_source_insert():
    from tests.synth.generator import inject_null_source_insert

    spec = dataclasses.replace(TRICKLE, dual_offset_days=2.0)
    df = generate(spec)
    out = inject_null_source_insert(df, 0.04, seed=1)
    assert out["source_insert_time"].isna().sum() == round(0.04 * len(df))
    assert df["source_insert_time"].notna().all()  # input never mutated


def test_inject_nulls():
    df = generate(TRICKLE)
    null_event = inject_null_event_time(df, 0.03, seed=1)
    assert null_event["event_time"].isna().sum() == round(0.03 * len(df))
    null_load = inject_null_load_time(df, 0.04, seed=1)
    assert null_load["load_time"].isna().sum() == round(0.04 * len(df))
    assert df["event_time"].notna().all()  # inputs never mutated


# ------------------------------------------------------------------------ loaders


def test_duckdb_loader_round_trip():
    df = generate(BATCHY)
    con = duckdb.connect()
    load_into_duckdb(df, con, table="events")
    assert con.execute('SELECT count(*) FROM "events"').fetchone()[0] == len(df)
    months = con.execute(
        "SELECT count(DISTINCT date_trunc('month', event_time)) FROM \"events\""
    ).fetchone()[0]
    assert months == 3


def test_sqlalchemy_loader_round_trip():
    df = generate(TRICKLE).head(500)
    engine = sqlalchemy.create_engine("duckdb:///:memory:")
    load_via_sqlalchemy(df, engine, table="events")
    with engine.connect() as conn:
        n = conn.execute(sqlalchemy.text('SELECT count(*) FROM "events"')).scalar_one()
    assert n == 500


# ---------------------------------------------------------------- scenario catalog


def test_catalog_has_every_planned_pathology():
    assert set(catalog()) == {
        "volume_spike",
        "volume_drop",
        "missing_month",
        "duplicate_keys",
        "straggler_batch",
        "raw_vs_corrected",
        "sustained_collapse",
        "sustained_collapse_short",
    }
    for pair in catalog().values():
        assert pair.description and pair.expected_detection


def test_twins_differ_only_in_the_injected_pathology():
    pairs = catalog()

    healthy = pairs["volume_spike"].healthy()
    spiked = pairs["volume_spike"].unhealthy()
    ratio = month_counts(spiked) / month_counts(healthy)
    assert (ratio > 5).sum() == 1 and (ratio == 1).sum() == len(ratio) - 1

    dropped = pairs["volume_drop"].unhealthy()
    drop_ratio = month_counts(dropped) / month_counts(pairs["volume_drop"].healthy())
    assert (drop_ratio < 0.15).sum() == 1 and (drop_ratio == 1).sum() == len(drop_ratio) - 1

    gap = pairs["missing_month"].unhealthy()
    assert len(month_counts(gap)) == len(month_counts(pairs["missing_month"].healthy())) - 1

    dups = pairs["duplicate_keys"].unhealthy()
    assert dups["row_id"].duplicated().any()
    assert not pairs["duplicate_keys"].healthy()["row_id"].duplicated().any()

    straggler = pairs["straggler_batch"].unhealthy()
    assert straggler["batch_id"].str.endswith("straggler").any()
    assert not pairs["straggler_batch"].healthy()["batch_id"].str.endswith("straggler").any()

    raw = pairs["raw_vs_corrected"].unhealthy()
    assert (raw["event_time_raw"] != raw["event_time"]).any()
    raw_healthy = pairs["raw_vs_corrected"].healthy()
    assert (raw_healthy["event_time_raw"] == raw_healthy["event_time"]).all()

    collapsed = pairs["sustained_collapse"].unhealthy()
    healthy_counts = month_counts(pairs["sustained_collapse"].healthy())
    collapsed_counts = month_counts(collapsed)
    tail_ratio = (collapsed_counts / healthy_counts).iloc[-3:]
    assert (tail_ratio < 0.15).all()
    # loads keep arriving on their normal cadence: batch timestamps unchanged
    assert set(collapsed["load_time"].unique()) == set(
        pairs["sustained_collapse"].healthy()["load_time"].unique()
    )
