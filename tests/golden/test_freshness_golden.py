"""Step 5 goldens: freshness — cadence from DISTINCT ARRIVAL EPOCHS, never
per-row gaps. Healthy feed passes; a stopped feed trips staleness; a
duplicate-timestamp bulk load does NOT corrupt the cadence; minimum-epochs and
zero-MAD fallback boundaries are hand-calculated."""

import pandas as pd
import pytest
from tests.support import probe, table_config
from tests.synth import generator as g

from metricprobe.metrics.freshness import assess_freshness, epoch_timestamps
from metricprobe.status import ReasonCode, Severity

WEEKLY_BATCHES = g.TableSpec(
    name="events",
    start_month="2024-01",
    n_months=12,
    rows_per_month=2000,
    # four batches per month, ~weekly epochs
    lag_model=g.StepBatches(schedule=((3.0, 0.25), (10.0, 0.25), (17.0, 0.25), (24.0, 0.25))),
    seed=95,
)


def weekly_days_frame(n_weeks: int, rows_per_epoch: int = 40) -> pd.DataFrame:
    """EXACTLY weekly load days (gaps all 7.0 -> MAD 0): the zero-MAD fixture."""
    frames = []
    for week in range(n_weeks):
        load = pd.Timestamp("2024-01-08") + pd.Timedelta(weeks=week)
        frames.append(
            pd.DataFrame(
                {
                    "row_id": range(week * rows_per_epoch, (week + 1) * rows_per_epoch),
                    "event_time": load - pd.Timedelta(days=3),
                    "load_time": load,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _fresh(df, config, as_of):
    canonical = probe(df, config, as_of)
    return assess_freshness(canonical, config, pd.Timestamp(as_of))


def test_healthy_weekly_feed_passes():
    config = table_config(load_batch_col="batch_id")
    # last batch: 2024-12's day-24 batch = 2025-01-25; as_of 3 days later
    result = _fresh(g.generate(WEEKLY_BATCHES), config, "2025-01-28")
    assert [s.severity for s in result.statuses] == [Severity.GREEN]
    assert result.cadence_median_days == pytest.approx(7, abs=1)


def test_feed_stopped_three_cadences_trips_staleness():
    config = table_config(load_batch_col="batch_id")
    result = _fresh(g.generate(WEEKLY_BATCHES), config, "2025-02-20")  # ~26d silent
    stale = [s for s in result.statuses if s.reason is ReasonCode.STALE_FEED]
    assert stale and stale[0].severity is Severity.RED
    assert result.days_since_last > 3 * result.cadence_median_days


def test_bulk_load_with_duplicate_timestamps_does_not_corrupt_cadence():
    # 5000 extra rows all at ONE existing epoch timestamp: per-row gaps would
    # collapse toward zero, but epochs deduplicate — the cadence is unchanged
    clean = weekly_days_frame(20)
    bulk_ts = pd.Timestamp("2024-03-04")  # an existing weekly epoch
    bulk = pd.concat(
        [
            clean,
            pd.DataFrame(
                {
                    "row_id": range(100_000, 105_000),
                    "event_time": bulk_ts - pd.Timedelta(days=3),
                    "load_time": bulk_ts,
                }
            ),
        ],
        ignore_index=True,
    )
    config = table_config()  # no batch col: epochs = distinct load DAYS
    as_of = "2024-05-29"
    clean_result = _fresh(clean, config, as_of)
    bulk_result = _fresh(bulk, config, as_of)
    assert bulk_result.epoch_count == clean_result.epoch_count
    assert bulk_result.cadence_median_days == clean_result.cadence_median_days == 7.0
    assert [s.severity for s in bulk_result.statuses] == [Severity.GREEN]


def test_minimum_epochs_required():
    config = table_config()
    result = _fresh(weekly_days_frame(3), config, "2024-02-01")
    assert result.epoch_count == 3
    assert [s.reason for s in result.statuses] == [ReasonCode.INSUFFICIENT_EPOCHS]
    assert result.cadence_median_days is None


def test_min_epochs_one_with_a_single_epoch_is_insufficient_not_a_crash():
    # freshness_min_epochs: 1 is a VALID config, but one epoch has no
    # inter-epoch gap: cadence is unlearnable, reported as insufficient
    config = table_config(analysis={"freshness_min_epochs": 1})
    result = _fresh(weekly_days_frame(1), config, "2024-02-01")
    assert result.epoch_count == 1
    assert result.last_epoch == pd.Timestamp("2024-01-08")
    assert [s.reason for s in result.statuses] == [ReasonCode.INSUFFICIENT_EPOCHS]
    assert result.cadence_median_days is None
    # two epochs = one gap: with min_epochs 1 satisfied, cadence IS learnable
    two = _fresh(weekly_days_frame(2), config, "2024-01-16")
    assert two.cadence_median_days == 7.0
    assert [s.severity for s in two.statuses] == [Severity.GREEN]


def test_freshness_thresholds_come_from_config():
    # same fixture as the boundary test, stricter configured thresholds:
    # amber > 7 + 1*1 = 8, red > 7 + 1.5*1 = 8.5 (defaults would stay GREEN)
    config = table_config(
        analysis={"freshness_amber_mads": 1.0, "freshness_red_mads": 1.5}
    )
    df = weekly_days_frame(20)  # last epoch 2024-05-20, sigma floored at 1.0
    amber = _fresh(df, config, "2024-05-28 06:00")  # 8.25 days
    assert [s.severity for s in amber.statuses] == [Severity.AMBER]
    red = _fresh(df, config, "2024-05-28 18:00")  # 8.75 days
    assert [s.severity for s in red.statuses] == [Severity.RED]


def test_zero_mad_fallback_boundaries_hand_calculated():
    # exactly weekly: gaps all 7.0 -> MAD 0 -> sigma = the configured fixed
    # tolerance (1.0 day). Thresholds: amber > 7 + 2*1 = 9, red > 7 + 3*1 = 10.
    config = table_config()
    df = weekly_days_frame(20)  # last epoch 2024-05-20
    green = _fresh(df, config, "2024-05-28 12:00")  # 8.5 days
    assert [s.severity for s in green.statuses] == [Severity.GREEN]
    amber = _fresh(df, config, "2024-05-29 12:00")  # 9.5 days
    assert [s.severity for s in amber.statuses] == [Severity.AMBER]
    red = _fresh(df, config, "2024-05-30 12:00")  # 10.5 days
    assert [s.severity for s in red.statuses] == [Severity.RED]
    assert green.cadence_sigma_days == 1.0  # the zero-MAD fixed tolerance
    # the inequalities are STRICT: exactly AT a threshold stays on the calm side
    at_amber = _fresh(df, config, "2024-05-29 00:00")  # exactly 9.0 days
    assert at_amber.days_since_last == 9.0
    assert [s.severity for s in at_amber.statuses] == [Severity.GREEN]
    at_red = _fresh(df, config, "2024-05-30 00:00")  # exactly 10.0 days
    assert at_red.days_since_last == 10.0
    assert [s.severity for s in at_red.statuses] == [Severity.AMBER]


def test_batches_with_corrupt_event_times_are_still_epochs():
    # the last two months' rows get NULL event times (a real upstream
    # corruption): those batches are ineligible for curves but they ARE
    # arrivals — freshness must not report a false STALE_FEED
    config = table_config(load_batch_col="batch_id")
    df = g.generate(WEEKLY_BATCHES)
    corrupted = df.copy()
    recent = corrupted["load_time"] >= pd.Timestamp("2024-11-01")
    corrupted.loc[recent, "event_time"] = pd.NaT
    as_of = "2025-01-28"
    clean = _fresh(df, config, as_of)
    with_corruption = _fresh(corrupted, config, as_of)
    assert with_corruption.epoch_count == clean.epoch_count  # no epoch lost
    assert with_corruption.last_epoch == clean.last_epoch
    assert [s.severity for s in with_corruption.statuses] == [Severity.GREEN]


def test_epochs_come_from_batches_when_configured():
    df = g.generate(WEEKLY_BATCHES)
    with_batches = table_config(load_batch_col="batch_id")
    canonical = probe(df, with_batches, "2025-01-28")
    stamps = epoch_timestamps(canonical, with_batches)
    assert len(stamps) == 48  # 4 physical batches x 12 months
