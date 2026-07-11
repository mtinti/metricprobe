"""Step 5 goldens: parity between two NAMED probes — twin tests for the full
outer join, the watermarked population, the common-mature-by-time rule,
prerequisites, and tolerance."""


import pandas as pd
import pytest
from tests.support import table_config
from tests.synth import generator as g

from metricprobe.extract.canonical import run_canonical
from metricprobe.metrics.completion import assess_completion
from metricprobe.metrics.parity import ParitySide, assess_parity
from metricprobe.metrics.volume import assess_volume
from metricprobe.status import ReasonCode, Severity

AS_OF = pd.Timestamp("2026-07-01")

SPEC = g.TableSpec(
    name="events",
    start_month="2023-01",
    n_months=30,
    rows_per_month=2000,
    lag_model=g.LognormalLag(mu=1.6, sigma=0.8),
    seed=91,
)


@pytest.fixture(scope="module")
def base_df():
    return g.generate(SPEC)


def _side(engine_df, table_name, as_of=AS_OF, **config_overrides) -> ParitySide:
    import sqlalchemy as sa

    config = table_config(
        probe_name=f"{table_name}_probe",
        table=table_name,
        key_cols=["row_id"],
        **config_overrides,
    )
    engine = sa.create_engine("duckdb:///:memory:")
    try:
        g.load_via_sqlalchemy(engine_df, engine, table_name)
        canonical = run_canonical(engine, config, as_of)
    finally:
        engine.dispose()
    completion = assess_completion(canonical, config, as_of)
    volume = assess_volume(canonical, config, as_of, completion)
    return ParitySide(config=config, canonical=canonical, completion=completion, volume=volume)


def _left(df, as_of=AS_OF, **overrides) -> ParitySide:
    return _side(df, "events_a", as_of=as_of, parity_with="events_b_probe", **overrides)


def _drop_month(df: pd.DataFrame, month: str) -> pd.DataFrame:
    return df[df["event_time"].dt.to_period("M") != month].reset_index(drop=True)


def test_identical_probes_are_green(base_df):
    left = _left(base_df)
    result = assess_parity(left, _side(base_df, "events_b"), AS_OF)
    assert [s.severity for s in result.statuses] == [Severity.GREEN]
    assert all(row.verdict == "match" and row.diff == 0 for row in result.rows)
    assert len(result.rows) == len(left.completion.mature_months)


def test_parity_with_must_declare_the_pairing(base_df):
    unpaired = _side(base_df, "events_a")  # no parity_with declared
    with pytest.raises(ValueError, match="parity_with"):
        assess_parity(unpaired, _side(base_df, "events_b"), AS_OF)


def test_missing_month_is_an_explicit_one_sided_red(base_df):
    # the divergence twin: 2024-01 (mature) missing on the right — it must be
    # an explicit RED diff, never silently dropped from the comparison
    result = assess_parity(
        _left(base_df), _side(_drop_month(base_df, "2024-01"), "events_b"), AS_OF
    )
    one_sided = [s for s in result.statuses if s.reason is ReasonCode.PARITY_ONE_SIDED_MONTH]
    assert len(one_sided) == 1 and one_sided[0].severity is Severity.RED
    assert "2024-01" in one_sided[0].detail and "left" in one_sided[0].detail
    by_month = {row.month: row for row in result.rows}
    assert by_month[pd.Period("2024-01", freq="M")].verdict == "left_only"


def test_count_mismatch_is_red_and_tolerance_absorbs_it(base_df):
    march = base_df["event_time"].dt.to_period("M") == "2024-03"
    right = base_df.drop(base_df[march].index[:50]).reset_index(drop=True)

    result = assess_parity(_left(base_df), _side(right, "events_b"), AS_OF)
    mismatches = [s for s in result.statuses if s.reason is ReasonCode.PARITY_MISMATCH]
    assert len(mismatches) == 1 and "2024-03" in mismatches[0].detail
    assert "+50" in mismatches[0].detail

    tolerant = assess_parity(
        _left(base_df, analysis={"parity_tolerance": 100}), _side(right, "events_b"), AS_OF
    )
    assert [s.severity for s in tolerant.statuses] == [Severity.GREEN]


def test_only_the_common_mature_population_is_compared(base_df):
    # 2025-05 is immature at this as_of: dropping it on the right must NOT
    # produce any verdict — immature months are outside the comparison
    as_of = pd.Timestamp("2025-07-10")
    result = assess_parity(
        _left(base_df, as_of=as_of),
        _side(_drop_month(base_df, "2025-05"), "events_b", as_of=as_of),
        as_of,
    )
    assert [s.severity for s in result.statuses] == [Severity.GREEN]
    assert pd.Period("2025-05", freq="M") not in {row.month for row in result.rows}


def test_different_horizons_on_identical_data_stay_green(base_df):
    # the left side learns under a STRICTER cutoff (400d vs 365d): identical
    # data must never manufacture one-sided months — the common mature
    # population is taken under the stricter of the two horizons
    left = _left(base_df, analysis={"training_cutoff_days": 400, "lag_cap_days": 365})
    right = _side(base_df, "events_b")
    assert left.completion.horizon != right.completion.horizon
    result = assess_parity(left, right, AS_OF)
    assert [s.severity for s in result.statuses] == [Severity.GREEN]
    # every compared month is mature under the STRICTER horizon
    strict_edge = AS_OF - pd.Timedelta(days=max(left.completion.horizon, right.completion.horizon))
    assert all((row.month + 1).start_time <= strict_edge for row in result.rows)


def test_watermarked_population_includes_allowed_negative_lags(base_df):
    # a below-threshold negative-lag excess on ONE side must not create false
    # diffs: those rows are still watermarked (non-NULL load <= as_of), only
    # excluded from curves. 0.05% < the 0.1% prerequisite threshold.
    corrupted = g.inject_negative_lags(base_df, fraction=0.0005, seed=11, skew_days=5.0)
    left = _left(base_df)
    right = _side(corrupted, "events_b")
    assert right.completion.negative_lag_excess_fraction > 0  # rows ARE excluded
    result = assess_parity(left, right, AS_OF)
    assert [s.severity for s in result.statuses] == [Severity.GREEN]


def test_injected_duplicates_yield_indeterminate_not_false_mismatch(base_df):
    # duplicates inflate the right side's counts — a naive diff would scream
    # MISMATCH; the violated append-only/uniqueness prerequisite must yield
    # INDETERMINATE instead
    right = g.inject_duplicate_keys(base_df, fraction=0.01, seed=5)
    result = assess_parity(_left(base_df), _side(right, "events_b"), AS_OF)
    assert [s.severity for s in result.statuses] == [Severity.INDETERMINATE]
    assert result.statuses[0].reason is ReasonCode.PARITY_PREREQ_UNIQUENESS
    assert not any(s.reason is ReasonCode.PARITY_MISMATCH for s in result.statuses)
    assert result.rows == []  # no month verdicts under a failed prerequisite


def test_read_uncommitted_yields_indeterminate(base_df):
    result = assess_parity(
        _left(base_df, read_uncommitted=True), _side(base_df, "events_b"), AS_OF
    )
    assert [s.severity for s in result.statuses] == [Severity.INDETERMINATE]
    assert result.statuses[0].reason is ReasonCode.PARITY_PREREQ_READ_UNCOMMITTED


def test_missing_key_cols_is_unverifiable_hence_indeterminate(base_df):
    left = _left(base_df)
    right = _side(base_df, "events_b")
    right.config = table_config(probe_name="events_b_probe", table="events_b")
    right.volume.duplicate_rows = None
    result = assess_parity(left, right, AS_OF)
    assert [s.severity for s in result.statuses] == [Severity.INDETERMINATE]
    assert result.statuses[0].reason is ReasonCode.PARITY_PREREQ_UNIQUENESS
    assert "unverifiable" in result.statuses[0].detail


def test_null_load_rows_are_informational_never_in_the_diff(base_df):
    # identical NULL-load injections on both sides: the watermarked counts
    # still match (GREEN) and the query-time NULL counts are reported
    nulled = g.inject_null_load_time(base_df, fraction=0.02, seed=7)
    result = assess_parity(_left(nulled), _side(nulled, "events_b"), AS_OF)
    assert [s.severity for s in result.statuses] == [Severity.GREEN]
    assert result.null_load_left == result.null_load_right > 0


def test_negative_lag_prereq_boundary_is_strict(base_df):
    """The contract requires the excess to be BELOW the threshold: sitting
    exactly AT it must fail the prerequisite (INDETERMINATE), never GREEN."""
    from metricprobe.metrics.parity import _failed_prerequisite

    left = _left(base_df)
    right = _side(base_df, "events_right")
    threshold = left.config.analysis.negative_lag_red_fraction  # 0.001
    left.completion.negative_lag_excess_fraction = threshold  # exactly AT it
    failed = _failed_prerequisite(left, right)
    assert failed is not None
    assert failed[0] is ReasonCode.PARITY_PREREQ_NEGATIVE_LAG
    # strictly below passes
    left.completion.negative_lag_excess_fraction = threshold / 2
    assert _failed_prerequisite(left, right) is None
