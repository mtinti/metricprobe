"""The () grouping-set REPRESENTATION proof (review follow-up).

The frozen contract requires the empty () grouping set carrying the global
scalars — COUNT(*) vs COUNT(DISTINCT key_hash) — because computed per-set
they would not be global. The production statement REALIZES that () set as a
UNION ALL branch of the same statement: measured on SQL Server, putting
COUNT(DISTINCT HASHBYTES(...)) inside the GROUPING SETS plan forces a
per-row spool (~150x one scan), which would blow the <=3x read budget the
same contract mandates.

This test proves the two representations are SEMANTICALLY IDENTICAL, row for
row, on DuckDB (where the spool pathology does not exist and the literal
form is executable): the production result equals a literal
GROUPING SETS ((month, lag), (epoch), (month, batch), (alt), ()) query on
every column of every row, including the ()-row's global distinct count.
"""

from __future__ import annotations

import pandas as pd
import sqlalchemy as sa
from tests.support import engine_with, table_config
from tests.synth import generator as g
from tests.synth.scenarios import catalog

from metricprobe.extract.canonical import (
    GROUPING_SET_IDS,
    GROUPING_WEIGHTS,
    GroupingSetsClause,
    _bucket_sums,
    _staging_clause,
    run_canonical,
    staging_sql,
)

AS_OF = "2026-07-01"


def _literal_grouping_sets_query(table):
    """The contract's LITERAL form: one GROUPING SETS clause including the
    empty () set, with the global distinct count computed inside it."""
    staging = _staging_clause("duckdb")
    month_key = staging.c.event_month
    lag_key = staging.c.lag_day
    epoch_key = staging.c.load_epoch_day
    batch_key = staging.c.batch_id
    alt_key = staging.c.alt_value
    sets = [[month_key, lag_key], [epoch_key]]
    grouping_id = (
        sa.func.grouping(month_key) * GROUPING_WEIGHTS["event_month"]
        + sa.func.grouping(lag_key) * GROUPING_WEIGHTS["lag_day"]
        + sa.func.grouping(epoch_key) * GROUPING_WEIGHTS["load_epoch_day"]
    )
    if table.load_batch_col:
        sets.append([month_key, batch_key])
        grouping_id = grouping_id + sa.func.grouping(batch_key) * GROUPING_WEIGHTS["batch_id"]
    else:
        grouping_id = grouping_id + GROUPING_WEIGHTS["batch_id"]
    if table.group_by_alt:
        sets.append([alt_key])
        grouping_id = grouping_id + sa.func.grouping(alt_key) * GROUPING_WEIGHTS["alt_value"]
    else:
        grouping_id = grouping_id + GROUPING_WEIGHTS["alt_value"]
    sets.append([])  # THE empty grouping set, literally
    select_batch = batch_key if table.load_batch_col else sa.func.max(batch_key)
    select_alt = alt_key if table.group_by_alt else sa.func.max(alt_key)
    distinct_keys = (
        sa.func.count(sa.distinct(staging.c.key_hash))
        if table.key_cols
        else sa.cast(sa.null(), sa.BigInteger())
    )
    return (
        sa.select(
            grouping_id.label("grouping_id"),
            month_key.label("event_month"),
            lag_key.label("lag_day"),
            epoch_key.label("load_epoch_day"),
            select_batch.label("batch_id"),
            select_alt.label("alt_value"),
            *_bucket_sums(staging),
            distinct_keys.label("distinct_keys"),
            sa.func.min(staging.c.load_time).label("min_load_time"),
        )
        .select_from(staging)
        .group_by(GroupingSetsClause(sets))
    )


def _sortable(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["grouping_id", "event_month", "lag_day", "load_epoch_day", "batch_id", "alt_value"]
    out = frame.copy()
    for column in ("event_month", "load_epoch_day", "min_load_time"):
        out[column] = pd.to_datetime(out[column])
    return out.sort_values(keys, na_position="first").reset_index(drop=True)


def test_union_all_branch_equals_the_literal_empty_grouping_set():
    df = g.inject_duplicate_keys(catalog()["straggler_batch"].healthy(), 0.02, seed=5)
    config = table_config(load_batch_col="batch_id", key_cols=["row_id"])
    engine = engine_with(df, "events")
    try:
        production = run_canonical(engine, config, pd.Timestamp(AS_OF)).frame
        # replay the same staging, then run the LITERAL grouping-sets form
        with engine.connect() as conn:
            key_types = {"row_id": "BIGINT"}
            conn.exec_driver_sql(
                staging_sql(config, "duckdb", as_of=pd.Timestamp(AS_OF), key_types=key_types)
            )
            literal = pd.DataFrame(
                conn.execute(_literal_grouping_sets_query(config)).mappings().all()
            )
    finally:
        engine.dispose()

    production = _sortable(production)
    literal = _sortable(literal)
    assert len(production) == len(literal)
    # the () row is IDENTICAL — including the global distinct count the
    # contract exists for
    prod_global = production[production["grouping_id"] == GROUPING_SET_IDS["global"]]
    lit_global = literal[literal["grouping_id"] == GROUPING_SET_IDS["global"]]
    assert len(prod_global) == len(lit_global) == 1
    for column in production.columns:
        left, right = prod_global.iloc[0][column], lit_global.iloc[0][column]
        assert (pd.isna(left) and pd.isna(right)) or left == right, column
    assert int(prod_global.iloc[0]["distinct_keys"]) < int(
        prod_global.iloc[0]["row_count"]
    )  # duplicates injected: the count is genuinely GLOBAL, not per-set
    # every dimensional row matches on every column except distinct_keys —
    # which is exactly WHY the () set exists: per-set distincts are not the
    # global answer (production intentionally leaves them NULL)
    comparable = [c for c in production.columns if c != "distinct_keys"]
    pd.testing.assert_frame_equal(
        production[comparable], literal[comparable], check_dtype=False
    )
