"""Step 2 tests: the complete config schema, frozen and versioned BEFORE any
metric work. Valid configs load; each invalid shape produces a named, specific
error; the digest is computed over the secret-redacted canonical form."""

import pytest
from pydantic import ValidationError

from metricprobe.config import (
    CONFIG_SCHEMA_VERSION,
    AnalysisParams,
    ConfigError,
    ProbeConfig,
    compose_campaign,
    config_digest,
    load_config,
)


def minimal_table(**overrides) -> dict:
    return {
        "probe_name": "orders_main",
        "database": "demo_retail",
        "schema": "dbo",
        "table": "orders",
        "event_time": "order_date",
        "load_time": "loaded_at",
        "resolution": {"order_date": "date", "loaded_at": "datetime"},
    } | overrides


def minimal_config(**overrides) -> dict:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "connection_url": "duckdb:///:memory:",
        "tables": [minimal_table()],
    } | overrides


def via_spec(**overrides) -> dict:
    # the shape frozen in CLAUDE.md: {join_table, on: [{base_col, lookup_col}], column}
    return {
        "join_table": "demo_health.dbo.referrals",
        "on": [{"base_col": "referral_id", "lookup_col": "id"}],
        "column": "referral_date",
    } | overrides


# ------------------------------------------------------------------ valid shapes


def test_minimal_config_loads():
    config = ProbeConfig.model_validate(minimal_config())
    assert config.tables[0].probe_name == "orders_main"
    assert config.tables[0].optional is False
    assert config.tables[0].suppress_small_counts is False  # off by default (hard rule)


def test_full_config_loads():
    config = ProbeConfig.model_validate(
        minimal_config(
            tables=[
                minimal_table(
                    probe_name="orders_main",
                    source_insert_time="src_inserted_at",
                    load_batch_col="batch_id",
                    group_by_alt="region",
                    key_cols=["order_id"],
                    compare_event_time="order_date_raw",
                    parity_with="orders_replica",
                    optional=True,
                    proxy=True,
                    expect_batchy=True,
                    resolution={"order_date": "date", "loaded_at": "datetime",
                                "src_inserted_at": "datetime"},
                    suppress_small_counts=True,
                    read_uncommitted=True,
                    analysis={"training_cutoff_days": 400, "lag_cap_days": 400},
                ),
                minimal_table(
                    probe_name="orders_replica",
                    database="demo_retail_copy",
                    parity_with="orders_main",
                ),
                minimal_table(
                    probe_name="episodes_via_lookup",
                    table="episodes",
                    event_time=None,
                    event_time_via=via_spec(),
                    resolution={"referral_date": "date", "loaded_at": "datetime"},
                ),
            ],
            campaign={"schedule": "0 6 * * 1", "timezone": "Europe/London"},
            store={"backend": "duckdb", "path": "./store"},
            delivery={
                "remotes": [
                    {"name": "origin", "url": "https://forge.example/dash.git",
                     "ref": "main", "token_env": "DASHBOARD_PUSH_TOKEN"}
                ],
            },
        )
    )
    via = config.tables[2].event_time_via
    assert via.on[0].base_col == "referral_id"
    # the dotted locator decomposes
    assert (via.database, via.table_schema, via.table) == ("demo_health", "dbo", "referrals")
    assert config.delivery.remotes[0].token_env == "DASHBOARD_PUSH_TOKEN"


def test_yaml_loader_expands_env_and_survives_bare_on_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_CONN_URL", "duckdb:///:memory:")
    path = tmp_path / "probe.yaml"
    # NOTE: a bare `on:` key parses as boolean true under YAML 1.1 — the loader
    # must normalize it back so the DOCUMENTED spelling works from YAML files.
    path.write_text(
        f"""
schema_version: {CONFIG_SCHEMA_VERSION}
# expansion is textual, so values with YAML-special characters need quoting
connection_url: "${{TEST_CONN_URL}}"
tables:
  - probe_name: orders_main
    database: demo_retail
    schema: dbo
    table: orders
    event_time: order_date
    load_time: loaded_at
    resolution:
      order_date: date
      loaded_at: datetime
  - probe_name: episodes_via
    database: demo_health
    schema: dbo
    table: episodes
    load_time: loaded_at
    event_time_via:
      join_table: demo_health.dbo.referrals
      on:
        - base_col: referral_id
          lookup_col: id
      column: referral_date
    resolution:
      referral_date: date
      loaded_at: datetime
"""
    )
    config = load_config(path)
    assert config.connection_url == "duckdb:///:memory:"
    assert config.tables[1].event_time_via.on[0].lookup_col == "id"


def test_missing_env_var_named_in_error(tmp_path):
    path = tmp_path / "probe.yaml"
    path.write_text("connection_url: ${TEST_UNDEFINED_VAR_XYZ}\n")
    with pytest.raises(ConfigError, match="TEST_UNDEFINED_VAR_XYZ"):
        load_config(path)


# ------------------------------------------------- invalid shapes, named errors


def test_unknown_fields_rejected_at_every_level():
    with pytest.raises(ValidationError, match="bogus_top"):
        ProbeConfig.model_validate(minimal_config(bogus_top=1))
    with pytest.raises(ValidationError, match="bogus_table"):
        ProbeConfig.model_validate(minimal_config(tables=[minimal_table(bogus_table=1)]))
    with pytest.raises(ValidationError, match="training_cutof_days"):  # the typo case
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(analysis={"training_cutof_days": 400})])
        )


def test_event_time_xor_event_time_via():
    with pytest.raises(ValidationError, match="exactly one of event_time or event_time_via"):
        ProbeConfig.model_validate(minimal_config(tables=[minimal_table(event_time=None)]))
    with pytest.raises(ValidationError, match="exactly one of event_time or event_time_via"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(event_time_via=via_spec())])
        )


def test_blank_required_strings_rejected():
    # "" and whitespace-only values are unusable mappings, not valid ones
    for field, value in [
        ("event_time", ""),
        ("load_time", "   "),
        ("database", " "),
        ("schema", ""),
        ("table", "\t"),
    ]:
        with pytest.raises(ValidationError, match="blank"):
            ProbeConfig.model_validate(minimal_config(tables=[minimal_table(**{field: value})]))


def test_join_spec_requires_key_pairs():
    bad = via_spec(on=[])
    with pytest.raises(ValidationError, match="on"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(event_time=None, event_time_via=bad)])
        )


def test_join_table_must_be_full_locator():
    bad = via_spec(join_table="demo_health.referrals")  # schema part missing
    with pytest.raises(ValidationError, match="database.schema.table"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(event_time=None, event_time_via=bad)])
        )


def test_duplicate_probe_names_rejected():
    with pytest.raises(ValidationError, match="duplicate probe_name"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(), minimal_table(database="demo_other")])
        )


def test_parity_with_must_reference_an_existing_campaign_probe():
    # existence is CAMPAIGN-WIDE (validated at composition, so the target may
    # live in another config file); self-reference is rejected per file
    dangling = ProbeConfig.model_validate(
        minimal_config(tables=[minimal_table(parity_with="no_such_probe")])
    )
    with pytest.raises(ConfigError, match="no_such_probe"):
        compose_campaign([dangling])
    with pytest.raises(ValidationError, match="itself"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(parity_with="orders_main")])
        )


def test_campaign_composition_across_files():
    left = ProbeConfig.model_validate(
        minimal_config(tables=[minimal_table(parity_with="orders_replica")])
    )
    right = ProbeConfig.model_validate(
        minimal_config(
            tables=[minimal_table(probe_name="orders_replica", database="demo_copy")]
        )
    )
    compose_campaign([left, right])  # cross-file parity reference is valid
    # campaign-wide duplicate probe names are rejected
    with pytest.raises(ConfigError, match="duplicate probe_name"):
        compose_campaign([left, left])
    # every file must share ONE store
    other_store = ProbeConfig.model_validate(
        minimal_config(
            tables=[minimal_table(probe_name="orders_replica")],
            store={"path": "./elsewhere"},
        )
    )
    with pytest.raises(ConfigError, match="SAME store"):
        compose_campaign([left, other_store])


def test_training_cutoff_must_cover_lag_support():
    with pytest.raises(ValidationError, match="training_cutoff_days.*lag_cap_days"):
        ProbeConfig.model_validate(
            minimal_config(
                tables=[minimal_table(analysis={"training_cutoff_days": 100, "lag_cap_days": 365})]
            )
        )


def test_red_threshold_cannot_be_below_amber():
    with pytest.raises(ValidationError, match="volume_red_mads"):
        ProbeConfig.model_validate(
            minimal_config(
                tables=[minimal_table(analysis={"volume_amber_mads": 3.0, "volume_red_mads": 2.0})]
            )
        )


def test_resolution_keys_must_be_configured_columns():
    with pytest.raises(ValidationError, match="no_such_col"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(resolution={"no_such_col": "date"})])
        )


def test_empty_key_cols_rejected():
    with pytest.raises(ValidationError, match="key_cols"):
        ProbeConfig.model_validate(minimal_config(tables=[minimal_table(key_cols=[])]))


def test_schema_version_is_enforced():
    with pytest.raises(ValidationError, match="unsupported"):
        ProbeConfig.model_validate(minimal_config(schema_version=99))
    with pytest.raises(ValidationError, match="schema_version"):
        data = minimal_config()
        del data["schema_version"]
        ProbeConfig.model_validate(data)


def test_bad_connection_url_rejected():
    with pytest.raises(ValidationError, match="connection_url"):
        ProbeConfig.model_validate(minimal_config(connection_url="not a url at all"))


def test_campaign_schedule_must_be_real_cron():
    with pytest.raises(ValidationError, match="cron"):
        ProbeConfig.model_validate(minimal_config(campaign={"schedule": "whenever"}))
    # five tokens are not enough — each field must be valid cron syntax
    with pytest.raises(ValidationError, match="cron"):
        ProbeConfig.model_validate(minimal_config(campaign={"schedule": "cat dog eel fox yak"}))
    # and within its field's numeric bounds
    with pytest.raises(ValidationError, match="range"):
        ProbeConfig.model_validate(minimal_config(campaign={"schedule": "99 * * * *"}))
    ok = ProbeConfig.model_validate(minimal_config(campaign={"schedule": "*/15 0-6 * * 1,3,5"}))
    assert ok.campaign.schedule == "*/15 0-6 * * 1,3,5"


def test_bad_timezone_rejected():
    with pytest.raises(ValidationError, match="timezone"):
        ProbeConfig.model_validate(minimal_config(campaign={"timezone": "Neverland/Nowhere"}))


def test_token_env_must_be_an_uppercase_name_not_a_token():
    for bad in ["ghp_actual$token!value", "ghp_ABC123"]:
        with pytest.raises(ValidationError, match="UPPER_CASE"):
            ProbeConfig.model_validate(
                minimal_config(
                    delivery={
                        "remotes": [
                            {"name": "origin", "url": "https://forge.example/d.git",
                             "token_env": bad}
                        ]
                    }
                )
            )


def test_delivery_urls_must_not_embed_credentials():
    with pytest.raises(ValidationError, match="embed credentials"):
        ProbeConfig.model_validate(
            minimal_config(
                delivery={
                    "remotes": [
                        {"name": "origin", "url": "https://demo_token_value@forge.example/d.git"}
                    ]
                }
            )
        )


def test_delivery_urls_must_not_carry_literal_query_secrets():
    # userinfo is not the only smuggling route: literal secret-shaped query
    # parameters violate the env-var-NAMES-only contract just the same
    def remote(url):
        return minimal_config(delivery={"remotes": [{"name": "origin", "url": url}]})

    for bad in [
        "https://forge.example/d.git?token=demo-literal-value",
        "https://forge.example/d.git?ref=main&access_token=demo123",
        "https://forge.example/d.git?PRIVATE_TOKEN=demo123",
        "https://forge.example/d.git?api_key=",
    ]:
        with pytest.raises(ValidationError, match="token_env"):
            ProbeConfig.model_validate(remote(bad))
    # ${VAR}/$VAR references are env var NAMES, not values: allowed
    for ok_url in [
        "https://forge.example/d.git?token=${DASHBOARD_TOKEN}",
        "https://forge.example/d.git?token=$DASHBOARD_TOKEN",
        "https://forge.example/d.git?branch=main",  # not secret-shaped
    ]:
        ok = ProbeConfig.model_validate(remote(ok_url))
        assert ok.delivery.remotes[0].url == ok_url


def test_mssql_store_requires_url():
    with pytest.raises(ValidationError, match="mssql_url"):
        ProbeConfig.model_validate(minimal_config(store={"backend": "mssql"}))
    # the store schema comes from CONFIG, never hardcoded; blank is rejected
    ok = ProbeConfig.model_validate(
        minimal_config(
            store={"backend": "mssql", "mssql_url": "mssql+pymssql://localhost/demo",
                   "mssql_schema": "metricprobe_meta"}
        )
    )
    assert ok.store.mssql_schema == "metricprobe_meta"
    assert ProbeConfig.model_validate(minimal_config()).store.mssql_schema == "dbo"
    with pytest.raises(ValidationError, match="blank"):
        ProbeConfig.model_validate(
            minimal_config(store={"backend": "duckdb", "mssql_schema": " "})
        )


# ------------------------------------------------------------ digest + defaults


def _cfg(url: str) -> ProbeConfig:
    return ProbeConfig.model_validate(minimal_config(connection_url=url))


def test_digest_is_stable_and_secret_redacted():
    base = _cfg("mssql+pymssql://sa:demo_pw_one@localhost/demo")
    assert config_digest(base) == config_digest(base)
    assert len(config_digest(base)) == 64  # sha256 hex

    # rotating a secret must NOT change the digest, wherever the secret lives:
    # userinfo password ...
    assert config_digest(base) == config_digest(_cfg("mssql+pymssql://sa:demo_pw_two@localhost/demo"))
    # ... a password-named query parameter ...
    assert config_digest(_cfg("mssql+pymssql://localhost/demo?password=demo_pw_one")) == (
        config_digest(_cfg("mssql+pymssql://localhost/demo?password=demo_pw_two"))
    )
    # ... or a percent-encoded ODBC PWD inside odbc_connect
    odbc = "mssql+pyodbc:///?odbc_connect=DRIVER%3DODBC+Driver+18%3BSERVER%3Dlocalhost%3BPWD%3D"
    assert config_digest(_cfg(odbc + "demo_pw_one")) == config_digest(_cfg(odbc + "demo_pw_two"))
    # ... including secrets that CONTAIN percent-escapes: the whole value is
    # redacted, not just the prefix before the first %XX
    assert config_digest(_cfg("mssql+pymssql://localhost/demo?password=demo%2Fone")) == (
        config_digest(_cfg("mssql+pymssql://localhost/demo?password=demo%2Ftwo"))
    )
    assert config_digest(_cfg(odbc + "demo%2Fpw%2Bone")) == config_digest(
        _cfg(odbc + "demo%2Fpw%2Btwo")
    )
    # encoded separators still bound the value: what follows %3B is NOT eaten
    bounded = odbc + "demo_pw_one%3BEncrypt%3D"
    assert config_digest(_cfg(bounded + "yes")) != config_digest(_cfg(bounded + "no"))

    # semantic changes DO change the digest
    assert config_digest(base) != config_digest(_cfg("mssql+pymssql://sa:demo_pw_one@127.0.0.1/demo"))
    other_analysis = ProbeConfig.model_validate(
        minimal_config(tables=[minimal_table(analysis={"lag_cap_days": 200})])
    )
    assert config_digest(base) != config_digest(other_analysis)


def test_digest_distinguishes_plus_from_space():
    # a global percent-decode would make 'a+b' and 'a b' hash identically —
    # two semantically different configs must never share a resume digest
    plus = ProbeConfig.model_validate(minimal_config(store={"path": "demo a+b"}))
    space = ProbeConfig.model_validate(minimal_config(store={"path": "demo a b"}))
    assert config_digest(plus) != config_digest(space)


def test_analysis_defaults_are_frozen_v1():
    # Versioned defaults: changing ANY of these is a deliberate, visible act.
    assert AnalysisParams().model_dump() == {
        "training_cutoff_days": 365,
        "lag_cap_days": 365,
        "clock_skew_tolerance_days": 1.0,
        "negative_lag_red_fraction": 0.001,
        "min_mature_months": 6,
        "evaluation_window_months": 3,
        "freshness_bucket": "day",
        "freshness_min_epochs": 5,
        "freshness_zero_mad_tolerance_days": 1.0,
        "freshness_amber_mads": 2.0,
        "freshness_red_mads": 3.0,
        "volume_amber_mads": 2.0,
        "volume_red_mads": 3.0,
        "expected_fill_band_mads": 2.0,
        "parity_tolerance": 0,
        "result_cell_cap": 100_000,
    }


def test_resolution_is_required_for_time_roles_and_labels_the_grain():
    # the per-column resolution declaration is part of the frozen contract:
    # a probe without it is an incomplete mapping, not a valid one
    with pytest.raises(ValidationError, match="resolution"):
        ProbeConfig.model_validate(minimal_config(tables=[minimal_table(resolution={})]))
    with pytest.raises(ValidationError, match="src_inserted_at"):
        ProbeConfig.model_validate(
            minimal_config(tables=[minimal_table(source_insert_time="src_inserted_at")])
        )
    # the declared resolutions drive the output grain label: a date column on
    # either side means sub-day arrival detail does not exist
    config = ProbeConfig.model_validate(minimal_config())
    assert config.tables[0].lag_resolution == "date"  # order_date is a date
    assert config.tables[0].dual_lag_resolution is None  # no source timestamp
    dual = ProbeConfig.model_validate(
        minimal_config(
            tables=[
                minimal_table(
                    source_insert_time="src_inserted_at",
                    resolution={"order_date": "datetime", "loaded_at": "datetime",
                                "src_inserted_at": "datetime"},
                )
            ]
        )
    )
    assert dual.tables[0].lag_resolution == "datetime"
    assert dual.tables[0].dual_lag_resolution == "datetime"
