"""Typed configuration (pydantic v2), frozen and versioned BEFORE any metric work.

Layers: ProbeConfig (connection + probe/table entries) plus CampaignConfig
(schedule, timezone, grace period, manual-run behavior), StoreConfig (backend,
path, retention) and DeliveryConfig (remotes, refs, worktree, token env-var
NAMES — never token values).

Contract highlights
  * Unknown fields are REJECTED at every level (typo safety).
  * schema_version is required and must match CONFIG_SCHEMA_VERSION.
  * Every probe entry needs a unique probe_name, a full table locator
    (database.schema.table) and EXACTLY ONE of event_time XOR event_time_via.
    The join spec uses `join_on` (not `on`: YAML 1.1 parses a bare `on` key as
    boolean true, the same foot-gun GitHub Actions lives with).
  * All analysis parameters are explicit with versioned defaults; validation
    enforces training_cutoff_days >= lag_cap_days so the training cohort covers
    the modeled lag support.
  * config_digest() hashes the secret-redacted canonical form: URL passwords
    are masked before hashing, so a credential rotation does not change the
    digest and the digest can be stored publicly.
  * The YAML loader expands ${ENV_VAR} references and fails loudly, naming
    every undefined variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zoneinfo
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from sqlalchemy.engine import make_url

CONFIG_SCHEMA_VERSION = 1


class ConfigError(Exception):
    """Raised for config-file problems: unreadable YAML, missing env vars,
    or validation failures surfaced through the loader."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class JoinKey(_Model):
    base_col: str = Field(min_length=1)
    lookup_col: str = Field(min_length=1)


class JoinSpec(_Model):
    """event_time_via: borrow the event time from a related table. The lookup
    side must be unique on the join key (asserted at probe time, Step 3)."""

    database: str = Field(min_length=1)
    table_schema: str = Field(min_length=1, alias="schema")
    table: str = Field(min_length=1)
    join_on: tuple[JoinKey, ...] = Field(min_length=1)
    column: str = Field(min_length=1)


class AnalysisParams(_Model):
    """All analysis parameters, explicit with versioned defaults (v1 values are
    pinned by test_analysis_defaults_are_frozen_v1)."""

    training_cutoff_days: int = Field(default=365, gt=0)
    lag_cap_days: int = Field(default=365, gt=0)
    clock_skew_tolerance_days: float = Field(default=1.0, ge=0)
    negative_lag_red_fraction: float = Field(default=0.001, gt=0, le=1)
    min_mature_months: int = Field(default=6, gt=0)
    evaluation_window_months: int = Field(default=3, gt=0)
    freshness_bucket: Literal["day", "hour"] = "day"
    freshness_min_epochs: int = Field(default=5, gt=0)
    freshness_zero_mad_tolerance_days: float = Field(default=1.0, gt=0)
    volume_amber_mads: float = Field(default=2.0, gt=0)
    volume_red_mads: float = Field(default=3.0, gt=0)
    expected_fill_band_mads: float = Field(default=2.0, gt=0)
    parity_tolerance: int = Field(default=0, ge=0)
    result_cell_cap: int = Field(default=100_000, gt=0)

    @model_validator(mode="after")
    def _cross_checks(self) -> AnalysisParams:
        if self.training_cutoff_days < self.lag_cap_days:
            raise ValueError(
                f"training_cutoff_days ({self.training_cutoff_days}) must be >= "
                f"lag_cap_days ({self.lag_cap_days}) so the training cohort covers "
                "the modeled lag support"
            )
        if self.volume_red_mads < self.volume_amber_mads:
            raise ValueError(
                f"volume_red_mads ({self.volume_red_mads}) must be >= "
                f"volume_amber_mads ({self.volume_amber_mads})"
            )
        return self


class TableConfig(_Model):
    """One probe entry. A table can be probed multiple times under different
    probe_names (variants are first-class in snapshots, dashboards, figures)."""

    probe_name: str = Field(min_length=1)
    database: str = Field(min_length=1)
    table_schema: str = Field(min_length=1, alias="schema")
    table: str = Field(min_length=1)
    event_time: str | None = None
    event_time_via: JoinSpec | None = None
    load_time: str = Field(min_length=1)
    source_insert_time: str | None = None
    load_batch_col: str | None = None
    group_by_alt: str | None = None
    key_cols: tuple[str, ...] | None = Field(default=None, min_length=1)
    compare_event_time: str | None = None
    parity_with: str | None = None
    expected_cadence_days: float | None = Field(default=None, gt=0)
    optional: bool = False
    proxy: bool = False
    expect_batchy: bool = False
    resolution: dict[str, Literal["date", "datetime"]] = Field(default_factory=dict)
    suppress_small_counts: bool = False  # off by default (hard rule)
    read_uncommitted: bool = False
    analysis: AnalysisParams = AnalysisParams()

    @model_validator(mode="after")
    def _cross_checks(self) -> TableConfig:
        if (self.event_time is None) == (self.event_time_via is None):
            raise ValueError(
                f"probe {self.probe_name!r}: exactly one of event_time or event_time_via "
                "must be set"
            )
        time_columns = {
            column
            for column in (
                self.event_time,
                self.load_time,
                self.source_insert_time,
                self.compare_event_time,
                self.event_time_via.column if self.event_time_via else None,
            )
            if column
        }
        unknown = sorted(set(self.resolution) - time_columns)
        if unknown:
            raise ValueError(
                f"probe {self.probe_name!r}: resolution declared for unknown column(s) "
                f"{unknown}; configured time columns are {sorted(time_columns)}"
            )
        return self


class CampaignConfig(_Model):
    schedule: str | None = None  # 5-field cron, None = manual-only
    timezone: str = "UTC"
    grace_period_hours: float = Field(default=6.0, ge=0)
    manual_run_behavior: Literal["allow", "forbid"] = "allow"

    @field_validator("schedule")
    @classmethod
    def _schedule_is_cron(cls, value: str | None) -> str | None:
        if value is not None and len(value.split()) != 5:
            raise ValueError(f"schedule must be a 5-field cron expression, got {value!r}")
        return value

    @field_validator("timezone")
    @classmethod
    def _timezone_exists(cls, value: str) -> str:
        try:
            zoneinfo.ZoneInfo(value)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value


class StoreConfig(_Model):
    backend: Literal["duckdb", "parquet", "mssql"] = "duckdb"
    path: str = "./metricprobe_store"
    retention_runs: int | None = Field(default=None, gt=0)  # None = keep all
    mssql_url: str | None = None  # env-expandable; required for backend "mssql"

    @model_validator(mode="after")
    def _mssql_needs_url(self) -> StoreConfig:
        if self.backend == "mssql" and not self.mssql_url:
            raise ValueError('store backend "mssql" requires mssql_url')
        return self


_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DeliveryRemote(_Model):
    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    ref: str = "main"
    token_env: str | None = None

    @field_validator("token_env")
    @classmethod
    def _token_env_is_a_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME.match(value):
            raise ValueError(
                f"token_env {value!r} must be the NAME of an environment variable "
                "(letters, digits, underscores), never the token value itself"
            )
        return value


class DeliveryConfig(_Model):
    remotes: tuple[DeliveryRemote, ...] = Field(min_length=1)
    worktree: str = "./dashboard_worktree"

    @model_validator(mode="after")
    def _unique_remote_names(self) -> DeliveryConfig:
        names = [remote.name for remote in self.remotes]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate delivery remote name(s): {duplicates}")
        return self


class ProbeConfig(_Model):
    """The complete, versioned configuration for one probe campaign."""

    schema_version: int
    connection_url: str = Field(min_length=1)
    tables: tuple[TableConfig, ...] = Field(min_length=1)
    campaign: CampaignConfig = CampaignConfig()
    store: StoreConfig = StoreConfig()
    delivery: DeliveryConfig | None = None  # None = render/delivery not configured

    @field_validator("schema_version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {value}; this version of metricprobe "
                f"supports {CONFIG_SCHEMA_VERSION}"
            )
        return value

    @field_validator("connection_url")
    @classmethod
    def _url_parses(cls, value: str) -> str:
        try:
            make_url(value)
        except Exception as exc:
            raise ValueError(f"connection_url is not a valid SQLAlchemy URL: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> ProbeConfig:
        names = [table.probe_name for table in self.tables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate probe_name(s): {duplicates}")
        for table in self.tables:
            if table.parity_with is None:
                continue
            if table.parity_with == table.probe_name:
                raise ValueError(
                    f"probe {table.probe_name!r}: parity_with must not reference itself"
                )
            if table.parity_with not in names:
                raise ValueError(
                    f"probe {table.probe_name!r}: parity_with references unknown probe "
                    f"{table.parity_with!r}; known probes: {sorted(names)}"
                )
        return self


# --------------------------------------------------------------------- loading

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(text: str, environ: dict[str, str] | None = None) -> str:
    """Expand ${VAR} references; fail loudly naming EVERY undefined variable."""
    env = dict(os.environ) if environ is None else environ
    missing = sorted({name for name in _ENV_REF.findall(text) if name not in env})
    if missing:
        raise ConfigError(
            f"config references undefined environment variable(s): {', '.join(missing)}"
        )
    return _ENV_REF.sub(lambda match: env[match.group(1)], text)


def load_config(path: str | Path) -> ProbeConfig:
    """Load and validate one YAML config file, expanding ${ENV_VAR} references."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    expanded = expand_env(text)
    try:
        data = yaml.safe_load(expanded)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    try:
        return ProbeConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc


# ---------------------------------------------------------------------- digest

# user:password@ credentials inside any URL-shaped string value
_URL_CREDENTIALS = re.compile(r"://([^:/@\s]+):([^@/\s]+)@")


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _URL_CREDENTIALS.sub(r"://\1:***@", value)
    return value


def config_digest(config: ProbeConfig) -> str:
    """SHA-256 over the secret-redacted canonical JSON form. Credential rotation
    does not change the digest; any semantic config change does."""
    canonical = _redact(config.model_dump(mode="json", by_alias=True))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
