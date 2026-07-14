"""Typed configuration (pydantic v2), frozen and versioned BEFORE any metric work.

Layers: ProbeConfig (connection + probe/table entries) plus CampaignConfig
(schedule, timezone, grace period, manual-run behavior), StoreConfig (backend,
path, retention) and DeliveryConfig (remotes, refs, worktree, token env-var
NAMES — never token values).

Contract highlights
  * Unknown fields are REJECTED at every level (typo safety); required string
    fields reject blank/whitespace-only values.
  * schema_version is required and must match CONFIG_SCHEMA_VERSION.
  * Every probe entry needs a unique probe_name, a full table locator
    (database.schema.table) and EXACTLY ONE of event_time XOR event_time_via.
    The join spec is the documented {join_table, on: [{base_col, lookup_col}],
    column} shape; because YAML 1.1 parses a bare `on` key as boolean true, a
    before-validator normalizes that (and accepts `join_on` as an alias).
  * All analysis parameters are explicit with versioned defaults; validation
    enforces training_cutoff_days >= lag_cap_days so the training cohort covers
    the modeled lag support.
  * config_digest() hashes the secret-redacted canonical form: URL userinfo and
    secret-named query parameters (password/pwd/token/..., raw or
    percent-encoded) are masked before hashing, so credential rotation does not
    change the digest and the digest can be stored publicly. Delivery remote
    URLs must not embed credentials at all — that is what token_env is for.
  * The YAML loader expands ${ENV_VAR} references and fails loudly, naming
    every undefined variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
import zoneinfo
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlalchemy.engine import make_url

# v2: StoreConfig.mssql_schema, freshness_amber_mads/freshness_red_mads,
# per-column resolution REQUIRED for the time roles, delivery query-secret
# rejection (v1 configs without resolution no longer validate).
# v3: compare_event_time joined the required-resolution set, connection_url
# restricted to the supported dialects (mssql/duckdb), non-finite analysis
# thresholds rejected, mssql_url validated — v2 configs can be rejected under
# these rules, so the version moved with them.
# v4: AnalysisParams.extraction_months (optional per-probe event-time bound,
# month-aligned; validated against training_cutoff/min_mature/evaluation so a
# bound can never construct an always-insufficient probe)
CONFIG_SCHEMA_VERSION = 4


class ConfigError(Exception):
    """Raised for config-file problems: unreadable YAML, missing env vars,
    or validation failures surfaced through the loader."""


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlankStr = Annotated[str, AfterValidator(_not_blank)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class JoinKey(_Model):
    base_col: NonBlankStr
    lookup_col: NonBlankStr


class JoinSpec(_Model):
    """event_time_via: borrow the event time from a related table, in the
    documented shape {join_table, on, column}. The lookup side must be unique
    on the join key (asserted at probe time, Step 3)."""

    join_table: NonBlankStr  # full "database.schema.table" locator
    on: tuple[JoinKey, ...] = Field(min_length=1)
    column: NonBlankStr

    @model_validator(mode="before")
    @classmethod
    def _normalize_on_key(cls, data):
        if isinstance(data, dict):
            # YAML 1.1 parses a bare `on:` mapping key as boolean true
            if True in data:
                data = {("on" if key is True else key): value for key, value in data.items()}
            if "join_on" in data and "on" not in data:
                data = dict(data)
                data["on"] = data.pop("join_on")
        return data

    @model_validator(mode="after")
    def _locator_shape(self) -> JoinSpec:
        parts = self.join_table.split(".")
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError(
                f'join_table {self.join_table!r} must be a full "database.schema.table" locator'
            )
        return self

    @property
    def database(self) -> str:
        return self.join_table.split(".")[0]

    @property
    def table_schema(self) -> str:
        return self.join_table.split(".")[1]

    @property
    def table(self) -> str:
        return self.join_table.split(".")[2]


class AnalysisParams(_Model):
    """All analysis parameters, explicit with versioned defaults (v1 values are
    pinned by test_analysis_defaults_are_frozen_v1)."""

    # float thresholds reject inf/NaN (allow_inf_nan=False): an infinite
    # tolerance or MAD multiplier would silently disable a required verdict
    training_cutoff_days: int = Field(default=365, gt=0)
    lag_cap_days: int = Field(default=365, gt=0)
    # None = full history (the default contract). A configured bound admits
    # only the last N calendar months of EVENT time (month-aligned so the
    # oldest admitted month is complete): it trades long-memory volume
    # baselines for bounded staging/cells, and is labelled on outputs.
    extraction_months: int | None = Field(default=None, gt=0)
    clock_skew_tolerance_days: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    negative_lag_red_fraction: float = Field(
        default=0.001, gt=0, le=1, allow_inf_nan=False
    )
    min_mature_months: int = Field(default=6, gt=0)
    evaluation_window_months: int = Field(default=3, gt=0)
    freshness_bucket: Literal["day", "hour"] = "day"
    freshness_min_epochs: int = Field(default=5, gt=0)
    freshness_zero_mad_tolerance_days: float = Field(
        default=1.0, gt=0, allow_inf_nan=False
    )
    freshness_amber_mads: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    freshness_red_mads: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    volume_amber_mads: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    volume_red_mads: float = Field(default=3.0, gt=0, allow_inf_nan=False)
    expected_fill_band_mads: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    parity_tolerance: int = Field(default=0, ge=0)
    result_cell_cap: int = Field(default=100_000, gt=0)

    @model_validator(mode="after")
    def _cross_checks(self) -> AnalysisParams:
        if self.extraction_months is not None:
            # the bound must leave a usable training cohort: cutoff months
            # + the minimum mature months + the open month, else every probe
            # would land in insufficient_history by construction
            floor = (
                int(self.training_cutoff_days / 30.4375)
                + self.min_mature_months
                + self.evaluation_window_months
                + 1
            )
            if self.extraction_months < floor:
                raise ValueError(
                    f"extraction_months ({self.extraction_months}) is too small: "
                    f"training_cutoff_days={self.training_cutoff_days} plus "
                    f"min_mature_months={self.min_mature_months} plus the "
                    f"evaluation window needs at least {floor} months"
                )
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
        if self.freshness_red_mads < self.freshness_amber_mads:
            raise ValueError(
                f"freshness_red_mads ({self.freshness_red_mads}) must be >= "
                f"freshness_amber_mads ({self.freshness_amber_mads})"
            )
        return self


class TableConfig(_Model):
    """One probe entry. A table can be probed multiple times under different
    probe_names (variants are first-class in snapshots, dashboards, figures)."""

    probe_name: NonBlankStr
    database: NonBlankStr
    table_schema: NonBlankStr = Field(alias="schema")
    table: NonBlankStr
    event_time: NonBlankStr | None = None
    event_time_via: JoinSpec | None = None
    load_time: NonBlankStr
    source_insert_time: NonBlankStr | None = None
    load_batch_col: NonBlankStr | None = None
    group_by_alt: NonBlankStr | None = None
    key_cols: tuple[NonBlankStr, ...] | None = Field(default=None, min_length=1)
    compare_event_time: NonBlankStr | None = None
    parity_with: NonBlankStr | None = None
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
        # the per-column resolution declaration is REQUIRED for every
        # configured time column — the metric roles (event/load/source) AND
        # compare_event_time: outputs label their grain from it
        required = {
            column
            for column in (
                self.event_time,
                self.event_time_via.column if self.event_time_via else None,
                self.load_time,
                self.source_insert_time,
                self.compare_event_time,
            )
            if column
        }
        missing = sorted(required - set(self.resolution))
        if missing:
            raise ValueError(
                f"probe {self.probe_name!r}: resolution (date | datetime) must be "
                f"declared for time column(s) {missing}"
            )
        return self

    @property
    def lag_resolution(self) -> str:
        """The lag grain basis for output labels: 'date' when either side of
        the lag is a date column (sub-day arrival detail does not exist),
        else 'datetime'."""
        event_column = self.event_time or self.event_time_via.column
        sides = (self.resolution[event_column], self.resolution[self.load_time])
        return "date" if "date" in sides else "datetime"

    @property
    def dual_lag_resolution(self) -> str | None:
        """Same label for the source-side lag (event -> source_insert_time)."""
        if not self.source_insert_time:
            return None
        event_column = self.event_time or self.event_time_via.column
        sides = (self.resolution[event_column], self.resolution[self.source_insert_time])
        return "date" if "date" in sides else "datetime"


# minute, hour, day-of-month, month, day-of-week (0-7, both 0 and 7 = Sunday)
_CRON_BOUNDS = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
_CRON_PART = re.compile(r"^(\*|\d+(?:-\d+)?)(?:/(\d+))?$")


def _validate_cron(value: str) -> None:
    fields = value.split()
    if len(fields) != 5:
        raise ValueError(f"schedule must be a 5-field cron expression, got {value!r}")
    for field_text, (low, high) in zip(fields, _CRON_BOUNDS, strict=True):
        for part in field_text.split(","):
            match = _CRON_PART.match(part)
            if not match:
                raise ValueError(
                    f"schedule {value!r} is not a valid cron expression: bad field {field_text!r}"
                )
            body, step = match.group(1), match.group(2)
            if step is not None and int(step) < 1:
                raise ValueError(
                    f"cron field {field_text!r} has a zero step in schedule {value!r}"
                )
            if body == "*":
                continue
            numbers = [int(number) for number in body.split("-")]
            if any(not low <= number <= high for number in numbers) or (
                len(numbers) == 2 and numbers[0] > numbers[1]
            ):
                raise ValueError(
                    f"cron field {field_text!r} is out of range {low}-{high} "
                    f"in schedule {value!r}"
                )


class CampaignConfig(_Model):
    schedule: str | None = None  # 5-field cron, None = manual-only
    timezone: str = "UTC"
    grace_period_hours: float = Field(default=6.0, ge=0, allow_inf_nan=False)
    manual_run_behavior: Literal["allow", "forbid"] = "allow"

    @field_validator("schedule")
    @classmethod
    def _schedule_is_cron(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_cron(value)
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
    path: NonBlankStr = "./metricprobe_store"
    retention_runs: int | None = Field(default=None, gt=0)  # None = keep all
    mssql_url: str | None = None  # env-expandable; required for backend "mssql"
    mssql_schema: NonBlankStr = "dbo"  # never hardcoded in SQL — always from config

    @model_validator(mode="after")
    def _mssql_needs_url(self) -> StoreConfig:
        if self.backend == "mssql":
            if not self.mssql_url or not self.mssql_url.strip():
                raise ValueError('store backend "mssql" requires mssql_url')
            # a malformed URL should fail at CONFIG time, not at the first
            # write; ${VAR} references are expanded by the loader before
            # validation, so what arrives here must already parse
            try:
                backend = make_url(self.mssql_url).get_backend_name()
            except Exception as exc:
                raise ValueError(
                    f"mssql_url is not a valid SQLAlchemy URL: {exc}"
                ) from exc
            if backend != "mssql":
                raise ValueError(
                    f"mssql_url must use an mssql dialect, got {backend!r}"
                )
        return self


_ENV_VAR_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_URL_WITH_USERINFO = re.compile(r"^[A-Za-z][\w+.-]*://[^/@\s]+@")
# secret-shaped query parameters in a delivery URL: literal values violate the
# token-env-var-NAMES-only contract; ${VAR}/$VAR references are names, not values
_SECRET_QUERY_PARAM = re.compile(
    r"[?&](token|access_token|private_token|oauth2?_token|password|passwd|pwd"
    r"|secret|client_secret|api_key|apikey|key|sig|signature|auth|authorization"
    r"|credential|credentials)=([^&#\s]*)",
    re.IGNORECASE,
)
_ENV_VAR_REFERENCE = re.compile(r"^\$(\{[A-Z_][A-Z0-9_]*\}|[A-Z_][A-Z0-9_]*)$")


class DeliveryRemote(_Model):
    name: NonBlankStr
    url: NonBlankStr
    ref: NonBlankStr = "main"
    token_env: str | None = None

    @field_validator("token_env")
    @classmethod
    def _token_env_is_a_name(cls, value: str | None) -> str | None:
        if value is not None and not _ENV_VAR_NAME.match(value):
            raise ValueError(
                f"token_env {value!r} must be the UPPER_CASE NAME of an environment "
                "variable (A-Z, digits, underscores), never the token value itself"
            )
        return value

    @field_validator("url")
    @classmethod
    def _no_embedded_credentials(cls, value: str) -> str:
        # detection runs on the raw URL AND its percent-DECODED form: encoding
        # a credential name (%74oken=...) or separator must not smuggle a
        # literal secret past the env-var-NAMES-only contract
        for candidate in dict.fromkeys((value, urllib.parse.unquote(value))):
            if _URL_WITH_USERINFO.match(candidate):
                raise ValueError(
                    f"delivery remote url {value!r} must not embed credentials; "
                    "supply the token through token_env instead"
                )
            for match in _SECRET_QUERY_PARAM.finditer(candidate):
                name, literal = match.group(1), match.group(2)
                if not _ENV_VAR_REFERENCE.match(literal):
                    raise ValueError(
                        f"delivery remote url carries a literal value for query "
                        f"parameter {name!r}; config holds env var NAMES, never "
                        "secret values — supply the token through token_env"
                    )
        return value


class DeliveryConfig(_Model):
    remotes: tuple[DeliveryRemote, ...] = Field(min_length=1)
    worktree: NonBlankStr = "./dashboard_worktree"

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
    connection_url: NonBlankStr
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
            backend = make_url(value).get_backend_name()
        except Exception as exc:
            raise ValueError(f"connection_url is not a valid SQLAlchemy URL: {exc}") from exc
        # extraction compiles mssql or duckdb SQL — any other dialect would
        # silently receive duckdb-flavoured statements
        if backend not in ("mssql", "duckdb"):
            raise ValueError(
                f"connection_url uses unsupported dialect {backend!r}; "
                "supported: mssql, duckdb"
            )
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> ProbeConfig:
        names = [table.probe_name for table in self.tables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate probe_name(s): {duplicates}")
        for table in self.tables:
            if table.parity_with == table.probe_name:
                raise ValueError(
                    f"probe {table.probe_name!r}: parity_with must not reference itself"
                )
        # parity_with targets are CAMPAIGN-WIDE and may live in another config
        # file: existence is validated by compose_campaign()
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

# URL userinfo: the PASSWORD is masked, the principal is KEPT — a reader and
# a writer login to the same server must hash differently (they may see
# different rows, and the digest guards resume identity). A single-component
# userinfo (bare token) is masked entirely: it is a credential.
_URL_USERINFO_WITH_PASSWORD = re.compile(r"://([^:@/\s]+):[^@/\s]*@")
_URL_USERINFO_BARE = re.compile(r"://[^:@/\s]+@")

_SECRET_NAMES = (
    "password", "passwd", "pwd", "token", "access_token", "secret",
    "api_key", "apikey", "sig", "signature",
)


def _encoded_spelling(name: str) -> str:
    """A regex matching the name with ANY subset of its characters
    percent-encoded (p%61ssword), so an encoded credential name cannot dodge
    redaction. Case-insensitivity comes from the compiled (?i) flag."""
    return "".join(f"(?:{re.escape(ch)}|%{ord(ch):02x})" for ch in name)


# secret-named query/connection-string parameters, matched in BOTH the raw
# and the percent-encoded spelling (encoded names, encoded assignments,
# behind raw or encoded separators & ; %26 %3B) WITHOUT transforming the rest
# of the string: a global decode would make distinct configs ('a+b' vs
# 'a b') hash identically. A matched secret token may CONTINUE with
# underscore-joined words (s3_secret_access_key, token_expiry) — a compound
# name containing a secret word is treated as a secret; over-redaction only
# makes the digest blind to that value, under-redaction hashes a credential.
_SECRET_PARAM = re.compile(
    r"(?i)(?:(?<![A-Za-z0-9])|(?<=%3[Bb])|(?<=%26))"
    r"(" + "|".join(_encoded_spelling(name) for name in _SECRET_NAMES) + r")"
    r"((?:(?:_|%5[Ff])(?:[A-Za-z0-9]|%[3-6][0-9A-Fa-f])*)*)"
    # the value may itself contain percent-escapes (a%2Fb): consume them so the
    # WHOLE secret is redacted, but stop at encoded separators %26/%3B exactly
    # as at their raw spellings
    r"(=|%3[Dd])((?:%(?!26|3[Bb])[0-9A-Fa-f]{2}|[^&;\s\"'%])*)"
)


def _redact(value):
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        value = _URL_USERINFO_WITH_PASSWORD.sub(r"://\1:***@", value)
        value = _URL_USERINFO_BARE.sub("://***@", value)
        return _SECRET_PARAM.sub(
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}***", value
        )
    return value


def config_digest(config: ProbeConfig) -> str:
    """SHA-256 over the secret-redacted canonical JSON form. Credential rotation
    does not change the digest; any semantic config change does."""
    canonical = _redact(config.model_dump(mode="json", by_alias=True))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compose_campaign(configs: list[ProbeConfig]) -> None:
    """Validate repeatable --config files as ONE campaign: probe names are
    campaign-wide unique, parity_with targets exist campaign-wide, and every
    file shares one store (the runner has exactly one snapshot store)."""
    if not configs:
        raise ConfigError("at least one config is required")
    names = [table.probe_name for config in configs for table in config.tables]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ConfigError(f"duplicate probe_name(s) across config files: {duplicates}")
    for config in configs:
        for table in config.tables:
            if table.parity_with is not None and table.parity_with not in names:
                raise ConfigError(
                    f"probe {table.probe_name!r}: parity_with references unknown "
                    f"probe {table.parity_with!r}; campaign probes: {sorted(names)}"
                )
    stores = {config.store for config in configs}
    if len(stores) > 1:
        raise ConfigError(
            "all config files in one campaign must declare the SAME store; "
            f"found {len(stores)} distinct store configurations"
        )
    campaigns = {config.campaign for config in configs}
    if len(campaigns) > 1:
        raise ConfigError(
            "all config files in one campaign must declare the SAME campaign "
            "settings (schedule/timezone/grace) — 'Next update expected by' is "
            f"a campaign property; found {len(campaigns)} distinct ones"
        )
    deliveries = {config.delivery for config in configs}
    if len(deliveries) > 1:
        raise ConfigError(
            "all config files in one campaign must declare the SAME delivery "
            "settings (remotes/refs/worktree) or none — delivery is owned by "
            f"the one campaign command; found {len(deliveries)} distinct ones"
        )
