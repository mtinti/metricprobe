# metricprobe

Data arrival latency & completeness probes for database tables.

When can you trust that a month of data is complete? `metricprobe` measures arrival
latency and completeness for SQL tables: completion curves, freshness and volume
checks, and a git-friendly status dashboard.

Given a table with an *event time* column (when the fact happened) and a *load
time* column (when the row landed in the warehouse), each probe answers three
questions per table: **healthy?** (traffic light), **updating?** (freshness) and
**complete back to: DATE** (months ending on or before this date are expected
≥95% complete). Only bounded aggregates ever leave the database, under an
enforced logical-reads budget; raw rows are never pulled.

Supported sources: **SQL Server** and **DuckDB** (the same pipeline runs on both;
CI proves the numbers match). Snapshots land in a parquet/DuckDB store by
default, or a SQL Server schema behind a config flag.

## Quickstart

```
python3 -m venv .venv
# the supported install path is the release tag (PyPI is phase 2):
.venv/bin/python -m pip install "metricprobe[export] @ git+https://github.com/mtinti/metricprobe@v0.1.7"
# (or, from a checkout: pip install -e ".[export]")
.venv/bin/metricprobe discover --url "$DB_URL" --database MyDb --out probe.yaml
# review the draft: confirm the event/load column guesses, then
.venv/bin/metricprobe run --config probe.yaml
.venv/bin/metricprobe report --config probe.yaml --out ./report
```

`metricprobe run` executes every configured probe sequentially under one run id
and commits an atomic snapshot. With a `delivery:` section configured it owns
the whole lifecycle in one process (analyse, commit, render the dashboard,
push it to your git remotes) and exits 0 (all green), 2 (data-health red) or
1 (a stage failed; earlier stages stay committed).

`metricprobe publish --config probe.yaml --out ./dashboard` emits a
forge-renderable markdown dashboard (README + SVG figures + self-contained
HTML report) from the latest committed run, so a scheduled workflow can make
the repo front page the dashboard.

**What the output looks like:** see the committed demo dashboard at
[`reports/README.md`](reports/README.md): four fixed-seed synthetic
databases (retail orders, IoT telemetry, card settlements, health episode
records) with healthy and unhealthy twins, so every verdict is visible.
A synthetic-data sample (an orders feed's completion curves):

![synthetic example: completion curves](reports/img/orders_completion_curves.svg)

Regenerate it with `python examples/demo.py --out reports` (synthetic data
only; CI diff-checks it). `examples/demo_config.yaml` is an annotated config
template and `examples/workflow.yml` a scheduled-campaign workflow template.

## Static image export (report/publish)

Rendering figures to PNG/SVG uses Plotly's **kaleido**, which requires an
installed **Chrome or Chromium**:

```
pip install "metricprobe[export] @ git+https://github.com/mtinti/metricprobe@v0.1.7"
```

- Linux/macOS/Windows with Chrome already installed: nothing else to do.
- **Locked-down Windows machines (no admin rights):** run `kaleido_get_chrome`
  once. It downloads a private Chromium build into your user profile; no
  system install or elevation required.
- CI runners: GitHub's Ubuntu images ship Chrome preinstalled.

`metricprobe report`/`publish` verify the export chain at startup and print
these instructions if anything is missing.

## Development

```
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -m "not equivalence"   # fast suite (DuckDB)
.venv/bin/python -m ruff check .
```

The equivalence suite compares results between DuckDB and a real SQL Server. It skips
unless `METRICPROBE_MSSQL_URL` points at a reachable SQL Server (CI runs it against
the official container on Linux x86-64):

```
docker run -e ACCEPT_EULA=Y -e MSSQL_SA_PASSWORD='Metricprobe1!' -p 1433:1433 \
  -d mcr.microsoft.com/mssql/server:2022-latest
export METRICPROBE_MSSQL_URL='mssql+pymssql://sa:Metricprobe1!@localhost:1433/tempdb'
.venv/bin/python -m pytest -m equivalence
```

The exact metric formulas are frozen in `docs/ALGORITHMS.md`; the ordered build
plan lives in `PLAN.md`.

## License

MIT
