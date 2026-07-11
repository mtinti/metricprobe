# metricprobe

Data arrival latency & completeness probes for database tables.

When can you trust that a month of data is complete? `metricprobe` measures arrival
latency and completeness for SQL tables — completion curves, freshness and volume
checks, and a git-friendly status dashboard.

**Status: pre-release skeleton.** The package modules are placeholders; metrics land
step by step behind a test-first workflow (see `PLAN.md`).

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

## License

MIT
