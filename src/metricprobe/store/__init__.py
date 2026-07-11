"""Snapshot writers: parquet/duckdb (default) and mssql schema (config-flagged),
sharing one interface. Runs are transactional: staging then atomic manifest
commit; readers only ever see manifest-committed runs.
"""
