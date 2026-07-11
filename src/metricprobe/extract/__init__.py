"""SQL builders (SQLAlchemy Core), dialect-aware: mssql + duckdb.

One canonical aggregation pass per table under an explicit scan budget; sargable
predicates; never raw rows into pandas; no hardcoded database targeting.
"""
