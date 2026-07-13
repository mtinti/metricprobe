"""metricprobe — data arrival latency & completeness probes for database tables.

Measures, per configured table: volume history, completion curves (lag from event
time to load time), dual lag, batch metrics, parity, and freshness. Only bounded
aggregate results ever leave the database.
"""

__version__ = "0.1.8"
