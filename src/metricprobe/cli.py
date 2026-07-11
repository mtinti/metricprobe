"""Command-line interface: discover | run | report | publish | serve.

Lands in Step 6 with the lifecycle orchestration. Exit codes separate code
failure from data failure: 0 = ran with no RED, 2 = ran with at least one
data-health RED (outputs committed first), 1 = execution error (nothing
partial becomes visible).
"""
