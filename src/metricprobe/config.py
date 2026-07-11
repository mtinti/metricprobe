"""Typed configuration: ProbeConfig / TableConfig plus campaign, store and delivery
layers (pydantic v2), the YAML loader with env-var expansion, and validation.

The complete config schema is frozen and versioned in Step 2, before any metric
work begins. Unknown fields are rejected (typo safety).
"""
