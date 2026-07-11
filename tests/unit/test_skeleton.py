"""Step 0 sanity: the package skeleton imports and is documented."""

import importlib

import metricprobe

MODULES = [
    "metricprobe",
    "metricprobe.config",
    "metricprobe.discover",
    "metricprobe.report",
    "metricprobe.publish",
    "metricprobe.app",
    "metricprobe.cli",
    "metricprobe.extract",
    "metricprobe.metrics",
    "metricprobe.metrics.volume",
    "metricprobe.metrics.completion",
    "metricprobe.metrics.dual_lag",
    "metricprobe.metrics.batch",
    "metricprobe.metrics.parity",
    "metricprobe.store",
    "metricprobe.viz",
]


def test_version_present():
    assert metricprobe.__version__


def test_all_modules_import_and_have_docstrings():
    undocumented = []
    for name in MODULES:
        module = importlib.import_module(name)
        if not (module.__doc__ or "").strip():
            undocumented.append(name)
    assert not undocumented, f"modules missing docstrings: {undocumented}"
