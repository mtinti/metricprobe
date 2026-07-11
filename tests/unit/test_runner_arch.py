"""Tests for the equivalence-runner platform assertion (scripts/assert_runner_arch.py)."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "assert_runner_arch", REPO_ROOT / "scripts" / "assert_runner_arch.py"
)
arch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arch)


def test_linux_x86_64_accepted():
    assert arch.check("Linux", "x86_64") is None
    assert arch.check("Linux", "AMD64") is None  # case-insensitive


def test_non_x86_64_rejected():
    assert "x86-64" in arch.check("Linux", "aarch64")
    assert "x86-64" in arch.check("Linux", "arm64")


def test_non_linux_rejected():
    assert "Linux" in arch.check("Darwin", "x86_64")
    assert "Linux" in arch.check("Windows", "AMD64")
