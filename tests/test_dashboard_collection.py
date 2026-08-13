"""Dashboard tests must collect without the observability extra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_test_dashboards_imports_without_prometheus_client() -> None:
    """``pytest -m integration`` collects every module; that job has no extra."""
    repo = Path(__file__).resolve().parents[1]
    script = r"""
import builtins
import importlib.util
from pathlib import Path

real_import = builtins.__import__


def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "prometheus_client" or name.startswith("prometheus_client."):
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _blocked
path = Path("tests/test_dashboards.py")
spec = importlib.util.spec_from_file_location("test_dashboards", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
