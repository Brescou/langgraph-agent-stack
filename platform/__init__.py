"""
platform — Platform kernel for the LangGraph agent stack.

Importing this package registers all built-in domain packs into PackRegistry.
Additional packs can be registered by calling PackRegistry.register() before
the FastAPI lifespan starts.

Implementation note — stdlib 'platform' shadowing
--------------------------------------------------
This package is named 'platform', which shadows the Python stdlib module of the
same name.  We resolve this by pre-importing 'uuid' (and any other stdlib module
that calls platform.system() at import time) while temporarily swapping the real
stdlib 'platform' module back into sys.modules.  After those imports are done we
restore our local package so downstream code can continue to do
'from platform.base_pack import BaseDomainPack' etc.
"""

import importlib.util
import sys

# ---------------------------------------------------------------------------
# Bootstrap: load the stdlib platform module by finding it on the stdlib path,
# bypassing our local package.  We do this before importing domain packs so
# that transitive stdlib imports (uuid -> platform.system) work correctly.
# ---------------------------------------------------------------------------

_THIS_PKG = sys.modules.get("platform")  # our partially-initialised local pkg

# Find the real stdlib platform.py by searching non-project paths
_stdlib_spec = None
_project_root = __file__.rsplit("/platform/", 1)[
    0
]  # e.g. /home/.../langgraph-agent-stack
for _path_entry in sys.path:
    if _path_entry == _project_root or _path_entry == "":
        continue
    import os as _os

    _candidate = _os.path.join(_path_entry, "platform.py")
    if _os.path.isfile(_candidate):
        _stdlib_spec = importlib.util.spec_from_file_location("platform", _candidate)
        break

if _stdlib_spec is not None and _stdlib_spec.loader is not None:
    _stdlib_platform_mod = importlib.util.module_from_spec(_stdlib_spec)
    # Temporarily replace our package with the real stdlib module so that
    # any import triggered below (e.g. uuid) can call platform.system() safely.
    sys.modules["platform"] = _stdlib_platform_mod
    _stdlib_spec.loader.exec_module(_stdlib_platform_mod)  # type: ignore[attr-defined]

    # Pre-import modules that call platform.system() at import time.
    import uuid as _uuid  # noqa: F401 — ensures uuid is cached in sys.modules

    # Restore our local package as the canonical 'platform' module.
    if _THIS_PKG is not None:
        sys.modules["platform"] = _THIS_PKG
        # Re-export stdlib ``platform`` API so third-party code (SQLAlchemy,
        # testcontainers, ``from platform import system``, etc.) keeps working
        # after this package replaces ``sys.modules['platform']``.
        for _name in dir(_stdlib_platform_mod):
            if _name.startswith("_"):
                continue
            setattr(_THIS_PKG, _name, getattr(_stdlib_platform_mod, _name))

from platform.base_pack import BaseDomainPack  # noqa: E402
from platform.registry import PackRegistry  # noqa: E402

__all__ = ["BaseDomainPack", "PackRegistry"]

# Register built-in packs explicitly — no magic, no auto-discovery.
from domain_packs.analysis_only.pack import AnalysisOnlyPack  # noqa: E402
from domain_packs.research_analysis.pack import ResearchAnalysisPack  # noqa: E402
from domain_packs.research_only.pack import ResearchOnlyPack  # noqa: E402
from domain_packs.summariser.pack import SummariserPack  # noqa: E402

PackRegistry.register(ResearchAnalysisPack)
PackRegistry.register(ResearchOnlyPack)
PackRegistry.register(SummariserPack)
PackRegistry.register(AnalysisOnlyPack)
