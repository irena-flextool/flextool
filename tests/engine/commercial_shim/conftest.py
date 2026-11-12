"""Item 230 checked against the commercial solvers' Python packages: opt-in, run with
``pytest -m commercial_shim tests/engine/commercial_shim -n 0``.

The tests put shims named like the solvers' command-line programs (``gurobi_cl``, ``cplex``, ``optimizer``,
``copt_cmd``; see ``shims/``) on ``PATH`` and run FlexTool's real commercial solve path through them.  They
need ``gurobipy`` / ``cplex`` / ``xpress`` / ``coptpy`` with a working (size-limited is enough) licence; a
solver whose package or licence is missing is skipped.

Every test collected under this folder gets the ``commercial_shim`` marker.  The default ``addopts``
deselects it; a command-line ``-m`` replaces the addopts one (CI's ``-m smoke`` / ``-m slow``), so the tests
are also skipped unless the ``-m`` expression names ``commercial_shim``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_NAMES_MARKER = re.compile(r"\bcommercial_shim\b")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark this folder's tests ``commercial_shim``; skip them unless ``-m`` names ``commercial_shim``."""
    named = bool(_NAMES_MARKER.search(config.getoption("markexpr", "") or ""))
    skip = pytest.mark.skip(
        reason="commercial-solver shim tests: run with -m commercial_shim (tests/engine/commercial_shim/)"
    )
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        if _HERE not in path.resolve().parents:
            continue
        item.add_marker(pytest.mark.commercial_shim)
        if not named:
            item.add_marker(skip)
