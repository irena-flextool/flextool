"""Axis-Enum mode parity run: opt-in, run with ``pytest -m axis_enums tests/engine/axis_enums -n 8``.

Axis-Enum mode (``FLEXTOOL_AXIS_ENUMS=1``) is experimental and off in production.  The tests in this folder
build the LP of many lp_snapshot fixtures twice, plain and in Enum mode, and require byte-identical MPS files;
they take minutes, so they are kept out of the normal suite.  The gating subset (one fixture per fixed site
family) is ``tests/engine/test_axis_enum_lp_parity_sites.py``.

Every test collected under this folder gets the ``axis_enums`` marker.  The default ``addopts``
(``-m "not csv_reload and not axis_enums"``) deselects them.  A command-line ``-m`` replaces the addopts one
(CI's ``-m smoke`` / ``-m slow``), so they are also skipped unless the ``-m`` expression names ``axis_enums``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_NAMES_AXIS_ENUMS = re.compile(r"\baxis_enums\b")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark this folder's tests ``axis_enums``; skip them unless ``-m`` names ``axis_enums``.

    ``tryfirst`` so the marker is on the items before ``-m`` deselects.
    """
    named = bool(_NAMES_AXIS_ENUMS.search(config.getoption("markexpr", "") or ""))
    skip = pytest.mark.skip(
        reason="axis-Enum mode parity run: run with -m axis_enums (tests/engine/axis_enums/)"
    )
    for item in items:
        path = Path(str(getattr(item, "path", "") or ""))
        if _HERE not in path.resolve().parents:
            continue
        item.add_marker(pytest.mark.axis_enums)
        if not named:
            item.add_marker(skip)
