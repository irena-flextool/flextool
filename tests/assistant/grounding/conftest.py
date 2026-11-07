"""The calibration tools (``tools/assistant_eval*.py``, not a package) for the grounding tests."""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"


def _tool(name: str) -> ModuleType:
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import importlib

    return importlib.import_module(name)


@pytest.fixture(scope="session")
def scoring() -> ModuleType:
    return _tool("assistant_eval_scoring")


@pytest.fixture(scope="session")
def evaluator() -> ModuleType:
    return _tool("assistant_eval")
