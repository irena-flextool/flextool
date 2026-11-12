"""What the append-only ``_augment_*.py`` scripts add to ``tests/fixtures/tests.json``.

Builders and tests that start from ``tests.json`` but must keep its earlier content
(``regen_lh2_three_region.py``, the assistant many-runs project) leave these out.  A new
augment script exports ``ADDED_ALTERNATIVES`` / ``ADDED_ENTITIES`` / ``ADDED_SCENARIOS``
and is listed in ``_AUGMENT_SCRIPTS``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent

_AUGMENT_SCRIPTS = ("_augment_phase3_examples", "_augment_wind_weighted")


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULES = tuple(_load(name) for name in _AUGMENT_SCRIPTS)

ADDED_ALTERNATIVES: tuple[str, ...] = tuple(a for m in _MODULES for a in m.ADDED_ALTERNATIVES)
ADDED_ENTITIES: tuple[tuple[str, tuple[str, ...]], ...] = tuple(e for m in _MODULES for e in m.ADDED_ENTITIES)
ADDED_SCENARIOS: tuple[str, ...] = tuple(s for m in _MODULES for s in m.ADDED_SCENARIOS)
