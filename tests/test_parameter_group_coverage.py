"""Verify every parameter_definition in the canonical DB has a parameter_group.

Templates exported via :mod:`flextool.export_to_tabular` can be filtered by
parameter group.  Any parameter without a group assignment would silently
disappear from a filtered export, so this test asserts full coverage.

The test is allowed to fail until every orphan is tagged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping

FLEXTOOL_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DB = FLEXTOOL_ROOT / "templates" / "examples.sqlite"
EXAMPLE_DB_URL = f"sqlite:///{EXAMPLE_DB}"


def test_every_parameter_has_a_group() -> None:
    assert EXAMPLE_DB.exists(), f"Example DB not found: {EXAMPLE_DB}"

    db = DatabaseMapping(EXAMPLE_DB_URL, create=False)
    try:
        db.fetch_all()
        orphans: list[tuple[str, str]] = []
        for pdef in db.get_parameter_definition_items():
            ext = pdef.extended()
            group = ext.get("parameter_group_name")
            if not group:
                orphans.append((ext["entity_class_name"], pdef["name"]))
    finally:
        db.close()

    if orphans:
        lines = "\n".join(f"  {cls}.{p}" for cls, p in sorted(orphans))
        pytest.fail(
            f"{len(orphans)} parameter(s) have no parameter_group_name:\n{lines}"
        )
