"""Verify every parameter_definition in the canonical DB has a parameter_group.

Templates exported via :mod:`flextool.export_to_tabular` can be filtered by
parameter group.  Any parameter without a group assignment would silently
disappear from a filtered export, so this test asserts full coverage.

The test is allowed to fail until every orphan is tagged.
"""

from __future__ import annotations

import pytest
from spinedb_api import DatabaseMapping


def test_every_parameter_has_a_group(examples_db_url: str) -> None:
    # ``examples_db_url`` (tests/conftest.py) is a per-worker ISOLATED copy
    # of ``templates/examples.sqlite`` — never the shared checked-in file —
    # so concurrent xdist workers cannot deadlock on a sqlite lock (see the
    # fixture docstring and CLAUDE.md invariant #3).
    with DatabaseMapping(examples_db_url, create=False) as db:
        db.fetch_all()
        orphans: list[tuple[str, str]] = []
        for pdef in db.get_parameter_definition_items():
            ext = pdef.extended()
            group = ext.get("parameter_group_name")
            if not group:
                orphans.append((ext["entity_class_name"], pdef["name"]))

    if orphans:
        lines = "\n".join(f"  {cls}.{p}" for cls, p in sorted(orphans))
        pytest.fail(
            f"{len(orphans)} parameter(s) have no parameter_group_name:\n{lines}"
        )
