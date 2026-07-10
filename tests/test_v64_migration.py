"""Tests for the v64 database migration.

v64 adds the per-solve ``solve.scaling`` parameter (the numerical
LP-autoscaler mode), promoting the previously run-time-only ``--scaling``
/ ``FLEXTOOL_SCALING`` control to a proper DB parameter.  It is bound to
the new ``scaling_modes`` value list (off / solver_only / basic / full)
and defaults to ``full`` (unchanged behaviour).  ``basic`` (or lower)
lets a model pin a mode that solves it correctly when Layer 2 mis-solves.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v64_version_constant_is_at_least_64() -> None:
    """The engine must report a schema version >= 64 — the scaling knob
    lower bound.  Later migrations keep raising the constant, so an exact
    equality assertion would regress every time the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 64


def test_migration_adds_scaling(tmp_path: Path) -> None:
    """Migrating a fixture DB yields a ``solve.scaling`` parameter
    definition defaulting to ``full``, bound to ``scaling_modes`` and
    grouped under ``solve_advanced``.
    """
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="scaling",
        )
        assert pdef, "scaling not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "full"
        ext = pdef.extended()
        assert ext.get("parameter_value_list_name") == "scaling_modes"
        assert ext.get("parameter_group_name") == "solve_advanced"
    finally:
        db.close()


def test_migration_adds_scaling_modes_value_list(tmp_path: Path) -> None:
    """The ``scaling_modes`` value list carries exactly the four CLI modes."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"] == "scaling_modes"
        }
        assert values == {"off", "solver_only", "basic", "full"}
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-running the migration over an already-current DB is a no-op:
    the definition stays single and at ``full``.
    """
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        matches = [
            p
            for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == "solve" and p["name"] == "scaling"
        ]
        assert len(matches) == 1
        default = from_database(
            matches[0]["default_value"], matches[0]["default_type"]
        )
        assert default == "full"
    finally:
        db.close()
