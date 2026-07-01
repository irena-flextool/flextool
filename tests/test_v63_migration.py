"""Tests for the v63 database migration.

v63 adds the per-solve ``solve.benders_in_out_weight`` parameter (the
Benders in-out separation stabilization weight ``lambda``), promoting the
previously machine-local env knob ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT`` to a
proper DB parameter.  Default ``0.0`` = OFF = exact Benders (byte-
identical); values in ``(0, 1)`` turn it on.

The migration mirrors the v62 "Add the Benders knobs" section: a single
``add_update_item`` upsert plus the ``solve_advanced`` group assignment.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v63_version_constant_is_at_least_63() -> None:
    """The engine must report a schema version >= 63 — the in-out-weight
    lower bound.  Later migrations keep raising the constant, so an exact
    equality assertion would regress every time the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 63


def test_migration_adds_benders_in_out_weight(tmp_path: Path) -> None:
    """Migrating a fixture DB to the current version yields a
    ``solve.benders_in_out_weight`` parameter definition defaulting to 0.0
    and grouped under ``solve_advanced``.
    """
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="benders_in_out_weight",
        )
        assert pdef, "benders_in_out_weight not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == 0.0
        assert pdef.extended().get("parameter_group_name") == "solve_advanced"
    finally:
        db.close()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    """Re-running the migration over an already-current DB is a no-op
    (``add_update_item`` upserts): the definition stays single and at 0.0.
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
            if p["entity_class_name"] == "solve"
            and p["name"] == "benders_in_out_weight"
        ]
        assert len(matches) == 1
        default = from_database(
            matches[0]["default_value"], matches[0]["default_type"]
        )
        assert default == 0.0
    finally:
        db.close()
