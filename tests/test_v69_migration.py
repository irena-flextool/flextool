"""Tests for the v69 database migration.

v69 backfills ``parameter_group_name`` on two parameter definitions that
earlier migrations added without a group:

* ``model.small_number_threshold`` (added in v59) -> ``model``.
* ``node.penalty_method`` (added in v67) -> ``basics``.

Every parameter must belong to a ``parameter_group`` so it is not dropped
from group-filtered tabular exports; this is guarded independently by
``tests/test_parameter_group_coverage.py``.  These tests assert the two
params carry their group after a full migration.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v69_version_constant_is_at_least_69() -> None:
    """The engine must report a schema version >= 69 — the parameter-group
    backfill lower bound.  Later migrations keep raising the constant, so an
    exact-equality assertion would regress as the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 69


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def test_small_number_threshold_in_model_group(tmp_path: Path) -> None:
    """``model.small_number_threshold`` is attached to the ``model`` group."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="model",
            name="small_number_threshold",
        )
        assert pdef, "small_number_threshold missing after migration"
        assert pdef["parameter_group_name"] == "model"
    finally:
        db.close()


def test_penalty_method_in_basics_group(tmp_path: Path) -> None:
    """``node.penalty_method`` is attached to the ``basics`` group."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="node",
            name="penalty_method",
        )
        assert pdef, "penalty_method missing after migration"
        assert pdef["parameter_group_name"] == "basics"
    finally:
        db.close()


def test_no_ungrouped_parameters_after_migration(tmp_path: Path) -> None:
    """After migration no parameter_definition is left without a group —
    the invariant that v69 restores."""
    db = _migrated_db(tmp_path)
    try:
        orphans = [
            (p["entity_class_name"], p["name"])
            for p in db.get_parameter_definition_items()
            if not p["parameter_group_name"]
        ]
        assert not orphans, f"ungrouped parameters remain: {sorted(orphans)}"
    finally:
        db.close()
