"""Tests for the v68 database migration.

v68 adds ``group.use_for_representative_periods`` — a per-group ``yes_no``
flag marking a region-group as an aggregation unit for net-load
representative-period selection:

1. Adds the ``group.use_for_representative_periods`` parameter definition
   (default ``"no"``) bound to the existing ``yes_no`` value list.
2. Attaches it to the ``solve_advanced`` parameter group.

The flag is consumed only by the representative-periods preprocessor
(``flextool.representative_periods.netload_inputs``); it is never read by
the LP model, so there is no engine-side gating to exercise here.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v68_version_constant_is_at_least_68() -> None:
    """The engine must report a schema version >= 68 — the
    use_for_representative_periods lower bound.  Later migrations keep
    raising the constant, so an exact equality assertion would regress as
    the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 68


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def test_rp_group_flag_definition_present(tmp_path: Path) -> None:
    """``group.use_for_representative_periods`` exists, defaults to ``"no"``
    and is bound to the ``yes_no`` value list."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="group",
            name="use_for_representative_periods",
        )
        assert pdef, "use_for_representative_periods not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "no"
        assert pdef["parameter_value_list_name"] == "yes_no"
    finally:
        db.close()


def test_rp_group_flag_in_solve_advanced_group(tmp_path: Path) -> None:
    """The new flag is attached to the ``solve_advanced`` parameter group."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="group",
            name="use_for_representative_periods",
        )
        assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_migrated_version_reaches_constant(tmp_path: Path) -> None:
    """After migration the DB's ``model.version`` equals FLEXTOOL_DB_VERSION."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="model",
            name="version",
        )
        version = from_database(pdef["default_value"], pdef["default_type"])
        assert float(version) == float(FLEXTOOL_DB_VERSION)
    finally:
        db.close()
