"""Tests for the v67 database migration.

v67 adds ``node.penalty_method`` — a per-node toggle for the balance
penalty (unserved-energy / over-supply slack) variables:

1. Creates the value list ``penalty_methods`` == {regular, off}.
2. Adds the ``node.penalty_method`` parameter definition (default
   ``"regular"``) bound to that value list.

``'off'`` removes the node's ``vq_state_up`` / ``vq_state_down`` slack
variables at model-build time; the engine-side gating is covered by
``tests/engine_polars/test_penalty_method_off.py``.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v67_version_constant_is_at_least_67() -> None:
    """The engine must report a schema version >= 67 — the penalty_method
    lower bound.  Later migrations keep raising the constant, so an exact
    equality assertion would regress as the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 67


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def test_penalty_method_definition_present(tmp_path: Path) -> None:
    """``node.penalty_method`` exists, defaults to ``"regular"`` and is bound
    to the ``penalty_methods`` value list."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="node",
            name="penalty_method",
        )
        assert pdef, "penalty_method not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "regular"
        assert pdef["parameter_value_list_name"] == "penalty_methods"
    finally:
        db.close()


def test_penalty_methods_value_list(tmp_path: Path) -> None:
    """``penalty_methods`` == {regular, off}."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"] == "penalty_methods"
        }
        assert values == {"regular", "off"}
    finally:
        db.close()
