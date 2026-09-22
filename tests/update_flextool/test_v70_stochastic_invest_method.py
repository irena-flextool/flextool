"""Tests for the v70 database migration (Slice C).

v70 adds ``solve.stochastic_invest_method`` — the opt-in enum selecting
how stochastic branches participate in investment:

1. Creates the value list ``stochastic_invest_methods`` == {none, recourse}.
2. Adds the ``solve.stochastic_invest_method`` definition (default
   ``"none"``) bound to that value list, grouped under ``solve_advanced``.

``'none'`` (default) is byte-identical to prior behaviour; ``'recourse'``
is hard-rejected by the Slice C guards until Slice D lands.  The
engine-side read/resolver + guards are covered by
``tests/engine_polars/test_recourse_guards.py``.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, from_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.sync_master_json_template import (
    sync_master_template,
)

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v70_version_constant_is_at_least_70() -> None:
    """The engine must report a schema version >= 70 — the
    stochastic_invest_method lower bound.  Later migrations keep raising
    the constant, so an exact-equality assertion would regress.
    """
    assert FLEXTOOL_DB_VERSION >= 70


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """Build a fixture DB from JSON (at the fixture's old version, which
    predates v70 so the parameter can ONLY appear via the v70 block),
    migrate it to HEAD, and return an open mapping."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    db.fetch_all()
    return db


def _db_version(url: str) -> int:
    """Read the migrated ``model.version`` parameter-definition default."""
    with DatabaseMapping(url, create=False) as db:
        sq = db.object_parameter_definition_sq
        row = (
            db.query(sq)
            .filter(sq.c.object_class_name == "model")
            .filter(sq.c.parameter_name == "version")
            .one_or_none()
        )
        assert row is not None, "model.version parameter definition missing"
        return int(from_database(row.default_value, row.default_type))


# --- M1 / M3: the migration adds the definition + value list ------------


def test_stochastic_invest_method_definition_present(tmp_path: Path) -> None:
    """``solve.stochastic_invest_method`` exists, defaults to ``"none"``,
    is bound to ``stochastic_invest_methods`` and grouped ``solve_advanced``."""
    db = _migrated_db(tmp_path)
    try:
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="stochastic_invest_method",
        )
        assert pdef, "stochastic_invest_method not added by migration"
        default = from_database(pdef["default_value"], pdef["default_type"])
        assert default == "none"
        assert pdef["parameter_value_list_name"] == "stochastic_invest_methods"
        assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_stochastic_invest_methods_value_list(tmp_path: Path) -> None:
    """``stochastic_invest_methods`` == {none, recourse}."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            from_database(v["value"], v["type"])
            for v in db.get_list_value_items()
            if v["parameter_value_list_name"] == "stochastic_invest_methods"
        }
        assert values == {"none", "recourse"}
    finally:
        db.close()


def test_migration_reaches_v70_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v70 fixture migrates to exactly v70 (>= 70), and re-running
    ``migrate_database`` is a no-op that keeps the parameter present —
    the ``add_value_list_manual`` / ``add_update_item`` helpers tolerate
    re-adds."""
    db_path = tmp_path / "lh2.sqlite"
    url = json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)
    start_version = _db_version(url)
    assert start_version < 70, (
        "fixture must start below v70 so the v70 block is exercised"
    )

    migrate_database(url)
    assert _db_version(url) == FLEXTOOL_DB_VERSION >= 70

    # Idempotent re-run — no crash, parameter still present.
    migrate_database(url)
    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        pdef = db.get_parameter_definition_item(
            entity_class_name="solve",
            name="stochastic_invest_method",
        )
        assert pdef, "parameter lost on idempotent re-migration"
        assert (
            from_database(pdef["default_value"], pdef["default_type"]) == "none"
        )


# --- M2: the committed master template is up to date --------------------


def test_master_template_up_to_date() -> None:
    """``sync_master_template(verify_only=True)`` returns True — the
    spinedb_schema.json regen (with the v70 parameter) was committed.
    This is the same assertion CI makes; pinning it locally catches a
    forgotten regen."""
    assert sync_master_template(verify_only=True) is True
