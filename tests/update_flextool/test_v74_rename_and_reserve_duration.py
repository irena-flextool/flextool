"""Tests for the v74 database migration — a single folded bump that:

1. **Renames** the two (unreleased) stochastic shared-decision window
   parameters on the ``solve`` class to positive, non-negated names:

   * ``non_anticipativity_periods``        -> ``shared_operation_periods``
   * ``non_anticipativity_invest_periods`` -> ``shared_invest_periods``

   The rename is collision-safe and carries each parameter's existing
   per-solve Array *values* by definition id, so a migrated DB keeps
   identical behaviour under the new names.

2. **Adds** ``reserve__upDown__group.reserve_duration`` for DBs that predate
   the parameter (``[h]`` scalar, default null, no value list; storage
   energy-adequacy coupling for reserves, issue #322).  Idempotent — a
   no-op when the definition is already present.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, from_database, import_data

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.sync_master_json_template import (
    sync_master_template,
)

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parents[1]
FIXTURES_DIR = TEST_DIR / "fixtures"

_OLD_TO_NEW = {
    "non_anticipativity_periods": "shared_operation_periods",
    "non_anticipativity_invest_periods": "shared_invest_periods",
}


def test_v74_version_constant_is_at_least_74() -> None:
    """The engine must report a schema version >= 74 — the rename +
    reserve_duration lower bound."""
    assert FLEXTOOL_DB_VERSION >= 74


def _migrated_db(tmp_path: Path, fixture: str) -> DatabaseMapping:
    """Build a fixture DB from JSON, migrate it to HEAD, return an open
    mapping."""
    db_path = tmp_path / "fx.sqlite"
    url = json_to_db(FIXTURES_DIR / fixture, db_path)
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


# --- rename: old names gone, new names present at HEAD -------------------


def test_na_params_renamed_at_head(tmp_path: Path) -> None:
    """At HEAD the two window params carry the positive names; the old
    non_anticipativity_* names are gone."""
    db = _migrated_db(tmp_path, "lh2_three_region.json")
    try:
        for old, new in _OLD_TO_NEW.items():
            assert not db.get_parameter_definition_item(
                entity_class_name="solve", name=old,
            ), f"{old} should have been renamed away at v74"
            pdef = db.get_parameter_definition_item(
                entity_class_name="solve", name=new,
            )
            assert pdef, f"{new} not present at HEAD"
            # Still an unset Array under solve_advanced.
            assert pdef["default_value"] is None
            assert pdef["default_type"] is None
            assert not pdef["parameter_value_list_name"]
            assert pdef["parameter_group_name"] == "solve_advanced"
    finally:
        db.close()


def test_rename_carries_values(tmp_path: Path) -> None:
    """A fixture that SETS the window params keeps those values under the
    new names after migrating to HEAD (values follow the definition id)."""
    db = _migrated_db(tmp_path, "stoch_two_period_hedge.json")
    try:
        names = {
            v["parameter_definition_name"]
            for v in db.get_parameter_value_items()
        }
        # The hedge fixture authors both window values on stoch_2p_hedge.
        assert "shared_invest_periods" in names, (
            "shared_invest_periods value lost in rename"
        )
        assert "shared_operation_periods" in names, (
            "shared_operation_periods value lost in rename"
        )
        for old in _OLD_TO_NEW:
            assert old not in names, f"stale {old} value survived the rename"
    finally:
        db.close()


# --- reserve_duration added on the reserve relationship class -----------


def _build_v73_db(url: str) -> None:
    """Create a minimal v73 DB carrying the OLD-named window params (with a
    value) on ``solve`` and a ``reserve__upDown__group`` class WITHOUT
    reserve_duration — so the v74 block is the ONLY path that renames /
    adds.  ``model.version`` default pinned to 73 so
    ``migrate_database(..., up_to=74)`` runs only the v74 step."""
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["solve", ()],
                ["reserve", ()],
                ["upDown", ()],
                ["group", ()],
                ["reserve__upDown__group", ("reserve", "upDown", "group")],
            ],
            parameter_definitions=[
                ["model", "version", 73.0, None, "Database version."],
                ["solve", "non_anticipativity_periods", None, None,
                 "Operational non-anticipativity window (pre-rename)."],
                ["solve", "non_anticipativity_invest_periods", None, None,
                 "Investment non-anticipativity window (pre-rename)."],
            ],
            alternatives=[["Base", ""]],
            entities=[
                ["solve", "s1"],
            ],
            parameter_values=[
                ["solve", "s1", "non_anticipativity_invest_periods",
                 Array(["p1"]), "Base"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v73 DB (old window names + reserve class)")


def test_migration_renames_adds_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v74 DB migrates to exactly v74: the window params are renamed
    (carrying their values), reserve_duration is added, and re-running the
    v74 step is a no-op."""
    db_path = tmp_path / "v73_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v73_db(url)
    start_version = _db_version(url)
    assert start_version < 74, (
        "seed DB must start below v74 so the v74 block is exercised"
    )

    migrate_database(url, up_to=74)
    assert _db_version(url) == 74

    def _assert_state(db: DatabaseMapping) -> None:
        # Rename: old gone, new present.
        for old, new in _OLD_TO_NEW.items():
            assert not db.get_parameter_definition_item(
                entity_class_name="solve", name=old,
            ), f"{old} not renamed"
            assert db.get_parameter_definition_item(
                entity_class_name="solve", name=new,
            ), f"{new} missing after rename"
        # Value carried to the new invest name.
        inv_vals = [
            v for v in db.get_parameter_value_items()
            if v["parameter_definition_name"] == "shared_invest_periods"
        ]
        assert len(inv_vals) == 1, "invest-window value not carried by rename"
        assert list(from_database(
            inv_vals[0]["value"], inv_vals[0]["type"]).values) == ["p1"]
        # reserve_duration added on the reserve relationship class.
        rd = db.get_parameter_definition_item(
            entity_class_name="reserve__upDown__group",
            name="reserve_duration",
        )
        assert rd, "reserve_duration not added by v74"
        assert rd["default_value"] is None
        assert rd["default_type"] is None
        assert not rd["parameter_value_list_name"]

    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        _assert_state(db)

    # Idempotent re-run of the v74 step.
    migrate_database(url, up_to=74)
    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        _assert_state(db)


# --- the committed master template is up to date ------------------------


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="sync_master_template runs migrate_database, which leaves a sqlite "
    "engine handle open; the TemporaryDirectory cleanup then hits WinError 32 "
    "on Windows. The same verification runs on the ubuntu-only template-check "
    "CI job (sync_master_json_template --verify).",
)
def test_master_template_up_to_date() -> None:
    """``sync_master_template(verify_only=True)`` returns True — the
    spinedb_schema.json regen (rename + reserve_duration) was committed."""
    assert sync_master_template(verify_only=True) is True
