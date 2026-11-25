"""Tests for the v75 database migration — a single folded bump that:

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



_OLD_TO_NEW = {
    "non_anticipativity_periods": "shared_operation_periods",
    "non_anticipativity_invest_periods": "shared_invest_periods",
}


def test_v75_version_constant_is_at_least_75() -> None:
    """The engine must report a schema version >= 75 — the rename +
    reserve_duration lower bound."""
    assert FLEXTOOL_DB_VERSION >= 75


def _migrated_db(tmp_path: Path) -> DatabaseMapping:
    """The minimal v74 database (:func:`_build_v74_db`: the old names, ``non_anticipativity_invest_periods``
    [p1] on ``s1``) migrated to 83, the last data version that has the renamed parameters (v84 replaces them by
    ``invest_solve_branching_method``: ``test_v84_branching``), as an open mapping.  The fixture corpus is at the
    head, so it cannot start below v75."""
    db_path = tmp_path / "v74.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v74_db(url, reserve=False)
    migrate_database(url, up_to=83)
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


# --- rename: old names gone, new names present (through 83) --------------


def test_na_params_renamed_through_83(tmp_path: Path) -> None:
    """Up to 83 the two window params carry the positive names (unset Arrays: no default, no value list); the
    old non_anticipativity_* names are gone."""
    db = _migrated_db(tmp_path)
    try:
        for old, new in _OLD_TO_NEW.items():
            assert not db.get_parameter_definition_item(
                entity_class_name="solve", name=old,
            ), f"{old} should have been renamed away at v75"
            pdef = db.get_parameter_definition_item(
                entity_class_name="solve", name=new,
            )
            assert pdef, f"{new} not present at 83"
            # No default (the seed's definition stores it as a JSON null).
            assert from_database(pdef["default_value"], pdef["default_type"]) is None
            assert not pdef["parameter_value_list_name"]
    finally:
        db.close()


def test_rename_carries_values(tmp_path: Path) -> None:
    """The seed's ``non_anticipativity_invest_periods`` [p1] on ``s1`` is kept under the new name through 83
    (values follow the definition id); no value under an old name survives."""
    db = _migrated_db(tmp_path)
    try:
        values = {
            v["parameter_definition_name"]: from_database(v["value"], v["type"])
            for v in db.get_parameter_value_items()
            if v["entity_class_name"] == "solve"
        }
        assert list(values["shared_invest_periods"].values) == ["p1"]
        for old in _OLD_TO_NEW:
            assert old not in values, f"stale {old} value survived the rename"
    finally:
        db.close()


# --- reserve_duration added on the reserve relationship class -----------


def _build_v74_db(url: str, reserve: bool = True) -> None:
    """Create a minimal v74 DB carrying the OLD-named window params (with a
    value) on ``solve`` and (with *reserve*) a ``reserve__upDown__group`` class WITHOUT
    reserve_duration — so the v75 block is the ONLY path that renames /
    adds.  ``model.version`` default pinned to 74 so
    ``migrate_database(..., up_to=75)`` runs only the v75 step.  Without *reserve* the later steps up to 83 run
    on it too (they need more of the schema for the reserve classes)."""
    reserve_classes = [
        ["reserve", ()],
        ["upDown", ()],
        ["group", ()],
        ["reserve__upDown__group", ("reserve", "upDown", "group")],
    ] if reserve else []
    with DatabaseMapping(url, create=True) as db:
        _count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["solve", ()],
                *reserve_classes,
            ],
            parameter_definitions=[
                ["model", "version", 74.0, None, "Database version."],
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
        db.commit_session("Seed v74 DB (old window names + reserve class)")


def test_migration_renames_adds_and_is_idempotent(tmp_path: Path) -> None:
    """A pre-v75 DB migrates to exactly v75: the window params are renamed
    (carrying their values), reserve_duration is added, and re-running the
    v75 step is a no-op."""
    db_path = tmp_path / "v74_base.sqlite"
    url = f"sqlite:///{db_path.resolve()}"
    _build_v74_db(url)
    start_version = _db_version(url)
    assert start_version < 75, (
        "seed DB must start below v75 so the v75 block is exercised"
    )

    migrate_database(url, up_to=75)
    assert _db_version(url) == 75

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
        assert rd, "reserve_duration not added by v75"
        assert rd["default_value"] is None
        assert rd["default_type"] is None
        assert not rd["parameter_value_list_name"]

    with DatabaseMapping(url, create=False) as db:
        db.fetch_all()
        _assert_state(db)

    # Idempotent re-run of the v75 step.
    migrate_database(url, up_to=75)
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
