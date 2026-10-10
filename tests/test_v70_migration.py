"""Tests for the v70 database migration.

Besides description updates, v70 renames the input-side capacity
coefficients on ``unit__inputNode``:

* ``capacity_max_coeff`` -> ``input_share_max`` (values move unchanged);
* ``capacity_min_coeff`` -> ``input_share_min``, default 0, and every old
  value is deleted (the new meaning — a mixing minimum — is unrelated).

``unit__outputNode.capacity_min_coeff`` becomes an optional per-output
floor (the minimum load is per unit, on the sum of outputs): default 0,
values outside [0, 1] and v36 backfill leftovers (``capacity_min_coeff ==
capacity_max_coeff != 1``) are deleted and listed; other values stay.

``reserve__upDown__group.penalty_reserve`` gets back its default 5000
(cleared by v56); authored values are untouched.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping, Map, from_database, import_data, to_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import (
    _V70_CAPACITY_MAX_COEFF_DESCRIPTION,
    _V70_CAPACITY_MIN_COEFF_DESCRIPTION,
    _V70_INPUT_SHARE_MIN_DESCRIPTION,
    _V70_PENALTY_RESERVE_DESCRIPTION,
    _migrate_v70_capacity_coefficient_descriptions,
    migrate_database,
)

from tests.db_utils import json_to_db

TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TEST_DIR / "fixtures"


def test_v70_version_constant_is_at_least_70() -> None:
    assert FLEXTOOL_DB_VERSION >= 70


def _build_v69_db(url: str) -> None:
    """Minimal v69 DB with input- and output-side capacity coefficients.

    Built from scratch (never a checked-in .sqlite).  ``model.version``
    default is pinned to 69 so ``migrate_database`` runs from the v70
    step.
    """
    with DatabaseMapping(url, create=True) as db:
        count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()],
                ["node", ()],
                ["unit", ()],
                ["unit__inputNode", ("unit", "node")],
                ["unit__outputNode", ("unit", "node")],
            ],
            parameter_definitions=[
                ["model", "version", 69.0, None, "Database version."],
                ["unit__inputNode", "capacity_max_coeff", 1.0, None,
                 "v69 input max coefficient."],
                ["unit__inputNode", "capacity_min_coeff", 1.0, None,
                 "v69 input min coefficient."],
                ["unit__outputNode", "capacity_max_coeff", 1.0, None,
                 "v69 output max coefficient."],
                ["unit__outputNode", "capacity_min_coeff", 1.0, None,
                 "v69 output min coefficient."],
            ],
            alternatives=[["Base", ""], ["alt", ""]],
            entities=[
                ["node", "fuel"],
                ["node", "elec"],
                ["unit", "plant"],
                ["unit__inputNode", ("plant", "fuel")],
                ["unit__outputNode", ("plant", "elec")],
            ],
            parameter_values=[
                ["unit__inputNode", ("plant", "fuel"), "capacity_max_coeff",
                 0.6, "alt"],
                ["unit__inputNode", ("plant", "fuel"), "capacity_min_coeff",
                 0.3, "alt"],
                ["unit__outputNode", ("plant", "elec"), "capacity_max_coeff",
                 0.8, "alt"],
                ["unit__outputNode", ("plant", "elec"), "capacity_min_coeff",
                 0.5, "Base"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.commit_session("Seed v69 DB with capacity coefficients")


def _values(db: DatabaseMapping, cls: str) -> dict:
    return {
        (pv["parameter_definition_name"], pv["entity_byname"],
         pv["alternative_name"]): from_database(pv["value"], pv["type"])
        for pv in db.get_parameter_value_items()
        if pv["entity_class_name"] == cls
    }


def _definitions(db: DatabaseMapping, cls: str) -> set[str]:
    return {p["name"] for p in db.get_parameter_definition_items()
            if p["entity_class_name"] == cls}


def test_input_values_move_to_input_share(tmp_path: Path, capsys) -> None:
    """``input_share_max`` keeps the old input max values; the old input
    min values are deleted (listed on stdout) and ``input_share_min``
    defaults to 0.  In-range output ``capacity_min_coeff`` values stay;
    the output default becomes 0."""
    url = f"sqlite:///{(tmp_path / 'v69.sqlite').resolve()}"
    _build_v69_db(url)
    migrate_database(url)
    printed = capsys.readouterr().out
    assert "plant/fuel [alt] = 0.3" in printed

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        in_defs = _definitions(db, "unit__inputNode")
        assert {"input_share_max", "input_share_min"} <= in_defs
        assert "capacity_max_coeff" not in in_defs
        assert "capacity_min_coeff" not in in_defs
        assert {"capacity_max_coeff", "capacity_min_coeff"} <= _definitions(
            db, "unit__outputNode")

        assert _values(db, "unit__inputNode") == {
            ("input_share_max", ("plant", "fuel"), "alt"): 0.6,
        }
        assert _values(db, "unit__outputNode") == {
            ("capacity_max_coeff", ("plant", "elec"), "alt"): 0.8,
            ("capacity_min_coeff", ("plant", "elec"), "Base"): 0.5,
        }

        share_max = db.get_parameter_definition_item(
            entity_class_name="unit__inputNode", name="input_share_max")
        assert from_database(share_max["default_value"],
                             share_max["default_type"]) == 1.0
        assert "full-load" in share_max["description"]
        share_min = db.get_parameter_definition_item(
            entity_class_name="unit__inputNode", name="input_share_min")
        assert from_database(share_min["default_value"],
                             share_min["default_type"]) == 0.0
        assert share_min["description"] == _V70_INPUT_SHARE_MIN_DESCRIPTION
        for d in (share_max, share_min):
            assert "fuel" not in d["description"].lower()
        out_max = db.get_parameter_definition_item(
            entity_class_name="unit__outputNode", name="capacity_max_coeff")
        assert out_max["description"] == _V70_CAPACITY_MAX_COEFF_DESCRIPTION
        out_min = db.get_parameter_definition_item(
            entity_class_name="unit__outputNode", name="capacity_min_coeff")
        assert out_min["description"] == _V70_CAPACITY_MIN_COEFF_DESCRIPTION
        assert from_database(out_min["default_value"],
                             out_min["default_type"]) == 0.0
    finally:
        db.close()


def _build_v69_output_db(url: str) -> None:
    """v69 DB with output ``capacity_min_coeff`` values of every kind."""
    bad_map = Map(["t1", "t2"], [0.5, 1.5], index_name="time")
    bad_map_val, bad_map_type = to_database(bad_map)
    with DatabaseMapping(url, create=True) as db:
        count, errors = import_data(
            db,
            entity_classes=[
                ["model", ()], ["node", ()], ["unit", ()],
                ["unit__inputNode", ("unit", "node")],
                ["unit__outputNode", ("unit", "node")],
            ],
            parameter_definitions=[
                ["model", "version", 69.0, None, "Database version."],
                ["unit__inputNode", "capacity_max_coeff", 1.0, None, ""],
                ["unit__inputNode", "capacity_min_coeff", 1.0, None, ""],
                ["unit__outputNode", "capacity_max_coeff", 1.0, None, ""],
                ["unit__outputNode", "capacity_min_coeff", 1.0, None, ""],
            ],
            alternatives=[["Base", ""], ["alt", ""]],
            entities=[
                ["node", "fuel"], ["node", "elec"], ["node", "heat"],
                ["node", "steam"],
                ["unit", "premium"], ["unit", "chp"], ["unit", "rivend"],
                ["unit", "keep"], ["unit", "mapped"],
                ["unit__outputNode", ("premium", "elec")],
                ["unit__outputNode", ("chp", "elec")],
                ["unit__outputNode", ("chp", "heat")],
                ["unit__outputNode", ("rivend", "elec")],
                ["unit__outputNode", ("rivend", "heat")],
                ["unit__outputNode", ("keep", "elec")],
                ["unit__outputNode", ("keep", "steam")],
                ["unit__outputNode", ("mapped", "elec")],
            ],
            parameter_values=[
                # v36-style out-of-range backfills -> deleted
                ["unit__outputNode", ("premium", "elec"),
                 "capacity_min_coeff", 2.0, "Base"],
                ["unit__outputNode", ("rivend", "heat"),
                 "capacity_min_coeff", -0.025, "Base"],
                # in-range v36 leftover (min == max != 1) -> deleted
                ["unit__outputNode", ("chp", "heat"),
                 "capacity_min_coeff", 0.5, "Base"],
                ["unit__outputNode", ("chp", "heat"),
                 "capacity_max_coeff", 0.5, "Base"],
                # authored in-range values -> kept
                ["unit__outputNode", ("chp", "elec"),
                 "capacity_min_coeff", 1.0, "Base"],
                ["unit__outputNode", ("rivend", "elec"),
                 "capacity_min_coeff", 0.0, "Base"],
                ["unit__outputNode", ("keep", "elec"),
                 "capacity_min_coeff", 0.4, "alt"],
                ["unit__outputNode", ("keep", "elec"),
                 "capacity_max_coeff", 0.4, "Base"],
            ],
        )
        assert not errors, f"seed import errors: {errors[:5]}"
        db.add_item("parameter_value", entity_class_name="unit__outputNode",
                    entity_byname=("mapped", "elec"),
                    parameter_definition_name="capacity_min_coeff",
                    alternative_name="Base", value=bad_map_val,
                    type=bad_map_type)
        db.commit_session("Seed v69 DB with output min coefficients")


def test_output_min_coeff_cleanup(tmp_path: Path, capsys) -> None:
    """Out-of-range output ``capacity_min_coeff`` values (incl. a map with
    one bad element) and v36 leftovers are deleted and listed; in-range
    authored values stay; units whose values change meaning are listed."""
    url = f"sqlite:///{(tmp_path / 'v69out.sqlite').resolve()}"
    _build_v69_output_db(url)
    migrate_database(url)
    printed = capsys.readouterr().out
    for frag in ("premium/elec [Base] = 2.0", "rivend/heat [Base] = -0.025",
                 "mapped/elec [Base]", "chp/heat [Base] = 0.5"):
        assert frag in printed, (frag, printed)
    # notices: rivend's only kept value is 0 (min load "switched off");
    # chp and keep carry > 0 on a multi-output unit.
    assert "lower min_load instead" in printed
    low = printed.split("lower min_load instead")[1].splitlines()[0]
    assert "rivend" in low and "keep" in low and "chp" not in low
    floor = printed.split("separate floor on that output")[1]
    assert "chp" in floor and "keep" in floor

    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        assert _values(db, "unit__outputNode") == {
            ("capacity_min_coeff", ("chp", "elec"), "Base"): 1.0,
            ("capacity_max_coeff", ("chp", "heat"), "Base"): 0.5,
            ("capacity_min_coeff", ("rivend", "elec"), "Base"): 0.0,
            ("capacity_min_coeff", ("keep", "elec"), "alt"): 0.4,
            ("capacity_max_coeff", ("keep", "elec"), "Base"): 0.4,
        }
    finally:
        db.close()


def test_step_is_idempotent(tmp_path: Path, capsys) -> None:
    """Re-running the step on a migrated DB changes and prints nothing."""
    url = f"sqlite:///{(tmp_path / 'v69out.sqlite').resolve()}"
    _build_v69_output_db(url)
    migrate_database(url)
    capsys.readouterr()
    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        before = _values(db, "unit__outputNode")
        _migrate_v70_capacity_coefficient_descriptions(db)
        db.fetch_all()
        assert _values(db, "unit__outputNode") == before
    finally:
        db.close()
    assert capsys.readouterr().out == ""


def test_existing_input_share_min_keeps_values(tmp_path: Path) -> None:
    """A DB that already has ``input_share_min`` (created from a current
    template) keeps its values: only the default / description change."""
    url = f"sqlite:///{(tmp_path / 'cur.sqlite').resolve()}"
    with DatabaseMapping(url, create=True) as db:
        _, errors = import_data(
            db,
            entity_classes=[["node", ()], ["unit", ()],
                            ["unit__inputNode", ("unit", "node")]],
            parameter_definitions=[
                ["unit__inputNode", "input_share_min", 1.0, None, "old"]],
            alternatives=[["Base", ""]],
            entities=[["node", "a"], ["unit", "u"],
                      ["unit__inputNode", ("u", "a")]],
            parameter_values=[["unit__inputNode", ("u", "a"),
                               "input_share_min", 0.3, "Base"]],
        )
        assert not errors
        db.commit_session("seed")
    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        _migrate_v70_capacity_coefficient_descriptions(db)
        db.fetch_all()
        assert _values(db, "unit__inputNode") == {
            ("input_share_min", ("u", "a"), "Base"): 0.3}
        d = db.get_parameter_definition_item(
            entity_class_name="unit__inputNode", name="input_share_min")
        assert from_database(d["default_value"], d["default_type"]) == 0.0
    finally:
        db.close()


def test_current_fixture_carries_new_names(tmp_path: Path) -> None:
    """A DB built from the (regenerated) test fixture has the new names
    on ``unit__inputNode`` only, and re-running the migration is a no-op
    that keeps every definition single."""
    url = json_to_db(FIXTURES_DIR / "tests.json", tmp_path / "tests.sqlite")
    migrate_database(url)
    migrate_database(url)
    db = DatabaseMapping(url, create=False)
    try:
        db.fetch_all()
        names = [(p["entity_class_name"], p["name"])
                 for p in db.get_parameter_definition_items()
                 if p["entity_class_name"] in ("unit__inputNode",
                                               "unit__outputNode")]
        assert len(names) == len(set(names))
        assert ("unit__inputNode", "input_share_max") in names
        assert ("unit__inputNode", "input_share_min") in names
        assert ("unit__inputNode", "capacity_max_coeff") not in names
        assert ("unit__outputNode", "capacity_max_coeff") in names
        assert ("unit__outputNode", "capacity_min_coeff") in names
    finally:
        db.close()


def test_penalty_reserve_default_restored(tmp_path: Path) -> None:
    """The v56-cleared ``penalty_reserve`` default comes back as 5000 with
    the new description; an authored value stays; re-running keeps it."""
    url = f"sqlite:///{(tmp_path / 'pen.sqlite').resolve()}"
    with DatabaseMapping(url, create=True) as db:
        _, errors = import_data(
            db,
            entity_classes=[["reserve", ()], ["upDown", ()], ["group", ()],
                            ["reserve__upDown__group",
                             ("reserve", "upDown", "group")]],
            parameter_definitions=[
                ["reserve__upDown__group", "penalty_reserve", None, None,
                 "[CUR/MW] Penalty for violating a reserve constraint. "
                 "Constant."]],
            alternatives=[["Base", ""]],
            entities=[["reserve", "r"], ["upDown", "up"], ["group", "g"],
                      ["reserve__upDown__group", ("r", "up", "g")]],
            parameter_values=[["reserve__upDown__group", ("r", "up", "g"),
                               "penalty_reserve", 123.0, "Base"]],
        )
        assert not errors
        db.commit_session("seed")
    for _ in range(2):
        db = DatabaseMapping(url, create=False)
        try:
            db.fetch_all()
            _migrate_v70_capacity_coefficient_descriptions(db)
            db.fetch_all()
            d = db.get_parameter_definition_item(
                entity_class_name="reserve__upDown__group",
                name="penalty_reserve")
            assert from_database(d["default_value"],
                                 d["default_type"]) == 5000.0
            assert d["description"] == _V70_PENALTY_RESERVE_DESCRIPTION
            assert _values(db, "reserve__upDown__group") == {
                ("penalty_reserve", ("r", "up", "g"), "Base"): 123.0}
        finally:
            db.close()
