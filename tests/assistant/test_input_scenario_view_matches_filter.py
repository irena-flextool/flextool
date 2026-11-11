"""The tools' own scenario view equals spinedb-api's scenario filter (the oracle), for every fixture scenario.

The view is computed from one unfiltered load (``flextool.assistant.inputs._db``); the oracle applies
``apply_scenario_filter_to_subqueries`` to the same database and reads the entities and parameter values that the
filter keeps.  Both must give the same active entities (including those decided by ``entity_alternative`` and by
inactive elements of multi-dimensional entities) and the same winning value of every parameter.  FlexTool's
``is_enabled`` gating sits on top of the Spine rules and is checked apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flextool.assistant.inputs import _db

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tests.json"
SCENARIOS = [row[0] for row in json.loads(FIXTURE.read_text(encoding="utf-8"))["scenarios"]]


def _oracle(db_path: Path, scenario: str) -> tuple[set, dict]:
    from spinedb_api.filters.scenario_filter import apply_scenario_filter_to_subqueries

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{db_path.as_posix()}") as db:
        classes = {c.id: c.name for c in db.query(db.entity_class_sq).all()}
        definitions = {p.id: p.name for p in db.query(db.parameter_definition_sq).all()}
        alternatives = {a.id: a.name for a in db.query(db.alternative_sq).all()}
        apply_scenario_filter_to_subqueries(db, scenario)
        entities = {}
        for e in db.query(db.wide_entity_sq).all():
            byname = tuple(e.element_name_list.split(",")) if e.element_name_list else (e.name,)
            entities[e.id] = (classes[e.class_id], byname)
        values = {}
        for v in db.query(db.parameter_value_sq).all():
            cls, byname = entities[v.entity_id]
            values[(cls, byname, definitions[v.parameter_definition_id])] = (alternatives[v.alternative_id], v.value)
    return set(entities.values()), values


@pytest.fixture(scope="module")
def input_db(tests_db: Path) -> _db.InputDB:
    return _db.load_uncached(tests_db)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_view_equals_the_scenario_filter(input_db: _db.InputDB, tests_db: Path, scenario: str) -> None:
    oracle_entities, oracle_values = _oracle(tests_db, scenario)
    view = input_db.view(scenario)
    ours = {(cls, byname) for cls, ents in input_db.entities.items() for byname in ents
            if view.spine_active(cls, byname)[0]}
    assert ours == oracle_entities
    our_values = {key: (item.alternative, item.value) for key, item in view.winners().items()
                  if view.spine_active(key[0], key[1])[0]}
    assert our_values == oracle_values


def test_the_fixture_exercises_entity_alternatives_and_elements(input_db: _db.InputDB) -> None:
    """The comparison above covers the cases the filter rules are about (tests.json): an entity made inactive by
    its own entity_alternative (reserve__upDown__group n_1/up/electricity is false in lossless_2way), and a
    multi-dimensional entity whose element is inactive (unit__outputNode coal_chp/heat in scenario coal, where
    unit coal_chp has no entity_alternative)."""
    view = input_db.view("lossless_2way")
    assert view.spine_active("reserve__upDown__group", ("n_1", "up", "electricity")) == (
        False, "it is set inactive in alternative 'lossless_2way'")
    coal = input_db.view("coal")
    active, reason = coal.spine_active("unit__outputNode", ("coal_chp", "heat"))
    assert active is False and "unit 'coal_chp' is not active" in reason


def test_is_enabled_no_drops_a_gated_entity(input_db: _db.InputDB) -> None:
    """tests.json: reserve__upDown__unit__node primary/up/coal_plant/west has is_enabled "no" in alternative reserve,
    primary/up/wind_plant/west "yes" there; scenario all has alternative reserve (and not Base).  Both are active by
    the Spine rules (the class is active by default, coal_plant / wind_plant / west are active in all)."""
    view = input_db.view("all")
    coal = ("primary", "up", "coal_plant", "west")
    wind = ("primary", "up", "wind_plant", "west")
    assert view.spine_active("reserve__upDown__unit__node", coal) == (True, None)
    assert view.active("reserve__upDown__unit__node", coal) == (
        False, "is_enabled is 'no' (from alternative 'reserve')")
    assert view.active("reserve__upDown__unit__node", wind) == (True, None)
