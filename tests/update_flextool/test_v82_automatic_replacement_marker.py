"""Part 5 of v82: the behaviour-change marker ``automatic_replacement_cost``.

A reinvest_automatic unit (explicit, or without a lifetime_method: the default) with a lifetime_technical is named
when it may invest and its investment cost now follows its replacements (a period map of invest_cost /
discount_rate / lifetime_technical / payback_time, a payback_time, a non-zero net_salvage_value), or when it may
retire (a retirement now avoids the replacements still to come).  On a hand-built database:

* ``mapped``: default lifetime_method, invest_no_limit, invest_cost a period map → named;
* ``salvaged``: reinvest_automatic, invest_no_limit, net_salvage_value 200 → named;
* ``retiring``: reinvest_automatic, retire_no_limit, constant values → named (a retirement option);
* ``plain``: reinvest_automatic, invest_no_limit, constant invest_cost, no salvage → not named (the chain of
  constant values charges what it did);
* ``bounded``: reinvest_choice with a map of invest_cost → not named (no replacement);
* ``lifeless``: invest_no_limit, a map of invest_cost but no lifetime_technical → not named (one plant for ever).
"""
from __future__ import annotations

import json

from spinedb_api import DatabaseMapping, Map, to_database

from flextool.update_flextool import db_migration as m
from flextool.update_flextool.db_migration import v82


def _db() -> DatabaseMapping:
    db = DatabaseMapping("sqlite://", create=True)
    db.add_item("entity_class", name="unit")
    db.add_item("alternative", name="base")
    for name in ("lifetime_method", "lifetime_technical", "invest_method", "invest_cost", "discount_rate",
                 "payback_time", "net_salvage_value"):
        db.add_item("parameter_definition", entity_class_name="unit", name=name)
    values = {
        "mapped": {"invest_method": "invest_no_limit", "lifetime_technical": 10.0,
                   "invest_cost": Map(["p2020", "p2030"], [400.0, 300.0], index_name="period")},
        "salvaged": {"lifetime_method": "reinvest_automatic", "invest_method": "invest_no_limit",
                     "lifetime_technical": 10.0, "invest_cost": 400.0, "net_salvage_value": 200.0},
        "retiring": {"lifetime_method": "reinvest_automatic", "invest_method": "retire_no_limit",
                     "lifetime_technical": 10.0},
        "plain": {"lifetime_method": "reinvest_automatic", "invest_method": "invest_no_limit",
                  "lifetime_technical": 10.0, "invest_cost": 400.0, "net_salvage_value": 0.0},
        "bounded": {"lifetime_method": "reinvest_choice", "invest_method": "invest_no_limit",
                    "lifetime_technical": 10.0,
                    "invest_cost": Map(["p2020", "p2030"], [400.0, 300.0], index_name="period")},
        "lifeless": {"invest_method": "invest_no_limit",
                     "invest_cost": Map(["p2020", "p2030"], [400.0, 300.0], index_name="period")},
    }
    for entity, params in values.items():
        db.add_item("entity", entity_class_name="unit", name=entity)
        for name, value in params.items():
            raw, typ = to_database(value)
            db.add_item("parameter_value", entity_class_name="unit", entity_byname=(entity,),
                        parameter_definition_name=name, alternative_name="base", value=raw, type=typ)
    return db


def _marker(db) -> dict | None:
    for md in db.get_items("metadata"):
        if md["name"] == m.BEHAVIOUR_CHANGE_MARKER_PREFIX + "automatic_replacement_cost":
            return json.loads(md["value"])
    return None


def test_marker_names_the_automatic_entities_whose_costs_follow_their_replacements() -> None:
    with _db() as db:
        v82._v82_automatic_replacement_marker(db)
        assert _marker(db) == {"since": 82, "entities": ["mapped", "retiring", "salvaged"]}
        # Idempotent: a rerun writes the same marker.
        v82._v82_automatic_replacement_marker(db)
        assert _marker(db) == {"since": 82, "entities": ["mapped", "retiring", "salvaged"]}
