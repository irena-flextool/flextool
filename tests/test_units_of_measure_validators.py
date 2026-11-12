"""``validate_units_of_measure``: warnings about ``node.unit_of_measure`` values that make labels mixed.

Each test builds a small database from ``schemas/spinedb_schema.json`` (the session ``schema_db_url``, copied) and
adds the entities it needs.  The default (every node MWh) gives no warning.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, to_database

from flextool.input_derivation._validators import node_units_of_measure, validate_units_of_measure

MIXED = "Mixed units of measure!"


def _db(tmp_path: Path, schema_db_url: str, entities: list[tuple[str, tuple[str, ...]]],
        values: list[tuple[str, tuple[str, ...], str, object]]) -> str:
    path = tmp_path / "uom.sqlite"
    shutil.copy(schema_db_url.removeprefix("sqlite:///"), path)
    url = f"sqlite:///{path}"
    with DatabaseMapping(url) as db:
        for cls, byname in entities:
            db.add_entity(entity_class_name=cls, entity_byname=byname)
        for cls, byname, param, value in values:
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=param,
                                   alternative_name="base", value=val, type=typ)
        db.commit_session("units of measure test")
    return url


def _warnings(url: str, caplog) -> list[str]:
    logger = logging.getLogger("test_units_of_measure_validators")
    caplog.set_level(logging.WARNING, logger=logger.name)
    with DatabaseMapping(url) as db:
        validate_units_of_measure(db, logger)
    return [r.getMessage() for r in caplog.records if r.name == logger.name]


NODES = [("node", ("elec",)), ("node", ("east",)), ("node", ("water",)), ("node", ("heat",))]
WATER = [("node", ("water",), "unit_of_measure", "m3"), ("node", ("heat",), "unit_of_measure", "MWh_th")]


def _uom(tmp_path, schema_db_url, caplog, entities, values=WATER) -> list[str]:
    return _warnings(_db(tmp_path, schema_db_url, NODES + entities, values), caplog)


def test_default_gives_no_warning(tmp_path, schema_db_url, caplog) -> None:
    entities = [("connection", ("c",)), ("connection__node__node", ("c", "elec", "east")),
                ("group", ("g",)), ("group__node", ("g", "elec")), ("group__node", ("g", "east"))]
    assert _uom(tmp_path, schema_db_url, caplog, entities, values=[]) == []


def test_node_units_map(tmp_path, schema_db_url) -> None:
    url = _db(tmp_path, schema_db_url, NODES, WATER + [("node", ("east",), "unit_of_measure", "MWh")])
    with DatabaseMapping(url) as db:
        assert node_units_of_measure(db) == {"water": "m3", "heat": "MWh_th"}


def test_connection_across_units(tmp_path, schema_db_url, caplog) -> None:
    entities = [("connection", ("pipe",)), ("connection__node__node", ("pipe", "elec", "water")),
                ("connection", ("line",)), ("connection__node__node", ("line", "elec", "east"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities)
    assert len(found) == 1, found
    assert "Connection 'pipe'" in found[0] and "MWh: elec; m3: water" in found[0] and MIXED in found[0]


def test_commodity_across_units(tmp_path, schema_db_url, caplog) -> None:
    entities = [("commodity", ("fuel",)), ("commodity__node", ("fuel", "elec")),
                ("commodity__node", ("fuel", "water")), ("commodity", ("power",)),
                ("commodity__node", ("power", "elec")), ("commodity__node", ("power", "east"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities)
    assert len(found) == 1, found
    assert "Commodity 'fuel'" in found[0] and "co2_content" in found[0]


def test_node_group_across_units_warns_once(tmp_path, schema_db_url, caplog) -> None:
    entities = [("group", ("all",)), ("group__node", ("all", "elec")), ("group__node", ("all", "water")),
                ("group__node", ("all", "east")), ("group__node", ("all", "heat")),
                ("group", ("power",)), ("group__node", ("power", "elec")), ("group__node", ("power", "east"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities)
    assert len(found) == 1, found
    assert found[0].startswith("Group 'all':") and MIXED in found[0]
    # Grouped by unit, so the m3 node that makes the sum mixed is named (WP1 review S3).
    assert "MWh: east, elec; MWh_th: heat; m3: water" in found[0]


def test_flow_group_across_units(tmp_path, schema_db_url, caplog) -> None:
    entities = [("unit", ("desal",)), ("unit__inputNode", ("desal", "elec")), ("unit__outputNode", ("desal", "water")),
                ("flowGroup", ("fg",)), ("flowGroup__unit__node", ("fg", "desal", "elec")),
                ("flowGroup__unit__node", ("fg", "desal", "water"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities)
    assert len(found) == 1, found
    assert "flowGroup 'fg'" in found[0] and MIXED in found[0]


@pytest.mark.parametrize("flag,value,feature", [
    ("has_inertia", "yes", "inertia"),
    ("has_non_synchronous", "yes", "the non-synchronous limit"),
    ("capacity_margin_method", "manual", "the capacity margin"),
])
def test_power_feature_group_with_a_non_power_node(tmp_path, schema_db_url, caplog, flag, value, feature) -> None:
    entities = [("group", ("g",)), ("group__node", ("g", "water"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities, WATER + [("group", ("g",), flag, value)])
    assert len(found) == 1, found
    assert f"it has {feature}; these group features are in MW (inertia in MW·s)" in found[0]
    assert "water (m3) is not in MWh" in found[0]


def test_reserve_group_with_a_non_power_node(tmp_path, schema_db_url, caplog) -> None:
    entities = [("group", ("g",)), ("group__node", ("g", "water")), ("reserve", ("primary",)),
                ("reserve__upDown__group", ("primary", "up", "g"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities)
    assert len(found) == 1, found
    assert "it has reserves" in found[0]


def test_power_feature_group_with_a_suffixed_mwh_node_is_fine(tmp_path, schema_db_url, caplog) -> None:
    """Critique S4: an ``MWh_<suffix>`` node is a power node."""
    entities = [("group", ("g",)), ("group__node", ("g", "heat")), ("group__node", ("g", "east"))]
    values = [("node", ("heat",), "unit_of_measure", "MWh_th"), ("node", ("east",), "unit_of_measure", "MWh_el"),
              ("group", ("g",), "has_inertia", "yes")]
    found = _uom(tmp_path, schema_db_url, caplog, entities, values)
    # Their units differ (a mixed sum), but neither is a non-power node.
    assert len(found) == 1 and "assume power nodes" not in found[0], found


def test_group_capacity_limit_over_different_units(tmp_path, schema_db_url, caplog) -> None:
    """Critique S5: a group invest limit sums the capacities of a water-side and an electricity-side unit."""
    entities = [("unit", ("desal",)), ("unit__inputNode", ("desal", "elec")), ("unit__outputNode", ("desal", "water")),
                ("unit", ("pv",)), ("unit__outputNode", ("pv", "elec")),
                ("group", ("cap",)), ("group__unit", ("cap", "desal")), ("group__unit", ("cap", "pv")),
                ("group", ("free",)), ("group__unit", ("free", "desal")), ("group__unit", ("free", "pv"))]
    found = _uom(tmp_path, schema_db_url, caplog, entities, WATER + [("group", ("cap",), "invest_max_total", 10.0)])
    assert len(found) == 1, found
    assert found[0].startswith("Group 'cap':") and "desal (m3)" in found[0] and "pv (MWh)" in found[0]


def test_multi_output_unit_gives_no_warning(tmp_path, schema_db_url, caplog) -> None:
    """Owner decision S1: a CHP with an MWh_th heat output is labelled ``MW + MW_th``, not warned about."""
    entities = [("unit", ("chp",)), ("unit__inputNode", ("chp", "east")), ("unit__outputNode", ("chp", "elec")),
                ("unit__outputNode", ("chp", "heat"))]
    assert _uom(tmp_path, schema_db_url, caplog, entities) == []


@pytest.mark.parametrize("value,shown,fix", [("m3/h", "m3/h/h", "m3"), ("kg/s", "kg/s/h", "kg"),
                                             ("MW", "MW/h", "MWh")])
def test_rate_like_value(tmp_path, schema_db_url, caplog, value: str, shown: str, fix: str) -> None:
    """The warning shows the label the value would give and the amount unit to set (WP1 review S5)."""
    found = _uom(tmp_path, schema_db_url, caplog, [], [("node", ("water",), "unit_of_measure", value)])
    assert len(found) == 1, found
    assert "looks like a rate" in found[0] and f"'{value}'" in found[0]
    assert f"flows would be labelled '{shown}'. Set '{fix}'." in found[0]


def test_a_capital_w_in_a_name_is_not_a_rate(tmp_path, schema_db_url, caplog) -> None:
    assert _uom(tmp_path, schema_db_url, caplog, [], [("node", ("water",), "unit_of_measure", "Wood_t")]) == []


def test_database_without_the_definition(tmp_path, caplog) -> None:
    """Critique S6: a database that lacks the definition (or classes) reads as all MWh, no warning, no error."""
    url = f"sqlite:///{tmp_path / 'bare.sqlite'}"
    with DatabaseMapping(url, create=True) as db:
        db.add_entity_class(name="node")
        db.add_entity(entity_class_name="node", name="n")
        db.commit_session("bare")
    assert _warnings(url, caplog) == []
    with DatabaseMapping(url) as db:
        assert node_units_of_measure(db) == {}


# ─── end to end: the label reaches the provider and changes nothing in the LP ───


def test_solve_carries_the_node_units_and_keeps_the_lp(tmp_path_factory, monkeypatch, caplog) -> None:
    """``tests.json`` chain ``init, west``: setting ``west`` to ``m3`` puts ``{west: m3}`` on the provider frame
    read by :func:`flextool.units_of_measure.read_node_units`, leaves the objective as it is (a label only), and
    solves under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (the parameter is registered)."""
    from flextool.units_of_measure import read_node_units
    from tests.engine._unit_db import b64, make_db, run

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    chain = ["init", "west"]
    objectives = {}
    for alt, values in (("uom_default", []), ("uom_m3", [["node", "west", "unit_of_measure", b64("m3", "str")]])):
        steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain))
        last = next(reversed(steps.values()))
        objectives[alt] = last.solution.obj
        provider = getattr(last, "flex_data_provider", None)
        assert provider is not None
        assert read_node_units(provider) == ({} if alt == "uom_default" else {"west": "m3"})
    assert objectives["uom_m3"] == objectives["uom_default"]


def test_group_capacity_limit_counts_storage_nodes_only(tmp_path, schema_db_url, caplog) -> None:
    """A balance node of a limited group has no capacity: a water balance node next to an electricity unit gives no
    capacity warning; a water storage node does."""
    entities = [("unit", ("pv",)), ("unit__outputNode", ("pv", "elec")),
                ("group", ("cap",)), ("group__unit", ("cap", "pv")), ("group__node", ("cap", "water"))]
    limit = [("group", ("cap",), "invest_max_total", 10.0)]
    found = _uom(tmp_path, schema_db_url, caplog, entities, WATER + limit)
    assert not any("capacity limits" in w for w in found), found
    caplog.clear()
    (tmp_path / "storage").mkdir()
    found = _uom(tmp_path / "storage", schema_db_url, caplog, entities,
                 WATER + limit + [("node", ("water",), "node_type", "storage")])
    assert len(found) == 1 and "capacity limits" in found[0] and "water (m3)" in found[0], found
