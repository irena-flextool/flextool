"""Workbook round-trip of the start-up input (``unit__inputNode.startup_input_method`` / ``startup_energy``,
database version 79).

The two parameters exist only on ``unit__inputNode``, while the workbook merges unit__inputNode and unit__outputNode
into the ``unit_node`` sheets.  A database built from the live master template (never a checked-in .sqlite) with a
``startup_only`` arc (period-map ``startup_energy``) and a ``startup_and_input`` arc (constant) is exported to xlsx
and imported into a fresh template: the values and the arcs come back.  The Excel and database readers see the arc
as an ordinary entity (only the engine hides a ``startup_only`` arc).
"""
from __future__ import annotations

import pytest
from spinedb_api import DatabaseMapping, Map, from_database, to_database

from flextool._resources import package_data_path
from flextool.export_to_tabular.export_to_excel import export_to_excel
from flextool.process_inputs.read_self_describing_excel import read_self_describing_excel
from flextool.process_inputs.write_self_describing_to_db import write_sheet_data_to_db
from flextool.update_flextool.initialize_database import initialize_database

MASTER_TEMPLATE = package_data_path("schemas/spinedb_schema.json")

ENTITIES = [("unit", ("cp",)), ("node", ("coal_n",)), ("node", ("oil_n",)), ("node", ("elec",)),
            ("unit__inputNode", ("cp", "coal_n")), ("unit__inputNode", ("cp", "oil_n")),
            ("unit__outputNode", ("cp", "elec"))]
VALUES = [
    ("unit__inputNode", ("cp", "oil_n"), "startup_input_method", "startup_only"),
    ("unit__inputNode", ("cp", "oil_n"), "startup_energy",
     Map(indexes=["p2030", "p2040"], values=[2.0, 1.5], index_name="period")),
    ("unit__inputNode", ("cp", "coal_n"), "startup_input_method", "startup_and_input"),
    ("unit__inputNode", ("cp", "coal_n"), "startup_energy", 0.5),
]


@pytest.fixture(scope="module")
def round_tripped(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, set]:
    out = tmp_path_factory.mktemp("startup_rt")
    src = out / "src.sqlite"
    initialize_database(str(MASTER_TEMPLATE), str(src))
    src_url = f"sqlite:///{src}"
    with DatabaseMapping(src_url) as db:
        for cls, byname in ENTITIES:
            db.add_entity(entity_class_name=cls, entity_byname=byname)
        for cls, byname, name, value in VALUES:
            v, t = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=name,
                                   alternative_name="base", value=v, type=t)
        db.commit_session("start-up input")

    xlsx = out / "rt.xlsx"
    export_to_excel(src_url, str(xlsx), include_advanced=True)
    rt = out / "rt.sqlite"
    initialize_database(str(MASTER_TEMPLATE), str(rt))
    rt_url = f"sqlite:///{rt}"
    sheets = read_self_describing_excel(str(xlsx), skip_sheets={"navigate", "version"})
    write_sheet_data_to_db(sheets, rt_url, purge_first=True, keep_entities=True)
    with DatabaseMapping(rt_url) as db:
        values = {(p["entity_class_name"], tuple(p["entity_byname"]), p["parameter_definition_name"]):
                  from_database(p["value"], p["type"]) for p in db.get_parameter_value_items()}
        entities = {(e["entity_class_name"], tuple(e["entity_byname"])) for e in db.get_items("entity")}
    return values, entities


def _flatten(value) -> object:
    if isinstance(value, Map):
        return tuple((str(i), _flatten(v)) for i, v in zip(value.indexes, value.values))
    return value


@pytest.mark.parametrize("cls,byname,name,value", VALUES, ids=[f"{v[2]}.{v[1][1]}" for v in VALUES])
def test_start_up_values_survive_the_workbook(round_tripped, cls, byname, name, value) -> None:
    values, _ = round_tripped
    key = (cls, byname, name)
    assert key in values, f"{name} of {byname} was lost on the workbook round-trip"
    assert _flatten(values[key]) == _flatten(value)


def test_start_up_arcs_survive_the_workbook(round_tripped) -> None:
    """Both input arcs (the startup_only one too) and the output arc come back as entities of their own class."""
    _, entities = round_tripped
    assert {("unit__inputNode", ("cp", "coal_n")), ("unit__inputNode", ("cp", "oil_n")),
            ("unit__outputNode", ("cp", "elec"))} <= entities
