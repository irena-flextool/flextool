"""The assistant's tools show a node's ``unit_of_measure`` (unit_of_measure WP3).

Inputs: ``parameter_unit_info`` rewrites the MWh / MW / kWh / kW of a description bracket for the nodes the
parameter is about, each token from its own source (critique M8).  Results: a column whose unit follows its node
takes the unit the writer put in the footer (review S8); a sum over nodes or rows of different units is refused,
and compare_runs' overview keeps the node level of a carrier quantity (critique M4).  Expected units are written
by hand from the rule (A = the node's unit, R = A per hour: m3 -> m3/h; per kW = per 0.001 of the rate).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd
import pytest

from flextool.assistant.context import make_context
from flextool.assistant.inputs._units import NodeTopology, parameter_unit_info
from flextool.assistant.registry import call_tool
from flextool.lean_parquet import write_lean_parquet
from flextool.process_outputs._output_meta import UnitContext, derive_column_meta, row_units

from tests.assistant.conftest import FIXTURES, make_project, record_run
from tests.assistant.results_tables import dt_index, succeed_writing

MIXED = "Mixed units of measure!"
SCENARIO = "desal_uom"

#: water (m3) is fed by a pump from elec; a CHP-like unit "wtp" burns water into elec; a connection elec-water.
TOPOLOGY = NodeTopology(
    node_units={"water": "m3", "heat": "MWh_th"},
    outputs={"pump": ("water",), "wtp": ("elec",), "chp": ("elec", "heat")},
    inputs={"pump": ("elec",), "wtp": ("water",)},
    connection_ends={"pipe": ("elec", "water"), "line": ("elec", "east")},
    commodity_nodes={"water_supply": ("water",)},
    group_members={"g_mixed": (("node", "elec"), ("unit", "pump")), "g_water": (("node", "water"),)},
    flowgroup_nodes={},
)


#: The member classes of the groups (the tools read them from the scenario).
MEMBERS = {"g_water": ["node"], "g_mixed": ["node", "unit"]}


def _unit(cls: str, parameter: str, byname: tuple[str, ...]) -> tuple[str | None, bool]:
    members = MEMBERS.get(byname[0]) if cls == "group" else None
    info = parameter_unit_info(cls, parameter, nodes=TOPOLOGY, byname=byname, members=members)
    return info.unit, info.node_set


@pytest.mark.parametrize("cls, parameter, byname, expected", [
    ("node", "inflow", ("water",), "m3 per time step"),
    ("node", "penalty_up", ("water",), "CUR/m3"),
    ("node", "invest_cost", ("water",), "CUR/(0.001 m3)"),
    ("node", "annual_flow", ("water",), "m3/a"),
    ("node", "invest_cost", ("heat",), "CUR/kWh_th"),
    ("unit", "invest_cost", ("pump",), "CUR/(0.001 m3/h)"),
    ("unit", "startup_cost", ("pump",), "CUR/(m3/h)"),
    ("unit", "fixed_cost", ("chp",), "CUR/(kW + kW_th)/a"),
    # Critique M8: MWh from the input node (water), MW from the unit's capacity side (elec).
    ("unit__inputNode", "startup_energy", ("wtp", "water"), "m3 per MW started"),
    ("unit__inputNode", "startup_energy", ("pump", "elec"), "MWh per m3/h started"),
    ("unit__outputNode", "other_operational_cost", ("pump", "water"), "CUR/m3"),
    ("commodity", "price", ("water_supply",), "CUR/m3"),
    ("commodity", "co2_content", ("water_supply",), "tCO2/m3"),
    ("connection", "invest_cost", ("pipe",), MIXED),
    ("group", "invest_max_total", ("g_water",), "m3"),
    ("group", "invest_max_total", ("g_mixed",), MIXED),
])
def test_input_units_follow_the_nodes(cls, parameter, byname, expected) -> None:
    unit, node_set = _unit(cls, parameter, byname)
    assert unit == expected
    assert node_set


@pytest.mark.parametrize("cls, parameter, byname, expected", [
    ("node", "inflow", ("elec",), "MWh per time step"),
    ("connection", "invest_cost", ("line",), "CUR/kW"),
    ("unit__outputNode", "inertia_constant", ("pump", "water"), "MWs/MW"),
    ("group", "capacity_margin", ("g_water",), "MW"),
    ("reserve__upDown__group", "reservation", ("r", "up", "g_water"), "MW"),
])
def test_mwh_nodes_and_power_features_keep_the_default(cls, parameter, byname, expected) -> None:
    unit, node_set = _unit(cls, parameter, byname)
    assert unit == expected and not node_set


def test_mixed_reason_names_the_nodes() -> None:
    info = parameter_unit_info("connection", "invest_cost", nodes=TOPOLOGY, byname=("pipe",))
    assert "MWh: elec" in info.reason and "m3: water" in info.reason


# ─── tools end to end: a water node in m3 ────────────────────────────────

def _b64(value: str) -> list[str]:
    return [base64.b64encode(json.dumps(value).encode()).decode(), "str"]


@pytest.fixture(scope="module")
def uom_db(tmp_path_factory) -> Path:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    root = tmp_path_factory.mktemp("uom_assistant")
    data = json.loads((FIXTURES / "tests.json").read_text(encoding="utf-8"))
    data["alternatives"].append(["uom", "water in m3"])
    data["parameter_values"].append(["node", "water", "unit_of_measure", _b64("m3"), "uom"])
    base = [sa for sa in data["scenario_alternatives"] if sa[0] == "desal_water_storage"]
    chain = [[SCENARIO, alt, before] for _s, alt, before in base]
    chain[-1][2] = "uom"
    chain.append([SCENARIO, "uom", None])
    data["scenario_alternatives"] += chain
    data["scenarios"].append([SCENARIO, False, ""])
    path = root / "case.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    db = json_to_db(path, root / "tests.sqlite")
    migrate_database(db)
    return root / "tests.sqlite"


def _call(ctx, tool: str, **args) -> dict:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


def _refused(ctx, tool: str, **args) -> str:
    out = call_tool(ctx, tool, args)
    assert not out.ok
    return out.message


CTX = UnitContext(node_units={"water": "m3"}, capacity_nodes={"med_desal": ("water",), "coal_plant": ("west",)})


def _tables(folder: Path) -> None:
    cols = pd.MultiIndex.from_tuples([("med_desal", "water"), ("coal_plant", "west")], names=["unit", "node"])
    flows = pd.DataFrame([[1.0, 10.0]] * 6, index=dt_index(), columns=cols)
    rows = pd.MultiIndex.from_tuples([("med_desal", "p2020"), ("coal_plant", "p2020")], names=["unit", "period"])
    capacity = pd.DataFrame({"total": [5.0, 7.0]}, index=rows)
    capacity.columns.name = "parameter"
    folder.mkdir(parents=True, exist_ok=True)
    for stem, frame in (("unit_outputNode_dt_ee", flows), ("unit_capacity_ed_p", capacity)):
        meta = {k: v.as_dict() for k, v in derive_column_meta(stem, frame.columns, ctx=CTX, rows=frame.index).items()}
        wrapped = pd.concat({SCENARIO: frame}, axis=1, names=["scenario"])
        write_lean_parquet(wrapped, folder / f"{stem}.parquet", column_metadata=meta,
                           row_units=row_units(stem, frame.index, frame.columns, CTX))


@pytest.fixture(scope="module")
def uom_ctx(tmp_path_factory, uom_db):
    project = make_project(tmp_path_factory.mktemp("uom_project"), uom_db)
    record_run(project, "desal_uom_1", SCENARIO, succeed_writing(_tables))
    return make_context(project)


def test_get_parameter_shows_the_node_unit(uom_ctx) -> None:
    rec = _call(uom_ctx, "get_parameter", entity_class="node", parameter="inflow", entity=["water"],
                scenario=SCENARIO)
    assert rec["unit"] == "m3 per time step"
    assert "water" in rec["unit_reason"] and "m3" in rec["unit_reason"]
    rec = _call(uom_ctx, "get_parameter", entity_class="node", parameter="penalty_up", entity=["water"],
                scenario=SCENARIO)
    assert rec["unit"] == "CUR/m3"
    # The same node in a scenario without the alternative keeps the default unit, MWh.
    rec = _call(uom_ctx, "get_parameter", entity_class="node", parameter="inflow", entity=["water"],
                scenario="desal_water_storage")
    assert rec["unit"] == "MWh per time step"


def test_result_units_come_from_the_footer(uom_ctx) -> None:
    out = _call(uom_ctx, "query_result", run="desal_uom_1", table="unit_outputNode_dt_ee",
                group_by=["unit", "node"], agg="max")
    assert out["unit"] == {"water": "m3/h", "west": "MW"}
    # A sum over the water and the electricity node has no unit: refused.
    message = _refused(uom_ctx, "query_result", run="desal_uom_1", table="unit_outputNode_dt_ee",
                       group_by=["period"], agg="sum")
    assert "different units" in message
    described = _call(uom_ctx, "describe_result_table", run="desal_uom_1", table="unit_capacity_ed_p")
    assert described["unit_by_row"] == {"unit": {"med_desal": "m3/h", "coal_plant": "MW"}}
    message = _refused(uom_ctx, "query_result", run="desal_uom_1", table="unit_capacity_ed_p",
                       group_by=["period"], agg="sum")
    assert "different units" in message


def test_overview_keeps_the_nodes_of_a_carrier() -> None:
    """Critique M4: a column whose unit follows its node (m3/h) is a carrier quantity, also when its text is not
    a power or energy unit, so compare_runs' overview does not add it up over nodes."""
    from flextool.assistant.results._catalog import Measure
    from flextool.assistant.results._query import Query, Table
    from flextool.assistant.tools.results import _overview
    from flextool.lean_parquet import LeanLayout

    layout = LeanLayout(idx=("period", "time"), index_physical=("period", "time"), col=("scenario", "unit", "node"),
                        columns=(("S", "desal", "water"), ("S", "pump", "water")), physical=("c0", "c1"),
                        column_meta=None, num_rows=6, lean=True)
    table = Table("unit_outputNode_dt_ee", Path("x"), layout,
                  [Measure("m3/h", "instantaneous", True)] * 2, False)
    query, notes = _overview(table, Query())
    assert "node" in query.group_by and any("not added up over node" in n for n in notes)
