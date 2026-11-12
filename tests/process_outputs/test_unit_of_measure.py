"""Result units follow ``node.unit_of_measure`` (unit_of_measure WP2: output wiring).

One solve of the desalination case of ``tests/fixtures/tests.json`` (``desal_water_storage``: a coal plant and a
back-pressure CHP feed the electricity node ``west``; the CHP's heat node ``desal_heat`` feeds the desalination
unit ``med_desal``, whose output is the water storage ``water``), with an alternative ``uom`` that sets
``water`` to ``m3`` and ``desal_heat`` to ``MWh_th``, and with ``water`` added to the nodeGroup ``electricity``
(so the group mixes MWh and m3).  The oracle units are written by hand from the rule (A = amount, R = A per
hour, A/a per year, CUR/A prices, the capacity label of owner decision S1), never copied from the output.

The pure tests resolve hand-built columns against a hand-built :class:`UnitContext`.
"""
from __future__ import annotations

import base64
import csv
import json
import shutil
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from flextool import units_of_measure as uom
from flextool.lean_parquet import read_lean_parquet_column_meta, read_lean_parquet_row_units
from flextool.process_outputs import _output_meta as meta
from flextool.process_outputs._output_meta import UNIT_CONTEXT_FILE, UnitContext, resolve_columns, row_units

TESTS = Path(__file__).resolve().parents[1]
REPO = TESTS.parent
MIXED = "Mixed units of measure!"
SCENARIO = "desal_uom"
DEFAULT_SCENARIO = "desal_water_storage"


# ─── pure: templates, registry, resolver ─────────────────────────────────

def _transforms() -> dict[str, meta.Transform]:
    return {name: tf for name, tf in vars(meta).items() if isinstance(tf, meta.Transform)}


def test_template_at_mwh_is_the_default_unit() -> None:
    """Every template filled for an MWh node gives today's literal unit (so default outputs cannot change)."""
    templated = {name: tf for name, tf in _transforms().items() if tf.template}
    assert set(templated) == {
        "RATE", "RAMP", "AVERAGE_MW", "ENERGY_ANNUAL", "PREENERGY_ANNUAL", "ENERGY_PERIOD", "ENERGY_PERSTEP",
        "ENERGY_HORIZON", "STORAGE_STATE", "CAPACITY_MWH", "CAPACITY_MW", "PRICE_ENERGY",
        "INVEST_COST_DISCOUNTED", "DUAL_INVEST_KW", "DUAL_INVEST_KWH"}
    for name, tf in templated.items():
        assert uom.fill(tf.template, "MWh") == tf.unit, name
    # The annuity of a storage node is per its amount unit (critique M2, also in a default run: CUR/MWh).
    assert meta.INVEST_COST_DISCOUNTED.node_template == "CUR/{A}"


def test_every_templated_table_has_a_unit_source() -> None:
    """A table whose transform (or map entry) has a template follows its nodes (UNIT_SOURCE) or is a power
    feature (DEFAULT_UNITS_ONLY): nothing is forgotten silently (registry basis = "has a template")."""
    templated = set()
    for key, spec in meta.OUTPUT_TRANSFORM.items():
        transforms = spec.values() if isinstance(spec, dict) else [spec]
        if any(tf.template for tf in transforms):
            templated.add(key)
    assert templated - set(meta.UNIT_SOURCE) - meta.DEFAULT_UNITS_ONLY == set()
    assert set(meta.UNIT_SOURCE) <= templated


CTX = UnitContext(
    node_units={"water": "m3", "heat": "MWh_th"},
    capacity_nodes={"desal": ("water",), "chp": ("elec", "heat"), "chp_arcs": ("elec", "heat"), "pv": ("elec",)},
    shared_capacity=frozenset({"chp"}),
    connection_nodes={"pipe": ("elec", "water"), "line": ("elec", "east")},
    group_nodes={"all": ("elec", "water"), "power": ("elec", "east"), "wet": ("water",)},
    flowgroup_nodes={"fg": ("elec", "water")},
)


def _units(key: str, columns, ctx=CTX, **kw) -> list[str]:
    return [m.unit for m in resolve_columns(key, columns, ctx=ctx, **kw)]


def test_node_level_columns() -> None:
    cols = pd.MultiIndex.from_tuples([("desal", "water"), ("pv", "elec"), ("chp", "heat")], names=["unit", "node"])
    assert _units("unit_outputNode_dt_ee", cols) == ["m3/h", "MW", "MW_th"]
    assert _units("unit_outputNode_d_ee", cols) == ["m3/a", "MWh/a", "MWh_th/a"]
    nodes = pd.Index(["water", "elec"], name="node")
    assert _units("node_state_dt_e", nodes) == ["m3", "MWh"]
    assert _units("node_inflow__dt", nodes) == ["m3", "MWh"]
    assert _units("node_prices_dt_e", nodes) == ["CUR/m3", "CUR/MWh"]
    # Without level names the declared position is used, also below a comparison's scenario level.
    assert _units("unit_outputNode_dt_ee", [("desal", "water")]) == ["m3/h"]
    assert _units("unit_outputNode_dt_ee", [("S1", "desal", "water")], ctx={"S1": CTX}) == ["m3/h"]


def test_connections_node_level_beats_both_ends() -> None:
    """Critique S2: leftward/rightward flows take their node's unit; net flow and losses over different ends
    are mixed; a connection between two MWh nodes stays MW."""
    cols = pd.MultiIndex.from_tuples([("pipe", "water"), ("pipe", "elec")], names=["process", "node"])
    assert _units("connection_leftward_dt_eee", cols) == ["m3/h", "MW"]
    procs = pd.Index(["pipe", "line"], name="process")
    assert _units("connection_dt_eee", procs) == [MIXED, "MW"]
    assert _units("connection_losses_d_eee", procs) == [MIXED, "MWh/a"]


def test_groups_and_power_features() -> None:
    groups = pd.Index(["all", "power", "wet"], name="group")
    assert _units("nodeGroup_flows_dt_g", groups) == [MIXED, "MWh", "m3"]
    assert _units("nodeGroup_flows_d_g", groups) == [MIXED, "MWh/a", "m3/a"]
    # Critique M1: the reserve / non-synchronous / inertia tables stay MW whatever their nodes.
    assert _units("nodeGroup_slack_nonsync_dt_g", groups) == ["MW"] * 3
    assert _units("nodeGroup_slack_nonsync_d_g", groups) == ["MWh/a"] * 3
    assert _units("nodeGroup_inertia_largest_flow_dt_g", groups) == ["MW"] * 3
    reserve = pd.MultiIndex.from_tuples([("desal", "primary", "up", "water")],
                                        names=["process", "reserve", "updown", "node"])
    assert _units("process_reserve_upDown_node_dt_eppe", reserve) == ["MW"]
    slack = pd.MultiIndex.from_tuples([("primary", "up", "wet")], names=["reserve", "updown", "node_group"])
    assert _units("nodeGroup_slack_reserve_dt_eeg", slack) == ["MW"]


def test_capacity_rows_and_mixed_column_rows() -> None:
    """Critique M3 + owner decision S1: row-keyed tables give each row entity its units, per column."""
    cols = pd.Index(["existing", "total"], name="parameter")
    rows = pd.MultiIndex.from_tuples([("desal", "p1"), ("desal", "p2"), ("chp", "p1"), ("chp_arcs", "p1"),
                                      ("pv", "p1")], names=["unit", "period"])
    by_row = row_units("unit_capacity_ed_p", rows, cols, CTX)
    assert by_row == {"desal": {"existing": "m3/h", "total": "m3/h"},
                      "chp": {"existing": "MW + MW_th", "total": "MW + MW_th"},
                      "chp_arcs": {"existing": "MW; MW_th", "total": "MW; MW_th"},
                      "pv": {"existing": "MW", "total": "MW"}}
    # The column's own unit joins the units of its rows.
    assert _units("unit_capacity_ed_p", cols, rows=rows) == ["MW; MW + MW_th; MW_th; m3/h"] * 2
    groups = pd.MultiIndex.from_tuples([("wet", "p1", "t1"), ("all", "p1", "t1")],
                                       names=["group", "period", "time"])
    gcols = pd.Index(["1. Loss of load", "6. Curtailed VRE of potential VRE", "7. Annualized inflow"],
                     name="parameter")
    assert row_units("nodeGroup_gdt_p", groups, gcols, CTX) == {
        "wet": {"1. Loss of load": "m3", "6. Curtailed VRE of potential VRE": "", "7. Annualized inflow": "m3/a"},
        "all": {"1. Loss of load": MIXED, "6. Curtailed VRE of potential VRE": "",
                "7. Annualized inflow": MIXED}}
    conn = pd.MultiIndex.from_tuples([("pipe", "p1"), ("line", "p1")], names=["connection", "period"])
    assert row_units("connection_capacity_ed_p", conn, cols, CTX)["pipe"]["total"] == MIXED
    nodes = pd.MultiIndex.from_tuples([("water", "p1")], names=["node", "period"])
    assert row_units("node_capacity_ed_p", nodes, cols, CTX) == {"water": {"existing": "m3", "total": "m3"}}
    fg = pd.MultiIndex.from_tuples([("fg", "p1")], names=["group", "period"])
    assert row_units("flowGroup_gd_p", fg, pd.Index(["cumulative_flow", "average_flow"], name="parameter"),
                     CTX) == {"fg": {"cumulative_flow": MIXED, "average_flow": MIXED}}


def test_annuity_and_duals_by_class() -> None:
    """entity_annuity_d_p holds units, connections and nodes: per capacity rate, per storage amount (M2)."""
    cols = pd.Index(["desal", "water", "pipe", "pv", "elec"], name="entity")
    assert _units("entity_annuity_d_p", cols) == ["CUR/(m3/h)", "CUR/m3", MIXED, "CUR/MW", "CUR/MWh"]
    assert _units("dual_invest_effective_unit_d_e", pd.Index(["desal", "chp"], name="entity")) == [
        "CUR/(0.001 m3/h)", "CUR/(kW + kW_th)"]
    assert _units("dual_invest_effective_node_d_e", pd.Index(["water", "heat"], name="entity")) == [
        "CUR/(0.001 m3)", "CUR/kWh_th"]


def test_default_context_is_the_default() -> None:
    """No node units: the transforms' units, no row units, whatever the other maps hold; only the annuity of a
    storage node is per its amount (CUR/MWh) once the context knows the units and connections (WP2 review 1)."""
    ctx = UnitContext(capacity_nodes=CTX.capacity_nodes, group_nodes=CTX.group_nodes)
    cols = pd.MultiIndex.from_tuples([("desal", "water")], names=["unit", "node"])
    assert _units("unit_outputNode_dt_ee", cols, ctx=ctx) == ["MW"]
    entities = pd.Index(["water", "desal"], name="entity")
    assert _units("entity_annuity_d_p", entities, ctx=ctx) == ["CUR/MWh", "CUR/MW"]
    # Without the class maps (no run context) every entity keeps the transform's CUR/MW.
    assert _units("entity_annuity_d_p", entities, ctx=UnitContext()) == ["CUR/MW", "CUR/MW"]
    assert _units("entity_annuity_d_p", entities, ctx=None) == ["CUR/MW", "CUR/MW"]
    rows = pd.MultiIndex.from_tuples([("desal", "p1")], names=["unit", "period"])
    assert row_units("unit_capacity_ed_p", rows, pd.Index(["total"], name="parameter"), ctx) is None
    assert row_units("unit_capacity_ed_p", rows, pd.Index(["total"], name="parameter"), None) is None


def test_context_json_round_trip(tmp_path) -> None:
    meta.write_unit_context(str(tmp_path), CTX)
    data = json.loads((tmp_path / UNIT_CONTEXT_FILE).read_text())
    assert data["version"] == meta.UNIT_CONTEXT_VERSION
    assert meta.load_unit_context(tmp_path) == CTX
    meta.write_unit_context(str(tmp_path), UnitContext())
    assert not (tmp_path / UNIT_CONTEXT_FILE).exists()
    assert meta.load_unit_context(tmp_path) is None


def test_startup_energy_unit() -> None:
    """Critique M8: startup_energy is the input node's amount per the unit's capacity rate."""
    assert uom.startup_energy_unit("m3", uom.capacity_units("{R}", ["elec"], CTX.node_units, shared=False)) == \
        "m3 per MW started"
    # A shared capacity is one rate (WP2 review NIT).
    assert uom.startup_energy_unit("m3", "MW + MW_th") == "m3 per (MW + MW_th) started"


# ─── one solve with water in m3 and heat in MWh_th ───────────────────────

def _b64(value: str) -> list[str]:
    return [base64.b64encode(json.dumps(value).encode()).decode(), "str"]


def _case_json(path: Path) -> Path:
    data = json.loads((TESTS / "fixtures" / "tests.json").read_text(encoding="utf-8"))
    data["alternatives"].append(["uom", "water in m3, the CHP heat in MWh_th"])
    data["parameter_values"] += [["node", "water", "unit_of_measure", _b64("m3"), "uom"],
                                 ["node", "desal_heat", "unit_of_measure", _b64("MWh_th"), "uom"]]
    data["entities"].append(["group__node", ["electricity", "water"], None])
    base = [sa for sa in data["scenario_alternatives"] if sa[0] == DEFAULT_SCENARIO]
    chain = [[SCENARIO, alt, before] for _s, alt, before in base]
    chain[-1][2] = "uom"
    chain.append([SCENARIO, "uom", None])
    data["scenario_alternatives"] += chain
    data["scenarios"].append([SCENARIO, False, ""])
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _solve_and_write(db_url, scenario, work, solver_dir, *, subdir, write_methods, single_result=None):
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    steps = run_chain_from_db(input_db_url=db_url, scenario_name=scenario, work_folder=work,
                              solver_config_dir=solver_dir, warm=True, keep_solutions=True)
    last = next(reversed(steps.values()))
    write_outputs(scenario_name=scenario, output_location=str(work), subdir=subdir,
                  output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"), write_methods=write_methods,
                  fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  solution=last.solution, solve_name=last.solve_name,
                  solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                  flex_data_provider=last.flex_data_provider, single_result=single_result)


@pytest.fixture(scope="module")
def case(tmp_path_factory, test_solver_config_dir):
    import sys

    sys.path.insert(0, str(TESTS))
    from db_utils import json_to_db
    from flextool.update_flextool.db_migration import migrate_database

    root = tmp_path_factory.mktemp("uom")
    url = json_to_db(_case_json(root / "case.json"), root / "case.sqlite")
    migrate_database(url)
    work = root / "work"
    work.mkdir()
    _solve_and_write(url, SCENARIO, work, test_solver_config_dir, subdir=SCENARIO,
                     write_methods=["parquet", "csv", "excel", "spinedb"])
    return {"url": url, "work": work, "solver": test_solver_config_dir,
            "parquet": work / "output_parquet" / SCENARIO, "csv": work / "output_csv" / SCENARIO}


def _footer_units(case, key: str) -> dict[str, str]:
    return {k: v["unit"] for k, v in read_lean_parquet_column_meta(case["parquet"] / f"{key}.parquet").items()
            if v["semantics"] != "dimension"}


def _units_csv(folder: Path) -> list[dict]:
    with open(folder / "units.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_footer_units_follow_the_nodes(case) -> None:
    out = _footer_units(case, "unit_outputNode_dt_ee")
    assert out[str(("med_desal", "water"))] == "m3/h"
    assert out[str(("coal_plant", "west"))] == "MW"
    assert out[str(("bp_chp", "desal_heat"))] == "MW_th"
    assert _footer_units(case, "unit_inputNode_dt_ee")[str(("med_desal", "desal_heat"))] == "MW_th"
    assert _footer_units(case, "unit_outputNode_d_ee")[str(("med_desal", "water"))] == "m3/a"
    assert _footer_units(case, "node_state_dt_e") == {"water": "m3"}
    assert _footer_units(case, "node_inflow__dt")["water"] == "m3"
    prices = _footer_units(case, "node_prices_dt_e")
    # The case writes no price column for the water storage; the m3 price is in the pure test.
    assert prices == {"west": "CUR/MWh", "desal_heat": "CUR/MWh_th"}
    assert _footer_units(case, "nodeGroup_flows_dt_g")["electricity"] == MIXED


def test_capacity_rows_in_footer_csv_and_excel(case) -> None:
    rows = read_lean_parquet_row_units(case["parquet"] / "unit_capacity_ed_p.parquet")
    assert rows["med_desal"]["total"] == "m3/h"
    assert rows["coal_plant"]["total"] == "MW"
    # The back-pressure CHP shares its capacity between its outputs (owner decision S1).
    assert sorted(rows["bp_chp"]["total"].split(" + ")) == ["MW", "MW_th"]
    assert read_lean_parquet_row_units(case["parquet"] / "node_capacity_ed_p.parquet")["water"]["total"] == "m3"
    group_rows = read_lean_parquet_row_units(case["parquet"] / "nodeGroup_gdt_p.parquet")
    assert group_rows["electricity"]["1. Loss of load"] == MIXED
    assert group_rows["electricity"]["6. Curtailed VRE of potential VRE"] == ""
    # units.csv: one more row per (column, row entity), with a ``row`` column.
    units = _units_csv(case["csv"])
    desal = [r for r in units if r["row"] == "med_desal" and r["header_1"] == "total"
             and r["file"].startswith("unit_capacity")]
    assert [r["unit"] for r in desal] == ["m3/h"]
    # Excel: the Table of the unit capacity sheet ends in a ``unit of measure`` column.
    wb = openpyxl.load_workbook(next((case["work"] / "output_excel").glob("*.xlsx")))
    sheet = next(row[0] for row in wb["Units"].iter_rows(min_row=2, values_only=True)
                 if row[1] == "unit_capacity_ed_p")
    values = list(wb[sheet].iter_rows(values_only=True))
    header = list(values[0])
    pos = header.index("unit of measure")
    by_unit = {row[0]: row[pos] for row in values[1:] if row[0] is not None}
    assert by_unit["med_desal"] == "m3/h" and by_unit["coal_plant"] == "MW"


def test_results_database_gets_the_node_units(case) -> None:
    from spinedb_api import DatabaseMapping, from_database

    with DatabaseMapping("sqlite:///" + str(case["work"] / "results.sqlite")) as db:
        values = {tuple(v["entity_byname"]): from_database(v["value"], v["type"])
                  for v in db.find_parameter_values(entity_class_name="node",
                                                    parameter_definition_name="unit_of_measure")}
    assert values == {("water",): "m3", ("desal_heat",): "MWh_th"}


def test_replay_gives_the_same_units(case, tmp_path) -> None:
    """A parquet replay reads _unit_context.json and writes the same units.csv as the solve did."""
    from flextool.process_outputs.write_outputs import write_outputs

    assert (case["parquet"] / UNIT_CONTEXT_FILE).exists()
    replay = tmp_path / "replay"
    shutil.copytree(case["work"] / "output_parquet", replay / "output_parquet")
    write_outputs(scenario_name=SCENARIO, output_location=str(replay), subdir=SCENARIO,
                  output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"), write_methods=["csv"],
                  fallback_output_location=str(replay), read_parquet_dir=True)
    replayed = {(r["file"], r["row"], r["header_1"], r["header_2"]): r["unit"]
                for r in _units_csv(replay / "output_csv" / SCENARIO)}
    solved = {(r["file"], r["row"], r["header_1"], r["header_2"]): r["unit"] for r in _units_csv(case["csv"])}
    shared = set(replayed) & set(solved)
    assert any(k[2] == "med_desal" and k[3] == "water" for k in shared)
    assert any(k[1] == "med_desal" for k in shared)  # the row units of the capacity table
    assert {k: replayed[k] for k in shared} == {k: solved[k] for k in shared}


def test_rerun_with_mwh_nodes_deletes_the_context(case) -> None:
    """Critique M6: a run whose nodes are all MWh deletes _unit_context.json, also a single-result run and a run
    without parquet output."""
    work, pq_dir = case["work"], case["parquet"]
    assert (pq_dir / UNIT_CONTEXT_FILE).exists()
    single = ("node_state_dt_e", "node_state__dt", "None", "None", "None", "None")
    _solve_and_write(case["url"], DEFAULT_SCENARIO, work, case["solver"], subdir=SCENARIO,
                     write_methods=["parquet"], single_result=single)
    assert not (pq_dir / UNIT_CONTEXT_FILE).exists()
    _solve_and_write(case["url"], SCENARIO, work, case["solver"], subdir=SCENARIO, write_methods=["parquet"])
    assert (pq_dir / UNIT_CONTEXT_FILE).exists()
    # A run that writes no parquets deletes the file even with node units: they must not sit beside an
    # earlier run's parquets (WP2 review 3; the footers hold each run's units).
    _solve_and_write(case["url"], SCENARIO, work, case["solver"], subdir=SCENARIO, write_methods=["csv"])
    assert not (pq_dir / UNIT_CONTEXT_FILE).exists()


def test_comparison_keeps_each_scenarios_units(tmp_path) -> None:
    """Critique S7: a comparison keys each scenario's context by its folder name; the same column with different
    units in two scenarios keeps both units (joined in the Units sheet) and gets one note."""
    from flextool.process_outputs._excel_writer import write_excel_with_metadata
    from flextool.scenario_comparison.orchestrator import comparison_unit_context, unit_differences

    a, b = tmp_path / "A_1", tmp_path / "B"
    meta.write_unit_context(str(a), UnitContext(node_units={"water": "m3"}))
    b.mkdir()
    ctx = comparison_unit_context({"A_1": "x", "B": "y"}, [a, b])
    assert set(ctx) == {"A_1", "B"} and ctx["B"].is_default
    assert unit_differences([], comparison_unit_context({"B": "y"}, [b])) == []
    cols = pd.MultiIndex.from_tuples([("A_1", "water"), ("B", "water")], names=["scenario", "node"])
    df = pd.DataFrame([[1.0, 2.0]], columns=cols, index=pd.MultiIndex.from_tuples([("p1", "t1")],
                                                                                  names=["period", "time"]))
    assert [m.unit for m in resolve_columns("node_state_dt_e", df.columns, ctx=ctx)] == ["m3", "MWh"]
    notes = unit_differences([("node_state__dt", "node_state_dt_e", df)], ctx)
    assert len(notes) == 1 and "node_state_dt_e" in notes[0] and "water" in notes[0]
    path = tmp_path / "cmp.xlsx"
    write_excel_with_metadata(path, [("node_state__dt", "node_state_dt_e", df)], ctx=ctx)
    units_row = next(r for r in openpyxl.load_workbook(path)["Units"].iter_rows(min_row=2, values_only=True))
    assert units_row[2] == "MWh; m3 (by column)"


def test_replay_reads_the_footers(case, tmp_path) -> None:
    """WP2 review 3: a replay takes the units the run wrote into the footers (here without _unit_context.json),
    and the results database defines node.unit_of_measure with the default MWh (review 2)."""
    from spinedb_api import DatabaseMapping, from_database
    from flextool.process_outputs.write_outputs import write_outputs

    replay = tmp_path / "replay"
    shutil.copytree(case["work"] / "output_parquet", replay / "output_parquet")
    (replay / "output_parquet" / SCENARIO / UNIT_CONTEXT_FILE).unlink(missing_ok=True)
    write_outputs(scenario_name=SCENARIO, output_location=str(replay), subdir=SCENARIO,
                  output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"), write_methods=["csv"],
                  fallback_output_location=str(replay), read_parquet_dir=True)
    units = _units_csv(replay / "output_csv" / SCENARIO)
    desal = [r["unit"] for r in units if r["file"] == "unit__outputNode__dt.csv" and r["header_1"] == "med_desal"
             and r["header_2"] == "water"]
    assert desal == ["m3/h"]
    capacity = [r["unit"] for r in units if r["row"] == "med_desal" and r["file"].startswith("unit_capacity")
                and r["header_1"] == "total"]
    assert capacity == ["m3/h"]
    with DatabaseMapping("sqlite:///" + str(case["work"] / "results.sqlite")) as db:
        definition = db.find_parameter_definitions(entity_class_name="node", name="unit_of_measure")[0]
        assert from_database(definition["default_value"], definition["default_type"]) == "MWh"


def test_comparison_parquets_carry_units(tmp_path) -> None:
    """WP2 review 4: a combined comparison table's footer holds each scenario's column units."""
    from flextool.lean_parquet import write_lean_parquet
    from flextool.scenario_comparison.orchestrator import comparison_column_units, comparison_unit_context

    a, b = tmp_path / "A_1", tmp_path / "B"
    meta.write_unit_context(str(a), UnitContext(node_units={"water": "m3"}))
    b.mkdir()
    cols = pd.MultiIndex.from_tuples([("A_1", "water"), ("B", "water")], names=["scenario", "node"])
    df = pd.DataFrame([[1.0, 2.0]], columns=cols, index=pd.MultiIndex.from_tuples([("p1", "t1")],
                                                                                  names=["period", "time"]))
    col_meta, rows = comparison_column_units("node_state_dt_e", df, comparison_unit_context(
        {"A_1": "x", "B": "y"}, [a, b]))
    assert rows is None
    path = tmp_path / "node_state_dt_e.parquet"
    write_lean_parquet(df, path, column_metadata=col_meta)
    written = read_lean_parquet_column_meta(path)
    assert written[str(("A_1", "water"))]["unit"] == "m3"
    assert written[str(("B", "water"))]["unit"] == "MWh"


def test_plot_plans_follow_the_nodes(case) -> None:
    """The solve's plot plans fill the label tokens per subplot: the water node's flows in m3/h, the
    electricity node's in MWh/h, the nodeGroup over electricity and water "Mixed units of measure!"."""
    from flextool.plot_outputs.plan import load_plot_plan

    def labels(plan) -> set[str]:
        out: set[str] = set()
        for label in [plan.ylabel, plan.xlabel, *(plan.batch_ylabels or []), *(plan.batch_xlabels or [])]:
            out.update(label if isinstance(label, list) else [label])
        return out - {None}

    plans = case["parquet"] / "plot_plans"
    flows = load_plot_plan(plans, "unit_outputNode_dt_ee", "default")
    assert {"m3/h", "MWh/h", "MWh_th/h"} <= labels(flows)
    assert MIXED in labels(load_plot_plan(plans, "nodeGroup_flows_dt_g", "default"))
    assert "m3" in labels(load_plot_plan(plans, "node_state_dt_e", "default"))
