"""FlexTool 2 (.xlsm) → FlexTool 4 importer, end to end.

The FT2 workbook is generated here with openpyxl in the 18-sheet FT2 layout
(the layout of ``InputData/*.xlsm`` in the FT2 distribution), so the test
needs no file outside the repository.  One small model, 24 hourly steps:

* node ``elec/nodeA``, demand 60 MW every step;
* ``coal`` (min load + startup cost), ``wind`` (cf profile);
* ``battery`` (charger / discharger, ``invested storage (MWh)`` = 20 forced,
  ``max invest (MWh)`` = 100, ``storage start`` = 0.5);
* ``gas_mustrun``: 40 MW, ``fix unit generation`` = 1 with ts_unit
  ``fix_generation`` 0.5 / 0.6 alternating;
* ``engine``: 30 MW, ``use min generation`` / ``use max generation`` = 1
  with ts_unit ``min_generation`` 0.2 and ``max_generation`` 0.8;
* ``chp``: 20 MW back-pressure CHP on coal (efficiency 0.4) with output2
  ``heat/nodeH`` (heat demand 30 MW): output2 gt coeff 1.4, lt coeff 2.0 with
  lt constant 0.1, max capacity ratio 1.6;
* ``coal`` has availability 0.9 (FT2: capacity margin only);
* ``time_in_years`` = 24 / 8760, the hours the 24 steps cover.

Expected values come from the FT2 definitions (``flexModel.mod``
``fix_unit_generation`` / ``min_generation_ts`` / ``upwardLimitNotOnline``:
the ts value is a share of capacity), not from importer output:
must-run flow = share × 40 MW = 20 / 24 MW.  CHP (``flexModel.mod``
``output2_*_output1_times_x``, ``upwardLimitOutput2``, ``fueluse``): with
C = 20 MW, 1.4 × elec <= heat <= 2.0 × elec + 0.1 × C and heat <= 1.6 × C;
fuel = elec / 0.4 (output2 has no fuel and is outside the capacity limit).
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest
from spinedb_api import DatabaseMapping, Map

from flextool.process_inputs.write_old_flextool_to_db import (
    ImportReport,
    _require_master_workbook,
    import_old_flextool_xlsm,
)
from flextool.update_flextool import FLEXTOOL_DB_VERSION

STEPS = [f"t{i:04d}" for i in range(24)]
FIX_SHARE = [0.5 if i % 2 == 0 else 0.6 for i in range(24)]

UNIT_TYPE_HEADER = [
    "unit type", "efficiency", "min load", "eff at min load",
    "ramp up (p.u. per min)", "ramp down (p.u. per min)", "O&M cost/MWh",
    "availability", "max reserve", "inertia constant (MWs/MW)",
    "fixed cost/kW/year", "inv.cost/kW", "inv.cost/kWh", "fixed kW/kWh ratio",
    "conversion eff", "startup cost", "min uptime (h)", "min downtime (h)",
    "eff charge", "self discharge loss", "lifetime", "interest", "annuity",
    "non synchronous",
]

UNITS_HEADER = [
    "unitGroup", "unit type", "fuel", "cf profile", "inflow", "input grid",
    "input node", "output grid", "output node", "capacity (MW)",
    "invested capacity (MW)", "max invest (MW)", "storage (MWh)",
    "invested storage (MWh)", "max invest (MWh)", "storage start",
    "storage finish", "reserve increase ratio", "use efficiency time series",
    "fix unit generation", "use min generation", "use max generation",
    "use min online", "inflow multiplier", "output2 grid", "output2 node",
    "output2 gt coeff", "output2 gt constant", "output2 lt coeff",
    "output2 lt constant", "output2_max_capacity_ratio",
    "fueluse increase eq x output2",
]


def _unit_type(name: str, **params: float) -> list:
    row = [name] + [None] * (len(UNIT_TYPE_HEADER) - 1)
    for key, value in params.items():
        row[UNIT_TYPE_HEADER.index(key)] = value
    return row


def _unit(**cols) -> list:
    row = [None] * len(UNITS_HEADER)
    for key, value in cols.items():
        row[UNITS_HEADER.index(key)] = value
    return row


def ft2_sheets() -> dict[str, list[list]]:
    """The sheets of the test workbook as rows (tests edit them before writing)."""
    master = [
        ["parameter", "value"],
        ["co2_cost", 0], ["loss_of_load_penalty", 10000],
        ["loss_of_reserves_penalty", 20000], ["lack_of_inertia_penalty", 30000],
        ["curtailment_penalty", 0], ["lack_of_capacity_penalty", 5000],
        ["time_in_years", 24 / 8760], ["time_period_duration", 60],
        ["reserve_duration", 0], ["use_capacity_margin", 0], ["use_online", 1],
        ["use_ramps", 0], ["use_non_synchronous", 0], ["use_inertia_limit", 0],
        ["mode_invest", 1], ["mode_dispatch", 1], ["print_duration", 0],
    ]
    unit_type = [
        UNIT_TYPE_HEADER,
        _unit_type("coal", **{"efficiency": 0.4, "min load": 0.3,
                              "eff at min load": 0.35, "O&M cost/MWh": 2,
                              "availability": 0.9, "startup cost": 50,
                              "lifetime": 40, "interest": 0.05}),
        _unit_type("wind", efficiency=1, availability=1, lifetime=25, interest=0.05),
        _unit_type("battery", **{"efficiency": 0.9, "eff charge": 0.9,
                                 "availability": 1, "inv.cost/kW": 300,
                                 "inv.cost/kWh": 100, "lifetime": 15,
                                 "interest": 0.05}),
        _unit_type("gas_mustrun", efficiency=0.5, availability=1),
        _unit_type("engine", efficiency=0.45, availability=1),
        _unit_type("chp", efficiency=0.4, availability=1),
    ]
    units = [
        UNITS_HEADER,
        _unit(**{"unit type": "coal", "fuel": "coal", "output grid": "elec",
                 "output node": "nodeA", "capacity (MW)": 100}),
        _unit(**{"unit type": "wind", "cf profile": "wind_A",
                 "output grid": "elec", "output node": "nodeA",
                 "capacity (MW)": 50}),
        _unit(**{"unit type": "battery", "output grid": "elec",
                 "output node": "nodeA", "capacity (MW)": 10,
                 "invested storage (MWh)": 20, "max invest (MWh)": 100,
                 "storage start": 0.5}),
        _unit(**{"unit type": "gas_mustrun", "fuel": "gas",
                 "output grid": "elec", "output node": "nodeA",
                 "capacity (MW)": 40, "fix unit generation": 1}),
        _unit(**{"unit type": "engine", "fuel": "gas", "output grid": "elec",
                 "output node": "nodeA", "capacity (MW)": 30,
                 "use min generation": 1, "use max generation": 1}),
        _unit(**{"unit type": "chp", "fuel": "coal", "output grid": "elec",
                 "output node": "nodeA", "capacity (MW)": 20,
                 "output2 grid": "heat", "output2 node": "nodeH",
                 "output2 gt coeff": 1.4, "output2 lt coeff": 2.0,
                 "output2 lt constant": 0.1,
                 "output2_max_capacity_ratio": 1.6}),
    ]
    return {
        "info": [["FlexTool 2 test workbook"]],
        "master": master,
        "nodeGroup": [["nodeGroup", "capacity margin (MW)", "non synchronous share",
                       "inertia limit (MWs)", "use ts_reserve", "use dynamic reserve"]],
        "gridNode": [
            ["grid", "node", "nodeGroup", "nodeGroup2", "nodeGroup3", "demand (MWh)",
             "import (MWh)", "capacity margin (MW)", "non synchronous share",
             "use ts_reserve", "use dynamic reserve", "print results"],
            ["elec", "nodeA", None, None, None, None, None, None, None, 0, 0, 1],
            ["heat", "nodeH", None, None, None, None, None, None, None, 0, 0, 1],
        ],
        "unit_type": unit_type,
        "fuel": [["fuel", "fuel (price/MWh)", "CO2 content (t/MWh)"],
                 ["coal", 10, 0.34], ["gas", 20, 0.2]],
        "unitGroup": [["unitGroup", "max invest MW", "min invest MW",
                       "max invest MWh", "min invest MWh", "print results"]],
        "units": units,
        "nodeNode": [["grid", "node1", "node2", "cap.rightward (MW)",
                      "cap.leftward (MW)", "invested capacity (MW)",
                      "max invest (MW)", "loss", "inv.cost/kW", "lifetime",
                      "interest", "annuity", "HVDC"]],
        "ts_cf": [[None, "cf_profile", "wind_A"], ["Time"]]
        + [[t, None, 0.3] for t in STEPS],
        "ts_inflow": [[None, "inflow"], ["time"]],
        "ts_energy": [[None, "grid", "elec", "heat"],
                      [None, "node", "nodeA", "nodeH"], ["time"]]
        + [[t, None, 60, 30] for t in STEPS],
        "ts_import": [[None, "grid"], [None, "node"], ["time"]],
        "ts_reserve_node": [[None, "node"], ["Time"]],
        "ts_reserve_nodeGroup": [[None, "nodeGroup"], ["Time"]],
        "ts_unit": [
            [None, "grid", "elec", "elec", "elec"],
            [None, "node", "nodeA", "nodeA", "nodeA"],
            [None, "unit", "gas_mustrun", "engine", "engine"],
            [None, "unit_ts_param", "fix_generation", "min_generation",
             "max_generation"],
            ["Time"],
        ] + [[t, None, FIX_SHARE[i], 0.2, 0.8] for i, t in enumerate(STEPS)],
        "ts_time": [["time", "in_use", "time_jump", "in_use_invest",
                     "time_jump_invest"]]
        + [[t, 1, 1, 1, 1] for t in STEPS],
    }


def write_workbook(path: Path, sheets: dict[str, list[list]]) -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(name)
        for row in rows:
            ws.append(row)
    wb.save(path)
    return path


def write_master_workbook(path: Path) -> Path:
    """An FT2 master workbook: no data sheets, one sensitivity that turns the
    capacity margin on (its v56 name was renamed at v65)."""
    return write_workbook(path, {
        "Getting Started": [["FlexTool 2"]],
        "Sensitivity definitions": [
            ["Scenario definitions master:", "Sheet", "use_capacity_margin"],
            ["Margin", "master", 1],
            [],
        ],
    })


def _values(url: str, cls: str, param: str) -> dict[tuple[tuple[str, ...], str], object]:
    with DatabaseMapping(url) as db:
        return {
            (tuple(pv["entity_byname"]), pv["alternative_name"]): pv["parsed_value"]
            for pv in db.get_items("parameter_value", entity_class_name=cls,
                                   parameter_definition_name=param)
        }


def _sheets_with_margin() -> dict[str, list[list]]:
    sheets = ft2_sheets()
    sheets["nodeGroup"].append(["margin_group", 10, None, None, 0, 0])
    sheets["gridNode"][1][2] = "margin_group"
    return sheets


@pytest.fixture(scope="module")
def imported_with_report(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, ImportReport]:
    folder = tmp_path_factory.mktemp("ft2_import")
    xlsm = write_workbook(folder / "model.xlsx", _sheets_with_margin())
    master = write_master_workbook(folder / "flexTool.xlsx")
    url = f"sqlite:///{folder / 'model.sqlite'}"
    report = import_old_flextool_xlsm(str(xlsm), url, master=str(master), strict=True)
    assert not report.failed
    return url, report


@pytest.fixture(scope="module")
def imported(imported_with_report: tuple[str, ImportReport]) -> str:
    return imported_with_report[0]


#: Behaviour-change markers the migration writes on a fresh import of the
#: test workbook (the import removes them).  DESK CHECK of each against the
#: importer's v56 mapping -- does the change alter what an FT2 input means?
#: * storage_default_fix_nothing: yes, the v56 default fix_start becomes
#:   fix_nothing.  The importer always writes storage_start_end_method, so the
#:   imported meaning does not depend on the default (test_storage_start_finish).
#: * user_constraint_prebuilt_nets_retirements: the importer writes no
#:   pre-built capacity coefficient (CHP rows and storage links use
#:   constraint_invested_capacity_coeff, FT2 has no retirements) -> n/a.
#: * dc_power_flow_lines_beside_dc_links, price_ladder_periods_from_solve_periods,
#:   construction_lead_time_commissioning, coarse_process_flows,
#:   stochastic_branch_period_years, ramp_branch_point_shutdown_once,
#:   retirement_of_existing_capacity_only: the importer writes no DC power flow,
#:   price ladder, lead time, new_stepduration, stochastics or retirement -> n/a.
#: * chained_solve_costs_from_run_start: cost reporting of an FT4 solve chain,
#:   not the meaning of an input -> n/a.
#: * acceptance_ceiling_follows_tolerance: solver acceptance -> n/a.
#: The data-dependent markers (storage_fix_method_rewritten,
#: solve_step_series_combined, energy_margin_adder_per_original_step,
#: ladder_tier_labels_as_written, unnamed_maps_read_per_depth) are NOT written:
#: the importer gives every storage its method and writes no step series,
#: energy margin, ladder or unnamed map.  A marker added by a later migration
#: fails this test: desk-check it the same way, then add it here.
EXPECTED_MARKERS = sorted("flextool_behaviour_change:" + name for name in (
    "acceptance_ceiling_follows_tolerance", "chained_solve_costs_from_run_start",
    "coarse_process_flows", "construction_lead_time_commissioning",
    "dc_power_flow_lines_beside_dc_links", "price_ladder_periods_from_solve_periods",
    "ramp_branch_point_shutdown_once", "retirement_of_existing_capacity_only",
    "stochastic_branch_period_years", "storage_default_fix_nothing",
    "user_constraint_prebuilt_nets_retirements",
))


def test_import_reaches_current_version_without_markers(
    imported_with_report: tuple[str, ImportReport],
) -> None:
    imported, report = imported_with_report
    with DatabaseMapping(imported) as db:
        version = db.get_parameter_definition_item(entity_class_name="model", name="version")
        from spinedb_api import from_database
        assert int(from_database(version["default_value"], version["default_type"])) == FLEXTOOL_DB_VERSION
        markers = [md["name"] for md in db.get_items("metadata")
                   if str(md["name"]).startswith("flextool_behaviour_change:")]
    assert markers == []
    assert report.markers_removed == EXPECTED_MARKERS


def test_forced_storage_investment_wins_over_max_invest(imported: str) -> None:
    node = ("battery_nodeA_storage",)
    assert _values(imported, "node", "invest_min_total")[(node, "base")] == 20
    assert _values(imported, "node", "invest_max_total")[(node, "base")] == 20


#: (storage start, storage finish) -> FT2 meaning (flexModel.mod stateStart,
#: stateEnd, storageBalance: a non-zero value fixes the state at that share of
#: the capacity, 0 / empty fixes nothing; the balance wraps unless both are set).
STORAGE_CASES = {
    "st_start": ((0.5, None), "fix_start", "bind_within_solve", 0.5, None),
    "st_zero": ((0, None), "fix_nothing", "bind_within_solve", None, None),
    "st_finish": ((None, 0.4), "fix_end", "bind_within_solve", None, 0.4),
    "st_both": ((0.5, 0.4), "fix_start_end", "bind_forward_only", 0.5, 0.4),
}


@pytest.fixture(scope="module")
def storage_imported(tmp_path_factory: pytest.TempPathFactory) -> str:
    sheets = ft2_sheets()
    for name, ((start, finish), *_rest) in STORAGE_CASES.items():
        sheets["unit_type"].append(_unit_type(name, **{
            "efficiency": 0.9, "eff charge": 0.9, "availability": 1}))
        sheets["units"].append(_unit(**{
            "unit type": name, "output grid": "elec", "output node": "nodeA",
            "capacity (MW)": 10, "storage (MWh)": 20, "storage start": start,
            "storage finish": finish}))
    folder = tmp_path_factory.mktemp("ft2_storage")
    xlsm = write_workbook(folder / "model.xlsx", sheets)
    url = f"sqlite:///{folder / 'model.sqlite'}"
    import_old_flextool_xlsm(str(xlsm), url, strict=True)
    return url


@pytest.mark.parametrize("unit_type", list(STORAGE_CASES))
def test_storage_start_finish(storage_imported: str, unit_type: str) -> None:
    # Before C-7: a start of 0 was written and fixed the state at 0, a finish
    # was not read, and both set kept the wrap (bind_within_solve).  A start
    # alone was already right after migration (v77 kept fix_start for a node
    # with its own start value).
    _, method, binding, start, finish = STORAGE_CASES[unit_type]
    node = (f"{unit_type}_nodeA_storage",)
    key = (node, "base")
    assert _values(storage_imported, "node", "storage_start_end_method")[key] == method
    assert _values(storage_imported, "node", "storage_binding_method")[key] == binding
    assert _values(storage_imported, "node", "storage_state_start").get(key) == start
    assert _values(storage_imported, "node", "storage_state_end").get(key) == finish


def test_must_run_and_generation_limits_become_profiles(imported: str) -> None:
    methods = _values(imported, "unit__node__profile", "profile_method")
    assert methods[(("gas_mustrun_nodeA", "nodeA", "gas_mustrun_nodeA_fix_generation"), "base")] == "fixed"
    assert methods[(("engine_nodeA", "nodeA", "engine_nodeA_min_generation"), "base")] == "lower_limit"
    assert methods[(("engine_nodeA", "nodeA", "engine_nodeA_max_generation"), "base")] == "upper_limit"
    profiles = _values(imported, "profile", "profile")
    fixed = profiles[(("gas_mustrun_nodeA_fix_generation",), "base")]
    assert isinstance(fixed, Map)
    assert list(fixed.indexes) == STEPS
    assert list(fixed.values) == FIX_SHARE
    assert set(profiles[(("engine_nodeA_min_generation",), "base")].values) == {0.2}
    assert set(profiles[(("engine_nodeA_max_generation",), "base")].values) == {0.8}


def test_sensitivity_survives_migration(imported: str) -> None:
    # Written as v56 has_capacity_margin = yes, then migrated (v65 rename).
    methods = _values(imported, "group", "capacity_margin_method")
    assert (("margin_group",), "Margin") in methods
    assert methods[(("margin_group",), "Margin")] == "manual"


def test_existing_current_version_target_refused(imported: str, tmp_path: Path) -> None:
    xlsm = write_workbook(tmp_path / "model.xlsx", ft2_sheets())
    with pytest.raises(ValueError, match=f"version {FLEXTOOL_DB_VERSION}"):
        import_old_flextool_xlsm(str(xlsm), imported, purge=False)


def test_non_flextool_database_target_refused_and_kept(tmp_path: Path) -> None:
    target = tmp_path / "other.sqlite"
    url = f"sqlite:///{target}"
    with DatabaseMapping(url, create=True) as db:
        db.add_entity_class(name="thing")
        db.commit_session("a user's database")
    xlsm = write_workbook(tmp_path / "model.xlsx", ft2_sheets())
    with pytest.raises(ValueError, match="not a FlexTool database"):
        import_old_flextool_xlsm(str(xlsm), url)
    assert target.exists()


def test_master_workbook_rejected(tmp_path: Path) -> None:
    master = write_master_workbook(tmp_path / "flexTool.xlsx")
    with pytest.raises(ValueError, match="master workbook.*--master"):
        import_old_flextool_xlsm(str(master), f"sqlite:///{tmp_path / 'x.sqlite'}")
    assert not (tmp_path / "x.sqlite").exists()


def test_chp_output2_rows_follow_ft2(imported: str) -> None:
    # FT2 rows with C = capacity 20 MW + v_invest (hand values from
    # flexModel.mod):  heat - 1.4 elec >= 0;  heat - 2.0 elec - 0.1 C <= 0;
    # heat - 1.6 C <= 0.  The existing 20 MW part of C moves to the constant.
    chp, elec, heat = "chp_nodeA", ("chp_nodeA", "nodeA"), ("chp_nodeA", "nodeH")
    sense = _values(imported, "constraint", "sense")
    constant = _values(imported, "constraint", "constant")
    rows = {"gt": ("greater_than", 0.0), "lt": ("less_than", 2.0),
            "output2_max": ("less_than", 32.0)}
    for suffix, (row_sense, row_constant) in rows.items():
        key = ((f"{chp}_{suffix}",), "base")
        assert sense[key] == row_sense
        assert constant[key] == pytest.approx(row_constant)
    flow = _values(imported, "unit__outputNode", "constraint_flow_coeff")
    assert dict(zip(flow[(heat, "base")].indexes, flow[(heat, "base")].values)) == {
        f"{chp}_gt": 1.0, f"{chp}_lt": 1.0, f"{chp}_output2_max": 1.0}
    assert dict(zip(flow[(elec, "base")].indexes, flow[(elec, "base")].values)) == {
        f"{chp}_gt": -1.4, f"{chp}_lt": -2.0}
    invested = _values(imported, "unit", "constraint_invested_capacity_coeff")[((chp,), "base")]
    assert dict(zip(invested.indexes, invested.values)) == pytest.approx(
        {f"{chp}_lt": -0.1, f"{chp}_output2_max": -1.6})
    # output2 has no fuel and is outside the capacity limit in FT2.
    coeff = _values(imported, "unit__outputNode", "conversion_flow_coeff")
    assert coeff[(heat, "base")] == 0.0
    assert (elec, "base") not in coeff


def test_availability_is_not_a_dispatch_derating(
    imported_with_report: tuple[str, ImportReport],
) -> None:
    # FT2 uses availability only in the capacity margin (flexModel.mod
    # capacityMarginNode); FT4 unit.availability would derate every flow.
    imported, report = imported_with_report
    assert _values(imported, "unit", "availability") == {}
    found = {(u.sheet, u.column): u.entities for u in report.unmapped}
    assert found[("unit_type", "availability")] == ["coal_nodeA 0.9"]


def test_time_in_years_matching_the_steps_is_not_reported(
    imported_with_report: tuple[str, ImportReport],
) -> None:
    # The workbook has time_in_years = 24 / 8760: the 24 steps of 1 h.
    _, report = imported_with_report
    assert not [u for u in report.unmapped if u.column.startswith("time_in_years")]


def test_report_lists_untranslated_inputs(tmp_path: Path) -> None:
    sheets = ft2_sheets()
    master = {row[0]: row for row in sheets["master"]}
    master["time_in_years"][1] = 1  # 24 h counted as one year
    master["curtailment_penalty"][1] = 100
    sheets["unit_type"].append(_unit_type("heat_pump", **{"conversion eff": 3}))
    sheets["units"].append(_unit(**{
        "unit type": "heat_pump", "input grid": "elec", "input node": "nodeA",
        "output grid": "elec", "output node": "nodeA", "capacity (MW)": 5,
        "use min generation": 1}))
    col = UNITS_HEADER.index
    sheets["units"][1][col("use min online")] = 1  # coal
    sheets["units"][2][col("use max generation")] = 1  # wind: has a cf profile
    sheets["units"][1][col("use min generation")] = 1  # coal: no series
    sheets["units"][3][col("fix unit generation")] = 1  # battery: storage
    # ts_unit min_online of a unit whose flag is 0: FT2 ignores it.
    for i, row in enumerate(sheets["ts_unit"]):
        row.append(["elec", "nodeA", "engine", "min_online", None][i] if i < 5 else 0.3)
    # cf in use only in the first half (dispatch): FT2 mean scaling 0.3 / 0.2.
    cf = sheets["ts_cf"]
    for i, row in enumerate(cf[2:]):
        row[2] = 0.2 if i < 12 else 0.4
    for i, row in enumerate(sheets["ts_time"][1:]):
        row[1] = 1 if i < 12 else 0
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    report = import_old_flextool_xlsm(str(xlsm), f"sqlite:///{tmp_path / 'm.sqlite'}")
    found = {(u.sheet, u.column): u for u in report.unmapped}

    first = report.unmapped[0]
    assert first.column == "time_in_years = 1"
    # FT2 weighs operating costs by 1 h / 1 year per step, FT4 by
    # 1 h / (24 h / 8760 h) per step: 8760 / 24 = 365 times more.
    assert "365 times" in first.reason
    assert ("master", "curtailment penalty") in found
    assert found[("unit_type", "conversion eff")].entities == ["heat_pump"]
    assert found[("units", "use min online")].entities == ["coal_nodeA"]
    assert "min_online" not in {column for sheet, column in found if sheet == "ts_unit"}
    by_unit = {(u.column, tuple(u.entities)): u.reason for u in report.unmapped
               if u.sheet == "units"}
    assert "ignores the maximum generation" in by_unit[("use max generation", ("wind_nodeA",))]
    assert "no 'min_generation' series" in by_unit[("use min generation", ("coal_nodeA",))]
    assert "storage unit" in by_unit[("fix unit generation", ("battery_nodeA",))]
    assert "minimum on the input flow" in by_unit[("use min generation", ("heat_pump_nodeA",))]
    assert "1.5 in the dispatch run" in found[("ts_cf", "wind_A")].reason
    text = report.format()
    assert "These FlexTool 2 inputs were not carried over" in text
    assert "Time and costs:" in text and "Units:" in text
    assert "FT2" not in text and "FT4" not in text


def test_require_master_workbook_rejects_an_input_file(tmp_path: Path) -> None:
    xlsm = write_workbook(tmp_path / "model.xlsx", ft2_sheets())
    with pytest.raises(ValueError, match="not a FlexTool 2 master workbook"):
        _require_master_workbook(str(xlsm))


@pytest.mark.solver
def test_imported_model_solves_with_must_run(
    imported: str,
    test_solver_config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import polars as pl

    from flextool.engine import run_chain_from_db

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(tmp_path)
    steps = run_chain_from_db(imported, "base", work_folder=tmp_path,
                              solver_config_dir=test_solver_config_dir)
    assert steps and all(step.optimal for step in steps.values())
    raw = tmp_path / "output_raw"
    flow = pl.read_parquet(raw / "v_flow__dispatch_p2020.parquet")
    # v_flow is per unit of unitsize; without virtual_unitsize the unitsize is
    # the existing capacity (40 MW), so MW = v_flow * 40.
    must_run = flow["('gas_mustrun_nodeA', 'gas_market', 'nodeA')"] * 40
    assert must_run.to_list() == pytest.approx([40 * s for s in FIX_SHARE], abs=1e-6)
    engine = flow["('engine_nodeA', 'gas_market', 'nodeA')"] * 30
    assert engine.min() >= 0.2 * 30 - 1e-6 and engine.max() <= 0.8 * 30 + 1e-6
    # CHP (C = 20 MW, unitsize 20): heat meets the 30 MW demand,
    # 1.4 elec <= heat <= 2 elec + 2, and fuel = elec / 0.4 (no heat fuel).
    chp_elec = flow["('chp_nodeA', 'chp_nodeA', 'nodeA')"] * 20
    chp_heat = flow["('chp_nodeA', 'chp_nodeA', 'nodeH')"] * 20
    chp_fuel = flow["('chp_nodeA', 'coal_market', 'chp_nodeA')"] * 20
    assert chp_heat.to_list() == pytest.approx([30.0] * 24, abs=1e-6)
    assert ((chp_heat - 1.4 * chp_elec).min()) >= -1e-6
    assert ((2.0 * chp_elec + 2.0 - chp_heat).min()) >= -1e-6
    assert chp_fuel.to_list() == pytest.approx((chp_elec / 0.4).to_list(), abs=1e-6)
    for slack in ("vq_state_up", "vq_state_down"):
        slacks = pl.read_parquet(raw / f"{slack}__dispatch_p2020.parquet")
        for node in ("nodeA", "nodeH"):
            assert slacks[node].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_invested_capacity_without_capacity_is_not_counted_twice(tmp_path: Path) -> None:
    """FT2 capacity C = capacity + invested capacity + v_invest (flexModel.mod).  With
    ``capacity (MW)`` empty and ``invested capacity (MW)`` = 25, C = 25 MW: the import
    forces 25 MW of investment, so ``existing`` must be empty (0), not 25 as well.
    The CHP's output2 row ``heat <= 1.6 C`` then has constant 1.6 × 0 = 0 and
    the invested part -1.6 on the forced 25 MW (hand values)."""
    sheets = ft2_sheets()
    sheets["unit_type"].append(_unit_type("peaker", efficiency=0.4, availability=1))
    sheets["unit_type"].append(_unit_type("chp_new", efficiency=0.4, availability=1))
    sheets["units"].append(_unit(**{
        "unit type": "peaker", "fuel": "gas", "output grid": "elec",
        "output node": "nodeA", "invested capacity (MW)": 25}))
    sheets["units"].append(_unit(**{
        "unit type": "chp_new", "fuel": "coal", "output grid": "elec",
        "output node": "nodeA", "invested capacity (MW)": 25,
        "output2 grid": "heat", "output2 node": "nodeH",
        "output2_max_capacity_ratio": 1.6}))
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    url = f"sqlite:///{tmp_path / 'model.sqlite'}"
    import_old_flextool_xlsm(str(xlsm), url, strict=True)

    existing = _values(url, "unit", "existing")
    for unit in ("peaker_nodeA", "chp_new_nodeA"):
        assert existing.get(((unit,), "base"), 0) == 0
        assert _values(url, "unit", "invest_min_total")[((unit,), "base")] == 25
        assert _values(url, "unit", "invest_max_total")[((unit,), "base")] == 25
    assert existing[(("coal_nodeA",), "base")] == 100
    chp = "chp_new_nodeA"
    assert _values(url, "constraint", "constant")[((f"{chp}_output2_max",), "base")] == pytest.approx(0.0)
    invested = _values(url, "unit", "constraint_invested_capacity_coeff")[((chp,), "base")]
    assert dict(zip(invested.indexes, invested.values)) == pytest.approx({f"{chp}_output2_max": -1.6})
