"""FlexTool 2 (.xlsm) → FlexTool 4 importer, end to end.

The FT2 workbook is generated here with openpyxl in the 18-sheet FT2 layout
(the layout of ``InputData/*.xlsm`` in the FT2 distribution), so the test
needs no file outside the repository.  One small model, 24 hourly steps:

* node ``elec/nodeA``, demand 60 MW every step;
* ``coal`` (min load + startup cost), ``wind`` (cf profile);
* ``battery`` (charger / discharger, ``invested storage (MWh)`` = 20,
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
    capacity margin on (its v56 name was renamed at v65) and one that gives
    the CHP 5 MW of invested capacity."""
    return write_workbook(path, {
        "Getting Started": [["FlexTool 2"]],
        "Sensitivity definitions": [
            ["Scenario definitions master:", "Sheet", "use_capacity_margin"],
            ["Margin", "master", 1],
            [],
            ["Scenario definitions units:", "Sheet", "unittype", "output node",
             "invested capacity (MW)"],
            ["MoreChp", "units", "chp", "nodeA", 5],
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
#: * user_constraint_prebuilt_nets_retirements: the CHP rows use
#:   constraint_cumulative_pre_built_capacity_coeff, which now nets earlier
#:   retirements; the importer writes no retirement (FT2 has none) -> n/a.
#: * dc_power_flow_lines_beside_dc_links, price_ladder_periods_from_solve_periods,
#:   construction_lead_time_commissioning, coarse_process_flows,
#:   stochastic_branch_period_years, ramp_branch_point_shutdown_once,
#:   retirement_of_existing_capacity_only: the importer writes no DC power flow,
#:   price ladder, lead time, new_stepduration, stochastics or retirement -> n/a.
#: * chained_solve_costs_from_run_start: cost reporting of an FT4 solve chain,
#:   not the meaning of an input -> n/a.
#: * acceptance_ceiling_follows_tolerance: solver acceptance -> n/a.
#: * capacity_margin_counts_by_flow: written because the importer maps an FT2
#:   capacity_margin_MW to capacity_margin_method manual (margin_group).  The
#:   new counting is FT2's own (flexModel capacityMarginNodeGroup): units at
#:   availability x capacity, the other units and transfers by their flow,
#:   exports on the demand side.  It is the pre-v79 counting that differed
#:   from FT2, so a new import has nothing to warn about -> removed like the rest.
#: The data-dependent markers (storage_fix_method_rewritten,
#: solve_step_series_combined, energy_margin_adder_per_original_step,
#: ladder_tier_labels_as_written, unnamed_maps_read_per_depth) are NOT written:
#: the importer gives every storage its method and writes no step series,
#: energy margin, ladder or unnamed map.  A marker added by a later migration
#: fails this test: desk-check it the same way, then add it here.
EXPECTED_MARKERS = sorted("flextool_behaviour_change:" + name for name in (
    "acceptance_ceiling_follows_tolerance", "capacity_margin_counts_by_flow",
    "chained_solve_costs_from_run_start",
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


def test_invested_storage_is_existing_beside_max_invest(imported: str) -> None:
    # FT2 storage capacity = storage + invested storage (20) + v_investStorage
    # with v_investStorage <= max invest (100) (flexModel.mod stateLimit).
    node = ("battery_nodeA_storage",)
    assert _values(imported, "node", "existing")[(node, "base")] == 20
    assert _values(imported, "node", "invest_max_total")[(node, "base")] == 100
    assert (node, "base") not in _values(imported, "node", "invest_min_total")


def test_storage_kw_per_kwh_link_follows_ft2(tmp_path: Path) -> None:
    # FT2 investStorage_kW_per_kWh: v_invest (MW) = ratio x v_investStorage
    # (MWh).  Ratio 0.25 (a 4 h storage): discharger invest - 0.25 x storage
    # invest = 0 (before: 0.25 x MW - MWh = 0, a 1/16 h storage).
    sheets = ft2_sheets()
    sheets["unit_type"].append(_unit_type("bat4h", **{
        "efficiency": 0.9, "eff charge": 0.9, "availability": 1,
        "inv.cost/kW": 300, "fixed kW/kWh ratio": 0.25}))
    sheets["units"].append(_unit(**{
        "unit type": "bat4h", "output grid": "elec", "output node": "nodeA",
        "max invest (MW)": 10, "max invest (MWh)": 40}))
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    url = f"sqlite:///{tmp_path / 'model.sqlite'}"
    import_old_flextool_xlsm(str(xlsm), url, strict=True)
    link = "bat4h_nodeA_storage_link"
    coeff = _values(url, "unit", "constraint_invested_capacity_coeff")
    discharger = coeff[(("bat4h_nodeA_discharger",), "base")]
    assert dict(zip(discharger.indexes, discharger.values))[link] == 1.0
    node = _values(url, "node", "constraint_invested_capacity_coeff")[(("bat4h_nodeA_storage",), "base")]
    assert dict(zip(node.indexes, node.values)) == {link: -0.25}


def test_invested_capacity_sensitivity_adds_to_capacity(imported: str) -> None:
    # Scenario MoreChp: invested capacity 5 on the 20 MW CHP -> C = 25 MW.
    existing = _values(imported, "unit", "existing")
    assert existing[(("chp_nodeA",), "base")] == 20
    assert existing[(("chp_nodeA",), "MoreChp")] == 25
    assert _values(imported, "unit", "invest_min_total") == {}


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
    # FT2 rows with C = capacity + invested capacity + v_invest (hand values
    # from flexModel.mod):  heat - 1.4 elec >= 0;  heat - 2.0 elec - 0.1 C <= 0;
    # heat - 1.6 C <= 0.  C is the capacity in service: pre-built (existing,
    # and what an earlier solve invested) + invested in the solve, so the
    # constants are 0.
    chp, elec, heat = "chp_nodeA", ("chp_nodeA", "nodeA"), ("chp_nodeA", "nodeH")
    sense = _values(imported, "constraint", "sense")
    constant = _values(imported, "constraint", "constant")
    rows = {"gt": ("greater_than", 0.0), "lt": ("less_than", 0.0),
            "output2_max": ("less_than", 0.0)}
    for suffix, (row_sense, row_constant) in rows.items():
        key = ((f"{chp}_{suffix}",), "base")
        assert sense[key] == row_sense
        assert constant[key] == pytest.approx(row_constant)
    flow = _values(imported, "unit__outputNode", "constraint_flow_coeff")
    assert dict(zip(flow[(heat, "base")].indexes, flow[(heat, "base")].values)) == {
        f"{chp}_gt": 1.0, f"{chp}_lt": 1.0, f"{chp}_output2_max": 1.0}
    assert dict(zip(flow[(elec, "base")].indexes, flow[(elec, "base")].values)) == {
        f"{chp}_gt": -1.4, f"{chp}_lt": -2.0}
    for param in ("constraint_cumulative_pre_built_capacity_coeff",
                  "constraint_invested_capacity_coeff"):
        coeffs = _values(imported, "unit", param)[((chp,), "base")]
        assert dict(zip(coeffs.indexes, coeffs.values)) == pytest.approx(
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
    ``capacity (MW)`` empty and ``invested capacity (MW)`` = 25, C = 25 MW: existing
    25 and no investment (before: existing 25 and a forced 25 MW investment).  The
    CHP's output2 row ``heat <= 1.6 C`` has constant 0 and -1.6 on the capacity in
    service (hand values).  The annuity report line lists only the peaker: the CHP
    has no investment cost, so FT2 adds no annuity for it."""
    sheets = ft2_sheets()
    sheets["unit_type"].append(_unit_type("peaker", **{
        "efficiency": 0.4, "availability": 1, "inv.cost/kW": 500}))
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
    report = import_old_flextool_xlsm(str(xlsm), url, strict=True)

    existing = _values(url, "unit", "existing")
    for unit in ("peaker_nodeA", "chp_new_nodeA"):
        assert existing[((unit,), "base")] == 25
        assert ((unit,), "base") not in _values(url, "unit", "invest_method")
    assert existing[(("coal_nodeA",), "base")] == 100
    chp = "chp_new_nodeA"
    assert _values(url, "constraint", "constant")[((f"{chp}_output2_max",), "base")] == pytest.approx(0.0)
    coeffs = _values(url, "unit", "constraint_cumulative_pre_built_capacity_coeff")[((chp,), "base")]
    assert dict(zip(coeffs.indexes, coeffs.values)) == pytest.approx({f"{chp}_output2_max": -1.6})
    found = {(u.sheet, u.column): u for u in report.unmapped}
    assert found[("units", "invested capacity (MW)")].entities == ["peaker_nodeA"]
    assert "annuity" in found[("units", "invested capacity (MW)")].reason


def test_output2_report_lines(tmp_path: Path) -> None:
    """An output2 without rows is free; constants of the older MW-form workbook are
    read as MW, and the report says so in 'How some inputs were read'."""
    sheets = ft2_sheets()
    sheets["units"][0] = UNITS_HEADER + ["output2 max capacity (MW)"]
    for row in sheets["units"][1:]:
        row.append(None)
    sheets["units"][-1][-1] = 800
    sheets["units"][-1][UNITS_HEADER.index("output2 gt constant")] = -100
    sheets["unit_type"].append(_unit_type("chp_free", efficiency=0.4, availability=1))
    sheets["units"].append(_unit(**{
        "unit type": "chp_free", "fuel": "coal", "output grid": "elec",
        "output node": "nodeA", "capacity (MW)": 10,
        "output2 grid": "heat", "output2 node": "nodeH"}) + [None])
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    report = import_old_flextool_xlsm(str(xlsm), f"sqlite:///{tmp_path / 'm.sqlite'}")
    found = {(u.sheet, u.column): u for u in report.unmapped}
    assert ("units", "output2 eq / gt / lt constant") not in found
    mw = [a for a in report.assumptions if "output2 eq / gt / lt constant" in a]
    assert len(mw) == 1 and "read as MW" in mw[0] and mw[0].endswith(": chp_nodeA.")
    free = found[("units", "output2 grid / node")]
    assert free.entities == ["chp_free_nodeA"]
    assert "free" in free.reason


# The units sheet of an older FT2 workbook: 'output2 max capacity (MW)' in place
# of 'output2 max capacity ratio', and the output2 constants are MW.
OLD_UNITS_HEADER = [
    "output2 max capacity (MW)" if h == "output2_max_capacity_ratio" else h
    for h in UNITS_HEADER] + ["output2 eq coeff", "output2 eq constant"]


@pytest.mark.solver
def test_old_header_output2_constants_are_mw(
    test_solver_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Older workbook (header 'output2 max capacity (MW)'): the CHP (20 MW) has
    heat = 1.0 elec + 5,  heat >= 1.0 elec - 4  and  heat <= 0 elec + 20, all
    constants in MW, and a plain cap heat <= 800 MW.  The LP rows have the MW
    values as their bounds and no capacity term.  Hand solve: CHP heat has no
    fuel and saves the gas boiler's 20 / 0.9 per MWh, CHP elec costs 10 / 0.4
    = 25 per MWh against 27 for the coal unit, so the CHP runs to the heat
    bound: heat 20 MW (binding), elec 15, fuel 15 / 0.4 = 37.5; the boiler
    makes the other 10 MW of the 30 MW heat demand.  Before: the constants
    were multiplied by the capacity (heat = elec + 100 > the demand)."""
    import polars as pl

    from flextool.engine import run_chain_from_db
    from tests.engine._lp_rows import row_bounds, row_terms
    from tests.engine._unit_db import build

    sheets = ft2_sheets()
    col = OLD_UNITS_HEADER.index
    width = len(OLD_UNITS_HEADER) - len(UNITS_HEADER)
    units = [OLD_UNITS_HEADER] + [row + [None] * width for row in sheets["units"][1:]]
    chp_row = units[-1]
    chp_row[col("output2 gt coeff")] = 1.0
    chp_row[col("output2 gt constant")] = -4
    chp_row[col("output2 lt coeff")] = None
    chp_row[col("output2 lt constant")] = 20
    chp_row[col("output2 eq coeff")] = 1.0
    chp_row[col("output2 eq constant")] = 5
    chp_row[col("output2 max capacity (MW)")] = 800
    boiler = [None] * len(OLD_UNITS_HEADER)
    for key, value in {"unit type": "boiler", "fuel": "gas", "output grid": "heat",
                       "output node": "nodeH", "capacity (MW)": 50}.items():
        boiler[col(key)] = value
    units.append(boiler)
    sheets["units"] = units
    sheets["unit_type"].append(_unit_type("boiler", efficiency=0.9, availability=1))
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    url = f"sqlite:///{tmp_path / 'model.sqlite'}"
    import_old_flextool_xlsm(str(xlsm), url, strict=True)

    chp = "chp_nodeA"
    assert ((chp,), "base") not in _values(url, "unit", "constraint_invested_capacity_coeff")
    assert ((chp,), "base") not in _values(
        url, "unit", "constraint_cumulative_pre_built_capacity_coeff")

    work = tmp_path / "run"
    work.mkdir()
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(work)
    steps = run_chain_from_db(url, "base", work_folder=work,
                              solver_config_dir=test_solver_config_dir,
                              keep_solutions=True)
    assert steps and all(step.optimal for step in steps.values())
    for step in steps.values():
        pb = build(step.flex_data)
        n_steps = step.flex_data.dt.height
        for family, row, bound in (
            ("process_constraint_equal", "eq", (5.0, 5.0)),
            ("process_constraint_greater_than", "gt", (-4.0, None)),
            ("process_constraint_less_than", "lt", (None, 20.0)),
            ("process_constraint_less_than", "output2_max", (None, 800.0)),
        ):
            bounds = row_bounds(pb, family, f"{chp}_{row},")
            assert len(bounds) == n_steps, (family, row)
            lower, upper = bound
            for lb, ub in bounds.values():
                if lower is not None:
                    assert lb == pytest.approx(lower)
                if upper is not None:
                    assert ub == pytest.approx(upper)
            for terms in row_terms(pb, family, f"{chp}_{row},").values():
                assert all(c.startswith("v_flow") for c in terms), terms

    raw = work / "output_raw"
    flow = pl.read_parquet(sorted(raw.glob("v_flow__dispatch*.parquet"))[-1])
    heat = flow[f"('{chp}', '{chp}', 'nodeH')"] * 20
    elec = flow[f"('{chp}', '{chp}', 'nodeA')"] * 20
    fuel = flow[f"('{chp}', 'coal_market', '{chp}')"] * 20
    boiler_heat = flow["('boiler_nodeH', 'gas_market', 'nodeH')"] * 50
    assert heat.to_list() == pytest.approx([20.0] * 24, abs=1e-6)
    assert elec.to_list() == pytest.approx([15.0] * 24, abs=1e-6)
    assert fuel.to_list() == pytest.approx([37.5] * 24, abs=1e-6)
    assert boiler_heat.to_list() == pytest.approx([10.0] * 24, abs=1e-6)
    slack = pl.read_parquet(sorted(raw.glob("vq_state_up__dispatch*.parquet"))[-1])
    assert slack["nodeH"].abs().max() == pytest.approx(0.0, abs=1e-9)


def _solve(url: str, solver_dir: Path, work: Path, monkeypatch: pytest.MonkeyPatch):
    from flextool.engine import run_chain_from_db

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(work)
    steps = run_chain_from_db(url, "base", work_folder=work, solver_config_dir=solver_dir)
    assert steps and all(step.optimal for step in steps.values())
    return work / "output_raw"


@pytest.mark.solver
def test_invested_chp_keeps_its_heat_in_the_dispatch_solve(
    test_solver_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invest solve, then dispatch solve.  CHP: no existing capacity, max invest
    50 MW, heat demand 40 MW and no other heat source.  FT2 (hand values):
    heat <= 1.6 C needs C = 40 / 1.6 = 25 MW; with C = 25,
    1.4 elec - 0.2 C <= heat = 40 <= 2.0 elec + 0.2 C gives
    17.5 <= elec <= 32.14; fuel = elec / 0.4.  Before: the dispatch solve's
    rows counted only capacity invested in that solve, so heat <= 0."""
    import polars as pl

    sheets = ft2_sheets()
    col = UNITS_HEADER.index
    chp_row = sheets["units"][-1]
    chp_row[col("capacity (MW)")] = None
    chp_row[col("max invest (MW)")] = 50
    chp_row[col("output2 gt constant")] = -0.2
    chp_row[col("output2 lt constant")] = 0.2
    sheets["unit_type"][-1] = _unit_type("chp", **{
        "efficiency": 0.4, "availability": 1, "inv.cost/kW": 1000,
        "lifetime": 30, "interest": 0.05})
    sheets["ts_energy"] = sheets["ts_energy"][:3] + [[t, None, 60, 40] for t in STEPS]
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    url = f"sqlite:///{tmp_path / 'model.sqlite'}"
    import_old_flextool_xlsm(str(xlsm), url, strict=True)
    work = tmp_path / "run"
    work.mkdir()
    raw = _solve(url, test_solver_config_dir, work, monkeypatch)

    invest = pl.read_parquet(raw / "v_invest__invest.parquet")
    # v_invest is per unit of unitsize (1000 MW for an investment-only unit).
    assert invest["chp_nodeA"][0] * 1000 == pytest.approx(25.0, abs=1e-6)
    flow = pl.read_parquet(raw / "v_flow__dispatch_p2020.parquet")
    heat = flow["('chp_nodeA', 'chp_nodeA', 'nodeH')"] * 1000
    elec = flow["('chp_nodeA', 'chp_nodeA', 'nodeA')"] * 1000
    fuel = flow["('chp_nodeA', 'coal_market', 'chp_nodeA')"] * 1000
    assert heat.to_list() == pytest.approx([40.0] * 24, abs=1e-6)
    assert elec.min() >= 17.5 - 1e-6 and elec.max() <= 45 / 1.4 + 1e-6
    assert fuel.to_list() == pytest.approx((elec / 0.4).to_list(), abs=1e-6)
    slack = pl.read_parquet(raw / "vq_state_up__dispatch_p2020.parquet")
    assert slack["nodeH"].abs().max() == pytest.approx(0.0, abs=1e-9)


@pytest.mark.solver
def test_dispatch_only_workbook_keeps_invested_capacity(
    test_solver_config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mode_invest 0: FT2 solves dispatch only, with C = capacity + invested
    capacity.  A gas unit with only 60 MW of invested capacity, demand 300 MW
    (more than all other units together, 193 MW at most), loss of load 10000
    per MWh against 20 / 0.4 = 50 for gas: the unit runs at 60 MW in every
    step (hand values).  Before: a forced investment that a dispatch-only
    solve never makes, so 0 MW."""
    import polars as pl

    sheets = ft2_sheets()
    for row in sheets["master"]:
        if row[0] == "mode_invest":
            row[1] = 0
    sheets["unit_type"].append(_unit_type("peaker", efficiency=0.4, availability=1))
    sheets["units"].append(_unit(**{
        "unit type": "peaker", "fuel": "gas", "output grid": "elec",
        "output node": "nodeA", "invested capacity (MW)": 60}))
    sheets["ts_energy"] = sheets["ts_energy"][:3] + [[t, None, 300, 30] for t in STEPS]
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    url = f"sqlite:///{tmp_path / 'model.sqlite'}"
    import_old_flextool_xlsm(str(xlsm), url, strict=True)
    work = tmp_path / "run"
    work.mkdir()
    raw = _solve(url, test_solver_config_dir, work, monkeypatch)

    flow = pl.read_parquet(raw / "v_flow__dispatch.parquet")
    peaker = flow["('peaker_nodeA', 'gas_market', 'nodeA')"] * 60
    assert peaker.to_list() == pytest.approx([60.0] * 24, abs=1e-6)


def test_storage_max_invest_derived_through_the_kw_kwh_ratio(tmp_path: Path) -> None:
    """FT2 p_unit_max_invest_MW / _MWh (flexModel.mod): a limit not given is the
    other one through the ratio (MW = ratio x MWh).  Ratio 0.25: max invest 10 MW
    alone gives 40 MWh; 40 MWh alone gives 10 MW (before: the side not given had no
    invest_method, so the storage link forced both to 0).  A storage with a
    max invest (MWh), no cost per kWh and no ratio invests in FT4 at no cost, as in
    FT2 (#218): it gets its invest_method and limit, and the report has no item."""
    sheets = ft2_sheets()
    for name in ("bat_mw", "bat_mwh"):
        sheets["unit_type"].append(_unit_type(name, **{
            "efficiency": 0.9, "eff charge": 0.9, "availability": 1,
            "inv.cost/kW": 300, "fixed kW/kWh ratio": 0.25}))
    sheets["unit_type"].append(_unit_type("bat_free", **{
        "efficiency": 0.9, "eff charge": 0.9, "availability": 1, "inv.cost/kW": 300}))
    sheets["units"].append(_unit(**{
        "unit type": "bat_mw", "output grid": "elec", "output node": "nodeA",
        "max invest (MW)": 10}))
    sheets["units"].append(_unit(**{
        "unit type": "bat_mwh", "output grid": "elec", "output node": "nodeA",
        "max invest (MWh)": 40}))
    sheets["units"].append(_unit(**{
        "unit type": "bat_free", "output grid": "elec", "output node": "nodeA",
        "max invest (MW)": 10, "max invest (MWh)": 40}))
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    master = write_workbook(tmp_path / "flexTool.xlsx", {
        "Sensitivity definitions": [
            ["Scenario definitions units:", "Sheet", "unittype", "output node",
             "max invest (MW)", "max invest (MWh)"],
            ["BigMW", "units", "bat_mw", "nodeA", 20, None],
            ["BigMWh", "units", "bat_mwh", "nodeA", None, 80],
            [],
        ]})
    url = f"sqlite:///{tmp_path / 'model.sqlite'}"
    report = import_old_flextool_xlsm(str(xlsm), url, master=str(master), strict=True)
    assert not report.failed

    node_max = _values(url, "node", "invest_max_total")
    unit_max = _values(url, "unit", "invest_max_total")
    node_method = _values(url, "node", "invest_method")
    assert node_max[(("bat_mw_nodeA_storage",), "base")] == pytest.approx(40.0)
    assert node_method[(("bat_mw_nodeA_storage",), "base")] == "invest_total"
    assert node_max[(("bat_mw_nodeA_storage",), "BigMW")] == pytest.approx(80.0)
    for part in ("charger", "discharger"):
        assert unit_max[((f"bat_mwh_nodeA_{part}",), "base")] == pytest.approx(10.0)
        assert unit_max[((f"bat_mwh_nodeA_{part}",), "BigMWh")] == pytest.approx(20.0)
        assert unit_max[((f"bat_mw_nodeA_{part}",), "BigMW")] == pytest.approx(20.0)
    assert node_max[(("bat_mwh_nodeA_storage",), "BigMWh")] == pytest.approx(80.0)
    assert node_method[(("bat_free_nodeA_storage",), "base")] == "invest_total"
    assert node_max[(("bat_free_nodeA_storage",), "base")] == pytest.approx(40.0)
    found = {(u.sheet, u.column): u for u in report.unmapped}
    assert ("units", "max invest (MWh)") not in found


def _write_sensitivity_master(path: Path) -> Path:
    """Overrides whose interplay the import must get right in one scenario."""
    return write_workbook(path, {
        "Sensitivity definitions": [
            ["Scenario definitions master:", "Sheet", "use_ramps"],
            ["NoRamps", "master", 0],
            ["Ramps", "master", 1],
            [],
            ["Scenario definitions unit_type:", "Sheet", "unit type", "inv.cost/kWh"],
            ["CheapBat", "unit_type", "battery", 50],
            [],
            ["Scenario definitions units:", "Sheet", "unittype", "output node",
             "invested storage (MWh)", "max invest (MWh)", "storage (MWh)",
             "invested capacity (MW)"],
            ["NoInvested", "units", "battery", "nodeA", 0, None, None, None],
            ["NoStorage", "units", "battery", "nodeA", 0, 0, None, None],
            ["StorageZero", "units", "battery", "nodeA", None, None, 0, None],
            ["MoreCap", "units", "battery", "nodeA", None, None, None, 5],
            ["MoreCap", "units", "coal", "nodeA", None, None, None, 5],
            [],
            ["Scenario definitions nodeNode:", "Sheet", "node1", "node2",
             "cap.rightward (MW)", "invested capacity (MW)"],
            ["Both", "nodeNode", "nodeA", "nodeB", 70, 20],
            ["NoInv", "nodeNode", "nodeA", "nodeB", None, 0],
            [],
        ]})


@pytest.fixture(scope="module")
def sensitivity_imported(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[str, ImportReport]:
    """The test workbook with ramps on coal and a connection nodeA -> nodeB
    (50 / 40 MW, invested capacity 10, inv.cost/kW 100), and the overrides of
    :func:`_write_sensitivity_master`."""
    sheets = ft2_sheets()
    sheets["unit_type"][1][UNIT_TYPE_HEADER.index("ramp up (p.u. per min)")] = 0.5
    sheets["gridNode"].append(["elec", "nodeB", None, None, None, None, None,
                               None, None, 0, 0, 1])
    sheets["nodeNode"].append(["elec", "nodeA", "nodeB", 50, 40, 10, None, 0.01,
                               100, 40, 0.05, None, 0])
    folder = tmp_path_factory.mktemp("ft2_sensitivities")
    xlsm = write_workbook(folder / "model.xlsx", sheets)
    master = _write_sensitivity_master(folder / "flexTool.xlsx")
    url = f"sqlite:///{folder / 'model.sqlite'}"
    report = import_old_flextool_xlsm(str(xlsm), url, master=str(master), strict=True)
    assert not report.failed
    return url, report


def test_zero_invested_storage_keeps_an_investable_storage(
    sensitivity_imported: tuple[str, ImportReport],
) -> None:
    """Battery: invested storage 20, max invest (MWh) 100.  Invested storage 0
    leaves FT2 C = 0 + v_investStorage <= 100: still a storage (before: balance,
    the investment lost).  Also max invest 0: nothing to store, balance.  Storage
    (MWh) 0: C = 0 + 20, a storage."""
    url, _ = sensitivity_imported
    node = ("battery_nodeA_storage",)
    node_type = _values(url, "node", "node_type")
    existing = _values(url, "node", "existing")
    assert node_type[(node, "NoInvested")] == "storage"
    assert existing[(node, "NoInvested")] == 0
    assert node_type[(node, "NoStorage")] == "balance"
    assert _values(url, "node", "invest_max_total")[(node, "NoStorage")] == 0
    assert node_type[(node, "StorageZero")] == "storage"
    assert existing[(node, "StorageZero")] == 20
    assert _values(url, "node", "invest_cost")[(node, "CheapBat")] == 50
    assert (("battery_nodeA_discharger",), "CheapBat") not in _values(url, "unit", "invest_cost")


def test_use_ramps_sensitivity_switches_the_ramp_method(
    sensitivity_imported: tuple[str, ImportReport],
) -> None:
    """use_ramps 0 writes the v56 method 'none' (before: 'no_ramp', not in v56,
    and the import crashed)."""
    url, _ = sensitivity_imported
    method = _values(url, "unit__outputNode", "ramp_method")
    edge = ("coal_nodeA", "nodeA")
    assert method[(edge, "NoRamps")] == "none"
    assert method[(edge, "Ramps")] == "ramp_limit"


def test_connection_capacity_and_invested_capacity_in_one_scenario(
    sensitivity_imported: tuple[str, ImportReport],
) -> None:
    """FT2 C = larger direction + invested capacity.  Base 50 + 10 = 60; scenario
    Both (rightward 70, invested 20) 90, written once; NoInv 50 + 0."""
    url, _ = sensitivity_imported
    existing = _values(url, "connection", "existing")
    conn = ("nodeA_nodeB",)
    assert existing[(conn, "base")] == 60
    assert existing[(conn, "Both")] == 90
    assert existing[(conn, "NoInv")] == 50


def test_annuity_report_covers_sensitivity_capacity(
    sensitivity_imported: tuple[str, ImportReport],
) -> None:
    """Invested capacity with an investment cost has an FT2 annuity: battery
    (storage 20, 100 per kWh), the connection (100 per kW) and, from MoreCap, the
    battery's 5 MW (300 per kW) and the connection's 20 MW in Both.  Not coal: no
    investment cost.  NoInv's 0 MW is no capacity."""
    _, report = sensitivity_imported
    found = {(u.sheet, u.column): u.entities for u in report.unmapped}
    assert found[("units", "invested storage (MWh)")] == ["battery_nodeA"]
    assert found[("nodeNode", "invested capacity (MW)")] == ["nodeA_nodeB"]
    assert found[("Sensitivity definitions / units", "invested capacity (MW)")] == [
        "battery_nodeA [MoreCap]"]
    assert found[("Sensitivity definitions / nodeNode", "invested capacity (MW)")] == [
        "nodeA_nodeB [Both]"]
    assert ("units", "invested capacity (MW)") not in found


def test_interest_zero_is_written_and_blank_interest_is_not(tmp_path: Path) -> None:
    """FT2's annuity is -PMT(interest, lifetime, 1), which at an interest of 0 is 1 / lifetime: FlexTool's
    discount_rate 0.  So a typed 0 is written as discount_rate 0 (unit, storage and connection); a blank interest
    writes nothing (the schema default)."""
    sheets = ft2_sheets()
    col = UNIT_TYPE_HEADER.index
    for row in sheets["unit_type"][1:]:
        if row[0] == "coal":
            row[col("interest")] = 0
        if row[0] == "battery":
            row[col("interest")] = 0
        if row[0] == "wind":
            row[col("interest")] = None
    sheets["nodeNode"].append(["elec", "nodeA", "nodeB", 10, 10, None, 50, 0.01, 100, 40, 0, None, 0])
    sheets["gridNode"].append(["elec", "nodeB", None, None, None, None, None, None, None, 0, 0, 1])
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    url = f"sqlite:///{tmp_path / 'm.sqlite'}"
    report = import_old_flextool_xlsm(str(xlsm), url)
    assert not report.failed
    unit = {name[0]: value for (name, _alt), value in _values(url, "unit", "discount_rate").items()}
    assert unit["coal_nodeA"] == 0
    assert "wind_nodeA" not in unit
    storage_rates = [v for k, v in unit.items() if k.startswith("battery")]
    assert storage_rates and all(v == 0 for v in storage_rates)
    assert all(v == 0 for v in _values(url, "node", "discount_rate").values())
    conn = _values(url, "connection", "discount_rate")
    assert list(conn.values()) == [0]


def test_ft2_annuity_matches_pmt() -> None:
    """Excel's -PMT(r, n, 1): r / (1 - (1 + r)^-n), and 1 / n at r = 0 (hand values)."""
    from flextool.process_inputs.write_old_flextool_to_db import ft2_annuity

    assert ft2_annuity(0.0, 25) == pytest.approx(0.04)
    assert ft2_annuity(0.08, 30) == pytest.approx(0.0888274334, rel=1e-9)


def test_annuity_typed_over_the_formula_is_reported(tmp_path: Path) -> None:
    """An annuity that is not -PMT(interest, lifetime, 1) is reported (FlexTool 4 ignores the column); one that
    matches, at interest 0.05 or 0, is not."""
    sheets = ft2_sheets()
    col = UNIT_TYPE_HEADER.index
    for row in sheets["unit_type"][1:]:
        if row[0] == "coal":  # 0.05, 40 years: the formula gives 0.0582782; 0.1 was typed over it
            row[col("annuity")] = 0.1
        if row[0] == "wind":  # 0.05, 25 years: the formula's value
            row[col("annuity")] = 0.0709524572
        if row[0] == "battery":  # interest 0, 15 years: 1 / 15
            row[col("interest")] = 0
            row[col("annuity")] = 1 / 15
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    report = import_old_flextool_xlsm(str(xlsm), f"sqlite:///{tmp_path / 'm.sqlite'}")
    found = {(u.sheet, u.column): u for u in report.unmapped}
    assert found[("unit_type", "annuity")].entities == ["coal"]
    assert "typed over the formula" in found[("unit_type", "annuity")].reason


def test_workbook_without_time_in_use_refused(tmp_path: Path) -> None:
    """FT2 solves the ts_time steps with in_use (flexModel.mod time_in_use): with none
    there is nothing to solve.  The import says so and writes no database (before: a
    SpineDB error on an empty timeset map, after the database was created)."""
    sheets = ft2_sheets()
    sheets["ts_time"] = sheets["ts_time"][:1] + [[t, 0, 0, 0, 0] for t in STEPS]
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    target = tmp_path / "model.sqlite"
    with pytest.raises(ValueError, match="no time step in use: column 'in_use'"):
        import_old_flextool_xlsm(str(xlsm), f"sqlite:///{target}")
    assert not target.exists()
