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
  with ts_unit ``min_generation`` 0.2 and ``max_generation`` 0.8.

Expected values come from the FT2 definitions (``flexModel.mod``
``fix_unit_generation`` / ``min_generation_ts`` / ``upwardLimitNotOnline``:
the ts value is a share of capacity), not from importer output:
must-run flow = share × 40 MW = 20 / 24 MW.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest
from spinedb_api import DatabaseMapping, Map

from flextool.process_inputs.write_old_flextool_to_db import (
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
        ["time_in_years", 1], ["time_period_duration", 60],
        ["reserve_duration", 0], ["use_capacity_margin", 0], ["use_online", 1],
        ["use_ramps", 0], ["use_non_synchronous", 0], ["use_inertia_limit", 0],
        ["mode_invest", 1], ["mode_dispatch", 1], ["print_duration", 0],
    ]
    unit_type = [
        UNIT_TYPE_HEADER,
        _unit_type("coal", **{"efficiency": 0.4, "min load": 0.3,
                              "eff at min load": 0.35, "O&M cost/MWh": 2,
                              "availability": 1, "startup cost": 50,
                              "lifetime": 40, "interest": 0.05}),
        _unit_type("wind", efficiency=1, availability=1, lifetime=25, interest=0.05),
        _unit_type("battery", **{"efficiency": 0.9, "eff charge": 0.9,
                                 "availability": 1, "inv.cost/kW": 300,
                                 "inv.cost/kWh": 100, "lifetime": 15,
                                 "interest": 0.05}),
        _unit_type("gas_mustrun", efficiency=0.5, availability=1),
        _unit_type("engine", efficiency=0.45, availability=1),
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
        "ts_energy": [[None, "grid", "elec"], [None, "node", "nodeA"], ["time"]]
        + [[t, None, 60] for t in STEPS],
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
def imported(tmp_path_factory: pytest.TempPathFactory) -> str:
    folder = tmp_path_factory.mktemp("ft2_import")
    xlsm = write_workbook(folder / "model.xlsx", _sheets_with_margin())
    master = write_master_workbook(folder / "flexTool.xlsx")
    url = f"sqlite:///{folder / 'model.sqlite'}"
    report = import_old_flextool_xlsm(str(xlsm), url, master=str(master), strict=True)
    assert not report.failed
    return url


def test_import_reaches_current_version_without_markers(imported: str) -> None:
    with DatabaseMapping(imported) as db:
        version = db.get_parameter_definition_item(entity_class_name="model", name="version")
        from spinedb_api import from_database
        assert int(from_database(version["default_value"], version["default_type"])) == FLEXTOOL_DB_VERSION
        markers = [md["name"] for md in db.get_items("metadata")
                   if str(md["name"]).startswith("flextool_behaviour_change:")]
    assert markers == []


def test_forced_storage_investment_wins_over_max_invest(imported: str) -> None:
    node = ("battery_nodeA_storage",)
    assert _values(imported, "node", "invest_min_total")[(node, "base")] == 20
    assert _values(imported, "node", "invest_max_total")[(node, "base")] == 20


def test_storage_start_keeps_its_ft2_meaning(imported: str) -> None:
    # FT2 fixes the first state at 'storage start' x capacity and wraps the
    # balance; the v56 default method would have become fix_nothing at v77.
    node = ("battery_nodeA_storage",)
    assert _values(imported, "node", "storage_start_end_method")[(node, "base")] == "fix_start"
    assert _values(imported, "node", "storage_state_start")[(node, "base")] == 0.5
    assert _values(imported, "node", "storage_binding_method")[(node, "base")] == "bind_within_solve"


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


def test_master_workbook_rejected(tmp_path: Path) -> None:
    master = write_master_workbook(tmp_path / "flexTool.xlsx")
    with pytest.raises(ValueError, match="master workbook.*--master"):
        import_old_flextool_xlsm(str(master), f"sqlite:///{tmp_path / 'x.sqlite'}")
    assert not (tmp_path / "x.sqlite").exists()


def test_report_lists_untranslated_inputs(tmp_path: Path) -> None:
    sheets = ft2_sheets()
    master = {row[0]: row for row in sheets["master"]}
    master["time_in_years"][1] = 0.5
    master["curtailment_penalty"][1] = 100
    sheets["unit_type"].append(_unit_type("heat_pump", **{"conversion eff": 3}))
    sheets["units"][1][UNITS_HEADER.index("use min online")] = 1
    xlsm = write_workbook(tmp_path / "model.xlsx", sheets)
    report = import_old_flextool_xlsm(str(xlsm), f"sqlite:///{tmp_path / 'm.sqlite'}")
    lines = [u.line() for u in report.unmapped]
    assert lines[0].startswith("master 'time_in_years = 0.5'")
    found = {(u.sheet, u.column): u.entities for u in report.unmapped}
    assert ("master", "curtailment penalty") in found
    assert found[("unit_type", "conversion eff")] == ["heat_pump"]
    assert found[("units", "use min online")] == ["coal_nodeA"]
    assert "use min online" in report.format()


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
    for slack in ("vq_state_up", "vq_state_down"):
        balance = pl.read_parquet(raw / f"{slack}__dispatch_p2020.parquet")["nodeA"]
        assert balance.abs().max() == pytest.approx(0.0, abs=1e-9)
