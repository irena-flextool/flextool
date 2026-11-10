"""Phase 3 how-to example augmentations of ``tests/fixtures/tests.json``.

Adds the scenarios of two how-to sections:

* **Must-run units** (``docs/how_to/thermal_chp.md``, "How to make a unit run
  all the time"): eight scenarios ``must_run_*`` that force a unit or a fleet
  to run with a profile floor, a user constraint on the flow, a user
  constraint on the online count, a flowGroup floor and a heat demand, plus
  two control cases (no must-run, ``min_load`` alone).
* **Emission trajectories** (``docs/how_to/emissions.md``, "How to compare
  emission trajectories"): three scenarios ``co2_*_measures`` with one
  ``co2_max_period`` trajectory each (without / with existing / with
  additional measures) on top of the same four-period system.
* **Cogeneration with desalination** (``docs/how_to/thermal_chp.md``, "How to
  model cogeneration with desalination"): four scenarios ``desal_*``, a
  backpressure CHP feeding a thermal desalination unit, the same with an
  extraction CHP, a reverse-osmosis investment and a water storage.
* **CSP with thermal storage** (``docs/how_to/investment.md``, "How to model
  CSP with thermal storage"): five scenarios ``csp_*``, a solar field, a heat
  storage and a turbine, existing, with a fixed design (solar multiple and
  storage hours tied by user constraints), with an optimised design, and
  with a gas boiler as backup.

Everything is append-only: new alternatives, entities and scenarios only.
No value is added to an existing alternative, and no new member is added to
an existing group or flowGroup, so every existing scenario keeps its data.
New units are turned on in their own alternative (the ``unit`` class is
inactive by default).

Idempotent: the rows go in with ``import_data``, which skips rows that are
already there, so re-running the script leaves ``tests.json`` unchanged.

After running it, regenerate the examples database JSON::

    python tests/fixtures/_augment_phase3_examples.py
    python -m flextool.update_flextool.generate_canonical templates_examples
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from db_utils import db_to_json, json_to_db  # noqa: E402

from flextool.update_flextool.db_handle import open_database_mapping  # noqa: E402
from spinedb_api import Array, Map, import_data  # noqa: E402

TESTS_JSON = REPO_ROOT / "tests" / "fixtures" / "tests.json"

STEAM_UNITS = ("steam_1", "steam_2", "steam_3")
PERIODS = ("p2020", "p2025", "p2030", "p2035")
MULTI_PERIOD_SOLVE = "y2020_2035_2day_dispatch"
DISPATCH_SOLVE = "y2020_2day_dispatch"
TIMESTEPS = [f"t{i:04d}" for i in range(1, 73)]


def _constraint_map(name: str, coeff: float = 1.0) -> Map:
    return Map([name], [coeff], index_name="constraint")


def _period_map(values: tuple[float, ...]) -> Map:
    return Map(list(PERIODS), [float(v) for v in values], index_name="period")


def _time_map(values: list[float]) -> Map:
    """A map over the 72 time steps of timeline ``y2020``."""
    assert len(values) == len(TIMESTEPS)
    return Map(list(TIMESTEPS), [float(v) for v in values], index_name="time")


def _invest_periods() -> Array:
    return Array(["p2020"], index_name="period")


def _chain_rows(scenario: str, chain: list[str]) -> list[list]:
    """``scenario_alternatives`` rows in the export layout: each alternative
    names the one ranked above it (the last names none)."""
    return [
        [scenario, alt, chain[i + 1] if i + 1 < len(chain) else None]
        for i, alt in enumerate(chain)
    ]


# ---------------------------------------------------------------------------
# Must-run units (E-1)
# ---------------------------------------------------------------------------


def _must_run_data() -> dict[str, list]:
    alternatives = [
        ["must_run_system",
         "Must-run how-to: constant 300 MW demand at west; wind costs 1 CUR/MWh."],
        ["must_run_fleet",
         "Must-run how-to: three identical 100 MW coal steam units steam_1..3 "
         "(other_operational_cost 0, 1, 2 CUR/MWh breaks the tie)."],
        ["must_run_profile",
         "Must-run how-to: a 500 MW coal unit baseload_plant whose output is at "
         "least 0.4 of its capacity (profile lower_limit)."],
        ["must_run_constraint_one",
         "Must-run how-to: user constraint coal_plant output >= 150 MW."],
        ["must_run_fleet_output_floor",
         "Must-run how-to: user constraint on the fleet output, sum >= 100 MW."],
        ["must_run_fleet_two_online",
         "Must-run how-to: binary online steam units with min_load 0.5 and a "
         "user constraint on the online count, sum >= 2 units."],
        ["must_run_flowgroup",
         "Must-run how-to: flowGroup steam_fleet with min_instant_flow 100 MW."],
        ["must_run_min_load_only",
         "Must-run how-to: binary online steam units with min_load 0.5 and no "
         "must-run (control case: min_load is not a must-run)."],
    ]
    entities: list[list] = [
        ["profile", "must_run_share", None],
        # A new unit, not coal_plant: unit__node__profile is active by default and its
        # profile_method defaults to upper_limit, so a (coal_plant, west, must_run_share)
        # entity would cap coal_plant at 0 in every existing scenario (no profile value).
        ["unit", "baseload_plant", None],
        ["unit__inputNode", ["baseload_plant", "coal_market"], None],
        ["unit__outputNode", ["baseload_plant", "west"], None],
        ["unit__node__profile", ["baseload_plant", "west", "must_run_share"], None],
        ["constraint", "coal_plant_floor", None],
        ["constraint", "steam_fleet_output_floor", None],
        ["constraint", "steam_fleet_two_online", None],
        ["flowGroup", "steam_fleet", None],
    ]
    entity_alternatives: list[list] = [["unit", ["baseload_plant"], "must_run_profile", True]]
    values: list[list] = [
        # must_run_system
        ["node", "west", "inflow", -300.0, "must_run_system"],
        ["unit__outputNode", ["wind_plant", "west"], "other_operational_cost", 1.0,
         "must_run_system"],
        # must_run_profile
        ["unit", "baseload_plant", "conversion_method", "constant_efficiency", "must_run_profile"],
        ["unit", "baseload_plant", "efficiency", 0.4, "must_run_profile"],
        ["unit", "baseload_plant", "existing", 500.0, "must_run_profile"],
        ["profile", "must_run_share", "profile", 0.4, "must_run_profile"],
        ["unit__node__profile", ["baseload_plant", "west", "must_run_share"], "profile_method",
         "lower_limit", "must_run_profile"],
        # must_run_constraint_one
        ["constraint", "coal_plant_floor", "sense", "greater_than", "must_run_constraint_one"],
        ["constraint", "coal_plant_floor", "constant", 150.0, "must_run_constraint_one"],
        ["unit__outputNode", ["coal_plant", "west"], "constraint_flow_coeff",
         _constraint_map("coal_plant_floor"), "must_run_constraint_one"],
        # must_run_fleet_output_floor
        ["constraint", "steam_fleet_output_floor", "sense", "greater_than",
         "must_run_fleet_output_floor"],
        ["constraint", "steam_fleet_output_floor", "constant", 100.0,
         "must_run_fleet_output_floor"],
        # must_run_fleet_two_online
        ["constraint", "steam_fleet_two_online", "sense", "greater_than",
         "must_run_fleet_two_online"],
        ["constraint", "steam_fleet_two_online", "constant", 2.0, "must_run_fleet_two_online"],
        ["solve", "y2020_2day_dispatch", "solver_mip_gap", 0.0, "must_run_fleet_two_online"],
        # must_run_flowgroup
        ["flowGroup", "steam_fleet", "min_instant_flow", 100.0, "must_run_flowgroup"],
        # must_run_min_load_only
        ["solve", "y2020_2day_dispatch", "solver_mip_gap", 0.0, "must_run_min_load_only"],
    ]
    for k, unit in enumerate(STEAM_UNITS):
        entities += [
            ["unit", unit, None],
            ["unit__inputNode", [unit, "coal_market"], None],
            ["unit__outputNode", [unit, "west"], None],
            ["flowGroup__unit__node", ["steam_fleet", unit, "west"], None],
        ]
        entity_alternatives.append(["unit", [unit], "must_run_fleet", True])
        values += [
            ["unit", unit, "conversion_method", "constant_efficiency", "must_run_fleet"],
            ["unit", unit, "efficiency", 0.4, "must_run_fleet"],
            ["unit", unit, "existing", 100.0, "must_run_fleet"],
            ["unit__outputNode", [unit, "west"], "other_operational_cost", float(k),
             "must_run_fleet"],
            ["unit__outputNode", [unit, "west"], "constraint_flow_coeff",
             _constraint_map("steam_fleet_output_floor"), "must_run_fleet_output_floor"],
            ["unit", unit, "constraint_online_coeff",
             _constraint_map("steam_fleet_two_online"), "must_run_fleet_two_online"],
        ]
        for alt in ("must_run_fleet_two_online", "must_run_min_load_only"):
            values += [
                ["unit", unit, "startup_method", "binary", alt],
                ["unit", unit, "min_load", 0.5, alt],
            ]

    base = ["init", "west", "coal", "wind", "must_run_system"]
    scenarios = {
        "must_run_none": base,
        "must_run_profile": base + ["must_run_profile"],
        "must_run_constraint_one": base + ["must_run_constraint_one"],
        "must_run_fleet_output_floor": base + ["must_run_fleet", "must_run_fleet_output_floor"],
        "must_run_fleet_two_online": base + ["must_run_fleet", "must_run_fleet_two_online"],
        "must_run_flowgroup": base + ["must_run_fleet", "must_run_flowgroup"],
        "must_run_heat_demand": base + ["coal_chp", "heat"],
        "must_run_min_load_only": base + ["must_run_fleet", "must_run_min_load_only"],
    }
    return {
        "alternatives": alternatives,
        "entities": entities,
        "entity_alternatives": entity_alternatives,
        "parameter_values": values,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Emission trajectories (E-4)
# ---------------------------------------------------------------------------


def _co2_trajectory_data() -> dict[str, list]:
    alternatives = [
        ["co2_trajectory_system",
         "Emission-trajectory how-to: four 5-year periods p2020..p2035 on the 2-day "
         "timeset, constant 300 MW demand at west, a carbon-free clean_plant at "
         "80 CUR/MWh and a per-period CO2 limit on group co2_limit."],
        ["co2_without_measures",
         "Emission-trajectory how-to: CO2 limit 3.0 Mt/a in every period (does not bind)."],
        ["co2_existing_measures",
         "Emission-trajectory how-to: CO2 limit 2.0, 1.8, 1.6, 1.4 Mt/a."],
        ["co2_additional_measures",
         "Emission-trajectory how-to: CO2 limit 2.0, 1.5, 1.0, 0.5 Mt/a."],
    ]
    entities = [
        ["solve", MULTI_PERIOD_SOLVE, None],
        ["unit", "clean_plant", None],
        ["unit__outputNode", ["clean_plant", "west"], None],
    ]
    entity_alternatives = [["unit", ["clean_plant"], "co2_trajectory_system", True]]
    sys_alt = "co2_trajectory_system"
    values: list[list] = [
        ["model", "flexTool", "solves",
         Array([MULTI_PERIOD_SOLVE], index_name="sequence_index"), sys_alt],
        ["solve", MULTI_PERIOD_SOLVE, "period_timeset",
         Map(list(PERIODS), ["2day"] * len(PERIODS), index_name="period"), sys_alt],
        ["solve", MULTI_PERIOD_SOLVE, "realized_periods",
         Array(list(PERIODS), index_name="period"), sys_alt],
        ["solve", MULTI_PERIOD_SOLVE, "years_represented",
         _period_map((5.0, 5.0, 5.0, 5.0)), sys_alt],
        ["solve", MULTI_PERIOD_SOLVE, "solve_mode", "single_solve", sys_alt],
        ["node", "west", "inflow", -300.0, sys_alt],
        ["unit", "clean_plant", "conversion_method", "none", sys_alt],
        ["unit", "clean_plant", "existing", 500.0, sys_alt],
        ["unit__outputNode", ["clean_plant", "west"], "other_operational_cost", 80.0, sys_alt],
        ["group", "co2_limit", "co2_method", "period", sys_alt],
        ["group", "co2_limit", "co2_max_period", _period_map((3.0e6,) * 4),
         "co2_without_measures"],
        ["group", "co2_limit", "co2_max_period", _period_map((2.0e6, 1.8e6, 1.6e6, 1.4e6)),
         "co2_existing_measures"],
        ["group", "co2_limit", "co2_max_period", _period_map((2.0e6, 1.5e6, 1.0e6, 0.5e6)),
         "co2_additional_measures"],
    ]
    base = ["init", "west", "coal", sys_alt]
    scenarios = {
        name: base + [name]
        for name in ("co2_without_measures", "co2_existing_measures", "co2_additional_measures")
    }
    return {
        "alternatives": alternatives,
        "entities": entities,
        "entity_alternatives": entity_alternatives,
        "parameter_values": values,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Cogeneration with desalination (E-2)
# ---------------------------------------------------------------------------


def _desalination_data() -> dict[str, list]:
    alternatives = [
        ["desal_system",
         "Desalination how-to: constant 300 MW demand at west; a backpressure CHP bp_chp "
         "(coal, heat = 2 x power) feeds desal_heat; a thermal desalination unit med_desal "
         "(12 m3/MWh_th, 3000 m3/h) serves a water demand of 3000 m3/h."],
        ["desal_extraction",
         "Desalination how-to: bp_chp off; an extraction CHP ext_chp with an operating "
         "region (power >= 0.5 x heat, heat <= 0.6 x capacity) instead."],
        ["desal_ro_invest",
         "Desalination how-to: a reverse-osmosis unit ro_desal (250 m3/MWh_e) that may be "
         "invested up to 2000 m3/h at 30 000 CUR per m3/h."],
        ["desal_water_storage",
         "Desalination how-to: water becomes a 15 000 m3 storage and the water demand "
         "varies (12 h at 2000, 12 h at 4000 m3/h)."],
    ]
    entities = [
        ["node", "desal_heat", None],
        ["node", "water", None],
        ["unit", "bp_chp", None],
        ["unit__inputNode", ["bp_chp", "coal_market"], None],
        ["unit__outputNode", ["bp_chp", "west"], None],
        ["unit__outputNode", ["bp_chp", "desal_heat"], None],
        ["constraint", "bp_chp_ratio", None],
        ["unit", "med_desal", None],
        ["unit__inputNode", ["med_desal", "desal_heat"], None],
        ["unit__outputNode", ["med_desal", "water"], None],
        ["unit", "ext_chp", None],
        ["unit__inputNode", ["ext_chp", "coal_market"], None],
        ["unit__outputNode", ["ext_chp", "west"], None],
        ["unit__outputNode", ["ext_chp", "desal_heat"], None],
        ["constraint", "ext_chp_backpressure_line", None],
        ["constraint", "ext_chp_max_heat", None],
        ["unit", "ro_desal", None],
        ["unit__inputNode", ["ro_desal", "west"], None],
        ["unit__outputNode", ["ro_desal", "water"], None],
    ]
    entity_alternatives = [
        ["node", ["desal_heat"], "desal_system", True],
        ["node", ["water"], "desal_system", True],
        ["unit", ["bp_chp"], "desal_system", True],
        ["unit", ["med_desal"], "desal_system", True],
        ["unit", ["bp_chp"], "desal_extraction", False],
        ["unit", ["ext_chp"], "desal_extraction", True],
        ["unit", ["ro_desal"], "desal_ro_invest", True],
    ]
    sys_alt = "desal_system"
    ext_alt = "desal_extraction"
    ro_alt = "desal_ro_invest"
    sto_alt = "desal_water_storage"
    water_demand = ([-2000.0] * 12 + [-4000.0] * 12) * 3
    values: list[list] = [
        # desal_system
        ["node", "west", "inflow", -300.0, sys_alt],
        ["node", "desal_heat", "node_type", "balance", sys_alt],
        ["node", "desal_heat", "penalty_up", 1000.0, sys_alt],
        ["node", "desal_heat", "penalty_down", 1000.0, sys_alt],
        ["node", "water", "node_type", "balance", sys_alt],
        ["node", "water", "inflow", -3000.0, sys_alt],
        ["node", "water", "penalty_up", 50.0, sys_alt],
        ["node", "water", "penalty_down", 50.0, sys_alt],
        ["unit", "bp_chp", "conversion_method", "constant_efficiency", sys_alt],
        ["unit", "bp_chp", "efficiency", 0.85, sys_alt],
        ["unit", "bp_chp", "existing", 600.0, sys_alt],
        ["constraint", "bp_chp_ratio", "sense", "equal", sys_alt],
        ["constraint", "bp_chp_ratio", "constant", 0.0, sys_alt],
        ["unit__outputNode", ["bp_chp", "west"], "constraint_flow_coeff",
         _constraint_map("bp_chp_ratio", 1.0), sys_alt],
        ["unit__outputNode", ["bp_chp", "desal_heat"], "constraint_flow_coeff",
         _constraint_map("bp_chp_ratio", -0.5), sys_alt],
        ["unit", "med_desal", "conversion_method", "constant_efficiency", sys_alt],
        ["unit", "med_desal", "efficiency", 12.0, sys_alt],
        ["unit", "med_desal", "existing", 3000.0, sys_alt],
        # desal_extraction
        ["unit", "ext_chp", "conversion_method", "constant_efficiency", ext_alt],
        ["unit", "ext_chp", "efficiency", 0.45, ext_alt],
        ["unit", "ext_chp", "existing", 1000.0, ext_alt],
        ["unit__outputNode", ["ext_chp", "west"], "conversion_flow_coeff", 1.0, ext_alt],
        ["unit__outputNode", ["ext_chp", "desal_heat"], "conversion_flow_coeff", 0.15, ext_alt],
        ["constraint", "ext_chp_backpressure_line", "sense", "greater_than", ext_alt],
        ["constraint", "ext_chp_backpressure_line", "constant", 0.0, ext_alt],
        ["constraint", "ext_chp_max_heat", "sense", "less_than", ext_alt],
        ["constraint", "ext_chp_max_heat", "constant", 0.0, ext_alt],
        ["unit__outputNode", ["ext_chp", "west"], "constraint_flow_coeff",
         _constraint_map("ext_chp_backpressure_line", 1.0), ext_alt],
        ["unit__outputNode", ["ext_chp", "desal_heat"], "constraint_flow_coeff",
         Map(["ext_chp_backpressure_line", "ext_chp_max_heat"], [-0.5, 1.0],
             index_name="constraint"), ext_alt],
        ["unit", "ext_chp", "constraint_capacity_coeff",
         _constraint_map("ext_chp_max_heat", -0.6), ext_alt],
        # desal_ro_invest
        ["unit", "ro_desal", "conversion_method", "constant_efficiency", ro_alt],
        ["unit", "ro_desal", "efficiency", 250.0, ro_alt],
        ["unit", "ro_desal", "invest_method", "invest_total", ro_alt],
        ["unit", "ro_desal", "invest_max_total", 2000.0, ro_alt],
        ["unit", "ro_desal", "invest_cost", 30.0, ro_alt],
        ["unit", "ro_desal", "lifetime", 20.0, ro_alt],
        ["unit", "ro_desal", "discount_rate", 0.05, ro_alt],
        ["solve", DISPATCH_SOLVE, "invest_periods", _invest_periods(), ro_alt],
        # desal_water_storage
        ["node", "water", "node_type", "storage", sto_alt],
        ["node", "water", "existing", 15000.0, sto_alt],
        ["node", "water", "storage_binding_method", "bind_within_timeblock", sto_alt],
        ["node", "water", "storage_start_end_method", "fix_start", sto_alt],
        ["node", "water", "storage_state_start", 0.0, sto_alt],
        ["node", "water", "inflow", _time_map(water_demand), sto_alt],
    ]
    base = ["init", "west", "coal", sys_alt]
    scenarios = {
        "desal_backpressure": base,
        "desal_extraction": base + [ext_alt],
        "desal_ro_invest": base + [ro_alt],
        "desal_water_storage": base + [sto_alt],
    }
    return {
        "alternatives": alternatives,
        "entities": entities,
        "entity_alternatives": entity_alternatives,
        "parameter_values": values,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# CSP with thermal storage (E-3)
# ---------------------------------------------------------------------------


def _csp_field_yield() -> list[float]:
    """Synthetic field yield [MW_th per MW_th of field]: a clear day (1.0 from t0007 to t0018),
    a cloudy day (0.4 from t0031 to t0042), then the clear day again."""
    clear = [1.0 if 7 <= h <= 18 else 0.0 for h in range(1, 25)]
    cloudy = [0.4 if 7 <= h <= 18 else 0.0 for h in range(1, 25)]
    return clear + cloudy + clear


def _csp_data() -> dict[str, list]:
    alternatives = [
        ["csp_system",
         "CSP how-to: constant 150 MW demand at west (coal_plant is the backup) and the "
         "field-yield profile csp_field_yield (a clear day, a cloudy day, a clear day)."],
        ["csp_plant",
         "CSP how-to: existing CSP plant: solar field csp_field 500 MW_th, heat storage "
         "csp_heat 2500 MWh_th, turbine csp_turbine 100 MW_e (efficiency 0.4)."],
        ["csp_invest_common",
         "CSP how-to: no existing capacity; field, storage and turbine may be invested "
         "(bounds 1000 MW_th, 5000 MWh_th, 100 MW_e)."],
        ["csp_fixed_design",
         "CSP how-to: solar multiple 2 and 10 storage hours at design efficiency 0.4, "
         "as user constraints on the capacities (field = 5 x turbine, storage = 25 x turbine)."],
        ["csp_gas_backup",
         "CSP how-to: a natural-gas boiler csp_backup_boiler (250 MW_th, efficiency 0.9) "
         "that can also charge the heat storage."],
    ]
    entities = [
        ["profile", "csp_field_yield", None],
        ["unit", "csp_field", None],
        ["unit__outputNode", ["csp_field", "csp_heat"], None],
        ["node", "csp_heat", None],
        ["unit__node__profile", ["csp_field", "csp_heat", "csp_field_yield"], None],
        ["unit", "csp_turbine", None],
        ["unit__inputNode", ["csp_turbine", "csp_heat"], None],
        ["unit__outputNode", ["csp_turbine", "west"], None],
        ["constraint", "csp_solar_multiple", None],
        ["constraint", "csp_storage_hours", None],
        ["commodity", "natural_gas", None],
        ["node", "natural_gas_market", None],
        ["commodity__node", ["natural_gas", "natural_gas_market"], None],
        ["unit", "csp_backup_boiler", None],
        ["unit__inputNode", ["csp_backup_boiler", "natural_gas_market"], None],
        ["unit__outputNode", ["csp_backup_boiler", "csp_heat"], None],
    ]
    entity_alternatives = [
        ["unit", ["csp_field"], "csp_plant", True],
        ["node", ["csp_heat"], "csp_plant", True],
        ["unit", ["csp_turbine"], "csp_plant", True],
        ["node", ["natural_gas_market"], "csp_gas_backup", True],
        ["unit", ["csp_backup_boiler"], "csp_gas_backup", True],
    ]
    sys_alt, plant_alt = "csp_system", "csp_plant"
    inv_alt, fix_alt, gas_alt = "csp_invest_common", "csp_fixed_design", "csp_gas_backup"
    values: list[list] = [
        # csp_system
        ["node", "west", "inflow", -150.0, sys_alt],
        ["profile", "csp_field_yield", "profile", _time_map(_csp_field_yield()), sys_alt],
        # csp_plant
        ["unit", "csp_field", "conversion_method", "none", plant_alt],
        ["unit", "csp_field", "existing", 500.0, plant_alt],
        ["unit__node__profile", ["csp_field", "csp_heat", "csp_field_yield"], "profile_method",
         "upper_limit", plant_alt],
        ["node", "csp_heat", "node_type", "storage", plant_alt],
        ["node", "csp_heat", "existing", 2500.0, plant_alt],
        ["node", "csp_heat", "storage_binding_method", "bind_within_timeblock", plant_alt],
        ["node", "csp_heat", "penalty_up", 1000.0, plant_alt],
        ["node", "csp_heat", "penalty_down", 1000.0, plant_alt],
        ["unit", "csp_turbine", "conversion_method", "constant_efficiency", plant_alt],
        ["unit", "csp_turbine", "efficiency", 0.4, plant_alt],
        ["unit", "csp_turbine", "existing", 100.0, plant_alt],
        # csp_invest_common
        ["solve", DISPATCH_SOLVE, "invest_periods", _invest_periods(), inv_alt],
        # csp_fixed_design: field - (SM / eta_d) x turbine = 0, storage - (h / eta_d) x turbine = 0
        ["constraint", "csp_solar_multiple", "sense", "equal", fix_alt],
        ["constraint", "csp_solar_multiple", "constant", 0.0, fix_alt],
        ["constraint", "csp_storage_hours", "sense", "equal", fix_alt],
        ["constraint", "csp_storage_hours", "constant", 0.0, fix_alt],
        ["unit", "csp_field", "constraint_capacity_coeff",
         _constraint_map("csp_solar_multiple", 1.0), fix_alt],
        ["node", "csp_heat", "constraint_capacity_coeff",
         _constraint_map("csp_storage_hours", 1.0), fix_alt],
        ["unit", "csp_turbine", "constraint_capacity_coeff",
         Map(["csp_solar_multiple", "csp_storage_hours"], [-5.0, -25.0], index_name="constraint"),
         fix_alt],
        # csp_gas_backup
        ["commodity", "natural_gas", "price", 12.0, gas_alt],
        ["node", "natural_gas_market", "node_type", "commodity", gas_alt],
        ["unit", "csp_backup_boiler", "conversion_method", "constant_efficiency", gas_alt],
        ["unit", "csp_backup_boiler", "efficiency", 0.9, gas_alt],
        ["unit", "csp_backup_boiler", "existing", 250.0, gas_alt],
    ]
    for cls, name, bound, cost in (
        ("unit", "csp_turbine", 100.0, 500.0),
        ("unit", "csp_field", 1000.0, 150.0),
        ("node", "csp_heat", 5000.0, 15.0),
    ):
        values += [
            [cls, name, "existing", 0.0, inv_alt],
            [cls, name, "invest_method", "invest_total", inv_alt],
            [cls, name, "invest_max_total", bound, inv_alt],
            [cls, name, "invest_cost", cost, inv_alt],
            [cls, name, "lifetime", 25.0, inv_alt],
            [cls, name, "discount_rate", 0.07, inv_alt],
        ]
    base = ["init", "west", "coal", sys_alt]
    scenarios = {
        "csp_without": base,
        "csp_existing": base + [plant_alt],
        "csp_fixed_design": base + [plant_alt, inv_alt, fix_alt],
        "csp_optimised_design": base + [plant_alt, inv_alt],
        "csp_gas_backup": base + [plant_alt, gas_alt],
    }
    return {
        "alternatives": alternatives,
        "entities": entities,
        "entity_alternatives": entity_alternatives,
        "parameter_values": values,
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _parts() -> tuple[dict[str, list], ...]:
    return (_must_run_data(), _co2_trajectory_data(), _desalination_data(), _csp_data())


# What this script adds to tests.json, for builders that start from tests.json
# and must not carry the Phase 3 examples (regen_lh2_three_region.py).
ADDED_ALTERNATIVES: tuple[str, ...] = tuple(
    row[0] for part in _parts() for row in part["alternatives"])
ADDED_ENTITIES: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (row[0], (row[1],) if isinstance(row[1], str) else tuple(row[1]))
    for part in _parts() for row in part["entities"])
ADDED_SCENARIOS: tuple[str, ...] = tuple(
    name for part in _parts() for name in part["scenarios"])


def _import(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    parts = _parts()
    with open_database_mapping(url) as db:
        existing_scenarios = {s["name"] for s in db.find_scenarios()}
        for part in parts:
            scenario_rows: list[list] = []
            chain_rows: list[list] = []
            for name, chain in part["scenarios"].items():
                if name in existing_scenarios:
                    continue  # already added on an earlier run: keep its chain as is
                scenario_rows.append([name, False, ""])
                chain_rows += _chain_rows(name, chain)
            count, errors = import_data(
                db,
                alternatives=part["alternatives"],
                entities=part["entities"],
                entity_alternatives=part["entity_alternatives"],
                parameter_values=part["parameter_values"],
                scenarios=scenario_rows,
                scenario_alternatives=chain_rows,
                on_conflict="keep",
            )
            if errors:
                raise RuntimeError(f"import errors: {errors}")
            print(f"imported {count} items")
        try:
            db.commit_session("Phase 3 how-to examples")
        except Exception as exc:  # nothing to commit on a re-run
            if "Nothing to commit" not in str(exc):
                raise
        for part in parts:
            for name, chain in part["scenarios"].items():
                got = [
                    r["alternative_name"]
                    for r in sorted(db.find_scenario_alternatives(scenario_name=name),
                                    key=lambda r: r["rank"])
                ]
                if got != chain:
                    raise RuntimeError(f"scenario {name}: chain {got} != {chain}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "tests_augment.sqlite"
        json_to_db(TESTS_JSON, tmp_db)
        _import(tmp_db)
        db_to_json(tmp_db, TESTS_JSON)
    print(f"wrote {TESTS_JSON}")


if __name__ == "__main__":
    main()
