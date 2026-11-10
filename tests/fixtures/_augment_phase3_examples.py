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


def _constraint_map(name: str, coeff: float = 1.0) -> Map:
    return Map([name], [coeff], index_name="constraint")


def _period_map(values: tuple[float, ...]) -> Map:
    return Map(list(PERIODS), [float(v) for v in values], index_name="period")


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
# Driver
# ---------------------------------------------------------------------------


def _import(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    parts = (_must_run_data(), _co2_trajectory_data())
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
            db.commit_session("Phase 3 how-to examples: must-run units, emission trajectories")
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
