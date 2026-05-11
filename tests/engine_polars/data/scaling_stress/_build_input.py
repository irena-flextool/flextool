"""Generator for ``tests/engine_polars/data/scaling_stress/input.json``.

Usage::

    python tests/engine_polars/data/scaling_stress/_build_input.py

Takes ``tests/fixtures/tests.json`` as the template (so the entire
class/parameter schema, timelines, and the ``multi_year_one_solve_battery``
solve chain come along for free) and layers a single new alternative
``scaling_stress`` on top with stress-test parameter overrides chosen
from *realistic* FlexTool modeling levers.

Realistic levers exercised here:

* **VOM (other_operational_cost) spread.**  wind_plant gets 0.01 €/MWh
  (essentially zero — modern wind), coal_plant gets 50 €/MWh (fuel +
  operating cost passthrough).  ~3.5 decades inside one parameter family.
* **Investment cost spread.**  coal_plant 1500 €/kW, wind_plant 1200
  €/kW, battery node 8000 €/MWh (storage capex), battery_inverter 250
  €/kW.  Realistic 5-30× spread between cheap inverter and big block of
  storage / thermal capacity.
* **Slack penalties.**  ``penalty_up`` / ``penalty_down`` set to 1e+5 on
  all nodes — kept high to deter slack use.  These hit every timestep
  so they multiply through the time-aggregation factors.
* **Period years_represented spread.**  Override the active solve
  ``y2020_2035_5week`` so that periods alternate 1 yr / 10 yr / 1 yr /
  10 yr — a 10× factor *within one model* on top of the per-timestep
  duration weight.  Combined with the 4-hour timestep_duration of the
  ``5weeks`` timeset, the per-flow objective coefficient picks up a
  realistic 40× to 400× annualisation factor depending on period.
* **Mix of unit sizes.**  coal_plant block 5000 MW, wind_plant 50 MW
  (a single turbine), battery_inverter 50 MW.  ~2 decades spread inside
  the unitsize family — also realistic (utility-scale thermal vs single
  wind module).

Nothing in this fixture is below 0.001 or above 1e+8.  All overrides
are values a real FlexTool user could plausibly type in.

Goal: reproducible HiGHS coefficient spread of roughly 4-6 decades on
the cost side, plus matrix / RHS spread driven by the time-aggregation
weighting — enough to exercise the scaler without resorting to fake
parameter values.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_JSON = REPO_ROOT / "tests" / "fixtures" / "tests.json"
TARGET_JSON = Path(__file__).parent / "input.json"

STRESS_ALT = "scaling_stress"
STRESS_SCEN = "scaling_stress"
BASE_SCEN = "multi_year_one_solve_battery"


def _enc_float(x: float) -> list:
    raw = json.dumps(x).encode("utf-8")
    return [base64.b64encode(raw).decode("ascii"), "float"]


def _enc_str(s: str) -> list:
    raw = json.dumps(s).encode("utf-8")
    return [base64.b64encode(raw).decode("ascii"), "str"]


def _enc_map(obj: dict) -> list:
    raw = json.dumps(obj).encode("utf-8")
    return [base64.b64encode(raw).decode("ascii"), "map"]


def main() -> None:
    with SOURCE_JSON.open() as fh:
        d = json.load(fh)

    # 1. Add the alternative.
    if not any(a[0] == STRESS_ALT for a in d["alternatives"]):
        d["alternatives"].append(
            [STRESS_ALT, "Scaling stress overrides for scaling_stress scenario."]
        )

    # 2. Add the scenario.
    if not any(s[0] == STRESS_SCEN for s in d["scenarios"]):
        d["scenarios"].append([STRESS_SCEN, False, ""])

    # 3. Replay multi_year_one_solve_battery's alternative chain under
    # the new scenario, with scaling_stress appended at the very end so
    # its overrides win.
    base_chain = [s[1] for s in d["scenario_alternatives"] if s[0] == BASE_SCEN]
    if not base_chain:
        raise RuntimeError(f"base scenario {BASE_SCEN!r} not found in tests.json")
    new_chain = base_chain + [STRESS_ALT]

    # Drop any pre-existing rows for our scenario before re-emitting.
    d["scenario_alternatives"] = [
        s for s in d["scenario_alternatives"] if s[0] != STRESS_SCEN
    ]
    for cur, nxt in zip(new_chain, new_chain[1:] + [None]):
        d["scenario_alternatives"].append([STRESS_SCEN, cur, nxt])

    # 4. Layer the stress overrides into parameter_values under the
    # new alternative.  Format: [class, entity, param, [b64, type], alt].

    # ---- Realistic per-period years_represented (1, 10, 1, 10) ----
    # The active solve in BASE_SCEN is `y2020_2035_5week` (4 periods,
    # base values [5,5,5,5]).  Mixing 1 / 10 within one model is a
    # well-documented FlexTool pattern (representative-period
    # weighting) and produces a 10× spread on every per-timestep
    # objective term *within one solve*.
    years_repr = {
        "index_type": "str",
        "data": [
            ["p2020", 1.0],
            ["p2025", 10.0],
            ["p2030", 1.0],
            ["p2035", 10.0],
        ],
        "index_name": "period",
    }

    overrides: list[list] = [
        # ----- Slack penalties (cost & RHS contribution, every step) -----
        ["node", "west", "penalty_up", _enc_float(1.0e5), STRESS_ALT],
        ["node", "west", "penalty_down", _enc_float(1.0e5), STRESS_ALT],
        ["node", "east", "penalty_up", _enc_float(1.0e5), STRESS_ALT],
        ["node", "east", "penalty_down", _enc_float(1.0e5), STRESS_ALT],
        ["node", "north", "penalty_up", _enc_float(1.0e5), STRESS_ALT],
        ["node", "north", "penalty_down", _enc_float(1.0e5), STRESS_ALT],

        # ----- Mix of unit sizes (realistic: utility thermal block vs
        #       single wind turbine vs small inverter module) -----
        ["unit", "coal_plant", "virtual_unitsize", _enc_float(5000.0),
         STRESS_ALT],
        ["unit", "wind_plant", "virtual_unitsize", _enc_float(50.0),
         STRESS_ALT],
        ["connection", "battery_inverter", "virtual_unitsize",
         _enc_float(50.0), STRESS_ALT],

        # ----- VOM spread on flows (realistic 0.01 vs 50 €/MWh) -----
        # Wind is essentially free to operate; coal carries fuel +
        # variable O&M.  This is the small-coefficient lever the user
        # called out — the 0.01 here is the smallest realistic cost a
        # FlexTool user would type in.
        ["unit__outputNode", ["wind_plant", "west"],
         "other_operational_cost", _enc_float(0.01), STRESS_ALT],
        ["unit__outputNode", ["coal_plant", "west"],
         "other_operational_cost", _enc_float(50.0), STRESS_ALT],

        # ----- Realistic investment costs (no fake 1e-3) -----
        # Spread: ~6× between cheapest (battery_inverter 250 €/kW) and
        # most expensive (battery storage node 8000 €/MWh).
        ["unit", "coal_plant", "invest_cost", _enc_float(1500.0),
         STRESS_ALT],
        ["unit", "wind_plant", "invest_cost", _enc_float(1200.0),
         STRESS_ALT],
        ["unit", "coal_plant", "fixed_cost", _enc_float(40.0), STRESS_ALT],
        ["connection", "battery_inverter", "invest_cost",
         _enc_float(250.0), STRESS_ALT],
        ["node", "battery", "invest_cost", _enc_float(8000.0), STRESS_ALT],

        # ----- Realistic efficiencies (restore base-style values) -----
        # Note: leaving these at the alt-chain values (coal 0.4,
        # wind 1.0, battery_inverter 0.95) is the realistic choice —
        # we do NOT crush them to 0.1.  Explicitly re-asserting in case
        # downstream alts change the defaults.
        ["unit", "coal_plant", "efficiency", _enc_float(0.4), STRESS_ALT],
        ["unit", "wind_plant", "efficiency", _enc_float(1.0), STRESS_ALT],
        ["connection", "battery_inverter", "efficiency",
         _enc_float(0.95), STRESS_ALT],

        # ----- Discounting + inflation (already in base; re-asserted) -----
        ["model", "flexTool", "inflation_rate", _enc_float(0.02),
         STRESS_ALT],
        ["unit", "coal_plant", "discount_rate", _enc_float(0.05),
         STRESS_ALT],
        ["unit", "wind_plant", "discount_rate", _enc_float(0.05),
         STRESS_ALT],
        ["connection", "battery_inverter", "discount_rate",
         _enc_float(0.05), STRESS_ALT],
        ["node", "battery", "discount_rate", _enc_float(0.05), STRESS_ALT],

        # ----- Investment lifetime (realistic, kept for clarity) -----
        ["unit", "coal_plant", "lifetime", _enc_float(30.0), STRESS_ALT],
        ["unit", "wind_plant", "lifetime", _enc_float(25.0), STRESS_ALT],
        ["connection", "battery_inverter", "lifetime",
         _enc_float(20.0), STRESS_ALT],
        ["node", "battery", "lifetime", _enc_float(10.0), STRESS_ALT],

        # ----- Period-years-represented spread: the *time-aggregation*
        #       lever the user called out.  1 yr vs 10 yr inside one
        #       model is realistic representative-period weighting. -----
        ["solve", "y2020_2035_5week", "years_represented",
         _enc_map(years_repr), STRESS_ALT],
    ]

    # Drop any pre-existing rows under the stress alt before re-emitting.
    d["parameter_values"] = [
        pv for pv in d["parameter_values"] if pv[4] != STRESS_ALT
    ]
    d["parameter_values"].extend(overrides)

    # entity_alternatives — needed?  tests.json has 38 rows; safest to
    # only add an entity_alternative if the same (class, entity, alt)
    # tuple isn't already present.  Spine doesn't strictly require these
    # for parameter_values to take effect, so we skip them.

    TARGET_JSON.parent.mkdir(parents=True, exist_ok=True)
    with TARGET_JSON.open("w") as fh:
        json.dump(d, fh, indent=2)
    print(f"Wrote {TARGET_JSON} ({TARGET_JSON.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
