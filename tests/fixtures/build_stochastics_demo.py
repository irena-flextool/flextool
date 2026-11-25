"""Add the example scenarios of ``docs/how_to/stochastic_examples.md`` to ``tests/fixtures/stochastics.json``.

Run ``python tests/fixtures/build_stochastics_demo.py`` from the repository root.  It reads ``stochastics.json``,
removes the rows the examples own (their scenarios, alternatives and entities, below) and adds them again from this
file, so it can be run again after a change here.  The examples have entities of their own and every value sits
in their own alternatives: the other scenarios of the file see nothing of them (nodes, units and connections are
inactive by default; the shared ``model`` entity ``flextool`` gets ``solves`` only in the examples' alternatives;
the commodities ``coal`` and ``gas`` are shared, with prices in ``rolling_demo``; the user constraint
``hydro_divide``, active by default, is made inactive in ``rolling_demo`` and ``invest_demo``).
``python -m flextool.update_flextool.generate_canonical`` then writes ``howto_stochastics.json`` (the GUI's
``stochastics.sqlite``).  Checks: ``tests/engine/realization_roll/test_stochastics_demo_rolling.py`` and
``tests/engine/test_stochastics_demo_invest.py``.

**Rolling dispatch.**  One timeline ``hourly`` of 216 one-hour steps ``t0001`` ...; three periods ``period1`` -
``period3`` of 72 h each, one after the other, each on its own timeset ``roll_ts_<period>`` and each representing
one year (``years_represented`` 1, ``social_discount_rate`` 0: a cost in a period's steps counts 8760 / 72 times).
One solve ``rolling_dispatch``: ``rolling_window``, jump 24 h, horizon 48 h; roll k realizes the day starting at
step 24k + 1; the horizon of the last roll ends with its jump; the rolls at the end of ``period1`` and ``period2``
look ahead into the next period.  One balance node ``elec`` (unserved / excess energy 3000 EUR/MWh):

* demand ``PERIOD_SCALE x (DEMAND_BASE + DEMAND_SWING x sin(2 pi (hour of day - 13) / 24))`` MW (peak at 19:00);
* ``wind_farm`` 650 MW, ``upper_limit`` profile ``wind_cf`` (stochastic, in group ``rolling_stochastic``);
* ``coal_units``: 3 units of 100 MW (``existing`` 3, ``virtual_unitsize`` 100) on ``coal_supply`` (coal 12
  EUR/MWh), ``min_load_efficiency``: efficiency 0.4 at full load, 0.35 at ``min_load`` 0.4 (a unit online burns
  fuel even at low output), ``startup_method`` ``integer_fix_in_uc``, ``startup_cost`` 50 EUR/MW (5000 EUR a unit
  start), ``minimum_time_method`` ``min_uptime``, ``min_uptime`` 8 h;
* ``gas_peaker``: 350 MW on ``gas_supply`` (gas 28 EUR/MWh), ``min_load_efficiency``: 0.35 at full load, 0.3 at
  ``min_load`` 0.2, ``startup_method`` ``linear``, ``startup_cost`` 5 EUR/MW;
* ``battery``: storage node, 400 MWh, ``self_discharge_loss`` 0.0005 per hour, start 50 % (``fix_start``), each
  roll's horizon ends at 50 % (``use_reference_value``), in group ``rolling_stochastic``; ``battery_inverter``
  connection elec - battery, 100 MW, efficiency 0.92, 0.5 EUR/MWh.

Wind (``wind_cf``, branch -> analysis time -> time): ``actual`` (``realized: yes``) is the actual series ``R(t)``,
given at each roll start for the 24 h of its jump (the series do not overlap).  At every roll start ``a`` two
forecasts cover ``a`` .. ``a + 47`` (cut at the last step, so every roll's horizon is covered): a centre ``C = R +
bias(a) x min(1, lead / 12)`` and ``high`` / ``low`` = ``C`` +/- ``(0.03 + 0.005 x lead)``, clipped to [0, 0.98]
(lead = hours after ``a``).  The forecasts made at the start of a roll serve the ``separate_dispatch`` stochastic
solve of that roll; the same forecasts, made at the end of the previous jump, serve the look-ahead of the
previous roll.  Alternatives: ``rolling_demo`` (the model, forecasts ``high`` 0.5 / ``low`` 0.5 and ``actual``
made at each roll start, ``dispatch_realization_method`` ``deterministic``); ``within_solve`` /
``separate_dispatch`` (the method); ``perfect_forecast`` (one forecast ``exact``, weight 1, equal to ``R`` over its
48 h).

**Investment under uncertainty.**  Timeline ``invest_hours`` t0001-t0008; periods ``y2030`` (t0001-t0004) and
``y2035`` (t0005-t0008), ``years_represented`` 5 each, ``social_discount_rate`` 0.  Node ``town`` (balance,
``penalty_up`` 3000 EUR/MWh, in group ``invest_stochastic``) with a stochastic demand (``inflow``): 100 MW in y2030
in every branch; in y2035 ``high`` 160 MW and ``low`` 120 MW (weight 0.5 each), ``actual`` (``realized: yes``) =
high.  Unit ``new_plant`` -> town, nothing existing, ``invest_no_limit``, ``invest_cost`` {y2030: 500, y2035: 1500}
EUR/kW, ``discount_rate`` 0, ``lifetime_technical`` 10, no operating cost.  Solve ``invest_plan``:
``single_solve`` over both periods, ``invest_periods`` both, ``realized_invest_periods`` = ``realized_periods`` =
[y2030] (y2035 is look-ahead), the forecasts made at y2030's first step.  Alternatives: ``invest_demo`` (the model,
``operations_only``); ``after_reveal_period`` (``invest_solve_branching_method``
``investments_after_reveal_period``); ``invest_then_dispatch`` (``model.solves`` [invest_first, dispatch_2030]:
the investment solve without ``realized_periods`` and without the ``actual`` branch, then a dispatch solve of y2030
with the forecasts and ``actual``, ``dispatch_realization_method`` ``separate_dispatch``).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.engine._solve_copy_models import array, nested, pack, pmap  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "stochastics.json"

# --- rolling dispatch ---------------------------------------------------------------------------------------------
ROLL_SOLVE = "rolling_dispatch"
PERIODS = ("period1", "period2", "period3")
PERIOD_HOURS = 72
N_STEPS = len(PERIODS) * PERIOD_HOURS
JUMP = 24
HORIZON = 48
DEMAND_BASE = 520.0
DEMAND_SWING = 130.0
PERIOD_SCALE = {"period1": 1.0, "period2": 1.08, "period3": 0.95}
WEIGHTS = {"high": 0.5, "low": 0.5}
ROLL_SCENARIOS = {
    "rolling_deterministic": ["rolling_demo"],
    "rolling_within_solve": ["rolling_demo", "within_solve"],
    "rolling_separate_dispatch": ["rolling_demo", "separate_dispatch"],
    "rolling_perfect_within_solve": ["rolling_demo", "perfect_forecast", "within_solve"],
    "rolling_perfect_separate_dispatch": ["rolling_demo", "perfect_forecast", "separate_dispatch"],
}

# --- investment ---------------------------------------------------------------------------------------------------
INVEST_PERIODS = ("y2030", "y2035")
INVEST_STEPS = 4                                     # per period
YEARS = 5.0                                          # years_represented of each period
DEMAND_2030 = 100.0
DEMAND_2035 = {"high": 160.0, "low": 120.0, "actual": 160.0}
INVEST_COST = {"y2030": 500.0, "y2035": 1500.0}      # EUR/kW
LIFETIME = 10.0
INVEST_SCENARIOS = {
    "invest_operations_only": ["invest_demo"],
    "invest_after_reveal_period": ["invest_demo", "after_reveal_period"],
    "invest_then_dispatch": ["invest_demo", "after_reveal_period", "invest_then_dispatch"],
}

SCENARIOS = {**ROLL_SCENARIOS, **INVEST_SCENARIOS}
SCENARIO_DESCRIPTIONS = {
    "rolling_deterministic": "Rolling dispatch on the actual wind (the forecasts are not solved)",
    "rolling_within_solve": "Rolling dispatch: each day on the actual wind, look-ahead with the forecasts",
    "rolling_separate_dispatch": "Rolling dispatch: commitment against the forecasts, then the day on the actual wind",
    "rolling_perfect_within_solve": "rolling_within_solve with a perfect forecast",
    "rolling_perfect_separate_dispatch": "rolling_separate_dispatch with a perfect forecast",
    "invest_operations_only": "Investment: one fleet for both demand futures",
    "invest_after_reveal_period": "Investment: invest in 2030, adapt the 2035 investment to the future",
    "invest_then_dispatch": "Investment solve, then a dispatch solve that realizes 2030",
}
ALTERNATIVES = {
    "rolling_demo": "Rolling dispatch example: the model, the wind forecasts, dispatch_realization_method deterministic",
    "within_solve": "dispatch_realization_method within_solve",
    "separate_dispatch": "dispatch_realization_method separate_dispatch",
    "perfect_forecast": "One wind forecast equal to the actual wind",
    "invest_demo": "Investment example: the model and the demand forecasts (operations_only)",
    "after_reveal_period": "invest_solve_branching_method investments_after_reveal_period",
    "invest_then_dispatch": "The investment solve followed by a dispatch solve",
}

ROLL_ENTITIES = [
    ("timeline", "hourly"), ("solve", ROLL_SOLVE), ("group", "rolling_stochastic"),
    *(("timeset", f"roll_ts_{p}") for p in PERIODS),
    ("node", "elec"), ("node", "battery"), ("node", "coal_supply"), ("node", "gas_supply"),
    ("unit", "wind_farm"), ("unit", "coal_units"), ("unit", "gas_peaker"), ("connection", "battery_inverter"),
    ("profile", "wind_cf"),
    ("commodity__node", ("coal", "coal_supply")), ("commodity__node", ("gas", "gas_supply")),
    ("unit__outputNode", ("wind_farm", "elec")), ("unit__outputNode", ("coal_units", "elec")),
    ("unit__outputNode", ("gas_peaker", "elec")),
    ("unit__inputNode", ("coal_units", "coal_supply")), ("unit__inputNode", ("gas_peaker", "gas_supply")),
    ("connection__node__node", ("battery_inverter", "elec", "battery")),
    ("unit__node__profile", ("wind_farm", "elec", "wind_cf")),
    ("group__unit", ("rolling_stochastic", "wind_farm")), ("group__node", ("rolling_stochastic", "battery")),
]
INVEST_ENTITIES = [
    ("timeline", "invest_hours"), ("solve", "invest_plan"), ("solve", "invest_first"), ("solve", "dispatch_2030"),
    ("group", "invest_stochastic"), *(("timeset", f"invest_ts_{p}") for p in INVEST_PERIODS),
    ("node", "town"), ("unit", "new_plant"),
    ("unit__outputNode", ("new_plant", "town")), ("group__node", ("invest_stochastic", "town")),
]
#: Every entity the examples own (the commodities and the model are shared).
ENTITIES = ROLL_ENTITIES + INVEST_ENTITIES


def step(i: int) -> str:
    return f"t{i + 1:04d}"


# --- rolling data -------------------------------------------------------------------------------------------------

def period_of(i: int) -> str:
    return PERIODS[i // PERIOD_HOURS]


def roll_starts() -> list[int]:
    return list(range(0, N_STEPS, JUMP))


def covered(a: int, length: int) -> range:
    return range(a, min(a + length, N_STEPS))


def demand(i: int) -> float:
    return round(PERIOD_SCALE[period_of(i)]
                 * (DEMAND_BASE + DEMAND_SWING * math.sin(2 * math.pi * ((i % 24) - 13) / 24)), 3)


def realized_wind(i: int) -> float:
    v = 0.45 + 0.3 * math.sin(2 * math.pi * i / (24 * 3.3)) + 0.12 * math.sin(2 * math.pi * i / 13 + 1.0)
    return round(min(0.95, max(0.02, v)), 3)


def _bias(a: int) -> float:
    return 0.1 * math.sin(1.7 * a / 24)


def forecast(branch: str, a: int, i: int) -> float:
    """The *branch* forecast made at step index *a* for step index *i* (``a <= i < a + HORIZON``)."""
    lead = i - a
    centre = realized_wind(i) + _bias(a) * min(1.0, lead / 12)
    spread = 0.03 + 0.005 * lead
    v = centre + spread if branch == "high" else centre - spread
    return round(min(0.98, max(0.0, v)), 3)


def _profile(branches: dict) -> list:
    """``{branch: {analysis step index: {step index: value}}}`` -> a packed branch / analysis_time / time map."""
    return pack(pmap([(b, nested([(step(a), nested([(step(i), v) for i, v in vals.items()], index_name="time"))
                                  for a, vals in made.items()], index_name="analysis_time"))
                      for b, made in branches.items()], index_name="branch"), "map")


def _roll_branches(rows: list[tuple[str, float, str]]) -> list:
    """``stochastic_branches``: each of *rows* ``(branch, weight, realized)`` made at every roll start, under the
    period of that step."""
    by_period: dict[str, list] = {}
    for a in roll_starts():
        by_period.setdefault(period_of(a), []).append(a)
    return pack(pmap([(p, nested([(b, nested([(step(a), nested([(yes, w)], index_name="realized"))
                                                for a in starts], index_name="analysis_time"))
                                   for b, w, yes in rows], index_name="branch"))
                      for p, starts in by_period.items()], index_name="period"), "map")


def _made(value, length: int) -> dict:
    """``{roll start: {step: value(roll start, step)}}`` over *length* steps from each roll start."""
    return {a: {i: value(a, i) for i in covered(a, length)} for a in roll_starts()}


def _f(v: float) -> list:
    return pack(float(v), "float")


def _s(v: str) -> list:
    return pack(v, "str")


def _time_map(values) -> list:
    return pack(pmap([(step(i), v) for i, v in values], index_name="time"), "map")


def _period_map(values: dict) -> list:
    return pack(pmap(list(values.items()), index_name="period"), "map")


def _rolling_values() -> list:
    def r(cls, name, par, value, alternative="rolling_demo"):
        return [cls, list(name) if isinstance(name, tuple) else name, par, value, alternative]

    return [
        r("model", "flextool", "solves", array([ROLL_SOLVE])),
        r("model", "flextool", "social_discount_rate", _f(0.0)),
        r("timeline", "hourly", "timestep_duration", _time_map((i, 1.0) for i in range(N_STEPS))),
        *(r("timeset", f"roll_ts_{p}", "timeline", _s("hourly")) for p in PERIODS),
        *(r("timeset", f"roll_ts_{p}", "timeset_duration", _time_map([(k * PERIOD_HOURS, float(PERIOD_HOURS))]))
          for k, p in enumerate(PERIODS)),
        r("group", "rolling_stochastic", "include_stochastics", _s("yes")),
        r("solve", ROLL_SOLVE, "solve_mode", _s("rolling_window")),
        r("solve", ROLL_SOLVE, "period_timeset", _period_map({p: f"roll_ts_{p}" for p in PERIODS})),
        r("solve", ROLL_SOLVE, "realized_periods", array(list(PERIODS))),
        r("solve", ROLL_SOLVE, "years_represented", _period_map({p: 1.0 for p in PERIODS})),
        r("solve", ROLL_SOLVE, "rolling_solve_jump", _f(JUMP)),
        r("solve", ROLL_SOLVE, "rolling_solve_horizon", _f(HORIZON)),
        r("solve", ROLL_SOLVE, "solver_mip_gap", _f(0.0)),
        r("solve", ROLL_SOLVE, "dispatch_realization_method", _s("deterministic")),
        r("solve", ROLL_SOLVE, "stochastic_branches", _roll_branches([
            ("high", WEIGHTS["high"], "no"), ("low", WEIGHTS["low"], "no"), ("actual", 0.0, "yes")])),
        r("solve", ROLL_SOLVE, "dispatch_realization_method", _s("within_solve"), "within_solve"),
        r("solve", ROLL_SOLVE, "dispatch_realization_method", _s("separate_dispatch"), "separate_dispatch"),
        r("solve", ROLL_SOLVE, "stochastic_branches", _roll_branches([("exact", 1.0, "no"), ("actual", 0.0, "yes")]),
          "perfect_forecast"),
        r("node", "elec", "node_type", _s("balance")),
        r("node", "elec", "inflow", _time_map((i, -demand(i)) for i in range(N_STEPS))),
        r("node", "elec", "penalty_up", _f(3000.0)),
        r("node", "elec", "penalty_down", _f(3000.0)),
        r("node", "coal_supply", "node_type", _s("commodity")),
        r("node", "gas_supply", "node_type", _s("commodity")),
        r("commodity", "coal", "price", _f(12.0)),
        r("commodity", "gas", "price", _f(28.0)),
        r("unit", "wind_farm", "existing", _f(650.0)),
        r("unit", "wind_farm", "efficiency", _f(1.0)),
        r("unit__node__profile", ("wind_farm", "elec", "wind_cf"), "profile_method", _s("upper_limit")),
        r("profile", "wind_cf", "profile", _profile({
            "high": _made(lambda a, i: forecast("high", a, i), HORIZON),
            "low": _made(lambda a, i: forecast("low", a, i), HORIZON),
            "actual": _made(lambda a, i: realized_wind(i), JUMP)})),
        r("profile", "wind_cf", "profile", _profile({
            "exact": _made(lambda a, i: realized_wind(i), HORIZON),
            "actual": _made(lambda a, i: realized_wind(i), JUMP)}), "perfect_forecast"),
        r("unit", "coal_units", "existing", _f(3.0)),  # units (virtual_unitsize 100 MW)
        r("unit", "coal_units", "virtual_unitsize", _f(100.0)),
        r("unit", "coal_units", "efficiency", _f(0.4)),
        r("unit", "coal_units", "min_load", _f(0.4)),
        r("unit", "coal_units", "conversion_method", _s("min_load_efficiency")),
        r("unit", "coal_units", "efficiency_at_min_load", _f(0.35)),
        r("unit", "coal_units", "startup_method", _s("integer_fix_in_uc")),
        r("unit", "coal_units", "startup_cost", _f(50.0)),
        r("unit", "coal_units", "minimum_time_method", _s("min_uptime")),
        r("unit", "coal_units", "min_uptime", _f(8.0)),
        r("unit", "gas_peaker", "existing", _f(350.0)),
        r("unit", "gas_peaker", "efficiency", _f(0.35)),
        r("unit", "gas_peaker", "min_load", _f(0.2)),
        r("unit", "gas_peaker", "conversion_method", _s("min_load_efficiency")),
        r("unit", "gas_peaker", "efficiency_at_min_load", _f(0.3)),
        r("unit", "gas_peaker", "startup_method", _s("linear")),
        r("unit", "gas_peaker", "startup_cost", _f(5.0)),
        r("node", "battery", "node_type", _s("storage")),
        r("node", "battery", "existing", _f(400.0)),
        r("node", "battery", "self_discharge_loss", _f(0.0005)),
        r("node", "battery", "storage_start_end_method", _s("fix_start")),
        r("node", "battery", "storage_state_start", _f(0.5)),
        r("node", "battery", "storage_solve_horizon_method", _s("use_reference_value")),
        r("node", "battery", "storage_state_reference_value", _f(0.5)),
        r("connection", "battery_inverter", "existing", _f(100.0)),
        r("connection", "battery_inverter", "efficiency", _f(0.92)),
        r("connection", "battery_inverter", "transfer_method", _s("regular")),
        r("connection", "battery_inverter", "other_operational_cost", _f(0.5)),
    ]


# --- investment data ----------------------------------------------------------------------------------------------

def invest_steps(period: str) -> list[str]:
    k = INVEST_PERIODS.index(period)
    return [step(k * INVEST_STEPS + j) for j in range(INVEST_STEPS)]


def invest_demand(branch: str, period: str) -> float:
    return DEMAND_2030 if period == "y2030" else DEMAND_2035[branch]


def _invest_branches(branches: tuple[str, ...]) -> list:
    """The fan at y2030's first step: the forecasts ``high`` / ``low`` (0.5 each) and, in *branches*, ``actual``."""
    rows = [(b, WEIGHTS[b], "no") for b in ("high", "low")] + [("actual", 0.0, "yes")] * ("actual" in branches)
    first = invest_steps("y2030")[0]
    return pack(pmap([("y2030", nested([(b, nested([(first, nested([(yes, w)], index_name="realized"))],
                                                    index_name="analysis_time")) for b, w, yes in rows],
                                       index_name="branch"))], index_name="period"), "map")


def _invest_inflow() -> list:
    """Demand as negative inflow, one series per branch made at y2030's first step covering both periods."""
    first = invest_steps("y2030")[0]
    return pack(pmap([(b, nested([(first, nested([(t, -invest_demand(b, p)) for p in INVEST_PERIODS
                                                  for t in invest_steps(p)], index_name="time"))],
                                 index_name="analysis_time"))
                      for b in ("high", "low", "actual")], index_name="branch"), "map")


def _investment_values() -> list:
    def v(cls, name, par, value, alternative="invest_demo"):
        return [cls, list(name) if isinstance(name, tuple) else name, par, value, alternative]

    every = list(INVEST_PERIODS)
    timesets = _period_map({p: f"invest_ts_{p}" for p in INVEST_PERIODS})
    years = _period_map({p: YEARS for p in INVEST_PERIODS})

    def invest_solve(name: str) -> list:
        return [v("solve", name, "solve_mode", _s("single_solve")),
                v("solve", name, "period_timeset", timesets),
                v("solve", name, "years_represented", years),
                v("solve", name, "invest_periods", array(every)),
                v("solve", name, "realized_invest_periods", array(["y2030"]))]

    return [
        v("model", "flextool", "solves", array(["invest_plan"])),
        v("model", "flextool", "social_discount_rate", _f(0.0)),
        v("timeline", "invest_hours", "timestep_duration",
          _time_map((i, 1.0) for i in range(len(INVEST_PERIODS) * INVEST_STEPS))),
        *(v("timeset", f"invest_ts_{p}", "timeline", _s("invest_hours")) for p in INVEST_PERIODS),
        *(v("timeset", f"invest_ts_{p}", "timeset_duration",
            _time_map([(k * INVEST_STEPS, float(INVEST_STEPS))])) for k, p in enumerate(INVEST_PERIODS)),
        v("group", "invest_stochastic", "include_stochastics", _s("yes")),
        *invest_solve("invest_plan"),
        v("solve", "invest_plan", "realized_periods", array(["y2030"])),
        v("solve", "invest_plan", "stochastic_branches", _invest_branches(("actual",))),
        v("node", "town", "node_type", _s("balance")),
        v("node", "town", "inflow", _invest_inflow()),
        v("node", "town", "penalty_up", _f(3000.0)),
        v("node", "town", "penalty_down", _f(3000.0)),
        v("unit", "new_plant", "existing", _f(0.0)),
        v("unit", "new_plant", "efficiency", _f(1.0)),
        v("unit", "new_plant", "virtual_unitsize", _f(1.0)),
        v("unit", "new_plant", "invest_method", _s("invest_no_limit")),
        v("unit", "new_plant", "invest_cost", _period_map(INVEST_COST)),
        v("unit", "new_plant", "discount_rate", _f(0.0)),
        v("unit", "new_plant", "lifetime_technical", _f(LIFETIME)),
        # investments_after_reveal_period, for invest_plan and invest_first
        v("solve", "invest_plan", "invest_solve_branching_method", _s("investments_after_reveal_period"),
          "after_reveal_period"),
        v("solve", "invest_first", "invest_solve_branching_method", _s("investments_after_reveal_period"),
          "after_reveal_period"),
        # the investment solve, then a dispatch solve of y2030
        v("model", "flextool", "solves", array(["invest_first", "dispatch_2030"]), "invest_then_dispatch"),
        *([*row[:4], "invest_then_dispatch"] for row in invest_solve("invest_first")),
        v("solve", "invest_first", "stochastic_branches", _invest_branches(()), "invest_then_dispatch"),
        v("solve", "dispatch_2030", "solve_mode", _s("single_solve"), "invest_then_dispatch"),
        v("solve", "dispatch_2030", "period_timeset", _period_map({"y2030": "invest_ts_y2030"}),
          "invest_then_dispatch"),
        v("solve", "dispatch_2030", "years_represented", _period_map({"y2030": YEARS}), "invest_then_dispatch"),
        v("solve", "dispatch_2030", "realized_periods", array(["y2030"]), "invest_then_dispatch"),
        v("solve", "dispatch_2030", "stochastic_branches", _invest_branches(("actual",)), "invest_then_dispatch"),
        v("solve", "dispatch_2030", "dispatch_realization_method", _s("separate_dispatch"), "invest_then_dispatch"),
    ]


# --- merge into stochastics.json ----------------------------------------------------------------------------------

def _norm(name) -> tuple:
    return tuple(name) if isinstance(name, (list, tuple)) else (name,)


OWNED = {(c, _norm(n)) for c, n in ENTITIES}


def strip(data: dict) -> dict:
    """*data* (a fixture's data sections) without the rows the examples own."""
    out = dict(data)
    out["alternatives"] = [a for a in data["alternatives"] if a[0] not in ALTERNATIVES]
    out["scenarios"] = [s for s in data["scenarios"] if s[0] not in SCENARIOS]
    out["scenario_alternatives"] = [s for s in data["scenario_alternatives"] if s[0] not in SCENARIOS]
    out["entities"] = [e for e in data["entities"] if (e[0], _norm(e[1])) not in OWNED]
    out["entity_alternatives"] = [e for e in data["entity_alternatives"]
                                  if e[2] not in ALTERNATIVES and (e[0], _norm(e[1])) not in OWNED]
    out["parameter_values"] = [p for p in data["parameter_values"]
                               if p[4] not in ALTERNATIVES and (p[0], _norm(p[1])) not in OWNED]
    return out


def build(source: Path = FIXTURE) -> dict:
    """The fixture *source* (``stochastics.json``) with the examples, as a packed spec."""
    from flextool.update_flextool.fixture_schema import schema_sections

    with open(source, encoding="utf-8") as f:
        data = strip(json.load(f))
    spec = schema_sections("packed")
    for section in ("alternatives", "scenarios", "scenario_alternatives", "entities", "entity_alternatives",
                    "parameter_values"):
        spec[section] = list(data[section])
    spec["alternatives"] += [[a, d] for a, d in ALTERNATIVES.items()]
    spec["scenarios"] += [[s, False, SCENARIO_DESCRIPTIONS[s]] for s in SCENARIOS]
    spec["scenario_alternatives"] += [[s, a, nxt] for s, alts in SCENARIOS.items()
                                      for a, nxt in zip(alts, alts[1:] + [None])]
    spec["entities"] += [[c, list(n) if isinstance(n, tuple) else n, None] for c, n in ENTITIES]
    alternative_of = {**{n: "rolling_demo" for c, n in ROLL_ENTITIES}, **{n: "invest_demo" for c, n in INVEST_ENTITIES}}
    spec["entity_alternatives"] += [[c, [n], alternative_of[n], True] for c, n in ENTITIES
                                    if c in ("node", "unit", "connection")]
    # The user constraint of the other scenarios is active by default; left in, every example solve would note
    # that it has no variable there (W-0004).
    spec["entity_alternatives"] += [["constraint", ["hydro_divide"], a, False] for a in ("rolling_demo", "invest_demo")]
    spec["parameter_values"] += _rolling_values() + _investment_values()
    return spec


def main() -> None:
    from flextool.update_flextool.test_fixtures import write_built_fixture

    write_built_fixture(build(), FIXTURE)
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
