"""LP snapshot harness: prove a model-layer refactor leaves the LP unchanged.

Builds the FlexTool LP for a fixed fixture set exactly as the production
solve path does (``run_chain_from_db`` -> ``run_orchestration`` ->
``build_flextool``) and writes every built :class:`polar_high.Problem` as
a named MPS file.  Two snapshots taken before and after a change are then
compared byte-for-byte (Tier A) or canonically up to row/column
permutation (Tier B).

Usage::

    # write OUT_DIR/<fixture>/<solve>.mps (+ manifest.json) for every fixture
    python tools/lp_snapshot.py snapshot _audit_reports/lp_baseline
    python tools/lp_snapshot.py snapshot OUT --fixtures base,coal_min_load -j 4

    # Tier A: byte compare (exit 1 on any difference / missing file)
    python tools/lp_snapshot.py compare _audit_reports/lp_baseline OUT
    # Tier B: same model up to row/column permutation, keyed by name
    python tools/lp_snapshot.py compare BASE NEW --canonical [--rtol 1e-12]

    # constraint-family coverage of a snapshot (family -> fixtures)
    python tools/lp_snapshot.py coverage _audit_reports/lp_baseline

    python tools/lp_snapshot.py list        # show the fixture set

How it works
------------
* Each fixture runs in its own ``python`` subprocess with a fresh random
  ``PYTHONHASHSEED`` so a snapshot also proves cross-process determinism.
* The fixture's Spine DB is built from a JSON fixture (``tests/fixtures/``
  or ``flextool/schemas/canonical_databases/``) with
  ``tests/db_utils.json_to_db`` (canonical how-to JSON:
  ``initialize_database``) + ``migrate_database`` (never from a
  checked-in ``.sqlite``), optionally edited by a named JSON patch that
  appends an override alternative and a new scenario.
* ``flextool.engine_polars.model.build_flextool`` is wrapped (in the
  worker process only; production code is untouched).  Right after each
  build returns -- before autoscale or any solver touches the Problem --
  the Problem is written with ``Problem.write_mps(emit_names=True)``.
  The objective constant (which MPS cannot carry) goes to
  ``manifest.json``.
* The chain runs with ``warm=False`` so every solve of a multi-solve
  chain (rolling, multi-period, stochastic, separate dispatch) is a cold
  ``build_flextool`` call and gets its own MPS.  Solves still run (the
  HiGHS ``cmd_solve_mps`` subprocess) because later solves of a chain
  are built from earlier solutions.  A solve name built more than once
  (rolling windows) gets ``__2``, ``__3``, ... suffixes in build order;
  ``manifest.json`` lists the files in build order and is compared too.
* ``coverage`` greps every ``add_cstr`` name in COVERAGE_SOURCES (plus
  the hand-resolved dynamic names in DYNAMIC_CSTR_NAMES) and reports
  which fixtures emit rows of each family; UNREACHABLE lists families no
  production build can emit.

Known non-determinism (production, not the harness): the RHS of
``node_storage_usage_fix_le`` sums several coarse-window usages with a
polars ``group_by().sum()`` whose input order follows an unordered
``.unique()`` (model.py, storage usage fix block), so
``nested_fix_usage/dispatch_fullYear_roll.mps`` can differ in the last
bits between processes.  Byte compare flags it; canonical compare with
``--rtol 1e-12`` accepts it.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO / "tests"
ENGINE_DIR = REPO / "flextool" / "engine_polars"
CANONICAL_PREFIX = "flextool/schemas/canonical_databases/"

# Files whose ``add_cstr`` calls define the constraint-family universe
# checked by ``coverage``.
COVERAGE_SOURCES = (
    "model.py", "_reserve.py", "_group_slack.py", "_cumulative_invest.py",
    "_delay.py", "_commodity_ladder.py", "_dc_power_flow.py",
    "_online.py", "_unit_floors.py",
)


# ---------------------------------------------------------------------------
# JSON patches (applied to the fixture JSON before import)
# ---------------------------------------------------------------------------

def _no_patch(data: dict) -> dict:
    return data


def _pack(x) -> list:
    """Spine import-format packed value (tests/fixtures JSON encoding)."""
    if isinstance(x, bool):
        raise TypeError("use 'yes'/'no' strings")
    if isinstance(x, (int, float)):
        payload, kind = float(x), "float"
    elif isinstance(x, str):
        payload, kind = x, "str"
    elif isinstance(x, dict) and x.get("type") == "array":
        payload = {k: v for k, v in x.items() if k != "type"}
        kind = "array"
    elif isinstance(x, dict):
        payload, kind = x, "map"
    else:
        raise TypeError(f"cannot pack {x!r}")
    return [base64.b64encode(json.dumps(payload).encode()).decode(), kind]


def _pmap(values: dict, index_name: str = "period") -> dict:
    """1-d str-indexed map."""
    return {"index_type": "str", "index_name": index_name,
            "data": [[k, v] for k, v in values.items()]}


def _tmap(values: list[float], start: int = 1) -> dict:
    """time map t0001.. over ``values``."""
    return _pmap({f"t{start + i:04d}": v for i, v in enumerate(values)},
                 "time")


def _chain(data: dict, scenario: str) -> list[str]:
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    if not rows:
        raise KeyError(f"scenario {scenario!r} not in fixture JSON")
    nxt = {a: b for _, a, b in rows}
    first = [a for a in nxt if a not in set(nxt.values())]
    assert len(first) == 1, f"ambiguous alternative chain for {scenario!r}"
    chain, alt = [], first[0]
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


_ACTIVATED_CLASSES = ("node", "unit", "connection")


def _overlay(data: dict, *, name: str, base: str | list[str],
             values: list, entities: list = (),
             param_defs: list = ()) -> dict:
    """Append override alternative ``name`` (entities, values) on top of
    the alternative chain of scenario ``base`` (or an explicit chain) and
    register scenario ``name``.  New node/unit/connection entities are
    activated in the override alternative."""
    chain = _chain(data, base) if isinstance(base, str) else list(base)
    data["alternatives"].append([name, ""])
    known = {(e[0], json.dumps(e[1])) for e in data["entities"]}
    for cls, ent in entities:
        if (cls, json.dumps(ent)) not in known:
            data["entities"].append([cls, ent, None])
            known.add((cls, json.dumps(ent)))
        if cls in _ACTIVATED_CLASSES:
            data["entity_alternatives"].append([cls, [ent], name, True])
    for cls, pname, desc in param_defs:
        data["parameter_definitions"].append(
            [cls, pname, ["bnVsbA==", None], None, desc, None])
    for cls, ent, param, val in values:
        data["parameter_values"].append([cls, ent, param, _pack(val), name])
    data["scenarios"].append([name, False, ""])
    full = chain + [name]
    for a, before in zip(full, full[1:] + [None]):
        data["scenario_alternatives"].append([name, a, before])
    return data


_P4 = ("p2020", "p2025", "p2030", "p2035")


def _per(v: float) -> dict:
    return _pmap({d: v for d in _P4})


def _invest_retire_values(cls: str, ent: str, scale: float) -> list:
    """invest_retire_period_total limits (min AND max, period AND total)."""
    return [
        [cls, ent, "invest_method", "invest_retire_period_total"],
        [cls, ent, "invest_max_period", _per(400 * scale)],
        [cls, ent, "invest_min_period", _per(1 * scale)],
        [cls, ent, "invest_max_total", 900 * scale],
        [cls, ent, "invest_min_total", 2 * scale],
        [cls, ent, "retire_max_period", _per(20 * scale)],
        [cls, ent, "retire_min_period", _per(0.5 * scale)],
        [cls, ent, "retire_max_total", 40 * scale],
        [cls, ent, "retire_min_total", 1 * scale],
        # without a salvage value the entity gets no divest variable
        [cls, ent, "salvage_value", 100.0],
    ]


# ``group`` has no retire_* parameters in the schema, yet the engine reads
# group retire_{max,min}_{period,total} (p_group_retire_*, g_divest_total)
# and invest_{max,min}_cumulative (p_group_invest_*_cumulative,
# g_invest_cumulative).  Registering them as extra parameter definitions
# in the patched JSON is the only way to reach those constraint families.
_OFF_SCHEMA_GROUP_PARAMS = [
    ["group", name, "off-schema: read by the engine, absent from the schema"]
    for name in ("retire_max_period", "retire_min_period",
                 "retire_max_total", "retire_min_total",
                 "invest_max_cumulative", "invest_min_cumulative")
]


def _group_limit_values(g: str) -> list:
    return [
        ["group", g, "invest_method", "invest_retire_period_total"],
        ["group", g, "invest_max_period", _per(5000.0)],
        ["group", g, "invest_min_period", _per(1.0)],
        ["group", g, "retire_max_period", _per(5000.0)],
        ["group", g, "retire_min_period", _per(0.1)],
        ["group", g, "invest_max_total", 9000.0],
        ["group", g, "invest_min_total", 2.0],
        ["group", g, "retire_max_total", 9000.0],
        ["group", g, "retire_min_total", 0.2],
    ]


def _patch_invest_limits(data: dict) -> dict:
    """Entity + group invest/retire limits (min and max, period and total),
    off-schema group cumulative invest caps, flowGroup cumulative/instant
    flow limits — multi-period single-solve invest model."""
    return _overlay(
        data, name="invest_limits",
        base="network_coal_wind_battery_invest_cumulative",
        entities=[["group", "inv_g"], ["group__unit", ["inv_g", "coal_plant"]],
                  ["group__node", ["inv_g", "battery"]],
                  ["group", "cum_g"], ["group__unit", ["cum_g", "coal_plant"]],
                  ["group__node", ["cum_g", "battery"]]],
        param_defs=_OFF_SCHEMA_GROUP_PARAMS,
        values=(_invest_retire_values("unit", "coal_plant", 1.0)
                + _invest_retire_values("node", "battery", 1.0)
                + _group_limit_values("inv_g")
                + [["group", "cum_g", "invest_method", "cumulative_limits"],
                   ["group", "cum_g", "invest_max_cumulative", 50000.0],
                   ["group", "cum_g", "invest_min_cumulative", 1.0],
                   ["flowGroup", "to_west_node", "max_cumulative_flow", 5000.0],
                   ["flowGroup", "to_west_node", "min_cumulative_flow", 1.0],
                   ["flowGroup", "to_west_node", "max_instant_flow", 5000.0],
                   ["flowGroup", "to_west_node", "min_instant_flow", 0.5]]))


def _patch_cumulative_capacity(data: dict) -> dict:
    """cumulative_min_capacity on a unit and a storage node (on top of the
    fixture's cumulative_max_capacity); flowGroup cumulative flow per
    period (map)."""
    return _overlay(
        data, name="cumulative_capacity",
        base="network_coal_wind_battery_invest_cumulative",
        values=[["unit", "coal_plant", "cumulative_min_capacity",
                 _per(510.0)],
                ["node", "battery", "cumulative_min_capacity", _per(55.0)],
                ["flowGroup", "to_west_node", "max_cumulative_flow",
                 _per(5000.0)],
                ["flowGroup", "to_west_node", "min_cumulative_flow",
                 _per(1.0)]])


def _patch_stoch_invest_storage(data: dict) -> dict:
    """Recourse + non-anticipative invest window with an invest/divest
    storage node and a group with invest/retire limits."""
    return _overlay(
        data, name="stoch_invest_storage", base="hedge_na_divest",
        param_defs=_OFF_SCHEMA_GROUP_PARAMS,
        entities=[["node", "store"], ["connection", "store_conn"],
                  ["connection__node__node", ["store_conn", "elec", "store"]],
                  ["group", "inv_g"], ["group__unit", ["inv_g", "base"]],
                  ["group__node", ["inv_g", "store"]]],
        values=[["node", "store", "node_type", "storage"],
                ["node", "store", "existing", 20.0],
                ["node", "store", "virtual_unitsize", 1.0],
                ["node", "store", "invest_method", "invest_retire_total"],
                # scalar: a period map here next to the fixture's map
                # without index_name crashes _per_entity_period_cost
                # (column-name axis detection, schema mismatch in concat)
                ["node", "store", "invest_cost", 1.0],
                ["node", "store", "invest_max_total", 50.0],
                ["node", "store", "retire_max_total", 10.0],
                ["node", "store", "salvage_value", 0.5],
                ["node", "store", "discount_rate", 0.05],
                ["node", "store", "lifetime", 1.0],
                ["node", "store", "penalty_up", 10000.0],
                ["node", "store", "penalty_down", 10000.0],
                ["connection", "store_conn", "existing", 50.0],
                ["connection", "store_conn", "efficiency", 0.9],
                ["group", "inv_g", "invest_method",
                 "invest_retire_period_total"],
                ["group", "inv_g", "invest_max_period",
                 _pmap({"p2035": 500.0, "p2040": 500.0})],
                ["group", "inv_g", "invest_min_period",
                 _pmap({"p2035": 0.1, "p2040": 0.1})],
                ["group", "inv_g", "retire_max_period",
                 _pmap({"p2035": 500.0, "p2040": 500.0})],
                ["group", "inv_g", "invest_max_total", 900.0],
                ["group", "inv_g", "invest_min_total", 0.2],
                ["group", "inv_g", "retire_max_total", 900.0],
                # no group retire minimum: infeasible on this fixture; the
                # >= divest-group families are covered by invest_limits
                # (a >= divest-group total never takes the per-path form)
                ])


_CHP_CHAIN = ["init", "west", "coal_chp", "heat"]
_GAS_ENTITIES = [["commodity", "gas"], ["node", "gas_market"],
                 ["commodity__node", ["gas", "gas_market"]],
                 ["unit__inputNode", ["coal_chp", "gas_market"]]]


def _reserve(r: str, ud: str, group: str, providers: list,
             method: str = "timeseries_only", reservation: float = 50.0,
             penalty: float = 1000.0) -> tuple[list, list]:
    """Entities + values of reserve ``r``/``ud`` on ``group`` offered by
    ``providers`` = [(class, unit_or_connection, node), ...]."""
    grp = [r, ud, group]
    ents = [["upDown", ud], ["reserve", r], ["reserve__upDown__group", grp]]
    vals = [["reserve__upDown__group", grp, "reserve_method", method],
            ["reserve__upDown__group", grp, "penalty_reserve", penalty]]
    if reservation:
        vals.append(["reserve__upDown__group", grp, "reservation",
                     reservation])
    for cls, p, n in providers:
        rel = f"reserve__upDown__{cls}__node"
        ents.append([rel, [r, ud, p, n]])
        vals += [[rel, [r, ud, p, n], "is_enabled", "yes"],
                 [rel, [r, ud, p, n], "max_share", 1.0],
                 [rel, [r, ud, p, n], "reliability", 1.0]]
    return ents, vals


def _patch_reserve_floors(data: dict) -> dict:
    """Downward reserve on a direct and a two-output unit without online
    variables: minFlow_reserve / minFlow_reserve_arc."""
    ents, vals = _reserve("rx", "down", "electricity",
                          [("unit", "coal_plant", "west"),
                           ("unit", "coal_chp", "west")])
    return _overlay(
        data, name="reserve_floors",
        base=["init", "west", "coal", "coal_chp", "heat"], entities=ents,
        values=vals + [["constraint", "coal_chp_fix", "is_enabled", "no"],
                       ["node", "west", "inflow", -300.0]])


def _chp_floor(data: dict, name: str, startup: str) -> dict:
    ents, vals = _reserve("rx", "down", "electricity",
                          [("unit", "coal_chp", "west")])
    return _overlay(
        data, name=name, base=_CHP_CHAIN, entities=_GAS_ENTITIES + ents,
        values=vals + [
            ["constraint", "coal_chp_fix", "is_enabled", "no"],
            ["node", "gas_market", "node_type", "commodity"],
            ["commodity", "gas", "price", 100.0],
            ["node", "west", "inflow", -300.0],
            ["node", "heat", "inflow", -500.0],
            ["unit__outputNode", ["coal_chp", "west"], "capacity_min_coeff",
             0.5],
            ["unit__inputNode", ["coal_chp", "gas_market"],
             "input_share_min", 0.3],
            ["unit", "coal_chp", "startup_method", startup],
            ["unit", "coal_chp", "min_load", 0.4]])


def _patch_chp_floor_linear(data: dict) -> dict:
    """Online (linear) two-output unit with a per-output floor, downward
    reserve and an input mixing minimum."""
    return _chp_floor(data, "chp_floor_linear", "linear")


def _patch_chp_floor_integer(data: dict) -> dict:
    """As chp_floor_linear with integer (binary) online variables."""
    return _chp_floor(data, "chp_floor_integer", "binary")


def _patch_reserve_dynamic_n1_down(data: dict) -> dict:
    """Dynamic reserve (increase_reserve_ratio) and an n-1 DOWN reserve."""
    dyn = ["dyn", "up", "electricity"]
    n1d = ["n1d", "down", "electricity"]
    rel = "reserve__upDown__unit__node"
    ents = [["reserve", "dyn"], ["reserve__upDown__group", dyn],
            [rel, ["dyn", "up", "wind_plant", "west"]],
            [rel, ["dyn", "up", "coal_plant", "west"]],
            ["reserve", "n1d"], ["reserve__upDown__group", n1d],
            [rel, ["n1d", "down", "dr_increase_demand", "west"]],
            [rel, ["n1d", "down", "coal_plant", "west"]]]
    vals = [
        ["reserve__upDown__group", dyn, "reserve_method", "dynamic_only"],
        ["reserve__upDown__group", dyn, "penalty_reserve", 1.0],
        [rel, ["dyn", "up", "wind_plant", "west"], "increase_reserve_ratio",
         0.5],
        [rel, ["dyn", "up", "wind_plant", "west"], "is_enabled", "yes"],
        [rel, ["dyn", "up", "coal_plant", "west"], "is_enabled", "yes"],
        [rel, ["dyn", "up", "coal_plant", "west"], "max_share", 0.01],
        [rel, ["dyn", "up", "coal_plant", "west"], "reliability", 1.0],
        ["reserve__upDown__group", n1d, "reserve_method",
         "large_failure_only"],
        ["reserve__upDown__group", n1d, "penalty_reserve", 20000.0],
        [rel, ["n1d", "down", "dr_increase_demand", "west"],
         "large_failure_ratio", 1.0],
        [rel, ["n1d", "down", "dr_increase_demand", "west"], "is_enabled",
         "yes"],
        [rel, ["n1d", "down", "coal_plant", "west"], "is_enabled", "yes"],
        [rel, ["n1d", "down", "coal_plant", "west"], "max_share", 1.0],
        [rel, ["n1d", "down", "coal_plant", "west"], "reliability", 1.0],
        ["unit", "dr_increase_demand", "min_load", 0.8],
        ["unit", "dr_increase_demand", "startup_method", "linear"],
    ]
    # timeseries "activator" requirements on west_north (activate the
    # reserves' relationships, as in tests/engine_polars)
    for r, ud in (("dyn", "up"), ("n1d", "down")):
        grp = [r, ud, "west_north"]
        ents.append(["reserve__upDown__group", grp])
        vals += [["reserve__upDown__group", grp, "reserve_method",
                  "timeseries_only"],
                 ["reserve__upDown__group", grp, "reservation", 1.0],
                 ["reserve__upDown__group", grp, "penalty_reserve", 20000.0]]
    return _overlay(
        data, name="reserve_dynamic_n1_down",
        base=["init", "west", "coal", "wind", "dr_increase_demand"],
        entities=ents, values=vals)


def _patch_reserve_storage_duration(data: dict) -> dict:
    """Up/down reserve from a storage-connected provider with a
    reserve_duration (energy-sustain constraints)."""
    e_up, v_up = _reserve("rs", "up", "electricity",
                          [("connection", "battery_inverter", "west")])
    e_dn, v_dn = _reserve("rs", "down", "electricity",
                          [("connection", "battery_inverter", "west")])
    vals = v_up + v_dn + [
        ["reserve__upDown__group", ["rs", "up", "electricity"],
         "reserve_duration", 2.0],
        ["reserve__upDown__group", ["rs", "down", "electricity"],
         "reserve_duration", 2.0]]
    return _overlay(data, name="reserve_storage_duration",
                    base="wind_battery", entities=e_up + e_dn, values=vals)


def _patch_stoch_na_reserve(data: dict) -> dict:
    """Reserve in the stochastic 2-day dispatch (non-anticipativity window
    over the realized part): non_anticipativity_reserve."""
    ents, vals = _reserve("r1", "up", "res_g",
                          [("unit", "coal_plant", "demand_node"),
                           ("unit", "gas_plant", "demand_node")],
                          reservation=100.0)
    return _overlay(data, name="stoch_na_reserve",
                    base="2_day_stochastic_dispatch",
                    entities=[["group", "res_g"],
                              ["group__node", ["res_g", "demand_node"]]]
                    + ents, values=vals)


def _patch_chp_misc(data: dict) -> dict:
    """greater_than user constraint, balance_within_period node and ramp
    limits on both the input and the output side of a unit."""
    return _overlay(
        data, name="chp_misc",
        base=["init", "west", "coal", "coal_chp", "heat"],
        values=[
            ["constraint", "coal_chp_fix", "sense", "greater_than"],
            ["node", "heat", "node_type", "balance_within_period"],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_method",
             "ramp_limit"],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_up",
             0.2],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_down",
             0.2],
            ["unit__inputNode", ["coal_plant", "coal_market"], "ramp_method",
             "ramp_limit"],
            ["unit__inputNode", ["coal_plant", "coal_market"],
             "ramp_speed_up", 0.3],
            ["unit__inputNode", ["coal_plant", "coal_market"],
             "ramp_speed_down", 0.3]])


def _patch_profiles(data: dict) -> dict:
    """fixed / lower_limit flow profiles and a fixed storage-state
    profile (the EV fixture keeps its upper/lower state profiles)."""
    return _overlay(
        data, name="profiles",
        base="coal_wind_ev",
        entities=[["profile", "coal_floor"], ["profile", "ev_fix"],
                  ["unit__node__profile", ["coal_plant", "west",
                                           "coal_floor"]],
                  ["node__profile", ["ev_battery", "ev_fix"]]],
        values=[["profile", "coal_floor", "profile", 0.1],
                ["unit__node__profile", ["coal_plant", "west", "coal_floor"],
                 "profile_method", "lower_limit"],
                ["unit__node__profile", ["wind_plant", "west",
                                         "wind_profile"],
                 "profile_method", "fixed"],
                # between the EV state lower (<= 0.5) and upper (>= 0.6)
                ["profile", "ev_fix", "profile", 0.55],
                ["node__profile", ["ev_battery", "ev_fix"],
                 "profile_method", "fixed"]])


def _patch_co2_period_total(data: dict) -> dict:
    """co2_method period_total: co2_max_period AND co2_max_total."""
    return _overlay(
        data, name="co2_period_total", base="multi_year_one_solve_co2_limit",
        values=[["group", "co2_limit", "co2_method", "period_total"],
                ["group", "co2_limit", "co2_max_total", 2.0e7]])


def _patch_negcap_indirect(data: dict) -> dict:
    """Negative-capacity unit made indirect (second output):
    maxOutputSum_negCap."""
    return _overlay(
        data, name="negcap_indirect",
        base=["init", "west", "heat", "anti_energy_plant"],
        entities=[["unit__outputNode", ["anti_energy_plant", "heat"]]],
        # each output arc is pinned to coeff x capacity (maxFlow AND
        # maxFlow_negCap); the coefficients must sum to 1 for the
        # conversion (efficiency 0.5, input pinned to 2 units) to hold
        values=[["unit__outputNode", ["anti_energy_plant", node],
                 "capacity_max_coeff", 0.5]
                for node in ("anti_energy_market", "heat")])


def _nested(data: dict, name: str, method: str) -> dict:
    """Nested invest -> storage -> rolling dispatch chain with fewer
    rolls (horizon 36 h, jump 24 h) and the given nested-fix method."""
    return _overlay(
        data, name=name,
        base="multi_fullYear_battery_nested_24h_invest_one_solve",
        values=[["solve", "dispatch_fullYear_roll", "rolling_solve_horizon",
                 36.0],
                ["solve", "dispatch_fullYear_roll", "rolling_solve_jump",
                 24.0],
                ["node", "battery", "storage_nested_fix_method", method]])


def _patch_nested_fix_quantity(data: dict) -> dict:
    return _nested(data, "nested_fix_quantity", "fix_quantity")


def _patch_nested_fix_usage(data: dict) -> dict:
    return _nested(data, "nested_fix_usage", "fix_usage")


def _patch_nested_fix_price(data: dict) -> dict:
    return _nested(data, "nested_fix_price", "fix_price")


def _rp_weights(bases: list[str], reps: tuple[str, str]) -> dict:
    data = []
    for i, b in enumerate(bases):
        w = (i + 1) / (len(bases) + 1)
        data.append([b, {"type": "map", "index_type": "str",
                         "index_name": "representative_period", "rank": 1,
                         "data": [[reps[0], w], [reps[1], 1.0 - w]]}])
    return {"type": "map", "index_type": "str", "index_name": "base_period",
            "rank": 1, "data": data}


def _rp_multi_base(data: dict, name: str, methods: dict) -> dict:
    """Representative periods with seven base periods (an inter-period
    storage chain, rp_inter_period_balance) and the given blended-weights
    storage binding method per LH2 storage node."""
    bases = [f"t{1 + 24 * k:04d}" for k in range(7)]
    sfx = "__lh2_three_region_rp_invest"
    return _overlay(
        data, name=name, base="lh2_three_region_rp_invest",
        values=[
            ["timeset", "rp_y2030" + sfx, "representative_period_weights",
             _rp_weights(bases, ("t0001", "t0025"))],
            ["timeset", "rp_y2040" + sfx, "representative_period_weights",
             _rp_weights(bases, ("t0049", "t0073"))],
        ] + [["node", n, "storage_binding_method", m]
             for n, m in methods.items()])


def _patch_rp_multi_base_period(data: dict) -> dict:
    return _rp_multi_base(data, "rp_multi_base_period", {
        n: "bind_within_period_blended_weights"
        for n in ("lh2_A", "lh2_B", "lh2_C")})


def _patch_rp_multi_base_solve(data: dict) -> dict:
    # within_solve and forward_only may share a solve; within_period may
    # not be mixed with either (FlexToolConfigError).
    return _rp_multi_base(data, "rp_multi_base_solve", {
        "lh2_A": "bind_within_solve_blended_weights",
        "lh2_B": "bind_forward_only_blended_weights",
        "lh2_C": "bind_within_solve_blended_weights"})


PATCHES: dict[str, Callable[[dict], dict]] = {
    "none": _no_patch,
    "invest_limits": _patch_invest_limits,
    "cumulative_capacity": _patch_cumulative_capacity,
    "stoch_invest_storage": _patch_stoch_invest_storage,
    "reserve_floors": _patch_reserve_floors,
    "chp_floor_linear": _patch_chp_floor_linear,
    "chp_floor_integer": _patch_chp_floor_integer,
    "reserve_dynamic_n1_down": _patch_reserve_dynamic_n1_down,
    "reserve_storage_duration": _patch_reserve_storage_duration,
    "stoch_na_reserve": _patch_stoch_na_reserve,
    "chp_misc": _patch_chp_misc,
    "profiles": _patch_profiles,
    "co2_period_total": _patch_co2_period_total,
    "negcap_indirect": _patch_negcap_indirect,
    "nested_fix_quantity": _patch_nested_fix_quantity,
    "nested_fix_usage": _patch_nested_fix_usage,
    "nested_fix_price": _patch_nested_fix_price,
    "rp_multi_base_period": _patch_rp_multi_base_period,
    "rp_multi_base_solve": _patch_rp_multi_base_solve,
}


# ---------------------------------------------------------------------------
# Fixture set
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fixture:
    name: str            # output sub-directory
    json: str            # JSON fixture, path relative to the repo root
    scenario: str
    patch: str = "none"  # key into PATCHES


_T = "tests/fixtures/tests.json"
_F = "tests/fixtures/"
_H = CANONICAL_PREFIX

# Unpatched scenarios: chosen for constraint-family coverage (see
# ``coverage``) plus breadth of objective terms, variables and solve
# structures (multi-solve chains, rolling, stochastic, separate dispatch,
# representative periods, blocks, nested solves).
_PLAIN: tuple[tuple[str, str, str], ...] = (
    # (fixture name, json, scenario)
    ("base", _T, "base"),
    ("all", _T, "all"),
    ("test_a_lot", _T, "test_a_lot"),
    ("network_all_tech", _T, "network_all_tech"),
    ("coal_chp", _T, "coal_chp"),
    ("coal_chp_extraction", _T, "coal_chp_extraction"),
    ("coal_min_load", _T, "coal_min_load"),
    ("coal_min_load_MIP_wind", _T, "coal_min_load_MIP_wind"),
    ("coal_wind_min_uptime", _T, "coal_wind_min_uptime"),
    ("coal_wind_min_uptime_MIP", _T, "coal_wind_min_uptime_MIP"),
    ("coal_ramp_limit", _T, "coal_ramp_limit"),
    ("coal_retire", _T, "coal_retire"),
    ("coal_co2_limit", _T, "coal_co2_limit"),
    ("coal_co2_price", _T, "coal_co2_price"),
    ("coal_ladder_annual", _T, "coal_ladder_annual"),
    ("coal_ladder_cumulative", _T, "coal_ladder_cumulative"),
    ("coal_wind_ev", _T, "coal_wind_ev"),
    ("coal_wind_inertia", _T, "coal_wind_inertia"),
    ("capacity_margin", _T, "capacity_margin"),
    ("lossless_2way", _T, "lossless_2way"),
    ("unidirectional_connection", _T, "unidirectional_connection"),
    ("water_pump_delayed", _T, "water_pump_delayed"),
    ("delay_source_coef", _T, "delay_source_coef"),
    ("dr_shift_demand", _T, "dr_shift_demand"),
    ("network_coal_wind_reserve_n_1", _T, "network_coal_wind_reserve_n_1"),
    ("network_coal_wind_battery_invest_cumulative", _T,
     "network_coal_wind_battery_invest_cumulative"),
    ("multi_year", _T, "multi_year"),
    ("multi_year_one_solve_battery", _T, "multi_year_one_solve_battery"),
    ("multi_year_wind_growth_cap", _T, "multi_year_wind_growth_cap"),
    ("wind_battery_invest_lifetime_renew_4solve", _T,
     "wind_battery_invest_lifetime_renew_4solve"),
    ("y2020_2029_2x5y", _T, "y2020_2029_2x5y"),
    ("5weeks_battery_intraperiod_blocks", _T,
     "5weeks_battery_intraperiod_blocks"),
    ("5weeks_invest_fullYear_dispatch_coal_wind", _T,
     "5weeks_invest_fullYear_dispatch_coal_wind"),
    ("lh2_three_region", _F + "lh2_three_region.json", "lh2_three_region"),
    ("lh2_three_region_invest", _F + "lh2_three_region.json",
     "lh2_three_region_invest"),
    ("lh2_three_region_rp_invest", _F + "lh2_three_region_rp_invest.json",
     "lh2_three_region_rp_invest"),
    ("h2_trade_parity", _F + "h2_trade_parity.json",
     "scenario_test_6h_no_carrier_storage"),
    ("dc_opf_test", _F + "case14_dc_power_flow.json", "dc_opf_test"),
    ("stochastic_2_day_dispatch", _F + "stochastics.json",
     "2_day_stochastic_dispatch"),
    ("stochastic_1_week_rolling_wind", _F + "stochastics.json",
     "1_week_rolling_wind"),
    ("stoch_two_period", _F + "stoch_two_period.json", "stoch"),
    ("stoch_recourse", _F + "stoch_two_period_invest.json", "recourse"),
    ("stoch_recourse_na", _F + "stoch_two_period_invest_na.json",
     "recourse_na"),
    ("stoch_hedge_na_window", _F + "stoch_two_period_hedge.json",
     "hedge_na_window"),
    ("stoch_pinned_ops", _F + "stoch_two_period_free_ops.json",
     "pinned_ops"),
    ("stoch_midhorizon_nreal0", _F + "stoch_two_period_midhorizon_nreal0.json",
     "midhorizon"),
    ("stoch_t0_commit", _F + "stoch_t0_commit.json", "stoch_t0_commit"),
    ("stoch_realized_dispatch", _F + "stoch_realized_dispatch.json",
     "w8_realized_dispatch"),
    ("separate_dispatch_noinvest", _F + "separate_dispatch_noinvest.json",
     "separate_dispatch_noinvest"),
    ("rolling_separate_dispatch", _F + "rolling_separate_dispatch.json",
     "rolling_separate_dispatch"),
    ("howto_non_sync_curtailment", _H + "howto_non_sync_and_curtailment.json",
     "curtailment"),
    ("howto_ramp_and_start_up", _H + "howto_ramp_and_start_up.json",
     "ramp_and_start_up"),
    ("howto_hydro_with_pump", _H + "howto_hydro_reservoir_with_pump.json",
     "hydro_with_pump"),
)

# Patched scenarios: an override alternative (PATCHES[name]) appended to
# a fixture scenario's alternative chain, registered as scenario ``name``.
_PATCHED: tuple[tuple[str, str], ...] = (
    ("invest_limits", _T),
    ("cumulative_capacity", _T),
    ("stoch_invest_storage", _F + "stoch_two_period_hedge.json"),
    ("reserve_floors", _T),
    ("chp_floor_linear", _T),
    ("chp_floor_integer", _T),
    ("reserve_dynamic_n1_down", _T),
    ("reserve_storage_duration", _T),
    ("stoch_na_reserve", _F + "stochastics.json"),
    ("chp_misc", _T),
    ("profiles", _T),
    ("co2_period_total", _T),
    ("negcap_indirect", _T),
    ("nested_fix_quantity", _T),
    ("nested_fix_usage", _T),
    ("nested_fix_price", _T),
    ("rp_multi_base_period", _F + "lh2_three_region_rp_invest.json"),
    ("rp_multi_base_solve", _F + "lh2_three_region_rp_invest.json"),
)

FIXTURES: tuple[Fixture, ...] = (
    *(Fixture(n, j, sc) for n, j, sc in _PLAIN),
    *(Fixture(n, j, n, patch=n) for n, j in _PATCHED),
)


def _fixture_by_name() -> dict[str, Fixture]:
    out: dict[str, Fixture] = {}
    for f in FIXTURES:
        if f.name in out:
            raise ValueError(f"duplicate fixture name {f.name!r}")
        out[f.name] = f
    return out


# ---------------------------------------------------------------------------
# Worker (runs inside the per-fixture subprocess)
# ---------------------------------------------------------------------------

def _solve_name_from_stack() -> str:
    """Return the ``complete_solve_name`` of the orchestration frame that
    is (transitively) calling ``build_flextool``."""
    frame = sys._getframe(2)
    while frame is not None:
        name = frame.f_locals.get("complete_solve_name")
        if isinstance(name, str) and name:
            return name
        frame = frame.f_back
    return "solve"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


def _worker(json_name: str, scenario: str, patch: str, out_dir: Path) -> int:
    os.environ.setdefault("FLEXTOOL_SKIP_SOLVER_PROBE", "1")
    sys.path.insert(0, str(TESTS_DIR))
    import warnings

    from db_utils import json_to_db  # tests/db_utils.py

    import flextool.engine_polars.model as model_mod
    from flextool.engine_polars import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from flextool.update_flextool.initialize_database import (
        initialize_database,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    used: dict[str, int] = {}
    original = model_mod.build_flextool

    def capturing_build_flextool(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        solve = _solve_name_from_stack()
        stem = _safe(solve)
        used[stem] = used.get(stem, 0) + 1
        if used[stem] > 1:
            stem = f"{stem}__{used[stem]}"
        path = out_dir / f"{stem}.mps"
        with warnings.catch_warnings():
            # objective offset is recorded in the manifest instead
            warnings.simplefilter("ignore", UserWarning)
            m.write_mps(path, emit_names=True)
        manifest.append({
            "file": path.name,
            "solve": solve,
            "obj_offset": repr(float(m._obj_offset)),
            "build_kwargs": {k: repr(v) for k, v in sorted(kwargs.items())},
        })
        return result

    model_mod.build_flextool = capturing_build_flextool

    with tempfile.TemporaryDirectory(prefix="lp_snapshot_") as tmp:
        tmp_path = Path(tmp)
        data = json.loads((REPO / json_name).read_text())
        data = PATCHES[patch](data)
        src_json = tmp_path / "fixture.json"
        src_json.write_text(json.dumps(data))
        db_path = tmp_path / "fixture.sqlite"
        if json_name.startswith(CANONICAL_PREFIX):
            # canonical how-to JSON (export_database format)
            if patch != "none":
                raise ValueError("patches apply to tests/fixtures JSON only")
            initialize_database(str(src_json), str(db_path))
            url = f"sqlite:///{db_path.resolve().as_posix()}"
        else:
            url = json_to_db(src_json, db_path)
        # same defensive migration the test DB fixtures apply
        migrate_database(url)
        solver_cfg = tmp_path / "solver_config"
        solver_cfg.mkdir()
        shutil.copy(TESTS_DIR / "highs.opt", solver_cfg / "highs.opt")
        work = tmp_path / "work"
        work.mkdir()
        run_chain_from_db(
            input_db_url=url,
            scenario_name=scenario,
            work_folder=work,
            solver_config_dir=solver_cfg,
            warm=False,
        )

    if not manifest:
        print("ERROR: no Problem was built through build_flextool",
              file=sys.stderr)
        return 2
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

def _run_fixture(fx: Fixture, out_root: Path) -> tuple[str, int, float, str]:
    out_dir = out_root / fx.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(secrets.randbelow(2**32 - 1) + 1)
    env["FLEXTOOL_SKIP_SOLVER_PROBE"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker",
           "--json", fx.json, "--scenario", fx.scenario,
           "--patch", fx.patch, "--out", str(out_dir)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO),
                          capture_output=True, text=True)
    dt = time.monotonic() - t0
    (out_dir / "_run.log").write_text(
        f"PYTHONHASHSEED={env['PYTHONHASHSEED']}\nexit={proc.returncode}\n"
        f"seconds={dt:.1f}\n--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}\n")
    tail = "" if proc.returncode == 0 else proc.stderr.strip()[-2000:]
    return fx.name, proc.returncode, dt, tail


def cmd_snapshot(args: argparse.Namespace) -> int:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    by_name = _fixture_by_name()
    if args.fixtures:
        names = [n.strip() for n in args.fixtures.split(",") if n.strip()]
        unknown = [n for n in names if n not in by_name]
        if unknown:
            print(f"unknown fixtures: {unknown}", file=sys.stderr)
            return 2
        fixtures = [by_name[n] for n in names]
    else:
        fixtures = list(FIXTURES)
    t0 = time.monotonic()
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run_fixture, fx, out_root) for fx in fixtures]
        for fut in cf.as_completed(futs):
            name, rc, dt, tail = fut.result()
            n_mps = len(list((out_root / name).glob("*.mps")))
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"{name:45s} {status:16s} {n_mps:3d} mps  {dt:6.1f}s",
                  flush=True)
            if rc != 0:
                failed.append(name)
                print(f"    {tail}", flush=True)
    total = sum(1 for _ in out_root.glob("*/*.mps"))
    print(f"\n{len(fixtures)} fixtures, {total} MPS files, "
          f"{time.monotonic() - t0:.1f}s wall")
    if failed:
        print(f"FAILED: {', '.join(sorted(failed))}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# MPS parsing (free format as written by polar_high's write_mps)
# ---------------------------------------------------------------------------

@dataclass
class MpsModel:
    obj_row: str
    obj_sense: str
    rows: dict[str, str]                       # row -> E/L/G
    cols: dict[str, bool]                      # col -> is_integer
    coefs: dict[tuple[str, str], float]        # (row, col) -> value
    obj: dict[str, float]                      # col -> objective coef
    rhs: dict[str, float]                      # row -> rhs
    obj_rhs: float                             # RHS on the objective row
    ranges: dict[str, float]
    bounds: dict[str, tuple[float, float]]     # col -> (lb, ub)


def parse_mps(path: Path) -> MpsModel:
    rows: dict[str, str] = {}
    cols: dict[str, bool] = {}
    coefs: dict[tuple[str, str], float] = {}
    obj: dict[str, float] = {}
    rhs: dict[str, float] = {}
    ranges: dict[str, float] = {}
    bound_recs: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    obj_row = ""
    obj_sense = "MIN"
    obj_rhs = 0.0
    section = ""
    integer = False
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                tok = line.split()
                section = tok[0]
                if section == "OBJSENSE" and len(tok) > 1:
                    obj_sense = tok[1]
                continue
            tok = line.split()
            if section == "OBJSENSE":
                obj_sense = tok[0]
            elif section == "ROWS":
                kind, name = tok[0], tok[1]
                if kind == "N":
                    if not obj_row:
                        obj_row = name
                else:
                    rows[name] = kind
            elif section == "COLUMNS":
                if len(tok) >= 3 and tok[1] == "'MARKER'":
                    integer = tok[2] == "'INTORG'"
                    continue
                col = tok[0]
                if col not in cols:
                    cols[col] = integer
                for i in range(1, len(tok) - 1, 2):
                    r, v = tok[i], float(tok[i + 1])
                    if r == obj_row:
                        obj[col] = obj.get(col, 0.0) + v
                    else:
                        coefs[(r, col)] = coefs.get((r, col), 0.0) + v
            elif section == "RHS":
                pairs = tok[1:] if len(tok) % 2 == 1 else tok
                for i in range(0, len(pairs) - 1, 2):
                    r, v = pairs[i], float(pairs[i + 1])
                    if r == obj_row:
                        obj_rhs = v
                    else:
                        rhs[r] = v
            elif section == "RANGES":
                pairs = tok[1:] if len(tok) % 2 == 1 else tok
                for i in range(0, len(pairs) - 1, 2):
                    ranges[pairs[i]] = float(pairs[i + 1])
            elif section == "BOUNDS":
                kind = tok[0]
                col = tok[2]
                val = float(tok[3]) if len(tok) > 3 else None
                bound_recs[col].append((kind, val))
    # a column with no nonzero (no row, no objective) appears only in
    # BOUNDS: declare it (continuous) so it is still compared
    for col in bound_recs:
        if col not in cols:
            cols[col] = False
    inf = math.inf
    bounds: dict[str, tuple[float, float]] = {}
    for col in cols:
        lb, ub = 0.0, inf
        for kind, val in bound_recs.get(col, ()):
            if kind == "UP":
                ub = val
            elif kind == "LO":
                lb = val
            elif kind == "FX":
                lb = ub = val
            elif kind == "FR":
                lb, ub = -inf, inf
            elif kind == "MI":
                lb = -inf
            elif kind == "PL":
                ub = inf
            elif kind == "BV":
                lb, ub = 0.0, 1.0
            elif kind == "LI":
                lb = val
            elif kind == "UI":
                ub = val
            else:
                raise ValueError(f"{path}: unknown bound type {kind!r}")
        bounds[col] = (lb, ub)
    return MpsModel(obj_row, obj_sense, rows, cols, coefs, obj, rhs, obj_rhs,
                    ranges, bounds)


_FAMILY_RE = re.compile(r"^([^\[(]+)")


def family_of(name: str) -> str:
    m = _FAMILY_RE.match(name)
    return m.group(1) if m else name


def _close(a: float, b: float, rtol: float) -> bool:
    if a == b:
        return True
    if rtol == 0.0 or math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def canonical_diff(a: MpsModel, b: MpsModel, rtol: float,
                   off_a: float, off_b: float) -> list[str]:
    """Return human-readable differences (empty list == equivalent)."""
    msgs: list[str] = []
    fam: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))

    if a.obj_sense != b.obj_sense:
        msgs.append(f"objective sense {a.obj_sense} != {b.obj_sense}")
    if not _close(off_a + a.obj_rhs, off_b + b.obj_rhs, rtol):
        msgs.append(f"objective constant {off_a + a.obj_rhs!r} != "
                    f"{off_b + b.obj_rhs!r}")

    for r in a.rows.keys() - b.rows.keys():
        fam[family_of(r)]["rows only in base"] += 1
    for r in b.rows.keys() - a.rows.keys():
        fam[family_of(r)]["rows only in new"] += 1
    for r in a.rows.keys() & b.rows.keys():
        if a.rows[r] != b.rows[r]:
            fam[family_of(r)]["sense differs"] += 1
        if not _close(a.rhs.get(r, 0.0), b.rhs.get(r, 0.0), rtol):
            fam[family_of(r)]["rhs differs"] += 1
        ra, rb = a.ranges.get(r), b.ranges.get(r)
        if (ra is None) != (rb is None) or (
                ra is not None and not _close(ra, rb, rtol)):
            fam[family_of(r)]["range differs"] += 1

    for c in a.cols.keys() - b.cols.keys():
        fam["col:" + family_of(c)]["cols only in base"] += 1
    for c in b.cols.keys() - a.cols.keys():
        fam["col:" + family_of(c)]["cols only in new"] += 1
    for c in a.cols.keys() & b.cols.keys():
        f = "col:" + family_of(c)
        if a.cols[c] != b.cols[c]:
            fam[f]["integrality differs"] += 1
        la, ua = a.bounds[c]
        lb, ub = b.bounds[c]
        if not (_close(la, lb, rtol) and _close(ua, ub, rtol)):
            fam[f]["bounds differ"] += 1
    for c in a.obj.keys() | b.obj.keys():
        if not _close(a.obj.get(c, 0.0), b.obj.get(c, 0.0), rtol):
            fam["col:" + family_of(c)]["objective coef differs"] += 1

    for key in a.coefs.keys() | b.coefs.keys():
        va, vb = a.coefs.get(key), b.coefs.get(key)
        f = family_of(key[0])
        if va is None:
            fam[f]["coef only in new"] += 1
        elif vb is None:
            fam[f]["coef only in base"] += 1
        elif not _close(va, vb, rtol):
            fam[f]["coef differs"] += 1

    for f in sorted(fam):
        detail = ", ".join(f"{k}={v}" for k, v in sorted(fam[f].items()))
        msgs.append(f"{f}: {detail}")
    return msgs


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def _offsets(fixture_dir: Path) -> dict[str, float]:
    mf = fixture_dir / "manifest.json"
    if not mf.is_file():
        return {}
    return {e["file"]: float(e["obj_offset"])
            for e in json.loads(mf.read_text())}


def cmd_compare(args: argparse.Namespace) -> int:
    base, new = Path(args.base_dir), Path(args.new_dir)
    for d in (base, new):
        if not d.is_dir():
            print(f"not a directory: {d}", file=sys.stderr)
            return 2
    rel_a = {p.relative_to(base) for pat in ("*/*.mps", "*/manifest.json")
             for p in base.glob(pat)}
    rel_b = {p.relative_to(new) for pat in ("*/*.mps", "*/manifest.json")
             for p in new.glob(pat)}
    n_same = n_diff = n_missing = 0
    for rel in sorted(rel_a | rel_b):
        pa, pb = base / rel, new / rel
        if rel not in rel_b:
            print(f"MISSING in new : {rel}")
            n_missing += 1
            continue
        if rel not in rel_a:
            print(f"MISSING in base: {rel}")
            n_missing += 1
            continue
        if not args.canonical:
            same = pa.read_bytes() == pb.read_bytes()
            print(f"{'identical' if same else 'DIFFERENT'}  {rel}")
            n_same += same
            n_diff += not same
            continue
        if rel.name == "manifest.json":
            # solve order / names matter; offsets are checked per MPS
            ma = [(e["file"], e["solve"]) for e in json.loads(pa.read_text())]
            mb = [(e["file"], e["solve"]) for e in json.loads(pb.read_text())]
            same = sorted(ma) == sorted(mb)
            print(f"{'equivalent' if same else 'DIFFERENT'}  {rel}")
            n_same += same
            n_diff += not same
            continue
        off_a = _offsets(pa.parent).get(rel.name, 0.0)
        off_b = _offsets(pb.parent).get(rel.name, 0.0)
        msgs = canonical_diff(parse_mps(pa), parse_mps(pb), args.rtol,
                              off_a, off_b)
        if msgs:
            n_diff += 1
            print(f"DIFFERENT   {rel}")
            for m in msgs:
                print(f"    {m}")
        else:
            n_same += 1
            print(f"equivalent  {rel}")
    label = "equivalent" if args.canonical else "identical"
    print(f"\n{n_same} {label}, {n_diff} different, {n_missing} missing")
    return 0 if (n_diff == 0 and n_missing == 0) else 1


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

_ADD_CSTR_LITERAL = re.compile(r"add_cstr\(\s*[\"']([A-Za-z_][A-Za-z0-9_]*)[\"']")

# Names passed to ``add_cstr`` through a variable or an f-string, resolved
# by reading the call sites (source file -> names).
DYNAMIC_CSTR_NAMES: dict[str, tuple[str, ...]] = {
    "model.py": (
        # _eq_invest_period_caps / _eq_invest_total_caps loop tables
        "maxInvest_entity_period_p", "maxInvest_entity_period_n",
        "maxDivest_entity_period_p", "maxDivest_entity_period_n",
        "maxInvest_entity_total", "maxInvest_entity_total_path",
        "maxDivest_entity_total", "maxDivest_entity_total_path",
        "maxInvest_entity_total_n", "maxInvest_entity_total_path_n",
        "maxDivest_entity_total_n", "maxDivest_entity_total_path_n",
        # _add_non_anticipativity_invest_constraints._tie(name, ...)
        "non_anticipativity_invest_p", "non_anticipativity_invest_n",
        "non_anticipativity_divest_p", "non_anticipativity_divest_n",
        # f"ramp_{side}_{dir_}_constraint"
        "ramp_sink_up_constraint", "ramp_sink_down_constraint",
        "ramp_source_up_constraint", "ramp_source_down_constraint",
        # user constraints (process_constraint_*)
        "process_constraint_equal", "process_constraint_less_than",
        "process_constraint_greater_than",
        # _eq_state_profiles loop table
        "profile_state_upper_limit", "profile_state_lower_limit",
        "profile_state_fixed",
        # _eq_profiles loop table
        "profile_flow_upper_limit", "profile_flow_lower_limit",
        "profile_flow_fixed",
    ),
    "_online.py": (
        # f"...{sfx}", sfx=_{kind}
        *(f"{base}_{kind}"
          for base in ("maxOnline", "maxStartup", "maxShutdown",
                       "online__startup", "online__shutdown",
                       "maxFlow_online", "maxOutputSum_online",
                       "minimum_uptime", "minimum_downtime",
                       "online_fix_eq")
          for kind in ("linear", "integer")),
    ),
    "_unit_floors.py": (
        # f"...{sfx}", sfx=_{kind}
        *(f"{base}_{kind}"
          for base in ("minFlow_minload", "minFlow_output_floor")
          for kind in ("linear", "integer")),
    ),
    "_reserve.py": (
        "reserveBalance_up_n_1_eq", "reserveBalance_down_n_1_eq",
        "reserve_process_upward", "reserve_process_downward",
    ),
    "_cumulative_invest.py": (
        # _emit_invest_period_minmax: f"min{Kind}_entity_period_{p,n}"
        "minInvest_entity_period_p", "minInvest_entity_period_n",
        "minDivest_entity_period_p", "minDivest_entity_period_n",
        # _emit_invest_total_minmax: f"min{Kind}_entity_total_{p,n}"
        "minInvest_entity_total_p", "minInvest_entity_total_n",
        "minDivest_entity_total_p", "minDivest_entity_total_n",
        # _emit_cumulative_capacity
        "maxCumulative_capacity_p", "maxCumulative_capacity_n",
        "minCumulative_capacity_p", "minCumulative_capacity_n",
        # _emit_group_invest_period: f"{pre}{Kind}Group_entity_period_{p,n}"
        *(f"{pre}{kind}Group_entity_period_{s}"
          for pre in ("max", "min") for kind in ("Invest", "Divest")
          for s in ("p", "n")),
        # _emit_group_invest_total: invest -> _{p,n}; divest -> legacy/path
        *(f"{pre}InvestGroup_entity_total_{s}"
          for pre in ("max", "min") for s in ("p", "n")),
        *(f"{pre}DivestGroup_entity_total_{s}"
          for pre in ("max", "min") for s in ("p", "n")),
        # _emit_group_total_divest_cap emits path_name only for "<="
        "maxDivestGroup_entity_total_path_p",
        "maxDivestGroup_entity_total_path_n",
        # _emit_group_invest_cumulative / cumulative flow / instant flow
        *(f"{pre}{tail}"
          for pre in ("max", "min")
          for tail in ("InvestGroup_entity_cumulative",
                       "Cumulative_flow_solve", "Cumulative_flow_period",
                       "Instant_flow")),
    ),
}

# Families no production build can emit, with the reason.  ``coverage``
# reports them separately and does not count them as gaps.
UNREACHABLE: dict[str, str] = {
    name: ("ed_invest_set (the v_invest_* domain) already anti-joins "
           "ed_invest_forbidden_no_investment (_derived_params."
           "ed_invest_set_from_source), so the row domain "
           "forbidden ∩ v_invest is always empty")
    for name in ("fix_v_invest_no_investment_eq_p",
                 "fix_v_invest_no_investment_eq_n")
}

# Literal names that are only ever expanded with a suffix in the source.
_LITERAL_IS_TEMPLATE = {"maxCumulative_capacity", "minCumulative_capacity"}


def constraint_universe() -> dict[str, str]:
    """family name -> source file, for every add_cstr in COVERAGE_SOURCES."""
    out: dict[str, str] = {}
    for fname in COVERAGE_SOURCES:
        text = (ENGINE_DIR / fname).read_text()
        for name in _ADD_CSTR_LITERAL.findall(text):
            if name not in _LITERAL_IS_TEMPLATE:
                out.setdefault(name, fname)
        for name in DYNAMIC_CSTR_NAMES.get(fname, ()):
            out.setdefault(name, fname)
    return out


def _row_families(mps: Path) -> set[str]:
    fams: set[str] = set()
    section = ""
    with open(mps) as fh:
        for line in fh:
            if line and not line[0].isspace():
                section = line.split()[0] if line.strip() else section
                if section != "ROWS" and fams:
                    break
                continue
            if section == "ROWS":
                tok = line.split()
                if tok and tok[0] != "N":
                    fams.add(family_of(tok[1]))
    return fams


def cmd_coverage(args: argparse.Namespace) -> int:
    root = Path(args.snapshot_dir)
    seen: dict[str, set[str]] = defaultdict(set)
    for mps in sorted(root.glob("*/*.mps")):
        for fam in _row_families(mps):
            seen[fam].add(mps.parent.name)
    universe = constraint_universe()
    missing = []
    print(f"{'family':48s} {'source':22s} fixtures")
    for fam in sorted(universe, key=lambda n: (universe[n], n)):
        fx = sorted(seen.get(fam, ()))
        if not fx and fam not in UNREACHABLE:
            missing.append(fam)
        shown = ", ".join(fx[:6]) + (f", +{len(fx) - 6}" if len(fx) > 6 else "")
        if not fx:
            shown = ("-- unreachable --" if fam in UNREACHABLE
                     else "-- NOT COVERED --")
        print(f"{fam:48s} {universe[fam]:22s} {shown}")
    extra = sorted(set(seen) - set(universe))
    if extra:
        print("\nrow families not from the coverage sources:")
        for fam in extra:
            print(f"  {fam}: {', '.join(sorted(seen[fam])[:6])}")
    if UNREACHABLE:
        print("\nunreachable families:")
        for fam, why in UNREACHABLE.items():
            print(f"  {fam}: {why}")
    n_unr = sum(1 for f in universe if f in UNREACHABLE and not seen.get(f))
    print(f"\n{len(universe) - len(missing) - n_unr}/{len(universe)} "
          f"families covered, {n_unr} unreachable, {len(missing)} missing")
    if missing:
        print("NOT COVERED: " + ", ".join(missing))
        return 1
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    for f in FIXTURES:
        patch = "" if f.patch == "none" else f"  patch={f.patch}"
        print(f"{f.name:45s} {f.json}:{f.scenario}{patch}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="write named MPS per fixture")
    sp.add_argument("out_dir")
    sp.add_argument("--fixtures", default="",
                    help="comma-separated subset of fixture names")
    sp.add_argument("-j", "--jobs", type=int,
                    default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("compare", help="compare two snapshots")
    sp.add_argument("base_dir")
    sp.add_argument("new_dir")
    sp.add_argument("--canonical", action="store_true",
                    help="compare up to row/column permutation, by name")
    sp.add_argument("--rtol", type=float, default=0.0,
                    help="relative float tolerance (canonical mode)")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("coverage", help="constraint family -> fixtures")
    sp.add_argument("snapshot_dir")
    sp.set_defaults(func=cmd_coverage)

    sp = sub.add_parser("list", help="list the fixture set")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("_worker", help=argparse.SUPPRESS)
    sp.add_argument("--json", required=True)
    sp.add_argument("--scenario", required=True)
    sp.add_argument("--patch", default="none")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=lambda a: _worker(a.json, a.scenario, a.patch,
                                           Path(a.out)))

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
