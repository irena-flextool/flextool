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

    # work-folder modes (any combination; plain = none of them)
    python tools/lp_snapshot.py snapshot OUT --fixtures base --csv-dump
    python tools/lp_snapshot.py snapshot OUT --fixtures conn_online_regular \
        --dirty-with conn_exact_flow
    python tools/lp_snapshot.py snapshot OUT --fixtures base,multi_year \
        --dirty-with self --dirty-sqlite self

    # Tier A: byte compare (exit 1 on any difference / missing file)
    python tools/lp_snapshot.py compare _audit_reports/lp_baseline OUT
    # Tier B: same model up to row/column permutation, keyed by name
    python tools/lp_snapshot.py compare BASE NEW --canonical [--rtol 1e-12]
    # restrict either tier to some fixture dirs (subset vs a full base)
    python tools/lp_snapshot.py compare BASE NEW --fixtures base,multi_year

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
* ``build_flextool`` in ``MODEL_MODULE`` is wrapped (in the
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
* Work-folder modes (``snapshot`` options; the captured target run is
  otherwise the plain run, and MPS naming / ``manifest.json`` are the
  same).  The mode line goes into each fixture's ``_run.log``.

  ``--csv-dump``
      The target runs with ``csv_dump=True`` and, after the run, the last
      step's provider is written with ``snapshot_processed_inputs(work)``,
      as the CLI ``--csv-dump`` does.
  ``--dirty-with FIXTURE|self``
      CSV dirt only.  Before the captured run, FIXTURE (``self`` = the
      target fixture itself) is built from its JSON (+ patch) into its own
      DB file and run in the same work folder, uncaptured, with
      ``csv_dump=True, keep_solutions=True, warm=False``.  Then every
      step's ``flex_data.dump_csvs(work, include_heavy=True)`` is written
      in step order (later steps win per file, as a per-sub-solve dump
      does) and the last step's provider ``snapshot_processed_inputs(work)``.
      No ``tests.sqlite`` is written.
  ``--dirty-sqlite FIXTURE|self``
      ``tests.sqlite`` dirt only: FIXTURE's DB file (``self`` = the
      target's own DB) is copied to ``work/tests.sqlite`` before the
      captured run (after any ``--dirty-with`` run).

  What the dirt can show: same-model CSV dirt (``--dirty-with self``)
  changes no LP even where the loader reads disk, because the dumped
  values equal the live ones; it is a regression guard, not a
  fail-before.  A different model's CSVs can (``conn_online_regular``
  after ``conn_exact_flow``).  A ``tests.sqlite`` that holds the target's
  scenario makes the loader build axis Enums from it; that changes the LP
  of some fixtures (``conn_online_regular`` with ``self``) and leaves
  others byte-identical; a DB without the scenario is ignored.  A target
  that fails under dirt is reported as FAILED like any other fixture.
* ``coverage`` greps every ``add_cstr`` name in COVERAGE_SOURCES (plus
  the hand-resolved dynamic names in DYNAMIC_CSTR_NAMES) and reports
  which fixtures emit rows of each family; UNREACHABLE lists families no
  production build can emit.

Known non-determinism (production, not the harness): the RHS of
``node_storage_usage_fix_le`` sums several coarse-window usages with a
polars ``group_by().sum()`` whose input order follows an unordered
``.unique()`` (model/core.py, storage usage fix block), so
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
ENGINE_DIR = REPO / "flextool" / "engine"
CANONICAL_PREFIX = "flextool/schemas/canonical_databases/"

# Module whose ``build_flextool`` attribute the worker wraps.  Every
# production caller must look it up there at call time (late
# ``from <MODEL_MODULE> import build_flextool``), so the patch is seen.
MODEL_MODULE = "flextool.engine.model.core"

# Files whose ``add_cstr`` calls define the constraint-family universe
# checked by ``coverage``: POSIX paths relative to ENGINE_DIR.
COVERAGE_SOURCES = (
    "model/core.py", "model/_reserve.py", "model/_group_slack.py",
    "model/_cumulative_invest.py", "model/_delay.py",
    "model/_commodity_ladder.py", "model/_dc_power_flow.py",
    "model/_online.py", "model/_unit_floors.py",
    "model/_non_anticipativity.py", "model/_exact.py",
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
                # kept scalar: fixture stability
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


def _patch_reserve_up_online(data: dict) -> dict:
    """Upward reserve from a direct and a two-output unit with linear
    online variables: the reserve terms of maxFlow_online /
    maxOutputSum_online."""
    ents, vals = _reserve("ru", "up", "electricity",
                          [("unit", "coal_plant", "west"),
                           ("unit", "coal_chp", "west")])
    return _overlay(
        data, name="reserve_up_online",
        base=["init", "west", "coal", "coal_chp", "heat"], entities=ents,
        values=vals + [["constraint", "coal_chp_fix", "is_enabled", "no"],
                       ["node", "west", "inflow", -300.0],
                       ["node", "heat", "inflow", -500.0],
                       ["unit", "coal_plant", "startup_method", "linear"],
                       ["unit", "coal_plant", "min_load", 0.5],
                       ["unit", "coal_chp", "startup_method", "linear"],
                       ["unit", "coal_chp", "min_load", 0.4]])


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
    # timeseries "activator" requirements on west_north: dynamic and n-1
    # reserves no longer need them to be active (demanded_reserves); kept
    # so the fixture's LP stays the same
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


_REL_UNIT = "reserve__upDown__unit__node"
# e_boiler: a direct unit west -> heat (efficiency 0.5, 100 MW)
_E_BOILER = [["unit", "e_boiler"],
             ["unit__inputNode", ["e_boiler", "west"]],
             ["unit__outputNode", ["e_boiler", "heat"]]]
_E_BOILER_VALUES = [["unit", "e_boiler", "efficiency", 0.5],
                    ["unit", "e_boiler", "existing", 100.0],
                    ["node", "heat", "penalty_up", 1000.0]]


def _with_max_share(vals: list, rel: list, share: float) -> list:
    return [v for v in vals if not (v[1] == rel and v[2] == "max_share")] + [
        [_REL_UNIT, rel, "max_share", share]]


def _patch_reserve_input_side(data: dict) -> dict:
    """Downward reserve at the input node of a direct unit (e_boiler, linear
    online) and of an indirect unit (coal_chp fuel at coal_market, linear
    online): the source-side reserve in the arc capacity rows and
    max_share of the input capacity."""
    e1, v1 = _reserve("rd", "down", "electricity",
                      [("unit", "e_boiler", "west")], penalty=20000.0)
    e2, v2 = _reserve("rf", "down", "fuel",
                      [("unit", "coal_chp", "coal_market")],
                      reservation=300.0, penalty=20000.0)
    v1 = _with_max_share(v1, ["rd", "down", "e_boiler", "west"], 0.2)
    v2 = _with_max_share(v2, ["rf", "down", "coal_chp", "coal_market"], 0.5)
    return _overlay(
        data, name="reserve_input_side",
        base=["init", "west", "coal", "coal_chp", "heat"],
        entities=_E_BOILER + e1 + e2,
        values=_E_BOILER_VALUES + v1 + v2 + [
            ["unit", "e_boiler", "conversion_method", "constant_efficiency"],
            ["constraint", "coal_chp_fix", "is_enabled", "no"],
            ["node", "west", "inflow", -250.0],
            ["unit", "e_boiler", "startup_method", "linear"],
            ["unit", "e_boiler", "min_load", 0.2],
            ["unit", "coal_chp", "startup_method", "linear"],
            ["unit", "coal_chp", "min_load", 0.4]])


def _patch_reserve_dynamic_demand(data: dict) -> dict:
    """Dynamic and n-1 requirements without any reservation: dz (up,
    dynamic_only, group increase_reserve_ratio 0.1 and e_boiler's
    consumption ratio 1 at west) and nz (down, large_failure_only,
    e_boiler's consumption fails); e_boiler is a min_load_efficiency unit
    with linear online variables."""
    grp_z = ["dz", "up", "electricity"]
    grp_n = ["nz", "down", "electricity"]
    ents = [["upDown", "up"], ["upDown", "down"], ["reserve", "dz"],
            ["reserve", "nz"], ["reserve__upDown__group", grp_z],
            ["reserve__upDown__group", grp_n]]
    vals = [["reserve__upDown__group", grp_z, "reserve_method", "dynamic_only"],
            ["reserve__upDown__group", grp_z, "penalty_reserve", 1000.0],
            ["reserve__upDown__group", grp_z, "increase_reserve_ratio", 0.1],
            ["reserve__upDown__group", grp_n, "reserve_method",
             "large_failure_only"],
            ["reserve__upDown__group", grp_n, "penalty_reserve", 1000.0]]
    for rel, extra in (
            (["dz", "up", "coal_plant", "west"], [("max_share", 0.1)]),
            (["dz", "up", "e_boiler", "west"],
             [("max_share", 0.0), ("increase_reserve_ratio", 1.0)]),
            (["nz", "down", "coal_plant", "west"], [("max_share", 1.0)]),
            (["nz", "down", "e_boiler", "west"],
             [("max_share", 0.0), ("large_failure_ratio", 1.0)])):
        ents.append([_REL_UNIT, rel])
        vals += [[_REL_UNIT, rel, "is_enabled", "yes"],
                 [_REL_UNIT, rel, "reliability", 1.0]]
        vals += [[_REL_UNIT, rel, k, v] for k, v in extra]
    return _overlay(
        data, name="reserve_dynamic_demand",
        base=["init", "west", "coal", "heat"], entities=_E_BOILER + ents,
        values=_E_BOILER_VALUES + vals + [
            ["unit", "e_boiler", "conversion_method", "min_load_efficiency"],
            ["unit", "e_boiler", "efficiency_at_min_load", 0.4],
            ["unit", "e_boiler", "min_load", 0.3],
            ["unit", "e_boiler", "startup_method", "linear"],
            ["node", "west", "inflow", -250.0]])


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


def _patch_co2_cap_minload(data: dict) -> dict:
    """CO2 price, period cap and total cap on a min_load_efficiency unit
    (linear online): the section fuel of the units online in
    co2_price_eff, co2_max_period and co2_max_total."""
    return _overlay(
        data, name="co2_cap_minload",
        base=["init", "west", "coal", "co2_limit", "coal_min_load"],
        values=[["group", "co2_limit", "co2_method", "price_period_total"],
                ["group", "co2_limit", "co2_price", 10.0],
                ["group", "co2_limit", "co2_max_total", 2.5e6]])


def _patch_co2_period_rolling(data: dict) -> dict:
    """co2_max_period in a rolling solve over two periods (p2020 and
    p2025, 2 days each, horizon 30 h / jump 6 h): rolls inside one
    period, rolls across the period boundary and the truncated tail."""
    s = "y2020_2day_dispatch"
    return _overlay(
        data, name="co2_period_rolling", base="coal_co2_limit",
        values=[["group", "co2_limit", "co2_max_period",
                 _pmap({"p2020": 1.0e6, "p2025": 0.5e6})],
                ["solve", s, "period_timeset",
                 _pmap({"p2020": "2day", "p2025": "2day"})],
                ["solve", s, "realized_periods",
                 {"type": "array", "value_type": "str",
                  "index_name": "period", "data": ["p2020", "p2025"]}],
                ["solve", s, "solve_mode", "rolling_window"],
                ["solve", s, "rolling_solve_horizon", 30.0],
                ["solve", s, "rolling_solve_jump", 6.0]])


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
    storage binding method per LH2 storage node.

    The base periods are the seven days of the week the timeline
    ``y2030_168h`` is named after (t0001, t0025, ..., t0145).  The fixture
    JSON keeps only the first 96 hours (the four representative days), so
    the overlay extends the timeline to 168 hours: a base period must start
    at a step of the timeline."""
    bases = [f"t{1 + 24 * k:04d}" for k in range(7)]
    sfx = "__lh2_three_region_rp_invest"
    week = _pmap({f"t{i:04d}": 1.0 for i in range(1, 169)}, "time")
    return _overlay(
        data, name=name, base="lh2_three_region_rp_invest",
        values=[
            ["timeline", "y2030_168h", "timestep_duration", week],
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
    # The across-solve variants: y2040's base periods continue from the last
    # base period of y2030 (rp_period_previous); lh2_A / lh2_C close the
    # two-period chain once, lh2_B (forward_only) not at all.  Any blended
    # variants may share a solve.
    return _rp_multi_base(data, "rp_multi_base_solve", {
        "lh2_A": "bind_within_solve_blended_weights",
        "lh2_B": "bind_forward_only_blended_weights",
        "lh2_C": "bind_within_solve_blended_weights"})


_RAMP_COAL = [
    ["unit__outputNode", ["coal_plant", "west"], "ramp_method", "ramp_limit"],
    ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_up", 0.002],
    ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_down", 0.002],
]


def _ramp_on(cls: str, rel: list, speed: float) -> list:
    return [[cls, rel, "ramp_method", "ramp_limit"],
            [cls, rel, "ramp_speed_up", speed],
            [cls, rel, "ramp_speed_down", speed]]


def _patch_ramp_step_duration(data: dict) -> dict:
    """coal ramp 0.002/min with 2-hour steps t0001-t0024 and 1-hour steps
    t0025-t0048: the ramp room scales with the step length."""
    vals = {f"t{i:04d}": (2.0 if i <= 24 else 1.0) for i in range(1, 49)}
    return _overlay(data, name="ramp_step_duration", base=["init", "west", "coal"],
                    values=_RAMP_COAL + [["timeline", "y2020", "timestep_duration", _pmap(vals, "time")]])


def _patch_ramp_blocks(data: dict) -> dict:
    """coal ramp on the 5weeks timeset (blocks with gaps in the timeline,
    two periods)."""
    return _overlay(data, name="ramp_blocks", base=["init", "west", "coal", "coal_ramp", "5weeks"],
                    values=_RAMP_COAL)


def _patch_ramp_reserve_online(data: dict) -> dict:
    """coal_plant (linear online, min_load 0.3, ramp 0.005/min) with upward
    and downward reserve at west; e_boiler (direct, linear online, min_load
    0.2) with a ramp on its input arc and downward reserve at west (consume
    more): reserve in the ramp rows and the startup / shutdown capacity
    reduction."""
    eu, vu = _reserve("ru", "up", "electricity", [("unit", "coal_plant", "west")], penalty=5000.0,
                      reservation=40.0)
    ed, vd = _reserve("rd", "down", "electricity", [("unit", "coal_plant", "west"), ("unit", "e_boiler", "west")],
                      penalty=5000.0, reservation=40.0)
    return _overlay(
        data, name="ramp_reserve_online", base=["init", "west", "coal", "coal_chp", "heat"],
        entities=_E_BOILER + eu + ed,
        values=_E_BOILER_VALUES + vu + vd
        + _ramp_on("unit__outputNode", ["coal_plant", "west"], 0.005)
        + _ramp_on("unit__inputNode", ["e_boiler", "west"], 0.002) + [
            ["unit", "e_boiler", "conversion_method", "constant_efficiency"],
            ["constraint", "coal_chp_fix", "is_enabled", "no"],
            ["node", "west", "inflow", -250.0],
            ["unit", "coal_plant", "startup_method", "linear"],
            ["unit", "coal_plant", "min_load", 0.3],
            ["unit", "e_boiler", "startup_method", "linear"],
            ["unit", "e_boiler", "min_load", 0.2],
            ["unit", "coal_plant", "startup_cost", 20.0],
            ["unit", "e_boiler", "startup_cost", 5.0]])


def _two_period_values() -> list:
    return [
        ["timeset", "ta", "timeline", "y2020"],
        ["timeset", "tb", "timeline", "y2020"],
        ["timeset", "ta", "timeset_duration", _pmap({"t0001": 24.0}, "time")],
        ["timeset", "tb", "timeset_duration", _pmap({"t0025": 24.0}, "time")],
        ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "ta", "p2025": "tb"})],
        ["solve", "y2020_2day_dispatch", "realized_periods",
         {"type": "array", "value_type": "str", "data": ["p2020", "p2025"]}],
    ]


def _patch_ramp_two_periods(data: dict) -> dict:
    """p2020 = t0001-t0024 and p2025 = t0025-t0048 of one timeline: the
    second period continues the first, so its first step ramps from
    p2020's last step."""
    return _overlay(data, name="ramp_two_periods", base=["init", "west", "coal"],
                    entities=[["timeset", "ta"], ["timeset", "tb"]],
                    values=_RAMP_COAL + _two_period_values())


def _patch_ramp_rolling(data: dict) -> dict:
    """coal ramp in the rolling fullYear dispatch with horizon 36 and jump
    24: every continuation roll ramps from the previous roll's flow."""
    return _overlay(data, name="ramp_rolling",
                    base=["init", "west", "battery", "wind", "coal", "fullYear", "dispatch_fullYear_roll"],
                    values=_RAMP_COAL + [
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 36.0],
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", 24.0]])


def _patch_ramp_chp_integer(data: dict) -> dict:
    """chp_floor_integer with ramp limits on coal_chp's fuel input and its
    electricity output: integer startup / shutdown capacity reduction on an
    output arc and on an indirect unit's input arc."""
    data = _chp_floor(data, "ramp_chp_integer", "binary")
    vals = (_ramp_on("unit__inputNode", ["coal_chp", "coal_market"], 0.002)
            + _ramp_on("unit__outputNode", ["coal_chp", "west"], 0.002))
    for cls, ent, param, val in vals:
        data["parameter_values"].append([cls, ent, param, _pack(val), "ramp_chp_integer"])
    return data


def _patch_ramp_rp(data: dict) -> dict:
    """coal ramp on a representative-period timeset whose two 24-step blocks
    (t0001, t0025) are adjacent on the timeline: no ramp row links them."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period",
                               "rank": 1, "data": [["t0001", 0.5], ["t0025", 0.5]]}]]}
    return _overlay(data, name="ramp_rp", base=["init", "west", "coal"],
                    entities=[["timeset", "rp2"]],
                    values=_RAMP_COAL + [
                        ["timeset", "rp2", "timeline", "y2020"],
                        ["timeset", "rp2", "timeset_duration", _pmap({"t0001": 24.0, "t0025": 24.0}, "time")],
                        ["timeset", "rp2", "representative_period_weights", rpw],
                        ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "rp2"})]])


def _patch_ramp_midhorizon(data: dict) -> dict:
    """Ramp on the invested unit of the mid-horizon stochastic fixture: the
    p2040 branches continue from p2035's last step."""
    return _overlay(data, name="ramp_midhorizon", base="midhorizon",
                    values=_ramp_on("unit__outputNode", ["base", "elec"], 0.002))


def _patch_ramp_stoch_rolling(data: dict) -> dict:
    """Ramp on coal_plant in the rolling stochastic 1-week fixture: every
    forecast branch of a continuation roll ramps from the realized flow."""
    return _overlay(data, name="ramp_stoch_rolling", base="1_week_rolling_wind",
                    values=_ramp_on("unit__outputNode", ["coal_plant", "demand_node"], 0.002))


def _patch_ramp_6h(data: dict) -> dict:
    """coal ramp with the solve's new_stepduration = 6 h (aggregated steps, adjacent in the solve's own
    step space)."""
    return _overlay(data, name="ramp_6h", base=["init", "west", "coal", "fullYear"],
                    values=_RAMP_COAL + [["solve", "y2020_fullYear_dispatch", "new_stepduration", 6.0]])


def _patch_ramp_blocks_2h(data: dict) -> dict:
    """coal ramp on the 5weeks timeset aggregated to 2-hour steps: gaps between blocks stay gaps."""
    return _overlay(data, name="ramp_blocks_2h", base=["init", "west", "coal", "coal_ramp", "5weeks"],
                    values=_RAMP_COAL + [["solve", "y2020_5week", "new_stepduration", 2.0]])


_BATTERY = ["init", "west", "wind", "battery"]


def _patch_agg_blocks_storage(data: dict) -> dict:
    """battery (bind_within_timeblock) on the 5weeks timeset aggregated to 2-hour steps: each block keeps its
    own state cycle and the gaps between blocks stay gaps."""
    return _overlay(data, name="agg_blocks_storage", base=_BATTERY + ["5weeks"],
                    values=[["solve", "y2020_5week", "new_stepduration", 2.0]])


def _patch_agg_intraperiod(data: dict) -> dict:
    """battery bind_intraperiod_blocks on the 5weeks timeset aggregated to 2-hour steps: four blocks per
    period (t0009 and t0013 touch)."""
    return _overlay(data, name="agg_intraperiod", base=_BATTERY + ["5weeks", "bind_battery_intraperiod_blocks"],
                    values=[["solve", "y2020_5week", "new_stepduration", 2.0]])


def _patch_agg_two_timesets(data: dict) -> dict:
    """Two periods on two timesets of one timeline (ta t0001-24, tb t0025-48) with the solve's
    new_stepduration = 2: both periods keep their steps."""
    return _overlay(data, name="agg_two_timesets", base=["init", "west", "coal"],
                    entities=[["timeset", "ta"], ["timeset", "tb"]],
                    values=_two_period_values() + [["solve", "y2020_2day_dispatch", "new_stepduration", 2.0]])


def _patch_agg_stoch_trunk(data: dict) -> dict:
    """Mid-horizon stochastic fixture (p2035 trunk, p2040 branches) with the solve's new_stepduration = 2:
    the trunk period keeps its steps."""
    return _overlay(data, name="agg_stoch_trunk", base="midhorizon",
                    values=[["solve", "stoch_2p_mid", "new_stepduration", 2.0]])


def _patch_agg_roll_blocks(data: dict) -> dict:
    """coal ramp, rolling dispatch (horizon 8 h, jump 4 h) over the 5weeks timeset aggregated to 2 h: only
    the roll that starts at t0013 (adjacent to the t0009 block) carries the previous roll's flow."""
    return _overlay(data, name="agg_roll_blocks", base=["init", "west", "coal", "fullYear", "dispatch_fullYear_roll"],
                    values=_RAMP_COAL + [
                        ["solve", "dispatch_fullYear_roll", "period_timeset", _pmap({"p2020": "5weeks"})],
                        ["solve", "dispatch_fullYear_roll", "realized_periods",
                         {"type": "array", "value_type": "str", "data": ["p2020"]}],
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 8.0],
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", 4.0],
                        ["solve", "dispatch_fullYear_roll", "new_stepduration", 2.0]])


def _agg_rp_values() -> list:
    """Representative-period timeset rpg on y2020: rep blocks t0001 and t0049 (24 h each, gap t0025-t0048),
    base periods t0001, t0025, t0049 blended from them."""
    def rep(w: float) -> dict:
        return {"type": "map", "index_type": "str", "index_name": "representative_period", "rank": 1,
                "data": [["t0001", w], ["t0049", 1.0 - w]]}
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", rep(1.0)], ["t0025", rep(0.5)], ["t0049", rep(0.0)]]}
    return [["timeset", "rpg", "timeline", "y2020"],
            ["timeset", "rpg", "timeset_duration", _pmap({"t0001": 24.0, "t0049": 24.0}, "time")],
            ["timeset", "rpg", "representative_period_weights", rpw],
            ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "rpg"})],
            ["node", "battery", "storage_binding_method", "bind_within_period_blended_weights"]]


def _patch_agg_rp(data: dict) -> dict:
    """battery bind_within_period_blended_weights on a gapped representative-period timeset with the solve's
    new_stepduration = 2: the rep blocks keep their first / last steps and the base periods their order."""
    return _overlay(data, name="agg_rp", base=_BATTERY + ["coal"], entities=[["timeset", "rpg"]],
                    values=_agg_rp_values() + [["solve", "y2020_2day_dispatch", "new_stepduration", 2.0]])


# ---- coarse new_stepduration groups (c15b: Q1 #11, #12, #33, #36, #8) ----

_ELZ_U = [["unit", "elz_u"], ["unit__inputNode", ["elz_u", "elec_A"]], ["unit__outputNode", ["elz_u", "h2_A"]],
          ["group__unit", ["daily_group", "elz_u"]]]


def _patch_coarse_explicit_unit(data: dict) -> dict:
    """lh2: unit elz_u (elec_A hourly -> h2_A daily) listed in daily_group: it still enters elec_A's hourly
    balance at every step (#36)."""
    return _overlay(data, name="coarse_explicit_unit", base="lh2_three_region", entities=_ELZ_U,
                    values=[["unit", "elz_u", "efficiency", 0.7], ["unit", "elz_u", "existing", 300.0],
                            ["unit", "elz_u", "conversion_method", "constant_efficiency"]])


def _patch_coarse_two_sink(data: dict) -> dict:
    """lh2: liquefier_A gets a second output lh2_B; both outputs are daily storage nodes and each node's coarse
    row holds only its own arcs (#8)."""
    return _overlay(data, name="coarse_two_sink", base="lh2_three_region",
                    entities=[["unit__outputNode", ["liquefier_A", "lh2_B"]]],
                    values=[["unit__outputNode", ["liquefier_A", "lh2_B"], "capacity_max_coeff", 0.5],
                            ["unit__outputNode", ["liquefier_A", "lh2_A"], "capacity_max_coeff", 0.5]])


def _coarse_dc_values(storage: bool) -> tuple[list, list]:
    four = _pmap({f"t000{i}": 1.0 for i in range(1, 5)}, "time")
    ents = [["timeline", "tl_dispatch"], ["group", "blk"], ["group__node", ["blk", "bus_14"]]]
    vals = [["timeline", "tl_dispatch", "timestep_duration", four],
            ["timeset", "ts_dispatch", "timeset_duration", _pmap({"t0001": 4.0}, "time")],
            ["node", "bus_14", "inflow", _tmap([-14.9, -40.0, -14.9, -40.0])],
            ["group", "blk", "new_stepduration", 2.0]]
    if storage:
        vals += [["node", "bus_14", "node_type", "storage"], ["node", "bus_14", "existing", 100.0],
                 ["node", "bus_14", "storage_state_start", 0.5]]
    return ents, vals


def _patch_coarse_dc_storage(data: dict) -> dict:
    """case14 DC power flow on 4 hourly steps, bus_14 a storage node in a 2-h group: its coarse rows hold
    v_flow and v_flow_back of both lines at every hour (#33)."""
    ents, vals = _coarse_dc_values(True)
    return _overlay(data, name="coarse_dc_storage", base="dc_opf_test", entities=ents, values=vals)


def _patch_coarse_dc_nostorage(data: dict) -> dict:
    """as coarse_dc_storage with bus_14 a balance node: a coarse group works without a storage node."""
    ents, vals = _coarse_dc_values(False)
    return _overlay(data, name="coarse_dc_nostorage", base="dc_opf_test", entities=ents, values=vals)


def _patch_coarse_uc(data: dict) -> dict:
    """lh2: liquefier_A (h2_A daily -> lh2_A daily) with linear online: online variables on every hour (#12)."""
    return _overlay(data, name="coarse_uc", base="lh2_three_region",
                    values=[["unit", "liquefier_A", "startup_method", "linear"],
                            ["unit", "liquefier_A", "min_load", 0.3],
                            ["unit", "liquefier_A", "virtual_unitsize", 50.0]])


_DAILY_BATTERY = [["group", "daily"], ["group__node", ["daily", "battery"]]]
_COARSE_BASE = ["init", "west", "wind", "coal", "battery"]


def _coarse_roll_values(jump: float) -> list:
    return [["group", "daily", "new_stepduration", 24.0],
            ["solve", "dispatch_fullYear_roll", "period_timeset", _pmap({"p2020": "fullYear"})],
            ["solve", "dispatch_fullYear_roll", "realized_periods",
             {"type": "array", "value_type": "str", "data": ["p2020"]}],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 48.0],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", jump]]


def _patch_coarse_roll(data: dict) -> dict:
    """battery in a 24-h group, rolling dispatch over fullYear (horizon 48 h, jump 24 h): every roll has
    coarse rows and carries the battery state."""
    return _overlay(data, name="coarse_roll", base=_COARSE_BASE + ["fullYear", "dispatch_fullYear_roll"],
                    entities=_DAILY_BATTERY, values=_coarse_roll_values(24.0))


def _patch_coarse_roll_jump12(data: dict) -> dict:
    """as coarse_roll with jump 12 h: a coarse step ends at each roll's last realized step."""
    return _overlay(data, name="coarse_roll_jump12", base=_COARSE_BASE + ["fullYear", "dispatch_fullYear_roll"],
                    entities=_DAILY_BATTERY, values=_coarse_roll_values(12.0))


def _patch_coarse_two_groups(data: dict) -> dict:
    """battery in a 24-h group and ev_battery in a 48-h group on the 2-day dispatch: each node has its own
    coarse steps."""
    return _overlay(data, name="coarse_two_groups", base=_COARSE_BASE + ["ev"],
                    entities=_DAILY_BATTERY + [["group", "twoday"], ["group__node", ["twoday", "ev_battery"]]],
                    values=[["group", "daily", "new_stepduration", 24.0],
                            ["group", "twoday", "new_stepduration", 48.0]])


def _patch_coarse_solve_step(data: dict) -> dict:
    """battery in a 24-h group with the solve's new_stepduration = 2: only the battery is coarse (coarse is
    relative to the solve's step)."""
    return _overlay(data, name="coarse_solve_step", base=_COARSE_BASE, entities=_DAILY_BATTERY,
                    values=[["group", "daily", "new_stepduration", 24.0],
                            ["solve", "y2020_2day_dispatch", "new_stepduration", 2.0]])


def _patch_coarse_not_coarse(data: dict) -> dict:
    """battery in a group whose step equals the solve's step (1 h): not coarse, LP as without the group."""
    return _overlay(data, name="coarse_not_coarse", base=_COARSE_BASE, entities=_DAILY_BATTERY,
                    values=[["group", "daily", "new_stepduration", 1.0]])


def _patch_coarse_nested_fix_price(data: dict) -> dict:
    """nested invest (24 h) -> storage (6 h) -> rolling dispatch with battery fix_price and the battery in a
    12-h group: not coarse in the 24-h invest solve, coarse in the storage and dispatch solves; the dispatch
    reads the storage solve's coarse price."""
    data = _nested(data, "coarse_nested_fix_price_base", "fix_price")
    return _overlay(data, name="coarse_nested_fix_price", base="coarse_nested_fix_price_base",
                    entities=[["group", "half_day"], ["group__node", ["half_day", "battery"]]],
                    values=[["group", "half_day", "new_stepduration", 12.0]])


def _patch_bind_within_period_2p(data: dict) -> dict:
    """battery bind_within_period on two periods sharing the 5weeks timeset (five blocks per period, fix_start,
    use_reference_value): each period cycles on itself, blocks chain inside the period (item 124)."""
    return _overlay(data, name="bind_within_period_2p", base="wind_battery_invest",
                    values=[["node", "battery", "storage_binding_method", "bind_within_period"]])


def _patch_rp_downgrade_within_solve(data: dict) -> dict:
    """lh2_three_region_rp_invest on the plain week168 timeset (no representative periods): the lh2 nodes'
    bind_within_solve_blended_weights is downgraded to bind_within_solve next to the plain bind_within_solve
    batteries; the downgraded nodes keep their state chain (item 127)."""
    sfx = "__lh2_three_region_rp_invest"
    return _overlay(data, name="rp_downgrade_within_solve", base="lh2_three_region_rp_invest", values=[
        ["solve", "lh2_rp_invest" + sfx, "period_timeset",
         {"type": "map", "index_type": "str", "index_name": "period", "rank": 1,
          "data": [["y2030", "week168"], ["y2040", "week168"]]}]]
        + [["node", n, "storage_binding_method", "bind_within_solve_blended_weights"]
           for n in ("lh2_A", "lh2_B", "lh2_C")])


def _patch_nested_rolling_storage_uc(data: dict) -> dict:
    """Nested invest -> rolling storage (48 h / 24 h, 6-h steps) -> rolling dispatch (24 h / 12 h) with the battery
    bind_forward_only + fix_start (no nested fix) and coal unit commitment (linear, min_load 0.4, min_uptime 8 h, ramp
    limit 0.002): the first dispatch roll under each later storage roll continues the storage level, the commitment,
    the ramp and the min up time history from the dispatch realised before it (carry routing)."""
    return _overlay(data, name="nested_rolling_storage_uc",
                    base="multi_fullYear_battery_nested_24h_invest_one_solve",
                    values=_RAMP_COAL + [
                        ["solve", "storage_fullYear_6h", "solve_mode", "rolling_window"],
                        ["solve", "storage_fullYear_6h", "rolling_solve_horizon", 48.0],
                        ["solve", "storage_fullYear_6h", "rolling_solve_jump", 24.0],
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 24.0],
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", 12.0],
                        ["node", "battery", "storage_nested_fix_method", "fix_nothing"],
                        ["unit", "coal_plant", "startup_method", "linear"],
                        ["unit", "coal_plant", "min_load", 0.4],
                        ["unit", "coal_plant", "min_uptime", 8.0]])


def _patch_storage_sequence_start_end(data: dict) -> dict:
    """The 4-solve investment sequence (y2020 -> y2035) with the battery bind_forward_only + fix_start_end (start 0.1,
    end 0.3): y2020 starts from fix_start, y2025-y2035 continue from the realised state, and the solves whose horizon
    holds the model's last step (y2030: p2030 + p2035, and y2035) have storage_state_end_binding at p2035's end."""
    return _overlay(data, name="storage_sequence_start_end", base="wind_battery_invest_lifetime_renew_4solve",
                    values=[["node", "battery", "storage_binding_method", "bind_forward_only"],
                            ["node", "battery", "storage_start_end_method", "fix_start_end"],
                            ["node", "battery", "storage_state_end", 0.3]])


def _patch_nested_fix_quantity_reference(data: dict) -> dict:
    """nested_fix_quantity with the battery use_reference_value 0.4: the dispatch rolls' ends are fixed by the
    storage solve, so the reference value yields there; the invest and storage solves keep it."""
    data = _nested(data, "nested_fix_quantity_reference_base", "fix_quantity")
    return _overlay(data, name="nested_fix_quantity_reference", base="nested_fix_quantity_reference_base",
                    values=[["node", "battery", "storage_solve_horizon_method", "use_reference_value"]])


def _patch_storage_fix_start_end_blocks(data: dict) -> dict:
    """5weeks_battery_intraperiod_blocks with the battery fix_start_end (start 0.1, end 0.3): the binding method's loop
    is ignored, so the blocks chain forward from p2020 into p2025, the start is in the first block's balance and the
    end pin is on the last block of each period_last period."""
    return _overlay(data, name="storage_fix_start_end_blocks", base="5weeks_battery_intraperiod_blocks",
                    values=[["node", "battery", "storage_start_end_method", "fix_start_end"],
                            ["node", "battery", "storage_state_end", 0.3]])


def _patch_storage_fix_start_end_within_solve(data: dict) -> dict:
    """wind_battery_invest_lifetime_renew_4solve with the battery bind_within_solve + fix_start_end (start 0.1, end
    0.3): the solve ignores the cyclic method (bind_forward_only), so y2020 starts from fix_start, the later solves
    continue from the realised state and y2030 / y2035 (whose horizons hold the model's last step) have the end pin."""
    return _overlay(data, name="storage_fix_start_end_within_solve", base="wind_battery_invest_lifetime_renew_4solve",
                    values=[["node", "battery", "storage_binding_method", "bind_within_solve"],
                            ["node", "battery", "storage_start_end_method", "fix_start_end"],
                            ["node", "battery", "storage_state_end", 0.3]])


def _patch_rolling_intraperiod_blocks(data: dict) -> dict:
    """battery bind_intraperiod_blocks in the rolling fullYear dispatch over p2020 + p2025 (72 h each, horizon 96 h,
    jump 48 h): roll 0 holds p2020 whole (loops, cut at its realized end t0048) and chains into p2025; roll 1 starts
    from the carried level inside p2020 and holds p2025 whole (loops); roll 2 chains p2025 from the carry."""
    return _overlay(data, name="rolling_intraperiod_blocks",
                    base=["init", "west", "wind", "coal", "battery", "fullYear", "dispatch_fullYear_roll",
                          "bind_battery_intraperiod_blocks"],
                    values=[["solve", "dispatch_fullYear_roll", "period_timeset",
                             _pmap({"p2020": "fullYear", "p2025": "fullYear"})],
                            ["solve", "dispatch_fullYear_roll", "realized_periods",
                             {"type": "array", "value_type": "str", "data": ["p2020", "p2025"]}],
                            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 96.0],
                            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", 48.0]])


def _patch_coarse_stoch(data: dict) -> dict:
    """stochastic 2-day dispatch with hydro_reservoir in a 24-h group: coarse rows in every period incl.
    the forecast branches."""
    return _overlay(data, name="coarse_stoch", base="2_day_stochastic_dispatch",
                    entities=[["group", "daily"], ["group__node", ["daily", "hydro_reservoir"]]],
                    values=[["group", "daily", "new_stepduration", 24.0]])


_MLE_CHP = [["unit", "coal_chp", "startup_method", "linear"],
            ["unit", "coal_chp", "min_load", 0.3],
            ["unit", "coal_chp", "conversion_method", "min_load_efficiency"],
            ["unit", "coal_chp", "efficiency_at_min_load", 0.6],
            ["unit", "coal_chp", "availability", 0.8],
            ["constraint", "coal_chp_fix", "is_enabled", "no"]]


def _patch_ramp_avail_section(data: dict) -> dict:
    """coal_chp (linear online, min_load_efficiency, availability 0.8, demand above capacity) with a ramp on
    its fuel input: the input-arc capacity rows must not cap the derated full output."""
    return _overlay(data, name="ramp_avail_section", base=["init", "west", "coal_chp", "heat"],
                    values=_MLE_CHP + _ramp_on("unit__inputNode", ["coal_chp", "coal_market"], 0.005) + [
                        ["node", "heat", "inflow", 0.0],
                        ["node", "west", "inflow", _tmap([-3000.0] * 48)]])


def _patch_reserve_input_avail(data: dict) -> dict:
    """reserve_input_side with coal_chp min_load_efficiency and availability 0.8: the "consume more" row of
    the fuel arc (maxInputFlow_online) at derated full output."""
    data = _patch_reserve_input_side(data)
    for cls, ent, param, val in _MLE_CHP[2:5]:
        data["parameter_values"].append([cls, ent, param, _pack(val), "reserve_input_side"])
    data["scenarios"].append(["reserve_input_avail", False, ""])
    full = _chain(data, "reserve_input_side")
    for a, before in zip(full, full[1:] + [None]):
        data["scenario_alternatives"].append(["reserve_input_avail", a, before])
    return data


def _patch_ramp_roll_span(data: dict) -> dict:
    """Two continuing periods p2020 (t0001-24) and a2025 (t0025-48, a name that sorts before p2020), coal
    linear online with a ramp, rolling horizon 16 / jump 10: one roll realizes steps of both periods; the
    carriers must take the last realized step in time."""
    vals = [v for v in _two_period_values() if v[2] not in ("period_timeset", "realized_periods")] + [
        ["solve", "y2020_2day_dispatch", "period_timeset", _pmap({"p2020": "ta", "a2025": "tb"})],
        ["solve", "y2020_2day_dispatch", "realized_periods",
         {"type": "array", "value_type": "str", "data": ["p2020", "a2025"]}],
        ["unit", "coal_plant", "startup_method", "linear"],
        ["unit", "coal_plant", "min_load", 0.2],
        ["solve", "y2020_2day_dispatch", "solve_mode", "rolling_window"],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", 16.0],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump", 10.0]]
    return _overlay(data, name="ramp_roll_span", base=["init", "west", "coal"],
                    entities=[["timeset", "ta"], ["timeset", "tb"]], values=_RAMP_COAL + vals)


_UNI = ["init", "west", "coal", "wind", "network", "unidirectional"]
_NET = ["init", "west", "coal", "wind", "network"]
_LL = ["Base", "lossless_2way"]


def _conn_startup(conn: str, method: str, cost: float = 3.0) -> list:
    return [["connection", conn, "startup_method", method], ["connection", conn, "startup_cost", cost]]


def _patch_conn_online_uni(data: dict) -> dict:
    """One-way west_east with linear online variables next to an online unit (coal_plant linear)."""
    return _overlay(data, name="conn_online_uni", base=_UNI + ["coal_min_load"],
                    values=_conn_startup("west_east", "linear"))


def _patch_conn_online_regular(data: dict) -> dict:
    """Two-way (regular, two variables) west_east with linear online variables."""
    return _overlay(data, name="conn_online_regular", base=_NET, values=_conn_startup("west_east", "linear"))


def _patch_conn_online_regular_mip(data: dict) -> dict:
    """Two-way (regular) west_east with binary online variables."""
    return _overlay(data, name="conn_online_regular_mip", base=_NET,
                    values=_conn_startup("west_east", "binary"))


def _patch_conn_online_1var(data: dict) -> dict:
    """One-variable (no_losses_no_variable_cost) west_east, linear online, holding up and down reserve at
    west."""
    eu, vu = _reserve("ru", "up", "electricity", [("connection", "west_east", "west")], reservation=100.0)
    ed, vd = _reserve("rd", "down", "electricity", [("connection", "west_east", "west")], reservation=100.0)
    return _overlay(data, name="conn_online_1var", base=_NET, entities=eu + ed,
                    values=vu + vd + _conn_startup("west_east", "linear") + [
                        ["connection", "west_east", "transfer_method", "no_losses_no_variable_cost"]])


def _patch_conn_exact(data: dict) -> dict:
    """exact ll (efficiency 0.9) with a 50 MW surplus at lw and costly spill: it must not dissipate the
    surplus by flowing both ways at once."""
    return _overlay(data, name="conn_exact", base=_LL, values=[
        ["connection", "ll", "transfer_method", "exact"], ["connection", "ll", "efficiency", 0.9],
        ["node", "lw", "inflow", 50.0], ["node", "le", "inflow", 0.0]])


def _patch_conn_exact_flow(data: dict) -> dict:
    """exact west_east in the plain network: the normal case."""
    return _overlay(data, name="conn_exact_flow", base=_NET,
                    values=[["connection", "west_east", "transfer_method", "exact"]])


def _patch_conn_exact_invest(data: dict) -> dict:
    """exact ll without existing capacity, investment allowed: the invested capacity flows both ways (one
    direction per step)."""
    return _overlay(data, name="conn_exact_invest", base=_LL, values=[
        ["connection", "ll", "transfer_method", "exact"], ["connection", "ll", "efficiency", 0.95],
        ["connection", "ll", "existing", 0.0], ["connection", "ll", "invest_method", "invest_no_limit"],
        ["connection", "ll", "invest_cost", 1.0], ["connection", "ll", "lifetime", 20.0],
        ["connection", "ll", "discount_rate", 0.05],
        ["node", "lw", "penalty_up", 1000.0], ["node", "lw", "penalty_down", 1000.0],
        ["node", "le", "penalty_up", 1000.0], ["node", "le", "penalty_down", 1000.0],
        ["solve", "lossless_2way_solve", "invest_periods",
         {"type": "array", "value_type": "str", "data": ["p2020"]}],
        ["solve", "lossless_2way_solve", "realized_invest_periods",
         {"type": "array", "value_type": "str", "data": ["p2020"]}]])


_GW = [["group", "gw"], ["group__node", ["gw", "lw"]]]


def _ll_reserve(r: str, ud: str, method: str, extra: list, reservation: float = 0.0) -> tuple[list, list]:
    ents, vals = _reserve(r, ud, "gw", [("connection", "ll", "lw")], method=method, reservation=reservation,
                          penalty=1000.0)
    rel = [r, ud, "ll", "lw"]
    vals += [["reserve__upDown__connection__node", rel, k, v] for k, v in extra]
    return ents, vals


def _patch_conn_reserve_1var(data: dict) -> dict:
    """One-variable ll (lossless_2way) at lw: dynamic up (ratio 0.5), n-1 up (ll fails), timeseries up and
    down 150 MW; ll exports 100 MW from lw at t0001 and imports 100 MW at t0002 (v_flow_back)."""
    e1, v1 = _ll_reserve("dyu", "up", "dynamic_only", [("increase_reserve_ratio", 0.5)])
    e2, v2 = _ll_reserve("n1u", "up", "large_failure_only", [("large_failure_ratio", 1.0)])
    e3, v3 = _ll_reserve("tsu", "up", "timeseries_only", [], reservation=150.0)
    e4, v4 = _ll_reserve("tsd", "down", "timeseries_only", [], reservation=150.0)
    return _overlay(data, name="conn_reserve_1var", base=_LL, values=v1 + v2 + v3 + v4,
                    entities=_GW + e1 + e2 + e3 + e4)


def _non_sync_group(g: str, nodes: list[str], limit: float, penalty: float = 4000.0) -> tuple[list, list]:
    """Entities + values of node group ``g`` with a non-synchronous limit."""
    ents = [["group", g]] + [["group__node", [g, n]] for n in nodes]
    vals = [["group", g, "has_non_synchronous", "yes"], ["group", g, "non_synchronous_limit", limit],
            ["group", g, "penalty_non_synchronous", penalty]]
    return ents, vals


def _patch_nonsync_internal_dc(data: dict) -> dict:
    """Non-synchronous limit 0.3 on {west, east} (wind flagged): west_east runs inside the group and
    west_north (is_DC) crosses its boundary."""
    ents, vals = _non_sync_group("gwe", ["west", "east"], 0.3)
    return _overlay(data, name="nonsync_internal_dc", base=_NET, entities=ents, values=vals + [
        ["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", "yes"],
        ["connection", "west_north", "is_DC", "yes"]])


def _patch_nonsync_1var_dc(data: dict) -> dict:
    """Non-synchronous limit 0.5 on {le}; ll (one variable, is_DC) imports to le through v_flow and exports
    through v_flow_back."""
    ents, vals = _non_sync_group("gle", ["le"], 0.5)
    return _overlay(data, name="nonsync_1var_dc", base=_LL, entities=ents,
                    values=vals + [["connection", "ll", "is_DC", "yes"]])


def _patch_nonsync_energy_margin(data: dict) -> dict:
    """Invest solve with an energy margin (inflow multiplier 1.1 on west) and a group {west} with a
    non-synchronous limit and a manual capacity margin."""
    ents = [["group", "gm"], ["group__node", ["gm", "west"]]]
    return _overlay(data, name="nonsync_energy_margin",
                    base=["init", "west", "wind", "battery", "battery_invest", "5weeks", "coal"],
                    entities=ents, values=[
                        ["group", "gm", "has_non_synchronous", "yes"], ["group", "gm", "non_synchronous_limit", 0.5],
                        ["group", "gm", "capacity_margin_method", "manual"], ["group", "gm", "capacity_margin", 0.0],
                        ["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", "yes"],
                        ["node", "west", "energy_margin_method", "inflow_multiplier"],
                        ["node", "west", "energy_margin_multiplier", 1.1]])


def _patch_conn_reserve_uni(data: dict) -> dict:
    """One-way west_east (west -> east) holding down reserve (send more) and up reserve (send less) at
    west."""
    ed, vd = _reserve("rdw", "down", "gww", [("connection", "west_east", "west")], reservation=300.0)
    eu, vu = _reserve("ruw", "up", "gww", [("connection", "west_east", "west")], reservation=300.0)
    return _overlay(data, name="conn_reserve_uni", base=_UNI, values=vd + vu,
                    entities=[["group", "gww"], ["group__node", ["gww", "west"]]] + ed + eu)


def _patch_conn_reserve_dc(data: dict) -> dict:
    """Upward and downward reserve at bus_10 from the two DC power-flow lines there (one-variable arcs with
    v_flow_back)."""
    provs = [("connection", "line_10_11", "bus_10"), ("connection", "line_9_10", "bus_10")]
    eu, vu = _reserve("ru", "up", "g10", provs, reservation=20.0)
    ed, vd = _reserve("rd", "down", "g10", provs, reservation=20.0)
    return _overlay(data, name="conn_reserve_dc", base=["base"], values=vu + vd,
                    entities=[["group", "g10"], ["group__node", ["g10", "bus_10"]]] + eu + ed)


def _patch_conn_online_roll(data: dict) -> dict:
    """One-way west_east with linear online variables in the rolling fullYear dispatch (horizon 36, jump
    24): every continuation roll carries the connection's online state."""
    return _overlay(data, name="conn_online_roll",
                    base=["init", "west", "battery", "wind", "coal", "network", "unidirectional", "fullYear",
                          "dispatch_fullYear_roll"],
                    values=_conn_startup("west_east", "linear") + [
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", 36.0],
                        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", 24.0]])


def _patch_conn_online_stoch(data: dict) -> dict:
    """Stochastic 2-day dispatch with node north fed from demand_node by a one-way connection (linear
    online) and a regular one (binary online): online ties across the forecast branches."""
    ents = [["node", "north"], ["connection", "dn_north"], ["connection", "dn_north2"],
            ["connection__node__node", ["dn_north", "demand_node", "north"]],
            ["connection__node__node", ["dn_north2", "demand_node", "north"]]]
    vals = [["node", "north", "node_type", "balance"], ["node", "north", "inflow", -150.0],
            ["node", "north", "penalty_up", 5000.0], ["node", "north", "penalty_down", 5000.0],
            ["connection", "dn_north", "transfer_method", "unidirectional"],
            ["connection", "dn_north", "existing", 100.0], ["connection", "dn_north", "efficiency", 0.95],
            ["connection", "dn_north2", "transfer_method", "regular"],
            ["connection", "dn_north2", "existing", 100.0], ["connection", "dn_north2", "efficiency", 0.9]]
    vals += _conn_startup("dn_north", "linear") + _conn_startup("dn_north2", "binary", 5.0)
    return _overlay(data, name="conn_online_stoch", base="2_day_stochastic_dispatch", entities=ents,
                    values=vals)


def _patch_conn_reserve_exact(data: dict) -> dict:
    """exact ll (forward lw -> le) with up and down reserve at lw: the direction-switch reserve rows."""
    e1, v1 = _ll_reserve("ru", "up", "timeseries_only", [], reservation=50.0)
    e2, v2 = _ll_reserve("rd", "down", "timeseries_only", [], reservation=50.0)
    return _overlay(data, name="conn_reserve_exact", base=_LL, values=v1 + v2 + [
        ["connection", "ll", "transfer_method", "exact"]], entities=_GW + e1 + e2)


def _patch_conn_online_1var_mip(data: dict) -> dict:
    """One-variable west_east with binary online variables and up / down reserve at west."""
    eu, vu = _reserve("ru", "up", "electricity", [("connection", "west_east", "west")], reservation=100.0)
    ed, vd = _reserve("rd", "down", "electricity", [("connection", "west_east", "west")], reservation=100.0)
    return _overlay(data, name="conn_online_1var_mip", base=_NET, entities=eu + ed,
                    values=vu + vd + _conn_startup("west_east", "binary") + [
                        ["connection", "west_east", "transfer_method", "no_losses_no_variable_cost"]])


def _group_override(data: dict, name: str, method: str) -> dict:
    return _overlay(data, name=name, base=_NET, values=[["group", "gwe", "transfer_method", method]],
                    entities=[["group", "gwe"], ["group__node", ["gwe", "west"]],
                              ["group__node", ["gwe", "east"]]])


def _patch_conn_group_override_uni(data: dict) -> dict:
    """group transfer_method = unidirectional over west, east (west_east regular): one arc."""
    return _group_override(data, "conn_group_override_uni", "unidirectional")


def _patch_conn_group_override_exact(data: dict) -> dict:
    """group transfer_method = exact over west, east: west_east gets the direction switch."""
    return _group_override(data, "conn_group_override_exact", "exact")


def _patch_online_bin_lin(data: dict) -> dict:
    """coal_plant binary online next to coal_chp linear online, with an inertia limit: both units online,
    one inertia term each."""
    return _overlay(data, name="online_bin_lin",
                    base=["init", "west", "coal", "heat", "coal_chp", "coal_min_load", "coal_min_load_MIP",
                          "inertia"],
                    values=[["unit", "coal_chp", "startup_method", "linear"]])


def _patch_online_mintime_lin(data: dict) -> dict:
    """coal_plant online only through minimum_time_method next to coal_chp linear online: coal_plant keeps
    its online variables and min up / down rows."""
    return _overlay(data, name="online_mintime_lin",
                    base=["init", "west", "coal", "heat", "coal_chp", "coal_min_updowntime"],
                    values=[["unit", "coal_chp", "startup_method", "linear"]])


# Silent-label inputs: a Map with Toolbox's default index name ``x`` whose keys are periods of other solves only
# (p2030 on a p2020 solve).  Each fixture has an unambiguous twin in tests/engine/test_silent_label_params.py
# (items 167-169 of specs/github_issues.md).

def _patch_pen_silent(data: dict) -> dict:
    """west penalty_up / penalty_down = {p2030: 5000} with index ``x``: no key in the solve, so the default 10000
    applies (item 167)."""
    m = _pmap({"p2030": 5000.0}, index_name="x")
    return _overlay(data, name="pen_silent", base="base",
                    values=[["node", "west", "penalty_up", m], ["node", "west", "penalty_down", m]])


def _patch_avail_silent(data: dict) -> dict:
    """coal_plant availability = {p2030: 0.5} with index ``x`` next to wind_plant availability 0.5: coal keeps the
    default and wind its scalar (item 168)."""
    return _overlay(data, name="avail_silent", base="coal_wind_ev", values=[
        ["unit", "coal_plant", "availability", _pmap({"p2030": 0.5}, index_name="x")],
        ["unit", "wind_plant", "availability", 0.5]])


def _patch_varcost_silent(data: dict) -> dict:
    """(coal_plant, west) other_operational_cost = {p2030: 7} with index ``x`` next to (wind_plant, west) 3.0: coal
    keeps no cost and wind its scalar (item 168)."""
    return _overlay(data, name="varcost_silent", base="coal_wind_ev", values=[
        ["unit__outputNode", ["wind_plant", "west"], "other_operational_cost", 3.0],
        ["unit__outputNode", ["coal_plant", "west"], "other_operational_cost",
         _pmap({"p2030": 7.0}, index_name="x")]])


def _patch_ladder_ann_depth2(data: dict) -> dict:
    """coal price_ladder_annual in the depth-2 form tier -> {price, quantity} (the same ladder in every period) in
    place of coal_ladder_annual's depth-3 period -> tier -> facet value with the same tiers (item 169)."""
    def tier(price: float, quantity: float) -> dict:
        return {"type": "map", "index_type": "str", "rank": 1, "index_name": "x",
                "data": [["price", price], ["quantity", quantity]]}

    ladder = {"index_type": "str", "rank": 2, "index_name": "tier",
              "data": [["1", tier(20.0, 1.0)], ["2", tier(30.0, 1e30)]]}
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == "price_ladder_annual" and v[4] == "ladder_ann_on")]
    return _overlay(data, name="ladder_ann_depth2", base="coal_ladder_annual",
                    values=[["commodity", "coal", "price_ladder_annual", ladder]])


def _patch_sd_silent(data: dict) -> dict:
    """coal_wind_ev + west self_discharge_loss = {p2030: 0.1} with index ``x``: west has no storage, so this
    value must change nothing, and ev_battery keeps its own self_discharge_loss (item 168).  Mirrors the
    ``sd_silent`` case of tests/engine/test_silent_label_params.py."""
    return _overlay(data, name="sd_silent", base="coal_wind_ev", values=[
        ["node", "west", "self_discharge_loss", _pmap({"p2030": 0.1}, index_name="x")]])


def _patch_timeline_default(data: dict) -> dict:
    """base with the 2day timeset's timeset_duration (alternative ``init``) removed: the scenario then has one
    timeline (y2020) and no ``timeset_duration`` anywhere, so the single-timeline default (item 171) must apply
    for the DB-side timeline readers to give the same ``dt`` as today's CSV-seeded path."""
    data = dict(data)
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "timeset" and v[2] == "timeset_duration" and v[4] == "init")]
    return _overlay(data, name="timeline_default", base="base", values=[])


def _patch_ladder_cum_silent(data: dict) -> dict:
    """coal price_ladder_cumulative with index_name ``x`` on both levels in place of coal_ladder_cumulative's
    labelled ``tier`` level (item 169)."""
    def tier(price: float, quantity: float) -> dict:
        return {"type": "map", "index_type": "str", "rank": 1, "index_name": "x",
                "data": [["price", price], ["quantity", quantity]]}

    ladder = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", tier(20.0, 1.0)], ["2", tier(30.0, 1e30)]]}
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == "price_ladder_cumulative"
                and v[4] == "ladder_cum_on")]
    return _overlay(data, name="ladder_cum_silent", base="coal_ladder_cumulative",
                    values=[["commodity", "coal", "price_ladder_cumulative", ladder]])


PATCHES: dict[str, Callable[[dict], dict]] = {
    "none": _no_patch,
    "invest_limits": _patch_invest_limits,
    "cumulative_capacity": _patch_cumulative_capacity,
    "stoch_invest_storage": _patch_stoch_invest_storage,
    "reserve_floors": _patch_reserve_floors,
    "chp_floor_linear": _patch_chp_floor_linear,
    "chp_floor_integer": _patch_chp_floor_integer,
    "reserve_dynamic_n1_down": _patch_reserve_dynamic_n1_down,
    "reserve_up_online": _patch_reserve_up_online,
    "reserve_storage_duration": _patch_reserve_storage_duration,
    "reserve_input_side": _patch_reserve_input_side,
    "reserve_dynamic_demand": _patch_reserve_dynamic_demand,
    "stoch_na_reserve": _patch_stoch_na_reserve,
    "chp_misc": _patch_chp_misc,
    "profiles": _patch_profiles,
    "co2_period_total": _patch_co2_period_total,
    "co2_cap_minload": _patch_co2_cap_minload,
    "co2_period_rolling": _patch_co2_period_rolling,
    "negcap_indirect": _patch_negcap_indirect,
    "nested_fix_quantity": _patch_nested_fix_quantity,
    "nested_fix_usage": _patch_nested_fix_usage,
    "nested_fix_price": _patch_nested_fix_price,
    "rp_multi_base_period": _patch_rp_multi_base_period,
    "rp_multi_base_solve": _patch_rp_multi_base_solve,
    "ramp_step_duration": _patch_ramp_step_duration,
    "ramp_blocks": _patch_ramp_blocks,
    "ramp_reserve_online": _patch_ramp_reserve_online,
    "ramp_two_periods": _patch_ramp_two_periods,
    "ramp_rolling": _patch_ramp_rolling,
    "ramp_chp_integer": _patch_ramp_chp_integer,
    "ramp_rp": _patch_ramp_rp,
    "ramp_midhorizon": _patch_ramp_midhorizon,
    "ramp_stoch_rolling": _patch_ramp_stoch_rolling,
    "ramp_6h": _patch_ramp_6h,
    "ramp_blocks_2h": _patch_ramp_blocks_2h,
    "ramp_avail_section": _patch_ramp_avail_section,
    "reserve_input_avail": _patch_reserve_input_avail,
    "ramp_roll_span": _patch_ramp_roll_span,
    "conn_online_uni": _patch_conn_online_uni,
    "conn_online_regular": _patch_conn_online_regular,
    "conn_online_regular_mip": _patch_conn_online_regular_mip,
    "conn_online_1var": _patch_conn_online_1var,
    "conn_exact": _patch_conn_exact,
    "conn_exact_flow": _patch_conn_exact_flow,
    "conn_exact_invest": _patch_conn_exact_invest,
    "conn_reserve_1var": _patch_conn_reserve_1var,
    "conn_reserve_uni": _patch_conn_reserve_uni,
    "nonsync_internal_dc": _patch_nonsync_internal_dc,
    "nonsync_1var_dc": _patch_nonsync_1var_dc,
    "nonsync_energy_margin": _patch_nonsync_energy_margin,
    "conn_reserve_dc": _patch_conn_reserve_dc,
    "conn_online_roll": _patch_conn_online_roll,
    "conn_online_stoch": _patch_conn_online_stoch,
    "conn_group_override_uni": _patch_conn_group_override_uni,
    "conn_group_override_exact": _patch_conn_group_override_exact,
    "online_bin_lin": _patch_online_bin_lin,
    "online_mintime_lin": _patch_online_mintime_lin,
    "conn_reserve_exact": _patch_conn_reserve_exact,
    "conn_online_1var_mip": _patch_conn_online_1var_mip,
    "agg_blocks_storage": _patch_agg_blocks_storage,
    "agg_intraperiod": _patch_agg_intraperiod,
    "agg_two_timesets": _patch_agg_two_timesets,
    "agg_stoch_trunk": _patch_agg_stoch_trunk,
    "agg_roll_blocks": _patch_agg_roll_blocks,
    "agg_rp": _patch_agg_rp,
    "coarse_explicit_unit": _patch_coarse_explicit_unit,
    "coarse_two_sink": _patch_coarse_two_sink,
    "coarse_dc_storage": _patch_coarse_dc_storage,
    "coarse_dc_nostorage": _patch_coarse_dc_nostorage,
    "coarse_uc": _patch_coarse_uc,
    "coarse_roll": _patch_coarse_roll,
    "coarse_roll_jump12": _patch_coarse_roll_jump12,
    "coarse_two_groups": _patch_coarse_two_groups,
    "coarse_solve_step": _patch_coarse_solve_step,
    "coarse_not_coarse": _patch_coarse_not_coarse,
    "coarse_nested_fix_price": _patch_coarse_nested_fix_price,
    "coarse_stoch": _patch_coarse_stoch,
    "bind_within_period_2p": _patch_bind_within_period_2p,
    "rp_downgrade_within_solve": _patch_rp_downgrade_within_solve,
    "rolling_intraperiod_blocks": _patch_rolling_intraperiod_blocks,
    "nested_rolling_storage_uc": _patch_nested_rolling_storage_uc,
    "storage_sequence_start_end": _patch_storage_sequence_start_end,
    "nested_fix_quantity_reference": _patch_nested_fix_quantity_reference,
    "storage_fix_start_end_blocks": _patch_storage_fix_start_end_blocks,
    "storage_fix_start_end_within_solve": _patch_storage_fix_start_end_within_solve,
    "pen_silent": _patch_pen_silent,
    "avail_silent": _patch_avail_silent,
    "varcost_silent": _patch_varcost_silent,
    "ladder_ann_depth2": _patch_ladder_ann_depth2,
    "sd_silent": _patch_sd_silent,
    "timeline_default": _patch_timeline_default,
    "ladder_cum_silent": _patch_ladder_cum_silent,
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
    ("nested_multi_invest", _T, "multi_fullYear_battery_nested_multi_invest"),
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
    ("reserve_up_online", _T),
    ("reserve_storage_duration", _T),
    ("reserve_input_side", _T),
    ("reserve_dynamic_demand", _T),
    ("stoch_na_reserve", _F + "stochastics.json"),
    ("chp_misc", _T),
    ("profiles", _T),
    ("co2_period_total", _T),
    ("co2_cap_minload", _T),
    ("co2_period_rolling", _T),
    ("negcap_indirect", _T),
    ("nested_fix_quantity", _T),
    ("nested_fix_usage", _T),
    ("nested_fix_price", _T),
    ("rp_multi_base_period", _F + "lh2_three_region_rp_invest.json"),
    ("rp_multi_base_solve", _F + "lh2_three_region_rp_invest.json"),
    ("ramp_step_duration", _T),
    ("ramp_blocks", _T),
    ("ramp_reserve_online", _T),
    ("ramp_two_periods", _T),
    ("ramp_rolling", _T),
    ("ramp_chp_integer", _T),
    ("ramp_rp", _T),
    ("ramp_midhorizon", _F + "stoch_two_period_midhorizon_nreal0.json"),
    ("ramp_stoch_rolling", _F + "stochastics.json"),
    ("ramp_6h", _T),
    ("ramp_blocks_2h", _T),
    ("ramp_avail_section", _T),
    ("reserve_input_avail", _T),
    ("ramp_roll_span", _T),
    ("conn_online_uni", _T),
    ("conn_online_regular", _T),
    ("conn_online_regular_mip", _T),
    ("conn_online_1var", _T),
    ("conn_exact", _T),
    ("conn_exact_flow", _T),
    ("conn_exact_invest", _T),
    ("conn_reserve_1var", _T),
    ("conn_reserve_uni", _T),
    ("nonsync_internal_dc", _T),
    ("nonsync_1var_dc", _T),
    ("nonsync_energy_margin", _T),
    ("conn_reserve_dc", _F + "case14_dc_power_flow.json"),
    ("conn_online_roll", _T),
    ("conn_online_stoch", _F + "stochastics.json"),
    ("conn_group_override_uni", _T),
    ("conn_group_override_exact", _T),
    ("online_bin_lin", _T),
    ("online_mintime_lin", _T),
    ("conn_reserve_exact", _T),
    ("conn_online_1var_mip", _T),
    ("agg_blocks_storage", _T),
    ("agg_intraperiod", _T),
    ("agg_two_timesets", _T),
    ("agg_stoch_trunk", _F + "stoch_two_period_midhorizon_nreal0.json"),
    ("agg_roll_blocks", _T),
    ("agg_rp", _T),
    ("coarse_explicit_unit", _F + "lh2_three_region.json"),
    ("coarse_two_sink", _F + "lh2_three_region.json"),
    ("coarse_dc_storage", _F + "case14_dc_power_flow.json"),
    ("coarse_dc_nostorage", _F + "case14_dc_power_flow.json"),
    ("coarse_uc", _F + "lh2_three_region.json"),
    ("coarse_roll", _T),
    ("coarse_roll_jump12", _T),
    ("coarse_two_groups", _T),
    ("coarse_solve_step", _T),
    ("coarse_not_coarse", _T),
    ("coarse_nested_fix_price", _T),
    ("coarse_stoch", _F + "stochastics.json"),
    ("bind_within_period_2p", _T),
    ("rp_downgrade_within_solve", _F + "lh2_three_region_rp_invest.json"),
    ("rolling_intraperiod_blocks", _T),
    ("nested_rolling_storage_uc", _T),
    ("storage_sequence_start_end", _T),
    ("nested_fix_quantity_reference", _T),
    ("storage_fix_start_end_blocks", _T),
    ("storage_fix_start_end_within_solve", _T),
    ("pen_silent", _T),
    ("avail_silent", _T),
    ("varcost_silent", _T),
    ("ladder_ann_depth2", _T),
    ("sd_silent", _T),
    ("timeline_default", _T),
    ("ladder_cum_silent", _T),
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


def _build_fixture_db(fx: Fixture, tmp_path: Path, stem: str) -> tuple[str, Path]:
    """Build ``fx``'s Spine DB (JSON + patch, then migrated) as
    ``tmp_path/<stem>.sqlite``; return its URL and file path."""
    from db_utils import json_to_db  # tests/db_utils.py

    from flextool.update_flextool.db_migration import migrate_database
    from flextool.update_flextool.initialize_database import (
        initialize_database,
    )

    data = json.loads((REPO / fx.json).read_text())
    data = PATCHES[fx.patch](data)
    src_json = tmp_path / f"{stem}.json"
    src_json.write_text(json.dumps(data))
    db_path = tmp_path / f"{stem}.sqlite"
    if fx.json.startswith(CANONICAL_PREFIX):
        # canonical how-to JSON (export_database format)
        if fx.patch != "none":
            raise ValueError("patches apply to tests/fixtures JSON only")
        initialize_database(str(src_json), str(db_path))
        url = f"sqlite:///{db_path.resolve().as_posix()}"
    else:
        url = json_to_db(src_json, db_path)
    # same defensive migration the test DB fixtures apply
    migrate_database(url)
    return url, db_path


def _dirty_work_folder(dirtier: Fixture, tmp_path: Path, work: Path,
                       solver_cfg: Path) -> None:
    """``--dirty-with``: run ``dirtier`` in ``work`` (uncaptured) and leave
    its CSVs there: every step's ``dump_csvs`` in step order, then the last
    step's provider snapshot."""
    from flextool.engine import run_chain_from_db

    url, _ = _build_fixture_db(dirtier, tmp_path, "dirtier")
    steps = run_chain_from_db(
        input_db_url=url,
        scenario_name=dirtier.scenario,
        work_folder=work,
        solver_config_dir=solver_cfg,
        warm=False,
        keep_solutions=True,
        csv_dump=True,
    )
    if not steps:
        raise RuntimeError(f"dirtier {dirtier.name!r} returned no steps")
    for step in steps.values():
        if step.flex_data is not None:
            step.flex_data.dump_csvs(work, include_heavy=True)
    last = next(reversed(list(steps.values())))
    if last.flex_data_provider is None:
        raise RuntimeError(f"dirtier {dirtier.name!r}: last step has no provider")
    last.flex_data_provider.snapshot_processed_inputs(work)


def _worker(json_name: str, scenario: str, patch: str, out_dir: Path, *,
            csv_dump: bool = False, dirty_with: str = "",
            dirty_sqlite: str = "") -> int:
    os.environ.setdefault("FLEXTOOL_SKIP_SOLVER_PROBE", "1")
    sys.path.insert(0, str(TESTS_DIR))
    import warnings

    import importlib

    from flextool.engine import run_chain_from_db

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    used: dict[str, int] = {}
    capture = {"on": False}  # off during a --dirty-with run
    model_mod = importlib.import_module(MODEL_MODULE)
    original = model_mod.build_flextool

    def capturing_build_flextool(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        if not capture["on"]:
            return result
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
    by_name = _fixture_by_name()
    target = Fixture("target", json_name, scenario, patch)

    with tempfile.TemporaryDirectory(prefix="lp_snapshot_") as tmp:
        tmp_path = Path(tmp)
        url, db_path = _build_fixture_db(target, tmp_path, "fixture")
        solver_cfg = tmp_path / "solver_config"
        solver_cfg.mkdir()
        shutil.copy(TESTS_DIR / "highs.opt", solver_cfg / "highs.opt")
        work = tmp_path / "work"
        work.mkdir()
        if dirty_with:
            _dirty_work_folder(by_name[dirty_with], tmp_path, work, solver_cfg)
        if dirty_sqlite:
            if dirty_sqlite == "self":
                sqlite_src = db_path
            else:
                _, sqlite_src = _build_fixture_db(by_name[dirty_sqlite],
                                                  tmp_path, "sqlite_dirt")
            shutil.copy(sqlite_src, work / "tests.sqlite")
        capture["on"] = True
        steps = run_chain_from_db(
            input_db_url=url,
            scenario_name=scenario,
            work_folder=work,
            solver_config_dir=solver_cfg,
            warm=False,
            csv_dump=csv_dump,
        )
        capture["on"] = False
        if csv_dump and steps:
            last = next(reversed(list(steps.values())))
            if last.flex_data_provider is not None:
                last.flex_data_provider.snapshot_processed_inputs(work)

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

@dataclass(frozen=True)
class WorkMode:
    """Work-folder mode of a snapshot (all off = plain)."""
    csv_dump: bool = False
    dirty_with: str = ""    # fixture name, "self" or "" (none)
    dirty_sqlite: str = ""  # fixture name, "self" or "" (none)

    def describe(self) -> str:
        parts = (["csv-dump"] if self.csv_dump else []) + (
            [f"dirty-with={self.dirty_with}"] if self.dirty_with else []) + (
            [f"dirty-sqlite={self.dirty_sqlite}"] if self.dirty_sqlite else [])
        return " ".join(parts) or "plain"

    def worker_args(self, fx: Fixture) -> list[str]:
        """Worker CLI flags; ``self`` resolves to ``fx`` (CSV dirt) or stays
        ``self`` (the worker copies the target's own DB file)."""
        out = ["--csv-dump"] if self.csv_dump else []
        if self.dirty_with:
            out += ["--dirty-with",
                    fx.name if self.dirty_with == "self" else self.dirty_with]
        if self.dirty_sqlite:
            out += ["--dirty-sqlite", self.dirty_sqlite]
        return out


def _run_fixture(fx: Fixture, out_root: Path,
                 mode: WorkMode = WorkMode()) -> tuple[str, int, float, str]:
    out_dir = out_root / fx.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(secrets.randbelow(2**32 - 1) + 1)
    env["FLEXTOOL_SKIP_SOLVER_PROBE"] = "1"
    cmd = [sys.executable, str(Path(__file__).resolve()), "_worker",
           "--json", fx.json, "--scenario", fx.scenario,
           "--patch", fx.patch, "--out", str(out_dir),
           *mode.worker_args(fx)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO),
                          capture_output=True, text=True)
    dt = time.monotonic() - t0
    (out_dir / "_run.log").write_text(
        f"PYTHONHASHSEED={env['PYTHONHASHSEED']}\nmode={mode.describe()}\n"
        f"exit={proc.returncode}\n"
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
    mode = WorkMode(csv_dump=args.csv_dump, dirty_with=args.dirty_with,
                    dirty_sqlite=args.dirty_sqlite)
    bad = [n for n in (mode.dirty_with, mode.dirty_sqlite)
           if n and n != "self" and n not in by_name]
    if bad:
        print(f"unknown dirt fixtures: {bad}", file=sys.stderr)
        return 2
    print(f"mode: {mode.describe()}", flush=True)
    t0 = time.monotonic()
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(_run_fixture, fx, out_root, mode)
                for fx in fixtures]
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
    if args.fixtures:
        wanted = {n.strip() for n in args.fixtures.split(",") if n.strip()}
        rel_a = {r for r in rel_a if r.parts[0] in wanted}
        rel_b = {r for r in rel_b if r.parts[0] in wanted}
        for name in sorted(wanted):
            for label, d in (("base", base), ("new", new)):
                if not (d / name / "manifest.json").is_file():
                    print(f"MISSING fixture in {label}: {name}")
                    n_missing += 1
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
# Keyed like COVERAGE_SOURCES (path relative to ENGINE_DIR).
DYNAMIC_CSTR_NAMES: dict[str, tuple[str, ...]] = {
    "model/core.py": (
        # _eq_invest_period_caps / _eq_invest_total_caps loop tables
        "maxInvest_entity_period_p", "maxInvest_entity_period_n",
        "maxDivest_entity_period_p", "maxDivest_entity_period_n",
        "maxInvest_entity_total", "maxInvest_entity_total_path",
        "maxDivest_entity_total", "maxDivest_entity_total_path",
        "maxInvest_entity_total_n", "maxInvest_entity_total_path_n",
        "maxDivest_entity_total_n", "maxDivest_entity_total_path_n",
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
    "model/_online.py": (
        # f"...{sfx}", sfx=_{kind}
        *(f"{base}_{kind}"
          for base in ("maxOnline", "maxStartup", "maxShutdown",
                       "online__startup", "online__shutdown",
                       "maxFlow_online", "maxOutputSum_online",
                       "maxInputFlow_online", "maxFlow_online_shutdown",
                       "maxInputFlow_online_shutdown", "maxFlow_back_online",
                       "minimum_uptime", "minimum_downtime",
                       "online_fix_eq")
          for kind in ("linear", "integer")),
    ),
    "model/_unit_floors.py": (
        # f"...{sfx}", sfx=_{kind}
        *(f"{base}_{kind}"
          for base in ("minFlow_minload", "minFlow_output_floor")
          for kind in ("linear", "integer")),
    ),
    "model/_non_anticipativity.py": (
        # f"non_anticipativity_online_{kind}"
        "non_anticipativity_online_integer", "non_anticipativity_online_linear",
        # add_invest_constraints loop table
        "non_anticipativity_invest_p", "non_anticipativity_invest_n",
        "non_anticipativity_divest_p", "non_anticipativity_divest_n",
    ),
    "model/_reserve.py": (
        "reserveBalance_up_n_1_eq", "reserveBalance_down_n_1_eq",
        "reserve_process_upward", "reserve_process_downward",
        # f"connection_reserve_{ud}[_online_{kind}|_exact]"
        *(f"connection_reserve_{ud}{sfx}" for ud in ("up", "down")
          for sfx in ("", "_online_linear", "_online_integer", "_exact")),
    ),
    "model/_cumulative_invest.py": (
        # every family passes its names from a loop table
        "minInvest_entity_period_p", "minInvest_entity_period_n",
        "minDivest_entity_period_p", "minDivest_entity_period_n",
        "minInvest_entity_total_p", "minInvest_entity_total_n",
        "minDivest_entity_total_p", "minDivest_entity_total_n",
        "maxCumulative_capacity_p", "maxCumulative_capacity_n",
        "minCumulative_capacity_p", "minCumulative_capacity_n",
        *(f"{pre}{kind}Group_entity_period_{s}"
          for pre in ("max", "min") for kind in ("Invest", "Divest")
          for s in ("p", "n")),
        *(f"{pre}{kind}Group_entity_total_{s}"
          for pre in ("max", "min") for kind in ("Invest", "Divest")
          for s in ("p", "n")),
        # the _path rows exist for the max divest cap only
        "maxDivestGroup_entity_total_path_p",
        "maxDivestGroup_entity_total_path_n",
        *(f"{pre}{tail}"
          for pre in ("max", "min")
          for tail in ("InvestGroup_entity_cumulative",
                       "Cumulative_flow_solve", "Cumulative_flow_period",
                       "Instant_flow")),
        "fix_v_invest_no_investment_eq_p", "fix_v_invest_no_investment_eq_n",
    ),
}

# Families no production build can emit, with the reason.  ``coverage``
# reports them separately and does not count them as gaps.
UNREACHABLE: dict[str, str] = {
    name: ("ed_invest_set (the v_invest_* domain) already anti-joins "
           "ed_invest_forbidden_no_investment (_derived."
           "ed_invest_set_from_source), so the row domain "
           "forbidden ∩ v_invest is always empty")
    for name in ("fix_v_invest_no_investment_eq_p",
                 "fix_v_invest_no_investment_eq_n")
}

def constraint_universe() -> dict[str, str]:
    """family name -> source file, for every add_cstr in COVERAGE_SOURCES."""
    unknown = set(DYNAMIC_CSTR_NAMES) - set(COVERAGE_SOURCES)
    if unknown:
        raise ValueError(
            f"DYNAMIC_CSTR_NAMES keys not in COVERAGE_SOURCES: {sorted(unknown)}"
        )
    out: dict[str, str] = {}
    for fname in COVERAGE_SOURCES:
        text = (ENGINE_DIR / fname).read_text()
        for name in _ADD_CSTR_LITERAL.findall(text):
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
    sp.add_argument("--csv-dump", action="store_true",
                    help="run with csv_dump=True + provider snapshot")
    sp.add_argument("--dirty-with", default="", metavar="FIXTURE|self",
                    help="first run FIXTURE in the work folder and leave its "
                         "CSVs (no tests.sqlite)")
    sp.add_argument("--dirty-sqlite", default="", metavar="FIXTURE|self",
                    help="copy FIXTURE's DB to work/tests.sqlite (no CSVs)")
    sp.set_defaults(func=cmd_snapshot)

    sp = sub.add_parser("compare", help="compare two snapshots")
    sp.add_argument("base_dir")
    sp.add_argument("new_dir")
    sp.add_argument("--canonical", action="store_true",
                    help="compare up to row/column permutation, by name")
    sp.add_argument("--rtol", type=float, default=0.0,
                    help="relative float tolerance (canonical mode)")
    sp.add_argument("--fixtures", default="",
                    help="comma-separated fixture dirs to compare (each must "
                         "exist in both)")
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
    sp.add_argument("--csv-dump", action="store_true")
    sp.add_argument("--dirty-with", default="")
    sp.add_argument("--dirty-sqlite", default="")
    sp.set_defaults(func=lambda a: _worker(
        a.json, a.scenario, a.patch, Path(a.out), csv_dump=a.csv_dump,
        dirty_with=a.dirty_with, dirty_sqlite=a.dirty_sqlite))

    args = ap.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
