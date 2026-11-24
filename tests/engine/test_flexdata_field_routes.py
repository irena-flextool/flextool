"""Every :class:`FlexData` field has a declared route through the CSV dump -> reload round trip.

The recurring mistake: a new FlexData field is written only by the DB overlay (``_apply_db_overrides``), so the
CSV dump (``flextool/engine/output/_dump_csvs.py``) does not write it and a reader-less reload
(``flextool.engine._csv_reload.load_from_dump``) leaves it None (``process_online_fix_in_uc``, commit 0c541f92).
This static test (no solve, no DB) puts every field in exactly one route:

1. **dumped and read back** (computed): a key of ``DIRECT_WRITES`` or a value of a ``_P*_SLICES`` map in
   ``_dump_csvs.py`` that the reload reads back: the loader's reader-less path (below) or the reload's
   ``SEEDED_FIELDS`` (``flextool/engine/_csv_reload/_field_seeds.py``).
2. **re-derived on reload**: set by ``load_flextool`` without a reader (computed from its source: the
   ``FlexData(...)`` call, the dicts its ``_load_*`` / ``load_data`` helpers return, and the ``flex_data.<f> =``
   writes outside the ``db_reader`` / ``handoff`` branches) from the run's CSVs that ``copy_runner_meta`` adds to
   the dump, or a seed of ``SEEDED_FIELDS`` that reads such a CSV, or :data:`REDERIVED` (this module).
3. **not reloaded**: :data:`NOT_RELOADED` (this module), with the reason.

Adding a field: give it a CSV seed in the loader (route 2, nothing to list), or write it in ``DIRECT_WRITES`` and
read it back in ``SEEDED_FIELDS`` (route 1, as 0c541f92 did), or list it in :data:`NOT_RELOADED` with the reason.
"""
from __future__ import annotations

import ast
import dataclasses
import re
from functools import cache
from pathlib import Path

from flextool.engine._csv_reload._field_seeds import SEEDED_FIELDS
from flextool.engine.output import _dump_csvs
from flextool.engine.preprocess.flex_data import FlexData

_ENGINE = Path(__file__).resolve().parents[2] / "flextool" / "engine"
_FLEX_DATA = _ENGINE / "preprocess" / "flex_data.py"
FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(FlexData))

#: Branches of ``load_flextool`` a reader-less reload does not run.
_SKIPPED_BRANCHES = ("db_reader is not None", "handoff is not None")


@cache
def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in ast.parse(path.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}


def _resolve(func: ast.expr) -> ast.FunctionDef | None:
    """The definition of a call to a ``flex_data.py`` function or to ``<module>.<f>`` of ``engine/model/``."""
    if isinstance(func, ast.Name):
        return _functions(_FLEX_DATA).get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        path = _ENGINE / "model" / f"{func.value.id}.py"
        return _functions(path).get(func.attr) if path.exists() else None
    return None


def _is_none(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _key_names(key: ast.expr, loops: dict[str, list[str]]) -> set[str]:
    """Field names a dict key can take: a string, an f-string (matched against the fields) or a loop variable."""
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return {key.value}
    if isinstance(key, ast.JoinedStr):
        pattern = "".join(re.escape(v.value) if isinstance(v, ast.Constant) else ".+" for v in key.values)
        return {f for f in FIELDS if re.fullmatch(pattern, f)}
    if isinstance(key, ast.Name):
        return set(loops.get(key.id, ()))
    return set()


def _returned_keys(fn: ast.FunctionDef) -> set[str]:
    """Keys a loader helper fills with something other than a literal None (its ``None`` placeholders are not
    seeds), following ``**helper(...)`` into the helper."""
    loops: dict[str, list[str]] = {}  # loop variable -> the strings it takes (``for name, key in ((...), ...)``)
    for n in ast.walk(fn):
        if isinstance(n, (ast.For, ast.comprehension)) and isinstance(n.iter, (ast.Tuple, ast.List)):
            targets = n.target.elts if isinstance(n.target, ast.Tuple) else [n.target]
            for i, target in enumerate(targets):
                rows = [e.elts[i] if isinstance(e, ast.Tuple) else e for e in n.iter.elts]
                loops.setdefault(getattr(target, "id", ""), []).extend(
                    e.value for e in rows if isinstance(e, ast.Constant))
    keys: set[str] = set()
    for n in ast.walk(fn):
        pairs: list[tuple[ast.expr | None, ast.expr]] = []
        if isinstance(n, ast.Dict):
            pairs = list(zip(n.keys, n.values))
        elif isinstance(n, ast.Call) and getattr(n.func, "id", None) == "dict":
            pairs = [(None if k.arg is None else ast.Constant(k.arg), k.value) for k in n.keywords]
        elif isinstance(n, ast.DictComp):
            pairs = [(n.key, n.value)]
        elif isinstance(n, ast.Assign):
            pairs = [(t.slice, n.value) for t in n.targets if isinstance(t, ast.Subscript)]
        for key, value in pairs:
            if key is None and isinstance(value, ast.Call) and (inner := _resolve(value.func)) is not None:
                keys |= _returned_keys(inner)
            elif key is not None and not _is_none(value):
                keys |= _key_names(key, loops)
    return keys & FIELDS


def _attribute_writes(body: list[ast.stmt], obj: str, seen: set[str]) -> set[str]:
    """``<obj>.<f> =`` / ``setattr(<obj>, "<f>", ...)`` in ``body`` outside :data:`_SKIPPED_BRANCHES`, following
    calls that pass ``obj`` positionally into the callee."""
    out: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.If) and any(s in ast.unparse(node.test) for s in _SKIPPED_BRANCHES):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(target, ast.Attribute) and getattr(target.value, "id", None) == obj:
                    out.add(target.attr)
        if isinstance(node, ast.Call):
            args = node.args
            if (getattr(node.func, "id", None) == "setattr" and len(args) > 1 and getattr(args[0], "id", None) == obj
                    and isinstance(args[1], ast.Constant)):
                out.add(args[1].value)
            for i, arg in enumerate(args):
                callee = _resolve(node.func) if getattr(arg, "id", None) == obj else None
                if callee is not None and callee.name not in seen and i < len(callee.args.args):
                    seen.add(callee.name)
                    out.update(_attribute_writes(callee.body, callee.args.args[i].arg, seen))
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in body:
        visit(stmt)
    return out


@cache
def loader_writes() -> frozenset[str]:
    """The fields ``load_flextool`` sets on a reader-less load (module docstring, route 2)."""
    lf = _functions(_FLEX_DATA)["load_flextool"]
    ctor = next(n for n in ast.walk(lf) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "FlexData")
    out = {k.arg for k in ctor.keywords if k.arg is not None and not _is_none(k.value)}
    spread = {k.value.id for k in ctor.keywords if k.arg is None}
    for n in ast.walk(lf):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.value, ast.Call):
            target = n.targets[0]
            if getattr(target, "id", None) in spread and (fn := _resolve(n.value.func)) is not None:
                out |= _returned_keys(fn)
        if isinstance(n, ast.Assign):  # startup["..."] = ... / startup.update(helper(...))
            out |= {t.slice.value for t in n.targets if isinstance(t, ast.Subscript)
                    and getattr(t.value, "id", None) in spread and isinstance(t.slice, ast.Constant)}
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "update"
                and getattr(n.func.value, "id", None) in spread and isinstance(n.args[0], ast.Call)
                and (fn := _resolve(n.args[0].func)) is not None):
            out |= _returned_keys(fn)
    out |= _attribute_writes(lf.body, "flex_data", set())
    return frozenset(out & FIELDS)


def dumped() -> frozenset[str]:
    """The fields ``dump_csvs`` writes under their own name (``DIRECT_WRITES`` and the ``_P*_SLICES`` maps)."""
    slices = [v for name, v in vars(_dump_csvs).items() if name.startswith("_P") and name.endswith("_SLICES")]
    return frozenset(_dump_csvs.DIRECT_WRITES).union(*(m.values() for m in slices))


#: Route 2 by hand: fields the reload re-derives from other dumped data that :func:`loader_writes` cannot see.
REDERIVED: dict[str, tuple[str, ...]] = {}

def _overlay(feature: str) -> str:
    return (f"unverified ({feature}): written only by the DB overlay, so a reader-less reload leaves it None "
            "(a reload with db_url + scenario runs the overlay)")


#: Route 3: reason -> the fields a reader-less reload does not bring back.
NOT_RELOADED: dict[str, tuple[str, ...]] = {
    "solve-to-solve carry (handoff, apply_existing_chain): a reload has no carry (_csv_reload.load)": (
        "p_entity_previously_invested_capacity", "p_entity_invested", "p_entity_divested",
        "p_entity_carried_commission", "p_entity_carried_in_construction", "p_entity_retired_carried"),
    "piecewise_linear units: dumped for inspection, a reader-less reload refuses them (_refuse_piecewise_units)": (
        "process_piecewise", "process_piecewise_needs_online", "process_piecewise_remap",
        "p_piecewise_slope", "p_piecewise_intercept", "p_piecewise_points"),
    "set by reduce_unit_detail_method, which a reload does not apply (_csv_reload.load)": ("relax_integrality",),
    "not set by the load: model/core.py computes it on the build": ("nodeState_dt",),
    "unverified (suspected gap): dumped, but neither the loader nor SEEDED_FIELDS reads it back": (
        "e_invest_total", "e_divest_total", "p_commodity_unitsize", "p_state_end"),
    _overlay("run-level costs, discounting"): (
        "p_years_represented_d", "p_inflation_op_run", "p_inflation_inv_run", "ed_entity_annual_discounted_run",
        "edl_invest_lines_run", "edl_divest_lines", "edl_existing_credit_lines", "edl_existing_replacement_lines",
        "cdl_divest_built_lines", "cdl_extend_lines"),
    _overlay("arc capacity and input shares"): (
        "p_arc_max_cap_coef", "p_indirect_input_cap", "process_source_sink_uncapped",
        "p_process_sink_max_capacity_coef", "p_process_sink_min_capacity_coef",
        "p_process_source_input_share_max", "p_process_source_input_share_min", "process_source_sink_2way_1var"),
    _overlay("user constraints"): (
        "constraint_period_total", "p_node_constraint_capacity_coeff", "p_process_constraint_capacity_coeff",
        "p_node_constraint_retired_capacity_coeff", "p_process_constraint_retired_capacity_coeff",
        "p_node_constraint_in_construction_capacity_coeff", "p_process_constraint_in_construction_capacity_coeff",
        "p_process_constraint_online_coeff"),
    _overlay("investment cohorts, lead time, salvage, extension"): (
        "dd_forward", "ed_invest_commission", "edd_invest_commission", "edd_invest_in_construction",
        "e_construction_lead_time", "ed_divest_salvage", "ed_existing_salvage_credit", "ed_existing_replacement",
        "cd_divest_built", "cdd_divest_built", "cohort_divest_built", "cd_divest_built_cost", "cd_divest_built_fixed",
        "cd_extend", "cdd_extend", "cohort_extend", "cd_extend_cost", "cd_extend_fixed", "p_entity_pre_existing"),
    _overlay("ramp speeds and costs"): (
        "process_source_sink_ramp_cost", "p_ramp_cost_arc", "p_ramp_speed_up_sink", "p_ramp_speed_down_sink",
        "p_ramp_speed_up_source", "p_ramp_speed_down_source"),
    _overlay("minimum up / down time"): (
        "pdt_uptime_set", "pdt_downtime_set", "uptime_lookback", "downtime_lookback", "uptime_carry",
        "downtime_carry"),
    _overlay("storage blocks"): ("dtttdt_block_lag",),
    _overlay("capacity margin, inertia, non-synchronous limits"): (
        "pdGroup_capacity_margin", "pdGroup_capacity_margin_relative", "pdGroup_penalty_capacity_margin",
        "pdtGroup_inertia_limit", "pdGroup_penalty_inertia", "pdtGroup_non_synchronous_limit",
        "pdGroup_penalty_non_synchronous", "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
        "p_positive_inflow", "p_negative_inflow", "pdtNodeInflow_per_step"),
    _overlay("reserves"): (
        "prundt", "process_reserve_upDown_node_active", "p_process_reserve_upDown_node_reliability",
        "p_reserve_upDown_group_penalty_reserve", "p_reserve_upDown_group_reserve_duration",
        "p_reserve_upDown_group_increase_reserve_ratio", "p_process_reserve_upDown_node_max_share",
        "p_process_reserve_upDown_node_large_failure_ratio_value",
        "p_process_reserve_upDown_node_increase_reserve_ratio_value"),
    _overlay("cumulative invest and flow limits"): (
        "ed_invest_min_period", "ed_divest_min_period", "ed_cumulative_max_capacity", "ed_cumulative_min_capacity",
        "p_group_invest_max_period", "p_group_invest_min_period", "p_group_retire_max_period",
        "p_group_retire_min_period", "p_group_invest_max_total", "p_group_invest_min_total",
        "p_group_retire_max_total", "p_group_retire_min_total", "pd_group_cumulative_max_capacity",
        "pd_group_cumulative_min_capacity", "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
        "pd_max_cumulative_flow", "pd_min_cumulative_flow"),
    _overlay("stochastic lineage, recourse"): ("dd_same_scenario", "pd_non_anticipativity", "d_leaf", "recourse_invest"),
}


def _routes() -> dict[str, set[str]]:
    read_back = loader_writes() | set(SEEDED_FIELDS)
    return {
        "dumped and read back": set(dumped() & read_back),
        "re-derived (loader / SEEDED_FIELDS)": set(read_back - dumped()),
        "REDERIVED": {f for names in REDERIVED.values() for f in names},
        "NOT_RELOADED": {f for names in NOT_RELOADED.values() for f in names},
    }


def test_every_field_has_a_route():
    """A FlexData field in no route fails with the three ways to give it one."""
    routes = _routes()
    missing = sorted(FIELDS - set().union(*routes.values()))
    assert not missing, (
        f"FlexData field(s) {missing} do not survive the CSV dump -> reload round trip.  Give each one route: "
        "(1) write it in DIRECT_WRITES (flextool/engine/output/_dump_csvs.py) and read it back in SEEDED_FIELDS "
        "(flextool/engine/_csv_reload/_field_seeds.py, plus its round trip in tests/csv_reload/test_dump_seeds.py); "
        "(2) seed it in load_flextool's reader-less path (flextool/engine/preprocess/flex_data.py), or list how the "
        "reload re-derives it in REDERIVED; (3) list it in NOT_RELOADED with the reason (both lists in "
        "tests/engine/test_flexdata_field_routes.py).")


def test_no_field_in_two_routes():
    """Hand-listed fields are not also in another route, and each is listed once."""
    routes = _routes()
    listed = [f for names in (*REDERIVED.values(), *NOT_RELOADED.values()) for f in names]
    assert len(listed) == len(set(listed)), sorted({f for f in listed if listed.count(f) > 1})
    pairs = [(a, b) for a in routes for b in routes if a < b and routes[a] & routes[b]]
    assert not pairs, {f"{a} & {b}": sorted(routes[a] & routes[b]) for a, b in pairs}


def test_no_stale_names():
    """Every listed name, and every field the dump or the reload seeds name, is a FlexData field."""
    listed = {f for names in (*REDERIVED.values(), *NOT_RELOADED.values()) for f in names}
    assert not listed - FIELDS, sorted(listed - FIELDS)
    assert not (dumped() | set(SEEDED_FIELDS)) - FIELDS, sorted((dumped() | set(SEEDED_FIELDS)) - FIELDS)
