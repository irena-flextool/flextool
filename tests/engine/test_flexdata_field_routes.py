"""Every :class:`FlexData` field is either in the CSV dump of the model contents or declared not to be.

The CSV dump (``FlexData.dump_csvs``, ``flextool/engine/output/_dump_csvs.py``) is maintained; the CSV *reload*
(``flextool/engine/_csv_reload/``) is shelved: don't extend it or add ``SEEDED_FIELDS`` seeds unless the owner asks.
This static test (no solve, no DB) puts every field in exactly one route:

1. **written by the dump** (computed): a key of ``DIRECT_WRITES`` or a value of a ``_P*_SLICES`` map.
2. **in the dump through the run's input CSVs** that the dump folder carries (``copy_runner_meta``), computed:
   fields ``load_flextool`` builds from those CSVs without a DB reader (the ``FlexData(...)`` call, the dicts its
   ``_load_*`` / ``load_data`` helpers return, the ``flex_data.<f> =`` writes outside the ``db_reader`` /
   ``handoff`` branches) and the existing ``SEEDED_FIELDS`` (each reads a run CSV).  ``SEEDED_FIELDS`` is read
   from the source of ``_csv_reload/_field_seeds.py``: nothing outside the shelved package imports it
   (``test_csv_reload_is_a_leaf``).
3. **not dumped**: :data:`NOT_DUMPED` (this module), with the reason per group.

Adding a field: add a ``DIRECT_WRITES`` / slice entry in ``_dump_csvs.py``, or a :data:`NOT_DUMPED` entry with
the reason.
"""
from __future__ import annotations

import ast
import dataclasses
import re
from functools import cache
from pathlib import Path

from flextool.engine.output import _dump_csvs
from flextool.engine.preprocess.flex_data import FlexData

_ENGINE = Path(__file__).resolve().parents[2] / "flextool" / "engine"
_FLEX_DATA = _ENGINE / "preprocess" / "flex_data.py"
_FIELD_SEEDS = _ENGINE / "_csv_reload" / "_field_seeds.py"
FIELDS: frozenset[str] = frozenset(f.name for f in dataclasses.fields(FlexData))

#: Branches of ``load_flextool`` that need a DB reader or a handoff (not built from the run CSVs).
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


@cache
def seeded_fields() -> frozenset[str]:
    """``SEEDED_FIELDS`` of ``_field_seeds.py``, read from its source: a tuple of strings and ``*<name>`` of the
    module's own string tuples."""
    tuples: dict[str, ast.expr] = {}
    for n in ast.parse(_FIELD_SEEDS.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            tuples[n.target.id] = n.value
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            tuples[n.targets[0].id] = n.value

    def strings(node: ast.expr) -> list[str]:
        assert isinstance(node, ast.Tuple), f"{_FIELD_SEEDS.name}: expected a tuple, got {ast.unparse(node)}"
        out: list[str] = []
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                out.append(e.value)
            elif isinstance(e, ast.Starred) and isinstance(e.value, ast.Name) and e.value.id in tuples:
                out.extend(strings(tuples[e.value.id]))
            else:
                raise AssertionError(f"{_FIELD_SEEDS.name}: SEEDED_FIELDS element {ast.unparse(e)} is not a "
                                     "string or *<module string tuple>")
        return out

    assert "SEEDED_FIELDS" in tuples, f"{_FIELD_SEEDS.name} no longer defines SEEDED_FIELDS"
    return frozenset(strings(tuples["SEEDED_FIELDS"]))


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


_OVERLAY = "from the DB overlay; not in the dump"

#: Route 3: reason -> the fields the dump does not hold.
NOT_DUMPED: dict[str, tuple[str, ...]] = {
    "solve-to-solve carry (handoff, apply_existing_chain), not model contents of the solve": (
        "p_entity_previously_invested_capacity", "p_entity_invested", "p_entity_divested",
        "p_entity_carried_commission", "p_entity_carried_in_construction", "p_entity_retired_carried"),
    "set by reduce_unit_detail_method on the loaded FlexData": ("relax_integrality",),
    "not set by the load: model/core/ computes it on the build": ("nodeState_dt",),
    f"{_OVERLAY} (run-level costs, discounting)": (
        "p_years_represented_d", "p_inflation_op_run", "p_inflation_inv_run", "ed_entity_annual_discounted_run",
        "edl_invest_lines_run", "edl_divest_lines", "edl_existing_credit_lines", "edl_existing_replacement_lines",
        "cdl_divest_built_lines", "cdl_extend_lines"),
    f"{_OVERLAY} (arc capacity and input shares)": (
        "p_arc_max_cap_coef", "p_indirect_input_cap", "process_source_sink_uncapped",
        "p_process_sink_max_capacity_coef", "p_process_sink_min_capacity_coef",
        "p_process_source_input_share_max", "p_process_source_input_share_min", "process_source_sink_2way_1var"),
    f"{_OVERLAY} (user constraints)": (
        "constraint_period_total", "p_node_constraint_capacity_coeff", "p_process_constraint_capacity_coeff",
        "p_node_constraint_retired_capacity_coeff", "p_process_constraint_retired_capacity_coeff",
        "p_node_constraint_in_construction_capacity_coeff", "p_process_constraint_in_construction_capacity_coeff",
        "p_process_constraint_online_coeff"),
    f"{_OVERLAY} (investment cohorts, lead time, salvage, extension)": (
        "dd_forward", "ed_invest_commission", "edd_invest_commission", "edd_invest_in_construction",
        "e_construction_lead_time", "ed_divest_salvage", "ed_existing_salvage_credit", "ed_existing_replacement",
        "cd_divest_built", "cdd_divest_built", "cohort_divest_built", "cd_divest_built_cost", "cd_divest_built_fixed",
        "cd_extend", "cdd_extend", "cohort_extend", "cd_extend_cost", "cd_extend_fixed", "p_entity_pre_existing"),
    f"{_OVERLAY} (ramp speeds and costs)": (
        "process_source_sink_ramp_cost", "p_ramp_cost_arc", "p_ramp_speed_up_sink", "p_ramp_speed_down_sink",
        "p_ramp_speed_up_source", "p_ramp_speed_down_source"),
    f"{_OVERLAY} (minimum up / down time)": (
        "pdt_uptime_set", "pdt_downtime_set", "uptime_lookback", "downtime_lookback", "uptime_carry",
        "downtime_carry"),
    f"{_OVERLAY} (storage blocks)": ("dtttdt_block_lag",),
    f"{_OVERLAY} (capacity margin, inertia, non-synchronous limits)": (
        "pdGroup_capacity_margin", "pdGroup_capacity_margin_relative", "pdGroup_penalty_capacity_margin",
        "pdtGroup_inertia_limit", "pdGroup_penalty_inertia", "pdtGroup_non_synchronous_limit",
        "pdGroup_penalty_non_synchronous", "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
        "p_positive_inflow", "p_negative_inflow", "pdtNodeInflow_per_step"),
    f"{_OVERLAY} (reserves)": (
        "prundt", "process_reserve_upDown_node_active", "p_process_reserve_upDown_node_reliability",
        "p_reserve_upDown_group_penalty_reserve", "p_reserve_upDown_group_reserve_duration",
        "p_reserve_upDown_group_increase_reserve_ratio", "p_process_reserve_upDown_node_max_share",
        "p_process_reserve_upDown_node_large_failure_ratio_value",
        "p_process_reserve_upDown_node_increase_reserve_ratio_value"),
    f"{_OVERLAY} (cumulative invest and flow limits)": (
        "ed_invest_min_period", "ed_divest_min_period", "ed_cumulative_max_capacity", "ed_cumulative_min_capacity",
        "p_group_invest_max_period", "p_group_invest_min_period", "p_group_retire_max_period",
        "p_group_retire_min_period", "p_group_invest_max_total", "p_group_invest_min_total",
        "p_group_retire_max_total", "p_group_retire_min_total", "pd_group_cumulative_max_capacity",
        "pd_group_cumulative_min_capacity", "p_group_max_cumulative_flow", "p_group_min_cumulative_flow",
        "pd_max_cumulative_flow", "pd_min_cumulative_flow"),
    f"{_OVERLAY} (stochastic lineage, recourse)": ("dd_same_scenario", "pd_non_anticipativity", "d_leaf",
                                                   "recourse_invest"),
}


def _routes() -> dict[str, set[str]]:
    return {
        "written by the dump": set(dumped()),
        "in the run's input CSVs": set((loader_writes() | seeded_fields()) - dumped()),
        "NOT_DUMPED": {f for names in NOT_DUMPED.values() for f in names},
    }


def test_every_field_has_a_route():
    """A FlexData field in no route fails with the two ways to give it one."""
    routes = _routes()
    missing = sorted(FIELDS - set().union(*routes.values()))
    assert not missing, (
        f"FlexData field(s) {missing} are not in the CSV dump of the model contents.  Either add a DIRECT_WRITES "
        "or _P*_SLICES entry in flextool/engine/output/_dump_csvs.py, or list the field in NOT_DUMPED with the "
        "reason (tests/engine/test_flexdata_field_routes.py).  The CSV reload (flextool/engine/_csv_reload/) is "
        "shelved: don't extend it or add SEEDED_FIELDS seeds unless the owner asks.")


def test_no_field_in_two_routes():
    """NOT_DUMPED lists each field once and none that the dump already holds."""
    routes = _routes()
    listed = [f for names in NOT_DUMPED.values() for f in names]
    assert len(listed) == len(set(listed)), sorted({f for f in listed if listed.count(f) > 1})
    pairs = [(a, b) for a in routes for b in routes if a < b and routes[a] & routes[b]]
    assert not pairs, {f"{a} & {b}": sorted(routes[a] & routes[b]) for a, b in pairs}


def test_no_stale_names():
    """Every name NOT_DUMPED, the dump or SEEDED_FIELDS lists is a FlexData field."""
    named = {f for names in NOT_DUMPED.values() for f in names} | dumped() | seeded_fields()
    assert not named - FIELDS, sorted(named - FIELDS)
