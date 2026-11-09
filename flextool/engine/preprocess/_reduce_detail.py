"""Solve-level detail reduction: ``solve.reduce_unit_detail_method`` / ``solve.reduce_system_detail_method``.

:func:`apply_reduce_detail` takes the :class:`FlexData` that ``load_flextool`` built for a sub-solve and returns the
reduced data the rest of the cascade sees (the build, warm-start reuse, Benders, the handoff and the per-roll
outputs).  It only removes or relaxes detail that the cascade produced from the entity data; it never adds any.
Every operation is one of: clear a set (its rows go), make the integer online columns continuous, or fold the
``min_load_efficiency`` section into the slope (an approximation: at part load less fuel than the curve, at full
load the same).  A level that finds nothing to reduce returns the data unchanged, so the LP is byte-identical.

Unit detail (``reduce_unit_detail_method``; the parts each value takes are in :data:`UNIT_PARTS`):

* ``integers`` — ``relax_integrality`` (``v_online_integer`` continuous, same names and rows) and ``process_exact``
  cleared: an ``exact`` connection behaves as ``regular`` (both directions keep their own flow and losses, only the
  direction binary and its two rows go).
* ``ramps`` — ramp limits and ramp costs: the four ramp-limit sets, the ramp speeds, the ramp-cost set and its
  ``p_ramp_cost_arc``, and the carried roll-end flow (``p_roll_continue_flow`` and its duration; nothing reads it
  without them).  ``dt_ramp_link`` / ``dt_online_wrap`` / ``dt_ramp_carry`` stay: online_change and the minimum
  up / down times use them as step predecessor.
* ``min_limits`` — minimum load (``process_minload``, ``p_min_load``), the ``capacity_min_coeff`` floors
  (``p_process_sink_min_capacity_coef``) and the minimum up / down times with their roll carry.
* ``online`` — the online variables (every online set, start-up cost sets, the online carry and pin) and with them
  ``min_limits``; ``min_load_efficiency`` units of the online set run at their full-load efficiency
  (``p_slope ← p_slope + p_section``, ``p_section ← 0``, ``process_min_load_eff`` cleared); user constraints with a
  ``constraint_online_coeff`` term are left out whole (dropping only the term could tighten the row).

The carried roll-end flow is kept under ``integers`` / ``online`` without ``ramps``: the ramp limits and the ramp
cost still need it at a continuation roll's first step.

System detail (``reduce_system_detail_method``; the value is its parts joined in the fixed order of
:data:`SYSTEM_TOKENS`, :func:`system_parts`):

* ``reserves`` — every reserve field (:data:`RESERVE_FIELDS`): the reserve groups (the gate of every reserve row),
  the per-process reserve sets and parameters, the realization pin ``p_fix_reserve`` and with them the reserve
  non-anticipativity ties.  The units' coupling to reserves (maxFlow, the floors, the ramp reserve rows) is gated
  on the reserve variables and drops out.
* ``inertia`` — the inertia groups, arcs, constants, limits and penalties (:data:`INERTIA_FIELDS`).
* ``non_synchronous`` — the non-synchronous groups, arcs, limits and penalties (:data:`NON_SYNC_FIELDS`).
* ``dc_power_flow`` — the angle model (:data:`DC_FIELDS`).  The lines of a ``dc_power_flow_with_angles`` group were
  given ``no_losses_no_variable_cost`` by the input derivation, so they keep their two-way flow, capacity and
  node-balance terms: lossless transport limited by their capacity.

The capacity margin is never reduced.  The report's ``notes`` name the interactions of the two levels (units whose
reserve now comes from built capacity, inertia groups counted from flows, downward reserve reaching down to zero
flow).
"""
from __future__ import annotations

import copy
import dataclasses
import logging
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param

from flextool.engine.input._solve_config import REDUCE_SYSTEM_DETAIL_METHODS, REDUCE_UNIT_DETAIL_METHODS
from flextool.engine.input._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine.preprocess.flex_data import FlexData

_LOG = logging.getLogger(__name__)

#: The parts of unit detail each ``reduce_unit_detail_method`` value leaves out.  ``online`` contains every
#: ``integers_min_limits`` effect (an ``online`` solve with a binary ``exact`` connection would stay a MIP).
UNIT_PARTS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "integers": frozenset({"integers"}),
    "integers_ramps": frozenset({"integers", "ramps"}),
    "integers_min_limits": frozenset({"integers", "min_limits"}),
    "integers_ramps_min_limits": frozenset({"integers", "ramps", "min_limits"}),
    "online": frozenset({"integers", "min_limits", "online"}),
    "online_ramps": frozenset({"integers", "ramps", "min_limits", "online"}),
}

#: The system parts, in the fixed word order of the value names.
SYSTEM_TOKENS: tuple[str, ...] = ("reserves", "inertia", "non_synchronous", "dc_power_flow")

#: FlexData fields cleared (set to None) by each unit part.
RAMP_FIELDS: tuple[str, ...] = (
    "process_source_sink_ramp_limit_sink_up", "process_source_sink_ramp_limit_sink_down",
    "process_source_sink_ramp_limit_source_up", "process_source_sink_ramp_limit_source_down",
    "p_ramp_speed_up_sink", "p_ramp_speed_down_sink", "p_ramp_speed_up_source", "p_ramp_speed_down_source",
    "process_source_sink_ramp_cost", "p_ramp_cost_arc",
    "p_roll_continue_flow", "p_roll_continue_flow_duration",
)
MIN_LOAD_FIELDS: tuple[str, ...] = ("process_minload", "p_min_load", "p_process_sink_min_capacity_coef")
MIN_TIME_FIELDS: tuple[str, ...] = (
    "pdt_uptime_set", "pdt_downtime_set", "uptime_lookback", "downtime_lookback",
    "uptime_carry", "downtime_carry", "online_startup_history", "online_min_time",
)
ONLINE_FIELDS: tuple[str, ...] = (
    "process_online", "process_online_linear", "process_online_integer", "p_online_dt",
    "pdt_online_linear", "pdt_online_integer", "p_roll_continue_online", "dt_online_carry", "p_fix_online",
)

#: FlexData fields cleared (set to None) by each system part.
RESERVE_FIELDS: tuple[str, ...] = (
    "reserve_upDown_group", "reserve_upDown_group_method_timeseries", "reserve_upDown_group_method_dynamic",
    "reserve_upDown_group_method_n_1", "prundt", "process_reserve_upDown_node_active", "p_fix_reserve",
    "process_reserve_upDown_node_increase_reserve_ratio", "process_reserve_upDown_node_large_failure_ratio",
    "p_process_reserve_upDown_node_reliability", "pdtReserve_upDown_group_reservation",
    "p_reserve_upDown_group_penalty_reserve", "p_reserve_upDown_group_reserve_duration",
    "p_reserve_upDown_group_increase_reserve_ratio", "p_process_reserve_upDown_node_max_share",
    "p_process_reserve_upDown_node_large_failure_ratio_value",
    "p_process_reserve_upDown_node_increase_reserve_ratio_value",
)
INERTIA_FIELDS: tuple[str, ...] = (
    "groupInertia", "process_sink_inertia", "process_source_inertia", "pdtGroup_inertia_limit",
    "pdGroup_penalty_inertia", "p_process_sink_inertia_constant", "p_process_source_inertia_constant",
)
NON_SYNC_FIELDS: tuple[str, ...] = (
    "groupNonSync", "process_sink_nonSync", "process_group_inside_nonSync", "pdtGroup_non_synchronous_limit",
    "pdGroup_penalty_non_synchronous", "groupNonSync_valued",
)
DC_FIELDS: tuple[str, ...] = (
    "node_dc_power_flow", "connection_dc_power_flow", "node_reference_angle", "p_connection_susceptance",
    "process_source_toSink_dc",
)
SYSTEM_FIELDS: dict[str, tuple[str, ...]] = {
    "reserves": RESERVE_FIELDS, "inertia": INERTIA_FIELDS, "non_synchronous": NON_SYNC_FIELDS,
    "dc_power_flow": DC_FIELDS,
}


@dataclasses.dataclass
class ReduceReport:
    """What :func:`apply_reduce_detail` left out of one sub-solve.

    ``removed``: feature -> sorted names of the entities (processes, or user constraints for
    ``user_constraints``) it was left out for; only features with something left out appear.  Keys: ``integer_online``
    (binary units / connections now continuous), ``exact`` (exact connections now regular), ``ramp_limits``,
    ``ramp_costs``, ``min_load``, ``capacity_min_coeff``, ``min_up_down``, ``online``, ``min_load_efficiency``
    (folded to full-load efficiency), ``user_constraints`` (left out whole); system keys ``reserves`` (the reserve
    groups with a requirement as ``reserve__upDown__group``, or every reserve group when none has one),
    ``inertia`` / ``non_synchronous`` (the groups), ``dc_power_flow`` (the connections now lossless transport).  ``notes``: interaction facts for the run log, each present only when both
    sides are in the solve: ``fractional_online_constraints`` (user constraints that now count fractional units
    online), ``reserves_from_capacity`` (units / connections left without online variables that keep a reserve:
    it now comes from built capacity, not online capacity), ``inertia_from_flows`` (inertia groups whose providers
    lost their online variables: inertia is counted from flows), ``downward_reserve_to_zero`` (processes left
    without min load that keep a downward reserve: it now reaches down to zero flow).  ``warnings``: the lines to
    log as warnings (a user constraint left out).  ``fields``: the FlexData fields the transform changed.
    """

    unit: str = "none"
    system: str = "none"
    removed: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    notes: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    warnings: list[str] = dataclasses.field(default_factory=list)
    fields: frozenset[str] = frozenset()

    @property
    def changed(self) -> bool:
        """True when the data was reduced (something was left out or relaxed, or a field was cleared)."""
        return bool(self.fields) or bool(self.removed)


def unit_parts(value: str) -> frozenset[str]:
    """The unit parts of a resolved ``reduce_unit_detail_method`` value; an unknown value raises."""
    parts = UNIT_PARTS.get(value)
    if parts is None:
        raise FlexToolConfigError(
            f"reduce_unit_detail_method = '{value}' is not one of {', '.join(REDUCE_UNIT_DETAIL_METHODS)}.")
    return parts


def system_parts(value: str) -> frozenset[str]:
    """The system parts (of :data:`SYSTEM_TOKENS`) of a resolved ``reduce_system_detail_method`` value: the value
    is its parts joined by ``_`` in the fixed order; ``none`` has none.  An unknown value raises."""
    if value not in REDUCE_SYSTEM_DETAIL_METHODS:
        raise FlexToolConfigError(
            f"reduce_system_detail_method = '{value}' is not one of {', '.join(REDUCE_SYSTEM_DETAIL_METHODS)}.")
    if value == "none":
        return frozenset()
    parts: list[str] = []
    rest = value
    for token in SYSTEM_TOKENS:
        if rest == token:
            parts.append(token)
            rest = ""
            break
        if rest.startswith(token + "_"):
            parts.append(token)
            rest = rest[len(token) + 1:]
    if rest:
        raise FlexToolConfigError(f"reduce_system_detail_method = '{value}': cannot read the part '{rest}'.")
    return frozenset(parts)


# ---------------------------------------------------------------------------
# helpers


def _frame(x) -> pl.DataFrame | None:
    if x is None:
        return None
    f = x.frame if isinstance(x, Param) else x
    return f.collect() if isinstance(f, pl.LazyFrame) else f


def _names(x, col: str = "p") -> set[str]:
    """The Utf8 values of column ``col`` of a set frame or a Param (empty when absent)."""
    f = _frame(x)
    if f is None or f.height == 0 or col not in f.columns:
        return set()
    return {str(v) for v in f.get_column(col).cast(pl.Utf8).unique().to_list()}


def _has_rows(x) -> bool:
    f = _frame(x)
    return f is not None and f.height > 0


def _positive_names(param, col: str = "p") -> set[str]:
    f = _frame(param)
    if f is None or f.height == 0:
        return set()
    return {str(v) for v in f.filter(pl.col("value") > 0).get_column(col).cast(pl.Utf8).unique().to_list()}


def _fold_min_load_efficiency(d: "FlexData", procs: set[str]) -> tuple[Param, Param]:
    """``(p_slope, p_section)`` with the section of ``procs`` folded into their slope.

        p_slope'[p,d,t]   = p_slope[p,d,t] + p_section[p,d,t]   p ∈ procs
        p_section'[p,d,t] = 0                                     p ∈ procs

    The fuel of such a unit becomes output × (slope + section): the full-load input per unit of output, so the
    same fuel as the curve at full load and less at part load.  Both already carry a direct unit's
    conversion_flow_coeff fold (slope⁰·k_sink/k_src, section⁰/k_src), and the full-load input of the
    conversion equation is (slope⁰·k_sink + section⁰)/k_src = slope + section, so the fold keeps full load for
    every coefficient.  Rows of other processes are unchanged; row order is kept.
    """
    slope, section = d.p_slope, d.p_section
    for name, prm in (("p_slope", slope), ("p_section", section)):
        if tuple(prm.dims) != ("p", "d", "t"):
            raise ValueError(f"reduce_unit_detail_method online: {name} has dims {prm.dims}, expected (p, d, t).")
    keys = ("p", "d", "t")
    u = [pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in keys]
    uk = [f"__{k}" for k in keys]
    sf, ef = _frame(slope), _frame(section)
    sec = (ef.select(*u, pl.col("value").cast(pl.Float64).alias("__sec"))
           .filter(pl.col("__p").is_in(sorted(procs))))
    dup = sec.group_by(uk).len().filter(pl.col("len") > 1)
    if dup.height > 0:
        raise ValueError("reduce_unit_detail_method online: p_section has duplicate (p, d, t) cells: "
                         f"{dup.select(uk).head(5).rows()}.")
    joined = (sf.with_row_index("__row").with_columns(*u)
              .join(sec, on=uk, how="left").sort("__row"))
    if joined.height != sf.height:
        raise ValueError(f"reduce_unit_detail_method online: folding the section changed the p_slope rows "
                         f"({sf.height} -> {joined.height}).")
    missing = sec.join(joined.select(uk), on=uk, how="anti")
    if missing.height > 0:
        raise ValueError("reduce_unit_detail_method online: min_load_efficiency cells with a section but no "
                         f"slope: {missing.select(uk).head(5).rows()}.")
    new_slope = joined.with_columns(
        pl.when(pl.col("__p").is_in(sorted(procs)))
        .then(pl.col("value") + pl.col("__sec").fill_null(0.0))
        .otherwise(pl.col("value")).cast(sf.schema["value"]).alias("value")).select(sf.columns)
    new_section = ef.with_columns(
        pl.when(pl.col("p").cast(pl.Utf8).is_in(sorted(procs))).then(0.0).otherwise(pl.col("value"))
        .cast(ef.schema["value"]).alias("value"))
    return Param(slope.dims, new_slope, name=slope.name), Param(section.dims, new_section, name=section.name)


def _online_constraints(d: "FlexData") -> list[str]:
    """User constraints of this solve with a ``constraint_online_coeff`` term of a process in the solve."""
    coef = _frame(d.p_process_constraint_online_coeff)
    if coef is None or coef.height == 0:
        return []
    in_solve = _names(d.process_source_sink)
    written = set().union(*(_names(f, "cn") for f in (d.cdt_eq, d.cdt_le, d.cdt_ge)))
    rows = coef.select(pl.col("p").cast(pl.Utf8), pl.col("cn").cast(pl.Utf8)).unique()
    return sorted({cn for p, cn in rows.iter_rows() if p in in_solve and cn in written})


def _drop_constraints(frame: pl.DataFrame | None, cns: list[str]) -> pl.DataFrame | None:
    frame = _frame(frame)
    if frame is None:
        return None
    out = frame.filter(~pl.col("cn").cast(pl.Utf8).is_in(cns))
    return out if out.height > 0 else None


# ---------------------------------------------------------------------------
# the transform


def _reduce_unit(d: "FlexData", parts: frozenset[str], report: ReduceReport) -> dict:
    """The changed fields of the unit parts ``parts`` (field -> new value); fills ``report``."""
    new: dict = {}
    removed = report.removed

    def clear(fields: tuple[str, ...]) -> None:
        for f in fields:
            if getattr(d, f) is not None:
                new[f] = None

    if "integers" in parts:
        integer_online = _names(d.process_online_integer)
        if integer_online and "online" not in parts:
            new["relax_integrality"] = True
        if integer_online:
            removed["integer_online"] = sorted(integer_online)
        exact = _names(d.process_exact)
        if exact:
            removed["exact"] = sorted(exact)
            new["process_exact"] = None
    if "ramps" in parts:
        limits = set().union(*(_names(getattr(d, f"process_source_sink_ramp_limit_{k}"))
                               for k in ("sink_up", "sink_down", "source_up", "source_down")))
        costs = _names(d.process_source_sink_ramp_cost)
        if limits:
            removed["ramp_limits"] = sorted(limits)
        if costs:
            removed["ramp_costs"] = sorted(costs)
        clear(RAMP_FIELDS)
    if "min_limits" in parts:
        min_load = _names(d.process_minload)
        cmin = _positive_names(d.p_process_sink_min_capacity_coef) & min_load
        min_time = _names(d.pdt_uptime_set) | _names(d.pdt_downtime_set)
        if min_load:
            removed["min_load"] = sorted(min_load)
        if cmin:
            removed["capacity_min_coeff"] = sorted(cmin)
        if min_time:
            removed["min_up_down"] = sorted(min_time)
        clear(MIN_LOAD_FIELDS + MIN_TIME_FIELDS)
    online = _names(d.process_online) | _names(d.process_online_linear) | _names(d.process_online_integer)
    online_cns = _online_constraints(d)
    if "online" in parts:
        if online:
            removed["online"] = sorted(online)
        clear(ONLINE_FIELDS)
        min_load_eff = _names(d.process_min_load_eff)
        not_online = sorted(min_load_eff - online)
        if not_online:
            # The method table gives a min_load_efficiency process an online variable; clearing the set without
            # folding would silently drop the section of these processes.
            raise ValueError(f"reduce_unit_detail_method {report.unit}: min_load_efficiency process(es) "
                             f"{', '.join(not_online)} have no online variable; their section cannot be folded.")
        folded = min_load_eff & online
        if folded and (d.p_slope is None or d.p_section is None):
            raise ValueError(f"reduce_unit_detail_method {report.unit}: min_load_efficiency process(es) "
                             f"{', '.join(sorted(folded))} have no p_slope / p_section to fold.")
        if folded:
            new["p_slope"], new["p_section"] = _fold_min_load_efficiency(d, folded)
            removed["min_load_efficiency"] = sorted(folded)
        if d.process_min_load_eff is not None:
            new["process_min_load_eff"] = None
        if online_cns:
            removed["user_constraints"] = list(online_cns)
            for f in ("cdt_eq", "cdt_le", "cdt_ge"):
                new[f] = _drop_constraints(getattr(d, f), online_cns)
            report.warnings.append(
                f"reduce_unit_detail_method = {report.unit}: user constraint(s) {', '.join(online_cns)} left out "
                "of this solve (they count units online, which this level leaves out).")
        if d.p_process_constraint_online_coeff is not None:
            new["p_process_constraint_online_coeff"] = None
    elif "integers" in parts and online_cns and _names(d.process_online_integer):
        report.notes["fractional_online_constraints"] = list(online_cns)
    return new


def _reduce_system(d: "FlexData", parts: frozenset[str], report: ReduceReport) -> dict:
    """The changed fields of the system parts ``parts`` (field -> None); fills ``report.removed``."""
    removed = report.removed
    if "reserves" in parts and _has_rows(d.reserve_upDown_group):
        # The groups with a requirement (a reserve method); without any, the groups of the gate (their providers
        # still have reserve variables, which go).
        frames = [f for f in (d.reserve_upDown_group_method_timeseries, d.reserve_upDown_group_method_dynamic,
                              d.reserve_upDown_group_method_n_1) if _has_rows(f)] or [d.reserve_upDown_group]
        groups = {"__".join(row) for f in frames
                  for row in _frame(f).select(*(pl.col(c).cast(pl.Utf8) for c in ("r", "ud", "g"))).iter_rows()}
        if groups:
            removed["reserves"] = sorted(groups)
    if "inertia" in parts and _names(d.groupInertia, "g"):
        removed["inertia"] = sorted(_names(d.groupInertia, "g"))
    if "non_synchronous" in parts and _names(d.groupNonSync, "g"):
        removed["non_synchronous"] = sorted(_names(d.groupNonSync, "g"))
    if "dc_power_flow" in parts and _has_rows(d.node_dc_power_flow) and _names(d.connection_dc_power_flow):
        removed["dc_power_flow"] = sorted(_names(d.connection_dc_power_flow))
    return {f: None for part in SYSTEM_TOKENS if part in parts for f in SYSTEM_FIELDS[part]
            if getattr(d, f) is not None}


def _interaction_notes(d: "FlexData", uparts: frozenset[str], sparts: frozenset[str],
                       report: ReduceReport) -> None:
    """The interactions of the unit level with the system detail the solve keeps (``report.notes``)."""
    online_gone = set(report.removed.get("online", ()))
    reserve = _frame(d.process_reserve_upDown_node_active)
    if "reserves" not in sparts and _has_rows(d.reserve_upDown_group) and reserve is not None:
        if online_gone:
            hit = sorted(online_gone & _names(reserve))
            if hit:
                report.notes["reserves_from_capacity"] = hit
        min_load_gone = set(report.removed.get("min_load", ()))
        if min_load_gone:
            down = _names(reserve.filter(pl.col("ud").cast(pl.Utf8) == "down"))
            hit = sorted(min_load_gone & down)
            if hit:
                report.notes["downward_reserve_to_zero"] = hit
    if "inertia" not in sparts and online_gone and _names(d.groupInertia, "g") and d.group_node is not None:
        group_node = _frame(d.group_node).select(pl.col("g").cast(pl.Utf8), pl.col("n").cast(pl.Utf8))
        arcs = [f.select(pl.col("p").cast(pl.Utf8), pl.col(side).cast(pl.Utf8).alias("n"))
                for f, side in ((_frame(d.process_sink_inertia), "sink"), (_frame(d.process_source_inertia), "source"))
                if f is not None and f.height > 0]
        if arcs:
            providers = pl.concat(arcs).filter(pl.col("p").is_in(sorted(online_gone)))
            groups = (providers.join(group_node, on="n", how="inner").get_column("g").unique().to_list())
            hit = sorted(set(groups) & _names(d.groupInertia, "g"))
            if hit:
                report.notes["inertia_from_flows"] = hit


def apply_reduce_detail(d: "FlexData", unit: str = "none", system: str = "none") -> tuple["FlexData", ReduceReport]:
    """The reduced data of one sub-solve and what was left out.

    ``unit`` / ``system`` are the resolved values (``SolveConfig.reduce_unit_detail_for`` /
    ``reduce_system_detail_for``).  Returns ``d`` itself when nothing is reduced; otherwise a shallow copy (frames
    not changed are shared, extra attributes such as the axis vocabulary are kept) with the changed fields replaced.
    An unknown value raises :class:`FlexToolConfigError`.
    """
    report = ReduceReport(unit=unit, system=system)
    uparts = unit_parts(unit)
    sparts = system_parts(system)
    if not uparts and not sparts:
        return d, report
    new = _reduce_unit(d, uparts, report) if uparts else {}
    new.update(_reduce_system(d, sparts, report))
    _interaction_notes(d, uparts, sparts, report)
    if not new:
        return d, report
    out = copy.copy(d)
    for f, v in new.items():
        setattr(out, f, v)
    report.fields = frozenset(new)
    return out, report


# ---------------------------------------------------------------------------
# the run-log line


#: How :func:`summary_line` words each key of ``ReduceReport.removed`` (unit keys, then system keys) and of
#: ``ReduceReport.notes``; a ``{}`` takes the entities.  Kept in this order in the line.
_REMOVED_TEXT: dict[str, str] = {
    "online": "online variables and start-up costs left out for {}",
    "integer_online": "integer units online made continuous for {}",
    "exact": "exact connections behave as regular for {}",
    "ramp_limits": "ramp limits left out for {}",
    "ramp_costs": "ramp costs left out for {}",
    "min_load": "min load left out for {}",
    "capacity_min_coeff": "capacity_min_coeff floors left out for {}",
    "min_up_down": "min up / down times left out for {}",
    "min_load_efficiency": "full-load efficiency (no part-load curve) for {}",
    "user_constraints": "{} left out (they count units online)",
    "reserves": "reserves left out for {}",
    "inertia": "inertia limits left out for {}",
    "non_synchronous": "non-synchronous limits left out for {}",
    "dc_power_flow": "DC power flow left out: lossless transport limited by capacity for {}",
}
_UNIT_KEYS = ("online", "integer_online", "exact", "ramp_limits", "ramp_costs", "min_load", "capacity_min_coeff",
              "min_up_down", "min_load_efficiency", "user_constraints")
_SYSTEM_KEYS = ("reserves", "inertia", "non_synchronous", "dc_power_flow")
_NOTE_TEXT: dict[str, str] = {
    "reserves_from_capacity": "the reserves of {} now come from built capacity, not online capacity",
    "inertia_from_flows": "inertia of {} is now counted from flows, not from units online",
    "downward_reserve_to_zero": "the downward reserve of {} now reaches down to zero flow, not to min load",
    "fractional_online_constraints": "{} now count fractional units online",
}
#: Names listed in full up to this count; above it the line gives the count only.
_NAMES_MAX = 5


def _counted(names: list[str], one: str, many: str) -> str:
    n = len(names)
    text = f"{n} {one if n == 1 else many}"
    return f"{text} ({', '.join(names)})" if n <= _NAMES_MAX else text


def _entities(key: str, names: list[str], units: set[str]) -> str:
    """``names`` worded for ``key``: processes as units and connections, the others by their kind."""
    if key == "user_constraints":
        return _counted(names, "user constraint", "user constraints")
    if key == "reserves":
        return _counted(names, "reserve group", "reserve groups")
    if key in ("inertia", "non_synchronous"):
        return _counted(names, "group", "groups")
    if key == "dc_power_flow":
        return _counted(names, "connection", "connections")
    unit = [p for p in names if p in units]
    conn = [p for p in names if p not in units]
    parts = ([_counted(unit, "unit", "units")] if unit else []) + ([_counted(conn, "connection", "connections")]
                                                                   if conn else [])
    return " and ".join(parts)


def _note_entities(key: str, names: list[str], units: set[str]) -> str:
    if key == "inertia_from_flows":
        return _counted(names, "group", "groups")
    if key == "fractional_online_constraints":
        return _counted(names, "user constraint", "user constraints")
    return _entities("online", names, units)


def summary_line(report: ReduceReport, d: "FlexData", solve: str) -> str:
    """The run-log line of one reduced sub-solve: per level what was left out, then the interactions.

    ``d`` is the data before the transform (for telling units from connections).  A level that found nothing to
    reduce says so.  Example: ``Solve invest: reduce_unit_detail_method = online: online variables and start-up
    costs left out for 1 unit (g1); min load left out for 1 unit (g1).  reduce_system_detail_method = reserves:
    reserves left out for 1 reserve group (rsv__up__electricity).``
    """
    units = _names(d.process_unit)
    removed = report.removed

    def level(param: str, value: str, keys: tuple[str, ...]) -> str:
        # Under online* the integer units online go with the online variables: no separate "made continuous".
        skip = {"integer_online"} if removed.get("online") else set()
        parts = [_REMOVED_TEXT[k].format(_entities(k, removed[k], units)) for k in keys
                 if removed.get(k) and k not in skip]
        return f"{param} = {value}: {'; '.join(parts) if parts else 'nothing to reduce'}."

    out = [f"Solve {solve}:"]
    if report.unit != "none":
        out.append(level("reduce_unit_detail_method", report.unit, _UNIT_KEYS))
    if report.system != "none":
        out.append(level("reduce_system_detail_method", report.system, _SYSTEM_KEYS))
    notes = [_NOTE_TEXT[k].format(_note_entities(k, v, units)) for k, v in report.notes.items() if v]
    if notes:
        text = "; ".join(notes)
        out.append(text[0].upper() + text[1:] + ".")
    return " ".join(out)


#: The features whose output cells a reduced solve leaves empty (``process_outputs.reduced_cells``), with the
#: FlexData fields that carry them: the solve has no variables of these features, so their cells are "not
#: modelled", not 0.
OUTPUT_FEATURES: dict[str, tuple[str, ...]] = {
    "online": ONLINE_FIELDS, "reserves": RESERVE_FIELDS, "inertia": INERTIA_FIELDS,
    "non_synchronous": NON_SYNC_FIELDS, "dc_power_flow": DC_FIELDS,
}


def output_features(report: ReduceReport, unreduced: "FlexData") -> list[str]:
    """The features of :data:`OUTPUT_FEATURES` of which the transform cleared a field that had rows in
    ``unreduced`` (the data before the transform), in that order.  Keyed on the cleared fields, not on ``removed``:
    a reserve set without a requirement still gives reserve variables when kept."""
    return [f for f, fields in OUTPUT_FEATURES.items()
            if any(_has_rows(getattr(unreduced, n)) for n in fields if n in report.fields)]


# ---------------------------------------------------------------------------
# across solves of different levels


def cold_start_online_flow_carry(d: "FlexData") -> tuple["FlexData", list[str]]:
    """Drop the carried roll-end flow of the online processes that have no online carry.

    For a solve that keeps online variables and continues from a solve whose ``reduce_unit_detail_method`` left
    them out (``online*``): the producer realised the flows of its ramp arcs but no units online, so this solve's
    online processes get a ramp-carry flow without an online carry, which
    ``params._ramp_links.check_online_ramp_carry_consistency`` refuses (the ramp split would read a false "0 units
    online before").  Their carried flow is dropped instead, so their ramp and commitment start cold, as at the
    start of a solve.  The carry routing itself is unchanged.  Other processes keep their carried flow.  Returns the
    data (``d`` itself when nothing is dropped) and the sorted names of the processes that start cold.
    """
    flow = d.p_roll_continue_flow
    if flow is None:
        return d, []
    online = _names(d.process_online)
    cold = sorted((_names(flow) & online) - _names(d.p_roll_continue_online))
    if not cold:
        return d, []
    out = copy.copy(d)
    for name in ("p_roll_continue_flow", "p_roll_continue_flow_duration"):
        prm = getattr(d, name)
        if prm is None:
            continue
        kept = _frame(prm).filter(~pl.col("p").cast(pl.Utf8).is_in(cold))
        setattr(out, name, Param(prm.dims, kept, name=prm.name) if kept.height > 0 else None)
    return out, cold


def output_static_view(d: "FlexData", unreduced: "FlexData", reduced_everywhere: "frozenset[str]") -> "FlexData":
    """The data the outputs read their static (solve-invariant) sets and parameters from, for a reduced sub-solve.

    The outputs take the solve-keyed sets and parameters of each sub-solve from its own (reduced) data, but the
    static ones (``process_online``, the reserve sets, ``groupInertia``, the DC sets, ...) from the last sub-solve
    only.  A reduced last solve would otherwise erase the tables of earlier full solves.  The view is the
    pre-transform data of the solve, except for the fields every sub-solve of the run so far reduced
    (``reduced_everywhere``): a feature no solve kept has no values to show and keeps no table.
    """
    if d is unreduced:
        return d
    view = copy.copy(unreduced)
    for f in reduced_everywhere:
        setattr(view, f, getattr(d, f))
    return view


def static_data(d: "FlexData") -> "FlexData":
    """The data of :func:`output_static_view` attached to ``d`` by the cascade (``d`` itself when none)."""
    view = getattr(d, "output_static_data", None)
    return d if view is None else view
