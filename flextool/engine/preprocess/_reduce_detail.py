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

System detail (``reduce_system_detail_method``) is parsed and validated (:func:`system_parts`) and recorded in the
report; this module changes no system field.
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
assert tuple(UNIT_PARTS) == REDUCE_UNIT_DETAIL_METHODS

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


@dataclasses.dataclass
class ReduceReport:
    """What :func:`apply_reduce_detail` left out of one sub-solve.

    ``removed``: feature -> sorted names of the entities (processes, or user constraints for
    ``user_constraints``) it was left out for; only features with something left out appear.  Keys: ``integer_online``
    (binary units / connections now continuous), ``exact`` (exact connections now regular), ``ramp_limits``,
    ``ramp_costs``, ``min_load``, ``capacity_min_coeff``, ``min_up_down``, ``online``, ``min_load_efficiency``
    (folded to full-load efficiency), ``user_constraints`` (left out whole).  ``notes``: interaction facts for the
    run log (``fractional_online_constraints``: user constraints that now count fractional units online).
    ``warnings``: the lines to log as warnings (a user constraint left out).
    """

    unit: str = "none"
    system: str = "none"
    removed: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    notes: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    warnings: list[str] = dataclasses.field(default_factory=list)

    @property
    def changed(self) -> bool:
        """True when the data was reduced (something was left out or relaxed)."""
        return bool(self.removed)


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
    joined = (sf.with_row_index("__row").with_columns(*u)
              .join(sec, on=uk, how="left").sort("__row"))
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
        folded = _names(d.process_min_load_eff) & online
        if folded and d.p_slope is not None and d.p_section is not None:
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


def apply_reduce_detail(d: "FlexData", unit: str = "none", system: str = "none") -> tuple["FlexData", ReduceReport]:
    """The reduced data of one sub-solve and what was left out.

    ``unit`` / ``system`` are the resolved values (``SolveConfig.reduce_unit_detail_for`` /
    ``reduce_system_detail_for``).  Returns ``d`` itself when nothing is reduced; otherwise a shallow copy (frames
    not changed are shared, extra attributes such as the axis vocabulary are kept) with the changed fields replaced.
    An unknown value raises :class:`FlexToolConfigError`.
    """
    report = ReduceReport(unit=unit, system=system)
    uparts = unit_parts(unit)
    system_parts(system)
    if not uparts:
        return d, report
    new = _reduce_unit(d, uparts, report)
    if not new:
        return d, report
    out = copy.copy(d)
    for f, v in new.items():
        setattr(out, f, v)
    return out, report
