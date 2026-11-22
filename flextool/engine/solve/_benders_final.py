"""The final solve of a Benders solve (item 287): the whole model solved once with the Benders investments fixed.

After the Benders loop, the undecomposed model of the solve is built as any monolithic solve and every investment,
retirement and life-extension decision is fixed to the Benders incumbent (:func:`pin_invest_decisions`).  That LP
then goes through the normal monolithic path, so the solve writes every result table, the full handoff and the
investment marginal values (the reduced costs of the fixed columns).  ``--benders-final-solve off`` /
``FLEXTOOL_BENDERS_FINAL_SOLVE=off`` skips it, for a model whose undecomposed LP does not fit in memory
(:func:`final_solve_on`).  The final solve's objective lies between the Benders bounds, since the incumbent is
feasible in the fixed-investment LP and the lower bound is valid (:func:`check_final_objective`).

The orchestration that calls these is in :mod:`flextool.engine.solve._orchestration` (``_run_one``); the
warning codes are W-0570..W-0572 (:mod:`flextool.run_warnings._engine_specs`).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Mapping

import numpy as np
import polars as pl

from flextool.engine.input._axis_enums import align_join_dtypes
from flextool.run_options import OPTIONS as RUN_OPTIONS
from flextool.run_options import RunOptions
from flextool.run_options import resolve as resolve_run_option

if TYPE_CHECKING:
    from polar_high import Problem

#: The decision families the final solve fixes: investments, retirements of existing capacity, retirements of
#: model-built cohorts and life extensions, of processes (``_p``) and storage nodes (``_n``).
PINNED_FAMILIES = ("v_invest_p", "v_invest_n", "v_divest_p", "v_divest_n",
                   "v_divest_built_p", "v_divest_built_n", "v_extend_p", "v_extend_n")
#: Families of the Benders result that are sums of the pinned ones (an equality row defines each), so they are not
#: fixed: they follow from the pinned columns.
AGGREGATE_FAMILIES = ("v_divested_built_p", "v_divested_built_n", "v_extended_p", "v_extended_n")

#: Half-width of the band a pinned column gets around its Benders value (:func:`pin_invest_decisions`): relative
#: to the value, absolute below 1.  The Benders master meets its rows only to the solver's feasibility tolerance,
#: so a value can lie that far outside a row bound of the whole model (an investment floor, a group or cumulative
#: limit); with every column of such a row fixed exactly, presolve would call the final solve infeasible.  The band
#: lets the final solve move the value by at most this much to meet the row, and changes the objective by a
#: relative amount of the same order, far inside the Benders gap.
PIN_BAND = 1e-6

#: Relative slack of :func:`check_final_objective`, on top of the solver's own tolerance.
FINAL_OBJECTIVE_REL_SLACK = 1e-6


def final_solve_on(run_options: RunOptions | None = None) -> bool:
    """True when a Benders solve ends with the final solve: ``--benders-final-solve`` >
    ``FLEXTOOL_BENDERS_FINAL_SOLVE`` > ``on``.  *run_options* ``None`` (a library call) reads the environment."""
    if run_options is not None:
        value = run_options.value("benders_final_solve")
    else:
        value = resolve_run_option(RUN_OPTIONS["benders_final_solve"]).value
    return value != "off"


def pin_invest_decisions(pb: "Problem", frames: Mapping[str, pl.DataFrame]) -> None:
    """Fix every investment / retirement / extension column of *pb* to its value in *frames*.

    *pb* is the undecomposed model of the solve, built (``build_flextool``) and not yet scaled (call this before
    Layer 2: the values are in the model's own units, as ``Solution.value`` gives them).  *frames* is
    ``BendersResult.invest_solution_vars``: per family, the Var's dims then ``value``.  For each family of
    :data:`PINNED_FAMILIES` that *pb* has, the value is first clamped into the column's own bounds (a Benders value
    a solver tolerance outside them would make the LP infeasible), then the column is bounded to
    ``value ± PIN_BAND · max(1, |value|)`` (:data:`PIN_BAND`; within the column's bounds): a value a solver
    tolerance outside a row bound of the whole model (a floor, a group or cumulative limit, all of whose columns
    are pinned) can then still meet the row.  A column of an integer family is fixed at the nearest whole number.
    The aggregates (:data:`AGGREGATE_FAMILIES`) are left free.

    Raises :class:`RuntimeError` (an internal error, not a configuration error) when a column of a pinned family
    has no value, when *frames* has a value for no column of *pb* (also for a family whose variable *pb* has with
    no column) or two values for one column, or when *frames* holds a family that *pb* does not have or that is no
    decision family at all.
    """
    for name in frames:
        if name in AGGREGATE_FAMILIES:
            continue
        if name not in PINNED_FAMILIES:
            raise RuntimeError(f"Benders final solve: {name!r} is not an investment decision family")
        if name not in pb._vars:
            raise RuntimeError(f"Benders final solve: the Benders solve has values for {name!r}, which the "
                               f"undecomposed model does not have")
    for name in PINNED_FAMILIES:
        var = pb._vars.get(name)
        frame = frames.get(name)
        if var is None or var.frame.height == 0:
            if var is not None and frame is not None and frame.height > 0:
                raise RuntimeError(f"Benders final solve: the Benders solve has {frame.height} values for {name}, "
                                   f"which has no column in the undecomposed model")
            continue
        if frame is None:
            raise RuntimeError(f"Benders final solve: no Benders value for {name} of "
                               f"{_key_text(var.frame.row(0), var.dims)}")
        values = _values_by_column(var, frame)
        lower = np.broadcast_to(np.asarray(var.col_lower(), dtype=np.float64), values.shape)
        upper = np.broadcast_to(np.asarray(var.col_upper(), dtype=np.float64), values.shape)
        values = np.minimum(np.maximum(values, lower), upper)
        if var.integer:
            values = np.round(values)
            var.lower = values
            var.upper = values.copy()
            continue
        band = PIN_BAND * np.maximum(1.0, np.abs(values))
        var.lower = np.maximum(values - band, lower)
        var.upper = np.minimum(values + band, upper)
    # The canonical matrix (built lazily by ``Problem.canonicalise`` for ``write_mps``) reads the bounds when it is
    # built; one built before this call would hold the old bounds.
    pb._canonical_dirty = True


def _values_by_column(var, frame: pl.DataFrame) -> np.ndarray:
    """The ``value`` of *frame* for each row of ``var.frame``, positionally aligned with it."""
    dims = list(var.dims)
    missing = [d for d in (*dims, "value") if d not in frame.columns]
    if missing:
        raise RuntimeError(f"Benders final solve: the Benders values of {var.name} have no column(s) {missing}")
    left = var.frame.select(dims).with_row_index("__row")
    original = frame.select(dims)
    right = frame.select(*dims, "value").with_row_index("__src")
    left, right = align_join_dtypes(left, right, dims)
    if right.select(dims).is_duplicated().any():
        dup = right.filter(right.select(dims).is_duplicated()).row(0)
        raise RuntimeError(f"Benders final solve: two Benders values for {var.name} of {_key_text(dup, dims)}")
    joined = left.join(right.drop("__src"), on=dims, how="left", nulls_equal=False).sort("__row")
    if joined.height != left.height:  # cannot happen with unique keys; a guard for the positional alignment
        raise RuntimeError(f"Benders final solve: the Benders values of {var.name} do not align with its columns")
    no_value = joined.filter(pl.col("value").is_null() | pl.col("value").is_nan())
    if no_value.height:
        raise RuntimeError(f"Benders final solve: no Benders value for {var.name} of "
                           f"{_key_text(no_value.select(dims).row(0), dims)} ({no_value.height} columns)")
    unmatched = right.join(left, on=dims, how="anti", nulls_equal=False)
    if unmatched.height:
        # The original key: one outside the model's Enum vocabulary is null after the alignment.
        key = original.row(int(unmatched["__src"][0]))
        raise RuntimeError(f"Benders final solve: a Benders value for {var.name} of "
                           f"{_key_text(key, dims)}, which the undecomposed model does "
                           f"not have ({unmatched.height} values)")
    return joined["value"].to_numpy().astype(np.float64, copy=True)


def _key_text(row: tuple, dims) -> str:
    """``'wind_A', d 'y2030'``: the entity of a column and its other dims."""
    parts = [f"'{row[0]}'"] + [f"{dim} '{value}'" for dim, value in zip(list(dims)[1:], row[1:])]
    return ", ".join(parts)


def check_final_objective(objective: float, lower_bound: float, upper_bound: float, *,
                          solver_tolerance: float = 0.0) -> bool:
    """True when the final solve's *objective* lies in the Benders bounds ``[lower_bound, upper_bound]``.

    The slack on each side is ``(FINAL_OBJECTIVE_REL_SLACK + solver_tolerance)`` times the largest of the three
    magnitudes (the final solve and the Benders solves stop at their own tolerances).  A bound that is not finite
    (no lower bound yet) is not checked.  A non-finite *objective* is outside.
    """
    if not math.isfinite(objective):
        return False
    scale = max(abs(objective), *(abs(b) for b in (lower_bound, upper_bound) if math.isfinite(b)))
    slack = (FINAL_OBJECTIVE_REL_SLACK + max(0.0, float(solver_tolerance))) * scale
    if math.isfinite(lower_bound) and objective < lower_bound - slack:
        return False
    if math.isfinite(upper_bound) and objective > upper_bound + slack:
        return False
    return True


__all__ = [
    "AGGREGATE_FAMILIES",
    "FINAL_OBJECTIVE_REL_SLACK",
    "PINNED_FAMILIES",
    "PIN_BAND",
    "check_final_objective",
    "final_solve_on",
    "pin_invest_decisions",
]
