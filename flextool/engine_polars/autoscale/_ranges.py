"""Layer 1 (detect): four-range LP coefficient readout.

This module computes the standard HiGHS coefficient-range diagnostic
(Matrix / Cost / Bound / RHS) for a polar-high :class:`Problem` and
decides whether the autoscaler should fire.  Detection only — no LP
modification happens here.  Layer 2 / Layer 3 (later phases) consume the
returned :class:`RangeReport` to pick per-quantity column scalers and a
``user_bound_scale`` value respectively.

Why three entry points
----------------------

* :func:`compute_ranges` — production wire-in.  Reads the four ranges
  from polar-high's own ``Solution.streamed_lp_ranges`` when available
  (no duplicate matrix assembly), and falls back to a passModel-style
  rebuild of the LP arrays from a pre-solve ``Problem`` when needed
  (e.g. when the autoscaler ultimately decides BEFORE solve in later
  phases, or when an alternate engine path doesn't populate
  ``streamed_lp_ranges``).

* :func:`ranges_from_arrays` — low-level kernel.  Accepts the four raw
  coefficient arrays directly.  Useful for tests (build a tiny LP, hand
  the arrays in) and for the post-solve path (extract via highspy if
  ``streamed_lp_ranges`` is absent).

* :func:`ranges_from_streamed` — adapter for the polar-high
  ``Solution.streamed_lp_ranges`` dict (the production hot path).

The same magnitude reduction (``finite & non-zero & |val|``) is used
across all three so the four ranges agree bit-for-bit with what HiGHS
itself prints in its "Coefficient ranges" block — by construction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from ._config import AutoScaleConfig


# Sentinel for "category has zero non-zero finite entries".  Reported as
# ``(nan, nan)`` per the spec; downstream code that wants to treat the
# group as absent should test ``math.isnan(range_tuple[0])``.
_NAN_PAIR: tuple[float, float] = (math.nan, math.nan)


@dataclass(frozen=True)
class RangeReport:
    """Layer 1 output: the four LP coefficient ranges + trigger decision.

    Each range tuple is ``(min |val|, max |val|)`` over **finite, non-zero
    absolute** entries of the corresponding LP component.  Zero-only
    groups return ``(nan, nan)`` and are excluded from the cross-group
    ratio.

    Attributes
    ----------
    matrix:
        Constraint matrix nonzero magnitudes.
    cost:
        Objective coefficient magnitudes.
    bound:
        Variable bound magnitudes (finite ``col_lower`` and ``col_upper``).
    rhs:
        Constraint row bound magnitudes (finite ``row_lower`` and
        ``row_upper``).
    cross_group_max_ratio:
        ``max(hi) / min(lo)`` across the four groups, ignoring any
        group that is ``(nan, nan)``.  ``math.nan`` when every group is
        empty (no LP).
    trigger:
        ``True`` iff any single-group ``hi/lo`` ratio OR the cross-group
        ratio exceeds ``10 ** config.threshold_decades``.
    """

    matrix: tuple[float, float]
    cost: tuple[float, float]
    bound: tuple[float, float]
    rhs: tuple[float, float]
    cross_group_max_ratio: float
    trigger: bool


def _abs_finite_nonzero_min_max(
    arrays: Iterable[Optional[np.ndarray]],
) -> tuple[float, float]:
    """Reduce a sequence of arrays to ``(min |a|, max |a|)`` over their
    finite, non-zero entries.

    Mirrors :func:`polar_high.engine._running_finite_nonzero_min_max` so
    Layer 1's outputs agree with the HiGHS-facing values polar-high
    computes during ``solve()``.  Accepts ``None`` entries (skipped) so
    the caller can pass ``(col_lower, col_upper)`` without filtering.
    Returns ``_NAN_PAIR`` if no array contributes a finite non-zero
    entry.
    """
    lo = math.inf
    hi = 0.0
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue
        mask = np.isfinite(a) & (a != 0)
        if not mask.any():
            continue
        m = np.abs(a[mask])
        lo = min(lo, float(m.min()))
        hi = max(hi, float(m.max()))
    if hi == 0.0:
        return _NAN_PAIR
    return (lo, hi)


def _ratio(span: tuple[float, float]) -> float:
    """Return ``hi / lo`` for a range tuple, or ``0.0`` for ``(nan, nan)``.

    ``0.0`` lets the trigger comparison ``ratio > 10**N`` evaluate to
    ``False`` for empty groups without a special case at the call site.
    """
    lo, hi = span
    if math.isnan(lo) or math.isnan(hi):
        return 0.0
    if lo == 0.0:
        # Defensive: ``_abs_finite_nonzero_min_max`` filters zeros, but if
        # an upstream adapter slipped one through, treat as "no ratio".
        return 0.0
    return hi / lo


def _build_report(
    matrix: tuple[float, float],
    cost: tuple[float, float],
    bound: tuple[float, float],
    rhs: tuple[float, float],
    config: AutoScaleConfig,
) -> RangeReport:
    """Assemble a :class:`RangeReport` from four already-reduced groups.

    Computes the cross-group ratio over non-empty groups only, then ORs
    the four per-group ratios with the cross-group ratio against the
    threshold to set ``trigger``.
    """
    groups = (matrix, cost, bound, rhs)
    non_empty_los = [g[0] for g in groups if not math.isnan(g[0])]
    non_empty_his = [g[1] for g in groups if not math.isnan(g[1])]
    if non_empty_los and non_empty_his:
        cross = max(non_empty_his) / min(non_empty_los)
    else:
        cross = math.nan

    threshold = 10.0 ** float(config.threshold_decades)
    per_group_max = max(_ratio(g) for g in groups)
    cross_for_trigger = 0.0 if math.isnan(cross) else cross
    trigger = bool(
        per_group_max > threshold or cross_for_trigger > threshold
    )

    return RangeReport(
        matrix=matrix,
        cost=cost,
        bound=bound,
        rhs=rhs,
        cross_group_max_ratio=cross,
        trigger=trigger,
    )


def ranges_from_arrays(
    *,
    matrix_values: Optional[np.ndarray],
    cost: Optional[np.ndarray],
    col_lower: Optional[np.ndarray],
    col_upper: Optional[np.ndarray],
    row_lower: Optional[np.ndarray],
    row_upper: Optional[np.ndarray],
    config: AutoScaleConfig,
) -> RangeReport:
    """Compute a :class:`RangeReport` directly from LP arrays.

    Parameters mirror the standard HiGHS/COIN/MPS terminology:

    * ``matrix_values`` — the constraint matrix nonzero coefficient
      array (``a_value_`` in HighsLp).  Pass the raw sparse values; the
      sparse structure (starts / indices) is irrelevant for ranges.
    * ``cost`` — the objective coefficient vector (``col_cost_``).
    * ``col_lower`` / ``col_upper`` — variable bounds, with ``±inf``
      sentinels (or ``±highspy.kHighsInf``) for "unbounded".
    * ``row_lower`` / ``row_upper`` — constraint row bounds, ditto.

    Inf / NaN / zero entries are filtered before the magnitude reduction
    so the four ranges match HiGHS' "Coefficient ranges" block.  Any
    array may be ``None`` (treated as empty) — useful when an LP genuinely
    has no objective term or no row bounds (degenerate but legal).
    """
    matrix = _abs_finite_nonzero_min_max([matrix_values])
    cost_r = _abs_finite_nonzero_min_max([cost])
    bound_r = _abs_finite_nonzero_min_max([col_lower, col_upper])
    rhs_r = _abs_finite_nonzero_min_max([row_lower, row_upper])
    return _build_report(matrix, cost_r, bound_r, rhs_r, config)


def ranges_from_streamed(
    streamed: dict[str, Optional[tuple[float, float]]],
    config: AutoScaleConfig,
) -> RangeReport:
    """Adapt a polar-high ``Solution.streamed_lp_ranges`` dict.

    polar-high already computed the four (min, max) pairs during
    assembly using the same magnitude filter (see
    ``_running_finite_nonzero_min_max`` in polar_high.engine).  Reusing
    its output avoids a duplicate matrix walk in production.  Keys this
    function consumes:

    * ``matrix`` — constraint matrix nonzeros.
    * ``cost`` — objective coefficients.
    * ``col_bound`` — variable bounds (combined ``col_lower`` /
      ``col_upper``).
    * ``row_bound`` — constraint row bounds.

    ``None`` values (polar-high's sentinel for "no finite non-zero
    entries") become ``(nan, nan)`` here, consistent with
    :func:`ranges_from_arrays`.
    """

    def _coerce(key: str) -> tuple[float, float]:
        v = streamed.get(key)
        if v is None:
            return _NAN_PAIR
        lo, hi = v
        return (float(lo), float(hi))

    matrix = _coerce("matrix")
    cost = _coerce("cost")
    bound = _coerce("col_bound")
    rhs = _coerce("row_bound")
    return _build_report(matrix, cost, bound, rhs, config)


def _ranges_via_passmodel(problem: Any, config: AutoScaleConfig) -> RangeReport:
    """Fallback path: assemble LP via polar-high's internal ``_build_lp_arrays``.

    Reached only when the caller hands :func:`compute_ranges` a
    pre-solve ``Problem`` and no ``streamed_lp_ranges`` are available.
    This duplicates the matrix walk polar-high does inside ``solve()``,
    so the production wire-in prefers :func:`ranges_from_streamed` to
    avoid the extra pass.  We document the cost explicitly rather than
    silently using it.
    """
    # Prepare the same n_cols / col_lb / col_ub arrays ``Problem.solve``
    # builds before calling ``_build_lp_arrays`` — see the streaming
    # solve path in polar_high.engine for the canonical sequence.  We
    # rely on the internal API here intentionally; the polar-high
    # repo is in scope for this project and any signature change will
    # show up as a Layer 1 test failure rather than silent drift.
    n_cols = problem._next_col
    col_lb = np.zeros(n_cols, dtype=np.float64)
    col_ub = np.zeros(n_cols, dtype=np.float64)
    for v in problem._vars.values():
        cids = v.frame["col_id"].to_numpy()
        col_lb[cids] = v.lower
        col_ub[cids] = v.upper

    (
        col_lb_h,
        col_ub_h,
        row_lb_h,
        row_ub_h,
        sorted_v,
        _sorted_r,
        _starts,
        _row_names,
        _n_rows,
    ) = problem._build_lp_arrays(
        n_cols=n_cols, col_lb=col_lb, col_ub=col_ub,
    )

    # The objective vector — built inline in ``_solve_passmodel`` /
    # ``_solve_streaming``; mirror that walk here.  We materialise
    # the objective term plans the same way ``solve()`` does.
    col_obj = np.zeros(n_cols, dtype=np.float64)
    import polars as pl  # local import — autoscale must not pull polars
    # at import time for environments that don't need this fallback.
    for t in problem._obj_terms:
        if t.lazy is None:
            continue
        f = t.lazy.collect() if isinstance(t.lazy, pl.LazyFrame) else t.lazy
        if f.height == 0:
            continue
        np.add.at(
            col_obj,
            f["col_id"].to_numpy(),
            f["coef"].to_numpy(),
        )

    # ``kHighsInf`` substitution in ``_build_lp_arrays`` replaces ±inf,
    # so we filter via ``np.isfinite`` and the HiGHS sentinel
    # explicitly.  HiGHS uses 1e30 as the kHighsInf value; treat anything
    # at that magnitude as "unbounded" for range purposes.
    import highspy
    inf_sentinel = float(highspy.kHighsInf)

    def _strip_inf(a: np.ndarray) -> np.ndarray:
        return np.where(np.abs(a) >= inf_sentinel, 0.0, a)

    return ranges_from_arrays(
        matrix_values=sorted_v,
        cost=col_obj,
        col_lower=_strip_inf(col_lb_h),
        col_upper=_strip_inf(col_ub_h),
        row_lower=_strip_inf(row_lb_h),
        row_upper=_strip_inf(row_ub_h),
        config=config,
    )


def compute_ranges(problem_or_solution: Any, config: AutoScaleConfig) -> RangeReport:
    """Compute the Layer 1 four-range report.

    Production callers pass either:

    * a polar-high ``Solution`` (post-solve) — the fast path; we just
      consume ``solution.streamed_lp_ranges``.
    * a polar-high ``Problem`` (pre-solve) — the fallback path; we
      assemble the LP arrays via ``Problem._build_lp_arrays`` and
      reduce them ourselves.  Slower; only used by callers that need
      the ranges before solve dispatch.

    The function dispatches on attribute presence, never on type, so a
    ``LiteSolution`` (commercial-solver wrapper) or a custom mock
    carrying ``streamed_lp_ranges`` works the same.

    Raises ``TypeError`` if neither shape is recognised — silently
    falling back would be the wrong call here; the autoscaler must know
    what kind of input it's been handed.
    """
    streamed = getattr(problem_or_solution, "streamed_lp_ranges", None)
    if isinstance(streamed, dict):
        return ranges_from_streamed(streamed, config)

    # Pre-solve Problem: detect via the streaming solve's private
    # fields.  We never mutate them.
    if (
        hasattr(problem_or_solution, "_vars")
        and hasattr(problem_or_solution, "_cstrs")
        and hasattr(problem_or_solution, "_build_lp_arrays")
    ):
        return _ranges_via_passmodel(problem_or_solution, config)

    raise TypeError(
        "compute_ranges expects a polar-high Problem (pre-solve) or a "
        "Solution carrying streamed_lp_ranges; got "
        f"{type(problem_or_solution).__name__}"
    )


__all__ = [
    "RangeReport",
    "compute_ranges",
    "ranges_from_arrays",
    "ranges_from_streamed",
]
