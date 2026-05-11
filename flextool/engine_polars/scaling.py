"""Per-solve scaling analyzer — polars / in-memory port of
:mod:`flextool.flextoolrunner.scaling`.

This module implements the same lightweight numerical diagnostic as the
original CSV-based analyzer, but consumes an in-memory
:class:`~flextool.engine_polars.input.FlexData` bag instead of reading
``*.csv`` files from disk.

The public API is :func:`analyze_solve`.  Everything else mirrors the
original module (same constants, same :class:`ScaleTable`,
:class:`FamilyStats`, same JSON helpers, same module-level
:data:`_scale_cache`).

Excluded from this port
-----------------------
* ``compute_bound_stats`` / ``decide_user_bound_scale`` /
  ``apply_bound_scale_decision`` / ``update_bound_scale_in_cache`` —
  those require a loaded HiGHS LP object (Agent-18c), not input data.
  They live in the original module and are imported from there where
  needed.
* CSV-reading helpers (``_iter_numeric_cells``, ``_scan_family``,
  ``_read_entity_unitsizes*``, ``_derive_entity_unitsizes*``,
  ``_sum_cost_params``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from flextool.engine_polars.input import FlexData
    from polar_high import Param


# ---------------------------------------------------------------------------
# Re-export constants & helpers that callers may import from this module too
# ---------------------------------------------------------------------------

AUTO_SCALE_ENV_VAR = "FLEXTOOL_AUTO_SCALE"
"""Environment-variable fallback for the ``--auto-scale`` CLI flag."""

DEFAULT_OBJECTIVE_SCALE = 1e-6
OBJECTIVE_SCALE_MIN = 1e-12
OBJECTIVE_SCALE_MAX = 1e0

UNITSIZE_SPREAD_THRESHOLD = 3.0
RHS_SPREAD_THRESHOLD = 6.0
COST_SPREAD_THRESHOLD = 5.0

RHS_FAMILIES: tuple[str, ...] = ("node_inflow", "node_annual_flow")
COST_FAMILIES: tuple[str, ...] = (
    "vom_and_op_costs",
    "capex_invest",
    "node_penalty",
)


# ---------------------------------------------------------------------------
# CLI / env-var resolution
# ---------------------------------------------------------------------------


def resolve_auto_scale(cli_flag: bool) -> bool:
    """True iff the CLI flag is set OR ``FLEXTOOL_AUTO_SCALE`` is truthy."""
    if cli_flag:
        return True
    raw = os.environ.get(AUTO_SCALE_ENV_VAR, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Data classes  (identical to the original; kept here so importers only need
# one import path)
# ---------------------------------------------------------------------------


@dataclass
class FamilyStats:
    """Summary statistics for one parameter family."""

    n_values: int
    n_zero: int
    n_nonzero: int
    log10_min: Optional[float] = None
    log10_max: Optional[float] = None
    log10_median: Optional[float] = None
    log10_p10: Optional[float] = None
    log10_p90: Optional[float] = None
    abs_min: Optional[float] = None
    abs_max: Optional[float] = None
    abs_median: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScaleTable:
    """Analyzer output — one per solve."""

    solve_name: str
    use_row_scaling: Literal["yes", "no"]
    scale_the_objective: float
    family_ranges: dict[str, FamilyStats]
    unitsize_spread_log10: float
    rough_obj_estimate: float
    timestamp: str
    source_dir: str
    scale_the_state: float = 1.0
    rhs_spread_log10: float = 0.0
    cost_spread_log10: float = 0.0
    row_scaling_trigger: Literal["unitsize", "rhs", "cost", "none"] = "none"
    bound_spread_log10: float = 0.0
    user_bound_scale: int = 0
    bound_abs_min: Optional[float] = None
    bound_abs_max: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


_scale_cache: dict[str, ScaleTable] = {}
"""Per-solve-name cache.  Rolling windows reuse the same entry."""


def clear_cache() -> None:
    """Reset the cache (primarily for tests)."""
    _scale_cache.clear()


# ---------------------------------------------------------------------------
# Core polars helper
# ---------------------------------------------------------------------------


def _extract_param_values(param: "Param | None") -> np.ndarray:
    """Extract all non-null, non-zero numeric values from a polars Param.

    Returns a float64 numpy array (empty when *param* is ``None`` or its
    frame is empty).
    """
    if param is None:
        return np.array([], dtype=np.float64)
    # Param.frame is a cached eager DataFrame property.
    try:
        frame = param.frame
    except Exception:
        return np.array([], dtype=np.float64)
    if frame is None or frame.is_empty():
        return np.array([], dtype=np.float64)
    if "value" not in frame.columns:
        return np.array([], dtype=np.float64)
    return (
        frame
        .select(pl.col("value"))
        .drop_nulls()
        .filter(pl.col("value") != 0.0)
        .to_series()
        .to_numpy()
    )


# ---------------------------------------------------------------------------
# FamilyStats computation  (pure math — stdlib only, same as original)
# ---------------------------------------------------------------------------


def _family_stats(values: list[float]) -> FamilyStats:
    """Compute a :class:`FamilyStats` summary from a list of floats."""
    n = len(values)
    if n == 0:
        return FamilyStats(n_values=0, n_zero=0, n_nonzero=0)
    zeros = sum(1 for v in values if v == 0.0)
    nonzero = [v for v in values if v != 0.0]
    if not nonzero:
        return FamilyStats(n_values=n, n_zero=zeros, n_nonzero=0)
    abs_vals = sorted(abs(v) for v in nonzero)
    log10_vals = sorted(math.log10(v) for v in abs_vals)

    def _pct(sorted_list: list[float], q: float) -> float:
        if not sorted_list:
            return math.nan
        if len(sorted_list) == 1:
            return sorted_list[0]
        k = q * (len(sorted_list) - 1)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return sorted_list[lo]
        return sorted_list[lo] + (sorted_list[hi] - sorted_list[lo]) * (k - lo)

    return FamilyStats(
        n_values=n,
        n_zero=zeros,
        n_nonzero=len(nonzero),
        log10_min=log10_vals[0],
        log10_max=log10_vals[-1],
        log10_median=_pct(log10_vals, 0.5),
        log10_p10=_pct(log10_vals, 0.10),
        log10_p90=_pct(log10_vals, 0.90),
        abs_min=abs_vals[0],
        abs_max=abs_vals[-1],
        abs_median=_pct(abs_vals, 0.5),
    )


def _family_stats_from_array(arr: np.ndarray) -> FamilyStats:
    """Variant that accepts a numpy float64 array for efficiency."""
    return _family_stats(arr.tolist())


# ---------------------------------------------------------------------------
# Pooled spread helper  (identical to original)
# ---------------------------------------------------------------------------


def _pooled_spread_log10(
    family_stats: dict[str, FamilyStats],
    families: tuple[str, ...] | list[str],
) -> float:
    """Return the pooled log10 spread across the named families."""
    abs_mins: list[float] = []
    abs_maxs: list[float] = []
    for name in families:
        stats = family_stats.get(name)
        if stats is None:
            continue
        if stats.abs_min is not None and stats.abs_min > 0.0:
            abs_mins.append(stats.abs_min)
        if stats.abs_max is not None and stats.abs_max > 0.0:
            abs_maxs.append(stats.abs_max)
    if not abs_mins or not abs_maxs:
        return 0.0
    overall_min = min(abs_mins)
    overall_max = max(abs_maxs)
    if overall_min <= 0.0 or overall_max <= 0.0:
        return 0.0
    try:
        return math.log10(overall_max) - math.log10(overall_min)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Rough objective estimate  (in-memory variant)
# ---------------------------------------------------------------------------


def _estimate_rough_obj_inmemory(
    family_arrays: dict[str, np.ndarray],
    flex_data: "FlexData",
) -> float:
    """Back-of-envelope total-cost magnitude from FlexData Params.

    Mirrors :func:`flextool.flextoolrunner.scaling._estimate_rough_obj`
    but pulls VOM / CAPEX sums directly from the in-memory Params instead
    of re-parsing CSV files.

    Formula::

        rough_obj ≈ (sum |VOM|) × typical_flow × n_timesteps
                  + (sum |CAPEX|) × typical_capacity
    """
    inflow_arr = family_arrays.get("node_inflow", np.array([], dtype=np.float64))
    unitsize_arr = family_arrays.get("entity_unitsize", np.array([], dtype=np.float64))

    # --- VOM: variable cost params + commodity price -----------------------
    vom_arr = np.concatenate([
        _extract_param_values(flex_data.p_pssdt_varCost),
        _extract_param_values(flex_data.p_pdt_varCost_source),
        _extract_param_values(flex_data.p_pdt_varCost_sink),
        _extract_param_values(flex_data.p_pdt_varCost_process),
        _extract_param_values(flex_data.p_startup_cost),
        _extract_param_values(flex_data.p_commodity_price),
    ])

    # --- CAPEX: lifetime fixed cost (invest / fixed cost annualised) -------
    capex_arr = _extract_param_values(flex_data.ed_lifetime_fixed_cost)

    sum_vom = float(np.sum(np.abs(vom_arr))) if vom_arr.size else 0.0
    sum_capex = float(np.sum(np.abs(capex_arr))) if capex_arr.size else 0.0

    # Typical flow: median of non-zero absolute inflows.
    nonzero_inflows = np.abs(inflow_arr[inflow_arr != 0.0]) if inflow_arr.size else np.array([])
    typical_flow = 0.0
    timesteps_like = 0
    if nonzero_inflows.size:
        nonzero_inflows.sort()
        typical_flow = float(nonzero_inflows[len(nonzero_inflows) // 2])
        timesteps_like = int(nonzero_inflows.size)

    # Typical capacity: median of non-zero absolute unitsizes.
    nonzero_unitsizes = np.abs(unitsize_arr[unitsize_arr != 0.0]) if unitsize_arr.size else np.array([])
    typical_cap = 0.0
    if nonzero_unitsizes.size:
        nonzero_unitsizes.sort()
        typical_cap = float(nonzero_unitsizes[len(nonzero_unitsizes) // 2])

    operational = sum_vom * typical_flow * max(1, timesteps_like)
    investment = sum_capex * typical_cap
    return operational + investment


# ---------------------------------------------------------------------------
# LP-level invest-cost coefficient upper bound (for the cost-floor guard)
# ---------------------------------------------------------------------------


def _invest_cost_lp_coef_upper_bound(flex_data: "FlexData") -> Optional[float]:
    """Upper-bound estimate for the LP cost coefficient on v_invest cols.

    The objective contribution for invest variables (model.py §8.0) is
    ``Sum(v_invest * p_unitsize * (ed_entity_annual_discounted +
    ed_lifetime_fixed_cost))`` so the LP coefficient on a v_invest
    column is bounded above by ``p_unitsize[e] × max(annu[e,d] +
    lf[e,d])``.

    Returns ``max(p_unitsize) × max(annu + lf)`` — a conservative upper
    bound that doesn't require joining the parameters on entity.  This
    overestimates when the entity with max unitsize is not the entity
    with max capex (the typical case), but the cost-floor guard only
    needs an upper bound, not the exact value.

    Returns ``None`` when neither invest-cost param is available.
    """
    annu_arr = _extract_param_values(flex_data.ed_entity_annual_discounted)
    lf_arr = _extract_param_values(flex_data.ed_lifetime_fixed_cost)
    if annu_arr.size == 0 and lf_arr.size == 0:
        return None
    annu_max = float(np.abs(annu_arr).max()) if annu_arr.size > 0 else 0.0
    lf_max = float(np.abs(lf_arr).max()) if lf_arr.size > 0 else 0.0
    # Conservative: ``max(annu) + max(lf)`` rather than ``max(annu + lf)``
    # — would require an aligned join; the sum bound is tighter than
    # either alone and adequate for floor-guard purposes.
    param_upper = annu_max + lf_max
    unitsize_arr = _extract_param_values(flex_data.p_all_entity_unitsize)
    unitsize_max = (
        float(np.abs(unitsize_arr).max()) if unitsize_arr.size > 0 else 1.0
    )
    return unitsize_max * param_upper


# ---------------------------------------------------------------------------
# Objective scalar recommendation
# ---------------------------------------------------------------------------


# Two-sided cost-band guard thresholds.  HiGHS' "Coefficient ranges"
# diagnostic warns on cost coefficients below ~1e-7 ("excessively
# small costs") OR above ~1e+5 ("excessively large costs").  The
# analyzer's cost_abs_min/max values are pre-row-multiplier
# (analyzer-side) PROXIES — per-row multipliers (RP weights,
# discount factors, period_share, years_represented) further
# amplify both ends of the range BUT by approximately the same
# factor at both ends, so the analyzer's log10-spread tracks the
# LP's log10-spread closely.  Setting the guards to HiGHS' literal
# thresholds keeps the analyzer's pre-multiplier band centered
# inside HiGHS' tolerable zone.
LP_COST_AFTER_SCALE_MIN_FLOOR = 1e-7
LP_COST_AFTER_SCALE_MAX_CEILING = 1e+5

# Legacy alias kept for downstream callers.  The new two-sided
# logic uses LP_COST_AFTER_SCALE_*_FLOOR/CEILING; this constant
# remains for ``_recommend_scale_the_objective`` callers that
# still pass only ``cost_abs_max``.
COST_AFTER_SCALE_MIN_FLOOR = 1e-3


def _recommend_scale_the_objective(
    rough_obj: float,
    cost_abs_max: Optional[float] = None,
    cost_abs_min: Optional[float] = None,
) -> float:
    """Recommend ``scale_the_objective`` from the rough objective magnitude.

    Three-step rule:

    1. Pick a power-of-10 scalar that brings the rough total cost close
       to ``O(1)`` — i.e. ``scale = 10**-round(log10(rough_obj))``.

    2. **Two-sided cost-band guard**.  When BOTH ``cost_abs_min`` and
       ``cost_abs_max`` are provided, constrain ``scale`` so the
       largest cost coefficient stays below
       :data:`LP_COST_AFTER_SCALE_MAX_CEILING` AND the smallest cost
       coefficient stays above :data:`LP_COST_AFTER_SCALE_MIN_FLOOR`:

           cost_abs_min × scale  ≥  LP_COST_AFTER_SCALE_MIN_FLOOR
           cost_abs_max × scale  ≤  LP_COST_AFTER_SCALE_MAX_CEILING

       If the cost log10-spread fits inside HiGHS' band, BOTH
       constraints are satisfiable; we clamp ``scale`` to
       ``[scale_lo, scale_hi]``.

    3. **Over-wide-spread fallback**.  If the cost spread *exceeds*
       HiGHS' usable band (i.e. ``scale_lo > scale_hi``), NO scale
       can satisfy both ends.  We use **geometric centering**: place
       the geomean of the cost range at the geomean of HiGHS' band,
       distributing the (unavoidable) violation symmetrically on both
       sides:

           scale  =  sqrt(MIN_FLOOR × MAX_CEILING)
                  /  sqrt(cost_abs_min × cost_abs_max)

       This minimises the symmetric (log-scale) HiGHS-floor /
       ceiling violation rather than blowing out one end entirely.

    4. **Legacy single-sided fallback**.  When only ``cost_abs_max``
       is provided (no ``cost_abs_min``), fall back to the original
       single-sided guard against tiny scaled costs:
       ``cost_abs_max × scale ≥ COST_AFTER_SCALE_MIN_FLOOR``.

    Parameters
    ----------
    rough_obj
        Back-of-envelope total objective magnitude (positive, finite).
    cost_abs_max
        Largest absolute cost coefficient in the input (across all cost
        families).  When ``None``, both guards are skipped.
    cost_abs_min
        Smallest absolute cost coefficient in the input (across all
        cost families).  When ``None``, falls back to single-sided
        ``cost_abs_max`` guard.

    Returns
    -------
    float
        A scalar in ``[OBJECTIVE_SCALE_MIN, OBJECTIVE_SCALE_MAX]``.
        NOTE: when the two-sided guard binds (or the centering
        fallback fires), the returned value is NOT restricted to a
        power of 10 — the goal is to hit HiGHS' usable band as
        closely as possible.
    """
    if not math.isfinite(rough_obj) or rough_obj <= 0.0:
        return DEFAULT_OBJECTIVE_SCALE
    try:
        lg = math.log10(rough_obj)
    except ValueError:
        return DEFAULT_OBJECTIVE_SCALE
    scale = 10.0 ** -round(lg)

    have_max = (
        cost_abs_max is not None
        and math.isfinite(cost_abs_max)
        and cost_abs_max > 0.0
    )
    have_min = (
        cost_abs_min is not None
        and math.isfinite(cost_abs_min)
        and cost_abs_min > 0.0
    )

    if have_max and have_min:
        # Two-sided band guard.
        scale_lo = LP_COST_AFTER_SCALE_MIN_FLOOR / cost_abs_min
        scale_hi = LP_COST_AFTER_SCALE_MAX_CEILING / cost_abs_max
        if scale_lo <= scale_hi:
            # Cost spread fits inside HiGHS' band.  Clamp scale to
            # ``[scale_lo, scale_hi]`` — keep the rough-obj
            # recommendation when it's already inside the window.
            if scale < scale_lo:
                scale = scale_lo
            elif scale > scale_hi:
                scale = scale_hi
        else:
            # Cost spread exceeds HiGHS' band — geometric centering
            # distributes the violation symmetrically.
            geomean_cost = math.sqrt(cost_abs_min * cost_abs_max)
            geomean_band = math.sqrt(
                LP_COST_AFTER_SCALE_MIN_FLOOR * LP_COST_AFTER_SCALE_MAX_CEILING
            )
            scale = geomean_band / geomean_cost
    elif have_max:
        # Legacy single-sided guard (kept for backwards compat with
        # callers that don't yet supply cost_abs_min).
        scale_floor_for_costs = COST_AFTER_SCALE_MIN_FLOOR / cost_abs_max
        if scale < scale_floor_for_costs:
            scale = scale_floor_for_costs

    if scale < OBJECTIVE_SCALE_MIN:
        scale = OBJECTIVE_SCALE_MIN
    if scale > OBJECTIVE_SCALE_MAX:
        scale = OBJECTIVE_SCALE_MAX
    return scale


# ---------------------------------------------------------------------------
# HiGHS user-scaling option recommendations  (post-build heuristics)
# ---------------------------------------------------------------------------


# user_bound_scale clamp.  HiGHS interprets ``user_bound_scale`` as a
# **power of 2** (bounds × 2^N), so the magnitude per step is ~3×, not
# 10×.  HiGHS itself rejects values outside [-30, 30]; -30 (= × 2^-30
# ≈ 1e-9) is the practical floor for very wide row-bound spreads
# (e.g. cumulative-ladder caps at 1e+8 paired with annual fractions
# at 1e-6).  HiGHS' "Consider setting the user_bound_scale option
# to <N>" warning text was the source of the older |N| ≤ 10
# guidance — that was about the per-warning increment, not the
# safe operating range.
USER_BOUND_SCALE_MIN = -30
USER_BOUND_SCALE_MAX = 0

# Threshold (decades) above which we apply ``user_bound_scale``.
# Below this the LP bounds are tight enough that HiGHS' own scaling is
# sufficient.
USER_BOUND_SCALE_TRIGGER_DECADES = 6.0


def _max_input_bound_proxy(family_stats: dict[str, FamilyStats]) -> Optional[float]:
    """Estimate the largest absolute bound the LP will see, from input stats.

    The LP's column bounds come mostly from ``entity_unitsize`` (× max
    units) and the row bounds come mostly from inflows / annual flows.
    We can't know the actual LP RHS without inspecting the built LP,
    but the *largest* parameter across these families gives a reasonable
    lower bound on the expected LP coefficient magnitude.

    Returns ``None`` when no usable family stats exist.
    """
    candidates: list[float] = []
    for name in ("entity_unitsize", "node_inflow", "node_annual_flow"):
        stats = family_stats.get(name)
        if stats is None:
            continue
        if stats.abs_max is not None and stats.abs_max > 0.0:
            candidates.append(stats.abs_max)
    if not candidates:
        return None
    return max(candidates)


def recommend_user_bound_scale(
    family_stats: dict[str, FamilyStats],
    rough_obj: float,
) -> int:
    """Heuristic ``user_bound_scale`` based on input parameter ranges.

    HiGHS' ``user_bound_scale`` option scales every column / row bound
    by ``2**N`` *during LP loading* (so it composes with our
    ``scale_the_objective``).  A negative ``N`` shrinks bounds toward 1,
    which helps when a model has large physical capacities or annual
    flows.

    Heuristic:

    * If ``rough_obj`` is enormous (≥ 1e+12) we expect the LP RHS to be
      similarly large (energy balances aggregate inflows × duration).
      Pick ``N`` so ``2**N × bound_proxy ≈ 1`` — i.e.
      ``N = -round(log2(bound_proxy))`` — clamped to
      ``[USER_BOUND_SCALE_MIN, 0]``.
    * Otherwise return ``0`` (let HiGHS' own scaling handle it).

    This is intentionally a *coarse* heuristic — proper bound-stat
    analysis requires the built LP, which the polars engine path does
    not currently expose.  When the input data underestimates LP
    coefficient magnitudes (common with annual aggregations), HiGHS'
    own warning will still suggest a more aggressive value.
    """
    if not math.isfinite(rough_obj) or rough_obj < 1e12:
        return 0  # below trigger — leave HiGHS alone
    bound_proxy = _max_input_bound_proxy(family_stats)
    if bound_proxy is None:
        return 0
    try:
        n = -int(round(math.log2(bound_proxy)))
    except ValueError:
        return 0
    if n > USER_BOUND_SCALE_MAX:
        n = USER_BOUND_SCALE_MAX
    if n < USER_BOUND_SCALE_MIN:
        n = USER_BOUND_SCALE_MIN
    return n


def recommend_user_bound_scale_from_lp(
    lp_ranges: dict,
) -> int:
    """Like :func:`recommend_user_bound_scale`, but uses actual built-LP
    ranges instead of input-data heuristics.

    The dict comes from :meth:`polar_high.Problem.peek_lp_ranges`; it
    has the form ``{'matrix': (lo, hi)|None, 'cost': ..., 'col_bound':
    ..., 'row_bound': ...}``.  We look at the larger ``abs_max`` of the
    row and column bound ranges — that's what HiGHS itself flags when
    it prints "user-scaled problem has some excessively large row
    bounds" — and choose ``N`` so ``2**N`` brings that max close to
    ``1.0``.  HiGHS' ``user_bound_scale`` is a power of 2, NOT a power
    of 10 — using ``log10`` here was a longstanding bug that left the
    recommended scaling ~3.3× too gentle per "decade".

    Returns an integer in
    ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``.  ``0`` means
    "leave HiGHS' own scaling alone".
    """
    if not lp_ranges:
        return 0
    bounds: list[float] = []
    for key in ("row_bound", "col_bound"):
        rng = lp_ranges.get(key)
        if rng is None:
            continue
        try:
            lo, hi = rng
        except (TypeError, ValueError):
            continue
        if math.isfinite(hi) and hi > 0.0:
            bounds.append(float(hi))
    if not bounds:
        return 0
    max_bound = max(bounds)
    # Trigger: ~2^10 = 1024 — matches the "max_bound > 1e3" heuristic
    # used pre-fix, kept as 2^10 to make the threshold consistent with
    # the new log2 scaling.
    if max_bound <= 1024.0:
        return 0
    try:
        n = -int(round(math.log2(max_bound)))
    except ValueError:
        return 0
    if n > USER_BOUND_SCALE_MAX:
        n = USER_BOUND_SCALE_MAX
    if n < USER_BOUND_SCALE_MIN:
        n = USER_BOUND_SCALE_MIN
    return n


def recommended_highs_options(
    table: ScaleTable,
    *,
    apply_user_bound_scale: bool = True,
    user_bound_scale_override: Optional[int] = None,
    lp_ranges: dict | None = None,
) -> dict[str, int]:
    """Build the HiGHS solver-options dict from a :class:`ScaleTable`.

    Currently sets:

    * ``simplex_scale_strategy = SIMPLEX_SCALE_STRATEGY_ADVANCED`` —
      always.  HiGHS' default (1) is a basic equilibration; (2) adds
      Curtis–Reid which costs negligibly more but handles wide
      coefficient spreads much better.
    * ``user_bound_scale`` — when set.  Resolution order:
        1. If ``user_bound_scale_override`` is provided (typically the
           value HiGHS itself recommends in its scaling warning), use it.
        2. Otherwise, when ``apply_user_bound_scale`` is True, fall back
           to :func:`recommend_user_bound_scale` (input-data heuristic).
        3. ``0`` is omitted from the returned dict (HiGHS no-op).

    Note: ``user_cost_scale`` is intentionally NOT set — we already
    scale costs via ``scale_the_objective`` inside the LP build, and
    layering two cost-scale shifts compounds confusingly without a
    proven benefit.

    Parameters
    ----------
    table:
        Per-solve :class:`ScaleTable` from :func:`analyze_solve`.
    apply_user_bound_scale:
        When False, skip the heuristic entirely (the explicit override
        still wins if provided).
    user_bound_scale_override:
        Explicit per-solve override, typically loaded from the DB
        ``solve.user_bound_scale`` parameter.  HiGHS' scaling warning
        prints a recommended value in the form ``"Consider setting the
        user_bound_scale option to <N>"`` — passing that ``N`` here is
        the most reliable way to silence the warning.  ``None`` (or 0)
        defers to the heuristic.
    lp_ranges:
        Optional coefficient-range dict from
        :meth:`polar_high.Problem.peek_lp_ranges`.  When supplied (and
        no explicit override is set), use
        :func:`recommend_user_bound_scale_from_lp` instead of the
        input-data heuristic — the post-build ranges are what HiGHS
        actually sees, so the recommendation matches HiGHS' own
        "Consider setting the user_bound_scale option to <N>" advice.
    """
    options: dict[str, int] = {
        "simplex_scale_strategy": SIMPLEX_SCALE_STRATEGY_ADVANCED,
    }
    if user_bound_scale_override is not None and user_bound_scale_override != 0:
        # Clamp to the same range the heuristic uses.  HiGHS itself
        # rejects values outside [-30, 30].  ``user_bound_scale`` is a
        # power of 2 (× 2^N), not a power of 10, so [-30, 30] covers
        # roughly 9 decades each way — plenty for any realistic LP.
        n = int(user_bound_scale_override)
        if n > USER_BOUND_SCALE_MAX:
            n = USER_BOUND_SCALE_MAX
        if n < USER_BOUND_SCALE_MIN:
            n = USER_BOUND_SCALE_MIN
        options["user_bound_scale"] = n
    elif lp_ranges is not None:
        n = recommend_user_bound_scale_from_lp(lp_ranges)
        if n != 0:
            options["user_bound_scale"] = n
    elif apply_user_bound_scale:
        n = recommend_user_bound_scale(table.family_ranges, table.rough_obj_estimate)
        if n != 0:
            options["user_bound_scale"] = n
    return options


# ---------------------------------------------------------------------------
# The analyzer entry point
# ---------------------------------------------------------------------------


def analyze_solve(
    solve_name: str,
    flex_data: "FlexData",
    work_folder: Path | str | None = None,
    logger: logging.Logger | None = None,
) -> ScaleTable:
    """Analyse the inputs of *solve_name* and return a :class:`ScaleTable`.

    Parameters
    ----------
    solve_name:
        The solve name being analysed.  Used as the cache key.
    flex_data:
        The in-memory :class:`~flextool.engine_polars.input.FlexData`
        bag produced by :func:`~flextool.engine_polars.input.load_flextool`
        (or the equivalent in-memory pipeline).
    work_folder:
        Optional path to the solve's working directory.  When provided,
        the scaling analysis JSON is written to
        ``<work_folder>/solve_data/scaling_analysis.json`` (mirroring the
        original module's behaviour).  When ``None`` no file is written.
    logger:
        Optional logger for one-line debug summaries.

    Returns
    -------
    ScaleTable
        The per-solve analysis result.  Subsequent calls with the same
        *solve_name* return the cached table without re-reading any data.
    """
    if solve_name in _scale_cache:
        return _scale_cache[solve_name]

    # ---- Build per-family numpy arrays from FlexData Params ---------------

    # entity_unitsize: p_all_entity_unitsize has schema (e, value) — covers
    # processes, connections AND nodes.
    unitsize_arr = _extract_param_values(flex_data.p_all_entity_unitsize)

    # node_inflow: p_inflow has schema (n, d, t, value).
    inflow_arr = _extract_param_values(flex_data.p_inflow)

    # node_annual_flow: also comes from p_inflow (period-indexed subset).
    # We pool it with node_inflow — the distinction between the two
    # CSV families was an artifact of the CSV writer splitting period-
    # indexed from timestep-indexed rows into separate files.  At the
    # in-memory level they live in the same Param.
    annual_flow_arr = inflow_arr  # same source; intentional alias

    # vom_and_op_costs: pool all variable cost + startup + commodity params.
    vom_arr = np.concatenate([
        _extract_param_values(flex_data.p_pssdt_varCost),
        _extract_param_values(flex_data.p_pdt_varCost_source),
        _extract_param_values(flex_data.p_pdt_varCost_sink),
        _extract_param_values(flex_data.p_pdt_varCost_process),
        _extract_param_values(flex_data.p_startup_cost),
        _extract_param_values(flex_data.p_commodity_price),
    ])

    # capex_invest: lifetime fixed cost (annualised invest + fixed cost).
    capex_arr = _extract_param_values(flex_data.ed_lifetime_fixed_cost)

    # node_penalty: up / down slack penalties on nodes.
    penalty_arr = np.concatenate([
        _extract_param_values(flex_data.p_penalty_up),
        _extract_param_values(flex_data.p_penalty_down),
    ])

    family_arrays: dict[str, np.ndarray] = {
        "entity_unitsize": unitsize_arr,
        "node_inflow": inflow_arr,
        "node_annual_flow": annual_flow_arr,
        "vom_and_op_costs": vom_arr,
        "capex_invest": capex_arr,
        "node_penalty": penalty_arr,
    }

    # ---- Compute FamilyStats for each family ------------------------------
    family_stats: dict[str, FamilyStats] = {
        name: _family_stats_from_array(arr)
        for name, arr in family_arrays.items()
    }

    # ---- Unitsize spread (log10) ------------------------------------------
    unitsize_stats = family_stats.get("entity_unitsize")
    if (
        unitsize_stats is not None
        and unitsize_stats.log10_max is not None
        and unitsize_stats.log10_min is not None
    ):
        spread = unitsize_stats.log10_max - unitsize_stats.log10_min
    else:
        spread = 0.0

    # ---- RHS spread (log10) ----------------------------------------------
    rhs_spread = _pooled_spread_log10(family_stats, RHS_FAMILIES)

    # ---- Cost spread (log10) ---------------------------------------------
    cost_spread = _pooled_spread_log10(family_stats, COST_FAMILIES)

    # ---- Row-scaling trigger (first match wins: unitsize > rhs > cost) ---
    trigger: Literal["unitsize", "rhs", "cost", "none"]
    if spread > UNITSIZE_SPREAD_THRESHOLD:
        trigger = "unitsize"
    elif rhs_spread > RHS_SPREAD_THRESHOLD:
        trigger = "rhs"
    elif cost_spread > COST_SPREAD_THRESHOLD:
        trigger = "cost"
    else:
        trigger = "none"

    use_row_scaling: Literal["yes", "no"] = "yes" if trigger != "none" else "no"

    # ---- Objective scalar recommendation ---------------------------------
    rough_obj = _estimate_rough_obj_inmemory(family_arrays, flex_data)
    # Pool all cost families to find the largest AND smallest absolute
    # cost coefficients — these guard against pushing scaled costs
    # outside HiGHS' usable band.  ``cost_abs_max`` keeps the largest
    # scaled cost below HiGHS' "excessively large" ceiling; ``cost_abs_min``
    # keeps the smallest scaled cost above HiGHS' "excessively small"
    # floor.  When both bind, the two-sided guard finds the best
    # balanced scale (or, if the cost spread exceeds HiGHS' band,
    # falls back to geometric centering — see
    # :func:`_recommend_scale_the_objective`).
    cost_abs_max_pooled: Optional[float] = None
    cost_abs_min_pooled: Optional[float] = None
    for fam in COST_FAMILIES:
        stats = family_stats.get(fam)
        if stats is None:
            continue
        if stats.abs_max is not None and stats.abs_max > 0.0:
            if cost_abs_max_pooled is None or stats.abs_max > cost_abs_max_pooled:
                cost_abs_max_pooled = stats.abs_max
        if stats.abs_min is not None and stats.abs_min > 0.0:
            if cost_abs_min_pooled is None or stats.abs_min < cost_abs_min_pooled:
                cost_abs_min_pooled = stats.abs_min
    # The LP cost coefficient on v_invest is
    #     p_unitsize × (ed_entity_annual_discounted + ed_lifetime_fixed_cost)
    # The family-stat pool above only sees the *raw* params (no unitsize
    # multiplier), so for high-unitsize / high-capex models it understates
    # the actual LP cost magnitude and the cost-floor guard fails to keep
    # scaled coefs out of HiGHS' "excessively large costs" zone.  Include
    # ``p_unitsize × max(invest-cost params)`` here so the floor reflects
    # the LP coefficient.
    invest_lp_max = _invest_cost_lp_coef_upper_bound(flex_data)
    if invest_lp_max is not None and invest_lp_max > 0.0:
        if cost_abs_max_pooled is None or invest_lp_max > cost_abs_max_pooled:
            cost_abs_max_pooled = invest_lp_max
    scale_obj = _recommend_scale_the_objective(
        rough_obj, cost_abs_max_pooled, cost_abs_min_pooled,
    )

    source_label = str(work_folder) if work_folder is not None else "<in-memory>"

    table = ScaleTable(
        solve_name=solve_name,
        use_row_scaling=use_row_scaling,
        scale_the_objective=scale_obj,
        family_ranges=family_stats,
        unitsize_spread_log10=spread,
        rough_obj_estimate=rough_obj,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_dir=source_label,
        rhs_spread_log10=rhs_spread,
        cost_spread_log10=cost_spread,
        row_scaling_trigger=trigger,
    )
    _scale_cache[solve_name] = table

    # ---- Optional JSON write to disk -------------------------------------
    if work_folder is not None:
        try:
            write_scaling_analysis_json(table, Path(work_folder) / "solve_data")
        except OSError as exc:
            if logger is not None:
                logger.warning(
                    "[scaling] %s: could not write scaling_analysis.json: %s",
                    solve_name,
                    exc,
                )

    if logger is not None:
        logger.debug(
            "[scaling] %s: unitsize_spread=%.2f rhs_spread=%.2f cost_spread=%.2f "
            "decades → use_row_scaling=%s (trigger=%s); "
            "rough_obj=%.3g → scale_the_objective=%g",
            solve_name,
            spread,
            rhs_spread,
            cost_spread,
            use_row_scaling,
            trigger,
            rough_obj,
            scale_obj,
        )

    return table


# ---------------------------------------------------------------------------
# JSON emission  (identical to original)
# ---------------------------------------------------------------------------


def write_scaling_analysis_json(
    table: ScaleTable,
    solve_data_dir: Path | str,
    filename: str = "scaling_analysis.json",
) -> Path:
    """Serialise *table* under ``solve_data_dir / filename``."""
    sd = Path(solve_data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    payload = table.to_dict()
    path = sd / filename
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Auto-apply helper  (identical to original)
# ---------------------------------------------------------------------------


def maybe_auto_apply_row_scaling(
    solve_name: str,
    table: ScaleTable,
    user_setting: Optional[str],
    auto_scale: bool,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Decide whether to override the caller's ``use_row_scaling`` setting.

    Returns ``"yes"`` / ``"no"`` to override, or ``None`` to leave
    the DB value untouched.
    """
    if not auto_scale:
        return None
    if user_setting is not None:
        s = str(user_setting).strip().lower()
        if s in ("yes", "no"):
            if logger is not None:
                logger.info(
                    "[scaling] %s: user override detected "
                    "(use_row_scaling=%r), not auto-applying (recommended=%s).",
                    solve_name,
                    user_setting,
                    table.use_row_scaling,
                )
            return None
    if logger is not None:
        logger.info(
            "[scaling] %s: auto-applying recommended use_row_scaling=%s",
            solve_name,
            table.use_row_scaling,
        )
    return table.use_row_scaling


# ---------------------------------------------------------------------------
# Effective-setting resolution  (defensive: tolerates bad DB values)
# ---------------------------------------------------------------------------


# HiGHS solver-option key for advanced row scaling.  Centralised here so
# every call site uses the same value.
SIMPLEX_SCALE_STRATEGY_ADVANCED = 2


def resolve_effective_scaling(
    table: ScaleTable,
    user_row_scaling: object | None,
    user_obj_scale: object | None,
) -> tuple[str, float]:
    """Combine the user's per-solve overrides with the auto-recommendation.

    The DB columns ``use_row_scaling`` and ``scale_the_objective`` are
    optional per-solve overrides.  When present and well-formed they
    win; when absent / malformed we fall back to the analyzer's
    recommendation in *table*.

    Parameters
    ----------
    table:
        The :class:`ScaleTable` produced by :func:`analyze_solve`.
    user_row_scaling:
        Raw value from ``state.solve.use_row_scaling.get(solve_name)``
        (typically ``None``, ``"yes"``, or ``"no"``).
    user_obj_scale:
        Raw value from ``state.solve.scale_the_objective.get(solve_name)``.
        ``params_to_dict`` may return strings for numeric scalars, so we
        accept any object and coerce defensively.

    Returns
    -------
    tuple[str, float]
        ``(effective_use_row_scaling, effective_scale_the_objective)``.
        The first is always ``"yes"`` or ``"no"``; the second is always
        a finite, strictly-positive float.
    """
    # --- row scaling --------------------------------------------------------
    if isinstance(user_row_scaling, str) and user_row_scaling.strip().lower() in ("yes", "no"):
        effective_row = user_row_scaling.strip().lower()
    else:
        effective_row = table.use_row_scaling  # already "yes" / "no"

    # --- objective scale ----------------------------------------------------
    effective_obj = table.scale_the_objective  # default fallback
    if user_obj_scale is not None:
        try:
            candidate = float(user_obj_scale)
            if math.isfinite(candidate) and candidate > 0.0:
                effective_obj = candidate
            # Otherwise keep the analyzer's recommendation; a 0 / NaN /
            # negative override would break un-scaling on the way out.
        except (TypeError, ValueError):
            pass  # malformed user value — fall back to recommendation

    return effective_row, effective_obj


def resolve_user_bound_scale_override(
    user_value: object | None,
) -> Optional[int]:
    """Coerce a raw DB ``user_bound_scale`` value to a clamped int (or None).

    Accepts ``None``, integers, floats, or numeric strings and returns:
        * an integer in ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``
          when the user provided a non-zero, finite, parseable value;
        * ``None`` when the user provided no value, ``0``, or anything
          unparseable — caller should then fall back to the heuristic.

    Truncates non-integer floats toward zero (e.g. ``-3.7`` → ``-3``)
    rather than rounding, mirroring the conservative direction (smaller
    |N| is gentler scaling).
    """
    if user_value is None:
        return None
    try:
        as_float = float(user_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    n = int(as_float)  # truncates toward zero
    if n == 0:
        return None
    if n > USER_BOUND_SCALE_MAX:
        n = USER_BOUND_SCALE_MAX
    if n < USER_BOUND_SCALE_MIN:
        n = USER_BOUND_SCALE_MIN
    return n
