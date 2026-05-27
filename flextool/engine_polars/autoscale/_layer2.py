"""Layer 2 (semantic per-type scaling) — implementation.

Layer 2 mutates a polar-high :class:`Problem` *in place* before
``Problem.solve(...)`` and returns a :class:`Layer2Plan` describing the
inverse transform.  The plan is consumed by
:func:`unscale_solution` after the solver returns, so callers stay
oblivious to the scaling change.

Mathematics (forward semantics — see the bit-for-bit test for the
correctness anchor):

* ``col_factor[j]`` is a positive power-of-two.  The variable
  substitution that lands in the LP arrays passed to HiGHS is
  ``x_scaled[j] = col_factor[j] * x[j]`` ( ⇔  ``x[j] = x_scaled[j] /
  col_factor[j]``).
* ``row_factor[i]`` is a positive power-of-two.  Each row of the
  scaled LP is the original row **multiplied** by ``row_factor[i]``
  (LHS coefficients, RHS, both).

The four LP-array transforms implemented below are then:

==================  ============================================
Quantity            Scaled value
==================  ============================================
``matrix[i, j]``    ``matrix[i, j] * row_factor[i] / col_factor[j]``
``cost[j]``         ``cost[j] / col_factor[j]``
``lb[j], ub[j]``    ``lb[j] * col_factor[j]``, ``ub[j] * col_factor[j]``
``rhs[i]``          ``rhs[i] * row_factor[i]``
==================  ============================================

Unscale (post-solve):

==================  =============================================
Quantity            Original value
==================  =============================================
``x[j]``            ``x_scaled[j] / col_factor[j]``
``col_dual[j]``     ``col_dual_scaled[j] * col_factor[j]``
``row_dual[i]``     ``row_dual_scaled[i] * row_factor[i]``
``obj``             unchanged
==================  =============================================

The objective is invariant: ``c x = (c/cf) * (cf*x) = c_scaled *
x_scaled``.  We do not touch the objective offset, the model sense, or
the integrality flags.

Layer 2 deliberately **does not column-scale integer variables** —
shifting their bounds by a non-unit factor would break integrality of
the recovered solution.  Their bounds, cost, and matrix entries flow
through unchanged.

MPS-export consideration
------------------------

Because Layer 2 mutates the LP arrays *before* they are passed to HiGHS,
``Highs.writeModel('out.mps')`` exports the **scaled** model.  An
external solver consuming that MPS would therefore receive the
re-coordinatised LP, not the original; its results would live in scaled
coordinates and would need the inverse transform recorded in the
``autoscale_<solve>.yaml`` audit (per-type exponents) to be returned to
user units.

For a workflow that writes MPS and passes it to a third-party solver,
the safe defaults are either:

1. Run with ``--scaling=basic`` or ``--scaling=solver_only`` so the exported MPS reflects the
   unscaled problem, or
2. Read the per-type exponents from the autoscale YAML report and apply
   the inverse transform to the external solver's solution.

Layer 3 (``user_*_scale``) is HiGHS-internal and is **not** captured in
``writeModel`` output — that scaling is invisible to external solvers
regardless of FlexTool's autoscale setting.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ._config import ScalingConfig
from ._layer2_types import (
    VarFamily,
    lookup_var,
    resolve_cstr_rhs_type,
)
from ._quantity_types import QuantityType


_logger = logging.getLogger(__name__)


# Clamp on the chosen per-type exponents.  ±20 keeps the scale factors
# inside roughly [1e-6, 1e+6] which is a sane band for energy-system
# LPs; values past that risk losing precision when HiGHS' own
# equilibration multiplies on top.
_DEFAULT_CLAMP = 20


# ---------------------------------------------------------------------------
# QuantityType ↔ int round-trip helpers.
#
# ``QuantityType`` is a *string*-valued Enum (see ``_quantity_types.py``),
# so we cannot use ``QuantityType(int_id)`` directly.  Build a stable
# bijection once at import — keyed by the enum's insertion order — and
# reuse it everywhere polars needs an integer column for ``column_type``
# joins / group-bys.  The mapping is process-local; it is regenerated
# every time this module is imported, which is fine because callers
# always reconstruct ``QuantityType`` *inside* this module.

_QTY_TO_ID: dict[QuantityType, int] = {q: i for i, q in enumerate(QuantityType)}
_ID_TO_QTY: dict[int, QuantityType] = {i: q for q, i in _QTY_TO_ID.items()}


def _qty_to_id(q: QuantityType | None) -> int | None:
    """Stable Int32 id for a :class:`QuantityType`, or ``None``."""
    if q is None:
        return None
    return _QTY_TO_ID[q]


def _id_to_qty(i: int) -> QuantityType:
    return _ID_TO_QTY[int(i)]


@dataclass(frozen=True)
class Layer2Plan:
    """Forward + inverse Layer-2 transform.

    ``col_factors`` and ``row_factors`` are length-``n_cols`` /
    ``n_rows`` ``np.float64`` arrays.  All entries are positive
    powers of two (1.0 = identity); the bit-for-bit roundtrip relies
    on this.

    Attributes
    ----------
    col_factors:
        Per-column forward multipliers.
    row_factors:
        Per-row forward multipliers.
    type_exponents:
        Power-of-two exponent picked for each :class:`QuantityType`
        present in the LP (for reporting).
    type_buckets_before:
        Per-type (min, max) magnitude before scaling — fed to the
        Layer-2 section of the autoscale YAML.
    type_buckets_after:
        Per-type (min, max) magnitude after scaling.
    skipped_rows:
        Constraint-row names whose family declares ``rhs_type=None``
        (user-defined constraints / structural zero-RHS rows) and
        therefore receive no per-row factor.
    skipped_integer_cols:
        Column ids of integer variables that received no column
        scaling (their factor is 1.0 in ``col_factors``).
    """

    col_factors: np.ndarray
    row_factors: np.ndarray
    type_exponents: dict[QuantityType, int]
    type_buckets_before: dict[QuantityType, tuple[float, float]]
    type_buckets_after: dict[QuantityType, tuple[float, float]] = field(
        default_factory=dict,
    )
    skipped_rows: list[str] = field(default_factory=list)
    skipped_integer_cols: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bucketing


def _effective_matrix_type(
    var_family: VarFamily,
    row_type: QuantityType | None,
) -> QuantityType:
    """Return the bucket key for a matrix entry sitting in ``row_type``'s
    row and feeding ``var_family``'s column.

    Layer 2 buckets by the matrix entry's effective physical type — the
    product of the column type and the row's multiplier param's type —
    so ``v_flow``-times-``p_unitsize`` rows land in POWER even though
    the column itself is DIMENSIONLESS.

    Heuristic:

    * If the column carries a ``multiplier_param``, the matrix entry's
      type is *whichever of POWER / ENERGY the row demands*.  We trust
      the row's ``rhs_type`` for this — every flextool family
      multiplies the dimensionless column by exactly the row's units.
    * Otherwise the column's own type is the entry type.
    * When ``row_type`` is ``None`` (skip-per-row family), we fall
      back to the column's own type so the matrix entry still
      contributes to its column-side bucket.
    """
    if var_family.multiplier_param is not None and row_type is not None:
        # User constraints (rhs_type=None) hit the column-type branch
        # below; for them we don't push a multiplied effective type
        # because the multiplier param IS the user's coefficient.
        return row_type
    return var_family.column_type


def _per_var_column_type(name: str) -> QuantityType:
    return lookup_var(name).column_type


# Per-type accumulator value: ``(log2_sum, count, abs_min, abs_max)``.
#
# Layer 2 only needs ``(log2_sum, count)`` to compute the geometric mean
# (see :func:`choose_scale_powers`).  ``abs_min`` / ``abs_max`` are
# carried for the report-side ``type_buckets_before`` /
# ``type_buckets_after`` fields on :class:`Layer2Plan` — both come for
# free from the polars ``group_by`` so we propagate them rather than
# losing the information.
_AccVal = tuple[float, int, float, float]
_INIT_ACC: _AccVal = (0.0, 0, math.inf, 0.0)


def _build_col_id_classification(problem: Any) -> pl.DataFrame:
    """Return a per-``col_id`` classification table.

    Columns:

    * ``col_id``               — Int64
    * ``column_type_id``       — Int32 (see :data:`_QTY_TO_ID`)
    * ``has_multiplier_param`` — Boolean

    Built once per :func:`bucket_coefficients` call and joined into
    every objective / constraint term's lazy plan.  ``has_multiplier_param``
    drives :func:`_effective_matrix_type`'s branch — when set *and* the
    constraint family has a ``rhs_type``, the matrix entry is bucketed
    against the row's type rather than the column's.
    """
    col_ids: list[int] = []
    type_ids: list[int] = []
    has_mp: list[bool] = []
    for name, var in problem._vars.items():
        try:
            fam = lookup_var(name)
        except KeyError as exc:
            raise KeyError(
                f"Layer 2: variable {name!r} not in VARIABLE_FAMILIES — "
                "register it in _layer2_types.py before solving."
            ) from exc
        ct_id = _qty_to_id(fam.column_type)
        mp = fam.multiplier_param is not None
        ids = var.frame["col_id"].to_numpy().tolist()
        col_ids.extend(int(c) for c in ids)
        type_ids.extend([ct_id] * len(ids))
        has_mp.extend([mp] * len(ids))
    return pl.DataFrame(
        {
            "col_id": pl.Series(col_ids, dtype=pl.Int64),
            "column_type_id": pl.Series(type_ids, dtype=pl.Int32),
            "has_multiplier_param": pl.Series(has_mp, dtype=pl.Boolean),
        }
    )


def _collect_term_agg(
    term_lazy: pl.LazyFrame,
    *,
    classification_lazy: pl.LazyFrame,
    rhs_t_id: int | None,
) -> pl.DataFrame | None:
    """Aggregate one term to ``(eff_t, log_sum, n, log_min, log_max)``.

    Returns ``None`` for terms with no finite nonzero coefficient (the
    caller skips merging in that case).

    Streaming engine: we try the ``streaming`` engine first.  Polars'
    streaming planner does not support every expression (``.log()`` and
    ``.abs()`` historically had patches of unsupported ops); if the
    streaming collect raises, we fall back to a per-term in-memory
    collect.  Even in the fallback the peak is bounded by *one* term's
    materialised size — orders of magnitude smaller than the original
    code's all-terms-at-once Python-list accumulation.
    """
    if rhs_t_id is None:
        eff_t_expr = pl.col("column_type_id")
    else:
        eff_t_expr = (
            pl.when(pl.col("has_multiplier_param"))
            .then(pl.lit(rhs_t_id, dtype=pl.Int32))
            .otherwise(pl.col("column_type_id"))
        )
    plan = (
        term_lazy
        .join(classification_lazy, on="col_id", how="inner")
        .with_columns(eff_t=eff_t_expr)
        .filter(pl.col("coef") != 0.0)
        .with_columns(log_abs=pl.col("coef").abs().log(2.0))
        .filter(pl.col("log_abs").is_finite())
        .group_by("eff_t")
        .agg(
            pl.col("log_abs").sum().alias("log_sum"),
            pl.len().alias("n"),
            pl.col("log_abs").min().alias("log_min"),
            pl.col("log_abs").max().alias("log_max"),
        )
    )
    try:
        agg = plan.collect(engine="streaming")
    except Exception as exc:  # noqa: BLE001 — polars may raise diverse types
        _logger.debug(
            "Layer 2: streaming engine refused term aggregate (%s); "
            "falling back to in-memory collect for this term.",
            exc,
        )
        agg = plan.collect()
    if agg.height == 0:
        return None
    return agg


def _merge_into_accumulator(
    acc: dict[QuantityType, _AccVal],
    agg: pl.DataFrame,
) -> None:
    """Fold one term's aggregate into the per-type accumulator."""
    for eff_t_id, log_sum, n, log_min, log_max in zip(
        agg["eff_t"].to_list(),
        agg["log_sum"].to_list(),
        agg["n"].to_list(),
        agg["log_min"].to_list(),
        agg["log_max"].to_list(),
    ):
        if eff_t_id is None:
            continue
        t = _id_to_qty(int(eff_t_id))
        ps, pn, pmin, pmax = acc.get(t, _INIT_ACC)
        amin = float(2.0 ** float(log_min))
        amax = float(2.0 ** float(log_max))
        acc[t] = (
            ps + float(log_sum),
            pn + int(n),
            min(pmin, amin),
            max(pmax, amax),
        )


def bucket_coefficients(
    problem: Any,
) -> tuple[
    dict[QuantityType, _AccVal],
    dict[QuantityType, _AccVal],
    dict[QuantityType, _AccVal],
    dict[int, QuantityType],
]:
    """Walk ``problem._vars``, ``problem._cstrs``, ``problem._obj_terms``
    and aggregate every nonzero finite coefficient magnitude by its
    effective :class:`QuantityType`.

    Returns four maps:

    * ``matrix_acc[t]`` — constraint-matrix entries, packed as
      ``(log2_sum, count, abs_min, abs_max)``.
    * ``cost_acc[t]`` — objective coefficients (same packing).
    * ``bound_acc[t]`` — finite variable bounds (same packing).
      RHS magnitudes are not bucketed here — they are handled per-family
      via the row_factor logic.
    * ``col_id_to_type`` — column id → its column QuantityType.  Used
      by :func:`apply_layer2` to push per-column factors back into the
      lazy term plans.

    Implementation note (rewrite 2026-05-27): aggregation happens
    inside polars via ``group_by(eff_t).agg(sum/count/min/max of
    log2|coef|)`` *per term*, with the streaming engine preferred and
    a per-term non-streaming fallback.  We never materialise a
    Python-list of every nonzero — only the (tiny) per-type aggregate
    frame, dropping peak RSS from O(nnz) to O(types · terms).
    """
    classification = _build_col_id_classification(problem)
    classification_lazy = classification.lazy()

    # ── col_id → column QuantityType (consumed by apply_layer2).
    col_id_to_type: dict[int, QuantityType] = {}
    for name, var in problem._vars.items():
        fam = lookup_var(name)  # KeyError already filtered above.
        for cid in var.frame["col_id"].to_numpy().tolist():
            col_id_to_type[int(cid)] = fam.column_type

    matrix_acc: dict[QuantityType, _AccVal] = {}
    cost_acc: dict[QuantityType, _AccVal] = {}
    bound_acc: dict[QuantityType, _AccVal] = {}

    # ── Variable bounds: small (≤ 2 per var family); keep in Python
    # but match the (sum_log2, count, min, max) accumulator shape.
    for name, var in problem._vars.items():
        fam = lookup_var(name)
        for b in (var.lower, var.upper):
            if not math.isfinite(b) or b == 0.0:
                continue
            av = abs(float(b))
            lv = math.log2(av)
            if not math.isfinite(lv):
                continue
            ps, pn, pmin, pmax = bound_acc.get(fam.column_type, _INIT_ACC)
            bound_acc[fam.column_type] = (
                ps + lv, pn + 1, min(pmin, av), max(pmax, av),
            )

    # ── Objective: rhs_t is N/A → eff_t == column_type_id.
    for term in problem._obj_terms:
        agg = _collect_term_agg(
            term.lazy, classification_lazy=classification_lazy, rhs_t_id=None,
        )
        if agg is not None:
            _merge_into_accumulator(cost_acc, agg)

    # ── Matrix: per-family walk of the expression terms.
    for cname, proto, over in problem._cstrs:
        try:
            rhs_t = resolve_cstr_rhs_type(cname)
        except KeyError as exc:
            raise KeyError(
                f"Layer 2: constraint {cname!r} not in CONSTRAINT_FAMILIES "
                "— register it in _layer2_types.py before solving."
            ) from exc
        rhs_t_id = _qty_to_id(rhs_t)
        for term in proto.expr.terms:
            agg = _collect_term_agg(
                term.lazy,
                classification_lazy=classification_lazy,
                rhs_t_id=rhs_t_id,
            )
            if agg is not None:
                _merge_into_accumulator(matrix_acc, agg)

    return matrix_acc, cost_acc, bound_acc, col_id_to_type


# ---------------------------------------------------------------------------
# Small helpers


def _col_id_cache(problem: Any) -> dict[int, str]:
    cache: dict[int, str] = {}
    for name, var in problem._vars.items():
        for cid in var.frame["col_id"].to_numpy().tolist():
            cache[int(cid)] = name
    return cache


# Memoised per-problem; rebuilt by ``_col_id_to_var_name``'s first call.
_COL_ID_CACHE_ATTR = "_autoscale_col_id_cache"


def _col_id_to_var_name(problem: Any, col_id: int) -> str:
    cache = getattr(problem, _COL_ID_CACHE_ATTR, None)
    if cache is None:
        cache = _col_id_cache(problem)
        try:
            setattr(problem, _COL_ID_CACHE_ATTR, cache)
        except Exception:
            # If the Problem doesn't permit dynamic attrs, fall back
            # to per-call recomputation.  polar-high's Problem has a
            # regular __dict__, so this branch should not fire.
            pass
    return cache[col_id]


def _bucket_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    arr = np.asarray(values)
    return (float(arr.min()), float(arr.max()))


# ---------------------------------------------------------------------------
# Power-of-two exponent picking


def choose_scale_powers(
    matrix_acc: dict[QuantityType, _AccVal],
    cost_acc: dict[QuantityType, _AccVal],
    bound_acc: dict[QuantityType, _AccVal],
    *,
    clamp: int = _DEFAULT_CLAMP,
) -> dict[QuantityType, int]:
    """Pick a power-of-two exponent for each :class:`QuantityType` seen.

    Per Bröchin et al. (2024) and the autoscaler handoff
    (``specs/flextool-autoscaling-handoff.md``):

        s_t = round( log2( 1 / geomean(|values|) ) )

    where the geometric mean is computed via
    ``log_mean = sum(log2|values|) / count`` pooled across the matrix,
    cost, and bound accumulators of type ``t``.  We clamp into
    ``[-clamp, +clamp]`` to keep the factors inside double precision's
    comfort zone.

    Returns a dict of :class:`QuantityType` → int.  Types with no
    samples (``count == 0``) are absent (so callers can default their
    factor to 1.0).

    Mathematically equivalent to the pre-2026-05-27 implementation that
    pooled raw magnitude lists, modulo at most ±1 on values that sit on
    a half-integer ``round`` boundary (sum-then-divide vs.
    ``np.log2(arr).mean()`` reorder floating-point ops).
    """
    pool: dict[QuantityType, tuple[float, int]] = {}
    for src in (matrix_acc, cost_acc, bound_acc):
        for t, val in src.items():
            log_sum, count = val[0], val[1]
            ps, pn = pool.get(t, (0.0, 0))
            pool[t] = (ps + float(log_sum), pn + int(count))

    chosen: dict[QuantityType, int] = {}
    for t, (log_sum, count) in pool.items():
        if count == 0:
            continue
        log_mean = log_sum / count
        exp = int(round(-log_mean))
        if exp > clamp:
            exp = clamp
        elif exp < -clamp:
            exp = -clamp
        chosen[t] = exp

    return chosen


# ---------------------------------------------------------------------------
# Lazy-term rewriter


def _rewrite_term_lazy(
    term_lazy: pl.LazyFrame,
    *,
    inv_col_factor_df: pl.DataFrame,
    row_factor: float | None,
) -> pl.LazyFrame:
    """Rewrite ``term.lazy`` so its ``coef`` column carries the Layer-2
    multiplicative adjustments.

    Two stages:

    * Per-column: left-join ``inv_col_factor_df`` (columns
      ``col_id``, ``__l2_inv_cf``) on ``col_id``, then multiply
      ``coef *= __l2_inv_cf``.
    * Per-row (constant across the family): multiply ``coef *=
      row_factor`` if not None.

    The join is left so any col_id absent from the table (shouldn't
    happen — every variable's columns are bucketed) gets a null
    multiplier; we coalesce to 1.0 to avoid silent NULL coef rows.
    """
    plan = term_lazy.join(inv_col_factor_df.lazy(), on="col_id", how="left")
    plan = plan.with_columns(
        coef=(
            pl.col("coef")
            * pl.col("__l2_inv_cf").fill_null(1.0)
        ),
    ).drop("__l2_inv_cf")
    if row_factor is not None and row_factor != 1.0:
        plan = plan.with_columns(coef=pl.col("coef") * float(row_factor))
    return plan


def _rewrite_obj_term_lazy(
    term_lazy: pl.LazyFrame,
    *,
    inv_col_factor_df: pl.DataFrame,
) -> pl.LazyFrame:
    plan = term_lazy.join(inv_col_factor_df.lazy(), on="col_id", how="left")
    plan = plan.with_columns(
        coef=(
            pl.col("coef")
            * pl.col("__l2_inv_cf").fill_null(1.0)
        ),
    ).drop("__l2_inv_cf")
    return plan


# ---------------------------------------------------------------------------
# Public API


def apply_layer2(
    problem: Any,
    config: ScalingConfig,
) -> Layer2Plan:
    """Apply Layer 2 to ``problem`` in place.

    Mutates ``problem._vars`` (bound rescale on non-integer columns),
    ``problem._cstrs`` (each term's lazy plan + each proto's rhs),
    and ``problem._obj_terms`` (lazy plan rewrite for the cost
    vector).  Returns a :class:`Layer2Plan` carrying the inverse
    transform for :func:`unscale_solution`.
    """
    matrix_acc, cost_acc, bound_acc, col_id_to_type = bucket_coefficients(problem)
    exponents = choose_scale_powers(matrix_acc, cost_acc, bound_acc)

    n_cols = problem._next_col
    col_factors = np.ones(n_cols, dtype=np.float64)

    # ── Column factors -------------------------------------------------
    integer_cols: list[int] = []
    for name, var in problem._vars.items():
        if var.integer:
            ids = var.frame["col_id"].to_numpy()
            integer_cols.extend(int(c) for c in ids.tolist())
            continue
        fam = lookup_var(name)
        col_t = fam.column_type
        exp = exponents.get(col_t)
        if exp is None:
            continue
        f = float(2 ** exp)
        ids = var.frame["col_id"].to_numpy()
        col_factors[ids] = f

    # Variable bound mutation — multiply finite bounds by col_factor.
    # Skip integer columns (col_factors[j] == 1.0 there by construction).
    for name, var in problem._vars.items():
        if var.integer:
            continue
        fam = lookup_var(name)
        f = float(2 ** exponents[fam.column_type]) if fam.column_type in exponents else 1.0
        if f == 1.0:
            continue
        if math.isfinite(var.lower):
            var.lower = float(var.lower) * f
        if math.isfinite(var.upper):
            var.upper = float(var.upper) * f

    # Build the inv_col_factor DataFrame once.  ``inv_cf = 1 / cf``;
    # since cf is a power of two, inv_cf is exact in IEEE.
    inv_col_factor_df = pl.DataFrame(
        {
            "col_id": pl.Series(np.arange(n_cols, dtype=np.int64)),
            "__l2_inv_cf": pl.Series(1.0 / col_factors),
        }
    )

    # ── Row factors + per-family LHS rewrite ---------------------------
    row_factors_list: list[float] = []
    skipped_rows: list[str] = []

    new_cstrs: list[tuple[str, Any, Any]] = []
    for cname, proto, over in problem._cstrs:
        rhs_t = resolve_cstr_rhs_type(cname)
        if rhs_t is None:
            rf = 1.0
        else:
            exp = exponents.get(rhs_t)
            rf = float(2 ** exp) if exp is not None else 1.0
        # Row count for this family.
        row_count = 1 if over is None else int(over.height)
        if rhs_t is None and row_count > 0:
            skipped_rows.append(cname)
        row_factors_list.extend([rf] * row_count)

        # Rewrite each LHS term.
        new_terms = []
        for term in proto.expr.terms:
            new_lazy = _rewrite_term_lazy(
                term.lazy,
                inv_col_factor_df=inv_col_factor_df,
                row_factor=rf if rf != 1.0 else None,
            )
            # Construct new _Term preserving dims/param_sources.
            new_term = type(term)(
                new_lazy, term.dims,
                param_sources=getattr(term, "param_sources", None),
            )
            new_terms.append(new_term)
        new_expr = type(proto.expr)(new_terms)

        # Rewrite RHS.  May be (int, float) | Param | Var | Expr.
        new_rhs = proto.rhs
        if rf != 1.0:
            new_rhs = _scale_rhs(proto.rhs, rf, inv_col_factor_df)
        # ALSO must rewrite Var/Expr RHS for column factors (the rhs
        # variable becomes part of the matrix via _solve_streaming's
        # negation step).  Even if rf == 1.0, we must apply col
        # scaling to any Var/Expr RHS.
        elif _rhs_has_vars(proto.rhs):
            new_rhs = _scale_rhs(proto.rhs, 1.0, inv_col_factor_df)

        new_proto = type(proto)(new_expr, proto.sense, new_rhs)
        new_cstrs.append((cname, new_proto, over))

    problem._cstrs[:] = new_cstrs

    # ── Objective rewrite ---------------------------------------------
    new_obj_terms = []
    for term in problem._obj_terms:
        new_lazy = _rewrite_obj_term_lazy(
            term.lazy, inv_col_factor_df=inv_col_factor_df,
        )
        new_term = type(term)(
            new_lazy, term.dims,
            param_sources=getattr(term, "param_sources", None),
        )
        new_obj_terms.append(new_term)
    problem._obj_terms[:] = new_obj_terms

    row_factors = np.asarray(row_factors_list, dtype=np.float64)

    # Per-type bucket-range reporting.  The accumulators carry
    # (log_sum, count, abs_min, abs_max); to combine min/max across
    # matrix∪cost∪bound we take element-wise min/max over the four-tuples
    # for each type.
    all_types = set(matrix_acc) | set(cost_acc) | set(bound_acc)
    type_buckets_before: dict[QuantityType, tuple[float, float]] = {}
    type_buckets_after: dict[QuantityType, tuple[float, float]] = {}
    for t in all_types:
        amin = math.inf
        amax = 0.0
        any_seen = False
        for src in (matrix_acc, cost_acc, bound_acc):
            if t in src:
                _, count, mn, mx = src[t]
                if count > 0:
                    amin = min(amin, mn)
                    amax = max(amax, mx)
                    any_seen = True
        if not any_seen:
            continue
        type_buckets_before[t] = (float(amin), float(amax))
        exp = exponents.get(t, 0)
        f = float(2 ** exp)
        type_buckets_after[t] = (float(amin * f), float(amax * f))

    plan = Layer2Plan(
        col_factors=col_factors,
        row_factors=row_factors,
        type_exponents=exponents,
        type_buckets_before=type_buckets_before,
        type_buckets_after=type_buckets_after,
        skipped_rows=skipped_rows,
        skipped_integer_cols=integer_cols,
    )
    return plan


def _rhs_has_vars(rhs: Any) -> bool:
    # Var or Expr RHS will become matrix entries.  Param and scalars
    # become row bounds.
    cls_name = type(rhs).__name__
    return cls_name in ("Var", "Expr")


def _scale_rhs(
    rhs: Any,
    row_factor: float,
    inv_col_factor_df: pl.DataFrame,
) -> Any:
    """Multiply an RHS by ``row_factor``.

    Cases:

    * scalar (int / float): multiply directly.
    * Param: multiply the underlying frame's ``value`` column by
      ``row_factor`` by building a new Param wrapping the rescaled
      frame.
    * Var: convert to Expr; recurse.
    * Expr: apply col-factor join + ``coef *= row_factor`` to every
      term, then return a new Expr.

    The mutation is *out-of-place* — we never modify the caller's
    objects beyond what apply_layer2 explicitly does in
    ``problem._cstrs``.
    """
    cls_name = type(rhs).__name__
    if isinstance(rhs, (int, float)):
        if rhs == 0:
            return rhs
        return float(rhs) * row_factor
    # polar_high types — import lazily so this module doesn't drag the
    # polar-high import chain when only the registries are needed.
    from polar_high.engine import Expr, Param, Var, _Term  # noqa: WPS433

    if isinstance(rhs, Param):
        # Build a new Param with rescaled value column.
        old = rhs.frame
        if "value" not in old.columns:
            raise ValueError(
                f"Layer 2: Param frame missing 'value' column ({old.columns})"
            )
        new_frame = old.with_columns(value=pl.col("value") * float(row_factor))
        return Param(rhs.dims, new_frame)

    if isinstance(rhs, Var):
        rhs = rhs.to_expr()

    if isinstance(rhs, Expr):
        new_terms = []
        for term in rhs.terms:
            new_lazy = _rewrite_term_lazy(
                term.lazy,
                inv_col_factor_df=inv_col_factor_df,
                row_factor=row_factor if row_factor != 1.0 else None,
            )
            new_terms.append(
                _Term(
                    new_lazy, term.dims,
                    param_sources=getattr(term, "param_sources", None),
                )
            )
        return Expr(new_terms)

    raise TypeError(
        f"Layer 2 _scale_rhs: unsupported rhs type {cls_name}"
    )


# ---------------------------------------------------------------------------
# Unscale


def unscale_solution(sol: Any, plan: Layer2Plan) -> None:
    """In-place: undo the Layer-2 forward transform on ``sol``.

    Mutates ``sol.col_value``, ``sol.col_dual``, ``sol.row_dual``.
    ``sol.obj`` is invariant under the (c→c/cf, x→cf·x) substitution
    and is left untouched.

    Length checks ensure the plan matches the solution.  Mismatch
    indicates the caller wired the plan to the wrong solve; we raise
    rather than silently corrupt the results.
    """
    cv = np.asarray(sol.col_value, dtype=np.float64)
    if cv.shape[0] != plan.col_factors.shape[0]:
        raise ValueError(
            f"Layer 2 unscale: col_value length {cv.shape[0]} != "
            f"col_factors length {plan.col_factors.shape[0]}"
        )
    sol.col_value = cv / plan.col_factors

    cd = np.asarray(getattr(sol, "col_dual", None), dtype=np.float64) \
        if getattr(sol, "col_dual", None) is not None else None
    if cd is not None and cd.size > 0:
        if cd.shape[0] != plan.col_factors.shape[0]:
            raise ValueError(
                f"Layer 2 unscale: col_dual length {cd.shape[0]} != "
                f"col_factors length {plan.col_factors.shape[0]}"
            )
        sol.col_dual = cd * plan.col_factors

    rd = np.asarray(getattr(sol, "row_dual", None), dtype=np.float64) \
        if getattr(sol, "row_dual", None) is not None else None
    if rd is not None and rd.size > 0:
        if rd.shape[0] != plan.row_factors.shape[0]:
            raise ValueError(
                f"Layer 2 unscale: row_dual length {rd.shape[0]} != "
                f"row_factors length {plan.row_factors.shape[0]}"
            )
        sol.row_dual = rd * plan.row_factors


__all__ = [
    "Layer2Plan",
    "apply_layer2",
    "bucket_coefficients",
    "choose_scale_powers",
    "unscale_solution",
]
