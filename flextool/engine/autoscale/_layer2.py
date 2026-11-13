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
import os
import sys
import time
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


# ---------------------------------------------------------------------------
# Capability gate for the bounded coefficient-walk (Phase D-5 step 3).
#
# ``bucket_coefficients`` prefers ``polar_high.autoscale._coef_walk`` to walk
# the block-COO ``(rid/col_id, coef)`` stream in bounded slices instead of
# materialising the full ``Var ⋈ P1 ⋈ P2 …`` product (see its docstring).
# The walk ships in polar-high>=2.4.0 (pinned in pyproject), so the import
# is guaranteed.  The capability detect below is retained as a defensive
# no-op: if a user force-downgrades polar-high, ``bucket_coefficients``
# falls back to the materialising ``_collect_term_agg`` collect for every
# term (correct, just unbounded) rather than raising ``ImportError``.
try:  # pragma: no cover - exercised by both-polar_high verification runs
    from polar_high.autoscale._coef_walk import (  # noqa: F401
        ArrayClassifier as _ArrayClassifier,
        CoefWalkRecipe as _CoefWalkRecipe,
        Log2HistogramReducer as _Log2HistogramReducer,
        bounded_coefficient_walk as _bounded_coefficient_walk,
    )

    _HAVE_COEF_WALK = True
except ImportError:
    _HAVE_COEF_WALK = False


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


def _var_col_ids(problem: Any) -> list[tuple[np.ndarray, QuantityType, bool]]:
    """Per Var, in ``problem._vars`` order: ``(col_ids, column_type,
    has_multiplier_param)``.

    Raises :class:`KeyError` naming the variable when it is not registered
    in ``VARIABLE_FAMILIES``.
    """
    out: list[tuple[np.ndarray, QuantityType, bool]] = []
    for name, var in problem._vars.items():
        try:
            fam = lookup_var(name)
        except KeyError as exc:
            raise KeyError(
                f"Layer 2: variable {name!r} not in VARIABLE_FAMILIES — "
                "register it in _layer2_types.py before solving."
            ) from exc
        out.append((
            var.frame["col_id"].to_numpy().astype(np.int64, copy=False),
            fam.column_type,
            fam.multiplier_param is not None,
        ))
    return out


def _col_type_table(
    problem: Any,
) -> tuple[np.ndarray, np.ndarray, list[QuantityType]]:
    """Per-``col_id`` column type and multiplier flag as numpy arrays.

    Returns ``(type_idx, has_mp, types)``: ``type_idx[c]`` (int16) is the
    index into ``types`` of column ``c``'s :class:`QuantityType` (``-1`` for
    a ``col_id`` no Var owns), ``has_mp[c]`` whether its family carries a
    ``multiplier_param``.  ``types`` lists each distinct column type once,
    in first-appearance order over ``problem._vars``.  One small integer per
    LP column instead of a Python object per column; a ``col_id`` owned by
    two Vars keeps the later one (Var order).
    """
    per_var = _var_col_ids(problem)
    types: list[QuantityType] = []
    idx_of: dict[QuantityType, int] = {}
    n_cols = 0
    for cids, ct, _mp in per_var:
        if ct not in idx_of:
            idx_of[ct] = len(types)
            types.append(ct)
        if cids.size:
            n_cols = max(n_cols, int(cids.max()) + 1)
    type_idx = np.full(n_cols, -1, dtype=np.int16)
    has_mp = np.zeros(n_cols, dtype=bool)
    for cids, ct, mp in per_var:
        type_idx[cids] = idx_of[ct]
        has_mp[cids] = mp
    return type_idx, has_mp, types


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
    against the row's type rather than the column's.  Rows follow
    ``problem._vars`` order, each Var's ``col_id`` in frame order.
    """
    per_var = _var_col_ids(problem)
    if not per_var:
        col_ids = np.zeros(0, dtype=np.int64)
        type_ids = np.zeros(0, dtype=np.int32)
        has_mp = np.zeros(0, dtype=bool)
    else:
        col_ids = np.concatenate([cids for cids, _ct, _mp in per_var])
        type_ids = np.concatenate([
            np.full(cids.size, _QTY_TO_ID[ct], dtype=np.int32)
            for cids, ct, _mp in per_var
        ])
        has_mp = np.concatenate([
            np.full(cids.size, mp, dtype=bool) for cids, _ct, mp in per_var
        ])
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

    Determinism: ``log_sum`` is the correctly rounded sum
    (:func:`_group_fsums`) of each group's ``log_abs`` values, not a polars
    ``sum()``: the streaming engine hands rows to the sum in an order that
    depends on the thread schedule, so a running float sum gave different
    last bits per run and per ``POLARS_MAX_THREADS`` (item 253).  The plan
    therefore collects the two narrow columns ``(eff_t, log_abs)`` (12
    bytes a row) instead of the per-group aggregate; count / min / max are
    exact in any order and stay in polars.
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
        .filter(pl.col("log_abs").is_finite() & pl.col("eff_t").is_not_null())
        .select("eff_t", "log_abs")
    )
    try:
        vals = plan.collect(engine="streaming")
    except Exception as exc:  # noqa: BLE001 — polars may raise diverse types
        _logger.debug(
            "Layer 2: streaming engine refused term aggregate (%s); "
            "falling back to in-memory collect for this term.",
            exc,
        )
        vals = plan.collect()
    if vals.height == 0:
        return None
    agg = (
        vals.group_by("eff_t")
        .agg(
            pl.len().alias("n"),
            pl.col("log_abs").min().alias("log_min"),
            pl.col("log_abs").max().alias("log_max"),
        )
        .sort("eff_t")
    )
    eff = vals["eff_t"].to_numpy()
    logs = vals["log_abs"].to_numpy()
    del vals
    sums = _group_fsums(eff, logs, agg["eff_t"].to_numpy(), agg["n"].to_numpy())
    return agg.with_columns(log_sum=pl.Series(sums, dtype=pl.Float64))


# Rows handed to ``math.fsum`` per ``tolist()`` slice: bounds the transient
# Python-float list of a big group (~32 bytes a value) to a few MB.
_FSUM_CHUNK = 1 << 18


def _group_fsums(
    keys: np.ndarray,
    logs: np.ndarray,
    groups: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Per group in ``groups`` (sorted ascending, the distinct ``keys``) the
    correctly rounded sum (:func:`math.fsum`) of its ``logs``.

    ``fsum`` is exact before its one rounding, so the result depends only on
    which values a group holds, never on their order — the same bits for
    any thread count, as polar-high's ``Log2HistogramReducer`` does for the
    walked terms.  Rows are grouped with one stable argsort (the groups are
    the few quantity types); each group is fed to a single ``fsum`` through
    bounded ``tolist()`` slices so a large group never becomes one big list.
    """
    if groups.size == 1:
        ordered = logs
    else:
        ordered = logs[np.argsort(keys, kind="stable")]
    ends = np.cumsum(counts.astype(np.int64))
    starts = ends - counts
    out = np.empty(groups.size, dtype=np.float64)
    for g, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        out[g] = math.fsum(
            v
            for c in range(s, e, _FSUM_CHUNK)
            for v in ordered[c:min(c + _FSUM_CHUNK, e)].tolist()
        )
    return out


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


# ---------------------------------------------------------------------------
# Bounded coefficient-walk wiring (Phase D-5 step 3).
#
# ``_collect_term_agg`` above materialises the merged ``Var ⋈ P1 ⋈ P2 …``
# (or RHS Param) chain per term to reduce it to a per-type
# ``(Σlog2|coef|, count, min, max)`` histogram.  On the FlexTool DES LP the
# polars streaming planner cannot push the group-by into the deep product,
# so the product materialises — the residual ~46 GB autoscale peak.
#
# The walk below replaces that materialising collect with
# :func:`polar_high.autoscale._coef_walk.bounded_coefficient_walk` driven by
# :class:`Log2HistogramReducer`: each term's ``(rid/col_id, coef)`` stream is
# walked in bounded ``_WALK_BATCH_ROWS`` slices and folded into the same
# per-bucket ``(Σlog2, count, min, max)`` accumulator.  ``scale=(None,0,None)``
# — bucketing uses RAW ``|coef|`` (no side-vector scaling).  The reducer's
# ``classify`` reproduces the ``col_id → eff_t`` mapping the old
# ``_collect_term_agg`` joined: for the objective ``eff_t = column_type``; for
# a matrix family ``eff_t = rhs_type`` when the column carries a multiplier
# param (and the family has an rhs_type), else ``column_type``.
#
# Terms with no recoverable block-COO recipe (a fully-collapsed
# ``Sum(over=ALL)`` term clears ``var_source`` / ``sum_block_meta`` and ends
# up ``over is None`` / ``dims == ()``) cannot be rebuilt by the walk; they
# keep the existing ``_collect_term_agg`` collect as a backstop — bounded by
# the (tiny) per-type aggregate, the same envelope as before.

# 256k keeps each batch's block-COO product comfortably small while
# amortising per-batch overhead.  The histogram's per-batch Σlog2 reassociates
# vs a single whole-collect sum, so a coefficient on a half-integer log2
# boundary may shift a chosen exponent by ±1 → a different (objective-
# invariant) scaling.  Accepted per the step-3 correctness bar.
_WALK_BATCH_ROWS = 256_000


def _layer2_bucket_profiler() -> Any:
    """Return an ``emit(family, term_idx, **extras)`` callable when
    ``POLAR_HIGH_LAYER2_PROFILE=1`` and ``psutil`` is importable, else
    ``None``.

    Mirrors the ``[ranges-stream profile]`` instrument in
    :mod:`polar_high.autoscale._ranges`: one ``[layer2-bucket profile]``
    stderr line per walked term carrying ``family``, ``term_idx``,
    ``over_height`` / ``n``, the post-walk wall clock, and an RSS sample —
    so the final DES run can CONFIRM this site was the ~46 GB driver and is
    now bounded.
    """
    if os.environ.get("POLAR_HIGH_LAYER2_PROFILE") != "1":
        return None
    try:
        import psutil
    except ImportError:
        return None
    proc = psutil.Process()
    t0 = time.monotonic()

    def _emit(family: str, term_idx: int, **extras: Any) -> None:
        rss = proc.memory_info().rss / (1024 ** 3)
        wall = time.monotonic() - t0
        extras_str = "\t".join(f"{k}={v}" for k, v in extras.items())
        print(
            f"[layer2-bucket profile]\tfamily={family}\tterm_idx={term_idx}"
            f"\trss_gb={rss:.2f}\twall_s={wall:.2f}"
            + (f"\t{extras_str}" if extras_str else ""),
            file=sys.stderr,
            flush=True,
        )

    return _emit


def _cost_classifier(
    type_idx: np.ndarray, types: list[QuantityType],
) -> Any:
    """``ArrayClassifier`` for the objective (cost) walk: ``eff_t ==
    column_type`` (no rhs_type); a ``col_id`` no Var owns has no bucket."""
    return _ArrayClassifier(type_idx, types)


def _matrix_classifier(
    type_idx: np.ndarray,
    has_mp: np.ndarray,
    types: list[QuantityType],
    rhs_t: QuantityType | None,
) -> Any:
    """``ArrayClassifier`` for a matrix family with row type ``rhs_t``.

    Reproduces :func:`_effective_matrix_type`'s branch per column: when the
    column carries a multiplier param AND the family has an ``rhs_type`` the
    entry buckets against the row's type, else against the column's own type.
    A ``col_id`` with no registered family has no bucket (the reducer drops
    it — same as the old inner-join missing the row).
    """
    if rhs_t is None:
        return _ArrayClassifier(type_idx, types)
    keys = list(types)
    if rhs_t in keys:
        rhs_idx = keys.index(rhs_t)
    else:
        rhs_idx = len(keys)
        keys.append(rhs_t)
    bucket = np.where(has_mp & (type_idx >= 0), np.int16(rhs_idx), type_idx)
    return _ArrayClassifier(bucket.astype(np.int16, copy=False), keys)


def _merge_hist_into_accumulator(
    acc: dict[QuantityType, _AccVal],
    hist: dict[QuantityType, tuple[float, int, float, float]],
) -> None:
    """Fold one walked term's :class:`Log2HistogramReducer` result into the
    per-type accumulator.

    The reducer keys directly by :class:`QuantityType` (the classify
    closures return ``QuantityType`` values), and packs each bucket as
    ``(Σlog2|coef|, count, abs_min, abs_max)`` — the SAME packing the
    accumulator carries — so the fold is a direct combine.
    """
    for t, (slog, cnt, amin, amax) in hist.items():
        if t is None or cnt == 0:
            continue
        ps, pn, pmin, pmax = acc.get(t, _INIT_ACC)
        acc[t] = (
            ps + float(slog),
            pn + int(cnt),
            min(pmin, float(amin)),
            max(pmax, float(amax)),
        )


def _obj_term_recipe(term: Any):
    """Return a column-mode ``(recipe, spine)`` for an objective term, or
    ``None`` if the term cannot route through the walk.

    Routes when the term carries a Var seed the column-mode walk can rebuild:
    a non-Sum term (``var_source`` set) or a pure-RELABEL Sum term
    (``sum_block_meta`` set, ``reduce_dims ⊆ var.dims``, no map-effect Where)
    — exactly the regime ``_ranges._obj_chain_bounded`` admits, where every
    ``col_id`` group is single-element so the per-cell product equals the
    reduced coef.  A fully-collapsed ``Sum(over=ALL)`` term (``var_source``
    and ``sum_block_meta`` both cleared) returns ``None`` → the caller keeps
    the existing collect.
    """
    from polar_high.autoscale._coef_walk import CoefWalkRecipe

    meta = getattr(term, "sum_block_meta", None)
    if meta is not None:
        var = meta.var_source
        if var is None:
            return None
        if meta.where_map_frames is not None:
            return None
        if not set(meta.reduce_dims).issubset(set(var.dims)):
            return None
        recipe = CoefWalkRecipe.from_term(term)
        return recipe, var.frame
    var = getattr(term, "var_source", None)
    if var is None:
        return None
    recipe = CoefWalkRecipe.from_term(term)
    return recipe, var.frame


def bucket_coefficients(
    problem: Any,
) -> tuple[
    dict[QuantityType, _AccVal],
    dict[QuantityType, _AccVal],
    dict[QuantityType, _AccVal],
    tuple[np.ndarray, list[QuantityType]],
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
    * ``(type_idx, types)`` — per-``col_id`` column type as an index
      array into ``types`` (``-1`` = no Var owns the column); see
      :func:`_col_type_table`.

    Implementation note (rewrite 2026-05-31, Phase D-5 step 3): per term
    the per-type histogram is accumulated by walking the block-COO
    ``(rid/col_id, coef)`` stream in bounded ``_WALK_BATCH_ROWS`` slices via
    :func:`polar_high.autoscale._coef_walk.bounded_coefficient_walk` +
    :class:`Log2HistogramReducer`, NEVER materialising the merged
    ``Var ⋈ P1 ⋈ P2 …`` product.  Peak RSS is bounded by one batch's product
    (not the full chain) — this is the change that removes the residual DES
    autoscale spike and lets the previously-skipped huge families be bucketed
    at all.  Terms with no recoverable recipe (fully-collapsed
    ``Sum(over=ALL)`` ⇒ ``over is None`` / ``dims == ()``) keep the bounded
    ``_collect_term_agg`` per-term collect as a backstop.

    Capability gate: the bounded walk requires ``polar_high.autoscale.
    _coef_walk`` (block-COO).  When that module is absent (older polar_high,
    detected once at import as ``_HAVE_COEF_WALK``) this falls back to the
    pre-step-3 behaviour — the materialising ``_collect_term_agg`` collect for
    every term, including the >1M per-family size skip — so the solve still
    autoscales correctly (just unbounded) instead of raising ``ImportError``.
    """
    classification = _build_col_id_classification(problem)
    classification_lazy = classification.lazy()

    # ── col_id → column type as numpy arrays.  Built before the capability
    # branch so both paths return it.
    type_idx, has_mp, col_types = _col_type_table(problem)

    matrix_acc: dict[QuantityType, _AccVal] = {}
    cost_acc: dict[QuantityType, _AccVal] = {}
    bound_acc: dict[QuantityType, _AccVal] = {}

    # ── Variable bounds: small (≤ 2 per scalar-bound var family); keep in
    # Python but match the (sum_log2, count, min, max) accumulator shape.
    # Per-element (array) bounds (polar_high ``add_var(upper=Param)``)
    # contribute every finite, non-zero entry — the same filter applied
    # per value.  Bounds never went through the walk, so this is identical
    # on both paths.
    for name, var in problem._vars.items():
        fam = lookup_var(name)
        for b in (var.lower, var.upper):
            if isinstance(b, np.ndarray):
                av_arr = np.abs(b[np.isfinite(b) & (b != 0.0)])
                if av_arr.size == 0:
                    continue
                lv_arr = np.log2(av_arr)
                ok = np.isfinite(lv_arr)
                if not ok.any():
                    continue
                av_arr = av_arr[ok]
                ps, pn, pmin, pmax = bound_acc.get(fam.column_type, _INIT_ACC)
                bound_acc[fam.column_type] = (
                    ps + float(lv_arr[ok].sum()), pn + int(av_arr.size),
                    min(pmin, float(av_arr.min())),
                    max(pmax, float(av_arr.max())),
                )
                continue
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

    if not _HAVE_COEF_WALK:
        # ── Capability fallback: this polar_high has no block-COO
        # ``_coef_walk`` (e.g. released ``main``).  Reproduce the pre-step-3
        # ``bucket_coefficients`` exactly — the materialising
        # ``_collect_term_agg`` collect for every objective and matrix term,
        # including the >1M per-family size skip (driven by the same
        # ``POLAR_HIGH_RANGES_MAX_FAMILY_ROWS`` env var as Layer 1's skip).
        _logger.debug(
            "Layer 2: polar_high lacks _coef_walk; bucketing via the "
            "pre-step-3 per-term collect (unbounded peak, but correct)."
        )
        for term in problem._obj_terms:
            agg = _collect_term_agg(
                term.lazy,
                classification_lazy=classification_lazy,
                rhs_t_id=None,
            )
            if agg is not None:
                _merge_into_accumulator(cost_acc, agg)

        try:
            _max_family_rows = int(
                os.environ.get("POLAR_HIGH_RANGES_MAX_FAMILY_ROWS", "1000000")
            )
        except (ValueError, TypeError):
            _max_family_rows = 1_000_000

        for cname, proto, over in problem._cstrs:
            row_count = 0 if over is None else int(over.height)
            if _max_family_rows > 0 and row_count > _max_family_rows:
                continue
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

        return matrix_acc, cost_acc, bound_acc, (type_idx, col_types)

    # ── Bounded coefficient-walk path (Phase D-5 step 3).  Reached only when
    # this polar_high ships ``_coef_walk`` (gated above at import).
    dense_axes = getattr(problem, "_dense_axes", None)
    profile = _layer2_bucket_profiler()

    from polar_high.autoscale._coef_walk import (
        Log2HistogramReducer,
        bounded_coefficient_walk,
    )

    # ``scale=(None, 0, None)`` — bucketing uses RAW |coef| (no side-vector
    # scaling); the reducer's ``_scaled_abs`` then returns ``|coef|`` verbatim.
    _RAW_SCALE: tuple[Any, int, Any] = (None, 0, None)

    # ── Objective: rhs_t is N/A → eff_t == column_type.  Route each term
    # through the bounded column-mode walk when it carries a Var seed the
    # walk can rebuild; otherwise (fully-collapsed Sum) keep the collect.
    cost_classify = _cost_classifier(type_idx, col_types)
    for ti, term in enumerate(problem._obj_terms):
        if term.lazy is None:
            continue
        routed = _obj_term_recipe(term)
        if routed is not None:
            recipe, spine = routed
            (hist,) = bounded_coefficient_walk(
                spine,
                recipe,
                _RAW_SCALE,
                [Log2HistogramReducer(_RAW_SCALE, cost_classify)],
                batch_rows=_WALK_BATCH_ROWS,
                dense_axes=dense_axes,
            )
            _merge_hist_into_accumulator(cost_acc, hist)
            if profile is not None:
                profile(
                    "<objective>", ti, n=int(spine.height), path="walk",
                )
        else:
            agg = _collect_term_agg(
                term.lazy,
                classification_lazy=classification_lazy,
                rhs_t_id=None,
            )
            if agg is not None:
                _merge_into_accumulator(cost_acc, agg)
            if profile is not None:
                profile("<objective>", ti, path="collect")

    # ── Matrix: per-family walk of the expression terms.
    #
    # NO blanket family-size skip (Phase D-5 step 3): the bounded
    # coefficient walk below caps the per-term peak at one batch's product,
    # so the old >1M skip — which silently dropped the biggest families
    # (e.g. the DES LP's ``profile_flow_upper_limit``, 1.5M rows × multi-
    # Param) from the scaling decision — is no longer needed.  Every family's
    # dim-bound LHS terms are now folded into the histogram.  Terms the walk
    # cannot rebuild (scalar / no-over / fully-collapsed Sum) take the bounded
    # ``_collect_term_agg`` collect; those are small by construction (no deep
    # product to materialise — the Sum already reduced it).
    from polar_high.autoscale._coef_walk import CoefWalkRecipe

    # One classifier per distinct row type (only a few), built on first use.
    matrix_classifiers: dict[QuantityType | None, Any] = {}
    for cname, proto, over in problem._cstrs:
        try:
            rhs_t = resolve_cstr_rhs_type(cname)
        except KeyError as exc:
            raise KeyError(
                f"Layer 2: constraint {cname!r} not in CONSTRAINT_FAMILIES "
                "— register it in _layer2_types.py before solving."
            ) from exc
        rhs_t_id = _qty_to_id(rhs_t)
        matrix_classify = matrix_classifiers.get(rhs_t)
        if matrix_classify is None:
            matrix_classify = _matrix_classifier(
                type_idx, has_mp, col_types, rhs_t,
            )
            matrix_classifiers[rhs_t] = matrix_classify
        for ti, term in enumerate(proto.expr.terms):
            # Route a dim-bound LHS term (real ``over`` grid, open dims, a
            # rebuildable Var/Sum recipe) through the bounded walk; anything
            # else (scalar, no ``over``, fully-collapsed Sum with no recipe)
            # keeps the bounded per-term collect backstop.
            # Routability mirrors ``CoefWalkRecipe.from_term``'s exact
            # precondition via ``is_buildable`` (meta present →
            # ``meta.var_source is not None``; else ``term.var_source is not
            # None``).  The earlier SHALLOW ``var_source or sum_block_meta``
            # check let a fully-collapsed ``Sum`` (meta present, but
            # ``meta.var_source`` None) through, then ``from_term`` raised.
            routable = (
                over is not None
                and bool(term.dims)
                and CoefWalkRecipe.is_buildable(term)
            )
            if routable:
                recipe = CoefWalkRecipe.from_term(term)
                (hist,) = bounded_coefficient_walk(
                    over,
                    recipe,
                    _RAW_SCALE,
                    [Log2HistogramReducer(_RAW_SCALE, matrix_classify)],
                    batch_rows=_WALK_BATCH_ROWS,
                    dense_axes=dense_axes,
                )
                _merge_hist_into_accumulator(matrix_acc, hist)
                if profile is not None:
                    profile(
                        cname, ti, over_height=int(over.height), path="walk",
                    )
            else:
                agg = _collect_term_agg(
                    term.lazy,
                    classification_lazy=classification_lazy,
                    rhs_t_id=rhs_t_id,
                )
                if agg is not None:
                    _merge_into_accumulator(matrix_acc, agg)
                if profile is not None:
                    profile(
                        cname, ti,
                        over_height=(0 if over is None else int(over.height)),
                        path="collect",
                    )

    return matrix_acc, cost_acc, bound_acc, (type_idx, col_types)


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
# Public API


def apply_layer2(
    problem: Any,
    config: ScalingConfig,
) -> Layer2Plan:
    """Apply Layer 2 to ``problem`` in place.

    Mutates ``problem._vars`` (bound rescale on non-integer columns)
    and writes the two side vectors
    ``problem._layer2_col_factor`` / ``problem._layer2_row_factor``
    that the polar-high consumers (``write_mps``, ``_build_lp_arrays``,
    ``_solve_streaming``, ``WarmProblem._initial_build``,
    ``LpView.from_problem``) multiply into the emitted LHS / cost /
    RHS at consumption time.  Also sets ``problem._layer2_locked =
    True`` to prevent post-Layer-2 structural changes that would
    invalidate the side-vector sizes.

    Does NOT mutate ``problem._cstrs`` or ``problem._obj_terms`` —
    the GLPK-style "scaling lives as metadata, coefficients are
    immutable" property.  Returns a :class:`Layer2Plan` carrying the
    inverse transform for :func:`unscale_solution`.
    """
    matrix_acc, cost_acc, bound_acc, _ = bucket_coefficients(problem)
    exponents = choose_scale_powers(matrix_acc, cost_acc, bound_acc)

    # Per-type bucket-range reporting (computed from the accumulators that
    # the walk above produced).  These feed the YAML / console report only;
    # they are NOT needed to install the side vectors, so the
    # exponents-only replay path (:func:`apply_layer2_with_exponents`)
    # passes ``None`` and the plan carries empty buckets.
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

    return apply_layer2_with_exponents(
        problem,
        exponents,
        type_buckets_before=type_buckets_before,
        type_buckets_after=type_buckets_after,
    )


def apply_layer2_with_exponents(
    problem: Any,
    exponents: dict[QuantityType, int],
    *,
    type_buckets_before: dict[QuantityType, tuple[float, float]] | None = None,
    type_buckets_after: dict[QuantityType, tuple[float, float]] | None = None,
) -> Layer2Plan:
    """Install Layer-2 side vectors on ``problem`` from KNOWN exponents.

    This is the cheap second half of :func:`apply_layer2` — it walks
    only ``problem._vars`` (O(#var families)) and ``problem._cstrs``
    (O(#constraint families)) to map the per-type ``exponents`` onto
    column / row factors.  It performs NO coefficient traversal
    (``bucket_coefficients`` / ``detect_ranges`` / ``_ranges_via_streaming``)
    and therefore none of the multi-GB transient working set those walks
    spike.

    Used by the orchestrator's per-roll autoscale cache: the first solve
    of a structural shape runs the full :func:`apply_layer2` (deriving
    ``exponents``); every subsequent same-shape roll replays the decision
    here against THIS roll's freshly-built ``Problem``, producing
    byte-identical scaled coefficients (the side vectors depend only on
    the per-type exponents and the column/row family layout, both of
    which are invariant for a fixed structural fingerprint).

    ``type_buckets_*`` are optional reporting fields; when omitted the
    returned plan carries empty buckets (the per-roll YAML report's
    Layer-2 section is then range-free, but the scaling applied to the LP
    is identical).
    """
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
    # This is the one place Layer 2 mutates state that's not behind the
    # side vectors; intentional because the col factor is uniform per
    # family.  ``Var.scale_bounds`` keeps scalar bounds scalar (the exact
    # historical ``float(b) * f``) and scales per-element arrays entry-wise.
    for name, var in problem._vars.items():
        if var.integer:
            continue
        fam = lookup_var(name)
        f = float(2 ** exponents[fam.column_type]) if fam.column_type in exponents else 1.0
        if f == 1.0:
            continue
        var.scale_bounds(f)

    # ── Row factors ----------------------------------------------------
    # Walk ``_cstrs`` in the same order consumers do; ``row_factors_list``
    # is built 0-based per constraint row.  The cost row is NOT in this
    # vector (objective gets column scaling only — GLPK convention).
    row_factors_list: list[float] = []
    skipped_rows: list[str] = []

    for cname, proto, over in problem._cstrs:
        rhs_t = resolve_cstr_rhs_type(cname)
        if rhs_t is None:
            rf = 1.0
        else:
            exp = exponents.get(rhs_t)
            rf = float(2 ** exp) if exp is not None else 1.0
        row_count = 1 if over is None else int(over.height)
        if rhs_t is None and row_count > 0:
            skipped_rows.append(cname)
        row_factors_list.extend([rf] * row_count)

    row_factors = np.asarray(row_factors_list, dtype=np.float64)

    # ── Install side vectors on the Problem ---------------------------
    # Size assertion: col_factors must be exactly n_cols.  row_factors
    # is indexed 0-based by constraint row in the order consumers walk
    # ``_cstrs`` (the cost row is not in this vector).
    #
    # IMPORTANT — convention asymmetry.  The math (see module docstring):
    #
    #   x_scaled[j] = col_factor[j] * x[j]
    #   matrix_scaled[i,j] = matrix[i,j] * row_factor[i] / col_factor[j]
    #   cost_scaled[j]     = cost[j] / col_factor[j]
    #   rhs_scaled[i]      = rhs[i] * row_factor[i]
    #
    # Consumers (write_mps, _build_lp_arrays, _solve_streaming,
    # WarmProblem._initial_build, LpView.from_problem) multiply emitted
    # values by ``_layer2_row_factor[i]`` and ``_layer2_col_factor[j]``
    # *directly* — no inversion.  For matrix and cost that means the
    # value we install in ``_layer2_col_factor`` is ``1 / col_factor``
    # (the inverse of the math ``cf``), so that ``vals * _cf[j]`` yields
    # the math-correct ``vals / cf[j]``.  ``_layer2_row_factor`` is the
    # forward ``rf`` — applied unchanged to matrix LHS and RHS.
    #
    # ``Layer2Plan.col_factors`` keeps the FORWARD ``cf`` (used by
    # ``unscale_solution`` as ``cv / cf``); only the side vector on the
    # Problem is inverted.  Since every ``cf`` is a power of two,
    # ``1/cf`` is exact in IEEE.
    assert col_factors.shape[0] == problem._next_col, (
        f"Layer 2: col_factors length {col_factors.shape[0]} != "
        f"problem._next_col {problem._next_col}"
    )
    problem._layer2_col_factor = 1.0 / col_factors
    problem._layer2_row_factor = row_factors
    # Lock AFTER writing both arrays — otherwise the writes themselves
    # could trip future guards if they touch any locked code path.
    problem._layer2_locked = True
    # Invalidate any cached canonical matrix on the Problem (polar-high
    # B1, commit a3dd35f).  The side vectors above are baked into
    # ``_matrix.val`` at ``canonicalise()`` time, so an existing cached
    # matrix built BEFORE this call is stale.  No current call path
    # triggers the stale-cache scenario (canonicalise is normally first
    # reached AFTER apply_layer2 via write_mps / _build_lp_arrays), but
    # B2's _build_lp_arrays migration would route every non-streaming
    # solve through canonicalise(), making a sequence like
    # ``write_mps → apply_layer2 → solve`` produce silently wrong
    # results without this flag.
    problem._canonical_dirty = True

    plan = Layer2Plan(
        col_factors=col_factors,
        row_factors=row_factors,
        type_exponents=dict(exponents),
        type_buckets_before=dict(type_buckets_before or {}),
        type_buckets_after=dict(type_buckets_after or {}),
        skipped_rows=skipped_rows,
        skipped_integer_cols=integer_cols,
    )
    return plan


# ---------------------------------------------------------------------------
# Unscale


def unscale_solution(sol: Any, plan: Layer2Plan) -> None:
    """In-place: undo the Layer-2 forward transform on ``sol``.

    Mutates ``sol.col_value``, ``sol.col_dual``, ``sol.row_dual``.  Also
    mirrors the unscaled values onto ``sol.highs`` (the live solver
    handle) so downstream writers that read ``h.getSolution().col_value``
    see physical-coordinate values rather than the scaled solver state.
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
    new_col_value = cv / plan.col_factors
    sol.col_value = new_col_value

    new_col_dual: "np.ndarray | None" = None
    cd = np.asarray(getattr(sol, "col_dual", None), dtype=np.float64) \
        if getattr(sol, "col_dual", None) is not None else None
    if cd is not None and cd.size > 0:
        if cd.shape[0] != plan.col_factors.shape[0]:
            raise ValueError(
                f"Layer 2 unscale: col_dual length {cd.shape[0]} != "
                f"col_factors length {plan.col_factors.shape[0]}"
            )
        new_col_dual = cd * plan.col_factors
        sol.col_dual = new_col_dual

    new_row_dual: "np.ndarray | None" = None
    rd = np.asarray(getattr(sol, "row_dual", None), dtype=np.float64) \
        if getattr(sol, "row_dual", None) is not None else None
    if rd is not None and rd.size > 0:
        if rd.shape[0] != plan.row_factors.shape[0]:
            raise ValueError(
                f"Layer 2 unscale: row_dual length {rd.shape[0]} != "
                f"row_factors length {plan.row_factors.shape[0]}"
            )
        new_row_dual = rd * plan.row_factors
        sol.row_dual = new_row_dual

    _push_unscaled_to_highs(
        sol,
        new_col_value=new_col_value,
        new_col_dual=new_col_dual,
        new_row_dual=new_row_dual,
    )


def _push_unscaled_to_highs(
    sol: Any,
    *,
    new_col_value: "np.ndarray | None" = None,
    new_col_dual: "np.ndarray | None" = None,
    new_row_dual: "np.ndarray | None" = None,
) -> None:
    """Mirror the unscaled values onto ``sol.highs``.

    Downstream output writers (``process_outputs.read_highs_solution``)
    consume ``h.getSolution().col_value`` directly off the solver handle;
    without this push they would see the scaled solver state and write
    physically-meaningless values.  Two handle types appear in practice:

    * The duck-typed ``_SolHighsShim`` (cold HiGHS subprocess path) —
      direct attribute assignment on its ``_SolutionView``.
    * A real :class:`highspy.Highs` (in-process warm path) — round-trip
      through a fresh ``HighsSolution`` pushed via ``setSolution``.

    On a real handle three things need care (items 194, 239):

    * ``setSolution`` clears the basis-valid flag (the statuses stay).  A
      later re-run of the handle after a data change would then start
      cold.  The basis is put back with ``setBasis``: the handle still
      holds the scaled LP the statuses belong to, and a positive diagonal
      row/column scaling keeps every status (basic, at-lower, at-upper).
    * HiGHS recomputes ``col_dual`` (and ``row_value``) from the pushed
      ``row_dual`` with the handle's *scaled* cost and matrix, whatever
      ``col_dual`` the pushed solution carries, so
      ``h.getSolution().col_dual`` is wrong after the push.  The unscaled
      reduced costs are kept on the handle as
      ``_flextool_unscaled_col_dual``; the output readers take them via
      :func:`flextool.process_outputs.read_highs_solution.handle_col_dual`
      while the handle is unsolved since the push (model status
      ``kNotset``).  ``row_value`` is not read by any writer.
    * ``setSolution`` zeroes ``getObjectiveValue()`` (see below).
    """
    h = getattr(sol, "highs", None)
    if h is None:
        return

    try:
        sv = h.getSolution()
    except Exception:
        sv = None

    # Shim path: the ``_SolHighsShim._SolutionView`` is a tiny class
    # whose ``__slots__`` advertise ``col_value`` / ``col_dual`` /
    # ``row_dual``.  Direct assignment makes ``h.getSolution()`` return
    # the unscaled arrays on the next call.
    sv_slots = getattr(sv, "__slots__", None) if sv is not None else None
    if sv_slots is not None and "col_value" in sv_slots:
        if new_col_value is not None:
            sv.col_value = new_col_value
        if new_col_dual is not None:
            sv.col_dual = new_col_dual
        if new_row_dual is not None:
            sv.row_dual = new_row_dual
        return

    # Real highspy.Highs path: round-trip through HighsSolution +
    # setSolution.  ``setSolution`` after ``run`` overwrites the solver's
    # stored solution — pinned by
    # ``tests/engine/autoscale/test_layer2_roundtrip.py``
    # (``test_layer2_unscale_handle_*``).
    #
    # BUG (fixed): ``setSolution`` AFTER ``run`` also resets HiGHS's cached
    # ``getObjectiveValue()`` to 0.0 (verified empirically against highspy
    # 1.14.0).  ``write_v_obj`` reads that very value to derive
    # ``v_obj__{solve}.parquet`` / ``total_cost.val``, so without the
    # rescue path below the post-autoscale objective collapses to zero
    # whenever Layer 2 fires on a warm-path solve (this manifested as
    # the three ``test_commodity_ladder_rolling`` failures in v56).  Stash
    # the pre-setSolution objective on the handle so the writer can
    # prefer it over the zeroed-out cache.
    try:
        import highspy
    except ImportError:  # pragma: no cover — highspy is a hard dep
        return
    try:
        # Capture HiGHS's post-run objective BEFORE setSolution wipes it.
        # ``sol.obj`` may already be the unscaled-by-Layer-2 value
        # (Layer 2 cost-side substitution is c -> c / cf, x -> cf · x,
        # so c^T x is invariant; ``sol.obj`` is left untouched by
        # ``unscale_solution``).  Read straight off the handle so the
        # rescue path captures whatever the live HiGHS just reported,
        # including HiGHS's own user_bound_scale unscale.
        # Plus the objective constant the handle leaves out (polar-high
        # >= 3.14.0 keeps it out of HiGHS; ``Solution.obj_offset``, absent
        # on an older polar-high whose handle includes it).
        pre_obj = float(h.getObjectiveValue()) + float(getattr(sol, "obj_offset", 0.0) or 0.0)
        # ``setSolution`` also resets the model status to ``kNotset`` and the
        # info counters (infeasibilities, primal-dual objective error) to "not
        # computed".  Keep a snapshot of the solve's diagnostics on the
        # Solution, so the accept/reject gate (``classify_acceptance``) judges
        # the solve that ran, not the wiped handle.
        _diag_fn = getattr(sol, "solve_diagnostics", None)
        if callable(_diag_fn):
            try:
                sol.pre_unscale_diagnostics = _diag_fn()
            except Exception:  # pragma: no cover — Solution may forbid setattr
                pass
        # The basis of the solve that ran (statuses of the scaled LP on the
        # handle).  Only a valid one is restored: a crossover-off IPM solve
        # has none worth keeping.
        try:
            basis = h.getBasis()
            basis_valid = bool(basis.valid)
        except Exception:  # pragma: no cover — handle without a basis API
            basis, basis_valid = None, False
        hs = highspy.HighsSolution()
        cv_push = new_col_value if new_col_value is not None else sol.col_value
        hs.col_value = np.asarray(cv_push, dtype=np.float64).tolist()
        hs.value_valid = True
        rd_push = new_row_dual if new_row_dual is not None else getattr(sol, "row_dual", None)
        if rd_push is not None and np.asarray(rd_push).size > 0:
            hs.row_dual = np.asarray(rd_push, dtype=np.float64).tolist()
            hs.dual_valid = True
        h.setSolution(hs)
        if basis is not None and basis_valid:
            try:
                basis.valid = True
                h.setBasis(basis)
            except Exception as exc:  # pragma: no cover — version-specific
                _logger.debug("Layer 2 unscale: basis not restored on the handle: %s", exc)
        # The unscaled reduced costs, which HiGHS cannot hold (it recomputed
        # ``col_dual`` in scaled coordinates inside ``setSolution``).
        cd_push = new_col_dual if new_col_dual is not None else getattr(sol, "col_dual", None)
        if cd_push is not None:
            cd_arr = np.asarray(cd_push, dtype=np.float64)
            if cd_arr.shape[0] == len(hs.col_value):
                try:
                    h._flextool_unscaled_col_dual = cd_arr  # type: ignore[attr-defined]
                except Exception:  # pragma: no cover — handle may forbid setattr
                    pass
        # Re-attach the captured objective so writers (see
        # ``process_outputs.read_highs_solution.write_v_obj``) can
        # bypass the zeroed ``getObjectiveValue()`` cache.
        try:
            h._flextool_unscaled_objective = pre_obj  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover — handle may forbid setattr
            pass
    except Exception:  # pragma: no cover — version-specific highspy quirk
        pass


__all__ = [
    "Layer2Plan",
    "apply_layer2",
    "bucket_coefficients",
    "choose_scale_powers",
    "unscale_solution",
]
