"""Δ.12c-fix2 — Inflow-method scaling cascade for ``p_inflow``.

Native polars-lazy port of
``flextool/flextoolrunner/preprocessing/node_inflow_scaling_params.py``
(~430 LOC procedural — read once, re-implemented here lazy/joined).

Domain
------
``p_inflow`` (``pdtNodeInflow`` in flextool.mod L1280) is a per-(node,
period, t) frame restricted to ``dt`` AND nodes whose ``inflow_method``
is anything BUT ``no_inflow``.  For each (n, d, t) the value is the
**additive sum** over whichever of the four scaling methods are
declared on n:

* ``use_original``                — pass through ``ptNode_inflow[n, t]``.
* ``scale_to_annual_flow``        — multiply by ``period_flow_annual_multiplier[n, d]``.
* ``scale_in_proportion``         — multiply by ``period_flow_proportional_multiplier[n, d]``.
* ``scale_to_annual_and_peak_flow`` — affine: ``new_old_slope[n, d] * pti - new_old_section[n, d]``.

Stochastic branches (``pbt_node_inflow`` fold-in) and parent-period folds
mirror flextool's ``write_pdtNodeInflow`` branches 1 & 2; we honour
those when the source carries a ``pbt_node_inflow`` shape.  For the
default deterministic dispatch fixtures (Δ.12c-fix2 target) only
branch 3 fires.

Scaling parameters
------------------

* ``ptNode_inflow[n, t]``: equals ``node.inflow[n, t]`` when present,
  else the entity's scalar default ``p_node_inflow_default = 0.0``.
* ``period_share_of_annual_flow[n, d]`` =
  ``abs(sum_t ptNode_inflow[n, t]) / annual_flow[n, d]``.
* ``period_flow_annual_multiplier[n, d]`` =
  ``complete_period_share_of_year[d] / period_share_of_annual_flow[n, d]``.
* ``period_flow_proportional_multiplier[n, d]`` =
  ``annual_flow[n, d] / (abs(sum_t ptNode_inflow[n, t]) /
  sum_{tl in period__timeline[d]} p_timeline_duration_in_years[tl])``.
* ``new_peak_sign / old_peak_*`` and ``new_old_slope / new_old_section``:
  see the inline comments in :func:`_compute_peak_scaling`.

The implementation builds every per-(n, d) scaling param as a lazy
frame and joins per-(n, d, t) at the end via a single ``.collect()``.

Caller
------
:func:`apply_p_inflow_with_scaling` is the single integration point.
``_derived_params.apply_derived_a`` calls it BEFORE checking the
fall-through to ``p_inflow_from_source`` (which returns None when any
``scale_to_*`` method is present).  When the scaling cascade succeeds
the result replaces ``flex_data.p_inflow`` and the conditional guard
is bypassed.

Reference
---------
``flextool/flextoolrunner/preprocessing/node_inflow_scaling_params.py``
(read-only mirror).
``flextool.mod`` lines 1395-1453 (peak/flow scaling family),
1280-1336 (pdtNodeInflow domain), 1601-1675 (positive/negative inflow).
``flextool/flextoolrunner/preprocessing/entity_period_calc_params.py:568``
(``write_pdtNodeInflow`` — the canonical additive-sum reference).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from ._input_source import _read_csv_file

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource
    from flextool.engine_polars._per_solve_sets import PerSolveAggregates


__all__ = [
    "apply_p_inflow_with_scaling",
    "p_inflow_with_scaling_from_source",
]


# ---------------------------------------------------------------------------
# Internal helpers (mirror _derived_params._try_param without importing it).
# ---------------------------------------------------------------------------


def _try_param(source: "InputSource", entity_class: str,
               parameter_name: str) -> pl.DataFrame | None:
    try:
        df = source.parameter(entity_class, parameter_name)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


def _scalar_default(source: "InputSource", entity_class: str,
                    parameter_name: str, fallback: float) -> float:
    """Schema default; falls back to *fallback* when source returns None."""
    try:
        d = source.parameter_default(entity_class, parameter_name)
    except KeyError:
        return fallback
    if d is None:
        return fallback
    try:
        return float(d)
    except (TypeError, ValueError):
        return fallback


def _try_entities(source: "InputSource", entity_class: str
                  ) -> pl.DataFrame | None:
    try:
        df = source.entities(entity_class)
    except KeyError:
        return None
    if df.height == 0:
        return None
    return df


# ---------------------------------------------------------------------------
# pdNode-style 1d_map(period) lookup with scalar fallback.
# ---------------------------------------------------------------------------


def _node_period_scalar_lf(source: "InputSource", parameter_name: str,
                            dt_lf: pl.LazyFrame
                            ) -> pl.LazyFrame:
    """Return ``[n, d, value]`` for a ``node.<parameter_name>`` 1d_map(period)
    OR scalar.  Scalar broadcasts across the active periods (taken from
    ``dt_lf.select('d').unique()``).

    Returns an empty 3-col LazyFrame when the parameter is absent / empty.
    """
    raw = _try_param(source, "node", parameter_name)
    schema = {"n": pl.Utf8, "d": pl.Utf8, "value": pl.Float64}
    empty = pl.LazyFrame(schema=schema)
    if raw is None:
        return empty
    cols = raw.columns
    period_col = next((c for c in ("period", "x") if c in cols), None)
    if period_col is not None:
        return (raw.lazy()
                   .select(pl.col("name").alias("n"),
                           pl.col(period_col).alias("d"),
                           pl.col("value").cast(pl.Float64)))
    # Scalar — broadcast across periods.
    periods = dt_lf.select("d").unique()
    return (raw.lazy()
               .select(pl.col("name").alias("n"),
                       pl.col("value").cast(pl.Float64))
               .join(periods, how="cross")
               .select("n", "d", "value"))


# ---------------------------------------------------------------------------
# ptNode_inflow — per-(n, t) raw inflow with scalar fallback.
# ---------------------------------------------------------------------------


def _pt_node_inflow_lf(source: "InputSource",
                        nodes_lf: pl.LazyFrame,
                        time_lf: pl.LazyFrame,
                        ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Return (full ``[n, t, value]`` frame, ``[n, t]`` "explicitly set" set).

    Mirrors ``write_node_inflow_scaling_params``'s ``ptNode_inflow``
    fall-through:

        if (n, t) ∈ node__time_inflow:
            ptNode_inflow[n, t] = pt_node_inflow[n, t]
        else:
            ptNode_inflow[n, t] = p_node_inflow_default[n] = 0.0

    The fall-through is a per-entity scalar default looked up via
    ``node.inflow`` (which doubles as a scalar parameter when no time
    series is supplied).  Schema default for ``inflow`` is ``None`` →
    fallback ``0.0``.
    """
    raw = _try_param(source, "node", "inflow")
    schema = {"n": pl.Utf8, "t": pl.Utf8, "value": pl.Float64}
    empty_explicit_lf = pl.LazyFrame(schema={"n": pl.Utf8, "t": pl.Utf8})
    if raw is None:
        # All zeros over (nodes × time).
        zero_lf = (nodes_lf.join(time_lf, how="cross")
                            .with_columns(value=pl.lit(0.0)))
        return zero_lf.select("n", "t", "value"), empty_explicit_lf
    cols = raw.columns
    if "branch" in cols:
        # Stochastic 3d_map — caller falls through to flextool branch 1/2.
        return None, None  # type: ignore[return-value]
    has_t = "t" in cols
    has_period = "period" in cols
    if has_t and not has_period:
        # 1d_map(t) — entries with non-null t are direct (n, t, value);
        # entries with t=null are scalar broadcasts for that entity (the
        # Spine source plugin emits scalar parameters with all index
        # columns null).  Mirrors flextool preprocessing's mixed-shape
        # handling in entity_period_calc_params.write_pdtNodeInflow:
        # ``ptNode_inflow[n, t]`` falls through to the entity's scalar
        # default when no (n, t) row is present.
        raw_lf = raw.lazy().select(
            pl.col("name").alias("n"),
            pl.col("t"),
            pl.col("value").cast(pl.Float64),
        )
        explicit_lf = raw_lf.filter(pl.col("t").is_not_null())
        scalar_lf = raw_lf.filter(pl.col("t").is_null()) \
                           .select("n", pl.col("value").alias("scalar"))
        # Default broadcast across time for nodes without explicit (n, t).
        full_lf = (nodes_lf.join(time_lf, how="cross")
                           .join(explicit_lf, on=["n", "t"], how="left")
                           .join(scalar_lf, on="n", how="left")
                           .with_columns(
                               value=pl.col("value")
                                       .fill_null(pl.col("scalar"))
                                       .fill_null(0.0))
                           .select("n", "t", "value"))
        # The "explicit" set for peak-domain detection: any (n, t) the
        # source has for that node — including scalar broadcasts (which
        # cover ALL t).
        explicit_t_set = explicit_lf.select("n", "t").unique()
        scalar_n = scalar_lf.select("n").unique()
        scalar_explicit = scalar_n.join(time_lf, how="cross") \
                                   .select("n", "t")
        return (full_lf,
                pl.concat([explicit_t_set, scalar_explicit]).unique())
    if has_period and has_t:
        # 2d_map(period, t) — fold over period (sum) for the (n, t) raw.
        # Used by some stochastic-adjacent fixtures; drop period.
        explicit_lf = (raw.lazy()
                          .select(pl.col("name").alias("n"),
                                  pl.col("t"),
                                  pl.col("value").cast(pl.Float64))
                          .group_by("n", "t").agg(pl.col("value").sum()))
        full_lf = (nodes_lf.join(time_lf, how="cross")
                           .join(explicit_lf, on=["n", "t"], how="left")
                           .with_columns(
                               value=pl.col("value").fill_null(0.0)))
        return (full_lf.select("n", "t", "value"),
                explicit_lf.select("n", "t").unique())
    # Scalar — broadcast across (nodes × time) using per-node value when
    # explicit, else default 0.0.
    explicit_scalar = (raw.lazy()
                          .select(pl.col("name").alias("n"),
                                  pl.col("value").cast(pl.Float64)
                                                  .alias("scalar")))
    full_lf = (nodes_lf.join(time_lf, how="cross")
                       .join(explicit_scalar, on="n", how="left")
                       .with_columns(value=pl.col("scalar").fill_null(0.0))
                       .select("n", "t", "value"))
    explicit_lf = (explicit_scalar.select("n")
                                  .join(time_lf, how="cross")
                                  .select("n", "t"))
    return full_lf, explicit_lf


# ---------------------------------------------------------------------------
# Inflow-method partition: which methods is each node assigned?
# ---------------------------------------------------------------------------


def _inflow_method_lf(source: "InputSource") -> pl.LazyFrame:
    """Return ``[n, method]`` covering every node entity.

    Mirrors
    :func:`flextool.flextoolrunner.preprocessing.method_with_fallback_sets.write_node_inflow_method`:
    nodes with an explicit method use it; nodes WITHOUT an explicit
    method inherit the schema default (``use_original``).  Both the
    explicit and the default-broadcast rows are emitted so the
    downstream additive sum and the default-pdtNodeInflow domain
    (every n NOT in ``no_inflow``) match flextool's pre-existing
    output.

    The schema default for ``node.inflow_method`` is None on the Spine
    side (the Spine source plugin doesn't auto-broadcast); flextool's
    preprocessing (per ``method_with_fallback_sets._INFLOW_METHOD_DEFAULT``)
    fills in ``use_original`` for every node lacking an explicit row.
    We mirror that here.
    """
    raw = _try_param(source, "node", "inflow_method")
    nodes = _try_entities(source, "node")
    schema = {"n": pl.Utf8, "method": pl.Utf8}
    if nodes is None:
        if raw is None:
            return pl.LazyFrame(schema=schema)
        return (raw.lazy()
                   .select(pl.col("name").alias("n"),
                           pl.col("value").alias("method")))
    nodes_lf = nodes.lazy().select(pl.col("name").alias("n"))
    if raw is None:
        # Every node falls back to the schema default.
        return nodes_lf.with_columns(method=pl.lit("use_original"))
    explicit = (raw.lazy()
                   .select(pl.col("name").alias("n"),
                           pl.col("value").alias("method")))
    explicit_n = explicit.select("n").unique() \
                          .with_columns(_has_explicit=pl.lit(True))
    fallback = (nodes_lf
                  .join(explicit_n, on="n", how="left")
                  .filter(pl.col("_has_explicit").fill_null(False).not_())
                  .select(pl.col("n"))
                  .with_columns(method=pl.lit("use_original")))
    return pl.concat([explicit, fallback])


# ---------------------------------------------------------------------------
# Per-(n, d) scaling-parameter computations.
# ---------------------------------------------------------------------------


def _compute_period_share_of_annual_flow(
    pti_lf: pl.LazyFrame,
    annual_flow_lf: pl.LazyFrame,
    method_lf: pl.LazyFrame,
    dt_complete_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """``period_share_of_annual_flow[n, d] =
        abs(sum_{t in dt_complete[d]} ptNode_inflow[n, t]) / annual_flow[n, d]``.

    Domain: (n, d) where n has ``scale_to_annual_flow`` OR
    ``scale_to_annual_and_peak_flow`` AND ``annual_flow[n, d] != 0``.

    Args:
        pti_lf: ``[n, t, value]``.
        annual_flow_lf: ``[n, d, value]``.
        method_lf: ``[n, method]``.
        dt_complete_lf: ``[d, t]`` — complete-time-in-use pairs.
    """
    eligible = (method_lf
                  .filter(pl.col("method").is_in(
                      ["scale_to_annual_flow",
                       "scale_to_annual_and_peak_flow"]))
                  .select("n").unique())
    af_nz = annual_flow_lf.filter(pl.col("value") != 0.0).rename({"value": "af"})
    # Sum ptNode_inflow over (d, t) ∈ dt_complete per (n, d).
    sums = (pti_lf
              .join(dt_complete_lf, on="t", how="inner")
              .group_by("n", "d").agg(pl.col("value").sum().alias("ti_sum")))
    return (eligible
              .join(af_nz, on="n", how="inner")
              .join(sums, on=["n", "d"], how="left")
              .with_columns(ti_sum=pl.col("ti_sum").fill_null(0.0))
              .with_columns(value=pl.col("ti_sum").abs() / pl.col("af"))
              .select("n", "d", "value"))


def _compute_period_flow_annual_multiplier(
    psaf_lf: pl.LazyFrame,
    cpsoy_lf: pl.LazyFrame,
    annual_flow_lf: pl.LazyFrame,
    method_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """``period_flow_annual_multiplier[n, d] =
        complete_period_share_of_year[d] / period_share_of_annual_flow[n, d]``.

    Domain: (n, d) where n has ``scale_to_annual_flow`` AND
    ``annual_flow[n, d] != 0``.

    Returns ``[n, d, value]``.
    """
    eligible = method_lf.filter(pl.col("method") == "scale_to_annual_flow") \
                         .select("n").unique()
    af_nz = annual_flow_lf.filter(pl.col("value") != 0.0).select("n", "d")
    return (eligible
              .join(af_nz, on="n", how="inner")
              .join(psaf_lf.rename({"value": "psaf"}),
                    on=["n", "d"], how="inner")
              .filter(pl.col("psaf") != 0.0)
              .join(cpsoy_lf.rename({"value": "cpsoy"}), on="d", how="inner")
              .with_columns(value=pl.col("cpsoy") / pl.col("psaf"))
              .select("n", "d", "value"))


def _compute_period_flow_proportional_multiplier(
    pti_lf: pl.LazyFrame,
    annual_flow_lf: pl.LazyFrame,
    method_lf: pl.LazyFrame,
    p_tdy_lf: pl.LazyFrame,           # [timeline, value]
    period_timeline_lf: pl.LazyFrame, # [d, timeline]
    time_lf: pl.LazyFrame,            # [t]
) -> pl.LazyFrame:
    """``period_flow_proportional_multiplier[n, d]`` =
        ``annual_flow[n, d] / (abs(sum_t ptNode_inflow[n, t]) / tdy_sum[d])``
    where ``tdy_sum[d] = sum_{tl in period__timeline[d]} p_timeline_duration_in_years[tl]``.

    Domain: (n, d) where n has ``scale_in_proportion`` AND
    ``annual_flow[n, d] != 0`` AND time_sum != 0 AND tdy_sum != 0.
    """
    eligible = method_lf.filter(pl.col("method") == "scale_in_proportion") \
                         .select("n").unique()
    af_nz = annual_flow_lf.filter(pl.col("value") != 0.0).rename({"value": "af"})
    # time_sum: sum_t ptNode_inflow[n, t] over the **full timeline** (all t)
    time_sum = (pti_lf.join(time_lf, on="t", how="inner")
                       .group_by("n").agg(pl.col("value").sum()
                                          .alias("ti_sum")))
    tdy_sum = (period_timeline_lf
                  .join(p_tdy_lf.rename({"value": "tdy"}),
                        on="timeline", how="inner")
                  .group_by("d").agg(pl.col("tdy").sum().alias("tdy_sum")))
    return (eligible
              .join(af_nz, on="n", how="inner")
              .join(time_sum, on="n", how="left")
              .with_columns(ti_sum=pl.col("ti_sum").fill_null(0.0))
              .filter(pl.col("ti_sum") != 0.0)
              .join(tdy_sum, on="d", how="left")
              .with_columns(tdy_sum=pl.col("tdy_sum").fill_null(0.0))
              .filter(pl.col("tdy_sum") != 0.0)
              .with_columns(value=pl.col("af") / (pl.col("ti_sum").abs()
                                                  / pl.col("tdy_sum")))
              .select("n", "d", "value"))


def _compute_peak_scaling(
    pti_lf: pl.LazyFrame,
    annual_flow_lf: pl.LazyFrame,
    peak_inflow_lf: pl.LazyFrame,
    method_lf: pl.LazyFrame,
    explicit_pti_lf: pl.LazyFrame,
    p_node_inflow_default: float,
    cpsoy_lf: pl.LazyFrame,
    dt_complete_lf: pl.LazyFrame,
    time_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Compute ``new_old_slope[n, d]`` and ``new_old_section[n, d]``.

    Domain: (n, d) where n has ``scale_to_annual_and_peak_flow`` AND
    ``annual_flow[n, d] != 0`` AND ``peak_inflow[n, d] != 0``.

    Returns ``[n, d, slope, section]``.
    """
    eligible = method_lf \
        .filter(pl.col("method") == "scale_to_annual_and_peak_flow") \
        .select("n").unique()
    af = annual_flow_lf.filter(pl.col("value") != 0.0).rename({"value": "af"})
    peak = peak_inflow_lf.filter(pl.col("value") != 0.0) \
                          .rename({"value": "peak"})
    base = eligible.join(af, on="n", how="inner") \
                    .join(peak, on=["n", "d"], how="inner")
    # Per-node availability of explicit pt rows (any (n, t) present).
    has_explicit_lf = explicit_pti_lf.select("n").unique() \
                                       .with_columns(has_pti=pl.lit(True))
    # op_max / op_min over the FULL time domain (matches mod L1395-1453's
    # iteration over ``t in time``, not ``dt_complete``).
    pti_full = pti_lf.join(time_lf, on="t", how="inner")
    aggs = pti_full.group_by("n").agg(
        pl.col("value").max().alias("op_max"),
        pl.col("value").min().alias("op_min"),
    )
    # When NO explicit pt row exists for a node, op_max = op_min = scalar
    # default.  Empty-aggs path: substitute the scalar default.
    base_with_aggs = (base
        .join(has_explicit_lf, on="n", how="left")
        .with_columns(has_pti=pl.col("has_pti").fill_null(False))
        .join(aggs, on="n", how="left")
        .with_columns(
            op_max=pl.when(pl.col("has_pti"))
                     .then(pl.col("op_max"))
                     .otherwise(pl.lit(p_node_inflow_default)),
            op_min=pl.when(pl.col("has_pti"))
                     .then(pl.col("op_min"))
                     .otherwise(pl.lit(p_node_inflow_default)),
        )
    )
    # op_sign: when has_pti, +1 if abs(op_max) >= abs(op_min) else -1.
    # When NOT has_pti, +1 if scalar_default >= 0 else -1 (mod L347).
    base_signed = base_with_aggs.with_columns(
        op_sign=pl.when(pl.col("has_pti"))
                  .then(pl.when(pl.col("op_max").abs()
                                >= pl.col("op_min").abs())
                          .then(pl.lit(1.0))
                          .otherwise(pl.lit(-1.0)))
                  .otherwise(pl.when(pl.col("op_max") >= 0.0)
                                  .then(pl.lit(1.0))
                                  .otherwise(pl.lit(-1.0))),
    )
    base_op = base_signed.with_columns(
        old_peak=pl.when(pl.col("op_sign") >= 0.0)
                   .then(pl.col("op_max"))
                   .otherwise(pl.col("op_min"))
    ).filter(pl.col("old_peak") != 0.0)
    # npop = peak / old_peak
    base_npop = base_op.with_columns(
        npop=pl.col("peak") / pl.col("old_peak"),
    )
    # orig_flow_sum: sum_{t in complete_time_in_use} ptNode_inflow[n, t]
    #   (per (n, d) — but the inner sum is over complete_time_in_use,
    #    the period axis only restricts the cross-product domain).
    ofs = (pti_lf.join(dt_complete_lf, on="t", how="inner")
                  .group_by("n", "d").agg(pl.col("value").sum().alias("ofs")))
    base_ofs = (base_npop
                  .join(ofs, on=["n", "d"], how="left")
                  .with_columns(ofs=pl.col("ofs").fill_null(0.0)))
    # npopis = npop * ofs / cpsoy[d] when cpsoy != 0 else 0
    base_npopis = (base_ofs
                     .join(cpsoy_lf.rename({"value": "cpsoy"}),
                           on="d", how="left")
                     .with_columns(cpsoy=pl.col("cpsoy").fill_null(0.0))
                     .with_columns(
                         npopis=pl.when(pl.col("cpsoy") != 0.0)
                                  .then(pl.col("npop") * pl.col("ofs")
                                        / pl.col("cpsoy"))
                                  .otherwise(pl.lit(0.0))))
    # npis = peak_inflow * 8760
    base_npis = base_npopis.with_columns(npis=pl.col("peak") * 8760.0)
    # new_old_multiplier:
    #   denom = npis - npopis;
    #   v = 0 if denom == 0 else op_sign * (op_sign * npopis - af) / denom.
    base_nom = base_npis.with_columns(
        nom=pl.when(pl.col("npis") - pl.col("npopis") != 0.0)
              .then(pl.col("op_sign")
                    * (pl.col("op_sign") * pl.col("npopis") - pl.col("af"))
                    / (pl.col("npis") - pl.col("npopis")))
              .otherwise(pl.lit(0.0)),
    )
    # new_old_slope = npop * (1 + nom)
    # new_old_section = peak_inflow * nom
    return base_nom.with_columns(
        slope=pl.col("npop") * (pl.lit(1.0) + pl.col("nom")),
        section=pl.col("peak") * pl.col("nom"),
    ).select("n", "d", "slope", "section")


# ---------------------------------------------------------------------------
# pbt_node_inflow stochastic / parent-period folds (branch 1 + 2 of
# entity_period_calc_params.write_pdtNodeInflow).  Currently a thin
# placeholder: when the source carries pbt_node_inflow data the helper
# returns None and the caller falls back to the CSV path (preserves
# pre-Δ.12c-fix2 behaviour for stochastic fixtures).
# ---------------------------------------------------------------------------


def _has_pbt_node_inflow(source: "InputSource") -> bool:
    """True if the source carries a ``pbt_node_inflow`` (stochastic) shape.

    Detection heuristic: ``node.inflow`` returns a frame with a
    ``branch`` column (3d_map(period, branch, t)).  In that case we
    can't yet compose the additive sum lazily — stochastic fold-in
    requires the full preprocessing topology (group__node /
    groupIncludeStochastics / first_timesteps / solve_branch__time_branch).
    """
    raw = _try_param(source, "node", "inflow")
    if raw is None:
        return False
    return "branch" in raw.columns


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def _balance_nodes_lf(
    workdir: Path | None,
    balance_set: pl.DataFrame | None,
) -> pl.LazyFrame | None:
    """Return the ``nodeBalance ∪ nodeBalancePeriod`` set as a lazy
    ``[n]`` frame, or ``None`` when the set can't be determined.

    Mirrors flextool's ``write_pdtNodeInflow`` ``balance_union`` filter:
    nodes outside the balance set get ``pdtNodeInflow = 0`` even if they
    have explicit inflow.
    """
    parts: list[pl.LazyFrame] = []
    if balance_set is not None and balance_set.height > 0:
        col = next((c for c in ("n", "node", "name")
                     if c in balance_set.columns), None)
        if col is not None:
            parts.append(balance_set.lazy().select(pl.col(col).alias("n")))
    if workdir is not None:
        for fname in ("nodeBalance.csv", "nodeBalancePeriod.csv"):
            p = Path(workdir) / "solve_data" / fname
            if not p.exists():
                continue
            try:
                df = _read_csv_file(p)
            except Exception:
                continue
            if df.height == 0:
                continue
            col = next((c for c in ("node", "n", "name")
                         if c in df.columns), None)
            if col is None:
                continue
            parts.append(df.lazy().select(pl.col(col).alias("n")))
    if not parts:
        return None
    return pl.concat(parts).unique()


def p_inflow_with_scaling_from_source(
    source: "InputSource",
    dt: pl.DataFrame,
    *,
    workdir: Path | None = None,
    balance_set: pl.DataFrame | None = None,
    per_solve_aggs: "PerSolveAggregates | None" = None,
) -> Param | None:
    """Compute the per-(n, d, t) scaled inflow Param.

    Returns ``None`` when:
      * the source has stochastic ``pbt_node_inflow`` (branch 1/2 of
        ``write_pdtNodeInflow`` — caller falls back to CSV); or
      * the source lacks the timeline scaffolding required for
        ``complete_period_share_of_year`` / ``p_timeline_duration_in_years``
        (preprocessing-only fixtures — out of scope for this helper).

    The returned frame includes one row per (n, d, t) ∈ dt for every
    node whose ``inflow_method`` is anything BUT ``no_inflow``,
    including nodes outside ``nodeBalance`` (mirrors flextool's
    ``pdtNodeInflow`` domain — the model-side `nodeBalance` filter
    happens at constraint-emission time via ``model.py``'s
    ``flow_to_n`` join, not here).

    The value is the additive sum of whichever methods apply per node.
    For nodes with ``use_original``, the result is just
    ``ptNode_inflow[n, t]`` broadcast across the active periods.

    Δ.13: when *per_solve_aggs* is supplied (the new
    :mod:`flextool.engine_polars._per_solve_sets`-derived view of the
    cpsoy / p_tdy / period_timeline frames + dt_complete), the helper
    uses it directly and skips the workdir-CSV path entirely.  The
    workdir path is preserved as a fallback for callers that don't
    pass ``per_solve_aggs``.
    """
    if dt is None or dt.height == 0:
        return None

    if _has_pbt_node_inflow(source):
        # Stochastic — defer to CSV (caller handles None).
        return None

    method_lf = _inflow_method_lf(source)
    method_eager = method_lf.collect()
    if method_eager.height == 0:
        # No explicit method anywhere — caller's simpler
        # ``p_inflow_from_source`` handles use_original-only fixtures.
        return None

    # Drop nodes whose method is exactly 'no_inflow' AND have no other
    # method declared — they're excluded from the pdtNodeInflow domain.
    no_inflow_only = (method_eager
        .group_by("n")
        .agg(pl.col("method").alias("methods"))
        .filter(pl.col("methods").list.contains("no_inflow")
                & (pl.col("methods").list.len() == 1))
        .select("n"))
    eligible_nodes = (method_eager.select("n").unique()
                                   .join(no_inflow_only, on="n", how="anti"))
    if eligible_nodes.height == 0:
        return None

    nodes_lf = eligible_nodes.lazy()
    dt_lf = dt.lazy()
    period_lf = dt_lf.select("d").unique()
    # dt_complete: per-period × full-timeline timesteps.  Used by the
    # ``period_share_of_annual_flow`` sum (mod L1395).
    #
    # Δ.13: when ``per_solve_aggs`` is supplied, use its native
    # dt_complete frame directly (no workdir CSV).  Otherwise prefer the
    # workdir's ``solve_data/steps_complete_solve.csv``; falls back to dt.
    if per_solve_aggs is not None and per_solve_aggs.dt_complete.height > 0:
        dt_complete_lf = per_solve_aggs.dt_complete.lazy()
    else:
        dt_complete_lf = _dt_complete_lf(workdir, dt_lf)
    # time_lf must cover the FULL timeline so ``ptNode_inflow`` is
    # computed over every t (not just active dt).  flextool's
    # preprocessing iterates over ``time`` set (== union of timeline
    # timesteps), not ``time_in_use``.  We use dt_complete's t set as
    # a proxy — it carries the full-period timesteps for every period
    # in use.  When the workdir doesn't supply
    # ``steps_complete_solve.csv``, fall back to dt's t-set.
    time_lf = dt_complete_lf.select("t").unique()

    # ── ptNode_inflow ────────────────────────────────────────────────
    p_node_inflow_default = _scalar_default(source, "node", "inflow", 0.0)
    pt_pair = _pt_node_inflow_lf(source, nodes_lf, time_lf)
    if pt_pair is None or pt_pair[0] is None:
        return None
    pti_lf, explicit_pti_lf = pt_pair

    # ── Annual / peak inflows (1d_map(period) or scalar) ─────────────
    af_lf = _node_period_scalar_lf(source, "annual_flow", dt_lf)
    pk_lf = _node_period_scalar_lf(source, "peak_inflow", dt_lf)

    # Restrict annual / peak to active periods (drops noise).
    af_lf = af_lf.join(period_lf, on="d", how="inner")
    pk_lf = pk_lf.join(period_lf, on="d", how="inner")

    # ── complete_period_share_of_year + p_timeline_duration_in_years ─
    # Δ.13: when ``per_solve_aggs`` is supplied, use its frames directly
    # (no workdir CSV).  Otherwise the legacy two-stage path applies:
    # workdir CSVs first, source-side derivation as fallback.
    if per_solve_aggs is not None:
        cpsoy_lf = per_solve_aggs.complete_period_share_of_year.lazy()
        p_tdy_lf = per_solve_aggs.p_timeline_duration_in_years.lazy()
        period_timeline_lf = per_solve_aggs.period_timeline.lazy()
    else:
        cpsoy_lf, p_tdy_lf, period_timeline_lf = _timeline_aggregates(
            source, workdir, dt_lf,
        )

    # ── Per-(n, d) scaling parameters ─────────────────────────────────

    psaf_lf = _compute_period_share_of_annual_flow(
        pti_lf, af_lf, method_lf, dt_complete_lf,
    )
    pfa_lf = _compute_period_flow_annual_multiplier(
        psaf_lf, cpsoy_lf, af_lf, method_lf,
    )
    pfp_lf = _compute_period_flow_proportional_multiplier(
        pti_lf, af_lf, method_lf, p_tdy_lf, period_timeline_lf, time_lf,
    )
    peak_lf = _compute_peak_scaling(
        pti_lf, af_lf, pk_lf, method_lf, explicit_pti_lf,
        p_node_inflow_default, cpsoy_lf, dt_complete_lf, time_lf,
    )

    # ── Build per-(n, d, t) result ───────────────────────────────────
    # Domain: (n, d, t) ∈ eligible_nodes × dt.  Non-balance-union nodes
    # get pti=0 (mirrors flextool's ``write_pdtNodeInflow`` branch 3
    # gate: ``value = 0.0; if in_balance: value += ...``).
    balance_lf = _balance_nodes_lf(workdir, balance_set)
    base = nodes_lf.join(dt_lf, how="cross") \
                    .join(pti_lf, on=["n", "t"], how="left") \
                    .with_columns(value=pl.col("value").fill_null(0.0)) \
                    .rename({"value": "pti"})
    if balance_lf is not None:
        base = (base.join(balance_lf.with_columns(_in_balance=pl.lit(True)),
                          on="n", how="left")
                    .with_columns(
                        pti=pl.when(pl.col("_in_balance").fill_null(False))
                              .then(pl.col("pti"))
                              .otherwise(pl.lit(0.0)))
                    .drop("_in_balance"))

    method_pivot = (method_lf.with_columns(present=pl.lit(True))
        .collect()
        .pivot(values="present", index="n", on="method",
               aggregate_function="first")
        .lazy())
    method_cols = (method_lf.collect()["method"].unique().to_list())

    # Add boolean columns for each method we care about; missing → False.
    base_m = base.join(method_pivot, on="n", how="left")
    for col in ("use_original", "scale_to_annual_flow", "scale_in_proportion",
                "scale_to_annual_and_peak_flow"):
        if col not in method_cols:
            base_m = base_m.with_columns(pl.lit(False).alias(col))
        else:
            base_m = base_m.with_columns(pl.col(col).fill_null(False))

    # Join the per-(n, d) scaling parameters.
    enriched = (base_m
        .join(pfa_lf.rename({"value": "pfa"}), on=["n", "d"], how="left")
        .join(pfp_lf.rename({"value": "pfp"}), on=["n", "d"], how="left")
        .join(peak_lf, on=["n", "d"], how="left")
        .with_columns(
            pfa=pl.col("pfa").fill_null(0.0),
            pfp=pl.col("pfp").fill_null(0.0),
            slope=pl.col("slope").fill_null(0.0),
            section=pl.col("section").fill_null(0.0),
        ))
    # Additive sum per the four method clauses.
    out_lf = enriched.with_columns(
        value=(
            pl.when(pl.col("scale_to_annual_flow"))
              .then(pl.col("pfa") * pl.col("pti"))
              .otherwise(pl.lit(0.0))
            + pl.when(pl.col("scale_in_proportion"))
                .then(pl.col("pfp") * pl.col("pti"))
                .otherwise(pl.lit(0.0))
            + pl.when(pl.col("scale_to_annual_and_peak_flow"))
                .then(pl.col("slope") * pl.col("pti") - pl.col("section"))
                .otherwise(pl.lit(0.0))
            + pl.when(pl.col("use_original"))
                .then(pl.col("pti"))
                .otherwise(pl.lit(0.0))
        )
    ).select("n", "d", "t", "value").sort("n", "d", "t")
    out = out_lf.collect()
    if out.height == 0:
        return None
    return Param(("n", "d", "t"), out)


# ---------------------------------------------------------------------------
# Workdir-aware aggregates (fall back to recomputing from source).
# ---------------------------------------------------------------------------


def _read_csv_eager(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = _read_csv_file(path)
    except Exception:
        return None
    return df if df.height > 0 else None


def _dt_complete_lf(workdir: Path | None,
                     dt_lf: pl.LazyFrame) -> pl.LazyFrame:
    """``dt_complete = [(d, t)]`` — complete-time-in-use pairs.

    Reads ``solve_data/steps_complete_solve.csv`` when available;
    otherwise falls back to the active ``dt`` (single-solve fixtures
    have ``dt_complete == dt``).
    """
    if workdir is not None:
        p = Path(workdir) / "solve_data" / "steps_complete_solve.csv"
        df = _read_csv_eager(p)
        if df is not None:
            cols = df.columns
            if "period" in cols and "step" in cols:
                return df.lazy().select(pl.col("period").alias("d"),
                                        pl.col("step").alias("t"))
            if "period" in cols and "time" in cols:
                return df.lazy().select(pl.col("period").alias("d"),
                                        pl.col("time").alias("t"))
    return dt_lf.select("d", "t")


def _timeline_aggregates(source: "InputSource",
                           workdir: Path | None,
                           dt_lf: pl.LazyFrame
                           ) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Return (cpsoy_lf [d, value], p_tdy_lf [timeline, value],
    period_timeline_lf [d, timeline]) — all lazy.

    Strategy:

    1. Prefer workdir's preprocessing CSVs when available (cheapest +
       canonical for the active solve).  This covers every dispatch
       fixture in tests/engine_polars/data.
    2. Otherwise derive from ``timeline.timestep_duration`` +
       ``solve.period_timeset`` + ``timeset.timeline``.

    For Δ.12c-fix2 the workdir path is sufficient — every
    ``scale_to_*`` fixture has flextool's preprocessing artefacts.
    The fall-through derivation handles single-solve fixtures whose
    workdir is freshly-materialised by load_flextool's tempdir path.
    """
    # Empty fall-throughs — preserved when neither path produces data.
    empty_cpsoy = pl.LazyFrame(schema={"d": pl.Utf8, "value": pl.Float64})
    empty_tdy = pl.LazyFrame(schema={"timeline": pl.Utf8,
                                       "value": pl.Float64})
    empty_pt = pl.LazyFrame(schema={"d": pl.Utf8, "timeline": pl.Utf8})

    cpsoy = empty_cpsoy
    p_tdy = empty_tdy
    period_timeline = empty_pt

    if workdir is not None:
        # complete_period_share_of_year_calc.csv (per_solve_sets.py output).
        cp_path = Path(workdir) / "solve_data" / "complete_period_share_of_year_calc.csv"
        cp_df = _read_csv_eager(cp_path)
        if cp_df is not None:
            cols = cp_df.columns
            period_col = next((c for c in ("period", "d") if c in cols), None)
            if period_col is not None and "value" in cols:
                cpsoy = cp_df.lazy().select(
                    pl.col(period_col).alias("d"),
                    pl.col("value").cast(pl.Float64))
        # p_timeline_duration_in_years.csv
        tdy_path = Path(workdir) / "solve_data" / "p_timeline_duration_in_years.csv"
        tdy_df = _read_csv_eager(tdy_path)
        if tdy_df is not None:
            cols = tdy_df.columns
            tl_col = next((c for c in ("timeline", "name") if c in cols),
                          None)
            if tl_col is not None and "value" in cols:
                p_tdy = tdy_df.lazy().select(
                    pl.col(tl_col).alias("timeline"),
                    pl.col("value").cast(pl.Float64))
        # period__timeline_set.csv (per_solve_sets.py output).
        pt_path = Path(workdir) / "solve_data" / "period__timeline_set.csv"
        pt_df = _read_csv_eager(pt_path)
        if pt_df is not None:
            cols = pt_df.columns
            if "period" in cols and "timeline" in cols:
                period_timeline = pt_df.lazy().select(
                    pl.col("period").alias("d"),
                    pl.col("timeline"),
                )

    # Source-side fallback: derive cpsoy + p_tdy from timeline.timestep_duration
    # and solve.period_timeset → timeset.timeline.  Only fires when the
    # workdir CSVs are absent (e.g. callers materialising via SpineDbSource
    # from scratch).
    if (cpsoy.collect().height == 0 or p_tdy.collect().height == 0
            or period_timeline.collect().height == 0):
        derived = _derive_timeline_aggregates(source, dt_lf)
        if derived is not None:
            d_cpsoy, d_tdy, d_pt = derived
            if cpsoy.collect().height == 0:
                cpsoy = d_cpsoy
            if p_tdy.collect().height == 0:
                p_tdy = d_tdy
            if period_timeline.collect().height == 0:
                period_timeline = d_pt
    return cpsoy, p_tdy, period_timeline


def _derive_timeline_aggregates(
    source: "InputSource", dt_lf: pl.LazyFrame
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame] | None:
    """Source-side derivation of timeline aggregates when the workdir
    CSVs are absent.

    * ``p_timeline_duration_in_years[tl] = sum_t step_duration[tl, t] / 8760``
    * ``period__timeline[d, tl]`` = unique pairs from
      ``solve.period_timeset[<solve>, d → ts]`` ⨝ ``timeset.timeline[ts → tl]``
      (we don't filter on solve here — the caller's dt_lf already
      restricts (d, t)).
    * ``complete_period_share_of_year[d]`` = sum_t step_duration[tl, t] / 8760
      restricted to (d, t) ∈ dt_lf.

    Returns None if any of the source-side params are missing.
    """
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return None
    cols = tl_dur.columns
    step_col = next((c for c in ("t", "step", "timestep", "x")
                     if c in cols and c not in ("name", "value")), None)
    if step_col is None:
        return None
    tl_lf = (tl_dur.lazy().select(
        pl.col("name").alias("timeline"),
        pl.col(step_col).alias("t"),
        pl.col("value").cast(pl.Float64).alias("step_dur"),
    ))
    p_tdy = (tl_lf.group_by("timeline")
                  .agg((pl.col("step_dur").sum() / 8760.0).alias("value")))

    # period__timeline derivation.
    p_ts = _try_param(source, "solve", "period_timeset")
    ts_tl = _try_param(source, "timeset", "timeline")
    if p_ts is None or ts_tl is None:
        return p_tdy.with_columns(value=pl.col("value")), \
               p_tdy.select("timeline", "value"), \
               pl.LazyFrame(schema={"d": pl.Utf8, "timeline": pl.Utf8})
    period_col = next((c for c in ("period", "x") if c in p_ts.columns),
                      None)
    if period_col is None:
        return None
    pt_lf = p_ts.lazy().select(
        pl.col(period_col).alias("d"),
        pl.col("value").alias("ts"),
    ).unique()
    ttl_lf = ts_tl.lazy().select(
        pl.col("name").alias("ts"),
        pl.col("value").alias("timeline"),
    )
    period_tl_lf = (pt_lf.join(ttl_lf, on="ts", how="inner")
                          .select("d", "timeline").unique())
    # Restrict to active periods.
    period_tl_lf = period_tl_lf.join(dt_lf.select("d").unique(),
                                       on="d", how="inner")
    # cpsoy: sum_t step_duration over (d, t) ∈ dt restricted to the
    # period's timeline.  Equivalent (matches mod L1399):
    #   complete_period_share_of_year[d] =
    #       sum_{t : (d, t) in dt} step_duration[timeline_for_d[d], t] / 8760
    cpsoy = (dt_lf
              .join(period_tl_lf, on="d", how="inner")
              .join(tl_lf, on=["timeline", "t"], how="inner")
              .group_by("d").agg((pl.col("step_dur").sum() / 8760.0)
                                  .alias("value")))
    return cpsoy, p_tdy, period_tl_lf


# ---------------------------------------------------------------------------
# Application — wire into apply_derived_a.
# ---------------------------------------------------------------------------


def apply_p_inflow_with_scaling(
    flex_data: object,
    source: "InputSource",
    workdir: Path | None,
    dt: pl.DataFrame,
    *,
    per_solve_aggs: "PerSolveAggregates | None" = None,
) -> bool:
    """Compute and assign ``flex_data.p_inflow`` via the scaling cascade.

    Returns True iff the helper produced a frame.  Returns False when:
      * the source has stochastic inflow (caller falls back to CSV); or
      * no node has a non-default inflow_method (caller falls back to
        the Γ.3.A ``p_inflow_from_source`` path which handles
        ``use_original``-only fixtures).

    The ``flex_data.nodeBalance`` set (loaded earlier in
    :func:`flextool.engine_polars.input.load_flextool`) is forwarded
    as the ``balance_set`` filter — non-balance nodes get
    ``pti = 0`` per flextool's ``write_pdtNodeInflow`` semantics.

    Non-destructive overlay against the seed
    (``solve_data/pdtNodeInflow.csv`` loaded earlier into
    ``flex_data.p_inflow``): when the helper would zero out a seed
    value because the source has no inflow data for a node BUT the
    seed has it (e.g. fixture-specific CSV-only patches like
    ``_gen_delay_source_coef::_patch_water_sink_demand`` which add
    ``water_sink,inflow,-1.0`` to ``input/p_node.csv`` post-write_input
    without mirroring it to Spine), the seed value survives via a
    per-(n, d, t) MAX(abs) overlay restricted to nodes the source
    doesn't know about.  Mirrors the safe-overlay pattern in
    ``_param_matches`` / Δ.6's ``parameter_explicit`` policy: the
    helper is authoritative for nodes the SOURCE explicitly carries;
    the seed survives for nodes it doesn't.
    """
    nb = getattr(flex_data, "nodeBalance", None)
    p = p_inflow_with_scaling_from_source(source, dt, workdir=workdir,
                                            balance_set=nb,
                                            per_solve_aggs=per_solve_aggs)
    if p is None:
        return False

    # Non-destructive overlay: the helper is authoritative for nodes
    # whose inflow the SOURCE has explicit rows for.  For nodes the
    # source doesn't know about (e.g. fixture-CSV-only injections),
    # restore the seed value via a per-(n, d, t) coalesce.
    seed = getattr(flex_data, "p_inflow", None)
    if seed is not None:
        seed_fr = seed.frame if hasattr(seed, "frame") else seed
        if (seed_fr is not None and seed_fr.height > 0
                and set(seed_fr.columns).issuperset({"n", "d", "t", "value"})):
            inflow_raw = _try_param(source, "node", "inflow")
            authoritative_nodes: list[str] = []
            if inflow_raw is not None:
                authoritative_nodes = (inflow_raw["name"].unique()
                                                          .to_list())
            # Seed rows for non-authoritative nodes.
            non_auth_seed = (seed_fr
                .filter(~pl.col("n").is_in(authoritative_nodes))
                .select("n", "d", "t",
                        pl.col("value").cast(pl.Float64).alias("seed_v")))
            if non_auth_seed.height > 0:
                helper_fr = p.frame
                merged = (helper_fr.lazy()
                            .join(non_auth_seed.lazy(),
                                  on=["n", "d", "t"], how="left")
                            .with_columns(
                                value=pl.when(pl.col("seed_v").is_null())
                                        .then(pl.col("value"))
                                        .otherwise(pl.col("seed_v")))
                            .select("n", "d", "t", "value")
                            .sort("n", "d", "t")
                            .collect())
                p = Param(("n", "d", "t"), merged)
    flex_data.p_inflow = p
    return True
