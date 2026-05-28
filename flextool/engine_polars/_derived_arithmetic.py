"""Cluster F — scalar Param-on-Param arithmetic (Δ.10).

Lazy-polars helpers for the "scalar arithmetic" cluster identified in
``audit/native_data_path_design_derived_clusters.md`` §F.  The cluster
groups Params whose source-driven build is a thin composition of the
already-cached primitives (``_entity_unitsize_lf`` /
``_node_unitsize_lf``, ``pss``, ``nodeBalance``, the ``dt`` (d, t)
grid).  They share the same shape: pull a scalar / per-entity value
out of Spine, broadcast over a structural set, optionally apply the
unitsize cascade.

Δ.10 lifts six helpers off ``input.py``'s CSV preprocessing:

* :func:`p_unitsize_from_source`            — ``(p,)`` per-process
  unitsize; cascade ``virtual_unitsize OR existing OR 1000`` filtered
  to processes appearing in ``pss``.
* :func:`p_state_unitsize_from_source`      — ``(n,)`` per-node
  unitsize; same cascade filtered to nodes in ``nodeState``.
* :func:`p_penalty_up_from_source` /
  :func:`p_penalty_down_from_source`        — ``(n, d, t)`` sentinel-
  default scalar broadcast over the node × dt grid.
* :func:`p_process_source_conversion_flow_coeff_from_source` /
  :func:`p_process_sink_conversion_flow_coeff_from_source` —
  ``(p, source)`` / ``(p, sink)`` conversion-flow coefficients with
  zero-drop semantics on the caller's input/output sets.

Architecture invariants (per the Δ.10 hand-off):

1. **Lazy polars throughout.**  Every helper returns a lazy chain
   collected once at the rim.
2. **None-default skip.**  When the parameter has no rows on the
   source side and no scalar default, the helper returns ``None``;
   the caller leaves the field untouched.
3. **No defensive gating.**  Helpers fail loudly if the cascade
   primitives drift; the parity sweep is the oracle.

The cluster F existing helpers (``p_slope``, ``p_section``,
``p_flow_upper_existing``, ``p_state_upper``, ``p_process_existing_count``)
remain in :mod:`._derived_params`; this module hosts only the new Δ.10
helpers — verification of the existing helpers is a code-review pass,
not a re-port.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine_polars._axis_enums import (
    alias_to_axis,
    cast_frame_axes,
    get_global_axis_enums,
    rename_to_axis,
)

from ._derived_params import (
    _entity_unitsize_lf,
    _node_unitsize_lf,
    _try_param,
)

if TYPE_CHECKING:
    from flextool.engine_polars._input_source import InputSource


# ---------------------------------------------------------------------------
# §F.1 — p_unitsize  (per-process unitsize, filtered to pss)
# ---------------------------------------------------------------------------


def p_unitsize_from_source(source: "InputSource",
                              pss: pl.DataFrame | None,
                              ) -> Param | None:
    """``p_unitsize[p]`` — per-process unitsize cascade restricted to
    processes appearing in *pss*.

    Mirrors ``flextool/engine_polars/input.py:800-825``::

        unitsize_long = _read_unitsize(p_entity_unitsize.csv)
        unitsize_p = unitsize_long.filter(p ∈ pss["p"].unique())

    The Spine source for ``p_entity_unitsize`` is the cascade::

        unitsize = virtual_unitsize (if explicitly set non-zero)
                  OR existing       (if explicitly set non-zero)
                  OR 1000.0

    Already lazified in :func:`._derived_params._entity_unitsize_lf`
    (cluster B/C primitive); we filter the result to processes in
    *pss*.  Returns ``None`` when *pss* is empty.
    """
    if pss is None or pss.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None:
        pss = cast_frame_axes(pss, _enums)
    procs = pss.lazy().select(pl.col("p")).unique()
    us_lf = _entity_unitsize_lf(source).pipe(rename_to_axis, {"e": "p"})
    df = (procs
            .join(us_lf, on="p", how="inner")
            .select("p", pl.col("us").alias("value"))
            .sort("p")
            .collect())
    if df.height == 0:
        return None
    return Param(("p",), df)


# ---------------------------------------------------------------------------
# §F.2 — p_state_unitsize  (per-node unitsize, filtered to nodeState)
# ---------------------------------------------------------------------------


def p_state_unitsize_from_source(source: "InputSource",
                                    nodeState_df: pl.DataFrame | None,
                                    ) -> Param | None:
    """``p_state_unitsize[n]`` — per-node unitsize restricted to nodes
    in *nodeState_df*.

    Mirrors ``input.py:1780-1789``::

        state_us_long = unitsize_long.filter(n ∈ nodeState["n"])

    Uses the canonical :func:`._derived_params._node_unitsize_lf`
    cascade.  Returns ``None`` when *nodeState_df* is empty.
    """
    if nodeState_df is None or nodeState_df.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None:
        nodeState_df = cast_frame_axes(nodeState_df, _enums)
    state = nodeState_df.lazy().select(pl.col("n")).unique()
    us_lf = _node_unitsize_lf(source)
    df = (state
            .join(us_lf, on="n", how="inner")
            .select("n", pl.col("us").alias("value"))
            .sort("n")
            .collect())
    if df.height == 0:
        return None
    return Param(("n",), df)


# ---------------------------------------------------------------------------
# §F.3 — p_penalty_up / p_penalty_down  (sentinel-default scalar broadcast)
# ---------------------------------------------------------------------------


def _penalty_param_from_source(source: "InputSource",
                                  parameter_name: str,
                                  nodeBalance_df: pl.DataFrame | None,
                                  dt: pl.DataFrame | None,
                                  ) -> Param | None:
    """Build ``p_penalty_<dir>[n, d, t]`` by broadcasting the per-node
    scalar / Map (period→time) over the (n, d, t) grid restricted to
    nodes in *nodeBalance_df*.

    ``node.penalty_up`` / ``node.penalty_down`` carry a sentinel
    default (e.g. 10000.0) on the schema; the source plugin returns one
    row per node with the scalar (default-broadcast).  The CSV path
    slices ``pdtNode.csv`` already broadcast to (n, d, t); we mirror by
    cross-joining the per-node value against the active-solve dt grid.

    Map-shaped inputs (per-period or per-(period, time) overrides) are
    passed through their own indices and joined on the matching subset
    of (d, t).  Scalar inputs broadcast to every (d, t).
    """
    if (nodeBalance_df is None or nodeBalance_df.height == 0
            or dt is None or dt.height == 0):
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _enums = get_global_axis_enums()
    if _enums is not None:
        nodeBalance_df = cast_frame_axes(nodeBalance_df, _enums)
        dt = cast_frame_axes(dt, _enums)
    df = _try_param(source, "node", parameter_name)
    if df is None or df.height == 0:
        return None
    nb_lf = nodeBalance_df.lazy().select(pl.col("n")).unique()
    # Defensive re-cast: re-cast d/t to canonical Enum so joins below
    # against ``base`` (which alias_to_axis-casts already) match dtype
    # even when ``dt`` arrives with Utf8 d/t.
    dt_lf = (dt.lazy()
                .select(alias_to_axis("d", "d"), alias_to_axis("t", "t"))
                .unique())
    cols = df.columns
    period_col = next((c for c in ("period", "d", "x") if c in cols), None)
    time_col = next((c for c in ("t", "time", "step") if c in cols), None)
    base = df.lazy().select(
        alias_to_axis("name", "n"),
        *([alias_to_axis(period_col, "d")] if period_col else []),
        *([alias_to_axis(time_col, "t")] if time_col else []),
        pl.col("value").cast(pl.Float64),
    )
    if period_col and time_col:
        out_lf = (nb_lf
                    .join(base, on="n", how="inner")
                    .join(dt_lf, on=["d", "t"], how="inner"))
    elif period_col:
        out_lf = (nb_lf
                    .join(base, on="n", how="inner")
                    .join(dt_lf, on="d", how="inner"))
    elif time_col:
        out_lf = (nb_lf
                    .join(base, on="n", how="inner")
                    .join(dt_lf, on="t", how="inner"))
    else:
        out_lf = (nb_lf
                    .join(base, on="n", how="inner")
                    .join(dt_lf, how="cross"))
    out = (out_lf
              .select("n", "d", "t", "value")
              .sort("n", "d", "t")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d", "t"), out)


def p_penalty_up_from_source(source: "InputSource",
                                nodeBalance_df: pl.DataFrame | None,
                                dt: pl.DataFrame | None,
                                ) -> Param | None:
    """``p_penalty_up[n, d, t]`` — broadcast ``node.penalty_up`` over
    nodeBalance × dt.  See :func:`_penalty_param_from_source`.
    """
    return _penalty_param_from_source(source, "penalty_up",
                                          nodeBalance_df, dt)


def p_penalty_down_from_source(source: "InputSource",
                                  nodeBalance_df: pl.DataFrame | None,
                                  dt: pl.DataFrame | None,
                                  ) -> Param | None:
    """``p_penalty_down[n, d, t]`` — broadcast ``node.penalty_down`` over
    nodeBalance × dt.  See :func:`_penalty_param_from_source`.
    """
    return _penalty_param_from_source(source, "penalty_down",
                                          nodeBalance_df, dt)


# ---------------------------------------------------------------------------
# §F.4 — p_process_source_conversion_flow_coeff /
# p_process_sink_conversion_flow_coeff
# ---------------------------------------------------------------------------
#
# Mirrors input.py:950-1002.  The CSV path reads
# p_process_source_conversion_flow_coeff.csv /
# p_process_sink_conversion_flow_coeff.csv (always emitted by flextool's
# input writer for indirect units), then:
#
#   1. Anti-joins zero-coef rows out of the inputs / outputs sets.
#   2. If any non-default (≠ 1.0) non-zero coef remains on a surviving
#      (p, source) / (p, sink) pair, builds a Param keyed on the *full*
#      surviving set (default-fill 1.0 where not listed).  The default-
#      fill is structural: ``v_flow * Param`` would inner-join and drop
#      rows otherwise.
#   3. If every coef is 1.0 (the trivial CHP-base case), the Param is
#      ``None`` — model.py's gate falls through to the no-coef path.
#
# Spine source: ``unit__inputNode.conversion_flow_coeff`` /
# ``unit__outputNode.conversion_flow_coeff``.  The default value on
# the schema is 1.0; values are stored on the relationship class.


def _flow_coef_from_source(source: "InputSource",
                              relationship_class: str,
                              node_role: str,
                              indirect_pairs: pl.DataFrame | None,
                              ) -> tuple[pl.DataFrame | None,
                                          Param | None]:
    """Build (zero_pairs, coef_param) for a relationship's flow
    coefficient.

    *relationship_class* — ``"unit__inputNode"`` or ``"unit__outputNode"``.
    *node_role* — ``"source"`` or ``"sink"`` (the column name to alias
    the node dim to in the output).
    *indirect_pairs* — the caller's surviving (p, source) / (p, sink)
    set (post upstream filters), used to gate Param generation and
    default-fill.

    Returns:
      * ``zero_pairs`` — ``(p, <node_role>)`` rows where coef == 0; the
        caller anti-joins these out of its own inputs / outputs set.
      * ``coef_param`` — ``Param((p, <node_role>), value)`` covering
        *every* surviving pair (default-fill 1.0) iff any non-default,
        non-zero coef is present; otherwise ``None``.
    """
    df = _try_param(source, relationship_class, "conversion_flow_coeff")
    if df is None or df.height == 0:
        return None, None
    cols = df.columns
    # The relationship has two dims: "unit" + "node".  source.parameter
    # returns them as columns of the same names.
    unit_col = "unit" if "unit" in cols else cols[0]
    node_col = "node" if "node" in cols else cols[1]
    base = df.lazy().select(
        alias_to_axis(unit_col, "p"),
        alias_to_axis(node_col, node_role),
        pl.col("value").cast(pl.Float64).alias("coef"),
    )
    zero = (base.filter(pl.col("coef") == 0.0)
                  .select("p", node_role)
                  .collect())
    if zero.height == 0:
        zero = None
    if indirect_pairs is None or indirect_pairs.height == 0:
        return zero, None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None:
        indirect_pairs = cast_frame_axes(indirect_pairs, _enums)
    nondef = base.filter(
        (pl.col("coef") != 0.0) & (pl.col("coef") != 1.0))
    if nondef.collect().height == 0:
        return zero, None
    pair_lf = (indirect_pairs.lazy()
                  .select("p", node_role).unique())
    merged = (pair_lf
                .join(base, on=["p", node_role], how="left")
                .with_columns(pl.col("coef").fill_null(1.0))
                .select("p", node_role, pl.col("coef").alias("value"))
                .sort("p", node_role)
                .collect())
    if merged.height == 0:
        return zero, None
    return zero, Param(("p", node_role), merged)


def p_process_source_conversion_flow_coeff_from_source(
    source: "InputSource",
    process_input_flows: pl.DataFrame | None,
) -> tuple[pl.DataFrame | None, Param | None]:
    """``p_process_source_conversion_flow_coeff`` for indirect units.

    Returns ``(zero_pairs, coef_param)`` mirroring
    ``input.py:_load_indirect``'s contract (lines 950-978).  The
    caller anti-joins ``zero_pairs`` from its inputs set and assigns
    ``coef_param`` to ``flex_data.p_process_source_conversion_flow_coeff``.

    *process_input_flows* — the caller's surviving (p, source) set
    (post upstream zero-drop / classifier filter).  Empty / None →
    return ``(None, None)``.
    """
    return _flow_coef_from_source(
        source, "unit__inputNode", "source", process_input_flows)


def p_process_sink_conversion_flow_coeff_from_source(
    source: "InputSource",
    process_output_flows: pl.DataFrame | None,
) -> tuple[pl.DataFrame | None, Param | None]:
    """``p_process_sink_conversion_flow_coeff`` for indirect units.

    Symmetric counterpart to
    :func:`p_process_source_conversion_flow_coeff_from_source` — see
    that helper's docstring for the contract.
    """
    return _flow_coef_from_source(
        source, "unit__outputNode", "sink", process_output_flows)
