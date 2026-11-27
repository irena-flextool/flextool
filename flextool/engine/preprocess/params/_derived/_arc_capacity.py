"""§3.3.5 — per-arc capacity: ``p_flow_upper``, the arc capacity coefficients, the indirect input-arc
limit and :func:`wire_arc_capacity_params`."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    cast_frame_axes,
    get_global_axis_enums,
    rename_to_axis,
    schema_dtype,
)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

from flextool.engine.preprocess.params._derived._scope import _enums, _try_entities, _try_param, _try_param_explicit
from flextool.engine.preprocess.params._derived._unitsize import _entity_unitsize_lf
from flextool.engine.preprocess.params._derived._process_method import (
    _METHOD_INDIRECT,
    _classify_process_method,
    _piecewise_flags,
    process_indirect_set,
    process_output_flows,
)
from flextool.engine.preprocess.params._derived._delay import _delay_distributions_from_source


# ---------------------------------------------------------------------------
# §3.3.5 — p_flow_upper  (Δ.26)
# ---------------------------------------------------------------------------
#
# Native port of flextool's
# ``preprocessing/process_arc_unions.py:write_p_flow_max`` (file:line
# `process_arc_unions.py:1469-1624` for `write_p_flow_max`).  Mirrors
# flextool.mod L1661-1677:
#
#     p_flow_max{(p, source, sink, d, t) in peedt} :=
#       if (p, source, sink) in process_source_sink_coeff_zero
#       then p_unconstrained_flow_cap
#       else (
#         if exists{(p, m) in process__method_indirect} 1
#            AND (p, source) in process_source
#         then ( if (p, 'min_load_efficiency') in process__ct_method
#                then pdtProcess_slope[p,d,t] + pdtProcess_section[p,d,t]
#                else pdtProcess_slope[p,d,t]
#              )
#              * (p_entity_dispatch_capacity_max[p,d] / p_entity_unitsize[p])
#              / p_process_source_capacity_max_coeff[p, source]
#         else (p_entity_dispatch_capacity_max[p,d] / p_entity_unitsize[p])
#       )
#       * (if (p, sink) in process_sink
#          then p_process_sink_capacity_max_coeff[p, sink] else 1)
#
# The indirect input-arc branch no longer follows the .mod (whose fuel
# arc was slope × capacity / source coefficient — too tight for
# multi-output units and inverted in the coefficient): it is the input
# limit of :func:`_indirect_input_cap_lf` (``input_share_max`` × the
# full-load fuel per unit of capacity) times the capacity count.
#
# Slow-path consumer: ``flex_data/_load_topology.py::_load_process_topology`` reads
# ``solve_data/p_flow_max.csv`` (preprocessed CSV).  Δ.26 ports the
# derivation natively as a polars-lazy helper, wired into
# :func:`apply_derived_c` so the fast single-solve path produces the
# correct ``p_flow_upper`` Param without any preprocessing.


def _arc_max_capacity_coef_lf(source: "InputSource",
                                  side: str,
                                  *,
                                  explicit: bool = False,
                                  ) -> pl.LazyFrame:
    """Per-(p, node) capacity coefficient of a unit edge.

    ``side='sink'`` → ``unit__outputNode.capacity_max_coeff`` as
    ``[p, sink, coef]``; ``side='source'`` → ``unit__inputNode.
    input_share_max`` (schema v70; was ``capacity_max_coeff``) as
    ``[p, source, coef]``.  ``explicit=True`` returns only authored rows
    (no schema-default broadcast).

    Connections don't carry these parameters (the per-arc coefficient is
    unit-only; connections always default to 1 — flextool.mod L686-687),
    so only unit-arc rows are emitted; callers fill the default 1.0 via
    left-join.
    """
    if side == "source":
        ec = "unit__inputNode"
        pname = "input_share_max"
        node_alias = "source"
    else:
        ec = "unit__outputNode"
        pname = "capacity_max_coeff"
        node_alias = "sink"
    df = (_try_param_explicit(source, ec, pname) if explicit
          else _try_param(source, ec, pname))
    if df is None or df.height == 0:
        from flextool.engine.preprocess.params._projection import remap_capacity_coef
        return remap_capacity_coef(source, pl.LazyFrame(
            schema={"p": schema_dtype(_enums, "p"),
                    node_alias: schema_dtype(_enums, node_alias),
                    "coef": pl.Float64}), side, node_alias, "coef")
    cols = df.columns
    rename: dict[str, str] = {}
    for c in cols:
        if c == "unit":
            rename[c] = "p"
        elif c == "node":
            rename[c] = node_alias
    out = (df.lazy().pipe(rename_to_axis, rename)
             .with_columns(pl.col("value").cast(pl.Float64).alias("coef"))
             .select("p", node_alias, "coef")
             .unique(subset=["p", node_alias], keep="last"))
    # One-input piecewise_linear units: their input_share_max is an output cap (_projection.remap_capacity_coef).
    from flextool.engine.preprocess.params._projection import remap_capacity_coef
    return remap_capacity_coef(source, out, side, node_alias, "coef")


def _arc_conversion_coef_lf(source: "InputSource",
                               side: str,
                               ) -> pl.LazyFrame:
    """Per-(p, node) ``conversion_flow_coeff`` from ``unit__inputNode``
    (``side='source'`` → ``[p, source, coef]``) or ``unit__outputNode``
    (``side='sink'`` → ``[p, sink, coef]``).  Only authored rows are
    returned; callers fill the default 1.0 via left-join.
    """
    ec = "unit__inputNode" if side == "source" else "unit__outputNode"
    df = _try_param(source, ec, "conversion_flow_coeff")
    if df is None or df.height == 0:
        return pl.LazyFrame(
            schema={"p": schema_dtype(_enums, "p"),
                    side: schema_dtype(_enums, side),
                    "coef": pl.Float64})
    rename = {c: ("p" if c == "unit" else side)
              for c in df.columns if c in ("unit", "node")}
    return (df.lazy().pipe(rename_to_axis, rename)
              .with_columns(pl.col("value").cast(pl.Float64).alias("coef"))
              .select("p", side, "coef")
              .unique(subset=["p", side], keep="last"))


def _delay_fuel_factor_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-process ``[p, delay_factor]`` = ``1 / max_td(delay_weight)``
    for delayed processes (a scalar ``delay`` has weight 1 → factor 1).

    One input step of a delayed process feeds several output steps with
    weights ``w_td``; with every output at its cap, a single input step
    can carry up to ``slope · Σ out / max(w_td)``.  Processes without a
    delay are absent (callers default to 1).
    """
    empty = pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                 "delay_factor": pl.Float64})
    dist = _delay_distributions_from_source(source)
    if dist is None:
        return empty
    weighted, single, processes = dist
    max_w: dict[str, float] = {p: 1.0 for p, _ in single}
    for (p, _), w in weighted.items():
        max_w[p] = max(max_w.get(p, 0.0), w)
    factors = {p: 1.0 / w for p, w in max_w.items()
               if p in processes and w > 0.0}
    if not factors:
        return empty
    return (pl.LazyFrame({"p": list(factors),
                          "delay_factor": list(factors.values())})
              .with_columns(alias_to_axis(pl.col("p"), "p")))


def _process_source_pairs_lf(source: "InputSource") -> pl.LazyFrame:
    """``process_source`` set ``[p, source]`` — union of unit input arcs
    and connection's first node.  Mirrors flextool.mod L686 + the
    expansion in ``input_writer.write_topology``."""
    parts: list[pl.LazyFrame] = []
    uin = _try_entities(source, "unit__inputNode")
    if uin is not None and uin.height > 0:
        parts.append(uin.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("node", "source"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None and cnn.height > 0:
        parts.append(cnn.lazy().select(
            alias_to_axis("connection", "p"),
            alias_to_axis("node_1", "source"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                     "source": schema_dtype(_enums, "source")})
    return pl.concat(parts).unique()


def _process_sink_pairs_lf(source: "InputSource") -> pl.LazyFrame:
    """``process_sink`` set ``[p, sink]`` — union of unit output arcs
    and connection's second node.  Mirrors flextool.mod L687."""
    parts: list[pl.LazyFrame] = []
    uout = _try_entities(source, "unit__outputNode")
    if uout is not None and uout.height > 0:
        parts.append(uout.lazy().select(
            alias_to_axis("unit", "p"),
            alias_to_axis("node", "sink"),
        ))
    cnn = _try_entities(source, "connection__node__node")
    if cnn is not None and cnn.height > 0:
        parts.append(cnn.lazy().select(
            alias_to_axis("connection", "p"),
            alias_to_axis("node_2", "sink"),
        ))
    if not parts:
        return pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                     "sink": schema_dtype(_enums, "sink")})
    return pl.concat(parts).unique()


def _process_source_sink_coeff_zero_lf(source: "InputSource",
                                            pss: pl.DataFrame,
                                            ) -> pl.LazyFrame:
    """``process_source_sink_coeff_zero`` set: rows of pss whose source-
    or sink-side ``conversion_flow_coeff`` is explicitly zero.

    A zero conversion coefficient takes the edge out of the conversion
    equation and out of every per-edge capacity / ramp / min-load
    constraint (the hydro-pass-through pattern); the edge's flow is
    bounded only by ``max_flow_for_unconstrained_variables``.  The set
    used to be keyed on ``capacity_max_coeff == 0`` — an artefact of the
    v35 rename of the old ``coefficient`` (which was the flow
    coefficient).  ``capacity_max_coeff = 0`` now means what its name
    says: a zero cap on the edge.
    """
    if pss is None or pss.height == 0:
        return pl.LazyFrame(schema={
            "p": schema_dtype(_enums, "p"),
            "source": schema_dtype(_enums, "source"),
            "sink": schema_dtype(_enums, "sink")})
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
    src_zero = (_arc_conversion_coef_lf(source, "source")
                  .filter(pl.col("coef") == 0.0)
                  .select("p", "source"))
    sink_zero = (_arc_conversion_coef_lf(source, "sink")
                   .filter(pl.col("coef") == 0.0)
                   .select("p", "sink"))
    src_match = (pss.lazy()
                   .join(src_zero, on=["p", "source"], how="inner")
                   .select("p", "source", "sink"))
    sink_match = (pss.lazy()
                    .join(sink_zero, on=["p", "sink"], how="inner")
                    .select("p", "source", "sink"))
    return pl.concat([src_match, sink_match]).unique()


def process_source_sink_uncapped_from_source(source: "InputSource",
                                              pss: pl.DataFrame | None,
                                              ) -> pl.DataFrame | None:
    """``(p, source, sink)`` arcs with ``conversion_flow_coeff = 0`` on
    either side — the uncapped (hydro-pass-through) edges.

    The model removes these arcs from every per-edge capacity / ramp /
    min-load constraint; their ``p_flow_upper`` is the unconstrained
    ``max_flow_for_unconstrained_variables`` value.  ``None`` when there
    are none.
    """
    if pss is None or pss.height == 0:
        return None
    out = (_process_source_sink_coeff_zero_lf(source, pss)
           .sort("p", "source", "sink")
           .collect())
    return out if out.height > 0 else None


def p_arc_max_cap_coef_from_source(source: "InputSource",
                                      pss: pl.DataFrame,
                                      classified: pl.DataFrame | None = None,
                                      ) -> "Param | None":
    """Per-(p, source, sink) ``capacity_max_coeff`` factor for **direct**
    arcs and indirect **output** arcs — the multiplier the ``maxFlow`` RHS drops when it uses
    ``p_flow_upper_existing`` (which is ``existing_count`` = capacity / unitsize only).

    Mirrors flextool.mod ``maxToSink`` / ``maxFromSource``:

    * ``maxToSink`` (``process__source__sinkIsNode``, the normal output
      arc) multiplies the existing-capacity RHS by
      ``p_process_sink_max_capacity_coefficient[p, sink]`` whenever
      ``(p, sink) ∈ process_sink``.  We carry that as ``sink_coef`` =
      ``unit__outputNode.capacity_max_coeff``.
    * ``maxFromSource`` (``process__sourceIsNode__sink_1way_noSink…``, a
      1-way unit whose node is the *source* and that has no output node)
      multiplies by ``p_process_source_max_capacity_coefficient[p,
      source]``.  The engine emits such an arc as a synthetic
      ``(p, source, p)`` row whose ``sink`` is the process itself, so
      ``(p, sink) ∉ process_sink``; we carry that as ``src_coef`` =
      ``unit__inputNode.input_share_max`` (schema v70; was
      ``capacity_max_coeff`` — same numbers).
    * A direct unit with an input AND an output node has a single flow
      variable (output units; its input is ``slope × v_flow``).  An
      AUTHORED ``input_share_max`` on its input caps that input at
      ``share × full-load input``, i.e. ``v_flow ≤ share × capacity``;
      combined with the output's ``capacity_max_coeff`` the arc factor is
      ``min(capacity_max_coeff_out, input_share_max_in)``.  The schema
      default (1, "no effective limit") is not applied, so an output
      coefficient above 1 is left untouched when the share is unset.

    Indirect (CHP / multi-flow) processes contribute only their OUTPUT
    arcs (``process_output_flows``): ``model/core/`` binds those with the
    existing-only ``maxToSink`` RHS (``p_flow_upper_existing``), so their
    ``capacity_max_coeff`` must be folded here or it is silently lost.
    Indirect input arcs stay excluded — their limit is
    :func:`p_indirect_input_cap_from_source` (``input_share_max`` × the
    full-load fuel per unit of capacity).  ``conversion_flow_coeff = 0`` arcs (direct or
    indirect, :func:`process_source_sink_uncapped_from_source`) are
    uncapped and excluded too.

    Returns a Param keyed ``(p, source, sink)`` carrying only the direct
    arcs whose effective coefficient differs from the schema default
    (1.0); absent arcs default to 1.0 at the consumer's densify.  Returns
    ``None`` when there is nothing to carry (no authored coefficient, or
    every authored row resolves to 1.0) so the consumer skips the fold.
    """
    if pss is None or pss.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)

    # Direct arcs + indirect OUTPUT arcs (see docstring).
    indirect_p = process_indirect_set(source, classified)
    direct_pss = (pss.lazy()
                    .select("p", "source", "sink")
                    .join(indirect_p.lazy().with_columns(_ind=pl.lit(True)),
                          on="p", how="left")
                    .filter(pl.col("_ind").is_null())
                    .select("p", "source", "sink"))
    out_flows = process_output_flows(source, pss, classified)
    if out_flows.height > 0:
        direct_pss = pl.concat([
            direct_pss,
            out_flows.lazy().select("p", "source", "sink"),
        ])
    # conversion_flow_coeff = 0 edges are uncapped (coeff_zero): model/core/
    # binds them with the unconstrained ``p_flow_upper`` value, so no
    # capacity coefficient applies to them.
    direct_pss = direct_pss.join(
        _process_source_sink_coeff_zero_lf(source, pss),
        on=["p", "source", "sink"], how="anti")

    sink_pairs = _process_sink_pairs_lf(source)
    src_coef_lf = _arc_max_capacity_coef_lf(source, "source")
    src_share_explicit_lf = _arc_max_capacity_coef_lf(
        source, "source", explicit=True)
    sink_coef_lf = _arc_max_capacity_coef_lf(source, "sink")

    out = (direct_pss
        # Tag (p, sink) ∈ process_sink — selects maxToSink vs maxFromSource.
        .join(sink_pairs.with_columns(_has_sink=pl.lit(True)),
              on=["p", "sink"], how="left")
        .with_columns(_has_sink=pl.col("_has_sink").fill_null(False))
        # Source-side input_share_max (default 1.0).
        .join(src_coef_lf.rename({"coef": "src_coef"}),
              on=["p", "source"], how="left")
        .with_columns(src_coef=pl.col("src_coef").fill_null(1.0))
        # Authored input_share_max only (direct units with an output).
        .join(src_share_explicit_lf.rename({"coef": "src_share"}),
              on=["p", "source"], how="left")
        # Sink-side capacity_max_coeff (default 1.0).
        .join(sink_coef_lf.rename({"coef": "sink_coef"}),
              on=["p", "sink"], how="left")
        .with_columns(sink_coef=pl.col("sink_coef").fill_null(1.0))
        .with_columns(
            value=pl.when(pl.col("_has_sink")
                          & pl.col("src_share").is_not_null())
                     # maxToSink of a direct unit with an authored
                     # input share: both caps apply to the one variable.
                     .then(pl.min_horizontal(pl.col("sink_coef"),
                                             pl.col("src_share")))
                     .when(pl.col("_has_sink"))
                     .then(pl.col("sink_coef"))     # maxToSink
                     .otherwise(pl.col("src_coef")))  # maxFromSource
        # Only carry rows that actually deviate from the 1.0 default; the
        # consumer fills absent (p, source, sink) with 1.0.
        .filter(pl.col("value") != 1.0)
        .select("p", "source", "sink", "value")
        .sort("p", "source", "sink")
        .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink"), out)


def wire_arc_capacity_params(flex_data, source: "InputSource") -> None:
    """Set the per-arc capacity Params ``model/core/``'s ``maxFlow`` needs:
    ``p_arc_max_cap_coef``, ``process_source_sink_uncapped`` and
    ``p_indirect_input_cap``.

    For the synthetic sub-solve path (rolling rolls ``<solve>_roll_<n>``,
    nested per-period sub-solves) that skips ``apply_derived_b..g``: all
    three producers are solve-agnostic (they read unit / edge parameters,
    the process topology, ``dt`` and the CSV-seeded ``p_slope`` /
    ``p_section``), so the result equals the full-cascade path.  Without
    this the roll sub-solves dropped the output ``capacity_max_coeff``,
    treated ``conversion_flow_coeff = 0`` edges as capped, and left the
    indirect input arcs without their limit.
    """
    pss = getattr(flex_data, "process_source_sink", None)
    if pss is None or pss.height == 0:
        return
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
    classified = _classify_process_method(source)
    flex_data.p_arc_max_cap_coef = p_arc_max_cap_coef_from_source(
        source, pss, classified)
    flex_data.process_source_sink_uncapped = (
        process_source_sink_uncapped_from_source(source, pss))
    flex_data.p_indirect_input_cap = p_indirect_input_cap_from_source(
        source, pss, getattr(flex_data, "dt", None),
        p_slope=getattr(flex_data, "p_slope", None),
        p_section=getattr(flex_data, "p_section", None),
        classified=classified)


def p_flow_upper_from_source(source: "InputSource",
                                pss: pl.DataFrame,
                                dt: pl.DataFrame,
                                p_slope: "Param | None",
                                p_section: "Param | None",
                                p_unitsize: "Param | None",
                                p_entity_max_units: "Param | None",
                                p_entity_all_existing: "Param | None",
                                classified: pl.DataFrame | None = None,
                                piecewise_lines: pl.DataFrame | None = None,
                                ) -> "Param | None":
    """§3.3.5 — per-(p, source, sink, d, t) structural max flow.

    Native port of flextool's
    ``preprocessing/process_arc_unions.py:write_p_flow_max``
    (file:line `process_arc_unions.py:1469-1624`).  Returns a Param
    keyed on ``(p, source, sink, d, t)`` whose values match
    ``solve_data/p_flow_max.csv`` written by the slow path.

    Inputs:

    ``pss`` — ``process_source_sink`` set; the (p, source, sink) index
    of the output Param.  Cross-joined with ``dt`` to form ``peedt``.

    ``p_slope`` / ``p_section`` — per-(p, d, t) slope / section terms
    from :func:`p_slope_from_source` / :func:`p_section_from_source`.
    Used in the indirect-method branch.

    ``p_unitsize`` — per-(p,) unitsize cascade from
    :func:`_arithmetic.p_unitsize_from_source`.  Caller may
    also pass the per-entity cascade keyed on ``(e,)``; we accept either
    column name.

    ``p_entity_max_units`` — per-(e, d) ``max_capacity / unitsize`` from
    :func:`p_entity_max_units_from_source`.  Provides
    ``p_entity_dispatch_capacity_max[p, d] / p_entity_unitsize[p]``
    directly, so we don't need to recompute the dispatch cap inline.

    ``piecewise_lines`` — the lines ``(p, d, seg, slope, intercept)`` of the
    multi-line ``piecewise_linear`` units (see :func:`_indirect_input_cap_lf`);
    None without such units.

    ``p_entity_all_existing`` — fallback when ``p_entity_max_units`` is
    absent (no invest configured); when both invest sets are empty
    flextool's preprocessing still emits ``p_flow_max`` rows valued at
    the existing unit count (capacity/unitsize) per arc.  This matches the slow path's
    behaviour in the no-invest case.

    Returns ``None`` when no usable inputs exist (empty pss or dt) —
    caller leaves the Param at its empty placeholder.
    """
    if pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return None

    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
        dt = cast_frame_axes(dt, _live)
        if classified is not None:
            classified = cast_frame_axes(classified, _live)

    # ── 1. Resolve the (p, d) cap-per-unitsize term ─────────────────
    # Prefer p_entity_max_units (= dispatch_capacity_max / unitsize per
    # (e, d)).  When absent (no invest in fixture), derive it from
    # p_entity_all_existing / p_unitsize.
    if p_entity_max_units is not None and p_entity_max_units.frame.height > 0:
        cap_per_unit_lf = (p_entity_max_units.frame.lazy()
            .pipe(rename_to_axis, {"e": "p"})
            .select("p", alias_to_axis("d", "d"), pl.col("value").alias("cap_per_unit")))
    else:
        # Derive from existing capacity / unitsize.  We only need rows that join
        # against pss, so we filter inline.
        if p_entity_all_existing is None or p_entity_all_existing.frame.height == 0:
            return None
        # p_unitsize may come keyed on (p,) (post-Δ.4b override) or (e,)
        # (raw cascade); accept both.
        if p_unitsize is None or p_unitsize.frame.height == 0:
            us_df = _entity_unitsize_lf(source).pipe(rename_to_axis, {"e": "p", "us": "us"}).collect()
        else:
            us_cols = p_unitsize.frame.columns
            us_key = "p" if "p" in us_cols else "e"
            us_df = p_unitsize.frame.pipe(rename_to_axis, {us_key: "p", "value": "us"})
        cap_per_unit_lf = (
            p_entity_all_existing.frame.lazy()
                .pipe(rename_to_axis, {"e": "p", "value": "cap"})
                .join(us_df.lazy(), on="p", how="left")
                .with_columns(
                    cap_per_unit=pl.when((pl.col("us").is_not_null())
                                          & (pl.col("us") != 0.0))
                                    .then(pl.col("cap") / pl.col("us"))
                                    .otherwise(0.0))
                .select("p", alias_to_axis("d", "d"), "cap_per_unit")
        )

    # ── 2. Indirect input (fuel) arc limit per unit of capacity ─────
    # (p, source, sink, d, t) for the input arcs of indirect units —
    # ``input_share_max × full-load fuel / src_conv``; see
    # :func:`_indirect_input_cap_lf`.  p_flow_upper on those arcs is that
    # limit times the structural capacity count.
    if classified is None:
        classified = _classify_process_method(source)
    in_cap_lf = _indirect_input_cap_lf(source, pss, dt, p_slope, p_section,
                                       classified, piecewise_lines)
    process_sink_lf = _process_sink_pairs_lf(source)

    # ── 3. Coefficient frames ───────────────────────────────────────
    sink_coef_lf = _arc_max_capacity_coef_lf(source, "sink")

    # ── 4. p_unconstrained_flow_cap ────────────────────────────────
    # DB stores the model param as ``max_flow_for_unconstrained_variables``
    # (no ``p_`` prefix — that prefix is only the CSV-emit alias).
    # The largest value over the model entities, used as given (also below
    # 1e6); 1e6 when none is set.
    p_unc = 1_000_000.0
    df_unc = _try_param(source, "model", "max_flow_for_unconstrained_variables")
    if df_unc is not None and df_unc.height > 0:
        try:
            p_unc = float(df_unc["value"].max())
        except Exception:
            pass

    # ── 5. coeff_zero set — rows that get the unconstrained value ─
    coeff_zero_lf = _process_source_sink_coeff_zero_lf(source, pss)

    # ── 6. Build peedt = pss × dt with cap-per-unit and arc tags ────
    pss_lf = pss.lazy().select("p", "source", "sink")
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # against pss (Enum p / source / sink) — keeps the (p, source, sink,
    # d, t) base frame Enum-typed across all axes.
    base = (pss_lf
        .join(dt.lazy()
                 .with_columns(alias_to_axis(pl.col("d"), "d"),
                               alias_to_axis(pl.col("t"), "t")),
              how="cross")  # adds d, t
        .join(cap_per_unit_lf, on=["p", "d"], how="left")
        .with_columns(
            cap_per_unit=pl.col("cap_per_unit").fill_null(0.0))
        # Indirect input arcs: fuel limit per unit of capacity.
        .join(in_cap_lf.rename({"value": "_in_cap"}),
              on=["p", "source", "sink", "d", "t"], how="left")
        # Sink-side capacity_max_coeff (default 1.0); also tag
        # (p, sink) ∈ process_sink (the .mod multiplies by sink_coef
        # only when (p, sink) ∈ process_sink, defaulting to 1 outside).
        .join(process_sink_lf.with_columns(_has_sink=pl.lit(True)),
                on=["p", "sink"], how="left")
        .with_columns(
            _has_sink=pl.col("_has_sink").fill_null(False))
        .join(sink_coef_lf.rename({"coef": "sink_coef"}),
                on=["p", "sink"], how="left")
        .with_columns(
            sink_coef=pl.col("sink_coef").fill_null(1.0))
        # Mark coeff_zero rows.
        .join(coeff_zero_lf.with_columns(_coeff_zero=pl.lit(True)),
                on=["p", "source", "sink"], how="left")
        .with_columns(
            _coeff_zero=pl.col("_coeff_zero").fill_null(False))
    )

    # ── 7. Compute the formula. ─────────────────────────────────────
    # indirect input arc: cap_per_unit · input-arc limit per capacity;
    # everything else:    cap_per_unit.
    # Multiply by sink_coef when (p, sink) ∈ process_sink, else *1.
    base = base.with_columns(
        _branch_value=pl.when(pl.col("_in_cap").is_not_null())
                          .then(pl.col("cap_per_unit") * pl.col("_in_cap"))
                          .otherwise(pl.col("cap_per_unit")),
    ).with_columns(
        # sink-coef multiplier — only when (p, sink) ∈ process_sink.
        _sink_factor=pl.when(pl.col("_has_sink"))
                          .then(pl.col("sink_coef"))
                          .otherwise(1.0),
    ).with_columns(
        value=pl.when(pl.col("_coeff_zero"))
                  .then(pl.lit(p_unc))
                  .otherwise(pl.col("_branch_value") * pl.col("_sink_factor")),
    )

    out = (base
        .select("p", "source", "sink", "d", "t", "value")
        .sort("p", "source", "sink", "d", "t")
        .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), out)


def _indirect_fuel_width_lf(source: "InputSource") -> pl.LazyFrame:
    """Per-process ``[p, fuel_width]`` = ``W``: the most output-side fuel
    energy per unit of capacity that the outputs can demand.

    ``conversion_indirect`` reads ``Σ_s src_conv_s·in_s = slope ·
    Σ_k conv_k·out_k (+ section · online)``.  Capacity caps the SUM of
    the outputs (``Σ_k out_k ≤ cap``, ``maxOutputSum``) and each output
    individually (``out_k ≤ capacity_max_coeff_k · cap``), so per unit of
    capacity ``W = max Σ_k conv_k·x_k`` s.t. ``Σ_k x_k ≤ 1``,
    ``0 ≤ x_k ≤ capacity_max_coeff_k`` — solved greedily: fill the
    outputs in descending ``conv_k`` order, each up to
    ``min(capacity_max_coeff_k, remaining)``.

    Outputs with ``conversion_flow_coeff ≤ 0`` draw no fuel (the
    ``= 0`` ones are the uncapped pass-through edges, outside the
    conversion equation and the sum cap) and are left out.  Only
    unit output edges (``unit__outputNode``) contribute; processes
    without any are absent (callers treat them as ``W = 0``).
    """
    empty = pl.LazyFrame(schema={"p": schema_dtype(_enums, "p"),
                                 "fuel_width": pl.Float64})
    outs = _try_entities(source, "unit__outputNode")
    if outs is None or outs.height == 0:
        return empty
    arcs = outs.lazy().select(alias_to_axis("unit", "p"),
                              alias_to_axis("node", "sink"))
    return (arcs
        .join(_arc_max_capacity_coef_lf(source, "sink"),
              on=["p", "sink"], how="left")
        .join(_arc_conversion_coef_lf(source, "sink")
                .rename({"coef": "conv"}),
              on=["p", "sink"], how="left")
        .with_columns(cap=pl.col("coef").fill_null(1.0).clip(lower_bound=0.0),
                      conv=pl.col("conv").fill_null(1.0))
        .filter(pl.col("conv") > 0.0)
        .sort(["p", "conv", "sink"], descending=[False, True, False])
        .with_columns(
            _prev=(pl.col("cap").cum_sum().over("p") - pl.col("cap")))
        .with_columns(
            _fill=pl.min_horizontal(
                pl.col("cap"),
                (1.0 - pl.col("_prev")).clip(lower_bound=0.0)))
        .group_by("p")
        .agg(fuel_width=(pl.col("conv") * pl.col("_fill")).sum()))


def _indirect_input_cap_lf(source: "InputSource",
                           pss: pl.DataFrame,
                           dt: pl.DataFrame,
                           p_slope: "Param | None",
                           p_section: "Param | None",
                           classified: pl.DataFrame,
                           piecewise_lines: pl.DataFrame | None = None,
                           ) -> pl.LazyFrame:
    """Input-arc limit per unit of capacity for indirect (multi-flow) units.

    One row per ``(p, source, sink, d, t)`` input arc of an indirect unit
    (``p ∈ process__method_indirect``, ``(p, source) ∈ process_source``,
    ``conversion_flow_coeff ≠ 0`` on the input — the zero-coefficient
    pass-through inputs are uncapped and absent)::

        value = input_share_max_s · L_s
        L_s   = (slope · W + section) · delay_factor / src_div_s

    ``L_s`` is the input flow (native units, per unit of unit capacity)
    that runs the unit at full output from input ``s`` alone: ``W`` is
    :func:`_indirect_fuel_width_lf`; ``section`` (the min-load-efficiency
    term, per unit online ≤ capacity) only for ``min_load_efficiency``
    processes and the single-line ``piecewise_linear`` units with a
    section (``single_line_rising`` of :func:`_piecewise_flags`: their
    ``p_section`` is the curve's intercept, set at the load boundary,
    ``params/_piecewise.py``); a multi-line ``piecewise_linear`` unit
    (``piecewise_lines``: its lines ``(p, d, seg, slope, intercept)``)
    needs at full output the most fuel of its lines,
    ``max_g(slope_g · W + intercept_g)``, in place of ``slope · W +
    section``; ``delay_factor = 1/max(delay weight)`` for delayed
    processes (one input step can feed a heavier-weighted output step);
    ``src_div_s`` is the input's ``conversion_flow_coeff`` (``1`` when
    ``≤ 0``), for delayed processes too: their delayed conversion term
    multiplies the input by ``conversion_flow_coeff`` exactly like the
    undelayed term (``_delay.delayed_input_expr``).  The full-load fuel is
    measured in fuel energy (flow × conversion_flow_coeff), so a poorer
    fuel automatically gets a larger flow limit.

    ``input_share_max`` (schema default 1: this input alone can run the
    unit at full output) is the largest share of the full-load fuel the
    input may supply.  Multiplied by availability × capacity (existing +
    invested − retired) in ``model/core/``'s ``maxFlow`` handling.
    """
    keys = ["p", "source", "sink", "d", "t"]
    indirect_p = (classified.lazy()
                    .filter((pl.col("klass") == "unit")
                            & pl.col("method").is_in(list(_METHOD_INDIRECT)))
                    .select("p")
                    .unique())
    # The section term: min_load_efficiency units, and the single-line piecewise_linear units whose curve has an
    # intercept (handled as min_load_efficiency units; their p_section is the curve's intercept).
    section_gate = pl.col("ct") == "min_load_efficiency"
    if (classified.get_column("ct").cast(pl.Utf8) == "piecewise_linear").any():
        rising = (_piecewise_flags(source).filter(pl.col("single_line_rising"))
                  .get_column("p").to_list())
        section_gate = section_gate | ((pl.col("ct") == "piecewise_linear")
                                       & pl.col("p").cast(pl.Utf8).is_in(rising))
    min_load_p = (classified.lazy()
                    .filter(section_gate)
                    .select("p")
                    .unique()
                    .with_columns(_has_min_load=pl.lit(True)))
    arcs = (pss.lazy().select("p", "source", "sink")
            .join(indirect_p, on="p", how="semi")
            .join(_process_source_pairs_lf(source), on=["p", "source"],
                  how="semi")
            .join(_process_source_sink_coeff_zero_lf(source, pss),
                  on=["p", "source", "sink"], how="anti"))
    base = (arcs
        .join(dt.lazy()
                 .select(alias_to_axis(pl.col("d"), "d"),
                         alias_to_axis(pl.col("t"), "t")),
              how="cross")
        .join(_arc_max_capacity_coef_lf(source, "source")
                .rename({"coef": "share"}),
              on=["p", "source"], how="left")
        .with_columns(share=pl.col("share").fill_null(1.0))
        .join(_arc_conversion_coef_lf(source, "source")
                .rename({"coef": "src_conv"}),
              on=["p", "source"], how="left")
        .with_columns(src_conv=pl.col("src_conv").fill_null(1.0))
        .join(_indirect_fuel_width_lf(source), on="p", how="left")
        .with_columns(fuel_width=pl.col("fuel_width").fill_null(0.0))
        .join(_delay_fuel_factor_lf(source), on="p", how="left")
        .with_columns(delay_factor=pl.col("delay_factor").fill_null(1.0))
        .join(min_load_p, on="p", how="left")
        .with_columns(
            _has_min_load=pl.col("_has_min_load").fill_null(False)))
    if p_slope is not None and p_slope.frame.height > 0:
        base = base.join(
            p_slope.frame.lazy().select(
                "p", "d", "t",
                pl.col("value").cast(pl.Float64).alias("slope")),
            on=["p", "d", "t"], how="left")
    else:
        base = base.with_columns(slope=pl.lit(None, dtype=pl.Float64))
    if p_section is not None and p_section.frame.height > 0:
        base = base.join(
            p_section.frame.lazy().select(
                "p", "d", "t",
                pl.col("value").cast(pl.Float64).alias("section")),
            on=["p", "d", "t"], how="left")
    else:
        base = base.with_columns(section=pl.lit(None, dtype=pl.Float64))
    #   fuel[p,d,t] = slope·W + section                       (section only behind the gate above)
    #   fuel[p,d,t] = max_g(pw_slope[p,g,d]·W + pw_intercept[p,g,d])   multi-line piecewise_linear units
    if piecewise_lines is not None and piecewise_lines.height > 0:
        pw_fuel = (piecewise_lines.lazy()
                   .select(pl.col("p").cast(pl.Utf8).alias("_pu"), pl.col("d").cast(pl.Utf8).alias("_du"),
                           pl.col("slope").cast(pl.Float64), pl.col("intercept").cast(pl.Float64))
                   .join(_indirect_fuel_width_lf(source)
                         .select(pl.col("p").cast(pl.Utf8).alias("_pu"), "fuel_width"),
                         on="_pu", how="left")
                   .with_columns(fuel_width=pl.col("fuel_width").fill_null(0.0))
                   .group_by("_pu", "_du")
                   .agg(_pw_fuel=(pl.col("slope") * pl.col("fuel_width") + pl.col("intercept")).max()))
        base = (base.with_columns(_pu=pl.col("p").cast(pl.Utf8), _du=pl.col("d").cast(pl.Utf8))
                .join(pw_fuel, on=["_pu", "_du"], how="left", maintain_order="left")
                .drop("_pu", "_du"))
    else:
        base = base.with_columns(_pw_fuel=pl.lit(None, dtype=pl.Float64))
    return (base
        .with_columns(
            _fuel=pl.when(pl.col("_pw_fuel").is_not_null())
                    .then(pl.col("_pw_fuel"))
                    .otherwise(pl.col("slope").fill_null(0.0) * pl.col("fuel_width")
                               + pl.when(pl.col("_has_min_load"))
                                   .then(pl.col("section").fill_null(0.0))
                                   .otherwise(0.0)),
            # src_conv ≤ 0 inputs would feed the conversion negatively
            # (not tied to the fuel need): keep a finite limit with a
            # unit divisor.  Delayed and undelayed inputs alike enter the
            # conversion as flow × src_conv.
            _src_div=pl.when(pl.col("src_conv") <= 0.0)
                        .then(1.0)
                        .otherwise(pl.col("src_conv")))
        .select(*keys,
                value=(pl.col("share") * pl.col("_fuel")
                       * pl.col("delay_factor") / pl.col("_src_div"))))


def p_indirect_input_cap_from_source(source: "InputSource",
                                     pss: "pl.DataFrame | None",
                                     dt: "pl.DataFrame | None",
                                     p_slope: "Param | None",
                                     p_section: "Param | None",
                                     classified: pl.DataFrame | None = None,
                                     piecewise_lines: pl.DataFrame | None = None,
                                     ) -> "Param | None":
    """``Param((p, source, sink, d, t))`` — the input-arc limit of indirect
    units per unit of capacity (:func:`_indirect_input_cap_lf`):
    ``input_share_max × full-load input flow per unit of capacity``.

    ``model/core/`` multiplies it by availability and the unit's capacity:
    a per-element ``v_flow`` upper bound when the unit has no invest /
    divest variables in the solve, else a ``maxFlow`` row whose invest /
    divest terms carry the same multiplier.  ``None`` when there are no
    such arcs.
    """
    if pss is None or pss.height == 0 or dt is None or dt.height == 0:
        return None
    _live = get_global_axis_enums()
    if _live is not None:
        pss = cast_frame_axes(pss, _live)
        dt = cast_frame_axes(dt, _live)
        if classified is not None:
            classified = cast_frame_axes(classified, _live)
    if classified is None:
        classified = _classify_process_method(source)
    out = (_indirect_input_cap_lf(source, pss, dt, p_slope, p_section,
                                  classified, piecewise_lines)
           .sort("p", "source", "sink", "d", "t")
           .collect())
    if out.height == 0:
        return None
    return Param(("p", "source", "sink", "d", "t"), out)
