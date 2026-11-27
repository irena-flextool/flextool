"""Γ.3.B — process flow parameters: ``flow_to_n`` / ``flow_from_n``, ``p_flow_constraint_coef``, the
conversion slope, section and minimum load, and :func:`apply_derived_b`."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import (
    alias_to_axis,
    cast_dim,
    cast_frame_axes,
    get_global_axis_enums,
    schema_dtype,
)
from flextool.engine.input._param_shapes import (
    broadcast_to_period_time,
    parameter_has_values,
    promote_param_to_dt,
    resolve_param_shape,
)

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import _enums, _read_active_solve, _try_param
from flextool.engine.preprocess.params._derived._unitsize import _entity_unitsize_lf
from flextool.engine.preprocess.params._derived._process_method import (
    _classify_process_method,
    process_indirect_set,
    process_input_flows,
    process_output_flows,
)
from flextool.engine.preprocess.params._derived._arc_capacity import (
    p_arc_max_cap_coef_from_source,
    process_source_sink_uncapped_from_source,
)
from flextool.engine.preprocess.params._derived._existing_capacity import p_flow_upper_existing_from_source


# ---------------------------------------------------------------------------
# §3.3.1 — flow_to_n / flow_from_n (block-aware filter on top of pss)
# ---------------------------------------------------------------------------


def flow_to_n(source: "InputSource",
              pss: pl.DataFrame) -> pl.DataFrame:
    """Augment pss with ``n = sink``.  Schema ``[p, source, sink, n]``.

    Every arc enters the balance of its sink node (a node of a coarse
    new_stepduration group sums its fine flows, nodeBalanceCoarse_eq).
    """
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink"),
                                      "n": schema_dtype(_enums, "n")})
    return (pss.lazy()
              .with_columns(n=pl.col("sink"))
              .select("p", "source", "sink", "n")
              .sort("p", "source", "sink", "n")
              .collect())


def flow_from_n(source: "InputSource",
                pss: pl.DataFrame) -> pl.DataFrame:
    """Augment pss with ``n = source``.  Schema ``[p, source, sink, n]``."""
    if pss is None or pss.height == 0:
        return pl.DataFrame(schema={"p": schema_dtype(_enums, "p"),
                                      "source": schema_dtype(_enums, "source"),
                                      "sink": schema_dtype(_enums, "sink"),
                                      "n": schema_dtype(_enums, "n")})
    return (pss.lazy()
              .with_columns(n=pl.col("source"))
              .select("p", "source", "sink", "n")
              .sort("p", "source", "sink", "n")
              .collect())


# ---------------------------------------------------------------------------
# §3.5.1 — p_flow_constraint_coef
# ---------------------------------------------------------------------------


def p_flow_constraint_coef_from_source(source: "InputSource",
                                          pss: pl.DataFrame,
                                          ) -> "Param | None":
    """User-defined per-arc (p, source, sink, cn) ``constraint_flow_coeff``.

    The constraint axis column is ``cn`` (not ``c``) to avoid collision
    with the commodity axis — see the c_collision review note in
    ``schemas/flextool_axis_contract.json``.

    Algorithm (audit §3.5.1, ``flex_data/_load_features.py::_load_user_constraints``):
      1. Pull unit__inputNode / unit__outputNode /
         connection__node.constraint_flow_coeff.  Each row carries
         ``(process, node, constraint, coef)``.
      2. Inner-join with pss on (p, n=source) → source-leg rows.
      3. Inner-join with pss on (p, n=sink) → sink-leg rows.
      4. Concat + group_by(p, source, sink, cn).sum.

    Returns ``None`` when no ``constraint_flow_coeff`` rows exist.
    """
    if pss is None or pss.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame param.
    _enums = get_global_axis_enums()
    if _enums is not None:
        pss = cast_frame_axes(pss, _enums)
    parts: list[pl.LazyFrame] = []
    for entity_class in ("unit__inputNode", "unit__outputNode",
                          "connection__node"):
        df = _try_param(source, entity_class, "constraint_flow_coeff")
        if df is None:
            continue
        # Schema: <process_col>, <node_col>, <constraint_index_col>, value.
        # constraint_flow_coeff is a 1d_map[constraint] → coef.
        cols = df.columns
        # Disambiguate process / node columns by entity_class layout.
        p_col = ("unit" if entity_class.startswith("unit")
                  else "connection")
        n_col = "node"
        c_col = next((c for c in cols
                       if c not in (p_col, n_col, "value")), None)
        if c_col is None:
            continue
        parts.append(df.lazy().select(
            alias_to_axis(p_col, "p"),
            alias_to_axis(n_col, "n"),
            alias_to_axis(c_col, "cn"),
            pl.col("value").cast(pl.Float64).alias("coef"),
        ))
    if not parts:
        return None
    coef = pl.concat(parts)

    # Cross-axis joins: pss carries ``source``/``sink`` (e-axis) joined
    # against ``coef.n`` (n-axis).  Per the contract n ⊂ e; up-cast
    # coef.n to e-Enum once and the join composes natively in Enum
    # without Utf8 materialisation at the join boundary.
    coef_e = coef.with_columns(cast_dim(pl.col("n"), None, "e"))
    src_match = (pss.lazy()
        .join(coef_e, left_on=["p", "source"], right_on=["p", "n"],
                how="inner")
        .select("p", "source", "sink", "cn", "coef"))
    sink_match = (pss.lazy()
        .join(coef_e, left_on=["p", "sink"], right_on=["p", "n"],
                how="inner")
        .select("p", "source", "sink", "cn", "coef"))
    joined = (pl.concat([src_match, sink_match], how="vertical")
                .group_by(["p", "source", "sink", "cn"])
                .agg(pl.col("coef").sum().alias("value"))
                .sort("p", "source", "sink", "cn")
                .collect())
    if joined.height == 0:
        return None
    return Param(("p", "source", "sink", "cn"), joined)


# ---------------------------------------------------------------------------
# §3.3.4 — p_slope (efficiency-curve slope)
# ---------------------------------------------------------------------------


def _eff_param_to_pdt(source: "InputSource",
                       entity_class: str,
                       parameter_name: str,
                       dt: pl.DataFrame,
                       value_col_alias: str,
                       ) -> "pl.LazyFrame | None":
    """Resolve an efficiency-family time parameter to a
    ``[p, d, t, <value_col_alias>]`` LazyFrame via the authoritative
    shape resolver (:mod:`._param_shapes`).

    Replaces the column-name-based ``_broadcast_param_to_dt``: the Spine
    Map index label is user-set (``"x"`` is the spinedb_api default when
    unset), so name matching mis-classified time-series parameters as
    scalars and broadcast them with a ``how="cross"`` cartesian product
    (process × full (d, t) grid) — which both overflowed polars' 2^32
    row-index on long timelines and, on short ones, silently flattened
    the time variation via the downstream ``unique(["p","d","t"])``.

    The resolver reads the parameter's nesting depth + per-level
    ``index_name`` from the DB and (for silent-default labels) probes the
    index values against ``dt``'s period / timestep domains, so a
    ``Map(time → value)`` resolves to ``Shape.MAP_TIME`` regardless of
    label.  :func:`broadcast_to_period_time` then keeps the Param at its
    authored shape (no cross-product) and :func:`promote_param_to_dt`
    widens it to ``(p, d, t)`` with a *narrowing* inner-join on the axis
    the source carries.

    A unit authoring a stochastic-time value takes the solve's emit-folded values
    (:func:`._param_shapes.broadcast_to_period_time`); the other units keep their own shapes.

    Returns ``None`` when the parameter has no usable rows (mirrors the
    old helper's ``None`` contract so callers' gating is unchanged).
    """
    resolved = resolve_param_shape(
        source, entity_class, parameter_name, period_filter=dt)
    param = broadcast_to_period_time(resolved, "p", dt)
    if param is None:
        return None
    return (promote_param_to_dt(param, dt)
                .select("p", "d", "t",
                        pl.col("value").cast(pl.Float64).alias(value_col_alias)))


def _mle_eta_min(eta_min: pl.Expr, eta: pl.Expr) -> pl.Expr:
    """Efficiency at minimum load of a min_load_efficiency cell: ``efficiency_at_min_load`` where given, else
    ``efficiency`` (a flat curve), cell by cell.  Shared by ``p_slope_from_source`` and
    ``p_section_from_source`` (the rolling emit seed ``_period_params._derive_conversion_trio`` applies the same
    fallback)."""
    return pl.coalesce(eta_min, eta)


def _mle_section(eta: pl.Expr, eta_min: pl.Expr, min_load: pl.Expr) -> pl.Expr:
    """Section (input per online unit) of a min_load_efficiency cell, the curve through (min_load, eta_min) and
    (1, eta):

        section = min_load · (1/eta_min − 1/eta) · (1 / (1 − min_load))    (min_load 0 → 0 whatever eta_min is)

    Unrounded.  Algebraically 1/eta − (1/eta − min_load/eta_min)/(1 − min_load), but written as a product of
    the efficiency difference so that a flat curve (eta_min = eta) gives exactly 0: the difference form left a
    float residue (−2.2e-16 for eta 0.9, min_load 0.3) on v_online that reached the objective as a ~1e-17 cost,
    a 16-decade cost range that autoscale centred with a 2^25 objective scale and HiGHS' dual simplex then failed
    on (excessive dual values, no model status).  The rolling emit seed
    ``_period_params._derive_conversion_trio`` evaluates the same float operations in the same order.

    The division by ``1 − min_load`` is a multiplication by its reciprocal.  Polars divides a column by a
    scalar-valued (broadcast) column as a multiplication by the reciprocal, but a column by a column as a true
    division, and whether ``min_load`` stays scalar-valued depends on how it was authored (a constant or a
    period map) and on the polars version; the two differ by an ulp.  With only multiplications and
    ``1.0 / x`` divisions the bits do not depend on the representation, so a constant and its equivalent map
    give the same section on every path."""
    return (pl.when(min_load == 0.0).then(0.0)
            .otherwise(min_load * (1.0 / eta_min - 1.0 / eta) * (1.0 / (1.0 - min_load))))


def _mle_slope(eta: pl.Expr, eta_min: pl.Expr, min_load: pl.Expr) -> pl.Expr:
    """Slope of input over output of a min_load_efficiency cell:

        slope = 1/eta − section      (= (1/eta − min_load/eta_min) / (1 − min_load);  section 0 → 1/eta)

    Shared by ``p_slope_from_source`` and ``p_section_from_source`` (via ``_mle_section``) so that
    slope + section = 1/eta on every cell and every path (single, rolling, nested, stochastic) reads identical
    values."""
    return 1.0 / eta - _mle_section(eta, eta_min, min_load)


def _mle_curve_inputs(source: "InputSource", dt: pl.DataFrame, base: pl.LazyFrame) -> pl.LazyFrame:
    """``base`` (p, d, t, …, eta) with ``eta_min`` and ``min_load`` added for the min_load_efficiency curve.

        min_load[p,d,t] = authored value, 0 where not authored (no minimum load)
        eta_min[p,d,t]  = efficiency_at_min_load where authored, else efficiency  (_mle_eta_min)
    """
    if parameter_has_values(source, "unit", "efficiency_at_min_load"):
        eta_min_lf = _eff_param_to_pdt(source, "unit", "efficiency_at_min_load", dt, "eta_min")
    else:
        eta_min_lf = None
    ml_lf = (_eff_param_to_pdt(source, "unit", "min_load", dt, "min_load")
             if parameter_has_values(source, "unit", "min_load") else None)
    base = (base.join(eta_min_lf, on=["p", "d", "t"], how="left") if eta_min_lf is not None
            else base.with_columns(eta_min=pl.lit(None).cast(pl.Float64)))
    base = (base.join(ml_lf, on=["p", "d", "t"], how="left") if ml_lf is not None
            else base.with_columns(min_load=pl.lit(None).cast(pl.Float64)))
    return base.with_columns(min_load=pl.col("min_load").fill_null(0.0),
                             eta_min=_mle_eta_min(pl.col("eta_min"), pl.col("eta")))


def _raise_on_null_mle_cells(df: pl.DataFrame, what: str) -> None:
    """A min_load_efficiency cell without a value would be a unit without fuel use there: refuse it."""
    if df.height:
        cells = ", ".join(f"({p}, {d}, {t})" for p, d, t in df.select(
            pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).head(5).iter_rows())
        raise ValueError(
            f"{what}: min_load_efficiency cells without a value (the unit would use no fuel there): {cells}; "
            f"this is a loader bug (min_load, efficiency_at_min_load and efficiency all have defaults).")


def p_slope_from_source(source: "InputSource",
                          dt: pl.DataFrame,
                          classified: pl.DataFrame | None = None,
                          ) -> "Param | None":
    """Conversion-curve slope per (p, d, t).

    For ``constant_efficiency`` ct_method: slope = 1 / efficiency.
    For ``min_load_efficiency`` ct_method: slope is linearised between
    (min_load, efficiency_at_min_load) and (1.0, efficiency); per
    audit §3.3.4 / entity_period_calc_params.py:1209-1306.

    On a direct (one-input, one-output) unit, the slope above is further scaled by
    ``k_sink/k_src``, the unit's output/input ``conversion_flow_coeff`` (default 1 each; #10 / Q12),
    folded in by :func:`_arithmetic._direct_flow_coef_ratio` — the same ratio the rolling emit seed
    (``emit._period_params._derive_conversion_trio``) applies.  A zero coefficient keeps its
    uncapped-edge meaning instead (ratio left at 1, with a warning naming the unit).

    Time-varying efficiency Maps are broadcast to the (d, t) grid;
    scalar efficiency is broadcast to the full grid.

    Returns ``None`` when no usable efficiency data is available.
    """
    if dt is None or dt.height == 0:
        return None
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _enums = get_global_axis_enums()
    if _enums is not None:
        dt = cast_frame_axes(dt, _enums)
        classified = cast_frame_axes(classified, _enums)
    # Pull efficiency / efficiency_at_min_load / min_load from units +
    # transfer connections (efficiency lives on both unit and connection
    # though connection.efficiency may be implicit via transfer_method).
    # Presence checks only (shape-agnostic: a stochastic-time unit value is not classified here).
    eff_unit = parameter_has_values(source, "unit", "efficiency")
    eff_conn = parameter_has_values(source, "connection", "efficiency")

    # Build a per-(p, d, t) efficiency lazyframe — union unit + connection.
    # Routed through the authoritative shape resolver so time-series
    # efficiency (e.g. connection.efficiency authored as Map(time→value)
    # with the spinedb_api silent-default ``"x"`` index label) is narrowed
    # to the active timeline instead of cross-joined; see
    # :func:`_eff_param_to_pdt`.
    eff_lfs: list[pl.LazyFrame] = []
    if eff_unit:
        e = _eff_param_to_pdt(source, "unit", "efficiency", dt, "eta")
        if e is not None:
            eff_lfs.append(e)
    if eff_conn:
        e = _eff_param_to_pdt(source, "connection", "efficiency", dt, "eta")
        if e is not None:
            eff_lfs.append(e)
    # Default-fill for processes in the classifier but missing from the
    # explicit efficiency rows.  flextool's default is 1.0 (constant_eff
    # ct_method baseline) — broadcast to the full classified × dt grid
    # then left-join with explicit rows so explicit overrides win.
    # Defensive re-cast: ensure d/t are canonical Enum on the cross-join
    # output (classified carries Enum p / ct / klass).
    classified_dt = (classified.lazy()
        .select("p", "ct", "klass")
        .join(dt.lazy()
                 .with_columns(alias_to_axis(pl.col("d"), "d"),
                               alias_to_axis(pl.col("t"), "t")),
              how="cross")
        .select("p", "d", "t", "ct", "klass"))
    if eff_lfs:
        explicit = pl.concat(eff_lfs).unique(subset=["p", "d", "t"])
        eff_lf = (classified_dt
                    .join(explicit, on=["p", "d", "t"], how="left")
                    .with_columns(eta=pl.col("eta").fill_null(1.0)))
    else:
        eff_lf = classified_dt.with_columns(eta=pl.lit(1.0))

    # constant_efficiency branch: slope = 1 / eta.
    base = eff_lf  # already has p, d, t, ct, klass, eta

    # min_load_efficiency rows: the slope of input over output between (min_load, eta_min) and (1, eta):
    #   slope[p,d,t] = 1/eta − section,  section = min_load·(1/eta_min − 1/eta) / (1 − min_load)  (_mle_slope)
    #   eta = efficiency (default 1.0);  min_load = 0 where not authored;
    #   eta_min = efficiency_at_min_load, else efficiency (_mle_eta_min: a flat curve, slope = 1/eta)
    # A min_load of 0 gives the constant-efficiency slope 1/eta whatever eta_min is.
    if classified.filter(pl.col("ct") == "min_load_efficiency").height > 0:
        base = _mle_curve_inputs(source, dt, base)
    else:
        base = base.with_columns(eta_min=pl.lit(None).cast(pl.Float64),
                                 min_load=pl.lit(0.0))

    # slope expression: branch on ct.
    base = base.with_columns(
        slope=pl.when(pl.col("ct") == "min_load_efficiency")
                .then(_mle_slope(pl.col("eta"), pl.col("eta_min"), pl.col("min_load")))
              .otherwise(1.0 / pl.col("eta"))
    )
    # #10 / Q12: fold a direct unit's output/input conversion_flow_coeff ratio into its slope —
    # slope = slope⁰·k_sink/k_src (Q-A; _arithmetic._direct_flow_coef_ratio is the shared direct-set
    # definition with the rolling emit seed, _period_params._derive_conversion_trio).
    from flextool.engine.preprocess.params._arithmetic import _direct_flow_coef_ratio
    ratio = _direct_flow_coef_ratio(source)
    if ratio.height > 0:
        base = (base.join(ratio.lazy(), on="p", how="left")
                     .with_columns(
                         slope=pl.when(pl.col("k_sink").is_not_null())
                                 .then(pl.col("slope") * pl.col("k_sink") / pl.col("k_src"))
                                 .otherwise(pl.col("slope"))))
    out = (base.select("p", "d", "t", pl.col("slope").alias("value"))
                .sort("p", "d", "t")
                .collect())
    _raise_on_null_mle_cells(out.filter(pl.col("value").is_null()), "p_slope")
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


# ---------------------------------------------------------------------------
# §3.3.4b — p_min_load (minFlow_minload floor), shape-correct (p, d, t)
# ---------------------------------------------------------------------------


def p_min_load_pdt_from_source(source: "InputSource",
                                 dt: pl.DataFrame,
                                 ) -> "Param | None":
    """``unit.min_load`` → ``Param(("p", "d", "t"), [p, d, t, value])``.

    Routed through the authoritative shape resolver (:func:`_eff_param_to_pdt`,
    the same path the efficiency-curve ``p_slope`` / ``p_section`` already use)
    so a period / time / period-time ``Map`` keeps its per-(d, t) semantics.

    This replaces the old ``(p,)`` direct param (``_direct``
    ``p_min_load_from_source`` → ``_entity_scalar_explicit``) whose
    ``.select(dim, "value")`` dropped the Map index column: a Map min_load
    then yielded one ``(unit, value)`` row per key, and polar_high **summed**
    the duplicate ``(p)`` keys in the ``minFlow_minload`` constraint
    (``model/core/``) — a period Map ``{p1: 0.4, p2: 0.3}`` produced a floor of
    0.7 in *every* period.  See ``specs/fix_map_reading.md`` §5.

    A **scalar** min_load is broadcast to the full ``(d, t)`` grid, so the
    constraint RHS (``v_online · min_load``) is byte-identical to the old
    ``(p,)`` broadcast — only Map authoring changes behaviour.

    Returns ``None`` when no unit authors ``min_load`` (no value = 0, no
    floor).  The online feature does not need this field; ``model/core/``
    checks it (``MIN_LOAD``) only when some unit has ``min_load > 0``
    (``process_minload``).
    """
    if dt is None or dt.height == 0:
        return None
    # Defend axis-aware join keys on the incoming frame param, mirroring
    # ``p_slope_from_source``'s treatment of ``dt`` before ``_eff_param_to_pdt``.
    _enums = get_global_axis_enums()
    if _enums is not None:
        dt = cast_frame_axes(dt, _enums)
    lf = _eff_param_to_pdt(source, "unit", "min_load", dt, "value")
    if lf is None:
        return None
    out = lf.sort("p", "d", "t").collect()
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)


def process_minload_from_p_min_load(p_min_load: "Param | None") -> pl.DataFrame | None:
    """``process_minload`` ``(p,)``: the units whose ``min_load`` is above 0 on some (d, t) of this solve.

    Read from the solve-resolved ``p_min_load`` (not the raw parameter), so a unit whose map has values only in
    other periods of the model gets no ``minFlow_minload`` floor, and the floor's units always have a
    ``p_min_load`` row.  None when no unit qualifies.
    """
    if p_min_load is None:
        return None
    out = (p_min_load.frame.lazy()
           .filter(pl.col("value") > 0)
           .select(pl.col("p"))
           .unique()
           .sort("p")
           .collect())
    return out if out.height > 0 else None


# ===========================================================================
# Γ.3.B integration entrypoint
# ===========================================================================


DERIVED_B_FIELDS = (
    "p_flow_upper_existing",
    "p_flow_constraint_coef",
    "p_slope",
    "p_min_load",
)


def apply_derived_b(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.B Derived Params, mutating ``flex_data`` in place.

    The classifier (``_classify_process_method``) is the single
    dependency for the reclassified ``process_indirect`` / ``flow_to_n``
    family; it's computed once per call and threaded through the
    per-Param helpers as a kwarg.

    Δ.3 replaced the previous ``derived_overrides_b`` dict-return;
    Δ.4 deleted the deprecated wrapper alias.
    """
    # Active solve (used for existing-period broadcast in some helpers).
    active_solve = ctx.solve_name if ctx is not None else _read_active_solve(workdir, provider=provider)

    # Build the classifier once.  Δ.12b: hard errors here would
    # indicate a malformed source — let them propagate instead of
    # silently producing an empty schema.
    classified = _classify_process_method(source)

    # Topology — already-computed CSV-side process_source_sink (a plain
    # DataFrame on FlexData; see ``flex_data/_load_topology.py::_load_process_topology``).
    pss_frame = getattr(flex_data, "process_source_sink", None)
    dt_csv = getattr(flex_data, "dt", None)

    # process_indirect / process_input_flows / process_output_flows /
    # process_indirect_dt.  Δ.12b: helpers produce or raise.  ``None``
    # / empty frame == "no indirect units" — same semantic the seed
    # produces.  We keep the ``height > 0`` filter on the assignment
    # so the SET-frame contract (None or non-empty) is preserved.
    if classified.height > 0:
        pi_db = process_indirect_set(source, classified)
        if pi_db is not None and pi_db.height > 0:
            flex_data.process_indirect = pi_db
        else:
            flex_data.process_indirect = None

        if pss_frame is not None and pss_frame.height > 0:
            pif_db = process_input_flows(source, pss_frame, classified)
            flex_data.process_input_flows = (
                pif_db if pif_db is not None and pif_db.height > 0
                else None)

            pof_db = process_output_flows(source, pss_frame, classified)
            flex_data.process_output_flows = (
                pof_db if pof_db is not None and pof_db.height > 0
                else None)

        # Phase E.3: ``process_indirect_dt`` is no longer materialised
        # eagerly; consumers call
        # ``_pdt_join.compute_process_indirect_dt`` on demand.  The
        # field stays declared on ``FlexData`` for ``getattr`` defenders
        # but defaults to ``None``.

    # flow_to_n / flow_from_n: every arc enters the balance of both its nodes (a coarse node sums its fine
    # flows over each coarse step, nodeBalanceCoarse_eq).
    if pss_frame is not None and pss_frame.height > 0:
        from flextool.engine.preprocess.params._block import (
            flow_from_n_frame,
            flow_from_nodeBalance_seed,
            flow_to_n_frame,
        )
        ftn_db = flow_to_n_frame(pss_frame)
        ffn_db = flow_from_n_frame(pss_frame)
        if ftn_db is not None and ftn_db.height > 0:
            flex_data.flow_to_n = ftn_db
        if ffn_db is not None and ffn_db.height > 0:
            flex_data.flow_from_n = ffn_db

        # ─── Δ.27 flow_from_nodeBalance_{eff,noEff} ─────────────────
        # Source-side nodeBalance topology — native port of the
        # inline derivation in ``flex_data/_load_storage.py::_load_storage`` lines
        # 1658-1705.  Without these, source-side flows from
        # nodeBalance nodes (e.g. transmission discharge into a
        # balance node) don't appear in nodeBalance_eq and the LP
        # decouples on multi-arc fixtures
        # (``work_lh2_three_region``).
        # Source-side flows must be gathered for BOTH plain ``nodeBalance``
        # nodes and ``balance_within_period`` (``nodeBalancePeriod``) nodes —
        # a period node that is a *source* on an arc (e.g. GAS → gas plant)
        # must contribute its source-side flow term to ``nodeBalancePeriod_eq``,
        # else the period balance is vacuous and the source draws for free.
        # Pass the dtype-safe union to the seed (which filters ``pss.source ∈
        # n``).  Both frames come from the same ``rename_to_axis({"node":"n"})``
        # in ``flex_data/_load_topology.py:_load_node`` so the ``n`` Enum aligns.  When the period
        # set is empty/None the union collapses to exactly ``nb_for_ffnb``
        # (no concat/unique/reorder) → byte-identical legacy behavior.
        nb_for_ffnb = getattr(flex_data, "nodeBalance", None)
        _nbp_for_ffnb = getattr(flex_data, "nodeBalancePeriod", None)
        if (nb_for_ffnb is not None and _nbp_for_ffnb is not None
                and _nbp_for_ffnb.height > 0):
            nb_for_ffnb = (pl.concat([nb_for_ffnb.select("n"),
                                      _nbp_for_ffnb.select("n")])
                             .unique()
                             .select("n"))
        pss_eff_frame = getattr(flex_data, "process_source_sink_eff", None)
        pss_noEff_frame = getattr(flex_data, "process_source_sink_noEff",
                                    None)
        if (nb_for_ffnb is not None and nb_for_ffnb.height > 0
                and pss_eff_frame is not None
                and pss_eff_frame.height > 0):
            ffnb_eff = flow_from_nodeBalance_seed(pss_eff_frame, nb_for_ffnb)
            if ffnb_eff is not None and ffnb_eff.height > 0:
                flex_data.flow_from_nodeBalance_eff = ffnb_eff
        if (nb_for_ffnb is not None and nb_for_ffnb.height > 0
                and pss_noEff_frame is not None
                and pss_noEff_frame.height > 0):
            ffnb_noEff = flow_from_nodeBalance_seed(pss_noEff_frame, nb_for_ffnb)
            if ffnb_noEff is not None and ffnb_noEff.height > 0:
                flex_data.flow_from_nodeBalance_noEff = ffnb_noEff

    # ─── §3.3.3 p_flow_upper_existing ──────────────────────────────────
    # Δ.12b: unconditional when pss is non-empty.  Helper returns None
    # for fixtures without entity-existing-capacity.
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_flow_upper_existing = p_flow_upper_existing_from_source(
            source, pss_frame, active_solve, workdir,
            ctx=ctx, provider=provider)

    # ─── §3.3.3b p_arc_max_cap_coef (per-arc capacity_max_coeff) ───────
    # ``p_flow_upper_existing`` is existing_count only and drops the
    # per-arc ``capacity_max_coeff`` that the .mod's maxToSink /
    # maxFromSource apply.  Carry the factor here so the model/core/ maxFlow
    # consumer can fold it onto every arc bound by the existing-only RHS
    # (direct arcs + indirect output arcs; indirect input arcs carry it
    # via ``p_flow_upper``).
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_arc_max_cap_coef = p_arc_max_cap_coef_from_source(
            source, pss_frame, classified)
        flex_data.process_source_sink_uncapped = (
            process_source_sink_uncapped_from_source(source, pss_frame))

    # ─── §3.5.1 p_flow_constraint_coef ─────────────────────────────────
    # Δ.12b: unconditional when pss is non-empty.
    if pss_frame is not None and pss_frame.height > 0:
        flex_data.p_flow_constraint_coef = p_flow_constraint_coef_from_source(
            source, pss_frame)

    # ─── §3.3.4 p_slope ────────────────────────────────────────────────
    # Δ.12b: unconditional when (dt, classified) are non-empty.
    if dt_csv is not None and dt_csv.height > 0 and classified.height > 0:
        flex_data.p_slope = p_slope_from_source(source, dt_csv, classified)

    # ─── §3.3.4b p_min_load (minFlow_minload floor, (p, d, t)) ──────────
    # Shape-correct (p, d, t) min_load routed through the efficiency
    # resolver so a Map isn't summed in the minFlow_minload constraint.
    # Gated on dt only (min_load applies to any online+minload unit,
    # independent of the ct_method classifier).  See
    # ``p_min_load_pdt_from_source`` / ``specs/fix_map_reading.md`` §5.
    if dt_csv is not None and dt_csv.height > 0:
        flex_data.p_min_load = p_min_load_pdt_from_source(source, dt_csv)

    # ─── §F.1 p_unitsize  (Δ.10 cluster F) ─────────────────────────────
    # Per-process unitsize cascade restricted to processes appearing in
    # ``pss``.  Mirrors ``flex_data/_load_topology.py::_load_process_topology``'s
    # ``unitsize_long.filter(p ∈ pss["p"].unique())``.
    # Δ.12b: unconditional when pss is non-empty.
    if pss_frame is not None and pss_frame.height > 0:
        from flextool.engine.preprocess.params._arithmetic import p_unitsize_from_source
        flex_data.p_unitsize = p_unitsize_from_source(source, pss_frame)

    # ─── §F.1b p_all_entity_unitsize  (scaling family — all entities) ────
    # Unfiltered entity unitsize covering processes + connections + nodes.
    # Used by the scaling analyzer (scaling.py:analyze_solve) to compute the
    # full entity-unitsize spread including node entries, and by the output
    # reader (``read_parameters._build_entity_unitsize_series``) as the
    # complete entity→unitsize carrier.
    # Note: _entity_unitsize_lf returns all entities (unit ∪ node ∪ connection)
    # with the cascade: virtual_unitsize OR existing OR 1000.0.
    #
    # This value is solve-invariant (computed against the whole-model,
    # unfiltered-scenario source).  In the DB cascade path it is computed
    # ONCE before the per-solve loop and seeded into the Provider under
    # ``input/p_all_entity_unitsize`` (see ``_orchestration._cascade._drive_cascade``),
    # so here we read it from the Provider rather than recompute it per roll.
    # The recompute is kept as the fallback for the CSV/fixture path and any
    # path that lacks the seeded Provider.
    from polar_high import Param as _Param
    _all_us_df = None
    if provider is not None:
        _seeded = provider.get("input/p_all_entity_unitsize")
        if _seeded is not None and _seeded.height > 0:
            _all_us_df = _seeded
    if _all_us_df is None:
        _all_us_lf = _entity_unitsize_lf(source)
        _all_us_df = _all_us_lf.rename({"us": "value"}).collect()
    if _all_us_df.height > 0:
        flex_data.p_all_entity_unitsize = _Param(("e",), _all_us_df)

    # ─── §F.4 p_process_source_conversion_flow_coeff /
    # p_process_sink_conversion_flow_coeff ─────────────────────────────
    # (Δ.10 cluster F).  Mirrors ``flex_data/_load_features.py:_load_indirect`` lines
    # 950-1002.  Anti-joins zero-coef rows out of the indirect-process
    # input/output flow sets and emits a Param keyed on (p, source) /
    # (p, sink) when any non-default, non-zero coef remains.
    # Δ.12b: helper exceptions propagate; the anti-join inner gate
    # (only when zero-rows exist) is preserved as a structural filter.
    from flextool.engine.preprocess.params._arithmetic import (
        p_process_source_conversion_flow_coeff_from_source,
        p_process_sink_conversion_flow_coeff_from_source,
    )
    pif = getattr(flex_data, "process_input_flows", None)
    pof = getattr(flex_data, "process_output_flows", None)
    z_src, p_src_coef = p_process_source_conversion_flow_coeff_from_source(
        source, pif)
    if z_src is not None and z_src.height > 0 and pif is not None \
            and pif.height > 0:
        new_pif = pif.join(z_src, on=["p", "source"], how="anti")
        if new_pif.height < pif.height:
            flex_data.process_input_flows = new_pif
    # TODO(Δ.12b helper-fix):
    # p_process_source_conversion_flow_coeff_from_source returns None when
    # no zero-coef rows exist in the source — but the seed-side
    # _load_indirect builds the coefficient Param from a different code
    # path (process__commodity__node_conversion_flow_coeff.csv).  Keep
    # the conditional assignment until the helpers converge on a single
    # producer.
    if p_src_coef is not None:
        flex_data.p_process_source_conversion_flow_coeff = p_src_coef

    z_sink, p_sink_coef = p_process_sink_conversion_flow_coeff_from_source(
        source, pof)
    if z_sink is not None and z_sink.height > 0 and pof is not None \
            and pof.height > 0:
        new_pof = pof.join(z_sink, on=["p", "sink"], how="anti")
        if new_pof.height < pof.height:
            flex_data.process_output_flows = new_pof
    if p_sink_coef is not None:
        flex_data.p_process_sink_conversion_flow_coeff = p_sink_coef


# ---------------------------------------------------------------------------
# §3.8.1 — p_section
# ---------------------------------------------------------------------------


def p_section_from_source(source: "InputSource",
                            dt: pl.DataFrame,
                            classified: pl.DataFrame | None = None,
                            ) -> "Param | None":
    """Section term (y-intercept) of the ``min_load_efficiency`` linearisation, one row per
    min_load_efficiency unit × (d, t):

        section = min_load · (1/eta_min − 1/eta) / (1 − min_load)   (_mle_section)
        eta = efficiency (default 1.0);  min_load = 0 where not authored (section 0);
        eta_min = efficiency_at_min_load, else efficiency (_mle_eta_min: a flat curve, section exactly 0)

    i.e. 1/eta − slope.  Unrounded, and the term ``p_slope_from_source`` subtracts from 1/eta, so the full-load
    input slope⁰ + section⁰ is 1/eta (the old flextool rounding of 1/eta and of the slope to 6 decimals left
    slope⁰ + section⁰ ≠ 1/eta, #140).  Other ct_methods → no section row; None
    when there is no min_load_efficiency unit.

    On a direct (one-input, one-output) unit, the section above is further divided by ``k_src``, the
    unit's input ``conversion_flow_coeff`` (default 1; #10 / Q12, Q-A — the section is per unit of
    native output capacity, not weighted by the sink coefficient).  With ``k_src`` ≠ 1 the folded
    slope + section no longer sums to 1/eta — the identity above is for the unfolded slope⁰/section⁰
    only.  Folded by :func:`_arithmetic._direct_flow_coef_ratio`, shared with
    ``p_slope_from_source`` and the rolling emit seed; a zero ``k_src``/``k_sink`` keeps its
    uncapped-edge meaning instead (ratio left at 1).
    """
    if dt is None or dt.height == 0:
        return None
    if classified is None:
        classified = _classify_process_method(source)
    if classified.height == 0:
        return None
    # Phase 4.8f: defend axis-aware join keys on incoming frame params.
    _enums = get_global_axis_enums()
    if _enums is not None:
        dt = cast_frame_axes(dt, _enums)
        classified = cast_frame_axes(classified, _enums)
    mle = classified.filter(pl.col("ct") == "min_load_efficiency").select("p").unique()
    if mle.height == 0:
        return None
    base = mle.lazy().join(dt.lazy().with_columns(alias_to_axis(pl.col("d"), "d"),
                                                  alias_to_axis(pl.col("t"), "t")).select("d", "t"),
                           how="cross")
    eta_lf = (_eff_param_to_pdt(source, "unit", "efficiency", dt, "eta")
              if parameter_has_values(source, "unit", "efficiency") else None)
    base = (base.join(eta_lf, on=["p", "d", "t"], how="left") if eta_lf is not None
            else base.with_columns(eta=pl.lit(None).cast(pl.Float64)))
    base = _mle_curve_inputs(source, dt, base.with_columns(eta=pl.col("eta").fill_null(1.0)))
    base = base.with_columns(section=_mle_section(pl.col("eta"), pl.col("eta_min"), pl.col("min_load")))
    # #10 / Q12: fold a direct unit's output/input conversion_flow_coeff ratio into its section —
    # section = section⁰/k_src (Q-A default; see p_slope_from_source).  No warning here — the DB-direct
    # zero-coefficient warning is emitted once, by p_slope_from_source.
    from flextool.engine.preprocess.params._arithmetic import _direct_flow_coef_ratio
    ratio = _direct_flow_coef_ratio(source, warn=False)
    if ratio.height > 0:
        base = (base.join(ratio.lazy().select("p", "k_src"), on="p", how="left")
                     .with_columns(
                         section=pl.when(pl.col("k_src").is_not_null())
                                   .then(pl.col("section") / pl.col("k_src"))
                                   .otherwise(pl.col("section"))))
    out = (base
                .select("p", "d", "t", pl.col("section").alias("value"))
                .sort("p", "d", "t")
                .collect())
    _raise_on_null_mle_cells(out.filter(pl.col("value").is_null()), "p_section")
    if out.height == 0:
        return None
    return Param(("p", "d", "t"), out)
