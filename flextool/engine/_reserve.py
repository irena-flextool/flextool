"""Reserves: capacity held back from the energy market to cover a reserve requirement of a node group.

Owns the variables v_reserve (reserve held per process, reserve, direction and node) and vq_reserve (the
shortfall of each reserve requirement), the requirement rows reserveBalance_timeseries_eq,
reserveBalance_dynamic_eq and reserveBalance_{up,down}_n_1_eq, the per-process limits
reserve_process_{upward,downward}, the storage energy rows reserve_storage_up_floor /
reserve_storage_down_headroom, and the shortfall penalty in the objective.  The coupling of v_reserve to
process capacity (maxFlow, maxOutputSum, profile limits, unit floors) lives with those families.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Var

from flextool.engine._axis_enums import rename_to_axis, schema_dtype
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine._model_common import _entity_rows, _has_rows, _var_view
from flextool.engine._param_shapes import promote_param_to_dt

if TYPE_CHECKING:
    from flextool.engine.model import _Build


def _check(d, fields: tuple[str, ...]) -> None:
    """Raise when the reserve feature is active but one of ``fields`` is not populated."""
    missing = [f for f in fields if getattr(d, f, None) is None]
    if missing:
        raise ValueError(
            f"_reserve: feature is active but data fields are not populated "
            f"(None): {missing}.  Populate them via _reserve.load_data()."
        )


# ---------------------------------------------------------------------------
# Loader


def _provider_get(provider, path: Path) -> "pl.DataFrame | None":
    """The provider's frame for ``path``, or None when the provider is missing or lacks it."""
    if provider is None:
        return None
    key = _provider_key(path)
    if not provider.has(key):
        return None
    return provider.get(key)


def _empty(cols: tuple[str, ...]) -> pl.DataFrame:
    """Empty frame with the axis dtype of every column in ``cols``."""
    return pl.DataFrame(schema={c: schema_dtype(None, c) for c in cols})


_PRUN = {"process": "p", "reserve": "r", "upDown": "ud", "node": "n"}
_RUG = {"reserve": "r", "upDown": "ud", "group": "g"}


def load_data(inp: Path | str, sd: Path | str, dt: pl.DataFrame | None = None, *, provider=None) -> dict:
    """Reserve sets and the reservation timeseries, as ``field_name -> DataFrame | Param``.

    The caller merges the dict into FlexData.  Empty when ``reserve__upDown__group`` has no rows (the
    feature is off).  ``inp`` / ``sd`` are the input and solve_data directories (files are read through
    ``provider``); ``dt`` clips the reservation to the steps of the horizon (None keeps every row).
    The remaining reserve fields (prundt, process_reserve_upDown_node_active, reliability, penalty,
    max_share, ratio values) are filled by the derived- and direct-parameter passes.
    """
    inp = Path(inp)
    sd = Path(sd)
    out: dict = {}

    rug = _provider_get(provider, sd / "reserve__upDown__group.csv")
    if rug is None or rug.height == 0:
        return out
    out["reserve_upDown_group"] = rug.pipe(rename_to_axis, _RUG).select("r", "ud", "g")

    # Requirement method partitions of reserve__upDown__group (always present, possibly empty).
    for method in ("timeseries", "dynamic", "n_1"):
        df = _provider_get(provider, sd / f"reserve__upDown__group__method_{method}.csv")
        if df is not None and df.height > 0:
            df = df.pipe(rename_to_axis, _RUG).select("r", "ud", "g", "method")
        else:
            df = pl.DataFrame(schema={"r": schema_dtype(None, "r"), "ud": schema_dtype(None, "ud"),
                                      "g": schema_dtype(None, "g"), "method": pl.Utf8})
        out[f"reserve_upDown_group_method_{method}"] = df

    # Processes whose flow drives a dynamic requirement / whose failure sets an n-1 requirement.
    for name in ("process_reserve_upDown_node_increase_reserve_ratio",
                 "process_reserve_upDown_node_large_failure_ratio"):
        df = _provider_get(provider, sd / f"{name}.csv")
        if df is not None:
            out[name] = (df.pipe(rename_to_axis, _PRUN).select("p", "r", "ud", "n") if df.height > 0
                         else _empty(("p", "r", "ud", "n")))

    # group_node: solve_data/group_node.csv, else the raw input/group__node.csv.  The first file with
    # rows wins; when the files exist but are empty the field is an empty frame.
    gn = None
    seen_any = False
    for path in (sd / "group_node.csv", inp / "group__node.csv"):
        df = _provider_get(provider, path)
        if df is None:
            continue
        seen_any = True
        if df.height > 0:
            gn = df.pipe(rename_to_axis, {"group": "g", "node": "n"}).select("g", "n").unique()
            break
    if gn is not None:
        out["group_node"] = gn
    elif seen_any:
        out["group_node"] = _empty(("g", "n"))

    # Reservation timeseries (r, ud, g, d, t), from the long-format pdtReserve_upDown_group.
    pdtR = _provider_get(provider, sd / "pdtReserve_upDown_group.csv")
    if pdtR is not None and pdtR.height > 0:
        pdtR = pdtR.pipe(rename_to_axis, {"reserve": "r", "upDown": "ud", "group": "g", "period": "d", "time": "t"})
        res_only = (pdtR.filter(pl.col("param") == "reservation")
                    .select("r", "ud", "g", "d", "t", "value")
                    .with_columns(value=pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0)))
        if dt is not None and dt.height > 0:
            res_only = res_only.join(dt, on=["d", "t"], how="inner")
        out["pdtReserve_upDown_group_reservation"] = Param(("r", "ud", "g", "d", "t"), res_only)

    return out


# ---------------------------------------------------------------------------
# Shortfall scale and penalty (shared with the output layer)


def _u8(lf: "pl.LazyFrame", keys) -> "pl.LazyFrame":
    """Cast ``keys`` to Utf8 (join keys across producers' Enum vocabularies)."""
    return lf.with_columns([pl.col(k).cast(pl.Utf8) for k in keys])


def _flow_requirement_bound(d, ratio_set, ratio_param, rug_sel, side: str) -> "pl.LazyFrame | None":
    """Largest value of one flow-driven reserve requirement, per ``(r, ud, g, d, t)`` (Utf8 keys).

    The requirement Σ v_flow · unitsize · ratio over the arcs whose ``side`` (``"sink"`` or ``"source"``)
    is a node n of group g with ``(p, r, ud, n)`` in ``ratio_set``, with v_flow replaced by its
    structural upper bound p_flow_upper (existing + the most the process can invest by that period, per
    unitsize; max_flow_for_unconstrained_variables for unlimited invest and conversion_flow_coeff = 0
    arcs).  None when there is nothing to sum.
    """
    if (ratio_set is None or ratio_set.height == 0 or ratio_param is None
            or rug_sel is None or rug_sel.height == 0):
        return None
    gn = getattr(d, "group_node", None)
    pss = getattr(d, "process_source_sink", None)
    fu = getattr(d, "p_flow_upper", None)
    us = getattr(d, "p_unitsize", None)
    if gn is None or gn.height == 0 or pss is None or pss.height == 0:
        return None
    if fu is None or us is None:
        raise ValueError(
            "_reserve: a dynamic / n-1 reserve requirement needs "
            "p_flow_upper and p_unitsize to size its shortfall variable.")
    sel = (_u8(ratio_set.lazy().select("p", "r", "ud", "n"), ["p", "r", "ud", "n"])
           .join(_u8(gn.lazy().select("g", "n"), ["g", "n"]), on="n", how="inner")
           .join(_u8(rug_sel.lazy().select("r", "ud", "g"), ["r", "ud", "g"]), on=["r", "ud", "g"], how="inner")
           .join(_u8(ratio_param.frame.lazy().select("p", "r", "ud", "n", "value"), ["p", "r", "ud", "n"])
                 .rename({"value": "__ratio"}),
                 on=["p", "r", "ud", "n"], how="inner"))
    arcs = _u8(pss.lazy().select("p", "source", "sink"), ["p", "source", "sink"]).with_columns(pl.col(side).alias("n"))
    upper = promote_param_to_dt(fu, d.dt)
    upper = _u8(upper.select("p", "source", "sink", "d", "t", "value"), ["p", "source", "sink", "d", "t"])
    us_lf = _u8(us.frame.lazy().select("p", "value"), ["p"]).rename({"value": "__us"})
    return (sel.join(arcs, on=["p", "n"], how="inner")
            .join(upper, on=["p", "source", "sink"], how="inner")
            .join(us_lf, on="p", how="inner")
            .with_columns((pl.col("value") * pl.col("__us") * pl.col("__ratio")).abs().alias("value"))
            .group_by("r", "ud", "g", "d", "t")
            .agg(pl.col("value").sum()))


def reserve_shortfall_scale(d) -> "Param | None":
    """Per-``(r, ud, g, d, t)`` multiplier of ``vq_reserve``.

    vq_reserve is a fraction in [0, 1] (bounded for numerics); the shortfall in MW is
    vq_reserve · scale, used identically in every reserveBalance row, in the objective penalty and in the
    output reconstruction.  scale is the largest requirement the group can have, so the shortfall can
    always cover the whole requirement:

    * reservation wherever it is authored (a timeseries-only model has scale = reservation);
    * dynamic groups: Σ over the flows that drive the requirement of
      p_flow_upper · unitsize · increase_reserve_ratio;
    * n-1 groups: Σ over the large-failure flows of p_flow_upper · unitsize · large_failure_ratio — the
      same sum over failing processes as the n-1 requirement (it must become a max if the requirement is
      ever made per failing process);
    * a group combining methods sums the applicable components.

    Returns the reservation Param itself when no dynamic / n-1 group exists; None when there is nothing
    to scale by.
    """
    rug = getattr(d, "reserve_upDown_group", None)
    if rug is None or rug.height == 0:
        return None
    res = getattr(d, "pdtReserve_upDown_group_reservation", None)
    irr_set = getattr(d, "process_reserve_upDown_node_increase_reserve_ratio", None)
    irr_val = getattr(d, "p_process_reserve_upDown_node_increase_reserve_ratio_value", None)
    lfr_set = getattr(d, "process_reserve_upDown_node_large_failure_ratio", None)
    lfr_val = getattr(d, "p_process_reserve_upDown_node_large_failure_ratio_value", None)
    parts: list = []
    method_dyn = getattr(d, "reserve_upDown_group_method_dynamic", None)
    if method_dyn is not None and method_dyn.height > 0:
        dyn_rug = method_dyn.select("r", "ud", "g").unique()
        for side in ("sink", "source"):
            part = _flow_requirement_bound(d, irr_set, irr_val, dyn_rug, side)
            if part is not None:
                parts.append(part)
    method_n1 = getattr(d, "reserve_upDown_group_method_n_1", None)
    if method_n1 is not None and method_n1.height > 0:
        for ud, side in (("up", "sink"), ("down", "source")):
            n1_rug = method_n1.filter(pl.col("ud").cast(pl.Utf8) == ud).select("r", "ud", "g").unique()
            part = _flow_requirement_bound(d, lfr_set, lfr_val, n1_rug, side)
            if part is not None:
                parts.append(part)
    if not parts:
        return res
    keys = ["r", "ud", "g", "d", "t"]
    if res is not None and res.frame.height > 0:
        parts.append(_u8(promote_param_to_dt(res, d.dt).select(*keys, "value"), keys))
    comp = (pl.concat([p.select(*keys, pl.col("value").cast(pl.Float64)) for p in parts])
            .group_by(keys).agg(pl.col("value").sum()))
    base = rug.select("r", "ud", "g").join(d.dt.select("d", "t"), how="cross")
    frame = (_u8(base.lazy(), ["r", "ud", "g"])
             .with_columns(pl.col("d").cast(pl.Utf8).alias("__d"), pl.col("t").cast(pl.Utf8).alias("__t"))
             .join(comp.rename({"d": "__d", "t": "__t"})
                   .with_columns(pl.col("__d").cast(pl.Utf8), pl.col("__t").cast(pl.Utf8)),
                   on=["r", "ud", "g", "__d", "__t"], how="left")
             .with_columns(pl.col("value").fill_null(0.0))
             .filter(pl.col("value") != 0.0)
             .collect())
    if frame.height == 0:
        return None
    # Restore the key dtypes of ``base`` (Enum axes).
    frame = (base.with_columns(pl.col("r").cast(pl.Utf8).alias("__r"),
                               pl.col("ud").cast(pl.Utf8).alias("__ud"),
                               pl.col("g").cast(pl.Utf8).alias("__g"),
                               pl.col("d").cast(pl.Utf8).alias("__d"),
                               pl.col("t").cast(pl.Utf8).alias("__t"))
             .join(frame.select(pl.col("r").alias("__r"), pl.col("ud").alias("__ud"), pl.col("g").alias("__g"),
                                "__d", "__t", "value"),
                   on=["__r", "__ud", "__g", "__d", "__t"], how="inner")
             .select(*keys, "value"))
    return Param(tuple(keys), frame)


def reserve_penalty(d) -> "Param | None":
    """``penalty_reserve`` dense over ``reserve_upDown_group``.

    Every ``(r, ud, g)`` gets a penalty: the authored ``p_reserve_upDown_group_penalty_reserve`` where
    present, else :data:`~flextool.engine._direct_params.PENALTY_RESERVE_DEFAULT` — a group with a
    requirement but no authored penalty must not get a free shortfall.  None only when the feature is off.
    """
    from flextool.engine._direct_params import PENALTY_RESERVE_DEFAULT

    rug = getattr(d, "reserve_upDown_group", None)
    if rug is None or rug.height == 0:
        return None
    keys = ["r", "ud", "g"]
    base = rug.lazy().select(keys).unique(maintain_order=True)
    pen = getattr(d, "p_reserve_upDown_group_penalty_reserve", None)
    if pen is not None:
        authored = _u8(pen.frame.lazy().select(*keys, "value"), keys)
        base = (base.with_columns(pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in keys)
                .join(authored.rename({k: f"__{k}" for k in keys}), on=[f"__{k}" for k in keys], how="left")
                .drop([f"__{k}" for k in keys]))
    else:
        base = base.with_columns(value=pl.lit(None, dtype=pl.Float64))
    out = base.with_columns(pl.col("value").cast(pl.Float64).fill_null(PENALTY_RESERVE_DEFAULT)).collect()
    return Param(("r", "ud", "g"), out.lazy())


# ---------------------------------------------------------------------------
# Variables


def add_variables(b: "_Build") -> None:
    """Declare v_reserve and vq_reserve."""
    if not b.has_reserve:
        return
    m, d = b.m, b.d
    _check(d, ("reserve_upDown_group", "prundt", "dt"))

    # ─── VAR. ─── v_reserve ──────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve[p,r,ud,n,d,t] ≥ 0   ∀ (p,r,ud,n,d,t) ∈ prundt
    # Reserve r in direction ud (up / down) that process p holds for node n,
    # in units of unitsize like v_flow: MW = v_reserve · unitsize[p].
    b.v_reserve = m.add_var("v_reserve", ("p", "r", "ud", "n", "d", "t"), d.prundt, lower=0.0)

    # ─── VAR. ─── vq_reserve ─────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # 0 ≤ vq_reserve[r,ud,g,d,t] ≤ 1   ∀ (r,ud,g) ∈ reserve_upDown_group, (d,t) ∈ dt
    # Shortfall of the reserve requirement of group g, as a fraction of the
    # largest requirement: MW = vq_reserve · scale[r,ud,g,d,t]
    # (reserve_shortfall_scale).
    b.vq_reserve = m.add_var("vq_reserve", ("r", "ud", "g", "d", "t"),
                             d.reserve_upDown_group.join(d.dt, how="cross"), lower=0.0, upper=1.0)


# ---------------------------------------------------------------------------
# Constraints


def _flow_node_view(v_flow: Var, side: str) -> Var:
    """``v_flow`` with its ``side`` axis (``"source"`` or ``"sink"``) relabelled to the node axis ``n``.

    The dynamic and n-1 requirements sum v_flow over the flows whose ``side`` node is the reserve node n.
    Relabelling ties that axis to n (and to the ratio Param's n), so the enclosing Sum collapses it;
    a separately renamed join column would leave ``source`` / ``sink`` open.  The view keeps the LP
    columns and the name ``v_flow`` (autoscale keys on it).
    """
    return Var(name=v_flow.name,
               dims=tuple("n" if c == side else c for c in v_flow.dims),
               frame=v_flow.frame.pipe(rename_to_axis, {side: "n"}))


def add_constraints(b: "_Build") -> None:
    """Reserve requirements of the node groups, per-process reserve limits and the storage energy rows."""
    if not b.has_reserve:
        return
    m, d = b.m, b.d
    _check(d, ("reserve_upDown_group", "prundt", "dt", "process_reserve_upDown_node_active", "group_node"))
    v_reserve, vq_reserve, v_flow = b.v_reserve, b.vq_reserve, b.v_flow
    p_unitsize = d.p_unitsize
    rug = d.reserve_upDown_group                           # (r, ud, g)
    gn = d.group_node.pipe(rename_to_axis, {"n": "n"})     # (g, n), n cast to the node axis

    # Left-hand side shared by every reserveBalance row, per (r, ud, g, d, t):
    #   reserve_sum = Σ_{(p,n) : (p,r,ud,n) ∈ process_reserve_upDown_node_active, (g,n) ∈ group_node}
    #                   v_reserve[p,r,ud,n,d,t] · unitsize[p] · reliability[p,r,ud,n]
    #   vq_term     = vq_reserve[r,ud,g,d,t] · scale[r,ud,g,d,t]
    pruna_g = (d.process_reserve_upDown_node_active
               .join(gn, on="n", how="inner")
               .join(rug, on=["r", "ud", "g"], how="inner")
               .select("p", "r", "ud", "n", "g"))
    lhs_reserve_core = None
    if d.p_process_reserve_upDown_node_reliability is not None and pruna_g.height > 0:
        lhs_reserve_core = Sum(Where(v_reserve * p_unitsize * d.p_process_reserve_upDown_node_reliability, pruna_g),
                               over=("p", "n"))
    scale = reserve_shortfall_scale(d)
    lhs_vq = vq_reserve * scale if scale is not None else None

    # ─── EQ. ─── reserveBalance_timeseries_eq ────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # reserve_sum[r,ud,g,d,t] + vq_term[r,ud,g,d,t]  ≥ reservation[r,ud,g,d,t]
    #   ∀ (r,ud,g) ∈ reserve_upDown_group_method_timeseries, (d,t) ∈ dt
    # Reserve held for a group covers its authored reservation timeseries,
    # or the shortfall is penalised.  RHS 0 when no reservation is authored.
    # Reserve of every provider counts at full MW (no efficiency slope).
    method_ts = d.reserve_upDown_group_method_timeseries
    if method_ts is not None and method_ts.height > 0:
        ts_rug = method_ts.select("r", "ud", "g").unique()
        rhs = (d.pdtReserve_upDown_group_reservation
               if d.pdtReserve_upDown_group_reservation is not None else 0.0)
        lhs_terms = {}
        if lhs_vq is not None:
            lhs_terms["vq_term"] = lhs_vq
        if lhs_reserve_core is not None:
            lhs_terms["reserve_sum"] = Where(lhs_reserve_core, ts_rug)
        m.add_cstr("reserveBalance_timeseries_eq", over=ts_rug.join(d.dt, how="cross"), sense=">=",
                   lhs_terms=lhs_terms, rhs_terms={"demand": rhs})

    # ─── EQ. ─── reserveBalance_dynamic_eq ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # reserve_sum[r,ud,g,d,t] + vq_term[r,ud,g,d,t]
    #   ≥ Σ_{(p,s,n) ∈ process_source_sink, (p,r,ud,n) ∈ irr_g} v_flow[p,s,n,d,t] · unitsize[p] · ratio[p,r,ud,n]
    #   + Σ_{(p,n,k) ∈ process_source_sink, (p,r,ud,n) ∈ irr_g} v_flow[p,n,k,d,t] · unitsize[p] · ratio[p,r,ud,n]
    #   ∀ (r,ud,g) ∈ reserve_upDown_group_method_dynamic, (d,t) ∈ dt
    #   irr_g = process_reserve_upDown_node_increase_reserve_ratio with n ∈ group g
    #   ratio = increase_reserve_ratio
    # The requirement grows with the flows into and out of the group's nodes
    # (e.g. variable generation and demand).  RHS 0 when no flow drives it.
    # Flows count at unitsize, without the efficiency slope.
    method_dyn = d.reserve_upDown_group_method_dynamic
    if method_dyn is not None and method_dyn.height > 0:
        dyn_rug = method_dyn.select("r", "ud", "g").unique()
        rhs_terms: dict = {}
        irr_set = getattr(d, "process_reserve_upDown_node_increase_reserve_ratio", None)
        irr_param = getattr(d, "p_process_reserve_upDown_node_increase_reserve_ratio_value", None)
        if (v_flow is not None and irr_set is not None and irr_set.height > 0
                and irr_param is not None and d.process_source_sink is not None):
            irr_g = (irr_set.join(gn, on="n", how="inner")
                     .join(dyn_rug, on=["r", "ud", "g"], how="inner")
                     .select("p", "r", "ud", "n", "g"))
            if irr_g.height > 0:
                pss = d.process_source_sink
                pss_as_sink = pss.pipe(rename_to_axis, {"sink": "n"}).select("p", "source", "n")
                into_n = Sum(Where(_flow_node_view(v_flow, "sink") * p_unitsize * irr_param,
                                   pss_as_sink.join(irr_g, on=["p", "n"], how="inner")),
                             over=("p", "source", "n"))
                pss_as_source = pss.pipe(rename_to_axis, {"source": "n"}).select("p", "n", "sink")
                out_of_n = Sum(Where(_flow_node_view(v_flow, "source") * p_unitsize * irr_param,
                                     pss_as_source.join(irr_g, on=["p", "n"], how="inner")),
                               over=("p", "sink", "n"))
                rhs_terms["dynamic_flow"] = into_n + out_of_n
        if not rhs_terms:
            rhs_terms = {"zero": 0.0}
        lhs_terms = {}
        if lhs_vq is not None:
            lhs_terms["vq_term"] = Where(lhs_vq, dyn_rug)
        if lhs_reserve_core is not None:
            lhs_terms["reserve_sum"] = Where(lhs_reserve_core, dyn_rug)
        m.add_cstr("reserveBalance_dynamic_eq", over=dyn_rug.join(d.dt, how="cross"), sense=">=",
                   lhs_terms=lhs_terms, rhs_terms=rhs_terms)

    # ─── EQ. ─── reserveBalance_up_n_1_eq / reserveBalance_down_n_1_eq ───────
    # ─────────────────────────────────────────────────────────────────────────
    # up:   reserve_sum[r,ud,g,d,t] + vq_term[r,ud,g,d,t]
    #         ≥ Σ_{(p,s,n) ∈ process_source_sink, (p,r,ud,n) ∈ lfr_g} v_flow[p,s,n,d,t] · unitsize[p] · ratio[p,r,ud,n]
    # down: reserve_sum[r,ud,g,d,t] + vq_term[r,ud,g,d,t]
    #         ≥ Σ_{(p,n,k) ∈ process_source_sink, (p,r,ud,n) ∈ lfr_g} v_flow[p,n,k,d,t] · unitsize[p] · ratio[p,r,ud,n]
    #   ∀ (r,ud,g) ∈ reserve_upDown_group_method_n_1 with ud = up / down, (d,t) ∈ dt
    #   lfr_g = process_reserve_upDown_node_large_failure_ratio with n ∈ group g
    #   ratio = large_failure_ratio
    # Reserve covers the loss of large processes: upward reserve the loss of
    # their output into n, downward reserve the loss of their consumption
    # from n.  One row per (r,ud,g,d,t): the requirement is the sum over all
    # large-failure processes, not a row per failing process.  RHS 0 when no
    # flow drives it.
    method_n1 = d.reserve_upDown_group_method_n_1
    if method_n1 is not None and method_n1.height > 0:
        lfr_set = getattr(d, "process_reserve_upDown_node_large_failure_ratio", None)
        lfr_param = getattr(d, "p_process_reserve_upDown_node_large_failure_ratio_value", None)
        for ud_filter, name, side, other in (("up", "reserveBalance_up_n_1_eq", "sink", "source"),
                                             ("down", "reserveBalance_down_n_1_eq", "source", "sink")):
            n1_rug = method_n1.filter(pl.col("ud") == ud_filter).select("r", "ud", "g").unique()
            if n1_rug.height == 0:
                continue
            rhs_terms: dict = {}
            if (v_flow is not None and lfr_set is not None and lfr_set.height > 0
                    and lfr_param is not None and d.process_source_sink is not None):
                lfr_g = (lfr_set.join(gn, on="n", how="inner")
                         .join(n1_rug, on=["r", "ud", "g"], how="inner")
                         .select("p", "r", "ud", "n", "g"))
                if lfr_g.height > 0:
                    # n is the sink of the failing flow (up) or its source (down).
                    pss_idx = (d.process_source_sink.pipe(rename_to_axis, {side: "n"})
                               .select("p", *(("source", "n") if side == "sink" else ("n", "sink"))))
                    rhs_terms["n_1_failure"] = Sum(
                        Where(_flow_node_view(v_flow, side) * p_unitsize * lfr_param,
                              pss_idx.join(lfr_g, on=["p", "n"], how="inner")),
                        over=("p", other, "n"))
            if not rhs_terms:
                rhs_terms = {"zero": 0.0}
            lhs_terms = {}
            if lhs_vq is not None:
                lhs_terms["vq_term"] = Where(lhs_vq, n1_rug)
            if lhs_reserve_core is not None:
                lhs_terms["reserve_sum"] = Where(lhs_reserve_core, n1_rug)
            m.add_cstr(name, over=n1_rug.join(d.dt, how="cross"), sense=">=",
                       lhs_terms=lhs_terms, rhs_terms=rhs_terms)

    # ─── EQ. ─── reserve_process_upward / reserve_process_downward ───────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve[p,r,ud,n,d,t] − max_share[p,r,ud,n] · Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + max_share[p,r,ud,n] · Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    #   ≤ max_share[p,r,ud,n] · existing_count[p,d]
    #   ∀ (p,r,ud,n,d,t) ∈ prundt with ud = up / down
    # A process holds at most max_share of its built units as reserve, in
    # units of unitsize.  The limit uses the built capacity, also for units
    # with online variables.
    max_share = getattr(d, "p_process_reserve_upDown_node_max_share", None)
    existing = getattr(d, "p_process_existing_count", None)
    if max_share is not None and d.prundt.height > 0 and existing is not None:
        edd_inv_set = getattr(d, "edd_invest_set", None)
        edd_div_act = getattr(d, "edd_divest_active", None)
        for ud_filter, name in (("up", "reserve_process_upward"), ("down", "reserve_process_downward")):
            idx = d.prundt.filter(pl.col("ud") == ud_filter)
            if idx.height == 0:
                continue
            rhs_param = max_share * existing
            lhs: dict = {"reserve": v_reserve}
            p_in_idx = idx.select("p").unique()
            if b.v_invest_p is not None and edd_inv_set is not None and edd_inv_set.height > 0:
                edd_p = edd_inv_set.pipe(rename_to_axis, {"e": "p"}).join(p_in_idx, on="p", how="inner")
                if edd_p.height > 0:
                    v_inv_at = _var_view(b.v_invest_p, f"__at_{name}", {"d": "d_invest"}, to_axis=True)
                    lhs["invest_neg"] = -(Sum(Where(v_inv_at, edd_p), over=("d_invest",)) * max_share)
            if b.v_divest_p is not None and edd_div_act is not None and edd_div_act.height > 0:
                edd_p_div = edd_div_act.join(p_in_idx, on="p", how="inner")
                if edd_p_div.height > 0:
                    v_div_at = _var_view(b.v_divest_p, f"__at_{name}", {"d": "d_divest"}, to_axis=True)
                    lhs["divest"] = Sum(Where(v_div_at, edd_p_div), over=("d_divest",)) * max_share
            m.add_cstr(name, over=idx, sense="<=", lhs_terms=lhs, rhs_terms={"max": rhs_param})

    _add_storage_energy_constraints(b)


def _add_storage_energy_constraints(b: "_Build") -> None:
    """Storage-backed reserve providers hold the energy to sustain their reserve for reserve_duration."""
    d = b.d
    dur = getattr(d, "p_reserve_upDown_group_reserve_duration", None)
    if (dur is None or not _has_rows(d.nodeState) or not _has_rows(d.process_source_sink)
            or b.v_state is None or d.p_state_unitsize is None):
        return
    m = b.m
    v_state = b.v_state
    us_n = d.p_state_unitsize
    dt = d.dt
    gn = d.group_node.pipe(rename_to_axis, {"n": "n"})     # (g, n), n cast to the node axis
    dur_set = dur.frame.select("r", "ud", "g")

    def _providers(ud_val: str, balance_side: str, storage_side: str) -> "pl.DataFrame | None":
        """(p, r, ud, nb, n, g): providers of reserve ud for balance node nb drawing on storage node n.

        ``ud`` is filtered before the topology join so the two orientations of a reversible battery do
        not mix.  Groups without a reserve_duration drop out.
        """
        pr = d.process_reserve_upDown_node_active.filter(pl.col("ud") == ud_val)
        if pr.height == 0:
            return None
        j = d.process_source_sink.pipe(rename_to_axis, {balance_side: "n"}).join(pr, on=["p", "n"], how="inner")
        if j.height == 0:
            return None
        j = (j.rename({"n": "nb"})
             .pipe(rename_to_axis, {storage_side: "n"})
             .join(gn.rename({"n": "nb"}), on="nb", how="inner")
             .join(d.reserve_upDown_group, on=["r", "ud", "g"], how="inner")
             .join(dur_set, on=["r", "ud", "g"], how="inner")
             .join(d.nodeState.select("n"), on="n", how="semi")
             .select("p", "r", "ud", "nb", "n", "g")
             .unique())
        return j if j.height > 0 else None

    def _slope_factor(providers: pl.DataFrame, reciprocal: bool) -> Param:
        """slope[p,d,t] over the providers × dt, 1.0 where p_slope has no value (noEff providers);
        1 / slope when ``reciprocal``."""
        grid = providers.join(dt, how="cross")
        if d.p_slope is not None:
            grid = grid.join(d.p_slope.frame, on=["p", "d", "t"], how="left")
            val = pl.col("value").fill_null(1.0)
        else:
            val = pl.lit(1.0)
        if reciprocal:
            val = 1.0 / val
        return Param(("p", "d", "t"), grid.with_columns(value=val).select("p", "d", "t", "value"))

    # v_reserve with its balance node relabelled nb: the storage node n added by the providers join is
    # the only n left after the sum, aligned with v_state.
    v_res_nb = Var(name=b.v_reserve.name,
                   dims=tuple("nb" if c == "n" else c for c in b.v_reserve.dims),
                   frame=b.v_reserve.frame.rename({"n": "nb"}))

    # ─── EQ. ─── reserve_storage_up_floor ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] · state_unitsize[n]
    #   − Σ_{(p,r,ud,nb,g) ∈ providers_up(n)} v_reserve[p,r,ud,nb,d,t] · unitsize[p] · duration[r,ud,g] · slope[p,d,t]
    #   ≥ 0
    #   ∀ n ∈ storage nodes of upward providers, (d,t) ∈ dt
    #   providers_up: upward reserve for nb by a process with source n ∈ nodeState and sink nb
    # Upward reserve from storage is backed by stored energy (MWh) for the
    # reserve duration (hours); slope = 1/η converts output to energy drawn.
    # Reliability is not applied: the energy matches the full committed MW.
    # No slack: with v_reserve = 0 the row is v_state ≥ 0, so a provider
    # that cannot sustain the duration commits less and the group pays the
    # shortfall penalty.  Couples to the intra-period state v_state only,
    # not to the representative-period state v_state_inter.
    join_up = _providers("up", balance_side="sink", storage_side="source")
    if join_up is not None:
        slope_up = _slope_factor(join_up.select("p").unique(), reciprocal=False)
        e_up = Sum(Where(v_res_nb * d.p_unitsize * dur * slope_up, join_up), over=("p", "r", "ud", "nb", "g"))
        m.add_cstr("reserve_storage_up_floor", over=join_up.select("n").unique().join(dt, how="cross"), sense=">=",
                   lhs_terms={"state": v_state * us_n, "energy_neg": -e_up}, rhs_terms={"zero": 0.0})

    # ─── EQ. ─── reserve_storage_down_headroom ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] · state_unitsize[n]
    #   + Σ_{(p,r,ud,nb,g) ∈ providers_down(n)} v_reserve[p,r,ud,nb,d,t] · unitsize[p] · duration[r,ud,g] / slope[p,d,t]
    #   + Σ_{(n,d′,d) ∈ edd_divest_active} v_divest_n[n,d′] · state_unitsize[n]
    #   − Σ_{(n,d′,d) ∈ edd_invest_set} v_invest_n[n,d′] · state_unitsize[n]
    #   ≤ state_upper[n,d,t] · state_unitsize[n]
    #   ∀ n ∈ storage nodes of downward providers, (d,t) ∈ dt
    #   providers_down: downward reserve for nb by a process with source nb and sink n ∈ nodeState
    # Downward reserve into storage needs free storage capacity (MWh) for the
    # reserve duration; 1/slope = η converts input to energy stored.  The
    # capacity terms are those of maxState, in MWh.  Same notes as the floor.
    join_dn = _providers("down", balance_side="source", storage_side="sink")
    if join_dn is not None:
        slope_dn = _slope_factor(join_dn.select("p").unique(), reciprocal=True)
        e_dn = Sum(Where(v_res_nb * d.p_unitsize * dur * slope_dn, join_dn), over=("p", "r", "ud", "nb", "g"))
        state_lhs: dict = {"state": v_state * us_n, "energy": e_dn}
        if b.v_divest_n is not None and d.edd_divest_active is not None:
            v_div_n_at = _var_view(b.v_divest_n, "__at_reserve_down_divest", {"d": "d_divest"}, to_axis=True)
            edd_div_n = _entity_rows(d.edd_divest_active, "p", d.nodeState, "n")
            if edd_div_n.height > 0:
                state_lhs["divest"] = Sum(Where(v_div_n_at, edd_div_n), over=("d_divest",)) * us_n
        if b.v_invest_n is not None and d.edd_invest_set is not None:
            v_inv_n_at = _var_view(b.v_invest_n, "__at_reserve_down_invest", {"d": "d_invest"}, to_axis=True)
            edd_inv_n = _entity_rows(d.edd_invest_set, "e", d.nodeState, "n")
            if edd_inv_n.height > 0:
                state_lhs["invest_neg"] = -(Sum(Where(v_inv_n_at, edd_inv_n), over=("d_invest",)) * us_n)
        m.add_cstr("reserve_storage_down_headroom", over=join_dn.select("n").unique().join(dt, how="cross"),
                   sense="<=", lhs_terms=state_lhs, rhs_terms={"upper": d.p_state_upper * us_n})


# ---------------------------------------------------------------------------
# Objective


def objective_terms(b: "_Build"):
    """Penalty of the reserve shortfall."""
    if not b.has_reserve:
        return None
    d = b.d

    # ─── OBJ. ─── reserve shortfall ──────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # + Σ_{(r,ud,g,d,t)} vq_reserve[r,ud,g,d,t] · scale[r,ud,g,d,t] · penalty_reserve[r,ud,g] · w_op[d,t]
    # Prices the shortfall in MW (vq_reserve · scale, the multiplier of the
    # reserveBalance rows) per hour of operation.  penalty_reserve is the
    # authored value, else the default (reserve_penalty).  w_op = b.op_factor.
    scale = reserve_shortfall_scale(d)
    pen = reserve_penalty(d)
    if scale is None or pen is None:
        return None
    return Sum(b.vq_reserve * scale * pen * b.op_factor)
