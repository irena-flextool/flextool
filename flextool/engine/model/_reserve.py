"""Reserves: capacity held back from the energy market to cover a reserve requirement of a node group.

Owns the variables v_reserve (reserve held per process, reserve, direction and node), vq_reserve (the
shortfall of each reserve requirement) and v_reserve_total (the total reserve of a group with n-1 rows), the
requirement rows reserveBalance_timeseries_eq, reserveBalance_dynamic_eq and reserveBalance_{up,down}_n_1_eq,
the bound reserve_total_upper, the per-process limits
reserve_process_{upward,downward}, the connection rows connection_reserve_{up,down}[_online_*|_exact] (a
connection's reserve limited by its net position at the node), the storage energy rows reserve_storage_up_floor /
reserve_storage_down_headroom, and the shortfall penalty in the objective.  The coupling of units' v_reserve to
process capacity (maxFlow, maxOutputSum, maxFlow_online, maxOutputSum_online, maxInputFlow_online, profile limits,
unit floors) lives with those families.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Var

from flextool.engine.input._axis_enums import rename_to_axis, schema_dtype
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.model._common import (
    _connection_procs, _drop_uncapped, _entity_rows, _has_rows, _indirect_input_cap, _utf8_keys, _var_view,
)
from flextool.engine.input._param_shapes import promote_param_to_dt

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


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


def _flow_requirement_bound(d, ratio_set, ratio_param, rug_sel, side: str, *,
                            per_process_max: bool = False) -> "pl.LazyFrame | None":
    """Largest value of one flow-driven reserve requirement, per ``(r, ud, g, d, t)`` (Utf8 keys).

    The requirement Σ v_flow · unitsize · ratio over the arcs whose ``side`` (``"sink"`` or ``"source"``)
    is a node n of group g with ``(p, r, ud, n)`` in ``ratio_set``, with v_flow replaced by its
    structural upper bound p_flow_upper (existing + the most the process can invest by that period, per
    unitsize; max_flow_for_unconstrained_variables for unlimited invest and conversion_flow_coeff = 0
    arcs).  With ``per_process_max`` the bound is summed over each process's own arcs and the result is
    the max over processes (the n-1 requirement: one row per failing process); otherwise the sum over
    all arcs (the dynamic requirement).  On the source side the bound of an arc in process_source_sink_eff is
    multiplied by the full-load draw per unit of flow, ``slope + section`` (:func:`_with_full_load_draw`):
    the requirement counts the flow drawn from the node, ``v_flow · slope + v_online · section``, and
    ``v_online ≤ p_flow_upper``.  None when there is nothing to sum.
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
    bound = (sel.join(arcs, on=["p", "n"], how="inner")
             .join(upper, on=["p", "source", "sink"], how="inner")
             .join(us_lf, on="p", how="inner")
             .with_columns((pl.col("value") * pl.col("__us") * pl.col("__ratio")).abs().alias("value")))
    if side == "source":
        bound = _with_full_load_draw(d, bound)
    # The reverse flow v_flow_back of one-variable and DC arcs arrives at the arc's source and leaves its sink;
    # its bound is the arc's (maxFlow_back), lossless (no full-load draw factor).
    back = _back_arcs(d)
    if back is not None:
        other = "source" if side == "sink" else "sink"
        b_arcs = _u8(back.lazy().select("p", "source", "sink"), ["p", "source", "sink"]).with_columns(
            pl.col(other).alias("n"))
        b_bound = (sel.join(b_arcs, on=["p", "n"], how="inner")
                   .join(upper, on=["p", "source", "sink"], how="inner")
                   .join(us_lf, on="p", how="inner")
                   .with_columns((pl.col("value") * pl.col("__us") * pl.col("__ratio")).abs().alias("value")))
        bound = pl.concat([bound, b_bound.select(bound.collect_schema().names())])
    if per_process_max:
        return (bound.group_by("r", "ud", "g", "p", "d", "t").agg(pl.col("value").sum())
                .group_by("r", "ud", "g", "d", "t").agg(pl.col("value").max()))
    return bound.group_by("r", "ud", "g", "d", "t").agg(pl.col("value").sum())


def _back_arcs(d) -> "pl.DataFrame | None":
    """(p, source, sink) of the arcs with a reverse-flow variable v_flow_back (one-variable and DC arcs), as
    model/core.py declares them; None when there are none."""
    from flextool.engine.model.core import _two_way_one_var_arcs

    if not _has_rows(getattr(d, "process_source_sink", None)):
        return None
    arcs = _two_way_one_var_arcs(d)
    return arcs if _has_rows(arcs) else None


def _with_full_load_draw(d, bound: "pl.LazyFrame") -> "pl.LazyFrame":
    """``bound`` (Utf8 keys p, source, sink, d, t; ``value``) with ``value`` × (slope + section) on the arcs of
    process_source_sink_eff: the input drawn at full load per unit of flow.  slope 1 where p_slope has no row;
    section only for min_load_efficiency units with online variables (as in nodeBalance_eq).  Unchanged
    without process_source_sink_eff or p_slope (hand-built data)."""
    eff = getattr(d, "process_source_sink_eff", None)
    slope = getattr(d, "p_slope", None)
    if eff is None or eff.height == 0 or slope is None:
        return bound
    keys = ["p", "source", "sink"]
    eff_lf = _u8(eff.lazy().select(keys), keys).unique().with_columns(pl.lit(True).alias("__eff"))
    slope_lf = _u8(promote_param_to_dt(slope, d.dt).select("p", "d", "t", pl.col("value").alias("__slope")),
                   ["p", "d", "t"])
    out = (bound.join(eff_lf, on=keys, how="left")
           .join(slope_lf, on=["p", "d", "t"], how="left"))
    out = _with_section(d, out.rename({"p": "__p"})).rename({"__p": "p"})
    factor = pl.col("__slope").fill_null(1.0) + pl.col("__sec").fill_null(0.0)
    return (out.with_columns(pl.when(pl.col("__eff").fill_null(False)).then(pl.col("value") * factor)
                             .otherwise(pl.col("value")).alias("value"))
            .drop("__eff", "__slope", "__sec"))


def group_demand_requirement(d) -> "Param | None":
    """The demand term of the dynamic requirement, MW per ``(r, ud, g, d, t)``.

    group_demand[r,ud,g,d,t] = ratio_g[r,ud,g] · Σ_{n : (g,n) ∈ group_node} max(−inflow[n,d,t], 0) / Δt[d,t]
    for the groups of ``reserve_upDown_group_method_dynamic`` with a ``reserve__upDown__group``
    ``increase_reserve_ratio`` (``p_reserve_upDown_group_increase_reserve_ratio``).  ``inflow`` is MWh per
    step (``nodeBalance_eq``), so dividing by the step duration gives MW in the step.  The demand of each
    node is its negative inflow (``p_negative_inflow``: the real demand, before the invest-solve energy
    margin, which is an adequacy device and not load), clipped per node before the group sum, so a node with
    positive inflow neither counts nor offsets another node's demand.  Zero rows are dropped.  Shared by
    the reserveBalance_dynamic_eq row and the shortfall scale.  None when no group has a ratio.
    """
    ratio = getattr(d, "p_reserve_upDown_group_increase_reserve_ratio", None)
    method_dyn = getattr(d, "reserve_upDown_group_method_dynamic", None)
    gn = getattr(d, "group_node", None)
    neg = getattr(d, "p_negative_inflow", None)
    step = getattr(d, "p_step_duration", None)
    if (ratio is None or method_dyn is None or method_dyn.height == 0 or gn is None or gn.height == 0
            or neg is None or step is None):
        return None
    keys = ["r", "ud", "g", "d", "t"]
    dyn_rug = method_dyn.select("r", "ud", "g").unique()
    sel = (_u8(ratio.frame.lazy().select("r", "ud", "g", pl.col("value").alias("__ratio")), ["r", "ud", "g"])
           .join(_u8(dyn_rug.lazy(), ["r", "ud", "g"]), on=["r", "ud", "g"], how="semi")
           .join(_u8(gn.lazy().select("g", "n"), ["g", "n"]).unique(), on="g", how="inner"))
    dt_u = _u8(d.dt.lazy().select("d", "t"), ["d", "t"])
    demand = (_u8(neg.frame.lazy().select("n", "d", "t", "value"), ["n", "d", "t"])
              .join(dt_u, on=["d", "t"], how="semi")
              .with_columns((-pl.col("value")).clip(lower_bound=0.0).alias("__dem")))
    step_lf = _u8(step.frame.lazy().select("d", "t", pl.col("value").alias("__dt")), ["d", "t"])
    comp = (sel.join(demand, on="n", how="inner")
            .group_by("r", "ud", "g", "d", "t", "__ratio").agg(pl.col("__dem").sum())
            .join(step_lf, on=["d", "t"], how="inner")
            .select("r", "ud", "g", "d", "t", value=pl.col("__ratio") * pl.col("__dem") / pl.col("__dt"))
            .filter(pl.col("value") != 0.0))
    base = dyn_rug.join(d.dt.select("d", "t"), how="cross")
    frame = (base.with_columns([pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in keys])
             .lazy()
             .join(comp.rename({k: f"__{k}" for k in keys}), on=[f"__{k}" for k in keys], how="inner")
             .select(*keys, "value")
             .sort(keys)
             .collect())
    if frame.height == 0:
        return None
    return Param(tuple(keys), frame)


def reserve_shortfall_scale(d) -> "Param | None":
    """Per-``(r, ud, g, d, t)`` multiplier of ``vq_reserve``.

    vq_reserve is a fraction in [0, 1] (bounded for numerics); the shortfall in MW is
    vq_reserve · scale, used identically in every reserveBalance row, in the objective penalty and in the
    output reconstruction.  scale is the largest requirement the group can have, so the shortfall can
    always cover the whole requirement:

    * reservation wherever it is authored (a timeseries-only model has scale = reservation);
    * dynamic groups: Σ over the flows that drive the requirement of
      p_flow_upper · unitsize · increase_reserve_ratio (× slope + section for a flow drawn on an
      efficiency arc), plus the group's demand term (:func:`group_demand_requirement`, exact);
    * n-1 groups: the max over failing processes of Σ over that process's large-failure flows of
      p_flow_upper · unitsize · large_failure_ratio — the largest single n-1 row (one row per failing
      process);
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
        demand = group_demand_requirement(d)
        if demand is not None:
            parts.append(_u8(demand.frame.lazy(), ["r", "ud", "g", "d", "t"]))
    method_n1 = getattr(d, "reserve_upDown_group_method_n_1", None)
    if method_n1 is not None and method_n1.height > 0:
        for ud, side in (("up", "sink"), ("down", "source")):
            n1_rug = method_n1.filter(pl.col("ud").cast(pl.Utf8) == ud).select("r", "ud", "g").unique()
            part = _flow_requirement_bound(d, lfr_set, lfr_val, n1_rug, side, per_process_max=True)
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
    present, else :data:`~flextool.engine.preprocess.params._direct.PENALTY_RESERVE_DEFAULT` — a group with a
    requirement but no authored penalty must not get a free shortfall.  None only when the feature is off.
    """
    from flextool.engine.preprocess.params._direct import PENALTY_RESERVE_DEFAULT

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
# Index sets shared by the variables and the rows

# (ud, n-1 row family, side of the failing flow that is a group node, the other side)
_N_1_DIRECTIONS = (("up", "reserveBalance_up_n_1_eq", "sink", "source"),
                   ("down", "reserveBalance_down_n_1_eq", "source", "sink"))


def _group_nodes(d) -> pl.DataFrame:
    """(g, n) of group_node, n cast to the node axis."""
    return d.group_node.pipe(rename_to_axis, {"n": "n"})


def _providers_by_group(d, gn: pl.DataFrame) -> pl.DataFrame:
    """pruna_g (p, r, ud, n, g): the reserve providers of each reserve group, at the group's nodes."""
    return (d.process_reserve_upDown_node_active
            .join(gn, on="n", how="inner")
            .join(d.reserve_upDown_group, on=["r", "ud", "g"], how="inner")
            .select("p", "r", "ud", "n", "g"))


def _n_1_failing(d, gn: pl.DataFrame) -> "dict[str, dict[str, pl.DataFrame]]":
    """Failing processes of the n-1 groups, per direction ``ud``.

    Each entry has ``n1_rug`` (r, ud, g) of the n-1 groups of that direction, ``lfr_g`` (p, r, ud, n, g) the
    large-failure relationships at the group's nodes, ``fail_arcs`` the failing v_flow arcs (p_fail and the arc's
    nodes, the group-side node as n), ``fail_back_arcs`` the failing v_flow_back arcs (the group-side node, the
    arc's source for up and its sink for down, as n; None without any) and ``fail_g`` (r, ud, g, p_fail) sorted.
    Empty when the n-1 method has no group, ratio or failing arc.  The caller checks that v_flow exists.
    """
    method_n1 = getattr(d, "reserve_upDown_group_method_n_1", None)
    lfr_set = getattr(d, "process_reserve_upDown_node_large_failure_ratio", None)
    lfr_param = getattr(d, "p_process_reserve_upDown_node_large_failure_ratio_value", None)
    if (method_n1 is None or method_n1.height == 0 or lfr_set is None or lfr_set.height == 0
            or lfr_param is None or d.process_source_sink is None):
        return {}
    out: dict[str, dict[str, pl.DataFrame]] = {}
    for ud_filter, _name, side, _other in _N_1_DIRECTIONS:
        n1_rug = method_n1.filter(pl.col("ud") == ud_filter).select("r", "ud", "g").unique()
        if n1_rug.height == 0:
            continue
        lfr_g = (lfr_set.join(gn, on="n", how="inner")
                 .join(n1_rug, on=["r", "ud", "g"], how="inner")
                 .select("p", "r", "ud", "n", "g"))
        if lfr_g.height == 0:
            continue
        # n is the sink of the failing flow (up) or its source (down).
        pss_idx = (d.process_source_sink.pipe(rename_to_axis, {side: "n"})
                   .select("p", *(("source", "n") if side == "sink" else ("n", "sink"))))
        fail_arcs = pss_idx.join(lfr_g, on=["p", "n"], how="inner").pipe(rename_to_axis, {"p": "p_fail"})
        # The reverse flow v_flow_back of a one-variable / DC arc arrives at the arc's source (up) and leaves its
        # sink (down): n is the other side of the arc, keyed on the large-failure relationship's (p, n).
        other = "source" if side == "sink" else "sink"
        fail_back_arcs = None
        back = _back_arcs(d)
        if back is not None:
            back_idx = (back.pipe(rename_to_axis, {other: "n"})
                        .select("p", *(("n", "sink") if other == "source" else ("source", "n"))))
            fail_back_arcs = (back_idx.join(lfr_g, on=["p", "n"], how="inner")
                              .pipe(rename_to_axis, {"p": "p_fail"}))
            if fail_back_arcs.height == 0:
                fail_back_arcs = None
        if fail_arcs.height == 0 and fail_back_arcs is None:
            continue
        failing = [fail_arcs.select("r", "ud", "g", "p_fail")]
        if fail_back_arcs is not None:
            failing.append(fail_back_arcs.select("r", "ud", "g", "p_fail"))
        fail_g = (pl.concat(failing)
                  .unique(maintain_order=True)
                  .sort("r", "ud", "g", "p_fail"))
        out[ud_filter] = {"n1_rug": n1_rug, "lfr_g": lfr_g, "fail_arcs": fail_arcs, "fail_back_arcs": fail_back_arcs,
                          "fail_g": fail_g}
    return out


def _reserve_total_rug(d, pruna_g: pl.DataFrame,
                       n1: "dict[str, dict[str, pl.DataFrame]]") -> "pl.DataFrame | None":
    """rug_total (r, ud, g), sorted: the groups with an n-1 row and at least one provider.

    The domain of v_reserve_total and reserve_total_upper.  None without reliability data (the reserve
    term is then left out of every reserveBalance row) or when no group qualifies.
    """
    if d.p_process_reserve_upDown_node_reliability is None or not n1 or pruna_g.height == 0:
        return None
    rug_total = (pl.concat([part["fail_g"].select("r", "ud", "g") for part in n1.values()])
                 .join(pruna_g.select("r", "ud", "g"), on=["r", "ud", "g"], how="semi")
                 .unique(maintain_order=True)
                 .sort("r", "ud", "g"))
    return rug_total if rug_total.height > 0 else None


def _rug_set(frame: pl.DataFrame) -> set[tuple[str, str, str]]:
    """(r, ud, g) of ``frame`` as label tuples."""
    return set(frame.select(pl.col(c).cast(pl.Utf8) for c in ("r", "ud", "g")).iter_rows())


# ---------------------------------------------------------------------------
# Variables


def add_variables(b: "_Build") -> None:
    """Declare v_reserve, vq_reserve and v_reserve_total."""
    if not b.has_reserve:
        return
    m, d = b.m, b.d
    _check(d, ("reserve_upDown_group", "prundt", "dt", "process_reserve_upDown_node_active", "group_node"))

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

    # ─── VAR. ─── v_reserve_total ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve_total[r,ud,g,d,t] ≥ 0   ∀ (r,ud,g) ∈ rug_total, (d,t) ∈ dt
    #   rug_total = (r,ud,g) with an n-1 row (in fail_g) and at least one provider in pruna_g
    # Reserve held for group g by all its providers, MW: at most reserve_sum
    # (reserve_total_upper).  Lets each n-1 row subtract the failing
    # process's own reserve instead of repeating every other provider, and
    # replaces reserve_sum in the timeseries / dynamic rows of the same
    # group.  No cost.
    if b.v_flow is not None:
        gn = _group_nodes(d)
        rug_total = _reserve_total_rug(d, _providers_by_group(d, gn), _n_1_failing(d, gn))
        if rug_total is not None:
            b.v_reserve_total = m.add_var("v_reserve_total", ("r", "ud", "g", "d", "t"),
                                          rug_total.join(d.dt, how="cross"), lower=0.0)


# ---------------------------------------------------------------------------
# Constraints


def _flow_node_view(v_flow: Var, side: str, extra: "dict[str, str] | None" = None) -> Var:
    """``v_flow`` with its ``side`` axis (``"source"`` or ``"sink"``) relabelled to the node axis ``n``.

    The dynamic and n-1 requirements sum v_flow over the flows whose ``side`` node is the reserve node n.
    Relabelling ties that axis to n (and to the ratio Param's n), so the enclosing Sum collapses it;
    a separately renamed join column would leave ``source`` / ``sink`` open.  ``extra`` relabels further
    axes (the n-1 rows: ``{"p": "p_fail"}``).  The view keeps the LP columns and the name ``v_flow``
    (autoscale keys on it).
    """
    mapping = {side: "n", **(extra or {})}
    return Var(name=v_flow.name,
               dims=tuple(mapping.get(c, c) for c in v_flow.dims),
               frame=v_flow.frame.pipe(rename_to_axis, mapping))


def _drawn_terms(b: "_Build", flow_view: Var, arcs: pl.DataFrame, us, ratio, *, p_col: str,
                 over: tuple[str, ...]) -> dict:
    """Σ over ``arcs`` of the flow a process draws from the arc's source node n, × ``ratio``, MW.

    ``arcs`` holds ``(p_col, n, sink, ...)`` with n the source node; ``flow_view`` is v_flow with its source
    relabelled n (and p relabelled ``p_col``).  As in nodeBalance_eq, an arc in process_source_sink_eff
    draws ``v_flow · unitsize · slope`` plus, for a min_load_efficiency unit with online variables,
    ``v_online · section · unitsize``; any other arc draws ``v_flow · unitsize``.  Keys ``noEff``, ``eff``,
    ``section_linear`` / ``section_integer``.  Without process_source_sink_eff or p_slope (hand-built data)
    every arc draws ``v_flow · unitsize``.
    """
    d = b.d

    def _flows(frame: pl.DataFrame, slope=None):
        expr = flow_view * us
        if slope is not None:
            expr = expr * slope
        return Sum(Where(expr * ratio, frame), over=over)

    eff = getattr(d, "process_source_sink_eff", None)
    slope = getattr(d, "p_slope", None)
    if eff is None or eff.height == 0 or slope is None:
        return {"noEff": _flows(arcs)}
    eff_keys = (eff.pipe(rename_to_axis, {"p": p_col, "source": "n"}).select(p_col, "n", "sink").unique())
    arcs_eff = arcs.join(eff_keys, on=[p_col, "n", "sink"], how="semi")
    if arcs_eff.height == 0:
        return {"noEff": _flows(arcs)}
    arcs_no = arcs.join(eff_keys, on=[p_col, "n", "sink"], how="anti")

    def _on_p(param: Param) -> Param:
        if p_col == "p":
            return param
        return Param((p_col, *param.dims[1:]), param.frame.pipe(rename_to_axis, {"p": p_col}))

    terms: dict = {}
    if arcs_no.height > 0:
        terms["noEff"] = _flows(arcs_no)
    terms["eff"] = _flows(arcs_eff, _on_p(slope))
    section = getattr(d, "p_section", None)
    pml = getattr(d, "process_min_load_eff", None)
    if b.has_minload_eff and section is not None and pml is not None:
        pml_p = pml.pipe(rename_to_axis, {"p": p_col}) if p_col != "p" else pml
        for kind, v_online in (("linear", b.v_online_linear), ("integer", b.v_online_integer)):
            if v_online is None:
                continue
            v_on = v_online if p_col == "p" else _var_view(v_online, "", {"p": p_col}, to_axis=True)
            terms[f"section_{kind}"] = Sum(Where(Where(v_on, pml_p) * _on_p(section) * us * ratio, arcs_eff),
                                           over=over)
    return terms


def add_constraints(b: "_Build") -> None:
    """Reserve requirements of the node groups, per-process reserve limits and the storage energy rows."""
    if not b.has_reserve:
        return
    m, d = b.m, b.d
    _check(d, ("reserve_upDown_group", "prundt", "dt", "process_reserve_upDown_node_active", "group_node"))
    v_reserve, vq_reserve, v_flow = b.v_reserve, b.vq_reserve, b.v_flow
    p_unitsize = d.p_unitsize
    gn = _group_nodes(d)                                   # (g, n), n cast to the node axis

    # Left-hand side shared by every reserveBalance row, per (r, ud, g, d, t):
    #   reserve_sum = Σ_{(p,n) : (p,r,ud,n) ∈ process_reserve_upDown_node_active, (g,n) ∈ group_node}
    #                   v_reserve[p,r,ud,n,d,t] · unitsize[p] · reliability[p,r,ud,n]
    #   vq_term     = vq_reserve[r,ud,g,d,t] · scale[r,ud,g,d,t]
    pruna_g = _providers_by_group(d, gn)
    lhs_reserve_core = None
    if d.p_process_reserve_upDown_node_reliability is not None and pruna_g.height > 0:
        lhs_reserve_core = Sum(Where(v_reserve * p_unitsize * d.p_process_reserve_upDown_node_reliability, pruna_g),
                               over=("p", "n"))
    scale = reserve_shortfall_scale(d)
    lhs_vq = vq_reserve * scale if scale is not None else None

    # Groups with n-1 rows (rug_total) carry their total reserve in v_reserve_total (see the n-1 rows below).
    n1 = _n_1_failing(d, gn) if v_flow is not None else {}
    rug_total = _reserve_total_rug(d, pruna_g, n1)
    total_rugs = _rug_set(rug_total) if rug_total is not None else set()
    declared = (_rug_set(b.v_reserve_total.frame) if b.v_reserve_total is not None else set())
    if total_rugs != declared:
        raise ValueError(
            f"_reserve: v_reserve_total is declared for groups {sorted(declared)} but the n-1 rows need it "
            f"for {sorted(total_rugs)}; add_variables and add_constraints must use the same domain.")

    def _reserve_terms(rug_sel: pl.DataFrame) -> dict:
        """The reserve term of the timeseries / dynamic rows of ``rug_sel``: v_reserve_total for the groups in
        rug_total, reserve_sum for the others."""
        if lhs_reserve_core is None:
            return {}
        with_total = (rug_sel.join(rug_total, on=["r", "ud", "g"], how="semi")
                      if rug_total is not None else None)
        if with_total is None or with_total.height == 0:
            return {"reserve_sum": Where(lhs_reserve_core, rug_sel)}
        terms = {"reserve_total": Where(b.v_reserve_total, with_total)}
        without = rug_sel.join(rug_total, on=["r", "ud", "g"], how="anti")
        if without.height > 0:
            terms["reserve_sum"] = Where(lhs_reserve_core, without)
        return terms

    # ─── EQ. ─── reserveBalance_timeseries_eq ────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # reserve_sum[r,ud,g,d,t] + vq_term[r,ud,g,d,t]  ≥ reservation[r,ud,g,d,t]
    #   ∀ (r,ud,g) ∈ reserve_upDown_group_method_timeseries, (d,t) ∈ dt
    #   reserve_sum is replaced by v_reserve_total for (r,ud,g) ∈ rug_total
    # Reserve held for a group covers its authored reservation timeseries,
    # or the shortfall is penalised.  RHS 0 when no reservation is authored.
    # Reserve of every provider counts at full MW (no efficiency slope).
    # A group that also has n-1 rows uses its total v_reserve_total: it is
    # at most reserve_sum, enters only ≥ rows and has no cost, so the rows
    # are equivalent (same optimal value for every right-hand side).
    method_ts = d.reserve_upDown_group_method_timeseries
    if method_ts is not None and method_ts.height > 0:
        ts_rug = method_ts.select("r", "ud", "g").unique()
        rhs = (d.pdtReserve_upDown_group_reservation
               if d.pdtReserve_upDown_group_reservation is not None else 0.0)
        lhs_terms = {}
        if lhs_vq is not None:
            lhs_terms["vq_term"] = lhs_vq
        lhs_terms |= _reserve_terms(ts_rug)
        m.add_cstr("reserveBalance_timeseries_eq", over=ts_rug.join(d.dt, how="cross"), sense=">=",
                   lhs_terms=lhs_terms, rhs_terms={"demand": rhs})

    # ─── EQ. ─── reserveBalance_dynamic_eq ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # reserve_sum[r,ud,g,d,t] + vq_term[r,ud,g,d,t]
    #   ≥ Σ_{(p,s,n) ∈ process_source_sink, (p,r,ud,n) ∈ irr_g} v_flow[p,s,n,d,t] · unitsize[p] · ratio[p,r,ud,n]
    #   + Σ_{(p,n,k) ∈ eff_out, (p,r,ud,n) ∈ irr_g}
    #       (v_flow[p,n,k,d,t] · unitsize[p] · slope[p,d,t] + v_online[p,d,t] · section[p,d,t] · unitsize[p])
    #       · ratio[p,r,ud,n]
    #   + Σ_{(p,n,k) ∈ process_source_sink \ eff_out, (p,r,ud,n) ∈ irr_g} v_flow[p,n,k,d,t] · unitsize[p]
    #       · ratio[p,r,ud,n]
    #   + Σ_{(p,n,k) ∈ twoway1var_arcs, (p,r,ud,n) ∈ irr_g} v_flow_back[p,n,k,d,t] · unitsize[p] · ratio[p,r,ud,n]
    #   + Σ_{(p,s,n) ∈ twoway1var_arcs, (p,r,ud,n) ∈ irr_g} v_flow_back[p,s,n,d,t] · unitsize[p] · ratio[p,r,ud,n]
    #   + group_demand[r,ud,g,d,t]
    #   ∀ (r,ud,g) ∈ reserve_upDown_group_method_dynamic, (d,t) ∈ dt
    #   group_demand = ratio_g[r,ud,g] · Σ_{n : (g,n) ∈ group_node} max(−inflow[n,d,t], 0) / Δt[d,t]
    #   ratio_g = reserve__upDown__group.increase_reserve_ratio;  inflow in MWh per step without the
    #             energy margin (p_negative_inflow, the real demand), so group_demand is MW in the step
    #   irr_g   = process_reserve_upDown_node_increase_reserve_ratio with n ∈ group g
    #   ratio   = increase_reserve_ratio;   eff_out = process_source_sink_eff
    #   section term only for min_load_efficiency units with online variables (v_online_linear / _integer)
    #   reserve_sum is replaced by v_reserve_total for (r,ud,g) ∈ rug_total
    # The requirement grows with the flows into and out of the group's nodes
    # (e.g. variable generation and consumption).  A flow out of a node counts
    # as drawn from it, as in nodeBalance_eq: on an efficiency arc the input is
    # v_flow · slope plus the min-load section (_drawn_terms).  The reverse flow
    # of a one-variable (and DC) arc arrives at the arc's source and leaves its
    # sink, as in nodeBalance_eq (lossless, no slope).  The group
    # term adds the demand of the group's nodes (negative inflow only; a
    # node with positive inflow does not count) times the group's ratio
    # (group_demand_requirement).  RHS 0 when nothing drives it.  A group that also has n-1 rows uses
    # v_reserve_total, as in the timeseries rows.
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
                drawn = _drawn_terms(b, _flow_node_view(v_flow, "source"),
                                     pss_as_source.join(irr_g, on=["p", "n"], how="inner"), p_unitsize, irr_param,
                                     p_col="p", over=("p", "sink", "n"))
                rhs_terms["dynamic_flow"] = into_n + drawn.pop("noEff") if "noEff" in drawn else into_n
                rhs_terms |= {f"dynamic_{k}": v for k, v in drawn.items()}
                if b.v_flow_back is not None and _has_rows(b.twoway1var_arcs):
                    back = b.twoway1var_arcs
                    # Reverse flow arrives at the arc's source (into n) and leaves its sink (drawn from n).
                    back_in = back.pipe(rename_to_axis, {"source": "n"}).select("p", "n", "sink").join(
                        irr_g, on=["p", "n"], how="inner")
                    back_out = back.pipe(rename_to_axis, {"sink": "n"}).select("p", "source", "n").join(
                        irr_g, on=["p", "n"], how="inner")
                    if back_in.height > 0:
                        rhs_terms["dynamic_back_in"] = Sum(Where(
                            _flow_node_view(b.v_flow_back, "source") * p_unitsize * irr_param, back_in),
                            over=("p", "n", "sink"))
                    if back_out.height > 0:
                        rhs_terms["dynamic_back_out"] = Sum(Where(
                            _flow_node_view(b.v_flow_back, "sink") * p_unitsize * irr_param, back_out),
                            over=("p", "source", "n"))
        demand = group_demand_requirement(d)
        if demand is not None:
            rhs_terms["group_demand"] = demand
        if not rhs_terms:
            rhs_terms = {"zero": 0.0}
        lhs_terms = {}
        if lhs_vq is not None:
            lhs_terms["vq_term"] = Where(lhs_vq, dyn_rug)
        lhs_terms |= _reserve_terms(dyn_rug)
        m.add_cstr("reserveBalance_dynamic_eq", over=dyn_rug.join(d.dt, how="cross"), sense=">=",
                   lhs_terms=lhs_terms, rhs_terms=rhs_terms)

    # ─── EQ. ─── reserveBalance_up_n_1_eq / reserveBalance_down_n_1_eq ───────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve_total[r,ud,g,d,t]
    #   − Σ_{n : (f,r,ud,n) ∈ pruna_g} v_reserve[f,r,ud,n,d,t] · unitsize[f] · reliability[f,r,ud,n]
    #   + vq_term[r,ud,g,d,t]
    #   ≥ up:   Σ_{(f,s,n) ∈ process_source_sink, (f,r,ud,n) ∈ lfr_g} v_flow[f,s,n,d,t] · unitsize[f] · ratio[f,r,ud,n]
    #     down: Σ_{(f,n,k) ∈ eff_out, (f,r,ud,n) ∈ lfr_g} (v_flow[f,n,k,d,t] · unitsize[f] · slope[f,d,t]
    #                                                    + v_online[f,d,t] · section[f,d,t] · unitsize[f])
    #                                                   · ratio[f,r,ud,n]
    #           + Σ_{(f,n,k) ∈ process_source_sink \ eff_out, (f,r,ud,n) ∈ lfr_g} v_flow[f,n,k,d,t] · unitsize[f]
    #               · ratio[f,r,ud,n]
    #   + up:   Σ_{(f,n,k) ∈ twoway1var_arcs, (f,r,ud,n) ∈ lfr_g} v_flow_back[f,n,k,d,t] · unitsize[f] · ratio[f,r,ud,n]
    #     down: Σ_{(f,s,n) ∈ twoway1var_arcs, (f,r,ud,n) ∈ lfr_g} v_flow_back[f,s,n,d,t] · unitsize[f] · ratio[f,r,ud,n]
    #   ∀ (r,ud,g,f) ∈ fail_g, (d,t) ∈ dt
    #   pruna_g = the providers of the reserve_sum above, per group
    #   lfr_g   = process_reserve_upDown_node_large_failure_ratio with n ∈ group g,
    #             (r,ud,g) ∈ reserve_upDown_group_method_n_1 with ud = up / down
    #   fail_g  = (r,ud,g,f): f in lfr_g with a flow (v_flow or v_flow_back) into (up) / out of (down) a node of g
    #   ratio   = large_failure_ratio;   f is the row dim p_fail;   eff_out, section as in the dynamic row
    # Reserve covers the loss of each large process f separately (the largest
    # failure binds): upward reserve the loss of its output into the group,
    # downward reserve the loss of its consumption from the group.  The
    # failing process's own reserve, at every node of the group, does not
    # count: it is subtracted from the group total.  All rows of a group
    # share vq_term, so the priced shortfall is the largest deficit.  The
    # lost consumption is the flow the process draws from the node, as in
    # nodeBalance_eq (output · slope plus the min-load section on an
    # efficiency arc); a lost output counts at unitsize.  A one-variable (or DC)
    # connection also fails while flowing backwards: its v_flow_back arrives at
    # the arc's source and leaves its sink (lossless).  No row for a group
    # without a failing process; a group without providers (or without
    # reliability data) has no v_reserve_total term.  Nonzeros grow with
    # providers + failing processes per group.
    if n1:
        lfr_param = d.p_process_reserve_upDown_node_large_failure_ratio_value
        # The failing process f is the dim p_fail: unitsize and ratio re-keyed on it.
        us_fail = Param(("p_fail",), p_unitsize.frame.pipe(rename_to_axis, {"p": "p_fail"}))
        ratio_fail = Param(("p_fail", "r", "ud", "n"), lfr_param.frame.pipe(rename_to_axis, {"p": "p_fail"}))
        for ud_filter, name, side, other in _N_1_DIRECTIONS:
            if ud_filter not in n1:
                continue
            n1_rug, fail_arcs, fail_g = (n1[ud_filter][k] for k in ("n1_rug", "fail_arcs", "fail_g"))
            if ud_filter == "down" and fail_arcs.height > 0:
                failure_terms = _drawn_terms(b, _flow_node_view(v_flow, side, {"p": "p_fail"}), fail_arcs, us_fail,
                                             ratio_fail, p_col="p_fail", over=(other, "n"))
                failure_terms = {("n_1_failure" if k == "noEff" else f"n_1_failure_{k}"): v
                                 for k, v in failure_terms.items()}
            elif fail_arcs.height > 0:
                failure_terms = {"n_1_failure": Sum(Where(_flow_node_view(v_flow, side, {"p": "p_fail"}) * us_fail
                                                          * ratio_fail, fail_arcs), over=(other, "n"))}
            else:
                failure_terms = {}
            fail_back = n1[ud_filter]["fail_back_arcs"]
            if fail_back is not None and b.v_flow_back is not None:
                failure_terms["n_1_failure_back"] = Sum(
                    Where(_flow_node_view(b.v_flow_back, other, {"p": "p_fail"}) * us_fail * ratio_fail, fail_back),
                    over=(side, "n"))
            lhs_terms = {}
            if lhs_vq is not None:
                lhs_terms["vq_term"] = Where(lhs_vq, n1_rug)
            if rug_total is not None:
                total_ud = rug_total.join(n1_rug, on=["r", "ud", "g"], how="semi")
                if total_ud.height > 0:
                    lhs_terms["total"] = Where(b.v_reserve_total, total_ud)
                    # The failing process's own reserve (p and p_fail compared on their labels).
                    pruna_own = (pruna_g.join(fail_g, on=["r", "ud", "g"], how="inner")
                                 .filter(pl.col("p").cast(pl.Utf8) == pl.col("p_fail").cast(pl.Utf8)))
                    if pruna_own.height > 0:
                        lhs_terms["reserve_own_neg"] = -Sum(
                            Where(v_reserve * p_unitsize * d.p_process_reserve_upDown_node_reliability, pruna_own),
                            over=("p", "n"))
            m.add_cstr(name, over=fail_g.join(d.dt, how="cross"), sense=">=",
                       lhs_terms=lhs_terms, rhs_terms=failure_terms)

    # ─── EQ. ─── reserve_total_upper ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve_total[r,ud,g,d,t] − reserve_sum[r,ud,g,d,t]  ≤  0   ∀ (r,ud,g) ∈ rug_total, (d,t) ∈ dt
    # v_reserve_total enters only ≥ requirement rows, with a positive sign,
    # and has no cost, so an upper bound is enough: the n-1 rows above are
    # exactly the per-provider form (one row per failing process, its own
    # reserve not counted) with v_reserve_total = reserve_sum, and the
    # timeseries / dynamic rows of the group are those with reserve_sum.
    # This needs reserve_sum ≥ 0, which holds because providers have
    # reliability > 0 (process_reserve_upDown_node_active) and unitsize > 0;
    # otherwise v_reserve_total ≥ 0 would add a constraint.  When
    # v_reserve_total > 0, the magnitude of this row's dual is the sum of the
    # duals of the group-step's rows that use v_reserve_total (opposite sign
    # convention: ≤ row against ≥ rows).
    if rug_total is not None:
        m.add_cstr("reserve_total_upper", over=rug_total.join(d.dt, how="cross"), sense="<=",
                   lhs_terms={"total": b.v_reserve_total, "reserve_neg": -Where(lhs_reserve_core, rug_total)},
                   rhs_terms={"zero": 0.0})

    # ─── EQ. ─── reserve_process_upward / reserve_process_downward ───────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_reserve[p,r,ud,n,d,t] − max_share[p,r,ud,n] · c[p,n,d,t] · Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + max_share[p,r,ud,n] · c[p,n,d,t] · Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
    #   ≤ max_share[p,r,ud,n] · c[p,n,d,t] · existing_count[p,d]
    #   ∀ (p,r,ud,n,d,t) ∈ prundt with ud = up / down
    #   c = 1 at a node the process feeds, and for connections;
    #   at an input node n of a unit (n the source of one of its arcs and not a sink):
    #     c = input_cap[p,n,p,d,t]           indirect unit (input_share_max × full-load input per unit)
    #     c = slope[p,d,t] + section[p,d,t]  direct unit's input→output arc (full-load input = capacity /
    #                                        efficiency; section only for min_load_efficiency units with
    #                                        online variables)
    #     c = 1                              sink-less unit (its capacity is the input), and
    #                                        conversion_flow_coeff = 0 inputs
    # A process holds at most max_share of the built capacity of its flow at
    # that node as reserve (units of unitsize; built = existing + invested −
    # retired, also for units with online variables).  At an input node the
    # capacity is the input's (_input_capacity_factor).  max_share not
    # authored → no reserve (RHS 0).
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
            c = _input_capacity_factor(d, idx)
            share = max_share if c is None else max_share * c
            if c is not None:
                rhs_param = share * existing
            lhs: dict = {"reserve": v_reserve}
            p_in_idx = idx.select("p").unique()
            if b.v_invest_p is not None and edd_inv_set is not None and edd_inv_set.height > 0:
                edd_p = edd_inv_set.pipe(rename_to_axis, {"e": "p"}).join(p_in_idx, on="p", how="inner")
                if edd_p.height > 0:
                    v_inv_at = _var_view(b.v_invest_p, f"__at_{name}", {"d": "d_invest"}, to_axis=True)
                    lhs["invest_neg"] = -(Sum(Where(v_inv_at, edd_p), over=("d_invest",)) * share)
            if b.v_divest_p is not None and edd_div_act is not None and edd_div_act.height > 0:
                edd_p_div = edd_div_act.join(p_in_idx, on="p", how="inner")
                if edd_p_div.height > 0:
                    v_div_at = _var_view(b.v_divest_p, f"__at_{name}", {"d": "d_divest"}, to_axis=True)
                    lhs["divest"] = Sum(Where(v_div_at, edd_p_div), over=("d_divest",)) * share
            m.add_cstr(name, over=idx, sense="<=", lhs_terms=lhs, rhs_terms={"max": rhs_param})

    _add_connection_reserve_constraints(b)
    _add_storage_energy_constraints(b)


# ---------------------------------------------------------------------------
# Connection reserve: net-position rows


def _connection_channels(b: "_Build", pn: pl.DataFrame) -> dict:
    """The flow channels of the connections at the reserve nodes ``pn`` (p, n).

    Returns ``{"in": ..., "out": ...}``; each is a frame ``(p, source, sink, n, kind, eff)`` in the dtypes of
    process_source_sink: ``kind`` "flow" (a v_flow arc) or "back" (the reverse flow v_flow_back of a one-variable /
    DC arc), ``eff`` True for a v_flow arc in process_source_sink_eff (its flow is measured at the sink and draws
    slope × flow at the source).  In-channels deliver into n (v_flow arcs with sink n, v_flow_back of arcs with
    source n), out-channels draw from n (v_flow arcs with source n, v_flow_back of arcs with sink n).  Raises when a
    (p, n) has more than one channel of a kind (the rows hold one channel's capacity).
    """
    d = b.d
    keys = ["p", "source", "sink"]
    pn_u = pn.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("n").cast(pl.Utf8).alias("_n")).unique()
    eff = getattr(d, "process_source_sink_eff", None)
    eff_u = (_utf8_keys(eff.lazy().select(keys), keys).select("_p", "_source", "_sink").unique()
             .with_columns(pl.lit(True).alias("eff")).collect()
             if _has_rows(eff) else None)

    def _side(arcs: "pl.DataFrame | None", node_col: str, kind: str) -> "pl.DataFrame | None":
        if not _has_rows(arcs):
            return None
        a = (_utf8_keys(arcs.lazy().select(keys), keys)
             .with_columns(pl.col(f"_{node_col}").alias("_n"))
             .join(pn_u.lazy(), on=["_p", "_n"], how="semi")
             .with_columns(pl.lit(kind).alias("kind")))
        if kind == "flow" and eff_u is not None:
            a = (a.join(eff_u.lazy(), on=["_p", "_source", "_sink"], how="left")
                 .with_columns(pl.col("eff").fill_null(False)))
        else:
            a = a.with_columns(pl.lit(False).alias("eff"))
        out = a.with_columns(pl.col(node_col).alias("n")).select(*keys, "n", "_p", "_source", "_sink", "_n", "kind",
                                                                "eff").collect()
        return out if out.height > 0 else None

    back = b.twoway1var_arcs if (b.v_flow_back is not None and _has_rows(b.twoway1var_arcs)) else None
    out: dict = {}
    for name, flow_side, back_side in (("in", "sink", "source"), ("out", "source", "sink")):
        parts = [x for x in (_side(d.process_source_sink, flow_side, "flow"), _side(back, back_side, "back"))
                 if x is not None]
        if not parts:
            out[name] = None
            continue
        frame = pl.concat(parts, how="vertical_relaxed")
        dup = frame.group_by("_p", "_n").len().filter(pl.col("len") > 1)
        if dup.height > 0:
            raise ValueError(f"_reserve: a connection has more than one {name}-channel at a reserve node (the "
                             f"connection reserve rows hold one channel's capacity): {dup.to_dicts()}")
        out[name] = frame
    return out


def _pndt_param(dom_u: pl.DataFrame, values: "pl.LazyFrame | None", fill: float) -> Param:
    """Param (p, n, d, t) dense over ``dom_u`` (typed p, n, d, t plus Utf8 _p, _n, _d, _t), ``values`` (Utf8 _p,
    _n, _d, _t, value) where present, ``fill`` elsewhere."""
    lf = dom_u.lazy()
    if values is not None:
        lf = lf.join(values, on=["_p", "_n", "_d", "_t"], how="left")
    else:
        lf = lf.with_columns(pl.lit(None, pl.Float64).alias("value"))
    frame = lf.with_columns(pl.col("value").cast(pl.Float64).fill_null(fill)).select("p", "n", "d", "t", "value")
    return Param(("p", "n", "d", "t"), frame.collect())


def _channel_values(chan: pl.DataFrame, arc_frame: "pl.LazyFrame", *, by_dt: bool = True) -> pl.LazyFrame:
    """Values of an arc-keyed frame (Utf8 _p, _source, _sink[, _d, _t], value) at the channels ``chan``, keyed on
    Utf8 (_p, _n[, _d, _t])."""
    on = ["_p", "_source", "_sink"]
    cols = ["_p", "_n", *(("_d", "_t") if by_dt else ()), "value"]
    return chan.lazy().select("_p", "_source", "_sink", "_n").join(arc_frame, on=on, how="inner").select(cols)


def _arc_dt_u(frame: "pl.DataFrame | pl.LazyFrame", dt: pl.DataFrame) -> pl.LazyFrame:
    """A (p, source, sink[, d], t...) Param frame promoted over dt, with Utf8 keys _p, _source, _sink, _d, _t."""
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return (_utf8_keys(lf, ["p", "source", "sink", "d", "t"])
            .select("_p", "_source", "_sink", "_d", "_t", pl.col("value").cast(pl.Float64)))


def _add_connection_reserve_constraints(b: "_Build") -> None:
    """connection_reserve_{up,down}[_online_{linear,integer}|_exact] (net-position rule, see the EQ. header)."""
    d = b.d
    conns = _connection_procs(d)
    if conns is None or conns.height == 0 or b.v_flow is None or d.prundt.height == 0:
        return
    m = b.m
    from flextool.engine.model import _exact, _online

    keys = ["p", "source", "sink"]
    dt_u = d.dt.select("d", "t").with_columns(pl.col("d").cast(pl.Utf8).alias("_d"),
                                              pl.col("t").cast(pl.Utf8).alias("_t"))
    rhs_u = _arc_dt_u(b.flow_upper_rhs.frame, d.dt)
    mult_u = _arc_dt_u(b.flow_cap_mult, d.dt) if b.flow_cap_mult is not None else None
    avail_u = (_utf8_keys(promote_param_to_dt(d.p_process_availability, d.dt).select("p", "d", "t", "value"),
                          ["p", "d", "t"]).select("_p", "_d", "_t", pl.col("value").alias("__av"))
               if d.p_process_availability is not None else None)
    coef_u = (_utf8_keys(d.p_arc_max_cap_coef.frame.lazy().select(*keys, "value"), keys)
              .select("_p", "_source", "_sink", pl.col("value").alias("__coef"))
              if d.p_arc_max_cap_coef is not None else None)
    slope_u = (_utf8_keys(promote_param_to_dt(d.p_slope, d.dt).select("p", "d", "t", "value"), ["p", "d", "t"])
               .select("_p", "_d", "_t", pl.col("value").alias("__slope"))
               if d.p_slope is not None else None)
    invest_procs = ({str(x) for x in b.invest_procs.get_column("p").to_list()}
                    if b.invest_procs is not None else set())
    exact = _exact.exact_arcs(d) if b.v_exact_forward is not None else None

    def _online_factor_u(chan: pl.DataFrame) -> pl.LazyFrame:
        """κ · availability of the channel arc over dt: (_p, _n, _d, _t, value)."""
        lf = chan.lazy().select("_p", "_source", "_sink", "_n").join(dt_u.lazy().select("_d", "_t"), how="cross")
        if coef_u is not None:
            lf = lf.join(coef_u, on=["_p", "_source", "_sink"], how="left")
        else:
            lf = lf.with_columns(pl.lit(None, pl.Float64).alias("__coef"))
        if avail_u is not None:
            lf = lf.join(avail_u, on=["_p", "_d", "_t"], how="left")
        else:
            lf = lf.with_columns(pl.lit(None, pl.Float64).alias("__av"))
        return lf.select("_p", "_n", "_d", "_t",
                         (pl.col("__coef").fill_null(1.0) * pl.col("__av").fill_null(1.0)).alias("value"))

    # ─── EQ. ─── connection_reserve_up / connection_reserve_down ─────────────
    # ─────────────────────────────────────────────────────────────────────────
    # x_in[p,n,d,t] − s_out[p,n,d,t] · x_out[p,n,d,t] + Σ_r v_reserve[p,r,up,n,d,t]
    #   − M_in · Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
    #   + M_in · Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]  ≤  C_in[p,n,d,t]   (up)
    # x_out[p,n,d,t] − x_in[p,n,d,t] / s_out[p,n,d,t] + Σ_r v_reserve[p,r,down,n,d,t] / s_out[p,n,d,t]
    #   − M_out · Σ v_invest_p[p,d′] + M_out · Σ v_divest_p[p,d′]  ≤  C_out[p,n,d,t]                         (down)
    #   ∀ connection p (process_source_sink \ process_unit) with (p,r,ud,n,d,t) ∈ prundt, ud = up (down)
    #   x_in  = flow of p arriving at n:  v_flow[p,m,n,d,t], or v_flow_back[p,n,m,d,t] of a one-variable / DC arc
    #   x_out = flow of p leaving n (measured where it arrives):  v_flow[p,n,m,d,t], or v_flow_back[p,m,n,d,t]
    #   s_out = slope[p,d,t] when x_out is v_flow on an arc of process_source_sink_eff, else 1 (power drawn at n per
    #           unit of x_out)
    #   C, M  = the channel arc's maxFlow (maxFlow_back) RHS and invest multiplier (b.flow_upper_rhs,
    #           b.flow_cap_mult: capacity_max_coeff · availability · existing / unitsize and coef · availability)
    #   a channel p does not have contributes nothing (x = 0, C = 0, M = 0, s_out = 1)
    # ─── EQ. ─── connection_reserve_{up,down}_online_{linear,integer} ────────
    #   same LHS without invest terms  ≤  κ · availability · v_online[p,d,t]   (κ, availability of the up row's
    #   in-channel arc / the down row's out-channel arc)   ∀ rows above with p online and that channel present
    # ─── EQ. ─── connection_reserve_{up,down}_exact ──────────────────────────
    #   same LHS − Ū · v_exact_forward[p,d,t]  ≤  0     when that channel is the exact connection's forward arc
    #   same LHS + Ū · v_exact_forward[p,d,t]  ≤  Ū     when it is the reverse arc
    #   ∀ rows above with p ∈ process_exact;  Ū = p_flow_upper · capacity_max_coeff · availability of the arc
    # A connection's reserve at a node is limited by its net position there: upward reserve by the free capacity
    # into n plus the flow out of n it can stop, downward reserve by the flow into n it can stop plus the free
    # capacity out of n (converted to power at n by the efficiency).  One row
    # per capacity bound (built, online, exact direction) of the channel whose capacity the reserve uses, so each
    # row holds one channel's capacity.  A one-way connection gets a floor at its sending node (upward reserve ≤
    # what it draws there) and its old capacity row at the receiving node; a one-variable connection's rows bound
    # the reserve by its signed flow (C ∓ f).  Reserve at a node the connection does not link is 0.  maxFlow /
    # maxFlow_online carry the upward reserve of units only (b.reserve_up_to_sink_units); the profile cap
    # profile_flow_upper_limit keeps the connection's in-arc term (conservative: it does not count stopping the
    # out-flow).
    for ud in ("up", "down"):
        sel = d.prundt.filter(pl.col("ud") == ud)
        sel = (sel.with_columns(pl.col("p").cast(pl.Utf8).alias("__p"))
               .join(conns.rename({"p": "__p"}), on="__p", how="semi").drop("__p"))
        if sel.height == 0:
            continue
        dom = sel.select("p", "n", "d", "t").unique(maintain_order=True)
        dom_u = dom.with_columns(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("n").cast(pl.Utf8).alias("_n"),
                                 pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"))
        ch = _connection_channels(b, dom.select("p", "n").unique())
        own = "in" if ud == "up" else "out"
        ch_own = ch[own]

        # s_out[p,n,d,t]: power drawn at n per unit of the out-channel (slope on an efficiency v_flow arc, else 1).
        s_vals = None
        if ch["out"] is not None and slope_u is not None:
            eff_out = ch["out"].filter(pl.col("eff"))
            if eff_out.height > 0:
                s_vals = (eff_out.lazy().select("_p", "_n").join(slope_u, on="_p", how="inner")
                          .select("_p", "_n", "_d", "_t", pl.col("__slope").alias("value")))
        s_out = _pndt_param(dom_u, s_vals, 1.0)
        inv_s = Param(("p", "n", "d", "t"), s_out.frame.with_columns(value=1.0 / pl.col("value")))

        def _channel_term(which: str, factor: "Param | None"):
            """Σ of the ``which`` ("in" / "out") channel's flow variable at (p, n, d, t), × factor."""
            chan = ch[which]
            if chan is None:
                return None
            terms = []
            for kind, var in (("flow", b.v_flow), ("back", b.v_flow_back)):
                part = chan.filter(pl.col("kind") == kind)
                if part.height == 0 or var is None:
                    continue
                # The node n of the channel: a v_flow arc delivers into its sink and draws from its source; the
                # reverse flow v_flow_back delivers into the arc's source and draws from its sink.
                side = {("in", "flow"): "sink", ("in", "back"): "source",
                        ("out", "flow"): "source", ("out", "back"): "sink"}[(which, kind)]
                frame = part.select(*keys).pipe(rename_to_axis, {side: "n"})
                view = _flow_node_view(var, side)
                expr = view if factor is None else view * factor
                terms.append(Sum(Where(expr, frame), over=("sink",) if side == "source" else ("source",)))
            if not terms:
                return None
            out = terms[0]
            for t in terms[1:]:
                out = out + t
            return out

        reserve = Sum(Where(b.v_reserve, sel.select("p", "r", "ud", "n").unique()), over=("r", "ud"))
        lhs: dict = {}
        if ud == "up":
            x_in = _channel_term("in", None)
            x_out = _channel_term("out", s_out)
            if x_in is not None:
                lhs["flow_in"] = x_in
            if x_out is not None:
                lhs["flow_out_neg"] = -x_out
            lhs["reserve"] = reserve
        else:
            x_out = _channel_term("out", None)
            x_in = _channel_term("in", inv_s)
            if x_out is not None:
                lhs["flow_out"] = x_out
            if x_in is not None:
                lhs["flow_in_neg"] = -x_in
            lhs["reserve"] = reserve * inv_s

        # Built capacity of the own channel: maxFlow's (maxFlow_back's) RHS and invest / divest terms.
        rhs_vals = _channel_values(ch_own, rhs_u) if ch_own is not None else None
        rhs = _pndt_param(dom_u, rhs_vals, 0.0)
        built = dict(lhs)
        if ch_own is not None and invest_procs:
            inv_chan = ch_own.filter(pl.col("_p").is_in(list(invest_procs)))
            if inv_chan.height > 0:
                mult_vals = (_channel_values(inv_chan, mult_u) if mult_u is not None else None)
                inv_dom = dom_u.join(inv_chan.select("_p", "_n").unique(), on=["_p", "_n"], how="semi")
                mult = _pndt_param(inv_dom, mult_vals, 1.0)
                if b.invest_in_dispatch_p is not None:
                    built["invest_neg"] = -(b.invest_in_dispatch_p * mult)
                if b.divest_in_dispatch_p is not None:
                    built["divest"] = b.divest_in_dispatch_p * mult
        m.add_cstr(f"connection_reserve_{ud}", over=dom, sense="<=", lhs_terms=built, rhs_terms={"cap": rhs})

        if ch_own is None:
            continue
        own_dom = dom_u.join(ch_own.select("_p", "_n").unique(), on=["_p", "_n"], how="semi")
        factor = _pndt_param(own_dom, _online_factor_u(ch_own), 1.0)
        # Online capacity of the own channel.
        for kind, _p_idx, online_set, v_online, _v_su, _v_sd in _online.uc_classes(b) if b.has_online else ():
            on_p = online_set.select(pl.col("p").cast(pl.Utf8).alias("_p")).unique()
            on_dom = own_dom.join(on_p, on="_p", how="semi")
            if on_dom.height == 0:
                continue
            m.add_cstr(f"connection_reserve_{ud}_online_{kind}", over=on_dom.select("p", "n", "d", "t"), sense="<=",
                       lhs_terms=dict(lhs), rhs_terms={"online": v_online * factor})
        # Direction switch of an exact connection: the own channel is its forward or its reverse arc.
        if exact is not None:
            ex_u = (_utf8_keys(exact.lazy(), keys).select("_p", "_source", "_sink", "forward").collect())
            ex_chan = ch_own.join(ex_u, on=["_p", "_source", "_sink"], how="inner")
            if ex_chan.height > 0:
                u_arc = _exact.exact_bound(d, ex_chan.select(*keys))
                u_vals = _channel_values(ex_chan, _arc_dt_u(u_arc.frame, d.dt))
                ex_dom = own_dom.join(ex_chan.select("_p", "_n").unique(), on=["_p", "_n"], how="semi")
                fwd = ex_chan.filter(pl.col("forward")).select("_p", "_n").unique().with_columns(
                    pl.lit(True).alias("__fwd"))
                # forward channel: LHS − Ū·y ≤ 0;  reverse channel: LHS + Ū·y ≤ Ū
                sign_u = (u_vals.join(fwd.lazy(), on=["_p", "_n"], how="left")
                          .with_columns(pl.col("__fwd").fill_null(False)))
                coef = _pndt_param(ex_dom, sign_u.select(
                    "_p", "_n", "_d", "_t",
                    pl.when(pl.col("__fwd")).then(-pl.col("value")).otherwise(pl.col("value")).alias("value")), 0.0)
                cap = _pndt_param(ex_dom, sign_u.select(
                    "_p", "_n", "_d", "_t",
                    pl.when(pl.col("__fwd")).then(0.0).otherwise(pl.col("value")).alias("value")), 0.0)
                ex_lhs = dict(lhs)
                ex_lhs["switch"] = b.v_exact_forward * coef
                m.add_cstr(f"connection_reserve_{ud}_exact", over=ex_dom.select("p", "n", "d", "t"), sense="<=",
                           lhs_terms=ex_lhs, rhs_terms={"cap": cap})


def _input_capacity_factor(d, idx: pl.DataFrame) -> "Param | None":
    """``c[p, n, d, t]`` of reserve_process_*: the capacity of the unit's flow at node n per unit of capacity,
    dense over the ``(p, n)`` of ``idx`` × dt with the dtypes of ``idx``.

    1 at a node the process feeds and for connections.  At an input node of a unit (n the source of one of
    its arcs and not the sink of any): ``p_indirect_input_cap[p, n, p]`` for an indirect unit, ``slope +
    section`` (section only for min_load_efficiency units with online variables) for a direct unit's
    input→output arc, 1 for a sink-less unit and conversion_flow_coeff = 0 inputs (they have no capacity).  A
    missing input cap of a capped indirect input raises (``_common._indirect_input_cap``).  A node that is both an
    input and an output of the same unit keeps the output rule.  A direct unit with several outputs has one
    arc per output from the same input; ``(p, n)`` is de-duplicated so each row gets one factor.  None when no
    row is at an input node (the rows keep the plain ``max_share`` form) or the data to tell is missing
    (hand-built FlexData).
    """
    from flextool.engine.model._unit_floors import all_unit_arcs

    arcs = all_unit_arcs(d)
    if arcs is None:
        return None
    keys = ["p", "source", "sink"]
    a = _u8(arcs.lazy(), keys)
    sinks = a.select("p", pl.col("sink").alias("n")).unique()
    inputs = (a.filter(pl.col("source") != pl.col("p"))
              .select("p", pl.col("source").alias("n"), "sink")
              .join(sinks, on=["p", "n"], how="anti"))
    rows = (idx.select("p", "n").unique(maintain_order=True)
            .with_columns(pl.col("p").cast(pl.Utf8).alias("__p"), pl.col("n").cast(pl.Utf8).alias("__n")))
    in_rows = rows.lazy().join(inputs.select(pl.col("p").alias("__p"), pl.col("n").alias("__n")).unique(),
                               on=["__p", "__n"], how="semi").collect()
    if in_rows.height == 0:
        return None
    dt_u = _u8(d.dt.lazy().select("d", "t"), ["d", "t"])
    # (p, n) → arc class: an input arc (sink == p: indirect / sink-less) or a direct arc; one per (p, n).
    kind = (inputs.with_columns((pl.col("sink") == pl.col("p")).alias("__in_arc"))
            .group_by("p", "n").agg(pl.col("__in_arc").all())
            .rename({"p": "__p", "n": "__n"}))
    indirect = getattr(d, "process_indirect", None)
    ind_p = (_u8(indirect.lazy().select("p"), ["p"]).unique().rename({"p": "__p"})
             .with_columns(pl.lit(True).alias("__ind"))
             if indirect is not None and indirect.height > 0 else None)
    grid = (in_rows.lazy().select("__p", "__n").join(dt_u, how="cross")
            .join(kind, on=["__p", "__n"], how="left"))
    grid = (grid.join(ind_p, on="__p", how="left") if ind_p is not None
            else grid.with_columns(pl.lit(None, pl.Boolean).alias("__ind")))
    # Capped input arcs of indirect units in the rows: their p_indirect_input_cap (raises when missing, as
    # maxInputFlow_online does).  conversion_flow_coeff = 0 inputs have no cap: factor 1 (__unc).
    ind_in = (inputs.filter(pl.col("sink") == pl.col("p"))
              .join(in_rows.lazy().select(pl.col("__p").alias("p"), pl.col("__n").alias("n")), on=["p", "n"],
                    how="semi"))
    if ind_p is not None:
        ind_in = ind_in.join(ind_p.select(pl.col("__p").alias("p")), on="p", how="semi")
    else:
        ind_in = ind_in.head(0)
    ind_in = ind_in.select("p", pl.col("n").alias("source"), "sink").collect()
    unc = getattr(d, "process_source_sink_uncapped", None)
    unc_keys = (_u8(unc.lazy().select(keys), keys).select(pl.col("p").alias("__p"), pl.col("source").alias("__n"))
                .unique().with_columns(pl.lit(True).alias("__unc"))
                if unc is not None and unc.height > 0 else None)
    capped_in = _drop_uncapped(d, ind_in)
    if capped_in.height > 0:
        cap = (_u8(_indirect_input_cap(d, capped_in).lazy(), [*keys, "d", "t"])
               .select(pl.col("p").alias("__p"), pl.col("source").alias("__n"), "d", "t",
                       pl.col("value").alias("__cap")))
        grid = grid.join(cap, on=["__p", "__n", "d", "t"], how="left")
    else:
        grid = grid.with_columns(pl.lit(None, pl.Float64).alias("__cap"))
    grid = (grid.join(unc_keys, on=["__p", "__n"], how="left") if unc_keys is not None
            else grid.with_columns(pl.lit(None, pl.Boolean).alias("__unc")))
    slope = getattr(d, "p_slope", None)
    if slope is not None:
        grid = grid.join(_u8(promote_param_to_dt(slope, d.dt).select(pl.col("p").alias("__p"), "d", "t",
                                                                      pl.col("value").alias("__slope")),
                             ["__p", "d", "t"]), on=["__p", "d", "t"], how="left")
    else:
        grid = grid.with_columns(pl.lit(None, pl.Float64).alias("__slope"))
    grid = _with_section(d, grid)
    direct_c = pl.col("__slope").fill_null(1.0) + pl.col("__sec").fill_null(0.0)
    value = (pl.when(pl.col("__in_arc") & pl.col("__ind").fill_null(False) & ~pl.col("__unc").fill_null(False))
             .then(pl.col("__cap"))
             .when(pl.col("__in_arc")).then(pl.lit(1.0))
             .otherwise(direct_c))
    c_in = grid.select("__p", "__n", "d", "t", value.alias("value"))
    base = rows.join(d.dt.select("d", "t"), how="cross")
    frame = (base.with_columns(pl.col("d").cast(pl.Utf8).alias("__d"), pl.col("t").cast(pl.Utf8).alias("__t"))
             .lazy()
             .join(c_in.rename({"d": "__d", "t": "__t"}), on=["__p", "__n", "__d", "__t"], how="left")
             .with_columns(pl.col("value").fill_null(1.0))
             .select("p", "n", "d", "t", "value")
             .collect())
    return Param(("p", "n", "d", "t"), frame)


def _with_section(d, grid: "pl.LazyFrame") -> "pl.LazyFrame":
    """``grid`` (Utf8 ``__p``, ``d``, ``t``) with ``__sec``: p_section of min_load_efficiency units with online
    variables, null elsewhere."""
    section = getattr(d, "p_section", None)
    pml = getattr(d, "process_min_load_eff", None)
    online = [f for f in (getattr(d, "process_online_linear", None), getattr(d, "process_online_integer", None))
              if f is not None and f.height > 0]
    if section is None or pml is None or pml.height == 0 or not online:
        return grid.with_columns(pl.lit(None, pl.Float64).alias("__sec"))
    on_p = pl.concat([_u8(f.lazy().select("p"), ["p"]) for f in online]).unique()
    sec = (_u8(promote_param_to_dt(section, d.dt).select("p", "d", "t", pl.col("value").alias("__sec")),
               ["p", "d", "t"])
           .join(_u8(pml.lazy().select("p"), ["p"]).unique(), on="p", how="semi")
           .join(on_p, on="p", how="semi")
           .rename({"p": "__p"}))
    return grid.join(sec, on=["__p", "d", "t"], how="left")


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
        parts = []
        orientations = [(d.process_source_sink, balance_side, storage_side, False)]
        back = _back_arcs(d)
        if back is not None:
            # The reverse flow of a one-variable / DC arc runs from its sink to its source.
            orientations.append((back, storage_side, balance_side, True))
        for arcs, bal, sto, is_back in orientations:
            j = arcs.pipe(rename_to_axis, {bal: "n"}).join(pr, on=["p", "n"], how="inner")
            if j.height == 0:
                continue
            parts.append(j.rename({"n": "nb"})
                         .pipe(rename_to_axis, {sto: "n"})
                         .join(gn.rename({"n": "nb"}), on="nb", how="inner")
                         .join(d.reserve_upDown_group, on=["r", "ud", "g"], how="inner")
                         .join(dur_set, on=["r", "ud", "g"], how="inner")
                         .join(d.nodeState.select("n"), on="n", how="semi")
                         .select("p", "r", "ud", "nb", "n", "g")
                         .with_columns(pl.lit(is_back).alias("__back")))
        if not parts:
            return None
        j = pl.concat(parts).unique(subset=["p", "r", "ud", "nb", "n", "g"], keep="first", maintain_order=True)
        if j.height == 0:
            return None
        if not j["__back"].any():
            # Byte-identical to the v_flow-only rows (row set and order).
            j = j.drop("__back").unique()
        return j

    def _slope_factor(providers: pl.DataFrame, reciprocal: bool, back: "pl.DataFrame | None" = None) -> Param:
        """slope[p,d,t] over the providers × dt, 1.0 where p_slope has no value (noEff providers) and for the
        providers in ``back`` (reverse flow of a one-variable / DC arc: lossless); 1 / slope when ``reciprocal``."""
        grid = providers.join(dt, how="cross")
        if d.p_slope is not None:
            grid = grid.join(d.p_slope.frame, on=["p", "d", "t"], how="left")
            val = pl.col("value").fill_null(1.0)
        else:
            val = pl.lit(1.0)
        if back is not None and back.height > 0:
            grid = grid.with_columns(pl.col("p").is_in(back["p"].implode()).alias("__b"))
            val = pl.when(pl.col("__b")).then(pl.lit(1.0)).otherwise(val)
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
    #   providers_up: upward reserve for nb by a process with source n ∈ nodeState and sink nb, or by a
    #     one-variable / DC connection whose arc runs nb → n (its reverse flow v_flow_back runs n → nb; slope 1)
    # Upward reserve from storage is backed by stored energy (MWh) for the
    # reserve duration (hours); slope = 1/η converts output to energy drawn.
    # Reliability is not applied: the energy matches the full committed MW.
    # No slack: with v_reserve = 0 the row is v_state ≥ 0, so a provider
    # that cannot sustain the duration commits less and the group pays the
    # shortfall penalty.  Couples to the intra-period state v_state only,
    # not to the representative-period state v_state_inter.
    join_up = _providers("up", balance_side="sink", storage_side="source")
    if join_up is not None:
        back_up = None
        if "__back" in join_up.columns:
            back_up = join_up.filter(pl.col("__back")).select("p").unique()
            join_up = join_up.drop("__back")
        slope_up = _slope_factor(join_up.select("p").unique(), reciprocal=False, back=back_up)
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
    #   providers_down: downward reserve for nb by a process with source nb and sink n ∈ nodeState, or by a
    #     one-variable / DC connection whose arc runs n → nb (its reverse flow runs nb → n; slope 1)
    # Downward reserve into storage needs free storage capacity (MWh) for the
    # reserve duration; 1/slope = η converts input to energy stored.  The
    # capacity terms are those of maxState, in MWh.  Same notes as the floor.
    join_dn = _providers("down", balance_side="source", storage_side="sink")
    if join_dn is not None:
        back_dn = None
        if "__back" in join_dn.columns:
            back_dn = join_dn.filter(pl.col("__back")).select("p").unique()
            join_dn = join_dn.drop("__back")
        slope_dn = _slope_factor(join_dn.select("p").unique(), reciprocal=True, back=back_dn)
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
