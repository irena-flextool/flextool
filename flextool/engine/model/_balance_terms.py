"""Flow terms of the node energy balance, shared by nodeBalance_eq and the non-anticipativity storage tie.

One construction for both, so the tie (the net flow into a stochastic node is the same in every forecast) can
never drift from the flow part of the balance it is meant to equal.
"""

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.input._axis_enums import join_aligned
from flextool.engine.model import _dc_power_flow
from flextool.engine.model._common import _has_rows, _var_view

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def _at_nodes(frame: "pl.DataFrame | None", names: "list[str] | None", col: str = "n") -> "pl.DataFrame | None":
    """Rows of ``frame`` whose ``col`` is one of ``names`` (compared as text); ``frame`` itself without names."""
    if frame is None or names is None:
        return frame
    return frame.filter(pl.col(col).cast(pl.Utf8).is_in(names))


def _arcs_for(frame: "pl.DataFrame | None", group_nodes: "pl.DataFrame | None",
              exclude_arcs: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """``frame`` (p, source, sink, n) at the ``(g, n)`` pairs of ``group_nodes`` (g added), without the arcs of
    ``exclude_arcs``; ``frame`` itself without either."""
    if frame is None:
        return None
    if group_nodes is not None:
        frame = join_aligned(frame, group_nodes.select("g", "n"), on="n", how="inner")
    if exclude_arcs is not None and exclude_arcs.height > 0 and frame.height > 0:
        keys = [k for k in ("g", "p", "source", "sink") if k in exclude_arcs.columns and k in frame.columns]
        frame = join_aligned(frame, exclude_arcs.select(keys).unique(), on=keys, how="anti")
    return frame


def balance_flow_terms(b: "_Build", *, at_branch: bool = False, nodes: "pl.DataFrame | None" = None,
                       group_nodes: "pl.DataFrame | None" = None, exclude_arcs: "pl.DataFrame | None" = None,
                       per_step: bool = False) -> dict:
    """Flow terms of nodeBalance_eq in MWh, open dims (n, d, t); (n, b, t) with ``at_branch``.

    sink_flow[n,d,t]          =   Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t] · unitsize[p] · Δt[d,t]
    source_eff[n,d,t]         = − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff} v_flow[p,n,k,d,t] · unitsize[p] · slope[p,d,t]
                                    · Δt[d,t]
    source_section_{lin,int}  = − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff, p ∈ process_min_load_eff}
                                    v_online[p,d,t] · section[p,d,t] · unitsize[p] · Δt[d,t]
    source_noEff[n,d,t]       = − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t] · unitsize[p] · Δt[d,t]
    back_at_sink / back_at_source: the v_flow_back terms of twoway1var_arcs
                                    (_dc_power_flow.nodeBalance_back_flow_terms)
    A unit with an efficiency draws flow · slope from its source node; a min_load_efficiency unit also draws its
    section input per online unit.  v_flow_back adds to the source node and takes from the sink node.

    ``at_branch`` indexes every variable and every (d,t) parameter at the forecast branch ``b`` instead of ``d``
    (the b side of non_anticipativity_storage_use).  ``nodes`` (a frame with ``n``) keeps only the arcs at those
    nodes; None keeps every arc.  The capacity margin's terms (capacityMargin):

    * ``group_nodes`` (g, n): each term at the pairs (g, n) only, open dims (g, n, d, t), so an arc counts at a
      node of g only for g (a back-flow term at an arc's other node outside g is left out);
    * ``exclude_arcs`` (p, source, sink[, g]): those arcs (for that g) are left out of every term;
    * ``per_step``: the terms are power (MW), without the Δt factor.
    """
    d = b.d
    terms: dict = {}
    if not b.has_proc:
        return terms
    names = None if nodes is None else sorted({str(x) for x in nodes.get_column("n").to_list()})

    def var(v):
        return _var_view(v, "__nab", {"d": "b"}) if at_branch else v

    def at_dt(param: Param, dims: tuple) -> Param:
        if not at_branch:
            return param
        return Param(tuple("b" if x == "d" else x for x in dims), param.frame.rename({"d": "b"}))

    def arcs(frame):
        return _arcs_for(_at_nodes(frame, names), group_nodes, exclude_arcs)

    v_flow = var(b.v_flow)
    step_dur = None if per_step else at_dt(d.p_step_duration, ("d", "t"))

    def energy(term):
        return term if step_dur is None else term * step_dur

    to_n = arcs(d.flow_to_n)
    if _has_rows(to_n):
        terms["sink_flow"] = Sum(
            energy(Where(v_flow * d.p_unitsize, to_n)),
            over=("p", "source", "sink"))
    from_eff = arcs(d.flow_from_nodeBalance_eff)
    if _has_rows(from_eff):
        slope = at_dt(d.p_slope, ("p", "d", "t"))
        terms["source_eff"] = -Sum(
            energy(Where(v_flow * d.p_unitsize * slope, from_eff)),
            over=("p", "source", "sink"))
        if b.has_minload_eff and d.p_section is not None:
            section_idx = from_eff.join(d.process_min_load_eff, on="p", how="inner")
            if section_idx.height > 0:
                section = at_dt(d.p_section, ("p", "d", "t"))
                for key, v_online in (("source_section_lin", b.v_online_linear),
                                      ("source_section_int", b.v_online_integer)):
                    if v_online is None:
                        continue
                    terms[key] = -Sum(
                        energy(Where(Where(var(v_online), d.process_min_load_eff)
                                     * section * d.p_unitsize,
                                     section_idx)),
                        over=("p", "source", "sink"))
    from_noEff = arcs(d.flow_from_nodeBalance_noEff)
    if _has_rows(from_noEff):
        terms["source_noEff"] = -Sum(
            energy(Where(v_flow * d.p_unitsize, from_noEff)),
            over=("p", "source", "sink"))
    if b.v_flow_back is not None and b.twoway1var_arcs is not None:
        back_arcs = b.twoway1var_arcs
        if names is not None:
            back_arcs = back_arcs.filter(pl.col("sink").cast(pl.Utf8).is_in(names)
                                         | pl.col("source").cast(pl.Utf8).is_in(names))
        if exclude_arcs is not None and exclude_arcs.height > 0 and "g" not in exclude_arcs.columns:
            back_arcs = join_aligned(back_arcs, exclude_arcs.select("p", "source", "sink").unique(),
                                     on=["p", "source", "sink"], how="anti")
        terms.update(_dc_power_flow.nodeBalance_back_flow_terms(
            back_arcs, var(b.v_flow_back), d.p_unitsize, step_dur, group_nodes=group_nodes,
            exclude_arcs=exclude_arcs if exclude_arcs is not None and "g" in exclude_arcs.columns else None))
    return terms
