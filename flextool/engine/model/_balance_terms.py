"""Flow terms of the node energy balance, shared by nodeBalance_eq and the non-anticipativity storage tie.

One construction for both, so the tie (the net flow into a stochastic node is the same in every forecast) can
never drift from the flow part of the balance it is meant to equal.
"""

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.model import _dc_power_flow
from flextool.engine.model._common import _has_rows, _var_view

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def _at_nodes(frame: "pl.DataFrame | None", names: "list[str] | None", col: str = "n") -> "pl.DataFrame | None":
    """Rows of ``frame`` whose ``col`` is one of ``names`` (compared as text); ``frame`` itself without names."""
    if frame is None or names is None:
        return frame
    return frame.filter(pl.col(col).cast(pl.Utf8).is_in(names))


def balance_flow_terms(b: "_Build", *, at_branch: bool = False, nodes: "pl.DataFrame | None" = None) -> dict:
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
    nodes; None keeps every arc.
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

    v_flow = var(b.v_flow)
    step_dur = at_dt(d.p_step_duration, ("d", "t"))
    to_n = _at_nodes(d.flow_to_n, names)
    if _has_rows(to_n):
        terms["sink_flow"] = Sum(
            Where(v_flow * d.p_unitsize, to_n) * step_dur,
            over=("p", "source", "sink"))
    from_eff = _at_nodes(d.flow_from_nodeBalance_eff, names)
    if _has_rows(from_eff):
        slope = at_dt(d.p_slope, ("p", "d", "t"))
        terms["source_eff"] = -Sum(
            Where(v_flow * d.p_unitsize * slope, from_eff) * step_dur,
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
                        Where(Where(var(v_online), d.process_min_load_eff)
                              * section * d.p_unitsize,
                              section_idx) * step_dur,
                        over=("p", "source", "sink"))
    from_noEff = _at_nodes(d.flow_from_nodeBalance_noEff, names)
    if _has_rows(from_noEff):
        terms["source_noEff"] = -Sum(
            Where(v_flow * d.p_unitsize, from_noEff) * step_dur,
            over=("p", "source", "sink"))
    if b.v_flow_back is not None and b.twoway1var_arcs is not None:
        arcs = b.twoway1var_arcs
        if names is not None:
            arcs = arcs.filter(pl.col("sink").cast(pl.Utf8).is_in(names)
                               | pl.col("source").cast(pl.Utf8).is_in(names))
        terms.update(_dc_power_flow.nodeBalance_back_flow_terms(
            arcs, var(b.v_flow_back), d.p_unitsize, step_dur))
    return terms
