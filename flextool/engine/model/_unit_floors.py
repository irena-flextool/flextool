"""Unit output floors: minimum load, per-output floors and downward-reserve headroom.

Owns the rows minFlow_minload_*, minFlow_output_floor_*, minFlow_reserve and minFlow_reserve_arc.  They keep
a unit's flow on its output measure A_p (:func:`unit_measured_arcs`) above a floor that already includes Rdn,
the reserve that reduces the arc (``down`` at the sink, ``up`` at the source; see
:func:`_reducing_reserve_selectors`).  Connections are excluded.  An indirect unit cutting its input gets
only the zero floor on that input arc, so with downward reserve on its outputs it can end up below its
minimum load.  See docs/dev/unit_floors_and_reserves.md.
"""

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.model._common import _drop_uncapped, _indirect_sum_arcs, _utf8_keys, _var_view
from flextool.engine.model._online import uc_classes
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.input._solve_state import FlexToolConfigError

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def add_constraints(b: "_Build") -> None:
    """Minimum load, per-output floors and the downward-reserve headroom of units."""
    if not b.has_proc:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow
    v_reserve = b.v_reserve
    arcs = unit_measured_arcs(d)
    if arcs is None:
        return
    keys = ["p", "source", "sink"]
    a_count = arcs.group_by("p", maintain_order=True).agg(pl.len().alias("__n"))
    multi_arcs = arcs.join(a_count.filter(pl.col("__n") >= 2).select("p"), on="p", how="semi")

    # Reducing-reserve selectors over all unit arcs (sel_all) and over A_p (sel_meas).
    all_unit_arcs = None
    sel_meas = (None, None, None)
    sel_all = (None, None, None)
    views = None
    if v_reserve is not None:
        units = getattr(d, "process_unit", None)
        pss = d.process_source_sink
        if units is not None and units.height > 0:
            all_unit_arcs = (_utf8_keys(pss.lazy().select(keys), ["p"])
                             .join(_utf8_keys(units.lazy().select("p"), ["p"]).select("_p").unique(),
                                   on="_p", how="semi")
                             .select(keys).collect())
            sel_all = _reducing_reserve_selectors(d, all_unit_arcs)
            if any(s is not None for s in sel_all):
                sel_meas = _restrict_selectors(sel_all, arcs)
                # v_reserve with its node axis relabelled sink / source (same columns and name).
                views = (_var_view(v_reserve, "", {"n": "sink"}, to_axis=True),
                         _var_view(v_reserve, "", {"n": "source"}, to_axis=True))

    minload = getattr(d, "process_minload", None)
    minload_rows: list = []
    floor_rows: list = []
    if minload is not None and minload.height > 0 and getattr(d, "p_min_load", None) is not None:
        # capacity_min_coeff of the outputs of multi-output units, keeping the > 0 ones.
        cmin = getattr(d, "p_process_sink_min_capacity_coef", None)
        cmin_arcs = None
        if cmin is not None and multi_arcs.height > 0:
            cmin_lf = (cmin.frame.lazy()
                       .with_columns(pl.col("p").cast(pl.Utf8).alias("_p"),
                                     pl.col("sink").cast(pl.Utf8).alias("_sink"))
                       .select("_p", "_sink", "value"))
            cmin_arcs = (_utf8_keys(multi_arcs.lazy(), keys)
                         .join(cmin_lf, on=["_p", "_sink"], how="inner")
                         .select(*keys, "value").collect())
            bad = cmin_arcs.filter((pl.col("value") < 0.0) | (pl.col("value") > 1.0))
            if bad.height > 0:
                raise FlexToolConfigError(
                    "capacity_min_coeff must lie in [0, 1]; found "
                    + ", ".join(f"{r['p']}->{r['sink']}: {r['value']}"
                                for r in bad.iter_rows(named=True)))
            cmin_arcs = cmin_arcs.filter(pl.col("value") > 0.0)
            if cmin_arcs.height == 0:
                cmin_arcs = None
        for kind, p_idx, online_set, v_online, _v_startup, _v_shutdown in uc_classes(b):
            sfx = f"_{kind}"
            on_ml = (_utf8_keys(online_set.lazy().select("p"), ["p"])
                     .join(_utf8_keys(minload.lazy().select("p"), ["p"]).select("_p").unique(),
                           on="_p", how="semi")
                     .select("_p").unique())
            a_on = (_utf8_keys(arcs.lazy(), ["p"])
                    .join(on_ml, on="_p", how="semi")
                    .select(keys).collect())
            _assert_minload_units_have_outputs(d, on_ml, a_on)
            if a_on.height == 0:
                continue
            floor = Where(v_online, d.process_minload) * d.p_min_load

            # ─── EQ. ─── minFlow_minload_{linear,integer} ────────────────────
            # ─────────────────────────────────────────────────────────────────
            # Σ_{(p,s,k) ∈ A_p} v_flow[p,s,k,d,t]  ≥  v_online[p,d,t] · min_load[p] + Rdn_p[p,d,t]
            #   ∀ p online with min_load, (p,d,t) ∈ p_idx
            #   process_minload = units with min_load > 0; min_load not authored = 0 (no floor)
            # Minimum load, once per unit on the sum of its output measure.  Rdn_p sums the
            # reducing reserves over A_p, so downward reserve cannot take the unit below it.
            # The floor is not scaled by availability.
            over = (a_on.select("p").unique(maintain_order=True)
                    .join(d.dt, how="cross")
                    .join(p_idx.select("p", "d", "t"), on=["p", "d", "t"], how="semi"))
            if over.height > 0:
                rhs: dict = {"floor": floor}
                if views is not None:
                    rhs.update(_rdn_terms(d, views, _restrict_selectors(sel_meas, a_on), per_arc=False))
                m.add_cstr(f"minFlow_minload{sfx}", over=over, sense=">=",
                           lhs_terms={"flow_sum": Sum(Where(v_flow, a_on), over=("source", "sink"))},
                           rhs_terms=rhs)
                minload_rows.append(over.select("p", "d", "t"))

            # ─── EQ. ─── minFlow_output_floor_{linear,integer} ───────────────
            # ─────────────────────────────────────────────────────────────────
            # v_flow[p,p,k,d,t]  ≥  v_online[p,d,t] · min_load[p] · capacity_min_coeff[p,k] + Rdn_k[p,k,d,t]
            #   ∀ (p,p,k) ∈ A_p of a multi-output unit with capacity_min_coeff[p,k] > 0, (p,d,t) ∈ p_idx
            # Per-output floor of a running unit; capacity_min_coeff must lie in [0, 1].
            if cmin_arcs is None:
                continue
            f_arcs = (_utf8_keys(cmin_arcs.lazy(), ["p"])
                      .join(on_ml, on="_p", how="semi")
                      .select(*keys, "value").collect())
            if f_arcs.height == 0:
                continue
            f_over = (f_arcs.select(keys)
                      .join(d.dt, how="cross")
                      .join(p_idx.select("p", "d", "t"), on=["p", "d", "t"], how="semi"))
            if f_over.height == 0:
                continue
            coef = Param(("p", "sink"), f_arcs.select("p", "sink", "value"))
            rhs = {"floor": floor * coef}
            if views is not None:
                rhs.update(_rdn_terms(d, views, _restrict_selectors(sel_meas, f_arcs.select(keys)), per_arc=True))
            m.add_cstr(f"minFlow_output_floor{sfx}", over=f_over,
                       sense=">=", lhs_terms={"flow": v_flow},
                       rhs_terms=rhs)
            floor_rows.append(f_over)

    if views is None:
        return
    res_pdt = d.prundt.select("p", "d", "t").unique(maintain_order=True)

    # ─── EQ. ─── minFlow_reserve ─────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k) ∈ A_p} v_flow[p,s,k,d,t]  ≥  Rdn_p[p,d,t]
    #   ∀ unit p with a reducing reserve on A_p, (p,d,t) ∈ prundt without a minFlow_minload row
    # Zero floor of units without a minimum-load row: a unit cannot offer more downward reserve
    # than its output.
    meas_hit = _selector_arcs(sel_meas, arcs)
    if meas_hit.height > 0:
        a_res = arcs.join(meas_hit.select("p").unique(), on="p", how="semi")
        over = (a_res.select("p").unique(maintain_order=True)
                .join(d.dt, how="cross")
                .join(res_pdt, on=["p", "d", "t"], how="semi"))
        if minload_rows:
            over = over.join(pl.concat(minload_rows).unique(), on=["p", "d", "t"], how="anti")
        if over.height > 0:
            m.add_cstr("minFlow_reserve", over=over, sense=">=",
                       lhs_terms={"flow_sum": Sum(Where(v_flow, a_res), over=("source", "sink"))},
                       rhs_terms=_rdn_terms(d, views, sel_meas, per_arc=False))

    # ─── EQ. ─── minFlow_reserve_arc ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t]  ≥  Rdn_a[p,s,k,d,t]
    #   ∀ unit arc (p,s,k) with a reducing reserve, outside A_p or one of several arcs in A_p,
    #     (p,d,t) ∈ prundt, without a minFlow_output_floor row
    # Per-arc zero floor where the per-unit rows cannot see the arc.
    all_hit = _selector_arcs(sel_all, all_unit_arcs)
    if all_hit.height == 0:
        return
    outside = (_utf8_keys(all_hit.lazy(), keys)
               .join(_utf8_keys(arcs.lazy(), keys).select("_p", "_source", "_sink"),
                     on=["_p", "_source", "_sink"], how="anti")
               .select(keys).collect())
    in_multi = (_utf8_keys(all_hit.lazy(), keys)
                .join(_utf8_keys(multi_arcs.lazy(), keys).select("_p", "_source", "_sink"),
                      on=["_p", "_source", "_sink"], how="semi")
                .select(keys).collect())
    arc_rows = pl.concat([outside, in_multi]).unique(maintain_order=True)
    if arc_rows.height == 0:
        return
    over = arc_rows.join(d.dt, how="cross").join(res_pdt, on=["p", "d", "t"], how="semi")
    if floor_rows:
        over = over.join(pl.concat(floor_rows).unique(), on=[*keys, "d", "t"], how="anti")
    if over.height == 0:
        return
    m.add_cstr("minFlow_reserve_arc", over=over, sense=">=",
               lhs_terms={"flow": v_flow},
               rhs_terms=_rdn_terms(d, views, _restrict_selectors(sel_all, arc_rows), per_arc=True))


def unit_measured_arcs(d) -> "pl.DataFrame | None":
    """The output-measure arcs ``A_p`` (``p, source, sink``) of every unit.

    * indirect units: their ``conversion_flow_coeff ≠ 0`` output arcs
      (:func:`_indirect_sum_arcs` — the ``maxOutputSum`` arc set);
    * direct units: their ``process_source_sink`` arcs minus the
      ``conversion_flow_coeff = 0`` (uncapped) ones — a single arc.

    Units are ``process_unit`` when populated; hand-built FlexData without
    it fall back to ``process_minload`` (min_load exists only on units).
    ``None`` when empty.
    """
    pss = getattr(d, "process_source_sink", None)
    if pss is None or pss.height == 0:
        return None
    units = getattr(d, "process_unit", None)
    if units is None or units.height == 0:
        units = getattr(d, "process_minload", None)
    if units is None or units.height == 0:
        return None
    keys = ["p", "source", "sink"]
    unit_p = _utf8_keys(units.lazy().select("p"), ["p"]).select("_p").unique()
    pss_u = (_utf8_keys(pss.lazy().select(keys), ["p"])
             .join(unit_p, on="_p", how="semi")
             .select(keys).collect())
    if pss_u.height == 0:
        return None
    indirect = getattr(d, "process_indirect", None)
    parts = []
    if indirect is not None and indirect.height > 0:
        ind_p = _utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique()
        direct = (_utf8_keys(pss_u.lazy(), ["p"])
                  .join(ind_p, on="_p", how="anti")
                  .select(keys).collect())
        sum_arcs = _indirect_sum_arcs(d)
        if sum_arcs is not None:
            parts.append(
                _utf8_keys(pss_u.lazy(), keys)
                .join(_utf8_keys(sum_arcs.lazy(), keys).select("_p", "_source", "_sink"),
                      on=["_p", "_source", "_sink"], how="semi")
                .select(keys).collect())
    else:
        direct = pss_u
    direct = _drop_uncapped(d, direct)
    if direct.height > 0:
        parts.append(direct)
    if not parts:
        return None
    out = (pl.concat(parts) if len(parts) > 1 else parts[0]).unique(maintain_order=True)
    return out if out.height > 0 else None


def _reducing_reserve_selectors(d, arcs: "pl.DataFrame"):
    """Selectors of the reserves that reduce the unit arcs ``arcs``.

    Returns ``(down_sink, up_src_flow, up_src_eff)`` — frames keyed
    ``(p, r, ud, sink)`` / ``(p, r, ud, source)`` / ``(p, r, ud, source)``
    (``None`` when empty):

    * ``down_sink``: ``down`` reserve at the arc's sink node;
    * ``up_src_flow``: ``up`` reserve at the source node of an arc whose
      ``v_flow`` is the input itself (``sink == p``) — factor 1;
    * ``up_src_eff``: ``up`` reserve at the source node of a direct
      input→output arc (``source ≠ p ≠ sink``) — factor ``1 / slope``.

    Only ``process_reserve_upDown_node_active`` rows that also index the
    ``v_reserve`` domain (``prundt``) are kept.
    """
    pruna = getattr(d, "process_reserve_upDown_node_active", None)
    prundt = getattr(d, "prundt", None)
    if (pruna is None or pruna.height == 0 or prundt is None
            or prundt.height == 0 or arcs is None or arcs.height == 0):
        return None, None, None
    rk = ["p", "r", "ud", "n"]
    act = (_utf8_keys(pruna.lazy().select(rk), rk)
           .join(_utf8_keys(prundt.lazy().select(rk), rk)
                 .select([f"_{k}" for k in rk]).unique(),
                 on=[f"_{k}" for k in rk], how="semi"))
    a = (_utf8_keys(arcs.lazy().select("p", "source", "sink"), ["p", "source", "sink"])
         .select("_p", "_source", "_sink"))

    def _sel(ud: str, side: str, arc_filter):
        cand = (act.filter(pl.col("ud").cast(pl.Utf8) == ud)
                .join(a.filter(arc_filter)
                      .select("_p", pl.col(f"_{side}").alias("_n"))
                      .unique(),
                      on=["_p", "_n"], how="semi")
                .select("p", "r", "ud", "n")
                .collect())
        if cand.height == 0:
            return None
        return cand.pipe(rename_to_axis, {"n": side}).unique(maintain_order=True)

    down_sink = _sel("down", "sink", pl.col("_sink") != pl.col("_p"))
    up_src_flow = _sel("up", "source",
                       (pl.col("_source") != pl.col("_p"))
                       & (pl.col("_sink") == pl.col("_p")))
    up_src_eff = _sel("up", "source",
                      (pl.col("_source") != pl.col("_p"))
                      & (pl.col("_sink") != pl.col("_p")))
    return down_sink, up_src_flow, up_src_eff


def _inverse_slope(d, procs: "pl.DataFrame") -> "Param":
    """``1 / slope`` over ``procs × dt`` (default slope 1).

    Converts a source-side reserve (node MW) into the output flow of a
    direct input→output unit.  It relies on the ``reserveBalance``
    convention of ``_reserve.add_constraints``, which counts every
    ``v_reserve`` as MW at the reserve node with no slope factor for
    source-side reserves of direct units.  If that factor is added there,
    this conversion must change with it (tripwire test:
    ``test_down_reserve_floor.py::test_direct_unit_input_side_up_reserve``).
    """
    base = procs.select("p").unique().join(d.dt, how="cross")
    base_lf = base.lazy().with_columns(pl.col("p").cast(pl.Utf8).alias("_k"))
    slope = getattr(d, "p_slope", None)
    if slope is not None:
        slope_lf = (promote_param_to_dt(slope, d.dt)
                    .with_columns(pl.col("p").cast(pl.Utf8).alias("_k"))
                    .select("_k", "d", "t", "value"))
        base_lf = base_lf.join(slope_lf, on=["_k", "d", "t"], how="left")
    else:
        base_lf = base_lf.with_columns(pl.lit(None, pl.Float64).alias("value"))
    frame = (base_lf
             .with_columns(pl.col("value").fill_null(1.0))
             .select("p", "d", "t", value=1.0 / pl.col("value"))
             .collect())
    return Param(("p", "d", "t"), frame)


def _rdn_terms(d, views, selectors, *, per_arc: bool) -> dict:
    """The reducing-reserve RHS terms for a floor row.

    ``views`` is ``v_reserve`` relabelled ``(at_sink, at_source)``.
    ``per_arc=False`` sums over every reserve and node of the selected
    arcs (leaves ``(p, d, t)``); ``per_arc=True`` keeps the node axis
    (leaves ``(p, sink, d, t)`` / ``(p, source, d, t)``) for rows over
    ``(p, source, sink, d, t)``.
    """
    at_sink, at_source = views
    down_sink, up_src_flow, up_src_eff = selectors
    terms: dict = {}
    if down_sink is not None:
        over = ("r", "ud") if per_arc else ("r", "ud", "sink")
        terms["reserve_down_sink"] = Sum(Where(at_sink, down_sink), over=over)
    if up_src_flow is not None:
        over = ("r", "ud") if per_arc else ("r", "ud", "source")
        terms["reserve_up_source"] = Sum(Where(at_source, up_src_flow), over=over)
    if up_src_eff is not None:
        over = ("r", "ud") if per_arc else ("r", "ud", "source")
        terms["reserve_up_source_eff"] = Sum(Where(at_source, up_src_eff) * _inverse_slope(d, up_src_eff), over=over)
    return terms


def _restrict_selectors(selectors, arcs: "pl.DataFrame"):
    """Keep the selector rows whose ``(p, node)`` belongs to ``arcs``."""
    out = []
    a = _utf8_keys(arcs.lazy().select("p", "source", "sink"), ["p", "source", "sink"])
    for sel in selectors:
        if sel is None:
            out.append(None)
            continue
        side = "sink" if "sink" in sel.columns else "source"
        keep = (_utf8_keys(sel.lazy(), ["p", side])
                .join(a.select("_p", f"_{side}").unique(), on=["_p", f"_{side}"], how="semi")
                .select(sel.columns).collect())
        out.append(keep if keep.height > 0 else None)
    return tuple(out)


def _selector_arcs(selectors, arcs: "pl.DataFrame") -> "pl.DataFrame":
    """The ``arcs`` rows that carry at least one selected reserve."""
    keys = ["p", "source", "sink"]
    a = _utf8_keys(arcs.lazy().select(keys), keys)
    hit = []
    for sel in selectors:
        if sel is None:
            continue
        side = "sink" if "sink" in sel.columns else "source"
        s = (_utf8_keys(sel.lazy().select("p", side), ["p", side])
             .select("_p", f"_{side}").unique())
        hit.append(a.join(s, on=["_p", f"_{side}"], how="semi").select(keys).collect())
    if not hit:
        return arcs.head(0).select(keys)
    return pl.concat(hit).unique(maintain_order=True)


def _assert_minload_units_have_outputs(d, on_ml: "pl.LazyFrame", a_on: "pl.DataFrame") -> None:
    """Every online min-load INDIRECT unit must have an output arc.

    An indirect unit whose outputs all carry ``conversion_flow_coeff = 0``
    has no output measure and gets no floor (the validator warns once per
    scenario).  A sink-less indirect unit (≥ 2 inputs, no output) cannot
    reach this point today (``model._max_flow_rhs`` raises for it); if that
    ever changes it must not silently lose its floor, hence the internal error.
    """
    indirect = getattr(d, "process_indirect", None)
    if indirect is None or indirect.height == 0:
        return
    missing = (on_ml
               .join(_utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi")
               .join(_utf8_keys(a_on.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="anti")
               .collect())
    if missing.height == 0:
        return
    pss = _utf8_keys(d.process_source_sink.lazy(), ["p", "source"])
    has_out = pss.filter(pl.col("_source") == pl.col("_p")).select("_p").unique().collect()
    sinkless = missing.join(has_out, on="_p", how="anti")
    if sinkless.height > 0:
        raise AssertionError(
            "internal error: online min-load indirect unit(s) without an "
            "output arc have no minimum-load floor: "
            + ", ".join(sinkless["_p"].to_list()))
