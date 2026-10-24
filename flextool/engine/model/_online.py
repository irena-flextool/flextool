"""Unit commitment: units online, startups, shutdowns and minimum up/down times.

Owns the variables v_online_*, v_startup_*, v_shutdown_* and the equations maxOnline, maxStartup,
maxShutdown, online__startup, online__shutdown, maxFlow_online, maxOutputSum_online, maxInputFlow_online,
minimum_uptime, minimum_downtime and online_fix_eq.  Each exists once per UC class present: ``linear`` (continuous
v_online) and ``integer`` (integer v_online), suffixed ``_linear`` / ``_integer``.
"""

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Lag, Param, Sum, Where

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.model._common import (
    _availability_factor, _capped_arcs, _coef_factor, _entity_rows, _indirect_sum_arcs, _utf8_keys, _var_view,
)

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


def add_variables(b: "_Build") -> None:
    """Declare the online, startup and shutdown variables of each UC class."""
    if not b.has_online:
        return
    m = b.m

    # ─── VAR. ─── v_online_linear / v_startup_linear / v_shutdown_linear ─────
    # ─────────────────────────────────────────────────────────────────────────
    # v_online_linear[p,d,t], v_startup_linear[p,d,t], v_shutdown_linear[p,d,t] ≥ 0
    #   ∀ (p,d,t) ∈ p_online_dt, p ∈ process_online_linear
    # Units online, starting up and shutting down (count of unitsize),
    # continuous.
    if b.has_online_lin:
        b.v_online_linear = m.add_var("v_online_linear", ("p", "d", "t"), b.p_olin_idx)
        b.v_startup_linear = m.add_var("v_startup_linear", ("p", "d", "t"), b.p_olin_idx)
        b.v_shutdown_linear = m.add_var("v_shutdown_linear", ("p", "d", "t"), b.p_olin_idx)

    # ─── VAR. ─── v_online_integer / v_startup_integer / v_shutdown_integer ──
    # ─────────────────────────────────────────────────────────────────────────
    # v_online_integer[p,d,t] ∈ ℤ≥0, v_startup_integer[p,d,t], v_shutdown_integer[p,d,t] ≥ 0
    #   ∀ (p,d,t) ∈ p_online_dt, p ∈ process_online_integer
    # As the linear family, but units online are integer; startups and
    # shutdowns stay continuous.
    if b.has_online_int:
        b.v_online_integer = m.add_var("v_online_integer", ("p", "d", "t"), b.p_oint_idx, integer=True)
        b.v_startup_integer = m.add_var("v_startup_integer", ("p", "d", "t"), b.p_oint_idx)
        b.v_shutdown_integer = m.add_var("v_shutdown_integer", ("p", "d", "t"), b.p_oint_idx)


def uc_classes(b: "_Build") -> list:
    """``[(kind, p_idx, online_set, v_online, v_startup, v_shutdown), …]`` for the UC classes present,
    linear first."""
    d = b.d
    classes = []
    if b.has_online_lin:
        classes.append(("linear", b.p_olin_idx, d.process_online_linear,
                        b.v_online_linear, b.v_startup_linear, b.v_shutdown_linear))
    if b.has_online_int:
        classes.append(("integer", b.p_oint_idx, d.process_online_integer,
                        b.v_online_integer, b.v_startup_integer, b.v_shutdown_integer))
    return classes


def add_constraints(b: "_Build") -> None:
    """Online units, startups, shutdowns, minimum up/down times and the parent-commitment pin.

    All rows of the linear class are emitted before those of the integer class; online_fix_eq
    follows both.
    """
    if not b.has_online:
        return
    m, d = b.m, b.d
    v_flow = b.v_flow

    for kind, p_idx, online_set, v_online, v_startup, v_shutdown in uc_classes(b):
        sfx = f"_{kind}"

        # Units invested before / divested before d, restricted to this class:
        #   invest_term = −Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
        #   divest_term = +Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
        # They sit on the LHS of the rows whose capacity is existing + invested − divested.
        invest_term = None
        divest_term = None
        if b.v_invest_p is not None and d.edd_invest_set is not None:
            v_inv_at_uc = _var_view(b.v_invest_p, f"__at_invest_uc{sfx}", {"d": "d_invest"}, to_axis=True)
            edd_inv_p_uc = _entity_rows(d.edd_invest_set, "e", online_set, "p")
            if edd_inv_p_uc.height > 0:
                invest_term = -Sum(Where(v_inv_at_uc, edd_inv_p_uc), over=("d_invest",))
        if b.v_divest_p is not None and d.edd_divest_active is not None:
            v_div_at_uc = _var_view(b.v_divest_p, f"__at_divest_uc{sfx}", {"d": "d_divest"}, to_axis=True)
            # online_set and edd_divest_active carry different p vocabularies: cast before the join.
            p_on = online_set.select(pl.col("p").cast(d.edd_divest_active.schema["p"], strict=False)).unique()
            edd_div_p_uc = d.edd_divest_active.join(p_on, on="p", how="semi")
            if edd_div_p_uc.height > 0:
                divest_term = Sum(Where(v_div_at_uc, edd_div_p_uc), over=("d_divest",))

        def _lhs_with_invdiv(var, key):
            lhs: dict = {key: var}
            if invest_term is not None:
                lhs["invest_neg"] = invest_term
            if divest_term is not None:
                lhs["divest"] = divest_term
            return lhs

        # ─── EQ. ─── maxOnline / maxStartup / maxShutdown_{linear,integer} ───
        # ─────────────────────────────────────────────────────────────────────
        # v_online[p,d,t] − Σ v_invest_p[p,d′] + Σ v_divest_p[p,d′]  ≤ existing_count[p,d]
        # v_startup[p,d,t]  (same invest/divest terms)                 ≤ existing_count[p,d]
        # v_shutdown[p,d,t] (same invest/divest terms)                 ≤ existing_count[p,d]
        #   ∀ (p,d,t) ∈ p_idx
        # Units online, starting up or shutting down cannot exceed the units built.
        m.add_cstr(f"maxOnline{sfx}",   over=p_idx, sense="<=",
                   lhs_terms=_lhs_with_invdiv(v_online, "online"),
                   rhs_terms={"max_units": d.p_process_existing_count})
        m.add_cstr(f"maxStartup{sfx}",  over=p_idx, sense="<=",
                   lhs_terms=_lhs_with_invdiv(v_startup, "startup"),
                   rhs_terms={"max_units": d.p_process_existing_count})
        m.add_cstr(f"maxShutdown{sfx}", over=p_idx, sense="<=",
                   lhs_terms=_lhs_with_invdiv(v_shutdown, "shutdown"),
                   rhs_terms={"max_units": d.p_process_existing_count})

        # ─── EQ. ─── online__startup / online__shutdown_{linear,integer} ─────
        # ─────────────────────────────────────────────────────────────────────
        # v_startup[p,d,t] + v_online[p,d′,t′] (+ carry[p,d,t])  ≥ v_online[p,d,t]
        # v_shutdown[p,d,t] + v_online[p,d,t]  ≥ v_online[p,d′,t′] (+ carry[p,d,t])
        #   ∀ (p,d,t) ∈ p_idx,   (d′,t′) = t_previous_within_solve of (d,t)
        # Startups and shutdowns count the change in units online.
        # The predecessor of the first step of a solve wraps cyclically to its last step.  On a
        # continuation roll of a rolling solve (d.p_roll_continue_online is set) that wrap would
        # point at the discarded look-ahead tail: the first step's lag is dropped instead and the
        # committed v_online carried from the previous roll (carry, p_roll_continue_online at the
        # first (d,t) only) is the predecessor, so a unit that stays committed is not charged a
        # new startup.  The gate is the presence of the carry: p_nested_solve_first is None on
        # every roll of the standalone rolling-window path.
        rco = d.p_roll_continue_online
        if rco is not None and d.dtttdt is not None and d.dtttdt.height > 0:
            dtttdt_cont = d.dtttdt.sort("d", "t").slice(1)
            first_dt = d.dtttdt.sort("d", "t").select("d", "t").head(1)
            # The carried p is String: cast to the online_set p Enum before the join.  A process
            # has one startup method, so the classes never share a p.
            carry_al = rco.frame.with_columns(pl.col("p").cast(online_set.schema["p"], strict=False))
            p_carry_rows = (online_set.select("p").unique()
                            .join(carry_al, on="p", how="inner")
                            .join(first_dt, how="cross")
                            .select("p", "d", "t", "value"))
            v_online_lag = Lag(v_online, dtttdt_cont, "t", "t_previous_within_solve")
            su_lhs = {"startup": v_startup, "online_prev": v_online_lag}
            sd_rhs = {"online_prev": v_online_lag}
            if p_carry_rows.height > 0:
                # Sparse Param at the boundary row only; Param + Param outer-joins with 0-fill.
                su_lhs["roll_continue_online"] = Param(("p", "d", "t"), p_carry_rows)
                sd_rhs["roll_continue_online"] = Param(("p", "d", "t"), p_carry_rows)
            m.add_cstr(f"online__startup{sfx}",  over=p_idx, sense=">=",
                       lhs_terms=su_lhs,
                       rhs_terms={"online_now":  v_online})
            m.add_cstr(f"online__shutdown{sfx}", over=p_idx, sense=">=",
                       lhs_terms={"shutdown":    v_shutdown,
                                  "online_now":  v_online},
                       rhs_terms=sd_rhs)
        else:
            v_online_lag = Lag(v_online, d.dtttdt, "t", "t_previous_within_solve")
            m.add_cstr(f"online__startup{sfx}",  over=p_idx, sense=">=",
                       lhs_terms={"startup":     v_startup,
                                  "online_prev": v_online_lag},
                       rhs_terms={"online_now":  v_online})
            m.add_cstr(f"online__shutdown{sfx}", over=p_idx, sense=">=",
                       lhs_terms={"shutdown":    v_shutdown,
                                  "online_now":  v_online},
                       rhs_terms={"online_prev": v_online_lag})

        # ─── EQ. ─── maxFlow_online_{linear,integer} ─────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_flow[p,s,k,d,t] + Σ_r v_reserve_up[p,r,k,d,t] + Σ_r v_reserve_down[p,r,s,d,t] · φ[p,s,k,d,t]
        #   ≤  v_online[p,d,t] · availability[p,d,t] · capacity_max_coeff[p,s,k]
        #   ∀ (p,s,k) capped arc of an online process, (d,t) ∈ dt;   φ as in maxFlow, units only
        # A unit's flow and the reserve that raises it — upward reserve at the arc's sink and
        # "consume more" downward reserve at its source — are limited by the units online
        # (spinning reserve: an offline unit holds neither).  The reserve terms are maxFlow's.
        # Only capped arcs (direct arcs and indirect
        # output arcs, see _capped_arcs): indirect input arcs are tied to the outputs by
        # conversion_indirect, and conversion_flow_coeff = 0 arcs are uncapped.  Availability and
        # capacity_max_coeff are densified to their 1.0 defaults so an unauthored unit keeps
        # its row.
        pss_online = _capped_arcs(d, d.process_source_sink.join(online_set, on="p", how="inner"))
        if pss_online.height > 0:
            over_pss_online = pss_online.join(d.dt, how="cross")
            if d.p_process_availability is not None:
                online_rhs = v_online * _availability_factor(d, pss_online)
            else:
                online_rhs = v_online
            if d.p_arc_max_cap_coef is not None:
                online_rhs = online_rhs * _coef_factor(pss_online, d.p_arc_max_cap_coef, ["p", "source", "sink"])
            online_lhs: dict = {"flow": v_flow}
            if b.reserve_up_to_sink is not None:
                online_lhs["reserve_up"] = b.reserve_up_to_sink
            online_lhs.update(b.reserve_down_at_source)
            m.add_cstr(f"maxFlow_online{sfx}",
                       over=over_pss_online, sense="<=",
                       lhs_terms=online_lhs,
                       rhs_terms={"online": online_rhs})

        # ─── EQ. ─── maxInputFlow_online_{linear,integer} ────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_flow[p,s,p,d,t] + Σ_r v_reserve_down[p,r,s,d,t]
        #   ≤ v_online[p,d,t] · availability[p,d,t] · input_cap[p,s,p,d,t]
        #   ∀ (p,s,p) ∈ reserve_down_input_arcs, an input arc of an online indirect unit, (d,t) ∈ dt
        #   input_cap = p_indirect_input_cap (input_share_max × full-load input per unit of capacity)
        # The online analogue of maxFlow on an indirect unit's input arc, built only where the arc
        # carries downward reserve at its source: consuming more needs online capacity.  The other
        # inputs and the outputs are not adjusted (the per-arc simplification of the input-side
        # floors).
        _add_max_input_flow_online(b, sfx, online_set, v_online)

        # ─── EQ. ─── maxOutputSum_online_{linear,integer} ────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # Σ_k v_flow[p,p,k,d,t] + Σ_k Σ_r v_reserve_up[p,r,k,d,t]  ≤  v_online[p,d,t] · availability[p,d,t]
        #   ∀ p ∈ process_indirect online with an output arc k, (p,d,t) ∈ p_idx
        # The online analogue of maxOutputSum: the sum of an indirect unit's outputs and of the
        # upward reserve it holds at those outputs is limited by the units online.  No
        # capacity_max_coeff term; availability defaults to 1.0.
        sum_arcs = _indirect_sum_arcs(d)
        if sum_arcs is not None:
            sum_arcs_on = sum_arcs.join(online_set.select("p").unique(), on="p", how="semi")
            if sum_arcs_on.height > 0:
                over_sum_on = (sum_arcs_on.select("p").unique(maintain_order=True)
                               .join(d.dt, how="cross")
                               .join(p_idx.select("p", "d", "t"), on=["p", "d", "t"], how="semi"))
                if over_sum_on.height > 0:
                    if d.p_process_availability is not None:
                        sum_online_rhs = v_online * _availability_factor(d, sum_arcs_on)
                    else:
                        sum_online_rhs = v_online
                    sum_lhs: dict = {"flow": Sum(Where(v_flow, sum_arcs_on), over=("source", "sink"))}
                    if b.reserve_up_sink_parts is not None:
                        reserve_at_sink, up_sink = b.reserve_up_sink_parts
                        up_sink = up_sink.join(sum_arcs_on.select("p", "sink").unique(), on=["p", "sink"],
                                               how="semi")
                        if up_sink.height > 0:
                            sum_lhs["reserve_up"] = Sum(Where(reserve_at_sink, up_sink), over=("r", "ud", "sink"))
                    m.add_cstr(f"maxOutputSum_online{sfx}",
                               over=over_sum_on, sense="<=",
                               lhs_terms=sum_lhs,
                               rhs_terms={"online": sum_online_rhs})

        # ─── EQ. ─── minimum_uptime_{linear,integer} ─────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_online[p,d,t]  ≥  Σ_{(p,d′,t′,d,t) ∈ uptime_lookback} v_startup[p,d′,t′] + carry_up[p,d,t]
        #   ∀ (p,d,t) ∈ pdt_uptime_set ∩ this class
        # A unit started within the minimum uptime stays online.  carry_up is the sum of the
        # previous roll's startups that fall in this roll's lookback window (d.uptime_carry,
        # 0-filled; absent outside continuation rolls).
        if (d.pdt_uptime_set is not None and d.pdt_uptime_set.height > 0
                and d.uptime_lookback is not None and d.uptime_lookback.height > 0):
            up_idx = d.pdt_uptime_set.join(online_set, on="p", how="inner")
            if up_idx.height > 0:
                # Lookback rows whose v_startup exists and whose process is in this class.
                p_idx_back = p_idx.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"})
                lkb = (d.uptime_lookback.join(online_set, on="p", how="inner")
                       .join(p_idx_back, on=["p", "d_back", "t_back"], how="inner")
                       .join(up_idx, on=["p", "d", "t"], how="inner"))
                if lkb.height > 0:
                    v_startup_at = _var_view(v_startup, f"__lookback_up{sfx}",
                                             {"d": "d_back", "t": "t_back"}, to_axis=True)
                    up_rhs = {"startup_sum": Sum(Where(v_startup_at, lkb), over=("d_back", "t_back"))}
                    up_carry = _online_mintime_carry_term(d.uptime_carry, up_idx)
                    if up_carry is not None:
                        up_rhs["carry_up"] = up_carry
                    m.add_cstr(f"minimum_uptime{sfx}",
                               over=up_idx, sense=">=",
                               lhs_terms={"online":      v_online},
                               rhs_terms=up_rhs)

        # ─── EQ. ─── minimum_downtime_{linear,integer} ───────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_online[p,d,t] + Σ_{(p,d′,t′,d,t) ∈ downtime_lookback} v_shutdown[p,d′,t′] + carry_dn[p,d,t]
        #   − Σ v_invest_p[p,d″] + Σ v_divest_p[p,d″]  ≤  existing_count[p,d]
        #   ∀ (p,d,t) ∈ pdt_downtime_set ∩ this class
        # A unit shut down within the minimum downtime stays offline: existing − v_online ≥
        # Σ v_shutdown, with the invest/divest terms of maxOnline (without them an invest-only
        # unit with existing = 0 could never be online).  carry_dn is the previous roll's
        # shutdowns in this roll's lookback window (d.downtime_carry, 0-filled).
        if (d.pdt_downtime_set is not None and d.pdt_downtime_set.height > 0
                and d.downtime_lookback is not None and d.downtime_lookback.height > 0):
            dn_idx = d.pdt_downtime_set.join(online_set, on="p", how="inner")
            if dn_idx.height > 0:
                p_idx_back = p_idx.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"})
                lkb = (d.downtime_lookback.join(online_set, on="p", how="inner")
                       .join(p_idx_back, on=["p", "d_back", "t_back"], how="inner")
                       .join(dn_idx, on=["p", "d", "t"], how="inner"))
                if lkb.height > 0:
                    v_shutdown_at = _var_view(v_shutdown, f"__lookback_dn{sfx}",
                                              {"d": "d_back", "t": "t_back"}, to_axis=True)
                    lhs_dn: dict = {"online":       v_online,
                                    "shutdown_sum": Sum(Where(v_shutdown_at, lkb), over=("d_back", "t_back"))}
                    if invest_term is not None:
                        lhs_dn["invest_neg"] = invest_term
                    if divest_term is not None:
                        lhs_dn["divest"] = divest_term
                    dn_carry = _online_mintime_carry_term(d.downtime_carry, dn_idx)
                    if dn_carry is not None:
                        lhs_dn["carry_dn"] = dn_carry
                    m.add_cstr(f"minimum_downtime{sfx}",
                               over=dn_idx, sense="<=",
                               lhs_terms=lhs_dn,
                               rhs_terms={"existing":     d.p_process_existing_count})

    # ─── EQ. ─── online_fix_eq_{linear,integer} ──────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_online[p,d,t] = p_fix_online[p,d,t]   ∀ (p,d,t) ∈ p_idx with a p_fix_online value
    # In a separate-dispatch solve, pins units online to the commitment of the parent solve.
    # Only declared v_online cells are pinned.  p_fix_online is Utf8-keyed: it is joined to
    # the native-typed index through Utf8 copies of (p, d, t).
    if d.p_fix_online is not None and d.p_fix_online.frame.height > 0:
        fix_u = d.p_fix_online.frame.select(
            pl.col("p").cast(pl.Utf8).alias("_pk"),
            pl.col("d").cast(pl.Utf8).alias("_dk"),
            pl.col("t").cast(pl.Utf8).alias("_tk"),
            "value")
        for kind, p_idx, _online_set, v_online, _v_startup, _v_shutdown in uc_classes(b):
            pin = (p_idx.select("p", "d", "t").unique()
                   .with_columns(_pk=pl.col("p").cast(pl.Utf8),
                                 _dk=pl.col("d").cast(pl.Utf8),
                                 _tk=pl.col("t").cast(pl.Utf8))
                   .join(fix_u, on=["_pk", "_dk", "_tk"], how="inner")
                   .select("p", "d", "t", "value"))
            if pin.height == 0:
                continue
            over = pin.select("p", "d", "t").unique()
            m.add_cstr(
                f"online_fix_eq_{kind}",
                over=over,
                sense="==",
                lhs_terms={"online": Where(v_online, over)},
                rhs_terms={"target": Param(("p", "d", "t"), pin)},
            )


def _add_max_input_flow_online(b: "_Build", sfx: str, online_set: "pl.DataFrame", v_online) -> None:
    """maxInputFlow_online rows of one UC class (see the EQ. header in :func:`add_constraints`)."""
    d = b.d
    arcs = b.reserve_down_input_arcs
    in_cap = getattr(d, "p_indirect_input_cap", None)
    indirect = getattr(d, "process_indirect", None)
    if (arcs is None or not b.reserve_down_at_source or in_cap is None
            or indirect is None or indirect.height == 0):
        return
    keys = ["p", "source", "sink"]
    sel = (_utf8_keys(arcs.lazy(), keys)
           .filter(pl.col("_sink") == pl.col("_p"))
           .join(_utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi")
           .join(_utf8_keys(online_set.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi"))
    # the arc rows of v_flow (process_source_sink dtypes), restricted to sel
    pss = d.process_source_sink.select(keys).unique()
    in_arcs = (_utf8_keys(pss.lazy(), keys)
               .join(sel.select("_p", "_source", "_sink"), on=["_p", "_source", "_sink"], how="semi")
               .select(keys).collect())
    if in_arcs.height == 0:
        return
    over = in_arcs.join(d.dt, how="cross")
    cap = (_utf8_keys(over.lazy(), keys + ["d", "t"])
           .join(_utf8_keys(in_cap.frame.lazy(), keys + ["d", "t"])
                 .select("_p", "_source", "_sink", "_d", "_t", "value"),
                 on=["_p", "_source", "_sink", "_d", "_t"], how="left")
           .select(*keys, "d", "t", "value").collect())
    missing = cap.filter(pl.col("value").is_null())
    if missing.height > 0:
        raise ValueError(
            "p_indirect_input_cap has no value for indirect input arc(s) "
            f"{missing.select(keys).unique().head(10).to_dicts()} — maxInputFlow_online would be dropped.")
    rhs = v_online * Param(("p", "source", "sink", "d", "t"), cap)
    if d.p_process_availability is not None:
        rhs = rhs * _availability_factor(d, in_arcs)
    lhs: dict = {"flow": b.v_flow}
    lhs.update(b.reserve_down_at_source)
    b.m.add_cstr(f"maxInputFlow_online{sfx}", over=over, sense="<=", lhs_terms=lhs, rhs_terms={"online": rhs})


def _online_mintime_carry_term(carry: "pl.DataFrame | None", idx: "pl.DataFrame") -> "Param | None":
    """The cross-roll minimum up/down-time carry as a ``Param((p, d, t))`` over ``idx``.

    ``carry`` is ``(p, d, t, value)`` with Utf8 axes (``online_mintime_carry_from_source``); its
    axes are cast to ``idx``'s Enum dtypes before the join.  The result is DENSE over ``idx``
    (0-filled): a term on a restricted domain would misalign the ``(p, d)`` ``existing`` RHS of
    ``minimum_downtime`` and make rows infeasible.  The left join also restricts the carry to
    the processes of ``idx``.  ``None`` when there is no carry (first roll, non-rolling solve).
    """
    if carry is None or carry.height == 0:
        return None
    carry_cast = carry.with_columns(
        pl.col("p").cast(idx.schema["p"], strict=False),
        pl.col("d").cast(idx.schema["d"], strict=False),
        pl.col("t").cast(idx.schema["t"], strict=False))
    dense = (idx.select("p", "d", "t")
             .join(carry_cast, on=["p", "d", "t"], how="left")
             .with_columns(pl.col("value").fill_null(0.0)))
    if dense.height == 0:
        return None
    return Param(("p", "d", "t"), dense)
