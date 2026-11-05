"""Unit commitment: units (and connections) online, startups, shutdowns and minimum up/down times.

Owns the variables v_online_*, v_startup_*, v_shutdown_* and the equations maxOnline, maxStartup,
maxShutdown, online_change, maxFlow_online, maxFlow_online_shutdown,
maxFlow_online_shutdown_branch, maxFlow_back_online, maxOutputSum_online, maxInputFlow_online,
maxInputFlow_online_shutdown, maxInputFlow_online_shutdown_branch, minimum_uptime, minimum_downtime and
online_fix_eq.  The ``_branch`` rows exist only at a branch point (several mid-horizon branches continuing from
the same trunk step); every other fixture never emits them.  Each exists once per UC
class present: ``linear`` (continuous v_online) and ``integer`` (integer
v_online), suffixed ``_linear`` / ``_integer``.  A process is online by its startup method (units: also by
``minimum_time_method``); a connection with a startup method has online capacity in service covering both flow
directions, and no minimum load, minimum up / down times or startup reduction (connections have no such
parameters).

A coarse process (``processCoarse``) has one units-online value per coarse step (an aliased level) and one
start-up / shut-down per coarse step at its start (events registered at the step's first fine step); its online
capacity and minimum-load rows are averaged over the step (``_coarse_flows.add_cstr``).  With ramp limits, its start-up /
shut-down capacity reduction uses the coarse step (``_cap_reduction``: the mean share over the whole step of a
unit that starts at the step's start or stops at its end) and its units online ramp between coarse steps
(``core._eq_ramp``).
"""

from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.model import _coarse_flows
from flextool.engine.model._common import (
    _availability_factor, _capped_arcs, _coef_factor, _drop_uncapped, _entity_rows, _has_rows, _indirect_input_cap,
    _indirect_sum_arcs, _utf8_keys, _var_view,
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
    # Units online, and units started / shut down: the change of units online, online_change
    # (count of unitsize), continuous.
    # ─── VAR. ─── v_online / v_startup / v_shutdown (coarse processes) ───────
    # on[p,d,b], su[p,d,b], sd[p,d,b]   ∀ p ∈ processCoarse ∩ process_online, (d,b) the coarse steps of p
    #   (b = its first fine step b_first)
    # Units online once per coarse step, held over its fine steps (a level: b.v_online_* is indexed on every fine
    # step, _coarse_flows.alias); the units started / shut down at the step's start (events: registered at b only,
    # not aliased, so a sum over t counts each once and a fine row of an interior step has no event term).
    if b.has_online_lin:
        b.v_online_linear = _coarse_flows.alias(
            b, m.add_var("v_online_linear", ("p", "d", "t"), b.p_olin_rows), b.p_olin_idx)
        b.v_startup_linear = m.add_var("v_startup_linear", ("p", "d", "t"), b.p_olin_rows)
        b.v_shutdown_linear = m.add_var("v_shutdown_linear", ("p", "d", "t"), b.p_olin_rows)

    # ─── VAR. ─── v_online_integer / v_startup_integer / v_shutdown_integer ──
    # ─────────────────────────────────────────────────────────────────────────
    # v_online_integer[p,d,t] ∈ ℤ≥0, v_startup_integer[p,d,t], v_shutdown_integer[p,d,t] ≥ 0
    #   ∀ (p,d,t) ∈ p_online_dt, p ∈ process_online_integer
    # As the linear family, but units online are integer; startups and
    # shutdowns stay continuous.
    if b.has_online_int:
        b.v_online_integer = _coarse_flows.alias(
            b, m.add_var("v_online_integer", ("p", "d", "t"), b.p_oint_rows, integer=True), b.p_oint_idx)
        b.v_startup_integer = m.add_var("v_startup_integer", ("p", "d", "t"), b.p_oint_rows)
        b.v_shutdown_integer = m.add_var("v_shutdown_integer", ("p", "d", "t"), b.p_oint_rows)


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


def online_rows(b: "_Build", kind: str) -> pl.DataFrame:
    """``(p, d, t)`` of the online rows of UC class ``kind``: every step of a fine process, the first fine step of
    each coarse step of a coarse process (the registered online / startup / shutdown columns)."""
    return b.p_olin_rows if kind == "linear" else b.p_oint_rows


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
        rows = online_rows(b, kind)

        # Units invested before / divested before d, restricted to this class:
        #   invest_term = −Σ_{(p,d′,d) ∈ edd_invest_set} v_invest_p[p,d′]
        #   divest_term = +Σ_{(p,d′,d) ∈ edd_divest_active} v_divest_p[p,d′]
        # They sit on the LHS of the rows whose capacity is existing + invested − divested.
        invest_term = None
        divest_term = None
        if b.v_invest_p is not None:
            v_inv_at_uc = _var_view(b.v_invest_p, f"__at_invest_uc{sfx}", {"d": "d_invest"}, to_axis=True)
            edd_inv_p_uc = _entity_rows(d.edd_invest_set, "e", online_set, "p")
            if edd_inv_p_uc.height > 0:
                invest_term = -Sum(Where(v_inv_at_uc, edd_inv_p_uc), over=("d_invest",))
        if b.v_divest_p is not None:
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

        # ─── EQ. ─── maxOnline_{linear,integer} ──────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_online[p,d,t] − Σ v_invest_p[p,d′] + Σ v_divest_p[p,d′]  ≤ existing_count[p,d]
        #   ∀ (p,d,t) ∈ p_idx
        # Units online cannot exceed the units built.
        # A retirement v_divest[e,d′] counts in d iff (e,d′,d) ∈ edd_divest_active (ndd_divest_active):
        #   d′ ≤ d < EOL[e], the end of life of e's pre-existing capacity (none for reinvest_automatic).
        m.add_cstr(f"maxOnline{sfx}",   over=rows, sense="<=",
                   lhs_terms=_lhs_with_invdiv(v_online, "online"),
                   rhs_terms={"max_units": d.p_process_existing_count})

        # The online predecessor v_online_prev of every (p,d,t) ∈ p_idx, one of three disjoint pieces:
        # v_online_prev[p,d,t] =
        #   v_online[p,d′,t′]         (d,t,d′,t′) ∈ dt_ramp_link                adjacent predecessor (also a continuing
        #                                                                         period, a mid-horizon branch's first step)
        #   p_roll_continue_online[p] (d,t) ∈ dt_online_carry                  the previous solve's realized commitment
        #                                                                         (dt_online_carry = dt_ramp_carry's steps)
        #   v_online[p,d,t″]          (d,t,d,t″) ∈ dt_online_wrap, (d,t) ∉ dt_online_carry
        #                                                                         stretch start: the last step of its stretch
        # A stretch is a run of steps of one period d along dt_ramp_link; it starts at the first step of a solve or of
        # a period that does not continue the previous one, after a gap in the timeline and at a representative-block
        # start.  No link across a gap, a representative-block start or a new year (as ramps and min up/down).  The
        # successor side does not wrap: a stretch end has no successor (_successor_steps).
        # A coarse process (processCoarse) has these rows, and maxOnline / maxStartup / maxShutdown, only at the first
        # fine step b of each of its coarse steps (online_rows); v_online_prev reads the alias of v_online, i.e. the
        # coarse step b⁻ that holds the fine predecessor of b (or the stretch's last coarse step on a wrap; a carried
        # first step is always a coarse step's first step, as coarse steps break at every roll and stretch start).
        # Units online, started and shut down are then decided once per coarse step, the start / stop at its start;
        # the rows of the interior steps would read 0 = 0 or be implied.
        rco = d.p_roll_continue_online
        d_dtype, t_dtype = p_idx.schema["d"], p_idx.schema["t"]

        carry_rows = None
        if d.dt_online_carry is not None and d.dt_online_carry.height > 0 and rco is not None:
            carry_dt = d.dt_online_carry.select(pl.col("d").cast(d_dtype, strict=False),
                                                pl.col("t").cast(t_dtype, strict=False)).unique()
            # The carried p is String: cast to the online_set p Enum before the join.  A process has one
            # startup method, so the classes never share a p.
            carry_al = rco.frame.with_columns(pl.col("p").cast(online_set.schema["p"], strict=False))
            p_carry_rows = (online_set.select("p").unique()
                            .join(carry_al, on="p", how="inner")
                            .join(carry_dt, how="cross")
                            .join(rows.select("p", "d", "t"), on=["p", "d", "t"], how="inner")
                            .select("p", "d", "t", "value"))
            if p_carry_rows.height > 0:
                carry_rows = p_carry_rows

        prev_links = []
        if d.dt_ramp_link is not None and d.dt_ramp_link.height > 0:
            prev_links.append(d.dt_ramp_link)
        if d.dt_online_wrap is not None and d.dt_online_wrap.height > 0:
            wrap = d.dt_online_wrap
            if carry_rows is not None:
                wrap = wrap.join(carry_rows.select("d", "t").unique(), on=["d", "t"], how="anti")
            prev_links.append(wrap)
        link_native = None
        if prev_links:
            link_cast = (pl.concat([f.select("d", "t", "d_previous", "t_previous") for f in prev_links])
                         .pipe(rename_to_axis,
                               {"d": "d", "t": "t", "d_previous": "d_previous", "t_previous": "t_previous"}))
            link_rows = (online_set.select("p").unique().join(link_cast, how="cross")
                        .join(rows.select("p", "d", "t"), on=["p", "d", "t"], how="inner"))
            if link_rows.height > 0:
                link_native = link_rows

        online_prev_terms: dict = {}
        if link_native is not None:
            v_online_prev_link = _var_view(v_online, f"__online_prev_link{sfx}",
                                           {"d": "d_previous", "t": "t_previous"}, to_axis=True)
            online_prev_terms["online_prev_link"] = Sum(Where(v_online_prev_link, link_native),
                                                        over=("d_previous", "t_previous"))
        if carry_rows is not None:
            # Sparse Param at the carried boundary row(s) only; Param + Param outer-joins with 0-fill.
            online_prev_terms["roll_continue_online"] = Param(("p", "d", "t"), carry_rows)

        # ─── EQ. ─── online_change_{linear,integer} ──────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_online[p,d,t] − v_online_prev[p,d,t] − v_startup[p,d,t] + v_shutdown[p,d,t]  =  0
        #   ∀ (p,d,t) ∈ p_idx;   v_online_prev as derived above (three pieces)
        # The change in units online is the units started minus the units shut down (Backbone's
        # q_startshut).  With maxStartup / maxShutdown below, a startup without a unit coming online, or a
        # shutdown without one going offline, is impossible; a fleet (unitsize < built) can still start and
        # stop units in the same step (both cost nothing extra on the ramp rows of a single-output arc, part
        # C) — a single integer unit cannot (maxStartup / maxShutdown forbid it there).
        m.add_cstr(f"online_change{sfx}", over=rows, sense="==",
                   lhs_terms={"online": v_online, "shutdown": v_shutdown},
                   rhs_terms={"startup": v_startup, **online_prev_terms})

        # ─── EQ. ─── maxStartup_{linear,integer} ─────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_startup[p,d,t] − v_online[p,d,t]  ≤  0                                 ∀ (p,d,t) ∈ p_idx
        # Units started in a step are online in it.
        m.add_cstr(f"maxStartup{sfx}", over=rows, sense="<=",
                   lhs_terms={"startup": v_startup, "online_neg": -v_online})

        # ─── EQ. ─── maxShutdown_{linear,integer} ─────────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_shutdown[p,d,t] + v_online[p,d,t] − Σ v_invest_p[p,d″] + Σ v_divest_p[p,d″]  ≤  existing_count[p,d]
        #   ∀ (p,d,t) ∈ p_idx whose online predecessor is v_online of the SAME period d (a dt_ramp_link with
        #   d_previous = d, or a dt_online_wrap; the same-period predecessor keeps the shutdown bound)
        # Units shut down in a step are offline in it.  Not built where the predecessor is in another period
        # or carried from an earlier roll (capacity may differ there — a unit online at the end of the
        # previous period / roll must still be able to "shut down" even if this step's capacity is lower);
        # sd ≤ v_online_prev follows there from online_change + maxStartup instead (su ≤ on_t ⇒
        # sd = su − on_t + on_prev ≤ on_prev).
        # A retirement v_divest[e,d′] counts in d iff (e,d′,d) ∈ edd_divest_active (ndd_divest_active):
        #   d′ ≤ d < EOL[e], the end of life of e's pre-existing capacity (none for reinvest_automatic).
        exclude_shutdown = None
        if link_native is not None:
            cross_period_link = link_native.filter(pl.col("d") != pl.col("d_previous")).select("p", "d", "t")
            if cross_period_link.height > 0:
                exclude_shutdown = cross_period_link
        if carry_rows is not None:
            carry_pdt = carry_rows.select("p", "d", "t")
            exclude_shutdown = (carry_pdt if exclude_shutdown is None
                                else pl.concat([exclude_shutdown, carry_pdt]).unique())
        max_shutdown_over = (rows if exclude_shutdown is None
                             else rows.join(exclude_shutdown, on=["p", "d", "t"], how="anti"))
        if max_shutdown_over.height > 0:
            sd_lhs = _lhs_with_invdiv(v_shutdown, "shutdown")
            sd_lhs["online"] = v_online
            m.add_cstr(f"maxShutdown{sfx}", over=max_shutdown_over, sense="<=",
                       lhs_terms=sd_lhs,
                       rhs_terms={"max_units": d.p_process_existing_count})

        # ─── EQ. ─── maxFlow_online_{linear,integer} ─────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_flow[p,s,k,d,t] + Σ_r v_reserve_up[p,r,k,d,t] + Σ_r v_reserve_down[p,r,s,d,t] · φ[p,s,k,d,t]
        #   ≤  κ[p,s,k] · ( v_online[p,d,t] · availability[p,d,t] − red_su[p,s,k,d,t] · v_startup[p,d,t]
        #                   − max(0, red_sd − red_su)[p,s,k,d,t] · v_shutdown[p,d,t⁺] )
        #   ∀ (p,s,k) capped arc of an online process, (d,t) ∈ dt;   φ as in maxFlow, units only;
        #   κ = capacity_max_coeff[p,s,k]
        # ─── EQ. ─── maxFlow_online_shutdown_{linear,integer} ────────────────
        # (maxFlow_online LHS)
        #   ≤  κ · ( v_online · availability − red_sd · v_shutdown[p,d,t⁺]
        #            − max(0, red_su − red_sd) · v_startup[p,d,t] )
        #   ∀ rows of maxFlow_online with red_sd > 0 and a UNIQUE successor t⁺
        # ─── EQ. ─── maxFlow_online_shutdown_branch_{linear,integer} ─────────
        # (maxFlow_online LHS)
        #   ≤  κ · ( v_online · availability − red_sd · v_shutdown[p,d⁺,t⁺]
        #            − max(0, red_su − red_sd) · v_startup[p,d,t] )
        #   ∀ (row of maxFlow_online, successor (d⁺,t⁺) of (d,t)) with red_sd > 0, at a (d,t) with SEVERAL
        #     successors (a branch point: several mid-horizon branches continuing from the same trunk step)
        # red_su = max(0, availability − SU),
        #   SU = min(1, min_load[p,d,t] + ½ · ramp_speed_up · 60 · step_duration[d,t])
        #   over the sides of the arc with an up ramp limit (process_source_sink_ramp_limit_{sink,source}_up; the
        #   smaller SU when both); red_sd likewise with ramp_speed_down at the pre-shutdown step (d,t); 0 without a
        #   ramp limit.  (d⁺,t⁺) = a successor whose ramp predecessor (dt_ramp_link) is (d,t) — the same period
        #   (today's only case) or the next continuing period (#98).
        # A coarse process (processCoarse): the averaged twins maxFlow_online{sfx}_coarse,
        #   maxFlow_online_shutdown{sfx}_coarse and maxFlow_online_shutdown_branch{sfx}_coarse (_coarse_flows.add_cstr)
        #   with the reduction on the coarse grid: red_su[p,s,k,d,b] = max(0, ā_b − SU_b) at the start-ups of b and
        #   red_sd = max(0, ā_b − SD_b) at the shut-downs of the coarse successor b⁺ (the coarse step whose first
        #   step's ramp predecessor lies in b, _coarse_flows.coarse_links); ā_b, m̄l_b the Δt-weighted means over b,
        #   SU_b = m̄l_b + ½·ρ_b while ρ_b ≤ 1 − m̄l_b, else 1 − (1 − m̄l_b)²/(2·ρ_b), ρ_b = ramp_speed_up · 60 · Δ_b
        #   (the mean share over the whole step of a unit started at its start that ramps up to full load within
        #   it; SD_b with the down speed: a unit stopping at the end of b); κ its step mean.
        # A unit's flow and the reserve that raises it — upward reserve at the arc's sink and
        # "consume more" downward reserve at its source — are limited by the units online
        # (spinning reserve: an offline unit holds neither).  The reserve terms are maxFlow's (units only; an online
        # connection's reserve is in connection_reserve_*_online_*).
        # Units starting in step t reach at most SU of their capacity on average over the step (synchronised at
        # min load, then ramping as fast as they may: half a step's ramp on average; owner question Q-c14a-A),
        # units shutting down after step t at most SD in t.  Two rows, so capacity online for a single step gets
        # the smaller of the two limits instead of both reductions (exact for any mix of starts and stops; GMPL's
        # one row with both reductions forbids one-step runs).  At a branch point (several successors of one
        # (d,t): several mid-horizon branches continuing from the same trunk step) the shutdown cross-term cannot
        # be split across the single maxFlow_online row (one v_flow[t] variable, one row); it is dropped there
        # (maxFlow_online keeps only its startup term, a valid — looser — bound) and maxFlow_online_shutdown_branch
        # gives the tight, correctly-isolated bound per branch instead: summing the branches' v_shutdown into one
        # row would let several mutually-exclusive branches each buy back a whole unit of ramp room from the
        # trunk step (critique item 1).  Only capped arcs (direct arcs and indirect
        # output arcs, see _capped_arcs): indirect input arcs are tied to the outputs by
        # conversion_indirect, and conversion_flow_coeff = 0 arcs are uncapped.  Availability and
        # capacity_max_coeff are densified to their 1.0 defaults so an unauthored unit keeps
        # its row.  The reduction is per arc (a multi-output unit's maxOutputSum_online is not reduced).
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
            if b.reserve_up_to_sink_units is not None:
                online_lhs["reserve_up"] = b.reserve_up_to_sink_units
            online_lhs.update(b.reserve_down_at_source)
            online_rhs_terms: dict = {"online": online_rhs}
            shutdown_rows = None
            branch_rows = None
            fine_red: dict = {}
            coarse_red: dict = {}
            coarse_shutdown = None
            coarse_branch = None
            red, red_c = _cap_reduction(b, pss_online, input_side=False)
            if red is not None or red_c is not None:
                kappa = (_coef_factor(pss_online, d.p_arc_max_cap_coef, ["p", "source", "sink"]).frame
                         if d.p_arc_max_cap_coef is not None
                         else pss_online.select("p", "source", "sink").unique().with_columns(value=pl.lit(1.0)))
            if red is not None:
                row1, row2, over2, over3, row3 = _reduction_terms(red, kappa, v_startup, v_shutdown, sfx)
                fine_red = row1
                if over2.height > 0:
                    shutdown_rows = (over2, {"online": online_rhs, **row2})
                if over3.height > 0:
                    branch_rows = (over3, {"online": online_rhs, **row3})
            if red_c is not None:
                # A coarse process: the reduction on the coarse grid (start-ups at b, shut-downs at b⁺).
                coarse_red, row2, over2, over3, row3 = _reduction_terms(
                    red_c, kappa, _coarse_flows.at_step(b, v_startup), v_shutdown, f"_coarse{sfx}", tcol="b_first")
                if over2.height > 0:
                    coarse_shutdown = (over2, row2)
                if over3.height > 0:
                    coarse_branch = (over3, row3)
            _coarse_flows.add_cstr(b, f"maxFlow_online{sfx}",
                                   over=over_pss_online, sense="<=",
                                   lhs_terms=online_lhs,
                                   rhs_terms=online_rhs_terms, fine_rhs=fine_red, coarse_rhs=coarse_red)
            if shutdown_rows is not None:
                m.add_cstr(f"maxFlow_online_shutdown{sfx}",
                           over=shutdown_rows[0], sense="<=",
                           lhs_terms=online_lhs,
                           rhs_terms=shutdown_rows[1])
            if coarse_shutdown is not None:
                _coarse_flows.add_twin(b, f"maxFlow_online_shutdown{sfx}", over=coarse_shutdown[0], sense="<=",
                                       lhs_terms=online_lhs, rhs_terms={"online": online_rhs},
                                       coarse_rhs=coarse_shutdown[1])
            if branch_rows is not None:
                m.add_cstr(f"maxFlow_online_shutdown_branch{sfx}",
                           over=branch_rows[0], sense="<=",
                           lhs_terms=online_lhs,
                           rhs_terms=branch_rows[1])
            if coarse_branch is not None:
                _coarse_flows.add_twin(b, f"maxFlow_online_shutdown_branch{sfx}", over=coarse_branch[0], sense="<=",
                                       lhs_terms=online_lhs, rhs_terms={"online": online_rhs},
                                       coarse_rhs=coarse_branch[1])

        # ─── EQ. ─── maxFlow_back_online_{linear,integer} ────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_flow_back[p,s,k,d,t]  ≤  κ[p,s,k] · availability[p,d,t] · v_online[p,d,t]
        #   ∀ (p,s,k) ∈ twoway1var_arcs with p online, (d,t) ∈ dt;   κ = capacity_max_coeff (1.0 defaults)
        # The reverse flow of a one-variable connection is limited by the units online, as its v_flow is by
        # maxFlow_online: one online status covers both directions (the mirror of maxFlow_back).  No reserve
        # term (connection reserve has its own rows, connection_reserve_*).  DC power-flow lines never get here
        # (a startup_method on them raises in core._prepare).
        if b.v_flow_back is not None and _has_rows(b.twoway1var_arcs):
            keys = ["p", "source", "sink"]
            back_on = (_utf8_keys(b.twoway1var_arcs.lazy().select(keys), ["p"])
                       .join(_utf8_keys(online_set.lazy().select("p"), ["p"]).select("_p").unique(), on="_p",
                             how="semi")
                       .select(keys).collect())
            if back_on.height > 0:
                back_rhs = v_online
                if d.p_process_availability is not None:
                    back_rhs = back_rhs * _availability_factor(d, back_on)
                if d.p_arc_max_cap_coef is not None:
                    back_rhs = back_rhs * _coef_factor(back_on, d.p_arc_max_cap_coef, keys)
                _coarse_flows.add_cstr(b, f"maxFlow_back_online{sfx}", over=back_on.join(d.dt, how="cross"),
                                       sense="<=", lhs_terms={"flow_back": b.v_flow_back},
                                       rhs_terms={"online": back_rhs})

        # ─── EQ. ─── maxInputFlow_online_{linear,integer} ────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_flow[p,s,p,d,t] + Σ_r v_reserve_down[p,r,s,d,t]
        #   ≤ input_cap[p,s,p,d,t] · ( v_online[p,d,t] · x(availability)[p,d,t] − red_su · v_startup[p,d,t]
        #                              − max(0, red_sd − red_su) · v_shutdown[p,d,t⁺] )
        #   ∀ (p,s,p) input arc of an online indirect unit that is in reserve_down_input_arcs or has a source-side
        #     ramp limit with a reduction, (d,t) ∈ dt
        #   input_cap = p_indirect_input_cap (input_share_max × full-load input per unit of capacity)
        # ─── EQ. ─── maxInputFlow_online_shutdown_{linear,integer} ───────────
        # (maxInputFlow_online LHS) ≤ input_cap · ( v_online · x(availability) − red_sd · v_shutdown[p,d,t⁺]
        #                                           − max(0, red_su − red_sd) · v_startup[p,d,t] )
        #   ∀ rows of maxInputFlow_online with red_sd > 0 and a UNIQUE successor t⁺
        # ─── EQ. ─── maxInputFlow_online_shutdown_branch_{linear,integer} ────
        # (maxInputFlow_online LHS) ≤ input_cap · ( v_online · x(availability) − red_sd · v_shutdown[p,d⁺,t⁺]
        #                                           − max(0, red_su − red_sd) · v_startup[p,d,t] )
        #   ∀ (row of maxInputFlow_online, successor (d⁺,t⁺) of (d,t)) with red_sd > 0, at a (d,t) with several
        #     successors (a branch point) — split per successor exactly as maxFlow_online_shutdown_branch
        #     (critique item 1): never summed across mutually-exclusive branches.
        # x(y) = (slope · W(y) + section) / (slope · W(1) + section): the input drawn at output level y per unit of
        #   full-load input (W(y): the output-side fuel width filled up to y, as in p_indirect_input_cap; section
        #   only for min_load_efficiency units).  x(availability) replaces availability: the section is drawn per
        #   unit online, so availability × full-load input would cap the derated full output (x = availability
        #   without a section).
        # red_su = max(0, x(availability) − SU_in), SU_in = min(1, x(min_load) / input_share_max[p,s]
        #   + ½ · ramp_speed_up · 60 · step_duration) from the input node's ramp speeds (red_sd likewise), so a
        #   started unit can always draw what it needs at min load.  (d⁺,t⁺) = a successor whose ramp
        #   predecessor (dt_ramp_link) is (d,t) — the same period (today's only case) or the next continuing
        #   period (#98).
        # A coarse process: the averaged twins maxInputFlow_online{sfx}_coarse (+ _shutdown, _shutdown_branch), with
        #   input_cap and x(availability) as step means and the reduction on the coarse grid as in maxFlow_online
        #   (SU_in_b, SD_in_b from the step-mean x(min_load) / input_share_max and ρ_b).
        # The online analogue of maxFlow on an indirect unit's input arc: consuming more needs online capacity,
        # and a starting (stopping) unit draws at most its startup (pre-shutdown) share of the input.  The other
        # inputs and the outputs are not adjusted (the per-arc simplification of the input-side floors).
        _add_max_input_flow_online(b, sfx, online_set, v_online, v_startup, v_shutdown)

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
                    _coarse_flows.add_cstr(b, f"maxOutputSum_online{sfx}",
                                           over=over_sum_on, sense="<=",
                                           lhs_terms=sum_lhs,
                                           rhs_terms={"online": sum_online_rhs})

        # ─── EQ. ─── minimum_uptime_{linear,integer} ─────────────────────────
        # ─────────────────────────────────────────────────────────────────────
        # v_online[p,d,t]  ≥  Σ_{(p,d′,t′,d,t) ∈ uptime_lookback} v_startup[p,d′,t′] + carry_up[p,d,t]
        #   ∀ (p,d,t) ∈ pdt_uptime_set ∩ this class ∩ online_rows
        # A unit started within the minimum uptime stays online.
        # uptime_lookback = {(p,d′,t′,d,t) : (d′,t′) = (d,t), or (d′,t′) an ancestor of (d,t) along dt_ramp_link
        #   with elapsed(d′,t′ → d,t) = Σ step_duration over the chain from (d′,t′) up to, not including, (d,t)
        #   < min_uptime[p]} — the chain stops at a step without a ramp predecessor (solve start, a period that
        #   does not continue the previous one, a gap in the timeline, a representative-block start, a forecast
        #   branch that starts the solve); no cyclic wrap.
        # carry_up[p,d,t] = Σ_{h ∈ online_history} startup[p,h] over the root r ∈ dt_ramp_carry that is (d,t) or an
        #   ancestor of it along dt_ramp_link, with elapsed(r → d,t) + hours_to_end[h] < min_uptime[p]
        #   (d.uptime_carry, 0-filled; absent where no step continues the previous solve's last realized step).
        # A coarse process: rows and look-back only at the first fine steps of its coarse steps (its only start-ups),
        #   so elapsed(b′ → b) = Σ of the coarse durations from b′ up to b: a start-up counts while the steps since
        #   its start last less than min_uptime, i.e. min_uptime in whole coarse steps, rounded up.  A row whose
        #   look-back holds only its own step and no carry (the previous coarse step lasts ≥ min_uptime) is maxStartup
        #   and is not built.
        if (d.pdt_uptime_set is not None and d.pdt_uptime_set.height > 0
                and d.uptime_lookback is not None and d.uptime_lookback.height > 0):
            up_idx = d.pdt_uptime_set.join(online_set, on="p", how="inner")
            if b.has_coarse_proc:
                up_idx = up_idx.join(rows, on=["p", "d", "t"], how="semi")
            if up_idx.height > 0:
                # Lookback rows whose v_startup exists and whose process is in this class.
                p_idx_back = rows.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"})
                lkb = (d.uptime_lookback.join(online_set, on="p", how="inner")
                       .join(p_idx_back, on=["p", "d_back", "t_back"], how="inner")
                       .join(up_idx, on=["p", "d", "t"], how="inner"))
                up_idx, lkb = _drop_own_step_rows(b, up_idx, lkb, d.uptime_carry)
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
        # unit with existing = 0 could never be online).
        # downtime_lookback: as uptime_lookback with min_downtime[p] (a coarse process as in minimum_uptime; a row
        #   whose look-back holds only its own step and no carry is implied by maxShutdown / online_change).
        # carry_dn: as carry_up with shutdown[p,h] and min_downtime[p] (d.downtime_carry, 0-filled).
        # A retirement v_divest[e,d′] counts in d iff (e,d′,d) ∈ edd_divest_active (ndd_divest_active):
        #   d′ ≤ d < EOL[e], the end of life of e's pre-existing capacity (none for reinvest_automatic).
        if (d.pdt_downtime_set is not None and d.pdt_downtime_set.height > 0
                and d.downtime_lookback is not None and d.downtime_lookback.height > 0):
            dn_idx = d.pdt_downtime_set.join(online_set, on="p", how="inner")
            if b.has_coarse_proc:
                dn_idx = dn_idx.join(rows, on=["p", "d", "t"], how="semi")
            if dn_idx.height > 0:
                p_idx_back = rows.pipe(rename_to_axis, {"d": "d_back", "t": "t_back"})
                lkb = (d.downtime_lookback.join(online_set, on="p", how="inner")
                       .join(p_idx_back, on=["p", "d_back", "t_back"], how="inner")
                       .join(dn_idx, on=["p", "d", "t"], how="inner"))
                dn_idx, lkb = _drop_own_step_rows(b, dn_idx, lkb, d.downtime_carry)
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
    # v_online[p,d,t] = p_fix_online[p,d,t]   ∀ (p,d,t) ∈ online_rows with a p_fix_online value
    #   (a coarse process: at the first fine step of each coarse step; its pin is the same on every fine step of
    #   the step, or the process keeps the solve's step, flex_data.coarse_pin_varies)
    # In a separate-dispatch solve, pins units online to the commitment of the parent solve.
    # Only declared v_online cells are pinned.  p_fix_online is Utf8-keyed: it is joined to
    # the native-typed index through Utf8 copies of (p, d, t).
    if d.p_fix_online is not None and d.p_fix_online.frame.height > 0:
        fix_u = d.p_fix_online.frame.select(
            pl.col("p").cast(pl.Utf8).alias("_pk"),
            pl.col("d").cast(pl.Utf8).alias("_dk"),
            pl.col("t").cast(pl.Utf8).alias("_tk"),
            "value")
        for kind, _p_idx, _online_set, v_online, _v_startup, _v_shutdown in uc_classes(b):
            pin = (online_rows(b, kind).select("p", "d", "t").unique()
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


def _drop_own_step_rows(b: "_Build", idx: pl.DataFrame, lkb: pl.DataFrame,
                        carry: "pl.DataFrame | None") -> "tuple[pl.DataFrame, pl.DataFrame]":
    """``idx`` / ``lkb`` (minimum up / down time rows and their look-back) without the rows of coarse processes
    whose look-back holds only their own step and nothing is carried: such a row reads on ≥ su (or on + sd ≤
    count), which maxStartup (maxShutdown and online_change) already state.  That is the case where the previous
    coarse step lasts at least the minimum time (its first fine step is then outside the look-back)."""
    if not b.has_coarse_proc or idx.height == 0:
        return idx, lkb
    keys = ["p", "d", "t"]
    coarse = _coarse_flows.split_arcs(b, idx.select("p").unique())[1]
    if coarse.height == 0:
        return idx, lkb
    own = (lkb.join(coarse, on="p", how="semi")
           .group_by(keys).agg(pl.len().alias("__n"))
           .filter(pl.col("__n") == 1)
           .select(keys))
    if carry is not None and carry.height > 0 and own.height > 0:
        carried = carry.select(*(pl.col(k).cast(pl.Utf8) for k in keys))
        own = (_utf8_keys(own.lazy(), keys).join(carried.rename({k: f"_{k}" for k in keys}).lazy(),
                                                 on=[f"_{k}" for k in keys], how="anti")
               .select(keys).collect())
    if own.height == 0:
        return idx, lkb
    return idx.join(own, on=keys, how="anti"), lkb.join(own, on=keys, how="anti")


def _add_max_input_flow_online(b: "_Build", sfx: str, online_set: "pl.DataFrame", v_online, v_startup,
                               v_shutdown) -> None:
    """maxInputFlow_online (+ _shutdown) rows of one UC class (see the EQ. header in :func:`add_constraints`)."""
    d = b.d
    indirect = getattr(d, "process_indirect", None)
    if indirect is None or indirect.height == 0:
        return
    keys = ["p", "source", "sink"]
    pss = d.process_source_sink.select(keys).unique()

    def _online_inputs(arcs: "pl.DataFrame | None") -> "pl.DataFrame":
        """The rows of v_flow's arcs (process_source_sink dtypes) among ``arcs`` that are capped input arcs of
        online indirect units (conversion_flow_coeff = 0 inputs are uncapped: no capacity row)."""
        if arcs is None or arcs.height == 0:
            return pss.head(0)
        sel = (_utf8_keys(arcs.lazy().select(keys), keys)
               .filter(pl.col("_sink") == pl.col("_p"))
               .join(_utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi")
               .join(_utf8_keys(online_set.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi"))
        return _drop_uncapped(d, _utf8_keys(pss.lazy(), keys)
                              .join(sel.select("_p", "_source", "_sink"), on=["_p", "_source", "_sink"], how="semi")
                              .select(keys).collect())

    reserve_arcs = (_online_inputs(b.reserve_down_input_arcs) if b.reserve_down_at_source
                    else pss.head(0))
    ramp_sets = [f for f in (d.process_source_sink_ramp_limit_source_up, d.process_source_sink_ramp_limit_source_down)
                 if _has_rows(f)]
    ramp_arcs = _online_inputs(pl.concat([f.select(keys) for f in ramp_sets]).unique() if ramp_sets else None)
    red, red_c = (_cap_reduction(b, ramp_arcs, input_side=True) if ramp_arcs.height > 0 else (None, None))
    in_arcs = reserve_arcs
    for r in (red, red_c):
        if r is not None:
            in_arcs = pl.concat([in_arcs, r.select(keys).unique(maintain_order=True)]).unique(maintain_order=True)
    if in_arcs.height == 0:
        return
    over = in_arcs.join(d.dt, how="cross")
    cap = _indirect_input_cap(d, in_arcs)
    rhs = v_online * Param(("p", "source", "sink", "d", "t"), cap)
    if d.p_process_availability is not None:
        rhs = rhs * Param(("p", "source", "sink", "d", "t"), _input_availability_share(b, in_arcs))
    lhs: dict = {"flow": b.v_flow}
    lhs.update(b.reserve_down_at_source)
    rhs_terms: dict = {"online": rhs}
    over2 = over3 = None
    row1 = row2 = row3 = {}
    if red is not None:
        row1, row2, over2, over3, row3 = _reduction_terms(red, cap, v_startup, v_shutdown, f"_in{sfx}")
    c1 = c2 = c3 = {}
    c_over2 = c_over3 = None
    if red_c is not None:
        # A coarse process: the input limit's step mean, the reduction on the coarse grid.
        cap_c = _coarse_flows.average(b, Param(("p", "source", "sink", "d", "t"), cap)).frame
        c1, c2, c_over2, c_over3, c3 = _reduction_terms(red_c, cap_c, _coarse_flows.at_step(b, v_startup),
                                                        v_shutdown, f"_in_coarse{sfx}", tcol="b_first")
    _coarse_flows.add_cstr(b, f"maxInputFlow_online{sfx}", over=over, sense="<=", lhs_terms=lhs,
                           rhs_terms=rhs_terms, fine_rhs=row1, coarse_rhs=c1)
    if over2 is not None and over2.height > 0:
        b.m.add_cstr(f"maxInputFlow_online_shutdown{sfx}", over=over2, sense="<=", lhs_terms=lhs,
                     rhs_terms={"online": rhs, **row2})
    if c_over2 is not None and c_over2.height > 0:
        _coarse_flows.add_twin(b, f"maxInputFlow_online_shutdown{sfx}", over=c_over2, sense="<=", lhs_terms=lhs,
                               rhs_terms={"online": rhs}, coarse_rhs=c2)
    if over3 is not None and over3.height > 0:
        b.m.add_cstr(f"maxInputFlow_online_shutdown_branch{sfx}", over=over3, sense="<=", lhs_terms=lhs,
                     rhs_terms={"online": rhs, **row3})
    if c_over3 is not None and c_over3.height > 0:
        _coarse_flows.add_twin(b, f"maxInputFlow_online_shutdown_branch{sfx}", over=c_over3, sense="<=",
                               lhs_terms=lhs, rhs_terms={"online": rhs}, coarse_rhs=c3)


def _cap_reduction(b: "_Build", arcs: "pl.DataFrame", *,
                   input_side: bool) -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """Startup / shutdown capacity reduction of online arcs ``arcs`` (``p, source, sink``): ``(fine, coarse)``.

    ``fine`` covers the arcs of fine processes over dt: ``(p, source, sink, d, t, red_su, red_sd, d_next, t_next,
    __nsucc)`` in the dtypes of ``arcs`` / ``d.dt`` (``d_next``, ``t_next`` Utf8, null where the step has no
    successor; ``__nsucc`` = the number of successors of ``(d, t)``, 0 without one), one row PER SUCCESSOR at a
    branch point (several rows sharing ``(p, source, sink, d, t)``, critique item 1), only rows with ``red_su > 0``
    or (a successor and ``red_sd > 0``); ``None`` when there are none.  See the EQ. header of maxFlow_online:
    ``SU = min(1, ml + ½ · ramp_speed_up · 60 · step_duration)`` over the sides of the arc with an up ramp limit,
    ``red_su = max(0, availability − SU)``; ``SD`` / ``red_sd`` likewise with the down speeds.  ``ml`` is
    ``min_load`` on an output-measured arc (``input_side`` False) and the input the unit draws at min load as a
    share of the arc's input limit on an indirect unit's input arc (``input_side`` True).

    ``coarse`` covers the arcs of coarse processes in the same columns, one row per coarse step b (``t`` = its
    first fine step b_first) and coarse successor b⁺ (``d_next``, ``t_next`` = b⁺'s first fine step,
    :func:`_coarse_flows.coarse_links`): ml and availability are their Δt-weighted means over b and SU / SD the
    mean share over the whole step of a unit that starts at its start / stops at its end,
    :func:`_coarse_flows.step_limit_share` with ρ_b = ramp_speed · 60 · Δ_b (the exact mean of a linear ramp that
    saturates at full load).
    """
    d = b.d
    keys = ["p", "source", "sink"]
    sides = []
    for dir_ in ("up", "down"):
        for side in (("source",) if input_side else ("sink", "source")):
            idx = getattr(d, f"process_source_sink_ramp_limit_{side}_{dir_}", None)
            speed = getattr(d, f"p_ramp_speed_{dir_}_{side}", None)
            if idx is None or idx.height == 0 or speed is None:
                continue
            hit = (_utf8_keys(idx.lazy().select(keys), keys).select([f"_{k}" for k in keys]).unique()
                   .with_columns(pl.col(f"_{side}").alias("_n"))
                   .join(speed.frame.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                                   pl.col(side).cast(pl.Utf8).alias("_n"),
                                                   pl.col("value").cast(pl.Float64).alias("__speed")),
                         on=["_p", "_n"], how="inner")
                   .select("_p", "_source", "_sink", "__speed")
                   .with_columns(__dir=pl.lit(dir_)))
            sides.append(hit)
    if not sides:
        return None, None
    speeds = pl.concat(sides)
    if b.has_coarse_proc:
        fine_arcs, coarse_arcs = _coarse_flows.split_arcs(b, arcs)
    else:
        fine_arcs, coarse_arcs = arcs, arcs.head(0)
    fine = (_reduction_rows(b, fine_arcs, speeds, input_side=input_side, coarse=False)
            if fine_arcs.height > 0 else None)
    coarse = (_reduction_rows(b, coarse_arcs, speeds, input_side=input_side, coarse=True)
              if coarse_arcs.height > 0 else None)
    return fine, coarse


def _reduction_rows(b: "_Build", arcs: "pl.DataFrame", speeds: "pl.LazyFrame", *, input_side: bool,
                    coarse: bool) -> "pl.DataFrame | None":
    """The rows of :func:`_cap_reduction` for ``arcs`` (all fine or all coarse processes, ``coarse``)."""
    d = b.d
    keys = ["p", "source", "sink"]
    base = (_utf8_keys(arcs.select(keys).unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
                       .lazy(), keys + ["d", "t"])
            .join(speeds.select("_p", "_source", "_sink").unique(), on=["_p", "_source", "_sink"], how="semi"))
    dur = d.p_step_duration.frame.lazy().select(pl.col("d").cast(pl.Utf8).alias("_d"),
                                                pl.col("t").cast(pl.Utf8).alias("_t"),
                                                pl.col("value").cast(pl.Float64).alias("__dur"))
    ml = _min_load_share(b, base, input_side=input_side)
    if input_side:
        avail = (_input_availability_share(b, arcs).lazy()
                 .select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("source").cast(pl.Utf8).alias("_source"),
                         pl.col("sink").cast(pl.Utf8).alias("_sink"), pl.col("d").cast(pl.Utf8).alias("_d"),
                         pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__av")))
    elif d.p_process_availability is not None:
        avail = (_availability_factor(d, arcs).frame.lazy()
                 .select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                         pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__av")))
    else:
        avail = None
    grid = base.join(dur, on=["_d", "_t"], how="inner").join(ml, on=["_p", "_source", "_sink", "_d", "_t"],
                                                              how="left")
    grid = grid.with_columns(pl.col("__ml").fill_null(0.0))
    if avail is not None:
        on = ["_p", "_source", "_sink", "_d", "_t"] if input_side else ["_p", "_d", "_t"]
        grid = grid.join(avail, on=on, how="left").with_columns(pl.col("__av").fill_null(1.0))
    else:
        grid = grid.with_columns(__av=pl.lit(1.0))
    if coarse:
        # One row per coarse step b (named by its first fine step): Δ_b and the step means of ml and availability.
        st = b.coarse_steps.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                                          pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("b_first").alias("_b"),
                                          pl.col("w").alias("__w"))
        t_dtype = d.dt.schema["t"]
        grid = (grid.join(st, on=["_p", "_d", "_t"], how="inner")
                .group_by(*keys, "d", "_p", "_source", "_sink", "_d", "_b")
                .agg(pl.col("__dur").sum(), (pl.col("__w") * pl.col("__ml")).sum().alias("__ml"),
                     (pl.col("__w") * pl.col("__av")).sum().alias("__av"))
                .with_columns(pl.col("_b").cast(t_dtype).alias("t"), pl.col("_b").alias("_t")))
        # The mean share over the whole step of a unit that starts at its start (stops at its end).
        lim_expr = _coarse_flows.step_limit_share(pl.col("__ml"), pl.col("__speed") * 60.0 * pl.col("__dur"))
        from flextool.engine.model.core import ramp_step_links

        links = _coarse_flows.coarse_links(b, ramp_step_links(d)[0])
        succ = links.lazy().select("_p", pl.col("__d_prev").alias("_d"), pl.col("__t_prev").alias("_t"),
                                   pl.col("_d").alias("d_next"), pl.col("_b").alias("t_next"))
        succ_on = ["_p", "_d", "_t"]
        nsucc_over = ["_p", "_source", "_sink", "_d", "_t"]
    else:
        # One startup (shutdown) step reaches ml + half a step's ramp on average (Q-c14a-A).
        lim_expr = pl.min_horizontal(pl.lit(1.0), pl.col("__ml") + 0.5 * pl.col("__speed") * 60.0 * pl.col("__dur"))
        succ = _successor_steps(d)
        succ_on = ["_d", "_t"]
        nsucc_over = ["_p", "_source", "_sink", "_d", "_t"]
    # The tightest side wins.
    lim = (grid.join(speeds, on=["_p", "_source", "_sink"], how="inner")
           .with_columns(__lim=lim_expr)
           .group_by("_p", "_source", "_sink", "_d", "_t", "__dir")
           .agg(pl.col("__lim").min()))
    su = lim.filter(pl.col("__dir") == "up").select("_p", "_source", "_sink", "_d", "_t",
                                                     pl.col("__lim").alias("__su"))
    sd = lim.filter(pl.col("__dir") == "down").select("_p", "_source", "_sink", "_d", "_t",
                                                       pl.col("__lim").alias("__sd"))
    out = (grid.join(su, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
           .join(sd, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
           .join(succ, on=succ_on, how="left")
           .with_columns(red_su=(pl.col("__av") - pl.col("__su")).clip(lower_bound=0.0).fill_null(0.0),
                         red_sd=pl.when(pl.col("t_next").is_not_null())
                         .then((pl.col("__av") - pl.col("__sd")).clip(lower_bound=0.0).fill_null(0.0))
                         .otherwise(0.0))
           .with_columns(__nsucc=pl.when(pl.col("t_next").is_not_null())
                         .then(pl.col("t_next").count().over(*nsucc_over))
                         .otherwise(0))
           .filter((pl.col("red_su") > 0.0) | (pl.col("red_sd") > 0.0))
           .select(*keys, "d", "t", "red_su", "red_sd", "d_next", "t_next", "__nsucc")
           .collect())
    return out if out.height > 0 else None


def _successor_steps(d) -> "pl.LazyFrame":
    """``(_d, _t, d_next, t_next)`` Utf8: every successor ``(d_next, t_next)`` whose ramp predecessor
    (``dt_ramp_link``) is ``(_d, _t)`` — the same period (today's only case) or the next continuing period
    (no ``d_previous == d`` filter, so the pre-shutdown limit also applies across a continuing period boundary).  Several rows per predecessor at a branch point (several mid-horizon
    branches continuing from the same trunk step) — callers must not sum across them: that
    would let several branches each buy back a whole unit of ramp room from the trunk step instead of each
    being bound by its own (mutually exclusive) branch."""
    link = getattr(d, "dt_ramp_link", None)
    if link is None or link.height == 0:
        return pl.LazyFrame(schema={"_d": pl.Utf8, "_t": pl.Utf8, "d_next": pl.Utf8, "t_next": pl.Utf8})
    return (link.lazy()
            .select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                    pl.col("d_previous").cast(pl.Utf8), pl.col("t_previous").cast(pl.Utf8))
            .select(pl.col("d_previous").alias("_d"), pl.col("t_previous").alias("_t"),
                    pl.col("d").alias("d_next"), pl.col("t").alias("t_next")))


def _min_load_share(b: "_Build", base: "pl.LazyFrame", *, input_side: bool) -> "pl.LazyFrame":
    """``(_p, _source, _sink, _d, _t, __ml)``: the share of the arc's capacity limit a unit online runs at
    at min load.  Output-measured arcs: ``min_load``.  An indirect unit's input arc (``input_side``): the input
    drawn at min load per unit of input limit, ``x(min_load) / input_share_max`` (:func:`_fuel_share`).  0 where
    the unit has no min_load."""
    d = b.d
    cols = ["_p", "_source", "_sink", "_d", "_t"]
    if d.p_min_load is None:
        return base.select(cols).with_columns(__ml=pl.lit(0.0))
    ml = (promote_param_to_dt(d.p_min_load, d.dt)
          .select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                  pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__y")))
    rows = base.select(cols).join(ml, on=["_p", "_d", "_t"], how="left").with_columns(pl.col("__y").fill_null(0.0))
    if not input_side:
        return rows.select(*cols, pl.col("__y").alias("__ml"))
    out = rows.join(_fuel_share(d, rows.select("_p", "_d", "_t", "__y").unique()), on=["_p", "_d", "_t"], how="left")
    share = getattr(d, "p_process_source_input_share_max", None)
    if share is not None:
        out = (out.join(share.frame.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                                  pl.col("source").cast(pl.Utf8).alias("_source"),
                                                  pl.col("value").cast(pl.Float64).alias("__ism")),
                        on=["_p", "_source"], how="left"))
    else:
        out = out.with_columns(__ism=pl.lit(None, pl.Float64))
    # No fuel width (no output data): no reduction is safe only with share 1 (SU = 1).
    return out.select(*cols, __ml=pl.when(pl.col("__x").is_null()).then(1.0)
                      .otherwise(pl.col("__x") / pl.col("__ism").fill_null(1.0)))


def _input_availability_share(b: "_Build", arcs: "pl.DataFrame") -> "pl.DataFrame":
    """``(p, source, sink, d, t, value)`` over ``arcs`` (indirect units' input arcs) × dt, dtypes of ``arcs`` /
    ``dt``: the input drawn at derated full output per unit of full-load input, ``x(availability)``
    (:func:`_fuel_share`).  It replaces availability on the input arc's online limit: the min-load section is drawn
    per unit online, not per available unit, so ``availability · full-load input`` would cap the derated full
    output.  Equal to availability without a section (one output); availability where there is no fuel width."""
    d = b.d
    keys = ["p", "source", "sink"]
    over = arcs.select(keys).unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
    ou = _utf8_keys(over.lazy(), keys + ["d", "t"])
    if d.p_process_availability is not None:
        av = (_availability_factor(d, arcs).frame.lazy()
              .select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                      pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__y")))
        ou = ou.join(av, on=["_p", "_d", "_t"], how="left").with_columns(pl.col("__y").fill_null(1.0))
    else:
        ou = ou.with_columns(__y=pl.lit(1.0))
    x = _fuel_share(d, ou.select("_p", "_d", "_t", "__y").unique())
    return (ou.join(x, on=["_p", "_d", "_t"], how="left")
            .select(*keys, "d", "t", value=pl.coalesce(pl.col("__x"), pl.col("__y")))
            .collect())


def _fuel_share(d, levels: "pl.LazyFrame", online_p: "pl.LazyFrame | None" = None) -> "pl.LazyFrame":
    """``(_p, _d, _t, __x)`` for ``levels`` ``(_p, _d, _t, __y)`` (Utf8 keys): the input an indirect unit
    draws at output level y per unit of its full-load input,
    ``x(y) = (slope · W(y) + section) / (slope · W(1) + section)``, ``W(y)`` the output-side fuel width filled up to
    y (outputs in descending conversion_flow_coeff order, each up to its capacity_max_coeff — the most fuel the
    outputs can draw at that output level, as in ``p_indirect_input_cap``), section only for a
    ``min_load_efficiency`` unit that also has an online variable — its balance equation applies no section
    without one, so crediting it here would let the input bound run ahead of the actual fuel draw.  ``online_p``
    (``_p`` Utf8) restricts the section credit to those units; ``None`` keeps every ``min_load_efficiency`` unit
    (the caller has already restricted ``levels`` to online units).  ``__x`` null where the unit has no fuel width
    (no output data)."""
    empty = pl.LazyFrame(schema={"_p": pl.Utf8, "_d": pl.Utf8, "_t": pl.Utf8, "__x": pl.Float64})
    outs = getattr(d, "process_output_flows", None)
    if outs is None or outs.height == 0:
        return empty
    k = (outs.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("sink").cast(pl.Utf8).alias("_k"))
         .unique())
    for name, col in ((getattr(d, "p_process_sink_max_capacity_coef", None), "__cap"),
                      (getattr(d, "p_process_sink_conversion_flow_coeff", None), "__conv")):
        if name is not None:
            k = k.join(name.frame.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                                pl.col("sink").cast(pl.Utf8).alias("_k"),
                                                pl.col("value").cast(pl.Float64).alias(col)),
                       on=["_p", "_k"], how="left")
        else:
            k = k.with_columns(pl.lit(None, pl.Float64).alias(col))
    k = (k.with_columns(__cap=pl.col("__cap").fill_null(1.0).clip(lower_bound=0.0),
                        __conv=pl.col("__conv").fill_null(1.0))
         .filter(pl.col("__conv") > 0.0)
         .sort(["_p", "__conv", "_k"], descending=[False, True, False])
         .with_columns(__prev=pl.col("__cap").cum_sum().over("_p") - pl.col("__cap")))
    rows = levels.with_columns(__one=pl.lit(1.0))

    def _width(y: str, out: str) -> "pl.LazyFrame":
        return (rows.select("_p", "_d", "_t", y).unique().join(k, on="_p", how="inner")
                .with_columns(__fill=pl.min_horizontal(pl.col("__cap"),
                                                       (pl.col(y) - pl.col("__prev")).clip(lower_bound=0.0)))
                .group_by("_p", "_d", "_t").agg((pl.col("__conv") * pl.col("__fill")).sum().alias(out)))

    w = _width("__y", "__w_y").join(_width("__one", "__w_full"), on=["_p", "_d", "_t"], how="inner")

    def _pdt(param, out: str) -> "pl.LazyFrame | None":
        if param is None:
            return None
        return (promote_param_to_dt(param, d.dt)
                .select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                        pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias(out)))

    slope = _pdt(d.p_slope, "__slope")
    sec = _pdt(getattr(d, "p_section", None), "__sec")
    w = w.join(slope, on=["_p", "_d", "_t"], how="left") if slope is not None else w.with_columns(__slope=pl.lit(None))
    if sec is not None and _has_rows(getattr(d, "process_min_load_eff", None)):
        mle = d.process_min_load_eff.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p")).unique()
        if online_p is not None:
            mle = mle.join(online_p, on="_p", how="semi")
        w = (w.join(sec, on=["_p", "_d", "_t"], how="left")
             .join(mle.with_columns(__mle=pl.lit(True)), on="_p", how="left")
             .with_columns(__sec=pl.when(pl.col("__mle")).then(pl.col("__sec").fill_null(0.0)).otherwise(0.0)))
    else:
        w = w.with_columns(__sec=pl.lit(0.0))
    w = w.with_columns(pl.col("__slope").cast(pl.Float64).fill_null(0.0))
    return w.select("_p", "_d", "_t",
                    __x=pl.when(pl.col("__slope") * pl.col("__w_full") + pl.col("__sec") > 0.0)
                    .then((pl.col("__slope") * pl.col("__w_y") + pl.col("__sec"))
                          / (pl.col("__slope") * pl.col("__w_full") + pl.col("__sec")))
                    .otherwise(None))


def _reduction_terms(red: "pl.DataFrame", kappa: "pl.DataFrame", v_startup, v_shutdown, sfx: str, *,
                     tcol: str = "t"):
    """RHS terms of maxFlow_online / maxFlow_online_shutdown / maxFlow_online_shutdown_branch (or the input
    triple) from ``red`` (:func:`_cap_reduction`) and the arc limit factor ``kappa``
    ``(p, source, sink, d, t, value)``: returns ``(terms_row1, terms_row2, over_row2, over_row3, terms_row3)``.
    ``tcol="b_first"``: the coarse rows of :func:`_cap_reduction` (``v_startup`` the :func:`_coarse_flows.at_step`
    view, ``kappa`` keyed ``b_first`` where it varies in time): terms and rows keyed ``(p, source, sink, d,
    b_first)``, the coarse grid of the ``<family>_coarse`` twins, with the shut-down at the coarse successor b⁺.

    Predecessor rows with a UNIQUE successor (``__nsucc <= 1``, the common case — today's structure, byte-
    unchanged) feed row1 / row2 exactly as before.  Predecessor rows at a branch point (``__nsucc > 1``,
    several mid-horizon branches continuing from the same trunk step) feed row3 instead: ONE
    maxFlow_online_shutdown_branch row per (predecessor, successor) pair, never summed across branches
    (critique item 1 — summing would let several branches each buy back a whole unit of ramp room from the
    trunk step, instead of each being bound by its own mutually-exclusive scenario).  row1 omits the
    shutdown_reduction cross-term at a branch point: it bounds the single v_flow[t] variable and cannot itself
    be split per branch; row3's rows give the tight, correctly-isolated bound there instead (row1 alone, without
    the cross-term, is still a valid — just looser — bound, so nothing is under-constrained).
    """
    keys = ["p", "source", "sink", "d", tcol]
    if tcol != "t":
        # The coarse grid: the row's step named by its first fine step (Utf8), the column of :func:`at_step`.
        red = red.with_columns(pl.col("t").cast(pl.Utf8)).rename({"t": tcol})
    on = [k for k in keys if k in kappa.columns]
    r = (_utf8_keys(red.lazy(), on)
         .join(_utf8_keys(kappa.lazy(), on).select([f"_{k}" for k in on] + [pl.col("value").alias("__k")]),
               on=[f"_{k}" for k in on], how="inner")
         .drop([f"_{k}" for k in on]).collect()
         .with_columns(__a=pl.col("__k") * pl.col("red_su"),
                       __b=pl.col("__k") * (pl.col("red_sd") - pl.col("red_su")).clip(lower_bound=0.0),
                       __c=pl.col("__k") * pl.col("red_sd"),
                       __e=pl.col("__k") * (pl.col("red_su") - pl.col("red_sd")).clip(lower_bound=0.0)))
    r_unique = r.filter(pl.col("__nsucc") <= 1)
    r_branch = r.filter(pl.col("__nsucc") > 1)

    def _su(col: str, rows_df: "pl.DataFrame"):
        # .unique(): at a branch point rows_df carries one row per successor sharing the same (p,s,k,d,t) and
        # the same value of ``col`` (it does not depend on which successor) — collapse back to one Param row.
        rows = rows_df.filter(pl.col(col) > 0.0).select(*keys, value=pl.col(col)).unique()
        if rows.height == 0:
            return None
        return v_startup * Param(tuple(keys), rows)

    # Renames BOTH d and t (not just t): the unique successor can now be in the next continuing period
    # (#98), so the view must follow it there — v_shutdown[p, d_next, t_next], not v_shutdown[p, d, t_next].
    sd_next = _var_view(v_shutdown, f"__sd_next{sfx}", {"d": "d_next", "t": "t_next"}, to_axis=True)
    d_next_dtype, t_next_dtype = sd_next.frame.schema["d_next"], sd_next.frame.schema["t_next"]

    def _sd_unique(col: str):
        rows = r_unique.filter((pl.col(col) > 0.0) & pl.col("t_next").is_not_null())
        if rows.height == 0:
            return None
        coef = Param(tuple(keys) + ("d_next", "t_next"),
                     rows.select(*keys, pl.col("d_next").cast(d_next_dtype), pl.col("t_next").cast(t_next_dtype),
                                 value=pl.col(col)))
        return Sum(sd_next * coef, over=("d_next", "t_next"))

    row1 = {k: -v for k, v in (("startup_reduction", _su("__a", r)), ("shutdown_reduction", _sd_unique("__b")))
            if v is not None}
    over2 = r_unique.filter((pl.col("red_sd") > 0.0) & pl.col("t_next").is_not_null()).select(keys)
    row2 = {k: -v for k, v in (("shutdown_reduction", _sd_unique("__c")),
                               ("startup_reduction", _su("__e", r_unique)))
            if v is not None}

    over3 = r_branch.filter((pl.col("red_sd") > 0.0) & pl.col("t_next").is_not_null()).select(
        *keys, "d_next", "t_next")
    row3: dict = {}
    if over3.height > 0:
        # over3's own index already carries (d_next, t_next) — one row per (predecessor, successor) pair — so
        # the shutdown term here is a plain product (no Sum/contraction: each row has exactly one successor).
        c_rows = r_branch.filter((pl.col("__c") > 0.0) & pl.col("t_next").is_not_null())
        if c_rows.height > 0:
            row3["shutdown_reduction"] = -sd_next * Param(
                tuple(keys) + ("d_next", "t_next"),
                c_rows.select(*keys, pl.col("d_next").cast(d_next_dtype), pl.col("t_next").cast(t_next_dtype),
                              value=pl.col("__c")))
        e_term = _su("__e", r_branch)
        if e_term is not None:
            row3["startup_reduction"] = -e_term
    return row1, row2, over2, over3, row3


def _online_mintime_carry_term(carry: "pl.DataFrame | None", idx: "pl.DataFrame") -> "Param | None":
    """The cross-roll minimum up/down-time carry as a ``Param((p, d, t))`` over ``idx``.

    ``carry`` is ``(p, d, t, value)`` with Utf8 axes (``_derived.online_mintime_carry_from_source``); its
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
