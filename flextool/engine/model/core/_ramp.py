"""Ramp limits and ramp cost rows of process flows."""

import polars as pl
from polar_high import Sum, Where, Param
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _online
from flextool.engine.model import _unit_floors
from flextool.engine.model._common import _coef_factor, _drop_uncapped, _has_rows, _utf8_keys, _var_view
from flextool.engine.model.core._context import _Build


def _eq_ramp(b: _Build) -> None:
    """Ramp limits of process flows."""
    if not b.has_ramp:
        return
    d = b.d
    link, carry = ramp_step_links(d)

    # ─── EQ. ─── ramp_{sink,source}_{up,down}_constraint ─────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Units without online variables (as c14a):
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] + R↑[p,s,k,d,t]
    #   ≤ +ρ̄ · c[p,s,k,d,t] · (existing_count[p,d] + Σ_{d″} v_invest_p[p,d″] − Σ_{d″} v_divest_p[p,d″])        (up)
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − R↓[p,s,k,d,t]
    #   ≥ −ρ̄ · c[p,s,k,d,t] · (existing_count[p,d] + Σ v_invest_p − Σ v_divest_p)                              (down)
    # Units with online variables (linear or integer class), room on the units online, split by unit state:
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] + R↑[p,s,k,d,t]
    #   ≤ c · ( ρ̄ · (v_online[p,d,t] − v_startup[p,d,t]) + SU[p,s,k,d,t] · v_startup[p,d,t] )
    #     − F[p,s,k,d′,t′] · v_shutdown[p,d,t]                                                                  (up)
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − R↓[p,s,k,d,t]
    #   ≥ −c · ( ρ̄ · (v_online[p,d,t] − v_startup[p,d,t]) + SD′[p,s,k,d′,t′] · v_shutdown[p,d,t] )
    #     + F[p,s,k,d,t] · v_startup[p,d,t]                                                                     (down)
    #   ρ̄ = ramp_speed_{dir}_{side}[p,node] · 60 · (step_duration[d′,t′] + step_duration[d,t]) / 2
    #   SU = min(1, ml[p,s,k,d,t] + ½ · ramp_speed_up_{side} · 60 · step_duration[d,t])
    #   SD′ = min(1, ml[p,s,k,d′,t′] + ½ · ramp_speed_down_{side} · 60 · step_duration[d′,t′])
    #   ml = min_load (output-measured arc) or x(min_load) / input_share_max (indirect unit's input arc), 0
    #     without min_load;  F = the arc's floor per unit online (arc_floor_share): min_load on the only arc of
    #     a unit's output measure, min_load · capacity_min_coeff on an output of a multi-output unit with
    #     capacity_min_coeff > 0, else 0
    #   ∀ (p,s,k) ∈ process_source_sink_ramp_limit_{side}_{dir}, (d,t,d′,t′) ∈ dt_ramp_link; on a link inside one
    #     period (d′ = d) only where ρ̄ < 1
    # On a continuation roll the first step of a period ramps from the previous roll's realized flow
    #   (dt_ramp_carry): v_flow[p,s,k,d′,t′] → f̂[p,s,k,d,t] (p_roll_continue_flow, a constant), step_duration
    #   [d′,t′] → its carried duration; for units with online variables F[d′,t′] → 0 (no shutdown term on the
    #   up row) and SD′ → max(1, f̂[p,s,k,d,t] / (c[p,s,k,d,t] · ĉ[p])) (1 where the carried online count
    #   ĉ = p_roll_continue_online is 0 — no stoppers then; exact and always feasible, unlike a fixed κ/c).
    #   On the carried down row of a unit without online variables f̂ → min(f̂, U[p,s,k,d,t] + ρ̄ · c ·
    #   existing_count[p,d]) (fine steps), U the arc's maxFlow RHS (its flow ceiling at the existing capacity): a
    #   carried ramp-down yields to an outage the previous solve did not see — the row then asks at most U (W-0592).
    # A coarse process (processCoarse, _coarse_flows.ramp_rows) ramps between its coarse steps: t = b (named by its
    #   first fine step), (d′,t′) = b⁻, the coarse step that holds the fine predecessor of b's first step, and
    #   step_duration → Δ_b = Σ_{t∈b} Δt, so ρ̄ = speed · 60 · (Δ_b⁻ + Δ_b)/2 (no row inside a coarse step; with
    #   24-h steps an in-period row exists only for a speed below 1/1440 per minute); a time-varying source factor
    #   c is its coarse-step mean; the carried step is the arc's last realized coarse step (its length carried).
    #   With online variables v_online is the step's value, v_startup / v_shutdown the events at b (start-ups and
    #   shut-downs relative to b⁻, online_change); c and F are their means over b (F[b⁻] over b⁻) and
    #   SU[b] = mean share over b of a unit started at b's start: ml + ½·ρ_b while ρ_b ≤ 1 − ml, else
    #   1 − (1 − ml)²/(2·ρ_b), ρ_b = ramp_speed_up · 60 · Δ_b, ml the mean over b (SD′[b⁻] likewise with the down
    #   speed over b⁻: the exact mean of a linear ramp that reaches full load within the step).
    # Steps without a predecessor, on arcs that hold reserve in that direction there (prundt):
    #   R↑ ≤ ρ·c·cap (up),  −R↓ ≥ −ρ·c·cap (down),  ρ = ramp_speed · 60 · step_duration[d,t] < 1;
    #   cap = v_online[p,d,t] for units with online variables, built units otherwise.
    # ─── EQ. ─── ramp_{sink,source}_{up,down}_reserve_coarse ─────────────────
    # R↑[p,s,k,d,b] ≤ room_b · cap (up),  −R↓[p,s,k,d,b] ≥ −room_b · cap (down)
    #   ∀ arc of a coarse process with the ramp limit, holding reserve in that direction on its coarse step (d,b)
    #   (b = its first fine step) with some ρ_t < 1;  room_b = min_{t∈b, ρ_t<1} ρ_t · c[p,s,k,d,t], ρ_t the room of
    #   the fine row at t: ramp_speed · 60 · (step_duration[d′,t′] + step_duration[d,t]) / 2 with t's ramp
    #   predecessor (d′,t′) (a carried one: the arc's carried duration p_roll_continue_flow_duration, as the
    #   carried ramp row), ramp_speed · 60 · step_duration[d,t] without one;  c the arc's capacity factor at t;
    #   cap = v_online (units online of the step) or the built units
    # A coarse process holds one reserve and one flow per coarse step: the fine rows inside the step (no flow change)
    # would bound the reserve by each step's own room, so the smallest binds; the ramp between coarse steps above
    # still holds the reserve together with the flow change.
    # R↑ = Σ_r v_reserve[p,r,up,k,d,t] + φ·Σ_r v_reserve[p,r,down,s,d,t]
    #   (raises the arc: up at its sink, down at its source)
    # R↓ = Σ_r v_reserve[p,r,down,k,d,t] + φ·Σ_r v_reserve[p,r,up,s,d,t]
    #   (lowers it: down at its sink, up at its source)
    #   φ = 1 where v_flow is the input (indirect unit's input arc, sink-less unit), 1/slope on a direct arc;
    #   units only.
    # Limits the change of a flow between two consecutive steps, plus the reserve it holds in that direction, to its
    # ramp room: ramp_speed is a share of capacity per minute and flows are step averages, so the room between two
    # step averages is the speed × the distance between the two step centres (a valid bound for every trajectory
    # that obeys the speed; it is the tight one for a linear ramp).  For a unit with online variables the units
    # online in both steps change by at most ρ̄ each, a unit starting in step t reaches at most SU (half a step's
    # ramp above min load, the startup-step limit of maxFlow_online) and a stopping unit drops from at least its
    # floor F; downwards a stopping unit drops at most from SD′ (the pre-shutdown limit) and a starting unit rises
    # to at least F.  v_startup / v_shutdown are relative to the same predecessor (online_change), so a start/stop
    # pair with online unchanged gains c·(ml − ½·ρ·Δ) − F ≤ 0 of room on the only output of a unit
    # (research_backbone_startup_ramp.md); on an input, or on one output of a multi-output unit, a pair can still
    # gain up to the min-load share, paid by the startup cost.  A step has a predecessor (dt_ramp_link) only where
    # it is adjacent on the timeline in the un-rolled solve: no cyclic wrap at the first step of a solve or of a
    # period that does not continue the previous one, no row across a gap or at the start of a representative
    # block.  Rows inside one period with ρ̄ ≥ 1 cannot bind and are not built (unchanged from c14a: every term of
    # the online-capacity split is still implied there, design §1.4).  Cross-period and carried rows are always
    # built (the capacity, or the online predecessor, can differ).
    # c scales the room as the arc's capacity is scaled: capacity_max_coeff[p,k] on the sink side; on the source
    # side input_share_max[p,s], or on an indirect unit's input arc p_indirect_input_cap.  v_flow and v_reserve
    # are in units of unitsize, existing_count = capacity / unitsize.  Arcs with conversion_flow_coeff = 0 get no
    # row.
    # Capacity in service: invested[e,d], retired[e,d] (and prebuilt[e,d]) of _capacity, the retirements of
    # existing capacity counting until its end of life (none for reinvest_automatic).
    keys = ["p", "source", "sink"]
    for side, dir_, idx_set, ramp_speed in (
        ("sink",   "up",   d.process_source_sink_ramp_limit_sink_up,     d.p_ramp_speed_up_sink),
        ("sink",   "down", d.process_source_sink_ramp_limit_sink_down,   d.p_ramp_speed_down_sink),
        ("source", "up",   d.process_source_sink_ramp_limit_source_up,   d.p_ramp_speed_up_source),
        ("source", "down", d.process_source_sink_ramp_limit_source_down, d.p_ramp_speed_down_source),
    ):
        if not _has_rows(idx_set) or ramp_speed is None:
            continue
        idx_set = _drop_uncapped(d, idx_set)
        if idx_set.height == 0:
            continue
        up = dir_ == "up"
        room_rows = None
        if b.has_coarse_proc:
            # A coarse process ramps between its coarse steps (_coarse_flows.ramp_rows); the reserve it holds in this
            # direction has its own room at the solve's steps of each coarse step (_coarse_flows.reserve_room_rows).
            fine_arcs, coarse_arcs = _coarse_flows.split_arcs(b, idx_set)
            parts = []
            if fine_arcs.height > 0:
                parts.append(_ramp_rows(d, fine_arcs, ramp_speed, side, dir_, link, carry))
            if coarse_arcs.height > 0:
                parts.append(_coarse_flows.ramp_rows(b, coarse_arcs, ramp_speed, side, link, carry))
                room_rows = _coarse_flows.reserve_room_rows(b, coarse_arcs, ramp_speed, side, dir_, link, carry)
            rows = pl.concat(parts) if parts else None
        else:
            rows = _ramp_rows(d, idx_set, ramp_speed, side, dir_, link, carry)
        for name, family_rows in ((f"ramp_{side}_{dir_}_constraint", rows),
                                  (f"ramp_{side}_{dir_}_reserve_coarse", room_rows)):
            if family_rows is not None and family_rows.height > 0:
                _ramp_family(b, name, family_rows, idx_set, ramp_speed, side, up, keys)


def _ramp_cost_rows(b: _Build) -> pl.DataFrame:
    """The rows R of ``ramp_change_eq`` (:func:`_ramp_rows` columns, ``__kind`` ``link`` / ``carry``): every arc of
    ``process_source_sink_ramp_cost`` at every step with a ramp predecessor, the same predecessors as the ramp
    limits (:func:`ramp_step_links`) but without their room filter; a coarse process at its coarse steps
    (``_coarse_flows.ramp_rows``)."""
    d = b.d
    link, carry = ramp_step_links(d)
    idx_set = d.process_source_sink_ramp_cost.select("p", "source", "sink")
    if b.has_coarse_proc:
        fine_arcs, coarse_arcs = _coarse_flows.split_arcs(b, idx_set)
        parts = []
        if fine_arcs.height > 0:
            parts.append(_ramp_rows(d, fine_arcs, None, "sink", "up", link, carry))
        if coarse_arcs.height > 0:
            parts.append(_coarse_flows.ramp_rows(b, coarse_arcs, None, "sink", link, carry))
        return pl.concat(parts)
    return _ramp_rows(d, idx_set, None, "sink", "up", link, carry)


def _eq_ramp_cost(b: _Build) -> None:
    """The change of a priced flow from its ramp predecessor, split into v_ramp_up and v_ramp_down."""
    if b.v_ramp_up is None:
        return
    rows = b.ramp_cost_rows
    keys = ["p", "source", "sink"]
    over = rows.select(*keys, "d", "t")
    carried = rows.filter(pl.col("__kind") == "carry")

    # ─── EQ. ─── ramp_change_eq ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,s,k,d,t] − v_flow[p,s,k,d′,t′] − v_ramp_up[p,s,k,d,t] + v_ramp_down[p,s,k,d,t] = 0
    #   ∀ (p,s,k) ∈ process_source_sink_ramp_cost, (d,t,d′,t′) ∈ dt_ramp_link   (every link, no room filter)
    # carried row (dt_ramp_carry): v_flow[p,s,k,d,t] − v_ramp_up + v_ramp_down = f̂[p,s,k,d,t] (p_roll_continue_flow)
    # coarse process: t = b (its coarse step, named by the first fine step), (d′,t′) = b⁻, as _coarse_flows.ramp_rows
    # Splits the change of a flow between a step and its ramp predecessor into an increase and a decrease, both
    # priced by the ramp cost (OBJ. ramp_cost), in units of unitsize.  The predecessors are those of the ramp limits:
    # no row at the first step of a solve, after a gap in the timeline, at a period that does not continue the
    # previous one or at a representative-block start; cross-period links are kept; a stochastic branch's first step
    # links to the shared predecessor; a continuation roll's first step ramps from the flow the previous roll
    # realized, so a change at a roll boundary is charged once, in the roll that realizes it.  Arcs with
    # conversion_flow_coeff = 0 have rows too (that parameter removes the ramp limits, not the cost).  Reserves are
    # not part of the change.  An indirect unit's input and output arcs are priced separately, each by its own side.
    lhs = _ramp_flow_change(b, rows, keys)
    lhs["ramp_up"] = -Where(b.v_ramp_up, over)
    lhs["ramp_down"] = Where(b.v_ramp_down, over)
    rhs: dict = {}
    if carried.height > 0:
        rhs["roll_continue_flow"] = Param(("p", "source", "sink", "d", "t"),
                                          carried.select(*keys, "d", "t", value=pl.col("__f_hat")))
    b.m.add_cstr(
        "ramp_change_eq",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _ramp_family(b: _Build, name: str, rows: pl.DataFrame, idx_set: pl.DataFrame, ramp_speed: Param, side: str,
                 up: bool, keys: list) -> None:
    """The rows ``rows`` (:func:`_ramp_rows` columns) of the ramp family ``name`` (EQ. header in :func:`_eq_ramp`)."""
    m, d = b.m, b.d
    sign = 1.0 if up else -1.0
    over = rows.select(*keys, "d", "t")
    carried = rows.filter(pl.col("__kind") == "carry")

    lhs = _ramp_flow_change(b, rows, keys)
    if up:
        if b.reserve_up_to_sink is not None:
            lhs["reserve_up"] = b.reserve_up_to_sink
        lhs.update(b.reserve_down_at_source)
    else:
        if b.reserve_down_to_sink is not None:
            lhs["reserve_down"] = -b.reserve_down_to_sink
        lhs.update({k: -v for k, v in b.reserve_up_at_source.items()})

    # Split rows by UC class membership (design §3.2): a process has one startup method, so it belongs to
    # at most one online class; the rest are built rows (c14a's existing_count / invest / divest terms).
    online_rows: dict = {}
    online_p_frames = []
    for kind, _p_idx_k, online_set_k, v_online_k, v_startup_k, v_shutdown_k in _online.uc_classes(b):
        sub = (_utf8_keys(rows.lazy(), ["p"])
               .join(_utf8_keys(online_set_k.lazy().select("p"), ["p"]).select("_p").unique(),
                     on="_p", how="semi")
               .select(rows.columns).collect())
        if sub.height > 0:
            online_rows[kind] = (sub, v_online_k, v_startup_k, v_shutdown_k)
            online_p_frames.append(sub.select("p").unique())
    if online_p_frames:
        excl_p = pl.concat(online_p_frames).unique()
        built_rows = (_utf8_keys(rows.lazy(), ["p"])
                      .join(_utf8_keys(excl_p.lazy(), ["p"]).select("_p"), on="_p", how="anti")
                      .select(rows.columns).collect())
    else:
        built_rows = rows

    rhs: dict = {}
    if built_rows.height > 0:
        rate_built = _ramp_rate(d, idx_set, side, built_rows, keys, b=b)
        rhs["limit"] = (rate_built * b.cap.existing_count_p) * sign
        if b.has_invest_p:
            rhs["invest"] = (b.cap.invested_p * rate_built) * sign
        if b.cap.retired_p is not None:
            rhs["divest"] = (b.cap.retired_p * rate_built) * (-sign)
    for kind, (sub, v_online_k, v_startup_k, v_shutdown_k) in online_rows.items():
        for term_name, term in _online_ramp_rhs(b, idx_set, ramp_speed, side, up, sub,
                                                v_online_k, v_startup_k, v_shutdown_k, keys).items():
            rhs[f"{term_name}_{kind}"] = term
    if carried.height > 0:
        f_hat = carried.select(*keys, "d", "t", value=pl.col("__f_hat"))
        if not up and built_rows.height > 0:
            f_hat = _carried_ramp_down_yields(b, idx_set, side, built_rows, f_hat, keys)
        rhs["roll_continue_flow"] = Param(("p", "source", "sink", "d", "t"), f_hat)
    m.add_cstr(
        name,
        over      = over,
        sense     = "<=" if up else ">=",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )


def _carried_ramp_down_yields(b: _Build, idx_set: pl.DataFrame, side: str, built_rows: pl.DataFrame,
                              f_hat: pl.DataFrame, keys: list) -> pl.DataFrame:
    """*f_hat* ``(p, source, sink, d, t, value)`` of the carried down-ramp rows with f̂ ≤ U + ρ̄ · c · ex on the fine
    carried rows of units without online variables (*built_rows*), U = the arc's maxFlow RHS (``b.flow_upper_rhs``,
    also the v_flow bound of a bound arc), ex = existing_count[p,d]: the row then asks flow ≥ U at most, the arc's
    own flow ceiling.  A higher f̂ (an outage the previous solve did not see) makes the row infeasible without
    investment; the carried ramp-down ends there instead (W-0592).  Every other row keeps its f̂."""
    d = b.d
    rows = built_rows.filter(pl.col("__kind") == "carry")
    if b.has_coarse_proc and rows.height > 0:
        rows = _coarse_flows.split_arcs(b, rows)[0]
    ex = b.cap.existing_count_p
    if rows.height == 0 or ex is None or b.flow_upper_rhs is None or not _has_rows(d.process_unit):
        return f_hat
    rows = _utf8_keys(rows.lazy(), ["p"]).join(
        _utf8_keys(d.process_unit.lazy().select("p"), ["p"]).select("_p").unique(), on="_p", how="semi"
    ).select(rows.columns).collect()
    if rows.height == 0:
        return f_hat
    rows = _c_values(rows, _c_param_plain(d, idx_set, side), keys)
    all_keys = [*keys, "d", "t"]
    upper = b.flow_upper_rhs.frame
    cap = (_utf8_keys(rows.lazy(), all_keys)
           .join(_utf8_keys(upper.lazy(), all_keys).select(*(f"_{k}" for k in all_keys),
                                                           pl.col("value").alias("__upper")),
                 on=[f"_{k}" for k in all_keys], how="inner")
           .join(_utf8_keys(ex.frame.lazy(), ["p", "d"]).select("_p", "_d", pl.col("value").alias("__ex")),
                 on=["_p", "_d"], how="inner")
           .select(*(f"_{k}" for k in all_keys),
                   (pl.col("__upper") + pl.col("__c") * pl.col("__ex") * pl.col("__room")).alias("__cap"),
                   pl.col("__f_hat"))
           .filter(pl.col("__f_hat") > pl.col("__cap"))
           .collect())
    if cap.height == 0:
        return f_hat
    from flextool.engine.preprocess.params._mintime import note_carried_obligation_cut
    note_carried_obligation_cut({r["_p"]: f"solve {b.solve_name}: the carried ramp-down ends at step "
                                          f"({r['_d']}, {r['_t']})" for r in cap.iter_rows(named=True)})
    return (_utf8_keys(f_hat.lazy(), all_keys)
            .join(cap.lazy().select(*(f"_{k}" for k in all_keys), "__cap"), on=[f"_{k}" for k in all_keys],
                  how="left", maintain_order="left")
            .select(*all_keys, pl.min_horizontal("value", pl.col("__cap").fill_null(pl.col("value"))).alias("value"))
            .collect())


def _ramp_flow_change(b: _Build, rows: pl.DataFrame, keys: list) -> dict:
    """The flow change of the ramp rows ``rows`` (:func:`_ramp_rows` columns): ``flow`` = v_flow at (d,t) on the
    ``link`` and ``carry`` rows, ``flow_prev`` = −v_flow at the predecessor (d′,t′) on the ``link`` rows (a carried
    predecessor is the constant f̂ on the right side).  Shared by the ramp limits and the ramp cost."""
    v_flow = b.v_flow
    flow_rows = rows.filter(pl.col("__kind") != "reserve").select(*keys, "d", "t")
    linked = rows.filter(pl.col("__kind") == "link")
    lhs: dict = {}
    if flow_rows.height > 0:
        lhs["flow"] = Where(v_flow, flow_rows)
    if linked.height > 0:
        prev = _var_view(v_flow, "__ramp_prev", {"d": "d_previous", "t": "t_previous"}, to_axis=True)
        pairs = linked.select(*keys, "d", "t",
                              pl.col("__d_prev").cast(prev.frame.schema["d_previous"]).alias("d_previous"),
                              pl.col("__t_prev").cast(prev.frame.schema["t_previous"]).alias("t_previous"))
        lhs["flow_prev"] = -Sum(Where(prev, pairs), over=("d_previous", "t_previous"))
    return lhs


def ramp_step_links(d) -> "tuple[pl.DataFrame, pl.DataFrame]":
    """The ramp predecessors of the steps of ``dt``, Utf8 keys ``_d, _t``.

    ``link``: ``(_d, _t, __d_prev, __t_prev, __dur, __dur_prev)`` from ``dt_ramp_link`` (both steps in dt).
    ``carry``: ``(_d, _t, __d_prev, __t_prev, __dur, __dur_prev)`` from ``dt_ramp_carry`` (the predecessor is
    the previous roll's step; its duration is carried), without steps that also have a link.
    """
    dur = d.p_step_duration.frame.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                                         pl.col("value").cast(pl.Float64).alias("__dur"))
    schema = {"_d": pl.Utf8, "_t": pl.Utf8, "__d_prev": pl.Utf8, "__t_prev": pl.Utf8,
              "__dur": pl.Float64, "__dur_prev": pl.Float64}
    link = pl.DataFrame(schema=schema)
    raw = getattr(d, "dt_ramp_link", None)
    if raw is not None and raw.height > 0:
        link = (raw.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                           pl.col("d_previous").cast(pl.Utf8).alias("__d_prev"),
                           pl.col("t_previous").cast(pl.Utf8).alias("__t_prev"))
                .join(dur, on=["_d", "_t"], how="inner")
                .join(dur.rename({"_d": "__d_prev", "_t": "__t_prev", "__dur": "__dur_prev"}),
                      on=["__d_prev", "__t_prev"], how="inner")
                .select(list(schema)))
    carry = pl.DataFrame(schema=schema)
    raw_c = getattr(d, "dt_ramp_carry", None)
    if raw_c is not None and raw_c.height > 0 and getattr(d, "p_roll_continue_flow", None) is not None:
        carry = (raw_c.select(pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
                              pl.col("d_previous").cast(pl.Utf8).alias("__d_prev"),
                              pl.col("t_previous").cast(pl.Utf8).alias("__t_prev"),
                              pl.col("duration_previous").cast(pl.Float64).alias("__dur_prev"))
                 .join(dur, on=["_d", "_t"], how="inner")
                 .join(link.select("_d", "_t"), on=["_d", "_t"], how="anti")
                 .select(list(schema)))
    return link, carry


def _ramp_rows(d, idx_set: pl.DataFrame, ramp_speed: "Param | None", side: str, dir_: str,
               link: pl.DataFrame, carry: pl.DataFrame) -> pl.DataFrame:
    """The rows of one ramp family: ``(p, source, sink, d, t)`` in the dtypes of ``idx_set`` / ``dt`` with
    ``__kind`` (``link`` / ``carry`` / ``reserve``), ``__room`` (ρ), ``__d_prev`` / ``__t_prev`` (Utf8) and
    ``__f_hat`` (the carried flow of ``carry`` rows).  See the EQ. header of :func:`_eq_ramp`.  With
    ``ramp_speed=None`` (the ramp cost, :func:`_eq_ramp_cost`) only the ``link`` and ``carry`` rows, every one of
    them (no room filter), and ``__room`` null."""
    keys = ["p", "source", "sink"]
    base = _utf8_keys(idx_set.select(keys).unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
                      .lazy(), keys + ["d", "t"])
    if ramp_speed is not None:
        speed = ramp_speed.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                        pl.col(side).cast(pl.Utf8).alias("_n"),
                                        pl.col("value").cast(pl.Float64).alias("__speed"))
        base = (base.with_columns(pl.col(f"_{side}").alias("_n"))
                .join(speed.lazy(), on=["_p", "_n"], how="inner"))
    with_link = (base.join(link.lazy(), on=["_d", "_t"], how="inner")
                 .with_columns(__kind=pl.lit("link"), __f_hat=pl.lit(None, pl.Float64)))
    if ramp_speed is None:
        parts = [with_link]
    else:
        parts = [with_link.filter((pl.col("__d_prev") != pl.col("_d"))
                                  | (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0
                                     < 1.0))]
    if carry.height > 0:
        f_hat = d.p_roll_continue_flow.frame.select(
            pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("source").cast(pl.Utf8).alias("_source"),
            pl.col("sink").cast(pl.Utf8).alias("_sink"), pl.col("d").cast(pl.Utf8).alias("_d"),
            pl.col("t").cast(pl.Utf8).alias("_t"), pl.col("value").cast(pl.Float64).alias("__f_hat"))
        parts.append(base.join(carry.lazy(), on=["_d", "_t"], how="inner")
                     .join(f_hat.lazy(), on=["_p", "_source", "_sink", "_d", "_t"], how="inner")
                     .with_columns(__kind=pl.lit("carry")))
    prundt = getattr(d, "prundt", None)
    if ramp_speed is None:
        cols = [*keys, "d", "t", "__kind", "__d_prev", "__t_prev", "__f_hat", pl.lit(None, pl.Float64).alias("__room")]
        return pl.concat([p.select(cols) for p in parts]).collect()
    if prundt is not None and prundt.height > 0:
        # Steps without a predecessor where the arc holds reserve in this direction: at its sink in the same
        # direction, at its source in the opposite one.
        held = prundt.lazy().select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("ud").cast(pl.Utf8).alias("_ud"),
                                    pl.col("n").cast(pl.Utf8).alias("_rn"), pl.col("d").cast(pl.Utf8).alias("_d"),
                                    pl.col("t").cast(pl.Utf8).alias("_t"))
        at_sink = held.filter(pl.col("_ud") == dir_).select("_p", pl.col("_rn").alias("_sink"), "_d", "_t").unique()
        at_source = (held.filter(pl.col("_ud") != dir_)
                     .select("_p", pl.col("_rn").alias("_source"), "_d", "_t").unique())
        lone = base.join(link.lazy().select("_d", "_t"), on=["_d", "_t"], how="anti")
        if carry.height > 0:
            lone = lone.join(carry.lazy().select("_d", "_t"), on=["_d", "_t"], how="anti")
        lone = lone.join(d.p_step_duration.frame.lazy().select(
            pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
            pl.col("value").cast(pl.Float64).alias("__dur")), on=["_d", "_t"], how="inner")
        held_rows = pl.concat([
            lone.join(at_sink, on=["_p", "_sink", "_d", "_t"], how="semi"),
            lone.join(at_source, on=["_p", "_source", "_d", "_t"], how="semi"),
        ]).unique(subset=["_p", "_source", "_sink", "_d", "_t"], keep="first", maintain_order=True)
        parts.append(held_rows
                     .with_columns(__d_prev=pl.lit(None, pl.Utf8), __t_prev=pl.lit(None, pl.Utf8),
                                   __dur_prev=pl.col("__dur"), __kind=pl.lit("reserve"),
                                   __f_hat=pl.lit(None, pl.Float64))
                     .filter(pl.col("__speed") * 60.0 * pl.col("__dur") < 1.0))
    cols = [*keys, "d", "t", "__kind", "__d_prev", "__t_prev", "__f_hat",
            (pl.col("__speed") * 60.0 * (pl.col("__dur") + pl.col("__dur_prev")) / 2.0).alias("__room")]
    return pl.concat([p.select(cols) for p in parts]).collect()


def _source_ramp_factor(d, idx_set: "pl.DataFrame") -> "Param | None":
    """Source-side ramp capacity factor c over ``idx_set × dt``.

    Indirect-unit input arcs (rows of ``p_indirect_input_cap``) take that input limit per unit of
    capacity; every other arc takes ``input_share_max`` keyed ``(p, source)`` (default 1).  ``None``
    when neither applies (factor 1 everywhere, the RHS is left unmultiplied).
    """
    keys = ["p", "source", "sink"]
    share = getattr(d, "p_process_source_input_share_max", None)
    in_cap = getattr(d, "p_indirect_input_cap", None)
    cap_rows = None
    if in_cap is not None and in_cap.frame.height > 0:
        cap_rows = (_utf8_keys(in_cap.frame.lazy(), keys)
                    .join(_utf8_keys(idx_set.lazy(), keys).select([f"_{k}" for k in keys]),
                          on=[f"_{k}" for k in keys], how="semi")
                    .select([f"_{k}" for k in keys] + ["d", "t", pl.col("value").alias("__cap")])
                    .collect())
        if cap_rows.height == 0:
            cap_rows = None
    if cap_rows is None:
        if share is None:
            return None
        return _coef_factor(idx_set, share, ["p", "source"])
    base = _utf8_keys(idx_set.select(keys).unique().join(d.dt.select("d", "t"), how="cross").lazy(), keys)
    if share is not None:
        base = base.join(share.frame.lazy()
                         .select(pl.col("p").cast(pl.Utf8).alias("_p"),
                                 pl.col("source").cast(pl.Utf8).alias("_source"),
                                 pl.col("value").alias("__share")),
                         on=["_p", "_source"], how="left")
    else:
        base = base.with_columns(pl.lit(None, dtype=pl.Float64).alias("__share"))
    frame = (base
             .join(cap_rows.lazy(), on=[f"_{k}" for k in keys] + ["d", "t"], how="left")
             .select(*keys, "d", "t", value=pl.coalesce(pl.col("__cap"), pl.col("__share"), pl.lit(1.0)))
             .collect())
    return Param(("p", "source", "sink", "d", "t"), frame)


def _ramp_rate(d, idx_set: "pl.DataFrame", side: str, rows_subset: "pl.DataFrame", keys: list,
               b: "_Build | None" = None) -> Param:
    """``c · ρ̄`` (or ``c · ρ`` on reserve-only rows) over ``rows_subset`` (:func:`_ramp_rows` output, any
    ``__kind``): the per-row ramp room already scaled by the arc's capacity factor ``c`` (c14c design §3,
    EQ. header of :func:`_eq_ramp`).  A coarse process's time-varying source factor is its coarse step's mean
    (``b`` given: :func:`_coarse_flows.at_first_steps`).  The reserve-room rows of a coarse process
    (:func:`_coarse_flows.reserve_room_rows`) already hold c in their room, min_t ρ_t · c_t, and are not scaled
    again."""
    if b is not None and b.has_coarse_proc and "__kind" in rows_subset.columns:
        fine_rows, coarse_rows = _coarse_flows.split_arcs(b, rows_subset)
        is_room = pl.col("__kind") == "reserve"
        room_rows = coarse_rows.filter(is_room)
        if room_rows.height > 0:
            rest = pl.concat([fine_rows, coarse_rows.filter(~is_room)])
            room = room_rows.select(*keys, "d", "t", value=pl.col("__room").cast(pl.Float64))
            if rest.height > 0:
                other = _ramp_rate(d, idx_set, side, rest, keys, b=b).frame.select(*keys, "d", "t", "value")
                room = pl.concat([other, room.select(*(pl.col(c).cast(other.schema[c]) for c in other.columns))])
            return Param(("p", "source", "sink", "d", "t"), room)
    rate = Param(("p", "source", "sink", "d", "t"), rows_subset.select(*keys, "d", "t", value=pl.col("__room")))
    if side == "sink":
        if d.p_process_sink_max_capacity_coef is not None:
            rate = rate * _coef_factor(idx_set, d.p_process_sink_max_capacity_coef, ["p", "sink"])
    else:
        src_factor = _source_ramp_factor(d, idx_set)
        if b is not None and b.has_coarse_proc:
            src_factor = _coarse_flows.at_first_steps(b, src_factor)
        if src_factor is not None:
            rate = rate * src_factor
    return rate


def _c_param_plain(d, idx_set: "pl.DataFrame", side: str) -> "Param | None":
    """The ramp row's capacity factor ``c`` alone (without ``ρ̄``): :func:`_coef_factor` on the sink side,
    :func:`_source_ramp_factor` on the source side.  ``None`` means 1.0 everywhere."""
    if side == "sink":
        if d.p_process_sink_max_capacity_coef is None:
            return None
        return _coef_factor(idx_set, d.p_process_sink_max_capacity_coef, ["p", "sink"])
    return _source_ramp_factor(d, idx_set)


def _c_values(rows: "pl.DataFrame", c_param: "Param | None", keys: list) -> "pl.DataFrame":
    """``rows`` + ``__c`` (float, default 1.0): the plain capacity factor of each row, from ``c_param``
    (:func:`_c_param_plain`'s output, possibly ``None`` meaning 1.0 everywhere)."""
    if c_param is None:
        return rows.with_columns(__c=pl.lit(1.0))
    cf = c_param.frame
    on = [k for k in ("p", "source", "sink", "d", "t") if k in cf.columns]
    joined = (_utf8_keys(rows.lazy(), on)
              .join(_utf8_keys(cf.lazy(), on).select([f"_{k}" for k in on] + [pl.col("value").alias("__c")]),
                    on=[f"_{k}" for k in on], how="left")
              .with_columns(pl.col("__c").fill_null(1.0))
              .collect())
    return joined.select(rows.columns + ["__c"])


def _ramp_limit_share(b: "_Build", ramp_speed: Param, side: str, points: "pl.DataFrame") -> "pl.DataFrame":
    """``(p, source, sink, d, t, value)`` over ``points`` (``p, source, sink, d, t``, dtypes of ``idx_set`` /
    ``dt``): ``min(1, ml + ½ · ramp_speed · 60 · step_duration[d,t])`` — the startup-step (SU, called with the
    family's own up-speed) or pre-shutdown (SD′, down-speed) output limit of a unit starting / stopping at
    ``points`` (c14c design §1.2/§3.1).  ``ml`` via :func:`flextool.engine.model._online._min_load_share`,
    ``input_side`` per arc (an indirect unit's input arc: ``source != p``, ``sink == p``, ``p`` in
    ``process_indirect``); this row's own ``ramp_speed`` only — no min over sides (unlike ``maxFlow_online``,
    which keeps the tighter of the two).  A point of a coarse process (``t`` = the first fine step of its coarse
    step b) takes the mean share over the whole step, ``_coarse_flows.step_limit_share`` of the step-mean ml and
    ρ_b = ramp_speed · 60 · Δ_b."""
    if b.has_coarse_proc:
        fine_pts, coarse_pts = _coarse_flows.split_arcs(b, points)
        if coarse_pts.height > 0:
            parts = [_coarse_ramp_limit_share(b, ramp_speed, side, coarse_pts)]
            if fine_pts.height > 0:
                parts.insert(0, _ramp_limit_share(b, ramp_speed, side, fine_pts))
            return pl.concat(parts)
    d = b.d
    keys = ["p", "source", "sink"]
    pts = _utf8_keys(points.lazy(), keys + ["d", "t"]).collect()
    if pts.height == 0:
        return pts.select(*keys, "d", "t").with_columns(value=pl.lit(0.0))
    indirect = getattr(d, "process_indirect", None)
    if indirect is not None and indirect.height > 0:
        ind_p = _utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique().collect()
        is_in = ((pts["_source"] != pts["_p"]) & (pts["_sink"] == pts["_p"])
                 & pts["_p"].is_in(ind_p["_p"].to_list()))
    else:
        is_in = pl.Series([False] * pts.height)
    in_rows = pts.filter(is_in)
    out_rows = pts.filter(~is_in)
    ml_parts = []
    if in_rows.height > 0:
        ml_parts.append(_online._min_load_share(b, in_rows.lazy(), input_side=True).collect())
    if out_rows.height > 0:
        ml_parts.append(_online._min_load_share(b, out_rows.lazy(), input_side=False).collect())
    ml = (pl.concat(ml_parts) if ml_parts
          else pl.DataFrame(schema={"_p": pl.Utf8, "_source": pl.Utf8, "_sink": pl.Utf8, "_d": pl.Utf8,
                                    "_t": pl.Utf8, "__ml": pl.Float64}))
    dur = d.p_step_duration.frame.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
        pl.col("value").cast(pl.Float64).alias("__dur"))
    speed = ramp_speed.frame.select(
        pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
        pl.col("value").cast(pl.Float64).alias("__speed"))
    out = (pts.join(ml, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
           .with_columns(pl.col("__ml").fill_null(0.0))
           .join(dur, on=["_d", "_t"], how="inner")
           .with_columns(pl.col(f"_{side}").alias("_n"))
           .join(speed, on=["_p", "_n"], how="inner")
           .with_columns(value=pl.min_horizontal(
               pl.lit(1.0), pl.col("__ml") + 0.5 * pl.col("__speed") * 60.0 * pl.col("__dur")))
           .select(*keys, "d", "t", "value"))
    return out


def _coarse_ramp_limit_share(b: "_Build", ramp_speed: Param, side: str, points: "pl.DataFrame") -> "pl.DataFrame":
    """:func:`_ramp_limit_share` at ``points`` of coarse processes (``t`` = b_first): over the fine steps t of each
    coarse step b, Δ_b = Σ Δt and the Δt-weighted mean ml, then the mean share of a unit starting at the step's
    start (stopping at its end) with ρ_b = ramp_speed · 60 · Δ_b (``_coarse_flows.step_limit_share``)."""
    d = b.d
    keys = ["p", "source", "sink"]
    st = b.coarse_steps.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("d").cast(pl.Utf8).alias("_d"),
                               pl.col("b_first").alias("_b"), pl.col("t").cast(pl.Utf8).alias("__tf"),
                               pl.col("w").alias("__w"))
    pts = (_utf8_keys(points.select(*keys, "d", "t").unique().lazy(), keys + ["d", "t"]).collect()
           .rename({"_t": "_b"})
           .join(st, on=["_p", "_d", "_b"], how="inner")
           .rename({"__tf": "_t"}))
    fine = pts.select(*keys, pl.col("_d").cast(d.dt.schema["d"]).alias("d"),
                      pl.col("_t").cast(d.dt.schema["t"]).alias("t")).unique()
    share = _ramp_limit_share_terms(b, side, fine)
    dur = d.p_step_duration.frame.select(
        pl.col("d").cast(pl.Utf8).alias("_d"), pl.col("t").cast(pl.Utf8).alias("_t"),
        pl.col("value").cast(pl.Float64).alias("__dur"))
    speed = ramp_speed.frame.select(
        pl.col("p").cast(pl.Utf8).alias("_p"), pl.col(side).cast(pl.Utf8).alias("_n"),
        pl.col("value").cast(pl.Float64).alias("__speed"))
    step_rows = (pts.join(share, on=["_p", "_source", "_sink", "_d", "_t"], how="left")
                 .with_columns(pl.col("__ml").fill_null(0.0))
                 .join(dur, on=["_d", "_t"], how="inner")
                 .with_columns((pl.col("__w") * pl.col("__ml")).alias("__ml")))
    return (_coarse_flows.ordered_sum(step_rows, [*keys, "d", "t", "_p", "_source", "_sink"], "_t",
                                      values=("__dur", "__ml"))
            .with_columns(pl.col(f"_{side}").alias("_n"))
            .join(speed, on=["_p", "_n"], how="inner")
            .with_columns(value=_coarse_flows.step_limit_share(pl.col("__ml"),
                                                               pl.col("__speed") * 60.0 * pl.col("__dur")))
            .select(*keys, "d", "t", "value"))


def _ramp_limit_share_terms(b: "_Build", side: str, points: "pl.DataFrame") -> "pl.DataFrame":
    """``(_p, _source, _sink, _d, _t, __ml)``: the min-load share ml of :func:`_ramp_limit_share` at ``points``
    (``p, source, sink, d, t``), input-side per arc."""
    d = b.d
    keys = ["p", "source", "sink"]
    pts = _utf8_keys(points.lazy(), keys + ["d", "t"]).collect()
    indirect = getattr(d, "process_indirect", None)
    if indirect is not None and indirect.height > 0:
        ind_p = _utf8_keys(indirect.lazy().select("p"), ["p"]).select("_p").unique().collect()
        is_in = ((pts["_source"] != pts["_p"]) & (pts["_sink"] == pts["_p"])
                 & pts["_p"].is_in(ind_p["_p"].to_list()))
    else:
        is_in = pl.Series([False] * pts.height)
    parts = []
    if pts.filter(is_in).height > 0:
        parts.append(_online._min_load_share(b, pts.filter(is_in).lazy(), input_side=True).collect())
    if pts.filter(~is_in).height > 0:
        parts.append(_online._min_load_share(b, pts.filter(~is_in).lazy(), input_side=False).collect())
    return (pl.concat(parts) if parts
            else pl.DataFrame(schema={"_p": pl.Utf8, "_source": pl.Utf8, "_sink": pl.Utf8, "_d": pl.Utf8,
                                      "_t": pl.Utf8, "__ml": pl.Float64}))


def _predecessor_points(rows_subset: "pl.DataFrame", d) -> "pl.DataFrame":
    """``(p, source, sink, d, t)`` at the predecessor identity of the rows of ``rows_subset`` (:func:`_ramp_rows`
    link rows: ``__d_prev`` / ``__t_prev``), in the dtypes of ``d.dt``, each point once: at a branch point several
    rows (one per branch) share the predecessor, and a repeated point would repeat its value in every join on it."""
    return rows_subset.select(
        "p", "source", "sink",
        pl.col("__d_prev").cast(d.dt.schema["d"]).alias("d"),
        pl.col("__t_prev").cast(d.dt.schema["t"]).alias("t")).unique(maintain_order=True)


def _branch_point_notice(link_rows: "pl.DataFrame") -> None:
    """Behaviour-change notice naming the units whose downward ramp rows meet a branch point: several rows (one per
    branch) share the predecessor step, where the shut-down share used to be multiplied (:func:`_predecessor_points`
    counts it once)."""
    shared = (link_rows.select(pl.col("p").cast(pl.Utf8), pl.col("source").cast(pl.Utf8),
                               pl.col("sink").cast(pl.Utf8), "__d_prev", "__t_prev")
              .filter(pl.struct("p", "source", "sink", "__d_prev", "__t_prev").is_duplicated()))
    if shared.height > 0:
        from flextool.engine import _behaviour_changes

        _behaviour_changes.notice("ramp_branch_point_shutdown_once", "units", sorted(set(shared["p"].to_list())))


def _attach_at_predecessor(rows_subset: "pl.DataFrame", values: "pl.DataFrame", keys: list) -> "pl.DataFrame":
    """``values`` (``p, source, sink, d, t, value``), evaluated at the predecessor identity, reattached to
    ``rows_subset``'s OWN ``(d, t)`` via its ``__d_prev`` / ``__t_prev`` bridge (Utf8).  0 where ``rows_subset``
    has no match (no predecessor value — the carried-predecessor callers restrict to link rows only)."""
    left = _utf8_keys(rows_subset.lazy(), keys)
    right = (_utf8_keys(values.lazy(), keys + ["d", "t"])
             .select([f"_{k}" for k in keys] + [pl.col("_d").alias("__d_prev"), pl.col("_t").alias("__t_prev"),
                                                "value"]))
    return (left.join(right, on=[f"_{k}" for k in keys] + ["__d_prev", "__t_prev"], how="left")
            .select(*keys, "d", "t", value=pl.col("value").fill_null(0.0)).collect())


def _attach_at_self(rows_subset: "pl.DataFrame", values: "pl.DataFrame", keys: list) -> "pl.DataFrame":
    """``values`` (``p, source, sink, d, t, value``) left-joined onto ``rows_subset``'s own ``(d, t)``, 0
    where absent."""
    left = _utf8_keys(rows_subset.lazy(), keys + ["d", "t"])
    right = _utf8_keys(values.lazy(), keys + ["d", "t"]).select(
        [f"_{k}" for k in keys] + ["_d", "_t", "value"])
    return (left.join(right, on=[f"_{k}" for k in keys] + ["_d", "_t"], how="left")
            .select(*keys, "d", "t", value=pl.col("value").fill_null(0.0)).collect())


def _online_ramp_rhs(b: "_Build", idx_set: "pl.DataFrame", ramp_speed: Param, side: str, up: bool,
                     sub: "pl.DataFrame", v_online_k, v_startup_k, v_shutdown_k, keys: list) -> dict:
    """RHS terms of one UC class's rows of a ramp family, room on the units online split by unit state (c14c
    design §3): ``{"online": ..., "startup": ..., "shutdown": ...}`` (without the ``_{kind}`` suffix the
    caller adds; present only where non-empty).  ``sub`` is this class's rows of :func:`_ramp_rows` (any
    ``__kind``: link / carry / reserve) — one process belongs to at most one UC class.
    """
    d = b.d
    sign = 1.0 if up else -1.0
    terms: dict = {"online": (v_online_k * _ramp_rate(d, idx_set, side, sub, keys, b=b)) * sign}

    flow_kind = sub.filter(pl.col("__kind") != "reserve")
    if flow_kind.height == 0:
        return terms
    link_kind = flow_kind.filter(pl.col("__kind") == "link")
    carry_kind = flow_kind.filter(pl.col("__kind") == "carry")
    c_param = _c_param_plain(d, idx_set, side)
    floor_tbl = _unit_floors.arc_floor_share(d, idx_set)
    if b.has_coarse_proc:
        # A coarse process: c and F are their coarse-step means, keyed at each step's first fine step (b, b⁻).
        c_param = _coarse_flows.at_first_steps(b, c_param)
        if floor_tbl is not None:
            floor_tbl = _coarse_flows.at_first_steps(
                b, Param(("p", "source", "sink", "d", "t"), floor_tbl.select(*keys, "d", "t", "value"))).frame
    c_vals = _c_values(flow_kind, c_param, keys)

    if up:
        # startup: c·(SU_t − ρ̄) — can be 0 or negative (keep the sign, drop exact zeros).
        su = _ramp_limit_share(b, ramp_speed, side, flow_kind.select(*keys, "d", "t"))
        su_coef = (c_vals.join(su.rename({"value": "__su"}), on=[*keys, "d", "t"], how="left")
                   .with_columns(pl.col("__su").fill_null(0.0))
                   .with_columns(value=pl.col("__c") * (pl.col("__su") - pl.col("__room")))
                   .filter(pl.col("value") != 0.0)
                   .select(*keys, "d", "t", "value"))
        if su_coef.height > 0:
            terms["startup"] = v_startup_k * Param(tuple(keys) + ("d", "t"), su_coef)
        # shutdown: −F[d′,t′], link rows only (carry rows: F = 0, no term, design §3.3).
        if link_kind.height > 0 and floor_tbl is not None:
            f_prev = (_attach_at_predecessor(link_kind, floor_tbl, keys)
                      .filter(pl.col("value") > 0.0)
                      .with_columns(value=-pl.col("value")))
            if f_prev.height > 0:
                terms["shutdown"] = v_shutdown_k * Param(tuple(keys) + ("d", "t"), f_prev)
    else:
        # startup: c·ρ̄ + F[d,t] — the row's own step, link and carry rows alike.
        f_self = (_attach_at_self(flow_kind, floor_tbl, keys) if floor_tbl is not None
                 else flow_kind.select(*keys, "d", "t").with_columns(value=pl.lit(0.0)))
        su_coef = (c_vals.join(f_self.rename({"value": "__f"}), on=[*keys, "d", "t"], how="left")
                   .with_columns(pl.col("__f").fill_null(0.0))
                   .with_columns(value=pl.col("__c") * pl.col("__room") + pl.col("__f"))
                   .filter(pl.col("value") != 0.0)
                   .select(*keys, "d", "t", "value"))
        if su_coef.height > 0:
            terms["startup"] = v_startup_k * Param(tuple(keys) + ("d", "t"), su_coef)
        # shutdown: −c·SD′[d′,t′] on link rows (ml-based); −c·SD′_carry on carry rows:
        #   SD′_carry = max(1, f̂ / (c · ĉ)), ĉ = p_roll_continue_online, 1.0 where ĉ = 0 — no stoppers then.
        shutdown_parts = []
        if link_kind.height > 0:
            _branch_point_notice(link_kind)
            sd = _ramp_limit_share(b, ramp_speed, side, _predecessor_points(link_kind, d))
            sd_at_row = _attach_at_predecessor(link_kind, sd, keys)
            c_link = _c_values(link_kind, c_param, keys)
            shutdown_parts.append(
                c_link.join(sd_at_row.rename({"value": "__sd"}), on=[*keys, "d", "t"], how="left")
                .with_columns(pl.col("__sd").fill_null(0.0))
                .with_columns(value=-pl.col("__c") * pl.col("__sd"))
                .select(*keys, "d", "t", "value"))
        if carry_kind.height > 0:
            rco = d.p_roll_continue_online
            rco_df = (rco.frame.select(pl.col("p").cast(pl.Utf8).alias("_p"), pl.col("value").alias("__c_hat"))
                      if rco is not None else pl.DataFrame(schema={"_p": pl.Utf8, "__c_hat": pl.Float64}))
            c_carry = _c_values(carry_kind, c_param, keys)
            shutdown_parts.append(
                _utf8_keys(c_carry.lazy(), ["p"]).join(rco_df.lazy(), on="_p", how="left")
                .with_columns(pl.col("__c_hat").fill_null(0.0))
                .with_columns(__sd=pl.when(pl.col("__c_hat") > 0.0)
                              .then(pl.max_horizontal(
                                  pl.lit(1.0), pl.col("__f_hat") / (pl.col("__c") * pl.col("__c_hat"))))
                              .otherwise(1.0))
                .with_columns(value=-pl.col("__c") * pl.col("__sd"))
                .select(*keys, "d", "t", "value").collect())
        if shutdown_parts:
            sd_all = pl.concat(shutdown_parts).filter(pl.col("value") != 0.0)
            if sd_all.height > 0:
                terms["shutdown"] = v_shutdown_k * Param(tuple(keys) + ("d", "t"), sd_all)
    return terms
