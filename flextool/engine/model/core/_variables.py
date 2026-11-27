"""The decision variables of the core model and the expressions more than one family reads."""

import polars as pl
from polar_high import Sum, Where
from flextool.engine.model import _capacity
from flextool.engine.model import _coarse_flows
from flextool.engine.model import _commodity_ladder
from flextool.engine.model import _dc_power_flow
from flextool.engine.model import _exact
from flextool.engine.model import _group_slack
from flextool.engine.model import _online
from flextool.engine.model import _reserve
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.model import _unit_floors
from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.model._common import _connection_procs, _has_rows, _utf8_keys, _var_view
from flextool.engine.model.core._context import _Build
from flextool.engine.model.core._node_balance import _node_balance_flow_terms
from flextool.engine.model.core._ramp import _ramp_cost_rows


# ---------------------------------------------------------------------------
# Variables


def _declare_variables(b: _Build) -> None:
    """Declare every decision variable; the order fixes the LP column order."""
    m, d = b.m, b.d

    # ─── VAR. ─── v_flow ─────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow[p,source,sink,d,t] ≥ 0   ∀ (p,source,sink,d,t) ∈ pss_dt
    # Flow on a process arc in units of unitsize: MW = v_flow · unitsize[p].
    # Input arcs of indirect units without invest/divest variables carry a
    # per-element upper bound (their capacity is a constant); maxFlow caps
    # every other arc.
    # ─── VAR. ─── v_flow (coarse processes) ──────────────────────────────────
    # v_flow[p,s,k,d,b] ≥ 0   ∀ p ∈ processCoarse, (p,s,k) arcs, (d,b) the coarse steps of p (b = first fine step)
    # The flow of a coarse process: one rate per coarse step, the same on its fine steps.  b.v_flow is indexed
    # on every fine step (_coarse_flows.alias): a fine row of a coarse process reads its step's column.
    if b.has_proc:
        upper = b.flow_var_upper if b.flow_var_upper is not None else float("inf")
        b.v_flow = _coarse_flows.alias(
            b, m.add_var("v_flow", ("p", "source", "sink", "d", "t"), _coarse_flows.domain(b, b.pss_dt),
                         lower=0.0, upper=upper),
            b.pss_dt)

    # ─── VAR. ─── v_ramp_up / v_ramp_down ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_ramp_up[p,s,k,d,t] ≥ 0, v_ramp_down[p,s,k,d,t] ≥ 0   ∀ (p,s,k,d,t) ∈ R
    #   R = the rows of ramp_change_eq (_ramp_cost_rows): (p,s,k) ∈ process_source_sink_ramp_cost with a ramp
    #   predecessor at (d,t) (dt_ramp_link or dt_ramp_carry; for a coarse process t = b, its coarse step)
    # The increase and the decrease of v_flow from the ramp predecessor, in units of unitsize (as v_flow).  Priced
    # equally by the ramp cost, so at most one of them is > 0 at an optimum (up + down = |Δv_flow|).  Declared
    # only when some arc has a ramp cost.
    if b.has_proc and b.has_ramp_cost:
        b.ramp_cost_rows = _ramp_cost_rows(b)
        if b.ramp_cost_rows.height > 0:
            r_idx = b.ramp_cost_rows.select("p", "source", "sink", "d", "t")
            b.v_ramp_up = m.add_var("v_ramp_up", ("p", "source", "sink", "d", "t"), r_idx, lower=0.0)
            b.v_ramp_down = m.add_var("v_ramp_down", ("p", "source", "sink", "d", "t"), r_idx, lower=0.0)

    # ─── VAR. ─── vq_state_up / vq_state_down ────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # vq_state_up[n,d,t], vq_state_down[n,d,t] ≥ 0
    #   ∀ n ∈ nodeBalance ∪ nodeBalancePeriod, (d,t) ∈ dt
    # Balance slack in MW (shortfall / surplus); enters the balance times
    # step_duration and is penalised in the objective.  Nodes with
    # penalty_method = off have none, so their balance is a hard equality.
    slack_idx = _balance_slack_index(b)
    b.vq_state_up = m.add_var("vq_state_up", ("n", "d", "t"), slack_idx, lower=0.0)
    b.vq_state_down = m.add_var("vq_state_down", ("n", "d", "t"), slack_idx, lower=0.0)

    # ─── VAR. ─── v_state ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] ≥ 0   ∀ (n,d,t) ∈ nodeState_dt
    # Storage state in units of the node's state unitsize:
    # MWh = v_state · state_unitsize[n].  Its upper bound depends on
    # invest/divest and is the maxState row.
    if b.has_storage:
        b.v_state = m.add_var("v_state", ("n", "d", "t"), b.nodeState_dt, lower=0.0)

    # ─── VAR. ─── v_state_inter / v_state_rp_start ───────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state_inter[n,d,b] ≥ 0        ∀ n ∈ nodeState_rp, (d,b) ∈ rp_base_period
    # v_state_rp_start[n,d,t] ≥ 0     ∀ n ∈ nodeState_rp, (d,t) ∈ rp_block_first
    # Representative-period storage, in state units: the state at the end of
    # base period b of period d, and the free starting state of each
    # representative block.  Every period has its own base periods (two
    # periods may share base-period names).  Upper bounds are the rows
    # rp_inter_period_max_state and maxState_rp_start.
    if b.has_rp:
        b.v_state_inter = m.add_var("v_state_inter", ("n", "d", "b"),
                                    d.nodeState_rp.join(d.rp_base_period_set, how="cross").select("n", "d", "b"),
                                    lower=0.0)
        b.v_state_rp_start = m.add_var("v_state_rp_start", ("n", "d", "t"), b.rp_start_idx, lower=0.0)

    # Unit commitment (v_online, v_startup, v_shutdown): declared by _online.
    _online.add_variables(b)

    # ─── VAR. ─── v_invest_p / v_divest_p / v_invest_n / v_divest_n ──────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_invest_p[p,d] ≥ 0 ∀ (p,d) ∈ pd_invest_set;  v_divest_p[p,d] ≥ 0 ∀ (p,d) ∈ pd_divest_set
    # v_invest_n[n,d] ≥ 0 ∀ (n,d) ∈ nd_invest_set;  v_divest_n[n,d] ≥ 0 ∀ (n,d) ∈ nd_divest_set
    # Capacity built / retired in period d, as a count of unitsize (process)
    # or state unitsize (storage node).  Only pre-existing capacity can be retired:
    # pd/nd_divest_set hold the periods where the entity has some (before its end of life).  Upper bounds are the
    # maxInvest_var_bound / maxDivest_var_bound rows.
    if b.has_invest_p:
        b.v_invest_p = m.add_var("v_invest_p", ("p", "d"), d.pd_invest_set)
    if b.has_divest_p:
        b.v_divest_p = m.add_var("v_divest_p", ("p", "d"), d.pd_divest_set)
    if b.has_invest_n:
        b.v_invest_n = m.add_var("v_invest_n", ("n", "d"), d.nd_invest_set)
    if b.has_divest_n:
        b.v_divest_n = m.add_var("v_divest_n", ("n", "d"), d.nd_divest_set)
    # Retirement of model-built capacity (v_divest_built, v_divested_built): declared by _retire_built.
    _retire_built.add_variables(b)
    # Life extension (v_extend, v_extended): declared by _extend.
    _extend.add_variables(b)

    # Reserve variables (v_reserve, vq_reserve, v_reserve_total): declared by _reserve.
    _reserve.add_variables(b)

    # ─── VAR. ─── v_flow_back ────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_flow_back[p,source,sink,d,t] ≥ 0   ∀ (p,source,sink) ∈ twoway1var_arcs, (d,t) ∈ dt
    # Reverse flow (sink → source) of method_2way_1var_off and DC power-flow
    # arcs, in units of unitsize; the signed flow is v_flow − v_flow_back.
    # One variable serves both the balance and dc_flow_eq.  A coarse process has one per coarse step, as v_flow.
    if _has_rows(b.twoway1var_arcs):
        back_idx = b.twoway1var_arcs.join(d.dt, how="cross").select("p", "source", "sink", "d", "t")
        if back_idx.height > 0:
            b.v_flow_back = _coarse_flows.alias(
                b, m.add_var("v_flow_back", ("p", "source", "sink", "d", "t"), _coarse_flows.domain(b, back_idx),
                             lower=0.0),
                back_idx)

    # Direction switch of exact connections (v_exact_forward), DC power-flow angles (v_angle) and
    # commodity-ladder trades (v_trade): declared by their feature modules.
    _exact.add_variables(b)
    _dc_power_flow.add_variables(b)
    _commodity_ladder.add_variables(b)

    # Group soft-floor slacks (vq_capacity_margin, vq_inertia, vq_non_synchronous): declared by _group_slack.
    _group_slack.add_variables(b)


def _balance_slack_index(b: _Build) -> pl.DataFrame:
    """(n, d, t) of the balance slacks: nodeBalance_dt ∪ nodeBalancePeriod × dt,
    without penalty_method = off nodes."""
    d = b.d
    idx = b.nodeBalance_dt
    if _has_rows(d.nodeBalancePeriod):
        period_dt = (d.nodeBalancePeriod.lazy()
                     .join(d.dt.lazy(), how="cross")
                     .select("n", "d", "t")
                     .collect())
        ref_schema = b.nodeBalance_dt.schema
        for col in ("n", "d", "t"):
            if period_dt.schema[col] != ref_schema[col]:
                period_dt = period_dt.with_columns(
                    pl.col(col).cast(ref_schema[col], strict=False))
        idx = pl.concat([b.nodeBalance_dt, period_dt]).unique()
    penalty_off = getattr(d, "nodeBalance_penalty_off", None)
    if _has_rows(penalty_off):
        off_n = penalty_off.select("n")
        if off_n.schema["n"] != idx.schema["n"]:
            off_n = off_n.with_columns(
                pl.col("n").cast(idx.schema["n"], strict=False))
        idx = idx.join(off_n, on="n", how="anti")
    return idx


# ---------------------------------------------------------------------------
# Shared expressions


def _shared_expressions(b: _Build) -> None:
    """Expressions read by more than one equation family or objective term."""
    d = b.d
    _reserve_coupling(b)
    b.node_balance_flow_terms = _node_balance_flow_terms(b)
    b.cap = _capacity.build_capacity(b)
    # Weight of every operational cost term per (d, t): step_duration ·
    # timestep_weight · inflation_op / period_share, times the branch
    # weight in stochastic solves.
    b.op_factor = (d.p_step_duration * d.p_timestep_weight
                   * d.p_inflation_op / d.p_period_share)
    if d.pdt_branch_weight is not None:
        b.op_factor = b.op_factor * d.pdt_branch_weight
    # Weight of event cost terms (start-ups, ramp changes, start-up energy): timestep_weight · inflation_op /
    # period_share, times the branch weight; no step_duration (an event is not a rate).
    b.ev_factor = d.p_timestep_weight * d.p_inflation_op / d.p_period_share
    if d.pdt_branch_weight is not None:
        b.ev_factor = b.ev_factor * d.pdt_branch_weight


def _reserve_coupling(b: _Build) -> None:
    """Σ_r v_reserve aggregates that share process capacity with v_flow."""
    v_reserve = b.v_reserve
    if v_reserve is None:
        return
    pruna = b.d.process_reserve_upDown_node_active          # (p, r, ud, n)
    if not _has_rows(pruna):
        return
    # v_reserve is in units of unitsize, like v_flow.  Upward reserve held
    # at a sink competes with the flow into it (maxFlow, maxOutputSum,
    # profile_flow_upper_limit); downward reserve at a sink lowers the
    # floor of profile_flow_lower_limit.
    up_sink = (pruna.filter(pl.col("ud") == "up")
               .pipe(rename_to_axis, {"n": "sink"})
               .select("p", "r", "ud", "sink"))
    if up_sink.height > 0:
        at_sink = _var_view(v_reserve, "__at_sink", {"n": "sink"}, to_axis=True)
        b.reserve_up_to_sink = Sum(Where(at_sink, up_sink), over=("r", "ud"))
        b.reserve_up_sink_parts = (at_sink, up_sink)
        # maxFlow and maxFlow_online couple the upward reserve of units only: a connection's reserve is limited
        # by its net position at the node (connection_reserve_* rows, _reserve).  Hand-built data without
        # process_unit keeps the process-generic coupling (_common.connection_procs is then None).
        if _connection_procs(b.d) is None:
            b.reserve_up_to_sink_units = b.reserve_up_to_sink
        else:
            units = b.d.process_unit.select(pl.col("p").cast(pl.Utf8).alias("__p")).unique()
            unit_sink = (up_sink.with_columns(pl.col("p").cast(pl.Utf8).alias("__p"))
                         .join(units, on="__p", how="semi").drop("__p"))
            if unit_sink.height > 0:
                b.reserve_up_to_sink_units = Sum(Where(at_sink, unit_sink), over=("r", "ud"))

    down_sink = (pruna.filter(pl.col("ud") == "down")
                 .pipe(rename_to_axis, {"n": "sink"})
                 .select("p", "r", "ud", "sink"))
    if down_sink.height > 0:
        at_sink_dn = _var_view(v_reserve, "__at_sink_dn", {"n": "sink"},
                               to_axis=True)
        b.reserve_down_to_sink = Sum(Where(at_sink_dn, down_sink),
                                     over=("r", "ud"))

    # Source-side reserve of units (consume more / consume less), per arc and converted to the arc's
    # v_flow: factor 1 where v_flow is the input (an indirect unit's input arc, a sink-less unit), 1/slope
    # on a direct input→output arc.  Down at the source raises the flow (maxFlow, maxFlow_online,
    # maxInputFlow_online, profile_flow_upper_limit); up at the source lowers it
    # (profile_flow_lower_limit; the floors in _unit_floors).  Connections are not coupled.
    unit_arcs = _unit_floors.all_unit_arcs(b.d)
    if unit_arcs is None:
        return
    at_source = _var_view(v_reserve, "__at_source", {"n": "source"}, to_axis=True)
    down_flow, down_eff = _unit_floors.increasing_source_selectors(b.d, unit_arcs)
    _, up_flow, up_eff = _unit_floors._reducing_reserve_selectors(b.d, unit_arcs)
    down_arcs = []
    for target, sel, direct, key in ((b.reserve_down_at_source, down_flow, False, "reserve_down_source"),
                                     (b.reserve_down_at_source, down_eff, True, "reserve_down_source_eff"),
                                     (b.reserve_up_at_source, up_flow, False, "reserve_up_source"),
                                     (b.reserve_up_at_source, up_eff, True, "reserve_up_source_eff")):
        if sel is None:
            continue
        arcs = _source_reserve_arcs(unit_arcs, sel, direct=direct)
        if arcs.height == 0:
            continue
        term = Where(at_source, sel)
        if direct:
            term = term * _unit_floors._inverse_slope(b.d, sel)
        target[key] = Where(Sum(term, over=("r", "ud")), arcs)
        if target is b.reserve_down_at_source:
            down_arcs.append(arcs)
    if down_arcs:
        b.reserve_down_input_arcs = pl.concat(down_arcs).unique(maintain_order=True)


def _source_reserve_arcs(unit_arcs: pl.DataFrame, sel: pl.DataFrame, *, direct: bool) -> pl.DataFrame:
    """The unit arcs ``(p, source, sink)`` of the class of ``sel`` (direct input→output arcs, or arcs whose
    v_flow is the input) whose ``(p, source)`` carries a reserve of ``sel``."""
    keys = ["p", "source", "sink"]
    a = _utf8_keys(unit_arcs.lazy(), keys)
    a = a.filter(pl.col("_source") != pl.col("_p"))
    a = a.filter((pl.col("_sink") != pl.col("_p")) if direct else (pl.col("_sink") == pl.col("_p")))
    hit = _utf8_keys(sel.lazy().select("p", "source"), ["p", "source"]).select("_p", "_source").unique()
    return a.join(hit, on=["_p", "_source"], how="semi").select(keys).collect()
