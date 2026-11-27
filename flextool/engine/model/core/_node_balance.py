"""Node balances: per step, per coarse step, per period and per timeset block, and the storage targets an
upper-level solve fixes."""

import polars as pl
from polar_high import Sum, Where, Lag, Param
from polar_high.engine import Var
from flextool.engine.model import _balance_terms
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.model import _self_discharge
from flextool.engine.input._axis_enums import alias_to_axis, rename_to_axis
from flextool.engine.model._common import (
    _entity_rows,
    _has_rows,
    dispatch_periods,
    _var_view,
    d_period_path,
    on_paths,
    path_rows,
    path_values,
)
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model.core._context import _Build
from flextool.engine.model.core._storage_links import _fix_quantity_target, _fix_usage_target


def _eq_node_balance(b: _Build) -> None:
    """Per-step energy balance of balance and storage nodes."""
    m, d = b.m, b.d

    # ─── EQ. ─── nodeBalance_eq ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #       (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t]
    #        + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n) ∈ startup_to_balance} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]   (no Δt: an event)
    #   + back-flow terms of two-way one-variable arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d′,t⁻] − v_state[n,d,t])·state_unitsize[n]
    #   − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·state_unitsize[n]
    #   h = Δt·ln(1 + self_discharge_loss[n,d,t]),  α = h/(1 − e^{−h}) − 1,  β = 1 − h/(e^{h} − 1)
    #   S⁻ = the state the step starts from: v_state[n,d′,t⁻] by the binding method, v_state_rp_start at a
    #        representative block's first step, the fix_start start state / the roll carry at an open chain's
    #        first step, v_state[n,d,t] where the chain is open with neither
    #   + start terms of open-start nodes (fix_start in the first solve, the roll carry in continuation solves)
    #   = −inflow[n,d,t]
    #   ∀ (n,d,t) ∈ nodeBalance_dt, n ∉ nodeStateBlock ∪ nodeCoarse;   Δt = step_duration[d,t]
    #   section[p,d,t] = min_load·(1/eta_min − 1/eta) / (1 − min_load)
    #   slope[p,d,t] = 1/eta − section[p,d,t]   (= (1/eta − min_load/eta_min) / (1 − min_load))
    #     for min_load_efficiency units (eta = efficiency, default 1; min_load = 0 where not authored;
    #     eta_min = efficiency_at_min_load, else efficiency); slope = 1/eta and no section for the other units
    #   on a direct (one-input, one-output) unit, slope and section above are further scaled by
    #     k_sink/k_src (slope) and 1/k_src (section) — the unit's output/input conversion_flow_coeff
    #     (default 1 each; a zero coefficient keeps its uncapped-edge meaning instead, unscaled)
    # Energy conservation per node and time step, in MWh.  Storage nodes add
    # the state change; the previous step t⁻ depends on storage_binding_method.
    # nodeStateBlock nodes are balanced per timeset block (nodeBalanceBlock_eq),
    # nodeCoarse nodes per coarse step (nodeBalanceCoarse_eq).
    # The section term exists for min_load_efficiency units only.  The start-up
    # term (core._startup_draw) is the start-up energy of the units with a
    # start-up input node n (startup_input_method startup_and_input /
    # startup_only), MWh per start-up event; a coarse process starts at the
    # first fine step of its coarse step.  The flow, start-up and slack terms
    # are b.node_balance_flow_terms.  Self-discharge: the
    # content decays continuously at ln(1 + loss) per hour; flows within a
    # step are spread evenly over the step.  Exact for any step length while
    # the loss is constant within the step; a coarse step / block with a
    # time-varying loss uses h_b = Σ Δt·λ_t, exact for the idle decay only
    # (#152).  The decay is applied to a weighted average of the start and
    # the end state (model/_self_discharge.py); the model folds it into the
    # state change as
    # (S⁻·(1 − β) − S·(1 + α))·state_unitsize.
    over = b.nodeBalance_dt
    if b.has_nsb:
        over = over.join(d.nodeStateBlock, on="n", how="anti")
    if b.has_coarse:
        over = over.join(d.nodeCoarse, on="n", how="anti")
    m.add_cstr(
        "nodeBalance_eq",
        over      = over,
        sense     = "==",
        lhs_terms = _node_balance_terms(b),
        rhs_terms = {"neg_inflow": -d.p_inflow},
    )


def _node_balance_terms(b: _Build) -> dict:
    """Every LHS term of nodeBalance_eq, (n, d, t) in MWh: flows, slacks and, for storage nodes, the state
    change with its self-discharge, and the start / roll-continuation terms.  nodeBalanceCoarse_eq and
    nodeBalanceBlock_eq sum them over each coarse step / block."""
    terms = dict(b.node_balance_flow_terms)
    if b.has_storage:
        terms.update(_state_change_terms(b))
        terms.update(_fix_start_terms(b))
        terms.update(_roll_continue_terms(b))
        terms.update(_open_start_loss_terms(b))
    return terms


def _sum_over_coarse_step(term, step_of: pl.DataFrame):
    """``term`` (n, d, t) summed over the fine steps t of each coarse step: (n, d, b_first).  A constant
    (``Param``) term is summed in its frame (sorted before the group-by, so the sum order is fixed)."""
    if isinstance(term, Param):
        frame = (term.frame
                 .join(step_of, on=["n", "d", "t"], how="inner")
                 .sort("n", "d", "b_first", "t")
                 .group_by(["n", "d", "b_first"], maintain_order=True)
                 .agg(pl.col("value").sum())
                 .select("n", "d", "b_first", "value"))
        return Param(("n", "d", "b_first"), frame)
    return Sum(Where(term, step_of), over=("t",))


def _eq_node_balance_coarse(b: _Build) -> None:
    """Energy balance per coarse step of the nodes of coarse new_stepduration groups."""
    if not b.has_coarse:
        return
    m, d = b.m, b.d
    step_of = d.node_coarse_step

    # ─── EQ. ─── nodeBalanceCoarse_eq ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{t ∈ b} [ Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #       (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n) ∈ startup_to_balance} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]
    #   + back-flow terms of two-way one-variable and DC arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d,t⁻] − v_state[n,d,t])·state_unitsize[n] − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·su[n]
    #   + start terms of open-start nodes ]
    #   = − Σ_{t ∈ b} inflow[n,d,t]
    #   ∀ n ∈ nodeCoarse, (n,d,b) ∈ node_coarse_step;   t ∈ b: (n,d,b,t) ∈ node_coarse_step;   Δt = step_duration[d,t]
    #   t⁻, start and roll terms per storage_binding_method exactly as in nodeBalance_eq
    #   a coarse process (processCoarse) has one v_flow per coarse step of its nodes: its fine terms read the same
    #   column, so its leg is v_flow[p,·,·,d,b]·unitsize·Σ_{t∈b} Δt (·slope_t on an efficiency source)
    #   α, β, S⁻ as in nodeBalance_eq with h_b = Σ_{t∈b} Δt·ln(1 + loss[n,d,t]), Δ_b = Σ_{t∈b} Δt: α spread over
    #   the fine steps by Δt/Δ_b, β at the first fine step only (the decay of the coarse step's own state over the
    #   coarse step; the flow weighting assumes the loss constant within the coarse step)
    # nodeBalance_eq summed over the fine steps of each coarse step of the node's group: every flow stays on the
    # solve's steps and each coarse leg sums its fine flows, so an arc between a fine and a coarse node, or
    # between two coarse nodes, conserves energy at both ends.  The node may shift energy freely within a coarse
    # step.  With the state constant within the coarse step (stateConstantCoarse_eq) the state terms telescope to
    # v_state[n,d,b⁻] − v_state[n,d,b] (b⁻ = the coarse step before b by the node's binding method), so storage
    # binding methods, fix_start and the roll carry apply to coarse nodes as to fine ones.  Same orientation and
    # units as nodeBalance_eq: its dual is the node's price (currency/MWh) over the coarse step.
    lhs = {key: _sum_over_coarse_step(term, step_of) for key, term in _node_balance_terms(b).items()}
    inflow = _sum_over_coarse_step(-d.p_inflow, step_of)
    rhs = {"neg_inflow": inflow} if inflow.frame.height > 0 else {}
    m.add_cstr(
        "nodeBalanceCoarse_eq",
        over      = step_of.select("n", "d", "b_first").unique(maintain_order=True),
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )

    # ─── EQ. ─── stateConstantCoarse_eq ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − v_state[n,d,t_previous] = 0
    #   ∀ (n,d,t,t_previous) ∈ node_coarse_interior   (n ∈ nodeCoarse ∩ nodeState; t_previous: the step before t
    #   within its coarse step)
    # A coarse storage node keeps one state per coarse step.
    if b.has_storage and _has_rows(d.node_coarse_interior):
        interior = d.node_coarse_interior
        state_prev = _var_view(b.v_state, "__prev", {"t": "t_previous"})
        steps = interior.select("n", "d", "t")
        m.add_cstr(
            "stateConstantCoarse_eq",
            over      = steps,
            sense     = "==",
            lhs_terms = {"state":      Where(b.v_state, steps),
                         "state_prev": -Sum(Where(state_prev, interior), over=("t_previous",))},
            rhs_terms = {},
        )


def _node_balance_flow_terms(b: _Build) -> dict:
    """Flow and slack terms of nodeBalance_eq, (n, d, t) in MWh.

    The flow terms come from ``_balance_terms.balance_flow_terms``, which the non-anticipativity storage tie
    shares.  Also used by nodeBalancePeriod_eq: period nodes are never storage, so they take only these terms.
    """
    d = b.d
    terms: dict = {"slack_up": b.vq_state_up * d.p_step_duration,
                   "slack_down": -b.vq_state_down * d.p_step_duration}
    terms.update(_balance_terms.balance_flow_terms(b))
    return terms


def _state_change_terms(b: _Build) -> dict:
    """State-change terms of nodeBalance_eq, one per storage_binding_method, with the self-discharge folded in
    (_weighted_change).

    The balance reads (state[t⁻] − state[t]) + sink − source = −inflow, i.e. state[t] − state[t⁻] = sink − source +
    inflow.  The branches and their lags are _self_discharge.state_links (each node is in one binding set only,
    _validate_storage_binding_disjoint): bind_within_timeblock wraps within the timeblock, bind_forward_only crosses
    period boundaries and has no state change at the solve's first step, bind_within_solve wraps the solve's first
    step to its last (with a stochastic fan in the first period, each root forecast's first step to the last step of
    its own path: dtttdt_within_solve), bind_within_period wraps each period's first step to the period's own last
    step (in a split realization roll the lag crosses the seam: each branch period's first step continues from the
    label's last jump step, the label's first step from the last step of one branch period), bind_intraperiod_blocks
    (nodeStateBlock) follows dtttdt_block_lag (these terms reach nodeBalanceBlock_eq only, which sums them over
    each block), and the blended-weights nodes chain inside each representative block whose
    first step starts from the free state v_state_rp_start.
    """
    d, v_state = b.d, b.v_state
    terms: dict = {}
    for link in _self_discharge.state_links(d, has_rp=b.has_rp, has_nsb=b.has_nsb):
        if link.cross_period:
            prev = _state_lag_cross_period(v_state, link.lag)
        else:
            prev = Lag(v_state, link.lag, "t", link.col)
        now = v_state if link.all_steps else Where(v_state, link.lag.select("d", "t").unique())
        terms[link.key] = _weighted_change(b, Where(prev, link.nodes), Where(now, link.nodes))
    if b.has_rp:
        nodes = d.nodeState_rp
        first_dt = d.rp_block_first.select("d", "t").unique()
        now = Where(Where(v_state, first_dt), nodes)
        start = Where(b.v_state_rp_start, nodes)
        terms["state_change_rp_start"] = _weighted_change(b, start, now)
    return terms


def _weighted_change(b: _Build, prev, now):
    """(prev·(1 − β) − now·(1 + α))·state_unitsize: the state change of a step with its self-discharge
    (_self_discharge); (prev − now)·state_unitsize when no storage node has a loss."""
    su = b.d.p_state_unitsize
    loss = b.self_discharge
    if loss is None:
        return (prev - now) * su
    return (prev * loss.keep_prev - now * loss.keep_now) * su


def _state_lag_cross_period(v_state: Var, lag_frame: pl.DataFrame):
    """v_state at (d_previous, t_previous_within_solve) of each (d, t) in
    ``lag_frame``; unlike ``Lag``, the previous step may be in another period."""
    v_state_back = _var_view(v_state, "__back", {"d": "d_back", "t": "t_back"},
                             to_axis=True)
    lag = lag_frame.select("d", "t",
                           alias_to_axis("d_previous", "d_back"),
                           alias_to_axis("t_previous_within_solve", "t_back"))
    return Sum(Where(v_state_back, lag), over=("d_back", "t_back"))


def _cyclic_nodes(b: _Build) -> "pl.DataFrame | None":
    """Storage nodes whose chain closes on itself within the solve (_self_discharge.cyclic_nodes)."""
    return _self_discharge.cyclic_nodes(b.d)


def _open_start_nodes(b: _Build) -> "pl.DataFrame | None":
    """Storage nodes whose chain starts open at the solve's first step (_self_discharge.open_start_nodes)."""
    return _self_discharge.open_start_nodes(b.d)


def _keep_prev_values(b: _Build, frame: pl.DataFrame) -> pl.DataFrame:
    """``frame`` (n, d, t, value) × (1 − β) of its step: a start energy weighted as the step's previous state."""
    loss = b.self_discharge
    if loss is None:
        return frame
    return (frame.join(loss.keep_prev.frame.rename({"value": "_keep"}), on=["n", "d", "t"], how="left")
            .with_columns(value=pl.col("value") * pl.col("_keep").fill_null(1.0))
            .select("n", "d", "t", "value"))


def _fix_start_terms(b: _Build) -> dict:
    """nodeBalance_eq start terms of open-start nodes (_open_start_nodes) with fix_start."""
    # ─── fix_start terms (in nodeBalance_eq) ─────────────────────────────────
    #   − v_state[n,d,t]·(1 + α[n,d,t])·state_unitsize[n]
    #   + (1 − β[n,d,t])·state_start[n]·(existing[n,d] + invested[n,d]·su[n]
    #                                    − retired[n,d]·su[n])
    #   ∀ (n,d,t) ∈ nodeState_first_dt, n ∈ open-start nodes ∩ storage_fix_start, n ∉ carried
    # The first step of a node with no realised state from before the solve (carried: the nodes of
    # roll_continue_state; so the first solve of the model and of each nested level), where the state change is
    # omitted, starts from the start state (MWh); α, β: the self-discharge weights (_self_discharge, 0 without a
    # loss).  Other fix_start nodes are pinned by storage_state_start_binding_cyclic_period (cyclic) or
    # rp_inter_period_start_fix (representative-period nodes: the start of their base-period chain).
    d = b.d
    first = _self_discharge.fix_start_first(d)
    if first is None:
        return {}
    nodes = first.select("n").unique()
    start, su = d.p_state_start, d.p_state_unitsize
    loss = b.self_discharge
    state = Where(b.v_state, first)
    terms: dict = {"fwd_fix_state": -(state * su) if loss is None else -(state * loss.keep_now * su)}
    # A sparse Param term combines with the dense −inflow by outer join with
    # 0-fill, so rows outside ``first`` keep their balance.
    start_energy = ((start * b.cap.existing_state_n).frame
                    .join(first, on=["n", "d"], how="inner")
                    .select("n", "d", "t", "value"))
    if start_energy.height > 0:
        terms["fwd_fix_existing_pos"] = Param(("n", "d", "t"), _keep_prev_values(b, start_energy))
    if b.has_invest_n:
        invested = _var_view(b.v_invest_n, "__at_fwd_fix", {"d": "d_invest"},
                             to_axis=True)
        alive = (_entity_rows(b.cap.alive, "e", nodes, "n")
                 .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if alive.height > 0:
            term = Sum(Where(invested * start * su, alive), over=("d_invest",))
            terms["fwd_fix_invest_pos"] = term if loss is None else term * loss.keep_prev
    if b.has_divest_n:
        divested = _var_view(b.v_divest_n, "__at_fwd_fix", {"d": "d_divest"},
                             to_axis=True)
        active = (_entity_rows(b.cap.retired_frame("n"), "n", nodes, "n")
                  .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if active.height > 0:
            term = Sum(Where(divested * start * su, active), over=("d_divest",))
            terms["fwd_fix_divest_neg"] = -(term if loss is None else term * loss.keep_prev)
    built = _retire_built.retired(b, "n")
    if built is not None:
        rows = (_entity_rows(_retire_built.retired_frame(b, "n"), "n", nodes, "n")
                .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if rows.height > 0:
            term = Where(built * start * su, rows)
            terms["fwd_fix_divest_built_neg"] = -(term if loss is None else term * loss.keep_prev)
    ext = _extend.extended(b, "n")
    if ext is not None:
        # The extended storage capacity in service (_extend): a capacity term, as an investment's.
        rows = (_entity_rows(_extend.extended_frame(b, "n"), "n", nodes, "n")
                .join(first.select("n", "d", "t"), on=["n", "d"], how="inner"))
        if rows.height > 0:
            term = Where(ext * start * su, rows)
            terms["fwd_fix_extended_pos"] = term if loss is None else term * loss.keep_prev
    return terms


def _roll_continue_terms(b: _Build) -> dict:
    """nodeBalance_eq start terms of a solve that continues from a realised state (the carry)."""
    # ─── roll-continuation terms (in nodeBalance_eq) ─────────────────────────
    #   − v_state[n,d,t]·(1 + α[n,d,t])·state_unitsize[n] + (1 − β[n,d,t])·roll_continue_state[n]
    #   ∀ (n,d,t) ∈ nodeState_first_dt, n ∈ carried, n ∉ cyclic nodes ∪ nodeState_rp
    # A node continues from the latest state realised before this solve at its own or a lower nesting level (the
    # previous roll, the previous solve of a sequence, the dispatch below a storage solve; roll_continue_state, MWh,
    # routed by the cascade: solve._carry_routing).  α, β: the self-discharge weights of an open-start node
    # (_self_discharge).  A cyclic node (_cyclic_nodes) cycles within the solve and takes no carry (how_to/solve_sequences.md
    # rolling notes); a bind_within_solve node of a rolling roll that continues from a carried state is bind_forward_only
    # in that roll (_run_model._rolling_storage._anchor_cyclic_in_rolling) and takes the carry here, its horizon end in
    # storage_state_roll_anchor / _close; a representative-period node starts each block from v_state_rp_start (no carry).  Nodes of
    # neither kind outside the open-start set take the unweighted form.
    d = b.d
    first = _self_discharge.roll_first(d)
    if first is None:
        return {}
    su = d.p_state_unitsize
    loss = b.self_discharge
    handed_over = (d.p_roll_continue_state.frame
                   .join(first, on="n", how="inner")
                   .select("n", "d", "t", "value"))
    open_start = _open_start_nodes(b)
    if loss is None or open_start is None:
        terms: dict = {"roll_continue_state": -(Where(b.v_state, first) * su)}
        if handed_over.height > 0:
            terms["roll_continue_value"] = Param(("n", "d", "t"), handed_over)
        return terms
    weighted = first.join(open_start, on="n", how="semi")
    plain = first.join(open_start, on="n", how="anti")
    terms = {}
    if weighted.height > 0:
        terms["roll_continue_state"] = -(Where(b.v_state, weighted) * loss.keep_now * su)
    if plain.height > 0:
        terms["roll_continue_state_rp"] = -(Where(b.v_state, plain) * su)
    if handed_over.height > 0:
        value = pl.concat([_keep_prev_values(b, handed_over.join(weighted.select("n"), on="n", how="semi")),
                           handed_over.join(weighted.select("n"), on="n", how="anti")], how="vertical")
        terms["roll_continue_value"] = Param(("n", "d", "t"), value)
    return terms


def _open_start_loss_terms(b: _Build) -> dict:
    """nodeBalance_eq self-discharge of an open chain's first step that starts from neither fix_start nor the
    carry (_self_discharge.open_first): the step has no state change, its state is constant over the step, so the
    loss is h·v_state·su (h = α + β)."""
    loss = b.self_discharge
    if loss is None:
        return {}
    first = _self_discharge.open_first(b.d)
    if first is None:
        return {}
    return {"self_discharge_open": -(Where(b.v_state, first) * loss.h * b.d.p_state_unitsize)}


def _eq_node_balance_period(b: _Build) -> None:
    """Period energy balance of balance_within_period nodes."""
    d = b.d
    if not _has_rows(d.nodeBalancePeriod) or d.period_in_use_set is None:
        return

    # ─── EQ. ─── nodeBalancePeriod_eq ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_t w[d,t] · (flow, start-up and slack terms of nodeBalance_eq)[n,d,t]  =  −Σ_t w[d,t] · inflow[n,d,t]
    #   ∀ n ∈ nodeBalancePeriod, n ∉ nodeState, d ∈ periods of dt;   w = timestep_weight
    #   split realization roll (d_period_path): the row on a branch period p_f sums the cells d ∈ {p, p_f} (p its
    #   base period: the jump, then the tail), each at its own weight; no row on p, inflow likewise
    # Energy balance over a whole period instead of every step, in MWh.  The
    # terms are b.node_balance_flow_terms (Δt already applied); period nodes
    # are never storage, so there are no state terms.
    # Every term, slacks included, carries the timestep weight: a per-step
    # balance needs no weight, but a period sum without it would let
    # low-weight timeslices offset high-weight ones.  The other op_factor
    # parts (inflation_op, period_share) are per period and cancel in a row.
    # Inflow is energy per step (no Δt), as in nodeBalance_eq.
    # Path rows (split realization roll, d_period_path): the row of a branch period p_f balances the path, the
    # cells of its base period p (the jump) and of p_f (the tail); p has no row.
    over = (d.nodeBalancePeriod.select("n").lazy()
            .join(dispatch_periods(d).lazy(), how="cross"))
    if _has_rows(d.nodeState):
        over = over.join(d.nodeState.select("n").lazy(), on="n", how="anti")
    over = over.select("n", "d").collect()
    if over.height == 0:
        return
    lhs = {key: Sum(term * d.p_timestep_weight, over=("t",))
           for key, term in b.node_balance_flow_terms.items()}
    inflow_period = Param(("n", "d"), ((d.p_inflow * d.p_timestep_weight).frame.lazy()
                                       .join(d.nodeBalancePeriod.select("n").lazy(), on="n", how="inner")
                                       # Fixed summation order (the join leaves row order unspecified).
                                       .sort("n", "d", "t")
                                       .group_by("n", "d", maintain_order=True)
                                       .agg(pl.col("value").sum())
                                       .select("n", "d", "value")
                                       .collect()))
    if (path := d_period_path(b, "nodeBalancePeriod_eq")) is not None:
        over, lhs = path_rows(over, path), {key: on_paths(term, path) for key, term in lhs.items()}
        inflow_period = Param(("n", "d"), path_values(inflow_period.frame, ("n",), path))
    b.m.add_cstr(
        "nodeBalancePeriod_eq",
        over      = over,
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = {"neg_inflow": -inflow_period},
    )


def _eq_node_balance_fix_quantity(b: _Build) -> None:
    """Final storage state fixed by an upper-level solve."""
    d = b.d

    # ─── EQ. ─── node_balance_fix_quantity_eq_lower ──────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] · state_unitsize[n]
    #   = Σ_{(d_up,d) ∈ period_branch, (d,t,t_up) ∈ dtt_timeline_matching} fix_storage_quantity[n,d_up,t_up]
    #     (storage_roll_end(fix_quantity)[n] instead, where it is given)
    #   ∀ n ∈ n_fix_storage_quantity, d ∈ period_last, (n,d,t) ∈ nodeState_last_dt
    # Pins the storage state (MWh) at the last step of each last period to the
    # state an upper-level (coarser) solve reached at the matching time.  Where
    # the horizon ends inside an upper step, the cascade restates the target as
    # the upper level interpolated at that instant (the upper solve's level is
    # linear within its step; _run_model._nested_storage._nested_end_rows), not the step's end.
    # Despite the name, the row is an equality.
    target = _fix_quantity_target(b)
    if target is None:
        return
    over = target.select("n", "d", "t").unique()
    b.m.add_cstr(
        "node_balance_fix_quantity_eq_lower",
        over      = over,
        sense     = "==",
        lhs_terms = {"state": Where(b.v_state, over) * d.p_state_unitsize},
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _eq_storage_usage_fix(b: _Build) -> None:
    """Storage usage per period capped by an upper-level solve."""
    d = b.d
    target = _fix_usage_target(b)
    if target is None:
        return

    # ─── EQ. ─── node_storage_usage_fix_le ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,n) ∈ process_source_sink, t} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    # + Σ_{(p,n,k) ∈ process_source_sink_eff, t}
    #     (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    # + Σ_{(p,n,k) ∈ process_source_sink_noEff, t} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   ≤ Σ_{(d_up,d) ∈ period_branch, t_up : ∃t (d,t,t_up) ∈ dtt_timeline_matching} fix_storage_usage[n,d_up,t_up]
    #     (storage_roll_end(usage_cap)[n] instead, where it is given)
    #   ∀ n ∈ n_fix_storage_usage, d ∈ period_last, (n,d,t_last) ∈ nodeState_last_dt;   Δt = step_duration[d,t]
    #   and, in a roll, ∀ (n,d) ∈ storage_roll_end(usage_cap_period): d a period of the roll before its horizon's
    #     last one, RHS that cumulative cap
    # Caps the net energy drawn from storage node n over a whole period (MWh)
    # by the usage an upper-level (coarser) solve planned for the coarse
    # windows overlapping that period.  One row per period, at its last step;
    # each coarse window counts once however many fine steps it covers.  The
    # cascade restates the cap where that differs from the cumulative rule
    # (_run_model._nested_storage._nested_end_rows): the upper usage from the period's start in
    # the solve instance to the horizon end, each window pro rata to its hours
    # in that span, less the usage the instance's earlier rolls realised in the
    # period — so a rolling lower level uses at most what the upper planned.
    # The section term exists for min_load_efficiency units only.  The
    # process sink/source coefficient ratio is not applied (nor in the
    # source-side variable cost); it is 1 in every current dataset.
    nodes = d.n_fix_storage_usage.select("n")

    def _arcs_at_node(arcs: pl.DataFrame, side: str) -> pl.DataFrame:
        """Rows of ``arcs`` whose ``side`` node is in n_fix_storage_usage, with n = that node."""
        return (arcs.join(nodes.pipe(rename_to_axis, {"n": side}), on=side, how="inner")
                .with_columns(n=pl.col(side)))

    v_flow = b.v_flow
    over_arcs_t = ("p", "source", "sink", "t")
    lhs: dict = {}
    into_node = _arcs_at_node(d.process_source_sink, "sink")
    if into_node.height > 0:
        lhs["sink_flow"] = -Sum(Where(v_flow * d.p_unitsize, into_node) * d.p_step_duration, over=over_arcs_t)
    from_node_eff = (_arcs_at_node(d.process_source_sink_eff, "source")
                     if _has_rows(d.process_source_sink_eff) else None)
    if from_node_eff is not None and from_node_eff.height > 0 and d.p_slope is not None:
        lhs["source_eff"] = Sum(Where(v_flow * d.p_unitsize * d.p_slope, from_node_eff) * d.p_step_duration,
                                over=over_arcs_t)
        if b.has_minload_eff and d.p_section is not None:
            section_idx = from_node_eff.join(d.process_min_load_eff, on="p", how="inner")
            if section_idx.height > 0:
                for key, has_online, v_online in (("source_section_lin", b.has_online_lin, b.v_online_linear),
                                                  ("source_section_int", b.has_online_int, b.v_online_integer)):
                    if has_online:
                        lhs[key] = Sum(Where(Where(v_online, d.process_min_load_eff) * d.p_section * d.p_unitsize,
                                             section_idx) * d.p_step_duration,
                                       over=over_arcs_t)
    from_node_noEff = (_arcs_at_node(d.process_source_sink_noEff, "source")
                       if _has_rows(d.process_source_sink_noEff) else None)
    if from_node_noEff is not None and from_node_noEff.height > 0:
        lhs["source_noEff"] = Sum(Where(v_flow * d.p_unitsize, from_node_noEff) * d.p_step_duration,
                                  over=over_arcs_t)
    if not lhs:
        return
    b.m.add_cstr(
        "node_storage_usage_fix_le",
        over      = target.select("n", "d", "t").unique(),
        sense     = "<=",
        lhs_terms = lhs,
        rhs_terms = {"target": Param(("n", "d", "t"), target)},
    )


def _eq_block_coupling(b: _Build) -> None:
    """bind_intraperiod_blocks storage nodes (nodeStateBlock), balanced per timeset block."""
    if not (b.has_nsb and b.has_storage):
        return
    m, d = b.m, b.d
    nsb = d.nodeStateBlock
    v_state = b.v_state
    # nodeBalance_eq skips these nodes: without blocks they would have no energy balance at all.
    # An empty dtttdt_block_lag is legitimate: a roll whose only block step is a chain start (a period held in
    # part with one step at the solve's first step) has no lag row; that step starts from the open-start terms.
    if not (_has_rows(d.period_block) and _has_rows(d.period_block_time)):
        names = ", ".join(sorted(str(n) for n in nsb["n"].to_list()))
        raise FlexToolConfigError(f"bind_intraperiod_blocks node(s) {names} have no timeset blocks in this solve")

    # ─── EQ. ─── stateConstantWithinBlock_eq ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_state[n,d,t] − v_state[n,d,t_previous] = 0
    #   ∀ n ∈ nodeStateBlock, (d,t,t_previous) ∈ dtttdt_block_interior
    # Keeps the state of a nodeStateBlock node constant within each timeset
    # block.  Interior steps never cross a period boundary.
    if _has_rows(d.dtttdt_block_interior):
        interior = d.dtttdt_block_interior
        m.add_cstr(
            "stateConstantWithinBlock_eq",
            over      = nsb.join(interior.select("d", "t"), how="cross"),
            sense     = "==",
            lhs_terms = {"state":     Where(v_state, nsb),
                         "state_lag": -Where(Lag(v_state, interior, "t", "t_previous"), nsb)},
            rhs_terms = {},
        )

    # ─── EQ. ─── nodeBalanceBlock_eq ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{t ∈ b} [ Σ_{(p,s,n) ∈ flow_to_n} v_flow[p,s,n,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_eff}
    #       (v_flow[p,n,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])·Δt
    #   − Σ_{(p,n,k) ∈ flow_from_nodeBalance_noEff} v_flow[p,n,k,d,t]·unitsize[p]·Δt
    #   − Σ_{(p,n) ∈ startup_to_balance} v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p]
    #   + back-flow terms of two-way one-variable and DC arcs
    #   + (vq_state_up[n,d,t] − vq_state_down[n,d,t])·Δt
    #   + (v_state[n,d′,t⁻] − v_state[n,d,t])·state_unitsize[n]   ((d′,t⁻) of (d,t) in dtttdt_block_lag; for a
    #     fix_start_end node in block_forward_lag)
    #   − (α[n,d,t]·v_state[n,d,t] + β[n,d,t]·S⁻[n,d,t])·state_unitsize[n]
    #   + start terms of open-start nodes (fix_start in the first solve, the roll carry in a continuation solve) ]
    #   = − Σ_{t ∈ b} inflow[n,d,t]
    #   ∀ n ∈ nodeStateBlock, (d,b) ∈ period_block;   t ∈ b by period_block_time;   Δt = step_duration[d,t]
    #   α, β, S⁻ as in nodeBalanceCoarse_eq with the block b (h_b = Σ_{t∈b} Δt·ln(1 + loss[n,d,t])); S⁻ at the
    #   block's first step is the previous block's state
    # nodeBalance_eq summed over each timeset block of a bind_intraperiod_blocks node.  The state is constant within
    # a block (stateConstantWithinBlock_eq), so the state terms telescope to v_state[b⁻] − v_state[b]: v_state of a
    # block is its state at the end of the block.  b⁻ is the previous block of the period, the first block of a
    # period continuing from the period's last block (the blocks of a period form a loop).  In a rolling solve the
    # blocks also end at the roll's realized end; a period the roll holds whole loops as above, while in a period
    # the roll holds only in part (period_block_chain) the blocks chain forward: the period's first block continues
    # from the previous step of the solve, and at the solve's first step the chain starts from fix_start (first
    # roll), the carried level (later rolls) or, with neither, with no state change.  A fix_start_end node ignores
    # the loop (its start and end are pinned separately): its blocks chain forward in every period like that
    # (_self_discharge.block_forward_nodes / block_forward_lag).  Same orientation and units as nodeBalance_eq: its
    # dual is the node's price (currency/MWh) over the block.
    terms = _node_balance_terms(b)
    step_of = (nsb.select("n").join(d.period_block_time.select("d", "b_first", "t"), how="cross")
               .select("n", "d", "b_first", "t"))
    lhs = {key: _sum_over_coarse_step(term, step_of) for key, term in terms.items()}
    inflow = _sum_over_coarse_step(-d.p_inflow, step_of)
    rhs = {"neg_inflow": inflow} if inflow.frame.height > 0 else {}
    m.add_cstr(
        "nodeBalanceBlock_eq",
        over      = nsb.select("n").join(d.period_block.select("d", "b_first"), how="cross").select("n", "d", "b_first"),
        sense     = "==",
        lhs_terms = lhs,
        rhs_terms = rhs,
    )
