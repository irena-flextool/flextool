"""The objective terms of the core model."""

import polars as pl
from polar_high import Sum, Where, Param
from polar_high.engine import Expr
from flextool.engine.model import _capacity
from flextool.engine.model import _extend
from flextool.engine.model import _retire_built
from flextool.engine.input._axis_enums import align_join_dtypes, rename_to_axis
from flextool.engine.model._common import _has_rows
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.model.core._context import _Build
from flextool.engine.model.core._conversion import _eff_fuel_input, _startup_draw
from flextool.engine.model.core._storage_state import _horizon_end_steps


# ---------------------------------------------------------------------------
# Objective


def _sum_terms(terms: list) -> Expr | None:
    """t₁ + t₂ + … in list order; None for an empty list."""
    total = None
    for term in terms:
        total = term if total is None else total + term
    return total


# Every operational cost term is weighted by b.op_factor:
#   w_op[d,t] = step_duration[d,t] · timestep_weight[d,t] · inflation_op[d] / period_share[d] · branch_weight[d,t]
#   inflation_op[d] (the social discount factor of operating costs) = Σ_y pyr[d,y]·(1 + social_discount_rate)^−(
#     base[d,y] + pyr[d,y]·discount_offset_operations), y the years d represents, pyr their length, base[d,y] the
#     years from the run start to the start of y
# step_duration turns a rate (currency/h) into currency per step, timestep_weight scales a representative step
# to the time it stands for, / period_share scales the modelled part of the period to a full year, and
# inflation_op discounts that year's operating cost; branch_weight is 1 outside stochastic solves.  Terms for point events
# (startups, ramp changes, end-of-horizon valuation) use w_ev[d,t] = w_op[d,t] / step_duration[d,t].
# The commodity ladder (_commodity_ladder) uses the per-period form
# inflation_op[d] / period_share[d] · branch_weight[d], because its purchases are per period.


def _obj_balance_slack(b: _Build) -> Expr:
    """Penalty on the node balance slacks."""
    d = b.d
    vq_up = b.vq_state_up
    vq_down = b.vq_state_down
    op_factor = b.op_factor

    # ─── OBJ. ─── balance_slack_penalty ──────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(n,d,t)} vq_state_up[n,d,t] · penalty_up[n,d,t] · ncs[n,d] · w_op[d,t]
    #   + Σ_{(n,d,t)} vq_state_down[n,d,t] · penalty_down[n,d,t] · ncs[n,d] · w_op[d,t]
    #   ∀ (n,d,t) with balance slacks (nodeBalance ∪ nodeBalancePeriod, penalty_method ≠ off)
    #   penalty is defined on every slack cell: the node's authored value, else the schema default
    #   ncs = node_capacity_for_scaling, factor 1 when the parameter is absent;   currency
    # Prices the energy a node balance cannot supply (up) or absorb (down); penalty in currency/MWh.
    # The slacks enter the balance without ncs, so with node_capacity_for_scaling ≠ 1 the effective
    # penalty per MWh is penalty · ncs (unreachable today; known inconsistency).
    if d.p_node_capacity_for_scaling is not None:
        return (Sum(vq_up   * d.p_penalty_up
                    * d.p_node_capacity_for_scaling * op_factor)
                + Sum(vq_down * d.p_penalty_down
                      * d.p_node_capacity_for_scaling * op_factor))
    return (Sum(vq_up   * d.p_penalty_up   * op_factor)
            + Sum(vq_down * d.p_penalty_down * op_factor))


def _obj_commodity(b: _Build) -> Expr | None:
    """Commodity purchase cost and sales revenue."""
    if not b.has_proc:
        return None
    d = b.d
    v_flow = b.v_flow
    op_factor = b.op_factor
    # Commodities with a price ladder are priced by _commodity_ladder, not by the single commodity price.
    ladder_c = (d.commodity_with_ladder["c"]
                if b.has_ladder and d.commodity_with_ladder is not None else None)

    def _single_price(flows: "pl.DataFrame | None") -> "pl.DataFrame | None":
        if flows is None or ladder_c is None:
            return flows
        return flows.filter(~pl.col("c").is_in(ladder_c))

    buy_eff = _single_price(d.flow_from_commodity_eff)
    buy_noEff = _single_price(d.flow_from_commodity_noEff)
    sell = _single_price(d.flow_to_commodity)
    terms: list = []

    # ─── OBJ. ─── commodity_cost_eff ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c) ∈ flow_from_commodity_eff, (d,t)}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · commodity_price[c,d,t] · w_op[d,t]
    # Cost of the commodity c bought from node s by processes with an efficiency: the input is the
    # output flow times slope, plus the section input per online unit (min_load_efficiency units only,
    # linear and integer online alike).  Price in currency/MWh.
    if _has_rows(buy_eff):
        terms += [Sum(term * d.p_commodity_price * op_factor) for _key, term in _eff_fuel_input(b, buy_eff)]

    # ─── OBJ. ─── commodity_cost_noEff ───────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c) ∈ flow_from_commodity_noEff, (d,t)} v_flow[p,s,k,d,t]·unitsize[p] · commodity_price[c,d,t] · w_op[d,t]
    # Cost of the commodity bought by processes without an efficiency: the input equals the flow.
    if _has_rows(buy_noEff):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, buy_noEff)
            * d.p_commodity_price * op_factor))

    # ─── OBJ. ─── commodity_cost_startup ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,n,c) ∈ startup_from_commodity, (d,t)}
    #     v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p] · commodity_price[c,d,t] · w_ev[d,t]
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight   (as startup_cost)
    # Cost of the start-up energy that units draw from the commodity node n of commodity c (startup_input_method
    # startup_and_input / startup_only; core._startup_draw).  The draw is MWh per start-up event, so the weight
    # has no step_duration.  Commodities with a price ladder are priced by _commodity_ladder.
    if b.has_startup_draw:
        buy_su = _single_price(getattr(d, "startup_from_commodity", None))
        terms += [Sum(term * d.p_commodity_price * b.ev_factor) for _key, term in _startup_draw(b, buy_su)]

    # ─── OBJ. ─── commodity_sales ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,k,c) ∈ flow_to_commodity, (d,t)} v_flow[p,s,k,d,t]·unitsize[p] · commodity_price[c,d,t] · w_op[d,t]
    # Revenue from flows into a commodity node k, sold at the commodity price (negative cost).  Every
    # process arc counts, with or without efficiency: the sink-side flow is the delivered energy.
    if _has_rows(sell) and d.p_commodity_price is not None:
        terms.append(-Sum(
            Where(v_flow * d.p_unitsize, sell)
            * d.p_commodity_price * op_factor))
    return _sum_terms(terms)


def _obj_co2_price(b: _Build) -> Expr | None:
    """CO2 emission cost of processes with an efficiency, net of removal credit."""
    if not b.has_co2_price:
        return None
    d = b.d
    terms: list[Expr] = []

    # ─── OBJ. ─── co2_price_eff ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_priced, (d,t)}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # Cost of the CO2 emitted when processes with an efficiency draw a CO2-bearing commodity c from a
    # node of a CO2-priced group g; content in t/MWh, price in currency/t.  The section term exists
    # for min_load_efficiency units only.
    if _has_rows(d.flow_from_co2_priced):
        terms.extend(Sum(term * d.p_co2_content * d.p_co2_price * b.op_factor)
                     for _key, term in _eff_fuel_input(b, d.flow_from_co2_priced))

    # ─── OBJ. ─── co2_removal_credit ─────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,s,k,c,g) ∈ flow_to_co2_priced, (d,t)}
    #     v_flow[p,s,k,d,t]·unitsize[p] · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # Credit for flows into the group's priced commodity node(s) — removal (e.g. DAC/BECCS modelled
    # as a flow into a CO2-content commodity node of the priced group g).  No slope or section term:
    # the delivered flow is the physical quantity removed, not a fuel input (mirrors flow_to_commodity,
    # the commodity-sale term).  Ported from GMPL's "Receiving credits for removing CO2" (#19, Q14).
    if _has_rows(getattr(d, "flow_to_co2_priced", None)):
        terms.append(-Sum(
            Where(b.v_flow * d.p_unitsize, d.flow_to_co2_priced)
            * d.p_co2_content * d.p_co2_price * b.op_factor))

    # ─── OBJ. ─── co2_price_startup ──────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,n,c,g) ∈ startup_from_co2_priced, (d,t)}
    #     v_startup[p,d,t]·startup_energy[p,n,d]·unitsize[p] · co2_content[c] · co2_price[g,d,t] · w_ev[d,t]
    # Cost of the CO2 of the start-up energy drawn from a node of a CO2-priced group g that carries the
    # CO2-bearing commodity c (core._startup_draw); an event, so the weight w_ev has no step_duration.
    if b.has_startup_draw and _has_rows(getattr(d, "startup_from_co2_priced", None)):
        terms.extend(Sum(term * d.p_co2_content * d.p_co2_price * b.ev_factor)
                     for _key, term in _startup_draw(b, d.startup_from_co2_priced))
    return _sum_terms(terms)


def _obj_co2_price_noEff(b: _Build) -> Expr | None:
    """CO2 emission cost of processes without an efficiency."""
    d = b.d
    if not (_has_rows(d.flow_from_co2_priced_noEff)
            and d.p_co2_content is not None and d.p_co2_price is not None):
        return None

    # ─── OBJ. ─── co2_price_noEff ────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,c,g) ∈ flow_from_co2_priced_noEff, (d,t)}
    #     v_flow[p,s,k,d,t]·unitsize[p] · co2_content[c] · co2_price[g,d,t] · w_op[d,t]
    # As co2_price_eff for processes without an efficiency: the input equals the flow, no slope and
    # no section term.
    return Sum(
        Where(b.v_flow * d.p_unitsize, d.flow_from_co2_priced_noEff)
        * d.p_co2_content * d.p_co2_price * b.op_factor)


def _obj_variable_cost(b: _Build) -> Expr | None:
    """Variable operation and maintenance cost of process flows."""
    if not b.has_proc:
        return None
    d = b.d
    v_flow = b.v_flow
    v_online_lin = b.v_online_linear
    v_online_int = b.v_online_integer
    op_factor = b.op_factor
    terms: list = []

    # ─── OBJ. ─── variable_cost_noEff ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_noEff} v_flow[p,s,k,d,t]·unitsize[p] · varCost[p,s,k,d,t] · w_op[d,t]
    # pssdt_varCost_noEff = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_noEff, varCost[p,s,k,d,t] ≠ 0}
    # O&M cost of the flows of processes without an efficiency; varCost (currency/MWh) is the arc's
    # combined process, source-side and sink-side other_operational_cost (every authored shape: constant,
    # period, time, period × time, stochastic time).
    if (_has_rows(d.pssdt_varCost_noEff)
            and d.p_pssdt_varCost is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_noEff)
            * d.p_pssdt_varCost * op_factor))

    # ─── OBJ. ─── variable_cost_eff_source ───────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_unit_source}
    #     (v_flow[p,s,k,d,t]·unitsize[p]·slope[p,d,t] + v_online[p,d,t]·section[p,d,t]·unitsize[p])
    #     · other_operational_cost_source[p,s,d,t] · w_op[d,t]
    # pssdt_varCost_eff_unit_source = {(p,s,k) ∈ process_source_sink_eff} × dt where
    #     other_operational_cost_source[p,s,d,t] ≠ 0 (every authored shape: constant, period, time,
    #     period × time, stochastic time).
    # O&M cost on the input side of units with an efficiency, priced on the input energy (flow times
    # slope, plus the section input of min_load_efficiency units).  Added like the other O&M terms
    # (GMPL subtracted it).  The sink/source coefficient ratio of the input is not applied (refactor
    # bug 10).
    if (_has_rows(d.pssdt_varCost_eff_unit_source)
            and d.p_pdt_varCost_source is not None):
        flow_term = Where(v_flow * d.p_unitsize * d.p_slope,
                          d.pssdt_varCost_eff_unit_source)
        terms.append(Sum(flow_term * d.p_pdt_varCost_source * op_factor))
        if b.has_minload_eff:
            section_idx = (d.pssdt_varCost_eff_unit_source
                           .join(d.process_min_load_eff, on="p", how="inner"))
            if section_idx.height > 0:
                if b.has_online_lin:
                    terms.append(Sum(
                        Where(Where(v_online_lin, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor))
                if b.has_online_int:
                    terms.append(Sum(
                        Where(Where(v_online_int, d.process_min_load_eff)
                              * d.p_section * d.p_unitsize,
                              section_idx)
                        * d.p_pdt_varCost_source * op_factor))

    # ─── OBJ. ─── variable_cost_eff_sink ─────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_unit_sink} v_flow[p,s,k,d,t]·unitsize[p] · other_operational_cost_sink[p,k,d,t] · w_op[d,t]
    # pssdt_varCost_eff_unit_sink = {(p,s,k) ∈ process_source_sink_eff} × dt where
    #     other_operational_cost_sink[p,k,d,t] ≠ 0 (every authored shape, as above).
    # O&M cost on the output side of units with an efficiency, priced on the output flow.
    if (_has_rows(d.pssdt_varCost_eff_unit_sink)
            and d.p_pdt_varCost_sink is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_unit_sink)
            * d.p_pdt_varCost_sink * op_factor))

    # ─── OBJ. ─── variable_cost_eff_connection ───────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ pssdt_varCost_eff_connection} v_flow[p,s,k,d,t]·unitsize[p] · other_operational_cost[p,d,t] · w_op[d,t]
    # pssdt_varCost_eff_connection = {(p,s,k) ∈ process_source_sink_eff} × dt where
    #     other_operational_cost[p,d,t] ≠ 0 (constant, period, time or period × time).
    # O&M cost of connections with an efficiency, priced on the flow of each direction.
    if (_has_rows(d.pssdt_varCost_eff_connection)
            and d.p_pdt_varCost_process is not None):
        terms.append(Sum(
            Where(v_flow * d.p_unitsize, d.pssdt_varCost_eff_connection)
            * d.p_pdt_varCost_process * op_factor))
    return _sum_terms(terms)


def _obj_startup_cost(b: _Build) -> Expr | None:
    """Startup cost of unit-commitment processes."""
    if not (b.has_startup_cost_lin or b.has_startup_cost_int):
        return None
    d = b.d
    terms: list = []

    # ─── OBJ. ─── startup_cost ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,d,t) ∈ pdt_online_{linear,integer}} v_startup_{linear,integer}[p,d,t] · startup_cost[p,d] · unitsize[p] · w_ev[d,t]
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight
    # Cost of starting up units, startup_cost in currency/MW started.  A startup is an event, so the
    # weight has no step_duration.
    startup_factor = b.ev_factor
    if b.has_startup_cost_lin:
        terms.append(Sum(
            Where(b.v_startup_linear, d.pdt_online_linear)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    if b.has_startup_cost_int:
        terms.append(Sum(
            Where(b.v_startup_integer, d.pdt_online_integer)
            * d.p_startup_cost * d.p_unitsize * startup_factor))
    return _sum_terms(terms)


def _obj_ramp_cost(b: _Build) -> Expr | None:
    """Cost of changing the flow of the arcs with a ramp cost."""
    if b.v_ramp_up is None:
        return None
    d = b.d

    # ─── OBJ. ─── ramp_cost ──────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,s,k,d,t) ∈ R} (v_ramp_up + v_ramp_down)[p,s,k,d,t] · unitsize[p] · rc[p,s,k] · w_ev[d,t]
    #   rc[p,s,k] = p_ramp_cost_arc = Σ_{side ∈ {source s, sink k}, ramp_method[p,side] ∈ {ramp_cost, both}}
    #               ramp_cost[p,side]   (> 0; a direct unit's one arc takes both sides)
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight   (as startup_cost)
    # Cost of changing a flow, ramp_cost in currency per MW of change of the step-average flow, up and down
    # alike.  A change is an event, as a start-up, so the weight has no step_duration: ramping 0 → X → 0 costs
    # 2·rc·X whatever the step lengths.  A start-up from 0 pays both its startup cost and this cost.
    factor = b.ev_factor
    return _sum_terms([Sum(v * d.p_ramp_cost_arc * d.p_unitsize * factor)
                       for v in (b.v_ramp_up, b.v_ramp_down)])


def _obj_storage_reference_price(b: _Build) -> Expr | None:
    """Value of the storage content left at the end of the solve horizon."""
    d = b.d
    if not (b.has_storage
            and d.p_storage_state_reference_price is not None):
        return None
    ends = _horizon_end_steps(b)
    if ends is None:
        return None
    over = (ends
            .join(d.p_storage_state_reference_price.frame.select("n", "d"),
                  on=["n", "d"], how="inner")
            .select("n", "d", "t").unique())
    if over.height == 0:
        return None

    # ─── OBJ. ─── storage_reference_price ────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(n,d,t) ∈ horizon_end, (n,d) ∈ reference_price}
    #     v_state[n,d,t]·state_unitsize[n] · reference_price[n,d] · w_ev[d,t]
    #   horizon_end = nodeState_last_dt ∩ period_last  ∪  block_tail_end   (as storage_state_solve_horizon_
    #                 reference_value: the solve's last period and its branch siblings, and in a rolling solve the
    #                 end of a chained bind_intraperiod_blocks tail that a looping period follows)
    #   w_ev = timestep_weight · inflation_op / period_share · branch_weight
    # Values the energy left in storage at the end of the solve horizon (revenue, hence the minus),
    # reference_price in currency/MWh.  The price comes from use_reference_price nodes or from the
    # storage-price handoff of a previous solve.  An end-of-horizon valuation is a point event, so the
    # weight has no step_duration.  Rows with a zero price are kept.
    ref_price_factor = (d.p_timestep_weight * d.p_inflation_op
                        / d.p_period_share)
    if d.pdt_branch_weight is not None:
        ref_price_factor = ref_price_factor * d.pdt_branch_weight
    return -Sum(
        Where(b.v_state, over) * d.p_state_unitsize
        * d.p_storage_state_reference_price * ref_price_factor)


def _obj_invest_divest(b: _Build) -> Expr | None:
    """Investment and divestment costs of processes and storage nodes."""
    if not (b.has_invest_p or b.has_divest_p or b.has_invest_n or b.has_divest_n
            or b.has_divest_built_p or b.has_divest_built_n or b.has_extend_p or b.has_extend_n):
        return None
    d = b.d
    terms: list = []
    # Under recourse investment every scenario branch carries its own invest/divest variables from the
    # branching period on, so each cost is weighted by the branch probability pd_branch_weight[d] and the
    # term is the expected cost.  Otherwise bw = 1 and no weight is applied.
    recourse = getattr(d, "recourse_invest", False) and d.pd_branch_weight is not None

    def _cost(param: "Param | None", ent: str) -> "Param | None":
        """Entity cost param (e, d) on axis ``ent``, times pd_branch_weight under recourse; None stays None."""
        if param is None:
            return None
        cost = Param((ent, "d"), param.frame.pipe(rename_to_axis, {"e": ent}))
        if not recourse:
            return cost
        f = (cost.frame
             .join(d.pd_branch_weight.frame.rename({"value": "w"}), on="d", how="left")
             .with_columns(value=pl.col("value") * pl.col("w").fill_null(1.0))
             .drop("w"))
        return Param(cost.dims, f)

    def _costs(var: str, *fields: str) -> list["Param | None"]:
        """The cost frames ``fields`` of a solve that has ``var`` investment variables.  The preprocessing gives
        each of them (zero rows where no cost is authored); none at all means the cost derivation did not run for
        the solve, which would leave the investments free in the objective, so it is an error.  (A retirement
        without its frames only loses its saving: no check.)"""
        params = [getattr(d, f, None) for f in fields]
        if all(p is None for p in params):
            raise FlexToolConfigError(
                f"The solve has {var} variables but none of the cost coefficients {', '.join(fields)}: the "
                "decisions would be free in the objective. The cost derivation did not run for this solve; please "
                "report it as a FlexTool bug.")
        return params

    # ─── OBJ. ─── invest_cost_p ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(p,d) ∈ pd_invest_set} v_invest_p[p,d]·unitsize[p] · annual_discounted[p,d] · bw[d]
    #   + Σ_{(p,d) ∈ pd_invest_set} v_invest_p[p,d]·unitsize[p] · lifetime_fixed_cost[p,d] · bw[d]
    #   annual_discounted[p,d] = (a_cap[p,c] − a_salv[p,c]) · Σ_{d′ ∈ L(p,d)} inflation_inv[d′]
    #                                                                       (reinvest_choice / no_reinvestment)
    #   inflation_inv[d] (the social discount factor of investment payments) = Σ_y pyr[d,y]·(1 +
    #     social_discount_rate)^−(base[d,y] + pyr[d,y]·discount_offset_investment) (y, pyr, base as for inflation_op)
    #                          = Σ_{g ∈ gen(p,d)} a_b[g] · Σ_{d′ ∈ L_g(p,d)} inflation_inv[d′]   (reinvest_automatic)
    #   gen(p,d): the chain of the capacity decided in d: g1 online at c(d); g_{k+1} online at y_eol[g_k] =
    #     round_end_of_life(y_on[g_k] + lifetime_technical[p, d_on(g_k)], end_of_life_method[p]) (at least one
    #     period: the next seam when that is at or before y_on[g_k]), until the end of the horizon
    #   L_g(p,d) = the solve's periods d′ on the walk lineage with yr[c] ≤ yr[d′], y_on[g] ≤ yr[d′] < y_eol[g]
    #   a_b[g] = a_cap[g] − a_salv[g] as below, every value read at d_on(g) (the period g comes online in, on d′'s
    #     lineage inside a stochastic fan), S at d_eol(g)
    #   c = c(d) the period the capacity comes online; L(p,d) = the solve's periods d′ with yr[c] ≤ yr[d′] < y_eol[p,d]
    #     (no upper end for reinvest_automatic): the periods the cohort of the decision is in service in, y_eol =
    #     round_end_of_life(yr[c] + lifetime_technical[p,c], end_of_life_method[p]) (engine.md "Capacity symbols")
    #   a_cap[p,c] = A[p,c]·G(n)/G(L) (A[p,c] when n = L); A[p,c] = ann(invest_cost[p,c], discount_rate[p,c], n);
    #     n = payback_time[p,c] (lifetime_technical[p,c] where the entity has none); L = lifetime_technical[p,c];
    #     G(x) = Σ_{k<⌈x⌉} min(1, x−k)·(1+social_discount_rate)^−k
    #   a_salv[p,c] = S·(1+r)^−L·af(r, L), S = 1000·net_salvage_value[p,d_eol] (d_eol the period y_eol starts; the
    #     last period after the last period start), r = discount_rate[p,c]
    #   lifetime_fixed_cost[p,d] = 1000 · Σ_{d′ ∈ L(p,d)} fixed_cost[p,d′]·inflation_op[d′]
    #   ann(c, r, n) = c·1000·af(r, n);  af(r, x) = r / (1 − (1+r)^−x) · (1+r)^(off_inv − 1) (1 / x for r = 0);
    #     off_inv = discount_offset_investment: each payment falls off_inv years into its service year
    # Cost of process capacity decided in period d, currency/MW. Capital: the level yearly cost of a plant of its real
    # lifetime (the present value of the loan, paid over payback_time at discount_rate, spread evenly over
    # lifetime_technical years; the payments are sized for when they fall, so the loan taken when the capacity comes
    # online is worth invest_cost at discount_rate for every off_inv), charged for each year the cohort is in service
    # inside the horizon (discounted per period). end_of_life_method sets the years in service, not the price;
    # payback_time and discount_rate set only the size (with discount_rate = social_discount_rate payback_time changes
    # nothing for a whole number of years). The salvage value at the end of the real lifetime lowers the price: its
    # present value at discount_rate, levelled over the lifetime_technical years (the textbook capital recovery with
    # salvage when payback_time = lifetime_technical). A reinvest_automatic plant is replaced at the end of its life
    # by a plant at the invest cost (and discount_rate, payback_time, lifetime_technical, salvage value) of the period
    # the replacement comes online, and so on to the end of the horizon; each pays its own level yearly cost for its
    # years in service (with constant values the chain charges the first plant's annuity for every period).
    # The inflation-adjusted fixed cost of each period the capacity
    # operates in is charged over the same periods (it stops at the rounded end of life), as for existing capacity.
    # With a construction lead time the capacity operates from its start of operation c(d): both windows start there,
    # and the invest cost, interest rate, payback time and lifetime are those of c(d): their period maps describe the
    # plant as constructed (invest_cost is an overnight cost); without one c(d) = d. No w_op: both factors are already
    # per period. A solve with invest variables and neither cost frame is an error, not a free investment.
    if b.has_invest_p:
        annu, lf = (_cost(p, "p") for p in _costs("v_invest_p", "ed_entity_annual_discounted",
                                                    "ed_lifetime_fixed_cost"))
        if annu is not None:
            terms.append(Sum(b.v_invest_p * d.p_unitsize * annu))
        if lf is not None:
            terms.append(Sum(b.v_invest_p * d.p_unitsize * lf))

    # ─── OBJ. ─── divest_cost_p ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(p,d) ∈ pd_divest_set} v_divest_p[p,d]·unitsize[p] · lifetime_fixed_cost_divest[p,d] · bw[d]
    #   − Σ_{(p,d) ∈ pd_divest_set} v_divest_p[p,d]·unitsize[p] · divest_salvage[p,d] · bw[d]
    #   lifetime_fixed_cost_divest[p,d] = Σ_{d′ ∈ W(p,d)} fixed_cost[p,d′]·1000·inflation_op[d′]
    #   divest_salvage[p,d] = [g1 in service in d]·(D(d)·S_d − a_x[p]·S̄_x[p,d])
    #                         − [g a replacement, m > 0]·set[g,d] + Σ_{replacements g′ ≥ g} a_b[g′]·S̄[g′,d]
    #     (reinvest_automatic: g the generation of p's existing chain in service in d, g1 the existing plant;
    #      m = yr[d] − y_on[g]; set, S̄[g′,d] and a_b as in divest_built_cost (_retire_built), a_b of one plant
    #      built in d_on(g′); other entities: the first term, g1 the existing cohort)
    #   W(p,d) = the solve's periods d′ with yr[d] ≤ yr[d′] < EOL[p] (the window of retired[p,·]: until the
    #     end of life of p's pre-existing capacity, the end of the horizon for reinvest_automatic)
    #   S_d = 1000·net_salvage_value[p,d]; D(d) = (1+social_discount_rate)^−(years from the run start to the start of d)
    #   a_x[p] = S_x·(1+r_x)^−R_x·af(r_x, R_x) (af: invest_cost_p), S_x = 1000·net_salvage_value[p,d_eol] (the
    #     last period's value after the last period start), r_x = discount_rate[p,d0], R_x = y_eol[p] − y0 (y_end[p]
    #     − y0 for a life past the horizon; 0 for a plant that serves no period), y_eol / y_end of g1 for
    #     reinvest_automatic (a_x = 0 without a lifetime); y0 / d0 the start of the run
    #   S̄_x[p,d] = Σ_{d′ ∈ history ∪ solve, y0 ≤ yr[d′] < EOL[p], on d's lineage} inflation_inv[d′]
    # Saving from process capacity retired in period d, currency/MW: the fixed cost of the existing capacity it no
    # longer pays (each period's, with the factor the existing fixed cost has), and its salvage settlement.  The
    # investment in existing capacity is sunk; it holds its salvage value as a level credit a_x per year in service
    # (existing_salvage); a retired MW receives its net salvage value at the start of d and gives back every credit,
    # so it earns exactly S_d·D(d).  A kept plant's credits are counted inside the horizon only.  Existing
    # reinvest_automatic capacity is replaced at its end of life (existing_replacement): retiring it also avoids every
    # replacement still to come, and retiring a replacement settles that plant's contract as it ran (a replacement
    # retired at its own start was never bought).
    if b.has_divest_p:
        lfd = _cost(d.ed_lifetime_fixed_cost_divest, "p")
        salvage = _cost(d.ed_divest_salvage, "p")
        if lfd is not None:
            terms.append(-Sum(b.v_divest_p * d.p_unitsize * lfd))
        if salvage is not None:
            terms.append(-Sum(b.v_divest_p * d.p_unitsize * salvage))

    # ─── OBJ. ─── invest_cost_n ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(n,d) ∈ nd_invest_set} v_invest_n[n,d]·state_unitsize[n] · annual_discounted[n,d] · bw[d]
    #   + Σ_{(n,d) ∈ nd_invest_set} v_invest_n[n,d]·state_unitsize[n] · lifetime_fixed_cost[n,d] · bw[d]
    # As invest_cost_p for storage capacity, currency/MWh of storage.
    if b.has_invest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        annu_n, lf_n = (_cost(p, "n") for p in _costs("v_invest_n", "ed_entity_annual_discounted",
                                                        "ed_lifetime_fixed_cost"))
        if annu_n is not None:
            terms.append(Sum(b.v_invest_n * us_n * annu_n))
        if lf_n is not None:
            terms.append(Sum(b.v_invest_n * us_n * lf_n))

    # ─── OBJ. ─── divest_cost_n ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(n,d) ∈ nd_divest_set} v_divest_n[n,d]·state_unitsize[n] · lifetime_fixed_cost_divest[n,d] · bw[d]
    #   − Σ_{(n,d) ∈ nd_divest_set} v_divest_n[n,d]·state_unitsize[n] · divest_salvage[n,d] · bw[d]
    # As divest_cost_p for storage capacity (W over the window of retired[n,·]).
    if b.has_divest_n:
        us_n = Param(("n",), d.p_state_unitsize.frame)
        lfd_n = _cost(d.ed_lifetime_fixed_cost_divest, "n")
        salvage_n = _cost(d.ed_divest_salvage, "n")
        if lfd_n is not None:
            terms.append(-Sum(b.v_divest_n * us_n * lfd_n))
        if salvage_n is not None:
            terms.append(-Sum(b.v_divest_n * us_n * salvage_n))

    # The retirement of model-built capacity (divest_built_cost_{p,n}): _retire_built.
    terms.extend(_retire_built.objective_terms(b))
    # The life extensions (extend_cost_{p,n}): _extend.
    terms.extend(_extend.objective_terms(b))
    return _sum_terms(terms)


def _obj_existing_fixed_cost(b: _Build) -> None:
    """Fixed cost of existing capacity as an objective constant (opt-in)."""
    d = b.d
    if not (b.include_existing_fixed_cost
            and d.p_ed_fixed_cost is not None
            and b.cap.existing_mw_e is not None):
        return

    # ─── OBJ. ─── existing_fixed_cost ────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(e,d)} existing[e,d] · fixed_cost[e,d] · inflation_op[d] · bw[d]
    #   bw = pd_branch_weight, factor 1 when absent
    # Fixed cost of the capacity that exists in each period, currency (fixed_cost in currency/MW/year).
    # A constant: it moves the objective value but not the solution.  Added only with
    # include_existing_fixed_cost; when pd_branch_weight is present, periods without a weight row drop out.
    # Multiplied by scale_the_objective like every other objective term.
    fc_frame = (
        b.cap.existing_mw_e.frame.rename({"value": "exist"})
        .join(d.p_ed_fixed_cost.frame.rename({"value": "fc"}),
              on=["e", "d"], how="inner")
        .join(d.p_inflation_op.frame.rename({"value": "infl"}),
              on="d", how="inner")
    )
    if d.pd_branch_weight is not None:
        fc_frame = (fc_frame
                    .join(d.pd_branch_weight.frame.rename({"value": "pdbw"}), on="d", how="inner")
                    .with_columns(contrib=pl.col("exist") * pl.col("fc") * pl.col("infl") * pl.col("pdbw")))
    else:
        fc_frame = fc_frame.with_columns(
            contrib=pl.col("exist") * pl.col("fc") * pl.col("infl"))
    fc_const = fc_frame.sort("e", "d")["contrib"].sum()  # fixed summation order (the joins leave it unspecified)
    if fc_const:
        b.m.add_obj_constant(float(fc_const) * b.scale_the_objective)


def _obj_existing_salvage(b: _Build) -> None:
    """The salvage value of existing capacity as a level yearly credit, an objective constant (opt-in)."""
    d = b.d
    credit = getattr(d, "ed_existing_salvage_credit", None)
    if not (b.include_existing_fixed_cost and credit is not None and b.cap.existing_cohort_size_e is not None):
        return

    # ─── OBJ. ─── existing_salvage ───────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # − Σ_{(e,d) ∈ credit} (pre_existing[e,d] − retired_carried[e,d]) · a_x[e]·inflation_inv[d] · bw[d]
    #   pre_existing = p_entity_pre_existing (the existing cohort's size, lifetime-gated, MW, MWh),
    #   retired_carried = p_entity_retired_carried (its retirements realized by earlier solves of a chain)
    #   credit = ed_existing_salvage_credit[e,d] = a_x[e]·inflation_inv[d] over the periods y0 ≤ yr[d] < EOL[e] of
    #     an entity that may retire (a_x: divest_cost_p)
    #   bw = pd_branch_weight, factor 1 when absent
    # The salvage value of existing capacity, received at its end of life, as a level yearly income over its
    # service years from the run start, the same credit as a built plant's (net of the retirements of earlier
    # solves; a retirement in this solve gives its credits back through divest_cost_p).  A constant, added only with
    # include_existing_fixed_cost, multiplied by scale_the_objective like every other objective term.
    exist = _capacity.existing_not_retired(b).rename({"value": "exist"})
    frame, credit_frame = align_join_dtypes(exist, credit.frame.rename({"value": "credit"}), ["e", "d"])
    frame = frame.join(credit_frame, on=["e", "d"], how="inner")
    if d.pd_branch_weight is not None:
        frame, weights = align_join_dtypes(frame, d.pd_branch_weight.frame.rename({"value": "pdbw"}), ["d"])
        frame = (frame.join(weights, on="d", how="inner")
                      .with_columns(contrib=pl.col("exist") * pl.col("credit") * pl.col("pdbw")))
    else:
        frame = frame.with_columns(contrib=pl.col("exist") * pl.col("credit"))
    constant = frame.sort("e", "d")["contrib"].sum()  # fixed summation order (the joins leave it unspecified)
    if constant:
        b.m.add_obj_constant(-float(constant) * b.scale_the_objective)


def _obj_existing_replacement(b: _Build) -> None:
    """The replacement of existing reinvest_automatic capacity at its end of life, an objective constant
    (opt-in)."""
    d = b.d
    price = getattr(d, "ed_existing_replacement", None)
    if not (b.include_existing_fixed_cost and price is not None and b.cap.existing_cohort_size_e is not None):
        return

    # ─── OBJ. ─── existing_replacement ───────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Σ_{(e,d) ∈ price} (pre_existing[e,d] − retired_carried[e,d]) · a_b[g(e,d)]·inflation_inv[d] · bw[d]
    #   price = ed_existing_replacement[e,d] = a_b[g(e,d)]·inflation_inv[d], g(e,d) the replacement of e's existing
    #     capacity in service in d (a generation g ≥ 2 of the existing reinvest_automatic chain: online at the end of
    #     life of the one before, round_end_of_life(y_on + lifetime_technical[e, d_on(g)]), until the horizon end)
    #   a_b[g] = a_cap[g] − a_salv[g] (invest_cost_p) of one plant built in d_on(g) (on d's lineage in a stochastic
    #     fan), its salvage at its end of life; 0 without an invest_cost (and no salvage value)
    #   pre_existing, retired_carried as in existing_salvage; bw = pd_branch_weight, factor 1 when absent
    # The replacement of existing capacity at its end of life, at the invest cost (and the other values) of the period
    # the replacement comes online: no decision changes it except retiring the capacity first, whose retirement
    # coefficient (divest_cost_p) gives the replacement back whether or not this constant is added.  A constant,
    # added only with include_existing_fixed_cost, multiplied by scale_the_objective like every other objective term.
    exist = _capacity.existing_not_retired(b).rename({"value": "exist"})
    frame, price_frame = align_join_dtypes(exist, price.frame.rename({"value": "price"}), ["e", "d"])
    frame = frame.join(price_frame, on=["e", "d"], how="inner")
    if d.pd_branch_weight is not None:
        frame, weights = align_join_dtypes(frame, d.pd_branch_weight.frame.rename({"value": "pdbw"}), ["d"])
        frame = (frame.join(weights, on="d", how="inner")
                      .with_columns(contrib=pl.col("exist") * pl.col("price") * pl.col("pdbw")))
    else:
        frame = frame.with_columns(contrib=pl.col("exist") * pl.col("price"))
    constant = frame.sort("e", "d")["contrib"].sum()  # fixed summation order (the joins leave it unspecified)
    if constant:
        b.m.add_obj_constant(float(constant) * b.scale_the_objective)
