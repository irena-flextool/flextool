import pandas as pd


def generic(par, s, v, r, debug):
    if debug:
        results = []
        df = pd.concat([par.inflation_factor_operations_yearly, par.inflation_factor_investment_yearly], axis=1)
        df.columns = ["operations discount factor","investments discount factor"]
        df.columns.name = "param"
        results.append((df, 'discountFactors_d_p'))

        df = par.entity_annuity
        results.append((df, 'entity_annuity_d_p'))

        return results


def cost_summaries(par, s, v, r, debug):
    """Cost summaries for periods and timesteps"""

    results = []

    # Common calculations
    discount_ops = par.inflation_factor_operations_yearly
    discount_invs = par.inflation_factor_investment_yearly
    period_share = par.complete_period_share_of_year
    to_millions = 1000000

    # 1. Costs at timestep level (non-annualized)
    costs_dt = pd.DataFrame(index=s.dt_realize_dispatch, dtype=float)
    costs_dt.columns.name = 'category'
    costs_dt['commodity_cost'] = r.cost_commodity_dt.sum(axis=1)
    costs_dt['commodity_sales'] = r.sales_commodity_dt.sum(axis=1)
    costs_dt['co2'] = r.cost_co2_dt
    costs_dt['other operational'] = r.cost_process_other_operational_cost_dt.sum(axis=1)
    costs_dt['starts'] = r.cost_startup_dt.sum(axis=1)
    costs_dt['upward slack penalty'] = r.costPenalty_node_state_upDown_dt.xs('up', level='upDown', axis=1).sum(axis=1)
    costs_dt['downward slack penalty'] = r.costPenalty_node_state_upDown_dt.xs('down', level='upDown', axis=1).sum(axis=1)
    costs_dt['inertia slack penalty'] = r.costPenalty_inertia_dt.sum(axis=1)
    costs_dt['non-synchronous slack penalty'] = r.costPenalty_non_synchronous_dt.sum(axis=1)
    try:
        costs_dt['upward reserve slack penalty'] = r.costPenalty_reserve_upDown_dt.xs('up', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['upward reserve slack penalty'] = 0
    try:
        costs_dt['downward reserve slack penalty'] = r.costPenalty_reserve_upDown_dt.xs('down', level='updown', axis=1).sum(axis=1)
    except KeyError:
        costs_dt['downward reserve slack penalty'] = 0

    results.append((costs_dt, 'costs_dt_p'))

    # 2. Annualized, inflation adjusted and years represented (derived from costs_dt)
    dispatch_costs_pure_period = costs_dt.groupby(level='period').sum()
    dispatch_costs_annualized_period = dispatch_costs_pure_period.div(period_share, axis=0) / to_millions
    dispatch_costs_inflation_adjusted = dispatch_costs_annualized_period.mul(discount_ops, axis=0)

    # 3. Discounted and inflation adjusted (with years represented) investment costs (d_realize_invest only)
    investment_costs = pd.DataFrame(index=s.d_realize_invest, dtype=float)
    investment_costs.columns.name = 'category'
    investment_costs['unit investment & retirement'] = (r.costInvestUnit_d + r.costDivestUnit_d) / to_millions
    investment_costs['connection investment & retirement'] = (r.costInvestConnection_d + r.costDivestConnection_d) / to_millions
    investment_costs['storage investment & retirement'] = (r.costInvestState_d + r.costDivestState_d) / to_millions
    investment_costs['fixed cost pre-existing'] = r.costFixedPreExisting_d / to_millions
    investment_costs['fixed cost invested'] = r.costFixedInvested_d / to_millions
    investment_costs['fixed cost reduction of divestments'] = r.costFixedDivested_d / to_millions
    investment_costs['capacity margin penalty'] = r.costPenalty_capacity_margin_d / to_millions

    # Annualize back: Remove inflation adjustment and years represented
    annual_invest_costs = investment_costs.div(discount_invs, axis=0)
    annual_invest_costs['fixed cost pre-existing'] = investment_costs['fixed cost pre-existing'].div(discount_ops, axis=0)
    annual_invest_costs['fixed cost invested'] = investment_costs['fixed cost invested'].div(discount_ops, axis=0)
    annual_invest_costs['fixed cost reduction of divestments'] = investment_costs['fixed cost reduction of divestments'].div(discount_ops, axis=0)
    annual_invest_costs['capacity margin penalty'] = investment_costs['capacity margin penalty'].div(discount_ops, axis=0)

    # 4. Combined summary (investment + dispatch aggregated to period)
    all_periods = s.d_realized_period.union(s.d_realize_invest)
    summary = pd.DataFrame(index=all_periods, dtype=float)
    summary.columns.name = 'parameter'

    # Without inflation and years (so, pure annual results)
    summary_annualized = annual_invest_costs.join(dispatch_costs_annualized_period)
    results.append((summary_annualized, 'annualized_costs_d_p'))

    # With years_represented adjusted with inflation (same as model)
    summary_inflation_years = investment_costs.join(dispatch_costs_inflation_adjusted)
    results.append((summary_inflation_years, 'costs_discounted_d_p'))

    # With years_represented adjusted with inflation (same as model)
    summary_inflation_years = investment_costs.join(dispatch_costs_inflation_adjusted)
    results.append((summary_inflation_years.sum(axis=0), 'costs_discounted_p_'))

    return results


def CO2(par, s, v, r, debug):
    """Annualized CO2 Mt for groups by period"""
    results = []

    # Calculate CO2 emissions in Mt
    total_co2 = ((r.emissions_co2_d * par.years_represented_d) / 1000000).sum(axis=0)
    co2_summary = pd.DataFrame(index=["CO2 [Mt]"], columns=["model_wide"], data=total_co2)
    co2_summary.index.name = 'param_CO2'
    results.append((co2_summary, 'CO2__'))

    # Process co2 emissions
    process_co2 = r.process_emissions_co2_d.groupby(['period']).sum()
    results.append((process_co2, 'process_co2_d_eee'))

    # Group co2 emissions
    results.append((r.group_co2_d, 'CO2_d_g'))
    return results
