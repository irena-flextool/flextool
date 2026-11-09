# Investment and multi-year models

##  How to make investments (storage/unit) 
**(examples.sqlite scenario: wind_battery_invest)**
***init - west - wind - battery - battery_invest***

Here we will use the previous battery scenario to represent the investment options in the tool.

The solve will invest only if it has an array of `invest_periods` set, telling the periods where it is allowed to make investment decisions. In a multi solve investment model (rolling investments) it can be useful to separately set `realized_invest_periods` so that the investment results get reported only from the chosen periods from each solve (in order to avoid overlapping reporting of investment decisions that are replaced by  investment decisions in later solves). Furthermore, `realized_periods` will define the solves and periods from which the dispatch results are output into results.

Each period's investments should be realized by only one solve: do not list the same period in the `realized_invest_periods` of two solves. If you do, FlexTool logs a warning naming both solves and the period. Dispatch timesteps, on the other hand, may be realized by more than one solve (for example an investment solve and a dispatch solve nested inside it that both have the same period in `realized_periods`). For the overlapping timesteps the solve that runs later is kept and the earlier solve's results are not reported. FlexTool logs one warning per pair of solves and period, giving the number of overlapping timesteps. Timesteps that only the earlier solve realizes are still reported from the earlier solve.

First, the investment parameters need to be included both for the *battery_inverter* and *battery* entities:

- `invest_method` - the modeller has options to limit the investment and retirement. Options are *not_allowed*, invest, retire or invest and retire. These have the sub options of no limit *invest_no_limit*, limit the amount per period: *invest_period*, limit the total amount invested *invest_total* or limit both the total investment and the investment per period *invest_period_total*. 

  - If the investment is limited, it requires the corresponding maximum/minimum investment (power [MW] or energy [MWh]) to the virtual capacity. This can be set to individual entities or groups of entities:
    
    - `invest_max_total`
    - `invest_max_period`
    - `invest_min_total`
    - `invest_min_period`
    - `retire_max_total`
    - `retire_max_period`
    - `retire_min_total`
    - `retire_min_period`
 
- `invest_cost` - overnight investment cost new capacity [currency/kW] for the *battery_inverter* and [currency/kWh] for the *battery*. Other one can be left empty or zero, since they will be tied together in the next phase. Here we will assume a fixed relation between kW and kWh for this battery technology, but for example flow batteries could have separate investments for storage and charging capacities.
- `discount_rate` - discount rate (WACC) [e.g. 0.05 means 5%] for the technology that reflects the financing cost and risk. Should be nominal when model `inflation_rate` > 0, real when `inflation_rate` = 0.
- `lifetime` - technical lifetime of the technology to calculate investment annuity (together with the discount rate)

Additional parameters:

- `lifetime_method`: Model can either be forced to reinvest when the lifetime ends (`reinvest_automatic`), have a choice (`reinvest_choice`), or be restricted to a one-shot build that retires after its lifetime without rebuild (`no_investment` — for physical one-off additions like plant refurbishments).
- `construction_lead_time` [years]: time from the investment decision until the capacity starts to operate. The start is rounded to the start of a period by `construction_lead_time_method` (below), on the periods of the whole model (all its solves). Constant or period (a map may set a different lead time per decision period). Default 0. Applies to `unit`, `connection` and `node` investments, deterministic and stochastic alike.
  - **Terms.** The *investment decision* is taken in a period of the solve's `invest_periods`. The capacity is *in construction* from the decision period until it *comes online* at the *start of operation*; from then on it is in service. Without a lead time, the capacity comes online in the decision period.
  - **Results.** In the capacity results (`unit_capacity__d`, `connection_capacity__d`, `node_capacity__d`) *invested* is the capacity that comes online in the period and *in_construction* the capacity decided but not yet operating (empty without a lead time); *total* is the capacity in service and does not include capacity in construction. Over combined solves (a chain of solves, invest then dispatch) every realized investment is counted once, in the period it comes online.
  - **Limits.** `invest_max_period` / `invest_min_period` (also of groups) count the capacity coming online in the period; the investment is decided earlier. `invest_max_total` / `invest_min_total` count every investment decision.
  - **Costs.** `invest_cost` is the overnight cost: construction time costs with their interest must be included in it. The annuity and the `fixed_cost` of new capacity are paid only over the years it operates within the model horizon, and the `invest_cost` (and `lifetime`, `discount_rate`) of the period it comes online is used. The `lifetime` starts when the capacity comes online.
  - **Horizon.** An investment whose rounded start of operation is after the end of the solve (including its look-ahead periods) or after the model horizon is not offered, with a warning naming the entity, the lead time and the solve. A chain of myopic solves whose horizon is not longer than the lead time therefore never invests in that technology: use a longer solve horizon or `construction_lead_time_method = immediate`.
- `construction_lead_time_method`: how the start of operation (decision year plus `construction_lead_time`) is rounded to the start of a period:
  - `immediate` — no lead time; the capacity operates from the decision period. This is also the effective default when no lead time is set.
  - `closest_seam` — the nearest period start; a tie goes to the later one. This is the effective default once a lead time **is** set but no method is chosen.
  - `previous_seam` — the period start before.
  - `next_seam` — the period start after.

    **Construction lead time and stochastics compose automatically.** Under recourse (`stochastic_invest_method = recourse`) with an invest non-anticipativity window (`shared_invest_periods`, see the stochastics how-to), a long-lead technology that must be online *after* the branch reveal has to be *ordered* before it — in a pre-reveal, tied period — so its investment is a single shared here-and-now decision, while a short-lead technology can be ordered at/after the reveal and react per scenario. This staging is emergent (no extra parameter): set the window to the pre-reveal periods and the lead time stages the technologies for you. Note the two knobs stay orthogonal — the window is the *information* boundary (which order periods are shared), the lead time is the *physics* (how early you must order); you need both, set consistently, for staging.

- `salvage_value`: Sets the extra value that can be gained for retiring [CUR/kW]. Only capacity that existed before the model run (`existing`) can be retired, until its end of life; the retirement saves its fixed cost over the periods it counts in and earns the salvage value as an annuity over `lifetime` (see [retirements](../reference/units.md#retirements)).
- `fixed_cost`: Annual cost for capacity [CUR/kW]

In many cases some of the investment decisions are tied to each other. Here the battery capacity and the connection capacity of the *battery_inverter* will be tied as they are simultaneously limited by the choice of the battery technology to be invested in.

To model this, a new constraint needs to be created that ties together the storage capacity of the *battery* and the charging/discharging capacity of the *battery_inverter*. 

A new `constraint` entity *battery_tie_kW_kWh* is created and added to an alternative in the `Entity Alternative` sheet.

It is given parameters `constant` and `sense`. Constant could be left out, since it is zero. The `sense` of the constraint must be *equal* to enforce the kw/kWh relation.

Both *battery_inverter* and *battery* need a coefficient to tell the model how they relate to each other. The equation has the capacity variables on the left side of the equation and the constant on the right side.

```
sum_i(`constraint_invested_capacity_coeff` * `v_invest`) = `constant`
      where i is any unit, connection or node that is part of the constraint
```

When the `constraint_invested_capacity_coeff` for *battery* is set at 1 and for the *battery_inverter* at -8, then the equation will force *battery_inverter* `v_invest` to be 8 times smaller than the *battery* `v_invest`. The negative term can be arranged to the right side of the equation, which yields:

```1 x *battery* = 8 x *battery_inverter*, which can be true only if *battery_inverter* is 1/8 of *battery*```

`constraint_invested_capacity_coeff` is not a parameter with a single value, but a map type parameter (index: constraint name, value: coefficient). It allows the entity to participate in multiple constraints. (In older databases this parameter was `constraint_capacity_coefficient`; migration renames it.)

To tie cumulative, built-up capacity rather than the current period's new build, use `constraint_cumulative_pre_built_capacity_coeff`. It takes the capacity in service at the start of the period: existing capacity plus the capacity that came online in earlier periods, minus earlier retirements. Capacity coming online in the period itself is in the invested term (`constraint_invested_capacity_coeff`, the capacity that comes online in the period; without a construction lead time, the capacity invested in that period). Paired with `constraint_invested_capacity_coeff`, it expresses period-over-period growth limits:

```
v_invest[e, d] <= growth_rate * pre_built[e, d]
```

is written as: `constraint_invested_capacity_coeff = { c: 1.0 }` and `constraint_cumulative_pre_built_capacity_coeff = { c: -growth_rate }` on entity `e`, with constraint `c` having `sense = less_than` and `constant = 0`. A 10% growth cap starting from 1000 MW of existing wind would allow 100 MW in period 1, 110 in period 2, 121 in period 3, and so on.

To tie the capacity in service in the period (the capacity that limits the flows, for example for a share of the fleet or a ratio between technologies), use `constraint_capacity_coeff`; for the capacity retired in the period, `constraint_retired_capacity_coeff`; for the capacity in construction (decided but not yet operating, only with a construction lead time), `constraint_in_construction_capacity_coeff`. Capacity in service = pre-built + invested − retired, with or without a construction lead time; capacity in construction is not in service.

Finally, FlexTool can mix several types of constraint coefficients: `constraint_invested_capacity_coeff`, `constraint_retired_capacity_coeff`, `constraint_cumulative_pre_built_capacity_coeff`, `constraint_capacity_coeff`, `constraint_state_coeff`, `constraint_online_coeff` and `constraint_flow_coeff` allowing the user to create custom constraints between any types of entities in the model for the main variables in the model (*flow*, *state*, *online* as well as *invest*). So, the equation above is in full form:

```
  + sum_i [constraint_invested_capacity_coeff(i) * v_invest]
           where i contains [node, unit, connection] belonging to the constraint
  + sum_i [constraint_retired_capacity_coeff(i) * v_divest]
  + sum_i [constraint_cumulative_pre_built_capacity_coeff(i) * pre_built]
           where pre_built = existing + Σ v_invest of earlier periods still in service − Σ v_divest of earlier periods
  + sum_i [constraint_capacity_coeff(i) * capacity_in_service]
           where capacity_in_service = existing + Σ v_invest still in service − Σ v_divest up to this period
  + sum_j [constraint_flow_coeff(j) * capacity]
           where j contains [unit--node, connection--node] belonging to the constraint
  + sum_k [constraint_state_coeff(k) * capacity]
           where k contains [node] belonging to the constraint
  + sum_m [constraint_online_coeff(m) * v_online]
           where m contains [unit, connection] with online variables belonging to the constraint,
           and v_online is the number of units online (of virtual_unitsize) in the time step
  =
  constant (a constant, or a period map giving each period its own value)

  where 'capacity' is existing capacity plus invested capacity
```

![Add battery investments](../img/concept/battery_invest.png)

## How to create a multi-year model

A multi-year model is constructed from multiple periods, each presenting one year. In the example case, each year is otherwise the same, but the demand is increasing in the *west* `node`. This means that all periods can use the same timeset *5weeks* from the same timeline *y2020*, but one can also make separate timelines for each year, if data is available for this. The `inflow` time series are scaled to match the value in `annual_flow` that is mapped for each period. The model is using the `inflow_method` *scale_to_annual* in order to achieve this (default is *use_original* that would not perform scaling). There should also be an `inflation_rate` parameter set for the `model` entity *flexTool* if something else than the model default of 0 is to be used (set to expected inflation when inputs are in nominal terms).

![Multi-year inflow](../img/concept/multi_year_inflow.png)

A multi-year model could be solved at one go (multi_year_one_solve) or by rolling through several solves (multi-year) where each solve has a foresight horizon and a realisation horizon. Next we will go through both options.

### Multi year with one solve
**(examples.sqlite scenario: multi_year_one_solve)**

In this example, one solve is used for all the four periods. All the four periods need to be added to the solve arrays `invest_periods` and `realized_periods`. Here the same timeset is used for all the four periods, so only difference between them is the increasing inflow set above. The parameters that need to be added to the `solve` entity:

- `years_represented` parameter is used by the model to calculate the discounting factors for the periods in the model (often future years). It should state the number of years each period will be representing. For example, a period for 2025 could represent the years 2025-2029 if its `years_represented` is set to 5. Any investments would be taking place at the start of 2025 and discounted to the beginning of 2025, but the operational costs would accrue from each year in 2025-2029 each with a different discounting factor (decreasing based on the interest rate).
- `invest_periods` the periods in which the model is allowed to make investments.
- `realized_periods` the periods that will be realized in this solve (outputs dispatch results for these periods).
- `period_timeset` defines the set of representative 'periods' (timesets in FlexTool) to be used in each FlexTool `period`.

![Multi-year one solve](../img/concept/multi_year_one_solve.png)

### Multi year with rolling solves 
**(examples.sqlite scenario: multi_year)**

If the solving time gets too big, there is an option to split the timeline into overlapping parts and solve them separately. This shortens the solving time as it increases exponentially as the model grows. Therefore, collection of smaller solves is faster. The drawback is that the accuracy will be affected if there are dependencies with a larger time interval than the split size. 

When dealing with investments the splitting only works if the different periods are similar, in most cases complete years. If the first solve is really windy, it would invest too much on wind which wouldn't produce enough energy in later solves and then it would need to invest again to something else. It can also be challenging to consider lifetimes. If the option of retiring is allowed, it might retire something that is needed for later solves. In this example, the periods are complete years and the only difference between periods is increased demand.

The model rolls through several solves and therefore, the `model` entity *flexTool* has four values in the `solves` array. Each value represents one solve and it's position in the sequence of solves. The next figure illustrates the realization (blue) and foresight horizons (grey). The first solve will solve both the year 2020 and year 2025, but it will only output the year 2020. The next will solve both 2025 and 2030 but only output 2025.

![One vs multi solve](../img/concept/one_multi_solve.png)

Next figure shows the values needed to define one solve (out of the four solves in the example). Each of these need to be repeated for each solve in the model.

- `years_represented` parameter is used by the model to calculate the discounting factors for the periods in the model (often future years). It should state the number of years each period will be representing. For example, a period for 2025 could represent the years 2025-2029 if its `years_represented` is set to 5. Any investments would be taking place at the start of 2025 and discounted to the beginning of 2025, but the operational costs would accrue from each year in 2025-2029 each with a different discounting factor (decreasing based on the interest rate).
- `invest_periods` the periods in which the model is allowed to make investments. To be given for each solve.
- `realized_periods` the periods that will be realized in this solve (outputs dispatch results for these periods). To be given for each solve.
- `realized_invest_periods` parameter states the periods that will realize the investment decisions. If not stated, it will use `realized_periods`.
- `period_timeset` defines the set of representative 'periods' (timesets in FlexTool) to be used in each FlexTool `period`.

Note the the `solve_mode`: *rolling_window* is not used! This is not for investment runs (without nesting) as it rolls freely, and investments should only be made at the start of the period. This example is called 'manual rolling' later when those are discussed.

In the examples.sqlite, the solve entities can carry solver options in the `solver_arguments` map (e.g. `solver`, `parallel` and `presolve` for HiGHS). They only affect the speed and not the results, but usually the default values are good enough and the user should only change them if they understand how the solvers work.

![Solve data](../img/concept/data_for_one_solve.png)
