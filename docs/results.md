# Model outputs

After every successful solve, the HiGHS solution is handed back to FlexTool in-memory together with the polars `FlexData` bundle that produced the LP. `flextool.process_outputs.write_outputs` then turns that pair into the result tables documented on this page.

The result tables are stored as **parquet** (the canonical store) and a configurable mix of derived artefacts:

| Location | Format | When written | Notes |
| --- | --- | --- | --- |
| `output_parquet/<scenario>/*.parquet` | parquet | always (canonical) | one file per result table; read by the GUI, the scenario-comparison tools, and `read_parquet_dir=True` re-runs |
| `output_csv/<scenario>/*.csv` | CSV | when `output-csv` is enabled | one CSV per parquet, plus `summary_solve.csv` (diagnostic overview) |
| `output_excel/output_<scenario>.xlsx` | Excel | when `output-excel` is enabled | one workbook per scenario; alphabetically ordered sheets |
| `output_plots/<scenario>/*.png` (or svg/pdf) | image | when `output-plot` is enabled | driven by the bundled `flextool/schemas/default_plots.yaml` |
| `<output-location>/results.sqlite` (or `--results-db-url`) | SpineDB | when `output-spinedb` is enabled (`--write-methods spinedb`) | the processed result tables written into a SpineDB using the FlexTool results schema (`flextool/schemas/spinedb_results_schema.json`); one Spine **alternative** per run (named after the scenario) so multiple runs coexist in one file. Time series land as nested Spine `Map` values on the model entity classes (node / unit / connection / group / model). |

`output_parquet/<scenario>/timeline_breaks.parquet` (columns `period`, `time`) lists the realized steps at which time-series plots start a new line, so a plot leaves a gap there instead of drawing a straight line across it: the step after a gap in the timeline, the start of every representative period (also when two representative periods are adjacent on the timeline, since they represent different base periods), and the first step of a period that does not continue the previous one (also a later year that reuses the same timeset). Rolling windows that follow each other on the timeline, and periods that continue the previous one, are drawn as one line. The file is written also when there are no breaks. Results saved before this file had the `period` and `time` columns are drawn without gaps until the scenario is run again. In scenario comparison plots the scenarios share one time axis, so a break of any selected scenario cuts every scenario's line at that step.

Every result table, with its parquet and CSV file names, its index and column levels, the unit of each column and its name in the results database, is listed in the [Results catalogue](results_catalogue.md), generated from the code. Below, a result is named by its results-database parameter (`class` entity `parameter`) where it is written to the results database, else by its table name.

The CSV / Excel / plot files are **derived** from the parquets, so deleting or regenerating them is safe. `summary_solve.csv` is a diagnostic file aimed at a quick overview of a solve — it lists objective decomposition, period weighting, CO2 total, and any non-zero slack uses.


All annualized numbers are scaled to **a full year of operation** by dividing through by `complete_period_share_of_year`. All values are returned in user units — auto-scaling (see [LP scaling pipeline](dev/scaling.md)) and time-block aggregation under flex-temporal decomposition are unwound before they reach the parquets, so users always read MW / MWh / [CUR] in the original problem space.

For stochastic models the parquets carry only the **realised / committed** results of each solve by default — the investment decisions, the operations of the periods before the reveal and, where a branch is designated realized, the realization dispatch of the fan — not the individual forecast branches. A forecasts-only fan has no realized operation, so its operational results are empty (the period label of the fan has no dispatch) and its operating costs and CO2 total are zero; its results hold the investment decisions and the investment and fixed costs (see [Costs of a run of several solves](#costs-of-a-run-of-several-solves)). The same holds for a rolling solve: only the realized steps of each roll (up to `rolling_solve_jump`) are reported, never its look-ahead. This does not depend on the `model` parameter `output_horizon` — the unrealized rows are only in its separate raw dump, below (see also [How to use stochastics](how_to/stochastics.md#how-to-use-stochastics-representing-uncertainty)).

With `output_horizon = yes` (a debug option), `output_parquet/<scenario>/horizon/` holds a **raw LP dump**: one parquet per LP variable and constraint, named as in the model (`v_flow.parquet`, `v_invest_p.parquet`, `nodeBalance_eq.parquet`, …), with every row of every solve — realized and unrealized (forecast branches, rolling look-ahead) alike — appended over the solves and rolls. Columns: the LP index columns (`p`, `n`, `d`, `t`, …), `solve`, `roll`, `realized` (`true` for the rows that also reach the normal results; empty for a variable or constraint without a period index), and `value` + `reduced_cost` for a variable or `dual` for a constraint. The values are raw solver values: primal values per unitsize as in the LP, duals and reduced costs only divided by `scale_the_objective` (HiGHS sign convention, probability and time weights included, empty for a MIP solve) — no unitsize, price or cost conversion. A solve decomposed with Benders is not dumped. The dump keeps every constraint's index until the solve's outputs are written, so it raises peak memory on large models.

- [Costs](#costs)
- [Cost by entity](#cost-by-entity)
- [Prices](#prices)
- [Energy flows](#energy-flows)
- [Capacity factors](#capacity-factors)
- [Energy balance in nodes](#energy-balance-in-nodes)
- [Unit online and startup](#unit-online-and-startup)
- [Connection online and startup](#connection-online-and-startup)
- [Unit curtailment and VRE potential](#unit-curtailment-and-vre-potential)
- [Node group results](#node-group-results)
- [Flow group results](#flow-group-results)
- [Capacity and investment results](#capacity-and-investment-results)
- [CO2 emissions](#co2-emissions)
- [Reserves](#reserves)
- [Inertia and non-synchronous generation](#inertia-and-non-synchronous-generation)
- [Ramps](#ramps)
- [DC power flow](#dc-power-flow)
- [Slack and penalty values](#slack-and-penalty-values)
- [Plotted data as CSV](#plotted-data-as-csv)

## Costs

- `model` entity `cost_annualized` parameter — M[CUR] (millions of user-chosen currency); annualized total cost broken down by category:
    - *unit investment & retirement* — M[CUR] cost of investing in unit capacity or salvage from retiring it
    - *connection investment & retirement* — M[CUR] same for connections
    - *storage investment & retirement* — M[CUR] same for node states (storages)
    - *fixed cost pre-existing* — M[CUR] fixed O&M for pre-existing capacity
    - *fixed cost invested* — M[CUR] fixed O&M for newly invested capacity
    - *fixed cost reduction of divestments* — M[CUR] fixed O&M removed by divestments
    - *commodity cost* / *commodity sales* — M[CUR] cost of buying / revenue from selling commodities
    - *co2* — M[CUR] cost of CO2 emissions caused by commodities with CO2 content, net of the removal credit (negative cost) for flows into the same priced group's node(s)
    - *other operational* — M[CUR] other variable O&M costs
    - *starts* — M[CUR] start-up costs
    - *ramps* — M[CUR] ramp costs: the cost of changing the flow of units with a `ramp_cost`
    - *upward / downward slack penalty* — M[CUR] cost of involuntary demand reduction / increase
    - *inertia slack penalty* — M[CUR] cost of not meeting the inertia constraint
    - *non-synchronous slack penalty* — M[CUR] cost of not meeting the non-synchronous constraint
    - *capacity margin penalty* — M[CUR] cost of not meeting the capacity margin constraint
    - *upward / downward reserve slack penalty* — M[CUR] cost of not meeting the reserve constraint

    The annualized investment & retirement, *fixed cost invested* and *fixed cost reduction of divestments* are
    the yearly amounts of every vintage and retirement in effect in the period: an investment adds its annuity (the
    `invest_cost` annuity of the period the capacity starts to operate) and its yearly fixed cost in every period it
    operates within the run (within its `lifetime`, or every later period with `reinvest_automatic`); a retirement
    subtracts its salvage annuity in every period from the retirement until `lifetime` later, and the yearly fixed
    cost of the retired capacity in every period the retirement counts in (until the end of life of the existing
    capacity). They are reported only for capacity whose investment cost is reported (see the next section). For these
    columns the annualized value is therefore not the discounted value divided by the period's discount factor,
    which holds for the other columns.

- `model` entity `cost_t` parameter — [CUR] (currency, not millions) same categories as above but per timestep: the cost incurred during the step, with the step duration and timestep weight applied but not scaled to a full year (no investment / fixed terms). The CSV / parquet output `costs__dt` has the same values and unit.
- `model` entity `cost_discounted_solve` parameter — M[CUR] costs for the solve considering discounting and years represented (NPV currency)
- `model` entity `cost_discounted_total` parameter — M[CUR] same, totalled over all realised periods

### Costs of a run of several solves

When `model.solves` holds several solves (a chain of investment solves, an investment solve and its dispatch
solves, a stochastic solve and its realization solve), the reported costs are those of the run as a whole, the same
as one solve over the same periods with the same decisions would report:

- **One discount base, the start of the run.** Every discounted cost counts its years from the first period of the
  run, not from the first period of the solve that reports it.
- **Each realized period is reported once, by the solve that realizes it**: its operating and penalty costs come
  from that solve's dispatch.
- **New capacity** is reported in the period where it starts to operate (the decision period without a
  `construction_lead_time`; during the lead time it is only `in_construction`, at no cost). Its investment cost
  (the annuity over the years it operates within the run, at the `invest_cost`, `lifetime` and `discount_rate` of
  that period) and the fixed cost over the same years come once, from the solve that decided it, and are reported
  only once a solve realizes the period it starts to operate in. Capacity carried from an earlier solve is
  therefore not charged again as existing capacity.
- **Retirements** are reported in the retirement period: the salvage annuity over the years from the retirement
  until `lifetime` later, and the fixed cost saved in the periods the retirement counts in. The fixed cost of existing capacity (*fixed cost pre-existing*) is that of the
  capacity given in `existing`, aged over its lifetime, in every period of the run, also in the last solve's
  look-ahead periods that no solve realizes (the fixed cost of new capacity covers them too).
- **Inside a stochastic fan** the investment cost and the fixed cost of the new capacity that operates in the fan's
  branch periods are reported, weighted by each branch's probability weight, at the fan's period label of the
  period the capacity starts to operate in (with or without a `construction_lead_time`, and also without a
  realization solve). A shared investment (`stochastic_invest_method` none) is one decision whose branch weights
  sum to 1, so it is reported at its full amount; under `recourse` each forecast's own investment is reported
  times its branch weight (the expected investment, as in the objective). Retirements in the branch periods of a
  recourse fan are not reported.
- In a stochastic solve with a realization solve, the investments come from the stochastic solve and the
  operations and the fixed cost of existing capacity from the realization solve. A fan without a realization
  solve reports no operating or penalty costs; the fixed cost of existing capacity is reported in the fan's
  period labels.

The printed objective of each solve, the cumulative CO2 cap price and the investment duals (`invest_marginal`,
the value of one more MW of capacity) stay discounted from the start of that solve, so `cost_discounted_total` is
not the sum of the solves' objectives and the investment duals of later solves of a chain are on a later discount
base. The CO2 price of a period is its dual divided by the discount factor of the solve that realizes the period.
A CSV-dump reload has no run-level cost data: its reported costs are discounted from the start of each solve and
its annualized investment, retirement and fixed costs of new and retired capacity are 0 (a warning is printed).

## Cost by entity

The system [Costs](#costs) above are also broken down **per entity at the period level**, so you can see which unit, connection or node drives each cost category. Six result tables are produced — one for each entity type (`unit`, `connection`, `node`) in each of two flavours:

- *annualized* — M[CUR/a], scaled to a full year of operation (the same basis as `cost_annualized`)
- *discounted* — M[CUR], net present value over the horizon (the same basis as `cost_discounted_total`)

Each table is indexed by `(period, entity)` — the entity level is named after its type (`unit`, `connection` or `node`) — and carries one column per cost `category`:

- `unit` entity `cost_annualized` / `cost_discounted` parameters — per-unit cost break-down
- `connection` entity `cost_annualized` / `cost_discounted` parameters — per-connection cost break-down
- `node` entity `cost_annualized` / `cost_discounted` parameters — per-node cost break-down

The categories mirror the system [Costs](#costs) decomposition, restricted to those that apply to the entity type:

- *commodity cost* / *commodity sales* — M[CUR] cost of buying / revenue from selling commodities
- *co2* — M[CUR] cost of CO2 emissions caused by the entity's flows
- *other operational* — M[CUR] other variable O&M costs
- *starts* — M[CUR] start-up costs (units)
- *ramps* — M[CUR] ramp costs (units with a `ramp_cost`)
- *investment* / *retirement* — M[CUR] cost of investing in unit / connection capacity or salvage from retiring it (named *storage investment* / *storage retirement* in the `node` table, where the capacity is node-state storage energy)
- *fixed cost pre-existing* / *fixed cost invested* / *fixed cost reduction of divestments* — M[CUR] fixed O&M for pre-existing, newly invested and divested capacity
- in the *annualized* tables, *investment*, *retirement*, *fixed cost invested* and *fixed cost reduction of divestments* are the yearly amounts in every period the capacity operates or the retirement applies, as in the system [Costs](#costs)
- *upward slack penalty* / *downward slack penalty* (`node` table only) — M[CUR] cost of involuntary demand reduction / increase at the node

Attribution rules (matching the LP objective):

- **Fuel / commodity cost** is attributed to the **consuming process** — the unit or connection that draws the commodity — not to the source node.
- **CO2 cost** for a process is the **sum over every priced group the process touches**, so a flow that contributes emissions to several priced groups carries the full cost of each. A flow into a priced group's node (removal) carries a negative cost (credit) the same way.

By construction these tables are purely additive collapses of the same per-entity intermediates the system summary uses, so **summing any category over all entities reproduces the system summary**: the `unit` + `connection` + `node` contributions for a given category and period add up to the matching `cost_annualized` / `cost_discounted_total` category value (`annualized_costs_d_p` / `costs_discounted_d_p` in the parquet store).

These are period-level tables only — there are no per-timestep per-entity cost tables.

## Prices

- `node` entity `price_t` parameter — [CUR/MWh] the cost of one more MWh of demand at that node and time step (the dual of the node's energy balance; positive when more demand would cost more), for every node that maintains an energy balance. It is in the price base of the cost inputs: the inflation / discounting of the period and the years it represents are removed, and it is not weighted by the timestep weight (`timeset_weights`, representative-period weights) or by the branch probability — in a stochastic solve each forecast branch reports its own price, not its probability-weighted contribution (the cost outputs per step are probability-weighted, so price × flow is not the branch's cost row). A step that the objective does not weigh (e.g. a step that `timeset_weights` leaves out) has no price (empty). A node of a group with a coarser `new_stepduration` (and a `bind_intraperiod_blocks` storage node) has one balance per coarse step (per timeset block); its price is that balance's dual, the same at every time step of the coarse step when the steps have the same weight. Within a coarse step the energy may be moved freely between time steps, so for such a node the hour at which a slack (or any flow that only the coarse balance limits) is reported is one of several equally good choices; with a time-varying penalty the solver picks the cheapest hour. For a storage node the price is the value of stored energy.
- table `reserve_prices_dt_ppg` (`group__reserve__upDown`, not in the results database) — [CUR/MWh] the cost of one more MW of reserve requirement held for one hour (in the same price base and with the same weights removed as `price_t`): the sum of the duals of the group's requirement constraints at the step (the timeseries requirement, the dynamic requirement and the large-failure requirement of every failing unit or connection), i.e. the cost of one more MW of every requirement of the group at once. Reported for every group with a requirement. When the requirement is not met it equals `penalty_reserve`.
- table `co2_price_period_d_g` (`group`, not in the results database) — [CUR/tCO2] shadow price of the per-period CO2 cap (nominal currency, sign-flipped so a binding cap shows positive). In a rolling solve where a period's hours are split across more than one roll, the reported price is the hour-weighted average of the rolls that realize it (weighted by the hours each roll keeps of the period). In a stochastic solve each scenario path reports its own price, not its price times the path's branch probability.
- table `co2_price_total_d_g` (`group`, not in the results database) — [CUR/tCO2] shadow price of the cumulative CO2 cap, per tonne over the represented years (NPV currency, broadcast across periods)

## Energy flows

- `unit__node` entity `flow_annualized` parameter — [MWh] cumulative flow from the node (if node is input) or to the node (if node is output), annualized to a full year
- `unit__node` entity `flow_t` parameter — [MW] flow at each timestep
- `connection__node__node` entity `flow_annualized` parameter — [MWh] annualized cumulative flow through the connection (left-to-right is positive)
- `connection__node__node` entity `flow_t` parameter — [MW] flow through the connection at each timestep
- table `connection_losses_d_eee` (not in the results database) — [MWh] annualized losses on the connection
- table `connection_losses_dt_eee` (not in the results database) — [MW] losses on the connection at each timestep

### Optional output: output_connection_flows_separate

- `connection__node__node` entity `flow_to_first_node_annualized` parameter — [MWh] annualized cumulative flow only to the left (first) node
- `connection__node__node` entity `flow_to_second_node_annualized` parameter — [MWh] annualized cumulative flow only to the right (second) node
- `connection__node__node` entity `flow_to_first_node_t` parameter — [MW] flow to the left (first) node at each timestep
- `connection__node__node` entity `flow_to_second_node_t` parameter — [MW] flow to the right (second) node at each timestep

## Capacity factors

- `unit__node` entity `cf` parameter — [per unit] average capacity factor of the flow: time-average of flow [MWh/h] divided by the unit's capacity [MW], which is on its output side; so the input cf of a unit at full load is 1/`efficiency`. One table per direction (`unit_cf__inputNode`, `unit_cf__outputNode`).
- `connection__node__node` entity `cf` parameter — [per unit] average capacity factor of the absolute flow (flows in either direction count as utilization), divided by connection capacity

## Energy balance in nodes

- `node` entity `balance` parameter — [MWh] annualized period sum of all balance contributions: *From units*, *From connections*, *To units*, *To connections*, *Self discharge*, *Loss of load* (upward slack), *Excess load* (downward slack), *Inflow*
- `node` entity `balance_t` parameter — [MW] same categories per timestep. *Self discharge* is the energy the storage lost in the step (the loss the optimisation used) divided by the step length; the loss of a coarse step (`new_stepduration` group) or of a `bind_intraperiod_blocks` block is spread evenly over its time steps.
- `node` entity `state_t` parameter — [MWh] storage state of the node at each timestep (state nodes only; multiplied by `entity_unitsize` so the value is in storage MWh, not the scaled LP variable)

## Unit online and startup

- `unit` entity `online_t` parameter — [count] number of units online at each timestep
- `unit` entity `online_average` parameter — [count] average online status over the period (weighted by step duration)
- `unit` entity `startup_cumulative` parameter — [count] startups during the period scaled to a full year (weighted by `rp_cost_weight` to handle representative periods correctly)

## Connection online and startup

Only for connections with a `startup_method`:

- `connection` entity `online_t` parameter — [count] units of the connection online at each timestep
- `connection` entity `online_average` parameter — [count] average online status over the period (weighted by step duration)
- `connection` entity `startup_cumulative` parameter — [count] startups during the period scaled to a full year (as for units)

## Unit curtailment and VRE potential

For VRE units (units that use an `upper_limit` profile) the model also reports curtailment and potential generation against each output node:

- table `unit_VRE_potential_outputNode_dt_ee` (not in the results database) — [MW] potential VRE generation at the timestep (`capacity × profile`)
- table `unit_VRE_potential_outputNode_d_ee` (not in the results database) — [MWh] potential VRE generation summed over the period's time steps (with the timestep weights), not scaled to a full year
- `unit__node` entity `curtailment_t` parameter — [MW] potential minus actual generation at the timestep
- table `unit_curtailment_outputNode_d_ee` (not in the results database) — [MWh] curtailed energy summed over the period's time steps (with the timestep weights), not scaled to a full year
- table `unit_curtailment_share_outputNode_dt_ee` (not in the results database) — [0–1] timestep curtailment / potential
- `unit__node` entity `curtailment_share` parameter — [0–1] period curtailment / period potential

## Node group results

`group` entities with `print_indicators = yes` produce indicator tables aggregating flows / inflows / slacks over their member nodes:

- `group` entity `indicator` (period table) — gives a set of indicators for all `node` members:
    - *Loss of load share* — [0–1] upward slack relative to inflow
    - *VRE share of demand* — [0–1] share of inflow served by VRE sources
    - *Excess load share* — [0–1] downward slack relative to inflow
    - *Curtailed VRE of demand* — [0–1] curtailed VRE relative to inflow
    - *Annualized inflow* — [MWh] sum of `inflow` to member nodes scaled to a year
    - *Curtailed VRE of potential VRE* — [0–1] curtailed share of potential VRE
- table `nodeGroup_gdt_p` (timestep table, not in the results database) — exposes the timestep-level building blocks of the above: *Loss of load* [MWh/step], *VRE generation* [MWh/step], *Excess load* [MWh/step], *Curtailed VRE* [MWh/step], *Timestep inflow* [MWh/step], *Curtailed VRE of potential VRE* [0–1], *Annualized inflow* [MWh], *VRE share of demand* [0–1]
- `group` entity `VRE_share_t` parameter — [0–1] share of inflow served by VRE sources at each timestep (one column per node group)
- table `nodeGroup_VRE_share_d_g` (not in the results database) — [0–1] period-average VRE share
- `group` entity `flow_annualized` parameter — [MWh] annualized sum of `inflow` to member nodes
- `group` entity `flow_t` parameter — [MWh/step] timestep inflow to member nodes

`group` entities with `print_dispatch = yes` produce a multi-column dispatch table over the group:

- `group` entity `sum_flow_t` parameter — [MWh/step] dispatch decomposed by `(type, item)` where `type` is one of: *slack* (upward/downward), *from_unit*, *from_unitGroup*, *to_unit*, *to_unitGroup*, *from_connection*, *from_connectionGroup*, *to_connection*, *to_connectionGroup*, *inflow*, *internal_losses* (units/connections/storages)
- `group` entity `sum_flow_annualized` parameter — [MWh] same decomposition, annualized period totals

## Flow group results

`flowGroup` entities whose `flow_aggregator` is `standalone_aggregator_only` or `both` produce aggregate flow statistics over their member `(process, node)` legs (listed via `flowGroup__unit__node` / `flowGroup__connection__node`):

- `group_flow__d.csv` — per-(flowGroup, period) totals:
    - `cumulative_flow` parameter — [MWh] sum of |flow| over the period for all member legs
    - `average_flow` parameter — [MW] average power equivalent (`cumulative_flow` / period hours)
- `group_flow__dt.csv` — per-(flowGroup, period, time) **signed net flow** [MW] (sign convention: flow *into* the group's nodes is positive, flow *out of* them is negative). Intended for spreadsheet post-processing; because the values are signed they do not stack, so this series is not meant for stacked dispatch plots.

(For member flows shown as aggregated bands inside a node group's dispatch table, set `flow_aggregator` to `dispatch_plots_only` or `both` instead — see [Node group results](#node-group-results) and the [reference](reference/flow_groups.md).)

## Capacity and investment results

- `unit`, `connection`, `node` entities `capacity` parameter — [MW or MWh] decomposed into:
    - *existing* — capacity assumed at the start of the period
    - *invested* — capacity that comes online in the period (without a construction lead time, the capacity invested in that period)
    - *in_construction* — capacity decided but not yet operating in the period; always present, empty without a construction lead time
    - *divested* — capacity the model decided to retire at the start of the period
    - *total* — capacity in service: `existing + invested − divested`, cumulated over the periods; capacity in construction is not included

    In combined solves (e.g. a chain of solves, or invest then dispatch), each realized investment is counted once, in the period it comes online.

    Period axis covers every period in which an investment **or** a divestment decision occurs (both the invest-eligible and divest-eligible periods feed into `d_realize_invest`), so divest-only periods are visible even when no new capacity is added.

- `unit`, `connection`, `node` entities `invest_marginal` parameter — [CUR/MW or MWh] effective dual of the investment decision: zero means the model is at the unconstrained optimum; positive means an active upper bound (per-entity, per-group, period, total or cumulative caps are summed automatically); negative means the entity is not built and is unprofitable to build (the v_invest column's own reduced cost) — this is the ordinary reading for an unbuilt entity, not an artefact; a numerical artefact instead looks like a non-zero value on a *built* (or at-bound) entity, or an inconsistent sign across these regimes, and is worth investigating

## CO2 emissions

- table `CO2__` (not in the results database) — [Mt] horizon-total CO2 across all groups and units (uses `years_represented_d` to convert annualized emissions to horizon totals). In a stochastic solve with more than one scenario path this is reported per path (one column per path) instead of pooling every path's emissions into one number — the anchor's shared pre-reveal history and each forecast branch's own periods are alternative futures, not consecutive years to add together.
- `group` entity `CO2_annualized` parameter — [t/yr] annualized CO2 emissions caused (or removed) by units and connections in the group
- `unit` entity `CO2_annualized` parameter — [t/yr] annualized CO2 emissions of the unit's flows, by commodity and node

## Reserves

- `unit__reserve__upDown__node` and `connection__reserve__upDown__node` `reservation_t` parameter — [MW] reserve provision at each timestep
- `unit__reserve__upDown__node` and `connection__reserve__upDown__node` `reservation_average` parameter — [MW] period-average reserve provision

## Inertia and non-synchronous generation

- `group` entity `inertia_t` parameter — [MWs] total inertia in the group of nodes at each timestep
- `group` entity `inertia_largest_flow_t` parameter — [MW] largest individual flow coming into the group of nodes with `has_inertia`
- `group` entity `inertia_unit_node_t` parameter — [MWs] inertia contribution per (unit, node) at each timestep, one column per contributing `(unit, node)`

## Ramps

- `unit__node` entity `ramp_t` parameter — [MW] ramp of the unit input or output flow at each timestep: the change from the previous step (one table per direction: `unit_ramp__inputNode__dt`, `unit_ramp__outputNode__dt`). The ramp at a step with no previous step in the solve (the first step, the first step after a gap in the timeline, the first step of a representative period) is empty; at the first step of a rolling window it is measured from the previous window's last step.

The "ramp room" envelope on the `node` entity (additional headroom for upward / downward ramps from non-VRE units, VRE units, and connections) is currently not emitted by the new pipeline — it is on the roadmap for re-exposure once the corresponding post-processing is ported.

## DC power flow

When the network contains DC-power-flow connections / nodes (`is_dc_power_flow = yes`):

- table `dc_angle_dt_e` (not in the results database) — [rad] voltage angle at each timestep
- table `dc_angle_diff_dt_e` (not in the results database) — [rad] angle difference across the connection at each timestep

## Slack and penalty values

Slack uses are listed in the [Energy balance in nodes](#energy-balance-in-nodes) section (`Loss of load`, `Excess load` and per-node `slack_up` / `slack_down` parameters with `_t` and annualized variants) and aggregated in [Costs](#costs). Group-level slack tables:

- `group` entity `slack_capacity_margin` parameter — [MW] capacity-margin shortfall in investment periods
- `group` entity `slack_inertia_t` / `slack_inertia` — [MWs] inertia shortfall, timestep / annualized
- `group` entity `slack_nonsync_t` / `slack_nonsync` — [MW] non-synchronous-share shortfall, timestep / annualized
- `group__reserve__upDown` entity `slack_reserve_t` / `slack_reserve` — [MW] reserve shortfall, timestep / annualized

For the sign / direction conventions of slacks, see [Slack convention](dev/slack_convention.md). For how block-aware solves (flex-temporal decomposition) and spatial Benders decomposition feed into these tables, see [Decomposition](dev/decomposition.md) — the result writers always operate on the fine-timeline, full-spatial solution, so the parquet tables look identical to a monolithic run.

## Plotted data as CSV

With **Write plotted data (CSV)** in the GUI's Png settings, `--write-plot-data` on `flextool-run`,
`flextool-write-outputs` or `flextool-scenario-results` (Spine Toolbox users can add it to the tool's command-line
arguments), or a top-level `write_plot_data: true` in the project's `plot_settings.yaml`, each plot file gets a CSV
file next to it with the data drawn in it: `output_plots/<scenario>/<plot>.png` comes with `<plot>.csv`, a plot
split into several files writes `<plot>_01.csv`, `<plot>_02.csv`, … with the data of each file, and the comparison
plots and dispatch plots in `output_plot_comparisons/` get theirs too. Any one of these turns it on. It is off by
default and then costs nothing.

- Rows are the x axis of the plot (the time steps, or the bars), with the row index levels (for example `period`,
  `time`) as the first columns. Only the plotted time window is written, and the gap rows that leave a gap in a
  line at a [timeline break](#model-outputs) are not.
- Columns are the plotted series (lines, stacks, bars), with one header row. When a series is named by several
  dimensions, its header is their values joined with ` | `, in the order of the result's dimensions; the dimension
  names themselves are not written. For example, in a comparison plot of unit output with the scenario first and the
  unit second, the column `base | coal_plant` is unit `coal_plant` in scenario `base`. A series named like one of
  the first columns (for example an entity called `time`) gets ` (series)` added to its name.
- A leading `subplot` column names the subplot of each row when the file has more than one subplot.
- Values are as plotted, after `multiply_by` (the unit is in the plot's axis label). For a stacked plot, each
  column is that series' own value, not the stacked total, so do not add the columns up again. Where a line is drawn
  as the sum over a dimension the plot does not show, the column is that sum. A cell that is empty in the
  results stays empty, also where a bar plot draws it as zero.
- A dispatch plot's CSV has the flows of the plotted time window and the demand line as a `Demand` column, as in
  the legend (`Demand (line)` if a flow is already called `Demand`).
- If a CSV cannot be written, the log has a warning naming the file, and the plots are still made.
- In `plot_settings.yaml`, only `write_plot_data: true` (no quotes) turns this on; a quoted `"true"` or any value
  that is not `true` or `false` leaves it off with a warning.

The CSVs are removed with the plots when the plots are made again.
