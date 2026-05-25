# Model outputs

After every successful solve, the HiGHS solution is handed back to FlexTool in-memory together with the polars `FlexData` bundle that produced the LP. `flextool.process_outputs.write_outputs` then turns that pair into the result tables documented on this page.

The result tables are stored as **parquet** (the canonical store) and a configurable mix of derived artefacts:

| Location | Format | When written | Notes |
| --- | --- | --- | --- |
| `output_parquet/<scenario>/*.parquet` | parquet | always (canonical) | one file per result table; read by the GUI, the scenario-comparison tools, and `read_parquet_dir=True` re-runs |
| `output_csv/<scenario>/*.csv` | CSV | when `output-csv` is enabled | one CSV per parquet, plus `summary_solve.csv` (diagnostic overview) |
| `output_excel/output_<scenario>.xlsx` | Excel | when `output-excel` is enabled | one workbook per scenario; alphabetically ordered sheets |
| `output_plots/<scenario>/*.png` (or svg/pdf) | image | when `output-plot` is enabled | driven by `templates/default_plots.yaml` |

The CSV / Excel / plot files are **derived** from the parquets, so deleting or regenerating them is safe. `summary_solve.csv` is a diagnostic file aimed at a quick overview of a solve — it lists objective decomposition, period weighting, CO2 total, and any non-zero slack uses.

All annualized numbers are scaled to **a full year of operation** by dividing through by `complete_period_share_of_year`. All values are returned in user units — auto-scaling (see [LP scaling pipeline](dev/scaling.md)) and time-block aggregation under flex-temporal decomposition are unwound before they reach the parquets, so users always read MW / MWh / [CUR] in the original problem space.

- [Costs](#costs)
- [Prices](#prices)
- [Energy flows](#energy-flows)
- [Capacity factors](#capacity-factors)
- [Energy balance in nodes](#energy-balance-in-nodes)
- [Unit online and startup](#unit-online-and-startup)
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

## Costs

- `model` entity `cost_annualized` parameter — M[CUR] (millions of user-chosen currency); annualized total cost broken down by category:
    - *unit investment & retirement* — M[CUR] cost of investing in unit capacity or salvage from retiring it
    - *connection investment & retirement* — M[CUR] same for connections
    - *storage investment & retirement* — M[CUR] same for node states (storages)
    - *fixed cost pre-existing* — M[CUR] fixed O&M for pre-existing capacity
    - *fixed cost invested* — M[CUR] fixed O&M for newly invested capacity
    - *fixed cost reduction of divestments* — M[CUR] fixed O&M removed by divestments
    - *commodity cost* / *commodity sales* — M[CUR] cost of buying / revenue from selling commodities
    - *co2* — M[CUR] cost of CO2 emissions caused by commodities with CO2 content
    - *other operational* — M[CUR] other variable O&M costs
    - *starts* — M[CUR] start-up costs
    - *upward / downward slack penalty* — M[CUR] cost of involuntary demand reduction / increase
    - *inertia slack penalty* — M[CUR] cost of not meeting the inertia constraint
    - *non-synchronous slack penalty* — M[CUR] cost of not meeting the non-synchronous constraint
    - *capacity margin penalty* — M[CUR] cost of not meeting the capacity margin constraint
    - *upward / downward reserve slack penalty* — M[CUR] cost of not meeting the reserve constraint
- `model` entity `cost_t` parameter — M[CUR] same categories as above but per timestep (no investment / fixed terms)
- `model` entity `cost_discounted_solve` parameter — M[CUR] costs for the solve considering discounting and years represented (NPV currency)
- `model` entity `cost_discounted_total` parameter — M[CUR] same, totalled over all realised periods

## Prices

- `node` entity `price_t` parameter — [CUR/MWh] dual of the balance constraint for every node that maintains an energy balance (nominal currency at period `d`, un-discounted from NPV)
- `group__reserve__upDown` entity `reserve_price_t` parameter — [CUR/MWh] dual of the reserve balance constraint
- `group` entity `co2_price_period` parameter — [CUR/tCO2] shadow price of the per-period CO2 cap (nominal currency, sign-flipped so a binding cap shows positive)
- `group` entity `co2_price_total` parameter — [CUR/tCO2] shadow price of the cumulative CO2 cap (NPV currency, broadcast across periods)

## Energy flows

- `unit__node` entity `flow_annualized` parameter — [MWh] cumulative flow from the node (if node is input) or to the node (if node is output), annualized to a full year
- `unit__node` entity `flow_t` parameter — [MW] flow at each timestep
- `connection__node__node` entity `flow_annualized` parameter — [MWh] annualized cumulative flow through the connection (left-to-right is positive)
- `connection__node__node` entity `flow_t` parameter — [MW] flow through the connection at each timestep
- `connection__node__node` entity `connection_losses_annualized` parameter — [MWh] annualized losses on the connection
- `connection__node__node` entity `connection_losses_t` parameter — [MW] losses on the connection at each timestep

### Optional output: output_connection_flows_separate

- `connection__node__node` entity `flow_to_first_node_annualized` parameter — [MWh] annualized cumulative flow only to the left (first) node
- `connection__node__node` entity `flow_to_second_node_annualized` parameter — [MWh] annualized cumulative flow only to the right (second) node
- `connection__node__node` entity `flow_to_first_node_t` parameter — [MW] flow to the left (first) node at each timestep
- `connection__node__node` entity `flow_to_second_node_t` parameter — [MW] flow to the right (second) node at each timestep

## Capacity factors

- `unit__node` entity `cf` parameter — [per unit] average capacity factor of the flow: time-average of flow [MWh/h] divided by capacity [MW] of the unit input or output. One table per direction (`unit_cf__inputNode`, `unit_cf__outputNode`).
- `connection` entity `cf` parameter — [per unit] average capacity factor of the absolute flow (flows in either direction count as utilization), divided by connection capacity

## Energy balance in nodes

- `node` entity `balance` parameter — [MWh] annualized period sum of all balance contributions: *From units*, *From connections*, *To units*, *To connections*, *Self discharge*, *Loss of load* (upward slack), *Excess load* (downward slack), *Inflow*
- `node` entity `balance_t` parameter — [MW] same categories per timestep
- `node` entity `state_t` parameter — [MWh] storage state of the node at each timestep (state nodes only; multiplied by `entity_unitsize` so the value is in storage MWh, not the scaled LP variable)

## Unit online and startup

- `unit` entity `online_t` parameter — [count] number of units online at each timestep
- `unit` entity `online_average` parameter — [count] average online status over the period (weighted by step duration)
- `unit` entity `startup_annualized` parameter — [count] startups during the period scaled to a full year (weighted by `rp_cost_weight` to handle representative periods correctly)

## Unit curtailment and VRE potential

For VRE units (units that use an `upper_limit` profile) the model also reports curtailment and potential generation against each output node:

- `unit__node` entity `VRE_potential_t` parameter — [MW] potential VRE generation at the timestep (`capacity × profile`)
- `unit__node` entity `VRE_potential_annualized` parameter — [MWh] annualized potential VRE generation in the period
- `unit__node` entity `curtailment_t` parameter — [MW] potential minus actual generation at the timestep
- `unit__node` entity `curtailment_annualized` parameter — [MWh] annualized curtailed energy in the period
- `unit__node` entity `curtailment_share_t` parameter — [0–1] timestep curtailment / potential
- `unit__node` entity `curtailment_share` parameter — [0–1] period curtailment / period potential

## Node group results

`group` entities with `output_group_node_indicators = yes` produce indicator tables aggregating flows / inflows / slacks over their member nodes:

- `group` entity `indicator` (period table) — gives a set of indicators for all `node` members:
    - *Loss of load share* — [0–1] upward slack relative to inflow
    - *VRE share of demand* — [0–1] share of inflow served by VRE sources
    - *Excess load share* — [0–1] downward slack relative to inflow
    - *Curtailed VRE of demand* — [0–1] curtailed VRE relative to inflow
    - *Annualized inflow* — [MWh] sum of `inflow` to member nodes scaled to a year
    - *Curtailed VRE of potential VRE* — [0–1] curtailed share of potential VRE
- `group` entity `indicator_t` (timestep table) — exposes the timestep-level building blocks of the above: *Loss of load* [MWh/step], *VRE generation* [MWh/step], *Excess load* [MWh/step], *Curtailed VRE* [MWh/step], *Timestep inflow* [MWh/step], *Curtailed VRE of potential VRE* [0–1], *Annualized inflow* [MWh], *VRE share of demand* [0–1]
- `group` entity `VRE_share_t` parameter — [0–1] share of inflow served by VRE sources at each timestep (one column per node group)
- `group` entity `VRE_share` parameter — [0–1] period-average VRE share
- `group` entity `total_inflow_annualized` parameter — [MWh] annualized sum of `inflow` to member nodes
- `group` entity `total_inflow_t` parameter — [MWh/step] timestep inflow to member nodes

`group` entities with `output_group_node_flows = yes` produce a multi-column dispatch table over the group:

- `group` entity `flows_t` parameter — [MWh/step] dispatch decomposed by `(type, item)` where `type` is one of: *slack* (upward/downward), *from_unit*, *from_unitGroup*, *to_unit*, *to_unitGroup*, *from_connection*, *from_connectionGroup*, *to_connection*, *to_connectionGroup*, *inflow*, *internal_losses* (units/connections/storages)
- `group` entity `flows_annualized` parameter — [MWh] same decomposition, annualized period totals

## Flow group results

`group` entities with `output_flowGroup_indicators = yes` produce aggregate flow statistics over their member `(process, node)` legs (listed via `group__process__node`):

- `group` entity `cumulative_flow` parameter — [MWh] sum of |flow| over the period for all member legs
- `group` entity `average_flow` parameter — [MW] average power equivalent (`cumulative_flow` / period hours)

## Capacity and investment results

- `unit`, `connection`, `node` entities `capacity` parameter — [MW or MWh] decomposed into:
    - *existing* — capacity assumed at the start of the period
    - *invested* — capacity the model decided to invest in for the period
    - *divested* — capacity the model decided to retire at the start of the period
    - *total* — `existing + invested − divested`
- `unit`, `connection`, `node` entities `invest_marginal` parameter — [CUR/MW or MWh] effective dual of the investment decision: zero means the model is at the unconstrained optimum; positive means an active upper bound (per-entity, per-group, period, total or cumulative caps are summed automatically); negative is not expected and indicates a numerical artefact worth investigating

## CO2 emissions

- `model` entity `CO2` parameter — [Mt] horizon-total CO2 across all groups and units (uses `years_represented_d` to convert annualized emissions to horizon totals)
- `group` entity `CO2_annualized` parameter — [Mt/yr] annualized CO2 emissions caused (or removed) by units and connections in the group
- `unit__source__sink` entity `CO2_annualized` parameter — [Mt/yr] annualized CO2 emissions per flow leg

## Reserves

- `unit__reserve__upDown__node` and `connection__reserve__upDown__node` `reservation_t` parameter — [MW] reserve provision at each timestep
- `unit__reserve__upDown__node` and `connection__reserve__upDown__node` `reservation_average` parameter — [MW] period-average reserve provision

## Inertia and non-synchronous generation

- `group` entity `inertia_t` parameter — [MWs] total inertia in the group of nodes at each timestep
- `group` entity `inertia_largest_flow_t` parameter — [MW] largest individual flow coming into the group of nodes with `has_inertia`
- `group__unit__node` entity `inertia_t` parameter — [MWs] inertia contribution per (group, unit, node) at each timestep

## Ramps

- `unit__node` entity `ramp_t` parameter — [MW] ramp of the unit input or output flow at each timestep (one table per direction: `unit_ramp__inputNode__dt`, `unit_ramp__outputNode__dt`)

The "ramp room" envelope on the `node` entity (additional headroom for upward / downward ramps from non-VRE units, VRE units, and connections) is currently not emitted by the new pipeline — it is on the roadmap for re-exposure once the corresponding post-processing is ported.

## DC power flow

When the network contains DC-power-flow connections / nodes (`is_dc_power_flow = yes`):

- `node` entity `dc_angle_t` parameter — [rad] voltage angle at each timestep
- `connection` entity `dc_angle_diff_t` parameter — [rad] angle difference across the connection at each timestep

## Slack and penalty values

Slack uses are listed in the [Energy balance in nodes](#energy-balance-in-nodes) section (`Loss of load`, `Excess load` and per-node `slack_up` / `slack_down` parameters with `_t` and annualized variants) and aggregated in [Costs](#costs). Group-level slack tables:

- `group` entity `slack_capacity_margin` parameter — [MW] capacity-margin shortfall in investment periods
- `group` entity `slack_inertia_t` / `slack_inertia` — [MWs] inertia shortfall, timestep / annualized
- `group` entity `slack_nonsync_t` / `slack_nonsync` — [MW] non-synchronous-share shortfall, timestep / annualized
- `group__reserve__upDown` entity `slack_reserve_t` / `slack_reserve` — [MW] reserve shortfall, timestep / annualized

For the sign / direction conventions of slacks, see [Slack convention](dev/slack_convention.md). For how block-aware solves (flex-temporal decomposition) and spatial Lagrangian decomposition feed into these tables, see [Decomposition](dev/decomposition.md) — the result writers always operate on the fine-timeline, full-spatial solution, so the parquet tables look identical to a monolithic run.
