# Model outputs

After every successful solve, the HiGHS solution is handed back to FlexTool in-memory together with the polars `FlexData` bundle that produced the LP. `flextool.process_outputs.write_outputs` then turns that pair into the result tables documented on this page.

The result tables are stored as **parquet** (the canonical store) and a configurable mix of derived artefacts:

| Location | Format | When written | Notes |
| --- | --- | --- | --- |
| `output_parquet/<scenario>/*.parquet` | parquet | always (canonical) | one file per result table; read by the GUI, the scenario-comparison tools, and `read_parquet_dir=True` re-runs |
| `output_csv/<scenario>/*.csv` | CSV | when `output-csv` is enabled | one CSV per parquet, plus `summary_solve.csv` (diagnostic overview), `units.csv` and `datapackage.json` (see [Units in the files](#units-in-the-files)) |
| `output_excel/output_<scenario>.xlsx` | Excel | when `output-excel` is enabled | one workbook per result folder (in the FlexTool window `output_<scenario>_<source number>.xlsx`): the `Units` and `_output_metadata` sheets, then the result sheets in alphabetical order |
| `output_plots/<scenario>/*.png` (or svg/pdf) | image | when `output-plot` is enabled | driven by the bundled `flextool/schemas/default_plots.yaml` |
| `<output-location>/results.sqlite` (or `--results-db-url`) | SpineDB | when `output-spinedb` is enabled (`--write-methods spinedb`) | the processed result tables written into a SpineDB using the FlexTool results schema (`flextool/schemas/spinedb_results_schema.json`); one Spine **alternative** per run (named after the scenario) so multiple runs coexist in one file. Time series land as nested Spine `Map` values on the model entity classes (node / unit / connection / group / model). |

`output_parquet/<scenario>/timeline_breaks.parquet` (columns `period`, `time`) lists the realized steps at which time-series plots start a new line, so a plot leaves a gap there instead of drawing a straight line across it: the step after a gap in the timeline, the start of every representative period (also when two representative periods are adjacent on the timeline, since they represent different base periods), and the first step of a period that does not continue the previous one (also a later year that reuses the same timeset). Rolling windows that follow each other on the timeline, and periods that continue the previous one, are drawn as one line. The file is written also when there are no breaks. Results saved before this file had the `period` and `time` columns are drawn without gaps until the scenario is run again. In scenario comparison plots the scenarios share one time axis, so a break of any selected scenario cuts every scenario's line at that step.

In the per-period tables (`…__d`) the `solve` column names the last complete solve that realized the period's dispatch; a rolling solve is named without its roll number (`dispatch_fullYear_roll`, not `dispatch_fullYear_roll_roll_17`). The capacity tables (`unit_capacity__d`, `connection_capacity__d`, `node_capacity__d`) the investment marginal values (`…_invest_marginal__d`) and the capacity-margin shortfall (`group` `slack_capacity_margin`) name the solve that realized the period's investments instead, for example the investment solve of an investment → dispatch chain. The per-step tables (`…__dt`) name the solve or roll that realized each step. Results saved before this version label the capacity tables with the dispatch solve until the scenario is run again.

Excel limits a sheet name to 31 characters. The longer table names get short sheet names (for example `unit_curtailment__outputNode__dt` is the sheet `unit_VRE_curtailment__dt` and `process__reserve__upDown__node__dt` is `process__reserve__node__dt`; the full list is `SHEET_NAMES` in `flextool/process_outputs/_sheet_names.py`); a longer name of your own is shortened in front of its `__d` / `__dt` suffix. The CSV and parquet file names keep the table names.

Every result table, with its parquet and CSV file names, its index and column levels, the unit of each column and its name in the results database, is listed in the [Results catalogue](results_catalogue.md), generated from the code. Below, a result is named by its results-database parameter (`class` entity `parameter`) where it is written to the results database, else by its table name.

The Excel workbooks hold names (entities, scenarios, alternatives) as text: a name that starts with `=` stays that text and never becomes a formula or a link. The CSV files hold the names exactly as they are, for the programs that read them. A spreadsheet program that opens a CSV file can take a name that starts with `=`, `+`, `-` or `@` for a formula; open the CSV files of a model from someone else as data (an import that sets the columns to text), or use the Excel workbook.

The CSV / Excel / plot files are **derived** from the parquets, so deleting or regenerating them is safe. `summary_solve.csv` is a diagnostic file aimed at a quick overview of a solve — it lists objective decomposition, period weighting, CO2 total, and any non-zero slack uses.


All annualized numbers are scaled to **a full year of operation** by dividing through by `complete_period_share_of_year`. All values are returned in user units — auto-scaling (see [LP scaling pipeline](dev/scaling.md)) and time-block aggregation under flex-temporal decomposition are unwound before they reach the parquets, so users always read MW / MWh / [CUR] in the original problem space.

For stochastic models the parquets carry only the **realised / committed** results of each solve by default — the investment decisions, the operations of the periods before the reveal and, where a branch is designated realized, the realization dispatch of the fan — not the individual forecast branches. A forecasts-only fan has no realized operation, so its operational results are empty (the period label of the fan has no dispatch) and its operating costs and CO2 total are zero; its results hold the investment decisions and the investment and fixed costs (see [Costs of a run of several solves](#costs-of-a-run-of-several-solves)). The same holds for a rolling solve: only the realized steps of each roll (up to `rolling_solve_jump`) are reported, never its look-ahead. This does not depend on the `model` parameter `output_horizon` — the unrealized rows are only in its separate raw dump, below (see also [How to use stochastics](how_to/stochastics.md#how-to-use-stochastics-representing-uncertainty)).

With `output_horizon = yes` (a debug option), `output_parquet/<scenario>/horizon/` holds a **raw LP dump**: one parquet per LP variable and constraint, named as in the model (`v_flow.parquet`, `v_invest_p.parquet`, `nodeBalance_eq.parquet`, …), with every row of every solve — realized and unrealized (forecast branches, rolling look-ahead) alike — appended over the solves and rolls. Columns: the LP index columns (`p`, `n`, `d`, `t`, …), `solve`, `roll`, `realized` (`true` for the rows that also reach the normal results; empty for a variable or constraint without a period index), and `value` + `reduced_cost` for a variable or `dual` for a constraint. The values are raw solver values: primal values per unitsize as in the LP, duals and reduced costs only divided by `scale_the_objective` (HiGHS sign convention, probability and time weights included, empty for a MIP solve) — no unitsize, price or cost conversion. A solve decomposed with Benders is not dumped. The dump keeps every constraint's index until the solve's outputs are written, so it raises peak memory on large models.

- [Units in the files](#units-in-the-files)
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

## Units in the files

The unit of measure of every value column is written with the results; it follows the convention of [Units of measure](reference/entities_and_time.md#units-of-measure) (a column without a unit, such as a share, says `per unit`).

- **Excel, a sheet with several header rows** (a table with two or more column levels, such as `unit__outputNode__dt` with the levels `unit` and `node`; a scenario comparison adds the `scenario` level, so most of its sheets are such sheets): the last header row is `unit of measure` and gives the unit of each column right above its data. A sheet of a table without declared units gets no `unit of measure` row. `pd.read_excel(path, sheet_name=..., header=[0, 1, 2], index_col=[0, 1])` reads such a sheet back, with one more header row than the table has column levels.
- **Excel, a sheet with one header row** (for example `annualized_costs__d`, `node_state__dt`, and in a scenario comparison a table such as `costs_discounted` whose only columns are the scenarios): the header row at `A1` holds the column names, and the data is an Excel Table named `tbl_<sheet name>` (characters other than letters, digits, `_` and `.` become `_`). The row index values are repeated on every row, not merged. To the right of the Table, one empty column apart, a small block gives the units: `all value columns` and the unit when every value column has the same unit, else one line per value column. Power Query reads the Table (`Excel.Workbook(...){[Item="tbl_annualized_costs__d", Kind="Table"]}[Data]`), so the unit block does not enter the query. `pd.read_excel` reads the whole sheet, the unit block included: pass `usecols` with the Table's columns, or drop the columns after the first empty one.
- **Excel, the first sheet `Units`**: one row per result sheet with the sheet name (a link to the sheet), the table name, the unit(s) (a table whose units differ by column lists them with `(by column)`), the semantics (instantaneous, annualized, discounted, …) and a description. `_output_metadata` follows with one row per value column (unit, semantics, tooltip, formula; in a scenario comparison the scenario of the column in its own `scenario` column). The header cells carry no comments: LibreOffice cannot save a workbook with them.
- **CSV**: the CSV files are written as before; `units.csv` next to them has one row per value column: `file`, `position` (the 0-based column of the file), `header_1`, `header_2`, … (the labels of the column in the file's header rows, empty below the file's last header row), `unit` and `semantics`. `summary_solve.csv` has one row (`M CUR`) for its costs. `datapackage.json` is a Frictionless descriptor of every CSV file: the field names are the labels of a column joined with `|` as a Frictionless reader joins them (`coal|east`; a label equal to the one above it is a merged cell and is not repeated, so the index column `period` of a file with three header rows is `period|`), a file with several header rows says so in its `dialect` (`headerRows`, `headerJoin`; header rows that are blank in every column, as in a wide file without value columns, are `commentRows`), and each value field has its `unit`. `frictionless validate datapackage.json` passes. The names `units.csv` and `datapackage.json` cannot be given to a result in the `rename` section of the output settings.

**Nodes with their own unit of measure.** When a node sets `unit_of_measure` (see
[Units of measure](reference/entities_and_time.md#units-of-measure)), the units of its results follow it in every
file: a flow into a water node in m3 is `m3/h`, its storage `m3`, its annual flow `m3/a`, its price `CUR/m3`. A
sum over nodes of different units (a nodeGroup's flows over an electricity and a water node, a connection's net flow
and losses between them) is labelled `Mixed units of measure!`. Reserves, inertia, the non-synchronous share and
the capacity margin stay in MW. A table whose rows are of different units (`unit_capacity__d`,
`connection_capacity__d`, `node_capacity__d`, the nodeGroup and flowGroup tables) gets the units of each row: the
Excel Table ends in a column `unit of measure` with the row's units (for the nodeGroup tables, whose columns differ,
the units of the row's columns joined, for example `m3; m3/a`), `units.csv` gets a `row` column and one row per
column and row entity, and the parquet footer holds them (`flextool_row_units`). With every node in MWh, none of
these is written and the files are as before. The scenario's parquet folder then also holds `_unit_context.json`
(the node units and which nodes each result follows), which the plots, the GUI and the scenario comparison read; a
run whose nodes are all MWh, or a run that writes no parquets, deletes it. A replay (`flextool-write-outputs`) takes
the units from the parquet footers. In a scenario comparison each scenario's columns keep that scenario's units,
and the combined comparison parquets carry them in their footer; a column whose unit differs between the scenarios
gets a note.

In the results database, a run with such a node defines the parameter `node` `unit_of_measure` (default `MWh`) and
writes its value for the nodes that are not MWh; the descriptions of the other parameters give the units of an
MWh node. A results database without this definition comes from runs whose nodes were all MWh.

The plot labels follow the nodes too. The labels of `default_plots.yaml` are unit templates: `{A}` is the node's
amount unit, `{R}` its rate, `{kR}` and `{kA}` the per-kW and per-kWh forms (`{A}/h`, `{A} (annualized)`,
`Currency/{A}`, `CUR/{kR}`). For MWh nodes they read as before (`MWh/h`, `MWh (annualized)`, `CUR/kW`). With a node
in another unit each subplot gets the unit of its nodes; a plot that sums, stacks or averages nodes of different
units says `Mixed units of measure!`, and lines of several units list them (`MWh/h; m3/h`). The dispatch plots'
y-axis is `m3/h` for a water node and `MWh/h` by default. A label without tokens in your own `plot_settings.yaml`
(or in a copy of the old default settings) is shown as written and does not follow the nodes.

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
- table `years_represented__d` (not in the results database) — [a] the years each period represents; an annualized per-period value times these years gives the period's total over the horizon
- table `period_share_of_year__d` (not in the results database) — [per unit] for each realised period, the share of a year it covers (its complete hours / 8760, in the solve that realises it): the annualized per-period (`…__d`) results (flows, energy, annualized costs, emissions, start-ups) are the period's weighted sums over its time steps divided by this share; capacities, prices, averages, ratios and discounted costs are not scaled. A 48-hour period has 0.005479, so its annualized per-period values are 182.5 times its 48-hour sums; a full-year period has 1. Results saved before this version have no such table.

### Costs of a run of several solves

When `model.solves` holds several solves (a chain of investment solves, an investment solve and its dispatch
solves, a stochastic solve and its realization solve), the reported costs are those of the run as a whole, the same
as one solve over the same periods with the same decisions would report:

- **One discount base, the start of the run.** Every discounted cost counts its years from the first period of the
  run, not from the first period of the solve that reports it.
- **Each realized period is reported once, by the solve that realizes it**: its operating and penalty costs come
  from that solve's dispatch, except the capacity margin penalty.
- **The capacity margin penalty** (and its slack, `slack__capacity_margin__d`) of a period is that of the solve that
  decided the period's investments, also when a later dispatch solve realizes the period (a dispatch solve has no
  capacity margin). The forecast branch periods of a stochastic fan are not realized, so their shortfalls and
  penalties stay out of the costs; `capacity_margin__d` has them.
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

Every solve also discounts its own objective from the start of the run (as one solve over the same periods
does), so the printed objective of each solve, the cumulative CO2 cap price and the investment duals
(`invest_marginal`, the value of one more MW of capacity) of every solve are on the run's discount base. The
printed objectives still cover each solve's own horizon (its look-ahead periods too), so `cost_discounted_total` is
not the sum of the solves' objectives. The CO2 price of a period is its dual divided by the discount factor of the
solve that realizes the period. A CSV-dump reload has no run-level cost data: its reported costs are discounted
from the start of the run with the solve's own factors over the solve's own periods, and its annualized
investment, retirement and fixed costs of new and retired capacity are 0 (a warning is printed).

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
- `unit__node` entity `startup_energy_t` parameter — [MWh] energy that the unit's start-ups take from the node in each timestep (`unit__inputNode` `startup_input_method` *startup_and_input* or *startup_only*: `startup_energy` × MW started)
- `unit__node` entity `startup_energy_annualized` parameter — [MWh] the same summed over the period and scaled to a full year. The start-up input is also in the unit's input flow from the node (`flow_t`, `flow_annualized`), the commodity cost, the CO2 emissions and the node balance; a *startup_only* arc has no other flow

## Connection online and startup

Only for connections with a `startup_method`:

- `connection` entity `online_t` parameter — [count] units of the connection online at each timestep
- `connection` entity `online_average` parameter — [count] average online status over the period (weighted by step duration)
- `connection` entity `startup_cumulative` parameter — [count] startups during the period scaled to a full year (as for units)

## Unit curtailment and VRE potential

For VRE units (units that use an `upper_limit` profile) the model also reports curtailment and potential generation against each output node:

- table `unit_VRE_potential_outputNode_dt_ee` (not in the results database) — [MW] potential VRE generation at the timestep (`capacity × profile`)
- table `unit_VRE_potential_outputNode_d_ee` (not in the results database) — [MWh/a] potential VRE generation summed over the period's time steps (with the timestep weights) and scaled to a full year, like the flows
- `unit__node` entity `curtailment_t` parameter — [MW] potential minus actual generation at the timestep
- table `unit_curtailment_outputNode_d_ee` (not in the results database) — [MWh/a] curtailed energy summed over the period's time steps (with the timestep weights) and scaled to a full year, like the flows; the *total* plot multiplies each period by its years represented
- table `unit_curtailment_share_outputNode_dt_ee` (not in the results database) — [0–1] timestep curtailment / potential
- `unit__node` entity `curtailment_share` parameter — [0–1] period curtailment / period potential

## Node group results

`group` entities with `print_indicators = yes` produce indicator tables aggregating flows / inflows / slacks over their member nodes:

- `group` entity `indicator` (period table) — gives a set of indicators for all `node` members. Each share is a ratio of two energies summed over the period, weighted by the time-step weights: `share = Σ_t w·X / Σ_t w·Y` (never a sum or an average of the time-step shares):
    - *Loss of load share* — [0–1] upward slack ÷ inflow over the period
    - *VRE share of demand* — [≥ 0] VRE generation into the member nodes ÷ inflow over the period (can exceed 1 when VRE is exported out of the group)
    - *Excess load share* — [0–1] downward slack ÷ inflow over the period
    - *Curtailed VRE of demand* — [≥ 0] curtailed VRE ÷ inflow over the period (can exceed 1 when the curtailment is larger than the demand)
    - *Annualized inflow* — [MWh] sum of `inflow` to member nodes scaled to a year
    - *Curtailed VRE of potential VRE* — [0–1] curtailed VRE ÷ potential VRE generation over the period, both on the arcs of the VRE units with an `upper_limit` profile into the member nodes (the group sum of `unit_curtailment_outputNode` and `unit_VRE_potential_outputNode`)
- table `nodeGroup_g_p` (`group_node__horizon`, not in the results database) — the same indicators over the whole horizon: each share is the ratio of the energies over all periods, each period's weighted sum scaled to a year and multiplied by its years represented (`Σ_d y·Σ_t w·X / share_d ÷ Σ_d y·Σ_t w·Y / share_d`), not an average of the period shares; *Inflow (horizon total)* [MWh] replaces the annualized inflow. In a stochastic fan each branch period counts with its years, so the horizon total adds up all the paths
- table `nodeGroup_gdt_p` (timestep table, not in the results database) — exposes the timestep-level building blocks of the above: *Loss of load* [MWh/step], *VRE generation* [MWh/step], *Excess load* [MWh/step], *Curtailed VRE* [MWh/step], *Timestep inflow* [MWh/step], *Curtailed VRE of potential VRE* [0–1], *Annualized inflow* [MWh], *VRE share of demand* [0–1]
- `group` entity `VRE_share_t` parameter — [≥ 0] share of inflow served by VRE sources at each timestep (one column per node group; 0 at a time step without inflow)
- table `nodeGroup_VRE_share_d_g` (not in the results database) — [≥ 0] VRE share of the period's inflow: VRE generation ÷ inflow, both summed over the period with the time-step weights (equals *VRE share of demand* above)
- table `nodeGroup_VRE_share_g` (`group_node_VRE_share__horizon`, not in the results database) — [≥ 0] VRE share over the whole horizon: the total VRE energy ÷ the total inflow, each period scaled to a year and multiplied by its years represented
- `group` entity `flow_annualized` parameter — [MWh] annualized sum of `inflow` to member nodes
- `group` entity `flow_t` parameter — [MWh/step] timestep inflow to member nodes

`group` entities with `print_dispatch = yes` produce a multi-column dispatch table over the group:

- `group` entity `sum_flow_t` parameter — [MWh/step] dispatch decomposed by `(type, item)` where `type` is one of: *slack* (upward/downward), *from_unit*, *from_unitGroup*, *to_unit*, *to_unitGroup*, *from_connection*, *from_connectionGroup*, *to_connection*, *to_connectionGroup*, *inflow*, *internal_losses* (units/connections/storages)
- `group` entity `sum_flow_annualized` parameter — [MWh] same decomposition, annualized period totals

## Flow group results

`flowGroup` entities whose `flow_aggregator` is `standalone_aggregator_only` or `both` produce aggregate flow statistics over their member `(process, node)` legs (listed via `flowGroup__unit__node` / `flowGroup__connection__node`):

- `group_flow__d.csv` — per-(flowGroup, period) totals:
    - `cumulative_flow` parameter — [MWh/a] sum of |flow| over the period for all member legs, with the time-step weights, scaled to a full year
    - `average_flow` parameter — [MW] average power (`cumulative_flow` / 8760 h)
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

- `unit`, `connection`, `node` entities `invest_marginal` parameter — [CUR/kW, CUR/kWh for storage nodes, as `invest_cost`] effective dual of the investment decision: zero means the model is at the unconstrained optimum; positive means an active upper bound (per-entity, per-group, period, total or cumulative caps are summed automatically); negative means the entity is not built and is unprofitable to build (the v_invest column's own reduced cost) — this is the ordinary reading for an unbuilt entity, not an artefact; a numerical artefact instead looks like a non-zero value on a *built* (or at-bound) entity, or an inconsistent sign across these regimes, and is worth investigating

## CO2 emissions

- table `CO2__` (not in the results database) — [Mt] horizon-total CO2 across all groups and units (uses `years_represented_d` to convert annualized emissions to horizon totals). In a stochastic solve with more than one scenario path this is reported per path (one column per path) instead of pooling every path's emissions into one number — the anchor's shared pre-reveal history and each forecast branch's own periods are alternative futures, not consecutive years to add together.
- `group` entity `CO2_annualized` parameter — [t/yr] annualized CO2 emissions caused (or removed) by units and connections in the group
- `unit` entity `CO2_annualized` parameter — [t/yr] annualized CO2 emissions of the unit's flows, by commodity and node

## Reserves

- `unit__reserve__upDown__node` and `connection__reserve__upDown__node` `reservation_t` parameter — [MW] reserve provision at each timestep
- `unit__reserve__upDown__node` and `connection__reserve__upDown__node` `reservation_average` parameter — [MW] the time-step-weighted average MW over the period's realized steps: Σ_t w·Δt·MW / Σ_t w·Δt with `w` the time-step weight (representative periods weigh by the base periods they stand for). The steps of a solve that left reserves out (`reduce_system_detail_method`) are left out of both sums, not counted as 0 MW. This table does not multiply by `reliability`; the group tables below do.
- `group__reserve__upDown` entity `requirement_t` parameter — the reserve requirement of each reserve, direction and group at each time step (`reserve__upDown__group__dt`, table `nodeGroup_reserve_dt_eegp`), from the group's requirement constraints evaluated at the solution, by quantity [MW]:
    - *requirement_timeseries*, *requirement_dynamic*, *requirement_n_1* — the need of each method the group has (`reserve_method`; a column of a method the group does not have is left out): the `reservation`; the flows into and out of the group's nodes × `increase_reserve_ratio` plus the group's demand × its `increase_reserve_ratio`; the largest failure, the flow of a failing unit or connection × `large_failure_ratio` (one requirement per failing process; the raw parquet `output_raw/reserve_group__<solve>.parquet` names the largest one in `n_1_process`)
    - *requirement* — the need of the binding requirement, the one with the largest deficit (need − counted reserve; among equal ones timeseries, then dynamic, then the failing processes by name; the raw parquet names it in `binding_method` and, for a failure, `binding_process`). This is not always the largest need: when no requirement is short, the binding one is the one closest to being short, and its need can be below another method's. For example, a timeseries need of 90 MW with 100 MW counted (10 MW to spare) and a large-failure need of 80 MW with 85 MW counted (5 MW to spare) give *requirement* = 80 MW. The largest need of the group is the largest of the *requirement_timeseries*, *requirement_dynamic* and *requirement_n_1* columns
    - *provided* — the reserve of the group's providers: Σ reserve × unitsize × `reliability` over the units and connections that hold the reserve at a node of the group (a provider at a node counts in every group that has the node)
    - *counted* — the reserve the binding requirement counts: *provided*, but for a large-failure requirement without the failing process's own reserve (it cannot cover its own failure). So *shortfall* = *requirement* − *counted* when there is a shortfall
    - *shortfall* — [MW] the requirement not met (the same as `slack_reserve_t`)
- `group__reserve__upDown` entity `requirement` parameter — the same per period (`reserve__upDown__group__d`, table `nodeGroup_reserve_d_eegp`): *requirement_avg*, *provided_avg*, *counted_avg* and *shortfall_avg* are time-step-weighted averages over the period's realized steps (Σ_t w·Δt·X / Σ_t w·Δt, the steps of a solve that left reserves out not counted), *requirement_max* and *shortfall_max* the largest step values (not weighted). Compare *requirement_avg* with *counted_avg*, not with *provided_avg*: with a large-failure requirement the failing unit's own reserve is in *provided* but not in *counted*. *requirement_avg* averages the binding requirement, which can come from a different method at different steps.
- The reserve price of the group (the sum of the duals of its requirements) is `reserve_price__upDown__group__dt` (table `reserve_prices_dt_ppg`, see [Prices](#prices)).
- table `nodeGroup_reserve_provider_dt_eegee` (`reserve__group__provider__dt`, not in the results database) — [MW] the counted reserve (× `reliability`) of each unit or connection at each node for each reserve, direction and group at each time step; per group they add up to *provided*. With `reliability` below 1 it is less than the reserve the unit holds (`reservation_t`).

A step whose solve left reserves out has empty cells in these tables. A reserve held by storage with a `reserve_duration` adds no requirement; it shows as a lower *provided*. After each solve the run log prints `reserve requirement check: solve ..., reserve ..., group ..., step (...): deficit ... MW, shortfall ... MW` for a step where the evaluated requirement and the solver's shortfall disagree (that should not happen; on an accepted near-optimal solve it can be solver tolerance, and the line says so).

## Inertia and non-synchronous generation

- `group` entity `inertia_t` parameter — [MWs] total inertia in the group of nodes at each timestep
- `group` entity `inertia_largest_flow_t` parameter — [MW] largest individual flow coming into the group of nodes with `has_inertia`
- `group` entity `inertia_unit_node_t` parameter — [MWs] inertia contribution per (unit, node) at each timestep, one column per contributing `(unit, node)`

## Ramps

A ramp is how much a unit's flow changed from the previous time step: the flow at this step minus the flow at
the previous step. It is positive when the flow rose and negative when it fell. FlexTool writes it for every input
and output of every unit, whether or not the unit has ramp limits.

| Table (results database: `unit__node` `ramp_t`) | Unit | Sign | Empty when |
|---|---|---|---|
| `unit_ramp__outputNode__dt`: each unit output | MW per time step (or the node's unit) | + the output rose, − it fell | the step has no previous step: the first step of a solve, the step after a gap in the timeline, the first step of a representative period, or a period that does not continue the previous one |
| `unit_ramp__inputNode__dt`: each unit input | MW per time step (or the node's unit) | + the unit took more from the node, − less | as above |

In a rolling solve, the first step of a window is compared with the last realized step of the previous window. No
column is a sum of the others. Earlier versions also had a node-level "ramp room" table, `node` `ramp_t`, with
stacked headroom levels. It is no longer produced.

**From MW per step to the ramp speed.** `ramp_speed_up` and `ramp_speed_down` are a share of capacity per minute
(see [Units](reference/units.md)). To compare, divide the ramp by the capacity of that input or output and by the
minutes between the two step centres (60 × the average of the two step durations in hours). Example: a 500 MW
unit on hourly steps has `ramp_t` = −30 MW. That is 30 / 500 / (60 × 1) = 0.001, so it ramped down at 0.1 % of
its capacity per minute, or 30 / 60 = 0.5 MW per minute. With 3-hour steps the same −30 MW is
30 / 500 / 180 ≈ 0.00033. The ramp limit also counts the reserve the unit holds in that direction, so a unit can be
at its limit with a `ramp_t` smaller than its ramp speed allows.

## DC power flow

When the network contains DC-power-flow connections / nodes (`is_dc_power_flow = yes`):

- table `dc_angle_dt_e` (not in the results database) — [rad] voltage angle at each timestep
- table `dc_angle_diff_dt_e` (not in the results database) — [rad] angle difference across the connection at each timestep

## Slack and penalty values

Slack uses are listed in the [Energy balance in nodes](#energy-balance-in-nodes) section (`Loss of load`, `Excess load` and per-node `slack_up` / `slack_down` parameters with `_t` and annualized variants) and aggregated in [Costs](#costs). Group-level slack tables:

- `group` entity `slack_capacity_margin` parameter — [MW] capacity-margin shortfall in investment periods. It is the shortfall of the solve that decided the investments: in an investment → dispatch chain it is the investment solve's, shown under the investment solve, and its penalty is in the costs (*capacity margin penalty*). The forecast branch periods of a stochastic fan are not realized, so this table and the costs leave out their shortfalls; `capacity_margin__d` has them.
- `group` entity `capacity_margin` parameter — the capacity margin of every solve with investment periods (`capacity_margin__d`, table `nodeGroup_capacity_margin_d_g`), per solve, group and investment period (look-ahead and forecast branch periods included, not only realized ones) at the binding step `binding_time`, the time step with the largest deficit (requirement − counted; on a group whose row is averaged over coarse steps, the first time step of the coarse step): *margin* [MW] (the `capacity_margin`, or `capacity_margin_share` × the peak), *peak_demand* [MW] (the peak demand of the group's nodes over the period's original timeline), *demand* [MW] (at the binding step), *requirement* [MW] (margin + demand), *firm* and *vre* [MW] (the counted capacity of firm units and of VRE units at their profile, with availability), *storage_flow*, *unit_flow*, *connections_net* and *consumption* [MW] (the flows of storage-backed units minus the slack energy of the storage nodes feeding the group, of the other units counted by their flow, of connections and of the group's consumption, signed as in the node balance), *dumped* [MW] (minus the downward slack), *counted* [MW] (their sum), *shortfall* [MW] (max(0, requirement − counted) at the binding step: 0 when the margin holds), *penalty_cost* [M CUR] (shortfall × `penalty_capacity_margin` × 1000 × inflation, as in the costs) and *capacity_price* [CUR/kW/a] (the shadow price of the margin: what one more kW of margin would cost; it equals `penalty_capacity_margin` while there is a shortfall, and is empty for a MIP). In the results database the binding step is not carried (the value is category → solve → period); it is in the CSV and parquet tables.
- `group` entity `capacity_margin_contribution` parameter — [MW] what each unit, connection and node counts in the group's capacity margin at its binding step (`capacity_margin_entity__d`, table `nodeGroup_capacity_margin_entity_d_g`), by class: *firm*, *vre*, *storage*, *storage_slack* (minus the slack energy of a storage node feeding the group, per hour of the step), *unit_by_flow*, *connection*, *consumption* and *dumped* (a node's downward slack; a node that dumped nothing is left out). They add up to the columns of `capacity_margin__d`: *firm* → *firm*, *vre* → *vre*, *storage* and *storage_slack* → *storage_flow*, *unit_by_flow* → *unit_flow*, *connection* → *connections_net*, *consumption* → *consumption*, *dumped* → *dumped*. A row sums to *counted*. A unit that you expect to count but that is missing, or counts by its flow, shows which rule of the [capacity margin](reference/groups.md#the-capacity-margin) applies to it.
- `group` entity `slack_inertia_t` / `slack_inertia` — [MWs] inertia shortfall, timestep / annualized
- `group` entity `slack_nonsync_t` / `slack_nonsync` — [MW] non-synchronous-share shortfall, timestep / annualized
- `group__reserve__upDown` entity `slack_reserve_t` / `slack_reserve` — [MW] reserve shortfall, timestep / annualized; the requirement it falls short of, and what each provider counts, is in [Reserves](#reserves) (`reserve__upDown__group__dt`)

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
- A dispatch plot's CSV has a `period` and a `time` column for the steps of the plotted time window (the periods
  one after another), the flows, and the demand line as a `Demand` column, as in the legend (`Demand (line)` if a
  flow is already called `Demand`). `dispatch_data.xlsx` (scenario comparison with `--write-dispatch-xlsx`) has one
  sheet per dispatch plot with the same `period` and `time` columns, a value on every row.
- If a CSV cannot be written, the log has a warning naming the file, and the plots are still made.
- In `plot_settings.yaml`, only `write_plot_data: true` (no quotes) turns this on; a quoted `"true"` or any value
  that is not `true` or `false` leaves it off with a warning.

The CSVs are removed with the plots when the plots are made again.
