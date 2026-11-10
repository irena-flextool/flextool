# Reading results

A FlexTool run writes many result tables. This page says where to look first, in which order, and what the main tables and their units mean. [Model outputs](results.md) describes every result, and the [Results catalogue](results_catalogue.md) lists every table with its files, columns and units.

## Where the results are

- **Result viewer.** In the FlexTool window, the Result viewer shows the plots of one scenario (key V) or compares several (key X); see [Result viewer](flextool_gui_interface.md#result-viewer).
- **Files.** Each run writes its tables to `output_parquet/<scenario>/` (in the FlexTool window `<scenario>_<source number>`), and on request as CSV (`output_csv/`), Excel (`output_excel/`) and plots (`output_plots/`).
- **Plot data.** With **Write plotted data (CSV)**, each plot comes with a CSV file of the data drawn in it, which is often the easiest table to take into a report; see [Plotted data as CSV](results.md#plotted-data-as-csv).

**Table names.** The name says what a table holds: the entity classes first (`unit`, `node`, `connection`, `group`), then the resolution. `__dt` is per period and time step, `__d` per period. For example, `unit__outputNode__dt` is the output of each unit to each node in each time step, and `unit_capacity__d` the capacity of each unit in each period.

## Check the results in this order

### 1. Did the run succeed?

A run that did not finish has no usable results. The FlexTool window shows the run's state in the execution jobs; `output_parquet/<scenario>/_run_status.json` holds it, and `_run.log` the messages of the run. Read the warnings in the log: they name data that the model did not use as you may have meant.

`summary_solve.csv` (in the CSV output) gives an overview of each solve: the objective, the costs split into operations, investments, fixed costs and penalties, the discount factors of the periods, the CO2 total and every slack the model used.

### 2. Are there slacks?

A slack is a requirement the model could not meet, at a penalty cost: unserved demand, excess energy, a reserve, inertia or capacity margin shortfall. With a slack, the results show a system that does not work, so find the cause before reading on.

- `summary_solve.csv` lists the slacks used and the *Penalty (slack) costs*.
- `slack__upward__node_state__dt` and `slack__downward__node_state__dt` [MW]: unserved demand (*loss of load*) and excess energy at each node; the `__d` tables give the period totals.
- `slack__reserve__upDown__group__dt`, `slack__inertia_group__dt`, `slack__nonsync_group__dt` and `slack__capacity_margin__d`: the shortfalls of reserves, inertia, the non-synchronous limit and the capacity margin.
- For an adequacy shortfall (the capacity margin of a group), the run log prints a line after each investment solve that leaves one (`capacity margin shortfall: solve ..., group ..., period ...: ... MW at ...`). `capacity_margin__d` has, per solve, group and investment period, the requirement at the binding time step, what each class of units and connections counts there and the shortfall; `capacity_margin_entity__d` has what each unit, connection and node counts. An investment solve keeps its rows under its own name, also when a dispatch solve realizes the period.

Penalties are large on purpose, so a small slack can dominate the costs. See [Slack and penalty values](results.md#slack-and-penalty-values).

### 3. What does the system cost?

- `annualized_costs__d` [M CUR per year]: the costs of one year of each period by category (investment annuities, fixed costs, fuel, CO2, other operating costs, start-ups, penalties). Use it to compare periods.
- `costs_discounted__d` and `costs_discounted` [M CUR]: the same costs discounted with `inflation_rate` and multiplied by the years each period represents, per period and in total. This is what the model minimises.
- `cost_unit_annualized__d` and the tables for connections and nodes: the costs of each entity.

See [Costs](results.md#costs) and [Discounting between milestone years](advanced_concepts.md#discounting-between-milestone-years).

### 4. What capacity is there?

- `unit_capacity__d` and `connection_capacity__d` [MW], `node_capacity__d` [MWh, storages]: the *existing* capacity, the capacity *invested* (coming online in the period), *divested* and the *total* in service.
- `invest_marginal`: the value of one more MW of capacity. Negative for a technology the model does not build: it does not pay off.

See [Capacity and investment results](results.md#capacity-and-investment-results).

### 5. How is it operated?

- `unit__outputNode__dt` and `unit__inputNode__dt` [MW]: the output and input of each unit in each time step; `connection__dt` the flows on connections.
- `node__dt` [MW] and `node__d` [MWh per year]: the energy balance of each node by category (units, connections, storage losses, inflow, loss of load).
- `node_state__dt` [MWh]: the content of each storage.
- The dispatch plots of a node group with `print_dispatch` *yes* stack all flows into and out of the group.

See [Energy flows](results.md#energy-flows) and [Energy balance in nodes](results.md#energy-balance-in-nodes).

### 6. Prices and emissions

- `node_prices__dt` [CUR/MWh]: the price of energy at each node and time step, the cost of one more MWh of demand there. A price at the level of the penalty means there was a slack.
- `group_CO2__d` [t per year]: the CO2 of each CO2 group; `co2.csv` [Mt]: the CO2 total over the years the periods represent. The CO2 price of a binding cap is in the table `co2_price_period_d_g` [CUR/t].

See [Prices](results.md#prices) and [CO2 emissions](results.md#co2-emissions).

## Units

| Unit | Used for |
|---|---|
| MW | flows and capacities, in each time step |
| MWh | storage capacity and content; flows summed over a period (`__d`), usually scaled to a full year |
| M CUR | costs per period and in total (millions of the currency of your inputs) |
| CUR | costs per time step (`costs__dt`) |
| CUR/MWh | prices |
| t/a, Mt | CO2 per year, CO2 over the whole run |

**Annualized or discounted.** An *annualized* value is for one year of the period: the hours of the period are scaled to the 8760 hours of a year. A *discounted* value is a present value: it is multiplied by the years the period represents and by the period's discount factor.

**Your units.** FlexTool does not convert units: a flow is in MW only if your inputs are in MW and MWh. A water node in m³ gives flows in m³/h. The parameter units are in the [Parameter catalogue](parameter_catalogue.md).
