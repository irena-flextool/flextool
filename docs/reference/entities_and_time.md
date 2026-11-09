# Main entities to define a power/energy system

Elemental entities (one dimensional):

- [`node`](nodes.md): maintain a balance between generation, consumption, transfers and storage state changes (nodes can also represent storages)
- [`unit`](units.md): power plants or other conversion devices that take one or more inputs and turn them into one or more outputs
- [`connection`](connections.md): transmission lines or other transfer connections between nodes
- [`commodity`](commodities.md): fuels or other commodities that are either purchased or sold at a price outside of the model scope
- [`profile`](units.md#units-constrained-by-profiles): timeseries that can be used to constraint the behaviour of units, connections or storages
- [`reserve`](reserves.md): reserve categories to withhold capacity to cope with issues outside of model scope

Entities with two or more dimensions:

- `unit__inputNode` and `unit__outputNode`: defines the inputs, outputs and their properties for the conversion units
- `connection__node__node`: defines which nodes a connection will connect
- `unit__node__profile` and `connection__profile`: defines a profile limit (upper, lower or fixed) for an energy flow
- `node__profile`: defines a profile limit (upper, lower, or fixed) for the storage state of the node
- `commodity__node`: defines if a node is a source or sink for a commodity
- `reserve__upDown__unit__node` and `reserve__upDown__connection__node`: reserve capacity from a source to the target node

See the other pages of this section for more detailed explanations.

![Simple example grid](../img/tutorial/simple_grid.png)


## How to define the temporal properties of the model

### Timesteps and periods

FlexTool has two different kinds of time varying parameters. The first one represents a regular timeline based on timesteps. The duration of each timestep can be defined by the user. There can be multiple timelines in the database - the user needs to define which timeline to use (and what parts of the timeline should be used, as will be explained later). The timestep names in the timeline are defined by the user - they can be abstract like 't0001' or follow a datetime format of choice. However, the timestep names between different timelines must remain unique (usually there should be only one timeline in a database and therefore no issues).

The second time varying dimension is `period`, which is typically used to depict assumptions about the future. One model can include multiple `solves` that the model will solve in sequence (to allow multi-stage modelling). Each solve can include multiple `periods` (so that the user can change parameter values for different parts of the future).

A parameter of particular type can be either constant/time-varying or constant/period-based. For example `inflow` is either a constant or time-varying, but it cannot be period-based (a period-keyed `inflow` is rejected; use `annual_flow` for demand that varies by period).

**Recommendation — name the Map index for parameters that accept both `period` and `time`.** A few parameters accept *either* a period map or a time series (for example the flowGroup `min_instant_flow` / `max_instant_flow` limits, which accept "Constant, period, time or period+time"). Period and time Maps look structurally identical in Spine Toolbox — both are 1-dimensional Maps — so FlexTool tells them apart from the Map's `index_name`. Set it explicitly to **`period`** or **`time`** when authoring such a Map. If you leave the default (Spine stores it as `x`), FlexTool falls back to inspecting the Map's index *values* and matching them against the model's period and timestep names to guess the intended axis; this works when the keys clearly belong to one set, but is ambiguous if they don't. The guess is made per entity: a map whose keys are neither periods nor timesteps of the solve gives that entity no value in that solve (the same as if nothing had been authored for it, so the parameter's own default applies) — it never affects any other entity's value. Keys that match *both* a period and a timestep name are an error. Setting `index_name` removes the guesswork. (Parameters that accept only one of the two — e.g. `inflow` is time-only — are never ambiguous and need no `index_name`.)

### Timesets

Timesets pick one or more sections from the `timeline` to form a `timeset`. Each timeset defines a start and a duration. The aim of timesets is to allow the modeller to create models with representative periods often used in the investment planning.

![Time structure](../img/tutorial/time_structure.png)

### Definitions

- `model`: model defines the sequence of solves to be performed (e.g. first an investment solve and then a dispatch solve)

  - *solves*: sequence of solves in the model represented with an array of solve names.
  - *inflation_rate*: [e.g. 0.02 for 2%] Model-wide inflation rate applied to all future costs. When inputs are in real (constant-price) terms, set to 0 (default). When inputs are in nominal terms, set to expected inflation.
  - *inflation_offset_investment*: [years] Offset from the period (often year) start to the first payment of the investment cost annuity.
  - *inflation_offset_operations*: [years] Offset from the period (often year) start to the payment of operational costs.
  - *periods_available*: (Optional) Array of periods available for the model. Use this for periods that are in the data, but are not in period_timeset.
  - *max_flow_for_unconstrained_variables*: [MW] Upper bound for capacities and flows that have no other limit: how much capacity an entity with `invest_method` `invest_no_limit` or `invest_retire_no_limit` (or `cumulative_limits` without `cumulative_max_capacity`) can add on top of its `existing` capacity, and the flow on a unit's input or output whose `conversion_flow_coeff` is 0. The value is used as given; with several model entities the largest value is used. Keep it large enough not to limit the real solution and small enough to avoid numerical problems (see [dev/scaling.md](../dev/scaling.md)). Default 1,000,000.
  - *debug*: Instruction set for performing model debugging and testing.
  - *version*: Database schema version. Written by the migration chain — do not edit manually.
  
- `solve`: each solve is built from an array of periods (e.g. one period for 2025 and another for 2030). Periods use timesets to connect with a timeline. Parameters are split into two parameter-group tags in the master template — `solve_basics` (knobs a typical user touches) and `solve_advanced` (rolling/nested/stochastic structures, solver options, cross-solve handoff). The split is purely cosmetic in the Spine DB editor; behaviour is identical.

  - *period_timeset*: map of periods with associated timesets that will be included in the solve. Index: period name, value: timeset name.
  - *new_stepduration*: [hours] Creates a new `timeline` for this `solve`, with time steps of this length. See also the group-level `new_stepduration` for flex-temporal decomposition.

    **How the data is combined.** The new timeline sums `inflow` over the time steps that form each new time step. A constant `inflow` counts once per time step it combines. Other time series, like `profile`, are averaged over the steps, weighted by their durations.

    **How the steps are formed.** Steps are aggregated from the start of each `timeset_duration` block. A block that is not a multiple of the step ends with a shorter step. A gap between blocks stays a gap. When several timesets share the timeline, each period keeps its own time steps.

    **Rules and errors.** The `new_stepduration` must be a whole multiple of the timeline's time steps. On a timeline with time steps of different lengths, every new time step must end where a time step of the timeline ends. FlexTool stops with an error otherwise. All timesets used by the solve must resolve to the same underlying timeline. A timeset used by this solve cannot be used by a solve without `new_stepduration` or with a different one.

  - *realized_periods*: these are the periods the model will 'realize' - i.e., what periods will be reported in the results from this solve
  - *realized_invest_periods* Array of the periods that will realize the investment decisions. If this is not defined when the invest_periods exist, the realized_periods are used to realize the invests as well
  - *invest_periods*: array of periods in which investments can be decided in this solve (applies only to entities that can be invested in). With a construction lead time, the capacity comes online later.
  - *years_represented*: Map to indicate how many years the period represents before the next period in the solve. Used for discounting. Can be below one (multiple periods in one year). Index: period, value: years.
  - *solver*: choice of a solver. HiGHS is the default and only fully supported backend (called via the `highspy` Python bindings). CPLEX can still be wired in as an advanced option for licensed users via `solver_precommand` / `solver_arguments`; see the how-to.
  - *solver_arguments*: Map of solver option name → value, forwarded verbatim to the active solver. This is where the former `highs_*` shortcuts now live. For HiGHS, set `solver` to choose the method ('simplex' or 'ipm' which is interior point method; use 'choose' for MIP models, since 'simplex' and 'ipm' will not work), `parallel` ('on' or 'off') to control parallelisation of single solves (it can be better to turn it off when executing multiple scenarios in parallel), and `presolve` ('on' or 'off') to toggle presolve (which can have a large impact on solution time when solves are large). These only affect speed, not results.
  - *solve_mode*: a single solve or a set of rolling optimisation windows solved in a sequence
  - *timeline_hole_multiplier*: [-] default 1.0. Multiplier applied to the inverse-step-duration term in the `nodeBalance_eq` for variable-step timelines. Tunes how strongly an unrepresented gap between two timesteps is amortised into the storage state equation. Leave at the default unless investigating large-step / sparse-timeline numerical issues.
  - Rolling window parameters:

    - *rolling_solve_jump*: Hours, (Required if rolling_window solve). Interval between the start points of the rolls. Also the output interval. This should be smaller than the horizon
    - *rolling_solve_horizon*: Hours, (Required if rolling_window solve). The length of the horizon of the roll. How long into the future the roll sees. For an individual roll, horizon is the solve length and jump is the output length.
    - *rolling_duration*: Hours, (Optional). Duration of rolling, if not stated, assumed to be the whole timeline of the solve

  - Nested solve sequence parameters:

    - *contains_solve*: Array of solves that are run with after this solve using the realized data of this solve. Read 'How to use Nested Rolling window solves (investments and long term storage)'
    - *fix_storage_periods*: Array of periods where this solve produces a storage-state handoff for the contained solve at the end of each listed period. The handoff carries any subset of three metrics — `fix_storage_quantity`, `fix_storage_price`, `fix_storage_usage` — selected per node by the node parameter `storage_nested_fix_method`.
  
  - Stochastic parameters:

    - *stochastic_branches*: 4D-map defining the solve's **forecast branches** — the probability-weighted possible futures. Index columns: *period*, *branch*, *analysis_time* (the branch's start timestep), *realized* (yes/no), and the value is the branch's **probability weight**. The weights are normalized to sum to one across the forecast branches at each start time. The *realized* flag is an **optional designation** (at most one branch per start time) marking which branch's data are treated as the actuals for the committed / realized-dispatch results; a realized-flagged branch is **not** a weighted member of the forecast fan. A period-indexed value (a map keyed by period, e.g. efficiency, startup cost, commodity price, availability, variable cost or a per-period limit) also applies to that period's forecast branches; a value authored for a branch period itself takes precedence. See 'How to use stochastics' for more information.
    - *stochastic_invest_method*: `none` (default) or `recourse`. `none` makes a single shared here-and-now investment plan across all forecast branches (operations still fan per branch). `recourse` gives per-branch (wait-and-see) investment: each forecast branch invests on its own data at and after the branching period, with probability-weighted investment costs and per-branch investment limits, and reports EVPI/VSS-style stochastic value. Forecast-branch (unrealized) investments are not in the normal outputs, only in the raw dump of `model.output_horizon` (a debug flag). Incompatible with Benders decomposition. See 'How to use stochastics' for the interpretation caveats.
  
  - For commercial solvers:

    - *solver_precommand* the commandline text in front of the call for the commercial (CPLEX) solver. For a possibility of reserving a floating licence for the duration of the solve
    - *solver_arguments* Array of additional commands passed to the commercial solver. Made for setting optimization parameters.

- `timeset`: timesets are sets of time with a start (from timeline) and a duration (number of time steps)

  - *timeset_duration* a map with index *timestep_name* that starts the timeset and value that defines the duration of the timeset (how many timesteps)
  - *timeline* The name of the timeline that the timeset uses. (String)
  - *timeset_weights*: Optional per-timestep weight map (index: timestep name, `index_name` *time* or left empty; value: float) applied to cost and slack terms in the objective. Use for non-RP models where timesteps represent unequal fractions of the year — e.g. a coarse OSeMOSYS-style timeslice structure where wet-season steps cover more year-hours than dry-season ones. Weights are normalized per period to sum to 1 and then scaled by the number of active timesteps, so uniform input reproduces weight = 1 per step (the default). Must not be combined with `representative_period_weights` on the same timeset — the runner errors out if both are set. The stochastic forecast branches of a period use the weights of the period's timeset. Any other shape (a scalar, a map indexed by *period*) is an error.


- `timeline`: continuous timeline with a user-defined duration for each timestep. Timelines are used by time series data.

  - *timestep_duration*: a map with *timestep_name* as an index and *duration* as a value.
  - *timeline_duration_in_years* Total duration of the timeline in years. Used to relate operational part of the model with the annualized part of the model.

### Time structure assumptions

The tool includes some assumptions about the time structure, in case something parts are missing. These will work if there is only one option for the model to choose. The assuptions are the following:

 - If `timeset`: `timeline` is not set and only one `timeline` is defined, it is used
 - If no `timeset` is exist and only one `timeline` is defined, create a full timeline timeset
 - If `period_timeset` is defined, but no `realized_periods` or `invest_periods` exists, all periods are realized
 - If `period_timeset` does not exist and only one `timeset` exists, create `timesets` for all `realized_periods` and `invest_periods`
 - If `model`: `solves` does not exist, and only one `solve` exists, that it is used
 - If a `solve` in `model`: `solves` does not exist, create a `solve` where all `periods_available` are realized
 
 For example in the case above, it would have been possible to not fill the `timeset`: `timeline` as there is only one `timeline` to choose. Leaving out the `period_timeset` would result in an error as there are multiple `timesets` to choose.



## Units of measure
FlexTool does not check units. It combines the numbers you enter as they are, so any set of units works as
long as it is consistent. The parameter descriptions, the defaults and the result labels use a default
convention because it reads well for most energy models:

- power and flows: MW; energy and storage: MWh
- money: CUR, the currency of your model (results show costs in millions, M CUR)
- emissions: tonnes of CO2
- time: hours for time steps, years for lifetimes and years represented — these are fixed, not a convention.

You can use other units for some nodes, for example kg/h for hydrogen flows and kg for its storage. Then
every value connected to those nodes has to use them: prices in CUR/kg, efficiencies and conversion factors
between the units (MWh of electricity per kg), capacities in kg/h, and the results of those nodes are in kg/h
and kg even where the label says MW or MWh.

Two places need care:

- Values entered per kW or kWh — `invest_cost`, `fixed_cost` and `salvage_value` of units, connections and
  storage nodes, and the group `penalty_capacity_margin` — are multiplied by 1000 by FlexTool. With other
  units this means "currency per thousandth of the flow unit" (CUR per g/h when flows are in kg/h).
- Penalty defaults (for example `penalty_up` 10 000 CUR/MWh) are sized for MWh. Set them yourself when a node
  uses other units.
