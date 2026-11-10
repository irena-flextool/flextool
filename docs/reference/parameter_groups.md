# Parameter groups and conventions

## Parameter groups (metadata)

Spine's `parameter_definition` table carries an optional `parameter_group_name` field — a lightweight category label that the Spine DB editor uses to group related parameters visually. FlexTool historically left this field blank. Schema v43 introduced the mechanism with a narrow `Outputs` foothold, and schema v44 populates it fully: every parameter_definition row in the master template is now tagged with one of 15 groups.

The scheme has no semantic meaning to the solver — it is purely metadata consumed by the Spine DB editor and anything else that walks `parameter_group_name`. Names are lowercase snake_case (FlexTool convention). Priorities are sparse by design so future groups can slot in without renumbering.

The groups are organised in three informal tiers — **asset physics** (what the thing is and how it runs), **decision overlays** (investment, retirement, storage state, advanced dynamics, reserves, emissions, network, flow caps, user-defined constraints), and **model plumbing** (model-wide settings, solve configuration, timeline definitions, output toggles).

| Group | Priority | Colour | Purpose |
|---|---|---|---|
| `basics` | 10 | `b3cde3` | What the thing is and how it runs at steady state — default home for any parameter not clearly in a decision overlay. |
| `investment` | 20 | `fdbf6f` | Capacity expansion: invest costs, methods, cumulative bounds, lifetime, discount rate, annualised fixed costs, capacity margin. |
| `retirement` | 25 | `ffb870` | Retirement/divestment bounds and salvage value. |
| `storage` | 30 | `cab2d6` | Storage state variable, cross-solve and horizon-edge behaviour, reference price/value. |
| `tech_advanced` | 35 | `b2df8a` | Unit commitment (startup, min times), ramp rates, time delays — advanced operational dynamics. |
| `reserve` | 40 | `fb9a99` | Spinning / standing / N-1 reserve provision parameters. |
| `emission` | 45 | `ccebc5` | CO2 accounting: fuel content, group caps, prices. |
| `network` | 50 | `80b1d3` | DC power flow, inertia, non-synchronous share — network-aware modelling. |
| `flow_limit` | 55 | `fccde5` | Group-level aggregated flow caps / floors and loss-of-load share control. |
| `constraint` | 70 | `bc80bd` | User-authored linear constraints plus the coefficient parameters on other entities that couple into them. |
| `model` | 80 | `d9d9d9` | Whole-model scope, horizons, numeric framing (version, periods, inflation). |
| `solve_basics` | 85 | `bebada` | Per-solve knobs a typical user touches (solver, mode, periods, years). |
| `solve_advanced` | 87 | `9f94c6` | Per-solve internals: solver options, rolling/nested/stochastic structures, cross-solve storage handoff. |
| `timeline` | 90 | `ffed6f` | Time-structure definitions consumed by solves (timestep duration, timeset metadata). |
| `output` | 95 | `a6cee3` | Toggles that shape what the run writes to outputs. |

### Convention for new parameters

When you add a new parameter definition, also assign it a `parameter_group_name`. Use the table above to pick the group — err on the side of matching *why the user sets the parameter*, not where it is mathematically consumed (e.g. `constraint_state_coeff` on `node` belongs to `constraint`, because the user writes it to participate in a user-defined constraint, not to configure storage).

Both the group definition and the tag are set inside the same db_migration step:

```python
db.add_update_item("parameter_group", name="my_group",
                   color="a6cee3", priority=60)
db.add_update_item("parameter_definition",
                   entity_class_name="unit",
                   name="my_param",
                   parameter_group_name="my_group")
```

The colour is a 6-digit hex string with no leading `#`. Priority controls display order in the Spine DB editor — pick a value that fits the tiered scheme above (asset physics 10–15, decision overlays 20–55, constraints 70, plumbing 80–95).

After editing the migration, run `python -m flextool.update_flextool.sync_master_json_template` to regenerate `schemas/spinedb_schema.json` — the group assignment is exported as the 6th slot of the parameter-definition tuple. The coverage pytest (`tests/test_parameter_groups_coverage.py`) fails loudly if any parameter is left ungrouped.

## Behaviour-change notices

When a FlexTool release changes what an existing model does, the database upgrade marks the upgraded database with
a metadata item `flextool_behaviour_change:<change>` (visible in the database editor's Metadata tab). While the
marker is present, every run prints a notice for each affected entity, once per run: which FlexTool version changed
the behaviour (and when it was published), what changed, the entities whose results change, and how to keep the old
behaviour or silence the notice. Deleting the marker stops the notice. New databases carry no markers. The markers
travel with the database: copies, the Spine Toolbox editor and the Excel export / import keep them.

The markers:

- `storage_default_fix_nothing` — the default `storage_start_end_method` became *fix_nothing*. The notice names the
  storage nodes without their own `storage_start_end_method` in models where some node has a `storage_state_start`
  (before: fixed at the start; now: free). Setting `storage_start_end_method` on those nodes silences it.
- `storage_fix_method_rewritten` — the upgrade rewrote `storage_start_end_method` values that had no value to fix
  at, and wrote *fix_start* on the nodes that used the old default with their own `storage_state_start` (results
  unchanged); the notice lists node, alternative, old method and what was written. Its value also lists the stored
  methods the upgrade left because scenarios used them differently.
- `user_constraint_prebuilt_nets_retirements` — `constraint_cumulative_pre_built_capacity_coeff` now subtracts the
  retirements of earlier periods, also those decided in the same solve (before: only those of earlier solves of a
  chain). The notice names the user constraints whose pre-built entity can retire; results change only if the model
  retires capacity. Deleting the marker silences it.

## Input checks

Before it reads the parameters, FlexTool checks the data of the scenario for values it cannot use sensibly. An
error stops the run; one message lists every offending value of the period-key, profile-link and value checks
(class, entity, parameter, value, alternative and, for a map, the first key that is wrong) with what to do, so one
rerun fixes them all. A warning is printed once per scenario read in the run log; the run goes on.

**Errors**

- An *upper_limit* or *fixed* `unit__node__profile`, `node__profile` or `connection__profile` link whose `profile`
  has no value in the scenario (it would hold the flow or the storage state at 0). The message names the link and
  the profile, and the alternatives that hold the profile's value when the scenario does not include them.
  A *lower_limit* link without a value is allowed (a floor of 0).
- `efficiency` of a unit or a connection at or below 0.
- Unit `min_load` outside [0, 1]; `availability` below 0 (unit, connection, node).
- `capacity_max_coeff` below 0.
- A minimum above its maximum, when the `invest_method` uses that pair: `invest_min_period` / `invest_max_period`,
  `retire_min_period` / `retire_max_period`, `cumulative_min_capacity` / `cumulative_max_capacity` (compared per
  period; a constant applies to every period), `invest_min_total` / `invest_max_total`, `retire_min_total` /
  `retire_max_total` (unit, connection, node, group). No capacity could meet both.
- `lifetime` at or below 0 on an entity that can invest or retire (the investment would be charged for no years).
- A map key of a period parameter (`existing`, `invest_cost`, `startup_cost`, `co2_max_period`, `lifetime`, and
  every other parameter that takes a period map) that is not a period: the readers would drop the value without a
  word (*y2020* for *p2020* gave no capacity). Periods are the keys of every `period_timeset`, the solve period lists,
  `years_represented`, the first level of `stochastic_branches`, the stochastic branch periods (*p2035_high*: the
  period, an underscore and the branch name) and `model.periods_available`, over the **whole**
  database, so a key that is a period of another scenario is fine. The message suggests the closest period
  (*p2030x* → *p2030*). When the data covers more years than any solve names (a 2020–2050 trajectory for a 2030
  model), add the extra periods to `model.periods_available`. A map without an index name of a parameter that can
  also vary over time (`availability`, `self_discharge_loss`, ...) whose keys are neither periods nor time steps is
  read as a period map, unless its keys look like time steps (*t0099*, *T0001*, a date-time stamp: then the
  time-series check reports them). Maps keyed by something else (`constraint_flow_coeff` by constraint, `delay` by
  hours) are not period maps and are not checked.
- A time-step key, a `time` level or a time series on a period parameter that cannot vary over time
  (`existing`, `startup_cost`, `invest_cost`, `lifetime`, ...; use a constant or a period map), and numeric map
  keys (2020) on any period parameter (periods are names such as *p2020*). A name that is both a period and a time
  step is read as a period.
- A timeline used by a solve (through a timeset named in a `period_timeset`, or the timeset a solve without
  `period_timeset` takes by default) without `timestep_duration`, or with a step of 0 hours or less (or not a
  number). Only the timelines a solve uses need durations: a spare timeline in the database is
  ignored.

**Warnings**

- A `node__profile` or `connection__profile` link without a `profile_method` (the default *none* does not apply the
  profile).
- `availability` above 1.5, and a profile above 1.5 on an *upper_limit* / *fixed* unit link: values above 1.5 look
  like percentages (95 for 0.95). A capacity factor slightly above 1 (rounding, measured data) is not reported. The
  profile check runs right after the data is read.
- Connection `efficiency` above 1 (creates energy).
- `virtual_unitsize` at or below 0 (ignored).
- A map of a period parameter that can also vary over time (`availability`, `other_operational_cost`, ...) without
  an index name whose keys are neither periods nor time steps but look like time steps, on a parameter the
  time-series check does not cover (the value is not used).
- A unit flow with `conversion_flow_coeff` 0 that no user constraint, flowGroup maximum or *upper_limit* / *fixed*
  profile ties: only `max_flow_for_unconstrained_variables` bounds it. Such an input is a free dump load on its node,
  such an output a free source. Tie it with a user constraint (`constraint_flow_coeff`), or give it a non-zero
  `conversion_flow_coeff`. A unit with an input whose outputs all have
  coefficient 0 gets one warning (it uses no input). `capacity_max_coeff` / `capacity_min_coeff` on such an output
  has no effect.

Negative `existing` is not reported: it is the supported "anti-energy" unit. Other checks (time series keys,
timesets, input shares, minimum-load curves, ladders, group limits) are described with their parameters.
