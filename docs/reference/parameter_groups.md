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
