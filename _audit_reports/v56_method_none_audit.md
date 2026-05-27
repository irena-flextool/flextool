# v56 method "no-op" audit — proposal

## Goal

Every method-style parameter (one backed by a `parameter_value_list` of
enum-like choices) should ship an explicit *no-op* / off member.  Today
the user often has to delete the parameter_value row to turn a method
off, which is destructive (loses the row's history / structure in
the GUI) and asymmetric with parameters that *do* have an off option
(e.g. `co2_methods.no_method`, `transfer_methods_group.use_connection_transfer_methods`).

This document is **audit only** — no schema, engine, fixture or
migration edits.  The implementation (a separate v56 helper) is
sketched at the bottom.

Scope: every parameter in `flextool/schemas/spinedb_schema.json` whose
`parameter_value_list_name` column is set.  That is the
authoritative list of enum-backed parameters (44 parameters across
35 classes).  Non-enum parameters are not in scope.

Naming-convention reference (derived from existing schema usage):

| list                       | existing off-style member                              |
| -------------------------- | ------------------------------------------------------ |
| `co2_methods`              | `no_method`                                            |
| `conversion_methods`       | `none`                                                 |
| `decomposition_methods`    | `none`                                                 |
| `inflow_methods`           | `no_inflow`                                            |
| `invest_methods`           | `not_allowed`                                          |
| `lifetime_methods`         | `no_investment`                                        |
| `load_share_type`          | `no`                                                   |
| `minimum_time_methods`     | `none`                                                 |
| `reserve_methods`          | `no_reserve`                                           |
| `startup_methods`          | `no_startup`                                           |
| `storage_nested_fix_method`| `fix_nothing` (and the redundant `no`)                 |
| `storage_solve_horizon_methods` | `free`                                            |
| `storage_start_end_methods`| `fix_nothing`                                          |
| `transfer_methods_group`   | `use_connection_transfer_methods` (override-off)       |
| `yes_no`                   | `no`                                                   |

So the schema is not consistent — `no_method`, `none`, `no_inflow`,
`no_reserve`, `no_startup`, `fix_nothing`, `not_allowed`,
`no_investment`, `free` are all used.  The recommendation below
follows whatever style each list already uses; for lists with no
established style, the proposal is `none`.

---

## Per-parameter audit

### `commodity.price_method` — value list `price_method`

- **Current members**: `price`, `price_ladder_annual`, `price_ladder_cumulative`.
- **No-op already?**: no — but the *default* `price` is the de-facto
  off for the ladder feature (plain price-times-flow consumer).
- **Proposed no-op name**: n/a (see notes).
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1748`
  `commodity_with_ladder` filters commodities by `price_method ∈ {price_ladder_annual, price_ladder_cumulative}`; everything else falls back to scalar `price` x flow.
- **Engine impact of a no-op**: none (default `price` already behaves as "plain price").
- **Confidence**: surface
- **Notes**: `price` is **not** off — the commodity still gets priced.
  Removing pricing entirely means deleting the `price` parameter, not
  toggling `price_method`.  This parameter only chooses *how* the price
  enters the LP, so there is no meaningful "no pricing at all" member.
  Recommend: leave alone.  Document in description that `price` is the
  baseline / scalar consumer.

---

### `connection.invest_method` — value list `invest_methods`

- **Current members**: `not_allowed`, `invest_no_limit`, `invest_period`, `invest_total`, `invest_period_total`, `retire_no_limit`, `retire_period`, `retire_total`, `retire_period_total`, `invest_retire_no_limit`, `invest_retire_period`, `invest_retire_total`, `invest_retire_period_total`, `cumulative_limits`.
- **No-op already?**: yes (`not_allowed`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_derived_npv.py:603`
  (`_entity_method_lf("invest_method")`); default is `not_allowed`, entities without an explicit row are excluded from the investment domain.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).
- **Notes**: This list is the gold standard — `not_allowed` is a real off; default is `not_allowed`.

---

### `connection.is_DC` — value list `is_DC`

- **Current members**: `yes`.
- **No-op already?**: no — toggling off is "delete the row" (default `null` ≡ false).
- **Proposed no-op name**: `no` (i.e. promote `is_DC` from a single-member yes list to a `yes/no` list, or migrate to `yes_no`).
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1729` `connection_is_DC`; filter `value=='yes'`.
- **Engine impact of a no-op**: none — the filter is `is_in("yes")`, so adding `no` simply makes the off-state expressible without deleting the row.  An extra-safe pass would change the consumer to `is_in("yes")` (already the case).
- **Confidence**: high
- **Notes**: same applies to the parallel one-member lists `has_DC`, `has_capacity_margin`, `has_inertia`, `has_invest`, `has_non_synchronous`, `is_active`, `is_non_synchronous` — they are all boolean-as-enum.  Cheapest fix is to **collapse them to `yes_no`** in a follow-up; the immediate-batch fix is to add `no` to each list (keeps the named-list per parameter, no rename).

---

### `connection.lifetime_method` — value list `lifetime_methods`

- **Current members**: `reinvest_automatic`, `reinvest_choice`, `no_investment`.
- **No-op already?**: yes (`no_investment`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_derived_npv.py:297`
  `lifetime_method_lf`; default fill `reinvest_automatic`; `no_investment` skips reinvestment logic.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).
- **Notes**: Naming is awkward (`no_investment` reads as if it forbids investment — see also `invest_method`).  Rename to `none` is a separate concern.

---

### `connection.startup_method` — value list `startup_methods`

- **Current members**: `no_startup`, `linear`, `binary`.
- **No-op already?**: yes (`no_startup`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1062`
  `process_online`; filter `value ∈ {linear, integer}` so `no_startup` falls through silently.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).
- **Notes**: Description in v56 mentions `integer` instead of `binary` — separate concern (see Batch E).

---

### `connection.transfer_method` — value list `transfer_methods`

- **Current members**: `no_losses_no_variable_cost`, `regular`, `exact`, `variable_cost_only`, `unidirectional`.
- **No-op already?**: no — there is no "transfer disabled" member.  The
  default is `regular`, which **does** build flow variables.
- **Proposed no-op name**: n/a (see notes).
- **Engine consumer**: `flextool/engine_polars/_direct_params.py:123` `connection_transfer_method`; consumed in `_derived_params.py:1637` (efficiency family) and `_block_layout.py:737` (transfer arc partitions).
- **Engine impact of a no-op**: requires new engine branch — every consumer would need a "skip-this-connection" code path.
- **Confidence**: surface
- **Notes**: A connection without a transfer method has no LP meaning.
  "Off" means delete the connection, not toggle a method.  Recommend
  no change.  (`no_losses_no_variable_cost` is *not* off — flow still
  happens, just frictionlessly.)

---

### `connection__profile.profile_method` / `node__profile.profile_method` / `unit__node__profile.profile_method` — value list `profile_methods`

- **Current members**: `upper_limit`, `lower_limit`, `fixed`.
- **No-op already?**: no — but the parameter's *default is null*, so missing == off.
- **Proposed no-op name**: `none`.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:907`
  `_profile_method_arc` / `:1030` `node_profile_filter`; entries without a `profile_method` row are not collected into any of the three subsets.
- **Engine impact of a no-op**: none — current consumers already match `value == <specific>`, so a `none` row falls through.
- **Confidence**: high
- **Notes**: Users who add a profile relationship but want to disable it
  without removing it benefit directly.  Add `none` to `profile_methods`.

---

### `constraint.sense` — value list `senses`

- **Current members**: `greater_than`, `less_than`, `equal`.
- **No-op already?**: no.
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/` (constraint sense — used in `_emit_*` constraint LHS/RHS as the LP row sense character).
- **Engine impact of a no-op**: n/a — a constraint without a sense is not a constraint.
- **Confidence**: surface
- **Notes**: Structural; "no sense" means "no constraint".  Leave alone.

---

### `group.co2_method` — value list `co2_methods`

- **Current members**: `no_method`, `price`, `period`, `total`, `price_period`, `price_total`, `period_total`, `price_period_total`.
- **No-op already?**: yes (`no_method`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_emit_co2_accumulators.py:49`
  `group_co2_max_total`; filter `co2_method ∈ {total, price_total, period_total, price_period_total}`.  `no_method` is silently dropped.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `group.decomposition_method` — value list `decomposition_methods`

- **Current members**: `none`, `lagrangian_region`.
- **No-op already?**: yes (`none`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_blocks.py:299` /
  `_lagrangian.py:175` / `_block_layout.py:633` — only `lagrangian_region` groups are collected for decomposition; `none` falls through.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `group.flow_aggregator` / `group.include_stochastics` / `group.output_flowGroup_indicators` / `group.output_nodeGroup_dispatch` / `group.output_nodeGroup_indicators` — value list `yes_no`

- **Current members**: `yes`, `no`.
- **No-op already?**: yes (`no`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: output emitters / aggregators; each filters on `value == 'yes'`.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).
- **Notes**: All `yes_no`-typed parameters are already symmetric.  This row is just for completeness.

---

### `group.has_capacity_margin`, `group.has_inertia`, `group.has_non_synchronous` — single-member value lists

- **Current members**: only `yes`.
- **No-op already?**: no — toggling off means deleting the row.
- **Proposed no-op name**: `no` (per-list) OR migrate the parameter to `yes_no`.
- **Engine consumer**: each gates a feature subsystem (capacity margin block, inertia constraint, non-synchronous flow tracking); consumer filters `value == 'yes'`, so a `no` entry is silently dropped.
- **Engine impact of a no-op**: none.
- **Confidence**: high
- **Notes**: Same shape as `connection.is_DC` above.  Cleanest is to
  retype these parameters to the existing `yes_no` value list (5 parameters total: `has_DC`, `has_capacity_margin`, `has_inertia`, `has_invest`, `has_non_synchronous` — though `has_invest` and `has_DC` aren't currently referenced as parameter_definitions, so they may be dead lists).

---

### `group.invest_method` — value list `invest_methods`

- See `connection.invest_method` — same list, same default `not_allowed`.  Already has no-op.

---

### `group.share_loss_of_load` — value list `load_share_type`

- **Current members**: `equal`, `inflow_weighted`, `no`.
- **No-op already?**: yes (`no`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: not currently consumed by `engine_polars/` projection layer (only appears in `autoscale/_quantity_types.py:413` as a dimensionless marker).  Likely consumed downstream by GAMS-era `.mod` or a post-output emitter.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `group.transfer_method` — value list `transfer_methods_group`

- **Current members**: `use_connection_transfer_methods`, `no_losses_no_variable_cost`, `regular`, `exact`, `variable_cost_only`, `dc_power_flow_with_angles`, `unidirectional`.
- **No-op already?**: yes (`use_connection_transfer_methods` — semantically "no group-level override").
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_native_run_model.py` and DC-power-flow gating in `_emit_arc_unions.py`; group-level value overrides per-connection `transfer_method`.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).
- **Notes**: This is the **gold-standard for our pattern** — the no-op is named *what it does* (`use_connection_transfer_methods`) rather than `none`/`no_method`.  For lists that aren't override-style, `none` / `no_<thing>` reads better.

---

### `model.output_horizon` — value list `yes_no`

- **Current members**: `yes`, `no`.
- **No-op already?**: yes (`no`).
- **Confidence**: n/a (already done).

---

### `node.inflow_method` — value list `inflow_methods`

- **Current members**: `no_inflow`, `use_original`, `scale_in_proportion`, `scale_to_annual_flow`, `scale_to_annual_and_peak_flow`.
- **No-op already?**: yes (`no_inflow`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_inflow_scaling.py:336` `inflow_method_lf`; default fill `use_original`; `no_inflow` skips inflow injection.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `node.invest_method` — value list `invest_methods`

- See `connection.invest_method` — same list, same default `not_allowed`.  Already has no-op.

---

### `node.lifetime_method` — value list `lifetime_methods`

- See `connection.lifetime_method`.  Already has no-op (`no_investment`).

---

### `node.node_type` — value list `node_type`

- **Current members**: `balance_within_period`, `commodity`, `balance`, `storage`.
- **No-op already?**: no.
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_direct_params.py:95` `node_node_type` (default `balance`); `_projection_params.py:1131` `nodeState` (filter `storage`); `_emit_mid_sets.py:86` `p_node_type` materialization.
- **Engine impact of a no-op**: requires new engine branch.  A node without a node_type makes no LP sense (every node has to be one of the four).
- **Confidence**: surface
- **Notes**: Structural.  "Off" means delete the node.

---

### `node.storage_binding_method` — value list `storage_binding_methods`

- **Current members**: `bind_within_period`, `bind_within_solve`, `bind_forward_only`, `bind_intraperiod_blocks`, `bind_within_timeblock`, `bind_within_solve_blended_weights`, `bind_within_period_blended_weights`, `bind_forward_only_blended_weights`.
- **No-op already?**: partial — `bind_forward_only` (the default) is "no end-to-start closure", which is the closest thing to off; but storage *always* needs binding semantics so there is no full off.
- **Proposed no-op name**: n/a (see notes).
- **Engine consumer**: `flextool/engine_polars/_native_run_model.py:124` (per-solve provider rewrite that silently degrades `*_blended_weights` to non-RP when no representative_period_weights); `_derived_block.py:580` (intraperiod block handling); `_emit_solve_time.py:80` `emit_node_storage_binding_method`.
- **Engine impact of a no-op**: requires new engine branch.  A storage node without a binding method would have no state-continuity equation; the v_state variable would be uninitialised across boundaries.
- **Confidence**: surface
- **Notes**: Structural — every storage node must have a binding method.
  `bind_forward_only` is the "least binding" choice and is already the default.  Recommend leave alone, but document in the description that `bind_forward_only` is the minimum-overhead choice.

---

### `node.storage_nested_fix_method` — value list `storage_nested_fix_method`

- **Current members**: `fix_nothing`, `fix_quantity`, `fix_price`, `no`, `fix_usage`.
- **No-op already?**: yes — *twice*: `fix_nothing` and `no`.
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1186` `storage_nested_fix_quantity`; filter `value == 'fix_quantity'`.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done) — but **drop the redundant `no`**, leave only `fix_nothing` to match the rest of the storage family.  (List as a separate hygiene concern, not part of "add no-op" batch.)

---

### `node.storage_solve_horizon_method` — value list `storage_solve_horizon_methods`

- **Current members**: `free`, `use_reference_value`, `use_reference_price`.
- **No-op already?**: yes (`free` — default).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_emit_arc_unions.py:282` collects nodes with `use_reference_value` / `use_reference_price`; `free` falls through.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `node.storage_start_end_method` — value list `storage_start_end_methods`

- **Current members**: `fix_start`, `fix_end`, `fix_start_end`, `fix_nothing`.
- **No-op already?**: yes (`fix_nothing`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1173` `storage_fix_start`; filter on specific value, `fix_nothing` drops.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `reserve__upDown__group.reserve_method` — value list `reserve_methods`

- **Current members**: `no_reserve`, `timeseries_only`, `dynamic_only`, `large_failure_only`, `timeseries_and_dynamic`, `timeseries_and_large_failure`, `dynamic_and_large_failure`, `all`.
- **No-op already?**: yes (`no_reserve`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1304` `reserve_upDown_group`; missing or `no_reserve` falls through.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `solve.solve_mode` — value list `solve_mode`

- **Current members**: `single_solve`, `rolling_window`.
- **No-op already?**: no.
- **Proposed no-op name**: n/a.
- **Engine consumer**: structural — the solve plan dispatcher.
- **Engine impact of a no-op**: n/a — a solve without a mode is not a solve.
- **Confidence**: surface
- **Notes**: Structural.

---

### `solve.solver` — value list `solvers`

- **Current members**: `highs`, `gurobi`, `cplex`, `xpress`, `copt`.
- **No-op already?**: no.
- **Proposed no-op name**: n/a.
- **Engine consumer**: solver-binary dispatch.
- **Engine impact of a no-op**: n/a.
- **Confidence**: surface
- **Notes**: Per task brief — solver is structural, not behavioural.

---

### `unit.conversion_method` — value list `conversion_methods`

- **Current members**: `constant_efficiency`, `min_load_efficiency`, `none`.
- **No-op already?**: yes (`none`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1113` `process_min_load_eff`; `_derived_params.py:1583` efficiency-family derivation.  `none` drops the unit from both.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `unit.invest_method` — value list `invest_methods`

- See `connection.invest_method`.  Already has no-op.

---

### `unit.lifetime_method` — value list `lifetime_methods`

- See `connection.lifetime_method`.  Already has no-op.

---

### `unit.minimum_time_method` — value list `minimum_time_methods`

- **Current members**: `min_uptime`, `min_downtime`, `both`, `none`.
- **No-op already?**: yes (`none`).
- **Proposed no-op name**: n/a.
- **Engine consumer**: `flextool/engine_polars/_derived_params.py:1666` `derive_unit_online_method`; `none` causes no startup promotion / no min-time constraints.
- **Engine impact of a no-op**: none.
- **Confidence**: n/a (already done).

---

### `unit.startup_method` — value list `startup_methods`

- See `connection.startup_method`.  Already has no-op (`no_startup`).

---

### `unit__inputNode.is_non_synchronous` / `unit__outputNode.is_non_synchronous` — value list `is_non_synchronous`

- **Current members**: `yes`.
- **No-op already?**: no.
- **Proposed no-op name**: `no`.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:1282` filter `value=='yes'`.
- **Engine impact of a no-op**: none.
- **Confidence**: high
- **Notes**: Same pattern as `is_DC` / `has_*`.  Migrate to `yes_no` or add `no` member.

---

### `unit__inputNode.ramp_method` / `unit__outputNode.ramp_method` — value list `ramp_methods`

- **Current members**: `ramp_limit`, `ramp_cost`, `both`.
- **No-op already?**: no — but the parameter's *default is null*, so missing == off.
- **Proposed no-op name**: `none`.
- **Engine consumer**: `flextool/engine_polars/_projection_params.py:571` `_ramp_pairs`; filter `value ∈ method_set`.  A `none` row would simply fall through.
- **Engine impact of a no-op**: none.
- **Confidence**: high
- **Notes**: Description warns that `ramp_cost` is "NOT FUNCTIONAL AS OF 19.3.2023" — relevant to Batch E, not this audit.  Adding `none` lets a user keep a `unit__node` row but disable ramping without deleting it.

---

## Summary stats

- Total parameters with a value list audited: **44** (one per
  parameter_definition with a non-null `parameter_value_list_name`).
- Already have a no-op (existing off member): **31**.
- Need a no-op (high confidence — add a member, no engine work):
  **8**.
  - `connection.is_DC` (and the 4 sibling single-`yes` lists on `group`)
  - `connection__profile.profile_method`
  - `node__profile.profile_method`
  - `unit__node__profile.profile_method`
  - `unit__inputNode.is_non_synchronous`
  - `unit__outputNode.is_non_synchronous`
  - `unit__inputNode.ramp_method`
  - `unit__outputNode.ramp_method`
- Need a no-op (medium confidence — engine branch required):
  **0**.  Every parameter that needs an engine branch turned out
  to be structurally non-togglable (surface tier).
- Surface / structural (out of scope — no meaningful off):
  **5**.
  - `commodity.price_method`
  - `connection.transfer_method`
  - `constraint.sense`
  - `node.node_type`
  - `node.storage_binding_method`
  - `solve.solve_mode`
  - `solve.solver`

(High-confidence and surface counts overlap with several `yes`-only
single-member lists collapsed under one row above.)

## Top 3-5 user-impactful "add a no-op" rows

1. **`unit__inputNode.ramp_method` / `unit__outputNode.ramp_method`**: users frequently want to keep a `unit__node` relationship but stop ramping it (e.g. when testing scenarios).  Adding `none` lets them flip the method without deleting the row, preserving `ramp_speed_*` settings for re-enabling later.
2. **`<*>__profile.profile_method`** (3 parameters: `connection__profile`, `node__profile`, `unit__node__profile`): same pattern — keep the profile relationship attached but disable the upper/lower/fixed binding.  High impact in scenario sweeps.
3. **`connection.is_DC`** (and `group.has_capacity_margin` / `has_inertia` / `has_non_synchronous`): boolean toggles currently single-`yes` lists.  Promoting them to `yes_no` (or adding `no`) makes off a real value rather than "delete the row".
4. **`unit__inputNode.is_non_synchronous` / `unit__outputNode.is_non_synchronous`**: identical pattern to `is_DC`; symmetry argument is strong.

## Proposed implementation plan (a later batch — not this audit)

The implementation should land in a single v56 helper (cherry-pickable
per parameter only if the helpers are tiny):

1. **Per affected value list**, append the new member via
   `add_value_list_manual` in a new `elif next_version == NN` block in
   `flextool/update_flextool/db_migration.py`.  Specifically:
   ```python
   add_value_list_manual(db, [
       ["profile_methods", "none"],
       ["ramp_methods", "none"],
       ["is_DC", "no"],
       ["has_capacity_margin", "no"],
       ["has_inertia", "no"],
       ["has_non_synchronous", "no"],
       ["is_non_synchronous", "no"],
   ])
   ```
2. **Mirror the additions** in `flextool/schemas/spinedb_schema.json`
   under `parameter_value_lists` (same set of `[list_name, member]`
   pairs).
3. **Default-value updates**: none required.  Every affected
   parameter currently defaults to `null` (missing == off), so the new
   `none` / `no` member is purely a user-facing toggle.  No engine
   consumer needs to change its default-fill logic.
4. **Engine branches**: none required.  Every consumer for the eight
   high-confidence rows already filters `value == <specific-active-method>`,
   so the new no-op rows fall through silently.
5. **Recommended packaging**: a **single helper** (rather than per-parameter)
   because the changes are mechanical and the value-list additions are
   idempotent under `add_value_list_manual`.  Per-parameter helpers are
   only worth the cherry-pick cost if we expect some additions to land
   while others are reverted.
6. **Follow-up hygiene tasks** (out of scope here, file as Batch E or a
   separate concern):
   - Drop the redundant `storage_nested_fix_method.no` member (keep
     `fix_nothing`).
   - Consider collapsing the five single-`yes` lists (`is_DC`,
     `has_capacity_margin`, `has_inertia`, `has_invest`,
     `has_non_synchronous`, `is_active`, `is_non_synchronous`) into
     `yes_no` — that would be a parameter-retype, not just a list-extend.
   - Consider renaming `lifetime_methods.no_investment` to
     `lifetime_methods.none` for consistency with `conversion_methods`,
     `decomposition_methods`, `minimum_time_methods`.

## Out-of-scope follow-ups

- **No engine, schema, fixture, or migration edits in this audit.**
- No new commit.
- The Batch E description-hygiene tasks (e.g. `binary` vs `integer` in
  `startup_methods` description, `ramp_cost` "NOT FUNCTIONAL" note,
  inconsistent off-member naming) are tracked separately.
