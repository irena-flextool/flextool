# v56 method off-member table — user-editable

Five columns:

- **entity_class** — entity class the parameter lives on.
- **parameter** — the method-style parameter (those backed by a `parameter_value_list`).
- **current default** — the parameter's `default_value` in `flextool/schemas/spinedb_schema.json`. `null` means absent / unset.
- **current off name** — what the parameter's value-list currently offers as a no-op member, or `not assigned` when no such member exists.
- **suggestion** — the recommended off-name. When this matches the current column, no change is needed; when it differs, the suggestion is what should replace (or be added). `no off-member needed` flags structural parameters whose "off" means deleting the entity / row rather than toggling a method.

Edit freely. Once you've made your calls in the **suggestion** column, dispatch the implementation as Batch D.

| entity_class | parameter | current default | current off name | suggestion |
|---|---|---|---|---|
| `commodity` | `price_method` | `price` | not assigned | no off-member needed |
| `connection` | `invest_method` | `not_allowed` | `not_allowed` | `not_allowed` |
| `connection` | `is_DC` | `no` | not assigned | `no` |
| `connection` | `lifetime_method` | `reinvest_automatic` | `no_investment` | `no_investment` |
| `connection` | `startup_method` | `no_startup` | `no_startup` | `no_startup` |
| `connection` | `transfer_method` | `regular` | not assigned | no off-member needed |
| `connection__profile` | `profile_method` | `none` | not assigned | `none` |
| `constraint` | `sense` | `equal` | not assigned | no off-member needed |
| `group` | `co2_method` | `none` | `no_method` | `none` |
| `group` | `decomposition_method` | `none` | `none` | `none` |
| `group` | `flow_aggregator` | `no` | `no` | `no` |
| `group` | `has_capacity_margin` | `no` | not assigned | `no` |
| `group` | `has_inertia` | `no` | not assigned | `no` |
| `group` | `has_non_synchronous` | `no` | not assigned | `no` |
| `group` | `include_stochastics` | `no` | `no` | `no` |
| `group` | `invest_method` | `not_allowed` | `not_allowed` | `not_allowed` |
| `group` | `output_flowGroup_indicators` | `no` | `no` | `no` |
| `group` | `output_nodeGroup_dispatch` | `no` | `no` | `no` |
| `group` | `output_nodeGroup_indicators` | `no` | `no` | `no` |
| `group` | `share_loss_of_load` | `no` | `no` | `no` |
| `group` | `transfer_method` | `use_connection_transfer_methods` | `use_connection_transfer_methods` | `use_connection_transfer_methods` |
| `model` | `output_horizon` | `no` | `no` | `no` |
| `node` | `inflow_method` | `use_original` | `no_inflow` | `no_inflow` |
| `node` | `invest_method` | `not_allowed` | `not_allowed` | `not_allowed` |
| `node` | `lifetime_method` | `reinvest_automatic` | `no_investment` | `no_investment` |
| `node` | `node_type` | `balance` | not assigned | no off-member needed |
| `node` | `storage_binding_method` | `bind_forward_only` | not assigned | no off-member needed |
| `node` | `storage_nested_fix_method` | `fix_nothing` | `fix_nothing` (also `no`, redundant) | `fix_nothing` (drop redundant `no`) |
| `node` | `storage_solve_horizon_method` | `free` | `free` | `free` |
| `node` | `storage_start_end_method` | `fix_start` | `fix_nothing` | `fix_nothing` |
| `node__profile` | `profile_method` | `none` | not assigned | `none` |
| `reserve__upDown__group` | `reserve_method` | `no_reserve` | `no_reserve` | `no_reserve` |
| `solve` | `solve_mode` | `single_solve` | not assigned | no off-member needed |
| `solve` | `solver` | `highs` | not assigned | no off-member needed |
| `unit` | `conversion_method` | `constant_efficiency` | `none` | `none` |
| `unit` | `invest_method` | `not_allowed` | `not_allowed` | `not_allowed` |
| `unit` | `lifetime_method` | `reinvest_automatic` | `no_investment` | `no_investment` |
| `unit` | `minimum_time_method` | `none` | `none` | `none` |
| `unit` | `startup_method` | `no_startup` | `no_startup` | `no_startup` |
| `unit__inputNode` | `is_non_synchronous` | `no` | not assigned | `no` |
| `unit__inputNode` | `ramp_method` | `none` | not assigned | `none` |
| `unit__outputNode` | `is_non_synchronous` | `no` | not assigned | `no` |
| `unit__outputNode` | `ramp_method` | `none` | not assigned | `none` |
| `unit__node__profile` | `profile_method` | `upper_limit` | not assigned | `none` |

## Counts

- Total rows: **44**
- Already aligned (current == suggestion, no edit needed): **30**
- Add a new off-member (current is `not assigned`, suggestion is concrete): **8** rows
  - profile_method: 3 rows (`connection__profile`, `node__profile`, `unit__node__profile`)
  - ramp_method: 2 rows (`unit__inputNode`, `unit__outputNode`)
  - is_non_synchronous: 2 rows (`unit__inputNode`, `unit__outputNode`)
  - is_DC: 1 row
  - has_capacity_margin / has_inertia / has_non_synchronous on `group`: 3 rows
  - (some of these share a value list — see "value-list-level additions" below)
- Structural (no off-member needed): **6** rows
- Hygiene flag (drop redundant): **1** row (`node.storage_nested_fix_method.no`)

## Value-list-level additions (what the implementation actually touches)

Each member is added to its value list once; multiple parameters share a list:

- `profile_methods` ← add `none` (used by 3 parameters)
- `ramp_methods` ← add `none` (used by 2 parameters)
- `is_DC` ← add `no` (used by 1 parameter)
- `has_capacity_margin` ← add `no`
- `has_inertia` ← add `no`
- `has_non_synchronous` ← add `no`
- `is_non_synchronous` ← add `no` (used by 2 parameters)

So 7 value-list mutations in total.

## Default-value oddities worth flagging

A few rows where the **current default does not match the off-member** — these are where users get the feature *enabled* by default and have to set the parameter explicitly to turn it off:

- `connection.lifetime_method` defaults to `reinvest_automatic` (off is `no_investment`).
- `connection.startup_method` defaults to `no_startup` — already off-by-default. ✓
- `connection.transfer_method` defaults to `regular` (no off-member exists; structural).
- `constraint.sense` defaults to `equal` (no off-member; structural).
- `group.invest_method` defaults to `not_allowed` — off-by-default. ✓
- `node.inflow_method` defaults to `use_original` (off is `no_inflow`).
- `node.invest_method` defaults to `not_allowed` — off-by-default. ✓
- `node.lifetime_method` defaults to `reinvest_automatic` (off is `no_investment`).
- `node.node_type` defaults to `balance` (structural).
- `node.storage_binding_method` defaults to `bind_forward_only` (the lightest non-off choice; structural).
- `node.storage_start_end_method` defaults to `fix_start` (off is `fix_nothing`).
- `unit.conversion_method` defaults to `constant_efficiency` (off is `none`).
- `unit.invest_method` defaults to `not_allowed` — off-by-default. ✓
- `unit.lifetime_method` defaults to `reinvest_automatic` (off is `no_investment`).
- `unit.minimum_time_method` defaults to `null` (off is `none` — `null` is the de-facto off).

If you'd like any of these defaults flipped (e.g. `node.inflow_method` from `use_original` to `no_inflow` so users opt-in to inflow), that's a separate concern — flag and I'll handle it in the implementation pass.

## Optional hygiene follow-ups (not part of "add off-member" batch)

These are listed for completeness — the user can decide whether to fold them in:

- **Drop redundant `storage_nested_fix_method.no`** (keep `fix_nothing`) — naming inconsistency.
- **Collapse single-`yes` lists to `yes_no`** — `is_DC`, `has_capacity_margin`, `has_inertia`, `has_non_synchronous`, `is_non_synchronous` are all boolean-as-enum. Migrating them to the shared `yes_no` list (instead of each carrying its own) reduces schema noise. This is a parameter-retype, not just a list-extend.
- **Rename `lifetime_methods.no_investment` → `lifetime_methods.none`** — for consistency with `conversion_methods`, `decomposition_methods`, `minimum_time_methods` which all use `none`. Cosmetic; cascades to any user DBs that set `no_investment` explicitly.
- **Rename `invest_methods.not_allowed` → ?** — semantic. `not_allowed` is clear ("investment is forbidden for this entity") but distinct from the `none` family. Keep or rename — your call.
