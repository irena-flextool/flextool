# Specification: Splitting `flextool/process_outputs` into Smaller Files

## Motivation

The current module has four files with very uneven sizes. `result_writer.py` is ~1 790 lines and
`process_results.py` is ~789 lines, making them hard for AI agents to work with (too much context
needed to change one small thing). The goal is to split them into ~20 files that are each small
and have a clear, single responsibility.

---

## Architecture question: should the two-step (process_results → output functions) be kept?

**Yes. The two-step is architecturally sound and should be kept, but the current boundary needs
correction.**

### Why the separation is worth keeping

`r.flow_dt` (the most expensive calculation) is consumed by **9 output functions**. Without
pre-computing it, it would be recomputed 9 times. Similarly `r.entity_all_capacity` serves 6
functions and `r.process_source/sink_flow_d` serves 3. For any shared intermediate used by 2+
output functions, pre-computing it once in `post_process_results` saves duplicated work and keeps
output functions small and focused.

The other unavoidable coupling is `write_summary_csv()`, which reads 14 attributes from `r` (cost
totals, slack quantities, CO2 emissions) and is called directly from `write_outputs()`, not through
`ALL_OUTPUTS`. Because it shares data needs with `cost_summaries()` and `slack_variables()`, the
cost and slack intermediate results must be pre-computed in `r`.

For both human developers and AI agents, the separation is better because it lets you reason about
"how is flow_dt computed" and "how is flow_dt formatted for output" as completely separate tasks
in completely separate files. Merging them would interleave computation and formatting logic,
increasing the context needed to make any change.

### Real problems with the current boundary (found by tracing every r.* read/write)

**1. `r.connection_d` and `r.connection_losses_d` should only be computed in `process_results.py`.**
The `connection()` output function currently *overwrites* them. Remove the overwrites from the
output function and use the pre-computed values from `r` directly.

**2. `nodeGroup_flows()` mutates column MultiIndexes on `r` attributes — this is intentional.**
It adds group-level labelling to existing column structures (useful for plotting). Leave as-is.

**3. Several `r` attributes are computed in `process_results` but never read by any output
function (dead weight). Replace with local variables or remove:**
- `r.hours_in_realized_period`
- `r.realized_period_share_of_year`
- `r.node_state_change_dt` / `r.node_state_change_d`
- `r.node_inflow_d`

**4. Three output functions (`node_additional_results`, `investment_duals`, `input_sets`) do not
read `r` at all** — they only use raw `v`, `par`, `s`. Harmless but worth noting.

### Structural improvement: move `write_summary_csv` into ALL_OUTPUTS

Currently `write_summary_csv` is hard-wired in `write_outputs()` via a direct call. It could
instead be a regular output function that returns a result and is listed in `ALL_OUTPUTS`. This
removes the special-cased call in `write_outputs()` and makes the data-flow dependency on
cost/slack pre-computation visible through the same mechanism as everything else.

---

### Current state

| File | Lines | Main concerns |
|---|---|---|
| `__init__.py` | 6 | re-exports |
| `read_flextool_outputs.py` | 393 | read_variables, read_parameters, read_sets |
| `process_results.py` | 789 | drop_levels, post_process_results (huge) |
| `result_writer.py` | ~1 790 | 25 output functions + write_outputs orchestrator |
| `to_spine_db.py` | 565 | write DataFrames → Spine DB |

---

## Replacing `SimpleNamespace` with typed alternatives

All four namespaces (`v`, `par`, `s`, `r`) currently use `SimpleNamespace`. This trades
discoverability and safety for convenience. Given Python 3.11+, better options are available.

### `v`, `par`, `s` → `@dataclass`

These are constructed once and then (largely) read-only. All fields are known at construction
time. Replace each `SimpleNamespace` with a `@dataclass`:

```python
# in read_variables.py
from dataclasses import dataclass
import pandas as pd

@dataclass
class Variables:
    obj: pd.DataFrame
    flow: pd.DataFrame
    ramp: pd.DataFrame
    # ... all fields explicitly listed
```

Use plain `@dataclass` (not `frozen=True`), because `drop_levels()` reassigns fields
(`v.flow = v.flow.droplevel('solve')`). Frozen would break that.

Benefits:
- IDE autocomplete and AttributeError on typos
- The field list is the documentation — no separate docstring needed
- Zero runtime overhead compared to SimpleNamespace
- `par` and `s` similarly become `@dataclass Parameters` and `@dataclass Sets`

### `r` → `@dataclass PostProcessedResults` with all Optional fields

`r` is built incrementally across multiple calc functions, so all fields start as `None` and are
filled in. Declare the full dataclass in `process_results.py`:

```python
@dataclass
class PostProcessedResults:
    # calc_capacity_flows.py
    entity_all_capacity: pd.DataFrame | None = None
    process_online_dt: pd.DataFrame | None = None
    flow_dt: pd.DataFrame | None = None
    flow_d: pd.DataFrame | None = None
    process_source_flow_d: pd.DataFrame | None = None
    process_sink_flow_d: pd.DataFrame | None = None
    ramp_dtt: pd.DataFrame | None = None
    # calc_connections.py
    connection_dt: pd.DataFrame | None = None
    connection_losses_dt: pd.DataFrame | None = None
    connection_d: pd.DataFrame | None = None
    connection_losses_d: pd.DataFrame | None = None
    connection_to_left_node__dt: pd.DataFrame | None = None
    connection_to_right_node__dt: pd.DataFrame | None = None
    connection_to_left_node__d: pd.DataFrame | None = None
    connection_to_right_node__d: pd.DataFrame | None = None
    # calc_storage_vre.py
    self_discharge_loss_dt: pd.DataFrame | None = None
    self_discharge_loss_d: pd.DataFrame | None = None
    potentialVREgen_dt: pd.DataFrame | None = None
    potentialVREgen_d: pd.DataFrame | None = None
    storage_usage_dt: pd.DataFrame | None = None
    # calc_slacks.py
    reserves_dt: pd.DataFrame | None = None
    reserves_d: pd.DataFrame | None = None
    upward_node_slack_dt: pd.DataFrame | None = None
    downward_node_slack_dt: pd.DataFrame | None = None
    upward_node_slack_d: pd.DataFrame | None = None
    downward_node_slack_d: pd.DataFrame | None = None
    upward_node_slack_d_not_annualized: pd.DataFrame | None = None
    downward_node_slack_d_not_annualized: pd.DataFrame | None = None
    costPenalty_node_state_upDown_dt: pd.DataFrame | None = None
    costPenalty_node_state_upDown_d: pd.DataFrame | None = None
    q_inertia_dt: pd.DataFrame | None = None
    q_inertia_d_not_annualized: pd.DataFrame | None = None
    q_non_synchronous_dt: pd.DataFrame | None = None
    q_non_synchronous_d_not_annualized: pd.DataFrame | None = None
    q_capacity_margin_d_not_annualized: pd.DataFrame | None = None
    costPenalty_capacity_margin_d: pd.Series | None = None
    q_reserves_dt: pd.DataFrame | None = None
    q_reserves_d_not_annualized: pd.DataFrame | None = None
    costPenalty_inertia_dt: pd.DataFrame | None = None
    costPenalty_non_synchronous_dt: pd.DataFrame | None = None
    costPenalty_reserve_upDown_dt: pd.DataFrame | None = None
    # calc_costs.py
    cost_commodity_dt: pd.DataFrame | None = None
    cost_commodity_d: pd.DataFrame | None = None
    sales_commodity_dt: pd.DataFrame | None = None
    sales_commodity_d: pd.DataFrame | None = None
    process_emissions_co2_dt: pd.DataFrame | None = None
    process_emissions_co2_d: pd.DataFrame | None = None
    emissions_co2_d: pd.Series | None = None
    group_co2_d: pd.DataFrame | None = None
    cost_co2_dt: pd.Series | None = None
    cost_process_other_operational_cost_dt: pd.DataFrame | None = None
    cost_startup_dt: pd.DataFrame | None = None
    process_startup_dt: pd.DataFrame | None = None
    cost_entity_invest_d: pd.DataFrame | None = None
    cost_entity_divest_d: pd.DataFrame | None = None
    cost_entity_fixed_pre_existing: pd.DataFrame | None = None
    cost_entity_fixed_invested: pd.DataFrame | None = None
    cost_entity_fixed_divested: pd.DataFrame | None = None
    costOper_dt: pd.Series | None = None
    costPenalty_dt: pd.Series | None = None
    costOper_d: pd.Series | None = None
    costPenalty_d: pd.Series | None = None
    costOper_and_penalty_d: pd.Series | None = None
    costInvestUnit_d: pd.Series | None = None
    costDivestUnit_d: pd.Series | None = None
    costInvestConnection_d: pd.Series | None = None
    costDivestConnection_d: pd.Series | None = None
    costInvestState_d: pd.Series | None = None
    costDivestState_d: pd.Series | None = None
    costInvest_d: pd.Series | None = None
    costDivest_d: pd.Series | None = None
    costFixedPreExisting_d: pd.Series | None = None
    costFixedInvested_d: pd.Series | None = None
    costFixedDivested_d: pd.Series | None = None
    # calc_group_flows.py
    group_output__unit_to_node_not_in_aggregate__dt: pd.DataFrame | None = None
    group_output__node_to_unit_not_in_aggregate__dt: pd.DataFrame | None = None
    group_output__group_aggregate_Unit_to_group__dt: pd.DataFrame | None = None
    group_output__group_aggregate_Group_to_unit__dt: pd.DataFrame | None = None
    group_output__from_connection_not_in_aggregate__dt: pd.DataFrame | None = None
    group_output__to_connection_not_in_aggregate__dt: pd.DataFrame | None = None
    group_output__from_connection_aggregate__dt: pd.DataFrame | None = None
    group_output__to_connection_aggregate__dt: pd.DataFrame | None = None
    group_output_Internal_connection_losses__dt: pd.DataFrame | None = None
    group_output_Internal_unit_losses__dt: pd.DataFrame | None = None
    group_node_inflow_dt: pd.DataFrame | None = None
    group_node_inflow_d: pd.DataFrame | None = None
    group_node_state_losses__dt: pd.DataFrame | None = None
    group_node_up_slack__dt: pd.DataFrame | None = None
    group_node_down_slack__dt: pd.DataFrame | None = None
```

---

## Proposed file structure (20 files)

```
flextool/process_outputs/
├── __init__.py                  (keep, update imports)
│
│  ── I/O layer ──
├── read_variables.py            (NEW – extracted from read_flextool_outputs.py)
├── read_parameters.py           (NEW – extracted from read_flextool_outputs.py)
├── read_sets.py                 (NEW – extracted from read_flextool_outputs.py)
├── to_spine_db.py               (keep, minor cleanup)
│
│  ── Post-processing calculations ──
├── drop_levels.py               (NEW – extracted from process_results.py)
├── calc_capacity_flows.py       (NEW – extracted from process_results.py)
├── calc_connections.py          (NEW – extracted from process_results.py)
├── calc_group_flows.py          (NEW – extracted from process_results.py)
├── calc_costs.py                (NEW – extracted from process_results.py)
├── calc_slacks.py               (NEW – extracted from process_results.py)
├── calc_storage_vre.py          (NEW – extracted from process_results.py)
├── process_results.py           (KEEP, greatly reduced – thin coordinator)
│
│  ── Output functions ──
├── out_capacity.py              (NEW – extracted from result_writer.py)
├── out_flows.py                 (NEW – extracted from result_writer.py)
├── out_group.py                 (NEW – extracted from result_writer.py)
├── out_node.py                  (NEW – extracted from result_writer.py)
├── out_costs.py                 (NEW – extracted from result_writer.py)
├── out_ancillary.py             (NEW – extracted from result_writer.py)
└── write_outputs.py             (KEEP, reduced – orchestrator + __main__)
```

---

## File-by-file specification

### 1. `read_variables.py` (~115 lines)
**Extracted from**: `read_flextool_outputs.py`
**Public API**: `read_variables(output_dir) -> SimpleNamespace`

Contains the single function that reads all variable CSV files (`v_flow.csv`, etc.) and returns
a `SimpleNamespace v` with properly named indices and column MultiIndexes.

**Simplification opportunity**: The repetitive index-naming block
```python
v.flow.index.names = ['solve', 'period', 'time']
v.ramp.index.names = ['solve', 'period', 'time']
# ... 20 more identical lines
```
can be reduced to a loop over a list of attribute names:
```python
for attr in ['flow', 'ramp', 'reserve', 'state', ...]:
    getattr(v, attr).index.names = ['solve', 'period', 'time']
```

---

### 2. `read_parameters.py` (~145 lines)
**Extracted from**: `read_flextool_outputs.py`
**Public API**: `read_parameters(output_dir) -> SimpleNamespace`

Contains the single function that reads all parameter CSV files and returns a `SimpleNamespace p`
with properly named indices and column names.

**Simplification opportunity**: Same as above — the long block of `p.X.columns.name = 'Y'`
assignments can be driven by a `{attr: colname}` dict and a loop.

---

### 3. `read_sets.py` (~145 lines)
**Extracted from**: `read_flextool_outputs.py`
**Public API**: `read_sets(output_dir) -> SimpleNamespace`

Contains the single function that reads all set definition CSV files. Simple scalar sets become
`pd.Index`, tuple sets become `pd.MultiIndex`.

**Simplification opportunity**: The repeated pattern
```python
df = pd.read_csv(output_path / 'set_groupInertia.csv')
s.groupInertia = pd.Index(df.iloc[:, 0])
```
appears ~8 times for single-column sets. Replace with a helper function or dict-driven loop:
```python
_SINGLE_COL_SETS = {
    'groupInertia': 'set_groupInertia.csv',
    'groupNonSync': 'set_groupNonSync.csv',
    ...
}
for attr, filename in _SINGLE_COL_SETS.items():
    setattr(s, attr, pd.Index(pd.read_csv(output_path / filename).iloc[:, 0]))
```

---

### 4. `drop_levels.py` (~105 lines)
**Extracted from**: `process_results.py`
**Public API**: `drop_levels(par, s, v) -> (par, s, v)`

Strips the `'solve'` level from all time-indexed variables, parameters and sets. This is a pure
preprocessing step that must happen before any calculations.

**Simplification opportunity**: The current code has ~50 almost-identical lines:
```python
par.step_duration = par.step_duration.droplevel('solve')
par.flow_min = par.flow_min.droplevel('solve')
# ...
```
Many also deduplicate (`[~index.duplicated(keep='first')]`). Replace with:
```python
_VARS_3D = ['flow', 'ramp', 'reserve', 'state', ...]  # drop only
_PARS_DEDUP = ['entity_all_existing', 'entity_pre_existing', ...]  # drop + dedup
_PARS_3D = ['step_duration', 'flow_min', ...]  # drop only

for attr in _VARS_3D:
    obj = getattr(v, attr)
    setattr(v, attr, obj.droplevel('solve'))

for attr in _PARS_DEDUP:
    obj = getattr(par, attr)
    obj = obj.droplevel('solve')
    setattr(par, attr, obj[~obj.index.duplicated(keep='first')])
```
This would reduce the function from ~90 lines to ~30 lines.

---

### 5. `calc_capacity_flows.py` (~130 lines)
**Extracted from**: `process_results.py`
**Public API**: `compute_capacity_and_flows(par, s, v, r) -> None` (mutates `r`)

Computes:
- `r.hours_in_realized_period`, `r.realized_period_share_of_year`
- `r.entity_all_capacity` (existing + cumulative invest - divest)
- `r.process_online_dt`
- `r.flow_dt` (the most complex calculation: applies unit-size scaling, slope/section
  transformations for `method_1var_per_way` processes)
- `r.flow_d`, `r.process_source_flow_d`, `r.process_sink_flow_d`
- `r.ramp_dtt`
- `s.process_source_sink_alwaysProcess` (derived set used by many callers)
- `s.nb` (union of node_balance and node_balance_period)

**Note on dependencies**: receives `r` as empty `SimpleNamespace`, fills it in-place; called
first after `drop_levels`.

---

### 6. `calc_connections.py` (~90 lines)
**Extracted from**: `process_results.py`
**Public API**: `compute_connection_flows(par, s, v, r) -> None` (mutates `r`)

Computes all connection-related derived quantities:
- `r.connection_dt`, `r.connection_losses_dt`
- `r.connection_to_left_node__dt`, `r.connection_to_right_node__dt`
- `r.connection_d`, `r.connection_losses_d`
- `r.connection_to_left_node__d`, `r.connection_to_right_node__d`
- Also provides `from_conn` and `to_conn` intermediate frames used by group aggregations
  (expose via `r.from_conn`, `r.to_conn` for `calc_group_flows.py`)

**Dependencies**: `r.flow_dt`, `s.process_connection`, `s.process_source`, `s.process_sink`

---

### 7. `calc_group_flows.py` (~180 lines)
**Extracted from**: `process_results.py`
**Public API**: `compute_group_flows(par, s, v, r) -> None` (mutates `r`)

Computes all group-level aggregations (the large second half of `post_process_results`):
- `r.group_output__unit_to_node_not_in_aggregate__dt/d`
- `r.group_output__node_to_unit_not_in_aggregate__dt/d`
- `r.group_output__group_aggregate_Unit_to_group__dt/d` (with sign-flip logic for negatives)
- `r.group_output__group_aggregate_Group_to_unit__dt/d`
- `r.group_output__from/to_connection_not_in_aggregate__dt/d`
- `r.group_output__from/to_connection_aggregate__dt/d`
- `r.group_output_Internal_connection_losses__dt/d`
- `r.group_output_Internal_unit_losses__dt/d`
- `r.group_node_inflow_dt/d`
- `r.group_node_state_losses__dt/d`
- `r.group_node_up/down_slack__dt`

**Dependencies**: `r.flow_dt`, `r.from_conn`, `r.to_conn`, `r.connection_losses_dt`,
`r.self_discharge_loss_dt`, `r.upward_node_slack_dt`, `r.downward_node_slack_dt`, sets

---

### 8. `calc_costs.py` (~140 lines)
**Extracted from**: `process_results.py`
**Public API**: `compute_costs(par, s, v, r) -> None` (mutates `r`)

Computes all cost quantities:
- Commodity costs: `r.cost_commodity_dt/d`, `r.sales_commodity_dt/d`
- CO2 emissions: `r.process_emissions_co2_dt/d`, `r.emissions_co2_dt/d`
- Group CO2: `r.group_co2_dt/d`, `r.group_cost_co2_dt/d`, `r.cost_co2_dt/d`
- Operational: `r.cost_process_other_operational_cost_dt/d`
- Startup: `r.process_startup_dt`, `r.cost_startup_dt/d`
- Investments: `r.cost_entity_invest_d`, `r.cost_entity_divest_d`,
  `r.cost_entity_fixed_pre_existing/invested/divested`
- Aggregates: `r.costOper_dt/d`, `r.costPenalty_dt/d`, `r.costOper_and_penalty_d`
- Investment aggregates by type: `r.costInvestUnit_d`, `r.costInvestConnection_d`, etc.

**Dependencies**: `r.flow_dt`, `r.process_online_dt`, `r.process_startup_dt`, `r.group_co2_d`, sets

**Simplification opportunity**: The period-aggregation block for penalty costs repeats the same
`if not df.empty ... else pd.DataFrame(0.0, ...)` pattern four times. Extract a helper function:
```python
def _agg_period_or_empty(df: pd.DataFrame, realized_periods, period_level='period') -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(0.0, index=realized_periods, columns=df.columns)
    return df[df.index.get_level_values(period_level).isin(realized_periods)].groupby(period_level).sum()
```

---

### 9. `calc_slacks.py` (~100 lines)
**Extracted from**: `process_results.py`
**Public API**: `compute_slacks(par, s, v, r) -> None` (mutates `r`)

Computes slack and reserve quantities:
- `r.reserves_dt/d`
- `r.upward_node_slack_dt/d`, `r.downward_node_slack_dt/d` (both annualized and not)
- `r.costPenalty_node_state_upDown_dt/d`
- `r.q_inertia_dt/d`, `r.costPenalty_inertia_dt`
- `r.q_non_synchronous_dt/d`, `r.costPenalty_non_synchronous_dt`
- `r.q_capacity_margin_d_not_annualized`, `r.costPenalty_capacity_margin_d`
- `r.q_reserves_dt/d`, `r.costPenalty_reserve_upDown_dt/d`

**Dependencies**: `v.q_state_up`, `v.q_state_down`, `v.reserve`, `v.q_inertia`,
`v.q_non_synchronous`, `v.q_capacity_margin`, `v.q_reserve`, parameter scalings

---

### 10. `calc_storage_vre.py` (~110 lines)
**Extracted from**: `process_results.py`
**Public API**: `compute_storage_and_vre(par, s, v, r) -> None` (mutates `r`)

Computes:
- `r.node_state_change_dt/d` (storage state change, with all four binding methods)
- `r.self_discharge_loss_dt/d`
- `r.node_inflow_d`
- `r.potentialVREgen_dt/d` (VRE capacity × availability × profile)
- `r.storage_usage_dt` (for nested model storage fixing)

**Dependencies**: `v.state`, `r.flow_dt`, `r.entity_all_capacity`, `par.process_availability`,
`par.profile`, `s.process_VRE`, storage binding method sets

---

### 11. `process_results.py` (thin coordinator, ~50 lines)
**Keeps**: `post_process_results(par, s, v) -> SimpleNamespace`
**Moves out**: everything above into their own modules

The coordinator simply calls all stages in the right order:
```python
from flextool.process_outputs.drop_levels import drop_levels
from flextool.process_outputs.calc_capacity_flows import compute_capacity_and_flows
from flextool.process_outputs.calc_connections import compute_connection_flows
from flextool.process_outputs.calc_storage_vre import compute_storage_and_vre
from flextool.process_outputs.calc_slacks import compute_slacks
from flextool.process_outputs.calc_costs import compute_costs
from flextool.process_outputs.calc_group_flows import compute_group_flows

def post_process_results(par, s, v):
    r = SimpleNamespace()
    par, s, v = drop_levels(par, s, v)
    compute_capacity_and_flows(par, s, v, r)
    compute_connection_flows(par, s, v, r)
    compute_storage_and_vre(par, s, v, r)
    compute_slacks(par, s, v, r)
    compute_costs(par, s, v, r)
    compute_group_flows(par, s, v, r)
    return r
```

---

### 12. `out_capacity.py` (~120 lines)
**Extracted from**: `result_writer.py`
**Contains**: `unit_capacity()`, `connection_capacity()`, `node_capacity()`

**Bug found — `connection_capacity` (line 120)**:
```python
# WRONG: assigns divest to 'invested' column
results['invested'] = v.divest.unstack()[ed_conn_divest] * par.entity_unitsize[conn_divest]
# CORRECT: should be
results['divested'] = v.divest.unstack()[ed_conn_divest] * par.entity_unitsize[conn_divest]
```

**Bug found — `node_capacity` (line 162)**:
```python
# WRONG: reads v.invest instead of v.divest when processing divested capacity
results['invested'] = v.invest.unstack()[ed_node_divest] * par.entity_unitsize[node_divest]
# CORRECT:
results['divested'] = v.divest.unstack()[ed_node_divest] * par.entity_unitsize[node_divest]
```

**Simplification opportunity**: The three capacity functions (`unit_capacity`, `connection_capacity`,
`node_capacity`) are structurally identical. Extract a shared helper:
```python
def _entity_capacity_table(par, s, v, r, entities, index_name, entity_set, invest_set, divest_set):
    ...
```

---

### 13. `out_flows.py` (~200 lines)
**Extracted from**: `result_writer.py`
**Contains**:
- `unit_outputNode()` — unit output flows (dt and d)
- `unit_inputNode()` — unit input flows (dt and d)
- `unit_cf_outputNode()` — unit capacity factors by output node
- `unit_cf_inputNode()` — unit capacity factors by input node
- `unit_VRE_curtailment_and_potential()` — VRE curtailment/potential (dt and d)
- `unit_ramps()` — ramp results (dt)

---

### 14. `out_group.py` (~200 lines)
**Extracted from**: `result_writer.py`
**Contains**:
- `nodeGroup_flows()` — group flow breakdown (dt and d) — the largest output function
- `nodeGroup_total_inflow()` — total inflow to groups (dt and d)
- `nodeGroup_indicators()` — VRE share, curtailment, slack indicators (dt and d)
- `nodeGroup_VRE_share()` — VRE share time series (dt and d)

**Simplification opportunity**: `nodeGroup_indicators()` and `nodeGroup_VRE_share()` compute
almost the same quantities (VRE flow sum, total inflow, VRE share). Extract a shared helper to
compute these per-group and per-timestep, then call it from both functions.

---

### 15. `out_node.py` (~120 lines)
**Extracted from**: `result_writer.py`
**Contains**:
- `node_summary()` — node balance breakdown by category (dt and d)
- `node_additional_results()` — node prices, state, up/down slack (dt and d)

---

### 16. `out_costs.py` (~85 lines)
**Extracted from**: `result_writer.py`
**Contains**:
- `cost_summaries()` — annualized and discounted cost tables
- `CO2()` — CO2 totals and process CO2 emissions
- `generic()` — debug-mode discount factor tables

---

### 17. `out_ancillary.py` (~120 lines)
**Extracted from**: `result_writer.py`
**Contains**:
- `connection()` — connection flows (dt and d)
- `connection_wards()` — leftward/rightward connection flows (dt and d)
- `connection_cf()` — connection capacity factors
- `reserves()` — reserve results (dt and d)
- `investment_duals()` — investment dual variable outputs
- `inertia_results()` — unit inertia and group inertia (dt)
- `slack_variables()` — reserve/non-sync/inertia/capacity-margin slacks
- `input_sets()` — pass-through sets needed by `scenario_results`

---

### 18. `write_outputs.py` (~380 lines)
**Extracted from**: `result_writer.py` (the bottom half)
**Contains**:
- `ALL_OUTPUTS` list (imports functions from all `out_*.py` modules)
- `log_time()` — timing + progress CSV utility
- `print_namespace_structure()` — debug utility
- `write_summary_csv()` — writes summary_solve.csv
- `write_outputs()` — main orchestrator (settings resolution, read, compute, write)
- `__main__` argument-parsing block

**Simplification opportunity**: The `write_outputs()` function does three unrelated things:
(a) resolves settings from DB/defaults, (b) reads/processes data, (c) writes outputs. Extract
the settings resolution into a small `_resolve_settings()` private function to reduce visual
complexity and make testing easier.

---

### 19. `to_spine_db.py` (keep, minor cleanup)
**Bug found in `_add_arrays` (line 496)**:
```python
# WRONG: map(array, col) — 'array' is not a function here
entity_names = ['__'.join(map(array, col)) for col in df.columns]
# CORRECT:
entity_names = ['__'.join(map(str, col)) for col in df.columns]
```

**Simplification opportunity**: `_add_time_series`, `_add_strs`, and `_add_arrays` share ~70%
identical boilerplate (class-name parsing, parameter definition, iteration over entities). Extract
shared scaffolding into a helper and have each function supply only its type-specific logic.

---

### 20. `__init__.py` (update imports)

```python
"""Output data processing: reads solver CSV results, post-processes, and writes outputs."""
from flextool.process_outputs.read_variables import read_variables
from flextool.process_outputs.read_parameters import read_parameters
from flextool.process_outputs.read_sets import read_sets
from flextool.process_outputs.process_results import post_process_results
from flextool.process_outputs.write_outputs import write_outputs

__all__ = ['read_variables', 'read_parameters', 'read_sets', 'post_process_results', 'write_outputs']
```

---

## Summary of bugs found

| File | Location | Bug |
|---|---|---|
| `result_writer.py` | `connection_capacity()` ~line 120 | Assigns `v.divest` result to `results['invested']` instead of `results['divested']` |
| `result_writer.py` | `node_capacity()` ~line 162 | Reads `v.invest.unstack()` instead of `v.divest.unstack()` when computing divested capacity |
| `to_spine_db.py` | `_add_arrays()` ~line 496 | `map(array, col)` should be `map(str, col)` |

---

## Summary of simplification opportunities

| Category | Description | Estimated line reduction |
|---|---|---|
| `read_variables.py` | Loop over list to set index names instead of 20 repetitive lines | ~15 lines |
| `read_parameters.py` | Dict-driven loop for `columns.name` assignments | ~20 lines |
| `read_sets.py` | Dict-driven loop for single-column sets | ~15 lines |
| `drop_levels.py` | List-driven loops instead of 90 individual lines | ~60 lines |
| `calc_costs.py` | `_agg_period_or_empty` helper for repeated empty-check pattern | ~25 lines |
| `out_capacity.py` | Shared `_entity_capacity_table` helper for three near-identical functions | ~60 lines |
| `out_group.py` | Shared VRE-share helper for `nodeGroup_indicators` and `nodeGroup_VRE_share` | ~30 lines |
| `to_spine_db.py` | Shared scaffolding for `_add_time_series/strs/arrays` | ~40 lines |
| `write_outputs.py` | Extract `_resolve_settings()` from `write_outputs()` | readability |

Total estimated reduction: **~265 lines** removed by simplification alone.

---

## Implementation order for tasks

1. Split `read_flextool_outputs.py` → `read_variables.py`, `read_parameters.py`, `read_sets.py`
   (pure extraction, no logic change)
**Status**: implemented
2. Extract `drop_levels.py` from `process_results.py` (with simplification)
**Status**: implemented
3. Extract `calc_capacity_flows.py`, `calc_connections.py`, `calc_storage_vre.py` (no shared state)
**Status**: implemented
4. Extract `calc_slacks.py`, `calc_costs.py`, `calc_group_flows.py`
**Status**: implemented
5. Slim down `process_results.py` to coordinator
**Status**: implemented
6. Extract output functions: `out_capacity.py`, `out_flows.py`, `out_node.py`, `out_costs.py`,
   `out_ancillary.py` (with bug fixes)
**Status**: implemented
7. Extract `out_group.py` (most complex, with shared VRE helper)
**Status**: implemented
8. Extract `write_outputs.py` from `result_writer.py`, update `ALL_OUTPUTS` imports
**Status**: implemented
9. Fix `to_spine_db.py` bug, apply simplifications
**Status**: implemented
10. Update `__init__.py` and `ARCHITECTURE.md`
**Status**: not done
