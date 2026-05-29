# Slack convention

Reference for how every `vq_*` slack variable in `flextool.mod` is
declared, bounded, penalised in the objective, and written to output
CSV.

This document is the single source of truth when adding a new slack or
editing an existing one.

## One-line summary

**Single-variable slacks; the user-supplied penalty is the valve.**
Every `vq_*` is a single non-negative variable (with a scaler-relative
`<= 1` upper bound where that was historically natural).  The penalty
coefficient already keeps the slack quiescent on well-posed inputs and
also lets the solver absorb pathological inputs without returning false
infeasibility.  Outputs un-scale row-scaled slacks back to the user's
absolute units so downstream consumers see no change.

## Rationale

An earlier design split each slack into a bounded primary (`<= K_rel`)
plus an unbounded escape at 1000x penalty, with the idea that the
primary tier kept the LP matrix bounded and the escape tier absorbed
pathological input.  In practice the existing high penalty already
performs both roles on its own: a large penalty is preferred over
infeasibility, and the solver will drive slack to zero when there is
any feasible alternative.  Two columns with identical constraint
coefficients but different objective weights are also a degeneracy
generator; removing the escape tier eliminates that.

## Convention

For a slack `vq_foo` whose row appears in constraint `R[i, ...]` with
row-scaler `S[i, ...]`:

```ampl
# Declaration — single variable
var vq_foo {...indices...} >= 0;                # unbounded
# or, where historically bounded ``<= 1`` makes physical sense
var vq_foo {...indices...} >= 0, <= 1;

# In the constraint row
s.t. R {...indices...}:
    ...
    + vq_foo[...] * S[..., d] * step_duration[...]
    ...
    ;

# In the objective — single sum
  + sum {...} pdt_branch_weight[d, t]
            * vq_foo[...]
            * S[..., d]
            * pdGroup[..., 'penalty_foo', d]
            * step_duration[...]
            * p_rp_cost_weight[d, t]
            * p_inflation_factor_operations_yearly[d]
            / complete_period_share_of_year[d]
```

## The slack roster

Seven `vq_*` slacks.  "No t" means period-only indexing (no time
dimension).

| Slack | Index set | Row scaler (internal) | Upper bound |
|---|---|---|---|
| `vq_state_up`        | `(n, d, t)` for `n` in `nodeBalance ∪ nodeBalancePeriod` | `node_capacity_for_scaling[n, d]` | none |
| `vq_state_down`      | `(n, d, t)` for `n` in `nodeBalance ∪ nodeBalancePeriod` | `node_capacity_for_scaling[n, d]` | none |
| `vq_reserve`         | `(r, ud, ng, d, t)` for `(r, ud, ng)` in `reserve__upDown__group` | `pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t]` | `<= 1` |
| `vq_inertia`         | `(g, d, t)` for `g` in `groupInertia` | `pdGroup[g, 'inertia_limit', d]` | `<= 1` |
| `vq_non_synchronous` | `(g, d, t)` for `g` in `groupNonSync` | `group_capacity_for_scaling[g, d]` | none |
| `vq_capacity_margin` | `(g, d)` for `g` in `groupCapacityMargin` — no t | `group_capacity_for_scaling[g, d]` | none |
| `vq_state_up_group`  | `(g, d, t)` for `g` in `group_loss_share` | `group_capacity_for_scaling[g, d]` | none; penalised indirectly through `vq_state_up` via `group_loss_share_constraint` |

## Output CSV convention

For every slack, the written value per row is the single-variable value
expressed in **absolute units** — multiplied by the row scaler and
step duration where the writer does so.  The file name, column
headers, row structure, and numerical scale are identical to the
pre-scaling-project output so downstream pandas readers
(`process_outputs/read_variables.py`, scenario analysis, plotting)
work unchanged.  Row scaling un-scaling (`× node_cap` or `× group_cap`)
is applied by the writer when `p_use_row_scaling=1` (the legacy LP
row-scaling family — independent of the `autoscale/` package; the
former `solve.use_row_scaling` DB parameter was removed, so this is now
only forced via the `FLEXTOOL_FORCE_ROW_SCALING` test hook); in Mode A
the scaler defaults to 1 so the factor is a no-op.

## Editing checklist when adding a new slack

1. **Declaration** — one `var` line, `>= 0` and optionally `<= 1` where
   the historic row structure justifies a relative cap.
2. **Every constraint row** that includes the slack — add a term
   multiplied by the row scaler.
3. **Objective** — one sum term with the user-supplied penalty.
4. **Output writer** — write `vq_foo.val` multiplied by whatever
   row-scaler factors are in effect so the CSV stays in absolute units.
5. **Parquet registry** — add a `VariableSpec` in
   `flextool/process_outputs/read_highs_solution.py`; set
   `unscale_by="node_cap"` or `"group_cap"` if the slack appears in a
   row-scaled constraint.
6. **Baseline comparison** — run the engine_polars gate and confirm
   no material delta on objective, slack totals, matrix range.
