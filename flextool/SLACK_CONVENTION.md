# Slack convention

Reference for how every `vq_*` slack variable in `flextool.mod` is
declared, bounded, penalised in the objective, and written to output
CSV.  Established by the LP-scaling work started 2026-04-22 (see
`~/.claude/projects/-home-jkiviluo-sources-flextool/memory/project_lp_scaling_2026-04.md`
for the full project context).

This document is the single source of truth when adding a new slack or
editing an existing one.  Agents 2–4 of the scaling project incrementally
bring every slack under the convention described here.

## One-line summary

**Relative internal, absolute at API boundary.**  Every `vq_*` is
declared as a fraction of its row-scaler (node capacity, group capacity,
reservation, inertia limit, capacity margin — whichever matrix row the
slack sits in).  It is split into two tiers: a bounded **primary** slack
`≤ K_rel` plus an unbounded **escape** slack.  Both are added into the
same constraint row (with the same scaler multipliers); both are
penalised in the objective; the escape penalty is `primary penalty ×
1000`.  Outputs un-scale everything back to the user's absolute units so
downstream consumers see no change.

## Why two tiers

* **Primary (`≤ K_rel`)** keeps the LP matrix bounded.  With
  `K_rel = 1`, the slack column contributes coefficients on the order
  of the row-scaler — predictable, scaler-compatible, friendly to
  HiGHS's coefficient scaling and symmetry detection.
* **Escape (`≥ 0`, no upper bound)** absorbs pathological inputs that
  blow past the primary cap.  Without this, the solver would report
  false infeasibility.  Activity on the escape tier becomes a
  user-facing diagnostic — "the model is telling you something is
  wrong with your inputs," not "solver failed."
* **Combined feasibility** — any non-negative value the old single
  slack could reach is still reachable as `primary + escape`, so the
  feasible region is unchanged; only the numerical presentation
  differs.

## The `× 1000` escape multiplier

`escape_multiplier = 1000` is a **hard-coded internal constant**, not a
user parameter.  It is applied globally as
`objective escape term = objective primary term × 1000`.

Rationale:

* A factor of 1000 is large enough that the escape tier is never
  preferred over the primary tier when primary has headroom — any
  realistic user-supplied penalty is at most O(10) times the "natural"
  cost magnitude of the system (€/MWh ~ 10–1000), so the primary tier
  wins by a safe margin on any well-posed model.
* A factor of 1000 is small enough that, once the escape tier does
  activate, objective-range spread stays under four orders of
  magnitude — HiGHS can still scale cleanly.  Going to `1e6` or higher
  causes objective-range blowouts that trigger HiGHS scaling warnings.
* Users who want to further penalise pathological-input activity do so
  by increasing the user-facing `penalty_*` parameter — it scales both
  tiers together.  No need to expose a separate escape knob.

Keep the constant hard-coded until there is concrete evidence a user
needs to tune it.  Exposing it as a parameter now would multiply the
API surface without benefit.

## Convention (conversion target)

For a slack `vq_foo` whose row appears in constraint `R[i, …]` with
row-scaler `S[i, …]`:

```ampl
# Declaration — two variables
var vq_foo_primary {…indices…} >= 0, <= 1;
var vq_foo_escape  {…indices…} >= 0;

# In the constraint row — sum, multiplied by the row scaler
s.t. R {…indices…}:
    …
    + (vq_foo_primary[…] + vq_foo_escape[…]) * S[…, d] * step_duration[…]
    …
    ;

# In the objective — two parallel sums, same multipliers except penalty
  + sum {…} pdt_branch_weight[d,t]
            * vq_foo_primary[…]
            * S[…, d]
            * pdGroup[…, 'penalty_foo', d]
            * step_duration[…]
            * p_rp_cost_weight[d, t]
            * p_inflation_factor_operations_yearly[d]
            / complete_period_share_of_year[d]
  + sum {…} pdt_branch_weight[d,t]
            * vq_foo_escape[…]
            * S[…, d]
            * pdGroup[…, 'penalty_foo', d] * 1000   # escape multiplier
            * step_duration[…]
            * p_rp_cost_weight[d, t]
            * p_inflation_factor_operations_yearly[d]
            / complete_period_share_of_year[d]
```

Every multiplier other than the penalty parameter is identical between
the two sums — so the only difference between a primary unit and an
escape unit is the `× 1000` penalty.

## The slack roster

Seven `vq_*` slacks.  `K_rel` defaults to 1 for every slack; for
`vq_state_up` / `vq_state_down` it is replaced per-node-per-period by
`p_state_slack_k_rel[n, d]`, which the Python pre-solve helper
(`flextool/flextoolrunner/slack_bounds.py`) rounds up to the nearest
power of 10 of the node's max absolute demand divided by
`node_capacity_for_scaling[n, d] × min_step_duration[d]`.  Power-of-10
rounding keeps structurally-identical nodes on the same cap so HiGHS
symmetry detection survives.  "No t" means period-only indexing (no
time dimension); the primary bound applies per period–entity instead
of per timestep.

| Slack | Index set | Row scaler (internal) | Status |
|---|---|---|---|
| `vq_state_up`        | `(n, d, t)` for `n` in `nodeBalance ∪ nodeBalancePeriod` | `node_capacity_for_scaling[n, d]` | converted (Agent 3); primary cap `p_state_slack_k_rel[n, d]` computed per-solve by `flextool/flextoolrunner/slack_bounds.py` |
| `vq_state_down`      | `(n, d, t)` for `n` in `nodeBalance ∪ nodeBalancePeriod` | `node_capacity_for_scaling[n, d]` | converted (Agent 3); same `p_state_slack_k_rel` cap as `vq_state_up` |
| `vq_reserve`         | `(r, ud, ng, d, t)` for `(r, ud, ng)` in `reserve__upDown__group` | `pdtReserve_upDown_group[r, ud, ng, 'reservation', d, t]` | converted (Agent 4); primary keeps historic name (already ≤ 1), only `vq_reserve_escape` added — see naming asymmetry note below |
| `vq_inertia`         | `(g, d, t)` for `g` in `groupInertia` | `pdGroup[g, 'inertia_limit', d]` | converted (Agent 4); primary keeps historic name (already ≤ 1), only `vq_inertia_escape` added — see naming asymmetry note below |
| `vq_non_synchronous` | `(g, d, t)` for `g` in `groupNonSync` | `group_capacity_for_scaling[g, d]` | converted (Agent 2); parquet extractor fanned out (Agent 3a) |
| `vq_capacity_margin` | `(g, d)` for `g` in `groupCapacityMargin` — **no t** | `group_capacity_for_scaling[g, d]` | converted (Agent 4); historic `× 1000` hack removed — escape multiplier now provides that role; cost-family scaling (Agent 6) will handle period-scale vs timestep-scale balance |
| `vq_state_up_group`  | `(g, d, t)` for `g` in `group_loss_share` | `group_capacity_for_scaling[g, d]` | converted (Agent 4); no objective term of its own -- penalised indirectly through `vq_state_up` via `group_loss_share_constraint`; two-tier split still applied so bounds match the convention |

### Naming asymmetry: `vq_reserve` and `vq_inertia`

For every two-tier slack added by Agents 2 and 3, both tiers carry an
explicit suffix: `vq_foo_primary` + `vq_foo_escape`.  For `vq_reserve`
and `vq_inertia`, the primary already had the correct `≤ 1` bound
before Agent 4, and renaming either would require touching every
constraint, objective term, output, and extractor site.  Instead,
Agent 4 left the primary name as-is (still `vq_reserve`, still
`vq_inertia`) and only added the escape companion — `vq_reserve_escape`
and `vq_inertia_escape`.

All other conventions still hold: constraint rows sum primary +
escape, the objective carries two parallel terms with a `× 1000`
escape multiplier, the CSV writes `primary + escape`, and the parquet
extractor uses `derived_from=("vq_reserve", "vq_reserve_escape")`
(likewise for inertia) to fan out both HiGHS columns into the single
logical output.

## Output CSV convention

For every converted slack, the written value per row is

```
csv_value = primary + escape
```

expressed in **absolute units** — multiplied by the row-scaler and
step duration where the pre-conversion writer did so.  The file name,
column headers, row structure, and numerical scale are identical to the
pre-conversion output so downstream pandas readers
(`process_outputs/read_variables.py`, scenario analysis, plotting) work
unchanged.

Example for `vq_non_synchronous` (Agent 2 reference conversion):

```ampl
printf ",%.6g",
    (vq_non_synchronous_primary[g, d, t].val
     + vq_non_synchronous_escape[g, d, t].val)
    >> "output_raw/vq_non_synchronous.csv";
```

Note that `group_capacity_for_scaling[g, d]` defaults to 1 in the
pre-Agent-5 state, so today `primary + escape` numerically equals the
old single-variable value.  Once Agent 5 activates row scaling, the
written value remains in the old "fraction × capacity × duration"
absolute units because both tiers are multiplied back out inside the
writer (or the Python un-scaler).  Callers never see the internal
scaling.

## Escape-tier activity reporting

When any escape variable holds a value above the feasibility tolerance,
the model should surface that fact.  Feasibility tolerance is taken as
`1e-8` relative to the row scaler — i.e. an absolute threshold of
`1e-8 × S[…]` (when the scaler is 1, this is `1e-8` absolute).

Current agent-2 implementation emits a single stdout line per slack
after the solve:

```
WARNING: vq_foo escape tier active — sum=<value>.
This indicates input demand exceeded bounded-slack capacity;
check <slack-specific hint>.
```

The detailed per-entity diagnostic (which groups/nodes are driving the
escape activity, by how much, and what user-facing input to look at) is
Agent 10's work.  For now, the single stdout line is a sentinel —
enough for a user to know "something used the escape tier; go find
it" — but not a drill-down.

Once Agent 10 lands, the same condition will also append a block to
`scaling_report.txt`.

## When bounded slack saturates

"Saturating" means the primary slack reaches its upper bound (`= 1`)
and the escape tier takes the remainder.  Three cases:

1. **Primary not saturated** — no diagnostic; the model is behaving
   normally and slack usage is in the predictable range.
2. **Primary saturated, escape = 0** — the primary bound exactly
   matched demand.  Rare but benign.  No diagnostic.
3. **Primary saturated, escape > feasibility tolerance** — diagnostic
   fires.  The user's inputs are demanding more slack than `K_rel` per
   row-scaler per step allows.  Either the row scaler is too small for
   the scenario, or the underlying constraint is genuinely infeasible
   on the user's data and slack is absorbing the contradiction.  A
   well-scaled model with sensible inputs never fires case 3.

In the pre-Agent-5 world, `S = 1` everywhere, so case 3 means the
underlying slack demand (in absolute units) exceeds `1 × step_duration`
per row per timestep.  For scenarios where historical `vq_*` usage
stayed below this, nothing changes.  When Agent 5 activates row
scaling, `S` becomes the true row capacity, and case 3 becomes the
"something is genuinely wrong with the inputs" signal intended.

## Editing checklist when adding / converting a slack

1. **Declaration** — two `var` lines, primary with `>= 0, <= 1`, escape
   with `>= 0`.
2. **Every constraint row** that includes the slack — replace the
   single slack reference with the `(primary + escape)` sum,
   multiplied by the row scaler.
3. **Objective** — two parallel sums, identical multiplier chain, with
   escape penalty = primary penalty × 1000.
4. **Output writer** — keep the CSV file name / columns unchanged;
   write `(primary + escape).val` multiplied by whatever factors the
   pre-conversion writer used.
5. **Post-solve diagnostic** — one stdout line if `sum escape.val >
   1e-8 × scaler`.
6. **Baseline comparison** — run the four benchmark scenarios via
   `scaling_benchmark/run_benchmarks.py --compare …`; confirm zero
   material delta on objective, slack totals, matrix range (new column
   in LP may shift nnz and range slightly — record the new values).
7. **Grep for leftover references** to the old single-variable name —
   Python CSV readers should keep working because the CSV is
   unchanged; parquet readers may return an empty frame until the
   extractor registry is updated, but this is harmless so long as no
   baseline scenario had non-zero activity.
