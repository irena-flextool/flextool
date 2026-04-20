# Noted decisions — commit #5 (rolling cumulative-quota validation scenario)

Non-obvious choices made while implementing the end-to-end validation
scenario (step 4e of `PLAN_rolling_quota_handoff.md`).  One entry per
decision, with file:location where relevant.

## Test module (tests/test_commodity_ladder_rolling.py)

- **Programmatic scenario build instead of a `tests/scenarios.yaml`
  entry.**  The existing YAML-driven scenario tests compare against
  golden CSVs.  The rolling-cumulative validation needs to inspect the
  in-flight `solve_data/cumulative_ladder_remaining.csv` and assert
  the running-balance *formula* against per-roll `v_trade` parquets
  — golden-file equality is the wrong tool here because LP dispatch
  on this small topology can pick symmetric bases without changing
  the structural invariants.  Pattern copied from
  `test_commodity_ladder_smoke.py` (commit #2) which took the same
  programmatic-scenario route for the same reason.

- **Two distinct periods (p2020 + p2025), one roll each**, instead of
  the "split a single year into halves" design sketched in PLAN §8.2.
  Using rolls that *share a period* triggers a real writer bug (see
  FOUND BUG #1 below) where span_weight == total_weight for every
  roll and allotments sum to `N * total_cap`.  Cross-period rolling
  keeps allotments partitioning cleanly
  (`span_weight[roll_i] = weight_of_one_period`,
  `Σ span_weight = total_weight`).  When the writer is fixed to do
  per-`(period, step)` accounting, this test can optionally be
  retargeted at within-period rolling.

- **`realized_periods` is an `Array`, not a `Map`** — `solve_config.
  periods_to_tuples` iterates `param_value.values` and expects string
  period names.  A `Map(["p2020", "p2025"], [True, True])` yields
  `True`/`False` at the outer iteration, which silently produces an
  empty set of realized periods.  Matched the `dispatch_fullYear_roll`
  fixture encoding (`Array` of strings with index_name="period").

- **Overspend test split into two cases.**  The single-solve variant
  (roll-1-only) exercises the writer's negative-remaining path
  directly — the solve succeeds, the writer records a large negative
  `p_cumulative_ladder_remaining`, and the structural assertions
  fire.  The full two-roll overspend is marked `xfail(strict=True)`
  pending a mod-side fix (FOUND BUG #2 below).  Keeping it as xfail
  rather than deleting it means the fix can flip the marker rather
  than the test getting rewritten.

- **Bit-identity assertion (task brief §D5) is structural, not
  cross-commit.**  Single-solve cumulative with a large cap == legacy
  `coal` price run's objective within `rel=1e-6`.  That's what the
  existing `test_commodity_ladder_smoke.py::test_objective_matches_coal`
  assertion does for annual mode; the same guarantee holds for
  cumulative because the empty-seed first-solve defaults keep the
  cumulative constraint inactive on every single solve.

## Found bugs — DO NOT PAPER OVER (report before proceeding)

### FOUND BUG #1 — writer double-counts allotments when multiple rolls share a period

**Symptom**: run `fullYear_roll`-style scenario (4 periods, many
rolls per period via `rolling_solve_horizon=8 / jump=4`) with a
`price_ladder_cumulative` commodity.  Each roll's writer sees the
same period in `realized_dispatch.csv`, sums
`years_represented[d] / period_share[d]` over the same `{d}`, and
records `span_weight == total_weight` for every roll.  Total
allotment over N rolls = `N * total_cap` instead of `total_cap`.

**Root cause**: the commit-#4 writer explicitly chose period-level
accounting (NOTES_commit4 under "cumulative_handoffs.py"):

> Because `v_trade` is period-level (no time, no branch), the
> consumption formula has **no** `step_duration` and **no**
> `p_rp_cost_weight` — the plan document predates the v_trade design
> and its per-(d,t) formulas must be re-derived to match this period
> indexing.

This is correct **if rolls never share a period**, which isn't
enforced anywhere.  `fullYear_roll` (rolling inside a single period)
and hour-level rolling in a multi-year setup both break.

**Fix options** (pick one, all require a design revisit):

1. Add a `realized_dispatch_fraction[d]` = (hours_realized_in_period) /
   (total_hours_in_period) written by the mod, and multiply span_weight
   by this.  Would keep v_trade period-level in the LP but split the
   allotment accounting by how much of each period committed.
   Consumption attribution would need a parallel split — the writer
   would need to multiply v_trade by the same fraction.  Trouble:
   `v_trade` is a single LP decision per (c, n, d, i); splitting it
   across rolls at the Python layer doesn't match the LP's view.
2. Drop v_trade's period aggregation and make it `v_trade[c, n, d, t, i]`
   (per-step).  Per-step accounting falls out naturally.  Costly
   mod-side, but the "right" answer.
3. Document that `price_ladder_cumulative` is only safe for
   scenarios where every roll realizes *distinct* periods.  Add a
   runtime check in `solve_config` / `cumulative_handoffs.py` that
   raises on overlap.  Keeps the current formulation but shrinks the
   supported scope.

**Impact on the validation scenario**: minor.  This commit avoids
the bug by using cross-period rolling.  Needs a follow-up commit
(probably step 4f or a new bullet) to actually resolve before
real-world datasets with within-period rolling land.

### FOUND BUG #2 — negative remaining makes LP infeasible

**Symptom**: run the two-roll overspend scenario with `tier1_cap=1 MWh`.
Roll 1 consumes >> 1 MWh (the seed keeps the constraint inactive on
the first solve, by design).  Writer records
`p_cumulative_ladder_remaining[coal, 1] ≈ −10.6M`.  Roll 2's LP has:

```
sum over (d in period_in_use, n in commodity_node) of
    v_trade[coal, n, d, 1] * unitsize[coal] * years[d] / share[d]
<= -10.6M
```

All LHS terms are non-negative (`v_trade >= 0`), so the LP is
structurally infeasible.  HiGHS correctly reports `kInfeasible`.

**Plan §1.1 claimed** "the optimizer is then forced to skip that
tier (or the LP is infeasible if no tail tier exists — document
this)", but **a tail tier doesn't help** — tier 1's constraint is
independent of tier 2, and forcing `v_trade[tier=1] = 0` can't be
done with a `<= −X` upper bound on a non-negative variable.

**Fix options**:

1. Mod-side: clamp RHS via `max(0, p_cumulative_ladder_remaining)`
   — easy, but loses the ability to carry a deficit forward (which
   the plan wanted).
2. Mod-side: when `p_cumulative_ladder_remaining[c, i] < 0`, add a
   separate constraint `v_trade[c, n, d, i] = 0` for every
   `(c, n, d)` pair.  Keeps the deficit visible in the CSV.  Simple.
3. Writer-side: clamp `remaining` at 0 before writing.  Loses the
   overspend signal entirely.

Recommendation: option 2.  The deficit value in the CSV is diagnostic
info; the LP should reject any v_trade on that tier until a later
roll brings the balance back to non-negative (which for a
never-increasing total is only via adjusted `total_cap`, so
practically: tier stays at zero once overspent).

**Impact on the validation scenario**: one test marked `xfail`.
The single-solve writer half is still validated (test passes).

## Guardrail compliance

- Did NOT modify `rivendell/` except for this NOTES file.
- Did NOT push.
- Did NOT touch memory files.
- Did NOT modify the LP mod or handoff writer (two bugs found;
  reported above; follow-up commits needed).
- No docs changes.

## Red flags for step 4f (CO2 cumulative + nested guard + stochastic)

- **FOUND BUG #1 must land before nested rolling** — nested rolling
  makes the within-period share case worse (sibling sub-solves under
  one parent all realize the same parent-visible periods).
- **FOUND BUG #2 also affects CO2** — the planned CO2 cumulative RHS
  swap (§2.2) uses the same `<= remaining` pattern and will hit the
  same infeasibility issue on any roll where consumption overshoots.
  Fix the ladder case first, then port the fix into the CO2 path.
- Stochastic branching (`pdt_branch_weight`) is a non-issue here —
  validation scenario has no stochastics — but FOUND BUG #1's span-
  weight formula would need a branch-weight factor once stochastics
  land, to match the LP's expected-value cost accounting.
