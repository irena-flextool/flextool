# Baseline changelog

Records every time the four scenario baselines under
`scaling_benchmark/baseline/*.json` are overwritten with a fresh run.

## 2026-04-22 — Agent 18c validation (variable-bound scaling)

No baseline refresh needed — Agent 18c's `user_bound_scale` lever only
fires under `--auto-scale`, and the 4 project scenarios either stay
below the 6-decade bound-spread threshold or converge cleanly when it
fires. All four baselines (`small_building`, `medium_national`,
`continental`, `composite`) compare clean vs the Agent 12 refresh
under the new code path (`run_benchmarks.py --compare` EXIT=0 for
each).

**Rivendell S19 auto-scale end-to-end validation.** Running
`rivendell/rivendell.sqlite` scenario `S19_cascade_hourly_rp` with
`--auto-scale` now:
- Analyser auto-applies `use_row_scaling=yes` (Agent 18b trigger
  fires on the 5-decade unitsize spread).
- Analyser auto-applies `user_bound_scale=-20` (bound range
  `[2e-3, 1e+6]`, spread 8.70 decades; `N = -round(log2(1e6)) = -20`,
  below clamp).
- HiGHS accepts the option (`"Assessing costs and bounds after
  applying user_bound_scale option value of -20"`).
- The Markowitz stall the Agent 18 report described **still fires**
  at every tested N (-6, -8, -10, -12, -20). Stall is a characteristic
  of this model's dual-simplex behaviour, not of bound spread alone —
  details in `projects/rivendell/agent18c_bound_scale/BENCHMARK_REPORT.md`
  (local; gitignored).

**Takeaways:**
- Mechanism works: HiGHS 1.14 exposes `user_bound_scale` via highspy
  `setOptionValue`, solution-invariant as advertised.
- For rivendell-shaped stalls, bound scaling alone is not sufficient.
  A follow-up agent could tune the `N` heuristic (current choice is
  slightly more aggressive than HiGHS's own `-8` hint) or investigate
  IPM vs dual-simplex or Markowitz strategy changes.
- Merge criterion met: infrastructure correct, tests green,
  benchmarks invariant, opt-in/opt-out respected.

## 2026-04-22 — Agent 12 refresh (centralised scale_the_objective)

Baselines refreshed after Agent 12 centralised `scale_the_objective`
in Python.  The analyser's power-of-10 recommendation is now emitted
per solve (`solve_data/scale_the_objective.csv`) instead of reading
the hardcoded `1E-6` from `flextool_base.dat`.

**Invariant vs Agent 11 baselines:**
- `objective` — match within machine precision (`< 3e-16` relative)
  across all four scenarios.  User-facing values unchanged.
- `slack_totals.*` — exact match in all four scenarios.
- `matrix_range`, `bound_range`, `rhs_range`, `rows_*` / `cols_*` /
  `nnz_*` — no change.

**Changed vs Agent 11 baselines (expected and documented):**
- `matrix_range_from_mps`, `cost_range` — reflect the new internal
  `scale_the_objective` values the analyser picks per scenario:
  `1e-9` for `small_building`, `medium_national`, `composite`;
  `1e-6` (unchanged) for `continental`.  Output un-scaling in
  `read_highs_solution.py` was made dynamic at the same time, so
  user-facing objective / dual / slack CSV values are preserved.

## 2026-04-22 — LP-scaling-2026-04 final refresh (Agent 11)

Baselines refreshed after Agents 2-10 landed on branch `lp-scaling-2026-04`.

**Invariant vs Agent 1 (`fba96f7`) baselines:**
- `objective` — exact match in all four scenarios (no change to LP optimum).
- `slack_totals.*` — exact match in all four scenarios.

**Changed vs Agent 1 baselines (expected and documented):**
- `cost_range`, `bound_range` — reflect the two-tier slack split added by
  Agents 2-4. Bounded primary slacks (`<= K_rel`) added new `<= 1` upper
  bounds; escape slacks added new small coefficients in the cost row.
- `cols_initial`, `nnz_initial` — each converted slack gained an escape
  companion column, so column counts and nonzero counts grew by a
  fixed amount per slack.
- `rows_initial` — unchanged by the slack split (escape shares the same
  constraint row as primary) but may reflect Agent 5's row-scaling
  infrastructure when active.

The refreshed baselines capture state at commit `a99c1788` (Agent 10b
HEAD at the moment of baseline capture), with row scaling **off** by
default. To capture baselines under row scaling on, re-run with
`FLEXTOOL_FORCE_ROW_SCALING=1` before `--write-baseline`. See
`scaling_benchmark/VALIDATION_REPORT.md` for the full matrix.

After refresh, `python scaling_benchmark/run_benchmarks.py --scenario
<s> --compare scaling_benchmark/baseline/<s>.json` exits 0 for all four
scenarios.

## 2026-04-22 — Initial (Agent 1)

First baselines captured. Commit `fba96f7`.
