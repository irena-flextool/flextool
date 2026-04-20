# Noted decisions — commit #4 (rolling cumulative-quota handoff)

Non-obvious choices made while implementing the rolling cumulative
handoff for `price_ladder_cumulative` tiers.  One line per decision,
prefixed with the file:location it relates to.

## flextool/flextool.mod

- Near the existing cumulative-cap constraint — `p_cumulative_ladder_remaining`
  is declared `{c in commodity, i in tier} default 1e30` (NOT constrained
  to `commodity__tier`).  Reason: the CSV reader for this parameter
  keys off the parameter's own domain; restricting it to `commodity__tier`
  would require the domain to already include every `(c, i)` pair
  the reader might emit, and a header-only seed CSV must silently
  load zero rows regardless of domain.  `1e30` sentinel matches the
  commit #2 convention (NOTES_commit2) — constraint predicate uses
  `< 1e29` to treat defaults as inactive.

- Near `table data IN 'CSV' 'solve_data/p_roll_continue_state.csv'`
  (around line 697) — the cumulative handoff CSV is read next to the
  existing per-solve handoffs, matching the placement sketched in
  PLAN_rolling_quota_handoff.md §2.1.

- `ladder_tier_cap_cumulative` RHS swap: `p_commodity_ladder_quantity`
  → `p_cumulative_ladder_remaining`, filter swapped likewise.  First
  solve: empty seed → default 1e30 → constraint inactive, LP
  bit-identical to a single-solve run (verified by
  `test_commodity_ladder_smoke.py::test_objective_matches_coal`
  still passing).  The direct-input `p_commodity_ladder_quantity`
  remains the source of truth for the annual tier cap and is the
  input the Python writer reads — the mod no longer sees it on
  the cumulative path.

- Phase-1 block (near line 4340) — new first-solve printf writes
  `solve_data/cumulative_weight_total.csv` with the scalar
  `total_weight = sum over d in period appearing in dt_complete of
  p_years_represented_d[d] / (sum step_duration over dt_complete
  rows with that d / 8760)`.  **GMPL idiom**: `in setof{} (d)` is
  disallowed as an iteration domain (`implementation restriction;
  in/within setof{} not allowed` — discovered the hard way), so
  the iteration set is expressed as `{d2 in period : exists{(d2, tt)
  in dt_complete} 1}` which parses cleanly.  The per-period share
  is computed inline via `8760 / sum complete_step_duration` to
  avoid depending on the `complete_period_share_of_year[d]` param's
  `period_in_use` domain (which doesn't cover periods only in
  `dt_complete` on rolling solves).

- No `pdt_branch_weight` in the total-weight or the span-weight:
  stochastic-branch weighting is deferred (per task brief §G
  guardrails).  When stochastic support lands, the mirror formulas
  in the writer and the mod printf must be updated together.

## flextool/process_outputs/cumulative_handoffs.py

- `complete_period_share_of_year[d]` is loaded from
  `solve_data/complete_period_share_of_year.csv` (already written by
  the mod's phase-1 block) rather than recomputed from
  `steps_complete_solve.csv` — keeps the writer symmetric with the
  mod's constraint LHS and avoids duplicating the 8760-normalisation
  logic.

- v_trade extraction uses `has_time=False, trailing_col_names=("tier",)`
  matching the VariableSpec in `read_highs_solution.py` from commit #3.
  The row index is `(solve, period)`, the column tuple is
  `(commodity, node, tier)`.  Tier coerced to `int` on read —
  the MPS emits tiers as string tokens and the output CSV uses ints
  for determinism with `sorted()`.

- `_span_weight` guards against `share <= 0.0` and `None` — defensive
  against malformed per-period share CSVs (seen when a scenario
  doesn't exercise all periods in `period__branch`).  Degenerate
  periods silently contribute 0 to the weight.

- Header uses the parameter name (`p_cumulative_ladder_remaining`)
  rather than a CSV-friendly label (`remaining`).  Reason: the mod's
  `table data IN 'CSV' ... : [commodity, tier], p_cumulative_ladder_remaining;`
  reads by header match; keeping the header identical avoids the
  `~old_name` renaming dance.

- No nested-solve guard (no `_is_outermost_realizing_solve(...)`
  predicate).  Per project-memory "Nested rolling … deferred" and
  task brief §G.  Under nested, sibling sub-solves would race on
  the CSV — acceptable for now; document as known limitation in
  follow-up.

## flextool/flextoolrunner/solve_writers.py

- `write_empty_cumulative_files` uses `mkdir(exist_ok=True)` for
  `solve_data/` — matches the defensive pattern already in
  `write_empty_storage_fix_file`'s siblings.  The orchestration
  call-site always precedes solve-data population but mkdir costs
  nothing.

## tests/test_cumulative_handoffs.py

- Fifteen unit tests, no solver invocation.  The end-to-end
  two-roll validation is step 4e (separate commit).  Test coverage
  spans: finite/infinite tier filtering, method filtering, first-
  solve prior=0, second-solve prior carry, overspend → negative,
  empty cumulative commodity list (header-only output), missing
  total_weight CSV (header-only output), multi-node per-tier sum.

- `_fake_highs` mirrors the pattern in `test_handoff_writers.py` —
  same SimpleNamespace wrappers so the `extract_variable` code path
  reading `col_value` / `col_dual` / `row_dual` just works.
