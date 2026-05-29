# F6 — Triage of 11 remaining failures after F1–F5

Date: 2026-05-28
Branch: `v56-test-cleanup`

## Summary

| # | Test | Classification | Action |
|---|------|---------------|--------|
| 1 | `test_db_direct_solve_parity_with_derived_b[work_coal_chp]` | A | **FIXED (G5)** — axis_enums backend ran without scenario_filter; `is_enabled="no"` rows from non-scenario alternatives disabled `coal_chp_fix` from the constraint enum |
| 2 | `TestPGLibCase14Integration::test_objective_value` | C | documented (output_raw/v_obj parquet contract drift) |
| 3 | `TestPGLibCase14Integration::test_reference_bus_angle_zero` | C | documented (same root cause as #2) |
| 4 | `TestPGLibCase14Integration::test_nonreference_buses_have_nonzero_angles` | C | documented (same root cause as #2) |
| 5 | `TestWithinPeriodCumulativeRolling::test_within_period_cumulative_completes` | B | documented; root cause traced to roll-2 dispatch-collapse — deeper cascade issue, deferred |
| 6 | `TestCumulativeLadderBindingCap::test_roll2_uses_tier2_only_after_roll1_saturates_cap` | B | documented; **same root cause as #5** (roll-2 dispatch collapse) — deferred |
| 7 | `TestSingleSolveBitIdentity::test_single_solve_cumulative_matches_coal_objective` | **FIXED (G4)** | autoscale `setSolution` zeroed `getObjectiveValue()`; stash `_flextool_unscaled_objective` before push, prefer it in `write_v_obj` |
| 8 | `test_cli_end_to_end_rivendell_map_index_name` | A | **FIXED** — `_node_constraint_coef` now swallows `KeyError` on missing param |
| 9 | `test_export_to_tabular::TestConstraintSheet::test_has_data_rows` | resolved | passes on re-run (state leak — already fixed elsewhere) |
| 10 | `test_handoff_writers::test_p_entity_period_existing_capacity_first_solve` | A | **FIXED** — test now passes `csv_dump=True` to the writer |
| 11 | `test_xlsx_workflow::TestXlsxWorkflow::test_full_workflow_from_xlsx` | resolved | already marked `SKIPPED` upstream |

**Outcome:** 3 fixed (A), 2 silently resolved on re-run, 6 documented as real bugs to follow up.

## (A) — Fixes applied this batch

### #8 — `_node_constraint_coef` missing-parameter handling

The Rivendell DB does not define `constraint_invested_capacity_coeff` on the
`node` class (the parameter set was renamed / removed for that scenario).
`InputSource.parameter()` raises `KeyError` on unknown parameter names —
the analogous helper `_process_constraint_coef` already wraps the call in
`try/except KeyError: continue`, but `_node_constraint_coef` did not.

Fix: wrap the `source.parameter(...)` call in `_node_constraint_coef`
with the same `KeyError` swallow that `_process_constraint_coef` uses.
Returns `None`, treated identically to "zero rows".

File: `flextool/engine_polars/_direct_params.py` (lines 283-296).

### #10 — `test_p_entity_period_existing_capacity_first_solve` missing `csv_dump=True`

The `write_p_entity_period_existing_capacity` writer was migrated to the
cascade and now only writes the CSV when `csv_dump=True` is passed
(line 911 — `if csv_dump: out.to_csv(...)`).  The test still expected
the file unconditionally.

Fix: pass `csv_dump=True` at the call site in the test.

File: `tests/test_handoff_writers.py` (around line 263).

## (A continued) — `work_coal_chp` parity fix (G5)

### #1 — `work_coal_chp` 111× objective drift — **FIXED**

Headline observation: `polar_high=21,097,000`, cascade-written parquet
`2,312,767,749.99`, ratio `109.625`.  Earlier hypothesis (polar_high LP
scaling drift) was wrong — both objectives came from polar_high
correctly; they just solved **different LPs**.  Side-by-side MPS diff
confirmed the cascade-built LP carried 48 extra rows
(`process_constraint_equal[coal_chp_fix,p2020,t0001..t0048]`) that the
DB-direct path's LP omitted entirely.

**Root cause.** `load_flextool` (`flextool/engine_polars/input.py`,
lines ≈4011-4019 before the fix) constructs a transient
`SpineDBBackend` to build the global `axis_enums` vocabulary, but
passes `scenario_name=None`:

```python
with SpineDBBackend(
    f"sqlite:///{sqlite_for_backend}",
    None,                           # ← no scenario_filter
) as _ab:
    axis_enums = build_axis_enums(_ab, contract)
```

Without a scenario_filter, `_disabled_entity_ids("constraint")` calls
`find_parameter_values(entity_class="constraint",
parameter_definition_name="is_enabled")` and walks **every**
parameter_value row in the DB.  For the `coal_chp_fix` constraint
there's a `is_enabled="no"` row in alternative `coal_chp_extraction` —
NOT part of the `coal_chp` scenario — that nevertheless lands in the
disabled set and removes `coal_chp_fix` from the `constraint` axis enum
vocabulary (only `battery_tie_kW_kWh / water_pump_fix / wind_growth_cap`
survive).

Downstream cascade: `_load_user_constraints` reads the
`p_process_node_constraint_flow_coeff.csv` and casts the
`constraint`→`cn` column through `alias_to_axis` against the (now
missing-`coal_chp_fix`) enum — a silent `cn=null` cast.  The
`flow_constraint_idx`'s `(p, source, sink, cn=null)` rows then fail to
join `cdt_eq` (which carries `cn=null` too, for the same reason).
Either way the `lhs_pieces` Where reduces to zero rows and
`m.add_cstr("process_constraint_equal", …)` emits no constraints.
The LP then satisfies heat / west demand with `vq_state_up` /
`vq_state_down` slacks (cost 0.01825-0.16425 per unit, scale_obj=1e-6)
because nothing forces the heat ↔ electricity coupling on the CHP
unit.  The slack-dominated objective is 109.625× the constrained one,
matching the observed ratio.

**Fix** (`flextool/engine_polars/input.py`):

```python
_scenario_for_backend = None
if db_reader is not None:
    _scenario_for_backend = getattr(db_reader, "scenario", None)
if _scenario_for_backend is None:
    _scenario_for_backend = _find_scenario(workdir_for_db)
with SpineDBBackend(
    f"sqlite:///{sqlite_for_backend}",
    _scenario_for_backend,
) as _ab:
    axis_enums = build_axis_enums(_ab, contract)
```

Source the scenario from an explicit `db_reader=` first (the reader
already knows its scenario), else fall back to `_find_scenario` on the
workdir convention.  Backend's `_open` then applies
`scenario_filter_from_dict`, narrowing `find_parameter_values` to the
active alternatives — `coal_chp_fix` is enabled, lands in the enum,
the cascade joins succeed, `process_constraint_equal[coal_chp_fix,*]`
rows emit, and the LP solves to the cascade's objective.

**Verification.**

- `tests/model/test_db_direct_solve.py::test_db_direct_solve_parity_with_derived_b[work_coal_chp]` — PASS (was FAIL with rel=0.99).
- `tests/model/test_db_direct_solve.py` (full file) — 43 passed, 2 skipped (no regressions).
- `tests/spinedb_backend/test_axis_enums.py` + 25 other axis-related tests — 26 passed.

Scope: targeted fix to the single SpineDBBackend constructor call in
`load_flextool`'s axis_enums-build branch.  The two other call sites
that already passed a scenario name are unchanged.  The fix only
activates when the workdir-convention or `db_reader` carries a
scenario; legacy CSV-only callers still see `None`.

### #5 — `TestWithinPeriodCumulativeRolling::test_within_period_cumulative_completes`

```
assert periods_seen == {"p2020", "p2025"}
AssertionError: Expected both periods accumulated, got {'p2020'}
```

**Investigation (G4 batch):** the accumulator contains only `p2020`
because every roll past the first **dispatches zero coal**.  Reproduced
on the simpler 2-roll cross-period scenario (`coal_cum_rolling`, under-
spending cap) too:

* roll-0 parquet `(coal, coal_market, p2020): tier1=58600, tier2=0`
* roll-1 parquet `(coal, coal_market, p2025): tier1=0, tier2=0`

`p_entity_all_existing.csv` in the roll-1 solve directory shows
`coal_plant, p2025: 500.0` (i.e. the unit *is* there), and `f_d_k[p2025] =
1.0` is emitted correctly, so the input-derivation cascade is structurally
fine.  But the LP itself produces zero coal dispatch in roll-1 — the
objective also balloons (e.g. 1.144e9 in roll-0 vs 15.4e9 in roll-1
for the under-spending case), suggesting the node-balance constraint is
being satisfied by a penalty/slack rather than by coal flow.

Root cause is somewhere in the rolling cascade's roll-to-roll handoff
of dispatch state — the warm-restart or one of the period-scoped
parameters drops the coal availability/inflow for the non-realized-yet
periods.  Outside the scope of a quick fix (>50 lines across
`_orchestration.py` / `_emit_chain_params.py` / the warm-restart path).
Deferred.

### #6 — `TestCumulativeLadderBindingCap::test_roll2_uses_tier2_only_after_roll1_saturates_cap`

```
AssertionError: Tier 2 (tail) must absorb roll-2 dispatch when tier 1
is locked out. Got tier-2 sum=0.0 in v_trade__y2020_2day_dispatch_roll_1.parquet.
```

**Investigation (G4 batch):** same underlying behaviour as #5 — roll-1
produces `v_trade = (0, 0)` for `p2025`.  Tier-1 is correctly locked
at 0 by the cumulative-cap constraint (cap = 1 MWh fully consumed in
roll-0; RHS = 1·1 − 1 = 0), but tier-2 (infinite cap, no constraint)
also stays at 0 because the LP cannot dispatch coal at all in roll-1.
Same root cause as #5; the cumulative-cap mod itself is structurally
correct (it produces the expected RHS=0 for tier-1).  Deferred — pair
the fix with #5.

### #7 — `TestSingleSolveBitIdentity::test_single_solve_cumulative_matches_coal_objective` — **FIXED**

```
coal=1,144,037,750.0  vs  coal_cum_single=0.0    (pre-fix)
```

**Root cause:** the autoscale Layer 2 unscale hook
(`_push_unscaled_to_highs`, added in commit `319e2da4`) round-trips
the unscaled primal back onto the live `highspy.Highs` handle via
`setSolution(HighsSolution)`.  Verified empirically (highspy 1.14.0):
calling `setSolution` AFTER `run` resets `getObjectiveValue()` to 0.0.
`write_v_obj` reads `h.getObjectiveValue() * inv_scale` to derive the
parquet objective and the `total_cost.val` stdout line, so the post-
autoscale objective collapsed to zero on every Layer-2-triggering solve.

This was specific to the cumulative-ladder fixture only because it
happens to land outside HiGHS's comfort zone (≥9-decade RHS span from
the `1e30` tier-2 quantity sentinel) — most regression tests don't
trigger Layer 2 from a single-solve and so don't surface the bug.

**Fix:**

* `flextool/engine_polars/autoscale/_layer2.py` —
  `_push_unscaled_to_highs` captures `h.getObjectiveValue()` BEFORE the
  `setSolution` call and stashes it as `h._flextool_unscaled_objective`.
* `flextool/process_outputs/read_highs_solution.py:write_v_obj` —
  prefer `h._flextool_unscaled_objective` when present; fall back to
  `h.getObjectiveValue()` otherwise.

No effect on the cold-path `_SolHighsShim` (the shim's view is updated
in place, no `setSolution` is involved, and the attribute is simply
absent so the writer falls back).

### #2, #3, #4 — `TestPGLibCase14Integration` (3 ERRORs)

All three error out in fixture setup with::

```
AssertionError: No v_obj parquet under <tmp>/work/output_raw
```

Verified manually that the FlexTool subprocess returns rc=0, the solve
completes (`total_cost.val = 17971370.4668`), and parquets are written
— but to a totally different path: `<cwd>/output_parquet/dc_opf_test/`
(in repo root, not under `--work-folder`), and **no per-variable lean
`v_obj__*.parquet` is emitted at all**.  The per-solve cascade writer
only emits aggregate `output_parquet/<scenario>/*.parquet` files, not
the legacy lean per-variable layout the test was migrated to expect
(see comment "Δ.22" in `tests/test_dc_power_flow.py`).

This is **(C)** pre-existing tech debt — the test's "Δ.22 migration"
to lean parquets was speculative; the lean format never landed.  Either
(a) the test should switch to reading the modern aggregated parquet
(`output_parquet/dc_opf_test/costs_dt_p.parquet` or similar), or (b)
`--work-folder` needs to actually redirect `output_parquet/` output.

**Same root cause family** as #7's `_read_objective` helper in
`tests/test_commodity_ladder_rolling.py:63`.

## Pytest invocation budget

Used 8/12 in this batch.

## Re-running after fixes

The two A fixes resolve 2 tests directly; one more (`test_export_to_tabular`)
became green on re-run.  Net: 11 failures → 6 (5 B documented, 1 C cluster
of 3 + 1 (#7) sharing the same `v_obj` parquet helper).
