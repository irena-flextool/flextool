# python-preprocessing migration

Tooling and artifacts for moving derived sets and calculated parameters
from `flextool.mod` (MathProg) into Python preprocessing.

This branch (`python-preprocessing`) is **frozen** against `new-outputs`
for the duration of the migration. No upstream rebases until the
migration completes.

## Phase 0 artifacts (this directory)

| Artifact | Purpose |
|---|---|
| `mps_parity.py` | Order-independent MPS structural comparison. Validation gate for every step. |
| `inventory_mathprog_derivations.py` | Cataloger of every `set` / `param` derivation in `flextool.mod`. |
| `inventory.csv` | Output of the cataloger. 351 in-scope items. |
| `build_dag.py` | Dependency DAG over the inventory + topo-sorted layered ordering. |
| `dag.json` | Adjacency + reverse adjacency for the in-scope items. |
| `order.txt` | Layered migration order — one item per line, blank lines between layers. |
| `schema_defaults_audit.py` | Schema vs. `.mod` `default ...` clause cross-reference. |
| `schema_defaults.csv` | 225 schema params + 68 mod-side defaults. |
| `baselines/h2_trade_baseline.json` | Reference MPS hash on the H2 trade `scenario_test_6h_no_carrier_storage` fixture (commodity ladder + indirect conversion; ~2.3k rows). |
| `baselines/test_a_lot_baseline.json` | Complementary reference MPS hash on `tests/fixtures/tests.sqlite::test_a_lot` (UC/online_linear, reserves, capacity margin, inertia, co2 limit, profile_state, storage_state_start_binding, min_load; ~5.3k rows). |
| `baselines/fullYear_roll_baseline.json` | Solve-loop coverage: rolling-window solve on `tests/fixtures/tests.sqlite::fullYear_roll` (`solve_mode=rolling_window`, horizon=8h, jump=4h, full year ⇒ many rolls). Captures the LAST roll's MPS (~41 rows). |
| `baselines/5weeks_invest_fullYear_dispatch_coal_wind_baseline.json` | Solve-loop coverage: multi-solve via `model.solves` on `tests/fixtures/tests.sqlite::5weeks_invest_fullYear_dispatch_coal_wind` (`model.solves=[invest_1year_5weeks, y2020_fullYear_dispatch]`, two single-solves back-to-back, no `contains_solve`). Captures the LAST solve's MPS (~433 rows). |
| `baselines/multi_fullYear_battery_nested_24h_invest_one_solve_baseline.json` | Solve-loop coverage: nested `contains_solves` chain on `tests/fixtures/tests.sqlite::multi_fullYear_battery_nested_24h_invest_one_solve` (3-level chain `invest_24h → storage_fullYear_6h → dispatch_fullYear_roll`, with the leaf being `rolling_window`). Captures the LAST roll's MPS at the deepest nesting level (~54 rows). |

The lint that bans bare `set()` in preprocessing modules lives at
`tests/test_preprocessing_ordered_set_lint.py` — it is a no-op until
the first preprocessing module lands.

## Per-step workflow

For each item in `order.txt` (strict sequential, layer by layer):

1. Implement the computation in
   `flextool/flextoolrunner/preprocessing/<family>.py`. Use
   `dict.fromkeys(iterable)` for ordered deduplicated containers.
   Never bare `set()`. The lint test enforces this.
2. Wire into `input_writer.py` — call the new function, write a CSV in
   `solve_data/<set_name>.csv` with header columns matching the
   MathProg dimensions.
3. Replace the `:= setof {...}` body in `flextool.mod` with a
   data-loaded declaration (`set X dimen N;` or `param X{...};`) and
   add a `table data IN` reader matching the new CSV.
4. Run validation gate against ALL FIVE baselines. The H2 trade and
   `test_a_lot` baselines cover write-input-scope structure (commodity
   ladder, indirect conversion, UC, reserves, capacity margin, inertia,
   co2 limit, profile_state, min_load). The three solve-loop baselines
   add coverage for `orchestration.py`'s rolling / multi-solve /
   nested-solve code paths that the first two never exercise:
   ```bash
   # H2 trade — write-input-scope, single solve
   python -m flextool.cli.cmd_run_flextool <h2_fixture>
   python -m migration.mps_parity check <h2_work>/flextool.mps \
       --baseline migration/baselines/h2_trade_baseline.json

   # test_a_lot — wide structural coverage, single solve
   python -m flextool.cli.cmd_run_flextool <test_a_lot_fixture>
   python -m migration.mps_parity check <test_a_lot_work>/flextool.mps \
       --baseline migration/baselines/test_a_lot_baseline.json

   # fullYear_roll — rolling_window solve, many rolls
   python -m flextool.cli.cmd_run_flextool sqlite:////tmp/parity-coverage/tests.sqlite \
       --scenario-name fullYear_roll \
       --work-folder /tmp/parity-coverage/fullYear_roll \
       --output-location /tmp/parity-coverage --output-subdir fullYear_roll \
       --write-methods csv --highs-threads 1
   python -m migration.mps_parity check /tmp/parity-coverage/fullYear_roll/flextool.mps \
       --baseline migration/baselines/fullYear_roll_baseline.json

   # 5weeks_invest_fullYear_dispatch_coal_wind — multi-solve via model.solves
   python -m flextool.cli.cmd_run_flextool sqlite:////tmp/parity-coverage/tests.sqlite \
       --scenario-name 5weeks_invest_fullYear_dispatch_coal_wind \
       --work-folder /tmp/parity-coverage/5weeks_invest_fullYear_dispatch_coal_wind \
       --output-location /tmp/parity-coverage \
       --output-subdir 5weeks_invest_fullYear_dispatch_coal_wind \
       --write-methods csv --highs-threads 1
   python -m migration.mps_parity check \
       /tmp/parity-coverage/5weeks_invest_fullYear_dispatch_coal_wind/flextool.mps \
       --baseline migration/baselines/5weeks_invest_fullYear_dispatch_coal_wind_baseline.json

   # multi_fullYear_battery_nested_24h_invest_one_solve — contains_solve chain
   python -m flextool.cli.cmd_run_flextool sqlite:////tmp/parity-coverage/tests.sqlite \
       --scenario-name multi_fullYear_battery_nested_24h_invest_one_solve \
       --work-folder /tmp/parity-coverage/multi_fullYear_battery_nested_24h_invest_one_solve \
       --output-location /tmp/parity-coverage \
       --output-subdir multi_fullYear_battery_nested_24h_invest_one_solve \
       --write-methods csv --highs-threads 1
   python -m migration.mps_parity check \
       /tmp/parity-coverage/multi_fullYear_battery_nested_24h_invest_one_solve/flextool.mps \
       --baseline migration/baselines/multi_fullYear_battery_nested_24h_invest_one_solve_baseline.json
   ```
   All five must report `OK: MPS structurally identical to baseline`.

   **Known limitation — rolling and multi-solve fixtures capture the
   LAST MPS only.** `flextool.mps` is overwritten per roll and per solve
   inside `orchestration.py`; only the final roll's / final solve's
   matrix is on disk when the run finishes. The three solve-loop
   baselines therefore pin:
   - `fullYear_roll`: the last roll of `dispatch_fullYear_roll`
   - `5weeks_invest_fullYear_dispatch_coal_wind`: the second solve
     (`y2020_fullYear_dispatch`)
   - `multi_fullYear_battery_nested_24h_invest_one_solve`: the last
     roll of the deepest child (`dispatch_fullYear_roll`) at the leaf
     of the `invest_24h → storage_fullYear_6h → dispatch_fullYear_roll`
     contains-chain.

   This is sufficient to catch any preprocessing change that perturbs
   the per-roll/per-solve MPS structure deterministically (set
   ordering, coefficient values, row/column count) but NOT changes
   that affect only intermediate rolls — e.g., a regression in
   `solve_period_history` carryover between rolls would not surface
   if the last-roll matrix happens to coincide. Agents touching
   solve-loop code paths (rolling jump arithmetic, contains-solve
   recursion, per-solve refresh in `preprocessing/solve_time.py`)
   should additionally run the relevant `tests/test_*_rolling*.py`
   unit tests and a spot-check scenario with intermediate-roll
   inspection.
5. Commit on `python-preprocessing` with a message of the form
   `migration L<N>: <set-name> → Python` and include the per-constraint
   timing delta (if any) in the body.

## Phase gates

After each layer (L0 → L11) completes:

- Full `tests/test_scenarios.py` (5 min)
- Full `tests/test_blocks.py` + `test_bind_intraperiod_blocks.py` +
  `test_lh2_three_region.py` + `test_co2_rolling_handoff.py` (~3 min)
- MPS parity on ALL FIVE fixtures: H2 trade (`h2_trade_baseline.json`),
  `test_a_lot` (`test_a_lot_baseline.json`), `fullYear_roll`
  (`fullYear_roll_baseline.json`),
  `5weeks_invest_fullYear_dispatch_coal_wind`
  (`5weeks_invest_fullYear_dispatch_coal_wind_baseline.json`), and
  `multi_fullYear_battery_nested_24h_invest_one_solve`
  (`multi_fullYear_battery_nested_24h_invest_one_solve_baseline.json`).
  Together these cover:
  - commodity-ladder / indirect-conversion (h2_trade)
  - UC/reserves/capacity-margin/inertia/co2/profile_state/min_load
    (test_a_lot)
  - rolling-window solve loop (fullYear_roll)
  - multi-solve via `model.solves` chain (5weeks_invest_…)
  - nested `contains_solves` chain ending in a rolling leaf
    (multi_fullYear_battery_nested_24h_invest_one_solve)

  Still uncovered (so still potential blind spots after layer gates):
  `nodeBalanceBlock_eq` / `stateConstantWithinBlock_eq` (intraperiod
  blocks), integer UC variants (`online__startup_integer` /
  `online__shutdown_integer` / `non_anticipativity_online_integer`),
  `minimum_uptime` / `minimum_downtime`, ramp constraints
  (`ramp_sink_up_constraint` / `ramp_up_variable`), divest, and
  intermediate rolls (the rolling/nested baselines pin only the LAST
  roll's MPS — see workflow note above). Spot-check one of
  `5weeks_battery_intraperiod_blocks`, `coal_wind_min_uptime_MIP`, or
  `coal_ramp_limit` per layer if its sets/params plausibly affect
  those families.

A failed gate halts the migration until investigated.

## Phase 1 (defaults consolidation)

Before Phase 2 (the per-set migration) begins, the 68 mod-side
`default Y;` clauses are moved to the schema.

Cross-referencing schema params to mod params is **not automated** —
each Phase 1 step's owner reads `input_writer.py` to determine which
schema row populates which mod param table. The audit CSV
(`schema_defaults.csv`) lists both sides; the agent decides per row:

- Schema has default + mod has matching default → drop mod default.
- Schema lacks default + mod has default → add schema default via
  `flextool/update_flextool/db_migration.py` (one new `next_version`
  block per change), regenerate the master template via
  `python -m flextool.update_flextool.sync_master_json_template`, then
  drop mod default.
- Schema and mod defaults disagree → human review required.

See `CONTRIBUTING.md` for migration mechanics (`db_migration.py`,
`__init__.py` version bump, regeneration command).

## Recovering from a failed step

The branch is structured so that every commit is a working state. If
step N's MPS parity check fails:

1. Investigate the diff: `python -m migration.mps_parity diff
   <baseline_mps> <work>/flextool.mps`. The structured diff isolates
   the regression to specific rows / columns / coefficients.
2. Fix the Python preprocessing or revert the step.
3. Do not stack a "correction" commit on top of a broken step. Either
   amend (if not yet pushed) or revert and retry.
