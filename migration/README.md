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
| `baselines/mps_baseline.json` | Reference MPS hash on the H2 trade `scenario_test_6h_no_carrier_storage` fixture. |

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
4. Run validation gate:
   ```bash
   python -m flextool.cli.cmd_run_flextool <fixture> --glpsol-timing
   python -m migration.mps_parity check <work>/flextool.mps
   ```
   Must report `OK: MPS structurally identical to baseline`.
5. Commit on `python-preprocessing` with a message of the form
   `migration L<N>: <set-name> → Python` and include the per-constraint
   timing delta (if any) in the body.

## Phase gates

After each layer (L0 → L11) completes:

- Full `tests/test_scenarios.py` (5 min)
- Full `tests/test_blocks.py` + `test_bind_intraperiod_blocks.py` +
  `test_lh2_three_region.py` + `test_co2_rolling_handoff.py` (~3 min)
- MPS parity on the H2 trade fixture and at least one additional
  scenario from a different complexity class (multi-block, rolling).

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
