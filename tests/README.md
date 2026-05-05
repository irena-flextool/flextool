# FlexTool test suite

The FlexTool tests are organized into four layers, each catching a
different class of regression. Read the layer summaries below before
adding a new test — picking the right layer keeps the fast paths fast
and makes failures easier to diagnose.

| Layer | Driver | What it catches | Speed |
|-------|--------|-----------------|-------|
| 1 | `test_scenarios.py` (60 scenarios) | CSV output drift on small focused models | minutes (full sweep) |
| 2 | `test_mps_parity.py` (5 baselines) | MPS structural changes (rows/cols/coefs/bounds) | 5-60s per case |
| 3 | feature-specific files | Behavior of one feature with its own fixture | seconds to minutes |
| 4 | unit tests (no solver) | Writer / reader / utility logic | < 1s each |

The data flow that wires fixtures to layers is at the bottom
([Data flow](#data-flow)).

---

## Layer 1: scenario regression suite

End-to-end runs of small focused model scenarios. Each scenario locks a
specific FlexTool feature — storage, reserves, transmission, etc. —
from DB read through LP solve to post-processed CSV output, and the
test fails if any value in the selected CSVs drifts.

- **Driver:** `tests/test_scenarios.py`
- **Manifest:** `tests/scenarios.yaml` (60 scenarios as of this writing)
- **Goldens:** `tests/expected/<scenario>/`
- **Source DB:** `tests/fixtures/tests.json` → `tests.sqlite` via
  `tests/db_utils.py:json_to_db` (rebuilt fresh per session by the
  `test_db_url` fixture in `conftest.py`)
- **Solver:** real glpsol + HiGHS, with `tests/highs.opt` overriding
  `bin/highs.opt` for tighter tolerances and `threads=1` (deterministic)

### Running

```bash
# All 60 scenarios
pytest tests/test_scenarios.py

# One specific scenario (exact match — -k does substring matching)
pytest "tests/test_scenarios.py::test_scenario[coal]"

# Verbose, shows scenario names
pytest tests/test_scenarios.py -v

# Stop on first failure
pytest tests/test_scenarios.py -x
```

CSVs are rounded to 4 decimal places before comparison; that absorbs
sub-tolerance numerical noise across HiGHS versions while still
catching meaningful drift.

### Regenerating goldens

Use `--regenerate <scenario>` only when you have intentionally changed
model behaviour and need to update the expected outputs.

```bash
pytest tests/test_scenarios.py --regenerate coal
```

This runs **only** the `coal` test, writes the new CSVs to
`tests/expected/coal/`, and marks the test as skipped (not passed).
Inspect the diff before committing — regenerating silently overwrites
the previous golden files. To regenerate several:

```bash
for s in coal wind_battery coal_co2_price; do
    pytest tests/test_scenarios.py --regenerate $s
done
```

### Adding a new scenario

1. Open `tests/tests.sqlite` in Spine Toolbox and add the scenario.
2. Re-export the JSON fixture:
   `python tests/db_utils.py export tests/tests.sqlite tests/fixtures/tests.json`
3. Add an entry to `tests/scenarios.yaml` (2-3 CSVs from
   `templates/default_plots.yaml` that exercise the feature).
4. Generate goldens: `pytest tests/test_scenarios.py --regenerate my_new_scenario`
5. Inspect `tests/expected/my_new_scenario/` and commit `tests.json`
   and the new goldens together.

### Optional scenario-entry fields

A scenario entry in `tests/scenarios.yaml` can carry up to three
optional fields beyond the required `scenario` and `csvs`:

| Field | Default | Purpose |
|---|---|---|
| `smoke: true` | (absent) | Includes the scenario in `pytest -m smoke` — the per-commit fast gate. About 5 short scenarios are marked. |
| `expected_objective: <number>` | (no check) | Hand-derived total system cost (in M CUR) read from `summary_solve.csv`. About 10 scenarios where the optimum is easy to verify. Comment the derivation above the field so future edits can re-check. |
| `expected_objective_tolerance: <relative>` | `1.0e-3` | Relative tolerance for the objective check. Use `1.0e-5` for slack-only or fully-deterministic dispatch; relax for scenarios where penalty variables make the objective sensitive to small slack shifts. |
| `time_budget_seconds: <wall>` | (no check) | Asserts that the wrapped block (`runner.write_input` + `run_model` + `write_outputs`) completes within this wall-clock budget. Set to `~1.5 ×` the observed maximum on a representative machine, rounded up to 0.5 s. About 60 scenarios have budgets; the long multi-roll / nested ones are skipped because their wall-clock variance defeats the assertion. |

A failing timing assertion looks like:

```
AssertionError: timing regression: scenario=coal observed=6.34s budget=4.00s
(set in tests/scenarios.yaml; bump if the increase is intended)
```

Bump the budget when the regression is intentional (e.g. you knowingly
added preprocessing work). If you don't recognise the cause, treat it
as a real regression — the budgets are set with 50 % slack so they
shouldn't trip on noise.

### DB fixture maintenance

`tests/tests.sqlite` is the editable Spine DB (gitignored).
`tests/fixtures/tests.json` is its JSON export (committed). After
editing the DB, run `python tests/db_utils.py export ...`; after
pulling someone else's changes, run `python tests/db_utils.py import
...`. The test session always rebuilds a throwaway sqlite from the
JSON, so `tests.sqlite` is only needed when editing in Spine Toolbox.

---

## Layer 2: MPS parity baselines

Layer 1 catches CSV numeric drift but a refactor can produce
identical CSVs from a structurally different MPS — e.g. dropping a
redundant constraint, splitting a variable, or reordering blocks. The
parity layer locks the matrix shape itself.

- **Driver:** `tests/test_mps_parity.py`
- **Baselines:** `migration/baselines/*.json` (5 of them)
- **Source DBs:** 4 from `tests.sqlite` + 1 from a dedicated
  `h2_parity.sqlite` built from `tests/fixtures/h2_trade_parity.json`
- **What's asserted:** `migration.mps_parity.canonical_hash` of the
  produced `flextool.mps` matches the baseline hash. The canonical
  form is order-independent over rows / columns / RHS / bounds, with
  coefficients quantized to 7 significant figures (see the docstring
  on `migration/mps_parity.py:_f64_hex`).

| Baseline | Scenario | What it stress-tests |
|----------|----------|----------------------|
| `test_a_lot_baseline.json` | `test_a_lot` | broad-coverage scenario hitting many features at once |
| `fullYear_roll_baseline.json` | `fullYear_roll` | rolling-horizon orchestration over a full year |
| `5weeks_invest_fullYear_dispatch_coal_wind_baseline.json` | `5weeks_invest_fullYear_dispatch_coal_wind` | mixed invest+dispatch nesting (5w invest layer ⊂ full-year dispatch) |
| `multi_fullYear_battery_nested_24h_invest_one_solve_baseline.json` | `multi_fullYear_battery_nested_24h_invest_one_solve` | multi-year + 24h nested block + battery state |
| `h2_trade_baseline.json` | `scenario_test_6h_no_carrier_storage` | commodity ladder + indirect conversion (h2 trade slice) |

### Layer-1 vs Layer-2 — different bug classes

Four of the five parity scenarios (`test_a_lot`, `fullYear_roll`,
`5weeks_invest_fullYear_dispatch_coal_wind`,
`multi_fullYear_battery_nested_24h_invest_one_solve`) are also Layer-1
scenarios. **This overlap is intentional.** A change can pass Layer 1
(numerically equivalent CSVs) and fail Layer 2 (different MPS), or
vice versa:

- *Layer-1-only fail*: a writer rounds output CSVs differently —
  matrix is unchanged but the comparison fails.
- *Layer-2-only fail*: a redundant constraint is dropped — the LP
  optimum is unchanged so CSVs still match, but the MPS shrinks.

The parity layer was added during the python-preprocessing migration
to catch the second class without hand-diffing 14k-coefficient MPS
files.

### Running

```bash
pytest tests/test_mps_parity.py -v
```

Each case runs a real solver in an isolated work dir (5-60s). The
suite is marked `@pytest.mark.solver`, so:

```bash
# Skip parity tests in fast paths
pytest -m "not solver"

# Only parity tests
pytest -m solver
```

### Regenerating a baseline

When the migration intentionally changes the MPS (e.g. a Python
preprocessor replaces a `.mod` derivation), re-baseline the affected
case:

```bash
# 1. Run the scenario manually, collect flextool.mps
python -m flextool.cli.cmd_run_flextool sqlite:////absolute/path/to/tests.sqlite \
    --scenario-name fullYear_roll --work-folder /tmp/parity-out --highs-threads 1

# 2. Update the baseline summary
python -m migration.mps_parity baseline /tmp/parity-out/flextool.mps \
    --out migration/baselines/fullYear_roll_baseline.json
```

Commit the updated `*_baseline.json` with a message that explains
what changed in the MPS (rows? cols? coefficients?) and why it's
expected.

---

## Layer 3: feature-specific solver integration tests

These run a real solver but with a unique DB / fixture / setup that
doesn't fit cleanly into the Layer-1 manifest. One file per feature:

| File | Pins |
|------|------|
| `test_lh2_three_region.py` | flex-temporal stack on the 3-region LH2 fixture (hourly + daily blocks coexist) |
| `test_commodity_ladder_smoke.py` | commodity-ladder mechanism (price tiers, indirect conversion) — single-solve |
| `test_commodity_ladder_rolling.py` | commodity ladder under rolling-horizon orchestration |
| `test_cost_aggregation_semantics.py` | post-process cost aggregations match the LP objective |
| `test_years_represented.py` | inflation-factor scaling (years_represented) — kept for `solve_data/p_years_represented.csv` row-pattern assertions; the cost-scaling part was promoted to Layer-1 scenarios `years_represented_half` + `years_represented_2_5` |
| `test_regional_filter.py` | Lagrangian regional decomposition export (`--region`) |
| `test_xlsx_workflow.py` | full xlsx → DB → solve pipeline (slow, multi-minute) |
| `test_read_highs_solution.py` | HiGHS solution-file reader |

`test_hyphen_end_to_end.py` and `test_vq_penalties.py` were promoted to Layer-1
scenarios (`hyphenated_entity_names`, `unidirectional_connection`,
`years_represented_half`, `years_represented_2_5`) and deleted.  See
`tests/scenarios.yaml` for the new entries.

Some of these are candidates for promotion to Layer 1 — if the test
boils down to "scenario X produces CSV Y", it's cheaper as a
`scenarios.yaml` entry. That promotion is tracked separately.

### Running

```bash
# One file
pytest tests/test_lh2_three_region.py -v

# Several
pytest tests/test_lh2_three_region.py tests/test_hyphen_end_to_end.py
```

---

## Layer 4: pure unit tests (no solver invocation)

Mock-based; cover writers / readers / utilities. Fast (< 1s each).
No solver, no DB build — these are the smoke gate.

Roughly grouped:

- **Handoff / cumulative writers:** `test_handoff_writers.py`,
  `test_cumulative_handoffs.py`, `test_co2_rolling_handoff.py`,
  `test_drop_levels_rolling.py`
- **Scaling:** `test_scaling.py`, `test_scaling_report.py`
- **Plot pipeline:** `test_plot_pipeline.py`,
  `test_plot_plan_chunk_reconstruction.py`,
  `test_plot_plan_time_labels_roundtrip.py`,
  `test_color_template.py`, `test_shared_axis_manifest.py`,
  `test_bar_layout.py`, `test_flowgroup_indicators.py`
- **Parity tool itself:** `test_mps_parity_harness.py` (unit tests
  for `migration.mps_parity` on synthetic MPS fixtures)
- **Other utilities:** `test_blocks.py`, `test_precision.py`,
  `test_solve_config.py`, `test_solver_options.py`,
  `test_highs_handle.py`, `test_ensure_settings_db.py`,
  `test_export_to_tabular.py`, `test_self_describing_reader.py`,
  `test_comparison_excel.py`, `test_plan_union.py`,
  `test_parameter_groups_coverage.py`, `test_group_output_warnings.py`,
  `test_preprocessing_ordered_set_lint.py`,
  `test_timeset_weights.py`, `test_bind_intraperiod_blocks.py`,
  `test_dc_power_flow.py`, `test_unidirectional_connection.py`,
  `test_mod_varcost_node_as_process.py`,
  `test_representative_periods.py` (some `@pytest.mark.slow`),
  `test_gui_startup.py` (GUI-only smoke)

### Running

```bash
# Layer 4 = everything not slow and not solver-dependent
pytest -m "not slow and not solver"
```

---

## Data flow

```
                                                     +---- 60 scenarios (Layer 1)
                                                     |
tests/fixtures/tests.json --> tests.sqlite ----------+---- 4 of 5 parity baselines (Layer 2)
   (committed JSON)         (built per session,      |
                            test_db_url fixture)     +---- many Layer-3 tests
                                                          (commodity_ladder_*, years_represented,
                                                           vq_penalties, hyphen_end_to_end, ...)


tests/fixtures/h2_trade_parity.json --> h2_parity.sqlite ---- 1 parity baseline (Layer 2)
   (committed JSON)                  (built per session,
                                     h2_parity_db_url fixture)


tests/fixtures/build_lh2_three_region.py --> three_region.sqlite ---- test_lh2_three_region.py (Layer 3)
   (Python builder, no JSON yet)            (built per session,
                                            lh2_db_url fixture)


templates/example_input_template.xlsx ---------------------- test_xlsx_workflow.py (Layer 3)
   (committed xlsx)                                          test_self_describing_reader.py (Layer 4)


synthetic in-memory DBs / mocks ---------------------------- everything else in Layer 3, all of Layer 4
```

Fixture wiring is in `tests/conftest.py` (session-scoped `test_db_url`
+ `test_bin_dir`) and in the per-test files for layer-specific
fixtures.

---

## Running each layer

```bash
# Layer 1 (minutes)
pytest tests/test_scenarios.py

# Layer 2 (5-60s per case, 5 cases)
pytest tests/test_mps_parity.py

# Layer 3 selected
pytest tests/test_lh2_three_region.py tests/test_xlsx_workflow.py \
       tests/test_commodity_ladder_smoke.py tests/test_regional_filter.py

# Layer 4 (fast — the smoke gate)
pytest -m "not slow and not solver"
```

Layer 1 is unmarked (the scenario sweep is fast individually but slow
in aggregate); to exclude it from a smoke run use a path filter
(`--ignore=tests/test_scenarios.py`).

---

## Where does a new test go?

A short decision tree:

1. **Is it a small variation of an existing scenario** that just needs
   different DB values? → add a `tests/scenarios.yaml` entry
   (Layer 1).
2. **Does it need a fixture or setup that doesn't fit `tests.json`**
   (custom DB builder, xlsx file, multi-step setup, dedicated
   alternative stack)? → new test file in `tests/` (Layer 3).
3. **Is it a pure module unit test** with mocks instead of a solver?
   → new test file in `tests/`, no solver call (Layer 4).
4. **Does it need to lock the MPS structure**, not just numeric
   outputs (e.g. catch a refactor that drops a redundant constraint
   without changing the optimum)? → new entry in
   `tests/test_mps_parity.py::_CASES` + a baseline JSON in
   `migration/baselines/` (Layer 2).

When unsure, prefer Layer 1 — adding a row to `scenarios.yaml` is
cheaper than another bespoke file, and the layer-3 footprint is
already large.

---

## Solver precision

`tests/highs.opt` overrides `bin/highs.opt` during test runs:

- `threads=1` for deterministic results
- `primal/dual_feasibility_tolerance=1e-8` (tighter than production)
- `mip_rel_gap=0.0001` (near-exact for small test problems)

Output CSVs are rounded to 4 decimal places before comparison
(`tests/db_utils.py:round_for_comparison`); MPS coefficients are
compared at 7 significant figures (`migration/mps_parity.py:_f64_hex`).

For known-failure tracking after a refactor, see
`tests/KNOWN_FAILURES_LP_SCALING.md`.
