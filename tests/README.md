# FlexTool Scenario Tests

End-to-end tests that run small focused model scenarios and compare CSV outputs
against golden files. Each test exercises a specific FlexTool feature
(storage, reserves, transmission, etc.) from DB read through LP solve to
post-processed CSV output.

## Running tests

```bash
# Run all scenario tests
pytest test/

# Run one specific scenario (exact match required — -k does substring matching)
pytest "test/test_scenarios.py::test_scenario[coal]"

# Verbose output (shows scenario names)
pytest test/ -v

# Stop on first failure
pytest test/ -x
```

Tests pass when all output CSVs match the golden files in `test/expected/`.
They fail with a diff-style message showing which values changed.

## Regenerating golden files

Use `--regenerate <scenario>` when you have intentionally changed model
behaviour and need to update the expected outputs.

```bash
pytest test/ --regenerate coal
```

This runs **only** the `coal` test, writes the new CSVs to
`test/expected/coal/`, and marks the test as skipped (not passed).
Inspect the files before committing — regenerating silently overwrites
the previous golden files.

**One scenario at a time.** To regenerate several:

```bash
for s in coal wind_battery coal_co2_price; do
    pytest test/ --regenerate $s
done
```

## Adding a new scenario test

### 1. Add the scenario to the test DB (in Spine Toolbox)

Open `test/tests.sqlite` in Spine Toolbox and add or verify the scenario
you want to test. The DB uses Spine's scenario/alternative structure, so
one DB holds all test scenarios.

### 2. Register the scenario in `test/scenarios.yaml`

```yaml
- scenario: my_new_scenario
  csvs:
    - unit__outputNode__dt.csv
    - costs__dt.csv
```

Choose 2–3 CSV files that directly exercise the feature being tested.
Available filenames come from the `filenames:` section of
`templates/default_plots.yaml`.

### 3. Generate golden files

```bash
pytest test/ --regenerate my_new_scenario
```

### 4. Inspect and commit

```bash
ls test/expected/my_new_scenario/
# check the values look plausible for what the scenario should produce
git add test/expected/my_new_scenario/ test/test_scenarios.py
git commit -m "add test: my_new_scenario"
```

## DB fixture maintenance

`test/tests.sqlite` is the editable Spine DB (not committed to git).
`test/fixtures/tests.json` is its JSON export (committed to git).

**After editing the DB** (adding scenarios, changing data, running migrations):

```bash
python test/db_utils.py export test/tests.sqlite test/fixtures/tests.json
git add test/fixtures/tests.json
git commit -m "update test fixture: ..."
```

**After pulling changes** (to rebuild the SQLite from the JSON):

```bash
python test/db_utils.py import test/fixtures/tests.json test/tests.sqlite
```

The test suite always rebuilds the SQLite from JSON automatically at
session start — `test/tests.sqlite` is only needed when editing in
Spine Toolbox.

## Solver precision

`test/highs.opt` overrides `bin/highs.opt` during test runs:
- `threads=1` for deterministic results
- `primal/dual_feasibility_tolerance=1e-8` (tighter than production)
- `mip_rel_gap=0.0001` (near-exact for small test problems)

Output CSVs are rounded to 4 decimal places before comparison, which
absorbs any remaining numerical noise across HiGHS versions.
