# Build Scenario Test Suite — Specification

## Goal

Replace the ad-hoc `test/` comparison files and the never-implemented `execution_tests/` skeleton with a proper pytest-based scenario test suite. Tests run small focused FlexTool model scenarios end-to-end, compare CSV outputs against golden files, and live fully under version control.

## Background / Decisions Made

- **System tests only** — The primary source of bugs is `flextool/flextool.mod` (MathProg), which cannot be unit-tested in isolation. System tests are the right unit tests for the model.
- **Multiple focused scenarios from one DB** — Spine Toolbox's scenario/alternative structure lets one DB contain many small focused test scenarios without data repetition.
- **CSV comparison** — `write_outputs` already produces CSV files. These are stable and human-readable. Only a curated subset of CSVs is compared per scenario (not all outputs).
- **JSON fixture in git, SQLite DB not in git** — Devs work in the Spine DB GUI, then export to JSON for git. CI imports JSON → SQLite at test startup.
- **Floating-point tolerance** — A dedicated `test/highs.opt` sets solver precision; outputs are rounded before comparison to an order of magnitude higher than that precision.
- **Regenerate mode** — `pytest --regenerate SCENARIO_NAME` overwrites golden files for that scenario (not all at once).
- **Pytest** — chosen for its parameterization, CI integration, and `conftest.py` flexibility.

---

## Directory Layout (target state)

```
test/
  conftest.py                 pytest fixtures and --regenerate flag
  highs.opt                   HiGHS solver options (controls numerical precision)
  db_utils.py                 json_to_db(json_path, db_path) and db_to_json(db_path, json_path)
  fixtures/
    test_model.json           Spine DB exported to JSON (version controlled)
  expected/
    coal/                     golden CSVs for the 'coal' scenario
      unit_flow__dt.csv
      costs.csv
    wind_battery/             golden CSVs for the 'wind_battery' scenario
      ...
    <scenario_name>/
      ...
  test_scenarios.py           parameterized pytest tests
```

The existing `test/*.txt` comparison files and `execution_tests/` skeleton are **superseded** by this setup. The old files can be removed once the new suite is validated.

---

## Task 1: Understand FlexToolRunner CWD requirements — COMPLETE

**Findings:**

- `solve_data/` is **auto-created** by `FlexToolRunner.__init__` — no pre-setup needed
- `output_raw/` is **created by glpsol Phase 3** — holds raw solver output CSVs read by `write_outputs`
- `HiGHS.log` is written by HiGHS to **CWD** and read back to check for infeasibility
- `output/` is created by `write_outputs` for its own progress CSV — in CWD
- Intermediate solver files go to **`root_dir`** (not CWD): `flextool.mps`, `flextool.sol`, `glpsol_solution.txt`
- `flextool.mod` and `flextool_base.dat` come from **`flextool_dir`** (defaults to `<package>/flextool/`)
- `highs.opt` is read from **`bin_dir/highs.opt`** (NOT CWD) — passed as `--options_file=` to HiGHS
- `FlexToolRunner` **can be called directly from Python** — no CLI/subprocess needed; `output_db_url` is a CLI-only concern
- **Concurrency issue**: `solve_data/`, `HiGHS.log`, `output_raw/` all go to CWD → tests must use `monkeypatch.chdir(tmp_path)` and run sequentially (no pytest-xdist parallelism)

**Test harness pattern:**
```python
# Per test: monkeypatch.chdir(tmp_path) sets CWD for all CWD-relative writes
runner = FlexToolRunner(
    input_db_url=test_db_url,
    scenario_name=scenario,
    root_dir=tmp_path,       # intermediate solver files stay in temp dir
    bin_dir=test_bin_dir,    # points to temp dir with test highs.opt + symlinked binaries
)
# flextool_dir defaults to <package>/flextool/ — correct for installed package
```

---

## Task 2: Implement `test/db_utils.py` — COMPLETE

**Goal:** Two utility functions for the JSON ↔ SQLite DB round-trip.

```python
def json_to_db(json_path: Path, db_path: Path) -> str:
    """Import Spine DB JSON export into a new SQLite DB. Returns sqlite:/// URL."""

def db_to_json(db_path: Path, json_path: Path) -> None:
    """Export SQLite DB to JSON format (for committing to git)."""
```

**Investigate:**
- How does `spinedb_api` export/import JSON? Check `spinedb_api.export_db`, `spinedb_api.import_data`, or the Spine Toolbox exporter specs in `.spinetoolbox/specifications/`.
- The existing `.spinetoolbox/specifications/Importer/import_flex3.json` may show the import mapping format.
- Look at how `spinedb_api.DatabaseMapping` is used elsewhere in the codebase.

**Notes:**
- `json_to_db` is called by `conftest.py` at test session start (creates `test/test_model.sqlite` in a temp location).
- `db_to_json` is called manually by devs after editing the DB — not by the test suite itself.
- The SQLite DB should **not** be created inside the repo if avoidable; use a temp directory or the system temp dir.

---

## Task 3: Implement `test/highs.opt` and tolerance strategy — COMPLETE

**Goal:** Create a HiGHS options file that sets deterministic, tight numerical precision, and define the rounding strategy for CSV comparison.

**Decisions needed:**
- What precision does HiGHS use by default? What options control it? (Check HiGHS documentation or existing `bin/highs.opt` if present.)
- What decimal places should CSVs be rounded to for comparison? (Suggested: if HiGHS tolerance is 1e-7, round to 4 decimal places for comparison.)
- Should rounding happen at write time (in the test harness) or rely on the CSV output already being rounded?

**Output:**
- `test/highs.opt` with appropriate precision settings
- A `round_csv_for_comparison(df: pd.DataFrame) -> pd.DataFrame` helper in `test/conftest.py` or `test/db_utils.py`

---

## Task 4: Implement `test/conftest.py` — COMPLETE

**Goal:** pytest fixtures and custom CLI option.

**Required pieces:**

```python
# Custom flag
def pytest_addoption(parser):
    parser.addoption("--regenerate", metavar="SCENARIO", default=None,
                     help="Regenerate expected CSVs for the given scenario name")

# Session-scoped fixture: import JSON → tmp SQLite DB once per test session
@pytest.fixture(scope="session")
def test_db_url(tmp_path_factory) -> str:
    ...

# Function-scoped fixture: isolated working directory per test run
# (FlexToolRunner writes to CWD — each test needs its own CWD)
@pytest.fixture
def workdir(tmp_path) -> Path:
    ...
```

**Notes:**
- The `workdir` fixture must set up whatever CWD structure `FlexToolRunner` requires (from Task 1).
- The `test/highs.opt` should be copied into `workdir` so the solver picks it up.
- The `--regenerate` option is read in tests via `request.config.getoption("--regenerate")`.

---

## Task 5: Define initial test scenarios and golden CSVs

**Goal:** Choose the first set of focused test scenarios and generate their golden CSV files.

**Suggested initial scenarios** (subject to what the test DB contains):
- `coal` — basic thermal dispatch, no investment
- `wind_battery` — VRE + storage dispatch
- `coal_co2` — CO2 constraint
- `coal_min_load` — minimum load / unit commitment
- `network_coal_wind` — transmission network

For each scenario, decide **which CSVs** to compare. Start minimal — 2-3 key output files per scenario that cover the scenario's specific feature. Candidates: `unit_flow__dt.csv`, `costs.csv`, `node_balance__dt.csv`, `storage_state__dt.csv` (for storage scenarios).

**Process:**
1. Run the scenario manually (or via `--regenerate`)
2. Inspect the output — verify the results are analytically plausible (manually spot-check key values)
3. Commit the golden CSVs

**This task requires human review** — do not auto-generate and commit without inspecting values. See "Oracle problem" note below.

---

## Task 6: Implement `test/test_scenarios.py` — SKELETON COMPLETE (golden files pending Task 5)

**Goal:** Parameterized pytest tests.

```python
# Scenario definitions: (scenario_name, list_of_csv_filenames_to_compare)
SCENARIOS = [
    ("coal",             ["unit_flow__dt.csv", "costs.csv"]),
    ("wind_battery",     ["unit_flow__dt.csv", "storage_state__dt.csv", "costs.csv"]),
    ("coal_co2",         ["unit_flow__dt.csv", "costs.csv"]),
    ("coal_min_load",    ["unit_flow__dt.csv"]),
    ("network_coal_wind",["unit_flow__dt.csv", "node_balance__dt.csv", "costs.csv"]),
]

@pytest.mark.parametrize("scenario,csvs", SCENARIOS)
def test_scenario(scenario, csvs, test_db_url, workdir, request):
    regenerate = request.config.getoption("--regenerate")

    run_scenario(test_db_url, scenario, workdir)  # calls FlexToolRunner + write_outputs(write_methods=['csv'])

    for csv_name in csvs:
        actual = load_and_round(workdir / "output" / scenario / csv_name)
        expected_path = EXPECTED_DIR / scenario / csv_name

        if regenerate == scenario:
            expected_path.parent.mkdir(parents=True, exist_ok=True)
            actual.to_csv(expected_path, index=False)
        else:
            expected = pd.read_csv(expected_path)
            pd.testing.assert_frame_equal(actual, expected, check_exact=False)
```

**Notes:**
- `run_scenario` should call the Python functions directly (not subprocess) to avoid overhead and CWD complications — but this depends on Task 1 findings.
- If direct Python calls aren't feasible, use `subprocess.run` with `cwd=workdir`.

---

## Task 7: CI integration

**Goal:** GitHub Actions workflow that runs the test suite on push/PR.

**File:** `.github/workflows/tests.yml`

**Steps:**
1. Checkout repo
2. Set up Python (3.11)
3. Install dependencies (`pip install -e ".[dev]"` or `pip install -e . && pip install pytest pandas`)
4. Run `pytest test/ -v`

**Notes:**
- The CI environment has no pre-existing SQLite DB — `conftest.py` must create it from `test/fixtures/test_model.json` (this is the whole point of `json_to_db`).
- HiGHS binary must be available — check how it's currently distributed (in `bin/`).
- Add `pytest` (and optionally `pytest-xdist` for parallel test runs) to `pyproject.toml` dev dependencies.

---

## Oracle Problem Note

> "Correct results" must be verified against known-correct math, not bootstrapped from current model output.

For each scenario's golden CSVs:
- Simple scenarios (e.g., `coal` with 2 time steps, 1 unit) should be hand-verifiable against the LP formulation
- More complex scenarios are regression tests — they assert "the model hasn't changed" not "the model is correct"
- Document which golden files are analytically verified vs regression-only in a comment in `test_scenarios.py`

The `--regenerate` command exists for intentional changes; using it irresponsibly silently loses test coverage. It is the developer's responsibility.

---

## Recommended Task Order

1. Task 1 (CWD investigation) — blocks Tasks 4 and 6
2. Task 2 (db_utils) — can be done in parallel with Task 1
3. Task 3 (highs.opt + rounding) — can be done in parallel
4. Task 4 (conftest.py) — depends on Tasks 1, 2, 3
5. Task 5 (define scenarios + golden CSVs) — depends on Task 4 working
6. Task 6 (test_scenarios.py) — depends on Tasks 4, 5
7. Task 7 (CI) — depends on Task 6
