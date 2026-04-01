# Contributing to IRENA FlexTool

## Development Setup

1. Clone the repository and create a virtual environment:
   ```bash
   git clone https://github.com/irena-flextool/flextool.git
   cd flextool
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

2. Install in editable mode with dev dependencies:
   ```bash
   pip install -e .
   pip install pytest pyyaml matplotlib
   ```

3. Verify the setup by running the tests:
   ```bash
   pytest tests/
   ```

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the codebase.

## Running Tests

Run the full test suite:
```bash
pytest tests/
```

Run a specific test file:
```bash
pytest tests/test_scenarios.py -v
```

Run a single scenario test:
```bash
pytest tests/ -k "coal" -v
```

### Golden File Tests

Scenario tests compare solver outputs against golden files in `tests/expected/<scenario>/`. To regenerate golden files after intentional changes to the model or output processing:

```bash
# Regenerate one scenario
pytest tests/ --regenerate coal

# Regenerate all scenarios
for s in $(grep "^- scenario:" tests/scenarios.yaml | sed 's/- scenario: //'); do
  pytest tests/ --regenerate "$s"
done
```

Always inspect the changes in `tests/expected/` before committing to make sure the differences are expected.

## Updating the Data Structure

When the FlexTool database schema needs to change (new parameters, renamed entities, new value lists, etc.), follow this process:

1. **Edit `flextool/update_flextool/db_migration.py`** — add a new migration step at the end of the `migrate_database()` function. Increment the version number in `flextool/update_flextool/__init__.py`.

2. **Regenerate the master template** — this ensures `version/flextool_template_master.json` matches the latest schema:
   ```bash
   python -m flextool.update_flextool.sync_master_json_template
   ```
   This creates a temporary database from the current master template, runs all migrations on it, and exports the result back to JSON.

3. **Verify** (optional, also run by CI):
   ```bash
   python -m flextool.update_flextool.sync_master_json_template --verify
   ```

4. **Commit both files together** — the migration code and the regenerated template should be in the same commit or pull request.

## Pull Requests

### Branching

- Create a feature branch from `master` for your work.
- Use descriptive branch names: `fix-reserve-output`, `add-ramp-cost`, `update-excel-export`.

### What a PR Should Contain

- **Focused changes** — one feature, one bug fix, or one refactoring per PR. Don't mix unrelated changes.
- **Tests** — add or update tests for any new or changed functionality. If you're fixing a bug, add a test that would have caught it.
- **Updated golden files** — if your change affects solver outputs, regenerate the affected golden files and include them in the PR.
- **Updated master template** — if your change modifies the database schema, include the regenerated `version/flextool_template_master.json`.
- **Clear description** — explain what the PR does and why. Reference any related issues.

### PR Checklist

- [ ] Tests pass locally (`pytest tests/`)
- [ ] No unintended changes to golden files
- [ ] Master template is up-to-date (If schema changed. Check [CONTRIBUTING.md updating the data structure](#updating-the-data-structure))
- [ ] Code follows existing patterns and style

### Review Process

- PRs to `master` trigger CI tests on Linux, Windows, and macOS.
- At least one review approval is recommended before merging.
- Prefer merge commits over squash to preserve commit history.

## Solver Binaries

FlexTool uses two solvers:
- **HiGHS** — called via the `highspy` Python package (installed automatically via pip).
- **GLPK (glpsol)** — a custom build from [mingodad/GLPK](https://github.com/mingodad/GLPK). Platform-specific binaries are in `bin/`. The macOS binary can be rebuilt via the `build-glpsol.yml` GitHub Actions workflow.

## CI Workflows

- **`tests.yml`** — runs pytest on all three platforms and verifies the master template is up to date. Triggered on push/PR to `master`.
- **`build-glpsol.yml`** — builds the glpsol binary for macOS arm64. Triggered manually.
