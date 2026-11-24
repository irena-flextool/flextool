# CLAUDE.md

FlexTool: an energy-system optimisation model.  The engine (`flextool/engine/`) reads a Spine database, builds an LP
with polar-high and solves it with HiGHS.  Folder-level `CLAUDE.md` files (engine, model, emit, solve,
update_flextool, process_outputs, gui, tests) load when you work in that folder and hold that area's rules.

## Read only what the task needs
- `docs/dev/architecture.md` (whole system, cross-cutting invariants) and `docs/dev/engine.md` (engine internals)
  start with a "Where to read" table: read the sections it names for your kind of task, not the whole file.
- Large modules: `~/venv-spi/bin/python tools/code_index.py <file>` (or `--find <name>`), then read the line range.
- `CONTRIBUTING.md`: dev setup, schema changes and migrations, the PR checklist.

## Tools
- Python: `~/venv-spi/bin/python`.  `ruff check .` (rules in `pyproject.toml`; CI runs it).
- `tools/related_tests.py [--run]`: the test modules that import what you changed.  Run those, not sweeps.
- `tools/lp_snapshot.py`: proves what a change does to the LP (byte-identical for refactors; every diff explained).
- Hooks (`.claude/settings.json`): a Stop hook runs ruff and the static guard tests on a dirty tree; a Bash hook
  refuses full-suite pytest runs and `-n` > 8; deny rules block destructive git.  A refusal explains itself.

## Rules for every change
- No shortcuts.  Nothing deferred to TODOs or later phases; a real blocker is reported, not worked around.
- Grep for an existing helper before writing one; never duplicate a helper across modules.
- No new dependencies: polars / numpy only, also for packages that arrive transitively (scipy via spinedb-api).
- Never make a test pass by loosening it (tolerance, skip / xfail, deleted assert, golden regenerated "to match");
  report instead.  Expected values are derived by hand from the fixture and the docs, never copied from new output.
- Windows (CI runs it, local runs don't): no file names built from reprs or free text; compare `Path` objects, not
  path strings; close sqlite / DatabaseMapping handles in `finally`; `os.replace` over an existing file;
  `encoding="utf-8"` on text I/O and subprocess (ruff checks `open()`).
- User-facing text (docs, CHANGELOG, messages, templates): a CHANGELOG entry under the release being built
  (result-changing → "Behaviour changes"); no private model names (a local pre-commit hook checks); no session
  labels ("critique #2", "fix 4(b)") in code, comments or test names — describe the behaviour.  Questions and
  decisions for the owner never go in commits, PRs, CHANGELOG or code.
- Git: the owner edits in parallel.  A dirty file outside your working set is their work: leave it alone, never
  revert or stash it; stop only if a file you must edit is dirty.  Stage explicit paths.  `specs/`, `projects/` and
  `_audit_reports/` are local folders (gitignored).
- Terms: *scenario* = one model instance (never a roll, level or branch); *forecast* = a stochastic branch;
  *solve / roll* = one LP run of a solve hierarchy; `full_dt` = the union of `(d,t)` over the rolls of a solve.

## Invariants (detail in the folder CLAUDE.md; checked where noted)
1. New `add_var` / `add_cstr` and new schema parameters are registered for autoscale — `engine/model/CLAUDE.md`;
   `test_registry_coverage`, strict mode on in tests.
2. A parameter's `(d,t)` shape comes from `_param_shapes`, never from column names — `engine/CLAUDE.md`;
   `tests/test_invariant_guards.py`.
3. Tests build their DB from JSON, never read a template `.sqlite` — `tests/CLAUDE.md`; invariant guards.
4. Schema renames update readers, `PARAMETER_TYPES`, the migration and `split_params` — `update_flextool/CLAUDE.md`.
5. Vectorized emit stays byte-identical to the per-roll writer — `engine/preprocess/emit/CLAUDE.md`; parity tests.
6. A new FlexData field has a CSV dump/reload route — `tests/engine/test_flexdata_field_routes.py`.
