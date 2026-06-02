# CLAUDE.md — quick start for agents working in this repo

This file is the entry point. It does **not** duplicate the docs — read
the authoritative sources before non-trivial work:

- **`docs/dev/architecture.md`** — system structure, every subsystem, and
  the invariants below (in full, with the *why*).
- **`CONTRIBUTING.md`** — dev setup, running tests, changing the data
  structure / schema, migrations, and the PR checklist.

## Environment
- Python 3.11+, type hints throughout.
- Virtualenv at `~/venv-spi/` — invoke as `~/venv-spi/bin/python`.
- **Lint with ruff before committing**: `ruff check .` (use
  `ruff check --fix .` to apply safe autofixes). Keep the tree clean.

## Invariants that bite (don't learn these the hard way)

1. **Autoscale registries must stay complete.** Every new
   `add_var(...)` / `add_cstr(...)` and every new schema parameter MUST be
   registered in `flextool/engine_polars/autoscale/_layer2_types.py`
   (`VARIABLE_FAMILIES` / `CONSTRAINT_FAMILIES`) and `_quantity_types.py`
   (`PARAMETER_TYPES`). A missing entry does **not** crash — it's caught
   and the solve silently reverts to an un-scaled LP. Run autoscale-
   exercising tests with `FLEXTOOL_AUTOSCALE_STRICT=1` to make gaps loud.
   Dynamic f-string constraint names are pinned in
   `tests/engine_polars/autoscale/test_registry_coverage.py`.
   → architecture.md "Numerical scaling → Layer 2".

2. **Never broadcast a parameter over `(d,t)` by column name.** Route
   through `_param_shapes`: `resolve_param_shape` →
   `broadcast_to_period_time` → `promote_param_to_dt`. Column-name axis
   detection misreads Spine's silent-default `"x"` index and cross-joins
   against the whole timeline (2³² overflow on long runs; silent flatten
   on short ones). `_param_shapes` is the I/O contract only — model-
   internal axes (e.g. rolling `roll`) live in
   `schemas/flextool_axis_contract.json`, not the resolver.
   → architecture.md "Parameter shapes & (d,t) broadcasting".

3. **Tests build their DB from JSON/schema; never read a checked-in
   `.sqlite`.** Checked-in DBs (`templates/*.sqlite`) lag the schema and
   silently under-cover. Use a `tmp_path_factory` fixture (`json_to_db`,
   or the session-scoped `schema_db_url`).
   → architecture.md "JSON-fixture single source of truth".

4. **After a schema rename**, update the engine readers, `PARAMETER_TYPES`
   (carry old + new during the migration window), and the migration code
   in `flextool/update_flextool/db_migration.py`.

5. **Vectorized per-roll emit must stay byte-parity.** Render value
   columns with the `repr()` loop (`_render_value_column`), never
   `.cast(Utf8)` — a cast silently diverges on sci-notation padding and
   `NaN`. Lift cascade lookups from the de-duplicated dict
   (`lift_dict_to_lookup`), never raw CSV — duplicate keys explode the
   join. Preserve domain/period order (don't `.unique()` the source
   lists) and the `__eo` / `__to` sort keys. Every heavy `_emit_*`
   derive is gated by a `test_vectorize_<family>_parity.py` (Tier A
   `.equals` / Tier B `rtol≈1e-12`).
   → docs/dev/engine_polars.md "Vectorized per-roll emit".

## Running / testing
See `CONTRIBUTING.md` "Running Tests". Inner loop is the area-specific
suite (`tests/engine_polars/<area>/`); the full `tests/` integration
sweep runs rarely.
