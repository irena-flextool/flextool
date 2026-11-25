# Engine rules (`flextool/engine/`)

- **Parameter shapes.** A DB parameter's period / time axis comes from `input/_param_shapes.py`:
  `resolve_param_shape` → `broadcast_to_period_time` → `promote_param_to_dt`; a new parameter goes in
  `PARAM_ALLOWED_SHAPES`.  Never detect the axis by column name: Spine names an unnamed Map index `"x"`, so a
  `"t" in df.columns` check misreads a time series as a scalar and cross-joins it against the timeline (2³² overflow
  on long runs, a silent flatten on short ones).  `tests/test_invariant_guards.py` counts such checks.
  Model-internal axes (e.g. rolling `roll`) live in `flextool/schemas/flextool_axis_contract.json`.
- **Time aggregation.** A new time-series parameter needs its `TIME_AGGREGATION` kind in
  `input/_solve_step_aggregation.py` (how a solve's `new_stepduration` combines it).  Emit folds read
  `solve_data/pt_*` / `pbt_*`, never `input/`.
- **Dimension dtypes.** Dimension columns are Enum-typed; polars cannot join Utf8 to Enum.  A column holding several
  entity classes (e.g. `source` / `sink`: nodes and processes) uses the union axis `e`, never `n`.  Reconcile join
  dtypes with `align_join_dtypes` / `cast_dim` (`input/_axis_enums.py`); never cast Enum to Utf8 to make a join or a
  test pass, no `try: cast() except: pass` shims, and an empty branch has the same dtypes as the populated one.
  Process tokens never go in an `n` column.
- **Readers fetch only what the instance uses** (for rolling: the union of `(d,t)` over the rolls), filtering on `dt`
  early.  Exception: inflow scaling reads the whole timeline.  A `strict=False` cast that nulls rows is usually a
  missed `cast_dim`, not over-reading.
- **LP order is deterministic** because `flextool/engine/__init__.py::_install_polar_high_determinism` sorts the
  domains of `add_var` / `add_cstr`.  A script that builds an LP with `polar_high` directly must call it first.
  Check determinism across separate processes (`PYTHONHASHSEED` differs), not within one.
- **Memory:** `_RJEM_MALLOC_CONF` is set at the top of `flextool/__init__.py` (`flextool/_allocator.py`) before
  polars loads; never import polars above it.  Measure memory with `priv_dirty` (smaps_rollup).
- **A standalone repro that imports `tests/` helpers** (`tests/db_utils` → spinedb_api) turns the polars string
  cache off, so joins fail with "enum … str" although production is fine.  Reproduce through `flextool.engine` /
  `run_chain_from_db` without `tests/` imports.
- polar-high (`../polar-high-opt`) is ours: fix a generic problem there once rather than per site here; changes stay
  generic and leave FlexTool's equation calls unchanged.
