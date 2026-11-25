# Output rules (`flextool/process_outputs/`)

- **Vectorized, set-based calculations.** Build the column set first (`Index.intersection`, membership in
  `s.process_unit` / `s.process_connection` / `s.node_state`), then one vector op (`.mul`,
  `.groupby(level=...).sum()`).  No Python loops over entities, periods or time steps; the only accepted loop is over
  the small fixed list of named cost categories.  Output units come from `_output_meta`.
- **Unrealized cells are empty, never 0.** The `invested` / `divested` cells of an (entity, period) that was
  invest-eligible but not realized (e.g. a rolling slice superseded by another roll) are empty: 0 would mean "the
  model decided zero".  Don't `fillna(0)` them; a golden with 0 there is wrong (regenerate, note it in
  `tests/expected/REGEN_LOG.md`).
- **Comparison / dispatch identity is the output-folder name** (e.g. `S2_Dry_1`), applied at the load boundary
  (`combine_parquet_files`, `combine_dispatch_mappings`, the viewer's `_load_dispatch_data`), not the model scenario
  name in the parquet.  The run-index suffix flows to legends and file names on purpose.
