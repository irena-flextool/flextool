## Release 3.41.1 (20.5.2026) — Toolbox/cascade bug-fix sweep

Patch release on top of 3.41.0.  Clears four user-visible regressions that
surface when running from Spine Toolbox or comparing many Rivendell scenarios.

**Bug fixes**

- `cmd_run_flextool`: default work folder to cwd in both code paths.  When
  `--work-folder` was omitted the CLI kept two variables (`work_folder=None`
  for the solver, `wf=cwd` for `write_outputs`), so `run_chain_from_db`
  spun up its own `/tmp/flexpy_run_chain_*` while `write_outputs` looked
  for `output_raw/` under cwd and tripped a `FileNotFoundError` on
  `v_obj.csv` (`output_csv/`, `output_parquet/`, `output_excel/`,
  `output_plots/` were skipped on the failure path).  Visible from
  Toolbox runs as a `write_outputs failed` warning right after the solve
  reported optimal.
- Toolbox DB URL handling: six normalisation sites
  (`_orchestration`, `_solve_config`, `_spinedb_reader`, `_timeline`,
  `spinedb_backend._backend`) used to whitelist only `sqlite:` /
  `postgresql:` schemes and prepended `sqlite:///` to anything else,
  turning Toolbox's engine-server URLs (`http://127.0.0.1:<port>`) into
  `sqlite:///http://...` and erroring with
  `unable to open database file`.  The guard now passes through any URL
  that already carries a scheme and only prepends `sqlite:///` for bare
  filesystem paths — `spinedb_api`'s `DatabaseMapping` already resolves
  Toolbox server URLs natively via `get_db_url_from_server`.
- Scenario comparison: cross-scenario combine no longer fails with
  `KeyError: '[nan] not in index'` in `prepare_plot_data` when some
  scenarios produce empty result frames.  `calc_connections` used to
  build empty `connection_d` / `connection_losses_d` without a column-
  level name; once the per-scenario parquet writer wrapped them with
  `pd.concat({name: df}, axis=1, names=['scenario'])` they shipped with
  names `['scenario', None]`.  The cross-scenario combine then collided
  with `['scenario', 'process']` from scenarios that *do* have
  connections, pandas resolved the conflict to `NaN`, and the comparison
  plotter blew up on the level-name lookup.  Empty connection frames
  now carry the same `'process'` column name as the non-empty branch.
- Same anti-pattern (empty / scalar result frames shipping without a
  column-level name) cleared on four more producers that would have
  hit the same `KeyError` under different comparison configs:
  `out_ancillary.largest_flow` (`nodeGroup_inertia_largest_flow_dt_g`),
  `out_ancillary.years_d` (`years_represented__d`),
  `out_group.results_dt` (`nodeGroup_VRE_share_dt_g`, both empty and
  non-empty branches), `out_costs.co2_summary` (`CO2__`).

**Diagnostics**

- `[mem]` phase log: align columns into a single table, drop the
  duplicated `(rss=…, peak=…)` block, and split the `load_flextool`
  entry into five then three sub-checkpoints so the cascade-load
  breakdown is readable in the always-on output.

## Release 3.41.0 (20.5.2026) — Phase E: lazy broadcast cascade + always-on phase log

Built on 3.40.0.  Lands Phase E of the Step 3 cascade-memory work —
the lazy-broadcast rewrite of the eight target Params and the
on-demand rebuild of the persistent cross-join scratch frames
(`pss_dt`, `nodeBalance_dt`, `nodeState_dt`, `nodeState_first_dt`,
`process_indirect_dt`).  Together these drop the cascade-load peak by
~80 % on the H2_trade fixtures.  Also unifies all phase-progress
logging into a single user-visible format, with section timing and
ΔRSS attributed per phase out of the box (no env-var opt-in).

**Phase E.1 — lazy broadcast helpers** (`_param_shapes.py`,
`_direct_params.py`)

- `broadcast_to_period_time` / `broadcast_to_period` now return
  `Param` with dims matching the actual authored shape:
  `SCALAR → (entity,)`; `MAP_PERIOD → (entity, d)`;
  `MAP_TIME → (entity, t)`;
  `MAP_PERIOD_TIME / MAP_TIME_PERIOD → (entity, d, t)`.  No eager
  `.collect()` inside the helpers — the underlying LazyFrame chain
  carries the period filter as an inner-join on the natural-dim axis
  instead of a cross-join with `dt`.
- `_filter_param_by_periods` operates on `p.lazy` (NOT `p.frame`,
  which would trigger polar_high's eager cache), shape-aware
  (filters on whichever of `{d, t}` are present in `p.dims`),
  returns `Param(p.dims, lf)`.
- `p_process_availability_from_source` promotes per-class parts
  (unit + connection) to union dims via lazy joins on
  `period_filter` for missing axes, then concats lazily — preserves
  a single consistent dim shape across mixed authoring without
  forcing eager materialisation.

polar_high handles the `(entity, d, t)` broadcast lazily at
constraint emission via its shared-dim inner-join contract; the
dense cross-product only ever lives in the per-term collect that
polar_high streams to HiGHS one row at a time.

**Phase E.2 — consumer-side `promote_param_to_dt`**

The eight target Params (`p_node_availability`,
`p_storage_state_reference_value`, `p_co2_price`,
`p_co2_max_period`, `p_process_availability`, `p_commodity_price`,
`pdt_max_instant_flow`, `pdt_min_instant_flow`) have **zero**
direct `.frame.<op>` consumers in `engine_polars/` — they're all
consumed via polar_high algebra (`*`, `over=`, `rhs_terms=`) which
inner-joins on shared dims and cross-joins on disjoint.  But a
handful of cascade sites do hand-rolled eager joins for
"left-join with default 1.0" semantics that polar_high's `*`
doesn't natively express:

- `model.py` flow_upper_rhs availability fold, storage_state
  reference value chain.
- `_region_filter._inject_half_flows` concat of virtual half-flow
  availability rows.
- `_dump_csvs` `pdt*.csv` slice writers (the dump path needs the
  fully-expanded `(entity, d, t)` frame so output CSVs match the
  pre-Phase-E layout consumers expect).
- `process_outputs/read_parameters._pdtX_per_entity` pandas pivot.

For these, a new helper `promote_param_to_dt(param, dt)` in
`_param_shapes.py` returns a LazyFrame with both `d` and `t`
columns — joining on `dt` for whichever axes the Param is missing
(cross-join when both absent).  Each consumer calls the helper
once at the top of its function and re-uses the result.

**Phase E.3 — drop persistent cross-join scratch**

The five cross-join scratch frames previously held in `FlexData`
(`pss_dt`, `nodeBalance_dt`, `nodeState_dt`, `nodeState_first_dt`,
`process_indirect_dt`) were duplicates of the data that
`v_flow.frame` / `v_state.frame` / etc. already hold after
`add_var`.  On y2050 the `pss_dt` alone was ~1.75 GB (43M rows ×
5 cols × 8 bytes); the four siblings each contribute ~0.5-1 GB.

- New module `flextool/engine_polars/_pdt_join.py` with
  `compute_pss_dt(d)`, `compute_nodeBalance_dt(d)`,
  `compute_nodeState_dt(d)`, `compute_nodeState_first_dt(d)`,
  `compute_process_indirect_dt(d)`.  Each lazily joins
  constituents (`process_source_sink`, `nodeBalance`, `nodeState`,
  `dt`) and collects on demand.
- `_fast_load._populate_pss_dt_and_balance_dt` stops populating
  the fields; slow-path `FlexData(...)` calls pass `None`.
- 8 consumer sites in `model.py` (add_var domains for v_flow /
  vq_state / v_state; `Sum` / `add_cstr` `over=` arguments; one
  explicit join with `pd_neg_cap`) call the compute helpers.
  Per-function locals cache the result so repeated reads don't
  rebuild the cross-join.
- `_region_filter._inject_half_flows` uses `compute_pss_dt(rd)`.
- 12 test files updated to compute their own slices.

**Always-on phase log**

Previously `[mem]` phase-progress log lines were gated on
`FLEXTOOL_MEMORY_DIAGNOSTICS=1` — users following a normal run saw
only the cryptic `input pass NN: 0.007s` lines with no memory
attribution.

- `_MemoryRecorder` now emits log lines unconditionally (RSS reads
  from `/proc` are essentially free).
  `FLEXTOOL_MEMORY_DIAGNOSTICS=1` additionally enables tracemalloc
  (so `traced_peak` becomes a real number rather than `-`) and
  writes the per-checkpoint CSV under
  `solve_data/memory_diagnostics.csv`.  Without the env var the
  log lines still appear but show `peak=-` / `Δpeak=-`.
- Module-level `set_phase_recorder(rec) / get_phase_recorder()`
  lets deeper modules (`input.py::_apply_db_overrides`) emit
  checkpoints in the unified format without each carrying a
  recorder kwarg.
- `input.py::_timed` routes through the recorder when one is
  active, falling back to the legacy plain print only when no
  recorder is registered (e.g. unit tests outside
  `run_orchestration`).
- Three new sub-phase checkpoints inside `input_derivation.run`:
  *Spine DB loaded*, *DB-driven derivations done*,
  *Preprocessing writers done*.
- User-visible labels replace the cryptic internal identifiers:
  `cascade_start` → *Run start*; `write_workdir_inputs_end` →
  *Input data prepared (after malloc_trim)*;
  `first_load_flextool_end` → *Model cascade built*;
  `first_lp_build_end` → *LP problem built*; `first_solve_end` →
  *Solver finished*.
- Each line shows section time + Δrss + Δpeak + absolute
  (rss, peak).  Sizes auto-format MB ↔ GB.

CSV schema (`solve_data/memory_diagnostics.csv`) unchanged —
downstream tooling that parses it keeps working.

**Measured cascade-load impact** (H2_trade test_24h, fast path):

| Checkpoint | Pre-Phase-E (3.40.0) | Post-Phase-E (3.41.0) | Δ |
|---|---:|---:|---:|
| `cascade_start` | 276 MB | 275 MB | parity |
| `Spine DB loaded` (new) | — | 476 MB | first sub-checkpoint |
| `Preprocessing writers done` (new) | — | 489 MB | — |
| `Input data prepared` (post-trim) | 1284 MB | 392 MB | **−69 %** |
| `Model cascade built` (`first_load_flextool_end`) | **3320 MB** | **667 MB** | **−80 %** |
| `LP problem built` | 3340 MB | 667 MB | −80 % |

y2050 cascade-load (from Phase E.1+E.2 commit before E.3 landed):
3041 MB at `Model cascade built` vs. pre-Phase-E baseline of
16-20 GB (process was killed at the 5-min timeout).  E.3 is
expected to drop this further; the user re-measures on their side.

**Test results** — 0 regressions on the targeted Phase E gate.
631-test inner-loop gate and the full v3.32.0 byte-parity suite
will be re-verified before tagging.

---

## Release 3.40.0 (20.5.2026) — Step 3 cascade memory (Tracks A + B.1) + post-Phase-4 bug-fix sweep

Built on 3.39.0.  Lands the first two Step 3 memory tracks
(SpineDBBackend `parsed_value` eviction + chunked row accumulator;
`FlexDataProvider` lifetime + eviction infrastructure) plus the
post-Phase-4 bug-fix follow-on that cleared ~30+ tests across
Phase 4 dtype reconciliation, profile cluster parity, anchor-window
nested-Map indexing, capacity-margin × 1000 stale goldens, and the
fast-path / synthetic-solve cascade.

The remaining Step 3 broadcast-cascade peak (the `~3.3 GB`
`first_load_flextool_end` checkpoint on `test_24h`) is addressed by
Phase E in the next release — see `specs/phase_e_handoff.md`.

**Step 3 Track A — SpineDBBackend memory (`spinedb_backend/_backend.py`)**
- Track A: `SpineDBBackend.parameter_values` drops each
  `MappedItem`'s cached `parsed_value` (Map / TimeSeries / Array
  Python object) as the materialiser advances through `params`.
  Spinedb-api's `parsed_value` is a lazy property; setting
  `_parsed_value=None` releases the parsed object while keeping the
  raw value/type for any defensive re-access.  Each (class, param)
  is touched once per `input_derivation` pass, so the
  generator-wrapped eviction achieves the full win at zero
  re-parse cost.
- Track A.5: `row_origins` is only consulted by `_maybe_cast` on
  axis-enum cast failure.  When `axis_enums is None` (the default
  `input_derivation` path) the list is pure waste — replaced raw
  `list.append` with a bound `_origin_append` that no-ops when no
  cast is requested.
- Track A.6: chunked row accumulator — some specs (notably
  `('profile', 'profile')` on H2_trade.sqlite) flatten into
  multi-million Python row-lists.  Flush `rows` into a polars
  sub-frame every `_ROWS_FLUSH_THRESHOLD` (default 200 000) entries
  and clear the list.  Sub-frames are concatenated before
  `_maybe_cast` runs.  Gated to `axis_enums is None`; the
  axis-enum-supplied callsites are tests with small fixtures where
  the peak doesn't bind.

**Step 3 Track B.1 — `FlexDataProvider` lifetime + eviction infrastructure**
- `EvictedFrameError(KeyError)` — raised by `get()` on a name that
  `release_unused()` has dropped.  Carries frame name +
  responsible item-group token.  Deliberately not caught silently
  anywhere; a hit signals an incomplete READS declaration or a
  handler reading outside its declared scope.
- `FlexDataProvider(rss_budget_mb=, retain_all=)` — constructor
  takes the threshold budget and the CSV-dump retention flag;
  budget also reads from `FLEXTOOL_RSS_BUDGET_MB` env var.
- `register_handler(handler_id, *, reads, groups=None)` — handler
  declares its frame reads and (optionally) the item-groups at
  which it fires.  `groups=None` = pinned to the last group
  (conservative).
- `precompute_lifetimes(item_groups)` — walks registered handlers;
  computes `_last_needed[name] = (group_token, group_idx)` per
  frame.  Re-running resets the evicted-frame markers.
- `release_unused(*, after=group)` — drops frames whose
  `_last_needed` is at or before `after` in iteration order.
  No-op when `retain_all=True`, when the threshold gate is closed
  (`rss_estimate_mb() <= rss_budget_mb`), or when
  `precompute_lifetimes` hasn't been called.
- `rss_estimate_mb()`, `is_evicted()`, `reset_lifetimes()` —
  helpers.
- `get()` checks the evicted set first and raises
  `EvictedFrameError` (with the correct breadcrumb) on a hit; the
  bare/qualified key promotion logic in the existing lookup path
  is extended to also recognise an evicted qualified key.
- No production-call sites plumb the new API yet — Phase B.2 will
  declare READS on the cascade handlers; Phase B.4 will wire
  `release_unused` into the LP build loop (deferred — Phase E
  removes the underlying broadcast-cascade peak that motivated the
  eviction; Track B.2-B.8 becomes regression-protection rather
  than peak-reduction).

**Measured impact on H2_trade test_24h (Tracks A + A.5 + A.6 + B.1)**

| Checkpoint | Pre-Track-A | Post | Δ |
|---|---:|---:|---:|
| `input_derivation` peak (probe) | 3447 MB | 1091 MB | −2356 MB (−68 %) |
| process maxrss (probe) | 3625 MB | 1091 MB | −2534 MB (−70 %) |
| `write_workdir_inputs_end` (post-trim) | 4796 MB | 1284 MB | −3512 MB (−73 %) |
| traced_peak @ `write_workdir_inputs_end` | (high) | 516 MB | −82 % |

Wall-clock cost +4 s / +16 % on test_24h `input_derivation`
(25 → 29 s) — chunking adds a small fixed cost per spec.

**Bug fixes (post-Phase-4 sweep)**

- *Stochastic profile lazy cascade — forecast-branch period
  handling* (`_derived_profile.py` + `_writer_provider_io.py`).
  Cleared 4 stochastic-fixture test_profile_cluster_parity
  failures (max-abs-diff 0.987 → 0).  Lazy cascade now correctly
  handles `period1_upper / _realized / _lower / _mid` forecast
  branch rows.
- *autouse fixture resets global `axis_enums` ContextVar between
  tests.*  `load_flextool`'s finally clause sets the ContextVar on
  success so post-load consumers see the live vocabulary, but the
  ContextVar persisted across tests in the same pytest worker —
  pinning Enum dtypes that differ from the next test's fixture's
  vocabulary, surfacing as `enum on left does not match enum on
  right` SchemaError.  Autouse fixture in
  `tests/engine_polars/conftest.py` clears it before/after each
  test.  Unblocked ~20 previously-flaky tests across
  `test_warm_chain_runner`, `test_orchestration_parity`,
  `test_axis_rename_helpers`, Rivendell scaling, synthetic toys,
  loaders.
- *pbt_node_inflow + dump_csvs_roundtrip + anchor-window (10
  tests).*  Cluster A (3): cascade uses `capture_frames` so disk
  CSVs the pbt parity tests assumed weren't written — rebuilt the
  tests on top of a provider from the captured frames.  Cluster B
  (3): `apply_branch_cluster` overwrote five `flex_data` fields
  with provider-only reads, nuking seeds that
  `_load_branch_artefacts` had populated via disk fallback for
  static fixtures.  Made overlay seed-preserving (only overwrite
  when override yields non-None OR the seed is already None).
  Cluster C (3 invest_5weeks_p* sub-solves): examples.sqlite's
  `invest_5weeks.invest_periods` Map has BOTH levels named `"x"`
  so `SpineDbReader._discover_index_cols` emitted `["x", "x"]` and
  `_emit_leaf` collapsed them deepest-wins, silently dropping the
  outer anchor index.  Disambiguate colliding nested-Map index
  names by suffixing deeper levels with `_<depth+1>` (`x` outer +
  `x` inner → `x` + `x_2`).
- *network coal-wind capacity-margin + multi-year-no-invest (4
  tests).*  Sub-cluster A: capacity-margin × 1000 stale parquet
  goldens (cascade `_group_slack.py:1233` correctly applies the
  unit conversion per the canonical model spec; legacy GMPL
  parquet goldens missed it).  Regenerated.  Sub-cluster B:
  availability 1.44× real cascade bug in `_param_shapes.py` and
  `model.py` — producer-side fix applied.  Sub-cluster C:
  multi-year wind no-invest — `_cumulative_invest.py` join-key
  mismatch on `p` Enum (cross-vocab); applied
  `align_join_dtypes / cast_dim` pattern.
- *Phase-4 dtype reconciliation in `_load_process_topology`,
  `_group_slack`, `_dc_power_flow`.*  Three cascade producer-side
  fixes surfaced by post-Phase-4 enum activation, all following
  the established `cast_dim / align_join_dtypes` pattern.
  Cleared ~30 tests across `test_warm_chain_runner`,
  `test_warm_param_autoupdate`, Rivendell, `test_scaling_bench`,
  toys, `test_orchestration_parity`, `test_native_cascade_parity`,
  `test_output_writer`, emission, capacity-margin, loaders.
- *Phase-4 omnibus cleanup (15 tests).*  3 obsolete
  `@pytest.mark.xfail(strict=True)` on `test_scaling_parity`
  removed (the `p_unitsize` excludes-node-unitsizes bug was fixed
  by Phase 4 vocab union).  `_cumulative_invest.py` Phase 4.8h
  boundary-cast pattern applied to 4 more emit sites.  New
  `FlexData.process_source_toSink_dc` field carries the
  one-direction toSink frame for DC arcs.  `input.py:1787`
  threaded `provider=` to `_read_active_solve` in `_load_invest`
  synthetic-solve gate detection.  `cmd_run_flextool.py` passes
  `regions=regions_detected` to `solve_lagrangian`.  Various
  test-side `cast_dim` reads to match production Enum.
- *`rename_to_axis` entity→f wide-CSV cast + synthetic-solve dtt
  seeds.*  Two bugs in block-aware multi-fullYear storage —
  `_read_wide_per_entity`'s long-CSV branch and the synthetic-solve
  `dtt` seed routing.
- *Threaded `provider=` to chained-existing-capacity readers.*
  Restored the cross-solve handoff chain for
  `p_entity_all_existing` (similar shape to v3.39.0's
  `apply_derived_f` fix).
- *Unscale `sol.obj` in place so `step.solution.obj` matches
  `v_obj` parquet.*  Objective values written to parquet were
  unscaled but the in-memory `step.solution.obj` was left scaled,
  surfacing as a parity gap downstream of the solver step.
- *Lazy NPV port — add missing `ed_lifetime_fixed_cost` arms.*
  Three arms missing from the lazy NPV port (task #17).
- *Widen Enum vocab at injection in `_inject_half_flows`.*  Half-flow
  injection produced Enum dtypes with narrower vocabularies than
  their union axis, breaking downstream `is_in` semi-joins.
- *Fast path runs preprocessing in-memory, unifies with slow path.*
  The fast load path previously bypassed
  `input_derivation.run` — unified so fast + slow paths share the
  same preprocessing semantics.

**Test infrastructure**
- Cherry-picked Track A + B.1 commits from the `step-3-memory`
  worktree onto `new-outputs` (linear history, no conflicts).
  `tests/engine_polars/test_provider_lifetime.py` (15 tests),
  `tests/spinedb_backend/test_parsed_value_eviction.py` (4 tests),
  `tests/spinedb_backend/test_memory_budget.py` (1 test) added.

---

## Release 3.39.0 (19.5.2026) — `enum dtype` refactor + cascade perf

Built on 3.38.0.  Completes the `enum dtype` refactor (Phase 0 → 4.9
plus the `polar_high` companion) so that every cascade axis-aware
column carries a polars `Enum` dtype end-to-end.  The refactor +
follow-on perf work delivers a ~98 % reduction in per-loader cascade
memory and a ~9× speed-up in `input_derivation.run` on the H2_trade
test_24h fixture (the canonical large-fixture diagnostic).

**Phase 4 enum dtype refactor (final pass — 4.7d → 4.9)**
- 4.7d: `align_join_dtypes` adoption sweep + `_load_edd_history`
  consumer audit
- 4.7e: Pattern 5 `lit_axis` for `fill_null` on axis-aware columns
  (`input.py` block-compat joins)
- 4.7f: relax mixed-vocab `n` → `e` in `flow_to_n` / `flow_from_n`;
  restore axis types post-Utf8 compare for `d`/`b` cross-Enum filter
- 4.8a: structural Enum-typing in `_inflow_scaling`
  `pbt_node_inflow` producer/fold (dict-LazyFrames now declare
  explicit Enum schema)
- 4.8b: `schema_dtype(...)` in empty-frame fallbacks across
  `_solve_context.py` and writer modules — empty + populated frames
  now agree on dtype under activation
- 4.8c: `block` axis vocab `key_kind: "values_plus_default"`
  substrate fix (scalar-per-entity `new_stepduration` was returning
  empty vocab); cascade `block_coarse → "bk"` cascade fix; together
  clear 84 lh2_three_region failures
- 4.8d: defensive `d` / `t` Enum re-cast at `p_profile_value_lf`
  entry — stochastic dt_lf paths cleared 25+ `derived_e` failures
- 4.8e: defensive d/t re-cast across `_derived_arithmetic`,
  `_derived_branch`, `_inflow_scaling`, `_derived_params`,
  `_projection_params`, `_derived_block` — 22 sites
- 4.8f: `cast_frame_axes` at cascade-helper entry — 17 sites
- 4.8g: cross-Enum `is_in` semi-join in `model.py` + `_direct_params`
  entry cast
- 4.8h: `model.py` cross-Enum `is_in → semi-join` sweep across
  18 sibling sites
- 4.8i: cross-Enum-different-vocab join fixes for wind_battery /
  lh2 / network_coal_wind_reserve (`model.py` per-Var down-casts
  reading target dtype from the FlexData schema, since
  `_LIVE_AXIS_ENUMS_CTX` was empty inside `build_flextool`)
- 4.9 (substrate): `polar-high-opt` ships `_align_enum_join_keys` —
  a generic subset-aware up-cast helper invoked at every internal
  join site in `polar_high/engine.py`.  Cascade can now drop ~25
  per-Var down-cast lines in `model.py`; the DSL stays unchanged.
  Documented in `polar-high-opt/README.md` "Enum dtype handling"
  section + 12 new tests in
  `polar-high-opt/tests/test_enum_dtype_align.py`

**Cascade performance (perf-fix follow-on)**
- `SpineDBBackend.parameter_value` cache:
  `find_parameter_values(class=, param=)` under spinedb-api's
  scenario filter scales with the in-memory parameter_value table
  (~1.85 s per call on H2_trade), not with filter selectivity.  One
  bulk `find_parameter_values()` (~2.7 s for ~25 k rows) + Python
  partition into `{(class, param): [rows]}` replaces ~100 individual
  filtered calls in `input_derivation.run`.  **~60× speed-up** on
  the parameter-fetch hot path.  Cache is lazy per-backend-instance,
  invalidated on `close()`.
- Columnar `_unroll_rows` in `SpineDbReader`: replaced the
  row-of-dicts → `pl.DataFrame(out_rows)` pattern (~503 ms on the
  hot `profile/profile` param) with `dict[str, list]` →
  `pl.DataFrame(columns, schema_overrides=…)` (~58 ms).  Recursion
  uses a positional `idx_path` mutated via append/pop, eliminating
  per-recursion `dict(base)` copies.  Tolerant of mixed-type Map
  indexes via `strict=False` + per-leaf last-wins consolidation in
  `_emit_leaf`.  **~10 % per-parameter speed-up** end-to-end
- Persist `_LIVE_AXIS_ENUMS_CTX` past `load_flextool`'s return so
  cascade helpers calling `cast_dim(..., None, axis)` see the live
  enum vocabulary inside `build_flextool` (substrate-level fix that
  unblocked `_delay.py:324` and several latent cross-Enum joins)
- Forward `provider=` through `apply_derived_f`'s three
  `*_from_workdir` calls — restores the multi-solve handoff chain
  (`p_entity_all_existing` accumulates correctly across sub-solves)

**Measured wins on H2_trade test_24h (vs 2026-05-13 baseline)**
- `cascade_start`: 276 MB → 276 MB (parity)
- `input_derivation.run` wall-clock: **33 s → 25.9 s (~21 % faster)**
- Per-loader cascade ΔRSS: **~950 MB → ~21 MB (~98 % reduction)**:
  `_load_node` -99 %, `_load_process_topology` -96 %,
  `_load_varcost` -99 %, `_load_profiles` -32 MB
- Broadcast cascade (`write_workdir_inputs_end →
  first_load_flextool_end`): +2.2 GB growth in baseline → **−0.8 GB
  drop** here
- 631-test gate: ~23 s → ~17 s (incidental)

**Cascade robustness**
- `seed_provider_from_dir` is now tolerant of malformed
  `solve_data/*.csv` files — stray dev artefacts with bad headers
  no longer block the whole loader; a warning is logged per skipped
  file
- Numeric-column → Enum cast guard at SpineDBBackend +
  SpineDbReader emit boundaries: polars' numeric → Enum cast
  reinterprets values as positional indices into the enum's
  categories; the guard skips the cast for numeric source columns
  (which are never dim columns under the contract)
- `polar_high` (3.39.0-companion): defensive `if nm is not None:`
  guard on `h.passRowName(i, nm)` mirrors the existing col-name
  guard — null row names from `pl.format(...)` on null axis columns
  no longer crash highspy

**Test suite**
- **Retired** `tests/engine_polars/test_db_direct_parity.py` (~95 %
  of its 981 tests were CSV-vs-DB-direct migration scaffolding —
  the migration is established).  Salvaged 45 survivor tests into
  `tests/model/test_db_direct_solve_parity.py`: the LP-solve-vs-
  parquet-objective parametric matrix, `InMemoryReader` /
  `load_flextool` entry-point edge cases, `test_resolved_default_landed`
  default matrix, and seven literal-value singletons.  37 pass; 4 of
  the 6 remaining failures fixed mid-session (see "Bug fixes" below);
  the 4 leftover failures are documented in `specs/model_bugs.md`
  (two pre-existing LP-objective baseline issues, two stochastic
  period-vocab cascades — none are enum-pattern fixes)

**Bug fixes (parity-test survivors)**
- `_delay.py:324` cross-Enum compare: substrate ContextVar
  persistence (Fix 1 — see above)
- `apply_derived_f` missing `provider=` forwarding to three
  `*_from_workdir` handoff loaders (Fix 3 — pre-existing bug from
  commit `5f9c50809a` 2026-05-05, surfaced now)
- polar_high defensive `passRowName` guard (Fix 2 — masks but does
  not fix the underlying stochastic period-vocab gap; tracked in
  `specs/model_bugs.md::PARITY-3`)
- Mixed-type Map indexes regression from columnar refactor
  (commit `a98caa1c`) — `strict=False` on the two
  `pl.DataFrame(columns, schema_overrides=…)` call sites + last-wins
  consolidation in `_emit_leaf` for duplicate index-column names

**Documentation**
- `specs/memory_diagnostic_results.md` — appended "Post-perf-fix
  re-measure (2026-05-19 — commit `e6797b17`)" section with the
  numbers above
- `specs/model_bugs.md` — added PARITY-1/2/3 (3 LP-objective parity
  failures with legacy-trust discussion), RESERVE-1 (`prundt`
  populator missing on `fullYear_roll`), ARITH-1
  (`flow_upper_rhs * p_process_availability` Utf8 leaf on
  `test_a_lot`), REGRESS-1 (FIXED), PERIODS-1 (`_solve_periods`
  expects `"i"` column that `examples.sqlite::invest_5weeks` doesn't
  produce), and TASK COVERAGE-1 (add automated e2e for three
  `examples.sqlite` scenarios that surfaced four of the bugs above
  — fixture data already in place, pure test-wiring)

---

## Release 3.38.0 (18.5.2026) — Rivendell bug-fix suite

Built on 3.37.0's in-memory cascade. Fixes a cluster of regressions that surfaced on the Rivendell customer database after the cascade rewrite, plus a final round of test ports.

**This release is the last known-stable point before the in-flight `enum dtype` refactor (Phase 0 → 4.8); pin to 3.38.0 until that lands green.**

**Bug fixes**
- Rivendell bug 1+5+6: `engine_polars/scaling.user_bound_scale` recommendation is now column-only (a previous row+col recommendation collapsed solver progress on S17/B0)
- Rivendell bug 2 / BUG A2: `engine_polars` casts `user_constraints` coefficients to Float64 — previously left as int when authored as integer, breaking polars joins downstream
- Rivendell bug 3: `co2_price` shape probing uses value-domain inspection instead of column count when the shape is ambiguous
- Rivendell bug 4: regression test added for the `process_online_dt` UC column carry
- BUG A1: `blocks` + output writer route input/block reads through the Provider so they observe in-memory writer output rather than a stale on-disk snapshot
- BUG A3: `process_outputs.process_source_sink_varCost` keyed by `alwaysProcess` (previously dropped rows for processes that only exist in some subsolves)
- BUG A4 / A2-followup: branch-weight Provider-routed read + 1000× cap-margin restored on the cascade scaling/state path; provider threaded into cascade scaling/state helpers
- BUG B1: cascade fans rolling-ladder accumulators back into the FlexData Provider on each roll so per-roll cumulative quotas no longer drift
- BUG B2: MATPOWER → Spine converter sets `node_type=commodity` on the per-generator fuel nodes it synthesises (previously emitted with the default type, breaking `node_balance_eq` on imported MATPOWER cases)
- BUG `p_online_dt_empty_no_blocks`: `p_online_dt_set` falls back correctly when UC is enabled but no `process_block` is authored

**Test infrastructure**
- `tests/perturbation/*` ported to the native cascade harness
- `tests/emission/*` ported to polar_high `Problem` inspection (the underlying MPS-emission path was retired in Δ.22)
- Solver license probe at startup is skipped under test to silence the Xpress `LicenseWarning` on CI
- `tests/model`: invest-chain regression test (Pre-work 0b R1+R2) + LP-bound-range smoke for invest + `work_base` (Pre-work 0b R7)

---

## Release 3.37.0 (17.5.2026) — In-memory cascade (FlexData Provider/Accumulator + package refactor)

This release retires the on-disk CSV round-trip that previously sat between every cascade stage. FlexData is now built, threaded, and consumed entirely in memory across the loader, writers, solver, and output writers. `--csv-dump` is the only path that still emits the intermediate CSVs (for debugging / GUI inspection); the production path keeps everything in `polars.DataFrame` form.

Two new packages factor the cascade I/O surface: `spinedb_backend` (Spine DB → FlexData), and `input_derivation` (preprocessing derivations that previously lived in `flextoolrunner/preprocessing/*`). `flextoolrunner/preprocessing/*` and `flextoolrunner/input_writer.py` are deleted.

**FlexData Accumulator (Phases C–H)**
- Phase C: per-sub-solve `FlexDataAccumulator`
- Phase D: `load_flextool(seed=FlexDataAccumulator)`
- Phase E (a–j): cascade consumes accumulator via `seed=`; 8 batches of monolith-writer "lifts" so writers populate the accumulator rather than emit CSVs (`_writer_arc_unions`, `chain_params` + `co2_accumulators`, `_writer_pdt_params`, `_writer_period_params`, `dispatchers` + `calc_params`, the remaining 4 monolith writers, `_writer_solve_writers`, `period_calc` + `per_solve` + `reserve`); `--csv-dump` gating; cross-solve carriers lifted into `capture_frames`; seed-aware `csv.reader` + `path.exists()` audits in writer + loader modules; fixes for Rivendell `B0_base_hourly_rp` and `S08_co2cap_slice` regressions surfaced by the lift
- Phase C.5: slimmed `OrchestrationStep` with `keep_solutions` opt-in
- Phase F: parquet-bundle registry + `manifest.json` for hand-off between sub-solves
- Phase G: `process_outputs` disk reads migrated to in-memory kwargs
- Phase H: end-to-end verification pass

**FlexData Provider (Step 1 + Step 2 + Step 2.5)**
- Step 1-a → 1-g-7f: `FlexDataProvider` scaffolding; pilot migration of `_read_p_flow_max`; full migration of `input.py`, writer-side reads, `process_outputs` reads; Provider-exclusive writer population; per-writer `provider=` threading through `_writer_per_solve`, `_writer_period_calc`, `_writer_inflow_scaling`, `_writer_dispatchers`, `_writer_lp_scaling`, `_writer_chain_params`, `_writer_arc_unions`, `_writer_solve_time`, `_solve_context`, `_pdt_lookup`, `_derived_params`, `_derived_profile`
- Step 2a–c: deleted the seed funnel + transitional shims; rewired `--csv-dump` and removed the CSV-emission gate
- Step 2.5-E (Phases A+B+C): Provider through the timeline + averaged-timeseries cascade
- Meta-test for post-Step-2 cascade invariants
- Tighter `RAW_INPUT_FALLBACK_ALLOWLIST` after the disk-arm purge; meta-guard against new disk fallbacks creeping in

**Disk-fallback purge**
- Disk-fallback arms deleted from 12 writer modules: `_provider_open`, `_provider_lookup_positional`, `_derived_params`, `_invest_seeds`, `_dc_power_flow` + `_commodity_ladder`, `_derived_profile`, `_delay`, `_reserve`, `_group_slack`, `_derived_branch`, `_derived_existing`, `_timeline`, `_inflow_scaling`
- `_load_handoff_aux_pair` callers fixed; `path.open` retired in the same area
- `flextoolrunner/blocks` routes `write_block_data` through `_PATCH_MODULES`
- Test verification: workdir-empty smoke + `--csv-dump` round-trip

**`input_derivation` package**
- Skeleton package created; validators extracted to `_validators` submodule
- Ported derivations: `_write_process_method`, `_write_dc_power_flow_data`, commodity ladders + ladder_sets — all now in-memory
- `run()` entry point introduced; `write_input` rewired through it; `write_workdir_inputs` routes through `input_derivation` directly
- `flextoolrunner/preprocessing/*` and `flextoolrunner/input_writer.py` deleted (superseded)

**`spinedb_backend` package**
- Skeleton package + entity / parameter / default materialisers
- `_ENTITY_SPECS`, `_PARAMETER_SPECS`, `_DEFAULT_VALUES_SPECS` migrated onto `Backend.entities` / `Backend.parameter_values` / `Backend.parameter_defaults`
- Provider threaded into `write_input`, `_writer_leaf_sets`, `_writer_mid_sets`, `_writer_calc_params`
- `flextoolrunner` plumbs Provider through `solve_writers.write_all_branches`, averaged-timeseries, and `timeline_config.py` disk read

**In-memory region decomposition (item 2.6)**
- `flextoolrunner` region decomposition is now fully in-memory across all entry points

**Tests migrated to native cascade**
- Migrated: `test_commodity_ladder_smoke`, `test_commodity_ladder_rolling`, `test_years_represented`, `test_timings_csv`, `TestPGLibCase14Integration`, `test_cost_aggregation_semantics`, `test_non_anticipativity`, `test_lh2_three_region`, `test_obj_decomposition`
- Retired: `test_mps_parity.py` deleted (legacy MPS parity dead); `tests/emission/*` skipped (`flextool.mps` emission gone in Δ.22 — ported to polar_high in 3.38.0); perturbation/* skipped (harness incompatible — ported in 3.38.0); `test_solve_handoff.py` trimmed (9 retired PoC tests dropped)
- `test_integration_parquet_matches_csv_on_base_scenario` skipped (legacy parquet/CSV parity is no longer meaningful)

**Miscellaneous**
- Commodity ladder reads routed through the Provider
- `apply_npv` signature mismatch fixed; provider threaded
- `engine_polars/scaling`: geometric-midpoint `user_bound_scale` + 6-decade gate
- GUI: Debug checkbox passes `--debug --csv-dump` to scenario runs
- Dropped redundant `_native_leaf_set_override` in `_drive_cascade`
- `engine_polars`: ported `co2_max_total` + fixed `minimum_downtime` invest tightening
- `read_parameters`: densified `reserve_upDown_group_reservation` to LP domain
- `process_outputs` / `read_variables`: stream parts via `ParquetWriter` row-groups

---

## Release 3.36.0 (15.5.2026) — Output-writer hardening + LP determinism + Surface A audit

Hardening release. The `engine_polars` cascade is pinned to a deterministic HiGHS solve and the LP column/row ordering is now canonical, so re-runs and goldens are stable across `PYTHONHASHSEED` and across CI workers. Many `process_outputs` regressions surfaced during the polars-cascade adoption are fixed; golden CSVs that drifted on the glpsol→HiGHS transition are regenerated with REGEN_LOG documentation. Surface A loader audit lands 50+24+13 focused tests.

**LP determinism**
- `engine_polars/scaling`: pin HiGHS to deterministic simplex options
- Canonical LP column/row ordering via `add_var` / `add_cstr` wrappers
- Post-`unique()` set frames sorted before LP id assignment
- `engine_polars` keys `_all_steps` by per-roll solve name, not parent solve
- `tests/db_utils`: numeric columns cast to float in `round_for_comparison`
- `assert_frame_equal` gains `atol=1e-4` to absorb the rounding step
- `scaling._scale_cache` cleared between scenarios in tests
- Two scenarios' `time_budget` bumped to absorb the determinism overhead (incl. `wind_battery_invest_lifetime_renew_4solve` 4.0 → 6.0 s)

**Output writers (`process_outputs`)**
- `costs_discounted`: multi-roll inflation/lifetime values now correct
- `dtt` MultiIndex sorted by `(solve, period, time, t_previous)`
- Per-period `years_represented` threaded through `FlexData`
- `d_realize_invest` filtered to realized periods only; divest periods included
- `summary_solve.csv`: solve names natural-sorted; Investment discount factor written over `period_in_use`
- `VRE_potential` + `group_flows` CSVs restored after a regression
- `process__ct_method` built from `process_min_load_eff`
- `dt_realize_dispatch` / `d_realized_period` use `realized_dispatch`
- `unit_capacity__d.csv` lag in rolling solves fixed
- `(entity, period)` lookups filtered to `v.invest`'s actual index
- `par` / `s` unioned over `full_dt` for rolling/multi-solve
- Self-discharge multiply fixed on rolling/multi-solve scenarios
- `unit__{input,output}Node` wide-pivot columns sorted
- Both flow directions emitted for direct-method arcs
- `commodity_node` sets derived from FlexData `flow_*` frames
- `nodeGroupDispatch` arc-union sets backfilled for `group_flows`
- Sub-femto HiGHS LP residuals clipped in `out_flows` before CSV emit
- Native engine + `process_outputs` fix for nested invest+dispatch cascades
- Rolling-storage hand-off fix for non-forward-only storages
- Forecast-branch rows emitted when `output_horizon=yes`
- Reserve None-handling hardened; `process_reserve` column emission gated on scheduled reservation
- `calc_group_flows`: `.squeeze()` dropped — was crashing single-timestep solves
- `engine_polars`: detects period column under user-renamed `Map.index_name`

**Goldens regenerated (HiGHS-clean vs glpsol residuals)**
- 6 scenarios routed from retired glpsol to HiGHS
- `coal_co2_limit` goldens regenerated against alt-optima
- `cost-penalty` goldens regenerated (HiGHS-clean zeros vs glpsol residuals)
- `multi_year_wind_no_investment` `unit_capacity` golden regenerated
- 5 alt-optima goldens regenerated against canonical LP column ordering
- `test_a_lot` + `5weeks_battery_intraperiod_blocks` alt-optima goldens regenerated
- 3 + 4 `summary_solve.csv` goldens regenerated (HiGHS precision + chronological Investment-factor order)
- `hyphenated_entity_names` regen documented
- `tests/REGEN_LOG` documents Phase-2 candidate verification (1 regen + 5 flags)
- Test infrastructure handles two-row-header / `Unnamed:` CSV forms

**Surface A audit fixtures + tests**
- Base fixtures for loader / constraint / objective audit
- 50 focused loader tests for Surface A audit
- 24 model-run constraint tests for B.1–B.15
- 13 cost-term isolation tests for B.16–B.20
- `step_duration=3` derived-fixture regression guards
- Hand-computed `step_duration` LP objective tests
- `engine_polars/scaling`: two-sided guard on `user_bound_scale` recommendation
- `engine_polars`: cascade-vs-seed parity diagnostic

**`plot_outputs` performance**
- `_plot_simple_bars` vectorised; `FixedLocator` / `FixedFormatter` for bar tick axes
- `_sum_row_heights` vectorised
- Stacked & grouped bar paths vectorised
- Sub-pixel-wide bars skipped in draw path
- Single-level expand groups counted in pagination
- Stale on-disk plans invalidated via `schema_version`
- Phantom NaN rows/cols dropped after dim-rule pivot

**Other**
- `native_run_model`: write `timeline_matching_map.csv` from returned dict
- GUI: stable input-source numbering with orphan reuse + visual marker; source-number suffix always written in output folder names
- GUI: ResultViewer style overrides scoped to its own plot tree
- GUI: status / peak / timestamp columns widened in jobs tree
- `tests/scenarios` wired to `engine_polars` cascade

---

## Release 3.35.0 (13.5.2026) — Cascade memory + GUI font system + plot performance

Performance, GUI and developer-experience release on top of 3.34.0. Long cascade runs no longer balloon RSS across rolls; the GUI handles per-monitor DPI scaling and user-tunable font sizes; `dump_csvs` no longer writes 7 GB debug oracles by default. Includes preparatory work for the engine-wide dtype refactor (axis-vocabulary discovery + canonical `pl.Enum`s) — the end-of-load Enum sweep was activated and then parked in the same release as a precaution.

**Cascade memory**
- Scaling diagnostic gated to once per base solve — 38 % wall-clock speedup on large cascades
- Opt-in `tracemalloc` + RSS checkpoints to localise OOM phases (env-var gated)
- `libc.malloc_trim` called after heavy allocators — 38 % RSS drop
- Heap trimmed at end of each iter too (multi-roll heap accumulation)
- `dump_csvs`: skip 7 GB-scale CSVs by default; env var restores the debug oracle

**dtype refactor — preparatory**
- Phase 1+2: axis-vocabulary discovery + canonical `pl.Enum`s
- Phase 3+4+5 (partial): `load_flextool` end-of-load `pl.Enum` sweep — activated, then disabled in the same release as a guard while downstream fallout was understood
- Dtype-flexible scratch frames in `engine_polars` (preparatory for Enum activation)
- `cast_dim` helper + populated-branch alignment; `cast_dim` sweep across cascade (3 files + remainder)
- `process_outputs`: normalise enum/utf8 join keys in `_entity_all_capacity`

**Path B (rolling-handoff perf)**
- Rolling-handoff CSV reads deduplicated (Path B, Category A)
- `WriterSnapshot` for top-7 per-solve preprocessing (Path B, Cat B)
- `ctx` threaded through period/branch helpers + `_dt_period_active_steps` + `dtttdt`

**GUI font + DPI system**
- Phase 1: centralised font metrics in `ui_metrics.py`
- Phase 2: per-role named fonts via `setup_fonts`
- Phase 3: saved geometry + sash clamped on restore
- Phase 4: rescale saved positions by font metric on restore
- Phase 5: Reset window layout action
- Phase 6: em-ified dialog widths
- Phase 7 + 7c: user-tunable UI font size + UX size tweaks
- Phase 8: per-monitor DPI font rescale
- Header fonts bound to named fonts so live size changes propagate

**Other**
- `engine_polars`: respect `solve.new_stepduration` in `apply_derived_a`
- `engine_polars`: preserve legacy entity-level rows in `dump_csvs` slices
- `engine_polars`: native `p_inflow` override in `apply_derived_a` disabled (regression guard)
- `engine_polars`: end-of-load `FlexData → Enum` cast disabled (preparatory work parked)
- `tests`: `pbt_node_inflow` multi-`time_start` + parent-period fixtures (Phase C)

---

## Release 3.34.0 (12.5.2026) — Multi-solver support + documentation overhaul

This release ships the two user-facing additions on top of 3.33.0's engine swap: per-solve commercial-solver selection and a documentation refresh organised around the Spine Toolbox GUI as the primary FlexTool interface.

**Multi-solver support**
- New per-solve parameters (v52 migration) on the `solve` entity: `solver`, `solver_io_api`, `solver_options`, `solver_time_limit`, `solver_mip_gap`, `solver_threads`, `solver_log_level`. HiGHS remains the default — existing scenarios with no solver parameters are unaffected.
- Backends supported via [polar-high](https://pypi.org/project/polar-high/): **HiGHS** (default, ships with FlexTool), **Gurobi**, **CPLEX**, **Xpress**, **COPT**. Each commercial solver requires its own Python wrapper + license; FlexTool itself never imports the commercial wrappers and never inspects licenses (vendor's discovery path handles it).
- The three convenience knobs (`solver_time_limit`, `solver_mip_gap`, `solver_threads`) are normalised across solvers; FlexTool translates them to each backend's native parameter name. `solver_options` passes raw key/value pairs through unchanged.
- Cascade startup probes each available solver with a trivial 1-var LP and logs a per-solver license status line, e.g. `Solver license status: gurobi=licensed, cplex=licensed, xpress=licensed, copt=not-installed, highs=licensed`.
- Cold rebuilds in the cascade dispatch through `polar_high.solvers.solve(...)` and normalise the result via a new `LiteSolution` adapter so downstream output writers stay unchanged. Warm-start and Lagrangian decomposition remain HiGHS-only by polar-high design — selecting another solver on a warm cascade logs a warning and falls back to cold rebuilds; selecting another solver on a Lagrangian-decomposed scenario raises a clear configuration error.
- Per-solver documentation pages under `docs/solvers/` (install, licensing, common errors, how to set the solver in FlexTool).

**Documentation overhaul**
- Site reorganised around the Spine Toolbox GUI as the primary FlexTool interface; CLI flows are documented but no longer the default story.
- New developer guide and how-to recipes for the engine_polars-era codebase.
- `decomposition_method` parameter description tightened; superseded GUI screenshot removed.

**Bug fixes**
- `_writer_mid_sets.derive_commodity_node_co2` no longer crashes with `ComputeError: cannot compare string with numeric type (f64)` on customer DBs where `p_commodity.csv`'s value column is inferred as Float64 by polars. `_read_csv` now forces every column to Utf8 on read via `infer_schema_length=0`.
- `process_outputs.calc_storage_vre` no longer raises `KeyError` when `node_self_discharge_loss` is authored on nodes that have no `v_state` LP variable (supply-curve / commodity nodes). The self-discharge multiply is now restricted to the intersection of authored nodes and storage nodes.
- CLI cascade exception handler distinguishes `FlexToolUserError` (configuration problem — clean message, exit 1) from other exceptions (real flextool bug — full traceback). Users hitting unknown-solver / missing-license errors no longer see a stack trace.
- `update_flextool` now refreshes declared dependencies for editable installs too — `pip install -e .` with new core deps in `pyproject.toml` would previously be missed, causing `ModuleNotFoundError` on the next solver invocation. The `--upgrade` is now passed without `--upgrade-strategy=eager` so transitive dependencies aren't churned.

**Performance**
- Spine DB read: `SolveConfig.load_from_db_url` and `TimelineConfig.load_from_db_url` now pre-warm with `db.fetch_all("entity")` + `db.fetch_all("parameter_value")` before any `find_*` call, mirroring the legacy `FlexToolRunner.__init__` pattern. Measured 5.5–6.4× speedup on large customer DBs (~1.8 s saved per cascade run); sub-MB test DBs see no change.

**Other**
- v52 schema migration renames the legacy `solver` value list to `solvers` in place: members expanded from `[glpsol, highs, cplex]` to `[highs, gurobi, cplex, xpress, copt]`; pre-existing `solve.solver = "glpsol"` values are rewritten to `"highs"` (GLPK retired in Δ.22). Parameter_definition foreign key is preserved through the rename so no data is lost.

---

## Release 3.33.0 (12.5.2026) — `glpsol` retired; HiGHS via `polar-high` becomes the sole LP backend

This release closes out the largest architectural change since FlexTool went open source: the GLPK/GMPL pipeline that built the LP via `glpsol`, wrote MPS files to disk, and re-loaded them into HiGHS is gone.  The new path builds the LP in process via [`polar-high`](https://pypi.org/project/polar-high/) on top of the [`polars`](https://pola.rs/) DataFrame engine, hands it to HiGHS through `highspy`, and stays in memory for the entire solve → handoff → output-writer chain.

There are no binary build artefacts left to ship: `glpsol` is gone, the `bin/glpsol*` binaries that the previous releases bundled have been deleted, and the `flextool.mod` GMPL model file is deleted.  FlexTool is now a pure-Python install — `pip install flextool` is sufficient.

The version is held at 3.x rather than bumped to 4.0.0 until the new engine reaches full parity with the old one across the test suite.  A 4.0.0 semantic bump will follow once that bar is met; this release ships the backend swap behind the existing 3.x API surface.

**Breaking changes**
- `bin/glpsol*` removed; `flextool/flextool.mod` and `flextool/flextool_base.dat` removed
- `--engine=gmpl` hard-rejected with a clear retirement banner
- Legacy GMPL-pipeline CLI flags removed: `--use-old-raw-csv`, `--ipm`, `--auto-scale`, `--relax-feasibility`, `--glpsol-timing`, `--report-near-duplicates`
- The stubbed `flextool/flextoolrunner/lagrangian.py` (legacy GMPL coordinator) is deleted; `--decomposition lagrangian` now drives the native polars coordinator (see below)
- `--highs-threads` accepted as a no-op stub for GUI/Toolbox compat; the native cascade is single-threaded

**New solver backend (`engine_polars`)**
- Whole-system `FlexData` is built directly from a Spine DB or pre-staged workdir CSVs; the LP build, solve, and output handoff all happen in memory through `engine_polars.run_chain_from_db`
- HiGHS is consulted via `polar-high.Problem` / `WarmProblem` / `LagrangianProblem`; no MPS roundtrip
- New experimental `--fast-single-solve` CLI path for simple single-solve workloads — bypasses `write_input` entirely and reads inputs from Spine via `SpineDbReader`
- Native `engine_polars/_writer_*` ports of 150+ preprocessing sets and calculated parameters (Writer Phase 1-4: `L0-L9`, follow-ups 1-8, closeout, Phase 2 sub-dispatches 1-8, Phase 3 cascade adoption, Phase 4 Gap F handoff)
- Automatic LP scaling — analyser, scaling-report, two-sided cost-band guard, geometric-centering fallback, objective and bound scaling
- Δ.31: in-memory `FlexData` + `solution` threaded into `process_outputs.write_outputs` so the output writers no longer round-trip through `solve_data/*.csv`

**Spatial Lagrangian decomposition — native rewire**
- `--decomposition lagrangian` CLI rewired onto `engine_polars._lagrangian.solve_lagrangian`: per-region builds via `LagrangianProblem`, damped subgradient outer loop, primal averaging on the cross-region pipeline flows
- Smoke-test coverage on the LH2 three-region fixture pins the CLI contract within a 2 % gap-to-monolithic tolerance

**Flex-temporal decomposition** (carried over from `new-outputs` 3.29.0, fully integrated)
- Per-entity temporal blocks make mixed-resolution dispatch possible (hourly power + daily hydrogen in the same solve)
- `v50` + `v51` migrations land `solve.new_stepduration` and group-level `new_stepduration` + `decomposition_method`
- Constraints made block-aware: storage, conversion, flow capacity, DC flow, UC, ramp, profile, reserve; node balance generalised via overlap-set aggregation
- Output writers expand coarse-block variables back onto the fine timeline for the user

**Test infrastructure**
- Layer-1/2/3 test pyramid scaffolded, with golden objectives pinned on 10 scenarios and per-scenario timing budgets
- `@pytest.mark.solver` and `@pytest.mark.smoke` markers
- GitHub Actions CI on the smoke tier
- LH2 three-region JSON fixture + per-fixture native parity tests
- MPS-parity harness retired alongside the GMPL pipeline

**Packaging / installation**
- All binary dependencies retired — FlexTool is now a pure-Python install
- `polar-high` is now a core dependency (was an optional `engine-polars` extra)
- `pip install flextool` is sufficient on Linux, macOS, and Windows (HiGHS arrives via the `highspy` wheel)
- PyPI release in preparation (see `specs/pypi_release_checklist.md` for the remaining steps)

---

## Release 3.32.0 (5.5.2026)
**Bug fixes**
- Excel template link-sheet retention: drop link sheets where the link's own class has no surviving data (e.g. `connection_node` when `constraint` is unselected)
- `group.{include_stochastics, new_stepduration}` retagged from `model`/`timeline` to `solve_advanced`
- `timeset.{timeset_weights, representative_period_weights}` retagged to `solve_advanced`
- Quadratic linear scan in `blocks::derive_blocks::_assign_entity_to_group` replaced with a precomputed `(entity → group)` lookup

**New features**
- `--groups` CLI flag on `cmd_export_to_tabular` to restrict the Excel template to a parameter-group set; sheets with no surviving columns are dropped
- GUI parameter-group picker on "Add empty FlexTool input Excel" — DB-driven checkbox tree, required groups (`timeline`, `model`, `solve_basics`, `basics`) highlighted with hover tooltip; theme-aware colours
- Default-value row in v2 Excel sheets where any parameter on the sheet has a default (Map/Array defaults stringified)
- Representative-period clustering input drops scalar-valued profiles/inflows

**Note** — `new-outputs` also landed four hot-path optimisations on the legacy `flextool/flextoolrunner/preprocessing/*` writers (`is_static` fast-path on `PdtLookup`, sparse-emit + `default 0` on the six dense `pdt*` CSVs, `pandas.read_csv` swap in `_read_pdt_at_param`, sparse iteration in `pssdt_varCost_*`, the O(N²×T²) → O(N + |pairs|) fix on `node_inflow_scaling_params.has_node_time_inflow`, and side-node indexing for `process_arc_unions::write_node_group_dispatch_sets`).  Those edits target a code path that the `engine_polars` writer port (Writer Phase 1-4) is retiring, so they were not carried across in the merge — the performance lessons are preserved as actionable items for the native polars writers in `specs/sparse_writer_lessons_for_engine_polars.md`.

## Release 3.31.0 (4.5.2026)
**Bug fixes**
- Stochastic per-branch profile values previously lost in `pdtProfile` are restored
- Plot dataframe fragmentation in dispatch column-alignment fixed
- Dispatch view reads group flags from `nodeGroupDispatch` (not `nodeGroupIndicators`)
- `solve.timeline_hole_multiplier` now declared in the v44 schema (default 1.0)
- `db_migration` v42: robust legacy value-list cleanup via `find_parameter_value_lists`
- Migration steps tolerate `NothingToCommit` so a second run doesn't abort the chain
- RP clustering input: scalar-valued profiles/inflows dropped

**New features**
- `mod → Python` preprocessing migration: ~150+ sets and calculated parameters previously declared in `flextool.mod` now live under `flextool/flextoolrunner/preprocessing/`, written per-solve to `solve_data/`
- Migration scaffolding: phase-0 inventory + MPS parity harness + lint rules + DAG of derivations
- 70 incremental migration batches with MPS parity verified at 7-sig-fig precision across multiple baselines (rolling, multi-solve, contains_solve, h2_trade, test_a_lot)
- Per-class param taxonomy in `_param_taxonomy.py` (PROCESS_TIME_PARAM, NODE_PERIOD_PARAM, etc.)
- Order-determinism lint rule (`tests/test_preprocessing_ordered_set_lint.py`) — bare `set()` / set-literals / set-comprehensions blocked in `preprocessing/`
- `templates/examples.sqlite` rebuilt from `tests.json` and migrated to v51

## Release 3.30.0 (28.4.2026)
**Bug fixes**
- `read_highs_solution`: strip GLPSOL quotes from ISO 8601 timestamps in parsed variable names
- `handoff_writers`: load resolved-set CSVs from `solve_data/` (Phase A follow-up)
- `plot_canvas`: cap rendered DPI to keep figure pixmaps under X11's limit
- `RP`: representative-period indices sorted into chronological order
- `plot_bars`: bar thickness constant per row regardless of bar count
- GUI memory-watchdog: kill only when BOTH free-RAM reserve AND swap allowance are breached
- GUI: refuse to open the same input source twice (flash row red)

**New features**
- `result_viewer` Phases B–E: single rebuild path with filter-only tree toggles, per-scenario availability union, lazy plan-parquet union, comparison-only plans
- GUI per-job memory budget + watchdog + admission control + status bar
- GUI: drag-to-reorder + Alt+Up/Down on result-viewer scenarios tree
- GUI: shared `CheckTreeController` for every check tree + generation tokens for comparison-data updates
- GUI: View button always shown; opens single mode focused on the scenario
- GUI: persist per-variant duration with template default + clamp-safe save
- GUI: robust delete + prune dangling scenario state from `settings.yaml`
- `.mod`-resolved set files moved from `input/` to `solve_data/`; `set_` prefix dropped
- `--glpsol-timing` diagnostic flag for per-constraint matrix-gen timing
- `nodeBalance_eq` and `conversion_indirect` use tuple binding instead of overlap equality-filters

## Release 3.29.0 (24.4.2026)
**Bug fixes**
- v50 item-lookup bug fixed while restoring `unidirectional` in the master template
- `export_to_tabular` whitelists synced with master template schema
- Lagrangian: defensive name-map consistency check per iteration

**New features**
- **Flex-temporal decomposition**: per-entity temporal blocks for mixed-resolution dispatch (e.g. hourly power + daily hydrogen in the same solve)
- v50 migration: `new_stepduration` moved from `timeset` to `solve`
- v51 migration: group-level `new_stepduration` + `decomposition_method`
- `blocks.py`: derive per-entity blocks + overlap set + block predecessors + boundaries
- Generalised node balance via overlap-set (M-matrix) aggregation; constraints made block-aware (storage, conversion, flow capacity, DC flow, UC, ramp, profile, reserve)
- Output writers expand coarse-block variables back to the fine timeline for the user
- **Spatial Lagrangian decomposition**: regional input filter (cross-region flows as half-flows), `lagrangian.py` scaffolding + subgradient loop with primal recovery, CLI flag + docs
- LH2 three-region fixture + golden integration test (Lagrangian vs monolithic)
- `HighsModelHandle` persistence helper for repeated subgradient solves

## Release 3.28.0 (24.4.2026)
**Bug fixes**
- Default HiGHS to serial simplex (`parallel=off`, `threads=1`) to avoid non-determinism and occasional stalls on small models; defaults reinforced in `solver_runner` as belt-and-suspenders
- Restore 0.05 `discount_rate` default and related advisory defaults that had drifted
- Revert slack primary+escape split — back to single-variable slacks with penalty acting as a valve

**New features**
- HiGHS upgraded to 1.14 (`highspy>=1.14`)
- Auto-scaling of LP/MIP numerics: row scaling via `node_cap`/`group_cap`, objective scaling, and bound scaling with diagnostics printed
- `--auto-scale` CLI flag that gates objective/state auto-apply behaviour
- `--highs-threads` CLI flag
- HiGHS `mip_detect_symmetry` enabled — ~15× speed-up on unit-commitment with identical units
- `virtual_unitsize` documented as a speed lever for UC with identical units
- `unidirectional` transfer method for connections
- Cost-aggregation semantics fixed — weighting factors applied per variable class
- Divest salvage included in objective; `pre_existing` renamed to `all_existing`; storage valuation documented
- Auto-seed missing `output_info`, `output_settings` and `comparison_settings` databases on first run
- Seed `bin/highs.opt` from tracked template on first run
- `scaling_benchmark` harness relocated to `benchmarks/scaling`; unit tests moved to `tests/`
- `min_load_efficiency` section-term test promoted from xfail to passing

## Release 3.27.0 (22.4.2026)
**Bug fixes**
- Annualize `nodeGroup_flows_d_gpe` to match the `node_d_ep` convention
- Column misalignment in `ed_lifetime_fixed_cost[_divest].csv` writers
- Harden axis-manifest override against `None` active_scenarios
- Sum only over `stack_levels` in shared axis-bound resolution
- Reserve provision with a timeseries reserve requirement
- Reserve provision active only when there is demand
- Delete dead plot functions

**New features**
- Parameter groups — every parameter now assigned to a group; "Outputs" renamed to "output"
- Parameter-group colours re-tuned for readability in both light and dark theme
- Colour template infrastructure for plots: category-based colouring (costs, node_flows) and entity_class.group colouring (flowGroup)
- Cross-scenario axis-bounds manifest — viewer reads shared bounds so y-axes stay stable across scenarios
- Per-scenario axis manifest with subset filter
- Composite colour lookup for `nodeGroup_flows` plots
- Colour template plumbed into the batch render path
- Plot plan JSON cleanup (redundant timestep fields dropped)
- Group-output parameters renamed; `output_results` split; flow-group indicator stub added
- Alphabetical sorting of subplots
- Periods ordered left-to-right in vertical p-variant bar plots

## Release 3.26.0 (21.4.2026)
**Bug fixes**
- Negative-remaining infeasibility in cumulative price-ladder cap
- Bug in `realized_invest` uniqueness
- Parameter_type_list corrections across migrations and master template

**New features**
- Commodity price ladder — `price_method`, `unitsize`, `price_ladder` parameters for commodities
- `v_trade` variable (MWh × unitsize) with tier caps and objective routing
- Rolling cumulative-quota handoff for `price_ladder_cumulative` — per-period accumulators preserve remaining quota across rolls
- Two-roll cumulative-ladder validation scenario
- Rolling-aware `co2_max_total` with per-period accumulators
- Node-type consolidation aligned with the ladder work

## Release 3.25.0 (20.4.2026)
**Bug fixes**
- Align `fix_storage_*.csv` header between init and phase 3
- Carry fractional tail in `write_years_represented` (tested at R=2.5)
- `canonical_sort` sorts stably by `solve_pos` only, preserving within-solve order
- Drop `_round_to_sig` from the parquet read path
- Reindex `extract_variable` output to canonical phase-1 row order

**New features**
- Direct HiGHS → parquet extraction replaces the phase-3 glpsol reader round-trip for solver outputs and solve handoffs
- Phase-3 glpsol retirement — derived-parameter printfs moved pre-solve; readers repointed at `input/` and `solve_data/`
- Canonicalise row order by solve-creation order across all readers
- `has_balance`, `has_storage` and `node_type` consolidated into a single `node_type` enum
- Parquet `VariableSpec` column names aligned with CSV readers
- `empty_variable_frame` helper for same-shape empties

## Release 3.24.0 (19.4.2026)
**Bug fixes**
- `storage_state_start` first-timestep semantics corrected for cyclic bindings
- Goldens regenerated for the sink-flow coefficient flip

**New features**
- `coefficient` on capacity constraints split into `flow_coefficient`, `max_capacity_coefficient`, `min_capacity_coefficient`
- Sink-side `flow_coefficient` flipped from division to multiplication — both sides of the balance now use the same convention
- New `pre_built` capacity-coefficient term alongside the invested term
- Growth-cap recipe documented
- Lifetime-aware dispatch capacity bound
- `max_flow_for_unconstrained_variables` model parameter replaces the previous 1e6 literal
- CO2 duals exposed; `rp_cost_weight` threaded into constraints; horizon-vs-annual outputs distinguished
- `timeset_weights` wired through the runner (populates `rp_cost_weight.csv`)
- `no_investment` handled as a `lifetime_method`

## Release 3.23.0 (17.4.2026)
**Bug fixes**
- Pre-processing now also updates the solve `period_timeset`
- Step duration correctly applied in period aggregations of flow-derived outputs
- Reserve Excel handling fixed

**New features**
- Greedy convex-hull clustering for representative-period selection
- Representative-period storage binding via `bind_using_blended_weights` (renamed from `bind_using_rp_weights`)
- `bind_intraperiod_blocks` storage binding method for LTS-style intra-period blocks (blocks defined by gaps in timeline indices)
- Multi-year wind `no_investment` scenario test and goldens
- Base-weighted scenario test and goldens
- Delays for units and connections (constant or map)
- `5weeks_battery_intraperiod_blocks` scenario test and goldens

## Release 3.22.0 (7.4.2026)
**Bug fixes**
- Parquet loading strips the scenario column level
- `savefig` works for `Figure()` objects not registered with `pyplot`
- Threading errors from `plt.figure()` replaced with `Figure()` usage
- Dispatch colors and `nodeGroup` filtering
- Cancel pending `draw_idle` to prevent redundant re-renders
- Numerous canvas / toolbar jitter fixes (freeze/thaw around draws, layout locking)
- Legend clipping, tree navigation, canvas clearing
- Stacked-area plot axis limits
- Empty `years_represented` for rolling dispatch solves
- CO2 emissions parquet key mismatch
- Scenarios without connections

**New features**
- `PlotPlan` — pre-computed plot plans saved to disk
- Threaded figure building with prefetch cache and memory-based GB cache limit
- Time-series downsampling via `tsdownsample` with a numpy fallback
- Network graph visualisation from the Spine database
- Cross-scenario dispatch y-limits and column consistency
- Comparison-mode parquet pipeline
- Availability manifest for three-level variant display
- Redesigned three-level variant navigation (Shift+Up/Down to next row with data at focus column)
- `PlotPlan` unit tests
- YAML config restructured with entry names as top-level keys
- Per-variant `plot_name` and explicit `variant` field in configs
- Dispatch mode in Result Viewer
- `combine_scenario_parquets` and comparison checkboxes in the viewer
- Parquet file-size reduction by dropping DataFrame multi-index bloat

## Release 3.21.0 (1.4.2026)
**Bug fixes**
- Handle scenarios without connections
- Connection output for `method_2way_1var` (DC power flow)
- Guard against all-NaN values in plotting
- Reserve provision active only when there is demand for the reserve
- Mac glpsol binary naming and selection across platforms
- Fixed and cleaned-up test suite

**New features**
- DC power flow B-θ formulation in the GMPL model
- Method resolution moved from GMPL to Python
- DC power flow data pipeline and database migration
- Output processing and parquet passthrough for DC power flow
- MATPOWER import CLI wrapper
- PGLib-OPF IEEE 14-bus integration test
- 24-test DC power flow pytest suite
- glpsol built for macOS, Linux and Windows via GitHub Actions (including macos-15 arm64)

## Release 3.20.0 (29.3.2026)
**Bug fixes**
- Excel migration edge cases
- Direct execution from Excel (`.xlsx`) corrected
- GUI updated to handle different input Excel versions

**New features**
- FlexTool 2.0 importer — reads old FlexTool 2.0 Excel `.xlsm` files into a Spine database
- Import of sensitivity scenarios from FlexTool 2.0
- CLI entry point for FlexTool 2.0 sensitivity import
- Version-aware importer that handles older FlexTool 3.0 Excels as well
- `invest_method` support in the FT2 importer
- `_inflow` renamed to `_node`; `_storage` used where applicable
- `base` used instead of `Base` as the default alternative
- Stochastic sheets in Excel
- Improved Excel read/write roundtrip, including DB-to-Excel export
- Dialog to migrate Excel files
- Single source of truth for the FlexTool input DB version (`flextool/update_flextool/__init__.py`)

## Release 3.19.0 (14.3.2026)
**Bug fixes**
- Execution runs from the FlexTool root with isolated work folders
- Comparison Excel written to the correct directory
- All scenarios no longer produce identical results
- Output-actions column order, drag-select and spinbox sync
- Foreground-color crash on start
- Several threading and layout issues

**New features**
- Result Viewer window with scenario listbox, plot tree and variant panel
- `PlotCache` with memory-based GB limit
- `PlotCanvas` with PNG display wired into ResultViewer
- Dark mode with `sv_ttk` and a theme-toggle radio
- Font-relative and DPI-aware GUI sizing
- Plot-settings dialog with YAML config parsing
- Execution menu window with subprocess pool and parallel workers
- Database version checking and auto-upgrade from the GUI
- Custom file picker with last-modified column and sorting
- Visual feedback: green/grey/red highlights, boxed outputs, auto-check
- Keyboard shortcuts (including `Ctrl-A` to select all) and rearranged move arrows
- DB-editor integration with process tracking
- `--work-folder` support for parallel scenario execution
- `--parquet-base-dir` for comparison without a database
- `output_db_url` optional in `cmd_run_flextool`
- Dual variant state in the viewer: desired (dashed) + shown (solid)
- Animated hourglass spinner on output-action buttons
- Persisted checkboxes, View button, sizing adjustments
- Parent-directory button in the file-picker dialog

## Release 3.18.0 (27.2.2026)
**Bug fixes**
- Fix reading input files in `run_flextool.py`
- Many small issues surfaced by the refactor fixed so all examples execute

**New features**
- Major refactoring into deep modules — package structure reorganised for separation of concerns
- New subdirectories to keep the FlexTool root uncluttered
- Initialisation now also creates settings databases
- Updated `update_flextool.py`
- Walked back global `flextool` CLI commands — they do not work with multiple installs
- Added `output_info_template.sqlite`
- `flextool_location.txt` for runner discovery

## Release 3.17.0 (14.1.2026)
**Bug fixes**
- Float-to-string handling in the import spec
- Node-summary column alignment
- `nodeBalancePeriod` uses `pdtNodeInflow` (sum over period) instead of `annual_flow` (`annual_flow` should not carry a sign)
- Rolling-model fixes including `p_roll_continue` handling

**New features**
- Excel import specification
- Direct Excel-to-DB read pipeline
- Calamine engine for faster spreadsheet reading
- Full FlexTool execution from Python — no Toolbox required
- Removed data checks from later solves (they belong in the first solve)

## Release 3.16.0 (5.12.2025)
**Bug fixes**
- Fixed costs split off from investment costs and their calculation corrected
- `years_represented` with a discount factor >1 now calculated correctly
- Timestep durations checked to be positive and non-zero

**New features**
- Speed-up in Python output writing
- Result list refactored
- Old `write_outputs.mod` glpsol output code removed from `flextool.mod`
- Further plotting enhancements and minor refactoring
- `solve_progress.csv` with timings and scenario name
- Vertical plots, sums and means, better subplot spacing
- `write_outputs` can be used to plot a single ad-hoc time series

## Release 3.15.0 (2.12.2025)
**Bug fixes**
- `costs_discounted.csv` calculation for multi-year models
- Empty sets and rolling-model crashes
- Negative capacity units handling in outputs
- Inertia calculation in outputs
- Missing `fix_storage_time_lists`
- Several output-processing bugs (ramp, investment, storage, group results)

**New features**
- Python post-processing pipeline reaches parity with the old glpsol `write_outputs.mod` — all result CSVs replicated except node ramp envelopes
- Scenario-comparison specification (`scenario_comparison.json`) and combined-scenario results framework
- Removed redundant `fix_storage` constraint that used `dt_realized_dispatch`
- First ugly-duckling plotting on top of the Python pipeline

## Release 3.14.0 (6.8.2025)
**Bug fixes**
- Fixing problems with group results (wrong signs)
- Fix broken importer
- Fix calculation of VRE shares in the results
- Other operational costs for units had gone missing in 3.12.0 (22.5.). Now they are back.

**New features**
- There is a new migrate button in the workflow to update the active input database to the latest version.

## Release 3.13.0 (27.6.2025)
**Bug fixes**
- Added missing fixed costs to operational models
- Catching infeasible models better

**New features**
- Time structure has been simplified, see ´https://irena-flextool.github.io/flextool/reference/#how-to-define-the-temporal-properties-of-the-model'
- Changed the time structure to have timeset instead of timeblockSet
- Added periods_available for the model entity, to have periods in the data that are not used without domain errors
- Added assumptions about the model timestructure to ease the defining of it.

## Release 3.12.0 (22.5.2025)
**Bug fixes**
- Decimal points in node_balance_t were causing some grievance.
- Issues with lower case / upper case filenames when importing results
- Added missing parameters inertia_constant, ramp_cost, ramp_speed back to the model (caused by release 3.10.0)
- Fixed a output processing crash in a rolling model without VRE

**New features**
- Delays for units and one-way connections. New parameter delay, measured in time-steps (be careful), which can be a constant or a map of (integer timesteps, weight) where the weights should add to one. The map allows to spread the delay like it might happen in river systems.
- An order of magnitude speed-up in reading inputs

## Release 3.11.0 (4.3.2025)
**Bug fixes**
- Preventing crashes on curtailment calucaltions to units without capacity and MIP stochastic solving
- Correcting nested storage state passing

**New features**
- Timeseries prices for commodity and co2

## Release 3.10.0 (31.1.2025)
**Bug fixes**

- Prevent HiGHS from hanging after founding a solution
- Fix profile limits (issue when efficiency != 1)
- Made to work with new Toolbox DB API

**New features**

- New HiGHS versio (1.9.0)
- Script to see the differences in outputs (or inputs) between two model runs (meant for testing)
- FlexToolRunner.py reads input data directly from DB (speed-up)


## Release 3.9.0 (28.11.2024)
**Bug fixes**

- Fixing Excel input template

**New features**

- Updated documentation to match Toolbox changes
- New HiGHS version (1.8.1), should help with model getting stuck.
- Reorganising file structure to remove clutter from FlexTool root
- Some name changes in the workflow
- FlexTool has requirements.txt and pyproject.toml (allows pip install -e ., but not in pypi)
- Improved installation instructions (switched from miniconda to venv)
- Linux (Ubuntu tested) should work out of the box (but FlexTool execution only in work directory one scenario at a time!)


## Release 3.8.0 (29.8.2024)
**Bug fixes**

- Excel input file to use entity_alternative
- unit_curtailment_share fix to Excel input

**New features**

- Existing capacity can be set for periods
- Cumulative limits for investments in nodes, units and connections
- New node_type that has period limit for inflow
- New highs solver version
- All parameter now have valid types. These are not enforced, but Spine Toolbox highlights parameters with a wrong type.


## Release 3.7.0 (24.6.2024)
**Bug fixes**

- Several bugs related to the migration to Spine Toolbox 0.8.
- CPLEX call without pre_command fixed
- Timeline aggregarion without all timesteps
- User constraint including capacity fixed to include unitsize
- User constraint including flows fixed to include step_duration
- Creating multiple invests with 2D maps fixed
- Update_flextool.py/ migrate_database.py crashing if nothing to commit
- Nested structure:
  - Pass end state only from dispatch level
  - Storage fix_quantity overrides other storage options

**New features**

- Added non-anticipatory constraint to stochastic branches. The brnaches now start from the first timestep.
- Storage_reference_value: Now possible to set seperately for multiple end times as time step map.
- Nested structure storage_nested_fix_method:
  - Added fix_usage
  - Fix_price changed to set the price to the objective function of the lower level solve


## Release 3.6.0 (15.5.2024)
**New features**

- Upgraded FlexTool to Spine Toolbox v0.8
- Entity alternative replaces is_active parameter


## Release 3.5.0 (29.4.2024)
**Bug fixes**

- Stochastic and rolling weights


## Release 3.4.0 (13.4.2024)
**Bug fixes**

- Inertia with MIP units added missing unitsize
- Result ordering improvements
- Fixing how flow variables are limited
- Min_load to work with multiple inputs/outputs¨
- Capacity limit includes the efficiency

**New features**

- Output of inertia seperately for units
- Default penalty values for reserve, inertia, capacity margin and non-synchronous limits
- Large model data processing speed-ups


## Release 3.3.0 (23.3.2024)
**Bug fixes**

-	Investments: maxState to only sum over the investments in this node, not all the nodes
-	Ramp constraint now correctly calculates the investments done in previous periods
-	Min and Max Invest/Divest constraints to only affect investment periods
-	Efficiency works with profile units
-	Allow upward and downward slacks to be more than the inflow
-	Potential VRE generation now includes efficiency and availability
-	Non-sync group constraint now excludes the flows inside the group
-	Two-way MIP connection to allow only one direction at a time
-	Connection variable cost to the objective function
-	Allow investing with just `invest_periods` without `realized_periods`
- unit_inputNode: `coefficient` applied correctly to units with multiple inputs

**New features**

- Two-stage stochastic modelling of future uncertainty
-	Better infeasibility and required parameter checks to inform the user what is wrong with the model
-	Small changes to some result data parameters
-	Added availability (constant /timeseries) parameter for entities
-	Several outputs to be optional to hasten the data transfer.
-	Added group flow output parameter `output_aggregate_outputs` to give the flow of the desired nodes in a grouped format
-	Option for loss of load sharing `share_loss_of_load` between nodes in a group to better describe the system where loss of load is present
-	A workflow item to display the summary file from model runs
-	Database with pre-made time settings for the users to start with *time_settings_only.sqlite*
-	C02 max total costraint
-	Default values for:
    -	Penalty values
    -	Efficiency
    -	Constraint constant
    -	Reserve reliability

**Documentation**

-	Introduction
-	Improved tutorial (including how-tos split of into their own section)
-	Installation / update instructions
-	Theory slides
-	How to section to guide on making specific parts of the model

Building parts of the model:

-	How to create basic temporal structures for your model
-	How to create a PV, wind or run-of-river hydro power plant
-	How to connect nodes in the same energy network
-	How to set the demand in a node
-	How to add a storage unit (battery)
-	How to make investments (storage/unit)
-	How to create combined heat and power (CHP)
-	How to create a hydro reservoir
-	How to create a hydro pump storage
-	How to add a reserve
-	How to add a minimum load, start-up and ramp
-	How to add CO2 emissions, costs and limits
-	How to create a non-synchronous limit
-	How to see the VRE curtailment and VRE share results for a node

Setting different solves:

-	How to run solves in a sequence (investment + dispatch)
-	How to create a multi-year model
-	How to use stochastics (represent uncertainty)

General:

-	How to use CPLEX as the solver
-	How to create aggregate outputs
-	How to enable/disable outputs
-	How to make the Flextool run faster


## Release 3.2.0 (23.8.2023)
**Bug fixes**
- Storage state, unit flows and ramps were not properly limited in multi-period investment models.
- Removed limits from vq_state (they were causing infeasibilities in some case).
- Added time changing efficiency to profile based flows.
- Flows from node to unit in unit__node are marked negative (were positive).

**New features**
- *Breaking change*: `realized_periods` (solve objects) is supplemented with a `realized_invest_periods` so that the user can state from which solves the results should take investment results and from which solves dispatch results.
- New parameters for solve objects: `solve_mode`, `rolling_duration`, `rolling_solve_horizon`, `rolling_solve_jump` and `rolling_start_time` that enable to build rolling window models. See [How to use a rolling window for a dispatch model](how_to.md#how-to-use-a-rolling-window-for-a-dispatch-model).
- New parameter `contains_solves` that enables nesting solves inside solves (e.g. to calculate shadow values for long term storages or to implement rolling dispatch inside a multi-year investment model). See [How to use Nested Rolling window solves (investments and long term storage)](how_to.md#how-to-use-nested-rolling-window-solves-investments-and-long-term-storage).
- New outputs: For groups: `VRE_share_t`. For unit__nodes (VRE units): `curtailment_share`, `curtailment_share_t`.
- Name changes to outputs: `flow` to `flow_annualized`, `sum_flow` to `sum_flow_annualized`.
- Add migration for results database (parameter descriptions can be migrated).
- Documentation updates: new how-to sections.
- Faster outputting of ramp results.
- update_flextool.py will leave the Toolbox project untouched.
- Disable execute project button from the FlexTool workflow.

## Release 3.1.4 (14.6.2023)
**Bug fixes**
- Cancelling of plot_results will not freeze Toolbox

**New features**
- Lifetime_method (reinvest_automatic and reinvest_choice) so that user can choose whether assets will be automically renewed at the end of lifetime or the model has to make that choice
- Commercial solver support (CPLEX)
- Database migration. init.sqlite and input_data.sqlite will be updated to the latest version when update_flextool.py is run. It is also possible to update any database using migrate_database.py. After `git pull`, run `python -m update_flextool.py`. In future, just `python -m update_flextool.py` is sufficient. Best to update Spine Toolbox before doing this.

## Release 3.1.3 (1.6.2023)
**Bug fixes**
- Use of capacity margin caused an infeasibility
- Certain inflow time series settings crashed the model (complained about ptNode[n, 'inflow', t])
- Investments did not consider lifetime correctly - units stayed in the system even after lifetime ended
- Lifetime is now calculated using years_represented
- Fixed how retirements work over multiple solves

**New features**
- Added support for commercial solvers (CPLEX explicitly at this point)


## Release 3.1.2 (23.5.2023)
**Bug fixes**
- Node prices were million times smaller than they should have been (bug introduced when scaling the model in 3.1.0)
- Non-synchronous limit was not working.

**New features**
- Documentation structure was improved
- Model assumes precedence between storage_start_end_method, storage_binding_method, and storage_solve_horizon_method (in that order)


## Release 3.1.1 (13.5.2023)

**Bug fixes**
- The calculation of the p_entity_max_capacity did not consider which investment method was actually active. It also limited capacity in situations where investment method was supposed to be 'invest_no_limit'.


## Release 3.1.0 (13.5.2023)

**Bug fixes**
- Division by zero capacity for units with cf
- Output fixed costs for existing units
- Fix output calculation for annualized fixed costs
- Fix error in importing CO2_methods for groups
- Add other_variable_cost for flows from unit to sink

**New features**
- New investment method with a limited horizon (model used to assume infinite horizon)
  - *Braking change*: new parameter 'years_represented', which replaces 'discount_years'. It is a map to indicate how many years the period represents before the next period in the solve. Used for discounting. Can be below one (multiple periods in one year). Index: period, value: years.
- Added sidebar to documentation (gh-pages branch)
- Default solver changed from GLPSOL to HiGHS
- Documentation improvements
- Scaling of the model resulting in very big performance improvement for linear problems
- Improved plotting for the web browser version
- In Spine Toolbox, plotting is based on the specification from the browser version
- More results: group flows over time
- More results: costs for whole model run
- More result plots

## Release 3.0.1 (17.3.2023)

**Bug fixes**
- CO2_method for groups was not imported correctly from Excel inputs
- Fix other_variable_cost in units for flows from unit to sink (source to unit was working)

**New features**
- Output fixed costs for existing units
- Added outputs to show average capacity factors for periods

## Release 3.0.0 (11.1.2023)

All planned features implemented



TEMPLATE

## Release (dd.mm.yyyy)

**Bug fixes**
- foo

**New features**
- bar
