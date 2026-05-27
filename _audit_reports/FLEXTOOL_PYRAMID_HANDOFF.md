# FlexTool handoff: tame the FlexData-build memory pyramid

**Target repo:** `flextool-engine` (`/home/jkiviluo/sources/flextool-engine/`)
**Related work:** polar-high `write_mps` — **shipped in v2.1.1** (see `POLAR_HIGH_WRITE_MPS_HANDOFF.md`).
**Related work:** FlexTool Layer 2 autoscale rewrite — **shipped on `main`** (commit `d48f416f`, see `LAYER2_POLARS_NATIVE_PLAN.md`). That fix removed ~45 GB from the pre-`write_mps` phase; pyramid measurements below should be **re-validated** before this work starts, since the numerator has shifted.
**Status:** scoped, **re-profile required before implementation** (this-session fixes invalidated the prior measurements).
**Estimated effort:** investigation 1-2 days; fix 2-5 days depending on findings

---

## 1. Background — what we observed

The user ran DES (RETO-Africa) end-to-end on a 64 GB box. v3.8.0 (mathprog/glpsol → MPS → subprocess HiGHS) solves the same LP comfortably — peak ~30 GB system used across the cascade. v4 OOMs.

Two distinct memory cliffs in v4. Their fixes are independent:

| Cliff | Owner | Status |
|---|---|---|
| ~~`write_mps` triple-frame + parallel polars sort spikes to ~50 GB~~ Layer 2 autoscale's `bucket_coefficients` spikes to ~45 GB pre-`write_mps` | polar-high `write_mps` + FlexTool `_layer2.py` | **Both shipped.** polar-high v2.1.1; FlexTool commit `d48f416f`. |
| **FlexData build / `_emit_solve_time.run` pyramid to ~35 GB transient** | **FlexTool** | **This handoff** — re-measure first (see below). |

The "parallel polars sort" hypothesis that initially drove the write_mps work was wrong: polar-high pins `POLARS_MAX_THREADS=1` in its `__init__.py`, so polars is single-threaded throughout. The real ~50 GB cliff observed in the DES profile was **Layer 2's `bucket_coefficients`** materialising every constraint family's term LazyFrames concurrently into Python float lists, to compute a few dozen `log_mean` scalars — see §4-bis "Validated diagnostic lessons" below.

System-monitor traces consistently show a **~35 GB transient pyramid in system memory** during the cascade's input/preprocessing phase, well before any HiGHS or `write_mps` work. The pyramid rises from baseline (~5-7 GB), peaks somewhere mid-`_emit_solve_time.run`, and drops back ~5-7 GB after "FlexData built" — net retained ~9 GB, but the transient peak in between is much larger.

This pyramid is the second cliff that must be addressed before DES fits on a 64 GB box. With `write_mps` fixed to its ~2-3 GB target *but* a 35 GB transient FlexData peak still happening, the combined v4 footprint still spikes above what v3.8 needed.

## 2. The bookkeeping gap (read first — this hid the problem)

`flextool/engine_polars/_orchestration.py` has a phase recorder (`_MemoryRecorder`, lines ~548-893) that samples RSS at **named checkpoints only**. Default whitelist (line ~531):

```python
_MEM_WHITELIST_LABELS = frozenset({
    "Run start",
    "Inputs prepared",
    "FlexData built",
    "Matrix built by polar-high",
    "Solver",
    "Outputs written",
})
```

The "FlexData built" checkpoint reports a +9 GB Δ vs "Inputs prepared". **But that's the post-consolidation delta, not the transient peak between samples.** Polars/glibc release intermediate buffers between the start and end of the sub-emitter chain in `_emit_solve_time.run`, so the recorder sees only the residual ~9 GB. System monitor (sampling RAM continuously) sees the real ~35 GB transient.

**First diagnostic improvement: instrument finer-grained checkpoints inside `_emit_solve_time.run`.** That module (file `flextool/engine_polars/_emit_solve_time.py:35`, the `run()` function) is a ~70-batch sequential pipeline of `_emit_*` calls. Wrapping each batch (or each L1/L2/L3 block) with `memory_recorder.checkpoint(...)` will tell us *which sub-emitter* spikes. Until we have that, we're guessing.

The recorder is already threaded through `state._memory_recorder` (see `_orchestration.py` where it's published via `set_phase_recorder`). `_emit_solve_time.run` doesn't currently use it but could — add a non-whitelisted `checkpoint("batch_N_done")` per batch, and run with `FLEXTOOL_MEMORY_VERBOSE=1` to see them all in stdout.

## 3. Where the pyramid lives

The cascade does (per `_orchestration.py` and `_native_run_model.py`):

```
Inputs prepared          ← end of load_flextool / write_workdir_inputs
  load_flextool start    ← _emit_solve_time.run starts here (Phase E.3 work)
  load_flextool: time + node loaded
  load_flextool: process topology loaded
  load_flextool: varcost loaded
  input pass 1a direct_params_a
  input pass 2  projection_params
  input pass 3  derived_a     ← largest deltas in past traces
  input pass 1b direct_params_b
  input pass 4  derived_b
  input pass 5  derived_c
  input pass 6  derived_d
  input pass 7  derived_e
  input pass 8  derived_f
  input pass 9  derived_g
  input pass 10 existing_chain
FlexData built           ← here the recorder shows -4.7 GB drop
```

From historic verbose traces, the largest individual checkpoint deltas are at "input pass 3 derived_a" (~+782 MB) and "input pass 4 derived_b" (~+483 MB). But these are *post-step* deltas; the *transient peaks within a step* are what the system monitor captures and the recorder misses.

The CPU pattern during this phase (from system monitor) is **single-threaded** — no parallel polars work visible. So the pyramid is sequential allocate-then-free, not parallel-sort-with-worker-buffers. Different mechanism from the `write_mps` cliff.

## 4-bis. Validated diagnostic lessons (from the 2026-05-27 Layer 2 + write_mps work)

This handoff was written before the polar-high `write_mps` + FlexTool Layer 2 work. That work shipped four lessons that this phase should reuse:

1. **`MALLOC_ARENA_MAX=2` is the cheapest first try.** One env var, no code change. glibc's per-thread malloc arenas can fragment polars peak by 2-4×; capping them often gets a workload off the cliff before any instrumentation is added. **Try this before anything else.**
2. **The "polars parallel sort multiplies peak by core-count" hypothesis is refuted on this stack.** polar-high pins `POLARS_MAX_THREADS=1` at import time, and FlexTool inherits that. A synthetic-LP bench (`polar-high-opt/tests/_bench_write_mps_parallel.py`) confirmed sort is *not* a parallelism multiplier here. **Don't waste time on threadpool-toggle experiments**; if you see multi-core activity at OOM time it's glibc / kernel page-zeroing / OOM-killer scheduling, not polars.
3. **Gated env-var checkpoints are the right shape for per-phase RSS diagnosis.** `POLAR_HIGH_WRITE_MPS_PROFILE=1` is the polar-high precedent — when set, emits tab-separated `[write_mps profile] phase=X family=Y rss_gb=Z delta_gb=±W ...` lines to stderr; zero overhead when unset (no `psutil` import). Use the same shape for `_emit_solve_time.run`'s per-batch instrumentation — `FLEXTOOL_PYRAMID_PROFILE=1` would slot in alongside the existing `FLEXTOOL_MEMORY_VERBOSE=1`.
4. **The polars-streaming `group_by` + `agg(sum,count,min,max)` pattern works for "aggregate stats across a large frame" use cases.** See `flextool/engine_polars/autoscale/_layer2.py` (post commit `d48f416f`) for the canonical implementation: build a per-id classification frame once, join it into the lazy plan, `.group_by(eff_t).agg([...]).collect(engine="streaming")`. Per-term non-streaming fallback wrapped in a try/except handles polars nodes that don't yet support streaming. This is the **new reference pattern** for any FlexTool emit_* code that walks a large polars frame just to compute a few scalars per category — and likely covers a large fraction of the remaining pyramid.

## 4. Hypotheses

Ranked by likelihood given the single-threaded shape:

1. **Polars `.collect()` on a fused LazyFrame chain materializes a much larger intermediate than its final result.** The "input pass" emitters chain a lot of joins / projections; the final frame is small, but polars may not stream the whole chain — it eagerly materializes one of the wide intermediate joins. Same family of bug as `write_mps`'s materialize-then-sort, but in FlexTool's own code instead of polar-high's.
2. **A specific Param-product expression with 3+ Param factors creates a join-buffer explosion.** Polar-high's engine.py (line ~1814-1831) explicitly flags this pattern: "rhs.lazy may be a chain of Param * Param * ... whose eager materialisation (`rhs.frame`) explodes intermediate join buffers by orders of magnitude vs the constraint's row_count". The mitigation there is semi-join pruning + streaming engine. FlexTool may have similar Param-products inside the `_emit_*` modules without the same mitigation.
3. **Polars retains LazyFrame plan history.** Each `with_columns`/`join`/`group_by` builds the plan; the optimizer may hold all input frames alive until the final `.collect()`. If the chain accumulates without explicit `.collect()` boundaries, peak memory holds the full chain rather than the streaming subset.
4. **glibc heap fragmentation across the 70-batch sequence.** Each emit_* allocates polars frames, frees them, but glibc's malloc keeps the arena pages. The pyramid could be cumulative arena growth rather than any single batch peaking. `malloc_trim(0)` periodically would address this; the recorder already calls it after `write_workdir_inputs_end` (see `_orchestration.py:2365`), but maybe not in enough places.

We won't know which dominates until §2's instrumentation lands. **Don't guess the fix — measure first.**

## 5. What the consumer (FlexTool) already provides

Reusable machinery:

- **Phase recorder + verbose mode**: `_MemoryRecorder.checkpoint(label, logger, user_label=...)` in `_orchestration.py:548-893`. Calls already use it at file/CSV write points. Add per-batch calls inside `_emit_solve_time.run`.
- **`_try_malloc_trim()`** at `_orchestration.py:956-...`. Best-effort `malloc_trim(0)` to hand freed glibc arena pages back to OS. Currently called once between `write_workdir_inputs_end` and `load_flextool`; could be called more often if hypothesis 4 fits.
- **`FLEXTOOL_MEMORY_DIAGNOSTICS=1`**: enables `tracemalloc` + per-checkpoint CSV under `solve_data/memory_diagnostics.csv`. Combined with `--debug` it captures the full trace. Use this to identify which sub-step's Python-allocated state dominates (tracemalloc) vs which is C-side (RSS-only).
- **`FLEXTOOL_MEMORY_VERBOSE=1`**: bypasses the 6-label whitelist so every `checkpoint()` call emits to stdout.

These were added in commits `f1568912` (--debug → env var wiring) and `ac643dd1`-era memory work. Both are live in HEAD.

## 6. Investigation plan

### Phase 0 — Cheapest first try, no code change (5 minutes)

Before adding any instrumentation: re-run DES with **`MALLOC_ARENA_MAX=2`** prepended to the existing run command. If the pyramid drops under budget, you're done. (Per §4-bis lesson 1; this works often enough to be worth trying before any code change.)

### Phase A — Localize the peak (0.5 day)

1. Add `state._memory_recorder.checkpoint(f"emit_solve_time.{name}_done")` after each `_*.emit_*` call in `_emit_solve_time.run` (~70 call sites). Use non-whitelisted labels so they only emit under `FLEXTOOL_MEMORY_VERBOSE=1`. (Or follow the polar-high precedent and add a dedicated `FLEXTOOL_PYRAMID_PROFILE=1` env var per §4-bis lesson 3 for a tighter on/off without flooding all the existing verbose-mode lines.)
2. Reproduce: run DES with `MALLOC_ARENA_MAX=2`, `--save-memory off` (so no concurrent write_mps work masks the signal), `--debug`, and `FLEXTOOL_MEMORY_VERBOSE=1`. Watch the verbose trace and the system monitor side-by-side. **Use the post-Layer-2-fix baseline as the comparison point** — the prior ~35 GB pyramid number was measured before commit `d48f416f` and may have shrunk substantially.
3. Identify which 2-3 batches together account for the remaining transient.

### Phase B — Drill into the worst offenders (0.5-1 day)

For each identified batch, inspect the emitter function. Look for:
- `.collect()` calls — are they on small final outputs or on intermediate wide joins?
- Param multiplications chained more than 2 deep (hypothesis 2).
- Joins where the right-hand side is full-dim and the left-hand side is sparse (the polar-high lines 1814-1860 pattern).
- Missing `engine="streaming"` on `.collect()`.

### Phase C — Fix (1-3 days depending on findings)

Likely fixes by hypothesis (apply where supported by the measured data):

- **For "scan a large frame to compute a few category-keyed scalars"**: use the `group_by(eff_t).agg([sum, count, min, max]).collect(engine="streaming")` pattern. Canonical implementation: `flextool/engine_polars/autoscale/_layer2.py` (post `d48f416f`) — a per-id classification frame is built once, joined into each term's lazy plan, then a single streaming group-by produces the per-category aggregates without ever materialising the per-row coefficient frames. This is the pattern that dropped Layer 2's peak from ~45 GB to <500 MB; see §4-bis lesson 4. Add a per-term non-streaming fallback for polars nodes that don't yet stream.
- **For full-chain materializations**: insert `.collect(engine="streaming")` to bound intermediate buffers. Where streaming refuses (some node types don't stream), break the chain with an explicit `.collect()` at a narrow point — smaller frames flow downstream.
- **For Param-products**: replicate polar-high's pattern (semi-join with target row index, streaming collect). See `polar-high/src/polar_high/engine.py:1814-1860` for the canonical implementation.
- **For glibc fragmentation**: add `_try_malloc_trim()` after each L1/L2/L3 batch block. Cheap on Linux (single libc call), no-op elsewhere. Also document `MALLOC_ARENA_MAX=2` in user-facing docs as the cheapest first-try mitigation — it's a strict superset of the per-call `malloc_trim` strategy for callers willing to set an env var.

### Phase D — Verification (0.5 day)

1. Re-run DES with the same instrumentation. Target: pyramid peak <15 GB transient (down from ~35 GB).
2. Compare end-to-end run to glpsol/v3.8's ~30 GB total system peak. v4 with both cliffs fixed should be in the same range or better.
3. No regression on `tests/engine_polars/test_solver_integration.py` (all currently pass under `FLEXTOOL_SAVE_MEMORY=0,1`).
4. No regression in other scenarios — run the full `tests/test_scenarios.py` gate.

## 7. Open decisions

1. **How aggressive to be with `malloc_trim`?** Calling it after every batch is safe but adds tiny overhead (~ms each). After each L-block is a reasonable middle ground.
2. **Should the recorder be re-shaped to detect transients?** The current "sample at named checkpoint" model misses pyramids. An alternative: a separate sampler thread polling RSS every 1 s into `memory_diagnostics.csv`. Adds complexity but means future cliffs of this shape don't hide for as long. Defer until we have one more example.
3. ~~**Coordination with the polar-high `write_mps` fix.**~~ **Done.** polar-high `write_mps` shipped in v2.1.1. The FlexTool Layer 2 fix (`d48f416f`) is the first concrete precedent of the polars-streaming `group_by` pattern adopted into this codebase — emit_* code should follow the same shape (see §4-bis lesson 4 and §6 Phase C above). The original "polars parallel sort" hypothesis was refuted (see §4-bis lesson 2) — don't pursue it.
4. **Tests for the memory profile itself.** Currently no test asserts "this scenario stays under N GB". Should we add a memory-budget test for DES (or a synthetic equivalent) that runs under a tracked RSS cap? Would catch regressions of this shape early. Out of scope for the fix; worth scoping after.

## 8. Reference data points

- v3.8.0 (mathprog → glpsol → HiGHS subprocess), same LP, same machine: peak ~30 GB system used across full cascade.
- v4 without save-memory: ~58 GB peak during HiGHS run() (separate cliff).
- v4 with save-memory in-process: ~54 GB peak during HiGHS writeModel (separate cliff).
- v4 with save-memory subprocess (build_only/writeModel): ~60 GB peak during MPS write (separate cliff).
- v4 with save-memory subprocess (new write_mps): ~60 GB peak during polars sort (separate cliff, polar-high handoff).
- **v4 in any mode: ~35 GB transient pyramid during `_emit_solve_time.run`** (THIS handoff).

The LP for DES (solve_hydro sub-solve): 9.9M rows × 5M cols × 19.8M nonzeros pre-presolve. Captured from glpsol-emitted MPS dimensions in the v3.8 run.

## 9. Reference material — file:line citations

In `flextool-engine/flextool/engine_polars/`:

| File | Lines | What it is |
|---|---|---|
| `_emit_solve_time.py` | 35-(231) | The orchestrator function `run()` — the ~70-batch sequential pipeline that the pyramid lives inside. |
| `_orchestration.py` | 531-538 | The phase-recorder whitelist (only 6 labels emit to stdout in non-verbose mode). |
| `_orchestration.py` | 548-893 | `_MemoryRecorder` class — `checkpoint()` API and CSV/log emission. |
| `_orchestration.py` | 956-... | `_try_malloc_trim()` helper. |
| `_orchestration.py` | 2329-2342 | Where `FLEXTOOL_MEMORY_DIAGNOSTICS=1` env var is consulted at run setup. |
| `_orchestration.py` | 839 | Where `FLEXTOOL_MEMORY_VERBOSE=1` is consulted to bypass the label whitelist. |
| `_native_run_model.py` | 1143 | Where the "prep: _emit_solve_time.run done" user-facing checkpoint is emitted (the one that hides the transient). |
| `cli/cmd_run_flextool.py` | 348-349 | Where `--debug` sets both env vars. |

In polar-high (for the Param-product pattern reference):

| File | Lines | What it is |
|---|---|---|
| `polar-high-opt/src/polar_high/engine.py` | 1814-1867 | Canonical semi-join + streaming-engine pattern for bounding Param-product peak memory. **Copy this pattern into FlexTool emit_* code where hypothesis 2 applies.** |
| `polar-high-opt/src/polar_high/engine.py` (`Problem.write_mps`) | (search for `def write_mps`) | Reference for per-family streaming COO emission + streaming sort + chunked output. Pattern: bound peak by processing one family at a time and dropping intermediate frames before the next family's collect. |

In flextool-engine (new this session — the validated `group_by` pattern):

| File | Lines | What it is |
|---|---|---|
| `flextool/engine_polars/autoscale/_layer2.py` (`bucket_coefficients`, post `d48f416f`) | — | **Canonical polars-streaming `group_by` + `agg(sum, count, min, max)` pattern.** Build a per-id classification DataFrame once (here: `col_id → (column_type_id, has_multiplier_param)`), join into each term's lazy plan, run a single streaming group-by that escapes only the (small) aggregate frame. Per-term non-streaming fallback wrapped in try/except. **Copy this shape into emit_* code wherever the goal is "scan a large frame to compute a few category-keyed scalars".** |
| `tests/engine_polars/autoscale/test_layer2_polars_native.py` | — | Test pattern for verifying mathematical equivalence between a legacy and a polars-streaming rewrite of the same aggregation. Includes synthetic-LP memory smoke test with `tracemalloc`. |

## 10. Coordination

- Independent of `write_mps` work. Can ship in either order; combined effect of both fixes is the goal.
- No FlexTool API surface changes (this is internal — emit_* sub-routines + bookkeeping).
- No polar-high dependency bump needed (the patterns to copy are static reference code, not new APIs).

---

**One-line summary of intent:** Find which of the ~70 `_emit_*` batches inside `_emit_solve_time.run` is responsible for the ~35 GB transient memory pyramid, then apply polars streaming / semi-join / malloc_trim patches (analogous to polar-high engine.py:1814-1867) to bound it to <15 GB.

This handoff was produced 2026-05-27 from FlexTool-side investigation. Source conversation / system-monitor charts / OOM-killer dumps available on request.
