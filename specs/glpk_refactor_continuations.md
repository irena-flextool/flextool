# GLPK-style refactor — continuations handoff

**Written:** 2026-05-28
**Audience:** future me / future agent picking this up in a week or two.
**Scope:** what the GLPK-style refactor landed, what it deliberately did NOT
land, and what it exposed but did not investigate.

---

## 1. What landed

A two-repo refactor unified polar-high's LP build pipeline around a single
canonical per-problem matrix and folded flextool's Layer 2 autoscaler into
that matrix as side vectors that bake at canonicalisation time.  The Stage A
"multiply at emit" pattern was retired from every primary consumer
(`Problem._build_lp_arrays`, `Problem.write_mps`, `WarmProblem._initial_build`,
`LpView.from_problem`).  The only remaining multiply-at-emit site is
`Problem._solve_streaming`, intentionally kept for its per-family CSR
memory-bound design (see deferral 2.2 below).

**polar-high — v2.1.3 → v2.2.0 (Stage A + Stage B)**

| Commit | Role |
|---|---|
| `ca24b79` | Stage A: add Layer 2 side-vector storage on Problem + multiply at emit (groundwork) |
| `dbf6f0e` | extend `_layer2_locked` guard to `Problem.add_var` |
| `d4171fb` | autoscale `_ranges_via_streaming` honours side vectors |
| `27b2e19` | regression tests for Layer 2 side-vector emit |
| `a3dd35f` | Stage B1: canonical CSC matrix + `write_mps` migration |
| `0711749` | Stage B2: `_build_lp_arrays` consumes canonical matrix (delegates to `_build_canonical_matrix`) |
| `3634f94` | Stage B3: `WarmProblem._initial_build` consumes canonical matrix |
| `682ddd9` | `LpView.from_problem` consumes canonical `col_obj` |
| `a3f68ba` | drop `_build_lp_arrays` back-compat shim params |
| `9d410b1` | release v2.2.0 |
| `bbe8f69` | docs: env vars reference page |

**flextool — v4.0.0a2 → v4.0.0a3 + 6 follow-on commits**

| Commit | Role |
|---|---|
| `9d7aa64a` | keep `_scale_rhs` lazy + size-skip large-family row scaling (pre-rewrite) |
| `ae093e27` | rewrite `apply_layer2` to write side vectors (no plan rewrites; deletes `_rewrite_term_lazy`, `_rewrite_obj_term_lazy`, `_scale_rhs`, `_rhs_has_vars`) |
| `64f7fe1c` | `apply_layer2` invalidates `Problem._matrix` cache |
| `9e1b6d65` | `_orchestration` comment refresh for Layer 2 warm path |
| `44653ef0` | release v4.0.0a3 |
| `db94d92a` | log: wrap at 100 chars + redesigned memory/timer table |
| `375922af` | diagnostics: in-process memory sampler (`FLEXTOOL_MEM_SAMPLER=1`) |
| `7cb985c4` | diagnostics: between-solves memory snapshot at level boundaries |
| `52b67e69` | orchestration: release subprocess solve state immediately on cold path |
| `ce4dbbbe` | orchestration: keep one warm-restart `Solution.highs` + provider per level |
| `8193516b` | orchestration: `FLEXTOOL_COLD_KEEP_PROVIDER` env knob |
| `753c8c14` | docs: env vars reference |
| `a062d715` | docs: between-solves parameter injection guide |
| `8be7a8f7` | docs: fix stale `FLEXTOOL_AUTO_SCALE` references |

Net effect on DES-scale workloads: Layer 2 apply no longer rewrites the
families' `LazyFrame` plans (it writes side vectors instead), so the
peak-memory cliff that triggered this whole arc is gone.  Memory across a
roll of solves is bounded because the cold path drops solver state at solve
end and the warm path keeps at most one `Solution.highs` per level.

---

## 2. What did NOT land (deliberate deferrals)

### 2.1 Per-family canonicalisation

The "Mostly ✓" row on the GLPK-likeness scorecard for build/scale.  Today
`_build_canonical_matrix` walks every constraint family into one global
hash-table dedup pass, then a global sort into CSC.  Per-family
canonicalisation would process one family at a time and emit its CSC slice
directly into the assembled matrix, exploiting the structural fact that
different families always get disjoint row ranges — cross-family dedup is
unnecessary by construction, not by convention.

Estimated gain: the global hash-table peak drops from `~1.5 × nnz` to
`~1.5 × max_family_nnz`, and the global sort scratch is eliminated entirely
(per-family CSC emission is a one-pass radix on local row index).

Effort: ~3–5 person-days.  Starting point is
`src/polar_high/engine.py::_build_canonical_matrix`.

### 2.2 `_solve_streaming` migration to canonical matrix

Deferred per orchestrator decision D8: canonicalising would defeat
`_solve_streaming`'s per-family CSR memory-bound design (which loads
families into HiGHS one at a time).  Reconsider only after 2.1 ships —
then `_solve_streaming` could slice per-family from `Problem._matrix`
cheaply.  Until then, this is the one consumer that still multiplies at
emit (intentional; don't touch its comments).

### 2.3 Memory-pressure yielding for retained HiGHS instances

The warm path (per `ce4dbbbe`) keeps one `Solution.highs` plus one
`flex_data_provider` per level.  Under memory pressure these should be
droppable.  There's already a mechanism — `FLEXTOOL_RSS_BUDGET_MB` drives
`FlexDataProvider` eviction — and it could be extended to evict the kept
HiGHS instances too.  User noted this is "not first priority since there
is save-memory" (`FLEXTOOL_SAVE_MEMORY=1` already gives an escape hatch).

### 2.4 Bounds-of-product autoscale

Stage D from the original plan: compute coefficient magnitude ranges by
walking parameter-source trees multiplicatively without ever evaluating
the full chain.  Would let us remove `POLAR_HIGH_RANGES_MAX_FAMILY_ROWS`
(the family-size skip in flextool's `bucket_coefficients` and in
polar-high's `_ranges_via_streaming`).  Orthogonal to the matrix
architecture — can ship independently.

### 2.5 CLI flag for memory sampler

Discussed but not added.  Today the sampler is activated only via env var
(`FLEXTOOL_MEM_SAMPLER=1`); a `--mem-sampler` CLI flag would improve
discoverability via `--help`.  Implementation note: a CLI surface must
set the env var before `start_mem_sampler()` is invoked (the sampler
reads the env var, not a configuration object).

---

## 3. What got exposed but not investigated

### 3.1 DES solve 9/9 OOM

The first DES run that got far enough to reach solve 9 hit ~55 GB total
RSS at solve-9 start.  The Phase 1 cold-path slim (`52b67e69`) and
Phase 2 warm-per-level retention (`ce4dbbbe`) both target the cross-solve
accumulation that drove this; needs re-validation on a fresh DES run
end-to-end.

### 3.2 Pre-existing `v_obj` zero bug — now fixed

The cold-path `_v_obj__{solve}.parquet` writer was emitting zero
objectives before commit `52b67e69`.  Any analysis built on pre-fix
`v_obj` parquets is wrong.  Warm-path runs were unaffected.  Flagging
because external scripts may still be consuming the bad parquets without
realising they need to re-run.

### 3.3 Stale doc references — partial sweep done

Phase A1 of the 2026-05-28 cleanup fixed every `FLEXTOOL_AUTO_SCALE`
reference in dev docs (scaling.md, engine_polars.md, architecture.md,
env_vars.md) and rephrased / retargeted a few `_build_lp_arrays` line
references in polar-high source comments.  No other staleness was found
in this sweep, but the user-facing docs at `docs/user/` were not
exhaustively grepped — worth a quick pass if someone hits a docs report
of a non-existent env var.

---

## 4. Known pre-existing issues (NOT caused by this refactor)

* **`tests/engine_polars/test_warm_chain_runner.py::test_warm_native_rolling_speedup`** —
  hangs on baseline `c2b1f48` (polar-high pre-refactor) too, confirmed
  via stash test.  Needs separate investigation; deselect it in CI runs
  via `--deselect tests/test_warm_chain_runner.py::test_warm_native_rolling_speedup`.
* **`tests/engine_polars/autoscale/test_registry_coverage.py::test_parameter_registry_covers_template_db`** —
  fails because six `solver_*` quantity-type entries are missing from
  `flextool/engine_polars/autoscale/_layer2_types.py`.  Trivial fix; the
  registry simply needs the six entries appended.  Nobody has done it.
* **`_ranges_via_passmodel` (polar-high)** — the only production callers
  for the legacy ranges path are gone; remaining call sites live in
  `tests/autoscale/test_ranges.py` (testing the legacy path itself).
  Candidate for removal in a future cleanup pass; do NOT remove without
  first confirming no out-of-tree consumer (e.g. flextool branches) calls
  it directly.

---

## 5. Documents to read first

When picking this up, read in this order:

1. This file (`specs/glpk_refactor_continuations.md`).
2. `docs/dev/env_vars.md` — env-var reference (the source of truth for
   what knobs exist and what they mean).
3. `docs/dev/inject_between_solves.md` — the `override_provider` API
   landed alongside the refactor.
4. polar-high `CHANGELOG.md` entry for `[2.2.0]`.
5. flextool `RELEASE.md` entry for `4.0.0a3`.
6. `specs/architecture_provider_and_future_duckdb.md` — context on the
   provider layer (referenced by deferrals 2.3 and 2.5).

---

## 6. Worktrees and branches

Both repos kept the `glpk-refactor` worktree + branch as backup:

* `/home/jkiviluo/sources/polar-high-opt/.claude/worktrees/glpk-refactor`
* `/home/jkiviluo/sources/flextool-engine/.claude/worktrees/glpk-refactor`

Both branches are fully merged into `main`; the worktrees can be removed
at convenience with:

```bash
git -C /home/jkiviluo/sources/polar-high-opt worktree remove .claude/worktrees/glpk-refactor
git -C /home/jkiviluo/sources/polar-high-opt branch -d glpk-refactor

git -C /home/jkiviluo/sources/flextool-engine worktree remove .claude/worktrees/glpk-refactor
git -C /home/jkiviluo/sources/flextool-engine branch -d glpk-refactor
```
