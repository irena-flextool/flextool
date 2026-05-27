# Solver selection in FlexTool

FlexTool can dispatch each `solve` in a scenario tree to a different LP / MIP
solver. The default — and what every existing scenario continues to use
without any extra configuration — is **HiGHS**, the open-source solver that
ships with FlexTool. Commercial solvers (Gurobi, CPLEX, Xpress, COPT) are
opt-in: install the vendor's Python wrapper, obtain a licence, then set
`solver = "<name>"` on the affected `solve` entity in the Spine database.
Solver choice is per-solve, not global, so a scenario can mix (for example) a
large dispatch LP on Gurobi with a small validation solve on HiGHS.

## Supported solvers

| Solver  | License                                       |
|---------|-----------------------------------------------|
| HiGHS   | Free, open source                             |
| Gurobi  | Commercial / academic free / WLS              |
| CPLEX   | Commercial / academic free / subscription     |
| Xpress  | Commercial / community (5000 var/row cap)     |
| COPT    | Commercial / academic free                    |

Per-solver installation, licensing, and verification:

- [HiGHS](highs.md) — bundled, no setup
- [Gurobi](gurobi.md)
- [CPLEX](cplex.md)
- [Xpress](xpress.md)
- [COPT](copt.md)

PLEASE NOTE: The instructions can be out-of-date.
Let us know if this is the case by opening an issue.

## How to switch solvers

Solver selection lives on the `solve` entity in the Spine database, in the
`solve_advanced` parameter group. Each parameter is optional and falls back
to a documented default if omitted, so existing scenarios behave exactly as
they did before this feature was added.

The seven parameters are, with their per-solver defaults:

| Parameter            | Default     | One-line description                                         |
|----------------------|-------------|--------------------------------------------------------------|
| `solver`             | `"highs"`   | Solver name from `[highs, gurobi, cplex, xpress, copt]`      |
| `solver_io_api`      | `"direct"`  | How the model reaches the solver: `direct` / `mps` / `lp`    |
| `solver_options`     | empty Map   | Raw key/value pairs forwarded verbatim to the solver         |
| `solver_time_limit`  | unset       | Wall-clock seconds; normalised across solvers                |
| `solver_mip_gap`     | unset       | Relative MIP gap; normalised across solvers                  |
| `solver_threads`     | unset       | Thread count; normalised across solvers                      |
| `solver_log_level`   | `"normal"`  | One of `silent` / `normal` / `verbose`                       |

Example — set Gurobi with a 60-second wall-clock cap on a single solve:

```text
solve entity:  yearly_dispatch
  solve_advanced.solver             = "gurobi"
  solve_advanced.solver_time_limit  = 60
```

Set in Spine Toolbox by selecting the `solve` entity, opening its parameter
editor, finding the `solve_advanced` group, and entering the values above.
Unspecified parameters take their defaults.

## The seven parameters in detail

| Parameter            | Type   | Default     | Description                                                                                              |
|----------------------|--------|-------------|----------------------------------------------------------------------------------------------------------|
| `solver`             | str    | `"highs"`   | Solver name. Must be one of `highs`, `gurobi`, `cplex`, `xpress`, `copt`.                                |
| `solver_io_api`      | str    | `"direct"`  | Historically `direct` (in-process Python API) vs `mps` (file-based subprocess); after the in-process cold-solve retirement this knob is effectively informational — non-HiGHS solvers are always dispatched through the subprocess MPS path regardless of value. HiGHS still picks `direct` when warm reuse is active and falls back to subprocess otherwise. |
| `solver_options`     | map    | empty       | Map of key → value pairs passed verbatim to the underlying solver. Use this for any solver-native option that is not one of the three convenience knobs below. Raw options always override the convenience knobs on key collision. |
| `solver_time_limit`  | float  | unset       | Wall-clock time limit in seconds. FlexTool translates this to the right native parameter name per solver (`TimeLimit` / `timelimit` / `maxtime` / `TimeLimit` / `time_limit`). |
| `solver_mip_gap`     | float  | unset       | Relative MIP optimality gap (e.g. `0.01` for 1 %). Translated to each solver's native parameter (`MIPGap` / `mip.tolerances.mipgap` / `miprelstop` / `RelGap` / `mip_rel_gap`). |
| `solver_threads`     | int    | unset       | Maximum thread count. Translated to each solver's native parameter (`Threads` / `threads` / `threads` / `Threads` / `threads`). |
| `solver_log_level`   | str    | `"normal"`  | `silent` suppresses solver output, `normal` is the solver default, `verbose` enables the solver's most chatty mode. |

### Authoring `solver_options` in Spine

`solver_options` is a Spine **Map** of string keys to string-or-float values.
Keys are the solver's own parameter names (e.g. `Presolve` for Gurobi or
`presolve` for HiGHS), values are whatever that solver expects.

Example (Gurobi, in Spine Toolbox's map editor):

```text
solver_options:
  Presolve   = 2
  Method     = 2
  Heuristics = 0.1
```

These are passed through untouched; misspelt keys raise a solver error which
FlexTool surfaces as a `FlexToolUserError` with the vendor's text.

### Per-solver baseline `.opt` files

Every solver gets a baseline parameter file shipped in the FlexTool repo
under `solver_config/`:

| Solver  | Baseline file                | Native format                                       |
|---------|------------------------------|-----------------------------------------------------|
| HiGHS   | `solver_config/highs.opt`    | `key=value` per line, `#` comments                  |
| Gurobi  | `solver_config/gurobi.opt`   | `ParamName value` per line, `#` comments            |
| CPLEX   | `solver_config/cplex.opt`    | `name value` per line; `name` uses interactive `set` syntax (e.g. `mip tolerances mipgap`) |
| Xpress  | `solver_config/xpress.opt`   | `CONTROL value` per line                            |
| COPT    | `solver_config/copt.opt`     | `ParamName value` per line                          |

These files are user-editable.  FlexTool reads each solver's baseline at
solve time, overlays the scenario's `solver_options` Map (plus the
convenience-knob translations from `solver_time_limit` / `solver_mip_gap`
/ `solver_threads`) **line-by-line**, writes the merged result to a temp
file alongside the MPS, and feeds it to the solver via the per-solver
mechanism:

| Solver  | How FlexTool feeds the merged file to the CLI                                                  |
|---------|------------------------------------------------------------------------------------------------|
| HiGHS   | `--options=<file>` argv flag on the `flextool.cli.cmd_solve_mps` subprocess                    |
| Gurobi  | `gurobi_cl ReadParams=<file> ResultFile=<sol> <mps>` (Gurobi's native parameter-file slot)     |
| CPLEX   | Each line emitted as `set <name> <value>` on the interactive optimizer's stdin before `read`   |
| Xpress  | Each line emitted as `setControl <NAME> <value>` on the optimizer console's stdin before `readprob` |
| COPT    | Each line emitted as `set <ParamName> <value>` on `copt_cmd`'s stdin before `read`             |

**Override precedence** (highest wins):

1. Raw `solver_options` entries on the scenario / solve.
2. Translated convenience knobs (`solver_time_limit` / `solver_mip_gap` /
   `solver_threads`).
3. The baseline file under `solver_config/<solver>.opt`.

Override `solver_config/` location with the `FLEXTOOL_SOLVER_CONFIG_DIR`
environment variable when running outside the standard project layout
(for example in CI).

## Warm-start caveat

The cross-solver dispatch always builds a fresh solver instance per call.
HiGHS retains its existing warm-start fast path because FlexTool uses HiGHS
directly (not via the cross-solver `solve()` entry point) when no `solver`
parameter is set. **Picking any non-HiGHS solver disables warm-start for that
solve.** In a rolling-horizon scenario this means each sub-solve is built
cold; expect a slower per-iteration build, but the result is fully correct.

## Lagrangian-decomposition caveat

`decomposition_method = lagrangian_region` is currently **HiGHS-only**.
Setting a non-HiGHS `solver` on a Lagrangian-decomposed scenario raises a
`FlexToolUserError` at startup. The error names both the offending solve and
the two remedies (either set `solver = highs` for that solve, or remove the
Lagrangian decomposition from the group).

This is an upstream constraint in polar-high: its `LagrangianProblem.solve`
does not yet accept a `solver_name`. Lifting the restriction is tracked
separately; in the meantime FlexTool fails loudly rather than silently
running on HiGHS.

## Verifying which solvers are installed

To see which solver Python wrappers polar-high has detected on this system,
run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Example output on a developer machine with all five wrappers installed:

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

A solver appears in `available_solvers` if its **Python wrapper** is
importable. That is not the same as "the solver is usable" — license checks
happen later, when FlexTool actually dispatches a solve. The most common
failure mode is `available_solvers` lists (say) `gurobi` but the first solve
raises a `FlexToolUserError` about a missing licence file; see the relevant
per-solver page for the licence setup.

## Common pitfalls

- **Setting `solver` without installing the wrapper.** Run the verification
  command above before authoring `solver = "gurobi"` (or similar) on a
  solve. If the solver name is missing from `available_solvers`, the
  per-solver page has the `pip install` line you need.
- **Setting `solver` without a valid licence.** The wrapper installs without
  one; the licence check only fires at solve time. FlexTool's error will
  point you at the licensing section of the solver page.
- **Mixing solvers without checking.** A scenario tree can route each solve
  to a different solver, but every solve needs its own `solver` parameter
  if you want anything other than HiGHS. Unset solves stay on HiGHS.
- **Using `io_api = "mps"` for very large models.** The MPS fallback writes
  the full model to disk and shells out to the solver's CLI. After the
  in-process cold-solve retirement this is the only path for non-HiGHS
  solvers; `direct` is silently treated as a synonym for `mps` at the
  orchestrator level. The trade-off is bounded peak RSS (the
  `Problem.write_mps` writer caps ~2-3 GB on the largest LPs) at the cost
  of file I/O plus a fresh solver process per sub-solve.
- **Hand-writing `solver_options` and the convenience knobs together.** Raw
  options win on key collisions. If you set both `solver_time_limit = 60`
  and `solver_options = {TimeLimit: 30}` on a Gurobi solve, Gurobi sees
  `TimeLimit = 30`.
