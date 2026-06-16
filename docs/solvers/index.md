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

The DB-level solver parameters are, with their per-solver defaults:

| Parameter            | Default     | One-line description                                         |
|----------------------|-------------|--------------------------------------------------------------|
| `solver`             | `"highs"`   | Solver name from `[highs, gurobi, cplex, xpress, copt]`      |
| `solver_arguments`   | empty Map   | Raw option name → value pairs forwarded verbatim to the solver |
| `solver_mip_gap`     | unset       | Relative MIP gap; normalised across solvers                  |
| `solver_precommand`  | unset       | Shell command run before the solver subprocess (advanced)    |

A handful of options that used to be DB parameters are now controlled by
**CLI flags** on `flextool` (they apply to the whole run rather than per
solve):

| CLI flag               | Replaces former DB param | Values / notes                                  |
|------------------------|--------------------------|-------------------------------------------------|
| `--matrix-file-format` | `solver_io_api`          | `mps` (default) / `lp`; on-disk format used when the solver is dispatched via a matrix file |
| `--solver-time-limit`  | `solver_time_limit`      | Wall-clock seconds (HiGHS `time_limit`); unset means no limit |
| `--highs-threads`      | `solver_threads`         | Integer thread count (default 1); `>1` enables HiGHS parallel mode |
| `--solver-log-level`   | `solver_log_level`       | `silent` / `normal` / `verbose`                 |

Example — set Gurobi on a single solve, with a 60-second wall-clock cap for
the whole run:

```text
solve entity:  yearly_dispatch
  solve_advanced.solver = "gurobi"
```
```bash
flextool <input_db_url> --solver-time-limit 60
```

Set the DB parameter in Spine Toolbox by selecting the `solve` entity,
opening its parameter editor, finding the `solve_advanced` group, and
entering the values above. Unspecified parameters take their defaults.

## The parameters in detail

| Parameter            | Type   | Default     | Description                                                                                              |
|----------------------|--------|-------------|----------------------------------------------------------------------------------------------------------|
| `solver`             | str    | `"highs"`   | Solver name. Must be one of `highs`, `gurobi`, `cplex`, `xpress`, `copt`.                                |
| `solver_arguments`   | map    | empty       | Map of option name → value passed verbatim to the underlying solver. Use this for any solver-native option that is not one of the convenience knobs below. Raw options always override the convenience knobs on key collision. |
| `solver_mip_gap`     | float  | unset       | Relative MIP optimality gap (e.g. `0.01` for 1 %). Translated to each solver's native parameter (`MIPGap` / `mip.tolerances.mipgap` / `miprelstop` / `RelGap` / `mip_rel_gap`). |
| `solver_precommand`  | str    | unset       | Shell command run before the solver subprocess (advanced; e.g. to source a licence environment). |

The corresponding CLI flags (`--matrix-file-format`, `--solver-time-limit`,
`--highs-threads`, `--solver-log-level`) are documented in the table above.

### Authoring `solver_arguments` in Spine

`solver_arguments` is a Spine **Map** of string keys to string-or-float values.
Keys are the solver's own parameter names (e.g. `Presolve` for Gurobi or
`presolve` for HiGHS), values are whatever that solver expects.

Example (Gurobi, in Spine Toolbox's map editor):

```text
solver_arguments:
  Presolve   = 2
  Method     = 2
  Heuristics = 0.1
```

These are passed through untouched; misspelt keys raise a solver error which
FlexTool surfaces as a `FlexToolUserError` with the vendor's text.

### Per-solver baseline `.opt` files

Every solver gets a baseline parameter file under `solver_config/`.  These
are **user-local files, not committed to the repo**: each ships in the wheel
as a `solver_config/<solver>.opt.template`, and FlexTool seeds the runtime
`solver_config/<solver>.opt` from its template on first run (and re-seeds any
that are missing).  The runtime files are gitignored, so you can edit them —
or delete one to reset it to the bundled default — with zero git-committed
change.

| Solver  | Baseline file                | Native format                                       |
|---------|------------------------------|-----------------------------------------------------|
| HiGHS   | `solver_config/highs.opt`    | `key=value` per line, `#` comments                  |
| Gurobi  | `solver_config/gurobi.opt`   | `ParamName value` per line, `#` comments            |
| CPLEX   | `solver_config/cplex.opt`    | `name value` per line; `name` uses interactive `set` syntax (e.g. `mip tolerances mipgap`) |
| Xpress  | `solver_config/xpress.opt`   | `CONTROL value` per line                            |
| COPT    | `solver_config/copt.opt`     | `ParamName value` per line                          |

FlexTool reads each solver's baseline at
solve time, overlays the scenario's `solver_arguments` Map (plus the
convenience-knob translations from the `--solver-time-limit` /
`solver_mip_gap` / `--highs-threads` knobs) **line-by-line**, writes the
merged result to a temp file alongside the MPS, and feeds it to the solver
via the per-solver mechanism:

| Solver  | How FlexTool feeds the merged file to the CLI                                                  |
|---------|------------------------------------------------------------------------------------------------|
| HiGHS   | `--options=<file>` argv flag on the `flextool.cli.cmd_solve_mps` subprocess                    |
| Gurobi  | `gurobi_cl ReadParams=<file> ResultFile=<sol> <mps>` (Gurobi's native parameter-file slot)     |
| CPLEX   | Each line emitted as `set <name> <value>` on the interactive optimizer's stdin before `read`   |
| Xpress  | Each line emitted as `setControl <NAME> <value>` on the optimizer console's stdin before `readprob` |
| COPT    | Each line emitted as `set <ParamName> <value>` on `copt_cmd`'s stdin before `read`             |

**Override precedence** (highest wins):

1. CLI flags (`--solver-time-limit`, `--highs-threads`, …).
2. Raw `solver_arguments` entries on the scenario / solve.
3. Translated convenience knobs (`solver_mip_gap`).
4. The baseline file under `solver_config/<solver>.opt`.

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
- **Using `--matrix-file-format mps` for very large models.** The MPS
  fallback writes the full model to disk and shells out to the solver's CLI.
  After the in-process cold-solve retirement this is the only path for
  non-HiGHS solvers. The trade-off is bounded peak RSS (the
  `Problem.write_mps` writer caps ~2-3 GB on the largest LPs) at the cost
  of file I/O plus a fresh solver process per sub-solve.
- **Hand-writing `solver_arguments` and the convenience knobs together.** Raw
  options win on key collisions. If you pass `--solver-time-limit 60` and
  also set `solver_arguments = {TimeLimit: 30}` on a Gurobi solve, Gurobi
  sees `TimeLimit = 30`.
