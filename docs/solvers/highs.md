# HiGHS

## What it is

HiGHS is the high-performance, open-source linear programming, mixed-integer
programming, and quadratic programming solver developed at the University of
Edinburgh. It is the **default solver in FlexTool** and the only one that
ships with the FlexTool installation — no separate setup, no licence, no
environment variables. Every FlexTool scenario that does not explicitly set
the `solver` parameter runs on HiGHS.

Internally FlexTool runs HiGHS via one of two paths:

- **Warm-active path** — the default for cascade runs. Keeps a single live
  `highspy.Highs` instance across structurally-compatible sub-solves so each
  iteration's right-hand-side updates flow into the same in-memory model.
- **Subprocess path** — for cold solves, `--save-memory`, and single-solve
  invocations. FlexTool writes the LP to a temporary MPS via the polar-high
  polars→MPS writer, then runs `python -m flextool.cli.cmd_solve_mps` in a
  clean child process so HiGHS' simplex/IPM working set lives outside the
  FlexTool address space. The retired in-process cold-HiGHS path is no
  longer reachable from the orchestrator — set `FLEXTOOL_SAVE_MEMORY=1`
  explicitly to silence the soft-promote warning on the cold path.

Official site: <https://highs.dev/>

## License options

- **Free and open source** under the MIT licence. No commercial vs academic
  split, no node lock, no annual renewal — install and use without paperwork.

## Installation

Already installed. HiGHS is pulled in automatically when you install
FlexTool because `highspy` is a hard dependency. If for any reason it is
missing from your environment, install it directly:

```bash
pip install highspy
```

There are no licence files to place, no environment variables to set, and no
binaries to fetch separately from the vendor.

## Verification

Run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Expected output (the list ordering may vary depending on which other
wrappers are installed, but `'highs'` must appear):

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

If `'highs'` is missing, your FlexTool install is broken — reinstall
FlexTool or run `pip install highspy` to repair.

## The `--matrix-file-format` choices for HiGHS

The on-disk format used when a solver is dispatched via a matrix file is
selected by the `--matrix-file-format` CLI flag (it replaced the old
`solver_io_api` DB parameter):

| Value      | What it does                                                                  | When to use it                                                  |
|------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `mps`      | Default. On-disk MPS when a matrix file is written.                           | The default; required for commercial-solver dispatch.           |
| `lp`       | On-disk LP format instead of MPS.                                             | When you prefer human-readable LP files.                        |

For HiGHS without `--save-memory`, FlexTool dispatches in-process (the
fastest path) and the flag has no effect. With `--save-memory`, HiGHS
round-trips through MPS internally; the flag again has no effect on HiGHS.
The flag matters for commercial solvers, which are always dispatched via a
matrix file.

## HiGHS-specific options worth knowing

HiGHS exposes a long list of options; FlexTool surfaces them through the
generic `solver_arguments` Map plus the convenience knobs. The most
commonly useful HiGHS option names are:

| HiGHS option name | Type   | Notes                                                                        |
|-------------------|--------|------------------------------------------------------------------------------|
| `presolve`        | string | `"on"` (default), `"off"`, or `"choose"`.                                    |
| `parallel`        | string | `"on"`, `"off"`, or `"choose"` (default).                                    |
| `time_limit`      | float  | Wall-clock seconds. Equivalent to the `--solver-time-limit` CLI flag.        |
| `mip_rel_gap`     | float  | Relative MIP optimality gap. Equivalent to FlexTool's `solver_mip_gap`.      |
| `threads`         | int    | Thread cap. Equivalent to the `--highs-threads` CLI flag.                    |
| `solver`          | string | `"simplex"`, `"ipm"`, or `"choose"`. (Note: this is HiGHS's *own* `solver` option — not FlexTool's.) |

### The former `highs_*` parameters

Three older FlexTool parameters (`highs_presolve`, `highs_method`,
`highs_parallel`) predated the generic mechanism and have been **removed**.
Author the equivalent HiGHS option directly inside the `solver_arguments`
Map instead:

| Removed FlexTool parameter | HiGHS option key in `solver_arguments` | Type   |
|----------------------------|----------------------------------------|--------|
| `highs_presolve`           | `presolve`                             | string |
| `highs_method`             | `solver`                               | string |
| `highs_parallel`           | `parallel`                             | string |

For example, to turn presolve off:

```text
solver_arguments:
  presolve = "off"
```

## Common errors

HiGHS is unusual in that license, install, and environment errors are
essentially absent. The two failure modes you can realistically see are:

- **Model infeasibility.** FlexTool reports `INFEASIBLE` status from HiGHS;
  see the FlexTool log for the path to the matrix file (when a file-based
  dispatch was used) or use `keep_solver=True` inside the engine for
  in-process debugging.
- **Numerical issues.** The LP-scaling guard already in FlexTool catches the
  most common cases; if you see HiGHS warn about ill-conditioning, the
  scenario probably has cost or capacity coefficients spanning more than
  ~12 orders of magnitude.

If you suspect a HiGHS internal issue, pass `--solver-log-level verbose`
to capture HiGHS's own diagnostics in the FlexTool log.

## How to set it in FlexTool

HiGHS is the default. No parameter is needed. To make the choice explicit
(or to set HiGHS-specific options), author on the relevant `solve` entity:

```text
solve_advanced.solver = "highs"
solve_advanced.solver_arguments:
  presolve = "on"
  parallel = "on"
```

Run-wide options such as the wall-clock cap are passed as CLI flags:

```bash
flextool <input_db_url> --solver-time-limit 600
```
