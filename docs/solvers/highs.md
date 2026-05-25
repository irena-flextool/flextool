# HiGHS

## What it is

HiGHS is the high-performance, open-source linear programming, mixed-integer
programming, and quadratic programming solver developed at the University of
Edinburgh. It is the **default solver in FlexTool** and the only one that
ships with the FlexTool installation — no separate setup, no licence, no
environment variables. Every FlexTool scenario that does not explicitly set
the `solver` parameter runs on HiGHS via the in-process Python API.

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

## The `solver_io_api` choices for HiGHS

| Value      | What it does                                                                  | When to use it                                                  |
|------------|-------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `direct`   | In-process via `highspy.Highs.passModel`. Default. The fastest path.          | Always, unless you have a specific reason to switch.            |
| `mps`      | Not supported for HiGHS. Raises `ValueError` at dispatch time.                | Never. The in-memory path is strictly better for HiGHS.         |
| `lp`       | Reserved by the dispatch enum but not implemented for any solver yet.         | Never (today).                                                  |

The `mps` mode is intentionally refused for HiGHS — there is no scenario
where writing an MPS file just to read it back into the same HiGHS process
is faster or more correct than the direct path. Use `direct` (the default).

## HiGHS-specific options worth knowing

HiGHS exposes a long list of options; FlexTool surfaces them through the
generic `solver_options` Map plus the three convenience knobs. The most
commonly useful HiGHS option names are:

| HiGHS option name | Type   | Notes                                                                        |
|-------------------|--------|------------------------------------------------------------------------------|
| `presolve`        | string | `"on"` (default), `"off"`, or `"choose"`.                                    |
| `parallel`        | string | `"on"`, `"off"`, or `"choose"` (default).                                    |
| `time_limit`      | float  | Wall-clock seconds. Equivalent to FlexTool's `solver_time_limit`.            |
| `mip_rel_gap`     | float  | Relative MIP optimality gap. Equivalent to FlexTool's `solver_mip_gap`.      |
| `threads`         | int    | Thread cap. Equivalent to FlexTool's `solver_threads`.                       |
| `solver`          | string | `"simplex"`, `"ipm"`, or `"choose"`. (Note: this is HiGHS's *own* `solver` option — not FlexTool's.) |

### Legacy `highs_*` parameters (still supported)

Three older FlexTool parameters predate the generic `solver_*` mechanism and
remain available for backward compatibility:

| Legacy FlexTool parameter | Maps to HiGHS option | Type   |
|---------------------------|----------------------|--------|
| `highs_presolve`          | `presolve`           | string |
| `highs_method`            | `solver`             | string |
| `highs_parallel`          | `parallel`           | string |

Existing scenarios that author `highs_presolve = "off"` continue to work
unchanged. The v52 generic way to express the same thing is:

```text
solver_options:
  presolve = "off"
```

Either form works on HiGHS. Use the generic `solver_options` for new
scenarios so the same authoring style transfers to other solvers; keep the
legacy `highs_*` names if you are editing existing scenarios and prefer not
to migrate them.

## Common errors

HiGHS is unusual in that license, install, and environment errors are
essentially absent. The two failure modes you can realistically see are:

- **Model infeasibility.** FlexTool reports `INFEASIBLE` status from HiGHS;
  see the FlexTool log for the path to the LP file (when `io_api != direct`)
  or use `keep_solver=True` inside the engine for in-process debugging.
- **Numerical issues.** The LP-scaling guard already in FlexTool catches the
  most common cases; if you see HiGHS warn about ill-conditioning, the
  scenario probably has cost or capacity coefficients spanning more than
  ~12 orders of magnitude.

If you suspect a HiGHS internal issue, set `solver_log_level = "verbose"`
to capture HiGHS's own diagnostics in the FlexTool log.

## How to set it in FlexTool

HiGHS is the default. No parameter is needed. To make the choice explicit
(or to set HiGHS-specific options), author on the relevant `solve` entity:

```text
solve_advanced.solver        = "highs"
solve_advanced.solver_io_api = "direct"
solve_advanced.solver_options:
  presolve     = "on"
  parallel     = "on"
solve_advanced.solver_time_limit = 600
```
