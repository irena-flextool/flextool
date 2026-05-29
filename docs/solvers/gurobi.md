# Gurobi

## What it is

Gurobi is a commercial linear, mixed-integer, and quadratic programming
solver used widely on large energy-system optimisation models. FlexTool
dispatches to Gurobi through polar-high's adapter; FlexTool itself never
imports `gurobipy` directly and never inspects a licence. Pre-configured
`gurobipy.Env` objects (for example with WLS credentials) can be passed
through, but the licence-discovery path is Gurobi's own.

Official site: <https://www.gurobi.com/>

## License options

- **Commercial.** Named-user, floating, and machine licences via the Gurobi
  sales channel.
- **Academic.** Free for students, faculty, and staff at recognised
  degree-granting institutions. Renewable annually from an academic IP.
- **Free trial.** Time-limited evaluation licence available from the Gurobi
  website with vendor sign-off.
- **WLS (Web Licence Service).** Cloud-friendly token-based licensing for
  containers, CI, and shared infrastructure.

Licence acquisition page: <https://www.gurobi.com/solutions/licensing/>

## Installation

Install the Python wrapper:

```bash
pip install gurobipy
```

polar-high also requires `scipy` for the vectorised matrix load on the
Gurobi path; it is normally already present in a FlexTool environment, but
if `pip install gurobipy` complains about a missing dependency, add it:

```bash
pip install scipy
```

After installing the wrapper, set up a licence file. The simplest
single-machine path:

1. Sign in at <https://www.gurobi.com/> and request a licence appropriate
   for your situation (commercial, academic, or trial).
2. Use Gurobi's `grbgetkey` tool, or the equivalent web flow, to download a
   `gurobi.lic` file.
3. Place `gurobi.lic` at either of:
   - `$HOME/gurobi.lic` (user-scoped, the simplest), or
   - `/opt/gurobi/gurobi.lic` (system-wide), or
   - any path of your choice, then set the `GRB_LICENSE_FILE` environment
     variable to that path.

WLS and Compute Server users typically build a `gurobipy.Env` programmatically
with credentials from the Gurobi web portal; FlexTool's dispatch
accepts a pre-built env via the polar-high `env=` kwarg, but routing one
through from a FlexTool scenario is currently a developer-level operation
(no Spine parameter for the env object yet).

## Verification

After installing `gurobipy`, run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Expected: `'gurobi'` appears in the printed list:

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

`gurobi` in the list confirms only that **the wrapper is importable** —
licence discovery still happens at solve time. To check that the licence is
also working, point FlexTool at any small Gurobi-targeted scenario and run
it; or, outside FlexTool:

```bash
python -c "import gurobipy as gp; gp.Model().optimize()"
```

A clean exit (no `GurobiError`) means the licence is found.

## Common errors

The error messages below are what FlexTool surfaces as
`FlexToolUserError` — they wrap polar-high's exception, which in turn wraps
Gurobi's raw `GurobiError`. The raw vendor text appears in the FlexTool log
as the cause of the wrapped exception.

- **No licence file found.** Gurobi `GurobiError` with `errno` in
  `10009..10015` (most commonly `10009` "No Gurobi license found"). FlexTool
  shows: *"Solver 'gurobi' is installed but its license check failed.
  Details: Gurobi license check failed (code 10009): No Gurobi license
  found. Place gurobi.lic at $HOME or /opt/gurobi/, set GRB_LICENSE_FILE,
  or pass a configured gurobipy.Env via env=..."*. Fix by placing the
  `gurobi.lic` file or setting `GRB_LICENSE_FILE`.
- **Licence expired.** Same FlexTool message family with the vendor text
  *"License has expired"*. Renew through the Gurobi web portal.
- **Wrapper installed but `scipy` missing.** FlexTool shows: *"Solver
  'gurobi' is not installed on this system. ..."* with the underlying
  cause "*scipy is not installed (required by the Gurobi adapter for
  vectorized matrix load)*". Run `pip install scipy`.
- **Unknown Gurobi option.** Setting a typo'd key in `solver_arguments`
  (e.g. `TimeLImit` instead of `TimeLimit`) raises `GurobiError` from
  `Model.setParam`; FlexTool surfaces it as *"Solver 'gurobi' returned an
  error: Gurobi error (code ...): unknown parameter ..."*. Fix the typo.

## How to set it in FlexTool

On the `solve` entity for the solve you want Gurobi to handle:

```text
solve_advanced.solver         = "gurobi"
solve_advanced.solver_mip_gap = 0.005
solve_advanced.solver_arguments:
  Method     = 2
  Presolve   = 2
  MIPFocus   = 1
```
```bash
flextool <input_db_url> --solver-time-limit 60 --highs-threads 8
```

The convenience knobs translate to `TimeLimit`, `MIPGap`, and `Threads`
respectively. Raw `solver_arguments` entries pass through untouched and win
on any key collision with the convenience knobs.

---

*Verified install on: [empty]*
