# Gurobi

## What it is

Gurobi is a commercial linear, mixed-integer, and quadratic programming
solver used widely on large energy-system optimisation models. FlexTool
writes each solve's model to an MPS file and runs Gurobi's command-line
solver `gurobi_cl` on it in a separate process (found on `PATH`, or in
`~/gurobi/bin`), passing every parameter as a `Name=value` argument. FlexTool
never inspects a licence: the licence-discovery path is Gurobi's own.

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

`gurobipy` lets FlexTool detect Gurobi at start ("Solvers:" line); the solve
itself runs `gurobi_cl`, which comes with the Gurobi installation.

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

WLS and Compute Server users set up the licence the way `gurobi_cl` reads it
(a `gurobi.lic` with the WLS credentials, or `GRB_LICENSE_FILE`). Server,
cloud and licence parameters (`ComputeServer`, `CS...`, `Cloud...`, `WLS...`,
`TokenServer`, ...) are taken from `solver_config/gurobi.opt` only: the
input database's `solver_arguments` may not set them (see
[What the database may set](index.md#what-the-database-may-set)).

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

FlexTool runs Gurobi as the `gurobi_cl` command-line program. When it fails,
FlexTool stops the run with a `FlexToolUserError` whose message includes
`gurobi_cl`'s own output (its `stdout` and `stderr`), so Gurobi's text is in
the FlexTool log.

- **No licence, or the licence check failed.** `gurobi_cl` exits with an
  error and prints, for example, *"No Gurobi license found"*. FlexTool shows:
  *"Solver 'gurobi' subprocess failed a license check. Details: 'gurobi'
  CLI for solve '<solve>' exited with returncode=... --- stdout --- ..."*
  (see [License options](#license-options)). Fix by placing the
  `gurobi.lic` file in your home folder (or `/opt/gurobi/`) or by setting
  `GRB_LICENSE_FILE` in the environment FlexTool runs in.
- **Licence expired.** The same message, with Gurobi's text *"License has
  expired"*. Renew the licence through the Gurobi web portal.
- **`gurobi_cl` not found.** *"Solver 'gurobi' CLI binary is not installed on
  this system."* Install Gurobi and put `gurobi_cl` on the `PATH`.
- **Unknown Gurobi parameter.** A misspelt name in `solver_arguments` (e.g.
  `MIPGapp`) stops the run before its first solve, with the closest name
  (`MIPGap`) in the message: the database may name only parameters in
  FlexTool's list ([What the database may set](index.md#what-the-database-may-set)).
  In `gurobi.opt` (or from `solver_arguments` where this computer allows
  unlisted names) FlexTool warns and passes it; Gurobi ignores it, prints
  `Unknown parameter: 'MIPGapp' (ignored)` and solves on, and FlexTool repeats
  that as a warning naming where the parameter was set. Fix the name. With console logging off
  (`LogToConsole 0`) Gurobi prints no such line, and there is no warning.
- **File parameters.** Parameters that name a file or folder (`LogFile`,
  `SolFiles`, `NodefileDir`, `InputFile`, the `.prm` / `.json` file lists of
  tuning, concurrent and multi-objective runs) take a full path in
  `gurobi.opt`; in the database's `solver_arguments` they take only a plain
  file name, written in the solve's work folder, and a path is refused. See
  [What the database may set](index.md#what-the-database-may-set).
  `ResultFile` is ignored everywhere (with a warning): FlexTool names its own
  result file and reads the solution from it.
- **Time limit reached.** A MIP stopped at its time limit with a solution is
  kept with a warning that gives the gap reached; an LP, or a MIP without a
  solution, stops the run. See [Time limit](index.md#time-limit).

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
flextool <input_db_url> --solver-time-limit 60 --solver-mip-gap 0.001
```

`--solver-time-limit 60` reaches Gurobi as `TimeLimit=60` and
`--solver-mip-gap 0.001` as `MIPGap=0.001`, over the solve's
`solver_mip_gap` (0.005) and any `MIPGap` in `solver_arguments` or
`gurobi.opt`. Without the flags, `solver_mip_gap` gives `MIPGap=0.005`, and a
`MIPGap` (or `mipgap`: names match without regard to case) in
`solver_arguments` wins over it. The thread count comes from `Threads` in
`gurobi.opt` (`--threads` and `--highs-threads` do not change it). The run
prints what Gurobi receives, for example:

```text
Solver args (gurobi): Threads=1, MIPGap=0.001, LogToConsole=1, Method=2, Presolve=2, MIPFocus=1, TimeLimit=60
```

---

*Verified install on: [empty]*
