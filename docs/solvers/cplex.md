# CPLEX

## What it is

IBM ILOG CPLEX is a long-established commercial linear, mixed-integer, and
quadratic programming solver. FlexTool writes each solve's model to an MPS
file and runs the CPLEX interactive optimizer (`cplex`, found on `PATH` or in
`~/cplex/bin`) on it in a separate process, giving every parameter as a
`set <name> <value>` command. FlexTool itself never imports `cplex`
directly.

Official site: <https://www.ibm.com/products/ilog-cplex-optimization-studio>

## License options

- **Commercial.** Perpetual or subscription licences via IBM and IBM
  partners.
- **Academic.** Free for students, faculty, and staff at recognised
  institutions through the IBM Academic Initiative.
- **Subscription / cloud.** IBM offers a subscription model with bundled
  cloud entitlements; details and SKU naming change over time — refer to
  the IBM product page for the current options.
- **Community Edition.** A size-limited free version exists (limits change
  by release; check IBM's documentation). Not recommended for FlexTool
  models above toy size.

## Installation

Install the Python wrapper:

```bash
pip install cplex
```

The PyPI `cplex` wheel embeds the runtime needed for typical
single-machine use. If you have a system-wide CPLEX Studio installation
(commercial or academic), use the `setup.py` shipped in
`<CPLEX_STUDIO_DIR>/cplex/python/<py-version>/<platform>/` instead:

```bash
cd <CPLEX_STUDIO_DIR>/cplex/python/<py-version>/<platform>
python setup.py install
```

The system-wide install path is the right one when you need licence
features (FlexNet, ILM) that the wheel does not include.

Licence files are discovered automatically by the CPLEX runtime. For
FlexNet licences, set `ILOG_LICENSE_FILE` to the licence file or
`@server` syntax for a floating licence. Academic users typically install
via the IBM Academic Initiative installer, which writes a licence as part
of the install.

## Verification

After installing the wrapper, run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Expected: `'cplex'` appears in the list.

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

To verify the licence is found:

```bash
python -c "import cplex; cplex.Cplex().solve()"
```

A clean exit (no `CplexSolverError` / `CplexError`) means the licence is
discoverable.

## Authoring `solver_arguments` for CPLEX

Write the parameter names of the interactive optimizer's `set` command, in
full and with spaces (`mip tolerances mipgap`). A dotted name
(`mip.tolerances.mipgap`, the Python-API spelling) is read as the spaced
one. An abbreviation (`mip tol mipg`) is refused from `solver_arguments`,
with the full name in the message; in `cplex.opt` it gives a warning, is
passed as written and does not replace the full name from another layer. See
[What the database may set](index.md#what-the-database-may-set).

```text
solver_arguments:
  timelimit                = 60
  mip tolerances mipgap    = 0.01
  threads                  = 8
```

`--solver-time-limit` (CPLEX `timelimit`) and `--solver-mip-gap`
(`mip tolerances mipgap`) win over these; `solver_mip_gap` is below them.

## Common errors

The error messages below are what FlexTool surfaces as `FlexToolUserError`.

- **No licence found.** `CplexSolverError` with code `1016`
  (`CPXERR_NO_LICENSE`). FlexTool shows: *"Solver 'cplex' is installed
  but its license check failed. Details: ..."* with the vendor's message
  about a missing or unreadable licence. Fix by installing the licence
  file or setting `ILOG_LICENSE_FILE`.
- **ILM / subscription failure.** Code `32024` (and other 32000-range
  codes). Same FlexTool licence-error message family. Re-authenticate
  your IBM Cloud subscription or refresh the ILM token.
- **Unknown CPLEX parameter.** A `set` command CPLEX does not know is
  reported in the solver output; FlexTool does not check CPLEX's names. The
  names are in the IBM CPLEX parameters reference
  (<https://www.ibm.com/docs/en/icos/22.1.0?topic=cplex-list-parameters>).
- **Time limit reached.** A MIP stopped at its time limit with a solution is
  kept with a warning; an LP, or a MIP without a solution, stops the run. See
  [Time limit](index.md#time-limit).

## Licence holders and other programs in front of CPLEX

FlexTool 3 ran a solve's `solver_precommand` in front of CPLEX. FlexTool 4 never runs a command stored in a
database; set the program on your computer as the CPLEX solver wrapper instead (`FLEXTOOL_SOLVER_WRAPPER_CPLEX`, or
Solver options → On this computer in the FlexTool window). See
[FlexTool 3 precommand users](index.md#flextool-3-precommand-users).

## How to set it in FlexTool

On the `solve` entity:

```text
solve_advanced.solver         = "cplex"
solve_advanced.solver_mip_gap = 0.01
solve_advanced.solver_arguments:
  lpmethod                = 4
  preprocessing presolve  = 1
```
```bash
flextool <input_db_url> --solver-time-limit 60 --solver-mip-gap 0.001
```

`--solver-time-limit` and `--solver-mip-gap` reach CPLEX as `timelimit 60` and
`mip tolerances mipgap 0.001`, over the solve's `solver_mip_gap`, its `solver_arguments` and
`cplex.opt` (precedence in [Solver selection](index.md)). Parameter names
match without regard to case (and a dot is read as a space). The thread count comes from `threads` in
`cplex.opt`; `--threads` and `--highs-threads` do not change it. The run
prints what CPLEX receives in a `Solver args (cplex):` line before each
solve.

---

*Verified install on: [empty]*
