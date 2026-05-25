# CPLEX

## What it is

IBM ILOG CPLEX is a long-established commercial linear, mixed-integer, and
quadratic programming solver. Its Python API is more verbose than Gurobi's,
and FlexTool's polar-high adapter accommodates that by feeding row-major
sparse data and using CPLEX's native ranged-row support. FlexTool itself
never imports `cplex` directly.

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

## Authoring `solver_options` for CPLEX

CPLEX parameters live under a nested attribute namespace
(`c.parameters.<group>.<sub>...<name>`). polar-high's CPLEX adapter
expects option keys as **dotted parameter paths** relative to
`c.parameters`. In FlexTool's `solver_options` Map, write the dotted name
directly:

```text
solver_options:
  timelimit                = 60
  mip.tolerances.mipgap    = 0.01
  threads                  = 8
```

These three correspond to FlexTool's `solver_time_limit`,
`solver_mip_gap`, and `solver_threads` convenience knobs; setting either
form works.

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
- **Unknown CPLEX parameter.** Typing `timeLimit` instead of `timelimit`
  in `solver_options` raises `CplexError` from the parameter setter;
  FlexTool surfaces it as *"Solver 'cplex' returned an error: ..."*. The
  parameter names are documented at the IBM CPLEX parameters reference
  ([FIXME: confirm URL] — typically under the IBM CPLEX online
  documentation for the installed version).

## How to set it in FlexTool

On the `solve` entity:

```text
solve_advanced.solver            = "cplex"
solve_advanced.solver_time_limit = 60
solve_advanced.solver_mip_gap    = 0.01
solve_advanced.solver_threads    = 8
solve_advanced.solver_options:
  lpmethod   = 4
  preind     = 1
```

The convenience knobs translate to CPLEX's `timelimit`,
`mip.tolerances.mipgap`, and `threads`. Raw `solver_options` win on key
collision.

---

*Verified install on: [empty]*
