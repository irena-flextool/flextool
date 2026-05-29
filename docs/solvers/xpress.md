# Xpress

## What it is

FICO Xpress is a commercial linear, mixed-integer, and nonlinear programming
solver. Its Python API uses a column-oriented `loadproblem` entry point,
which polar-high's adapter feeds directly with the CSC arrays already
produced by the engine. FlexTool dispatches through polar-high; it never
imports `xpress` directly.

Official site: <https://www.fico.com/en/products/fico-xpress-optimization>

## License options

- **Commercial.** FICO sells perpetual and term licences directly and
  through partners.
- **Academic.** Free academic licences for teaching and research; apply
  through the FICO Xpress academic partnership.
- **Community Edition.** Free, no-registration size-limited build with a
  hard cap (historically **5000 variables / 5000 rows** for LPs and MIPs;
  exact cap and small-print may change by release — confirm against the
  current FICO documentation). Useful for tutorials
  and small validation models, too small for typical FlexTool dispatch
  scenarios.

Licence and download portal: FICO's Xpress product page above.

## Installation

Install the Python wrapper:

```bash
pip install xpress
```

The PyPI `xpress` wheel from FICO embeds the solver runtime, so no
additional download is needed for the Community Edition.

For licensed users:

1. Obtain `xpauth.xpr` from your FICO licence portal.
2. Place it at one of the default locations searched by Xpress, or set the
   environment variable `XPAUTH_PATH` to point at it. Refer to the vendor's
   installation guide for the precise discovery order on your platform
   ([FIXME: confirm URL]).
3. Floating-licence users point `XPAUTH_PATH` at the licence server's
   `xpauth.xpr` shim.

Unlike Gurobi/COPT, `xpress.problem()` does not currently accept a
pre-constructed environment object, so the polar-high `env=` pass-through
is accepted but unused for Xpress. Licence discovery is entirely vendor-side.

## Verification

After installing the wrapper, run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Expected: `'xpress'` appears in the list.

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

To verify the licence works (this also prints the Xpress banner with the
licence type):

```bash
python -c "import xpress; xpress.problem()"
```

Community Edition users see a banner stating the size cap. If the call
raises an exception mentioning *license*, *oem*, or *no token*, see
"Common errors" below.

## Common errors

The error messages below are what FlexTool surfaces as `FlexToolUserError`.
Xpress does not expose a stable numeric error code on its Python
exceptions, so polar-high classifies licence errors by message text rather
than code; the keyword set is `license`, `licence`, `oem`, `expired`,
`no token`.

- **No `xpauth.xpr` found.** Xpress raises with a message containing
  "license" or "no token". FlexTool shows: *"Solver 'xpress' is installed
  but its license check failed. Details: ..."*. Fix by placing
  `xpauth.xpr` or setting `XPAUTH_PATH`.
- **Community-Edition size cap exceeded.** Xpress raises with a message
  containing "oem" or "exceeds the size of an OEM problem". FlexTool
  surfaces this as the licence-error family. The fix is either to shrink
  the model or to upgrade to a full licence — Community Edition is
  unsuitable for production FlexTool scenarios.
- **Unknown Xpress control.** Typing a wrong `solver_arguments` key raises a
  vendor exception from `p.setControl(name, value)`; FlexTool shows
  *"Solver 'xpress' returned an error: ..."*. The control names are in the
  Xpress reference manual ([FIXME: confirm URL]).

## How to set it in FlexTool

On the `solve` entity:

```text
solve_advanced.solver         = "xpress"
solve_advanced.solver_mip_gap = 0.005
solve_advanced.solver_arguments:
  defaultalg = 4
  presolve   = 1
```
```bash
flextool <input_db_url> --solver-time-limit 60 --highs-threads 8
```

The convenience knobs translate to Xpress's `maxtime`, `miprelstop`, and
`threads`. Raw `solver_arguments` are passed verbatim to `setControl` and
win on key collision.

---

*Verified install on: [empty]*
