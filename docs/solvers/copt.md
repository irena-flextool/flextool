# COPT

## What it is

COPT (Cardinal Optimizer) from Cardinal Operations is a newer commercial
linear, mixed-integer, second-order cone, and semidefinite programming
solver. Its Python API is intentionally Gurobi-shaped, so polar-high's COPT
adapter mirrors the Gurobi adapter closely. FlexTool dispatches through
polar-high; it never imports `coptpy` directly.

Official site: <https://www.copt.de/> (Cardinal Operations product portal)
[FIXME: confirm canonical URL — also <https://www.shanshu.ai/> hosts COPT
information.]

## License options

- **Commercial.** Cardinal Operations sells perpetual and floating-server
  licences directly and through resellers.
- **Academic.** Free for academic users. Request through the Cardinal
  Operations academic programme on the official site.
- **Cloud / WLS-style token licences.** Cardinal supports floating-licence
  servers and token discovery, similar to Gurobi's WLS model.
- **Trial / community.** A free trial is typically offered; specifics
  change with release. Check the vendor site.

## Installation

Install the Python wrapper:

```bash
pip install coptpy
```

polar-high's COPT adapter also requires `scipy` for the vectorised matrix
load (same dependency as the Gurobi adapter); install it if missing:

```bash
pip install scipy
```

For licensing:

1. Obtain `copt.lic` (node-locked) or floating-licence credentials from
   Cardinal Operations.
2. Place `copt.lic` at the path indicated by the vendor installer, or set
   `COPT_LICENSE_DIR` to the directory containing it. Refer to the
   vendor's installation guide for the precise discovery order [FIXME:
   confirm URL].
3. Floating-licence users configure the licence server endpoint per the
   COPT documentation; polar-high accepts a pre-built `coptpy.Envr` via
   the `env=` pass-through.

### HiGHS / COPT process-coexistence warning

COPT 8.x ships native code (`coptpy.coptcore`) that conflicts with
`highspy` when both are loaded into the same Python interpreter:
`highspy.Highs.run()` can segfault once `coptpy` has been imported.

polar-high auto-detects this situation and transparently routes COPT
solves through the MPS-file fallback (HiGHS writes an MPS, the
`copt_cmd` CLI is invoked as a subprocess) so the in-process load does
not happen. This requires the standalone `copt_cmd` binary on `PATH`. If
it is missing, FlexTool surfaces a clear "binary not found" error. The
binary ships with the COPT distribution.

In a FlexTool scenario that uses only COPT (no HiGHS solves anywhere in
the run), the in-process direct path is used.

## Verification

After installing `coptpy`, run:

```bash
python -c "from polar_high.solvers import available_solvers; print(available_solvers)"
```

Expected: `'copt'` appears in the list.

```text
['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

To verify the licence works:

```bash
python -c "import coptpy as cp; cp.Envr().createModel('test')"
```

A clean exit (no `CoptError`) means the licence is found.

## Common errors

The error messages below are what FlexTool surfaces as `FlexToolUserError`.
polar-high's COPT adapter classifies licence errors by both numeric errno
and message keywords (because the published errno table is not as cleanly
documented as Gurobi's).

- **No licence file found.** `CoptError` with errno in the provisional
  licence range (`2`, `7`, `8`, `9`) or a message containing "license",
  "licence", "token", or "expired". FlexTool shows: *"Solver 'copt' is
  installed but its license check failed. Details: ..."*. Fix by placing
  `copt.lic` or setting `COPT_LICENSE_DIR`.
- **Licence expired.** Same FlexTool message family with the vendor's
  expiry text. Renew through the Cardinal Operations licence portal.
- **`scipy` missing.** FlexTool surfaces: *"Solver 'copt' is not
  installed on this system. ..."* with the cause *"scipy is not installed
  (required by the COPT adapter for vectorized matrix load)"*. Install
  scipy with `pip install scipy`.
- **`copt_cmd` not on `PATH` (HiGHS-coexistence fallback).** FlexTool
  shows: *"Solver 'copt' returned an error: ..."* with the cause naming
  the missing binary. Add the COPT distribution's `bin` directory to
  `PATH`, or run COPT solves from a process that does not also use
  HiGHS.

## How to set it in FlexTool

On the `solve` entity:

```text
solve_advanced.solver            = "copt"
solve_advanced.solver_time_limit = 60
solve_advanced.solver_mip_gap    = 0.005
solve_advanced.solver_threads    = 8
solve_advanced.solver_options:
  Presolve   = 1
  LpMethod   = 2
```

The convenience knobs translate to COPT's `TimeLimit`, `RelGap`, and
`Threads`. Raw `solver_options` win on key collision.

---

*Verified install on: [empty]*
