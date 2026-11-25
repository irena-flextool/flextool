# Solve rules (`flextool/engine/solve/`)

- **Acceptance** (`_acceptance.py::classify_acceptance`): kOptimal is accepted; kUnknown (crossover-off IPM) is
  accepted as near-optimal only when primal feasible and the primal-dual gap is small (`FLEXTOOL_ACCEPT_*` knobs,
  `docs/dev/env_vars.md`).  Without `solve_diagnostics()` (subprocess / commercial solvers) only kOptimal passes.
  `run_crossover` is the model author's choice; never hard-code it on.
- **Console markers around the solve** use `print(..., flush=True)`, not the logger: stdout is redirected there so
  HiGHS' banner owns fd 1, and the GUI execution window parses these printed lines.
- Only HiGHS for anything beyond toy size: the commercial solvers here are licensed for tiny test models only.
