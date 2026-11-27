# Solve rules (`flextool/engine/solve/`)

- **Acceptance** (`_acceptance.py::classify_acceptance`): kOptimal is accepted; kUnknown (crossover-off IPM) is
  accepted as near-optimal only when primal feasible and the primal-dual gap is small (`FLEXTOOL_ACCEPT_*` knobs,
  `docs/dev/env_vars.md`).  Without `solve_diagnostics()` (subprocess / commercial solvers) only kOptimal passes.
  `run_crossover` is the model author's choice; never hard-code it on.
- **Console markers around the solve** use `print(..., flush=True)`, not the logger: stdout is redirected there so
  HiGHS' banner owns fd 1, and the GUI execution window parses these printed lines.
- Only HiGHS for anything beyond toy size: the commercial solvers here are licensed for tiny test models only.
- **The per-solve loop** is `_orchestration/` (a package): `_cascade._drive_cascade` runs every sub-solve
  (`_run_one`), `_entry` holds `run_chain_from_db` / `run_orchestration`; the HiGHS options, autoscale steps,
  guards, step records and memory diagnostics each have their own module (map in `_orchestration/__init__.py`).
  Patch a name on the submodule whose code reads it, not on the package.
- **The cascade walk** is `_run_model/` (a package): `_native_run_model.native_run_model` expands the solve tree
  and sets up each sub-solve; the storage steps it calls live in `_rolling_storage`, `_nested_storage`,
  `_rp_methods` and `_storage_frames`, the run-scoped timelines in `_run_timelines` (map in
  `_run_model/__init__.py`).  Tests patch the seams (`_roll_tail_plans`, `make_step_jump`) on `_native_run_model`.
- **The Benders region split** is `_region_filter/` (a package): `_split.split` slices a whole-system `FlexData`
  into regions (`_region_data` builds one region and injects its half-flows), `_master_network.master_network_data`
  builds the master's reduced `FlexData`; arcs, master side, checks and region membership have their own modules
  (map in `_region_filter/__init__.py`).  `_benders` reads the public names through the package, so tests patch
  `master_network_data` on the package (or `_benders._region_filter`); private names live on their submodule.
