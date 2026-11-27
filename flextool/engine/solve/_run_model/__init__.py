"""Native cascade driver.

This module owns the per-solve cascade loop.  :func:`native_run_model`
is invoked by :func:`._orchestration._cascade._drive_cascade`.

Design decisions
----------------

* **Cascade loop is native; inner preprocessing is not (yet).**
  Scope: own the cascade walk and inline the solve-tree expansion /
  stochastic branching / per-solve setup; not re-port
  ``preprocessing.solve_time`` or the ``solve_writers`` module.  Those
  remain authoritative and are still called as functions — the
  ``_native_leaf_set_override()`` context (wired by
  :func:`._orchestration._cascade._drive_cascade`) intercepts the already-native
  preprocessing helpers.
* **Native solve-tree expansion** via
  :class:`flextool.engine.solve._recursive_solve.RecursiveSolveBuilder`
  + :class:`._stochastic.StochasticSolver`.
* **Direct handoff construction.**  The post-solve ``SolveHandoff`` is
  built by ``build_handoff_from_solution`` directly from the polar_high
  ``Solution`` object — no CSV round-trip.
* **Optional state fields tolerated.**  Native :class:`RunnerState`
  may lack ``timing_recorder`` / ``auto_scale``.  Callers that consume
  those guard with ``getattr(state, name, default)``.

Modules
-------

* ``_native_run_model`` — the per-solve cascade loop, :func:`native_run_model`.
* ``_storage_frames`` — the per-solve storage-method readers, :class:`._storage_frames._RollingStorage`, the coded
  storage warnings, the module logger.
* ``_rp_methods`` — storage-binding methods on representative periods: checks, the across-solve chain, the
  downgrade on solves without representative periods, the forward chain of fix_start_end nodes.
* ``_rolling_storage`` — the storage of rolling_window solves: carry, anchor and close of a roll, the per-period
  scopes, the rolling checks.
* ``_nested_storage`` — the storage of nested solves: the upper solve's planned levels and the rows that restate
  them in a lower level.
* ``_run_timelines`` — run-scoped timelines: the realization-roll plans, the realized-branch projection, the run's
  time steps and existing cohorts.
"""
from __future__ import annotations

from flextool.engine.solve._run_model._native_run_model import native_run_model
from flextool.engine.solve._run_model._run_timelines import (
    project_realized_branch_time_branch_rows as project_realized_branch_time_branch_rows,
    run_time_steps as run_time_steps,
)

__all__ = ["native_run_model"]
