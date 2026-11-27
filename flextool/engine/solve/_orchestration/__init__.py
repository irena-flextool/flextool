"""Native polar_high orchestrator — master loop conductor.

The orchestrator:

* Combines the foundation modules (``_solve_config`` + ``_timeline`` +
  ``_recursive_solve`` + ``_stochastic``).
* Drives the per-solve preprocessing through the L0-L9 batch +
  ``preprocessing_solve_time`` + ``solve_writers``.
* Runs the actual solve via ``polar_high.Problem.solve`` (HiGHS).
* Captures :class:`SolveHandoff` per solve via the native
  :func:`build_handoff_from_solution`, threads it forward as
  ``prior_handoff``, and routes it into the in-memory handoff slot of
  the runner so the consume side (``preprocessing_solve_time``,
  ``handoff_writers``) reads from it.

Design choices
--------------

* The orchestrator drives ``_run_model.native_run_model`` once
  per top-level invocation, with a **polar_high-as-inner-solver** wrapper
  that:
    - Reads the per-solve snapshot via ``load_flextool``.
    - Builds the LP via ``build_flextool``.
    - Solves via ``polar_high`` (HiGHS).
    - Captures handoff via ``build_handoff_from_solution``.
    - Deposits handoff into ``state.handoffs`` for the next iteration's
      preprocessing.

* **Storage-fixing handoff is in-memory by default** when
  ``state.handoffs`` is non-None.  The file-copy path
  (``shutil.copy`` of ``solve_data_<parent>/fix_storage_*.csv``) is
  consulted only when ``state.handoffs is None``.

* **Roll-counter reset semantics**: every top-level
  :func:`run_orchestration` call invokes
  ``state.solve.roll_counter = state.solve.make_roll_counter()`` first
  so test re-use of the same SolveConfig doesn't desync (R-O5).

* **``model_solve`` validation**: enforced loud-and-early.  Empty
  ``model_solve`` or more-than-one model raises
  :class:`FlexToolConfigError`.

* **``run_chain``** is a thin compat shim that always delegates here.

Modules
-------

* ``_highs_options`` — the HiGHS options of a sub-solve, the ``Solver args`` echo, the basis cache gate.
* ``_autoscale_hooks`` — the autoscale steps around a sub-solve (Layer 1 / 2 / 3, report, guard, unscale).
* ``_memory`` — memory / phase diagnostics, heap release, the cross-level retention audit.
* ``_steps`` — :class:`OrchestrationStep`, :class:`SnapshotSolution`, the objective-scale outputs.
* ``_guards`` — guards of the solve chain and the realization solves of stochastic solves.
* ``_cascade`` — the master loop, :func:`_drive_cascade`.
* ``_entry`` — :func:`run_orchestration` and :func:`run_chain_from_db`.
"""
from __future__ import annotations

from flextool.engine.solve._orchestration._entry import (
    AXIS_ENUMS_ENV as AXIS_ENUMS_ENV,
    AXIS_ENUMS_WARNING as AXIS_ENUMS_WARNING,
    run_chain_from_db,
    run_orchestration,
)
from flextool.engine.solve._orchestration._memory import (
    get_phase_recorder as get_phase_recorder,
    set_phase_recorder as set_phase_recorder,
)
from flextool.engine.solve._orchestration._steps import OrchestrationStep
from flextool.engine.solve._orchestration._steps import SnapshotSolution as SnapshotSolution

__all__ = [
    "OrchestrationStep",
    "run_orchestration",
    "run_chain_from_db",
]
