"""HiGHS determinism pin + advanced simplex-scaling strategy constant.

Extracted from the legacy :mod:`flextool.engine_polars.scaling` module so the
orchestrator can keep importing these knobs after ``scaling.py`` is retired
in favour of the :mod:`flextool.engine_polars.autoscale` package.

HiGHS' default-of-``choose`` for ``parallel`` / ``presolve`` / ``solver`` plus
a time-varying ``random_seed`` lets HiGHS pick different code paths between
runs.  For LPs with multiple optimal vertices that flips which optimum is
returned, which in turn flips golden-file comparisons (e.g. the
``coal_wind_ev`` / ``network_coal_wind_reserve_co2_capacity_margin``
scenarios in ``test_scenarios.py``).

Pinned:

* ``random_seed=0``  — fully deterministic seed.
* ``parallel="off"`` — no multi-threading parallelism inside HiGHS.
  This alone forces the serial dual simplex (HiGHS' default solver), so we
  deliberately do NOT also set ``threads=1`` here.  The user can override via
  ``--highs-threads N`` (N > 1), which flips ``parallel`` to ``on`` and sets
  ``threads=N`` in ``_finalise_highs_options`` — N == 1 leaves this pin
  intact and keeps default behaviour byte-identical.  HiGHS still initialises
  a single, process-global thread scheduler on the FIRST ``Highs::run()`` of
  the process; subsequent ``Highs`` instances that try to set ``threads`` to
  a different value are rejected with
  ``"global scheduler has already been initialized"``.  We avoid that error
  path because ``_finalise_highs_options`` resolves ``FLEXTOOL_HIGHS_THREADS``
  from a process-level env var set by the CLI before any solve runs — every
  sub-solve in the cascade sees the same value, so the scheduler is
  initialised once with the final thread count and never reconfigured.
  ``parallel=off`` (the default) is sufficient for determinism (no
  concurrent simplex trajectories); users who opt in to N > 1 explicitly
  accept the determinism / wall-clock trade-off.
* ``solver="simplex"`` — pick simplex unconditionally (avoids HiGHS' internal
  ``choose`` heuristic flipping between simplex / IPM).
* ``presolve="on"``  — force presolve on (vs the non-deterministic
  ``choose``).  We do NOT disable presolve — turning it off makes the gate
  ~3x slower and changes a great many LP solutions; ``on`` is deterministic
  and matches HiGHS' usual recommendation.

These propagate through the orchestrator's HiGHS-options builders to the
``set_solver_options(highs_options)`` call sites in
:mod:`flextool.engine_polars._orchestration` (both the warm path and the
cold path).  This is the actual control point — ``tests/highs.opt`` is
copied into the bin-dir fixture for CLI-style runs but is NOT read by
``polar_high.Problem`` (which is what ``test_scenarios.py`` drives via
``run_chain_from_db``).
"""
from __future__ import annotations


DETERMINISM_OPTIONS: dict[str, object] = {
    "random_seed": 0,
    "parallel": "off",
    "solver": "simplex",
    "presolve": "on",
}
"""HiGHS solver-option keys that pin byte-deterministic LP solutions."""


SIMPLEX_SCALE_STRATEGY_ADVANCED: int = 2
"""HiGHS ``simplex_scale_strategy`` value for Curtis-Reid row/col scaling.

HiGHS' default (1) is basic equilibration; (2) adds Curtis-Reid which costs
negligibly more but handles wide coefficient spreads much better.  Centralised
here so every call site uses the same value.
"""


__all__ = ["DETERMINISM_OPTIONS", "SIMPLEX_SCALE_STRATEGY_ADVANCED"]
