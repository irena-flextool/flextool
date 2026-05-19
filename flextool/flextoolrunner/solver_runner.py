"""SolverRunner shell + HiGHS solver-option resolvers.

Δ.22 — GMPL retired
===================

Δ.21 retired the ``--engine=gmpl`` CLI dispatch.  Δ.22 deleted the
GMPL artefacts (``flextool/flextool.mod``,
``flextool/flextool_base.dat``, ``bin/glpsol*``) and the legacy
Lagrangian coordinator that drove ``glpsol`` per region.  The
``SolverRunner.run()`` GMPL/HiGHS workflow that this module used to
own (``_run_glpsol_*``, ``_run_highs_or_cplex``, ``_run_phase_3``,
``_run_glpsol``, ``_run_cplex``, ``_cplex_to_glpsol``,
``_write_glpsol_solution``, ``_platform_binaries``, ``_run_highs``)
has been removed.

What's left
-----------

* :class:`SolverRunner` — minimal base class that the native cascade's
  ``_FlexpyCascadeSolver`` / ``_NoOpSolver`` subclasses (in
  :mod:`flextool.engine_polars._orchestration`) extend.  Its
  ``__init__`` stores ``state`` / ``logger`` for subclass use;
  ``run()`` is unimplemented by design — the cascade overrides it to
  drive the polar-high LP build/solve in-process and never reaches
  the legacy GMPL path.

* :func:`resolve_relax_feasibility` / :func:`resolve_ipm` — kept as
  pure helpers that map a CLI / env-var value to a HiGHS option.
  They are still exercised by ``tests/test_solver_options.py`` and
  may be re-wired into the native cascade's HiGHS configuration in
  a future dispatch (currently the cascade builds Problem instances
  via polar-high and configures HiGHS directly there).

The corresponding ``--ipm`` / ``--relax-feasibility`` CLI flags were
removed in Δ.22 phase C; the env-var fallbacks
(``FLEXTOOL_IPM`` / ``FLEXTOOL_RELAX_FEASIBILITY``) survive and are
still resolved by these helpers for callers that import them
directly.
"""
from __future__ import annotations

import os
from typing import Optional


# ---------------------------------------------------------------------------
# Solver-option resolvers (CLI / env-var → HiGHS option value)
# ---------------------------------------------------------------------------

RELAX_FEASIBILITY_ENV_VAR = "FLEXTOOL_RELAX_FEASIBILITY"
"""Environment-variable fallback for ``--relax-feasibility``.  Set to
an empty value or ``1`` / ``yes`` / ``on`` / ``true`` to request the
default relaxed tolerance (``1e-5``); set to a floating-point number to
request an explicit tolerance."""

IPM_ENV_VAR = "FLEXTOOL_IPM"
"""Environment-variable fallback for ``--ipm``.  Truthy values
(``1`` / ``yes`` / ``on`` / ``true``) switch HiGHS to the interior-point
solver; unset / falsy leaves HiGHS' default simplex method in place."""

DEFAULT_RELAX_FEASIBILITY = 1e-5
"""Default tolerance used when ``--relax-feasibility`` is passed
without a value.  Two orders of magnitude looser than HiGHS'
``1e-7`` default — loose enough to absorb typical sub-tolerance
residuals on wide-bound models without being irresponsibly loose."""


def resolve_relax_feasibility(cli_value) -> Optional[float]:
    """Resolve ``--relax-feasibility`` into an explicit tolerance.

    Accepts the CLI value (which may be ``None`` for absent, the string
    ``"default"`` for flag-with-no-value, or a float/numeric string for
    an explicit tolerance) and falls back to the
    :data:`RELAX_FEASIBILITY_ENV_VAR` env var when the CLI is silent.

    Returns ``None`` when the user did not request relaxation
    (HiGHS keeps its defaults), otherwise a positive float tolerance.
    Invalid values (non-numeric, <=0) return ``None``.
    """
    if cli_value is None:
        raw = os.environ.get(RELAX_FEASIBILITY_ENV_VAR, "").strip()
        if not raw:
            return None
        lowered = raw.lower()
        if lowered in ("1", "true", "yes", "on"):
            return DEFAULT_RELAX_FEASIBILITY
        try:
            tol = float(raw)
        except ValueError:
            return None
        return tol if tol > 0 else None
    # CLI path.  argparse used to set ``cli_value`` to the sentinel
    # string "default" when the flag was passed without ``=TOL``;
    # otherwise it was already a float.  The CLI flag itself was
    # removed in Δ.22, but the resolver still accepts the same shapes
    # so direct callers (and the unit tests) continue to work.
    if cli_value == "default":
        return DEFAULT_RELAX_FEASIBILITY
    try:
        tol = float(cli_value)
    except (TypeError, ValueError):
        return None
    return tol if tol > 0 else None


def resolve_ipm(cli_flag: bool) -> bool:
    """True iff ``cli_flag`` is set OR :data:`IPM_ENV_VAR` is truthy."""
    if cli_flag:
        return True
    raw = os.environ.get(IPM_ENV_VAR, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Minimal SolverRunner shell — base class for native-cascade subclasses
# ---------------------------------------------------------------------------


class SolverRunner:
    """Minimal shell that the native cascade's solver subclasses extend.

    The legacy GMPL/HiGHS workflow (``run`` driving glpsol then HiGHS
    on an MPS file) was removed in Δ.22.  ``run()`` here raises
    :class:`NotImplementedError`; subclasses (notably
    ``_FlexpyCascadeSolver`` / ``_NoOpSolver`` in
    :mod:`flextool.engine_polars._orchestration` and friends) override
    it to drive the polar-high LP in-process.

    ``__init__`` stores ``state`` and ``logger`` for the subclasses,
    matching the contract those subclasses already rely on via
    ``super().__init__(runner_state)``.
    """

    def __init__(self, state) -> None:
        self.state = state
        self.logger = state.logger

    def run(self, current_solve: str) -> int:  # noqa: ARG002
        """Unimplemented — the legacy GMPL/HiGHS workflow was removed in Δ.22.

        The native cascade subclasses override this method.  Direct
        instantiation + ``run()`` (the legacy ``FlexToolRunner.run_model``
        + ``SolverRunner(...)`` path) is no longer functional; see
        ``_archive/progress.md`` Δ.22 for context.
        """
        raise NotImplementedError(
            "SolverRunner.run was removed in Δ.22 along with the GMPL "
            "pipeline.  The native cascade subclasses this class and "
            "overrides run(); direct invocation is no longer supported. "
            "See specs/lagrangian_port_handoff.md for the deferred "
            "Lagrangian port that was the last consumer of this path."
        )


__all__ = [
    "DEFAULT_RELAX_FEASIBILITY",
    "IPM_ENV_VAR",
    "RELAX_FEASIBILITY_ENV_VAR",
    "SolverRunner",
    "resolve_ipm",
    "resolve_relax_feasibility",
]
