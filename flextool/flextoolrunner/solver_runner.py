"""SolverRunner shell — base class for the native cascade's solver subclasses.

The legacy GMPL/HiGHS workflow that this module used to own was
removed in Δ.22 (glpsol + .mod + flextool_base.dat) and the legacy
Lagrangian coordinator that drove glpsol per region.  The native
cascade's ``_FlexpyCascadeSolver`` / ``_NoOpSolver`` in
:mod:`flextool.engine_polars._orchestration` extend :class:`SolverRunner`
and override ``run`` to drive the polar-high LP build/solve
in-process.  Direct instantiation + ``run()`` is no longer functional.
"""
from __future__ import annotations


class SolverRunner:
    """Minimal shell that the native cascade's solver subclasses extend.

    ``__init__`` stores ``state`` and ``logger`` for the subclasses,
    matching the contract those subclasses already rely on via
    ``super().__init__(runner_state)``.  ``run`` is unimplemented by
    design — the cascade subclasses override it.
    """

    def __init__(self, state) -> None:
        self.state = state
        self.logger = state.logger

    def run(self, current_solve: str) -> int:  # noqa: ARG002
        """Unimplemented — the legacy GMPL/HiGHS workflow was removed in Δ.22.

        The native cascade subclasses override this method.  Direct
        invocation is not supported.
        """
        raise NotImplementedError(
            "SolverRunner.run was removed in Δ.22 along with the GMPL "
            "pipeline.  The native cascade subclasses this class and "
            "overrides run(); direct invocation is no longer supported."
        )


__all__ = ["SolverRunner"]
