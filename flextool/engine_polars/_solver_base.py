"""SolverRunner shell — base class for the native cascade's solver subclasses.

The cascade's ``_PolarHighCascadeSolver`` / ``_NoOpSolver`` in
:mod:`flextool.engine_polars._orchestration` extend :class:`SolverRunner`
and override ``run`` to drive the polar-high LP build/solve in-process.
Direct instantiation + ``run()`` is not supported.
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
        """Unimplemented — subclasses override this method.

        Direct invocation on the base class is not supported.
        """
        raise NotImplementedError(
            "SolverRunner.run is unimplemented on the base class.  "
            "The cascade subclasses this class and overrides run(); "
            "direct invocation is not supported."
        )


__all__ = ["SolverRunner"]
