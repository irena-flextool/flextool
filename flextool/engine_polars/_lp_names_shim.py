"""``highspy.Highs`` shim for the commercial-solver output path.

Phase 3 of the FlexTool multi-solver port.  Wraps a
(:class:`~polar_high.solvers._base.SolverResult`,
:class:`~polar_high.solvers._lp_view.LpView`) pair and exposes only the
subset of the ``highspy.Highs`` API that FlexTool's output writers
consume:

* :func:`flextool.process_outputs.read_highs_solution.extract_variable`
  uses ``h.allVariableNames()``, ``h.getLp().row_names_``,
  ``h.getSolution().col_value``, ``h.getSolution().col_dual``, and
  ``h.getSolution().row_dual``.
* :func:`flextool.process_outputs.read_highs_solution.write_v_obj` uses
  ``h.getObjectiveValue()``.
* :func:`flextool.engine_polars._output_writer._rename_invest_columns`
  uses ``h.passColName(cid, name)`` — but the rename is applied inside
  :meth:`flextool.engine_polars._solver_result_to_solution.LiteSolution.from_solver_result`
  on the commercial path, so the shim's ``passColName`` is a no-op
  (idempotent — the name already matches).

The shim is constructed by
:meth:`flextool.engine_polars._solver_result_to_solution.LiteSolution.make_shim`,
which the output writer adapter calls when ``sol.highs is None`` (the
commercial-solver path).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from flextool.engine_polars._solver_result_to_solution import LiteSolution


@dataclass
class _LpHandle:
    """Stand-in for ``highspy.Highs.getLp()`` return object.

    Only the two name attributes are exposed; the writers never call
    other LP-handle methods on the highspy side either.
    """

    row_names_: list[str]
    col_names_: list[str]


@dataclass
class _SolutionHandle:
    """Stand-in for ``highspy.Highs.getSolution()`` return object.

    ``col_dual`` is filled with NaN.  None of the writers exposed by
    Phase 3 read column duals on the commercial path — the variables
    that consume ``col_dual`` (``v_invest.dual`` / ``v_divest.dual``)
    are LP-only artefacts of HiGHS' barrier crossover and are not
    available from any commercial adapter today.  We return NaN so any
    consumer that does happen to touch the array gets an obvious
    sentinel rather than silently-zero values.
    """

    col_value: np.ndarray
    col_dual: np.ndarray
    row_dual: np.ndarray


@dataclass
class LpNamesShim:
    """Minimal ``highspy.Highs`` API shim built from a LiteSolution.

    Construct via :meth:`from_lite_solution` rather than calling the
    constructor directly — the factory builds the numpy arrays the
    output writers expect.
    """

    _lp_handle: _LpHandle
    _solution_handle: _SolutionHandle
    _col_names: list[str]
    _objective: float

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_lite_solution(cls, sol: "LiteSolution") -> "LpNamesShim":
        """Build a shim from a :class:`LiteSolution`.

        * ``allVariableNames()`` returns the LiteSolution's
          (already-renamed) ``col_names``.
        * ``getLp().row_names_`` / ``col_names_`` come from the
          underlying LpView, with ``col_names`` post-rename.
        * ``getSolution().col_value`` is the LiteSolution's
          ``col_value`` array.
        * ``getSolution().col_dual`` is NaN-filled (commercial adapters
          do not expose reduced costs by default; only structurally
          interesting consumer is ``v_invest.dual``, which is a
          HiGHS-only artefact).
        * ``getSolution().row_dual`` is built from ``SolverResult.dual``
          keyed by the original LpView row names (unchanged by the
          rename — only column names are renamed).
        * ``getObjectiveValue()`` returns ``result.objective`` (raises
          on access for non-optimal solves via the LiteSolution.obj
          property contract).
        """
        result = sol.result
        lp_view = sol._lp_view
        assert sol.col_value is not None
        n_cols = len(sol.col_names)
        n_rows = len(lp_view.row_names)

        # Column duals: NaN — no commercial adapter populates these.
        col_dual = np.full(n_cols, np.nan, dtype=np.float64)

        # Row duals: from result.dual keyed by row name; NaN for any
        # constraint not in the dict.  ``SolverResult.dual`` is None
        # for MIP solves (per polar-high _base.SolverResult docstring).
        dual = result.dual or {}
        row_dual = np.empty(n_rows, dtype=np.float64)
        for i, rn in enumerate(lp_view.row_names):
            v = dual.get(rn)
            row_dual[i] = float(v) if v is not None else np.nan

        lp_handle = _LpHandle(
            row_names_=list(lp_view.row_names),
            col_names_=list(sol.col_names),
        )
        sol_handle = _SolutionHandle(
            col_value=sol.col_value,
            col_dual=col_dual,
            row_dual=row_dual,
        )
        objective = (
            float(result.objective) if result.objective is not None else float("nan")
        )
        return cls(
            _lp_handle=lp_handle,
            _solution_handle=sol_handle,
            _col_names=list(sol.col_names),
            _objective=objective,
        )

    # ------------------------------------------------------------------
    # highspy.Highs surface
    # ------------------------------------------------------------------

    def allVariableNames(self) -> list[str]:  # noqa: N802 — highspy casing
        """Return the ordered list of LP column names (post-rename)."""
        return self._col_names

    def getLp(self) -> _LpHandle:  # noqa: N802 — highspy casing
        """Return a stand-in object exposing ``row_names_`` / ``col_names_``."""
        return self._lp_handle

    def getSolution(self) -> _SolutionHandle:  # noqa: N802 — highspy casing
        """Return a stand-in object exposing
        ``col_value`` / ``col_dual`` / ``row_dual``."""
        return self._solution_handle

    def getObjectiveValue(self) -> float:  # noqa: N802 — highspy casing
        """Return the solver objective (NaN if non-optimal)."""
        return self._objective

    def passColName(self, cid: int, name: str) -> None:  # noqa: N802 — highspy casing
        """No-op.

        The v_invest_p / v_invest_n → v_invest rename happens inside the
        :class:`LiteSolution` constructor on the commercial path; by the
        time any caller tries to ``passColName`` here the names already
        match.  Kept as a method so the
        :func:`flextool.engine_polars._output_writer._rename_invest_columns`
        adapter doesn't need a branch for the shim path.
        """
        return None


__all__ = ["LpNamesShim"]
