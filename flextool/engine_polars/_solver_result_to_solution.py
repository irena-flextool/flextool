"""``SolverResult`` → ``Solution`` normaliser (Phase 3 of FlexTool multi-solver port).

The polar-high ``Problem.solve`` path returns a :class:`polar_high.Solution`
that downstream FlexTool consumers (``engine_polars/input.py``,
``_emit_co2_accumulators.py``, ``process_outputs/read_parameters.py``,
``_output_writer.py``) rely on for:

* ``sol._vars`` — mapping ``var_name -> Var`` carrying ``.frame["col_id"]``.
* ``sol.value(name)`` — returns long-form ``(*dims, value)`` DataFrame.
* ``sol.obj`` — scalar objective (raises if non-optimal — same as Solution.obj).
* ``sol.optimal`` — bool.
* ``sol.col_names`` — ordered list of rendered LP column names.
* ``sol.highs`` — live ``highspy.Highs`` instance OR ``None`` for the
  commercial path (the output writer adapter routes via a shim instead).
* ``sol.status`` — termination status enum.

The commercial path (gurobi/cplex/xpress/copt + io_api=mps) flows through
:func:`polar_high.solvers.solve`, which returns the solver-agnostic
:class:`polar_high.solvers.SolverResult`.  This module rebuilds the
Solution-shaped surface from ``(SolverResult, Problem)`` so the same
downstream code paths work without per-call branching.

Variable-name reconciliation
---------------------------
The polars LP splits each invest/divest decision into ``v_invest_p`` /
``v_invest_n`` / ``v_divest_p`` / ``v_divest_n`` columns (rendered into
LP column names like ``v_invest_p[ent,period]``).  FlexTool writers
expect a unified ``v_invest[ent,period]`` / ``v_divest[ent,period]``
family.  In the HiGHS path the rename happens after the solve via
``passColName`` on the live solver (see
``_output_writer._rename_invest_columns``).  Commercial-path solvers
don't keep a live instance, so the rename happens here at construction
time — the LiteSolution's ``col_names``, ``_vars`` and ``primal`` dicts
all expose the unified names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from polar_high import Problem
    from polar_high.solvers._base import SolverResult
    from polar_high.solvers._lp_view import LpView


# ---------------------------------------------------------------------------
# Variable-name rename (matches ``_output_writer._VAR_RENAME``)
# ---------------------------------------------------------------------------

# Mapping mirrors ``flextool.engine_polars._output_writer._VAR_RENAME``.
# Kept duplicated rather than imported to avoid the late-import cycle
# (``_output_writer`` imports from this module via the LpNamesShim
# pathway).
_VAR_RENAME: dict[str, str] = {
    "v_invest_p": "v_invest",
    "v_invest_n": "v_invest",
    "v_divest_p": "v_divest",
    "v_divest_n": "v_divest",
}


def _rename_prefix(name: str) -> str:
    """Replace a leading ``v_invest_p[`` / ``v_invest_n[`` / ``v_divest_*[``
    prefix with the unified ``v_invest[`` / ``v_divest[`` form.

    Returns *name* unchanged if no rename applies.
    """
    for src, dst in _VAR_RENAME.items():
        if name.startswith(src + "["):
            return dst + name[len(src):]
        if name == src:
            return dst
    return name


# ---------------------------------------------------------------------------
# LiteSolution
# ---------------------------------------------------------------------------


@dataclass
class _LiteVar:
    """Minimal stand-in for :class:`polar_high.engine.Var`.

    Only carries the fields downstream FlexTool consumers actually
    touch: ``.frame`` (with ``col_id`` column) and ``.dims``.
    """

    name: str
    dims: tuple[str, ...]
    frame: pl.DataFrame


@dataclass
class LiteSolution:
    """Solution-shaped view over a :class:`SolverResult`.

    Built by :meth:`from_solver_result` after a commercial-solver call.
    Exposes the subset of the :class:`polar_high.Solution` API that
    FlexTool consumers rely on:

    * ``_vars`` — dict[str, _LiteVar] (rebuilt from the source Problem,
      with the v_invest_p/n → v_invest rename applied).
    * ``value(name)`` — long-form ``(*dims, value)`` DataFrame.
    * ``col_names`` — list of rendered LP column names (with the rename
      applied), aligned with ``LpView.col_names``.
    * ``obj`` — float; raises ``ValueError`` if the underlying
      ``SolverResult.objective`` is ``None`` (matches Solution.obj).
    * ``optimal`` — bool.
    * ``status`` — :class:`SolverStatus` from the SolverResult.
    * ``highs`` — always ``None`` (commercial path doesn't keep a
      live HiGHS instance).  The output writer adapter constructs a
      :class:`flextool.engine_polars._lp_names_shim.LpNamesShim` via
      :meth:`make_shim` when it needs the highspy-Highs surface.
    """

    result: "SolverResult"
    _lp_view: "LpView"
    _vars: dict[str, _LiteVar] = field(default_factory=dict)
    # ordered name vector (LpView.col_names with rename applied).
    col_names: list[str] = field(default_factory=list)
    row_names: list[str] = field(default_factory=list)
    # cached numpy view of primal values, aligned with col_names.
    col_value: np.ndarray | None = None
    # ``highs`` is always ``None`` for the LiteSolution surface; callers
    # that need the highspy.Highs API surface call :meth:`make_shim`.
    highs: Any = None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_solver_result(
        cls, result: "SolverResult", problem: "Problem",
    ) -> "LiteSolution":
        """Build a LiteSolution from ``(SolverResult, Problem)``.

        The Problem is needed for its ``_vars`` map and the LP view.
        We rebuild the LpView once here (same construction polar-high's
        dispatch already ran inside ``solve()``; rebuilding is cheap
        relative to the solve and avoids requiring the dispatch to
        thread it back out).
        """
        from polar_high.solvers._lp_view import LpView

        lp_view = LpView.from_problem(problem)

        # Apply rename to col_names so downstream sees unified names.
        renamed_col_names = [_rename_prefix(nm) for nm in lp_view.col_names]
        renamed_row_names = list(lp_view.row_names)

        # Build LiteVars.  After rename, ``v_invest_p`` and ``v_invest_n``
        # both fold into a single ``v_invest`` entry whose frame is the
        # union of the two (disjoint by construction — process-side vs
        # node-side).  Same for v_divest_*.
        lite_vars: dict[str, _LiteVar] = {}
        for v in problem._vars.values():
            dst_name = _VAR_RENAME.get(v.name, v.name)
            new_var = _LiteVar(
                name=dst_name, dims=tuple(v.dims), frame=v.frame,
            )
            if dst_name in lite_vars:
                # Fold v_invest_p + v_invest_n into the same _LiteVar.
                prior = lite_vars[dst_name]
                merged_frame = pl.concat(
                    [prior.frame, new_var.frame], how="vertical_relaxed",
                )
                lite_vars[dst_name] = _LiteVar(
                    name=dst_name, dims=prior.dims, frame=merged_frame,
                )
            else:
                lite_vars[dst_name] = new_var

        # Materialise the col_value array from the primal dict, aligned
        # with the (possibly renamed) col_names.  The primal dict keys
        # are the *unrenamed* names (the solver saw the original LP);
        # so look up by the original lp_view name and store under the
        # renamed index.
        primal = result.primal or {}
        n = len(lp_view.col_names)
        col_value = np.empty(n, dtype=np.float64)
        for i, nm in enumerate(lp_view.col_names):
            v = primal.get(nm)
            col_value[i] = float(v) if v is not None else 0.0

        return cls(
            result=result,
            _lp_view=lp_view,
            _vars=lite_vars,
            col_names=renamed_col_names,
            row_names=renamed_row_names,
            col_value=col_value,
            highs=None,
        )

    # ------------------------------------------------------------------
    # Solution-shaped surface
    # ------------------------------------------------------------------

    def value(self, var_name: str) -> pl.DataFrame:
        """Long-form per-variable solution: ``(*dims, value)``.

        Mirrors :meth:`polar_high.Solution.value`.
        """
        v = self._vars[var_name]
        ids = v.frame["col_id"].to_numpy()
        assert self.col_value is not None
        vals = self.col_value[ids]
        if v.dims:
            return v.frame.select(*v.dims).with_columns(value=pl.Series(vals))
        return pl.DataFrame({"value": vals})

    @property
    def obj(self) -> float:
        """Objective value; raises if the solver returned no value
        (e.g. infeasible)."""
        if self.result.objective is None:
            raise ValueError(
                f"solver {self.result.solver_name!r} returned no objective "
                f"(status={self.result.status!s})"
            )
        return float(self.result.objective)

    @property
    def optimal(self) -> bool:
        """``True`` iff the solver reported OPTIMAL status."""
        from polar_high.solvers._base import SolverStatus
        return self.result.status == SolverStatus.OPTIMAL

    @property
    def status(self):
        """Forward the underlying :class:`SolverStatus`."""
        return self.result.status

    # ------------------------------------------------------------------
    # Shim construction
    # ------------------------------------------------------------------

    def make_shim(self):
        """Return a :class:`LpNamesShim` wrapping this LiteSolution.

        The shim exposes the subset of the ``highspy.Highs`` API that
        ``flextool.process_outputs.read_highs_solution.extract_variable``
        and ``handoff_writers`` consume.  Called by ``_output_writer``
        on the commercial-solve path where ``self.highs is None``.
        """
        from flextool.engine_polars._lp_names_shim import LpNamesShim
        return LpNamesShim.from_lite_solution(self)


__all__ = ["LiteSolution", "_LiteVar"]
