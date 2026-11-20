"""The investment-only results of a Benders solve (item 287 P3).

A Benders solve whose final solve (``--benders-final-solve``) is off, or did not give an accepted solution, has
no solution of the whole model: only its investment decisions at the incumbent
(``BendersResult.invest_solution_vars``, frames in ``polar_high.Solution.value(name)`` form) and its objective (the
best upper bound).  :func:`write_benders_invest_slice` writes these under the solve's name to ``output_raw/`` the
way a monolithic solve writes them, so the union of the run's solves reports the Benders solve's investments,
retirements and their costs, and its ``summary_solve`` row:

* ``_solve_order.txt`` entry (the union orders the solve before a later dispatch solve);
* ``v_obj`` (the objective = the Benders upper bound);
* the investment decision parquets ``v_invest`` / ``v_divest`` and the per-cohort retirement / life-extension
  ones (``write_v_divest_built`` / ``write_v_extend``), through :class:`_FrameHighs`, a read-only stand-in for the
  HiGHS handle that holds only these columns;
* the cohort capacity and the per-solve parameter and set slices (the capacity and cost tables read them).

Not written: dispatch variables, prices and other duals, the investment marginal values (W-0571 says so), the
capacity margin and reserve reports, the handoff CSVs.  A family the solve does not have is absent, never an empty
part.  Periods the Benders solve realizes but no later solve does have no dispatch rows.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    import polars as pl

_logger = logging.getLogger(__name__)

#: Frame names of ``invest_solution_vars`` → the column name the output writers read (``_writer._VAR_RENAME``).
_COLUMN_NAME = {"v_invest_p": "v_invest", "v_invest_n": "v_invest", "v_divest_p": "v_divest",
                "v_divest_n": "v_divest"}
#: The variable specs and custom writers (``write_all_variables``) of the investment decisions.
_SPECS = ("v_invest", "v_divest")
_CUSTOM_WRITERS = ("v_obj", "v_divest_built", "v_extend")


class _FrameHighs:
    """The investment frames of a Benders solve as a solved HiGHS handle for the output writers: the columns
    ``<name>[<dim values>]`` (the names polar-high gives them, after ``_writer._rename_invest_columns``) with their
    values, reduced costs NaN, no rows, and the objective."""

    def __init__(self, frames: Mapping[str, "pl.DataFrame"], scaled_objective: float):
        names: list[str] = []
        values: list[float] = []
        for family, frame in frames.items():
            column = _COLUMN_NAME.get(family, family)
            dims = [c for c in frame.columns if c != "value"]
            for row in frame.select(*dims, "value").iter_rows():
                names.append(f"{column}[{','.join(str(x) for x in row[:-1])}]")
                values.append(float(row[-1]))
        self._names = names
        self._solution = SimpleNamespace(col_value=np.asarray(values, dtype=float),
                                         col_dual=np.full(len(values), np.nan), row_dual=np.empty(0))
        # ``write_v_obj`` reads this before ``getObjectiveValue`` (the autoscale stash; already in solver units).
        self._flextool_unscaled_objective = scaled_objective

    def allVariableNames(self) -> list[str]:  # noqa: N802 - highspy API
        return self._names

    def getSolution(self) -> SimpleNamespace:  # noqa: N802 - highspy API
        return self._solution

    def getLp(self) -> SimpleNamespace:  # noqa: N802 - highspy API
        return SimpleNamespace(row_names_=[])

    def getObjectiveValue(self) -> float:  # noqa: N802 - highspy API
        return self._flextool_unscaled_objective


def write_benders_invest_slice(frames: Mapping[str, "pl.DataFrame"], objective: float, *, work_folder: Path | str,
                               solve_name: str, flex_data, provider, scale_the_objective: float) -> None:
    """Write the investment-only results of Benders solve *solve_name* (module docstring) to
    ``<work_folder>/output_raw``.  *frames*: ``BendersResult.invest_solution_vars`` (may be empty: a model without
    investment then gets its ``summary_solve`` row only); *objective*: the Benders upper bound (real units);
    *scale_the_objective*: the solve's objective scale (the writers unscale with it)."""
    from flextool.engine.output._cohort_capacity import write_cohort_capacity
    from flextool.engine.solve._orchestration import SnapshotSolution
    from flextool.process_outputs.persist_realized_slice import (
        write_all_params_realized_slice,
        write_all_sets_realized_slice,
    )
    from flextool.process_outputs.read_highs_solution import (
        VARIABLE_SPECS,
        _actual_solve_name,
        _load_realized_periods,
        write_all_variables,
    )
    from flextool.process_outputs.solve_order import append_solve_order

    output_dir = Path(work_folder) / "output_raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    roll_name = _actual_solve_name(solve_name, provider=provider)
    append_solve_order(output_dir, roll_name)

    left_out = getattr(flex_data, "reduced_output_features", None)
    left_out_of = getattr(flex_data, "reduced_output_entities", None)
    if left_out or left_out_of:
        from flextool.process_outputs.reduced_cells import write_reduced_marker

        write_reduced_marker(output_dir, roll_name, list(left_out or ()), left_out_of)

    def _frame(name: str) -> Path | None:
        return (Path(work_folder) / f"{name}.csv"
                if provider is not None and provider.has(name) else None)

    realized_periods_csv = _frame("solve_data/realized_invest_periods_of_current_solve")
    h = _FrameHighs(frames, float(objective) * float(scale_the_objective))
    write_all_variables(
        h, solve_name=roll_name, output_dir=output_dir,
        realized_dispatch_csv=_frame("solve_data/realized_dispatch"),
        realized_periods_csv=realized_periods_csv,
        specs=[s for s in VARIABLE_SPECS if s.name in _SPECS],
        flex_data=flex_data, scale_the_objective=scale_the_objective, provider=provider,
        custom_writers=_CUSTOM_WRITERS,
    )
    # The cohort capacity and the parameter / set slices read the decisions through ``Solution.value(name)``.
    snap = SnapshotSolution(_vars=dict(frames))
    write_cohort_capacity(snap, flex_data, output_dir=output_dir, solve_name=roll_name, provider=provider)
    if flex_data is None:
        return
    invest_periods = _load_realized_periods(realized_periods_csv, provider=provider)
    write_all_params_realized_slice(flex_data, snap, solve_name=roll_name, output_dir=output_dir,
                                    realized_invest_periods=invest_periods)
    write_all_sets_realized_slice(flex_data, snap, solve_name=roll_name, output_dir=output_dir,
                                  realized_invest_periods=invest_periods)


__all__ = ["write_benders_invest_slice"]
