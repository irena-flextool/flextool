"""Marker lines of a FlexTool run's console output that the GUI's run log and the log tests read.

Standard library only and no other ``flextool`` import, so the GUI (``flextool.gui.log_view``), the engine and
the tests share one copy.
"""
from __future__ import annotations

#: A line that opens a solver's console block, which the run log greys until the next phase row: HiGHS' banner and
#: the line FlexTool prints before a commercial solver's log (``Running <Solver> (<program>)``; the names are the
#: engine's ``_subprocess_solve._DISPLAY_NAMES``, pinned by a test).
SOLVER_BLOCK_STARTS = ("Running HiGHS", "Running Gurobi", "Running CPLEX", "Running Xpress", "Running COPT")

#: The lines around the solve summary printed at the end of a run (the content of ``summary_solve.csv``); the GUI
#: folds similar rows between them.
SOLVE_SUMMARY_START = "--- Solve summary ---"
SOLVE_SUMMARY_END = "--- End of solve summary ---"
