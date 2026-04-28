"""Per-solve preprocessing entry point.

Called from ``orchestration.run_model``'s solve loop, after
``solve_writers``/``blocks.write_block_data_for_solve`` have populated
``solve_data/`` and immediately before ``solver.run`` invokes glpsol.

Two responsibilities:

1. Refresh write_input-scope preprocessing outputs that mod's
   ``if p_model['solveFirst']`` printfs overwrite (they only run on the
   first solve, so subsequent solves would see corrupted CSVs without
   this re-write). The cost is small — these functions read input/
   files and project columns; sub-second on any realistic scenario.

2. Compute and write the per-solve-specific sets (``per_solve_sets.py``)
   whose inputs come from per-solve CSVs.

This is the single hook the migration relies on for solve-loop
correctness; all per-set families dispatch from here.
"""
from __future__ import annotations

from pathlib import Path

from flextool.flextoolrunner.runner_state import RunnerState


def run(state: RunnerState, solve_name: str) -> None:
    """Execute per-solve preprocessing for ``solve_name``.

    The function is idempotent: calling it twice with the same
    ``solve_name`` produces the same outputs.
    """
    wf = state.paths.work_folder
    input_dir = wf / "input"
    solve_data_dir = wf / "solve_data"

    # Per-solve-only sets: inputs in solve_data/, written above by
    # orchestration / solve_writers / blocks before this hook.
    from flextool.flextoolrunner.preprocessing import per_solve_sets
    per_solve_sets.write_per_solve_sets(solve_data_dir)

    # Re-run write_input-scope preprocessing whose output CSV is
    # overwritten by mod's `if p_model['solveFirst']` printfs (which
    # fire after MPS gen on solve 1). Without this refresh, multi-solve
    # scenarios would read the post-printf state of these files on
    # solve 2 and find the wrong header columns.
    #
    # Safe to call any number of times — each function is a pure
    # function of input/ CSVs, which don't change between solves.
    from flextool.flextoolrunner.preprocessing import (
        method_with_fallback_sets,
    )
    method_with_fallback_sets.write_node_inflow_method(input_dir, solve_data_dir)
    method_with_fallback_sets.write_node_storage_binding_method(input_dir, solve_data_dir)
