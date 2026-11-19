"""The run output greys a commercial solver's log as it greys HiGHS' (item 238); no Tk window is made."""
from __future__ import annotations

from types import SimpleNamespace

from flextool.engine.solve._subprocess_solve import _DISPLAY_NAMES
from flextool.console_markers import SOLVER_BLOCK_STARTS as _SOLVER_BLOCK_STARTS
from flextool.gui.log_view import LogView


def _classify(state: SimpleNamespace, line: str) -> list[str]:
    return LogView.classify(state, line)  # type: ignore[arg-type]


def test_commercial_solver_log_is_one_grey_block() -> None:
    state = SimpleNamespace(highs_block_active=False)
    assert _classify(state, "Solver args (gurobi): Threads=1") == []
    assert _classify(state, "Running Gurobi (/x/gurobi_cl)") == ["highs"]
    assert state.highs_block_active
    assert _classify(state, "Optimal objective  1.000000000e+00") == ["highs"]
    assert _classify(state, "") == ["highs"]
    # Inside the block only a line starting with a warning / error is flagged: warnings amber, errors red.
    assert _classify(state, "Warning: large coefficient range") == ["highs", "solverwarn"]
    assert _classify(state, "WARNING: Number of ranged constraints is 3") == ["highs", "solverwarn"]
    assert _classify(state, "  ERROR:   failed to read") == ["highs", "error"]
    assert _classify(state, "Primal feasibility error 1e-9") == ["highs"]
    # The next phase row ends the block.
    assert _classify(state, "Solver                       1.81s (+0.01) |  191 MB (+0.56) |    18.1 GB (+0) |"
                            "       1 MB (+0)") == ["mem"]
    assert not state.highs_block_active
    assert _classify(state, "after the block") == []


def test_every_commercial_marker_opens_a_block() -> None:
    """The engine's marker names and the shared block starts (``flextool.console_markers``) agree."""
    for name in _DISPLAY_NAMES.values():
        state = SimpleNamespace(highs_block_active=False)
        line = f"Running {name} (/opt/{name.lower()}/bin/solver)"
        assert line.startswith(_SOLVER_BLOCK_STARTS), name
        assert _classify(state, line) == ["highs"] and state.highs_block_active


def test_the_gui_folds_between_the_markers_the_run_prints() -> None:
    """The run's solve summary (``write_outputs.solve_summary_lines``) opens and closes with the GUI's fold markers,
    and its slack rows of one node over several periods make a foldable run."""
    from flextool.gui import log_view
    import importlib

    write_outputs = importlib.import_module("flextool.process_outputs.write_outputs")
    from tests._support import log_lint

    assert log_view.SUMMARY_START == write_outputs.SOLVE_SUMMARY_START == log_lint.SUMMARY_START
    assert log_view.SUMMARY_END == write_outputs.SOLVE_SUMMARY_END == log_lint.SUMMARY_END
    text = ('"' + write_outputs.SUMMARY_TITLE + '2026-11-19 10:00:00+00:00"\n\n\n'
            '"Solve","Objective","Total cost from solver, includes all penalty costs"\ninv,1.5\n\n'
            + "".join(f"Created, n1, p{y}, 0.5\n" for y in range(4)) + "Removed, n1, p0, 1\n")
    lines = write_outputs.solve_summary_lines(text)
    assert lines[0] == log_view.SUMMARY_START and lines[-1] == log_view.SUMMARY_END
    assert lines[1].startswith('"Solve","Objective"') and not any("Output at (UTC)" in line for line in lines)
    keys = [log_view.similar_row_key(line) for line in lines]
    first = lines.index("Created, n1, p0, 0.5")
    assert keys[first:first + 4] == ["Created"] * 4 and keys[first - 1] != "Created" != keys[first + 4]
