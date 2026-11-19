"""The run output greys a commercial solver's log as it greys HiGHS' (item 238); no Tk window is made."""
from __future__ import annotations

from types import SimpleNamespace

from flextool.engine.solve._subprocess_solve import _DISPLAY_NAMES
from flextool.gui.execution_window import _SOLVER_BLOCK_STARTS
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
    """The engine's marker names and the GUI's block starts agree (the GUI does not import the engine)."""
    for name in _DISPLAY_NAMES.values():
        state = SimpleNamespace(highs_block_active=False)
        line = f"Running {name} (/opt/{name.lower()}/bin/solver)"
        assert line.startswith(_SOLVER_BLOCK_STARTS), name
        assert _classify(state, line) == ["highs"] and state.highs_block_active
