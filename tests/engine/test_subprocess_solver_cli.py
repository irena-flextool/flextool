"""Command lines and output checks of the commercial-solver CLI driver.

No solver binary or licence is needed: the argv builders are pure functions and the licence check reads text.
"""
from __future__ import annotations

from pathlib import Path

from flextool.engine.solve._subprocess_solve import _gurobi_script


def test_gurobi_parameters_are_name_value_arguments(tmp_path: Path) -> None:
    """``gurobi_cl`` takes parameters only as ``Name=value`` arguments before the model file; it rejects
    ``ReadParams=<file>`` ("Unknown parameter: 'ReadParams' (ignored)") and then solves at its defaults."""
    opt = tmp_path / "solve.opt"
    opt.write_text("# merged baseline + solver_options\nThreads 0\nMIPGap 0.01\nTimeLimit 3600\n")
    binary, mps, sol = Path("/opt/gurobi/bin/gurobi_cl"), tmp_path / "solve.mps", tmp_path / "solve.sol"

    argv, stdin = _gurobi_script(binary, mps, sol, opt)

    assert argv == [str(binary), "Threads=0", "MIPGap=0.01", "TimeLimit=3600", f"ResultFile={sol}", str(mps)]
    assert stdin is None


def test_gurobi_without_parameters(tmp_path: Path) -> None:
    """No merged file (no baseline and no solver_options): only the result file and the model."""
    binary, mps, sol = Path("gurobi_cl"), tmp_path / "m.mps", tmp_path / "m.sol"

    argv, _ = _gurobi_script(binary, mps, sol, None)

    assert argv == ["gurobi_cl", f"ResultFile={sol}", str(mps)]
