"""Command lines and output checks of the commercial-solver CLI driver.

No solver binary or licence is needed: the argv builders are pure functions and the licence check reads text.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine.solve._subprocess_solve import _gurobi_script, _looks_like_license_error


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


# A normal Gurobi 13 log of an infeasible model: the banner names the licence file and the licence type.
_GUROBI_INFEASIBLE_LOG = """\
Set parameter LogToConsole to value 1
Using license file /Library/gurobi/gurobi.lic
Set parameter ResultFile to value "solve.sol"
Gurobi Optimizer version 13.0.2 build v13.0.2rc0 (mac64[arm] - Darwin 24.6.0 24G90)
Academic license 2655372 - for non-commercial use only - registered to someone@example.org
Restricted license - for non-production use only - expires 2027-11-24
Optimize a model with 1520 rows, 1104 columns and 4300 nonzeros
Presolve time: 0.01s

Model is infeasible
Best objective -, best bound -, gap -
"""


def test_licence_banner_of_an_infeasible_solve_is_not_a_licence_error() -> None:
    assert not _looks_like_license_error(_GUROBI_INFEASIBLE_LOG, "")
    assert not _looks_like_license_error("", _GUROBI_INFEASIBLE_LOG)


@pytest.mark.parametrize("text", [
    "ERROR 10009: No Gurobi license found (user someone, host build01, hostid 1a2b3c4d, cores 8)",
    "Error 10009: License expired 2026-10-31",
    "Error 10009: Request denied: license not valid for this host",
    "Error 10010: HostID mismatch (licensed to 1a2b3c4d, hostid is 5e6f7a8b)",
    "Error 10030: Model too large for size-limited license; visit https://gurobi.com/unrestricted",
    "Error 10022: Unable to contact token server 'tokens.example.org'",
    "Failed to obtain license from the Web License Service",
    "CPLEX Error  32201: ilm: CPLEX: no license found: no such file or directory.",
    "CPLEX Error  1016: Promotional version. Problem size limits exceeded.",
    "?2: Xpress licensing error 8: license file not found",
    "Could not find license file /opt/copt/license.dat",
    "COPT license error: licence has expired",
])
def test_licence_failure_is_a_licence_error(text: str) -> None:
    assert _looks_like_license_error("", text)
    assert _looks_like_license_error(_GUROBI_INFEASIBLE_LOG + text + "\n", "")
