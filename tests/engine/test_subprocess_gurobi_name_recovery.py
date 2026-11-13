"""Regression tests for the commercial-solver (gurobi/cplex/…) subprocess
name-recovery path.

Two problems drove this path's design:

1. The path used to re-read the just-written MPS through
   ``highspy.Highs.readModel`` in the parent to recover variable names —
   which OOM'd on large LPs and hard-failed (``parent failed to read MPS
   back``) on FlexTool's own MPS when a name contained a character HiGHS'
   reader rejects.  Fixed by rebuilding names from the released Problem's
   surviving ``_vars`` / pre-release ``_cstrs`` and wrapping in a
   :class:`_SolHighsShim` — no ``readModel``.

2. Free-format MPS is whitespace-delimited with no portable quoting, so a
   real entity name containing a space (common — e.g. ``Battery Farm``)
   corrupts the file: the solver silently mis-parses the column and
   returns a *wrong* answer.  Fixed by writing the MPS with GENERIC
   ``C0000001`` / ``R0000002`` names (``emit_names=False``) and mapping
   the solver's ``.sol`` back by *index* onto the real names held in
   memory.  The commercial path is now whitespace-agnostic, like HiGHS.

These tests pin the name rebuilders, the generic-id parsers, the
end-to-end index mapping through a fake ``gurobi_cl``, and — when a real
Gurobi (the pip ``gurobipy`` restricted licence is enough) is available —
a genuine solve of a model whose entity name contains a space.
"""
from __future__ import annotations

import importlib.util
import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from polar_high import Param, Problem, Sum

from flextool.engine.solve import _subprocess_solve as sps
from flextool.engine.solve._subprocess_solve import (
    _col_names_from_vars,
    _generic_col_id,
    _generic_row_id,
    _row_names_from_cstrs,
    _SolHighsShim,
    _solve_commercial_subprocess,
)

# The commercial-solver stubs are POSIX shebang scripts made executable via
# chmod; Windows can't spawn them (WinError 193). Skip the module there — the
# name-recovery logic itself is platform-agnostic and covered on Linux/macOS.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="commercial-solver subprocess stubs are POSIX shebang scripts",
)


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------


def _cost_lp(nodes: list[str], costs: list[float] | None = None) -> Problem:
    """``min Σ cost·x[node,p0]  s.t.  Σ x == 15,  0 <= x <= 10``.

    A tiny transport LP whose unique optimum fills cheapest-first.  With
    ``costs=[1,2,3]``, ``cap=10``, ``demand=15`` the optimum is
    ``x=[10,5,0]``, objective ``20``.  ``nodes`` lets a caller inject
    awkward characters (a space) into an entity name.
    """
    if costs is None:
        costs = [1.0, 2.0, 3.0][: len(nodes)]
    periods = ["p0"] * len(nodes)
    idx = pl.DataFrame({"node": nodes, "period": periods})
    pb = Problem()
    x = pb.add_var("x", ("node", "period"), idx, lower=0.0, upper=10.0)
    pb.add_cstr(
        "bal", sense="==",
        lhs_terms={"x": Sum(x, over=("node", "period"))},
        rhs_terms={"d": 15.0},
    )
    cost_frame = pl.DataFrame(
        {"node": nodes, "period": periods, "value": costs}
    )
    pb.set_objective(
        Sum(x * Param(("node", "period"), cost_frame, name="cost"),
            over=("node", "period"))
    )
    return pb


def _expected_names(nodes: list[str]) -> list[str]:
    return [f"x[{n},p0]" for n in nodes]


# ---------------------------------------------------------------------------
# 1. Name rebuilders + generic-id parsers (unit)
# ---------------------------------------------------------------------------


def test_col_names_dimensioned_dense_and_ordered() -> None:
    pb = _cost_lp(["A", "B", "C"])
    names = _col_names_from_vars(pb)
    assert names == _expected_names(["A", "B", "C"])
    assert "" not in names and len(names) == 3


def test_col_names_survive_bracket_comma_space() -> None:
    # Entity names with a space (the character that corrupts free-format
    # MPS) and a comma; brackets come from the family format.
    pb = _cost_lp(["Battery Farm", "Grid,North"], costs=[1.0, 2.0])
    assert _col_names_from_vars(pb) == ["x[Battery Farm,p0]", "x[Grid,North,p0]"]


def test_col_names_recoverable_after_release(tmp_path: Path) -> None:
    """``write_mps(release=True)`` must leave ``_vars`` intact for column
    name recovery — the state the commercial path recovers names in."""
    pb = _cost_lp(["A", "B", "C"])
    pb.write_mps(str(tmp_path / "m.mps"), release=True, emit_names=False)
    assert getattr(pb, "_released", False) is True
    assert _col_names_from_vars(pb) == _expected_names(["A", "B", "C"])


def test_row_names_from_cstrs_before_release() -> None:
    """Row names mirror the ``family[dims]`` format, in declaration /
    row-id order — and must be read BEFORE release drops ``_cstrs``."""
    pb = _cost_lp(["A", "B"], costs=[1.0, 2.0])
    # The single scalar-``over`` constraint 'bal' has no dims → bare name.
    assert _row_names_from_cstrs(pb) == ["bal"]

    # A dimensioned constraint yields one row name per row.
    idx = pl.DataFrame({"node": ["A", "B"], "period": ["p0", "p0"]})
    pb2 = Problem()
    y = pb2.add_var("y", ("node", "period"), idx, lower=0.0)
    pb2.add_cstr(
        "cap", over=idx, sense="<=",
        lhs_terms={"y": y}, rhs_terms={"c": 5.0},
    )
    pb2.set_objective(Sum(y, over=("node", "period")))
    assert _row_names_from_cstrs(pb2) == ["cap[A,p0]", "cap[B,p0]"]


def test_row_names_dropped_by_release(tmp_path: Path) -> None:
    pb = _cost_lp(["A", "B"], costs=[1.0, 2.0])
    pb.write_mps(str(tmp_path / "m.mps"), release=True, emit_names=False)
    # _cstrs is cleared on release → empty; this is WHY the path snapshots
    # row names before writing the MPS.
    assert _row_names_from_cstrs(pb) == []


def test_col_names_scalar_var_uses_bare_family_name() -> None:
    """A scalar (0-dim) variable maps to the bare family name at its
    ``col_id`` (mirrors ``engine._canonicalise``'s else-branch)."""
    class _FakeVar:
        def __init__(self, name, dims, frame):
            self.name, self.dims, self.frame = name, dims, frame

    class _FakeProblem:
        def __init__(self, vars_map):
            self._vars = vars_map

    x = _FakeVar("x", ("node",),
                 pl.DataFrame({"node": ["A", "B"], "col_id": [0, 1]}))
    slack = _FakeVar("slack", (), pl.DataFrame({"col_id": [2]}))
    names = _col_names_from_vars(_FakeProblem({"x": x, "slack": slack}))
    assert names == ["x[A]", "x[B]", "slack"]


def test_generic_id_parsers() -> None:
    # write_mps(emit_names=False): col_id j -> C{j+1:07d}; row_id i -> R{i+2:07d}
    assert _generic_col_id("C0000001") == 0
    assert _generic_col_id("C0000042") == 41
    assert _generic_row_id("R0000002") == 0
    assert _generic_row_id("R0000003") == 1
    # Wider fields (past 7 digits, >10M rows/cols) still parse.
    assert _generic_col_id("C10000001") == 10_000_000
    # Non-generic names (reserved 'cost' row, real names) → None.
    assert _generic_col_id("cost") is None
    assert _generic_row_id("cost") is None
    assert _generic_col_id("x[coal,p0]") is None


# ---------------------------------------------------------------------------
# 2. End-to-end commercial path via a fake gurobi_cl (no solver needed)
# ---------------------------------------------------------------------------


def _write_fake_gurobi(tmp_path: Path, sol_lines: list[str], stdout: str = "", *, solve_mps: bool = False) -> Path:
    """Executable stub mimicking ``gurobi_cl``: read ``ResultFile=<path>``
    from argv, write *sol_lines* there (Gurobi ResultFile format), print
    *stdout*, record its argv in ``<tmp_path>/argv.json``, exit 0.

    With *solve_mps* it instead solves the model file (the last argument) with
    highspy and writes a Gurobi JSON result (status 2, objective, the non-zero
    values), so a whole FlexTool run can go through it."""
    body = textwrap.dedent(
        """\
        #!{py}
        import json, sys
        result = None
        for a in sys.argv[1:]:
            if a.startswith("ResultFile="):
                result = a.split("=", 1)[1]
        assert result is not None, "fake gurobi_cl: no ResultFile= in argv"
        with open({argv_log!r}, "a") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\\n")
        payload = {payload!r}
        if {solve_mps!r}:
            import highspy
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(sys.argv[-1])
            h.run()
            assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
            names = h.allVariableNames()
            values = h.getSolution().col_value
            payload = json.dumps({{
                "SolutionInfo": {{"Status": 2, "ObjVal": h.getInfo().objective_function_value, "SolCount": 1}},
                "Vars": [{{"VarName": n, "X": v}} for n, v in zip(names, values) if v != 0.0],
            }})
        with open(result, "w") as fh:
            fh.write(payload)
        sys.stdout.write({stdout!r})
        sys.exit(0)
        """
    ).format(py=sys.executable, payload="\n".join(sol_lines) + "\n", stdout=stdout,
             argv_log=str(tmp_path / "argv.json"), solve_mps=solve_mps)
    stub = tmp_path / "fake_gurobi_cl"
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return stub


def _argv_of_calls(tmp_path: Path) -> list[list[str]]:
    """The argv of each call of the stub, in order."""
    return [json.loads(line) for line in (tmp_path / "argv.json").read_text().splitlines()]


def test_commercial_path_maps_generic_primal_by_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The MPS carries generic names, so the ``.sol`` does too; the path
    maps them back by index onto the real ``_vars`` names.  Entity names
    here contain a space — which is exactly why generic names are used."""
    nodes = ["coal fired", "gas", "wind"]      # space in first name
    pb = _cost_lp(nodes)
    # Gurobi echoes MPS (generic) names: C{col_id+1:07d}.
    sol_lines = [
        "# Objective value = 20",
        "C0000001 10",   # col_id 0 -> x[coal fired,p0]
        "C0000002 5",    # col_id 1 -> x[gas,p0]
        "C0000003 0",    # col_id 2 -> x[wind,p0]
    ]
    # A text result file has no status: Gurobi's console line says the LP finished.
    stub = _write_fake_gurobi(tmp_path, sol_lines, stdout="Optimal objective  2.000000000e+01\n")
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)

    sol = _solve_commercial_subprocess(
        pb, "gurobi", options=None, solve_name="unit",
        logger=None, work_folder=None,
    )
    assert sol.optimal is True
    assert isinstance(sol.highs, _SolHighsShim)
    # Real (space-bearing) names recovered from _vars, primal by index.
    assert list(sol.highs.allVariableNames()) == _expected_names(nodes)
    np.testing.assert_allclose(sol.col_value, [10.0, 5.0, 0.0])
    assert sol.row_names == []          # Gurobi .sol carries no duals
    assert sol.row_dual.size == 0


def test_commercial_path_missing_primal_defaults_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.sol`` omitting a column defaults it to 0.0 — columns come from
    ``_vars``, not the .sol."""
    pb = _cost_lp(["A", "B"], costs=[1.0, 2.0])
    sol_lines = ["# Objective value = 5", "C0000001 5"]   # C0000002 absent
    stub = _write_fake_gurobi(tmp_path, sol_lines)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)

    sol = _solve_commercial_subprocess(
        pb, "gurobi", options=None, solve_name="unit",
        logger=None, work_folder=None,
    )
    assert list(sol.highs.allVariableNames()) == _expected_names(["A", "B"])
    np.testing.assert_allclose(sol.col_value, [5.0, 0.0])


def test_commercial_path_no_real_names_in_mps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The written MPS must contain only generic names — never a real
    (space-bearing) entity name that would corrupt free-format MPS."""
    nodes = ["coal fired", "gas", "wind"]
    pb = _cost_lp(nodes)
    stub = _write_fake_gurobi(tmp_path, ["# Objective value = 0", "C0000001 0"])
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)

    work = tmp_path / "work"
    _solve_commercial_subprocess(
        pb, "gurobi", options=None, solve_name="keep",
        logger=None, work_folder=work,
    )
    mps_text = (work / "solve_data" / "subprocess" / "keep.mps").read_text()
    assert "coal fired" not in mps_text
    assert "coal" not in mps_text          # no real entity token at all
    assert "C0000001" in mps_text          # generic names present



# ---------------------------------------------------------------------------
# 3. Real commercial solvers (community / restricted licences) — the true
#    whitespace test, one parametrized case per solver.
# ---------------------------------------------------------------------------
#
# FlexTool's commercial path spawns a *CLI binary* (gurobi_cl / cplex /
# optimizer / copt_cmd), never the in-process Python API.  The pip wheels
# (gurobipy / cplex / xpress / coptpy) ship the optimizer + a size-limited
# community licence + the Python API but NOT that CLI binary.  Each shim
# below is a CLI-shaped entry point backed by the wheel, so the real
# subprocess path runs end-to-end with a genuine solve wherever the wheel is
# installed; the case skips cleanly otherwise.  Each shim parses exactly the
# argv / stdin script FlexTool's ``_SCRIPTS[solver]`` builder emits.

_SOLVER_MODULE = {
    "gurobi": "gurobipy",
    "cplex": "cplex",
    "xpress": "xpress",
    "copt": "coptpy",
}

# Shim bodies (plain strings — no test-time interpolation; the shim reads
# argv / stdin at its own runtime).  Shebang + sys.executable is prepended
# when the file is written.
_SHIM_BODY = {
    "gurobi": '''
import sys, gurobipy as gp
model = result = None
for a in sys.argv[1:]:
    if a.startswith("ResultFile="):
        result = a.split("=", 1)[1]
    elif "=" not in a or a.lower().endswith((".mps", ".lp")):
        model = a
m = gp.read(model)
m.setParam("OutputFlag", 0)
m.optimize()
if result is not None:
    m.write(result)
''',
    "cplex": '''
import sys, cplex
c = cplex.Cplex()
for s in (c.set_log_stream, c.set_results_stream,
          c.set_warning_stream, c.set_error_stream):
    s(None)
for line in sys.stdin.read().splitlines():
    t = line.split()
    if not t:
        continue
    k = t[0].lower()
    if k == "read":
        c.read(t[1])
    elif k == "optimize":
        c.solve()
    elif k == "write":
        c.solution.write(t[1])
''',
    "xpress": '''
import sys, xpress as xp
p = xp.problem()
p.controls.outputlog = 0
for line in sys.stdin.read().splitlines():
    t = line.split()
    if not t:
        continue
    k = t[0].lower()
    if k == "readprob":
        p.readProb(t[1])
    elif k == "lpoptimize":
        p.optimize()
    elif k == "writeslxsol":
        p.writeSlxSol(t[1], "")
    elif k == "writeprtsol":
        p.writePrtSol(t[1])
''',
    "copt": '''
import sys, coptpy as cp
env = cp.Envr()
m = env.createModel()
# The log stays on, as copt_cmd's is by default: FlexTool reads how the solve ended from it.
for line in sys.stdin.read().splitlines():
    t = line.split()
    if not t:
        continue
    k = t[0].lower()
    if k == "read":
        m.read(t[1])
    elif k == "optimize":
        m.solve()
    elif k == "write":
        m.write(t[1])
''',
}

_LICENCE_SKIP_HINTS = (
    "license", "licence", "size-limited", "size limit",
    "too large", "size limitations",
)


def _make_solver_shim(solver: str, tmp_path: Path) -> Path | None:
    """Write a CLI shim for *solver* backed by its Python wheel, or return
    ``None`` when the wheel isn't importable (→ skip)."""
    if importlib.util.find_spec(_SOLVER_MODULE[solver]) is None:
        return None
    stub = tmp_path / f"{solver}_cli"
    stub.write_text(
        "#!" + sys.executable + "\n" + textwrap.dedent(_SHIM_BODY[solver])
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return stub


@pytest.mark.parametrize("solver", ["gurobi", "cplex", "xpress", "copt"])
def test_real_commercial_solver_handles_spaces(
    solver: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end with a REAL solve per commercial solver: a model whose
    entity name contains a space must solve to the correct optimum and map
    the primal back onto the real names.  Before the generic-name fix a
    space silently produced a WRONG answer (or crashed).  Skips where the
    solver's wheel / licence is unavailable."""
    shim = _make_solver_shim(solver, tmp_path)
    if shim is None:
        pytest.skip(f"{_SOLVER_MODULE[solver]} not installed")
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: shim)

    nodes = ["coal fired", "gas", "wind"]     # space in a real entity name
    pb = _cost_lp(nodes)
    try:
        sol = _solve_commercial_subprocess(
            pb, solver, options=None, solve_name="real",
            logger=None, work_folder=None,
        )
    except RuntimeError as exc:
        if any(h in str(exc).lower() for h in _LICENCE_SKIP_HINTS):
            pytest.skip(f"{solver}: no usable licence ({exc})")
        raise

    assert sol.optimal is True
    assert isinstance(sol.highs, _SolHighsShim)
    assert list(sol.highs.allVariableNames()) == _expected_names(nodes)
    by_name = dict(zip(sol.highs.allVariableNames(), sol.col_value))
    # cheapest-first fill: coal 10, gas 5, wind 0 — mapped onto real names.
    assert by_name["x[coal fired,p0]"] == pytest.approx(10.0, abs=1e-6)
    assert by_name["x[gas,p0]"] == pytest.approx(5.0, abs=1e-6)
    assert by_name["x[wind,p0]"] == pytest.approx(0.0, abs=1e-6)
    # Objective recovered for every solver (COPT via the '=?'-tolerant
    # regex, Xpress via writeprtsol).
    assert float(sol.obj) == pytest.approx(20.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Options, status and messages of the commercial path (item 230), through the fake gurobi_cl
# ---------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402

from flextool.cli.cmd_run_flextool import build_parser  # noqa: E402
from flextool.engine.input._solve_config import SolverConfig  # noqa: E402
from flextool.engine.solve._acceptance import classify_acceptance  # noqa: E402
from flextool.engine.solve._solver_dispatch import run_one_solve  # noqa: E402
from flextool.run_options import RunOptions  # noqa: E402

_GUROBI_TIME_LIMIT_LOG = (
    "Explored 2055 nodes (12153 simplex iterations) in 0.30 seconds (0.34 work units)\n"
    "\n"
    "Time limit reached\n"
    "Best objective 1.000000000000e+02, best bound 9.900000000000e+01, gap 1.0101%\n"
)
_SOL_20 = ["# Objective value = 20", "C0000001 10", "C0000002 5", "C0000003 0"]


def _mip_lp(nodes: list[str]) -> Problem:
    """``_cost_lp`` with an extra integer column, so the problem is a MIP."""
    pb = _cost_lp(nodes)
    pb.add_var("u", ("node",), pl.DataFrame({"node": ["n"]}), lower=0.0, upper=1.0, integer=True)
    return pb


def _gurobi(stub: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)


def test_gurobi_time_limit_from_the_console(tmp_path, monkeypatch) -> None:
    """A text result file has no status: ``Time limit reached`` on the console makes it TIME_LIMIT, and the
    gap line gives the gap; a MIP is then kept with a warning naming it."""
    _gurobi(_write_fake_gurobi(tmp_path, _SOL_20, stdout=_GUROBI_TIME_LIMIT_LOG), monkeypatch)
    sol = _solve_commercial_subprocess(
        _mip_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit", logger=None, work_folder=None,
    )
    assert (sol.shim_status, sol.optimal) == ("TIME_LIMIT", False)
    assert sol.shim_mip_gap == pytest.approx(0.010101)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert acc.accepted and acc.limit_stopped and "1.0101%" in acc.message


def test_gurobi_without_the_time_limit_line_is_optimal(tmp_path, monkeypatch) -> None:
    _gurobi(_write_fake_gurobi(tmp_path, _SOL_20, stdout="Optimal solution found\n"), monkeypatch)
    sol = _solve_commercial_subprocess(
        _cost_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit", logger=None, work_folder=None,
    )
    assert (sol.shim_status, sol.optimal) == ("OPTIMAL", True)


def test_gurobi_json_result_gives_status_and_gap_with_logging_off(tmp_path, monkeypatch, caplog) -> None:
    """The JSON result file carries the status whatever the console says; it lists non-zero values only, so a
    missing column is a zero, not a warning."""
    payload = json.dumps({
        "SolutionInfo": {"Status": 9, "ObjVal": 20, "ObjBound": 19.5, "MIPGap": 0.025, "SolCount": 2},
        "Vars": [{"VarName": "C0000001", "X": 10}, {"VarName": "C0000002", "X": 5}],
    })
    _gurobi(_write_fake_gurobi(tmp_path, [payload]), monkeypatch)
    with caplog.at_level(logging.WARNING):
        sol = _solve_commercial_subprocess(
            _cost_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit",
            logger=logging.getLogger("t"), work_folder=None,
        )
    assert sol.shim_status == "TIME_LIMIT" and sol.shim_mip_gap == pytest.approx(0.025)
    np.testing.assert_allclose(sol.col_value, [10.0, 5.0, 0.0])
    assert sol.obj == pytest.approx(20.0)
    assert not [r for r in caplog.records if "absent" in r.getMessage()]
    # An LP stopped at its limit stops the run.
    assert not classify_acceptance(sol, ranges_post=None, solve_name="unit").accepted


_GUROBI_JSON_FIXTURES = Path(__file__).parent / "fixtures" / "gurobi_json"


def _knapsack_value(i: int) -> int:
    """Item values of ``time_limited_mip.json`` (``fixtures/gurobi_json/make_fixtures.py``)."""
    return 100 + (i * 7919) % 901


def _knapsack_weight(i: int, k: int) -> int:
    return 50 + ((i + 1) * (k + 3) * 104729) % 951


def test_gurobi_json_files_written_by_gurobi_parse() -> None:
    """Result files written by gurobipy 13.0.2 (``Model.write("<x>.json")``, the format of ``gurobi_cl
    ResultFile=<x>.json``), generated once by ``fixtures/gurobi_json/make_fixtures.py`` and committed.

    * optimal MIP: min -5 x - 4 y, 6 x + 4 y <= 24, x + 2 y <= 6, integer: x = 4, y = 0 (the only non-zero, so
      the only ``Vars`` entry), objective -20, gap 0;
    * time-limited 1500-item, 40-row binary knapsack: status 9, the incumbent's objective is the sum of the
      chosen items' (negated) values, each row holds, and the gap is (ObjVal - ObjBound) / |ObjVal|;
    * infeasible LP (x + y >= 3, x + y <= 1): status 3 and no ``Vars``, so no solution.
    """
    optimal = sps._parse_gurobi_json(_GUROBI_JSON_FIXTURES / "optimal_mip.json")
    assert optimal is not None
    assert (optimal.status, optimal.objective, optimal.primal, optimal.mip_gap) == (
        "OPTIMAL", -20.0, {"C0000001": 4.0}, 0.0,
    )

    limited = sps._parse_gurobi_json(_GUROBI_JSON_FIXTURES / "time_limited_mip.json")
    assert limited is not None and limited.status == "TIME_LIMIT"
    chosen = [int(name[1:]) - 1 for name, x in limited.primal.items() if x == 1.0]
    assert len(chosen) == len(limited.primal) > 0           # binaries: every listed value is 1
    assert limited.objective == pytest.approx(-sum(_knapsack_value(i) for i in chosen))
    for k in range(40):
        capacity = sum(_knapsack_weight(i, k) for i in range(1500)) // 2
        assert sum(_knapsack_weight(i, k) for i in chosen) <= capacity
    info = json.loads((_GUROBI_JSON_FIXTURES / "time_limited_mip.json").read_text())["SolutionInfo"]
    assert limited.mip_gap == pytest.approx((info["ObjVal"] - info["ObjBound"]) / abs(info["ObjVal"]))
    assert 0.0 < limited.mip_gap < 0.01

    infeasible = sps._parse_gurobi_json(_GUROBI_JSON_FIXTURES / "infeasible_lp.json")
    assert infeasible is not None
    assert (infeasible.status, infeasible.objective, infeasible.primal, infeasible.mip_gap) == (
        "INFEASIBLE", None, None, None,
    )


def test_gurobi_text_result_with_another_limit_keeps_a_mip(tmp_path, monkeypatch) -> None:
    """A text result file with ``Node limit reached`` on the console is ``NODE_LIMIT`` with the console's gap: a
    MIP keeps the solution with a warning naming the limit, its option and the gap (item 246)."""
    log = (
        "Node limit reached\n"
        "Best objective 2.000000000000e+01, best bound 1.900000000000e+01, gap 5.0000%\n"
    )
    _gurobi(_write_fake_gurobi(tmp_path, _SOL_20, stdout=log), monkeypatch)
    sol = _solve_commercial_subprocess(
        _mip_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit", logger=None, work_folder=None,
    )
    assert (sol.shim_status, sol.optimal, sol.shim_solver) == ("NODE_LIMIT", False, "gurobi")
    assert sol.shim_mip_gap == pytest.approx(0.05)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert acc.accepted and acc.limit_stopped and acc.model_status == "kSolutionLimit", acc
    assert "stopped at its node limit" in acc.message and "Gurobi NodeLimit" in acc.message
    assert "(relative MIP gap 5.0000%)" in acc.message


def _gurobi_json(status: int, *, with_vars: bool) -> str:
    info = {"Status": status, "ObjVal": 20, "ObjBound": 19, "MIPGap": 0.05, "SolCount": int(with_vars)}
    payload: dict = {"SolutionInfo": info}
    if with_vars:
        payload["Vars"] = [{"VarName": "C0000001", "X": 10}, {"VarName": "C0000002", "X": 5}]
    return json.dumps(payload)


@pytest.mark.parametrize(("status", "name", "model_status", "options"), [
    (8, "NODE_LIMIT", "kSolutionLimit", "Gurobi NodeLimit"),
    (10, "SOLUTION_LIMIT", "kSolutionLimit", "Gurobi SolutionLimit"),
    (16, "WORK_LIMIT", "kTimeLimit", "Gurobi WorkLimit"),
    (15, "USER_OBJ_LIMIT", "kObjectiveTarget", "Gurobi BestObjStop or BestBdStop"),
])
def test_gurobi_json_limit_with_a_solution_keeps_a_mip(status, name, model_status, options, tmp_path,
                                                       monkeypatch) -> None:
    _gurobi(_write_fake_gurobi(tmp_path, [_gurobi_json(status, with_vars=True)]), monkeypatch)
    sol = _solve_commercial_subprocess(
        _mip_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit", logger=None, work_folder=None,
    )
    assert (sol.shim_status, sol.shim_mip_gap) == (name, pytest.approx(0.05))
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert acc.accepted and acc.limit_stopped and acc.model_status == model_status, acc
    assert f"({options}, in the solver's option file" in acc.message and "5.0000%" in acc.message


@pytest.mark.parametrize(("status", "reason"), [
    (10, "stopped at its solution limit before finding a solution"),
    (15, "stopped at its objective limit (BestObjStop or BestBdStop) before finding a solution"),
    (6, "found no solution better than its objective cutoff (Cutoff)"),
])
def test_gurobi_json_stop_without_a_solution_names_it(status, reason, tmp_path, monkeypatch) -> None:
    _gurobi(_write_fake_gurobi(tmp_path, [_gurobi_json(status, with_vars=False)]), monkeypatch)
    with pytest.raises(RuntimeError, match=re.escape(reason)):
        _solve_commercial_subprocess(
            _mip_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit", logger=None, work_folder=None,
        )


def test_gurobi_text_result_without_a_status_line_is_not_optimal(tmp_path, monkeypatch) -> None:
    """Review L2-2: a text result file and a console without a status line (``LogToConsole 0``): how the solve
    ended is unknown (a node limit would read the same), so it is ``NO_STATUS`` and the run stops, saying why."""
    _gurobi(_write_fake_gurobi(tmp_path, _SOL_20, stdout=""), monkeypatch)
    sol = _solve_commercial_subprocess(
        _mip_lp(["A", "B", "C"]), "gurobi", options=None, solve_name="unit", logger=None, work_folder=None,
    )
    assert (sol.shim_status, sol.optimal) == ("NO_STATUS", False)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert not acc.accepted and "LogToConsole 1" in acc.message and "(solver status NO_STATUS)" in acc.message


def _cplex_file(tmp_path: Path, status_string: str, status_value: str | None) -> Path:
    path = tmp_path / "x.sol"
    value = "" if status_value is None else f' solutionStatusValue="{status_value}"'
    path.write_text(
        '<?xml version="1.0"?>\n<CPLEXSolution version="1.2">\n'
        f' <header objectiveValue="20"{value} solutionStatusString="{status_string}"/>\n'
        ' <variables><variable name="C0000001" index="0" value="10"/></variables>\n'
        '</CPLEXSolution>\n'
    )
    return path


@pytest.mark.parametrize("status_value, status_string, expected", [
    # The codes of the CPLEX reference manual ("Solution status codes"), with the string CPLEX writes beside each.
    ("1", "optimal", "OPTIMAL"),
    ("101", "integer optimal solution", "OPTIMAL"),
    ("102", "integer optimal, tolerance", "OPTIMAL"),
    ("5", "optimal with unscaled infeasibilities", "OPTIMAL_INFEASIBLE"),
    ("115", "integer optimal with unscaled infeasibilities", "OPTIMAL_INFEASIBLE"),
    ("20", "optimal face unbounded", "OPTIMAL_FACE_UNBOUNDED"),
    ("11", "time limit exceeded", "TIME_LIMIT"),
    ("107", "time limit exceeded, integer feasible", "TIME_LIMIT"),
    ("131", "deterministic time limit exceeded, integer feasible", "TIME_LIMIT"),
    ("105", "node limit exceeded, integer feasible", "NODE_LIMIT"),
    ("113", "aborted, integer feasible", "INTERRUPTED"),
    # The code decides when the string says something else.
    ("115", "integer optimal solution", "OPTIMAL_INFEASIBLE"),
    # An unlisted code falls back to the string.
    ("999", "integer optimal solution", "OPTIMAL"),
    # No code (an older writer): the string, the unscaled cases before plain "optimal".
    (None, "integer optimal with unscaled infeasibilities", "OPTIMAL_INFEASIBLE"),
    (None, "optimal face unbounded", "OPTIMAL_FACE_UNBOUNDED"),
])
def test_cplex_status_code_with_a_solution(status_value, status_string, expected, tmp_path) -> None:
    """Review L2-3: the numeric ``solutionStatusValue`` first; "optimal with unscaled infeasibilities" and
    "optimal face unbounded" are not optimal."""
    parsed = sps._parse_cplex_sol(_cplex_file(tmp_path, status_string, status_value))
    assert (parsed.status, parsed.primal) == (expected, {"C0000001": 10.0})


@pytest.mark.parametrize("status, reason", [
    ("OPTIMAL_INFEASIBLE", "once unscaled, the solution breaks the feasibility tolerances"),
    ("OPTIMAL_FACE_UNBOUNDED", "has an unbounded optimal face"),
])
def test_cplex_uncertified_optimum_is_rejected_with_its_reason(status, reason) -> None:
    from types import SimpleNamespace

    sol = SimpleNamespace(optimal=False, shim_status=status, shim_mip_gap=None, _vars=_cost_lp(["A"])._vars)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert not acc.accepted and reason in acc.message and f"(solver status {status})" in acc.message


def test_cplex_without_a_solution_file_at_its_time_limit_says_so(tmp_path) -> None:
    """CPLEX writes no file without a solution; its console's ``time limit exceeded`` makes the reason the time
    limit (the run says "stopped at its time limit before finding a solution"), else nothing is known."""
    missing = tmp_path / "none.sol"
    log = "Dual simplex - Time limit exceeded, no solution.\nCPLEX Error  1217: No solution exists.\n"
    assert sps._parse_cplex_sol(missing, log) == sps._ParsedSolution("TIME_LIMIT", None, None, None)
    assert sps._parse_cplex_sol(missing, "CPLEX Error  1217: No solution exists.\n").status == "OTHER"
    assert sps._NO_SOLUTION_REASONS["TIME_LIMIT"] == "stopped at its time limit before finding a solution"


@pytest.mark.parametrize("status_string, expected", [
    ("integer optimal solution", "OPTIMAL"),
    ("integer optimal, tolerance", "OPTIMAL"),
    ("time limit exceeded, integer feasible", "TIME_LIMIT"),
    ("aborted, integer feasible", "INTERRUPTED"),
    ("node limit exceeded, integer feasible", "NODE_LIMIT"),
    ("solution limit exceeded", "SOLUTION_LIMIT"),
    ("integer infeasible or unbounded", "INF_OR_UNBD"),
    ("something new", "OTHER"),
    ("", "OPTIMAL"),     # no status string at all: the values are all there is
])
def test_cplex_status_string_with_a_solution(status_string, expected, tmp_path) -> None:
    """Review L3: a CPLEX file with values is "optimal" only when its status string says so (or has none)."""
    path = tmp_path / "x.sol"
    path.write_text(
        '<?xml version="1.0"?>\n<CPLEXSolution version="1.2">\n'
        f' <header objectiveValue="20" solutionStatusString="{status_string}"/>\n'
        ' <variables><variable name="C0000001" index="0" value="10"/></variables>\n'
        '</CPLEXSolution>\n'
    )
    parsed = sps._parse_cplex_sol(path)
    assert (parsed.status, parsed.primal) == (expected, {"C0000001": 10.0})


def test_cplex_aborted_mip_is_rejected_naming_the_status() -> None:
    """The status of an aborted CPLEX MIP with an incumbent stops the run (it used to be read as optimal)."""
    from types import SimpleNamespace

    sol = SimpleNamespace(optimal=False, shim_status=sps._cplex_status("aborted, integer feasible"),
                          shim_mip_gap=None, _vars=_mip_lp(["A"])._vars)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="unit")
    assert not acc.accepted and "interrupted" in acc.message


def test_gurobi_unknown_parameter_is_a_warning_naming_its_layer(tmp_path, monkeypatch, caplog) -> None:
    _gurobi(_write_fake_gurobi(tmp_path, _SOL_20, stdout="Unknown parameter: 'MIPGapp' (ignored)\n"), monkeypatch)
    with caplog.at_level(logging.WARNING):
        _solve_commercial_subprocess(
            _cost_lp(["A", "B", "C"]), "gurobi", options={"MIPGapp": "0.01"}, solve_name="unit",
            logger=logging.getLogger("t"), work_folder=None,
            option_layers={"MIPGapp": "solver_arguments of solve unit"},
        )
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "MIPGapp" in warnings[0] and "solver_arguments of solve unit" in warnings[0]


def test_flags_reach_gurobi_and_set_the_watchdog(tmp_path, monkeypatch) -> None:
    """End to end through ``run_one_solve``: the flags become Gurobi parameters and the subprocess timeout is
    2 x 60 + 600 = 720 s."""
    _gurobi(_write_fake_gurobi(tmp_path, _SOL_20), monkeypatch)
    (tmp_path / "cfg").mkdir()
    (tmp_path / "cfg" / "gurobi.opt").write_text("Threads 0\nMIPGap 0.002\n")
    timeouts: list = []
    real_run = sps._run_solver_process

    def spy(*args, **kwargs):
        timeouts.append(kwargs.get("watchdog"))
        return real_run(*args, **kwargs)

    monkeypatch.setattr(sps, "_run_solver_process", spy)
    flags = ["sqlite:///x.sqlite", "--solver-time-limit", "60", "--solver-mip-gap", "0.001"]
    run_one_solve(
        _cost_lp(["A", "B", "C"]), SolverConfig(name="gurobi"), solve_name="unit",
        run_options=RunOptions.from_cli(build_parser().parse_args(flags), environ={}),
        solver_config_dir=tmp_path / "cfg",
    )
    (argv,) = _argv_of_calls(tmp_path)
    assert argv[:-2] == ["Threads=0", "MIPGap=0.001", "TimeLimit=60"]
    assert timeouts == [720.0]
