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

from flextool.engine_polars import _subprocess_solve as sps
from flextool.engine_polars._subprocess_solve import (
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


def _write_fake_gurobi(tmp_path: Path, sol_lines: list[str]) -> Path:
    """Executable stub mimicking ``gurobi_cl``: read ``ResultFile=<path>``
    from argv, write *sol_lines* there (Gurobi ResultFile format), exit 0."""
    body = textwrap.dedent(
        """\
        #!{py}
        import sys
        result = None
        for a in sys.argv[1:]:
            if a.startswith("ResultFile="):
                result = a.split("=", 1)[1]
        assert result is not None, "fake gurobi_cl: no ResultFile= in argv"
        with open(result, "w") as fh:
            fh.write({payload!r})
        sys.exit(0)
        """
    ).format(py=sys.executable, payload="\n".join(sol_lines) + "\n")
    stub = tmp_path / "fake_gurobi_cl"
    stub.write_text(body)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IRWXU)
    return stub


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
    stub = _write_fake_gurobi(tmp_path, sol_lines)
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
try:
    m.setParam("Logging", 0)
except Exception:
    pass
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
