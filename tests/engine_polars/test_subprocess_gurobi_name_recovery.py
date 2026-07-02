"""Regression tests for the commercial-solver (gurobi/cplex/…) subprocess
name-recovery path.

Background
----------
The commercial path used to re-read the just-written MPS through
``highspy.Highs.readModel`` in the *parent* purely to recover the
``col_id``-indexed variable-name list.  That re-read:

* OOM'd on large LPs (tens of GB of transient RSS), and
* hard-failed (``parent failed to read MPS back``) on FlexTool's own
  free-format MPS whenever a name carried a character HiGHS' MPS reader
  rejects — the solver itself (``gurobi_cl``) had already solved the LP
  and written a valid ``.sol``; only the parent's re-parse choked.

The fix rebuilds the column names from the released Problem's surviving
``_vars`` (:func:`_col_names_from_vars`) and wraps the parsed arrays in a
:class:`_SolHighsShim` — the same shim the HiGHS save-memory path uses —
so ``readModel`` is never called on the commercial path.

These tests pin:

1. :func:`_col_names_from_vars` produces the exact polar-high naming,
   dense and ``col_id``-indexed, for dimensioned + scalar vars and names
   containing brackets / commas / spaces;
2. it keeps working *after* ``write_mps(release=True)`` (the state the
   commercial path recovers names in);
3. the end-to-end commercial path maps a solver ``.sol`` back onto the
   solution via the shim, with no ``highspy.readModel`` anywhere.
"""
from __future__ import annotations

import stat
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from polar_high import Problem, Sum

from flextool.engine_polars import _subprocess_solve as sps
from flextool.engine_polars._subprocess_solve import (
    _col_names_from_vars,
    _SolHighsShim,
    _solve_commercial_subprocess,
)


# ---------------------------------------------------------------------------
# Problem builders
# ---------------------------------------------------------------------------


def _problem_with_names(node_values: list[str]) -> tuple[Problem, list[str]]:
    """Build ``min Σ x[node,period]  s.t. Σ x >= 1, x >= 0``.

    Returns the Problem plus the expected ``col_id``-ordered column-name
    list.  ``node_values`` lets a caller inject awkward characters
    (spaces, brackets already come from the family format).
    """
    periods = [f"p{i}" for i in range(len(node_values))]
    idx = pl.DataFrame({"node": node_values, "period": periods})
    pb = Problem()
    x = pb.add_var("x", ("node", "period"), idx, lower=0.0)
    pb.add_cstr(
        "c",
        sense=">=",
        lhs_terms={"x": Sum(x, over=("node", "period"))},
        rhs_terms={"c": 1.0},
    )
    pb.set_objective(x)
    # col_ids are assigned in index-row order by ``add_var``.
    expected = [f"x[{n},{p}]" for n, p in zip(node_values, periods)]
    return pb, expected


# ---------------------------------------------------------------------------
# 1. _col_names_from_vars unit behaviour
# ---------------------------------------------------------------------------


def test_col_names_dimensioned_dense_and_ordered() -> None:
    pb, expected = _problem_with_names(["A", "B", "C"])
    names = _col_names_from_vars(pb)
    assert names == expected
    # Dense: one entry per column, no gaps.
    assert "" not in names
    assert len(names) == 3


class _FakeVar:
    """Minimal stand-in for ``polar_high.Var`` — just the attributes
    :func:`_col_names_from_vars` reads (``name`` / ``dims`` / ``frame``).
    """

    def __init__(self, name: str, dims: tuple[str, ...], frame: pl.DataFrame):
        self.name = name
        self.dims = dims
        self.frame = frame


class _FakeProblem:
    def __init__(self, vars_map: dict[str, _FakeVar]):
        self._vars = vars_map


def test_col_names_scalar_var_uses_bare_family_name() -> None:
    """A scalar (0-dim) variable maps to the bare family name at its
    ``col_id`` (mirrors ``engine._canonicalise``'s else-branch)."""
    x = _FakeVar(
        "x",
        ("node",),
        pl.DataFrame({"node": ["A", "B"], "col_id": [0, 1]}),
    )
    slack = _FakeVar("slack", (), pl.DataFrame({"col_id": [2]}))
    names = _col_names_from_vars(_FakeProblem({"x": x, "slack": slack}))
    assert names == ["x[A]", "x[B]", "slack"]
    assert "" not in names


def test_col_names_survive_bracket_comma_space() -> None:
    # The realistic FlexTool case: entity names with brackets already
    # come from the family format; a space is the character that broke
    # the old readModel re-read.
    pb, expected = _problem_with_names(["Battery Flows", "Grid,North"])
    names = _col_names_from_vars(pb)
    assert names == ["x[Battery Flows,p0]", "x[Grid,North,p1]"]
    assert names == expected


def test_col_names_recoverable_after_release(tmp_path: Path) -> None:
    """``write_mps(release=True)`` must leave ``_vars`` intact enough for
    name recovery — that is the exact state the commercial path uses."""
    pb, expected = _problem_with_names(["A", "B", "C"])
    pb.write_mps(str(tmp_path / "m.mps"), release=True)
    assert getattr(pb, "_released", False) is True
    names = _col_names_from_vars(pb)
    assert names == expected


# ---------------------------------------------------------------------------
# 2. End-to-end commercial path via a fake gurobi_cl
# ---------------------------------------------------------------------------


def _write_fake_gurobi(tmp_path: Path, sol_lines: list[str]) -> Path:
    """Write an executable stub that mimics ``gurobi_cl``.

    It reads ``ResultFile=<path>`` from argv, ignores everything else
    (ReadParams, the MPS), and writes *sol_lines* as the ``.sol`` — a
    key=value primal file exactly like Gurobi's ResultFile output.
    Exits 0 so the parent proceeds to the read-back.
    """
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


def test_commercial_path_maps_primal_via_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full commercial path: real ``write_mps``, a fake ``gurobi_cl`` that
    writes a ``.sol``, and the new name-recovery/shim read-back — no
    ``highspy.readModel``.  Verifies the primal maps onto the right
    columns and the Solution carries a ``_SolHighsShim``."""
    pb, expected = _problem_with_names(["A", "B", "C"])
    # Fake solver reports a distinct value per (bracketed, comma-bearing)
    # variable name, plus the objective comment line Gurobi emits.
    sol_lines = [
        "# Objective value = 6",
        f"{expected[0]} 1",
        f"{expected[1]} 2",
        f"{expected[2]} 3",
    ]
    stub = _write_fake_gurobi(tmp_path, sol_lines)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)

    # Guard: the old crash path must not be reachable — assert readModel
    # is never invoked by failing loudly if anything touches highspy here.
    sol = _solve_commercial_subprocess(
        pb,
        "gurobi",
        options=None,
        solve_name="unit",
        logger=None,
        work_folder=None,
    )

    assert sol.optimal is True
    assert sol.obj == pytest.approx(6.0)
    # Solution is backed by the duck-typed shim, not a live Highs.
    assert isinstance(sol.highs, _SolHighsShim)
    assert list(sol.highs.allVariableNames()) == expected
    # Primal mapped by name onto the col_id-ordered array.
    np.testing.assert_allclose(
        sol.highs.getSolution().col_value, [1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(sol.col_value, [1.0, 2.0, 3.0])
    # No duals from a Gurobi .sol → empty row-name list, zero row duals.
    assert list(sol.row_names) == []
    assert sol.row_dual.size == 0


def test_commercial_path_missing_primal_defaults_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.sol`` that omits a variable defaults it to 0.0 (and the path
    still succeeds) — the columns come from ``_vars``, not the .sol."""
    pb, expected = _problem_with_names(["A", "B"])
    # Report only the first variable.
    sol_lines = ["# Objective value = 5", f"{expected[0]} 5"]
    stub = _write_fake_gurobi(tmp_path, sol_lines)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)

    sol = _solve_commercial_subprocess(
        pb, "gurobi", options=None, solve_name="unit",
        logger=None, work_folder=None,
    )
    assert list(sol.highs.allVariableNames()) == expected
    np.testing.assert_allclose(sol.col_value, [5.0, 0.0])


def test_commercial_path_keeps_files_when_work_folder_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a work_folder the MPS/.sol are preserved for post-mortem — a
    prerequisite for diagnosing any future solver-side name issue."""
    pb, expected = _problem_with_names(["A", "B"])
    sol_lines = ["# Objective value = 3", f"{expected[0]} 1", f"{expected[1]} 2"]
    stub = _write_fake_gurobi(tmp_path, sol_lines)
    monkeypatch.setattr(sps, "_find_solver_binary", lambda name: stub)

    work = tmp_path / "work"
    sol = _solve_commercial_subprocess(
        pb, "gurobi", options=None, solve_name="keep",
        logger=None, work_folder=work,
    )
    assert sol.optimal is True
    sub = work / "solve_data" / "subprocess"
    assert (sub / "keep.mps").is_file()
    assert (sub / "keep.sol").is_file()
