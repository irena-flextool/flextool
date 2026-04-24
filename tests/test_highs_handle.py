"""Tests for :class:`flextool.flextoolrunner.highs_handle.HighsModelHandle`.

Uses a tiny synthetic LP rather than the full FlexTool model so the
persistence helper can be exercised in isolation::

    max  3*x1 + 2*x2 + 1*x3
    s.t. x1 + x2 + x3 <= 10
         0 <= x_i <= 10

Encoded as a minimisation with ``c = [-3, -2, -1]``.  Optimum:
``x = (10, 0, 0)``, objective ``-30``.
"""
from __future__ import annotations

import highspy
import numpy as np
import pytest

from flextool.flextoolrunner.highs_handle import HighsModelHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_lp() -> highspy.Highs:
    """Build the shared synthetic LP and return a silent ``Highs`` instance."""
    h = highspy.Highs()
    h.silent()
    lp = highspy.HighsLp()
    lp.num_col_ = 3
    lp.num_row_ = 1
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = np.array([-3.0, -2.0, -1.0])
    lp.col_lower_ = np.array([0.0, 0.0, 0.0])
    lp.col_upper_ = np.array([10.0, 10.0, 10.0])
    lp.row_lower_ = np.array([-highspy.kHighsInf])
    lp.row_upper_ = np.array([10.0])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array([0, 1, 2, 3])
    lp.a_matrix_.index_ = np.array([0, 0, 0])
    lp.a_matrix_.value_ = np.array([1.0, 1.0, 1.0])
    lp.col_names_ = ["x[1]", "x[2]", "x[3]"]
    lp.row_names_ = ["cap"]
    h.passModel(lp)
    return h


@pytest.fixture
def handle() -> HighsModelHandle:
    """Fresh post-solve ``HighsModelHandle`` on the tiny LP."""
    h = _tiny_lp()
    hndl = HighsModelHandle(h)
    hndl.solve()
    hndl.build_name_maps()
    return hndl


# ---------------------------------------------------------------------------
# Name maps + pattern lookup
# ---------------------------------------------------------------------------
def test_build_name_maps_after_solve(handle: HighsModelHandle) -> None:
    """Name maps populate post-solve with the original column/row names."""
    assert handle.col_by_name == {"x[1]": 0, "x[2]": 1, "x[3]": 2}
    assert handle.row_by_name == {"cap": 0}


def test_cols_matching_pattern(handle: HighsModelHandle) -> None:
    """Glob lookup returns all three synthetic columns."""
    assert handle.cols_matching("x[*]") == [0, 1, 2]
    assert handle.cols_matching("x[1]") == [0]
    assert handle.cols_matching("y[*]") == []
    # '?' wildcard matches exactly one character in the bracketed index.
    assert handle.cols_matching("x[?]") == [0, 1, 2]


def test_rows_matching_pattern(handle: HighsModelHandle) -> None:
    assert handle.rows_matching("cap") == [0]
    assert handle.rows_matching("cap*") == [0]
    assert handle.rows_matching("nope") == []


# ---------------------------------------------------------------------------
# Fix / unfix
# ---------------------------------------------------------------------------
def test_fix_and_unfix_cols(handle: HighsModelHandle) -> None:
    """Fix shifts optimum, unfix restores it."""
    assert handle.objective() == pytest.approx(-30.0)
    x1 = handle.col_by_name["x[1]"]

    # Fix x[1] = 0. Optimum becomes x[2] = 10, obj = -20.
    handle.fix_cols([x1], 0.0)
    handle.solve()
    assert handle.is_optimal()
    assert handle.objective() == pytest.approx(-20.0)
    assert handle.primal([x1])[0] == pytest.approx(0.0)

    # Unfix (default: restore snapshot) returns to original optimum.
    handle.unfix_cols([x1])
    handle.solve()
    assert handle.objective() == pytest.approx(-30.0)
    assert handle.primal([x1])[0] == pytest.approx(10.0)


def test_fix_cols_scalar_vs_sequence(handle: HighsModelHandle) -> None:
    """Both scalar broadcast and per-column sequences accepted."""
    cols = handle.cols_matching("x[*]")
    # Fix all three to zero via scalar.
    handle.fix_cols(cols, 0.0)
    handle.solve()
    assert handle.objective() == pytest.approx(0.0)
    # Unfix with an explicit sequence, then confirm the constraint re-binds.
    handle.unfix_cols(cols, lb=[0.0, 0.0, 0.0], ub=[10.0, 10.0, 10.0])
    handle.solve()
    assert handle.objective() == pytest.approx(-30.0)


def test_unfix_without_snapshot_raises() -> None:
    """unfix_cols with None bounds and no prior fix_cols is an error."""
    h = _tiny_lp()
    hndl = HighsModelHandle(h)
    hndl.solve()
    hndl.build_name_maps()
    with pytest.raises(RuntimeError, match="no snapshot"):
        hndl.unfix_cols([0])


# ---------------------------------------------------------------------------
# Cost modification
# ---------------------------------------------------------------------------
def test_change_costs(handle: HighsModelHandle) -> None:
    """Making x[3] the most attractive variable swings the optimum to it."""
    x3 = handle.col_by_name["x[3]"]
    handle.change_costs([x3], -5.0)  # now x[3] has the largest reward
    handle.solve()
    assert handle.is_optimal()
    assert handle.objective() == pytest.approx(-50.0)
    assert handle.primal([x3])[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Row addition
# ---------------------------------------------------------------------------
def test_add_row_constrains_future_solve(handle: HighsModelHandle) -> None:
    """Adding ``x[1] <= 3`` forces x[2] to pick up the slack."""
    x1 = handle.col_by_name["x[1]"]
    new_idx = handle.add_row("cap_x1", {x1: 1.0}, upper=3.0)
    # Row-index tracking: exactly one new row past the original `cap`.
    assert new_idx == 1
    assert handle.row_by_name["cap_x1"] == new_idx
    assert handle.rows_matching("cap_x1") == [new_idx]

    handle.solve()
    assert handle.is_optimal()
    # x[1] = 3, x[2] = 7, x[3] = 0  →  obj = -3*3 - 2*7 = -23
    assert handle.objective() == pytest.approx(-23.0)
    primals = handle.primal([x1, handle.col_by_name["x[2]"]])
    assert primals[0] == pytest.approx(3.0)
    assert primals[1] == pytest.approx(7.0)


def test_add_row_equality(handle: HighsModelHandle) -> None:
    """lower == upper encodes an equality constraint."""
    x2 = handle.col_by_name["x[2]"]
    handle.add_row("x2_eq_4", {x2: 1.0}, lower=4.0, upper=4.0)
    handle.solve()
    assert handle.is_optimal()
    assert handle.primal([x2])[0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Duals / reduced costs
# ---------------------------------------------------------------------------
def test_reduced_costs_on_fixed_col(handle: HighsModelHandle) -> None:
    """Reduced costs at optimum: zero for degenerate (rc=0) columns,
    nonzero for columns at a bound that are suboptimal to enter.

    At the LP optimum ``x = (10, 0, 0)``:
    * ``x[1] = 10`` is nonbasic at its upper bound, reduced cost = -1.
    * ``x[2] = 0`` is degenerate (reward equal to the row-dual), rc = 0.
    * ``x[3] = 0`` is nonbasic at its lower bound, rc = +1.
    """
    x1 = handle.col_by_name["x[1]"]
    x2 = handle.col_by_name["x[2]"]
    x3 = handle.col_by_name["x[3]"]

    rc = handle.reduced_costs([x1, x2, x3])
    assert rc[0] == pytest.approx(-1.0, abs=1e-9)
    assert rc[1] == pytest.approx(0.0, abs=1e-9)
    assert rc[2] == pytest.approx(1.0, abs=1e-9)

    # Fix x[3] to a suboptimal positive value: x[3] = 2 steals capacity
    # from x[1] (which drops to 8, off its upper bound).  The row-dual
    # now reflects x[1]'s gradient (-3), and x[3]'s reduced cost
    # becomes c3 - lambda = -1 - (-3) = +2, signalling the fix is
    # strictly suboptimal by 2 per unit.  Objective degrades by +4.
    handle.fix_cols([x3], 2.0)
    handle.solve()
    assert handle.objective() == pytest.approx(-26.0)
    assert handle.reduced_costs([x3])[0] == pytest.approx(2.0, abs=1e-6)


def test_row_duals(handle: HighsModelHandle) -> None:
    """Capacity row's dual equals the shadow price of one extra unit
    of capacity.  For our LP that is -2 (under minimisation of the
    negated objective, i.e. ``-2`` per extra slack unit on ``cap``)."""
    cap = handle.row_by_name["cap"]
    dual = handle.row_duals([cap])[0]
    # HiGHS convention: row dual for `cap` at optimum = -2 (min sense,
    # upper bound binding).
    assert dual == pytest.approx(-2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------
def test_warm_start_reuses_basis(handle: HighsModelHandle) -> None:
    """A cost perturbation that keeps the same optimal basis should
    re-solve in zero iterations."""
    x3 = handle.col_by_name["x[3]"]
    # Tiny cost bump that does not move the optimum: x[3] stays nonbasic.
    handle.change_costs([x3], -0.5)
    handle.solve()
    assert handle.is_optimal()
    assert handle.iteration_count() == 0


# ---------------------------------------------------------------------------
# Utility accessors
# ---------------------------------------------------------------------------
def test_num_cols_and_rows(handle: HighsModelHandle) -> None:
    assert handle.num_cols() == 3
    assert handle.num_rows() == 1
    handle.add_row("extra", {0: 1.0}, upper=100.0)
    assert handle.num_rows() == 2


# ---------------------------------------------------------------------------
# Defensive consistency check (per Juha 2026-04-24 heads-up on scaling paths)
# ---------------------------------------------------------------------------
def test_check_consistency_reports_ok_for_fresh_map(handle: HighsModelHandle) -> None:
    """Freshly-built name maps match the live Lp → no diagnostic."""
    assert handle.check_consistency() is None


def test_check_consistency_detects_stale_cached_index(
    handle: HighsModelHandle,
) -> None:
    """Corrupting the cache simulates a column reorder; the check should
    surface a diagnostic rather than silently returning wrong indices."""
    # Swap two cached indices — emulates a post-reorder stale map.
    handle.col_by_name["x[1]"], handle.col_by_name["x[2]"] = (
        handle.col_by_name["x[2]"],
        handle.col_by_name["x[1]"],
    )
    diag = handle.check_consistency()
    assert diag is not None
    assert "x[" in diag

    # Rebuild restores consistency.
    handle.build_name_maps()
    assert handle.check_consistency() is None
