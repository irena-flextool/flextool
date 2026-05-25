"""Self-tests for the shared fixtures in ``constraints/conftest.py``.

Each test confirms the fixture builds a ``FlexData`` that ``build_flextool``
accepts and HiGHS solves to optimality.  Placeholder fixtures (those still
to be authored as a flex_toy_*.py file) ``pytest.skip`` with a clear
reason — the corresponding self-test asserts the skip happened.
"""
from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool

from .conftest import solver_options


def _solve(data) -> "Solution":
    pb = Problem()
    build_flextool(pb, data)
    return pb.solve(options=solver_options())


def test_solver_options_keys():
    opts = solver_options()
    assert opts["random_seed"] == 42
    assert opts["parallel"] == "off"


def test_toy_1n1p_1d2t_solves(toy_1n1p_1d2t):
    sol = _solve(toy_1n1p_1d2t)
    assert sol.optimal


def test_toy_storage_2t_solves(toy_storage_2t):
    sol = _solve(toy_storage_2t)
    assert sol.optimal


# Placeholder fixtures: each currently skips with a reason.  The body
# below is reached only if a future agent populates the fixture, in
# which case the test should solve.
def test_toy_storage_blocks_solves(toy_storage_blocks):
    sol = _solve(toy_storage_blocks)
    assert sol.optimal


def test_toy_uc_3t_solves(toy_uc_3t):
    sol = _solve(toy_uc_3t)
    assert sol.optimal


def test_toy_invest_3d_solves(toy_invest_3d):
    sol = _solve(toy_invest_3d)
    assert sol.optimal


def test_toy_2branch_2d_solves(toy_2branch_2d):
    sol = _solve(toy_2branch_2d)
    assert sol.optimal


def test_toy_2node_chp_solves(toy_2node_chp):
    sol = _solve(toy_2node_chp)
    assert sol.optimal


def test_toy_group_reserve_solves(toy_group_reserve):
    sol = _solve(toy_group_reserve)
    assert sol.optimal
