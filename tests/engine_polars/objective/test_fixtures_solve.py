"""Self-tests for the shared fixtures in ``objective/conftest.py``."""
from __future__ import annotations

import pytest

from .conftest import solve_problem


def test_toy_costs_only_solves(toy_costs_only_1d2t):
    pb, sol = solve_problem(toy_costs_only_1d2t)
    assert sol.optimal
    # Zero inflow + zero state ⇒ obj should be 0 (no slack, no buy).
    assert abs(sol.obj) < 1e-9


def test_reuse_constraints_fixture_works(toy_1n1p_1d2t):
    """Verifies pytest_plugins re-export of constraints fixtures."""
    pb, sol = solve_problem(toy_1n1p_1d2t)
    assert sol.optimal
