"""Write the Gurobi JSON result files the parser tests read (run by hand; the files are committed).

    ~/venv-spi/bin/python tests/engine/fixtures/gurobi_json/make_fixtures.py

Needs gurobipy (the pip wheel's size-limited licence is enough).  ``Model.write("<x>.json")`` writes the same
JSON solution format as ``gurobi_cl ResultFile=<x>.json``.  Columns are named ``C0000001`` ... like the generic
names of FlexTool's MPS file.
"""
from __future__ import annotations

from pathlib import Path

import gurobipy as gp

HERE = Path(__file__).resolve().parent

#: The time-limited knapsack: item i has value ``value(i)`` and weight ``weight(i, k)`` in constraint k.
N_ITEMS = 1500
N_ROWS = 40


def value(i: int) -> int:
    return 100 + (i * 7919) % 901


def weight(i: int, k: int) -> int:
    return 50 + ((i + 1) * (k + 3) * 104729) % 951


def capacity(k: int) -> int:
    return sum(weight(i, k) for i in range(N_ITEMS)) // 2


def _model(env: gp.Env) -> gp.Model:
    m = gp.Model(env=env)
    m.setParam("OutputFlag", 0)
    return m


def optimal_mip(env: gp.Env) -> None:
    """min -5 x - 4 y, 6 x + 4 y <= 24, x + 2 y <= 6, x, y >= 0 integer: optimum x = 4, y = 0, objective -20."""
    m = _model(env)
    x = m.addVar(vtype=gp.GRB.INTEGER, name="C0000001")
    y = m.addVar(vtype=gp.GRB.INTEGER, name="C0000002")
    m.setObjective(-5 * x - 4 * y)
    m.addConstr(6 * x + 4 * y <= 24, name="R0000001")
    m.addConstr(x + 2 * y <= 6, name="R0000002")
    m.optimize()
    assert m.Status == gp.GRB.OPTIMAL
    m.write(str(HERE / "optimal_mip.json"))


def time_limited_mip(env: gp.Env) -> None:
    """max sum value(i) x_i (as min of its negative), sum_i weight(i, k) x_i <= capacity(k), x binary, stopped
    at a 0.5 s time limit with an incumbent."""
    m = _model(env)
    xs = [m.addVar(vtype=gp.GRB.BINARY, name=f"C{i + 1:07d}") for i in range(N_ITEMS)]
    m.setObjective(gp.quicksum(-value(i) * xs[i] for i in range(N_ITEMS)))
    for k in range(N_ROWS):
        m.addConstr(gp.quicksum(weight(i, k) * xs[i] for i in range(N_ITEMS)) <= capacity(k), name=f"R{k + 1:07d}")
    m.setParam("TimeLimit", 0.5)
    m.setParam("Threads", 1)
    m.optimize()
    assert m.Status == gp.GRB.TIME_LIMIT and m.SolCount > 0, (m.Status, m.SolCount)
    m.write(str(HERE / "time_limited_mip.json"))


def infeasible_lp(env: gp.Env) -> None:
    """x + y >= 3 and x + y <= 1: infeasible."""
    m = _model(env)
    x = m.addVar(name="C0000001")
    y = m.addVar(name="C0000002")
    m.setObjective(x + y)
    m.addConstr(x + y >= 3, name="R0000001")
    m.addConstr(x + y <= 1, name="R0000002")
    m.setParam("DualReductions", 0)  # a definite INFEASIBLE, not INF_OR_UNBD
    m.optimize()
    assert m.Status == gp.GRB.INFEASIBLE
    m.write(str(HERE / "infeasible_lp.json"))


if __name__ == "__main__":
    with gp.Env(params={"OutputFlag": 0}) as env:
        optimal_mip(env)
        time_limited_mip(env)
        infeasible_lp(env)
