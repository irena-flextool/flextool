"""User constraints count a one-variable connection's reverse flow (WP-H2, the user-constraint side of item 138).

docs/reference/user_constraints.md: ``constraint_flow_coeff`` on ``connection__node`` puts the connection's flow in the
row in both directions.  A two-variable connection (``variable_cost_only`` / ``regular`` / ``exact``) has one arc per
direction, and the coefficient of (c, n) lands on both arcs (each touches n, as source or sink), so the row reads
coef·(forward + reverse).  A one-variable connection (``no_losses_no_variable_cost``, every DC power-flow line)
carries its reverse flow in ``v_flow_back``, which the row never saw: the reverse direction was free of any user
constraint.  The fix gives v_flow_back the arc's coefficient, the same as v_flow: coef·(v_flow + v_flow_back).

Fixture: ``tests.json`` ``lossless_2way`` (from JSON): line ``ll`` lw → le, 200 MW, two 1-h steps, no units and no
costs; inflow lw +100 / le −100 at t0001 and lw −100 / le +100 at t0002, so without a constraint ll carries 100 MW
lw → le at t0001 and 100 MW le → lw at t0002 and nothing is unserved (objective 0; test_flowgroup_reverse_flow).
Energy the line does not carry is unserved at the receiving node and spilled at the other.

Hand values, constraint ``uc``: 1 × flow of (ll, le) ≤ 30 (``less_than``, ``constant`` 30):

* per time step: the line carries 30 MW each step, +30 into le at t0001 and −30 at t0002, 140 MWh held back.
  Before the fix the one-variable line carried the full 100 MW back at t0002 (−100 into le, 70 MWh held back).
* the coefficient on (ll, lw) instead of (ll, le) gives the same: both arcs touch both nodes.
* the line authored le → lw (``ll_rev``): the same.
* as a DC power-flow line (susceptance 1000 MW/rad): the same; θ_le − θ_lw = −0.03 / +0.03 rad (power flows
  from the higher angle).
* ``period_total`` (one period of 2 h, period_share 2/8760, w = 4380 per step): Σ_t 4380·(fwd + back) ≤ 4380·60 →
  60 MWh over the two steps, 140 MWh held back (the per-step objective; the split between the steps is degenerate).
"""
from __future__ import annotations

import pytest

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import b64, build
from tests.engine.test_flowgroup_reverse_flow import (
    TIMES,
    TOL,
    _dc_values,
    _method,
    _net_into,
    _reversed_line,
    _run,
)

pytestmark = pytest.mark.solver

REL = 1e-9
CAP = 30.0
W = 8760.0 / 2.0
LE = "process_constraint_less_than"


def _uc(conn: str, node: str, constant: float = CAP, method: str | None = None) -> tuple[list, list]:
    ents = [["constraint", "uc", None], ["connection__node", [conn, node], None]]
    vals = [["constraint", "uc", "sense", b64("less_than", "str")],
            ["constraint", "uc", "constant", b64(constant, "float")],
            ["connection__node", [conn, node], "constraint_flow_coeff",
             b64({"index_type": "str", "index_name": "constraint", "data": [["uc", 1.0]]}, "map")]]
    if method is not None:
        vals.append(["constraint", "uc", "constraint_period_method", b64(method, "str")])
    return ents, vals


@pytest.fixture(scope="module")
def cases(tmp_path_factory) -> dict:
    tpf = tmp_path_factory
    out = {}
    ents, vals = _uc("ll", "le")
    out["le_1var"] = _run(tpf, "ucr_le_1var", vals, ents)
    out["le_2var"] = _run(tpf, "ucr_le_2var", vals + _method("variable_cost_only"), ents)
    ents, vals = _uc("ll", "lw")
    out["lw_1var"] = _run(tpf, "ucr_lw_1var", vals, ents)
    out["lw_2var"] = _run(tpf, "ucr_lw_2var", vals + _method("variable_cost_only"), ents)
    ents, vals = _uc("ll_rev", "le")
    rents, rvals = _reversed_line()
    out["rev"] = _run(tpf, "ucr_rev", vals + rvals + _method("no_losses_no_variable_cost", "ll_rev"),
                      ents + rents, inactive=("ll",))
    ents, vals = _uc("ll", "le")
    dents, dvals = _dc_values()
    out["dc"] = _run(tpf, "ucr_dc", vals + dvals, ents + dents)
    ents, vals = _uc("ll", "le", constant=2.0 * CAP * W, method="period_total")
    out["pt_1var"] = _run(tpf, "ucr_pt_1var", vals, ents)
    out["pt_2var"] = _run(tpf, "ucr_pt_2var", vals + _method("variable_cost_only"), ents)
    return out


@pytest.mark.parametrize("node", ["le", "lw"])
def test_one_variable_line_equals_two_variable_line(cases, node) -> None:
    """The same line as ``no_losses_no_variable_cost`` (reverse in v_flow_back) and as a zero-cost lossless
    ``variable_cost_only`` line (two arcs) gives the same objective and net flows, with the coefficient on either
    end.  The constraint holds energy back (objective > 0).  Before: the one-variable line's reverse flow was free."""
    one, two = cases[f"{node}_1var"], cases[f"{node}_2var"]
    assert two.solution.obj > 1.0
    assert one.solution.obj == pytest.approx(two.solution.obj, rel=REL)
    for n in ("lw", "le"):
        n1, n2 = _net_into(one, n), _net_into(two, n)
        for t in TIMES:
            assert n1[t] == pytest.approx(n2[t], abs=TOL), (node, n, t)


@pytest.mark.parametrize("case", ["le_1var", "lw_1var", "rev", "dc"])
def test_hand_values(cases, case) -> None:
    """flow ≤ 30 in both directions: +30 MW into le at t0001, −30 at t0002 (was −100 before the fix); the same
    objective whichever end carries the coefficient, whichever way the line is authored, and as a DC line."""
    net = _net_into(cases[case], "le")
    assert net["t0001"] == pytest.approx(CAP, abs=TOL), case
    assert net["t0002"] == pytest.approx(-CAP, abs=TOL), case
    assert cases[case].solution.obj == pytest.approx(cases["le_2var"].solution.obj, rel=1e-7)


def test_dc_angles(cases) -> None:
    """The DC line's angle difference follows the limited flow, 30 MW / 1000 MW/rad = 0.03 rad, power flowing from
    the higher angle: θ_le − θ_lw = −0.03 at t0001 (lw → le), +0.03 at t0002 (le → lw)."""
    from tests.engine._unit_db import utf8

    angle = utf8(cases["dc"].solution.value("v_angle"))
    theta = {(n, t): x for n, _d, t, x in angle.select("n", "d", "t", "value").rows()}
    assert theta[("le", "t0001")] - theta[("lw", "t0001")] == pytest.approx(-0.03, abs=1e-6)
    assert theta[("le", "t0002")] - theta[("lw", "t0002")] == pytest.approx(0.03, abs=1e-6)


def test_row_reads_back_flow_with_the_forward_coefficient(cases) -> None:
    """Each per-step row carries v_flow and v_flow_back of ll with the same coefficient (1 × unitsize)."""
    rows = row_terms(build(cases["le_1var"].flex_data), LE)
    assert len(rows) == len(TIMES)
    for row, terms in rows.items():
        fwd = [c for col, c in terms.items() if col.startswith("v_flow[") and "ll" in col]
        back = [c for col, c in terms.items() if col.startswith("v_flow_back[") and "ll" in col]
        assert len(fwd) == 1 and len(back) == 1, (row, terms)
        assert back[0] == pytest.approx(fwd[0], rel=1e-12), (row, terms)


def test_period_total(cases) -> None:
    """``period_total``: the throughput of both directions over the period is ≤ 60 MWh for the one-variable line,
    as for the two-variable line; 140 MWh held back, the per-step objective."""
    one, two = cases["pt_1var"], cases["pt_2var"]
    assert one.solution.obj == pytest.approx(two.solution.obj, rel=REL)
    assert one.solution.obj == pytest.approx(cases["le_2var"].solution.obj, rel=1e-7)
    net = _net_into(one, "le")
    assert net["t0001"] - net["t0002"] == pytest.approx(2.0 * CAP, abs=TOL)
