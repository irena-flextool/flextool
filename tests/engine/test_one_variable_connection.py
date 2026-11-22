"""A one-variable connection (``no_losses_no_variable_cost``, DC power-flow lines) has one signed flow (package K).

* ``other_operational_cost`` is ignored (it would be a cost one way and a revenue the other): the LP and the
  objective are those without it, with W-0590 once per solve.
* ``efficiency`` ≠ 1 is a conversion, the same in both directions: the forward flow takes f · slope from the
  source, the reverse flow gives f · slope back there (before, the reverse flow gave f, so with an efficiency
  above 1 sending the same amount both ways created energy at the source), with W-0589 once per solve.

The model is ``lossless_2way`` of ``tests.json``: ``ll`` (200 MW) between lw and le, lw exports 100 MW at t0001
and imports 100 MW at t0002.
"""
from __future__ import annotations

import pytest

from tests._support.coded_warnings import capture_coded
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import b64, build, make_db, run

pytestmark = pytest.mark.solver

LL = ["Base", "lossless_2way"]
ROLLING = [["solve", "lossless_2way_solve", "solve_mode", b64("rolling_window", "str")],
           ["solve", "lossless_2way_solve", "rolling_solve_horizon", b64(1.0, "float")],
           ["solve", "lossless_2way_solve", "rolling_solve_jump", b64(1.0, "float")]]


def _solve(tmp_path_factory, alt: str, efficiency: float, cost: float | None, extra: list | None = None):
    vals = [["connection", "ll", "efficiency", b64(efficiency, "float")]]
    if cost is not None:
        vals.append(["connection", "ll", "other_operational_cost", b64(cost, "float")])
    with capture_coded() as got:
        url, scen = make_db(tmp_path_factory, alt=alt, values=vals + (extra or []), base_chain=LL)
        steps = run(tmp_path_factory, url, scen)
    return steps, got


def _lp(step) -> tuple[dict, dict]:
    """(nodeBalance_eq rows, objective coefficients by column) of the step's LP."""
    pb = build(step.flex_data)
    m = pb.canonicalise()
    return row_terms(pb, "nodeBalance_eq"), {m.col_names[i]: float(m.col_obj[i]) for i in range(m.n_cols)}


def test_variable_cost_is_ignored(tmp_path_factory) -> None:
    """With other_operational_cost 5 the LP (balance rows and objective coefficients) and the objective are
    those without a cost (before: 5 · Δt on the forward flow only, objective 2.19e9 with efficiency 0.5); one
    W-0590 names ll, no W-0589 with efficiency 1."""
    with_cost, got = _solve(tmp_path_factory, "k_cost", 1.0, 5.0)
    without, got0 = _solve(tmp_path_factory, "k_nocost", 1.0, None)
    a, b = list(with_cost.values())[-1], list(without.values())[-1]
    assert _lp(a) == _lp(b)
    assert a.solution.obj == pytest.approx(b.solution.obj, abs=1e-6)
    assert a.flex_data.p_pdt_varCost_process is None
    assert [c.names for c in got.calls if c.code == "W-0590"] == [["ll"]]
    assert not [c for c in got.calls + got0.calls if c.code == "W-0589"]
    assert not [c for c in got0.calls if c.code == "W-0590"]


@pytest.mark.parametrize("efficiency", [0.5, 2.0])
def test_efficiency_is_the_same_conversion_both_ways(tmp_path_factory, efficiency: float) -> None:
    """At lw (the source) the forward flow takes 200 / efficiency per unit and the reverse flow gives the same back;
    at le (the sink) both are ±200.  So the net transfer is one signed flow and sending the same amount both ways
    changes no node (before: the reverse flow gave 200 at lw, with efficiency 2 a free 100 MW there).  One W-0589
    names ll with its efficiency."""
    steps, got = _solve(tmp_path_factory, f"k_eff_{int(efficiency * 10)}", efficiency, None)
    rows, _obj = _lp(list(steps.values())[-1])
    for t in ("t0001", "t0002"):
        lw = rows[f"nodeBalance_eq[lw,p2020,{t}]"]
        le = rows[f"nodeBalance_eq[le,p2020,{t}]"]
        fwd, back = f"v_flow[ll,lw,le,p2020,{t}]", f"v_flow_back[ll,lw,le,p2020,{t}]"
        assert lw[fwd] == pytest.approx(-200.0 / efficiency) and lw[back] == pytest.approx(200.0 / efficiency)
        assert le[fwd] == pytest.approx(200.0) and le[back] == pytest.approx(-200.0)
    calls = [c for c in got.calls if c.code == "W-0589"]
    assert [c.names for c in calls] == [["ll"]]
    assert got.detail("W-0589", "ll") == f"efficiency {efficiency:g}"


def test_warnings_once_per_rolling_solve(tmp_path_factory) -> None:
    """A rolling solve of two rolls prints each warning once (the rolls count as one solve)."""
    steps, got = _solve(tmp_path_factory, "k_roll", 0.5, 5.0, ROLLING)
    assert len(steps) >= 2, list(steps)
    for code in ("W-0589", "W-0590"):
        lines = [line for line in got.lines() if line.endswith(f"[{code}]")]
        assert len(lines) == 1, (code, got.lines())
        assert "'ll'" in lines[0], lines


def test_dc_power_flow_line_cost_is_ignored(tmp_path_factory) -> None:
    """A line of a dc_power_flow_with_angles group is a one-variable connection too: its other_operational_cost
    is dropped with W-0590."""
    ents = [["group", "gdc", None], ["group__node", ["gdc", "west"], None], ["group__node", ["gdc", "north"], None]]
    vals = [["group", "gdc", "transfer_method", b64("dc_power_flow_with_angles", "str")],
            ["connection", "west_north", "other_operational_cost", b64(3.0, "float")]]
    with capture_coded() as got:
        url, scen = make_db(tmp_path_factory, alt="k_dcpf", values=vals, entities=ents,
                            base_chain=["init", "west", "coal", "wind", "network"])
        steps = run(tmp_path_factory, url, scen)
    assert "west_north" in got.names("W-0590")
    for step in steps.values():
        cost = step.flex_data.p_pdt_varCost_process
        assert cost is None or "west_north" not in {str(p) for p in cost.frame.get_column("p").to_list()}
