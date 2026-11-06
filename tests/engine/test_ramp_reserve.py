"""Held reserve uses ramp room (#1).

Upward reserve must be deliverable on top of the flow, so the level ``v_flow[t] + R↑[t]`` may differ from the
previous step's flow by at most the step's upward ramp room; downward reserve likewise with ``v_flow[t] − R↓[t]``
and the downward room.  The raising reserve of an arc is upward reserve at its sink plus "consume more" downward
reserve at its source (converted to the arc's flow: 1/slope on a direct unit), the lowering reserve the opposite.
A step without a predecessor (the first step of the solve) still limits the reserve to one step's room.

Units without online variables (startups add ramp room; see owner question Q-c14a-B).  Fixtures:
``tests.json`` (coal_plant 500 MW, 1 unit; e_boiler west → heat, efficiency 0.5, 100 MW) plus override
alternatives, built from JSON.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_bounds, row_terms, step_of
from tests.engine._unit_db import PLANT, b64, build, flow, reserve, solve, unitsize

pytestmark = pytest.mark.solver

ARC = f"{PLANT},coal_market,west,"
EB = "e_boiler"
TOL = 1e-6


def _ramp(speed: float, dirs, cls: str = "unit__outputNode", rel=None) -> list:
    rel = rel or [PLANT, "west"]
    return ([[cls, rel, "ramp_method", b64("ramp_limit", "str")]]
            + [[cls, rel, f"ramp_speed_{d}", b64(speed, "float")] for d in dirs])


def _reserve(r: str, ud: str, providers: list[tuple[str, str]], reservation: float) -> tuple[list, list]:
    """Reserve ``r`` on group electricity with penalty 300 per MW short: below the balance penalties (800 / 900
    per MWh at west, 1000 at heat), so the model does not over- or under-supply a node to make ramp room (a step
    ramping down by Δ could hold Δ more reserve)."""
    grp = [r, ud, "electricity"]
    ents = [["reserve", r, None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64("timeseries_only", "str")],
            ["reserve__upDown__group", grp, "reservation", b64(reservation, "float")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(300.0, "float")]]
    for p, n in providers:
        key = [r, ud, p, n]
        ents.append(["reserve__upDown__unit__node", key, None])
        vals += [["reserve__upDown__unit__node", key, "is_enabled", b64("yes", "str")],
                 ["reserve__upDown__unit__node", key, "max_share", b64(1.0, "float")],
                 ["reserve__upDown__unit__node", key, "reliability", b64(1.0, "float")]]
    return ents, vals


FLAT = [["node", "west", "inflow", b64(-300.0, "float")]]


@pytest.fixture(scope="module")
def up_case(tmp_path_factory):
    """coal_plant runs flat at 300 MW (200 MW headroom), ramp up 0.002/min = 60 MW per hour, 100 MW of upward
    reserve asked only from it."""
    ents, vals = _reserve("ru", "up", [(PLANT, "west")], 100.0)
    return solve(tmp_path_factory, alt="rr_up", base_chain=["init", "west", "coal"], entities=ents,
                 values=FLAT + vals + _ramp(0.002, ("up",)))


def test_up_reserve_uses_up_ramp_room(up_case) -> None:
    """The reserve is held to one step's ramp room, 60 MW, at every step (including the first, which has no
    predecessor), and the up rows carry it.  Before: 100 MW held (capacity headroom only)."""
    step = up_case
    res = reserve(step, PLANT, "ru", "west")
    assert res.max() == pytest.approx(60.0, abs=1e-6)
    assert (res <= 60.0 + TOL).all()
    rows = row_terms(build(step.flex_data), "ramp_sink_up_constraint", ARC)
    assert len(rows) == step.flex_data.dt.height
    for row, terms in rows.items():
        assert coef(terms, f"v_reserve[{PLANT},ru,up,west,") == pytest.approx(1.0), row


def test_reserve_only_row_at_first_step(up_case) -> None:
    """The first step has no predecessor: its row has only the reserve term, with room 0.12 (× 1 unit)."""
    pb = build(up_case.flex_data)
    row = f"ramp_sink_up_constraint[{ARC}p2020,t0001]"
    terms = row_terms(pb, "ramp_sink_up_constraint", ARC)[row]
    assert set(terms) == {f"v_reserve[{PLANT},ru,up,west,p2020,t0001]"}
    assert row_bounds(pb, "ramp_sink_up_constraint", ARC)[row][1] == pytest.approx(0.12)


def test_down_reserve_uses_down_ramp_room(tmp_path_factory) -> None:
    """Same with a downward ramp limit and 100 MW of downward reserve: held to 60 MW; the down rows subtract
    it (GMPL added it there, loosening the row).  Before: 100 MW held."""
    ents, vals = _reserve("rd", "down", [(PLANT, "west")], 100.0)
    step = solve(tmp_path_factory, alt="rr_dn", base_chain=["init", "west", "coal"], entities=ents,
                 values=FLAT + vals + _ramp(0.002, ("down",)))
    res = reserve(step, PLANT, "rd", "west")
    assert res.max() == pytest.approx(60.0, abs=1e-6)
    assert (res <= 60.0 + TOL).all()
    for row, terms in row_terms(build(step.flex_data), "ramp_sink_down_constraint", ARC).items():
        assert coef(terms, f"v_reserve[{PLANT},rd,down,west,") == pytest.approx(-1.0), row


def test_up_reserve_after_a_ramp_down(up_case) -> None:
    """The reserve shares the room with the flow change: at every step v_flow[t] − v_flow[t−1] + R↑[t] ≤ 0.12
    units (a unit ramping down by Δ in a step could hold Δ more; with a costly surplus the flow stays flat)."""
    step = up_case
    us = unitsize(step, PLANT)
    f = flow(step, PLANT, "coal_market", "west").to_list()
    res = reserve(step, PLANT, "ru", "west").to_list()
    for i in range(1, len(f)):
        assert (f[i] - f[i - 1] + res[i]) / us <= 0.12 + TOL


def test_consume_more_uses_input_ramp_room(tmp_path_factory) -> None:
    """e_boiler (direct west → heat, efficiency 0.5) with a ramp on its input arc and downward ("consume
    more") reserve at west: the source-up rows carry the reserve with factor 1/slope = 0.5, so it is held to
    0.12 / 0.5 = 0.24 units of input (24 MW).  Before: 40 MW held (the input headroom)."""
    eb_ents = [["unit", EB, None], ["unit__inputNode", [EB, "west"], None], ["unit__outputNode", [EB, "heat"], None]]
    eb_vals = [["unit", EB, "efficiency", b64(0.5, "float")],
               ["unit", EB, "existing", b64(100.0, "float")],
               ["unit", EB, "conversion_method", b64("constant_efficiency", "str")],
               ["node", "heat", "penalty_up", b64(1000.0, "float")],
               ["node", "heat", "penalty_down", b64(1000.0, "float")],
               ["node", "heat", "inflow", b64(-80.0, "float")]]
    ents, vals = _reserve("rd", "down", [(EB, "west")], 40.0)
    step = solve(tmp_path_factory, alt="rr_eb", base_chain=["init", "west", "coal", "heat"],
                 entities=eb_ents + ents, values=FLAT + eb_vals + vals
                 + _ramp(0.002, ("up",), cls="unit__inputNode", rel=[EB, "west"]))
    us = unitsize(step, EB)
    rows = row_terms(build(step.flex_data), "ramp_source_up_constraint", f"{EB},west,heat,")
    assert rows
    for row, terms in rows.items():
        assert coef(terms, f"v_reserve[{EB},rd,down,west,") == pytest.approx(0.5), row
    res = reserve(step, EB, "rd", "west")
    assert res.max() == pytest.approx(0.24 * us, abs=1e-6)
    assert ("p2020", "t0001") in {step_of(r) for r in rows}


def test_reserve_terms_only_where_reserve_is_held(up_case) -> None:
    """The reserve-only rows exist only for the reserve's direction: no down rows without a down ramp limit,
    and the up rows' reserve column is ru's (the provider's v_reserve domain)."""
    pb = build(up_case.flex_data)
    assert not row_terms(pb, "ramp_sink_down_constraint")
    cols = {c for terms in row_terms(pb, "ramp_sink_up_constraint", ARC).values() for c in terms}
    assert {c.split("[")[0] for c in cols} == {"v_flow", "v_reserve"}
    assert all(",ru,up,west," in c for c in cols if c.startswith("v_reserve"))
    assert pl.Series([c for c in cols if c.startswith("v_reserve")]).len() == up_case.flex_data.dt.height
