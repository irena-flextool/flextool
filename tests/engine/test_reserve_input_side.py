"""Reserve at a unit's input node: headroom for "consume more", ``max_share`` of the input capacity (#42, #23(a)).

``max_share`` is set on the unit–node relationship, so it is a share of THAT arc's
capacity (input arc → input capacity), and a unit consuming at full capacity cannot offer more "consume more"
(downward) reserve at its input.  Downward reserve at a unit's source node now sits on the LHS of the arc's
capacity rows (``maxFlow``, ``maxFlow_online_*``, the new ``maxInputFlow_online_*`` of an indirect unit's input
arc, ``profile_flow_upper_limit``), converted to the arc's flow (factor 1 where v_flow is the input, 1/slope on a
direct input→output arc).  ``reserve_process_*`` at an input node caps the reserve at ``max_share`` × the input
capacity (``_reserve._input_capacity_factor``).  Output-side ``max_share`` stays ``max_share`` × built capacity
(about how far the asset can be trusted).  Connections are not coupled yet (planned with the rewrite of the
connection capacity rows).

Fixtures: ``tests.json`` (``coal`` 500 MW at west, ``coal_chp`` indirect coal_market → west + heat, ``heat``
node) plus ``e_boiler``: west → heat, efficiency 0.5, 100 MW; heat −300 MW with ``penalty_up`` 1000, west
−250 MW; reserve ``rd`` (down, 50 MW, penalty 20 000) at west offered only by e_boiler.  Built from JSON.
"""
from __future__ import annotations

import collections

import polars as pl
import pytest

from tests.engine._unit_db import (
    CHP, COAL, PLANT, b64, build, flow, online, reserve, solve, unitsize, utf8,
)

pytestmark = pytest.mark.solver

REL = "reserve__upDown__unit__node"
EB = "e_boiler"
DR = "dr_increase_demand"
BASE = ["init", "west", "coal", "heat"]
EB_ENTS = [["unit", EB, None], ["unit__inputNode", [EB, "west"], None], ["unit__outputNode", [EB, "heat"], None]]
EB_VALS = [["unit", EB, "efficiency", b64(0.5, "float")],
           ["unit", EB, "existing", b64(100.0, "float")],
           ["unit", EB, "conversion_method", b64("constant_efficiency", "str")],
           ["node", "heat", "penalty_up", b64(1000.0, "float")],
           ["node", "west", "inflow", b64(-250.0, "float")]]
TOL = 1e-6


def _reserve(r: str, ud: str, group: str, providers: list[tuple[str, str, float]], reservation: float,
             penalty: float = 20000.0, cls: str = "unit") -> tuple[list, list]:
    grp = [r, ud, group]
    rel_cls = f"reserve__upDown__{cls}__node"
    ents = [["reserve", r, None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64("timeseries_only", "str")],
            ["reserve__upDown__group", grp, "reservation", b64(reservation, "float")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(penalty, "float")]]
    for p, n, share in providers:
        key = [r, ud, p, n]
        ents.append([rel_cls, key, None])
        vals += [[rel_cls, key, "is_enabled", b64("yes", "str")],
                 [rel_cls, key, "max_share", b64(share, "float")],
                 [rel_cls, key, "reliability", b64(1.0, "float")]]
    return ents, vals


def _rd(share: float) -> tuple[list, list]:
    return _reserve("rd", "down", "electricity", [(EB, "west", share)], 50.0)


def _row_terms(pb, family: str, prefix: str = "") -> dict[str, dict[str, float]]:
    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(f"{family}[{prefix}")}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            if m.row_idx[k] in rows:
                out[rows[m.row_idx[k]]][m.col_names[c]] = m.val[k]
    return out


def _row_rhs(pb, family: str, prefix: str = "") -> dict[str, float]:
    m = pb.canonicalise()
    return {nm: float(m.row_ub[i]) for i, nm in enumerate(m.row_names) if nm.startswith(f"{family}[{prefix}")}


def _coef(terms: dict[str, float], prefix: str) -> float:
    hit = [v for c, v in terms.items() if c.startswith(prefix)]
    assert len(hit) == 1, (prefix, terms)
    return hit[0]


def _shortfall(step, r: str, group: str = "electricity") -> pl.Series:
    from flextool.engine.input._param_shapes import promote_param_to_dt
    from flextool.engine.model._reserve import reserve_shortfall_scale

    fd = step.flex_data
    frame = promote_param_to_dt(reserve_shortfall_scale(fd), fd.dt).collect()
    sc = utf8(frame).filter(pl.col("r") == r).sort("d", "t")["value"]
    vq = utf8(step.solution.value("vq_reserve")).filter((pl.col("r") == r) & (pl.col("g") == group)) \
        .sort("d", "t")["value"]
    return vq * sc


# ---------------------------------------------------------------------------
# direct unit at full consumption


def test_full_consumer_offers_no_consume_more_reserve(tmp_path_factory) -> None:
    """e_boiler would run at full output (heat demand 300 > 100 MW); holding the 50 MW "consume more"
    reserve needs input headroom: input + reserve ≤ 200 MW, so it runs at 75 MW out and the reserve is met.
    The maxFlow row of its arc carries the reserve with factor 1/slope = 0.5.  Before: full output and
    100 MW of reserve."""
    ents, vals = _rd(1.0)
    step = solve(tmp_path_factory, alt="in_full", base_chain=BASE, entities=EB_ENTS + ents, values=EB_VALS + vals)
    us = unitsize(step, EB)
    out = flow(step, EB, "west", "heat")
    res = reserve(step, EB, "rd", "west")
    assert ((out * 2.0 + res) <= 2.0 * us + TOL).all()
    assert res.to_list() == pytest.approx([50.0] * len(res), abs=1e-6)
    assert out.to_list() == pytest.approx([75.0] * len(out), abs=1e-6)
    assert (_shortfall(step, "rd").abs() < 1e-6).all()
    for row, terms in _row_terms(build(step.flex_data), "maxFlow", f"{EB},west,heat,").items():
        assert _coef(terms, f"v_reserve[{EB},rd,down,west,") == pytest.approx(0.5), row


# ---------------------------------------------------------------------------
# online units: direct (maxFlow_online) and indirect (maxInputFlow_online), max_share of the input capacity


@pytest.fixture(scope="module")
def input_side(tmp_path_factory):
    """e_boiler linear UC (min_load 0.2) holds rd at west with max_share 0.2; coal_chp (indirect, linear
    UC, min_load 0.4) holds rf (down, 300 MW) at its fuel node coal_market (group fuel) with max_share 0.5."""
    e1, v1 = _rd(0.2)
    e2, v2 = _reserve("rf", "down", "fuel", [(CHP, COAL, 0.5)], 300.0)
    return solve(tmp_path_factory, alt="in_side", base_chain=BASE[:3] + ["coal_chp", "heat"],
                 entities=EB_ENTS + e1 + e2,
                 values=EB_VALS + v1 + v2 + [
                     ["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
                     ["unit", EB, "startup_method", b64("linear", "str")],
                     ["unit", EB, "min_load", b64(0.2, "float")],
                     ["unit", CHP, "startup_method", b64("linear", "str")],
                     ["unit", CHP, "min_load", b64(0.4, "float")]])


def test_online_consumer_headroom(input_side) -> None:
    """maxFlow_online_linear of e_boiler's arc carries the reserve (factor 0.5), so the reserve needs online
    capacity: reserve ≤ (online − output) · slope at every step (an offline unit holds none).  Before an
    offline e_boiler held 20 MW."""
    step = input_side
    rows = _row_terms(build(step.flex_data), "maxFlow_online_linear", f"{EB},west,heat,")
    assert rows
    for row, terms in rows.items():
        assert _coef(terms, f"v_reserve[{EB},rd,down,west,") == pytest.approx(0.5), row
    res = reserve(step, EB, "rd", "west")
    room = (online(step, EB) - flow(step, EB, "west", "heat")) * 2.0
    assert ((res - room) <= TOL).all()


def test_input_side_max_share_is_share_of_input_capacity(input_side) -> None:
    """max_share 0.2 at e_boiler's input: the reserve is at most 0.2 × 200 MW (input capacity) = 40 MW, and
    it binds (40 MW held, 10 MW short).  coal_chp: rf ≤ 0.5 × input capacity (input_cap × existing).
    Before: 0.2 × 100 MW (output) = 20 MW."""
    step = input_side
    us = unitsize(step, EB)
    res = reserve(step, EB, "rd", "west")
    assert (res <= 0.2 * 2.0 * us + TOL).all()
    assert res.max() == pytest.approx(40.0, abs=1e-6)
    pb = build(step.flex_data)
    rhs = _row_rhs(pb, "reserve_process_downward", f"{EB},rd,down,west,")
    assert rhs and set(round(v, 12) for v in rhs.values()) == {round(0.2 * 2.0 * 100.0 / us, 12)}
    cap = utf8(step.flex_data.p_indirect_input_cap.frame).filter(
        (pl.col("p") == CHP) & (pl.col("source") == COAL))["value"]
    rhs_chp = _row_rhs(pb, "reserve_process_downward", f"{CHP},rf,down,{COAL},")
    existing = float(utf8(step.flex_data.p_process_existing_count.frame).filter(pl.col("p") == CHP)["value"][0])
    assert set(round(v, 9) for v in rhs_chp.values()) == {round(0.5 * float(cap[0]) * existing, 9)}


def test_indirect_input_arc_headroom(input_side) -> None:
    """coal_chp (no invest: its fuel arc is bounded on v_flow) gets a maxFlow row on the fuel arc with the
    reserve, and a maxInputFlow_online_linear row: fuel + reserve ≤ input_cap × online capacity (MW).
    Before: no row; an offline-able unit could offer the reserve."""
    step = input_side
    pb = build(step.flex_data)
    mf = _row_terms(pb, "maxFlow", f"{CHP},{COAL},{CHP},")
    on = _row_terms(pb, "maxInputFlow_online_linear", f"{CHP},{COAL},{CHP},")
    assert len(mf) == len(on) == step.flex_data.dt.height
    for rows in (mf, on):
        for row, terms in rows.items():
            assert _coef(terms, f"v_reserve[{CHP},rf,down,{COAL},") == pytest.approx(1.0), row
            assert _coef(terms, f"v_flow[{CHP},{COAL},{CHP},") == pytest.approx(1.0), row
    cap = float(utf8(step.flex_data.p_indirect_input_cap.frame).filter(
        (pl.col("p") == CHP) & (pl.col("source") == COAL))["value"][0])
    fuel = flow(step, CHP, COAL, CHP)
    res = reserve(step, CHP, "rf", COAL)
    assert ((fuel + res - cap * online(step, CHP)) <= 1e-5).all()
    assert res.max() > 1.0


# ---------------------------------------------------------------------------
# sink-less consumer, profile rows, guards


def test_sink_less_consumer_headroom(tmp_path_factory) -> None:
    """dr_increase_demand (50 MW, consumes at west, no output) offering down reserve at west: its maxFlow row
    carries the reserve with factor 1 and consumption + reserve ≤ 50 MW."""
    ents, vals = _reserve("rd", "down", "electricity", [(DR, "west", 1.0)], 50.0)
    step = solve(tmp_path_factory, alt="in_dr", base_chain=["init", "west", "coal", "dr_increase_demand"],
                 entities=ents, values=vals)
    rows = _row_terms(build(step.flex_data), "maxFlow", f"{DR},west,{DR},")
    assert rows
    for row, terms in rows.items():
        assert _coef(terms, f"v_reserve[{DR},rd,down,west,") == pytest.approx(1.0), row
    total = flow(step, DR, "west", DR) + reserve(step, DR, "rd", "west")
    assert (total <= 50.0 + TOL).all()


def test_profile_rows_carry_source_reserve(tmp_path_factory) -> None:
    """A profile upper limit on e_boiler's arc: profile_flow_upper_limit carries the down reserve at its input
    with factor 1/slope = 0.5."""
    ents, vals = _rd(1.0)
    ents.append(["unit__node__profile", [EB, "heat", "wind_profile"], None])
    vals.append(["unit__node__profile", [EB, "heat", "wind_profile"], "profile_method", b64("upper_limit", "str")])
    step = solve(tmp_path_factory, alt="in_prof", base_chain=BASE + ["wind"], entities=EB_ENTS + ents,
                 values=EB_VALS + vals)
    rows = _row_terms(build(step.flex_data), "profile_flow_upper_limit", f"{EB},west,heat,")
    assert rows
    for row, terms in rows.items():
        assert _coef(terms, f"v_reserve[{EB},rd,down,west,") == pytest.approx(0.5), row


def test_output_side_max_share_unchanged(tmp_path_factory) -> None:
    """Guard: coal_plant (linear UC) offering up reserve at its output west with max_share 0.2 keeps
    reserve_process_upward ≤ 0.2 · existing_count (built capacity, not online capacity)."""
    ents, vals = _reserve("ru", "up", "electricity", [(PLANT, "west", 0.2)], 50.0)
    step = solve(tmp_path_factory, alt="in_out", base_chain=["init", "west", "coal"], entities=ents,
                 values=vals + [["unit", PLANT, "startup_method", b64("linear", "str")],
                                ["unit", PLANT, "min_load", b64(0.5, "float")]])
    pb = build(step.flex_data)
    existing = float(utf8(step.flex_data.p_process_existing_count.frame).filter(pl.col("p") == PLANT)["value"][0])
    rhs = _row_rhs(pb, "reserve_process_upward", f"{PLANT},ru,up,west,")
    assert rhs and all(v == pytest.approx(0.2 * existing) for v in rhs.values())
    for row, terms in _row_terms(pb, "reserve_process_upward", f"{PLANT},ru,up,west,").items():
        assert terms == {next(iter(terms)): 1.0} and next(iter(terms)).startswith(f"v_reserve[{PLANT},"), row


def test_connection_source_reserve_unchanged(tmp_path_factory) -> None:
    """Connections are not coupled yet: west_east offering down reserve at its source west gets no reserve term in its
    maxFlow rows (connections are coupled in the connection work, not here)."""
    ents, vals = _reserve("rd", "down", "electricity", [("west_east", "west", 1.0)], 50.0, cls="connection")
    step = solve(tmp_path_factory, alt="in_conn", base_chain=["init", "west", "coal", "network"], entities=ents,
                 values=vals)
    rows = _row_terms(build(step.flex_data), "maxFlow", "west_east,west,east,")
    assert rows
    for row, terms in rows.items():
        assert not [c for c in terms if c.startswith("v_reserve[")], row


def test_profile_lower_limit_subtracts_source_up_reserve(tmp_path_factory) -> None:
    """A profile lower limit on e_boiler's arc with an upward ("consume less") reserve at its input west:
    profile_flow_lower_limit carries the reserve with factor −1/slope = −0.5 (the reserve lowers the flow)."""
    ents, vals = _reserve("ru", "up", "electricity", [(EB, "west", 1.0)], 20.0)
    ents.append(["unit__node__profile", [EB, "heat", "wind_profile"], None])
    vals.append(["unit__node__profile", [EB, "heat", "wind_profile"], "profile_method", b64("lower_limit", "str")])
    step = solve(tmp_path_factory, alt="in_prof_lo", base_chain=BASE + ["wind"], entities=EB_ENTS + ents,
                 values=EB_VALS + vals)
    rows = _row_terms(build(step.flex_data), "profile_flow_lower_limit", f"{EB},west,heat,")
    assert len(rows) == step.flex_data.dt.height
    for row, terms in rows.items():
        assert _coef(terms, f"v_reserve[{EB},ru,up,west,") == pytest.approx(-0.5), row


def test_indirect_input_arc_headroom_integer(tmp_path_factory) -> None:
    """As test_indirect_input_arc_headroom with binary online variables on coal_chp: the
    maxInputFlow_online_integer rows carry the fuel flow, the reserve (1) and −input_cap on v_online_integer,
    and fuel + reserve ≤ input_cap × online capacity at every step."""
    e2, v2 = _reserve("rf", "down", "fuel", [(CHP, COAL, 0.5)], 300.0)
    step = solve(tmp_path_factory, alt="in_side_int", base_chain=BASE[:3] + ["coal_chp", "heat"], entities=e2,
                 values=v2 + [["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
                              ["node", "west", "inflow", b64(-250.0, "float")],
                              ["unit", CHP, "startup_method", b64("binary", "str")],
                              ["unit", CHP, "min_load", b64(0.4, "float")]])
    rows = _row_terms(build(step.flex_data), "maxInputFlow_online_integer", f"{CHP},{COAL},{CHP},")
    assert len(rows) == step.flex_data.dt.height
    cap = float(utf8(step.flex_data.p_indirect_input_cap.frame).filter(
        (pl.col("p") == CHP) & (pl.col("source") == COAL))["value"][0])
    for row, terms in rows.items():
        assert _coef(terms, f"v_reserve[{CHP},rf,down,{COAL},") == pytest.approx(1.0), row
        assert _coef(terms, f"v_flow[{CHP},{COAL},{CHP},") == pytest.approx(1.0), row
        assert _coef(terms, f"v_online_integer[{CHP},") == pytest.approx(-cap), row
    fuel = flow(step, CHP, COAL, CHP)
    res = reserve(step, CHP, "rf", COAL)
    assert ((fuel + res - cap * online(step, CHP, "integer")) <= 1e-5).all()


def test_source_side_rows_in_rolls(tmp_path_factory) -> None:
    """Rolling solve (12 h horizon, 6 h jump): in every roll e_boiler's arc rows carry the input-side down
    reserve with 1/efficiency⁻¹ = 0.5 and reserve_process_downward caps it at max_share 0.2 × the input
    capacity (100 MW / 0.5) — the numbers of the JSON input."""
    from tests.engine._unit_db import make_db, run

    ents, vals = _rd(0.2)
    url, scen = make_db(tmp_path_factory, alt="in_roll", base_chain=BASE, entities=EB_ENTS + ents,
                        values=EB_VALS + vals + [
                            ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
                            ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(12.0, "float")],
                            ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(6.0, "float")]])
    steps = run(tmp_path_factory, url, scen)
    assert len(steps) > 2
    for name, step in steps.items():
        fd = step.flex_data
        us = unitsize(step, EB)
        pb = build(fd)
        rows = _row_terms(pb, "maxFlow", f"{EB},west,heat,")
        assert len(rows) == fd.dt.height, name
        for row, terms in rows.items():
            assert _coef(terms, f"v_reserve[{EB},rd,down,west,") == pytest.approx(0.5), (name, row)
        rhs = _row_rhs(pb, "reserve_process_downward", f"{EB},rd,down,west,")
        assert len(rhs) == fd.dt.height, name
        assert all(v == pytest.approx(0.2 * (1 / 0.5) * 100.0 / us) for v in rhs.values()), name
