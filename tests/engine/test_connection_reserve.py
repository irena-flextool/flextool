"""Reserve rows and connections (#96).

(1) The reverse flow ``v_flow_back`` of one-variable and DC connections counts in the reserve requirements, as in
``nodeBalance_eq`` (it arrives at the arc's source and leaves its sink): in the dynamic requirement, in the n-1
requirement of a connection failing while it flows backwards, in the shortfall scale, and in the storage energy
rows of a one-variable connection to a storage node.  Before, these rows summed ``v_flow`` only.

Fixtures (JSON): ``tests.json`` lossless_2way — ll (no_losses_no_variable_cost, 200 MW) between lw and le; ll
exports 100 MW from lw at t0001 (v_flow) and imports 100 MW into lw at t0002 (v_flow_back) — and wind_battery
(battery_inverter west → battery).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8

pytestmark = pytest.mark.solver

LL = ["Base", "lossless_2way"]
TOL = 1e-6
REL = "reserve__upDown__connection__node"


def _reserve(r: str, ud: str, group: str, providers: list, *, method: str = "timeseries_only",
             reservation: float | None = None, extra: dict | None = None) -> tuple[list, list]:
    """Entities + values of reserve ``r``/``ud`` on ``group`` offered by connections [(conn, node), ...]."""
    grp = [r, ud, group]
    ents = [["upDown", ud, None], ["reserve", r, None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", b64(method, "str")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(1000.0, "float")]]
    if reservation is not None:
        vals.append(["reserve__upDown__group", grp, "reservation", b64(reservation, "float")])
    for conn, node in providers:
        rel = [r, ud, conn, node]
        ents.append([REL, rel, None])
        vals += [[REL, rel, "is_enabled", b64("yes", "str")], [REL, rel, "max_share", b64(1.0, "float")],
                 [REL, rel, "reliability", b64(1.0, "float")]]
        for k, v in (extra or {}).items():
            vals.append([REL, rel, k, b64(v, "float")])
    return ents, vals


def _group(name: str, node: str) -> list:
    return [["group", name, None], ["group__node", [name, node], None]]


def _steps(tpf, alt, ents, vals, chain=LL):
    url, scen = make_db(tpf, alt=alt, values=vals, base_chain=list(chain), entities=ents)
    return list(run(tpf, url, scen).values())


def test_dynamic_counts_reverse_flow(tmp_path_factory) -> None:
    """C1: dynamic up reserve at lw with ll's increase_reserve_ratio 0.5: the requirement row has v_flow_back of
    ll (the import into lw) with −100 (0.5 × unitsize 200) as well as its v_flow; the shortfall scale holds both
    directions' bounds (200 MW).  Its v_flow (the export from lw) has −100 too: the docs count either direction.
    Solved: the requirement is 50 MW at both steps (0.5 × the 100 MW exported at t0001 and imported at t0002) and
    is met.  Before: no v_flow_back term, the import raised no requirement."""
    ents, vals = _reserve("dyu", "up", "gw", [("ll", "lw")], method="dynamic_only",
                          extra={"increase_reserve_ratio": 0.5})
    step = _steps(tmp_path_factory, "cr_dyn", _group("gw", "lw") + ents, vals)[-1]
    rows = row_terms(build(step.flex_data), "reserveBalance_dynamic_eq", "dyu,up,gw,")
    assert len(rows) == 2
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(-100.0)
        assert coef(terms, f"v_flow[ll,lw,le,p2020,{t}]") == pytest.approx(-100.0)
        assert coef(terms, f"vq_reserve[dyu,up,gw,p2020,{t}]") == pytest.approx(200.0)
    flow = {t: x for p, s, k, d, t, x in utf8(step.solution.value("v_flow")).rows()}
    back = {t: x for p, s, k, d, t, x in utf8(step.solution.value("v_flow_back")).rows()}
    req = {t: 100.0 * (flow[t] + back[t]) for t in flow}
    assert req["t0001"] == pytest.approx(50.0, abs=1e-6)   # 0.5 × the 100 MW exported
    assert req["t0002"] == pytest.approx(50.0, abs=1e-6)   # 0.5 × the 100 MW imported
    res = {t: x for p, r, ud, n, d, t, x in utf8(step.solution.value("v_reserve")).rows()}
    q = {t: x for r, ud, g, d, t, x in utf8(step.solution.value("vq_reserve")).rows()}
    for t in ("t0001", "t0002"):
        assert res[t] * 200.0 + q[t] * 200.0 >= req[t] - 1e-6, t


def test_n_1_counts_reverse_flow(tmp_path_factory) -> None:
    """C2 (up): ll fails while importing into lw (v_flow_back at t0002): reserveBalance_up_n_1_eq rows for ll with
    v_flow_back −200 (ratio 1 × unitsize 200).  ll is the only provider and its own reserve does not count, so
    the 100 MW import is a 0.5 shortfall of the 200 MW scale at t0002.  Before: no row."""
    ents, vals = _reserve("n1u", "up", "gw", [("ll", "lw")], method="large_failure_only",
                          extra={"large_failure_ratio": 1.0})
    step = _steps(tmp_path_factory, "cr_n1u", _group("gw", "lw") + ents, vals)[-1]
    rows = row_terms(build(step.flex_data), "reserveBalance_up_n_1_eq", "n1u,up,gw,ll,")
    assert len(rows) == 2
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(-200.0)
        assert coef(terms, f"vq_reserve[n1u,up,gw,p2020,{t}]") == pytest.approx(200.0)
    q = {t: x for r, ud, g, d, t, x in utf8(step.solution.value("vq_reserve")).rows()}
    assert q["t0002"] == pytest.approx(0.5, abs=1e-6)
    assert q["t0001"] == pytest.approx(0.0, abs=1e-6)


def test_n_1_down_counts_reverse_flow(tmp_path_factory) -> None:
    """C2 (down): at le, ll draws power from le by its reverse flow (le → lw at t0002): a downward n-1 row for ll
    at le with v_flow_back −200.  Before: no row (ll has no v_flow arc leaving le)."""
    ents, vals = _reserve("n1d", "down", "ge", [("ll", "le")], method="large_failure_only",
                          extra={"large_failure_ratio": 1.0})
    step = _steps(tmp_path_factory, "cr_n1d", _group("ge", "le") + ents, vals)[-1]
    rows = row_terms(build(step.flex_data), "reserveBalance_down_n_1_eq", "n1d,down,ge,ll,")
    assert len(rows) == 2
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(-200.0)
        assert not [c for c in terms if c.startswith("v_flow[")]


def test_shortfall_scale_includes_reverse_bound(tmp_path_factory) -> None:
    """The shortfall scale of the n-1 group is the reverse flow's bound (200 MW) although ll has no v_flow arc
    into lw.  Before: no scale (no failing v_flow arc)."""
    from flextool.engine.model._reserve import reserve_shortfall_scale

    ents, vals = _reserve("n1u", "up", "gw", [("ll", "lw")], method="large_failure_only",
                          extra={"large_failure_ratio": 1.0})
    step = _steps(tmp_path_factory, "cr_scale", _group("gw", "lw") + ents, vals)[-1]
    scale = reserve_shortfall_scale(step.flex_data)
    assert scale is not None
    assert sorted(utf8(scale.frame)["value"].to_list()) == [pytest.approx(200.0)] * 2


def test_storage_reserve_through_reverse_flow(tmp_path_factory) -> None:
    """C3: battery_inverter as a one-variable connection (west → battery; discharging is its reverse flow) holds
    upward reserve at west with reserve_duration 2: reserve_storage_up_floor of the battery holds that reserve
    (−unitsize × 2 × slope 1).  Before: no row (only v_flow arcs from the storage counted)."""
    ents, vals = _reserve("rs", "up", "electricity", [("battery_inverter", "west")], reservation=5.0)
    vals += [["reserve__upDown__group", ["rs", "up", "electricity"], "reserve_duration", b64(2.0, "float")],
             ["connection", "battery_inverter", "transfer_method", b64("no_losses_no_variable_cost", "str")]]
    step = _steps(tmp_path_factory, "cr_sto", ents, vals, chain=["init", "west", "wind", "battery"])[-1]
    fd = step.flex_data
    us = float(utf8(fd.p_unitsize.frame).filter(pl.col("p") == "battery_inverter")["value"][0])
    rows = row_terms(build(fd), "reserve_storage_up_floor", "battery,")
    assert rows
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_reserve[battery_inverter,rs,up,west,p2020,{t}]") == pytest.approx(-us * 2.0)


# ---------------------------------------------------------------------------
# (2) Connection reserve limited by its net position (connection_reserve_* rows)

UNI = ["init", "west", "coal", "wind", "network", "unidirectional"]
WE = "west_east"


def _vals(step, var: str, **flt) -> dict[str, float]:
    v = utf8(step.solution.value(var))
    for k, x in flt.items():
        v = v.filter(pl.col(k) == x)
    return {t: x for t, x in v.select("t", "value").rows()}


@pytest.mark.parametrize("ud,expect,short_t", [("up", {"t0001": 0.75, "t0002": 0.5}, "t0002"),
                                               ("down", {"t0001": 0.5, "t0002": 0.75}, "t0001")])
def test_one_variable_reserve_net_position(tmp_path_factory, ud, expect, short_t) -> None:
    """D1: ll (200 MW) holds a 150 MW reserve at lw.  Up: room = capacity into lw − import + export, 1.5 units at
    t0001 (export 100 MW: the reservation 0.75 is met) and 0.5 at t0002 (import 100 MW: shortfall 50 MW = 1/3 of
    the 150 MW scale); down mirrored.  Before: 1 unit (200 MW) in both steps whatever the flow."""
    ents, vals = _reserve(f"ts{ud[0]}", ud, "gw", [("ll", "lw")], reservation=150.0)
    step = _steps(tmp_path_factory, f"cr_d1_{ud}", _group("gw", "lw") + ents, vals)[-1]
    res = _vals(step, "v_reserve", p="ll")
    q = _vals(step, "vq_reserve")
    for t, x in expect.items():
        assert res[t] == pytest.approx(x, abs=1e-6), t
    assert q[short_t] == pytest.approx(1.0 / 3.0, abs=1e-6)
    pb = build(step.flex_data)
    rows = row_terms(pb, f"connection_reserve_{ud}", "ll,lw,")
    assert len(rows) == 2
    sign = 1.0 if ud == "up" else -1.0
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(sign)
        assert coef(terms, f"v_flow[ll,lw,le,p2020,{t}]") == pytest.approx(-sign)
        assert coef(terms, f"v_reserve[ll,ts{ud[0]},{ud},lw,p2020,{t}]") == pytest.approx(1.0)


def test_one_way_connection_floor_and_headroom(tmp_path_factory) -> None:
    """D2: one-way west_east (400 MW, efficiency 0.9, west → east) at its sending node west.  Down reserve (send
    more): ``v_flow + R·0.9 ≤ 1`` (R/s, s = 1/0.9); up reserve (send less): ``−v_flow/0.9 + R ≤ 0`` (at most what it
    draws).  Solved (300 MW each): the flow and the down reserve share the line, the up reserve is at most what
    the line draws.  Before: no row, both reserves 1 unit at every step (test_reserve_input_side's
    "connections are not coupled")."""
    ed, vd = _reserve("rdw", "down", "gww", [(WE, "west")], reservation=300.0)
    eu, vu = _reserve("ruw", "up", "gww", [(WE, "west")], reservation=300.0)
    step = _steps(tmp_path_factory, "cr_d2", _group("gww", "west") + ed + eu, vd + vu, chain=UNI)[-1]
    pb = build(step.flex_data)
    dn = row_terms(pb, "connection_reserve_down", f"{WE},west,")
    up = row_terms(pb, "connection_reserve_up", f"{WE},west,")
    assert len(dn) == len(up) == 48
    for row, terms in dn.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[{WE},west,east,p2020,{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_reserve[{WE},rdw,down,west,p2020,{t}]") == pytest.approx(0.9)
    for row, terms in up.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[{WE},west,east,p2020,{t}]") == pytest.approx(-1.0 / 0.9)
        assert coef(terms, f"v_reserve[{WE},ruw,up,west,p2020,{t}]") == pytest.approx(1.0)
    f = _vals(step, "v_flow", p=WE)
    rd = _vals(step, "v_reserve", p=WE, r="rdw")
    ru = _vals(step, "v_reserve", p=WE, r="ruw")
    for t in f:
        assert f[t] + 0.9 * rd[t] <= 1.0 + TOL
        assert ru[t] <= f[t] / 0.9 + TOL
    # The flow uses capacity the down reserve cannot (before: rd = 1 at every step whatever the flow).
    shared = [t for t in f if f[t] > 0.1 and f[t] + 0.9 * rd[t] >= 1.0 - TOL]
    assert shared and all(rd[t] < 1.0 - 0.1 for t in shared)


def test_two_way_reserve_rows(tmp_path_factory) -> None:
    """D3: battery_inverter (regular, η 0.95) with up and down reserve at west (reserve_duration 2): maxFlow of
    its arc into west has no v_reserve; up row (x_in 1, x_out −1/0.95, R 1), down row (x_out 1, x_in −0.95, R 0.95);
    objective 1030.991 M (was 1027.312 M: the down reserve now needs charging headroom, the up reserve may count
    stopping the charging).  The battery is bind_within_timeblock with fix_start 0.1, so 5 MWh is pinned on the
    wrap target v_state[t0048] (#135; 1030.137 M when it sat on the end of step 1).  Derived independently with a
    highspy LP of the fixture (exact self-discharge, pin on the last step of the 48-step block): 1030990802.5756."""
    eu, vu = _reserve("rs", "up", "electricity", [("battery_inverter", "west")], reservation=50.0)
    ed, vd = _reserve("rs", "down", "electricity", [("battery_inverter", "west")], reservation=50.0)
    vals = vu + vd + [["reserve__upDown__group", ["rs", ud, "electricity"], "reserve_duration", b64(2.0, "float")]
                      for ud in ("up", "down")]
    step = _steps(tmp_path_factory, "cr_d3", eu + ed, vals, chain=["init", "west", "wind", "battery"])[-1]
    pb = build(step.flex_data)
    mf = row_terms(pb, "maxFlow", "battery_inverter,battery,west,")
    assert mf and all(not [c for c in t if c.startswith("v_reserve[")] for t in mf.values())
    bi = "battery_inverter"
    for row, terms in row_terms(pb, "connection_reserve_up", f"{bi},west,").items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[{bi},battery,west,p2020,{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_flow[{bi},west,battery,p2020,{t}]") == pytest.approx(-1.0 / 0.95)
        assert coef(terms, f"v_reserve[{bi},rs,up,west,p2020,{t}]") == pytest.approx(1.0)
    for row, terms in row_terms(pb, "connection_reserve_down", f"{bi},west,").items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[{bi},west,battery,p2020,{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_flow[{bi},battery,west,p2020,{t}]") == pytest.approx(-0.95)
        assert coef(terms, f"v_reserve[{bi},rs,down,west,p2020,{t}]") == pytest.approx(0.95)
    assert step.solution.obj == pytest.approx(1030.9908025756e6, rel=1e-8)


def test_online_connection_reserve(tmp_path_factory) -> None:
    """D4 (replaces test_up_reserve_online::test_online_connection_up_reserve): west_east with linear online
    variables (JSON startup_method) holds upward reserve at its receiving node east and at its sending node west.
    connection_reserve_up_online_linear[west_east,east] = v_flow + R − v_online ≤ 0 (the old maxFlow_online
    coupling, now from its own row), maxFlow_online_linear has no v_reserve; at west the floor −v_flow/0.9 + R ≤ 0
    (no online row: west_east has no in-channel at west)."""
    ents = []
    vals = []
    for node in ("east", "west"):
        e, v = _reserve("ru", "up", f"g_{node}", [(WE, node)], reservation=10.0)
        ents += _group(f"g_{node}", node) + e
        vals += v
    vals += [["connection", WE, "startup_method", b64("linear", "str")]]
    step = _steps(tmp_path_factory, "cr_d4", ents, vals, chain=["init", "west", "coal", "coal_min_load", "wind",
                                                                "network", "unidirectional"])[-1]
    pb = build(step.flex_data)
    on = row_terms(pb, "connection_reserve_up_online_linear", f"{WE},east,")
    assert len(on) == 48
    for row, terms in on.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[{WE},west,east,p2020,{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_reserve[{WE},ru,up,east,p2020,{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_online_linear[{WE},p2020,{t}]") < 0.0
    assert not row_terms(pb, "connection_reserve_up_online_linear", f"{WE},west,")
    for terms in row_terms(pb, "maxFlow_online_linear", f"{WE},").values():
        assert not [c for c in terms if c.startswith("v_reserve[")]
    west = row_terms(pb, "connection_reserve_up", f"{WE},west,")
    assert len(west) == 48
    for row, terms in west.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow[{WE},west,east,p2020,{t}]") == pytest.approx(-1.0 / 0.9)


def test_exact_connection_reserve(tmp_path_factory) -> None:
    """D5: exact ll (forward lw → le) with up and down reserve at lw.  The up row's channel is the reverse arc
    le → lw: ``… + Ū·y ≤ Ū``; the down row's channel is the forward arc lw → le: ``… − Ū·y ≤ 0``."""
    eu, vu = _reserve("ru", "up", "gw", [("ll", "lw")], reservation=50.0)
    ed, vd = _reserve("rd", "down", "gw", [("ll", "lw")], reservation=50.0)
    vals = vu + vd + [["connection", "ll", "transfer_method", b64("exact", "str")]]
    step = _steps(tmp_path_factory, "cr_d5", _group("gw", "lw") + eu + ed, vals)[-1]
    pb = build(step.flex_data)
    from tests.engine._lp_rows import row_bounds
    for ud, sign, cap in (("up", 1.0, 1.0), ("down", -1.0, 0.0)):
        rows = row_terms(pb, f"connection_reserve_{ud}_exact", "ll,lw,")
        bounds = row_bounds(pb, f"connection_reserve_{ud}_exact", "ll,lw,")
        assert len(rows) == 2, ud
        for row, terms in rows.items():
            t = row.split(",")[-1].rstrip("]")
            assert coef(terms, f"v_exact_forward[ll,p2020,{t}]") == pytest.approx(sign), (ud, terms)
            assert bounds[row][1] == pytest.approx(cap)


def test_connection_reserve_rolling_and_stochastic(tmp_path_factory) -> None:
    """Every roll of the rolling fullYear dispatch and every branch period of the stochastic 2-day dispatch get the
    connection reserve rows (sending-node floor of a one-way connection)."""
    eu, vu = _reserve("ruw", "up", "gww", [(WE, "west")], reservation=100.0)
    rolls = _steps(tmp_path_factory, "cr_roll", _group("gww", "west") + eu,
                   vu + [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
                         ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]],
                   chain=["init", "west", "battery", "wind", "coal", "network", "unidirectional", "fullYear",
                          "dispatch_fullYear_roll"])
    assert len(rolls) > 1
    for s in rolls:
        rows = row_terms(build(s.flex_data), "connection_reserve_up", f"{WE},west,")
        assert len(rows) == s.flex_data.dt.height
    from tests.engine._unit_db import FLEXTOOL_ROOT
    ents = [["node", "north", None], ["connection", "dn_north", None],
            ["connection__node__node", ["dn_north", "demand_node", "north"], None]]
    vals = [["node", "north", "node_type", b64("balance", "str")], ["node", "north", "inflow", b64(-150.0, "float")],
            ["node", "north", "penalty_up", b64(5000.0, "float")],
            ["connection", "dn_north", "transfer_method", b64("unidirectional", "str")],
            ["connection", "dn_north", "existing", b64(200.0, "float")]]
    er, vr = _reserve("rdn", "up", "gdn", [("dn_north", "demand_node")], reservation=50.0)
    url, scen = make_db(tmp_path_factory, alt="cr_stoch", values=vals + vr, base_chain=["base", "init", "system",
                                                                                          "2day"],
                        entities=ents + _group("gdn", "demand_node") + er,
                        fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json")
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    rows = row_terms(build(step.flex_data), "connection_reserve_up", "dn_north,demand_node,")
    periods = {r.split(",")[-2] for r in rows}
    assert len(rows) == step.flex_data.dt.height and len(periods) > 1


def test_connection_reserve_with_investment(tmp_path_factory) -> None:
    """The built rows carry the channel arc's invest terms (as its maxFlow row): battery_inverter can invest
    (wind_battery_invest), its connection_reserve_up row at west has −M·v_invest_p with the maxFlow multiplier of
    its in-arc battery → west."""
    eu, vu = _reserve("ru", "up", "electricity", [("battery_inverter", "west")], reservation=5.0)
    step = _steps(tmp_path_factory, "cr_inv", eu, vu,
                  chain=["init", "west", "wind", "battery", "battery_invest", "5weeks"])[-1]
    pb = build(step.flex_data)
    up = row_terms(pb, "connection_reserve_up", "battery_inverter,west,")
    mf = row_terms(pb, "maxFlow", "battery_inverter,battery,west,")
    assert up and mf
    for row, terms in up.items():
        d, t = row.split(",")[-2], row.split(",")[-1].rstrip("]")
        inv = {c: v for c, v in terms.items() if c.startswith("v_invest_p[battery_inverter,")}
        mf_inv = {c: v for c, v in mf[f"maxFlow[battery_inverter,battery,west,{d},{t}]"].items()
                  if c.startswith("v_invest_p[battery_inverter,")}
        assert inv and inv == pytest.approx(mf_inv)


# ---------------------------------------------------------------------------
# (3) The net-position headroom reaches 2 × capacity summed over the products (Q-c13-A)

ONE_VAR = "no_losses_no_variable_cost"


def _swing(mw: float, tm: str, products: list[tuple[str, str, str, str, float]]) -> tuple[list, list]:
    """ll (200 MW, transfer_method ``tm``) exports ``mw`` from lw at t0001 and imports ``mw`` into lw at t0002;
    timeseries ``products`` [(reserve, upDown, group, node, reservation)] offered by ll alone (max_share 1)."""
    ents = _group("gw", "lw") + _group("ge", "le")
    vals = [["node", "lw", "inflow", b64({"index_type": "str", "data": [["t0001", mw], ["t0002", -mw]]}, "map")],
            ["node", "le", "inflow", b64({"index_type": "str", "data": [["t0001", -mw], ["t0002", mw]]}, "map")],
            ["connection", "ll", "transfer_method", b64(tm, "str")]]
    for r, ud, g, n, res in products:
        e, v = _reserve(r, ud, g, [("ll", n)], reservation=res)
        ents, vals = ents + e, vals + v
    return ents, vals


def _units(step, r: str) -> dict[str, float]:
    return _vals(step, "v_reserve", p="ll", r=r)


@pytest.mark.parametrize("tm", [ONE_VAR, "regular"])
@pytest.mark.parametrize("mw", [200.0, 100.0])
def test_net_position_two_products_reach_twice_capacity(tmp_path_factory, tm: str, mw: float) -> None:
    """Two up products r1, r2 and two down products d1, d2 at lw, two down products e1, e2 at le, 200 MW each.
    The headroom is capacity − net signed flow, linear in the flow and summed over the products: exporting mw
    from lw, lw's up headroom is 200 + mw MW (stop exporting, then import) and its down headroom 200 − mw; le
    (importing mw) has a down headroom of 200 + mw.  At full export (mw = 200) that is 2 × capacity up at lw
    and down at le, and nothing down at lw; t0002 is the mirror.  Rows: ``capacity_in − net flow + Σ R ≤ 1``
    (one-variable: ``−v_flow + v_flow_back``; regular: ``−v_flow[lw→le] + v_flow[le→lw]``)."""
    prods = [("r1", "up", "gw", "lw", 200.0), ("r2", "up", "gw", "lw", 200.0),
             ("d1", "down", "gw", "lw", 200.0), ("d2", "down", "gw", "lw", 200.0),
             ("e1", "down", "ge", "le", 200.0), ("e2", "down", "ge", "le", 200.0)]
    ents, vals = _swing(mw, tm, prods)
    step = _steps(tmp_path_factory, f"cr_2x_{tm[:3]}_{int(mw)}", ents, vals)[-1]
    u = {r: _units(step, r) for r, *_ in prods}
    share = mw / 200.0
    expect = {"t0001": {"up_lw": 1.0 + share, "down_lw": 1.0 - share, "down_le": 1.0 + share},
              "t0002": {"up_lw": 1.0 - share, "down_lw": 1.0 + share, "down_le": 1.0 - share}}
    for t, e in expect.items():
        assert u["r1"][t] + u["r2"][t] == pytest.approx(e["up_lw"], abs=TOL), t
        assert u["d1"][t] + u["d2"][t] == pytest.approx(e["down_lw"], abs=TOL), t
        assert u["e1"][t] + u["e2"][t] == pytest.approx(e["down_le"], abs=TOL), t
        for r in u:
            assert u[r][t] <= 1.0 + TOL, (r, t)      # each product ≤ max_share × capacity
    pb = build(step.flex_data)
    back_in = "v_flow_back[ll,lw,le,p2020,{t}]" if tm == ONE_VAR else "v_flow[ll,le,lw,p2020,{t}]"
    for fam, node, sign, rs in (("connection_reserve_up", "lw", 1.0, ("r1", "r2")),
                                ("connection_reserve_down", "lw", -1.0, ("d1", "d2")),
                                ("connection_reserve_down", "le", 1.0, ("e1", "e2"))):
        rows = row_terms(pb, fam, f"ll,{node},")
        assert len(rows) == 2, (fam, node)
        for row, terms in rows.items():
            t = row.split(",")[-1].rstrip("]")
            assert coef(terms, back_in.format(t=t)) == pytest.approx(sign)
            assert coef(terms, f"v_flow[ll,lw,le,p2020,{t}]") == pytest.approx(-sign)
            ud = "up" if fam.endswith("up") else "down"
            for r in rs:
                assert coef(terms, f"v_reserve[ll,{r},{ud},{node},p2020,{t}]") == pytest.approx(1.0)


def test_net_position_single_product_capped_by_max_share(tmp_path_factory) -> None:
    """One up product r3 and one down product d3 at lw, 400 MW each, ll swinging 200 MW: a single product stays
    at most max_share × capacity (1 unit = 200 MW; ``max_share`` is a share of the provider's capacity), so half
    of r3 is short at t0001 (full export) and half of d3 at t0002 (full import).  The owner's 2 × capacity rule is
    about the sum over products (test above)."""
    prods = [("r3", "up", "gw", "lw", 400.0), ("d3", "down", "gw", "lw", 400.0)]
    ents, vals = _swing(200.0, ONE_VAR, prods)
    step = _steps(tmp_path_factory, "cr_2x_one", ents, vals)[-1]
    r3, d3 = _units(step, "r3"), _units(step, "d3")
    q3 = _vals(step, "vq_reserve", r="r3")
    qd = _vals(step, "vq_reserve", r="d3")
    assert r3["t0001"] == pytest.approx(1.0, abs=TOL) and q3["t0001"] == pytest.approx(0.5, abs=TOL)
    assert d3["t0002"] == pytest.approx(1.0, abs=TOL) and qd["t0002"] == pytest.approx(0.5, abs=TOL)
    pb = build(step.flex_data)
    for fam, r in (("reserve_process_upward", "r3"), ("reserve_process_downward", "d3")):
        rows = row_terms(pb, fam, f"ll,{r},")
        assert rows, fam
        for row, terms in rows.items():
            t = row.split(",")[-1].rstrip("]")
            assert coef(terms, f"v_reserve[ll,{r},") == pytest.approx(1.0), (row, t)
