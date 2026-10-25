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
    directions' bounds (200 MW).  Solved: the requirement is 50 MW at t0002 (100 MW imported) and is met.  Before: no
    v_flow_back term, the import raised no requirement."""
    ents, vals = _reserve("dyu", "up", "gw", [("ll", "lw")], method="dynamic_only",
                          extra={"increase_reserve_ratio": 0.5})
    step = _steps(tmp_path_factory, "cr_dyn", _group("gw", "lw") + ents, vals)[-1]
    rows = row_terms(build(step.flex_data), "reserveBalance_dynamic_eq", "dyu,up,gw,")
    assert len(rows) == 2
    for row, terms in rows.items():
        t = row.split(",")[-1].rstrip("]")
        assert coef(terms, f"v_flow_back[ll,lw,le,p2020,{t}]") == pytest.approx(-100.0)
        assert coef(terms, f"vq_reserve[dyu,up,gw,p2020,{t}]") == pytest.approx(200.0)
    flow = {t: x for p, s, k, d, t, x in utf8(step.solution.value("v_flow")).rows()}
    back = {t: x for p, s, k, d, t, x in utf8(step.solution.value("v_flow_back")).rows()}
    req = {t: 100.0 * (flow[t] + back[t]) for t in flow}
    assert req["t0002"] == pytest.approx(50.0, abs=1e-6)   # 0.5 × the 100 MW imported
    res = {t: x for p, r, ud, n, d, t, x in utf8(step.solution.value("v_reserve")).rows()}
    q = {t: x for r, ud, g, d, t, x in utf8(step.solution.value("vq_reserve")).rows()}
    assert res["t0002"] * 200.0 + q["t0002"] * 200.0 >= req["t0002"] - 1e-6


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
