"""The ``exact`` transfer method: a connection flows in one direction per time step.

docs/how_to.md: "*Exact* method does not allow flow in both directions at the same time, but it requires a binary
variable".  The engine mapped ``exact`` to an online integer that capped both arcs (v_flow ≤ cap·v_online_integer)
but did not stop simultaneous flow, so a lossy exact line could dissipate a surplus by flowing both ways, and it
got free "startups".  Now one binary per connection and step, v_exact_forward, switches the direction:
``forward ≤ Ū·y``, ``reverse ≤ Ū·(1 − y)``, Ū the arc's structural upper bound (GMPL's integer unit count
fails when the built capacity is not a whole number of units, e.g. after a continuous investment).

Fixtures: ``tests.json`` lossless_2way (ll 200 MW, lw ↔ le, two steps) and the network chain, from JSON.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._lp_rows import coef, row_bounds, row_terms, step_of
from tests.engine._unit_db import b64, build, last_stochastic, make_db, run, utf8

pytestmark = pytest.mark.solver

LL = ["Base", "lossless_2way"]
NET = ["init", "west", "coal", "wind", "network"]
TOL = 1e-6


def _solve(tpf, alt, values, chain=LL):
    url, scen = make_db(tpf, alt=alt, values=values, base_chain=list(chain))
    return list(run(tpf, url, scen).values())[-1]


def _tmap(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[t, v] for t, v in values.items()]}, "map")


def _flows(step, conn: str) -> dict[tuple[str, str], dict[str, float]]:
    """{(source, sink): {t: MW}} of ``conn``."""
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == conn)["value"][0])
    out: dict = {}
    for p, s, k, d, t, x in utf8(step.solution.value("v_flow")).filter(pl.col("p") == conn).rows():
        out.setdefault((s, k), {})[t] = x * us
    return out


def _dissipate(method: str) -> list:
    return [["connection", "ll", "transfer_method", b64(method, "str")],
            ["connection", "ll", "efficiency", b64(0.9, "float")],
            ["node", "lw", "inflow", b64(50.0, "float")], ["node", "le", "inflow", b64(0.0, "float")]]


def test_exact_no_simultaneous_flow(tmp_path_factory) -> None:
    """B1: a 50 MW surplus at lw with no consumer and a costly spill.  A regular line dissipates it by flowing
    both ways (forward 200 MW, reverse 180 MW: 42.2 MW of losses, 7.78 MW spilled, objective 681.3 M); an exact
    line sends the 50 MW to le, where 45 MW is spilled (objective 3942 M, the ratio 45 / 7.78 of the spill).
    Before: the exact line behaved like the regular one."""
    regular = _solve(tmp_path_factory, "ex_reg", _dissipate("regular"))
    exact = _solve(tmp_path_factory, "ex_exact", _dissipate("exact"))
    assert regular.solution.obj == pytest.approx(681.3333333e6, rel=1e-6)
    assert exact.solution.obj == pytest.approx(3942.0e6, rel=1e-6)
    f = _flows(exact, "ll")
    for t in ("t0001", "t0002"):
        assert min(f[("lw", "le")][t], f[("le", "lw")][t]) <= TOL, (t, f)
    rf = _flows(regular, "ll")
    assert min(rf[("lw", "le")]["t0001"], rf[("le", "lw")]["t0001"]) > 1.0


def test_exact_rows(tmp_path_factory) -> None:
    """The direction rows: ``v_flow[fwd] − Ū·y ≤ 0`` and ``v_flow[rev] + Ū·y ≤ Ū`` with Ū = 1 (existing capacity,
    units of unitsize), y binary; no online variables without a startup method (before: v_online_integer)."""
    step = _solve(tmp_path_factory, "ex_rows", _dissipate("exact"))
    pb = build(step.flex_data)
    fwd = row_terms(pb, "maxFlow_exact_forward", "ll,lw,le,")
    rev = row_terms(pb, "maxFlow_exact_reverse", "ll,le,lw,")
    rb = row_bounds(pb, "maxFlow_exact_reverse", "ll,le,lw,")
    assert len(fwd) == len(rev) == 2
    for row, terms in fwd.items():
        d, t = step_of(row)
        assert coef(terms, f"v_flow[ll,lw,le,{d},{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_exact_forward[ll,{d},{t}]") == pytest.approx(-1.0)
    for row, terms in rev.items():
        d, t = step_of(row)
        assert coef(terms, f"v_flow[ll,le,lw,{d},{t}]") == pytest.approx(1.0)
        assert coef(terms, f"v_exact_forward[ll,{d},{t}]") == pytest.approx(1.0)
        assert rb[row][1] == pytest.approx(1.0)
    m = pb.canonicalise()
    ints = {nm for i, nm in enumerate(m.col_names) if m.col_int[i]}
    assert {nm for nm in ints if nm.startswith("v_exact_forward[ll,")} == {
        "v_exact_forward[ll,p2020,t0001]", "v_exact_forward[ll,p2020,t0002]"}
    assert all(m.col_ub[i] == 1.0 for i, nm in enumerate(m.col_names) if nm.startswith("v_exact_forward["))
    assert not [nm for nm in m.col_names if nm.startswith("v_online_")]


def _invest(existing: float, extra: list) -> list:
    return [["connection", "ll", "transfer_method", b64("exact", "str")],
            ["connection", "ll", "existing", b64(existing, "float")],
            ["connection", "ll", "invest_method", b64("invest_total", "str")],
            ["connection", "ll", "invest_max_total", b64(100.0, "float")],
            ["connection", "ll", "invest_cost", b64(1.0, "float")],
            ["connection", "ll", "lifetime", b64(20.0, "float")],
            ["connection", "ll", "discount_rate", b64(0.05, "float")],
            ["solve", "lossless_2way_solve", "invest_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
            ["solve", "lossless_2way_solve", "realized_invest_periods",
             b64({"value_type": "str", "data": ["p2020"]}, "array")]] + extra


@pytest.mark.parametrize("existing,mw", [(0.0, 100.0), (200.0, 300.0)], ids=["greenfield", "fractional_units"])
def test_exact_invested_capacity_both_directions(tmp_path_factory, existing, mw) -> None:
    """B2′: invested capacity flows in both directions, one direction per step.  Investment is capped at 100 MW.
    Greenfield (existing 0, default unitsize 1000, 100 MW invested: 0.1 units) and existing 200 MW (default
    unitsize = existing, 200 MW) plus 100 MW invested (1.5 units).  lw and le swap a surplus / demand of ``mw``
    between the two steps.  The GMPL integer unit count (y ∈ ℤ, y ≤ N, forward ≤ y, reverse ≤ N − y) allows no
    forward flow in the first case and caps it at 200 MW in the second (both fail under a prototype of that form)."""
    extra = [["node", "lw", "inflow", _tmap({"t0001": mw, "t0002": -mw})],
             ["node", "le", "inflow", _tmap({"t0001": -mw, "t0002": mw})]]
    step = _solve(tmp_path_factory, f"ex_inv_{int(existing)}", _invest(existing, extra))
    f = _flows(step, "ll")
    assert f[("lw", "le")]["t0001"] == pytest.approx(mw, rel=1e-6)
    assert f[("le", "lw")]["t0002"] == pytest.approx(mw, rel=1e-6)
    for t in ("t0001", "t0002"):
        assert min(f[("lw", "le")][t], f[("le", "lw")][t]) <= TOL


def test_exact_with_online(tmp_path_factory) -> None:
    """B3: an exact connection with a linear startup method has both the online rows and the direction rows."""
    step = _solve(tmp_path_factory, "ex_online", [
        ["connection", "west_east", "transfer_method", b64("exact", "str")],
        ["connection", "west_east", "startup_method", b64("linear", "str")]], chain=NET)
    pb = build(step.flex_data)
    for fam in ("maxFlow_exact_forward", "maxFlow_exact_reverse", "maxFlow_online_linear"):
        assert row_terms(pb, fam, "west_east,"), fam
    m = pb.canonicalise()
    assert [nm for nm in m.col_names if nm.startswith("v_online_linear[west_east,")]
    assert not [nm for nm in m.col_names if nm.startswith("v_online_integer[")]


def test_process_exact_dump_writes_csv(tmp_path_factory) -> None:
    """Critique 8f, writer half: the solved FlexData has process_exact = {ll}, and dump_csvs writes it to
    solve_data/process_exact.csv.  The CSV reload half is ``tests/csv_reload/test_exact_connection.py``
    (c24 S1-8)."""
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt="ex_dump", values=_dissipate("exact"), base_chain=LL)
    wf = tmp_path_factory.mktemp("ex_dump_wf")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    original = list(steps.values())[-1].flex_data
    assert set(utf8(original.process_exact)["p"].to_list()) == {"ll"}
    out = tmp_path_factory.mktemp("ex_dump_out")
    original.dump_csvs(out, include_heavy=True)
    path = out / "solve_data" / "process_exact.csv"
    assert path.exists()
    assert set(pl.read_csv(path).to_series(0).to_list()) == {"ll"}


def _no_simultaneous(step, conn: str, a: str, b: str) -> int:
    rows = {}
    for p, s, k, d, t, x in utf8(step.solution.value("v_flow")).filter(pl.col("p") == conn).rows():
        rows.setdefault((d, t), {})[(s, k)] = x
    for key, f in rows.items():
        assert min(f.get((a, b), 0.0), f.get((b, a), 0.0)) <= TOL, key
    return len(rows)


def test_exact_rolling(tmp_path_factory) -> None:
    """Every roll of the rolling fullYear dispatch has the direction rows of the exact west_east over its steps
    (no inter-temporal link, nothing carried) and its solution never flows both ways."""
    steps = list(run(tmp_path_factory, *make_db(
        tmp_path_factory, alt="ex_roll",
        values=[["connection", "west_east", "transfer_method", b64("exact", "str")],
                ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
                ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]],
        base_chain=["init", "west", "battery", "wind", "coal", "network", "fullYear",
                    "dispatch_fullYear_roll"])).values())
    assert len(steps) > 1
    for s in steps:
        n = s.flex_data.dt.height
        pb = build(s.flex_data)
        assert len(row_terms(pb, "maxFlow_exact_forward", "west_east,west,east,")) == n
        assert len(row_terms(pb, "maxFlow_exact_reverse", "west_east,east,west,")) == n
        assert _no_simultaneous(s, "west_east", "west", "east") == n


def test_exact_stochastic(tmp_path_factory) -> None:
    """Stochastic 2-day dispatch with an exact connection demand_node ↔ north: direction rows in every period of
    the trunk and the forecast branches, and no simultaneous flow in any of them."""
    from tests.engine._unit_db import FLEXTOOL_ROOT

    ents = [["node", "north", None], ["connection", "dn_x", None],
            ["connection__node__node", ["dn_x", "demand_node", "north"], None]]
    vals = [["node", "north", "node_type", b64("balance", "str")], ["node", "north", "inflow", b64(-150.0, "float")],
            ["node", "north", "penalty_up", b64(5000.0, "float")],
            ["node", "north", "penalty_down", b64(5000.0, "float")],
            ["connection", "dn_x", "transfer_method", b64("exact", "str")],
            ["connection", "dn_x", "existing", b64(200.0, "float")],
            ["connection", "dn_x", "efficiency", b64(0.9, "float")]]
    url, scen = make_db(tmp_path_factory, alt="ex_stoch", values=vals, base_chain=["base", "init", "system", "2day"],
                        entities=ents, fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json")
    step = last_stochastic(run(tmp_path_factory, url, scen))
    pb = build(step.flex_data)
    fwd = row_terms(pb, "maxFlow_exact_forward", "dn_x,demand_node,north,")
    assert len(fwd) == step.flex_data.dt.height
    assert len({r.split(",")[-2] for r in fwd}) > 1
    assert _no_simultaneous(step, "dn_x", "demand_node", "north") == step.flex_data.dt.height
