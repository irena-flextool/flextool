"""The group ``increase_reserve_ratio`` adds demand to the dynamic requirement (#23(c), owner Q4 2026-10-24).

``reserveBalance_dynamic_eq[r, ud, g, d, t]`` gets ``+ ratio_g · Σ_{n ∈ g} max(−inflow[n,d,t], 0) / Δt[d,t]``
(``_reserve.group_demand_requirement``): the demand of the group's nodes in MW in each step, each node's
negative inflow clipped before the group sum, without the invest-solve energy margin (Q-c13-C default: the
margin is an adequacy device, not load).  The parameter alone activates the reserve, the shortfall scale
covers the term, and a ratio on a group without a dynamic method is ignored with a warning.  Before, the
parameter was exported but never read.

Fixtures: ``tests.json`` (``coal`` 500 MW at ``west``; west inflow a 72-step time series, −589 MW at t0001),
``stochastics.json`` (a per-branch inflow is authored), built from JSON.
"""
from __future__ import annotations

import base64
import collections
import json
import logging

import polars as pl
import pytest

from tests.engine._unit_db import (
    BASE_FIXTURE_JSON, FLEXTOOL_ROOT, PLANT, PLANT_CHAIN, b64, build, make_db, run, solve, time_map, utf8,
)

pytestmark = pytest.mark.solver

REL = "reserve__upDown__unit__node"
GRP = ["dz", "up", "electricity"]
SOLVE = "y2020_2day_dispatch"


def _json_inflow(alt: str = "west", node: str = "west") -> dict[str, float]:
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    for cls, ent, param, val, a in data["parameter_values"]:
        if cls == "node" and ent == node and param == "inflow" and a == alt:
            v = json.loads(base64.b64decode(val[0]))
            return {t: float(x) for t, x in v["data"]}
    raise KeyError((alt, node))


WEST = _json_inflow()


def _dz(ratio: float = 0.1, method: str = "dynamic_only", max_share: float = 0.1,
        penalty: float = 1000.0) -> tuple[list, list]:
    """``dz`` (up) on electricity with the group ratio ``ratio`` and coal as the only provider (no process
    ratio, no reservation)."""
    rel = ["dz", "up", PLANT, "west"]
    ents = [["reserve", "dz", None], ["reserve__upDown__group", GRP, None], [REL, rel, None]]
    vals = [["reserve__upDown__group", GRP, "reserve_method", b64(method, "str")],
            ["reserve__upDown__group", GRP, "penalty_reserve", b64(penalty, "float")],
            [REL, rel, "is_enabled", b64("yes", "str")],
            [REL, rel, "max_share", b64(max_share, "float")],
            [REL, rel, "reliability", b64(1.0, "float")]]
    if ratio:
        vals.append(["reserve__upDown__group", GRP, "increase_reserve_ratio", b64(ratio, "float")])
    return ents, vals


def _rows(pb, family: str = "reserveBalance_dynamic_eq") -> dict[tuple[str, str], tuple[float, dict]]:
    """``{(d, t): (row lower bound, {column: coefficient})}`` of the dz rows of ``family``."""
    m = pb.canonicalise()
    idx = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(family + "[dz,")}
    terms: dict[int, dict[str, float]] = collections.defaultdict(dict)
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            if m.row_idx[k] in idx:
                terms[m.row_idx[k]][m.col_names[c]] = m.val[k]
    out = {}
    for i, nm in idx.items():
        d, t = nm.rstrip("]").split(",")[-2:]
        out[(d, t)] = (float(m.row_lb[i]), terms[i])
    return out


def _rhs(fd) -> dict[tuple[str, str], float]:
    return {k: v[0] for k, v in _rows(build(fd)).items()}


# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def one_positive(tmp_path_factory):
    """West inflow −300 except +100 at t0005."""
    ents, vals = _dz()
    return solve(tmp_path_factory, alt="gd_pos", base_chain=PLANT_CHAIN, entities=ents,
                 values=vals + [["node", "west", "inflow", time_map({"t0005": 100.0}, -300.0)]])


def test_group_ratio_alone_activates(one_positive) -> None:
    """A dynamic_only group whose only driver is its own increase_reserve_ratio is active: coal gets v_reserve
    and is in the dynamic rows."""
    fd = one_positive.flex_data
    assert set(utf8(fd.prundt.select("p", "r")).unique().iter_rows()) == {(PLANT, "dz")}
    for _, (_, terms) in _rows(build(fd)).items():
        assert [c for c in terms if c.startswith(f"v_reserve[{PLANT},dz,up,west,")]


def test_group_ratio_adds_demand_to_dynamic_row(one_positive) -> None:
    """RHS = 0.1 · max(−inflow, 0) / Δt at every step: 30 MW, and 0 at the step with positive inflow.
    Before: 0 (the parameter was not read)."""
    rhs = _rhs(one_positive.flex_data)
    assert len(rhs) == one_positive.flex_data.dt.height
    for (_d, t), lb in rhs.items():
        assert lb == pytest.approx(0.0 if t == "t0005" else 30.0, abs=1e-9), t


def test_positive_inflow_does_not_offset(tmp_path_factory) -> None:
    """A second node of the group with positive inflow (+200) neither counts nor offsets west's demand."""
    ents, vals = _dz()
    ents += [["node", "spring", None], ["group__node", ["electricity", "spring"], None]]
    vals += [["node", "spring", "node_type", b64("balance", "str")],
             ["node", "spring", "inflow", b64(200.0, "float")],
             ["node", "spring", "penalty_down", b64(1.0, "float")]]
    step = solve(tmp_path_factory, alt="gd_spring", base_chain=PLANT_CHAIN, entities=ents, values=vals)
    for (_d, t), lb in _rhs(step.flex_data).items():
        assert lb == pytest.approx(0.1 * -WEST[t], rel=1e-12), t


def test_demand_is_per_step_mw(tmp_path_factory) -> None:
    """With 2-hour steps the inflow is MWh per step: RHS = 0.1 · (−inflow of the two hours) / 2."""
    ents, vals = _dz()
    step = solve(tmp_path_factory, alt="gd_2h", base_chain=PLANT_CHAIN, entities=ents,
                 values=vals + [["solve", SOLVE, "new_stepduration", b64(2.0, "float")]])
    fd = step.flex_data
    dur = {(d, t): v for d, t, v in utf8(fd.p_step_duration.frame).select("d", "t", "value").iter_rows()}
    assert set(dur.values()) == {2.0}
    hours = list(WEST)
    rhs = _rhs(fd)
    assert rhs
    for (d, t), lb in rhs.items():
        i = hours.index(t)
        expected = 0.1 * -(WEST[hours[i]] + WEST[hours[i + 1]]) / 2.0
        assert lb == pytest.approx(expected, rel=1e-12), t


def test_scale_covers_group_demand(tmp_path_factory) -> None:
    """Without a provider able to hold reserve (max_share 0) the whole requirement is shortfall: the scale
    equals the demand term, vq · scale = requirement and the shortfall is priced."""
    from flextool.engine.model._reserve import group_demand_requirement, reserve_shortfall_scale

    ents, vals = _dz(max_share=0.0)
    step = solve(tmp_path_factory, alt="gd_short", base_chain=PLANT_CHAIN, entities=ents, values=vals)
    fd = step.flex_data
    scale = {(d, t): v for d, t, v in utf8(reserve_shortfall_scale(fd).frame).filter(pl.col("r") == "dz")
             .select("d", "t", "value").iter_rows()}
    demand = {(d, t): v for d, t, v in utf8(group_demand_requirement(fd).frame).select("d", "t", "value").iter_rows()}
    assert scale == pytest.approx(demand)
    vq = {(d, t): v for d, t, v in utf8(step.solution.value("vq_reserve")).filter(pl.col("r") == "dz")
          .select("d", "t", "value").iter_rows()}
    for key, req in demand.items():
        assert req > 0.0
        assert vq[key] * scale[key] == pytest.approx(req, rel=1e-6), key
    pen = [c for c in build(fd).canonicalise().col_names if c.startswith("vq_reserve[dz,")]
    assert pen


def test_constant_and_map_inflow_build_the_same_row(tmp_path_factory) -> None:
    """West inflow −300 as a constant and as a time map of −300 everywhere give identical dynamic rows."""
    ents, vals = _dz()
    rows = []
    for alt, inflow in (("gd_const", b64(-300.0, "float")), ("gd_map", time_map({}, -300.0))):
        step = solve(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN, entities=ents,
                     values=vals + [["node", "west", "inflow", inflow]])
        rows.append(_rows(build(step.flex_data)))
    assert rows[0] == rows[1]
    assert {lb for lb, _ in rows[0].values()} == {30.0}


def test_group_demand_in_rolls(tmp_path_factory) -> None:
    """Rolling solve (horizon 12 h, jump 6 h): every roll has the ratio field and its dynamic RHS equals
    0.1 · max(−inflow, 0) of the JSON inflow series at that roll's steps."""
    ents, vals = _dz()
    url, scen = make_db(tmp_path_factory, alt="gd_roll", base_chain=PLANT_CHAIN, entities=ents, values=vals + [
        ["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
        ["solve", SOLVE, "rolling_solve_horizon", b64(12.0, "float")],
        ["solve", SOLVE, "rolling_solve_jump", b64(6.0, "float")]])
    steps = run(tmp_path_factory, url, scen)
    assert len(steps) > 2
    seen: set[str] = set()
    for name, step in steps.items():
        fd = step.flex_data
        assert fd.p_reserve_upDown_group_increase_reserve_ratio is not None, name
        rhs = _rhs(fd)
        assert len(rhs) == fd.dt.height, name
        for (_d, t), lb in rhs.items():
            assert lb == pytest.approx(0.1 * max(-WEST[t], 0.0), rel=1e-12), (name, t)
            seen.add(t)
    assert len(seen) >= 12


def test_group_demand_per_branch(tmp_path_factory) -> None:
    """Stochastic solve with a per-branch demand at demand_node: each branch period's dynamic RHS is
    0.1 × that branch's demand (from the JSON series), so the branches get different requirements."""
    steps_t = [f"t{i:04d}" for i in range(1, 49)]
    bases = {"realized": -3000.0, "upper": -3300.0, "lower": -2700.0, "mid": -3100.0}

    def series(base: float) -> dict:
        return {"type": "map", "index_type": "str", "data": [[t, base - 10.0 * i] for i, t in enumerate(steps_t)]}

    inflow = b64({"index_type": "str", "data": [[br, {"type": "map", "index_type": "str",
                                                      "data": [["t0001", series(b)]]}] for br, b in bases.items()]},
                 "map")
    grp = ["dz", "up", "res_g"]
    rel = ["dz", "up", "gas_plant", "demand_node"]
    ents = [["group", "res_g", None], ["group__node", ["res_g", "demand_node"], None],
            ["group__node", ["add_stochastics", "demand_node"], None],
            ["reserve", "dz", None], ["upDown", "up", None], ["reserve__upDown__group", grp, None], [REL, rel, None]]
    vals = [["node", "demand_node", "inflow", inflow],
            ["reserve__upDown__group", grp, "reserve_method", b64("dynamic_only", "str")],
            ["reserve__upDown__group", grp, "increase_reserve_ratio", b64(0.1, "float")],
            ["reserve__upDown__group", grp, "penalty_reserve", b64(1000.0, "float")],
            [REL, rel, "is_enabled", b64("yes", "str")],
            [REL, rel, "max_share", b64(0.1, "float")]]
    url, scen = make_db(tmp_path_factory, alt="gd_stoch", values=vals, base_chain=["base", "init", "system", "2day"],
                        entities=ents, fixture_json=FLEXTOOL_ROOT / "tests" / "fixtures" / "stochastics.json")
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    rhs = _rhs(step.flex_data)
    by_branch: dict[str, list[float]] = collections.defaultdict(list)
    for (d, t), lb in rhs.items():
        branch = d.split("_", 1)[1] if "_" in d else None
        if branch is None:
            assert lb == pytest.approx(0.0), (d, t)  # the anchor period carries no demand
            continue
        i = steps_t.index(t)
        assert lb == pytest.approx(0.1 * -(bases[branch] - 10.0 * i), rel=1e-12), (d, t)
        by_branch[branch].append(lb)
    assert set(by_branch) == set(bases)
    assert len({round(v[0], 6) for v in by_branch.values()}) == len(bases)


@pytest.mark.parametrize("method", ["inflow_multiplier", "inflow_adder"])
def test_energy_margin_not_in_group_demand(tmp_path_factory, method: str) -> None:
    """Invest solve with an energy margin: the node balance sees the margined demand, the dynamic reserve
    requirement only the real demand.  inflow_multiplier 1.1 on west; inflow_adder 50 on a group node
    ``spare`` that has no inflow (it creates demand where there was none, and no requirement)."""
    ents, vals = _dz()
    if method == "inflow_multiplier":
        vals += [["node", "west", "energy_margin_method", b64("inflow_multiplier", "str")],
                 ["node", "west", "energy_margin_multiplier", b64(1.1, "float")]]
    else:
        ents += [["node", "spare", None], ["group__node", ["electricity", "spare"], None]]
        vals += [["node", "spare", "node_type", b64("balance", "str")],
                 ["node", "spare", "energy_margin_method", b64("inflow_adder", "str")],
                 ["node", "spare", "energy_margin_adder", b64(50.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt=f"gd_em_{method}", entities=ents, values=vals,
                        base_chain=["init", "west", "wind", "battery", "battery_invest", "5weeks", "coal"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    inflow = {(n, d, t): v for n, d, t, v in utf8(fd.p_inflow.frame).select("n", "d", "t", "value").iter_rows()}
    dur = {(d, t): v for d, t, v in utf8(fd.p_step_duration.frame).select("d", "t", "value").iter_rows()}
    real = {(n, d, t): v for n, d, t, v in utf8(fd.p_inflow_unmargined.frame)
            .select("n", "d", "t", "value").iter_rows()}
    rhs = _rhs(fd)
    assert rhs
    margin_seen = False
    for (d, t), lb in rhs.items():
        west_real = real[("west", d, t)]
        assert lb == pytest.approx(0.1 * max(-west_real, 0.0) / dur[(d, t)], rel=1e-9), (d, t)
        if method == "inflow_multiplier":
            assert inflow[("west", d, t)] == pytest.approx(1.1 * west_real, rel=1e-9)
            margin_seen |= west_real < 0.0
        else:
            assert inflow[("spare", d, t)] < 0.0
            margin_seen = True
    assert margin_seen
    # The node balance reads the margined inflow.
    m = build(fd).canonicalise()
    node = "west" if method == "inflow_multiplier" else "spare"
    for i, nm in enumerate(m.row_names):
        if nm.startswith(f"nodeBalance_eq[{node},"):
            d, t = nm.rstrip("]").split(",")[-2:]
            assert float(m.row_lb[i]) == pytest.approx(-inflow[(node, d, t)], rel=1e-9), nm


def test_ratio_without_dynamic_method_warns(tmp_path_factory, caplog) -> None:
    """A group ratio on a timeseries_only group is ignored, with one warning naming the group."""
    ents, vals = _dz(method="timeseries_only")
    vals.append(["reserve__upDown__group", GRP, "reservation", b64(10.0, "float")])
    with caplog.at_level(logging.WARNING):
        step = solve(tmp_path_factory, alt="gd_warn", base_chain=PLANT_CHAIN, entities=ents, values=vals)
    msgs = [r.getMessage() for r in caplog.records if "increase_reserve_ratio of reserve group" in r.getMessage()]
    assert len(msgs) == 1 and "dz__up__electricity" in msgs[0], msgs
    assert not _rows(build(step.flex_data))
