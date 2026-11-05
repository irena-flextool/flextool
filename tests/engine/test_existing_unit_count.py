"""With ``virtual_unitsize`` set, ``existing`` is the number of units: capacity = existing × virtual_unitsize.

The existing unit count is capacity / unitsize, so it equals ``existing`` with virtual_unitsize and 1 without it
(unitsize = existing).  These tests pin the provider-less fallbacks of ``params._derived`` (no chained
``p_entity_all_existing``: workdir None) on an in-memory source.  Unit ``u`` (a → b) has existing 2 and
virtual_unitsize 250 (two 250 MW units, 500 MW): count 2.  Its twin ``w`` has existing 500 MW and no
virtual_unitsize: unitsize 500, count 500 / 500 = 1.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine import InMemoryReader
from flextool.engine.preprocess.params import _derived
from tests.engine._lp_rows import coef, row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, COAL, PLANT, b64, build, demand, flow, make_db, run, solve, utf8

PERIOD = "p2020"


def _source(existing_shape: str) -> InMemoryReader:
    if existing_shape == "constant":
        existing = pl.DataFrame({"name": ["u", "w"], "value": [2.0, 500.0]})
    else:
        existing = pl.DataFrame({"name": ["u", "w"], "period": [PERIOD, PERIOD], "value": [2.0, 500.0]})
    return InMemoryReader(entities={
        "unit": pl.DataFrame({"name": ["u", "w"]}),
        "node": pl.DataFrame({"name": ["a", "b"]}),
        "connection": pl.DataFrame(schema={"name": pl.Utf8}),
        "solve": pl.DataFrame({"name": ["s"]}),
        "unit__inputNode": pl.DataFrame({"unit": ["u", "w"], "node": ["a", "a"]}),
        "unit__outputNode": pl.DataFrame({"unit": ["u", "w"], "node": ["b", "b"]}),
    }, parameters={
        ("unit", "existing"): existing,
        ("unit", "virtual_unitsize"): pl.DataFrame({"name": ["u"], "value": [250.0]}),
        ("solve", "realized_periods"): pl.DataFrame({"name": ["s"], "value": [PERIOD]}),
    })


def _values(frame: pl.DataFrame) -> dict[str, float]:
    assert frame["d"].cast(pl.Utf8).to_list() == [PERIOD] * frame.height
    return {str(p): float(v) for p, v in zip(frame["p"].to_list(), frame["value"].to_list())}


@pytest.mark.parametrize("existing_shape", ["constant", "period_map"])
def test_existing_count_fallback_is_existing_with_virtual_unitsize(existing_shape):
    dt = pl.DataFrame({"d": [PERIOD], "t": ["t0001"]})
    count = _derived.p_process_existing_count_from_source(_source(existing_shape), dt, None, None)
    assert count is not None
    assert _values(count.frame) == {"u": 2.0, "w": 1.0}


@pytest.mark.parametrize("existing_shape", ["constant", "period_map"])
def test_flow_upper_existing_fallback_is_existing_count_per_arc(existing_shape):
    pss = pl.DataFrame({"p": ["u", "w"], "source": ["a", "a"], "sink": ["b", "b"]})
    upper = _derived.p_flow_upper_existing_from_source(_source(existing_shape), pss, None, None)
    assert upper is not None
    assert upper.frame.select("p", "source", "sink").rows() == [("u", "a", "b"), ("w", "a", "b")]
    assert _values(upper.frame) == {"u": 2.0, "w": 1.0}


# ---------------------------------------------------------------------------
# Semantic pins on solved models.  tests.json (2day: 48 hourly steps t0001-t0048, period p2020), chain init, west,
# coal; coal_plant (coal_market -> west, efficiency 0.4, fuel 20 / 0.4 = 50 €/MWh) and a constant west demand of
# 800 MW with penalty_up 10000.  capacity = existing × virtual_unitsize, unitsize = virtual_unitsize, existing count
# = capacity / unitsize = existing; flow in MW = v_flow × unitsize, storage in MWh = v_state × state_unitsize.
# ---------------------------------------------------------------------------

STEPS = 48
_BASE = ["init", "west", "coal"]


def _pinned_values(*extra: list) -> list:
    return [demand("west", 800.0), ["node", "west", "penalty_up", b64(10000.0, "float")], *extra]


def _value(param, **keys) -> float:
    frame = utf8(param.frame)
    for col, val in keys.items():
        frame = frame.filter(pl.col(col) == val)
    assert frame.height == 1, (keys, frame)
    return float(frame["value"][0])


def _var(step, var: str, **keys) -> pl.DataFrame:
    frame = utf8(step.solution.value(var))
    for col, val in keys.items():
        frame = frame.filter(pl.col(col) == val)
    assert frame.height > 0, (var, keys)
    return frame.sort([c for c in ("d", "t") if c in frame.columns])


def _unmet(step, node: str) -> list[float]:
    return _var(step, "vq_state_up", n=node)["value"].to_list()


@pytest.mark.solver
def test_unit_capacity_is_count_times_virtual_unitsize(tmp_path_factory):
    """T1: coal_plant existing 2, virtual_unitsize 250 (2 × 250 MW) and its twin existing 500 MW without
    virtual_unitsize (unitsize 500, count 1): both have 500 MW, the LP differs only by the unitsize scaling.  At
    800 MW demand both run at 500 MW every step and leave 300 MW unserved."""
    cases = {
        "count": ([["unit", PLANT, "existing", b64(2.0, "float")],
                   ["unit", PLANT, "virtual_unitsize", b64(250.0, "float")]], 2.0, 250.0),
        "mw": ([["unit", PLANT, "existing", b64(500.0, "float")]], 1.0, 500.0),
    }
    for alt, (vals, count, size) in cases.items():
        step = solve(tmp_path_factory, alt=f"t1_{alt}", values=_pinned_values(*vals), base_chain=_BASE)
        fd = step.flex_data
        assert _value(fd.p_process_existing_count, p=PLANT) == pytest.approx(count)
        assert _value(fd.p_unitsize, p=PLANT) == pytest.approx(size)
        assert _value(fd.p_entity_all_existing, e=PLANT) == pytest.approx(500.0)
        out = _var(step, "v_flow", p=PLANT, source=COAL, sink="west")
        assert out.height == STEPS
        assert out["value"].to_list() == pytest.approx([count] * STEPS, rel=1e-9)
        assert flow(step, PLANT, COAL, "west").to_list() == pytest.approx([500.0] * STEPS, rel=1e-9)
        assert _unmet(step, "west") == pytest.approx([300.0] * STEPS, rel=1e-9)


@pytest.mark.solver
def test_storage_capacity_is_count_times_virtual_unitsize(tmp_path_factory):
    """T3: battery existing 3, virtual_unitsize 100: 300 MWh in 100 MWh units, so maxState caps v_state at 3.
    fix_start 0.5 with the fixture's bind_within_timeblock pins the wrap target, the last step of the timeblock:
    v_state(t0048) = 0.5 × 300 / 100 = 1.5 units = 150 MWh."""
    step = solve(tmp_path_factory, alt="t3_battery", base_chain=_BASE + ["battery"], values=_pinned_values(
        ["node", "battery", "existing", b64(3.0, "float")],
        ["node", "battery", "virtual_unitsize", b64(100.0, "float")],
        ["node", "battery", "storage_state_start", b64(0.5, "float")]))
    fd = step.flex_data
    assert _value(fd.p_state_existing_capacity, n="battery", d="p2020") == pytest.approx(300.0)
    su = _value(fd.p_state_unitsize, n="battery")
    assert su == pytest.approx(100.0)
    bounds = row_bounds(build(fd), "maxState", "battery,")
    assert len(bounds) == STEPS
    assert [ub for _, ub in bounds.values()] == pytest.approx([3.0] * STEPS)
    state = _var(step, "v_state", n="battery")
    assert float(state["value"].max()) * su <= 300.0 + 1e-6
    end = state.filter(pl.col("t") == "t0048")["value"][0]
    assert end * su == pytest.approx(150.0, rel=1e-9)


@pytest.mark.solver
def test_connection_capacity_is_count_times_virtual_unitsize(tmp_path_factory):
    """T4: west_east existing 2, virtual_unitsize 50 (100 MW).  coal_plant 5000 MW (no virtual_unitsize) keeps west
    supplied, north is cut off (west_north, east_north existing 0).  East (300 MW demand, penalty 500 €/MWh, no own
    supply) values each delivered MW at 500 €/MWh, more than the 50 / 0.9 €/MWh it costs at west, so the connection
    runs at its 100 MW every step.  A connection's flow is the flow into its sink (nodeBalance_eq: the source gives
    flow / efficiency), so east gets 100 MW and 300 − 100 = 200 MW stay unserved."""
    step = solve(tmp_path_factory, alt="t4_network", base_chain=_BASE + ["network"], values=_pinned_values(
        ["unit", PLANT, "existing", b64(5000.0, "float")],
        ["connection", "west_east", "existing", b64(2.0, "float")],
        ["connection", "west_east", "virtual_unitsize", b64(50.0, "float")],
        ["connection", "west_north", "existing", b64(0.0, "float")],
        ["connection", "east_north", "existing", b64(0.0, "float")]))
    fd = step.flex_data
    assert _value(fd.p_process_existing_count, p="west_east") == pytest.approx(2.0)
    assert _value(fd.p_unitsize, p="west_east") == pytest.approx(50.0)
    rows = row_bounds(build(fd), "maxFlow", "west_east,")
    arcs = {tuple(nm[nm.index("[") + 1:].split(",")[1:3]) for nm in rows}
    assert arcs == {("west", "east"), ("east", "west")}
    assert [ub for _, ub in rows.values()] == pytest.approx([2.0] * len(rows))
    assert flow(step, "west_east", "west", "east").to_list() == pytest.approx([100.0] * STEPS, rel=1e-9)
    assert flow(step, "west_east", "east", "west").to_list() == pytest.approx([0.0] * STEPS, abs=1e-9)
    assert _unmet(step, "east") == pytest.approx([200.0] * STEPS, rel=1e-9)


@pytest.mark.solver
@pytest.mark.parametrize(("west_mw", "units", "out_mw", "unmet_mw"),
                         [(800.0, 3.0, 600.0, 200.0), (450.0, 3.0, 450.0, 0.0), (100.0, 1.0, 100.0, 0.0)])
def test_binary_online_commits_whole_units_of_virtual_unitsize(tmp_path_factory, west_mw, units, out_mw, unmet_mw):
    """T5: coal_plant binary online, existing 3, virtual_unitsize 200 (three 200 MW units), min_load 0.4: at most 3
    units online, each between 80 and 200 MW.  800 MW: 3 units at 600 MW, 200 MW unserved.  450 MW: 200·on ≥ 450
    needs on = 3 (floor 240 ≤ 450).  100 MW: on = 1 (on = 2 would force 160 MW, dumped at penalty_down 800)."""
    step = solve(tmp_path_factory, alt=f"t5_binary_{int(west_mw)}", base_chain=_BASE, values=[
        demand("west", west_mw), ["node", "west", "penalty_up", b64(10000.0, "float")],
        ["unit", PLANT, "existing", b64(3.0, "float")],
        ["unit", PLANT, "virtual_unitsize", b64(200.0, "float")],
        ["unit", PLANT, "startup_method", b64("binary", "str")],
        ["unit", PLANT, "min_load", b64(0.4, "float")]])
    fd = step.flex_data
    pb = build(fd)
    online_ub = row_bounds(pb, "maxOnline_integer", f"{PLANT},")
    assert len(online_ub) == STEPS
    assert [ub for _, ub in online_ub.values()] == pytest.approx([3.0] * STEPS)
    floors = row_terms(pb, "minFlow_minload_integer", f"{PLANT},")
    assert len(floors) == STEPS
    for terms in floors.values():
        # v_flow (units of 200 MW) ≥ 0.4 · v_online: 80 MW per unit online.
        ratio = coef(terms, "v_online_integer[") / coef(terms, "v_flow[")
        assert ratio == pytest.approx(-0.4, rel=1e-12)
    on = _var(step, "v_online_integer", p=PLANT)["value"].to_list()
    assert on == pytest.approx([units] * STEPS, abs=1e-6)
    assert flow(step, PLANT, COAL, "west").to_list() == pytest.approx([out_mw] * STEPS, rel=1e-9, abs=1e-6)
    assert _unmet(step, "west") == pytest.approx([unmet_mw] * STEPS, abs=1e-6)


@pytest.mark.solver
@pytest.mark.parametrize(("invest_max_mw", "v_invest"), [(500.0, 2.0), (400.0, 1.6)])
def test_investment_limit_is_mw_and_units_are_continuous(tmp_path_factory, invest_max_mw, v_invest):
    """T6: coal_plant existing 1, virtual_unitsize 250 (250 MW), invest_total with invest_max_total in MW and a
    cheap invest_cost.  maxInvest_entity_total: 250·v_invest ≤ invest_max_total, so v_invest ≤ 2 (500 MW) or ≤ 1.6
    (400 MW, not a whole number).  maxInvest_var_bound caps v_invest at (existing capacity + invest_max_total) /
    unitsize = 3 (2.6): the existing capacity is in it.  Unserved load is worth far more than the annuity, so the
    model builds to the limit: 250 + 250·v_invest MW (750 or 650) runs every step, the rest of 800 MW is unmet."""
    step = solve(tmp_path_factory, alt=f"t6_invest_{int(invest_max_mw)}", base_chain=_BASE, values=_pinned_values(
        ["unit", PLANT, "existing", b64(1.0, "float")],
        ["unit", PLANT, "virtual_unitsize", b64(250.0, "float")],
        ["unit", PLANT, "invest_method", b64("invest_total", "str")],
        ["unit", PLANT, "invest_max_total", b64(invest_max_mw, "float")],
        ["unit", PLANT, "invest_cost", b64(1.0, "float")],
        ["unit", PLANT, "lifetime", b64(30.0, "float")],
        ["unit", PLANT, "discount_rate", b64(0.05, "float")],
        ["solve", "y2020_2day_dispatch", "invest_periods",
         b64({"type": "array", "value_type": "str", "data": ["p2020"]}, "array")],
        ["solve", "y2020_2day_dispatch", "realized_invest_periods",
         b64({"type": "array", "value_type": "str", "data": ["p2020"]}, "array")]))
    fd = step.flex_data
    assert _value(fd.p_process_existing_count, p=PLANT) == pytest.approx(1.0)
    pb = build(fd)
    var_bound = row_bounds(pb, "maxInvest_var_bound", f"{PLANT},")
    assert [ub for _, ub in var_bound.values()] == [pytest.approx((250.0 + invest_max_mw) / 250.0)]
    total_terms = row_terms(pb, "maxInvest_entity_total", f"{PLANT}")
    total_bounds = row_bounds(pb, "maxInvest_entity_total", f"{PLANT}")
    assert len(total_terms) == 1
    (name, terms), = total_terms.items()
    assert total_bounds[name][1] / coef(terms, "v_invest_p[") == pytest.approx(invest_max_mw / 250.0, rel=1e-12)
    inv = _var(step, "v_invest_p", p=PLANT)["value"].to_list()
    assert inv == pytest.approx([v_invest], rel=1e-9)
    built = 250.0 + 250.0 * v_invest
    assert flow(step, PLANT, COAL, "west").to_list() == pytest.approx([built] * STEPS, rel=1e-9)
    assert _unmet(step, "west") == pytest.approx([800.0 - built] * STEPS, rel=1e-9)


_FIXTURES = FLEXTOOL_ROOT / "tests" / "fixtures"
_ROLLING = (_FIXTURES / "rolling_separate_dispatch.json", ["init", "base", "system", "1week_rolling"])
_STOCH = (_FIXTURES / "stoch_t0_commit.json", ["init", "base", "stoch"])


def _fixture_steps(tmp_path_factory, alt: str, fixture: tuple, values: list) -> dict:
    fixture_json, chain = fixture
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=fixture_json)
    return run(tmp_path_factory, url, scen)


@pytest.mark.solver
def test_rolling_dispatch_chain_keeps_unit_counts(tmp_path_factory):
    """T7: rolling_separate_dispatch has coal_plant existing 3 × 600 MW and gas_plant 2 × 250 MW (binary online, no
    investment).  The capacity handed from roll to roll and to the realized dispatch is 1800 / 500 MW, so every
    roll and the dispatch solve keep the counts 3 and 2 and cap the units online at them."""
    steps = _fixture_steps(tmp_path_factory, "t7_rolling", _ROLLING, [])
    assert len(steps) > 1
    for name, step in steps.items():
        fd = step.flex_data
        pb = build(fd)
        for unit, count in (("coal_plant", 3.0), ("gas_plant", 2.0)):
            counts = utf8(fd.p_process_existing_count.frame).filter(pl.col("p") == unit)["value"].to_list()
            assert counts and counts == pytest.approx([count] * len(counts)), (name, unit)
            ub = [u for _, u in row_bounds(pb, "maxOnline_integer", f"{unit},").values()]
            assert ub and ub == pytest.approx([count] * len(ub)), (name, unit)


def _canonical_lp(step) -> tuple:
    """The step's LP independent of row and column order: rows by name (terms by column name, bounds) and columns
    by name (bounds, objective, integrality)."""
    m = build(step.flex_data).canonicalise()
    terms: list[dict[str, float]] = [{} for _ in range(m.n_rows)]
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            terms[int(m.row_idx[k])][m.col_names[c]] = float(m.val[k])
    rows = {nm: (terms[i], float(m.row_lb[i]), float(m.row_ub[i])) for i, nm in enumerate(m.row_names)}
    cols = {nm: (float(m.col_lb[i]), float(m.col_ub[i]), float(m.col_obj[i]), bool(m.col_int[i]))
            for i, nm in enumerate(m.col_names)}
    return rows, cols


def _period_map(period: str, value: float) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[period, value]]}, "map")


_SHAPE_CASES = {
    "single": (None, [(PLANT, "p2020", 2.0)], [["unit", PLANT, "virtual_unitsize", b64(250.0, "float")]]),
    "rolling": (_ROLLING, [("coal_plant", "period1", 3.0), ("gas_plant", "period1", 2.0)], []),
    "stochastic": (_STOCH, [("peaker", "p2035", 2.0)], []),
}


@pytest.mark.solver
@pytest.mark.parametrize("solve_kind", list(_SHAPE_CASES))
def test_existing_count_constant_and_period_map_give_the_same_lp(tmp_path_factory, solve_kind):
    """T8: with virtual_unitsize, existing as a constant and as the equivalent period map give the same count in
    every period, so every step's LP is the same in a single, a rolling and a stochastic solve."""
    fixture, units, extra = _SHAPE_CASES[solve_kind]
    lps = {}
    for shape in ("constant", "map"):
        values = list(extra) + [
            ["unit", unit, "existing", b64(v, "float") if shape == "constant" else _period_map(period, v)]
            for unit, period, v in units]
        alt = f"t8_{solve_kind}_{shape}"
        if fixture is None:
            url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_BASE)
            steps = run(tmp_path_factory, url, scen)
        else:
            steps = _fixture_steps(tmp_path_factory, alt, fixture, values)
        for unit, period, v in units:
            count = utf8(list(steps.values())[0].flex_data.p_process_existing_count.frame)
            assert count.filter((pl.col("p") == unit) & (pl.col("d") == period))["value"].to_list() == [v]
        lps[shape] = {name: _canonical_lp(step) for name, step in steps.items()}
    assert list(lps["constant"]) == list(lps["map"])
    for name in lps["constant"]:
        assert lps["constant"][name] == lps["map"][name], name
