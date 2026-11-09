"""A solve's ``new_stepduration`` combines every time series by its parameter's kind (C-13), end to end.

The solve ``y2020_2day_dispatch`` (48 hourly steps) with ``new_stepduration`` = 2 h has 24 steps t0001, t0003, ...;
each combines two hours of equal length.  Every time map below alternates a (odd hours) / b (even hours), so each
2 h step takes, by the kind of its parameter (``_solve_step_aggregation.TIME_AGGREGATION``):

    MEAN      (a + b)/2
    HARMONIC  1/η = round((1/a + 1/b)/2, 6)       (the step's fuel per output; η = its reciprocal)
    SUM       a + b
    LAST      b                                   (the step ends at its second hour)

Each registry key is checked on the FlexData field its reader fills (one value on all 24 steps), and the LP of the
maps equals the LP of the equivalent constants in a single, a rolling (24 h / 12 h) and a stochastic solve.  The
group-level ``new_stepduration`` path (fine parameters, averaged rows) gives the same objective as the solve-level
path where a single parameter varies.  Values worked out by hand in each docstring; databases built from JSON.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.input._solve_step_aggregation import TIME_AGGREGATION
from tests.engine._unit_db import RunCache, b64, build, last_stochastic, utf8

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCHASTICS_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
T48 = [f"t{i:04d}" for i in range(1, 49)]
STEPS = T48[0::2]
SOLVE = "y2020_2day_dispatch"
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def S(v: float) -> list:
    return b64(float(v), "float")


def tm(a: float, b: float, steps: list[str] = T48) -> list:
    """Time map: a in the odd hours (t0001, t0003, ...), b in the even hours."""
    return b64({"index_type": "str", "index_name": "time",
                "data": [[t, (a if i % 2 == 0 else b)] for i, t in enumerate(steps)]}, "map")


def ptm(period: str, a: float, b: float) -> list:
    """Period + time map of :func:`tm` under ``period``."""
    return b64({"index_type": "str", "index_name": "period", "data": [[period, {
        "type": "map", "index_type": "str", "index_name": "time",
        "data": [[t, (a if i % 2 == 0 else b)] for i, t in enumerate(T48)]}]]}, "map")


def _chain(fixture: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


STEP2 = [["solve", SOLVE, "new_stepduration", S(2.0)]]
ROLLING = [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
           ["solve", SOLVE, "rolling_solve_horizon", S(24.0)],
           ["solve", SOLVE, "rolling_solve_jump", S(12.0)]]

# Vehicle A: coal_wind_inertia (west, coal_plant from coal_market, wind_plant, group electricity, flowGroup
# to_west_node).  (class, entity, parameter, map, the equivalent constant)
CHAIN_A = _chain(TESTS_JSON, "coal_wind_inertia")
SERIES_A = [
    ("group", "electricity", "inertia_limit", tm(100.0, 300.0), 200.0),
    ("group", "electricity", "non_synchronous_limit", tm(0.2, 0.6), 0.4),
    ("group", "electricity", "co2_price", tm(10.0, 30.0), 20.0),
    ("commodity", "coal", "price", tm(20.0, 40.0), 30.0),
    ("unit", "coal_plant", "availability", tm(0.9, 0.7), 0.8),
    ("unit", "coal_plant", "efficiency", tm(0.4, 0.5), 1.0 / 2.25),
    ("unit__outputNode", ["coal_plant", "west"], "other_operational_cost", tm(1.0, 3.0), 2.0),
    ("node", "west", "penalty_up", tm(900.0, 1100.0), 1000.0),
    ("node", "west", "penalty_down", tm(800.0, 1000.0), 900.0),
    ("flowGroup", "to_west_node", "max_instant_flow", tm(5000.0, 7000.0), 6000.0),
    ("flowGroup", "to_west_node", "min_instant_flow", tm(10.0, 30.0), 20.0),
    ("profile", "wind_profile", "profile", ptm("p2020", 0.2, 0.4), 0.3),
]
FLAGS_A = [["group", "electricity", "has_non_synchronous", b64("yes", "str")],
           ["group", "electricity", "penalty_non_synchronous", S(4000.0)],
           ["group", "electricity", "co2_method", b64("price", "str")],
           ["unit__outputNode", ["wind_plant", "west"], "is_non_synchronous", b64("yes", "str")]]

# Vehicle B: init / west / coal / wind / network / coal_min_load / reserve / battery.  coal_plant has
# min_load_efficiency with efficiency 0.4 and (here) a constant min_load 0.3.
CHAIN_B = ["init", "west", "coal", "wind", "network", "coal_min_load", "reserve", "battery"]
SERIES_B = [
    ("node", "battery", "availability", tm(0.9, 0.7), 0.8),
    ("node", "battery", "self_discharge_loss", tm(0.01, 0.03), 0.02),
    ("node", "battery", "storage_state_reference_value", tm(0.5, 0.7), 0.7),
    ("reserve__upDown__group", ["primary", "up", "electricity"], "reservation", tm(10.0, 30.0), 20.0),
    ("unit", "coal_plant", "efficiency_at_min_load", tm(0.3, 0.36), 1.0 / 3.055556),
    ("connection", "west_east", "availability", tm(0.9, 0.7), 0.8),
    ("connection", "west_east", "efficiency", tm(0.95, 0.85), 1.0 / 1.114551),
    ("connection", "west_east", "other_operational_cost", tm(1.0, 3.0), 2.0),
    ("unit__inputNode", ["coal_plant", "coal_market"], "other_operational_cost", tm(2.0, 4.0), 3.0),
]
FLAGS_B = [["unit", "coal_plant", "min_load", S(0.3)]]


def _values(series: list, *, constant: bool) -> list:
    return [[c, e, p, S(k) if constant else m] for c, e, p, m, k in series]


def _run(tmp_path_factory, alt: str, values: list, chain: list[str], fixture: Path = TESTS_JSON,
         entities: list | None = None) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=fixture,
                            entities=entities)
    assert steps and all(s.solution is not None for s in steps.values())
    return steps


def _vehicle(tmp_path_factory, which: str, *, constant: bool = False, extra: list = ()) -> dict:
    series, flags, chain = (SERIES_A, FLAGS_A, CHAIN_A) if which == "A" else (SERIES_B, FLAGS_B, CHAIN_B)
    tag = f"c13_{which}_{'const' if constant else 'map'}_{len(extra)}"
    return _run(tmp_path_factory, tag, STEP2 + list(extra) + flags + _values(series, constant=constant), chain)


def _cells(fd, field: str, keys: dict[str, str]) -> dict[str, float]:
    """``t -> value`` of FlexData ``field`` on the solve's (d, t), for the entity ``keys`` (dim -> name)."""
    param = getattr(fd, field)
    frame = utf8(promote_param_to_dt(param, fd.dt).collect())
    for dim, name in keys.items():
        frame = frame.filter(pl.col(dim) == name)
    assert not frame.select("d", "t").is_duplicated().any(), field
    return {t: float(v) for t, v in frame.select("t", "value").iter_rows()}


# key -> (vehicle, FlexData field, entity keys, the value on every 2 h step)
FIELDS: dict[tuple[str, str], tuple[str, str, dict, float]] = {
    # (100 + 300)/2
    ("group", "inertia_limit"): ("A", "pdtGroup_inertia_limit", {"g": "electricity"}, 200.0),
    # (0.2 + 0.6)/2
    ("group", "non_synchronous_limit"): ("A", "pdtGroup_non_synchronous_limit", {"g": "electricity"}, 0.4),
    # (10 + 30)/2
    ("group", "co2_price"): ("A", "p_co2_price", {"g": "electricity"}, 20.0),
    # (20 + 40)/2
    ("commodity", "price"): ("A", "p_commodity_price", {"c": "coal"}, 30.0),
    # (0.9 + 0.7)/2
    ("unit", "availability"): ("A", "p_process_availability", {"p": "coal_plant"}, 0.8),
    # 1/η = (1/0.4 + 1/0.5)/2 = (2.5 + 2.0)/2 = 2.25 (not 1/0.45 = 2.2̅)
    ("unit", "efficiency"): ("A", "p_slope", {"p": "coal_plant"}, 2.25),
    # (1 + 3)/2
    ("unit__outputNode", "other_operational_cost"): ("A", "p_pdt_varCost_sink", {"p": "coal_plant"}, 2.0),
    # (900 + 1100)/2
    ("node", "penalty_up"): ("A", "p_penalty_up", {"n": "west"}, 1000.0),
    # (800 + 1000)/2
    ("node", "penalty_down"): ("A", "p_penalty_down", {"n": "west"}, 900.0),
    # (5000 + 7000)/2
    ("flowGroup", "max_instant_flow"): ("A", "pdt_max_instant_flow", {"g": "to_west_node"}, 6000.0),
    # (10 + 30)/2
    ("flowGroup", "min_instant_flow"): ("A", "pdt_min_instant_flow", {"g": "to_west_node"}, 20.0),
    # period + time profile (0.2 + 0.4)/2
    ("profile", "profile"): ("A", "p_profile_value", {"f": "wind_profile"}, 0.3),
    # (0.9 + 0.7)/2
    ("node", "availability"): ("B", "p_node_availability", {"n": "battery"}, 0.8),
    # (0.01 + 0.03)/2
    ("node", "self_discharge_loss"): ("B", "p_state_self_discharge", {"n": "battery"}, 0.02),
    # LAST: the second hour's 0.7
    ("node", "storage_state_reference_value"): ("B", "p_storage_state_reference_value", {"n": "battery"}, 0.7),
    # (10 + 30)/2
    ("reserve__upDown__group", "reservation"): ("B", "pdtReserve_upDown_group_reservation",
                                                {"r": "primary", "ud": "up", "g": "electricity"}, 20.0),
    # 1/η_min = round((1/0.3 + 1/0.36)/2, 6) = round((3.3333333 + 2.7777778)/2, 6) = 3.055556; the min-load section
    # with efficiency 0.4 and min_load 0.3: 0.3 · (3.055556 − 2.5) / 0.7 = 0.2380954…
    ("unit", "efficiency_at_min_load"): ("B", "p_section", {"p": "coal_plant"}, 0.3 * (3.055556 - 2.5) * (1 / 0.7)),
    # (0.9 + 0.7)/2
    ("connection", "availability"): ("B", "p_process_availability", {"p": "west_east"}, 0.8),
    # 1/η = round((1/0.95 + 1/0.85)/2, 6) = round((1.0526316 + 1.1764706)/2, 6) = 1.114551
    ("connection", "efficiency"): ("B", "p_slope", {"p": "west_east"}, 1.114551),
    # (1 + 3)/2
    ("connection", "other_operational_cost"): ("B", "p_pdt_varCost_process", {"p": "west_east"}, 2.0),
    # (2 + 4)/2
    ("unit__inputNode", "other_operational_cost"): ("B", "p_pdt_varCost_source", {"p": "coal_plant"}, 3.0),
}
# The keys checked by their own tests below: node.inflow (SUM), the time-map profile (CSV path), unit.min_load,
# timeset.timeset_weights and node.energy_margin_adder.
OWN_TESTS = {("node", "inflow"), ("unit", "min_load"), ("timeset", "timeset_weights"),
             ("node", "energy_margin_adder")}


def test_every_registry_key_has_an_end_to_end_check() -> None:
    assert set(TIME_AGGREGATION) == set(FIELDS) | OWN_TESTS


@pytest.mark.parametrize("key", sorted(FIELDS), ids=lambda k: f"{k[0]}.{k[1]}")
def test_each_kind_reaches_flexdata(tmp_path_factory, key) -> None:
    which, field, keys, expected = FIELDS[key]
    fd = list(_vehicle(tmp_path_factory, which).values())[-1].flex_data
    assert utf8(fd.dt).height == 24
    cells = _cells(fd, field, keys)
    assert sorted(cells) == STEPS, field
    assert all(v == pytest.approx(expected, rel=1e-12) for v in cells.values()), (field, cells)


def test_inflow_and_time_profile_csv_path(tmp_path_factory) -> None:
    """node.inflow (SUM) and a time-map profile (MEAN) through the provider's averaged frames: west's inflow in
    tests.json is −589 / −537 at t0001 / t0002 → −1126, −506 / −482 → −988 at t0003; wind_profile 0.38 / 0.40 →
    0.39 at t0001 (vehicle B keeps the fixture's time-map profile)."""
    fd = list(_vehicle(tmp_path_factory, "B").values())[-1].flex_data
    inflow = _cells(fd, "p_inflow", {"n": "west"})
    assert (inflow["t0001"], inflow["t0003"]) == (-1126.0, -988.0)
    assert _cells(fd, "p_profile_value", {"f": "wind_profile"})["t0001"] == pytest.approx(0.39, rel=1e-12)


def test_varying_min_load_takes_the_mean_and_warns(tmp_path_factory, caplog) -> None:
    """unit.min_load 0.3 / 0.5 → (0.3 + 0.5)/2 = 0.4 per step, with the warning that the min_load_efficiency curve of
    such a step is approximate (critique M2)."""
    caplog.set_level(logging.WARNING)
    values = STEP2 + [["unit", "coal_plant", "min_load", tm(0.3, 0.5)]]
    steps = _run(tmp_path_factory, "c13_minload", values, ["init", "west", "coal", "coal_min_load"])
    fd = list(steps.values())[-1].flex_data
    cells = _cells(fd, "p_min_load", {"p": "coal_plant"})
    assert sorted(cells) == STEPS and set(cells.values()) == {0.4}
    assert any("min_load varies within a solve step" in r.message and "coal_plant" in r.message
               for r in caplog.records)


def test_timeset_weights_take_the_step_mean(tmp_path_factory) -> None:
    """timeset 2day weights: 1 / 3 alternating in t0001-t0024 (step mean 2), 1 in t0025-t0048 (step mean 1).  The
    weights are normalised per period to the number of steps: 24 steps, Σ = 12·2 + 12·1 = 36, so the steps of the
    first day weigh 2·24/36 = 4/3 and those of the second day 1·24/36 = 2/3.  (Sampling the first hour would give
    1 everywhere.)"""
    weights = b64({"index_type": "str", "index_name": "time",
                   "data": [[t, (1.0 if i >= 24 or i % 2 == 0 else 3.0)] for i, t in enumerate(T48)]}, "map")
    steps = _run(tmp_path_factory, "c13_tsw", STEP2 + [["timeset", "2day", "timeset_weights", weights]],
                 ["init", "west", "coal"])
    fd = list(steps.values())[-1].flex_data
    cells = utf8(fd.p_timestep_weight.frame)
    got = {t: float(v) for t, v in cells.select("t", "value").iter_rows()}
    assert sorted(got) == STEPS
    for t, v in got.items():
        assert v == pytest.approx(4.0 / 3.0 if int(t[1:]) < 25 else 2.0 / 3.0, rel=1e-12), t


@pytest.mark.parametrize(("label", "adder", "expected"), [
    ("period_time_map", ptm("p2020", 10.0, 30.0), -1166.0),
    ("constant", S(10.0), -1146.0),
    ("period_map", b64({"index_type": "str", "index_name": "period", "data": [["p2020", 10.0]]}, "map"), -1146.0),
])
def test_energy_margin_adder_sums_over_the_step(tmp_path_factory, label, adder, expected) -> None:
    """energy_margin_adder (MWh per original step, SUM) on west in an investment solve with 2 h steps.  West's own
    inflow at t0001 is −589 − 537 = −1126.  A period+time map 10 / 30 adds 10 + 30 = 40 MWh to each 2 h step
    (−1166); a constant 10, and a period map p2020 → 10, are the same 10 in each of the two hours: 2 · 10 = 20
    (−1146)."""
    values = STEP2 + [["solve", SOLVE, "invest_periods", b64({"value_type": "str", "data": ["p2020"],
                                                              "index_name": "period"}, "array")],
                      ["node", "west", "energy_margin_method", b64("inflow_adder", "str")],
                      ["node", "west", "energy_margin_adder", adder]]
    steps = _run(tmp_path_factory, f"c13_adder_{label}", values, ["init", "west", "coal"])
    fd = list(steps.values())[-1].flex_data
    assert _cells(fd, "p_inflow", {"n": "west"})["t0001"] == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# The LP of the maps equals the LP of the equivalent constants


def _lp(fd) -> dict:
    m = build(fd).canonicalise()
    rows: dict = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    cols: dict = {}
    for c in range(m.n_cols):
        name = m.col_names[c]
        cols[name] = (float(m.col_obj[c]), float(m.col_lb[c]), float(m.col_ub[c]))
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][name] = float(m.val[k])
    return {"rows": rows, "cols": cols}


def _assert_same_lp(steps: dict, ref: dict) -> None:
    assert list(steps) == list(ref)
    for name in ref:
        lp, lp_ref = _lp(steps[name].flex_data), _lp(ref[name].flex_data)
        assert lp["rows"].keys() == lp_ref["rows"].keys(), name
        assert lp["cols"].keys() == lp_ref["cols"].keys(), name
        for row, (lo, hi, terms) in lp_ref["rows"].items():
            got = lp["rows"][row]
            assert (got[0], got[1]) == pytest.approx((lo, hi), rel=1e-12), (name, row)
            assert got[2].keys() == terms.keys(), (name, row)
            for col, v in terms.items():
                assert got[2][col] == pytest.approx(v, rel=1e-12), (name, row, col)
        for col, v in lp_ref["cols"].items():
            assert lp["cols"][col] == pytest.approx(v, rel=1e-12), (name, col)
        assert float(steps[name].solution.obj) == pytest.approx(float(ref[name].solution.obj), rel=1e-9)


@pytest.mark.parametrize("which", ["A", "B"])
def test_single_solve_maps_equal_constants(tmp_path_factory, which) -> None:
    _assert_same_lp(_vehicle(tmp_path_factory, which), _vehicle(tmp_path_factory, which, constant=True))


@pytest.mark.parametrize("which", ["A", "B"])
def test_rolling_solve_maps_equal_constants(tmp_path_factory, which) -> None:
    """The 2 h solve rolled with a 24 h horizon and a 12 h jump: every roll's LP equals the constants' LP."""
    steps = _vehicle(tmp_path_factory, which, extra=ROLLING)
    assert len([k for k in steps if "_roll_" in k]) >= 3
    _assert_same_lp(steps, _vehicle(tmp_path_factory, which, constant=True, extra=ROLLING))


STOCH_CHAIN = _chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch")
STOCH_ENTITIES = [["group", "sys", None], ["group__node", ["sys", "demand_node"], None]]
SERIES_S = [
    ("group", "sys", "inertia_limit", ptm("period1", 100.0, 300.0), 200.0),
    ("group", "sys", "non_synchronous_limit", tm(0.2, 0.6), 0.4),
    ("commodity", "coal", "price", tm(20.0, 40.0), 30.0),
    ("unit", "coal_plant", "availability", tm(0.9, 0.7), 0.8),
    ("unit", "coal_plant", "efficiency", tm(0.4, 0.5), 1.0 / 2.25),
    ("node", "demand_node", "penalty_up", tm(900.0, 1100.0), 1000.0),
]
FLAGS_S = [["solve", "2day_dispatch", "new_stepduration", S(2.0)],
           ["group", "sys", "has_inertia", b64("yes", "str")],
           ["group", "sys", "penalty_inertia", S(5000.0)],
           ["unit__outputNode", ["coal_plant", "demand_node"], "inertia_constant", S(4.0)],
           ["group", "sys", "has_non_synchronous", b64("yes", "str")],
           ["group", "sys", "penalty_non_synchronous", S(4000.0)],
           ["unit__outputNode", ["wind_plant", "demand_node"], "is_non_synchronous", b64("yes", "str")]]


def test_stochastic_solve_maps_equal_constants(tmp_path_factory) -> None:
    """2_day_stochastic_dispatch with new_stepduration 2: the branch periods take the anchor's step means, and the
    LP of the maps equals the constants' LP."""
    steps = _run(tmp_path_factory, "c13_stoch_map", FLAGS_S + _values(SERIES_S, constant=False), STOCH_CHAIN,
                 STOCHASTICS_JSON, STOCH_ENTITIES)
    ref = _run(tmp_path_factory, "c13_stoch_const", FLAGS_S + _values(SERIES_S, constant=True), STOCH_CHAIN,
               STOCHASTICS_JSON, STOCH_ENTITIES)
    fd = last_stochastic(steps).flex_data
    assert len(set(utf8(fd.dt)["d"].to_list())) > 1, "expected the branch periods"
    _assert_same_lp(steps, ref)


def _branch_map(per_branch: dict[str, tuple[float, float]]) -> list:
    """Stochastic-time map (branch → analysis_time t0001 → time): a in the odd hours, b in the even hours."""
    return b64({"index_type": "str", "index_name": "branch", "rank": 3, "data": [[branch, {
        "type": "map", "index_type": "str", "index_name": "analysis_time", "rank": 2, "data": [["t0001", {
            "type": "map", "index_type": "str", "index_name": "time", "rank": 1,
            "data": [[t, (a if i % 2 == 0 else b)] for i, t in enumerate(T48)]}]]}]
        for branch, (a, b) in per_branch.items()]}, "map")


def test_stochastic_branch_series_combine_by_kind(tmp_path_factory) -> None:
    """2_day_stochastic_dispatch with new_stepduration 2 and stochastic-time maps (one series per branch) on the
    stochastic entities: each branch period takes its own branch's step value, by the parameter's kind.

    hydro_reservoir inflow (SUM): realized −100 / −300 → −400; upper −200 / −400 → −600; lower −50 / −70 → −120;
    mid −10 / −30 → −40.  wind_plant availability (MEAN): 0.9 / 0.7 → 0.8; 0.6 / 0.4 → 0.5; 0.5 / 0.3 → 0.4;
    1.0 / 0.8 → 0.9.  wind profile (MEAN): 0.2 / 0.4 → 0.3; 0.1 / 0.3 → 0.2; 0.5 / 0.7 → 0.6; 0.0 / 0.2 → 0.1."""
    inflow = {"realized": (-100.0, -300.0), "upper": (-200.0, -400.0), "lower": (-50.0, -70.0), "mid": (-10.0, -30.0)}
    avail = {"realized": (0.9, 0.7), "upper": (0.6, 0.4), "lower": (0.5, 0.3), "mid": (1.0, 0.8)}
    prof = {"realized": (0.2, 0.4), "upper": (0.1, 0.3), "lower": (0.5, 0.7), "mid": (0.0, 0.2)}
    values = [["solve", "2day_dispatch", "new_stepduration", S(2.0)],
              ["node", "hydro_reservoir", "inflow", _branch_map(inflow)],
              ["unit", "wind_plant", "availability", _branch_map(avail)],
              ["profile", "wind", "profile", _branch_map(prof)]]
    fd = last_stochastic(_run(tmp_path_factory, "c13_stoch_branch", values, STOCH_CHAIN, STOCHASTICS_JSON)).flex_data
    expected = {
        ("p_inflow", "n", "hydro_reservoir"): {"realized": -400.0, "upper": -600.0, "lower": -120.0, "mid": -40.0},
        ("p_process_availability", "p", "wind_plant"): {"realized": 0.8, "upper": 0.5, "lower": 0.4, "mid": 0.9},
        ("p_profile_value", "f", "wind"): {"realized": 0.3, "upper": 0.2, "lower": 0.6, "mid": 0.1},
    }
    for (field, dim, name), per_branch in expected.items():
        frame = utf8(promote_param_to_dt(getattr(fd, field), fd.dt).collect()).filter(pl.col(dim) == name)
        got = {(d, t): float(v) for d, t, v in frame.select("d", "t", "value").iter_rows()}
        for branch, want in per_branch.items():
            cells = {t: v for (d, t), v in got.items() if d == f"period1_{branch}"}
            assert sorted(cells) == STEPS, (field, branch)
            assert all(v == pytest.approx(want, rel=1e-12) for v in cells.values()), (field, branch, cells)


# ---------------------------------------------------------------------------
# Group-level new_stepduration (fine parameters, averaged rows) vs solve-level (aggregated parameters)

ORACLE_CHAIN = ["init", "west", "coal"]
ALL_NODES = [["group", "all2h", None], ["group__node", ["all2h", "west"], None],
             ["group__node", ["all2h", "coal_market"], None]]


@pytest.mark.parametrize(("label", "values"), [
    ("price", [["commodity", "coal", "price", tm(20.0, 40.0)]]),
    ("availability", [["unit", "coal_plant", "availability", tm(0.9, 0.7)]]),
    ("efficiency", [["unit", "coal_plant", "efficiency", tm(0.4, 0.5)]]),
])
def test_group_step_and_solve_step_agree(tmp_path_factory, label, values) -> None:
    """Every node of init / west / coal in a 2 h group (the group path keeps the hourly values and averages the
    rows over each coarse step: the mean price, availability and slope) gives the same objective as the solve's
    new_stepduration = 2 h (the aggregated values: price 30, availability 0.8, slope 2.25), one time-varying
    parameter at a time (with two, the mean of a product is not the product of the means)."""
    group = _run(tmp_path_factory, f"c13_grp_{label}", [["group", "all2h", "new_stepduration", S(2.0)]] + values,
                 ORACLE_CHAIN, entities=ALL_NODES)
    solve = _run(tmp_path_factory, f"c13_slv_{label}", STEP2 + values, ORACLE_CHAIN)
    assert float(list(group.values())[-1].solution.obj) == pytest.approx(
        float(list(solve.values())[-1].solution.obj), rel=1e-9)
