"""Rolling and nested sub-solves use period-map values of efficiency and profiles (#61).

A roll of a ``rolling_window`` solve and a per-period sub-solve of a nested investment are windows of the parent
solve, so a value given as a period map ``{d: v}`` must reach them exactly as it reaches a single solve:

    value_roll[k, d, t] = value_single[k, d, t]      for every (d, t) of the roll

These sub-solves skip the DB-direct ``apply_derived_*`` passes and keep the CSV seeds of the per-solve emit.
Before the fix:
- the emit ``pdtProcess`` cascade had no period tier for ``efficiency`` / ``efficiency_at_min_load`` /
  ``min_load`` (``pt_process`` drops period-labelled maps and ``pd_process`` did not list them), so
  ``p_slope``, ``p_flow_upper`` and ``p_indirect_input_cap`` used efficiency 1 and no min-load section;
- the emit ``pdtProfile`` had no period tier, so a period-keyed profile (1-D period or 2-D period x time) read
  as 0.

The criterion: inside one solve mode, a period map gives the same parameters and objective as the equivalent
scalar (or time curve), and a roll's parameters equal the matching slice of the single solve.

A silent-``x`` **time** map of one of these parameters must not leak its timestep names into ``pd_process`` or
the period universe (``periodAll``).

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``1_week_rolling_wind`` of ``tests/fixtures/stochastics.json`` with its stochastic data removed: one period
  ``period1`` (168 h), rolls of 96 h jumping 24 h, so 7 synthetic rolls ``1week_rolling_roll_<n>``.
- ``invest_5weeks`` of ``tests/fixtures/tests.json`` with nested ``invest_periods``: the synthetic sub-solves
  ``invest_5weeks_<p>``.
- ``network_coal_wind_battery_co2_fullYear_availability`` of ``tests/fixtures/tests.json`` (x-labelled time
  maps of unit / connection ``availability``).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, is_realization, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
STOCHASTICS_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"

ROLL_CHAIN = ["init", "base", "system", "1week_rolling"]
PERIOD = "period1"
T168 = [f"t{i:04d}" for i in range(1, 169)]
CURVE = {t: round(0.2 + 0.001 * i, 6) for i, t in enumerate(T168)}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Value / database helpers


def scalar(v: float) -> list:
    return b64(float(v), "float")


def pmap(values: dict, index_name: str | None = "period") -> list:
    """1-D map (``index_name=None``: Spine's silent ``x`` index)."""
    m = {"type": "map", "index_type": "str", "data": [[k, v] for k, v in values.items()]}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


def tmap(tv: dict) -> list:
    return pmap(tv, index_name="time")


def ptmap(per: dict) -> list:
    """2-D map period -> time -> value."""
    return b64({"type": "map", "index_type": "str", "index_name": "period", "data": [
        [d, {"type": "map", "index_type": "str", "index_name": "time",
             "data": [[t, v] for t, v in tv.items()]}] for d, tv in per.items()]}, "map")


def _deterministic_rolling_json(tmp_path_factory, alt: str) -> Path:
    """``stochastics.json`` without its stochastic data, plus the connection ``link`` (activated in ``alt``)."""
    data = json.loads(STOCHASTICS_JSON.read_text())
    data["parameter_values"] = [
        pv for pv in data["parameter_values"]
        if pv[2] not in ("stochastic_branches", "include_stochastics")
        and not (pv[0] == "profile" and pv[1] == "wind" and pv[4] in ("1week_rolling", "2day"))]
    data["entities"] += [["connection", "link", None],
                         ["connection__node__node", ["link", "gas_market", "coal_market"], None]]
    data["entity_alternatives"].append(["connection", ["link"], alt, True])
    path = tmp_path_factory.mktemp(f"{alt}_json") / "stochastics_det.json"
    path.write_text(json.dumps(data))
    return path


_RUNS: dict = {}


def _roll(tmp_path_factory, values: list, *, single: bool = False) -> dict:
    key = json.dumps([values, single])
    if key not in _RUNS:
        alt = f"c8_{len(_RUNS)}"
        values = list(values)
        if single:
            values.append(["solve", "1week_rolling", "solve_mode", b64("single_solve", "str")])
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=ROLL_CHAIN,
                            fixture_json=_deterministic_rolling_json(tmp_path_factory, alt))
        _RUNS[key] = run(tmp_path_factory, url, scen)
    return _RUNS[key]


def _frame(x) -> pl.DataFrame:
    assert x is not None, "parameter is None"
    f = getattr(x, "frame", x)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    return utf8(f)


def _canon(x) -> pl.DataFrame:
    f = _frame(x)
    return f.sort(f.columns)


def _value(x, **keys) -> set[float]:
    f = _frame(x)
    for col, val in keys.items():
        f = f.filter(pl.col(col) == val)
    return {round(float(v), 9) for v in f["value"]}


# ---------------------------------------------------------------------------
# Rolling vehicle: efficiency family and profile, period map vs scalar


def _values(as_map: bool) -> list:
    def v(x):
        return pmap({PERIOD: x}) if as_map else scalar(x)

    return [
        ["unit", "coal_plant", "efficiency", v(0.5)],                      # direct unit
        ["unit", "hydro_plant", "efficiency", v(1.5)],                     # two outputs: indirect
        ["unit", "gas_plant", "conversion_method", b64("min_load_efficiency", "str")],
        ["unit", "gas_plant", "startup_method", b64("linear", "str")],
        ["unit", "gas_plant", "min_load", v(0.3)],
        ["unit", "gas_plant", "efficiency_at_min_load", scalar(0.6)],      # schema: constant only
        ["connection", "link", "existing", scalar(100.0)],
        ["connection", "link", "transfer_method", b64("regular", "str")],
        ["connection", "link", "efficiency", v(0.7)],
        ["profile", "wind", "profile", v(0.35)],
    ]


@pytest.fixture(scope="module")
def roll_map(tmp_path_factory):
    return _roll(tmp_path_factory, _values(as_map=True))


@pytest.fixture(scope="module")
def roll_scalar(tmp_path_factory):
    return _roll(tmp_path_factory, _values(as_map=False))


@pytest.fixture(scope="module")
def single_map(tmp_path_factory):
    return _roll(tmp_path_factory, _values(as_map=True), single=True)


def _assert_rolls(steps: dict) -> None:
    assert len(steps) > 1
    assert all("_roll_" in name for name in steps), list(steps)


def test_efficiency_maps_equal_scalars_in_rolls(roll_map, roll_scalar) -> None:
    _assert_rolls(roll_map)
    assert list(roll_map) == list(roll_scalar)
    for name, s_map in roll_map.items():
        s_scalar = roll_scalar[name]
        dm, ds = s_map.flex_data, s_scalar.flex_data
        # Unrounded (#140), compared at 9 decimals: slope = 1/efficiency; gas_plant (efficiency 0.9, min_load 0.3
        # at efficiency 0.6) slope = (1/0.9 - 0.3/0.6) / (1 - 0.3).
        assert _value(dm.p_slope, p="coal_plant") == {2.0}, name
        assert _value(dm.p_slope, p="hydro_plant") == {round(1 / 1.5, 9)}, name
        assert _value(dm.p_slope, p="gas_plant") == {round((1 / 0.9 - 0.3 / 0.6) / 0.7, 9)}, name
        assert _value(dm.p_slope, p="link") == {round(1 / 0.7, 9)}, name
        assert _value(dm.p_flow_upper, p="hydro_plant", source="hydro_reservoir", sink="hydro_plant") == {
            round(1 / 1.5, 9)}
        for field in ("p_slope", "p_section", "p_flow_upper", "p_indirect_input_cap"):
            assert _canon(getattr(dm, field)).equals(_canon(getattr(ds, field))), (name, field)
        assert s_map.solution.obj == pytest.approx(s_scalar.solution.obj, rel=1e-9), name


def test_profile_period_map_equals_scalar_in_rolls(roll_map, roll_scalar) -> None:
    for name, s_map in roll_map.items():
        assert _value(s_map.flex_data.p_profile_value, f="wind") == {0.35}, name
        assert _canon(s_map.flex_data.p_profile_value).equals(
            _canon(roll_scalar[name].flex_data.p_profile_value)), name


def test_profile_period_curve_equals_time_curve_in_rolls(tmp_path_factory) -> None:
    curve = _roll(tmp_path_factory, [["profile", "wind", "profile", ptmap({PERIOD: CURVE})]])
    ref = _roll(tmp_path_factory, [["profile", "wind", "profile", tmap(CURVE)]])
    _assert_rolls(curve)
    assert list(curve) == list(ref)
    for name, s_curve in curve.items():
        assert _frame(s_curve.flex_data.p_profile_value).height > 0, name
        assert _canon(s_curve.flex_data.p_profile_value).equals(
            _canon(ref[name].flex_data.p_profile_value)), name
        assert s_curve.solution.obj == pytest.approx(ref[name].solution.obj, rel=1e-9), name


def _slice(single, roll, field: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``field`` of the single solve restricted to the roll's (d, t), and the roll's ``field``; both sorted."""
    r = _canon(getattr(roll, field))
    s = _canon(getattr(single, field)).join(r.select("d", "t").unique(), on=["d", "t"], how="semi")
    return s.sort(s.columns), r


def test_rolls_match_single_solve_parameters(roll_map, single_map) -> None:
    assert len(single_map) == 1
    single = next(iter(single_map.values())).flex_data
    for name, step in roll_map.items():
        s, r = _slice(single, step.flex_data, "p_profile_value")
        assert r.height > 0 and s.equals(r), name
        for field in ("p_slope", "p_flow_upper"):
            s, r = _slice(single, step.flex_data, field)
            keys = [c for c in r.columns if c != "value"]
            assert s.select(keys).equals(r.select(keys)), (name, field)
            assert s["value"].to_list() == pytest.approx(r["value"].to_list(), rel=1e-12), (name, field)


# ---------------------------------------------------------------------------
# Nested investment: the per-period sub-solves ``invest_5weeks_<p>``

NESTED_CHAIN = ["init", "west", "coal", "coal_invest", "wind", "battery", "5weeks", "invest_solveSequence_5weeks",
                "invest_allYears_5weeks", "invest_oneYear4Times_5weeks", "invest_twoYears4Times_5weeks"]
NESTED_PERIODS = ["p2020", "p2025", "p2030", "p2035"]


def _nested(tmp_path_factory, as_map: bool) -> dict:
    def v(x):
        return pmap({p: x for p in NESTED_PERIODS}) if as_map else scalar(x)

    values = [["unit", "coal_plant", "efficiency", v(0.5)],
              ["unit", "coal_plant", "conversion_method", b64("min_load_efficiency", "str")],
              ["unit", "coal_plant", "startup_method", b64("linear", "str")],
              ["unit", "coal_plant", "min_load", v(0.3)],
              ["unit", "coal_plant", "efficiency_at_min_load", scalar(0.4)],
              ["profile", "wind_profile", "profile", v(0.35)],
              ["solve", "invest_5weeks", "contains_solves",
               b64({"type": "array", "value_type": "str", "data": []}, "array")]]
    key = json.dumps(["nested", values])
    if key not in _RUNS:
        url, scen = make_db(tmp_path_factory, alt=f"c8_{len(_RUNS)}", values=values, base_chain=NESTED_CHAIN,
                            fixture_json=TESTS_JSON)
        _RUNS[key] = run(tmp_path_factory, url, scen)
    return _RUNS[key]


def test_period_maps_on_nested_investment_sub_solves(tmp_path_factory) -> None:
    as_map = _nested(tmp_path_factory, as_map=True)
    as_scalar = _nested(tmp_path_factory, as_map=False)
    assert list(as_map) == [f"invest_5weeks_{p}" for p in NESTED_PERIODS]
    assert list(as_map) == list(as_scalar)
    for name, s_map in as_map.items():
        dm, ds = s_map.flex_data, as_scalar[name].flex_data
        # efficiency 0.5, min_load 0.3 at efficiency 0.4: slope = (1/0.5 - 0.3/0.4) / 0.7
        assert _value(dm.p_slope, p="coal_plant") == {round((2.0 - 0.75) / 0.7, 9)}, name
        assert _value(dm.p_profile_value, f="wind_profile") == {0.35}, name
        for field in ("p_slope", "p_section", "p_flow_upper", "p_profile_value"):
            assert _canon(getattr(dm, field)).equals(_canon(getattr(ds, field))), (name, field)
        assert s_map.solution.obj == pytest.approx(as_scalar[name].solution.obj, rel=1e-9), name


# ---------------------------------------------------------------------------
# Q13: a genuine period x time (2-D) map of the efficiency family reaches every path, the same as the
# equivalent 1-D time map (single solve was already equal before the fix; rolling / nested / stochastic
# rolling fell through to the scalar / def1 / 0 default because neither ``pd_process`` nor ``pt_process``
# carries a 2-D map).


def _mild_curve(base: float, times: list) -> dict:
    """A +-5% curve of *base* over *times* — any shape works; what matters is that the 2-D map and the
    1-D time map author the SAME per-time value, so an equal result proves the 2-D map reached the
    right tier."""
    return {t: round(base * (0.95 + 0.0006 * i), 6) for i, t in enumerate(times)}


def _eff_family_values(times: list, as_2d: bool) -> list:
    def v(base):
        curve = _mild_curve(base, times)
        return ptmap({PERIOD: curve}) if as_2d else tmap(curve)

    return [
        ["unit", "coal_plant", "efficiency", v(0.5)],
        ["unit", "hydro_plant", "efficiency", v(1.5)],
        ["unit", "gas_plant", "conversion_method", b64("min_load_efficiency", "str")],
        ["unit", "gas_plant", "startup_method", b64("linear", "str")],
        ["unit", "gas_plant", "min_load", v(0.3)],
        ["unit", "gas_plant", "efficiency_at_min_load", v(0.6)],
        ["unit", "gas_plant", "availability", v(0.8)],
        ["connection", "link", "existing", scalar(100.0)],
        ["connection", "link", "transfer_method", b64("regular", "str")],
        ["connection", "link", "efficiency", v(0.7)],
    ]


def test_period_time_2d_maps_equal_time_maps_in_rolls(tmp_path_factory) -> None:
    """Rolling (``1week_rolling``): a 2-D map ``{period1: {t: v}}`` of unit efficiency / min_load /
    efficiency_at_min_load / availability and connection efficiency gives the same ``p_slope``,
    ``p_section``, ``p_flow_upper``, ``p_indirect_input_cap`` and objective as the equivalent 1-D time
    map, in every roll."""
    as_2d = _roll(tmp_path_factory, _eff_family_values(T168, as_2d=True))
    as_1d = _roll(tmp_path_factory, _eff_family_values(T168, as_2d=False))
    _assert_rolls(as_2d)
    assert list(as_2d) == list(as_1d)
    for name, s2 in as_2d.items():
        d2, d1 = s2.flex_data, as_1d[name].flex_data
        for field in ("p_slope", "p_section", "p_flow_upper", "p_indirect_input_cap"):
            assert _canon(getattr(d2, field)).equals(_canon(getattr(d1, field))), (name, field)
        assert s2.solution.obj == pytest.approx(as_1d[name].solution.obj, rel=1e-9), name


def test_period_time_2d_maps_equal_time_maps_on_nested_investment_sub_solves(tmp_path_factory) -> None:
    """Nested investment (``invest_5weeks``): the same property on the per-period sub-solves. The
    representative timestep names are shared across ``NESTED_PERIODS`` (discovered from the existing
    scalar baseline), so the same curve is authored as one ``pdt_process`` block per period (2-D) and as
    a single ``pt_process`` curve (1-D)."""
    baseline = _nested(tmp_path_factory, as_map=False)
    times = sorted({t for step in baseline.values() for t in _frame(step.flex_data.dt)["t"].to_list()})
    assert times

    def v(base, as_2d):
        curve = _mild_curve(base, times)
        return ptmap({p: curve for p in NESTED_PERIODS}) if as_2d else tmap(curve)

    def values(as_2d: bool) -> list:
        return [["unit", "coal_plant", "efficiency", v(0.5, as_2d)],
                ["unit", "coal_plant", "conversion_method", b64("min_load_efficiency", "str")],
                ["unit", "coal_plant", "startup_method", b64("linear", "str")],
                ["unit", "coal_plant", "min_load", v(0.3, as_2d)],
                ["unit", "coal_plant", "efficiency_at_min_load", v(0.4, as_2d)],
                ["solve", "invest_5weeks", "contains_solves",
                 b64({"type": "array", "value_type": "str", "data": []}, "array")]]

    def run_nested(as_2d: bool) -> dict:
        key = json.dumps(["nested_q13", as_2d, times])
        if key not in _RUNS:
            url, scen = make_db(tmp_path_factory, alt=f"c8_{len(_RUNS)}", values=values(as_2d),
                                base_chain=NESTED_CHAIN, fixture_json=TESTS_JSON)
            _RUNS[key] = run(tmp_path_factory, url, scen)
        return _RUNS[key]

    as_2d, as_1d = run_nested(True), run_nested(False)
    assert list(as_2d) == list(as_1d) == [f"invest_5weeks_{p}" for p in NESTED_PERIODS]
    for name, s2 in as_2d.items():
        d2, d1 = s2.flex_data, as_1d[name].flex_data
        for field in ("p_slope", "p_section", "p_flow_upper"):
            assert _canon(getattr(d2, field)).equals(_canon(getattr(d1, field))), (name, field)
        assert s2.solution.obj == pytest.approx(as_1d[name].solution.obj, rel=1e-9), name


def test_period_time_2d_maps_equal_time_maps_in_stochastic_rolling(tmp_path_factory) -> None:
    """Stochastic rolling (``test_stochastic_time_params``'s ``1_week_rolling_wind`` vehicle):
    ``coal_plant`` is a stochastic-group entity, so its rolls mint branch periods even though the 2-D
    map authored here is not itself stochastic — a branch period without its own ``pdt`` row must
    inherit the anchor's, exactly like the ``pd`` tier does."""
    from tests.model.test_stochastic_time_params import (
        _reserve_entities, _roll_values, _run, ROLLING, STOCH_NODE, STOCH_UNIT,
    )

    def v(base, as_2d):
        curve = _mild_curve(base, T168)
        return ptmap({PERIOD: curve}) if as_2d else tmap(curve)

    def values(as_2d: bool) -> list:
        return _roll_values(labelled=False) + [
            ["unit", "coal_plant", "conversion_method", b64("min_load_efficiency", "str")],
            ["unit", "coal_plant", "startup_method", b64("linear", "str")],
            ["unit", "coal_plant", "efficiency", v(0.5, as_2d)],
            ["unit", "coal_plant", "min_load", v(0.3, as_2d)],
            ["unit", "coal_plant", "efficiency_at_min_load", v(0.6, as_2d)],
        ]

    entities = STOCH_UNIT + STOCH_NODE + _reserve_entities(second=True)
    as_2d = _run(tmp_path_factory, ROLLING, values(True), entities)
    as_1d = _run(tmp_path_factory, ROLLING, values(False), entities)
    assert list(as_2d) == list(as_1d)
    # The stochastic rolls (the realization rolls the designation interleaves have no branch periods).
    rolls = [k for k in as_2d if "_roll_" in k and not is_realization(k)]
    assert len(rolls) == 7, list(as_2d)
    for name in rolls:
        d2, d1 = as_2d[name].flex_data, as_1d[name].flex_data
        periods = set(_canon(d2.p_slope).filter(pl.col("p") == "coal_plant")["d"].to_list())
        assert periods - {PERIOD}, (name, periods)  # branch periods present: inheritance tier exercised
        for field in ("p_slope", "p_section"):
            c2 = _canon(getattr(d2, field)).filter(pl.col("p") == "coal_plant")
            c1 = _canon(getattr(d1, field)).filter(pl.col("p") == "coal_plant")
            assert c2.equals(c1), (name, field)
        assert as_2d[name].solution.obj == pytest.approx(as_1d[name].solution.obj, rel=1e-9), name


# ---------------------------------------------------------------------------
# Silent-``x`` time maps of the efficiency family stay out of ``pd_process`` and ``periodAll``


@pytest.fixture(scope="module")
def availability_db(tmp_path_factory) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    url = json_to_db(TESTS_JSON, tmp_path_factory.mktemp("c8_avail") / "tests.sqlite")
    migrate_database(url)
    return url


AVAILABILITY_SCENARIO = "network_coal_wind_battery_co2_fullYear_availability"
TIMESTEPS_72 = [f"t{i:04d}" for i in range(1, 73)]


def test_silent_x_time_maps_stay_out_of_pd_process(availability_db, tmp_path) -> None:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.input_derivation import run as input_derivation_run

    provider = FlexDataProvider()
    input_derivation_run(availability_db, provider, scenario_name=AVAILABILITY_SCENARIO, work_folder=tmp_path)
    # the vehicle authors x-labelled time maps of unit / connection availability, read by pt_process
    pt = provider.get("input/pt_process")
    assert pt.filter(pl.col("processParam") == "availability").height > 0
    pd_process = provider.get("input/pd_process")
    leaked = pd_process.filter(pl.col("period").is_in(TIMESTEPS_72))
    assert leaked.height == 0, leaked


def test_silent_x_time_maps_stay_out_of_period_universe(availability_db, tmp_path_factory, monkeypatch) -> None:
    from flextool.engine.preprocess.emit import _per_solve

    seen: list[list[str]] = []
    orig = _per_solve._emit_singles

    def spy(provider, key, header, values):
        if key == "solve_data/periodAll_set.csv":
            seen.append(list(values))
        return orig(provider, key, header, values)

    monkeypatch.setattr(_per_solve, "_emit_singles", spy)
    run(tmp_path_factory, availability_db, AVAILABILITY_SCENARIO)
    assert seen and all(v == ["p2020"] for v in seen), seen


# ---------------------------------------------------------------------------
# Node availability, penalty_up / penalty_down and self_discharge_loss given as a period x time (2-D) map.
#
# On a roll / nested sub-solve these fields come from the DB-direct producers (``apply_direct_params_b`` resolves
# ``p_node_availability`` and ``p_state_self_discharge`` through ``_param_shapes`` over the step's own ``dt``;
# ``apply_penalty_params`` replaces the ``pdtNode`` seed of ``p_penalty_up`` / ``_down``), so a 2-D map must give
# the same per-(n, d, t) value as the equivalent 1-D time map and, for a constant, as the scalar:
#
#     field_2d[n, d, t] = field_1d[n, d, t] (= field_scalar[n, d, t] for a constant)   every (d, t) of the step
#
# The fields keep their authored axes ((n, t) for a time map, (n, d, t) for a 2-D map, (n) for a scalar), so
# every comparison is made after broadcasting the field over the step's ``dt``.

NODE_FIELDS = ("p_node_availability", "p_state_self_discharge", "p_penalty_up", "p_penalty_down")
NODE_BASE = {("hydro_reservoir", "availability"): 0.8, ("hydro_reservoir", "self_discharge_loss"): 0.001,
             ("demand_node", "penalty_up"): 5000.0, ("demand_node", "penalty_down"): 4000.0}


def _node_cells(field, dt) -> pl.DataFrame:
    """``field`` broadcast over ``dt``: sorted ``(n, d, t, value)``, values rounded to 9 decimals."""
    f = _frame(field)
    grid = _frame(dt).select("d", "t").unique()
    if {"d", "t"} <= set(f.columns):
        f = f.join(grid, on=["d", "t"], how="semi")
    elif "t" in f.columns:
        f = f.join(grid, on="t", how="inner")
    elif "d" in f.columns:
        f = f.join(grid, on="d", how="inner")
    else:
        f = f.join(grid, how="cross")
    return (f.select("n", "d", "t", pl.col("value").cast(pl.Float64).round(9))
             .sort("n", "d", "t"))


def _node_values(shape: str, times: list, periods: list, base: dict = NODE_BASE, *, constant: bool = False) -> list:
    """``shape``: ``"2d"`` (one ``{t: v}`` block per period), ``"1d"`` (one ``{t: v}`` time map) or ``"scalar"``."""
    out = []
    for (node, param), b in base.items():
        curve = {t: b for t in times} if constant else _mild_curve(b, times)
        if shape == "2d":
            val = ptmap({p: curve for p in periods})
        elif shape == "1d":
            val = tmap(curve)
        else:
            val = scalar(b)
        out.append(["node", node, param, val])
    return out


def _assert_node_steps_equal(case: dict, twin: dict, fields=NODE_FIELDS, steps=None) -> None:
    assert list(case) == list(twin)
    for name in steps if steps is not None else list(case):
        dc, dw = case[name].flex_data, twin[name].flex_data
        for field in fields:
            c, w = _node_cells(getattr(dc, field), dc.dt), _node_cells(getattr(dw, field), dw.dt)
            assert c.height > 0, (name, field)
            assert c.equals(w), (name, field)
        assert case[name].solution.obj == pytest.approx(twin[name].solution.obj, rel=1e-9), name


@pytest.mark.parametrize("single", [False, True], ids=["rolling", "single"])
def test_node_period_time_2d_maps_equal_time_maps(tmp_path_factory, single: bool) -> None:
    """``1week_rolling`` (7 rolls) and the same solve as one single solve: a 2-D ``{period1: {t: v}}`` map of
    hydro_reservoir availability / self_discharge_loss and demand_node penalty_up / penalty_down gives the same
    four fields on every (d, t) and the same objective as the equivalent 1-D time map."""
    as_2d = _roll(tmp_path_factory, _node_values("2d", T168, [PERIOD]), single=single)
    as_1d = _roll(tmp_path_factory, _node_values("1d", T168, [PERIOD]), single=single)
    if single:
        assert len(as_2d) == 1
    else:
        _assert_rolls(as_2d)
    _assert_node_steps_equal(as_2d, as_1d)


@pytest.mark.parametrize("mode", ["single", "rolling", "stochastic_rolling"])
def test_node_constant_equals_equivalent_maps(tmp_path_factory, mode: str) -> None:
    """A constant c authored as a scalar, as a 2-D period x time map and as a 1-D time map gives the same four
    node fields and objective, in a single solve, in rolls and in stochastic rolls."""
    if mode == "stochastic_rolling":
        from tests.model.test_stochastic_time_params import ROLLING, STOCH_NODE, STOCH_UNIT, _run

        def runs(shape):
            return _run(tmp_path_factory, ROLLING, _node_values(shape, T168, [PERIOD], constant=True),
                        STOCH_UNIT + STOCH_NODE)
    else:
        def runs(shape):
            return _roll(tmp_path_factory, _node_values(shape, T168, [PERIOD], constant=True),
                         single=mode == "single")

    sc, as_2d, as_1d = runs("scalar"), runs("2d"), runs("1d")
    _assert_node_steps_equal(as_2d, sc)
    _assert_node_steps_equal(as_1d, sc)


def test_node_period_time_2d_maps_equal_time_maps_on_nested_investment_sub_solves(tmp_path_factory) -> None:
    """Nested investment (``invest_5weeks``): battery availability / self_discharge_loss and west penalty_up as
    one ``{t: v}`` block per ``NESTED_PERIODS`` period (2-D) vs one time curve (1-D), on every
    ``invest_5weeks_<p>`` sub-solve.  The timestep names are discovered from the scalar baseline."""
    baseline = _nested(tmp_path_factory, as_map=False)
    times = sorted({t for step in baseline.values() for t in _frame(step.flex_data.dt)["t"].to_list()})
    assert times
    base = {("battery", "availability"): 0.9, ("battery", "self_discharge_loss"): 0.002,
            ("west", "penalty_up"): 6000.0}
    no_nested_children = ["solve", "invest_5weeks", "contains_solves",
                          b64({"type": "array", "value_type": "str", "data": []}, "array")]

    def run_nested(shape: str) -> dict:
        values = _node_values(shape, times, NESTED_PERIODS, base) + [no_nested_children]
        key = json.dumps(["nested_node", values])
        if key not in _RUNS:
            url, scen = make_db(tmp_path_factory, alt=f"c8_{len(_RUNS)}", values=values,
                                base_chain=NESTED_CHAIN, fixture_json=TESTS_JSON)
            _RUNS[key] = run(tmp_path_factory, url, scen)
        return _RUNS[key]

    as_2d, as_1d = run_nested("2d"), run_nested("1d")
    assert list(as_2d) == [f"invest_5weeks_{p}" for p in NESTED_PERIODS]
    _assert_node_steps_equal(as_2d, as_1d, fields=("p_node_availability", "p_state_self_discharge", "p_penalty_up"))


def test_node_period_time_2d_maps_equal_time_maps_in_stochastic_rolling(tmp_path_factory) -> None:
    """Stochastic rolling (``1_week_rolling_wind``): hydro_reservoir availability / self_discharge_loss as a 2-D
    map vs a 1-D time map, next to demand_node's stochastic-time penalty_up (same in both runs).  Each roll mints
    branch periods; the 2-D map has rows only for ``period1``, so a branch period must take its anchor's values
    (the anchor inheritance of ``broadcast_to_period_time``).  The stochastic-time node keeps its branch values,
    read from the ``pdtNode`` emit (``_param_shapes.STOCHASTIC_TIME_PARAMS``)."""
    from tests.model.test_stochastic_time_params import (
        AT_ROLL, ROLLING, STOCH_NODE, STOCH_UNIT, _assert_branch_values, _cells, _r3, _run,
    )

    hydro = {k: v for k, v in NODE_BASE.items() if k[0] == "hydro_reservoir"}
    penalty = [["node", "demand_node", "penalty_up", _r3(10000.0, AT_ROLL, True)]]
    as_2d = _run(tmp_path_factory, ROLLING, _node_values("2d", T168, [PERIOD], hydro) + penalty,
                 STOCH_UNIT + STOCH_NODE)
    as_1d = _run(tmp_path_factory, ROLLING, _node_values("1d", T168, [PERIOD], hydro) + penalty,
                 STOCH_UNIT + STOCH_NODE)
    rolls = [k for k in as_2d if "_roll_" in k and not is_realization(k)]
    assert len(rolls) == 7, list(as_2d)
    for name in rolls:
        step = as_2d[name]
        branch_periods = set(_frame(step.flex_data.dt)["d"].to_list()) - {PERIOD}
        assert branch_periods, name
        for field in ("p_node_availability", "p_state_self_discharge"):
            cells = _node_cells(getattr(step.flex_data, field), step.flex_data.dt).filter(
                pl.col("n") == "hydro_reservoir")
            assert branch_periods <= set(cells["d"].to_list()), (name, field)
        _assert_branch_values(_cells(step, "p_penalty_up", ["n"]), ("demand_node",), 10000.0,
                              periods=branch_periods)
    _assert_node_steps_equal(as_2d, as_1d, fields=("p_node_availability", "p_state_self_discharge",
                                                   "p_penalty_up"), steps=rolls)


def _csv_dump_pdt_node(tmp_path_factory, shape: str) -> pl.DataFrame:
    """Run ``1week_rolling`` (node values of *shape*) with ``csv_dump=True`` and snapshot the last roll's Provider the way ``--csv-dump``
    does (``cmd_run_flextool``); return that dump's ``solve_data/pdtNode.csv`` rows of the ``NODE_BASE``
    (node, param) pairs, sorted."""
    from flextool.engine import run_chain_from_db

    alt = f"c8_dump_{shape}"
    url, scen = make_db(tmp_path_factory, alt=alt, values=_node_values(shape, T168, [PERIOD]), base_chain=ROLL_CHAIN,
                        fixture_json=_deterministic_rolling_json(tmp_path_factory, alt))
    work = tmp_path_factory.mktemp(f"{alt}_w")
    steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True, csv_dump=True)
    last = next(reversed(list(steps.values())))
    assert "_roll_" in next(reversed(list(steps))), list(steps)
    last.flex_data_provider.snapshot_processed_inputs(work)
    pairs = pl.DataFrame({"node": [n for n, _ in NODE_BASE], "param": [p for _, p in NODE_BASE]})
    dump = pl.read_csv(work / "solve_data" / "pdtNode.csv", infer_schema_length=0)
    return (dump.join(pairs, on=["node", "param"], how="semi")
                .with_columns(pl.col("value").cast(pl.Float64))
                .sort("node", "param", "period", "time"))


def test_csv_dump_pdt_node_shows_period_time_2d_maps(tmp_path_factory) -> None:
    """``--csv-dump`` of a roll: ``solve_data/pdtNode.csv`` shows a 2-D period x time map of node availability,
    self_discharge_loss and penalties exactly like the equivalent 1-D time map, i.e. the authored value at each
    (period1, t) of the roll — not the class default."""
    as_2d = _csv_dump_pdt_node(tmp_path_factory, "2d")
    as_1d = _csv_dump_pdt_node(tmp_path_factory, "1d")
    assert as_2d.height > 0
    assert set(as_2d.select("node", "param").unique().iter_rows()) == set(NODE_BASE)
    authored = {k: _mild_curve(b, T168) for k, b in NODE_BASE.items()}
    for node, param, _period, t, v in as_2d.iter_rows():
        assert v == pytest.approx(authored[(node, param)][t], rel=1e-12), (node, param, t, v)
    assert as_2d.equals(as_1d)
