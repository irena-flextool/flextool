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

from tests.engine._unit_db import b64, make_db, run, utf8

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
        assert _value(dm.p_slope, p="coal_plant") == {2.0}, name
        assert _value(dm.p_slope, p="hydro_plant") == {0.666667}, name
        assert _value(dm.p_slope, p="gas_plant") == {0.873016}, name
        assert _value(dm.p_slope, p="link") == {round(1 / 0.7, 6)}, name
        assert _value(dm.p_flow_upper, p="hydro_plant", source="hydro_reservoir", sink="hydro_plant") == {0.666667}
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
            assert s["value"].to_list() == pytest.approx(r["value"].to_list(), rel=1e-6), (name, field)


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
        assert _value(dm.p_slope, p="coal_plant") == {round((2.0 - 0.75) / 0.7, 6)}, name
        assert _value(dm.p_profile_value, f="wind_profile") == {0.35}, name
        for field in ("p_slope", "p_section", "p_flow_upper", "p_profile_value"):
            assert _canon(getattr(dm, field)).equals(_canon(getattr(ds, field))), (name, field)
        assert s_map.solution.obj == pytest.approx(as_scalar[name].solution.obj, rel=1e-9), name


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
