"""Online units without ``min_load``; ``min_load_efficiency`` defaults (#70).

``min_load`` is the minimum-load floor's parameter: no value means 0 (no floor).  An online unit needs no
``min_load``, and a ``min_load_efficiency`` unit reads

    slope[p,d,t]   = (1/eta − min_load/eta_min) / (1 − min_load)
    section[p,d,t] = round(1/eta, 6) − round((round(1/eta, 6) − min_load/eta_min) / (1 − min_load), 6)
    eta = efficiency (default 1);  min_load = 0 where not authored;
    eta_min = efficiency_at_min_load where given, else efficiency (owner decision c11: a flat curve)

``process_minload`` (the floor's units) = units with ``min_load > 0``.

Before the fix an online unit without any ``min_load`` in the model stopped the build (``p_min_load`` was an
online gate field); a ``min_load_efficiency`` unit without ``min_load`` stopped it on ``p_slope`` / ``p_section``
or, next to other ``min_load_efficiency`` units, lost its slope row and used no fuel; a unit without
``efficiency`` lost its section row; a ``min_load_efficiency`` unit without ``min_load`` got a floor row; a unit
with ``min_load > 0`` and no ``efficiency_at_min_load`` used no fuel on single / stochastic solves and a wrong
curve in rolls.

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``single``: ``tests/fixtures/tests.json`` init / west / coal, unit ``coal_plant`` (efficiency 0.4).
- ``roll``: ``stochastics.json`` ``1_week_rolling_wind`` made deterministic (7 rolls), unit ``gas_plant``
  (efficiency 0.9; second unit ``coal_plant``, efficiency 0.9).
- ``stoch``: ``stochastics.json`` ``2_day_stochastic_dispatch``, same units.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, build, make_db, utf8

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TESTS_JSON = FIXTURES / "tests.json"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"

UNIT = {"single": "coal_plant", "roll": "gas_plant", "stoch": "gas_plant"}
EFF = {"single": 0.4, "roll": 0.9, "stoch": 0.9}
OTHER = "coal_plant"          # second unit of the stochastics vehicles
VEHICLES = ("single", "roll", "stoch")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def scalar(v: float) -> list:
    return b64(float(v), "float")


def s(v: str) -> list:
    return b64(v, "str")


def tmap(values: dict) -> list:
    return b64({"index_type": "str", "index_name": "time", "data": [[k, v] for k, v in values.items()]}, "map")


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _fixture(tmp_path_factory, veh: str, *, strip_efficiency: str | None) -> tuple[Path, list[str]]:
    """``(fixture json, base chain)``; ``strip_efficiency``: drop that unit's ``efficiency`` values."""
    if veh == "single":
        src, chain = TESTS_JSON, ["init", "west", "coal"]
    else:
        src = STOCHASTICS_JSON
        chain = _chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch" if veh == "stoch" else "1_week_rolling_wind")
    if veh != "roll" and strip_efficiency is None:
        return src, chain
    data = json.loads(src.read_text())
    if veh == "roll":
        data["parameter_values"] = [
            pv for pv in data["parameter_values"]
            if pv[2] not in ("stochastic_branches", "include_stochastics")
            and not (pv[0] == "profile" and pv[1] == "wind" and pv[4] in ("1week_rolling", "2day"))]
        data["entities"] += [["connection", "link", None],
                             ["connection__node__node", ["link", "gas_market", "coal_market"], None]]
        data["entity_alternatives"].append(["connection", ["link"], "init", True])
    if strip_efficiency is not None:
        data["parameter_values"] = [pv for pv in data["parameter_values"]
                                    if not (pv[0] == "unit" and pv[1] == strip_efficiency and pv[2] == "efficiency")]
    out = tmp_path_factory.mktemp("c11c_json") / src.name
    out.write_text(json.dumps(data))
    return out, chain


_RUNS: dict = {}


def _run(tmp_path_factory, veh: str, values: list, *, strip_efficiency: str | None = None) -> dict:
    from flextool.engine import run_chain_from_db

    key = json.dumps([veh, values, strip_efficiency])
    if key not in _RUNS:
        fj, chain = _fixture(tmp_path_factory, veh, strip_efficiency=strip_efficiency)
        alt = f"c11c_{len(_RUNS)}"
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=fj)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)
        assert steps, "run_chain_from_db produced no orchestration steps"
        for step in steps.values():
            assert step.solution is not None, "solve produced no solution"
        _RUNS[key] = steps
    return _RUNS[key]


def _families(step) -> dict[str, int]:
    """Row count of every constraint family of the solve's LP."""
    pb = build(step.flex_data)
    out: dict[str, int] = {}
    for name, _proto, over in pb._cstrs:
        out[name] = out.get(name, 0) + len(over)
    return out


def _assert_same_model(a: dict, b: dict) -> None:
    """Same solves, objectives (rel 1e-9) and constraint families with the same row counts."""
    assert list(a) == list(b)
    for name in a:
        assert float(a[name].solution.obj) == pytest.approx(float(b[name].solution.obj), rel=1e-9), name
        assert _families(a[name]) == _families(b[name]), name


def _unit_rows(param, unit: str) -> pl.DataFrame:
    assert param is not None
    return utf8(param.frame).filter(pl.col("p") == unit)


def _floor_rows(step, unit: str) -> int:
    pb = build(step.flex_data)
    n = 0
    for name in ("minFlow_minload_linear", "minFlow_minload_integer"):
        for r in pb.cstrs_named(name):
            if r.name == name:
                n += utf8(r.over).filter(pl.col("p") == unit).height
    return n


def _assert_floor_holds(step, unit: str, min_load: float) -> None:
    """The unit runs at or above min_load · online in every step (and is online somewhere)."""
    flow = (utf8(step.solution.value("v_flow")).filter(pl.col("p") == unit)
            .group_by("d", "t").agg(pl.col("value").sum().alias("flow")))
    on = utf8(step.solution.value("v_online_linear")).filter(pl.col("p") == unit).rename({"value": "online"})
    j = on.join(flow, on=["d", "t"], how="left").with_columns(pl.col("flow").fill_null(0.0))
    assert j.height > 0 and j["online"].max() > 1e-6, "unit never online: the floor is not exercised"
    assert (j["flow"] - min_load * j["online"]).min() >= -1e-6


def _online(veh: str, method: str = "linear") -> list:
    u = UNIT[veh]
    vals = [["unit", u, "startup_method", s("binary" if method == "binary" else "linear")]]
    if method == "min_uptime":
        vals += [["unit", u, "minimum_time_method", s("min_uptime")], ["unit", u, "min_uptime", scalar(3.0)]]
    return vals


def _mle(unit: str, *, eta_min: float | None = None, min_load: float | None = None) -> list:
    vals = [["unit", unit, "conversion_method", s("min_load_efficiency")]]
    if eta_min is not None:
        vals.append(["unit", unit, "efficiency_at_min_load", scalar(eta_min)])
    if min_load is not None:
        vals.append(["unit", unit, "min_load", scalar(min_load)])
    return vals


# ---------------------------------------------------------------------------
# Online units without min_load


@pytest.mark.parametrize("method", ["linear", "binary", "min_uptime"])
@pytest.mark.parametrize("veh", VEHICLES)
def test_online_without_min_load(tmp_path_factory, veh, method) -> None:
    """An online unit with no ``min_load`` anywhere builds and solves like the same model with ``min_load 0``.
    Before: ``feature 'online' … not populated (None): ['p_min_load']``."""
    steps = _run(tmp_path_factory, veh, _online(veh, method))
    ref = _run(tmp_path_factory, veh, _online(veh, method) + [["unit", UNIT[veh], "min_load", scalar(0.0)]])
    _assert_same_model(steps, ref)
    st = list(steps.values())[-1]
    assert st.flex_data.p_min_load is None
    assert st.flex_data.process_minload is None


# ---------------------------------------------------------------------------
# min_load_efficiency without min_load / efficiency


@pytest.mark.parametrize("eta_min", [None, 0.3], ids=["no_eta_min", "eta_min"])
@pytest.mark.parametrize("veh", VEHICLES)
def test_min_load_efficiency_without_min_load(tmp_path_factory, veh, eta_min) -> None:
    """A min_load_efficiency online unit without ``min_load`` equals the same unit with ``min_load 0``: slope
    1/efficiency, section 0.  Before: crash on ``p_min_load`` / ``p_slope``."""
    u = UNIT[veh]
    steps = _run(tmp_path_factory, veh, _online(veh) + _mle(u, eta_min=eta_min))
    ref = _run(tmp_path_factory, veh, _online(veh) + _mle(u, eta_min=eta_min, min_load=0.0))
    _assert_same_model(steps, ref)
    st = list(steps.values())[-1]
    slope = _unit_rows(st.flex_data.p_slope, u)
    assert slope.height == st.flex_data.dt.height
    assert slope["value"].to_list() == pytest.approx([1.0 / EFF[veh]] * slope.height, rel=1e-6)
    section = _unit_rows(st.flex_data.p_section, u)
    assert section.height == st.flex_data.dt.height
    assert section["value"].abs().max() < 1e-12


@pytest.mark.parametrize("veh", ["roll", "stoch"])
def test_min_load_efficiency_unit_next_to_one_with_min_load(tmp_path_factory, veh) -> None:
    """Both units min_load_efficiency (efficiency_at_min_load 0.3); coal has min_load 0.4, gas none: gas has its
    slope and section rows and the model equals gas with ``min_load 0``.  Before (stoch): gas had no slope row and
    used no fuel (127 824 057.32 vs 144 593 994.67); (roll) no gas section row."""
    common = (_online(veh) + [["unit", OTHER, "startup_method", s("linear")]]
              + _mle(OTHER, eta_min=0.3, min_load=0.4))
    steps = _run(tmp_path_factory, veh, common + _mle(UNIT[veh], eta_min=0.3))
    ref = _run(tmp_path_factory, veh, common + _mle(UNIT[veh], eta_min=0.3, min_load=0.0))
    _assert_same_model(steps, ref)
    for st in steps.values():
        fd = st.flex_data
        assert _unit_rows(fd.p_slope, UNIT[veh]).height == fd.dt.height
        assert _unit_rows(fd.p_section, UNIT[veh]).height == fd.dt.height
        assert sorted(utf8(fd.process_minload)["p"].to_list()) == [OTHER]


@pytest.mark.parametrize("veh", ["roll", "stoch"])
def test_min_load_efficiency_default_efficiency(tmp_path_factory, veh) -> None:
    """A min_load_efficiency unit without ``efficiency`` (schema default 1), ``min_load 0.3`` and
    ``efficiency_at_min_load 0.8`` equals ``efficiency 1`` given explicitly (stoch 69 357 129.64; roll 0
    18 996 631.46, slope 0.892857, section 0.107143).  Before (stoch): crash on ``p_section``."""
    u = UNIT[veh]
    base = _online(veh) + _mle(u, eta_min=0.8, min_load=0.3)
    steps = _run(tmp_path_factory, veh, base, strip_efficiency=u)
    ref = _run(tmp_path_factory, veh, base + [["unit", u, "efficiency", scalar(1.0)]], strip_efficiency=u)
    _assert_same_model(steps, ref)
    st = list(steps.values())[0]
    assert _unit_rows(st.flex_data.p_slope, u)["value"].to_list() == pytest.approx(
        [(1.0 - 0.3 / 0.8) / 0.7] * st.flex_data.dt.height, rel=1e-6)
    assert _unit_rows(st.flex_data.p_section, u)["value"].to_list() == pytest.approx(
        [0.107143] * st.flex_data.dt.height, abs=1e-6)


def test_no_floor_for_min_load_efficiency_unit_without_min_load(tmp_path_factory) -> None:
    """gas min_load_efficiency without ``min_load`` next to coal online with ``min_load 0``: no floor rows.
    Before: 240 ``minFlow_minload_linear`` rows for gas (the emit's min_load_efficiency seed)."""
    steps = _run(tmp_path_factory, "stoch", _online("stoch") + _mle("gas_plant", eta_min=0.3)
                 + [["unit", OTHER, "startup_method", s("linear")], ["unit", OTHER, "min_load", scalar(0.0)]])
    st = list(steps.values())[-1]
    assert st.flex_data.process_minload is None
    assert _floor_rows(st, "gas_plant") == 0
    assert _floor_rows(st, OTHER) == 0


# ---------------------------------------------------------------------------
# min_load > 0 without efficiency_at_min_load: efficiency at minimum load too (owner decision c11)


@pytest.mark.parametrize("veh", VEHICLES)
def test_min_load_efficiency_without_efficiency_at_min_load_is_flat(tmp_path_factory, veh, caplog) -> None:
    """``min_load 0.3`` and no ``efficiency_at_min_load``: the unit uses its efficiency at minimum load too, so
    the model equals ``efficiency_at_min_load = efficiency`` given explicitly, the slope is 1/efficiency and the
    minimum load still applies.  A warning names the unit.  Before: no fuel use (single, stoch) or a curve with
    efficiency_at_min_load 0 (roll)."""
    u = UNIT[veh]
    with caplog.at_level(logging.WARNING):
        steps = _run(tmp_path_factory, veh, _online(veh) + _mle(u, min_load=0.3))
    ref = _run(tmp_path_factory, veh, _online(veh) + _mle(u, min_load=0.3, eta_min=EFF[veh]))
    _assert_same_model(steps, ref)
    for name, st in steps.items():
        fd = st.flex_data
        slope = _unit_rows(fd.p_slope, u)
        assert slope.height == fd.dt.height, name
        assert slope["value"].to_list() == pytest.approx([1.0 / EFF[veh]] * slope.height, rel=1e-5), name
        assert _floor_rows(st, u) == fd.dt.height, name
        _assert_floor_holds(st, u, 0.3)
    if veh == "stoch":
        # The floor binds here (on single and roll a linear online variable without start costs can follow the
        # flow, so the floor rows hold without changing the optimum): without it the model differs.
        no_floor = _run(tmp_path_factory, veh, _online(veh) + _mle(u))
        assert float(list(steps.values())[-1].solution.obj) != pytest.approx(
            float(list(no_floor.values())[-1].solution.obj), rel=1e-6)
    assert (f"Unit '{u}' uses min_load_efficiency with min_load > 0 but has no efficiency_at_min_load"
            in caplog.text)


@pytest.mark.parametrize("veh", VEHICLES)
def test_efficiency_at_min_load_map_leaves_cells_out(tmp_path_factory, veh) -> None:
    """``efficiency_at_min_load`` as a time map that covers only the first 12 steps: the other steps use the
    efficiency, as if the map gave it there."""
    u = UNIT[veh]
    steps_all = [f"t{i:04d}" for i in range(1, 169 if veh == "roll" else 49)]
    first = steps_all[:12]
    partial = tmap({t: 0.3 for t in first})
    full = tmap({t: (0.3 if t in first else EFF[veh]) for t in steps_all})
    steps = _run(tmp_path_factory, veh, _online(veh) + _mle(u, min_load=0.3)
                 + [["unit", u, "efficiency_at_min_load", partial]])
    ref = _run(tmp_path_factory, veh, _online(veh) + _mle(u, min_load=0.3)
               + [["unit", u, "efficiency_at_min_load", full]])
    _assert_same_model(steps, ref)
    for st in steps.values():
        assert _unit_rows(st.flex_data.p_slope, u).height == st.flex_data.dt.height


def test_min_load_efficiency_warning_folds_maps(tmp_path_factory, caplog) -> None:
    """The warning sees ``min_load > 0`` in a period map and in a stochastic-time map, and not in a map of
    zeros or when ``efficiency_at_min_load`` is given."""
    from spinedb_api import DatabaseMapping
    from spinedb_api.filters.scenario_filter import scenario_filter_config, scenario_filter_from_dict

    from flextool.input_derivation._validators import validate_min_load_efficiency_inputs

    stoch = b64({"index_type": "str", "index_name": "branch", "data": [
        ["upper", {"type": "map", "index_type": "str", "index_name": "analysis_time", "data": [
            ["t0001", {"type": "map", "index_type": "str", "index_name": "time",
                       "data": [["t0001", 0.0], ["t0002", 0.2]]}]]}]]}, "map")
    period_map = b64({"index_type": "str", "index_name": "period", "data": [["p2020", 0.3]]}, "map")
    zeros = b64({"index_type": "str", "index_name": "period", "data": [["p2020", 0.0]]}, "map")
    values = (_mle("coal_plant") + [["unit", "coal_plant", "min_load", period_map]]
              + _mle("coal_chp") + [["unit", "coal_chp", "min_load", stoch]]
              + _mle("wind_plant") + [["unit", "wind_plant", "min_load", zeros]]
              + _mle("fusion_plant", eta_min=0.3, min_load=0.3))
    url, scen = make_db(tmp_path_factory, alt="c11c_warn", values=values,
                        base_chain=["init", "west", "coal", "coal_chp", "wind", "fusion"])
    logger = logging.getLogger("c11c_warn")
    with caplog.at_level(logging.WARNING, logger="c11c_warn"):
        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            scenario_filter_from_dict(db, scenario_filter_config(scen))
            validate_min_load_efficiency_inputs(db, logger)
    warned = sorted(r.getMessage().split("'")[1] for r in caplog.records if r.name == "c11c_warn")
    assert warned == ["coal_chp", "coal_plant"]
