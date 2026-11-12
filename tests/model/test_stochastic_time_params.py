"""Stochastic-time maps of availability, efficiency, min_load, other operational costs, reservation and penalties
(#78).

``docs/how_to/stochastics.md``: "The tool recognizes a stochastic timeseries from the dimensions of the data" — a three-level
map ``branch → analysis_time → time``.  For the parameters whose schema allows it (``node.availability``,
``node.penalty_up`` / ``penalty_down``, ``unit.availability`` / ``efficiency`` / ``min_load``, the unit arcs'
``other_operational_cost``, ``reserve__upDown__group.reservation``):

    value[e, d, t] = folded stochastic value of e at (d, t)    e's value is a stochastic-time map
                   = the DB-direct value of e at (d, t)        any other shape

"Folded" is what the per-solve emit computes from ``pbt_*``: on a branch period of an ``include_stochastics``
group, the branch's series for the solve's analysis time.  Labelled and silent maps give the same result, and one
entity's stochastic-time map does not change how the other entities are read.

Before the fix the labelled map stopped the model (``index_name 'branch' which is not a valid axis``) for every
parameter but the penalties, and a silent map sent the whole parameter to the emit seed: another entity's period ×
time map was lost, ``node.availability`` was lost entirely, every unit's efficiency became 1, and a stochastic
``min_load`` stopped the build.

Vehicles (databases built from JSON, never a checked-in ``.sqlite``): ``2_day_stochastic_dispatch`` (``period1``
and the branch periods ``period1_{realized, upper, lower, mid}``) and the 7 rolls of ``1_week_rolling_wind`` of
``tests/fixtures/stochastics.json``.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._unit_db import b64, is_realization, make_db, utf8
from tests.model.test_balance_penalty_shapes import (
    BRANCH_PERIODS,
    BRANCHES,
    STOCHASTICS_JSON,
    T48,
    T168,
    _chain,
    scalar,
    stoch_map,
)

ANCHOR = "period1"
SINGLE = "2_day_stochastic_dispatch"
ROLLING = "1_week_rolling_wind"
BR = dict(zip(BRANCHES, (0.9, 0.8, 0.7, 0.6)))
AT_SINGLE = {"t0001": T48}
AT_ROLL = {f"t{s:04d}": T168[s - 1:] for s in range(1, 169, 24)}
COAL_IN = ["coal_plant", "coal_market"]
COAL_OUT = ["coal_plant", "demand_node"]
GAS_IN = ["gas_plant", "gas_market"]
STOCH_UNIT = [["group__unit", ["add_stochastics", "coal_plant"], None]]
STOCH_NODE = [["group__node", ["add_stochastics", "demand_node"], None]]
ROLLING_SEPARATE_DISPATCH_JSON = STOCHASTICS_JSON.parent / "rolling_separate_dispatch.json"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _alt(v: float, i: int) -> float:
    """The second entity's period × time curve: ``1.1 v`` on even steps, ``v`` on odd ones."""
    return v if i % 2 else v * 1.1


def _ptmap(v: float, timesteps: list[str]) -> list:
    return b64({"type": "map", "index_type": "str", "index_name": "period", "data": [
        [ANCHOR, {"type": "map", "index_type": "str", "index_name": "time",
                  "data": [[t, _alt(v, i)] for i, t in enumerate(timesteps)]}]]}, "map")


def _pt_value(v: float, t: str) -> float:
    return _alt(v, int(t[1:]) - 1)


def _r3(mult: float, at: dict, labelled: bool) -> list:
    return stoch_map({b: x * mult for b, x in BR.items()}, at, labelled=labelled)


def _reserve_entities(second: bool) -> list:
    ents = [["upDown", "up", None], ["reserve", "r1", None], ["group", "res_g", None],
            ["group__node", ["res_g", "demand_node"], None],
            ["reserve__upDown__group", ["r1", "up", "res_g"], None],
            ["reserve__upDown__unit__node", ["r1", "up", "coal_plant", "demand_node"], None],
            ["reserve__upDown__unit__node", ["r1", "up", "gas_plant", "demand_node"], None]]
    if second:
        ents += [["reserve", "r2", None], ["reserve__upDown__group", ["r2", "up", "res_g"], None],
                 ["reserve__upDown__unit__node", ["r2", "up", "gas_plant", "demand_node"], None]]
    return ents


def _reserve_values(r1, r2, *, grouped: bool = True) -> list:
    vals = [["group", "res_g", "include_stochastics", b64("yes", "str")]] if grouped else []
    for r, units in (("r1", ("coal_plant", "gas_plant")), ("r2", ("gas_plant",))):
        if r == "r2" and r2 is None:
            continue
        vals += [["reserve__upDown__group", [r, "up", "res_g"], "reserve_method", b64("timeseries_only", "str")],
                 ["reserve__upDown__group", [r, "up", "res_g"], "penalty_reserve", scalar(5000.0)],
                 ["reserve__upDown__group", [r, "up", "res_g"], "reservation", r1 if r == "r1" else r2]]
        for u in units:
            vals += [["reserve__upDown__unit__node", [r, "up", u, "demand_node"], "max_share", scalar(1.0)],
                     ["reserve__upDown__unit__node", [r, "up", u, "demand_node"], "reliability", scalar(1.0)]]
    return vals


_RUNS: dict = {}


def _run(tmp_path_factory, scenario: str, values: list, entities: list, *, warm: bool = True) -> dict:
    from flextool.engine import run_chain_from_db

    key = json.dumps([scenario, values, entities, warm])
    if key not in _RUNS:
        alt = f"c10b_{len(_RUNS)}"
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_chain(STOCHASTICS_JSON, scenario),
                            entities=entities, fixture_json=STOCHASTICS_JSON)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"),
                                  keep_solutions=True, warm=warm)
        assert steps and all(s.solution is not None for s in steps.values())
        _RUNS[key] = steps
    return _RUNS[key]


def _frame(x) -> pl.DataFrame:
    assert x is not None, "parameter is None"
    f = getattr(x, "frame", x)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    return utf8(f)


def _cells(step, field: str, keys: list[str]) -> dict:
    f = _frame(getattr(step.flex_data, field))
    if not {"d", "t"} <= set(f.columns):
        f = f.join(utf8(step.flex_data.dt).select("d", "t"), how="cross")
    f = f.select(*keys, "d", "t", "value")
    assert not f.select(*keys, "d", "t").is_duplicated().any(), f"{field}: duplicate cells"
    return {tuple(r[:-1]): float(r[-1]) for r in f.iter_rows()}


def _dt(step) -> list[tuple[str, str]]:
    return list(utf8(step.flex_data.dt).select("d", "t").iter_rows())


def _first(steps: dict):
    return next(iter(steps.values()))


def _stochastic_steps(steps: dict) -> list:
    """The steps of the stochastic solve, without the engine-added realization solve that follows it."""
    return [s for k, s in steps.items() if "__realized_dispatch" not in k]


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Single stochastic solve: availability, both arc costs, node availability, penalty, reservation

# field -> (key columns, stochastic entity, second entity with a period × time map or None, branch value multiplier,
#           second entity's base value, emit key, emit entity columns, emit parameter, cost-like (zero rows dropped))
SINGLE_FIELDS = {
    "p_process_availability": (["p"], ("coal_plant",), ("gas_plant",), 1.0, 0.5,
                               "solve_data/pdtProcess", ["process"], "availability", False),
    "p_pdt_varCost_source": (["p", "source"], tuple(COAL_IN), tuple(GAS_IN), 10.0, 5.0,
                             "solve_data/pdtProcess_source", ["process", "source"], "other_operational_cost", True),
    "p_pdt_varCost_sink": (["p", "sink"], tuple(COAL_OUT), None, 10.0, None,
                           "solve_data/pdtProcess_sink", ["process", "sink"], "other_operational_cost", True),
    "p_node_availability": (["n"], ("hydro_reservoir",), ("demand_node",), 1.0, 0.5,
                            "solve_data/pdtNode", ["node"], "availability", False),
    "p_penalty_up": (["n"], ("demand_node",), ("downriver",), 10000.0, 7000.0,
                     "solve_data/pdtNode", ["node"], "penalty_up", False),
    "pdtReserve_upDown_group_reservation": (["r", "ud", "g"], ("r1", "up", "res_g"), ("r2", "up", "res_g"), 100.0,
                                            50.0, "solve_data/pdtReserve_upDown_group",
                                            ["reserve", "upDown", "group"], "reservation", False),
}


def _single_values(labelled: bool, at: dict, timesteps: list[str]) -> list:
    def r3(mult):
        return _r3(mult, at, labelled)
    return ([["unit", "coal_plant", "availability", r3(1.0)],
             ["unit", "gas_plant", "availability", _ptmap(0.5, timesteps)],
             ["unit__inputNode", COAL_IN, "other_operational_cost", r3(10.0)],
             ["unit__inputNode", GAS_IN, "other_operational_cost", _ptmap(5.0, timesteps)],
             ["unit__outputNode", COAL_OUT, "other_operational_cost", r3(10.0)],
             ["node", "hydro_reservoir", "availability", r3(1.0)],
             ["node", "demand_node", "availability", _ptmap(0.5, timesteps)],
             ["node", "demand_node", "penalty_up", r3(10000.0)],
             ["node", "downriver", "penalty_up", _ptmap(7000.0, timesteps)]]
            + _reserve_values(r3(100.0), _ptmap(50.0, timesteps)))


def _single(tmp_path_factory, labelled: bool) -> dict:
    return _run(tmp_path_factory, SINGLE, _single_values(labelled, AT_SINGLE, T48),
                STOCH_UNIT + STOCH_NODE + _reserve_entities(second=True))


def _assert_branch_values(cells: dict, ent: tuple, mult: float, periods=BRANCH_PERIODS) -> None:
    for (*k, d, t), v in cells.items():
        if tuple(k) == ent and d in periods:
            assert v == pytest.approx(BR[d.removeprefix("period1_")] * mult, rel=1e-12), (ent, d, t, v)


@pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "silent"])
def test_stochastic_time_single_solve(tmp_path_factory, labelled: bool) -> None:
    """Each parameter: the rank-3 entity carries its branch values on the branch periods and exactly the emit's
    fold on every (d, t); the second entity keeps its period × time map on every (d, t) (before the fix: crash
    when labelled; silent: the second entity's map lost, node availability lost entirely)."""
    step = _first(_single(tmp_path_factory, labelled))
    provider = step.flex_data_provider
    for field, (keys, ent, other, mult, base, emit_key, emit_cols, param, costlike) in SINGLE_FIELDS.items():
        cells = _cells(step, field, keys)
        mine = {k: v for k, v in cells.items() if k[:len(keys)] == ent}
        assert {k[-2] for k in mine} >= set(BRANCH_PERIODS), (field, sorted({k[-2] for k in mine}))
        _assert_branch_values(cells, ent, mult)
        emit = utf8(provider.get(emit_key))
        emit = emit.filter(pl.col("param") == param)
        for c, e in zip(emit_cols, ent):
            emit = emit.filter(pl.col(c) == e)
        folded = {(*ent, d, t): float(v) for d, t, v in emit.select("period", "time", "value").iter_rows()}
        if costlike:
            folded = {k: v for k, v in folded.items() if v != 0.0}
        dt = set(_dt(step))
        folded = {k: v for k, v in folded.items() if k[-2:] in dt}
        assert mine == folded, field
        if other is not None:
            for d, t in _dt(step):
                assert cells[(*other, d, t)] == pytest.approx(_pt_value(base, t), rel=1e-12), (field, other, d, t)


def test_labelled_equals_silent_single(tmp_path_factory) -> None:
    lab, sil = _first(_single(tmp_path_factory, True)), _first(_single(tmp_path_factory, False))
    assert _rel(float(lab.solution.obj), float(sil.solution.obj)) < 1e-9
    for field, (keys, *_rest) in SINGLE_FIELDS.items():
        assert _cells(lab, field, keys) == _cells(sil, field, keys), field


def test_penalty_stochastic_node_does_not_hide_other_maps(tmp_path_factory) -> None:
    """Before the fix the emit seed was kept for every node as soon as one node authored rank 3: downriver lost its
    period × time map (default 10 000).  Now it keeps 7000 / 7700, and the third balance node keeps the default."""
    step = _first(_single(tmp_path_factory, True))
    cells = _cells(step, "p_penalty_up", ["n"])
    for d, t in _dt(step):
        assert cells[("downriver", d, t)] == pytest.approx(_pt_value(7000.0, t), rel=1e-12)
        assert cells[("hydro_reservoir", d, t)] == 10000.0


# ---------------------------------------------------------------------------
# Efficiency and min_load (both units min_load_efficiency + linear startup)


def _eff_values(coal_eff, coal_ml) -> list:
    vals = []
    for u in ("coal_plant", "gas_plant"):
        vals += [["unit", u, "startup_method", b64("linear", "str")],
                 ["unit", u, "conversion_method", b64("min_load_efficiency", "str")],
                 ["unit", u, "efficiency_at_min_load", scalar(0.6)]]
    return vals + [["unit", "coal_plant", "efficiency", coal_eff],
                   ["unit", "gas_plant", "efficiency", _ptmap(0.5, T48)],
                   ["unit", "coal_plant", "min_load", coal_ml],
                   ["unit", "gas_plant", "min_load", _ptmap(0.3, T48)]]


def _eff(tmp_path_factory, labelled: bool | None) -> dict:
    """``labelled=None``: coal's efficiency / min_load as scalars (the reference for the other units)."""
    if labelled is None:
        vals = _eff_values(scalar(0.9), scalar(0.45))
    else:
        vals = _eff_values(_r3(1.0, AT_SINGLE, labelled), _r3(0.5, AT_SINGLE, labelled))
    return _run(tmp_path_factory, SINGLE, vals, STOCH_UNIT)


@pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "silent"])
def test_stochastic_efficiency_and_min_load(tmp_path_factory, labelled: bool) -> None:
    """coal's branch efficiency and min_load reach ``p_min_load`` (branch values) and the slope / section; gas
    keeps its period × time maps; ``p_min_load`` and ``p_section`` are dense for both units (before the fix:
    crash when labelled; silent: ONLINE gate error, and every unit's efficiency 1)."""
    step = _first(_eff(tmp_path_factory, labelled))
    ml = _cells(step, "p_min_load", ["p"])
    _assert_branch_values(ml, ("coal_plant",), 0.5)
    for d, t in _dt(step):
        assert ml[("gas_plant", d, t)] == pytest.approx(_pt_value(0.3, t), rel=1e-12)
    for field in ("p_min_load", "p_section", "p_slope"):
        cells = _cells(step, field, ["p"])
        for u in ("coal_plant", "gas_plant"):
            assert all((u, d, t) in cells for d, t in _dt(step)), (field, u)
    slope = _cells(step, "p_slope", ["p"])
    for d in BRANCH_PERIODS:
        eta, m = BR[d.removeprefix("period1_")], BR[d.removeprefix("period1_")] * 0.5
        want = (1 / eta - m / 0.6) / (1 - m)
        assert slope[("coal_plant", d, "t0001")] == pytest.approx(want, rel=1e-9), d
    assert set(utf8(step.flex_data.process_minload)["p"]) >= {"coal_plant", "gas_plant"}


def test_stochastic_efficiency_keeps_other_units(tmp_path_factory) -> None:
    """A silent rank-3 efficiency on coal: every other unit's slope is the one of the run where coal's values are
    scalars (before the fix every unit's efficiency was 1)."""
    step = _first(_eff(tmp_path_factory, False))
    ref = _first(_eff(tmp_path_factory, None))
    for field in ("p_slope", "p_section"):
        mine = {k: v for k, v in _cells(step, field, ["p"]).items() if k[0] != "coal_plant"}
        want = {k: v for k, v in _cells(ref, field, ["p"]).items() if k[0] != "coal_plant"}
        assert mine and mine == want, field
    assert any(k[0] == "gas_plant" for k in _cells(step, "p_slope", ["p"]))


def test_labelled_equals_silent_efficiency(tmp_path_factory) -> None:
    lab, sil = _first(_eff(tmp_path_factory, True)), _first(_eff(tmp_path_factory, False))
    assert _rel(float(lab.solution.obj), float(sil.solution.obj)) < 1e-9
    for field in ("p_min_load", "p_section", "p_slope"):
        assert _cells(lab, field, ["p"]) == _cells(sil, field, ["p"]), field


# ---------------------------------------------------------------------------
# min_load on an online unit without min_load_efficiency (the emit's min_load domain covers online units)


def _ml_online(tmp_path_factory, scenario: str, labelled: bool) -> dict:
    at = AT_SINGLE if scenario == SINGLE else AT_ROLL
    vals = [["unit", u, "startup_method", b64("linear", "str")] for u in ("coal_plant", "gas_plant")]
    vals += [["unit", "coal_plant", "min_load", _r3(0.5, at, labelled)],
             ["unit", "gas_plant", "min_load", scalar(0.3)]]
    return _run(tmp_path_factory, scenario, vals, STOCH_UNIT)


@pytest.mark.parametrize("scenario", [SINGLE, ROLLING])
@pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "silent"])
def test_stochastic_min_load_online_constant_efficiency(tmp_path_factory, scenario: str, labelled: bool) -> None:
    """coal is online with a constant efficiency: its stochastic min_load reaches the minFlow_minload floor (every
    cell carries the fold, branch periods their branch value) and ``process_minload`` contains it."""
    steps = _ml_online(tmp_path_factory, scenario, labelled)
    for step in _stochastic_steps(steps)[:3]:
        cells = _cells(step, "p_min_load", ["p"])
        assert all(("coal_plant", d, t) in cells for d, t in _dt(step))
        periods = {d for d, _t in _dt(step)} - {ANCHOR}
        assert periods, "no branch periods"
        _assert_branch_values(cells, ("coal_plant",), 0.5, periods=periods)
        mins = utf8(step.flex_data.process_minload)["p"].to_list()
        assert "coal_plant" in mins and len(mins) == len(set(mins))


# ---------------------------------------------------------------------------
# Rolling


def _roll_values(labelled: bool) -> list:
    return ([["unit", "coal_plant", "availability", _r3(1.0, AT_ROLL, labelled)],
             ["unit", "gas_plant", "availability", _ptmap(0.5, T168)],
             ["unit__inputNode", COAL_IN, "other_operational_cost", _r3(10.0, AT_ROLL, labelled)],
             ["unit__inputNode", GAS_IN, "other_operational_cost", _ptmap(5.0, T168)],
             ["node", "demand_node", "penalty_up", _r3(10000.0, AT_ROLL, labelled)],
             ["node", "downriver", "penalty_up", _ptmap(7000.0, T168)]]
            + _reserve_values(_r3(100.0, AT_ROLL, labelled), _ptmap(50.0, T168)))


ROLL_FIELDS = {k: SINGLE_FIELDS[k] for k in ("p_process_availability", "p_pdt_varCost_source", "p_penalty_up",
                                             "pdtReserve_upDown_group_reservation")}


def _rolling(tmp_path_factory, labelled: bool, *, warm: bool = True) -> dict:
    return _run(tmp_path_factory, ROLLING, _roll_values(labelled),
                STOCH_UNIT + STOCH_NODE + _reserve_entities(second=True), warm=warm)


def _rolls(steps: dict) -> list:
    rolls = [s for k, s in steps.items() if "_roll_" in k and "__realized_dispatch" not in k]
    assert len(rolls) == 7, list(steps)
    return rolls


@pytest.mark.parametrize("labelled", [True, False], ids=["labelled", "silent"])
def test_rolling_stochastic_time_params(tmp_path_factory, labelled: bool) -> None:
    """Every roll solves; the rank-3 entity carries its branch values on the roll's branch periods; the second
    entity keeps its period × time map on every (d, t) of every roll."""
    for step in _rolls(_rolling(tmp_path_factory, labelled)):
        periods = {d for d, _t in _dt(step)} - {ANCHOR}
        assert periods
        for field, (keys, ent, other, mult, base, *_rest) in ROLL_FIELDS.items():
            cells = _cells(step, field, keys)
            _assert_branch_values(cells, ent, mult, periods=periods)
            assert any(k[:len(keys)] == ent for k in cells), field
            for d, t in _dt(step):
                assert cells[(*other, d, t)] == pytest.approx(_pt_value(base, t), rel=1e-12), (field, d, t)


def test_rolling_labelled_equals_silent_and_warm_equals_cold(tmp_path_factory) -> None:
    lab = _rolling(tmp_path_factory, True)
    sil = _rolling(tmp_path_factory, False)
    cold = _rolling(tmp_path_factory, True, warm=False)
    assert list(lab) == list(sil) == list(cold)
    for name in lab:
        assert _rel(float(lab[name].solution.obj), float(sil[name].solution.obj)) < 1e-9, name
        assert _rel(float(lab[name].solution.obj), float(cold[name].solution.obj)) < 1e-9, name
        for field, (keys, *_rest) in ROLL_FIELDS.items():
            assert _cells(lab[name], field, keys) == _cells(sil[name], field, keys), (name, field)


# ---------------------------------------------------------------------------
# A rank-3 map equal everywhere to v versus the scalar v


def _const(tmp_path_factory, scenario: str, rank3: bool) -> dict:
    at = AT_SINGLE if scenario == SINGLE else AT_ROLL
    if rank3:
        avail = stoch_map({b: 0.7 for b in BRANCHES}, at, labelled=True)
        cost = stoch_map({b: 4.0 for b in BRANCHES}, at, labelled=True)
    else:
        avail, cost = scalar(0.7), scalar(4.0)
    return _run(tmp_path_factory, scenario, [["unit", "coal_plant", "availability", avail],
                                             ["unit__inputNode", COAL_IN, "other_operational_cost", cost]],
                STOCH_UNIT)


def test_rank3_constant_equals_scalar_single_branch_periods(tmp_path_factory) -> None:
    """Single stochastic solve, a forecasts-only fan: every cell of the four forecast branch periods equals the
    scalar run's and so does the objective.  The period label ``period1`` has no dispatch, so neither run has a
    cell on it."""
    r3, sc = _first(_const(tmp_path_factory, SINGLE, True)), _first(_const(tmp_path_factory, SINGLE, False))
    for field, keys, ent in (("p_process_availability", ["p"], ("coal_plant",)),
                             ("p_pdt_varCost_source", ["p", "source"], tuple(COAL_IN))):
        a = {k: v for k, v in _cells(r3, field, keys).items() if k[:len(keys)] == ent}
        b = {k: v for k, v in _cells(sc, field, keys).items() if k[:len(keys)] == ent}
        assert {k[-2] for k in a} == set(BRANCH_PERIODS), field
        assert a == b, field
    assert _rel(float(r3.solution.obj), float(sc.solution.obj)) < 1e-9


def test_rank3_constant_equals_scalar_rolling(tmp_path_factory) -> None:
    """Rolling: every cell (the anchor takes the realized branch) and every roll's objective equal the scalar's."""
    r3, sc = _const(tmp_path_factory, ROLLING, True), _const(tmp_path_factory, ROLLING, False)
    assert list(r3) == list(sc)
    for name in r3:
        for field, keys in (("p_process_availability", ["p"]), ("p_pdt_varCost_source", ["p", "source"])):
            a = {k: v for k, v in _cells(r3[name], field, keys).items() if k[0] == "coal_plant"}
            b = {k: v for k, v in _cells(sc[name], field, keys).items() if k[0] == "coal_plant"}
            assert a == b, (name, field)
        assert _rel(float(r3[name].solution.obj), float(sc[name].solution.obj)) < 1e-9, name


# ---------------------------------------------------------------------------
# An entity outside every include_stochastics group


def test_rank3_outside_stochastic_group_reads_realized_branch(tmp_path_factory) -> None:
    """``gas_plant`` (in no ``include_stochastics`` group) authors a rank-3 availability.  Outside such a group
    every branch period reads the single value of the realized branch (docs/how_to/stochastics.md "Single solve
    stochastics"): ``1_week_rolling_wind`` designates branch ``realized`` (0.9) ``realized: yes`` in every
    roll, so every cell of every stochastic roll is 0.9."""
    steps = _run(tmp_path_factory, ROLLING, [["unit", "gas_plant", "availability", _r3(1.0, AT_ROLL, True)]], [])
    for step in _stochastic_steps(steps)[:3]:
        cells = {k: v for k, v in _cells(step, "p_process_availability", ["p"]).items() if k[0] == "gas_plant"}
        assert len(cells) == len(_dt(step))
        assert set(cells.values()) == {BR["realized"]}


def test_rank3_outside_stochastic_group_forecasts_only_raises(tmp_path_factory) -> None:
    """The same value in ``2_day_stochastic_dispatch``, a forecasts-only fan: there is no realized branch for
    the ungrouped entity to read, so the solve raises naming the entity instead of silently taking the class
    default (1.0) on every branch period."""
    with pytest.raises(FlexToolConfigError,
                       match="unit or connection 'gas_plant', parameter 'availability'.*include_stochastics"):
        _run(tmp_path_factory, SINGLE, [["unit", "gas_plant", "availability", _r3(1.0, AT_SINGLE, True)]], [])


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "ungrouped"])
def test_rank3_reservation_forecasts_only_fan(tmp_path_factory, grouped: bool) -> None:
    """A rank-3 ``reservation`` of ``res_g`` in ``2_day_stochastic_dispatch``, a forecasts-only fan.  With
    ``include_stochastics`` on ``res_g`` each forecast reads its own branch (100 x the branch value).  Without
    it the group would read the realized branch, which this fan does not have, so the solve raises naming the
    reserve instead of silently taking the reservation default 0 on every branch period."""
    values = _reserve_values(_r3(100.0, AT_SINGLE, True), None, grouped=grouped)
    entities = _reserve_entities(second=False)
    if not grouped:
        with pytest.raises(FlexToolConfigError, match=r"reserve 'r1' \(up\) of group 'res_g', parameter "
                           r"'reservation'.*Set include_stochastics on group 'res_g'"):
            _run(tmp_path_factory, SINGLE, values, entities)
        return
    cells = _cells(_first(_run(tmp_path_factory, SINGLE, values, entities)),
                   "pdtReserve_upDown_group_reservation", ["r", "ud", "g"])
    assert {k[-2] for k in cells} == set(BRANCH_PERIODS)
    _assert_branch_values(cells, ("r1", "up", "res_g"), 100.0)


# ---------------------------------------------------------------------------
# A stochastic-time value authoring only SOME of a cohort's active branches


def test_rank3_partial_branch_coverage_raises(tmp_path_factory) -> None:
    """A stochastic-time value that authors SOME of a reveal cohort's active branches but not ALL of them is
    an input error (owner decision #82, point 2) -- never a silent fallback to the class default for the
    branches left out.  ``2_day_stochastic_dispatch``'s ``period1`` cohort has 4 active branches (``realized``,
    ``upper``, ``lower``, ``mid``; none flagged ``realized: yes``); authoring only 2 of them must raise."""
    partial = stoch_map({"upper": 0.9, "lower": 0.7}, AT_SINGLE, labelled=True)
    with pytest.raises(FlexToolConfigError, match="missing"):
        _run(tmp_path_factory, SINGLE, [["unit", "coal_plant", "availability", partial]], STOCH_UNIT)


# ---------------------------------------------------------------------------
# A solve without its own stochastic branches


@pytest.mark.parametrize("single", [True, False], ids=["single", "rolls"])
def test_rank3_in_deterministic_solve_raises_no_realized_branch(tmp_path_factory, single: bool) -> None:
    """``1_week_rolling_wind`` without its stochastic data at all: no branch periods and no
    ``stochastic_branches`` row anywhere, so there is no realized-branch designation for
    ``coal_plant``'s rank-3 availability to take its value from.  Owner decision #82(4) raises a
    :class:`FlexToolConfigError` naming the entity and parameter instead of silently falling back
    to the class default (as a pre-fix run used to, with a WARNING)."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.input._solve_state import FlexToolConfigError
    from tests.model.test_rolling_period_maps import ROLL_CHAIN, _deterministic_rolling_json

    values = [["unit", "coal_plant", "availability", _r3(1.0, AT_ROLL, True)]]
    if single:
        values.append(["solve", "1week_rolling", "solve_mode", b64("single_solve", "str")])
    alt = f"det_r3_{single}"
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=ROLL_CHAIN,
                        fixture_json=_deterministic_rolling_json(tmp_path_factory, alt))
    with pytest.raises(FlexToolConfigError, match="no realized branch"):
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)


def test_rank3_dispatch_sibling_uses_realized_branch(tmp_path_factory) -> None:
    """A rolling + ``separate_dispatch`` solve's auto-spawned realized-dispatch sibling has no
    ``stochastic_branches`` / branch-anchor pairs of its own (it is deterministic -- ``v_online`` is pinned
    from its UC parent), but its parent's realized branch is projected onto it
    (``_run_model.project_realized_branch_time_branch_rows``).  A rank-3 availability authored on
    ``coal_plant`` must resolve to the REALIZED branch's own value on the sibling (owner decision #82(4):
    a realized branch projected from elsewhere still counts; only the TRUE no-realized-branch-anywhere case
    (above) raises), not the class default (1.0) and not a raise."""
    from flextool.engine import run_chain_from_db

    at = {f"t{s:04d}": [f"t{i:04d}" for i in range(s, 73)] for s in (1, 25, 49)}
    avail = stoch_map({"realized": 0.73, "upper": 0.81, "lower": 0.62, "mid": 0.95}, at, labelled=True)
    alt = "disp_sibling_r3"
    url, scen = make_db(tmp_path_factory, alt=alt, values=[["unit", "coal_plant", "availability", avail]],
                        base_chain=_chain(ROLLING_SEPARATE_DISPATCH_JSON, "rolling_separate_dispatch"),
                        entities=[["group__unit", ["add_stochastics", "coal_plant"], None]],
                        fixture_json=ROLLING_SEPARATE_DISPATCH_JSON)
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"),
                              keep_solutions=True, warm=False)
    dispatch = [s for name, s in steps.items() if "realized_dispatch" in name]
    assert dispatch, list(steps)
    for step in dispatch:
        coal = {k: v for k, v in _cells(step, "p_process_availability", ["p"]).items() if k[0] == "coal_plant"}
        assert coal, "coal_plant missing from the dispatch sibling's availability"
        assert set(coal.values()) == {0.73}, coal


# ---------------------------------------------------------------------------
# Stochastic-time profiles: profile.profile resolves through its own cascade
# (flextool.engine.preprocess.params._profiles), not the pbt_* STOCHASTIC_TIME_PARAMS path above.


def _wind_branch_series(fixture_json, alt: str, branch: str, analysis_time: str) -> dict:
    """Branch *branch*'s own authored ``t -> value`` series at *analysis_time* of the ``wind`` profile in
    alternative *alt* (``2day`` for the single solve, ``1week_rolling`` for the rolls) -- the independent oracle
    of the profile tests below, read straight from the fixture JSON, never from the engine's own resolved
    output."""
    import base64

    for cls, ent, param, value, a in json.loads(fixture_json.read_text())["parameter_values"]:
        if (cls, ent, param, a) != ("profile", "wind", "profile", alt):
            continue
        raw = json.loads(base64.b64decode(value[0]).decode())
        for b, branch_map in raw["data"]:
            if b == branch:
                for ts, ts_map in branch_map["data"]:
                    if ts == analysis_time:
                        return dict(ts_map["data"])
    raise AssertionError(f"profile.wind (alternative {alt}, branch {branch}, {analysis_time}) not in the fixture")


def _wind_cells(step) -> dict:
    """``d -> {t: value}`` of the ``wind`` profile of *step*."""
    out: dict = {}
    for (f, d, t), v in _cells(step, "p_profile_value", ["f"]).items():
        if f == "wind":
            out.setdefault(d, {})[t] = v
    return out


def test_rank3_profile_realization_solve_takes_realized_branch(tmp_path_factory) -> None:
    """``1_week_rolling_wind`` authors a rank-3 ``wind`` profile and designates branch ``realized`` in every
    roll.  The period label ``period1`` has no dispatch in the stochastic rolls (no profile rows there); each
    realization roll dispatches ``period1`` with the realized branch's own series for that roll's analysis time
    (its first step), read straight from the fixture JSON."""
    steps = _run(tmp_path_factory, ROLLING, [], [])
    for step in _stochastic_steps(steps):
        assert ANCHOR not in _wind_cells(step), "the period label has profile rows in a stochastic roll"
    realization = [s for k, s in steps.items() if is_realization(k)]
    assert len(realization) == 7, list(steps)
    for step in realization:
        cells = _wind_cells(step)
        assert set(cells) == {ANCHOR}, sorted(cells)
        analysis_time = min(cells[ANCHOR])
        want = _wind_branch_series(STOCHASTICS_JSON, "1week_rolling", "realized", analysis_time)
        assert cells[ANCHOR] == {t: want[t] for t in cells[ANCHOR]}, analysis_time


def test_rank3_profile_forecasts_only_fan_solves(tmp_path_factory) -> None:
    """``2_day_stochastic_dispatch``'s forecasts-only fan (the four forecasts ``realized`` / ``upper`` /
    ``lower`` / ``mid``, none designated) with its rank-3 ``wind`` profile solves: every forecast branch period
    takes its own branch's authored series, the period label ``period1`` has no profile rows (no dispatch, so it
    needs no value of its own) and no realization solve is added."""
    steps = _run(tmp_path_factory, SINGLE, [], [])
    assert not any(is_realization(k) for k in steps), "a forecasts-only fan got a realization solve"
    cells = _wind_cells(_first(steps))
    assert ANCHOR not in cells
    for branch in BRANCHES:
        want = _wind_branch_series(STOCHASTICS_JSON, "2day", branch, "t0001")
        got = cells.get(f"{ANCHOR}_{branch}")
        assert got, f"wind profile missing on {ANCHOR}_{branch}"
        assert got == {t: want[t] for t in got}, f"{ANCHOR}_{branch}: profile != the branch's own series"
