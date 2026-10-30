"""Node balance penalties keep their own value or the default next to maps (#62).

The balance slacks of every node in ``nodeBalance ∪ nodeBalancePeriod`` are priced in the objective:

    Σ vq_state_up[n,d,t] · penalty_up[n,d,t] · w_op[d,t]  (+ the same for down)

``penalty`` must be defined on every slack cell:

    penalty[n, d, t] = authored value of n at (d, t)      scalar, period, time or period × time map
                     = authored[n, anchor(d), t]          d a stochastic branch period without own row
                     = schema default (10 000)            n has no value, or its map leaves out d / t

Before the fix, as soon as one node authored its penalty as a map, the DB-direct reader returned the explicit rows
only and joined them by column name: nodes with a constant penalty or the default lost their rows, so their slack
was free (the objective term is an inner-join product).  A stochastic-time (rank 3) penalty crashed the single
solve, and on rolling sub-solves a time or period × time ``penalty_up`` fell back to the default (the emit seed's
``pt_node`` / ``pbt_node`` inputs did not list ``penalty_up`` and have no period × time tier).

The criterion: a map gives the same parameters and objective as the equivalent scalar, every balance node has a
penalty on every ``(d, t)`` of the solve, and a node without a value of its own gets the default.

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``network_coal_wind`` (``tests/fixtures/tests.json``): nodes west / east / north with authored penalties.
- ``2_day_stochastic_dispatch`` (``stochastics.json``): ``period1`` and four branch periods; ``demand_node``
  authors its penalties, ``downriver`` authors ``penalty_down = 0`` only, ``hydro_reservoir`` nothing.
- ``stoch`` (``stoch_two_period.json``): two anchor periods with branches.
- ``1_week_rolling_wind`` (``stochastics.json``): 7 stochastic rolls.
- ``rolling_separate_dispatch`` (``rolling_separate_dispatch.json``): stochastic rolls plus the realized-dispatch
  sibling solves.
- ``y2020_2day_dispatch`` of ``tests.json`` rolled 24 h / 12 h (west + heat): warm-start vehicle.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import TIMESTEPS, b64, make_db, utf8

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TESTS_JSON = FIXTURES / "tests.json"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"
TWO_PERIOD_JSON = FIXTURES / "stoch_two_period.json"
SEPARATE_DISPATCH_JSON = FIXTURES / "rolling_separate_dispatch.json"

DEFAULT = 10000.0
ANCHOR = "period1"
BRANCHES = ("realized", "upper", "lower", "mid")
BRANCH_PERIODS = tuple(f"period1_{b}" for b in BRANCHES)
ALL_PERIODS = (ANCHOR,) + BRANCH_PERIODS
T48 = [f"t{i:04d}" for i in range(1, 49)]
T168 = [f"t{i:04d}" for i in range(1, 169)]
STOCH_NODES = ("demand_node", "downriver", "hydro_reservoir")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Value / database helpers


def scalar(v: float) -> list:
    return b64(float(v), "float")


def pmap(values: dict, index_name: str | None = "period") -> list:
    """1-D map (``index_name=None``: Spine's silent ``x`` index)."""
    m = {"index_type": "str", "data": [[k, v] for k, v in values.items()]}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


def tmap(tv: dict) -> list:
    return pmap(tv, index_name="time")


def ptmap(per: dict) -> list:
    """2-D map period -> time -> value."""
    return b64({"index_type": "str", "index_name": "period", "data": [
        [d, {"type": "map", "index_type": "str", "index_name": "time",
             "data": [[t, v] for t, v in tv.items()]}] for d, tv in per.items()]}, "map")


def stoch_map(branch_values: dict, analysis_times: dict, *, labelled: bool) -> list:
    """3-D stochastic-time map branch -> analysis time -> time -> value (constant per branch).

    ``analysis_times`` maps each analysis time to its timesteps.  ``labelled``: index names
    ``branch / analysis_time / time`` (as ``docs/how_to.md`` names them); otherwise Spine's silent ``x``.
    """
    def level(name: str, data: list) -> dict:
        m = {"type": "map", "index_type": "str", "data": data}
        if labelled:
            m["index_name"] = name
        return m
    top = level("branch", [
        [b, level("analysis_time", [[at, level("time", [[t, v] for t in ts])]
                                    for at, ts in analysis_times.items()])]
        for b, v in branch_values.items()])
    top.pop("type")
    return b64(top, "map")


def realized_branch_override(solve: str = "2day_dispatch", analysis_time: str = "t0001") -> list:
    """Flag ``solve``'s ``realized`` branch ``realized: yes`` on ``ANCHOR``.

    The shared ``stochastics.json`` fixture leaves every one of ``2day_dispatch``'s branches
    ``realized: no`` -- a legitimate, separately-tested forecasts-only configuration many
    unrelated tests build on -- but a stochastic-time (rank-3) value authored in a test that uses
    this solve needs a committed branch to resolve the anchor's own value from (owner decision
    #82: no reference-forecast fallback).  Shared with ``tests/model/test_stochastic_time_params.py``,
    whose rolling solve names its analysis time differently.
    """
    branch_map = {
        "type": "map", "index_type": "str",
        "data": [[b, {"type": "map", "index_type": "str",
                      "data": [[analysis_time, {"type": "map", "index_type": "str",
                                                "data": [["yes" if b == "realized" else "no", 1.0]]}]]}]
                 for b in BRANCHES],
    }
    value = {"index_type": "str", "data": [[ANCHOR, branch_map]]}
    return [["solve", solve, "stochastic_branches", b64(value, "map")]]


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


_RUNS: dict = {}


def _run(tmp_path_factory, fixture_json: Path, scenario: str, values: list, *, entities: list | None = None,
         warm: bool = True) -> dict:
    """Run ``scenario`` of ``fixture_json`` + ``values`` (memoised per module)."""
    from flextool.engine import run_chain_from_db

    key = json.dumps([str(fixture_json), scenario, values, entities, warm])
    if key not in _RUNS:
        alt = f"c9_{len(_RUNS)}"
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_chain(fixture_json, scenario),
                            entities=entities, fixture_json=fixture_json)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"),
                                  keep_solutions=True, warm=warm)
        assert steps, "run_chain_from_db produced no orchestration steps"
        for step in steps.values():
            assert step.solution is not None, "solve produced no solution"
        _RUNS[key] = steps
    return _RUNS[key]


def _frame(x) -> pl.DataFrame:
    assert x is not None, "parameter is None"
    f = getattr(x, "frame", x)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    return utf8(f)


def _cells(step, field: str) -> dict[tuple[str, str, str], float]:
    """``(n, d, t) -> value``; fails on a duplicate cell."""
    f = _frame(getattr(step.flex_data, field)).select("n", "d", "t", "value")
    assert not f.select("n", "d", "t").is_duplicated().any(), f"{field}: duplicate (n, d, t) cells"
    return {(n, d, t): float(v) for n, d, t, v in f.iter_rows()}


def _dt(step) -> list[tuple[str, str]]:
    return [(d, t) for d, t in utf8(step.flex_data.dt).select("d", "t").iter_rows()]


def _assert_dense(step, field: str, nodes, expected) -> None:
    """Every ``(n, d, t)`` of ``nodes`` × the solve's dt has a value; ``expected(n, d, t)`` gives it."""
    cells = _cells(step, field)
    for n in nodes:
        for d, t in _dt(step):
            assert (n, d, t) in cells, f"{field}: no value for ({n}, {d}, {t})"
            want = expected(n, d, t)
            assert cells[(n, d, t)] == pytest.approx(want, rel=1e-12), (
                f"{field}[{n}, {d}, {t}] = {cells[(n, d, t)]}, expected {want}")


def _last(steps: dict):
    return list(steps.values())[-1]


def _obj(steps: dict) -> float:
    return float(_last(steps).solution.obj)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


# ---------------------------------------------------------------------------
# Deterministic single solve


def _det(tmp_path_factory, east):
    return _run(tmp_path_factory, TESTS_JSON, "network_coal_wind", [["node", "east", "penalty_up", east]])


def test_det_map_keeps_other_nodes_penalties(tmp_path_factory) -> None:
    """``east.penalty_up`` as a period map next to scalar west / north: everybody keeps a penalty.

    Before the fix west and north lost their rows and the objective collapsed to 0.
    """
    steps = _det(tmp_path_factory, pmap({"p2020": 7000.0}))
    ref = _det(tmp_path_factory, scalar(7000.0))
    want = {"east": 7000.0, "north": 1000.0, "west": 900.0}
    _assert_dense(_last(steps), "p_penalty_up", want, lambda n, d, t: want[n])
    assert _cells(_last(steps), "p_penalty_up") == _cells(_last(ref), "p_penalty_up")
    assert _obj(ref) > 1e8
    assert _rel(_obj(steps), _obj(ref)) < 1e-9


# ---------------------------------------------------------------------------
# Stochastic single solve (2-day)


def _stoch(tmp_path_factory, up, down=None, *, extra: list | None = None, entities: list | None = None):
    values = [["node", "demand_node", "penalty_up", up]]
    if down is not None:
        values.append(["node", "demand_node", "penalty_down", down])
    values += extra or []
    return _run(tmp_path_factory, STOCHASTICS_JSON, "2_day_stochastic_dispatch", values, entities=entities)


def test_stoch_map_default_for_nodes_without_value(tmp_path_factory) -> None:
    """Period maps keyed by the anchor on ``demand_node``: the branch periods inherit them, ``hydro_reservoir``
    (no value) gets the default, ``downriver``'s explicit ``penalty_down = 0`` stays 0 (not the default).
    """
    steps = _stoch(tmp_path_factory, pmap({ANCHOR: 6000.0}), pmap({ANCHOR: 4000.0}))
    ref = _stoch(tmp_path_factory, scalar(6000.0), scalar(4000.0))
    step = _last(steps)
    up = {"demand_node": 6000.0, "downriver": DEFAULT, "hydro_reservoir": DEFAULT}
    down = {"demand_node": 4000.0, "downriver": 0.0, "hydro_reservoir": DEFAULT}
    assert {d for d, _t in _dt(step)} == set(ALL_PERIODS)
    _assert_dense(step, "p_penalty_up", STOCH_NODES, lambda n, d, t: up[n])
    _assert_dense(step, "p_penalty_down", STOCH_NODES, lambda n, d, t: down[n])
    assert _cells(step, "p_penalty_up") == _cells(_last(ref), "p_penalty_up")
    assert _cells(step, "p_penalty_down") == _cells(_last(ref), "p_penalty_down")
    assert _rel(_obj(steps), _obj(ref)) < 1e-9


def test_time_and_period_time_maps(tmp_path_factory) -> None:
    """A time map and a period × time map on ``demand_node``: its per-step values on every period, the default
    for the others; a constant time map equals the scalar."""
    curve = {t: (6000.0 if i % 2 else 6500.0) for i, t in enumerate(T48)}
    for value in (tmap(curve), ptmap({ANCHOR: curve})):
        step = _last(_stoch(tmp_path_factory, value))
        _assert_dense(step, "p_penalty_up", STOCH_NODES,
                      lambda n, d, t: curve[t] if n == "demand_node" else DEFAULT)
    const = _stoch(tmp_path_factory, tmap({t: 6000.0 for t in T48}))
    ref = _stoch(tmp_path_factory, scalar(6000.0))
    assert _cells(_last(const), "p_penalty_up") == _cells(_last(ref), "p_penalty_up")
    assert _rel(_obj(const), _obj(ref)) < 1e-9


def test_silent_label_map(tmp_path_factory) -> None:
    """A period map without ``index_name`` (Spine's silent ``x``) reads like the labelled one."""
    silent = _stoch(tmp_path_factory, pmap({ANCHOR: 6000.0}, index_name=None))
    labelled = _stoch(tmp_path_factory, pmap({ANCHOR: 6000.0}))
    _assert_dense(_last(silent), "p_penalty_up", STOCH_NODES,
                  lambda n, d, t: 6000.0 if n == "demand_node" else DEFAULT)
    assert _cells(_last(silent), "p_penalty_up") == _cells(_last(labelled), "p_penalty_up")
    assert _rel(_obj(silent), _obj(labelled)) < 1e-9


def test_partial_period_map_gets_default(tmp_path_factory) -> None:
    """``city.penalty_up = {p2035: 8000}``: p2035 and its branches 8000, p2040 and its branches the default."""
    step = _last(_run(tmp_path_factory, TWO_PERIOD_JSON, "stoch",
                      [["node", "city", "penalty_up", pmap({"p2035": 8000.0})]]))
    periods = {d for d, _t in _dt(step)}
    assert {"p2035", "p2035_high", "p2035_low", "p2040", "p2040_high", "p2040_low"} <= periods
    _assert_dense(step, "p_penalty_up", ("city",),
                  lambda n, d, t: 8000.0 if d.startswith("p2035") else DEFAULT)


def test_map_next_to_penalty_method_off(tmp_path_factory) -> None:
    """``hydro_reservoir.penalty_method = off`` next to a map on ``demand_node``: no slack for the off node, and
    the same solve as with the scalar."""
    off = [["node", "hydro_reservoir", "penalty_method", b64("off", "str")]]
    steps = _stoch(tmp_path_factory, pmap({ANCHOR: 6000.0}), extra=off)
    ref = _stoch(tmp_path_factory, scalar(6000.0), extra=off)
    for name in ("vq_state_up", "vq_state_down"):
        sol = utf8(_last(steps).solution.value(name))
        assert sol.height > 0
        assert "hydro_reservoir" not in set(sol["n"]), f"{name} has a column for the off node"
    _assert_dense(_last(steps), "p_penalty_up", ("demand_node", "downriver"),
                  lambda n, d, t: 6000.0 if n == "demand_node" else DEFAULT)
    assert _rel(_obj(steps), _obj(ref)) < 1e-9


@pytest.mark.parametrize("labelled", [False, True], ids=["silent_labels", "branch_labels"])
def test_stochastic_time_penalty(tmp_path_factory, labelled: bool) -> None:
    """Rank-3 ``penalty_up`` / ``penalty_down`` on ``demand_node`` (in the stochastic group): each branch period
    carries its branch's value, the anchor carries the realized branch's value, the other nodes keep theirs
    (48 rows per (n, d), no duplicates).

    Before the fix the silent-label case crashed (``p_penalty_*`` None) and the labelled case got every branch
    value on every cell (192 rows per (n, d)) with the other nodes' slack free.  The ``realized`` branch is
    designated ``realized: yes`` (owner decision #82: a forecasts-only fan authoring a rank-3 value with no
    committed branch is a configuration error) -- the anchor then takes that branch's own value rather than
    the schema default.
    """
    ups = dict(zip(BRANCHES, (6000.0, 6100.0, 6200.0, 6300.0)))
    downs = dict(zip(BRANCHES, (4000.0, 4100.0, 4200.0, 4300.0)))
    at = {"t0001": T48}
    steps = _stoch(tmp_path_factory, stoch_map(ups, at, labelled=labelled), stoch_map(downs, at, labelled=labelled),
                   extra=realized_branch_override(),
                   entities=[["group__node", ["add_stochastics", "demand_node"], None]])
    step = _last(steps)

    def want(table, other):
        def f(n, d, t):
            if n != "demand_node":
                return other[n]
            if d in BRANCH_PERIODS:
                return table[d.removeprefix("period1_")]
            return table["realized"]
        return f
    _assert_dense(step, "p_penalty_up", STOCH_NODES, want(ups, {"downriver": DEFAULT, "hydro_reservoir": DEFAULT}))
    _assert_dense(step, "p_penalty_down", STOCH_NODES, want(downs, {"downriver": 0.0, "hydro_reservoir": DEFAULT}))
    for field in ("p_penalty_up", "p_penalty_down"):
        counts = _frame(getattr(step.flex_data, field)).group_by("n", "d").len()
        assert set(counts["len"]) == {48}, f"{field}: rows per (n, d) {sorted(set(counts['len']))}"


# ---------------------------------------------------------------------------
# Rolling (synthetic sub-solves)


def _rolls(steps: dict) -> list:
    """The roll steps; their solve names carry ``_roll_`` (the synthetic early return was taken)."""
    rolls = [s for k, s in steps.items() if "_roll_" in k]
    assert len(rolls) >= 3, f"expected roll sub-solves, got {list(steps)}"
    return rolls


def _roll(tmp_path_factory, value, *, entities: list | None = None, warm: bool = True) -> dict:
    return _run(tmp_path_factory, STOCHASTICS_JSON, "1_week_rolling_wind",
                [["node", "demand_node", "penalty_up", value]], entities=entities, warm=warm)


def test_rolling_time_and_period_time_maps(tmp_path_factory) -> None:
    """A time map and a period × time map of ``demand_node.penalty_up`` reach every roll on its own steps; the
    other nodes keep the default.  A period map (correct before the fix too) is the guard."""
    curve = {t: (6000.0 if i % 2 else 6500.0) for i, t in enumerate(T168)}
    for value in (tmap(curve), ptmap({ANCHOR: curve})):
        for step in _rolls(_roll(tmp_path_factory, value)):
            _assert_dense(step, "p_penalty_up", STOCH_NODES,
                          lambda n, d, t: curve[t] if n == "demand_node" else DEFAULT)
    for step in _rolls(_roll(tmp_path_factory, pmap({ANCHOR: 6000.0}))):
        _assert_dense(step, "p_penalty_up", STOCH_NODES,
                      lambda n, d, t: 6000.0 if n == "demand_node" else DEFAULT)


@pytest.mark.parametrize("labelled", [False, True], ids=["silent_labels", "branch_labels"])
def test_rolling_stochastic_time_penalty(tmp_path_factory, labelled: bool) -> None:
    """Rank-3 ``penalty_up`` with one analysis time per roll start: the rolls carry each branch's value on its
    branch period (the emit seed's pbt fold; ``pbt_node`` did not list ``penalty_up`` before the fix)."""
    ups = dict(zip(BRANCHES, (6000.0, 6100.0, 6200.0, 6300.0)))
    at = {f"t{s:04d}": T168 for s in range(1, 169, 24)}
    rolls = _rolls(_roll(tmp_path_factory, stoch_map(ups, at, labelled=labelled),
                         entities=[["group__node", ["add_stochastics", "demand_node"], None]]))
    for step in rolls[:2]:
        cells = _cells(step, "p_penalty_up")
        branch_periods = {d for _n, d, _t in cells if d.startswith("period1_")}
        assert branch_periods, "no branch periods in the roll"
        for (n, d, _t), v in cells.items():
            if n == "demand_node" and d in branch_periods:
                assert v == ups[d.removeprefix("period1_")], f"({n}, {d}): {v}"
            elif n != "demand_node":
                assert v == DEFAULT, f"({n}, {d}): {v}"


def test_realized_dispatch_sibling_covers_every_node(tmp_path_factory) -> None:
    """``rolling_separate_dispatch``: a time map on ``demand_node`` reaches the rolls and the realized-dispatch
    solves, and every balance node keeps its own penalty or the default there."""
    curve = {t: (6000.0 if i % 2 else 6500.0) for i, t in enumerate(T168)}
    steps = _run(tmp_path_factory, SEPARATE_DISPATCH_JSON, "rolling_separate_dispatch",
                 [["node", "demand_node", "penalty_up", tmap(curve)]])
    dispatch = [k for k in steps if "realized_dispatch" in k]
    assert dispatch, f"no realized-dispatch solves in {list(steps)}"
    for step in steps.values():
        nodes = set(utf8(step.flex_data.nodeBalance)["n"])
        assert nodes == {"demand_node", "downriver"}, nodes
        _assert_dense(step, "p_penalty_up", nodes,
                      lambda n, d, t: curve[t] if n == "demand_node" else DEFAULT)


# ---------------------------------------------------------------------------
# Warm start and the run-level memo


WARM_VALUES = [
    ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")],
    ["node", "west", "inflow", b64(-1000.0, "float")],
    ["node", "heat", "inflow", b64(0.0, "float")],
]
WARM_CHAIN_JSON = TESTS_JSON


def _warm_run(tmp_path_factory, curve: dict, warm: bool) -> dict:
    from flextool.engine import run_chain_from_db

    key = json.dumps(["warm", curve, warm])
    if key not in _RUNS:
        alt = f"c9w_{len(_RUNS)}"
        url, scen = make_db(tmp_path_factory, alt=alt,
                            values=WARM_VALUES + [["node", "west", "penalty_up", tmap(curve)]],
                            base_chain=["init", "west", "coal_chp", "heat"])
        _RUNS[key] = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"),
                                       keep_solutions=True, warm=warm)
    return _RUNS[key]


def test_warm_rolling_time_varying_penalty_matches_cold(tmp_path_factory) -> None:
    """A time-varying ``penalty_up`` that prices real slack (cheap on odd hours) differs between rolls: every warm
    roll's objective equals the cold one.  No roll reuses the previous roll's LP (its ``dt`` labels shift), so the
    changed penalty cells are always built, never pushed into a stale LP."""
    varying = {t: (5.0 + 0.1 * i) if i % 2 else 5000.0 for i, t in enumerate(TIMESTEPS)}
    cold = _warm_run(tmp_path_factory, varying, warm=False)
    hot = _warm_run(tmp_path_factory, varying, warm=True)
    rolls = [k for k in cold if "_roll_" in k]
    assert len(rolls) >= 3 and list(cold) == list(hot)
    assert any(float(utf8(cold[k].solution.value("vq_state_up"))["value"].sum()) > 0 for k in rolls), (
        "no roll uses the up slack: the penalty would not reach the objective")
    assert not any(hot[k].warm_used for k in rolls), "a roll reused an LP built for another window"
    for k in rolls:
        for run in (cold, hot):
            _assert_dense(run[k], "p_penalty_up", ("west",), lambda n, d, t: varying[t])
        assert _rel(float(hot[k].solution.obj), float(cold[k].solution.obj)) < 1e-9, k


def test_rolling_reads_penalty_values_once_per_run(tmp_path_factory, monkeypatch) -> None:
    """The authored penalty values are resolved once per run and direction, not in every roll."""
    from flextool.engine.preprocess.params import _arithmetic

    calls: list[str] = []
    original = _arithmetic._resolved_penalty

    def counting(source, parameter_name, dt):
        calls.append(parameter_name)
        return original(source, parameter_name, dt)

    monkeypatch.setattr(_arithmetic, "_resolved_penalty", counting)
    curve = {t: (6000.0 if i % 2 else 6500.0) for i, t in enumerate(T168)}
    url, scen = make_db(tmp_path_factory, alt="c9_memo", values=[["node", "demand_node", "penalty_up", tmap(curve)]],
                        base_chain=_chain(STOCHASTICS_JSON, "1_week_rolling_wind"), fixture_json=STOCHASTICS_JSON)
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("c9_memo_w"), keep_solutions=True)
    assert len(_rolls(steps)) == 7
    assert sorted(calls) == ["penalty_down", "penalty_up"], calls


NESTED_PENALTIES = {"p2020": 1000.0, "p2025": 2000.0, "p2030": 3000.0, "p2035": 4000.0}


@pytest.mark.parametrize("label", ["period", None], ids=["labelled_memo", "silent_nomemo"])
def test_nested_period_values_per_subsolve(tmp_path_factory, monkeypatch, label) -> None:
    """``west.penalty_up`` as a period map on the nested ``invest_5weeks`` sub-solves: every sub-solve has exactly
    its period's value on all its (d, t) and a row for every balance node.  With labels the resolved values are
    memoised on the run's reader (one resolve of ``penalty_up`` for the 4 sub-solves); a silent label is probed
    against each sub-solve's periods, so it is resolved per sub-solve (and still right)."""
    from flextool.engine.preprocess.params import _arithmetic
    from tests.engine._unit_db import run
    from tests.model.test_rolling_period_maps import NESTED_CHAIN, NESTED_PERIODS

    calls: list[str] = []
    original = _arithmetic._resolved_penalty

    def counting(source, parameter_name, dt):
        calls.append(parameter_name)
        return original(source, parameter_name, dt)

    monkeypatch.setattr(_arithmetic, "_resolved_penalty", counting)
    values = [["node", "west", "penalty_up", pmap(NESTED_PENALTIES, index_name=label)],
              ["solve", "invest_5weeks", "contains_solves",
               b64({"type": "array", "value_type": "str", "data": []}, "array")]]
    url, scen = make_db(tmp_path_factory, alt=f"c9_nested_{label}", values=values, base_chain=NESTED_CHAIN,
                        fixture_json=TESTS_JSON)
    steps = run(tmp_path_factory, url, scen)
    assert list(steps) == [f"invest_5weeks_{p}" for p in NESTED_PERIODS]
    seen = set()
    for step in steps.values():
        cells = _cells(step, "p_penalty_up")
        dt = _dt(step)
        west = {(d, t): v for (n, d, t), v in cells.items() if n == "west"}
        assert set(west) == set(dt)
        for (d, _t), v in west.items():
            assert v == NESTED_PENALTIES[d]
            seen.add(d)
        assert {n for n, _d, _t in cells} >= set(utf8(step.flex_data.nodeBalance)["n"])
    assert seen == set(NESTED_PERIODS)
    assert calls.count("penalty_up") == (1 if label else len(NESTED_PERIODS)), calls


def test_rolling_silent_time_map(tmp_path_factory) -> None:
    """A silent-label time map of ``demand_node.penalty_up`` (probed per roll, not memoised) reaches all 7 rolls
    densely: the curve for demand_node, the default for the other balance nodes."""
    curve = {t: (6000.0 if i % 2 else 6500.0) for i, t in enumerate(T168)}
    rolls = _rolls(_roll(tmp_path_factory, pmap(curve, index_name=None)))
    assert len(rolls) == 7
    for step in rolls:
        _assert_dense(step, "p_penalty_up", STOCH_NODES,
                      lambda n, d, t: curve[t] if n == "demand_node" else DEFAULT)


# ---------------------------------------------------------------------------
# Group penalties (inertia; capacity margin and non-synchronous share the producer)


def _inertia(tmp_path_factory, g2_penalty, *, g2_limit: bool = True):
    """``coal_wind_inertia`` with ``electricity.penalty_inertia = 3000`` and a second inertia group ``g2``
    (west) whose limit cannot be met, so its slack is used."""
    values = [["group", "electricity", "penalty_inertia", scalar(3000.0)]]
    if g2_limit:
        values += [["group", "g2", "has_inertia", b64("yes", "str")],
                   ["group", "g2", "inertia_limit", scalar(1.0e5)]]
    if g2_penalty is not None:
        values.append(["group", "g2", "penalty_inertia", g2_penalty])
    return _run(tmp_path_factory, TESTS_JSON, "coal_wind_inertia", values,
                entities=[["group", "g2", None], ["group__node", ["g2", "west"], None]])


def _group_rows(step, field: str) -> dict[tuple[str, str], float]:
    """Rows of ``electricity`` and ``g2`` (every other group gets the default too; they have no limit)."""
    f = _frame(getattr(step.flex_data, field)).select("g", "d", "value")
    assert not f.select("g", "d").is_duplicated().any()
    return {(g, d): float(v) for g, d, v in f.iter_rows() if g in ("electricity", "g2")}


def test_group_penalty_default_next_to_authored(tmp_path_factory) -> None:
    """``g2`` has an inertia limit and no penalty of its own while ``electricity`` authors one: ``g2`` gets the
    default 5000, so its slack is priced as with an explicit 5000.  Before the fix ``g2`` had no row and its
    inertia slack was free."""
    steps = _inertia(tmp_path_factory, None)
    ref = _inertia(tmp_path_factory, scalar(5000.0))
    rows = _group_rows(_last(steps), "pdGroup_penalty_inertia")
    assert rows == {("electricity", "p2020"): 3000.0, ("g2", "p2020"): 5000.0}
    assert rows == _group_rows(_last(ref), "pdGroup_penalty_inertia")
    slack = utf8(_last(steps).solution.value("vq_inertia"))
    assert float(slack.filter(pl.col("g") == "g2")["value"].sum()) > 0
    assert _rel(_obj(steps), _obj(ref)) < 1e-9
    free = _inertia(tmp_path_factory, None, g2_limit=False)
    assert _obj(steps) > _obj(free) * (1 + 1e-6)


def test_group_explicit_zero_penalty_stays_free(tmp_path_factory) -> None:
    """An explicit ``penalty_inertia = 0`` keeps meaning "violation is free": ``g2`` has no penalty row and the
    objective equals the run without ``g2``'s limit (guard: the default fill must not turn 0 into 5000)."""
    steps = _inertia(tmp_path_factory, scalar(0.0))
    rows = _group_rows(_last(steps), "pdGroup_penalty_inertia")
    assert rows == {("electricity", "p2020"): 3000.0}
    free = _inertia(tmp_path_factory, None, g2_limit=False)
    assert _rel(_obj(steps), _obj(free)) < 1e-9
