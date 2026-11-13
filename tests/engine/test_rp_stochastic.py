"""Representative periods in stochastic solves (item 125) and in rolling solves (item 126).

Expected values are the hand-derived oracle of ``specs/bugfix/item125/oracle.md`` (H1b, H3, H4, H5, O1, O6; H2b's
domains and drift), never read from the engine.  Databases are built in the test: a tiny one-storage model from
the schema JSON (``_tiny``) and overlays of ``tests/fixtures/lh2_three_region_rp_invest.json`` (``tools/
lp_snapshot.py`` helpers); never a checked-in ``.sqlite``.

Tiny model: storage node ``s`` (``virtual_unitsize`` 1, so v_state is in MWh; existing = C), constant inflow −D;
unit ``src`` → ``s`` (``virtual_unitsize`` 1, existing K MW) with a ``fixed`` stochastic profile p(t), so the
step change is c(t) = K·p(t) − D.  ``src`` is in the include_stochastics group ``g`` (``s`` only where said),
timeline ``tl`` with 1-h steps, forecasts ``high`` / ``low`` (``realized: no``) with weight 0.5 each.  A fan after
the trunk period also has ``real`` (``realized: yes``): the trunk reads its series, given at the solve's start
(a forecasts-only fan has no series before the reveal, item 250).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from tests.engine import _rp_hourly_ref as H
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build

REPO = Path(__file__).resolve().parents[2]
SCHEMA_JSON = REPO / "flextool" / "schemas" / "spinedb_schema.json"
RP_JSON = REPO / "tests" / "fixtures" / "lh2_three_region_rp_invest.json"
SFX = "__lh2_three_region_rp_invest"
LH2_SOLVE = "lh2_rp_invest" + SFX
LH2 = ("lh2_A", "lh2_B", "lh2_C")
BATTERIES = ("battery_A", "battery_B", "battery_C")
L = H.L
TOL = 1e-6
FO_B, WS_B, WP_B = ("bind_forward_only_blended_weights", "bind_within_solve_blended_weights",
                    "bind_within_period_blended_weights")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Tiny one-storage model
# ---------------------------------------------------------------------------


def _m(data: dict, name: str) -> api.Map:
    return api.Map(list(data), list(data.values()), index_name=name)


def _steps(n: int) -> list[str]:
    return [f"t{i:04d}" for i in range(1, n + 1)]


def _rp_weights(weights: dict) -> api.Map:
    return _m({b: _m(reps, "representative_period") for b, reps in weights.items()}, "base_period")


def _tiny(tmp_path_factory, *, n_steps: int, timesets: dict, period_timeset: dict, fan: tuple,
          profile: dict, s: dict, K: float, D: float, s_in_group: bool = False,
          inflow: dict | None = None, branch_weights: dict | None = None,
          realized: str | None = None, with_work: bool = False) -> dict:
    """Build, solve and return the steps of the tiny model.

    ``timesets`` = {name: (timeset_duration, representative_period_weights or None)}; ``fan`` = (period, start step);
    ``profile`` = {branch: {analysis_time: {step: p}}}; ``s`` = extra parameters of node ``s``; ``inflow`` (same
    nesting as ``profile``) replaces the constant inflow −D of ``s`` by a stochastic one; ``branch_weights``
    {branch: weight} (default 0.5 each); ``realized``: the branch of ``profile`` marked realized: yes (weight 0, not
    a forecast; it gives the data before a mid-horizon fan); ``with_work``: return (steps, work folder)."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp("rp_stoch_tiny")
    db_path = work / "tiny.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db_path))
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    entities = [("model", "flextool"), ("timeline", "tl"), ("node", "s"), ("unit", "src"),
                ("unit__outputNode", ("src", "s")), ("profile", "prof"), ("unit__node__profile", ("src", "s", "prof")),
                ("group", "g"), ("group__unit", ("g", "src")), ("solve", "sol")]
    entities += [("timeset", ts) for ts in timesets]
    if s_in_group:
        entities.append(("group__node", ("g", "s")))
    periods = list(period_timeset)
    fan_period, fan_step = fan
    bw = branch_weights or {b: 0.5 for b in profile if b != realized}
    branches = _m({fan_period: _m({b: _m({fan_step: _m({"yes": 0.0} if b == realized else {"no": bw[b]},
                                                         "realized")}, "analysis_time")
                                   for b in profile}, "branch")}, "period")
    prof = _m({b: _m({at: _m(series, "time") for at, series in windows.items()}, "analysis_time")
               for b, windows in profile.items()}, "branch")
    values = [
        ("model", "flextool", "solves", api.Array(["sol"])),
        ("timeline", "tl", "timestep_duration", _m({t: 1.0 for t in _steps(n_steps)}, "time")),
        ("node", "s", "node_type", "storage"),
        ("node", "s", "virtual_unitsize", 1.0),
        ("node", "s", "inflow", -D if inflow is None else
         _m({b: _m({at: _m(series, "time") for at, series in windows.items()}, "analysis_time")
             for b, windows in inflow.items()}, "branch")),
        ("unit", "src", "existing", K),
        ("unit", "src", "virtual_unitsize", 1.0),
        ("unit", "src", "efficiency", 1.0),
        ("unit__node__profile", ("src", "s", "prof"), "profile_method", "fixed"),
        ("profile", "prof", "profile", prof),
        ("group", "g", "include_stochastics", "yes"),
        ("solve", "sol", "solve_mode", "single_solve"),
        ("solve", "sol", "period_timeset", _m(period_timeset, "period")),
        ("solve", "sol", "realized_periods", api.Array(periods)),
        ("solve", "sol", "years_represented", _m({p: 1.0 for p in periods}, "period")),
        ("solve", "sol", "stochastic_branches", branches),
    ]
    for ts, (duration, weights) in timesets.items():
        values += [("timeset", ts, "timeline", "tl"),
                   ("timeset", ts, "timeset_duration", _m(duration, "time"))]
        if weights is not None:
            values.append(("timeset", ts, "representative_period_weights", _rp_weights(weights)))
    values += [("node", "s", k, v) for k, v in s.items()]
    with open_database_mapping(url) as db:
        _count, errors = api.import_data(
            db, alternatives=[("tiny", "")], scenarios=[("tiny_s", False, "")],
            scenario_alternatives=[("tiny_s", "tiny", None)], entities=entities,
            entity_alternatives=[("node", ("s",), "tiny", True), ("unit", ("src",), "tiny", True)],
            parameter_values=[(c, e, p, v, "tiny") for c, e, p, v in values])
        assert not errors, errors[:5]
        db.commit_session("rp stochastic tiny model")
    steps = run_chain_from_db(url, "tiny_s", work_folder=work / "w", keep_solutions=True)
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal
    return (steps, work / "w") if with_work else steps


def _u(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns if c != "value"])


def _values(step, var: str) -> dict[tuple, float]:
    f = _u(step.solution.value(var))
    keys = [c for c in f.columns if c != "value"]
    return {tuple(r[:-1]): r[-1] for r in f.select(*keys, "value").iter_rows()}


def _rows(pb, family: str) -> list[str]:
    return sorted(row_terms(pb, family))


def _p(c: list[float], K: float, D: float) -> list[float]:
    """The profile values that give the step changes c (c = K·p − D)."""
    return [(x + D) / K for x in c]


def _series(steps: list[str], c: list[float], K: float, D: float) -> dict:
    return dict(zip(steps, _p(c, K, D)))


# ---------------------------------------------------------------------------
# H1b / H4: a fan in the solve's first period, bind_forward_only_blended_weights
# ---------------------------------------------------------------------------

# Blocks A = t0001-t0002, B = t0003-t0004; base periods b1 = t0001 (← A), b2 = t0003 (← B); tsw = 1.
_FIRST_FAN_TS = {"rp": ({"t0001": 2.0, "t0003": 2.0}, {"t0001": {"t0001": 1.0}, "t0003": {"t0003": 1.0}})}
_H1B_HIGH = [0.0, -1.0, 1.0, 1.0]     # A net −1, B net +2
_H1B_LOW = [0.0, 1.0, 0.5, 0.5]       # A net +1, B net +1


def _first_fan(tmp_path_factory, *, high, low, s: dict, s_in_group: bool = False) -> dict:
    K, D = 2.0, 1.0
    st = _steps(4)
    return _tiny(tmp_path_factory, n_steps=4, timesets=_FIRST_FAN_TS, period_timeset={"p1": "rp"},
                 fan=("p1", "t0001"), K=K, D=D, s=s, s_in_group=s_in_group,
                 profile={"high": {"t0001": _series(st, high, K, D)}, "low": {"t0001": _series(st, low, K, D)}})


def test_h1b_forward_only_blended_first_period_fan_starts_equal(tmp_path_factory):
    """H1b: fix_start pins v_state at t0001 of every root forecast (1.5 = 0.5 · C); the chain starts
    X = v_inter[b1] − drift[b1] are equal (rp_inter_period_fan_start) ⇒ X = 1: v_inter high (0, 2), low (2, 3)."""
    steps = _first_fan(tmp_path_factory, high=_H1B_HIGH, low=_H1B_LOW,
                       s={"existing": 3.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_start",
                          "storage_state_start": 0.5})
    step = steps["sol"]
    state = _values(step, "v_state")
    assert state[("s", "p1_high", "t0001")] == pytest.approx(1.5, abs=TOL)
    assert state[("s", "p1_low", "t0001")] == pytest.approx(1.5, abs=TOL)
    inter = _values(step, "v_state_inter")
    assert inter == pytest.approx({("s", "p1_high", "t0001"): 0.0, ("s", "p1_high", "t0003"): 2.0,
                                   ("s", "p1_low", "t0001"): 2.0, ("s", "p1_low", "t0003"): 3.0}, abs=TOL)
    assert float(step.solution.obj) == pytest.approx(0.0, abs=TOL)
    # H5 (first-period fan; label p1): no rp data on the label.
    fd = step.flex_data
    pb = build(fd)
    assert len(_rows(pb, "rp_inter_period_fan_start")) == 1
    assert len(_rows(pb, "rp_inter_period_balance")) == 2
    assert _rows(pb, "rp_inter_period_cyclic") == []
    assert len(_rows(pb, "rp_inter_period_max_state")) == 4
    assert len(_rows(pb, "maxState_rp_start")) == 4
    assert len(_values(step, "v_state_rp_start")) == 4
    assert fd.rp_period_previous is None or fd.rp_period_previous.height == 0
    assert "p1" not in set(_u(fd.rp_block_first)["d"].to_list())


def test_h4_block_starts_tied_in_the_tie_window(tmp_path_factory):
    """H4: s in the include_stochastics group, the whole first period tied (default window, single solve):
    no non_anticipativity_storage_state_first row, so both blocks' starts are tied by
    non_anticipativity_rp_block_start (2 blocks × (2 forecasts − 1) = 2 rows); identical forecasts ⇒ equal states."""
    steps = _first_fan(tmp_path_factory, high=_H1B_HIGH, low=_H1B_HIGH, s_in_group=True,
                       s={"existing": 3.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_nothing"})
    step = steps["sol"]
    pb = build(step.flex_data)
    assert _rows(pb, "non_anticipativity_rp_block_start") == [
        "non_anticipativity_rp_block_start[s,p1_high,p1_low,t0001]",
        "non_anticipativity_rp_block_start[s,p1_high,p1_low,t0003]"]
    assert _rows(pb, "non_anticipativity_storage_state_first") == []
    assert len(_rows(pb, "rp_inter_period_fan_start")) == 1
    start = _values(step, "v_state_rp_start")
    for t in ("t0001", "t0003"):
        assert start[("s", "p1_high", t)] == pytest.approx(start[("s", "p1_low", t)], abs=TOL)
    state = _values(step, "v_state")
    for t in _steps(4):
        assert state[("s", "p1_high", t)] == pytest.approx(state[("s", "p1_low", t)], abs=TOL)
    assert float(step.solution.obj) == pytest.approx(0.0, abs=TOL)


def test_h4_tied_block_start_costs_a_spill(tmp_path_factory):
    """H4 (values): the block-start tie decides the objective, not only the row list.

    H4's timeline, timeset (blocks A = t0001-2, B = t0003-4; base b1 ← A, b2 ← B; tsw = 1) and first-period fan
    p1 @ t0001 (0.5 / 0.5); s: fo_blended, fix_nothing, C = 3, in the include_stochastics group, penalty_down
    P_d = 100, penalty_up P_u = 1000.  src delivers K·p = 2 · 0.5 = 1 MW in both forecasts (the net-flow tie
    non_anticipativity_storage_use holds; it leaves inflow and slacks out), and s's inflow differs per forecast,
    so the step change is c = 1 + inflow:
      high inflow −1, −1, +1, −3 ⇒ c = 0, 0, +2, −2;   low inflow −1, −1, −3, +1 ⇒ c = 0, 0, −2, +2.
    Block B from its start S_x = v_state_rp_start[s,p1_x,t0003]: high levels S_high + 2, S_high (S_high ≤ 1 for
    C = 3); low S_low − 2, S_low (S_low ≥ 2).  Every block nets 0, so every drift is 0 and the base chain
    (v_inter[b2] = v_inter[b1], one fan start) binds nothing.
    - Untied (non_anticipativity_rp_block_start removed): S_high ∈ [0, 1], S_low ∈ [2, 3], no slack, objective 0.
    - Tied (S_high = S_low = S): S + 2 − spill ≤ 3 and S − 2 + up ≥ 0 ⇒ spill + up ≥ 1 MWh.  Spill is the
      cheaper (P_d < P_u at the same weight) ⇒ S = 2, spill 1 MWh at (p1_high, t0003), up 0; v_state high
      t0003 = 3, t0004 = 1; low t0003 = 0, t0004 = 2.
    op_factor[p1_high, t0003] = Δt (1 h) · tsw (1) · inflation_op[p1] / period_share[p1] · branch weight:
    inflation_rate 0 and years_represented 1 ⇒ inflation_op = 1; p1 represents b1 + b2 = 4 h ⇒ period_share =
    4 / 8760; weight 0.5 ⇒ op_factor = 8760 / 4 · 0.5 = 1095.  **Objective = P_d · 1 MWh · 1095 = 109 500.**
    """
    st = _steps(4)
    flat = {t: 0.5 for t in st}
    steps = _tiny(tmp_path_factory, n_steps=4, timesets=_FIRST_FAN_TS, period_timeset={"p1": "rp"},
                  fan=("p1", "t0001"), K=2.0, D=1.0, s_in_group=True,
                  s={"existing": 3.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_nothing",
                     "penalty_down": 100.0, "penalty_up": 1000.0},
                  profile={"high": {"t0001": flat}, "low": {"t0001": flat}},
                  inflow={"high": {"t0001": dict(zip(st, [-1.0, -1.0, 1.0, -3.0]))},
                          "low": {"t0001": dict(zip(st, [-1.0, -1.0, -3.0, 1.0]))}})
    step = steps["sol"]
    # Values first: removing the tie leaves an objective of 0 and different block-B starts.
    assert float(step.solution.obj) == pytest.approx(100.0 * 1095.0, rel=1e-9)
    start = _values(step, "v_state_rp_start")
    assert start[("s", "p1_high", "t0003")] == pytest.approx(2.0, abs=TOL)
    assert start[("s", "p1_low", "t0003")] == pytest.approx(2.0, abs=TOL)
    state = _values(step, "v_state")
    assert [state[("s", "p1_high", t)] for t in ("t0003", "t0004")] == pytest.approx([3.0, 1.0], abs=TOL)
    assert [state[("s", "p1_low", t)] for t in ("t0003", "t0004")] == pytest.approx([0.0, 2.0], abs=TOL)
    spill = {k: v for k, v in _values(step, "vq_state_down").items() if abs(v) > TOL}
    assert spill == pytest.approx({("s", "p1_high", "t0003"): 1.0}, abs=TOL)
    assert len(_rows(build(step.flex_data), "non_anticipativity_rp_block_start")) == 2


# ---------------------------------------------------------------------------
# H3 / H5: a fan after the trunk period, bind_forward_only_blended_weights
# ---------------------------------------------------------------------------


def test_h3_forward_only_blended_trunk_to_branch_chain(tmp_path_factory):
    """H3 (C = 3): drifts p1 b1 3, b2 0.75, b3 −2, p2_high −1, p2_low +1; every branch's first base period
    continues from the trunk period's last ⇒ unique v_inter 2.25, 3.0, 1.0, 0.0 (high), 2.0 (low).  H5: the
    domains carry no label (p2) row."""
    K, D = 4.0, 2.0
    trunk = _series(_steps(6), [1.5, 1.5, -1.0, -1.0, 0.0, 0.0], K, D)
    steps = _tiny(
        tmp_path_factory, n_steps=8,
        timesets={"ts1": ({"t0001": 2.0, "t0003": 2.0},
                          {"t0001": {"t0001": 1.0}, "t0003": {"t0001": 0.55, "t0003": 0.45},
                           "t0005": {"t0003": 1.0}}),
                  "ts2": ({"t0007": 2.0}, {"t0007": {"t0007": 1.0}})},
        period_timeset={"p1": "ts1", "p2": "ts2"}, fan=("p2", "t0007"), K=K, D=D,
        s={"existing": 3.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_nothing"},
        profile={"high": {"t0007": _series(["t0007", "t0008"], [-0.5, -0.5], K, D)},
                 "low": {"t0007": _series(["t0007", "t0008"], [0.5, 0.5], K, D)},
                 "real": {"t0001": trunk, "t0007": _series(["t0007", "t0008"], [0.5, 0.5], K, D)}},
        realized="real")
    step = steps["sol"]
    inter = _values(step, "v_state_inter")
    assert inter == pytest.approx({("s", "p1", "t0001"): 2.25, ("s", "p1", "t0003"): 3.0,
                                   ("s", "p1", "t0005"): 1.0, ("s", "p2_high", "t0007"): 0.0,
                                   ("s", "p2_low", "t0007"): 2.0}, abs=TOL)
    assert float(step.solution.obj) == pytest.approx(0.0, abs=TOL)
    fd = step.flex_data
    assert set(_values(step, "v_state_rp_start")) == {("s", "p1", "t0001"), ("s", "p1", "t0003"),
                                                       ("s", "p2_high", "t0007"), ("s", "p2_low", "t0007")}
    assert sorted(_u(fd.rp_period_previous).select("d", "d_prev").iter_rows()) == [("p2_high", "p1"),
                                                                                  ("p2_low", "p1")]
    assert sorted(_u(fd.rp_block_first).select("d", "t").iter_rows()) == [
        ("p1", "t0001"), ("p1", "t0003"), ("p2_high", "t0007"), ("p2_low", "t0007")]
    tw = {(d, t): v for d, t, v in _u(fd.p_timestep_weight.frame).select("d", "t", "value").iter_rows()}
    assert tw == pytest.approx({("p1", "t0001"): 1.55 * 2 / 3, ("p1", "t0002"): 1.55 * 2 / 3,
                                ("p1", "t0003"): 1.45 * 2 / 3, ("p1", "t0004"): 1.45 * 2 / 3,
                                ("p2_high", "t0007"): 1.0, ("p2_high", "t0008"): 1.0,
                                ("p2_low", "t0007"): 1.0, ("p2_low", "t0008"): 1.0}, abs=1e-9)
    pb = build(fd)
    assert len(_rows(pb, "rp_inter_period_balance")) == 4
    assert _rows(pb, "rp_inter_period_cyclic") == []
    assert _rows(pb, "rp_inter_period_fan_start") == []
    assert len(_rows(pb, "rp_inter_period_max_state")) == 5
    assert len(_rows(pb, "maxState_rp_start")) == 4


def test_h2b_within_solve_blended_every_branch_closes_on_the_trunk_start(tmp_path_factory):
    """H2b (H5 domains): ws_blended, fan after the trunk period: balance 2 (each branch from the trunk), cyclic 2
    (d_first × d_last = p1 × {p2_high, p2_low}); drift[p1] = +2 ⇒ drift of every branch = −2, so the high block
    spills 1 MWh (its flows net −1); v_inter p1 = 2, p2_high = p2_low = 0.
    Objective = P_d (100) · 1 MWh · op_factor[p2_high, t]; op_factor = Δt (1 h) · tsw (1) · inflation_op[p2] /
    period_share[p2] · branch weight: inflation_rate 0 and years_represented 1 ⇒ inflation_op = 1; p2 (and its
    branch periods) represents one 2-h base period ⇒ period_share = 2 / 8760; weight 0.5 ⇒ op_factor = 8760 / 2 ·
    0.5 = 2190 ⇒ **objective = 219 000**."""
    K, D = 2.0, 1.0
    trunk = _series(["t0001", "t0002"], [1.0, 1.0], K, D)
    steps = _tiny(
        tmp_path_factory, n_steps=4,
        timesets={"ts1": ({"t0001": 2.0}, {"t0001": {"t0001": 1.0}}),
                  "ts2": ({"t0003": 2.0}, {"t0003": {"t0003": 1.0}})},
        period_timeset={"p1": "ts1", "p2": "ts2"}, fan=("p2", "t0003"), K=K, D=D,
        s={"existing": 2.0, "storage_binding_method": WS_B, "storage_start_end_method": "fix_nothing",
           "penalty_down": 100.0, "penalty_up": 1000.0},
        profile={"high": {"t0003": _series(["t0003", "t0004"], [-1.0, 0.0], K, D)},
                 "low": {"t0003": _series(["t0003", "t0004"], [-1.0, -1.0], K, D)},
                 "real": {"t0001": trunk, "t0003": _series(["t0003", "t0004"], [-1.0, -1.0], K, D)}},
        realized="real")
    step = steps["sol"]
    inter = _values(step, "v_state_inter")
    assert inter == pytest.approx({("s", "p1", "t0001"): 2.0, ("s", "p2_high", "t0003"): 0.0,
                                   ("s", "p2_low", "t0003"): 0.0}, abs=TOL)
    state, start = _values(step, "v_state"), _values(step, "v_state_rp_start")
    for br in ("p2_high", "p2_low"):
        assert state[("s", br, "t0004")] - start[("s", br, "t0003")] == pytest.approx(-2.0, abs=TOL)
    pb = build(step.flex_data)
    assert len(_rows(pb, "rp_inter_period_balance")) == 2
    assert _rows(pb, "rp_inter_period_cyclic") == [
        "rp_inter_period_cyclic[s,p1,t0001,p2_high,t0003]", "rp_inter_period_cyclic[s,p1,t0001,p2_low,t0003]"]
    assert float(step.solution.obj) == pytest.approx(100.0 * 2190.0, rel=1e-9)


# ---------------------------------------------------------------------------
# H1 / H2: plain bind_forward_only and bind_within_solve in a fan (no representative periods)
# ---------------------------------------------------------------------------

_PLAIN_FAN_TS = {"ts": ({"t0001": 4.0}, None)}


def _h1(tmp_path_factory, *, high, low, s_in_group: bool = False) -> dict:
    K, D = 5.0, 3.0
    st = _steps(4)
    return _tiny(tmp_path_factory, n_steps=4, timesets=_PLAIN_FAN_TS, period_timeset={"p1": "ts"},
                 fan=("p1", "t0001"), K=K, D=D, s_in_group=s_in_group,
                 s={"existing": 5.0, "storage_binding_method": "bind_forward_only",
                    "storage_start_end_method": "fix_nothing"},
                 profile={"high": {"t0001": _series(st, high, K, D)}, "low": {"t0001": _series(st, low, K, D)}})


def test_h1_forward_only_first_period_fan_starts_equal(tmp_path_factory):
    """H1a: plain bind_forward_only, fix_nothing, C = 5, first-period fan p1 @ t0001; c high 0, −1, −1, +2,
    low 0, +2, +1, −3.  t0001 is open (no state change) ⇒ high S, S−1, S−2, S; low S, S+2, S+3, S.  The equal
    start (storage_state_fan_start) gives S ≥ 2 and S ≤ 2 ⇒ S = 2: v_high 2, 1, 0, 2; v_low 2, 4, 5, 2."""
    step = _h1(tmp_path_factory, high=[0.0, -1.0, -1.0, 2.0], low=[0.0, 2.0, 1.0, -3.0])["sol"]
    state = _values(step, "v_state")
    assert [state[("s", "p1_high", t)] for t in _steps(4)] == pytest.approx([2.0, 1.0, 0.0, 2.0], abs=TOL)
    assert [state[("s", "p1_low", t)] for t in _steps(4)] == pytest.approx([2.0, 4.0, 5.0, 2.0], abs=TOL)
    assert float(step.solution.obj) == pytest.approx(0.0, abs=TOL)
    assert _rows(build(step.flex_data), "storage_state_fan_start") == [
        "storage_state_fan_start[s,p1_high,t0001,p1_low,t0001]"]


def test_h1_fan_start_kept_when_the_window_ties_the_whole_root(tmp_path_factory):
    """H4 note on S4: s in the include_stochastics group with the default window (all of p1 tied) has no
    non_anticipativity_storage_state_first row, so the start is tied by storage_state_fan_start alone (identical
    series: the net-flow tie holds)."""
    c = [0.0, -1.0, -1.0, 2.0]
    step = _h1(tmp_path_factory, high=c, low=c, s_in_group=True)["sol"]
    pb = build(step.flex_data)
    assert _rows(pb, "non_anticipativity_storage_state_first") == []
    assert len(_rows(pb, "storage_state_fan_start")) == 1
    state = _values(step, "v_state")
    assert state[("s", "p1_high", "t0001")] == pytest.approx(state[("s", "p1_low", "t0001")], abs=TOL)


def _h2(tmp_path_factory, *, K: float, D: float, low: list[float]) -> dict:
    trunk = _series(["t0001", "t0002"], [1.0, 1.0], K, D)
    return _tiny(
        tmp_path_factory, n_steps=4, timesets={"ts1": ({"t0001": 2.0}, None), "ts2": ({"t0003": 2.0}, None)},
        period_timeset={"p1": "ts1", "p2": "ts2"}, fan=("p2", "t0003"), K=K, D=D,
        branch_weights={"high": 0.3, "low": 0.7},
        s={"existing": 2.0, "storage_binding_method": "bind_within_solve", "storage_start_end_method": "fix_nothing",
           "penalty_down": 100.0, "penalty_up": 1000.0},
        profile={"high": {"t0003": _series(["t0003", "t0004"], [-1.0, 0.0], K, D)},
                 "low": {"t0003": _series(["t0003", "t0004"], low, K, D)},
                 "real": {"t0001": trunk, "t0003": _series(["t0003", "t0004"], low, K, D)}},
        realized="real")["sol"]


# op_factor[p2_x, t] = Δt (1 h) · tsw (1) · inflation_op[p2] / period_share[p2] · branch weight: inflation_rate 0 and
# years_represented 1 ⇒ inflation_op = 1; p2 (and its branch periods) holds 2 h ⇒ period_share = 2 / 8760 ⇒
# op_factor = 4380 · w: high (0.3) 1314, low (0.7) 3066.
_OP_HIGH, _OP_LOW = 4380.0 * 0.3, 4380.0 * 0.7


def test_h2_within_solve_every_branch_ends_at_the_trunk_start(tmp_path_factory):
    """H2a: plain bind_within_solve, fan at p2 @ t0003 (0.3 / 0.7), C = 2; c trunk +1, +1; high −1, 0; low −1, −1.
    One loop per path: every leaf ends at E, the trunk's first step starts from E ⇒ trunk E+1, E+2 ⇒ E = 0;
    low 1, 0 closes; high needs a spill of 1 MWh (timing within the period degenerate).
    Objective = P_d (100) · 1 MWh · op_factor[p2_high] = 131 400."""
    step = _h2(tmp_path_factory, K=2.0, D=1.0, low=[-1.0, -1.0])
    state = _values(step, "v_state")
    assert [state[("s", "p1", t)] for t in ("t0001", "t0002")] == pytest.approx([1.0, 2.0], abs=TOL)
    assert [state[("s", "p2_low", t)] for t in ("t0003", "t0004")] == pytest.approx([1.0, 0.0], abs=TOL)
    assert state[("s", "p2_high", "t0004")] == pytest.approx(0.0, abs=TOL)
    spill = _values(step, "vq_state_down")
    assert sum(v for k, v in spill.items() if k[1] == "p2_high") == pytest.approx(1.0, abs=TOL)
    assert sum(v for k, v in spill.items() if k[1] != "p2_high") == pytest.approx(0.0, abs=TOL)
    assert float(step.solution.obj) == pytest.approx(100.0 * _OP_HIGH, rel=1e-9)
    assert _rows(build(step.flex_data), "storage_within_solve_path_end") == [
        "storage_within_solve_path_end[s,p2_high,t0004,p2_low,t0004]"]


def test_h2a_prime_within_solve_closes_both_leaves(tmp_path_factory):
    """H2a': D = 2, K = 4: trunk +1, +1; high −1, 0; low −1, −2; C = 2.  Ends equal E = 0 ⇒ high spills 1 MWh and
    low takes 1 MWh of up-slack (a trunk slack costs more), whichever leaf the trunk wraps to.
    Objective = P_d · op_factor[p2_high] + P_u · op_factor[p2_low] = 100 · 1314 + 1000 · 3066 = 3 197 400."""
    step = _h2(tmp_path_factory, K=4.0, D=2.0, low=[-1.0, -2.0])
    state = _values(step, "v_state")
    assert [state[("s", "p1", t)] for t in ("t0001", "t0002")] == pytest.approx([1.0, 2.0], abs=TOL)
    assert state[("s", "p2_high", "t0004")] == pytest.approx(0.0, abs=TOL)
    assert state[("s", "p2_low", "t0004")] == pytest.approx(0.0, abs=TOL)
    assert sum(v for k, v in _values(step, "vq_state_down").items() if k[1] == "p2_high") == pytest.approx(
        1.0, abs=TOL)
    assert sum(v for k, v in _values(step, "vq_state_up").items() if k[1] == "p2_low") == pytest.approx(
        1.0, abs=TOL)
    assert float(step.solution.obj) == pytest.approx(100.0 * _OP_HIGH + 1000.0 * _OP_LOW, rel=1e-9)


def test_within_solve_output_state_change_of_a_first_period_fan(tmp_path_factory, monkeypatch):
    """node_state_change (calc_storage_vre) of plain bind_within_solve with a fan in the solve's first period: the
    first step of each root reports v[root, t0001] − v[leaf, t0004] (the model's lag, dtttdt_within_solve), not
    the root's own last step.  p1 (t0001-t0002) fans at t0001 into high / low, p2 (t0003-t0004) continues them;
    C = 1, fix_nothing; c high +1, 0, −1, 0; low 0, +1, 0, −1.  Both leaves end at E and each root starts from E
    + c(t0001) ≤ 1 ⇒ E = 0: v_high 1, 1, 0, 0; v_low 0, 1, 1, 0 (unique, no slack).  State change (unitsize 1)
    high +1, 0, −1, 0; low 0, +1, 0, −1 (the old wrap to the root's own last step gave 0 and −1 at t0001).
    The forecast periods are not realized (normal outputs drop them), so the test keeps them, reads their v_state
    from the solution and reports every step."""
    from flextool.process_outputs import drop_levels as dl
    from flextool.process_outputs.calc_storage_vre import node_state_change_dt
    from flextool.process_outputs.write_outputs import _read_outputs

    K, D = 2.0, 1.0
    steps, work = _tiny(
        tmp_path_factory, n_steps=4, timesets={"ts1": ({"t0001": 2.0}, None), "ts2": ({"t0003": 2.0}, None)},
        period_timeset={"p1": "ts1", "p2": "ts2"}, fan=("p1", "t0001"), K=K, D=D, with_work=True,
        s={"existing": 1.0, "storage_binding_method": "bind_within_solve", "storage_start_end_method": "fix_nothing",
           "penalty_down": 100.0, "penalty_up": 1000.0},
        # One forecast per branch, made at the fan's analysis time t0001 and covering p1 and p2 (item 255).
        profile={"high": {"t0001": _series(["t0001", "t0002", "t0003", "t0004"], [1.0, 0.0, -1.0, 0.0], K, D)},
                 "low": {"t0001": _series(["t0001", "t0002", "t0003", "t0004"], [0.0, 1.0, 0.0, -1.0], K, D)}})
    step = steps["sol"]
    state = _values(step, "v_state")
    assert [state[("s", "p1_high", t)] for t in ("t0001", "t0002")] == pytest.approx([1.0, 1.0], abs=TOL)
    assert [state[("s", "p2_high", t)] for t in ("t0003", "t0004")] == pytest.approx([0.0, 0.0], abs=TOL)
    assert [state[("s", "p1_low", t)] for t in ("t0001", "t0002")] == pytest.approx([0.0, 1.0], abs=TOL)
    assert [state[("s", "p2_low", t)] for t in ("t0003", "t0004")] == pytest.approx([1.0, 0.0], abs=TOL)
    par, sets, v = _read_outputs(
        str(work / "output_raw"), flex_data=step.flex_data, solution=step.solution, solve_name=step.solve_name,
        solve_steps=[(st.solve_name, st.flex_data, st.effective_solution) for st in steps.values()],
        flex_data_provider=step.flex_data_provider)
    monkeypatch.setattr(dl, "_drop_unrealized_forecast_periods", lambda *_: None)
    sets.dt_realize_dispatch = sets.dt
    par, sets, v = dl.drop_levels(par, sets, v)
    # output_raw holds realized rows only: v_state of the forecast periods from the solution.
    v.state = (step.solution.value("v_state").to_pandas().astype({"n": str, "d": str, "t": str})
               .pivot_table(index=["d", "t"], columns="n", values="value").rename_axis(["period", "time"]))
    change = node_state_change_dt(sets, v, par.entity_unitsize)["s"]
    got = {(str(d), str(t)): float(x) for (d, t), x in change.items()}
    assert got == pytest.approx({
        ("p1_high", "t0001"): 1.0, ("p1_high", "t0002"): 0.0, ("p2_high", "t0003"): -1.0, ("p2_high", "t0004"): 0.0,
        ("p1_low", "t0001"): 0.0, ("p1_low", "t0002"): 1.0, ("p2_low", "t0003"): 0.0, ("p2_low", "t0004"): -1.0},
        abs=TOL)


# ---------------------------------------------------------------------------
# O1: identical forecasts = the deterministic model (lh2_three_region_rp_invest)
# ---------------------------------------------------------------------------


def _lh2_run(tmp_path_factory, values: list, *, drop: tuple = (), patch=None) -> dict:
    """Solve the lh2 fixture with the overlay ``values``; ``patch`` (an ``lp_snapshot`` ``_patch_*`` function)
    first adds its own scenario, which the overlay then builds on."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads(RP_JSON.read_text(encoding="utf-8"))
    data["parameter_values"] = [v for v in data["parameter_values"]
                                if not (v[0] == "solve" and v[1] == LH2_SOLVE and v[2] in drop)]
    base = "lh2_three_region_rp_invest"
    if patch is not None:
        data = patch(data)
        base = data["scenarios"][-1][0]
    data = L._overlay(data, name="custom", base=base, values=values)
    work = tmp_path_factory.mktemp("rp_stoch_lh2")
    (work / "f.json").write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, "custom", work_folder=work / "w", keep_solutions=True)
    assert steps
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal
    return steps


def _branches(period: str, step: str) -> dict:
    def level(name, data):
        return {"type": "map", "index_type": "str", "index_name": name, "data": data}
    return level("period", [[period, level("branch", [
        [b, level("analysis_time", [[step, level("realized", [["no", 0.5]])]])] for b in ("a", "b")])]])


def _lh2_methods(method: str) -> list:
    # O1 compares the blended methods: bind_within_timeblock and fix_nothing keep the batteries (plain
    # bind_within_solve + fix_start in the fixture) out of it; test_lh2_identical_forecasts_plain_within_solve
    # covers them.
    return ([["node", n, "storage_binding_method", method] for n in LH2]
            + [["node", n, "storage_binding_method", "bind_within_timeblock"] for n in BATTERIES]
            + [["node", n, "storage_start_end_method", "fix_nothing"] for n in BATTERIES])


_O1_DET: dict[str, float] = {}


@pytest.mark.parametrize("fan", [("y2040", "t0049"), ("y2030", "t0001")], ids=["midhorizon", "first_period"])
@pytest.mark.parametrize("method", [WS_B, FO_B, WP_B])
def test_o1_identical_forecasts_equal_the_deterministic_objective(tmp_path_factory, method, fan):
    """O1: two identical forecasts (0.5 / 0.5, no stochastic data) give the deterministic objective, with the fan
    after the trunk period (y2040) and in the first period (y2030); the lh2 nodes keep fix_start."""
    if method not in _O1_DET:
        det = _lh2_run(tmp_path_factory, _lh2_methods(method))[LH2_SOLVE]
        _O1_DET[method] = float(det.solution.obj)
    steps = _lh2_run(tmp_path_factory, _lh2_methods(method)
                     + [["solve", LH2_SOLVE, "stochastic_branches", _branches(*fan)]])
    step = steps[LH2_SOLVE]
    assert float(step.solution.obj) == pytest.approx(_O1_DET[method], rel=1e-7)
    branch_periods = {d for d in _u(step.flex_data.rp_block_first)["d"].unique().to_list() if d.endswith(("_a", "_b"))}
    fan_periods = ("y2030", "y2040") if fan[0] == "y2030" else ("y2040",)
    assert branch_periods == {f"{p}_{b}" for p in fan_periods for b in ("a", "b")}
    if method == FO_B and fan[0] == "y2030":
        # The lh2 nodes' fix_start pins the first block, not the chain start: the fan start ties them too.
        assert len(_rows(build(step.flex_data), "rp_inter_period_fan_start")) == len(LH2)


_FANS = [("y2040", "t0049"), ("y2030", "t0001")]
_FAN_IDS = ["midhorizon", "first_period"]
_LH2_DET: list[float] = []


@pytest.mark.parametrize("fan", _FANS, ids=_FAN_IDS)
def test_lh2_identical_forecasts_plain_within_solve(tmp_path_factory, fan):
    """Identical forecasts (0.5 / 0.5) of the lh2 fixture as it is (batteries bind_within_solve + fix_start, the lh2
    nodes bind_within_period_blended_weights) give the deterministic objective: every forecast path closes its own
    loop (before, only the last-listed branch did and the other battery ends were free: 8.93e9 vs 9.39e9)."""
    if not _LH2_DET:
        _LH2_DET.append(float(_lh2_run(tmp_path_factory, [])[LH2_SOLVE].solution.obj))
    step = _lh2_run(tmp_path_factory, [["solve", LH2_SOLVE, "stochastic_branches", _branches(*fan)]])[LH2_SOLVE]
    assert float(step.solution.obj) == pytest.approx(_LH2_DET[0], rel=1e-9)


@pytest.mark.parametrize("fan", _FANS, ids=_FAN_IDS)
def test_plain_within_solve_every_leaf_ends_at_the_wrap_target(tmp_path_factory, fan):
    """The rp_multi_base_solve topology (lp_snapshot) with forecasts a / b and the batteries (plain
    bind_within_solve) on fix_nothing: the last step of every leaf (y2040_a, y2040_b) ends at the level of the
    trunk's wrap target d* (the last period key, y2040_b).  With the fan in the first period each root's first
    step continues from the last step of its own path (y2030_x t0001 ← y2040_x t0096)."""
    steps = _lh2_run(tmp_path_factory,
                     [["solve", LH2_SOLVE, "stochastic_branches", _branches(*fan)]]
                     + [["node", n, "storage_start_end_method", "fix_nothing"] for n in BATTERIES],
                     patch=L._patch_rp_multi_base_solve)
    step = steps[LH2_SOLVE]
    state = _values(step, "v_state")
    for n in BATTERIES:
        assert state[(n, "y2040_a", "t0096")] == pytest.approx(state[(n, "y2040_b", "t0096")], abs=1e-6), n
    assert _rows(build(step.flex_data), "storage_within_solve_path_end") == [
        f"storage_within_solve_path_end[{n},y2040_a,t0096,y2040_b,t0096]" for n in BATTERIES]
    lag = step.flex_data.dtttdt_within_solve
    if fan[0] == "y2030":
        wrap = {(d, t): (dp, tp) for d, t, dp, tp in _u(lag).select(
            "d", "t", "d_previous", "t_previous_within_solve").iter_rows() if d.startswith("y2030")}
        assert wrap[("y2030_a", "t0001")] == ("y2040_a", "t0096")
        assert wrap[("y2030_b", "t0001")] == ("y2040_b", "t0096")
    else:
        assert lag is None


# ---------------------------------------------------------------------------
# O6: rolling + representative periods + blended (item 126)
# ---------------------------------------------------------------------------

# daily_group's 24-h coarse steps would be cut by a roll that starts inside a day: 12-h coarse steps align with
# the 36-h jump and divide the 24-h representative blocks.
_ROLLING = [["solve", LH2_SOLVE, "solve_mode", "rolling_window"],
            ["solve", LH2_SOLVE, "rolling_solve_horizon", 48.0],
            ["solve", LH2_SOLVE, "rolling_solve_jump", 36.0],
            ["group", "daily_group", "new_stepduration", 12.0]]
_NO_INVEST = ("invest_periods", "realized_invest_periods")


def test_o6_day_long_coarse_step_cut_by_the_roll_names_the_jump(tmp_path_factory):
    """daily_group's own 24-h coarse step divides the 24-h representative blocks, but roll 0 (jump 36 h) realizes
    up to t0036, 12 h into the block t0025-t0048: the error names the roll's realized end, not the
    representative-period length."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    rolling = [r for r in _ROLLING if r[1] != "daily_group"] + [["group", "daily_group", "new_stepduration", 24.0]]
    with pytest.raises(FlexToolConfigError) as exc:
        _lh2_run(tmp_path_factory, rolling, drop=_NO_INVEST)
    msg = str(exc.value)
    assert ("the realized part of this roll ends at step 't0036' (rolling_solve_jump), inside a representative "
            "period and inside the coarse step starting 't0025' (period 'y2030')") in msg
    assert "'daily_group'" in msg
    assert "does not evenly divide" not in msg


def _rolling(tmp_path_factory, method: str) -> dict:
    return _lh2_run(tmp_path_factory, _ROLLING + [["node", n, "storage_binding_method", method] for n in LH2],
                    drop=_NO_INVEST)


def _mps(step, path: Path) -> bytes:
    build(step.flex_data).write_mps(path)
    return path.read_bytes()


_O6_REF: dict[str, bytes] = {}


@pytest.mark.parametrize("method", [WS_B, FO_B, WP_B])
def test_o6_rolling_blended_binds_forward_only_with_one_warning(tmp_path_factory, caplog, method):
    """O6: horizon 48 h, jump 36 h over the 96 rp steps ⇒ 3 rolls (roll 0 realises up to t0036, inside the
    second y2030 block); one warning naming the lh2 nodes; no representative-period storage rows; rolls 1 and 2
    continue the lh2 states; every roll's LP is the LP of the same model with bind_forward_only."""
    caplog.set_level(logging.WARNING, logger="flextool.engine.solve._run_model")
    steps = _rolling(tmp_path_factory, method)
    warnings = [r.getMessage() for r in caplog.records if "storage_binding_method is not applied" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert "lh2_A, lh2_B, lh2_C" in warnings[0]
    assert ("storage_binding_method is not applied in a rolling solve: they bind forward only and the storage "
            "state is passed directly from roll to roll") in warnings[0]
    names = list(steps)
    assert names == [f"{LH2_SOLVE}_roll_{k}" for k in range(3)]
    for k, name in enumerate(names):
        step = steps[name]
        assert step.flex_data.nodeState_rp is None or step.flex_data.nodeState_rp.height == 0
        for var in ("v_state_inter", "v_state_rp_start"):
            with pytest.raises(KeyError):
                step.solution.value(var)
        pb = build(step.flex_data)
        for fam in ("rp_inter_period_balance", "rp_inter_period_cyclic", "rp_inter_period_fan_start",
                    "rp_inter_period_max_state", "maxState_rp_start"):
            assert _rows(pb, fam) == [], (name, fam)
        if k:
            carry = {n: v for n, v in _u(step.flex_data.p_roll_continue_state.frame)
                     .select("n", "value").iter_rows()}
            prev = steps[names[k - 1]]
            end = ("t0036", "y2030") if k == 1 else ("t0072", "y2040")
            su = {n: v for n, v in _u(prev.flex_data.p_state_unitsize.frame).select("n", "value").iter_rows()}
            state = _values(prev, "v_state")
            for n in LH2:
                assert carry[n] == pytest.approx(state[(n, end[1], end[0])] * su[n], rel=1e-9, abs=1e-6)
    if not _O6_REF:
        before = len(caplog.records)
        ref = _rolling(tmp_path_factory, "bind_forward_only")
        assert not [r for r in caplog.records[before:] if "storage_binding_method is not applied" in r.getMessage()]
        out = tmp_path_factory.mktemp("o6_ref_mps")
        for k, name in enumerate(ref):
            _O6_REF[name] = _mps(ref[name], out / f"{k}.mps")
        _O6_REF["_obj"] = [float(s.solution.obj) for s in ref.values()]
    out = tmp_path_factory.mktemp("o6_mps")
    for k, name in enumerate(names):
        assert _mps(steps[name], out / f"{k}.mps") == _O6_REF[name], name
    assert [float(s.solution.obj) for s in steps.values()] == pytest.approx(_O6_REF["_obj"], rel=1e-9)
