"""Price outputs in currency per physical unit (item 116).

``price_t`` (``v_dual_node_balance``), ``reserve_price_t`` and the nested ``fix_price`` handoff are the cost of one
more MWh (one more MW of reserve for one hour) at that node / group and step::

    price[n,d,t]          = σ · dual / s / (tw[d,t] · inflation_op[d] / period_share[d] · bw[d,t])
    reserve_price[...d,t] =     dual / s / (Δt[d,t] · tw[d,t] · inflation_op[d] / period_share[d] · bw[d,t])

(``flextool/engine/output/_prices.py``).  Before, node prices were negated and multiplied by
``period_share · tw · bw`` (e.g. −164 250 for a 900 CUR/MWh penalty step), reserve prices lacked the ``1/s``
(0.0009 for 900) and the ``fix_price`` handoff was ``s`` times the price (4.75e-5 for 47.5).

Every expected value is derived by hand from the fixture data in the test's docstring.  Fixtures:
``tests/fixtures/tests.json`` (``west``: penalty_up 900; ``coal``: coal 20 CUR/MWh fuel, efficiency 0.4 → 50,
500 MW; ``network``: east penalty_up 500, east_north efficiency 0.8, west_east 0.9; ``reserve``: primary/up on
electricity, wind reliability 0.5), ``stochastics.json`` and ``lh2_three_region_rp_invest.json`` with override
alternatives, all built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, RunCache, b64, make_db, utf8

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"
STOCH = FIX / "stochastics.json"
LH2_RP = FIX / "lh2_three_region_rp_invest.json"
COAL = ["init", "west", "coal"]
NET_RES = ["init", "west", "coal", "wind", "network", "reserve"]
ROLL = ["init", "west", "coal", "fullYear", "dispatch_fullYear_roll"]
NESTED_FP = [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
             ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")],
             ["node", "battery", "storage_nested_fix_method", b64("fix_price", "str")]]
STEP_2H = [["solve", "y2020_2day_dispatch", "new_stepduration", b64(2.0, "float")]]


def _chain(fixture: Path, scenario: str) -> list[str]:
    data = json.loads(fixture.read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


def _run(tmp_path_factory, alt, chain, values=(), fixture_json=None, entities=None):
    from flextool.engine import run_chain_from_db

    kw = {} if fixture_json is None else {"fixture_json": fixture_json}
    url, scen = make_db(tmp_path_factory, alt=alt, values=list(values), base_chain=list(chain),
                        entities=entities, **kw)
    wf = tmp_path_factory.mktemp(f"{alt}_w")
    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    assert steps and all(s.solution is not None for s in steps.values())
    return steps, wf


# Unpatched runs of one model are shared by the tests that read it (the network + reserve model, the rolling coal
# model); a run made under a monkeypatch goes through ``_run``.
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _shared_run(tmp_path_factory, alt, chain):
    """``_run`` of an unpatched model with no overrides, shared within the module (read only)."""
    steps, wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=[], base_chain=list(chain))
    assert steps and all(s.solution is not None for s in steps.values())
    return steps, wf


def _node_prices(wf: Path, solve: str | None = None) -> dict[tuple[str, str, str], dict[str, float | None]]:
    """``{(solve, d, t): {node: price}}`` from the written ``v_dual_node_balance`` parquets (all solves, or one)."""
    pattern = f"v_dual_node_balance__{solve}.parquet" if solve else "v_dual_node_balance__*.parquet"
    files = sorted((wf / "output_raw").glob(pattern))
    assert files, f"no {pattern} written"
    out: dict = {}
    for f in files:
        df = pl.read_parquet(f)
        assert df.height > 0, f"{f.name} is empty"
        nodes = [c for c in df.columns if c not in ("solve", "period", "time")]
        for row in df.iter_rows(named=True):
            out[(str(row["solve"]), str(row["period"]), str(row["time"]))] = {n: row[n] for n in nodes}
    return out


def _node_prices_over_all_lp_steps(step, wf: Path) -> dict[tuple[str, str, str], dict[str, float | None]]:
    """:func:`_node_prices` of ``step`` written over every ``(d, t)`` of its LP (``v_flow`` columns) as the
    realized rows, with the run's scale_the_objective: a solve that realizes no step (a nested parent, a
    forecasts-only fan) has no rows in its normal price output."""
    import dataclasses

    from flextool.engine.output._writer import highs_with_names
    from flextool.process_outputs.read_highs_solution import write_v_dual_node_balance

    rows = (utf8(step.solution.value("v_flow")).select(pl.col("d").alias("period"), pl.col("t").alias("step"))
            .unique(maintain_order=True))
    fd = dataclasses.replace(step.flex_data, realized_dispatch=rows)
    scale = float(pl.read_csv(Path(wf) / "solve_data" / "scale_the_objective.csv")["value"][0])
    out_dir = Path(wf) / "all_lp_steps"
    write_v_dual_node_balance(highs_with_names(step.solution), solve_name=step.solve_name,
                              output_dir=out_dir / "output_raw", flex_data=fd, scale_the_objective=scale)
    return _node_prices(out_dir, step.solve_name)


def _reserve_prices(wf: Path) -> dict[tuple[str, str, str], dict[tuple[str, ...], float]]:
    from flextool.lean_parquet import read_lean_parquet

    files = sorted((wf / "output_raw").glob("v_dual_reserve__upDown__group__period__t__*.parquet"))
    assert files, "no reserve price parquet written"
    out: dict = {}
    for f in files:
        df = read_lean_parquet(f)
        assert df.shape[0] > 0 and df.shape[1] > 0, f"{f.name} is empty"
        for idx, row in df.iterrows():
            out[tuple(str(x) for x in idx)] = {tuple(str(x) for x in c): float(v) for c, v in row.items()}
    return out


def _values(prices: dict, node: str) -> list[float]:
    return [v[node] for v in prices.values() if v.get(node) is not None]


def _approx_set(values, expected: set[float]) -> set[float]:
    """The members of ``expected`` that ``values`` hit; asserts every value is one of them (rel 1e-9)."""
    hit: set[float] = set()
    for v in values:
        match = [e for e in expected if v == pytest.approx(e, rel=1e-9)]
        assert match, f"price {v} is none of {sorted(expected)}"
        hit.add(match[0])
    return hit


# ---------------------------------------------------------------------------
# node prices, hand-calculated


def test_node_price_hand_calculated(tmp_path_factory) -> None:
    """coal: west demand 589 / 537 / 506 MW at t0001–t0003 exceeds coal_plant's 500 MW → unserved energy at
    penalty_up 900; elsewhere coal (20 CUR/MWh fuel / efficiency 0.4 = 50) is marginal.  So west is 900 or 50 at
    every step, both occur, and 900 at t0001.  Before: −164 250 / −9 125 (negated, × period share 1/182.5)."""
    _steps, wf = _run(tmp_path_factory, "pr_coal", COAL)
    prices = _node_prices(wf)
    west = _values(prices, "west")
    assert len(west) == 48
    assert _approx_set(west, {900.0, 50.0}) == {900.0, 50.0}
    t1 = next(v for (s, d, t), v in prices.items() if t == "t0001")
    assert t1["west"] == pytest.approx(900.0, rel=1e-9)


def test_node_price_timestep_weights(tmp_path_factory) -> None:
    """weighted_2day without generation: west is served only by the slack, so its price is penalty_up 900 at
    every step although timeset_weights vary per step (0.5, 1.0, …).  Before: −900 · tw / period_share."""
    steps, wf = _run(tmp_path_factory, "pr_tw", ["init", "west", "weighted_2day"])
    tw = set(utf8(next(iter(steps.values())).flex_data.p_timestep_weight.frame)["value"].to_list())
    assert len(tw) > 1, "the fixture must have several timestep weights"
    west = _values(_node_prices(wf), "west")
    assert len(west) == 48
    assert _approx_set(west, {900.0}) == {900.0}


def test_node_price_multi_hour_step(tmp_path_factory) -> None:
    """coal with a 2-h solve step (new_stepduration 2): a price is per MWh, not per step, so west is still 900
    (unserved, the first steps' demand exceeds 500 MW) or 50 (coal marginal), with no factor 2."""
    steps, wf = _run(tmp_path_factory, "pr_2h", COAL, STEP_2H)
    sd = set(utf8(next(iter(steps.values())).flex_data.p_step_duration.frame)["value"].to_list())
    assert sd == {2.0}
    west = _values(_node_prices(wf), "west")
    assert len(west) == 24
    assert _approx_set(west, {900.0, 50.0}) == {900.0, 50.0}


def test_network_and_reserve_prices(tmp_path_factory) -> None:
    """network + reserve at t0001: east is short (penalty_up 500); north is fed from east over east_north
    (efficiency 0.8): 500 / 0.8 = 625; west exports to east over west_east (0.9): 500 · 0.9 = 450.  Reserve
    primary/up/electricity: every MW is held back on wind (reliability 0.5 → 2 MW of wind, each worth the west
    price 450): 900.  Before: −91 250 / −114 062.5 / −82 125 and a reserve price of 0.0009 (no 1/s)."""
    _steps, wf = _shared_run(tmp_path_factory, "pr_net", NET_RES)
    prices = _node_prices(wf)
    t1 = next(v for (s, d, t), v in prices.items() if t == "t0001")
    assert t1["east"] == pytest.approx(500.0, rel=1e-9)
    assert t1["north"] == pytest.approx(625.0, rel=1e-9)
    assert t1["west"] == pytest.approx(450.0, rel=1e-9)
    res = _reserve_prices(wf)
    r1 = next(v for (s, d, t), v in res.items() if t == "t0001")
    assert r1[("primary", "up", "electricity")] == pytest.approx(900.0, rel=1e-9)


def test_reserve_price_weighted_and_multi_hour(tmp_path_factory) -> None:
    """The reserve price is per MW held for one hour: with timeset_weights (weighted_2day) the dispatch of each
    step is unchanged (no storage links the steps), so every node and reserve price equals the unweighted run's;
    with a 2-h step the t0001 situation of the network test holds again (east short, west exporting, wind
    holding the reserve): east 500, west 450, reserve 2 · 450 = 900, not 1800 (no Δt in the price).
    Before: the reserve price was ×1e-6, × tw and × Δt off."""
    _s0, wf0 = _shared_run(tmp_path_factory, "pr_rw0", NET_RES)
    _s1, wf1 = _run(tmp_path_factory, "pr_rw1", NET_RES + ["weighted_2day"])
    n0, n1 = _node_prices(wf0), _node_prices(wf1)
    r0, r1 = _reserve_prices(wf0), _reserve_prices(wf1)
    assert len(n1) == len(n0) == 48 and len(r1) == len(r0) == 48
    for key, row in n0.items():
        for node, v in row.items():
            assert n1[key][node] == pytest.approx(v, rel=1e-7, abs=1e-7), (key, node)
    for key, row in r0.items():
        for g, v in row.items():
            assert r1[key][g] == pytest.approx(v, rel=1e-7, abs=1e-7), (key, g)

    _s2, wf2 = _run(tmp_path_factory, "pr_r2h", NET_RES, STEP_2H)
    p2 = _node_prices(wf2)
    t1 = next(v for (s, d, t), v in p2.items() if t == "t0001")
    assert t1["east"] == pytest.approx(500.0, rel=1e-9)
    assert t1["west"] == pytest.approx(450.0, rel=1e-9)
    rr = next(v for (s, d, t), v in _reserve_prices(wf2).items() if t == "t0001")
    assert rr[("primary", "up", "electricity")] == pytest.approx(900.0, rel=1e-9)


# ---------------------------------------------------------------------------
# generic hand check: a step served by the slack has price = penalty_up


def _slack_cells(steps) -> list[tuple[str, str, str, str, float]]:
    """``(solve, n, d, t, penalty_up)`` of every realized cell whose up-slack is in use (strictly above its lower
    bound 0; it has no upper bound), where the price must be penalty_up."""
    out = []
    for name, st in steps.items():
        try:
            vq = utf8(st.solution.value("vq_state_up"))
        except KeyError:
            continue
        pen = utf8(st.flex_data.p_penalty_up.frame).select("n", "d", "t", pl.col("value").alias("pen"))
        hit = vq.filter(pl.col("value") > 1e-6).join(pen, on=["n", "d", "t"], how="inner")
        out += [(name, n, d, t, p) for n, d, t, p in hit.select("n", "d", "t", "pen").iter_rows()]
    return out


@pytest.mark.parametrize("case", ["rp", "rolling", "rolling_discounted", "stochastic"])
def test_slack_step_price_equals_penalty(tmp_path_factory, case) -> None:
    """At every realized (n,d,t) whose up-slack is in use, one more MWh is one more MWh of slack, so the price is
    penalty_up[n] — in representative periods (rp weights in tw), in every roll (period share of the complete
    solve, inflation 5 = years represented), with discounting (inflation_rate 0.05 on the four 5-year periods of
    dispatch_fullYear_roll) and in each forecast branch (bw 0.25).  The rp and stochastic cases lower one node's
    penalty_up so that its slack is in use at many steps.  Before: negated and × share · tw · bw."""
    if case == "rp":
        # A cheap elec_A slack (30 CUR/MWh) is used wherever supplying elec_A would cost more.
        steps, wf = _run(tmp_path_factory, "pr_sl_rp", _chain(LH2_RP, "lh2_three_region_rp_invest"),
                         [["node", "elec_A", "penalty_up", b64(30.0, "float")]], fixture_json=LH2_RP)
    elif case == "rolling":
        steps, wf = _shared_run(tmp_path_factory, "pr_sl_roll", ROLL)
    elif case == "rolling_discounted":
        steps, wf = _run(tmp_path_factory, "pr_sl_disc", ROLL,
                         [["model", "flexTool", "inflation_rate", b64(0.05, "float")]])
        infl = {}
        for st in steps.values():
            infl.update(dict(utf8(st.flex_data.p_inflation_op.frame).select("d", "value").iter_rows()))
        assert len({round(v, 9) for v in infl.values()}) > 1, "the periods must be discounted differently"
    else:
        # demand_node slack at 10 CUR/MWh: cheaper than coal (17.78), dearer than gas (6.67).
        steps, wf = _run(tmp_path_factory, "pr_sl_st", _chain(STOCH, "2_day_stochastic_dispatch"),
                         [["node", "demand_node", "penalty_up", b64(10.0, "float")]], fixture_json=STOCH)
    # The forecasts-only fan realizes no step: its prices are checked over every step of its LP.
    prices = (_node_prices_over_all_lp_steps(steps["2day_dispatch"], wf) if case == "stochastic"
              else _node_prices(wf))
    checked = 0
    for solve, n, d, t, pen in _slack_cells(steps):
        row = prices.get((solve, d, t))
        if row is None or row.get(n) is None:
            continue                       # not a realized step of this solve (look-ahead) / zero-weight step
        assert row[n] == pytest.approx(pen, rel=1e-7), (solve, n, d, t)
        checked += 1
    assert checked >= 5, f"only {checked} slack cells checked"


# ---------------------------------------------------------------------------
# stochastic, rolling, intraperiod blocks, coarse


def test_stochastic_branch_price_not_probability_weighted(tmp_path_factory) -> None:
    """2_day_stochastic_dispatch_no_storage: in each of the four branch periods (branch weight 0.25) demand_node
    is served at the margin by coal (16 CUR/MWh fuel / efficiency 0.9 = 17.78), gas (6 / 0.9 = 6.667) or free
    (zero-cost) supply that is being spilled (0); the price is the branch's own price, not × 0.25.  The period
    label ``period1`` has no dispatch: no operational weight and no price rows.  The fan realizes no step, so its
    normal price output has no rows and the price is checked over every step of its LP.  Before: × 0.25 · period
    share."""
    steps, wf = _run(tmp_path_factory, "pr_stoch", _chain(STOCH, "2_day_stochastic_dispatch_no_storage"),
                     fixture_json=STOCH)
    fd = next(iter(steps.values())).flex_data
    bw = utf8(fd.pdt_branch_weight.frame)
    assert "period1" not in set(bw["d"].to_list())
    assert pl.read_parquet(Path(wf) / "output_raw" / "v_dual_node_balance__2day_dispatch.parquet").height == 0
    prices = _node_prices_over_all_lp_steps(steps["2day_dispatch"], wf)
    periods = {d for (_s, d, _t) in prices}
    assert "period1" not in periods and len(periods) >= 4, periods
    zero = {(d, t) for d, t in bw.filter(pl.col("value") == 0.0).select("d", "t").iter_rows()}
    hit: set[float] = set()
    for (_s, d, t), row in prices.items():
        v = row["demand_node"]
        if (d, t) in zero:
            assert v is None or math.isnan(v), (d, t, v)
            continue
        hit |= _approx_set([v], {16.0 / 0.9, 6.0 / 0.9, 0.0}) if v != 0.0 else {0.0}
    assert {16.0 / 0.9, 6.0 / 0.9} <= hit
    # A forecasts-only fan (no realized: yes branch) adds no realization solve.
    assert list(steps) == ["2day_dispatch"]


def test_rolling_price_per_roll(tmp_path_factory) -> None:
    """Rolling coal (dispatch_fullYear_roll, 8 h window / 4 h jump, 4 periods of 5 years): every roll writes west
    = 900 (unserved) or 50 (coal), with the complete solve's period share and inflation 5 removed.  Before:
    −109 500 / −6 083."""
    steps, wf = _shared_run(tmp_path_factory, "pr_roll", ROLL)
    assert len(steps) > 4
    for name in steps:
        west = _values(_node_prices(wf, name), "west")
        assert west, name
        _approx_set(west, {900.0, 50.0})


def test_intraperiod_block_price_absolute(tmp_path_factory) -> None:
    """5weeks with a bind_intraperiod_blocks battery: west at t0001 is short (900, unserved); the battery's block
    price is the west price × the inverter efficiency 0.95 (one MWh from the battery delivers 0.95 to west):
    855, positive (nodeBalanceBlock_eq has the sign of nodeBalance_eq: σ = +1).  Before: −394 200 / −374 490."""
    _steps, wf = _run(tmp_path_factory, "pr_ipb", ["init", "west", "wind", "battery", "5weeks",
                                                   "bind_battery_intraperiod_blocks"])
    prices = _node_prices(wf)
    t1 = next(v for (s, d, t), v in prices.items() if t == "t0001" and d == "p2020")
    assert t1["west"] == pytest.approx(900.0, rel=1e-9)
    assert t1["battery"] == pytest.approx(855.0, rel=1e-9)


def test_coarse_price_absolute(tmp_path_factory) -> None:
    """west hourly with coal (5000 MW: never short, marginal 20 / 0.4 = 50); h2 in a 24-h group served by an
    electrolyser (efficiency 0.7) from west: west = 50 and h2 = 50 / 0.7 at every step.  Before: −9 125 and
    −13 035.7 (negated, × share)."""
    ents = [["node", "h2", None], ["unit", "elz", None], ["unit__inputNode", ["elz", "west"], None],
            ["unit__outputNode", ["elz", "h2"], None], ["group", "daily", None], ["group__node", ["daily", "h2"], None]]
    vals = [["node", "h2", "node_type", b64("balance", "str")], ["node", "h2", "inflow", b64(-10.0, "float")],
            ["node", "h2", "penalty_up", b64(10000.0, "float")], ["node", "h2", "penalty_down", b64(10000.0, "float")],
            ["unit", "elz", "efficiency", b64(0.7, "float")], ["unit", "elz", "existing", b64(100.0, "float")],
            ["unit", "elz", "conversion_method", b64("constant_efficiency", "str")],
            ["unit", "coal_plant", "existing", b64(5000.0, "float")],
            ["group", "daily", "new_stepduration", b64(24.0, "float")]]
    _steps, wf = _run(tmp_path_factory, "pr_coarse", ["init", "west", "coal", "battery"], vals, entities=ents)
    prices = _node_prices(wf)
    west, h2 = _values(prices, "west"), _values(prices, "h2")
    assert len(west) == len(h2) == 48
    _approx_set(west, {50.0})
    _approx_set(h2, {50.0 / 0.7})


# ---------------------------------------------------------------------------
# the objective scale and the LP scaling do not reach the prices


def test_price_independent_of_objective_scale(tmp_path_factory, monkeypatch) -> None:
    """network + reserve solved with scale_the_objective 1 instead of the default 1e-6: identical node and
    reserve prices (the live scale is divided out).  Before: the reserve price differed by 1e6."""
    _s0, wf0 = _shared_run(tmp_path_factory, "pr_s6", NET_RES)
    import flextool.engine.solve._orchestration as orch
    monkeypatch.setattr(orch, "_resolve_effective_obj_scale", lambda _v: 1.0)
    _s1, wf1 = _run(tmp_path_factory, "pr_s1", NET_RES)
    for get in (_node_prices, _reserve_prices):
        a, b = get(wf0), get(wf1)
        assert a.keys() == b.keys() and len(a) == 48
        for key, row in a.items():
            for col, v in row.items():
                assert b[key][col] == pytest.approx(v, rel=1e-7, abs=1e-7), (key, col)


def test_price_independent_of_lp_scaling(tmp_path_factory, monkeypatch) -> None:
    """network + reserve with the autoscaler forced to act (Layer 1 threshold 1 decade, so Layer 2 scales rows by
    factors ≠ 1; a Layer 3 user_objective_scale of 3), both recorded, and with FLEXTOOL_SCALING=off: the same node
    and reserve prices (rel 1e-6) — Layer 2 is un-scaled before the writers read the duals and Layer 3 by HiGHS."""
    import dataclasses

    import numpy as np

    import flextool.engine.solve._orchestration as orch

    seen = {"l2": [], "l3": []}
    real_cfg = orch._autoscale_resolve_config
    real2, real3 = orch._autoscale_apply_layer2_pre_solve, orch._autoscale_apply_layer3_pre_solve

    def forced_cfg(cli_args=None, **kwargs):
        return dataclasses.replace(real_cfg(cli_args, **kwargs), threshold_decades=1.0, user_objective_scale=3)

    def spy2(*a, **k):
        plan, ranges = real2(*a, **k)
        seen["l2"].append(plan)
        return plan, ranges

    def spy3(*a, **k):
        plan = real3(*a, **k)
        seen["l3"].append(plan)
        return plan

    monkeypatch.setattr(orch, "_autoscale_resolve_config", forced_cfg)
    monkeypatch.setattr(orch, "_autoscale_apply_layer2_pre_solve", spy2)
    monkeypatch.setattr(orch, "_autoscale_apply_layer3_pre_solve", spy3)
    _s0, wf0 = _run(tmp_path_factory, "pr_sc_on", NET_RES)
    plans2 = [p for p in seen["l2"] if p is not None]
    assert plans2 and any(np.any(p.row_factors != 1.0) for p in plans2), "Layer 2 did not scale any row"
    assert any(p is not None and p.user_objective_scale != 0 for p in seen["l3"]), "Layer 3 set no objective scale"

    monkeypatch.setattr(orch, "_autoscale_resolve_config", real_cfg)
    monkeypatch.setenv("FLEXTOOL_SCALING", "off")
    _s1, wf1 = _run(tmp_path_factory, "pr_sc_off", NET_RES)
    for get in (_node_prices, _reserve_prices):
        a, b = get(wf0), get(wf1)
        assert a.keys() == b.keys() and len(a) == 48
        for key, row in a.items():
            for col, v in row.items():
                assert b[key][col] == pytest.approx(v, rel=1e-6, abs=1e-6), (key, col)


# ---------------------------------------------------------------------------
# nested fix_price handoff


@pytest.mark.parametrize("scale", [None, 1.0])
def test_fix_price_handoff_equals_parent_price(tmp_path_factory, monkeypatch, scale) -> None:
    """Nested invest → storage (6 h) → rolling dispatch (36 h / 24 h) with battery fix_price: the storage solve
    hands, at each fix step, exactly its own battery price_t (rel 1e-9; the price writer over every step of the
    storage LP, as the storage solve realizes no step), at the default scale_the_objective and at 1.  At p2020/t0067 the battery discharges at the margin, so its price is the west
    price × the inverter efficiency 0.95 (one MWh from the battery delivers 0.95 MWh to west; 47.5 = 0.95 × 50 in
    this run, but the west level rests on the nested invest result and is not pinned here).  Before: s × the
    price (4.75e-5) at the default scale, ÷ nothing else."""
    if scale is not None:
        import flextool.engine.solve._orchestration as orch
        monkeypatch.setattr(orch, "_resolve_effective_obj_scale", lambda _v: scale)
    alt = "pr_fp_def" if scale is None else "pr_fp_one"
    steps, wf = _run(tmp_path_factory, alt, _chain(FIX / "tests.json",
                                                   "multi_fullYear_battery_nested_24h_invest_one_solve"),
                     NESTED_FP)
    parent = steps["storage_fullYear_6h"]
    fp = (parent.handoff.fix_storage_price
          .with_columns(pl.col(c).cast(pl.Utf8) for c in ("node", "period", "step"))
          .filter(pl.col("node") == "battery"))
    assert fp.height > 0
    prices = _node_prices_over_all_lp_steps(parent, wf)
    checked = 0
    for _n, d, t, v in fp.select("node", "period", "step", "p_fix_storage_price").iter_rows():
        assert v == pytest.approx(prices[("storage_fullYear_6h", d, t)]["battery"], rel=1e-9), (d, t)
        checked += 1
    assert checked > 0
    at = fp.filter((pl.col("period") == "p2020") & (pl.col("step") == "t0067"))
    assert at.height == 1
    west = prices[("storage_fullYear_6h", "p2020", "t0067")]["west"]
    assert west > 0.0
    assert at["p_fix_storage_price"][0] == pytest.approx(0.95 * west, rel=1e-9)


# ---------------------------------------------------------------------------
# empty price cells pass through the output writers


def test_null_prices_reach_output_writers(tmp_path_factory) -> None:
    """Empty price cells pass through the output processing and the parquet / csv / excel / SpineDB writers
    without error and are never written as a number (before the c15c reader change they were filled with 0; the
    SpineDB writer may leave them out).

    Vehicle: the coal case (west: penalty_up 900; coal 20 CUR/MWh / efficiency 0.4 = 50) with ``timeset_weights``
    on its timeset ``2day`` listing only t0001-t0024.  A step the map leaves out has weight 0
    (``p_timestep_weight``: raw weight 0, the listed steps scaled by 48 / 24), so its price weight is 0 and its
    price is empty (``_prices.to_price``); the other steps keep their merit price, 50 or 900 (the weight cancels
    between the dual and the price weight)."""
    from spinedb_api import DatabaseMapping, from_database

    from flextool.process_outputs.write_outputs import write_outputs

    weights = b64({"index_type": "str", "data": [[f"t{i:04d}", 1.0] for i in range(1, 25)]}, "map")
    steps, wf = _run(tmp_path_factory, "pr_null", COAL, [["timeset", "2day", "timeset_weights", weights]])
    prices = _node_prices(wf)
    for (_s, _d, t), row in prices.items():
        v = row["west"]
        if int(t[1:]) > 24:
            assert v is None or math.isnan(v), (t, v)
        else:
            _approx_set([v], {50.0, 900.0})
    assert any(int(t[1:]) > 24 for (_s, _d, t) in prices)
    last = next(reversed(steps.values()))
    db_url = "sqlite:///" + str(wf / "results.sqlite")
    write_outputs(
        scenario_name="pr_null_test", output_location=str(wf), subdir="pr_null_test",
        output_config_path=str(FLEXTOOL_ROOT / "flextool" / "schemas" / "default_plots.yaml"),
        fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
        write_methods=["parquet", "csv", "excel", "spinedb"], results_db_url=db_url,
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=last.flex_data_provider,
    )
    leaves: list[tuple[str, float]] = []

    def walk(value, path):
        idx = getattr(value, "indexes", None)
        if idx is None:
            leaves.append((path, value))
            return
        for k in idx:
            walk(value.get_value(k), path + (str(k),))

    with DatabaseMapping(db_url) as db:
        for pv in db.get_parameter_value_items(entity_class_name="node"):
            if pv["entity_byname"] == ("west",) and pv["parameter_definition_name"] == "price_t":
                walk(from_database(pv["value"], pv["type"]), ())
    assert leaves, "price_t of west not written"
    for path, v in leaves:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        assert not any(p.startswith("t") and int(p[1:]) > 24 for p in path if p[1:].isdigit()), (path, v)
        _approx_set([v], {50.0, 900.0})
