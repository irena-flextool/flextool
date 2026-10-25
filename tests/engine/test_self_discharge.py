"""Storage self-discharge is exact for any step length (FA-1, item 3; owner decision option (a)).

The content decays continuously at λ = ln(1 + self_discharge_loss) per hour and the flows of a step are spread evenly
over the step.  The balance row applies the loss to a weighted average of the start and end state,
``(α·S_t + β·S⁻)·su`` with h = Δt·λ, α = h/(1 − e^{−h}) − 1, β = 1 − h/(e^h − 1), which gives the exact one-step
solution S_t = e^{−h}·S⁻ + (1 − e^{−h})/h·E_t.  Coarse new_stepduration groups and bind_intraperiod_blocks blocks apply
it once per coarse step / block (h = Δ_b·λ).  Before, the row charged ``loss·S_t`` per step whatever its length (a
24 h step lost one hour's worth), the coarse and block rows ``loss·Δt·S_t``, and the outputs reported
``state·loss·su`` (not the LP's loss).

Every expected value is a closed form (``specs/bugfix/fa1_self_discharge_analysis.md``), not a captured output.
Case: the tests.json battery (inverter off, existing 1000 MWh), 10 MWh of inflow into the storage in t0003 only,
``bind_forward_only`` with fix_start 0 and horizon ``free``, the 48-h y2020_2day_dispatch solve.  Databases are built
from JSON fixtures plus an override alternative (never a checked-in sqlite).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flextool.engine.model import _self_discharge
from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
CHAIN = ["init", "west", "wind", "coal", "battery"]
E = 10.0
SU = 1000.0


def _tmap(vals: dict, n: int = 72) -> list:
    return b64({"index_type": "str", "index_name": "time",
                "data": [[f"t{i:04d}", vals.get(f"t{i:04d}", 0.0)] for i in range(1, n + 1)]}, "map")


def _values(loss: float, *, method: str = "bind_forward_only", start: str = "fix_start") -> list:
    return [["node", "battery", "self_discharge_loss", b64(loss, "float")],
            ["node", "battery", "storage_binding_method", b64(method, "str")],
            ["node", "battery", "storage_solve_horizon_method", b64("free", "str")],
            ["node", "battery", "storage_start_end_method", b64(start, "str")],
            ["node", "battery", "storage_state_start", b64(0.0, "float")],
            ["node", "battery", "existing", b64(SU, "float")],
            ["node", "battery", "inflow", _tmap({"t0003": E})],
            ["connection", "battery_inverter", "existing", b64(0.0, "float")]]


def _solve(tmp_path_factory, alt: str, values: list, *, entities=None, chain=CHAIN, all_steps: bool = False):
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities)
    steps = run(tmp_path_factory, url, scen)
    return list(steps.values()) if all_steps else list(steps.values())[-1]


def _states(step) -> dict[tuple[str, str], float]:
    """Battery state in MWh by (d, t)."""
    fd = step.flex_data
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    frame = utf8(step.solution.value("v_state")).filter(pl.col("n") == "battery")
    return {(d, t): v * su for d, t, v in frame.select("d", "t", "value").iter_rows()}


def _lam(loss: float) -> float:
    return math.log1p(loss)


def _uniform(loss: float, hours: float) -> float:
    """10 MWh charged at a constant rate over a step of ``hours``: the state at the end of the step."""
    x = hours * _lam(loss)
    return E * (1 - math.exp(-x)) / x


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("h", [1e-8, 1e-4, 9.99e-3, 1e-2, 0.0597, 0.239, 1.17, 1.67, 12.0])
def test_alpha_beta_closed_form(h):
    """α + β = h; the row reproduces e^{−h} and (1 − e^{−h})/h; series and closed form agree at the switch."""
    a, b = _self_discharge.alpha_beta(np.array([h]))
    a, b = float(a[0]), float(b[0])
    assert a + b == pytest.approx(h, rel=1e-15)
    assert a >= 0 and b >= 0
    assert (1 - b) / (1 + a) == pytest.approx(math.exp(-h), rel=1e-13)
    assert 1 / (1 + a) == pytest.approx(-math.expm1(-h) / h, rel=1e-13)


# ---------------------------------------------------------------------------
# Solved states against the closed forms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss,want", [(0.01, 8.0741), (0.05, 3.5033)])
def test_hourly(tmp_path_factory, loss, want):
    """Hourly steps, charge during t0003: S(t0024) = 10·e^{−21λ}·(1 − e^{−λ})/λ (before 8.0340 / 3.4185)."""
    s = _states(_solve(tmp_path_factory, f"sd_h{int(loss * 100)}", _values(loss)))
    exact = E * math.exp(-21 * _lam(loss)) * (1 - math.exp(-_lam(loss))) / _lam(loss)
    assert exact == pytest.approx(want, abs=5e-5)
    assert s[("p2020", "t0024")] == pytest.approx(exact, rel=1e-7)


@pytest.mark.parametrize("loss", [0.01, 0.05])
def test_step6(tmp_path_factory, loss):
    """6-h solve steps: the end of the first step = 10·(1 − e^{−6λ})/(6λ) (9.7073 / 8.6693; before 9.9010 at 0.01),
    and 18 h later that times e^{−18λ} (8.1155 / 3.6023)."""
    vals = _values(loss) + [["solve", "y2020_2day_dispatch", "new_stepduration", b64(6.0, "float")]]
    s = _states(_solve(tmp_path_factory, f"sd_s6{int(loss * 100)}", vals))
    p = sorted(t for d, t in s if d == "p2020")
    first = _uniform(loss, 6.0)
    assert first == pytest.approx({0.01: 9.7073, 0.05: 8.6693}[loss], abs=5e-5)
    assert s[("p2020", p[0])] == pytest.approx(first, rel=1e-7)
    assert s[("p2020", p[3])] == pytest.approx(first * math.exp(-18 * _lam(loss)), rel=1e-7)


@pytest.mark.parametrize("loss,first_want,idle_want", [(0.01, 8.8956, 0.78757), (0.05, 5.8920, 0.31007)])
def test_step24(tmp_path_factory, loss, first_want, idle_want):
    """24-h solve steps: S(24 h) = 10·(1 − e^{−24λ})/(24λ) and S(48 h)/S(24 h) = e^{−24λ} (before 9.9010 / 0.9901)."""
    vals = _values(loss) + [["solve", "y2020_2day_dispatch", "new_stepduration", b64(24.0, "float")]]
    s = _states(_solve(tmp_path_factory, f"sd_s24{int(loss * 100)}", vals))
    p = sorted(t for d, t in s if d == "p2020")
    assert len(p) == 2
    assert s[("p2020", p[0])] == pytest.approx(_uniform(loss, 24.0), rel=1e-7)
    assert _uniform(loss, 24.0) == pytest.approx(first_want, abs=5e-5)
    assert s[("p2020", p[1])] / s[("p2020", p[0])] == pytest.approx(math.exp(-24 * _lam(loss)), rel=1e-7)
    assert math.exp(-24 * _lam(loss)) == pytest.approx(idle_want, abs=5e-6)


def _coarse_values(loss: float) -> tuple[list, list]:
    entities = [["group", "daily", None], ["group__node", ["daily", "battery"], None]]
    return _values(loss) + [["group", "daily", "new_stepduration", b64(24.0, "float")]], entities


@pytest.mark.parametrize("loss", [0.01, 0.05])
def test_coarse24(tmp_path_factory, loss):
    """The battery in a 24-h coarse group next to the hourly west: the same closed forms at coarse-step level
    (8.8956 / 5.8920, idle factor 0.78757 / 0.31007; before 8.0645 / 4.5455 and 0.8065 / 0.4545)."""
    vals, ents = _coarse_values(loss)
    step = _solve(tmp_path_factory, f"sd_c24{int(loss * 100)}", vals, entities=ents)
    assert step.flex_data.nodeCoarse is not None
    s = _states(step)
    assert s[("p2020", "t0024")] == pytest.approx(_uniform(loss, 24.0), rel=1e-7)
    assert s[("p2020", "t0001")] == pytest.approx(s[("p2020", "t0024")], rel=1e-12)
    assert s[("p2020", "t0048")] / s[("p2020", "t0024")] == pytest.approx(math.exp(-24 * _lam(loss)), rel=1e-7)


def _block_values(loss: float) -> tuple[list, list]:
    """bind_intraperiod_blocks over two 24-h timeset blocks t0001 and t0026 (a one-step gap keeps them apart)."""
    entities = [["timeset", "two_days", None]]
    vals = _values(loss, method="bind_intraperiod_blocks", start="fix_nothing") + [
        ["timeset", "two_days", "timeline", b64("y2020", "str")],
        ["timeset", "two_days", "timeset_duration",
         b64({"index_type": "str", "index_name": "time", "data": [["t0001", 24.0], ["t0026", 24.0]]}, "map")],
        ["solve", "y2020_2day_dispatch", "period_timeset",
         b64({"index_type": "str", "index_name": "period", "data": [["p2020", "two_days"]]}, "map")]]
    return vals, entities


def test_block_cycle(tmp_path_factory):
    """The two blocks loop: S_A = g·10/(1 − a²), S_B = a·S_A, a = e^{−24λ}, g = (1 − a)/(24λ): loss 0.01 →
    23.425 / 18.449 (block end states; before the block row charged loss·Δt·S on the block's start state)."""
    vals, ents = _block_values(0.01)
    s = _states(_solve(tmp_path_factory, "sd_blk", vals, entities=ents))
    a = math.exp(-24 * _lam(0.01))
    g = (1 - a) / (24 * _lam(0.01))
    s_a = g * E / (1 - a * a)
    assert s_a == pytest.approx(23.4255, abs=5e-4)
    assert s[("p2020", "t0024")] == pytest.approx(s_a, rel=1e-7)
    assert s[("p2020", "t0049")] == pytest.approx(a * s_a, rel=1e-7)


def test_fix_start_with_investment(tmp_path_factory):
    """forward_only, fix_start 0.5, a forced storage build (invest_min_total), loss 0.05: the first-step row carries
    the invested start energy as the step's previous state, (1 − β)·0.5·su on v_invest_n, the existing start energy
    (1 − β)·0.5·existing in the bound and −(1 + α)·su on the first state."""
    vals = [v for v in _values(0.05) if v[2] not in ("storage_state_start", "existing")] + [
        ["node", "battery", "storage_state_start", b64(0.5, "float")],
        ["node", "battery", "existing", b64(100.0, "float")],
        ["node", "battery", "invest_method", b64("invest_total", "str")],
        ["node", "battery", "invest_max_total", b64(1000.0, "float")],
        ["node", "battery", "invest_min_total", b64(200.0, "float")],
        ["node", "battery", "invest_cost", b64(1.0, "float")],
        ["node", "battery", "lifetime", b64(10.0, "float")],
        ["node", "battery", "virtual_unitsize", b64(100.0, "float")]]
    step = _solve(tmp_path_factory, "sd_inv", vals, chain=CHAIN + ["battery_invest", "5weeks"])
    fd = step.flex_data
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    d0, t0 = utf8(fd.dt).sort("d", "t").row(0)
    a, b = (float(x[0]) for x in _self_discharge.alpha_beta(np.array([math.log1p(0.05)])))
    pb = build(fd)
    name = f"nodeBalance_eq[battery,{d0},{t0}]"
    row = row_terms(pb, "nodeBalance_eq", "battery,")[name]
    assert row[f"v_state[battery,{d0},{t0}]"] == pytest.approx(-(1 + a) * su, rel=1e-12)
    inv = {c: v for c, v in row.items() if c.startswith("v_invest_n[battery,")}
    assert inv, row
    for v in inv.values():
        assert v == pytest.approx((1 - b) * 0.5 * su, rel=1e-12)
    lo, _ = row_bounds(pb, "nodeBalance_eq", "battery,")[name]
    inflow = float(utf8(fd.p_inflow.frame).filter((pl.col("n") == "battery") & (pl.col("d") == d0)
                                                  & (pl.col("t") == t0))["value"].sum())
    existing = float(utf8(fd.p_state_existing_capacity.frame).filter((pl.col("n") == "battery")
                                                                     & (pl.col("d") == d0))["value"][0])
    assert lo == pytest.approx(-inflow - (1 - b) * 0.5 * existing, rel=1e-12)


def _roll_values(loss: float) -> list:
    return _values(loss) + [
        ["solve", "dispatch_fullYear_roll", "period_timeset",
         b64({"index_type": "str", "index_name": "period", "data": [["p2020", "fullYear"]]}, "map")],
        ["solve", "dispatch_fullYear_roll", "realized_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
        ["solve", "dispatch_fullYear_roll", "rolling_duration", b64(24.0, "float")]]


def test_roll_carry_weighting(tmp_path_factory):
    """bind_forward_only, rolling 8 h / 4 h, loss 0.05: the realized states equal the single solve's (the exact
    one-step solution is time-consistent: the roll's first step uses the carry as its previous state), and the
    first-step row of roll 1 has −(1 + α)·su on its state and −(1 − β)·carry in its bound."""
    rolls = _solve(tmp_path_factory, "sd_roll", _roll_values(0.05),
                   chain=CHAIN + ["fullYear", "dispatch_fullYear_roll"], all_steps=True)
    single = _states(_solve(tmp_path_factory, "sd_roll1", _values(0.05), chain=CHAIN + ["fullYear"]))
    realized: dict = {}
    for step in rolls:
        rd = utf8(step.flex_data.realized_dispatch)
        keep = set(zip(rd["period"], rd["step"]))
        realized.update({k: v for k, v in _states(step).items() if k in keep})
    assert len(realized) == 24
    for k, v in realized.items():
        assert v == pytest.approx(single[k], rel=1e-9, abs=1e-9), k
    fd = rolls[1].flex_data
    a, b = (float(x[0]) for x in _self_discharge.alpha_beta(np.array([math.log1p(0.05)])))
    d0, t0 = utf8(fd.dt).sort("d", "t").row(0)
    carry = float(utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == "battery")["value"][0])
    assert carry > 1.0
    pb = build(fd)
    name = f"nodeBalance_eq[battery,{d0},{t0}]"
    row = row_terms(pb, "nodeBalance_eq", "battery,")[name]
    assert {c: v for c, v in row.items() if c.startswith("v_state[")} == {
        f"v_state[battery,{d0},{t0}]": pytest.approx(-(1 + a) * SU, rel=1e-12)}
    lo, _ = row_bounds(pb, "nodeBalance_eq", "battery,")[name]
    assert lo == pytest.approx(-(1 - b) * carry, rel=1e-12)


def _lp_snapshot():
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_rp_block_start_weighted(tmp_path_factory):
    """Representative periods (lp_snapshot ``agg_rp``): at a representative block's first step the free start
    state v_state_rp_start carries su·(1 − β) and the step's own state −su·(1 + α) (before su and −su·(1 + loss))."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = _lp_snapshot()._patch_agg_rp(json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text()))
    work = tmp_path_factory.mktemp("sd_rp")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    step = list(run_chain_from_db(url, "agg_rp", work_folder=work / "w", keep_solutions=True).values())[-1]
    fd = step.flex_data
    loss = float(utf8(fd.p_state_self_discharge.frame).filter(pl.col("n") == "battery")["value"][0])
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    assert loss > 0
    dur = {(d, t): v for d, t, v in utf8(fd.p_step_duration.frame).select("d", "t", "value").iter_rows()}
    rows = row_terms(build(fd), "nodeBalance_eq", "battery,")
    firsts = utf8(fd.rp_block_first).select("d", "t").unique().rows()
    assert firsts
    for d, t in firsts:
        a, b = (float(x[0]) for x in _self_discharge.alpha_beta(np.array([dur[(d, t)] * math.log1p(loss)])))
        row = rows[f"nodeBalance_eq[battery,{d},{t}]"]
        assert row[f"v_state_rp_start[battery,{d},{t}]"] == pytest.approx(su * (1 - b), rel=1e-12)
        assert row[f"v_state[battery,{d},{t}]"] == pytest.approx(-su * (1 + a), rel=1e-12)


# ---------------------------------------------------------------------------
# The reported self-discharge is the LP's loss (output_raw/v_self_discharge, node__dt "Self discharge")
# ---------------------------------------------------------------------------


def _run_wf(tmp_path_factory, alt: str, values: list, *, entities=None, chain=CHAIN):
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities)
    wf = tmp_path_factory.mktemp(f"{alt}_wf")
    steps = list(run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True).values())
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps)
    return steps, wf, scen


def _reported(wf: Path, solve: str) -> dict[tuple[str, str], float]:
    """output_raw/v_self_discharge__<solve>.parquet, battery column, by (d, t) — the file must exist and hold rows."""
    from flextool.lean_parquet import read_lean_parquet

    path = wf / "output_raw" / f"v_self_discharge__{solve}.parquet"
    assert path.exists(), f"no v_self_discharge output for {solve}"
    df = read_lean_parquet(path)
    assert len(df) > 0 and "battery" in df.columns, df
    return {(str(d), str(t)): float(v) for (_s, d, t), v in df["battery"].items()}


def _ab(hours: float, loss: float) -> tuple[float, float]:
    a, b = _self_discharge.alpha_beta(np.array([hours * math.log1p(loss)]))
    return float(a[0]), float(b[0])


def _hand_loss(steps: list[tuple[str, str]], dur: dict, state: dict, loss: float, groups=None, loop=False,
               start: float = 0.0) -> dict:
    """α·S_b + β·S_{b⁻} per (coarse) step b in MWh, spread over its fine steps by Δt/Δ_b; b⁻ = the previous step
    (the last one when ``loop``), S before the first step = ``start``."""
    groups = groups or [[s] for s in steps]
    out: dict = {}
    for i, g in enumerate(groups):
        span = sum(dur[s] for s in g)
        a, b = _ab(span, loss)
        prev = state[groups[i - 1][-1]] if (i > 0 or loop) else start
        total = a * state[g[-1]] + b * prev
        for s in g:
            out[s] = total * dur[s] / span
    return out


def _steps_dur(fd) -> tuple[list, dict]:
    dur = {(d, t): v for d, t, v in utf8(fd.p_step_duration.frame).select("d", "t", "value").iter_rows()}
    return sorted(dur), dur


@pytest.mark.parametrize("case", ["hourly", "step24", "coarse24", "block"])
def test_output_equals_lp(tmp_path_factory, case):
    """The written loss per step equals α·S + β·S⁻ of the solution (hand-built per case) and its sum closes the
    storage's energy balance (start 0 + 10 MWh inflow − end state; the block loop loses all 10 MWh).  Before: the
    output was state·loss·su (step24: 4.729 reported against 0.197 in the LP at 0.01)."""
    loss = 0.01
    ents = None
    vals = _values(loss)
    if case == "step24":
        vals += [["solve", "y2020_2day_dispatch", "new_stepduration", b64(24.0, "float")]]
    elif case == "coarse24":
        vals, ents = _coarse_values(loss)
    elif case == "block":
        vals, ents = _block_values(loss)
    steps, wf, _ = _run_wf(tmp_path_factory, f"sdo_{case}", vals, entities=ents)
    step = steps[-1]
    rep = _reported(wf, step.solve_name)
    st = _states(step)
    order, dur = _steps_dur(step.flex_data)
    groups = None
    if case == "coarse24":
        groups = [order[:24], order[24:48]]
    elif case == "block":
        groups = [[s for s in order if s[1] <= "t0024"], [s for s in order if s[1] >= "t0026"]]
    want = _hand_loss(order, dur, st, loss, groups=groups, loop=(case == "block"))
    assert set(rep) == set(want)
    for k, v in want.items():
        assert rep[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k
    closure = E if case == "block" else E - st[order[-1]]
    assert sum(rep.values()) == pytest.approx(closure, rel=1e-9)


def test_output_node_dt_column(tmp_path_factory):
    """node__dt "Self discharge" (MW) × step duration sums to the LP's loss (step24 case, write_outputs csv)."""
    import pandas as pd

    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    vals = _values(0.01) + [["solve", "y2020_2day_dispatch", "new_stepduration", b64(24.0, "float")]]
    steps, wf, scen = _run_wf(tmp_path_factory, "sdo_csv", vals)
    step = steps[-1]
    write_outputs(scenario_name=scen, output_location=str(wf), subdir=scen,
                  output_config_path=str(REPO / "templates" / "default_plots.yaml"), write_methods=["csv"],
                  fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
                  solution=step.solution, solve_name=step.solve_name,
                  solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps],
                  flex_data_provider=FlexDataProvider(), debug=True)
    nd = pd.read_csv(next((wf / "output_csv").rglob("node__dt.csv")), header=[0, 1])
    col = [c for c in nd.columns if c[0] == "battery" and c[1] == "Self discharge"][0]
    tcol = [c for c in nd.columns if c[0] == "time"][0]
    _, dur = _steps_dur(step.flex_data)
    reported = sum(abs(float(v)) * dur[("p2020", t)] for t, v in zip(nd[tcol], nd[col]))
    lp = sum(_reported(wf, step.solve_name).values())
    assert lp > 1.0
    assert reported == pytest.approx(lp, rel=1e-3)      # node__dt is rounded to 4 decimals


def test_output_roll(tmp_path_factory):
    """bind_forward_only rolling 8 h / 4 h, loss 0.05: each roll's written loss at its first step uses the carry
    as the previous state, (α·S + β·carry)·… — the realized union of the outputs equals the single solve's loss."""
    steps, wf, _ = _run_wf(tmp_path_factory, "sdo_roll", _roll_values(0.05),
                           chain=CHAIN + ["fullYear", "dispatch_fullYear_roll"])
    single_steps, wf1, _ = _run_wf(tmp_path_factory, "sdo_roll1", _values(0.05), chain=CHAIN + ["fullYear"])
    single = _reported(wf1, single_steps[-1].solve_name)
    union: dict = {}
    for step in steps:
        union.update(_reported(wf, step.solve_name))
    assert len(union) == 24
    for k, v in union.items():
        assert v == pytest.approx(single[k], rel=1e-8, abs=1e-12), k


def test_output_cyclic_roll(tmp_path_factory):
    """bind_within_timeblock rolling 8 h / 4 h: a roll's first realized step starts from the roll's own wrap (its
    last, look-ahead step), which the realized union of the outputs does not hold — the writer computes it from the
    roll's own solution."""
    vals = [v for v in _roll_values(0.05) if v[2] not in ("storage_binding_method", "storage_start_end_method")]
    vals += [["node", "battery", "storage_binding_method", b64("bind_within_timeblock", "str")],
             ["node", "battery", "storage_start_end_method", b64("fix_nothing", "str")],
             ["connection", "battery_inverter", "existing", b64(10.0, "float")]]
    vals = [v for v in vals if not (v[0] == "connection" and v[2] == "existing" and v[3][0] == b64(0.0, "float")[0])]
    steps, wf, _ = _run_wf(tmp_path_factory, "sdo_croll", vals, chain=CHAIN + ["fullYear", "dispatch_fullYear_roll"])
    assert len(steps) >= 3
    for step in steps:
        fd = step.flex_data
        st = _states(step)
        order, dur = _steps_dur(fd)
        rep = _reported(wf, step.solve_name)
        want = _hand_loss(order, dur, st, 0.05, loop=True)        # the roll is one timeblock part: it loops
        assert rep
        for k, v in rep.items():
            assert v == pytest.approx(want[k], rel=1e-9, abs=1e-12), (step.solve_name, k)


def _guard_links_match_lp(fd) -> int:
    """Every storage nodeBalance_eq row's previous-state column (the v_state column with a positive coefficient)
    is the step _self_discharge.previous_steps gives for that row — the β weight and the state change point at the
    same state.  Returns the number of rows checked."""
    links = _self_discharge.state_links(fd, has_rp=fd.nodeState_rp is not None and fd.nodeState_rp.height > 0,
                                        has_nsb=fd.nodeStateBlock is not None and fd.nodeStateBlock.height > 0)
    prev = {(n, d, t): (dp, tp) for n, d, t, dp, tp in _self_discharge.previous_steps(links).iter_rows()}
    rows = row_terms(build(fd), "nodeBalance_eq")
    storage = set(utf8(fd.nodeState)["n"].to_list())     # the binding sets also list balance nodes
    checked = 0
    for name, cols in rows.items():
        n, d, t = name[name.index("[") + 1:-1].split(",")
        if n not in storage:
            continue
        pos = [c for c, v in cols.items() if c.startswith("v_state[") and v > 0]
        if (n, d, t) in prev and prev[(n, d, t)] != (d, t):
            dp, tp = prev[(n, d, t)]
            assert pos == [f"v_state[{n},{dp},{tp}]"], (name, pos, prev[(n, d, t)])
            checked += 1
    return checked


def test_output_stochastic(tmp_path_factory):
    """Stochastic 2-day dispatch with a lossy hydro reservoir: v_self_discharge has exactly the row keys of the
    v_state output (the realized and branch (d, t) the state output writes) and non-zero values; the model's
    previous-state columns are the writer's links."""
    from flextool.engine import run_chain_from_db
    from flextool.lean_parquet import read_lean_parquet
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((REPO / "tests" / "fixtures" / "stochastics.json").read_text())
    data = _lp_snapshot()._overlay(data, name="custom", base="2_day_stochastic_dispatch",
                                   values=[["node", "hydro_reservoir", "self_discharge_loss", 0.01]])
    work = tmp_path_factory.mktemp("sd_stoch")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = list(run_chain_from_db(url, "custom", work_folder=work / "w", keep_solutions=True).values())
    for step in steps:
        out = work / "w" / "output_raw"
        sd = read_lean_parquet(out / f"v_self_discharge__{step.solve_name}.parquet")
        st = read_lean_parquet(out / f"v_state__{step.solve_name}.parquet")
        assert list(sd.index) == list(st.index)
        assert "hydro_reservoir" in sd.columns and (sd["hydro_reservoir"].abs() > 0).any()
        assert any("_" in str(d) for _s, d, _t in sd.index) or len(steps) > 1  # branch periods are written
        assert _guard_links_match_lp(step.flex_data) > 0


@pytest.mark.parametrize("fixture", ["within_solve", "rp"])
def test_links_match_lp(tmp_path_factory, fixture):
    """The previous-state links of the writer are the model's state change on a two-period bind_within_solve
    solve (5weeks: p2020's first step wraps to p2025's last) and an rp solve."""
    if fixture == "within_solve":
        vals = [["node", "battery", "storage_binding_method", b64("bind_within_solve", "str")]]
        fd = _solve(tmp_path_factory, "sd_lk_ws", vals, chain=["init", "west", "wind", "battery", "5weeks"]).flex_data
        assert len(set(utf8(fd.dt)["d"].to_list())) > 1
    else:
        from flextool.engine import run_chain_from_db
        from flextool.update_flextool.db_migration import migrate_database
        from tests.db_utils import json_to_db

        data = _lp_snapshot()._patch_agg_rp(json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text()))
        work = tmp_path_factory.mktemp("sd_lk_rp")
        (work / "f.json").write_text(json.dumps(data))
        url = json_to_db(work / "f.json", work / "f.sqlite")
        migrate_database(url)
        fd = list(run_chain_from_db(url, "agg_rp", work_folder=work / "w", keep_solutions=True).values())[-1].flex_data
    assert _guard_links_match_lp(fd) > 0
