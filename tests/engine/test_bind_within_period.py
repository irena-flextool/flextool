"""``bind_within_period`` storage: the state cycles within each FlexTool period, blocks chain inside it (#124).

reference.md: "state cycles within each FlexTool period; blocks chain inside the period" (GMPL:
``v_state[n,d,t] − v_state[n,d,t_previous]`` with the period's first step wrapping to its own last step).  Before
c22 the method had no implementation: such a node got no state-change term and no error.  Databases are built
from JSON fixtures plus an override alternative (``tools/lp_snapshot.py`` overlays), never a checked-in sqlite.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.model import _self_discharge
from flextool.engine.model._pdt_join import compute_nodeState_dt
from flextool.engine.preprocess.flex_data import _within_period_lag
from tests.engine import _rp_hourly_ref as H
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures"
L = H.L
WP = "bind_within_period"
RP_SOLVE = "lh2_rp_invest__lh2_three_region_rp_invest"


def _run(tmp_path_factory, json_name: str, patch, scenario: str = "custom", *, with_work: bool = False):
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = patch(json.loads((FIX / json_name).read_text()))
    work = tmp_path_factory.mktemp("bind_within_period")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "w", keep_solutions=True)
    assert steps
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal
    return (steps, work / "w") if with_work else steps


def _overlay(base, values, entities=()):
    def patch(data):
        return L._overlay(data, name="custom", base=base, values=values, entities=entities)
    return patch


def _state_cols(cols: dict) -> dict:
    return {c: v for c, v in cols.items() if c.startswith("v_state[")}


# ---------------------------------------------------------------------------
# The lag
# ---------------------------------------------------------------------------


def test_within_period_lag_wraps_each_period_to_its_own_last_step():
    """p1 (two blocks, the solve's first period: wraps to the solve's last period p2), p2 (continues from p1,
    shares p1's step labels), p3 (one step), p4 (wraps onto itself)."""
    rows = [
        # d, t, t_previous, d_previous
        ("p1", "t1", "t4", "p2"), ("p1", "t2", "t1", "p1"), ("p1", "t3", "t2", "p1"), ("p1", "t4", "t3", "p1"),
        ("p2", "t1", "t4", "p1"), ("p2", "t2", "t1", "p2"), ("p2", "t3", "t2", "p2"), ("p2", "t4", "t3", "p2"),
        ("p3", "t9", "t4", "p2"),
        ("p4", "t5", "t6", "p4"), ("p4", "t6", "t5", "p4"),
    ]
    dtttdt = pl.DataFrame(rows, schema=["d", "t", "t_previous", "d_previous"], orient="row")
    lag = _within_period_lag(dtttdt)
    got = {(d, t): tp for d, t, tp in lag.select("d", "t", "t_previous_within_period").iter_rows()}
    assert got == {
        ("p1", "t1"): "t4", ("p1", "t2"): "t1", ("p1", "t3"): "t2", ("p1", "t4"): "t3",
        ("p2", "t1"): "t4", ("p2", "t2"): "t1", ("p2", "t3"): "t2", ("p2", "t4"): "t3",
        ("p3", "t9"): "t9",
        ("p4", "t5"): "t6", ("p4", "t6"): "t5",
    }


# ---------------------------------------------------------------------------
# Two periods on one timeline, five blocks per period (tests.json, wind_battery_invest)
# ---------------------------------------------------------------------------

_BLOCK_FIRSTS = ("t0001", "t0009", "t0013", "t0025", "t0037")


@pytest.fixture(scope="module")
def two_period(tmp_path_factory):
    steps, work = _run(tmp_path_factory, "tests.json",
                       _overlay("wind_battery_invest", [["node", "battery", "storage_binding_method", WP]]),
                       scenario="custom", with_work=True)
    step = steps["y2020_5week"]
    return step, build(step.flex_data), (steps, work)


def test_state_change_wraps_and_chains_within_each_period(two_period):
    step, pb, _ = two_period
    su = float(step.flex_data.p_state_unitsize.frame["value"][0])
    # The previous state carries su·(1 − β) (its self-discharge weight; hourly steps, one β for every step).
    beta = float(_self_discharge.loss_coefficients(step.flex_data, compute_nodeState_dt(step.flex_data)).frame["beta"].max())
    for d in ("p2020", "p2025"):
        rows = row_terms(pb, "nodeBalance_eq", f"battery,{d},")
        # First step of the period: the state change from the period's own last step.
        first = _state_cols(rows[f"nodeBalance_eq[battery,{d},t0001]"])
        assert first[f"v_state[battery,{d},t0040]"] == pytest.approx(su * (1 - beta), rel=1e-12)
        assert first[f"v_state[battery,{d},t0001]"] < 0
        # A block boundary: the previous block's last step (blocks chain inside the period).
        boundary = _state_cols(rows[f"nodeBalance_eq[battery,{d},t0009]"])
        assert boundary[f"v_state[battery,{d},t0004]"] == pytest.approx(su * (1 - beta), rel=1e-12)
        # No lag into the other period.
        for row, cols in rows.items():
            assert all(f",{d}," in c for c in _state_cols(cols)), (row, cols)


def test_fix_start_pins_the_wrap_step_and_no_reference_end_pin(two_period):
    """fix_start: the state at the last step of the first period (the step the first period wraps to);
    use_reference_value has no end pin for a cyclic method (as bind_within_solve)."""
    _, pb, _ = two_period
    cyc = row_terms(pb, "storage_state_start_binding_cyclic_period", "battery,")
    assert set(cyc) == {"storage_state_start_binding_cyclic_period[battery,p2020,t0040]"}
    assert not row_terms(pb, "storage_state_start_binding", "battery,")
    assert not row_terms(pb, "storage_state_solve_horizon_reference_value", "battery,")


def test_output_state_change_uses_the_within_period_lag(two_period):
    """node_state_change (calc_storage_vre) of a bind_within_period node: (v_state[t] − v_state[t⁻])·unitsize
    with the model's lag (the first step from the period's own last step), summing to zero over the period."""
    from flextool.process_outputs.process_results import post_process_results
    from flextool.process_outputs.write_outputs import _read_outputs

    step, _, (steps, work) = two_period
    par, sets, v = _read_outputs(
        str(work / "output_raw"), flex_data=step.flex_data, solution=step.solution, solve_name=step.solve_name,
        solve_steps=[(st.solve_name, st.flex_data, st.effective_solution) for st in steps.values()],
        flex_data_provider=step.flex_data_provider)
    r = post_process_results(par, sets, v)
    change = r.node_state_change_dt["battery"]
    su = float(step.flex_data.p_state_unitsize.frame["value"][0])
    state = {(str(d), str(t)): float(x) for d, t, x in (
        step.solution.value("v_state").with_columns(pl.col("n").cast(pl.Utf8))
        .filter(pl.col("n") == "battery").select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8), "value")
        .iter_rows())}
    lag = {(str(d), str(t)): str(tp) for d, t, tp in step.flex_data.dtttdt_within_period.select(
        "d", "t", "t_previous_within_period").iter_rows()}
    assert len(change) > 0
    for (d, t), value in change.items():
        expected = (state[(d, t)] - state[(d, lag[(d, t)])]) * su
        assert value == pytest.approx(expected, abs=1e-6), (d, t)
    assert change.loc[("p2020", "t0001")] == pytest.approx((state[("p2020", "t0001")] - state[("p2020", "t0040")]) * su)
    assert abs(change).max() > 1e-6
    assert change.groupby(level="period").sum().abs().max() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Coarse (24 h group) nodes, shared base names, and the RP downgrade
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exp", ["e1", "e2"])
def test_hourly_matches_one_block_per_period_timeblock(tmp_path_factory, exp):
    """Plain bind_within_period on the hourly twins of the rp experiments (lh2 nodes in the fixture's 24 h
    group, several blocks per period) equals one 168 h bind_within_timeblock block per period."""
    fn = H.EXPERIMENTS[exp]
    wp = _run(tmp_path_factory, "lh2_three_region_rp_invest.json", fn("hourly", WP))[RP_SOLVE]
    tb = _run(tmp_path_factory, "lh2_three_region_rp_invest.json",
              H.one_block(fn, "hourly", "bind_within_timeblock"))[RP_SOLVE]
    assert float(wp.solution.obj) == pytest.approx(float(tb.solution.obj), rel=1e-8)


def _week168(method: str):
    return _overlay("lh2_three_region_rp_invest", [
        ["solve", RP_SOLVE, "period_timeset",
         {"type": "map", "index_type": "str", "index_name": "period", "rank": 1,
          "data": [["y2030", "week168"], ["y2040", "week168"]]}]]
        + [["node", n, "storage_binding_method", method] for n in ("lh2_A", "lh2_B", "lh2_C")])


def test_downgraded_blended_node_cycles_within_each_period(tmp_path_factory):
    """On a non-RP solve bind_within_period_blended_weights becomes bind_within_period: the lh2 nodes keep a
    state chain (before: none), the same LP as the plain method."""
    down = _run(tmp_path_factory, "lh2_three_region_rp_invest.json",
                _week168("bind_within_period_blended_weights"))[RP_SOLVE]
    plain = _run(tmp_path_factory, "lh2_three_region_rp_invest.json", _week168(WP))[RP_SOLVE]
    fd = down.flex_data
    assert sorted(str(n) for n in fd.storage_bind_within_period["n"].to_list()) == ["lh2_A", "lh2_B", "lh2_C"]
    rows = row_terms(build(fd), "nodeBalanceCoarse_eq", "lh2_A,y2040,t0001")
    cols = _state_cols(rows["nodeBalanceCoarse_eq[lh2_A,y2040,t0001]"])
    assert cols["v_state[lh2_A,y2040,t0096]"] > 0          # y2040 wraps to its own last step, not y2030's
    assert not any(",y2030," in c for c in cols)
    assert float(down.solution.obj) == pytest.approx(float(plain.solution.obj), rel=1e-9)


# ---------------------------------------------------------------------------
# Stochastic and rolling solves
# ---------------------------------------------------------------------------


def test_stochastic_forecast_periods_cycle_on_themselves(tmp_path_factory):
    steps = _run(tmp_path_factory, "stochastics.json",
                 _overlay("2_day_stochastic_dispatch", [["node", "hydro_reservoir", "storage_binding_method", WP]]))
    step = steps["2day_dispatch"]
    pb = build(step.flex_data)
    for d in ("period1_lower", "period1_mid", "period1_upper"):
        cols = _state_cols(row_terms(pb, "nodeBalance_eq", f"hydro_reservoir,{d},t0001")[
            f"nodeBalance_eq[hydro_reservoir,{d},t0001]"])
        assert cols.get(f"v_state[hydro_reservoir,{d},t0048]", 0.0) > 0, (d, cols)


def test_rolling_roll_wraps_to_its_last_step_in_the_period(tmp_path_factory):
    """In a roll the period's steps are the roll's: its first step wraps to the roll's last step of the period
    (how_to.md: use whole periods as rolling_jump)."""
    steps = _run(tmp_path_factory, "stochastics.json",
                 _overlay("1_week_rolling_wind", [["node", "hydro_reservoir", "storage_binding_method", WP]]))
    assert len(steps) > 1
    for step in steps.values():
        fd = step.flex_data
        lag = fd.dtttdt_within_period.with_columns([pl.col(c).cast(pl.Utf8) for c in fd.dtttdt_within_period.columns])
        dt = fd.dt.with_columns(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
        for d in lag["d"].unique().to_list():
            steps_d = dt.filter(pl.col("d") == d)["t"].sort().to_list()
            first = lag.filter((pl.col("d") == d) & (pl.col("t") == steps_d[0]))
            assert first["t_previous_within_period"].to_list() == [steps_d[-1]], (d, steps_d[0])

