"""``bind_intraperiod_blocks`` storage nodes keep their block balance in every sub-solve that can have one (item 113).

A ``bind_intraperiod_blocks`` node (``nodeStateBlock``) is balanced once per timeset block by
``nodeBalanceBlock_eq`` (``nodeBalance_eq`` skips it) and its state is constant within each block
(``stateConstantWithinBlock_eq``).  Before, a sub-solve that is not a Spine solve (every roll, every nested anchor
``<solve>_<period>``) never got the block frames, so the node had no energy balance at all (energy from nothing).
Now: a nested anchor sub-solve gets its blocks and the model refuses a block node without blocks.  In a rolling
solve (owner decisions Q-c15c-A option B and Q-c15e-A option (b)) the blocks of a roll are also cut at its realized
end; a period the roll holds whole loops as in a solve that is not rolling, and in a period the roll holds only in
part the blocks chain forward and the storage level is carried from roll to roll.

Fixtures: ``tests/fixtures/tests.json`` (``5weeks`` = five 4-step blocks; ``bind_battery_intraperiod_blocks`` on the
battery) with override alternatives, built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, run, utf8

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"


def _chain(scenario: str) -> list[str]:
    data = json.loads((FIX / "tests.json").read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


# ---------------------------------------------------------------------------
# Rolling solves (Q-c15c-A option B, Q-c15e-A option (b))

ROLL_CHAIN = ["init", "west", "wind", "coal", "battery", "fullYear", "dispatch_fullYear_roll",
              "bind_battery_intraperiod_blocks"]
ROLL = "dispatch_fullYear_roll"


def _roll_values(horizon: float, jump: float, periods: tuple[str, ...] = ("p2020",), *, loss: float = 0.0) -> list:
    return [["solve", ROLL, "rolling_solve_horizon", b64(horizon, "float")],
            ["solve", ROLL, "rolling_solve_jump", b64(jump, "float")],
            ["solve", ROLL, "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [[p, "fullYear"] for p in periods]}, "map")],
            ["solve", ROLL, "realized_periods", b64({"value_type": "str", "data": list(periods)}, "array")],
            ["node", "battery", "self_discharge_loss", b64(loss, "float")]]


def _q4_values(*, loss: float = 0.0, inflow: dict | None = None, start: float = 0.0,
               extra: list | None = None) -> list:
    """48-h rolls jumping 24 h over the 72-h p2020 of fullYear, inverter off, 10 MWh into the battery in t0003
    (or *inflow*), fix_start *start*, horizon free: rolls hold t0001–t0048, t0025–t0072 and t0049–t0072."""
    from tests.engine._unit_db import time_map

    return (_roll_values(48.0, 24.0, loss=loss)
            + [["node", "battery", "storage_solve_horizon_method", b64("free", "str")],
               ["node", "battery", "storage_state_start", b64(start, "float")],
               ["node", "battery", "inflow", time_map(inflow if inflow is not None else {"t0003": 10.0}, 0.0)],
               ["connection", "battery_inverter", "existing", b64(0.0, "float")]]
            + (extra or []))


def _rolls(tmp_path_factory, alt: str, values: list) -> list:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=ROLL_CHAIN)
    steps = run(tmp_path_factory, url, scen)
    return [steps[k] for k in sorted((k for k in steps if k.startswith(f"{ROLL}_roll_")),
                                     key=lambda k: int(k.rsplit("_", 1)[1]))]


def _block_sizes(fd) -> dict[tuple[str, str], int]:
    return {(d, b): n for d, b, n in utf8(fd.period_block_time).group_by("d", "b_first").len().iter_rows()}


def _lag(fd) -> dict[tuple[str, str], tuple[str, str]]:
    return {(d, t): (dp, tp) for d, t, dp, tp in utf8(fd.dtttdt_block_lag).iter_rows()}


def _states_mwh(step) -> dict[tuple[str, str], float]:
    v = utf8(step.solution.value("v_state")).filter(pl.col("n") == "battery")
    return {(d, t): x * SU for d, t, x in v.select("d", "t", "value").iter_rows()}


def _state_cols(row: dict[str, float]) -> dict[str, float]:
    return {c: v for c, v in row.items() if c.startswith("v_state[") and abs(v) > 1e-12}


def _carry_mwh(fd) -> float:
    return float(utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == "battery")["value"][0])


def test_rolling_blocks_cut_at_realized_end(tmp_path_factory) -> None:
    """The blocks of each roll are its timeset blocks cut after the last realized step when the roll continues past
    it: {t0001: 24, t0025: 24}, {t0025: 24, t0049: 24}, {t0049: 24}.  p2020 (72 h) is held in part by every roll, so
    its blocks chain forward: the roll's first step has no lag row, a later block continues from the previous step.
    Before: the run stopped with an error (c15c interim)."""
    rolls = _rolls(tmp_path_factory, "ibr_cut", _q4_values())
    want = [{("p2020", "t0001"): 24, ("p2020", "t0025"): 24}, {("p2020", "t0025"): 24, ("p2020", "t0049"): 24},
            {("p2020", "t0049"): 24}]
    assert len(rolls) == 3
    for step, sizes in zip(rolls, want):
        fd = step.flex_data
        assert fd.solve_is_rolling and utf8(fd.period_block_chain)["d"].to_list() == ["p2020"]
        assert _block_sizes(fd) == sizes
        lag = _lag(fd)
        firsts = sorted(b for _d, b in sizes)
        assert ("p2020", firsts[0]) not in lag
        for b in firsts[1:]:
            assert lag[("p2020", b)] == ("p2020", f"t{int(b[1:]) - 1:04d}")


@pytest.mark.parametrize("loss", [0.0, 0.01])
def test_rolling_blocks_carry(tmp_path_factory, loss) -> None:
    """The level is carried from roll to roll and each block decays by its own 24 h (a = e^{−24λ}, g = (1 − a)/(24λ),
    λ = ln(1 + loss)): the realized blocks hold 10·g, 10·g·a, 10·g·a² MWh (loss 0.01: 8.8956, 7.0059, 5.5176; loss 0:
    10 everywhere) — the trajectory of one forward chain.  Roll 1's first block row holds −su on its own state, the
    carried level (the state at roll 0's realized end) in its bounds and no state of a later block (no loop).
    Before: the run stopped with an error (c15c interim)."""
    import math

    rolls = _rolls(tmp_path_factory, f"ibr_carry{int(loss * 100)}", _q4_values(loss=loss))
    h = 24 * math.log1p(loss)
    a, g = (math.exp(-h), (1 - math.exp(-h)) / h) if loss > 0 else (1.0, 1.0)
    for i, (step, b_first) in enumerate(zip(rolls, ("t0001", "t0025", "t0049"))):
        s = _states_mwh(step)
        for k in range(int(b_first[1:]), int(b_first[1:]) + 24):
            assert s[("p2020", f"t{k:04d}")] == pytest.approx(10.0 * g * a ** i, rel=1e-7), (i, k)
    # The carry handed to a roll is the previous roll's state at its realized end (the step before the roll's first).
    for prev, step, first in zip(rolls, rolls[1:], (25, 49)):
        assert _carry_mwh(step.flex_data) == pytest.approx(_states_mwh(prev)[("p2020", f"t{first - 1:04d}")],
                                                           rel=1e-9)
    if loss == 0.0:
        pb = build(rolls[1].flex_data)
        row = "nodeBalanceBlock_eq[battery,p2020,t0025]"
        assert _state_cols(row_terms(pb, "nodeBalanceBlock_eq", "battery,")[row]) == {
            "v_state[battery,p2020,t0048]": pytest.approx(-SU, rel=1e-12)}
        lo, hi = row_bounds(pb, "nodeBalanceBlock_eq", "battery,")[row]
        assert lo == hi == pytest.approx(-_carry_mwh(rolls[1].flex_data), rel=1e-12) and lo == pytest.approx(-10.0)


@pytest.mark.parametrize("loss", [0.0, 0.01])
def test_rolling_blocks_open_start(tmp_path_factory, loss) -> None:
    """fix_nothing: roll 0's first block has no state change — its state columns sum to −h_b·su (0 without a loss:
    the constant state of an open block only decays) and its bounds hold no start constant — while roll 1 starts
    from the carried level.  No start pin of any family."""
    import math
    vals = _q4_values(loss=loss, inflow={"t0030": 10.0},
                      extra=[["node", "battery", "storage_start_end_method", b64("fix_nothing", "str")]])
    rolls = _rolls(tmp_path_factory, f"ibr_open{int(loss * 100)}", vals)
    pb0 = build(rolls[0].flex_data)
    row = "nodeBalanceBlock_eq[battery,p2020,t0001]"
    h_b = 24 * math.log1p(loss)
    assert sum(_state_cols(row_terms(pb0, "nodeBalanceBlock_eq", "battery,")[row]).values()) == pytest.approx(
        -h_b * SU, abs=1e-9)
    assert row_bounds(pb0, "nodeBalanceBlock_eq", "battery,")[row] == (0.0, 0.0)
    for fam in ("storage_state_start_binding", "storage_state_start_binding_cyclic_period"):
        assert row_terms(pb0, fam, "battery,") == {}
    pb1 = build(rolls[1].flex_data)
    carry = _carry_mwh(rolls[1].flex_data)
    assert carry == pytest.approx(_states_mwh(rolls[0])[("p2020", "t0024")], abs=1e-9)
    if loss > 0:
        # The open first block has no flows (inflow only at t0030) and loses h_b·S·su = 0, so its state is 0.
        assert carry == pytest.approx(0.0, abs=1e-9)
    lo, _hi = row_bounds(pb1, "nodeBalanceBlock_eq", "battery,")["nodeBalanceBlock_eq[battery,p2020,t0025]"]
    # The carried level enters the row weighted by 1 − β of the block (β = 1 − h_b/(e^{h_b} − 1); 0 without a loss).
    beta = 0.0 if loss == 0 else 1 - h_b / math.expm1(h_b)
    assert lo == pytest.approx(-(1 - beta) * carry - 10.0, abs=1e-9)


def test_rolling_blocks_fix_start(tmp_path_factory) -> None:
    """fix_start 0.1, no inflow, inverter off, no loss: roll 0's first block starts from 0.1 · 50 = 5 MWh (the start
    energy in its bounds; no start pin row), and every block of every roll holds 5 MWh."""
    vals = _q4_values(inflow={}, start=0.1)
    rolls = _rolls(tmp_path_factory, "ibr_fix", vals)
    pb0 = build(rolls[0].flex_data)
    assert row_bounds(pb0, "nodeBalanceBlock_eq", "battery,")["nodeBalanceBlock_eq[battery,p2020,t0001]"] == (
        pytest.approx(-5.0), pytest.approx(-5.0))
    for fam in ("storage_state_start_binding", "storage_state_start_binding_cyclic_period"):
        assert row_terms(pb0, fam, "battery,") == {}
    for step in rolls:
        for v in _states_mwh(step).values():
            assert v == pytest.approx(5.0, abs=1e-9)
    # Later rolls chain on from the carried level (5 MWh), with no fix_start pin and no start energy.
    for step in rolls[1:]:
        assert _carry_mwh(step.flex_data) == pytest.approx(5.0, abs=1e-9)
        pb = build(step.flex_data)
        for fam in ("storage_state_start_binding", "storage_state_start_binding_cyclic_period"):
            assert row_terms(pb, fam, "battery,") == {}


def test_rolling_blocks_reference_value(tmp_path_factory) -> None:
    """use_reference_value 0.4 (tests.json battery, inverter on): each roll's last step is pinned to 0.4 · 50 = 20 MWh
    — the roll holds p2020 in part, so the blocks chain forward and the end is not tied to the start.  Before (in a
    solve that is not rolling the node is excluded from the pin): the run stopped with an error."""
    rolls = _rolls(tmp_path_factory, "ibr_ref", _roll_values(48.0, 24.0))
    for step, last in zip(rolls, ("t0048", "t0072", "t0072")):
        pb = build(step.flex_data)
        rows = row_terms(pb, "storage_state_solve_horizon_reference_value", "battery,")
        assert set(rows) == {f"storage_state_solve_horizon_reference_value[battery,p2020,{last}]"}
        assert _states_mwh(step)[("p2020", last)] == pytest.approx(20.0, abs=1e-6)


def test_rolling_blocks_cross_period(tmp_path_factory) -> None:
    """p2020 + p2025 (72 h each), 48-h rolls jumping 24 h: roll 2 holds p2020 t0049–t0072 and p2025 t0001–t0024,
    both in part, so p2025's first block continues from p2020's last step (forward chain across the period boundary,
    no loop) and the roll's first step has no lag row."""
    rolls = _rolls(tmp_path_factory, "ibr_cross", _roll_values(48.0, 24.0, ("p2020", "p2025")))
    fd = rolls[2].flex_data
    assert sorted(utf8(fd.period_block_chain)["d"].to_list()) == ["p2020", "p2025"]
    lag = _lag(fd)
    assert ("p2020", "t0049") not in lag
    assert lag[("p2025", "t0001")] == ("p2020", "t0072")
    row = row_terms(build(fd), "nodeBalanceBlock_eq", "battery,")["nodeBalanceBlock_eq[battery,p2025,t0001]"]
    assert set(_state_cols(row)) == {"v_state[battery,p2020,t0072]", "v_state[battery,p2025,t0024]"}


def test_rolling_whole_period_loops(tmp_path_factory, caplog) -> None:
    """p2020 + p2025 (72 h each), 96-h rolls jumping 72 h (a whole period): roll 0 holds p2020 whole — it loops, with
    the fix_start pin on its wrap (p2020, t0072) as in a solve that is not rolling — and p2025 in part (chained from
    p2020's end, with the reference-value pin at the roll's end); roll 1 holds p2025 whole: it loops and takes no
    carry although the previous roll handed one over, and has no reference-value pin.  No warning: the jump is a
    whole period."""
    import logging
    caplog.set_level(logging.WARNING)
    rolls = _rolls(tmp_path_factory, "ibr_whole", _roll_values(96.0, 72.0, ("p2020", "p2025")))
    assert len(rolls) == 2
    fd0, fd1 = rolls[0].flex_data, rolls[1].flex_data
    assert utf8(fd0.period_block_chain)["d"].to_list() == ["p2025"] and fd1.period_block_chain is None
    lag0, lag1 = _lag(fd0), _lag(fd1)
    assert lag0[("p2020", "t0001")] == ("p2020", "t0072") and lag0[("p2025", "t0001")] == ("p2020", "t0072")
    assert lag1[("p2025", "t0001")] == ("p2025", "t0072")
    pb0, pb1 = build(fd0), build(fd1)
    assert set(row_terms(pb0, "storage_state_start_binding_cyclic_period", "battery,")) == {
        "storage_state_start_binding_cyclic_period[battery,p2020,t0072]"}
    assert set(row_terms(pb0, "storage_state_solve_horizon_reference_value", "battery,")) == {
        "storage_state_solve_horizon_reference_value[battery,p2025,t0024]"}
    # The carry is roll 0's realized end, p2020's last step: the fix_start-pinned wrap, 0.1 · 50 = 5 MWh.
    assert fd1.p_roll_continue_state is not None and _carry_mwh(fd1) == pytest.approx(5.0, abs=1e-9)
    assert row_bounds(pb1, "nodeBalanceBlock_eq", "battery,")["nodeBalanceBlock_eq[battery,p2025,t0001]"] == (0.0, 0.0)
    assert row_terms(pb1, "storage_state_solve_horizon_reference_value", "battery,") == {}
    # Looping and carried (a state was realised before roll 1): neither the carry nor a fix_start pin.
    for fam in ("storage_state_start_binding", "storage_state_start_binding_cyclic_period"):
        assert row_terms(pb1, fam, "battery,") == {}
    assert not [r for r in caplog.records if "whole number of periods" in r.getMessage()]


def test_rolling_jump_not_whole_periods_warns(tmp_path_factory, caplog) -> None:
    """48-h rolls jumping 24 h over the 72-h p2020: rolls 1 and 2 start inside the period; one warning names the
    solve, the first such roll and the node."""
    import logging

    caplog.set_level(logging.WARNING)
    _rolls(tmp_path_factory, "ibr_warn", _q4_values())
    hits = [r.getMessage() for r in caplog.records if "whole number of periods" in r.getMessage()]
    assert len(hits) == 1
    assert f"'{ROLL}'" in hits[0] and f"'{ROLL}_roll_1'" in hits[0] and "battery" in hits[0]


def test_rolling_single_roll_equals_single_solve(tmp_path_factory) -> None:
    """A rolling solve with one roll over the whole p2020 holds the period whole, so its blocks loop exactly as in the
    same solve run as single_solve: the same block rows, fix_start pin and objective (the state columns identical;
    other coefficients to the precision the roll's per-solve inputs carry, e.g. 1/0.95 written as 1.052632)."""
    roll = _rolls(tmp_path_factory, "ibr_one", _roll_values(72.0, 72.0))
    assert len(roll) == 1
    url, scen = make_db(tmp_path_factory, alt="ibr_one_single",
                        values=_roll_values(72.0, 72.0) + [["solve", ROLL, "solve_mode", b64("single_solve", "str")]],
                        base_chain=ROLL_CHAIN)
    single = list(run(tmp_path_factory, url, scen).values())[-1]
    pb_r, pb_s = build(roll[0].flex_data), build(single.flex_data)
    for fam in ("nodeBalanceBlock_eq", "stateConstantWithinBlock_eq", "storage_state_start_binding_cyclic_period",
                "storage_state_solve_horizon_reference_value"):
        rows_r, rows_s = row_terms(pb_r, fam, "battery,"), row_terms(pb_s, fam, "battery,")
        assert set(rows_r) == set(rows_s), fam
        for row, terms in rows_s.items():
            assert set(rows_r[row]) == set(terms), (fam, row)
            assert _state_cols(rows_r[row]) == _state_cols(terms), (fam, row)
            for col, v in terms.items():
                assert rows_r[row][col] == pytest.approx(v, rel=1e-5), (fam, row, col)
        bounds_r = row_bounds(pb_r, fam, "battery,")
        for row, (lo, hi) in row_bounds(pb_s, fam, "battery,").items():
            assert bounds_r[row] == (pytest.approx(lo, rel=1e-9), pytest.approx(hi, rel=1e-9)), (fam, row)
    assert set(row_terms(pb_s, "storage_state_start_binding_cyclic_period", "battery,")) == {
        "storage_state_start_binding_cyclic_period[battery,p2020,t0072]"}
    assert single.obj is not None and roll[0].obj == pytest.approx(single.obj, rel=1e-5)


def test_rolling_blocks_stochastic_follow_forward_only(tmp_path_factory) -> None:
    """Rolling stochastic dispatch (stochastics.json ``1_week_rolling_wind``) with a bind_intraperiod_blocks
    hydro_reservoir: the rolls hold the week in part, so every block lag row is the bind_forward_only lag of that step
    (``dtttdt_forward_only``, whatever it links a forecast branch to — critique c15e 13: one place to change)."""
    from flextool.engine.preprocess.params._derived import dtttdt_forward_only_from_dtttdt
    from tests.engine.test_self_discharge import STOCH_JSON, _run_json

    steps = _run_json(tmp_path_factory, STOCH_JSON, "1_week_rolling_wind",
                      [["node", "hydro_reservoir", "storage_binding_method", "bind_intraperiod_blocks"]], "ibr_stoch")
    assert len(steps) > 1
    for step in steps:
        fd = step.flex_data
        assert fd.solve_is_rolling and fd.period_block_chain is not None
        lag = _lag(fd)
        fwd = {(d, t): (dp, tp) for d, t, dp, tp in utf8(dtttdt_forward_only_from_dtttdt(fd.dtttdt))
               .select("d", "t", "d_previous", "t_previous_within_solve").iter_rows()}
        block_steps = set(utf8(fd.period_block_time).select("d", "t").iter_rows())
        assert lag == {k: v for k, v in fwd.items() if k in block_steps}
        assert row_terms(build(fd), "nodeBalanceBlock_eq", "hydro_reservoir,")


def test_synthetic_subsolve_gets_blocks(tmp_path_factory) -> None:
    """Nested invest (5weeks, one solve per period) → storage (fullYear 6 h, single solve) with a
    bind_intraperiod_blocks battery: every sub-solve is a synthetic anchor ``<solve>_<period>`` and has the block
    frames of its own steps — by hand: 5weeks → blocks t0001 (4 steps), t0009 (8: t0009–t0016, the t0009 and
    t0013 weeks touch), t0025 (4), t0037 (4); fullYear_6h → one block of 12 steps from t0001 — and equal to what
    the same two solves get as Spine solves (``apply_derived_e``'s ``period_block_family_from_source``) when they
    run un-nested over all four periods.  The battery has one ``nodeBalanceBlock_eq`` row per block,
    ``stateConstantWithinBlock_eq`` rows and no ``nodeBalance_eq`` row.  Before: the anchors had no block
    frames, so the battery had 0 balance rows of any family."""
    data = json.loads((FIX / "tests.json").read_text())
    # The storage solve keeps no rolling child: the anchors are then the only sub-solves.
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if not (pv[1] == "storage_fullYear_6h" and pv[2] == "contains_solves")]
    fixture = tmp_path_factory.mktemp("ib_fx") / "tests_no_child.json"
    fixture.write_text(json.dumps(data))
    url, scen = make_db(tmp_path_factory, alt="ib_syn", values=[], fixture_json=fixture,
                        base_chain=_chain("multi_fullYear_battery_nested_multi_invest")
                        + ["bind_battery_intraperiod_blocks"])
    steps = run(tmp_path_factory, url, scen)
    assert len(steps) == 8 and all(k.rsplit("_", 1)[-1].startswith("p20") for k in steps), list(steps)
    # The same solves as Spine solves (no per-period anchors): invest_5weeks contains storage_fullYear_6h.
    url_s, scen_s = make_db(tmp_path_factory, alt="ib_spine", values=[], fixture_json=fixture,
                            base_chain=["init", "west", "wind", "coal", "battery", "5weeks", "fullYear",
                                        "storage_fullYear_6h", "invest_oneYear4Times_5weeks",
                                        "bind_battery_intraperiod_blocks"])
    spine = run(tmp_path_factory, url_s, scen_s)
    assert set(spine) == {"invest_5weeks", "storage_fullYear_6h"}, list(spine)
    for name, step in steps.items():
        fd = step.flex_data
        assert fd.nodeStateBlock is not None and utf8(fd.nodeStateBlock)["n"].to_list() == ["battery"], name
        assert fd.period_block_time is not None, name
        got = utf8(fd.period_block_time).sort("d", "b_first", "t")
        periods = sorted(set(got["d"].to_list()))
        base = name.rsplit("_", 1)[0]
        expected = ({"t0001": 4, "t0009": 8, "t0025": 4, "t0037": 4} if base == "invest_5weeks"
                    else {"t0001": 12})
        for d in periods:
            sizes = dict(got.filter(pl.col("d") == d).group_by("b_first").agg(pl.len()).rows())
            assert sizes == expected, (name, d, sizes)
        ref = utf8(spine[base].flex_data.period_block_time).filter(pl.col("d").is_in(periods))
        assert got.equals(ref.sort("d", "b_first", "t")), name
        ref_succ = utf8(spine[base].flex_data.period_block_succ).filter(pl.col("d").is_in(periods))
        assert utf8(fd.period_block_succ).sort("d", "b_first").equals(ref_succ.sort("d", "b_first")), name
        n_blocks = utf8(fd.period_block).height
        assert n_blocks == len(expected) * len(periods)
        pb = build(fd)
        assert len(row_terms(pb, "nodeBalanceBlock_eq", "battery,")) == n_blocks, name
        # One stateConstantWithinBlock_eq row per interior step of the Spine solve's blocks in these periods.
        ref_interior = utf8(spine[base].flex_data.dtttdt_block_interior).filter(pl.col("d").is_in(periods))
        assert len(row_terms(pb, "stateConstantWithinBlock_eq", "battery,")) == ref_interior.height > 0, name
        assert len(row_terms(pb, "nodeBalance_eq", "battery,")) == 0, name


# ---------------------------------------------------------------------------
# #117 — nodeBalanceBlock_eq is nodeBalance_eq summed over each block
# ---------------------------------------------------------------------------

BLOCK_CHAIN = ["init", "west", "wind", "battery", "5weeks", "bind_battery_intraperiod_blocks"]
# 5weeks blocks (the t0009 and t0013 weeks touch and merge): first step → steps of the block.
BLOCKS = {"t0001": [f"t{i:04d}" for i in range(1, 5)], "t0009": [f"t{i:04d}" for i in range(9, 17)],
          "t0025": [f"t{i:04d}" for i in range(25, 29)], "t0037": [f"t{i:04d}" for i in range(37, 41)]}
SU = 50.0          # battery existing 50 MWh → state_unitsize 50


def _no_loss() -> list:
    return [["node", "battery", "self_discharge_loss", b64(0.0, "float")]]


def _inner(name: str) -> list[str]:
    return name[name.index("[") + 1:-1].split(",")


def test_block_state_is_end_of_block(tmp_path_factory) -> None:
    """Inverter off, 10 MWh into the battery in t0003 and out in t0026, fix_start 0: the state of each block is its
    end state — p2020 blocks t0001 / t0009 / t0025 / t0037 hold 10 / 10 / 0 / 0 MWh — and the fix_start pin sits on
    the period's last step t0040 (the state the first block starts from).  Before: start-of-block labelling
    (0 / 10 / 10 / 0) and the pin on t0001."""
    from tests.engine._unit_db import time_map

    vals = _no_loss() + [["connection", "battery_inverter", "existing", b64(0.0, "float")],
                         ["node", "battery", "storage_state_start", b64(0.0, "float")],
                         ["node", "battery", "inflow", time_map({"t0003": 10.0, "t0026": -10.0}, 0.0)]]
    url, scen = make_db(tmp_path_factory, alt="ib117_end", values=vals, base_chain=BLOCK_CHAIN)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    state = utf8(step.solution.value("v_state")).filter((pl.col("n") == "battery") & (pl.col("d") == "p2020"))
    got = {t: v * SU for t, v in state.select("t", "value").iter_rows()}
    for b_first, ts in BLOCKS.items():
        want = 10.0 if b_first in ("t0001", "t0009") else 0.0
        for t in ts:
            assert got[t] == pytest.approx(want, abs=1e-6), (b_first, t, got[t])
    pb = build(step.flex_data)
    assert row_terms(pb, "storage_state_start_binding", "battery,") == {}
    pins = row_terms(pb, "storage_state_start_binding_cyclic_period", "battery,")
    assert set(pins) == {"storage_state_start_binding_cyclic_period[battery,p2020,t0040]"}, sorted(pins)


def _overlay_values() -> tuple[list, list]:
    """A lossless two-way inverter (one flow variable: v_flow_back) and a min_load_efficiency unit ``bgen``
    drawing from the battery (section input v_online · section)."""
    entities = [["unit", "bgen", None], ["unit__inputNode", ["bgen", "battery"], None],
                ["unit__outputNode", ["bgen", "west"], None]]
    vals = _no_loss() + [
        ["connection", "battery_inverter", "transfer_method", b64("no_losses_no_variable_cost", "str")],
        ["unit", "bgen", "existing", b64(10.0, "float")],
        ["unit", "bgen", "conversion_method", b64("min_load_efficiency", "str")],
        ["unit", "bgen", "efficiency", b64(0.9, "float")],
        ["unit", "bgen", "efficiency_at_min_load", b64(0.8, "float")],
        ["unit", "bgen", "min_load", b64(0.3, "float")],
        ["unit", "bgen", "startup_method", b64("linear", "str")]]
    return entities, vals


def _built(tmp_path_factory, alt: str, extra: list):
    entities, vals = _overlay_values()
    url, scen = make_db(tmp_path_factory, alt=alt, values=vals + extra, base_chain=BLOCK_CHAIN, entities=entities)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    return step.flex_data, build(step.flex_data)


def test_block_row_is_summed_node_balance(tmp_path_factory) -> None:
    """Every non-state column of a block row (flows, the inverter's v_flow_back, bgen's v_online section input,
    slacks) has the coefficient the same column has in the battery's nodeBalance_eq row when the battery is a
    plain bind_within_period node, the RHS is the summed −inflow, and the state terms are −su on the block's
    state and +su on the previous block's state (the first block of a period: the period's last block).  Before:
    no v_flow_back and no section terms, the row negated, the state change from the block to the next block."""
    fd, pb = _built(tmp_path_factory, "ib117_sum", [])
    _, pb_wp = _built(tmp_path_factory, "ib117_wp",
                      [["node", "battery", "storage_binding_method", b64("bind_within_period", "str")]])
    blocks = row_terms(pb, "nodeBalanceBlock_eq", "battery,")
    fine = row_terms(pb_wp, "nodeBalance_eq", "battery,")
    fine_bounds = row_bounds(pb_wp, "nodeBalance_eq", "battery,")
    bounds = row_bounds(pb, "nodeBalanceBlock_eq", "battery,")
    periods = sorted(set(utf8(fd.period_block)["d"].to_list()))
    assert len(blocks) == len(BLOCKS) * len(periods)
    order = list(BLOCKS)
    seen_back = seen_online = 0
    for d in periods:
        for i, (b_first, ts) in enumerate(BLOCKS.items()):
            row = blocks[f"nodeBalanceBlock_eq[battery,{d},{b_first}]"]
            want: dict[str, float] = {}
            for t in ts:
                for col, v in fine[f"nodeBalance_eq[battery,{d},{t}]"].items():
                    if not col.startswith("v_state["):
                        want[col] = v
            got = {c: v for c, v in row.items() if not c.startswith("v_state[")}
            assert set(got) == set(want), (d, b_first, sorted(set(got) ^ set(want)))
            for c, v in want.items():
                assert got[c] == pytest.approx(v, rel=1e-12, abs=1e-12), (d, b_first, c)
            seen_back += sum(c.startswith("v_flow_back[battery_inverter,") for c in got)
            seen_online += sum(c.startswith("v_online_linear[bgen,") for c in got)
            lo = sum(fine_bounds[f"nodeBalance_eq[battery,{d},{t}]"][0] for t in ts)
            assert bounds[f"nodeBalanceBlock_eq[battery,{d},{b_first}]"][0] == pytest.approx(lo, abs=1e-9)
            states = {c: v for c, v in row.items() if c.startswith("v_state[") and v != 0.0}
            prev_last = BLOCKS[order[i - 1]][-1]          # i = 0 → the period's last block (the loop)
            assert states == {f"v_state[battery,{d},{ts[-1]}]": pytest.approx(-SU, rel=1e-12),
                              f"v_state[battery,{d},{prev_last}]": pytest.approx(SU, rel=1e-12)}, (d, b_first)
    assert seen_back == sum(len(ts) for ts in BLOCKS.values()) * len(periods)
    assert seen_online == seen_back
