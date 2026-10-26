"""Where the storage chain starts and ends across solves (fix_start, the carry between solves, fix_end).

* ``bind_within_timeblock`` + ``fix_start`` pins the state the first step starts from: the wrap target of the first
  step (the last step of the first timeblock), not the state at the end of the first step.
* Later solves and rolls continue from the realised state; fix_start only where a level first appears.
* ``fix_end`` / ``fix_start_end`` pin the end of the last sub-solve of each nesting level; an upper-level nested
  fix_quantity / fix_usage at that step wins.  The horizon-end reference value yields to any nested fix.
* ``fix_start_end`` ignores the storage_binding_method: a cyclic node chains forward, a bind_intraperiod_blocks node
  keeps its blocks and chains them forward.

Databases: ``tests/fixtures/tests.json`` / ``stochastics.json`` plus an override alternative (never a checked-in
sqlite).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import b64, build, make_db, run, time_map, utf8

REPO = Path(__file__).resolve().parents[2]
PIN = "storage_state_start_binding_cyclic_period"
BARE = "storage_state_start_binding"
WIND_BATTERY = ["init", "west", "wind", "battery"]
FULLYEAR_ROLL = ["init", "west", "battery", "wind", "fullYear", "dispatch_fullYear_roll"]
NESTED = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest",
          "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
          "storage_fullYear_6h", "invest_24h", "bind_battery_forward", "unlimited_investment"]


def _f(x: float) -> list:
    return b64(x, "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _bare(pb, node: str) -> dict:
    """Rows of the exact family storage_state_start_binding (row_terms is a prefix match)."""
    return {k: v for k, v in row_terms(pb, BARE, f"{node},").items() if k.startswith(f"{BARE}[")}


def _lag_of_first(fd, node: str, col: str = "t_previous_within_timeset") -> set[tuple[str, str]]:
    first = utf8(fd.nodeState_first_dt).filter(pl.col("n") == node).select("d", "t")
    lag = utf8(fd.dtttdt).join(first, on=["d", "t"], how="inner")
    return {(d, t) for d, t in lag.select("d", col).iter_rows()}


def _state_mwh(step, node: str) -> dict[str, float]:
    su = float(utf8(step.flex_data.p_state_unitsize.frame).filter(pl.col("n") == node)["value"][0])
    rows = utf8(step.solution.value("v_state")).filter(pl.col("n") == node)
    return {t: v * su for t, v in rows.select("t", "value").iter_rows()}


# ---------------------------------------------------------------------------
# bind_within_timeblock + fix_start (N8)
# ---------------------------------------------------------------------------


def test_timeblock_fix_start_pins_the_state_the_first_step_starts_from(tmp_path_factory):
    """wind_battery (y2020_2day_dispatch: p2020 t0001–t0048, one timeblock), battery bind_within_timeblock +
    fix_start 0.1 (existing 50 MWh, so 5 MWh), no self-discharge, inverter capacity 0 (the battery has no flows),
    inflow +10 MWh at t0001 and −10 MWh at t0003.

    The first step's state change is S[t0001] − S[t0048] (t0048 = its wrap target in the timeblock), so the start
    state is S[t0048] = 5 and the balance gives S[t0001] = S[t0002] = 5 + 10 = 15, S[t0003..t0048] = 15 − 10 = 5:
    no slack.  Before, the pin sat on S[t0001] = 5, which needs S[t0048] = −5: the balance needed slack."""
    url, scen = make_db(tmp_path_factory, alt="sse_tb", base_chain=WIND_BATTERY, values=[
        ["node", "battery", "self_discharge_loss", _f(0.0)],
        ["node", "battery", "inflow", time_map({"t0001": 10.0, "t0003": -10.0}, 0.0)],
        ["connection", "battery_inverter", "existing", _f(0.0)]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    assert _lag_of_first(fd, "battery") == {("p2020", "t0048")}
    pb = build(fd)
    assert set(row_terms(pb, PIN, "battery,")) == {f"{PIN}[battery,p2020,t0048]"}
    assert not _bare(pb, "battery")
    lo, hi = row_bounds(pb, PIN, "battery,")[f"{PIN}[battery,p2020,t0048]"]
    su = float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == "battery")["value"][0])
    assert lo == pytest.approx(0.1 * 50.0 / su) and hi == pytest.approx(lo)
    state = _state_mwh(step, "battery")
    assert state["t0048"] == pytest.approx(5.0, abs=1e-7)
    assert state["t0001"] == pytest.approx(15.0, abs=1e-7)
    assert state["t0002"] == pytest.approx(15.0, abs=1e-7)
    assert state["t0003"] == pytest.approx(5.0, abs=1e-7)
    for slack in ("vq_state_up", "vq_state_down"):
        rows = utf8(step.solution.value(slack)).filter(pl.col("n") == "battery")
        assert rows["value"].abs().max() == pytest.approx(0.0, abs=1e-9), slack


def test_timeblock_fix_start_rolling_pins_the_end_of_the_rolls_timeblock_part(tmp_path_factory):
    """fullYear_roll with horizon 36 h / jump 24 h: the first roll holds t0001–t0036 of the fullYear timeblock, so its
    first step wraps to t0036 (the end of the timeblock's part in the roll) and fix_start is pinned there.  Later
    rolls are cyclic continuation rolls: no pin (and no carry, #119)."""
    url, scen = make_db(tmp_path_factory, alt="sse_tb_roll", base_chain=FULLYEAR_ROLL, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)]])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) > 2
    fd0 = steps[0].flex_data
    assert _lag_of_first(fd0, "battery") == {("p2020", "t0036")}
    pb0 = build(fd0)
    assert set(row_terms(pb0, PIN, "battery,")) == {f"{PIN}[battery,p2020,t0036]"}
    assert not _bare(pb0, "battery")
    for step in steps[1:3]:
        pb = build(step.flex_data)
        assert not row_terms(pb, PIN, "battery,") and not _bare(pb, "battery")


def _stochastic_db(tmp_path_factory, alt: str, values: list) -> tuple[str, str]:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db
    from tests.engine.test_cyclic_fix_start import _lp_snapshot

    data = json.loads((REPO / "tests" / "fixtures" / "stochastics.json").read_text())
    data = _lp_snapshot()._overlay(data, name=alt, base="2_day_stochastic_dispatch", values=values)
    work = tmp_path_factory.mktemp(alt)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    return url, alt


def test_timeblock_fix_start_stochastic_pins_in_the_trunk(tmp_path_factory):
    """2_day_stochastic_dispatch, hydro_reservoir bind_within_timeblock + fix_start 0.5: the timeblock lag stays in
    its period, so each first step of the solve (the trunk's and each forecast branch's) is pinned at the wrap
    target inside its own period, and the solve is optimal."""
    from flextool.engine import run_chain_from_db

    url, scen = _stochastic_db(tmp_path_factory, "sse_tb_stoch", [
        ["node", "hydro_reservoir", "storage_binding_method", "bind_within_timeblock"],
        ["node", "hydro_reservoir", "storage_start_end_method", "fix_start"],
        ["node", "hydro_reservoir", "storage_state_start", 0.5]])
    steps = list(run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("sse_tb_stoch_w"),
                                   keep_solutions=True).values())
    assert all(s.solution.optimal for s in steps)
    fd = steps[0].flex_data
    first = utf8(fd.nodeState_first_dt).filter(pl.col("n") == "hydro_reservoir").select("d", "t")
    lag = utf8(fd.dtttdt).join(first, on=["d", "t"], how="inner")
    assert lag.height == first.height > 1
    want = {(d, t) for d, t in lag.select("d", "t_previous_within_timeset").iter_rows()}
    assert {d for d, _t in want} == set(first["d"].to_list())
    got = set(row_terms(build(fd), PIN, "hydro_reservoir,"))
    assert got == {f"{PIN}[hydro_reservoir,{d},{t}]" for d, t in want}


def test_timeblock_fix_start_yields_to_a_nested_fix_at_the_wrap_target(tmp_path_factory):
    """Nested invest → storage → dispatch (36 h / 24 h) with nested fix_quantity and the battery
    bind_within_timeblock + fix_start.  The first dispatch roll holds t0001–t0036 of the fullYear timeblock: its wrap
    target is t0036, the step where node_balance_fix_quantity_eq_lower fixes the state to the storage solve's.  The
    upper value wins: the roll has the fix row and no start pin on that variable, and it solves."""
    url, scen = make_db(tmp_path_factory, alt="sse_tb_nested", base_chain=NESTED, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
        ["node", "battery", "storage_nested_fix_method", _s("fix_quantity")],
        ["node", "battery", "storage_binding_method", _s("bind_within_timeblock")]])
    steps = run(tmp_path_factory, url, scen)
    roll0 = steps["dispatch_fullYear_roll_roll_0"]
    assert roll0.solution.optimal
    fd = roll0.flex_data
    assert _lag_of_first(fd, "battery") == {("p2020", "t0036")}
    pb = build(fd)
    assert set(row_terms(pb, "node_balance_fix_quantity_eq_lower", "battery,")) == {
        "node_balance_fix_quantity_eq_lower[battery,p2020,t0036]"}
    assert not row_terms(pb, PIN, "battery,")
    assert not _bare(pb, "battery")
    # The storage solve itself (no nested fix above it) keeps its start pin at its own wrap target.
    fd_storage = steps["storage_fullYear_6h"].flex_data
    assert set(row_terms(build(fd_storage), PIN, "battery,")) == {
        f"{PIN}[battery,{d},{t}]" for d, t in _lag_of_first(fd_storage, "battery")}


# ---------------------------------------------------------------------------
# Later solves continue from the realised state; fix_start only where a level first appears (N5)
# ---------------------------------------------------------------------------

SEQ = ["init", "west", "wind", "wind_invest", "battery", "battery_invest", "5weeks", "lifetime_choice",
       "invest_solveSequence_5weeks", "lifetime_renew"]
MULTI_INVEST = ["init", "west", "coal", "coal_invest", "wind", "wind_invest", "battery", "battery_invest", "5weeks",
                "invest_solveSequence_5weeks", "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll",
                "storage_fullYear_6h", "invest_oneYear4Times_5weeks", "bind_battery_forward",
                "invest_twoYears4Times_5weeks", "unlimited_investment"]
NO_LOSS = ["node", "battery", "self_discharge_loss", b64(0.0, "float")]


def _su(fd, node: str = "battery") -> float:
    return float(utf8(fd.p_state_unitsize.frame).filter(pl.col("n") == node)["value"][0])


def _realised_end(step, node: str = "battery") -> float:
    """State (MWh) of ``node`` at the solve's last realised step, read from its own solution."""
    end = step.handoff.roll_end_step
    d, t = str(end["period"][0]), str(end["step"][0])
    v = utf8(step.solution.value("v_state")).filter((pl.col("n") == node) & (pl.col("d") == d) & (pl.col("t") == t))
    assert v.height == 1, (d, t)
    return float(v["value"][0]) * _su(step.flex_data, node)


def _first_row(step, node: str = "battery") -> tuple[dict, tuple[float, float], tuple[str, str]]:
    """The node's first-step nodeBalance_eq row: (terms, bounds, (d, t))."""
    fd = step.flex_data
    first = utf8(fd.nodeState_first_dt).filter(pl.col("n") == node).sort("d", "t").select("d", "t").rows()
    assert len(first) == 1, first
    d, t = first[0]
    name = f"nodeBalance_eq[{node},{d},{t}]"
    pb = build(fd)
    return row_terms(pb, "nodeBalance_eq", f"{node},{d},{t}")[name], \
        row_bounds(pb, "nodeBalance_eq", f"{node},{d},{t}")[name], (d, t)


def _carry(fd, node: str = "battery") -> "float | None":
    if fd.p_roll_continue_state is None:
        return None
    hit = utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == node)
    return float(hit["value"][0]) if hit.height else None


def _assert_continues(step, prev, node: str = "battery") -> None:
    """``step`` starts from ``prev``'s realised end state: the carry is that state, the first-step row has −su on the
    first state and the bound −carry (no self-discharge, no inflow on the node), and no fix_start terms."""
    want = _realised_end(prev, node)
    assert _carry(step.flex_data, node) == pytest.approx(want, abs=1e-9)
    terms, (lo, hi), (d, t) = _first_row(step, node)
    su = _su(step.flex_data, node)
    assert terms[f"v_state[{node},{d},{t}]"] == pytest.approx(-su)
    assert not [c for c in terms if c.startswith(("v_invest_n", "v_divest_n"))], terms
    assert lo == pytest.approx(-want, abs=1e-9) and hi == pytest.approx(lo)


def test_sequence_forward_carries_realized_state(tmp_path_factory):
    """Manual 4-solve sequence (y2020 → y2035, one realised period each), battery bind_forward_only + fix_start 0.1,
    horizon free, no self-discharge.  Each later solve starts from the state its predecessor realised at its last
    realised step (fix_start is for the first step of the whole model only, reference.md).  Before, each solve was
    re-pinned to 0.1·(50 + invested) = 85 MWh: the first-step row carried the invest column and bound −5."""
    url, scen = make_db(tmp_path_factory, alt="sse_seq_fo", base_chain=SEQ, values=[
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")],
        ["node", "battery", "storage_solve_horizon_method", _s("free")], NO_LOSS])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert [s.solve_name for s in steps] == ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]
    first = steps[0].flex_data
    assert _carry(first) is None
    terms, (lo, _hi), _ = _first_row(steps[0])
    assert [c for c in terms if c.startswith("v_invest_n")], "y2020 starts from fix_start (with investment)"
    assert lo == pytest.approx(-0.1 * 50.0)
    for prev, step in zip(steps, steps[1:]):
        _assert_continues(step, prev)
        assert not _bare(build(step.flex_data), "battery")


def test_sequence_cyclic_no_repin(tmp_path_factory):
    """The golden sequence (battery bind_within_timeblock + fix_start 0.1): y2020 has the start pin at its wrap
    target; y2025–y2035 have a realised state before them, so no pin, and no carry either (a cyclic node cycles
    within its solve, #119).  Before, every solve was re-pinned."""
    url, scen = make_db(tmp_path_factory, alt="sse_seq_tb", base_chain=SEQ, values=[])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert set(row_terms(build(steps[0].flex_data), PIN, "battery,")) == {f"{PIN}[battery,p2020,t0004]"}
    for step in steps[1:]:
        fd = step.flex_data
        assert _carry(fd) is not None
        pb = build(fd)
        assert not row_terms(pb, PIN, "battery,") and not _bare(pb, "battery")
        terms, (lo, hi), _ = _first_row(step)
        assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0, abs=1e-12)


def _nested_rolling_parent(tmp_path_factory, alt: str) -> dict:
    """Rolling storage solve (48 h / 24 h, 6-h steps) containing a rolling dispatch (12 h / 6 h); battery
    bind_forward_only + fix_start 0.1, use_reference_value 0.4, no nested fix, no self-discharge."""
    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=NESTED, values=[
        ["solve", "storage_fullYear_6h", "solve_mode", _s("rolling_window")],
        ["solve", "storage_fullYear_6h", "rolling_solve_horizon", _f(48.0)],
        ["solve", "storage_fullYear_6h", "rolling_solve_jump", _f(24.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(12.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(6.0)],
        ["node", "battery", "storage_solve_horizon_method", _s("use_reference_value")],
        ["node", "battery", "storage_nested_fix_method", _s("fix_nothing")], NO_LOSS])
    return run(tmp_path_factory, url, scen)


def test_nested_child_under_later_parent_roll_carries(tmp_path_factory):
    """The first dispatch roll under storage roll 1 (dispatch roll 4, p2020 t0025) continues from the state the
    dispatch realised last (roll 3, ending at t0024), as the other dispatch rolls do.  Before it had neither the carry
    nor fix_start: its first state was free (free energy up to the capacity)."""
    steps = _nested_rolling_parent(tmp_path_factory, "sse_nest_roll")
    names = list(steps)
    s1 = names.index("storage_fullYear_6h_roll_1")
    child, prev = steps[names[s1 + 1]], steps[names[s1 - 1]]
    assert names[s1 + 1] == "dispatch_fullYear_roll_roll_4" and names[s1 - 1] == "dispatch_fullYear_roll_roll_3"
    assert _first_row(child)[2] == ("p2020", "t0025")
    _assert_continues(child, prev)
    assert not _bare(build(child.flex_data), "battery")


def test_nested_first_roll_keeps_fix_start(tmp_path_factory):
    """Same chain: the invest solve, storage roll 0 and dispatch roll 0 are the first of their levels and start from
    fix_start (0.1 of the built capacity, bound −0.1·existing in the rolls); dispatch rolls 1–3 continue from the
    previous roll; storage roll 1 continues from the dispatch realised below it (roll 3: the upward feedback)."""
    steps = _nested_rolling_parent(tmp_path_factory, "sse_nest_first")
    for name in ("storage_fullYear_6h_roll_0", "dispatch_fullYear_roll_roll_0"):
        fd = steps[name].flex_data
        assert _carry(fd) is None, name
        existing = float(utf8(fd.p_state_existing_capacity.frame)
                         .filter((pl.col("n") == "battery") & (pl.col("d") == "p2020"))["value"][0])
        _terms, (lo, hi), _ = _first_row(steps[name])
        assert lo == pytest.approx(-0.1 * existing) and hi == pytest.approx(lo), name
    assert _carry(steps["invest_24h"].flex_data) is None
    for k in (1, 2, 3):
        _assert_continues(steps[f"dispatch_fullYear_roll_roll_{k}"], steps[f"dispatch_fullYear_roll_roll_{k - 1}"])
    _assert_continues(steps["storage_fullYear_6h_roll_1"], steps["dispatch_fullYear_roll_roll_3"])


def test_nested_multi_instance_levels_continue(tmp_path_factory):
    """Invest solve split per period (invest_5weeks_p20xx ⊃ storage_fullYear_6h_p20xx ⊃ dispatch 36 h / 24 h).  Only
    the p2020 instances are the first of their levels; invest_p2025, storage_p2025 and the first p2025 dispatch roll
    continue from the state the last p2020 dispatch roll realised (owner rule: fix_start at the first step of a lower
    nested level, i.e. its first appearance).  Before, each period's instances were re-pinned to 0.1·capacity."""
    url, scen = make_db(tmp_path_factory, alt="sse_multi", base_chain=MULTI_INVEST, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)], NO_LOSS])
    steps = run(tmp_path_factory, url, scen)
    names = list(steps)
    inv25 = names.index("invest_5weeks_p2025")
    last_2020 = steps[names[inv25 - 1]]
    assert names[inv25 - 1].startswith("dispatch_fullYear_roll_roll_")
    assert _realised_end_period(last_2020) == "p2020"
    for name in ("invest_5weeks_p2020", "storage_fullYear_6h_p2020", "dispatch_fullYear_roll_roll_0"):
        assert _carry(steps[name].flex_data) is None, name
    for name in ("invest_5weeks_p2025", "storage_fullYear_6h_p2025", names[inv25 + 2]):
        assert _carry(steps[name].flex_data) == pytest.approx(_realised_end(last_2020), abs=1e-9), name
    assert names[inv25 + 1] == "storage_fullYear_6h_p2025"
    _assert_continues(steps[names[inv25 + 2]], last_2020)


def _realised_end_period(step) -> str:
    return str(step.handoff.roll_end_step["period"][0])


def test_two_chains_in_one_process_start_from_fix_start(tmp_path_factory):
    """The carry registry is scoped to one cascade run: a second run in the same process starts from fix_start
    again (its first solve's first-step row equals the first run's), not from the first run's last state."""
    url, scen = make_db(tmp_path_factory, alt="sse_two", base_chain=SEQ, values=[
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")],
        ["node", "battery", "storage_solve_horizon_method", _s("free")], NO_LOSS])
    a = list(run(tmp_path_factory, url, scen).values())
    b = list(run(tmp_path_factory, url, scen).values())
    assert _realised_end(a[-1]) != pytest.approx(0.1 * 50.0)
    assert _carry(b[0].flex_data) is None
    assert _first_row(b[0]) == _first_row(a[0])


def test_realized_dispatch_siblings_continue_from_the_previous_dispatch(tmp_path_factory):
    """Rolling separate_dispatch (stochastic UC rolls 48 h / 24 h, each followed by its realised-dispatch sibling over
    the jump), hydro_reservoir made a bind_forward_only storage with fix_start 0.5.  The first UC roll and its
    sibling start from fix_start; every later UC roll and its sibling continue from the state the previous sibling
    realised (the realised dispatch; the UC rolls realise nothing themselves).  A sibling never starts from its own
    UC roll's state."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db
    from tests.engine.test_cyclic_fix_start import _lp_snapshot

    data = json.loads((REPO / "tests" / "fixtures" / "rolling_separate_dispatch.json").read_text())
    data = _lp_snapshot()._overlay(data, name="sse_rsd", base="rolling_separate_dispatch", values=[
        ["node", "hydro_reservoir", "node_type", "storage"],
        ["node", "hydro_reservoir", "storage_binding_method", "bind_forward_only"],
        ["node", "hydro_reservoir", "storage_solve_horizon_method", "free"]])
    work = tmp_path_factory.mktemp("sse_rsd")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, "sse_rsd", work_folder=work / "w", keep_solutions=True)
    names = list(steps)
    sib = [n for n in names if "realized_dispatch" in n]
    uc = [n for n in names if n not in sib]
    assert len(sib) == len(uc) > 2, names
    node = "hydro_reservoir"
    for name in (uc[0], sib[0]):
        fd = steps[name].flex_data
        assert _carry(fd, node) is None, name
        assert utf8(fd.storage_fix_start)["n"].to_list() == [node]
    for k in range(1, len(uc)):
        want = _realised_end(steps[sib[k - 1]], node)
        for name in (uc[k], sib[k]):
            assert _carry(steps[name].flex_data, node) == pytest.approx(want, abs=1e-6), (name, k)


def _rp_sequence(tmp_path_factory, alt: str, second_rp: bool) -> list:
    """y2020_5week → y2025_5week, battery bind_forward_only_blended_weights + fix_start 0.1, horizon free, no loss.
    y2020 runs on a representative-period timeset (blocks t0001 and t0009, 4 h each, both representing base period
    t0001); y2025 on the same (``second_rp``) or on the chronological 5weeks timeset (the battery is then downgraded
    to bind_forward_only)."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db
    from tests.engine.test_cyclic_fix_start import _lp_snapshot

    snap = _lp_snapshot()
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [["t0001", {"type": "map", "index_type": "str", "index_name": "representative_period", "rank": 1,
                               "data": [["t0001", 0.5], ["t0009", 0.5]]}]]}
    second = "rpw" if second_rp else "5weeks"
    data = json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text())
    data = snap._overlay(data, name=alt, base=["init", "west", "wind", "battery", "5weeks",
                                               "invest_solveSequence_5weeks"],
                         entities=[["timeset", "rpw"]], values=[
        ["timeset", "rpw", "timeline", "y2020"],
        ["timeset", "rpw", "timeset_duration", snap._pmap({"t0001": 4.0, "t0009": 4.0}, "time")],
        ["timeset", "rpw", "representative_period_weights", rpw],
        ["model", "flexTool", "solves", {"type": "array", "value_type": "str", "index_name": "sequence_index",
                                         "data": ["y2020_5week", "y2025_5week"]}],
        ["solve", "y2020_5week", "period_timeset", snap._pmap({"p2020": "rpw", "p2025": "rpw"})],
        ["solve", "y2025_5week", "period_timeset", snap._pmap({"p2025": second, "p2030": second})],
        ["node", "battery", "storage_binding_method", "bind_forward_only_blended_weights"],
        ["node", "battery", "storage_solve_horizon_method", "free"],
        ["node", "battery", "self_discharge_loss", 0.0]])
    work = tmp_path_factory.mktemp(alt)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = list(run_chain_from_db(url, alt, work_folder=work / "w", keep_solutions=True).values())
    assert [s.solve_name for s in steps] == ["y2020_5week", "y2025_5week"]
    assert "battery" in utf8(steps[0].flex_data.nodeState_rp)["n"].to_list()
    return steps


def test_representative_period_node_takes_neither_carry_nor_pin(tmp_path_factory):
    """rp → rp: in y2025 the battery is a representative-period node again.  Its state at y2020's last realised step
    is a state inside a representative block, not a chronological level, so it is not carried; and since a state
    was realised before y2025, fix_start does not re-pin it either: each block starts from v_state_rp_start, free.
    Before, y2025 was re-pinned (storage_state_start_binding at its first step)."""
    steps = _rp_sequence(tmp_path_factory, "sse_rp_rp", second_rp=True)
    fd = steps[1].flex_data
    assert "battery" in utf8(fd.nodeState_rp)["n"].to_list()
    assert _carry(fd) is not None
    pb = build(fd)
    assert _bare(build(steps[0].flex_data), "battery"), "the first rp solve keeps its fix_start pin"
    assert not _bare(pb, "battery") and not row_terms(pb, PIN, "battery,")
    rows, bounds = row_terms(pb, "nodeBalance_eq", "battery,"), row_bounds(pb, "nodeBalance_eq", "battery,")
    firsts = utf8(fd.rp_block_first).select("d", "t").unique().rows()
    assert firsts
    for d, t in firsts:
        name = f"nodeBalance_eq[battery,{d},{t}]"
        assert f"v_state_rp_start[battery,{d},{t}]" in rows[name]
        assert bounds[name] == (pytest.approx(0.0, abs=1e-12), pytest.approx(0.0, abs=1e-12)), name


def test_representative_period_node_downgraded_later_continues(tmp_path_factory):
    """rp → chronological: in y2025 the battery is downgraded to bind_forward_only and continues from the state y2020
    realised at its last realised step (the end of its last representative block, the best chronological level
    available).  Before, y2025 was re-pinned to fix_start."""
    steps = _rp_sequence(tmp_path_factory, "sse_rp_chrono", second_rp=False)
    fd = steps[1].flex_data
    assert fd.nodeState_rp is None or "battery" not in utf8(fd.nodeState_rp)["n"].to_list()
    assert "battery" in utf8(fd.storage_bind_forward_only)["n"].to_list()
    _assert_continues(steps[1], steps[0])


def test_self_discharge_output_of_a_continuing_solve(tmp_path_factory):
    """Forward sequence with self_discharge_loss 0.05: the written loss of y2025's first step is α·S·su + β·carry,
    the carry being y2020's realised end (the writer decides the start of the chain as the model does: the step
    starts from the carry, not from fix_start)."""
    from flextool.engine import run_chain_from_db
    from tests.engine.test_self_discharge import _ab, _reported

    url, scen = make_db(tmp_path_factory, alt="sse_sd", base_chain=SEQ, values=[
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")],
        ["node", "battery", "storage_solve_horizon_method", _s("free")],
        ["node", "battery", "self_discharge_loss", _f(0.05)]])
    wf = tmp_path_factory.mktemp("sse_sd_wf")
    steps = list(run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True).values())
    prev, step = steps[0], steps[1]
    carry = _realised_end(prev)
    assert carry > 1.0
    assert _carry(step.flex_data) == pytest.approx(carry, abs=1e-9)
    fd = step.flex_data
    d, t = utf8(fd.nodeState_first_dt).filter(pl.col("n") == "battery").row(0)[1:]
    dur = float(utf8(fd.p_step_duration.frame).filter((pl.col("d") == d) & (pl.col("t") == t))["value"][0])
    a, b = _ab(dur, 0.05)
    s = utf8(step.solution.value("v_state")).filter((pl.col("n") == "battery") & (pl.col("d") == d)
                                                     & (pl.col("t") == t))["value"][0] * _su(fd)
    assert _reported(wf, step.solve_name)[(d, t)] == pytest.approx(a * s + b * carry, rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------
# The other roll-forward carriers of a nested dispatch roll (N14)
# ---------------------------------------------------------------------------


def _rows(frame, cols: list[str]) -> list[tuple]:
    if frame is None:
        return []
    f = frame.frame if hasattr(frame, "frame") else frame
    return sorted(utf8(f).select(cols).rows())


def _close(a: list[tuple], b: list[tuple]) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x[:-1] != y[:-1] or abs(float(x[-1]) - float(y[-1])) > 1e-9:
            return False
    return True


@pytest.mark.parametrize("parent", ["single", "rolling"])
def test_nested_dispatch_rolls_continue_commitment_ramps_and_totals(tmp_path_factory, parent):
    """Nested invest → storage (single, or rolling 48 h / 24 h) → dispatch with coal unit commitment (linear,
    min_load 0.4, min_uptime 8 h, ramp 0.002).  Every dispatch roll after the first continues from the dispatch roll
    realised before it — also the first roll under a later storage roll: its committed online state, the ramp from
    that roll's last flow (when the steps are adjacent) and the min up time history.  Before, the parent solve's
    handoff (a storage solve realises no dispatch) was translated over these carriers in every nested dispatch roll,
    so each roll ≥ 1 restarted the commitment from its own cyclic wrap, ramped from nothing and forgot the min up time
    history."""
    vals = [["unit", "coal_plant", "startup_method", _s("linear")],
            ["unit", "coal_plant", "min_load", _f(0.4)],
            ["unit", "coal_plant", "min_uptime", _f(8.0)],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_method", _s("ramp_limit")],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_up", _f(0.002)],
            ["unit__outputNode", ["coal_plant", "west"], "ramp_speed_down", _f(0.002)],
            ["node", "battery", "storage_nested_fix_method", _s("fix_nothing")]]
    if parent == "single":
        vals += [["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
                 ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)]]
    else:
        vals += [["solve", "storage_fullYear_6h", "solve_mode", _s("rolling_window")],
                 ["solve", "storage_fullYear_6h", "rolling_solve_horizon", _f(48.0)],
                 ["solve", "storage_fullYear_6h", "rolling_solve_jump", _f(24.0)],
                 ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(12.0)],
                 ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(6.0)]]
    url, scen = make_db(tmp_path_factory, alt=f"sse_n14_{parent}", base_chain=NESTED, values=vals)
    steps = run(tmp_path_factory, url, scen)
    rolls = [k for k in steps if k.startswith("dispatch_fullYear_roll_roll_")][:8]
    assert len(rolls) == 8
    ramp_checked = online_checked = 0
    for prev_name, name in zip(rolls, rolls[1:]):
        prev, fd = steps[prev_name].handoff, steps[name].flex_data
        assert _close(_rows(fd.p_roll_continue_online, ["p", "value"]),
                      _rows(prev.roll_end_online, ["process", "value"])), name
        online_checked += 1
        assert _rows(fd.online_startup_history, ["p", "d", "t"]) == _rows(prev.online_history,
                                                                          ["process", "period", "step"]), name
        end = prev.roll_end_step.rows()[0][:2]
        first = _first_row(steps[name])[2]
        if end[0] == first[0] and int(first[1][1:]) == int(end[1][1:]) + 1:
            # Adjacent in the same period: the first step ramps from the previous roll's last realised flow.
            assert _rows(fd.dt_ramp_carry, ["d", "t", "d_previous", "t_previous"])[:1] == [(first[0], first[1], *end)]
            flow = [r[-1] for r in _rows(prev.roll_end_flow, ["process", "source", "sink", "value"])]
            assert [r[-1] for r in _rows(fd.p_roll_continue_flow, ["p", "source", "sink", "value"])] == \
                pytest.approx(flow, abs=1e-12)
            ramp_checked += 1
    assert online_checked == 7 and ramp_checked >= 4
    assert any(steps[r].handoff.online_history is not None for r in rolls)
    if parent == "rolling":
        # The first roll under storage roll 1 and storage roll 1 itself (upward) continue from dispatch roll 3.
        names = list(steps)
        s1 = names.index("storage_fullYear_6h_roll_1")
        for name in (names[s1], names[s1 + 1]):
            assert _close(_rows(steps[name].flex_data.p_roll_continue_online, ["p", "value"]),
                          _rows(steps[names[s1 - 1]].handoff.roll_end_online, ["process", "value"])), name


# ---------------------------------------------------------------------------
# fix_end / fix_start_end pin the end of the last sub-solve of each level (N6); the reference value yields to a
# nested fix (N11)
# ---------------------------------------------------------------------------

END = "storage_state_end_binding"
REF = "storage_state_solve_horizon_reference_value"
FIX_Q = "node_balance_fix_quantity_eq_lower"
FIX_U = "node_storage_usage_fix_le"


def _last_steps(fd, node: str = "battery") -> set[tuple[str, str]]:
    """(d, t) of nodeState_last_dt ∩ period_last for ``node``."""
    last = utf8(fd.nodeState_last_dt).filter(pl.col("n") == node)
    periods = set(utf8(fd.period_last)["d"].to_list())
    return {(d, t) for d, t in last.select("d", "t").iter_rows() if d in periods}


def _existing(fd, node: str, d: str) -> float:
    return float(utf8(fd.p_state_existing_capacity.frame)
                 .filter((pl.col("n") == node) & (pl.col("d") == d))["value"][0])


def test_fix_end_pins_the_last_step(tmp_path_factory):
    """wind_battery (y2020_2day_dispatch, p2020 t0001–t0048), battery bind_forward_only + fix_end with
    storage_state_end 0.3 and existing 50 MWh: the state at the end of t0048 is pinned to 0.3·50 = 15 MWh (bounds
    15/su in state units), with no start pin (fix_end fixes only the end).  Before, no row pinned the end and the
    state ended at 9.999 MWh."""
    url, scen = make_db(tmp_path_factory, alt="sse_end", base_chain=WIND_BATTERY, values=[
        ["node", "battery", "storage_start_end_method", _s("fix_end")],
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")],
        ["node", "battery", "storage_state_end", _f(0.3)]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    assert _last_steps(fd) == {("p2020", "t0048")}
    pb = build(fd)
    rows = row_bounds(pb, END, "battery,")
    assert set(rows) == {f"{END}[battery,p2020,t0048]"}
    su = _su(fd)
    assert rows[f"{END}[battery,p2020,t0048]"] == (pytest.approx(0.3 * 50.0 / su), pytest.approx(0.3 * 50.0 / su))
    assert not _bare(pb, "battery") and not row_terms(pb, PIN, "battery,")
    assert _state_mwh(step, "battery")["t0048"] == pytest.approx(15.0, abs=1e-6)


def test_fix_end_only_in_the_last_roll(tmp_path_factory):
    """fullYear_roll (p2020–p2035, 72 h each) rolling 36 h / 24 h, battery bind_forward_only + fix_end 0.3: only the
    last roll ends the model, so only it has the end row (at the end of p2035) and its realised end is 0.3 of the
    capacity; the earlier rolls end free (reference.md: the last timestep of the last roll)."""
    url, scen = make_db(tmp_path_factory, alt="sse_end_roll", base_chain=FULLYEAR_ROLL, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
        ["node", "battery", "storage_start_end_method", _s("fix_end")],
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")],
        ["node", "battery", "storage_state_end", _f(0.3)]])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert len(steps) > 3
    for step in steps[:-1]:
        assert step.flex_data.storage_solve_last is False, step.solve_name
        assert not row_terms(build(step.flex_data), END, "battery,"), step.solve_name
    last = steps[-1]
    fd = last.flex_data
    assert fd.storage_solve_last is True
    assert {r.split("[", 1)[1].rstrip("]") for r in row_terms(build(fd), END, "battery,")} == {
        f"battery,{d},{t}" for d, t in _last_steps(fd)}
    assert {d for d, _t in _last_steps(fd)} == {"p2035"}
    assert _realised_end(last) == pytest.approx(0.3 * _existing(fd, "battery", "p2035"), abs=1e-6)


def test_fix_end_with_investment(tmp_path_factory):
    """wind_battery_invest (5weeks, battery invest_total ≤ 800 MWh), battery bind_forward_only + fix_end 0.5: the end
    row carries the built capacity, −0.5 on each v_invest_n column in service, so the realised end state is half of
    the existing plus the invested storage."""
    url, scen = make_db(tmp_path_factory, alt="sse_end_inv",
                        base_chain=["init", "west", "wind", "battery", "battery_invest", "5weeks"], values=[
        ["node", "battery", "storage_start_end_method", _s("fix_end")],
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")],
        ["node", "battery", "storage_solve_horizon_method", _s("free")],
        ["node", "battery", "storage_state_end", _f(0.5)]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    fd = step.flex_data
    pb = build(fd)
    rows = row_terms(pb, END, "battery,")
    assert rows and set(rows) == {f"{END}[battery,{d},{t}]" for d, t in _last_steps(fd)}
    su = _su(fd)
    inv = utf8(step.solution.value("v_invest_n")).filter(pl.col("n") == "battery")
    invested = {d: v for d, v in inv.select("d", "value").iter_rows()}
    assert sum(invested.values()) > 1e-6, "the battery is built"
    for name, terms in rows.items():
        d, t = name[:-1].split(",")[-2:]
        cols = {c: v for c, v in terms.items() if c.startswith("v_invest_n[battery,")}
        assert cols and all(v == pytest.approx(-0.5) for v in cols.values()), terms
        built = sum(invested[c[:-1].split(",")[-1]] for c in cols)
        state = _state_mwh(step, "battery")[t]
        assert state == pytest.approx(0.5 * (_existing(fd, "battery", d) + built * su), rel=1e-7), (d, t)


@pytest.mark.parametrize("method", ["fix_quantity", "fix_usage", "fix_price"])
def test_fix_end_nested(tmp_path_factory, method):
    """Nested invest → storage → dispatch (36 h / 24 h), battery bind_forward_only + fix_start_end (end 0.3) and the
    given nested fix method.  The invest solve and the storage solve are the last of their levels and have the end
    row.  The last dispatch roll ends the dispatch level: with fix_quantity / fix_usage the upper solve fixes its end
    step and wins (the fix row, no end row); a fix_price only values that end, so the end row stays.  Earlier dispatch
    rolls have no end row."""
    url, scen = make_db(tmp_path_factory, alt=f"sse_end_nest_{method}", base_chain=NESTED, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
        ["node", "battery", "storage_start_end_method", _s("fix_start_end")],
        ["node", "battery", "storage_state_end", _f(0.3)],
        ["node", "battery", "storage_nested_fix_method", _s(method)]])
    steps = run(tmp_path_factory, url, scen)
    rolls = [k for k in steps if k.startswith("dispatch_fullYear_roll_roll_")]
    assert len(rolls) > 3
    for name in ("invest_24h", "storage_fullYear_6h"):
        fd = steps[name].flex_data
        assert fd.storage_solve_last is True, name
        assert set(row_terms(build(fd), END, "battery,")) == {
            f"{END}[battery,{d},{t}]" for d, t in _last_steps(fd)}, name
    for name in rolls[:-1]:
        assert steps[name].flex_data.storage_solve_last is False, name
        assert not row_terms(build(steps[name].flex_data), END, "battery,"), name
    last = steps[rolls[-1]]
    fd = last.flex_data
    assert fd.storage_solve_last is True
    pb = build(fd)
    ends = {f"battery,{d},{t}" for d, t in _last_steps(fd)}
    if method == "fix_price":
        assert {r.split("[", 1)[1][:-1] for r in row_terms(pb, END, "battery,")} == ends
        assert _realised_end(last) == pytest.approx(0.3 * _existing(fd, "battery", "p2035"), abs=1e-6)
    else:
        fam = FIX_Q if method == "fix_quantity" else FIX_U
        assert {r.split("[", 1)[1][:-1] for r in row_terms(pb, fam, "battery,")} == ends
        assert not row_terms(pb, END, "battery,")
    assert all(s.solution.optimal for s in steps.values())


def test_fix_end_stochastic(tmp_path_factory):
    """2_day_stochastic_dispatch, hydro_reservoir bind_forward_only + fix_end 0.4: each terminal branch period ends
    at the target, one row per terminal period, and each branch's end state is 0.4 of the existing 5000."""
    from flextool.engine import run_chain_from_db

    url, scen = _stochastic_db(tmp_path_factory, "sse_end_stoch", [
        ["node", "hydro_reservoir", "storage_binding_method", "bind_forward_only"],
        ["node", "hydro_reservoir", "storage_start_end_method", "fix_end"],
        ["node", "hydro_reservoir", "storage_state_end", 0.4]])
    steps = list(run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp("sse_end_stoch_w"),
                                   keep_solutions=True).values())
    assert all(s.solution.optimal for s in steps)
    step = steps[-1]
    fd = step.flex_data
    node = "hydro_reservoir"
    ends = _last_steps(fd, node)
    assert len(ends) > 1 and len({d for d, _t in ends}) == len(ends)
    assert set(row_terms(build(fd), END, f"{node},")) == {f"{END}[{node},{d},{t}]" for d, t in ends}
    su = _su(fd, node)
    v = utf8(step.solution.value("v_state")).filter(pl.col("n") == node)
    for d, t in ends:
        got = float(v.filter((pl.col("d") == d) & (pl.col("t") == t))["value"][0]) * su
        assert got == pytest.approx(0.4 * _existing(fd, node, d), rel=1e-7), (d, t)


@pytest.mark.parametrize("method", ["fix_nothing", "fix_quantity", "fix_usage", "fix_price"])
def test_reference_value_yields_to_a_nested_fix(tmp_path_factory, method):
    """Nested invest → storage → dispatch (36 h / 24 h), battery bind_forward_only + use_reference_value 0.4 and the
    given nested fix method.  The storage solve fixes (or prices) every dispatch roll's end step, so with a nested fix
    no dispatch roll has the reference row (how_to.md: the fixed storage value overrides the horizon end value); the
    fix sits at each roll end instead.  Without a nested fix (control) every roll end has the reference row.  The
    invest and storage solves have no upper fix and keep their reference rows; every solve is optimal.  Before, roll 0
    had both the fix row and the reference row on v_state[t0036] (20 vs the storage state) and was infeasible."""
    url, scen = make_db(tmp_path_factory, alt=f"sse_ref_{method}", base_chain=NESTED, values=[
        ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", _f(36.0)],
        ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", _f(24.0)],
        ["node", "battery", "storage_solve_horizon_method", _s("use_reference_value")],
        ["node", "battery", "storage_nested_fix_method", _s(method)]])
    steps = run(tmp_path_factory, url, scen)
    assert all(s.solution.optimal for s in steps.values())
    rolls = [k for k in steps if k.startswith("dispatch_fullYear_roll_roll_")]
    assert len(rolls) > 3
    fixed_rolls = 0
    for name in rolls:
        fd = steps[name].flex_data
        pb = build(fd)
        ends = {f"battery,{d},{t}" for d, t in _last_steps(fd)}
        assert ends, name
        refs = {r.split("[", 1)[1][:-1] for r in row_terms(pb, REF, "battery,")}
        if method == "fix_nothing":
            assert refs == ends, name
            continue
        if method == "fix_price":
            assert not refs, name
            assert fd.ndt_fix_storage_price is not None, name
            fixed_rolls += 1
            continue
        # Each roll end has either the upper fix or (where the upper solve sets none there) the reference row, never
        # both.  A fix_usage window whose upper usage is 0 has no cap row (the handoff drops zero usages, item 150),
        # so that roll end keeps the reference row.
        fam = FIX_Q if method == "fix_quantity" else FIX_U
        fixed = {r.split("[", 1)[1][:-1] for r in row_terms(pb, fam, "battery,")}
        assert not (refs & fixed) and refs | fixed == ends, (name, refs, fixed)
        fixed_rolls += bool(fixed)
    if method != "fix_nothing":
        assert fixed_rolls >= len(rolls) // 2, (fixed_rolls, len(rolls))
    for name in ("invest_24h", "storage_fullYear_6h"):
        fd = steps[name].flex_data
        assert {r.split("[", 1)[1][:-1] for r in row_terms(build(fd), REF, "battery,")} == {
            f"battery,{d},{t}" for d, t in _last_steps(fd)}, name


def test_storage_solve_last_per_level():
    """The last sub-solve of each nesting level: a later sub-solve at the same level clears the flag; a realised-
    dispatch sibling is not counted and takes its parent's flag."""
    from flextool.engine.solve._run_model import _storage_solve_last

    solves = ["inv", "sto_0", "d_0", "d_1", "sto_1", "d_2", "d_3", "uc_0", "rd_0", "uc_1", "rd_1"]
    level = {"inv": 0, "sto_0": 1, "sto_1": 1, "d_0": 2, "d_1": 2, "d_2": 2, "d_3": 2,
             "uc_0": 0, "rd_0": 0, "uc_1": 0, "rd_1": 0}
    siblings = {"rd_0": "uc_0", "rd_1": "uc_1"}
    got = dict(zip(solves, _storage_solve_last(solves, level.__getitem__, siblings)))
    assert got == {"inv": False, "sto_0": False, "d_0": False, "d_1": False, "sto_1": True, "d_2": False,
                   "d_3": True, "uc_0": False, "rd_0": False, "uc_1": True, "rd_1": True}


# ---------------------------------------------------------------------------
# fix_start_end ignores the storage_binding_method (N12)
# ---------------------------------------------------------------------------

FIVE_WEEKS = ["init", "west", "wind", "battery", "5weeks"]


def _n12_state(method: str, d: str, t: str) -> float:
    """The forward chain of the N12 fixture (MWh): 10 at the start, 20 from t0003 of p2020, 30 from t0003 of p2025.
    A block node's state is the state at the end of its block (t0001–t0004, t0009–t0016, t0025–t0028,
    t0037–t0040), so 20 in every p2020 block and 30 in every p2025 block."""
    if d == "p2020":
        return 20.0 if method == "bind_intraperiod_blocks" or t >= "t0003" else 10.0
    return 30.0 if method == "bind_intraperiod_blocks" or t >= "t0003" else 20.0


@pytest.mark.parametrize("method", ["bind_within_solve", "bind_within_timeblock", "bind_within_period",
                                    "bind_within_period_blended_weights", "bind_intraperiod_blocks"])
def test_fix_start_end_cyclic_chains_forward(tmp_path_factory, method):
    """y2020_5week (p2020, p2025; steps t0001–t0004, t0009–t0016, t0025–t0028, t0037–t0040 in each), battery
    fix_start_end with storage_state_start 0.25 and storage_state_end 0.5, existing 40 MWh in p2020 and 60 MWh in
    p2025, no self-discharge, inverter capacity 0 (no flows), inflow +10 MWh at t0003 of each period.  With
    fix_start_end the binding method is ignored (reference.md), so the chain runs forward through both periods: it
    starts from 0.25·40 = 10 MWh, holds 10 + 10 = 20 from t0003 of p2020, continues into p2025 at 20, holds
    20 + 10 = 30 from its t0003 and ends at 30 = 0.5·60 MWh, the end pin, with no slack (_n12_state).  Both periods
    end at t0040, but only the solve's last period is last (#37): p2020's end (20 MWh) is not pinned.  A
    bind_intraperiod_blocks node keeps its blocks.  The
    blended method has no representative periods here: blended → plain (downgrade) → forward-only.  Before, each
    cycle needed a net inflow of 0 (each period has +10: slack), and with bind_within_solve the start and end pins
    sat on one state."""
    from flextool.engine.model import _self_discharge

    url, scen = make_db(tmp_path_factory, alt=f"sse_n12_{method[5:]}", base_chain=FIVE_WEEKS, values=[
        ["node", "battery", "storage_binding_method", _s(method)],
        ["node", "battery", "storage_start_end_method", _s("fix_start_end")],
        ["node", "battery", "storage_state_start", _f(0.25)],
        ["node", "battery", "storage_state_end", _f(0.5)], NO_LOSS,
        ["node", "battery", "existing", b64({"index_type": "str", "index_name": "period",
                                             "data": [["p2020", 40.0], ["p2025", 60.0]]}, "map")],
        ["node", "battery", "inflow", time_map({"t0003": 10.0}, 0.0)],
        ["connection", "battery_inverter", "existing", _f(0.0)]])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    assert step.solution.optimal
    fd = step.flex_data
    existing = {d: _existing(fd, "battery", d) for d in ("p2020", "p2025")}
    assert existing == {"p2020": pytest.approx(40.0), "p2025": pytest.approx(60.0)}
    for cyclic in ("storage_bind_within_solve", "storage_bind_within_timeblock", "storage_bind_within_period",
                   "storage_bind_within_period_blended_weights"):
        frame = getattr(fd, cyclic)
        assert frame is None or "battery" not in utf8(frame)["n"].to_list(), cyclic
    if method == "bind_intraperiod_blocks":
        assert "battery" in utf8(fd.nodeStateBlock)["n"].to_list()
        assert utf8(_self_discharge.block_forward_nodes(fd))["n"].to_list() == ["battery"]
        assert _self_discharge.block_cyclic_nodes(fd) is None
    else:
        assert "battery" in utf8(fd.storage_bind_forward_only)["n"].to_list()
    pb = build(fd)
    assert not row_terms(pb, PIN, "battery,") and not _bare(pb, "battery")
    su = _su(fd)
    last = _last_steps(fd)
    assert last == {("p2025", "t0040")}
    ends = row_bounds(pb, END, "battery,")
    assert set(ends) == {f"{END}[battery,{d},{t}]" for d, t in last}
    for d, t in last:
        lo, hi = ends[f"{END}[battery,{d},{t}]"]
        assert lo == pytest.approx(0.5 * existing[d] / su) and hi == pytest.approx(lo)
    rows = utf8(step.solution.value("v_state")).filter(pl.col("n") == "battery")
    assert rows.height == 40
    for d, t, v in rows.select("d", "t", "value").iter_rows():
        assert v * su == pytest.approx(_n12_state(method, d, t), abs=1e-6), (d, t, v * su)
    for slack in ("vq_state_up", "vq_state_down"):
        hit = utf8(step.solution.value(slack)).filter(pl.col("n") == "battery")
        assert hit.height == 0 or hit["value"].abs().max() == pytest.approx(0.0, abs=1e-9), slack


@pytest.mark.parametrize("method", ["use_reference_value", "use_reference_price", "fix_end"])
def test_only_the_solves_last_period_ends_the_horizon(tmp_path_factory, method):
    """y2020_5week holds p2020 and p2025, both ending at t0040 (one shared timeset).  Only the solve's last period
    ends the horizon (#37, owner Q17): with battery bind_forward_only (existing 50 MWh, no self-discharge) the
    reference value 0.4 pins only the end of p2025 (0.4·50 = 20 MWh), the reference price 30 values only that state
    (a negative objective coefficient; none on the end of p2020), and fix_end 0.3 pins only that state (0.3·50 =
    15 MWh).  The end of p2020 continues into p2025 (forward chain) and is free.  Before, period_last held both
    periods, so the end of p2020 was pinned / valued too, mid-horizon."""
    values = {"use_reference_value": [], "use_reference_price": [
        ["node", "battery", "storage_solve_horizon_method", _s("use_reference_price")]], "fix_end": [
        ["node", "battery", "storage_solve_horizon_method", _s("free")],
        ["node", "battery", "storage_start_end_method", _s("fix_end")]]}[method]
    url, scen = make_db(tmp_path_factory, alt=f"sse_37_{method}", base_chain=FIVE_WEEKS, values=[
        ["node", "battery", "storage_binding_method", _s("bind_forward_only")], NO_LOSS] + values)
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    assert step.solution.optimal
    fd = step.flex_data
    assert {(d, t) for d, t in utf8(fd.nodeState_last_dt).filter(pl.col("n") == "battery")
            .select("d", "t").iter_rows()} == {("p2020", "t0040"), ("p2025", "t0040")}
    assert utf8(fd.period_last)["d"].to_list() == ["p2025"]
    pb = build(fd)
    family = {"use_reference_value": REF, "fix_end": END}.get(method)
    for fam in (REF, END):
        want = {f"{fam}[battery,p2025,t0040]"} if fam == family else set()
        assert set(row_terms(pb, fam, "battery,")) == want, fam
    if family is not None:
        share = 0.4 if method == "use_reference_value" else 0.3
        end = utf8(step.solution.value("v_state")).filter(
            (pl.col("n") == "battery") & (pl.col("d") == "p2025") & (pl.col("t") == "t0040"))
        assert float(end["value"][0]) * _su(fd) == pytest.approx(share * 50.0, abs=1e-6)
    else:
        m = pb.canonicalise()
        obj = {nm: float(m.col_obj[i]) for i, nm in enumerate(m.col_names) if nm.startswith("v_state[battery,")}
        assert obj["v_state[battery,p2025,t0040]"] < 0.0
        assert {nm for nm, c in obj.items() if c != 0.0} == {"v_state[battery,p2025,t0040]"}


def test_period_last_is_the_last_period_and_its_branches():
    """``_compute_first_and_last_periods``: periods sharing a timeset are not all last — only the solve's last period
    (#37); in a stochastic solve the last period's branch siblings that end on the same step are last with it (the
    anchor group of the last period, as listed in period__branch), an earlier anchor's group is not."""
    from types import SimpleNamespace

    from flextool.engine.preprocess.emit._solve_writers import _compute_first_and_last_periods

    def atl(spec: dict) -> dict:
        return {p: [SimpleNamespace(timestep=a), SimpleNamespace(timestep=b)] for p, (a, b) in spec.items()}

    shared = atl({"p2020": ("t0001", "t0040"), "p2025": ("t0001", "t0040"), "p2030": ("t0001", "t0040")})
    assert _compute_first_and_last_periods(shared, [("p2020", "ts")], [])[0] == ["p2030"]
    stoch = atl({"p2035": ("t0001", "t0003"), "p2035_high": ("t0001", "t0003"), "p2040": ("t0004", "t0006"),
                 "p2040_high": ("t0004", "t0006"), "p2040_low": ("t0004", "t0006")})
    branches = [("p2035", "p2035"), ("p2035", "p2035_high"), ("p2040", "p2040"), ("p2040", "p2040_high"),
                ("p2040", "p2040_low")]
    assert _compute_first_and_last_periods(stoch, [("p2035", "ts")], branches)[0] == [
        "p2040_low", "p2040", "p2040_high"]
    trunk = atl({"period1": ("t0001", "t0048"), "period1_upper": ("t0001", "t0048"),
                 "period1_lower": ("t0001", "t0048")})
    tb = [("period1", "period1"), ("period1", "period1_realized"), ("period1", "period1_upper"),
          ("period1", "period1_lower")]
    assert _compute_first_and_last_periods(trunk, [("period1", "ts")], tb)[0] == [
        "period1_lower", "period1", "period1_upper"]


def test_fix_start_end_blocks_continue_between_solves(tmp_path_factory):
    """The manual 4-solve sequence (y2020 → y2035, one period each), battery bind_intraperiod_blocks + fix_start_end
    (start 0.1, end 0.3), no self-discharge, no inflow on the battery.  The blocks chain forward, so the node is not
    cyclic: each later solve continues from the state its predecessor realised (the first block's balance holds
    −carry, as a forward-only node's first step), no solve has the wrap-target pin, and only the last solve has the
    end pin.  Before, the blocks looped in each solve: no carry in the balance (bound 0)."""
    url, scen = make_db(tmp_path_factory, alt="sse_n12_seq_blocks", base_chain=SEQ, values=[
        ["node", "battery", "storage_binding_method", _s("bind_intraperiod_blocks")],
        ["node", "battery", "storage_start_end_method", _s("fix_start_end")], NO_LOSS])
    steps = list(run(tmp_path_factory, url, scen).values())
    assert [s.solve_name for s in steps] == ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]
    assert all(s.solution.optimal for s in steps)
    assert _carry(steps[0].flex_data) is None
    for i, step in enumerate(steps):
        fd = step.flex_data
        pb = build(fd)
        assert not row_terms(pb, PIN, "battery,"), step.solve_name
        assert bool(row_terms(pb, END, "battery,")) == (i == len(steps) - 1), step.solve_name
        if i == 0:
            continue
        want = _realised_end(steps[i - 1])
        assert _carry(fd) == pytest.approx(want, abs=1e-9)
        d, t = utf8(fd.nodeState_first_dt).filter(pl.col("n") == "battery").select("d", "t").row(0)
        name = f"nodeBalanceBlock_eq[battery,{d},{t}]"
        lo, hi = row_bounds(pb, "nodeBalanceBlock_eq", f"battery,{d},{t}")[name]
        assert lo == pytest.approx(-want, abs=1e-9) and hi == pytest.approx(lo), step.solve_name


class _Provider:
    def __init__(self, frames: dict):
        self.frames = dict(frames)

    def has(self, key: str) -> bool:
        return key in self.frames

    def get(self, key: str):
        return self.frames[key]

    def put(self, key: str, frame) -> None:
        self.frames[key] = frame


def test_fix_start_end_mapping_rewrites_only_fix_start_end_nodes():
    """The per-solve mapping: a fix_start_end node's cyclic method becomes its forward-only equivalent (blended →
    forward-only blended); bind_forward_only, bind_intraperiod_blocks (the model chains its blocks forward) and the
    nodes of the other start / end methods keep their method."""
    import logging

    from flextool.engine.solve._run_model import _fix_start_end_chains_forward

    methods = {"a": "bind_within_solve", "b": "bind_within_timeblock", "c": "bind_within_period",
               "d": "bind_within_solve_blended_weights", "e": "bind_within_period_blended_weights",
               "f": "bind_forward_only", "g": "bind_intraperiod_blocks", "h": "bind_within_solve",
               "i": "bind_within_solve"}
    start_end = {n: "fix_start_end" for n in "abcdefg"} | {"h": "fix_start", "i": "fix_end"}
    provider = _Provider({
        "input/node__storage_binding_method": pl.DataFrame({"node": list(methods),
                                                            "storage_binding_method": list(methods.values())}),
        "input/node__storage_start_end_method": pl.DataFrame({"node": list(start_end),
                                                              "storage_start_end_method": list(start_end.values())})})
    _fix_start_end_chains_forward(solve="s", provider=provider, logger=logging.getLogger("test"))
    got = dict(provider.get("input/node__storage_binding_method").iter_rows())
    assert got == {"a": "bind_forward_only", "b": "bind_forward_only", "c": "bind_forward_only",
                   "d": "bind_forward_only_blended_weights", "e": "bind_forward_only_blended_weights",
                   "f": "bind_forward_only", "g": "bind_intraperiod_blocks", "h": "bind_within_solve",
                   "i": "bind_within_solve"}


# ---------------------------------------------------------------------------
# A LP rebuilt after the solve is the LP that was solved (item 149)
# ---------------------------------------------------------------------------


def _run_keeping_workdir(tmp_path_factory, url: str, scen: str) -> tuple[list, Path]:
    """Run the chain in a ``work_<scenario>`` folder next to a copy of the database (the layout ``load_flextool``
    expects for a reload) and return ``(steps, work_folder)``."""
    import shutil
    from urllib.parse import urlparse

    from flextool.engine import run_chain_from_db

    wf = tmp_path_factory.mktemp(f"{scen}_root") / f"work_{scen}"
    wf.mkdir()
    steps = list(run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True).values())
    shutil.copy(urlparse(url).path, wf / "tests.sqlite")
    return steps, wf


@pytest.mark.parametrize("chain", ["single", "sequence"])
def test_rebuild_from_the_post_solve_provider_is_the_solved_lp(tmp_path_factory, chain):
    """After a solve the Provider keeps the carriers the solve consumed (``handoff/*``) and exposes its own end state
    beside them (``produced_handoff/*``).  ``load_flextool(work_folder, provider=<post-solve provider>)`` then rebuilds
    the last solve's LP: the same carry (none in a single solve, which starts from fix_start 0.1·50 = 5 MWh; the
    previous solve's realised end in the 4-solve sequence) and the same optimum; the csv dump holds the same carry in
    ``handoff/roll_end_state.csv`` and the solve's own end in ``produced_handoff/roll_end_state.csv``.  Before, the
    post-solve provider held the solve's own realised end as ``handoff/roll_end_state``, so the rebuild started from
    it (single: no fix_start, carry = its own end)."""
    values = [["node", "battery", "storage_binding_method", _s("bind_forward_only")],
              ["node", "battery", "storage_solve_horizon_method", _s("free")], NO_LOSS]
    url, scen = make_db(tmp_path_factory, alt=f"sse_rebuild_{chain}",
                        base_chain=SEQ if chain == "sequence" else WIND_BATTERY, values=values)
    steps, wf = _run_keeping_workdir(tmp_path_factory, url, scen)
    last = steps[-1]
    want_carry = _realised_end(steps[-2]) if chain == "sequence" else None
    own_end = _realised_end(last)
    assert want_carry is None or abs(want_carry - own_end) > 1.0, "the test needs a carry that differs from the end"

    def _as_carry(value: "float | None"):
        return pytest.approx(value, abs=1e-9) if value is not None else None

    assert _carry(last.flex_data) == _as_carry(want_carry)

    from polar_high import Problem

    from flextool.engine import build_flextool, load_flextool
    from flextool.engine.preprocess import _provider_keys as K

    provider = last.flex_data_provider
    rebuilt = load_flextool(wf, provider=provider)
    assert _carry(rebuilt) == _as_carry(want_carry)
    pb = Problem()
    build_flextool(pb, rebuilt)
    sol = pb.solve()
    assert sol.optimal
    assert sol.obj == pytest.approx(last.solution.obj, rel=1e-9)

    provider.snapshot_processed_inputs(wf)
    for folder, want in (("handoff", want_carry), ("produced_handoff", own_end)):
        dumped = pl.read_csv(wf / folder / "roll_end_state.csv").filter(pl.col("node") == "battery")
        got = float(dumped["value"][0]) if dumped.height else None
        assert got == _as_carry(want), folder
    produced = utf8(provider.get(K.produced_handoff_key(K.HANDOFF_ROLL_END_STATE))).filter(pl.col("node") == "battery")
    assert float(produced["value"][0]) == pytest.approx(own_end, abs=1e-9)
