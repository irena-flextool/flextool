"""Where the storage chain starts and ends across solves (fix_start, the carry between solves, fix_end).

* ``bind_within_timeblock`` + ``fix_start`` pins the state the first step starts from: the wrap target of the first
  step (the last step of the first timeblock), not the state at the end of the first step.

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
