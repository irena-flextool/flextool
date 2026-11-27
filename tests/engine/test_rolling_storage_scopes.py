"""Rolling storage per period / timeblock, the cumulative nested fix_usage cap and the interpolated nested target.

A tiny model built from the schema JSON (never a checked-in ``.sqlite``), like ``test_rolling_storage_anchor``:
timeline ``tl`` t01..t16 (1 h steps), storage node ``tank`` (100 MWh, ``fix_start`` 0.5 = 50 MWh, no loss) feeding
the balance node ``load`` through ``dis``; ``gen`` charges the tank at 1 CUR/MWh when available.  Penalties are
10 000 CUR/MWh (the tank's own slacks 20 000).

* Per-period carry (bind_within_period, bind_intraperiod_blocks; bind_within_timeblock, which a rolling solve uses as
  bind_within_period): a rolling solve whose roll boundaries fall inside a period continues that period from the
  carried level, anchors the roll's part and closes the period at the level it started from in the roll(s) holding
  its end (logic doc Q3 / Q4, Lemma 4:
  free energy arises iff a realised roll boundary lies strictly inside a cyclic scope).
* Cumulative fix_usage (logic doc §4.6): a rolling lower level under an upper fix_usage is capped by the upper
  usage from the start of the period less what its earlier rolls realised.
* Q6: a lower horizon that ends inside an upper step takes the upper level interpolated at that instant.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from tests.engine._lp_rows import row_bounds
from tests.engine._unit_db import build, utf8

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_JSON = FLEXTOOL_ROOT / "flextool" / "schemas" / "spinedb_schema.json"
ALT, SCENARIO = "tank", "tank_s"
STEPS = [f"t{i:02d}" for i in range(1, 17)]
ANCHOR, CLOSE = "storage_state_roll_anchor", "storage_state_roll_close"


def tmap(data: dict, index_name: str = "time") -> api.Map:
    return api.Map(list(data), list(data.values()), index_name=index_name)


def arr(values: list[str]) -> api.Array:
    return api.Array(list(values))


def series(values: dict, default: float) -> api.Map:
    return tmap({t: values.get(t, default) for t in STEPS})


def build_db(tmp_dir: Path, solves: dict, *, top: list[str], tank: dict, load: dict, gen: dict,
             timesets: dict | None = None) -> str:
    """The database (url): ``load`` {step: MW} drawn from the tank, ``gen`` {step: availability} of the 10 MW
    charger, ``timesets`` {name: {first step: hours}} (``full`` and ``coarse`` cover the 16 h by default)."""
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    db_path = Path(tmp_dir) / "tank.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db_path))
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    timesets = {"full": {"t01": 16.0}, "coarse": {"t01": 16.0}, **(timesets or {})}
    entities = [("model", "flextool"), ("timeline", "tl"), ("node", "tank"), ("node", "load"),
                ("unit", "dis"), ("unit", "gen"), ("unit__inputNode", ("dis", "tank")),
                ("unit__outputNode", ("dis", "load")), ("unit__outputNode", ("gen", "tank"))]
    entities += [("timeset", ts) for ts in timesets] + [("solve", s) for s in solves]
    tank_values = {"node_type": "storage", "existing": 100.0, "storage_binding_method": "bind_within_solve",
                   "storage_start_end_method": "fix_start", "storage_state_start": 0.5,
                   "penalty_up": 20000.0, "penalty_down": 20000.0, **tank}
    values = [
        ("model", "flextool", "solves", arr(top)),
        ("timeline", "tl", "timestep_duration", tmap({t: 1.0 for t in STEPS})),
        ("node", "load", "node_type", "balance"),
        ("node", "load", "inflow", series({t: -v for t, v in load.items()}, 0.0)),
        ("node", "load", "penalty_up", 10000.0),
        ("node", "load", "penalty_down", 10000.0),
        ("unit", "dis", "existing", 100.0),
        ("unit", "dis", "efficiency", 1.0),
        ("unit", "gen", "existing", 10.0),
        ("unit", "gen", "efficiency", 1.0),
        ("unit", "gen", "availability", series(gen, 0.0)),
        ("unit__outputNode", ("gen", "tank"), "other_operational_cost", 1.0),
    ] + [("node", "tank", k, v) for k, v in tank_values.items()]
    for ts, blocks in timesets.items():
        values += [("timeset", ts, "timeline", "tl"), ("timeset", ts, "timeset_duration", tmap(blocks))]
    for solve, params in solves.items():
        values += [("solve", solve, par, val) for par, val in params.items()]
    with open_database_mapping(url) as db:
        _count, errors = api.import_data(
            db, alternatives=[(ALT, "")], scenarios=[(SCENARIO, False, "")],
            scenario_alternatives=[(SCENARIO, ALT, None)], entities=entities,
            entity_alternatives=[(c, (e,), ALT, True) for c, e in (("node", "tank"), ("node", "load"),
                                                                   ("unit", "dis"), ("unit", "gen"))],
            parameter_values=[(c, e, p, v, ALT) for c, e, p, v in values])
        if errors:
            raise RuntimeError(f"import errors: {errors[:5]}")
        db.commit_session("tank model")
    return url


def run(tmp_path, solves: dict, *, top: list[str] | None = None, tank: dict | None = None, load: dict | None = None,
        gen: dict | None = None, timesets: dict | None = None) -> dict:
    from flextool.engine import run_chain_from_db

    tmp_path.mkdir(parents=True, exist_ok=True)
    url = build_db(tmp_path, solves, top=top or list(solves), tank=tank or {},
                   load={t: 5.0 for t in ("t05", "t06", "t07", "t08")} if load is None else load,
                   gen={t: 1.0 for t in ("t09", "t10", "t11", "t12")} if gen is None else gen, timesets=timesets)
    steps = run_chain_from_db(url, SCENARIO, work_folder=tmp_path / "work", keep_solutions=True)
    for step in steps.values():
        assert step.solution is not None
    return steps


def rolls(steps: dict, solve: str = "st") -> list:
    out = [s for name, s in steps.items() if name.startswith(f"{solve}_") and "_roll_" in name]
    return sorted(out, key=lambda s: int(s.solve_name.rsplit("_", 1)[1]))


def levels(step, node: str = "tank") -> dict:
    """``{(period, step): MWh}`` of the node's v_state in a solved roll."""
    su = float(utf8(step.flex_data.p_state_unitsize.frame).filter(pl.col("n") == node)["value"][0])
    rows = utf8(step.solution.value("v_state")).filter(pl.col("n") == node)
    return {(d, t): v * su for d, t, v in zip(rows["d"].to_list(), rows["t"].to_list(), rows["value"].to_list())}


def kinds(step, node: str = "tank") -> list:
    sre = step.flex_data.storage_roll_end
    if sre is None:
        return []
    return sorted(utf8(sre).filter(pl.col("n") == node)["kind"].to_list())


def carry(step, node: str = "tank") -> "float | None":
    c = step.flex_data.p_roll_continue_state
    if c is None:
        return None
    hit = utf8(c.frame).filter(pl.col("n") == node)
    return float(hit["value"][0]) if hit.height > 0 else None


def two_periods(**extra) -> dict:
    """A rolling solve over periods p1 and p2 (the 16 h each), jump 6 h, horizon 8 h: the rolls start at p1 t01,
    t07, t13 (into p2 t01..t04), p2 t03, t09 and t15, so roll boundaries fall inside both periods."""
    return {"solve_mode": "rolling_window", "rolling_solve_jump": 6.0, "rolling_solve_horizon": 8.0,
            "rolling_duration": 32.0, "period_timeset": tmap({"p1": "full", "p2": "full"}, "period"),
            "years_represented": tmap({"p1": 1.0, "p2": 1.0}, "period"), "realized_periods": arr(["p1", "p2"]),
            **extra}


def realised(rs: list, jump: int = 6) -> list:
    """``[(period, step, MWh)]``: the realised level at the end of each roll's realised part."""
    out = []
    for k, r in enumerate(rs):
        lv = levels(r)
        order = sorted(lv)
        out.append((*order[min(jump, len(order)) - 1], lv[order[min(jump, len(order)) - 1]]))
    return out


# ---------------------------------------------------------------------------
# bind_within_period / bind_intraperiod_blocks (and bind_within_timeblock as bind_within_period) across roll
# boundaries; rolling over a timeline with gaps is an error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["bind_within_period", "bind_within_timeblock"])
def test_period_scope_carried_anchored_and_closed(tmp_path, method):
    """Each roll that starts inside a period continues it from the carry (no wrap of its first step), anchors
    its part and closes the period at the level the period started from where it holds the period's end.  A
    timeblock that is the whole period is the same scope."""
    rs = rolls(run(tmp_path, {"st": two_periods()}, tank={"storage_binding_method": method}))
    assert [kinds(r) for r in rs] == [
        [], ["scope_anchor", "scope_carry"], ["scope_carry", "scope_close"], ["scope_anchor", "scope_carry"],
        ["scope_carry", "scope_close"], ["scope_carry", "scope_close"]]
    # roll 1 starts from roll 0's realised t06 level and takes it as its carry (no wrap at its first step)
    assert carry(rs[1]) == pytest.approx(levels(rs[0])[("p1", "t06")], abs=1e-6)
    pb1 = build(rs[1].flex_data)
    assert row_bounds(pb1, ANCHOR, "tank") == {f"{ANCHOR}[tank,p1,t14]": (pytest.approx(carry(rs[1])),
                                                                           float("inf"))}
    # roll 2 holds p1's end: p1 closes at its start (fix_start, 50 MWh); p2 begins in roll 2 and loops on its part
    assert row_bounds(build(rs[2].flex_data), CLOSE, "tank") == {
        f"{CLOSE}[tank,p1,t16]": (pytest.approx(50.0), pytest.approx(50.0))}
    p2_start = levels(rs[2])[("p2", "t04")]                    # p2's first step wraps to p2 t04 in roll 2
    for r in rs[4:]:
        (key, (lo, hi)), = row_bounds(build(r.flex_data), CLOSE, "tank").items()
        assert key == f"{CLOSE}[tank,p2,t16]" and lo == pytest.approx(p2_start) and hi == pytest.approx(p2_start)
    # the realised path is continuous and each period ends where it started
    for prev, nxt in zip(rs, rs[1:]):
        assert carry(nxt) == pytest.approx(realised([prev])[0][2], abs=1e-6)
    assert levels(rs[2])[("p1", "t16")] == pytest.approx(50.0, abs=1e-6)
    assert levels(rs[-1])[("p2", "t16")] == pytest.approx(p2_start, abs=1e-6)


def test_aligned_period_rolls_unchanged(tmp_path):
    """A jump of a whole period puts no realised roll boundary inside a period: no storage_roll_end rows."""
    rs = rolls(run(tmp_path, {"st": two_periods(rolling_solve_jump=16.0, rolling_solve_horizon=16.0)},
                   tank={"storage_binding_method": "bind_within_period"}))
    assert len(rs) == 2 and all(r.flex_data.storage_roll_end is None for r in rs)


def test_timeblock_in_rolling_is_within_period(tmp_path, coded_log):
    """Owner decision (Phase 5): a rolling solve uses bind_within_timeblock as bind_within_period (each period of a
    rolling solve is one continuous timeblock).  The same model with the two methods gives, roll by roll, the same
    storage_roll_end rows, the same anchor / close bounds and the same objective; one printed line names the
    conversion."""
    rs = {}
    for method in ("bind_within_timeblock", "bind_within_period"):
        rs[method] = rolls(run(tmp_path / method, {"st": two_periods()}, tank={"storage_binding_method": method}))
        out = coded_log()
        lines = [ln for ln in out.splitlines() if "[W-0532]" in ln]
        assert lines == (["Note: rolling 'st': bind_within_timeblock used as bind_within_period ('tank') [W-0532]"]
                         if method == "bind_within_timeblock" else []), out[-2000:]
    tb, wp = rs["bind_within_timeblock"], rs["bind_within_period"]
    assert len(tb) == len(wp)
    for a, b in zip(tb, wp):
        assert kinds(a) == kinds(b)
        tb_set = a.flex_data.storage_bind_within_timeblock
        assert tb_set is None or tb_set.height == 0
        pa, pb = build(a.flex_data), build(b.flex_data)
        for name in (ANCHOR, CLOSE):
            ba, bb = row_bounds(pa, name, "tank"), row_bounds(pb, name, "tank")
            # pytest >= 9.2 approx() rejects a mapping of tuples: compare the keys, then each (lower, upper) pair.
            assert ba.keys() == bb.keys()
            for row in ba:
                assert ba[row] == pytest.approx(bb[row]), row
        assert a.solution.objective_in_user_units == pytest.approx(b.solution.objective_in_user_units, rel=1e-9,
                                                                   abs=1e-6)
        assert levels(a) == pytest.approx(levels(b), abs=1e-6)


def test_timeblock_upper_solve_keeps_its_cycle_above_rolling(tmp_path):
    """An upper solve that is not rolling keeps bind_within_timeblock (it cycles within its timeblock), while the
    rolling solve nested in it uses bind_within_period."""
    solves, tank = nested("fix_quantity", {})
    steps = run(tmp_path, solves, top=["up"], tank={**tank, "storage_binding_method": "bind_within_timeblock"})
    up = steps["up"].flex_data
    assert up.storage_bind_within_timeblock is not None and up.storage_bind_within_timeblock.height == 1
    assert up.storage_bind_within_period is None or up.storage_bind_within_period.height == 0
    for r in rolls(steps):
        fd = r.flex_data
        assert fd.storage_bind_within_timeblock is None or fd.storage_bind_within_timeblock.height == 0
        in_roll = set()
        for frame in (fd.storage_bind_within_period, fd.storage_bind_forward_only):
            if frame is not None:
                in_roll |= set(utf8(frame)["n"].to_list())
        assert "tank" in in_roll


def test_rolling_over_sampled_timeblocks_is_an_error(tmp_path):
    """Owner decision (Phase 5): a rolling solve over a timeline with gaps (two sampled blocks t01-t06 and t09-t14
    in one period) is a configuration error, whatever the storage method."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    solve = {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
             "rolling_duration": 12.0, "period_timeset": tmap({"p1": "blocks"}, "period"),
             "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"])}
    with pytest.raises(FlexToolConfigError, match=r"Rolling solve 'st' runs over a timeline with gaps \(period 'p1' "
                                                  r"has 2 timeblocks\)"):
        run(tmp_path, {"st": solve}, tank={"storage_binding_method": "bind_forward_only"},
            timesets={"blocks": {"t01": 6.0, "t09": 6.0}})


def test_rolling_over_adjacent_timeset_blocks_runs(tmp_path):
    """Timeset blocks that follow each other on the timeline (t01-t08 and t09-t16) are one continuous timeblock: no
    error."""
    solve = {**two_periods(), "period_timeset": tmap({"p1": "split", "p2": "split"}, "period")}
    rs = rolls(run(tmp_path, {"st": solve}, tank={"storage_binding_method": "bind_within_period"},
                   timesets={"split": {"t01": 8.0, "t09": 8.0}}))
    assert len(rs) == 6


def test_rolling_with_coarse_steps_counts_one_timeblock(tmp_path):
    """A rolling solve with new_stepduration (2 h steps): the aggregated steps are consecutive, so the period is one
    timeblock (no gap error) and the timeblock storage runs as bind_within_period."""
    solve = {**two_periods(rolling_solve_jump=6.0, rolling_solve_horizon=8.0), "new_stepduration": 2.0}
    rs = rolls(run(tmp_path, {"st": solve}, tank={"storage_binding_method": "bind_within_timeblock"}))
    assert len(rs) >= 2
    assert any("scope_carry" in kinds(r) for r in rs)


def test_gap_counter_counts_blocks_per_period():
    """The gap check counts timeblocks per period (index jumps), not periods."""
    from types import SimpleNamespace

    from flextool.engine.input._solve_state import ActiveTimeEntry, FlexToolConfigError
    from flextool.engine.solve._run_model._rolling_storage import _reject_rolling_with_gaps

    state = SimpleNamespace(solve=SimpleNamespace(solve_modes={"st": "rolling_window", "one": "single_solve"}))

    def steps(*idx):
        return [ActiveTimeEntry(f"t{i:02d}", i, "1") for i in idx]

    _reject_rolling_with_gaps(state=state, complete_solve_name="st",
                              complete_active_time_list={"p1": steps(0, 1, 2), "p2": steps(0, 1, 2)})
    _reject_rolling_with_gaps(state=state, complete_solve_name="one",
                              complete_active_time_list={"p1": steps(0, 1, 5)})
    with pytest.raises(FlexToolConfigError, match="period 'p2' has 3 timeblocks"):
        _reject_rolling_with_gaps(state=state, complete_solve_name="st",
                                  complete_active_time_list={"p1": steps(0, 1), "p2": steps(0, 2, 3, 7)})


def test_many_storage_nodes_print_one_line(coded_log):
    """Owner rule (format 6b): a storage message prints one line per message type and solve, naming how many nodes
    and the first only, whatever the number of nodes and rolls."""
    from types import SimpleNamespace

    from flextool.engine import _behaviour_changes
    from flextool.engine.solve._run_model._rolling_storage import _timeblock_binds_within_period_in_roll

    class Provider:
        def __init__(self, frame):
            self.frames = {"input/node__storage_binding_method": frame}

        def has(self, key):
            return key in self.frames

        def get(self, key):
            return self.frames[key]

        def put(self, key, frame):
            self.frames[key] = frame

    _behaviour_changes.start_run({})
    names = [f"battery_{i:02d}" for i in range(12)]
    state = SimpleNamespace(solve=SimpleNamespace(solve_modes={"st": "rolling_window"}),
                            timeline=SimpleNamespace(rp_weights={}))
    for roll in range(3):
        provider = Provider(pl.DataFrame({"node": names, "storage_binding_method": ["bind_within_timeblock"] * 12}))
        _timeblock_binds_within_period_in_roll(state=state, complete_solve_name="st", active_timeset_names=["full"],
                                               provider=provider)
        assert set(provider.get("input/node__storage_binding_method")["storage_binding_method"].to_list()) == {
            "bind_within_period"}
    from flextool.engine.solve._run_model._storage_frames import _warn_storages
    for roll in range(3):
        _warn_storages("W-0582", f"st_roll_{roll}", names, f"roll st_roll_{roll}")
    log = coded_log()
    assert [ln for ln in log.splitlines() if "[W-0532]" in ln] == [
        "Note: rolling 'st': bind_within_timeblock used as bind_within_period (12 cases, e.g. 'battery_00') [W-0532]"]
    assert [ln for ln in log.splitlines() if ln.endswith("[W-0582]")] == [
        "Warning: rolling 'st': period start unknown; storage closes at the carry (12 cases, e.g. 'battery_00') "
        "[W-0582]"]


def test_intraperiod_blocks_close_the_period(tmp_path):
    """bind_intraperiod_blocks held in part by every roll (logic doc §4.3 gap): the blocks chain forward from roll
    to roll, and the roll(s) holding a period's end now close it at the level it started from (before, the
    horizon method set that end and the period's cycle never closed)."""
    rs = rolls(run(tmp_path, {"st": two_periods()}, tank={"storage_binding_method": "bind_intraperiod_blocks"}))
    assert kinds(rs[0]) == ["anchor_share"]
    assert kinds(rs[1]) == ["scope_anchor"] and kinds(rs[2]) == ["scope_close"]
    assert row_bounds(build(rs[2].flex_data), CLOSE, "tank") == {
        f"{CLOSE}[tank,p1,t16]": (pytest.approx(50.0), pytest.approx(50.0))}
    p2_start = levels(rs[2])[("p1", "t16")]                     # p2's blocks chain on from p1's end in roll 2
    assert levels(rs[-1])[("p2", "t16")] == pytest.approx(p2_start, abs=1e-6)


# ---------------------------------------------------------------------------
# Nested: an upper storage solve ``up`` (4 h steps) fixes the tank of the rolling ``st``
# ---------------------------------------------------------------------------


def nested(method: str, st: dict) -> tuple[dict, dict]:
    up = {"solve_mode": "single_solve", "new_stepduration": 4.0, "period_timeset": tmap({"p1": "coarse"}, "period"),
          "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"]),
          "fix_storage_periods": arr(["p1"]), "contains_solves": arr(["st"])}
    base = {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
            "rolling_duration": 16.0, "period_timeset": tmap({"p1": "full"}, "period"),
            "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"])}
    return {"up": up, "st": {**base, **st}}, {"storage_nested_fix_method": method}


def usage_mwh(step, steps: list[str]) -> float:
    """The tank's net usage (dis input − gen output, MWh) over ``steps`` of a solved roll."""
    flows = utf8(step.solution.value("v_flow")).filter(pl.col("t").is_in(steps))
    us = dict(zip(*[utf8(step.flex_data.p_unitsize.frame)[c].to_list() for c in ("p", "value")]))
    out = flows.filter(pl.col("p") == "dis")["value"].sum() * us["dis"]
    return float(out - flows.filter(pl.col("p") == "gen")["value"].sum() * us["gen"])


def test_fix_usage_is_cumulative_over_rolls(tmp_path):
    """Logic doc §4.6.  The load (10 MW in t01-t02) comes before the charger (10 MW in t03-t04): the 4 h upper
    step averages both and draws nothing from the tank in t01-t04 (usage 0 in every window).  The hourly rolls
    (jump 2, horizon 4) must serve t01-t02 from the tank.  With a cap per horizon, roll 1 [t03-t06] sees a cap of
    0 and need not recharge: the realised rolls would draw 20 MWh the upper solve never planned.  The cumulative
    cap of roll 1 is the upper usage from t01 (0) less roll 0's realised 20 MWh, so it recharges."""
    solves, tank = nested("fix_usage", {"rolling_solve_jump": 2.0, "rolling_solve_horizon": 4.0})
    steps = run(tmp_path, solves, top=["up"], tank=tank, load={"t01": 10.0, "t02": 10.0},
                gen={"t03": 1.0, "t04": 1.0})
    rs = rolls(steps)
    upper_total = usage_mwh(steps["up"], STEPS)
    assert upper_total == pytest.approx(0.0, abs=1e-6)
    assert usage_mwh(rs[0], ["t01", "t02"]) == pytest.approx(20.0, abs=1e-6)
    caps = row_bounds(build(rs[1].flex_data), "node_storage_usage_fix_le", "tank")
    assert [hi for _lo, hi in caps.values()] == [pytest.approx(-20.0)]
    realised_total = sum(usage_mwh(r, [f"t{2 * k + 1:02d}", f"t{2 * k + 2:02d}"]) for k, r in enumerate(rs))
    assert realised_total <= upper_total + 1e-6
    assert levels(rs[-1])[("p1", "t16")] == pytest.approx(50.0, abs=1e-6)


def test_fix_quantity_interpolated_inside_an_upper_step(tmp_path):
    """Q6: a roll whose horizon ends inside an upper 4 h step (jump 2, horizon 6: the first roll ends at t06,
    halfway through the upper step t05-t08) is fixed at the upper level interpolated there, not at the step's
    end level."""
    solves, tank = nested("fix_quantity", {"rolling_solve_jump": 2.0, "rolling_solve_horizon": 6.0})
    steps = run(tmp_path, solves, top=["up"], tank=tank)
    up = levels(steps["up"])
    up_steps = sorted(up)
    at_t04 = up[next(k for k in up_steps if k[1] == "t01")]     # the 4 h step t01-t04 is named t01
    at_t08 = up[next(k for k in up_steps if k[1] == "t05")]
    rs = rolls(steps)
    (key, (lo, hi)), = row_bounds(build(rs[0].flex_data), "node_balance_fix_quantity_eq_lower", "tank").items()
    expected = at_t04 + 0.5 * (at_t08 - at_t04)
    assert key.endswith(",p1,t06]")
    assert lo == pytest.approx(expected) and hi == pytest.approx(expected)
    assert "fix_quantity" in kinds(rs[0])
    # an aligned roll end (t08) keeps the step's end level and gets no restated row
    aligned = next(r for r in rs if max(t for _d, t in levels(r)) == "t08")
    assert "fix_quantity" not in kinds(aligned)


def test_upper_level_interpolation_helper():
    """_upper_level_interp: linear inside a step from the level the step starts at (the step before, or the upper
    start level for the first step) to its end; the end level where a step ends; the start at the first step."""
    from types import SimpleNamespace

    from flextool.engine.solve._run_model._nested_storage import _upper_level_interp

    handoff = SimpleNamespace(
        storage_level=pl.DataFrame({"node": ["n", "n"], "period": ["p", "p"], "step": ["a", "b"],
                                    "value": [10.0, 30.0]}),
        cycle_start_state=pl.DataFrame({"node": ["n"], "value": [4.0]}))
    index = SimpleNamespace(hours=lambda root, a, b: float(b - a))
    steps = [(0, "tl", 0, 4, "p", "a"), (0, "tl", 4, 8, "p", "b")]

    def hit(i: int):
        return _upper_level_interp(node="n", at=(0, i), steps=steps, parent_handoff=handoff, time_index=index)

    assert hit(0) == (4.0, True)
    assert hit(1) == (pytest.approx(5.5), False)
    assert hit(4) == (10.0, True)
    assert hit(6) == (pytest.approx(20.0), False)
    assert hit(8) == (30.0, True)
    assert hit(9) is None


def test_rolling_forward_free_start_matches_its_end(tmp_path):
    """A bind_forward_only node with a free start (fix_nothing) in a rolling solve: the first roll's horizon end is
    held to the level it starts from (storage_state_end_match) and the rolls reaching the end of the model end at
    least at that level (storage_state_roll_match), so the realised run cannot start full and end empty."""
    solve = {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
             "rolling_duration": 16.0, "period_timeset": tmap({"p1": "full"}, "period"),
             "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"])}
    rs = rolls(run(tmp_path, {"st": solve}, tank={"storage_binding_method": "bind_forward_only",
                                                  "storage_start_end_method": "fix_nothing"}))
    assert [kinds(r) for r in rs] == [["match_first"], [], ["match"], ["match"]]
    assert list(row_bounds(build(rs[0].flex_data), "storage_state_end_match", "tank")) == [
        "storage_state_end_match[tank,p1,t01,p1,t08]"]
    start = levels(rs[0])[("p1", "t01")]
    for r in rs[2:]:
        (key, (lo, hi)), = row_bounds(build(r.flex_data), "storage_state_roll_match", "tank").items()
        assert key.endswith(",p1,t16]") and lo == pytest.approx(start) and hi == float("inf")
    assert levels(rs[-1])[("p1", "t16")] >= start - 1e-6


def test_fix_storage_timesteps_output_set(tmp_path):
    """The output set dt_fix_storage_timesteps holds the steps of a solve's own fix_storage_periods (it was always
    empty), and is empty in a solve without them."""
    from flextool.process_outputs.read_sets import read_sets

    solves, tank = nested("fix_usage", {})
    steps = run(tmp_path, solves, top=["up"], tank=tank)
    up = read_sets(steps["up"].flex_data, steps["up"].solution, solve_name="up")
    assert sorted(t for _s, _d, t in up.dt_fix_storage_timesteps) == ["t01", "t05", "t09", "t13"]
    roll = rolls(steps)[0]
    assert len(read_sets(roll.flex_data, roll.solution, solve_name=roll.solve_name).dt_fix_storage_timesteps) == 0


def test_storage_usage_output(tmp_path):
    """node_storage_usage_dt_e: the realised net use of the fix_usage tank in each step (MWh, out − in), written by
    the output writer (parquet and csv): the rolls of the cumulative fix_usage case above draw 10 MWh an hour in
    t01-t02 and recharge it in t03-t04; 0 elsewhere."""
    import contextlib
    import io

    from flextool.lean_parquet import read_lean_parquet
    from flextool.process_outputs.out_node import node_storage_usage
    from flextool.process_outputs.write_outputs import write_outputs

    solves, tank = nested("fix_usage", {"rolling_solve_jump": 2.0, "rolling_solve_horizon": 4.0})
    steps = run(tmp_path, solves, top=["up"], tank=tank, load={"t01": 10.0, "t02": 10.0},
                gen={"t03": 1.0, "t04": 1.0})
    last = next(reversed(steps.values()))
    out = tmp_path / "out"
    with contextlib.redirect_stdout(io.StringIO()):
        write_outputs(SCENARIO, output_location=str(out), subdir=SCENARIO,
                      output_config_path=str(FLEXTOOL_ROOT / "flextool" / "schemas" / "default_plots.yaml"),
                      output_funcs=[node_storage_usage], write_methods=["parquet", "csv"],
                      fallback_output_location=str(tmp_path / "work"),
                      raw_output_dir=str(tmp_path / "work" / "output_raw"), solution=last.solution,
                      solve_name=last.solve_name, flex_data=last.flex_data,
                      flex_data_provider=last.flex_data_provider)
    frame = read_lean_parquet(out / "output_parquet" / SCENARIO / "node_storage_usage_dt_e.parquet")
    usage = {t: v for (_d, t), v in frame.droplevel("scenario", axis=1)["tank"].items()}
    assert usage == pytest.approx({t: {"t01": 10.0, "t02": 10.0, "t03": -10.0, "t04": -10.0}.get(t, 0.0)
                                   for t in STEPS}, abs=1e-6)
    assert (out / "output_csv" / SCENARIO / "node_storage_usage__dt.csv").is_file()


@pytest.mark.parametrize("start", ["fix_start", "fix_nothing"])
def test_nested_first_roll_of_a_period_starts_from_the_upper_start(tmp_path, start):
    """bind_within_period under an upper fix_quantity: the first roll's horizon cuts the period, so its first step
    would wrap to the horizon end, which the upper target fixes — the year would start at the upper level of t08,
    not at the upper start.  It starts instead from fix_start (forward, no wrap) or, without it, from the level the
    upper solve starts from (borrowed), and every roll ends at the upper target."""
    solves, tank = nested("fix_quantity", {})
    steps = run(tmp_path, solves, top=["up"],
                tank={**tank, "storage_binding_method": "bind_within_period", "storage_start_end_method": start})
    rs = rolls(steps)
    up = steps["up"]
    up_start = float(utf8(up.handoff.cycle_start_state).filter(pl.col("node") == "tank")["value"][0])
    assert "scope_carry" in kinds(rs[0])
    pb0 = build(rs[0].flex_data)
    assert not row_bounds(pb0, "storage_state_start_binding_cyclic_period", "tank")
    if start == "fix_start":
        assert carry(rs[0]) is None
        assert up_start == pytest.approx(50.0, abs=1e-6)
    else:
        assert carry(rs[0]) == pytest.approx(up_start, abs=1e-6)
    up_levels = levels(up)
    assert levels(rs[-1])[("p1", "t16")] == pytest.approx(up_levels[max(up_levels)], abs=1e-6)


@pytest.mark.parametrize("p2_capacity, closing", [(40.0, 40.0), (200.0, 50.0)])
def test_closing_level_capped_at_the_closing_period_capacity(tmp_path, p2_capacity, closing):
    """bind_within_solve, fix_start 0.5 on 100 MWh in p1, rolling over p1 and p2: the closing rolls end at the MWh
    the first roll started from (50; no new energy when p2's capacity grows to 200) but never above the storage p2
    holds (40 MWh when it shrinks; owner decision).  Before: always 50, so a shrinking store could not close."""
    rs = rolls(run(tmp_path, {"st": two_periods(rolling_solve_jump=4.0)},
                   tank={"existing": tmap({"p1": 100.0, "p2": p2_capacity}, "period")}))
    closes = [row_bounds(build(r.flex_data), CLOSE, "tank") for r in rs if "close" in kinds(r)]
    assert closes
    for rows in closes:
        (key, (lo, hi)), = rows.items()
        assert key.endswith(",p2,t16]") and lo == pytest.approx(closing) and hi == pytest.approx(closing)
    assert levels(rs[-1])[("p2", "t16")] == pytest.approx(closing, abs=1e-6)


@pytest.mark.parametrize("rolling", [False, True])
def test_capacity_added_later_starts_empty(tmp_path, rolling):
    """Storage capacity added in a later period (existing 100 → 200 MWh in p2) holds no energy of its own: a forward
    chain with fix_start 0.5 starts p1 at 50 MWh (the share of p1's capacity) and enters p2 at the level p1 left
    (no load or charging at p2 t01), not at a share of the larger store — in one solve and across rolls (the carry
    is MWh)."""
    solve = two_periods(rolling_solve_jump=4.0)
    if not rolling:
        solve = {k: v for k, v in solve.items() if not k.startswith("rolling")} | {"solve_mode": "single_solve"}
    steps = run(tmp_path, {"st": solve}, tank={"storage_binding_method": "bind_forward_only",
                                               "existing": tmap({"p1": 100.0, "p2": 200.0}, "period")})
    seq = rolls(steps) if rolling else [steps["st"]]
    lv: dict = {}
    for r in seq:                                            # the realised part of each roll (jump 4 h)
        own = levels(r)
        lv.update({k: own[k] for k in (sorted(own)[:4] if rolling else own)})
    assert lv[("p2", "t01")] == pytest.approx(lv[("p1", "t16")], abs=1e-6)
    assert lv[("p1", "t16")] <= 50.0 + 1e-6


def one_period_roll(period: str, **extra) -> dict:
    """A rolling solve over ``period`` (16 h), jump 4 h, horizon 8 h."""
    return {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
            "rolling_duration": 16.0, "period_timeset": tmap({period: "full"}, "period"),
            "years_represented": tmap({period: 1.0}, "period"), "realized_periods": arr([period]), **extra}


FREE_FORWARD = {"storage_binding_method": "bind_forward_only", "storage_start_end_method": "fix_nothing"}


def _start_level(step, node: str = "tank") -> float:
    return float(utf8(step.handoff.cycle_start_state).filter(pl.col("node") == node)["value"][0])


def test_free_forward_start_matched_over_a_sequence_of_rolling_solves(tmp_path):
    """Review H1: two rolling solves in sequence (st1 over p1, st2 over p2), a bind_forward_only tank with a free
    start and a 10 MW load at the start of p1.  The first roll of st1 holds its horizon end at its start
    (match_first); the closing rolls of st2, which reach the end of the model and continue the tank from st1's
    carry, end at least at the level the model started it from (match).  Before, st2 got no match: the tank could
    start p1 full and end p2 empty (40 MWh never produced)."""
    steps = run(tmp_path, {"st1": one_period_roll("p1"), "st2": one_period_roll("p2")}, top=["st1", "st2"],
                tank=FREE_FORWARD, load={t: 10.0 for t in ("t01", "t02", "t03", "t04")},
                gen={t: 1.0 for t in ("t05", "t06", "t07", "t08")})
    r1, r2 = rolls(steps, "st1"), rolls(steps, "st2")
    assert [kinds(r) for r in r1] == [["match_first"], [], [], []]
    assert [kinds(r) for r in r2] == [[], [], ["match"], ["match"]]
    start = _start_level(r1[0])
    for r in r2[2:]:
        (key, (lo, hi)), = row_bounds(build(r.flex_data), "storage_state_roll_match", "tank").items()
        assert key.endswith(",p2,t16]") and lo == pytest.approx(start) and hi == float("inf")
    assert levels(r2[-1])[("p2", "t16")] >= start - 1e-6


def test_free_forward_start_matched_in_a_later_instance_of_a_nested_rolling_solve(tmp_path):
    """Review H1: a rolling solve nested under a rolling upper solve of two rolls (jump 8 h): each upper roll has its
    own instance of the lower rolling solve.  The second instance reaches the end of the model and continues the
    tank from the first one; its closing roll ends at least at the level the model started the tank from at the
    lower level (a regression guard: this setup passed before the fix too)."""
    up = one_period_roll("p1", rolling_solve_jump=8.0, rolling_solve_horizon=8.0, contains_solves=arr(["st"]))
    st = one_period_roll("p1", rolling_solve_horizon=4.0)
    steps = run(tmp_path, {"up": up, "st": st}, top=["up"], tank=FREE_FORWARD,
                load={t: 10.0 for t in ("t01", "t02", "t03", "t04")},
                gen={t: 1.0 for t in ("t05", "t06", "t07", "t08")})
    lower = sorted((s for name, s in steps.items() if name.startswith("st_")), key=lambda s: s.solve_name)
    at_end = [s for s in lower if ("p1", "t16") in levels(s)]
    assert at_end and all("match" in kinds(s) for s in at_end)
    start = _start_level(lower[0])
    for s in at_end:
        (key, (lo, _hi)), = row_bounds(build(s.flex_data), "storage_state_roll_match", "tank").items()
        assert key.endswith(",p1,t16]") and lo == pytest.approx(start)
        assert levels(s)[("p1", "t16")] >= start - 1e-6


def test_model_start_is_kept_per_nesting_level():
    """Review M1: the level the model started a node from is recorded per nesting level, so a lower level's
    matching end compares with its own start, not with the upper level's; a level without its own record takes the
    nearest one."""
    from types import SimpleNamespace

    from flextool.engine.solve._run_model._storage_frames import _RollingStorage
    from flextool.engine.solve._run_model._nested_storage import _record_model_start

    store = _RollingStorage()
    for level, value in ((0, 80.0), (1, 30.0)):
        _record_model_start(store, SimpleNamespace(cycle_start_state=pl.DataFrame({"node": ["tank"],
                                                                                 "value": [value]})),
                            None, level=level)
    assert store.model_start_of(0, "tank") == 80.0
    assert store.model_start_of(1, "tank") == 30.0
    assert store.model_start_of(2, "tank") == 30.0
    assert store.model_start_of(0, "other") is None


def test_fix_usage_cap_covers_the_period_a_roll_realises_into_the_next(tmp_path):
    """Review M3: a rolling lower solve over p1 and p2 (jump 6 h, horizon 8 h) under an upper fix_usage on both
    periods.  The roll starting at p1 t13 realises p1's last four hours while its horizon ends in p2.  The load
    (10 MW at p1 t13-t14) and the charger (p1 t15-t16) cancel in the upper 4 h step, so the upper plans no usage in
    p1.  Before, that roll had only p2's cap: it served the load from the tank without recharging, and the realised
    rolls drew 20 MWh from p1 the upper solve never planned.  Now each period of a roll before the horizon's last
    one gets its own cumulative cap (usage_cap_period), so p1's realised usage stays within the upper plan."""
    up = {"solve_mode": "single_solve", "new_stepduration": 4.0,
          "period_timeset": tmap({"p1": "coarse", "p2": "coarse"}, "period"),
          "years_represented": tmap({"p1": 1.0, "p2": 1.0}, "period"), "realized_periods": arr(["p1", "p2"]),
          "fix_storage_periods": arr(["p1", "p2"]), "contains_solves": arr(["st"])}
    steps = run(tmp_path, {"up": up, "st": two_periods()}, top=["up"],
                tank={"storage_nested_fix_method": "fix_usage", "storage_binding_method": "bind_within_solve"},
                load={"t13": 10.0, "t14": 10.0}, gen={"t15": 1.0, "t16": 1.0})
    rs = rolls(steps)
    crossing = next(r for r in rs if ("p1", "t13") in levels(r) and any(d == "p2" for d, _t in levels(r)))
    assert "usage_cap_period" in kinds(crossing)
    caps = row_bounds(build(crossing.flex_data), "node_storage_usage_fix_le", "tank")
    assert any(key.endswith(",p1,t16]") for key in caps), caps
    realised_p1 = 0.0
    for r in rs:
        ts = [t for d, t in sorted(levels(r))[:6] if d == "p1"]     # the realised part (jump 6 h) in p1
        if ts:
            realised_p1 += usage_mwh_period(r, "p1", ts)
    upper_p1 = usage_mwh_period(steps["up"], "p1", STEPS)
    assert upper_p1 == pytest.approx(0.0, abs=1e-6)
    assert realised_p1 <= upper_p1 + 1e-6


def usage_mwh_period(step, period: str, steps: list[str]) -> float:
    """The tank's net usage (dis input − gen output, MWh) over ``steps`` of ``period`` in a solved step."""
    flows = utf8(step.solution.value("v_flow")).filter((pl.col("d") == period) & pl.col("t").is_in(steps))
    us = dict(zip(*[utf8(step.flex_data.p_unitsize.frame)[c].to_list() for c in ("p", "value")]))
    dur = utf8(step.flex_data.p_step_duration.frame).filter(pl.col("d") == period)
    hours = dict(zip(dur["t"].to_list(), dur["value"].to_list()))
    flows = flows.with_columns(h=pl.col("t").replace_strict(hours, default=1.0).cast(pl.Float64))
    out = (flows.filter(pl.col("p") == "dis")["value"] * flows.filter(pl.col("p") == "dis")["h"]).sum() * us["dis"]
    gen = flows.filter(pl.col("p") == "gen")
    return float(out - (gen["value"] * gen["h"]).sum() * us["gen"])


def test_core_storage_notices_print_one_line_per_solve(coded_log):
    """Review M4: the storage notices built in the model (rp_storage_chain_pinned, storage_end_matches_start) print
    one line per change and complete solve, naming how many nodes and the first, however many rows of the build
    name them (before: the full list, again for every new name).  The pre-branch storage_default_fix_nothing notice
    is in the same format (once per run)."""
    from types import SimpleNamespace

    from flextool.engine import _behaviour_changes
    from flextool.engine.model.core._storage_links import _notice_nodes

    marker = '{"since": 82}'
    _behaviour_changes.start_run({"rp_storage_chain_pinned": marker, "storage_end_matches_start": marker,
                                  "storage_default_fix_nothing": marker})
    names = [f"h2_{i:02d}" for i in range(12)]
    for solve in ("lt_rp", "lt_rp", "mt"):
        b = SimpleNamespace(solve_name=solve)
        _notice_nodes("rp_storage_chain_pinned", names[:6], b)        # start pins
        _notice_nodes("rp_storage_chain_pinned", names[6:], b)        # end pins
        _notice_nodes("storage_end_matches_start", pl.DataFrame({"n": names}), b)
    for _ in range(2):
        _behaviour_changes.notice("storage_default_fix_nothing", "storage nodes", names, solve="")
    lines = [ln for ln in coded_log().splitlines() if ln.startswith("Note: ")]
    assert len(lines) == 5, lines
    pinned = [ln for ln in lines if "storage chain start / end set (6 cases, e.g. 'h2_00')" in ln]
    assert len(pinned) == 2                                            # lt_rp and mt
    assert sum("ends at least at its start (12 cases, e.g. 'h2_00')" in ln for ln in lines) == 2
    assert sum("storage starts free by default (fix_nothing) (12 cases, e.g. 'h2_00')" in ln for ln in lines) == 1
