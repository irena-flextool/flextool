"""Rolling storage per period / timeblock, the cumulative nested fix_usage cap and the interpolated nested target.

A tiny model built from the schema JSON (never a checked-in ``.sqlite``), like ``test_rolling_storage_anchor``:
timeline ``tl`` t01..t16 (1 h steps), storage node ``tank`` (100 MWh, ``fix_start`` 0.5 = 50 MWh, no loss) feeding
the balance node ``load`` through ``dis``; ``gen`` charges the tank at 1 CUR/MWh when available.  Penalties are
10 000 CUR/MWh (the tank's own slacks 20 000).

* Per-scope carry (bind_within_period, bind_within_timeblock, bind_intraperiod_blocks): a rolling solve whose roll
  boundaries fall inside a period / timeblock continues that scope from the carried level, anchors the roll's part
  and closes the scope at the level it started from in the roll(s) holding its end (logic doc Q3 / Q4, Lemma 4:
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
# bind_within_period / bind_within_timeblock / bind_intraperiod_blocks across roll boundaries
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


def test_sampled_timeblocks_cut_by_rolls(tmp_path):
    """Two sampled blocks per period (t01-t06 and t09-t14): the roll starting at t05 continues the first block
    from the carry and closes it at t06 at its start level; the second block, begun in that roll, loops; the roll
    starting at t13 continues and closes it."""
    solve = {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
             "rolling_duration": 12.0, "period_timeset": tmap({"p1": "blocks"}, "period"),
             "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"])}
    rs = rolls(run(tmp_path, {"st": solve}, tank={"storage_binding_method": "bind_within_timeblock"},
                   timesets={"blocks": {"t01": 6.0, "t09": 6.0}}))
    steps = [sorted(t for _d, t in levels(r)) for r in rs]
    first_block = next(k for k, s in enumerate(steps) if s[0] in ("t02", "t03", "t04", "t05", "t06"))
    r = rs[first_block]
    assert "scope_carry" in kinds(r) and "scope_close" in kinds(r)
    (key, (lo, hi)), = row_bounds(build(r.flex_data), CLOSE, "tank").items()
    assert key == f"{CLOSE}[tank,p1,t06]" and lo == pytest.approx(50.0) and hi == pytest.approx(50.0)
    assert levels(r)[("p1", "t06")] == pytest.approx(50.0, abs=1e-6)
    # the second block began in that roll (its first step wraps to t14) and the next roll closes it at that level
    nxt = rs[first_block + 1]
    assert "scope_carry" in kinds(nxt)
    (key, (lo, hi)), = row_bounds(build(nxt.flex_data), CLOSE, "tank").items()
    assert key == f"{CLOSE}[tank,p1,t14]" and lo == pytest.approx(levels(r)[("p1", "t14")])


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

    from flextool.engine.solve._run_model import _upper_level_interp

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
