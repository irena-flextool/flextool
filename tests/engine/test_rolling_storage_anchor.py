"""Rolling bind_within_solve storage: carry, horizon-end anchor and closure; a nested level's borrowed start.

A tiny model built from the schema JSON (never a checked-in ``.sqlite``).  Timeline ``tl``: t01..t16, 1 h steps.
Storage node ``tank`` (100 MWh, ``bind_within_solve``, ``fix_start`` 0.5 = 50 MWh, no loss), feeding the balance
node ``load`` (5 MW in t05..t08) through ``dis`` (100 MW, efficiency 1).  ``gen`` charges the tank (10 MW, cost
1 CUR/MWh) in t09..t12 only (``availability``).  Penalties are 10 000 CUR/MWh (the tank's own slacks 20 000).

The rolling solve ``st``: jump 4 h, horizon 8 h, duration 16 h, so the rolls are t01-08, t05-12, t09-16 and
t13-16; the last two reach the end of the solve (closing rolls).  Expected (design §8 test 2):

* roll 0 cycles on itself (start = end = 50): it cannot refill before t08, so it leaves the look-ahead's load
  unserved; its realised t04 level is 50;
* roll 1 continues from 50 and serves the load from the tank (t08: 30); the anchor ends its horizon at ≥ 50, so
  it buys 20 MWh in t09-12;
* roll 2 (closing) continues from 30 and closes at 50 (the level roll 0 started from): it buys 20 MWh;
* roll 3 (closing) continues from 50 and ends at 50.

Before, every roll cycled on itself from a free level: roll 1 could start at 70 with no carry.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from tests.engine._lp_rows import row_bounds, row_terms
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


def rolling(**extra) -> dict:
    return {"solve_mode": "rolling_window", "rolling_solve_jump": 4.0, "rolling_solve_horizon": 8.0,
            "rolling_duration": 16.0, "period_timeset": tmap({"p1": "full"}, "period"),
            "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"]), **extra}


def build_db(tmp_dir: Path, solves: dict[str, dict], *, top: list[str], tank: dict | None = None,
             periods: tuple[str, ...] = ("p1",)) -> str:
    """The database (url) of the tiny model: ``solves`` ``{solve: {parameter: value}}``, ``top`` the model's
    solves, ``tank`` extra parameters of the tank (they replace the defaults)."""
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    db_path = Path(tmp_dir) / "tank.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db_path))
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    entities = [("model", "flextool"), ("timeline", "tl"), ("timeset", "full"), ("timeset", "coarse"),
                ("node", "tank"), ("node", "load"),
                ("unit", "dis"), ("unit", "gen"), ("unit__inputNode", ("dis", "tank")),
                ("unit__outputNode", ("dis", "load")), ("unit__outputNode", ("gen", "tank"))]
    entities += [("solve", s) for s in solves]
    tank_values = {"node_type": "storage", "existing": 100.0, "storage_binding_method": "bind_within_solve",
                   "storage_start_end_method": "fix_start", "storage_state_start": 0.5,
                   "penalty_up": 20000.0, "penalty_down": 20000.0}
    tank_values.update(tank or {})
    values = [
        ("model", "flextool", "solves", arr(top)),
        ("timeline", "tl", "timestep_duration", tmap({t: 1.0 for t in STEPS})),
        ("timeset", "full", "timeline", "tl"),
        ("timeset", "full", "timeset_duration", tmap({"t01": 16.0})),
        ("timeset", "coarse", "timeline", "tl"),
        ("timeset", "coarse", "timeset_duration", tmap({"t01": 16.0})),
        ("node", "load", "node_type", "balance"),
        ("node", "load", "inflow", series({t: -5.0 for t in ("t05", "t06", "t07", "t08")}, 0.0)),
        ("node", "load", "penalty_up", 10000.0),
        ("node", "load", "penalty_down", 10000.0),
        ("unit", "dis", "existing", 100.0),
        ("unit", "dis", "efficiency", 1.0),
        ("unit", "gen", "existing", 10.0),
        ("unit", "gen", "efficiency", 1.0),
        ("unit", "gen", "availability", series({t: 1.0 for t in ("t09", "t10", "t11", "t12")}, 0.0)),
        ("unit__outputNode", ("gen", "tank"), "other_operational_cost", 1.0),
    ] + [("node", "tank", k, v) for k, v in tank_values.items()]
    for solve, params in solves.items():
        for par, val in params.items():
            values.append(("solve", solve, par, val))
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


def run(tmp_path, solves: dict, *, top: list[str] | None = None, tank: dict | None = None) -> dict:
    from flextool.engine import run_chain_from_db

    tmp_path.mkdir(parents=True, exist_ok=True)
    url = build_db(tmp_path, solves, top=top or list(solves), tank=tank)
    steps = run_chain_from_db(url, SCENARIO, work_folder=tmp_path / "work", keep_solutions=True)
    for step in steps.values():
        assert step.solution is not None
    return steps


def state(step, node: str = "tank") -> dict:
    """``{step: MWh}`` of the node's v_state in a solved roll."""
    su = float(utf8(step.flex_data.p_state_unitsize.frame).filter(pl.col("n") == node)["value"][0])
    rows = utf8(step.solution.value("v_state")).filter(pl.col("n") == node).sort("d", "t")
    return {t: v * su for t, v in zip(rows["t"].to_list(), rows["value"].to_list())}


def gen_mwh(step, steps: tuple[str, ...]) -> float:
    rows = utf8(step.solution.value("v_flow")).filter((pl.col("p") == "gen") & pl.col("t").is_in(list(steps)))
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == "gen")["value"][0])
    return float(rows["value"].sum()) * us


def rolls(steps: dict, solve: str = "st") -> list:
    """The rolls of ``solve`` in roll order (``st_roll_<k>``; nested in a solve of period p1: ``st_p1_roll_<k>``)."""
    out = [s for name, s in steps.items() if name.startswith(f"{solve}_") and "_roll_" in name]
    return sorted(out, key=lambda s: int(s.solve_name.rsplit("_", 1)[1]))


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    return rolls(run(tmp_path_factory.mktemp("plain"), {"st": rolling()}))


def test_rolls_and_closing_flags(plain):
    """Four rolls; rows: anchor in roll 1, closure in rolls 2 and 3 (both reach t16), none in roll 0 (its wrap)."""
    assert len(plain) == 4
    kinds = [None if s.flex_data.storage_roll_end is None else
             utf8(s.flex_data.storage_roll_end).filter(pl.col("n") == "tank")["kind"].to_list() for s in plain]
    assert kinds == [None, ["anchor"], ["close"], ["close"]]
    pb0 = build(plain[0].flex_data)
    assert not row_terms(pb0, ANCHOR, "tank") and not row_terms(pb0, CLOSE, "tank")


def test_anchor_binds_hand_checked(plain):
    """Roll 1 continues from 50, serves the load (t08: 30) and its anchor ends t12 at 50: it buys 20 MWh."""
    r1 = plain[1]
    fd = r1.flex_data
    carry = float(utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == "tank")["value"][0])
    assert carry == pytest.approx(50.0, abs=1e-6)
    lo, hi = row_bounds(build(fd), ANCHOR, "tank")[f"{ANCHOR}[tank,p1,t12]"]
    assert lo == pytest.approx(50.0) and hi == float("inf")
    s1 = state(r1)
    assert s1["t08"] == pytest.approx(30.0, abs=1e-6)
    assert s1["t12"] == pytest.approx(50.0, abs=1e-6)
    assert gen_mwh(r1, ("t09", "t10", "t11", "t12")) == pytest.approx(20.0, abs=1e-6)


def test_two_closing_rolls_close_to_the_start(plain):
    """Rolls 2 and 3 both reach t16 and close to 50 MWh (roll 0's fix_start pin), whatever their carry."""
    for r in plain[2:]:
        assert row_bounds(build(r.flex_data), CLOSE, "tank") == {
            f"{CLOSE}[tank,p1,t16]": (pytest.approx(50.0), pytest.approx(50.0))}
        assert state(r)["t16"] == pytest.approx(50.0, abs=1e-6)
    assert state(plain[2])["t12"] == pytest.approx(50.0, abs=1e-6)
    assert gen_mwh(plain[2], ("t09", "t10", "t11", "t12")) == pytest.approx(20.0, abs=1e-6)


def test_realised_levels_continue_and_balance(plain):
    """The realised level is continuous across every roll boundary (each roll starts from the previous roll's
    realised end) and the realised hours balance: Σ supply = Σ demand + (final − initial) = 20 + 0."""
    ends = {0: "t04", 1: "t08", 2: "t12", 3: "t16"}
    realised = [state(r)[ends[k]] for k, r in enumerate(plain)]
    assert realised == pytest.approx([50.0, 30.0, 50.0, 50.0], abs=1e-6)
    for prev, nxt in zip(realised, plain[1:]):
        carry = float(utf8(nxt.flex_data.p_roll_continue_state.frame).filter(pl.col("n") == "tank")["value"][0])
        assert carry == pytest.approx(prev, abs=1e-6)
    supply = sum(gen_mwh(r, tuple(STEPS[4 * k: 4 * k + 4])) for k, r in enumerate(plain))
    assert supply == pytest.approx(20.0 + (realised[-1] - 50.0), abs=1e-6)


def test_single_roll_equals_single_solve(tmp_path):
    """Horizon = duration: one roll, no carry, the same objective as the solve that is not rolling."""
    one = rolls(run(tmp_path / "one", {"st": rolling(rolling_solve_jump=16.0, rolling_solve_horizon=16.0)}))
    single = list(run(tmp_path / "single", {"st": {k: v for k, v in rolling().items()
                                                   if not k.startswith("rolling")} | {"solve_mode": "single_solve"}})
                  .values())[-1]
    assert len(one) == 1 and one[0].flex_data.storage_roll_end is None
    assert float(one[0].solution.obj) == pytest.approx(float(single.solution.obj), rel=1e-9)


def test_self_discharge_anchor_is_reachable(tmp_path):
    """With a loss the anchor is the lower of the carry and the level the tank decays to by doing nothing: a loss of
    1 %/h decays the content by 1/1.01 per hour (the balance rows' continuous decay), eight hours to t12."""
    r = rolls(run(tmp_path, {"st": rolling()}, tank={"self_discharge_loss": 0.01}))
    fd = r[1].flex_data
    carry = float(utf8(fd.p_roll_continue_state.frame).filter(pl.col("n") == "tank")["value"][0])
    lo, _hi = row_bounds(build(fd), ANCHOR, "tank")[f"{ANCHOR}[tank,p1,t12]"]
    assert lo == pytest.approx(carry / 1.01 ** 8, rel=1e-9)


def test_multi_period_capacity_change_gives_no_free_energy(tmp_path):
    """Two periods (p1 100 MWh, p2 60 MWh): the closing rolls close to the MWh roll 0 started from (50), not to the
    share of p2's capacity (30), so the realised year returns what it took."""
    solve = rolling(period_timeset=tmap({"p1": "full", "p2": "full"}, "period"),
                    years_represented=tmap({"p1": 1.0, "p2": 1.0}, "period"), realized_periods=arr(["p1", "p2"]),
                    rolling_duration=32.0)
    r = rolls(run(tmp_path, {"st": solve}, tank={"existing": tmap({"p1": 100.0, "p2": 60.0}, "period")}))
    closing = [x for x in r if x.flex_data.storage_roll_end is not None
               and "close" in utf8(x.flex_data.storage_roll_end)["kind"].to_list()]
    assert closing, "no closing roll"
    for x in closing:
        (key, (lo, hi)), = row_bounds(build(x.flex_data), CLOSE, "tank").items()
        assert key.endswith(",p2,t16]") and lo == pytest.approx(50.0) and hi == pytest.approx(50.0)
    last = state(r[-1])
    assert last[max(last)] == pytest.approx(50.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Nested: an upper storage solve ``up`` (4 h steps, timeset ``coarse``) fixes the tank of the rolling ``st``
# ---------------------------------------------------------------------------


def nested(method: str, **tank) -> tuple[dict, dict]:
    up = {"solve_mode": "single_solve", "new_stepduration": 4.0, "period_timeset": tmap({"p1": "coarse"}, "period"),
          "years_represented": tmap({"p1": 1.0}, "period"), "realized_periods": arr(["p1"]),
          "fix_storage_periods": arr(["p1"]), "contains_solves": arr(["st"])}
    return {"up": up, "st": rolling()}, {"storage_nested_fix_method": method, **tank}


def test_nested_fix_quantity_first_roll_starts_from_fix_start(tmp_path):
    """Under fix_quantity the first st roll starts from fix_start (50 MWh, chained forward: its wrap target would
    be the upper target at t08) and every roll's horizon end follows the upper target, with no anchor or closure."""
    solves, tank = nested("fix_quantity")
    steps = run(tmp_path, solves, top=["up"], tank=tank)
    r = rolls(steps)
    fd0 = r[0].flex_data
    assert "tank" in utf8(fd0.storage_bind_forward_only)["n"].to_list()
    pb0 = build(fd0)
    assert not row_terms(pb0, "storage_state_start_binding_cyclic_period", "tank")
    assert set(row_terms(pb0, "node_balance_fix_quantity_eq_lower", "tank")) == {
        "node_balance_fix_quantity_eq_lower[tank,p1,t08]"}
    up_state = state(steps["up"])
    for x in r:
        pb = build(x.flex_data)
        assert not row_terms(pb, ANCHOR, "tank") and not row_terms(pb, CLOSE, "tank")
        fixed = row_bounds(pb, "node_balance_fix_quantity_eq_lower", "tank")
        assert len(fixed) == 1
    # the realised year starts at 50 and ends at the upper plan's end
    assert state(r[-1])["t16"] == pytest.approx(up_state[max(up_state)], abs=1e-6)


def test_nested_fix_quantity_free_start_borrows_the_upper_start(tmp_path):
    """Without fix_start the first st roll starts from the level the upper solve starts from (its cycle level),
    not from a free level: borrowed end ⇒ borrowed start."""
    solves, tank = nested("fix_quantity", storage_start_end_method="fix_nothing")
    steps = run(tmp_path, solves, top=["up"], tank=tank)
    r = rolls(steps)
    upper = steps["up"]
    up_start = state(upper)[max(state(upper))]          # bind_within_solve: the first step wraps to the last
    fd0 = r[0].flex_data
    carry = float(utf8(fd0.p_roll_continue_state.frame).filter(pl.col("n") == "tank")["value"][0])
    assert carry == pytest.approx(up_start, abs=1e-6)
    assert state(r[-1])["t16"] == pytest.approx(up_start, abs=1e-6)


def test_nested_fix_price_keeps_the_closure(tmp_path):
    """A nested fix_price values the roll ends but does not replace the closure: the closing rolls still end at
    the start level; at a non-closing roll end the price replaces the anchor."""
    solves, tank = nested("fix_price", storage_state_reference_price=2.0)
    r = rolls(run(tmp_path, solves, top=["up"], tank=tank))
    for x in r[2:]:
        assert row_bounds(build(x.flex_data), CLOSE, "tank") == {
            f"{CLOSE}[tank,p1,t16]": (pytest.approx(50.0), pytest.approx(50.0))}
        assert state(x)["t16"] == pytest.approx(50.0, abs=1e-6)
    assert not row_terms(build(r[1].flex_data), ANCHOR, "tank")


def test_notice_names_the_anchored_node_with_the_marker(tmp_path, coded_log):
    """With the marker the run prints the rolling_cyclic_storage_anchored notice once, naming the tank; without it
    (a new database) nothing."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping

    url = build_db(tmp_path, {"st": rolling()}, top=["st"])
    run_chain_from_db(url, SCENARIO, work_folder=tmp_path / "w0")
    assert "rolling_cyclic_storage_anchored" not in coded_log()
    with open_database_mapping(url) as db:
        db.add_item("metadata", name="flextool_behaviour_change:rolling_cyclic_storage_anchored",
                    value='{"since": 82, "nodes": ["tank"]}')
        db.commit_session("marker")
    run_chain_from_db(url, SCENARIO, work_folder=tmp_path / "w1")
    lines = [ln for ln in coded_log().splitlines() if "rolling_cyclic_storage_anchored" in ln]
    assert len(lines) == 1 and "Affected nodes: tank." in lines[0]
