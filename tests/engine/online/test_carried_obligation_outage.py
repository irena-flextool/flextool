"""Obligations carried in from an earlier solve yield to an outage that solve did not see (fix (a), W-0592).

B = {(p, d, t) : min_load > 0 and availability · κ < min_load} (``_mintime.online_blocked_cells``, κ the summed
capacity_max_coeff of the unit's output measure): there min load (≥ min_load · online) and the online capacity
(≤ κ · availability · online) leave only online = 0.  (a1) the carried minimum up-time ends at the first step of B it
reaches and does not resume; (a2) a realization pin > 0 on B drops all pins of that unit; (a3) the carried ramp-down
of a unit without online variables asks at most its output ceiling.  Inside a solve an outage it sees is obeyed.

Hand-derived oracles (``tests.json``: ``coal_plant`` one 500 MW unit, ``dispatch_fullYear_roll`` cut to p2020):

* T-a1u: chain t21..t28 (1 h), one start-up with hours_to_end 2, L = 8: the carry covers elapsed + 2 < 8, so
  t21..t26 (elapsed 0..5); with t25 in B only t21..t24, also where t26 is available again.
* T-a1 oA: horizon = jump = 4 h, min_uptime 6 h, outage t0021-24.  Roll 4 (t17-20) sees no outage, so it equals the
  run without one: a start at t0019.  The carry (t19 + 6 h → t0024) meets the outage at t0021: roll 5 has the unit
  offline over t0021-24.  oB: horizon 8 h, jump 4 h, min_uptime 8 h, outage t0025-30.  Roll 4 (t17-24) equals the
  run without the outage; the start-ups of t19 / t20 carry to t0026 / t0027, cut at t0025: roll 5 keeps the
  carried units online at t0021-24 (their sum, roll 4's start-ups) and 0 at t0025-28.  Today both roll 5s are
  infeasible.
* T-a2: two ``integer_fix_in_uc`` peakers of 100 MW (10 and 20 €/MWh, min_load 0.5), demand 150 MW at t1-t3,
  balance penalties 10 000 €/MWh.  Every forecast sees both available: cheap 100 + dear 50, both online (1, 1, 1).
  The realized branch has the cheap one unavailable at t2: its pin 1 there is infeasible (today), so it loses all
  its pins and is decided again: online (1, 0, 1); the dear one keeps its three pins (1, 1, 1).
* T-a3: no online variables, ramp_speed_down 0.002 / min (ρ̄ = 0.12 of capacity per 1 h step), horizon = jump =
  4 h, outage t0021-24.  Roll 4 ends at 0.332 (as without the outage); the carried down row would ask flow(t0021) ≥
  0.332 − 0.12 = 0.212 at availability 0.  Capped at min(0.332, 0 + 0.12): flow ≥ 0, so flow(t0021) = 0.
* T-a4: one W-0592 line per run, naming coal_plant (the cut also repeats in the next roll).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param

from tests._support.coded_warnings import capture_coded
from tests.engine._unit_db import TIMESTEPS, b64, make_db, utf8

FLEXTOOL_ROOT = Path(__file__).resolve().parents[3]
T0_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_t0_commit.json"
ROLL = "dispatch_fullYear_roll"
PLANT = "coal_plant"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ─── T-a1u: the carry cut on a hand-built chain ─────────────────────────────


def _chain(steps: list[tuple[str, str]], links: list[tuple[tuple[str, str], tuple[str, str]]]):
    dt = pl.DataFrame({"d": [d for d, _ in steps], "t": [t for _, t in steps], "step_duration": [1.0] * len(steps)})
    link = pl.DataFrame({"d": [a[0] for a, _ in links], "t": [a[1] for a, _ in links],
                         "d_previous": [b[0] for _, b in links], "t_previous": [b[1] for _, b in links]})
    return dt, link


def _carry(dt, link, roots, blocked, cut=None):
    from flextool.engine.preprocess.params._mintime import mintime_roll_carry

    carry = pl.DataFrame({"d": [d for d, _ in roots], "t": [t for _, t in roots],
                          "d_previous": ["p"] * len(roots), "t_previous": ["t20"] * len(roots)})
    # A start-up at t19 (2 h from its start to the end of the carried step t20).
    history = pl.DataFrame({"p": ["u", "u"], "d": ["p", "p"], "t": ["t19", "t20"], "startup": [1.0, 0.0],
                            "shutdown": [0.0, 0.0], "duration": [1.0, 1.0], "hours_to_end": [2.0, 1.0]})
    b = None if blocked is None else pl.DataFrame({"p": ["u"] * len(blocked), "d": [d for d, _ in blocked],
                                                     "t": [t for _, t in blocked]})
    out = mintime_roll_carry(dt, None, link, carry, history, [("u", 8.0)], "startup", blocked=b, cut=cut)
    return [] if out is None else [(r["d"], r["t"]) for r in out.sort("d", "t").iter_rows(named=True)]


def test_t_a1u_carry_ends_at_the_first_blocked_step() -> None:
    steps = [("p", f"t{i}") for i in range(21, 29)]
    dt, link = _chain(steps, list(zip(steps[1:], steps[:-1])))
    assert _carry(dt, link, steps[:1], None) == steps[:6]
    cut: list = []
    assert _carry(dt, link, steps[:1], [("p", "t25")], cut) == steps[:4]
    assert cut == [("u", "p", "t25")]
    # A later blocked step changes nothing more; the cut does not resume at t26 although it is available.
    assert _carry(dt, link, steps[:1], [("p", "t25"), ("p", "t27")]) == steps[:4]


def test_t_a1u_cut_follows_its_branch() -> None:
    """Two forecast branches a, b from one root step: an outage on branch a ends only a's carry."""
    a = [("a", f"t{i}") for i in range(21, 25)]
    b = [("b", f"t{i}") for i in range(21, 25)]
    dt, link = _chain(a + b, list(zip(a[1:], a[:-1])) + list(zip(b[1:], b[:-1])))
    got = _carry(dt, link, [a[0], b[0]], [("a", "t23")])
    assert got == a[:2] + b


# ─── T-a1b: the blocked cells ───────────────────────────────────────────────


def test_t_a1b_blocked_cells_boundary() -> None:
    """u: min_load 0.5, availability 0.4 at t1 (blocked: no unit of it can run at 0.4 < 0.5, however many units
    it has), 0.6 at t2 (not); z: availability 0 and min_load 0 (online at zero output is feasible: not blocked);
    k: capacity_max_coeff 0.5 and availability 0.9, 0.45 < min_load 0.5 (blocked); w: availability not authored
    (1), min_load 0.5 (not)."""
    from flextool.engine.preprocess.params._mintime import online_blocked_cells

    units = ["u", "z", "k", "w"]
    dt = pl.DataFrame({"d": ["p"] * 2, "t": ["t1", "t2"]})
    cells = [(p, t) for p in units for t in ("t1", "t2")]
    ml = {"u": 0.5, "z": 0.0, "k": 0.5, "w": 0.5}
    av = {("u", "t1"): 0.4, ("u", "t2"): 0.6, ("z", "t1"): 0.0, ("z", "t2"): 0.0, ("k", "t1"): 0.9, ("k", "t2"): 1.0}
    d = SimpleNamespace(
        dt=dt,
        p_min_load=Param(("p", "d", "t"), pl.DataFrame({"p": [p for p, _ in cells], "d": ["p"] * len(cells),
                                                         "t": [t for _, t in cells],
                                                         "value": [ml[p] for p, _ in cells]})),
        p_process_availability=Param(("p", "d", "t"), pl.DataFrame(
            {"p": [p for p, _ in av], "d": ["p"] * len(av), "t": [t for _, t in av], "value": list(av.values())})),
        p_arc_max_cap_coef=Param(("p", "source", "sink"),
                                 pl.DataFrame({"p": ["k"], "source": ["k"], "sink": ["n"], "value": [0.5]})),
        process_source_sink=pl.DataFrame({"p": units, "source": units, "sink": ["n"] * 4}),
        process_unit=pl.DataFrame({"p": units}), process_minload=None, process_indirect=None,
        process_source_sink_uncapped=None)
    got = online_blocked_cells(d)
    assert got.rows() == [("k", "p", "t1"), ("u", "p", "t1")]


# ─── T-a1 / T-a3 / T-a4: rolling single-stage toys ──────────────────────────


def _roll_values(horizon: float, jump: float, duration: float, outage: list[str]) -> list:
    vals = [["solve", ROLL, "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", "fullYear"]]}, "map")],
            ["solve", ROLL, "realized_periods", b64({"value_type": "str", "data": ["p2020"]}, "array")],
            ["solve", ROLL, "rolling_solve_horizon", b64(horizon, "float")],
            ["solve", ROLL, "rolling_solve_jump", b64(jump, "float")],
            ["solve", ROLL, "rolling_duration", b64(duration, "float")]]
    if outage:
        vals.append(["unit", PLANT, "availability", b64(
            {"index_type": "str", "index_name": "time",
             "data": [[t, 0.0 if t in outage else 1.0] for t in TIMESTEPS]}, "map")])
    return vals


def _rolls(tmp_path_factory, alt: str, values: list, chain: list[str]):
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain)
    with capture_coded() as got:
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)
    assert all(s.solution.optimal for s in steps.values())
    return list(steps.values()), got


def _series(step, var: str, kind: str = "") -> dict[str, float]:
    df = utf8(step.solution.value(f"{var}{kind}")).filter(pl.col("p") == PLANT)
    if "sink" in df.columns:
        df = df.filter(pl.col("sink") == "west")
    return {t: round(v, 6) + 0.0 for t, v in zip(df["t"], df["value"])}


UC_CHAIN = ["init", "west", "coal", "coal_min_load", "coal_min_updowntime", "wind", "fullYear", ROLL]
OUT_A = ["t0021", "t0022", "t0023", "t0024"]
OUT_B = ["t0025", "t0026", "t0027", "t0028", "t0029", "t0030"]


@pytest.mark.solver
@pytest.mark.parametrize("toy, startup", [("oA", "integer_fix_in_uc"), ("oB", "integer_fix_in_uc"),
                                          ("oB", "linear")])
def test_t_a1_carried_min_uptime_ends_at_the_outage(tmp_path_factory, toy, startup) -> None:
    horizon, uptime, outage = (4.0, 6.0, OUT_A) if toy == "oA" else (8.0, 8.0, OUT_B)
    kind = "_integer" if startup.startswith("integer") else "_linear"
    unit = [["unit", PLANT, "min_uptime", b64(uptime, "float")],
            ["unit", PLANT, "min_downtime", b64(1.0, "float")],
            ["unit", PLANT, "startup_method", b64(startup, "str")]]
    rolls, got = _rolls(tmp_path_factory, f"{toy}_{startup}", unit + _roll_values(horizon, 4.0, 32.0, outage),
                        UC_CHAIN)
    # Without the outage, cut where roll 4's whole horizon still fits.
    base, _ = _rolls(tmp_path_factory, f"{toy}_{startup}_no_outage",
                     unit + _roll_values(horizon, 4.0, 16.0 + horizon, []), UC_CHAIN)
    assert len(rolls) == 8 and len(base) >= 5
    # Roll 4 sees no outage: the same window data and carry as the run without one.
    for var in ("v_online", "v_startup"):
        assert _series(rolls[4], var, kind) == pytest.approx(_series(base[4], var, kind), abs=1e-6), var
    online5 = _series(rolls[5], "v_online", kind)
    if toy == "oA":
        assert _series(rolls[4], "v_startup", kind)["t0019"] == pytest.approx(1.0)
        assert [online5[t] for t in OUT_A] == [0.0] * 4
    else:
        su4 = _series(rolls[4], "v_startup", kind)
        started = su4["t0019"] + su4["t0020"]   # realized in roll 4's jump (t17-20), both within 8 h of t0024
        assert started > 0.0 and sum(su4[t] for t in ("t0017", "t0018")) == 0.0
        assert [online5[t] for t in OUT_A] == pytest.approx([started] * 4, abs=1e-6)
        assert [online5[t] for t in ("t0025", "t0026", "t0027", "t0028")] == [0.0] * 4
    # T-a4: one note for the run, naming the unit (the next roll's carry is cut again, without a new line).
    assert got.names("W-0592") == [PLANT]
    assert sum("[W-0592]" in line for line in got.lines()) == 1
    assert "carried minimum up-time ends at step (p2020, t00" in got.detail("W-0592", PLANT)


@pytest.mark.solver
def test_t_a3_carried_ramp_down_yields_to_the_outage(tmp_path_factory) -> None:
    values = _roll_values(4.0, 4.0, 28.0, OUT_A)
    rolls, got = _rolls_with_ramp(tmp_path_factory, "ramp_outage", values)
    assert len(rolls) == 7
    assert _series(rolls[4], "v_flow")["t0020"] == pytest.approx(0.332, abs=1e-9)
    assert _series(rolls[5], "v_flow")["t0021"] == 0.0
    assert got.names("W-0592") == [PLANT]
    assert "carried ramp-down ends at step (p2020, t0021)" in got.detail("W-0592", PLANT)


def _rolls_with_ramp(tmp_path_factory, alt: str, values: list):
    """The roll toy with a ramp-down limit on coal_plant → west (no online variables)."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db
    from tests.engine._unit_db import BASE_FIXTURE_JSON, add_override_alternative

    data = json.loads(BASE_FIXTURE_JSON.read_text(encoding="utf-8"))
    for k, v in (("ramp_method", b64("ramp_limit", "str")), ("ramp_speed_up", b64(1.0, "float")),
                 ("ramp_speed_down", b64(0.002, "float"))):
        data["parameter_values"].append(["unit__outputNode", [PLANT, "west"], k, v, alt])
    scen = add_override_alternative(data, alt, values, ["init", "west", "coal", "wind", "fullYear", ROLL])
    work = tmp_path_factory.mktemp(f"{alt}_db")
    (work / "t.json").write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(work / "t.json", work / "t.sqlite")
    migrate_database(url)
    with capture_coded() as got:
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)
    assert all(s.solution.optimal for s in steps.values())
    return list(steps.values()), got


# ─── T-a2: the realization pin yields to a realized-only outage ─────────────

PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"


def _stoch_series(per_branch: dict[str, list[float]]) -> list:
    return b64({"index_type": "str", "data": [
        [br, {"type": "map", "index_type": "str", "data": [["t0001", {"type": "map", "index_type": "str", "data": [
            [f"t000{i + 1}", v] for i, v in enumerate(vals)]}]]}] for br, vals in per_branch.items()]}, "map")


def _two_peakers(data: dict) -> None:
    """``stoch_t0_commit`` reshaped as the T-a2 model (module docstring)."""
    wind = {"wind", "wprof"}
    data["entities"] = [e for e in data["entities"]
                        if not (set(e[1] if isinstance(e[1], list) else [e[1]]) & wind)]
    # The cheap peaker reads its availability per branch (stochastic group stoch_g).
    data["entities"] += [["unit", "dear", None], ["unit__outputNode", ["dear", "elec"], None],
                         ["group__unit", ["stoch_g", "peaker"], None]]
    data["entity_alternatives"] = [e for e in data.get("entity_alternatives", []) if not (set(e[1]) & wind)]
    data["entity_alternatives"] += [["unit", ["dear"], "base", True]]
    drop = {"invest_method", "invest_cost", "invest_max_total", "lifetime_technical", "discount_rate",
            "startup_cost", "existing", "virtual_unitsize", "startup_method", "min_uptime", "min_downtime",
            "minimum_time_method", "efficiency", "min_load"}
    keep = [r for r in data["parameter_values"]
            if not (set(r[1] if isinstance(r[1], list) else [r[1]]) & wind) and not (r[1] == "peaker" and r[2] in drop)]
    for r in keep:
        if r[1] == "elec" and r[2] == "inflow":
            r[3] = b64({"index_type": "str", "data": [["t0001", -150.0], ["t0002", -150.0], ["t0003", -150.0]]},
                       "map")
    for u, cost in (("peaker", 10.0), ("dear", 20.0)):
        keep += [["unit", u, "existing", b64(1.0, "float"), "base"],
                 ["unit", u, "virtual_unitsize", b64(100.0, "float"), "base"],
                 ["unit", u, "efficiency", b64(1.0, "float"), "base"],
                 ["unit", u, "min_load", b64(0.5, "float"), "base"],
                 ["unit", u, "startup_method", b64("integer_fix_in_uc", "str"), "base"]]
        if u == "dear":
            keep.append(["unit__outputNode", ["dear", "elec"], "other_operational_cost", b64(cost, "float"), "base"])
    keep.append(["unit", "peaker", "availability",
                 _stoch_series({"rlz": [1.0, 0.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0]}), "stoch"])
    data["parameter_values"] = keep


@pytest.mark.solver
def test_t_a2_pin_yields_to_a_realized_outage(tmp_path) -> None:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads(T0_FIXTURE.read_text(encoding="utf-8"))
    _two_peakers(data)
    (tmp_path / "t.json").write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(tmp_path / "t.json", tmp_path / "t.sqlite")
    migrate_database(url)
    with capture_coded() as got:
        steps = run_chain_from_db(url, "stoch_t0_commit", work_folder=tmp_path / "work", warm=False,
                                  keep_solutions=True)
    assert list(steps) == [PARENT, SIBLING]
    assert all(s.solution.optimal for s in steps.values())

    def online(step, d: str) -> dict:
        df = utf8(step.solution.value("v_online_integer")).filter(pl.col("d") == d)
        return {p: [round(v, 9) + 0.0 for v in g.sort("t")["value"]] for (p,), g in df.group_by("p")}

    for d in ("p2035_high", "p2035_low"):
        assert online(steps[PARENT], d) == {"peaker": [1.0, 1.0, 1.0], "dear": [1.0, 1.0, 1.0]}, d
    assert online(steps[SIBLING], "p2035") == {"peaker": [1.0, 0.0, 1.0], "dear": [1.0, 1.0, 1.0]}
    pins = utf8(steps[SIBLING].flex_data.p_fix_online.frame)
    assert sorted(set(pins["p"])) == ["dear"] and pins.height == 3
    assert got.names("W-0592") == ["peaker"]
