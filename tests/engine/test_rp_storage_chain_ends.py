"""The absolute storage level at the ends of a representative-period chain, and the matching end of forward chains.

In a representative-period solve a ``*_blended_weights`` node has the base-period chain v_state_inter (the
chronological level at the end of each base period) and block-internal states that start from a free offset
(v_state_rp_start).  storage_state_start / storage_state_end / use_reference_value now act on the chain (the level the
first base period starts from, the level at the end of the last one), the chain start of a forward chain is a level
within the capacity, and a forward chain whose start the solve decides ends at least at that start
(``rp_inter_period_end_match``; ``storage_state_end_match`` without representative periods).

Tiny model (built from the schema JSON, never a checked-in sqlite): storage node ``s`` (virtual_unitsize 1, so the
states are in MWh) with a constant inflow of −1 MW (a demand on the store), unit ``src`` → ``s`` (10 MW, efficiency 1,
other_operational_cost 10 per MWh), four 1-h steps.  ``rp``: two blocks t0001-t0002 and t0003-t0004, each the exact
representative of its own base period (weight 1), so the chain of two base periods is the chronological year;
``hourly``: the same four steps without representative periods.  The demand is 4 MWh, so every MWh the store
delivers net (start − end) is a MWh ``src`` does not produce.

The fuel tank (owner's lt_rp finding): storage that the solve can invest in, bind_forward_only_blended_weights,
fix_start 0.5 and a free end.  Before, the chain start and end were free whatever fix_start said (fix_start pinned a
block-internal offset), so each MWh of storage built delivered up to two MWh once and the solve bought storage
instead of production (here 2 MWh of storage; on the hourly timeset, where fix_start did pin the start, 8 MWh, whose
0.5 share is the whole demand).  Now the chain start is pinned at 0.5 × capacity and the end returns at least the
start on the new capacity: no storage, production 4.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build

REPO = Path(__file__).resolve().parents[2]
SCHEMA_JSON = REPO / "flextool" / "schemas" / "spinedb_schema.json"
STEPS = ["t0001", "t0002", "t0003", "t0004"]
FO_B, WS_B = "bind_forward_only_blended_weights", "bind_within_solve_blended_weights"
TOL = 1e-6


def _m(data: dict, name: str) -> api.Map:
    return api.Map(list(data), list(data.values()), index_name=name)


def _run(tmp_path_factory, *, s: dict, rp: bool = True, invest: bool = False, mixed: bool = False):
    """Build, solve and return the step of the tiny model; ``s`` = extra parameters of node ``s``.  ``mixed``: the
    solve also holds period p2 on timeset ``plain`` (the four steps, no representative periods)."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp("rp_chain_ends")
    db_path = work / "tiny.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db_path))
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    entities = [("model", "flextool"), ("timeline", "tl"), ("timeset", "ts"), ("node", "s"), ("unit", "src"),
                ("unit__outputNode", ("src", "s")), ("solve", "sol")]
    values = [
        ("model", "flextool", "solves", api.Array(["sol"])),
        ("timeline", "tl", "timestep_duration", _m({t: 1.0 for t in STEPS}, "time")),
        ("timeset", "ts", "timeline", "tl"),
        ("timeset", "ts", "timeset_duration", _m({"t0001": 2.0, "t0003": 2.0}, "time")),
        ("node", "s", "node_type", "storage"),
        ("node", "s", "virtual_unitsize", 1.0),
        ("node", "s", "inflow", -1.0),
        ("node", "s", "penalty_up", 10000.0),
        ("node", "s", "penalty_down", 10000.0),
        ("unit", "src", "existing", 10.0),
        ("unit", "src", "virtual_unitsize", 1.0),
        ("unit", "src", "efficiency", 1.0),
        ("unit__outputNode", ("src", "s"), "other_operational_cost", 10.0),
        ("solve", "sol", "solve_mode", "single_solve"),
        ("solve", "sol", "period_timeset", _m({"p1": "ts"}, "period")),
        ("solve", "sol", "realized_periods", api.Array(["p1"])),
        ("solve", "sol", "years_represented", _m({"p1": 1.0}, "period")),
    ]
    if rp:
        values.append(("timeset", "ts", "representative_period_weights",
                       _m({b: _m({b: 1.0}, "representative_period") for b in ("t0001", "t0003")}, "base_period")))
    if invest:
        values += [("solve", "sol", "invest_periods", api.Array(["p1"])),
                   ("solve", "sol", "realized_invest_periods", api.Array(["p1"])),
                   ("node", "s", "invest_method", "invest_no_limit"),
                   ("node", "s", "invest_cost", 1.0),
                   ("node", "s", "lifetime_technical", 1.0),
                   ("node", "s", "discount_rate", 0.0)]
    if mixed:
        entities.append(("timeset", "plain"))
        values = [v for v in values if not (v[0] == "solve" and v[2] in ("period_timeset", "realized_periods",
                                                                         "years_represented"))]
        values += [("timeset", "plain", "timeline", "tl"),
                   ("timeset", "plain", "timeset_duration", _m({"t0001": 4.0}, "time")),
                   ("solve", "sol", "period_timeset", _m({"p1": "ts", "p2": "plain"}, "period")),
                   ("solve", "sol", "realized_periods", api.Array(["p1", "p2"])),
                   ("solve", "sol", "years_represented", _m({"p1": 1.0, "p2": 1.0}, "period"))]
    values += [("node", "s", k, v) for k, v in s.items()]
    with open_database_mapping(url) as db:
        _count, errors = api.import_data(
            db, alternatives=[("tiny", "")], scenarios=[("tiny_s", False, "")],
            scenario_alternatives=[("tiny_s", "tiny", None)], entities=entities,
            entity_alternatives=[("node", ("s",), "tiny", True), ("unit", ("src",), "tiny", True)],
            parameter_values=[(c, e, p, v, "tiny") for c, e, p, v in values])
        assert not errors, errors[:5]
        db.commit_session("rp chain ends tiny model")
    steps = run_chain_from_db(url, "tiny_s", work_folder=work / "w", keep_solutions=True)
    step = steps["sol"]
    assert step.solution is not None and step.solution.optimal
    return step


def _values(step, var: str) -> dict[tuple, float]:
    frame = step.solution.value(var)
    if frame is None or frame.height == 0:
        return {}
    frame = frame.with_columns([pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value"])
    keys = [c for c in frame.columns if c != "value"]
    return {tuple(r[:-1]): r[-1] for r in frame.select(*keys, "value").iter_rows()}


def _production(step) -> float:
    """MWh src produces over the four steps (each representative step stands for one chronological step)."""
    return sum(v for k, v in _values(step, "v_flow").items() if k[0] == "src")


def _invested(step) -> float:
    return sum(_values(step, "v_invest_n").values())


def _chain(step) -> tuple[float, float]:
    """(level the chain starts from, level at its end): v_inter[b1] − drift[b1], v_inter[b2]."""
    inter = _values(step, "v_state_inter")
    state, start = _values(step, "v_state"), _values(step, "v_state_rp_start")
    drift1 = state[("s", "p1", "t0002")] - start[("s", "p1", "t0001")]
    return inter[("s", "p1", "t0001")] - drift1, inter[("s", "p1", "t0003")]


def _no_slack(step) -> None:
    for var in ("vq_state_up", "vq_state_down"):
        assert all(abs(v) < TOL for v in _values(step, var).values()), var


def test_fuel_tank_is_not_bought(tmp_path_factory):
    """fix_start 0.5 on storage the solve can invest in, forward chain, free end: the start on the new capacity
    must come back, so storage brings no free energy and none is built; src produces the whole demand.  Before:
    2 MWh of storage (see the module docstring)."""
    step = _run(tmp_path_factory, invest=True, s={"existing": 0.0, "storage_binding_method": FO_B,
                                                  "storage_start_end_method": "fix_start",
                                                  "storage_state_start": 0.5})
    assert _invested(step) == pytest.approx(0.0, abs=TOL)
    assert _production(step) == pytest.approx(4.0, abs=TOL)
    _no_slack(step)
    pb = build(step.flex_data)
    assert list(row_terms(pb, "rp_inter_period_start_fix")) == ["rp_inter_period_start_fix[s,p1,t0001]"]
    assert list(row_terms(pb, "rp_inter_period_end_match")) == ["rp_inter_period_end_match[s,p1,t0001,p1,t0003]"]
    assert not row_terms(pb, "storage_state_start_binding")


def test_fuel_tank_is_not_bought_without_representative_periods(tmp_path_factory):
    """The same on the hourly timeset (the blended node is bind_forward_only there): storage_state_end_match_share
    returns the start on the new capacity (before: 8 MWh built)."""
    step = _run(tmp_path_factory, rp=False, invest=True,
                s={"existing": 0.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_start",
                   "storage_state_start": 0.5})
    assert _invested(step) == pytest.approx(0.0, abs=TOL)
    assert _production(step) == pytest.approx(4.0, abs=TOL)
    assert list(row_terms(build(step.flex_data), "storage_state_end_match_share"))


def test_cyclic_blended_fix_start_pins_the_cycle_level(tmp_path_factory):
    """bind_within_solve_blended_weights + fix_start 0.5 on 2 MWh: the chain start (and, closed, its end) is 1 MWh;
    the store returns what it delivers, so production is the demand (as before: the closure already held)."""
    step = _run(tmp_path_factory, s={"existing": 2.0, "storage_binding_method": WS_B,
                                     "storage_start_end_method": "fix_start", "storage_state_start": 0.5})
    start, end = _chain(step)
    assert (start, end) == pytest.approx((1.0, 1.0), abs=TOL)
    assert _production(step) == pytest.approx(4.0, abs=TOL)
    assert not row_terms(build(step.flex_data), "rp_inter_period_end_match")


def test_fix_start_on_existing_capacity_is_an_endowment(tmp_path_factory, capsys):
    """fix_start 0.5 on 2 MWh the solve cannot change, forward chain, free end: the chain starts at 1 MWh (pinned on
    the chain, not inside the first block) and may end empty, the user's explicit endowment (a printed warning with
    its size that calls it a valid initial stock, owner decision; not closed): production 3."""
    step = _run(tmp_path_factory, s={"existing": 2.0, "storage_binding_method": FO_B,
                                     "storage_start_end_method": "fix_start", "storage_state_start": 0.5})
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "storage_state_start on capacity the solve cannot change" in ln]
    assert len(lines) == 1 and "1 storage node (s)" in lines[0] and "(1 MWh in total)" in lines[0]
    assert "endowment" in lines[0]
    start, end = _chain(step)
    assert (start, end) == pytest.approx((1.0, 0.0), abs=TOL)
    assert _production(step) == pytest.approx(3.0, abs=TOL)
    assert not row_terms(build(step.flex_data), "rp_inter_period_end_match")


def test_free_start_ends_at_least_at_its_start(tmp_path_factory):
    """fix_nothing, forward chain on 2 MWh: the chain start is a level in [0, 2] and the end is at least the start,
    so the store delivers nothing net: production 4 (before: start 2 or more, end 0, production at most 2)."""
    step = _run(tmp_path_factory, s={"existing": 2.0, "storage_binding_method": FO_B,
                                     "storage_start_end_method": "fix_nothing"})
    start, end = _chain(step)
    assert -TOL <= start <= 2.0 + TOL and end >= start - TOL
    assert _production(step) == pytest.approx(4.0, abs=TOL)
    pb = build(step.flex_data)
    assert list(row_terms(pb, "rp_inter_period_start_max_state")) == ["rp_inter_period_start_max_state[s,p1,t0001]"]
    assert list(row_terms(pb, "rp_inter_period_start_min_state")) == ["rp_inter_period_start_min_state[s,p1,t0001]"]


def test_fix_start_end_pins_both_ends_of_the_chain(tmp_path_factory):
    """fix_start_end 0.5 / 0.25 on 2 MWh the solve cannot change: the chain starts at 1 and ends at 0.5 (both on the
    chain): the explicit drain of 0.5 MWh, production 3.5."""
    step = _run(tmp_path_factory, s={"existing": 2.0, "storage_binding_method": FO_B,
                                     "storage_start_end_method": "fix_start_end", "storage_state_start": 0.5,
                                     "storage_state_end": 0.25})
    assert _chain(step) == pytest.approx((1.0, 0.5), abs=TOL)
    assert _production(step) == pytest.approx(3.5, abs=TOL)
    pb = build(step.flex_data)
    assert list(row_terms(pb, "rp_inter_period_end_fix")) == ["rp_inter_period_end_fix[s,p1,t0003]"]
    assert not row_terms(pb, "storage_state_end_binding")


@pytest.mark.parametrize("rp", [True, False], ids=["rp", "hourly"])
@pytest.mark.parametrize("existing", [0.0, 2.0])
def test_fix_start_end_drain_on_investable_capacity_warns(tmp_path_factory, capsys, rp, existing):
    """fix_start_end 0.5 / 0.25 on storage the solve can invest in (owner decision, Phase 5; before a configuration
    error): the run solves and prints one warning line with the MWh drained from existing capacity
    (0.5·E − 0.25·E).  The matching end stays, so storage built must return its start energy: with the end pin
    0.25·(E + I) ≥ 0.5·I the investment is capped at I ≤ 0.25·E / (0.5 − 0.25) = E.  Each MWh built under the cap
    lets the end hold less of the existing start energy, so the store delivers 0.25·(E + I), at most the user's own
    start energy on existing capacity (0.5·E): here storage at 1 per MWh is cheaper than production at 10, so the
    solve builds to the cap (E), never beyond (no free energy)."""
    step = _run(tmp_path_factory, invest=True, rp=rp,
                s={"existing": existing, "storage_binding_method": FO_B if rp else "bind_forward_only",
                   "storage_start_end_method": "fix_start_end", "storage_state_start": 0.5,
                   "storage_state_end": 0.25})
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "fix_start_end with storage_state_end below" in ln]
    assert len(lines) == 1, out[-2000:]
    assert "1 storage node (s)" in lines[0] and f"drain {0.25 * existing:.6g} MWh" in lines[0]
    assert "can be invested in" in lines[0]
    invested = _invested(step)
    assert invested == pytest.approx(existing, abs=TOL)
    delivered = 4.0 - _production(step)
    assert delivered == pytest.approx(0.25 * (existing + invested), abs=TOL)
    assert delivered <= 0.5 * existing + TOL
    _no_slack(step)
    pb = build(step.flex_data)
    match = "rp_inter_period_end_match" if rp else "storage_state_end_match_share"
    assert row_terms(pb, match)


@pytest.mark.parametrize("rp", [True, False], ids=["rp", "hourly"])
def test_reference_series_starts_from_its_first_value(tmp_path_factory, rp):
    """use_reference_value from a time series that falls from 0.8 to 0.3 (a historical drawdown), no fix_start,
    forward chain on 2 MWh the solve cannot change (owner decision, Phase 5): the start is the reference at the
    model's first step (0.8 → 1.6 MWh, not free) and the end the reference at the last (0.3 → 0.6 MWh), so the store
    delivers the explicit 1 MWh drawdown of existing capacity and production is 3 (before: a free start bounded by
    the matching end, production 4)."""
    step = _run(tmp_path_factory, rp=rp,
                s={"existing": 2.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_nothing",
                   "storage_solve_horizon_method": "use_reference_value",
                   "storage_state_reference_value": _m(dict(zip(STEPS, [0.8, 0.7, 0.5, 0.3])), "time")})
    assert _production(step) == pytest.approx(3.0, abs=TOL)
    pb = build(step.flex_data)
    if rp:
        start, end = _chain(step)
        assert (start, end) == pytest.approx((1.6, 0.6), abs=TOL)
        assert list(row_terms(pb, "rp_inter_period_reference_value")) == [
            "rp_inter_period_reference_value[s,p1,t0003]"]
        assert list(row_terms(pb, "rp_inter_period_start_fix")) == ["rp_inter_period_start_fix[s,p1,t0001]"]
        assert not row_terms(pb, "storage_state_solve_horizon_reference_value")
    else:
        state = _values(step, "v_state")
        assert state[("s", "p1", "t0004")] == pytest.approx(0.6, abs=TOL)
        assert state[("s", "p1", "t0001")] == pytest.approx(1.6 - 1.0 + _production_at(step, "t0001"), abs=TOL)
        assert not row_terms(pb, "storage_state_end_match")


def _production_at(step, t: str) -> float:
    return sum(v for k, v in _values(step, "v_flow").items() if k[0] == "src" and k[-1] == t)


@pytest.mark.parametrize("fix_start", [None, 0.8, 0.5], ids=["reference", "same", "differs"])
def test_reference_start_and_fix_start(tmp_path_factory, capsys, fix_start):
    """A use_reference_value node without fix_start starts at the reference (0.8); with fix_start the share of
    fix_start wins, and a printed warning names the node where the two differ (one line)."""
    s = {"existing": 2.0, "storage_binding_method": "bind_forward_only",
         "storage_start_end_method": "fix_nothing" if fix_start is None else "fix_start",
         "storage_solve_horizon_method": "use_reference_value", "storage_state_reference_value": 0.8}
    if fix_start is not None:
        s["storage_state_start"] = fix_start
    step = _run(tmp_path_factory, rp=False, s=s)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "differs from the storage_state_reference_value" in ln]
    share = 0.8 if fix_start is None else fix_start
    # start share·2 MWh, end 0.8·2 MWh: the store gives (share − 0.8)·2 net
    assert _production(step) == pytest.approx(4.0 - (share - 0.8) * 2.0, abs=TOL)
    if fix_start == 0.5:
        assert len(lines) == 1 and "1 storage node (s)" in lines[0]
    else:
        assert not lines


@pytest.mark.parametrize("rp", [True, False])
@pytest.mark.parametrize("fix_start", [0.5, None], ids=["fix_start", "reference_start"])
def test_reference_below_the_start_on_investable_storage_warns(tmp_path_factory, capsys, rp, fix_start):
    """use_reference_value end below the start (fix_start 0.5, or a reference series 0.5 → 0.3) on storage the solve
    can invest in (owner decision, Phase 5; before a configuration error): the run solves with the matching end,
    which caps the investment (here at 0: nothing exists), and prints one warning line."""
    s = {"existing": 0.0, "storage_binding_method": FO_B if rp else "bind_forward_only",
         "storage_solve_horizon_method": "use_reference_value"}
    if fix_start is None:
        s.update({"storage_start_end_method": "fix_nothing",
                  "storage_state_reference_value": _m(dict(zip(STEPS, [0.5, 0.5, 0.4, 0.3])), "time")})
    else:
        s.update({"storage_start_end_method": "fix_start", "storage_state_start": fix_start,
                  "storage_state_reference_value": 0.3})
    step = _run(tmp_path_factory, invest=True, rp=rp, s=s)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "use_reference_value end below their start" in ln]
    assert len(lines) == 1 and "1 storage node (s)" in lines[0], out[-2000:]
    assert _invested(step) == pytest.approx(0.0, abs=TOL)
    assert _production(step) == pytest.approx(4.0, abs=TOL)
    _no_slack(step)


@pytest.mark.parametrize("start", ["fix_nothing", "fix_start"])
def test_investment_solve_short_of_the_model_end_warns(tmp_path_factory, capsys, start):
    """A sequence of investment solves (y2020_5week … y2035_5week of tests.json) with an investable battery that
    chains forward: the first solve does not reach the end of the model, so it cannot hold the end at its start.

    * A free start (fix_nothing, the default): the last solve continues from the carry and ends the battery at
      least at the level the first solve started it from (storage_state_roll_match), so there is no warning
      (review M5: before, every such run warned that built storage could deliver its start energy for free).
    * fix_start: the start on built capacity is not matched over the sequence; a warning names the battery once.
    """
    from tests.engine._unit_db import b64, make_db, run

    values = [["node", "battery", "storage_binding_method", b64("bind_forward_only", "str")],
              ["node", "battery", "storage_start_end_method", b64(start, "str")],
              ["node", "battery", "storage_solve_horizon_method", b64("free", "str")]]
    if start == "fix_start":
        values.append(["node", "battery", "storage_state_start", b64(0.5, "float")])
    url, scen = make_db(tmp_path_factory, alt=f"end_match_warn_{start}",
                        base_chain=["init", "west", "wind", "battery", "battery_invest", "5weeks",
                                    "invest_solveSequence_5weeks"],
                        values=values)
    steps = run(tmp_path_factory, url, scen)
    out = capsys.readouterr().out
    warned = out.count("the solve does not reach the end of the model, so the end is not held")
    if start == "fix_nothing":
        assert warned == 0, out[-3000:]
        last = list(steps.values())[-1]
        assert row_terms(build(last.flex_data), "storage_state_roll_match", "battery,")
    else:
        assert warned == 1, out[-3000:]
        assert "1 storage node (battery) start" in out
    for step in steps.values():
        assert not row_terms(build(step.flex_data), "storage_state_end_match")


def test_life_extension_in_the_start_period_is_a_start_decision(monkeypatch):
    """Review M2: a storage node whose cohort the solve can extend in the start period has its start capacity
    decided by the solve, as for an investment (fix_start_energy and node_in_service count the extended capacity):
    both investability checks (the warnings of _check_storage_accounting and the matching end rows through
    _invest_at_start_nodes) count it.  Before, only nd_invest_set was read: such a fix_start start was taken for an
    endowment and got no matching end."""
    from types import SimpleNamespace

    from flextool.engine.model import core

    extend = pl.DataFrame({"n": ["s", "s"], "cohort": ["existing", "existing"], "d": ["p1", "p2"]})
    monkeypatch.setattr(core._extend, "options", lambda d, axis: extend if axis == "n" else None)
    b = SimpleNamespace(d=SimpleNamespace(nd_invest_set=pl.DataFrame({"n": ["t"], "d": ["p2"]})), has_invest_n=True)
    assert core._start_decision_periods(b) == {"s": {"p1", "p2"}, "t": {"p2"}}
    nodes = pl.DataFrame({"n": ["s", "t"]})
    assert core._invest_at_start_nodes(b, nodes, ["p1"]) == {"s"}
    assert core._invest_at_start_nodes(b, nodes, ["p2"]) == {"s", "t"}
    b.has_invest_n = False
    assert core._invest_at_start_nodes(b, nodes, ["p1"]) == {"s"}


def test_timeblock_fix_start_end_in_a_solve_mixing_representative_and_plain_periods_runs(tmp_path_factory):
    """Review M6: bind_within_timeblock with fix_start_end in a solve whose p1 has representative periods and p2 not.
    Before, the engine mapped the node to bind_forward_only_blended_weights and _check_rp_solve then stopped the
    solve with an error about a *_blended_weights method the user never set.  Now the node keeps cycling within each
    timeblock (no energy moves between the representative periods): the solve runs and needs no slack."""
    step = _run(tmp_path_factory, mixed=True,
                s={"storage_binding_method": "bind_within_timeblock", "storage_start_end_method": "fix_start_end",
                   "storage_state_start": 0.5, "storage_state_end": 0.5, "existing": 2.0})
    _no_slack(step)
    assert "v_state_inter" not in step.solution._vars
    # every timeblock returns to its start: the store gives no energy, src produces the whole demand
    assert _production(step) == pytest.approx(8.0, abs=TOL)


@pytest.mark.parametrize("method", ["bind_within_solve", "bind_within_period"])
def test_cyclic_fix_start_end_on_representative_periods_chains_on_the_blended_chain(tmp_path_factory, method):
    """Review M6: with fix_start_end the binding method is ignored, so a plain cyclic method is not the
    configuration error of plain cycles on representative periods (before: the error, which named the method the
    run ignores anyway); the node chains forward on the base-period chain, where the start and end pins act."""
    step = _run(tmp_path_factory, s={"storage_binding_method": method, "storage_start_end_method": "fix_start_end",
                                     "storage_state_start": 0.5, "storage_state_end": 0.5, "existing": 2.0})
    _no_slack(step)
    start, end = _chain(step)
    assert start == pytest.approx(1.0, abs=TOL) and end == pytest.approx(1.0, abs=TOL)


def test_plain_cyclic_error_on_representative_periods_names_the_count_and_the_first(tmp_path_factory):
    """Review L4: the plain-cyclic error on representative periods is in the one-line format (how many nodes and the
    first, with its method)."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError) as exc:
        _run(tmp_path_factory, s={"storage_binding_method": "bind_within_solve"})
    msg = str(exc.value)
    assert "1 storage node (s) use a storage_binding_method (e.g. s: bind_within_solve)" in msg, msg
