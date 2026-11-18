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


def _run(tmp_path_factory, *, s: dict, rp: bool = True, invest: bool = False):
    """Build, solve and return the step of the tiny model; ``s`` = extra parameters of node ``s``."""
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
    assert len(lines) == 1 and "s 1 MWh" in lines[0] and "endowment" in lines[0]
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


def test_fix_start_end_drain_on_investable_capacity_is_an_error(tmp_path_factory):
    """fix_start_end with storage_state_end below storage_state_start on storage the solve can invest in: the drain
    grows with the storage built, which has no consistent reading (validation error)."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    with pytest.raises(FlexToolConfigError, match="storage_state_end below storage_state_start"):
        _run(tmp_path_factory, invest=True, s={"existing": 0.0, "storage_binding_method": FO_B,
                                               "storage_start_end_method": "fix_start_end",
                                               "storage_state_start": 0.5, "storage_state_end": 0.25})


@pytest.mark.parametrize("rp", [True, False], ids=["rp", "hourly"])
def test_reference_series_below_the_start_does_not_drain(tmp_path_factory, rp):
    """use_reference_value from a time series that falls from 0.8 to 0.3 (a historical drawdown), free start,
    forward chain on 2 MWh: the reference sets the end at 0.6 MWh, and the matching end bounds the start by it, so
    the store delivers nothing net and production is the demand (before: start full, end 0.6, 1.4 MWh free)."""
    step = _run(tmp_path_factory, rp=rp,
                s={"existing": 2.0, "storage_binding_method": FO_B, "storage_start_end_method": "fix_nothing",
                   "storage_solve_horizon_method": "use_reference_value",
                   "storage_state_reference_value": _m(dict(zip(STEPS, [0.8, 0.7, 0.5, 0.3])), "time")})
    assert _production(step) == pytest.approx(4.0, abs=TOL)
    pb = build(step.flex_data)
    if rp:
        start, end = _chain(step)
        assert end == pytest.approx(0.6, abs=TOL) and start <= end + TOL
        assert list(row_terms(pb, "rp_inter_period_reference_value")) == [
            "rp_inter_period_reference_value[s,p1,t0003]"]
        assert not row_terms(pb, "storage_state_solve_horizon_reference_value")
    else:
        state = _values(step, "v_state")
        assert state[("s", "p1", "t0004")] == pytest.approx(0.6, abs=TOL)
        assert state[("s", "p1", "t0001")] <= 0.6 + TOL
        assert list(row_terms(pb, "storage_state_end_match")) == ["storage_state_end_match[s,p1,t0001,p1,t0004]"]


@pytest.mark.parametrize("rp", [True, False])
def test_reference_below_fix_start_share_on_investable_storage_is_an_error(tmp_path_factory, rp):
    """use_reference_value 0.3 below fix_start 0.5 on storage the solve can invest in: ending at the reference would
    let built storage deliver part of its start energy for free, against the matching end — a configuration error
    (owner decision).  Before: the reference became a lower bound (rp_inter_period_reference_value_min /
    storage_state_solve_horizon_reference_value_min)."""
    from flextool.engine.input._solve_state import FlexToolConfigError

    with pytest.raises(FlexToolConfigError, match="storage_state_reference_value below storage_state_start"):
        _run(tmp_path_factory, invest=True, rp=rp,
             s={"existing": 0.0, "storage_binding_method": FO_B if rp else "bind_forward_only",
                "storage_start_end_method": "fix_start", "storage_state_start": 0.5,
                "storage_solve_horizon_method": "use_reference_value", "storage_state_reference_value": 0.3})


def test_investment_solve_short_of_the_model_end_warns(tmp_path_factory, capsys):
    """A sequence of investment solves (y2020_5week … y2035_5week of tests.json) with an investable battery that
    chains forward from a free start: the first solve does not reach the end of the model and the later ones
    continue from a carried level, so no solve holds both ends and the matching end cannot be applied; a warning
    names the battery once."""
    from tests.engine._unit_db import b64, make_db, run

    url, scen = make_db(tmp_path_factory, alt="end_match_warn",
                        base_chain=["init", "west", "wind", "battery", "battery_invest", "5weeks",
                                    "invest_solveSequence_5weeks"],
                        values=[["node", "battery", "storage_binding_method", b64("bind_forward_only", "str")],
                                ["node", "battery", "storage_start_end_method", b64("fix_nothing", "str")],
                                ["node", "battery", "storage_solve_horizon_method", b64("free", "str")]])
    steps = run(tmp_path_factory, url, scen)
    out = capsys.readouterr().out
    assert out.count("the solve does not reach the end of the model, so the end is not held") == 1, out[-3000:]
    assert "storage node(s) battery start" in out
    for step in steps.values():
        assert not row_terms(build(step.flex_data), "storage_state_end_match")
