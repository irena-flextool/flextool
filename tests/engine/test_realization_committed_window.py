"""No information from the future in a stochastic single solve and its realization (item 255).

A single solve decides at its fan's analysis time: every period after the fan's period reads the forecast made
then, never one made later (rule (a), the same as a roll of a rolling solve).  Its realization solve dispatches the
committed periods (``realized_periods``) only, with the realized branch's data: no look-ahead period, so the
committed periods' storage is not decided knowing the later actuals (rule (b), the same as a realization roll).

Fixture (built from the schema JSON, never a checked-in sqlite): steps t0001..t0008 of 1 h, period p2030 =
t0001..t0004, p2040 = t0005..t0008, one year each.  Node ``elec`` (penalties 10 000), unit ``gas`` (200 MW, 50
CUR/MWh), unit ``wind`` (upper-limit stochastic profile), storage node ``bat`` (100 MWh, starts at 50 MWh,
end value 30 CUR/MWh at the end of the solve) behind connection ``bconn`` (50 MW, efficiency 0.9).  Solve ``s30``
fans at p2030 / t0001 into forecasts ``high`` / ``low`` (weights 0.5) and the realized branch ``rlz``, and realizes
p2030 with p2040 as its look-ahead.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

import flextool

SCHEMA_JSON = Path(flextool.__file__).parent / "schemas" / "spinedb_schema.json"
STEPS = [f"t{i:04d}" for i in range(1, 9)]
P30, P40 = STEPS[:4], STEPS[4:]


def _m(d: dict, name: str) -> api.Map:
    return api.Map(list(d), list(d.values()), index_name=name)


def _fan(period: str, at: str) -> api.Map:
    return _m({period: _m({b: _m({at: _m({r: w}, "realized")}, "analysis_time")
                           for b, (r, w) in {"high": ("no", 0.5), "low": ("no", 0.5),
                                             "rlz": ("yes", 0.0)}.items()}, "branch")}, "period")


def _solve(name: str, periods: list[str], realized: str, at: str, invest: bool) -> list[tuple]:
    timeset = {"p2030": "ts30", "p2040": "ts40"}
    rows = [("solve", name, "solve_mode", "single_solve"),
            ("solve", name, "period_timeset", _m({p: timeset[p] for p in periods}, "period")),
            ("solve", name, "years_represented", _m({p: 1.0 for p in periods}, "period")),
            ("solve", name, "realized_periods", api.Array([realized])),
            ("solve", name, "stochastic_branches", _fan(realized, at))]
    if invest:
        rows += [("solve", name, "invest_periods", api.Array(periods)),
                 ("solve", name, "realized_invest_periods", api.Array([realized]))]
    return rows


def _run(tmp_path_factory, *, solves: list[str], profile: api.Map, wind: list[tuple], inflow,
         s40: bool) -> dict:
    """Build the fixture DB and run it; ``{solve name: OrchestrationStep}``."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp("committed_window")
    db = work / "t.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    url = f"sqlite:///{db.resolve().as_posix()}"
    entities = [("model", "flextool"), ("timeline", "tl"), ("timeset", "ts30"), ("timeset", "ts40"),
                ("node", "elec"), ("node", "bat"), ("unit", "wind"), ("unit", "gas"),
                ("unit__outputNode", ("wind", "elec")), ("unit__outputNode", ("gas", "elec")),
                ("profile", "wprof"), ("unit__node__profile", ("wind", "elec", "wprof")),
                ("group", "g"), ("group__unit", ("g", "wind")),
                ("connection", "bconn"), ("connection__node__node", ("bconn", "elec", "bat")),
                *[("solve", s) for s in solves]]
    values = [
        ("model", "flextool", "solves", api.Array(solves)),
        ("timeline", "tl", "timestep_duration", _m({t: 1.0 for t in STEPS}, "time")),
        ("timeset", "ts30", "timeline", "tl"), ("timeset", "ts30", "timeset_duration", _m({"t0001": 4.0}, "time")),
        ("timeset", "ts40", "timeline", "tl"), ("timeset", "ts40", "timeset_duration", _m({"t0005": 4.0}, "time")),
        ("node", "elec", "node_type", "balance"), ("node", "elec", "inflow", inflow),
        ("node", "elec", "penalty_up", 10000.0), ("node", "elec", "penalty_down", 10000.0),
        ("node", "bat", "node_type", "storage"), ("node", "bat", "existing", 100.0),
        ("node", "bat", "storage_start_end_method", "fix_start"), ("node", "bat", "storage_state_start", 0.5),
        ("node", "bat", "storage_solve_horizon_method", "use_reference_price"),
        ("node", "bat", "storage_state_reference_price", 30.0),
        ("node", "bat", "penalty_up", 10000.0), ("node", "bat", "penalty_down", 10000.0),
        ("connection", "bconn", "existing", 50.0), ("connection", "bconn", "efficiency", 0.9),
        ("unit", "gas", "existing", 200.0), ("unit", "gas", "efficiency", 1.0),
        ("unit__outputNode", ("gas", "elec"), "other_operational_cost", 50.0),
        ("unit", "wind", "efficiency", 1.0), ("unit", "wind", "virtual_unitsize", 1.0),
        *[("unit", "wind", p, v) for p, v in wind],
        ("unit__node__profile", ("wind", "elec", "wprof"), "profile_method", "upper_limit"),
        ("profile", "wprof", "profile", profile),
        ("group", "g", "include_stochastics", "yes"),
        *_solve("s30", ["p2030", "p2040"], "p2030", "t0001", invest=s40),
    ]
    if s40:
        values += _solve("s40", ["p2040"], "p2040", "t0005", invest=True)
    with open_database_mapping(url) as mp:
        _, errors = api.import_data(
            mp, alternatives=[("a", "")], scenarios=[("sc", False, "")], scenario_alternatives=[("sc", "a", None)],
            entities=entities,
            entity_alternatives=[(c, (e,), "a", True) for c, e in
                                 [("node", "elec"), ("node", "bat"), ("unit", "wind"), ("unit", "gas"),
                                  ("connection", "bconn")]],
            parameter_values=[(c, e, p, v, "a") for c, e, p, v in values])
        assert not errors, errors
        mp.commit_session("fixture")
    return run_chain_from_db(url, "sc", work_folder=work / "w", keep_solutions=True)


def _state(step) -> dict[tuple[str, str], float]:
    """``(period, step) -> MWh`` of ``bat`` (``v_state`` is in units of the 100 MWh storage)."""
    df = step.solution.value("v_state").with_columns(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    return {(d, t): 100.0 * v for d, t, v in df.select("d", "t", "value").iter_rows()}


@pytest.mark.solver
def test_chain_lookahead_reads_the_fan_forecast_and_realizes_the_committed_period(tmp_path_factory) -> None:
    """Chain s30 (p2030 realized, p2040 look-ahead, fan at t0001) -> s30__realized_dispatch -> s40 (p2040, fan at
    t0005) -> s40__realized_dispatch.  Wind is invested (invest_total, 1000 CUR/kW, 30 years at 5 %, at most
    400 MW).  The t0001 forecast covers both periods (p2040 at half the CF of the t0005 forecast: high 0.45, low
    0.15, rlz 0.3 / 0.1 / 0.45 / 0.3); the t0005 forecast, made for s40, covers p2040 (high 0.9, low 0.3).

    (a) s30 decides at t0001, so its p2040 reads the t0001 forecast (before: the t0005 one, made for s40):
    hand-calculated (critique of item 255) s30 invests W = 222.22 MW (before: 295.83).  (b) The realization of s30
    dispatches p2030 only (t0001..t0004, realized CF 0.5, 0.5, 0.5, 0.1): t0001..t0003 wind 111.11 MW, surplus
    11.11 MW -> +10 MWh each -> 80 MWh; t0004 wind 22.22 MW, deficit 77.78 MW, discharge capped at 50 MW
    delivered = 55.56 MWh drawn (45 CUR/MWh saved > end value 30) -> 24.44 MWh, gas 27.78 MW.  s40 starts there."""
    cf = {"rlz": ([0.5, 0.5, 0.5, 0.1], [0.6, 0.2, 0.9, 0.6]),
          "high": ([0.9] * 4, [0.9] * 4), "low": ([0.3] * 4, [0.3] * 4)}
    profile = _m({b: _m({"t0001": _m(dict(zip(STEPS, c30 + [0.5 * x for x in c40])), "time"),
                         "t0005": _m(dict(zip(P40, c40)), "time")}, "analysis_time")
                  for b, (c30, c40) in cf.items()}, "branch")
    wind = [("existing", 0.0), ("invest_method", "invest_total"), ("invest_cost", 1000.0), ("lifetime", 30.0),
            ("discount_rate", 0.05), ("invest_max_total", 400.0)]
    steps = _run(tmp_path_factory, solves=["s30", "s40"], profile=profile, wind=wind, inflow=-100.0, s40=True)
    assert list(steps) == ["s30", "s30__realized_dispatch", "s40", "s40__realized_dispatch"]
    invest = steps["s30"].solution.value("v_invest_p").with_columns(pl.col("d").cast(pl.Utf8))
    assert invest.filter(pl.col("d") == "p2030")["value"].item() == pytest.approx(2000.0 / 9.0, rel=1e-6)
    realization = steps["s30__realized_dispatch"]
    state = _state(realization)
    assert sorted(state) == [("p2030", t) for t in P30]
    assert [state[("p2030", t)] for t in P30] == pytest.approx([60.0, 70.0, 80.0, 220.0 / 9.0], rel=1e-6)
    profile_cells = realization.flex_data_provider.get("solve_data/pdtProfile")
    assert set(profile_cells["period"].cast(pl.Utf8).to_list()) == {"p2030"}
    carried = steps["s40"].flex_data.p_roll_continue_state
    carried = carried.frame if hasattr(carried, "frame") else carried
    assert carried.filter(pl.col("n").cast(pl.Utf8) == "bat")["value"].item() == pytest.approx(220.0 / 9.0,
                                                                                               rel=1e-6)


def _scarce_lookahead(tmp_path_factory) -> dict:
    """``s30`` alone, wind 110 MW existing; the t0001 forecast covers both periods: rlz CF 1, 1, 1, 0.5 in p2030
    and 0 in p2040 (high 1.0, low 0.5 throughout); demand 100 MW in p2030 and 300 MW in p2040, so the realized
    p2040 is short of 100 MW per step against 200 MW of gas."""
    cf = {"rlz": [1.0, 1.0, 1.0, 0.5] + [0.0] * 4, "high": [1.0] * 8, "low": [0.5] * 8}
    profile = _m({b: _m({"t0001": _m(dict(zip(STEPS, c)), "time")}, "analysis_time") for b, c in cf.items()},
                 "branch")
    inflow = _m({t: -100.0 if t in P30 else -300.0 for t in STEPS}, "time")
    return _run(tmp_path_factory, solves=["s30"], profile=profile, wind=[("existing", 110.0)], inflow=inflow,
                s40=False)


@pytest.mark.solver
def test_realization_does_not_save_storage_for_the_realized_lookahead(tmp_path_factory) -> None:
    """The realization of ``s30`` dispatches p2030 only; with the scarce realized p2040 in its horizon it would keep
    the battery full for p2040.  Oracle (a separate hand calculation, item 255; efficiency 0.9 each way, 50 MW on
    the receiving side, end value 30 CUR/MWh at the last step of the solve, start 50 MWh):

    * p2030 only (now): t0001..t0003 wind 110 MW, surplus 10 MW drawn -> +9 MWh each: 59, 68, 77 MWh; t0004 wind
      55 MW, the 45 MW deficit is discharged (33.3 CUR per MWh delivered < gas 50): 50 MWh drawn -> 27 MWh, gas 0.
    * with the realized p2040 (before): each stored MWh avoids 0.9 x 10 000 of unserved energy in p2040, so the
      battery is full (100 MWh) at the end of t0004 -- decided knowing the later actuals."""
    steps = _scarce_lookahead(tmp_path_factory)
    assert list(steps) == ["s30", "s30__realized_dispatch"]
    realization = steps["s30__realized_dispatch"]
    state = _state(realization)
    assert sorted(state) == [("p2030", t) for t in P30]
    assert [state[("p2030", t)] for t in P30] == pytest.approx([59.0, 68.0, 77.0, 27.0], rel=1e-6)
    flow = realization.solution.value("v_flow").with_columns(
        pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    assert flow.filter(pl.col("p") == "gas")["value"].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert set(flow["d"].to_list()) == {"p2030"}
