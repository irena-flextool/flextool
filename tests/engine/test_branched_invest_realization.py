"""The realization dispatch of an investment solve (``invest_solve_branching_method``, data version 84).

Model: one balance node ``elec`` (demand 100 MW, penalty 10 000 EUR/MWh), the investable unit ``gen`` (existing 0,
invest_cost 1 EUR/kW, lifetime 20 a, discount rate 0.05: an annuity of about 80 EUR/MW a year, far below the
penalty of a missing MW, 20 000 EUR per 2 h period; other_operational_cost 10 EUR/MWh) and 100 MW of free ``wind``
whose upper-limit profile is stochastic (group ``sg`` with ``include_stochastics``).  Two periods ``p1`` (t01-t02)
and ``p2`` (t03-t04) of 1 h steps, one year each.  Wind share: forecast ``high`` 0.8 (weight 1), ``low`` 0.2 in p1
and 0.0 in p2 (weight 3), the realized branch ``real`` 0.1, so the residual demand ``gen`` must cover is high 20 / 20,
low 80 / 100, real 90 / 90 MW (p1 / p2).

With ``storage``: a node ``res`` (storage 100 MWh, ``fix_start`` at 0, ``bind_forward_only``) that receives 20 MWh
each hour (inflow) and feeds ``elec`` through ``turbine`` (100 MW, efficiency 1), which has no capacity in p1
(``existing`` p1 0, p2 100).  So nothing leaves ``res`` in p1: every solve that dispatches p1 ends it at
0 + 2·20 = **40 MWh**.
"""
from __future__ import annotations

import logging

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm

STEPS = ("t01", "t02", "t03", "t04")
PERIOD_STEPS = {"p1": ("t01", "t02"), "p2": ("t03", "t04")}
FIRST_STEP = {"p1": "t01", "p2": "t03"}
#: The wind share per branch and period.
WIND = {"high": {"p1": 0.8, "p2": 0.8}, "low": {"p1": 0.2, "p2": 0.0}, "real": {"p1": 0.1, "p2": 0.1}}
PERIOD_OF = {t: p for p, steps in PERIOD_STEPS.items() for t in steps}
WEIGHT = {"high": 1.0, "low": 3.0}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _spec(*, reveal: str, method: str | None, realized: list[str], realized_invest: list[str],
          invest: list[str] = ("p1", "p2"), storage: bool = False, two_d: "dict[str, list[str]] | None" = None,
          two_d_invest: "dict[str, list[str]] | None" = None) -> dict:
    """The model with its fan at *reveal* (``p1``: the solve's first period, or ``p2``: mid-horizon) and
    ``invest_solve_branching_method`` *method* (``None``: absent, ``operations_only``).  *two_d*: 2-D maps of
    ``realized_periods`` / ``realized_invest_periods`` (``{row: [period]}``) and *two_d_invest* of
    ``invest_periods`` instead of the 1-D lists; ``stochastic_branches`` then holds a fan at every period."""
    from flextool.update_flextool.fixture_schema import schema_sections

    pack, pmap, nested = scm.pack, scm.pmap, scm.nested
    spec = schema_sections("packed")
    spec["alternatives"] = [["init", ""]]
    spec["scenarios"] = [["base", False, ""]]
    spec["scenario_alternatives"] = [["base", "init", None]]
    spec["entities"] = [["model", "flextool", None], ["node", "elec", None], ["timeline", "tl", None],
                        ["timeset", "ts1", None], ["timeset", "ts2", None], ["unit", "gen", None],
                        ["unit", "wind", None], ["group", "sg", None], ["profile", "wp", None],
                        ["solve", "st", None],
                        ["unit__outputNode", ["gen", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                        ["group__unit", ["sg", "wind"], None],
                        ["unit__node__profile", ["wind", "elec", "wp"], None]]
    spec["entity_alternatives"] = [["node", ["elec"], "init", True], ["unit", ["gen"], "init", True],
                                   ["unit", ["wind"], "init", True]]
    fans = sorted({p for rows in (two_d or {}).values() for p in rows}) if two_d else [reveal]
    # The realized branch: a series at the solve's start (the periods before the reveal) and one at each fan.
    starts = ["t01"] + [FIRST_STEP[p] for p in fans if p != "p1"]
    real = [(at, nested([(t, WIND["real"][PERIOD_OF[t]]) for t in STEPS[STEPS.index(at):]])) for at in starts]
    forecasts = [(b, nested([(FIRST_STEP[p], nested([(t, WIND[b][PERIOD_OF[t]])
                                                     for t in STEPS[STEPS.index(FIRST_STEP[p]):]]))
                             for p in fans])) for b in WEIGHT]
    profile = pack(pmap(forecasts + [("real", nested(real))]), "map")
    fan = pmap([(p, nested([(b, nested([(FIRST_STEP[p], nested([("no", w)]))])) for b, w in WEIGHT.items()]
                           + [("real", nested([(FIRST_STEP[p], nested([("yes", 0.0)]))]))])) for p in fans])
    pv = [
        ["model", "flextool", "solves", scm.array(["st"]), "init"],
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in STEPS], index_name="time"), "map"),
         "init"],
        ["timeset", "ts1", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts1", "timeset_duration", pack(pmap([("t01", 2.0)], index_name="time"), "map"), "init"],
        ["timeset", "ts2", "timeline", pack("tl", "str"), "init"],
        ["timeset", "ts2", "timeset_duration", pack(pmap([("t03", 2.0)], index_name="time"), "map"), "init"],
        ["node", "elec", "node_type", pack("balance", "str"), "init"],
        ["node", "elec", "penalty_up", pack(10000.0, "float"), "init"],
        ["node", "elec", "penalty_down", pack(10000.0, "float"), "init"],
        ["node", "elec", "inflow", pack(-100.0, "float"), "init"],
        ["unit", "gen", "efficiency", pack(1.0, "float"), "init"],
        ["unit", "gen", "existing", pack(0.0, "float"), "init"],
        ["unit", "gen", "virtual_unitsize", pack(1.0, "float"), "init"],
        ["unit", "gen", "invest_method", pack("invest_total", "str"), "init"],
        ["unit", "gen", "invest_cost", pack(1.0, "float"), "init"],
        ["unit", "gen", "discount_rate", pack(0.05, "float"), "init"],
        ["unit", "gen", "lifetime_technical", pack(20.0, "float"), "init"],
        ["unit", "gen", "invest_max_total", pack(1000.0, "float"), "init"],
        ["unit__outputNode", ["gen", "elec"], "other_operational_cost", pack(10.0, "float"), "init"],
        ["unit", "wind", "efficiency", pack(1.0, "float"), "init"],
        ["unit", "wind", "existing", pack(100.0, "float"), "init"],
        ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
        ["profile", "wp", "profile", profile, "init"],
        ["group", "sg", "include_stochastics", pack("yes", "str"), "init"],
        ["solve", "st", "solve_mode", pack("single_solve", "str"), "init"],
        ["solve", "st", "period_timeset", pack(pmap([("p1", "ts1"), ("p2", "ts2")], index_name="period"), "map"),
         "init"],
        ["solve", "st", "years_represented", pack(pmap([("p1", 1.0), ("p2", 1.0)], index_name="period"), "map"),
         "init"],
        ["solve", "st", "stochastic_branches", pack(fan, "map"), "init"],
    ]
    if two_d is None:
        pv += [["solve", "st", "invest_periods", scm.array(list(invest)), "init"],
               ["solve", "st", "realized_periods", scm.array(list(realized)), "init"],
               ["solve", "st", "realized_invest_periods", scm.array(list(realized_invest)), "init"]]
    else:
        pv += [["solve", "st", "invest_periods", scm.two_d(two_d_invest or two_d), "init"],
               ["solve", "st", "realized_periods", scm.two_d(two_d), "init"],
               ["solve", "st", "realized_invest_periods", scm.two_d(two_d), "init"]]
    if method is not None:
        pv.append(["solve", "st", "invest_solve_branching_method", pack(method, "str"), "init"])
    if storage:
        spec["entities"] += [["node", "res", None], ["unit", "turbine", None],
                             ["unit__inputNode", ["turbine", "res"], None],
                             ["unit__outputNode", ["turbine", "elec"], None]]
        spec["entity_alternatives"] += [["node", ["res"], "init", True], ["unit", ["turbine"], "init", True]]
        pv += [["node", "res", "node_type", pack("storage", "str"), "init"],
               ["node", "res", "existing", pack(100.0, "float"), "init"],
               ["node", "res", "inflow", pack(20.0, "float"), "init"],
               ["node", "res", "storage_start_end_method", pack("fix_start", "str"), "init"],
               ["node", "res", "storage_state_start", pack(0.0, "float"), "init"],
               ["node", "res", "storage_binding_method", pack("bind_forward_only", "str"), "init"],
               ["unit", "turbine", "efficiency", pack(1.0, "float"), "init"],
               ["unit", "turbine", "existing",
                pack(pmap([("p1", 0.0), ("p2", 100.0)], index_name="period"), "map"), "init"]]
    spec["parameter_values"] = pv
    return spec


def _value(steps, solve: str, var: str) -> pl.DataFrame:
    assert solve in steps, f"no {solve!r} step in {list(steps)}"
    sol = steps[solve].solution
    assert var in sol._vars, f"{var} not in {solve}: {sorted(sol._vars)}"
    return sol.value(var).with_columns(pl.col(c).cast(pl.Utf8) for c in sol._vars[var].dims)


def _w0639(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[W-0639]" in r.getMessage()]


def _sum(frame: pl.DataFrame, **keys) -> float:
    return float(frame.filter(pl.all_horizontal(*(pl.col(k) == v for k, v in keys.items())))["value"].sum())


def _periods(frame: pl.DataFrame) -> set[str]:
    return set(frame["d"].to_list())


def _realized_dispatch(folder, solve: str) -> list[tuple[str, str]]:
    """The ``(period, time)`` cells *solve* reports as realized dispatch (its raw ``dt_realize_dispatch``)."""
    frame = pl.read_parquet(folder / "w" / "output_raw" / f"dt_realize_dispatch__{solve}.parquet")
    return sorted(zip(frame["period"].cast(pl.Utf8).to_list(), frame["time"].cast(pl.Utf8).to_list()))


@pytest.mark.solver
def test_after_reveal_realization_keeps_the_tied_investment(tmp_path_factory, caplog) -> None:
    """``investments_after_reveal_period`` with the fan at p1 (the reveal) and p1 realized: the p1 investment x is
    one decision of both forecasts and must cover low's residual 80 (a missing MW costs far more than a MW built);
    building more at p1 does not pay, because p2's extra need (low 100) is per forecast and a p2 MW costs one year,
    a p1 MW two.  So x = 80 on both branch copies.  The engine adds ``st__realized_dispatch``: it dispatches p1 only
    (the realized periods from the reveal on), with the realized data, and keeps the tied investment as existing
    capacity, so ``gen`` runs at 80 MW of the realized residual 90 (the rest is the shortfall).  No W-0639."""
    caplog.set_level(logging.WARNING)
    folder = tmp_path_factory.mktemp("after_reveal")
    built, steps = scm.run_spec(_spec(reveal="p1", method="investments_after_reveal_period", realized=["p1"],
                                      realized_invest=["p1"]), folder)
    assert [s for s, _, _ in built] == ["st", "st__realized_dispatch"]
    invest = _value(steps, "st", "v_invest_p")
    assert _sum(invest, p="gen", d="p1_high") == pytest.approx(80.0, abs=1e-6)
    assert _sum(invest, p="gen", d="p1_low") == pytest.approx(80.0, abs=1e-6)
    flow = _value(steps, "st__realized_dispatch", "v_flow")
    assert _periods(flow) == {"p1"}
    for t in PERIOD_STEPS["p1"]:
        assert _sum(flow, p="gen", d="p1", t=t) == pytest.approx(80.0, abs=1e-6)
    assert _realized_dispatch(folder, "st__realized_dispatch") == [("p1", "t01"), ("p1", "t02")]
    assert _w0639(caplog) == []


@pytest.mark.solver
def test_midhorizon_realization_starts_at_the_reveal(tmp_path_factory) -> None:
    """An investment solve (``operations_only``) with its fan at p2 that realizes p1 and p2: it reports p1's
    operations itself, and its realization dispatches p2 only (from the reveal on), so p1 is not dispatched twice.
    The realization starts its storage from the investment solve's level at the end of p1, 40 MWh (``res`` gets
    20 MWh an hour and cannot discharge in p1).  In p2 it gets 2·20 more, and the water replaces ``gen``
    (10 EUR/MWh) within the turbine's 100 MW, so the turbine delivers 40 + 40 = **80 MWh** over p2 and ``res`` ends
    empty.  (Started empty instead, it would deliver 40.)"""
    folder = tmp_path_factory.mktemp("midhorizon")
    built, steps = scm.run_spec(_spec(reveal="p2", method=None, realized=["p1", "p2"],
                                      realized_invest=["p1", "p2"], storage=True), folder)
    assert [s for s, _, _ in built] == ["st", "st__realized_dispatch"]
    assert _sum(_value(steps, "st", "v_state"), n="res", d="p1", t="t02") == pytest.approx(0.4, abs=1e-9)  # 40 MWh / 100
    flow = _value(steps, "st__realized_dispatch", "v_flow")
    assert _periods(flow) == {"p2"}
    # Flows of the turbine are per unit of its 100 MW capacity.
    assert _sum(flow, p="turbine") * 100.0 == pytest.approx(80.0, abs=1e-6)
    assert _realized_dispatch(folder, "st") == [("p1", "t01"), ("p1", "t02")]
    assert _realized_dispatch(folder, "st__realized_dispatch") == [("p2", "t03"), ("p2", "t04")]


@pytest.mark.solver
def test_year_by_year_continues_from_the_realized_capacity(tmp_path_factory) -> None:
    """A 2-D map, one row per year (``y1``: p1, ``y2``: p2), each with its fan at its own period and
    ``investments_after_reveal_period``.  ``st_y1`` ties its p1 investment at low's 80; its realization keeps it.
    ``st_y2`` starts from that 80 MW (the tied value handed on under the real period name p1) and ties a p2
    investment of 100 − 80 = **20** (low's p2 residual); its realization has 100 MW against the realized residual
    90, so ``gen`` runs at 90.  Without the handoff ``st_y2`` would build 100."""
    folder = tmp_path_factory.mktemp("year_by_year")
    built, steps = scm.run_spec(_spec(reveal="p1", method="investments_after_reveal_period", realized=[],
                                      realized_invest=[], two_d={"y1": ["p1"], "y2": ["p2"]}), folder)
    assert [s for s, _, _ in built] == ["st_y1", "st_y1__realized_dispatch", "st_y2", "st_y2__realized_dispatch"]
    for solve, period, value in (("st_y1", "p1", 80.0), ("st_y2", "p2", 20.0)):
        invest = _value(steps, solve, "v_invest_p")
        for b in WEIGHT:
            assert _sum(invest, p="gen", d=f"{period}_{b}") == pytest.approx(value, abs=1e-6), (solve, b)
    flow = _value(steps, "st_y2__realized_dispatch", "v_flow")
    for t in PERIOD_STEPS["p2"]:
        assert _sum(flow, p="gen", d="p2", t=t) == pytest.approx(90.0, abs=1e-6)


@pytest.mark.solver
@pytest.mark.parametrize("reveal", ["p1", "p2"])
def test_at_reveal_period_has_no_realization(tmp_path_factory, caplog, reveal) -> None:
    """``investments_at_reveal_period``: every forecast invests on its own from the reveal on, so nothing after it
    can be realized (the solve realizes the periods before the reveal only) and no realization solve is added.  With
    the reveal at the first period (nothing realized) one W-0639 line names the solve; with a later reveal (p1
    realized, the "act, then learn" structure) none."""
    caplog.set_level(logging.WARNING)
    realized = [] if reveal == "p1" else ["p1"]
    built, _ = scm.run_spec(_spec(reveal=reveal, method="investments_at_reveal_period", realized=realized,
                                  realized_invest=realized), tmp_path_factory.mktemp(f"at_reveal_{reveal}"))
    assert [s for s, _, _ in built] == ["st"]
    expected = (["Warning: investment solve 'st': investments branch at its first period (expert option) [W-0639]"]
                if reveal == "p1" else [])
    assert _w0639(caplog) == expected
