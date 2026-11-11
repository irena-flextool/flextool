"""The result tools on synthetic result tables: wide aggregation gives what the long form gives, filters, the
``*`` wildcard, ``time_range``, ``group_by`` / ``agg``, ``top_n``, the per-series summary, the budget message, the
aggregation guard (a rate summed over time is refused and names the annual table; an average summed over entities
is allowed), compare_runs with a key one run lacks, and a large wide table that is aggregated without being
expanded to rows.

The tables are written as ``write_outputs`` writes them (``results_tables.write_result``); every expected value
below is computed by hand from the numbers in :data:`OUTPUT_DT` and the other frames, as the comments show.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from flextool.assistant.context import make_context
from flextool.assistant.registry import call_tool
from flextool.lean_parquet import lean_parquet_layout, read_lean_parquet_wide, to_long, write_lean_parquet

from tests.assistant.conftest import make_project, record_run
from tests.assistant.results_tables import (
    PERIODS,
    TIMES,
    dt_index,
    succeed_writing,
    write_membership,
    write_result,
)

# unit_outputNode_dt_ee of run base_1 [MW] — (unit, node): p2020 t01..t03, p2025 t01..t03 (None = empty).
OUTPUT_DT = {
    ("coal_plant", "west"): [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    ("wind_plant", "west"): [5.0, 0.0, 15.0, 0.0, 0.0, 0.0],
    ("gas_plant", "east"): [1.0, 2.0, 3.0, 4.0, 5.0, None],
}
# unit_outputNode_d_ee [MWh/a] of base_1 and high_1 (high_1 has no gas_plant).
OUTPUT_D_BASE = {("coal_plant", "west"): [1000.0, 2000.0], ("wind_plant", "west"): [300.0, 0.0],
                 ("gas_plant", "east"): [50.0, 60.0]}
OUTPUT_D_HIGH = {("coal_plant", "west"): [2000.0, 4000.0], ("wind_plant", "west"): [300.0, 0.0]}


def _frame(data: dict, index: pd.Index, names: list[str]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_tuples(list(data), names=names) if len(names) > 1 else pd.Index(
        [k[0] if isinstance(k, tuple) else k for k in data], name=names[0])
    return pd.DataFrame(np.array([[np.nan if v is None else v for v in vals] for vals in data.values()]).T,
                        index=index, columns=columns)


def _base_tables(folder: Path) -> None:
    period_index = pd.Index(PERIODS, name="period")
    write_result(folder, "unit_outputNode_dt_ee", _frame(OUTPUT_DT, dt_index(), ["unit", "node"]), "base")
    write_result(folder, "unit_outputNode_d_ee", _frame(OUTPUT_D_BASE, period_index, ["unit", "node"]), "base")
    # unit_online_average_d_e [units, average over the period].
    write_result(folder, "unit_online_average_d_e",
                 _frame({("coal_plant",): [0.5, 1.0], ("gas_plant",): [0.25, 0.0]}, period_index, ["process"]),
                 "base")
    # annualized_costs_d_p [M CUR/a].
    write_result(folder, "annualized_costs_d_p",
                 _frame({("investment",): [100.0, 0.0], ("upward slack penalty",): [2.5, 0.0]}, period_index,
                        ["category"]), "base")
    # node_prices_dt_e [CUR/MWh], rows (solve, period, time).
    solve_index = pd.MultiIndex.from_tuples([("s1", p, t) for p in PERIODS for t in TIMES],
                                            names=["solve", "period", "time"])
    write_result(folder, "node_prices_dt_e",
                 _frame({("west",): [30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                         ("east",): [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}, solve_index, ["node"]), "base")
    # nodeGroup_gdt_p: '1. Loss of load' [MWh per step], '8. VRE share of demand' [ratio]; rows (group, period, t).
    group_index = pd.MultiIndex.from_tuples([("g1", p, t) for p in PERIODS for t in TIMES],
                                            names=["group", "period", "time"])
    write_result(folder, "nodeGroup_gdt_p",
                 _frame({("1. Loss of load",): [0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
                         ("8. VRE share of demand",): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, group_index, ["parameter"]),
                 "base")
    write_membership(folder, "group_node", [("g1", "west"), ("g1", "east"), ("g2", "north")], ["group", "node"],
                     "base")
    # unit_capacity_ed_p [MW], rows (unit, period), columns parameter: five different quantities.
    capacity_index = pd.MultiIndex.from_tuples([("u1", "p2020"), ("u2", "p2020")], names=["unit", "period"])
    write_result(folder, "unit_capacity_ed_p",
                 _frame({("existing",): [100.0, 200.0], ("invested",): [50.0, 0.0], ("in_construction",): [0.0, 0.0],
                         ("divested",): [0.0, 0.0], ("total",): [150.0, 200.0]}, capacity_index, ["parameter"]),
                 "base")
    # CO2_d_g [t/a]: two groups that can share members.
    write_result(folder, "CO2_d_g", _frame({("g1",): [10.0, 12.0], ("g2",): [5.0, 6.0]}, period_index, ["group"]),
                 "base")
    # costs_dt_p [CUR per step]: commodity cost 1..6, CO2 0.5 in every step.
    write_result(folder, "costs_dt_p",
                 _frame({("commodity cost",): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ("co2 cost",): [0.5] * 6}, dt_index(),
                        ["category"]), "base")


def _high_tables(folder: Path) -> None:
    high_dt = {("coal_plant", "west"): [20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
               ("wind_plant", "west"): [5.0, 0.0, 15.0, 0.0, 0.0, 0.0]}
    write_result(folder, "unit_outputNode_dt_ee", _frame(high_dt, dt_index(), ["unit", "node"]), "high")
    write_result(folder, "unit_outputNode_d_ee",
                 _frame(OUTPUT_D_HIGH, pd.Index(PERIODS, name="period"), ["unit", "node"]), "high")


@pytest.fixture(scope="module")
def results_project(tmp_path_factory: pytest.TempPathFactory, tests_db: Path) -> Path:
    project = make_project(tmp_path_factory.mktemp("results"), tests_db)
    record_run(project, "base_1", "base", succeed_writing(_base_tables))
    record_run(project, "high_1", "high", succeed_writing(_high_tables))
    return project


@pytest.fixture
def rctx(results_project: Path):
    return make_context(results_project)


def _call(ctx, tool: str, **args) -> dict:
    out = call_tool(ctx, tool, args)
    assert out.ok, out.message
    return json.loads(out.text)


def _refused(ctx, tool: str, **args) -> str:
    out = call_tool(ctx, tool, args)
    assert not out.ok, out.text
    return out.message


def _rows(out: dict) -> list[tuple]:
    return [tuple(r) for r in out["rows"]]


def test_wide_aggregation_equals_the_long_form(rctx, results_project: Path) -> None:
    """The same statistics computed the slow way (every value as a row, then group_by) agree with the tool.  A
    mean, min or max over units and time (or period) is that of the units' total at each step: the long form
    adds up over the dropped entity levels per step first."""
    path = results_project / "output_parquet" / "base_1" / "unit_outputNode_dt_ee.parquet"
    layout = lean_parquet_layout(path)
    long = to_long(read_lean_parquet_wide(path, layout=layout), layout).filter(pl.col("value").is_not_null())
    for group_by, agg in ((["period", "node"], "mean"), (["period", "time", "node"], "sum"),
                          (["unit"], "max"), (["period"], "count"), (["node", "time"], "min")):
        out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=group_by, agg=agg)
        timed = [level for level in ("period", "time") if level not in group_by]
        source = long
        if agg in ("mean", "min", "max") and timed:
            source = long.group_by([*group_by, *timed]).agg(pl.col("value").sum())
        expected = source.group_by(group_by).agg(getattr(pl.col("value"), agg)().cast(pl.Float64))
        expected_rows = sorted(tuple(r) for r in expected.select([*group_by, "value"]).iter_rows())
        got = sorted(tuple(r) for r in out["rows"])
        assert [g[:-1] for g in got] == [e[:-1] for e in expected_rows], (group_by, agg)
        assert [g[-1] for g in got] == pytest.approx([e[-1] for e in expected_rows], rel=1e-9), (group_by, agg)
    # By hand, group_by [period, node] mean: the mean over time of the units' sum at each step.  p2020 west
    # (10+5, 20+0, 30+15) = 15, 20, 45 → 80/3 = 26.667; p2020 east (1+2+3)/3 = 2; p2025 west (40, 50, 60) → 50;
    # p2025 east (4+5)/2 = 4.5 (the empty step does not count).
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period", "node"],
                agg="mean")
    assert out["columns"] == ["period", "node", "value"]
    assert _rows(out) == [("p2020", "west", 26.66666667), ("p2020", "east", 2.0), ("p2025", "west", 50.0),
                          ("p2025", "east", 4.5)]
    assert out["applied"]["how"].startswith("mean of the sum over unit: ")
    assert out["unit"] == "MW" and out["semantics"] == "instantaneous"
    assert out["applied"]["combined_over"] == ["entity", "time"]
    assert any("equal step lengths" in n for n in out["notes"])


def test_filters_wildcard_time_range_and_time_of_max(rctx) -> None:
    # coal* keeps coal_plant; t02..t03: p2020 max(20, 30) = 30, p2025 max(50, 60) = 60.
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", filters={"unit": ["coal*"]},
                time_range={"from": "t02", "to": "t03"}, group_by=["period", "unit"], agg="max")
    assert _rows(out) == [("p2020", "coal_plant", 30.0), ("p2025", "coal_plant", 60.0)]
    # wind_plant: p2020 maximum 15 at t03; p2025 all 0, the first maximum is at t01.
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", filters={"unit": ["wind_plant"]},
                group_by=["period", "unit", "node"], agg="time_of_max")
    assert out["columns"] == ["period", "unit", "node", "value", "time_of_max"]
    assert _rows(out) == [("p2020", "wind_plant", "west", 15.0, "t03"), ("p2025", "wind_plant", "west", 0.0, "t01")]
    message = _refused(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", filters={"unit": ["oil*"]})
    assert message.startswith('No value of the level "unit" of "unit_outputNode_dt_ee" matches the filter.')
    message = _refused(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", filters={"colour": ["x"]})
    assert 'no level "colour"' in message and '"period", "time", "unit", "node"' in message


def test_sum_over_entities_with_top_n(rctx) -> None:
    # Per step, all units: p2020 t01..t03 = 16, 22, 48; p2025 = 44, 55, 60 (gas_plant's empty t03 not counted).
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period", "time"],
                agg="sum", top_n=2)
    assert _rows(out) == [("p2025", "t03", 60.0), ("p2025", "t02", 55.0)]
    assert out["total_rows"] == 2 and out["applied"]["combined_over"] == ["entity"]


def test_per_series_summary_of_a_time_table(rctx) -> None:
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee")
    assert out["applied"]["mode"] == "summary"
    # No sum column: power summed over time is not energy.
    assert out["columns"] == ["period", "unit", "node", "mean", "min", "max", "count", "time_of_max"]
    # Largest |mean| first: coal p2025 (40+50+60)/3 = 50, coal p2020 20, wind p2020 (5+0+15)/3 = 6.667,
    # gas p2025 (4+5)/2 = 4.5, gas p2020 2, wind p2025 0.
    assert _rows(out) == [
        ("p2025", "coal_plant", "west", 50.0, 40.0, 60.0, 3.0, "t03"),
        ("p2020", "coal_plant", "west", 20.0, 10.0, 30.0, 3.0, "t03"),
        ("p2020", "wind_plant", "west", 6.666666667, 0.0, 15.0, 3.0, "t03"),
        ("p2025", "gas_plant", "east", 4.5, 4.0, 5.0, 2.0, "t02"),
        ("p2020", "gas_plant", "east", 2.0, 1.0, 3.0, 3.0, "t03"),
        ("p2025", "wind_plant", "west", 0.0, 0.0, 0.0, 3.0, "t01"),
    ]


def test_the_aggregation_guard(rctx) -> None:
    # Power summed over time: refused, naming the annual table (OUTPUT_DIMS counterpart).
    message = _refused(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period"],
                       agg="sum")
    assert message == ("This table holds power at each timestep. Its sum over time is not energy unless each step "
                       "is weighted by its duration, which the result files do not record. Use the annual table "
                       "unit_outputNode_d_ee instead.")
    # An average summed over entities is allowed: p2020 0.5 + 0.25 = 0.75, p2025 1.0 + 0 = 1.0 …
    out = _call(rctx, "query_result", run="base_1", table="unit_online_average_d_e", group_by=["period"],
                agg="sum")
    assert _rows(out) == [("p2020", 0.75), ("p2025", 1.0)]
    # … but not over periods.
    assert "levels or averages" in _refused(rctx, "query_result", run="base_1", table="unit_online_average_d_e",
                                            group_by=["process"], agg="sum")
    # Annual costs: summed over categories per period (100 + 2.5 = 102.5) yes, over periods no.
    out = _call(rctx, "query_result", run="base_1", table="annualized_costs_d_p", group_by=["period"], agg="sum")
    assert _rows(out) == [("p2020", 102.5), ("p2025", 0.0)]
    assert out["unit"] == "M CUR/a"
    assert _refused(rctx, "query_result", run="base_1", table="annualized_costs_d_p", group_by=[],
                    agg="sum").startswith("These are annual values; adding them across periods")
    # Prices do not add up; their mean over time and nodes is the mean of the single values: p2020
    # (30+40+50+10+20+30)/6 = 30, p2025 (60+70+80+40+50+60)/6 = 60.
    assert "prices" in _refused(rctx, "query_result", run="base_1", table="node_prices_dt_e", group_by=["period"],
                                agg="sum")
    out = _call(rctx, "query_result", run="base_1", table="node_prices_dt_e", group_by=["period"], agg="mean")
    assert _rows(out) == [("p2020", 30.0), ("p2025", 60.0)]
    assert out["applied"]["how"] == "mean of the single values over time, node; these values do not add up over node"


def test_capacity_parameters_are_not_combined(rctx) -> None:
    """existing, invested, in_construction, divested and total are different quantities: no statistic combines
    them (before: sum 100+50+150+200+200 = 700, mean 700/10 = 70, max 200)."""
    for agg in ("sum", None, "max"):
        args = {"agg": agg} if agg else {}
        message = _refused(rctx, "query_result", run="base_1", table="unit_capacity_ed_p", group_by=["period"],
                           **args)
        assert message.startswith('The level "parameter" of "unit_capacity_ed_p" holds different quantities ('), agg
        assert '"existing"' in message and '"total"' in message
        assert message.endswith("or keep the level and add the values you need yourself."), agg
    # One quantity is added up over units: total 150 + 200 = 350.
    out = _call(rctx, "query_result", run="base_1", table="unit_capacity_ed_p", filters={"parameter": ["total"]},
                group_by=["period"], agg="sum")
    assert _rows(out) == [("p2020", 350.0)] and out["applied"]["combined_over"] == ["entity"]
    # Keeping parameter: each quantity summed over units (existing 100 + 200, invested 50 + 0, …).
    out = _call(rctx, "query_result", run="base_1", table="unit_capacity_ed_p", group_by=["period", "parameter"],
                agg="sum")
    assert _rows(out) == [("p2020", "existing", 300.0), ("p2020", "invested", 50.0),
                          ("p2020", "in_construction", 0.0), ("p2020", "divested", 0.0), ("p2020", "total", 350.0)]


def test_cost_categories_add_up_but_have_no_mean(rctx) -> None:
    out = _call(rctx, "query_result", run="base_1", table="annualized_costs_d_p", group_by=["period"], agg="sum")
    assert _rows(out) == [("p2020", 102.5), ("p2025", 0.0)] and out["applied"]["combined_over"] == ["category"]
    # The mean across categories (before: 102.5 / 2 = 51.25) is refused …
    message = _refused(rctx, "query_result", run="base_1", table="annualized_costs_d_p", group_by=["period"],
                       agg="mean")
    assert message == ('The "category" values of "annualized_costs_d_p" are parts of one total; their mean is not a '
                       'meaningful figure. Use sum, or keep "category" in group_by.')
    # … the mean over periods of the total is not: (102.5 + 0) / 2 = 51.25.
    out = _call(rctx, "query_result", run="base_1", table="annualized_costs_d_p", group_by=[], agg="mean")
    assert _rows(out) == [(51.25,)]
    assert out["applied"]["how"].startswith("mean of the sum over category: ")


def test_overlapping_groups_are_not_added_up(rctx) -> None:
    # g1 and g2 can share members: their sum (before: 10 + 5 = 15) is refused; mean, max, count are fine.
    message = _refused(rctx, "query_result", run="base_1", table="CO2_d_g", group_by=["period"], agg="sum")
    assert message.startswith('Groups of "CO2_d_g" can share members, so adding them up may count a member more')
    out = _call(rctx, "query_result", run="base_1", table="CO2_d_g", group_by=["period"], agg="max")
    assert _rows(out) == [("p2020", 10.0), ("p2025", 12.0)] and out["applied"]["combined_over"] == ["group"]
    out = _call(rctx, "query_result", run="base_1", table="CO2_d_g", group_by=["period"], agg="mean")
    assert _rows(out) == [("p2020", 7.5), ("p2025", 9.0)]
    out = _call(rctx, "query_result", run="base_1", table="CO2_d_g", group_by=["period"], agg="count")
    assert _rows(out) == [("p2020", 2.0), ("p2025", 2.0)]
    out = _call(rctx, "query_result", run="base_1", table="CO2_d_g", filters={"group": ["g1"]}, group_by=[],
                agg="max")
    assert _rows(out) == [(12.0,)]


def test_peak_of_the_total_over_units_and_time(rctx) -> None:
    """Max over units and time is the peak of the units' sum per step, not the largest single value.  Per step,
    all units: p2020 16, 22, 48; p2025 44, 55, 60 (gas_plant empty at t03).  The largest single values were
    p2020 30 (coal t03) and p2025 60; the smallest single values 0 (wind)."""
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period"], agg="max")
    assert _rows(out) == [("p2020", 48.0), ("p2025", 60.0)]
    assert out["applied"]["how"] == ("peak of the sum over unit, node: the values are added up over unit, node at "
                                     "each time step first, then the max is taken over time")
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period"], agg="min")
    assert _rows(out) == [("p2020", 16.0), ("p2025", 44.0)]
    assert out["applied"]["how"].startswith("lowest value of the sum over unit, node: ")
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period"],
                agg="time_of_max")
    assert _rows(out) == [("p2020", 48.0, "t03"), ("p2025", 60.0, "t03")]
    # Mean over time of the sum: p2020 (16+22+48)/3 = 28.667, p2025 (44+55+60)/3 = 53.
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period"], agg="mean")
    assert _rows(out) == [("p2020", 28.66666667), ("p2025", 53.0)]
    # Over units, time and periods: the peak 60 at (p2025, t03).
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=[], agg="time_of_max")
    assert out["columns"] == ["value", "period_of_max", "time_of_max"]
    assert _rows(out) == [(60.0, "p2025", "t03")]
    # Keeping the unit gives each unit's own peak.
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period", "unit"],
                agg="max")
    assert _rows(out)[:3] == [("p2020", "coal_plant", 30.0), ("p2020", "wind_plant", 15.0),
                              ("p2020", "gas_plant", 3.0)]
    assert out["applied"]["how"] is None
    # Prices do not add up over nodes: the largest single price, p2020 50 (west t03), p2025 80.
    out = _call(rctx, "query_result", run="base_1", table="node_prices_dt_e", group_by=["period"], agg="max")
    assert _rows(out) == [("p2020", 50.0), ("p2025", 80.0)]
    assert out["applied"]["how"] == "largest single value over time, node; these values do not add up over node"


def test_per_step_values_are_not_summed_over_periods(rctx) -> None:
    # Before: 21 + 6 × 0.5 = 22 over both periods.
    message = _refused(rctx, "query_result", run="base_1", table="costs_dt_p", group_by=[], agg="sum")
    assert message == ("These values are sums over the sampled time steps of each period, not scaled to the years "
                       "a period represents; adding them across periods does not give a total over the horizon. "
                       "Compare periods side by side. Use the annual table annualized_costs_d_p instead.")
    # Per period: p2020 1+2+3 + 3 × 0.5 = 7.5, p2025 4+5+6 + 1.5 = 16.5.
    out = _call(rctx, "query_result", run="base_1", table="costs_dt_p", group_by=["period"], agg="sum")
    assert _rows(out) == [("p2020", 7.5), ("p2025", 16.5)]
    assert any("not scaled to a full year" in n for n in out["notes"])


def test_mixed_unit_tables(rctx) -> None:
    message = _refused(rctx, "query_result", run="base_1", table="nodeGroup_gdt_p", group_by=["group", "period"],
                       agg="mean")
    assert "different units" in message and "parameter" in message
    # Loss of load is MWh per step: its sum over time is allowed (p2020 0 + 2 + 3 = 5, p2025 0 + 0 + 1 = 1).
    out = _call(rctx, "query_result", run="base_1", table="nodeGroup_gdt_p",
                filters={"parameter": ["1. Loss of load"]}, group_by=["group", "period"], agg="sum")
    assert _rows(out) == [("g1", "p2020", 5.0), ("g1", "p2025", 1.0)]
    assert out["unit"] == "MWh" and out["semantics"] == "per_step"
    assert any("not scaled to a full year" in n for n in out["notes"])


def test_values_of_a_period_table_and_a_membership_set(rctx) -> None:
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_d_ee", filters={"node": ["west"]})
    assert out["applied"]["mode"] == "values"
    assert sorted(_rows(out)) == [("p2020", "coal_plant", "west", 1000.0), ("p2020", "wind_plant", "west", 300.0),
                                  ("p2025", "coal_plant", "west", 2000.0), ("p2025", "wind_plant", "west", 0.0)]
    out = _call(rctx, "query_result", run="base_1", table="group_node", filters={"group": ["g1"]})
    assert out["columns"] == ["group", "node"] and _rows(out) == [("g1", "west"), ("g1", "east")]
    assert "set of names" in _refused(rctx, "query_result", run="base_1", table="group_node", group_by=["group"])


def test_describe_list_and_find(rctx) -> None:
    out = _call(rctx, "describe_result_table", run="base_1", table="unit_outputNode_dt_ee")
    assert out["levels"]["unit"] == {"axis": "column", "distinct": 3, "first": ["coal_plant", "wind_plant",
                                                                                "gas_plant"]}
    assert out["levels"]["time"]["distinct"] == 3 and out["time"] == {"steps": 3, "first": "t01", "last": "t03"}
    assert out["periods"] == ["p2020", "p2025"] and out["rows"] == 6 and out["data_columns"] == 3
    assert out["counterpart"] == "unit_outputNode_d_ee"
    listed = _call(rctx, "list_result_tables", run="base_1")
    assert listed["total"] == 10
    names = {t["table"] for tables in listed["topics"].values() for t in tables}
    assert names == {"unit_outputNode_dt_ee", "unit_outputNode_d_ee", "unit_online_average_d_e",
                     "annualized_costs_d_p", "node_prices_dt_e", "nodeGroup_gdt_p", "group_node",
                     "unit_capacity_ed_p", "CO2_d_g", "costs_dt_p"}
    hits = _call(rctx, "find_result_tables", query="nodal energy price")["tables"]
    assert hits[0]["table"] == "node_prices_dt_e" and hits[0]["unit"] == "CUR/MWh"
    message = _refused(rctx, "query_result", run="base_1", table="unit_outputNode_dt")
    assert message.startswith('There is no result table "unit_outputNode_dt" in "base_1".')


def test_compare_runs_with_a_missing_key(rctx) -> None:
    out = _call(rctx, "compare_runs", table="unit_outputNode_d_ee", runs=["base_1", "high_1"], by_difference=True)
    assert out["columns"] == ["period", "unit", "node", "base_1", "high_1", "difference_vs_first"]
    # Largest difference first: coal p2025 4000 − 2000, coal p2020 2000 − 1000, wind 0 and 0; gas_plant is only in
    # base_1 (no difference, last).
    assert _rows(out) == [
        ("p2025", "coal_plant", "west", 2000.0, 4000.0, 2000.0),
        ("p2020", "coal_plant", "west", 1000.0, 2000.0, 1000.0),
        ("p2020", "wind_plant", "west", 300.0, 300.0, 0.0),
        ("p2025", "wind_plant", "west", 0.0, 0.0, 0.0),
        ("p2020", "gas_plant", "east", 50.0, None, None),
        ("p2025", "gas_plant", "east", 60.0, None, None),
    ]
    assert out["missing"] == {"high_1": "2 keys without a value"} and out["source"] == "run_folders"
    # A time table: each series' mean per period (power: the mean, not a sum): coal p2025 (80+100+120)/3 = 100
    # against 50.
    out = _call(rctx, "compare_runs", table="unit_outputNode_dt_ee", runs=["base_1", "high_1"],
                filters={"unit": ["coal_plant"]}, by_difference=True)
    assert _rows(out) == [("p2025", "coal_plant", "west", 50.0, 100.0, 50.0),
                          ("p2020", "coal_plant", "west", 20.0, 40.0, 20.0)]
    assert _refused(rctx, "compare_runs", table="unit_outputNode_d_ee", runs=["base_1"]).startswith(
        "compare_runs needs two or more result folders")


def test_budget_and_a_large_table_aggregated_without_expanding(tmp_path: Path, tests_db: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """600 series × 8760 time steps = 5 256 000 values, over the 5 000 000 budget: keeping every value is refused
    with the budget message; a mean per period and node is computed on the wide frame, and no step turns more
    values into rows than the reduced result has (2 nodes × 1 period)."""
    n_units, n_steps = 600, 8760
    times = [f"t{i:04d}" for i in range(1, n_steps + 1)]
    index = pd.MultiIndex.from_tuples([("p2020", t) for t in times], names=["period", "time"])
    columns = pd.MultiIndex.from_tuples([(f"u{i:03d}", "west" if i % 2 == 0 else "east") for i in range(n_units)],
                                        names=["unit", "node"])
    # Unit i produces i MW in every step, so the west units (even i) produce 0 + 2 + … + 598 = 89 700 MW together
    # in every step and the east units (odd i) 1 + 3 + … + 599 = 90 000 MW: the mean over time of the sum.
    values = np.tile(np.arange(n_units, dtype=float), (n_steps, 1))

    def tables(folder: Path) -> None:
        write_result(folder, "unit_outputNode_dt_ee", pd.DataFrame(values, index=index, columns=columns), "big")

    project = make_project(tmp_path, tests_db)
    record_run(project, "big_1", "big", succeed_writing(tables))
    ctx = make_context(project)
    message = _refused(ctx, "query_result", run="big_1", table="unit_outputNode_dt_ee",
                       group_by=["period", "time", "unit", "node"], agg="max")
    assert message == "This would return about 5 256 000 values. Add filters, group_by or top_n."

    unpivoted: list[int] = []
    real = pl.DataFrame.unpivot

    def counting(self, *args, **kwargs):
        out = real(self, *args, **kwargs)
        unpivoted.append(out.height)
        return out

    monkeypatch.setattr(pl.DataFrame, "unpivot", counting)
    out = _call(ctx, "query_result", run="big_1", table="unit_outputNode_dt_ee", group_by=["period", "node"],
                agg="mean")
    assert _rows(out) == [("p2020", "west", 89700.0), ("p2020", "east", 90000.0)]
    assert unpivoted and max(unpivoted) <= 2


def test_a_table_with_a_row_and_a_column_level_of_one_name(tmp_path: Path, tests_db: Path) -> None:
    def tables(folder: Path) -> None:
        frame = pd.DataFrame([[1.0]], index=pd.Index(["east"], name="node"),
                             columns=pd.MultiIndex.from_tuples([("clash", "west")], names=["scenario", "node"]))
        folder.mkdir(parents=True, exist_ok=True)
        write_lean_parquet(frame, folder / "node_state_dt_e.parquet")

    project = make_project(tmp_path, tests_db)
    record_run(project, "clash_1", "clash", succeed_writing(tables))
    ctx = make_context(project)
    for args in ({}, {"group_by": ["node"], "agg": "max"}):
        message = _refused(ctx, "query_result", run="clash_1", table="node_state_dt_e", **args)
        assert message == ('The table "node_state_dt_e" uses the name "node" for a row level and a column level, so '
                           'it cannot be queried.')


def _overview_tables(scale: dict) -> callable:
    """The tables of one run of the overview test; *scale* holds that run's numbers."""
    def write(folder: Path) -> None:
        period_index = pd.Index(PERIODS, name="period")
        write_result(folder, "unit_outputNode_dt_ee", _frame(scale["dt"], dt_index(), ["unit", "node"]),
                     scale["scenario"])
        rows = pd.MultiIndex.from_tuples([(p, u) for p in PERIODS for u in ("u1", "u2")], names=["period", "unit"])
        write_result(folder, "cost_unit_annualized_d_ec", _frame(scale["cost"], rows, ["category"]),
                     scale["scenario"])
        write_result(folder, "costs_discounted_d_p", _frame(scale["discounted"], period_index, ["category"]),
                     scale["scenario"])
        solve_index = pd.MultiIndex.from_tuples([("s1", p, t) for p in PERIODS for t in TIMES],
                                                names=["solve", "period", "time"])
        write_result(folder, "node_prices_dt_e", _frame(scale["prices"], solve_index, ["node"]), scale["scenario"])
        write_result(folder, "connection_dt_eee", _frame(scale["connections"], dt_index(), ["process"]),
                     scale["scenario"])
    return write


# Rows of cost_unit_annualized_d_ec: (p2020, u1), (p2020, u2), (p2025, u1), (p2025, u2).
OVERVIEW_BASE = {"scenario": "base", "dt": OUTPUT_DT,
                 "cost": {("commodity",): [10.0, 5.0, 20.0, 6.0], ("investment",): [100.0, 0.0, 0.0, 50.0]},
                 "discounted": {("commodity",): [3.0, 2.0], ("investment",): [40.0, 10.0]},
                 "prices": {("west",): [30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
                            ("east",): [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]},
                 "connections": {("c1",): [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ("c2",): [10.0] * 6}}
OVERVIEW_HIGH = {"scenario": "high",
                 "dt": {("coal_plant", "west"): [20.0, 40.0, 60.0, 80.0, 100.0, 120.0],
                        ("wind_plant", "west"): [5.0, 0.0, 15.0, 0.0, 0.0, 0.0]},
                 "cost": {("commodity",): [12.0, 5.0, 25.0, 6.0], ("investment",): [100.0, 10.0, 0.0, 50.0]},
                 "discounted": {("commodity",): [4.0, 2.0], ("investment",): [40.0, 20.0]},
                 "prices": {("west",): [40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
                            ("east",): [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]},
                 "connections": {("c1",): [2.0] * 6, ("c2",): [10.0] * 6}}


def test_compare_runs_overview(tmp_path: Path, tests_db: Path) -> None:
    """Agent feedback waste 2: without group_by, agg, sort, top_n and by_difference, each run's totals per period
    and category, in key order (not by difference); entities added up where that is meaningful.  Before: one row
    per series, largest difference first."""
    project = make_project(tmp_path, tests_db)
    record_run(project, "base_1", "base", succeed_writing(_overview_tables(OVERVIEW_BASE)))
    record_run(project, "high_1", "high", succeed_writing(_overview_tables(OVERVIEW_HIGH)))
    ctx = make_context(project)
    runs = ["base_1", "high_1"]

    # Annualized costs per (period, category), units added up: p2020 commodity 10 + 5 = 15 against 12 + 5 = 17;
    # investment 100 + 0 against 100 + 10; p2025 commodity 20 + 6 against 25 + 6; investment 0 + 50 both.
    out = _call(ctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=runs)
    assert out["columns"] == ["period", "category", "base_1", "high_1", "difference_vs_first"]
    assert _rows(out) == [("p2020", "commodity", 15.0, 17.0, 2.0), ("p2020", "investment", 100.0, 110.0, 10.0),
                          ("p2025", "commodity", 26.0, 31.0, 5.0), ("p2025", "investment", 50.0, 50.0, 0.0)]
    # Totals per period (categories added up too): 115 against 127, 76 against 81.
    assert out["totals"] == {"columns": ["period", "base_1", "high_1", "difference_vs_first"],
                             "rows": [["p2020", 115.0, 127.0, 12.0], ["p2025", 76.0, 81.0, 5.0]]}
    assert out["applied"]["mode"] == "overview" and out["applied"]["group_by"] == ["period", "category"]
    assert out["notes"][0] == ("Totals per period, category (sum over unit). by_difference true gives each "
                               "series, largest difference first.")

    # Discounted costs add up over periods too: commodity 3 + 2 against 4 + 2, investment 40 + 10 against 40 + 20;
    # in total 55 against 66.
    out = _call(ctx, "compare_runs", table="costs_discounted_d_p", runs=runs)
    assert _rows(out) == [("commodity", 5.0, 6.0, 1.0), ("investment", 50.0, 60.0, 10.0)]
    assert out["totals"]["rows"] == [[55.0, 66.0, 11.0]]

    # Prices are not added up over nodes: the mean over time per period and node.  west p2020 (30+40+50)/3 = 40
    # against 50; east 20 both; west p2025 70 against 80; east 50 both.
    out = _call(ctx, "compare_runs", table="node_prices_dt_e", runs=runs)
    assert _rows(out) == [("p2020", "west", 40.0, 50.0, 10.0), ("p2020", "east", 20.0, 20.0, 0.0),
                          ("p2025", "west", 70.0, 80.0, 10.0), ("p2025", "east", 50.0, 50.0, 0.0)]
    assert "totals" not in out
    assert "The values are not added up over node (they are prices or ratios); each node is shown." in out["notes"]

    # Power per node (review M3: nodes may hold different carriers, so power is not added up over them): the
    # mean over time of the node's units' sum at each step.  base west p2020 (15 + 20 + 45)/3 = 26.67, p2025
    # (40 + 50 + 60)/3 = 50; base east (gas_plant) p2020 (1 + 2 + 3)/3 = 2, p2025 (4 + 5)/2 = 4.5 (no value at
    # the last step); high west (25 + 40 + 75)/3 = 46.67 and (80 + 100 + 120)/3 = 100; high has no east.
    out = _call(ctx, "compare_runs", table="unit_outputNode_dt_ee", runs=runs)
    assert out["columns"] == ["period", "node", "base_1", "high_1", "difference_vs_first"]
    assert _rows(out) == [("p2020", "west", pytest.approx(80 / 3), pytest.approx(140 / 3), pytest.approx(20.0)),
                          ("p2020", "east", 2.0, None, None), ("p2025", "west", 50.0, 100.0, 50.0),
                          ("p2025", "east", 4.5, None, None)]
    assert out["applied"]["agg"] == "mean" and out["applied"]["group_by"] == ["period", "node"]
    # Review M2: the first note says what was computed (the mean of the units' sum), not a mean over units.
    assert out["notes"][0] == ("Totals per period, node (mean of the sum over unit: the values are added up over "
                               "unit at each time step first, then the mean is taken over time). by_difference true "
                               "gives each series, largest difference first.")
    assert out["notes"][1] == ("Energy and power are not added up over node: a model's nodes may hold different "
                               "carriers (electricity, heat, hydrogen); each node is shown.")
    # Connections without a node level keep each connection: base c1 p2020 (1 + 2 + 3)/3 = 2, p2025 5; c2 10;
    # high c1 2, c2 10.
    out = _call(ctx, "compare_runs", table="connection_dt_eee", runs=runs)
    assert _rows(out) == [("p2020", "c1", 2.0, 2.0, 0.0), ("p2020", "c2", 10.0, 10.0, 0.0),
                          ("p2025", "c1", 5.0, 2.0, -3.0), ("p2025", "c2", 10.0, 10.0, 0.0)]
    assert out["notes"][0] == ("Totals per period, process (mean over time). by_difference true gives each series, "
                               "largest difference first.")
    assert out["notes"][1].startswith("Energy and power are not added up over process:")

    # by_difference: every series, largest difference first; group_by keeps that order too.
    out = _call(ctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=runs, by_difference=True)
    assert out["total_rows"] == 8 and _rows(out)[0] == ("p2020", "u2", "investment", 0.0, 10.0, 10.0)
    # Per (period, unit), categories added up: p2020 u1 110 against 112, u2 5 against 15; p2025 u1 20 against
    # 25, u2 56 both.
    out = _call(ctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=runs, group_by=["period", "unit"])
    assert out["applied"]["mode"] == "aggregate"
    assert _rows(out) == [("p2020", "u2", 5.0, 15.0, 10.0), ("p2025", "u1", 20.0, 25.0, 5.0),
                          ("p2020", "u1", 110.0, 112.0, 2.0), ("p2025", "u2", 56.0, 56.0, 0.0)]


@pytest.fixture
def overview_ctx(tmp_path: Path, tests_db: Path):
    project = make_project(tmp_path, tests_db)
    record_run(project, "base_1", "base", succeed_writing(_overview_tables(OVERVIEW_BASE)))
    record_run(project, "high_1", "high", succeed_writing(_overview_tables(OVERVIEW_HIGH)))
    return make_context(project)


def test_compare_runs_overview_totals_say_when_they_are_cut(overview_ctx) -> None:
    """Cycle 1 review m4: totals are cut at max_rows and to the result cap like the rows, and say so.  Before:
    cut at max_rows without a note, and never fitted (one long totals list could push out every row)."""
    from flextool.assistant.tools.results import _fit_totals

    runs = ["base_1", "high_1"]
    out = _call(overview_ctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=runs, max_rows=1)
    assert out["totals"] == {"columns": ["period", "base_1", "high_1", "difference_vs_first"],
                             "rows": [["p2020", 115.0, 127.0, 12.0]], "shown": "totals rows 1–1 of 2"}
    payload = {"table": "t", "rows": [], "totals": {"columns": ["period", "v"],
                                                     "rows": [[f"p{i:03d}", float(i)] for i in range(100)]}}
    def redact(text: str) -> str:
        return text

    fitted = _fit_totals(payload, 800, redact, 100)
    k = len(fitted["totals"]["rows"])
    assert 0 < k < 100 and fitted["totals"]["shown"] == f"totals rows 1–{k} of 100"
    assert len(json.dumps(fitted, ensure_ascii=False, separators=(",", ":"))) <= 800 - 200
    assert _fit_totals(payload, 100_000, redact, 100)["totals"] == payload["totals"]


def test_compare_runs_overview_says_when_it_falls_back(overview_ctx, monkeypatch) -> None:
    """m5: when both overview queries are refused, each series is shown with a note that says why; any other
    refusal (not an aggregation's) is the caller's at once.  Before: no note, and a refusal of any kind was
    taken as "totals not meaningful"."""
    from flextool.assistant.errors import ToolError
    from flextool.assistant.tools import results

    real = results._side_by_side
    calls: list[bool] = []

    def refusing(present, q, *, value_only):
        calls.append(value_only)
        if not value_only:
            raise ToolError("sum_ratio", what="ratios")
        return real(present, q, value_only=value_only)

    monkeypatch.setattr(results, "_side_by_side", refusing)
    out = _call(overview_ctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=["base_1", "high_1"])
    assert out["notes"][0] == ("Totals are not meaningful for this table (This table holds ratios; adding them "
                               "gives no meaningful total. Use mean, min or max.); each series is shown.")
    assert out["applied"]["mode"] != "overview" and out["total_rows"] == 8 and calls == [False, False, True]

    def not_found(present, q, *, value_only):
        calls.append(value_only)
        raise ToolError("no_runs")

    calls.clear()
    monkeypatch.setattr(results, "_side_by_side", not_found)
    message = _refused(overview_ctx, "compare_runs", table="cost_unit_annualized_d_ec", runs=["base_1", "high_1"])
    assert message.startswith("The project has no scenario results") and calls == [False]


def test_energy_units() -> None:
    """M3: power and energy units (of a node's carrier) against units that add up across carriers."""
    from flextool.assistant.unit_wording import is_energy_unit

    assert all(is_energy_unit(u) for u in ("MW", "MWh", "MWh/a", "MW·s", "MW·s/a", "kW", "GWh", "MVA"))
    assert not any(is_energy_unit(u) for u in ("M CUR", "M CUR/a", "CUR", "t/a", "Mt", "units", "units/a", "rad",
                                               "a", "", None))


def test_compare_runs_description_matches_its_rule() -> None:
    """Critique M2: compare_runs states its own rule for a call without group_by and agg, and not query_result's
    per-series summary; filters and time_range narrow the overview."""
    from flextool.assistant.registry import all_tools
    from flextool.assistant.tools import results

    compare = all_tools()["compare_runs"].description
    assert "Without group_by, agg, sort and top_n (filters and time_range narrow): each run's totals" in compare
    assert "by_difference true: each series" in compare
    assert "With by_difference, group_by or agg: largest difference first." in compare
    assert results._QUERY_SUMMARY not in compare
    assert results._QUERY_SUMMARY in all_tools()["query_result"].description
