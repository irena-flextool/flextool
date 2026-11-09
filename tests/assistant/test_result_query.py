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
from flextool.lean_parquet import lean_parquet_layout, read_lean_parquet_wide, to_long

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
                 _frame({("west",): [30.0, 40.0, 50.0, 60.0, 70.0, 80.0]}, solve_index, ["node"]), "base")
    # nodeGroup_gdt_p: '1. Loss of load' [MWh per step], '8. VRE share of demand' [ratio]; rows (group, period, t).
    group_index = pd.MultiIndex.from_tuples([("g1", p, t) for p in PERIODS for t in TIMES],
                                            names=["group", "period", "time"])
    write_result(folder, "nodeGroup_gdt_p",
                 _frame({("1. Loss of load",): [0.0, 2.0, 3.0, 0.0, 0.0, 1.0],
                         ("8. VRE share of demand",): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}, group_index, ["parameter"]),
                 "base")
    write_membership(folder, "group_node", [("g1", "west"), ("g1", "east"), ("g2", "north")], ["group", "node"],
                     "base")


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
    """The same statistics computed the slow way (every value as a row, then group_by) agree with the tool."""
    path = results_project / "output_parquet" / "base_1" / "unit_outputNode_dt_ee.parquet"
    layout = lean_parquet_layout(path)
    long = to_long(read_lean_parquet_wide(path, layout=layout), layout).filter(pl.col("value").is_not_null())
    for group_by, agg in ((["period", "node"], "mean"), (["period", "time", "node"], "sum"),
                          (["unit"], "max"), (["period"], "count"), (["node", "time"], "min")):
        out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=group_by, agg=agg)
        expected = long.group_by(group_by).agg(getattr(pl.col("value"), agg)().cast(pl.Float64))
        expected_rows = sorted(tuple(r) for r in expected.select([*group_by, "value"]).iter_rows())
        got = sorted(tuple(r) for r in out["rows"])
        assert [g[:-1] for g in got] == [e[:-1] for e in expected_rows], (group_by, agg)
        assert [g[-1] for g in got] == pytest.approx([e[-1] for e in expected_rows], rel=1e-9), (group_by, agg)
    # By hand, group_by [period, node] mean: p2020 west (10+20+30+5+0+15)/6 = 13.333…; p2020 east (1+2+3)/3 = 2;
    # p2025 west (40+50+60+0+0+0)/6 = 25; p2025 east (4+5)/2 = 4.5 (the empty cell does not count).
    out = _call(rctx, "query_result", run="base_1", table="unit_outputNode_dt_ee", group_by=["period", "node"],
                agg="mean")
    assert out["columns"] == ["period", "node", "value"]
    assert _rows(out) == [("p2020", "west", 13.33333333), ("p2020", "east", 2.0), ("p2025", "west", 25.0),
                          ("p2025", "east", 4.5)]
    assert out["unit"] == "MW (default convention)" and out["semantics"] == "instantaneous"
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
    assert out["unit"] == "M CUR/a (default convention)"
    assert _refused(rctx, "query_result", run="base_1", table="annualized_costs_d_p", group_by=[],
                    agg="sum").startswith("These are annual values; adding them across periods")
    # Prices do not add up; their mean over time and nodes is allowed: p2020 (30+40+50)/3 = 40, p2025 70.
    assert "prices" in _refused(rctx, "query_result", run="base_1", table="node_prices_dt_e", group_by=["period"],
                                agg="sum")
    out = _call(rctx, "query_result", run="base_1", table="node_prices_dt_e", group_by=["period"], agg="mean")
    assert _rows(out) == [("p2020", 40.0), ("p2025", 70.0)]


def test_mixed_unit_tables(rctx) -> None:
    message = _refused(rctx, "query_result", run="base_1", table="nodeGroup_gdt_p", group_by=["group", "period"],
                       agg="mean")
    assert "different units" in message and "parameter" in message
    # Loss of load is MWh per step: its sum over time is allowed (p2020 0 + 2 + 3 = 5, p2025 0 + 0 + 1 = 1).
    out = _call(rctx, "query_result", run="base_1", table="nodeGroup_gdt_p",
                filters={"parameter": ["1. Loss of load"]}, group_by=["group", "period"], agg="sum")
    assert _rows(out) == [("g1", "p2020", 5.0), ("g1", "p2025", 1.0)]
    assert out["unit"] == "MWh (default convention)" and out["semantics"] == "per_step"
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
    assert listed["total"] == 7
    names = {t["table"] for tables in listed["topics"].values() for t in tables}
    assert names == {"unit_outputNode_dt_ee", "unit_outputNode_d_ee", "unit_online_average_d_e",
                     "annualized_costs_d_p", "node_prices_dt_e", "nodeGroup_gdt_p", "group_node"}
    hits = _call(rctx, "find_result_tables", query="nodal energy price")["tables"]
    assert hits[0]["table"] == "node_prices_dt_e" and hits[0]["unit"] == "CUR/MWh (default convention)"
    message = _refused(rctx, "query_result", run="base_1", table="unit_outputNode_dt")
    assert message.startswith('There is no result table "unit_outputNode_dt" in "base_1".')


def test_compare_runs_with_a_missing_key(rctx) -> None:
    out = _call(rctx, "compare_runs", table="unit_outputNode_d_ee", runs=["base_1", "high_1"])
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
                filters={"unit": ["coal_plant"]})
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
    # Unit i produces i MW in every step, so the mean of the west units (even i) is the mean of 0, 2, …, 598 = 299
    # and of the east units (odd i) the mean of 1, 3, …, 599 = 300.
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
    assert _rows(out) == [("p2020", "west", 299.0), ("p2020", "east", 300.0)]
    assert unpivoted and max(unpivoted) <= 2
