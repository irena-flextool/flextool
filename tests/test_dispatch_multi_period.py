"""Comparison dispatch of a multi-period scenario (item 181).

The combined comparison frames are indexed ``(period, time)``.  Steps of the
same name in different periods must stay apart, in the order of the file, and
a scenario's frame must hold only its own steps (the combined row index is the
union of all scenarios' steps).

Data, one scenario, periods y2030 and y2050 with steps t1, t2:
pv = 10, 20 | 100, 200; slack = 1, 0 | 3, 0; curtailment = 0, 5 | 0, 50;
inflow = -11, -20 | -103, -200 (demand = -inflow = 11, 20, 103, 200).
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from flextool.scenario_comparison.data_models import (  # noqa: E402
    DispatchMappings,
    TimeSeriesResults,
)
from flextool.scenario_comparison.dispatch_data import (  # noqa: E402
    prepare_dispatch_data,
    prepare_node_dispatch_data,
)
from flextool.scenario_comparison.dispatch_plots import (  # noqa: E402
    _build_dispatch_figure,
    dispatch_plot_data,
)

NODE = "n"
GROUP = "g"
PV_COL = f"(pv, {NODE})"  # the group dispatch column of the pv flow into n
STEPS = [("y2030", "t1"), ("y2030", "t2"), ("y2050", "t1"), ("y2050", "t2")]


def _index(steps):
    return pd.MultiIndex.from_tuples(steps, names=["period", "time"])


def _frame(scenario, steps, key, values, names):
    cols = pd.MultiIndex.from_tuples([(scenario, *key)], names=["scenario", *names])
    return pd.DataFrame([[v] for v in values], index=_index(steps), columns=cols)


def _results(scenarios):
    """``scenarios``: {name: (steps, pv, slack, curtailment, inflow)}, combined side by side."""
    def combined(pos, key, names):
        return pd.concat(
            [_frame(sc, data[0], key, data[pos], names) for sc, data in scenarios.items()], axis=1,
        )
    res = TimeSeriesResults()
    res.unit_outputNode_dt_ee = combined(1, ("pv", NODE), ["e0", "e1"])
    res.node_slack_up_dt_e = combined(2, (NODE,), ["node"])
    res.unit_curtailment_outputNode_dt_ee = combined(3, ("pv", NODE), ["unit", "sink"])
    res.node_inflow__dt = combined(4, (NODE,), ["node"])
    return res


def _mappings(scenarios):
    def rows(row):
        return pd.DataFrame([{**row, "scenario": sc} for sc in scenarios]).set_index("scenario")
    m = DispatchMappings()
    m.dispatch_groups = rows({"group": GROUP})
    m.group_node = rows({"group": GROUP, "node": NODE})
    m.not_in_aggregate_unit_to_node = rows({"group": GROUP, "process": "pv", "unit": "pv", "node": NODE})
    return m


ONE = {"s": (STEPS, [10, 20, 100, 200], [1, 0, 3, 0], [0, 5, 0, 50], [-11, -20, -103, -200])}


def test_group_dispatch_keeps_periods_apart():
    df, demand = prepare_dispatch_data(_results(ONE), _mappings(ONE), "s", GROUP)
    assert list(df.index) == STEPS
    assert df[PV_COL].tolist() == [10, 20, 100, 200]
    assert df["LossOfLoad"].tolist() == [1, 0, 3, 0]
    assert df["Curtailed"].tolist() == [0, 5, 0, 50]
    assert list(demand.index) == STEPS
    assert demand.tolist() == [11, 20, 103, 200]


def test_node_dispatch_keeps_periods_apart():
    df, demand = prepare_node_dispatch_data(_results(ONE), "s", NODE)
    assert list(df.index) == STEPS
    assert df["pv_out"].tolist() == [10, 20, 100, 200]
    assert df["LossOfLoad"].tolist() == [1, 0, 3, 0]
    assert demand.tolist() == [11, 20, 103, 200]


def test_file_order_is_kept():
    # y2050 before y2030 in the file; t1 in both periods is not merged
    steps = [STEPS[2], STEPS[3], STEPS[0], STEPS[1]]
    data = {"s": (steps, [100, 200, 10, 20], [3, 0, 1, 0], [0, 50, 0, 5], [-103, -200, -11, -20])}
    df, demand = prepare_dispatch_data(_results(data), _mappings(data), "s", GROUP)
    assert list(df.index) == steps
    assert df[PV_COL].tolist() == [100, 200, 10, 20]
    assert list(demand.index) == steps
    df_n, demand_n = prepare_node_dispatch_data(_results(data), "s", NODE)
    assert list(df_n.index) == steps
    assert df_n["pv_out"].tolist() == [100, 200, 10, 20]


def test_scenario_frames_hold_only_their_own_steps():
    a_steps = [("y2050", "t1"), ("y2050", "t2")]
    b_steps = [("y2030", "t1"), ("y2030", "t2"), ("y2040", "t1"), ("y2040", "t2")]
    data = {
        "A": (a_steps, [7, 8], [0, 1], [2, 0], [-7, -9]),
        "B": (b_steps, [1, 2, 3, 4], [0, 0, 0, 1], [0, 1, 0, 0], [-1, -2, -3, -5]),
    }
    res, maps = _results(data), _mappings(data)
    for sc, (steps, pv, slack, curt, inflow) in data.items():
        df, demand = prepare_dispatch_data(res, maps, sc, GROUP)
        assert list(df.index) == steps, sc
        assert df[PV_COL].tolist() == pv
        assert df["LossOfLoad"].tolist() == slack
        assert df["Curtailed"].tolist() == curt
        assert list(demand.index) == steps
        assert demand.tolist() == [-x for x in inflow]
        df_n, demand_n = prepare_node_dispatch_data(res, sc, NODE)
        assert list(df_n.index) == steps, sc
        assert df_n["pv_out"].tolist() == pv
        assert list(demand_n.index) == steps


def test_figure_gap_before_the_period_that_breaks():
    df, demand = prepare_dispatch_data(_results(ONE), _mappings(ONE), "s", GROUP)
    breaks = _index([("y2050", "t1")])
    fig = _build_dispatch_figure(df, demand, "g", timeline=(0, 4), break_times=breaks)
    ax = fig.axes[0]
    line = next(ln for ln in ax.get_lines() if ln.get_label() == "Demand")
    y = np.asarray(line.get_ydata(), dtype=float)
    # 3 NaN gap rows before row 2 (the first step of y2050)
    assert len(y) == 7
    assert np.isnan(y[2:5]).all()
    assert y[[0, 1, 5, 6]].tolist() == [11, 20, 103, 200]
    # The x axis names the time steps (not "(period, time)" tuples) ...
    fmt = ax.xaxis.get_major_formatter()
    assert [fmt(x, None) for x in (0, 1, 2, 5, 6)] == ["t1", "t2", "", "t1", "t2"]
    # ... and the first step of each period carries the period name
    period_axis = ax.child_axes[0]
    assert list(period_axis.get_xticks()) == [0, 5]
    assert [t.get_text() for t in period_axis.get_xticklabels()] == ["y2030", "y2050"]


def test_plot_data_has_period_and_time_columns():
    df, demand = prepare_dispatch_data(_results(ONE), _mappings(ONE), "s", GROUP)
    out = dispatch_plot_data(df, demand, (0, 4))
    assert list(out.columns[:2]) == ["period", "time"]
    assert list(zip(out["period"], out["time"])) == STEPS
    assert out["Demand"].tolist() == [11, 20, 103, 200]


def test_combined_parquet_scenarios_with_different_periods(tmp_path):
    """Through the real load boundary: the outer-joined rows of the other scenario are not zero rows."""
    from flextool.lean_parquet import write_lean_parquet
    from flextool.scenario_comparison.db_reader import combine_parquet_files

    a_steps = [("y2050", "t1"), ("y2050", "t2")]
    b_steps = [("y2030", "t1"), ("y2030", "t2"), ("y2040", "t1"), ("y2040", "t2")]
    data = {"A": (a_steps, [7.0, 8.0]), "B": (b_steps, [1.0, 2.0, 3.0, 4.0])}
    files = {"unit_outputNode_dt_ee.parquet": []}
    for sc, (steps, pv) in data.items():
        d = tmp_path / sc
        d.mkdir()
        path = d / "unit_outputNode_dt_ee.parquet"
        write_lean_parquet(_frame(sc, steps, ("pv", NODE), pv, ["unit", "node"]), path)
        files["unit_outputNode_dt_ee.parquet"].append((sc, path))
    res = TimeSeriesResults.from_dict(combine_parquet_files(files, num_scenarios=2))
    for sc, (steps, pv) in data.items():
        df, _ = prepare_node_dispatch_data(res, sc, NODE)
        assert list(df.index) == steps, sc
        assert df["pv_out"].tolist() == pv


def _write_combined(tmp_path, data):
    """``data``: {scenario: (steps, pv, slack, curtailment, inflow)} written as parquets and combined
    through the real load boundary (``get_scenario_results``), A first."""
    from flextool.lean_parquet import write_lean_parquet
    from flextool.scenario_comparison.db_reader import get_scenario_results

    specs = [
        ("unit_outputNode_dt_ee", 1, ("pv", NODE), ["unit", "node"]),
        ("node_slack_up_dt_e", 2, (NODE,), ["node"]),
        ("unit_curtailment_outputNode_dt_ee", 3, ("pv", NODE), ["unit", "sink"]),
        ("node_inflow__dt", 4, (NODE,), ["node"]),
    ]
    for sc, row in data.items():
        d = tmp_path / sc
        d.mkdir()
        for name, pos, key, names in specs:
            write_lean_parquet(_frame(sc, row[0], key, row[pos], names), d / f"{name}.parquet")
    _, res = get_scenario_results(scenario_folders={sc: str(tmp_path) for sc in data}, parquet_subdir="")
    return res


# Sampled weeks (A: steps 1-2 and 5-6, a break before t05) against the full timeline (B: t01-t06); the
# combined row index is A's steps, then t03, t04 at the end.
SAMPLED = [("y2030", "t01"), ("y2030", "t02"), ("y2030", "t05"), ("y2030", "t06")]
FULL = [("y2030", f"t0{i}") for i in range(1, 7)]
SHARED = {
    "A": (SAMPLED, [1.0, 2.0, 5.0, 6.0], [0.0] * 4, [0.0, 1.0, 0.0, 0.0], [-1.0, -2.0, -5.0, -6.0]),
    "B": (FULL, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [0.0] * 6, [0.0] * 5 + [2.0],
          [-10.0, -20.0, -30.0, -40.0, -50.0, -60.0]),
}


def test_scenarios_sharing_steps_keep_their_own_order(tmp_path):
    res = _write_combined(tmp_path, SHARED)
    assert list(res.unit_outputNode_dt_ee.index) == SAMPLED + FULL[2:4]  # the union order
    maps = _mappings(SHARED)
    for sc, (steps, pv, _slack, curt, inflow) in SHARED.items():
        df, demand = prepare_dispatch_data(res, maps, sc, GROUP)
        assert list(df.index) == steps, sc
        assert df[PV_COL].tolist() == pv
        assert df["Curtailed"].tolist() == curt
        assert list(demand.index) == steps
        assert demand.tolist() == [-x for x in inflow]
        df_n, demand_n = prepare_node_dispatch_data(res, sc, NODE)
        assert list(df_n.index) == steps, sc
        assert df_n["pv_out"].tolist() == pv
        assert demand_n.tolist() == [-x for x in inflow]


def test_each_scenario_breaks_where_its_own_timeline_does(tmp_path, monkeypatch):
    """create_dispatch_plots with per-scenario breaks: A gets its gap before t05, B (full timeline) none."""
    from pathlib import Path

    from flextool.scenario_comparison import dispatch_plots

    res = _write_combined(tmp_path, SHARED)
    (tmp_path / "plots").mkdir()
    breaks = {"A": _index([("y2030", "t05")]), "B": _index([])}
    drawn = {}

    def capture(df, inflow, path, **kw):
        fig = _build_dispatch_figure(df, inflow, "g", timeline=kw["timeline"], break_times=kw["break_times"])
        line = next(ln for ln in fig.axes[0].get_lines() if ln.get_label() == "Demand")
        drawn[Path(path).stem] = np.asarray(line.get_ydata(), dtype=float)

    monkeypatch.setattr(dispatch_plots, "plot_dispatch_area", capture)
    dispatch_plots.create_dispatch_plots(
        res, _mappings(SHARED), tmp_path / "plots", scenarios=["A", "B"], break_times=breaks,
        settings_path=tmp_path / "none.yaml", write_xlsx=True,
    )
    # dispatch_data.xlsx: every row shows its period (no merged cells)
    sheet = pd.read_excel(tmp_path / "plots" / "dispatch_data.xlsx", sheet_name=f"{GROUP}_A", engine="openpyxl")
    assert list(zip(sheet["period"], sheet["time"])) == SAMPLED
    a = drawn[f"dispatch_nodeGroup_{GROUP}_A"]
    assert len(a) == 7 and np.isnan(a[2:5]).all()
    assert a[[0, 1, 5, 6]].tolist() == [1, 2, 5, 6]
    assert drawn[f"dispatch_nodeGroup_{GROUP}_B"].tolist() == [10, 20, 30, 40, 50, 60]


def test_own_all_empty_step_is_kept(tmp_path):
    """A step whose cells are all empty in the scenario's own file stays a row: the demand line stays aligned."""
    steps = [("p", "t1"), ("p", "t2"), ("p", "t3")]
    nan = float("nan")
    data = {
        "A": (steps, [1.0, nan, 3.0], [0.0, nan, 0.0], [0.0, nan, 0.0], [-1.0, -2.0, -3.0]),
        "B": (steps[:1], [5.0], [0.0], [0.0], [-5.0]),
    }
    res = _write_combined(tmp_path, data)
    df, demand = prepare_node_dispatch_data(res, "A", NODE)
    assert list(df.index) == steps
    assert list(demand.index) == steps
    assert demand.tolist() == [1.0, 2.0, 3.0]
