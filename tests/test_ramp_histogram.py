"""Ramp histogram (#25) and ramp envelope plots (``flextool/scenario_comparison/ramp_plots.py``).

Expected values are the hand-computed oracle of ``specs/ramp_envelope_oracle_v2.md`` Cases 6 and 6b (bin rule of
spec §2.11: symmetric range −M…+M, one bin centred on 0, bins [lo, hi) with the last one closed, width rounded
up to 1/2/5·10ᵏ, one bin set per plot over all its series; room lines in MW/h).
"""
from __future__ import annotations

import math

import matplotlib
import numpy as np
import pandas as pd
import pytest

from flextool.scenario_comparison.data_models import TimeSeriesResults
from flextool.scenario_comparison.ramp_plots import (
    RAMP_BAND_KEYS,
    build_ramp_envelope_figure,
    create_ramp_plots,
    histogram_data,
    ramp_band_colors,
    ramp_histogram,
    ramp_histogram_bins,
    room_lines,
)

matplotlib.use("Agg")

BANDS = ("ramp_net_load", "ramp_demand", "units_up", "storage_up", "vre_up", "connections_up",
         "units_down", "storage_down", "vre_down", "connections_down")
ROWS = pd.MultiIndex.from_tuples([("p", "t1"), ("p", "t2"), ("p", "t3"), ("p", "t4")], names=["period", "time"])
NAN_ROW = [math.nan] * len(BANDS)
#: Case 6 entity A = the Case 1 node: rows t1 and t3 (t2, t4 empty: block ends).
ENTITY_A = pd.DataFrame([[35, 20, 65, 30, 10, 20, -30, -7, -5, -40], NAN_ROW,
                         [-50, -20, 60, 3, 0, 25, -50, -35, -30, -35], NAN_ROW], index=ROWS, columns=BANDS,
                        dtype=float)
#: Hours a year (blocks A = {t1, t2} weight 100, B = {t3, t4} weight 50) and the transition lengths (one-hour
#: steps: (1 + 1)/2 = 1 h; none at the block ends).
WEIGHTS = pd.Series([100.0, 100.0, 50.0, 50.0], index=ROWS)
TRANS = pd.Series([1.0, math.nan, 1.0, math.nan], index=ROWS)


def _entity_b(net_t3: float) -> pd.DataFrame:
    """Case 6 / 6b entity B: demand 6 (t1), 12 (t3); net load −8 (t1), *net_t3* (t3); no room data."""
    frame = pd.DataFrame(np.nan, index=ROWS, columns=BANDS)
    frame.loc[("p", "t1"), ["ramp_net_load", "ramp_demand"]] = [-8.0, 6.0]
    frame.loc[("p", "t3"), ["ramp_net_load", "ramp_demand"]] = [net_t3, 12.0]
    return frame


def _hist(envelopes: dict) -> dict:
    return histogram_data(envelopes, {s: TRANS for s in envelopes}, {s: WEIGHTS for s in envelopes})["p"]


def _nonzero(edges: np.ndarray, hours: np.ndarray) -> dict:
    return {(float(edges[i]), float(edges[i + 1])): float(h) for i, h in enumerate(hours) if h != 0.0}


def test_case6_entity_a_bins_and_hours() -> None:
    """A: M = 50 → 2·50/40 = 2.5 → w = 5; K = ⌈50/5 − ½⌉ = 10 → 21 bins, edges −52.5 … 52.5.  demand 20 → [17.5,
    22.5) 100 h, −20 → [−22.5, −17.5) 50 h; net load 35 → [32.5, 37.5) 100 h, −50 → [−52.5, −47.5) 50 h."""
    data = _hist({"S": ENTITY_A})
    edges = data["edges"]
    assert len(edges) - 1 == 21
    assert edges[0] == -52.5 and edges[-1] == 52.5
    assert np.allclose(np.diff(edges), 5.0)
    assert _nonzero(edges, data["hours"][("S", "ramp_demand")]) == {(17.5, 22.5): 100.0, (-22.5, -17.5): 50.0}
    assert _nonzero(edges, data["hours"][("S", "ramp_net_load")]) == {(32.5, 37.5): 100.0, (-52.5, -47.5): 50.0}
    # Every series holds all 150 h.
    assert all(hours.sum() == 150.0 for hours in data["hours"].values())


def test_case6_entity_a_room_lines() -> None:
    """Room lines: min Σup = min(125, 88) = 88 MW/h, min Σdown = min(82, 150) = 82 MW/h → lines +88 and −82."""
    assert _hist({"S": ENTITY_A})["room"] == pytest.approx((88.0, -82.0))
    assert room_lines([ENTITY_A], [TRANS]) == pytest.approx((88.0, -82.0))
    # Rates per MW/h: a two-hour transition halves the room and the line.
    up, down = room_lines([ENTITY_A], [TRANS * 2.0])
    assert (up, down) == pytest.approx((44.0, -41.0))


def test_case6_entity_b_bins_and_hours() -> None:
    """B: M = max(6, 12, 8, 30) = 30 → 60/40 = 1.5 → w = 2; K = ⌈15 − ½⌉ = 15 → 31 bins, the odd integers −31 … 31.
    demand 6 → [5, 7) 100 h, 12 → [11, 13) 50 h; net load −8 → [−9, −7) 100 h, 30 → [29, 31] 50 h; no room."""
    data = _hist({"S": _entity_b(30.0)})
    edges = data["edges"]
    assert len(edges) - 1 == 31
    assert edges.tolist() == list(np.arange(-31.0, 32.0, 2.0))
    assert _nonzero(edges, data["hours"][("S", "ramp_demand")]) == {(5.0, 7.0): 100.0, (11.0, 13.0): 50.0}
    assert _nonzero(edges, data["hours"][("S", "ramp_net_load")]) == {(-9.0, -7.0): 100.0, (29.0, 31.0): 50.0}
    up, down = data["room"]
    assert math.isnan(up) and math.isnan(down)


def test_case6b_shared_range_over_two_scenarios() -> None:
    """6b: S2 net load at t3 44 → M = 44 → 88/40 = 2.2 → w = 5; K = ⌈44/5 − ½⌉ = 9 → 19 bins, −47.5 … 47.5.  S1
    demand 6 → [2.5, 7.5) 100 h, 12 → [7.5, 12.5) 50 h; S1 net load −8 → [−12.5, −7.5) 100 h, 30 → [27.5, 32.5)
    50 h; S2 demand as S1; S2 net load −8 → [−12.5, −7.5) 100 h, 44 → [42.5, 47.5] 50 h.  S1 alone keeps w = 2."""
    data = _hist({"S1": _entity_b(30.0), "S2": _entity_b(44.0)})
    edges = data["edges"]
    assert len(edges) - 1 == 19
    assert edges[0] == -47.5 and edges[-1] == 47.5
    demand = {(2.5, 7.5): 100.0, (7.5, 12.5): 50.0}
    assert _nonzero(edges, data["hours"][("S1", "ramp_demand")]) == demand
    assert _nonzero(edges, data["hours"][("S2", "ramp_demand")]) == demand
    assert _nonzero(edges, data["hours"][("S1", "ramp_net_load")]) == {(-12.5, -7.5): 100.0, (27.5, 32.5): 50.0}
    assert _nonzero(edges, data["hours"][("S2", "ramp_net_load")]) == {(-12.5, -7.5): 100.0, (42.5, 47.5): 50.0}
    alone = _hist({"S1": _entity_b(30.0)})["edges"]
    assert len(alone) - 1 == 31


def test_bins_degenerate_and_closed_last_bin() -> None:
    """M = 0: one bin [−½, ½]; no rows: no bins.  A value at +M = (K + ½)·w falls in the last (closed) bin, a
    value at −M = −(K + ½)·w in the first: M = 47.5 → w = 5 (2.375 → 5), K = ⌈9.5 − ½⌉ = 9, edges ±47.5."""
    assert ramp_histogram_bins([np.zeros(3)]).tolist() == [-0.5, 0.5]
    assert ramp_histogram_bins([np.array([math.nan])]) is None
    assert ramp_histogram_bins([]) is None
    edges = ramp_histogram_bins([np.array([47.5, -47.5, 0.0])])
    assert edges[0] == -47.5 and edges[-1] == 47.5 and len(edges) - 1 == 19
    hours = ramp_histogram([47.5, -47.5, 0.0], [1.0, 2.0, 4.0], edges)
    assert hours[-1] == 1.0 and hours[0] == 2.0 and hours[9] == 4.0


def test_band_colors_from_the_template_and_the_default() -> None:
    """``categories.ramp_envelope`` of plot_settings.yaml colours a band; a band it does not list keeps the
    default."""
    colors = ramp_band_colors({"categories": {"ramp_envelope": {"vre": "#ff0000"}}})
    assert colors["vre"] == (1.0, 0.0, 0.0)
    assert colors["units"] == "#4e79a7"
    assert set(colors) == set(RAMP_BAND_KEYS)


def _results() -> TimeSeriesResults:
    """Combined comparison frames of one scenario: node n (entity A of Case 6) and group G (the same rows)."""
    def wide(frame: pd.DataFrame, entity: str, level: str) -> pd.DataFrame:
        out = frame.copy()
        out.columns = pd.MultiIndex.from_product([["S"], [entity], BANDS], names=["scenario", level, "band"])
        return out

    res = TimeSeriesResults()
    res.node_ramp_envelope_dt_ep = wide(ENTITY_A, "n", "node")
    res.nodeGroup_ramp_envelope_dt_gp = wide(ENTITY_A, "G", "group")
    res.timestep_hours_dt = pd.DataFrame(
        {("S", "timestep_hours"): WEIGHTS}).rename_axis(columns=["scenario", "param"])
    res.transition_hours_dt = pd.DataFrame(
        {("S", "transition_hours"): TRANS}).rename_axis(columns=["scenario", "param"])
    return res


def test_envelope_figure_draws_the_bands_and_lines() -> None:
    """The envelope of Case 1's node: one legend entry per band plus the two lines; empty rows are gaps."""
    fig = build_ramp_envelope_figure(ENTITY_A, "n", timeline=(0, 4), keep_plot_data=True)
    ax = fig.axes[0]
    labels = ax.get_legend().get_texts()
    assert [t.get_text() for t in labels] == ["Net load ramp", "Demand ramp", "Connections", "VRE", "Storage",
                                              "Units"]
    net = ax.lines[0].get_ydata()
    assert net[0] == 35.0 and math.isnan(net[1]) and net[2] == -50.0
    assert "change toward the next step" in ax.get_ylabel()
    assert fig.flextool_plot_batch[0][1].shape == (4, 10)


def test_comparison_run_writes_png_and_csv(tmp_path) -> None:
    """One envelope PNG per scenario and one frequency PNG per entity, with their CSVs: the node only when the
    caller lists it (all-nodes option)."""
    written = create_ramp_plots(_results(), tmp_path, ["S"], groups=["G"], nodes=[], plot_rows=(0, 3),
                                write_plot_data=True)
    names = sorted(p.name for p in written)
    assert names == ["ramp_envelope_nodeGroup_G_S.png", "ramp_frequency_nodeGroup_G.png"]
    envelope_csv = pd.read_csv(tmp_path / "ramp_envelope_nodeGroup_G_S.csv")
    assert list(envelope_csv.columns) == ["period", "time", *BANDS[2:], "ramp_net_load", "ramp_demand"]
    assert envelope_csv["units_up"].tolist()[0] == 65.0
    freq = pd.read_csv(tmp_path / "ramp_frequency_nodeGroup_G.csv")
    assert list(freq.columns) == ["bin_from", "bin_to", "S | net load [h/year]", "S | demand [h/year]",
                                  "smallest up room", "smallest down room"]
    assert len(freq) == 21
    assert freq["S | demand [h/year]"].sum() == 150.0
    assert freq["smallest up room"].iloc[0] == 88.0
    with_nodes = create_ramp_plots(_results(), tmp_path / "debug", ["S"], groups=["G"], nodes=["n"])
    assert sorted(p.name for p in with_nodes) == ["ramp_envelope_nodeGroup_G_S.png", "ramp_envelope_node_n_S.png",
                                                  "ramp_frequency_nodeGroup_G.png", "ramp_frequency_node_n.png"]
