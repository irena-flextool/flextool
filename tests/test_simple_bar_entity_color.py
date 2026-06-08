"""Simple (non-stacked, non-grouped) bar charts colored by entity class.

Covers the ``color_entity_class`` → simple-bar coloring path: the resolver
that maps an entity class to the column/index level carrying it, the map
build, the per-bar color lookup, and an end-to-end render where the bars
take their entity colors instead of the default steelblue.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.orchestrator import prepare_plot_data
from flextool.plot_outputs.plan import (
    resolve_color_bar_level,
    _color_level_values,
    build_simple_bar_color_map,
)
from flextool.plot_outputs.plot_bars_detail import (
    _resolve_bar_color,
    DEFAULT_SIMPLE_BAR_COLOR,
)

RED = "#ff0000"
GREEN = "#00ff00"
STEELBLUE_RGB = to_rgb(DEFAULT_SIMPLE_BAR_COLOR)


# ---------------------------------------------------------------------------
# resolve_color_bar_level — engine-owned level-name → class map
# ---------------------------------------------------------------------------

class TestResolveColorBarLevel:
    def test_process_resolves_for_unit(self):
        assert resolve_color_bar_level(["process", "node"], "unit") == "process"

    def test_process_resolves_for_connection(self):
        # 'process' masks both unit and connection.
        assert resolve_color_bar_level(["process", "node"], "connection") == "process"

    def test_node_resolves_for_node(self):
        assert resolve_color_bar_level(["process", "node"], "node") == "node"

    def test_source_and_sink_count_as_node(self):
        assert resolve_color_bar_level(["source"], "node") == "source"
        assert resolve_color_bar_level(["sink"], "node") == "sink"

    def test_first_match_wins_on_tie(self):
        # Two node-bearing levels — first in order wins (simple tiebreak).
        assert resolve_color_bar_level(["source", "sink"], "node") == "source"

    def test_no_match_returns_none(self):
        assert resolve_color_bar_level(["node"], "unit") is None

    def test_no_class_returns_none(self):
        assert resolve_color_bar_level(["process"], None) is None


# ---------------------------------------------------------------------------
# _color_level_values — distinct values from columns or index
# ---------------------------------------------------------------------------

class TestColorLevelValues:
    def test_reads_from_columns(self):
        cols = pd.MultiIndex.from_tuples(
            [("u1", "n1"), ("u2", "n1")], names=["process", "node"]
        )
        df = pd.DataFrame([[1.0, 2.0]], index=pd.Index(["p0"], name="period"), columns=cols)
        assert _color_level_values(df, "process") == ["u1", "u2"]

    def test_reads_from_index(self):
        # 'a'-style: the entity sits on the (bar) row index.
        idx = pd.Index(["u1", "u2"], name="process")
        df = pd.DataFrame([[1.0], [2.0]], index=idx,
                          columns=pd.Index(["n1"], name="node"))
        assert _color_level_values(df, "process") == ["u1", "u2"]

    def test_absent_level_returns_empty(self):
        df = pd.DataFrame([[1.0]], index=pd.Index(["p0"], name="period"),
                          columns=pd.Index(["n1"], name="node"))
        assert _color_level_values(df, "process") == []


# ---------------------------------------------------------------------------
# _resolve_bar_color — per-bar color from the right location
# ---------------------------------------------------------------------------

def _cmap():
    return {"u1": (1.0, 0.0, 0.0), "u2": (0.0, 1.0, 0.0)}


class TestResolveBarColor:
    def test_expand_level_uses_group(self):
        df = pd.DataFrame(
            [[1.0]], index=pd.Index(["p0"], name="period"),
            columns=pd.MultiIndex.from_tuples([("u1", "n1")], names=["process", "node"]),
        )
        c = _resolve_bar_color("u1", "p0", df, ["process"], "process", _cmap())
        assert c == (1.0, 0.0, 0.0)

    def test_index_level_uses_period(self):
        # The bar IS the entity (row index); color keyed by the bar's index value.
        df = pd.DataFrame(
            [[1.0], [2.0]], index=pd.Index(["u1", "u2"], name="process"),
            columns=pd.Index(["n1"], name="node"),
        )
        assert _resolve_bar_color("n1", "u2", df, [], "process", _cmap()) == (0.0, 1.0, 0.0)

    def test_subplot_level_constant_across_subplot(self):
        # color_bar_level survives as a single-valued column level (subplot).
        df = pd.DataFrame(
            [[1.0]], index=pd.Index(["p0"], name="period"),
            columns=pd.MultiIndex.from_tuples([("n1", "u1")], names=["node", "process"]),
        )
        # expand level is 'node'; color level 'process' is the (single-valued) other level.
        c = _resolve_bar_color("n1", "p0", df, ["node"], "process", _cmap())
        assert c == (1.0, 0.0, 0.0)

    def test_unresolved_value_falls_back_to_default(self):
        df = pd.DataFrame(
            [[1.0]], index=pd.Index(["p0"], name="period"),
            columns=pd.MultiIndex.from_tuples([("u9", "n1")], names=["process", "node"]),
        )
        c = _resolve_bar_color("u9", "p0", df, ["process"], "process", _cmap())
        assert c == DEFAULT_SIMPLE_BAR_COLOR


# ---------------------------------------------------------------------------
# build_simple_bar_color_map — resolution + map, shared by both render paths
# ---------------------------------------------------------------------------

class TestBuildSimpleBarColorMap:
    def test_map_built_for_column_entity(self):
        cols = pd.MultiIndex.from_tuples(
            [("u1", "n1"), ("u2", "n1")], names=["process", "node"]
        )
        df = pd.DataFrame([[1.0, 2.0]], index=pd.Index(["p0"], name="period"), columns=cols)
        tpl = {"entities": {"unit": {"u1": RED, "u2": GREEN}}}
        cmap, level = build_simple_bar_color_map(df, "unit", tpl)
        assert level == "process"
        assert cmap["u1"] == pytest.approx(to_rgb(RED))
        assert cmap["u2"] == pytest.approx(to_rgb(GREEN))

    def test_no_level_returns_none(self):
        df = pd.DataFrame([[1.0]], index=pd.Index(["p0"], name="period"),
                          columns=pd.Index(["n1"], name="node"))
        cmap, level = build_simple_bar_color_map(df, "unit", {})
        assert cmap is None and level is None

    def test_process_resolves_against_unit_and_connection(self):
        # 'process' column carries both a unit (u1) and a connection (c1);
        # each is resolved against its own class section.
        cols = pd.MultiIndex.from_tuples(
            [("u1", "n1"), ("c1", "n1")], names=["process", "node"]
        )
        df = pd.DataFrame([[1.0, 2.0]], index=pd.Index(["p0"], name="period"), columns=cols)
        tpl = {"entities": {"unit": {"u1": RED}, "connection": {"c1": GREEN}}}
        cmap, level = build_simple_bar_color_map(df, "process", tpl)
        assert level == "process"
        assert cmap["u1"] == pytest.approx(to_rgb(RED))
        assert cmap["c1"] == pytest.approx(to_rgb(GREEN))


# ---------------------------------------------------------------------------
# End-to-end render: bars take entity colors, not steelblue
# ---------------------------------------------------------------------------

def _flow_df():
    """unit→node flow shape: period index, [process, node] columns."""
    cols = pd.MultiIndex.from_tuples(
        [("u1", "n1"), ("u2", "n1")], names=["process", "node"]
    )
    index = pd.Index(["2020", "2021"], name="period")
    return pd.DataFrame(
        np.array([[10.0, 20.0], [12.0, 22.0]]), index=index, columns=cols
    )


def _write_template(tmp_path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n  unit:\n    u1: '%s'\n    u2: '%s'\n" % (RED, GREEN)
    )
    return p


def _bar_facecolors(figures):
    colors = set()
    for _name, fig in figures:
        for ax in fig.axes:
            for patch in ax.patches:
                if isinstance(patch, Rectangle):
                    w, h = patch.get_width(), patch.get_height()
                    if w and h:  # a real bar, not a spine artifact
                        colors.add(tuple(round(c, 3) for c in patch.get_facecolor()[:3]))
    return colors


class TestEndToEndColoring:
    def test_period_variant_colors_bars_by_unit(self, tmp_path):
        # 'p' variant: unit is the expand level → per-bar color.
        df = _flow_df()
        cfg = PlotConfig(
            plot_name="unit to node",
            map_dimensions_for_plots=["d_ee", "b_eu"],
            color_entity_class="unit",
        )
        figures, total = prepare_plot_data(
            df, cfg, plot_name="unit to node", color_path=_write_template(tmp_path)
        )
        assert total >= 1
        colors = _bar_facecolors(figures)
        assert tuple(round(c, 3) for c in to_rgb(RED)) in colors
        assert tuple(round(c, 3) for c in to_rgb(GREEN)) in colors
        assert tuple(round(c, 3) for c in STEELBLUE_RGB) not in colors

    def test_total_variant_colors_bars_by_unit(self, tmp_path):
        # 'a' variant: period summed, unit becomes the bar (row index).
        df = _flow_df()
        cfg = PlotConfig(
            plot_name="unit to node total",
            map_dimensions_for_plots=["d_ee", "y_be"],
            color_entity_class="unit",
        )
        figures, total = prepare_plot_data(
            df, cfg, plot_name="unit to node total",
            color_path=_write_template(tmp_path),
        )
        assert total >= 1
        colors = _bar_facecolors(figures)
        assert tuple(round(c, 3) for c in to_rgb(RED)) in colors
        assert tuple(round(c, 3) for c in to_rgb(GREEN)) in colors
        assert tuple(round(c, 3) for c in STEELBLUE_RGB) not in colors

    def test_without_color_entity_class_bars_stay_default(self, tmp_path):
        df = _flow_df()
        cfg = PlotConfig(
            plot_name="unit to node",
            map_dimensions_for_plots=["d_ee", "b_eu"],
        )
        figures, _ = prepare_plot_data(
            df, cfg, plot_name="unit to node", color_path=_write_template(tmp_path)
        )
        colors = _bar_facecolors(figures)
        assert tuple(round(c, 3) for c in STEELBLUE_RGB) in colors
        assert tuple(round(c, 3) for c in to_rgb(RED)) not in colors

    def test_subplot_level_colors_each_subplot(self, tmp_path):
        # Reserve-slack shape: bars are (reserve, updown), the node group is
        # the SUBPLOT — color each subplot's bars by its group. Exercises the
        # live path's re-attach of the dropped subplot level.
        cols = pd.MultiIndex.from_tuples(
            [("primary", "up", "g1"), ("primary", "up", "g2"),
             ("primary", "down", "g1"), ("primary", "down", "g2")],
            names=["reserve", "updown", "group"],
        )
        df = pd.DataFrame(
            np.array([[5.0, 7.0, 3.0, 2.0], [6.0, 8.0, 4.0, 2.5]]),
            index=pd.Index(["2020", "2021"], name="period"), columns=cols,
        )
        p = tmp_path / "plot_settings.yaml"
        p.write_text("entities:\n  group:\n    g1: '%s'\n    g2: '%s'\n" % (RED, GREEN))
        cfg = PlotConfig(
            plot_name="reserve", map_dimensions_for_plots=["d_eeg", "y_bbu"],
            color_entity_class="group",
        )
        figures, total = prepare_plot_data(df, cfg, plot_name="reserve", color_path=p)
        assert total >= 1
        colors = _bar_facecolors(figures)
        assert tuple(round(c, 3) for c in to_rgb(RED)) in colors
        assert tuple(round(c, 3) for c in to_rgb(GREEN)) in colors
        assert tuple(round(c, 3) for c in STEELBLUE_RGB) not in colors

    def test_process_variant_colors_units_and_connections(self, tmp_path):
        # process_co2 shape: a 'process' bar level mixing a unit and a
        # connection; color_entity_class: process resolves each against its
        # own class section.
        cols = pd.MultiIndex.from_tuples(
            [("unit", "u1", "co2", "n1"), ("connection", "c1", "co2", "n1")],
            names=["type", "process", "commodity", "node"],
        )
        df = pd.DataFrame(
            np.array([[10.0, 20.0], [12.0, 22.0]]),
            index=pd.Index(["2020", "2021"], name="period"), columns=cols,
        )
        p = tmp_path / "plot_settings.yaml"
        p.write_text(
            "entities:\n  unit:\n    u1: '%s'\n  connection:\n    c1: '%s'\n"
            % (RED, GREEN)
        )
        cfg = PlotConfig(
            plot_name="process co2",
            map_dimensions_for_plots=["d_eeee", "b_xbxx"],
            color_entity_class="process",
        )
        figures, total = prepare_plot_data(
            df, cfg, plot_name="process co2", color_path=p
        )
        assert total >= 1
        colors = _bar_facecolors(figures)
        assert tuple(round(c, 3) for c in to_rgb(RED)) in colors      # unit u1
        assert tuple(round(c, 3) for c in to_rgb(GREEN)) in colors    # connection c1
        assert tuple(round(c, 3) for c in STEELBLUE_RGB) not in colors
