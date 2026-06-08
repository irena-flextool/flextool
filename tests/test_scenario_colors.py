"""Scenario coloring: comparison plots color the scenario series from the
project's ``scenarios`` section of plot_settings, but only when the scenario
dim is the colored (legend) series — not when it's a subplot/file split.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb

from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.orchestrator import _apply_dimension_rules
from flextool.plot_outputs import plan as P
from flextool.plot_outputs.color_template import (
    resolve_label_color,
    order_labels_by_template,
)
from flextool.plot_outputs.legend_helpers import build_shared_color_map

RED = "#ff0000"
GREEN = "#00ff00"
TPL = {"scenarios": {"scenA": RED, "scenB": GREEN}}


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class TestScenarioResolver:
    def test_resolve_hit(self):
        assert resolve_label_color("scenA", TPL, scenario=True) == pytest.approx(to_rgb(RED))

    def test_resolve_case_insensitive(self):
        assert resolve_label_color("SCENA", TPL, scenario=True) == pytest.approx(to_rgb(RED))

    def test_resolve_miss(self):
        assert resolve_label_color("scenZ", TPL, scenario=True) is None

    def test_not_a_scenario_lookup_without_flag(self):
        # Without scenario=True the scenarios section is not consulted.
        assert resolve_label_color("scenA", TPL) is None

    def test_order_listed_first_then_tail(self):
        assert order_labels_by_template(
            ["scenB", "x", "scenA"], TPL, scenario=True) == ["scenA", "scenB", "x"]

    def test_shared_map_scenario(self):
        cm = build_shared_color_map(
            ["scenA", "scenB", "extra"], color_template=TPL, scenario=True)
        assert cm["scenA"] == pytest.approx(to_rgb(RED))
        assert cm["scenB"] == pytest.approx(to_rgb(GREEN))
        assert cm["extra"] not in (to_rgb(RED), to_rgb(GREEN))  # palette


# ---------------------------------------------------------------------------
# Plan wiring — scenario colored only when it is the legend series
# ---------------------------------------------------------------------------

def _bar_plan(df, cfg, color_path):
    dfp, rules, _ctype, *_ = _apply_dimension_rules(df, cfg, (0, 2))
    cr = rules[dfp.index.nlevels:]
    g = [i for i, c in enumerate(cr) if c == 'g']
    s = [i for i, c in enumerate(cr) if c == 's']
    e = [i for i, c in enumerate(cr) if c == 'e']
    u = [i for i, c in enumerate(cr) if c == 'u']
    return P._compute_bar_plan(dfp, "cmp", cfg, s, e, u, g, None, color_path=color_path)


def _write_tpl(tmp_path, body):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(body)
    return p


class TestScenarioInLegend:
    def test_grouped_scenario_uses_scenario_colors(self, tmp_path):
        # [d_se, b_ge]: period=bar, scenario=grouped (legend), entity=expand.
        cols = pd.MultiIndex.from_tuples(
            [("scenA", "nodeX"), ("scenB", "nodeX")], names=["scenario", "entity"])
        df = pd.DataFrame(np.array([[10.0, 20.0], [12.0, 22.0]]),
                          index=pd.Index(["2020", "2021"], name="period"), columns=cols)
        p = _write_tpl(tmp_path, "scenarios:\n  scenA: '%s'\n  scenB: '%s'\n" % (RED, GREEN))
        cfg = PlotConfig(map_dimensions_for_plots=["d_se", "b_ge"], legend="shared")
        plan = _bar_plan(df, cfg, p)
        assert plan.shared_color_map["scenA"] == pytest.approx(to_rgb(RED))
        assert plan.shared_color_map["scenB"] == pytest.approx(to_rgb(GREEN))

    def test_scenario_as_subplot_does_not_use_scenario_colors(self, tmp_path):
        # [d_se, b_eu]: period=bar, entity=expand, scenario=SUBPLOT (not legend).
        # No grouped/stack legend series -> simple bar; scenario colors must
        # NOT apply (the scenarios section is only for the legend series).
        cols = pd.MultiIndex.from_tuples(
            [("scenA", "nodeX"), ("scenB", "nodeX")], names=["scenario", "entity"])
        df = pd.DataFrame(np.array([[10.0, 20.0], [12.0, 22.0]]),
                          index=pd.Index(["2020", "2021"], name="period"), columns=cols)
        p = _write_tpl(tmp_path, "scenarios:\n  scenA: '%s'\n  scenB: '%s'\n" % (RED, GREEN))
        cfg = PlotConfig(map_dimensions_for_plots=["d_se", "b_eu"], legend="shared")
        plan = _bar_plan(df, cfg, p)
        # Simple bar (no grouped/stack legend) -> no scenario-colored map.
        scen_colored = bool(plan.shared_color_map) and any(
            v == pytest.approx(to_rgb(RED)) for v in plan.shared_color_map.values())
        assert not scen_colored


class TestScenarioLineLegend:
    def test_line_scenario_uses_scenario_colors(self, tmp_path):
        # [dt_se, tt_l]: time=x, scenario=line (legend), entity summed-in.
        idx = pd.MultiIndex.from_product(
            [["2020"], range(4)], names=["period", "time"])
        cols = pd.Index(["scenA", "scenB"], name="scenario")
        df = pd.DataFrame(np.random.default_rng(0).random((4, 2)) * 10,
                          index=idx, columns=cols)
        p = _write_tpl(tmp_path, "scenarios:\n  scenA: '%s'\n  scenB: '%s'\n" % (RED, GREEN))
        cfg = PlotConfig(map_dimensions_for_plots=["dt_e", "tt_l"], legend="shared")
        dfp, rules, _ctype, *_ = _apply_dimension_rules(df, cfg, (0, 4))
        cr = rules[dfp.index.nlevels:]
        line = [i for i, c in enumerate(cr) if c == 'l']
        sub = [i for i, c in enumerate(cr) if c == 'u']
        stack = [i for i, c in enumerate(cr) if c == 's']
        plan = P._compute_time_plan(dfp, "cmp", cfg, stack, sub, line, None, (0, 4),
                                    color_path=p)
        assert plan.shared_color_map["scenA"] == pytest.approx(to_rgb(RED))
        assert plan.shared_color_map["scenB"] == pytest.approx(to_rgb(GREEN))
