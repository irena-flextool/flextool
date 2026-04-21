"""Tests for the color-template infrastructure (Chunk D).

Covers:
* ``load_color_template`` — missing file, valid file, caching.
* ``resolve_label_color`` — category vs entity-class lookup, case rules,
  hex/RGB list parsing, missing sections, malformed values.
* ``build_shared_color_map`` — backward-compat when no template is
  provided, template+palette interleaving, integration with a
  default-shape ``PlotConfig``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
import yaml

from flextool.plot_outputs import color_template as ct
from flextool.plot_outputs.config import PlotConfig
from flextool.plot_outputs.legend_helpers import build_shared_color_map


# ---------------------------------------------------------------------------
# load_color_template
# ---------------------------------------------------------------------------


class TestLoadColorTemplate:
    def setup_method(self) -> None:
        ct._clear_cache()

    def test_missing_file_returns_empty_dict(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        assert ct.load_color_template(missing) == {}

    def test_valid_file_parses(self, tmp_path):
        p = tmp_path / "colors.yaml"
        p.write_text(
            yaml.safe_dump(
                {
                    "category": {"costs": {"Solar cost": "#112233"}},
                    "entity_class": {"group": {"Solar": [255, 128, 0]}},
                }
            ),
            encoding="utf-8",
        )
        data = ct.load_color_template(p)
        assert data["category"]["costs"]["Solar cost"] == "#112233"
        assert data["entity_class"]["group"]["Solar"] == [255, 128, 0]

    def test_caching_honored_on_unchanged_file(self, tmp_path):
        p = tmp_path / "colors.yaml"
        p.write_text(yaml.safe_dump({"category": {"x": {"a": "#010203"}}}), encoding="utf-8")
        first = ct.load_color_template(p)
        # Mutate the file contents but keep mtime the same by restoring the stat.
        # Simpler: rewrite file identically and assert cache returned same obj.
        second = ct.load_color_template(p)
        assert first is second  # cache hit returns the same dict object

    def test_cache_refreshes_when_file_changes(self, tmp_path):
        p = tmp_path / "colors.yaml"
        p.write_text(
            yaml.safe_dump({"category": {"x": {"a": "#010203"}}}), encoding="utf-8"
        )
        first = ct.load_color_template(p)
        # Bump mtime by writing new content with a forced-later mtime.
        import os
        import time
        time.sleep(0.01)
        p.write_text(
            yaml.safe_dump({"category": {"x": {"a": "#ffffff"}}}), encoding="utf-8"
        )
        os.utime(p, None)  # ensure mtime updates even on fast filesystems
        second = ct.load_color_template(p)
        assert first is not second
        assert second["category"]["x"]["a"] == "#ffffff"

    def test_malformed_yaml_returns_empty(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("::::not valid yaml:::\n  - [", encoding="utf-8")
        assert ct.load_color_template(p) == {}

    def test_non_dict_top_level_returns_empty(self, tmp_path):
        p = tmp_path / "list.yaml"
        p.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
        assert ct.load_color_template(p) == {}


# ---------------------------------------------------------------------------
# resolve_label_color
# ---------------------------------------------------------------------------


class TestResolveLabelColor:
    def test_category_match_exact(self):
        tpl = {"category": {"costs": {"Solar cost": "#102030"}}}
        c = ct.resolve_label_color("Solar cost", tpl, category="costs")
        assert c == pytest.approx((0x10 / 255, 0x20 / 255, 0x30 / 255))

    def test_category_match_is_case_sensitive(self):
        tpl = {"category": {"costs": {"Solar cost": "#102030"}}}
        assert ct.resolve_label_color("solar cost", tpl, category="costs") is None

    def test_entity_class_match_is_case_insensitive(self):
        tpl = {"entity_class": {"group": {"Solar": "#abcdef"}}}
        for label in ("Solar", "solar", "SOLAR"):
            c = ct.resolve_label_color(label, tpl, entity_class="group")
            assert c == pytest.approx(
                (0xAB / 255, 0xCD / 255, 0xEF / 255)
            ), f"failed for {label!r}"

    def test_both_hints_none_returns_none(self):
        tpl = {"category": {"costs": {"Solar cost": "#102030"}}}
        assert ct.resolve_label_color("Solar cost", tpl) is None

    def test_hex_without_hash(self):
        tpl = {"category": {"x": {"a": "ff0080"}}}
        c = ct.resolve_label_color("a", tpl, category="x")
        assert c == pytest.approx((1.0, 0.0, 128 / 255))

    def test_rgb_list_int_0_255(self):
        tpl = {"category": {"x": {"a": [255, 128, 0]}}}
        c = ct.resolve_label_color("a", tpl, category="x")
        assert c == pytest.approx((1.0, 128 / 255, 0.0))

    def test_rgb_list_float_0_1(self):
        tpl = {"category": {"x": {"a": [1.0, 0.5, 0.0]}}}
        c = ct.resolve_label_color("a", tpl, category="x")
        assert c == pytest.approx((1.0, 0.5, 0.0))

    def test_rgb_tuple_accepted(self):
        tpl = {"category": {"x": {"a": (0, 255, 0)}}}
        c = ct.resolve_label_color("a", tpl, category="x")
        assert c == pytest.approx((0.0, 1.0, 0.0))

    def test_missing_category_returns_none(self):
        tpl = {"category": {"other": {"a": "#101010"}}}
        assert ct.resolve_label_color("a", tpl, category="missing") is None

    def test_missing_entity_class_returns_none(self):
        tpl = {"entity_class": {"other": {"a": "#101010"}}}
        assert ct.resolve_label_color("a", tpl, entity_class="missing") is None

    def test_missing_label_returns_none(self):
        tpl = {"category": {"x": {"a": "#101010"}}}
        assert ct.resolve_label_color("b", tpl, category="x") is None

    @pytest.mark.parametrize(
        "bad",
        [
            "not-a-color",
            "#12345",        # too short
            "#1234567",      # too long
            [1, 2],          # wrong length
            [-1, 0, 0],      # negative
            "zzzzzz",        # non-hex
            123,             # not a string or list
            None,
        ],
    )
    def test_malformed_value_returns_none(self, bad):
        tpl = {"category": {"x": {"a": bad}}}
        assert ct.resolve_label_color("a", tpl, category="x") is None

    def test_non_dict_template_returns_none(self):
        assert ct.resolve_label_color("a", None, category="x") is None
        assert ct.resolve_label_color("a", [], category="x") is None

    def test_entity_class_preferred_over_category(self):
        """When both hints are supplied, entity_class wins per the spec."""
        tpl = {
            "category": {"costs": {"solar": "#000000"}},
            "entity_class": {"group": {"solar": "#ffffff"}},
        }
        c = ct.resolve_label_color(
            "solar", tpl, category="costs", entity_class="group"
        )
        assert c == pytest.approx((1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# build_shared_color_map
# ---------------------------------------------------------------------------


class TestBuildSharedColorMap:
    def test_no_template_matches_old_output(self):
        """Without a template, behaviour matches the pre-refactor code."""
        labels = [f"l{i}" for i in range(5)]
        result = build_shared_color_map(labels)
        expected = plt.colormaps["tab10"].colors
        for i, label in enumerate(labels):
            assert result[label] == expected[i]

    def test_no_template_uses_tab20_when_more_than_10(self):
        labels = [f"l{i}" for i in range(12)]
        result = build_shared_color_map(labels)
        expected = plt.colormaps["tab20"].colors
        for i, label in enumerate(labels):
            assert result[label] == expected[i]

    def test_template_hits_do_not_consume_palette_slots(self):
        """Template-matched labels get explicit colors; palette labels get
        tab10[0], tab10[1], ... regardless of interleaving."""
        tpl = {
            "category": {
                "costs": {
                    "Templated A": "#ff0000",
                    "Templated B": "#00ff00",
                }
            }
        }
        labels = ["Templated A", "palette1", "Templated B", "palette2", "palette3"]
        result = build_shared_color_map(
            labels, color_template=tpl, category="costs"
        )
        assert result["Templated A"] == pytest.approx((1.0, 0.0, 0.0))
        assert result["Templated B"] == pytest.approx((0.0, 1.0, 0.0))
        tab10 = plt.colormaps["tab10"].colors
        assert result["palette1"] == tab10[0]
        assert result["palette2"] == tab10[1]
        assert result["palette3"] == tab10[2]

    def test_template_empty_category_falls_back_to_palette(self):
        tpl = {"category": {"costs": {}}}
        labels = ["a", "b", "c"]
        result = build_shared_color_map(
            labels, color_template=tpl, category="costs"
        )
        tab10 = plt.colormaps["tab10"].colors
        for i, label in enumerate(labels):
            assert result[label] == tab10[i]

    def test_entity_class_case_insensitive_integration(self):
        tpl = {"entity_class": {"group": {"Solar": "#010203"}}}
        labels = ["solar", "wind"]
        result = build_shared_color_map(
            labels, color_template=tpl, entity_class="group"
        )
        assert result["solar"] == pytest.approx(
            (1 / 255, 2 / 255, 3 / 255)
        )
        assert result["wind"] == plt.colormaps["tab10"].colors[0]

    def test_default_plotconfig_yields_unchanged_output(self):
        """Constructing with an all-default PlotConfig should leave the
        palette-only output untouched — the relevant hint fields default
        to ``None``."""
        cfg = PlotConfig()
        assert cfg.color_category is None
        assert cfg.color_entity_class is None

        labels = ["a", "b", "c", "d"]
        # Default cfg: no hints → identical to old palette-only call.
        result = build_shared_color_map(
            labels,
            color_template={"category": {"costs": {"a": "#ff0000"}}},
            category=cfg.color_category,
            entity_class=cfg.color_entity_class,
        )
        tab10 = plt.colormaps["tab10"].colors
        for i, label in enumerate(labels):
            assert result[label] == tab10[i]

    def test_palette_selection_based_on_palette_needed_count(self):
        """When enough labels are templated that only ≤10 palette slots
        remain, we should use tab10 even if the total label count is >10.
        This is a behavioural choice we pin down in a test."""
        tpl = {
            "category": {
                "costs": {f"tpl_{i}": "#ff0000" for i in range(5)}
            }
        }
        labels = [f"tpl_{i}" for i in range(5)] + [f"pal_{i}" for i in range(9)]
        result = build_shared_color_map(
            labels, color_template=tpl, category="costs"
        )
        tab10 = plt.colormaps["tab10"].colors
        # 9 palette labels, palette_needed <= 10, so tab10 should be used.
        for i in range(9):
            assert result[f"pal_{i}"] == tab10[i]

    def test_existing_call_sites_keep_working(self):
        """The old single-arg signature still works unchanged."""
        labels = ["a", "b"]
        result = build_shared_color_map(labels)
        tab10 = plt.colormaps["tab10"].colors
        assert result == {"a": tab10[0], "b": tab10[1]}


# ---------------------------------------------------------------------------
# Chunk E: cost color-template integration (through the plan pipeline)
# ---------------------------------------------------------------------------


class TestCostCategoryIntegration:
    """End-to-end: a bar plan computed with ``color_category='costs'`` pulls
    colors from ``templates/default_colors.yaml`` for known cost labels and
    falls back to the palette for unknown labels.

    This locks in the wiring between :class:`PlotConfig`,
    :func:`_compute_bar_plan`, and the real shipped YAML.  It's the
    canonical regression test for Chunk E.
    """

    def setup_method(self) -> None:
        ct._clear_cache()

    def test_shipped_template_has_expected_cost_labels(self):
        """Sanity check on the YAML: the labels we plan to tag must all
        resolve.  Keeps the YAML honest if someone renames a cost key."""
        tpl = ct.load_color_template()
        # A representative sample spanning semantic groups (fuel, invest,
        # slack, CO2).  If any of these disappear, the YAML drifted.
        expected_labels = [
            "unit investment & retirement",
            "commodity_cost",
            "co2",
            "upward slack penalty",
            "other operational",
            "starts",
            "fixed cost pre-existing",
        ]
        for lbl in expected_labels:
            assert ct.resolve_label_color(lbl, tpl, category="costs") is not None, (
                f"Shipped template is missing cost label {lbl!r}"
            )

    def test_bar_plan_uses_template_colors_for_cost_labels(self):
        """A bar plan with ``color_category='costs'`` and ``legend='shared'``
        should pull template colors for cost labels that exist in the
        shipped YAML and fall back to tab10 for any label that doesn't."""
        import numpy as np
        import pandas as pd

        from flextool.plot_outputs.color_template import resolve_label_color
        from flextool.plot_outputs.plan import compute_live_plan

        tpl = ct.load_color_template()

        # Simulate the ``annualized_costs_d_p`` shape: periods on the row,
        # a single ``parameter`` column level holding cost labels.
        cost_labels = [
            "commodity_cost",      # templated
            "co2",                 # templated
            "upward slack penalty",  # templated
            "not_a_real_cost",     # NOT templated → falls back to tab10
        ]
        periods = ["p1", "p2", "p3"]
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.random((len(periods), len(cost_labels))) * 100,
            index=pd.Index(periods, name="period"),
            columns=pd.MultiIndex.from_arrays([cost_labels], names=["parameter"]),
        )

        cfg = PlotConfig(
            plot_name="Cost bars",
            # Bar chart: row=period (b), col=parameter (s=stack).
            map_dimensions_for_plots=["d_p", "b_s"],
            # Shared legend triggers shared_color_map construction.
            legend="shared",
            color_category="costs",
        )
        plan = compute_live_plan(df, cfg, plot_name="Cost bars")
        assert plan is not None, "compute_live_plan returned None"
        assert plan.shared_color_map is not None, (
            "shared_color_map not populated; check legend='shared' path"
        )

        # Templated labels match the YAML; untemplated ones come from tab10.
        for lbl in ("commodity_cost", "co2", "upward slack penalty"):
            expected = resolve_label_color(lbl, tpl, category="costs")
            assert expected is not None
            got = plan.shared_color_map[lbl]
            assert tuple(got) == pytest.approx(expected), (
                f"Cost label {lbl!r} got {got} but template says {expected}"
            )

        tab10 = plt.colormaps["tab10"].colors
        assert plan.shared_color_map["not_a_real_cost"] == tab10[0], (
            "Untemplated label did not fall back to tab10[0]"
        )

    def test_bar_plan_without_category_uses_palette_only(self):
        """Without ``color_category``, cost labels still get palette colors
        (not template colors) — opt-in only, no silent coupling."""
        import numpy as np
        import pandas as pd

        from flextool.plot_outputs.plan import compute_live_plan

        df = pd.DataFrame(
            np.arange(6, dtype=float).reshape(3, 2),
            index=pd.Index(["p1", "p2", "p3"], name="period"),
            columns=pd.MultiIndex.from_arrays(
                [["commodity_cost", "co2"]], names=["parameter"]
            ),
        )
        cfg = PlotConfig(
            plot_name="Cost bars (no hint)",
            map_dimensions_for_plots=["d_p", "b_s"],
            legend="shared",
            # color_category intentionally unset
        )
        plan = compute_live_plan(df, cfg, plot_name="Cost bars (no hint)")
        assert plan is not None
        assert plan.shared_color_map is not None

        tab10 = plt.colormaps["tab10"].colors
        # Both labels fall back to the palette (alphabetical: co2 then
        # commodity_cost after sort inside _compute_bar_plan).
        assert plan.shared_color_map["co2"] == tab10[0]
        assert plan.shared_color_map["commodity_cost"] == tab10[1]


# ---------------------------------------------------------------------------
# Chunk F: node-flow color-template integration (through the plan pipeline)
# ---------------------------------------------------------------------------


class TestNodeFlowsCategoryIntegration:
    """End-to-end: a bar plan computed with ``color_category='node_flows'``
    pulls colors from ``templates/default_colors.yaml`` for the eight
    node-balance categories emitted by
    :func:`flextool.process_outputs.out_node.node_summary` and falls back
    to the palette for labels not in the template.

    This locks in the wiring for Chunk F the same way
    :class:`TestCostCategoryIntegration` locks in Chunk E.
    """

    def setup_method(self) -> None:
        ct._clear_cache()

    def test_shipped_template_has_all_node_flow_labels(self):
        """The eight category strings that ``out_node.node_summary``
        writes must all resolve.  If any of these drift (rename in
        out_node.py, typo in the YAML), this is the canary."""
        tpl = ct.load_color_template()
        # Exact labels from out_node.node_summary:
        #   flextool/process_outputs/out_node.py:8
        expected_labels = [
            "From units",
            "From connections",
            "Loss of load",
            "To units",
            "To connections",
            "Self discharge",
            "Excess load",
            "Inflow",
        ]
        for lbl in expected_labels:
            assert ct.resolve_label_color(lbl, tpl, category="node_flows") is not None, (
                f"Shipped template is missing node-flow label {lbl!r}"
            )

    def test_bar_plan_uses_template_colors_for_node_flow_labels(self):
        """A bar plan with ``color_category='node_flows'`` and
        ``legend='shared'`` should pull template colors for the node-flow
        category labels that exist in the shipped YAML and fall back to
        tab10 for any label that doesn't."""
        import numpy as np
        import pandas as pd

        from flextool.plot_outputs.color_template import resolve_label_color
        from flextool.plot_outputs.plan import compute_live_plan

        tpl = ct.load_color_template()

        # Simulate the ``node_d_ep`` single-node shape after entity has
        # been expanded out: periods on the row, a single ``category``
        # column level holding the node-flow vocabulary.
        node_flow_labels = [
            "From units",         # templated (green)
            "Loss of load",       # templated (red slack)
            "Inflow",             # templated (light green)
            "not_a_real_flow",    # NOT templated → falls back to tab10
        ]
        periods = ["p1", "p2", "p3"]
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.random((len(periods), len(node_flow_labels))) * 100,
            index=pd.Index(periods, name="period"),
            columns=pd.MultiIndex.from_arrays([node_flow_labels], names=["category"]),
        )

        cfg = PlotConfig(
            plot_name="Node flow bars",
            # Bar chart: row=period (b), col=category (s=stack) — matches
            # the real ``node_d_ep`` rules ``[d_ep, b_es]`` collapsed to a
            # single entity.
            map_dimensions_for_plots=["d_p", "b_s"],
            legend="shared",
            color_category="node_flows",
        )
        plan = compute_live_plan(df, cfg, plot_name="Node flow bars")
        assert plan is not None, "compute_live_plan returned None"
        assert plan.shared_color_map is not None, (
            "shared_color_map not populated; check legend='shared' path"
        )

        for lbl in ("From units", "Loss of load", "Inflow"):
            expected = resolve_label_color(lbl, tpl, category="node_flows")
            assert expected is not None
            got = plan.shared_color_map[lbl]
            assert tuple(got) == pytest.approx(expected), (
                f"Node-flow label {lbl!r} got {got} but template says {expected}"
            )

        tab10 = plt.colormaps["tab10"].colors
        assert plan.shared_color_map["not_a_real_flow"] == tab10[0], (
            "Untemplated label did not fall back to tab10[0]"
        )

    def test_node_flows_and_costs_slacks_use_same_red(self):
        """Cross-category consistency check: the strong-red slack hue used
        for the ``costs`` category (``upward slack penalty``) is re-used
        for the ``node_flows`` category (``Loss of load``), so users learn
        'red = slack / infeasibility' once across the whole viewer."""
        tpl = ct.load_color_template()
        costs_red = ct.resolve_label_color(
            "upward slack penalty", tpl, category="costs",
        )
        flows_red = ct.resolve_label_color(
            "Loss of load", tpl, category="node_flows",
        )
        assert costs_red is not None
        assert flows_red is not None
        assert costs_red == pytest.approx(flows_red), (
            "Slack hues drifted between costs and node_flows categories"
        )


# ---------------------------------------------------------------------------
# Chunk G: flowGroup entity-class color-template integration
# ---------------------------------------------------------------------------


class TestFlowGroupEntityClassIntegration:
    """End-to-end: a bar plan computed with ``color_entity_class='group'``
    pulls colors from ``templates/default_colors.yaml`` for well-known
    flowGroup technology / fuel names (solar, wind, coal, …) and falls
    back to the palette for project-specific names not in the template.

    flowGroup labels are **user-chosen entity names** — a given project's
    group might be called "solar" or "Solar" or "solar_park_DE".  The
    template uses case-insensitive matching so the first two hit the same
    color; the third falls through to tab10.

    This is the flowGroup counterpart to
    :class:`TestCostCategoryIntegration` /
    :class:`TestNodeFlowsCategoryIntegration`.
    """

    def setup_method(self) -> None:
        ct._clear_cache()

    def test_shipped_template_has_recommended_flowgroup_labels(self):
        """Sanity check on the YAML: the conventional flowGroup names that
        the template ships with must all resolve.  Canary if someone
        trims / renames entries in ``default_colors.yaml``."""
        tpl = ct.load_color_template()
        # A representative sample across the semantic groups (renewables,
        # conventional, storage, other).
        expected_labels = [
            "solar", "wind", "hydro", "biomass",
            "coal", "gas", "nuclear",
            "battery", "h2_storage",
            "chp", "import",
        ]
        for lbl in expected_labels:
            assert ct.resolve_label_color(
                lbl, tpl, entity_class="group",
            ) is not None, (
                f"Shipped template is missing flowGroup label {lbl!r}"
            )

    def test_flowgroup_labels_are_case_insensitive(self):
        """User-chosen entity names come in at whatever case the project
        picked; the template must match regardless."""
        tpl = ct.load_color_template()
        for canonical, variants in [
            ("solar", ["Solar", "SOLAR", "sOlAr"]),
            ("wind", ["Wind", "WIND"]),
            ("h2_storage", ["H2_Storage", "H2_STORAGE"]),
        ]:
            base = ct.resolve_label_color(canonical, tpl, entity_class="group")
            assert base is not None
            for v in variants:
                got = ct.resolve_label_color(v, tpl, entity_class="group")
                assert got == pytest.approx(base), (
                    f"{v!r} did not resolve to same color as {canonical!r}"
                )

    def test_bar_plan_uses_template_colors_for_flowgroup_entities(self):
        """A bar plan with ``color_entity_class='group'`` and
        ``legend='shared'`` should pull template colors for well-known
        flowGroup names (case-insensitively) and fall back to tab10 for
        project-specific names the template doesn't know about."""
        import numpy as np
        import pandas as pd

        from flextool.plot_outputs.color_template import resolve_label_color
        from flextool.plot_outputs.plan import compute_live_plan

        tpl = ct.load_color_template()

        # Simulate a flowGroup-indicator frame reshaped so the legend
        # axis carries group names.  Matches the dimension shape
        # ``[d_g, b_s]`` used by group-indicator bar plots that stack
        # by group.
        group_labels = [
            "solar",               # templated (gold)
            "Wind",                # templated (mixed case — still hits)
            "coal_lignite_mix",    # NOT templated → palette fallback
        ]
        periods = ["p1", "p2", "p3"]
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.random((len(periods), len(group_labels))) * 100,
            index=pd.Index(periods, name="period"),
            columns=pd.MultiIndex.from_arrays([group_labels], names=["group"]),
        )

        cfg = PlotConfig(
            plot_name="FlowGroup bars",
            map_dimensions_for_plots=["d_g", "b_s"],
            legend="shared",
            color_entity_class="group",
        )
        plan = compute_live_plan(df, cfg, plot_name="FlowGroup bars")
        assert plan is not None, "compute_live_plan returned None"
        assert plan.shared_color_map is not None, (
            "shared_color_map not populated; check legend='shared' path"
        )

        # Templated labels match the YAML (case-insensitive).
        for lbl in ("solar", "Wind"):
            expected = resolve_label_color(lbl, tpl, entity_class="group")
            assert expected is not None
            got = plan.shared_color_map[lbl]
            assert tuple(got) == pytest.approx(expected), (
                f"FlowGroup label {lbl!r} got {got} but template says {expected}"
            )

        # Project-specific name falls back to tab10[0] (first palette slot
        # since no other untemplated labels precede it).
        tab10 = plt.colormaps["tab10"].colors
        assert plan.shared_color_map["coal_lignite_mix"] == tab10[0], (
            "Untemplated flowGroup label did not fall back to tab10[0]"
        )

    def test_category_and_entity_class_paths_independent(self):
        """Opt-in only: passing ``color_entity_class='group'`` should not
        accidentally pick up colors from the ``category`` section, and
        vice versa.  Locks down the precedence documented in
        :func:`resolve_label_color`."""
        tpl = ct.load_color_template()
        # "solar" is an entity_class.group key; it should resolve under
        # entity_class='group' but NOT under category='costs'.
        assert ct.resolve_label_color(
            "solar", tpl, entity_class="group",
        ) is not None
        assert ct.resolve_label_color(
            "solar", tpl, category="costs",
        ) is None

        # "commodity_cost" is a category.costs key; it should resolve
        # under category='costs' but NOT under entity_class='group'.
        assert ct.resolve_label_color(
            "commodity_cost", tpl, category="costs",
        ) is not None
        assert ct.resolve_label_color(
            "commodity_cost", tpl, entity_class="group",
        ) is None
