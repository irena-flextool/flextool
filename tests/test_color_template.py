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


class TestResolveLabelColorNewSchema:
    """The new top-level keys ``categories`` / ``entities``, dict entity
    entries with ``neg_color``, and the ``negative`` flag."""

    def test_new_categories_key_exact(self):
        tpl = {"categories": {"costs": {"Solar cost": "#102030"}}}
        c = ct.resolve_label_color("Solar cost", tpl, category="costs")
        assert c == pytest.approx((0x10 / 255, 0x20 / 255, 0x30 / 255))

    def test_new_entities_key_case_insensitive(self):
        tpl = {"entities": {"group": {"Solar": "#abcdef"}}}
        for label in ("Solar", "solar", "SOLAR"):
            c = ct.resolve_label_color(label, tpl, entity_class="group")
            assert c == pytest.approx(
                (0xAB / 255, 0xCD / 255, 0xEF / 255)
            ), f"failed for {label!r}"

    def test_new_key_preferred_over_legacy(self):
        """When both the new and legacy keys are present, the new key wins."""
        tpl = {
            "categories": {"costs": {"a": "#ffffff"}},
            "category": {"costs": {"a": "#000000"}},
        }
        c = ct.resolve_label_color("a", tpl, category="costs")
        assert c == pytest.approx((1.0, 1.0, 1.0))
        tpl_e = {
            "entities": {"group": {"x": "#ffffff"}},
            "entity_class": {"group": {"x": "#000000"}},
        }
        c_e = ct.resolve_label_color("x", tpl_e, entity_class="group")
        assert c_e == pytest.approx((1.0, 1.0, 1.0))

    def test_entity_dict_entry_color_only(self):
        tpl = {"entities": {"unit": {"battery": {"color": "#43A047"}}}}
        c = ct.resolve_label_color("battery", tpl, entity_class="unit")
        assert c == pytest.approx((0x43 / 255, 0xA0 / 255, 0x47 / 255))
        # negative=True with no neg_color → still the positive color.
        c_neg = ct.resolve_label_color(
            "battery", tpl, entity_class="unit", negative=True,
        )
        assert c_neg == pytest.approx((0x43 / 255, 0xA0 / 255, 0x47 / 255))

    def test_entity_dict_entry_neg_color(self):
        tpl = {
            "entities": {
                "unit": {"chp": {"color": "#E64A19", "neg_color": "#9c3010"}}
            }
        }
        pos = ct.resolve_label_color("chp", tpl, entity_class="unit")
        neg = ct.resolve_label_color(
            "chp", tpl, entity_class="unit", negative=True,
        )
        assert pos == pytest.approx((0xE6 / 255, 0x4A / 255, 0x19 / 255))
        assert neg == pytest.approx((0x9C / 255, 0x30 / 255, 0x10 / 255))

    def test_entity_scalar_negative_is_same_color(self):
        """A bare scalar uses the same color on both sides."""
        tpl = {"entities": {"unit": {"coal": "#212121"}}}
        pos = ct.resolve_label_color("coal", tpl, entity_class="unit")
        neg = ct.resolve_label_color(
            "coal", tpl, entity_class="unit", negative=True,
        )
        assert pos == pytest.approx((0x21 / 255, 0x21 / 255, 0x21 / 255))
        assert neg == pytest.approx(pos)

    def test_categories_negative_flag_ignored(self):
        """Categories are scalar; ``negative`` makes no difference."""
        tpl = {"categories": {"costs": {"co2": "#4d4d4d"}}}
        pos = ct.resolve_label_color("co2", tpl, category="costs")
        neg = ct.resolve_label_color(
            "co2", tpl, category="costs", negative=True,
        )
        assert pos == pytest.approx(neg)


class TestShippedTemplateNewSchema:
    """The bundled ``default_plot_settings.yaml`` now uses the new schema
    (``categories`` / ``entities``) and resolves the same colors as before."""

    def setup_method(self) -> None:
        ct._clear_cache()

    def test_bundled_uses_new_top_level_keys(self):
        tpl = ct.load_color_template()
        assert "categories" in tpl
        assert "entities" in tpl
        assert "scenarios" in tpl
        # Legacy keys are gone from the bundled file.
        assert "category" not in tpl
        assert "entity_class" not in tpl
        # entities subsections present; flowGroup ships convention colors,
        # nodeGroup/unit/connection/node start empty.
        entities = tpl["entities"]
        assert "flowGroup" in entities and isinstance(entities["flowGroup"], dict)
        for sub in ("nodeGroup", "unit", "connection", "node"):
            assert sub in entities
            assert entities[sub] in (None, {}), (
                f"{sub} subsection should ship empty"
            )

    def test_bundled_costs_label_color_unchanged(self):
        tpl = ct.load_color_template()
        c = ct.resolve_label_color(
            "commodity_cost", tpl, category="costs",
        )
        assert c == pytest.approx(
            (0xA0 / 255, 0x52 / 255, 0x2D / 255)
        ), "commodity_cost color drifted after schema rename"

    def test_bundled_group_entity_color_unchanged(self):
        tpl = ct.load_color_template()
        c = ct.resolve_label_color("solar", tpl, entity_class="flowGroup")
        assert c == pytest.approx(
            (0xF4 / 255, 0xB4 / 255, 0x00 / 255)
        ), "solar flowGroup color drifted after schema rename"


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
    colors from ``schemas/default_plot_settings.yaml`` for known cost labels and
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
    pulls colors from ``schemas/default_plot_settings.yaml`` for the eight
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
    pulls colors from ``schemas/default_plot_settings.yaml`` for well-known
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
        trims / renames entries in ``default_plot_settings.yaml``."""
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
                lbl, tpl, entity_class="flowGroup",
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
            base = ct.resolve_label_color(canonical, tpl, entity_class="flowGroup")
            assert base is not None
            for v in variants:
                got = ct.resolve_label_color(v, tpl, entity_class="flowGroup")
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
            color_entity_class="flowGroup",
        )
        plan = compute_live_plan(df, cfg, plot_name="FlowGroup bars")
        assert plan is not None, "compute_live_plan returned None"
        assert plan.shared_color_map is not None, (
            "shared_color_map not populated; check legend='shared' path"
        )

        # Templated labels match the YAML (case-insensitive).
        for lbl in ("solar", "Wind"):
            expected = resolve_label_color(lbl, tpl, entity_class="flowGroup")
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
        # "solar" is an entities.flowGroup key; it should resolve under
        # entity_class='flowGroup' but NOT under category='costs'.
        assert ct.resolve_label_color(
            "solar", tpl, entity_class="flowGroup",
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


# ---------------------------------------------------------------------------
# nodeGroup_flows composite-label color lookup
# ---------------------------------------------------------------------------


class TestNodeGroupFlowsComposite:
    """The nodeGroup_flows_* plots stack on two column levels (``type``,
    ``item``) which the legend renders as either ``"<type> | <item>"``
    (bar plans) or Python tuple repr ``"('<type>', '<item>')"``
    (time-series stacks).  The ``category.nodegroup_flows`` section of
    the template resolves both forms by splitting on the separator, then
    trying the fully-qualified ``<type>_<item>`` key (e.g.
    ``slack_upward``) and falling back to the ``<type>`` key alone.
    """

    def setup_method(self) -> None:
        ct._clear_cache()

    def test_shipped_template_group_property_types_resolve(self):
        """The group-property types emitted by
        ``flextool/process_outputs/out_group.py::nodeGroup_flows`` —
        ``inflow``, the three ``internal_losses`` item subtypes and the
        two ``slack`` items — stay category-colored and must resolve on
        the shipped template (``<type>_<item>`` then ``<type>``)."""
        tpl = ct.load_color_template()
        group_property_labels = [
            "inflow | elec",
            "internal_losses | storages",
            "internal_losses | units",
            "internal_losses | connections",
            "slack | upward",
            "slack | downward",
        ]
        for label in group_property_labels:
            assert ct.resolve_label_color(
                label, tpl, category="nodegroup_flows",
            ) is not None, (
                f"Shipped template missing nodegroup_flows label {label!r}"
            )

    def test_shipped_internal_losses_subtypes_are_distinct(self):
        """The three ``internal_losses`` item subtypes must each get their
        own color (requirement: color internal losses separately)."""
        tpl = ct.load_color_template()
        storages = ct.resolve_label_color(
            "internal_losses | storages", tpl, category="nodegroup_flows")
        units = ct.resolve_label_color(
            "internal_losses | units", tpl, category="nodegroup_flows")
        connections = ct.resolve_label_color(
            "internal_losses | connections", tpl, category="nodegroup_flows")
        assert storages != units != connections and storages != connections

    def test_entity_types_route_to_entities_section(self):
        """A per-participant flow type is colored by its ``item`` via the
        entity class fixed by the type — ``from_unitGroup`` items resolve
        in ``entities.flowGroup`` (shipped with convention colors), not via a
        type key (there is none)."""
        tpl = ct.load_color_template()
        # 'wind' ships in entities.flowGroup; the unitGroup item 'Wind'
        # resolves to that exact color (case-insensitive).
        wind = ct.resolve_label_color(
            "from_unitGroup | Wind", tpl, category="nodegroup_flows")
        assert wind == ct._parse_color_value("#4FC3F7")
        # An un-listed unit item falls through to the palette (entities.unit
        # ships empty).
        assert ct.resolve_label_color(
            "from_unit | coal_plant_DE", tpl, category="nodegroup_flows",
        ) is None

    def test_composite_pipe_label_resolves_type_qualified(self):
        """``"slack | upward"`` should hit ``slack_upward`` (more specific
        than the generic ``slack`` fallback)."""
        tpl = ct.load_color_template()
        c = ct.resolve_label_color(
            "slack | upward", tpl, category="nodegroup_flows",
        )
        expected = ct.resolve_label_color(
            "slack_upward", tpl, category="nodegroup_flows",
        )
        assert c is not None
        assert expected is not None
        assert c == pytest.approx(expected)

        c_down = ct.resolve_label_color(
            "slack | downward", tpl, category="nodegroup_flows",
        )
        expected_down = ct.resolve_label_color(
            "slack_downward", tpl, category="nodegroup_flows",
        )
        assert c_down is not None
        assert c_down == pytest.approx(expected_down)
        # And the two slack colors are different (upward vs downward).
        assert c != pytest.approx(c_down)

    def test_composite_tuple_repr_label_resolves(self):
        """Time-series stacks format the label as ``str(tuple)``; the
        resolver must recognize this shape."""
        tpl = ct.load_color_template()
        c = ct.resolve_label_color(
            "('slack', 'upward')", tpl, category="nodegroup_flows",
        )
        expected = ct.resolve_label_color(
            "slack_upward", tpl, category="nodegroup_flows",
        )
        assert c is not None
        assert c == pytest.approx(expected)

    def test_entity_type_pos_neg_colors(self):
        """Entity-routed types color by item; the ``to_*`` (draw) side uses
        the entity's ``neg_color`` when set, else the same ``color``.  Both
        the bare-color and ``{color, neg_color}`` YAML forms must work."""
        tpl = {
            "categories": {"nodegroup_flows": {"inflow": "#7dd87a"}},
            "entities": {
                "unit": {
                    "coal_DE": "#111111",                       # bare color
                    "batt_DE": {"color": "#222222", "neg_color": "#333333"},
                },
                "flowGroup": {"Wind": "#4fc3f7"},
            },
        }
        red = ct._parse_color_value
        # Bare color: from_ and to_ both use it.
        assert ct.resolve_label_color(
            "from_unit | coal_DE", tpl, category="nodegroup_flows") == red("#111111")
        assert ct.resolve_label_color(
            "to_unit | coal_DE", tpl, category="nodegroup_flows") == red("#111111")
        # Dict color: from_ uses color, to_ uses neg_color.
        assert ct.resolve_label_color(
            "from_unit | batt_DE", tpl, category="nodegroup_flows") == red("#222222")
        assert ct.resolve_label_color(
            "to_unit | batt_DE", tpl, category="nodegroup_flows") == red("#333333")
        # unitGroup item routes to entities.flowGroup, not entities.unit.
        assert ct.resolve_label_color(
            "from_unitGroup | Wind", tpl, category="nodegroup_flows") == red("#4fc3f7")

    def test_entity_type_does_not_cross_classes(self):
        """The type fixes the class: a ``from_unit`` item is looked up only
        in ``entities.unit`` — a same-named ``entities.flowGroup`` entry must
        NOT leak in (and vice-versa)."""
        tpl = {
            "entities": {
                "flowGroup": {"shared_name": "#aaaaaa"},
                "unit": {},
            },
        }
        # from_unit looks in entities.unit only → miss → None (palette).
        assert ct.resolve_label_color(
            "from_unit | shared_name", tpl, category="nodegroup_flows") is None
        # from_unitGroup looks in entities.flowGroup → hit.
        assert ct.resolve_label_color(
            "from_unitGroup | shared_name", tpl, category="nodegroup_flows",
        ) == ct._parse_color_value("#aaaaaa")

    def test_composite_unknown_type_returns_none(self):
        """A composite label whose type isn't in the template falls
        through — caller uses the palette."""
        tpl = ct.load_color_template()
        c = ct.resolve_label_color(
            "totally_unknown | foo", tpl, category="nodegroup_flows",
        )
        assert c is None

    def test_non_composite_label_still_resolves(self):
        """Plain (non-composite) labels still work via the exact-key
        lookup — covers the case where a project has a single-level
        column index on this category for some reason."""
        tpl = ct.load_color_template()
        c = ct.resolve_label_color(
            "inflow", tpl, category="nodegroup_flows",
        )
        assert c is not None

    def test_composite_logic_off_for_other_categories(self):
        """Composite splitting is gated to the ``nodegroup_flows``
        category.  For ``costs`` / ``node_flows`` a label containing
        `` | `` must still be matched literally (or return None), not
        split into type/item — protecting existing categories from
        accidental aliasing."""
        tpl = {
            "category": {
                "costs": {
                    # A literal key that happens to contain " | ".
                    "fuel | coal": "#101010",
                    "fuel": "#202020",
                }
            }
        }
        # Exact match on the literal key wins.
        c = ct.resolve_label_color(
            "fuel | coal", tpl, category="costs",
        )
        assert c == pytest.approx((0x10 / 255, 0x10 / 255, 0x10 / 255))
        # And a label whose left half matches "fuel" but without an
        # exact key must NOT silently fall back to "fuel" under costs.
        c2 = ct.resolve_label_color(
            "fuel | unknown", tpl, category="costs",
        )
        assert c2 is None

    def test_cross_category_slack_consistency(self):
        """The upward-slack red is identical across ``costs.upward slack
        penalty``, ``node_flows.Loss of load``, and
        ``nodegroup_flows.slack_upward`` so users learn a single 'red =
        infeasibility' vocabulary.  Same check for the downward slack."""
        tpl = ct.load_color_template()
        costs_red = ct.resolve_label_color(
            "upward slack penalty", tpl, category="costs",
        )
        flows_red = ct.resolve_label_color(
            "Loss of load", tpl, category="node_flows",
        )
        group_red = ct.resolve_label_color(
            "slack_upward", tpl, category="nodegroup_flows",
        )
        group_red_composite = ct.resolve_label_color(
            "slack | upward", tpl, category="nodegroup_flows",
        )
        assert costs_red is not None
        assert flows_red is not None
        assert group_red is not None
        assert group_red_composite is not None
        assert costs_red == pytest.approx(flows_red)
        assert costs_red == pytest.approx(group_red)
        assert costs_red == pytest.approx(group_red_composite), (
            "Composite-label lookup drifted from direct-key lookup"
        )

        # Downward slack: same check against costs.downward slack
        # penalty + node_flows.Excess load.
        costs_pink = ct.resolve_label_color(
            "downward slack penalty", tpl, category="costs",
        )
        flows_pink = ct.resolve_label_color(
            "Excess load", tpl, category="node_flows",
        )
        group_pink_composite = ct.resolve_label_color(
            "slack | downward", tpl, category="nodegroup_flows",
        )
        assert costs_pink == pytest.approx(flows_pink)
        assert costs_pink == pytest.approx(group_pink_composite)


class TestNodeGroupFlowsPlanIntegration:
    """End-to-end: a bar plan computed with
    ``color_category='nodegroup_flows'`` and a 2-level column MultiIndex
    (``type``, ``item``) pulls template colors for known composite
    labels and falls back to the palette for unknown ones.

    Mirrors :class:`TestCostCategoryIntegration` /
    :class:`TestNodeFlowsCategoryIntegration` for the nodeGroup_flows_*
    plots.
    """

    def setup_method(self) -> None:
        ct._clear_cache()

    def test_bar_plan_colors_composite_labels_by_item(self, tmp_path):
        import numpy as np
        import pandas as pd

        from flextool.plot_outputs.color_template import resolve_label_color
        from flextool.plot_outputs.plan import compute_live_plan

        # Custom template: entity colors for the two units (so from_unit
        # routes to them), plus the group-property specials.
        color_yaml = tmp_path / "plot_settings.yaml"
        color_yaml.write_text(
            "categories:\n"
            "  nodegroup_flows:\n"
            "    inflow: '#7dd87a'\n"
            "    slack_upward: '#d62728'\n"
            "    slack_downward: '#ff7f7f'\n"
            "entities:\n"
            "  unit:\n"
            "    coal_DE: '#111111'\n"
            "    gas_DE: '#999999'\n",
            encoding="utf-8",
        )
        tpl = ct.load_color_template(color_yaml)

        # Simulate the ``nodeGroup_flows_d_gpe`` shape (single-group
        # slice): periods on the row; columns have ``type`` and ``item``
        # levels that stack on the bar.
        type_item_pairs = [
            ("slack", "upward"),        # → slack_upward (category red)
            ("slack", "downward"),      # → slack_downward (category light red)
            ("from_unit", "coal_DE"),   # → entities.unit.coal_DE (by item)
            ("from_unit", "gas_DE"),    # → entities.unit.gas_DE (distinct)
            ("inflow", "nodeA"),        # → category inflow
            ("mystery", "thing"),       # → no match, palette fallback
        ]
        periods = ["p1", "p2", "p3"]
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            rng.random((len(periods), len(type_item_pairs))) * 100,
            index=pd.Index(periods, name="period"),
            columns=pd.MultiIndex.from_tuples(
                type_item_pairs, names=["type", "item"],
            ),
        )

        cfg = PlotConfig(
            plot_name="NodeGroup flows",
            map_dimensions_for_plots=["d_pe", "b_ss"],
            legend="shared",
            color_category="nodegroup_flows",
        )
        plan = compute_live_plan(
            df, cfg, plot_name="NodeGroup flows", color_path=color_yaml,
        )
        assert plan is not None, "compute_live_plan returned None"
        assert plan.shared_color_map is not None

        cmap = plan.shared_color_map
        # Specials keep their category colors.
        assert tuple(cmap["slack | upward"]) == pytest.approx(
            resolve_label_color("slack_upward", tpl, category="nodegroup_flows"))
        assert tuple(cmap["slack | downward"]) == pytest.approx(
            resolve_label_color("slack_downward", tpl, category="nodegroup_flows"))
        assert tuple(cmap["inflow | nodeA"]) == pytest.approx(
            resolve_label_color("inflow", tpl, category="nodegroup_flows"))
        # Entity types colored by item — coal_DE and gas_DE get their OWN
        # (distinct) entity colors, no longer collapsed to one type color.
        assert tuple(cmap["from_unit | coal_DE"]) == pytest.approx(
            ct._parse_color_value("#111111"))
        assert tuple(cmap["from_unit | gas_DE"]) == pytest.approx(
            ct._parse_color_value("#999999"))
        assert cmap["from_unit | coal_DE"] != cmap["from_unit | gas_DE"]
        # Unknown type — palette fallback (only un-templated label → tab10[0]).
        tab10 = plt.colormaps["tab10"].colors
        assert cmap["mystery | thing"] == tab10[0]


# ---------------------------------------------------------------------------
# template_label_order / order_labels_by_template  (file-order stacking)
# ---------------------------------------------------------------------------


class TestTemplateLabelOrder:
    def test_category_keys_in_file_order(self):
        tpl = {"categories": {"costs": {"b": 1, "a": 2, "c": 3}}}
        assert ct.template_label_order(tpl, category="costs") == ["b", "a", "c"]

    def test_entity_keys_in_file_order(self):
        tpl = {"entities": {"unit": {"Zeb": "#111", "abe": "#222"}}}
        assert ct.template_label_order(tpl, entity_class="unit") == ["Zeb", "abe"]

    def test_legacy_keys_honored(self):
        tpl = {"category": {"costs": {"x": 1, "y": 2}}}
        assert ct.template_label_order(tpl, category="costs") == ["x", "y"]
        tpl2 = {"entity_class": {"unit": {"x": 1, "y": 2}}}
        assert ct.template_label_order(tpl2, entity_class="unit") == ["x", "y"]

    def test_entity_class_wins_over_category(self):
        tpl = {
            "categories": {"costs": {"c1": 1}},
            "entities": {"unit": {"u1": 1, "u2": 2}},
        }
        assert ct.template_label_order(
            tpl, category="costs", entity_class="unit"
        ) == ["u1", "u2"]

    def test_missing_section_returns_empty(self):
        assert ct.template_label_order({}, category="costs") == []
        assert ct.template_label_order({"categories": {}}, category="costs") == []
        assert ct.template_label_order({"x": 1}) == []


class TestOrderLabelsByTemplate:
    def test_listed_first_then_alpha_tail(self):
        tpl = {"categories": {"costs": {"b": 1, "a": 2, "c": 3}}}
        labels = ["z", "a", "y", "c", "b"]  # z,y not listed
        out = ct.order_labels_by_template(labels, tpl, category="costs")
        # listed in file order b,a,c then unlisted alpha y,z
        assert out == ["b", "a", "c", "y", "z"]

    def test_no_section_falls_back_to_alpha(self):
        labels = ["c", "a", "b"]
        assert ct.order_labels_by_template(labels, {}, category="costs") == [
            "a", "b", "c",
        ]

    def test_entity_case_insensitive_returns_data_label(self):
        # File lists canonical 'Solar','Wind'; data labels differ in case.
        tpl = {"entities": {"group": {"Solar": "#1", "Wind": "#2"}}}
        labels = ["wind", "SOLAR", "extra"]
        out = ct.order_labels_by_template(labels, tpl, entity_class="group")
        # Ordered by file order (Solar then Wind), preserving the DATA label
        # spelling; 'extra' unlisted appended.
        assert out == ["SOLAR", "wind", "extra"]

    def test_only_listed_present_subset(self):
        tpl = {"categories": {"costs": {"a": 1, "b": 2, "c": 3}}}
        labels = ["c", "a"]  # b absent from data
        assert ct.order_labels_by_template(labels, tpl, category="costs") == [
            "a", "c",
        ]

    def test_composite_category_ordering(self):
        # nodegroup_flows orders by fixed type band order, then within an
        # entity-routed band by that entity class's file order, and within a
        # special band by categories file order.
        tpl = {
            "categories": {
                "nodegroup_flows": {
                    "inflow": "#1", "slack_upward": "#2", "slack_downward": "#3",
                }
            },
            "entities": {"flowGroup": {"Wind": "#a", "Solar": "#b"}},
        }
        labels = [
            "to_unitGroup | Solar",      # draw band (last), flowGroup order Solar
            "from_unitGroup | Solar",    # supply band, flowGroup order Solar (1)
            "from_unitGroup | Wind",     # supply band, flowGroup order Wind (0)
            "inflow | n",                # inflow band
            "slack | downward",          # slack band
        ]
        out = ct.order_labels_by_template(
            labels, tpl, category="nodegroup_flows"
        )
        assert out == [
            "from_unitGroup | Wind",
            "from_unitGroup | Solar",
            "inflow | n",
            "slack | downward",
            "to_unitGroup | Solar",
        ]

    def test_all_unlisted_is_pure_alpha(self):
        tpl = {"categories": {"costs": {"x": 1}}}
        labels = ["c", "a", "b"]
        assert ct.order_labels_by_template(labels, tpl, category="costs") == [
            "a", "b", "c",
        ]

    def test_no_label_dropped_or_duplicated(self):
        tpl = {"categories": {"costs": {"a": 1, "b": 2}}}
        labels = ["b", "x", "a", "y"]
        out = ct.order_labels_by_template(labels, tpl, category="costs")
        assert sorted(out) == sorted(labels)
        assert len(out) == len(labels)


# ---------------------------------------------------------------------------
# Dispatch color / order resolution (Stage 4.1)
# ---------------------------------------------------------------------------


class TestResolveDispatchColorsAndOrder:
    """`resolve_dispatch_colors_and_order` — the three dispatch column kinds.

    Kind 1: bare entity names (processGroup aggregates / units / connections)
            → ``entities`` (case-insensitive across group/unit/connection/node).
    Kind 2: special tokens (LossOfLoad, Charge, internal_losses, …)
            → ``categories.dispatch`` (exact); fixed pos/neg positions, so
            NOT placed in ``config_order``.
    Kind 3: composite ``(process, node)`` / ``(connection)`` reprs and
            node-level ``<unit>_out`` / ``<conn>_left`` columns (with optional
            ``_pos`` / ``_neg``) → entity name extracted, then ``entities``.
    """

    TEMPLATE = {
        "categories": {
            "dispatch": {
                "LossOfLoad": "crimson",
                "Charge": "lime",
                "internal_losses": "darkgray",
            }
        },
        "entities": {
            "flowGroup": {"wind": "#4FC3F7", "coal": "#212121"},
            "unit": {
                "chp": {"color": "#E64A19", "neg_color": "#9c3010"},
                "battery": "#43A047",
            },
            "connection": {"AC_link": "#888888"},
        },
    }

    def test_special_tokens_resolve_to_dispatch_category(self):
        colors, order = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["LossOfLoad", "Charge", "internal_losses"]
        )
        assert colors["LossOfLoad"] == "crimson"
        assert colors["Charge"] == "lime"
        assert colors["internal_losses"] == "darkgray"
        # Special tokens keep their pipeline-fixed positions → not in order.
        assert order == []

    def test_bare_entity_names_resolve_and_order(self):
        colors, order = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["coal", "wind"]
        )
        assert colors["coal"] == "#212121"
        assert colors["wind"] == "#4FC3F7"
        # File order is wind, coal (group section) → top→bottom then reversed.
        assert order == ["coal", "wind"]

    def test_case_insensitive_entity_lookup(self):
        colors, _ = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["WIND", "Coal"]
        )
        assert colors["WIND"] == "#4FC3F7"
        assert colors["Coal"] == "#212121"

    def test_composite_process_node_column(self):
        colors, _ = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["(coal, elecA)", "(AC_link)"]
        )
        assert colors["(coal, elecA)"] == "#212121"
        assert colors["(AC_link)"] == "#888888"

    def test_node_level_suffix_columns(self):
        colors, _ = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE,
            ["coal_out", "battery_in", "AC_link_left", "AC_link_right"],
        )
        assert colors["coal_out"] == "#212121"
        assert colors["battery_in"] == "#43A047"
        assert colors["AC_link_left"] == "#888888"
        assert colors["AC_link_right"] == "#888888"

    def test_neg_color_on_split_parts(self):
        # Post-split columns as emitted by _order_dispatch_columns.
        colors, _ = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["chp_pos", "chp_neg"]
        )
        assert colors["chp_pos"] == "#E64A19"   # positive side
        assert colors["chp_neg"] == "#9c3010"   # distinct neg_color

    def test_neg_color_registered_for_whole_mixed_column(self):
        # A whole (not-yet-split) mixed column registers <col>_neg too.
        colors, _ = ct.resolve_dispatch_colors_and_order(self.TEMPLATE, ["chp"])
        assert colors["chp"] == "#E64A19"
        assert colors["chp_neg"] == "#9c3010"

    def test_bare_color_uses_same_color_both_sides(self):
        colors, _ = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["battery_pos", "battery_neg"]
        )
        # battery has no neg_color → both sides identical, no separate key.
        assert colors["battery_pos"] == "#43A047"
        assert colors["battery_neg"] == "#43A047"

    def test_unresolved_columns_omitted(self):
        colors, order = ct.resolve_dispatch_colors_and_order(
            self.TEMPLATE, ["unknown_unit", "(mystery, n1)"]
        )
        assert "unknown_unit" not in colors
        assert "(mystery, n1)" not in colors
        assert order == []

    def test_empty_template_is_pure_fallback(self):
        colors, order = ct.resolve_dispatch_colors_and_order(
            {}, ["coal", "LossOfLoad", "chp_neg"]
        )
        assert colors == {}
        assert order == []

    def test_template_entity_names_file_order(self):
        names = ct.template_entity_names(self.TEMPLATE)
        # flowGroup (wind, coal) then unit (chp, battery) then connection.
        assert names == ["wind", "coal", "chp", "battery", "AC_link"]
