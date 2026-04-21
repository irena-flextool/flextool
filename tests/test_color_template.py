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
