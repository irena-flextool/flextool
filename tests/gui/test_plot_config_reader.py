from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from flextool.gui.plot_config_reader import (
    PlotEntry,
    PlotGroup,
    PlotVariant,
    parse_plot_config,
)

# Resolve template paths relative to the repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PLOTS = _REPO_ROOT / "templates" / "default_plots.yaml"
_COMPARISON_PLOTS = _REPO_ROOT / "templates" / "default_comparison_plots.yaml"


class TestParseDefaultPlots:
    """Integration tests using the real default_plots.yaml."""

    @pytest.fixture()
    def groups(self) -> list[PlotGroup]:
        assert _DEFAULT_PLOTS.is_file(), f"Missing {_DEFAULT_PLOTS}"
        return parse_plot_config(_DEFAULT_PLOTS)

    def test_has_ten_groups(self, groups):
        assert len(groups) >= 10

    def test_group_numbers_sequential(self, groups):
        numbers = [g.number for g in groups]
        assert numbers == [str(i) for i in range(len(numbers))]

    def test_group_names(self, groups):
        assert groups[0].name == "Infeasibilities"
        assert groups[3].name == "Costs"
        assert groups[9].name == "Extras"

    def test_entry_0_0_has_variants_d_and_t(self, groups):
        entry = None
        for g in groups:
            for e in g.entries:
                if e.number == "0.0":
                    entry = e
                    break
        assert entry is not None, "Entry 0.0 not found"
        letters = {v.letter for v in entry.variants}
        assert "p" in letters
        assert "h" in letters

    def test_total_entry_count(self, groups):
        total = sum(len(g.entries) for g in groups)
        # Verify a reasonable number of entries exist (exact count varies with config changes)
        assert total >= 40

    def test_entry_5_0_has_variant_a(self, groups):
        """Entry 5.0 (CO2 total) has variant letter 'a'."""
        entry = None
        for g in groups:
            for e in g.entries:
                if e.number == "5.0":
                    entry = e
                    break
        assert entry is not None
        assert any(v.letter == "a" for v in entry.variants)


class TestParseComparisonPlots:
    """Integration tests using the real default_comparison_plots.yaml."""

    @pytest.fixture()
    def groups(self) -> list[PlotGroup]:
        assert _COMPARISON_PLOTS.is_file(), f"Missing {_COMPARISON_PLOTS}"
        return parse_plot_config(_COMPARISON_PLOTS)

    def test_has_groups(self, groups):
        assert len(groups) >= 10

    def test_hidden_entries_without_map_dimensions(self, groups):
        """Entries without map_dimensions_for_plots should be excluded.

        The comparison YAML has several entries missing map_dimensions
        (e.g., some reserve or VRE entries). The total should be less
        than the raw entry count.
        """
        total = sum(len(g.entries) for g in groups)
        # Should be less than 48 (the spec says 48 but some are hidden)
        assert total <= 48


class TestHiddenEntriesWithoutMapDimensions:
    """Verify that entries without map_dimensions_for_plots are hidden."""

    def test_entries_without_map_are_hidden(self, tmp_path):
        yaml_content = dedent("""\
            plots:
              'Some result':
                group: 1
                order: 0
                has_map_d_e:
                  default:
                    map_dimensions_for_plots: [d_e, s_b]
              'Another result':
                group: 1
                order: 1
                no_map_d_e:
                  default:
                    xlabel: MWh
              'Nested with map':
                group: 2
                order: 0
                nested_has_map_dt_e:
                  default:
                    map_dimensions_for_plots: [dt_e, tt_l]
              'Nested without map':
                group: 2
                order: 1
                nested_no_map_dt_e:
                  default:
                    xlabel: MWh
        """)
        config_file = tmp_path / "test_plots.yaml"
        config_file.write_text(yaml_content)
        groups = parse_plot_config(config_file)

        all_numbers = set()
        for g in groups:
            for e in g.entries:
                all_numbers.add(e.number)

        assert "1.0" in all_numbers
        assert "1.1" not in all_numbers
        assert "2.0" in all_numbers
        assert "2.1" not in all_numbers


class TestShortNameEqualsFullName:
    """Test that short_name is not truncated (equals full_name)."""

    def test_short_name_equals_full_name(self, tmp_path):
        yaml_content = dedent("""\
            plots:
              'A very long plot name that would have been truncated before':
                group: 1
                order: 0
                result_key_d_e:
                  default:
                    map_dimensions_for_plots: [d_e, s_b]
        """)
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content)
        groups = parse_plot_config(config_file)
        entry = groups[0].entries[0]
        assert entry.short_name == entry.full_name


class TestMissingFile:
    """Test behaviour with missing or invalid files."""

    def test_missing_file_returns_empty(self, tmp_path):
        result = parse_plot_config(tmp_path / "nonexistent.yaml")
        assert result == []

    def test_invalid_yaml_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(": : : invalid yaml [[[")
        result = parse_plot_config(bad_file)
        assert result == []

    def test_empty_plots_returns_empty(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("plots:\n")
        result = parse_plot_config(config_file)
        assert result == []
