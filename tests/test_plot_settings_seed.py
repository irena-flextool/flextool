"""Tests for Stage 4.2 dispatch color auto-seeding.

Covers the append-only ``plot_settings.yaml`` writer
(:mod:`flextool.scenario_comparison.plot_settings_seed`) and the
discovery / classification helper
(:func:`flextool.scenario_comparison.config_builder.discover_dispatch_entities`).

Key invariants asserted:

* the writer preserves every existing line (comments, commented examples,
  existing entries, formatting) byte-for-byte in untouched regions;
* it inserts missing names under the correct subsection at the correct
  indentation;
* a second run with no new names is byte-identical (idempotent);
* discovery classifies units vs connections vs groups from a representative
  mappings fixture;
* the bundled package file is never written (always the project copy).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flextool.scenario_comparison.config_builder import (
    assign_palette_colors,
    discover_dispatch_entities,
)
from flextool.scenario_comparison.data_models import DispatchMappings
from flextool.scenario_comparison.plot_settings_seed import (
    seed_colors_into_plot_settings,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# A representative settings file shaped like the bundled default: entities
# with a populated ``group`` subsection and comment-only ``unit`` /
# ``connection`` subsections, plus a comment-only ``scenarios`` section.
_BASE_FILE = """\
# Plot settings for this project.

scenarios:
  # Colors for comparison scenarios, keyed by scenario name, e.g.
  #   S04_battery: "#1f77b4"
  # (Left empty here — fill in per project.)

categories:
  costs:
    co2: '#4d4d4d'

entities:
  group:
    # Recommended colors for user-named flowGroup entities.
    solar:          '#F4B400'   # gold / sun
    wind:           '#4FC3F7'   # sky blue
  unit:
    # Per-project unit colors, keyed by unit name (case-insensitive).
    #   coal_plant: "#212121"
  connection:
    # Per-project connection colors, keyed by connection name.
    #   AC_link:  "#888888"
  node:
    # Per-project node colors.
    #   elec_A:   "#4FC3F7"
"""


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    p = tmp_path / "plot_settings.yaml"
    p.write_text(_BASE_FILE, encoding="utf-8")
    return p


def _make_mappings() -> DispatchMappings:
    """Build a representative mappings object across the discovery fields."""
    m = DispatchMappings()
    # Group aggregates (processGroups / unit- and connectionGroups)
    m.dispatch_groups = pd.DataFrame({"group": ["elec_grid", "heat_grid"]})
    m.processGroup_Unit_to_group = pd.DataFrame(
        {"group_aggregate": ["thermal_units"]}
    )
    m.processGroup_Connection = pd.DataFrame(
        {"group_aggregate": ["transmission"]}
    )
    # Individual units (bare entity names)
    m.processGroup_unit_to_node_members = pd.DataFrame(
        {"unit": ["coal_plant", "gas_turbine"], "node": ["elecA", "elecB"]}
    )
    m.not_in_aggregate_unit_to_node = pd.DataFrame(
        {"process": ["wind_farm"], "node": ["elecA"]}
    )
    # Individual connections (bare entity names)
    m.processGroup_connection_to_node_members = pd.DataFrame(
        {"process": ["AC_link"], "node": ["elecA"]}
    )
    m.not_in_aggregate_connection = pd.DataFrame(
        {"connection": ["DC_tie"]}
    )
    return m


# --------------------------------------------------------------------------
# Discovery / classification
# --------------------------------------------------------------------------

def test_discover_classifies_units_connections_groups():
    m = _make_mappings()
    discovered = discover_dispatch_entities(m, ["S1", "S2"])

    assert set(discovered["group"]) == {
        "elec_grid", "heat_grid", "thermal_units", "transmission",
    }
    assert set(discovered["unit"]) == {"coal_plant", "gas_turbine", "wind_farm"}
    assert set(discovered["connection"]) == {"AC_link", "DC_tie"}
    assert discovered["scenarios"] == ["S1", "S2"]


def test_discover_group_wins_over_individual():
    """A name appearing as both an aggregate and a member is a group."""
    m = DispatchMappings()
    m.processGroup_Unit_to_group = pd.DataFrame(
        {"group_aggregate": ["coal_plant"]}
    )
    m.processGroup_unit_to_node_members = pd.DataFrame(
        {"unit": ["coal_plant"], "node": ["elecA"]}
    )
    discovered = discover_dispatch_entities(m, [])
    assert "coal_plant" in discovered["group"]
    assert "coal_plant" not in discovered["unit"]


def test_discover_empty_scenarios_filtered():
    m = DispatchMappings()
    discovered = discover_dispatch_entities(m, ["S1", "", None])
    assert discovered["scenarios"] == ["S1"]


def test_assign_palette_colors_stable_and_special():
    colors = assign_palette_colors(["a", "b", "internal_losses"])
    assert colors["internal_losses"] == "darkgray"  # special fixed
    assert colors["a"].startswith("#") and len(colors["a"]) == 7
    # Stable: same start index gives same colors.
    assert assign_palette_colors(["a", "b"]) == {
        "a": colors["a"], "b": colors["b"],
    }


# --------------------------------------------------------------------------
# Append-only writer: insertion + preservation
# --------------------------------------------------------------------------

def test_writer_inserts_under_correct_subsections(settings_file: Path):
    changed = seed_colors_into_plot_settings(
        settings_file,
        {
            "group": {"thermal": "#111111"},
            "unit": {"gas_turbine": "#222222"},
            "connection": {"DC_tie": "#333333"},
        },
        {"S1": "#444444"},
    )
    assert changed is True
    out = settings_file.read_text(encoding="utf-8")
    lines = out.splitlines()

    def _sub_block(name: str) -> list[str]:
        start = next(i for i, ln in enumerate(lines) if ln == f"  {name}:")
        block = []
        for ln in lines[start + 1:]:
            if ln and not ln.startswith("   ") and not ln.startswith("  #"):
                if len(ln) - len(ln.lstrip()) <= 2 and ln.strip():
                    break
            block.append(ln)
        return block

    # New group entry sits under group, after the existing solar/wind.
    grp = _sub_block("group")
    assert '    thermal: "#111111"' in grp
    assert any("solar:" in ln for ln in grp)  # existing preserved
    # Unit / connection entries land under their own subsections.
    assert '    gas_turbine: "#222222"' in _sub_block("unit")
    assert '    DC_tie: "#333333"' in _sub_block("connection")
    # Scenario entry under scenarios at 2-space indent.
    assert '  S1: "#444444"' in lines


def test_writer_preserves_existing_content_verbatim(settings_file: Path):
    """Every original line must still be present, in order, untouched."""
    original = settings_file.read_text(encoding="utf-8")
    seed_colors_into_plot_settings(
        settings_file,
        {"unit": {"new_unit": "#abcabc"}},
        {},
    )
    out = settings_file.read_text(encoding="utf-8")
    # Original lines appear as a subsequence (only additions, no edits).
    orig_lines = original.splitlines()
    out_lines = out.splitlines()
    it = iter(out_lines)
    for ol in orig_lines:
        assert ol in it, f"original line lost or reordered: {ol!r}"


def test_writer_idempotent(settings_file: Path):
    seed_colors_into_plot_settings(
        settings_file,
        {"unit": {"coal_plant": "#222222"}},
        {"S1": "#444444"},
    )
    after_first = settings_file.read_text(encoding="utf-8")
    changed = seed_colors_into_plot_settings(
        settings_file,
        {"unit": {"coal_plant": "#999999"}},  # same name, different color
        {"S1": "#999999"},
    )
    assert changed is False
    assert settings_file.read_text(encoding="utf-8") == after_first


def test_writer_skips_commented_example_keys(tmp_path: Path):
    """A name already present only as a commented example is not duplicated."""
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n"
        "  unit:\n"
        "    # coal_plant: \"#212121\"\n",
        encoding="utf-8",
    )
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"coal_plant": "#222222"}}, {}
    )
    assert changed is False
    assert p.read_text(encoding="utf-8").count("coal_plant") == 1


def test_writer_case_insensitive_dedup(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n"
        "  unit:\n"
        "    Coal: \"#212121\"\n",
        encoding="utf-8",
    )
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"coal": "#222222"}}, {}
    )
    assert changed is False


def test_writer_creates_missing_subsection(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n"
        "  group:\n"
        "    solar: \"#F4B400\"\n",
        encoding="utf-8",
    )
    changed = seed_colors_into_plot_settings(
        p, {"connection": {"AC_link": "#333333"}}, {}
    )
    assert changed is True
    out = p.read_text(encoding="utf-8")
    assert "  connection:" in out
    assert '    AC_link: "#333333"' in out
    # Existing group content intact.
    assert '    solar: "#F4B400"' in out


def test_writer_creates_missing_sections(tmp_path: Path):
    """File with no entities / scenarios sections at all."""
    p = tmp_path / "plot_settings.yaml"
    p.write_text("categories:\n  costs:\n    co2: '#4d4d4d'\n", encoding="utf-8")
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"u1": "#111111"}}, {"S1": "#222222"}
    )
    assert changed is True
    out = p.read_text(encoding="utf-8")
    assert "entities:" in out and "  unit:" in out
    assert '    u1: "#111111"' in out
    assert "scenarios:" in out and '  S1: "#222222"' in out
    # Original category content untouched.
    assert "    co2: '#4d4d4d'" in out


def test_writer_handles_section_at_eof_no_trailing_newline(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n  unit:",  # subsection header at EOF, no newline
        encoding="utf-8",
    )
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"u1": "#111111"}}, {}
    )
    assert changed is True
    out = p.read_text(encoding="utf-8")
    assert '    u1: "#111111"' in out


def test_writer_no_change_returns_false(settings_file: Path):
    before = settings_file.read_text(encoding="utf-8")
    changed = seed_colors_into_plot_settings(settings_file, {}, {})
    assert changed is False
    assert settings_file.read_text(encoding="utf-8") == before


def test_writer_never_touches_bundled_package_file():
    """Seeding the orchestrator path must write the project copy, not the
    packaged default."""
    from flextool.plot_outputs.color_template import _default_path

    bundled = Path(_default_path())
    before = bundled.read_text(encoding="utf-8")
    # The seed writer is only ever handed a project path; assert directly that
    # the bundled file is byte-stable across a seed of a separate copy.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td) / "plot_settings.yaml"
        proj.write_text(before, encoding="utf-8")
        seed_colors_into_plot_settings(
            proj, {"unit": {"brand_new_unit": "#abcabc"}}, {}
        )
    assert bundled.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# Orchestrator hook
# --------------------------------------------------------------------------

def test_orchestrator_hook_seeds_project_copy_not_bundled(tmp_path: Path):
    """The orchestrator seed helper, given the bundled fallback path, seeds a
    fresh project ``plot_settings.yaml`` and never writes the package file."""
    from flextool.plot_outputs.color_template import _default_path
    from flextool.scenario_comparison.orchestrator import (
        _seed_dispatch_colors_into_plot_settings,
    )

    bundled = Path(_default_path())
    bundled_before = bundled.read_text(encoding="utf-8")

    project = tmp_path / "proj"
    project.mkdir()
    m = _make_mappings()

    # color_path is the bundled default (no project file yet) → helper must
    # seed the project copy.
    _seed_dispatch_colors_into_plot_settings(
        project, bundled, m, ["S1", "S2"]
    )

    project_file = project / "plot_settings.yaml"
    assert project_file.is_file()
    out = project_file.read_text(encoding="utf-8")
    # Discovered names landed in the project copy.
    assert "coal_plant:" in out
    assert "AC_link:" in out or "DC_tie:" in out
    assert "S1:" in out and "S2:" in out
    # Bundled package file untouched.
    assert bundled.read_text(encoding="utf-8") == bundled_before


def test_orchestrator_hook_idempotent(tmp_path: Path):
    from flextool.plot_outputs.color_template import _default_path
    from flextool.scenario_comparison.orchestrator import (
        _seed_dispatch_colors_into_plot_settings,
    )

    project = tmp_path / "proj"
    project.mkdir()
    m = _make_mappings()
    bundled = Path(_default_path())

    _seed_dispatch_colors_into_plot_settings(project, bundled, m, ["S1"])
    project_file = project / "plot_settings.yaml"
    first = project_file.read_text(encoding="utf-8")

    # Second run resolves to the now-existing project file; nothing new.
    _seed_dispatch_colors_into_plot_settings(
        project, project_file, m, ["S1"]
    )
    assert project_file.read_text(encoding="utf-8") == first
