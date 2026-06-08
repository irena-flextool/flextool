"""Tests for the structured ``plot_settings.yaml`` color seeder (Stage 6.1).

Covers the structured read/modify/write writer
(:func:`flextool.scenario_comparison.plot_settings_seed.seed_colors_into_plot_settings`)
and the discovery / classification helper
(:func:`flextool.scenario_comparison.config_builder.discover_dispatch_entities`).

The writer is now a plain pyyaml load -> dict -> dump (``sort_keys=False``):
comments are NOT preserved in the project file (the GUI is the editor), but key
ORDER and existing VALUES are.

Key invariants asserted:

* ADD: missing names are appended AFTER existing entries; existing entry order
  and values are never overwritten;
* PRUNE (opt-in): stale entity names are removed from ``entities.*`` but
  ``categories.*`` and ``scenarios`` are never pruned;
* key order is preserved on dump (= stacking order);
* bare ``"#RRGGBB"`` and ``{color, neg_color}`` values round-trip;
* a pass that adds + prunes nothing is a no-op (returns ``False``, file
  untouched);
* the dispatch seed is add-only (does not prune);
* discovery classifies units vs connections vs groups from a representative
  mappings fixture;
* the bundled package file is never written (always the project copy).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

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

# A representative structured settings file: scenarios + categories + entities
# with a populated ``group`` subsection and empty unit / connection / node.
_BASE_FILE = """\
scenarios:
  S0: "#aaaaaa"

categories:
  costs:
    co2: '#4d4d4d'

entities:
  nodeGroup: {}
  flowGroup:
    solar: '#F4B400'
    wind: '#4FC3F7'
  unit: {}
  connection: {}
  node: {}
"""


@pytest.fixture
def settings_file(tmp_path: Path) -> Path:
    p = tmp_path / "plot_settings.yaml"
    p.write_text(_BASE_FILE, encoding="utf-8")
    return p


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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

    # Dispatch group names → nodeGroup; processGroup aggregates → flowGroup.
    assert set(discovered["nodeGroup"]) == {"elec_grid", "heat_grid"}
    assert set(discovered["flowGroup"]) == {"thermal_units", "transmission"}
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
    assert "coal_plant" in discovered["flowGroup"]
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
# Structured writer: ADD
# --------------------------------------------------------------------------

def test_writer_inserts_under_correct_subsections(settings_file: Path):
    changed = seed_colors_into_plot_settings(
        settings_file,
        {
            "flowGroup": {"thermal": "#111111"},
            "unit": {"gas_turbine": "#222222"},
            "connection": {"DC_tie": "#333333"},
        },
        {"S1": "#444444"},
    )
    assert changed is True
    data = _load(settings_file)

    assert data["entities"]["flowGroup"]["thermal"] == "#111111"
    assert data["entities"]["unit"]["gas_turbine"] == "#222222"
    assert data["entities"]["connection"]["DC_tie"] == "#333333"
    assert data["scenarios"]["S1"] == "#444444"


def test_writer_add_preserves_existing_entries_order_and_values(
    settings_file: Path,
):
    """New names are appended AFTER existing ones; existing order + values
    are untouched."""
    seed_colors_into_plot_settings(
        settings_file,
        {"flowGroup": {"thermal": "#111111", "battery": "#999999"}},
        {},
    )
    data = _load(settings_file)
    grp = data["entities"]["flowGroup"]
    # Existing entries first, in original order, with original values.
    assert list(grp.keys()) == ["solar", "wind", "thermal", "battery"]
    assert grp["solar"] == "#F4B400"
    assert grp["wind"] == "#4FC3F7"
    # New entries appended with the supplied colors.
    assert grp["thermal"] == "#111111"
    assert grp["battery"] == "#999999"


def test_writer_never_overwrites_existing_color(settings_file: Path):
    """A wanted name that already exists keeps the USER's color, not the
    seed's."""
    changed = seed_colors_into_plot_settings(
        settings_file,
        {"flowGroup": {"solar": "#000000"}},  # different color, already present
        {},
    )
    assert changed is False  # nothing added
    data = _load(settings_file)
    assert data["entities"]["flowGroup"]["solar"] == "#F4B400"  # unchanged


def test_writer_case_insensitive_dedup(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        'entities:\n  unit:\n    Coal: "#212121"\n', encoding="utf-8"
    )
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"coal": "#222222"}}, {}
    )
    assert changed is False
    data = _load(p)
    assert list(data["entities"]["unit"].keys()) == ["Coal"]


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


def test_writer_no_change_returns_false(settings_file: Path):
    before = settings_file.read_text(encoding="utf-8")
    changed = seed_colors_into_plot_settings(settings_file, {}, {})
    assert changed is False
    assert settings_file.read_text(encoding="utf-8") == before


def test_writer_creates_missing_subsection(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        'entities:\n  group:\n    solar: "#F4B400"\n', encoding="utf-8"
    )
    changed = seed_colors_into_plot_settings(
        p, {"connection": {"AC_link": "#333333"}}, {}
    )
    assert changed is True
    data = _load(p)
    assert data["entities"]["connection"]["AC_link"] == "#333333"
    assert data["entities"]["group"]["solar"] == "#F4B400"


def test_writer_creates_missing_sections(tmp_path: Path):
    """File with no entities / scenarios sections at all."""
    p = tmp_path / "plot_settings.yaml"
    p.write_text("categories:\n  costs:\n    co2: '#4d4d4d'\n", encoding="utf-8")
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"u1": "#111111"}}, {"S1": "#222222"}
    )
    assert changed is True
    data = _load(p)
    assert data["entities"]["unit"]["u1"] == "#111111"
    assert data["scenarios"]["S1"] == "#222222"
    # Original category content preserved.
    assert data["categories"]["costs"]["co2"] == "#4d4d4d"


def test_writer_empty_file(tmp_path: Path):
    """An empty (None) file loads as {} and seeds cleanly."""
    p = tmp_path / "plot_settings.yaml"
    p.write_text("", encoding="utf-8")
    changed = seed_colors_into_plot_settings(
        p, {"unit": {"u1": "#111111"}}, {}
    )
    assert changed is True
    data = _load(p)
    assert data["entities"]["unit"]["u1"] == "#111111"


# --------------------------------------------------------------------------
# Structured writer: order + value round-trip
# --------------------------------------------------------------------------

def test_writer_preserves_key_order_on_dump(tmp_path: Path):
    """sort_keys=False preserves dict insertion order (= stacking order)."""
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n"
        "  unit:\n"
        '    zzz: "#010101"\n'
        '    aaa: "#020202"\n'
        '    mmm: "#030303"\n',
        encoding="utf-8",
    )
    # Add nothing to unit but trigger a write via a new connection entry, so
    # the dump round-trips the unit subsection.
    seed_colors_into_plot_settings(
        p, {"connection": {"c1": "#040404"}}, {}
    )
    data = _load(p)
    # Original (non-alphabetical) order is preserved exactly.
    assert list(data["entities"]["unit"].keys()) == ["zzz", "aaa", "mmm"]


def test_writer_hex_and_neg_color_round_trip(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        "entities:\n"
        "  unit:\n"
        '    coal_plant: "#1f4ea8"\n'
        "    chp:\n"
        '      color: "#E64A19"\n'
        '      neg_color: "#9c3010"\n',
        encoding="utf-8",
    )
    seed_colors_into_plot_settings(p, {"unit": {"new_u": "#abcdef"}}, {})
    data = _load(p)
    # Bare hex round-trips identically.
    assert data["entities"]["unit"]["coal_plant"] == "#1f4ea8"
    # {color, neg_color} dict round-trips identically.
    assert data["entities"]["unit"]["chp"] == {
        "color": "#E64A19",
        "neg_color": "#9c3010",
    }


def test_writer_accepts_dict_color_values(tmp_path: Path):
    """A seeded value may itself be a {color, neg_color} dict."""
    p = tmp_path / "plot_settings.yaml"
    p.write_text("entities:\n  unit: {}\n", encoding="utf-8")
    seed_colors_into_plot_settings(
        p,
        {"unit": {"chp": {"color": "#E64A19", "neg_color": "#9c3010"}}},
        {},
    )
    data = _load(p)
    assert data["entities"]["unit"]["chp"] == {
        "color": "#E64A19",
        "neg_color": "#9c3010",
    }


# --------------------------------------------------------------------------
# Structured writer: PRUNE
# --------------------------------------------------------------------------

def test_writer_prune_removes_stale_entities(settings_file: Path):
    """Entity names not in the keep set are removed; kept ones survive."""
    # Keep only 'solar' (drop 'wind'), add 'newgrp'.
    changed = seed_colors_into_plot_settings(
        settings_file,
        {"flowGroup": {"newgrp": "#111111"}},
        {},
        prune_entities={"flowGroup": {"solar", "newgrp"}},
    )
    assert changed is True
    data = _load(settings_file)
    grp = data["entities"]["flowGroup"]
    assert "wind" not in grp  # pruned
    assert grp["solar"] == "#F4B400"  # kept
    assert grp["newgrp"] == "#111111"  # added


def test_writer_prune_is_case_insensitive(tmp_path: Path):
    p = tmp_path / "plot_settings.yaml"
    p.write_text(
        'entities:\n  unit:\n    Coal: "#212121"\n    Gas: "#333333"\n',
        encoding="utf-8",
    )
    # keep set is lowercased; 'Coal' should survive, 'Gas' should be pruned.
    changed = seed_colors_into_plot_settings(
        p, {}, {}, prune_entities={"unit": {"coal"}}
    )
    assert changed is True
    data = _load(p)
    assert list(data["entities"]["unit"].keys()) == ["Coal"]


def test_writer_prune_never_touches_categories_or_scenarios(
    settings_file: Path,
):
    """Prune only affects entities.*; categories.* and scenarios are safe."""
    seed_colors_into_plot_settings(
        settings_file,
        {},
        {},
        # Empty keep sets for every class -> would prune everything in
        # entities, but categories / scenarios must be untouched.
        prune_entities={"flowGroup": set(), "unit": set(),
                        "connection": set(), "node": set()},
    )
    data = _load(settings_file)
    # Entities fully pruned.
    assert data["entities"]["flowGroup"] == {}
    # categories + scenarios preserved.
    assert data["categories"]["costs"]["co2"] == "#4d4d4d"
    assert data["scenarios"]["S0"] == "#aaaaaa"


def test_writer_prune_none_is_add_only(settings_file: Path):
    """prune_entities=None must not remove anything."""
    seed_colors_into_plot_settings(
        settings_file,
        {"flowGroup": {"newgrp": "#111111"}},
        {},
        prune_entities=None,
    )
    data = _load(settings_file)
    grp = data["entities"]["flowGroup"]
    assert "solar" in grp and "wind" in grp  # nothing pruned
    assert grp["newgrp"] == "#111111"


def test_writer_prune_only_no_add_returns_true_when_stale(settings_file: Path):
    """A pure prune (no adds) that removes something still writes."""
    changed = seed_colors_into_plot_settings(
        settings_file,
        {},
        {},
        prune_entities={"flowGroup": {"solar"}},  # drop wind
    )
    assert changed is True
    data = _load(settings_file)
    assert list(data["entities"]["flowGroup"].keys()) == ["solar"]


def test_writer_prune_only_no_stale_returns_false(settings_file: Path):
    """A prune whose keep set already covers everything is a no-op."""
    before = settings_file.read_text(encoding="utf-8")
    changed = seed_colors_into_plot_settings(
        settings_file,
        {},
        {},
        prune_entities={"flowGroup": {"solar", "wind"}},
    )
    assert changed is False
    assert settings_file.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# Bundled package file safety
# --------------------------------------------------------------------------

def test_writer_never_touches_bundled_package_file():
    """Seeding writes the project copy, never the packaged default."""
    from flextool.plot_outputs.color_template import _default_path

    bundled = Path(_default_path())
    before = bundled.read_text(encoding="utf-8")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        proj = Path(td) / "plot_settings.yaml"
        proj.write_text(before, encoding="utf-8")
        seed_colors_into_plot_settings(
            proj, {"unit": {"brand_new_unit": "#abcabc"}}, {}
        )
    assert bundled.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------
# Orchestrator (dispatch) hook — ADD-ONLY (no prune)
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

    _seed_dispatch_colors_into_plot_settings(
        project, bundled, m, ["S1", "S2"]
    )

    project_file = project / "plot_settings.yaml"
    assert project_file.is_file()
    data = _load(project_file)
    assert "coal_plant" in data["entities"]["unit"]
    assert (
        "AC_link" in data["entities"]["connection"]
        or "DC_tie" in data["entities"]["connection"]
    )
    assert "S1" in data["scenarios"] and "S2" in data["scenarios"]
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


def test_orchestrator_hook_does_not_prune(tmp_path: Path):
    """The dispatch seed sees only a SUBSET of entities, so it must never
    remove entities it didn't discover (add-only)."""
    from flextool.scenario_comparison.orchestrator import (
        _seed_dispatch_colors_into_plot_settings,
    )

    project = tmp_path / "proj"
    project.mkdir()
    # Pre-seed an existing unit that the dispatch mappings do NOT mention.
    pf = project / "plot_settings.yaml"
    pf.write_text(
        'entities:\n  unit:\n    pre_existing_unit: "#abcdef"\n',
        encoding="utf-8",
    )
    m = _make_mappings()  # discovers coal_plant / gas_turbine / wind_farm

    _seed_dispatch_colors_into_plot_settings(project, pf, m, ["S1"])

    data = _load(pf)
    units = data["entities"]["unit"]
    # The un-discovered pre-existing unit must NOT be pruned.
    assert "pre_existing_unit" in units
    # Discovered units were added.
    assert "coal_plant" in units
