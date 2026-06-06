"""Tests for Stage 5 upfront input-DB entity color seeding.

Covers :mod:`flextool.scenario_comparison.input_entity_colors`:

* ``fetch_entities_by_class`` opens one input DB and returns ALL entities in
  the relevant classes grouped by class (unit / connection / node / group),
  in a single pass;
* ``seed_input_entity_colors`` additively seeds the fetched names into a
  project ``plot_settings.yaml`` (preserving existing entries / comments),
  is idempotent on a second run, never writes the bundled package file, and
  skips a non-existent DB URL without error.

The input DB is a tiny throwaway Spine DB built with ``spinedb_api`` in a
tmp fixture — no checked-in ``.sqlite`` is read.
"""

from __future__ import annotations

from pathlib import Path

from spinedb_api import DatabaseMapping

from flextool.scenario_comparison.input_entity_colors import (
    RELEVANT_ENTITY_CLASSES,
    fetch_entities_by_class,
    seed_input_entity_colors,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _build_input_db(path: Path, entities: dict[str, list[str]]) -> str:
    """Build a minimal FlexTool-shaped Spine DB and return its sqlite URL.

    *entities* maps entity_class_name -> list of entity names.  Includes the
    four relevant classes plus an extra irrelevant class to prove filtering.
    """
    url = "sqlite:///" + str(path)
    with DatabaseMapping(url, create=True) as db:
        classes = set(entities) | {"unit", "connection", "node", "group", "commodity"}
        for cls in classes:
            db.add_update_item("entity_class", name=cls)
        for cls, names in entities.items():
            for name in names:
                db.add_update_item(
                    "entity", entity_class_name=cls, name=name, entity_byname=(name,)
                )
        db.commit_session("input entity color test fixture")
    return url


# A project plot_settings.yaml with one pre-existing user entry per class and
# representative comments, to prove additive / preserving behavior.
_SEED_FILE = """\
# Project plot settings (test fixture).

scenarios:

entities:
  group:
    solar: "#F4B400"   # user-set; must be preserved
  unit:
    # commented example: coal_plant: "#212121"
  connection:
  node:
"""


# --------------------------------------------------------------------------
# fetch_entities_by_class
# --------------------------------------------------------------------------


def test_fetch_groups_by_relevant_class_and_filters(tmp_path):
    url = _build_input_db(
        tmp_path / "in.sqlite",
        {
            "unit": ["coal_plant", "battery"],
            "connection": ["AC_link"],
            "node": ["elec_A", "elec_B"],
            "group": ["solar", "wind"],
            "commodity": ["coal", "gas"],  # irrelevant class — must be dropped
        },
    )
    result = fetch_entities_by_class(url)

    assert set(result) == {"unit", "connection", "node", "group"}
    assert "commodity" not in result
    # Names returned sorted + complete (all entities, one pass).
    assert result["unit"] == ["battery", "coal_plant"]
    assert result["connection"] == ["AC_link"]
    assert result["node"] == ["elec_A", "elec_B"]
    assert result["group"] == ["solar", "wind"]


def test_fetch_omits_empty_classes(tmp_path):
    url = _build_input_db(tmp_path / "in.sqlite", {"unit": ["u1"]})
    result = fetch_entities_by_class(url)
    assert result == {"unit": ["u1"]}


def test_relevant_classes_match_schema_subsections():
    # The four classes verified against flextool/schemas/spinedb_schema.json
    # and the plot_settings.yaml entities subsections.
    assert set(RELEVANT_ENTITY_CLASSES) == {"unit", "connection", "node", "group"}


# --------------------------------------------------------------------------
# seed_input_entity_colors
# --------------------------------------------------------------------------


def _project_with_settings(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "plot_settings.yaml").write_text(_SEED_FILE, encoding="utf-8")
    return project


def test_seed_adds_entities_additively(tmp_path):
    url = _build_input_db(
        tmp_path / "in.sqlite",
        {
            "unit": ["coal_plant", "battery"],
            "connection": ["AC_link"],
            "node": ["elec_A"],
            "group": ["solar", "wind"],  # solar already present
        },
    )
    project = _project_with_settings(tmp_path)
    before = (project / "plot_settings.yaml").read_text(encoding="utf-8")

    changed = seed_input_entity_colors(project, [url])
    assert changed is True

    after = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    # Existing user entry preserved verbatim.
    assert 'solar: "#F4B400"   # user-set; must be preserved' in after
    # New names added under each class.
    assert "coal_plant:" in after
    assert "battery:" in after
    assert "AC_link:" in after
    assert "elec_A:" in after
    assert "wind:" in after
    # solar (already present) NOT duplicated.
    assert after.count("solar:") == 1
    # Bundled default file is NOT what was edited.
    assert before != after


def test_seed_is_idempotent(tmp_path):
    url = _build_input_db(
        tmp_path / "in.sqlite",
        {"unit": ["coal_plant"], "node": ["elec_A"], "group": ["wind"]},
    )
    project = _project_with_settings(tmp_path)

    assert seed_input_entity_colors(project, [url]) is True
    first = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    # Second run with the same DB makes no change (byte-identical).
    assert seed_input_entity_colors(project, [url]) is False
    second = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    assert first == second


def test_seed_unions_multiple_distinct_dbs(tmp_path):
    url_a = _build_input_db(tmp_path / "a.sqlite", {"unit": ["u_a"]})
    url_b = _build_input_db(tmp_path / "b.sqlite", {"unit": ["u_b"], "node": ["n_b"]})
    project = _project_with_settings(tmp_path)

    changed = seed_input_entity_colors(project, [url_a, url_b, url_a])
    assert changed is True
    after = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    assert "u_a:" in after
    assert "u_b:" in after
    assert "n_b:" in after


def test_seed_skips_nonexistent_db(tmp_path):
    project = _project_with_settings(tmp_path)
    before = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    missing = "sqlite:///" + str(tmp_path / "does_not_exist.sqlite")

    # No error; nothing seeded.
    changed = seed_input_entity_colors(project, [missing])
    assert changed is False
    after = (project / "plot_settings.yaml").read_text(encoding="utf-8")
    assert before == after


def test_seed_never_writes_bundled_default(tmp_path):
    # A project with NO plot_settings.yaml resolves to the bundled default;
    # seeding must create + write the PROJECT copy, never the package file.
    from flextool.plot_outputs import color_template

    bundled = color_template._default_path()
    bundled_before = bundled.read_text(encoding="utf-8")

    project = tmp_path / "fresh_project"
    project.mkdir()
    assert not (project / "plot_settings.yaml").exists()

    url = _build_input_db(tmp_path / "in.sqlite", {"unit": ["new_unit"]})
    changed = seed_input_entity_colors(project, [url])
    assert changed is True

    # Project copy created and contains the new entity.
    proj_file = project / "plot_settings.yaml"
    assert proj_file.is_file()
    assert "new_unit:" in proj_file.read_text(encoding="utf-8")
    # Bundled package file untouched.
    assert bundled.read_text(encoding="utf-8") == bundled_before


def test_seed_no_urls_returns_false(tmp_path):
    project = _project_with_settings(tmp_path)
    assert seed_input_entity_colors(project, []) is False
