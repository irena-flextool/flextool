"""Tests for Stage 5 upfront input-DB entity color seeding.

Covers :mod:`flextool.scenario_comparison.input_entity_colors`:

* ``fetch_entities_by_class`` opens one input DB and returns ALL entities in
  the relevant classes grouped by class (unit / connection / node / group),
  in a single pass;
* ``seed_input_entity_colors`` structurally seeds the fetched names into a
  project ``plot_settings.yaml`` (preserving existing entries + their values),
  PRUNES entities the DB no longer has (the input DB is the authoritative full
  set), is idempotent on a second run, never writes the bundled package file,
  and skips a non-existent DB URL without error.

The input DB is a tiny throwaway Spine DB built with ``spinedb_api`` in a
tmp fixture — no checked-in ``.sqlite`` is read.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from spinedb_api import DatabaseMapping

from flextool.scenario_comparison.input_entity_colors import (
    RELEVANT_ENTITY_CLASSES,
    fetch_entities_by_class,
    seed_input_entity_colors,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _build_input_db(
    path: Path,
    entities: dict[str, list[str]],
    memberships: dict[str, list[tuple[str, ...]]] | None = None,
) -> str:
    """Build a minimal FlexTool-shaped Spine DB and return its sqlite URL.

    *entities* maps entity_class_name -> list of entity names.  *memberships*
    maps a group-membership relationship class (e.g. ``group__node`` or
    ``group__unit__node``) -> list of ``entity_byname`` tuples; the member
    entities they reference must already appear in *entities*.  Group
    classification (nodeGroup vs flowGroup) is derived from these.  An extra
    irrelevant class is always present to prove filtering.

    Classification is membership-driven: pass ``group__node`` membership to
    make a group a nodeGroup, or a ``group__unit*`` / ``group__connection*``
    membership to make it a flowGroup.  A group with NO membership belongs to
    neither bucket and is dropped.
    """
    memberships = memberships or {}
    url = "sqlite:///" + str(path)
    with DatabaseMapping(url, create=True) as db:
        # 0-dimensional element classes first…
        base_classes = (
            set(entities) | {"unit", "connection", "node", "group", "commodity"}
        )
        for cls in base_classes:
            db.add_update_item("entity_class", name=cls)
        for cls, names in entities.items():
            for name in names:
                db.add_update_item(
                    "entity", entity_class_name=cls, name=name, entity_byname=(name,)
                )
        # …then the multi-dimensional membership relationship classes (their
        # dimensions are the names split on "__") + their member entities.
        for cls in memberships:
            db.add_update_item(
                "entity_class", name=cls, dimension_name_list=tuple(cls.split("__")),
            )
        for cls, bynames in memberships.items():
            for byname in bynames:
                db.add_update_item(
                    "entity", entity_class_name=cls, entity_byname=tuple(byname),
                )
        db.commit_session("input entity color test fixture")
    return url


# A structured project plot_settings.yaml with one pre-existing user entry to
# prove additive / value-preserving behavior.
_SEED_FILE = """\
scenarios: {}

entities:
  nodeGroup: {}
  flowGroup:
    solar: "#F4B400"
  unit: {}
  connection: {}
  node: {}
"""


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
            "group": ["solar", "wind", "elec"],
            "commodity": ["coal", "gas"],  # irrelevant class — must be dropped
        },
        memberships={
            # solar/wind aggregate flows → flowGroup; elec collects nodes →
            # nodeGroup.
            "group__unit__node": [
                ("solar", "coal_plant", "elec_A"),
                ("wind", "battery", "elec_A"),
            ],
            "group__node": [("elec", "elec_A")],
        },
    )
    result = fetch_entities_by_class(url)

    assert set(result) == {"nodeGroup", "flowGroup", "unit", "connection", "node"}
    assert "commodity" not in result and "group" not in result
    # Names returned sorted + complete (all entities, one pass).
    assert result["unit"] == ["battery", "coal_plant"]
    assert result["connection"] == ["AC_link"]
    assert result["node"] == ["elec_A", "elec_B"]
    # Groups split by membership.
    assert result["flowGroup"] == ["solar", "wind"]
    assert result["nodeGroup"] == ["elec"]


def test_group_without_flow_membership_is_nodegroup(tmp_path):
    url = _build_input_db(
        tmp_path / "in.sqlite",
        {"group": ["plain"], "node": ["n1"]},
        memberships={"group__node": [("plain", "n1")]},
    )
    result = fetch_entities_by_class(url)
    assert result["nodeGroup"] == ["plain"]
    assert "flowGroup" not in result


def test_group_without_any_membership_is_dropped(tmp_path):
    """A group with neither node nor flow membership (e.g. ``oil`` /
    ``pumphydro`` with no ``group__node`` / ``group__unit*`` /
    ``group__connection*`` entries) has nothing to colour and must NOT appear
    in nodeGroup (nor flowGroup)."""
    url = _build_input_db(
        tmp_path / "in.sqlite",
        {"group": ["oil", "pumphydro", "elec"], "node": ["n1"]},
        # Only "elec" is a real nodeGroup; oil/pumphydro have no membership.
        memberships={"group__node": [("elec", "n1")]},
    )
    result = fetch_entities_by_class(url)
    assert result["nodeGroup"] == ["elec"]
    assert "oil" not in result.get("nodeGroup", [])
    assert "pumphydro" not in result.get("nodeGroup", [])
    assert "flowGroup" not in result


def test_fetch_omits_empty_classes(tmp_path):
    url = _build_input_db(tmp_path / "in.sqlite", {"unit": ["u1"]})
    result = fetch_entities_by_class(url)
    assert result == {"unit": ["u1"]}


def test_relevant_classes_match_schema_subsections():
    # The output buckets: group is split into nodeGroup/flowGroup; the rest
    # map 1:1 to top-level SpineDB classes.
    assert set(RELEVANT_ENTITY_CLASSES) == {
        "nodeGroup", "flowGroup", "unit", "connection", "node",
    }


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
            "group": ["solar", "wind"],  # solar already present (flowGroup)
        },
        memberships={"group__unit": [("solar", "coal_plant"), ("wind", "battery")]},
    )
    project = _project_with_settings(tmp_path)
    before = (project / "plot_settings.yaml").read_text(encoding="utf-8")

    changed = seed_input_entity_colors(project, [url])
    assert changed is True

    data = _load(project / "plot_settings.yaml")
    # Existing user entry preserved with its value.
    assert data["entities"]["flowGroup"]["solar"] == "#F4B400"
    # New names added under each class.
    assert "coal_plant" in data["entities"]["unit"]
    assert "battery" in data["entities"]["unit"]
    assert "AC_link" in data["entities"]["connection"]
    assert "elec_A" in data["entities"]["node"]
    assert "wind" in data["entities"]["flowGroup"]
    # solar (already present) NOT duplicated.
    assert list(data["entities"]["flowGroup"].keys()).count("solar") == 1
    # The file changed.
    assert before != (project / "plot_settings.yaml").read_text(encoding="utf-8")


def test_seed_prunes_entities_no_longer_in_db(tmp_path):
    """The input DB is the authoritative full set: an entity present in the
    file but absent from the DB is removed on seed."""
    project = _project_with_settings(tmp_path)
    # Add a flowGroup entry that the DB will NOT contain.
    pf = project / "plot_settings.yaml"
    data = _load(pf)
    data["entities"]["flowGroup"]["stale_group"] = "#dddddd"
    pf.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    url = _build_input_db(
        tmp_path / "in.sqlite",
        {"group": ["solar", "wind"], "unit": ["coal_plant"]},
        memberships={"group__unit": [("solar", "coal_plant"), ("wind", "coal_plant")]},
    )
    changed = seed_input_entity_colors(project, [url])
    assert changed is True

    after = _load(pf)
    grp = after["entities"]["flowGroup"]
    # stale_group pruned (not in DB), solar kept, wind added.
    assert "stale_group" not in grp
    assert grp["solar"] == "#F4B400"
    assert "wind" in grp
    assert "coal_plant" in after["entities"]["unit"]


def test_seed_is_idempotent(tmp_path):
    url = _build_input_db(
        tmp_path / "in.sqlite",
        {"unit": ["coal_plant"], "node": ["elec_A"], "group": ["wind"]},
        memberships={"group__unit": [("wind", "coal_plant")]},
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
