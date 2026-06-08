"""Upfront seeding of input-DB entity colors into ``plot_settings.yaml``.

The dispatch pipeline already seeds the *dispatch* entities it discovers
(:mod:`flextool.scenario_comparison.config_builder` /
:mod:`flextool.scenario_comparison.orchestrator`).  This module covers the
*regular* (non-dispatch) plot pipeline: when a GUI scenario-execution batch
starts, do one upfront fetch of ALL entities in the relevant entity classes
from each project INPUT DB and additively seed their default-palette colors
into the project's ``plot_settings.yaml``.  The normal time-series / stack
plots then get a complete, editable color list, and the user can reorder /
recolor while the scenarios run.

Design (decided with the user):

* **Upfront, at job start** — source is the INPUT DB, not the output parquet.
* **One DB fetch of ALL entities** in the relevant classes per distinct input
  DB — never per scenario / per alternative (DB fetches are slow; one pass
  gets them all).  If a batch references multiple distinct input DBs, each
  distinct one is fetched once and the results unioned.
* **Single writer in the GUI process** — scenarios run as parallel
  subprocesses, so this seeding must happen once, in the GUI thread, before
  (or while) those subprocesses run.  This module does not write concurrently.

The relevant FlexTool entity classes — verified against the model schema
(``flextool/schemas/spinedb_schema.json``) and the ``entities`` subsections of
``schemas/default_plot_settings.yaml`` — are ``unit``, ``connection``,
``node`` and ``group``: exactly the classes whose entity names appear as plot
legend labels.

This is the GUI-execution path only (the color-editing workflow is GUI).  The
headless CLI path is intentionally out of scope.
"""

from __future__ import annotations

import logging
from pathlib import Path

from spinedb_api import DatabaseMapping

logger = logging.getLogger(__name__)


# ``plot_settings.yaml`` ``entities`` subsections that plot legend labels map
# to.  ``group`` (a top-level SpineDB class) is split by membership into
# ``nodeGroup`` / ``flowGroup`` (see :func:`fetch_entities_by_class`); the
# rest map 1:1 to top-level SpineDB classes.
RELEVANT_ENTITY_CLASSES = ("nodeGroup", "flowGroup", "unit", "connection", "node")

# SpineDB membership relationship classes that make a ``group`` a flowGroup
# (it aggregates FLOWS).  A group with none of these is a nodeGroup (a
# collection of nodes — typically ``group__node``).  The group is the FIRST
# element of the membership entity's byname.
_FLOWGROUP_MEMBERSHIP_CLASSES = (
    "group__unit__node",
    "group__unit",
    "group__connection__node",
    "group__connection",
)


def fetch_entities_by_class(db_url: str) -> dict[str, list[str]]:
    """Open one input DB and return entity names grouped by plot entity class.

    Opens a single :class:`~spinedb_api.DatabaseMapping` (``create=False``),
    reads ALL entity items in ONE pass (units / connections / nodes / groups
    plus the group-membership relationship classes), classifies each
    ``group`` as ``nodeGroup`` or ``flowGroup`` by its membership, and closes
    the mapping.  Returns ``{class -> sorted unique names}`` for the classes
    in :data:`RELEVANT_ENTITY_CLASSES` that have at least one entity.  Never
    queries per scenario / alternative — one fetch returns everything.

    Classification: a group appearing in any
    :data:`_FLOWGROUP_MEMBERSHIP_CLASSES` membership is a **flowGroup**;
    every other group is a **nodeGroup**.

    The base pattern mirrors
    :func:`flextool.export_to_tabular.db_reader._read_entities`
    (``db.get_entity_items()`` grouped by ``entity_class_name``).
    """
    groups: set[str] = set()
    units: set[str] = set()
    connections: set[str] = set()
    nodes: set[str] = set()
    flow_group_names: set[str] = set()

    db = DatabaseMapping(db_url, create=False)
    try:
        for item in db.get_entity_items():
            cls = item["entity_class_name"]
            if cls == "group":
                groups.add(item["name"])
            elif cls == "unit":
                units.add(item["name"])
            elif cls == "connection":
                connections.add(item["name"])
            elif cls == "node":
                nodes.add(item["name"])
            elif cls in _FLOWGROUP_MEMBERSHIP_CLASSES:
                byname = item.get("entity_byname")
                if byname:
                    flow_group_names.add(byname[0])  # group is the 1st member
    finally:
        db.close()

    by_class: dict[str, list[str]] = {
        "nodeGroup": sorted(g for g in groups if g not in flow_group_names),
        "flowGroup": sorted(g for g in groups if g in flow_group_names),
        "unit": sorted(units),
        "connection": sorted(connections),
        "node": sorted(nodes),
    }
    return {cls: names for cls, names in by_class.items() if names}


def _url_to_path(db_url: str) -> Path | None:
    """Return the on-disk path for a ``sqlite:///`` URL, else ``None``.

    Non-sqlite URLs (which FlexTool does not currently use for input DBs)
    return ``None`` and are skipped by the existence check.
    """
    prefix = "sqlite:///"
    if db_url.startswith(prefix):
        return Path(db_url[len(prefix):])
    return None


def seed_input_entity_colors(
    project_path: Path,
    input_db_urls: list[str],
) -> bool:
    """Fetch input-DB entities and seed (add + prune) their colors in settings.

    For each DISTINCT *input_db_url* whose backing sqlite file exists on disk,
    fetch all entities in the relevant classes (one DB open each), union the
    results across DBs, assign default palette colors with the same logic the
    dispatch seeding uses
    (:func:`flextool.scenario_comparison.config_builder.assign_palette_colors`),
    and write them into the project ``plot_settings.yaml`` via the structured
    writer
    (:func:`flextool.scenario_comparison.plot_settings_seed.seed_colors_into_plot_settings`).
    Because the input DB is the AUTHORITATIVE full entity set, this seed passes
    ``prune_entities`` (the full per-class DB name set) so entities no longer
    in the DB are REMOVED as well as new ones added.

    The project file is resolved via
    :func:`flextool.plot_outputs.color_template.resolve_plot_settings_path`;
    if the resolver returns the bundled default (the project has no own file
    yet) the project copy is seeded first via
    :func:`flextool.gui.project_utils.seed_plot_settings` so the packaged file
    is never written.  When the file changes, the color-template cache is
    cleared so any reader in this same process sees the new colors.

    Returns ``True`` when the project file was modified, ``False`` otherwise
    (no new entities / nothing to seed — the idempotent case).

    URLs whose backing file does not exist are skipped (e.g. xlsx sources that
    point at a not-yet-created ``intermediate/<stem>.sqlite``).  This is a
    known limitation: such sources are seeded later, on their next batch, once
    the intermediate DB has been built.
    """
    from flextool.plot_outputs.color_template import (
        _clear_cache,
        _default_path,
        resolve_plot_settings_path,
    )
    from flextool.scenario_comparison.config_builder import assign_palette_colors
    from flextool.scenario_comparison.plot_settings_seed import (
        seed_colors_into_plot_settings,
    )

    # Distinct, existing-on-disk input DBs only.
    seen: set[str] = set()
    distinct_urls: list[str] = []
    for url in input_db_urls:
        if not url or url in seen:
            continue
        seen.add(url)
        path = _url_to_path(url)
        if path is None or not path.is_file():
            # xlsx source -> intermediate/<stem>.sqlite not yet created; skip
            # (known limitation, see docstring).
            continue
        distinct_urls.append(url)

    if not distinct_urls:
        return False

    # Union all entities across the distinct input DBs (one fetch each).
    union: dict[str, set[str]] = {cls: set() for cls in RELEVANT_ENTITY_CLASSES}
    for url in distinct_urls:
        per_db = fetch_entities_by_class(url)
        for cls, names in per_db.items():
            union[cls].update(names)

    entity_colors: dict[str, dict[str, str]] = {}
    prune_entities: dict[str, set[str]] = {}
    for cls in RELEVANT_ENTITY_CLASSES:
        names = sorted(union[cls])
        if names:
            entity_colors[cls] = assign_palette_colors(names)
        # The input DB is the AUTHORITATIVE full entity set per class: prune
        # any entity in the file that the DB no longer has.  Pass the full
        # (possibly empty) lowercased name set for every relevant class so a
        # class that lost all its entities is fully cleared.
        prune_entities[cls] = {n.lower() for n in names}

    if not any(entity_colors.values()):
        return False

    # Resolve to the project's own file; if the resolver handed back the
    # bundled default, seed the project copy first (never write the package
    # file).
    target = resolve_plot_settings_path(project_path)
    if target == Path(_default_path()):
        from flextool.gui.project_utils import seed_plot_settings
        target = seed_plot_settings(project_path)

    changed = seed_colors_into_plot_settings(
        target, entity_colors, {}, prune_entities=prune_entities
    )
    if changed:
        _clear_cache()
        logger.info("Seeded input-DB entity colors into %s", target)
    return changed
