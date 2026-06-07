"""Structured seeding of discovered colors into ``plot_settings.yaml``.

The GUI color/order picker (Stage 6) is now the canonical editor for the
project ``plot_settings.yaml``, so the file is treated as a plain STRUCTURED
data file: load with :mod:`pyyaml`, modify the dict, dump with
``sort_keys=False`` to preserve key order (= stacking order).  Comments are no
longer preserved in the PROJECT file — they live only in the bundled template
(schema reference) and as future UI tooltips/labels.

This module provides the single read/modify/write helper used by both seeds:

* the upfront **input-DB** seed
  (:mod:`flextool.scenario_comparison.input_entity_colors`), which knows the
  FULL per-class entity set from the input DB and therefore ADDS new names and
  PRUNES stale ones; and
* the **dispatch** seed
  (:func:`flextool.scenario_comparison.orchestrator._seed_dispatch_colors_into_plot_settings`),
  which discovers only a SUBSET of entities and therefore ADDS only (never
  prunes).

Add semantics (always): for each entity class / scenario, names not already
present (case-insensitive) are appended AFTER the existing entries in that
subsection.  Existing entry ORDER and existing VALUES are never disturbed (a
user's hand-picked color or order is preserved).

Prune semantics (opt-in, ``prune_entities``): names whose lowercased form is
NOT in the supplied full set for that class are removed from ``entities.<class>``.
Only ``entities.*`` is ever pruned; ``categories.*`` (schema-fixed) and
``scenarios`` are never pruned.

A pass that adds nothing and prunes nothing makes no write (returns ``False``),
so a freshly-copied, still-commented project file keeps its comments until the
first real change.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_ENTITY_CLASSES = ("group", "unit", "connection", "node")


def _load_mapping(path: Path) -> dict:
    """Load *path* as a YAML mapping (``None`` / empty -> ``{}``).

    Raises :class:`ValueError` on a parse error or a non-mapping top level.
    """
    text = path.read_text(encoding="utf-8")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - corrupt file
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return parsed


def _existing_lc(section: dict) -> set[str]:
    """Lowercased key set of *section* (for case-insensitive de-dup)."""
    return {str(k).lower() for k in section}


def _add_entries(
    section: dict,
    wanted: dict[str, str],
) -> bool:
    """Append names from *wanted* not already in *section* (case-insensitive).

    Mutates *section* in place; existing entries (order + value) are left
    untouched.  Returns ``True`` if anything was added.
    """
    existing = _existing_lc(section)
    added = False
    for name, color in wanted.items():
        if str(name).lower() in existing:
            continue
        section[name] = color
        existing.add(str(name).lower())
        added = True
    return added


def _prune_entries(section: dict, keep_lc: set[str]) -> bool:
    """Remove keys of *section* whose lowercase is NOT in *keep_lc*.

    Mutates *section* in place.  Returns ``True`` if anything was removed.
    """
    stale = [k for k in section if str(k).lower() not in keep_lc]
    for k in stale:
        del section[k]
    return bool(stale)


def seed_colors_into_plot_settings(
    path: Path,
    entity_colors: dict[str, dict[str, str]],
    scenario_colors: dict[str, str],
    prune_entities: dict[str, set[str]] | None = None,
) -> bool:
    """Structurally add (and optionally prune) colors in ``plot_settings.yaml``.

    *entity_colors* maps entity class (``group`` / ``unit`` / ``connection`` /
    ``node``) to an ordered ``{name -> color}`` mapping; *scenario_colors* is
    the ``{name -> color}`` mapping for the ``scenarios`` section.  Color
    values are written verbatim (``"#RRGGBB"`` strings, matplotlib color
    names, or ``{color, neg_color}`` dicts — whatever the caller supplies).

    ADD: names not already present (active key, case-insensitively) in their
    target (sub)section are appended AFTER the existing entries.  Existing
    entries — their order and their values — are never overwritten.

    PRUNE (only when *prune_entities* is given): for each class present in
    *prune_entities*, entity names in ``entities.<class>`` whose lowercased
    name is NOT in ``prune_entities[class]`` are removed.  This is how the
    input-DB seed drops entities that no longer exist in the DB.  Only
    ``entities.*`` is pruned — never ``categories.*`` (schema-fixed) nor
    ``scenarios``.  When *prune_entities* is ``None`` the call is add-only.

    The file is loaded with :func:`yaml.safe_load`, modified, and re-dumped
    with ``sort_keys=False`` so KEY ORDER (= stacking order) is preserved and
    ``"#RRGGBB"`` values round-trip as quoted strings.  ``categories.*``,
    ``scenarios``, and any unmanaged keys are preserved.

    Returns ``True`` when the file was modified, ``False`` when nothing was
    added or pruned (the no-op / idempotent case — the file is left untouched
    on disk so a still-commented fresh copy keeps its comments).
    """
    path = Path(path)
    data = _load_mapping(path)

    modified = False

    # --- Entities: add + (optional) prune -----------------------------------
    entities = data.get("entities")
    entities_existed = isinstance(entities, dict)
    if not entities_existed:
        entities = {}

    for cls in _ENTITY_CLASSES:
        wanted = entity_colors.get(cls) or {}
        do_prune = prune_entities is not None and cls in prune_entities
        if not wanted and not do_prune:
            continue

        class_map = entities.get(cls)
        if not isinstance(class_map, dict):
            class_map = {}

        changed_here = False
        if wanted and _add_entries(class_map, wanted):
            changed_here = True
        if do_prune and _prune_entries(class_map, prune_entities[cls]):
            changed_here = True

        if changed_here:
            # Attach the (possibly newly created) class map to the section.
            entities[cls] = class_map
            modified = True

    # Only attach the entities section if we actually changed something and
    # there is content to write — never inject an empty section on a no-op.
    if modified and entities and not entities_existed:
        data["entities"] = entities

    # --- Scenarios: add only -------------------------------------------------
    if scenario_colors:
        scenarios = data.get("scenarios")
        if not isinstance(scenarios, dict):
            scenarios = {}
        if _add_entries(scenarios, scenario_colors):
            data["scenarios"] = scenarios
            modified = True

    if not modified:
        return False

    out = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    path.write_text(out, encoding="utf-8")
    return True
