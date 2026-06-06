"""Loader and resolver for the per-project plot settings (colors).

The settings file maps legend labels to explicit RGB colors.  It has
these label sections:

* ``categories`` — keyed by a category name declared by a plot entry via
  ``color_category``.  Label lookup is *exact* (these are parameter or
  result names whose casing is controlled by the pipeline).
* ``entities`` — keyed by an entity-class name (``group`` / ``unit`` /
  ``connection`` / ``node``) declared via ``color_entity_class``.  Label
  lookup is *case-insensitive* (entity names come from the data and can be
  mixed-case).  An entity entry may be a bare color or a mapping
  ``{color: ..., neg_color: ...}`` whose ``neg_color`` colors the
  entity's negative-side part of a mixed (both-signs) stacked column.

For backward compatibility the resolver also reads the legacy top-level
keys ``category`` / ``entity_class`` written by older projects' seeded
``plot_settings.yaml`` (migration window).

Any label not found falls back to the tab10/tab20 palette used by
:func:`build_shared_color_map`.

The bundled starting-point file lives at
``schemas/default_plot_settings.yaml`` and is copied into each project as
``plot_settings.yaml``.  The loaded result is cached at the module level
keyed by ``(path, mtime)`` so repeated plot builds within the same process
don't re-read it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# (resolved_path, mtime_ns) -> parsed dict
_TEMPLATE_CACHE: dict[tuple[str, int], dict] = {}


def _default_path() -> Path:
    from flextool._resources import package_data_path
    return package_data_path("schemas/default_plot_settings.yaml")


def resolve_plot_settings_path(project_path: Path | None) -> Path:
    """Resolve the plot color template path for *project_path*.

    Returns ``<project_path>/plot_settings.yaml`` when that file exists,
    otherwise the bundled starting point
    (``schemas/default_plot_settings.yaml``).  This
    keeps behavior byte-identical to before for any project that has no
    ``plot_settings.yaml`` while letting a per-project file override the
    template for both PNG export and the result viewer.
    """
    if project_path is not None:
        candidate = Path(project_path) / "plot_settings.yaml"
        if candidate.is_file():
            return candidate
    return _default_path()


def load_color_template(path: Path | None = None) -> dict:
    """Load the color template YAML from *path* (or the default location).

    Returns an empty dict if the file is missing or unreadable, so
    callers can treat "no template" uniformly.  The result is cached
    keyed by ``(resolved path, mtime_ns)`` — a subsequent call with an
    unchanged file hits the cache.
    """
    resolved = Path(path) if path is not None else _default_path()
    try:
        st = resolved.stat()
    except OSError:
        return {}

    cache_key = (str(resolved), st.st_mtime_ns)
    cached = _TEMPLATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        logger.warning("Failed to load color template %s: %s", resolved, exc)
        _TEMPLATE_CACHE[cache_key] = {}
        return {}

    if not isinstance(data, dict):
        data = {}

    # Also prune any stale entries for the same path with a different
    # mtime — keeps the cache from growing unbounded over long sessions.
    for key in list(_TEMPLATE_CACHE):
        if key[0] == str(resolved) and key != cache_key:
            del _TEMPLATE_CACHE[key]

    _TEMPLATE_CACHE[cache_key] = data
    return data


def _parse_color_value(value) -> tuple[float, float, float] | None:
    """Normalize a YAML color value to an ``(r, g, b)`` float tuple in 0..1.

    Accepts:
    * ``#RRGGBB`` hex strings (with or without the leading ``#``).
    * ``[r, g, b]`` lists/tuples of ints (0..255) or floats (0..1).

    Returns ``None`` for anything malformed; callers log the context.
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().lstrip("#")
        if len(s) != 6:
            return None
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
        except ValueError:
            return None
        return (r / 255.0, g / 255.0, b / 255.0)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            parts = [float(x) for x in value]
        except (TypeError, ValueError):
            return None
        # Decide 0..1 vs 0..255 by the max component.
        max_part = max(parts)
        min_part = min(parts)
        if min_part < 0:
            return None
        if max_part <= 1.0:
            # floats in 0..1
            if any(p > 1.0 for p in parts):
                return None
            return (parts[0], parts[1], parts[2])
        if max_part <= 255.0:
            return (parts[0] / 255.0, parts[1] / 255.0, parts[2] / 255.0)
        return None
    return None


# Categories whose labels are composite ``(type, item)`` tuples rendered
# into legend strings in one of two shapes:
#   * ``"<type> | <item>"``   — bar plans (via _format_legend_labels)
#   * ``"('<type>', '<item>')"`` — time-series stacks (str(tuple))
# For these categories we split the label, then try
#   1. ``<type>_<item>``   (e.g. ``slack_upward``)
#   2. ``<type>`` alone    (e.g. ``unit``)
# before falling through.  Plain (non-composite) labels still fall through
# to the exact-key lookup below.
_COMPOSITE_CATEGORIES: frozenset[str] = frozenset({"nodegroup_flows"})


def _split_composite_label(label: str) -> tuple[str, str] | None:
    """Split a composite legend label into ``(type, item)`` if possible.

    Recognizes the two formats emitted by the plot pipeline for two-level
    column stacks:

    * ``"<type> | <item>"`` — produced by ``_format_legend_labels`` in the
      bar-plan path.
    * ``"('<type>', '<item>')"`` — produced by ``str(item)`` in the
      time-series stack path when ``item`` is a 2-tuple.

    Returns ``None`` for labels that don't match either shape (callers
    should treat the label as non-composite and try a plain key lookup).
    """
    if not isinstance(label, str):
        return None

    # " | " separator (bar plans).
    if " | " in label:
        parts = label.split(" | ", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

    # Python tuple repr (time-series stacks).  Use ast.literal_eval for
    # safety — the string came from str(tuple(...)) of hashable scalars so
    # it should parse cleanly; anything else falls through.
    s = label.strip()
    if s.startswith("(") and s.endswith(")"):
        try:
            import ast
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, tuple) and len(parsed) == 2:
            a, b = parsed
            if isinstance(a, str) and isinstance(b, str) and a and b:
                return a, b
    return None


def _resolve_entity_value(raw):
    """Normalize an entity entry into ``(color_value, neg_color_value)``.

    An entity entry may be either a bare color scalar (``"#RRGGBB"`` or
    ``[r, g, b]``) or a mapping ``{color: ..., neg_color: ...}``.  Returns
    the positive-side value and the optional negative-side value (the
    latter ``None`` when the entry is a scalar or has no ``neg_color``).
    """
    if isinstance(raw, dict):
        return raw.get("color"), raw.get("neg_color")
    return raw, None


def resolve_label_color(
    label: str,
    template: dict,
    category: str | None = None,
    entity_class: str | None = None,
    negative: bool = False,
) -> tuple[float, float, float] | None:
    """Resolve *label* to a color via the plot-settings *template*.

    Lookup precedence:
    * If *entity_class* is given, check the ``entities`` section
      (legacy ``entity_class``) under ``[entity_class]`` with a
      case-insensitive key match.  An entity entry may be a bare color or
      a mapping ``{color: ..., neg_color: ...}``.  When *negative* is True
      and the entry has a ``neg_color``, that color is returned; otherwise
      the positive ``color`` is used.
    * Else if *category* is given, check the ``categories`` section
      (legacy ``category``) under ``[category]`` with an exact key match.
      For composite-label categories (see ``_COMPOSITE_CATEGORIES``) the
      label is first split on ``" | "`` or parsed as a 2-tuple repr and
      looked up as ``<type>_<item>`` then ``<type>`` before falling back
      to the exact-key lookup.  Categories are always scalar; *negative*
      has no effect.
    * Otherwise return ``None`` and let the caller fall back to the palette.

    Both the new top-level keys (``categories`` / ``entities``) and the
    legacy keys (``category`` / ``entity_class``) are accepted so that
    projects whose seeded ``plot_settings.yaml`` still uses the old schema
    keep working (migration window).

    Returns a normalized ``(r, g, b)`` float tuple in ``0..1`` or
    ``None`` if the template has no entry (or the entry is malformed).
    Malformed values are logged but do not raise.
    """
    if not isinstance(template, dict) or label is None:
        return None

    raw = None
    if entity_class:
        section = template.get("entities")
        if not isinstance(section, dict):
            section = template.get("entity_class")
        if isinstance(section, dict):
            class_map = section.get(entity_class)
            if isinstance(class_map, dict):
                # Case-insensitive lookup.  Build a lowercase view on the
                # fly — these dicts are small (tens of entries at most).
                label_lc = str(label).lower()
                for key, val in class_map.items():
                    if str(key).lower() == label_lc:
                        color_val, neg_val = _resolve_entity_value(val)
                        raw = neg_val if (negative and neg_val is not None) else color_val
                        break
    elif category:
        section = template.get("categories")
        if not isinstance(section, dict):
            section = template.get("category")
        if isinstance(section, dict):
            cat_map = section.get(category)
            if isinstance(cat_map, dict):
                # Composite categories: split the label and try the
                # type-qualified key first, then the type alone.  This
                # lets the YAML express "all unit flows are green" while
                # also letting specific sub-types (slack_upward vs
                # slack_downward) pin their own color.
                if category in _COMPOSITE_CATEGORIES:
                    parts = _split_composite_label(label)
                    if parts is not None:
                        type_key, item_key = parts
                        raw = cat_map.get(f"{type_key}_{item_key}")
                        if raw is None:
                            raw = cat_map.get(type_key)
                # Fall back to exact-key lookup on the original label
                # (covers non-composite labels and any YAML entries that
                # happen to match the joined form verbatim).
                if raw is None:
                    raw = cat_map.get(label)

    if raw is None:
        return None

    color = _parse_color_value(raw)
    if color is None:
        logger.warning(
            "Color template: malformed value %r for label %r "
            "(category=%r, entity_class=%r)",
            raw, label, category, entity_class,
        )
    return color


def template_label_order(
    template: dict,
    category: str | None = None,
    entity_class: str | None = None,
) -> list[str]:
    """Return the label keys of a template section in **file order**.

    This is the single source of file-defined stacking/legend order.  Given
    a loaded *template* and an optional *category* or *entity_class*, it
    returns the keys of ``categories[category]`` (exact) or
    ``entities[entity_class]`` (case-insensitive, but the file's canonical
    key spelling is returned) in YAML insertion order — which is dict
    insertion order, preserved by the loader.

    Both the new top-level keys (``categories`` / ``entities``) and the
    legacy keys (``category`` / ``entity_class``) are honored (migration
    window), matching :func:`resolve_label_color`.

    Returns an empty list when the template has no matching section.  When
    both *category* and *entity_class* are given, *entity_class* wins
    (matching the lookup precedence in :func:`resolve_label_color`).
    """
    if not isinstance(template, dict):
        return []

    if entity_class:
        section = template.get("entities")
        if not isinstance(section, dict):
            section = template.get("entity_class")
        if isinstance(section, dict):
            class_map = section.get(entity_class)
            if isinstance(class_map, dict):
                return [str(k) for k in class_map.keys()]
        return []

    if category:
        section = template.get("categories")
        if not isinstance(section, dict):
            section = template.get("category")
        if isinstance(section, dict):
            cat_map = section.get(category)
            if isinstance(cat_map, dict):
                return [str(k) for k in cat_map.keys()]
    return []


def order_labels_by_template(
    labels: list[str],
    template: dict,
    category: str | None = None,
    entity_class: str | None = None,
) -> list[str]:
    """Order *labels* by template file order, appending unlisted tail sorted.

    Labels present in the template section (per
    :func:`template_label_order`) come first in **file order**; labels not
    in the section are appended **sorted alphabetically** (the historical
    fallback for the tail).

    Matching mirrors :func:`resolve_label_color`:

    * ``entity_class`` lookups are **case-insensitive**.
    * ``category`` lookups are **exact**, except composite categories (see
      ``_COMPOSITE_CATEGORIES``) whose labels are matched on
      ``<type>_<item>`` then ``<type>`` after splitting.

    The returned list is a permutation of *labels* (no labels are dropped
    or duplicated even if a file key has no matching label).
    """
    file_order = template_label_order(template, category=category, entity_class=entity_class)
    if not file_order:
        return sorted(labels, key=str)

    remaining = list(labels)
    ordered: list[str] = []

    if entity_class:
        # Case-insensitive: bucket remaining labels by lowercase.
        by_lc: dict[str, list[str]] = {}
        for lbl in remaining:
            by_lc.setdefault(str(lbl).lower(), []).append(lbl)
        consumed: set[int] = set()
        for key in file_order:
            bucket = by_lc.get(str(key).lower())
            if not bucket:
                continue
            for lbl in bucket:
                lid = id(lbl)
                if lid in consumed:
                    continue
                consumed.add(lid)
                ordered.append(lbl)
        tail = [lbl for lbl in remaining if id(lbl) not in consumed]
        return ordered + sorted(tail, key=str)

    # Category path (exact + composite split).
    composite = bool(category) and category in _COMPOSITE_CATEGORIES

    def _match_key(label) -> str | None:
        """Return the file_order key this label matches, or None."""
        if composite:
            parts = _split_composite_label(label)
            if parts is not None:
                type_key, item_key = parts
                joined = f"{type_key}_{item_key}"
                if joined in file_order_set:
                    return joined
                if type_key in file_order_set:
                    return type_key
        if label in file_order_set:
            return label
        return None

    file_order_set = set(file_order)
    # Group labels by the file key they match (in file order).
    matched: dict[str, list[str]] = {}
    consumed_idx: set[int] = set()
    for i, lbl in enumerate(remaining):
        key = _match_key(lbl)
        if key is not None:
            matched.setdefault(key, []).append(lbl)
            consumed_idx.add(i)
    for key in file_order:
        for lbl in matched.get(key, []):
            ordered.append(lbl)
    tail = [lbl for i, lbl in enumerate(remaining) if i not in consumed_idx]
    return ordered + sorted(tail, key=str)


# ---------------------------------------------------------------------------
# Dispatch color / order resolution
# ---------------------------------------------------------------------------

# Entity classes the dispatch resolver searches, in precedence order.  A
# dispatch column that is an entity name (a processGroup aggregate, a unit,
# a connection, …) may live in any of these ``entities`` subsections; the
# first one with a matching (case-insensitive) key wins.
_DISPATCH_ENTITY_CLASSES: tuple[str, ...] = (
    "group",
    "unit",
    "connection",
    "node",
)


def _entities_section(template: dict) -> dict:
    """Return the ``entities`` (or legacy ``entity_class``) mapping or {}."""
    if not isinstance(template, dict):
        return {}
    section = template.get("entities")
    if not isinstance(section, dict):
        section = template.get("entity_class")
    return section if isinstance(section, dict) else {}


def _dispatch_special_section(template: dict) -> dict:
    """Return the ``categories.dispatch`` mapping or {}."""
    if not isinstance(template, dict):
        return {}
    cats = template.get("categories")
    if not isinstance(cats, dict):
        cats = template.get("category")
    if not isinstance(cats, dict):
        return {}
    block = cats.get("dispatch")
    return block if isinstance(block, dict) else {}


def _extract_dispatch_entity_name(column: str) -> str | None:
    """Extract the entity name from a dispatch column name.

    Dispatch column names come in several shapes (see
    ``scenario_comparison/dispatch_data.py``):

    * bare entity name — a processGroup aggregate, e.g. ``"coal"`` ;
    * ``"(process, node)"`` or ``"(connection)"`` repr composites for
      not-in-aggregate flows — the *process*/*connection* is the entity ;
    * node-level columns ``"<unit>_out"`` / ``"<unit>_in"`` /
      ``"<conn>_left"`` / ``"<conn>_right"`` from the single-node path ;
    * any of the above with a trailing ``_pos`` / ``_neg`` added by the
      mixed-sign split in ``_order_dispatch_columns``.

    Returns the bare entity name, or ``None`` if *column* is not a string.
    """
    if not isinstance(column, str):
        return None
    name = column

    # Strip the mixed-sign split suffix first (it is appended last).
    if name.endswith("_pos"):
        name = name[:-4]
    elif name.endswith("_neg"):
        name = name[:-4]

    # ``(process, node)`` / ``(connection)`` repr composites: take the
    # first element (the process / connection entity).
    s = name.strip()
    if s.startswith("(") and s.endswith(")"):
        inner = s[1:-1]
        # First comma-separated field, stripped of surrounding quotes.
        first = inner.split(",", 1)[0].strip().strip("'\"")
        return first or None

    # Node-level flow suffixes from the single-node dispatch path.
    for suffix in ("_left_right", "_left", "_right", "_out", "_in"):
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            if base:
                return base

    return name or None


def _lookup_entity_color(entities: dict, name: str, negative: bool):
    """Look up *name* across the dispatch entity classes (case-insensitive).

    Returns the raw color value (positive side, or ``neg_color`` when
    *negative* and present) or ``None`` if no class lists *name*.
    """
    name_lc = str(name).lower()
    for cls in _DISPATCH_ENTITY_CLASSES:
        class_map = entities.get(cls)
        if not isinstance(class_map, dict):
            continue
        for key, val in class_map.items():
            if str(key).lower() == name_lc:
                color_val, neg_val = _resolve_entity_value(val)
                if negative and neg_val is not None:
                    return neg_val
                return color_val
    return None


def template_entity_names(template: dict) -> list[str]:
    """Return all entity names listed in the template, in file order.

    Concatenates the keys of every dispatch entity class
    (``group`` / ``unit`` / ``connection`` / ``node``) in file order.  This
    is the data-independent stacking order seed for the dispatch resolver:
    bare-entity dispatch columns match these names, so passing them to
    :func:`resolve_dispatch_colors_and_order` yields a stable
    ``config_order`` without first enumerating the data columns.
    """
    names: list[str] = []
    for cls in _DISPATCH_ENTITY_CLASSES:
        names.extend(template_label_order(template, entity_class=cls))
    return names


def resolve_dispatch_colors_and_order(
    template: dict,
    columns,
) -> tuple[dict[str, str], list[str]]:
    """Build dispatch ``colors`` + ``config_order`` from a plot-settings template.

    *columns* is the set of dispatch column names that may appear (the
    union across scenarios, base names — i.e. before the mixed-sign
    ``_pos`` / ``_neg`` split).  The result reproduces the shape that the
    legacy ``config['positive'|'negative']`` parsing produced:

    * ``colors`` maps each column name to its color value (a ``#RRGGBB``
      hex string, an ``[r, g, b]`` list, or a matplotlib color name).  A
      mixed-sign entity whose ``entities`` entry has ``neg_color`` also gets
      a ``"<column>_neg"`` entry carrying that distinct negative-side color;
      the ``"<column>_pos"`` part inherits the base color via
      ``get_color_for_column``.
    * ``config_order`` lists the **entity** column base names in matplotlib
      stacking order (bottom-to-top), following ``entities`` file order.  As
      in the legacy path the list is built top→bottom then reversed for
      matplotlib.  Special tokens are deliberately **omitted** from
      ``config_order`` so that ``_order_dispatch_columns`` keeps placing them
      at their fixed ``POSITIVE_SPECIAL`` / ``NEGATIVE_SPECIAL`` positions
      (its ``else`` branch is byte-for-byte the historical behavior when no
      entity ordering is supplied).  Special-token *colors* still flow
      through ``colors``.

    Resolution per column:

    1. **Special token** (exact key in ``categories.dispatch``) — e.g.
       ``LossOfLoad``, ``Charge``, ``internal_losses``.  Contributes a color
       only; ordering stays pipeline-fixed.
    2. **Entity** — extract the entity name (bare / composite /
       node-level, see :func:`_extract_dispatch_entity_name`) and look it
       up case-insensitively across ``entities`` ``group`` / ``unit`` /
       ``connection`` / ``node``.  Contributes both a color and a
       ``config_order`` position.
    3. **Fallback** — left out of ``colors`` and ``config_order`` so the
       downstream ``_auto_assign_node_colors_with_existing`` palette (seeded
       from the historical ``DEFAULT_SPECIAL_COLORS``) assigns it and
       std-dev ordering places it, exactly as before.

    For a project whose template has no dispatch entities (the common
    case — the bundled default ships ``entities`` empty), only the special
    tokens resolve (to the ``categories.dispatch`` values, which equal the
    old ``DEFAULT_SPECIAL_COLORS``); ``config_order`` comes back empty, so
    ``_order_dispatch_columns`` takes its ``else`` branch and every column —
    specials included — is ordered exactly as before.
    """
    special = _dispatch_special_section(template)
    entities = _entities_section(template)
    cols = [str(c) for c in (columns or [])]

    colors: dict[str, str] = {}

    # Entity columns in template file order, then the unlisted tail.  We
    # reuse the case-insensitive ordering machinery so file order is the
    # single source of truth.  Build the union of entity-class file orders
    # (group, unit, connection, node) as the order key.
    entity_file_order: list[str] = []
    for cls in _DISPATCH_ENTITY_CLASSES:
        entity_file_order.extend(template_label_order(template, entity_class=cls))
    entity_order_index = {
        str(k).lower(): i for i, k in enumerate(entity_file_order)
    }

    entity_cols: list[str] = []
    for col in cols:
        if col in special:
            color = special.get(col)
            if color is not None:
                colors[col] = color
            continue
        name = _extract_dispatch_entity_name(col)
        if name is None:
            continue
        pos_color = _lookup_entity_color(entities, name, negative=False)
        if pos_color is None:
            continue
        neg_color = _lookup_entity_color(entities, name, negative=True)
        is_neg_part = isinstance(col, str) and col.endswith("_neg")
        # The column may itself already be the negative-side split part
        # (``<col>_neg``) — color it with ``neg_color``; otherwise it is the
        # positive / whole column → ``color``.  ``get_color_for_column`` does
        # an exact-key lookup, so we register the precise rendered name.
        if is_neg_part and neg_color is not None and neg_color != pos_color:
            colors[col] = neg_color
        else:
            colors[col] = pos_color
            # A whole (not-yet-split) mixed column is split downstream into
            # ``<col>_pos`` / ``<col>_neg``; register the ``<col>_neg`` key so
            # its negative part picks up a distinct ``neg_color`` when set.
            if (
                not is_neg_part
                and not col.endswith("_pos")
                and neg_color is not None
                and neg_color != pos_color
            ):
                colors[f"{col}_neg"] = neg_color
        entity_cols.append(col)

    # --- Order (top → bottom; reversed at the end for matplotlib) ---
    # Only entity columns drive config_order, in template file order.
    # Special tokens are left out so their fixed pos/neg placement (and the
    # historical std-dev ordering of unlisted columns) is preserved.
    ordered_entities = sorted(
        entity_cols,
        key=lambda c: entity_order_index.get(
            str(_extract_dispatch_entity_name(c)).lower(), len(entity_order_index)
        ),
    )

    config_order = list(ordered_entities)
    config_order.reverse()

    return colors, config_order


def _clear_cache() -> None:
    """Clear the module-level cache (test hook)."""
    _TEMPLATE_CACHE.clear()


__all__ = [
    "load_color_template",
    "resolve_plot_settings_path",
    "resolve_label_color",
    "template_label_order",
    "order_labels_by_template",
    "resolve_dispatch_colors_and_order",
    "template_entity_names",
    "_clear_cache",
]
