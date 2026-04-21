"""Loader and resolver for the global plot color template.

The color template maps legend labels to explicit RGB colors.  It has
two sections:

* ``category`` — keyed by a category name declared by a plot entry via
  ``color_category``.  Label lookup is *exact* (these are parameter or
  result names whose casing is controlled by the pipeline).
* ``entity_class`` — keyed by an entity-class name declared via
  ``color_entity_class``.  Label lookup is *case-insensitive* (entity
  names come from the data and can be mixed-case).

Any label not found falls back to the tab10/tab20 palette used by
:func:`build_shared_color_map`.

The YAML file lives at ``templates/default_colors.yaml`` by default and
is cached at the module level keyed by ``(path, mtime)`` so repeated
plot builds within the same process don't re-read it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


# (resolved_path, mtime_ns) -> parsed dict
_TEMPLATE_CACHE: dict[tuple[str, int], dict] = {}


def _repo_root() -> Path:
    """Return the FlexTool repository root.

    ``color_template.py`` lives at ``flextool/plot_outputs/``; go two
    levels up to reach the repo root alongside ``templates/``.
    """
    return Path(__file__).resolve().parent.parent.parent


def _default_path() -> Path:
    return _repo_root() / "templates" / "default_colors.yaml"


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


def resolve_label_color(
    label: str,
    template: dict,
    category: str | None = None,
    entity_class: str | None = None,
) -> tuple[float, float, float] | None:
    """Resolve *label* to a color via the color *template*.

    Lookup precedence:
    * If *entity_class* is given, check ``template['entity_class'][entity_class]``
      with a case-insensitive key match.
    * Else if *category* is given, check ``template['category'][category]``
      with an exact key match.
    * Otherwise return ``None`` and let the caller fall back to the palette.

    Returns a normalized ``(r, g, b)`` float tuple in ``0..1`` or
    ``None`` if the template has no entry (or the entry is malformed).
    Malformed values are logged but do not raise.
    """
    if not isinstance(template, dict) or label is None:
        return None

    raw = None
    if entity_class:
        section = template.get("entity_class")
        if isinstance(section, dict):
            class_map = section.get(entity_class)
            if isinstance(class_map, dict):
                # Case-insensitive lookup.  Build a lowercase view on the
                # fly — these dicts are small (tens of entries at most).
                label_lc = str(label).lower()
                for key, val in class_map.items():
                    if str(key).lower() == label_lc:
                        raw = val
                        break
    elif category:
        section = template.get("category")
        if isinstance(section, dict):
            cat_map = section.get(category)
            if isinstance(cat_map, dict):
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


def _clear_cache() -> None:
    """Clear the module-level cache (test hook)."""
    _TEMPLATE_CACHE.clear()


__all__ = [
    "load_color_template",
    "resolve_label_color",
    "_clear_cache",
]
