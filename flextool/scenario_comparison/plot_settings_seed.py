"""Append-only seeding of discovered colors into ``plot_settings.yaml``.

Dispatch runs discover entity (unit / connection / group) and scenario names
from the data and want stable default colors for any name not already in the
project's ``plot_settings.yaml``.  We must *add* those without disturbing the
user's file: every existing line — comments, commented examples, existing
entries, hand edits, blank lines, formatting — is preserved byte-for-byte.

This rules out a YAML round-trip (``yaml.safe_dump`` rewrites the whole file
and drops comments; ``ruamel.yaml`` would add a dependency, which is
forbidden).  Instead we:

1. Parse the file with :mod:`pyyaml` (read-only) to learn which entity /
   scenario names already exist *as active keys*.
2. Scan the raw text to also learn which names appear as *commented* example
   keys, so we never duplicate a commented placeholder.
3. Compute the names that are genuinely missing.
4. If any are missing, splice new ``    name: "#RRGGBB"`` lines into the
   relevant subsection at the subsection's own indentation, inserting after
   the subsection's existing entries (or, for an empty subsection, right
   after its header / trailing comment block), leaving every untouched line
   exactly as it was.

A run that discovers nothing new makes no change (byte-identical), so the
seeding is idempotent.
"""

from __future__ import annotations

from pathlib import Path

import yaml


# --- Indentation conventions matching schemas/default_plot_settings.yaml ----
# Top-level sections (``scenarios:``, ``categories:``, ``entities:``) sit at
# column 0; entity-class subsections (``group:`` …) are indented two spaces
# under ``entities:``; leaf entries are indented two further spaces.
_TOP_INDENT = 0
_SUB_INDENT = 2
_ENTRY_UNDER_SUB_INDENT = 4
_ENTRY_UNDER_TOP_INDENT = 2

_ENTITY_CLASSES = ("group", "unit", "connection", "node")


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_comment_marker(line: str) -> str:
    """Return *line* with a leading ``#`` (and one optional space) removed."""
    body = line.lstrip(" ")
    if body.startswith("#"):
        body = body[1:]
        if body.startswith(" "):
            body = body[1:]
    return body


def _key_of(text: str) -> str | None:
    """Extract a bare mapping key from a ``key: value`` text fragment.

    Returns the unquoted key, or ``None`` if *text* is not a ``key:`` line.
    Used both for active entries and (after stripping the ``#``) for commented
    example entries, so a commented placeholder is not duplicated.
    """
    stripped = text.strip()
    if not stripped or ":" not in stripped:
        return None
    key_part = stripped.split(":", 1)[0].strip()
    if not key_part:
        return None
    # Unwrap quotes.
    if len(key_part) >= 2 and key_part[0] in "'\"" and key_part[-1] == key_part[0]:
        key_part = key_part[1:-1]
    return key_part or None


def _existing_names_in_block(lines: list[str], start: int, end: int) -> set[str]:
    """Collect active + commented example keys within ``lines[start:end]``.

    Case-folded so the lookup is case-insensitive (matching the resolver).
    """
    names: set[str] = set()
    for raw in lines[start:end]:
        body = raw.strip()
        if not body:
            continue
        if body.startswith("#"):
            key = _key_of(_strip_comment_marker(raw))
        else:
            key = _key_of(raw)
        if key is not None:
            names.add(key.lower())
    return names


def _find_top_section(lines: list[str], name: str) -> tuple[int, int] | None:
    """Return ``(header_idx, end_idx)`` for top-level section ``name:``.

    ``end_idx`` is the index of the first line that belongs to the *next*
    top-level section (or ``len(lines)`` at end of file).  Returns ``None``
    when the section header is absent.
    """
    header = None
    for i, line in enumerate(lines):
        if _leading_spaces(line) == _TOP_INDENT and not line.lstrip().startswith("#"):
            key = _key_of(line)
            if key == name:
                header = i
                break
    if header is None:
        return None
    end = len(lines)
    for j in range(header + 1, len(lines)):
        line = lines[j]
        if (
            line.strip()
            and _leading_spaces(line) == _TOP_INDENT
            and not line.lstrip().startswith("#")
        ):
            end = j
            break
    return header, end


def _find_subsection(
    lines: list[str], sec_start: int, sec_end: int, sub_name: str
) -> tuple[int, int] | None:
    """Return ``(header_idx, end_idx)`` for a ``  <sub_name>:`` subsection.

    Searches within ``lines[sec_start+1:sec_end]`` for a subsection header at
    ``_SUB_INDENT``.  ``end_idx`` is the first line that starts the next
    subsection (indent ``<= _SUB_INDENT`` active line) within the section, or
    ``sec_end``.
    """
    header = None
    for i in range(sec_start + 1, sec_end):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _leading_spaces(line) == _SUB_INDENT and _key_of(line) == sub_name:
            header = i
            break
    if header is None:
        return None
    end = sec_end
    for j in range(header + 1, sec_end):
        line = lines[j]
        if (
            line.strip()
            and not line.lstrip().startswith("#")
            and _leading_spaces(line) <= _SUB_INDENT
        ):
            end = j
            break
    return header, end


def _block_end_insert_point(lines: list[str], start: int, end: int) -> int:
    """Index after the last *content* (comment or entry) line in the block.

    Auto-added entries are appended at the END of a (sub)section, after both
    the instruction comments and any existing entries, so the comments stay
    ABOVE the seeded entities.  Trailing blank lines (the separator before the
    next section) are skipped so new entries sit directly under the last
    content line.  Returns *start* when the block is entirely blank.
    """
    insert_at = start
    for i in range(start, end):
        if lines[i].strip():  # any non-blank line (comment or entry)
            insert_at = i + 1
    return insert_at


def _format_entry(indent: int, name: str, color: str) -> str:
    """Render a single ``<indent>name: "#RRGGBB"`` entry line (no newline)."""
    key = name
    # Quote keys that YAML would otherwise misparse (rare for entity names,
    # but keep it safe for names with special leading chars or spaces).
    if name == "" or name[0] in "!&*?|>%@`\"'#,[]{}:" or name != name.strip():
        key = '"' + name.replace('"', '\\"') + '"'
    return f"{' ' * indent}{key}: \"{color}\""


def _line_sep(text: str) -> str:
    """Detect the line terminator used by *text* (default ``\\n``)."""
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def seed_colors_into_plot_settings(
    path: Path,
    entity_colors: dict[str, dict[str, str]],
    scenario_colors: dict[str, str],
) -> bool:
    """Additively splice missing colors into ``plot_settings.yaml`` at *path*.

    *entity_colors* maps entity class (``group`` / ``unit`` / ``connection``)
    to an ordered ``{name -> '#RRGGBB'}`` mapping; *scenario_colors* is the
    ``{name -> '#RRGGBB'}`` mapping for the ``scenarios`` section.  Only names
    that are not already present (active *or* commented, case-insensitively)
    in their target (sub)section are inserted; existing content is preserved
    byte-for-byte.

    Returns ``True`` when the file was modified, ``False`` when nothing was
    missing (the no-op / idempotent case).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # Parse to a dict purely to validate structure / surface load errors; the
    # raw-text scan below is what drives de-duplication (it also sees
    # commented examples that pyyaml cannot).
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:  # pragma: no cover - corrupt file
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if parsed is not None and not isinstance(parsed, dict):
        raise ValueError(f"{path} is not a YAML mapping")

    sep = _line_sep(text)
    # ``splitlines(keepends=False)`` for editing; we rejoin with *sep* and
    # restore a trailing newline if the original had one.
    had_trailing_nl = text.endswith(("\n", "\r"))
    lines = text.splitlines()

    modified = False

    # --- Entities ------------------------------------------------------------
    # Locate (or create) the ``entities:`` section once.
    ent_loc = _find_top_section(lines, "entities")
    if any(entity_colors.get(cls) for cls in _ENTITY_CLASSES):
        if ent_loc is None:
            # No entities section at all — create it at end of file.
            if lines and lines[-1].strip():
                lines.append("")
            lines.append("entities:")
            ent_loc = (len(lines) - 1, len(lines))
            modified = True

    for cls in _ENTITY_CLASSES:
        wanted = entity_colors.get(cls) or {}
        if not wanted:
            continue
        # Re-resolve section bounds each pass (earlier insertions shift them).
        ent_loc = _find_top_section(lines, "entities")
        if ent_loc is None:  # pragma: no cover - created above
            continue
        sec_start, sec_end = ent_loc
        sub = _find_subsection(lines, sec_start, sec_end, cls)
        if sub is None:
            # Subsection missing — create a minimal header at the end of the
            # entities section, then treat it as an empty block.
            insert_pos = sec_end
            lines.insert(insert_pos, f"{' ' * _SUB_INDENT}{cls}:")
            modified = True
            sub = (insert_pos, insert_pos + 1)
            sec_end += 1

        sub_start, sub_end = sub
        existing = _existing_names_in_block(lines, sub_start + 1, sub_end)
        missing = [
            (n, c) for n, c in wanted.items() if n.lower() not in existing
        ]
        if not missing:
            continue
        insert_at = _block_end_insert_point(lines, sub_start + 1, sub_end)
        new_lines = [
            _format_entry(_ENTRY_UNDER_SUB_INDENT, n, c) for n, c in missing
        ]
        lines[insert_at:insert_at] = new_lines
        modified = True

    # --- Scenarios -----------------------------------------------------------
    if scenario_colors:
        scen_loc = _find_top_section(lines, "scenarios")
        if scen_loc is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append("scenarios:")
            scen_loc = (len(lines) - 1, len(lines))
            modified = True
        sec_start, sec_end = scen_loc
        existing = _existing_names_in_block(lines, sec_start + 1, sec_end)
        missing = [
            (n, c) for n, c in scenario_colors.items() if n.lower() not in existing
        ]
        if missing:
            insert_at = _block_end_insert_point(lines, sec_start + 1, sec_end)
            new_lines = [
                _format_entry(_ENTRY_UNDER_TOP_INDENT, n, c) for n, c in missing
            ]
            lines[insert_at:insert_at] = new_lines
            modified = True

    if not modified:
        return False

    out = sep.join(lines)
    if had_trailing_nl:
        out += sep
    path.write_text(out, encoding="utf-8")
    return True
