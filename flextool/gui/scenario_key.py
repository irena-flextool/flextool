"""Helpers for scenario identity across source and display boundaries.

Two scenarios with the same name from different input sources must not
collide on disk. **New folders are always written with the source-number
suffix** (``output_parquet/base_3/`` rather than ``output_parquet/base/``)
so the folder name itself records the source identity — losing the
project settings.yaml or moving the project elsewhere does not strand
the data.

Encodings
---------

* **Compound key** (settings) — ``"<source_number>|<scenario_name>"``.
  Used in project settings (checked_executed_scenarios, comp_*_scenarios,
  etc.). ``|`` cannot appear in a scenario name.

* **Subdirectory name** (on disk) — ``<scenario_name>_<source_number>``
  (e.g. ``base_2``). Always. There is no bare-name shortcut for newly-
  written folders.

Legacy bare folders
-------------------

Pre-existing projects may carry folders without the suffix (``base/``)
from the previous "bare-owner" convention. Those are still recognised
on the READ side via :attr:`ProjectSettings.bare_output_owners`:
:func:`resolve_source_number` consults that map first so the old bare
folder still maps to its original source. New writes never reuse a
bare folder — if scenario "base" runs again it writes to
``base_<source_number>/`` regardless of whether ``base/`` exists.
Users with legacy projects should rename or delete the old bare
folders manually; the GUI flags them as orphans (foreign source
number) so they are visible.

If the user happens to have a scenario name that itself ends in
``_<digits>`` (e.g. literally named ``foo_3``), parsing is ambiguous
when there is no owners record — :func:`parse_subdir` will read
``foo_3`` as "scenario foo from source 3". The owners map disambiguates
for legacy data; for new writes the suffix is appended, producing
``foo_3_<source_number>`` which parses unambiguously.
"""

from __future__ import annotations

from pathlib import Path

LEGACY_SOURCE_NUMBER = 0


def format_subdir(source_number: int, scenario_name: str) -> str:
    """Return the explicit suffixed subdirectory name.

    This is the "collision" form; prefer :func:`resolve_subdir_for_read`
    or :func:`choose_output_subdir_for_write` which honour bare-name
    ownership.
    """
    return f"{scenario_name}_{source_number}"


def parse_subdir(subdir: str) -> tuple[int, str]:
    """Split ``<name>_<N>`` into ``(N, name)``.

    A subdir without a trailing numeric suffix returns
    ``(LEGACY_SOURCE_NUMBER, subdir)`` — callers are expected to consult
    the ownership map to recover the correct source number for such bare
    names.
    """
    base, sep, suffix = subdir.rpartition("_")
    if sep and suffix.isdigit():
        return int(suffix), base
    return LEGACY_SOURCE_NUMBER, subdir


def format_key(source_number: int, scenario_name: str) -> str:
    """Return the settings-level compound key."""
    return f"{source_number}|{scenario_name}"


def parse_key(key: str) -> tuple[int, str]:
    """Split a compound key into ``(source_number, scenario_name)``."""
    prefix, sep, rest = key.partition("|")
    if sep and prefix.isdigit():
        return int(prefix), rest
    return LEGACY_SOURCE_NUMBER, key


# ---------------------------------------------------------------------------
# Ownership-aware resolution
# ---------------------------------------------------------------------------

def resolve_source_number(
    subdir: str,
    bare_output_owners: dict[str, int],
) -> tuple[int, str]:
    """Return ``(source_number, scenario_name)`` for an on-disk subdir.

    Uses ``bare_output_owners`` (a snapshot of the ownership map) to
    recover the source number for bare names that aren't self-describing.
    Falls back to suffix parsing when the subdir isn't a recorded bare
    name.
    """
    # An explicit ownership record wins — even if the subdir happens to
    # end in ``_<N>``, if that full string is recorded as a bare owner
    # (e.g. a scenario legitimately named "foo_1"), trust the record.
    if subdir in bare_output_owners:
        return bare_output_owners[subdir], subdir
    src, name = parse_subdir(subdir)
    if src != LEGACY_SOURCE_NUMBER:
        return src, name
    # No suffix, no ownership record → legacy bare.
    return LEGACY_SOURCE_NUMBER, subdir


def resolve_subdir_for_read(
    bare_output_owners: dict[str, int],
    source_number: int,
    scenario_name: str,
) -> str:
    """Return the subdir that holds this scenario's outputs (no mutation).

    Use this when the folder is expected to already exist: reading plots,
    checking output presence, deleting, etc.
    """
    if bare_output_owners.get(scenario_name) == source_number:
        return scenario_name
    return format_subdir(source_number, scenario_name)


def choose_output_subdir_for_write(
    project_path: Path,
    bare_output_owners: dict[str, int],
    source_number: int,
    scenario_name: str,
) -> str:
    """Pick the subdir to write outputs into.

    Always returns ``<scenario_name>_<source_number>``. The folder name
    is self-describing — losing or corrupting ``bare_output_owners`` no
    longer makes the folder ambiguous about which source it came from.

    The ``project_path`` and ``bare_output_owners`` parameters are kept
    in the signature for API stability with callers but are unused; the
    only legitimate purpose of the bare-owner map now is to recognise
    legacy bare folders on the READ side (see :func:`resolve_source_number`).
    """
    del project_path, bare_output_owners  # unused — kept for API stability
    return format_subdir(source_number, scenario_name)


def release_bare_owner(
    bare_output_owners: dict[str, int],
    source_number: int,
    scenario_name: str,
) -> bool:
    """Drop the ownership record if *source_number* owns ``scenario_name``.

    Returns True if a record was removed.
    """
    if bare_output_owners.get(scenario_name) == source_number:
        del bare_output_owners[scenario_name]
        return True
    return False
