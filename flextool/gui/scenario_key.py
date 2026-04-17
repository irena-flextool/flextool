"""Helpers for scenario identity across source and display boundaries.

Two scenarios with the same name from different input sources must not
collide on disk. To keep the common case clean (``output_parquet/base/``,
not ``output_parquet/base_1/``), we only apply a source suffix when
another input source has already claimed the bare name.

Encodings
---------

* **Compound key** (settings) — ``"<source_number>|<scenario_name>"``.
  Used in project settings (checked_executed_scenarios, comp_*_scenarios,
  etc.). ``|`` cannot appear in a scenario name.

* **Subdirectory name** (on disk) — either the bare scenario name
  (``base``) when this source owns it, or ``<scenario_name>_<source_number>``
  (``base_2``) when another source owns the bare name. Ownership is
  tracked in :attr:`ProjectSettings.bare_output_owners`.

Legacy / ambiguity
------------------

Pre-existing directories that happen to end in ``_<digits>`` will be
parsed as suffixed entries. If the user has scenario names ending in
``_<small integer>`` those will false-positive — but the ownership map
takes precedence when it records the full name, so this only matters
for folders that have no ownership record at all (legacy data).
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
    """Pick the subdir to write outputs into, claiming ownership when free.

    Rules:
      1. If an explicit suffixed folder already exists for *source_number*,
         reuse it (avoids orphaning data from prior runs that used the
         suffix form).
      2. If the bare name is already owned by this source, use bare.
      3. If the bare name is owned by another source, use the suffix form.
      4. If nobody owns the bare name yet, claim it for this source and
         use bare.

    ``bare_output_owners`` is mutated in place when ownership is claimed
    (case 4). The caller is responsible for persisting settings after.
    """
    parquet_root = project_path / "output_parquet"
    suffixed = parquet_root / format_subdir(source_number, scenario_name)
    if suffixed.is_dir():
        return suffixed.name

    owner = bare_output_owners.get(scenario_name)
    if owner == source_number:
        return scenario_name
    if owner is not None:
        return format_subdir(source_number, scenario_name)

    # Unclaimed — claim for this source.
    bare_output_owners[scenario_name] = source_number
    return scenario_name


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
