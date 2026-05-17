"""Shared Provider-aware I/O helpers for the writer-port modules.

Step 1-g introduced this small helper module to deduplicate the
``_provider_open`` / ``_provider_key`` pattern that originated in
:mod:`._writer_arc_unions` (Step 1-d) and now needs to apply to every
other writer module that previously used :func:`_seed_open` from
:mod:`._input_source`.

The helpers mirror those in :mod:`._writer_arc_unions` exactly so the
migration is a mechanical one-line swap of ``_seed_open(path)`` for
``_provider_open(provider, _provider_key(path), path)``.  After Step 2
deletes the seed funnel, the transitional middle tier
(:func:`_seed_lookup`) goes away and this module collapses to a thin
Provider-first + disk-fallback shim — at which point callers can decide
whether to keep the helpers here or inline them per-module.

Caller convention (identical to :func:`_seed_open`):

* When the Provider carries the named artefact, returns a ``StringIO``
  wrapping the serialised frame so the caller's ``csv.reader`` chain is
  unchanged.
* Otherwise falls back to disk; ``None`` is returned only when neither
  the Provider nor the disk has the file (matching the missing-file
  sentinel ``_seed_open`` returns).
"""
from __future__ import annotations

import io
from pathlib import Path


def _provider_key(path: "Path | str") -> str:
    """Build the canonical Provider key for *path*.

    Returns ``"<parent>/<stem>"`` when *path* has a parent dir, else the
    bare stem.  Mirrors the parent-qualified dual-key semantics used by
    the writers when populating the Provider via ``capture_frames``.
    """
    p = Path(path)
    parent = p.parent.name
    stem = p.stem
    if parent:
        return f"{parent}/{stem}"
    return stem


def _provider_open(provider: "object | None", name: str,
                   path: "Path | str"):
    """Open a file-like handle for *name* via the Provider, the legacy
    in-memory seed, or disk.  Returns ``None`` when none of the three
    has the file.

    Mirrors :func:`._writer_arc_unions._provider_open` exactly, with the
    same transitional middle tier (:func:`_seed_lookup`) so callsites
    that haven't yet been plumbed with an explicit ``provider`` kwarg
    still resolve their frames from the in-memory carrier during the
    Step 1-g migration window.  Step 2 drops the seed-lookup branch
    once every caller threads the Provider.
    """
    if provider is not None and provider.has(name):
        df = provider.get(name)
        buf = io.StringIO()
        df.write_csv(buf)
        buf.seek(0)
        return buf
    # Transitional — fall back to the legacy seed lookup so unplumbed
    # callsites continue to find in-memory frames during the migration.
    from flextool.engine_polars._input_source import _seed_lookup
    seeded = _seed_lookup(path)
    if seeded is not None:
        buf = io.StringIO()
        seeded.write_csv(buf)
        buf.seek(0)
        return buf
    p = Path(path)
    if p.exists():
        return p.open()
    return None


def _provider_lookup_positional(
    provider: "object | None",
    name: str,
    path: "Path | str",
    columns: list[str],
):
    """Provider-first counterpart to
    :func:`._input_source._seed_lookup_positional`.

    Resolution order:

    1. If *provider* has *name*, slice its frame to the first
       ``len(columns)`` columns and rename them by position to
       *columns*.
    2. Otherwise fall through to the legacy seed lookup so unplumbed
       sites still resolve frames during the Step 1-g migration window.

    Returns ``None`` when neither the Provider nor the seed carries the
    frame; the caller then falls back to its on-disk
    ``polars.read_csv(path)`` branch — identical contract to
    ``_seed_lookup_positional``.
    """
    if provider is not None and provider.has(name):
        df = provider.get(name)
        keep = df.columns[: len(columns)]
        out = df.select(keep)
        out.columns = columns
        return out
    from flextool.engine_polars._input_source import _seed_lookup_positional
    return _seed_lookup_positional(path, columns)


__all__ = ["_provider_key", "_provider_open", "_provider_lookup_positional"]
