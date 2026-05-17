"""Shared Provider-aware I/O helpers for the writer-port modules.

These helpers wrap a :class:`FlexDataProvider` so writer modules can read
their upstream artefacts uniformly: ask the Provider first, fall back to
the on-disk CSV when the Provider doesn't carry the frame.  Returns
``None`` only when neither the Provider nor disk has the file.
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
    """Open a file-like handle for *name* via the Provider or disk.

    Returns ``None`` when neither has the file.
    """
    if provider is not None and provider.has(name):
        df = provider.get(name)
        buf = io.StringIO()
        df.write_csv(buf)
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
    """Return a frame sliced to *columns* from the Provider, or ``None``.

    Resolution: if *provider* has *name*, slice its frame to the first
    ``len(columns)`` columns and rename them by position to *columns*.
    Otherwise return ``None`` so the caller falls back to its own
    on-disk ``polars.read_csv(path)`` branch.
    """
    if provider is not None and provider.has(name):
        df = provider.get(name)
        # Guard against a width mismatch between the Provider's stored
        # frame and the caller's expected schema.  When widths diverge,
        # treat as a miss so the caller's disk fallback wins.
        if df.width >= len(columns):
            keep = df.columns[: len(columns)]
            out = df.select(keep)
            out.columns = columns
            return out
    return None


__all__ = ["_provider_key", "_provider_open", "_provider_lookup_positional"]
