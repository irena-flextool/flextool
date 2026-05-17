"""Shared Provider I/O helpers for the writer-port modules.

Post-Step-2.5 (items 17/19/20) the cascade has a single data pathway:
the :class:`FlexDataProvider` carries every frame a writer needs.
Disk reads are forbidden in cascade modules — the meta-test
``tests/engine_polars/test_meta_provider_invariants.py`` enforces it.

These helpers therefore consult the Provider only.  When the Provider
does not carry the requested key they return ``None`` (or the caller's
sentinel) — callers tolerate the miss by returning their own empty
result.  The disk-fallback arm that used to exist for the (deleted)
``test_writer_port_phase1`` byte-parity tests was removed in Step 2.5
Phase B.

The *path* argument is retained at every helper signature so call
sites can keep using the canonical ``input/<stem>.csv`` /
``solve_data/<stem>.csv`` paths to derive the Provider key via
:func:`_provider_key`.  No I/O against the path occurs here.
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
                   path: "Path | str"):  # noqa: ARG001 — path kept for API
    """Open an in-memory CSV handle for *name* from the Provider.

    Returns a ``StringIO`` containing the frame's CSV serialisation
    when the Provider carries *name*, else ``None``.  *path* is kept
    in the signature so call sites can pass the canonical workdir
    path alongside the key without restructuring — it is not read.

    Raises :class:`ValueError` when *provider* is missing: cascade
    callers must always thread a Provider, and a ``None`` here is a
    bug, not a fallback.
    """
    if provider is None:
        raise ValueError(
            "_provider_open requires a FlexDataProvider; the disk-fallback "
            "arm was removed in Step 2.5.  Plumb a provider through to "
            "the caller."
        )
    if not provider.has(name):
        return None
    df = provider.get(name)
    buf = io.StringIO()
    df.write_csv(buf)
    buf.seek(0)
    return buf


def _provider_lookup_positional(
    provider: "object | None",
    name: str,
    path: "Path | str",   # noqa: ARG001 — path kept for API
    columns: list[str],
):
    """Return a frame sliced to *columns* from the Provider, or ``None``.

    Resolution: if *provider* has *name*, slice its frame to the first
    ``len(columns)`` columns and rename them by position to *columns*.
    Otherwise return ``None`` so the caller falls back to its own
    empty-frame branch.  *path* is retained in the signature for
    symmetry with :func:`_provider_open`; no disk I/O happens here.

    Raises :class:`ValueError` when *provider* is missing.
    """
    if provider is None:
        raise ValueError(
            "_provider_lookup_positional requires a FlexDataProvider; the "
            "disk-fallback arm was removed in Step 2.5.  Plumb a provider "
            "through to the caller."
        )
    if not provider.has(name):
        return None
    df = provider.get(name)
    if df.width < len(columns):
        return None
    keep = df.columns[: len(columns)]
    out = df.select(keep)
    out.columns = columns
    return out


__all__ = ["_provider_key", "_provider_open", "_provider_lookup_positional"]
