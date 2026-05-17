"""Shared Provider-aware I/O helpers for the writer-port modules.

These helpers let writer modules read upstream artefacts uniformly: the
live :class:`FlexDataProvider` is consulted first, and a disk-fallback
serves the byte-parity test gate (``test_writer_port_phase1``) which
exercises writer modules outside any cascade Provider context.

In a real cascade run the Provider always carries the frame the writer
needs (Step 2 invariant); the disk arm is therefore unreachable
in-cascade and is preserved exclusively for the off-cascade test
harness that seeds inputs to disk and runs writers without a Provider.
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

    In-cascade the Provider always supplies the frame; the disk arm is
    reserved for the off-cascade byte-parity test harness that seeds
    inputs to disk and invokes writers without a Provider.  Returns
    ``None`` when neither source has the file.
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
    on-disk ``polars.read_csv(path)`` branch.  The disk arm is
    off-cascade only (writer-port byte-parity tests); in-cascade the
    Provider always carries the frame.
    """
    if provider is not None and provider.has(name):
        df = provider.get(name)
        if df.width >= len(columns):
            keep = df.columns[: len(columns)]
            out = df.select(keep)
            out.columns = columns
            return out
    return None


__all__ = ["_provider_key", "_provider_open", "_provider_lookup_positional"]
