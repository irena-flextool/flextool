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

import polars as pl


def _emit(provider, key: str, df: pl.DataFrame) -> None:
    """Register *df* in *provider* under *key* (canonical form).

    *key* must be of the form 'parent/basename' (e.g. 'solve_data/foo.csv').
    The Provider's bidirectional lookup (see
    :class:`FlexDataProvider.get`) resolves bare-basename consumer
    queries to the qualified key, so a single registration is
    sufficient for every consumer.
    """
    provider.put(key, df)


def _provider_key(path: "Path | str") -> str:
    """Build the canonical Provider key for *path*.

    Returns ``"<parent>/<stem>"`` when *path* has a parent dir, else the
    bare stem.  Matches the parent-qualified dual-key semantics that
    :func:`_emit` registers each frame under.
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


def workdir_provider_for_paths(
    workdir: "Path | str",
    paths: "list[Path | str]",
):
    """Build a :class:`FlexDataProvider` carrying the listed *paths*
    from *workdir*.

    Used by cascade entry points that accept a bare ``workdir`` and need
    to materialise Provider keys for the per-solve scaffolding the
    cascade reads via :func:`_provider_open` / :func:`_provider_read`.
    Each entry in *paths* is interpreted relative to *workdir* (or
    treated as absolute if already absolute).  Missing files are skipped
    silently — the cascade modules tolerate missing keys by their own
    empty-frame contract.

    This helper lives in ``_emit_provider_io.py`` (which is on the
    meta-test's ``PROVIDER_IMPL_ALLOWLIST``) so the cascade modules
    themselves remain free of disk reads.  The frames are stored under
    the parent-qualified Provider key (``"<parent>/<stem>"``) computed
    by :func:`_provider_key`.
    """
    from ._flex_data_provider import FlexDataProvider

    workdir = Path(workdir)
    provider = FlexDataProvider()
    for entry in paths:
        p = Path(entry)
        if not p.is_absolute():
            p = workdir / p
        if not p.exists():
            continue
        try:
            df = pl.read_csv(p)
        except Exception:
            continue
        provider.put(_provider_key(p), df)
    return provider


__all__ = [
    "_emit",
    "_provider_key",
    "_provider_open",
    "_provider_lookup_positional",
    "workdir_provider_for_paths",
]
