"""Process topology (``pss`` family) of a reloaded dump, read from the seeded provider.

``load_flextool`` builds the topology from the DB reader when it has one.  A dump reloaded without a DB has only
the preprocessed CSVs, so :func:`pss_from_dump` reads ``solve_data/process_source_sink{,_eff,_noEff}`` and
``solve_data/process_{source,sink}`` from the provider that :func:`seed_provider_from_dump` filled.  The loader
calls it through its private ``_dump_topology`` hook.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.input._axis_enums import cast_dim, rename_to_axis, schema_dtype


def _empty_pss() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "p": schema_dtype(None, "p"),
        "source": schema_dtype(None, "source"),
        "sink": schema_dtype(None, "sink"),
    })


def _read_pss(provider, key: str) -> pl.DataFrame:
    """``(p, source, sink)`` of ``key``, unique and sorted; empty when the key is absent or holds no processes."""
    if not provider.has(key):
        return _empty_pss()
    df = provider.get(key)
    if df.height == 0 or "process" not in df.columns:
        return _empty_pss()
    return (df.pipe(rename_to_axis, {"process": "p"})
              .with_columns(cast_dim(pl.col("source"), None, "source"),
                            cast_dim(pl.col("sink"), None, "sink"))
              .select("p", "source", "sink")
              .unique()
              .sort("p", "source", "sink"))


def _read_canonical_set(provider, key: str, side: str) -> pl.DataFrame | None:
    """``(p, side)`` of ``key`` (the canonical source or sink per process), or ``None`` when absent or empty."""
    if not provider.has(key):
        return None
    df = provider.get(key)
    if df.height == 0 or "process" not in df.columns or side not in df.columns:
        return None
    return (df.pipe(rename_to_axis, {"process": "p"})
              .with_columns(cast_dim(pl.col(side), None, side))
              .select("p", side)
              .unique()
              .sort("p", side))


def pss_from_dump(sd: Path, provider) -> dict | None:
    """The five topology frames of a dump, or ``None`` when it has no processes.

    Returns ``{"pss", "pss_eff", "pss_noEff", "pss_source_canonical", "pss_sink_canonical"}``.  ``None`` when the
    provider has no ``solve_data/process_source_sink`` or it holds no process.  ``pss_eff`` / ``pss_noEff`` are empty
    frames when their key is absent; the canonical sets are ``None`` when absent or empty.  ``sd`` is the dump's
    ``solve_data`` folder (part of the hook signature; every frame is read from ``provider``).
    """
    del sd  # the provider was seeded from the dump; nothing is read from disk here
    if not provider.has("solve_data/process_source_sink"):
        return None
    pss = _read_pss(provider, "solve_data/process_source_sink")
    if pss.height == 0:
        return None
    return {
        "pss": pss,
        "pss_eff": _read_pss(provider, "solve_data/process_source_sink_eff"),
        "pss_noEff": _read_pss(provider, "solve_data/process_source_sink_noEff"),
        "pss_source_canonical": _read_canonical_set(provider, "solve_data/process_source", "source"),
        "pss_sink_canonical": _read_canonical_set(provider, "solve_data/process_sink", "sink"),
    }


__all__ = ["pss_from_dump"]
