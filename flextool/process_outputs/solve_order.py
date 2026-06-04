"""Solve creation-order helper.

Reads ``solve_data/solve__p_entity_pre_existing.csv`` (a per-solve parameter
that's appended on every solve and that's present for every realistic
model) to recover the order in which solves were executed.

Used by :mod:`read_variables._read_from_parquet` to concatenate per-solve
parquets in solve creation order, so the resulting row order matches
the CSV-reader pathway (where rows are naturally in creation order via
phase-1 ``for {s in solve_current, ...}`` printf appends).  Matching
row order at the source avoids needing any post-concat sort or
per-call-site reindex.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# Creation-order manifest written incrementally by the per-roll output
# hook (one line per sub-solve, in execution order).  This is the
# authoritative order source on the default ``keep_solutions=False`` flow,
# where ``solve_data/solve__p_entity_pre_existing.csv`` is absent.
_MANIFEST_NAME = "_solve_order.txt"


def append_solve_order(output_dir: Path | str, solve_name: str) -> None:
    """Append ``solve_name`` to the creation-order manifest in ``output_dir``.

    Idempotent within a run only in the sense that re-running the cascade
    into the same folder must start from a clean manifest — callers that
    re-run must clear ``output_dir`` first (the cascade does).  Called
    once per sub-solve, in execution order, so the file's line order is
    the cascade creation order both the variable reader and the stage-3
    param/set union read back via :func:`load_solve_order`.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / _MANIFEST_NAME).open("a", encoding="utf-8") as f:
        f.write(f"{solve_name}\n")


def _load_from_manifest(output_dir: Path) -> dict[str, int]:
    path = output_dir / _MANIFEST_NAME
    if not path.exists():
        return {}
    seen: list[str] = []
    seen_set: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and name not in seen_set:
            seen_set.add(name)
            seen.append(name)
    return {s: i for i, s in enumerate(seen)}


def load_solve_order(work_folder: Path | str) -> dict[str, int]:
    """Map solve names to their creation order (0-indexed).

    Sources, in priority:

    1. ``output_raw/_solve_order.txt`` — the per-roll manifest appended by
       the output hook (:func:`append_solve_order`).  Authoritative on the
       default ``keep_solutions=False`` flow.
    2. ``solve_data/solve__p_entity_pre_existing.csv`` — legacy per-solve
       appendage; unique solves in row order = creation order.

    Returns ``{}`` if neither source is present.
    """
    work = Path(work_folder)
    manifest = _load_from_manifest(work / "output_raw")
    if manifest:
        return manifest

    path = work / "solve_data" / "solve__p_entity_pre_existing.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["solve"])
    solves = df["solve"].astype(str).drop_duplicates().tolist()
    return {s: i for i, s in enumerate(solves)}
