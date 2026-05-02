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


def load_solve_order(work_folder: Path | str) -> dict[str, int]:
    """Map solve names to their creation order (0-indexed).

    Source: ``solve_data/solve__p_entity_pre_existing.csv`` — appended on
    every solve; unique solves in row order = creation order.

    Returns ``{}`` if the file is absent.
    """
    path = Path(work_folder) / "solve_data" / "solve__p_entity_pre_existing.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, usecols=["solve"])
    solves = df["solve"].astype(str).drop_duplicates().tolist()
    return {s: i for i, s in enumerate(solves)}
