"""Solve creation-order helper.

Reads ``solve_data/solve_progress.csv`` (appended-per-solve in
``solver_runner._run_highs_or_cplex``) to recover the order in which
solves were executed.  Used to give every reader the same canonical row
order so cross-reader operations (``DataFrame.mul`` with ``level=``)
align cleanly.

Why this matters
----------------
``DataFrame.mul(other, axis=1, level=0)`` raises "Join on level
between two MultiIndex objects is ambiguous" when the operands' row
MultiIndexes have different lexsort depths OR different row orders.
Plain ``sort_index()`` would fix that — but lexicographic sort puts
``dispatch_fullYear_roll_roll_10`` before ``roll_2`` and breaks
``drop_levels.py`` which uses ``keep='first'`` on dedup, expecting
parent solves (e.g. ``invest_24h``, ``invest_5weeks_p2020``) to appear
before child rolls.

``canonical_sort`` solves both: rows are reordered by solve creation
order (parent first, then numerically-sequenced rolls), so the
``keep='first'`` semantics + cross-reader alignment both hold.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_solve_order(work_folder: Path | str) -> dict[str, int]:
    """Map solve names to their creation order (0-indexed).

    Source: ``solve_data/p_entity_pre_existing.csv`` — a per-solve
    parameter that's appended to on every solve and that's present
    for every realistic model.  Unique solves in row order = creation
    order.

    ``solve_progress.csv`` would be more direct but it is NOT a single
    pandas table — it interleaves headers like ``Init time,...`` with
    the actual per-solve table, which trips ``pd.read_csv``.

    Returns ``{}`` if the file is absent (legacy paths, fresh worktree).
    """
    path = Path(work_folder) / "solve_data" / "p_entity_pre_existing.csv"
    if not path.exists():
        return {}
    # Only need the ``solve`` column; ``usecols`` keeps memory low for
    # scenarios with thousands of solve rows.
    df = pd.read_csv(path, usecols=["solve"])
    solves = df["solve"].astype(str).drop_duplicates().tolist()
    return {s: i for i, s in enumerate(solves)}


def canonical_sort(
    df: pd.DataFrame, solve_order: dict[str, int],
) -> pd.DataFrame:
    """Reorder rows by solve creation order; preserve within-solve order.

    Operates only on frames whose row MultiIndex contains ``solve`` as
    a level.  For other frames (or rowless frames) the input is returned
    unchanged.

    **Within-solve row order is preserved** — important for representative-
    period scenarios where the model emits timesteps in selected-RP order
    (e.g. ``t0001..t0024, t0121..t0144, ...``) rather than lex order.
    A naive ``sort_index()`` would lex-sort timesteps and split a frame
    that only differs from another by being empty (so it skipped sort)
    out of alignment.

    Implementation: stable ``argsort`` by ``solve_pos`` only — solves
    missing from ``solve_order`` get position ``-1`` and sort first
    (defensive; every solve appears in ``solve_progress.csv``).
    """
    if not isinstance(df.index, pd.MultiIndex) or "solve" not in df.index.names:
        return df
    if len(df) == 0:
        return df
    solve_pos = (
        df.index.get_level_values("solve").map(solve_order).fillna(-1).astype(int)
    )
    order = np.argsort(np.asarray(solve_pos), kind="stable")
    return df.iloc[order]
