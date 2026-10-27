"""Canonical output-tree compare — item 163 (design_163.md F7).

Run-to-run order noise (row/column permutation, the dropped ``__set_marker__`` helper column) is not a
real difference: the F7 canonical compare removes it so ``tools/output_snapshot.py`` and
``tests/test_output_work_folder_isolation.py`` are not polluted by it.  This module holds that one
compare so both use the same rule.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from flextool.lean_parquet import read_lean_parquet

__all__ = ["diff_output_trees"]

# Compared verbatim by name only (never by content): the build-time manifest and the plot-plan
# metadata JSON (its sibling ``*_plan.parquet`` data file IS compared below).
_SKIP_JSON_EXACT = {"manifest.json", "_dispatch_metadata.json"}


def _is_skipped_json(name: str) -> bool:
    return name in _SKIP_JSON_EXACT or name.endswith("_plan.json")


def _canonicalize_parquet(path: Path) -> pd.DataFrame:
    """*path* with ``__set_marker__`` dropped and columns/rows sorted — the F7 canonical form.

    ``DataFrame.empty`` is true when EITHER axis has length 0 (e.g. a pure-set frame whose only
    column was the dropped ``__set_marker__``), so the row sort is gated on the row count, not on
    ``.empty`` — otherwise a 0-column, N>0-row frame would keep its (order-only) row order.
    """
    df = read_lean_parquet(path)
    if "__set_marker__" in df.columns:
        df = df.drop(columns=["__set_marker__"])
    df = df.sort_index(axis=1)
    if len(df.index) > 0:
        df = df.sort_index(axis=0)
    return df


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if list(a.columns) != list(b.columns) or a.shape != b.shape:
        return False
    try:
        return a.equals(b)
    except Exception:
        return False


def _diff_tree(a_dir: Path, b_dir: Path) -> list[str]:
    """Canonical compare of every file under two directories (relative paths)."""
    diffs: list[str] = []
    a_files = (
        {p.relative_to(a_dir).as_posix() for p in a_dir.rglob("*") if p.is_file()}
        if a_dir.is_dir() else set()
    )
    b_files = (
        {p.relative_to(b_dir).as_posix() for p in b_dir.rglob("*") if p.is_file()}
        if b_dir.is_dir() else set()
    )
    for rel in sorted(a_files - b_files):
        if not _is_skipped_json(Path(rel).name):
            diffs.append(f"{rel}: missing in {b_dir}")
    for rel in sorted(b_files - a_files):
        if not _is_skipped_json(Path(rel).name):
            diffs.append(f"{rel}: missing in {a_dir}")
    for rel in sorted(a_files & b_files):
        name = Path(rel).name
        if _is_skipped_json(name):
            continue
        a_path, b_path = a_dir / rel, b_dir / rel
        if name.endswith(".parquet"):
            try:
                a_df = _canonicalize_parquet(a_path)
                b_df = _canonicalize_parquet(b_path)
            except Exception as exc:
                diffs.append(f"{rel}: could not compare ({exc})")
                continue
            if not _frames_equal(a_df, b_df):
                diffs.append(f"{rel}: differs")
        else:
            if a_path.read_bytes() != b_path.read_bytes():
                diffs.append(f"{rel}: differs")
    return diffs


def diff_output_trees(a_work: "Path | str", b_work: "Path | str", scenario: str) -> list[str]:
    """Canonical diff of two work folders' output sets (item 163, F7).

    Compares ``output_raw/`` (the cascade's per-roll parquet stash: per-solve parquets,
    ``_solve_order.txt``, ``manifest.json``) and ``out/output_parquet/<scenario>/`` (the final
    per-scenario bundle a CLI-shaped ``write_outputs`` call produces) between ``a_work`` and
    ``b_work``.  Every parquet is compared with ``__set_marker__`` dropped and rows/columns sorted
    (F7 — row/column order is not a real difference).  ``manifest.json``, ``_dispatch_metadata.json``
    and every ``*_plan.json`` (plot-plan metadata — its ``*_plan.parquet`` sibling IS compared) are
    skipped.

    Returns the sorted list of differing or missing relative paths, each prefixed with the tree it
    came from (``output_raw/...`` or ``out/output_parquet/<scenario>/...``).
    """
    a_work, b_work = Path(a_work), Path(b_work)
    diffs: list[str] = []
    for a_root, b_root, prefix in (
        (a_work / "output_raw", b_work / "output_raw", "output_raw"),
        (
            a_work / "out" / "output_parquet" / scenario,
            b_work / "out" / "output_parquet" / scenario,
            f"out/output_parquet/{scenario}",
        ),
    ):
        diffs.extend(f"{prefix}/{d}" for d in _diff_tree(a_root, b_root))
    return sorted(diffs)
