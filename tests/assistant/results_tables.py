"""Synthetic result tables for the result-tool tests, written the way ``write_outputs`` writes them: the frame's
columns under a top ``scenario`` level, the column metadata derived from the output metadata, as a lean parquet;
a membership set as a flat table with a ``scenario`` column."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from flextool.lean_parquet import write_lean_parquet
from flextool.process_outputs._output_meta import derive_column_meta

PERIODS = ["p2020", "p2025"]
TIMES = ["t01", "t02", "t03"]


def dt_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples([(p, t) for p in PERIODS for t in TIMES], names=["period", "time"])


def write_result(folder: Path, stem: str, frame: pd.DataFrame, scenario: str) -> None:
    derived = derive_column_meta(stem, frame.columns)
    meta = {k: v.as_dict() for k, v in derived.items()} if derived else None
    wrapped = pd.concat({scenario: frame}, axis=1, names=["scenario"])
    folder.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(wrapped, folder / f"{stem}.parquet", column_metadata=meta)


def write_membership(folder: Path, stem: str, rows: list[tuple[str, ...]], names: list[str], scenario: str) -> None:
    frame = pd.DataFrame(rows, columns=names)
    frame.insert(0, "scenario", scenario)
    folder.mkdir(parents=True, exist_ok=True)
    write_lean_parquet(frame, folder / f"{stem}.parquet")


def succeed_writing(write_tables: Callable[[Path], None]) -> Callable:
    """A run record outcome: one optimal solve, then *write_tables* writes the run's result files."""
    def outcome(rec) -> None:
        rec.solve_started("y2020", "y2020", 0, "Solve start: y2020, 1/1")
        rec.solve_classified(model_status="kOptimal", acceptance="optimal", message="", objective=1.0,
                             warm_used=False)
        rec.solve_finished(0)
        rec.outputs_cleared()
        write_tables(rec.folder)
        rec.outputs_written()
        rec.finish("succeeded", exit_code=0, error=None)
    return outcome
