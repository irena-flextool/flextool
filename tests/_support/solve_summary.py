"""Read the solve summary a FlexTool run prints at its end (``--- Solve summary ---`` ...
``--- End of solve summary ---``: the content of ``summary_solve.csv``) from the run's stdout."""
from __future__ import annotations

from flextool.process_outputs.write_outputs import SOLVE_SUMMARY_END, SOLVE_SUMMARY_START


def summary_rows(stdout: str) -> list[str]:
    """The rows of the last solve summary in *stdout* (without the marker lines); ``[]`` when there is none."""
    lines = [line.rstrip("\r") for line in stdout.splitlines()]
    starts = [i for i, line in enumerate(lines) if line.strip() == SOLVE_SUMMARY_START]
    if not starts:
        return []
    rows = lines[starts[-1] + 1:]
    end = next((i for i, line in enumerate(rows) if line.strip() == SOLVE_SUMMARY_END), None)
    assert end is not None, "the solve summary has no end marker"
    return rows[:end]


def summary_objectives(stdout: str) -> dict[str, float]:
    """``{solve: objective}`` from the summary's "Solve","Objective" table, in currency (the summary has M CUR)."""
    rows = summary_rows(stdout)
    header = next(i for i, row in enumerate(rows) if row.startswith('"Solve","Objective"'))
    out: dict[str, float] = {}
    for row in rows[header + 1:]:
        if not row.strip() or row.startswith('"'):
            break
        solve, value = row.rsplit(",", 1)
        out[solve] = float(value) * 1e6
    return out
