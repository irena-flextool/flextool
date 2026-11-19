"""Lint a FlexTool run's console log (the stdout of ``cmd_run_flextool``, as the GUI shows it).

:func:`lint` splits the log into *solves* (a ``Solve start: <solve>, k/n`` line opens one; the rolls of a
rolling solve and the repeated starts of one solve count as one solve, the lines before the first start are
the ``<run>`` scope), drops what is not FlexTool's own text (the solver's console block, which opens at
``Running HiGHS`` / ``Running <solver>`` and ends at the next timing row; the timing rows themselves; blank
lines; the solve summary at the end of a run, ``--- Solve summary ---`` ... ``--- End of solve summary ---``,
which is the content of ``summary_solve.csv``, a table, not messages) and reports

* **repeats**: a line template (:func:`template`: quoted names → ``<n>``, numbers → ``<x>``) that appears more
  than :data:`MAX_REPEATS` times within one solve (a per-entity warning that should be one aggregated line);
* **long lines**: a FlexTool line longer than :data:`MAX_LINE` characters once quoted names are shortened to
  ``'<n>'`` and paths and URLs to ``<path>`` (:func:`measured`: the user's names and folders are as long as they
  are; the limit is on FlexTool's own text).

``tests/test_execution_log_lint.py`` runs it on a set of test scenarios with an allow-list of the known
offenders.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

MAX_REPEATS = 3
MAX_LINE = 100

#: The lines that open a solver's console dump (the GUI's ``_SOLVER_BLOCK_STARTS``).
SOLVER_BLOCK_STARTS = ("Running HiGHS", "Running Gurobi", "Running CPLEX", "Running Xpress", "Running COPT")
SUMMARY_START = "--- Solve summary ---"
SUMMARY_END = "--- End of solve summary ---"
_SOLVE_START = re.compile(r"^Solve start: (.+?), \d+/\d+\s*$")
_ROLL_SUFFIX = re.compile(r"_roll_\d+$")
# A timing / memory row (or its header): cells separated by "|" with a size or the word "memory".
_SIZE = re.compile(r"\d+(?:\.\d+)? ?[kMGT]B\b")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|(?<=[A-Za-z_])\d+")
# An absolute path or a URL (sqlite:///..., https://..., /tmp/..., C:\...).
_PATH = re.compile(r"(?:[a-z][a-z0-9+.-]*://\S*|(?<![\w.])(?:[A-Za-z]:)?[\\/][^\s'\",;()]*[\\/][^\s'\",;()]*)")


def is_timing_row(line: str) -> bool:
    return "|" in line and (_SIZE.search(line) is not None or "memory" in line)


def template(line: str) -> str:
    """*line* with quoted names as ``<n>`` and numbers as ``<x>`` (``p2020`` → ``p<x>``)."""
    return _NUMBER.sub("<x>", _QUOTED.sub("<n>", line.strip()))


def measured(line: str) -> str:
    """*line* as its length is checked: quoted names as ``'<n>'``, paths and URLs as ``<path>``."""
    return _PATH.sub("<path>", _QUOTED.sub("'<n>'", line))


@dataclass
class LintResult:
    #: ``{(solve, template): count}`` of the templates seen more than MAX_REPEATS times in one solve.
    repeats: dict[tuple[str, str], int] = field(default_factory=dict)
    #: ``[(template, length)]`` of the FlexTool lines longer than MAX_LINE (:func:`measured`), one per template.
    long_lines: list[tuple[str, int]] = field(default_factory=list)


def flextool_lines(log: str) -> list[tuple[str, str]]:
    """``[(solve, line)]`` of FlexTool's own non-blank lines of *log*, with the solve each belongs to."""
    out: list[tuple[str, str]] = []
    solve = "<run>"
    in_solver = False
    in_summary = False
    for raw in log.splitlines():
        line = raw.rstrip()
        if line.strip() == SUMMARY_START:
            in_summary = True
        if in_summary:
            in_summary = line.strip() != SUMMARY_END
            continue
        match = _SOLVE_START.match(line)
        if match:
            solve = _ROLL_SUFFIX.sub("", match.group(1))
            in_solver = False
            continue
        if is_timing_row(line):
            in_solver = False
            continue
        if line.startswith(SOLVER_BLOCK_STARTS):
            in_solver = True
        if in_solver or not line.strip():
            continue
        out.append((solve, line))
    return out


def lint(log: str, max_repeats: int = MAX_REPEATS, max_line: int = MAX_LINE) -> LintResult:
    result = LintResult()
    counts: Counter[tuple[str, str]] = Counter()
    long_seen: dict[str, int] = {}
    for solve, line in flextool_lines(log):
        tmpl = template(line)
        counts[(solve, tmpl)] += 1
        length = len(measured(line))
        if length > max_line:
            long_seen[tmpl] = max(length, long_seen.get(tmpl, 0))
    result.repeats = {key: n for key, n in counts.items() if n > max_repeats}
    result.long_lines = sorted(long_seen.items())
    return result
