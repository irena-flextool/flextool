"""Regression guard for the execution log (specs/execution_log_redesign.md, Phase 2).

Runs a set of cheap test scenarios through ``cmd_run_flextool`` (a subprocess: the console log exactly as the GUI
and the run log get it, HiGHS' output included) and lints each log with ``tests/_support/log_lint.py``:

* no FlexTool line template (quoted names → ``<n>``, numbers → ``<x>``) more than 3 times within one solve
  (a per-entity message must be one aggregated line: ``flextool.run_warnings.warn_grouped``);
* no FlexTool line longer than 100 characters (quoted names shortened to ``'<n>'``, paths and URLs to ``<path>``);
* every ``[W-dddd]`` code is in the registry, and a run with coded warnings ends with the tally.

The solver's own console block is not linted.  The known offenders are allow-listed below (:data:`KNOWN_REPEATS`,
:data:`KNOWN_LONG`) — each entry is a regular expression matched against the line template.
TODO(P3): convert the sites behind these entries and delete the entries; ``test_allow_list_has_no_stale_entries``
fails when an entry no longer matches anything, so a fixed site's entry cannot linger.

The rolling scenario (72 rolls, ~100 s) is marked ``slow``.
"""
from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from flextool.run_warnings import WARNINGS
from flextool.process_outputs.write_outputs import SOLVE_SUMMARY_END, SOLVE_SUMMARY_START, solve_summary_lines
from tests._support.log_lint import MAX_LINE, MAX_REPEATS, flextool_lines, lint, template

# Chosen to cover the warnings the test scenarios print (the survey of every tests/scenarios.yaml scenario on the
# main fixture, 2026-11-19): unconnected nodes and empty flowGroups (base), timeset notes (base_weighted),
# capacity margin shortfall (capacity_margin), timeset blocks (coal_wind_min_uptime), overlapping realized
# dispatch (coal_unit_size_MIP_wind), period-map notes (network_coal_wind_battery_invest_cumulative), multi-period
# user constraints and anti-energy units (test_a_lot), several solves (wind_battery_invest_lifetime_renew_4solve).
FAST_SCENARIOS = (
    "base",
    "base_weighted",
    "capacity_margin",
    "coal_wind_min_uptime",
    "coal_unit_size_MIP_wind",
    "network_coal_wind_battery_invest_cumulative",
    "test_a_lot",
    "wind_battery_invest_lifetime_renew_4solve",
)
SLOW_SCENARIOS = ("fullYear_roll",)

# (template regex, why it is allowed).  A new offender: fix the site rather than adding an entry.
KNOWN_REPEATS: list[tuple[str, str]] = []
KNOWN_LONG: list[tuple[str, str]] = [
    # Kept on one line on purpose: it is printed once per run (again only when it changes), it is the option list
    # a user copies into solver_arguments or compares between runs, and tests of every solver match it whole.
    (r"^Solver args: ", "the solver's whole option list, once per run (copyable as is; the GUI wraps it)"),
]
# Entries that only the slow scenarios exercise (the stale check skips them when those did not run).
SLOW_ONLY: set[str] = set()

_CODE = re.compile(r"\[(W-\d{4})\]")
_LOGS: dict[str, str] = {}
_ROOT: list[Path] = []  # the runs' folder (one per scenario)
_USED: set[str] = set()


def _run(scenario: str, db_url: str, solver_config_dir: Path, root: Path) -> str:
    folder = root / scenario
    folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "flextool.cli.cmd_run_flextool", db_url, "--scenario-name", scenario,
         "--solver-config-dir", str(solver_config_dir), "--work-folder", str(folder / "work"),
         "--output-location", str(folder / "out"), "--write-methods", "csv", "--threads", "1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=folder, timeout=850,
    )
    log = result.stdout + result.stderr
    assert result.returncode == 0, f"{scenario} exited {result.returncode}:\n{log[-3000:]}"
    return log


@pytest.fixture(scope="module")
def _runner(test_db_url, test_solver_config_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("log_lint")
    _ROOT[:] = [root]

    def log_of(scenario: str) -> str:
        if scenario not in _LOGS:
            _LOGS[scenario] = _run(scenario, test_db_url, test_solver_config_dir, root)
        return _LOGS[scenario]

    missing = [s for s in FAST_SCENARIOS if s not in _LOGS]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for scenario, log in zip(missing, pool.map(log_of, missing)):
            _LOGS[scenario] = log
    yield log_of
    _LOGS.clear()
    _USED.clear()


def _allowed(tmpl: str, entries: list[tuple[str, str]]) -> bool:
    for pattern, _why in entries:
        if re.search(pattern, tmpl):
            _USED.add(pattern)
            return True
    return False


_PARAMS = [*FAST_SCENARIOS, *(pytest.param(s, marks=pytest.mark.slow) for s in SLOW_SCENARIOS)]


@pytest.mark.solver
@pytest.mark.parametrize("scenario", _PARAMS)
def test_log_has_no_repeated_lines(_runner, scenario):
    result = lint(_runner(scenario))
    bad = {f"{solve}: {tmpl} (×{n})" for (solve, tmpl), n in result.repeats.items()
           if not _allowed(tmpl, KNOWN_REPEATS)}
    assert not bad, (f"lines repeated more than {MAX_REPEATS} times within a solve (aggregate them with "
                     "flextool.run_warnings.warn_grouped):\n" + "\n".join(sorted(bad)))


@pytest.mark.solver
@pytest.mark.parametrize("scenario", _PARAMS)
def test_log_has_no_long_lines(_runner, scenario):
    result = lint(_runner(scenario))
    bad = [f"{tmpl} ({length} chars)" for tmpl, length in result.long_lines if not _allowed(tmpl, KNOWN_LONG)]
    assert not bad, f"FlexTool lines longer than {MAX_LINE} characters:\n" + "\n".join(bad)


@pytest.mark.solver
@pytest.mark.parametrize("scenario", _PARAMS)
def test_codes_are_registered_and_tallied(_runner, scenario):
    log = _runner(scenario)
    codes = [c for _, line in flextool_lines(log) if not line.startswith("Warnings: ")
             for c in _CODE.findall(line)]
    assert set(codes) <= set(WARNINGS), sorted(set(codes) - set(WARNINGS))
    warned = [c for c in codes if WARNINGS[c].severity == "warning"]
    tally = [line for line in log.splitlines() if line.startswith("Warnings: ")]
    if warned:
        assert len(tally) == 1 and all(f"{c} ×" in tally[0] for c in set(warned)), tally
        assert "Warnings explained: https://" in log
    else:
        assert tally == []


@pytest.mark.solver
def test_allow_list_has_no_stale_entries(_runner):
    """Every allow-list entry still matches a line of the scenarios (delete an entry once its site is fixed)."""
    for scenario in FAST_SCENARIOS:
        result = lint(_runner(scenario))
        for (_solve, tmpl) in result.repeats:
            _allowed(tmpl, KNOWN_REPEATS)
        for tmpl, _length in result.long_lines:
            _allowed(tmpl, KNOWN_LONG)
    ran_slow = all(s in _LOGS for s in SLOW_SCENARIOS)
    if ran_slow:
        for scenario in SLOW_SCENARIOS:
            result = lint(_LOGS[scenario])
            for (_solve, tmpl) in result.repeats:
                _allowed(tmpl, KNOWN_REPEATS)
            for tmpl, _length in result.long_lines:
                _allowed(tmpl, KNOWN_LONG)
    stale = [p for p, _ in KNOWN_REPEATS + KNOWN_LONG if p not in _USED and (ran_slow or p not in SLOW_ONLY)]
    assert not stale, "allow-list entries that match nothing any more (delete them):\n" + "\n".join(stale)


@pytest.mark.solver
@pytest.mark.parametrize("scenario", _PARAMS)
def test_run_ends_with_the_solve_summary(_runner, scenario):
    """A successful run prints the whole summary_solve.csv (without its time line) after "Full execution time",
    between the summary markers the GUI folds on."""
    log = _runner(scenario)
    lines = log.splitlines()
    start, end = lines.index(SOLVE_SUMMARY_START), lines.index(SOLVE_SUMMARY_END)
    assert lines.count(SOLVE_SUMMARY_START) == 1
    assert any(line.startswith("--- Full execution time") for line in lines[:start])
    # The rows are those of summary_solve.csv (the runs write csv), less its title line and blank lines.
    (csv_file,) = (_ROOT[0] / scenario / "out").rglob("summary_solve.csv")
    assert lines[start:end + 1] == solve_summary_lines(csv_file.read_text(encoding="utf-8"))


# -- the linter itself (no solver) -------------------------------------------------------------------------------

_SYNTHETIC = """\
Run start                           0s (-) |   163 MB (+163) | 17.9 GB (+17.9) |       1 MB (+1)
Warning: node 'a' is odd.
Warning: node 'b' is odd.
Solve start: dispatch, 1/2
Warning: thing 'x' at 1.5 is odd.
Warning: thing 'y' at 2 is odd.
Warning: thing 'z' at 3e-4 is odd.
Running HiGHS 1.15.1 (git hash: 04024d7): Copyright (c) 2026 under MIT licence terms
WARNING: a solver line that is far too long for the limit but the solver's block is not linted at all, ever ok
Solver                       2.06s (+0.05) |   190 MB (+0.9) |    18.1 GB (+0) |       1 MB (+0)
Solve start: dispatch_roll_2, 2/2
Warning: thing 'w' at 4 is odd.
Warning: node 'c' is odd.
Output: /a/very/long/path/that/would/break/the/limit/on/its/own/without/shortening/it/to/a/marker/ok/yes
--- Solve summary ---
"Total cost counted in the objective (M CUR)",1,"a description far longer than the limit of one hundred characters"
Created, a, p1, 1
Created, a, p2, 1
Created, a, p3, 1
Created, a, p4, 1
--- End of solve summary ---
"""


def test_lint_splits_solves_and_skips_the_solver_block():
    assert template("Warning: thing 'x' in p2020 at -1.5e+03 MW") == "Warning: thing <n> in p<x> at <x> MW"
    lines = flextool_lines(_SYNTHETIC)
    assert [s for s, _ in lines] == ["<run>"] * 2 + ["dispatch"] * 6
    assert not any("HiGHS" in line or line.startswith("WARNING") for _, line in lines)
    result = lint(_SYNTHETIC)
    # 4 'thing' lines in solve dispatch (its roll 2 included); 3 'node' lines but in two solves.
    assert result.repeats == {("dispatch", "Warning: thing <n> at <x> is odd."): 4}
    assert result.long_lines == []
    long_one = "Warning: " + "x" * 100
    assert lint(long_one).long_lines == [(long_one, len(long_one))]
