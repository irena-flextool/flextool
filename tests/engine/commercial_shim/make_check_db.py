"""Build the small FlexTool database of the item 230 solver checks.

Used by ``test_item230_commercial_shims.py`` and, with real solver licences, by hand (see
``specs/item230_colleague_checklist.md``)::

    python tests/engine/commercial_shim/make_check_db.py <folder>

writes ``<folder>/item230_check.sqlite`` (migrated to the current schema) from ``tests/fixtures/tests.json``,
with these scenarios for each solver ``<s>`` in gurobi, cplex, xpress, copt (solve ``y2020_2day_dispatch``):

* ``lp_<s>``: the ``coal`` LP (two days, hourly);
* ``mip_<s>``: ``coal_min_load_MIP_wind`` (integer online units);
* ``stack_<s>``: the LP with ``solver_mip_gap`` 0.2 and ``solver_arguments`` gap 0.1 / time limit 800;
* ``smg_<s>``: the LP with ``solver_mip_gap`` 0.2;

* ``paths_<s>``: the LP with a ``solver_precommand`` that would create ``<folder>/precommand_ran_<s>`` and, for
  each solver but COPT, a ``solver_arguments`` parameter that would make the solver write at
  ``<folder>/path_option_<s>`` (item 236: the precommand is never run, and a path from the input database is
  refused before the solver starts);
* ``split_<s>``: the LP over the periods ``p2020`` and ``p2025`` with ``realized_periods`` a 2-D map, so FlexTool
  splits the solve into ``y2020_2day_dispatch_p2020`` and ``y2020_2day_dispatch_p2025``; with ``solver_mip_gap``
  0.2 and ``solver_arguments`` time limit 800 (each copy must keep the solver and all of its settings);

plus ``unknown_gurobi`` (``solver_arguments`` ``NoSuchParameter`` 1, not a Gurobi parameter: refused unless this
computer allows unlisted names) and ``iterlimit_xpress`` (``LPITERLIMIT``
1, ``PRESOLVE`` 0).  The scenarios ``coal`` and ``coal_min_load_MIP_wind`` themselves solve with HiGHS.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOLVE = "y2020_2day_dispatch"
SOLVERS = ("gurobi", "cplex", "xpress", "copt")
LP_BASE = ["init", "west", "coal"]
#: How each solver's ``solver_arguments`` spell the gap and the time limit in ``stack_<s>``: not the option
#: file's spelling, so the merge by the solver's own name rule is exercised too (case; CPLEX dotted; Gurobi
#: ``_``).  Xpress keeps the upper-case names: its Python ``setControl`` (the shim) takes no other case, and the
#: console's case rule is not known here.
ARG_GAP = {"gurobi": "Mip_Gap", "cplex": "mip.tolerances.mipgap", "xpress": "MIPRELSTOP", "copt": "relgap"}
ARG_TIME = {"gurobi": "time_limit", "cplex": "TimeLimit", "xpress": "MAXTIME", "copt": "timelimit"}
#: A parameter that makes the solver write at a path (each solver's parameter reference; COPT has none), in a
#: spelling the solver's library also reads (Gurobi ignores ``_``, Xpress an ``XPRS_`` prefix).
ARG_PATH = {"gurobi": "Log_File", "cplex": "logfile", "xpress": "XPRS_TUNEROUTPUTPATH"}


def precommand_marker(folder: Path, solver: str) -> Path:
    """The file the ``solver_precommand`` of ``paths_<solver>`` would create if it were run."""
    return Path(folder) / f"precommand_ran_{solver}"


def path_option_target(folder: Path, solver: str) -> Path:
    """Where the ``solver_arguments`` path parameter of ``paths_<solver>`` would make the solver write."""
    return Path(folder) / f"path_option_{solver}"


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _arguments(pairs: list[list]) -> dict:
    return {"index_type": "str", "index_name": "x", "data": pairs}


def _period_map(pairs: list[list]) -> dict:
    return {"type": "map", "index_type": "str", "index_name": "period", "data": pairs}


#: The periods of ``split_<s>`` and the names of the solves FlexTool makes of ``y2020_2day_dispatch`` for them.
SPLIT_PERIODS = ("p2020", "p2025")
SPLIT_SOLVES = tuple(f"{SOLVE}_{p}" for p in SPLIT_PERIODS)


def build(folder: Path) -> str:
    """Write ``<folder>/item230_check.sqlite``; return its ``sqlite:///`` URL."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    lps = _lp_snapshot()
    folder = Path(folder)
    data = json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text(encoding="utf-8"))
    for s in SOLVERS:
        data = lps._overlay(data, name=f"lp_{s}", base=LP_BASE, values=[["solve", SOLVE, "solver", s]])
        data = lps._overlay(data, name=f"mip_{s}", base="coal_min_load_MIP_wind",
                            values=[["solve", SOLVE, "solver", s]])
        data = lps._overlay(data, name=f"stack_{s}", base=LP_BASE, values=[
            ["solve", SOLVE, "solver", s],
            ["solve", SOLVE, "solver_mip_gap", 0.2],
            ["solve", SOLVE, "solver_arguments", _arguments([[ARG_GAP[s], 0.1], [ARG_TIME[s], 800]])],
        ])
        data = lps._overlay(data, name=f"smg_{s}", base=LP_BASE, values=[
            ["solve", SOLVE, "solver", s], ["solve", SOLVE, "solver_mip_gap", 0.2]])
        paths = [["solve", SOLVE, "solver", s],
                 ["solve", SOLVE, "solver_precommand", f"touch {precommand_marker(folder, s)}"]]
        if s in ARG_PATH:
            paths.append(["solve", SOLVE, "solver_arguments",
                          _arguments([[ARG_PATH[s], str(path_option_target(folder, s))]])])
        data = lps._overlay(data, name=f"paths_{s}", base=LP_BASE, values=paths)
        data = lps._overlay(data, name=f"split_{s}", base=LP_BASE, values=[
            ["solve", SOLVE, "solver", s],
            ["solve", SOLVE, "solver_mip_gap", 0.2],
            ["solve", SOLVE, "solver_arguments", _arguments([[ARG_TIME[s], 800]])],
            ["solve", SOLVE, "period_timeset", _period_map([[p, "2day"] for p in SPLIT_PERIODS])],
            ["solve", SOLVE, "years_represented", _period_map([[p, 5.0] for p in SPLIT_PERIODS])],
            ["solve", SOLVE, "realized_periods",
             _period_map([[p, _period_map([[p, "yes"]])] for p in SPLIT_PERIODS])],
        ])
    data = lps._overlay(data, name="unknown_gurobi", base=LP_BASE, values=[
        ["solve", SOLVE, "solver", "gurobi"],
        ["solve", SOLVE, "solver_arguments", _arguments([["NoSuchParameter", 1]])]])
    data = lps._overlay(data, name="iterlimit_xpress", base=LP_BASE, values=[
        ["solve", SOLVE, "solver", "xpress"],
        ["solve", SOLVE, "solver_arguments", _arguments([["LPITERLIMIT", 1], ["PRESOLVE", 0]])]])
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / "item230_check.json"
    json_path.write_text(json.dumps(data), encoding="utf-8")
    url = json_to_db(json_path, folder / "item230_check.sqlite")
    json_path.unlink()
    migrate_database(url)
    return url


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/engine/commercial_shim/make_check_db.py <folder>")
    print(build(Path(sys.argv[1])))
