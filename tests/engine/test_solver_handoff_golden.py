"""What each solver is handed: a golden record of the argv, the stdin script and the option file (item 240 B4).

The commercial solvers get their resolved options as ``Name=value`` arguments (Gurobi) or as lines of a stdin
command script (CPLEX, Xpress, COPT); the ``--save-memory`` HiGHS child gets them through the files it is passed.
This module runs a fixed corpus of option sets through the real hand-off, with no solver program:

* commercial: the option layers (option file < ``solver_mip_gap`` < ``solver_arguments`` < flags) through
  ``resolve_commercial_options`` and ``_solve_commercial_subprocess`` up to the point where the solver would start
  (``_run_solver_process`` is replaced by a recorder): the argv, the stdin script and the ``<solve>.opt`` record;
* HiGHS ``--save-memory``: the effective option dict through ``_solve_highs_subprocess`` and the child
  ``cmd_solve_mps.main`` (run in this process), up to the point where the child reads the model: the ``<solve>.opt``
  record, whether the child took the options, and the options HiGHS then holds (``writeOptions``, the lines that
  differ from a fresh ``Highs()``).

The golden file was generated on the code before the B4 refactor (which hands the option list to the builders and
the child directly instead of writing a file and reading it back), so a refactor must reproduce it exactly.
Regenerate it only for a deliberate change: ``FLEXTOOL_WRITE_SOLVER_HANDOFF_GOLDEN=1 pytest <this file>``.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import polars as pl
import pytest

from polar_high import Param, Problem

from flextool.cli.cmd_run_flextool import build_parser
from flextool.engine.input._solve_config import SolverConfig
from flextool.engine.solve import _solver_dispatch, _subprocess_solve
from flextool.run_options import RunOptions

GOLDEN = Path(__file__).parent / "data" / "solver_handoff_golden.json"
_TEMPLATES = Path(_subprocess_solve.__file__).parents[2] / "solver_config"
_FLAGS = ["--solver-mip-gap", "0.001", "--solver-time-limit", "1800"]


def _template(solver: str) -> str:
    return (_TEMPLATES / f"{solver}.opt.template").read_text()


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------

#: (case id, solver, option-file text or None, SolverConfig keyword arguments, flags).
_COMMERCIAL_CORPUS: list[tuple[str, str, str | None, dict, list[str]]] = [
    *[(f"{s}_template", s, _template(s), {}, []) for s in ("gurobi", "cplex", "xpress", "copt")],
    *[(f"{s}_nothing", s, None, {}, []) for s in ("gurobi", "cplex", "xpress", "copt")],
    *[(f"{s}_flags_only", s, None, {}, _FLAGS) for s in ("gurobi", "cplex", "xpress", "copt")],
    *[(f"{s}_stacked", s, _template(s), {"mip_gap": 0.01, "time_limit": 600.0, "threads": 4,
                                         "options": {"Presolve": 2}}, _FLAGS)
      for s in ("gurobi", "cplex", "xpress", "copt")],
    ("gurobi_spellings", "gurobi", "Threads 0\nMIPGap 0.002\n",
     {"mip_gap": 0.01, "options": {"Mip_Gap": 0.005, "mipgap": "0.004", "Presolve": 2.0}}, ["--solver-mip-gap",
                                                                                            "0.001"]),
    ("gurobi_numbers", "gurobi", None, {"options": {
        "Threads": 1.0, "Method": "-3.00", "Heuristics": 0.05, "TimeLimit": "inf", "FeasibilityTol": 1e-4,
        "OutputFlag": True, "Quad": False, "IntFeasTol": "1e-9", "Cuts": -1, "NodeLimit": "1e+20",
    }}, []),
    ("gurobi_paths_and_result_file", "gurobi", "LogFile C:\\runs\\x.log\nNodefileDir /srv/nodes\n"
     "ResultFile mine.sol\n", {"options": {"SolFiles": "sols", "ResultFile": "other.json", "Record": 1}}, []),
    ("cplex_names", "cplex", "threads 1\nmip.tolerances.mipgap 0.002\nmip tolerances absmipgap 0.5\n",
     {"mip_gap": 0.01, "options": {"mip.strategy.file": 3, "Emphasis.MIP": "2", "preprocessing presolve": "yes",
                                    "workdir": "scratch"}}, []),
    ("cplex_numbers", "cplex", None, {"options": {
        "timelimit": 3600.0, "mip tolerances mipgap": "1e-4", "simplex tolerances optimality": 1e-9,
        "parallel": -1, "emphasis numerical": True, "lpmethod": "4.0",
    }}, ["--solver-time-limit", "100.5"]),
    ("xpress_names", "xpress", "THREADS 1\nXPRS_MIPRELSTOP 0.002\nxprs_maxtime 300\n",
     {"mip_gap": 0.01, "time_limit": 99.2, "options": {"XPRS_PRESOLVE": 0, "maxtime": "120",
                                                        "TUNEROUTPUTPATH": "tuner"}}, []),
    ("xpress_numbers", "xpress", None, {"options": {
        "MIPRELSTOP": 1e-4, "FEASTOL": "1e-9", "DEFAULTALG": 4.0, "OUTPUTLOG": True, "MAXNODE": "-1",
        "BARGAPSTOP": "inf",
    }}, _FLAGS),
    ("copt_names", "copt", "Threads 1\nrelgap 0.002\nLogging 1\n",
     {"mip_gap": 0.01, "options": {"RELGAP": 0.004, "Presolve": "-1", "TimeLimit": 1e+20, "LogFile": "copt.txt"}},
     ["--solver-time-limit", "60"]),
    ("copt_numbers", "copt", None, {"options": {
        "FeasTol": 1e-6, "LpMethod": 2.0, "Crossover": False, "HeurLevel": "3.000", "IntTol": "1e-7",
    }}, []),
]

#: (case id, the effective HiGHS option dict handed to the ``--save-memory`` solve).
_HIGHS_CORPUS: list[tuple[str, dict]] = [
    ("highs_template", _solver_dispatch._parse_highs_opt_text(_template("highs"))),
    ("highs_nothing", {}),
    ("highs_types", {
        "presolve": "off", "solver": "ipm", "parallel": "on", "threads": 2, "mip_rel_gap": 0.01,
        "time_limit": 3600.0, "output_flag": False, "log_to_console": True, "random_seed": 7,
        "primal_feasibility_tolerance": 1e-9, "run_crossover": "choose",
    }),
    ("highs_text_values", {
        "threads": "+2", "mip_rel_gap": ".5", "presolve": "ON", "output_flag": "True", "log_to_console": "1",
        "mip_detect_symmetry": "on", "time_limit": "inf", "dual_feasibility_tolerance": "5e-1",
    }),
    ("highs_paths", {"log_file": "C:\\runs\\x.log", "write_model_file": "/srv/m.mps", "solution_file": "sol.txt",
                     "write_solution_to_file": True}),
    ("highs_empty_solution_file", {"presolve": "off", "solution_file": "", "threads": 1}),
    ("highs_empty_log_file", {"log_file": ""}),
    ("highs_whole_float_for_int", {"threads": 1.0}),
    ("highs_whole_float_text_for_int", {"threads": "1.0"}),
    ("highs_unknown_name", {"presolve": "off", "Presolve": "on"}),
    ("highs_out_of_range", {"mip_rel_gap": -1}),
    ("highs_minus_inf_time", {"time_limit": "-inf"}),
]


# ---------------------------------------------------------------------------
# Running one case through the hand-off
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """Raised by the recorders where the solver (or the child's model read) would start."""


def _tiny_problem() -> Problem:
    p = Problem()
    i = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", i, lower=0.0)
    p.add_cstr("c", over=i, sense=">=", lhs_terms={"x": x},
               rhs_terms={"r": Param(("i",), pl.DataFrame({"i": [0], "value": [1.0]}))})
    p.set_objective(x.to_expr(), sense="min")
    return p


def _run_options(flags: list[str]) -> RunOptions:
    return RunOptions.from_cli(build_parser().parse_args(["sqlite:///x.sqlite", *flags]), environ={})


def _commercial_case(solver: str, opt_text: str | None, cfg: dict, flags: list[str], work: Path,
                     monkeypatch: pytest.MonkeyPatch) -> dict:
    _solver_dispatch.reset_run_warnings()
    run_options = _run_options(flags)
    resolved = _solver_dispatch.resolve_commercial_options(
        solver, opt_file="cfg/x.opt" if opt_text is not None else None, opt_text=opt_text,
        solver_config=SolverConfig(name=solver, **cfg), run_options=run_options, solve_name="invest",
    )
    seen: dict = {}

    def record(argv, *, stdin_text=None, cwd=None, **_kwargs):
        seen.update(argv=list(argv), stdin=stdin_text)
        raise _Stop

    monkeypatch.setattr(_subprocess_solve, "_find_solver_binary", lambda name: Path("bin"))
    monkeypatch.setattr(_subprocess_solve, "_run_solver_process", record)
    with pytest.raises(_Stop):
        _subprocess_solve._solve_commercial_subprocess(
            _tiny_problem(), solver, {o.name: o.value for o in resolved}, solve_name="invest", logger=None,
            work_folder=work, option_layers={o.name: o.layer for o in resolved}, run_options=run_options,
        )
    record_file = work / "solve_data" / "subprocess" / "invest.opt"
    seen["opt"] = record_file.read_text() if record_file.is_file() else None
    return seen


def _options_dump(highs) -> list[str]:
    """The option lines of *highs* (``writeOptions``)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "options.txt"
        highs.writeOptions(str(path))
        return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def _highs_case(options: dict, work: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    highspy = pytest.importorskip("highspy")
    from flextool.cli import cmd_solve_mps

    default = set(_options_dump(highspy.Highs()))
    seen: dict = {}

    class Recorder(highspy.Highs):
        def readModel(self, *args, **kwargs):  # noqa: N802 - HiGHS' name
            seen["deviations"] = [line for line in _options_dump(self) if line not in default]
            raise _Stop

    def child(cmd, **_kwargs):
        args = list(cmd)[list(cmd).index("flextool.cli.cmd_solve_mps") + 1:]
        try:
            code = cmd_solve_mps.main(args)
        except _Stop:
            code = None
        seen["accepted"] = code is None
        if code is not None:
            seen["exit_code"] = code
        raise _Stop

    monkeypatch.setattr(highspy, "Highs", Recorder)
    monkeypatch.setattr(subprocess, "run", child)
    with pytest.raises(_Stop):
        _subprocess_solve._solve_highs_subprocess(
            _tiny_problem(), dict(options), solve_name="invest", logger=None, work_folder=work,
            run_options=_run_options([]),
        )
    seen["opt"] = (work / "solve_data" / "subprocess" / "invest.opt").read_text()
    return seen


def _all_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for case, solver, opt_text, cfg, flags in _COMMERCIAL_CORPUS:
        out[case] = _commercial_case(solver, opt_text, cfg, flags, tmp_path / case, monkeypatch)
    for case, options in _HIGHS_CORPUS:
        out[case] = _highs_case(options, tmp_path / case, monkeypatch)
    return out


def test_solver_handoff_matches_the_golden_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    # The record is how options are handed over, made before item 240 B1 refused names that are not in FlexTool's
    # lists (the corpus has CPLEX ``Presolve``); allowing such names keeps it the same record.
    monkeypatch.setenv("FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS", "1")
    monkeypatch.chdir(tmp_path)  # HiGHS opens a log_file as soon as it is set
    cases = _all_cases(tmp_path, monkeypatch)
    if os.environ.get("FLEXTOOL_WRITE_SOLVER_HANDOFF_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"wrote {GOLDEN}")
    golden = json.loads(GOLDEN.read_text())
    assert sorted(cases) == sorted(golden)
    for case in golden:
        assert cases[case] == golden[case], case


def test_golden_record_pins_the_known_cases() -> None:
    """Hand-checked entries of the golden record (so a regenerated file cannot silently drift): the Gurobi
    template's argv, the CPLEX stack's mip gap line, and the two HiGHS cases where reading the option file and
    setting each option differ (an empty value is refused by the child)."""
    golden = json.loads(GOLDEN.read_text())
    assert golden["gurobi_template"]["argv"] == [
        "bin", "Threads=1", "MIPGap=0.002", "LogToConsole=1", "ResultFile=invest.json", "invest.mps"]
    assert golden["gurobi_template"]["stdin"] is None
    assert golden["gurobi_nothing"] == {"argv": ["bin", "ResultFile=invest.json", "invest.mps"], "opt": None,
                                        "stdin": None}
    assert golden["cplex_stacked"]["stdin"].splitlines() == [
        "set threads 4", "set mip tolerances mipgap 0.001", "set timelimit 1800", "set presolve 2",
        "read invest.mps", "optimize", "write invest.sol sol", "quit"]
    for case in ("highs_empty_solution_file", "highs_empty_log_file"):
        assert golden[case]["accepted"] is False and golden[case]["exit_code"] == 2, case
    assert golden["highs_text_values"]["accepted"] is True
    assert "threads = 2" in golden["highs_text_values"]["deviations"]


# ---------------------------------------------------------------------------
# Random option lists: the pairs handed over directly = the file written and read back (item 240 B4)
# ---------------------------------------------------------------------------
#
# Before B4 the builders re-read the ``<solve>.opt`` file written from the options, and the ``--save-memory``
# HiGHS child read its ``key=value`` file with ``readOptions``.  Both readers are still in the code (the user's
# option files use the first; HiGHS keeps the second), so the old chain is rebuilt here and compared, over seeded
# random option lists of the accepted grammar, with what the solver is handed now.

_NAMES = {
    "gurobi": ["Threads", "MIPGap", "Mip_Gap", "mipgap", "Presolve", "LogFile", "Method", "TimeLimit",
               "NodefileStart", "Heuristics", "OutputFlag"],
    "cplex": ["threads", "mip tolerances mipgap", "mip.tolerances.mipgap", "timelimit", "workdir", "emphasis mip",
              "preprocessing presolve", "lpmethod", "cpumask", "read fileencoding"],
    "xpress": ["THREADS", "XPRS_MAXTIME", "maxtime", "MIPRELSTOP", "OUTPUTLOG", "TUNEROUTPUTPATH", "FEASTOL"],
    "copt": ["Threads", "RelGap", "TimeLimit", "LogFile", "Presolve", "FeasTol"],
}
_HIGHS_NAMES = ["presolve", "solver", "parallel", "threads", "mip_rel_gap", "time_limit", "output_flag",
                "log_to_console", "random_seed", "primal_feasibility_tolerance", "run_crossover",
                "mip_heuristic_effort", "log_file", "simplex_strategy", "bogus_option", "Presolve"]
_WORDS = ["on", "off", "auto", "choose", "ipm", "simplex", "scratch", "x.log", "ON", "True", "yes", "ffff", "UTF-8"]
_TEXT_NUMBERS = ["inf", "-inf", "1e-4", "1e+20", "+2", ".5", "5e-1", "1.0", "-3.00", "0", "1", "-1", "100.5"]
_PATHS = ["C:\\runs\\x.log", "/srv/x", "a:b", "sub/y.txt"]


def _random_value(rng):
    kind = rng.randrange(8)
    if kind == 0:
        return rng.randint(-5, 200)
    if kind == 1:
        return round(rng.uniform(-1, 1) * 10 ** rng.randint(-9, 9), rng.randint(0, 12))
    if kind == 2:
        return float(rng.randint(-20, 20))
    if kind == 3:
        return rng.choice([True, False])
    if kind == 4:
        return rng.choice(_TEXT_NUMBERS)
    if kind == 5:
        return rng.choice(_WORDS)
    if kind == 6:
        return rng.choice(_PATHS)
    return rng.choice([None, ""])


def _random_options(rng, names: list[str], accepted) -> dict:
    options = {}
    for _ in range(rng.randint(0, 8)):
        name, value = rng.choice(names), _random_value(rng)
        if accepted(name, value):
            options[name] = value
    return options


def test_commercial_pairs_equal_the_file_read_back(tmp_path: Path) -> None:
    import random

    rng = random.Random(240)
    record = tmp_path / "s.opt"
    compared = 0
    for _ in range(2000):
        solver = rng.choice(sorted(_NAMES))

        def accepted(name, value, solver=solver) -> bool:
            if value is None:
                return True
            try:
                _solver_dispatch.validate_solver_option(
                    solver, name, _subprocess_solve._format_native_opt_value(value), "test")
            except _solver_dispatch.FlexToolUserError:
                return False
            return True

        options = _random_options(rng, _NAMES[solver], accepted)
        build = _subprocess_solve._SCRIPTS[solver]
        written = _subprocess_solve._write_commercial_opt_file(options, record)
        old_params = tuple(_subprocess_solve._parse_native_opt_file(written).items()) if written else ()
        old = build(Path("bin"), Path("m.mps"), Path("m.sol"), old_params)
        new = build(Path("bin"), Path("m.mps"), Path("m.sol"), _subprocess_solve.commercial_params(options))
        assert new == old, (solver, options)
        compared += bool(options)
    assert compared > 1500


def test_highs_pairs_set_one_by_one_equal_the_option_file_read(tmp_path: Path, monkeypatch) -> None:
    import random

    monkeypatch.chdir(tmp_path)  # HiGHS opens a log_file as soon as it is set

    highspy = pytest.importorskip("highspy")
    from flextool.cli import cmd_solve_mps

    def ok(status, what) -> bool:  # cmd_solve_mps.main's rule: kWarning is taken
        return status in (highspy.HighsStatus.kOk, highspy.HighsStatus.kWarning)

    default = set(_options_dump(highspy.Highs()))

    def outcome(highs, taken: bool) -> tuple[bool, list[str] | None]:
        return taken, ([line for line in _options_dump(highs) if line not in default] if taken else None)

    rng = random.Random(2400)
    opt_path, json_path = tmp_path / "s.opt", tmp_path / "s.options.json"
    outcomes = set()
    for _ in range(2000):
        options = _random_options(rng, _HIGHS_NAMES, lambda name, value: True)
        _subprocess_solve._write_highs_option_files(_subprocess_solve._highs_option_pairs(options), opt_path,
                                                    json_path)
        old_h, new_h = highspy.Highs(), highspy.Highs()
        old = outcome(old_h, ok(old_h.readOptions(str(opt_path)), "readOptions"))
        new = outcome(new_h, cmd_solve_mps.apply_option_pairs(new_h, json_path, ok) is None)
        assert new == old, options
        outcomes.add(old[0])
    assert outcomes == {True, False}
