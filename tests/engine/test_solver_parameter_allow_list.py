"""Solver parameter names from the input database must be in FlexTool's list (item 240 B1).

A solve's ``solver_arguments`` may name only parameters of FlexTool's list of the solve's solver
(``solver_catalogue/<solver>.json``), matched as the solver reads names.  Checked here, with values set by hand:

* an unlisted name is refused with the closest listed one; a CPLEX abbreviation with its full name (D2);
* HiGHS names the installed HiGHS knows as a number or a switch pass, a text one does not (D1);
* this computer may allow unlisted names (environment variable, else ``solver_settings.json``; D4), with one
  warning per run; paths and servers stay refused;
* an unlisted name in the run's own option file warns and passes (D3);
* FlexTool 3's ``solver`` / ``parallel`` / ``presolve`` are left out of a commercial solve with one warning;
* every solve is checked before the first one (``check_run_solver_options``); the run record names the switch.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from flextool.engine.input._solve_config import SolverConfig
from flextool.engine.solve import _solver_catalogue, _solver_dispatch
from flextool.engine.solve._solver_dispatch import (
    FlexToolUserError,
    _resolve_effective_highs_options,
    check_run_solver_options,
    reset_run_warnings,
    resolve_commercial_options,
)
from flextool.run_options import RunOptions

REPO = Path(__file__).resolve().parents[2]
_NO_FLAGS = RunOptions.from_env({})
ALLOW = "FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """No allow switch from the environment, an empty user config folder, fresh once-per-run warnings."""
    monkeypatch.delenv(ALLOW, raising=False)
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(tmp_path / "user"))
    reset_run_warnings()
    yield
    reset_run_warnings()


def _db(solver: str, options: dict, *, solve: str = "invest", opt_text: str | None = None):
    """Resolve *options* as the ``solver_arguments`` of *solve*: ``{name: value}`` as the solver gets them."""
    if solver == "highs":
        return _resolve_effective_highs_options(
            solver_arguments_map=options, highs_opt_path=Path("cfg/highs.opt") if opt_text is not None else None,
            highs_opt_text=opt_text, cli_overrides=None, solve_name=solve)
    resolved = resolve_commercial_options(
        solver, opt_file="cfg/x.opt" if opt_text is not None else None, opt_text=opt_text,
        solver_config=SolverConfig(name=solver, options=options), run_options=_NO_FLAGS, solve_name=solve)
    return {o.name: o.value for o in resolved}


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _settings_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "user" / "config" / "solver_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ---------------------------------------------------------------------------
# Listed names pass, unlisted ones are refused with the closest name
# ---------------------------------------------------------------------------


def test_misspelt_name_is_refused_with_the_closest_name() -> None:
    with pytest.raises(FlexToolUserError) as err:
        _db("gurobi", {"MIPGapp": 0.01})
    message = str(err.value)
    assert "Refusing gurobi option 'MIPGapp' (value '0.01') from solver_arguments of solve invest" in message
    assert "not in FlexTool's list of Gurobi parameters (made from Gurobi 13.0.2)" in message
    assert "did you mean 'MIPGap'?" in message
    assert ALLOW in message and "solver_config/gurobi.opt" in message and "solver_settings.json" in message


@pytest.mark.parametrize("solver, name", [
    ("gurobi", "Mip_Gap"), ("gurobi", "MIPGAP"), ("xpress", "XPRS_MAXTIME"), ("xpress", "maxtime"),
    ("cplex", "mip.tolerances.mipgap"), ("cplex", "MIP Tolerances MIPGap"), ("copt", "relgap"),
    ("highs", "mip_rel_gap"),
])
def test_spellings_the_solver_reads_as_a_listed_name_pass(solver, name) -> None:
    assert list(_db(solver, {name: "0.01"}).values()) == ["0.01"]


def test_cplex_abbreviation_is_refused_with_its_full_name() -> None:
    with pytest.raises(FlexToolUserError, match="not a full CPLEX parameter name; write 'mip tolerances mipgap'"):
        _db("cplex", {"mip tol mipgap": "0.01"})
    # ``mip tol`` abbreviates no listed name of two words; ``mip limits`` + ``s`` abbreviates two.
    with pytest.raises(FlexToolUserError, match="abbreviates 'mip limits solutions', 'mip limits strongcand', "
                                                "'mip limits strongit'"):
        _db("cplex", {"mip limits s": "1"})


def test_highs_name_in_another_case_is_refused_naming_the_listed_one() -> None:
    """HiGHS takes its option names in lower case only."""
    with pytest.raises(FlexToolUserError, match="did you mean 'presolve'"):
        _db("highs", {"Presolve": "off"})


@pytest.mark.parametrize("solver, name, value, words", [
    ("xpress", "COMPUTEEXECSERVICE", "svc1", "another computer"),
    ("gurobi", "JobID", "job1", "not checked for use from an input database"),
    ("cplex", "defaults", "yes", "a command of the CPLEX console"),
    ("cplex", "distmip config", "mpi", "another computer"),
])
def test_listed_local_only_names_are_refused(solver, name, value, words) -> None:
    with pytest.raises(FlexToolUserError, match=words):
        _db(solver, {name: value})


@pytest.mark.parametrize("solver, name, value", [
    ("cplex", "read fileencoding", "/tmp/evil"), ("cplex", "cpumask", "C:x"), ("xpress", "MPSRHSNAME", r"..\x"),
])
def test_a_text_parameter_from_the_database_takes_no_path_characters(solver, name, value) -> None:
    """Review b1 F4: a listed ``plain`` text parameter takes a word (``UTF-8``, ``ffff``, ``RHS1``) from the
    database, never a value with ``/``, ``\\`` or ``:``."""
    with pytest.raises(FlexToolUserError, match="a text parameter takes no"):
        _db(solver, {name: value})
    assert _db(solver, {name: "RHS1"}) == {name: "RHS1"}


# ---------------------------------------------------------------------------
# HiGHS names the installed HiGHS knows (D1)
# ---------------------------------------------------------------------------


def _without(solver: str, key: str, load=_solver_catalogue.load):
    """*solver*'s catalogue without *key* (as an older FlexTool list would be)."""
    full = load(solver)
    entries = {k: e for k, e in full.entries.items() if k != key}
    return _solver_catalogue.Catalogue(solver, full.versions, entries)


def test_highs_number_the_installed_highs_knows_passes_a_text_one_does_not(monkeypatch) -> None:
    pytest.importorskip("highspy")
    real = _solver_catalogue.load
    for key in ("mip_heuristic_effort", "log_file"):
        monkeypatch.setattr(_solver_catalogue, "load", lambda s, key=key: _without(s, key) if s == "highs" else real(s))
        if key == "mip_heuristic_effort":  # a double in highspy 1.15.1
            assert _db("highs", {key: 0.3}) == {key: 0.3}
            continue
        for allow in ("0", "1"):  # a text option nobody reviewed is refused whatever the switch
            monkeypatch.setenv(ALLOW, allow)
            with pytest.raises(FlexToolUserError, match="text option of the installed HiGHS"):
                _db("highs", {key: "run.log"})


# ---------------------------------------------------------------------------
# This computer's switch (D4)
# ---------------------------------------------------------------------------


def test_switch_from_the_environment_passes_an_unlisted_name_with_one_warning(monkeypatch, caplog) -> None:
    monkeypatch.setenv(ALLOW, "1")
    with caplog.at_level(logging.WARNING):
        for solve in ("invest", "dispatch"):
            assert _db("gurobi", {"MIPGapp": "0.01"}, solve=solve) == {"MIPGapp": "0.01"}
    allowed = [w for w in _warnings(caplog) if w.startswith("Passing gurobi option 'MIPGapp'")]
    assert len(allowed) == 1 and f"environment variable {ALLOW}" in allowed[0], _warnings(caplog)


def test_switch_from_the_settings_file_and_the_environment_wins(monkeypatch, tmp_path, caplog) -> None:
    path = _settings_file(tmp_path, '{"format": 1, "allow_unlisted_parameters": true}')
    with caplog.at_level(logging.WARNING):
        assert _db("copt", {"RelGapp": "0.01"}) == {"RelGapp": "0.01"}
    assert any(str(path) in w for w in _warnings(caplog))
    monkeypatch.setenv(ALLOW, "0")
    with pytest.raises(FlexToolUserError, match="not in FlexTool's list of COPT parameters"):
        _db("copt", {"RelGapp": "0.01"})


def test_switch_keeps_servers_and_paths_refused(monkeypatch) -> None:
    monkeypatch.setenv(ALLOW, "1")
    with pytest.raises(FlexToolUserError, match="another computer"):
        _db("gurobi", {"CSFoo": "host1"})  # unlisted, server prefix
    with pytest.raises(FlexToolUserError, match="another computer"):
        _db("gurobi", {"ComputeServer": "host1"})  # listed
    with pytest.raises(FlexToolUserError, match="plain file name"):
        _db("cplex", {"write": "/tmp/x"})  # unlisted, CPLEX write command
    with pytest.raises(FlexToolUserError, match="plain file name"):
        _db("gurobi", {"NewLogFile": "/tmp/x.log"})  # unlisted, file in the name


@pytest.mark.parametrize("text, words", [
    ("{not json", "is not valid JSON"),
    ('{"format": 2, "allow_unlisted_parameters": true}', "written by a newer FlexTool"),
    ('{"allow_unlisted_parameters": "yes"}', "write true or false"),
])
def test_unreadable_settings_file_stops_only_a_run_that_needs_it(tmp_path, text, words) -> None:
    path = _settings_file(tmp_path, text)
    assert _db("gurobi", {"MIPGap": "0.01"}) == {"MIPGap": "0.01"}  # listed: the file is not read
    with pytest.raises(FlexToolUserError, match=words) as err:
        _db("gurobi", {"MIPGapp": "0.01"})
    assert str(path) in str(err.value)


def test_environment_value_other_than_one_or_zero_is_an_error(monkeypatch) -> None:
    monkeypatch.setenv(ALLOW, "maybe")
    with pytest.raises(FlexToolUserError, match=f"{ALLOW}='maybe' is not 1 or 0"):
        _db("gurobi", {"MIPGapp": "0.01"})


def test_run_record_names_the_switch_and_its_source(monkeypatch, tmp_path) -> None:
    from flextool._local_solver_settings import allow_unlisted_record

    assert allow_unlisted_record() == {"value": False, "source": "default"}
    path = _settings_file(tmp_path, '{"allow_unlisted_parameters": true}')
    assert allow_unlisted_record() == {"value": True, "source": str(path)}
    monkeypatch.setenv(ALLOW, "0")
    assert allow_unlisted_record() == {"value": False, "source": f"environment variable {ALLOW}"}
    _settings_file(tmp_path, "{not json")
    monkeypatch.delenv(ALLOW)
    record = allow_unlisted_record()
    assert record["value"] is None and "is not valid JSON" in record["source"]


# ---------------------------------------------------------------------------
# The run's own option files warn and pass (D3)
# ---------------------------------------------------------------------------


def test_unlisted_name_in_the_option_file_warns_once_and_passes(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        for solve in ("invest", "dispatch"):
            assert _db("gurobi", {}, solve=solve, opt_text="Threads 2\nMIPGapp 0.01\n") == {
                "Threads": "2", "MIPGapp": "0.01"}
    unlisted = [w for w in _warnings(caplog) if "MIPGapp" in w]
    assert unlisted == ["cfg/x.opt, line 2: gurobi option 'MIPGapp': it is not in FlexTool's list of Gurobi "
                        "parameters (made from Gurobi 13.0.2); did you mean 'MIPGap'? Passed to Gurobi as written."]


def test_cplex_abbreviation_in_the_option_file_warns_naming_the_full_name(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert _db("cplex", {}, opt_text="mip tol mipgap 0.01\n") == {"mip tol mipgap": "0.01"}
    assert any("write 'mip tolerances mipgap'" in w and "line 1" in w for w in _warnings(caplog))


def test_highs_option_file_and_flags_warn_on_an_unknown_name(caplog) -> None:
    pytest.importorskip("highspy")
    with caplog.at_level(logging.WARNING):
        options = _resolve_effective_highs_options(
            solver_arguments_map=None, highs_opt_path=Path("cfg/highs.opt"), highs_opt_text="mip_rel_gapp=0.1\n",
            cli_overrides={"thread": 2})
    assert options == {"mip_rel_gapp": "0.1", "thread": 2}
    warned = _warnings(caplog)
    assert any(w.startswith("cfg/highs.opt, line 1: highs option 'mip_rel_gapp'") for w in warned), warned
    assert any(w.startswith("the run's flags or environment variables: highs option 'thread'") for w in warned)


# ---------------------------------------------------------------------------
# FlexTool 3's HiGHS settings in a commercial solve
# ---------------------------------------------------------------------------


def test_flextool3_highs_settings_are_left_out_of_a_commercial_solve(caplog) -> None:
    """The v56 migration wrote ``solver`` / ``parallel`` / ``presolve`` with HiGHS words for every solve.  On Gurobi
    they are left out with one warning each per run; a Gurobi ``Presolve`` number stays; HiGHS keeps them."""
    ft3 = {"solver": "ipm", "parallel": "on", "presolve": "off"}
    with caplog.at_level(logging.WARNING):
        for solve in ("invest", "dispatch"):
            assert _db("gurobi", {**ft3, "Threads": "2"}, solve=solve) == {"Threads": "2"}
    left_out = [w for w in _warnings(caplog) if w.startswith("Not passing")]
    assert len(left_out) == 3 and all("this solve uses Gurobi" in w for w in left_out), left_out
    assert _db("gurobi", {"presolve": "2"}) == {"presolve": "2"}  # a number for Gurobi's own Presolve
    assert _db("cplex", {"parallel": "1"}) == {"parallel": "1"}  # CPLEX's own parallel takes -1 / 0 / 1
    assert _db("highs", ft3) == ft3


def test_flextool3_highs_names_are_the_migrations() -> None:
    """The names the v56 migration folds FlexTool 3's highs_method / highs_parallel / highs_presolve into."""
    source = (REPO / "flextool" / "update_flextool" / "db_migration.py").read_text()
    assert set(re.findall(r'highs_key="(\w+)"', source)) == _solver_dispatch._FLEXTOOL3_HIGHS_OPTIONS


# ---------------------------------------------------------------------------
# Every solve before the first one
# ---------------------------------------------------------------------------


def test_every_solve_is_checked_before_the_first(caplog) -> None:
    settings = SimpleNamespace(
        solvers={"invest": "gurobi", "dispatch": "cplex"},
        arguments={"invest": {"MIPGap": "0.01", "parallel": "on"}, "dispatch": {"mip tol mipgap": "0.01"},
                   "highs_solve": {"mip_rel_gap": "0.01"}},
    )
    with pytest.raises(FlexToolUserError, match="solver_arguments of solve dispatch: it is not a full CPLEX"):
        check_run_solver_options(settings)
    settings.arguments["dispatch"] = {"mip tolerances mipgap": "0.01"}
    with caplog.at_level(logging.WARNING):
        check_run_solver_options(settings)
    assert [w for w in _warnings(caplog) if w.startswith("Not passing parallel")]
    settings.arguments["highs_solve"] = {"MIPGap": "0.01"}  # a solve without a solver parameter is HiGHS
    with pytest.raises(FlexToolUserError, match="highs option 'MIPGap'"):
        check_run_solver_options(settings)
    # Review b1 F7: only the solves the run runs; a 2-D period map's copies are named (and checked) as theirs.
    check_run_solver_options(settings, solves=["invest", "dispatch"])
    settings.solvers.update({"highs_solve_p1": "highs", "highs_solve_p2": "highs"})
    settings.arguments.update({"highs_solve_p1": {"MIPGap": "1"}, "highs_solve_p2": {"MIPGap": "1"}})
    with pytest.raises(FlexToolUserError, match="from solver_arguments of solve highs_solve: "):
        check_run_solver_options(settings, solves=["invest", "highs_solve_p1", "highs_solve_p2"],
                                 shown_as={"highs_solve_p1": "highs_solve", "highs_solve_p2": "highs_solve"})


def test_highs_options_of_a_commercial_solve_leave_out_its_arguments() -> None:
    """Review b1 F8 (question 3): the HiGHS option set of a Gurobi solve (the warm-LP key, the echo line, Benders)
    does not read its ``solver_arguments``: ``MIPGap`` is not handed to the HiGHS resolver, so it is neither in
    the result nor refused as an unlisted HiGHS name."""
    from flextool.engine.input._solve_config import SolverSettings
    from flextool.engine.solve._orchestration import _solve_highs_options

    gurobi = SolverConfig(name="gurobi", options={"MIPGap": "0.01"})
    solve = SimpleNamespace(
        solver_config_for=lambda name: gurobi,
        solver_settings=SolverSettings(solvers={"invest": "gurobi"}, precommand={},
                                       arguments={"invest": {"MIPGap": "0.01"}}),
    )
    state = SimpleNamespace(solve=solve, run_options=_NO_FLAGS, paths=SimpleNamespace(
        solver_option_files=None, solver_config_dir=None, work_folder=None))
    options = _solve_highs_options(state, "invest", baseline=None)
    assert not {k for k in options if k.casefold() == "mipgap"}, options


def test_ft3_highs_settings_of_a_migrated_database_are_left_out_of_a_gurobi_solve(tmp_path, caplog) -> None:
    """Review b1 F8 (critique P4): FlexTool 3's ``highs_method`` / ``highs_parallel`` / ``highs_presolve`` on a
    Gurobi solve, folded into ``solver_arguments`` (as ``solver`` / ``parallel`` / ``presolve``) by the v56
    migration's own steps: the run's upfront check passes with one warning per name, and the solve's Gurobi
    options leave the three out.  No fixture is older than v56 (they are re-exported at the current version), so
    the FlexTool 3 rows are planted in a current database and the v56 steps are run on it."""
    from spinedb_api import DatabaseMapping, import_data
    from spinedb_api.filters.scenario_filter import apply_scenario_filter_to_subqueries

    from flextool.engine.input._solve_config import SolveConfig
    from flextool.update_flextool import db_migration
    from tests.db_utils import json_to_db

    url = json_to_db(REPO / "tests" / "fixtures" / "stochastics.json", tmp_path / "ft3.sqlite")
    solve, scenario, alternative = "1week_rolling", "1_week_rolling_wind", "1week_rolling"
    with DatabaseMapping(url) as db:
        _count, errors = import_data(db, parameter_definitions=[
            ("solve", "highs_method"), ("solve", "highs_parallel"), ("solve", "highs_presolve")], parameter_values=[
            ("solve", solve, "solver", "gurobi", alternative),
            ("solve", solve, "highs_method", "simplex", alternative),
            ("solve", solve, "highs_parallel", "on", alternative),
            ("solve", solve, "highs_presolve", "on", alternative),
        ])
        assert not errors, errors
        db.commit_session("FlexTool 3 HiGHS settings on a Gurobi solve")
        for step in ("method", "parallel", "presolve"):
            getattr(db_migration, f"_migrate_v56_fold_highs_{step}_into_solver_arguments")(db)
            getattr(db_migration, f"_migrate_v56_remove_highs_{step}")(db)
    with DatabaseMapping(url) as db:
        apply_scenario_filter_to_subqueries(db, scenario)
        sc = SolveConfig.load_from_db(db, logging.getLogger("test.ft3"))
    assert sc.solver_settings.arguments[solve] == {"solver": "simplex", "parallel": "on", "presolve": "on"}
    with caplog.at_level(logging.WARNING):
        check_run_solver_options(sc.solver_settings, solves=sc.solves_in_run())
        assert _db("gurobi", sc.solver_settings.arguments[solve], solve=solve) == {}
    left_out = [w for w in _warnings(caplog) if w.startswith("Not passing")]
    assert sorted(w.split("=", 1)[0] for w in left_out) == [
        "Not passing parallel", "Not passing presolve", "Not passing solver"], left_out


def test_shipped_databases_set_only_listed_names() -> None:
    """The JSON sources of the shipped examples, templates and test databases (CLAUDE.md invariant 3): every
    ``solver_arguments`` value names a listed parameter of its solve's solver (as of item 240 they set none, so
    this guards the fixtures that gain values)."""
    sources = [*(REPO / "tests" / "fixtures").glob("*.json"),
               *(REPO / "flextool" / "schemas" / "canonical_databases").glob("*.json")]
    assert sources
    for path in sources:
        data = json.loads(path.read_text())
        values = data.get("parameter_values", []) if isinstance(data, dict) else []
        solvers = {v[1]: v[3] for v in values if v[0] == "solve" and v[2] == "solver" and isinstance(v[3], str)}
        for v in values:
            if v[0] == "solve" and v[2] == "solver_arguments":
                settings = SimpleNamespace(solvers=solvers, arguments={v[1]: _map_of(v[3])})
                check_run_solver_options(settings)


def _map_of(value) -> dict:
    import base64

    raw = json.loads(base64.b64decode(value[0])) if isinstance(value, list) else value
    return {str(k): str(x) for k, x in raw.get("data", [])}


# ---------------------------------------------------------------------------
# The upfront check in a run (design test 9; review b1 F7 / F8)
# ---------------------------------------------------------------------------

_CHECK_SOLVE = "y2020_2day_dispatch"
_OTHER_SOLVE = "y2020_fullYear_dispatch"


@pytest.fixture(scope="module")
def two_solve_db(tmp_path_factory) -> str:
    """``tests/fixtures/tests.json`` with two scenarios on the ``coal`` LP chain (``init``, ``west``, ``coal``):

    * ``second_refused``: ``model.solves`` = [``y2020_2day_dispatch`` on Gurobi, ``y2020_fullYear_dispatch``],
      the second with ``solver_arguments`` ``MIPGap`` (not a HiGHS name);
    * ``unused_refused``: the model runs ``y2020_2day_dispatch`` only; ``y2020_fullYear_dispatch``, which it does
      not run, has the same ``MIPGap``."""
    import importlib.util
    import sys

    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    name = "_lp_snapshot_for_tests"  # the fixture overlay helpers, loaded as make_check_db.py loads them
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    lps = sys.modules[name]
    bad = ["solve", _OTHER_SOLVE, "solver_arguments", {"index_type": "str", "index_name": "x", "data": [["MIPGap", 0.01]]}]
    data = json.loads((REPO / "tests" / "fixtures" / "tests.json").read_text())
    base = ["init", "west", "coal"]
    data = lps._overlay(data, name="second_refused", base=base, values=[
        ["model", "flexTool", "solves", {"type": "array", "value_type": "str", "data": [_CHECK_SOLVE, _OTHER_SOLVE]}],
        ["solve", _CHECK_SOLVE, "solver", "gurobi"], bad])
    data = lps._overlay(data, name="unused_refused", base=base, values=[bad])
    folder = tmp_path_factory.mktemp("upfront_check")
    json_path = folder / "db.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, folder / "db.sqlite")
    migrate_database(url)
    return url


def test_a_refused_option_of_a_later_solve_stops_the_run_before_the_first_solve(two_solve_db, tmp_path) -> None:
    """Design test 9: the second solve's ``MIPGap`` is refused before the first (Gurobi) solve writes its model."""
    from flextool.engine import run_chain_from_db

    with pytest.raises(FlexToolUserError, match=f"highs option 'MIPGap'.*solver_arguments of solve {_OTHER_SOLVE}"):
        run_chain_from_db(input_db_url=two_solve_db, scenario_name="second_refused", work_folder=tmp_path / "w")
    assert not list((tmp_path / "w").rglob("*.mps"))


def test_an_option_of_a_solve_the_model_does_not_run_does_not_stop_it(two_solve_db, tmp_path) -> None:
    """Review b1 F7: ``y2020_fullYear_dispatch`` is not in ``model.solves``, so its options never reach a solver and
    are not checked; the run solves ``y2020_2day_dispatch``."""
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(input_db_url=two_solve_db, scenario_name="unused_refused", work_folder=tmp_path / "w")
    assert list(steps) == [_CHECK_SOLVE] and steps[_CHECK_SOLVE].optimal
