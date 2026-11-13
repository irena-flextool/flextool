"""``cmd_run_flextool`` hands the run's options to the engine: an unset flag lets the environment variable through.

``main`` is driven in-process with ``run_chain_from_db`` replaced by a recorder, so nothing is solved; the recorder
reads the HiGHS operator layer at the moment the cascade would start.
"""
from __future__ import annotations

import os
import sys

import pytest

import flextool.engine
from flextool._solver_config_source import UNTRUSTED_SHORT
from flextool.cli import cmd_run_flextool
from flextool.engine.solve._orchestration import _operator_highs_overrides

_OPTION_ENV_VARS = (
    "FLEXTOOL_HIGHS_THREADS", "FLEXTOOL_HIGHS_PRESOLVE", "FLEXTOOL_SOLVER_LOG_LEVEL", "FLEXTOOL_HIGHS_TIME_LIMIT",
    "FLEXTOOL_HIGHS_MIP_GAP", "FLEXTOOL_SOLVER_TIME_LIMIT", "FLEXTOOL_SOLVER_MIP_GAP", "FLEXTOOL_SCALING", "FLEXTOOL_USER_BOUND_SCALE",
    "FLEXTOOL_SAVE_MEMORY", "FLEXTOOL_WARM_START", "FLEXTOOL_BASIS_CACHE_DIR", "FLEXTOOL_BENDERS_WORKERS",
    "FLEXTOOL_THREADS", "FLEXTOOL_SOLVER_CONFIG_DIR",
)


@pytest.fixture
def clean_env():
    """No option variables set at the start; the whole environment restored afterwards (``main`` writes it)."""
    saved = dict(os.environ)
    for name in _OPTION_ENV_VARS:
        os.environ.pop(name, None)
    yield os.environ
    os.environ.clear()
    os.environ.update(saved)


def _run_main(monkeypatch, tmp_path, *flags):
    """Run ``cmd_run_flextool.main`` with *flags*; return what the cascade entry point received."""
    seen: dict = {}

    def recorder(input_db_url, scenario_name, **kwargs):
        seen["kwargs"] = kwargs
        seen["highs_overrides"] = _operator_highs_overrides(kwargs["run_options"])
        return {}

    monkeypatch.setattr(flextool.engine, "run_chain_from_db", recorder)
    db = tmp_path / "input.sqlite"
    db.write_bytes(b"")
    argv = [
        "cmd_run_flextool", f"sqlite:///{db}", "--scenario-name", "base",
        "--work-folder", str(tmp_path / "work"), "--output-location", str(tmp_path / "out"), *flags,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    # The recorder runs no solve, and a run without solve steps exits 1.
    with pytest.raises(SystemExit) as exc:
        cmd_run_flextool.main()
    assert exc.value.code == 1
    return seen


def test_env_threads_reach_highs_when_the_flag_is_not_given(clean_env, monkeypatch, tmp_path):
    clean_env["FLEXTOOL_HIGHS_THREADS"] = "4"
    seen = _run_main(monkeypatch, tmp_path)
    assert seen["highs_overrides"].get("threads") == 4
    assert seen["highs_overrides"].get("parallel") == "on"


def test_threads_flag_wins_over_env(clean_env, monkeypatch, tmp_path):
    clean_env["FLEXTOOL_HIGHS_THREADS"] = "4"
    seen = _run_main(monkeypatch, tmp_path, "--highs-threads", "1")
    assert "threads" not in seen["highs_overrides"]
    assert "parallel" not in seen["highs_overrides"]
    opts = seen["kwargs"]["run_options"]
    assert (opts.value("highs_threads"), opts.source("highs_threads")) == (1, "cli")


def test_unset_flags_leave_option_env_vars_unset(clean_env, monkeypatch, tmp_path):
    seen = _run_main(monkeypatch, tmp_path)
    assert [name for name in _OPTION_ENV_VARS if name in os.environ] == []
    assert seen["highs_overrides"] == {}
    opts = seen["kwargs"]["run_options"]
    assert opts.summary_lines() == []


def test_run_options_reach_the_engine_with_their_sources(clean_env, monkeypatch, tmp_path, capsys):
    clean_env["FLEXTOOL_HIGHS_PRESOLVE"] = "off"
    seen = _run_main(monkeypatch, tmp_path, "--scaling", "basic", "--save-memory")
    opts = seen["kwargs"]["run_options"]
    assert (opts.value("scaling"), opts.source("scaling")) == ("basic", "cli")
    assert (opts.value("save_memory"), opts.source("save_memory")) == (True, "cli")
    assert (opts.value("presolve"), opts.source("presolve")) == ("off", "env")
    header = capsys.readouterr().out
    assert "presolve=off (env FLEXTOOL_HIGHS_PRESOLVE)" in header
    assert "scaling=basic (flag)" in header
    assert "Threads:" in header


def test_given_flags_are_not_written_into_the_environment(clean_env, monkeypatch, tmp_path):
    seen = _run_main(monkeypatch, tmp_path, "--scaling", "basic", "--save-memory", "--warm-start",
                     "--highs-threads", "3", "--benders-workers", "2", "--basis-cache-dir", str(tmp_path / "bc"),
                     "--presolve", "off", "--user-bound-scale", "-3")
    assert [name for name in _OPTION_ENV_VARS if name in os.environ] == []
    opts = seen["kwargs"]["run_options"]
    assert {key: opts.value(key) for key in ("scaling", "save_memory", "warm_start", "highs_threads",
                                              "benders_workers", "presolve", "user_bound_scale")} == {
        "scaling": "basic", "save_memory": True, "warm_start": True, "highs_threads": 3, "benders_workers": 2,
        "presolve": "off", "user_bound_scale": -3}
    assert seen["highs_overrides"] == {"threads": 3, "parallel": "on", "presolve": "off"}


def test_threads_flag_reaches_highs_and_the_header(clean_env, monkeypatch, tmp_path, capsys):
    """``--threads 2``: HiGHS gets ``threads=2`` without its parallel simplex, the run header names the budget, and
    the flag is not written into the environment."""
    seen = _run_main(monkeypatch, tmp_path, "--threads", "2")
    opts = seen["kwargs"]["run_options"]
    assert (opts.value("threads"), opts.source("threads")) == (2, "cli")
    assert seen["highs_overrides"] == {"threads": 2}
    assert "FLEXTOOL_THREADS" not in os.environ
    header = capsys.readouterr().out
    line = next(ln for ln in header.splitlines() if ln.startswith("Threads:"))
    assert "2 (flag)" in line and "HiGHS 2" in line


def test_solver_config_folder_is_resolved_once_and_shown_in_the_header(clean_env, monkeypatch, tmp_path, capsys):
    """No ``--solver-config-dir``: the ``solver_config`` next to the input database wins over the current folder;
    the header names it with its reason and the engine gets the same resolved source.  The flag then wins."""
    (tmp_path / "solver_config").mkdir()
    (tmp_path / "solver_config" / "highs.opt").write_text("threads=1\n")
    cwd = tmp_path / "elsewhere"
    (cwd / "solver_config").mkdir(parents=True)
    (cwd / "solver_config" / "highs.opt").write_text("threads=2\n")
    monkeypatch.chdir(cwd)
    seen = _run_main(monkeypatch, tmp_path)
    source = seen["kwargs"]["solver_config_source"]
    assert (source.dir, source.reason) == (tmp_path / "solver_config", "next to the input database")
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Solver config:"))
    # Found on its own, outside a workspace this computer set up: its files get the database's rules (item 240 B3).
    assert line.endswith(f"{tmp_path / 'solver_config'} (next to the input database; {UNTRUSTED_SHORT})")
    chosen = tmp_path / "chosen"
    seen = _run_main(monkeypatch, tmp_path, "--solver-config-dir", str(chosen))
    source = seen["kwargs"]["solver_config_source"]
    assert (source.dir, source.reason) == (chosen, "--solver-config-dir")


def test_header_names_each_file_of_a_several_folder_ladder(clean_env, monkeypatch, tmp_path, capsys):
    """Two ``--solver-config-dir`` folders: the header lists, per folder in ladder order, the files it gives,
    then the solvers on FlexTool's defaults (hand-written expected line)."""
    upper, lower = tmp_path / "upper", tmp_path / "lower"
    upper.mkdir()
    lower.mkdir()
    (upper / "gurobi.opt").write_text("Threads 2\n")
    (lower / "gurobi.opt").write_text("Threads 4\n")
    (lower / "highs.opt").write_text("threads=1\n")
    monkeypatch.chdir(tmp_path)
    seen = _run_main(monkeypatch, tmp_path, "--solver-config-dir", str(upper), "--solver-config-dir", str(lower))
    source = seen["kwargs"]["solver_config_source"]
    assert source.layers == ((upper, "--solver-config-dir"), (lower, "--solver-config-dir"))
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("Solver config:"))
    assert line.endswith(
        f"{upper} (--solver-config-dir): gurobi.opt; {lower} (--solver-config-dir): highs.opt; "
        "FlexTool defaults: cplex.opt, xpress.opt, copt.opt"
    ), line
