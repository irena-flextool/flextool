"""Child FlexTool runs get the parent's choices as flags and inherit its environment untouched.

The calibrator and the workflow runner each launch an engine run as a subprocess, through the ``flextool-run``
supervisor (``python -m flextool_run_supervisor``), which passes its arguments to ``cmd_run_flextool`` unchanged.
Their command lines must parse with the real ``cmd_run_flextool`` parser, an option they choose must arrive as a
flag (source ``cli``) rather than as an edited copy of the environment, and each names itself as the run's
launcher (``--run-launcher``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from flextool._child_python import child_env
from flextool.cli import cmd_execute_flextool_workflow as workflow
from flextool.cli.cmd_run_flextool import build_parser
from flextool.run_options import OPTIONS, RunOptions


def _capture_subprocess(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    def fake_run(argv, **kwargs):
        calls.append({"argv": list(argv), **kwargs})
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr=None)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _child_options(argv: list[str], environ: dict[str, str]) -> RunOptions:
    # argv = [python, "-P", "-m", "flextool_run_supervisor", *args]; the supervisor passes args to the engine
    # unchanged.
    assert argv[1:4] == ["-P", "-m", "flextool_run_supervisor"]
    return RunOptions.from_cli(build_parser().parse_args(argv[4:]), environ=environ)


def test_calibrator_requests_warm_start_by_flag_and_inherits_the_environment(monkeypatch, tmp_path):
    from flextool.calibrate._solve import run_solve

    monkeypatch.delenv("FLEXTOOL_WARM_START", raising=False)
    monkeypatch.delenv("FLEXTOOL_BASIS_CACHE_DIR", raising=False)
    monkeypatch.setenv("FLEXTOOL_HIGHS_THREADS", "3")
    calls = _capture_subprocess(monkeypatch)
    cache_dir = tmp_path / "basis_cache"

    run_solve("input.sqlite", "base", work_dir=tmp_path / "work", out_root=tmp_path / "out", cache_dir=cache_dir)

    (call,) = calls
    # No environment of its own: the child inherits the calibrator's environment as the user set it, plus only
    # PYTHONSAFEPATH (the current folder stays off its sys.path).
    assert call.get("env") == child_env()
    child = _child_options(call["argv"], environ={"FLEXTOOL_HIGHS_THREADS": "3"})
    assert child.get("warm_start").source == "cli" and child.value("warm_start") is True
    assert child.get("basis_cache_dir").source == "cli"
    assert Path(child.value("basis_cache_dir")) == cache_dir
    # An option the calibrator does not choose still comes from the inherited environment.
    assert child.get("highs_threads").source == "env" and child.value("highs_threads") == 3
    assert {k for k, r in child.resolved.items() if r.source == "cli"} == {"warm_start", "basis_cache_dir"}
    assert build_parser().parse_args(call["argv"][4:]).run_launcher == "calibrator"


def test_workflow_child_command_parses_and_sets_no_solver_option():
    args = workflow.build_parser().parse_args(
        ["in.sqlite", "out.sqlite", "base", "--write-methods", "parquet", "csv", "--debug"]
    )
    child_args = workflow.run_flextool_args(args, output_config="plots.yaml", output_subdir="base")

    parsed = build_parser().parse_args(child_args)
    assert (parsed.input_db_url, parsed.output_db_url, parsed.scenario_name) == ("in.sqlite", "out.sqlite", "base")
    assert parsed.write_methods == ["parquet", "csv"]
    assert parsed.debug == "basic"
    assert parsed.run_launcher == "workflow"
    child = RunOptions.from_cli(parsed, environ={})
    assert all(child.source(key) == "default" for key in OPTIONS)


def test_workflow_forwards_the_thread_budget_as_a_flag():
    args = workflow.build_parser().parse_args(["in.sqlite", "out.sqlite", "base", "--threads", "3"])
    child_args = workflow.run_flextool_args(args, output_config="plots.yaml", output_subdir="base")
    child = RunOptions.from_cli(build_parser().parse_args(child_args), environ={"FLEXTOOL_THREADS": "7"})
    assert (child.value("threads"), child.source("threads")) == (3, "cli")


def test_calibrator_forwards_the_thread_budget_as_a_flag(monkeypatch, tmp_path):
    from flextool.calibrate._cli import _config_from_args, build_parser as calibrate_parser
    from flextool.calibrate._solve import run_solve

    config = _config_from_args(calibrate_parser().parse_args(
        ["input.sqlite", "base", "--iterations", "1", "--threads", "4", "--work-dir", str(tmp_path / "w"),
         "--output-location", str(tmp_path / "o")]))
    assert config.threads == 4
    calls = _capture_subprocess(monkeypatch)
    run_solve("input.sqlite", "base", work_dir=tmp_path / "work", out_root=tmp_path / "out",
              cache_dir=tmp_path / "bc", threads=config.threads)
    (call,) = calls
    child = _child_options(call["argv"], environ={})
    assert (child.value("threads"), child.source("threads")) == (4, "cli")


def test_workflow_runs_the_engine_under_the_supervisor(monkeypatch, tmp_path):
    """Phase 2 of the workflow starts ``python -P -m flextool_run_supervisor`` with no environment of its own
    (only ``PYTHONSAFEPATH`` added)."""
    calls = _capture_subprocess(monkeypatch)
    monkeypatch.setattr(workflow, "ensure_settings_db", lambda *a, **k: None)
    db = tmp_path / "in.sqlite"
    db.touch()
    monkeypatch.setattr("sys.argv", ["cmd_execute_flextool_workflow", f"sqlite:///{db}", "out.sqlite", "base",
                                     "--write-methods", "parquet"])
    workflow.main()
    runs = [c for c in calls if c["argv"][1:4] == ["-P", "-m", "flextool_run_supervisor"]]
    assert len(runs) == 1
    assert runs[0].get("env") == child_env()
    parsed = build_parser().parse_args(runs[0]["argv"][4:])
    assert parsed.scenario_name == "base" and parsed.run_launcher == "workflow"
