"""Child FlexTool runs get the parent's choices as flags and inherit its environment untouched.

The calibrator and the workflow runner each launch ``cmd_run_flextool`` as a subprocess.  Their command lines must
parse with the real ``cmd_run_flextool`` parser, and an option they choose must arrive as a flag (source ``cli``)
rather than as an edited copy of the environment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

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
    # argv = [python, "-m", "flextool.cli.cmd_run_flextool", *args]
    assert argv[1:3] == ["-m", "flextool.cli.cmd_run_flextool"]
    return RunOptions.from_cli(build_parser().parse_args(argv[3:]), environ=environ)


def test_calibrator_requests_warm_start_by_flag_and_inherits_the_environment(monkeypatch, tmp_path):
    from flextool.calibrate._solve import run_solve

    monkeypatch.delenv("FLEXTOOL_WARM_START", raising=False)
    monkeypatch.delenv("FLEXTOOL_BASIS_CACHE_DIR", raising=False)
    monkeypatch.setenv("FLEXTOOL_HIGHS_THREADS", "3")
    calls = _capture_subprocess(monkeypatch)
    cache_dir = tmp_path / "basis_cache"

    run_solve("input.sqlite", "base", work_dir=tmp_path / "work", out_root=tmp_path / "out", cache_dir=cache_dir)

    (call,) = calls
    # No environment of its own: the child inherits the calibrator's environment as the user set it.
    assert call.get("env") is None
    child = _child_options(call["argv"], environ={"FLEXTOOL_HIGHS_THREADS": "3"})
    assert child.get("warm_start").source == "cli" and child.value("warm_start") is True
    assert child.get("basis_cache_dir").source == "cli"
    assert Path(child.value("basis_cache_dir")) == cache_dir
    # An option the calibrator does not choose still comes from the inherited environment.
    assert child.get("highs_threads").source == "env" and child.value("highs_threads") == 3
    assert {k for k, r in child.resolved.items() if r.source == "cli"} == {"warm_start", "basis_cache_dir"}


def test_workflow_child_command_parses_and_sets_no_solver_option():
    args = workflow.build_parser().parse_args(
        ["in.sqlite", "out.sqlite", "base", "--write-methods", "parquet", "csv", "--debug"]
    )
    child_args = workflow.run_flextool_args(args, output_config="plots.yaml", output_subdir="base")

    parsed = build_parser().parse_args(child_args)
    assert (parsed.input_db_url, parsed.output_db_url, parsed.scenario_name) == ("in.sqlite", "out.sqlite", "base")
    assert parsed.write_methods == ["parquet", "csv"]
    assert parsed.debug == "basic"
    child = RunOptions.from_cli(parsed, environ={})
    assert all(child.source(key) == "default" for key in OPTIONS)
