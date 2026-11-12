"""Every Python process FlexTool starts keeps the current folder off ``sys.path`` (``python -P`` and
``PYTHONSAFEPATH=1``, ``flextool._child_python``), so a ``flextool/`` folder in a project or workspace folder is never
imported instead of the installed FlexTool.

The end-to-end check uses a project folder holding a dummy ``flextool/__init__.py`` that writes a marker file: the
representative-period job of the FlexTool window and the ``flextool-run`` engine child must not run it.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import flextool_run_supervisor as supervisor
from flextool._child_python import child_env, python_module_argv
from flextool.calibrate._solve import solve_argv
from flextool.gui.calibrate_commands import build_calibrate_command, build_rp_command
from flextool.gui.platform_utils import spine_db_editor_argv

REPO = Path(__file__).resolve().parents[1]


def _shadowing_project(tmp_path: Path) -> tuple[Path, Path]:
    """A project folder with a ``flextool`` package that writes ``marker`` when imported."""
    project = tmp_path / "Shared project"
    (project / "flextool").mkdir(parents=True)
    marker = tmp_path / "marker"
    (project / "flextool" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('ran')\n")
    return project, marker


def test_the_dummy_package_would_run_without_the_safe_path(tmp_path: Path) -> None:
    """The control: a plain ``python -m`` in that folder imports the dummy (so the checks below mean something)."""
    project, marker = _shadowing_project(tmp_path)
    env = {k: v for k, v in child_env().items() if k != "PYTHONSAFEPATH"}
    subprocess.run([sys.executable, "-m", "flextool.representative_periods.preprocess", "--help"], cwd=project,
                   env=env, capture_output=True, timeout=120)
    assert marker.exists()


def test_building_representative_periods_does_not_run_code_from_the_project(tmp_path: Path, monkeypatch) -> None:
    """The window's RP job, even when started with the project folder as the current folder."""
    from flextool.gui.calibrate_jobs import RpJobSpec, _SerialRunner, launch_rp_jobs
    from flextool.gui.data_models import ProjectSettings
    from flextool.gui.execution_manager import ExecutionManager

    project, marker = _shadowing_project(tmp_path)
    monkeypatch.chdir(project)
    mgr = ExecutionManager(project_path=project, settings=ProjectSettings())
    runner = _SerialRunner("test")
    spec = RpJobSpec(db_url=f"sqlite:///{tmp_path / 'missing.sqlite'}", scenario="base", n_rp=2, period_length=24,
                     force_sustained=False, force_peak=False, force_window=None, solves=(), alternative_name=None)
    launch_rp_jobs(mgr, python_exe=sys.executable, project_path=project, jobs=[spec], runner=runner)
    runner.wait(timeout=180)
    (job,) = mgr.get_jobs()
    assert job.stdout_lines[0].split()[1:3] == ["-P", "-m"]
    assert not marker.exists(), "code from the project folder ran"


def test_the_flextool_run_engine_child_does_not_run_code_from_the_folder(tmp_path: Path) -> None:
    """``flextool-run`` started from inside a shared project folder."""
    project, marker = _shadowing_project(tmp_path)
    cmd = [*supervisor.engine_command(), "--help"]
    done = subprocess.run(cmd, cwd=project, env=supervisor.child_environment(), capture_output=True, timeout=180)
    assert done.returncode == 0, done.stdout.decode(errors="replace")[-2000:]
    assert not marker.exists(), "code from the project folder ran"


def test_every_launcher_starts_python_with_the_safe_path(tmp_path: Path) -> None:
    exe = sys.executable
    argvs = {
        "helper": python_module_argv("x"),
        "rp": build_rp_command(exe, "sqlite:///m.sqlite", "base", n_rp=2, period_length=24, force_sustained=False,
                               force_peak=False, force_window=None, solves=[], alternative_name=None),
        "calibrate": build_calibrate_command(
            exe, "sqlite:///m.sqlite", "base", iterations=1, sizing="", overshoot=1.0, damping_first=1.0,
            damping_remaining=1.0, stall_fraction=0.0, warm_start_cache_dir=None, work_dir=None,
            output_location=None, debug=False),
        "calibrator solve": solve_argv("m.sqlite", "base", work_dir=tmp_path, out_root=tmp_path, cache_dir=tmp_path),
        "db editor": spine_db_editor_argv("sqlite:///m.sqlite"),
        "flextool-run": supervisor.engine_command(),
    }
    for what, argv in argvs.items():
        assert argv[:3] == [exe, "-P", "-m"], what
    assert child_env({})["PYTHONSAFEPATH"] == "1"
    assert supervisor.child_environment({})["PYTHONSAFEPATH"] == "1"


# A list or call that names the interpreter and then "-m" / "-c" right away (no "-P" between them).
_UNSAFE = re.compile(r"""(?:sys\.executable|python_exe|\bpy)\s*,\s*["']-[mc]["']""")


def test_no_source_starts_python_without_the_safe_path() -> None:
    sources = [*(REPO / "flextool").rglob("*.py"), REPO / "flextool_run_supervisor.py"]
    offenders = [p.relative_to(REPO).as_posix() for p in sources if _UNSAFE.search(p.read_text(encoding="utf-8"))]
    assert offenders == [], "these start Python without -P; use flextool._child_python"
