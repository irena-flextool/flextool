"""Tests for the Solver-options flag emission in ``_build_run_command``.

Covers the GUI → ``cmd_run_flextool`` command-line translation of the
five Solver-options-dialog knobs (log level, time limit, MIP relative
gap, matrix file format, scaling) plus presolve.  Two behaviours are
pinned here:

* ``--solver-mip-gap`` is appended only when ``solver_mip_gap > 0`` and
  carries a lossless ``repr(float)`` value, mirroring the sibling
  ``--solver-time-limit`` knob.
* ``--presolve`` is *always* forwarded, including the ``"choose"``
  default — the engine's determinism pin lives on the in-process
  baseline (used by the golden test gate), not this CLI path, so
  ``choose`` here genuinely lets HiGHS decide per-problem.

The manager constructor only stores ``project_path`` / ``settings`` and
reads package-bundled plot configs, so these tests are hermetic: no real
subprocess, DB, or network access.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.gui.data_models import ProjectSettings
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager
from flextool.gui.settings_io import (
    load_project_settings,
    save_project_settings,
)


def _cmd_for(tmp_path: Path, **setting_overrides) -> list[str]:
    """Build the engine command line for a SCENARIO job under *settings*."""
    settings = ProjectSettings(**setting_overrides)
    mgr = ExecutionManager(project_path=tmp_path, settings=settings)
    job = ExecutionJob(
        job_id=0,
        scenario_name="scen",
        output_subdir="scen",
        input_db_url="sqlite:///input.sqlite",
    )
    return mgr._build_run_command(job, tmp_path / "work")


def test_defaults_forward_presolve_choose_only(tmp_path: Path):
    """At GUI defaults, presolve=choose and the 0.001 MIP gap are
    forwarded; the other non-default-gated knobs are absent."""
    cmd = _cmd_for(tmp_path)
    assert "--presolve=choose" in cmd
    # MIP gap defaults to 0.001, so it is emitted at defaults.
    i = cmd.index("--solver-mip-gap")
    assert cmd[i + 1] == repr(0.001)
    # The other default-gated knobs emit nothing at their defaults.
    assert "--solver-time-limit" not in cmd
    assert not any(a.startswith("--scaling") for a in cmd)
    assert not any(a.startswith("--matrix-file-format") for a in cmd)
    assert not any(a.startswith("--solver-log-level") for a in cmd)


@pytest.mark.parametrize("ps", ["on", "off", "choose"])
def test_presolve_always_forwarded(tmp_path: Path, ps: str):
    cmd = _cmd_for(tmp_path, presolve=ps)
    assert f"--presolve={ps}" in cmd


def test_mip_gap_emitted_with_lossless_value(tmp_path: Path):
    cmd = _cmd_for(tmp_path, solver_mip_gap=0.01)
    i = cmd.index("--solver-mip-gap")
    assert cmd[i + 1] == repr(0.01)


def test_mip_gap_omitted_when_disabled(tmp_path: Path):
    # Checkbox off → no override emitted, regardless of the stored value.
    cmd = _cmd_for(tmp_path, solver_mip_gap_set=False, solver_mip_gap=0.01)
    assert "--solver-mip-gap" not in cmd


def test_mip_gap_zero_is_emitted_when_enabled(tmp_path: Path):
    # 0 is a valid gap (proven exact optimum) and must be sent when the
    # checkbox is on — it is no longer a "defer to .opt" sentinel.
    cmd = _cmd_for(tmp_path, solver_mip_gap_set=True, solver_mip_gap=0.0)
    i = cmd.index("--solver-mip-gap")
    assert cmd[i + 1] == repr(0.0)


# --- settings_io round-trip -------------------------------------------


def test_settings_io_roundtrips_mip_gap(tmp_path: Path):
    settings = ProjectSettings(solver_mip_gap=0.005)
    save_project_settings(tmp_path, settings)
    loaded = load_project_settings(tmp_path)
    assert loaded.solver_mip_gap == pytest.approx(0.005)


def test_settings_io_rejects_negative_mip_gap(tmp_path: Path):
    # A hand-edited / corrupt settings.yaml must not poison the GUI: a
    # negative gap falls back to the dataclass default rather than load.
    default_gap = ProjectSettings().solver_mip_gap
    settings = ProjectSettings(solver_mip_gap=0.5)
    save_project_settings(tmp_path, settings)
    yaml_path = tmp_path / next(
        p.name for p in tmp_path.iterdir() if p.suffix in (".yaml", ".yml")
    )
    text = yaml_path.read_text(encoding="utf-8")
    yaml_path.write_text(
        text.replace("solver_mip_gap: 0.5", "solver_mip_gap: -3.0"),
        encoding="utf-8",
    )
    loaded = load_project_settings(tmp_path)
    assert loaded.solver_mip_gap == default_gap
