"""The GUI's offer to install a compatible solver build (no window opened:
``_offer_solver_fix`` is called on a stand-in object, with the dialog
mocked)."""

import sys
import types

from flextool import env_check
from flextool.gui import main_window


def _stand_in(jobs):
    return types.SimpleNamespace(
        _run_cli_job=lambda steps, **kw: jobs.append((steps, kw)),
        _on_solver_fix_finished=lambda *a: None,
    )


def test_proactive_offer_installs_compat_runtime(monkeypatch):
    monkeypatch.setattr(env_check, "_dist_version", {"polars": "1.40.1"}.get)
    monkeypatch.setattr(
        env_check, "cpu_verdict", lambda: env_check.CpuVerdict(env_check.CPU_NEEDS_COMPAT, "t", ["avx2"])
    )
    asked = []
    monkeypatch.setattr(main_window.messagebox, "askyesno", lambda title, text, **kw: asked.append(text) or True)
    jobs = []
    main_window.MainWindow._offer_solver_fix(_stand_in(jobs), None, True)
    assert "avx2" in asked[0]
    (steps, kw), = jobs
    assert steps == [[sys.executable, "-m", "pip", "install", "polars[rtcompat]==1.40.1"]]
    assert kw["intro"] == env_check.proactive_banner()


def test_crash_offer_uses_component_remedy(monkeypatch):
    monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **kw: True)
    jobs = []
    probe = env_check.StackProbe(env_check.NATIVE_FAULT, 3221225477, failed_component="highspy")
    main_window.MainWindow._offer_solver_fix(_stand_in(jobs), probe)
    (steps, _kw), = jobs
    assert steps == env_check.downgrade_highspy_steps(sys.executable)


def test_declined_offer_runs_nothing(monkeypatch):
    monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **kw: False)
    jobs = []
    probe = env_check.StackProbe(env_check.NATIVE_FAULT, 3221225477, failed_component="highspy")
    main_window.MainWindow._offer_solver_fix(_stand_in(jobs), probe)
    assert jobs == []
