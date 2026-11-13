"""The GUI's offer to install a compatible solver build (no window opened:
``_offer_solver_fix`` is called on a stand-in object, with the dialog
mocked)."""

import sys
import types

import pytest

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
    (steps, kw), = jobs
    assert steps == [[sys.executable, "-P", "-m", "pip", "install", "polars[rtcompat]==1.40.1"]]
    # The dialog states the reason and the exact command it will run.
    assert "avx2" in asked[0]
    assert env_check.format_command(steps[0]) in asked[0]
    assert asked[0].endswith("Run it now?")
    assert kw["intro"] == env_check.proactive_banner()


def test_declined_proactive_offer_runs_nothing(monkeypatch):
    monkeypatch.setattr(env_check, "_dist_version", {"polars": "1.40.1"}.get)
    monkeypatch.setattr(
        env_check, "cpu_verdict", lambda: env_check.CpuVerdict(env_check.CPU_NEEDS_COMPAT, "t", ["avx2"])
    )
    monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **kw: False)
    jobs = []
    main_window.MainWindow._offer_solver_fix(_stand_in(jobs), None, True)
    assert jobs == []


def test_crash_offer_uses_component_remedy(monkeypatch):
    monkeypatch.setattr(env_check, "_dist_version", {"highspy": "1.14.0"}.get)
    monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **kw: True)
    jobs = []
    probe = env_check.StackProbe(env_check.NATIVE_FAULT, 3221225477, failed_component="highspy")
    main_window.MainWindow._offer_solver_fix(_stand_in(jobs), probe)
    (steps, _kw), = jobs
    assert steps == env_check.highspy_upgrade_steps(sys.executable)


def test_declined_offer_runs_nothing(monkeypatch):
    monkeypatch.setattr(env_check, "_dist_version", {"highspy": "1.14.0"}.get)
    monkeypatch.setattr(main_window.messagebox, "askyesno", lambda *a, **kw: False)
    jobs = []
    probe = env_check.StackProbe(env_check.NATIVE_FAULT, 3221225477, failed_component="highspy")
    main_window.MainWindow._offer_solver_fix(_stand_in(jobs), probe)
    assert jobs == []


# ── the import-time dialog of a GUI started without a terminal ────
# A real (hidden) Tk root and progress window, so run under xvfb-run; the
# message boxes are recorded and pip is a stand-in.

_STEP = [sys.executable, "-P", "-m", "pip", "install", "polars[rtcompat]==1.40.1"]


class _Pip:
    def __init__(self, returncode):
        self.returncode = returncode
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return types.SimpleNamespace(returncode=self.returncode, stdout="", stderr="ERROR: no matching distribution\n")


def _boxes(monkeypatch, answer):
    import tkinter.messagebox as mb

    shown = []
    monkeypatch.setattr(mb, "askyesno", lambda title, text, **kw: shown.append(("ask", text)) or answer)
    monkeypatch.setattr(mb, "showinfo", lambda title, text, **kw: shown.append(("info", text)))
    monkeypatch.setattr(mb, "showerror", lambda title, text, **kw: shown.append(("error", text)))
    return shown


@pytest.fixture
def dialog_root(monkeypatch):
    """One hidden Tk root, handed to the dialog in place of the one it would create.

    Creating a Tk root can fail intermittently on a headless runner (the Windows CI runner), and the dialog
    reports that as "unavailable".  The root is created here once: when it cannot be, the test is skipped with
    the reason; when it can, the dialog under test runs on it and cannot fail to open its window.
    """
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"tkinter cannot open a window here (run under xvfb-run): {exc}")
    root.withdraw()
    monkeypatch.setattr(env_check, "_open_dialog_root", lambda: root)
    yield root
    try:
        root.destroy()  # the dialog destroys it on success
    except tkinter.TclError:
        pass


def test_dialog_no_runs_nothing(monkeypatch, dialog_root):
    shown = _boxes(monkeypatch, False)
    pip = _Pip(0)
    assert env_check.compat_install_dialog("REASON + COMMAND", [_STEP], pip) == "declined"
    assert pip.calls == []
    assert shown == [("ask", "REASON + COMMAND\n\nRun it now?")]


def test_dialog_yes_installs_and_says_so(monkeypatch, dialog_root):
    shown = _boxes(monkeypatch, True)
    pip = _Pip(0)
    assert env_check.compat_install_dialog("REASON", [_STEP], pip) == "installed"
    assert [argv for argv, _kw in pip.calls] == [_STEP]
    assert pip.calls[0][1]["capture_output"] is True
    assert [kind for kind, _text in shown] == ["ask", "info"]


def test_dialog_failed_install_shows_pip_output(monkeypatch, dialog_root):
    shown = _boxes(monkeypatch, True)
    assert env_check.compat_install_dialog("REASON", [_STEP], _Pip(1)) == "failed"
    kind, text = shown[-1]
    assert kind == "error"
    assert "exit 1" in text and "no matching distribution" in text and "REASON" in text
