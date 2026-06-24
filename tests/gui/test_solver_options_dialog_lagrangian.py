"""GUI wiring test for the Lagrangian "Parallel workers" solver-option.

Constructs the real :class:`MainWindow` (under a headless display) and
exercises the two plumbing methods that move the value between the
project settings and the Tk variable backing the Solver-options dialog:

* ``_load_auto_gen_vars`` pushes ``ProjectSettings.lagrangian_workers``
  into ``lagrangian_workers_var`` (settings → widget), under the
  ``_suppress_auto_gen_save`` guard.
* ``_on_auto_gen_toggled`` writes ``lagrangian_workers_var`` back into
  ``ProjectSettings.lagrangian_workers`` and persists via
  ``save_project_settings`` (widget → settings, auto-save trace).

The same test asserts an existing field (``solver_time_limit``) still
round-trips through the very same methods, so a botched edit to the
trace-registration tuple or the load/save bodies is caught here too.

Run headless via ``xvfb-run -a``.
"""

from __future__ import annotations

import pytest

from flextool.gui import main_window as main_window_mod
from flextool.gui.main_window import MainWindow


@pytest.fixture()
def window():
    """Construct the real MainWindow, withdrawn; skip if no display."""
    try:
        w = MainWindow()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"MainWindow unavailable: {exc}")
    w.withdraw()
    try:
        yield w
    finally:
        w.destroy()


def test_lagrangian_workers_load_sync_and_autosave(window, monkeypatch):
    w = window

    # Capture persistence without touching the real projects dir.
    saved: list[object] = []
    monkeypatch.setattr(
        main_window_mod,
        "save_project_settings",
        lambda _path, settings: saved.append(settings),
    )
    # Ensure the auto-save path actually calls save (needs a project).
    w.current_project = w.current_project or "test-project"

    # settings → widget
    w.project_settings.lagrangian_workers = 5
    w.project_settings.solver_time_limit = 42
    w._load_auto_gen_vars()
    assert w.lagrangian_workers_var.get() == 5
    # Existing field still round-trips through the same method.
    assert w.solver_time_limit_var.get() == 42

    # widget → settings (auto-save trace)
    w.lagrangian_workers_var.set(3)
    w.solver_time_limit_var.set(7)
    w._on_auto_gen_toggled()
    assert w.project_settings.lagrangian_workers == 3
    # Existing field still written back.
    assert w.project_settings.solver_time_limit == 7
    # Persisted at least once with the new value present.
    assert saved, "save_project_settings was never invoked"
    assert saved[-1].lagrangian_workers == 3


def test_lagrangian_workers_zero_is_default(window):
    w = window
    w.project_settings.lagrangian_workers = 0
    w._load_auto_gen_vars()
    assert w.lagrangian_workers_var.get() == 0
