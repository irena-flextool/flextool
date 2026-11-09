"""One X display per pytest-xdist worker for the Tk tests (``--run-gui -n N``).

The workers of one ``xvfb-run -a`` share its display, and with it the keyboard focus, the pointer and the window
stacking: the windows of one worker map under the pointer, take the focus and are destroyed while another worker's
test sends keys.  Beyond keys going to the wrong process, this crashes Tk 8.6.14 (a segmentation fault in
``focus -force``): when the X focus is on the root window (it reverts there when another process's focused window
dies), an Enter event gives a Tk application an *implicit* focus, and the LeaveNotify that ends it clears the
application's focus window but not the display's ``focusPtr``.  Destroying that window leaves ``focusPtr``
dangling, and the next ``focus -force`` reads the freed window (``TkSetFocusWin`` → ``FindDisplayFocusInfo``).
Tk's 8.6 branch after 8.6.14 clears ``focusPtr`` there too.

So each worker starts its own Xvfb (and openbox when the inherited display runs a window manager, as the CI job's
does) before any test module opens Tk, and points ``DISPLAY`` at it; a serial run (``-n 0``) keeps the display it
was given.  Without Xvfb installed the workers keep the shared display.

Every test module's leftover cyclic garbage is collected when the module ends, on the Tk (main) thread: a
``tkinter.Variable`` of a destroyed window collected later on a worker thread (the project hooks of
test_project_change_store_hooks.py, say) waits about a second per variable for a main loop that never runs.
"""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import time

import pytest

_SCREEN = "1280x1024x24"            # what xvfb-run -a gives
_started: list[subprocess.Popen] = []


def _has_window_manager(display: str) -> bool:
    """Whether an EWMH window manager runs on *display* (read with ``xprop``; no Tk on the inherited display)."""
    if not display or shutil.which("xprop") is None:
        return False
    try:
        out = subprocess.run(["xprop", "-display", display, "-root", "_NET_SUPPORTING_WM_CHECK"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "window id" in out.stdout


def _start_xvfb() -> str:
    """Start an Xvfb on a free display number; returns its ``:N``."""
    read_end, write_end = os.pipe()
    try:
        server = subprocess.Popen(
            ["Xvfb", "-displayfd", str(write_end), "-screen", "0", _SCREEN, "-nolisten", "tcp"],
            pass_fds=(write_end,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    finally:
        os.close(write_end)
    _started.append(server)
    with os.fdopen(read_end) as reader:
        number = reader.readline().strip()        # written once the server accepts connections
    if not number.isdigit():
        raise pytest.UsageError("Xvfb did not start for this pytest worker")
    return f":{number}"


def _start_openbox(display: str) -> None:
    """Start openbox on *display* and wait until it has taken the screen."""
    env = {**os.environ, "DISPLAY": display}
    _started.append(subprocess.Popen(["openbox"], env=env, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    deadline = time.monotonic() + 10
    while not _has_window_manager(display):
        if time.monotonic() >= deadline:
            raise pytest.UsageError(f"openbox did not start on {display} for this pytest worker")
        time.sleep(0.05)


@pytest.fixture(scope="module", autouse=True)
def _collect_garbage_on_the_tk_thread():
    """Collect what the module left (its Tk roots are destroyed by now: this fixture is torn down after them)."""
    yield
    gc.collect()


def pytest_configure(config: pytest.Config) -> None:
    if not os.environ.get("PYTEST_XDIST_WORKER") or not config.getoption("--run-gui", default=False):
        return
    if shutil.which("Xvfb") is None:
        return
    with_wm = _has_window_manager(os.environ.get("DISPLAY", "")) and shutil.which("openbox") is not None
    display = _start_xvfb()
    os.environ["DISPLAY"] = display
    os.environ.pop("XAUTHORITY", None)          # the inherited cookie is for the shared display
    if with_wm:
        _start_openbox(display)


def pytest_unconfigure(config: pytest.Config) -> None:
    while _started:
        process = _started.pop()                 # openbox before its display
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
