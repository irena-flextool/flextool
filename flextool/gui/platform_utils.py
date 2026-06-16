"""Cross-platform helper functions for opening files and launching external tools."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tkinter as tk

logger = logging.getLogger(__name__)


def _clean_env() -> dict[str, str]:
    """Return a copy of os.environ without Python virtualenv variables.

    LibreOffice and other desktop apps bundle their own Python and crash
    (std::bad_alloc) if they inherit PYTHONPATH/VIRTUAL_ENV from the
    parent process.  This only affects the child process — the current
    Python process keeps its full environment.
    """
    env = {
        k: v for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "VIRTUAL_ENV", "VIRTUAL_ENV_DISABLE_PROMPT",
                     "PYTHONHOME", "CONDA_PREFIX", "CONDA_DEFAULT_ENV")
    }
    if "VIRTUAL_ENV" in os.environ:
        venv_bin = os.path.join(os.environ["VIRTUAL_ENV"], "bin")
        env["PATH"] = os.pathsep.join(
            p for p in os.environ.get("PATH", "").split(os.pathsep)
            if p != venv_bin
        )
    return env


def open_file_in_default_app(filepath: Path) -> None:
    """Open a file using the OS default application.

    Windows: os.startfile
    macOS: subprocess.Popen(['open', str(filepath)])
    Linux: subprocess.Popen(['xdg-open', str(filepath)])

    Raises:
        OSError: If the file does not exist or the platform command fails.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise OSError(f"File does not exist: {filepath}")

    if sys.platform == "win32":
        os.startfile(str(filepath))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(filepath)])
    else:
        subprocess.Popen(
            ["xdg-open", str(filepath)],
            start_new_session=True,
            env=_clean_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def open_folder(dirpath: Path) -> None:
    """Open a folder in the OS file manager.

    Windows: os.startfile
    macOS: subprocess.Popen(['open', str(dirpath)])
    Linux: subprocess.Popen(['xdg-open', str(dirpath)])

    Raises:
        OSError: If the directory does not exist.
    """
    dirpath = Path(dirpath)
    if not dirpath.is_dir():
        raise OSError(f"Directory does not exist: {dirpath}")

    if sys.platform == "win32":
        os.startfile(str(dirpath))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(dirpath)])
    else:
        subprocess.Popen(
            ["xdg-open", str(dirpath)],
            start_new_session=True,
            env=_clean_env(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def open_spine_db_editor(db_url: str) -> subprocess.Popen | None:
    """Launch spine-db-editor as a subprocess.

    Command: ``spine-db-editor <db_url>``

    Returns:
        The :class:`subprocess.Popen` object, or ``None`` if
        ``spine-db-editor`` is not found on the system PATH.
    """
    exe = shutil.which("spine-db-editor")
    if exe is None:
        logger.warning("spine-db-editor not found on PATH")
        return None

    try:
        proc = subprocess.Popen([exe, db_url])
        return proc
    except OSError:
        logger.warning("Failed to launch spine-db-editor", exc_info=True)
        return None


def set_process_dpi_awareness() -> None:
    """Mark the process DPI-aware on Windows, before the first Tk window.

    MUST be called **before** the Tk root (HWND) exists — Windows only
    honours ``SetProcessDpiAwareness`` while the process owns no windows.
    Called afterwards it is a silent no-op, leaving the process DPI-
    *unaware*: the OS then hands Tk a virtualized, bitmap-upscaled (blurry)
    screen and lies about its size. Setting awareness here makes Tk render
    at native pixels and report the true DPI, which ``apply_dpi_scaling``
    then converts into the matching ``tk scaling``. No-op off Windows
    (macOS handles Retina internally; Linux scales via ``tk scaling``).
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # 1 = PROCESS_SYSTEM_DPI_AWARE.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:
        pass


def apply_dpi_scaling(root: tk.Tk) -> float:
    """Detect the OS display scale factor and apply it to tkinter.

    Must be called **before** any widgets or themes are created so that
    default fonts (and everything derived from them) pick up the correct
    size.

    Resolution order:

    1. ``FLEXTOOL_DPI`` environment variable — explicit user override
       (e.g. ``FLEXTOOL_DPI=144``) for setups where auto-detection fails.
    2. On Windows, ``ctypes`` queries the system DPI directly (the process
       must already be DPI-aware via :func:`set_process_dpi_awareness`,
       called before the Tk root existed) and sets ``tk scaling`` to
       ``dpi / 72`` so fonts match the OS scale.
    3. On Linux/X11, ``Xft.dpi`` (set by GNOME/KDE/Xfce) is the most
       reliable hint, followed by ``GDK_SCALE`` and finally an
       ``xrandr``-derived DPI.  The xrandr fallback uses the monitor
       under the cursor at startup (which is where the WM will open
       the window on every desktop I've tested), and falls back to
       the primary monitor when the cursor's monitor reports a clearly
       bogus physical size (e.g. virtual / mirrored displays often
       report 0mm or copy a real monitor's dimensions verbatim).
    4. On macOS, Tk handles Retina internally — nothing to do.

    Once-at-startup is intentional.  ``tk scaling`` doesn't re-flow
    widgets that are already laid out, so changing it mid-session
    would only partially update the UI.

    A best-effort per-monitor handler in :class:`MainWindow` reapplies
    ``setup_fonts`` when ``winfo_fpixels('1i')`` swings by more than 10 %,
    so text rescales on monitor changes; widget positions still need a
    restart to fully re-flow.

    Returns:
        The ratio by which tk scaling was increased (1.0 means no change).
        Use this to rescale theme fonts that use hardcoded pixel sizes.
    """
    if sys.platform == "darwin":
        return 1.0

    dpi: float | None = None

    # Explicit override always wins.
    flextool_dpi = os.environ.get("FLEXTOOL_DPI")
    if flextool_dpi:
        try:
            dpi = float(flextool_dpi)
        except ValueError:
            pass

    if dpi is None and sys.platform == "win32":
        # Awareness is set earlier (set_process_dpi_awareness, before the Tk
        # root existed); here we only read the now-truthful system DPI.
        try:
            import ctypes
            dpi = ctypes.windll.user32.GetDpiForSystem()  # type: ignore[attr-defined]
        except Exception:
            pass
    elif dpi is None:
        # Linux / X11: try xrdb for Xft.dpi (set by GNOME, KDE, Xfce)
        try:
            out = subprocess.check_output(
                ["xrdb", "-query"], text=True, timeout=2, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if line.startswith("Xft.dpi:"):
                    dpi = float(line.split(":", 1)[1].strip())
                    break
        except Exception:
            pass
        # Fallback: GDK_SCALE environment variable (integer scale factor)
        if dpi is None:
            gdk_scale = os.environ.get("GDK_SCALE")
            if gdk_scale:
                try:
                    dpi = 96.0 * float(gdk_scale)
                except ValueError:
                    pass
        # Last resort: ask xrandr which monitor the cursor is on and
        # use that monitor's DPI.  Falls through to primary if the
        # cursor's monitor is suspect (mirrored / virtual / headless).
        if dpi is None:
            try:
                cursor_x, cursor_y = root.winfo_pointerxy()
            except Exception:
                cursor_x, cursor_y = (None, None)
            dpi = _xrandr_dpi_for_window(cursor_x, cursor_y)

    if dpi is not None and dpi > 0:
        current_scaling = float(root.tk.call("tk", "scaling"))
        if sys.platform == "win32":
            # Tk on Windows does NOT auto-scale ``tk scaling`` for the
            # display DPI — it stays pinned at the 96-DPI baseline (~1.33)
            # regardless of the real monitor. So point-sized fonts render
            # tiny on a scaled screen and, conversely, the old ">current"
            # guard made the result wildly machine-dependent (huge on one
            # box, tiny on another). The process is DPI-aware (set early),
            # so set scaling to the documented pixels-per-point UNCONDITION-
            # ALLY: every point-sized font and the row height derived from
            # it then track the OS scale factor together (no sparse rows /
            # tiny text). ``FLEXTOOL_DPI`` overrides the detected DPI.
            new_scaling = dpi / 72.0
            root.tk.call("tk", "scaling", new_scaling)
            return new_scaling / current_scaling
        # Linux/X11: desktop apps treat 96 DPI as "scale = 1.0", and a
        # 0.85 fudge keeps sv_ttk's heavier metrics in line with
        # surrounding Qt/GTK content. Only raise scaling, never shrink it
        # below tk's default on systems that already got it right.
        new_scaling = (dpi / 96.0) * 0.85
        if new_scaling > current_scaling:
            root.tk.call("tk", "scaling", new_scaling)
            return new_scaling / current_scaling
    return 1.0


def _xrandr_monitors() -> list[dict]:
    """Parse all connected monitors from xrandr.

    Returns a list of dicts with keys: name, primary, x, y, w, h,
    mm_w, mm_h, dpi.  Empty list if xrandr is unavailable.
    """
    import re
    try:
        out = subprocess.check_output(
            ["xrandr"], text=True, timeout=2, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    monitors: list[dict] = []
    # Match e.g. "DP-0 connected primary 3840x2160+1920+0 ... 700mm x 390mm"
    pattern = re.compile(
        r"^(\S+)\s+connected\s+(primary\s+)?(\d+)x(\d+)\+(\d+)\+(\d+)"
        r".*?(\d+)mm\s+x\s+(\d+)mm",
    )
    for line in out.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        name = m.group(1)
        primary = bool(m.group(2))
        w, h, x, y = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
        mm_w, mm_h = int(m.group(7)), int(m.group(8))
        dpi = (w / mm_w) * 25.4 if mm_w > 0 else 0.0
        monitors.append({
            "name": name, "primary": primary,
            "x": x, "y": y, "w": w, "h": h,
            "mm_w": mm_w, "mm_h": mm_h, "dpi": dpi,
        })
    return monitors


def _xrandr_dpi_for_window(x: int | None, y: int | None) -> float | None:
    """DPI for the monitor at (x, y); falls back to primary when sensible.

    A monitor is "suspect" when its physical-size readings are 0
    (virtual / mirrored / headless) or when its derived DPI is < 80
    (impossibly low — typically a virtual buffer claiming a real
    monitor's mm dimensions but at a fraction of the resolution).
    Suspect monitors are skipped in favour of the primary or the
    highest-DPI real monitor.
    """
    monitors = _xrandr_monitors()
    if not monitors:
        return None

    def is_real(m: dict) -> bool:
        return m["mm_w"] > 0 and m["dpi"] >= 80.0

    chosen = None
    if x is not None and y is not None:
        for m in monitors:
            if (m["x"] <= x < m["x"] + m["w"]
                    and m["y"] <= y < m["y"] + m["h"]):
                chosen = m
                break
    if chosen is not None and is_real(chosen):
        return chosen["dpi"]

    primary = next((m for m in monitors if m["primary"]), None)
    if primary is not None and is_real(primary):
        return primary["dpi"]

    real = [m for m in monitors if is_real(m)]
    if real:
        # Pick the highest-DPI real monitor as the safest default —
        # users on hi-DPI displays would rather have slightly oversized
        # widgets than unreadable ones.
        return max(real, key=lambda m: m["dpi"])["dpi"]

    return None


def scale_theme_fonts(root: tk.Tk, factor: float) -> None:
    """Rescale hardcoded theme fonts (e.g. sv_ttk's SunValley* fonts).

    sv_ttk defines its own fonts with absolute pixel sizes that do not
    respond to ``tk scaling``.  This function finds all ``SunValley*``
    fonts and multiplies their size by *factor*.

    Must be called **after** ``sv_ttk.set_theme()`` (which creates the
    fonts).
    """
    if factor <= 1.0:
        return
    import tkinter.font as tkfont

    for name in tkfont.names(root):
        if name.startswith("SunValley"):
            font = tkfont.Font(root=root, name=name, exists=True)
            current_size = font.cget("size")
            # Negative size = pixels, positive = points; scale either way
            font.configure(size=round(current_size * factor))
