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


def apply_dpi_scaling(root: tk.Tk) -> float:
    """Detect the OS display scale factor and apply it to tkinter.

    Must be called **before** any widgets or themes are created so that
    default fonts (and everything derived from them) pick up the correct
    size.

    On Linux/X11 the ``Xft.dpi`` X resource is the most reliable source
    (GNOME, KDE, and Xfce all set it).  The ``GDK_SCALE`` environment
    variable is checked as a fallback.

    On Windows, ``ctypes`` is used to query the system DPI.

    On macOS, Tk handles Retina scaling internally so nothing is needed.

    Returns:
        The ratio by which tk scaling was increased (1.0 means no change).
        Use this to rescale theme fonts that use hardcoded pixel sizes.
    """
    if sys.platform == "darwin":
        return 1.0

    dpi: float | None = None

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
            dpi = ctypes.windll.user32.GetDpiForSystem()  # type: ignore[attr-defined]
        except Exception:
            pass
    else:
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

    if dpi is not None and dpi > 0:
        # tk scaling is in units of pixels-per-point where 1.0 ≈ 72 DPI.
        # The formula is: scaling = dpi / 72
        new_scaling = dpi / 72.0
        current_scaling = float(root.tk.call("tk", "scaling"))
        # Only apply if the OS requests a higher scale than tk's default,
        # to avoid shrinking fonts on systems where tk already got it right.
        if new_scaling > current_scaling:
            root.tk.call("tk", "scaling", new_scaling)
            return new_scaling / current_scaling
    return 1.0


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
