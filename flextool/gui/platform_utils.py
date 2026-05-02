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

    Resolution order:

    1. ``FLEXTOOL_DPI`` environment variable — explicit user override
       (e.g. ``FLEXTOOL_DPI=144``) for setups where auto-detection fails.
    2. On Windows, ``ctypes`` queries the system DPI directly.
    3. On Linux/X11, ``Xft.dpi`` (set by GNOME/KDE/Xfce) is the most
       reliable hint, followed by ``GDK_SCALE`` and finally an
       ``xrandr``-derived DPI computed from the primary monitor's
       physical dimensions.
    4. On macOS, Tk handles Retina internally — nothing to do.

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
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
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
        # Last resort: derive from xrandr's primary-monitor physical
        # dimensions.  Catches setups (e.g. plain X11 or some KDE
        # installs) where neither Xft.dpi nor GDK_SCALE is exported.
        if dpi is None:
            dpi = _xrandr_primary_dpi()

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


def _xrandr_primary_dpi() -> float | None:
    """Compute DPI from xrandr's primary-display physical width.

    Returns ``None`` when xrandr is unavailable, no primary display is
    flagged, or the reported physical width is 0 (some virtual /
    headless setups report 0mm).
    """
    import re
    try:
        out = subprocess.check_output(
            ["xrandr"], text=True, timeout=2, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    # Match e.g. "DP-0 connected primary 3840x2160+1920+0 ... 700mm x 390mm"
    pattern = re.compile(
        r"\bprimary\s+(\d+)x(\d+)\+\d+\+\d+.*?(\d+)mm\s+x\s+(\d+)mm"
    )
    for line in out.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        px_w = int(m.group(1))
        mm_w = int(m.group(3))
        if px_w > 0 and mm_w > 0:
            return (px_w / mm_w) * 25.4
    return None


def normalize_default_font_size(root: tk.Tk, size: int = 10) -> None:
    """Force TkDefaultFont and TkFixedFont to a consistent size across OS.

    Tk picks a platform-specific default (Segoe UI 9 on Windows,
    DejaVu Sans 10 on Linux, etc.) which causes layout differences.
    Calling this after theme initialisation normalises the size so that
    widget geometry is identical regardless of the operating system.
    """
    import tkinter.font as tkfont

    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont",
                 "TkMenuFont", "TkHeadingFont", "TkTooltipFont"):
        try:
            font = tkfont.nametofont(name)
            font.configure(size=size)
        except Exception:
            pass


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
