"""Cross-platform helper functions for opening files and launching external tools."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

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
