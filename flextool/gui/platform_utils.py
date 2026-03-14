"""Cross-platform helper functions for opening files and launching external tools."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


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
        subprocess.Popen(["xdg-open", str(filepath)])


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
        subprocess.Popen(["xdg-open", str(dirpath)])


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
