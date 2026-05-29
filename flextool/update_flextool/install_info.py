"""Detect how FlexTool is installed and build its self-upgrade command.

Two supported install shapes:

* an editable **git checkout** (``pip install -e .``) — upgraded with
  ``git pull`` followed by an editable reinstall;
* a **PyPI wheel** install — upgraded with ``pip install --upgrade``.

The standalone Tkinter GUI uses this to offer an in-app "Update FlexTool"
action that works regardless of which shape is in place.  The richer Spine
Toolbox workflow updater (``git restore`` + project-file rebuild) stays in
:mod:`flextool.update_flextool.self_update`; that behaviour is intentionally
*not* used from the GUI.
"""

from __future__ import annotations

import importlib.metadata as _im
import shutil
import sys
from pathlib import Path


def git_checkout_root() -> Path | None:
    """Return the repo root if FlexTool runs from an editable git checkout.

    Relies on the imported package's own location: an editable install imports
    ``flextool`` straight from the source tree, so its parent directory holds
    both ``.git`` and ``pyproject.toml``.  A wheel install imports from
    ``site-packages`` and fails this test (returns ``None``).
    """
    import flextool

    pkg_dir = Path(flextool.__file__).resolve().parent  # …/flextool
    repo_root = pkg_dir.parent
    if (repo_root / ".git").is_dir() and (repo_root / "pyproject.toml").is_file():
        return repo_root
    return None


def is_git_install() -> bool:
    """Whether FlexTool runs from an editable git checkout."""
    return git_checkout_root() is not None


def flextool_version() -> str:
    """Installed FlexTool version, or ``"unknown"`` if it cannot be read."""
    try:
        return _im.version("flextool")
    except _im.PackageNotFoundError:
        return "unknown"


def toolbox_installed() -> bool:
    """Whether Spine Toolbox (and thus the Spine DB Editor) is installed."""
    try:
        _im.distribution("spinetoolbox")
        return True
    except _im.PackageNotFoundError:
        return shutil.which("spine-db-editor") is not None


def describe_install() -> str:
    """One-line, user-facing description of the current install."""
    root = git_checkout_root()
    if root is not None:
        return f"git checkout at {root}\n(version {flextool_version()})"
    return f"PyPI install (version {flextool_version()})"


def upgrade_steps(include_toolbox: bool) -> tuple[list[list[str]], Path | None]:
    """Build the command(s) that upgrade FlexTool in place.

    Args:
        include_toolbox: when ``True`` the upgrade targets the ``[toolbox]``
            extra, so Spine Toolbox is installed (or kept) alongside FlexTool.

    Returns:
        ``(steps, cwd)`` where *steps* is a list of ``argv`` lists to run in
        order (stop at the first failure) and *cwd* is the directory they run
        in — the repo root for a git checkout, otherwise ``None``.
    """
    extra = "[toolbox]" if include_toolbox else ""
    py = sys.executable
    root = git_checkout_root()
    if root is not None:
        steps = [
            ["git", "pull", "--ff-only"],
            [py, "-m", "pip", "install", "--upgrade", "-e", f".{extra}"],
        ]
        return steps, root
    steps = [[py, "-m", "pip", "install", "--upgrade", f"flextool{extra}"]]
    return steps, None
