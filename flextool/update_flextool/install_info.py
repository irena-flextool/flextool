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
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


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
        if _has_upstream(root):
            pull = ["git", "pull", "--ff-only"]
        else:
            # The branch has no upstream tracking ref, so a bare `git pull`
            # fails with "no tracking information"; name the remote and branch
            # explicitly instead.
            remote = _default_remote(root)
            branch = current_branch(root) or "main"
            pull = ["git", "pull", "--ff-only", remote, branch]
        steps = [
            pull,
            [py, "-m", "pip", "install", "--upgrade", "-e", f".{extra}"],
        ]
        return steps, root
    steps = [[py, "-m", "pip", "install", "--upgrade", f"flextool{extra}"]]
    return steps, None


def _git(root: Path, *args: str, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=root, timeout=timeout, capture_output=True, text=True
    )


def current_branch(root: Path) -> str | None:
    """Current branch name, or ``None`` (detached HEAD / error)."""
    try:
        result = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
        name = result.stdout.strip()
        return name if result.returncode == 0 and name and name != "HEAD" else None
    except Exception:
        return None


def _has_upstream(root: Path) -> bool:
    """Whether the current branch has an upstream tracking ref."""
    try:
        return _git(
            root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
        ).returncode == 0
    except Exception:
        return False


def _default_remote(root: Path) -> str:
    """Pick a remote to pull from — ``origin`` if present, else the first."""
    try:
        remotes = _git(root, "remote").stdout.split()
    except Exception:
        remotes = []
    if "origin" in remotes:
        return "origin"
    return remotes[0] if remotes else "origin"


def update_available(timeout: float = 5.0) -> bool:
    """Best-effort check for a newer FlexTool version.

    For a git checkout this fetches and reports whether the tracked upstream
    branch is ahead; for a PyPI install it compares the installed version with
    the latest on PyPI. Returns ``True`` only on a confident positive — any
    error, offline state, or ambiguity returns ``False`` so the UI never nags
    spuriously. Performs network I/O; call it off the UI thread.
    """
    root = git_checkout_root()
    if root is not None:
        return _git_behind_upstream(root, timeout)
    return _pypi_has_newer(timeout)


def _git_behind_upstream(root: Path, timeout: float) -> bool:
    """Whether the checkout's tracked upstream has commits we don't have."""
    import os

    # Never let fetch block on an interactive credential / host-key prompt;
    # it must fail fast and silently if auth is needed.
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_SSH_COMMAND": "ssh -oBatchMode=yes",
    }
    try:
        subprocess.run(
            ["git", "fetch", "--quiet"],
            cwd=root, timeout=timeout, check=True, capture_output=True, env=env,
        )
        # Prefer the configured upstream; fall back to the remote branch of the
        # same name when the branch has no tracking ref set.
        if _has_upstream(root):
            ref = "@{u}"
        else:
            branch = current_branch(root)
            if branch is None:
                return False
            ref = f"{_default_remote(root)}/{branch}"
        result = subprocess.run(
            ["git", "rev-list", "--count", f"HEAD..{ref}"],
            cwd=root, timeout=timeout, check=True, capture_output=True, text=True,
        )
        return int(result.stdout.strip() or "0") > 0
    except Exception:
        # No upstream, no git, no network, auth needed, detached HEAD, … —
        # don't nag.
        logger.debug("git update check inconclusive for %s", root, exc_info=True)
        return False


def _pypi_latest_version(timeout: float) -> str | None:
    """Latest FlexTool version string on PyPI, or ``None`` on any failure."""
    request = Request(
        "https://pypi.org/pypi/flextool/json",
        headers={"User-Agent": "flextool-gui"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)["info"]["version"]
    except Exception:
        logger.debug("PyPI update check failed", exc_info=True)
        return None


def _pypi_has_newer(timeout: float) -> bool:
    latest = _pypi_latest_version(timeout)
    if latest is None:
        return False
    try:
        from packaging.version import Version

        return Version(latest) > Version(flextool_version())
    except Exception:
        logger.debug("Version comparison failed (latest=%s)", latest, exc_info=True)
        return False
