"""Where FlexTool keeps its per-user files: the platform's user config and cache folders.

``FLEXTOOL_USER_HOME`` names a folder that stands in for both: the config folder is ``<it>/config`` and the cache
folder ``<it>/cache``.  The test suite sets it for every test and for every process a test starts, so no test
reads or writes the real user folders.

A folder named by ``FLEXTOOL_USER_HOME``, ``XDG_CONFIG_HOME`` or ``XDG_CACHE_HOME`` that is not a full path, or that
lies inside the current folder (a workspace), is ignored with a warning (review b2b3 L2): the solver settings file
(:mod:`flextool._local_solver_settings`) holds which workspaces this computer trusts, so it must never be a file
that a received workspace carries.  platformdirs would take a relative ``XDG_CONFIG_HOME`` as it is, against the
XDG specification.  The platform's own folder is used instead.

Only the standard library is imported at module level (``platformdirs`` inside the functions).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

USER_HOME_ENV = "FLEXTOOL_USER_HOME"

_logger = logging.getLogger(__name__)
_warned: set[tuple[str, str]] = set()


def _usable(name: str, value: str) -> Path | None:
    """*value* of the environment variable *name* as a folder, or ``None`` (with one warning) when it is not a full
    path or lies inside the current folder."""
    path = Path(value)
    problem = None
    if not path.is_absolute():
        problem = "is not a full path"
    else:
        try:
            if path.resolve().is_relative_to(Path.cwd().resolve()):
                problem = f"lies inside the current folder {Path.cwd()}, a workspace that could carry its own settings"
        except (OSError, RuntimeError):
            pass
    if problem is None:
        return path
    if (name, value) not in _warned:
        _warned.add((name, value))
        _logger.warning("Ignoring %s=%s: it %s. FlexTool uses the platform's user folder instead.", name, value, problem)
    return None


def _user_home() -> Path | None:
    value = os.environ.get(USER_HOME_ENV, "").strip()
    return _usable(USER_HOME_ENV, value) if value else None


def _platform_dir(kind: str) -> Path:
    import platformdirs

    xdg = "XDG_CONFIG_HOME" if kind == "config" else "XDG_CACHE_HOME"
    value = os.environ.get(xdg, "").strip()
    getter = platformdirs.user_config_dir if kind == "config" else platformdirs.user_cache_dir
    if value and _usable(xdg, value) is None and sys.platform not in ("win32", "darwin"):
        # platformdirs reads the variable on Linux and other Unix systems only: their XDG default stands in.
        return Path.home() / (".config" if kind == "config" else ".cache") / "flextool"
    return Path(getter("flextool", appauthor=False))


def user_config_dir() -> Path:
    """FlexTool's user config folder (``<user config folder>/flextool``)."""
    home = _user_home()
    if home is not None:
        return home / "config"
    return _platform_dir("config")


def user_cache_dir() -> Path:
    """FlexTool's user cache folder (``<user cache folder>/flextool``)."""
    home = _user_home()
    if home is not None:
        return home / "cache"
    return _platform_dir("cache")


def config_dir_overridden() -> bool:
    """Whether the user config folder comes from ``FLEXTOOL_USER_HOME`` or ``XDG_CONFIG_HOME`` (a folder anyone can
    name), not the platform's default."""
    return _user_home() is not None or (
        bool(os.environ.get("XDG_CONFIG_HOME", "").strip())
        and _usable("XDG_CONFIG_HOME", os.environ["XDG_CONFIG_HOME"].strip()) is not None
    )
