"""Where FlexTool keeps its per-user files: the platform's user config and cache folders.

``FLEXTOOL_USER_HOME`` names a folder that stands in for both: the config folder is ``<it>/config`` and the cache
folder ``<it>/cache``.  The test suite sets it for every test and for every process a test starts, so no test
reads or writes the real user folders.

A folder named by ``FLEXTOOL_USER_HOME``, ``XDG_CONFIG_HOME`` or ``XDG_CACHE_HOME`` that is not a full path is
ignored with a warning, and the platform's own folder is used (review b2b3 L2): a relative folder would be looked up
in whatever folder FlexTool starts in, and platformdirs would take a relative ``XDG_CONFIG_HOME`` as it is, against
the XDG specification.  A full path is used wherever FlexTool starts (item 240 final review M1): an earlier rule that
also ignored a folder inside the current folder made the settings, the assistant's store and the cache depend on the
start folder (a start from ``~`` or ``/`` ignored every such variable).  That the solver settings file, which holds
the folders this computer trusts, never lies inside a folder judged for trust is checked for each such folder
(:func:`flextool._local_solver_settings.trust_status`), which is the check that matters there.

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
    path.  Never depends on the current folder."""
    path = Path(value)
    if path.is_absolute():
        return path
    if (name, value) not in _warned:
        _warned.add((name, value))
        _logger.warning("Ignoring %s=%s: it is not a full path. FlexTool uses the platform's user folder instead.",
                        name, value)
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
