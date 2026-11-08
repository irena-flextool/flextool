"""Where FlexTool keeps its per-user files: the platform's user config and cache folders.

``FLEXTOOL_USER_HOME`` names a folder that stands in for both: the config folder is ``<it>/config`` and the cache
folder ``<it>/cache``.  The test suite sets it for every test and for every process a test starts, so no test
reads or writes the real user folders.

Only the standard library is imported at module level (``platformdirs`` inside the functions).
"""
from __future__ import annotations

import os
from pathlib import Path

USER_HOME_ENV = "FLEXTOOL_USER_HOME"


def _user_home() -> Path | None:
    value = os.environ.get(USER_HOME_ENV, "").strip()
    return Path(value) if value else None


def user_config_dir() -> Path:
    """FlexTool's user config folder (``<user config folder>/flextool``)."""
    home = _user_home()
    if home is not None:
        return home / "config"
    import platformdirs

    return Path(platformdirs.user_config_dir("flextool", appauthor=False))


def user_cache_dir() -> Path:
    """FlexTool's user cache folder (``<user cache folder>/flextool``)."""
    home = _user_home()
    if home is not None:
        return home / "cache"
    import platformdirs

    return Path(platformdirs.user_cache_dir("flextool"))
