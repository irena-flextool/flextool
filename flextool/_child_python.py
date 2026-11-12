"""The command lines and environments of the Python processes FlexTool starts.

``python -m module`` and ``python -c code`` put the current folder first on ``sys.path``, so a ``flextool/`` (or
``pip/``, ``polars/``, ...) folder in a project or workspace folder would be imported instead of the installed
package, and its code would run.  Every Python child FlexTool starts therefore runs with ``-P`` (Python 3.11+: do
not add the current folder or the script's folder to ``sys.path``), and its environment carries
``PYTHONSAFEPATH=1`` as a second guard, which also covers the Python processes that child starts in turn.
``PYTHONPATH`` keeps working.  The pip steps (updates, solver library repairs) get ``-P`` but not the variable, also
when FlexTool itself runs with it (:func:`pip_env`): pip runs the setup scripts of the packages it builds, and those
may import from their own folder.

This module uses the standard library only.  Two files cannot import it and spell the flag themselves:
``flextool_run_supervisor.py`` and ``flextool/env_check.py`` (both stdlib-only, loaded without importing the
``flextool`` package).  ``tests/test_child_python_safe_path.py`` checks every launcher.
"""
from __future__ import annotations

import os
import sys
from typing import Mapping

#: The interpreter flag that keeps the current folder off ``sys.path``.
SAFE_PATH_FLAG = "-P"
#: The environment variable with the same effect, inherited by grandchildren.
SAFE_PATH_ENV = "PYTHONSAFEPATH"


def python_argv(*args: str, python: str | None = None) -> list[str]:
    """``[python, "-P", *args]``; *python* defaults to :data:`sys.executable`."""
    return [python or sys.executable, SAFE_PATH_FLAG, *args]


def python_module_argv(module: str, *args: str, python: str | None = None) -> list[str]:
    """``[python, "-P", "-m", module, *args]``; *python* defaults to :data:`sys.executable`."""
    return python_argv("-m", module, *args, python=python)


def child_env(base: Mapping[str, str] | None = None, **extra: str) -> dict[str, str]:
    """A copy of *base* (default :data:`os.environ`) with ``PYTHONSAFEPATH=1`` and the *extra* variables."""
    env = dict(os.environ if base is None else base)
    env.update(extra)
    env[SAFE_PATH_ENV] = "1"
    return env


def pip_env(base: Mapping[str, str] | None = None, **extra: str) -> dict[str, str]:
    """A copy of *base* (default :data:`os.environ`) with the *extra* variables and without ``PYTHONSAFEPATH``, for
    the pip steps (see the module docstring)."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k != SAFE_PATH_ENV}
    env.update(extra)
    return env


__all__ = ["SAFE_PATH_ENV", "SAFE_PATH_FLAG", "child_env", "pip_env", "python_argv", "python_module_argv"]
