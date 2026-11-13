"""Solver settings of this computer, never of a project or database (item 240 B1).

One setting today: whether solver parameter names that are not in FlexTool's list of the solver's parameters
(:mod:`flextool.engine.solve._solver_catalogue`) are taken from input databases.  Off by default.  Two places,
the environment first:

1. ``FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS`` = ``1`` / ``0`` (also ``true`` / ``false``, ``yes`` / ``no``);
2. ``<user config folder>/solver_settings.json`` (:func:`flextool.user_dirs.user_config_dir`), key
   ``allow_unlisted_parameters`` (``true`` / ``false``), with ``"format": 1``.

Not in the database, a project folder, ``settings.yaml`` or ``projects.yaml``: those travel with a project or a
workspace.  No command-line flag either: a Spine Toolbox project carries tool command lines and can come from
someone else.  The value is read when it is needed (an unlisted name from a database), so a damaged file stops
only a run that depends on it, with a message naming it.

Only the standard library and :mod:`flextool.user_dirs` are imported.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ALLOW_UNLISTED_ENV = "FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS"
SETTINGS_FILE_NAME = "solver_settings.json"
FORMAT = 1
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class LocalSolverSettingsError(ValueError):
    """The environment variable or the settings file holds a value FlexTool cannot read."""


@dataclass(frozen=True)
class AllowUnlisted:
    """Whether unlisted solver parameter names pass from databases, and where that was set."""

    value: bool
    source: str  # ``environment variable FLEXTOOL_...``, the settings file's path, or ``default``

    def record(self) -> dict[str, object]:
        return {"value": self.value, "source": self.source}


def settings_path() -> Path:
    """This computer's solver settings file (it need not exist)."""
    from flextool.user_dirs import user_config_dir

    return user_config_dir() / SETTINGS_FILE_NAME


def read_settings_file(path: Path | None = None) -> dict:
    """The settings file's content (``{}`` when there is none); raises :class:`LocalSolverSettingsError` when it
    is not a FlexTool solver settings file of a format this FlexTool reads."""
    path = settings_path() if path is None else path
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise LocalSolverSettingsError(f"FlexTool cannot read its solver settings file {path} ({exc}).") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path} is not valid JSON ({exc}). Correct or delete it.") from exc
    if not isinstance(data, dict):
        raise LocalSolverSettingsError(f"FlexTool's solver settings file {path} does not hold a JSON object.")
    fmt = data.get("format", FORMAT)
    if not isinstance(fmt, int) or isinstance(fmt, bool) or fmt > FORMAT:
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path} has format {fmt!r}, written by a newer FlexTool; this "
            f"FlexTool reads format {FORMAT}. Update FlexTool, or delete the file.")
    return data


def allow_unlisted_parameters(environ: Mapping[str, str] | None = None) -> AllowUnlisted:
    """The allow switch for unlisted solver parameter names from databases (environment first, then the
    settings file, else off)."""
    env = os.environ if environ is None else environ
    raw = env.get(ALLOW_UNLISTED_ENV, "").strip()
    if raw:
        word = raw.casefold()
        if word not in _TRUE | _FALSE:
            raise LocalSolverSettingsError(
                f"{ALLOW_UNLISTED_ENV}={raw!r} is not 1 or 0. Set it to 1 to allow solver parameter names that "
                "are not in FlexTool's list from input databases, or to 0 (or unset it) to refuse them.")
        return AllowUnlisted(word in _TRUE, f"environment variable {ALLOW_UNLISTED_ENV}")
    path = settings_path()
    data = read_settings_file(path)
    if "allow_unlisted_parameters" not in data:
        return AllowUnlisted(False, "default")
    value = data["allow_unlisted_parameters"]
    if not isinstance(value, bool):
        raise LocalSolverSettingsError(
            f"FlexTool's solver settings file {path}: allow_unlisted_parameters is {value!r}; write true or false.")
    return AllowUnlisted(value, str(path))


def allow_unlisted_record(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """The run record's ``allow_unlisted_parameters``: ``{"value": bool, "source": ...}``; an unreadable setting
    gives ``{"value": None, "source": <the message>}`` (the run stops only if it needs the setting)."""
    try:
        return allow_unlisted_parameters(environ).record()
    except LocalSolverSettingsError as exc:
        return {"value": None, "source": str(exc)}
