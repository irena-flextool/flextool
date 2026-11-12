"""Names from input data that become file and folder names, and the folders FlexTool may empty or remove.

A scenario name is the name of its result folders (``output_parquet/<name>/``, ``output_csv/<name>/``,
``output_plots/<name>/``, and ``work/<name>_<N>/`` in the FlexTool window) and is passed to FlexTool's own commands
as an argument.  Input databases do not limit names, and they can come from someone else, so every place where a
run starts checks the name with :func:`output_name_problem`; the folder cleanups check with :func:`require_inside`
that what they empty or remove is inside the folder it belongs to; plot file names go through
:func:`safe_file_component`.

The rules (:func:`output_name_problem`) refuse only what cannot be a single folder name, or would be read as
something else: an empty name, ``.`` and ``..``, ``/`` and ``\\``, a control character, a leading ``-`` (it would be
read as an option of a FlexTool command), a drive prefix such as ``C:``, and the names Windows reserves for devices
(``CON``, ``NUL``, ``COM1``, ...).  On Windows also the characters ``< > : " | ? *`` and a trailing dot or space,
which Windows does not allow in file names.  Letters of any alphabet, digits, spaces and punctuation such as
``_ - . ( ) , + = @`` stay allowed.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

#: Device names Windows reserves, with or without an extension (``CON``, ``nul.txt``).
_WINDOWS_RESERVED = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                               *(f"LPT{i}" for i in range(1, 10))})
_WINDOWS_BAD_CHARS = frozenset('<>:"|?*')
_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f]")
_DRIVE = re.compile(r"^[A-Za-z]:")


class UnsafeNameError(ValueError):
    """A name from the input data cannot be used as a file or folder name."""


class UnsafePathError(RuntimeError):
    """A folder FlexTool was about to empty or remove is not inside the folder it belongs to."""


def output_name_problem(name: object, *, windows: bool | None = None) -> str | None:
    """Why *name* cannot name a result folder or be passed to a FlexTool command (``None`` when it can).

    *windows* (default: whether this is Windows) adds the characters and trailing dots or spaces Windows does not
    allow in file names.
    """
    if not isinstance(name, str) or not name:
        return "is empty"
    if name in (".", ".."):
        return "is '.' or '..'"
    if "/" in name or "\\" in name:
        return "holds '/' or '\\'"
    if _CONTROL.search(name):
        return "holds a control character (such as a line break or an escape character)"
    if name.startswith("-"):
        return "starts with '-'"
    if _DRIVE.match(name):
        return "starts with a drive letter and ':'"
    if name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED:
        return "is a name Windows reserves for devices (CON, PRN, AUX, NUL, COM1-9, LPT1-9)"
    if sys.platform == "win32" if windows is None else windows:
        if any(c in _WINDOWS_BAD_CHARS for c in name):
            return 'holds a character Windows does not allow in file names (< > : " | ? *)'
        if name.endswith((".", " ")):
            return "ends with a dot or a space, which Windows drops"
    return None


def describe_name(name: object) -> str:
    """*name* for a message: quoted, with control characters shown as ``\\xNN`` (Python's ``repr``, which also shows
    other characters that do not print, such as bidi controls, as ``\\uNNNN``, and keeps letters of any script)."""
    return repr(name)


def check_output_name(name: object, what: str = "scenario name") -> str:
    """*name* when it can name a result folder; else :class:`UnsafeNameError` with a message for the user."""
    reason = output_name_problem(name)
    if reason is not None:
        raise UnsafeNameError(
            f"The {what} {describe_name(name)} {reason}, so it cannot be used as a folder name. Rename it in the "
            "input data and run again.")
    return name  # type: ignore[return-value]


def require_inside(target: str | os.PathLike, root: str | os.PathLike) -> Path:
    """*target*, resolved, when it is *root* or inside it (both resolved); else :class:`UnsafePathError`.

    Call it before emptying or removing *target*.
    """
    resolved = Path(target).resolve()
    base = Path(root).resolve()
    if resolved != base and not resolved.is_relative_to(base):
        raise UnsafePathError(f"FlexTool did not empty or remove {target}: it is not inside {root}.")
    return resolved


def require_strictly_inside(target: str | os.PathLike, root: str | os.PathLike) -> Path:
    """Like :func:`require_inside`, but *target* must not be *root* itself."""
    resolved = require_inside(target, root)
    if resolved == Path(root).resolve():
        raise UnsafePathError(f"FlexTool did not remove {target}: it is the folder {root} itself.")
    return resolved


def safe_file_component(text: str, *, windows: bool | None = None) -> str:
    """*text* as one file name part, so a file named with it stays in its folder: ``/``, ``\\`` and control
    characters become ``_``, and so do ``< > : " | ? *`` on Windows (*windows*, default: whether this is Windows;
    elsewhere they are kept, so the plot file names there stay as they were); ``.`` or ``..`` as the whole text
    becomes ``_``; never empty."""
    bad = "/\\" + ("".join(sorted(_WINDOWS_BAD_CHARS)) if (sys.platform == "win32" if windows is None else windows)
                    else "")
    out = "".join("_" if (c in bad or _CONTROL.match(c)) else c for c in str(text))
    if out in ("", ".", ".."):
        return "_"
    return out


def result_workbook_name(folder: str) -> str:
    """File name of the Excel workbook of the results in ``output_parquet/<folder>/``: ``output_<folder>.xlsx``.

    Named after the result folder, not the scenario: the window's folders are ``<scenario>_<source number>``, so
    two input sources, or a scenario and one whose name extends it (``base_1``, ``base_10``), each get their own.
    """
    return f"output_{folder}.xlsx"


__all__ = [
    "UnsafeNameError", "UnsafePathError", "check_output_name", "describe_name", "output_name_problem",
    "require_inside", "require_strictly_inside", "result_workbook_name", "safe_file_component",
]
