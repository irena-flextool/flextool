"""Show text from input data on a console or in a log window without its control characters taking effect.

Names and descriptions come from databases and project files that can be someone else's.  Printed raw, a control
character in them reaches the terminal (or the Spine Toolbox console) as a command: an escape sequence can hide or
rewrite earlier lines, retitle the window or, in some terminals, write to the clipboard.  :func:`printable` shows
each such character as ``\\xNN`` (``\\x1b[31m``), so the text is still readable and nothing is lost from the meaning.

Shown as ``\\xNN``: the C0 controls (U+0000-U+001F) except the tab (and the line feed where lines are kept), DEL
(U+007F) and the C1 controls (U+0080-U+009F, which some terminals read as the escape sequences' 8-bit form).

Used by the ``flextool-run`` supervisor (its relay to the console and ``_run.log``), the engine's console log
handler and its own ``_run.log`` copy, and the FlexTool window's job log and run log window.  This module uses the
standard library only: the supervisor loads it by path, without importing the ``flextool`` package.
"""
from __future__ import annotations

import logging
import re

#: The characters :func:`printable` shows as ``\\xNN`` (the line feed separately, see ``keep_newlines``).
_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_CONTROL_OR_NEWLINE = re.compile("[\x00-\x08\x0a-\x1f\x7f-\x9f]")


def _escape(match: re.Match) -> str:
    return f"\\x{ord(match.group()):02x}"


def printable(text: str, *, keep_newlines: bool = False) -> str:
    """*text* with its control characters shown as ``\\xNN``; tabs stay, and line feeds stay when
    *keep_newlines*.  A carriage return just before a kept line feed (a Windows line end) stays too."""
    if keep_newlines:
        text = text.replace("\r\n", "\n")
        return _CONTROL.sub(_escape, text)
    return _CONTROL_OR_NEWLINE.sub(_escape, text)


class PrintableFormatter(logging.Formatter):
    """A formatter that formats with *inner* (default: the plain ``%(message)s`` one) and then applies
    :func:`printable`, keeping line feeds (multi-line messages and tracebacks)."""

    def __init__(self, inner: logging.Formatter | None = None) -> None:
        super().__init__()
        self._inner = inner or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        return printable(self._inner.format(record), keep_newlines=True)


def make_printable(handler: logging.Handler) -> logging.Handler:
    """Give *handler* a :class:`PrintableFormatter` around its current formatter (once); return *handler*."""
    if not isinstance(handler.formatter, PrintableFormatter):
        handler.setFormatter(PrintableFormatter(handler.formatter))
    return handler


__all__ = ["PrintableFormatter", "make_printable", "printable"]
