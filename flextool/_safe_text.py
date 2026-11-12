"""Show text from input data on a console or in a log window without its control characters taking effect.

Names and descriptions come from databases and project files that can be someone else's.  Printed raw, a control
character in them reaches the terminal (or the Spine Toolbox console) as a command: an escape sequence can hide or
rewrite earlier lines, retitle the window or, in some terminals, write to the clipboard.  :func:`printable` shows
each such character as ``\\xNN`` (``\\x1b[31m``), so the text is still readable and nothing is lost from the meaning.

Shown as ``\\xNN``: the C0 controls (U+0000-U+001F) except the tab (and the line feed where lines are kept), DEL
(U+007F) and the C1 controls (U+0080-U+009F, which some terminals read as the escape sequences' 8-bit form).  Shown
as ``\\uNNNN``: the bidirectional embeddings, overrides and isolates (U+202A-U+202E, U+2066-U+2069), which can
reorder what a line shows, and the line and paragraph separators (U+2028, U+2029).  The joiners U+200C / U+200D stay:
names in Persian and Indic scripts need them.  A carriage return is kept only as part of a Windows line end, which
becomes a line feed; :class:`PrintableText` also joins one split across two writes (``print("x\\r")`` writes ``"x\\r"`` and then ``"\\n"``).

Used by the ``flextool-run`` supervisor (its relay to the console and ``_run.log``), the engine's console log
handler and its own ``_run.log`` copy, and the FlexTool window's job log and run log window.  This module uses the
standard library only: the supervisor loads it by path, without importing the ``flextool`` package.
"""
from __future__ import annotations

import logging
import re
import threading

#: The characters :func:`printable` shows as ``\\xNN`` / ``\\uNNNN`` (the line feed separately, see ``keep_newlines``).
_BIDI_AND_SEPARATORS = "\u2028\u2029\u202a-\u202e\u2066-\u2069"
_CONTROL = re.compile(f"[\x00-\x08\x0b-\x1f\x7f-\x9f{_BIDI_AND_SEPARATORS}]")
_CONTROL_OR_NEWLINE = re.compile(f"[\x00-\x08\x0a-\x1f\x7f-\x9f{_BIDI_AND_SEPARATORS}]")


def _escape(match: re.Match) -> str:
    code = ord(match.group())
    return f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"


def printable(text: str, *, keep_newlines: bool = False) -> str:
    """*text* with its control characters shown as ``\\xNN`` (bidi controls as ``\\uNNNN``); tabs stay, and line
    feeds stay when *keep_newlines*.  A carriage return just before a kept line feed (a Windows line end) becomes
    part of that line feed."""
    if keep_newlines:
        text = text.replace("\r\n", "\n")
        return _CONTROL.sub(_escape, text)
    return _CONTROL_OR_NEWLINE.sub(_escape, text)


class PrintableText:
    """:func:`printable` (lines kept) for text written in pieces: a carriage return that ends a write is held back
    until the next write by the same thread, so a Windows line end split across two writes stays a line end."""

    def __init__(self) -> None:
        self._held_cr: set[int] = set()

    def __call__(self, text: str) -> str:
        ident = threading.get_ident()
        if ident in self._held_cr:
            self._held_cr.discard(ident)
            text = "\r" + text
        if text.endswith("\r"):
            self._held_cr.add(ident)
            text = text[:-1]
        return printable(text, keep_newlines=True)


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


__all__ = ["PrintableFormatter", "PrintableText", "make_printable", "printable"]
