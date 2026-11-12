"""Copy the engine's Python ``stdout`` / ``stderr`` into ``_run.log`` when no pipe owner writes the log.

A run started from Spine Toolbox, ``python -m flextool.cli.cmd_run_flextool`` or the root ``run_flextool.py`` has
nobody holding its output pipe, so the engine records its own console output (``log_capture: "python"``).  Only
what goes through ``sys.stdout`` / ``sys.stderr`` is seen: the native solver's lines and a crash dump write to
file descriptors 1 / 2 directly and are not in such a log.

:class:`TeeStream` passes every write to the wrapped stream and assembles lines per thread, so lines written by two
threads at once are never interleaved inside a line.  Control characters other than the tab and the line feed (an
escape sequence from a name in the input data) are shown as ``\\xNN`` on the console and in the log
(:func:`flextool._safe_text.printable`).  ``fileno`` / ``isatty`` / ``encoding`` come
from the wrapped stream, so code that routes native output by file descriptor (polar-high) behaves as before.
This module never imports the CLI.
"""
from __future__ import annotations

import logging
import sys
import threading
from typing import Any, Callable

from flextool._safe_text import printable
from flextool.run_record._owner import RunLogWriter

_logger = logging.getLogger(__name__)


class TeeStream:
    """A text stream that writes to *inner* and copies each complete line to *writer*."""

    def __init__(self, inner: Any, writer: RunLogWriter) -> None:
        self._inner = inner
        self._writer = writer
        self._lock = threading.Lock()
        self._partial: dict[int, list[str]] = {}
        self._broken = False

    @property
    def inner(self) -> Any:
        return self._inner

    def write(self, text: str) -> int:
        size = len(text)
        if text:
            text = printable(text, keep_newlines=True)
        self._inner.write(text)
        if text and not self._broken:
            self._copy(text)
        # The caller's length: a shown control character is longer than the character it stands for.
        return size

    def _copy(self, text: str) -> None:
        ident = threading.get_ident()
        with self._lock:
            parts = self._partial.setdefault(ident, [])
            if "\n" not in text:
                parts.append(text)
                return
            head, *middle, rest = text.split("\n")
            parts.append(head)
            lines = ["".join(parts), *middle]
            if rest:
                self._partial[ident] = [rest]
            else:
                del self._partial[ident]
        for line in lines:
            self._write_line(line)

    def _write_line(self, line: str) -> None:
        try:
            self._writer.write_line(line)
        except (OSError, ValueError) as exc:
            # The log is a copy: a full disk or a closed file must never break the console.
            self._broken = True
            try:
                self._inner.write(f"Run log copy stopped: {exc}\n")
            except (OSError, ValueError):
                pass

    def flush_partial(self) -> None:
        """Write the unfinished lines of every thread to the log."""
        with self._lock:
            pending = ["".join(parts) for parts in self._partial.values() if "".join(parts)]
            self._partial.clear()
        if not self._broken:
            for line in pending:
                self._write_line(line)

    def flush(self) -> None:
        self._inner.flush()

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def fileno(self) -> int:
        return self._inner.fileno()

    def isatty(self) -> bool:
        return self._inner.isatty()

    @property
    def encoding(self) -> str | None:
        return getattr(self._inner, "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._inner, "errors", None)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._inner, attr)


def install_tee(writer: RunLogWriter) -> Callable[[], None]:
    """Replace ``sys.stdout`` / ``sys.stderr`` by :class:`TeeStream` copies into *writer*; return ``restore``.

    Root-logger ``StreamHandler`` s bound to the old streams are re-pointed to the tees with ``setStream``.
    ``restore()`` writes the unfinished lines, puts the streams and the handlers back and is idempotent; it does
    not close *writer*.
    """
    old_out, old_err = sys.stdout, sys.stderr
    tee_out = TeeStream(old_out, writer) if old_out is not None else None
    tee_err = TeeStream(old_err, writer) if old_err is not None else None
    if tee_out is not None:
        sys.stdout = tee_out
    if tee_err is not None:
        sys.stderr = tee_err
    moved: list[tuple[logging.StreamHandler, Any]] = []
    for handler in logging.getLogger().handlers:
        if not isinstance(handler, logging.StreamHandler) or isinstance(handler, logging.FileHandler):
            continue
        stream = handler.stream
        if tee_out is not None and stream is old_out:
            handler.setStream(tee_out)
            moved.append((handler, old_out))
        elif tee_err is not None and stream is old_err:
            handler.setStream(tee_err)
            moved.append((handler, old_err))
    done = False

    def restore() -> None:
        nonlocal done
        if done:
            return
        done = True
        for tee in (tee_out, tee_err):
            if tee is not None:
                tee.flush_partial()
        for handler, stream in moved:
            if handler.stream is tee_out or handler.stream is tee_err:
                handler.setStream(stream)
        if tee_out is not None and sys.stdout is tee_out:
            sys.stdout = old_out
        if tee_err is not None and sys.stderr is tee_err:
            sys.stderr = old_err

    return restore


__all__ = ["TeeStream", "install_tee"]
