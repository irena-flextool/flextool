"""Shared ``__main__`` shim for FlexTool CLI Tools.

Spine Toolbox's *Basic Console* does not run a Python Tool as a separate
``python script.py`` process.  It exec's the Tool's file inside a persistent
``python -i`` REPL (``sys.flags.interactive`` is set).  A ``sys.exit()``
there — whether at the end of the program or raised deep inside ``main()`` —
raises ``SystemExit``, which TERMINATES that REPL; Toolbox then pings the dead
process and reports a spurious ``Kernel died (×_×)``.  The Basic Console
decides success/failure from an *uncaught exception*, not the process exit
code.

``run_tool`` reconciles both runtimes.  It invokes the Tool's entry point and
swallows the resulting ``SystemExit``:

* Under ``-i`` (Basic Console) it returns quietly on a zero/``None`` code and
  re-raises a ``RuntimeError`` on a non-zero one, so Toolbox marks the Tool
  failed while the REPL stays alive — no "Kernel died".
* As a standalone CLI (``sys.flags.interactive == 0``) it preserves normal
  shell exit-code semantics (re-raises the original ``SystemExit`` / exits with
  the entry point's return value).

While ``entry`` runs, what it writes to ``sys.stdout`` / ``sys.stderr`` and what
the root logger's handlers write is shown with its control characters escaped
(:func:`flextool._safe_text.printable`), so that a name from the input data
cannot act on the terminal or the Toolbox console as a command.

Use it at a Tool's ``__main__`` boundary, and as the console script
(``cli``)::

    def cli():
        run_tool(main)

    if __name__ == "__main__":
        cli()
"""
import logging
import sys
from contextlib import contextmanager

from flextool._safe_text import PrintableFormatter, PrintableText


class _PrintableStream:
    """Pass-through text stream that writes :func:`flextool._safe_text.printable` text to *inner*."""

    def __init__(self, inner):
        self._inner = inner
        self._printable = PrintableText()

    @property
    def inner(self):
        return self._inner

    def write(self, text: str) -> int:
        self._inner.write(self._printable(text) if text else text)
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._inner, name)


@contextmanager
def printable_console():
    """Escape control characters in console output while the block runs (nested use is a no-op)."""
    if isinstance(sys.stdout, _PrintableStream) or isinstance(sys.stderr, _PrintableStream):
        yield
        return
    old_out, old_err = sys.stdout, sys.stderr
    root = logging.getLogger()
    old_formatters = [(h, h.formatter) for h in root.handlers if not isinstance(h.formatter, PrintableFormatter)]
    if old_out is not None:
        sys.stdout = _PrintableStream(old_out)
    if old_err is not None:
        sys.stderr = _PrintableStream(old_err)
    for handler, formatter in old_formatters:
        handler.setFormatter(PrintableFormatter(formatter))
    try:
        yield
    finally:
        for handler, formatter in old_formatters:
            handler.setFormatter(formatter)
        if isinstance(sys.stdout, _PrintableStream):
            sys.stdout = old_out
        if isinstance(sys.stderr, _PrintableStream):
            sys.stderr = old_err


def run_tool(entry):
    """Run ``entry`` (a zero-arg callable) reconciling CLI vs Basic Console.

    See the module docstring for the rationale.
    """
    try:
        with printable_console():
            rc = entry()
    except SystemExit as exc:
        if not sys.flags.interactive:
            raise  # standalone CLI: let the original exit code propagate
        code = exc.code
        rc = 0 if code is None else code
    if sys.flags.interactive:
        # Basic Console: NEVER sys.exit() — it would kill the persistent REPL.
        # Signal failure via an exception (Toolbox catches it); succeed quietly.
        if isinstance(rc, int):
            if rc != 0:
                raise RuntimeError(f"FlexTool Tool failed (exit code {rc}).")
        elif rc is not None:  # e.g. sys.exit("some message")
            raise RuntimeError(f"FlexTool Tool failed: {rc}")
        return rc
    sys.exit(rc)
