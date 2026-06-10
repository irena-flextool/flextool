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

Use it at a Tool's ``__main__`` boundary::

    if __name__ == "__main__":
        run_tool(main)
"""
import sys


def run_tool(entry):
    """Run ``entry`` (a zero-arg callable) reconciling CLI vs Basic Console.

    See the module docstring for the rationale.
    """
    try:
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
