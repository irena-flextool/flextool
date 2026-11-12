"""Shim for the FICO Xpress Optimizer console (``optimizer``), built on the ``xpress`` Python package.

Invocation, as FlexTool builds it (``_xpress_script``): no arguments, a command script on stdin::

    setControl <NAME> <value>      (one per resolved parameter)
    readprob <solve>.mps
    lpoptimize
    writeslxsol <solve>.sol
    writeprtsol <solve>.sol
    quit

What it does:

* ``setControl``: the control is looked up in Xpress (``problem.getControl``, which is case-sensitive: the
  controls are upper case), the value parsed for the control's type (an integer control refuses ``1.5`` and
  ``1.0``) and set.  An unknown control or a bad value prints an
  error line and the script goes on without it.
* ``readprob`` / ``lpoptimize``: ``problem.readProb`` / ``problem.lpOptimize``.  On a MIP, ``lpoptimize``
  solves only the LP relaxation, as the console command does (FlexTool refuses MIPs on Xpress before it gets
  here).  The package's log goes to stdout.
* ``writeslxsol`` / ``writeprtsol``: ``problem.writeSlxSol`` / ``problem.writePrtSol`` — the library writers
  the console commands call.  The package writes the SLX file under the given name and the print file with
  ``.prt`` appended; ``_parse_xpress_sol`` looks for both spellings.

What it does NOT reproduce:

* Whether the console takes ``setControl <NAME> <value>`` at all, and its parsing of names (case), values
  (``1.0`` for an integer control) and paths.
  The real console's control syntax is the first thing to check with a licensed ``optimizer``.
* Console-only log lines, in particular whatever the console prints when ``MAXTIME`` stops a solve (FlexTool
  looks for ``MAXTIME limit reached`` / ``*** Time limit reached ***`` / ``time limit hit``; the library log
  here says ``Problem is unfinished``).
* The console's exit code and its banner.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shim_common import record, script_lines, single_path, typed  # noqa: E402


def main(stdin_text: str) -> int:
    warnings.simplefilter("ignore")  # the package's "Using the Community license" notice
    import xpress as xp

    prob = xp.problem()
    entry: dict = {"solver": "xpress", "stdin": stdin_text, "params": {}, "unknown": []}
    set_names: list[str] = []
    for line in script_lines(stdin_text):
        command, _, rest = line.partition(" ")
        command = command.lower()
        try:
            if command == "setcontrol":
                name, text = rest.split()
                try:
                    current = prob.getControl(name)
                except (xp.InterfaceError, xp.ModelError, xp.SolverError) as exc:
                    print(f"?? setControl {name}: {exc} (not set)", flush=True)
                    entry["unknown"].append(name)
                    continue
                try:
                    # Integer controls come back as ``int`` or an ``IntEnum`` (``PRESOLVE``).
                    kind = int if isinstance(current, int) else float if isinstance(current, float) else str
                    value = typed(text, kind)
                except ValueError:
                    print(f"?? setControl {name}: invalid value '{text}' (not set)", flush=True)
                    entry["unknown"].append(name)
                    continue
                prob.setControl(name, value)
                set_names.append(name)
            elif command == "readprob":
                prob.readProb(single_path("readprob", rest))
            elif command == "lpoptimize":
                prob.lpOptimize()
                entry["status"] = int(prob.attributes.lpstatus)
            elif command == "writeslxsol":
                prob.writeSlxSol(single_path("writeslxsol", rest))
            elif command == "writeprtsol":
                prob.writePrtSol(single_path("writeprtsol", rest))
            elif command == "quit":
                break
            else:
                print(f"?? unknown command '{command}' (shim)", flush=True)
        except (xp.InterfaceError, xp.ModelError, xp.SolverError) as exc:
            print(f"?? {command}: {exc}", flush=True)
            entry["error"] = str(exc)
    entry["params"] = {name: prob.getControl(name) for name in set_names}
    entry["is_mip"] = int(prob.attributes.mipents) > 0
    record(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.stdin.read()))
