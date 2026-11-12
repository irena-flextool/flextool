"""Shim for COPT's command-line program (``copt_cmd``), built on ``coptpy``.

Invocation, as FlexTool builds it (``_copt_script``): no arguments, a command script on stdin::

    set <ParamName> <value>        (one per resolved parameter)
    read <solve>.mps
    optimize
    write <solve>.sol
    quit

What it does:

* ``set``: the parameter is looked up in COPT (``Model.getParamInfo``, which takes any case), the value parsed
  for its type (an integer parameter refuses ``1.5`` and ``1.0``) and set.  An unknown name or a bad value prints an error line and the
  script goes on without it.
* ``read`` / ``optimize`` / ``write``: ``Model.read`` / ``Model.solve`` / ``Model.write`` (a ``.sol`` name
  writes COPT's solution text, ``# Objective value <v>`` then ``<name> <value>`` lines, the layout
  ``_parse_copt_sol`` reads).  The package's log goes to stdout; its summary carries the lines FlexTool reads
  (``MIP status : stopped (time limit reached)``, ``Best gap : X%``).
* With no solution ``write`` writes nothing (as the library does).

What it does NOT reproduce:

* ``copt_cmd``'s own command parser (case, abbreviations, quoting, paths with spaces), its banner, the exact
  text of its error messages and its exit code.
* What ``copt_cmd`` prints for ``write`` without a solution: the shim's error line is its own.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shim_common import record, script_lines, single_path, typed  # noqa: E402


def main(stdin_text: str) -> int:
    import coptpy as cp

    env = cp.Envr()
    model = env.createModel("copt_cmd_shim")
    # COPT's names are case-insensitive; record each by its canonical spelling (``COPT.Param``).
    canonical = {n.casefold(): n for n in dir(cp.COPT.Param) if not n.startswith("_")}
    entry: dict = {"solver": "copt", "stdin": stdin_text, "params": {}, "unknown": []}
    set_names: list[str] = []
    for line in script_lines(stdin_text):
        command, _, rest = line.partition(" ")
        command = command.lower()
        try:
            if command == "set":
                name, text = rest.split()
                try:
                    info = model.getParamInfo(name)
                except (AttributeError, cp.CoptError) as exc:
                    print(f"ERROR: set {name}: {exc} (not set)", flush=True)
                    entry["unknown"].append(name)
                    continue
                try:
                    value = typed(text, type(info[1]))
                except ValueError:
                    print(f"ERROR: set {name}: invalid value '{text}' (not set)", flush=True)
                    entry["unknown"].append(name)
                    continue
                name = canonical.get(name.casefold(), name)
                model.setParam(name, value)
                set_names.append(name)
            elif command == "read":
                model.read(single_path("read", rest))
            elif command == "optimize":
                model.solve()
                entry["status"] = int(model.status)
                entry["is_mip"] = bool(model.ismip)
                if model.ismip and model.hasmipsol:
                    entry["mip_gap"] = model.getAttr("BestGap")
            elif command == "write":
                target = single_path("write", rest)
                if not (model.hasmipsol if model.ismip else model.haslpsol):
                    print(f"ERROR: write {target}: no solution to write", flush=True)
                    continue
                model.write(target)
            elif command == "quit":
                break
            else:
                print(f"ERROR: unknown command '{command}' (shim)", flush=True)
        except cp.CoptError as exc:
            print(f"ERROR: {command}: {exc}", flush=True)
            entry["error"] = str(exc)
    entry["params"] = {name: model.getParam(name) for name in set_names}
    record(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.stdin.read()))
