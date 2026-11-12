"""Shim for the CPLEX Interactive Optimizer (``cplex``), built on the ``cplex`` Python package.

Invocation, as FlexTool builds it (``_cplex_script``): no arguments, a command script on stdin::

    set <parameter path words> <value>     (one per resolved parameter, e.g. ``set mip tolerances mipgap 0.01``)
    read <solve>.mps
    optimize
    write <solve>.sol sol
    quit

What it does:

* ``set``: the words before the value name the parameter the way the interactive optimizer does, which is the
  attribute path of the Python package (``set mip tolerances mipgap`` = ``parameters.mip.tolerances.mipgap``).
  Words are case-insensitive.  The value is parsed for the parameter's own type (an integer parameter refuses
  ``1.0``; whether the interactive optimizer does is not known here).  An unknown path or a value that does not parse prints a
  ``CPLEX Error`` line and the script goes on (an interactive session reports a bad command and reads the next
  one); the parameter is then not set.
* ``optimize`` solves (LP or MIP, as the model is).  The package's log goes to stdout.  The interactive
  optimizer's own result lines are then printed from the package's numbers:
  ``Solution status: <solutionStatusString>`` and, for a MIP with a solution,
  ``Current MIP best bound = <bound> (gap = <abs>, <rel>%)`` — the line ``_parse_cplex_sol`` reads the gap of a
  time-limited MIP from.
* ``write <file> sol`` writes CPLEX's XML solution file (``Cplex.solution.write``, the same writer the
  interactive command uses); with no solution it prints ``CPLEX Error  1217: No solution exists.``.

What it does NOT reproduce:

* The interactive parser: unique-prefix abbreviations (``set mip tol mipg``), the ``CPLEX>`` prompts, the
  exact text of its error messages, and its exit code.
* The interactive optimizer's own summary text beyond the two lines above (it is synthesised, not CPLEX's).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shim_common import record, script_lines, single_path, typed  # noqa: E402


def _parameter(cpx, words: list[str]):
    """The ``cplex`` parameter object of an interactive ``set`` path, or ``None`` when there is none."""
    node = cpx.parameters
    for word in words:
        if word.startswith("_"):
            return None
        node = getattr(node, word, None)
        if node is None:
            return None
    return node if hasattr(node, "set") and hasattr(node, "type") else None


def main(stdin_text: str) -> int:
    import cplex
    from cplex.exceptions import CplexError

    cpx = cplex.Cplex()
    entry: dict = {"solver": "cplex", "stdin": stdin_text, "params": {}, "unknown": []}
    set_params: dict[str, object] = {}
    is_mip = False
    for line in script_lines(stdin_text):
        command, _, rest = line.partition(" ")
        command = command.lower()
        try:
            if command == "set":
                *words, text = rest.split()
                param = _parameter(cpx, [w.lower() for w in words])
                if param is None:
                    print(f"CPLEX Error: unknown parameter 'set {' '.join(words)}' (not set)", flush=True)
                    entry["unknown"].append(" ".join(words))
                    continue
                try:
                    value = typed(text, param.type())
                except ValueError:
                    print(f"CPLEX Error: invalid value '{text}' for '{' '.join(words)}' (not set)", flush=True)
                    entry["unknown"].append(" ".join(words))
                    continue
                param.set(value)
                set_params[" ".join(w.lower() for w in words)] = param
                print(f"New value for {' '.join(words)}: {value}", flush=True)
            elif command == "read":
                cpx.read(single_path("read", rest))
                is_mip = cpx.problem_type[cpx.get_problem_type()] not in ("LP", "QP")
            elif command == "optimize":
                cpx.solve()
                sol = cpx.solution
                print(f"Solution status: {sol.get_status_string()}", flush=True)
                entry["status"] = sol.get_status_string()
                has_solution = sol.is_primal_feasible()
                if is_mip and has_solution:
                    obj, bound = sol.get_objective_value(), sol.MIP.get_best_objective()
                    rel = sol.MIP.get_mip_relative_gap()
                    print(f"Current MIP best bound = {bound:.10e} (gap = {abs(obj - bound):g}, {rel * 100:.2f}%)",
                          flush=True)
                    entry["mip_gap"] = rel
            elif command == "write":
                parts = rest.split()
                target = single_path("write", parts[0]) if parts else ""
                if parts[1:] != ["sol"]:
                    print(f"CPLEX Error: shim supports only 'write <file> sol', got {line!r}", flush=True)
                    continue
                if not cpx.solution.is_primal_feasible():
                    print("CPLEX Error  1217: No solution exists.", flush=True)
                    continue
                cpx.solution.write(target)
            elif command == "quit":
                break
            else:
                print(f"CPLEX Error: command '{command}' does not exist (shim)", flush=True)
        except CplexError as exc:
            print(str(exc), flush=True)
            entry["error"] = str(exc)
    entry["is_mip"] = is_mip
    entry["params"] = {name: param.get() for name, param in set_params.items()}
    record(entry)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.stdin.read()))
