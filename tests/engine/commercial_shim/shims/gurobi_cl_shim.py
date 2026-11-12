"""Shim for ``gurobi_cl``, built on gurobipy.

Invocation, as FlexTool builds it (``_gurobi_script``)::

    gurobi_cl [<Name>=<value> ...] ResultFile=<solve>.json <solve>.mps

What it does, in ``gurobi_cl``'s order:

1. Every ``Name=value`` argument before the model file is a parameter (any case, ``_`` ignored, as in
   Gurobi).  Its type
   comes from Gurobi's own parameter table (``Model.getParamInfo``); an integer parameter refuses ``1.0``
   (strict: whether ``gurobi_cl`` accepts that is not known here).  It is set on the environment before the model is read, which
   prints Gurobi's ``Set parameter <Name> to value <v>`` line.  A name Gurobi does not know prints
   ``Unknown parameter: '<Name>' (ignored)`` and the solve goes on without it (the behaviour FlexTool's
   ``_warn_unknown_gurobi_parameters`` reads).  A value that does not parse for the parameter's type is an
   error: exit 1.
2. ``ResultFile`` is an ordinary Gurobi parameter: Gurobi itself writes the result file at the end of
   ``optimize()``.  For a ``.json`` name that is the JSON solution format ``_parse_gurobi_json`` reads.
3. The log is Gurobi's library log on stdout (``Time limit reached``, ``Best objective ..., gap X%``), the same
   library text ``gurobi_cl`` prints.
4. A Gurobi error (licence, size limit, unreadable model) prints ``Error <code>: <message>`` and exits 1.

What it does NOT reproduce:

* ``gurobi_cl``'s own argument parsing: its ``--`` options, its handling of a value with spaces or quotes,
  case rules for ``Name`` beyond what ``getParamInfo`` accepts, and the exact text of an invalid-value error.
* The exit code of the real program in each failure case (this shim: 0 after a solve, 1 on an error).
* Banner and summary lines that only the command-line program prints.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shim_common import record, typed  # noqa: E402


def main(argv: list[str]) -> int:
    import gurobipy as gp

    if not argv:
        print("Usage: gurobi_cl [--command]* [param=value]* filename", flush=True)
        return 1
    model_file, raw_params = argv[-1], argv[:-1]
    entry: dict = {"solver": "gurobi", "argv": argv, "params": {}, "unknown": []}
    try:
        probe_env = gp.Env(empty=True)
        probe_env.setParam("OutputFlag", 0)
        probe_env.start()
        probe = gp.Model(env=probe_env)
        # Gurobi reads names without case and without ``_`` after the first character (``Mip_Gap`` is
        # ``MIPGap``); record each by its canonical spelling (``GRB.Param``).
        def fold(n: str) -> str:
            return n[:1].casefold() + n[1:].replace("_", "").casefold()

        canonical = {fold(n): n for n in dir(gp.GRB.Param) if not n.startswith("_")}
        env = gp.Env(empty=True)
        passed: list[str] = []
        for token in raw_params:
            if "=" not in token:
                print(f"Unknown option: {token}", flush=True)
                return 1
            name, text = token.split("=", 1)
            info = probe.getParamInfo(name)
            if info is None:
                print(f"Unknown parameter: '{name}' (ignored)", flush=True)
                entry["unknown"].append(name)
                continue
            try:
                value = typed(text, info[1])
            except ValueError:
                print(f"Error: invalid value '{text}' for parameter {info[0]}", flush=True)
                return 1
            name = canonical.get(fold(name), info[0])
            env.setParam(name, value)
            passed.append(name)
        probe.dispose()
        probe_env.dispose()
        env.start()
        model = gp.read(model_file, env)
        model.optimize()
        # ``ResultFile`` is the file FlexTool asked for, not a solver option: recorded on its own.
        entry["result_file"] = model.getParamInfo("ResultFile")[2]
        entry["params"] = {name: model.getParamInfo(name)[2] for name in passed if name != "ResultFile"}
        entry["status"] = model.Status
        entry["is_mip"] = bool(model.IsMIP)
        entry["sol_count"] = model.SolCount
        entry["mip_gap"] = model.MIPGap if model.IsMIP and model.SolCount else None
        record(entry)
    except gp.GurobiError as exc:
        print(f"Error {exc.errno}: {exc}", flush=True)
        entry["error"] = str(exc)
        record(entry)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
