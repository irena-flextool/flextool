"""Helpers shared by the commercial-CLI shims (``gurobi_cl``, ``cplex``, ``optimizer``, ``copt_cmd``).

Each shim stands in for one solver's command-line program and is driven exactly as FlexTool drives the real one
(``flextool/engine/solve/_subprocess_solve.py``: ``_gurobi_script`` / ``_cplex_script`` / ``_xpress_script`` /
``_copt_script``), but solves with the solver's Python package.  The test puts a launcher named like the real
program on ``PATH``; FlexTool finds it with ``shutil.which``.

Every call appends one JSON line to the file named by ``FLEXTOOL_SHIM_RECORD`` (when set): the arguments, the
stdin script, each parameter **as the solver package reports it after the solve** (read back, not echoed), the
solver's status and its relative MIP gap.  Tests assert on that record, so "what the solver got" is the
package's own view, not FlexTool's.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any


def record(entry: dict[str, Any]) -> None:
    """Append *entry* as one JSON line to ``$FLEXTOOL_SHIM_RECORD`` (no-op when unset)."""
    path = os.environ.get("FLEXTOOL_SHIM_RECORD")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def script_lines(text: str) -> list[str]:
    """The non-blank lines of a stdin script, stripped."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def single_path(command: str, rest: str) -> str:
    """The file argument of a console command.

    FlexTool writes paths unquoted.  How each real console tokenises a path with a space in it is not known
    here, so a shim refuses one instead of guessing (exit 2), which makes such a case visible in a test.
    """
    if any(ch.isspace() for ch in rest):
        print(
            f"shim: the path of {command!r} contains whitespace ({rest!r}); the real console's handling of an "
            "unquoted path with spaces is not reproduced",
            flush=True,
        )
        sys.exit(2)
    return rest


def typed(value: str, kind: type) -> Any:
    """*value* (text) as *kind* (``int`` / ``float`` / ``str``); raises ``ValueError`` when it does not parse.

    Strict on purpose: ``int("1.0")`` fails, as an integer parameter given a decimal would be refused.
    """
    if kind is bool:
        kind = int
    if kind is int:
        return int(value)
    if kind is float:
        return float(value)
    return value
