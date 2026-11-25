"""Claude Code Stop / SubagentStop hook: lint and the fast static guards before an agent reports done.

Runs only when the working tree has changes (a read-only session stops at once):

* ``ruff check`` on the changed Python files;
* the static guard tests (seconds, no solves): ``tests/test_invariant_guards.py``, the FlexData field routes, and
  the autoscale registry coverage / EQ-header grammar when the model, autoscale or schema changed.

A failure exits 2 and prints the output, which sends the agent back to fix it.  The second stop in a row
(``stop_hook_active``) is let through with the output still shown, so a failure the agent cannot fix (e.g. in a
file someone else is editing) never loops.  Wired in ``.claude/settings.json``; runnable by hand:
``python tools/agent_gate.py < /dev/null``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

GUARDS = ["tests/test_invariant_guards.py", "tests/engine/test_flexdata_field_routes.py"]
#: Extra guards when a changed path starts with one of these prefixes.
CONDITIONAL = {
    ("flextool/engine/model/", "flextool/engine/autoscale/", "flextool/schemas/"):
        ["tests/engine/autoscale/test_registry_coverage.py"],
    ("flextool/engine/model/",): ["tests/docs/test_header_grammar.py"],
}


def _run(cmd: list[str]) -> tuple[int, str]:
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8")
    return out.returncode, (out.stdout + out.stderr).strip()


def changed_files() -> list[str]:
    _, diff = _run(["git", "diff", "--name-only", "HEAD"])
    _, new = _run(["git", "ls-files", "--others", "--exclude-standard"])
    return [f for f in (diff + "\n" + new).splitlines() if f and (REPO / f).exists()]


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        event = {}
    files = changed_files()
    py = [f for f in files if f.endswith(".py")]
    if not py:
        return 0
    failures = []
    rc, out = _run([PY, "-m", "ruff", "check", "--output-format", "concise", "-q", *py])
    if rc:
        failures.append(f"ruff check (changed files):\n{out}")
    guards = [g for g in GUARDS if (REPO / g).exists()]
    for prefixes, extra in CONDITIONAL.items():
        if any(f.startswith(prefixes) for f in files):
            guards += [g for g in extra if g not in guards]
    rc, out = _run([PY, "-m", "pytest", "-n", "0", "-q", "--tb=line", "-p", "no:cacheprovider", *guards])
    if rc:
        failures.append(f"static guards ({' '.join(guards)}):\n{out[-4000:]}")
    if not failures:
        return 0
    print("agent_gate: fix these before reporting done.\n\n" + "\n\n".join(failures), file=sys.stderr)
    return 0 if event.get("stop_hook_active") else 2


if __name__ == "__main__":
    sys.exit(main())
