"""Claude Code PreToolUse hook for Bash: refuse the pytest runs that cost an hour instead of minutes.

Refused (exit 2, the reason goes back to the agent):

* pytest with no test path (with or without ``-k``; ``-m smoke``, the one-minute gate, is fine), or on ``tests`` / ``tests/engine`` as a whole (the full
  suite is about 5 h of worker time);
* the whole ``tests/test_scenarios.py`` (name the scenarios: ``tests/test_scenarios.py::test_x`` or ``-k``);
* ``-n`` above 8.

Pick the modules with ``python tools/related_tests.py``.  When the full suite is really wanted (a milestone gate the
owner asked for), prefix the command with ``FLEXTOOL_FULL_SUITE=1``.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

BROAD_PATHS = {"tests", "tests/", "./tests", "tests/engine", "tests/engine/", "."}
MAX_WORKERS = 8
#: Words that may precede the pytest command: wrappers, their options and VAR=value settings.
LAUNCHERS = {"timeout", "heavy", "/usr/local/bin/heavy", "xvfb-run", "-a", "nice", "env", "time"}


def _is_launcher(word: str) -> bool:
    return word in LAUNCHERS or "=" in word or re.fullmatch(r"\d+[smh]?", word) is not None


def problems(command: str) -> list[str]:
    if "FLEXTOOL_FULL_SUITE=1" in command:
        return []
    found = []
    for segment in re.split(r"&&|\|\||;|\|", command):
        try:
            words = shlex.split(segment)
        except ValueError:
            continue
        i = next((k for k, w in enumerate(words) if w == "pytest" or w.endswith("/pytest")), None)
        if i is None or not (words[i - 1:i] == ["-m"] or all(_is_launcher(w) for w in words[:i])):
            continue
        args = words[i + 1:]
        paths = [a for a in args if not a.startswith("-") and ("tests" in a or a.endswith(".py") or a == ".")]
        smoke = any(args[k:k + 2] == ["-m", "smoke"] for k in range(len(args)))
        if not paths and not smoke and "--pyargs" not in args:
            found.append("pytest without a test path runs the whole suite (a -k keyword sweep too)")
        broad = [p for p in paths if p.rstrip("/") in {b.rstrip("/") for b in BROAD_PATHS}]
        if broad:
            found.append(f"pytest on {broad[0]!r} is a directory sweep")
        if any(p.rstrip("/").endswith("tests/test_scenarios.py") for p in paths) and "-k" not in args:
            found.append("the whole tests/test_scenarios.py: name the scenarios (::test_x or -k)")
        for k, a in enumerate(args):
            n = a[2:] if a.startswith("-n") and a != "-n" else (args[k + 1] if a == "-n" and k + 1 < len(args) else "")
            if n.isdigit() and int(n) > MAX_WORKERS:
                found.append(f"-n {n}: at most -n {MAX_WORKERS}")
    return found


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0
    found = problems(event.get("tool_input", {}).get("command", ""))
    if not found:
        return 0
    print("Refused: " + "; ".join(found) + ".  Run only the modules your change touches "
          "(python tools/related_tests.py lists them); prefix FLEXTOOL_FULL_SUITE=1 only for a full-suite gate "
          "the owner asked for.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
