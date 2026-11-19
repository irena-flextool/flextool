"""The warnings page of the web docs (``reference/warnings.md``), generated from the registry
``flextool.run_warnings.registry``: every code with its console line, what it means and how to fix it.

A code's section has the anchor ``#w-0123`` (the lowercase code), which ``registry.docs_url`` and the GUI's
links point to.
"""
from __future__ import annotations

from pathlib import Path

from flextool.run_warnings.registry import NOTE, WARNINGS, WARNINGS_DOCS_URL

_INTRO = f"""\
# Warnings and notes

A FlexTool run reports a problem it can work around as one console line per kind of problem and solve, ending
in a code:

```
Warning: <what is wrong> (<N> cases, e.g. '<first entity>') [W-0123]
```

The line names how many entities are affected and the first of them; the full list (and the values behind each
case) is in the run's log at debug level (run with `--debug`). At the end of a run a summary counts the warnings
of each code. `Note:` lines are information, not problems.

This page explains each code: what it means and how to fix it. Its address is
[{WARNINGS_DOCS_URL}]({WARNINGS_DOCS_URL}), and the section of a code is `#` followed by the code in lowercase
(for example `#w-0001`). The FlexTool GUI opens it when you click a code in the run output.

Not every FlexTool warning has a code yet; the lines without one explain themselves.
"""


def _console_line(code: str) -> str:
    spec = WARNINGS[code]
    prefix = "Note" if spec.severity == NOTE else "Warning"
    message = spec.message.replace("{", "<").replace("}", ">")
    return f"{prefix}: {message} (<N> cases, e.g. '<name>') [{code}]"


def render_warnings_page(markdown_root: Path | None = None) -> str:
    """The Markdown of the warnings page (*markdown_root* is unused: the page comes from the registry only)."""
    topics: dict[str, list[str]] = {}
    for code, spec in WARNINGS.items():
        topics.setdefault(spec.topic, []).append(code)
    parts = [_INTRO]
    for topic in sorted(topics):
        parts.append(f"## {topic}\n")
        for code in topics[topic]:
            spec = WARNINGS[code]
            kind = "Note" if spec.severity == NOTE else "Warning"
            # The heading is the code alone, so its anchor is the lowercase code (python-markdown's toc id).
            parts.append(f"### {code}\n")
            parts.append(f"**{spec.title}** ({kind.lower()}). Console line:\n\n```\n{_console_line(code)}\n```\n")
            parts.append(spec.body.strip() + "\n")
    return "\n".join(parts).rstrip("\n") + "\n"
