"""Format CLI command lists for readable log output."""
from __future__ import annotations


def format_cmd_for_log(cmd: list[str]) -> str:
    """Format a command list into a multi-line string with bash \\ continuations.

    Groups each ``--flag`` with its argument values on the same line.
    The result is copy-pasteable into a bash terminal.

    Example::

        python -m flextool.cli.cmd_scenario_results \\
          --parquet-base-dir /path/to/dir \\
          --alternatives DES DES-Invest PES TES \\
          --plot-dir /path/to/plots
    """
    if not cmd:
        return ''

    lines: list[str] = []
    current: list[str] = []

    for token in cmd:
        if token.startswith('--') and current:
            # Flush previous group
            lines.append(' '.join(current))
            current = [token]
        else:
            current.append(token)

    if current:
        lines.append(' '.join(current))

    if len(lines) <= 1:
        return lines[0] if lines else ''

    # Join with \ continuation; indent continuation lines
    return ' \\\n  '.join(lines)
