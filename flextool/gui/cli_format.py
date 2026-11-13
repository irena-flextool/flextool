"""Format CLI command lists for readable log output."""
from __future__ import annotations

#: The engine module the GUI runs; a terminal runs the same run as ``flextool-run``.
ENGINE_MODULE = "flextool.cli.cmd_run_flextool"
#: The console script that supervises an engine run from a terminal.
RUN_SCRIPT = "flextool-run"


def terminal_command(cmd: list[str]) -> list[str]:
    """The command a user can paste into a terminal for the same engine run as *cmd*.

    The GUI starts the engine itself (``<python> -m flextool.cli.cmd_run_flextool ...``) and writes its log; in a
    terminal ``flextool-run`` does that, so the interpreter and module become ``flextool-run``.  The GUI's
    ``--run-launcher gui`` is dropped: the pasted run is not a GUI run.  Any other command is returned unchanged.
    """
    out = list(cmd)
    for i in range(1, len(out) - 1):
        if out[i] == "-m" and out[i + 1] == ENGINE_MODULE:
            # The interpreter, its ``-P`` (``flextool._child_python``) and ``-m module`` become ``flextool-run``.
            start = i - 2 if i >= 2 and out[i - 1] == "-P" else i - 1
            out[start:i + 2] = [RUN_SCRIPT]
            break
    else:
        return out
    for i, tok in enumerate(out):
        if tok == "--run-launcher" and i + 1 < len(out):
            del out[i:i + 2]
            break
        if tok.startswith("--run-launcher="):
            del out[i]
            break
    return out


def _is_program_group(group: list[str]) -> bool:
    """True when *group* holds the program invocation (``python -m mod``,
    a ``…/script.py`` form or ``flextool-run``)."""
    return any(tok == "-m" or tok.endswith(".py") or tok == RUN_SCRIPT for tok in group)


def _split_program_positionals(group: list[str]) -> tuple[list[str], list[str]]:
    """Split a program-invocation *group* into ``(head, positionals)``.

    ``head`` is the interpreter + ``-m module`` (or ``script.py``, or ``flextool-run``) prefix;
    ``positionals`` are the obligatory positional arguments that follow it
    (e.g. the input-DB URL passed to ``cmd_run_flextool``).  Flags already
    start their own group upstream, so everything after the program entry
    in this group is positional.
    """
    end: int | None = None
    for i, tok in enumerate(group):
        if tok == "-m" and i + 1 < len(group):
            end = i + 2  # interpreter … -m module
            break
        if tok.endswith(".py") or tok == RUN_SCRIPT:
            end = i + 1
            break
    if end is None:
        return group, []
    return group[:end], group[end:]


def format_cmd_for_log(cmd: list[str]) -> str:
    """Format a command list into a multi-line string with bash \\ continuations.

    Groups each ``--flag`` with its argument values on the same line, and
    breaks the program's obligatory positional argument(s) onto their own
    lines, indented one space deeper than flag continuations so they read
    as arguments to the program rather than flags.  The result is
    copy-pasteable into a bash terminal.

    Example::

        systemd-run --user --scope --quiet \\
          -- flextool-run \\
             sqlite:////path/to/input_data.sqlite \\
          --scenario-name DES-Inv \\
          --work-folder /path/to/work
    """
    if not cmd:
        return ''

    # Group tokens: each ``--flag`` (and the bare ``--`` separator) starts a
    # new group that carries its trailing argument values.
    groups: list[list[str]] = []
    current: list[str] = []
    for token in cmd:
        if token.startswith('--') and current:
            groups.append(current)
            current = [token]
        else:
            current.append(token)
    if current:
        groups.append(current)

    # Render each group to a (indent, text) continuation line; positional
    # arguments of the program get a deeper indent than flag continuations.
    rendered: list[str] = []
    for idx, group in enumerate(groups):
        indent = '' if idx == 0 else '  '
        if _is_program_group(group):
            head, positionals = _split_program_positionals(group)
            rendered.append(indent + ' '.join(head))
            for pos in positionals:
                rendered.append('   ' + pos)
        else:
            rendered.append(indent + ' '.join(group))

    if len(rendered) <= 1:
        return rendered[0].strip() if rendered else ''

    # Join with ``\`` continuations; indents are already baked into each line.
    return ' \\\n'.join(rendered)
