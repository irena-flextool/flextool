"""Format CLI command lists for readable log output."""
from __future__ import annotations


def _is_program_group(group: list[str]) -> bool:
    """True when *group* holds the program invocation (``python -m mod`` or
    a ``…/script.py`` form)."""
    return any(tok == "-m" or tok.endswith(".py") for tok in group)


def _split_program_positionals(group: list[str]) -> tuple[list[str], list[str]]:
    """Split a program-invocation *group* into ``(head, positionals)``.

    ``head`` is the interpreter + ``-m module`` (or ``script.py``) prefix;
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
        if tok.endswith(".py"):
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
          -- /path/python -m flextool.cli.cmd_run_flextool \\
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
