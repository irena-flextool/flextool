"""``flextool-repair-value-list``: repair a database that W-0490 names, and restore lost shipped defaults (W-0491).

The tool never changes the database it is given: it writes a repaired copy (default ``<stem>_repaired.sqlite`` next
to it) with two side files, the applied answers (``.value_list_repair.csv``, replayable with ``--answers``) and a
report (``.value_list_repair.txt``).  The work is done by ``flextool.update_flextool.value_list_*`` and
``list_default_restore``; this module is the command line: one question per similarity group (``e`` answers each
item of the group), a summary and a confirm, the answers-file route when there is no terminal, and the exit codes:

====  ==========================================================================================================
0     copy written (with or without a record); nothing to do; W-0490 does not apply; rerun on a repaired copy;
      dry run finished
1     the user quit (a partial answers file is written); newer data version or SpineDB schema; archive refused;
      answers file refused; output exists or is the source; verify failed; any other error
2     usage error (argparse)
3     answers needed, but no terminal and no ``--answers``
4     the list matches no known pattern (``unclear``): nothing written, use the W-0490 checklist
====  ==========================================================================================================

:func:`repair_command` builds the command line the GUI copies and the warning texts show.  The module imports only
the standard library and ``flextool._child_python`` at top level, so the GUI can import it cheaply.

TODO(v5.0.0): remove with W-0490 (see ``update_flextool/value_list_history``), except the part B route.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from flextool._child_python import python_module_argv

MODULE = "flextool.cli.cmd_repair_value_list"
PROG = "flextool-repair-value-list"
EXIT_OK, EXIT_ERROR, EXIT_NEEDS_ANSWERS, EXIT_UNCLEAR = 0, 1, 3, 4
QUIT_SUFFIX = ".value_list_answers.csv"
NOTHING = "no value-list overwrite and no empty shipped default found; nothing to do"
NO_RECORD = "No repair record was written: the warning stays until every item has an answer."


def repair_command(db_path: str | Path, *, python: str | None = None, shell: str | None = None) -> str:
    """The command that runs this tool on *db_path* (made absolute), as one line for *shell*: ``"posix"``
    (``shlex``), ``"cmd"`` (``subprocess.list2cmdline``; cmd cannot escape ``%`` inside quotes) or ``"powershell"``
    (``& 'arg' ...``).  Default: ``"cmd"`` on Windows, ``"posix"`` elsewhere.  *python* defaults to
    :data:`sys.executable`; a ``pythonw.exe`` is swapped for the ``python.exe`` next to it (a pythonw process has no
    console, so it cannot ask)."""
    exe = Path(python or sys.executable)
    if exe.name.lower() == "pythonw.exe" and (console := exe.with_name("python.exe")).exists():
        exe = console
    argv = python_module_argv(MODULE, str(Path(db_path).resolve()), python=str(exe))
    shell = shell or ("cmd" if os.name == "nt" else "posix")
    if shell == "posix":
        return shlex.join(argv)
    if shell == "cmd":
        return subprocess.list2cmdline(argv)
    if shell == "powershell":
        return "& " + " ".join("'" + a.replace("'", "''") + "'" for a in argv)
    raise ValueError(f"unknown shell {shell!r} (posix, cmd or powershell)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG, description="Repair a FlexTool database whose invest_methods value list an old Spine library "
        "overwrote (W-0490), and restore shipped defaults an old upgrade left empty (W-0491). The database itself "
        "is never changed: a repaired copy is written next to it.")
    p.add_argument("database", help="the .sqlite file")
    p.add_argument("--output", help="the repaired copy (default: <name>_repaired.sqlite next to the database)")
    p.add_argument("--archive", help="an older copy of the database (data version below 22) to use as evidence")
    p.add_argument("--dry-run", action="store_true", help="show what would be asked and written; write nothing")
    p.add_argument("--write-answers", metavar="FILE.csv", help="with --dry-run: write the questions to an answers "
                   "file to fill in (answer column: keep, set:<member> or skip)")
    p.add_argument("--answers", metavar="FILE.csv", help="take the answers from this file (no questions)")
    p.add_argument("--accept-certain", action="store_true", help="answer the items the evidence makes certain with "
                   "their suggestion (class defaults and storage questions are always asked)")
    return p


class _Quit(Exception):
    pass


def _interactive() -> bool:
    return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())


def _where(s) -> str:
    return f" {s.entity} [{s.alternative}]" if s.entity else " (class default)"


def _line(s, answer=None) -> str:
    then = f"answer {answer.text()}" if answer else (f"suggested {s.suggestion}" if s.suggestion else "no suggestion")
    return f"  {s.item}  {s.cls}.{s.parameter}{_where(s)}: {s.label}; {then}"


def _show_unclear(a) -> None:
    from flextool.update_flextool.frozen import _connect_ro

    print(f"The invest_methods list matches no pattern this tool knows ({a.reason}). Nothing was written.")
    for m in a.members:
        print(f"  list_value {m.id}: index {m.index}, {m.value!r}, commit {m.commit}")
    if a.finding is not None:
        con = _connect_ro(a.path)
        try:
            for row in con.execute('SELECT id, CAST(date AS TEXT), comment FROM "commit" WHERE id >= ? AND id <= ? '
                                   "ORDER BY id", (a.finding.start_commit, a.finding.end_commit)):
                print("  commit {}: {} {}".format(*row))
        finally:
            con.close()
    print("Use the manual checklist of W-0490 in the FlexTool documentation (warnings reference).")


def _parse(text: str, s, allowed: tuple[str, ...]):
    """The prompt's answer for *s*: an ``Answer``, ``"each"``, or ``None`` (ask again).  Raises ``_Quit``."""
    from flextool.update_flextool import value_list_answers as va

    word = text.strip()
    low = word.lower()
    if low in ("q", "quit"):
        raise _Quit
    if low in ("e", "each"):
        return "each"
    if not low:
        return va.check_answer(s, s.suggestion, allowed) if s.suggestion else None
    word = {"k": "keep", "x": "skip"}.get(low, word)
    if word.isdigit() and 1 <= int(word) <= len(allowed):
        word = f"set:{allowed[int(word) - 1]}"
    return va.check_answer(s, word, allowed)


def _ask_group(items, choices: dict, ask, group: bool) -> dict:
    """Ask one question for *items* (one similarity group, or one item); returns ``{item: Answer}``."""
    from flextool.update_flextool import value_list_answers as va

    s = items[0]
    same = all(choices[t.item] == choices[s.item] for t in items)
    if len(items) == 1:
        print(f"\n{s.kind} {s.cls}.{s.parameter}{_where(s)}: {s.label}\n  {s.evidence}\n  {s.context.text()}"
              + (f"\n  {s.archive_note}" if s.archive_note else ""))
    else:
        one = len({t.evidence for t in items}) == 1
        print(f"\n{len(items)} items, {s.kind}: {s.label}" + (f"\n  {s.evidence}" if one else ""))
        for t in items[:8]:
            print(f"  - {t.cls}.{t.parameter}{_where(t)}: " + "; ".join(
                x for x in ("" if one else t.evidence, t.context.text(), t.archive_note) if x))
        if len(items) > 8:
            print(f"  ... and {len(items) - 8} more")
    if s.reason:
        print(f"  {s.reason}")
    members = choices[s.item] if same else ()
    if members:
        print("  members: " + ", ".join(f"{i} = {m}" for i, m in enumerate(members, 1)))
    enter = "Enter = each item's suggestion" if len({t.suggestion for t in items}) > 1 and all(
        t.suggestion for t in items) else (f"Enter = {s.suggestion}" if s.suggestion and all(
            t.suggestion == s.suggestion for t in items) else "")
    keep = s.keep_text if len(items) == 1 else "keep each as it is"
    opts = [o for o in (enter, f"k = {keep}", "a number = set that member" if members else "", "x = skip",
                        "e = answer each" if group and len(items) > 1 else "", "q = quit") if o]
    while True:
        text = ask(f"  [{'; '.join(opts)}] > ")
        try:
            got = [_parse(text, t, choices[t.item]) for t in items]
        except va.AnswersError as exc:
            print(f"  {exc}")
            continue
        if "each" in got and group and len(items) > 1:
            out = {}
            for t in items:
                out |= _ask_group([t], choices, ask, False)
            return out
        if all(isinstance(g, va.Answer) for g in got):
            return {t.item: g for t, g in zip(items, got)}
        print("  No answer is suggested here: please choose." if s.kind != "storage" else f"  {s.reason}")


def _summary(plan, output: Path) -> None:
    from flextool.update_flextool import value_list_answers as va

    sets = plan.sets
    keeps = [d for d in plan.decisions if d.action == "keep"]
    left = [d for d in plan.decisions if d.answer is None or d.action == "skip"]
    print(f"\nSummary: {len(sets)} set, {len(keeps)} kept, {len(left)} left as they are (skipped or not answered)"
          + (f"; restores {', '.join(plan.restore)} in invest_methods" if plan.restore else ""))
    for d in sets:
        print(f"  {d.suspect.cls}.{d.suspect.parameter}{_where(d.suspect)}: {d.suspect.current or 'empty'} -> "
              f"{d.answer.member}")
    for item, why in plan.stale:
        print(f"  stale (not applied): {item}: {why}")
    for w in plan.warnings:
        print(f"  warning: {w}")
    if plan.held:
        print(f"  {va.STORAGE_HELD}")
    print(f"  repair record: {'yes' if plan.part_a_record or plan.part_b_record else 'no'}; copy: {output}")


def _quit_file(output: Path, suspects, key: str, answers: dict) -> Path:
    from flextool.update_flextool import value_list_answers as va

    for folder in (output.parent, Path.cwd()):
        path = va.free_path(folder / (output.name + QUIT_SUFFIX))
        try:
            va.write_answers(path, suspects, key, answers)
            return path
        except OSError:
            continue
    raise OSError("the partial answers could not be written")


def main(argv: list[str] | None = None, *, ask=None) -> int:
    """Run the tool; *ask* (a callable like :func:`input`) makes it interactive without a terminal (tests)."""
    args = build_parser().parse_args(argv)
    if args.write_answers and not args.dry_run:
        build_parser().error("--write-answers needs --dry-run")
    from flextool.update_flextool import value_list_answers as va
    from flextool.update_flextool import value_list_repair as vr

    try:
        return _run(args, ask if ask is not None else (input if _interactive() else None))
    except (vr.RepairError, va.AnswersError, OSError) as exc:
        print(f"{PROG}: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print(f"{PROG}: interrupted; nothing was written.", file=sys.stderr)
    return EXIT_ERROR


def _run(args, ask) -> int:
    from spinedb_api.exception import SpineDBVersionError

    from flextool.update_flextool import value_list_repair as vr
    from flextool.update_flextool import value_list_write as vw
    from flextool.update_flextool.frozen import sqlite_path_from

    source = sqlite_path_from(args.database)
    if source is None or not source.is_file():
        raise vr.RepairError(f"No sqlite file at {args.database}.")
    output = Path(args.output) if args.output else vw.default_output(source)
    if args.dry_run:
        if output.resolve() == source.resolve():
            raise vr.RepairError("The repaired copy must be a new file, not the original.")
    else:
        vw.check_output(source, output)
    if args.write_answers and Path(args.write_answers).exists():
        raise vr.RepairError(f"{args.write_answers} exists already; give another name (it is never overwritten).")
    vw.check_schema(source)
    with tempfile.TemporaryDirectory(prefix="flextool-repair-") as tmp:
        snap = vr.snapshot(source, tmp if args.dry_run else output.parent, name=output.name)
        try:
            return _work(args, ask, source, output, vr.analyse(snap))
        except SpineDBVersionError as exc:
            raise vr.RepairError(f"The Spine database schema of {source.name} is older than its data version "
                                 "expects: run flextool-migrate-database on it (or open it in FlexTool), then run "
                                 "this tool again.") from exc
        finally:
            vw.discard(snap)  # a no-op once the copy is published (the snapshot became the output)


def _work(args, ask, source: Path, output: Path, a) -> int:
    from flextool.update_flextool import value_list_answers as va
    from flextool.update_flextool import value_list_archive as varc
    from flextool.update_flextool import value_list_repair as vr
    from flextool.update_flextool import value_list_suspects as vs
    from flextool.update_flextool import value_list_write as vw

    if a.state == vr.UNCLEAR:
        _show_unclear(a)
        return EXIT_UNCLEAR
    if a.state == vr.NO_HISTORY:
        print("W-0490 does not apply to this file (it has no upgrade history).")
    elif a.state == vr.NO_FINDING:
        print("No value-list overwrite (W-0490) in this file.")
    else:
        print(f"W-0490: state {a.state}" + (f", lost member {a.lost}" if a.lost else ""))
    if line := vr.part_b_line(a):
        print(line)
    found = vr.all_suspects(a)
    archive = None
    if args.archive:
        archive = varc.read_archive(args.archive, a)
        if archive.refused:
            raise vr.RepairError(f"The archive is refused: {archive.reason}.")
        if archive.reason:
            print(f"Archive: {archive.reason}")
        found = varc.apply_archive(a, found, archive)
    prev = vw.previous_repair(a, found)
    for done in (prev.part_a, prev.part_b):
        if done:
            print(f"This copy was {done}.")
    for s in prev.kept:
        print(f"  kept:{_where(s)} {s.cls}.{s.parameter}: {s.keep_text}")
    suspects = prev.suspects
    if not suspects and (prev.part_a or prev.part_b or a.state in (vr.NO_HISTORY, vr.NO_FINDING)):
        if not (prev.part_a or prev.part_b):
            print(NOTHING if not vr.part_b_line(a) else "Nothing to do for W-0490.")
        return EXIT_OK
    key, answers, stale, warnings = va.file_key(a), {}, (), ()
    if args.answers:
        read = va.read_answers(args.answers, a, suspects)
        answers, stale, warnings = read.answers, read.stale, read.warnings
    if args.accept_certain:
        answers = va.accept_certain(suspects) | answers
    stale_items = {item for item, _ in stale}
    needed = [s for s in suspects if s.item not in answers and s.item not in stale_items]
    if args.dry_run:
        print(f"Dry run: {len(suspects)} item(s); nothing was written.")
        for s in suspects:
            print(_line(s, answers.get(s.item)))
        if args.write_answers:
            va.write_answers(args.write_answers, suspects, key, answers)
            print(f"Answers file: {args.write_answers} (fill in the answer column, then run with --answers).")
        return EXIT_OK
    if needed and not args.answers:
        if ask is None:
            print(f"{len(needed)} item(s) need an answer, but there is no terminal to ask in:")
            for s in needed:
                print(_line(s))
            db = shlex.quote(str(source)) if os.name != "nt" else f'"{source}"'
            print(f"Answer them in a file:\n  1. {PROG} {db} --dry-run --write-answers answers.csv\n  2. fill in "
                  f"the answer column (keep, set:<member> or skip)\n  3. {PROG} {db} --answers answers.csv")
            return EXIT_NEEDS_ANSWERS
        choices = va.choices(a, suspects)
        try:
            for g in vs.groups(needed):
                answers |= _ask_group(list(g.suspects), choices, ask, True)
        except (_Quit, EOFError, KeyboardInterrupt):
            return _quit(output, suspects, key, answers)
    plan = va.make_plan(a, suspects, answers, stale=stale, warnings=warnings, key=key)
    if ask is not None:
        _summary(plan, output)
        try:
            if ask(f"Write to {output}? [y/N] ").strip().lower() not in ("y", "yes"):
                raise _Quit
        except (_Quit, EOFError, KeyboardInterrupt):
            return _quit(output, suspects, key, answers)
    w = vw.write_copy(a, plan, output, source=source, archive=archive, previous=prev)
    for line in w.lines:
        print(f"  {line}")
    print(f"Repaired copy: {w.output}\n  answers: {w.answers}\n  report: {w.report}")
    if not (w.part_a_record or w.part_b_record) and any(d.answer is None or d.action == va.SKIP
                                                         for d in plan.decisions):
        print(NO_RECORD)
    return EXIT_OK


def _quit(output: Path, suspects, key: str, answers: dict) -> int:
    path = _quit_file(output, suspects, key, answers)
    print(f"\nQuit: nothing was written to a database. The answers so far are in {path} (finish it and run with "
          "--answers).")
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
