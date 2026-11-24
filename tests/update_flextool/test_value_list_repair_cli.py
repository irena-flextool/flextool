"""The value-list repair tool's command line and repair command (package P6).

Every database is a made-up copy of the schema database (``_value_list_cases``).  Oracles from the construction
(written by hand, never from the tool's output):

- ``_hit``: three default suspects (unit, connection, node ``invest_method`` point to R; one block question),
  U1 written before the overwrite (certain, ``set:not_allowed``), U2 and U3 written after it (no suggestion; same
  context, so one group).  U4 has no ``invest_method`` value, so it relies on the unit default in ``s1``: Enter
  changes results, and U4 is listed to check (no cumulative cap).  The prompts come in that order: the block, U1,
  then U2+U3.  Enter on the block and on U1 takes the suggestions; U2 ``set:invest_total`` and U3 ``keep`` give the
  record ``set`` 5 (3 defaults + U1 + U2) and ``kept`` 1 (U3).
- ``_restored``: as ``_hit`` without U2/U3, with ``not_allowed`` added back by hand (index 13, after the session)
  and the three defaults pointed to it: the defaults are the shipped default (no suspect), U1 is the only suspect
  and it is certain.
- ``_lost`` (part B): ``unit.conversion_method``, ``solve.solve_mode`` and ``node.storage_start_end_method``
  emptied by a migration step after the fingerprint (label "probably …"), one more reference default the bug could
  not reach ("no sign"), storage node A (start before the v77 commit) and B (start after it): one block question
  (results change: storage nodes), defaults first, then A and B.
"""
from __future__ import annotations

import contextlib
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from flextool.cli import cmd_repair_value_list as cli
from flextool.update_flextool import frozen as fz
from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_write as vw
from tests.update_flextool._value_list_cases import (VERSION_PREFIX, add_commit, clear_defaults, make_hit,
                                                     make_lost_defaults, member_id, sql, write_items)

ROOT = Path(__file__).resolve().parents[2]
LOST = ["unit.conversion_method", "solve.solve_mode", "node.storage_start_end_method"]


def _hit(tmp_path, schema_db_url, name="hit.sqlite", after=True):
    folder = tmp_path / name.removesuffix(".sqlite")
    folder.mkdir()
    after_values = [("unit", "U2", "base", "cumulative_limits"), ("unit", "U3", "base", "cumulative_limits")]
    case = make_hit(folder, schema_db_url, before=[("unit", "U1", "base", "not_allowed")],
                    after=after_values if after else ())
    write_items(case.path, [("unit", u, "efficiency", "base", 0.5) for u in ("U1", "U2", "U3", "U4")],
                {"s1": ["base"]})
    return case


def _restored(tmp_path, schema_db_url, after=False):
    case = _hit(tmp_path, schema_db_url, "restored.sqlite", after=after)
    h = add_commit(case.path, "Updated", "2025-01-10")
    sql(case.path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type, commit_id) '
                   "VALUES (?, 13, ?, 'str', ?)", (case.list_id, b'"not_allowed"', h))
    new = member_id(case.path, "invest_methods", "not_allowed")
    sql(case.path, "UPDATE parameter_definition SET default_value = ? WHERE default_type = 'list_value_ref' AND "
                   "CAST(default_value AS TEXT) = ?", (str(new).encode(), str(case.r_id)))
    return case


def _lost(tmp_path, schema_db_url, names=LOST):
    (extra,) = sorted(f"{c}.{p}" for c, p in ldr.reference().keys() - ldr.bug_reachable())[:1]
    folder = tmp_path / "lost"
    folder.mkdir()
    return make_lost_defaults(folder, list(names) + [extra], schema_db=schema_db_url,
                              before=[("node", "A", "storage_state_start", "base", 0.5)],
                              after=[("node", "B", "storage_state_start", "base", 0.3)], scenarios={"s1": ["base"]})


class _Ask:
    """A stub for ``input``: gives *replies* in order, then raises ``EOFError``; keeps the prompts."""

    def __init__(self, *replies: str) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.replies:
            raise EOFError
        return self.replies.pop(0)


def _files(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}


def _record_row(path: Path, key: str):
    return sql(path, "SELECT id, value FROM metadata WHERE name = ?", (key,))


def test_no_terminal_exits_3_and_names_the_answers_route(tmp_path, schema_db_url) -> None:
    """A real child process with an open stdin pipe (not a terminal): exit 3, nothing written, the recipe shown."""
    case = _hit(tmp_path, schema_db_url)
    sha = fz.sha256_of(case.path)
    proc = subprocess.Popen([sys.executable, "-P", "-m", cli.MODULE, str(case.path)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=_env(), cwd=tmp_path)
    out, _ = proc.communicate(timeout=300)
    text = out.decode("utf-8")
    assert proc.returncode == 3, text
    assert "--dry-run --write-answers" in text and "--answers answers.csv" in text
    assert _files(case.path.parent) == ["hit.sqlite"] and fz.sha256_of(case.path) == sha


def test_accept_certain_without_terminal(tmp_path, schema_db_url, capsys) -> None:
    """``--accept-certain`` with no terminal: a hit with class-default suspects still exits 3; a hit whose only
    suspect is certain (``_restored``) writes the copy with the record (exit 0)."""
    case = _hit(tmp_path, schema_db_url)
    assert cli.main([str(case.path), "--accept-certain"]) == 3
    assert "def:" in capsys.readouterr().out and _files(case.path.parent) == ["hit.sqlite"]
    restored = _restored(tmp_path, schema_db_url)
    assert cli.main([str(restored.path), "--accept-certain"]) == 0
    out = vw.default_output(restored.path)
    assert vlh.record_ends(vlh.session_findings_ro(out)[0], out)
    assert vlh.read_record(out).value["set"] == 1 and vlh.read_record(out).value["kept"] == 0


def test_dry_run_answers_round_trip_and_refusals(tmp_path, schema_db_url, capsys) -> None:
    """``--dry-run --write-answers`` writes no database; the file filled in (suggestions taken, the two later values
    kept) applies with ``--answers``; the same file on a copy with one more commit is refused (exit 1)."""
    case = _hit(tmp_path, schema_db_url)
    answers = tmp_path / "answers.csv"
    assert cli.main([str(case.path), "--dry-run", "--write-answers", str(answers)]) == 0
    assert _files(case.path.parent) == ["hit.sqlite"]
    assert cli.main([str(case.path), "--dry-run", "--write-answers", str(answers)]) == 1  # never overwritten
    with open(answers, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    for row in rows:
        row["answer"] = row["suggestion"].split(" (")[0] or "keep"
    with open(answers, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sibling = tmp_path / "sibling.sqlite"
    shutil.copy2(case.path, sibling)
    add_commit(sibling, "Updated", "2025-02-01")
    assert cli.main([str(sibling), "--answers", str(answers)]) == 1
    assert "another copy" in capsys.readouterr().err and not vw.default_output(sibling).exists()
    assert cli.main([str(case.path), "--answers", str(answers)]) == 0
    out = vw.default_output(case.path)
    assert vlh.read_record(out).value["set"] == 4 and vlh.read_record(out).value["kept"] == 2
    with pytest.raises(SystemExit) as exc:
        cli.main([str(case.path), "--write-answers", str(tmp_path / "x.csv")])
    assert exc.value.code == 2


@pytest.mark.parametrize("replies", [(), ("", "", "e", "keep", "keep", "n")], ids=["eof", "no"])
def test_quit_writes_partial_answers_only(tmp_path, schema_db_url, replies) -> None:
    """EOF at the first question, or "n" at the confirm: exit 1, the partial answers file, no database, no
    snapshot.  A second quit never overwrites the first file."""
    case = _hit(tmp_path, schema_db_url)
    sha = fz.sha256_of(case.path)
    for n in range(2):
        assert cli.main([str(case.path)], ask=_Ask(*replies)) == 1
    name = "hit_repaired.sqlite" + cli.QUIT_SUFFIX
    assert _files(case.path.parent) == sorted(["hit.sqlite", name, name.replace(".csv", "-1.csv")])
    assert fz.sha256_of(case.path) == sha
    with open(case.path.parent / name, encoding="utf-8-sig", newline="") as f:
        answered = [r["answer"] for r in csv.DictReader(f)]
    assert answered == ([""] * 6 if not replies else ["set:not_allowed"] * 4 + ["keep", "keep"])


def test_interactive_groups_rerun_and_part_a_record_kept(tmp_path, schema_db_url, capsys) -> None:
    """Block first: a number there is no answer ("Type Enter, e, x or q."), Enter takes the suggestions and the block
    lists U4 to check.  Then the groups: Enter takes a suggestion, Enter on a group without one asks again, ``e``
    drills down.  The rerun asks nothing.  A later run on the copy that restores an emptied default leaves the part A
    record row as it was (P5's open gap) and adds the part B record."""
    case = _hit(tmp_path, schema_db_url)
    ask = _Ask("1", "", "", "", "e", "set:invest_total", "k", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert len(ask.prompts) == 8 and ask.prompts[-1].startswith("Write to ") and ask.prompts[0] == ask.prompts[1]
    text = capsys.readouterr().out
    assert "No answer is suggested here" in text and "  Type Enter, e, x or q." in text
    assert "  unit.invest_method, no cumulative cap: U4 could invest and retire up to 1e6 MW until now" in text
    out = vw.default_output(case.path)
    assert vlh.read_record(out).value == {"v": 1, "step": 22, "end_commit": case.end_commit,
                                          "end_date": "2024-09-19", "set": 5, "kept": 1}
    report = out.with_name(out.name + vw.SIDE_TXT).read_text(encoding="utf-8")
    assert "\n  unit.invest_method, no cumulative cap: U4 could invest and retire up to 1e6 MW until now\n" in report
    sha = fz.sha256_of(out)
    ask = _Ask()
    assert cli.main([str(out)], ask=ask) == 0 and ask.prompts == []
    text = capsys.readouterr().out
    assert "This copy was repaired on" in text and "U3 [base]" in text and fz.sha256_of(out) == sha
    assert _files(out.parent) == sorted(["hit.sqlite", out.name, out.name + vw.SIDE_CSV, out.name + vw.SIDE_TXT])
    later = tmp_path / "later.sqlite"
    shutil.copy2(out, later)
    clear_defaults(later, ["unit.conversion_method"], add_commit(later, "Updated", "2026-11-24"))
    before = _record_row(later, vlh.RECORD_KEY)
    assert cli.main([str(later)], ask=_Ask("", "y")) == 0
    again = vw.default_output(later)
    assert _record_row(again, vlh.RECORD_KEY) == before and len(before) == 1
    assert len(_record_row(again, ldr.PART_B_RECORD_KEY)) == 1


def test_exit_codes(tmp_path, schema_db_url, capsys) -> None:
    """One row of the exit-code table each; the source is never changed and nothing is left next to it."""
    no_history = _hit(tmp_path, schema_db_url, "nohist.sqlite")
    sql(no_history.path, 'DELETE FROM "commit" WHERE comment LIKE ?', (VERSION_PREFIX + "%",))
    no_finding = _hit(tmp_path, schema_db_url, "late.sqlite")
    sql(no_finding.path, 'UPDATE "commit" SET date = ? WHERE id > ?', ("2025-08-01 12:00:00", no_finding.start_commit))
    old = _hit(tmp_path, schema_db_url, "old.sqlite")
    sql(old.path, 'UPDATE "commit" SET date = ? WHERE id > ?', ("2025-08-01 12:00:00", old.start_commit))
    newer = _hit(tmp_path, schema_db_url, "newer.sqlite")
    unclear = make_hit(tmp_path, schema_db_url, name="unclear.sqlite", legacy=True)
    sql(unclear.path, "UPDATE list_value SET commit_id = ? WHERE id = ?",
        (add_commit(unclear.path, "Updated", "2025-01-10"), unclear.r_id))
    for path, version in ((old.path, b"76.0"), (newer.path, b"99.0")):
        sql(path, "UPDATE parameter_definition SET default_value = ?, default_type = 'float' WHERE name = 'version' "
                  "AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'model')", (version,))
    archive = _hit(tmp_path, schema_db_url, "archived.sqlite")
    cases = [([no_history.path], 0, "W-0490 does not apply"), ([no_finding.path], 0, cli.NOTHING),
             ([old.path], 0, ldr.MIGRATE_FIRST), ([newer.path], 1, "update FlexTool"),
             ([unclear.path], 4, "checklist"), ([archive.path, "--archive", str(no_finding.path)], 1, "data version"),
             ([archive.path, "--output", str(archive.path)], 1, "not the original"),
             ([archive.path, "--output", str(no_finding.path)], 1, "exists already")]
    for argv, code, says in cases:
        sha, folder = fz.sha256_of(argv[0]), _files(Path(argv[0]).parent)
        assert cli.main([str(a) for a in argv]) == code, argv
        captured = capsys.readouterr()
        assert says in captured.out + captured.err, (argv, captured)
        assert fz.sha256_of(argv[0]) == sha and _files(Path(argv[0]).parent) == folder


def _part_b_record(path: Path):
    with contextlib.closing(fz._connect_ro(vw.default_output(path))) as con:
        return ldr.parse_part_b_record(con)


def test_part_b(tmp_path, schema_db_url, capsys) -> None:
    """A part B-only file: no terminal → 3 (also with ``--accept-certain``: defaults and storage questions are never
    certain); ``--dry-run`` prints the block.  The storage nodes make the block take the result-change closing,
    listing A and B; Enter restores the four defaults and keeps both starts free, ``y`` → the part B record
    ``(set, kept, storage) == (4, 2, 0)``, and the report lists A and B to check."""
    case = _lost(tmp_path, schema_db_url)
    for extra in ([], ["--accept-certain"]):
        assert cli.main([str(case.path), *extra]) == 3
        assert capsys.readouterr().out.count("  sn:") == 2
    a_line = "  node A [base] (scenarios: s1): an unaffected database fixes its start (fix_start); here it stays free"
    b_line = "  node B [base] (scenarios: s1): start added after the upgrade; it stays free"
    assert cli.main([str(case.path), "--dry-run"]) == 0
    assert a_line in capsys.readouterr().out
    ask = _Ask("", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert len(ask.prompts) == 2 and ask.prompts[0] == "  [Enter = restore; e = one by one; x = skip; q = quit] > "
    text = capsys.readouterr().out
    assert cli.NO_RECORD not in text and a_line in text and b_line in text
    assert ldr.CHECK_INTRO.strip() in " ".join(text.split())  # the block is printed wrapped
    rec = _part_b_record(case.path)
    assert (rec.set, rec.kept, rec.storage) == (4, 2, 0)
    out = vw.default_output(case.path)
    report = out.with_name(out.name + vw.SIDE_TXT).read_text(encoding="utf-8")
    assert a_line in report and b_line in report


def test_part_b_without_storage_changes_no_result(tmp_path, schema_db_url, capsys) -> None:
    """``_lost`` without ``node.storage_start_end_method``: no storage node, so the block says Enter changes no
    result and offers no ``e``; Enter, ``y`` → three defaults set."""
    case = _lost(tmp_path, schema_db_url, names=LOST[:2])
    ask = _Ask("", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert ask.prompts[0] == "  [Enter = restore; x = skip; q = quit] > "
    assert "FlexTool already uses these defaults when none is set." in " ".join(capsys.readouterr().out.split())
    rec = _part_b_record(case.path)
    assert (rec.set, rec.kept, rec.storage) == (3, 0, 0)


def test_block_one_by_one(tmp_path, schema_db_url, capsys) -> None:
    """At the block ``2`` is no answer ("Type Enter, e, x or q."), ``e`` goes through the items: the four defaults
    (Enter each), A ``fix_start`` (``1``, the only member offered), B kept, ``y``: 9 prompts and the record
    ``(4, 1, 1)``.  The summary names a default "(method parameter default)"."""
    case = _lost(tmp_path, schema_db_url)
    ask = _Ask("2", "e", "", "", "", "", "1", "k", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert len(ask.prompts) == 9 and ask.prompts[0] == ask.prompts[1]
    text = capsys.readouterr().out
    assert "  Type Enter, e, x or q." in text and "solve.solve_mode (method parameter default): empty -> single_solve" \
        in text
    rec = _part_b_record(case.path)
    assert (rec.set, rec.kept, rec.storage) == (4, 1, 1)


def test_set_to_the_current_member_and_a_spine_library_error(tmp_path, schema_db_url, monkeypatch, capsys) -> None:
    """Review finding 3.  ``_restored`` with U2/U3: U1 skipped, U2 ``set:cumulative_limits`` (what it reads), U3
    skipped, ``y``: the copy is written with nothing changed (no spinedb_api ``NothingToCommit``), no record, and
    the applied answers say ``keep`` for U2.  A spinedb_api error inside the write: exit 1 with one line, no
    traceback, nothing left next to the source."""
    from spinedb_api.exception import SpineDBAPIError

    case = _restored(tmp_path, schema_db_url, after=True)
    sha = fz.sha256_of(case.path)
    ask = _Ask("x", "e", "set:cumulative_limits", "x", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert "counted as keep" in capsys.readouterr().out
    out = vw.default_output(case.path)
    assert vlh.read_record(out) is None and fz.sha256_of(case.path) == sha
    with open(out.with_name(out.name + vw.SIDE_CSV), encoding="utf-8-sig", newline="") as f:
        answered = {r["entity"]: r["answer"] for r in csv.DictReader(f)}
    assert answered == {"U1": "skip", "U2": "keep", "U3": "skip"}
    other = tmp_path / "other"
    other.mkdir()
    source = _restored(other, schema_db_url).path  # U1 only: certain, so --accept-certain answers everything
    folder = _files(source.parent)

    def refuse(*args, **kwargs):
        raise SpineDBAPIError("boom")

    monkeypatch.setattr(vw, "_apply", refuse)
    assert cli.main([str(source), "--accept-certain"]) == 1
    err = capsys.readouterr().err
    assert "the Spine library refused the database (boom); nothing was written" in err and "Traceback" not in err
    assert _files(source.parent) == folder == ["hit.sqlite"]


def test_repair_command(tmp_path, monkeypatch) -> None:
    """Expected strings written by hand for each shell; ``-P`` always; the path made absolute; ``pythonw.exe``
    swapped for ``python.exe`` only when that file exists."""
    db, py, m = tmp_path / "my db %x# ä.sqlite", "/opt/py 3/bin/python", cli.MODULE
    assert cli.repair_command(db, python=py, shell="posix") == f"'/opt/py 3/bin/python' -P -m {m} '{db}'"
    assert cli.repair_command(db, python=py, shell="cmd") == f'"/opt/py 3/bin/python" "-P" "-m" "{m}" "{db}"'
    quote = tmp_path / "it's.sqlite"
    assert cli.repair_command(quote, python=py, shell="powershell") == \
        f"& '/opt/py 3/bin/python' '-P' '-m' '{m}' '{tmp_path}{os.sep}it''s.sqlite'"
    monkeypatch.chdir(tmp_path)
    assert cli.repair_command("rel.sqlite", python="py", shell="cmd") == f'"py" "-P" "-m" "{m}" "{tmp_path / "rel.sqlite"}"'
    if os.name != "nt":
        assert cli.repair_command(db, python=py) == cli.repair_command(db, python=py, shell="posix")
    with pytest.raises(ValueError):
        cli.repair_command(db, python=py, shell="fish")
    (tmp_path / "PythonW.exe").write_bytes(b"")
    w = str(tmp_path / "PythonW.exe")
    assert cli.repair_command("x", python=w, shell="cmd").startswith(f'"{w}" "-P" ')
    (tmp_path / "python.exe").write_bytes(b"")
    assert cli.repair_command("x", python=w, shell="cmd").startswith(f'"{tmp_path / "python.exe"}" "-P" ')


def test_repair_command_quotes_shell_characters(tmp_path) -> None:
    """Review finding 1: a folder ``R&D``, spaces and curly quotes, for each shell; every expected line written by
    hand.  cmd: each argument in double quotes (``&`` is then text, cmd does not cut the line); PowerShell: ``'`` and
    the curly single quotes U+2018 / U+2019 / U+201B (and U+201A) each doubled inside single quotes; posix:
    ``shlex`` quotes the whole argument in single quotes."""
    m = cli.MODULE
    folder = tmp_path / "Sales & R&D"
    db = folder / "Juha\u2019s \u2018model\u2019 \u201bx\u201a.sqlite"
    py = "/opt/Py 3/python"
    exe = py  # used as given (fc8dacc1): no Path(), which would turn / into \ on Windows
    assert cli.repair_command(db, python=py, shell="cmd") == f'"{exe}" "-P" "-m" "{m}" "{db}"'
    ps_name = "Juha\u2019\u2019s \u2018\u2018model\u2019\u2019 \u201b\u201bx\u201a\u201a.sqlite"
    assert cli.repair_command(db, python=py, shell="powershell") == \
        f"& '{exe}' '-P' '-m' '{m}' '{folder}{os.sep}{ps_name}'"
    assert cli.repair_command(db, python=py, shell="posix") == f"'{exe}' -P -m {m} '{db}'"
    plain = tmp_path / "R&D" / "db.sqlite"
    win_py = r"C:\Py\python.exe"
    assert cli.repair_command(plain, python=win_py, shell="cmd") == f'"{win_py}" "-P" "-m" "{m}" "{plain}"'
    assert cli.repair_command(tmp_path / "it's.sqlite", python=py, shell="posix") == \
        f"'{exe}' -P -m {m} '{tmp_path}{os.sep}it'\"'\"'s.sqlite'"


def test_import_is_light_and_module_help() -> None:
    """Importing the module adds no ``flextool`` module but itself (the package and ``_child_python`` are loaded
    anyway by the GUI), so the GUI's "Copy repair command" stays cheap; ``python -P -m … --help`` exits 0."""
    code = ("import sys, flextool.cli, flextool._child_python; before = set(sys.modules); "
            "import flextool.cli.cmd_repair_value_list; "
            "print(sorted(m for m in set(sys.modules) - before if m.split('.')[0] in ('flextool', 'polars', "
            "'spinedb_api')))")
    out = subprocess.run([sys.executable, "-P", "-c", code], capture_output=True, text=True, env=_env(),
                         timeout=300, check=True).stdout
    assert out.strip() == "['flextool.cli.cmd_repair_value_list']"
    proc = subprocess.run([sys.executable, "-P", "-m", cli.MODULE, "--help"], capture_output=True, text=True,
                          env=_env(), timeout=300)
    assert proc.returncode == 0 and "--write-answers" in proc.stdout
