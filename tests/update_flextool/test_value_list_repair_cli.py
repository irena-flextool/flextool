"""The value-list repair tool's command line and repair command (package P6).

Every database is a made-up copy of the schema database (``_value_list_cases``).  Oracles from the construction
(written by hand, never from the tool's output):

- ``_hit``: three default suspects (unit, connection, node ``invest_method`` point to R; three similarity groups,
  in definition order), U1 written before the overwrite (certain, ``set:not_allowed``), U2 and U3 written after it
  (no suggestion; same context, so one group).  The prompts come in that order: unit, connection, node defaults,
  U1, then U2+U3.  Enter on the first four takes the suggestion; U2 ``set:invest_total`` and U3 ``keep`` give the
  record ``set`` 5 (3 defaults + U1 + U2) and ``kept`` 1 (U3).
- ``_restored``: as ``_hit`` without U2/U3, with ``not_allowed`` added back by hand (index 13, after the session)
  and the three defaults pointed to it: the defaults are the shipped default (no suspect), U1 is the only suspect
  and it is certain.
- ``_lost`` (part B): ``unit.conversion_method``, ``solve.solve_mode`` and ``node.storage_start_end_method``
  emptied by a migration step after the fingerprint (label "probably …", one group), one more reference default the
  bug could not reach ("no sign", its own group), storage node A (start before the v77 commit) and B (start after
  it): four prompts, defaults first.
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


def _restored(tmp_path, schema_db_url):
    case = _hit(tmp_path, schema_db_url, "restored.sqlite", after=False)
    h = add_commit(case.path, "Updated", "2025-01-10")
    sql(case.path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type, commit_id) '
                   "VALUES (?, 13, ?, 'str', ?)", (case.list_id, b'"not_allowed"', h))
    new = member_id(case.path, "invest_methods", "not_allowed")
    sql(case.path, "UPDATE parameter_definition SET default_value = ? WHERE default_type = 'list_value_ref' AND "
                   "CAST(default_value AS TEXT) = ?", (str(new).encode(), str(case.r_id)))
    return case


def _lost(tmp_path, schema_db_url):
    (extra,) = sorted(f"{c}.{p}" for c, p in ldr.reference().keys() - ldr.bug_reachable())[:1]
    folder = tmp_path / "lost"
    folder.mkdir()
    return make_lost_defaults(folder, LOST + [extra], schema_db=schema_db_url,
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


@pytest.mark.parametrize("replies", [(), ("", "", "", "", "e", "keep", "keep", "n")], ids=["eof", "no"])
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
    """Group-first prompt: Enter takes a suggestion, Enter on a group without one asks again, ``e`` drills down.
    The rerun asks nothing.  A later run on the copy that restores an emptied default leaves the part A record row
    as it was (P5's open gap) and adds the part B record."""
    case = _hit(tmp_path, schema_db_url)
    ask = _Ask("", "", "", "", "", "e", "set:invest_total", "k", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert len(ask.prompts) == 9 and ask.prompts[-1].startswith("Write to ")
    assert "No answer is suggested here" in capsys.readouterr().out
    out = vw.default_output(case.path)
    assert vlh.read_record(out).value == {"v": 1, "step": 22, "end_commit": case.end_commit,
                                          "end_date": "2024-09-19", "set": 5, "kept": 1}
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


def test_part_b(tmp_path, schema_db_url, capsys) -> None:
    """A part B-only file: no terminal → 3 (also with ``--accept-certain``: defaults and storage questions are never
    certain).  At the prompt Enter on a storage question asks again with "No answer is recommended"; A
    ``fix_start`` (``1``, the only member offered), B kept → the copy with the part B record (storage 1)."""
    case = _lost(tmp_path, schema_db_url)
    for extra in ([], ["--accept-certain"]):
        assert cli.main([str(case.path), *extra]) == 3
        assert capsys.readouterr().out.count("  sn:") == 2
    ask = _Ask("", "", "", "1", "k", "y")
    assert cli.main([str(case.path)], ask=ask) == 0
    assert len(ask.prompts) == 6
    assert cli.NO_RECORD not in (text := capsys.readouterr().out)
    assert text.count(ldr.STORAGE_NOTE) >= 3  # A's question, its repeat, B's question
    out = vw.default_output(case.path)
    with contextlib.closing(fz._connect_ro(out)) as con:
        rec = ldr.parse_part_b_record(con)
    assert (rec.set, rec.kept, rec.storage) == (4, 1, 1)


def test_repair_command(tmp_path, monkeypatch) -> None:
    """Expected strings written by hand for each shell; ``-P`` always; the path made absolute; ``pythonw.exe``
    swapped for ``python.exe`` only when that file exists."""
    db, py, m = tmp_path / "my db %x# ä.sqlite", "/opt/py 3/bin/python", cli.MODULE
    assert cli.repair_command(db, python=py, shell="posix") == f"'/opt/py 3/bin/python' -P -m {m} '{db}'"
    assert cli.repair_command(db, python=py, shell="cmd") == f'"/opt/py 3/bin/python" -P -m {m} "{db}"'
    quote = tmp_path / "it's.sqlite"
    assert cli.repair_command(quote, python=py, shell="powershell") == \
        f"& '/opt/py 3/bin/python' '-P' '-m' '{m}' '{tmp_path}{os.sep}it''s.sqlite'"
    monkeypatch.chdir(tmp_path)
    assert cli.repair_command("rel.sqlite", python="py", shell="cmd") == f"py -P -m {m} {tmp_path / 'rel.sqlite'}"
    if os.name != "nt":
        assert cli.repair_command(db, python=py) == cli.repair_command(db, python=py, shell="posix")
    with pytest.raises(ValueError):
        cli.repair_command(db, python=py, shell="fish")
    (tmp_path / "PythonW.exe").write_bytes(b"")
    w = str(tmp_path / "PythonW.exe")
    assert cli.repair_command("x", python=w, shell="cmd").startswith(f"{w} -P ")
    (tmp_path / "python.exe").write_bytes(b"")
    assert cli.repair_command("x", python=w, shell="cmd").startswith(f"{tmp_path / 'python.exe'} -P ")


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
