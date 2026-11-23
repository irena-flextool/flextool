"""The value-list repair tool's answers: the answers CSV, answer checks and the plan of a run.

**Answers CSV** (``--write-answers`` / ``--answers``, the partial answers file on a quit, the applied-answers side
file).  Written in UTF-8 with a BOM and ``,``; read as ``utf-8-sig`` with ``,`` or ``;`` sniffed (Excel with a
Finnish locale saves ``;``), any other encoding refused ("save as CSV UTF-8").  One row per suspect, keyed by
``item`` (``pv:<id>``, ``def:<id>``, ``sn:<entity id>:<alternative id>``); rows are never matched by line.

- ``file_key`` (every row) = ``k<end_commit>-<end date YYYYMMDD>-<snapshot max commit id>``, or
  ``k0-00000000-<max id>`` without a W-0490 finding (a part B-only run).  The letter keeps it text in Excel.  A
  file whose key differs was made for another copy of the database: refused.
- Strict per row: ``kind``, ``class``, ``parameter``, ``current`` (the member the suspect reads now in its own
  list).  A mismatch makes the row stale: reported, not applied, no record.  An ``item`` that is no suspect is an
  error (the file is refused).
- Lenient: ``entity`` / ``alternative`` that differ only the way Excel changes a value (surrounding spaces, or both
  parse as the same number, ``001`` vs ``1``, ``1.5`` vs ``1,5``) give a warning, and the row is applied.
- ``answer``: ``keep``, ``set:<member>``, ``skip``, or empty (no answer), in any case.  ``set:`` must name a member
  of the suspect's own list, or a member the write restores into R's list; a storage item (``sn:``) takes only
  ``set:fix_start`` or ``keep``.  ``--write-answers`` leaves ``answer`` empty: a pre-selected answer (the
  ``suggestion`` column) is never an answer until the user writes it there.

**Plan** (:func:`make_plan`): what the write sets, keeps and skips, and whether each record is due.  It counts only
explicit answers.  A record is due when every suspect of its part has ``keep`` or ``set:`` (no skip, no stale row,
no missing answer).  While any storage question has no answer, the ``node.storage_start_end_method`` default is held
back even if it is answered ``set:`` (it is what keeps those questions alive).

Standard library and the tool's own modules only; no database is written here.

TODO(v5.0.0): remove with W-0490 (see ``value_list_history``), except the part B rules.
"""
from __future__ import annotations

import contextlib
import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path

from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs

COLUMNS = ("item", "file_key", "kind", "class", "parameter", "entity", "alternative", "current", "answer",
           "evidence", "context", "archive", "suggestion")
_STRICT = ("kind", "class", "parameter", "current")
KEEP, SET, SKIP = "keep", "set", "skip"
STORAGE_HELD = "storage_start_end_method default left empty until every storage question has an answer"
NOT_UTF8 = "{path} is not UTF-8 text: save it as \"CSV UTF-8\" (in Excel: Save As, CSV UTF-8) and try again."


class AnswersError(Exception):
    """An answers file or an answer that cannot be used; the message says why.  Nothing is written."""


@dataclass(frozen=True)
class Answer:
    action: str  # KEEP, SET or SKIP
    member: str | None = None

    def text(self) -> str:
        return f"set:{self.member}" if self.action == SET else self.action


def file_key(a: vr.Analysis) -> str:
    """The answers file's fingerprint for the analysed snapshot (module docstring)."""
    from flextool.update_flextool.frozen import _connect_ro

    with contextlib.closing(_connect_ro(a.path)) as con:
        (top,) = con.execute('SELECT max(id) FROM "commit"').fetchone()
    f = a.finding
    if f is None:
        return f"k0-00000000-{top or 0}"
    return f"k{f.end_commit}-{f.date.replace('-', '')}-{top or 0}"


def choices(a: vr.Analysis, suspects) -> dict[str, tuple[str, ...]]:
    """The members ``set:`` may name, per suspect item: its own list's members, plus the members the write restores
    when that list is R's; ``fix_start`` only for a storage item."""
    from flextool.update_flextool.frozen import _connect_ro

    with contextlib.closing(_connect_ro(a.path)) as con:
        lists: dict[int, list[str]] = {}
        for lid, value in con.execute('SELECT parameter_value_list_id, value FROM list_value ORDER BY "index", id'):
            if (text := vr._text(value)) is not None:
                lists.setdefault(lid, []).append(text)
    out = {}
    for s in suspects:
        if s.kind == ldr.STORAGE_KIND:
            out[s.item] = (ldr.OLD_STORAGE_DEFAULT,)
            continue
        own = list(lists.get(s.list_id, ()))
        if s.list_id == a.list_id:
            own += [m for m in a.restore if m not in own]
        out[s.item] = tuple(own)
    return out


def suggestion_cell(s: vs.Suspect) -> str:
    """The CSV ``suggestion`` cell: the pre-selected answer with its reason (empty when there is none)."""
    if s.suggestion is None:
        return ""
    return f"{s.suggestion} ({s.reason})" if s.reason else s.suggestion


def check_answer(s: vs.Suspect, text: str, allowed: tuple[str, ...]) -> Answer | None:
    """The answer *text* for suspect *s* (``None`` when empty), checked against the members *allowed* for it
    (:func:`choices`); raises :class:`AnswersError` for anything else.  The whole ``suggestion`` cell counts as its
    answer (copied over in a spreadsheet)."""
    word = text.strip()
    if word and s.suggestion is not None and word == suggestion_cell(s):
        word = s.suggestion
    low = word.lower()
    if not low:
        return None
    if low in (KEEP, SKIP):
        return Answer(low)
    if low.startswith("set:"):
        wanted = word[4:].strip()
        hits = [m for m in allowed if m == wanted] or [m for m in allowed if m.lower() == wanted.lower()]
        if len(hits) == 1:
            return Answer(SET, hits[0])
        raise AnswersError(f"{s.item}: '{wanted}' is not a member of its list ({', '.join(allowed) or 'none'}).")
    raise AnswersError(f"{s.item}: '{text.strip()}' is not an answer (keep, set:<member>, skip or empty).")


def accept_certain(suspects) -> dict[str, Answer]:
    """``--accept-certain``: the pre-selected answer of every "certain" suspect, as an explicit answer.  Class
    defaults, part B and storage suspects are never certain, so they are never answered here."""
    return {s.item: Answer(SET, s.suggestion[4:]) for s in suspects
            if s.certain and not s.part_b and s.suggestion and s.suggestion.startswith("set:")}


def write_answers(path: str | Path, suspects, key: str, answers=None) -> None:
    """Write the answers file: one row per suspect, ``answer`` from *answers* (``{item: Answer}``; empty when
    missing)."""
    answers = answers or {}
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        out = csv.writer(f)
        out.writerow(COLUMNS)
        for s in suspects:
            ans = answers.get(s.item)
            out.writerow([s.item, key, s.kind, s.cls, s.parameter, s.entity, s.alternative, s.current,
                          ans.text() if ans else "", s.label, s.context.text(), s.archive_note, suggestion_cell(s)])


def free_path(path: str | Path) -> Path:
    """*path*, or the first of ``<stem>-1<suffix>``, ``-2``, … that does not exist (never overwrite)."""
    path = Path(path)
    n = 0
    while (p := path if n == 0 else path.with_name(f"{path.stem}-{n}{path.suffix}")).exists():
        n += 1
    return p


def _same_number(x: str, y: str) -> bool:
    try:
        a, b = (float(v.strip().replace(",", ".")) for v in (x, y))
    except ValueError:
        return False
    return math.isfinite(a) and a == b


@dataclass(frozen=True)
class ReadAnswers:
    answers: dict[str, Answer]
    stale: tuple[tuple[str, str], ...]  # (item, why)
    warnings: tuple[str, ...]


def read_answers(path: str | Path, a: vr.Analysis, suspects) -> ReadAnswers:
    """Read and check an answers file against the analysed snapshot's *suspects* (module docstring).  Raises
    :class:`AnswersError` for a file that cannot be used (encoding, columns, fingerprint, unknown or repeated item,
    an invalid answer)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise AnswersError(NOT_UTF8.format(path=path)) from None
    header = text.splitlines()[0] if text else ""
    names = {d: next(csv.reader([header], delimiter=d), []) for d in (",", ";")}
    delim = next((d for d, cols in names.items() if "item" in cols), None)
    if delim is None:
        raise AnswersError(f"{path} has no 'item' column: it is not a value-list answers file.")
    if missing := [c for c in COLUMNS[:9] if c not in names[delim]]:
        raise AnswersError(f"{path} lacks the column(s) {', '.join(missing)}.")
    by_item = {s.item: s for s in suspects}
    key, allowed = file_key(a), choices(a, suspects)
    answers, stale, warnings, seen, bad = {}, [], [], set(), []
    for row in csv.DictReader(io.StringIO(text, newline=""), delimiter=delim):
        cells = {k: (v or "") for k, v in row.items() if k is not None}
        if not any(v.strip() for v in cells.values()):
            continue  # an empty line Excel left
        item = cells["item"].strip()
        if cells["file_key"].strip() != key:
            raise AnswersError(f"{path} was made for another copy of the database (its key "
                               f"'{cells['file_key'].strip()}' is not this file's '{key}'): write a new one with "
                               "--dry-run --write-answers.")
        s = by_item.get(item)
        if s is None:
            raise AnswersError(f"{path}: '{item}' is not an item of this database.")
        if item in seen:
            raise AnswersError(f"{path}: '{item}' appears twice.")
        seen.add(item)
        now = {"kind": s.kind, "class": s.cls, "parameter": s.parameter, "current": s.current}
        changed = [c for c in _STRICT if cells[c] != now[c]]
        for col, value in (("entity", s.entity), ("alternative", s.alternative)):
            if cells[col] == value:
                continue
            if cells[col].strip() == value or _same_number(cells[col], value):
                warnings.append(f"{item}: {col} '{cells[col]}' read as '{value}' (changed by a spreadsheet)")
            else:
                changed.append(col)
        if changed:
            stale.append((item, "changed since the file was written: " + ", ".join(
                f"{c} '{cells[c]}' is now '{now.get(c, getattr(s, c, ''))}'" for c in changed)))
            continue
        try:
            ans = check_answer(s, cells["answer"], allowed[item])
        except AnswersError as exc:
            bad.append(str(exc))
            continue
        if ans is not None:
            answers[item] = ans
    if bad:
        raise AnswersError(f"{path}: " + " ".join(bad))
    return ReadAnswers(answers, tuple(stale), tuple(warnings))


@dataclass(frozen=True)
class Decision:
    suspect: vs.Suspect
    answer: Answer | None  # None: no answer (empty, missing or stale); the write leaves it as it is

    @property
    def action(self) -> str:
        return self.answer.action if self.answer else SKIP


@dataclass(frozen=True)
class Plan:
    """What a run writes.  *decisions* has one entry per suspect, in suspect order; *held* the items whose
    ``set:`` is not applied (the storage default while a storage question has no answer, :data:`STORAGE_HELD`);
    *restore* the members the write adds to R's list."""

    key: str
    restore: tuple[str, ...]
    decisions: tuple[Decision, ...]
    stale: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]
    held: tuple[str, ...]
    part_a_record: bool
    part_b_record: bool

    @property
    def answers(self) -> dict[str, Answer]:
        """The explicit answers (for the applied-answers CSV)."""
        return {d.suspect.item: d.answer for d in self.decisions if d.answer}

    @property
    def sets(self) -> tuple[Decision, ...]:
        """The ``set:`` decisions the write applies (held items left out)."""
        return tuple(d for d in self.decisions if d.action == SET and d.suspect.item not in self.held)

    def counts(self, part_b: bool) -> dict[str, int]:
        """``set`` / ``kept`` of one part (part B: ``set`` counts defaults, ``storage`` the ``fix_start`` writes;
        ``kept`` both), for the records and the summary."""
        mine = [d for d in self.decisions if d.suspect.part_b == part_b]
        applied = {d.suspect.item for d in self.sets}
        is_storage = [d.suspect.kind == ldr.STORAGE_KIND for d in mine]
        out = {"set": sum(d.suspect.item in applied for d, st in zip(mine, is_storage) if not st),
               "kept": sum(d.action == KEEP for d in mine)}
        if part_b:
            out["storage"] = sum(d.suspect.item in applied for d, st in zip(mine, is_storage) if st)
        return out


def make_plan(a: vr.Analysis, suspects, answers, *, stale=(), warnings=(), key: str | None = None) -> Plan:
    """The plan for the analysed snapshot from the explicit *answers* (``{item: Answer}``, from the prompt,
    :func:`accept_certain` or :func:`read_answers`; a suggestion is never one).  Each answer is checked again
    (:class:`AnswersError`)."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    if a.state == vr.UNCLEAR:
        raise AnswersError("The state is unclear: nothing can be written (use the checklist).")
    by_item = {s.item: s for s in suspects}
    if unknown := sorted(set(answers) - set(by_item)):
        raise AnswersError(f"Not an item of this database: {', '.join(unknown)}.")
    allowed = choices(a, suspects)
    stale_items = {item for item, _ in stale}
    decisions = []
    for s in suspects:
        ans = None if s.item in stale_items else answers.get(s.item)
        if ans is not None and ans.action == SET:  # check again: a prompt or a caller may pass any member
            ans = check_answer(s, ans.text(), allowed[s.item])
        decisions.append(Decision(s, ans))

    def done(part_b: bool) -> bool:
        return all(d.action in (KEEP, SET) for d in decisions if d.suspect.part_b == part_b)

    storage_open = any(d.action not in (KEEP, SET) for d in decisions if d.suspect.kind == ldr.STORAGE_KIND)
    held = tuple(d.suspect.item for d in decisions if storage_open and d.action == SET and d.suspect.part_b
                 and d.suspect.kind == vs.DEFAULT and (d.suspect.cls, d.suspect.parameter) == ldr.STORAGE)
    part_a = a.finding is not None and a.state in (vr.HIT, vr.HIT_UNANCHORED, vr.RENAMED_BACK, vr.NOT_HIT)
    reference = ldr.reference()
    empty_default = any(s.part_b or (s.kind == vs.DEFAULT and not s.current and (s.cls, s.parameter) in reference)
                        for s in suspects)
    part_b = a.version == FLEXTOOL_DB_VERSION and empty_default
    return Plan(key if key is not None else file_key(a), a.restore, tuple(decisions), tuple(stale), tuple(warnings),
                held, part_a and done(False), part_b and done(True))
