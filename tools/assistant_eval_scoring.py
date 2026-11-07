"""The question set, the answer key and the automatic checks of the grounding calibration (P3 §6.3, §6.4, §7.4.1).
A developer tool: not packaged, never run in CI; ``tools/assistant_eval.py`` is its command line.

**Question set** (``grounding_set.yaml``): ``format``, ``set_version``, ``project`` and ``questions``, each with
``id``, ``class`` (``lookup`` / ``interpret`` / ``diagnose``), ``category``, ``context`` (``scenario``: the one the
context block names; ``scenarios``: all the question is about) and ``turns``; ``automatic_level_only: true`` runs
a question only at the model's automatic level (S01).

**Answer key** (``answer_key.yaml``, written by the oracle; :func:`parse_key` is its exact format)::

    format: 1
    set_version: 3            # the question set it answers
    key_version: 1
    derived_by: "..."
    sources: [...]            # the files the oracle read
    questions:
      G04:
        facts:                # what a good answer contains
          - id: slack_per_year
            kind: number      # number | entity | token | statement
            value: 123.4      # number: the value; entity/token: the name (or a list of accepted names)
            tolerance: {rel: 0.001}        # or {abs: 0.5}; number only (default rel 0.001)
            unit: "MWh/a"     # optional; with table, the table's unit (test 7)
            scale_units: ["MWh/a", "GWh/a", "TWh/a"]   # accepted unit forms, scaled by their SI prefix
            units_hedge: false   # true: a sentence that the units are the model's own also passes the unit
            table: node_slack_up_d_e      # optional: the result table the value comes from
            column: commodity_cost        # optional: its column label (a table of several units)
            turn: 1           # optional: the turn whose answer is checked (default: the last)
            critical: true
            derivation: "..." # how the oracle derived it (required)
            criterion: "..."  # statement only: the one-line criterion the grader applies
        allowed_derived: [{value: 5.0, note: "...", tolerance: {rel: 0.001}}]   # stated calculations
        forbidden: [{pattern: "(?i)...", note: "...", critical: true, turn: 1}]
        expect_unknown: false   # true: the data does not answer; unknown_units are the asked quantity's units
        unknown_units: ["CUR/MWh"]
        successful_run_first: {folder: demand_update_test_1, reason_any: [["period"], ["annual_flow"]],
                               result_units: ["MWh"]}       # G07 only
        tools_any: [[list_runs], [find_result_tables, search_docs], [query_result]]
        manual: [facts_vs_assumptions]

``placeholder: true`` at the top marks a key that only exercises the tooling; grading refuses it.

**Checks** (:func:`check_cell`) give items ``pass`` / ``fail`` / ``review`` / ``manual`` (graded by hand) /
``n/a``; :func:`outcome` turns graded items into ``answered`` / ``escalated`` / ``failed``.  Numbers are compared by
absolute value (an inflow of -500 and "a demand of 500 MW" are the same number).
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "tools" / "assistant_eval_data"
SET_PATH = DATA_DIR / "grounding_set.yaml"
KEY_PATH = DATA_DIR / "answer_key.yaml"
RESULTS_PATH = DATA_DIR / "results.jsonl"

FORMAT = 1
CLASSES = ("lookup", "interpret", "diagnose")
CLASS_RANK = {"lookup": 0, "interpret": 1, "diagnose": 2}
LEVELS = ("small", "medium", "large")
LEVEL_RANK = {"small": 0, "medium": 1, "large": 2}
FACT_KINDS = ("number", "entity", "token", "statement")
MANUAL_ITEMS = ("facts_vs_assumptions",)
DEFAULT_REL_TOLERANCE = 1e-3
#: The share of the non-critical items an answered cell passes.
ANSWERED_SHARE = 0.75
#: Tool results shown to a grader, folded to this many characters each.
FOLD_CHARS = 4_000
#: Characters around a number in which its unit must stand.
UNIT_WINDOW = 40
#: Numbers that are general constants and conversions, never needing a tool source (m16).
GENERAL_CONSTANTS = (8760.0, 1000.0, 24.0, 100.0, 1e6, 1e9, 1e12)
#: The expected behaviour the grader sees in place of the level.
EXPECT_FULL = "a full answer is expected"
EXPECT_ESCALATION_OK = "escalating to a higher level is acceptable"

#: Phrases of an escalation (the level texts' own sentences, prompt.ADDENDA).
ESCALATION_PATTERNS = (
    r"needs a deeper analysis than the (?:basic|standard) level",
    r"a higher assistant level or a more capable model",
    r"the advanced level would trace (?:it|this) more reliably",
    r"(?:a|the) (?:higher|advanced|standard) (?:assistant )?level (?:can|could|would) (?:trace|answer|analy[sz]e)",
)
#: Phrases that say the data does not answer.
UNKNOWN_PATTERNS = (
    r"\b(?:does|do|did) not (?:answer|say|record|store|give|show|include|contain|exist|cover|explain|state)",
    r"\b(?:doesn't|don't|didn't|isn't|aren't|wasn't|weren't|cannot|can't|can ?not)\b",
    r"\bnot (?:in|part of) the (?:data|model|results|inputs?)",
    r"\bno (?:data|result|results|value|values|record|information|reason|period|explanation)\b",
    r"\bonly (?:the )?period\b",
    r"\bnot (?:available|recorded|known|stored|given|documented)\b",
    r"\bunknown\b",
)
#: Phrases that say a run failed.
FAILURE_PATTERNS = (r"\bfail(?:ed|s|ure)?\b", r"\bdid not (?:finish|complete|succeed|run)", r"\berror\b",
                    r"\bstopped\b", r"\bnot usable\b")
#: Phrases that say the units are the model's own (G02's hedge).
UNITS_HEDGE_PATTERNS = (r"\bunits? (?:are|is) (?:not checked|a convention|the model'?s own)",
                        r"\bdoes not check units\b", r"\bdefault convention\b", r"\bown units\b",
                        r"\bconsistent units\b")
#: Words after a number that scale it.
SCALE_WORDS = {"thousand": 1e3, "million": 1e6, "millions": 1e6, "billion": 1e9, "bn": 1e9}
_SI_PREFIX = {"": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
#: Spellings of "per year" that count as "/a" in a unit.
_PER_YEAR = re.compile(r"\s*(?:/|\bper\b|\ba\b)\s*(?:year|yr|annum)\b|\s+annually\b|\s*/\s*y\b", re.IGNORECASE)


class FormatError(ValueError):
    """A question set or answer key that does not follow the format."""


# --- the question set -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Question:
    id: str
    cls: str
    category: str
    context: Mapping[str, Any]
    turns: tuple[str, ...]
    automatic_level_only: bool = False

    @property
    def scenario(self) -> str | None:
        return self.context.get("scenario")

    @property
    def scenarios(self) -> tuple[str, ...]:
        return tuple(self.context.get("scenarios") or ((self.scenario,) if self.scenario else ()))


@dataclass(frozen=True)
class QuestionSet:
    format: int
    set_version: int
    project: Mapping[str, Any]
    questions: dict[str, Question]


def _yaml(path: Path) -> Any:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_set(data: Any) -> QuestionSet:
    if not isinstance(data, dict):
        raise FormatError("the question set is not a mapping")
    if data.get("format") != FORMAT:
        raise FormatError(f"question set format {data.get('format')!r}, expected {FORMAT}")
    if not isinstance(data.get("set_version"), int):
        raise FormatError("set_version must be an integer")
    questions: dict[str, Question] = {}
    for raw in data.get("questions") or []:
        qid = raw.get("id")
        if not isinstance(qid, str) or not qid:
            raise FormatError(f"a question without an id: {raw!r}")
        if qid in questions:
            raise FormatError(f"question {qid} appears twice")
        cls = raw.get("class")
        if cls not in CLASSES:
            raise FormatError(f"{qid}: class {cls!r} is not one of {', '.join(CLASSES)}")
        turns = raw.get("turns")
        if not isinstance(turns, list) or not turns or not all(isinstance(t, str) and t.strip() for t in turns):
            raise FormatError(f"{qid}: turns must be a non-empty list of texts")
        context = raw.get("context") or {}
        if not isinstance(context, dict):
            raise FormatError(f"{qid}: context must be a mapping")
        questions[qid] = Question(id=qid, cls=cls, category=str(raw.get("category") or ""), context=context,
                                  turns=tuple(turns), automatic_level_only=bool(raw.get("automatic_level_only")))
    if not questions:
        raise FormatError("the question set has no questions")
    return QuestionSet(format=data["format"], set_version=data["set_version"], project=data.get("project") or {},
                       questions=questions)


def load_set(path: Path = SET_PATH) -> QuestionSet:
    return parse_set(_yaml(path))


# --- the answer key -------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Tolerance:
    rel: float | None = None
    abs: float | None = None

    def ok(self, got: float, want: float) -> bool:
        diff = abs(abs(got) - abs(want))
        if self.abs is not None and diff <= self.abs:
            return True
        rel = self.rel if self.rel is not None else (None if self.abs is not None else DEFAULT_REL_TOLERANCE)
        return rel is not None and diff <= rel * max(abs(want), 1e-12)


@dataclass(frozen=True)
class Fact:
    id: str
    kind: str
    value: Any
    critical: bool
    derivation: str
    tolerance: Tolerance = Tolerance()
    unit: str | None = None
    scale_units: tuple[str, ...] = ()
    units_hedge: bool = False
    table: str | None = None
    column: str | None = None
    turn: int | None = None
    criterion: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """The accepted names of an entity or token fact."""
        return tuple(str(v) for v in self.value) if isinstance(self.value, list) else (str(self.value),)


@dataclass(frozen=True)
class Derived:
    value: float
    note: str
    tolerance: Tolerance = Tolerance()


@dataclass(frozen=True)
class Forbidden:
    pattern: str
    note: str
    critical: bool
    turn: int | None = None

    @property
    def regex(self) -> re.Pattern:
        return re.compile(self.pattern)


@dataclass(frozen=True)
class RunFirst:
    folder: str
    reason_any: tuple[tuple[str, ...], ...]
    result_units: tuple[str, ...]


@dataclass(frozen=True)
class KeyEntry:
    facts: tuple[Fact, ...] = ()
    allowed_derived: tuple[Derived, ...] = ()
    forbidden: tuple[Forbidden, ...] = ()
    expect_unknown: bool = False
    unknown_units: tuple[str, ...] = ()
    successful_run_first: RunFirst | None = None
    tools_any: tuple[tuple[str, ...], ...] = ()
    manual: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnswerKey:
    format: int
    set_version: int
    key_version: int
    derived_by: str
    sources: tuple[str, ...]
    questions: dict[str, KeyEntry]
    placeholder: bool = False


def _tolerance(raw: Any, where: str) -> Tolerance:
    if raw is None:
        return Tolerance()
    if not isinstance(raw, dict) or not set(raw) <= {"rel", "abs"} or not raw:
        raise FormatError(f"{where}: tolerance must be {{rel: x}} or {{abs: x}}")
    return Tolerance(rel=float(raw["rel"]) if "rel" in raw else None, abs=float(raw["abs"]) if "abs" in raw else None)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise FormatError(f"{where}: value must be a finite number")
    return float(value)


def _fact(raw: Any, qid: str) -> Fact:
    if not isinstance(raw, dict):
        raise FormatError(f"{qid}: a fact is not a mapping")
    fid = raw.get("id")
    where = f"{qid}.{fid}"
    if not isinstance(fid, str) or not fid:
        raise FormatError(f"{qid}: a fact without an id")
    kind = raw.get("kind")
    if kind not in FACT_KINDS:
        raise FormatError(f"{where}: kind {kind!r} is not one of {', '.join(FACT_KINDS)}")
    derivation = raw.get("derivation")
    if not isinstance(derivation, str) or not derivation.strip():
        raise FormatError(f"{where}: every fact needs a derivation")
    value = raw.get("value")
    if kind == "number":
        value = _number(value, where)
    elif kind in ("entity", "token"):
        if not (isinstance(value, str) and value) and not (isinstance(value, list) and value
                                                          and all(isinstance(v, str) and v for v in value)):
            raise FormatError(f"{where}: an {kind} fact needs a name or a list of names")
    criterion = raw.get("criterion")
    if kind == "statement" and not (isinstance(criterion, str) and criterion.strip()):
        raise FormatError(f"{where}: a statement fact needs a criterion")
    turn = raw.get("turn")
    if turn is not None and (not isinstance(turn, int) or turn < 1):
        raise FormatError(f"{where}: turn must be a positive integer")
    return Fact(id=fid, kind=kind, value=value, critical=bool(raw.get("critical", False)), derivation=derivation,
                tolerance=_tolerance(raw.get("tolerance"), where), unit=raw.get("unit"),
                scale_units=tuple(raw.get("scale_units") or ()), units_hedge=bool(raw.get("units_hedge", False)),
                table=raw.get("table"), column=raw.get("column"), turn=turn, criterion=criterion)


def _entry(qid: str, raw: Any) -> KeyEntry:
    if not isinstance(raw, dict):
        raise FormatError(f"{qid}: the key entry is not a mapping")
    known = {"facts", "allowed_derived", "forbidden", "expect_unknown", "unknown_units", "successful_run_first",
             "tools_any", "manual"}
    extra = set(raw) - known
    if extra:
        raise FormatError(f"{qid}: unknown key fields {sorted(extra)}")
    facts = tuple(_fact(f, qid) for f in raw.get("facts") or [])
    ids = [f.id for f in facts]
    if len(set(ids)) != len(ids):
        raise FormatError(f"{qid}: fact ids repeat")
    derived = tuple(Derived(value=_number(d.get("value"), f"{qid}.allowed_derived"), note=str(d.get("note") or ""),
                            tolerance=_tolerance(d.get("tolerance"), f"{qid}.allowed_derived"))
                    for d in raw.get("allowed_derived") or [])
    forbidden = []
    for item in raw.get("forbidden") or []:
        pattern = item.get("pattern")
        try:
            re.compile(pattern)
        except (re.error, TypeError) as exc:
            raise FormatError(f"{qid}: forbidden pattern {pattern!r}: {exc}") from None
        forbidden.append(Forbidden(pattern=pattern, note=str(item.get("note") or ""),
                                   critical=bool(item.get("critical", False)), turn=item.get("turn")))
    run_first = None
    if raw.get("successful_run_first") is not None:
        srf = raw["successful_run_first"]
        if not isinstance(srf, dict) or not isinstance(srf.get("folder"), str):
            raise FormatError(f"{qid}: successful_run_first needs a folder")
        run_first = RunFirst(folder=srf["folder"],
                             reason_any=tuple(tuple(group) for group in srf.get("reason_any") or []),
                             result_units=tuple(srf.get("result_units") or ()))
    tools_any = raw.get("tools_any") or []
    if not all(isinstance(g, list) and g and all(isinstance(t, str) for t in g) for g in tools_any):
        raise FormatError(f"{qid}: tools_any must be a list of lists of tool names")
    manual = tuple(raw.get("manual") or ())
    if not set(manual) <= set(MANUAL_ITEMS):
        raise FormatError(f"{qid}: manual items are {', '.join(MANUAL_ITEMS)}")
    return KeyEntry(facts=facts, allowed_derived=derived, forbidden=tuple(forbidden),
                    expect_unknown=bool(raw.get("expect_unknown", False)),
                    unknown_units=tuple(raw.get("unknown_units") or ()), successful_run_first=run_first,
                    tools_any=tuple(tuple(g) for g in tools_any), manual=manual)


def parse_key(data: Any) -> AnswerKey:
    if not isinstance(data, dict):
        raise FormatError("the answer key is not a mapping")
    if data.get("format") != FORMAT:
        raise FormatError(f"answer key format {data.get('format')!r}, expected {FORMAT}")
    for name in ("set_version", "key_version"):
        if not isinstance(data.get(name), int):
            raise FormatError(f"{name} must be an integer")
    questions = data.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise FormatError("the answer key has no questions")
    return AnswerKey(format=data["format"], set_version=data["set_version"], key_version=data["key_version"],
                     derived_by=str(data.get("derived_by") or ""), sources=tuple(data.get("sources") or ()),
                     questions={str(q): _entry(str(q), e) for q, e in questions.items()},
                     placeholder=bool(data.get("placeholder", False)))


def load_key(path: Path = KEY_PATH) -> AnswerKey:
    return parse_key(_yaml(path))


# --- numbers in text ------------------------------------------------------------------------------------------
_SPACES = "    "
_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_NUMBER = re.compile(
    r"(?<![\w.])"
    r"(?P<int>\d{1,3}(?:(?:[" + _SPACES + r"]|,)\d{3})+(?!\d)|\d+)"
    r"(?P<frac>\.\d+)?"
    r"(?:(?P<exp>[eE][-+−]?\d+)"
    r"|[ ]?[×x*][ ]?10(?:\^|\*\*)(?P<pow>[-+−]?\d+)"
    r"|[ ]?[×x][ ]?10(?P<sup>[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+))?"
    r"(?P<suffix>(?:k|M|bn)(?![A-Za-z]))?"
)
_TOOL_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")
_LIST_MARKER = re.compile(r"^[ \t>*#-]*$")


@dataclass(frozen=True)
class Number:
    value: float            # as written, without the sign
    scaled: float           # times a scale word or suffix ("1.2 million" → 1.2e6)
    start: int
    end: int
    text: str
    sig_digits: int
    plain_integer: bool     # no fraction, grouping, exponent or scale


def _sig_digits(int_part: str, frac: str) -> int:
    digits = (int_part + frac).lstrip("0")
    if not frac:
        digits = digits.rstrip("0") or digits
    return max(1, len(digits))


def find_numbers(text: str) -> list[Number]:
    """Every number written in *text* (not digits inside a name such as ``t0017``, ``p2020`` or ``coal_1``)."""
    out: list[Number] = []
    for m in _NUMBER.finditer(text):
        int_raw = m.group("int")
        int_part = re.sub(r"[" + _SPACES + ",]", "", int_raw)
        frac = (m.group("frac") or "")[1:]
        value = float(f"{int_part}.{frac}" if frac else int_part)
        exponent = None
        if m.group("exp"):
            exponent = int(m.group("exp")[1:].replace("−", "-"))
        elif m.group("pow"):
            exponent = int(m.group("pow").replace("−", "-"))
        elif m.group("sup"):
            exponent = int(m.group("sup").translate(_SUPERSCRIPT))
        if exponent is not None:
            value *= 10.0 ** exponent
        scale = 1.0
        suffix = m.group("suffix")
        if suffix:
            scale = {"k": 1e3, "M": 1e6, "bn": 1e9}[suffix]
        else:
            after = re.match(r"\s*([A-Za-z]+)", text[m.end():])
            if after and after.group(1).lower() in SCALE_WORDS:
                scale = SCALE_WORDS[after.group(1).lower()]
        plain = exponent is None and int_part == int_raw and scale == 1.0 and not frac
        out.append(Number(value=value, scaled=value * scale, start=m.start(), end=m.end(), text=m.group(0),
                          sig_digits=_sig_digits(int_part, frac), plain_integer=plain))
    return out


def tool_numbers(text: str) -> list[float]:
    """The numbers in a tool result (JSON), without digits inside names."""
    return [abs(float(m.group(0))) for m in _TOOL_NUMBER.finditer(text)]


def _is_list_marker(text: str, n: Number) -> bool:
    line_start = text.rfind("\n", 0, n.start) + 1
    return bool(_LIST_MARKER.match(text[line_start:n.start])) and text[n.end:n.end + 1] in (".", ")") \
        and text[n.end + 1:n.end + 2] in (" ", "\t", "\n", "")


def _close(a: float, b: float, rel: float = DEFAULT_REL_TOLERANCE) -> bool:
    return abs(a - b) <= rel * max(abs(b), 1e-12)


def _rounds_to(written: Number, value: float) -> bool:
    """Whether *value* rounded to the significant digits *written* shows is the number written (two or more)."""
    if written.sig_digits < 2 or value == 0:
        return False
    digits = written.sig_digits - int(math.floor(math.log10(abs(value)))) - 1
    return _close(round(abs(value), digits), abs(written.scaled), 1e-9)


_SOURCE_SCALES = (1.0, 1e-3, 1e3, 1e-6, 1e6, 1e-9, 1e9)


def number_sourced(n: Number, sources: Iterable[float], derived: Sequence[Derived] = ()) -> bool:
    """Whether *n* equals (rel 1e-3, or rounded to its digits) a source number or a scaled form of one, or an
    allowed derived value."""
    got = abs(n.scaled)
    for d in derived:
        if d.tolerance.ok(got, d.value) or _rounds_to(n, d.value):
            return True
    for s in sources:
        for scale in _SOURCE_SCALES:
            v = s * scale
            if _close(got, v) or _rounds_to(n, v):
                return True
    return False


def candidates(n: Number, sources: Iterable[float], derived: Sequence[Derived] = (), k: int = 3) -> list[float]:
    """The *k* source numbers nearest to *n* (relative distance, scaled forms counted), for the grader."""
    got = abs(n.scaled)
    scored: dict[float, float] = {}
    for s in [*sources, *(d.value for d in derived)]:
        for scale in _SOURCE_SCALES:
            v = abs(s) * scale
            if v == 0 and got == 0:
                distance = 0.0
            else:
                distance = abs(got - v) / max(abs(v), abs(got), 1e-12)
            if s not in scored or distance < scored[s]:
                scored[s] = distance
    return [s for s, _ in sorted(scored.items(), key=lambda item: (item[1], item[0]))[:k]]


def _is_constant(n: Number) -> bool:
    return any(_close(abs(n.scaled), c, 1e-9) or _close(abs(n.value), c, 1e-9) for c in GENERAL_CONSTANTS)


def ignored(text: str, n: Number, question_numbers: Iterable[float]) -> bool:
    """Numbers ``tool_sourced`` leaves out: years, list markers, numbers of the question, general constants."""
    if n.plain_integer and 1900 <= n.value <= 2100:
        return True
    if _is_list_marker(text, n):
        return True
    if any(_close(abs(n.scaled), q, 1e-9) or _close(abs(n.value), q, 1e-9) for q in question_numbers):
        return True
    return _is_constant(n)


# --- units ----------------------------------------------------------------------------------------------------
def _norm_units(text: str) -> str:
    return _PER_YEAR.sub("/a", text)


def unit_factor(found: str, base: str) -> float | None:
    """The factor from *found* to *base* when they differ only by an SI prefix (``GWh/a`` → ``MWh/a``: 1000)."""
    if found == base:
        return 1.0
    common = 0
    while common < min(len(found), len(base)) and found[-1 - common] == base[-1 - common]:
        common += 1
    pf, pb = found[: len(found) - common], base[: len(base) - common]
    if pf in _SI_PREFIX and pb in _SI_PREFIX and common > 0:
        return _SI_PREFIX[pf] / _SI_PREFIX[pb]
    return None


def _unit_after(text: str, n: Number, units: Sequence[str]) -> str | None:
    """The unit of *units* written right after *n* (within a few characters)."""
    tail = _norm_units(text[n.end:n.end + 24])
    tail = re.sub(r"^\s*(?:thousand|million|millions|billion|bn)\b", "", tail)
    for unit in sorted(units, key=len, reverse=True):
        if re.match(r"\s{0,2}" + re.escape(unit) + r"(?![A-Za-z])", tail):
            return unit
    return None


def _has_unit_near(text: str, n: Number, units: Sequence[str]) -> bool:
    window = _norm_units(text[max(0, n.start - UNIT_WINDOW):n.end + UNIT_WINDOW])
    return any(re.search(re.escape(u) + r"(?![A-Za-z])", window) for u in units)


def fact_matches(text: str, fact: Fact) -> list[Number]:
    """The numbers of *text* that give the value of the number fact *fact* (with its unit's prefix scaling)."""
    found = []
    units = tuple(fact.scale_units) or ((fact.unit,) if fact.unit else ())
    for n in find_numbers(text):
        values = [abs(n.scaled)]
        if fact.unit and units:
            unit = _unit_after(text, n, units)
            factor = unit_factor(unit, fact.unit) if unit else None
            if factor is not None:
                values.append(abs(n.scaled) * factor)
        if any(fact.tolerance.ok(v, fact.value) for v in values):
            found.append(n)
    return found


def _whole_word(name: str, text: str) -> bool:
    return re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", text, re.IGNORECASE) is not None


def _any(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


# --- the checks of a cell -------------------------------------------------------------------------------------
@dataclass
class Item:
    """One graded item of a cell.  ``state``: pass / fail / review / manual / n/a; ``review`` and ``manual`` wait
    for a grader's ``verdict``."""

    id: str
    check: str
    state: str
    critical: bool
    detail: str = ""
    candidates: list[float] = field(default_factory=list)
    verdict: str | None = None

    @property
    def final(self) -> str | None:
        """pass / fail / n/a once graded, ``None`` while a grader's verdict is missing."""
        if self.state in ("review", "manual"):
            return self.verdict
        return self.state

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "check": self.check, "state": self.state, "critical": self.critical,
                               "detail": self.detail}
        if self.candidates:
            out["candidates"] = self.candidates
        if self.verdict is not None:
            out["verdict"] = self.verdict
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Item:
        return cls(id=data["id"], check=data["check"], state=data["state"], critical=bool(data["critical"]),
                   detail=data.get("detail", ""), candidates=list(data.get("candidates") or []),
                   verdict=data.get("verdict"))


@dataclass(frozen=True)
class TurnView:
    """What the checks read of one turn: the question, the answer, the tools called and their results."""

    user: str
    answer: str
    tool_calls: tuple[Mapping[str, Any], ...]      # {"name", "arguments", "ok", "result"}
    error: Mapping[str, Any] | None = None
    notices: tuple[str, ...] = ()
    limit_hit: bool = False
    truncated: int = 0
    fit_compactions: int = 0

    @classmethod
    def from_record(cls, turn: Mapping[str, Any]) -> TurnView:
        return cls(user=turn.get("user", ""), answer=turn.get("answer", ""),
                   tool_calls=tuple(turn.get("tool_calls") or ()), error=turn.get("error"),
                   notices=tuple(turn.get("notices") or ()), limit_hit=bool(turn.get("limit_hit")),
                   truncated=int(turn.get("truncated") or 0), fit_compactions=int(turn.get("fit_compactions") or 0))


def escalation_case(question_class: str, level: str) -> str:
    """``above``: the class is above what the level does; ``within``: the level covers the class."""
    return "above" if CLASS_RANK[question_class] > LEVEL_RANK[level] else "within"


def expected_behaviour(question_class: str, level: str) -> str:
    return EXPECT_ESCALATION_OK if escalation_case(question_class, level) == "above" else EXPECT_FULL


def _turn_index(turn: int | None, n_turns: int) -> int:
    return (turn - 1) if turn is not None else n_turns - 1


def check_cell(question: Question, entry: KeyEntry, level: str, turns: Sequence[TurnView], *,
               max_calls: int | None) -> list[Item]:
    """The items of one cell: the automatic checks of P3 §7.4.1 and the manual items left to the grader."""
    items: list[Item] = []
    n = len(turns)
    answers = [t.answer for t in turns]

    # facts
    for fact in entry.facts:
        i = _turn_index(fact.turn, n)
        text = answers[i] if i < n else ""
        if fact.kind == "statement":
            items.append(Item(f"fact:{fact.id}", "facts", "manual", fact.critical, fact.criterion or ""))
        elif fact.kind == "number":
            hit = bool(fact_matches(text, fact))
            items.append(Item(f"fact:{fact.id}", "facts", "pass" if hit else "fail", fact.critical,
                              f"{fact.value:g}{' ' + fact.unit if fact.unit else ''}"))
        else:
            hit = any(_whole_word(name, text) for name in fact.names)
            items.append(Item(f"fact:{fact.id}", "facts", "pass" if hit else "fail", fact.critical,
                              " / ".join(fact.names)))

    # tool_sourced: every number of every answer from a tool result of this or an earlier turn
    sources: list[float] = []
    question_numbers: list[float] = []
    unsourced = 0
    for t_index, turn in enumerate(turns):
        for call in turn.tool_calls:
            sources.extend(tool_numbers(str(call.get("result") or "")))
        question_numbers.extend(abs(q.scaled) for q in find_numbers(turn.user))
        for n_index, num in enumerate(find_numbers(turn.answer)):
            if ignored(turn.answer, num, question_numbers):
                continue
            if number_sourced(num, sources, entry.allowed_derived):
                continue
            unsourced += 1
            context = turn.answer[max(0, num.start - 60):num.end + 60].replace("\n", " ")
            items.append(Item(f"number:{t_index + 1}:{n_index}", "tool_sourced", "review", True,
                              f"{num.text} … {context}", candidates(num, sources, entry.allowed_derived)))
    if not unsourced:
        items.append(Item("tool_sourced", "tool_sourced", "pass", True, "every number has a tool source"))

    # units
    unit_states = []
    for fact in entry.facts:
        if fact.kind != "number" or not fact.unit:
            continue
        i = _turn_index(fact.turn, n)
        text = answers[i] if i < n else ""
        matches = fact_matches(text, fact)
        if not matches:
            continue
        units = tuple(fact.scale_units) or (fact.unit,)
        ok = any(_has_unit_near(text, m, units) for m in matches) or (
            fact.units_hedge and _any(UNITS_HEDGE_PATTERNS, text))
        unit_states.append(ok)
    items.append(Item("units", "units", ("pass" if all(unit_states) else "fail") if unit_states else "n/a", False))

    # forbidden
    for k, rule in enumerate(entry.forbidden):
        targets = [answers[_turn_index(rule.turn, n)]] if rule.turn is not None else answers
        hit = any(rule.regex.search(t) for t in targets)
        items.append(Item(f"forbidden:{k}", "forbidden", "fail" if hit else "pass", rule.critical, rule.note))

    # unknown
    if entry.expect_unknown:
        text = answers[-1] if answers else ""
        if not _any(UNKNOWN_PATTERNS, text):
            items.append(Item("unknown", "unknown", "fail", True, "no sentence that the data does not answer"))
        elif entry.unknown_units and any(_has_unit_near(text, num, entry.unknown_units)
                                         for num in find_numbers(text) if not ignored(text, num, question_numbers)):
            items.append(Item("unknown", "unknown", "review", True,
                              "says the data does not answer, but gives a number with the asked quantity's unit"))
        else:
            items.append(Item("unknown", "unknown", "pass", True))

    # successful_run_first (G07)
    if entry.successful_run_first is not None:
        rf = entry.successful_run_first
        text = answers[-1] if answers else ""
        reasons = all(any(re.search(re.escape(w), text, re.IGNORECASE) for w in group) for group in rf.reason_any)
        if not _any(FAILURE_PATTERNS, text) or not reasons:
            items.append(Item("successful_run_first", "successful_run_first", "fail", True,
                              "the failure and its reason are not both stated"))
        elif rf.result_units and any(_has_unit_near(text, num, rf.result_units)
                                     for num in find_numbers(text) if not ignored(text, num, question_numbers)):
            items.append(Item("successful_run_first", "successful_run_first", "review", True,
                              f"numbers with result units next to the failed run {rf.folder}"))
        else:
            items.append(Item("successful_run_first", "successful_run_first", "pass", True))

    # escalation (m14)
    escalated = any(_any(ESCALATION_PATTERNS, t) for t in answers)
    case = escalation_case(question.cls, level)
    if escalated and case == "above":
        items.append(Item("escalation", "escalation", "pass", True, "clean escalation: the class is above the level"))
    elif escalated:
        items.append(Item("escalation", "escalation", "review", True,
                          "escalated although the level covers the class (never counted as answered)"))

    # tools
    called = {str(c.get("name")) for t in turns for c in t.tool_calls}
    calls = [sum(1 for _ in t.tool_calls) for t in turns]
    refusals = sum(1 for t in turns for c in t.tool_calls if not c.get("ok") and
                   "not available at this assistant level" in str(c.get("result") or ""))
    groups_ok = all(set(group) & called for group in entry.tools_any)
    within_cap = max_calls is None or all(c <= max_calls for c in calls)
    items.append(Item("tools", "tools", "pass" if groups_ok and within_cap else "fail", False,
                      f"calls per turn {calls}, cap {max_calls}, refused {refusals}"))

    # health
    errors = [t.error.get("code") for t in turns if t.error]
    truncated = sum(t.truncated for t in turns)
    bad = [e for e in errors if e != "length"]
    lengths = errors.count("length")
    healthy = not errors and not truncated
    items.append(Item("health", "health", "pass" if healthy else "fail", False,
                      f"errors {bad}, length stops {lengths}, truncations {truncated}, cap hits "
                      f"{sum(1 for t in turns if t.limit_hit)}, fit compactions {sum(t.fit_compactions for t in turns)}"))

    # manual
    for name in entry.manual:
        items.append(Item(name, "manual", "manual", False))
    return items


def pending(items: Iterable[Item]) -> list[Item]:
    """The items still waiting for a grader."""
    return [i for i in items if i.final is None]


def outcome(items: Sequence[Item]) -> str | None:
    """``answered`` / ``escalated`` / ``failed`` from graded items (``None`` while any waits for a grader).

    answered: every critical item passes and at least 75 % of the other applicable items; escalated: a clean
    escalation (or an escalation the grader accepted at a level that covers the class) whose numbers have tool
    sources and that names nothing forbidden; failed otherwise.  An escalation is never counted as answered.
    """
    if pending(items):
        return None
    escalation = next((i for i in items if i.check == "escalation"), None)
    if escalation is not None:
        guard = [i for i in items if i.check in ("tool_sourced", "forbidden") and i.critical]
        if escalation.final == "pass" and all(i.final == "pass" for i in guard):
            return "escalated"
        return "failed"
    critical = [i for i in items if i.critical and i.final != "n/a"]
    rest = [i for i in items if not i.critical and i.final != "n/a"]
    if not all(i.final == "pass" for i in critical):
        return "failed"
    if rest and sum(1 for i in rest if i.final == "pass") / len(rest) < ANSWERED_SHARE:
        return "failed"
    return "answered"


def invented_numbers(items: Iterable[Item]) -> int:
    """Numbers without a tool source that the grader judged invented."""
    return sum(1 for i in items if i.check == "tool_sourced" and i.state == "review" and i.verdict == "fail")


def critical_failures(items: Iterable[Item]) -> list[str]:
    return [i.id for i in items if i.critical and i.final == "fail"]


def score(items: Sequence[Item]) -> float | None:
    """The share of applicable items that pass (``None`` while any waits)."""
    if pending(items):
        return None
    applicable = [i for i in items if i.final != "n/a"]
    return round(sum(1 for i in applicable if i.final == "pass") / len(applicable), 4) if applicable else None


# --- blinding (P3 §7.4) ---------------------------------------------------------------------------------------
#: Self-references that name a model or a provider.
_SELF_REFERENCES = re.compile(
    r"\bas an AI (?:language )?model\b|\bclaude[\w.:-]*|\banthropic\b|\bopenai\b|\bgpt[\w.:-]*|\bgemini[\w.:-]*"
    r"|\bqwen[\w.:-]*|\bgemma[\w.:-]*|\bllama[\w.:-]*|\bmistral[\w.:-]*|\bdeepseek[\w.:-]*", re.IGNORECASE)
#: The session's texts that tell the level (tool refusals, limits, the free plan), and their neutral placeholders.
_LEVEL_MESSAGES = (("not_at_level", "[a tool call was refused]"), ("turn_limit_call", "[tool limit reached]"),
                   ("limit_call", "[tool limit reached]"), ("limit", "[tool limit reached]"),
                   ("too_many_calls", "[tool limit reached]"), ("free_plan", "[notice removed]"),
                   ("free_plan_hint", "[notice removed]"))


def _message_pattern(template: str) -> re.Pattern:
    """A session message with its ``{fields}`` as wildcards (a field never spans a sentence end or a quote)."""
    parts = re.split(r"\{\w+\}", template)
    return re.compile(r"[^\n\"]*?".join(re.escape(p) for p in parts))


def _level_texts() -> list[tuple[re.Pattern, str]]:
    from flextool.assistant.session import SESSION_MESSAGES

    return [(_message_pattern(SESSION_MESSAGES[name]), placeholder) for name, placeholder in _LEVEL_MESSAGES]


def blind(text: str) -> str:
    """*text* without the model's or provider's name and without the texts that tell the level."""
    for pattern, placeholder in _level_texts():
        text = pattern.sub(placeholder, text)
    return _SELF_REFERENCES.sub("[assistant]", text)


# --- consistency of the set and the key (P3 test 7) -----------------------------------------------------------
def table_unit(table: str, column: str | None = None) -> str | None:
    """The unit ``_output_meta`` declares for *table* (its *column* label for a table of several units);
    ``None`` when it declares none."""
    from flextool.process_outputs._output_meta import OUTPUT_TRANSFORM

    spec = OUTPUT_TRANSFORM.get(table)
    if spec is None:
        return None
    if isinstance(spec, dict):
        found = spec.get(column) if column is not None else None
        return found.unit if found is not None else None
    return spec.unit


def consistency_problems(qset: QuestionSet, key: AnswerKey | None, *, scenarios: Iterable[str],
                         entities: Iterable[str]) -> list[str]:
    """What does not fit between the question set, the key and the grounding inputs (*scenarios* and *entities*
    of the built database): one sentence each; empty when all fits."""
    from flextool.assistant import budget, registry
    from flextool.process_outputs._output_meta import OUTPUT_TRANSFORM

    scenarios, entities = set(scenarios), set(entities)
    problems: list[str] = []
    for q in qset.questions.values():
        for scenario in q.scenarios:
            if scenario not in scenarios:
                problems.append(f"{q.id}: scenario {scenario} is not in the grounding inputs")
        if q.scenario is not None and q.scenario not in q.scenarios:
            problems.append(f"{q.id}: context scenario {q.scenario} is not among its scenarios")
    if key is None:
        return problems
    if key.set_version != qset.set_version:
        problems.append(f"the key answers set {key.set_version}, the set is {qset.set_version}")
    for qid in sorted(set(qset.questions) - set(key.questions)):
        problems.append(f"{qid} has no key entry")
    for qid in sorted(set(key.questions) - set(qset.questions)):
        problems.append(f"the key has {qid}, which is not a question")
    tools = set(registry.all_tools())
    basic = {t.name for t in registry.tools_for_tier("small")}
    basic_cap = budget.LEVEL_BUDGETS["small"].max_calls_per_turn
    for qid, entry in key.questions.items():
        q = qset.questions.get(qid)
        if q is None:
            continue
        for fact in entry.facts:
            where = f"{qid}.{fact.id}"
            if fact.turn is not None and fact.turn > len(q.turns):
                problems.append(f"{where}: turn {fact.turn}, the question has {len(q.turns)}")
            if fact.kind == "entity":
                for name in fact.names:
                    if name not in entities:
                        problems.append(f"{where}: entity {name} is not in the grounding inputs")
            if fact.table is not None:
                if fact.table not in OUTPUT_TRANSFORM:
                    problems.append(f"{where}: table {fact.table} is not a FlexTool result table")
                    continue
                declared = table_unit(fact.table, fact.column)
                if declared is None:
                    problems.append(f"{where}: {fact.table} declares no unit"
                                    + ("" if fact.column else " for the whole table; name the column"))
                elif fact.kind == "number" and fact.unit != declared:
                    problems.append(f"{where}: unit {fact.unit!r}, {fact.table} is in {declared!r}")
            if fact.kind == "number" and fact.unit and fact.scale_units and fact.unit not in fact.scale_units:
                problems.append(f"{where}: scale_units must include the unit {fact.unit}")
        for group in entry.tools_any:
            for name in group:
                if name not in tools:
                    problems.append(f"{qid}: tools_any names {name}, which is not a tool")
        if q.cls == "lookup" and entry.tools_any:
            if len(entry.tools_any) > basic_cap:
                problems.append(f"{qid}: the expected path has {len(entry.tools_any)} calls, the basic level "
                                f"allows {basic_cap}")
            for group in entry.tools_any:
                if not set(group) & basic:
                    problems.append(f"{qid}: no basic tool in the expected call {group}")
    return problems
