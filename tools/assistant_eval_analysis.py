"""The calibration analysis and the agreement figures of the grounding calibration (P3 §7.5, §7.6).  A developer
tool: not packaged, never run in CI; ``tools/assistant_eval.py calibrate`` and ``... summary --dir`` are its
command line.

**calibrate** reads ``results.jsonl`` (grading v2 lines, ``line_format`` 2; the latest line of a cell is its
grade) and prints, per model, a setting × class table of ``answered_rate`` (answered / applicable),
``incomplete_rate`` (incomplete / applicable: nothing false said, something left out; never safe) and
``safe_rate`` (answered or escalated with no invented number), each with n and a Wilson 95 % interval, the windows
of each setting and the cells left out as not applicable or skipped by the owner (``run --skip``: in no rate,
listed apart); per model the cells incomplete on a critical item, the
cells without an answer and the share the owner graded; per setting the calls, rounds and output tokens of
answered cells (median, 90th percentile by nearest rank), cap hits, length stops, fit compactions, truncations, the
prompt ratios and the understandability ratings; the free-plan cells in a block of their own, not used for the
caps.  Then the evidence for the owner (P3 §7.6 (a)–(c)): tasks per level, the proposed call caps and each model's
evidence level against its automatic level.  With round 1's few cells per class these are evidence for the owner's
review, not pass/fail decisions.

**Failed requests** (:func:`failed_attempt_codes`, :func:`failed_attempts_line`): per model, the failed request
attempts of the saved turns by code, which ``summary`` prints as the service's and the format's problems (an
overload, a tool call the service could not read), never as part of a grade or rate.

**Agreement** (:func:`agreement_report`): the owner's verdicts, given before Opus's were shown, against Opus's on
the reviewed cells: per item (right / missing / wrong), per outcome (four) and for understandability (1–3, also
linear-weighted kappa): n, agreement, Cohen's kappa with a bootstrap interval resampled by question, the confusion
table and the lenient (Opus kinder) and strict counts; for the random sample, for all reviewed cells and, as raw
counts, for Opus's own cells that no other reason queued against the random sample (self-preference at matched
difficulty).
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import assistant_eval_scoring as scoring

Z95 = 1.959963984540054
#: The starting bars of P3-O5: answered_rate per level setting, and no invented number.
BARS = {"small": 0.60, "medium": 0.75, "large": 0.75}
#: The basic floor of a proposed call cap (P3 §7.6 (b)).
CAP_FLOOR = 3
LINE_FORMAT = 2

GRADER_MODEL = "claude-opus-5-5"
GRADER_PROVIDER = "anthropic"
GRADER_MAX_OUTPUT = 16_000
BOOTSTRAP_REPS = 2_000
#: Above this share of undefined bootstrap resamples the interval is called unreliable.
UNDEFINED_SHARE = 0.05
#: The random review sample: this share of the cells no other reason queued, and at least REVIEW_FLOOR of them.
REVIEW_SHARE = 0.15
#: The smallest sample whose Wilson upper bound can fall below REVIEW_ALL_BOUND: 0 of 35 gives 9.9 %, 0 of 34
#: gives 10.2 %.
REVIEW_FLOOR = 35
#: Opus's outcome disagreement on the random sample: a Wilson upper bound above this → review every cell.
REVIEW_ALL_BOUND = 0.10

# --- statistics -----------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """The Wilson score interval of *k* of *n* (``None`` for n = 0)."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def nearest_rank(values: Sequence[float], share: float) -> float | None:
    """The *share* percentile by nearest rank (sorted, the value at rank ceil(share · n))."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(round(share * len(ordered), 9))) - 1]


def kappa(pairs: Sequence[tuple[str, str]]) -> tuple[float, float, float | None]:
    """``(observed agreement, expected agreement, Cohen's kappa)`` of two raters' verdicts *pairs* over any set of
    categories (kappa ``None`` when the expected agreement is 1)."""
    n = len(pairs)
    if n == 0:
        return math.nan, math.nan, None
    observed = sum(1 for a, b in pairs if a == b) / n
    categories = {a for a, _ in pairs} | {b for _, b in pairs}
    expected = sum((sum(1 for a, _ in pairs if a == c) / n) * (sum(1 for _, b in pairs if b == c) / n)
                   for c in categories)
    if expected >= 1.0:
        return observed, expected, None
    return observed, expected, (observed - expected) / (1 - expected)


def weighted_kappa(pairs: Sequence[tuple[int, int]], categories: Sequence[int] = scoring.UNDERSTANDABILITY
                   ) -> float | None:
    """Cohen's kappa with linear weights over the ordered *categories* (a 1↔3 disagreement counts twice a 1↔2
    one): agreement weight ``1 − |i − j| / (k − 1)``; ``None`` for no pairs or an expected weighted agreement of 1."""
    n = len(pairs)
    if n == 0:
        return None
    index = {c: k for k, c in enumerate(categories)}
    span = len(categories) - 1

    def weight(a: int, b: int) -> float:
        return 1.0 - abs(index[a] - index[b]) / span

    observed = sum(weight(a, b) for a, b in pairs) / n
    rows = {c: sum(1 for a, _ in pairs if a == c) / n for c in categories}
    cols = {c: sum(1 for _, b in pairs if b == c) / n for c in categories}
    expected = sum(rows[a] * cols[b] * weight(a, b) for a in categories for b in categories)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class Bootstrap:
    """A bootstrap of kappa: the 95 % percentile interval of the defined resamples (``None`` when none), and how
    many of the *reps* resamples had no kappa (all their verdicts alike)."""

    interval: tuple[float, float] | None
    undefined: int
    reps: int


def bootstrap_kappa(clusters: Mapping[str, Sequence[tuple[Any, Any]]], *, reps: int = BOOTSTRAP_REPS,
                    seed: int = 0, stat: Callable[[list], float | None] | None = None) -> Bootstrap:
    """A 95 % percentile interval of kappa (or *stat* of the pairs), resampling whole clusters (questions) with
    replacement; a resample without a value is counted, not dropped silently.  Fewer than two clusters: no
    resampling (``reps`` 0)."""
    stat = stat or (lambda pairs: kappa(pairs)[2])
    keys = sorted(clusters)
    if len(keys) < 2:
        return Bootstrap(None, 0, 0)
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        pairs = [p for k in (rng.choice(keys) for _ in keys) for p in clusters[k]]
        value = stat(pairs)
        if value is not None:
            values.append(value)
    undefined = reps - len(values)
    if not values:
        return Bootstrap(None, undefined, reps)
    values.sort()
    return Bootstrap((values[int(0.025 * (len(values) - 1))], values[int(math.ceil(0.975 * (len(values) - 1)))]),
                     undefined, reps)


def bootstrap_note(boot: Bootstrap) -> str:
    """The report's words on the undefined resamples ("" when there are none)."""
    if not boot.undefined:
        return ""
    note = f"; {boot.undefined} of {boot.reps} resamples undefined (all verdicts alike)"
    return note + (", interval unreliable" if boot.undefined > UNDEFINED_SHARE * boot.reps else "")


# --- calibrate ------------------------------------------------------------------------------------------------
def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def authoritative(lines: Iterable[Mapping[str, Any]]) -> list[dict]:
    """The authoritative lines, the latest per cell (a re-grade replaces the first pass).  A cell is the model with
    its free-plan setting, the versions, the level setting, the question and the repeat; a ``not_applicable``
    or ``skipped`` line of a cell is dropped once a later line of that cell exists (it ran after all, under any
    key version)."""
    latest: dict[tuple, dict] = {}
    for line in lines:
        if not line.get("authoritative"):
            continue
        base = (line.get("provider"), line.get("model"), bool(line.get("free_tier")), line.get("set_version"),
                line.get("level_setting"), line.get("question"), line.get("repeat", 1))
        if line.get("outcome") not in LEFT_OUT:
            for cell in [c for c, x in latest.items() if c[:-1] == base and x.get("outcome") in LEFT_OUT]:
                del latest[cell]
        latest[(*base, line.get("key_version"))] = dict(line)
    return list(latest.values())


#: The outcomes of the cells left out of every rate: not run for their context, skipped by the owner.
LEFT_OUT = frozenset({"not_applicable", "skipped"})


@dataclass
class Rate:
    k: int
    n: int

    @property
    def rate(self) -> float | None:
        return self.k / self.n if self.n else None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.k, self.n)


@dataclass
class ModelEvidence:
    provider: str
    model: str
    automatic_level: str | None
    free_tier: bool
    answered: dict[tuple[str, str], Rate] = field(default_factory=dict)
    incomplete: dict[tuple[str, str], Rate] = field(default_factory=dict)
    safe: dict[tuple[str, str], Rate] = field(default_factory=dict)
    escalated: dict[tuple[str, str], Rate] = field(default_factory=dict)
    invented: dict[tuple[str, str], int] = field(default_factory=dict)
    windows: dict[str, list[int]] = field(default_factory=dict)
    not_applicable: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    #: Cells incomplete on a critical item, cells without a final answer, cells the owner graded.
    critical_incomplete: int = 0
    no_answer: int = 0
    owner_graded: int = 0
    cells: int = 0


def evidence(lines: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, bool], ModelEvidence]:
    """Per (provider, model, free plan): the rates per (setting, class)."""
    out: dict[tuple[str, str, bool], ModelEvidence] = {}
    for line in lines:
        key = (line["provider"], line["model"], bool(line.get("free_tier")))
        ev = out.setdefault(key, ModelEvidence(line["provider"], line["model"], line.get("automatic_level"),
                                               bool(line.get("free_tier"))))
        if line.get("automatic_level") and not ev.automatic_level:
            ev.automatic_level = line["automatic_level"]
        cell = (line["level_setting"], line["class"])
        if line.get("outcome") == "not_applicable":
            ev.not_applicable.append(f"{line['question']} at {line['level_setting']}")
            continue
        if line.get("outcome") == "skipped":
            ev.skipped.append(f"{line['question']} at {line['level_setting']}")
            continue
        if line.get("window") is not None and line["window"] not in ev.windows.setdefault(line["level_setting"], []):
            ev.windows[line["level_setting"]].append(line["window"])
        for table in (ev.answered, ev.incomplete, ev.safe, ev.escalated):
            table.setdefault(cell, Rate(0, 0)).n += 1
        invented = int(line.get("invented_numbers") or 0)
        ev.invented[cell] = ev.invented.get(cell, 0) + invented
        ev.cells += 1
        ev.owner_graded += line.get("grader") == "owner"
        ev.no_answer += bool(line.get("no_answer"))
        if line["outcome"] == "answered":
            ev.answered[cell].k += 1
        if line["outcome"] == "incomplete":
            ev.incomplete[cell].k += 1
            ev.critical_incomplete += bool(line.get("critical_missing"))
        if line["outcome"] == "escalated":
            ev.escalated[cell].k += 1
        if line["outcome"] in ("answered", "escalated") and invented == 0:
            ev.safe[cell].k += 1
    return out


#: The start of Ollama's message for a tool call it could not read (an older record holds it as ``busy``).
UNREADABLE_TOOL_CALL_MESSAGE = "error parsing tool call"


def attempt_code(attempt: Mapping[str, Any]) -> str:
    """The code of one failed attempt: ``tool_call_unreadable`` also for one recorded before that code existed."""
    code = str(attempt.get("code"))
    if code == "busy" and attempt.get("status") == 500 \
            and str(attempt.get("message") or "").startswith(UNREADABLE_TOOL_CALL_MESSAGE):
        return "tool_call_unreadable"
    return code


def failed_attempt_codes(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, int]]:
    """Per (provider, model): how many failed request attempts of each code the records' turns hold
    (``failed_attempts``: retried and recovered, or the stopping one of a turn), codes in alphabetical order.  These
    are the service's and the format's problems (an overload, a rate limit, a tool call the service could not read),
    not the model's grade: no rate counts them.  An attempt recorded before ``tool_call_unreadable`` was its own
    code (``busy``, status 500, a message starting "error parsing tool call": Ollama's) is counted as one."""
    out: dict[tuple[str, str], dict[str, int]] = {}
    for record in records:
        model = record.get("model") or {}
        key = (str(model.get("preset")), str(model.get("model") or (record.get("cell") or {}).get("model")))
        counts = out.setdefault(key, {})
        for turn in record.get("turns") or []:
            for attempt in turn.get("failed_attempts") or []:
                code = attempt_code(attempt)
                counts[code] = counts.get(code, 0) + 1
    return {key: dict(sorted(counts.items())) for key, counts in out.items()}


#: The hashes of a cell's level setting (``setting_key`` of its record) whose mix within one model and level the
#: summary and grade-ai warn of: cells asked with other instructions, tools or tool behaviour (``behaviour``,
#: :data:`flextool.assistant.ASSISTANT_BEHAVIOUR`; a record before it was recorded: unknown) are not of one setting.
SETTING_HASHES = (("tools_hash", "tools"), ("system_hash", "instructions"), ("behaviour", "tool behaviours"))


def mixed_settings(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """One warning per (provider, model, level) and hash of :data:`SETTING_HASHES` that more than one value holds
    among the records' ``setting_key`` (a record without it, written before the key was recorded: ``unknown``,
    which alone warns of nothing).  In provider, model, level order."""
    seen: dict[tuple[str, str, str], dict[str, set[str]]] = {}
    for record in records:
        cell = record.get("cell") or {}
        model = record.get("model") or {}
        key = (str(model.get("preset")), str(cell.get("model") or model.get("model")), str(cell.get("level_setting")))
        setting = record.get("setting_key")
        hashes = seen.setdefault(key, {name: set() for name, _what in SETTING_HASHES})
        for name, _what in SETTING_HASHES:
            value = setting.get(name) if isinstance(setting, Mapping) else None
            hashes[name].add(str(value) if value else "unknown")
    out = []
    for (preset, model, level), hashes in sorted(seen.items()):
        for name, what in SETTING_HASHES:
            if len(hashes[name]) > 1:
                values = sorted(hashes[name], key=lambda v: (v == "unknown", v))
                out.append(f"Warning: the cells of {preset} {model} at level {level} were asked with "
                           f"{len(values)} different {what} ({name} {', '.join(values)}): their grades are not "
                           "of one setting.")
    return out


def failed_attempts_line(counts: Mapping[tuple[str, str], Mapping[str, int]]) -> str | None:
    """The summary's line of :func:`failed_attempt_codes` for one model: ``None`` when no request failed."""
    merged: dict[str, int] = {}
    for by_code in counts.values():
        for code, n in by_code.items():
            merged[code] = merged.get(code, 0) + n
    if not merged:
        return None
    listed = ", ".join(f"{code} {merged[code]}" for code in sorted(merged))
    return f"  failed requests (service or format problems, not part of the grade): {listed}"


def _pct(rate: Rate) -> str:
    if not rate.n:
        return "   -"
    low, high = rate.interval or (0.0, 0.0)
    return f"{rate.k}/{rate.n} {100 * rate.rate:.0f}% [{100 * low:.0f}–{100 * high:.0f}]"


def setting_stats(lines: Sequence[Mapping[str, Any]], level: str) -> dict[str, Any]:
    """The answered cells' calls, rounds and output tokens (median, 90th percentile) and the health counts of
    one level setting."""
    rows = [x for x in lines if x["level_setting"] == level and x.get("outcome") not in LEFT_OUT]
    answered = [x for x in rows if x["outcome"] == "answered"]

    def spread(name: str) -> tuple[float | None, float | None]:
        values = [float(x.get(name) or 0) for x in answered]
        return (statistics.median(values) if values else None, nearest_rank(values, 0.9))

    ratios_text = [x["prompt_ratio"]["text"] for x in rows if (x.get("prompt_ratio") or {}).get("text") is not None]
    ratios_tool = [x["prompt_ratio"]["tool_results"] for x in rows
                   if (x.get("prompt_ratio") or {}).get("tool_results") is not None]
    ratings = [int(x["understandability"]) for x in rows if x.get("understandability") is not None]
    helpful = [int(x["helpfulness"]) for x in rows if x.get("helpfulness") is not None]
    return {"cells": len(rows), "answered": len(answered), "calls": spread("calls"), "rounds": spread("rounds"),
            "understandability": {u: ratings.count(u) for u in scoring.UNDERSTANDABILITY},
            "understandability_median": statistics.median(ratings) if ratings else None,
            "helpfulness": {u: helpful.count(u) for u in (1, 2, 3)},
            "tokens_out": spread("tokens_out"),
            **{name: sum(int(x.get(name) or 0) for x in rows)
               for name in ("cap_hits", "length_stops", "fit_compactions", "truncations")},
            "prompt_ratio_text": statistics.median(ratios_text) if ratios_text else None,
            "prompt_ratio_tool_results": statistics.median(ratios_tool) if ratios_tool else None,
            "refused_tools": sorted({t for x in rows if x["outcome"] == "failed" for t in x.get("refused_tools") or []})}


def proposed_cap(stats: Mapping[str, Any]) -> int | None:
    """90th percentile of the answered cells' calls + 1, never below the basic floor (P3 §7.6 (b))."""
    p90 = stats["calls"][1]
    return None if p90 is None else max(CAP_FLOOR, int(p90) + 1)


def evidence_level(ev: ModelEvidence) -> str | None:
    """The highest level L whose classes (and the lower ones) are answered at setting L at L's bar with no
    invented number (P3 §7.6 (c)); ``None`` when not even basic."""
    best = None
    for level in scoring.LEVELS:
        classes = [c for c in scoring.CLASSES if scoring.CLASS_RANK[c] <= scoring.LEVEL_RANK[level]]
        ok = True
        for cls in classes:
            rate = ev.answered.get((level, cls))
            if rate is None or not rate.n or rate.rate < BARS[level] or ev.invented.get((level, cls), 0):
                ok = False
        if not ok:
            break
        best = level
    return best


def task_support(models: Sequence[ModelEvidence]) -> list[str]:
    """P3 §7.6 (a): per level and class it covers, whether the models placed at it answer the class there and the
    models below escalate it cleanly at their own setting."""
    out = []
    for level in scoring.LEVELS:
        at = [m for m in models if m.automatic_level == level]
        below = [m for m in models if m.automatic_level is not None
                 and scoring.LEVEL_RANK[m.automatic_level] < scoring.LEVEL_RANK[level]]
        for cls in [c for c in scoring.CLASSES if scoring.CLASS_RANK[c] <= scoring.LEVEL_RANK[level]]:
            answered = [m.answered.get((level, cls)) for m in at]
            answered = [r for r in answered if r is not None and r.n]
            # each rate with its own model's level, so its bar does not depend on the order of the lines
            escalated = [(r, m.automatic_level) for m in below
                         if scoring.CLASS_RANK[cls] > scoring.LEVEL_RANK[m.automatic_level]
                         for r in [m.escalated.get((m.automatic_level, cls))] if r is not None and r.n]
            if not answered:
                out.append(f"  {level} / {cls}: no model placed at {level} answered it at {level}; read the "
                           "transcripts")
                continue
            ok_at = all(r.rate >= BARS[level] for r in answered)
            ok_below = all(r.rate >= BARS[own] for r, own in escalated)
            text = "supports the level text as drafted" if ok_at and ok_below else \
                "read the transcripts and propose a revised level text to the owner"
            out.append(f"  {level} / {cls}: answered " + ", ".join(_pct(r) for r in answered)
                       + (("; escalated below " + ", ".join(_pct(r) for r, _own in escalated)) if escalated else "")
                       + f" → {text}")
    return out


def calibrate_report(lines: Sequence[Mapping[str, Any]]) -> str:
    """The §7.6 evidence tables as text."""
    lines = authoritative(lines)
    if not lines:
        return "No authoritative lines."
    out: list[str] = ["Calibration evidence (round 1: read qualitatively; the bars are starting values, not "
                      "decisions)", f"Bars: answered_rate basic {BARS['small']:.0%}, standard {BARS['medium']:.0%}, "
                      f"advanced {BARS['large']:.0%}; no invented number."]
    groups = evidence(lines)
    paid = [m for (_p, _m, free), m in groups.items() if not free]
    free = [m for (_p, _m, is_free), m in groups.items() if is_free]
    for block, models in (("Models", paid), ("Free-plan models (not used for the caps)", free)):
        if not models:
            continue
        out.append("")
        out.append(f"== {block} ==")
        for m in models:
            out.append(f"{m.provider} {m.model} (automatic level {m.automatic_level or 'unknown'})")
            out.append(f"  {'setting':<8} {'class':<9} {'answered_rate':<26} {'incomplete_rate':<26} "
                       f"{'safe_rate':<26} invented  window")
            for level in scoring.LEVELS:
                for cls in scoring.CLASSES:
                    rate = m.answered.get((level, cls))
                    if rate is None:
                        continue
                    windows = ",".join(str(w) for w in m.windows.get(level, []))
                    out.append(f"  {level:<8} {cls:<9} {_pct(rate):<26} {_pct(m.incomplete[(level, cls)]):<26} "
                               f"{_pct(m.safe[(level, cls)]):<26} {m.invented.get((level, cls), 0):<9} {windows}")
            out.append(f"  incomplete on a critical item: {m.critical_incomplete}; no answer (failed): "
                       f"{m.no_answer}; graded by the owner: {m.owner_graded} of {m.cells} (the rest by "
                       f"{GRADER_MODEL})")
            if m.not_applicable:
                out.append("  not applicable (context): " + ", ".join(m.not_applicable))
            if m.skipped:
                out.append(f"  skipped: {len(m.skipped)} (by the owner, in no rate: " + ", ".join(m.skipped) + ")")
            level = evidence_level(m)
            note = "" if level == m.automatic_level else \
                (f" — differs from its automatic level {m.automatic_level}: propose a seed change for its name "
                 "token or size band, for the owner to decide on the transcripts")
            out.append(f"  evidence level: {level or 'none'}{note}")
    out.append("")
    out.append("== Per setting (models above; free-plan cells left out) ==")
    used = [x for x in lines if not x.get("free_tier")]
    for level in scoring.LEVELS:
        stats = setting_stats(used, level)
        if not stats["cells"]:
            continue

        def two(pair: tuple) -> str:
            return "-" if pair[0] is None else f"median {pair[0]:g}, p90 {pair[1]:g}"

        out.append(f"{level}: {stats['cells']} cells, {stats['answered']} answered; calls {two(stats['calls'])}; "
                   f"rounds {two(stats['rounds'])}; output tokens {two(stats['tokens_out'])}; cap hits "
                   f"{stats['cap_hits']}, length stops {stats['length_stops']}, fit compactions "
                   f"{stats['fit_compactions']}, truncations {stats['truncations']}; prompt ratio text "
                   f"{stats['prompt_ratio_text']}, tool results {stats['prompt_ratio_tool_results']}")
        dist = stats["understandability"]
        if sum(dist.values()):
            out.append(f"  understandability 1 / 2 / 3: {dist[1]} / {dist[2]} / {dist[3]}, median "
                       f"{stats['understandability_median']:g}")
        dist = stats["helpfulness"]
        if sum(dist.values()):
            out.append(f"  helpfulness (the owner's sample) 1 / 2 / 3: {dist[1]} / {dist[2]} / {dist[3]}")
        cap = proposed_cap(stats)
        if cap is not None:
            out.append(f"  proposed calls per question: {cap} (p90 of answered cells + 1, at least {CAP_FLOOR})")
        if stats["refused_tools"]:
            out.append(f"  refused tools in failed cells: {', '.join(stats['refused_tools'])} — check whether "
                       "they were needed")
        if stats["length_stops"]:
            out.append("  length stops: give this level more output tokens")
        ratio = stats["prompt_ratio_tool_results"]
        if ratio is not None and ratio > 1.0:
            out.append("  the tool-result estimate is below the reported tokens: lower the JSON divisor")
    s01 = [x for x in used if x["question"] == "S01" and x["outcome"] == "failed"]
    if s01:
        out.append("S01 failed for " + ", ".join(sorted({x["model"] for x in s01}))
                   + ": check the 12k–32k compaction row")
    out.append("")
    out.append("== Tasks per level (P3 §7.6 (a)) ==")
    out.extend(task_support(paid))
    return "\n".join(out)


def cmd_calibrate(args: Any, *, say: Callable[[str], None] = print) -> int:
    path = Path(args.results_file)
    if not path.is_file():
        print(f"No results file {path}.", file=sys.stderr)
        return 2
    lines = read_lines(path)
    old = sum(1 for x in lines if x.get("line_format") != LINE_FORMAT)
    if old:
        print(f"{path} has {old} line(s) without line_format {LINE_FORMAT}, from the earlier grading (no 'missing' "
              "verdict, no 'incomplete' outcome): calibrate reads grading v2 lines only. Move those lines to "
              "another file.", file=sys.stderr)
        return 2
    say(calibrate_report(lines))
    return 0


# --- agreement (owner first, then Opus) ------------------------------------------------------------------------
#: How kind a verdict is: Opus is lenient where its verdict ranks above the owner's, strict where below.
KIND_RANK = {"item": {"right": 2, "missing": 1, "wrong": 0},
             "outcome": {"answered": 2, "escalated": 2, "incomplete": 1, "failed": 0},
             "understandability": {1: 2, 2: 1, 3: 0}}
CATEGORIES = {"item": scoring.VERDICTS, "outcome": scoring.OUTCOMES, "understandability": scoring.UNDERSTANDABILITY}


@dataclass(frozen=True)
class ReviewedCell:
    """One reviewed cell Opus graded: the owner's verdicts given before Opus's were shown (*first*) and Opus's, as
    pairs ``(owner, opus)``.  *sample*: drawn at random from the cells no other reason queued; *own_matched*: a cell
    Opus wrote that no other reason queued (it would have been among those)."""

    token: str
    question: str
    items: tuple[tuple[str, str], ...]
    outcome: tuple[str, str]
    understandability: tuple[int, int]
    sample: bool = False
    own_matched: bool = False
    #: How Opus's grade came: ``api`` or ``file`` (a Claude Code subagent through grade-ai's file route).
    route: str = "api"


def measure(cells: Sequence[ReviewedCell], name: str, *, seed: int = 0) -> dict[str, Any]:
    """n, agreement, kappa (with its bootstrap by question), the confusion table ``{(owner, opus): count}`` and the
    lenient / strict counts of *name* (item, outcome or understandability) over *cells*; understandability also
    gets the linear-weighted kappa."""
    clusters: dict[str, list[tuple[Any, Any]]] = {}
    for c in cells:
        pairs = list(c.items) if name == "item" else [getattr(c, name)]
        clusters.setdefault(c.question, []).extend(pairs)
    pairs = [p for c in cells for p in (c.items if name == "item" else [getattr(c, name)])]
    out: dict[str, Any] = {"n": len(pairs)}
    if not pairs:
        return out
    observed, _expected, k = kappa(pairs)
    rank = KIND_RANK[name]
    out.update(agreement=observed, kappa=k, bootstrap=bootstrap_kappa(clusters, seed=seed),
               confusion={(a, b): sum(1 for p in pairs if p == (a, b)) for a in CATEGORIES[name]
                          for b in CATEGORIES[name]},
               lenient=sum(1 for a, b in pairs if rank[b] > rank[a]),
               strict=sum(1 for a, b in pairs if rank[b] < rank[a]))
    if name == "understandability":
        out["weighted_kappa"] = weighted_kappa(pairs)
        out["weighted_bootstrap"] = bootstrap_kappa(clusters, seed=seed, stat=weighted_kappa)
    return out


def _share(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f} %"


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _interval(value: tuple[float, float] | None) -> str:
    return "-" if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"


def _measure_lines(cells: Sequence[ReviewedCell], *, seed: int) -> list[str]:
    out = []
    for name, label in (("item", "items (right / missing / wrong)"), ("outcome", "outcome (4)"),
                        ("understandability", "understandability (1–3)")):
        m = measure(cells, name, seed=seed)
        if not m["n"]:
            out.append(f"  {label}: none")
            continue
        text = (f"  {label}: n {m['n']}, agreement {_share(m['agreement'])}, kappa {_num(m['kappa'])} (bootstrap "
                f"by question {_interval(m['bootstrap'].interval)}{bootstrap_note(m['bootstrap'])})")
        if name == "understandability":
            text += (f", linear-weighted kappa {_num(m['weighted_kappa'])} "
                     f"{_interval(m['weighted_bootstrap'].interval)}")
        out.append(text + f"; lenient (Opus kinder) {m['lenient']}, strict {m['strict']}")
        cats = CATEGORIES[name]
        out.append("    owner \\ Opus  " + "  ".join(f"{c!s:>10}" for c in cats))
        for a in cats:
            out.append(f"    {a!s:<12}  " + "  ".join(f"{m['confusion'][(a, b)]:>10}" for b in cats))
    return out


def review_all_needed(sample: Sequence[ReviewedCell]) -> tuple[int, int, float | None, bool]:
    """``(outcome disagreements, sample size, Wilson upper bound, review every cell?)`` of the random sample: the
    bound above :data:`REVIEW_ALL_BOUND` (or no sample) → review every cell."""
    d = sum(1 for c in sample if c.outcome[0] != c.outcome[1])
    bound = wilson(d, len(sample))
    upper = bound[1] if bound else None
    return d, len(sample), upper, upper is None or upper > REVIEW_ALL_BOUND


def _raw(cells: Sequence[ReviewedCell]) -> str:
    agree = sum(1 for c in cells if c.outcome[0] == c.outcome[1])
    rank = KIND_RANK["outcome"]
    lenient = sum(1 for c in cells if rank[c.outcome[1]] > rank[c.outcome[0]])
    strict = sum(1 for c in cells if rank[c.outcome[1]] < rank[c.outcome[0]])
    irank = KIND_RANK["item"]
    items = [p for c in cells for p in c.items]
    return (f"{len(cells)} cells: outcome {agree} agree, {lenient} lenient, {strict} strict; items {len(items)}: "
            f"{sum(1 for a, b in items if irank[b] > irank[a])} lenient, "
            f"{sum(1 for a, b in items if irank[b] < irank[a])} strict")


def agreement_report(reviewed: Sequence[ReviewedCell], *, unreviewed: int, whole_rest: bool, unparsed: int,
                     seed: int) -> str:
    """The agreement text (``agreement.txt``): the random sample (the estimate for the *unreviewed* cells Opus
    alone graded) with the review-all rule, all reviewed cells, and self-preference at matched difficulty."""
    sample = [c for c in reviewed if c.sample]
    out = [f"Agreement of {GRADER_MODEL} with the owner: the owner's verdicts given before Opus's were shown, "
           "against Opus's. Blinding is partial: style, tool names and answer length still hint at the model."]
    out.append(f"== The random sample ({len(sample)} cells; the estimate for the {unreviewed} cells Opus alone "
               "graded) ==")
    out += _measure_lines(sample, seed=seed)
    if whole_rest:
        out.append("  The random sample is every cell no other reason queued: every graded cell is reviewed.")
    elif sample:
        d, n, upper, needed = review_all_needed(sample)
        out.append(f"  outcome disagreements {d} of {n}, Wilson 95 % upper bound {_share(upper)}"
                   + (": Review every cell: review --all." if needed else
                      f": at most {REVIEW_ALL_BOUND:.0%} of the unreviewed outcomes are expected to differ."))
    out.append(f"== All reviewed cells ({len(reviewed)}) ==")
    out += _measure_lines(reviewed, seed=seed)
    own = [c for c in reviewed if c.own_matched]
    out.append("== Self-preference: the cells Opus wrote that no other reason queued, against the random sample "
               "(raw counts; matched difficulty) ==")
    out.append(f"  Opus's own: {_raw(own)}")
    out.append(f"  random sample: {_raw(sample)}")
    routes = sorted({c.route for c in reviewed})
    if routes != ["api"] and routes:
        # The same model by another route (the file route: Claude Code subagents): its agreement apart.
        for route in routes:
            of_route = [c for c in reviewed if c.route == route]
            in_sample = [c for c in of_route if c.sample]
            out.append(f"== Graded through the {route} route: all reviewed cells ({len(of_route)}; "
                       f"{len(in_sample)} of the random sample) ==")
            out += _measure_lines(of_route, seed=seed)
            if in_sample:
                d = sum(1 for c in in_sample if c.outcome[0] != c.outcome[1])
                out.append(f"  random sample outcome disagreements {d} of {len(in_sample)}")
    if unparsed:
        out.append(f"{unparsed} cell(s) Opus's reply could not be read for: graded by the owner alone, left out.")
    return "\n".join(out)
