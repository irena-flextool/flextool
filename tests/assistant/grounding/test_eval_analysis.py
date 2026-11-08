"""The calibration runner and its analysis, without a model (P3 test 12).

``run`` against a scripted backend and probe: a project its own build did not make (or whose source changed) is
refused with exit 2 before any request; a results folder of another model or project is refused; the daily
request budget stops after the cell that reaches it and ``--resume`` goes on the next UTC day; a record carries
the knobs sent, the level setting and the calibration tag, and the ledger lists the requests apart.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.assistant import grounding_projects as gp
from tests.assistant.llm.fake_backend import FakeBackend

MODEL = "vendor/model-a:free"


@pytest.fixture(scope="module")
def grounding(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A grounding project of inputs only (the runner needs its marker, the scripted model no results)."""
    return gp.build_grounding_project(tmp_path_factory.mktemp("grounding"), solve=False).path


class Harness:
    """A scripted probe and backend: every backend answers each request with one short text."""

    def __init__(self) -> None:
        self.backends: list[FakeBackend] = []
        self.probes = 0

    def prober(self, preset, conn, cancel, *, needed):
        from flextool.assistant.probes import ModelFacts

        self.probes += 1
        return ModelFacts(context_window=131_072, tools=True, source="openrouter")

    def factory(self, preset, conn):
        from flextool.assistant.llm_adapter import Message, Reply, Usage

        backend = FakeBackend([Reply(Message("assistant", "coal_plant has 500 MW."), "end", None,
                                     Usage(1200, 0, 0, 20))] * 8, origin=(preset.id, conn.base_url, conn.model))
        self.backends.append(backend)
        return backend

    @property
    def requests(self) -> int:
        return sum(len(b.requests) for b in self.backends)


def run(evaluator, project: Path, out: Path, *extra: str, harness: Harness, model: str = MODEL) -> int:
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(project), "--provider", "openrouter", "--model", model, "--out", str(out),
         "--accept-data-notice", *extra])
    return evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=lambda _text: None)


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")


def test_plain_project_refused_before_any_request(evaluator, assistant_project: Path, tmp_path: Path,
                                                  capsys) -> None:
    harness = Harness()
    assert run(evaluator, assistant_project, tmp_path / "out", harness=harness) == 2
    assert evaluator.NOT_GROUNDING in capsys.readouterr().err
    assert harness.probes == 0 and harness.backends == []
    assert not (tmp_path / "out").exists()


def test_changed_source_refused(evaluator, tmp_path: Path, capsys) -> None:
    project = gp.build_grounding_project(tmp_path / "p", solve=False).path
    with open(project / "input_sources" / gp.SOURCE_NAME, "ab") as f:
        f.write(b"\0")
    harness = Harness()
    assert run(evaluator, project, tmp_path / "out", harness=harness) == 2
    assert evaluator.NOT_GROUNDING in capsys.readouterr().err
    assert harness.probes == 0 and harness.backends == []


def test_resume_refuses_another_model_or_project(evaluator, grounding: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "out"
    harness = Harness()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=harness) == 0
    assert harness.requests == 1
    capsys.readouterr()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=harness) == 2
    assert evaluator.HAS_RESULTS in capsys.readouterr().err
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01", harness=harness,
               model="vendor/model-b:free") == 2
    assert evaluator.OTHER_RESULTS in capsys.readouterr().err
    run_file = out / evaluator.RUN_FILE
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    run_file.write_text(json.dumps({**saved, "project_fingerprint": "0" * 64}), encoding="utf-8")
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01", harness=harness) == 2
    assert evaluator.OTHER_RESULTS in capsys.readouterr().err
    assert harness.requests == 1


def test_daily_budget_stops_and_resumes_next_day(evaluator, grounding: Path, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    said: list[str] = []
    harness = Harness()

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "openrouter", "--model", MODEL, "--out", str(out),
             "--accept-data-notice", "--questions", "G01,G03,G04", "--levels", "basic",
             "--max-requests-per-day", "2", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                                 say=said.append)

    assert go() == 0
    assert said[-1] == evaluator.DAILY_DONE
    assert sorted(p.stem for p in (out / "cells").glob("*.json")) == ["G01__small__r1", "G03__small__r1"]
    assert harness.requests == 2
    assert go("--resume") == 0                              # the same day: nothing more is sent
    assert said[-1] == evaluator.DAILY_DONE and harness.requests == 2
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(evaluator, "utc_now", lambda: tomorrow)
    assert go("--resume") == 0
    assert harness.requests == 3
    assert len(list((out / "cells").glob("*.json"))) == 3


def test_daily_budget_defaults(evaluator) -> None:
    assert evaluator.daily_budget("openrouter", "vendor/m:free", None) == 45
    assert evaluator.daily_budget("openrouter", "vendor/m", None) is None
    assert evaluator.daily_budget("gemini", "gemini-flash", None) is None
    assert evaluator.daily_budget("gemini", "gemini-flash", 100) == 100


def test_record_and_ledger(evaluator, grounding: Path, tmp_path: Path) -> None:
    from flextool.assistant.usage import Ledger

    out = tmp_path / "out"
    harness = Harness()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic,advanced",
               "--relax-data-collection", harness=harness) == 0
    small = json.loads((out / "cells" / "G01__small__r1.json").read_text(encoding="utf-8"))
    large = json.loads((out / "cells" / "G01__large__r1.json").read_text(encoding="utf-8"))
    assert small["cell"] == {"model": MODEL, "level_setting": "small", "question": "G01", "class": "lookup",
                             "repeat": 1}
    assert small["model"]["knobs_sent"] == {"provider": {"data_collection": "allow"}}
    assert small["model"]["knobs_relaxed"] is True and small["model"]["free_tier"] is True
    assert small["purpose"] == "calibration" and small["requests_sent"] == 1
    assert small["budget"]["caps"]["max_calls_per_turn"] == 6
    assert large["budget"]["caps"]["max_calls_per_turn"] > 6
    turn = small["turns"][0]
    assert turn["answer"] == "coal_plant has 500 MW." and turn["stop"] == "end" and turn["tool_calls"] == []
    assert small["project_fingerprint"] == json.loads((grounding / gp.MARKER_FILE).read_text())["fingerprint"]
    assert harness.probes == 1
    months = Ledger().read()["months"]
    keys = {k for month in months.values() for k in month}
    assert keys == {"openrouter" + evaluator.CALIBRATION_SUFFIX}


def test_ollama_context_not_applicable(evaluator, grounding: Path, tmp_path: Path) -> None:
    from flextool.assistant.probes import ModelFacts

    out = tmp_path / "out"
    harness = Harness()
    loaded = {"gpt-oss-20b-16k": 16_384, "gpt-oss-20b-32k": 24_576}

    def prober(preset, conn, cancel, *, needed):
        return ModelFacts(context_window=131_072, loaded_context=loaded[conn.model], tools=True, source="ollama",
                          parameter_count=20_900_000_000)

    def info(base_url: str, model: str) -> dict:
        return {"num_ctx": {"gpt-oss-20b-16k": 16_384, "gpt-oss-20b-32k": 32_768}[model], "size": 14_000,
                "size_vram": 14_000}

    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "ollama", "--model", "gpt-oss:20b", "--out", str(out),
         "--questions", "G01", "--levels", "basic,standard", "--ollama-context", "basic=16384,standard=32768"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=prober, info=info,
                             say=lambda _t: None) == 0
    basic = json.loads((out / "cells" / "G01__small__r1.json").read_text(encoding="utf-8"))
    standard = json.loads((out / "cells" / "G01__medium__r1.json").read_text(encoding="utf-8"))
    assert basic["not_applicable"] is None and basic["model"]["model"] == "gpt-oss-20b-16k"
    assert basic["model"]["ollama"] == {"variant": "gpt-oss-20b-16k", "num_ctx": 16_384, "loaded_context": 16_384,
                                        "size": 14_000, "size_vram": 14_000}
    assert standard["not_applicable"] == "context" and standard["turns"] == []
    assert standard["model"]["ollama"]["loaded_context"] == 24_576
    assert harness.requests == 1


def _write_key(path: Path, *, placeholder: bool) -> Path:
    """A key in the oracle's format for G01 that only exercises grading (never the oracle's values)."""
    import yaml

    note = "exercises the grading only"
    path.write_text(yaml.safe_dump({
        "format": 1, "set_version": gp.SET_VERSION, "key_version": 7, "placeholder": placeholder,
        "derived_by": note, "sources": [],
        "questions": {"G01": {"facts": [{"id": "unit", "kind": "token", "value": "coal_plant", "critical": True,
                                         "derivation": note}],
                              "manual": ["facts_vs_assumptions"]}}}), encoding="utf-8")
    return path


def test_grade_and_record(evaluator, grounding: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=Harness()) == 0
    results = tmp_path / "results.jsonl"
    asked: list[str] = []
    shown: list[str] = []

    def grade(key: Path) -> int:
        args = evaluator.build_parser().parse_args(["grade", str(out), "--key", str(key)])
        return evaluator.cmd_grade(args, ask=lambda text: asked.append(text) or "pass", say=shown.append)

    def summary(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(["summary", str(out), "--results-file", str(results), *extra])
        return evaluator.cmd_summary(args, say=shown.append)

    assert grade(_write_key(tmp_path / "placeholder.yaml", placeholder=True)) == 0
    assert summary("--record") == 2 and not results.exists()
    shown.clear()
    assert grade(_write_key(tmp_path / "key.yaml", placeholder=False)) == 0
    view = "\n".join(shown)
    assert "Expected behaviour: a full answer is expected" in view
    assert MODEL not in view and "openrouter" not in view.lower() and "small" not in view
    record = json.loads((out / "cells" / "G01__small__r1.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "answered" and record["key_version"] == 7
    # "500 MW" came from no tool result (the scripted model called none): a number for the grader
    assert record["manual"]["verdicts"] == {"number:1:0": "pass", "facts_vs_assumptions": "pass"}
    assert "Number without a tool source: 500" in view
    assert summary("--record") == 0 and summary("--record") == 0          # the second appends nothing
    lines = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert {k: line[k] for k in ("grader", "authoritative", "first_pass", "provider", "model", "level_setting",
                                 "question", "class", "outcome", "invented_numbers", "key_version")} == {
        "grader": "owner", "authoritative": True, "first_pass": True, "provider": "openrouter", "model": MODEL,
        "level_setting": "small", "question": "G01", "class": "lookup", "outcome": "answered",
        "invented_numbers": 0, "key_version": 7}
    assert list(line) == ["date", "grader", "authoritative", "first_pass", "flextool_version", "commit",
                          "set_version", "key_version", "provider", "model", "automatic_level", "level_setting",
                          "window", "free_tier", "knobs_relaxed", "question", "class", "outcome", "score",
                          "critical_failures", "invented_numbers", "calls", "rounds", "tokens_in", "tokens_out",
                          "tokens_not_reported", "prompt_ratio", "repeat", "level_source", "cap_hits",
                          "length_stops", "fit_compactions", "truncations", "refused_tools"]
    assert "coal_plant has" not in json.dumps(line)


def test_not_applicable_cells_are_recorded_apart(evaluator, grounding: Path, tmp_path: Path) -> None:
    from flextool.assistant.probes import ModelFacts

    out = tmp_path / "out"

    def prober(preset, conn, cancel, *, needed):
        return ModelFacts(context_window=131_072, loaded_context=16_384, tools=True, source="ollama")

    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "ollama", "--model", "gpt-oss:20b", "--out", str(out),
         "--questions", "G01", "--levels", "standard", "--ollama-context", "standard=32768"])
    sizes = {"num_ctx": 32_768, "size": 14_000, "size_vram": 14_000}
    assert evaluator.cmd_run(args, backend_factory=Harness().factory, prober=prober, info=lambda _b, _m: sizes,
                             say=lambda _t: None) == 0
    results = tmp_path / "results.jsonl"
    args = evaluator.build_parser().parse_args(["summary", str(out), "--record", "--results-file", str(results)])
    assert evaluator.cmd_summary(args, say=lambda _t: None) == 0
    (line,) = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines()]
    assert (line["grader"], line["outcome"], line["level_setting"], line["score"]) == (
        "none", "not_applicable", "medium", None)


def test_tool_calls_and_results_recorded(evaluator, grounding: Path, tmp_path: Path) -> None:
    from flextool.assistant.llm_adapter import Message, Reply, ToolCall, Usage

    class Calling(Harness):
        def factory(self, preset, conn):
            call = ToolCall("call1", "list_runs", {}, "{}")
            backend = FakeBackend([Reply(Message("assistant", "", tool_calls=[call]), "tools", None,
                                         Usage(1500, 0, 0, 10)),
                                   Reply(Message("assistant", "No runs yet."), "end", None, Usage(1800, 0, 0, 5))],
                                  origin=(preset.id, conn.base_url, conn.model))
            self.backends.append(backend)
            return backend

    out = tmp_path / "out"
    assert run(evaluator, grounding, out, "--questions", "G04", "--levels", "basic", harness=Calling()) == 0
    record = json.loads((out / "cells" / "G04__small__r1.json").read_text(encoding="utf-8"))
    (turn,) = record["turns"]
    (call,) = turn["tool_calls"]
    assert call["name"] == "list_runs" and call["ok"] is True and call["arguments"] == {}
    assert json.loads(call["result"])["runs"] == []
    assert turn["rounds"] == 2 and turn["calls"] == 1 and turn["answer"] == "No runs yet."
    assert record["requests_sent"] == 2
    assert str(grounding) not in json.dumps(record, ensure_ascii=False)       # paths go through redaction
    ratio = record["budget"]["prompt_ratio"]
    assert ratio["text"] is not None and ratio["tool_results"] is not None


# --- calibrate (expected values computed by hand by an independent agent) -------------------------------------
@pytest.mark.parametrize("k, n, low, high", [(6, 8, 0.4093, 0.9285), (0, 2, 0.0, 0.6576), (2, 2, 0.3424, 1.0),
                                             (1, 2, 0.0945, 0.9055), (2, 4, 0.15, 0.85), (0, 8, 0.0, 0.3244),
                                             (8, 8, 0.6756, 1.0)])
def test_wilson_interval(analysis, k: int, n: int, low: float, high: float) -> None:
    got = analysis.wilson(k, n)
    assert got == pytest.approx((low, high), abs=5e-5)


def test_percentiles_and_proposed_cap(analysis) -> None:
    import statistics

    for values, median, p90, cap in (([2, 3, 3, 1, 2, 3], 2.5, 3, 4), ([1, 4, 2, 6, 3, 5, 2, 8, 2, 7], 3.5, 7, 8)):
        assert statistics.median(values) == median
        assert analysis.nearest_rank(values, 0.9) == p90
        assert analysis.proposed_cap({"calls": (median, p90)}) == cap
    assert analysis.proposed_cap({"calls": (1, 1)}) == analysis.CAP_FLOOR


@pytest.mark.parametrize("table, observed, expected, kappa", [
    ((20, 5, 10, 15), 0.7, 0.5, 0.4), ((9, 1, 0, 2), 0.9167, 0.6667, 0.75)])
def test_kappa(analysis, table, observed, expected, kappa) -> None:
    both, a_only, b_only, neither = table
    pairs = ([("pass", "pass")] * both + [("pass", "fail")] * a_only + [("fail", "pass")] * b_only
             + [("fail", "fail")] * neither)
    assert analysis.kappa(pairs) == pytest.approx((observed, expected, kappa), abs=5e-5)


def _line(**fields) -> dict:
    base = {"date": "2026-11-10", "grader": "owner", "authoritative": True, "first_pass": True, "set_version": 3,
            "key_version": 1, "provider": "ollama", "model": "m-basic", "automatic_level": "small",
            "level_setting": "small", "window": 16384, "free_tier": False, "knobs_relaxed": False, "class": "lookup",
            "outcome": "answered", "score": 1.0, "critical_failures": [], "invented_numbers": 0, "calls": 1,
            "rounds": 2, "tokens_in": 1000, "tokens_out": 100, "tokens_not_reported": 0,
            "prompt_ratio": {"text": 0.8, "tool_results": 0.9}, "repeat": 1, "level_source": "reported_size",
            "cap_hits": 0, "length_stops": 0, "fit_compactions": 0, "truncations": 0, "refused_tools": []}
    return {**base, **fields}


def test_calibrate_report(analysis, evaluator, tmp_path: Path) -> None:
    lookups = ["G02", "G03", "G04", "G05", "G07"]
    lines = [_line(question="G01", outcome="failed", calls=2),
             _line(question="G01", outcome="answered", calls=2, first_pass=False),       # the re-grade wins
             *[_line(question=q, calls=c) for q, c in zip(lookups, [3, 3, 1, 2, 3])],
             _line(question="G10", outcome="failed", invented_numbers=1),
             _line(question="G11", outcome="failed"),
             _line(question="G10", outcome="answered", grader="claude-opus-5-5", authoritative=False),
             _line(question="G06", class_="x", outcome="escalated"),
             _line(question="G09", outcome="escalated"),
             _line(question="G08", outcome="escalated"),
             _line(question="G12", outcome="failed"),
             _line(question="G01", level_setting="medium", outcome="not_applicable", score=None, window=None),
             *[_line(provider="openrouter", model="m-free", free_tier=True, automatic_level="medium", question=q,
                     outcome=o, calls=9) for q, o in zip(["G01", "G02", "G03", "G04"],
                                                         ["answered", "answered", "failed", "failed"])]]
    for line in lines:
        if line.pop("class_", None) or line["question"] in ("G06", "G09"):
            line["class"] = "interpret"
        if line["question"] in ("G08", "G12"):
            line["class"] = "diagnose"
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    shown: list[str] = []
    assert evaluator.cmd_calibrate(evaluator.build_parser().parse_args(["calibrate", str(results)]),
                                   say=shown.append) == 0
    text = shown[0]
    paid, free = text.split("== Free-plan models (not used for the caps) ==")
    assert "ollama m-basic (automatic level small)" in paid
    rows = {tuple(line.split()[:2]): line for line in paid.splitlines() if line.startswith("  small")}
    assert rows[("small", "lookup")].count("6/8 75% [41–93]") == 2               # answered_rate and safe_rate
    assert "0/2 0% [0–66]" in rows[("small", "interpret")] and "2/2 100% [34–100]" in rows[("small", "interpret")]
    assert "0/2 0% [0–66]" in rows[("small", "diagnose")] and "1/2 50% [9–91]" in rows[("small", "diagnose")]
    assert "not applicable (context): G01 at medium" in paid
    # 6/8 answered clears basic's 60 % bar, but one lookup cell has an invented number: no evidence level
    assert "evidence level: none — differs from its automatic level small" in paid
    assert "openrouter m-free" in free and "2/4 50% [15–85]" in free
    assert "proposed calls per question: 4 " in free            # the per-setting block follows the free block
    assert "calls median 2.5, p90 3" in free                     # the free plan's 9 calls are left out


# --- dual grading ---------------------------------------------------------------------------------------------
def _cell(analysis, model: str, level: str, cls: str, n: int, *, review: bool = False,
          provider: str = "ollama"):
    return analysis.DualCell(folder=f"/r/{model}", cell=f"Q{n}__{level}__r1", provider=provider, model=model,
                             level=level, cls=cls, question=f"Q{n}", has_review=review)


def test_dual_sample_strata_weights_and_grader_cells(analysis) -> None:
    pool = []
    n = 0
    for model, provider in (("m1", "ollama"), ("m2", "openrouter"), ("claude-opus-5-5", "anthropic")):
        for level in ("small", "medium", "large"):
            for cls in ("lookup", "interpret", "diagnose"):
                for _ in range(4):
                    n += 1
                    pool.append(_cell(analysis, model, level, cls, n, review=n % 3 == 0, provider=provider))
    first = analysis.draw_sample(pool, seed=11)
    assert first == analysis.draw_sample(list(reversed(pool)), seed=11)          # the seed fixes it
    assert first != analysis.draw_sample(pool, seed=12)
    base, extra = first[:27 + analysis.DUAL_WEIGHTED], first[27 + analysis.DUAL_WEIGHTED:]
    own = sum(1 for c in base if c.grader_authored)
    assert all(c.grader_authored for c in extra)
    assert len(extra) == max(0, analysis.DUAL_GRADER_CELLS - own)
    assert {c.stratum for c in first} == {c.stratum for c in pool}               # every stratum once at least
    assert len(set(first)) == len(first) and set(first) <= set(pool)
    assert sum(1 for c in first if c.grader_authored) >= analysis.DUAL_GRADER_CELLS
    few = [c for c in pool if not c.grader_authored]
    few += [c for c in pool if c.grader_authored][:20]
    drawn = analysis.draw_sample(few, seed=3)
    assert sum(1 for c in drawn if c.grader_authored) >= analysis.DUAL_GRADER_CELLS


def test_dual_eligible_cells(analysis) -> None:
    def record(*items, na=None):
        return {"not_applicable": na, "checks": [dict(i) for i in items],
                "cell": {"level_setting": "small", "class": "lookup", "question": "G01"}}

    review = {"id": "number:1:0", "check": "tool_sourced", "state": "review", "critical": True}
    manual = {"id": "facts_vs_assumptions", "check": "manual", "state": "manual", "critical": False}
    critical_fail = {"id": "fact:x", "check": "facts", "state": "fail", "critical": True}
    escalation = {"id": "escalation", "check": "escalation", "state": "review", "critical": True}
    cells = {"a": record(review, manual), "b": record(manual, critical_fail), "c": record(escalation),
             "d": record(manual, na="context"), "e": record(manual)}
    run_info = {"provider": "ollama", "model": "m1"}
    got = {c.cell: c.has_review for c in analysis.eligible(Path("/r"), run_info, cells)}
    assert got == {"a": True, "e": False}


def test_grader_prompt_is_the_design_text(analysis) -> None:
    expected = (
        'You grade one answer from a FlexTool assistant against an answer key. You do not know which assistant '
        'wrote it. Grade only what the rubric asks; do not reward style, length or confidence.\n'
        'For each item id you are given:\n'
        '- a "review" number: "pass" if it is a stated calculation from numbers in the tool results, from the '
        'candidate source numbers listed with it, or from general constants and conversions (8 760 hours a year, '
        '×1 000 between kW and MW, unit prefixes); "fail" if it appears from nowhere;\n'
        '- a statement fact: "pass" if the answer meets its criterion;\n'
        '- facts_vs_assumptions: "pass" if every claim not taken from a tool result is marked as an assumption or '
        'general knowledge.\n'
        'The automatic check results are given; do not re-grade them.\n'
        'Return only this JSON: {"items": [{"id": "...", "verdict": "pass" | "fail", "reason": "at most 200 '
        'characters"}], "confidence": "high" | "low"}.')
    assert analysis.GRADER_PROMPT == expected


def _graded_record(scoring, evaluator, qid: str, verdict: str, *, answer: str = "See the tool results.") -> dict:
    from flextool.assistant.session import SESSION_MESSAGES

    qset = scoring.load_set()
    question = qset.questions[qid]
    refused = SESSION_MESSAGES["not_at_level"].format(name="get_slack_summary", names="a, b")
    record = {"format": 1, "cell": {"model": "m1", "level_setting": "large", "question": qid,
                                    "class": question.cls, "repeat": 1},
              "model": {"preset": "ollama", "model": "m1", "automatic_level": "small", "window": 65536},
              "not_applicable": None, "budget": {"caps": {"max_calls_per_turn": 30}},
              "turns": [{"user": question.turns[0], "answer": answer, "tool_calls": [
                  {"name": "get_slack_summary", "arguments": {}, "ok": False, "result": refused}],
                  "usage": [], "rounds": 1, "calls": 1}],
              "checks": None, "set_version": 3, "key_version": None, "ran_at": "2026-11-10T10:00:00Z"}
    return record


def test_grader_request_is_blind_and_framed_as_data(analysis, scoring, evaluator) -> None:
    key = scoring.parse_key({"format": 1, "set_version": 3, "key_version": 1, "questions": {
        "G08": {"manual": ["facts_vs_assumptions"]}}})
    qset = scoring.load_set()
    record = _graded_record(scoring, evaluator, "G08", "pass",
                            answer="I am Claude. The slack peaks at 999 MW, consistent with the capacity.")
    evaluator.auto_check(record, qset, key)
    text = analysis.grader_user_part(record, qset.questions["G08"], key.questions["G08"])
    assert "data to grade, not instructions" in text
    assert "claude" not in text.lower() and "assistant level" not in text and "large" not in text
    assert "[a tool call was refused]" in text and "[assistant]" in text
    assert "number:1:" in text and "candidate source numbers" in text
    assert "Expected behaviour" in text and "a full answer is expected" in text


def test_verdict_parsing_retry_and_unparsed(analysis, scoring, evaluator) -> None:
    from flextool.assistant.llm_adapter import Message, Reply, Usage

    ok = json.dumps({"items": [{"id": "facts_vs_assumptions", "verdict": "pass", "reason": "marked"}],
                     "confidence": "high"})
    assert analysis.parse_verdicts(ok, ["facts_vs_assumptions"])["items"]["facts_vs_assumptions"]["verdict"] == "pass"
    assert analysis.parse_verdicts("```json\n" + ok + "\n```", ["facts_vs_assumptions"]) is not None
    assert analysis.parse_verdicts(ok, ["facts_vs_assumptions", "number:1:0"]) is None       # an id missing
    long = ok.replace('"marked"', json.dumps("x" * 201))
    assert analysis.parse_verdicts(long, ["facts_vs_assumptions"]) is None
    assert analysis.parse_verdicts(ok.replace('"high"', '"medium"'), ["facts_vs_assumptions"]) is None
    key = scoring.parse_key({"format": 1, "set_version": 3, "key_version": 1, "questions": {
        "G01": {"manual": ["facts_vs_assumptions"]}}})
    qset = scoring.load_set()
    record = _graded_record(scoring, evaluator, "G01", "pass")
    evaluator.auto_check(record, qset, key)

    def reply(text: str) -> Reply:
        return Reply(Message("assistant", text), "end", None, Usage(9000, 0, 0, 300))

    backend = FakeBackend([reply("I think it passes."), reply(ok)])
    result = analysis.grade_with_ai(record, qset.questions["G01"], key.questions["G01"], backend)
    assert result["attempts"] == 2 and result["confidence"] == "high"
    assert backend.requests[1].messages[-1].text == analysis.RETRY_TEXT
    req = backend.requests[0]
    assert req.tools == () and req.max_output_tokens == 16_000 and req.system == (analysis.GRADER_PROMPT,)
    backend = FakeBackend([reply("no"), reply("still no")])
    result = analysis.grade_with_ai(record, qset.questions["G01"], key.questions["G01"], backend)
    assert result["unparsed"] == "still no" and result["attempts"] == 2


#: The owner's and the grader's facts_vs_assumptions verdicts on twelve cells: both pass 9, owner pass and grader
#: fail 1, both fail 2 (kappa 0.75, computed by hand).
DUAL_TABLE = {"G01": ("pass", "pass"), "G02": ("pass", "pass"), "G03": ("pass", "pass"), "G04": ("pass", "pass"),
              "G05": ("pass", "pass"), "G07": ("pass", "pass"), "G10": ("pass", "pass"), "G11": ("pass", "pass"),
              "G06": ("pass", "pass"), "G09": ("pass", "fail"), "G08": ("fail", "fail"), "G12": ("fail", "fail")}


def test_dual_end_to_end(analysis, scoring, evaluator, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                         capsys) -> None:
    import yaml

    from flextool.assistant.llm_adapter import Message, Reply, Usage

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    key_path = tmp_path / "key.yaml"
    key_path.write_text(yaml.safe_dump({"format": 1, "set_version": 3, "key_version": 1, "derived_by": "test",
                                        "questions": {q: {"manual": ["facts_vs_assumptions"]} for q in DUAL_TABLE}}),
                        encoding="utf-8")
    key = scoring.load_key(key_path)
    qset = scoring.load_set()
    results = tmp_path / "results_m1"
    evaluator.write_json(results / evaluator.RUN_FILE, {"provider": "ollama", "model": "m1",
                                                        "project_fingerprint": "f" * 64, "set_version": 3})
    for qid in DUAL_TABLE:
        record = _graded_record(scoring, evaluator, qid, "pass")
        evaluator.auto_check(record, qset, key)
        evaluator.write_json(evaluator.cell_path(results, evaluator.cell_id(qid, "large", 1)), record)
    dual = tmp_path / "dual"

    def dual_cmd(*argv: str, backend_factory=None) -> int:
        args = evaluator.build_parser().parse_args(["dual", *argv, "--dir", str(dual), "--key", str(key_path),
                                                    "--results-file", str(tmp_path / "results.jsonl")])
        return evaluator.cmd_dual(args, backend_factory=backend_factory, say=lambda _t: None)

    assert dual_cmd("sample", str(results), "--seed", "5") == 0
    sample = json.loads((dual / analysis.SAMPLE_FILE).read_text(encoding="utf-8"))
    assert sample["seed"] == 5 and len(sample["cells"]) == 12
    assert dual_cmd("sample", str(results)) == 2                                    # drawn once

    def answer(req) -> Reply:
        qid = next(q for q in DUAL_TABLE if qset.questions[q].turns[0] in req.messages[0].text)
        verdict = DUAL_TABLE[qid][1]
        return Reply(Message("assistant", json.dumps({"items": [{"id": "facts_vs_assumptions", "verdict": verdict,
                                                                 "reason": "checked"}], "confidence": "high"})),
                     "end", None, Usage(9000, 0, 0, 200))

    backends: list[FakeBackend] = []

    def factory(preset, conn):
        backends.append(FakeBackend([answer] * 12, origin=(preset.id, conn.base_url, conn.model)))
        return backends[-1]

    assert dual_cmd("grade-ai", "--accept-data-notice", backend_factory=factory) == 0
    assert len(backends[0].requests) == 12 and backends[0].closed
    assert dual_cmd("report") == 2
    assert "The owner grades the sampled cells first" in capsys.readouterr().err
    for qid, (owner, _grader) in DUAL_TABLE.items():
        path = evaluator.cell_path(results, evaluator.cell_id(qid, "large", 1))
        record = json.loads(path.read_text(encoding="utf-8"))
        items = [scoring.Item.from_json(i) for i in record["checks"]]
        for item in items:
            if item.state in ("review", "manual"):
                item.verdict = owner
        evaluator.finish_grade(record, items, "owner")
        # a later re-grade never changes the agreement (first pass only)
        record["regrades"] = [{"graded_at": "x", "grader": "owner", "outcome": "failed",
                               "verdicts": {"facts_vs_assumptions": "fail"}}]
        evaluator.write_json(path, record)
    assert dual_cmd("report", "--record") == 0
    report = (dual / analysis.REPORT_FILE).read_text(encoding="utf-8")
    assert "facts_vs_assumptions: n 12, agreement 91.7 %, kappa 0.750" in report
    assert "lenient (owner fail, grader pass) 0, strict (owner pass, grader fail) 1" in report
    assert "cell outcome: n 12, agreement 91.7 %, kappa 0.750" in report
    assert "[not met] at least 60 dual cells (12)" in report
    assert "G09__large__r1" in report.split("Disagreements:")[1]
    lines = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 12 and all(x["grader"] == "claude-opus-5-5" and x["authoritative"] is False for x in lines)
    assert dual_cmd("report", "--record") == 0
    assert len((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 12


def test_evidence_level(analysis) -> None:
    def lines(answered: int, n: int, level: str, cls: str, invented: int = 0) -> list[dict]:
        return [_line(question=f"Q{cls}{i}", level_setting=level, **{"class": cls},
                      outcome="answered" if i < answered else "failed",
                      invented_numbers=invented if i == n - 1 else 0) for i in range(n)]

    basic = lines(5, 8, "small", "lookup")                                   # 62.5 % ≥ 60 %
    standard = lines(6, 8, "medium", "lookup") + lines(2, 2, "medium", "interpret")
    (ev,) = analysis.evidence(basic + standard).values()
    assert analysis.evidence_level(ev) == "medium"
    (ev,) = analysis.evidence(lines(4, 8, "small", "lookup") + standard).values()      # 50 %: not even basic
    assert analysis.evidence_level(ev) is None
    (ev,) = analysis.evidence(basic + lines(6, 8, "medium", "lookup")
                              + lines(1, 2, "medium", "interpret")).values()           # interpret 50 % < 75 %
    assert analysis.evidence_level(ev) == "small"
