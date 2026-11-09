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
                             say=lambda _text: None, allow_unsolved=True)


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
    assert "This is not a grounding project made by assistant_eval build" in capsys.readouterr().err
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


def _openrouter_key(model: str):
    from flextool.assistant import endpoint
    from flextool.assistant.providers import PRESETS

    preset = PRESETS["openrouter"]
    return endpoint.consent_key("openrouter", preset.base_url, endpoint.route_for(preset, preset.base_url,
                                                                                  model=model))


def test_accept_data_notice_is_not_kept(evaluator, grounding: Path, tmp_path: Path) -> None:
    """C6-M2: the calibration's acceptance is for that run only; the chat still asks."""
    from flextool.assistant import endpoint

    harness = Harness()
    assert run(evaluator, grounding, tmp_path / "out", "--questions", "G01", "--levels", "basic",
               harness=harness) == 0
    assert harness.requests == 1
    assert not endpoint.acks_path().exists()
    assert endpoint.needs_consent(_openrouter_key("other/model"))


def test_a_chat_acceptance_does_not_cover_the_calibration(evaluator, grounding: Path, tmp_path: Path,
                                                         capsys) -> None:
    """C6-M2: an acceptance kept by the chat is not read by the calibration, and a refused first run writes no
    results folder (the rerun with the flag is not refused as "has results")."""
    from flextool.assistant import endpoint

    endpoint.record_consent(_openrouter_key(MODEL))
    harness = Harness()
    out = tmp_path / "out"
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "openrouter", "--model", MODEL, "--out", str(out),
         "--questions", "G01", "--levels", "basic"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=lambda _t: None, allow_unsolved=True) == 3
    err = capsys.readouterr().err
    assert "Run again with --accept-data-notice" in err
    assert harness.requests == 0 and harness.probes == 0
    assert not (out / evaluator.RUN_FILE).exists()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=harness) == 0
    assert harness.requests == 1


def test_a_forwarded_ollama_model_asks_the_notice(evaluator, grounding: Path, tmp_path: Path, capsys) -> None:
    """C6-m1: the probe shows that the local Ollama forwards the model to Ollama cloud; that route's notice is
    asked before any question is sent, and accepting it keeps nothing."""
    from flextool.assistant import endpoint
    from flextool.assistant.probes import ModelFacts

    harness = Harness()

    def prober(preset, conn, cancel, *, needed):
        harness.probes += 1
        return ModelFacts(context_window=131_072, tools=True, remote_host="ollama.com", source="ollama")

    out = tmp_path / "out"

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "ollama", "--model", "gpt-oss:120b", "--out",
             str(out), "--questions", "G01", "--levels", "basic", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=prober, info=None,
                                 say=lambda _t: None, allow_unsolved=True)

    assert go() == 3
    err = capsys.readouterr().err
    assert "Ollama cloud" in err and "--resume --accept-data-notice" in err
    assert harness.probes == 1 and harness.requests == 0
    assert go("--resume", "--accept-data-notice") == 0
    assert harness.requests == 1
    assert harness.backends[0].origin[0] == "ollama"
    assert not endpoint.acks_path().exists()


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
                                 say=said.append, allow_unsolved=True)

    assert go() == 0
    assert said[-1] == evaluator.DAILY_DONE_REMOTE                  # OpenRouter: the rerun needs the flag again
    assert sorted(p.stem for p in (out / "cells").glob("*.json")) == ["G01__small__r1", "G03__small__r1"]
    assert harness.requests == 2
    assert go("--resume") == 0                              # the same day: nothing more is sent
    assert said[-1] == evaluator.DAILY_DONE_REMOTE and harness.requests == 2
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
                             say=lambda _t: None, allow_unsolved=True) == 0
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
        args = evaluator.build_parser().parse_args(["grade", str(out), "--single-model", "--dir",
                                                    str(tmp_path / "grading"), "--key", str(key)])
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
                             say=lambda _t: None, allow_unsolved=True) == 0
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
    from flextool.assistant import endpoint

    assert not endpoint.acks_path().exists()                    # C6-M2: the acceptance was for that run only
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


def test_the_calibration_tools_keep_no_acceptance() -> None:
    """C6-M2: no calibration path reads or writes the chat's acknowledgements."""
    tools = Path(__file__).resolve().parents[3] / "tools"
    for path in sorted(tools.glob("assistant_eval*.py")):
        text = path.read_text(encoding="utf-8")
        assert "needs_consent" not in text and "record_consent" not in text, path.name


# --- G14: blind grading across models (C6-M1, m2, n3) -------------------------------------------------------
#: Not free-plan models, so their requests are not paced.
ALPHA = "vendor/alpha-one"
BETA = "vendor/beta-two"
PAID = "vendor/model-a"


def _key_file(path: Path, qids: tuple[str, ...]) -> Path:
    """A non-placeholder key for *qids* that only exercises grading: the token coal_plant, critical."""
    import yaml

    note = "exercises the grading only"
    path.write_text(yaml.safe_dump({
        "format": 1, "set_version": gp.SET_VERSION, "key_version": 7, "derived_by": note, "sources": [],
        "questions": {q: {"facts": [{"id": "unit", "kind": "token", "value": "coal_plant", "critical": True,
                                     "derivation": note}], "manual": ["facts_vs_assumptions"]} for q in qids}}),
        encoding="utf-8")
    return path


def _results(evaluator, grounding: Path, out: Path, model: str, levels: str = "basic") -> Path:
    assert run(evaluator, grounding, out, "--questions", "G01,G03,G04", "--levels", levels, harness=Harness(),
               model=model) == 0
    return out


def _grade(evaluator, key: Path, grading: Path, *folders: Path, ask, said: list[str], extra=()) -> int:
    args = evaluator.build_parser().parse_args(["grade", *map(str, folders), "--dir", str(grading), "--key",
                                                str(key), *extra])
    return evaluator.cmd_grade(args, ask=ask, say=said.append)


def _cells_shown(said: list[str]) -> list[str]:
    return [line for line in said if line.startswith("Cell #")]


def test_grade_shuffles_across_folders(evaluator, grounding: Path, tmp_path: Path, capsys) -> None:
    """C6-M1: the cells of two models are graded in one blind order; nothing shown names a model or a cell."""
    alpha = _results(evaluator, grounding, tmp_path / "res_alpha", ALPHA)
    beta = _results(evaluator, grounding, tmp_path / "res_beta", BETA)
    key = _key_file(tmp_path / "key.yaml", ("G01", "G03", "G04"))
    grading = tmp_path / "grading"
    evaluator.write_json(grading / evaluator.GRADING_FILE, {"seed": 11, "order": [], "tokens": {}})
    said: list[str] = []
    assert _grade(evaluator, key, grading, alpha, ask=lambda _t: "p", said=said) == 2
    assert "Grading one model is not blind" in capsys.readouterr().err
    assert _grade(evaluator, key, grading, alpha, beta, ask=lambda t: said.append(t) or "p", said=said) == 0
    state = json.loads((grading / evaluator.GRADING_FILE).read_text(encoding="utf-8"))
    order = [tuple(k) for k in state["order"]]
    cids = ("G01__small__r1", "G03__small__r1", "G04__small__r1")
    assert sorted(order) == sorted((str(f.resolve()), c) for f in (alpha, beta) for c in cids)
    folders = [Path(f).name for f, _c in order]
    assert folders != sorted(folders) and folders != sorted(folders, reverse=True)        # mixed for seed 11
    assert _cells_shown(said) == [f"Cell #{n}" for n in range(1, 7)]
    assert {name: tuple(k) for name, k in state["tokens"].items()} == {f"#{n}": k for n, k in enumerate(order, 1)}
    view = "\n".join(said)
    for hidden in ("alpha-one", "beta-two", "vendor", "openrouter", "OpenRouter", "res_alpha", "res_beta",
                   "__small__", "small", *cids):
        assert hidden not in view, hidden
    for f in (alpha, beta):
        assert all(r["outcome"] == "answered" for r in evaluator.read_cells(f).values())


def test_late_folder_is_mixed_into_the_tail(evaluator, grounding: Path, tmp_path: Path) -> None:
    """C6-M1: a folder added later is shuffled into the cells not yet shown, not appended as one block."""
    alpha = _results(evaluator, grounding, tmp_path / "res_alpha", ALPHA, levels="basic,standard")
    beta = _results(evaluator, grounding, tmp_path / "res_beta", BETA)
    key = _key_file(tmp_path / "key.yaml", ("G01", "G03", "G04"))
    grading = tmp_path / "grading"
    evaluator.write_json(grading / evaluator.GRADING_FILE, {"seed": 11, "order": [], "tokens": {}})
    said: list[str] = []
    assert _grade(evaluator, key, grading, alpha, ask=lambda _t: "p", said=said,
                  extra=("--single-model", "--sitting", "2")) == 0
    first = json.loads((grading / evaluator.GRADING_FILE).read_text(encoding="utf-8"))["order"]
    assert len(first) == 6 and _cells_shown(said) == ["Cell #1", "Cell #2"]
    said.clear()
    assert _grade(evaluator, key, grading, alpha, beta, ask=lambda _t: "p", said=said) == 0
    order = json.loads((grading / evaluator.GRADING_FILE).read_text(encoding="utf-8"))["order"]
    assert order[:2] == first[:2]
    assert [k for k in order if k in first] == first                    # alpha's rest keeps its order
    beta_at = [n for n, (f, _c) in enumerate(order, 1) if Path(f).name == "res_beta"]
    assert len(beta_at) == 3 and min(beta_at) > 2 and beta_at != [7, 8, 9]     # interleaved for seed 11
    assert _cells_shown(said) == [f"Cell #{n}" for n in range(3, 10)]


@pytest.mark.parametrize("text, blinded", [
    ("As Sonnet 4.5 I checked it.", "As [assistant] 4.5 I checked it."),
    ("(Opus checked)", "([assistant] checked)"),
    ("GPT-oss said so", "[assistant] said so"),
    ("Kimi thinks", "[assistant] thinks"),
    ("Your level: basic.", "Your [level]."),
    ("the advanced level would trace it", "the [level] would trace it"),
    ("showing 20 of 340 rows", "showing [part] of 340 rows"),
    ("showing the first 40 of 900 characters", "showing [part] of 900 characters"),
    ("The result is larger than 12000 characters", "The result is larger than [limit] characters"),
    ("alpha-one wrote this", "[assistant] wrote this"),
    ("a large share of demand", "a large share of demand"),
    ("the metadata of a meta-analysis", "the metadata of a meta-analysis"),
])
def test_blind_names_and_levels(scoring, text: str, blinded: str) -> None:
    """C6-m2: family names, level phrases, result caps and the folder's own names are blinded; ordinary words
    stay."""
    assert scoring.blind(text, ["alpha-one"]) == blinded


def test_grader_view_blinds_arguments(evaluator, scoring) -> None:
    record = _lookup_record(scoring, evaluator, "small", 6, arguments={"note": "claude"})
    view = evaluator.grader_view(record, *_g01(scoring), evaluator.auto_check(record, *_g01_set(scoring)))
    assert '{"note": "[assistant]"}' in view and "claude" not in view.lower()


def _g01_set(scoring):
    key = scoring.parse_key({"format": 1, "set_version": 3, "key_version": 1, "questions": {
        "G01": {"facts": [{"id": "unit", "kind": "token", "value": "coal_plant", "critical": True,
                           "derivation": "test"}], "manual": ["facts_vs_assumptions"]}}})
    return scoring.load_set(), key


def _g01(scoring):
    qset, key = _g01_set(scoring)
    return qset.questions["G01"], key.questions["G01"]


def _lookup_record(scoring, evaluator, level: str, cap: int, *, arguments=None, answer: str = "coal_plant.") -> dict:
    question = scoring.load_set().questions["G01"]
    return {"format": 1, "cell": {"model": "m1", "level_setting": level, "question": "G01", "class": "lookup",
                                  "repeat": 1},
            "model": {"preset": "ollama", "model": "m1"}, "not_applicable": None,
            "budget": {"caps": {"max_calls_per_turn": cap}},
            "turns": [{"user": question.turns[0], "answer": answer, "tool_calls": [
                {"name": "list_runs", "arguments": arguments or {}, "ok": True, "result": '{"runs": []}'}],
                "usage": [], "rounds": 2, "calls": 1}],
            "checks": None, "set_version": 3, "key_version": None, "ran_at": "2026-11-10T10:00:00Z"}


def test_grader_material_has_no_cap(evaluator, analysis, scoring) -> None:
    """C6-m2: a lookup cell at basic (cap 6) and at advanced (cap 30) looks the same to both graders."""
    qset, key = _g01_set(scoring)
    small = _lookup_record(scoring, evaluator, "small", 6)
    large = _lookup_record(scoring, evaluator, "large", 30)
    views = [evaluator.grader_view(r, *_g01(scoring), evaluator.auto_check(r, qset, key)) for r in (small, large)]
    assert views[0] == views[1] and "cap 6" not in views[0]
    parts = [analysis.grader_user_part(r, qset.questions["G01"], key.questions["G01"]) for r in (small, large)]
    assert parts[0] == parts[1] and "cap 6" not in parts[0] and "calls per turn" not in parts[0]


@pytest.mark.parametrize("leak", ["ran_at", "glued_name"])
def test_leak_scan_blocks_grading(evaluator, grounding: Path, tmp_path: Path, capsys, leak: str) -> None:
    """C6-m2: material the blinding cannot clean stops the whole sitting before the first question."""
    alpha = _results(evaluator, grounding, tmp_path / "res_alpha", ALPHA)
    path = evaluator.cell_path(alpha, "G03__small__r1")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["turns"][0]["answer"] += (f" Checked at {record['ran_at']}." if leak == "ran_at"
                                     else " Signed: teamalpha-one.")
    evaluator.write_json(path, record)
    before = {p.name: p.read_bytes() for p in (alpha / "cells").glob("*.json")}
    key = _key_file(tmp_path / "key.yaml", ("G01", "G03", "G04"))
    asked: list[str] = []
    assert _grade(evaluator, key, tmp_path / "grading", alpha, ask=lambda t: asked.append(t) or "p", said=[],
                  extra=("--single-model",)) == 2
    err = capsys.readouterr().err
    assert "the grader material reveals" in err and "nothing was graded" in err
    assert ("the time of the run" if leak == "ran_at" else "the model's name") in err
    assert "alpha-one" not in err and record["ran_at"] not in err
    assert asked == [] and before == {p.name: p.read_bytes() for p in (alpha / "cells").glob("*.json")}


@pytest.mark.parametrize("material, kinds", [
    ("plain text", []),
    ("[assistant] wrote it, showing [part] of 30 rows", []),
    ("Sonnet wrote it", ["a model or provider name"]),
    ("at the standard level", ["the level"]),
    ("calls per turn [2], cap 6", ["a call cap"]),
    ("showing 20 of 30 rows", ["a result cap"]),
    ("The tool limit for this question is reached. Answer with what you have, and say what is still unknown.",
     ["a level message"]),
    ("by beta-two", ["the model's name"]),
    ("at 2026-11-10T10:00:00Z", ["the time of the run", "a time"]),
    ("run at 2026-01-02T03:04", ["a time"]),
])
def test_leak_scan_kinds(scoring, material: str, kinds: list[str]) -> None:
    """The planted leaks: family names, levels, call and result caps, level messages, names, times."""
    assert scoring.leak_scan(material, names=["beta-two"], stamps=["2026-11-10T10:00:00Z"]) == kinds
    assert scoring.leak_scan("run at 2026-01-02T03:04", names=[], stamps=[], sources='{"t": "2026-01-02T03:04"}') == []


def test_grade_interrupt_keeps_given_verdicts(evaluator, grounding: Path, tmp_path: Path, capsys,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    """C6-n3: Ctrl-C keeps the verdicts already given; an interrupted re-grade changes nothing; an unknown cell
    is refused."""
    alpha = _results(evaluator, grounding, tmp_path / "res_alpha", ALPHA)
    key = _key_file(tmp_path / "key.yaml", ("G01", "G03", "G04"))
    grading = tmp_path / "grading"
    answers: list = ["pass", KeyboardInterrupt]
    prompts: list[str] = []

    def fake_input(text: str) -> str:
        prompts.append(text)
        item = answers.pop(0)
        if item is KeyboardInterrupt:
            raise KeyboardInterrupt
        return item

    monkeypatch.setattr("builtins.input", fake_input)

    def grade(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(["grade", str(alpha), "--single-model", "--dir", str(grading),
                                                    "--key", str(key), *extra])
        return evaluator.cmd_grade(args, say=lambda _t: None)

    assert grade() == 0
    order = json.loads((grading / evaluator.GRADING_FILE).read_text(encoding="utf-8"))["order"]
    first = evaluator.cell_path(alpha, order[0][1])
    record = json.loads(first.read_text(encoding="utf-8"))
    verdicts = {i["id"]: i.get("verdict") for i in record["checks"]}
    assert verdicts["number:1:0"] == "pass" and verdicts["facts_vs_assumptions"] is None
    assert record["outcome"] is None
    assert grade("--regrade", "#1") == 2 and "Grade cell #1 first." in capsys.readouterr().err
    prompts.clear()
    answers[:] = ["pass"] * 5
    assert grade() == 0
    assert len(prompts) == 5                                  # the first cell's one left, two per other cell
    assert all(r["outcome"] == "answered" for r in evaluator.read_cells(alpha).values())
    graded = first.read_bytes()
    answers[:] = [KeyboardInterrupt]
    assert grade("--regrade", "#1") == 0
    assert first.read_bytes() == graded
    assert grade("--regrade", "#99") == 2 and "No graded cell #99." in capsys.readouterr().err
    answers[:] = ["fail", "pass"]
    assert grade("--regrade", "#1") == 0
    record = json.loads(first.read_text(encoding="utf-8"))
    assert record["outcome"] == "answered" and record["regrades"][0]["outcome"] == "failed"


# --- G14: dual pool, task support and statistics (C6-m3, m6, n1, m2) ----------------------------------------
def _items(scoring, *specs):
    return [scoring.Item(id=i, check=c, state=st, critical=cr) for i, c, st, cr in specs]


def test_dual_pool_keeps_escalations(analysis, scoring) -> None:
    """C6-m3: an escalation cell whose critical facts fail as expected is still in the pool; cells that no
    verdict can change are not."""
    import itertools

    def record(*items):
        return {"not_applicable": None, "checks": [i.to_json() for i in items],
                "cell": {"level_setting": "small", "class": "diagnose", "question": "G08"}}

    fact_fail = ("fact:x", "facts", "fail", True)
    number = ("number:1:0", "tool_sourced", "review", True)
    escalation = ("escalation", "escalation", "pass", True)
    fva = ("facts_vs_assumptions", "manual", "manual", False)
    forbidden = ("forbidden:0", "forbidden", "fail", True)
    cells = {"esc": record(*_items(scoring, fact_fail, number, escalation, fva)),
             "esc_forbidden": record(*_items(scoring, fact_fail, number, escalation, fva, forbidden)),
             "fact_fail": record(*_items(scoring, fact_fail, number, fva))}
    got = [c.cell for c in analysis.eligible(Path("/r"), {"provider": "ollama", "model": "m1"}, cells)]
    assert got == ["esc"]
    # outcome_fixed is outcome()'s own rule: brute force over every verdict of up to three graded items
    pool = [fact_fail, number, escalation, fva, forbidden, ("tool_sourced", "tool_sourced", "pass", True),
            ("number:1:1", "tool_sourced", "review", True), ("units", "units", "pass", False),
            ("escalation", "escalation", "review", True), ("fact:s", "facts", "manual", True),
            ("tools", "tools", "fail", False)]
    checked = 0
    for size in range(1, 5):
        for combo in itertools.combinations(pool, size):
            if len({c[0] for c in combo}) < size:
                continue
            items = _items(scoring, *combo)
            graded = [i for i in items if i.state in ("review", "manual")]
            if len(graded) > 3:
                continue
            outcomes = set()
            for verdicts in itertools.product(("pass", "fail"), repeat=len(graded)):
                for i, v in zip(graded, verdicts):
                    i.verdict = v
                outcomes.add(scoring.outcome(items))
            for i in graded:
                i.verdict = None
            assert scoring.outcome_fixed(items) == (len(outcomes) == 1), combo
            checked += 1
    assert checked > 100


def test_task_support_order_independent(analysis) -> None:
    """C6-m6: each escalation rate is held to its own model's bar, whatever the order of the lines."""
    def lines(model: str, auto: str, level: str, cls: str, outcome: str, k: int, n: int) -> list[dict]:
        return [_line(provider="ollama", model=model, automatic_level=auto, level_setting=level, question=f"Q{i}",
                      outcome=outcome if i < k else "failed", **{"class": cls}) for i in range(n)]

    a = lines("A", "small", "small", "interpret", "escalated", 13, 20)          # 65 %: basic bar 60 %
    b = lines("B", "medium", "medium", "lookup", "answered", 20, 20)
    big = lines("L", "large", "large", "interpret", "answered", 16, 20)         # 80 %: advanced bar 75 %
    for order in ([b, a, big], [a, b, big]):
        models = list(analysis.evidence([x for part in order for x in part]).values())
        (row,) = [t for t in analysis.task_support(models) if t.startswith("  large / interpret")]
        assert row.endswith("supports the level text as drafted"), row
        assert "escalated below 13/20 65%" in row


def test_bootstrap_counts_undefined(analysis) -> None:
    """C6-n1: resamples without a kappa are counted and reported."""
    p, f = "pass", "fail"
    boot = analysis.bootstrap_kappa({"q1": [(p, p)], "q2": [(f, f)]}, reps=2000, seed=3)
    assert boot.interval == (1.0, 1.0) and boot.reps == 2000
    # a resample is undefined when both draws are the same question: P = 1/2, sd = sqrt(2000 / 4) ≈ 22
    assert 900 <= boot.undefined <= 1100
    note = analysis.bootstrap_note(boot)
    assert f"; {boot.undefined} of 2000 resamples undefined (all verdicts alike), interval unreliable" == note
    mixed = analysis.bootstrap_kappa({"q1": [(p, p), (f, f)], "q2": [(p, p), (f, f)]}, reps=500)
    assert mixed.undefined == 0 and mixed.interval == (1.0, 1.0) and analysis.bootstrap_note(mixed) == ""
    alike = analysis.bootstrap_kappa({"q1": [(p, p)], "q2": [(p, p)]}, reps=300)
    assert alike.interval is None and alike.undefined == 300
    rows = [{"owner": o, "grader": g, "question": q} for q, (o, g) in (("q1", (p, p)), ("q2", (f, f)))]
    summary = analysis.criterion_summary(rows, seed=3)
    assert summary["bootstrap"].undefined == boot.undefined


def test_statistics_by_hand(analysis) -> None:
    """Wilson 13/20: z² = 3.8415, centre (0.65 + 0.0960) / 1.1921 = 0.6258, half 1.6442 · 0.11737 = 0.1930.
    Kappa of (p,p), (f,f), (p,unparsed), (f,f): observed 3/4 = 0.75; expected 0.5·0.25 (pass) + 0.5·0.5 (fail)
    + 0·0.25 (unparsed) = 0.375; kappa (0.75 − 0.375) / 0.625 = 0.6."""
    assert analysis.wilson(13, 20) == pytest.approx((0.4329, 0.8188), abs=5e-5)
    pairs = [("pass", "pass"), ("fail", "fail"), ("pass", "unparsed"), ("fail", "fail")]
    assert analysis.kappa(pairs) == pytest.approx((0.75, 0.375, 0.6), abs=1e-12)


def test_dual_grade_ai_leak_scan_sends_nothing(analysis, scoring, evaluator, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """C6-m2: a sampled cell whose material the blinding cannot clean stops grade-ai before any request."""
    import yaml

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
    key_path = tmp_path / "key.yaml"
    key_path.write_text(yaml.safe_dump({"format": 1, "set_version": 3, "key_version": 1, "derived_by": "test",
                                        "questions": {"G01": {"manual": ["facts_vs_assumptions"]}}}),
                        encoding="utf-8")
    key = scoring.load_key(key_path)
    results = tmp_path / "results_m1"
    evaluator.write_json(results / evaluator.RUN_FILE, {"provider": "ollama", "model": "gamma-three:14b",
                                                        "project_fingerprint": "f" * 64, "set_version": 3})
    record = _graded_record(scoring, evaluator, "G01", "pass", answer="Written by gamma-three-like me.")
    evaluator.auto_check(record, scoring.load_set(), key)
    evaluator.write_json(evaluator.cell_path(results, "G01__large__r1"), record)
    dual = tmp_path / "dual"
    argv = ["--dir", str(dual), "--key", str(key_path), "--results-file", str(tmp_path / "r.jsonl")]
    assert evaluator.cmd_dual(evaluator.build_parser().parse_args(["dual", "sample", str(results), *argv,
                                                                   "--seed", "1"]), say=lambda _t: None) == 0
    backends: list[FakeBackend] = []

    def factory(preset, conn):
        backends.append(FakeBackend([]))
        return backends[-1]

    blinded = analysis.grader_user_part(record, scoring.load_set().questions["G01"], key.questions["G01"],
                                        ["gamma-three"])
    assert "gamma-three" not in blinded                       # the token itself is blinded ...
    record["turns"][0]["answer"] = "Written by gamma-threelike me."    # ... a glued one is not: the scan stops it
    evaluator.write_json(evaluator.cell_path(results, "G01__large__r1"), record)
    args = evaluator.build_parser().parse_args(["dual", "grade-ai", "--accept-data-notice", *argv])
    assert evaluator.cmd_dual(args, backend_factory=factory, say=lambda _t: None) == 2
    err = capsys.readouterr().err
    assert "the grader material reveals the model's name; nothing was sent" in err
    assert backends == [] and not (dual / analysis.SEALED_FILE).exists()


# --- G15: runs, resume and the marker (C6-M5, m4, m7) -------------------------------------------------------
class Scripted(Harness):
    """Each backend takes the next script of *scripts* (one list of items per cell); then plain answers."""

    def __init__(self, scripts: list[list]) -> None:
        super().__init__()
        self.scripts = list(scripts)

    def factory(self, preset, conn):
        if not self.scripts:
            return super().factory(preset, conn)
        backend = FakeBackend(self.scripts.pop(0), origin=(preset.id, conn.base_url, conn.model))
        self.backends.append(backend)
        return backend


def _answer(text: str = "coal_plant has 500 MW."):
    from flextool.assistant.llm_adapter import Message, Reply, Usage

    return Reply(Message("assistant", text), "end", None, Usage(1200, 0, 0, 20))


def _error(code: str):
    from flextool.assistant import llm_errors

    cls = {"connection": llm_errors.Unreachable, "timeout": llm_errors.TimedOut, "busy": llm_errors.ServerBusy,
           "auth": llm_errors.AuthFailed, "model": llm_errors.ModelNotFound,
           "routing": llm_errors.NoMatchingEndpoint}[code]
    key = "connection_remote" if code == "connection" else code
    return llm_errors.make_error(cls, key, llm_errors.ErrorContext(provider="OpenRouter", host="openrouter.ai"))


@pytest.fixture
def no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session's own retries without their waits (a connection retry after 0 s, no busy retries)."""
    import functools

    from flextool.assistant import session
    from flextool.assistant.llm_retry import RetryPolicy

    monkeypatch.setattr(session, "RetryPolicy", functools.partial(RetryPolicy, max_retries=0, connection_wait=0.0))


@pytest.mark.parametrize("code", ["connection", "timeout", "busy", "auth", "model", "routing"])
def test_transient_error_not_saved_and_resumed(evaluator, grounding: Path, tmp_path: Path, code: str,
                                               no_waits) -> None:
    """C6-M5: an error that is not the model's behaviour stops the run with the cell unsaved; --resume asks it."""
    out = tmp_path / "out"
    said: list[str] = []
    harness = Scripted([[_answer()], [_error(code)] * 4])

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
             "--accept-data-notice", "--questions", "G01,G03,G04", "--levels", "basic", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                                 say=said.append, allow_unsolved=True)

    assert go() == 1
    assert sorted(p.stem for p in (out / "cells").glob("*.json")) == ["G01__small__r1"]
    assert said[-1] == ("G03__small__r1: nothing was saved for this cell. Run again with --resume "
                        "--accept-data-notice.")
    assert go("--resume") == 0                                    # the fixed backend: plain answers
    cells = evaluator.read_cells(out)
    assert sorted(cells) == ["G01__small__r1", "G03__small__r1", "G04__small__r1"]
    assert all(t["stop"] == "end" for r in cells.values() for t in r["turns"])


def test_saved_error_cell_reruns_and_graded_stop_kept(evaluator, grounding: Path, tmp_path: Path) -> None:
    """C6-M5: a cell an older folder saved with a connection stop is asked again; a length stop is a graded
    outcome and kept."""
    out = tmp_path / "out"
    harness = Harness()
    assert run(evaluator, grounding, out, "--questions", "G01,G03", "--levels", "basic", harness=harness, model=PAID) == 0
    for cid, code in (("G01__small__r1", "connection"), ("G03__small__r1", "length")):
        path = evaluator.cell_path(out, cid)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["turns"][0].update(stop=code, error={"code": code, "message": "scripted"})
        evaluator.write_json(path, record)
    length_before = evaluator.cell_path(out, "G03__small__r1").read_bytes()
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01,G03", "--levels", "basic",
               harness=harness, model=PAID) == 0
    assert harness.requests == 3
    assert json.loads(evaluator.cell_path(out, "G01__small__r1").read_text())["turns"][0]["stop"] == "end"
    assert evaluator.cell_path(out, "G03__small__r1").read_bytes() == length_before
    spent = json.loads((out / evaluator.SPENT_FILE).read_text(encoding="utf-8"))
    assert [(e["cell"], e["requests"], e["code"]) for e in spent] == [("G01__small__r1", 1, "replaced")]


def test_spend_of_a_failed_cell_counts(evaluator, grounding: Path, tmp_path: Path, no_waits) -> None:
    """C6-M5: the requests a stopped cell sent count against the day: 1 (the tool call) + 1 (the resumed cell)
    reaches the budget of 2."""
    from flextool.assistant.llm_adapter import Message, Reply, ToolCall, Usage

    out = tmp_path / "out"
    said: list[str] = []
    call = Reply(Message("assistant", "", tool_calls=[ToolCall("c1", "list_runs", {}, "{}")]), "tools", None,
                 Usage(1500, 0, 0, 10))
    harness = Scripted([[call, _error("connection"), _error("connection"), _error("connection")]])

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
             "--accept-data-notice", "--questions", "G01,G03", "--levels", "basic", "--max-requests-per-day", "2",
             *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                                 say=said.append, allow_unsolved=True)

    assert go() == 1
    spent = json.loads((out / evaluator.SPENT_FILE).read_text(encoding="utf-8"))
    assert [(e["cell"], e["requests"], e["code"]) for e in spent] == [("G01__small__r1", 1, "connection")]
    assert not (out / "cells").exists() or not any((out / "cells").iterdir())
    sent = harness.requests
    assert go("--resume") == 0
    assert harness.requests - sent == 1 and said[-1] == evaluator.DAILY_DONE_REMOTE
    assert sorted(p.stem for p in (out / "cells").glob("*.json")) == ["G01__small__r1"]


def test_resume_identity(evaluator, grounding: Path, tmp_path: Path, capsys) -> None:
    """C6-m4: the address, the free plan, the Ollama contexts and the data-collection knob are part of a folder's
    identity; the levels only pick cells."""
    from flextool.assistant.probes import ModelFacts

    out = tmp_path / "out"
    harness = Harness()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=harness,
               model=PAID) == 0
    capsys.readouterr()
    for extra in (("--base-url", "https://proxy.example.test/api/v1"), ("--free-tier", "on"),
                  ("--relax-data-collection",)):
        assert run(evaluator, grounding, out, "--resume", "--questions", "G01", "--levels", "basic", *extra,
                   harness=harness, model=PAID) == 2, extra
        assert evaluator.OTHER_RESULTS in capsys.readouterr().err
    assert harness.requests == 1 and harness.probes == 1
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01", "--levels", "basic,standard",
               harness=harness, model=PAID) == 0
    assert harness.requests == 2
    # an older run.json without the new fields is never continued
    run_file = out / evaluator.RUN_FILE
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    run_file.write_text(json.dumps({k: v for k, v in saved.items() if k != "ollama_context"}), encoding="utf-8")
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01", harness=harness, model=PAID) == 2
    assert evaluator.OTHER_RESULTS in capsys.readouterr().err

    def prober(preset, conn, cancel, *, needed):
        return ModelFacts(context_window=131_072, loaded_context=16_384, tools=True, source="ollama")

    local = tmp_path / "local"

    def ollama(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "ollama", "--model", "gpt-oss:20b", "--out",
             str(local), "--questions", "G01", "--levels", "basic", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=prober,
                                 info=lambda _b, _m: {"num_ctx": 16_384, "size": 1, "size_vram": 1},
                                 say=lambda _t: None, allow_unsolved=True)

    assert ollama() == 0
    sent = harness.requests
    assert ollama("--resume", "--ollama-context", "basic=16384") == 2
    assert evaluator.OTHER_RESULTS in capsys.readouterr().err and harness.requests == sent


def test_not_applicable_rerun(evaluator, analysis, grounding: Path, tmp_path: Path) -> None:
    """C6-m4: --resume runs a not-applicable cell whose setting now fits; a later line of a cell replaces its
    not_applicable line, and two free-plan settings stay apart."""
    from flextool.assistant.probes import ModelFacts

    out = tmp_path / "out"
    harness = Harness()
    sizes = {"num_ctx": 16_384, "size": 14_000, "size_vram": 10_000}           # does not fit in graphics memory

    def prober(preset, conn, cancel, *, needed):
        return ModelFacts(context_window=131_072, loaded_context=16_384, tools=True, source="ollama")

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "ollama", "--model", "gpt-oss:20b", "--out",
             str(out), "--questions", "G01,G03", "--levels", "basic", "--ollama-context", "basic=16384", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=prober, info=lambda _b, _m: sizes,
                                 say=lambda _t: None, allow_unsolved=True)

    assert go() == 0
    assert all(r["not_applicable"] == "context" for r in evaluator.read_cells(out).values())
    assert harness.requests == 0
    still = evaluator.cell_path(out, "G01__small__r1").read_bytes()
    assert go("--resume") == 0 and harness.requests == 0                      # still not applicable: kept
    assert evaluator.cell_path(out, "G01__small__r1").read_bytes() == still
    sizes["size_vram"] = 14_000
    assert go("--resume") == 0 and harness.requests == 2
    assert all(r["not_applicable"] is None and r["turns"] for r in evaluator.read_cells(out).values())

    na = _line(question="G01", outcome="not_applicable", key_version=None, score=None)
    answered = _line(question="G01", outcome="answered", key_version=2)
    assert analysis.authoritative([na, answered]) == [answered]
    free = _line(question="G01", free_tier=True)
    assert len(analysis.authoritative([_line(question="G01"), free])) == 2


def _write_status(folder: Path, run_id: str) -> None:
    from flextool.run_record._owner import RUN_STATUS_FILE

    folder.mkdir(parents=True, exist_ok=True)
    (folder / RUN_STATUS_FILE).write_text(json.dumps({"run_id": run_id, "state": "succeeded"}), encoding="utf-8")


def test_marker_checks(evaluator, tmp_path: Path, capsys) -> None:
    """C6-m7: an unsolved build, another input source, another run's results and a changed run are refused;
    the marker's own runs and the database's sidecar files are not."""
    project = gp.build_grounding_project(tmp_path / "p", solve=False).path
    with pytest.raises(evaluator.Refused, match="built without its runs"):
        evaluator.check_marker(project)
    assert evaluator.check_marker(project, allow_unsolved=True)["solved"] is False
    marker_path = project / gp.MARKER_FILE
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_path.write_text(json.dumps({**marker, "solved": True, "runs": {"coal_1": "id-a"}}), encoding="utf-8")
    _write_status(project / "output_parquet" / "coal_1", "id-b")
    with pytest.raises(evaluator.Refused, match=r"changed since: the run in output_parquet/coal_1"):
        evaluator.check_marker(project)
    _write_status(project / "output_parquet" / "coal_1", "id-a")
    (project / "input_sources" / (gp.SOURCE_NAME + "-journal")).write_bytes(b"")
    (project / "output_csv" / "empty_1").mkdir()
    assert evaluator.check_marker(project)["runs"] == {"coal_1": "id-a"}
    _write_status(project / "output_parquet" / "x_1", "id-x")
    with pytest.raises(evaluator.Refused, match=r"changed since: results output_parquet/x_1 of another run"):
        evaluator.check_marker(project)
    for f in (project / "output_parquet" / "x_1").iterdir():
        f.unlink()
    (project / "input_sources" / "other.sqlite").write_bytes(b"")
    with pytest.raises(evaluator.Refused, match="changed since: another input source other.sqlite"):
        evaluator.check_marker(project)
    marker_path.write_text(json.dumps({**marker, "format": 1}), encoding="utf-8")
    with pytest.raises(evaluator.Refused, match=r"^This is not a grounding project made by assistant_eval build\.$"):
        evaluator.check_marker(project)
    harness = Harness()
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(project), "--provider", "openrouter", "--model", PAID, "--out",
         str(tmp_path / "out"), "--accept-data-notice"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=lambda _t: None) == 2
    assert harness.requests == 0 and harness.probes == 0
