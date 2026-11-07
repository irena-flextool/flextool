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
    assert small["budget"]["caps"]["max_calls_per_turn"] == 3
    assert large["budget"]["caps"]["max_calls_per_turn"] > 3
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
