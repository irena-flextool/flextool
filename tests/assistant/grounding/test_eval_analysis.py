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


#: A gpt-oss:20b variant on a 16 GB card the desktop uses 4 GB of: 1.3 GB of the 14 GB stays on the CPU.
OFFLOADED = {"size": 14_000_000_000, "size_vram": 12_700_000_000}


@pytest.fixture
def ollama_server():
    """A fake Ollama on 127.0.0.1: ``/api/show`` gives each variant its ``num_ctx``, ``/api/ps`` reports it
    loaded with :data:`OFFLOADED`'s sizes (``size_vram`` below ``size``)."""
    from tests.assistant.llm.local_server import FakeServer

    server = FakeServer()
    contexts = {"gpt-oss-20b-16k": 16_384, "gpt-oss-20b-32k": 32_768}
    server.routes[("POST", "/api/show")] = lambda body: (200, {"parameters": f"num_ctx {contexts[body['model']]}"})
    server.routes[("GET", "/api/ps")] = (200, {"models": [{"name": name, "model": name, **OFFLOADED}
                                                          for name in contexts]})
    yield server
    server.close()


def _offload_run(evaluator, grounding: Path, out: Path, server, harness: Harness, *extra: str,
                 loaded: int = 16_384, level: str = "basic=16384") -> int:
    from flextool.assistant.probes import ModelFacts

    def prober(preset, conn, cancel, *, needed):
        return ModelFacts(context_window=131_072, loaded_context=loaded, tools=True, source="ollama",
                          parameter_count=20_900_000_000)

    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "ollama", "--model", "gpt-oss:20b", "--out", str(out),
         "--base-url", f"http://127.0.0.1:{server.port}/v1", "--questions", "G01",
         "--levels", level.split("=")[0], "--ollama-context", level, *extra])
    return evaluator.cmd_run(args, backend_factory=harness.factory, prober=prober, info=None,
                             say=lambda _t: None, allow_unsolved=True)


def test_cpu_offload_not_applicable_without_the_flag(evaluator, grounding: Path, tmp_path: Path,
                                                     ollama_server) -> None:
    harness = Harness()
    out = tmp_path / "out"
    assert _offload_run(evaluator, grounding, out, ollama_server, harness) == 0
    record = json.loads((out / "cells" / "G01__small__r1.json").read_text(encoding="utf-8"))
    assert record["not_applicable"] == "context" and record["turns"] == [] and harness.requests == 0
    assert record["model"]["ollama"]["size_vram"] == 12_700_000_000      # read from the fake /api/ps
    assert record["model"]["cpu_offload"] is None                       # not run: nothing ran offloaded
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    assert run_json["cpu_offload_allowed"] is False
    assert ("GET", "/api/ps") in [(m, p) for m, p, _b, _h in ollama_server.requests]


def test_cpu_offload_runs_and_is_recorded_with_the_flag(evaluator, grounding: Path, tmp_path: Path,
                                                        ollama_server) -> None:
    harness = Harness()
    out = tmp_path / "out"
    assert _offload_run(evaluator, grounding, out, ollama_server, harness, "--allow-cpu-offload") == 0
    record = json.loads((out / "cells" / "G01__small__r1.json").read_text(encoding="utf-8"))
    assert record["not_applicable"] is None and record["turns"] and harness.requests == 1
    # 1.3 GB of 14 GB on the CPU: 1.3 / 14 = 0.092857... -> 0.0929
    offload = {"size": 14_000_000_000, "size_vram": 12_700_000_000, "cpu_share": 0.0929}
    assert record["model"]["cpu_offload"] == offload
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    assert run_json["cpu_offload_allowed"] is True
    line = evaluator.summary_line(record, run_json, key_version=None)
    assert line["cpu_offload"] == offload


def test_cpu_offload_smaller_context_stays_not_applicable(evaluator, grounding: Path, tmp_path: Path,
                                                          ollama_server) -> None:
    harness = Harness()
    out = tmp_path / "out"
    assert _offload_run(evaluator, grounding, out, ollama_server, harness, "--allow-cpu-offload",
                        loaded=24_576, level="standard=32768") == 0
    record = json.loads((out / "cells" / "G01__medium__r1.json").read_text(encoding="utf-8"))
    assert record["not_applicable"] == "context" and harness.requests == 0
    assert record["model"]["ollama"]["loaded_context"] == 24_576 and record["model"]["cpu_offload"] is None


def test_cpu_offload_is_part_of_the_run_identity(evaluator, grounding: Path, tmp_path: Path, ollama_server,
                                                 capsys) -> None:
    harness = Harness()
    out = tmp_path / "out"
    assert _offload_run(evaluator, grounding, out, ollama_server, harness) == 0          # not applicable
    assert _offload_run(evaluator, grounding, out, ollama_server, harness, "--resume",
                        "--allow-cpu-offload") == 2
    assert evaluator.OTHER_RESULTS in capsys.readouterr().err and harness.requests == 0
    allowed = tmp_path / "allowed"
    assert _offload_run(evaluator, grounding, allowed, ollama_server, harness, "--allow-cpu-offload") == 0
    assert _offload_run(evaluator, grounding, allowed, ollama_server, harness, "--resume") == 2
    assert evaluator.OTHER_RESULTS in capsys.readouterr().err
    assert _offload_run(evaluator, grounding, allowed, ollama_server, harness, "--resume",
                        "--allow-cpu-offload") == 0
    assert harness.requests == 1                                    # the finished cell is not asked again
    # a run.json from before the flag existed is a run without it
    run_file = out / evaluator.RUN_FILE
    saved = json.loads(run_file.read_text(encoding="utf-8"))
    run_file.write_text(json.dumps({k: v for k, v in saved.items() if k != "cpu_offload_allowed"}),
                        encoding="utf-8")
    assert _offload_run(evaluator, grounding, out, ollama_server, harness, "--resume") == 0
    assert _offload_run(evaluator, grounding, out, ollama_server, harness, "--resume",
                        "--allow-cpu-offload") == 2


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
    base = {"line_format": 2, "rubric_version": 2, "date": "2026-11-10", "grader": "owner", "authoritative": True,
            "set_version": 3, "key_version": 1, "provider": "ollama", "model": "m-basic", "automatic_level": "small",
            "level_setting": "small", "window": 16384, "free_tier": False, "knobs_relaxed": False, "class": "lookup",
            "outcome": "answered", "missing": [], "critical_missing": [], "invented_numbers": 0, "calls": 1,
            "rounds": 2, "tokens_in": 1000, "tokens_out": 100, "tokens_not_reported": 0,
            "prompt_ratio": {"text": 0.8, "tool_results": 0.9}, "repeat": 1, "level_source": "reported_size",
            "cap_hits": 0, "length_stops": 0, "fit_compactions": 0, "truncations": 0, "refused_tools": []}
    return {**base, **fields}


def test_calibrate_report(analysis, evaluator, tmp_path: Path) -> None:
    lookups = ["G02", "G03", "G04", "G05", "G07"]
    lines = [_line(question="G01", outcome="failed", calls=2),
             _line(question="G01", outcome="answered", calls=2),                          # the later line wins
             *[_line(question=q, calls=c) for q, c in zip(lookups, [3, 3, 1, 2, 3])],
             _line(question="G10", outcome="failed", invented_numbers=1),
             _line(question="G11", outcome="failed"),
             _line(question="G10", outcome="answered", grader="claude-opus-5-5", authoritative=False),
             _line(question="G06", class_="x", outcome="escalated"),
             _line(question="G09", outcome="escalated"),
             _line(question="G08", outcome="escalated"),
             _line(question="G12", outcome="failed"),
             _line(question="G01", level_setting="medium", outcome="not_applicable", window=None),
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


# --- G14: blinding (C6-m2; the grading itself: test_grading_v2.py) -------------------------------------------
#: Not a free-plan model, so its requests are not paced.
PAID = "vendor/model-a"


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


# --- G14: task support and statistics (C6-m6, n1, m2) ------------------------------------------------------
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
    cells = [analysis.ReviewedCell(q, q, (pair,), ("answered", "answered"), (1, 1))
             for q, pair in (("q1", ("right", "right")), ("q2", ("wrong", "wrong")))]
    assert analysis.measure(cells, "item", seed=3)["bootstrap"].undefined == boot.undefined


def test_failed_attempts_are_counted_by_model_and_code_not_graded(analysis, evaluator) -> None:
    """Two records of one model hold 2 + 1 unreadable tool calls and 1 overload, a record of another model 1
    connection error: counted per model and code (alphabetical), and the line lists them for one model."""
    def record(model: str, *turns: list[str]) -> dict:
        return {"model": {"preset": "ollama", "model": model}, "cell": {"model": model},
                "turns": [{"failed_attempts": [{"code": c} for c in codes]} for codes in turns]}

    records = [record("gpt-oss:20b", ["tool_call_unreadable", "tool_call_unreadable"], []),
               record("gpt-oss:20b", ["busy", "tool_call_unreadable"]), record("qwen3:14b", ["connection"]),
               record("qwen3:14b")]
    counts = analysis.failed_attempt_codes(records)
    assert counts == {("ollama", "gpt-oss:20b"): {"busy": 1, "tool_call_unreadable": 3},
                      ("ollama", "qwen3:14b"): {"connection": 1}}
    assert analysis.failed_attempts_line({k: v for k, v in counts.items() if k[1] == "gpt-oss:20b"}) == (
        "  failed requests (service or format problems, not part of the grade): busy 1, tool_call_unreadable 3")
    assert analysis.failed_attempts_line(analysis.failed_attempt_codes([record("m", [])])) is None
    assert "tool_call_unreadable" in evaluator.TEMPORARY_STOPS
    assert "tool_call_unreadable" not in evaluator.GRADED_STOPS


def test_an_older_records_unreadable_tool_call_is_counted_as_one(analysis) -> None:
    """A record written before ``tool_call_unreadable`` existed holds Ollama's "error parsing tool call" as
    ``busy`` with status 500: counted as ``tool_call_unreadable``; a real overload (another message, or another
    status with that message) stays ``busy``."""
    message = ("error parsing tool call: raw='{\"agg\":\"sum\",\"max_rows=10?\"}', "
               "err=invalid character '}' after object key")
    attempts = [{"code": "busy", "status": 500, "provider_code": "api_error", "message": message, "quota": [],
                 "retry_delay": None},
                {"code": "busy", "status": 503, "provider_code": None, "message": "server overloaded", "quota": [],
                 "retry_delay": None},
                {"code": "busy", "status": 429, "provider_code": None, "message": message, "quota": [],
                 "retry_delay": None}]
    record = {"model": {"preset": "ollama", "model": "gpt-oss:20b"}, "cell": {"model": "gpt-oss:20b"},
              "turns": [{"failed_attempts": attempts}]}
    assert analysis.failed_attempt_codes([record]) == {("ollama", "gpt-oss:20b"): {"busy": 2,
                                                                                   "tool_call_unreadable": 1}}


def test_statistics_by_hand(analysis) -> None:
    """Wilson 13/20: z² = 3.8415, centre (0.65 + 0.0960) / 1.1921 = 0.6258, half 1.6442 · 0.11737 = 0.1930.
    Kappa of (p,p), (f,f), (p,unparsed), (f,f): observed 3/4 = 0.75; expected 0.5·0.25 (pass) + 0.5·0.5 (fail)
    + 0·0.25 (unparsed) = 0.375; kappa (0.75 − 0.375) / 0.625 = 0.6."""
    assert analysis.wilson(13, 20) == pytest.approx((0.4329, 0.8188), abs=5e-5)
    pairs = [("pass", "pass"), ("fail", "fail"), ("pass", "unparsed"), ("fail", "fail")]
    assert analysis.kappa(pairs) == pytest.approx((0.75, 0.375, 0.6), abs=1e-12)


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
           "routing": llm_errors.NoMatchingEndpoint, "tool_call_unreadable": llm_errors.UnreadableToolCall}[code]
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
    """C6-M5: an error that is not the model's behaviour stops the run with the cell unsaved; --resume asks it.
    Item 216: a temporary stop (connection, timeout, busy) keeps the cell's checkpoint, which --resume continues;
    any other (key, model, routing) throws it away."""
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
    kept = code in ("connection", "timeout", "busy")
    assert said[-1] == ("G03__small__r1: kept up to turn 1, request 1; run again with --resume --accept-data-notice "
                        "to continue it" if kept else
                        "G03__small__r1: nothing was saved for this cell. Run again with --resume "
                        "--accept-data-notice.")
    assert evaluator.partial_path(out, "G03__small__r1").is_file() == kept
    assert go("--resume") == 0                                    # the fixed backend: plain answers
    assert not (out / evaluator.PARTIAL_DIR).exists() or not any((out / evaluator.PARTIAL_DIR).iterdir())
    cells = evaluator.read_cells(out)
    assert sorted(cells) == ["G01__small__r1", "G03__small__r1", "G04__small__r1"]
    assert all(t["stop"] == "end" for r in cells.values() for t in r["turns"])


def test_an_unreadable_tool_call_keeps_the_cell_and_the_run_goes_on(evaluator, grounding: Path, tmp_path: Path,
                                                                    no_waits) -> None:
    """A tool call the service could not read (after the session's retries) keeps G03's checkpoint but does not end
    the run: G04 is asked and saved; the run ends with exit code 1 and names the kept cell; --resume continues it
    (exit 0, every cell saved)."""
    out = tmp_path / "out"
    said: list[str] = []
    harness = Scripted([[_answer()], [_error("tool_call_unreadable")] * 4])

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
             "--accept-data-notice", "--questions", "G01,G03,G04", "--levels", "basic", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                                 say=said.append, allow_unsolved=True)

    assert go() == 1
    assert sorted(p.stem for p in (out / "cells").glob("*.json")) == ["G01__small__r1", "G04__small__r1"]
    assert evaluator.partial_path(out, "G03__small__r1").is_file()
    assert ("G03__small__r1: kept up to turn 1, request 1; run again with --resume --accept-data-notice to "
            "continue it") in said
    assert "G04__small__r1: saved" in said
    assert said[-2:] == [f"The other cells of {PAID} are in {out}.",
                         "1 cell(s) kept after a tool call the service could not read: G03__small__r1; run again "
                         "with --resume --accept-data-notice to continue them."]
    said.clear()
    assert go("--resume") == 0
    assert said[-1] == f"All cells of {PAID} are in {out}."
    cells = evaluator.read_cells(out)
    assert sorted(cells) == ["G01__small__r1", "G03__small__r1", "G04__small__r1"]
    assert not any((out / evaluator.PARTIAL_DIR).iterdir())
    # every finished record holds its level setting, the one its checkpoint was checked against
    keys = [r["setting_key"] for r in cells.values()]
    assert keys[0]["level"] == "small" and len(keys[0]["tools_hash"]) == 16 and len(keys[0]["system_hash"]) == 16
    assert all(k == keys[0] for k in keys)


def test_run_plans_its_matrix_and_a_stopped_folder_is_incomplete(evaluator, scoring, grounding: Path,
                                                                 tmp_path: Path, no_waits) -> None:
    """Grading v2 B1: run.json records the questions, levels and repeats a run plans (and whether the
    automatic-level question is asked at each level reached) before any cell is asked, so a folder a stop left
    short names its missing cells: G01 saved, G03 stopped (its checkpoint named), G04 never asked → G03 and G04
    missing."""
    out = tmp_path / "out"
    harness = Scripted([[_answer()], [_error("connection")] * 4])
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
         "--accept-data-notice", "--questions", "G01,G03,G04", "--levels", "basic"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=lambda _t: None, allow_unsolved=True) == 1
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    matrix = run_json[evaluator.MATRIX_KEY]
    assert (matrix["questions"], matrix["levels"], matrix["repeat"]) == (["G01", "G03", "G04"], ["small"], 1)
    assert set(matrix["automatic_at"]) == {"small"}
    cells = evaluator.read_cells(out)
    assert sorted(cells) == ["G01__small__r1"]
    with pytest.raises(evaluator.Refused) as refused:
        evaluator._refuse_incomplete([(out, run_json, cells)], scoring.load_set())
    text = str(refused.value)
    assert text.startswith("2 cell(s) are missing or not finished")
    assert ("out: G03__small__r1 (unfinished: checkpoint at turn 1, request 1), G04__small__r1 (missing)"
            in text)
    assert f"run --resume --project PROJECT --provider openrouter --model {PAID} --out {out}" in text


def test_a_cell_asked_again_stays_until_replaced(evaluator, grounding: Path, tmp_path: Path, no_waits) -> None:
    """Grading v2 B1: a saved cell asked again is replaced only by its new record; a stop on the way leaves the old
    record (the folder never lacks the cell), its requests counted once; the rerun replaces it."""
    out = tmp_path / "out"
    harness = Harness()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=harness,
               model=PAID) == 0
    path = evaluator.cell_path(out, "G01__small__r1")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["turns"][0].update(stop="connection", error={"code": "connection", "message": "scripted"})
    evaluator.write_json(path, record)
    old = path.read_bytes()
    stopping = Scripted([[_error("connection")] * 4])
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01", "--levels", "basic", harness=stopping,
               model=PAID) == 1
    assert path.read_bytes() == old
    spent = json.loads((out / evaluator.SPENT_FILE).read_text(encoding="utf-8"))
    assert [(e["cell"], e["code"]) for e in spent] == [("G01__small__r1", "connection")]
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01", "--levels", "basic", harness=harness,
               model=PAID) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["turns"][0]["stop"] == "end"
    spent = json.loads((out / evaluator.SPENT_FILE).read_text(encoding="utf-8"))
    assert [(e["cell"], e["code"]) for e in spent] == [("G01__small__r1", "connection"),
                                                       ("G01__small__r1", "replaced")]


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


def test_a_provider_failure_saved_as_cut_off_is_asked_again(evaluator, grounding: Path, tmp_path: Path) -> None:
    """Calibration 2026-11-10: a turn saved with "the connection ended before the answer was complete", no answer
    text and no output tokens in its last request (0, or no usage at all) is a provider failure: --resume asks
    the cell again and says how many.  A cut-off answer with text and output is a graded stop and kept."""
    from flextool.assistant.llm_errors import MESSAGES

    out = tmp_path / "out"
    harness = Harness()
    qids = "G01,G03,G04,G05"
    assert run(evaluator, grounding, out, "--questions", qids, "--levels", "basic", harness=harness, model=PAID) == 0
    ended = {"code": "length", "message": MESSAGES["length_no_finish"]}
    zero = {"input_uncached": 1437, "cache_read": 0, "cache_write": None, "output": 0, "cost": None,
            "reported": True}
    edits = {"G01__small__r1": {"answer": "", "usage": [zero]},          # thought, then the break
             "G03__small__r1": {"answer": "", "usage": []},              # no usage at all
             "G04__small__r1": {"answer": "In scenario coal the energy", "usage": [{**zero, "output": 14}]},
             "G05__small__r1": {"answer": "", "usage": [{**zero, "output": 30}]}}   # a tool call came
    for cid, edit in edits.items():
        path = evaluator.cell_path(out, cid)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["turns"][0].update(stop="length", error=ended, **edit)
        evaluator.write_json(path, record)
    kept = {cid: evaluator.cell_path(out, cid).read_bytes() for cid in ("G04__small__r1", "G05__small__r1")}
    said: list[str] = []
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
         "--accept-data-notice", "--questions", qids, "--levels", "basic", "--resume"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=said.append, allow_unsolved=True) == 0
    assert harness.requests == 4 + 2
    assert ("2 saved cell(s) are a provider failure saved as cut off (the answer broke off before any of it "
            "came); they are asked again: G01__small__r1, G03__small__r1.") in said
    assert "G01__small__r1: asked again (a provider failure saved as cut off)" in said
    cells = evaluator.read_cells(out)
    assert [cells[c]["turns"][0]["stop"] for c in ("G01__small__r1", "G03__small__r1")] == ["end", "end"]
    assert {cid: evaluator.cell_path(out, cid).read_bytes() for cid in kept} == kept


def test_spend_of_a_failed_cell_counts(evaluator, grounding: Path, tmp_path: Path, no_waits) -> None:
    """C6-M5: the requests a stopped cell sent count against the day: 1 (the tool call) + 1 (the resumed cell)
    reaches the budget of 2.  Item 216: the answered tool call is held by the cell's checkpoint, not by the stop's
    spent.json entry (each request counted once), and --resume continues the cell with the one request left."""
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
    assert [(e["cell"], e["requests"], e["code"]) for e in spent] == [("G01__small__r1", 0, "connection")]
    partial = json.loads(evaluator.partial_path(out, "G01__small__r1").read_text(encoding="utf-8"))
    assert list(partial["requests_by_day"].values()) == [1]
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

    na = _line(question="G01", outcome="not_applicable", key_version=None)
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


def test_reuse(evaluator) -> None:
    """C6-n4: the time helpers are the store's own (the graders' folding: test_grading_v2.test_reuse_folds_results)."""
    from flextool.assistant import _files

    assert evaluator.iso is _files.iso and evaluator.utc_now is _files.utc_now


# --- run --skip / --unskip (the owner leaves a cell that never completes out) -----------------------------------
def test_skip_then_unskip_and_resume_asks_the_cell(evaluator, grounding: Path, tmp_path: Path, no_waits) -> None:
    """G03 stops (1 attempt in spent.json); --skip records it and asks nothing (no probe, no request); --resume
    leaves it out; --unskip takes it back and the next --resume asks and saves it."""
    out = tmp_path / "out"
    harness = Scripted([[_answer()], [_error("connection")] * 4])
    said: list[str] = []

    def go(*extra: str) -> int:
        args = evaluator.build_parser().parse_args(
            ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
             "--questions", "G01,G03", "--levels", "basic", *extra])
        return evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                                 say=said.append, allow_unsolved=True)

    assert go("--accept-data-notice") == 1
    probes, sent = harness.probes, harness.requests
    assert go("--resume", "--skip", "G03__small__r1", "--skip-reason", "the endpoint breaks off") == 0
    assert (harness.probes, harness.requests) == (probes, sent)
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    assert run_json["skipped"]["G03__small__r1"]["attempts"] == 1
    assert go("--resume", "--accept-data-notice") == 0                     # resume is not refused
    assert harness.requests == sent and "G03__small__r1" not in evaluator.read_cells(out)
    assert said[-1] == f"All cells of {PAID} are in {out} (1 skipped by the owner)."
    assert go("--resume", "--unskip", "G03__small__r1") == 0
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    assert run_json["skipped"] == {} and [e["cell"] for e in run_json["unskipped"]] == ["G03__small__r1"]
    assert go("--resume", "--accept-data-notice") == 0
    assert harness.requests == sent + 1
    assert evaluator.read_cells(out)["G03__small__r1"]["turns"][0]["stop"] == "end"


def test_skip_of_an_unknown_cell_is_refused(evaluator, grounding: Path, tmp_path: Path, capsys) -> None:
    """A cell id the folder cannot have is refused with the closest ids; nothing is written."""
    out = tmp_path / "out"
    harness = Harness()
    assert run(evaluator, grounding, out, "--questions", "G01,G03", "--levels", "basic", harness=harness,
               model=PAID) == 0
    before = (out / evaluator.RUN_FILE).read_bytes()
    assert run(evaluator, grounding, out, "--resume", "--skip", "G03__smal__r1", "--skip-reason", "x",
               harness=harness, model=PAID) == 2
    assert "No cell G03__smal__r1 in out (closest: G03__small__r1" in capsys.readouterr().err
    assert (out / evaluator.RUN_FILE).read_bytes() == before


# --- calibration fixes 2026-11-10: the provider's own words, --pace, --no-stream --------------------------------
GOOGLE_QUOTA = ("You exceeded your current quota. Quota exceeded for metric: generativelanguage.googleapis.com/"
                "generate_content_free_tier_input_token_count, limit: 15000, model: gemma-4-31b.")
GOOGLE_BUSY = "The model is overloaded. Please try again later."


def _google(status: int, name: str, message: str, headers: dict | None = None):
    """The error Google's endpoint answers, classified as the backend does (no wait before its retry)."""
    from flextool.assistant import llm_errors

    body = [{"error": {"code": status, "message": message, "status": name,
                       "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"}]}}]
    return llm_errors.classify_status(status, body, headers, llm_errors.ErrorContext(
        preset_id="gemini", provider="Google Gemini", host="generativelanguage.googleapis.com", secret=SECRET_KEY))


SECRET_KEY = "AIzaSy-test-key-0123456789"


def test_a_stop_keeps_what_the_provider_said(evaluator, grounding: Path, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Calibration 2026-11-10: a cell stopped by Google's errors keeps, for every failed attempt, the HTTP status,
    Google's status name and its message in spent.json, and the console shows them after FlexTool's message.  A
    429 (retried once), then a 503 that stops the cell.  Gemini quota fixes 2026-11-10: the quota Google names
    (metric, limit, model) and the wait it suggests lead the message, and spent.json keeps them."""
    import functools

    from flextool.assistant import session
    from flextool.assistant.llm_retry import RetryPolicy

    monkeypatch.setattr(session, "RetryPolicy", functools.partial(RetryPolicy, max_retries=1))
    out = tmp_path / "out"
    said: list[str] = []
    harness = Scripted([[_google(429, "RESOURCE_EXHAUSTED", GOOGLE_QUOTA + f" key {SECRET_KEY}"),
                         _google(503, "UNAVAILABLE", GOOGLE_BUSY, {"retry-after": "0"})]])
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
         "--accept-data-notice", "--questions", "G01", "--levels", "basic"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=said.append, allow_unsolved=True) == 1
    quota = ("quota: input tokens (limit 15000, model gemma-4-31b); Google suggests waiting 0 s. You exceeded your "
             "current quota. key <key>")
    assert said[-4:-1] == ["Google Gemini is still overloaded. Ask again in a moment.",
                           f"Google Gemini said: 503 UNAVAILABLE — {GOOGLE_BUSY}",
                           f"  earlier attempt (1): 429 RESOURCE_EXHAUSTED — {quota}"]
    spent = json.loads((out / evaluator.SPENT_FILE).read_text(encoding="utf-8"))
    busy = {"code": "busy", "status": 503, "provider_code": "UNAVAILABLE", "message": GOOGLE_BUSY, "quota": [],
            "retry_delay": None}
    metric = "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count"
    assert [{k: e[k] for k in ("cell", "requests", "code", "provider", "attempts")} for e in spent] == [
        {"cell": "G01__small__r1", "requests": 0, "code": "busy", "provider": busy,
         "attempts": [{"code": "tpm_input", "status": 429, "provider_code": "RESOURCE_EXHAUSTED", "message": quota,
                       "quota": [{"metric": metric, "quota_id": "", "limit": "15000", "model": "gemma-4-31b"}],
                       "retry_delay": 0.0},
                      busy]}]
    assert SECRET_KEY not in (out / evaluator.SPENT_FILE).read_text(encoding="utf-8")


def test_a_saved_turn_keeps_its_failed_attempts(evaluator, grounding: Path, tmp_path: Path, no_waits) -> None:
    """A connection error retried once, then the answer: the saved turn names the failed attempt (no provider
    answer, so no status or message)."""
    out = tmp_path / "out"
    harness = Scripted([[_error("connection"), _answer()]])
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", harness=harness,
               model=PAID) == 0
    turn = evaluator.read_cells(out)["G01__small__r1"]["turns"][0]
    assert turn["stop"] == "end"
    assert turn["failed_attempts"] == [{"code": "connection", "status": None, "provider_code": None,
                                        "message": None, "quota": [], "retry_delay": None}]


class Timed(Harness):
    """Plain answers; the time of each request and the connection of each backend."""

    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []
        self.conns: list = []

    def factory(self, preset, conn):
        import time

        backend = super().factory(preset, conn)
        self.conns.append(conn)
        stream = backend.stream

        def timed(req, on_event, cancel):
            self.times.append(time.monotonic())
            return stream(req, on_event, cancel)

        backend.stream = timed
        return backend


@pytest.fixture
def no_floor():
    from flextool.assistant.pacing import PROCESS_PACER

    yield
    for state in PROCESS_PACER._states.values():
        state.floor = None


def test_pace_spaces_the_requests_and_may_change_on_resume(evaluator, grounding: Path, tmp_path: Path,
                                                           no_floor) -> None:
    """``run --pace 0.6``: no two requests of the run start closer than 0.6 s (the pacing alone would send them
    back to back).  The pace is recorded in run.json (each run) and in each cell; a --resume with another pace is
    not refused."""
    out = tmp_path / "out"
    harness = Timed()
    assert run(evaluator, grounding, out, "--questions", "G01,G03,G04", "--levels", "basic", "--pace", "0.6",
               harness=harness, model=PAID) == 0
    gaps = [b - a for a, b in zip(harness.times, harness.times[1:])]
    assert len(gaps) == 2 and min(gaps) >= 0.6 - 0.01
    cells = evaluator.read_cells(out)
    assert {c["model"]["pace"] for c in cells.values()} == {0.6}
    assert {c["model"]["stream"] for c in cells.values()} == {True}
    assert run(evaluator, grounding, out, "--resume", "--questions", "G01,G05", "--levels", "basic",
               harness=harness, model=PAID) == 0
    assert evaluator.read_cells(out)["G05__small__r1"]["model"]["pace"] == 0.0
    runs = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))[evaluator.RUNS_KEY]
    assert [(r["pace"], r["stream"]) for r in runs] == [(0.6, True), (0.0, True)]


def test_pace_is_a_floor_over_the_pacing():
    """The floor lengthens the spacing of request starts over the learned interval and the seed and leaves the
    learned interval as it is."""
    from flextool.assistant.pacing import Pacer

    p = Pacer(clock=lambda: 0.0)
    p.prepare("k", 1.1)
    p.set_floor("k", 20.0)
    assert p.interval("k") == 20.0
    assert p.rate_limited("k", None) == pytest.approx(2.2)          # learned from the seed, not the floor
    assert p.interval("k") == 20.0
    p.set_floor("k", 0)
    assert p.interval("k") == pytest.approx(2.2)


def test_no_stream_asks_for_whole_answers(evaluator, grounding: Path, tmp_path: Path, capsys) -> None:
    """``run --no-stream``: every connection asks for whole answers, recorded per cell and run; a --resume may
    stream again.  Refused for a provider whose backend has no whole answers (nothing is written)."""
    out = tmp_path / "out"
    harness = Timed()
    assert run(evaluator, grounding, out, "--questions", "G01", "--levels", "basic", "--no-stream",
               harness=harness, model=PAID) == 0
    assert harness.conns and not any(c.stream for c in harness.conns)
    assert evaluator.read_cells(out)["G01__small__r1"]["model"]["stream"] is False
    assert run(evaluator, grounding, out, "--resume", "--questions", "G03", "--levels", "basic",
               harness=harness, model=PAID) == 0
    assert harness.conns[-1].stream is True
    runs = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))[evaluator.RUNS_KEY]
    assert [r["stream"] for r in runs] == [False, True]
    capsys.readouterr()
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "anthropic", "--model", "claude-x", "--out",
         str(tmp_path / "other"), "--accept-data-notice", "--no-stream"])
    assert evaluator.cmd_run(args, backend_factory=harness.factory, prober=harness.prober, info=None,
                             say=lambda _t: None, allow_unsolved=True) == 2
    assert "--no-stream is for the OpenAI-compatible providers" in capsys.readouterr().err
    assert not (tmp_path / "other").exists()


def test_cells_of_two_tool_behaviours_are_warned_of(analysis) -> None:
    """B1 EVAL: one m1 small cell asked with behaviour b1, one before it was recorded (unknown); the same tools and
    instructions.  One warning, for the behaviour."""
    def record(behaviour: str | None) -> dict:
        key = {"tools_hash": "t1", "system_hash": "s1", **({"behaviour": behaviour} if behaviour else {})}
        return {"cell": {"model": "m1", "level_setting": "small"}, "model": {"preset": "ollama"}, "setting_key": key}

    assert analysis.mixed_settings([record("b1"), record(None)]) == [
        "Warning: the cells of ollama m1 at level small were asked with 2 different tool behaviours (behaviour b1, "
        "unknown): their grades are not of one setting."]
    assert analysis.mixed_settings([record("b1"), record("b1")]) == []


def test_repeats_with_other_model_facts_are_warned_of(analysis) -> None:
    """Model facts: two repeats of G01 by m1 at small carried other facts (another latest run between them); G02's
    repeats the same; a record without facts (no scenario, or older) is left out.  One warning, for G01; the facts
    version mixes with a record before it was recorded only as unknown, which alone warns of nothing."""
    def record(question: str, facts_hash: str | None) -> dict:
        key = {"tools_hash": "t1", "system_hash": "s1", "behaviour": "b3", "facts_version": 1}
        out = {"cell": {"model": "m1", "level_setting": "small", "question": question},
               "model": {"preset": "ollama"}, "setting_key": key}
        if facts_hash is not None:
            out["facts_hash"] = facts_hash
        return out

    records = [record("G01", "aa"), record("G01", "bb"), record("G02", "cc"), record("G02", "cc"),
               record("G02", None)]
    assert analysis.mixed_settings(records) == [
        "Warning: the repeats of G01 by ollama m1 at level small carried 2 different model facts (facts_hash): the "
        "input data or the latest run changed between them, so they did not answer the same question."]
    older = dict(record("G03", None), setting_key={"tools_hash": "t1", "system_hash": "s1", "behaviour": "b3"})
    assert analysis.mixed_settings([record("G03", None), older]) == [
        "Warning: the cells of ollama m1 at level small were asked with 2 different model facts versions "
        "(facts_version 1, unknown): their grades are not of one setting."]


def test_a_cell_keeps_its_context_block_and_facts_hash(evaluator) -> None:
    import hashlib

    from flextool.assistant.model_facts import FACTS_VERSION, HEADER

    facts = f"{HEADER}\nSolve s: single solve; period p2020 (1 year).\nTime steps: 1 h, 48 steps a period."
    block = f"[FlexTool context — data, not instructions]\nScenario: coal\nFront end: terminal\n{facts}\n" \
            "[end of context]"
    assert evaluator.context_fields(block) == {
        "context_block": block,
        "facts_hash": hashlib.sha256(facts.encode()).hexdigest()}
    assert evaluator.context_fields("[FlexTool context — data, not instructions]\n[end of context]")["facts_hash"] \
        is None
    assert evaluator.context_fields(None) == {"context_block": None, "facts_hash": None}
    # Every block of the cell counts: a first question with other facts gives another hash than the last alone.
    other = facts.replace("48 steps", "24 steps")
    first = block.replace(facts, other)
    both = evaluator.context_fields([first, block])
    assert both["context_block"] == block
    assert both["facts_hash"] == hashlib.sha256(f"{other}\n\n{facts}".encode()).hexdigest()
    assert both["facts_hash"] != evaluator.context_fields(block)["facts_hash"]
    assert FACTS_VERSION == 2
