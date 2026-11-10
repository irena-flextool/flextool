"""The calibration runner checkpoints every cell before each request and ``run --resume`` continues it from the
request that failed (calibration fixes 2026-11-10, item 216; design specs/review/cycles/eval_checkpoint_design.md
with its critique).

Against scripted backends (no provider is reached): a cell interrupted at any request and resumed sends the
uninterrupted run's requests byte for byte and saves the same record apart from its resume metadata (T2, S01 per
turn); the stops that keep or throw away the checkpoint, and the checks that start a cell again (T3); a process that
died (T4); the day's budget reached inside a cell (T5); skip and unskip (T6); the folder's readers (T7); the
grading material and the redaction (T8); a misclassified record asked again (T9); a pace or stream change (T10);
Google's per-minute input-token quota, kept once and not applicable after the same request was refused twice
(T11, T12).  Every request is counted once against the day.
"""
from __future__ import annotations

import copy
import functools
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.assistant import grounding_projects as gp
from tests.assistant.llm.fake_backend import FakeBackend

PAID = "vendor/model-a"


@pytest.fixture(scope="module")
def grounding(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A grounding project of inputs only (the runner needs its marker, the scripted model no results)."""
    return gp.build_grounding_project(tmp_path_factory.mktemp("grounding"), solve=False).path


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")


@pytest.fixture(autouse=True)
def retry_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session's retries: one retry of a busy or limited answer, without waiting."""
    from flextool.assistant import session
    from flextool.assistant.llm_retry import RetryPolicy

    monkeypatch.setattr(session, "RetryPolicy", functools.partial(
        RetryPolicy, max_retries=1, free_plan_rate_retries=1, backoff_start=0.0, jitter=0.0, connection_wait=0.0))


@pytest.fixture(autouse=True)
def no_floor():
    from flextool.assistant.pacing import PROCESS_PACER

    yield
    for state in PROCESS_PACER._states.values():
        state.floor = None


# --- scripts ------------------------------------------------------------------------------------------------------
def answer(text: str):
    from flextool.assistant.llm_adapter import Message, Reply, Usage

    return Reply(Message("assistant", text), "end", None, Usage(1200, 0, 0, 20))


def tool_round(n: int):
    from flextool.assistant.llm_adapter import Message, Reply, ToolCall, Usage

    return Reply(Message("assistant", "", tool_calls=[ToolCall(f"call{n:03d}", "list_runs", {}, "{}")]), "tools",
                 None, Usage(1500, 0, 0, 10))


def failing(code: str, **kw) -> Callable:
    """A fresh error each time it is reached (a retry changes an error's message)."""
    from flextool.assistant import llm_errors

    cls = {"busy": llm_errors.ServerBusy, "auth": llm_errors.AuthFailed}[code]
    return lambda _req: llm_errors.make_error(cls, code, llm_errors.ErrorContext(provider="OpenRouter"), **kw)


GOOGLE_QUOTA = ("You exceeded your current quota. Quota exceeded for metric: generativelanguage.googleapis.com/"
                "generate_content_free_tier_input_token_count, limit: 15000, model: gemma-4-31b.")


def google(status: int, name: str, message: str, quota: bool = False) -> Callable:
    """Google's error answer as the backend classifies it (no wait before its retry)."""
    from flextool.assistant import llm_errors

    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "0s"}]
    if quota:
        details.insert(0, {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{
            "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_input_token_count",
            "quotaId": "GenerateContentInputTokensPerModelPerMinute-FreeTier",
            "quotaDimensions": {"model": "gemma-4-31b"}, "quotaValue": "15000"}]})
    body = [{"error": {"code": status, "message": message, "status": name, "details": details}}]
    return lambda _req: llm_errors.classify_status(status, body, None, llm_errors.ErrorContext(
        preset_id="gemini", provider="Google Gemini", host="generativelanguage.googleapis.com"))


def interrupt(_req):
    raise KeyboardInterrupt


#: S01's eight turns: turn 2 and turn 5 call a tool (turn 5 twice).
S01_TURNS = [[answer("t1")], [tool_round(1), answer("t2")], [answer("t3")], [answer("t4")],
             [tool_round(2), tool_round(3), answer("t5")], [answer("t6")], [answer("t7")], [answer("t8")]]
S01_SCRIPT = [item for turn in S01_TURNS for item in turn]
S01_TURN_OF = [n for n, turn in enumerate(S01_TURNS, start=1) for _ in turn]
#: A one-turn cell of four requests (standard).
G01_SCRIPT = [tool_round(1), tool_round(2), tool_round(3), answer("coal_plant has 500 MW.")]


class Runs:
    """The backends of every run: each cell's backend takes the next script; every request's body is kept."""

    def __init__(self, scripts: list[list] | None = None, window: int = 131_072) -> None:
        self.scripts = [list(s) for s in scripts or []]
        self.bodies: list[str] = []
        self.window = window

    def prober(self, preset, conn, cancel, *, needed):
        from flextool.assistant.probes import ModelFacts

        return ModelFacts(context_window=self.window, tools=True, source="openrouter")

    def factory(self, preset, conn):
        from flextool.assistant.llm_adapter import make_backend

        script = self.scripts.pop(0) if self.scripts else [answer("plain")] * 8
        backend = FakeBackend(script, origin=(preset.id, conn.base_url, conn.model))
        wire = make_backend(preset, conn)          # renders the request bodies; sends nothing
        stream, close = backend.stream, backend.close

        def recorded(req, on_event, cancel):
            self.bodies.append(json.dumps(wire.request_body(req), sort_keys=True, ensure_ascii=False))
            return stream(req, on_event, cancel)

        def closed() -> None:
            close()
            wire.close()

        backend.stream, backend.close = recorded, closed
        return backend

    def add(self, *scripts: list) -> Runs:
        self.scripts += [list(s) for s in scripts]
        return self


def run(evaluator, grounding: Path, out: Path, runs: Runs, *extra: str, said: list | None = None,
        questions: str = "G01", levels: str = "standard") -> int:
    args = evaluator.build_parser().parse_args(
        ["run", "--project", str(grounding), "--provider", "openrouter", "--model", PAID, "--out", str(out),
         "--accept-data-notice", "--questions", questions, "--levels", levels, *extra])
    return evaluator.cmd_run(args, backend_factory=runs.factory, prober=runs.prober, info=None,
                             say=(said.append if said is not None else lambda _t: None), allow_unsolved=True)


def s01(evaluator, grounding, out, runs, *extra, said=None) -> int:
    return run(evaluator, grounding, out, runs, *extra, said=said, questions="S01", levels="basic,standard,advanced")


def comparable(record: dict) -> dict:
    """A record without what differs between an interrupted and an uninterrupted run: the resume metadata, the
    times, the commit, the requests per day, and each turn's seconds and failed attempts (the stopping attempts)."""
    r = copy.deepcopy(record)
    for k in ("resume", "ran_at", "git", "requests_by_day"):
        r.pop(k)
    for t in r["turns"]:
        t.pop("seconds")
        t.pop("failed_attempts")
    return r


def only(cells: dict) -> tuple[str, dict]:
    (cid, record), = cells.items()
    return cid, record


def spent_of(evaluator, out: Path) -> list[dict]:
    return json.loads((out / evaluator.SPENT_FILE).read_text(encoding="utf-8"))


def partial_of(evaluator, out: Path, cid: str) -> dict:
    return json.loads(evaluator.partial_path(out, cid).read_text(encoding="utf-8"))


def no_partials(evaluator, out: Path) -> bool:
    folder = out / evaluator.PARTIAL_DIR
    return not folder.exists() or not any(folder.iterdir())


def counted(evaluator, out: Path, day: str) -> int:
    """The requests the folder counts for *day* (its records, spent.json and the unfinished checkpoints)."""
    partials = [evaluator.held_requests(p) for p in evaluator.read_partials(out).values()]
    return evaluator.requests_today(evaluator.read_cells(out).values(), day,
                                    spent_of(evaluator, out) if (out / evaluator.SPENT_FILE).is_file() else [],
                                    partials)


def today() -> str:
    from flextool.assistant._files import iso, utc_now

    return iso(utc_now())[:10]


# --- T2: the equivalence, S01 per turn ----------------------------------------------------------------------------
def test_an_interrupted_and_resumed_s01_sends_and_saves_what_an_uninterrupted_run_does(evaluator, grounding,
                                                                                       tmp_path):
    """For every request k of S01 (eight turns, three tool rounds): run A retries request k once; run B stops at k
    (its retries used up: exit 1, the cell kept up to its turn and request), and --resume continues it from that
    request, the finished turns kept, not asked again.  B's requests (the stopped attempt once) are A's, body for
    body; the records are equal apart from the resume metadata; B's resume point names the turn, the request and
    the code; every request is counted once (the record holds all of them, the stop none)."""
    for k in range(len(S01_SCRIPT)):
        a_out, b_out = tmp_path / f"a{k}", tmp_path / f"b{k}"
        a = Runs([S01_SCRIPT[:k] + [failing("busy")] + S01_SCRIPT[k:]])
        assert s01(evaluator, grounding, a_out, a) == 0
        cid, a_record = only(evaluator.read_cells(a_out))
        said: list[str] = []
        b = Runs([S01_SCRIPT[:k] + [failing("busy"), failing("busy")]])
        assert s01(evaluator, grounding, b_out, b, said=said) == 1
        turn = S01_TURN_OF[k]
        assert said[-1] == (f"{cid}: kept up to turn {turn}, request {k + 1}; run again with --resume "
                            "--accept-data-notice to continue it")
        first = len(b.bodies)
        assert s01(evaluator, grounding, b_out, b.add(S01_SCRIPT[k:]), "--resume", said=said) == 0
        assert f"{cid}: continued from turn {turn}, request {k + 1} (busy)" in said
        assert b.bodies[:first - 1] + b.bodies[first:] == a.bodies, f"interrupted at request {k}"
        assert b.bodies[first] == a.bodies[k + 1]           # the resume starts with the request that failed
        b_cid, b_record = only(evaluator.read_cells(b_out))
        assert b_cid == cid and comparable(b_record) == comparable(a_record), f"interrupted at request {k}"
        assert [a["code"] for a in b_record["turns"][turn - 1]["failed_attempts"]] == ["busy", "busy"]
        assert a_record["resume"] == {"count": 0, "points": []}
        (point,) = b_record["resume"]["points"]
        assert b_record["resume"]["count"] == 1
        assert {k_: point[k_] for k_ in ("turn", "request", "code", "pace", "stream")} == {
            "turn": turn, "request": k + 1, "code": "busy", "pace": 0.0, "stream": True}
        assert set(point) == {"turn", "round", "request", "code", "provider", "lost_usage", "stopped_at",
                              "resumed_at", "pace", "stream", "git", "stopped_git", "request_sha256",
                              "body_sha256"}
        assert all(tuple(t) == evaluator.TURN_KEYS for t in b_record["turns"])     # what a checkpoint must hold
        n = len(S01_SCRIPT)
        assert b_record["requests_sent"] == n == sum(b_record["requests_by_day"].values())
        assert [(e["code"], e["requests"]) for e in spent_of(evaluator, b_out)] == [("busy", 0)]
        assert counted(evaluator, b_out, today()) == n == counted(evaluator, a_out, today())
        assert no_partials(evaluator, b_out)


def test_two_stops_give_two_resume_points(evaluator, grounding, tmp_path):
    """T2: S01 stopped in turn 2 and again in turn 5: the record counts two resumes with their turns and codes."""
    out = tmp_path / "out"
    runs = Runs([S01_SCRIPT[:2] + [failing("busy")] * 2])
    assert s01(evaluator, grounding, out, runs) == 1
    assert s01(evaluator, grounding, out, runs.add(S01_SCRIPT[2:6] + [failing("busy")] * 2), "--resume") == 1
    assert s01(evaluator, grounding, out, runs.add(S01_SCRIPT[6:]), "--resume") == 0
    _cid, record = only(evaluator.read_cells(out))
    assert record["resume"]["count"] == 2
    assert [(p["turn"], p["request"], p["code"]) for p in record["resume"]["points"]] == [(2, 3, "busy"),
                                                                                           (5, 7, "busy")]
    assert counted(evaluator, out, today()) == len(S01_SCRIPT) == record["requests_sent"]


# --- T3: what throws a checkpoint away, and what starts a cell again ----------------------------------------------
def test_a_key_error_throws_the_checkpoint_away(evaluator, grounding, tmp_path):
    """auth is not temporary: the checkpoint goes (its two answered requests written to spent.json as
    discarded), the console says nothing was saved, and the next --resume asks the cell from its first request."""
    out = tmp_path / "out"
    said: list[str] = []
    runs = Runs([G01_SCRIPT[:2] + [failing("auth")]])
    assert run(evaluator, grounding, out, runs, said=said) == 1
    assert said[-1] == "G01__medium__r1: nothing was saved for this cell. Run again with --resume --accept-data-notice."
    assert no_partials(evaluator, out)
    assert [(e["code"], e["requests"]) for e in spent_of(evaluator, out)] == [("auth", 0), ("discarded", 2)]
    first = runs.bodies[0]
    sent = len(runs.bodies)
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT), "--resume") == 0
    assert runs.bodies[sent] == first
    assert counted(evaluator, out, today()) == 2 + 4


def _stopped_at_two(evaluator, grounding, out: Path, runs: Runs) -> None:
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[:2] + [failing("busy")] * 2)) == 1
    assert evaluator.partial_path(out, "G01__medium__r1").is_file()


@pytest.mark.parametrize("change", ["window", "instructions", "format", "missing", "tampered"])
def test_a_checkpoint_that_no_longer_fits_starts_the_cell_again(evaluator, grounding, tmp_path, monkeypatch,
                                                                change):
    """A changed window (another setting), changed instructions, a checkpoint of another format, one with a field
    missing (never a default) or one whose history no longer gives its request: the cell starts again from its
    first request, with one line saying why; the checkpoint's requests stay counted (discarded)."""
    from flextool.assistant import prompt

    out = tmp_path / "out"
    runs = Runs()
    _stopped_at_two(evaluator, grounding, out, runs)
    path = evaluator.partial_path(out, "G01__medium__r1")
    partial = json.loads(path.read_text(encoding="utf-8"))
    why = {"window": "the level setting changed: window", "instructions": "the instructions or tools changed",
           "format": "a checkpoint of another format", "missing": "a checkpoint of another format",
           "tampered": "the checkpoint no longer gives the same request"}[change]
    if change == "window":
        runs.window = 200_000
    elif change == "instructions":
        layers = prompt.system_layers
        monkeypatch.setattr(prompt, "system_layers", lambda *a, **k: (*layers(*a, **k), "One more layer."))
    elif change == "format":
        partial["format"] = 0
    elif change == "missing":
        del partial["current"]["notices"]
    else:
        partial["session"]["conversation"]["messages"][0]["text"] += " (changed)"
    evaluator.write_json(path, partial)
    said: list[str] = []
    first = runs.bodies[0]
    sent = len(runs.bodies)
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT), "--resume", said=said) == 0
    line = next(t for t in said if "started again" in t)
    assert line.startswith(f"G01__medium__r1: started again from the start ({why}")
    if change != "instructions":
        assert runs.bodies[sent] == first
    assert [(e["code"], e["requests"]) for e in spent_of(evaluator, out)] == [("busy", 0), ("discarded", 2)]
    assert evaluator.read_cells(out)["G01__medium__r1"]["resume"]["count"] == 0
    assert counted(evaluator, out, today()) == 2 + 4


# --- T4: a process that died ------------------------------------------------------------------------------------
def test_a_process_that_died_is_continued_and_its_request_counted(evaluator, grounding, tmp_path):
    """Ctrl-C (or a crash) while request 3 is under way: no stop is written; --resume continues the cell from
    that request, counts it once on its day (it may have been sent) and records the point as crashed."""
    out = tmp_path / "out"
    runs = Runs([G01_SCRIPT[:2] + [interrupt]])
    with pytest.raises(KeyboardInterrupt):
        run(evaluator, grounding, out, runs)
    partial = partial_of(evaluator, out, "G01__medium__r1")
    assert partial["stopped"] is False and partial["stops"] == []
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[2:]), "--resume") == 0
    record = evaluator.read_cells(out)["G01__medium__r1"]
    assert [(p["code"], p["request"]) for p in record["resume"]["points"]] == [("crashed", 3)]
    assert [(e["code"], e["requests"]) for e in spent_of(evaluator, out)] == [("crashed", 1)]
    assert record["requests_sent"] == 4 and counted(evaluator, out, today()) == 5


# --- T5: the day's budget inside a cell ---------------------------------------------------------------------------
def test_the_days_budget_stops_inside_a_cell_and_the_next_day_continues_it(evaluator, grounding, tmp_path,
                                                                            monkeypatch):
    """--max-requests-per-day 2 and a cell of four requests: two are sent, the third is not (exit 0, the daily
    message); the checkpoint survives (not thrown away as "stopped"), its pending request and its requests per
    day unchanged.  The same day sends nothing more; the next day continues the cell with request 3."""
    reference = Runs([G01_SCRIPT])
    assert run(evaluator, grounding, tmp_path / "ref", reference) == 0
    out = tmp_path / "out"
    said: list[str] = []
    runs = Runs([G01_SCRIPT])
    assert run(evaluator, grounding, out, runs, "--max-requests-per-day", "2", said=said) == 0
    assert said[-1] == evaluator.DAILY_DONE_REMOTE and len(runs.bodies) == 2
    partial = partial_of(evaluator, out, "G01__medium__r1")
    assert partial["requests_by_day"] == {today(): 2} and partial["stopped"] is True
    assert [s["code"] for s in partial["stops"]] == ["budget"]
    assert not (out / evaluator.SPENT_FILE).exists()                 # nothing was sent after the checkpoint
    assert run(evaluator, grounding, out, runs, "--resume", "--max-requests-per-day", "2", said=said) == 0
    assert said[-1] == evaluator.DAILY_DONE_REMOTE and len(runs.bodies) == 2
    first_day = today()
    tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
    monkeypatch.setattr(evaluator, "utc_now", lambda: tomorrow)
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[2:]), "--resume", "--max-requests-per-day", "2") == 0
    assert runs.bodies == reference.bodies
    record = evaluator.read_cells(out)["G01__medium__r1"]
    assert record["requests_by_day"] == {first_day: 2, tomorrow.strftime("%Y-%m-%d"): 2}
    assert [p["code"] for p in record["resume"]["points"]] == ["budget"]
    assert counted(evaluator, out, first_day) == 2 and counted(evaluator, out, tomorrow.strftime("%Y-%m-%d")) == 2


# --- T6: skip and unskip --------------------------------------------------------------------------------------------
def test_skip_keeps_the_checkpoint_and_unskip_continues_it(evaluator, grounding, tmp_path):
    """--skip on a cell with a checkpoint keeps it (its requests in the skip, one attempt: the checkpoint's own
    stops are not counted again); --resume leaves the cell out; after --unskip, --resume continues it from the
    request that failed."""
    out = tmp_path / "out"
    runs = Runs()
    _stopped_at_two(evaluator, grounding, out, runs)
    assert run(evaluator, grounding, out, runs, "--resume", "--skip", "G01__medium__r1", "--skip-reason", "x") == 0
    entry = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))["skipped"]["G01__medium__r1"]
    assert (entry["attempts"], entry["requests"]) == (1, 2)
    assert evaluator.partial_path(out, "G01__medium__r1").is_file()
    sent = len(runs.bodies)
    assert run(evaluator, grounding, out, runs, "--resume") == 0 and len(runs.bodies) == sent
    assert run(evaluator, grounding, out, runs, "--resume", "--unskip", "G01__medium__r1") == 0
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[2:]), "--resume") == 0
    assert runs.bodies[sent] == runs.bodies[2]                        # request 3 again, as it was
    assert evaluator.read_cells(out)["G01__medium__r1"]["resume"]["count"] == 1


# --- T7: the folder's readers -----------------------------------------------------------------------------------
def test_readers_ignore_checkpoints_and_the_grading_names_an_unfinished_cell(evaluator, scoring, grounding,
                                                                             tmp_path):
    """read_cells reads cells/ only (and a temp file in partial/ is no checkpoint); the grading refuses a folder
    with an unfinished cell and names its checkpoint; a stale checkpoint beside a finished record is removed by the
    next run and not counted."""
    out = tmp_path / "out"
    runs = Runs()
    _stopped_at_two(evaluator, grounding, out, runs)
    (out / evaluator.PARTIAL_DIR / "G01__medium__r1.json.123-456.abcd.tmp").write_text("{}", encoding="utf-8")
    assert evaluator.read_cells(out) == {}
    assert list(evaluator.read_partials(out)) == ["G01__medium__r1"]
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    with pytest.raises(evaluator.Refused) as refused:
        evaluator._refuse_incomplete([(out, run_json, {})], scoring.load_set())
    assert "G01__medium__r1 (unfinished: checkpoint at turn 1, request 3)" in str(refused.value)
    stale = evaluator.partial_path(out, "G01__medium__r1").read_bytes()
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[2:]), "--resume") == 0
    evaluator.partial_path(out, "G01__medium__r1").write_bytes(stale)       # a stop between record and removal
    assert counted(evaluator, out, today()) == 4 + 2                         # before: the stale one counts
    sent = len(runs.bodies)
    assert run(evaluator, grounding, out, runs, "--resume") == 0
    assert len(runs.bodies) == sent and not evaluator.partial_path(out, "G01__medium__r1").exists()
    assert counted(evaluator, out, today()) == 4
    evaluator._refuse_incomplete([(out, run_json, evaluator.read_cells(out))], scoring.load_set())


# --- T8: what the graders see, and the redaction ----------------------------------------------------------------
def test_a_resumed_records_grading_material_and_line(evaluator, scoring, grounding, tmp_path):
    """The grading material of a resumed S01 record is the material without its resume metadata, the leak scan
    (with the resume times among its stamps) finds nothing, and the results.jsonl line counts the resume."""
    out = tmp_path / "out"
    runs = Runs([S01_SCRIPT[:4] + [failing("busy")] * 2])
    assert s01(evaluator, grounding, out, runs) == 1
    assert s01(evaluator, grounding, out, runs.add(S01_SCRIPT[4:]), "--resume") == 0
    cid, record = only(evaluator.read_cells(out))
    bare = {k: v for k, v in record.items() if k not in ("resume", "requests_by_day")}
    question = scoring.load_set().questions["S01"]
    key = scoring.load_key()
    entry = key.questions["S01"]
    items, _ = scoring.grading_items(question, entry, record["cell"]["level_setting"], evaluator._turn_views(record),
                                     max_calls=record["budget"]["caps"]["max_calls_per_turn"])
    tag = scoring.data_tag(7, "#1")
    material = scoring.grader_material(record, question, entry, items, [], tag)
    assert material == scoring.grader_material(bare, question, entry, items, [], tag)
    stamps = evaluator._stamps([record])
    point = record["resume"]["points"][0]
    assert {point["stopped_at"], point["resumed_at"]} <= stamps
    assert scoring.leak_scan(material, names=[], stamps=stamps, sources=evaluator._tool_text(record)) == []
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    line = evaluator.summary_line(record, run_json, key_version=key.key_version)
    assert (line["resumes"], line["resume_points"], line["resume_codes"]) == (1, [[4, 0]], ["busy"])


EXTERNAL = Path("/elsewhere/case data/outside.sqlite")


@pytest.fixture
def source_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """One tool, ``source``: it registers :data:`EXTERNAL` with the redaction (as listing an external input source
    does) and names it."""
    from flextool.assistant import registry
    from flextool.assistant.registry import make_tool

    def source(ctx, _args: dict) -> dict:
        ctx.redactor.add_known(EXTERNAL)
        return {"path": str(EXTERNAL)}

    monkeypatch.setattr(registry, "_tools", {"source": make_tool(
        name="source", title="source", description="The source tool.", properties={}, handler=source,
        area="orientation", small_tier=True)})


def test_the_redaction_learned_before_a_stop_holds_after_the_resume(evaluator, grounding, tmp_path, source_tool):
    """M4: a path the redaction learned from a tool call before the stop (an external source) is still redacted
    after the resume, in the record of the turn being asked: the folder is "<source folder>", never its name."""
    from flextool.assistant.llm_adapter import Message, Reply, ToolCall, Usage

    said_path = f"The file is in {EXTERNAL.parent}."
    script = [Reply(Message("assistant", "", tool_calls=[ToolCall("c1", "source", {}, "{}")]), "tools", None,
                    Usage(1500, 0, 0, 10)), answer(said_path)]
    a_out, b_out = tmp_path / "a", tmp_path / "b"
    assert run(evaluator, grounding, a_out, Runs([script])) == 0
    runs = Runs([script[:1] + [failing("busy")] * 2])
    assert run(evaluator, grounding, b_out, runs) == 1
    assert run(evaluator, grounding, b_out, runs.add(script[1:]), "--resume") == 0
    a = evaluator.read_cells(a_out)["G01__medium__r1"]
    b = evaluator.read_cells(b_out)["G01__medium__r1"]
    assert b["turns"][0]["answer"] == a["turns"][0]["answer"] == "The file is in <source folder>."
    assert "case data" not in json.dumps(b)
    assert comparable(b) == comparable(a)


# --- T9, T10: a misclassified record; pace and stream ---------------------------------------------------------------
def test_a_misclassified_record_whose_rerun_stops_is_continued(evaluator, grounding, tmp_path):
    """A saved cell a provider failure left as cut off is asked again; that rerun stops: the old record stays and
    the checkpoint is continued by the next --resume, whose record replaces it (its requests stay counted)."""
    from flextool.assistant.llm_errors import MESSAGES

    out = tmp_path / "out"
    runs = Runs([[answer("x")]])
    assert run(evaluator, grounding, out, runs) == 0
    path = evaluator.cell_path(out, "G01__medium__r1")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["turns"][0].update(stop="length", answer="", usage=[{"output": 0}],
                              error={"code": "length", "message": MESSAGES["length_no_finish"]})
    evaluator.write_json(path, record)
    old = path.read_bytes()
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[:2] + [failing("busy")] * 2), "--resume") == 1
    assert path.read_bytes() == old and evaluator.partial_path(out, "G01__medium__r1").is_file()
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[2:]), "--resume") == 0
    new = evaluator.read_cells(out)["G01__medium__r1"]
    assert new["resume"]["count"] == 1 and new["turns"][0]["answer"] == "coal_plant has 500 MW."
    assert [(e["code"], e["requests"]) for e in spent_of(evaluator, out)] == [("busy", 0), ("replaced", 1)]
    assert counted(evaluator, out, today()) == 1 + 4


def test_pace_and_stream_changed_on_resume_are_recorded(evaluator, grounding, tmp_path):
    """T10: a resume with another --pace and --no-stream is not refused; its point records them, and the record
    the run that finished the cell."""
    out = tmp_path / "out"
    runs = Runs()
    _stopped_at_two(evaluator, grounding, out, runs)
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT[2:]), "--resume", "--pace", "0.01",
               "--no-stream") == 0
    record = evaluator.read_cells(out)["G01__medium__r1"]
    (point,) = record["resume"]["points"]
    assert (point["pace"], point["stream"]) == (0.01, False)
    assert (record["model"]["pace"], record["model"]["stream"]) == (0.01, False)
    assert point["request_sha256"] and point["body_sha256"]


# --- T11, T12: Google's per-minute input-token quota --------------------------------------------------------------
def test_googles_input_token_quota_stops_with_its_own_code_and_is_kept(evaluator, grounding, tmp_path):
    """T11: a 429 naming Google's input-token per-minute quota (retried once) stops the cell with ``tpm_input``,
    no graded stop: the cell is kept (exit 1), and its spent.json stop names the request (its hash and corrected
    estimate) and the quota."""
    out = tmp_path / "out"
    said: list[str] = []
    quota = google(429, "RESOURCE_EXHAUSTED", GOOGLE_QUOTA, quota=True)
    runs = Runs([G01_SCRIPT[:2] + [quota, quota]])
    assert run(evaluator, grounding, out, runs, said=said) == 1
    assert said[-1].startswith("G01__medium__r1: kept up to turn 1, request 3")
    (stop,) = spent_of(evaluator, out)
    assert stop["code"] == "tpm_input" and stop["provider"]["quota"][0]["limit"] == "15000"
    partial = partial_of(evaluator, out, "G01__medium__r1")
    assert stop["request_sha256"] == partial["session"]["pending"]["request_sha256"]
    assert stop["est_input"] == partial["session"]["pending"]["est_input"] > 0


def test_the_same_request_refused_twice_for_its_input_tokens_is_not_applicable(evaluator, scoring, grounding,
                                                                                tmp_path):
    """T12: the same request refused again for Google's input tokens per minute: the cell is not applicable
    ("provider quota per request", with the quota), its finished turns kept and every request counted, the
    checkpoint gone; the run goes on (exit 0).  No later --resume asks it (no third send); the grading takes it as
    not applicable.  --unskip takes it back and the next --resume asks it from the start."""
    out = tmp_path / "out"
    said: list[str] = []
    quota = google(429, "RESOURCE_EXHAUSTED", GOOGLE_QUOTA, quota=True)
    runs = Runs([G01_SCRIPT[:2] + [quota, quota]])
    assert run(evaluator, grounding, out, runs) == 1
    assert run(evaluator, grounding, out, runs.add([quota, quota]), "--resume", said=said) == 0
    assert any("not applicable (provider quota per request)" in t for t in said)
    record = evaluator.read_cells(out)["G01__medium__r1"]
    assert record["not_applicable"] == evaluator.QUOTA_PER_REQUEST and record[evaluator.NA_SCOPE_KEY] == "cell"
    assert record["quota"][0]["quota_id"] == "GenerateContentInputTokensPerModelPerMinute-FreeTier"
    assert [s["code"] for s in record["stops"]] == ["tpm_input", "tpm_input"]
    assert record["requests_by_day"] == {today(): 2} and record["requests_sent"] == 2
    assert no_partials(evaluator, out)
    sent = len(runs.bodies)
    assert run(evaluator, grounding, out, runs, "--resume", said=said) == 0
    assert len(runs.bodies) == sent
    assert said[-2] == ("G01__medium__r1: not applicable (provider quota per request); run --unskip "
                        "G01__medium__r1 to ask it again")
    run_json = json.loads((out / evaluator.RUN_FILE).read_text(encoding="utf-8"))
    cells = evaluator.read_cells(out)
    evaluator._refuse_incomplete([(out, run_json, cells)], scoring.load_set())     # complete: not applicable
    assert evaluator.summary_line(record, run_json, key_version=1)["outcome"] == "not_applicable"
    assert counted(evaluator, out, today()) == 2
    assert run(evaluator, grounding, out, runs, "--resume", "--unskip", "G01__medium__r1") == 0
    assert evaluator.read_cells(out) == {} and counted(evaluator, out, today()) == 2
    assert run(evaluator, grounding, out, runs.add(G01_SCRIPT), "--resume") == 0
    assert runs.bodies[sent] == runs.bodies[0]
    assert counted(evaluator, out, today()) == 2 + 4


def test_the_same_request_stopped_three_times_is_named(evaluator, grounding, tmp_path):
    """Busy three times at the same request: still kept (no automatic not applicable), and the console says so."""
    out = tmp_path / "out"
    said: list[str] = []
    runs = Runs()
    _stopped_at_two(evaluator, grounding, out, runs)
    for _ in range(2):
        assert run(evaluator, grounding, out, runs.add([failing("busy")] * 2), "--resume", said=said) == 1
    assert "G01__medium__r1: the same request failed 3 times; --skip it or wait" in said
    assert evaluator.partial_path(out, "G01__medium__r1").is_file()


# --- the minor items ----------------------------------------------------------------------------------------------
def test_a_stop_is_written_into_the_checkpoint_before_spent(evaluator, grounding, tmp_path, monkeypatch):
    """m3: a stop goes into the checkpoint first, then spent.json: a crash between them counts its requests twice,
    never not at all."""
    out = tmp_path / "out"
    write = evaluator.write_json

    def failing_spent(path, data):
        if Path(path).name == evaluator.SPENT_FILE:
            raise OSError("disk full")
        write(path, data)

    monkeypatch.setattr(evaluator, "write_json", failing_spent)
    with pytest.raises(OSError):
        run(evaluator, grounding, out, Runs([G01_SCRIPT[:2] + [failing("busy")] * 2]))
    partial = partial_of(evaluator, out, "G01__medium__r1")
    assert partial["stopped"] is True and [s["code"] for s in partial["stops"]] == ["busy"]


def test_the_earlier_runs_attempts_are_said_and_kept(evaluator, grounding, tmp_path):
    """m4: after a resume, the stopped turn's earlier attempts (kept as JSON in the checkpoint) lead its failed
    attempts and the console's "earlier attempt" lines."""
    out = tmp_path / "out"
    first = google(503, "UNAVAILABLE", "First overload.")
    second = google(503, "UNAVAILABLE", "Second overload.")
    runs = Runs([G01_SCRIPT[:2] + [first, first]])
    assert run(evaluator, grounding, out, runs) == 1
    said: list[str] = []
    assert run(evaluator, grounding, out, runs.add([second, second]), "--resume", said=said) == 1
    assert "  earlier attempts (2): 503 UNAVAILABLE — First overload." in said
    current = partial_of(evaluator, out, "G01__medium__r1")["current"]
    assert [a["message"] for a in current["failed_attempts"]] == ["First overload."] * 2 + ["Second overload."] * 2


def test_a_checkpoint_that_cannot_be_written_is_said_once(evaluator, grounding, tmp_path, monkeypatch):
    """A checkpoint write that fails (the disk) is said once; the cell goes on without one and its record counts
    every request."""
    from flextool.assistant import _files

    out = tmp_path / "out"
    said: list[str] = []
    write = _files.write_text_atomic

    def failing_partial(path, text, **kw):
        if Path(path).parent.name == evaluator.PARTIAL_DIR:
            raise OSError("read-only folder")
        write(path, text, **kw)

    monkeypatch.setattr(_files, "write_text_atomic", failing_partial)
    assert run(evaluator, grounding, out, Runs([G01_SCRIPT]), said=said) == 0
    assert [t for t in said if "checkpoint could not be written" in t] == [
        "G01__medium__r1: the checkpoint could not be written (read-only folder); the cell goes on without one."]
    record = evaluator.read_cells(out)["G01__medium__r1"]
    assert record["requests_by_day"] == {today(): 4} and no_partials(evaluator, out)
