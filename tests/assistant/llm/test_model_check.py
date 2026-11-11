"""[Test model] / ``--test-model`` (owner decisions 2026-11-08): one tiny request through the backend, the retries
and the pacing; what it reports; and the "not available to your account" marks of the saved model list.

Expected texts are written from the owner's decisions, not read from the module."""
from __future__ import annotations

import itertools
import threading
import time

import pytest

from flextool.assistant import model_list_cache as cache
from flextool.assistant.llm_adapter import Message, Reply, Usage
from flextool.assistant.llm_errors import ErrorContext, RateLimited, TimedOut, classify_status, make_error
from flextool.assistant.model_check import check_model
from flextool.assistant.pacing import Pacer
from flextool.assistant.providers import PRESETS
from flextool.assistant.usage import Ledger
from tests.assistant.llm.fake_backend import Block, FakeBackend

ORIGIN = ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-pro")
CTX = ErrorContext(preset_id="gemini", provider="Google Gemini", model="gemini-2.5-pro")


def ok(text: str = "OK") -> Reply:
    return Reply(Message("assistant", text), "end", None, Usage(12, 0, 0, 1))


def ticking(*times: float):  # noqa: ANN201 — a clock giving *times*, then the last one
    values = itertools.chain(times, itertools.repeat(times[-1]))
    return lambda: next(values)


def run(backend: FakeBackend, user_config_in_tmp, **kwargs):  # noqa: ANN001, ANN003, ANN201
    kwargs.setdefault("clock", ticking(0.0))
    return check_model(PRESETS["gemini"], backend, kwargs.pop("cancel", threading.Event()), label="Google Gemini",
                       ledger=Ledger(user_config_in_tmp), ledger_key="gemini", pacer=Pacer(),
                       sleep=lambda _s: False, **kwargs)


def _counts(user_config_in_tmp) -> dict:  # noqa: ANN001
    months = Ledger(user_config_in_tmp).read()["months"]
    (month,) = months.values()
    return month["gemini"]["models"]["gemini-2.5-pro"]


def test_an_answer_says_available_with_its_times_and_counts_its_tokens(user_config_in_tmp):
    backend = FakeBackend([ok()], origin=ORIGIN)
    # The clock: the request starts at 10, the first text at 12.5, the end at 13.0.
    result = run(backend, user_config_in_tmp, clock=ticking(10.0, 12.5, 13.0))
    assert result.outcome == "available"
    assert (result.first_byte, result.total) == (2.5, 3.0)
    assert result.message == "gemini-2.5-pro is available: the answer started after 2.5 s and ended after 3.0 s."
    request = backend.requests[0]
    assert request.tools == () and request.max_output_tokens <= 32
    assert all("OK" in part or "test" in part for part in request.system)       # nothing of a project
    assert _counts(user_config_in_tmp)["output"] == 1


def test_an_answer_without_text_says_only_the_time(user_config_in_tmp):
    backend = FakeBackend([Reply(None, "length", None, Usage(12, 0, 0, 16))], origin=ORIGIN)
    result = run(backend, user_config_in_tmp, clock=ticking(0.0, 4.0))
    assert result.outcome == "available" and result.message == "gemini-2.5-pro is available: it answered after 4.0 s."


def test_a_model_closed_to_the_account_is_refused_in_the_providers_words(user_config_in_tmp):
    body = [{"error": {"code": 404, "status": "NOT_FOUND",
                       "message": "This model models/gemini-2.5-pro is no longer available to new users."}}]
    backend = FakeBackend([classify_status(404, body, None, CTX)], origin=ORIGIN)
    result = run(backend, user_config_in_tmp)
    assert result.outcome == "refused" and result.unavailable
    assert result.message.startswith("Google Gemini refused the test request: This model models/gemini-2.5-pro is "
                                     "no longer available to new users")
    assert "not available to your account" in result.message


def test_another_refusal_is_quoted_and_marks_nothing(user_config_in_tmp):
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Thinking budget is invalid."}}
    backend = FakeBackend([classify_status(400, body, None, CTX)], origin=ORIGIN)
    result = run(backend, user_config_in_tmp)
    assert result.outcome == "refused" and not result.unavailable
    assert result.message == "Google Gemini refused the test request: Thinking budget is invalid."


def test_no_answer_in_time_is_a_timeout(user_config_in_tmp):
    backend = FakeBackend([make_error(TimedOut, "timeout", CTX)], origin=ORIGIN)
    result = run(backend, user_config_in_tmp)
    assert result.outcome == "timeout"
    assert result.message == "Google Gemini did not answer within 300 seconds."


def test_a_rate_limit_is_retried_and_counted(user_config_in_tmp):
    limit = make_error(RateLimited, "rpm", CTX, retry_after=0.01)
    backend = FakeBackend([limit, ok()], origin=ORIGIN)
    result = run(backend, user_config_in_tmp, clock=time.monotonic)        # the countdown runs on a real clock
    assert result.outcome == "available" and len(backend.requests) == 2
    assert _counts(user_config_in_tmp)["rejected_attempts"] == 1


def test_stop_ends_the_wait(user_config_in_tmp):
    block = Block(then=ok())
    backend = FakeBackend([block], origin=ORIGIN)
    cancel = threading.Event()
    box = {}
    thread = threading.Thread(target=lambda: box.update(result=run(backend, user_config_in_tmp, cancel=cancel)))
    thread.start()
    assert block.started.wait(5)
    cancel.set()
    thread.join(5)
    assert box["result"].outcome == "stopped" and box["result"].message == "Stopped."


# --- the marks ------------------------------------------------------------------------------------------------
URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
FP = "0123456789ab"          # the fingerprint of the key of the tests below


def test_a_mark_hides_the_model_until_a_refresh_a_new_key_success_or_seven_days(tmp_path):
    from flextool.assistant.providers import hidden_reason, shown_models

    ids = ["gemini-2.5-pro", "gemini-3.8-flash"]
    cache.remember("gemini", URL, ids, now=100.0, root=tmp_path)
    cache.mark_unavailable("gemini", URL, "gemini-2.5-pro", FP, now=1000.0, root=tmp_path)
    marks = cache.load_marks(tmp_path)
    closed = cache.unavailable(marks, "gemini", URL, FP, now=1000.0 + 6 * 86400)
    assert closed == {"gemini-2.5-pro"}
    assert shown_models(PRESETS["gemini"], ids, unavailable=closed) == ["gemini-3.8-flash"]
    assert shown_models(PRESETS["gemini"], ids, show_all=True, unavailable=closed) == ids
    assert hidden_reason("gemini-2.5-pro", closed) == "unavailable"
    assert cache.unavailable(marks, "gemini", URL, FP, now=1000.0 + 8 * 86400) == set()        # 7 days
    assert cache.unavailable(marks, "openrouter", URL, FP, now=1000.0) == set()                # another list
    assert cache.lookup(cache.load_all(tmp_path), "gemini", URL).ids == tuple(ids)        # the list is kept
    # A success of the model removes its mark; a new key every mark of the provider and address; a refresh too.
    cache.mark_unavailable("gemini", URL, "gemini-3.8-flash", FP, now=1000.0, root=tmp_path)
    cache.clear_marks("gemini", URL, "gemini-2.5-pro", root=tmp_path)
    assert set(cache.load_marks(tmp_path)[cache.key("gemini", URL)]) == {"gemini-3.8-flash"}
    cache.clear_marks("gemini", URL, root=tmp_path)
    assert cache.load_marks(tmp_path) == {}
    cache.mark_unavailable("gemini", URL, "gemini-2.5-pro", FP, now=1000.0, root=tmp_path)
    cache.remember("gemini", URL, ids, now=2000.0, root=tmp_path)
    assert cache.load_marks(tmp_path) == {}


@pytest.mark.parametrize("damaged", ["{not json", '{"format": 1, "lists": [], "unavailable": 3}'])
def test_a_damaged_file_loses_only_the_marks(tmp_path, damaged):
    (tmp_path / cache.FILE).write_text(damaged, encoding="utf-8")
    assert cache.load_marks(tmp_path) == {}
    cache.mark_unavailable("gemini", URL, "gemini-2.5-pro", FP, now=1.0, root=tmp_path)
    assert cache.unavailable(cache.load_marks(tmp_path), "gemini", URL, FP, now=2.0) == {"gemini-2.5-pro"}


KEY = "sk-test-0123456789abcdefghij"


def test_a_mark_counts_only_for_the_key_it_was_found_with(tmp_path, user_config_in_tmp):
    """A mark keeps a one-way fingerprint of the key: another key (or none) does not see it, the same key does,
    and marking again drops the marks of another key and those of an older FlexTool without a fingerprint."""
    import json

    from flextool.assistant.keys import NO_KEY, SALT_FILE, key_fingerprint

    mine, other = key_fingerprint(KEY), key_fingerprint(KEY + "x")
    assert mine == key_fingerprint(KEY) and len(mine) == 12 and mine != other
    assert key_fingerprint(None) == key_fingerprint("") == NO_KEY
    assert (user_config_in_tmp / SALT_FILE).exists()
    cache.mark_unavailable("gemini", URL, "gemini-2.5-pro", mine, now=1000.0, root=tmp_path)
    marks = cache.load_marks(tmp_path)
    assert cache.unavailable(marks, "gemini", URL, mine, now=1001.0) == {"gemini-2.5-pro"}
    assert cache.unavailable(marks, "gemini", URL, other, now=1001.0) == set()
    assert cache.unavailable(marks, "gemini", URL, NO_KEY, now=1001.0) == set()
    assert cache.unavailable(marks, "gemini", URL, None, now=1001.0) == set()      # the key not known yet
    # Neither the key nor any 6 characters of it are in the file.
    text = (tmp_path / cache.FILE).read_text(encoding="utf-8")
    assert all(KEY[i:i + 6] not in text for i in range(len(KEY) - 5))
    # A mark of an older FlexTool (a bare time) never matches; marking with another key drops it and the old key's.
    data = json.loads(text)
    data["unavailable"][cache.key("gemini", URL)]["gemini-1.5-pro"] = 1000.0
    (tmp_path / cache.FILE).write_text(json.dumps(data), encoding="utf-8")
    assert cache.unavailable(cache.load_marks(tmp_path), "gemini", URL, mine, now=1001.0) == {"gemini-2.5-pro"}
    cache.mark_unavailable("gemini", URL, "gemini-3.8-flash", other, now=1002.0, root=tmp_path)
    assert set(cache.load_marks(tmp_path)[cache.key("gemini", URL)]) == {"gemini-3.8-flash"}


def test_the_salt_is_kept_per_installation(user_config_in_tmp):
    """The fingerprint is salted: the same key gives another fingerprint with another salt file."""
    from flextool.assistant.keys import SALT_FILE, key_fingerprint

    first = key_fingerprint(KEY)
    salt = (user_config_in_tmp / SALT_FILE).read_text(encoding="utf-8")
    assert KEY not in salt and key_fingerprint(KEY) == first
    (user_config_in_tmp / SALT_FILE).unlink()
    assert key_fingerprint(KEY) != first
