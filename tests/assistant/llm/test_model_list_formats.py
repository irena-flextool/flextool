"""The live model list of each preset, from synthetic payloads in the provider's documented list format, and one
model-id rule for the list, the saved setting and the request (owner feedback 2026-11-08).

Formats (checked against the provider documentation on 2026-11-08 unless marked):

* Gemini (OpenAI-compatible ``GET …/v1beta/openai/models``): OpenAI-style items whose id carries ``models/``
  (``{"id": "models/gemini-1.5-flash", "object": "model", "owned_by": "google"}``, Google AI developer forum);
  the chat endpoint is documented with the bare name (``gemini-2.5-flash``).
* OpenRouter (``GET /api/v1/models``): ``id`` such as ``openai/gpt-4``, ``context_length``,
  ``supported_parameters``; the id is the chat model.
* Mistral (``GET /v1/models``): ``id``, ``object``, ``capabilities``, ``max_context_length``, ``aliases``, ``type``.
* Anthropic (``GET /v1/models``): ``id``, ``type``, ``display_name``, ``created_at``, ``lifecycle`` (active,
  deprecated: closed to new adopters; retired: requests fail), ``max_input_tokens``; ``has_more``, ``first_id``,
  ``last_id``.
* OpenAI (``GET /v1/models``): ``{"object": "list", "data": [{"id", "object", "created", "owned_by"}]}`` —
  NOT VERIFIED this session (the reference page refused the fetch); the long-standing format.
* Ollama (``GET /v1/models``): OpenAI-style items, ids with their tag (``llama3.2:latest``), ``owned_by``
  "library" — fields partly verified (the page names ``created`` and ``owned_by`` and shows tagged ids).
* LM Studio (``GET /v1/models``): OpenAI-style items — NOT VERIFIED (the page shows no response example).
"""
from __future__ import annotations

import threading

import pytest

from flextool.assistant.backends.anthropic_messages import AnthropicMessagesBackend
from flextool.assistant.backends.openai_chat import OpenAIChatBackend
from flextool.assistant.llm_adapter import Message
from flextool.assistant.llm_errors import ModelNotFound, message_for
from flextool.assistant.providers import PRESETS, model_id
from tests.assistant.llm.wire import Recorder, connection, json_response, openai_sse, request, sse_response

LISTS: dict[str, tuple[dict, list[str]]] = {
    "gemini": ({"object": "list", "data": [
        {"id": "models/gemini-2.5-flash", "object": "model", "owned_by": "google"},
        {"id": "models/gemini-2.5-pro", "object": "model", "owned_by": "google"}]},
        ["gemini-2.5-flash", "gemini-2.5-pro"]),
    "openrouter": ({"data": [
        {"id": "openai/gpt-4", "canonical_slug": "openai/gpt-4", "name": "OpenAI: GPT-4", "context_length": 8191,
         "supported_parameters": ["tools", "tool_choice", "max_tokens"]},
        {"id": "example/no-tools", "canonical_slug": "example/no-tools", "name": "No tools", "context_length": 4096,
         "supported_parameters": ["max_tokens"]}]},
        ["openai/gpt-4"]),
    "mistral": ({"object": "list", "data": [
        {"id": "mistral-large-latest", "object": "model", "created": 1, "owned_by": "mistralai",
         "capabilities": {"completion_chat": True, "function_calling": True}, "max_context_length": 131072,
         "aliases": ["mistral-large-2411"], "type": "base"}]},
        ["mistral-large-latest"]),
    "openai": ({"object": "list", "data": [
        {"id": "gpt-4.1", "object": "model", "created": 1, "owned_by": "system"},
        {"id": "text-embedding-3-small", "object": "model", "created": 1, "owned_by": "system"}]},
        ["gpt-4.1"]),
    "ollama": ({"object": "list", "data": [
        {"id": "llama3.2:latest", "object": "model", "created": 1, "owned_by": "library"},
        {"id": "gpt-oss:120b-cloud", "object": "model", "created": 1, "owned_by": "library"}]},
        ["gpt-oss:120b-cloud", "llama3.2:latest"]),
    "lmstudio": ({"object": "list", "data": [
        {"id": "qwen2.5-7b-instruct", "object": "model", "owned_by": "organization_owner"}]},
        ["qwen2.5-7b-instruct"]),
}


@pytest.mark.parametrize("preset_id", sorted(LISTS))
def test_the_openai_style_lists_give_the_ids_that_are_sent(preset_id):
    payload, expected = LISTS[preset_id]
    chosen = expected[0]
    wire = Recorder([json_response(200, payload), sse_response(openai_sse([
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}]))])
    backend = OpenAIChatBackend(PRESETS[preset_id], connection(preset_id, model=model_id(preset_id, chosen)),
                                transport=wire.transport)
    try:
        listed = backend.list_models()
        assert [e.id for e in listed] == expected
        assert wire.requests[0].url.rstrip("/").endswith("/models")
        backend.stream(request([Message("user", "hi")]), lambda _e: None, threading.Event())
    finally:
        backend.close()
    # The id picked from the list is the id saved and the id sent.
    assert model_id(preset_id, chosen) == chosen
    assert wire.last.body["model"] == chosen


def test_gemini_ids_with_or_without_the_prefix_are_one_model():
    assert model_id("gemini", "models/gemini-2.5-flash") == "gemini-2.5-flash"
    assert model_id("gemini", " gemini-2.5-flash ") == "gemini-2.5-flash"
    # Other presets keep the id as listed: a slash is part of OpenRouter's ids.
    assert model_id("openrouter", "openai/gpt-4") == "openai/gpt-4"
    assert model_id("openai_compatible", "models/x") == "models/x"


def test_ollama_cloud_models_are_noted():
    payload, _expected = LISTS["ollama"]
    wire = Recorder([json_response(200, payload)])
    backend = OpenAIChatBackend(PRESETS["ollama"], connection("ollama"), transport=wire.transport)
    try:
        notes = {e.id: e.note for e in backend.list_models()}
    finally:
        backend.close()
    assert notes == {"gpt-oss:120b-cloud": "runs on Ollama cloud", "llama3.2:latest": None}


def test_anthropic_leaves_out_retired_models_and_notes_deprecated_ones():
    payload = {"data": [
        {"id": "claude-new", "type": "model", "display_name": "New", "created_at": "2026-09-21T00:00:00Z",
         "lifecycle": "active", "max_input_tokens": 200000, "max_tokens": 64000},
        {"id": "claude-old", "type": "model", "display_name": "Old", "created_at": "2025-01-01T00:00:00Z",
         "lifecycle": "deprecated", "max_input_tokens": 200000, "max_tokens": 8192},
        {"id": "claude-gone", "type": "model", "display_name": "Gone", "created_at": "2024-01-01T00:00:00Z",
         "lifecycle": "retired", "max_input_tokens": 0, "max_tokens": 0}],
        "first_id": "claude-new", "has_more": False, "last_id": "claude-gone"}
    wire = Recorder([json_response(200, payload)])
    backend = AnthropicMessagesBackend(PRESETS["anthropic"], connection("anthropic"), transport=wire.transport)
    try:
        listed = {e.id: (e.note, e.context_window) for e in backend.list_models()}
    finally:
        backend.close()
    assert listed == {"claude-new": (None, 200000), "claude-old": ("deprecated: closed to new accounts", 200000)}


def test_a_listed_model_the_provider_does_not_serve_quotes_its_answer():
    """Gemini answers 404 for a model it lists but does not serve to the account: the message says so in the
    provider's words instead of claiming the model does not exist."""
    answer = [{"error": {"code": 404, "status": "NOT_FOUND",
                         "message": "This model models/gemini-2.5-flash is no longer available to new users."}}]
    wire = Recorder([json_response(404, answer)])
    backend = OpenAIChatBackend(PRESETS["gemini"], connection("gemini", model="gemini-2.5-flash"),
                                transport=wire.transport)
    try:
        with pytest.raises(ModelNotFound) as caught:
            backend.stream(request([Message("user", "hi")]), lambda _e: None, threading.Event())
    finally:
        backend.close()
    gui = message_for(caught.value, "gui")
    assert gui == ("Google Gemini did not find the model gemini-2.5-flash (its answer: This model "
                   "models/gemini-2.5-flash is no longer available to new users). A listed model can still be "
                   "closed to your account: pick another one from the list in the assistant settings.")


# --- the lists the settings keep -----------------------------------------------------------------------------------
def test_saved_lists_are_kept_per_preset_and_address(tmp_path):
    from flextool.assistant import model_list_cache as cache

    assert cache.load_all(tmp_path) == {}
    cache.remember("ollama", "http://LocalHost:11434/v1/", ["a:latest"], now=100.0, root=tmp_path)
    cache.remember("ollama", "http://otherhost:11434/v1", ["b:latest"], now=200.0, root=tmp_path)
    lists = cache.load_all(tmp_path)
    assert cache.lookup(lists, "ollama", "http://localhost:11434/v1") == cache.SavedList(("a:latest",), 100.0)
    assert cache.lookup(lists, "ollama", "http://otherhost:11434/v1").ids == ("b:latest",)
    assert cache.lookup(lists, "lmstudio", "http://localhost:11434/v1") is None
    # A damaged file loses the saved lists only, and the next fetch writes it again.
    (tmp_path / cache.FILE).write_text("{not json", encoding="utf-8")
    assert cache.load_all(tmp_path) == {}
    cache.remember("ollama", "http://localhost:11434/v1", ["c:latest"], now=300.0, root=tmp_path)
    assert list(cache.load_all(tmp_path)) == ["ollama|http://localhost:11434/v1"]


# --- the newest stable version of each model -----------------------------------------------------------------------
# Synthetic ids in each service's id format; the expected survivors are worked out by hand from the rules in
# flextool/assistant/model_versions.py, not from its output.
FAMILY_LISTS: dict[str, tuple[list[str], list[str]]] = {
    "gemini": ([
        "models/gemini-2.0-flash-001", "models/gemini-2.0-flash-exp", "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite", "models/gemini-2.5-flash-preview-09-2025", "models/gemini-2.5-pro",
        "models/gemini-2.5-pro-preview-05-06", "models/gemini-3-flash", "models/gemini-3-flash-lite",
        "models/gemini-3-pro-preview", "models/gemini-3.1-flash-preview", "models/gemini-3.1-pro-preview",
        "models/gemini-exp-1206", "models/gemini-flash-latest", "models/gemma-3-27b-it"],
        # flash: 3 is the newest release (the 3.1 preview, the exp and the -latest alias are hidden); pro: 2.5,
        # its 3 and 3.1 previews hidden; flash-lite: 3; gemini-exp-1206 is a family of previews only.
        ["gemini-2.5-pro", "gemini-3-flash", "gemini-3-flash-lite", "gemma-3-27b-it"]),
    "anthropic": ([
        "claude-3-5-haiku-20241022", "claude-3-5-sonnet-latest", "claude-3-7-sonnet-20250219", "claude-haiku-4-5",
        "claude-haiku-4-5-20251001", "claude-opus-4-1", "claude-opus-4-1-20250805", "claude-opus-4-5",
        "claude-opus-4-5-20251101", "claude-opus-5-20260201", "claude-opus-5-20260301", "claude-sonnet-4-20250514",
        "claude-sonnet-4-5", "claude-sonnet-4-5-20250929", "claude-sonnet-5-20260115"],
        # haiku 4-5: the undated name before its snapshot; opus 5: the newer of two snapshots; the -latest alias
        # is hidden.
        ["claude-haiku-4-5", "claude-opus-5-20260301", "claude-sonnet-5-20260115"]),
    "openai": ([
        "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini",
        "gpt-4o-search-preview", "gpt-5", "gpt-5-2025-08-07", "gpt-5-mini", "gpt-5-nano", "gpt-5.1",
        "gpt-5.1-2025-11-13", "gpt-5.1-codex", "gpt-5.2-preview", "o1-preview", "o3", "o4-mini"],
        # gpt 4.1 < 5 < 5.1 (the undated name before its snapshot; the 5.2 preview hidden); mini and nano are
        # their own families; "4o" is a name, not a version: gpt-4o keeps its undated name over its snapshot;
        # gpt-4o-search and o1 have previews only.
        ["gpt-4o", "gpt-4o-mini", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.1-codex", "o3", "o4-mini"]),
    "mistral": ([
        "codestral-2501", "codestral-2508", "magistral-medium-2509", "ministral-8b-2410", "ministral-8b-latest",
        "mistral-large-2407", "mistral-large-2411", "mistral-large-latest", "mistral-small-2503",
        "mistral-small-latest", "pixtral-large-latest"],
        # -latest is hidden, so the dated snapshots compare normally; pixtral-large has only an alias.
        ["codestral-2508", "magistral-medium-2509", "ministral-8b-2410", "mistral-large-2411",
         "mistral-small-2503"]),
    "openrouter": ([
        "anthropic/claude-opus-4.1", "anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4.5",
        "deepseek/deepseek-chat-v3-0324:free", "deepseek/deepseek-chat-v3.1:free", "google/gemini-2.0-flash-exp:free",
        "google/gemini-2.5-flash", "google/gemini-3-flash-preview", "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-oss-120b",
        "openai/gpt-oss-120b:free", "qwen/qwen3-coder"],
        # The vendor path and :free are part of the family; google/gemini-flash:free has only an exp.
        ["anthropic/claude-opus-4.1", "anthropic/claude-sonnet-4.5", "deepseek/deepseek-chat-v3.1:free",
         "google/gemini-2.5-flash", "meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct:free",
         "openai/gpt-oss-120b", "openai/gpt-oss-120b:free", "qwen/qwen3-coder"]),
}


@pytest.mark.parametrize("preset_id", sorted(FAMILY_LISTS))
def test_the_list_shows_the_newest_stable_version_of_each_model(preset_id):
    from flextool.assistant.providers import model_entries, shown_models

    ids, expected = FAMILY_LISTS[preset_id]
    items = [{"id": i, "supported_parameters": ["tools"]} for i in ids]
    listed = [e.id for e in model_entries(preset_id, items)]
    assert len(listed) == len(ids)                              # the whole list is what is saved
    preset = PRESETS[preset_id]
    assert shown_models(preset, listed) == expected
    assert shown_models(preset, listed, show_all=True) == listed


def test_family_and_version_of_an_id():
    from flextool.assistant.model_versions import model_key

    key = model_key("models/gemini-2.5-flash-preview-09-2025")
    assert (key.family, key.version, key.snapshot, key.preview) == ("gemini-flash", (2, 5), (20250900,), True)
    key = model_key("claude-opus-4-5-20251101")
    assert (key.family, key.version, key.snapshot, key.preview) == ("claude-opus", (4, 5), (20251101,), False)
    key = model_key("gpt-5-2025-08-07")
    assert (key.family, key.version, key.snapshot) == ("gpt", (5,), (20250807,))
    assert model_key("gemini-3-flash-lite").family == "gemini-flash-lite"
    assert model_key("meta-llama/llama-3.3-70b-instruct:free").family == "meta-llama/llama-70b-instruct:free"
    assert model_key("gpt-oss:120b-cloud").family == "gpt-oss:120b-cloud"
    assert model_key("mistral-large-latest").alias
    for model in ("tool-2-rc1", "tool-2-beta", "tool-2-alpha2", "tool-2-preview-05-06", "tool-exp-1206",
                  "tool-2-experimental", "tool-2-nightly"):
        assert model_key(model).preview, model
    assert not model_key("devstral-small-2505").preview         # a marker is a whole word, not part of one


def test_pre_releases_and_aliases_are_hidden_and_why():
    from flextool.assistant.model_versions import hidden_kind, latest_ids

    # A family of pre-releases only, or of a -latest alias only, is not shown at all.
    assert latest_ids(["tool-3-preview", "tool-3.1-preview", "tool-exp-1206", "other-latest"]) == []
    assert latest_ids(["tool-2", "tool-3-rc1", "tool-latest"]) == ["tool-2"]
    assert [hidden_kind(m) for m in ("tool-3-rc1", "tool-latest", "tool-1")] == ["pre-release", "alias", "older"]


def test_an_id_without_a_version_is_its_own_family():
    from flextool.assistant.model_versions import latest_ids, model_key

    assert model_key("assistant-model").family == "assistant-model"
    assert model_key("assistant-model").version == ()
    assert latest_ids(["assistant-model", "assistant-model-2", "assistant-model-3"]) == [
        "assistant-model", "assistant-model-3"]


def test_servers_on_this_computer_show_every_model():
    from flextool.assistant.providers import shown_models

    ids = ["qwen-2.5-7b", "qwen-3-7b"]
    assert shown_models(PRESETS["ollama"], ids) == ids
    assert shown_models(PRESETS["lmstudio"], ids) == ids
    assert shown_models(PRESETS["openai_compatible"], ids) == ["qwen-3-7b"]


# --- models that are not chat models ----------------------------------------------------------------------------
# Each service's list as (id, kept) pairs; the kept flags are worked out by hand from the word list in
# flextool/assistant/providers.py (whole words only), not from its output.  Gemini: the ids of the real list
# (2026-11-08) that are not chat models, and three of its chat models.
NOT_CHAT_LISTS: dict[str, list[tuple[str, bool]]] = {
    "gemini": [
        ("models/aqa", False), ("models/gemini-2.5-pro", True), ("models/gemini-3-pro-image", False),
        ("models/gemini-3.1-flash-image", False), ("models/gemini-3.1-flash-lite-image", False),
        ("models/gemini-3.5-transcribe", False), ("models/gemini-3.5-transcribe-live", False),
        ("models/gemini-3.8-flash", True), ("models/gemini-3.8-flash-lite", True),
        ("models/gemini-3.8-flash-lite-tts", False), ("models/gemini-3.8-flash-tts", False),
        ("models/gemini-3.8-live", False), ("models/gemini-3.8-live-extended-thinking", False),
        ("models/gemini-embedding-001", False), ("models/gemini-embedding-2", False),
        ("models/gemini-nano-banana-2.1", True),            # an image model by its name only: left in
        ("models/lyria-3.5", False)],
    "openai": [
        ("babbage-002", True), ("chatgpt-4o-latest", True), ("codex-mini-latest", True),
        ("computer-use-preview", True), ("dall-e-3", False), ("gpt-4o", True), ("gpt-4o-audio-preview", True),
        ("gpt-4o-mini-tts", False), ("gpt-4o-transcribe", False), ("gpt-5-mini", True), ("gpt-5.1", True),
        ("gpt-image-1", False), ("gpt-live-1", False), ("gpt-realtime", False), ("gpt-realtime-mini", False),
        ("o3", True), ("omni-moderation-latest", False), ("sora-2", True), ("text-embedding-3-large", False),
        ("tts-1-hd", False), ("whisper-1", False)],
    "mistral": [
        ("codestral-2508", True), ("codestral-embed-2505", False), ("devstral-medium-2507", True),
        ("magistral-medium-2509", True), ("mistral-embed-2312", False), ("mistral-large-2512", True),
        ("mistral-moderation-2411", False), ("mistral-ocr-2505", True), ("mistral-small-2506", True),
        ("pixtral-large-2411", True), ("voxtral-mini-2507", True), ("voxtral-mini-transcribe-2507", False),
        ("voxtral-mini-transcribe-realtime-2602", False), ("voxtral-mini-tts-3-0", False),
        ("voxtral-small-2507", True)],
    "openrouter": [
        ("anthropic/claude-sonnet-4.5", True), ("deepseek/deepseek-chat-v3.1", True), ("google/gemini-2.5-flash", True),
        ("google/gemini-2.5-flash-image", False), ("liquid/lfm-2.2-6b", True),
        ("meta-llama/llama-3.3-70b-instruct:free", True), ("mistralai/voxtral-small-24b-2507", True),
        ("openai/gpt-4o-audio-preview", True), ("openai/gpt-5.1", True), ("openai/gpt-oss-120b", True),
        ("qwen/qwen3-coder", True), ("x-ai/grok-4", True)],
    "anthropic": [
        ("claude-haiku-4-5-20251001", True), ("claude-opus-4-5-20251101", True),
        ("claude-sonnet-4-5-20250929", True)],
}


@pytest.mark.parametrize("preset_id", sorted(NOT_CHAT_LISTS))
def test_models_that_are_not_chat_models_are_left_out(preset_id):
    from flextool.assistant.providers import hidden_reason, is_chat_model, model_id, shown_models

    pairs = [(model_id(preset_id, raw), kept) for raw, kept in NOT_CHAT_LISTS[preset_id]]
    assert {m: is_chat_model(m) for m, _ in pairs} == dict(pairs)
    ids = [m for m, _ in pairs]
    preset = PRESETS[preset_id]
    shown = shown_models(preset, ids)
    assert not {m for m, kept in pairs if not kept} & set(shown)
    assert all(hidden_reason(m) == "not-chat" for m, kept in pairs if not kept)
    assert shown_models(preset, ids, show_all=True) == ids          # "Show all models" shows every one


def test_the_real_gemini_list_shows_its_chat_models():
    from flextool.assistant.providers import model_id, shown_models

    ids = sorted(model_id("gemini", raw) for raw, _ in NOT_CHAT_LISTS["gemini"])
    assert shown_models(PRESETS["gemini"], ids) == [
        "gemini-2.5-pro", "gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-nano-banana-2.1"]


def test_a_word_counts_only_whole():
    from flextool.assistant.providers import is_chat_model

    # Words that only contain a listed word are not it.
    for model in ("olive-7b", "deliver-2", "imagery-chat", "embedder-chat", "videogame-llm", "veolia-3", "alive"):
        assert is_chat_model(model), model
    assert not is_chat_model("vendor/some-model:embedding")
    assert not is_chat_model("Model_Video_2")                    # case and underscores do not hide a word
    assert is_chat_model("dall-3") and not is_chat_model("dall-e-2")


def test_a_server_on_this_computer_shows_its_non_chat_models():
    from flextool.assistant.providers import shown_models

    ids = ["nomic-embed-text:latest", "qwen3:8b"]
    assert shown_models(PRESETS["ollama"], ids) == ids
