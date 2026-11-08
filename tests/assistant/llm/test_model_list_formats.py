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
