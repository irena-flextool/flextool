"""The presets (P1 §7): no model names, knobs inside the allowed fields, the defaults the owner chose.  The level
of a model is ``test_tier_resolution.py``."""
from __future__ import annotations

import dataclasses

import pytest

from flextool.assistant.providers import (
    PRESETS,
    hidden_reason,
    knob_on,
    knob_reports,
    model_entries,
    relaxed_knobs,
    request_fields,
    shown_models,
)


def test_presets_name_no_models():
    fields = {f.name for f in dataclasses.fields(next(iter(PRESETS.values())))}
    assert not {f for f in fields if "model" in f}


def test_the_key_variables():
    assert {p.key_env for p in PRESETS.values() if p.key_env} == {
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "OPENROUTER_API_KEY", "GEMINI_API_KEY"}
    for preset in PRESETS.values():
        assert preset.key_required != preset.key_optional
        assert preset.key_required == (preset.key_env is not None)


@pytest.mark.parametrize("preset", PRESETS.values(), ids=lambda p: p.id)
def test_knob_fields_are_allowed(preset):
    for knob in preset.request_knobs:
        assert knob.path[0] in preset.allowed_params
    if preset.require_parameters_with_tools:
        assert "provider" in preset.allowed_params
    assert preset.local == (preset.server_name is not None)
    assert preset.last_checked


def test_openrouter_defaults_deny_data_collection_and_leave_zdr_off():
    openrouter = PRESETS["openrouter"]
    assert request_fields(openrouter, {}) == {"provider": {"data_collection": "deny"}}
    assert request_fields(openrouter, {"openrouter.zdr": True}) == {"provider": {"data_collection": "deny",
                                                                                 "zdr": True}}
    assert request_fields(openrouter, {"openrouter.data_collection": "allow", "openrouter.zdr": False}) == {
        "provider": {"data_collection": "allow"}}
    assert knob_reports(openrouter, {}) == ["request sent with `provider.data_collection: deny`"]
    assert relaxed_knobs(openrouter, {"openrouter.data_collection": False}) == [
        "privacy setting relaxed: data collection allowed"]
    assert relaxed_knobs(openrouter, {"openrouter.zdr": False}) == []
    with pytest.raises(ValueError):
        knob_on(openrouter, openrouter.knob("data_collection"), {"openrouter.data_collection": "sometimes"})


def test_openai_sends_store_false_by_default():
    openai = PRESETS["openai"]
    assert request_fields(openai, {}) == {"store": False}
    assert request_fields(openai, {"openai.store": False}) == {}
    assert knob_reports(openai, {}) == ["request sent with `store: false`"]


def test_openai_list_keeps_every_id():
    """C4-n2: the saved list keeps every id (RF-D12); the shown list hides the non-chat kinds by whole words, and
    ``audio`` is not one of them."""
    ids = ["gpt-4o-audio-preview", "gpt-image-1", "text-embedding-3-small", "gpt-4o", "gpt-audio"]
    saved = [e.id for e in model_entries("openai", [{"id": i, "object": "model"} for i in ids])]
    assert saved == ["gpt-4o", "gpt-4o-audio-preview", "gpt-audio", "gpt-image-1", "text-embedding-3-small"]
    assert shown_models(PRESETS["openai"], saved) == ["gpt-4o", "gpt-audio"]
    assert hidden_reason("gpt-4o-audio-preview") == "pre-release"
    assert hidden_reason("gpt-image-1") == hidden_reason("text-embedding-3-small") == "not-chat"
    assert shown_models(PRESETS["openai"], saved, show_all=True) == saved
