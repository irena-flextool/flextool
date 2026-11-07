"""The presets (P1 §7): no model names, knobs inside the allowed fields, the defaults the owner chose.  The level
of a model is ``test_tier_resolution.py``."""
from __future__ import annotations

import dataclasses

import pytest

from flextool.assistant.providers import (
    PRESETS,
    knob_on,
    knob_reports,
    relaxed_knobs,
    request_fields,
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
