"""The level of a model from its name, size and context (P3 §8 test 2; §3).

The expected levels are the ones the design states (P3 §3.2's examples column and the critique's M8 table of
real ids, both written before the code), never read back from the code: every new rule adds an example here.
Checked besides: the normalisation (vendor paths, tags, date stamps, quantisation suffixes), the size rules
(expert and active counts ignored), the reported size before the id, the window classes and their ceiling, the
allowed levels (never above the automatic level for a recognised model; higher, at the user's risk, for an
unrecognised one while the constant allows it), a stored higher level lowered with a notice, a window below
12 288 tokens refused, the probe's parameter count and its cached facts, and that nothing under ``flextool/``
can force a level past these rules.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

import pytest

from flextool.assistant import endpoint, probes, tiers
from flextool.assistant.llm_adapter import Connection
from flextool.assistant.probes import ModelFacts
from flextool.assistant.providers import PRESETS, tier_from_model
from tests.assistant.llm.local_server import FakeServer

REPO = Path(__file__).resolve().parents[3]

#: (model id, level) on a remote route without facts: P3 §3.2's examples and the critique's ids.
NAME_LEVELS = [
    # rule 1: small-model words, also joined to a version
    ("gpt-5-mini", "small"), ("o4-mini", "small"), ("gemini-2.5-flash-lite", "small"),
    ("mistral-small3.2", "small"), ("nova-micro-v1", "small"), ("amazon/nova-micro-v1", "small"),
    ("claude-haiku-4-5", "small"), ("gpt-5-nano", "small"), ("mistral-small-latest", "small"),
    # rule 2: size in the id (total parameters; active and effective counts ignored)
    ("qwen3:30b-a3b", "small"), ("gpt-oss:20b", "small"), ("qwen3-235b-a22b", "large"),
    ("command-r7b", "small"), ("example-8x7b", "medium"), ("example-70b", "medium"),
    # rule 3: fast or mid-tier variants
    ("grok-code-fast-1", "medium"), ("llama-4-maverick", "medium"), ("meta-llama/llama-4-maverick", "medium"),
    ("llama-4-maverick-17b-128e-instruct", "medium"), ("llama-4-scout-17b-16e-instruct", "medium"),
    ("nova-pro-v1", "medium"), ("amazon/nova-pro-v1", "medium"), ("sonar-pro", "medium"),
    ("perplexity/sonar-pro", "medium"), ("deepseek-coder", "medium"),
    # rule 4: large families
    ("claude-opus-5-5", "large"), ("claude-fable-5-1", "large"), ("claude-fable-5", "large"),
    ("claude-mythos-5-1", "large"), ("claude-opus-5", "large"), ("claude-sonnet-5", "large"),
    ("gemini-2.5-pro", "large"), ("models/gemini-2.5-pro", "large"), ("mistral-large-latest", "large"),
    ("o3", "large"), ("o1", "large"), ("deepseek-v3.2", "large"), ("deepseek-r1", "large"),
    ("gpt-5-pro", "large"),
    # rule 5: gpt with major 5 or more
    ("gpt-5", "large"), ("gpt-5.1", "large"), ("gpt-5.2", "large"), ("gpt-5-codex", "large"),
    # rule 6: other known families
    ("gemini-2.5-flash", "medium"), ("gemini-flash-latest", "medium"), ("mistral-medium-2508", "medium"),
    ("gpt-4o", "medium"), ("gpt-4.1", "medium"),
    # rule 7: not recognised, remote
    ("minimax-m2", "medium"),
]


@pytest.mark.parametrize(("model", "level"), NAME_LEVELS)
def test_level_from_the_name(model, level):
    choice = tiers.resolve_tier(PRESETS["openrouter"], model, None, None, "remote")
    assert choice.automatic == level, model
    assert choice.window_ceiling is None and not choice.window_known
    assert tiers.NAME_ONLY in choice.lines()


@pytest.mark.parametrize(("model", "normalised", "tokens"), [
    ("Anthropic/Claude-Opus-5-5", "claude-opus-5-5", ("claude", "opus", "5", "5")),
    ("openai/gpt-oss-20b:free", "gpt-oss-20b", ("gpt", "oss", "20b")),
    ("deepseek-r1:latest", "deepseek-r1", ("deepseek", "r1")),
    ("qwen3:30b-a3b", "qwen3:30b-a3b", ("qwen3", "30b", "a3b")),
    ("qwen2.5:14b-instruct-q4_K_M", "qwen2.5:14b-instruct", ("qwen2.5", "14b", "instruct")),
    ("llama3.1:8b-instruct-fp16", "llama3.1:8b-instruct", ("llama3.1", "8b", "instruct")),
    ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5", ("claude", "sonnet", "4", "5")),
    ("gpt-4o-2024-08-06", "gpt-4o", ("gpt", "4o")),
    ("example@20250101", "example", ("example",)),
    ("gpt-oss:120b-cloud", "gpt-oss:120b", ("gpt", "oss", "120b")),
    ("glm-4.6:cloud", "glm-4.6", ("glm", "4.6")),
])
def test_normalisation(model, normalised, tokens):
    assert tiers.normalise_model_id(model) == (normalised, tokens)


def test_whole_tokens_only():
    # gemini and minimax never match the word mini; ministral is its own word.
    assert tiers.resolve_tier(None, "gemini-2.5-flash", None, None, "remote").automatic == "medium"
    assert tiers.resolve_tier(None, "minimax-m2", None, None, "remote").source == "fallback"
    assert tiers.resolve_tier(None, "ministral-8b", None, None, "remote").reason == (
        "a small model of its family (ministral)")


@pytest.mark.parametrize(("model", "size"), [
    ("example-8x7b", 56.0), ("example-1.5b", 1.5), ("example-7b-and-14b", 14.0), ("command-r7b", 7.0),
    ("qwen3-30b-a3b", 30.0), ("example-a3b", None), ("gemma-3n-e4b-it", None), ("llama-4-scout-17b-16e", None),
    ("example", None),
])
def test_size_tag(model, size):
    assert tiers.size_tag(model) == size


def test_the_reported_size_comes_before_the_name():
    """Ollama reports the size of a model whose name has none (critique M8): the default deepseek-r1 is 8 B."""
    facts = ModelFacts(context_window=131_072, loaded_context=65_536, source="ollama", parameter_count=8_030_000_000)
    choice = tiers.resolve_tier(PRESETS["ollama"], "deepseek-r1", facts, None, "loopback")
    assert (choice.automatic, choice.source, choice.reason) == (
        "small", "reported_size", "a model of about 8 billion parameters")
    assert choice.placement() == ("This model is placed at the basic level from its size (a model of about 8 "
                                  "billion parameters). Higher levels are not available for it.")
    facts = ModelFacts(loaded_context=65_536, source="ollama", parameter_count=70_600_000_000)
    assert tiers.resolve_tier(PRESETS["ollama"], "llama3.3", facts, None, "loopback").automatic == "medium"
    # Without the reported size, llama3.3 on this computer is not recognised: a basic guess.
    choice = tiers.resolve_tier(PRESETS["ollama"], "llama3.3", ModelFacts(loaded_context=65_536), None, "loopback")
    assert (choice.automatic, choice.source) == ("small", "fallback")


def test_the_fallback_guesses_from_where_the_model_runs():
    local = tiers.resolve_tier(PRESETS["openai_compatible"], "my-finetune", None, None, "loopback")
    lan = tiers.resolve_tier(PRESETS["openai_compatible"], "my-finetune", None, None, "local_network")
    remote = tiers.resolve_tier(PRESETS["openai_compatible"], "my-finetune", None, None, "remote")
    assert (local.automatic, lan.automatic, remote.automatic) == ("small", "small", "medium")
    assert local.reason == "not recognised; guessed from where it runs"


@pytest.mark.parametrize(("window", "ceiling"), [
    (12_288, "small"), (16_384, "small"), (32_767, "small"), (32_768, "medium"), (65_535, "medium"),
    (65_536, "large"), (199_999, "large"), (200_000, "large"), (1_000_000, "large"),
])
def test_window_ceiling(window, ceiling):
    facts = ModelFacts(context_window=window, source="anthropic")
    choice = tiers.resolve_tier(PRESETS["anthropic"], "claude-opus-5-5", facts, None, "remote")
    assert choice.window_ceiling == ceiling and choice.automatic == ceiling
    assert choice.allowed == tiers.LEVEL_ORDER[: tiers.LEVEL_ORDER.index(ceiling) + 1]
    assert choice.window == window and choice.window_known


def test_the_window_ceiling_and_its_sentence():
    facts = ModelFacts(context_window=131_072, loaded_context=16_384, source="ollama")
    choice = tiers.resolve_tier(PRESETS["ollama"], "example:70b", facts, None, "loopback")
    assert (choice.name_level, choice.automatic, choice.allowed) == ("medium", "small", ("small",))
    assert choice.window_source == "loaded"
    assert choice.lines() == [
        "This model is placed at the standard level from its size (a model of about 70 billion parameters).",
        "This model runs with a 16384-token context, so the highest level is basic. A larger context allows a "
        "higher level. Set OLLAMA_CONTEXT_LENGTH or the model's context length in its load settings."]
    remote = tiers.resolve_tier(PRESETS["anthropic"], "claude-opus-5-5", ModelFacts(context_window=40_000),
                                None, "remote")
    assert remote.window_line() == ("This model runs with a 40000-token context, so the highest level is "
                                    "standard. A larger context allows a higher level.")


def test_below_the_minimum_no_session():
    facts = ModelFacts(loaded_context=12_287, source="ollama")
    with pytest.raises(tiers.WindowTooSmall) as info:
        tiers.resolve_tier(PRESETS["ollama"], "qwen2.5:14b", facts, None, "loopback")
    assert (info.value.window, info.value.local) == (12_287, True)
    preset = PRESETS["ollama"]
    conn = Connection(base_url="http://localhost:11434/v1", model="qwen2.5:14b", api_key="", key_source="none",
                      knobs={}, route=endpoint.route_for(preset, "http://localhost:11434/v1", model="qwen2.5:14b"))
    err = probes.window_error(preset, conn, info.value)
    assert err.code == "small_context"
    assert "12287-token context" in err.user_message and "OLLAMA_CONTEXT_LENGTH=16384" in err.user_message


@pytest.mark.parametrize(("facts", "local", "window", "source"), [
    (ModelFacts(context_window=131_072, loaded_context=32_768), True, 32_768, "loaded"),
    (ModelFacts(context_window=131_072), False, 131_072, "model"),
    (ModelFacts(listed_context=40_960), True, 40_960, "model_list"),
    (ModelFacts(), True, 16_384, "assumed"),
    (ModelFacts(), False, 128_000, "assumed"),
])
def test_where_the_window_comes_from(facts, local, window, source):
    choice = tiers.resolve_tier(None, "claude-opus-5-5", facts, None, "loopback" if local else "remote")
    assert (choice.window, choice.window_source, choice.window_known) == (window, source, source != "assumed")


@pytest.mark.parametrize("model", [m for m, _ in NAME_LEVELS if not m.startswith("minimax")])
def test_a_recognised_model_is_never_offered_above_its_level(model):
    for window in (16_384, 40_000, 100_000, 400_000):
        choice = tiers.resolve_tier(None, model, ModelFacts(context_window=window), None, "remote")
        assert choice.source != "fallback"
        assert choice.allowed == tiers.LEVEL_ORDER[: tiers.LEVEL_ORDER.index(choice.automatic) + 1]
        choice = tiers.resolve_tier(None, model, ModelFacts(context_window=window), "advanced", "remote")
        assert tiers.LEVEL_ORDER.index(choice.chosen) <= tiers.LEVEL_ORDER.index(choice.automatic)


def test_a_stored_higher_level_is_lowered_with_a_notice():
    choice = tiers.resolve_tier(PRESETS["openai"], "gpt-5-mini", None, "advanced", "remote")
    assert (choice.chosen, choice.automatic) == ("small", "small")
    assert choice.notice == ("The level chosen before (advanced) is above this model's level, so the automatic "
                             "level is used.")
    assert choice.notice in choice.lines()
    # The same after a probe lowered the level for the context (resume after the model's context changed).
    facts = ModelFacts(context_window=20_000)
    choice = tiers.resolve_tier(PRESETS["anthropic"], "claude-opus-5-5", facts, "standard", "remote")
    assert (choice.chosen, choice.notice is not None) == ("small", True)
    # A lower level is kept, by either name.
    for name in ("basic", "small"):
        choice = tiers.resolve_tier(PRESETS["openai"], "gpt-5.1", None, name, "remote")
        assert (choice.chosen, choice.notice, choice.level_name) == ("small", None, "basic")


def test_an_unrecognised_model_may_go_higher_at_the_users_risk():
    assert tiers.ALLOW_HIGHER_WHEN_UNRECOGNISED is True        # P3-O12, final
    facts = ModelFacts(context_window=40_000)
    choice = tiers.resolve_tier(None, "my-finetune", facts, None, "remote")
    assert (choice.automatic, choice.allowed) == ("medium", ("small", "medium"))     # up to the window's level
    choice = tiers.resolve_tier(None, "my-finetune", None, "advanced", "loopback")
    assert (choice.automatic, choice.chosen, choice.allowed) == ("small", "large", tiers.LEVEL_ORDER)
    assert choice.chosen_note == tiers.UNRECOGNISED and choice.notice is None
    assert choice.lines().count(tiers.UNRECOGNISED) == 1


def test_an_unrecognised_model_stays_at_its_guess_when_the_constant_is_off(monkeypatch):
    monkeypatch.setattr(tiers, "ALLOW_HIGHER_WHEN_UNRECOGNISED", False)
    choice = tiers.resolve_tier(None, "my-finetune", None, "advanced", "loopback")
    assert (choice.allowed, choice.chosen, choice.chosen_note) == (("small",), "small", None)
    assert choice.notice is not None


def test_tier_from_model_keeps_p1_callers():
    assert tier_from_model("openai", "gpt-5-mini") == "small"
    assert tier_from_model("anthropic", "claude-opus-5-5") == "large"
    assert tier_from_model("ollama", "my-finetune") == "small"           # a local preset: the local guess
    assert tier_from_model("openrouter", "minimax/minimax-m2") == "medium"


def test_nothing_forces_a_level_past_the_rules():
    """P3-D9: the forced level choice of the calibration lives in tools/, never in the shipped code."""
    pattern = re.compile(r"forced_choice|force_level|forced_level|bypass_level|level_bypass")
    hits = [str(p.relative_to(REPO)) for p in (REPO / "flextool").rglob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))]
    assert hits == []


# --- the facts the level is resolved from ---------------------------------------------------------------------
def test_ollama_parameter_count():
    assert probes.ollama_parameter_count({"model_info": {"general.parameter_count": 20_914_757_184}}) == \
        20_914_757_184
    assert probes.ollama_parameter_count({"details": {"parameter_size": "14.8B"}}) == 14_800_000_000
    assert probes.ollama_parameter_count({"details": {"parameter_size": "568M"}}) == 568_000_000
    assert probes.ollama_parameter_count({"details": {"parameter_size": "large"}}) is None
    assert probes.ollama_parameter_count({}) is None


def _ollama_server(server: FakeServer, model: str, loaded: int) -> Connection:
    server.routes[("POST", "/api/show")] = (200, {
        "capabilities": ["completion", "tools"], "details": {"parameter_size": "8.0B"},
        "model_info": {"general.architecture": "qwen2", "qwen2.context_length": 131_072,
                       "general.parameter_count": 8_030_261_248}})
    server.routes[("POST", "/api/generate")] = (200, {"model": model, "done": True})
    server.routes[("GET", "/api/ps")] = (200, {"models": [
        {"name": model, "model": model, "size": 5, "size_vram": 5, "context_length": loaded}]})
    base = f"http://127.0.0.1:{server.port}/v1"
    return Connection(base_url=base, model=model, api_key="", key_source="none", knobs={},
                      route=endpoint.route_for(PRESETS["ollama"], base, model=model))


def test_the_probe_reports_the_size_and_keeps_the_facts():
    server = FakeServer()
    try:
        conn = _ollama_server(server, "deepseek-r1:latest", 65_536)
        facts = probes.probe(PRESETS["ollama"], conn, threading.Event(), needed=12_288)
    finally:
        server.close()
    assert (facts.parameter_count, facts.loaded_context) == (8_030_261_248, 65_536)
    choice = tiers.resolve_tier(PRESETS["ollama"], conn.model, facts, None, conn.route)
    assert (choice.automatic, choice.source) == ("small", "reported_size")
    # The settings read the same facts later without a probe: from this process, then from the user cache.
    assert probes.cached_facts("ollama", conn.base_url, conn.model) == facts
    probes._RECENT.clear()
    cached = probes.cached_facts("ollama", conn.base_url, conn.model)
    assert cached is not None and (cached.parameter_count, cached.loaded_context, cached.context_window) == (
        8_030_261_248, 65_536, 131_072)
    assert probes.cached_facts("ollama", conn.base_url, "another:7b") is None


def test_a_custom_endpoint_lists_the_context():
    server = FakeServer()
    try:
        base = f"http://127.0.0.1:{server.port}/v1"
        server.routes[("GET", "/v1/models")] = (200, {"data": [
            {"id": "other", "max_model_len": 4096}, {"id": "served-model", "max_model_len": 32_768}]})
        preset = PRESETS["openai_compatible"]
        conn = Connection(base_url=base, model="served-model", api_key="", key_source="none", knobs={},
                          route=endpoint.route_for(preset, base, model="served-model"))
        facts = probes.probe(preset, conn, threading.Event())
        missing = probes.probe(preset, Connection(base_url=base, model="not-served", api_key="", key_source="none",
                                                  knobs={}, route=conn.route), threading.Event())
    finally:
        server.close()
    assert facts.listed_context == 32_768
    choice = tiers.resolve_tier(preset, "served-model", facts, None, conn.route)
    assert (choice.window, choice.window_source, choice.window_ceiling) == (32_768, "model_list", "medium")
    assert missing.listed_context is None


# --- advanced models before and after the probe (owner GUI feedback 2026-11-08 (2)) -------------------------------
#: Ids the design places at advanced (P3 §3.2 rule 4 and 5), on the provider they run at.
ADVANCED_IDS = [("anthropic", "claude-opus-5-5"), ("anthropic", "claude-sonnet-5"), ("anthropic", "claude-fable-5-1"),
                ("openai", "gpt-5"), ("openai", "o3"), ("gemini", "gemini-2.5-pro"),
                ("openrouter", "anthropic/claude-opus-5-5")]
#: (facts, why): no probe yet; a probe that found no context (a remote service: assumed 128 000); the model's
#: own context; the context the model list gives.
ADVANCED_FACTS = [(None, "before the probe"), (ModelFacts(source="none"), "unknown remote context"),
                  (ModelFacts(context_window=1_000_000, source="anthropic"), "1M context"),
                  (ModelFacts(listed_context=200_000), "listed context")]


@pytest.mark.parametrize(("preset_id", "model"), ADVANCED_IDS)
@pytest.mark.parametrize(("facts", "why"), ADVANCED_FACTS)
def test_an_advanced_model_is_advanced_before_and_after_the_probe(preset_id, model, facts, why):
    preset = PRESETS[preset_id]
    route = endpoint.route_for(preset, preset.base_url, model=model, facts=facts)
    choice = tiers.resolve_tier(preset, model, facts, None, route)
    assert (choice.automatic, choice.chosen, choice.window_line()) == ("large", "large", None), why


@pytest.mark.parametrize(("preset_id", "model"), ADVANCED_IDS)
def test_the_gui_and_the_terminal_give_an_advanced_model_advanced_before_the_probe(preset_id, model):
    from flextool.assistant.chat_cli import Choice
    from flextool.gui.assistant.worker import _level_before_probe

    preset = PRESETS[preset_id]
    route = endpoint.route_for(preset, preset.base_url, model=model)
    choice = Choice(preset=preset, base_url=preset.base_url, model=model, tier=None, knobs={})
    assert _level_before_probe(preset, preset.base_url, model, None, route, None)[0] == "large"
    assert choice.level == "large"
    # With the facts a probe found (kept in the user cache), and after the probe itself.
    probes.remember_facts(preset_id, preset.base_url, model, ModelFacts(context_window=1_000_000, source="x"))
    cached = probes.cached_facts(preset_id, preset.base_url, model)
    assert _level_before_probe(preset, preset.base_url, model, None, route, cached)[0] == "large"
    assert choice.level == "large"
    assert choice.tier_choice(ModelFacts(source="none")).chosen == "large"
    assert choice.tier_choice(ModelFacts(context_window=1_000_000)).chosen == "large"
