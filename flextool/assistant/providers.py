"""The AI services FlexTool's chat can use (P1 §7): one :class:`Preset` each, and the level a model name suggests.

A preset names no models (P1-O1): the model is chosen from the service's live model list.  It says which request
fields the service accepts (``allowed_params``; a field outside it is never sent), which privacy switches FlexTool
sends with every request (``request_knobs``), and which ones only the user's account can set (``account_knobs``,
shown as "set in your provider console").  ``last_checked`` is the date the preset's facts were last checked.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from flextool.assistant.llm_adapter import ModelEntry


@dataclass(frozen=True)
class Knob:
    """A field FlexTool sends with every request to ask the service to handle the data in a certain way."""

    id: str
    label: str
    path: tuple[str, ...]           # ("store",) or ("provider", "data_collection")
    on_value: object
    off_value: object | None        # None: the field is left out when the knob is off
    default_on: bool
    report: str                     # how the request carries it when on, e.g. "request sent with `store: false`"
    note: str | None = None
    relaxed: str | None = None      # the status-line text when a default-on knob is turned off


@dataclass(frozen=True)
class Preset:
    id: str
    label: str
    backend: Literal["anthropic", "openai"]
    base_url: str | None
    key_env: str | None
    key_required: bool
    key_optional: bool
    allowed_params: frozenset[str]
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"]
    default_max_output_tokens: int
    request_knobs: tuple[Knob, ...]
    account_knobs: tuple[str, ...]
    required_headers: frozenset[str]
    capture_tool_call_keys: frozenset[str]
    capture_message_keys: frozenset[str]
    tool_message_name: bool
    cost_field: str | None
    context_probe: Literal["anthropic", "openrouter", "ollama", "lmstudio"] | None
    local: bool
    server_name: str | None
    last_checked: str
    require_parameters_with_tools: bool = False     # OpenRouter: provider.require_parameters whenever tools go

    def knob(self, knob_id: str) -> Knob | None:
        return next((k for k in self.request_knobs if k.id == knob_id), None)


_CHECKED = "2026-11-06"

STORE_KNOB = Knob(
    id="store", label="Ask OpenAI not to store the requests for later viewing in your account", path=("store",),
    on_value=False, off_value=None, default_on=True, report="request sent with `store: false`",
    relaxed="privacy setting relaxed: requests may be stored in your OpenAI account")
DATA_COLLECTION_KNOB = Knob(
    id="data_collection", label="Use only providers that do not collect the data of your requests",
    path=("provider", "data_collection"), on_value="deny", off_value="allow", default_on=True,
    report="request sent with `provider.data_collection: deny`",
    relaxed="privacy setting relaxed: data collection allowed")
ZDR_KNOB = Knob(
    id="zdr", label="Use only providers that keep no copy of your requests", path=("provider", "zdr"),
    on_value=True, off_value=None, default_on=False, report="request sent with `provider.zdr: true`",
    note=("Only use providers that keep no copy of your requests. This narrows the list of models you can use, "
          "free models especially."))

PRESETS: dict[str, Preset] = {p.id: p for p in (
    Preset(
        id="openai", label="OpenAI", backend="openai", base_url="https://api.openai.com/v1",
        key_env="OPENAI_API_KEY", key_required=True, key_optional=False,
        allowed_params=frozenset({"store", "stream_options", "tool_choice", "parallel_tool_calls",
                                  "prompt_cache_key"}),
        max_tokens_param="max_completion_tokens", default_max_output_tokens=16000,
        request_knobs=(STORE_KNOB,),
        account_knobs=("the data sharing and training settings of your organisation",
                       "zero data retention, which needs approval from OpenAI"),
        required_headers=frozenset(), capture_tool_call_keys=frozenset(), capture_message_keys=frozenset(),
        tool_message_name=False, cost_field=None, context_probe=None, local=False, server_name=None,
        last_checked=_CHECKED),
    Preset(
        id="anthropic", label="Anthropic", backend="anthropic", base_url="https://api.anthropic.com",
        key_env="ANTHROPIC_API_KEY", key_required=True, key_optional=False,
        allowed_params=frozenset({"tool_choice", "cache_control"}),
        max_tokens_param="max_tokens", default_max_output_tokens=16000, request_knobs=(),
        account_knobs=("the data retention arrangements and training terms of your account",),
        required_headers=frozenset(), capture_tool_call_keys=frozenset(), capture_message_keys=frozenset(),
        tool_message_name=False, cost_field=None, context_probe="anthropic", local=False, server_name=None,
        last_checked=_CHECKED),
    Preset(
        id="mistral", label="Mistral", backend="openai", base_url="https://api.mistral.ai/v1",
        key_env="MISTRAL_API_KEY", key_required=True, key_optional=False,
        allowed_params=frozenset({"tool_choice", "parallel_tool_calls"}),
        max_tokens_param="max_tokens", default_max_output_tokens=8192, request_knobs=(),
        account_knobs=("whether your plan allows training on your data, and the opt-out in your account",),
        required_headers=frozenset(), capture_tool_call_keys=frozenset(), capture_message_keys=frozenset(),
        tool_message_name=True, cost_field=None, context_probe=None, local=False, server_name=None,
        last_checked=_CHECKED),
    Preset(
        id="openrouter", label="OpenRouter", backend="openai", base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY", key_required=True, key_optional=False,
        allowed_params=frozenset({"tool_choice", "parallel_tool_calls", "provider"}),
        max_tokens_param="max_tokens", default_max_output_tokens=8192,
        request_knobs=(DATA_COLLECTION_KNOB, ZDR_KNOB),
        account_knobs=("your OpenRouter privacy settings, which apply on top",),
        required_headers=frozenset(), capture_tool_call_keys=frozenset({"extra_content"}),
        capture_message_keys=frozenset({"reasoning_details"}), tool_message_name=False, cost_field="usage.cost",
        context_probe="openrouter", local=False, server_name=None, last_checked=_CHECKED,
        require_parameters_with_tools=True),
    Preset(
        id="gemini", label="Google Gemini", backend="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/", key_env="GEMINI_API_KEY",
        key_required=True, key_optional=False, allowed_params=frozenset({"stream_options", "tool_choice"}),
        max_tokens_param="max_tokens", default_max_output_tokens=16000, request_knobs=(),
        account_knobs=(("free tier: Google may use prompts and answers to improve its products; paid tier terms "
                        "differ"),),
        required_headers=frozenset(), capture_tool_call_keys=frozenset({"extra_content"}),
        capture_message_keys=frozenset(), tool_message_name=False, cost_field=None, context_probe=None,
        local=False, server_name=None, last_checked=_CHECKED),
    Preset(
        id="ollama", label="Ollama", backend="openai", base_url="http://localhost:11434/v1", key_env=None,
        key_required=False, key_optional=True, allowed_params=frozenset({"stream_options", "tool_choice"}),
        max_tokens_param="max_tokens", default_max_output_tokens=4096, request_knobs=(),
        account_knobs=("Ollama cloud models: check Ollama's terms for cloud models yourself",),
        required_headers=frozenset(), capture_tool_call_keys=frozenset(), capture_message_keys=frozenset(),
        tool_message_name=False, cost_field=None, context_probe="ollama", local=True, server_name="Ollama",
        last_checked=_CHECKED),
    Preset(
        id="lmstudio", label="LM Studio", backend="openai", base_url="http://localhost:1234/v1", key_env=None,
        key_required=False, key_optional=True, allowed_params=frozenset({"stream_options", "tool_choice"}),
        max_tokens_param="max_tokens", default_max_output_tokens=4096, request_knobs=(), account_knobs=(),
        required_headers=frozenset(), capture_tool_call_keys=frozenset(), capture_message_keys=frozenset(),
        tool_message_name=False, cost_field=None, context_probe="lmstudio", local=True, server_name="LM Studio",
        last_checked=_CHECKED),
    Preset(
        id="openai_compatible", label="Custom endpoint", backend="openai", base_url=None, key_env=None,
        key_required=False, key_optional=True, allowed_params=frozenset({"tool_choice"}),
        max_tokens_param="max_tokens", default_max_output_tokens=4096, request_knobs=(),
        account_knobs=("Check how this service handles data yourself.",),
        required_headers=frozenset(), capture_tool_call_keys=frozenset({"extra_content"}),
        capture_message_keys=frozenset(), tool_message_name=False, cost_field=None, context_probe=None,
        local=False, server_name=None, last_checked=_CHECKED),
)}


# --- knobs --------------------------------------------------------------------------------------------------------
def knob_key(preset: Preset, knob: Knob) -> str:
    """The settings key of *knob*: ``"<preset>.<knob>"``."""
    return f"{preset.id}.{knob.id}"


def knob_on(preset: Preset, knob: Knob, knobs: Mapping[str, object]) -> bool:
    """Whether *knob* is on in the settings *knobs*: ``True``/``False``, or the knob's on or off value as sent
    (``"deny"``); absent means the knob's default."""
    value = knobs.get(knob_key(preset, knob))
    if value is None:
        return knob.default_on
    if isinstance(value, bool):
        return value
    if value == knob.on_value:
        return True
    if value == knob.off_value:
        return False
    raise ValueError(f"the setting {knob_key(preset, knob)} has the value {value!r}, which is neither "
                     f"{knob.on_value!r} nor {knob.off_value!r}")


def request_fields(preset: Preset, knobs: Mapping[str, object]) -> dict[str, Any]:
    """The request fields the knobs of *preset* add, nested by their paths."""
    out: dict[str, Any] = {}
    for knob in preset.request_knobs:
        value = knob.on_value if knob_on(preset, knob, knobs) else knob.off_value
        if value is None:
            continue
        target = out
        for part in knob.path[:-1]:
            target = target.setdefault(part, {})
        target[knob.path[-1]] = value
    return out


def knob_reports(preset: Preset, knobs: Mapping[str, object]) -> list[str]:
    """How each knob that is on is carried by the requests."""
    return [k.report for k in preset.request_knobs if knob_on(preset, k, knobs)]


def relaxed_knobs(preset: Preset, knobs: Mapping[str, object]) -> list[str]:
    """The status-line texts of the default-on knobs the user turned off."""
    return [k.relaxed or k.label for k in preset.request_knobs
            if k.default_on and not knob_on(preset, k, knobs)]


# --- the level a model name suggests ------------------------------------------------------------------------------
Tier = Literal["small", "medium", "large"]

#: (pattern, tier), tried in order on the lower-case model name; the first match wins, else medium.  Size tags
#: are handled by :func:`tier_from_model` before these.
TIER_PATTERNS: tuple[tuple[str, Tier], ...] = tuple((rf"(?<![a-z0-9]){word}(?![a-z0-9])", tier) for word, tier in (
    ("mini", "small"),
    ("nano", "small"),
    ("haiku", "small"),
    ("flash-lite", "small"),
    ("small", "small"),
    ("tiny", "small"),
    ("opus", "large"),
    ("fable", "large"),
    ("pro", "large"),
    ("large", "large"),
    (r"gpt-5(\.\d+)?", "large"),
    ("o3", "large"),
))
#: A parameter-count tag such as ``7b``, ``14b``, ``8x7b`` or ``1.5b``.
_SIZE_TAG = re.compile(r"(?<![a-z0-9.])(?:\d+x)?(\d+(?:\.\d+)?)b(?![a-z0-9])")
SMALL_SIZE_LIMIT = 32.0         # billion parameters


def tier_from_model(preset_id: str, model: str) -> Tier:
    """The level a model name suggests: small for size tags up to about 30 billion parameters and small-model
    words, large for the providers' top families, else medium.  The user's choice always wins over this."""
    name = model.lower()
    sizes = [float(m.group(1)) for m in _SIZE_TAG.finditer(name)]
    if sizes and max(sizes) <= SMALL_SIZE_LIMIT:
        return "small"
    for pattern, tier in TIER_PATTERNS:
        if re.search(pattern, name):
            return tier
    return "medium"


# --- the live model list ------------------------------------------------------------------------------------------
_NOT_CHAT = re.compile(r"embed|audio|image|moderation|realtime|transcribe|tts", re.IGNORECASE)


def is_cloud_name(model: str) -> bool:
    """Whether an Ollama model name marks a cloud model (``:cloud`` or ``-cloud`` at the end)."""
    return model.endswith((":cloud", "-cloud"))


def model_entries(preset_id: str, items: Iterable[Mapping[str, Any]]) -> list[ModelEntry]:
    """The usable chat models of a service's model list (the raw items of its list endpoint), sorted by id."""
    out: dict[str, ModelEntry] = {}
    for item in items:
        model_id = str(item.get("id") or item.get("name") or "")
        if not model_id:
            continue
        note = None
        context = item.get("context_length") or item.get("max_input_tokens")
        if preset_id == "openai" and _NOT_CHAT.search(model_id):
            continue
        if preset_id == "openrouter" and "tools" not in (item.get("supported_parameters") or []):
            continue
        if preset_id == "gemini":
            model_id = model_id.removeprefix("models/")
        if preset_id == "ollama" and is_cloud_name(model_id):
            note = "runs on Ollama cloud"
        out[model_id] = ModelEntry(model_id, note, int(context) if isinstance(context, int) else None)
    return [out[k] for k in sorted(out)]
