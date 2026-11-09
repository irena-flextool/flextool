"""The AI services FlexTool's chat can use (P1 §7): one :class:`Preset` each.  The level of a model is
:mod:`flextool.assistant.tiers`.

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
    #: Seconds between request starts while the free plan is on, from the provider's documented free-plan limit
    #: (a rate hint for the pacing, P1 §5.1; not a model name), and where and when it was checked.
    free_plan_interval: float | None = None
    free_plan_interval_source: str = ""
    #: Seconds between request starts always, free plan or not (owner decision 2026-11-08), and its source.
    min_interval: float | None = None
    min_interval_source: str = ""

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
        last_checked=_CHECKED, min_interval=1.1,
        min_interval_source=("docs.mistral.ai/admin/workspaces/usage-limits, checked 2026-11-08: the free mode is "
                             "limited in requests per second (the number is shown only in the account's limits "
                             "page); 1.1 s keeps within any limit of one or more per second.  Applied also off the "
                             "free plan (owner decision 2026-11-08): a paid account loses about a second per tool "
                             "round")),
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
        require_parameters_with_tools=True, free_plan_interval=3.0,
        free_plan_interval_source=("openrouter.ai/docs/api/reference/limits, checked 2026-11-08: free models "
                                   "(ids ending :free) allow 20 requests per minute")),
    # No free-plan interval: Gemini's free-tier limits differ by model and are shown only in Google AI Studio
    # (aistudio.google.com/docs/rate-limits, checked 2026-11-08); its rate-limit answers name the wait instead.
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


def pacing_seed(preset: Preset, free_plan: bool) -> float | None:
    """The interval the pacing starts from and never goes below (:class:`flextool.assistant.pacing.Pacer`): the
    preset's :attr:`~Preset.min_interval`, and with the free plan on also its free-plan interval (the larger);
    ``None`` when neither applies."""
    seeds = [v for v in (preset.min_interval, preset.free_plan_interval if free_plan else None) if v]
    return max(seeds) if seeds else None


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


def tier_from_model(preset_id: str, model: str) -> Tier:
    """The level the name of *model* gives (internal name), before any probe: :func:`flextool.assistant.tiers.
    resolve_tier` without facts, an unrecognised name guessed from where the preset's models run.  P1's callers
    keep this name; the level in use comes from ``resolve_tier`` with the probe's facts."""
    from flextool.assistant import tiers

    preset = PRESETS.get(preset_id)
    return tiers.resolve_tier(preset, model, None, None).automatic  # type: ignore[return-value]


# --- the live model list ------------------------------------------------------------------------------------------
def is_cloud_name(model: str) -> bool:
    """Whether an Ollama model name marks a cloud model (``:cloud`` or ``-cloud`` at the end)."""
    return model.endswith((":cloud", "-cloud"))


def model_id(preset_id: str, name: str) -> str:
    """The id FlexTool stores, checks and sends for the model *name* of the preset *preset_id*: one rule for the
    model list, the saved setting, the probe and the request.  Gemini's OpenAI-compatible list gives
    ``models/gemini-2.5-flash`` where its chat endpoint is documented with ``gemini-2.5-flash``, so the prefix
    is dropped; every other preset uses its list's ids as they are (spaces around them dropped)."""
    name = name.strip()
    if preset_id == "gemini":
        name = name.removeprefix("models/")
    return name


def model_entries(preset_id: str, items: Iterable[Mapping[str, Any]]) -> list[ModelEntry]:
    """The usable models of a service's model list (the raw items of its list endpoint), sorted by id.  Every id
    is kept, whatever kind of model it names: :func:`shown_models` hides the models for something other than a chat
    when the list is shown (RF-D12).  Anthropic's retired models (``lifecycle``) are left out and its deprecated
    ones noted: they are closed to new accounts; OpenRouter's models without tool calls are left out."""
    out: dict[str, ModelEntry] = {}
    for item in items:
        model_id_ = model_id(preset_id, str(item.get("id") or item.get("name") or ""))
        if not model_id_:
            continue
        note = None
        lifecycle = item.get("lifecycle")
        if lifecycle == "retired":
            continue
        if lifecycle == "deprecated":
            note = "deprecated: closed to new accounts"
        context = next((item[k] for k in ("context_length", "max_input_tokens", "max_context_length")
                        if isinstance(item.get(k), int) and not isinstance(item.get(k), bool) and item[k] > 0), None)
        if preset_id == "openrouter" and "tools" not in (item.get("supported_parameters") or []):
            continue
        if preset_id == "ollama" and is_cloud_name(model_id_):
            note = "runs on Ollama cloud"
        out[model_id_] = ModelEntry(model_id_, note, int(context) if isinstance(context, int) else None)
    return [out[k] for k in sorted(out)]


#: Whole words of a model id that only a model for something other than a chat has (owner decisions 2026-11-08):
#: speech, sound, images, video, music, embeddings, moderation and ranking — the kinds of model, not names of
#: models (P1-O1).  A word is matched against the id's words (split at ``-``, ``_``, ``.``, ``/`` and ``:``), never
#: as part of a word.  Only words that never name a chat model are here: a non-chat model the list keeps is better
#: than a chat model it drops.  ``live`` and ``realtime`` are the realtime voice models (Gemini's Live API, OpenAI's
#: Realtime API, Mistral's realtime transcription), which no chat request reaches.  ``audio`` is not a word here:
#: OpenAI's audio models (``gpt-4o-audio-preview``) answer chat requests with tools.
NOT_CHAT_WORDS: tuple[tuple[str, ...], ...] = tuple(tuple(w.split("-")) for w in (
    "embedding", "embeddings", "embed", "tts", "transcribe", "transcription", "speech", "image", "images",
    "imagen", "veo", "video", "lyria", "music", "aqa", "moderation", "rerank", "reranker", "whisper", "dall-e",
    "live", "realtime"))
_ID_WORDS = re.compile(r"[-_./:]+")


def is_chat_model(model: str) -> bool:
    """Whether the id *model* has none of :data:`NOT_CHAT_WORDS` as a whole word (or run of words)."""
    words = [w for w in _ID_WORDS.split(model.strip().lower().removeprefix("models/")) if w]
    for target in NOT_CHAT_WORDS:
        n = len(target)
        if any(tuple(words[i:i + n]) == target for i in range(len(words) - n + 1)):
            return False
    return True


def lists_latest_only(preset: Preset) -> bool:
    """Whether the model list of *preset* is shown with only the newest stable version of each family by default
    (:mod:`flextool.assistant.model_versions`): the remote services.  Ollama and LM Studio list exactly the models
    the user installed, so all of them are shown."""
    return not preset.local


def shown_models(preset: Preset, ids: Iterable[str], *, show_all: bool = False,
                 unavailable: Iterable[str] = ()) -> list[str]:
    """The ids of a model list as shown: all of them with *show_all*; else without the models a test found not
    available to the account (*unavailable*), and for a remote service only the chat models (:func:`is_chat_model`)
    and of those the newest stable version of each family (no pre-releases, no ``latest`` aliases).  The saved list
    keeps every id; this runs when the list is shown."""
    from flextool.assistant.model_versions import latest_ids

    ids = list(ids)
    if show_all:
        return ids
    closed = set(unavailable)
    ids = [i for i in ids if i not in closed]
    if not lists_latest_only(preset):
        return ids
    return latest_ids(i for i in ids if is_chat_model(i))


def hidden_reason(model: str, unavailable: Iterable[str] = ()) -> str:
    """Why the default list leaves *model* out: ``"unavailable"`` (a test found it not available to the account),
    ``"not-chat"`` (:func:`is_chat_model`), else :func:`flextool.assistant.model_versions.hidden_kind`
    (``"pre-release"``, ``"alias"`` or ``"older"``)."""
    from flextool.assistant.model_versions import hidden_kind

    if model in set(unavailable):
        return "unavailable"
    return "not-chat" if not is_chat_model(model) else hidden_kind(model)
