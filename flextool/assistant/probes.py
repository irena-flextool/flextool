"""What FlexTool finds out about a model before the first question is sent (P1 §7.2, §9.2).

:func:`probe` returns the model's :class:`ModelFacts` — its context window, the context it is loaded with, whether
it supports tool calls, and the host a local server forwards it to — or raises the error that stops the chat before
anything reaches the model:

* **Ollama** (the base URL without ``/v1``): ``POST /api/show`` gives the capabilities (no ``tools`` ->
  :class:`ToolsUnsupported`), the maximum context and ``remote_host`` / ``remote_model``.  A model with a
  ``remote_host``, or a name ending in ``:cloud`` or ``-cloud``, runs on another host (Ollama cloud): it is not
  loaded here and its route is remote.  A local model is loaded first (``POST /api/generate`` with
  ``keep_alive``, the documented load call; up to 900 seconds, cancellable, :class:`Waiting` every 5 seconds), then
  ``GET /api/ps`` gives the context it runs with; below what the instructions and tools need,
  :class:`ContextTooSmall` (Ollama would silently drop the start of the prompt).  If ``/api/show`` cannot be read,
  nothing is sent: a model FlexTool could not classify is never treated as local.  No thread or graphics-card
  limits are sent (P1-O10); ``size_vram < size`` gives the hint that the model runs partly on the processor.
* **LM Studio**: ``GET /api/v1/models`` (context length, loaded), then ``GET /api/v0/models/<model>`` (the
  loaded context length, the ``tool_use`` capability); below need, :class:`ContextTooSmall`; no ``tool_use`` is a
  note only (LM Studio can prompt a model for tools).
* **OpenRouter**: the public model list (``GET <base>/models``), cached for a day in the user cache folder: the
  context length and whether ``tools`` is a supported parameter.
* **Anthropic**: the model's ``max_input_tokens``.
* **Custom endpoint** (``openai_compatible``): the model list (``GET <base>/models``); the model's entry may give
  its context as ``context_length``, ``max_model_len`` or ``context_window`` (vLLM, llama.cpp and others; the
  field differs per server).  A list that cannot be read gives no facts.  Other services: nothing is checked.

Ollama's ``/api/show`` also gives the model's parameter count (``general.parameter_count``, else
``details.parameter_size``), which places a model whose name has no size tag (P3 §3.1).  The facts of every probe
are kept in this process and in ``<user cache>/flextool/assistant/model_facts.json`` with the time they were
found, so the settings show the level and context of a model without asking any server (:func:`cached_facts`).

Local servers are asked through a proxy-less opener (a local endpoint never goes through a proxy, whatever the
environment says), with a 2 second timeout; no redirect is followed.  Only the standard library is used, apart
from the Anthropic library for the Anthropic check.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

from flextool.assistant import net
from flextool.assistant.endpoint import OLLAMA_CLOUD_HOST
from flextool.assistant.llm_adapter import Connection, OnEvent, ToolSpec, Waiting
from flextool.assistant.llm_errors import (
    ContextTooSmall,
    ErrorContext,
    LLMError,
    ModelNotFound,
    ProviderError,
    TimedOut,
    ToolsUnsupported,
    Unreachable,
    cancelled,
    classify_status,
    make_error,
    tools_message,
)
from flextool.assistant.providers import Preset, is_cloud_name

#: Timeout of a question to a local server, and of the call that loads a model (P1-D9).
LOCAL_TIMEOUT = 2.0
LOAD_TIMEOUT = 900.0
#: Timeout of the remote model list (OpenRouter).
REMOTE_TIMEOUT = net.CONNECT_TIMEOUT
#: How long Ollama keeps the model loaded after the probe's load call.
KEEP_ALIVE = "10m"
#: Seconds between two :class:`Waiting` events while a model loads.
WAIT_INTERVAL = 5.0
#: Tokens kept free for the answer on top of the instructions and tools.
ANSWER_MARGIN = 4_000
#: The OpenRouter model list is fetched again after this many seconds.
MODEL_LIST_MAX_AGE = 24 * 3600.0
MODEL_LIST_CACHE = "openrouter_models.json"
FACTS_CACHE = "model_facts.json"

#: The settings hint for an Ollama model that does not fit in graphics memory (P1-O10).
PARTLY_ON_CPU = "This model doesn't fit fully in graphics memory and will be slower."
#: The note for an LM Studio model that does not list tool use.
LMSTUDIO_NO_TOOL_USE = ("LM Studio does not list tool use for {model}. The assistant reads your case data with tool "
                        "calls, so the answers may not use your data; a model that lists tool use works better.")


@dataclass(frozen=True)
class ModelFacts:
    """What the probe found.  ``None`` means not known; *source* says where the facts came from."""

    context_window: int | None = None       # the model's maximum context
    loaded_context: int | None = None       # the context the local server runs it with
    tools: bool | None = None
    remote_host: str | None = None          # the host a local server forwards the model to (Ollama cloud)
    source: str = "none"
    partly_on_cpu: bool = False
    notes: tuple[str, ...] = ()             # shown to the user, never sent to the model
    parameter_count: int | None = None      # the parameters the local server reports (Ollama)
    listed_context: int | None = None       # the context a custom endpoint's model list gives

    @property
    def context(self) -> int | None:
        """The context the requests have: the loaded context of a local model, else the model's maximum."""
        return self.loaded_context if self.loaded_context is not None else self.context_window


def needed_tokens(system: tuple[str, ...], tools: tuple[ToolSpec, ...]) -> int:
    """About how many tokens the instructions and tools take, plus room for the answer: a quarter of their JSON
    length + :data:`ANSWER_MARGIN`."""
    payload = json.dumps({"system": list(system),
                          "tools": [{"name": t.name, "description": t.description, "parameters": t.input_schema}
                                    for t in tools]}, ensure_ascii=False)
    return len(payload) // 4 + ANSWER_MARGIN


# --- asking a server ----------------------------------------------------------------------------------------------
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 — urllib's signature
        return None


def _opener(proxy: str | None) -> urllib.request.OpenerDirector:
    """An opener that uses *proxy* (``None``: no proxy at all, whatever the environment says) and follows no
    redirect."""
    proxies = {} if proxy is None else {"http": proxy, "https": proxy}
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies), _NoRedirect())


@dataclass
class _Answer:
    status: int
    body: Any                   # decoded JSON, or the text when it is not JSON
    headers: Any


class _Asker:
    """Questions to one server: JSON in, JSON out, cancellable, with the connection's error context."""

    def __init__(self, base: str, conn: Connection, ctx: ErrorContext, cancel: threading.Event,
                 on_event: OnEvent | None, *, proxy: str | None = None) -> None:
        self.base = base.rstrip("/")
        self.conn = conn
        self.ctx = ctx
        self.cancel = cancel
        self.on_event = on_event
        self.opener = _opener(proxy)

    def _headers(self, body: bytes | None) -> dict[str, str]:
        headers = {"User-Agent": net.USER_AGENT, "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.conn.api_key:
            headers["Authorization"] = net.key_header_value(self.conn.api_key)
        return headers

    def _fetch(self, method: str, path: str, payload: dict | None, timeout: float) -> _Answer:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=self._headers(data))
        try:
            with self.opener.open(request, timeout=timeout) as response:
                status, raw, headers = response.status, response.read(), response.headers
        except urllib.error.HTTPError as exc:
            status, raw, headers = exc.code, exc.read() if exc.fp is not None else b"", exc.headers
        text = raw.decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(text) if text.strip() else None
        except ValueError:
            body = text
        return _Answer(status, body, headers)

    def ask(self, method: str, path: str, payload: dict | None = None, *, timeout: float = LOCAL_TIMEOUT,
            waiting: bool = False) -> _Answer:
        """The answer to one request; a status other than 200 raises the classified error.  The request runs
        in a daemon thread, so *cancel* stops the wait at once (the thread is left to finish on its own)."""
        if self.cancel.is_set():
            raise cancelled(self.ctx)
        result: dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                result["answer"] = self._fetch(method, path, payload, timeout)
            except BaseException as exc:  # noqa: BLE001 — handed to the waiting thread
                result["error"] = exc
            finally:
                done.set()

        threading.Thread(target=run, name="flextool-assistant-probe", daemon=True).start()
        started = time.monotonic()
        next_wait = started + WAIT_INTERVAL
        while not done.wait(0.05):
            if self.cancel.is_set():
                raise cancelled(self.ctx)
            now = time.monotonic()
            if waiting and self.on_event is not None and now >= next_wait:
                self.on_event(Waiting(int(now - started)))
                next_wait = now + WAIT_INTERVAL
        if self.cancel.is_set():
            raise cancelled(self.ctx)
        if "error" in result:
            raise self._connection_error(result["error"], timeout)
        answer: _Answer = result["answer"]
        if answer.status != 200:
            raise classify_status(answer.status, answer.body, answer.headers, self.ctx)
        return answer

    def _connection_error(self, exc: BaseException, timeout: float) -> LLMError:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, (socket.timeout, TimeoutError)):
            return make_error(TimedOut, "timeout", self.ctx, message_fields={"seconds": f"{timeout:g}"})
        if isinstance(reason, OSError):
            return make_error(Unreachable, "connection_local" if self.ctx.local else "connection_remote",
                              self.ctx)
        return self.unreadable(f"{type(exc).__name__}")

    def unreadable(self, what: str) -> LLMError:
        return make_error(ProviderError, "probe_unreadable", self.ctx, message_fields={"what": what})

    def ask_object(self, method: str, path: str, payload: dict | None = None, **kwargs: Any) -> dict:
        answer = self.ask(method, path, payload, **kwargs)
        if not isinstance(answer.body, dict):
            raise self.unreadable(f"{method} {path} did not return a JSON object")
        return answer.body


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _without_v1(url: str) -> str:
    url = url.rstrip("/")
    return url[: -len("/v1")] if url.endswith("/v1") else url


def _small_context(key: str, ctx: ErrorContext, n: int, needed: int) -> LLMError:
    return make_error(ContextTooSmall, key, ctx, message_fields={"n": n, "needed": needed})


def window_error(preset: Preset, conn: Connection, err: Exception) -> LLMError:
    """The :class:`ContextTooSmall` error of a model whose context is below the assistant's minimum
    (:class:`flextool.assistant.tiers.WindowTooSmall`): the local server's advice when it runs here."""
    from flextool.assistant.backends._common import error_context
    from flextool.assistant.tiers import MIN_WINDOW

    ctx = error_context(preset, conn)
    window = getattr(err, "window", 0)
    key = {"ollama": "small_context_ollama", "lmstudio": "small_context_lmstudio"}.get(preset.context_probe or "")
    if key is None or not getattr(err, "local", False):
        key = "small_context"
    return _small_context(key, ctx, window, MIN_WINDOW)


def _tools_unsupported(ctx: ErrorContext) -> LLMError:
    err = make_error(ToolsUnsupported, "tools", ctx)
    err.user_message = tools_message(ctx)
    err.args = (err.user_message,)
    return err


# --- Ollama -------------------------------------------------------------------------------------------------------
def remote_host_of(show: Mapping[str, Any], model: str) -> str | None:
    """The host an Ollama model is forwarded to: the host of ``remote_host`` in the ``/api/show`` answer; for a
    name ending in ``:cloud`` / ``-cloud`` without one, Ollama cloud; else ``None`` (the model runs here)."""
    remote = show.get("remote_host")
    if isinstance(remote, str) and remote.strip():
        remote = remote.strip()
        return (urlsplit(remote if "//" in remote else "//" + remote).hostname or remote).lower()
    if is_cloud_name(model):
        return OLLAMA_CLOUD_HOST
    return None


def _ollama_context_window(show: Mapping[str, Any]) -> int | None:
    info = show.get("model_info")
    if not isinstance(info, dict):
        return None
    arch = info.get("general.architecture")
    if isinstance(arch, str) and _int(info.get(f"{arch}.context_length")):
        return _int(info[f"{arch}.context_length"])
    return next((_int(v) for k, v in info.items() if k.endswith(".context_length") and _int(v)), None)


_PARAMETER_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([KMBT])\s*", re.IGNORECASE)
_SCALE = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def ollama_parameter_count(show: Mapping[str, Any]) -> int | None:
    """The parameter count in an ``/api/show`` answer: ``model_info["general.parameter_count"]``, else
    ``details.parameter_size`` (``"20.9B"``, ``"568M"``); ``None`` when neither is there."""
    info = show.get("model_info")
    if isinstance(info, dict) and _int(info.get("general.parameter_count")):
        return _int(info["general.parameter_count"])
    details = show.get("details")
    size = details.get("parameter_size") if isinstance(details, dict) else None
    if isinstance(size, str):
        match = _PARAMETER_SIZE.fullmatch(size)
        if match:
            return int(float(match.group(1)) * _SCALE[match.group(2).lower()])
    return None


def _same_model(entry: Mapping[str, Any], model: str) -> bool:
    names = {entry.get("name"), entry.get("model")}
    wanted = {model, model if ":" in model else f"{model}:latest"}
    return bool(names & wanted)


def _probe_ollama(conn: Connection, ctx: ErrorContext, cancel: threading.Event, on_event: OnEvent | None,
                  needed: int | None) -> ModelFacts:
    asker = _Asker(_without_v1(conn.base_url), conn, ctx, cancel, on_event)
    show = asker.ask_object("POST", "/api/show", {"model": conn.model})
    capabilities = show.get("capabilities")
    tools = ("tools" in capabilities) if isinstance(capabilities, list) else None
    if tools is False:
        raise _tools_unsupported(ctx)
    context_window = _ollama_context_window(show)
    parameters = ollama_parameter_count(show)
    remote = remote_host_of(show, conn.model)
    if remote is not None:
        return ModelFacts(context_window=context_window, tools=tools, remote_host=remote, source="ollama",
                          parameter_count=parameters)
    # Load the model, so the context it runs with can be read (P1 §7.2).  Nothing but the model and how long to
    # keep it is sent: the server uses the graphics card and its own threads as it sees fit (P1-O10).
    asker.ask("POST", "/api/generate", {"model": conn.model, "keep_alive": KEEP_ALIVE}, timeout=LOAD_TIMEOUT,
              waiting=True)
    ps = asker.ask_object("GET", "/api/ps")
    entries = ps.get("models")
    entry = next((e for e in entries if isinstance(e, dict) and _same_model(e, conn.model)), None) \
        if isinstance(entries, list) else None
    if entry is None:
        raise asker.unreadable(f"{conn.model} was not listed as loaded after loading it")
    loaded = _int(entry.get("context_length"))
    if loaded is None:
        raise asker.unreadable(f"the context length of {conn.model} was not listed")
    if needed is not None and loaded < needed:
        raise _small_context("small_context_ollama", ctx, loaded, needed)
    size, vram = _int(entry.get("size")), entry.get("size_vram")
    partly_on_cpu = size is not None and isinstance(vram, int) and not isinstance(vram, bool) and vram < size
    return ModelFacts(context_window=context_window, loaded_context=loaded, tools=tools, source="ollama",
                      partly_on_cpu=partly_on_cpu, notes=(PARTLY_ON_CPU,) if partly_on_cpu else (),
                      parameter_count=parameters)


# --- LM Studio ----------------------------------------------------------------------------------------------------
def _entry_id(entry: Mapping[str, Any]) -> str | None:
    for key in ("id", "key", "model_key", "modelKey"):
        if isinstance(entry.get(key), str):
            return entry[key]
    return None


def _is_loaded(entry: Mapping[str, Any]) -> bool | None:
    if isinstance(entry.get("loaded"), bool):
        return entry["loaded"]
    if isinstance(entry.get("state"), str):
        return entry["state"] == "loaded"
    if isinstance(entry.get("loaded_instances"), list):
        return bool(entry["loaded_instances"])
    return None


def _probe_lmstudio(conn: Connection, ctx: ErrorContext, cancel: threading.Event, on_event: OnEvent | None,
                    needed: int | None) -> ModelFacts:
    asker = _Asker(_without_v1(conn.base_url), conn, ctx, cancel, on_event)
    context_window: int | None = None
    loaded: bool | None = None
    try:
        listing = asker.ask_object("GET", "/api/v1/models")
    except (ModelNotFound, ProviderError):
        listing = {}                     # an older LM Studio without the v1 list: the v0 answer has it all
    items = listing.get("data", listing.get("models"))
    entry = next((e for e in items if isinstance(e, dict) and _entry_id(e) == conn.model), None) \
        if isinstance(items, list) else None
    if entry is not None:
        context_window = _int(entry.get("context_length")) or _int(entry.get("max_context_length"))
        loaded = _is_loaded(entry)
    details = asker.ask_object("GET", "/api/v0/models/" + quote(conn.model, safe="/"))
    context_window = context_window or _int(details.get("max_context_length"))
    if loaded is None:
        loaded = _is_loaded(details)
    loaded_context = _int(details.get("loaded_context_length")) if loaded is not False else None
    if needed is not None and loaded_context is not None and loaded_context < needed:
        raise _small_context("small_context_lmstudio", ctx, loaded_context, needed)
    capabilities = details.get("capabilities")
    tools = ("tool_use" in capabilities) if isinstance(capabilities, list) else None
    notes = (LMSTUDIO_NO_TOOL_USE.format(model=conn.model),) if tools is False else ()
    return ModelFacts(context_window=context_window, loaded_context=loaded_context, tools=tools, source="lmstudio",
                      notes=notes)


# --- OpenRouter ---------------------------------------------------------------------------------------------------
def cache_root() -> Path:
    """``<user cache dir of flextool>/assistant``."""
    import platformdirs

    return Path(platformdirs.user_cache_dir("flextool")) / "assistant"


def _cached_models(base_url: str, now: float) -> list[dict] | None:
    from flextool.assistant import _files

    try:
        data = _files.read_json(cache_root() / MODEL_LIST_CACHE, check_format=False)
    except (OSError, _files.UnreadableFile):
        return None
    entry = (data or {}).get(net.normalise_endpoint(base_url))
    if not isinstance(entry, dict) or not isinstance(entry.get("models"), list):
        return None
    fetched = entry.get("fetched")
    if not isinstance(fetched, (int, float)) or not 0 <= now - fetched < MODEL_LIST_MAX_AGE:
        return None
    return entry["models"]


def _store_models(base_url: str, models: list[dict], now: float) -> None:
    from flextool.assistant import _files

    path = cache_root() / MODEL_LIST_CACHE
    try:
        with _files.locked(path):
            try:
                data = _files.read_json(path, check_format=False) or {}
            except _files.UnreadableFile:
                data = {}
            data[net.normalise_endpoint(base_url)] = {"fetched": now, "models": models}
            _files.write_json_atomic(path, data)
    except (OSError, _files.LockTimeout):
        return                          # the list is fetched again next time


def _probe_openrouter(conn: Connection, ctx: ErrorContext, cancel: threading.Event,
                      clock: Callable[[], float]) -> ModelFacts:
    now = clock()
    models = _cached_models(conn.base_url, now)
    if models is None:
        asker = _Asker(conn.base_url, conn, ctx, cancel, None, proxy=net.proxy_for(conn.base_url))
        try:
            listing = asker.ask_object("GET", "/models", timeout=REMOTE_TIMEOUT)
        except LLMError as exc:
            if exc.code == "cancelled":
                raise
            return ModelFacts(source="openrouter (the model list could not be fetched)")
        models = [{"id": m.get("id"), "context_length": m.get("context_length"),
                   "supported_parameters": m.get("supported_parameters")}
                  for m in listing.get("data") or [] if isinstance(m, dict)]
        _store_models(conn.base_url, models, now)
    entry = next((m for m in models if m.get("id") == conn.model), None)
    if entry is None:
        return ModelFacts(source="openrouter (the model is not in the model list)")
    parameters = entry.get("supported_parameters")
    tools = ("tools" in parameters) if isinstance(parameters, list) else None
    if tools is False:
        raise _tools_unsupported(ctx)
    return ModelFacts(context_window=_int(entry.get("context_length")), tools=tools, source="openrouter")


# --- a custom endpoint -----------------------------------------------------------------------------------------
#: The fields a custom endpoint's model list may give the context in, tried in order (P3 §3.3).
LISTED_CONTEXT_FIELDS = ("context_length", "max_model_len", "context_window")


def _probe_openai_compatible(conn: Connection, ctx: ErrorContext, cancel: threading.Event) -> ModelFacts:
    proxy = None if ctx.local else net.proxy_for(conn.base_url)
    asker = _Asker(conn.base_url, conn, ctx, cancel, None, proxy=proxy)
    try:
        listing = asker.ask_object("GET", "/models", timeout=LOCAL_TIMEOUT if ctx.local else REMOTE_TIMEOUT)
    except LLMError as exc:
        if exc.code == "cancelled":
            raise
        return ModelFacts(source="openai_compatible (the model list could not be read)")
    items = listing.get("data", listing.get("models"))
    entry = next((m for m in items if isinstance(m, dict) and (m.get("id") or m.get("name")) == conn.model), None) \
        if isinstance(items, list) else None
    if entry is None:
        return ModelFacts(source="openai_compatible (the model is not in the model list)")
    listed = next((_int(entry.get(name)) for name in LISTED_CONTEXT_FIELDS if _int(entry.get(name))), None)
    return ModelFacts(listed_context=listed, source="openai_compatible")


# --- the probe ----------------------------------------------------------------------------------------------------
def probe(preset: Preset, conn: Connection, cancel: threading.Event, *, needed: int | None = None,
          on_event: OnEvent | None = None, backend: Any = None,
          clock: Callable[[], float] = time.time) -> ModelFacts:
    """The facts of the model of *conn*.  *needed* is the context the instructions and tools need
    (:func:`needed_tokens`; ``None``: not checked).  *backend* is the Anthropic backend to ask for the model's
    input limit (one is built and closed when ``None``).

    Raises, and nothing is sent to the model: :class:`ToolsUnsupported`; :class:`ContextTooSmall` for a local
    model loaded with too small a context; for a local server that does not answer or whose answer cannot be read,
    :class:`Unreachable`, :class:`TimedOut`, :class:`ModelNotFound` or :class:`ProviderError`; :class:`NoKey`
    when the Anthropic check has no key; :class:`Cancelled`.
    """
    from flextool.assistant.backends._common import error_context

    ctx = error_context(preset, conn)
    if preset.context_probe == "ollama":
        facts = _probe_ollama(conn, ctx, cancel, on_event, needed)
    elif preset.context_probe == "lmstudio":
        facts = _probe_lmstudio(conn, ctx, cancel, on_event, needed)
    elif preset.context_probe == "openrouter":
        facts = _probe_openrouter(conn, ctx, cancel, clock)
    elif preset.context_probe == "anthropic":
        facts = _probe_anthropic(preset, conn, cancel, backend)
    elif preset.id == "openai_compatible":
        facts = _probe_openai_compatible(conn, ctx, cancel)
    else:
        facts = ModelFacts()
    remember_facts(preset.id, conn.base_url, conn.model, facts)
    return facts


# --- the facts of the last probes, for the settings (which never probe) -------------------------------------------
_RECENT: dict[tuple[str, str, str], ModelFacts] = {}
_RECENT_LOCK = threading.Lock()


def _recent_key(preset_id: str, base_url: str, model: str) -> tuple[str, str, str]:
    return preset_id, net.normalise_endpoint(base_url), model


_STORED_FIELDS = ("context_window", "loaded_context", "tools", "remote_host", "source", "partly_on_cpu",
                  "parameter_count", "listed_context")


def remember_facts(preset_id: str, base_url: str, model: str, facts: ModelFacts, *,
                   clock: Callable[[], float] = time.time) -> None:
    """Keep *facts* as the last probe's of *model* at *base_url*: in this process, and in the user cache with the
    time (a file that cannot be written is skipped; the next probe tries again)."""
    try:
        key = _recent_key(preset_id, base_url, model)
    except ValueError:
        return
    with _RECENT_LOCK:
        _RECENT[key] = facts
    _store_facts("|".join(key), facts, clock())


def _store_facts(key: str, facts: ModelFacts, now: float) -> None:
    from flextool.assistant import _files

    path = cache_root() / FACTS_CACHE
    try:
        with _files.locked(path):
            try:
                data = _files.read_json(path, check_format=False) or {}
            except _files.UnreadableFile:
                data = {}
            data[key] = {"found": now, **{name: getattr(facts, name) for name in _STORED_FIELDS}}
            _files.write_json_atomic(path, data)
    except (OSError, _files.LockTimeout):
        return


def _stored_facts(key: str) -> ModelFacts | None:
    from flextool.assistant import _files

    try:
        data = _files.read_json(cache_root() / FACTS_CACHE, check_format=False)
    except (OSError, _files.UnreadableFile):
        return None
    entry = (data or {}).get(key)
    if not isinstance(entry, dict):
        return None
    values: dict[str, Any] = {}
    for name in _STORED_FIELDS:
        value = entry.get(name)
        if name in ("source", "remote_host"):
            values[name] = value if isinstance(value, str) else (None if name == "remote_host" else "cache")
        elif name in ("tools",):
            values[name] = value if isinstance(value, bool) else None
        elif name == "partly_on_cpu":
            values[name] = value is True
        else:
            values[name] = _int(value)
    return ModelFacts(**values)


def cached_facts(preset_id: str, base_url: str, model: str, *,
                 clock: Callable[[], float] = time.time) -> ModelFacts | None:
    """The facts FlexTool last found for *model* at *base_url* without asking any server (P3 §3.3, for the
    settings): this process's last probe, else the user cache, else — for OpenRouter — the model's entry in the
    day's cached model list; ``None`` when there are none."""
    try:
        key = _recent_key(preset_id, base_url, model)
    except ValueError:
        return None
    with _RECENT_LOCK:
        recent = _RECENT.get(key)
    if recent is not None:
        return recent
    stored = _stored_facts("|".join(key))
    if stored is not None:
        return stored
    if preset_id == "openrouter":
        models = _cached_models(base_url, clock())
        entry = next((m for m in models or [] if m.get("id") == model), None)
        if entry is not None and _int(entry.get("context_length")):
            return ModelFacts(context_window=_int(entry["context_length"]), source="openrouter (cached model list)")
    return None


def recent_facts(preset_id: str, base_url: str, model: str) -> ModelFacts | None:
    """The facts the last probe in this process found for *model* at *base_url*, ``None`` when none ran.  The
    settings show its hints (a model partly on the processor, P1-O10) without asking any server."""
    try:
        key = _recent_key(preset_id, base_url, model)
    except ValueError:
        return None
    with _RECENT_LOCK:
        return _RECENT.get(key)


def _probe_anthropic(preset: Preset, conn: Connection, cancel: threading.Event, backend: Any) -> ModelFacts:
    if cancel.is_set():
        raise cancelled()
    own = backend is None
    if own:
        from flextool.assistant.llm_adapter import make_backend

        backend = make_backend(preset, conn)
    try:
        window = backend.context_window()
    except LLMError as exc:
        if exc.code == "cancelled":
            raise
        return ModelFacts(source="anthropic (the model's limits could not be read)")
    finally:
        if own:
            backend.close()
    return ModelFacts(context_window=window, source="anthropic")
