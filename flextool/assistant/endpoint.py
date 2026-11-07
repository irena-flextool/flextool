"""Where the data of a chat goes, and the line that tells the user (P1 §9.1, §9.2).

:func:`route_for` computes the :class:`~flextool.assistant.llm_adapter.Route` of a connection once: the endpoint
kind from the literal host of the base URL (never a name lookup), the proxy a remote endpoint goes through
(:func:`flextool.assistant.net.proxy_for`), and — for Ollama — the host the probe found the model is forwarded
to: an Ollama cloud model (or one a Modelfile forwards elsewhere) is **remote**, labelled "Ollama cloud" (or
"Ollama, forwarded to {host}").  :func:`indicator_text` is the line every front end shows.
:func:`route_from_settings` gives the route of the saved settings without asking any server (the FlexTool window
reads it when it opens); there an Ollama model counts as forwarded by its name only, and the probe before the first
question settles it.

**The data notice and the acknowledgements** (P1 §9.3).  Before the first question goes to a remote route (a
service on the internet, a model a local server forwards, such as an Ollama cloud model), the user accepts
:func:`consent_notice`.  The acceptance is kept in ``<config>/assistant/acknowledgements.json`` under its
:class:`ConsentKey` — preset, normalised endpoint, proxy host, forwarded host and the hash of the notice's
wording — so another endpoint, a new or changed proxy, a cloud model or a changed notice asks again.  An endpoint
on this computer or the local network needs no acceptance (the indicator says where the data goes).  The file
also records when the welcome was first shown (:func:`welcome_seen`, :func:`mark_welcome_seen`).  Every
read-modify-write runs under the file's lock: several chats and front ends write it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flextool.assistant import _files
from flextool.assistant.llm_adapter import Route
from flextool.assistant.net import normalise_endpoint
from flextool.assistant.net import route_for as _route_for
from flextool.assistant.providers import PRESETS, Preset, is_cloud_name

if TYPE_CHECKING:
    from flextool.assistant.probes import ModelFacts
    from flextool.assistant.settings import Settings

OLLAMA_CLOUD = "Ollama cloud"
OLLAMA_CLOUD_HOST = "ollama.com"

INDICATOR_LOOPBACK = ("Endpoint on this computer ({host}). FlexTool sends data only there; if that server forwards "
                      "requests, data leaves this computer.")
INDICATOR_LOCAL_NETWORK = "Endpoint on your local network ({host})."
INDICATOR_REMOTE = "Data goes to: {provider} ({host})."
INDICATOR_PROXY = " It is sent through the proxy {proxy_host}."
INDICATOR_OLLAMA_CLOUD = "Data goes to: Ollama cloud ({forwarded_host}), forwarded by Ollama on this computer."
INDICATOR_FORWARDED = "Data goes to: {forwarded_host}, forwarded by {server} on this computer."


def forwarded_label(preset: Preset, forwarded_to: str) -> str:
    """The label of a model a local server forwards to *forwarded_to*."""
    if forwarded_to == OLLAMA_CLOUD_HOST or forwarded_to.endswith("." + OLLAMA_CLOUD_HOST):
        return OLLAMA_CLOUD
    return f"{preset.label}, forwarded to {forwarded_to}"


def route_for(preset: Preset, base_url: str, *, facts: ModelFacts | None = None, model: str | None = None) -> Route:
    """The route of a connection of *preset* to *base_url*.  For Ollama the forwarded host comes from the probe's
    *facts*, or — without facts — from the *model* name (``:cloud`` / ``-cloud``)."""
    forwarded = None
    if preset.id == "ollama":
        if facts is not None:
            forwarded = facts.remote_host
        elif model and is_cloud_name(model):
            forwarded = OLLAMA_CLOUD_HOST
    label = forwarded_label(preset, forwarded) if forwarded else preset.label
    return _route_for(base_url, label, forwarded_to=forwarded)


def route_from_settings(settings: Settings) -> Route | None:
    """The route of the saved provider settings, ``None`` when no provider (or no address) is set."""
    provider = settings.provider
    preset = PRESETS.get(provider.preset or "")
    if preset is None:
        return None
    base_url = provider.base_url or preset.base_url
    if not base_url:
        return None
    return route_for(preset, base_url, model=provider.model)


def indicator_text(route: Route) -> str:
    """The line that says where the data goes."""
    if route.forwarded_to:
        if route.label == OLLAMA_CLOUD:
            return INDICATOR_OLLAMA_CLOUD.format(forwarded_host=route.forwarded_to)
        server = route.label.split(",")[0]
        return INDICATOR_FORWARDED.format(forwarded_host=route.forwarded_to, server=server)
    if route.kind == "loopback":
        return INDICATOR_LOOPBACK.format(host=route.host)
    if route.kind == "local_network":
        return INDICATOR_LOCAL_NETWORK.format(host=route.host)
    text = INDICATOR_REMOTE.format(provider=route.label, host=route.host)
    if route.proxy_host:
        text += INDICATOR_PROXY.format(proxy_host=route.proxy_host)
    return text


# --- the data notice ----------------------------------------------------------------------------------------------
#: The data notice (outline §8, owner's draft; "e.g." written out, plain text without emphasis).
NOTICE = (
    "Your questions, the case data the assistant reads (input parameters, results, logs) and the names of your "
    "project and scenarios are sent to {provider} at {endpoint}. FlexTool cannot guarantee how this data is "
    "stored or used \u2014 it may be retained, reviewed, or used for training depending on the provider, your "
    "account type and your plan. Free tiers may allow training on your data. Check the provider's current "
    "policies yourself. The only reliable way to keep data on this computer is a local model (for example Ollama "
    "or LM Studio), which requires considerable local compute."
)
NOTICE_PROXY = " Your network settings send this data through the proxy {proxy_host}, which can also see it."
NOTICE_HEADERS = " {provider} requires FlexTool to send these request headers: {headers}."
#: The hash of the notice's wording: a changed notice is asked again.
NOTICE_HASH = hashlib.sha256((NOTICE + NOTICE_PROXY + NOTICE_HEADERS).encode("utf-8")).hexdigest()[:12]

ACKS_FILE = "acknowledgements.json"


def consent_notice(route: Route, *, required_headers: frozenset[str] | tuple[str, ...] = ()) -> str:
    """The data notice of *route*: the service and the host the data goes to (the forwarded host of a model a
    local server forwards), the proxy sentence when a proxy applies, and the headers a preset requires."""
    endpoint = route.forwarded_to or route.host
    text = NOTICE.format(provider=route.label, endpoint=endpoint)
    if route.proxy_host:
        text += NOTICE_PROXY.format(proxy_host=route.proxy_host)
    if required_headers:
        text += NOTICE_HEADERS.format(provider=route.label, headers=", ".join(sorted(required_headers)))
    return text


@dataclass(frozen=True)
class ConsentKey:
    """What an acceptance of the data notice is kept under.  *remote* is not part of the match: it says whether
    the route needs an acceptance at all."""

    preset: str
    endpoint: str
    proxy: str | None
    forwarded_to: str | None
    notice: str
    remote: bool

    def to_json(self, at: str) -> dict[str, Any]:
        return {"preset": self.preset, "endpoint": self.endpoint, "proxy": self.proxy,
                "forwarded_to": self.forwarded_to, "notice": self.notice, "at": at}

    def matches(self, ack: object) -> bool:
        return isinstance(ack, dict) and all(ack.get(name) == value for name, value in (
            ("preset", self.preset), ("endpoint", self.endpoint), ("proxy", self.proxy),
            ("forwarded_to", self.forwarded_to), ("notice", self.notice)))


def consent_key(preset_id: str, base_url: str, route: Route) -> ConsentKey:
    """The key of the data notice of *route* (the connection of *preset_id* to *base_url*)."""
    return ConsentKey(preset=preset_id, endpoint=normalise_endpoint(base_url), proxy=route.proxy_host,
                      forwarded_to=route.forwarded_to, notice=NOTICE_HASH, remote=route.kind == "remote")


def acks_path(config_root: Path | None = None) -> Path:
    return (config_root or _files.config_root()) / ACKS_FILE


def _read_acks(path: Path) -> dict[str, Any]:
    """The acknowledgements file; a damaged one reads as empty (every notice is asked again)."""
    try:
        return _files.read_json(path) or {}
    except (OSError, _files.UnreadableFile):
        return {}


def _change_acks(config_root: Path | None, change) -> None:
    """Read, change and write the acknowledgements under the file's lock.  A file of a newer format is not
    changed (:class:`~flextool.assistant._files.NewerFormat`)."""
    path = acks_path(config_root)
    with _files.locked(path):
        data = _read_acks(path)
        if _files.is_newer(data):
            raise _files.NewerFormat(path, data.get("format"))
        data["format"] = _files.FORMAT
        change(data)
        _files.write_json_atomic(path, data)


def needs_consent(key: ConsentKey, config_root: Path | None = None) -> bool:
    """Whether the data notice must be accepted before the first question goes on this route: a remote route with
    no acceptance under the same key."""
    if not key.remote:
        return False
    acks = _read_acks(acks_path(config_root)).get("acks")
    return not (isinstance(acks, list) and any(key.matches(a) for a in acks))


def record_consent(key: ConsentKey, config_root: Path | None = None) -> None:
    """Keep the user's acceptance of the notice under *key* (a lasting one: it is not asked again for the same
    key)."""
    def change(data: dict[str, Any]) -> None:
        acks = data.get("acks") if isinstance(data.get("acks"), list) else []
        acks = [a for a in acks if not key.matches(a)]
        acks.append(key.to_json(_files.iso(_files.utc_now())))
        data["acks"] = acks
    _change_acks(config_root, change)


def welcome_seen(config_root: Path | None = None) -> bool:
    """Whether the welcome has been shown (in the terminal or the FlexTool window)."""
    return bool(_read_acks(acks_path(config_root)).get("welcome_seen"))


def mark_welcome_seen(config_root: Path | None = None) -> None:
    """Record that the welcome has been shown (the first time only)."""
    def change(data: dict[str, Any]) -> None:
        data.setdefault("welcome_seen", _files.iso(_files.utc_now()))
    _change_acks(config_root, change)
