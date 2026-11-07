"""Where the data of a chat goes, and the line that tells the user (P1 §9.1, §9.2).

:func:`route_for` computes the :class:`~flextool.assistant.llm_adapter.Route` of a connection once: the endpoint
kind from the literal host of the base URL (never a name lookup), the proxy a remote endpoint goes through
(:func:`flextool.assistant.net.proxy_for`), and — for Ollama — the host the probe found the model is forwarded
to: an Ollama cloud model (or one a Modelfile forwards elsewhere) is **remote**, labelled "Ollama cloud" (or
"Ollama, forwarded to {host}").  :func:`indicator_text` is the line every front end shows.
:func:`route_from_settings` gives the route of the saved settings without asking any server (the FlexTool window
reads it when it opens); there an Ollama model counts as forwarded by its name only, and the probe before the first
question settles it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from flextool.assistant.llm_adapter import Route
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
