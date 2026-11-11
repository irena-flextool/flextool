"""What both backends share: the error context of a connection, the key sent, merging streamed provider data."""
from __future__ import annotations

import contextlib
import copy
import threading
from typing import Any
from urllib.parse import urlsplit

from flextool.assistant.llm_adapter import Connection
from flextool.assistant.llm_errors import ErrorContext, NoKey, make_error
from flextool.assistant.net import NO_KEY, is_local, read_timeout_for
from flextool.assistant.providers import Preset, knob_on


def error_context(preset: Preset, conn: Connection) -> ErrorContext:
    parts = urlsplit(conn.base_url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    zdr_knob = preset.knob("zdr")
    return ErrorContext(
        preset_id=preset.id, provider=conn.route.label, model=conn.model, key_source=conn.key_source,
        host=parts.hostname or "", port=port, local=is_local(conn.base_url), server_name=preset.server_name,
        key_env=preset.key_env, zdr=bool(zdr_knob and knob_on(preset, zdr_knob, conn.knobs)),
        read_timeout=read_timeout_for(conn.base_url), secret=conn.api_key)


def api_key_for(preset: Preset, conn: Connection, ctx: ErrorContext) -> str:
    """The key the requests carry: the resolved one, or :data:`~flextool.assistant.net.NO_KEY` for an endpoint
    that needs none (never empty, so no library falls back to a key of its own).  A missing required key raises
    :class:`NoKey`."""
    if conn.api_key:
        return conn.api_key
    if preset.key_required:
        raise make_error(NoKey, "no_key", ctx)
    return NO_KEY


def deep_merge(target: dict, extra: dict) -> dict:
    """Merge *extra* into *target*: nested objects merged, any other value replaced by the later one."""
    for key, value in extra.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            deep_merge(current, value)
        else:
            target[key] = copy.deepcopy(value)
    return target


class InFlight:
    """The response a backend is reading, so another thread can close it (cancel)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._response: Any = None
        self.aborted = threading.Event()

    def begin(self) -> None:
        self.aborted.clear()
        with self._lock:
            self._response = None

    def set(self, response: Any) -> None:
        with self._lock:
            self._response = response

    def clear(self) -> None:
        with self._lock:
            self._response = None

    def abort(self) -> None:
        self.aborted.set()
        with self._lock:
            response = self._response
        if response is not None:
            # Closing a response another thread is reading may fail; the socket shutdown that follows stops the
            # read anyway.
            with contextlib.suppress(Exception):
                response.close()
