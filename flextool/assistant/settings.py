"""The assistant's settings (P1 §10.1): ``<config>/assistant/settings.json``.

```
{"format": 1,
 "provider": {"preset", "base_url", "model", "tier": null, "knobs": {"openrouter.data_collection": "deny", …},
              "key_storage": "keyring"},
 "conversation_expiry_days": 30, "compaction_every": 6, "keep_recent_turns": 2, "parallel_requests": 1}
```

Command-line flags override the settings for one session; ``/save`` (and the window's settings dialog) writes them
with :func:`save_settings`, a read-modify-write under the file's lock that keeps keys this FlexTool does not know.
A value that is missing or out of range reads as its default, with a note (:attr:`Settings.notes`).  A file of a
newer format is read but not written (:attr:`Settings.read_only`).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from flextool.assistant import _files

FILE_NAME = "settings.json"

DEFAULT_EXPIRY_DAYS = 30
DEFAULT_COMPACTION_EVERY = 6
DEFAULT_KEEP_RECENT_TURNS = 2
DEFAULT_PARALLEL_REQUESTS = 1
MAX_PARALLEL_REQUESTS = 4
DEFAULT_KNOBS: dict[str, object] = {"openrouter.data_collection": "deny", "openrouter.zdr": False}
KeyStorage = Literal["keyring", "file"]
TIERS = ("basic", "standard", "advanced")


@dataclass
class ProviderSettings:
    preset: str | None = None
    base_url: str | None = None
    model: str | None = None
    tier: str | None = None                     # None: from the model name
    knobs: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_KNOBS))
    key_storage: KeyStorage = "keyring"


@dataclass
class Settings:
    provider: ProviderSettings = field(default_factory=ProviderSettings)
    conversation_expiry_days: int = DEFAULT_EXPIRY_DAYS     # 0: conversations are not saved
    compaction_every: int = DEFAULT_COMPACTION_EVERY
    keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS
    parallel_requests: int = DEFAULT_PARALLEL_REQUESTS     # requests at once to one provider (window chats)
    read_only: bool = False                                 # a newer FlexTool wrote the file
    notes: list[str] = field(default_factory=list)          # values read as their default, and why

    def to_json(self) -> dict[str, Any]:
        p = self.provider
        return {
            "format": _files.FORMAT,
            "provider": {"preset": p.preset, "base_url": p.base_url, "model": p.model, "tier": p.tier,
                         "knobs": dict(p.knobs), "key_storage": p.key_storage},
            "conversation_expiry_days": self.conversation_expiry_days,
            "compaction_every": self.compaction_every,
            "keep_recent_turns": self.keep_recent_turns,
            "parallel_requests": self.parallel_requests,
        }


def settings_path(config_root: Path | None = None) -> Path:
    return (config_root or _files.config_root()) / FILE_NAME


def _int(raw: dict[str, Any], key: str, default: int, low: int, high: int | None, notes: list[str]) -> int:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        limit = f"from {low} to {high}" if high is not None else f"{low} or more"
        notes.append(f"The setting {key} must be a whole number {limit}; it was {value!r}, so {default} is used.")
        return default
    return value


def _text(raw: dict[str, Any], key: str, notes: list[str]) -> str | None:
    value = raw.get(key)
    if value is None or isinstance(value, str):
        return value or None
    notes.append(f"The setting provider.{key} must be text; it was {value!r}, so it is not used.")
    return None


def from_json(raw: dict[str, Any]) -> Settings:
    """The settings in *raw* (the file's object); every wrong value reads as its default with a note."""
    notes: list[str] = []
    prov_raw = raw.get("provider")
    if prov_raw is None:
        prov_raw = {}
    elif not isinstance(prov_raw, dict):
        notes.append("The setting provider must be an object; the provider is not set.")
        prov_raw = {}
    knobs_raw = prov_raw.get("knobs", {})
    knobs = dict(DEFAULT_KNOBS)
    if isinstance(knobs_raw, dict):
        knobs.update({str(k): v for k, v in knobs_raw.items()})
    else:
        notes.append("The setting provider.knobs must be an object; the default privacy settings are used.")
    tier = _text(prov_raw, "tier", notes)
    if tier is not None and tier not in TIERS:
        notes.append(f"The setting provider.tier must be one of {', '.join(TIERS)}; it was {tier!r}, so the "
                     "level comes from the model name.")
        tier = None
    storage = prov_raw.get("key_storage", "keyring")
    if storage not in ("keyring", "file"):
        notes.append(f"The setting provider.key_storage must be keyring or file; it was {storage!r}, so keyring "
                     "is used.")
        storage = "keyring"
    provider = ProviderSettings(preset=_text(prov_raw, "preset", notes), base_url=_text(prov_raw, "base_url", notes),
                                model=_text(prov_raw, "model", notes), tier=tier, knobs=knobs, key_storage=storage)
    return Settings(
        provider=provider,
        conversation_expiry_days=_int(raw, "conversation_expiry_days", DEFAULT_EXPIRY_DAYS, 0, None, notes),
        compaction_every=_int(raw, "compaction_every", DEFAULT_COMPACTION_EVERY, 1, None, notes),
        keep_recent_turns=_int(raw, "keep_recent_turns", DEFAULT_KEEP_RECENT_TURNS, 0, None, notes),
        parallel_requests=_int(raw, "parallel_requests", DEFAULT_PARALLEL_REQUESTS, 1, MAX_PARALLEL_REQUESTS, notes),
        read_only=_files.is_newer(raw),
        notes=notes,
    )


def load_settings(config_root: Path | None = None) -> Settings:
    """The saved settings, or the defaults when there are none.  A damaged file reads as the defaults with a
    note, and is not overwritten until the user saves."""
    path = settings_path(config_root)
    try:
        raw = _files.read_json(path)
    except _files.UnreadableFile as exc:
        settings = Settings()
        settings.notes.append(f"{exc} The default settings are used.")
        return settings
    if raw is None:
        return Settings()
    settings = from_json(raw)
    if settings.read_only:
        settings.notes.append(str(_files.NewerFormat(path, raw.get("format"))))
    return settings


def expiry_days(config_root: Path | None = None) -> int:
    """The conversation expiry in days (0: conversations are not saved)."""
    return load_settings(config_root).conversation_expiry_days


def save_settings(settings: Settings, config_root: Path | None = None) -> None:
    """Write *settings*: under the file's lock, over the current file, keeping the keys this FlexTool does not
    know.  Raises :class:`~flextool.assistant._files.NewerFormat` for a file of a newer format and
    :class:`~flextool.assistant._files.LockTimeout` when another process holds the lock too long."""
    path = settings_path(config_root)
    with _files.locked(path):
        try:
            current = _files.read_json(path) or {}
        except _files.UnreadableFile:
            current = {}
        if _files.is_newer(current):
            raise _files.NewerFormat(path, current.get("format"))
        merged = copy.deepcopy(current)
        new = settings.to_json()
        provider = merged.get("provider") if isinstance(merged.get("provider"), dict) else {}
        provider.update(new.pop("provider"))
        merged.update(new)
        merged["provider"] = provider
        _files.write_json_atomic(path, merged)
