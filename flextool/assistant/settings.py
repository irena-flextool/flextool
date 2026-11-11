"""The assistant's settings (P1 §10.1): ``<config>/assistant/settings.json``.

```
{"format": 1,
 "provider": {"preset", "base_url", "model", "tier": null, "knobs": {"openrouter.data_collection": "deny", …},
              "key_storage": "keyring"},
 "conversation_expiry_days": 30, "compaction_every": null, "keep_recent_turns": null, "parallel_requests": 1,
 "free_tier": null, "free_tier_suggested": []}
```

``compaction_every`` / ``keep_recent_turns``: ``null`` = automatic, from the model's context (P3 §3.3, §4.5); a
number overrides it.  The values P1's ``/save`` wrote by default (6 and 2) read as automatic.  ``free_tier``:
``null`` = automatic (on only for an OpenRouter ``:free`` model), ``true`` / ``false`` = the user's choice
(P3 §4.6).  ``free_tier_suggested`` lists the routes (usage-ledger keys) where the free-plan hint was shown once.

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
#: The values P1 wrote by default; stored, they read as automatic (P3 §4.5).
P1_COMPACTION_EVERY = 6
P1_KEEP_RECENT_TURNS = 2
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
    compaction_every: int | None = None                     # None: from the model's context
    keep_recent_turns: int | None = None                    # None: from the model's context
    parallel_requests: int = DEFAULT_PARALLEL_REQUESTS     # requests at once to one provider (window chats)
    free_tier: bool | None = None                           # None: on only for an OpenRouter :free model
    free_tier_suggested: list[str] = field(default_factory=list)    # routes where the free-plan hint was shown
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
            "free_tier": self.free_tier,
            "free_tier_suggested": list(self.free_tier_suggested),
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


def _automatic_int(raw: dict[str, Any], key: str, p1_default: int, low: int, notes: list[str]) -> int | None:
    """A number that overrides an automatic value: ``None`` when absent, ``null`` or P1's stored default."""
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        notes.append(f"The setting {key} must be a whole number {low} or more, or null; it was {value!r}, so it is "
                     "chosen from the model's context.")
        return None
    return None if value == p1_default else value


def _free_tier(raw: dict[str, Any], notes: list[str]) -> bool | None:
    value = raw.get("free_tier")
    if value is None or isinstance(value, bool):
        return value
    notes.append(f"The setting free_tier must be true, false or null; it was {value!r}, so it is automatic.")
    return None


def _suggested(raw: dict[str, Any], notes: list[str]) -> list[str]:
    value = raw.get("free_tier_suggested", [])
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    notes.append("The setting free_tier_suggested must be a list of texts; it is read as empty.")
    return []


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
        compaction_every=_automatic_int(raw, "compaction_every", P1_COMPACTION_EVERY, 1, notes),
        keep_recent_turns=_automatic_int(raw, "keep_recent_turns", P1_KEEP_RECENT_TURNS, 0, notes),
        parallel_requests=_int(raw, "parallel_requests", DEFAULT_PARALLEL_REQUESTS, 1, MAX_PARALLEL_REQUESTS, notes),
        free_tier=_free_tier(raw, notes),
        free_tier_suggested=_suggested(raw, notes),
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


def mark_free_tier_suggested(route_key: str, config_root: Path | None = None) -> bool:
    """Remember that the free-plan hint was shown for *route_key*; ``True`` when it was not shown there before
    (the hint is shown once per route, P3 §4.6).  A file that cannot be written (a newer format, a lock held too
    long) answers ``False``: no hint rather than one at every rate limit."""
    path = settings_path(config_root)
    try:
        with _files.locked(path):
            try:
                current = _files.read_json(path) or {}
            except _files.UnreadableFile:
                return False
            if _files.is_newer(current):
                return False
            shown = current.get("free_tier_suggested")
            shown = list(shown) if isinstance(shown, list) else []
            if route_key in shown:
                return False
            current = copy.deepcopy(current)
            current.setdefault("format", _files.FORMAT)
            current["free_tier_suggested"] = [*shown, route_key]
            _files.write_json_atomic(path, current)
            return True
    except (OSError, _files.LockTimeout):
        return False


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
        shown = current.get("free_tier_suggested")
        if isinstance(shown, list):         # a hint shown since these settings were read stays shown
            new["free_tier_suggested"] = list(dict.fromkeys([*new["free_tier_suggested"], *shown]))
        merged.update(new)
        merged["provider"] = provider
        _files.write_json_atomic(path, merged)
