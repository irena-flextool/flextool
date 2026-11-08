"""The keys of the AI services (P1 §8.1): where FlexTool finds one, and where it stores one.

A key is looked for in this order (Q2): the preset's environment variable (``GEMINI_API_KEY`` …), the system
keyring (service ``flextool-assistant``), the key file.  The environment wins, and every front end says where the
key came from ("from the environment variable GEMINI_API_KEY").

* **Environment**: :func:`read_key_env` is the only function under ``flextool/assistant/`` that reads the
  environment, and only by a variable name a preset gives (``tests/test_env_knob_registry.py`` checks both).
* **Keyring**: every call runs in a daemon thread that is given up after 5 seconds, because a stuck D-Bus or
  Keychain call never returns.  A keyring that cannot keep a key safely is not used: the fail and null backends
  (a Linux computer without a Secret Service gives the fail backend), any ``keyrings.alt`` backend and any backend
  that stores keys unencrypted (for example one chosen with ``PYTHON_KEYRING_BACKEND``), a priority below 1, a
  chain without usable backends.
* **Key file**: only when the user chooses it after the warning (:func:`key_file_warning`):
  ``<config>/assistant/keys.json``, created readable by the owner only (0600 on POSIX) before the key is written.
  Never in a project folder.

Keys stored for a service with a fixed address are kept under the preset id; an optional key of a local server or
a custom endpoint is kept per endpoint (``"<preset>:<normalised address>"``).  A key never appears in a message.

What is saved about a key elsewhere (the "not available to your account" marks of the model list) is only its
fingerprint (:func:`key_fingerprint`): a one-way HMAC-SHA-256 of the key with a random salt of this installation
(``<config>/assistant/key_salt.json``), cut to 12 hex digits, so a mark can tell that the key changed without
holding anything the key can be read from.
``keyring`` is imported only here, inside the functions that use it.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import threading
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from flextool.assistant import _files
from flextool.assistant.net import normalise_endpoint

if TYPE_CHECKING:
    from flextool.assistant.providers import Preset

SERVICE = "flextool-assistant"
KEY_FILE = "keys.json"
SALT_FILE = "key_salt.json"
#: The fingerprint of "no key" (a local server, an optional key not set).
NO_KEY = "none"
FINGERPRINT_LENGTH = 12
KEYRING_TIMEOUT = 5.0

KEYRING_SLOW = "The system keyring did not answer within {seconds} seconds."
KEYRING_PLAINTEXT = ("The keyring setting points to a backend that stores keys unencrypted; FlexTool does not use "
                     "it. Use --key-file if you accept an unencrypted file.")
KEYRING_NONE = ("There is no usable system keyring on this computer ({backend}). Set the environment variable for "
                "the key, or use --key-file if you accept an unencrypted file.")
KEYRING_FAILED = "The system keyring could not {action} the key: {reason}"
KEY_FILE_WARNING = ("The key will be stored unencrypted at {path}. Anyone who can read your files can read it. "
                    "Store it there? [y/N]")

#: The terminal's advice in the keyring texts, and the settings window's in its place.
_WINDOW_ADVICE = (
    ("Use --key-file if you accept an unencrypted file.",
     "Choose a file on this computer if you accept an unencrypted file."),
    ("or use --key-file if you accept an unencrypted file.",
     "or choose a file on this computer if you accept an unencrypted file."),
)

T = TypeVar("T")


def for_window(text: str) -> str:
    """A keyring text of this module with the settings window's advice in place of the terminal option."""
    for terminal, window in _WINDOW_ADVICE:
        text = text.replace(terminal, window)
    return text


class KeySource(Enum):
    ENV = "environment variable {name}"
    KEYRING = "system keyring"
    FILE = "key file"
    NONE = "none"

    def describe(self, preset: Preset | None = None) -> str:
        """Where a key came from, for "from {…}": ``the environment variable GEMINI_API_KEY``."""
        if self is KeySource.ENV:
            return "the " + self.value.format(name=preset.key_env if preset is not None else "")
        if self is KeySource.NONE:
            return "nowhere"
        return "the " + self.value


class KeyringUnavailable(Exception):
    """The keyring cannot be used here; the message says why."""


class KeyringTimeout(KeyringUnavailable):
    pass


class KeyStoreError(Exception):
    """A key could not be stored; the message says why and what to do."""


# --- the environment ---------------------------------------------------------------------------------------------
def read_key_env(name: str | None) -> str | None:
    """The value of the environment variable *name* (a preset's ``key_env``), stripped; ``None`` when it is not set
    or empty.  The only reader of the environment in the assistant."""
    if not name:
        return None
    value = os.environ.get(name, "").strip()
    return value or None


# --- where a key is kept -----------------------------------------------------------------------------------------
def key_account(preset: Preset, endpoint: str | None = None) -> str:
    """The keyring user name (and key file entry) of the key of *preset* at *endpoint*."""
    if preset.key_optional and (endpoint or preset.base_url):
        return f"{preset.id}:{normalise_endpoint(endpoint or preset.base_url or '')}"
    return preset.id


def key_file_path(config_root: Path | None = None) -> Path:
    return (config_root or _files.config_root()) / KEY_FILE


def key_file_warning(config_root: Path | None = None) -> str:
    """The question asked before a key goes to the key file."""
    return KEY_FILE_WARNING.format(path=key_file_path(config_root))


# --- the keyring, in a thread that may be given up ---------------------------------------------------------------
def _with_timeout(fn: Callable[[], T], timeout: float) -> T:
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:        # noqa: BLE001  (handed to the caller's thread)
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name="flextool-assistant-keyring", daemon=True).start()
    if not done.wait(timeout):
        raise KeyringTimeout(KEYRING_SLOW.format(seconds=f"{timeout:g}"))
    if "error" in box:
        raise box["error"]
    return box["value"]


def _is_plaintext(backend: object) -> bool:
    cls = type(backend)
    return cls.__module__.startswith("keyrings.alt") or "plaintext" in cls.__name__.lower()


def _problem(backend: object) -> str | None:
    """Why *backend* is not used, or ``None`` when it is."""
    cls = type(backend)
    if _is_plaintext(backend):
        return KEYRING_PLAINTEXT
    if cls.__module__ in ("keyring.backends.fail", "keyring.backends.null"):
        return KEYRING_NONE.format(backend=_name(backend))
    try:
        priority = float(getattr(backend, "priority"))
    except Exception as exc:                # noqa: BLE001  (a backend says it does not fit by raising)
        return KEYRING_NONE.format(backend=f"{_name(backend)}: {exc}")
    if priority < 1:
        return KEYRING_NONE.format(backend=_name(backend))
    return None


def _name(backend: object) -> str:
    return f"{type(backend).__module__}.{type(backend).__name__}"


def _usable_backend() -> Any:
    """The keyring backend FlexTool uses; raises :class:`KeyringUnavailable`.  A chain of backends is reduced to
    its first usable member, so a key is never read from or written to an unencrypted one."""
    import keyring
    from keyring.backends.chainer import ChainerBackend

    backend = keyring.get_keyring()
    if isinstance(backend, ChainerBackend):
        members = list(backend.backends)
        plaintext_only = bool(members) and all(_is_plaintext(m) for m in members)
        for member in members:
            if _problem(member) is None:
                return member
        raise KeyringUnavailable(KEYRING_PLAINTEXT if plaintext_only else
                                 KEYRING_NONE.format(backend="no usable backend in the keyring chain"))
    problem = _problem(backend)
    if problem is not None:
        raise KeyringUnavailable(problem)
    return backend


def keyring_status(timeout: float | None = None) -> tuple[bool, str]:
    """Whether the system keyring can keep keys here, and the text to show."""
    try:
        backend = _with_timeout(_usable_backend, KEYRING_TIMEOUT if timeout is None else timeout)
    except KeyringUnavailable as exc:
        return False, str(exc)
    except Exception as exc:                # noqa: BLE001  (an import or D-Bus failure is "not available")
        return False, KEYRING_NONE.format(backend=str(exc) or type(exc).__name__)
    return True, f"The system keyring is used ({getattr(backend, 'name', _name(backend))})."


def _keyring_call(action: str, fn: Callable[[Any], T], timeout: float | None) -> T:
    def run() -> T:
        return fn(_usable_backend())
    try:
        return _with_timeout(run, KEYRING_TIMEOUT if timeout is None else timeout)
    except KeyringUnavailable:
        raise
    except Exception as exc:                # noqa: BLE001  (locked collection, dismissed prompt, D-Bus error)
        raise KeyringUnavailable(KEYRING_FAILED.format(action=action, reason=str(exc) or type(exc).__name__)) \
            from None


# --- the key file ------------------------------------------------------------------------------------------------
def _read_file(config_root: Path | None) -> dict[str, str]:
    try:
        data = _files.read_json(key_file_path(config_root))
    except (OSError, _files.UnreadableFile):
        return {}
    keys = (data or {}).get("keys")
    return {str(k): v for k, v in keys.items() if isinstance(v, str)} if isinstance(keys, dict) else {}


def _change_file(config_root: Path | None, change: Callable[[dict[str, str]], bool]) -> None:
    path = key_file_path(config_root)
    with _files.locked(path):
        try:
            data = _files.read_json(path) or {"format": _files.FORMAT, "keys": {}}
        except _files.UnreadableFile as exc:
            raise KeyStoreError(f"{exc} Remove it, then store the key again.") from None
        if _files.is_newer(data):
            raise KeyStoreError(str(_files.NewerFormat(path, data.get("format"))))
        keys = data.get("keys")
        if not isinstance(keys, dict):
            keys = data["keys"] = {}
        if not change(keys):
            return
        if keys:
            _files.write_json_atomic(path, data, private=True)
        else:
            path.unlink(missing_ok=True)


def _remove_from_file(config_root: Path | None, account: str) -> None:
    if key_file_path(config_root).exists():
        _change_file(config_root, lambda keys: keys.pop(account, None) is not None)


# --- the fingerprint of a key ------------------------------------------------------------------------------------
#: The salt of this process when the salt file can neither be read nor written: fingerprints then hold for this
#: process only, so what was saved with them stops matching (a mark is dropped, never kept for another key).
_PROCESS_SALT = secrets.token_hex(16)


def _salt(config_root: Path | None) -> str:
    path = (config_root or _files.config_root()) / SALT_FILE

    def read() -> str | None:
        data = _files.read_json(path, check_format=False) or {}
        salt = data.get("salt")
        return salt if isinstance(salt, str) and len(salt) >= 32 else None

    try:
        salt = read()
        if salt:
            return salt
        with _files.locked(path):
            salt = read()                   # another process may have written it meanwhile
            if not salt:
                salt = secrets.token_hex(16)
                _files.write_json_atomic(path, {"format": _files.FORMAT, "salt": salt}, private=True)
        return salt
    except (OSError, _files.UnreadableFile, _files.LockTimeout, _files.NewerFormat):
        return _PROCESS_SALT


def key_fingerprint(key: str | None, *, config_root: Path | None = None) -> str:
    """A short one-way fingerprint of *key*: ``HMAC-SHA-256(salt of this installation, key)``, 12 hex digits;
    :data:`NO_KEY` without a key.  Tells whether the key changed; the key cannot be read back from it."""
    key = (key or "").strip()
    if not key:
        return NO_KEY
    digest = hmac.new(_salt(config_root).encode("utf-8"), key.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:FINGERPRINT_LENGTH]


# --- the public calls --------------------------------------------------------------------------------------------
def resolve_key(preset: Preset, endpoint: str | None = None, *, config_root: Path | None = None,
                timeout: float | None = None) -> tuple[str | None, KeySource, str | None]:
    """The key of *preset* at *endpoint*: ``(key, where it came from, note)``.  The note says why the keyring was
    not used, when no key was found there."""
    from_env = read_key_env(preset.key_env)
    if from_env:
        return from_env, KeySource.ENV, None
    account = key_account(preset, endpoint)
    note = None
    try:
        key = _keyring_call("read", lambda backend: backend.get_password(SERVICE, account), timeout)
    except KeyringUnavailable as exc:
        key, note = None, str(exc)
    if key:
        return key, KeySource.KEYRING, None
    key = _read_file(config_root).get(account)
    if key:
        return key, KeySource.FILE, None
    return None, KeySource.NONE, note


def store_key(preset: Preset, endpoint: str | None, key: str, *, to_file: bool = False,
              config_root: Path | None = None, timeout: float | None = None) -> KeySource:
    """Store *key*: in the system keyring, or with *to_file* (after the user accepted
    :func:`key_file_warning`) in the key file.  A key in the other place is removed, so the one just stored is the
    one found.  Raises :class:`KeyStoreError` when the keyring cannot be used: nothing falls back to the file
    without the user's choice."""
    key = key.strip()
    if not key:
        raise KeyStoreError("The key is empty.")
    account = key_account(preset, endpoint)
    if to_file:
        def put(keys: dict[str, str]) -> bool:
            keys[account] = key
            return True
        _change_file(config_root, put)
        try:
            _keyring_call("remove", lambda backend: _delete(backend, account), timeout)
        except KeyringUnavailable:
            pass                            # no keyring here, so nothing there to shadow the file
        return KeySource.FILE
    try:
        _keyring_call("store", lambda backend: backend.set_password(SERVICE, account, key), timeout)
    except KeyringUnavailable as exc:
        raise KeyStoreError(str(exc)) from None
    _remove_from_file(config_root, account)
    return KeySource.KEYRING


def _delete(backend: Any, account: str) -> None:
    from keyring.errors import PasswordDeleteError

    try:
        backend.delete_password(SERVICE, account)
    except PasswordDeleteError:
        pass                                # there was none


def forget_key(preset: Preset, endpoint: str | None = None, *, config_root: Path | None = None,
               timeout: float | None = None) -> str | None:
    """Remove the stored key of *preset* at *endpoint* from the keyring and the key file; returns a note when the
    keyring could not be reached (a key may then still be there)."""
    account = key_account(preset, endpoint)
    note = None
    try:
        _keyring_call("remove", lambda backend: _delete(backend, account), timeout)
    except KeyringUnavailable as exc:
        note = str(exc)
    _remove_from_file(config_root, account)
    return note
