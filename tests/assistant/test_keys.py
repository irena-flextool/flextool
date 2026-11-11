"""Where the assistant finds and stores the AI services' keys (P1 §8.1).

The system keyring is the in-memory ``fake_keyring`` of the conftest; the user config folder is in ``tmp_path``.
Expected texts are copied from P1 §8.1.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import keyring.backends.fail
import keyring.backends.null
import keyring.core
import pytest
from keyring.backends.chainer import ChainerBackend

from flextool.assistant import _files, keys
from flextool.assistant.keys import KeySource, KeyStoreError, resolve_key, store_key
from flextool.assistant.providers import PRESETS
from tests.assistant.fake_keyring import FakeKeyring, LowPriorityKeyring, PlaintextKeyring

GEMINI = PRESETS["gemini"]
OLLAMA = PRESETS["ollama"]
SECRET = "AIzaTESTKEYabcdefghijklmnopqrstuvwxyz0123"
PLAINTEXT_TEXT = ("The keyring setting points to a backend that stores keys unencrypted; FlexTool does not use it. "
                  "Use --key-file if you accept an unencrypted file.")


@pytest.fixture
def root(user_config_in_tmp: Path) -> Path:
    return user_config_in_tmp


@pytest.fixture(autouse=True)
def no_key_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    for preset in PRESETS.values():
        if preset.key_env:
            monkeypatch.delenv(preset.key_env, raising=False)


def files_under(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def test_the_environment_wins_over_the_keyring(fake_keyring, monkeypatch, root):
    fake_keyring.passwords[("flextool-assistant", "gemini")] = "from-keyring"
    assert resolve_key(GEMINI, config_root=root) == ("from-keyring", KeySource.KEYRING, None)
    monkeypatch.setenv("GEMINI_API_KEY", f"  {SECRET}\n")
    key, source, note = resolve_key(GEMINI, config_root=root)
    assert (key, source, note) == (SECRET, KeySource.ENV, None)
    assert source.describe(GEMINI) == "the environment variable GEMINI_API_KEY"
    assert KeySource.KEYRING.describe(GEMINI) == "the system keyring"
    assert KeySource.FILE.describe(GEMINI) == "the key file"
    monkeypatch.setenv("GEMINI_API_KEY", "")
    assert resolve_key(GEMINI, config_root=root)[1] is KeySource.KEYRING     # empty counts as unset


def test_a_stored_key_is_in_the_keyring_only(fake_keyring, root):
    assert store_key(GEMINI, None, f" {SECRET} ", config_root=root) is KeySource.KEYRING
    assert fake_keyring.passwords == {("flextool-assistant", "gemini"): SECRET}
    assert resolve_key(GEMINI, config_root=root) == (SECRET, KeySource.KEYRING, None)
    assert not any(SECRET in p.read_text(encoding="utf-8", errors="replace") for p in files_under(root))
    keys.forget_key(GEMINI, config_root=root)
    assert fake_keyring.passwords == {}
    assert resolve_key(GEMINI, config_root=root) == (None, KeySource.NONE, None)


def test_optional_keys_are_kept_per_endpoint(fake_keyring, root):
    store_key(OLLAMA, "HTTP://LocalHost:11434/v1/", "token-a", config_root=root)
    store_key(OLLAMA, "http://192.168.1.20:11434/v1", "token-b", config_root=root)
    assert set(fake_keyring.passwords) == {("flextool-assistant", "ollama:http://localhost:11434/v1"),
                                           ("flextool-assistant", "ollama:http://192.168.1.20:11434/v1")}
    assert resolve_key(OLLAMA, "http://localhost:11434/v1", config_root=root)[0] == "token-a"
    assert resolve_key(OLLAMA, None, config_root=root)[0] == "token-a"          # the preset's own address


@pytest.mark.parametrize("backend", [keyring.backends.fail.Keyring(), keyring.backends.null.Keyring(),
                                     LowPriorityKeyring()], ids=["fail", "null", "priority below 1"])
def test_a_keyring_that_cannot_keep_keys_is_not_used(backend, monkeypatch, root):
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend)
    ok, text = keys.keyring_status()
    assert not ok and text.startswith("There is no usable system keyring on this computer")
    key, source, note = resolve_key(GEMINI, config_root=root)
    assert (key, source) == (None, KeySource.NONE) and note == text
    with pytest.raises(KeyStoreError) as caught:
        store_key(GEMINI, None, SECRET, config_root=root)
    assert SECRET not in str(caught.value)
    assert not keys.key_file_path(root).exists()          # never falls back to the file by itself


def test_a_chain_without_usable_backends_is_not_used(monkeypatch, root):
    class EmptyChain(ChainerBackend):
        backends: list = []

    monkeypatch.setattr(keyring.core, "_keyring_backend", EmptyChain())
    assert keys.keyring_status()[0] is False


def test_a_chain_uses_its_first_encrypted_member(monkeypatch, root):
    good = FakeKeyring()
    plain = PlaintextKeyring()

    class Chain(ChainerBackend):
        backends = [plain, good]

    monkeypatch.setattr(keyring.core, "_keyring_backend", Chain())
    store_key(GEMINI, None, SECRET, config_root=root)
    assert plain.passwords == {} and good.passwords == {("flextool-assistant", "gemini"): SECRET}


@pytest.mark.parametrize("module", ["keyrings.alt.file", "tests.assistant.fake_keyring"])
def test_an_unencrypted_backend_is_refused(module, monkeypatch, root):
    backend_class = type("PlaintextKeyring" if module.startswith("tests") else "EncryptedKeyring",
                         (FakeKeyring,), {"__module__": module})
    monkeypatch.setattr(keyring.core, "_keyring_backend", backend_class())
    assert keys.keyring_status() == (False, PLAINTEXT_TEXT)
    with pytest.raises(KeyStoreError, match="stores keys unencrypted"):
        store_key(GEMINI, None, SECRET, config_root=root)


def test_python_keyring_backend_choosing_a_plaintext_backend_is_refused(monkeypatch, root):
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "tests.assistant.fake_keyring.PlaintextKeyring")
    monkeypatch.setattr(keyring.core, "_keyring_backend", None)     # keyring chooses again, from the variable
    assert keys.keyring_status() == (False, PLAINTEXT_TEXT)
    assert type(keyring.core._keyring_backend) is PlaintextKeyring


def test_a_keyring_that_does_not_answer_is_given_up(fake_keyring, root):
    fake_keyring.block = threading.Event()
    started = time.monotonic()
    key, source, note = resolve_key(GEMINI, config_root=root, timeout=0.2)
    assert time.monotonic() - started < 2.0
    assert (key, source) == (None, KeySource.NONE)
    assert note == "The system keyring did not answer within 0.2 seconds."
    assert keys.keyring_status(timeout=0.2)[0] is True       # choosing the backend does not call it
    with pytest.raises(KeyStoreError, match="did not answer"):
        store_key(GEMINI, None, SECRET, config_root=root, timeout=0.2)
    stuck = [t for t in threading.enumerate() if t.name == "flextool-assistant-keyring"]
    assert stuck and all(t.daemon for t in stuck)            # never keeps the process from exiting
    fake_keyring.block.set()
    assert keys.KEYRING_TIMEOUT == 5.0


class SlowFirstStore(FakeKeyring):
    """A keyring whose store of one key waits for :attr:`gate` (a store that is given up and lands later); every
    other call answers at once."""

    def __init__(self, slow_key: str) -> None:
        super().__init__()
        self.slow_key = slow_key
        self.gate = threading.Event()
        self.landed = threading.Event()

    def set_password(self, service: str, username: str, password: str) -> None:
        if password == self.slow_key:
            self.gate.wait(30)
            super().set_password(service, username, password)
            self.landed.set()
            return
        super().set_password(service, username, password)


@pytest.fixture
def slow_keyring(monkeypatch) -> object:
    fake = SlowFirstStore("late-key-A")
    monkeypatch.setattr(keyring.core, "_keyring_backend", fake)
    yield fake
    fake.gate.set()


def _settled(fake: SlowFirstStore, expected: dict) -> bool:
    """Whether the keyring holds *expected* within 2 s after the late store landed."""
    assert fake.landed.wait(2.0)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if fake.passwords == expected:
            return True
        time.sleep(0.01)
    return fake.passwords == expected


def test_a_late_keyring_store_does_not_shadow_the_file(slow_keyring, root):
    """C4-n1: a store that was given up lands after the key went to the file; the keyring entry is removed."""
    with pytest.raises(KeyStoreError, match="It may still store the key later"):
        store_key(GEMINI, None, "late-key-A", config_root=root, timeout=0.05)
    assert store_key(GEMINI, None, "file-key", to_file=True, config_root=root) is KeySource.FILE
    slow_keyring.gate.set()
    assert _settled(slow_keyring, {})
    assert resolve_key(GEMINI, config_root=root) == ("file-key", KeySource.FILE, None)


def test_a_late_keyring_store_never_removes_a_key_stored_since(slow_keyring, root):
    """The late key A lands after the user put a key in the file and then key B in the keyring: B stays (a
    compare-and-delete, never a blind removal), and A does not replace it."""
    with pytest.raises(KeyStoreError, match="did not answer"):
        store_key(GEMINI, None, "late-key-A", config_root=root, timeout=0.05)
    store_key(GEMINI, None, "file-key", to_file=True, config_root=root)
    assert store_key(GEMINI, None, "key-B", config_root=root) is KeySource.KEYRING
    slow_keyring.gate.set()
    assert _settled(slow_keyring, {("flextool-assistant", "gemini"): "key-B"})
    assert resolve_key(GEMINI, config_root=root) == ("key-B", KeySource.KEYRING, None)


def test_a_late_keyring_store_after_a_removal_is_removed(slow_keyring, root):
    with pytest.raises(KeyStoreError):
        store_key(GEMINI, None, "late-key-A", config_root=root, timeout=0.05)
    keys.forget_key(GEMINI, config_root=root)
    slow_keyring.gate.set()
    assert _settled(slow_keyring, {})
    assert resolve_key(GEMINI, config_root=root)[1] is KeySource.NONE


def test_the_key_file_only_after_the_choice_owner_only(fake_keyring, root, monkeypatch):
    assert keys.key_file_warning(root) == (
        f"The key will be stored unencrypted at {root / 'keys.json'}. Anyone who can read your files can read it. "
        "Store it there? [y/N]")
    fake_keyring.passwords[("flextool-assistant", "gemini")] = "old-keyring-key"
    modes: dict[str, int] = {}
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        if str(path).endswith(".tmp"):
            modes[str(path)] = mode
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(_files.os, "open", recording_open)
    assert store_key(GEMINI, None, SECRET, to_file=True, config_root=root) is KeySource.FILE
    assert list(modes.values()) == [0o600]                   # created owner-only, before the key was written
    path = keys.key_file_path(root)
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600
    assert fake_keyring.passwords == {}                      # the keyring would otherwise win over the file
    assert resolve_key(GEMINI, config_root=root) == (SECRET, KeySource.FILE, None)
    # Storing in the keyring again removes the copy in the file.
    store_key(GEMINI, None, "new-key", config_root=root)
    assert not path.exists()
    assert resolve_key(GEMINI, config_root=root) == ("new-key", KeySource.KEYRING, None)


def test_a_key_file_without_a_keyring(monkeypatch, root):
    monkeypatch.setattr(keyring.core, "_keyring_backend", keyring.backends.fail.Keyring())
    store_key(GEMINI, None, SECRET, to_file=True, config_root=root)
    key, source, note = resolve_key(GEMINI, config_root=root)
    assert (key, source, note) == (SECRET, KeySource.FILE, None)
    note = keys.forget_key(GEMINI, config_root=root)
    assert note is not None and SECRET not in note
    assert not keys.key_file_path(root).exists()


def test_the_connection_never_shows_the_key():
    from flextool.assistant.llm_adapter import Connection, Route

    conn = Connection("https://example.org/v1", "m", SECRET, "the environment variable GEMINI_API_KEY", {},
                      Route("remote", "example.org", None, None, "Google Gemini"))
    assert SECRET not in repr(conn)
