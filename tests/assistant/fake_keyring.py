"""An in-memory keyring for the assistant tests: no test reads or writes the real system keyring.

:class:`FakeKeyring` keeps passwords in a dict; with :attr:`FakeKeyring.block` set, every call waits for that
event first (a keyring that does not answer).  :class:`PlaintextKeyring` stands for a backend that stores keys
unencrypted (a ``keyrings.alt`` file backend, as ``PYTHON_KEYRING_BACKEND`` may choose).
"""
from __future__ import annotations

import threading

from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError


class FakeKeyring(KeyringBackend):
    priority = 5

    def __init__(self) -> None:
        super().__init__()
        self.passwords: dict[tuple[str, str], str] = {}
        self.block: threading.Event | None = None

    def _wait(self) -> None:
        if self.block is not None:
            self.block.wait(30)

    def get_password(self, service: str, username: str) -> str | None:
        self._wait()
        return self.passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._wait()
        self.passwords[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self._wait()
        if self.passwords.pop((service, username), None) is None:
            raise PasswordDeleteError("no such password")


class PlaintextKeyring(FakeKeyring):
    """Stores keys unencrypted (by its name, as keyrings.alt's ``PlaintextKeyring``)."""


class LowPriorityKeyring(FakeKeyring):
    priority = 0.5
