"""The model-request tests never reach the network: a connection to anything but this computer is refused.

The backends are tested through ``httpx2.MockTransport`` handlers that answer with synthetic provider payloads,
and the session through :class:`~tests.assistant.llm.fake_backend.FakeBackend`.  Blocking fakes are released when a
test ends, whatever it did.
"""
from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator

import pytest

from tests.assistant.llm.fake_backend import FakeBackend


def _loopback(address: object) -> bool:
    if not isinstance(address, tuple) or not address:
        return True                 # a Unix socket
    try:
        return ipaddress.ip_address(str(address[0]).split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """``socket.connect`` refuses every address that is not on this computer."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def connect(self: socket.socket, address: object) -> None:
        if not _loopback(address):
            raise OSError(f"test refused a network connection to {address!r}")
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        if not _loopback(address):
            raise OSError(f"test refused a network connection to {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture
def fake_backends() -> Iterator[list[FakeBackend]]:
    """Register every :class:`FakeBackend` a test builds here; each is released at teardown."""
    made: list[FakeBackend] = []
    yield made
    for backend in made:
        backend.release_all()
