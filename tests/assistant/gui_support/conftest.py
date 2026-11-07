"""The chat machinery of the FlexTool window (P2 §13.1) runs under the same rules as the model-request tests: no
connection leaves this computer, and every blocking fake backend is released when a test ends."""
from __future__ import annotations

from tests.assistant.llm.conftest import fake_backends, no_network  # noqa: F401 — the shared guard and teardown
