"""The coded warnings of a test (``flextool.run_warnings``): the console lines and their debug-log details.

A plugin of ``tests/conftest.py`` (``pytest_plugins``), so every test can ask for the fixture:
``coded_log()`` returns the text of the ``flextool.warnings`` records (DEBUG and up, one per line) logged since
the previous call: the ``Warning: ... [W-0123]`` / ``Note: ...`` lines and the ``W-0123 <name>: <detail>`` debug
lines.
"""
from __future__ import annotations

import logging

import pytest

from flextool import run_warnings


@pytest.fixture
def coded_log(caplog):
    caplog.set_level(logging.DEBUG, logger=run_warnings.LOGGER_NAME)
    seen = [0]

    def read() -> str:
        records = [r for r in caplog.records if r.name == run_warnings.LOGGER_NAME]
        new, seen[0] = records[seen[0]:], len(records)
        return "".join(r.getMessage() + "\n" for r in new)
    return read
