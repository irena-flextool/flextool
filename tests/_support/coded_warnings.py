"""The coded warnings of a test (``flextool.run_warnings``): the console lines and their debug-log details.

A plugin of ``tests/conftest.py`` (``pytest_plugins``), so every test can ask for the fixture:

* ``coded_log()`` returns the text of the ``flextool.warnings`` records (DEBUG and up, one per line) logged since
  the previous read: the ``Warning: ... [W-0123]`` / ``Note: ...`` lines and the ``W-0123 <name>: <detail>`` debug
  lines;
* ``coded_log.notices()`` returns, since the previous read, the full texts of the behaviour-change notices
  (``Behaviour changed in ...: <summary> Affected <kind>: <names>. <detail> <silence>``, the debug line of each
  ``W-06xx`` note, without its code), one per notice.
"""
from __future__ import annotations

import logging
import re

import pytest

from flextool import run_warnings

_NOTICE = re.compile(r"^W-\d{4} (?:In solve '[^']*': )?(Behaviour changed in .*)$", re.S)


class CodedLog:
    def __init__(self, caplog) -> None:
        self._caplog = caplog
        self._seen = 0

    def _new(self) -> list[str]:
        records = [r for r in self._caplog.records if r.name == run_warnings.LOGGER_NAME]
        new, self._seen = records[self._seen:], len(records)
        return [r.getMessage() for r in new]

    def __call__(self) -> str:
        return "".join(m + "\n" for m in self._new())

    def notices(self) -> list[str]:
        return [m.group(1) for m in map(_NOTICE.match, self._new()) if m]


@pytest.fixture
def coded_log(caplog) -> CodedLog:
    caplog.set_level(logging.DEBUG, logger=run_warnings.LOGGER_NAME)
    return CodedLog(caplog)
