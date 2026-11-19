"""The coded warnings of a test (``flextool.run_warnings``), in two views.

**Structured calls** — :func:`capture_coded`, a context manager::

    with capture_coded() as got:
        warn_node_connections(db)
    assert got.names("W-0300") == ["west"]
    assert "alternative 'base'" in got.detail("W-0300", "west")
    assert got.lines() == [...]                 # the console lines that were emitted

Every :func:`run_warnings.warn` / :func:`~run_warnings.note` call goes through ``warn_grouped``; the spy records
each call (code, names, *fmt*, details) and passes it on, so the lines are still emitted as usual.  Details given
as one text (not a ``{name: text}`` mapping) are kept under the name ``""``.  A call with an *empty* name list
emits nothing (the check found no case): it is kept apart in ``got.empty_calls``, never in ``got.calls`` /
``got.codes()``; a call without names (``names=None``, a code that names no entity) is in ``got.calls`` with
``entityless=True``.  ``got.lines()`` are the ``Warning:`` / ``Note:`` lines the calls actually printed, so a
test can tell "no call" and "a call that printed nothing" from "a line without a name".

**Log text** — the ``coded_log`` fixture (this module is a plugin of ``tests/conftest.py``, ``pytest_plugins``):

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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

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


# -- structured calls -------------------------------------------------------------------------------------------


@dataclass
class CodedCall:
    code: str
    names: list[str]
    fmt: dict[str, Any]
    details: dict[str, str]
    #: The call gave no names (``names=None``): a line without ``(...)``.
    entityless: bool = False


class _LineHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.INFO)
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record.getMessage())


@dataclass
class Captured:
    calls: list[CodedCall] = field(default_factory=list)
    #: Calls whose name list was given but empty (they print nothing).
    empty_calls: list[CodedCall] = field(default_factory=list)
    #: The ``Warning:`` / ``Note:`` lines printed while capturing.
    printed: list[str] = field(default_factory=list)

    def codes(self) -> list[str]:
        """The codes of the calls that name something (or no entity by design), in order, each once."""
        return list(dict.fromkeys(call.code for call in self.calls))

    def names(self, code: str, **fmt: Any) -> list[str]:
        """The names of *code* (in calls whose *fmt* has the given items), in order, each once."""
        out: dict[str, None] = {}
        for call in self.calls:
            if call.code == code and all(str(call.fmt.get(k)) == str(v) for k, v in fmt.items()):
                out.update(dict.fromkeys(call.names))
        return list(out)

    def entityless(self, code: str) -> bool:
        """Whether *code* was emitted without names (``names=None``)."""
        return any(call.code == code and call.entityless for call in self.calls)

    def detail(self, code: str, name: str) -> str:
        """The details of *name* under *code* (``""`` when none); several calls' details are joined by ``; ``."""
        return "; ".join(call.details[name] for call in self.calls if call.code == code and name in call.details)

    def lines(self, code: str | None = None) -> list[str]:
        """The console lines printed (of *code* only, when given)."""
        return [line for line in self.printed if code is None or line.endswith(f"[{code}]")]


def _texts(details: Any) -> dict[str, str]:
    if isinstance(details, Mapping):
        return {run_warnings._name_text(k): str(v) for k, v in details.items()}
    if details is None:
        return {}
    return {"": str(details)}


@contextmanager
def capture_coded() -> Iterator[Captured]:
    got = Captured()
    original = run_warnings.warn_grouped

    def spy(code: str, names=None, **kwargs: Any) -> bool:
        fmt = {k: v for k, v in kwargs.items() if k not in ("solve", "details", "per_run", "_stacklevel")}
        listed = None if names is None else list(names)
        call = CodedCall(code, [run_warnings._name_text(n) for n in listed or ()], fmt,
                         _texts(kwargs.get("details")), entityless=listed is None)
        (got.empty_calls if listed is not None and not listed else got.calls).append(call)
        return original(code, listed, **kwargs)

    logger = logging.getLogger(run_warnings.LOGGER_NAME)
    handler = _LineHandler(got.printed)
    old_level = logger.level
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    run_warnings.warn_grouped = spy
    try:
        yield got
    finally:
        run_warnings.warn_grouped = original
        logger.removeHandler(handler)
        logger.setLevel(old_level)
