"""Capture the coded warnings (:mod:`flextool.run_warnings`) a piece of code emits, as structured cases.

    with capture_coded() as got:
        warn_node_connections(db)
    assert got.names("W-0300") == ["west"]
    assert "alternative 'base'" in got.detail("W-0300", "west")

Every :func:`run_warnings.warn` / :func:`~run_warnings.note` call goes through ``warn_grouped``; the spy records
each call (code, names, *fmt*, details) and passes it on, so the lines are still emitted as usual.  Details given
as one text (not a ``{name: text}`` mapping) are kept under the name ``""``.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from flextool import run_warnings


@dataclass
class CodedCall:
    code: str
    names: list[str]
    fmt: dict[str, Any]
    details: dict[str, str]


@dataclass
class Captured:
    calls: list[CodedCall] = field(default_factory=list)

    def codes(self) -> list[str]:
        """The codes emitted, in order, each once."""
        return list(dict.fromkeys(call.code for call in self.calls))

    def names(self, code: str, **fmt: Any) -> list[str]:
        """The names of *code* (in calls whose *fmt* has the given items), in order, each once."""
        out: dict[str, None] = {}
        for call in self.calls:
            if call.code == code and all(str(call.fmt.get(k)) == str(v) for k, v in fmt.items()):
                out.update(dict.fromkeys(call.names))
        return list(out)

    def detail(self, code: str, name: str) -> str:
        """The details of *name* under *code* (``""`` when none); several calls' details are joined by ``; ``."""
        return "; ".join(call.details[name] for call in self.calls if call.code == code and name in call.details)


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

    def spy(code: str, names=(), **kwargs: Any) -> bool:
        fmt = {k: v for k, v in kwargs.items() if k not in ("solve", "details", "per_run", "_stacklevel")}
        names = list(names)
        got.calls.append(CodedCall(code, [run_warnings._name_text(n) for n in names], fmt,
                                   _texts(kwargs.get("details"))))
        return original(code, names, **kwargs)

    run_warnings.warn_grouped = spy
    try:
        yield got
    finally:
        run_warnings.warn_grouped = original
