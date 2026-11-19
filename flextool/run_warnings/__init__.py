"""Coded, aggregated warnings and notes of a FlexTool run.

One console line per warning code and solve (never one per entity)::

    Warning: <message> (<N> cases, e.g. '<first name>') [W-0123]
    Warning: <message> ('<the name>') [W-0123]                      # N == 1
    Warning: <message> [W-0123]                                    # no entity (names not given)

The message comes from the registry (:mod:`flextool.run_warnings.registry`), which also documents every code on
the web page :data:`WARNINGS_DOCS_URL`.  Lines go through the ``logging`` logger ``flextool.warnings`` (WARNING
level for a warning, INFO for a note), so the console, the run log and the run record (``_run_status.json``, which
stores the code and the number of cases with the line) all get them.  The full list of names, and the
per-name *details*, go to the same logger at DEBUG level.  When nothing configured ``logging`` (the engine used
as a library, e.g. from a notebook), Python's last-resort handler prints the warnings to stderr; the notes are
printed there too (to see the debug lines, configure ``logging``).

Emitting
--------
* :func:`warn_grouped` ``(code, names, *, solve=None, details=None, per_run=False, **fmt)``: one line for *names*.
  *names* given but empty (a check that found nothing): no line, no tally, no run-record entry.  Leave *names*
  out (``None``) for a code that names no entity: its line has no ``(...)`` part.
* :func:`warn` ``(code, name=None, ...)``: the same for one name (``None``: no entity); inside a
  :func:`collecting` scope the names of repeated calls join one line (the pattern for a check that finds
  entities one at a time in a loop).
* :func:`note`: the same for an informational ``Note:`` line.

Scope
-----
* Outside a :func:`collecting` scope a call prints at once.  With ``solve=`` a code prints once per complete
  solve (the rolls ``<solve>_roll_<k>`` of a rolling solve count as one solve): a later call for the same code,
  *fmt* and solve only adds its new names to the debug log and to the tally.
* ``with collecting():`` gathers the calls made inside it and prints each (code, *fmt*, solve) as one line when
  the scope ends (also when it ends by an exception).  Scopes nest: the outermost one prints.
* ``per_run=True``: a name already named under the code earlier in the run is left out (each entity once per
  run, however the solves group them); no line when no name is new.

Run
---
:func:`start_run` forgets what earlier runs printed (called by ``_behaviour_changes.start_run`` at the start of
every engine run).  :func:`tally_lines` gives the end-of-run summary that ``cmd_run_flextool`` prints::

    Warnings: 2 kinds, 15 cases — W-0004 ×12, W-0001 ×3
    Warnings explained: https://irena-flextool.github.io/flextool/reference/warnings/
    Other warnings (no code): 2, see the log above
"""
from __future__ import annotations

import logging
import re
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from typing import Any, Iterator

from flextool.run_warnings.registry import (
    NOTE,
    WARNING,
    WARNINGS,
    WARNINGS_DOCS_URL,
    WARNINGS_PAGE,
    WarningSpec,
    docs_url,
)

__all__ = [
    "CASES_ATTR", "CODE_ATTR", "LOGGER_NAME", "MAX_NAME", "WARNINGS", "WARNINGS_DOCS_URL", "WARNINGS_PAGE",
    "WarningSpec", "add_cases_listener", "collecting", "complete_solve", "docs_url", "flush", "format_line", "note",
    "remove_cases_listener", "start_run", "tally", "tally_lines", "uncoded_count", "warn", "warn_grouped",
]

LOGGER_NAME = "flextool.warnings"
#: ``LogRecord`` attributes of a coded line (``extra=``): the code and the number of cases.  The run record's
#: warning handler stores them with the line (``flextool.run_record._status``).
CODE_ATTR = "flextool_warning_code"
CASES_ATTR = "flextool_warning_cases"
#: A name longer than this is shortened in the console line (the debug log has it whole).
MAX_NAME = 40

_logger = logging.getLogger(LOGGER_NAME)
_lock = threading.RLock()
# (code, fmt key, solve) printed in this run -> the names it covered (the solve-scoped de-duplication).
_printed: dict[tuple, set[str]] = {}
# code -> names named under it in this run (per_run=True).
_named: dict[str, set[str]] = {}
# code -> number of cases in this run (the tally).
_cases: dict[str, int] = {}
# The open collecting() scope: (code, fmt key, solve) -> pending call; None when no scope is open.
_pending: dict[tuple, "_Pending"] | None = None
_depth = 0
# Called with (code, cases) when a call adds cases to a line printed earlier (no new line): the run record keeps
# its ``cases`` equal to the tally (:func:`add_cases_listener`).
_case_listeners: list[Callable[[str, int], None]] = []
# Uncoded WARNING-or-worse records of this run (any logger): the tally names how many there were.
_uncoded = 0


class _UncodedCounter(logging.Handler):
    """On the root logger: counts the warnings that carry no code (``logging.warning`` calls not converted to
    this module), so the tally does not read as complete when the log has others."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)

    def handle(self, record: logging.LogRecord) -> bool:   # no lock or formatting needed: one integer
        global _uncoded
        if record.levelno >= logging.WARNING and getattr(record, CODE_ATTR, None) is None:
            _uncoded += 1
        return True

    def emit(self, record: logging.LogRecord) -> None:     # pragma: no cover - handle() does the work
        pass


_counter = _UncodedCounter()


def _has_output_handler() -> bool:
    """Whether a handler (other than the uncoded counter) would get a record of ``flextool.warnings``."""
    logger: logging.Logger | None = _logger
    while logger is not None:
        if any(h is not _counter for h in logger.handlers):
            return True
        if not logger.propagate:
            return False
        logger = logger.parent
    return False


#: Where a line was asked for: ``(pathname, lineno, funcName)`` of the call site (the ``--debug`` log format shows
#: it; a line printed when a scope ends would otherwise point at :func:`flush`).
_Where = tuple[str, int, str]


def _call_site(depth: int) -> _Where | None:
    try:
        frame = sys._getframe(depth)   # 1: our caller (the logging module's stacklevel count)
    except ValueError:
        return None
    return frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name


class _Pending:
    __slots__ = ("code", "fmt", "solve", "names", "details", "per_run", "where")

    def __init__(self, code: str, fmt: dict[str, Any], solve: str | None, per_run: bool,
                 where: _Where | None) -> None:
        self.code, self.fmt, self.solve, self.per_run, self.where = code, fmt, solve, per_run, where
        self.names: dict[str, None] = {}
        self.details: list[tuple[str | None, str]] = []


_ROLL_SUFFIX = re.compile(r"_roll_\d+$")


def complete_solve(solve: str | None) -> str | None:
    """The complete solve *solve* belongs to: a roll ``<solve>_roll_<k>`` of a rolling solve counts as ``<solve>``
    (``None`` for no solve).  The one place that knows the roll-name suffix for the warnings and the log."""
    if not solve:
        return None
    return _ROLL_SUFFIX.sub("", str(solve))




def _name_text(name: Any) -> str:
    if isinstance(name, (tuple, list)):
        return ", ".join(str(part) for part in name)
    return str(name)


def _short(name: str) -> str:
    return name if len(name) <= MAX_NAME else name[:MAX_NAME - 1] + "…"


def _spec(code: str) -> WarningSpec:
    try:
        return WARNINGS[code]
    except KeyError:
        raise KeyError(f"unknown warning code {code!r}: add it to flextool/run_warnings/registry.py") from None


def format_line(code: str, names: Iterable[Any] = (), *, solve: str | None = None, **fmt: Any) -> str:
    """The console line of *code* for *names* (de-duplicated, in order), without printing it."""
    spec = _spec(code)
    texts = list(dict.fromkeys(_name_text(n) for n in names))
    fields = dict(fmt)
    fields.setdefault("solve", complete_solve(solve) or "")
    prefix = "Note" if spec.severity == NOTE else "Warning"
    line = f"{prefix}: {spec.message.format(**fields)}"
    if len(texts) == 1:
        line += f" ('{_short(texts[0])}')"
    elif texts:
        line += f" ({len(texts)} cases, e.g. '{_short(texts[0])}')"
    return f"{line} [{code}]"


def _details_text(details: str | Mapping[Any, Any] | None) -> list[tuple[str | None, str]]:
    """The debug-log lines of *details* with the name each is about: ``(name, "name: text")`` per item of a
    mapping, or ``(None, text)``."""
    if details is None:
        return []
    if isinstance(details, Mapping):
        return [(_name_text(key), f"{_name_text(key)}: {value}") for key, value in details.items()]
    return [(None, str(details))]


def _emit(code: str, names: list[str], fmt: dict[str, Any], solve: str | None, details: list[tuple[str | None, str]],
          stacklevel: int, where: _Where | None = None) -> None:
    """Print one line and log the full list (caller holds the lock).  *where*: the call site of a line printed
    when its scope ends (else the caller *stacklevel* frames up is)."""
    spec = _spec(code)
    cases = max(len(names), 1)
    _cases[code] = _cases.get(code, 0) + cases
    line = format_line(code, names, solve=solve, **fmt)
    level = logging.INFO if spec.severity == NOTE else logging.WARNING
    extra = {CODE_ATTR: code, CASES_ATTR: cases}
    if level < logging.WARNING and not _has_output_handler():
        # No logging configured (the engine used as a library): Python's last-resort handler shows only
        # warnings, so a note would vanish; it goes to stderr as the warnings do.
        print(line, file=sys.stderr, flush=True)
    elif where is None:
        _logger.log(level, line, extra=extra, stacklevel=stacklevel)
    elif _logger.isEnabledFor(level):
        _logger.handle(_logger.makeRecord(_logger.name, level, where[0], where[1], line, None, None, where[2],
                                          extra))
    if len(names) > 1:
        _logger.debug("%s all cases%s: %s", code, f" in solve '{solve}'" if solve else "", "; ".join(names))
    for _name, text in details:
        _logger.debug("%s %s", code, text)


def _fmt_key(fmt: Mapping[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in fmt.items()))


def _fresh(code: str, names: list[str], per_run: bool) -> list[str]:
    if not per_run:
        return names
    seen = _named.setdefault(code, set())
    fresh = [n for n in names if n not in seen]
    seen.update(fresh)
    return fresh


def _deliver(code: str, names: list[str], fmt: dict[str, Any], solve: str | None,
             details: list[tuple[str | None, str]],
             per_run: bool, stacklevel: int, where: _Where | None = None) -> bool:
    """Print now, or only log what a line of this solve did not cover yet (caller holds the lock)."""
    names = _fresh(code, names, per_run)
    if per_run and not names:
        return False
    if solve is None:
        _emit(code, names, fmt, None, details, stacklevel + 1, where)
        return True
    key = (code, _fmt_key(fmt), solve)
    covered = _printed.get(key)
    if covered is None:
        _printed[key] = set(names)
        _emit(code, names, fmt, solve, details, stacklevel + 1, where)
        return True
    extra = [n for n in names if n not in covered]
    if extra:
        covered.update(extra)
        _cases[code] = _cases.get(code, 0) + len(extra)
        for listener in list(_case_listeners):
            try:
                listener(code, len(extra))
            except Exception:  # noqa: BLE001 - a listener must not break a warning
                _logger.debug("a case listener failed", exc_info=True)
        _logger.debug("%s also in solve '%s': %s", code, solve, "; ".join(extra))
        new = set(extra)
        for name, text in details:          # the details of the new names (a name-less text: as given)
            if name is None or name in new:
                _logger.debug("%s %s", code, text)
    return False


def warn_grouped(code: str, names: Iterable[Any] | None = None, *, solve: str | None = None,
                 details: str | Mapping[Any, Any] | None = None, per_run: bool = False,
                 _stacklevel: int = 2, **fmt: Any) -> bool:
    """One aggregated line of *code* for *names* (entity names, or tuples of them).

    *names* given but empty: nothing at all (the check found no case).  *names* ``None``: a line without names,
    for a code that names no entity.  *fmt* fills the registry message (and is part of the grouping: calls with other *fmt* values give other
    lines).  *details* (text, or ``{name: text}``) go to the debug log.  Returns True when a line was printed now
    (False inside a :func:`collecting` scope, which prints at its end).
    """
    _spec(code)
    texts = [] if names is None else list(dict.fromkeys(_name_text(n) for n in names))
    if names is not None and not texts:
        return False
    solve_key = complete_solve(solve)
    detail_lines = _details_text(details)
    with _lock:
        if _pending is not None:
            key = (code, _fmt_key(fmt), solve_key, per_run)
            pending = _pending.get(key)
            if pending is None:
                pending = _pending[key] = _Pending(code, dict(fmt), solve_key, per_run, _call_site(_stacklevel))
            pending.names.update(dict.fromkeys(texts))
            pending.details.extend(detail_lines)
            return False
        return _deliver(code, texts, dict(fmt), solve_key, detail_lines, per_run, _stacklevel + 1)


def warn(code: str, name: Any = None, *, solve: str | None = None, details: str | Mapping[Any, Any] | None = None,
         per_run: bool = False, **fmt: Any) -> bool:
    """:func:`warn_grouped` for one *name* (or none); inside :func:`collecting` the names of repeated calls join
    one line."""
    return warn_grouped(code, None if name is None else (name,), solve=solve, details=details, per_run=per_run,
                        _stacklevel=3, **fmt)


def note(code: str, names: Iterable[Any] | None = None, *, solve: str | None = None,
         details: str | Mapping[Any, Any] | None = None, per_run: bool = False, **fmt: Any) -> bool:
    """:func:`warn_grouped` for a code registered as a note (a ``Note:`` line at INFO level)."""
    if _spec(code).severity != NOTE:
        raise ValueError(f"{code} is a warning, not a note: use warn_grouped")
    return warn_grouped(code, names, solve=solve, details=details, per_run=per_run, _stacklevel=3, **fmt)


def flush() -> None:
    """Print what the open :func:`collecting` scope gathered so far (one line per code, *fmt* and solve)."""
    global _pending
    with _lock:
        if not _pending:
            return
        pending, _pending = _pending, {}
        for item in pending.values():
            _deliver(item.code, list(item.names), item.fmt, item.solve, item.details, item.per_run, 3, item.where)


@contextmanager
def collecting() -> Iterator[None]:
    """Gather the warnings of a pass (a validation pass, a loop over entities, one solve) and print each code once
    when the pass ends."""
    global _pending, _depth
    with _lock:
        _depth += 1
        if _pending is None:
            _pending = {}
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                try:
                    flush()
                finally:
                    _pending = None


def add_cases_listener(listener: Callable[[str, int], None]) -> None:
    """Call *listener* ``(code, cases)`` whenever cases join a line of *code* printed earlier (no new line, so no
    log record carries them); the run record adds them to the code's entry."""
    with _lock:
        _case_listeners.append(listener)


def remove_cases_listener(listener: Callable[[str, int], None]) -> None:
    with _lock:
        if listener in _case_listeners:
            _case_listeners.remove(listener)


def start_run() -> None:
    """Begin a run: forget the lines, names and tally of earlier runs (an open scope keeps its pending calls), and
    count the run's uncoded warnings from now on."""
    global _uncoded
    with _lock:
        _printed.clear()
        _named.clear()
        _cases.clear()
        _uncoded = 0
        root = logging.getLogger()
        if _counter not in root.handlers:
            root.addHandler(_counter)


def tally() -> dict[str, int]:
    """``{code: cases}`` of the warnings (not the notes) of this run, most cases first."""
    with _lock:
        items = [(c, n) for c, n in _cases.items() if WARNINGS[c].severity == WARNING]
    return dict(sorted(items, key=lambda item: (-item[1], item[0])))


def uncoded_count() -> int:
    """How many uncoded warnings (WARNING or worse, any logger) were logged since :func:`start_run`."""
    return _uncoded


def tally_lines() -> list[str]:
    """The end-of-run summary: ``Warnings: <k> kinds, <n> cases — W-0004 ×12, ...`` and the docs link, then
    ``Other warnings (no code): <m>, see the log above`` when uncoded ones were logged; ``[]`` when the run had
    no warning."""
    counts = tally()
    other = uncoded_count()
    out: list[str] = []
    if counts:
        total = sum(counts.values())
        kinds = len(counts)
        listed = ", ".join(f"{code} ×{n}" for code, n in counts.items())
        out += [f"Warnings: {kinds} kind{'s' if kinds != 1 else ''}, {total} case{'s' if total != 1 else ''} — "
                f"{listed}", f"Warnings explained: {WARNINGS_DOCS_URL}"]
    if other:
        out.append(f"Other warnings (no code): {other}, see the log above")
    return out
