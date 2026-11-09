"""The project's input sources as the FlexTool window numbers them, read without changing anything.

The sources are the spreadsheets and databases in ``input_sources/`` plus the files added from elsewhere
(``external_refs`` in ``settings.yaml``), as ``gui.input_sources.InputSourceManager.scan_input_sources`` lists
them.  The number of a source is the one in ``settings.yaml`` (the suffix of its result folders,
``gui.scenario_key``).  The settings are read, never saved, and the window's ``refresh`` (which saves them) is never
called.  A spreadsheet is read through its conversion ``intermediate/<stem>.sqlite``, which FlexTool writes when one
of its scenarios runs.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, close_names, name_list

SPREADSHEET_KINDS = ("xlsx", "ods")
CONVERTING_SECONDS = 10.0


@dataclass(frozen=True)
class Source:
    name: str                       # the file name, as the window shows it
    number: int | None              # the source number of settings.yaml, None when it has none yet
    kind: str                       # sqlite · xlsx · ods
    path: Path                      # the file itself
    external: bool

    @property
    def spreadsheet(self) -> bool:
        return self.kind in SPREADSHEET_KINDS

    def database_path(self, project_path: Path) -> Path:
        """The SQLite file the tools read: the source itself, or a spreadsheet's conversion."""
        if self.spreadsheet:
            return project_path / "intermediate" / f"{Path(self.name).stem}.sqlite"
        return self.path


def settings(ctx: AssistantContext):
    """The project's ``settings.yaml`` (defaults when it is missing); read only."""
    from flextool.gui.settings_io import load_project_settings

    return load_project_settings(ctx.project_path)


def sources(ctx: AssistantContext) -> list[Source]:
    """Every input source of the project, in the window's order."""
    from flextool.gui.input_sources import InputSourceManager

    project_settings = settings(ctx)
    manager = InputSourceManager(ctx.project_path, project_settings)
    numbers = dict(project_settings.input_source_numbers)
    by_path = {rec.path: int(n) for n, rec in project_settings.source_registry.items() if rec.path}
    out: list[Source] = []
    for info in manager.scan_input_sources():
        path = manager.resolve_path(info)
        rel = info.external_rel_path or f"input_sources/{info.name}"
        number = numbers.get(info.name)
        if number is None:
            number = by_path.get(rel)
        out.append(Source(info.name, int(number) if number is not None else None, info.file_type, path,
                          bool(info.external_rel_path)))
    return out


def source_names(all_sources: list[Source]) -> str:
    return name_list([s.name if s.number is None else f"{s.name} ({s.number})" for s in all_sources])


def resolve_source(ctx: AssistantContext, source: str | None) -> Source:
    """The source named *source* (file name, name without extension, or number); without one the connection's
    default, else the only source."""
    all_sources = sources(ctx)
    if not all_sources:
        raise ToolError("no_sources")
    wanted = source if source is not None else ctx.default_source
    if wanted is None:
        if len(all_sources) == 1:
            return all_sources[0]
        raise ToolError("choose_source", sources=source_names(all_sources))
    text = str(wanted).strip()
    for s in all_sources:
        if text in (s.name, Path(s.name).stem) or (s.number is not None and text == str(s.number)):
            return s
    lower = text.lower()
    matches = [s for s in all_sources if lower in (s.name.lower(), Path(s.name).stem.lower())]
    if len(matches) == 1:
        return matches[0]
    raise ToolError("unknown_source", source=text, sources=source_names(all_sources))


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def conversion_state(ctx: AssistantContext, source: Source) -> dict | None:
    """``{state, converted_at}`` of a spreadsheet's conversion (``None`` for a database source).

    ``missing`` — not converted; ``converting`` — the conversion is younger than 10 s or has a ``-journal`` next to
    it; ``stale`` — the spreadsheet changed after it; ``fresh`` otherwise.
    """
    if not source.spreadsheet:
        return None
    db_path = source.database_path(ctx.project_path)
    try:
        converted = db_path.stat().st_mtime
    except OSError:
        return {"state": "missing", "converted_at": None}
    journal = db_path.with_name(db_path.name + "-journal")
    if journal.exists() or time.time() - converted < CONVERTING_SECONDS:
        state = "converting"
    else:
        try:
            changed = source.path.stat().st_mtime
        except OSError:
            changed = converted
        state = "stale" if changed > converted else "fresh"
    return {"state": state, "converted_at": _iso(converted)}


def open_source(ctx: AssistantContext, source: Source):
    """``(InputDB, notes)`` of a source; a stale conversion is read with a note."""
    from flextool.assistant.inputs import _db

    notes: list[str] = []
    if source.spreadsheet:
        conversion = conversion_state(ctx, source)
        assert conversion is not None
        if conversion["state"] == "missing":
            raise ToolError("not_converted")
        if conversion["state"] == "converting":
            raise ToolError("converting")
        if conversion["state"] == "stale":
            notes.append(f"The spreadsheet was changed after its last conversion ({conversion['converted_at']}); "
                         "these are the values of that conversion. Run a scenario of it to convert it again.")
    elif not source.path.is_file():
        raise ToolError("source_missing", source=source.name, path=str(source.path))
    return _db.load(source.database_path(ctx.project_path)), notes


def default_scenario_for(ctx: AssistantContext, source: Source, db) -> str | None:
    """The connection's default scenario where it applies to *source*, else ``None``.

    With a default source (the FlexTool window's, or ``--input-source``) the default scenario belongs to it: it
    applies to that source only.  Without one (``--scenario`` alone names no source) it applies to any source that
    has that scenario.
    """
    wanted = ctx.default_scenario
    if wanted is None:
        return None
    if ctx.default_source is not None:
        try:
            default = resolve_source(ctx, None)
        except ToolError:
            return None
        return wanted if default.path == source.path else None
    return wanted if wanted in db.scenario_names() else None


def resolve_scenario(ctx: AssistantContext, source: Source, db, scenario: str | None) -> str:
    """The scenario named *scenario*; without one the connection's default where it applies to *source*
    (:func:`default_scenario_for`), else the only scenario."""
    names = db.scenario_names()
    if not names:
        raise ToolError("no_scenarios", source=source.name)
    wanted = scenario if scenario is not None else default_scenario_for(ctx, source, db)
    if wanted is None:
        if len(names) == 1:
            return names[0]
        raise ToolError("choose_scenario", source=source.name, n=len(names), scenarios=name_list(names))
    if wanted in names:
        return wanted
    raise ToolError("unknown_scenario", scenario=wanted, source=source.name,
                    scenarios=name_list(close_names(wanted, names) or names))


def relative(ctx: AssistantContext, path: Path) -> str:
    """*path* relative to the project (POSIX), or its name when it is elsewhere."""
    try:
        return Path(os.path.relpath(path, ctx.project_path)).as_posix()
    except ValueError:
        return path.name
