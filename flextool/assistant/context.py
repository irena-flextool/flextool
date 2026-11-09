"""What every tool call knows: the project, the defaults the connection was started with, the caps and the
redaction, and what the session has already been told; and the snapshot of what the user has chosen, which the chat
puts in its context block.

The FlexTool window's snapshot (:class:`GuiState`, taken on the Tk thread at each Send and every second for the
runtime context file) becomes one payload with :func:`snapshot_payload`: the same dictionary goes into the runtime
file, answers ``get_gui_context`` and is what the context block shows (P2-D4).  Standard library only.

:data:`TOOL_LOCK` is the one lock every context of this process uses by default, so tools never run at once
(the input-data cache is shared by the whole process; several chats, and a closing chat's worker, may call tools).
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from flextool.assistant.caps import DEFAULT_PRESET, NAME_MAX, Caps, Rendered, cap_text, clean, preset
from flextool.assistant.redact import Redactor

FrontEnd = Literal["agent", "terminal", "gui"]

#: The lock every tool call of this process runs under (P2 §5.5).
TOOL_LOCK = threading.Lock()


@dataclass(frozen=True)
class ResultRef:
    """A result folder the user checked or selected in the FlexTool window, with what its run record says."""

    folder: str
    source_number: int
    scenario: str
    state: str | None
    results_usable: bool
    unusable_code: str | None
    reason: str | None


@dataclass(frozen=True)
class ViewerState:
    """What the result viewer of the FlexTool window shows."""

    mode: str
    scenarios: tuple[str, ...]
    plot_entry: str | None
    plot_name: str | None
    variant: str | None
    result_key: str | None
    sub_config: str | None
    time_start: int | None
    time_steps: int | None
    time_start_label: str | None
    dispatch_item: str | None
    comparison_checked: tuple[str, ...]


@dataclass(frozen=True)
class GuiState:
    """The selections of the FlexTool window when a question was sent."""

    sources_checked: tuple[tuple[str, int], ...]
    source_selected: str | None
    scenarios_checked: tuple[tuple[str, str], ...]
    scenario_selected: tuple[str, str] | None
    results_checked: tuple[ResultRef, ...]
    results_selected: tuple[ResultRef, ...]
    viewer: ViewerState | None


@dataclass(frozen=True)
class ContextSnapshot:
    """What the user has chosen when a question is sent: the project, the input source and the scenario the tools
    use by default, the front end, and (in the FlexTool window) its selections.  *taken_at* (UTC) is never in the
    context block."""

    project: str
    input_source: str | None = None
    scenario: str | None = None
    front_end: FrontEnd = "terminal"
    gui: GuiState | None = None
    taken_at: str = ""


#: Entries of one list in :func:`snapshot_payload`; the rest is counted in ``<list>_more``.
PAYLOAD_LIST_MAX = 20


def primary_defaults(gui: GuiState) -> tuple[str | None, str | None]:
    """``(input source, scenario)`` the tools use by default; the scenario always belongs to that source.

    Source: the highlighted source row, else the source of the highlighted scenario row, else the only checked
    source.  Scenario: the highlighted scenario row when it is of that source, else the one checked scenario of
    that source; ``None`` without a source.
    """
    selected = gui.scenario_selected
    if gui.source_selected:
        source: str | None = gui.source_selected
    elif selected is not None:
        source = selected[0]
    else:
        source = gui.sources_checked[0][0] if len(gui.sources_checked) == 1 else None
    if source is None:
        return None, None
    if selected is not None and selected[0] == source:
        return source, selected[1]
    mine = [name for src, name in gui.scenarios_checked if src == source]
    return source, (mine[0] if len(mine) == 1 else None)


def gui_snapshot(project: str, gui: GuiState | None, *, taken_at: str | None = None) -> ContextSnapshot:
    """The snapshot of a question sent from the FlexTool window: its selections and the defaults they give."""
    when = taken_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if gui is None:
        return ContextSnapshot(project=project, front_end="gui", taken_at=when)
    source, scenario = primary_defaults(gui)
    return ContextSnapshot(project=project, input_source=source, scenario=scenario, front_end="gui", gui=gui,
                           taken_at=when)


def _payload_text(value: object, redact: Callable[[str], str] | None) -> str | None:
    """A name for the payload: redacted, then cut (a cut first could leave part of a folder name).  Redacted here,
    the text is :class:`~flextool.assistant.caps.Rendered`, so ``fit_json`` does not redact it again."""
    if value is None:
        return None
    text = clean(str(value))
    if redact is None:
        return cap_text(text, NAME_MAX)
    return Rendered(cap_text(redact(text), NAME_MAX))


def _payload_list(out: dict[str, Any], key: str, items: Iterable[Any], render: Callable[[Any], Any]) -> None:
    items = list(items)
    out[key] = [render(item) for item in items[:PAYLOAD_LIST_MAX]]
    if len(items) > PAYLOAD_LIST_MAX:
        out[f"{key}_more"] = len(items) - PAYLOAD_LIST_MAX


def _result_payload(ref: ResultRef, redact: Callable[[str], str] | None) -> dict[str, Any]:
    return {"folder": _payload_text(ref.folder, redact), "source_number": ref.source_number,
            "scenario": _payload_text(ref.scenario, redact), "state": ref.state,
            "results_usable": ref.results_usable, "unusable_code": ref.unusable_code,
            "reason": _payload_text(ref.reason, redact)}


def _viewer_payload(v: ViewerState, redact: Callable[[str], str] | None) -> dict[str, Any]:
    def text(value: object) -> str | None:
        return _payload_text(value, redact)

    out: dict[str, Any] = {"mode": text(v.mode)}
    _payload_list(out, "scenarios", v.scenarios, text)
    out.update({"plot_entry": text(v.plot_entry), "plot_name": text(v.plot_name), "variant": text(v.variant),
                "result_key": text(v.result_key), "sub_config": text(v.sub_config), "time_start": v.time_start,
                "time_steps": v.time_steps, "time_start_label": text(v.time_start_label),
                "dispatch_item": text(v.dispatch_item)})
    _payload_list(out, "comparison_checked", v.comparison_checked, text)
    return out


def snapshot_payload(s: ContextSnapshot | None, *, gui_running: str,
                     redact: Callable[[str], str] | None = None) -> dict[str, Any]:
    """The snapshot as plain data (JSON-ready): names and selections only, no paths and no data values.

    *gui_running* is ``"running"`` or ``"not running"``.  Every string is redacted (*redact*) and cut to
    :data:`~flextool.assistant.caps.NAME_MAX` characters; each list keeps :data:`PAYLOAD_LIST_MAX` entries and
    counts the rest in ``<list>_more``.  The result rows carry what their run record says (``results_usable``,
    ``unusable_code``, ``reason``), never a rule of their own (D5).
    """
    out: dict[str, Any] = {"gui": gui_running}
    if s is None:
        out.update({"front_end": None, "project": None, "input_source": None, "scenario": None})
        return out

    def text(value: object) -> str | None:
        return _payload_text(value, redact)

    out.update({"front_end": s.front_end, "project": text(s.project), "input_source": text(s.input_source),
                "scenario": text(s.scenario), "taken_at": s.taken_at or None})
    gui = s.gui
    if gui is None:
        return out
    _payload_list(out, "sources_checked", gui.sources_checked,
                  lambda item: {"name": text(item[0]), "number": item[1]})
    out["source_selected"] = text(gui.source_selected)
    _payload_list(out, "scenarios_checked", gui.scenarios_checked,
                  lambda item: {"source": text(item[0]), "scenario": text(item[1])})
    selected = gui.scenario_selected
    out["scenario_selected"] = None if selected is None else {"source": text(selected[0]),
                                                              "scenario": text(selected[1])}
    _payload_list(out, "results_checked", gui.results_checked, lambda ref: _result_payload(ref, redact))
    _payload_list(out, "results_selected", gui.results_selected, lambda ref: _result_payload(ref, redact))
    out["viewer"] = None if gui.viewer is None else _viewer_payload(gui.viewer, redact)
    return out


@dataclass
class AssistantContext:
    project_path: Path
    project_name: str
    default_source: str | None
    default_scenario: str | None
    caps: Caps
    front_end: FrontEnd
    setup_problem: str | None = None
    lock: threading.Lock = field(default_factory=lambda: TOOL_LOCK)
    notes_sent: set[str] = field(default_factory=set)       # long notes are sent once per session
    gui_snapshot: ContextSnapshot | None = None              # the FlexTool window's snapshot of the current question
    redactor: Redactor = field(init=False)

    def __post_init__(self) -> None:
        self.redactor = Redactor(self.project_path)

    def redact(self, text: str) -> str:
        return self.redactor.redact(text)

    def snapshot(self) -> ContextSnapshot:
        """The window's snapshot of the current question, else one from this context's defaults."""
        if self.gui_snapshot is not None:
            return self.gui_snapshot
        return ContextSnapshot(project=self.project_name, input_source=self.default_source,
                               scenario=self.default_scenario, front_end=self.front_end)


def make_context(project_path: Path | str, *, input_source: str | None = None, scenario: str | None = None,
                 result_cap: str = DEFAULT_PRESET, front_end: FrontEnd = "agent") -> AssistantContext:
    """The context of a connection to the project at *project_path* (an absolute folder)."""
    path = Path(project_path).absolute()
    return AssistantContext(
        project_path=path,
        project_name=path.name,
        default_source=input_source,
        default_scenario=scenario,
        caps=preset(result_cap),
        front_end=front_end,
    )
