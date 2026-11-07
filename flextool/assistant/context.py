"""What every tool call knows: the project, the defaults the connection was started with, the caps and the
redaction, and what the session has already been told; and the snapshot of what the user has chosen, which the chat
puts in its context block.

:data:`TOOL_LOCK` is the one lock every context of this process uses by default, so tools never run at once
(the input-data cache is shared by the whole process; several chats, and a closing chat's worker, may call tools).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from flextool.assistant.caps import DEFAULT_PRESET, Caps, preset
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
