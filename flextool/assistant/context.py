"""What every tool call knows: the project, the defaults the connection was started with, the caps and the
redaction, and what the session has already been told."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from flextool.assistant.caps import DEFAULT_PRESET, Caps, preset
from flextool.assistant.redact import Redactor

FrontEnd = Literal["agent", "terminal", "gui"]


@dataclass
class AssistantContext:
    project_path: Path
    project_name: str
    default_source: str | None
    default_scenario: str | None
    caps: Caps
    front_end: FrontEnd
    setup_problem: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    notes_sent: set[str] = field(default_factory=set)       # long notes are sent once per session
    redactor: Redactor = field(init=False)

    def __post_init__(self) -> None:
        self.redactor = Redactor(self.project_path)

    def redact(self, text: str) -> str:
        return self.redactor.redact(text)


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
