"""What the FlexTool window shows, for the AI assistant (P2 §6.2): the snapshot and the runtime-context publisher.

:func:`snapshot_from_main` reads the main window's selections on the Tk thread — the input sources checked and
highlighted, the available scenarios checked and highlighted, the result rows checked and highlighted (with what
their run record says, D5) and the open result viewer's :meth:`~flextool.gui.result_viewer.ResultViewer.view_state`.
Every part is read on its own: one that cannot be read now (a tree being rebuilt, a window going away) is empty and
logged once, the others still come.  Reading three trees of 500 rows costs about 2 ms (N3).

:class:`ContextPublisher` polls the snapshot once a second while a project is open and hands it to the runtime
context writer only when something changed (P2-D4: no hooks in the main window, no callbacks in the viewer).  Each
Send takes its own snapshot through the same function (P2-D3).
"""
from __future__ import annotations

import logging
import tkinter as tk
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flextool.assistant.context import ContextSnapshot, GuiState, ResultRef, ViewerState
    from flextool.assistant.runtime_context import RuntimeContextWriter

logger = logging.getLogger(__name__)

#: How often the publisher looks at the window.
POLL_MS = 1000
#: The errors a part of the window may raise while it is being rebuilt or closed.
_READ_ERRORS = (tk.TclError, AttributeError, LookupError, TypeError, ValueError)
_logged: set[str] = set()


def _log_once(part: str) -> None:
    """Log a part that could not be read, once per part and process (the poll would repeat it every second)."""
    if part in _logged:
        logger.debug("The FlexTool window's %s could not be read", part, exc_info=True)
        return
    _logged.add(part)
    logger.warning("The FlexTool window's %s could not be read for the AI assistant", part, exc_info=True)


def _check_on() -> str:
    from flextool.gui.scenario_lists import CHECK_ON

    return CHECK_ON


def _input_sources(main: Any) -> tuple[tuple[tuple[str, int], ...], str | None]:
    """(name, number) of the checked live input sources, and the highlighted one's name."""
    from flextool.gui.main_window import _is_ghost_iid, _source_name_from_iid

    tree = main.input_sources_tree
    on = _check_on()
    checked: list[tuple[str, int]] = []
    for iid in tree.get_children():
        if _is_ghost_iid(iid):
            continue
        values = tree.item(iid, "values")
        if values and values[0] == on:
            checked.append((_source_name_from_iid(iid), int(values[2])))
    selected = next((_source_name_from_iid(iid) for iid in tree.selection() if not _is_ghost_iid(iid)), None)
    return tuple(checked), selected


def _source_names(main: Any) -> dict[int, str]:
    """Input source number → name, from the available scenarios and the project's source numbers."""
    names: dict[int, str] = {}
    settings = getattr(main, "project_settings", None)
    for name, number in (getattr(settings, "input_source_numbers", None) or {}).items():
        names[int(number)] = name
    manager = getattr(main, "avail_scenario_mgr", None)
    if manager is not None:
        for info in manager.scenarios:
            names[info.source_number] = info.source_name
    return names


def _scenarios(main: Any) -> tuple[tuple[tuple[str, str], ...], tuple[str, str] | None]:
    """(source, scenario) of the available scenarios checked for running, and of the highlighted one."""
    tree = main.available_tree
    names = _source_names(main)
    on = _check_on()

    def pair(values: Any) -> tuple[str, str]:
        number = int(values[1])
        return names.get(number, str(number)), str(values[2])

    checked = tuple(pair(values) for iid in tree.get_children()
                    if (values := tree.item(iid, "values")) and values[0] == on)
    selected = None
    for iid in tree.selection():
        values = tree.item(iid, "values")
        if values and len(values) >= 3:
            selected = pair(values)
            break
    return checked, selected


def _result_ref(main: Any, iid: str) -> ResultRef | None:
    """The result row *iid* with what its run record says (``None`` for a row without its scan record)."""
    from flextool import run_record
    from flextool.assistant.context import ResultRef
    from flextool.gui.scenario_key import resolve_subdir_for_read

    info = (getattr(main, "_executed_row_info", None) or {}).get(iid)
    if info is None:
        return None
    owners = getattr(getattr(main, "project_settings", None), "bare_output_owners", None) or {}
    return ResultRef(folder=resolve_subdir_for_read(owners, info.source_number, info.name),
                     source_number=info.source_number, scenario=info.name, state=info.run_state,
                     results_usable=bool(info.checkable),
                     unusable_code=run_record.unusable_code(info.status, info.run_state),
                     reason=info.unusable_reason)


def _results(main: Any) -> tuple[tuple[ResultRef, ...], tuple[ResultRef, ...]]:
    """The result rows checked (in tree order) and highlighted."""
    tree = main.executed_tree
    on = _check_on()
    checked = [ref for iid in tree.get_children()
               if (values := tree.item(iid, "values")) and values[0] == on
               and (ref := _result_ref(main, iid)) is not None]
    selected = [ref for iid in tree.selection() if (ref := _result_ref(main, iid)) is not None]
    return tuple(checked), tuple(selected)


def _viewer(main: Any) -> ViewerState | None:
    viewer = getattr(main, "_result_viewer", None)
    if viewer is None or not viewer.winfo_exists():
        return None
    return viewer.view_state()


def snapshot_from_main(main: Any) -> GuiState | None:
    """The main window's selections now (Tk thread); ``None`` when no project is open."""
    from flextool.assistant.context import GuiState

    if not getattr(main, "current_project", None):
        return None
    sources: tuple[tuple[str, int], ...] = ()
    source_selected = None
    scenarios: tuple[tuple[str, str], ...] = ()
    scenario_selected = None
    results_checked: tuple[ResultRef, ...] = ()
    results_selected: tuple[ResultRef, ...] = ()
    viewer = None
    try:
        sources, source_selected = _input_sources(main)
    except _READ_ERRORS:
        _log_once("input sources")
    try:
        scenarios, scenario_selected = _scenarios(main)
    except _READ_ERRORS:
        _log_once("available scenarios")
    try:
        results_checked, results_selected = _results(main)
    except _READ_ERRORS:
        _log_once("results")
    try:
        viewer = _viewer(main)
    except _READ_ERRORS:
        _log_once("result viewer")
    return GuiState(sources_checked=sources, source_selected=source_selected, scenarios_checked=scenarios,
                    scenario_selected=scenario_selected, results_checked=results_checked,
                    results_selected=results_selected, viewer=viewer)


def context_snapshot(main: Any, project_name: str | None = None) -> ContextSnapshot:
    """The snapshot of a question sent now (P2-D3): the window's selections and the defaults they give."""
    from flextool.assistant.context import gui_snapshot

    name = project_name or getattr(main, "current_project", None) or ""
    return gui_snapshot(name, snapshot_from_main(main))


class ContextPublisher:
    """Publishes the window's selections to *writer* once a second while it runs, only when they changed.

    *main* is the main window (its ``after`` runs the poll); *snapshot(main)* reads the selections (default
    :func:`snapshot_from_main`).  :meth:`stop` ends the poll; the writer is the caller's to stop.
    """

    def __init__(self, main: tk.Misc, writer: RuntimeContextWriter, project_name: str, *,
                 interval_ms: int = POLL_MS, snapshot: Callable[[Any], GuiState | None] | None = None) -> None:
        self.main = main
        self.writer = writer
        self.project_name = project_name
        self.interval_ms = interval_ms
        self._snapshot = snapshot or snapshot_from_main
        self._after_id: str | None = None
        self._last: dict[str, Any] | None = None
        self.running = False
        #: How many payloads were handed to the writer (for the tests).
        self.published = 0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._after_id = self.main.after(0, self._tick)

    def stop(self) -> None:
        self.running = False
        if self._after_id is not None:
            try:
                self.main.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def publish_now(self) -> bool:
        """Read the window and publish when the payload changed; ``True`` when it did."""
        from flextool.assistant.context import gui_snapshot, snapshot_payload

        snapshot = gui_snapshot(self.project_name, self._snapshot(self.main))
        # Compared without the time it was taken: an unchanged window is not written again.
        payload = snapshot_payload(snapshot, gui_running="running")
        payload.pop("taken_at", None)
        if payload == self._last:
            return False
        self._last = payload
        self.writer.publish_snapshot(snapshot)
        self.published += 1
        return True

    def _tick(self) -> None:
        self._after_id = None
        if not self.running:
            return
        try:
            self.publish_now()
        except Exception:  # noqa: BLE001 — the file keeps its last payload; the poll goes on
            _log_once("selections")
        try:
            self._after_id = self.main.after(self.interval_ms, self._tick)
        except tk.TclError:             # the main window is gone
            self.running = False
