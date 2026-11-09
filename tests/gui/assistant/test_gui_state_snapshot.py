"""The FlexTool window's selections for the AI assistant (P2 §6.2): the snapshot of a fake main window and the 1 s
publisher that writes the runtime context file only when the selections change."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest

from flextool.assistant import _files
from flextool.assistant.context import ViewerState
from flextool.assistant.runtime_context import RuntimeContextWriter, runtime_path
from flextool.gui.assistant.gui_state import ContextPublisher, context_snapshot, snapshot_from_main
from flextool.gui.data_models import ExecutedScenarioInfo, ScenarioInfo
from tests.gui.assistant.tk_helpers import pump

pytestmark = pytest.mark.gui

ON, OFF = "■", "□"
FAILED = {"state": "failed", "error": {"message": "the solver found no solution"},
          "outputs": {"parquet_state": "previous_run"}}
SUCCEEDED = {"state": "succeeded", "outputs": {"parquet_state": "written"}}
VIEW = ViewerState(mode="single", scenarios=("base_1",), plot_entry="3", plot_name="Unit outputs", variant="a",
                   result_key="unit_outputNode_dt", sub_config=None, time_start=0, time_steps=168,
                   time_start_label="p2020 t0001", dispatch_item=None, comparison_checked=())


class FakeViewer:
    def __init__(self, state: ViewerState | None = VIEW, *, raises: bool = False) -> None:
        self.state = state
        self.raises = raises

    def winfo_exists(self) -> bool:
        return True

    def view_state(self) -> ViewerState | None:
        if self.raises:
            raise tk.TclError("invalid command name")
        return self.state


class FakeMain(tk.Toplevel):
    """The parts of the main window the snapshot reads, with the same row layouts."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.withdraw()
        self.current_project = "Project"
        self.input_sources_tree = ttk.Treeview(self, columns=("check", "name", "number", "status"))
        self.available_tree = ttk.Treeview(self, columns=("check", "number", "name"))
        self.executed_tree = ttk.Treeview(self, columns=("check", "number", "name", "view", "time"))
        self.project_settings = SimpleNamespace(input_source_numbers={"north.sqlite": 1, "south.xlsx": 2},
                                                bare_output_owners={"legacy": 1})
        self.avail_scenario_mgr = SimpleNamespace(scenarios=[ScenarioInfo("base", 1, "north.sqlite"),
                                                             ScenarioInfo("wet", 2, "south.xlsx")])
        self._executed_row_info: dict[str, ExecutedScenarioInfo] = {}
        self._result_viewer: FakeViewer | None = None
        sources = self.input_sources_tree
        sources.insert("", "end", iid="north.sqlite", values=(ON, "north.sqlite", 1, "✓"))
        sources.insert("", "end", iid="ext:south.xlsx", values=(OFF, "south.xlsx  →  ../south.xlsx", 2, "✓"))
        sources.insert("", "end", iid="ghost:3", values=("", "gone.sqlite", 3, "⊘"))
        available = self.available_tree
        available.insert("", "end", iid="a1", values=(ON, 1, "base"))
        available.insert("", "end", iid="a2", values=(ON, 2, "wet"))
        executed = self.executed_tree
        for iid, check, number, name, status, state, usable, reason in (
                ("r1", ON, 1, "base", SUCCEEDED, "succeeded", True, None),
                ("r2", OFF, 2, "wet", FAILED, "failed", False, "latest run failed: the solver found no solution"),
                ("r3", ON, 1, "legacy", None, None, True, None)):
            executed.insert("", "end", iid=iid, values=(check, number, name, "▶", "06.11.26 14:02"))
            self._executed_row_info[iid] = ExecutedScenarioInfo(
                name=name, source_number=number, timestamp="06.11.26 14:02", status=status, run_state=state,
                checkable=usable, unusable_reason=reason)


@pytest.fixture
def main(tk_root: tk.Tk):
    window = FakeMain(tk_root)
    yield window
    window.destroy()


def test_snapshot_reads_checks_selections_run_records_and_viewer(main: FakeMain) -> None:
    main.input_sources_tree.selection_set("ext:south.xlsx")
    main.available_tree.selection_set("a2")
    main.executed_tree.selection_set("r2")
    main._result_viewer = FakeViewer()
    gui = snapshot_from_main(main)
    assert gui is not None
    assert gui.sources_checked == (("north.sqlite", 1),)          # the ghost row is never a source
    assert gui.source_selected == "south.xlsx"                     # the ext: prefix is not part of the name
    assert gui.scenarios_checked == (("north.sqlite", "base"), ("south.xlsx", "wet"))
    assert gui.scenario_selected == ("south.xlsx", "wet")
    assert [r.folder for r in gui.results_checked] == ["base_1", "legacy"]   # a bare legacy folder keeps its name
    (failed,) = gui.results_selected
    assert (failed.folder, failed.state, failed.results_usable, failed.unusable_code) == (
        "wet_2", "failed", False, "failed")
    assert failed.reason == "latest run failed: the solver found no solution"
    assert gui.viewer == VIEW
    snapshot = context_snapshot(main)
    assert (snapshot.project, snapshot.input_source, snapshot.scenario, snapshot.front_end) == (
        "Project", "south.xlsx", "wet", "gui")


def test_a_part_that_cannot_be_read_leaves_the_others(main: FakeMain) -> None:
    main._result_viewer = FakeViewer(raises=True)
    gui = snapshot_from_main(main)
    assert gui is not None and gui.viewer is None
    assert gui.sources_checked == (("north.sqlite", 1),)
    main.current_project = None
    assert snapshot_from_main(main) is None


class RecordingWriter:
    def __init__(self) -> None:
        self.snapshots: list = []
        self.ticks: list[bool] = []

    def publish_snapshot(self, snapshot) -> None:  # noqa: ANN001
        self.snapshots.append(snapshot)

    def tick(self, active: bool = False) -> None:
        self.ticks.append(active)


def test_each_poll_ticks_the_writer_with_the_focus(main: FakeMain) -> None:
    """Review G18, C3-m5: every poll ticks the writer (its heartbeat goes on only then), also when nothing
    changed, and says whether FlexTool has the keyboard focus (the window in use)."""
    from tests.gui.assistant.tk_helpers import take_focus

    writer = RecordingWriter()
    publisher = ContextPublisher(main, writer, "Project", interval_ms=20)
    publisher.start()
    try:
        pump(main, lambda: len(writer.ticks) >= 5)
        assert len(writer.snapshots) == 1                # five polls, one change
        import time

        main.geometry("300x200+40+40")
        main.deiconify()
        pump(main, lambda: main.winfo_viewable())

        deadline = time.monotonic() + 5     # again while tests in other processes take the focus
        while writer.ticks[-1] is not True:
            assert time.monotonic() < deadline, "no tick with the focus"
            take_focus(main)
            pump(main, timeout=0.05)
    finally:
        publisher.stop()
    count = len(writer.ticks)
    pump(main, timeout=0.1)
    assert len(writer.ticks) == count                    # stopped: no more ticks


def test_publisher_writes_only_when_the_selections_change(main: FakeMain) -> None:
    writer = RecordingWriter()
    publisher = ContextPublisher(main, writer, "Project", interval_ms=20)
    publisher.start()
    try:
        pump(main, lambda: len(writer.snapshots) == 1)
        pump(main, timeout=0.2)                         # ten more polls of an unchanged window
        assert len(writer.snapshots) == 1
        main.available_tree.selection_set("a1")
        pump(main, lambda: len(writer.snapshots) == 2)
        assert writer.snapshots[-1].scenario == "base"
    finally:
        publisher.stop()
    count = len(writer.snapshots)
    main.available_tree.selection_set("a2")
    pump(main, timeout=0.1)
    assert len(writer.snapshots) == count               # stopped: nothing more


def test_publisher_feeds_the_runtime_file(main: FakeMain, tmp_path: Path) -> None:
    project = tmp_path / "Project"
    project.mkdir()
    writer = RuntimeContextWriter(project, "Project", heartbeat_s=60)
    writer.start()
    publisher = ContextPublisher(main, writer, "Project", interval_ms=20)
    path = runtime_path(project)
    try:
        main.executed_tree.selection_set("r2")
        publisher.start()

        def written() -> bool:
            try:
                data = _files.read_json(path)
            except (OSError, _files.UnreadableFile):
                return False
            return bool(data) and bool(data["context"].get("results_selected"))

        pump(main, written)
        context = _files.read_json(path)["context"]
        assert context["results_selected"][0]["unusable_code"] == "failed"
        assert context["scenarios_checked"][1] == {"source": "south.xlsx", "scenario": "wet"}
    finally:
        publisher.stop()
        assert writer.stop()
    assert not path.exists()
