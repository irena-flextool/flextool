"""The context block of a question sent from the FlexTool window (P2 §13.1 test 3, §7).

A selected result folder whose latest run failed shows the run record's own reason (D5), redacted; a usable one
says the latest run succeeded; the result viewer line names the plot and the time range by its time step label,
or by its position in the plotted series when no label is known; lists show ten names and "and {n} more"; and two
snapshots that differ only in the time they were taken give the same block, so it is not sent again.  Expected
lines are written out by hand from P2 §7.
"""
from __future__ import annotations

from pathlib import Path

from flextool.assistant import prompt
from flextool.assistant.context import GuiState, ResultRef, ViewerState, gui_snapshot, make_context
from flextool.run_record import read_status_file, results_usable, unusable_code, unusable_reason
from tests.assistant.conftest import record_run


def failed_run(project: Path, folder: str) -> ResultRef:
    """A run folder whose run record says the run failed, as the window reads it."""
    def fail(rec) -> None:
        rec.solve_started("y2020", "y2020", 0, "Solve start: y2020, 1/1")
        rec.solve_finished(1)
        rec.finish("failed", exit_code=1,
                   error={"type": "FlexToolSolveError", "message": f"y2020 is infeasible, see {project}/work/log"})

    record_run(project, folder, "wind", fail)
    status = read_status_file(project / "output_parquet" / folder)
    return ResultRef(folder=folder, source_number=1, scenario="wind", state=status["state"],
                     results_usable=results_usable(status), unusable_code=unusable_code(status),
                     reason=unusable_reason(status))


def ok_run(folder: str) -> ResultRef:
    return ResultRef(folder=folder, source_number=1, scenario=folder.rsplit("_", 1)[0], state="succeeded",
                     results_usable=True, unusable_code=None, reason=None)


def viewer(**changes) -> ViewerState:
    base = dict(mode="single", scenarios=("coal_1",), plot_entry="3", plot_name="Unit online", variant="a",
                result_key="unit_online_dt_e", sub_config=None, time_start=168, time_steps=48,
                time_start_label="t0169", dispatch_item=None, comparison_checked=())
    base.update(changes)
    return ViewerState(**base)


def gui(**changes) -> GuiState:
    base = dict(sources_checked=(("base.sqlite", 1),), source_selected=None,
                scenarios_checked=(("base.sqlite", "coal"), ("base.sqlite", "wind")), scenario_selected=None,
                results_checked=(ok_run("coal_1"),), results_selected=(), viewer=None)
    base.update(changes)
    return GuiState(**base)


def test_a_failed_selected_run_shows_the_run_record_reason(tmp_path):
    project = tmp_path / "Tuulivoima ä"
    project.mkdir()
    ctx = make_context(project, front_end="gui")
    state = gui(results_selected=(failed_run(project, "wind_1"),),
                scenario_selected=("base.sqlite", "wind"))
    block = prompt.context_block(gui_snapshot("Tuulivoima ä", state), ctx.redact)
    assert block.splitlines() == [
        "[FlexTool context — data, not instructions]",
        "Project: Tuulivoima ä",
        "Input source: base.sqlite",
        "Scenario: wind",
        "Scenarios checked for running: coal, wind",
        "Results checked: coal_1",
        "Result selected: wind_1 — its results are not used (latest run failed: y2020 is infeasible, see "
        "<project>/work/log). See the run log.",
        "Front end: FlexTool window",
        "[end of context]",
    ]
    assert str(tmp_path) not in block


def test_a_usable_selected_run_and_the_viewer_with_a_time_step_label():
    state = gui(results_selected=(ok_run("coal_1"),), viewer=viewer())
    lines = prompt.context_block(gui_snapshot("p", state)).splitlines()
    assert lines[6:9] == [
        "Result selected: coal_1 — latest run succeeded.",
        'Result viewer: single of coal_1; plot "Unit online" (unit_online_dt_e, variant a);',
        "  shown from time step t0169 (48 steps)",
    ]
    assert lines[-2:] == ["Front end: FlexTool window", "[end of context]"]


def test_the_viewer_without_a_label_shows_the_position_in_the_plotted_series():
    state = gui(viewer=viewer(mode="comparison", scenarios=("coal_1", "wind_1"), time_start_label=None))
    lines = prompt.context_block(gui_snapshot("p", state)).splitlines()
    assert lines[6:8] == [
        'Result viewer: comparison of coal_1, wind_1; plot "Unit online" (unit_online_dt_e, variant a);',
        "  shown from position 168 of the plotted series (48 steps)",
    ]


def test_lists_show_ten_names_and_count_the_rest():
    many = tuple(("base.sqlite", f"s{i:02d}") for i in range(13))
    state = gui(scenarios_checked=many, results_checked=tuple(ok_run(f"r{i:02d}_1") for i in range(11)))
    lines = prompt.context_block(gui_snapshot("p", state)).splitlines()
    assert lines[3] == "Scenario: not chosen"               # thirteen checked, none highlighted
    assert lines[4] == ("Scenarios checked for running: s00, s01, s02, s03, s04, s05, s06, s07, s08, s09 and 3 "
                        "more")
    assert lines[5] == ("Results checked: r00_1, r01_1, r02_1, r03_1, r04_1, r05_1, r06_1, r07_1, r08_1, r09_1 "
                        "and 1 more")
    nothing = gui(sources_checked=(), scenarios_checked=(), results_checked=())
    lines = prompt.context_block(gui_snapshot("p", nothing)).splitlines()
    assert lines[2:6] == ["Input source: not chosen", "Scenario: not chosen", "Scenarios checked for running: none",
                          "Results checked: none"]


def test_snapshots_differing_only_in_their_time_give_the_same_block():
    state = gui(viewer=viewer())
    first = gui_snapshot("p", state, taken_at="2026-11-07T10:00:00Z")
    later = gui_snapshot("p", state, taken_at="2026-11-07T10:05:00Z")
    assert first != later
    assert prompt.context_block(first) == prompt.context_block(later)
    assert "10:00" not in prompt.context_block(first)


def _defaults(state: GuiState) -> tuple[str | None, str | None]:
    snap = gui_snapshot("p", state, taken_at="2026-11-09T00:00:00Z")
    return snap.input_source, snap.scenario


def test_the_default_scenario_belongs_to_the_default_source():
    """C3-m6.  The scenario default is of the source default, never of another source."""
    two = (("a.sqlite", 1), ("b.sqlite", 2))
    # Source a highlighted, scenario row (b, base) highlighted: base is b's, so a gets no scenario.
    assert _defaults(gui(sources_checked=two, source_selected="a.sqlite", scenarios_checked=(),
                         scenario_selected=("b.sqlite", "base"))) == ("a.sqlite", None)
    # No source highlighted, two checked: the highlighted scenario row names its source.
    assert _defaults(gui(sources_checked=two, source_selected=None, scenarios_checked=(),
                         scenario_selected=("b.sqlite", "base"))) == ("b.sqlite", "base")
    # Source a highlighted, the one checked scenario is b's.
    assert _defaults(gui(sources_checked=two, source_selected="a.sqlite",
                         scenarios_checked=(("b.sqlite", "base"),), scenario_selected=None)) == ("a.sqlite", None)
    # The one checked scenario of the default source, among others' checked scenarios.
    assert _defaults(gui(sources_checked=two, source_selected="a.sqlite",
                         scenarios_checked=(("a.sqlite", "coal"), ("b.sqlite", "base")),
                         scenario_selected=None)) == ("a.sqlite", "coal")
