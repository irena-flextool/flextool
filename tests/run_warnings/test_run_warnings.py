"""``flextool.run_warnings``: the registry, the aggregated console line, the solve and pass scopes, the tally, the
run record's entry of a coded warning and the generated documentation page."""
from __future__ import annotations

import logging
import re
import string

import pytest

from flextool import run_warnings
from flextool.docs_build import scope
from flextool.docs_build.parameters import WEB_PAGE_URL
from flextool.docs_build.warnings import render_warnings_page
from flextool.run_warnings.registry import RETIRED, SEVERITIES, WARNINGS, WARNINGS_DOCS_URL, WARNINGS_PAGE, docs_url

CODE = re.compile(r"^W-\d{4}$")


@pytest.fixture
def records(caplog):
    """The ``flextool.warnings`` records (DEBUG and up) of the test, in a fresh run."""
    caplog.set_level(logging.DEBUG, logger=run_warnings.LOGGER_NAME)
    run_warnings.start_run()
    yield lambda level=logging.DEBUG: [r.getMessage() for r in caplog.records
                                       if r.name == run_warnings.LOGGER_NAME and r.levelno >= level]
    run_warnings.start_run()


# -- registry ---------------------------------------------------------------------------------------------------


def test_registry_codes_and_fields():
    assert list(WARNINGS) == sorted(WARNINGS), "keep the registry in code order"
    for code, spec in WARNINGS.items():
        assert CODE.match(code) and spec.code == code
        assert code not in RETIRED
        assert spec.severity in SEVERITIES
        assert spec.message and spec.message[0].islower() and not spec.message.endswith("."), code
        assert spec.title and spec.topic and "**How to fix:**" in spec.body, code
        fields = {f for _, f, _, _ in string.Formatter().parse(spec.message) if f}
        line = run_warnings.format_line(code, ["a_name"], solve="s", **{f: "x" for f in fields - {"solve"}})
        assert len(line) <= 100, line


def test_docs_url_and_page_path_agree_with_the_docs_build():
    assert WARNINGS_PAGE == scope.WARNINGS_PAGE and WARNINGS_PAGE in scope.WEB_PAGES
    site = WEB_PAGE_URL.rsplit("parameter_catalogue/", 1)[0]
    assert WARNINGS_DOCS_URL == site + WARNINGS_PAGE.removesuffix(".md") + "/"
    assert docs_url("W-0004") == WARNINGS_DOCS_URL + "#w-0004"


def test_the_page_has_an_anchor_per_code():
    page = render_warnings_page()
    for code, spec in WARNINGS.items():
        assert f"### {code}\n\n**{spec.title}** ({spec.severity})" in page
        assert f"[{code}]\n```" in page


# -- the line ---------------------------------------------------------------------------------------------------


def test_line_formats():
    assert run_warnings.format_line("W-0001", []) == "Warning: flowGroup has flow_aggregator but no flow members [W-0001]"
    assert run_warnings.format_line("W-0001", ["fg"]) == (
        "Warning: flowGroup has flow_aggregator but no flow members ('fg') [W-0001]")
    assert run_warnings.format_line("W-0001", ["b", "a", "b"]) == (
        "Warning: flowGroup has flow_aggregator but no flow members (2 cases, e.g. 'b') [W-0001]")
    assert run_warnings.format_line("W-0003", [("unit", "node")]).endswith("('unit, node') [W-0003]")
    long = "x" * 60
    assert f"'{'x' * (run_warnings.MAX_NAME - 1)}…'" in run_warnings.format_line("W-0001", [long])
    with pytest.raises(KeyError, match="registry"):
        run_warnings.format_line("W-9999", [])


def test_unknown_code_and_note_of_a_warning_raise():
    with pytest.raises(KeyError):
        run_warnings.warn_grouped("W-9999", ["a"])
    with pytest.raises(ValueError, match="not a note"):
        run_warnings.note("W-0001", ["a"])


def test_one_line_with_the_full_list_and_details_in_the_debug_log(records):
    assert run_warnings.warn_grouped("W-0001", ["a", "b"], details={"a": "both", "b": "both"})
    assert records(logging.WARNING) == [
        "Warning: flowGroup has flow_aggregator but no flow members (2 cases, e.g. 'a') [W-0001]"]
    assert records()[1:] == ["W-0001 all cases: a; b", "W-0001 a: both", "W-0001 b: both"]


def test_the_record_carries_code_and_cases(caplog):
    caplog.set_level(logging.WARNING, logger=run_warnings.LOGGER_NAME)
    run_warnings.start_run()
    run_warnings.warn_grouped("W-0001", ["a", "b", "c"])
    (record,) = [r for r in caplog.records if r.name == run_warnings.LOGGER_NAME]
    assert getattr(record, run_warnings.CODE_ATTR) == "W-0001"
    assert getattr(record, run_warnings.CASES_ATTR) == 3
    assert record.filename == "test_run_warnings.py"          # the call site, not the helper (--debug format)


# -- scopes -----------------------------------------------------------------------------------------------------


def test_once_per_complete_solve(records):
    """The rolls of a solve are one solve: the second roll's call adds its new names to the debug log only; another
    solve gets its own line."""
    assert run_warnings.warn_grouped("W-0004", ["c1", "c2"], solve="dispatch_roll_1")
    assert not run_warnings.warn_grouped("W-0004", ["c2", "c3"], solve="dispatch_roll_2")
    assert run_warnings.warn_grouped("W-0004", ["c1"], solve="invest")
    assert records(logging.WARNING) == [
        "Warning: user constraint has no variable in solve 'dispatch'; left out (2 cases, e.g. 'c1') [W-0004]",
        "Warning: user constraint has no variable in solve 'invest'; left out ('c1') [W-0004]",
    ]
    assert "W-0004 also in solve 'dispatch': c3" in records()
    assert run_warnings.tally() == {"W-0004": 4}


def test_per_run_names_each_entity_once(records):
    assert run_warnings.warn_grouped("W-0004", ["c1", "c2"], solve="a", per_run=True)
    assert not run_warnings.warn_grouped("W-0004", ["c1"], solve="b", per_run=True)
    assert run_warnings.warn_grouped("W-0004", ["c1", "c3"], solve="c", per_run=True)
    assert records(logging.WARNING)[-1] == (
        "Warning: user constraint has no variable in solve 'c'; left out ('c3') [W-0004]")


def test_collecting_joins_the_calls_of_a_pass(records):
    """A check that finds entities one at a time: one line per code and fmt when the pass ends, also when it ends
    by an exception; nested scopes print at the outermost end."""
    with pytest.raises(RuntimeError):
        with run_warnings.collecting():
            for name in ("g1", "g2", "g3"):
                assert not run_warnings.warn("W-0002", name, parameter="print_dispatch")
            with run_warnings.collecting():
                run_warnings.warn("W-0002", "g4", parameter="print_indicators")
                run_warnings.warn("W-0001", "fg")
            assert records() == []
            raise RuntimeError("the pass stops")
    assert records(logging.WARNING) == [
        "Warning: group has print_dispatch: yes but no group__node members (3 cases, e.g. 'g1') [W-0002]",
        "Warning: group has print_indicators: yes but no group__node members ('g4') [W-0002]",
        "Warning: flowGroup has flow_aggregator but no flow members ('fg') [W-0001]",
    ]
    run_warnings.warn("W-0001", "after")                     # the scope is closed: prints at once
    assert records(logging.WARNING)[-1].endswith("('after') [W-0001]")


def test_tally_lines(records):
    assert run_warnings.tally_lines() == []
    run_warnings.warn_grouped("W-0001", ["a"])
    run_warnings.warn_grouped("W-0004", ["c1", "c2", "c3"], solve="s")
    assert run_warnings.tally_lines() == [
        "Warnings: 2 kinds, 4 cases — W-0004 ×3, W-0001 ×1",
        f"Warnings explained: {WARNINGS_DOCS_URL}",
    ]
    run_warnings.start_run()
    assert run_warnings.tally_lines() == []


# -- run record -------------------------------------------------------------------------------------------------


def test_run_record_keeps_one_entry_per_code(tmp_path):
    """A coded warning takes one ``warnings`` entry: its later lines add to ``cases`` / ``lines``, so the per-solve
    lines of a long run do not fill the MAX_WARNINGS entries."""
    from flextool.run_record import RunRecorder, read_status_file
    from flextool.run_record._owner import write_status_atomic
    from flextool.run_record._status import MAX_WARNINGS

    folder = tmp_path / "base"
    folder.mkdir()
    rec = RunRecorder()                                     # no heartbeat thread: nothing writes on its own
    rec.folder = folder
    rec._status = {"schema": "flextool.run_status", "schema_version": 1, "run_id": rec.run_id,
                   "state": "running", "updated_at": None, "outputs": {"parquet_state": "none"},
                   "solves": [], "solves_omitted": 0, "warnings": [], "warnings_omitted": 0}
    write_status_atomic(folder, rec._status)
    logger = logging.getLogger("flextool")                  # the parent of flextool.warnings and of the uncoded one
    handler = rec.warning_handler()
    logger.addHandler(handler)
    run_warnings.start_run()
    try:
        for k in range(MAX_WARNINGS + 5):
            run_warnings.warn_grouped("W-0004", ["c1", "c2"], solve=f"solve_{k}")
        run_warnings.warn_grouped("W-0001", ["fg"])
        logging.getLogger("flextool.test.uncoded").warning("an uncoded warning")
    finally:
        logger.removeHandler(handler)
        run_warnings.start_run()
    rec.finish("succeeded", exit_code=0, error=None)
    status = read_status_file(tmp_path / "base")
    coded = {w["code"]: w for w in status["warnings"] if "code" in w}
    assert coded["W-0004"]["cases"] == 2 * (MAX_WARNINGS + 5) and coded["W-0004"]["lines"] == MAX_WARNINGS + 5
    assert coded["W-0004"]["message"].endswith("in solve 'solve_0'; left out (2 cases, e.g. 'c1') [W-0004]")
    assert coded["W-0001"]["cases"] == 1
    assert [w["message"] for w in status["warnings"] if "code" not in w] == ["an uncoded warning"]
    assert status["warnings_omitted"] == 0


def test_the_gui_links_codes_to_the_registry_page():
    """The GUI's run log links ``[W-dddd]`` with its own copy of the URL (TODO(P3): import it from
    ``flextool.run_warnings.registry``, which needs no engine); until then the two must agree."""
    pytest.importorskip("tkinter")
    from flextool.gui import log_view

    assert log_view.WARNINGS_DOCS_URL == WARNINGS_DOCS_URL
    assert log_view.warning_docs_url("[W-0004]") == docs_url("W-0004")
