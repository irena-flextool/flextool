"""``flextool.run_warnings``: the registry, the aggregated console line, the solve and pass scopes, the tally, the
run record's entry of a coded warning and the generated documentation page."""
from __future__ import annotations

import contextlib
import logging
import re
import string
import sys

import pytest

from flextool import run_warnings
from flextool.docs_build import scope
from flextool.docs_build.parameters import WEB_PAGE_URL
from flextool.docs_build.warnings import render_warnings_page
from flextool.run_warnings.registry import RETIRED, SEVERITIES, WARNINGS, WARNINGS_DOCS_URL, WARNINGS_PAGE, docs_url

CODE = re.compile(r"^W-\d{4}$")
#: Codes whose line is longer than 100 characters on purpose: W-0541 quotes the HiGHS warnings it explains (the
#: owner's wording; printed once per run).
LONG_LINE_OK = {"W-0541"}


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
        line = run_warnings.format_line(code, [] if spec.entityless else ["a_name"], solve="s",
                                        **{f: "x" for f in fields - {"solve"}})
        assert len(line) <= 100 or code in LONG_LINE_OK, line


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


def test_an_empty_name_list_is_no_case_and_no_names_is_an_entityless_line(records):
    """B1: ``warn_grouped(code, [])`` (a check that found nothing) prints nothing and counts nothing, also inside a
    scope and for a note; leaving the names out gives the line without ``(...)``."""
    assert not run_warnings.warn_grouped("W-0201", [])
    assert not run_warnings.warn_grouped("W-0004", (), solve="s")
    assert not run_warnings.note("W-0229", [])
    with run_warnings.collecting():
        run_warnings.warn_grouped("W-0200", [])
    assert records() == [] and run_warnings.tally() == {} and run_warnings.tally_lines() == []
    assert run_warnings.warn_grouped("W-0001")
    assert run_warnings.warn("W-0002", parameter="print_dispatch")
    assert records(logging.WARNING) == [
        "Warning: flowGroup has flow_aggregator but no flow members [W-0001]",
        "Warning: group has print_dispatch: yes but no group__node members [W-0002]"]
    assert run_warnings.tally() == {"W-0001": 1, "W-0002": 1}


def test_capture_tells_an_empty_call_from_a_nameless_line():
    from tests._support.coded_warnings import capture_coded

    run_warnings.start_run()
    try:
        with capture_coded() as got:
            run_warnings.warn_grouped("W-0201", [])
            run_warnings.warn_grouped("W-0001")
        assert got.codes() == ["W-0001"] and got.entityless("W-0001")
        assert [c.code for c in got.empty_calls] == ["W-0201"]
        assert got.lines() == ["Warning: flowGroup has flow_aggregator but no flow members [W-0001]"]
    finally:
        run_warnings.start_run()


def test_the_record_carries_code_and_cases(caplog):
    caplog.set_level(logging.WARNING, logger=run_warnings.LOGGER_NAME)
    run_warnings.start_run()
    run_warnings.warn_grouped("W-0001", ["a", "b", "c"])
    (record,) = [r for r in caplog.records if r.name == run_warnings.LOGGER_NAME]
    assert getattr(record, run_warnings.CODE_ATTR) == "W-0001"
    assert getattr(record, run_warnings.CASES_ATTR) == 3
    assert record.filename == "test_run_warnings.py"          # the call site, not the helper (--debug format)


def test_a_scoped_line_carries_its_call_site(caplog):
    """m1: a line printed when its collecting() scope ends names the first call's file and line (``--debug``)."""
    caplog.set_level(logging.WARNING, logger=run_warnings.LOGGER_NAME)
    run_warnings.start_run()
    with run_warnings.collecting():
        run_warnings.warn("W-0001", "fg1"); line = sys._getframe().f_lineno  # noqa: E702
        run_warnings.warn("W-0001", "fg2")
    (record,) = [r for r in caplog.records if r.name == run_warnings.LOGGER_NAME]
    assert (record.filename, record.lineno) == ("test_run_warnings.py", line)
    assert record.funcName == "test_a_scoped_line_carries_its_call_site"


def test_a_later_roll_logs_only_the_details_of_its_new_names(records):
    run_warnings.warn_grouped("W-0004", ["c1"], solve="d_roll_1", details={"c1": "first"})
    run_warnings.warn_grouped("W-0004", ["c1", "c2"], solve="d_roll_2", details={"c1": "again", "c2": "new"})
    debug = records()
    assert "W-0004 c2: new" in debug and "W-0004 c1: again" not in debug


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


def _recorder(tmp_path):
    from flextool.run_record import RunRecorder
    from flextool.run_record._owner import write_status_atomic

    folder = tmp_path / "base"
    folder.mkdir()
    rec = RunRecorder()                                     # no heartbeat thread: nothing writes on its own
    rec.folder = folder
    rec._status = {"schema": "flextool.run_status", "schema_version": 1, "run_id": rec.run_id,
                   "state": "running", "updated_at": None, "outputs": {"parquet_state": "none"},
                   "solves": [], "solves_omitted": 0, "warnings": [], "warnings_omitted": 0}
    write_status_atomic(folder, rec._status)
    return rec


@contextlib.contextmanager
def _recording(rec):
    """Route the run's warnings to *rec* as ``cmd_run_flextool`` does (handler on the logger, cases listener)."""
    logger = logging.getLogger("flextool")                  # the parent of flextool.warnings and of the uncoded one
    handler = rec.warning_handler()
    logger.addHandler(handler)
    run_warnings.add_cases_listener(rec.add_warning_cases)
    run_warnings.start_run()
    try:
        yield
    finally:
        logger.removeHandler(handler)
        run_warnings.remove_cases_listener(rec.add_warning_cases)
        run_warnings.start_run()


def test_run_record_keeps_one_entry_per_code(tmp_path):
    """A coded warning takes one ``warnings`` entry whose ``cases`` is the tally's count (also the names a later
    roll of the same solve added without a line), with its distinct lines (at most MAX_CODE_MESSAGES) — so the
    per-solve lines of a long run do not fill the MAX_WARNINGS entries, and no solve's line is lost."""
    from flextool.run_record import read_status_file
    from flextool.run_record._status import MAX_CODE_MESSAGES, MAX_WARNINGS

    rec = _recorder(tmp_path)
    with _recording(rec):
        for k in range(MAX_WARNINGS + 5):
            run_warnings.warn_grouped("W-0004", ["c1", "c2"], solve=f"solve_{k}")
        run_warnings.warn_grouped("W-0004", ["c1", "c3"], solve="solve_0_roll_2")   # no line: +1 case
        run_warnings.warn_grouped("W-0001", ["fg"])
        run_warnings.warn_grouped("W-0002", ["g1"], parameter="print_dispatch")
        run_warnings.warn_grouped("W-0002", ["g2"], parameter="print_indicators")
        run_warnings.warn_grouped("W-0201", [])                                     # nothing found: no entry
        logging.getLogger("flextool.test.uncoded").warning("an uncoded warning")
        tally = run_warnings.tally()
    rec.finish("succeeded", exit_code=0, error=None)
    status = read_status_file(tmp_path / "base")
    coded = {w["code"]: w for w in status["warnings"] if "code" in w}
    assert set(coded) == {"W-0001", "W-0002", "W-0004"}
    w4 = coded["W-0004"]
    assert w4["cases"] == tally["W-0004"] == 2 * (MAX_WARNINGS + 5) + 1 and w4["lines"] == MAX_WARNINGS + 5
    assert w4["message"].endswith("in solve 'solve_0'; left out (2 cases, e.g. 'c1') [W-0004]")
    assert len(w4["messages"]) == MAX_CODE_MESSAGES and w4["messages"][0] == w4["message"]
    assert "'solve_1'" in w4["messages"][1]
    assert w4["messages_omitted"] == MAX_WARNINGS + 5 - MAX_CODE_MESSAGES
    assert [m.split(" has ")[1].split(":")[0] for m in coded["W-0002"]["messages"]] == [
        "print_dispatch", "print_indicators"]
    assert coded["W-0002"]["cases"] == 2 and coded["W-0001"]["cases"] == 1
    assert [w["message"] for w in status["warnings"] if "code" not in w] == ["an uncoded warning"]
    assert status["warnings_omitted"] == 0


def test_run_record_cap_has_no_stale_code_entry(tmp_path):
    """A coded entry that does not fit in ``warnings`` is counted once in ``warnings_omitted``; its later lines do
    not count again nor land in a dropped entry."""
    from flextool.run_record import read_status_file
    from flextool.run_record._status import MAX_WARNINGS

    rec = _recorder(tmp_path)
    with _recording(rec):
        for k in range(MAX_WARNINGS):
            logging.getLogger("flextool.test.uncoded").warning("uncoded %d", k)
        for k in range(4):
            run_warnings.warn_grouped("W-0004", ["c1"], solve=f"s{k}")
        rec._drain_warnings()
        run_warnings.warn_grouped("W-0001", ["fg"])
        run_warnings.warn_grouped("W-0001", ["fg2"], solve="x")
    rec.finish("succeeded", exit_code=0, error=None)
    status = read_status_file(tmp_path / "base")
    assert len(status["warnings"]) == MAX_WARNINGS and not any("code" in w for w in status["warnings"])
    assert status["warnings_omitted"] == 2                 # one per code, not one per line


def test_the_gui_links_codes_to_the_registry_page():
    """The GUI's run log links ``[W-dddd]`` through the registry (a standard-library module)."""
    pytest.importorskip("tkinter")
    from flextool.gui import log_view

    assert log_view.warning_docs_url("[W-0004]") == docs_url("W-0004")


def test_a_note_shows_without_logging_configured(monkeypatch, capsys):
    """m3: with no handler anywhere (the engine as a library), a note goes to stderr instead of vanishing."""
    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])
    logger = logging.getLogger(run_warnings.LOGGER_NAME)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "level", logging.INFO)
    run_warnings.start_run()
    try:
        run_warnings.note("W-0229", ["p1"])
    finally:
        run_warnings.start_run()
    assert "[W-0229]" in capsys.readouterr().err


def test_the_tally_counts_the_uncoded_warnings_too(records):
    """m11: an uncoded ``logging.warning`` is not in the code list, but the tally says there were others."""
    logging.getLogger("flextool.test.uncoded").warning("an uncoded one")
    assert run_warnings.tally_lines() == ["Warnings: plus 1 other warning (no code; see the log above)"]
    run_warnings.warn_grouped("W-0001", ["a"])
    logging.getLogger("flextool.test.uncoded").error("another")
    assert run_warnings.tally_lines()[0] == (
        "Warnings: 1 kind, 1 case — W-0001 ×1; plus 2 other warnings (no code; see the log above)")
