"""Control characters from names in the input data reach no console or log window as commands
(``flextool._safe_text``): ``flextool-run``'s relay, the engine's own console copy and its console log handler, and
the FlexTool window's job log show them as ``\\xNN``.
"""
from __future__ import annotations

import io
import logging
import sys

import flextool_run_supervisor as supervisor
from flextool._safe_text import PrintableFormatter, PrintableText, make_printable, printable
from flextool.run_record import RunLogWriter, install_tee
from flextool.run_record._owner import write_log_header

NAME = "node\x1b]52;c;aGk=\x07\x1b[2J\x9b1m"


def test_printable_shows_control_characters_and_keeps_text() -> None:
    assert printable(NAME) == "node\\x1b]52;c;aGk=\\x07\\x1b[2J\\x9b1m"
    assert printable("a\tb ä 風") == "a\tb ä 風"
    assert printable("a\nb") == "a\\x0ab"
    assert printable("a\r\nb\rc\n", keep_newlines=True) == "a\nb\\x0dc\n"


def test_bidi_controls_are_shown_and_joiners_kept() -> None:
    assert printable("ok\u202eevil\u2066x\u2069\u2028") == "ok\\u202eevil\\u2066x\\u2069\\u2028"
    assert printable("\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645") == "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"


def test_a_windows_line_end_split_across_writes_stays_a_line_end() -> None:
    shown = PrintableText()
    assert shown("x\r") + shown("\n") == "x\n"
    assert shown("a\r") + shown("b\n") == "a\\x0db\n"  # a carriage return that is not a line end is shown


def test_the_console_log_handler_shows_them(tmp_path) -> None:
    stream = io.StringIO()
    handler = make_printable(logging.StreamHandler(stream))
    assert isinstance(handler.formatter, PrintableFormatter) and make_printable(handler).formatter is handler.formatter
    logger = logging.getLogger("flextool.test_safe_text")
    logger.addHandler(handler)
    try:
        logger.warning("unit %s has no capacity\nsecond line", NAME)
    finally:
        logger.removeHandler(handler)
    assert stream.getvalue() == f"unit {printable(NAME)} has no capacity\nsecond line\n"


def test_the_engine_console_copy_shows_them(tmp_path) -> None:
    log_path = tmp_path / "_run.log"
    write_log_header(log_path, "r1", "2026-11-12T10:00:00+02:00")
    writer = RunLogWriter(log_path, head=1 << 20, tail=1 << 20)
    inner = io.StringIO()
    old = sys.stdout
    sys.stdout = inner
    try:
        restore = install_tee(writer)
        print(f"Solve {NAME}")
        restore()
    finally:
        sys.stdout = old
        writer.close()
    assert inner.getvalue() == f"Solve {printable(NAME)}\n"
    assert f"Solve {printable(NAME)}" in log_path.read_text(encoding="utf-8").splitlines()
    assert "\x1b" not in log_path.read_text(encoding="utf-8")


def test_flextool_run_relays_them_shown(tmp_path) -> None:
    echo = io.StringIO()
    code = f"print({NAME!r} + ' warning', flush=True)"
    assert supervisor.run_supervised([], child_cmd=[sys.executable, "-c", code], stdout=echo) == 0
    # The child's line end is relayed as it is: "\r\n" on Windows.
    assert echo.getvalue().replace("\r\n", "\n") == printable(NAME) + " warning\n"


def test_the_window_job_log_shows_them(tmp_path) -> None:
    from flextool.gui.data_models import ProjectSettings
    from flextool.gui.execution_manager import ExecutionManager, JobType

    mgr = ExecutionManager(project_path=tmp_path, settings=ProjectSettings())
    job = mgr.add_auxiliary_job(JobType.OUTPUT_ACTION, "x", "x")
    mgr.append_stdout(job.job_id, f"Scenario {NAME}")
    assert job.stdout_lines == [f"Scenario {printable(NAME)}"]
