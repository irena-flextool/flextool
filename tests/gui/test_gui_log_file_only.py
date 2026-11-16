"""A record marked ``file_only`` reaches the GUI log file but not the console."""
import io
import logging
import sys

from flextool.gui import __main__ as gui_main


def test_file_only_record_skips_the_console(tmp_path, monkeypatch):
    console = io.StringIO()
    monkeypatch.setattr(sys, "stderr", console)
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers.clear()
    try:
        gui_main._configure_logging(tmp_path)
        log = logging.getLogger("flextool.gui.platform_utils")
        log.warning("DPI-DIAG: x", extra={"file_only": True})
        log.warning("ordinary warning")
        for h in root.handlers:
            h.flush()
        text = (tmp_path / "flextool_gui.log").read_text(encoding="utf-8")
    finally:
        for h in root.handlers:
            h.close()
        root.handlers[:] = saved
    assert "DPI-DIAG: x" in text and "ordinary warning" in text
    assert "DPI-DIAG" not in console.getvalue()
    assert "ordinary warning" in console.getvalue()
