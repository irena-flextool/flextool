"""The engine's console copy (``log_capture: "python"``): whole lines from concurrent threads, clean restore."""
from __future__ import annotations

import logging
import sys
import threading

from flextool.run_record import RunLogWriter, install_tee
from flextool.run_record._owner import write_log_header

N_LINES = 200


def test_print_and_logging_from_two_threads_land_as_whole_lines(tmp_path):
    log_path = tmp_path / "_run.log"
    write_log_header(log_path, "r1", "2026-11-06T10:00:00+02:00")
    root = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    writer = RunLogWriter(log_path, head=1 << 20, tail=1 << 20)
    try:
        restore = install_tee(writer)
        assert sys.stdout is not original_stdout and handler.stream is sys.stdout

        def work(tag: str) -> None:
            for i in range(N_LINES):
                # print() writes the text and the newline in two calls.
                print(f"print-{tag}-{i}")
                logging.getLogger("flextool.test_tee").warning("warn-%s-%d", tag, i)

        threads = [threading.Thread(target=work, args=(tag,)) for tag in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        restore()
        restore()  # idempotent
        assert sys.stdout is original_stdout and sys.stderr is original_stderr
        assert handler.stream is original_stdout
    finally:
        writer.close()
        root.removeHandler(handler)
        root.setLevel(old_level)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# FlexTool run r1 started 2026-11-06T10:00:00+02:00"
    expected = {f"{kind}-{tag}-{i}" for kind in ("print", "warn") for tag in ("a", "b") for i in range(N_LINES)}
    assert sorted(lines[1:]) == sorted(expected)
