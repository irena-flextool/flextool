"""A stand-in for the engine in a pipe-owner test (standard library only).

    python _stub_engine.py <folder> <mode>

Like ``cmd_run_flextool`` with a pipe owner it writes a ``running`` ``_run_status.json`` with one running solve and
the ``_run.log`` header into *folder*, prints the marker, and then, by *mode*:

* ``segv``: prints a Python line, writes a line straight to fd 1 and segfaults with ``faulthandler`` on;
* ``sigkill``: kills itself with SIGKILL;
* ``exit0``: exits 0 without finishing its status;
* ``exit1``: prints a line and exits 1 without finishing its status;
* ``sleep``: prints a line and sleeps for a minute (until it is killed);
* ``stubborn``: ignores SIGINT and SIGTERM, prints a line and sleeps for a minute.

It exits 3 at once when ``FLEXTOOL_RUN_LOG_OWNER`` is not ``1`` (the owner did not set it).
"""
from __future__ import annotations

import faulthandler
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path


def main() -> None:
    folder, mode = Path(sys.argv[1]), sys.argv[2]
    if os.environ.get("FLEXTOOL_RUN_LOG_OWNER") != "1":
        sys.exit(3)
    folder.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    status = {
        "schema": "flextool.run_status", "schema_version": 1, "run_id": run_id, "state": "running",
        "verdict": None, "exit_code": None, "error": None, "started_at": started, "updated_at": started,
        "heartbeat_at": started, "finished_at": None, "wall_time_s": None,
        "outputs": {"parquet_state": "none", "written_at": None, "by": "run", "previous": None},
        "solves": [{"name": "s1", "index": 0, "kind": "lp", "state": "running"}],
    }
    (folder / "_run_status.json").write_text(json.dumps(status), encoding="utf-8")
    (folder / "_run.log").write_bytes(f"# FlexTool run {run_id} started {started}\n".encode("utf-8"))
    print("##flextool-run-record## " + json.dumps({"folder": str(folder), "run_id": run_id}), flush=True)
    if mode == "segv":
        print("python line ä", flush=True)
        os.write(1, b"native line\n")
        faulthandler.enable()
        faulthandler._sigsegv()
    elif mode == "sigkill":
        print("about to be killed", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
    elif mode == "exit0":
        print("done", flush=True)
        sys.exit(0)
    elif mode == "exit1":
        print("failing", flush=True)
        sys.exit(1)
    elif mode in ("sleep", "stubborn"):
        if mode == "stubborn":
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print("sleeping", flush=True)
        time.sleep(60)
    sys.exit(2)


if __name__ == "__main__":
    main()
