"""Run ``tools/lp_snapshot.py`` from a test: the plain (Utf8) and the axis-Enum LP of the same fixtures.

``snapshot(out_dir, fixtures, axis_enums=...)`` runs ``lp_snapshot.py snapshot`` in a subprocess with
``FLEXTOOL_AXIS_ENUMS`` removed from the environment (the ``--axis-enums`` flag sets it in each worker), so the
plain run is a Utf8 run even when the calling shell has the switch on.  Each fixture's ``_run.log`` records its
worker exit code (:func:`fixture_exit`); :func:`compare` runs ``lp_snapshot.py compare`` (Tier A: every MPS file
and manifest byte-identical).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LP_SNAPSHOT = REPO / "tools" / "lp_snapshot.py"
_EXIT = re.compile(r"^exit=(-?\d+)$", re.MULTILINE)


def _env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k != "FLEXTOOL_AXIS_ENUMS"}


def snapshot(out_dir: Path, fixtures: tuple[str, ...], *, axis_enums: bool, jobs: int,
             timeout: float) -> subprocess.CompletedProcess:
    """``lp_snapshot.py snapshot [--axis-enums] -j jobs --fixtures ... out_dir``; non-zero rc if any fixture crashed."""
    cmd = [sys.executable, str(LP_SNAPSHOT), "snapshot", "-j", str(jobs), "--fixtures", ",".join(fixtures)]
    if axis_enums:
        cmd.append("--axis-enums")
    cmd.append(str(out_dir))
    return subprocess.run(cmd, cwd=REPO, env=_env(), capture_output=True, text=True, timeout=timeout)


def fixture_exit(out_dir: Path, fixture: str) -> tuple[int | None, str]:
    """Worker exit code of ``fixture`` in a snapshot (``None`` if it never ran) and the tail of its run log."""
    log = out_dir / fixture / "_run.log"
    if not log.is_file():
        return None, ""
    text = log.read_text()
    match = _EXIT.search(text)
    return (int(match.group(1)) if match else None), text[-3000:]


def compare(base_dir: Path, new_dir: Path, fixture: str, *, timeout: float = 300) -> subprocess.CompletedProcess:
    """``lp_snapshot.py compare --fixtures fixture base_dir new_dir`` (rc 0: Tier A identical, nothing missing)."""
    cmd = [sys.executable, str(LP_SNAPSHOT), "compare", "--fixtures", fixture, str(base_dir), str(new_dir)]
    return subprocess.run(cmd, cwd=REPO, env=_env(), capture_output=True, text=True, timeout=timeout)
