"""TimingRecorder — per-CLI-invocation phase timings.

Writes a single structured CSV at
``<work_folder>/solve_data/timings.csv``.

Schema (CSV header)::

    phase,subphase,solve,roll_index,seconds,started_at_iso,cumulative_s

One row per phase. ``record(...)`` appends immediately so a crash mid-run
still leaves usable data. ``section(...)`` is a context-manager helper
that times an arbitrary block. ``finalize(output_dir)`` copies the file
to its final per-scenario output destination once the run completes.

Always-on; cheap (one stat + one append per phase). Constructed once per
CLI invocation; tests calling :class:`FlexToolRunner` directly bootstrap
their own recorder if the CLI did not pass one in.
"""
from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


_HEADER = (
    "phase",
    "subphase",
    "solve",
    "roll_index",
    "seconds",
    "started_at_iso",
    "cumulative_s",
)


class TimingRecorder:
    """One per CLI invocation; lives on ``FlexToolRunner.state``.

    Captures ``time.perf_counter()`` deltas with absolute UTC timestamps
    and cumulative-since-start seconds for every phase of a flextool run.
    """

    def __init__(self, work_folder: Path | str, scenario: Optional[str] = None) -> None:
        self.t0 = time.perf_counter()
        self.t0_wall = datetime.now(timezone.utc)
        self.work_folder = Path(work_folder)
        self.scenario = scenario  # may be set later if cli didn't have it yet
        self._path = self.work_folder / "solve_data" / "timings.csv"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._open_for_write()

    def _open_for_write(self) -> None:
        """Create the file (truncate any pre-existing one) and write the header."""
        with open(self._path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_HEADER)

    def set_scenario(self, name: str) -> None:
        """Set/overwrite the scenario name (CLI may not know it at __init__)."""
        self.scenario = name

    def record(
        self,
        phase: str,
        *,
        subphase: str = "",
        solve: str = "",
        roll_index: int | str = "",
        seconds: float,
        t_start: Optional[float] = None,
    ) -> None:
        """Append one row.

        ``t_start`` is the ``perf_counter()`` value at the START of the
        phase (used to back-compute ``started_at_iso``); if ``None`` we
        assume the phase ended just now (``started_at_iso = now - seconds``).
        """
        now_perf = time.perf_counter()
        cumulative_s = now_perf - self.t0
        if t_start is not None:
            started_perf = t_start
        else:
            started_perf = now_perf - max(0.0, float(seconds))
        # Map perf_counter → wall time via the recorded t0/t0_wall pair.
        started_wall = self.t0_wall + timedelta(seconds=started_perf - self.t0)
        started_iso = started_wall.isoformat()
        row = (
            str(phase),
            str(subphase),
            str(solve),
            str(roll_index),
            f"{float(seconds):.6f}",
            started_iso,
            f"{cumulative_s:.6f}",
        )
        # Atomic append — open/write/close per row so a crash leaves the
        # file in a parseable state.
        with open(self._path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @contextmanager
    def section(
        self,
        phase: str,
        *,
        subphase: str = "",
        solve: str = "",
        roll_index: int | str = "",
    ):
        """Time a block: ``with recorder.section('write_input'): ...``."""
        t_start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t_start
            self.record(
                phase,
                subphase=subphase,
                solve=solve,
                roll_index=roll_index,
                seconds=elapsed,
                t_start=t_start,
            )

    @property
    def path(self) -> Path:
        """Current location of the timings.csv (in the work folder)."""
        return self._path

    def finalize(self, output_dir: Path | str) -> None:
        """Copy ``timings.csv`` into the final scenario output dir.

        Lives alongside ``summary_solve.csv`` so users have one place to
        look. Idempotent: if the source file is missing, this is a no-op.
        """
        src = self._path
        if not src.exists():
            return
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / "timings.csv"
        # Plain byte copy — file is small.
        dst.write_bytes(src.read_bytes())
