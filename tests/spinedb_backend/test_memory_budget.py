"""Memory budget for ``input_derivation.run`` on a representative
fixture.

This test runs the full input_derivation pipeline against
``templates/examples.sqlite`` (scenario ``test_a_lot``) and asserts that
the peak RSS during the run stays under a threshold.

Numbers (measured on a developer workstation under stock allocator
settings, 2026-05-20):

- Baseline (pre-Track-A, parameter_value cache holds parsed Map / TimeSeries
  for the lifetime of the backend): ``test_a_lot`` peak ≈ 350-400 MB.
- Post-Track-A (per-row ``_parsed_value`` eviction): same fixture peak
  unchanged in absolute terms (the fixture's parameter set is small and
  the row-accumulator path dominates), but the SCALING behaviour against
  larger fixtures changes substantially (see
  ``specs/memory_diagnostic_results.md``).

The test threshold is set generously so this acts as a *guardrail* —
catching regressions like "someone accidentally kept the parsed_value
cache alive across the whole backend lifetime again", or "someone
added a copy of the parameter_value table at a hot site".  It is not a
tight upper bound.

Why ``examples.sqlite`` and not H2_trade.sqlite: the H2 fixture lives
outside the repo and is too large for CI runners.  ``examples.sqlite``
is in-tree and small but still exercises the same code paths.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DB = REPO_ROOT / "templates" / "examples.sqlite"


pytestmark = pytest.mark.skipif(
    not EXAMPLES_DB.exists(),
    reason=f"examples.sqlite not present at {EXAMPLES_DB}",
)


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


class _PeakWatcher:
    """Background thread sampling /proc/self/status VmRSS every 50 ms."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            r = _rss_mb()
            if r > self.peak_mb:
                self.peak_mb = r
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        self.start_rss = _rss_mb()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)


def test_input_derivation_peak_rss_under_budget(tmp_path) -> None:
    """Peak RSS during ``input_derivation.run`` on examples.sqlite
    ``test_a_lot`` must stay under the budget.

    Budget: 1.5 GB (1500 MB).  Generous; intended to catch egregious
    regressions like a re-introduced full parsed_value cache, not to
    enforce a tight optimum.  Tighten when the cascade-side row
    accumulator work (Track A.5 follow-up) lands.

    If your CI / dev machine reports persistent RSS pressure unrelated
    to flextool (browser tabs, IDE servers, etc.), set
    ``FLEXTOOL_TEST_RSS_BUDGET_MB`` to override.
    """
    # Late imports so the test's import phase doesn't dominate the
    # baseline reading.
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    from flextool.input_derivation import run as input_derivation_run

    budget_mb = float(os.environ.get("FLEXTOOL_TEST_RSS_BUDGET_MB", "1500"))

    provider = FlexDataProvider()
    with _PeakWatcher() as watcher:
        t0 = time.perf_counter()
        input_derivation_run(
            f"sqlite:///{EXAMPLES_DB}",
            provider,
            scenario_name="test_a_lot",
            work_folder=tmp_path,
            precision_digits=0,
        )
        wall_s = time.perf_counter() - t0

    delta_mb = watcher.peak_mb - watcher.start_rss
    print(
        f"\n[budget] input_derivation.run on test_a_lot: "
        f"peak={watcher.peak_mb:.0f} MB  "
        f"start={watcher.start_rss:.0f} MB  "
        f"delta={delta_mb:.0f} MB  "
        f"wall={wall_s:.2f}s  "
        f"budget={budget_mb:.0f} MB",
    )

    # Use delta against the *start* RSS (not absolute peak) so the
    # budget isn't sensitive to unrelated process baselines (e.g. the
    # pytest interpreter's own footprint).
    assert delta_mb < budget_mb, (
        f"input_derivation.run on test_a_lot allocated {delta_mb:.0f} MB "
        f"during execution, exceeding the {budget_mb:.0f} MB regression "
        f"budget.  This typically indicates a memory-retention "
        f"regression in spinedb_backend (parsed_value cache survived "
        f"longer than intended) or in input_derivation (a derived "
        f"frame leak).  Investigate per "
        f"specs/memory_diagnostic_results.md."
    )
