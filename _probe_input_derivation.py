"""Peak-RSS probe for input_derivation.run on a Spine DB + scenario.

Untracked. Used by Step 3 Track A to baseline and re-measure the
SpineDBBackend._parameter_value_index peak.

Reports:
- RSS at startup (post-imports)
- RSS just before SpineDBBackend.close()  (the peak, while parameter
  values are still parsed)
- RSS after close + malloc_trim         (the trough)
- ru_maxrss process peak                (high-water mark)

Usage:

    python3 _probe_input_derivation.py \\
        sqlite:////home/jkiviluo/sources/flextool/projects/h2-imo/input_sources/H2_trade.sqlite \\
        test_24h
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import resource
import sys
import threading
import time
from pathlib import Path


def _rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _maxrss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class _PeakWatcher:
    """Background thread polling RSS every interval_s seconds."""

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

    def start(self) -> "_PeakWatcher":
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        self._thread.join(timeout=1.0)
        return self.peak_mb


def _malloc_trim() -> bool:
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("db_url")
    parser.add_argument("scenario")
    parser.add_argument("--precision-digits", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"[probe] start  rss={_rss_mb():.0f} MB  maxrss={_maxrss_mb():.0f} MB")

    t_import_0 = time.perf_counter()
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    from flextool.input_derivation import run as input_derivation_run
    from flextool.spinedb_backend import SpineDBBackend
    print(
        f"[probe] imports done   rss={_rss_mb():.0f} MB"
        f"  Δt={time.perf_counter() - t_import_0:.2f}s"
    )

    import tempfile

    provider = FlexDataProvider()
    watcher = _PeakWatcher().start()

    # Inline expansion of input_derivation.run's db_url branch so we can
    # checkpoint RSS at fetch_all / scenario_filter / spec-loops boundaries.
    import spinedb_api as api
    from spinedb_api import DatabaseMapping

    with tempfile.TemporaryDirectory(prefix="probe_") as tmp:
        wf = Path(tmp)
        t0 = time.perf_counter()
        scen_config = api.filters.scenario_filter.scenario_filter_config(
            args.scenario
        ) if args.scenario else None
        with DatabaseMapping(args.db_url) as db:
            print(f"[probe] DatabaseMapping open      rss={_rss_mb():.0f} MB"
                  f"  Δt={time.perf_counter()-t0:.2f}s")
            db.fetch_all("entity")
            print(f"[probe] fetch_all(entity)         rss={_rss_mb():.0f} MB"
                  f"  Δt={time.perf_counter()-t0:.2f}s")
            db.fetch_all("parameter_value")
            print(f"[probe] fetch_all(parameter_value) rss={_rss_mb():.0f} MB"
                  f"  Δt={time.perf_counter()-t0:.2f}s")
            if scen_config is not None:
                api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
                print(f"[probe] scenario_filter applied   rss={_rss_mb():.0f} MB"
                      f"  Δt={time.perf_counter()-t0:.2f}s")
            os.makedirs(wf / "input", exist_ok=True)

            # Re-use input_derivation._do via the non-str branch by passing db.
            input_derivation_run(
                db,
                provider,
                scenario_name=args.scenario,
                work_folder=wf,
                precision_digits=args.precision_digits,
            )
        rss_at_end = _rss_mb()
        watcher_peak_during = watcher.peak_mb
        print(
            f"[probe] derivation end            rss={rss_at_end:.0f} MB"
            f"  watcher_peak={watcher_peak_during:.0f} MB"
            f"  Δt={time.perf_counter() - t0:.2f}s"
        )
        print(f"[probe] provider keys             {len(provider.keys())}")

    # The DatabaseMapping was already exited via the ``with`` block inside
    # input_derivation.run.  Drop the stub backend wrapper, trim the heap.
    gc.collect()
    _malloc_trim()
    rss_after_trim = _rss_mb()
    watcher_peak_final = watcher.stop()
    print(
        f"[probe] post-derivation + trim    rss={rss_after_trim:.0f} MB"
        f"  watcher_peak={watcher_peak_final:.0f} MB"
    )
    print(
        f"[probe] process maxrss            {_maxrss_mb():.0f} MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
