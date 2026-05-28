"""In-process memory sampler — writes timestamped RSS/available/swap to a log file."""
from __future__ import annotations

import datetime
import os
import sys
import threading
import time


def start_mem_sampler() -> None:
    """Start a daemon thread that samples memory at a fixed interval if FLEXTOOL_MEM_SAMPLER is set."""
    if os.environ.get("FLEXTOOL_MEM_SAMPLER", "").strip().lower() in ("", "0", "false"):
        return

    pid = os.getpid()
    log_path = os.environ.get("FLEXTOOL_MEM_SAMPLER_LOG", f"/tmp/flextool_mem_sampler_{pid}.log")
    try:
        interval_ms = max(20, min(10000, int(os.environ.get("FLEXTOOL_MEM_SAMPLER_INTERVAL_MS", "100"))))
    except ValueError:
        interval_ms = 100
    interval_s = interval_ms / 1000.0

    try:
        import psutil
        _proc = psutil.Process(pid)

        def _sample():
            rss_gb = _proc.memory_info().rss / 2**30
            vm = psutil.virtual_memory()
            available_gb = vm.available / 2**30
            sw = psutil.swap_memory()
            swap_used_gb = sw.used / 2**30
            return rss_gb, available_gb, swap_used_gb

    except ImportError:
        if sys.platform != "linux":
            # No psutil and not Linux: write a warning and bail out.
            with open(log_path, "a", buffering=1) as _f:
                _f.write(
                    f"# flextool memory sampler pid={pid} WARNING: psutil not available "
                    f"and platform is not linux — sampler disabled\n"
                )
            return

        def _read_proc_value(path: str, key: str) -> int:
            with open(path) as fh:
                for line in fh:
                    if line.startswith(key):
                        return int(line.split()[1]) * 1024  # kB → bytes
            raise KeyError(key)

        def _sample():
            rss = _read_proc_value("/proc/self/status", "VmRSS:")
            rss_gb = rss / 2**30
            available = _read_proc_value("/proc/meminfo", "MemAvailable:")
            available_gb = available / 2**30
            swap_total = _read_proc_value("/proc/meminfo", "SwapTotal:")
            swap_free = _read_proc_value("/proc/meminfo", "SwapFree:")
            swap_used_gb = (swap_total - swap_free) / 2**30
            return rss_gb, available_gb, swap_used_gb

    started_at = datetime.datetime.now(datetime.UTC).isoformat()

    def _loop(log_path: str, interval_s: float) -> None:
        mono_start = time.monotonic()
        with open(log_path, "a", buffering=1) as f:
            f.write(
                f"# flextool memory sampler pid={pid} started_at={started_at} interval_ms={interval_ms}\n"
            )
            while True:
                ts = datetime.datetime.now(datetime.UTC).isoformat()
                mono_s = time.monotonic() - mono_start
                try:
                    rss_gb, available_gb, swap_used_gb = _sample()
                    line = (
                        f"ts={ts}\tmono_s={mono_s:.3f}\t"
                        f"rss_gb={rss_gb:.4f}\tavailable_gb={available_gb:.4f}\t"
                        f"swap_used_gb={swap_used_gb:.4f}\n"
                    )
                except Exception as exc:
                    line = f"ts={ts}\tmono_s={mono_s:.3f}\terror={repr(exc)}\n"
                f.write(line)
                # Explicit flush so the last samples before SIGKILL reach disk;
                # line-buffered mode alone won't flush until the process exits cleanly.
                f.flush()
                time.sleep(interval_s)

    t = threading.Thread(target=_loop, args=(log_path, interval_s), name="flextool-mem-sampler", daemon=True)
    t.start()
