"""In-process memory sampler — timestamped, allocator-aware memory trace.

RSS is a poor signal for this workload: it conflates shared, clean, and
allocator-reserved-but-free pages with genuinely-live private data.  This
sampler additionally records, per sample:

* ``pss_gb`` / ``priv_dirty_gb`` / ``proc_swap_gb`` — from
  ``/proc/self/smaps_rollup``.  ``Private_Dirty + Swap`` is the
  allocator-agnostic "what this process truly owns" number (counts dirty
  anon pages whether they live in glibc, mimalloc, or the C++ heap, and
  whether resident or swapped).
* ``glibc_inuse_gb`` / ``glibc_free_gb`` / ``glibc_mmap_gb`` — from
  glibc ``mallinfo2()`` (``uordblks`` / ``fordblks`` / ``hblkhd``).  The
  *live-vs-reserved* discriminator: ``fordblks`` is memory glibc has
  freed but is holding on its arena free-list rather than returning to
  the OS (what ``malloc_trim`` would release).  NOTE: only sees the glibc
  allocator — HiGHS/C++ go through glibc, but polars (Rust) may use its
  own allocator, so a large ``priv_dirty`` with small ``glibc_inuse``
  points at the non-glibc (polars) side.
* ``mem_used_gb`` — system used (``MemTotal - MemAvailable``), the number
  most desktop system monitors plot.  ``available_gb`` / ``swap_used_gb``
  kept for back-compat.

Gated by ``FLEXTOOL_MEM_SAMPLER``; interval via
``FLEXTOOL_MEM_SAMPLER_INTERVAL_MS`` (default 100 ms).
"""
from __future__ import annotations

import ctypes
import datetime
import os
import sys
import threading
import time


class _Mallinfo2(ctypes.Structure):
    # glibc >= 2.33.  All fields size_t (the legacy ``mallinfo`` used int
    # and overflows past 2 GB — useless here).
    _fields_ = [
        (n, ctypes.c_size_t)
        for n in (
            "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
            "fsmblks", "uordblks", "fordblks", "keepcost",
        )
    ]


def _make_mallinfo2():
    """Return a callable giving (inuse, free, mmap) bytes, or None."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        fn = libc.mallinfo2  # AttributeError on glibc < 2.33
        fn.restype = _Mallinfo2
        fn.argtypes = []
    except (OSError, AttributeError):
        return None

    def _read():
        mi = fn()
        return mi.uordblks, mi.fordblks, mi.hblkhd

    return _read


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

    if not sys.platform.startswith("linux"):
        with open(log_path, "a", buffering=1) as _f:
            _f.write(
                f"# flextool memory sampler pid={pid} WARNING: platform is not "
                f"linux — sampler disabled\n"
            )
        return

    _GB = float(2 ** 30)
    _mallinfo2 = _make_mallinfo2()

    def _read_kv_bytes(path: str, keys: tuple[str, ...]) -> dict[str, int]:
        """Parse ``Key:   N kB`` lines from a /proc file → bytes."""
        out: dict[str, int] = {}
        want = set(keys)
        with open(path) as fh:
            for line in fh:
                k = line.split(":", 1)[0]
                if k in want:
                    out[k] = int(line.split()[1]) * 1024
                    if len(out) == len(want):
                        break
        return out

    def _sample() -> dict[str, float]:
        # Per-process private cost — allocator-agnostic.
        rss = pss = priv_dirty = proc_swap = 0
        try:
            r = _read_kv_bytes(
                "/proc/self/smaps_rollup",
                ("Rss", "Pss", "Private_Dirty", "Swap"),
            )
            rss = r.get("Rss", 0)
            pss = r.get("Pss", 0)
            priv_dirty = r.get("Private_Dirty", 0)
            proc_swap = r.get("Swap", 0)
        except Exception:
            pass
        if rss == 0:
            # smaps_rollup unavailable — fall back to VmRSS.
            try:
                rss = _read_kv_bytes("/proc/self/status", ("VmRSS",)).get("VmRSS", 0)
            except Exception:
                pass

        # System-wide.
        mem_total = mem_avail = swap_total = swap_free = 0
        try:
            m = _read_kv_bytes(
                "/proc/meminfo",
                ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"),
            )
            mem_total = m.get("MemTotal", 0)
            mem_avail = m.get("MemAvailable", 0)
            swap_total = m.get("SwapTotal", 0)
            swap_free = m.get("SwapFree", 0)
        except Exception:
            pass

        # glibc allocator breakdown — live vs reserved-free.
        g_inuse = g_free = g_mmap = -1
        if _mallinfo2 is not None:
            try:
                g_inuse, g_free, g_mmap = _mallinfo2()
            except Exception:
                pass

        return {
            "rss_gb": rss / _GB,
            "pss_gb": pss / _GB,
            "priv_dirty_gb": priv_dirty / _GB,
            "proc_swap_gb": proc_swap / _GB,
            "available_gb": mem_avail / _GB,
            "mem_used_gb": (mem_total - mem_avail) / _GB,
            "swap_used_gb": (swap_total - swap_free) / _GB,
            "glibc_inuse_gb": g_inuse / _GB if g_inuse >= 0 else -1.0,
            "glibc_free_gb": g_free / _GB if g_free >= 0 else -1.0,
            "glibc_mmap_gb": g_mmap / _GB if g_mmap >= 0 else -1.0,
        }

    started_at = datetime.datetime.now(datetime.UTC).isoformat()
    # Stable column order so post-hoc parsing is trivial.
    _cols = (
        "rss_gb", "pss_gb", "priv_dirty_gb", "proc_swap_gb",
        "available_gb", "mem_used_gb", "swap_used_gb",
        "glibc_inuse_gb", "glibc_free_gb", "glibc_mmap_gb",
    )

    def _loop(log_path: str, interval_s: float) -> None:
        mono_start = time.monotonic()
        with open(log_path, "a", buffering=1) as f:
            f.write(
                f"# flextool memory sampler pid={pid} started_at={started_at} "
                f"interval_ms={interval_ms} mallinfo2={'yes' if _mallinfo2 else 'no'}\n"
            )
            f.write("# priv_dirty_gb+proc_swap_gb = true private cost; "
                    "glibc_free_gb = held-free (malloc_trim-able); "
                    "mem_used_gb = system used (monitor metric)\n")
            while True:
                ts = datetime.datetime.now(datetime.UTC).isoformat()
                mono_s = time.monotonic() - mono_start
                try:
                    s = _sample()
                    parts = "\t".join(f"{c}={s[c]:.4f}" for c in _cols)
                    line = f"ts={ts}\tmono_s={mono_s:.3f}\tepoch={time.time():.3f}\t{parts}\n"
                except Exception as exc:
                    line = f"ts={ts}\tmono_s={mono_s:.3f}\tepoch={time.time():.3f}\terror={repr(exc)}\n"
                f.write(line)
                # Explicit flush so the last samples before SIGKILL reach disk;
                # line-buffered mode alone won't flush until the process exits cleanly.
                f.flush()
                time.sleep(interval_s)

    t = threading.Thread(target=_loop, args=(log_path, interval_s), name="flextool-mem-sampler", daemon=True)
    t.start()
