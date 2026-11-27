"""Run diagnostics: the opt-in memory / phase recorder (``FLEXTOOL_MEMORY_DIAGNOSTICS``) and its run-wide
handle, the glibc heap release, the phase profile print and the cross-level retention audit."""
from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Opt-in memory diagnostics
# ---------------------------------------------------------------------------


# Whitelist of ``user_label`` strings that are emitted to the log in
# regular (non-verbose) mode.  Set ``FLEXTOOL_MEMORY_VERBOSE=1`` to emit
# every checkpoint (the full pre-cleanup trace).
#
# ``Solve start: <name>, k/N`` markers are emitted as plain text lines
# (no checkpoint), so they don't appear here.
_MEM_WHITELIST_LABELS: frozenset[str] = frozenset({
    "Run start",
    "Inputs prepared",
    "FlexData built",
    "Matrix built by polar-high",
    "polar-high scaling",
    "Solver",
    "Outputs written",
    "Solve cleanup",
})


def _is_whitelisted_mem_label(user_label: str | None) -> bool:
    """True when ``user_label`` matches a whitelisted phase label."""
    if not user_label:
        return False
    return user_label in _MEM_WHITELIST_LABELS


class _MemoryRecorder:
    """Opt-in tracemalloc + RSS checkpoint recorder.

    Activated by ``FLEXTOOL_MEMORY_DIAGNOSTICS=1``.  When the env var is
    not set, callers should construct :class:`_NoopMemoryRecorder`
    instead (or simply skip construction); :meth:`checkpoint` here is the
    hot-path no-op fallback only when ``enabled`` is False.

    Each :meth:`checkpoint` call appends one row to
    ``<work_folder>/solve_data/memory_diagnostics.csv`` with schema::

        checkpoint,t_elapsed_s,traced_current_mb,traced_peak_mb,rss_mb

    The file is open/append/closed per row (same atomicity pattern as
    :class:`flextool.cli._timing.TimingRecorder.record`)
    so a crash mid-cascade still leaves a parseable trail.

    A one-liner is also logged at INFO level via the supplied logger so
    progress is visible in stdout even when the GUI buffers.

    ``tracemalloc.start()`` is invoked lazily on first checkpoint to keep
    the cost localised to instrumented runs.
    """

    _HEADER = (
        "checkpoint",
        "t_elapsed_s",
        "traced_current_mb",
        "traced_peak_mb",
        "rss_mb",
    )

    def __init__(self, csv_path: Path | None = None,
                 enabled: bool = True,
                 verbose: bool = True) -> None:
        """Construct a phase-progress recorder.

        Parameters
        ----------
        csv_path
            Where to write the per-checkpoint CSV.  ``None`` skips CSV
            emission (verbose log lines still fire).
        enabled
            Full diagnostic mode — starts tracemalloc on first checkpoint
            so ``traced_peak`` becomes meaningful, and writes the CSV.
            When ``False`` we still emit human-readable log lines with
            RSS + section time + Δrss (RSS reads from ``/proc`` are
            essentially free); ``peak`` shows as ``-`` since tracemalloc
            isn't running.
        verbose
            Emit log lines (one per checkpoint).  Set ``False`` only if
            you want a fully silent recorder (rare; debugging).
        """
        self.enabled = enabled
        self.verbose = verbose
        self.t0 = time.perf_counter()
        self._t_prev = self.t0
        self._rss_prev_mb: float = 0.0
        self._peak_prev_mb: float = 0.0
        self._sys_prev_mb: float = 0.0
        self._swap_prev_mb: float = 0.0
        self._header_emitted: bool = False
        self._path = Path(csv_path) if csv_path is not None else None
        self._started = False
        # FLEXTOOL_PYRAMID_PROFILE=1: previous RSS for per-batch delta.
        self._pyramid_prev_rss_mb: float = 0.0
        if self.enabled and self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            import csv as _csv
            with open(self._path, "w", encoding="utf-8", newline="") as f:
                _csv.writer(f).writerow(self._HEADER)

    @staticmethod
    def _read_rss_mb() -> float:
        """Read committed memory (anon RSS + swap, in MB) from
        ``/proc/self/status``.

        We deliberately don't report ``VmRSS`` (= ``RssAnon`` +
        ``RssFile`` + ``RssShmem``) because file-backed pages are
        evictable cache from the kernel's POV and don't reflect the
        process's true memory commitment.  ``RssAnon`` (anonymous
        resident, i.e. heap + private mappings) plus ``VmSwap`` (the
        same anonymous pages that have been swapped out) gives the
        right picture of "memory this process actually needs" —
        what systemd-oomd's PSI signal effectively responds to, and
        what tracks the system monitor's "Used" number more closely
        than raw ``VmRSS``.

        Returns 0.0 if /proc isn't available (non-Linux) or the
        relevant lines aren't found.  We never want diagnostics to
        raise.
        """
        anon_kb = 0.0
        swap_kb = 0.0
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            anon_kb = float(parts[1])
                    elif line.startswith("VmSwap:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            swap_kb = float(parts[1])
        except OSError:
            pass
        return (anon_kb + swap_kb) / 1024.0

    @staticmethod
    def _read_sys_swap_used_mb() -> float | None:
        """System-level used swap (MB) = ``SwapTotal - SwapFree`` from
        ``/proc/meminfo``.  Returns 0.0 when there's no swap configured
        (``SwapTotal == 0``), ``None`` when ``/proc/meminfo`` isn't
        available (Windows, macOS: the cell shows ``-``, not a fake 0).
        """
        total_kb = 0.0
        free_kb = 0.0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("SwapTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kb = float(parts[1])
                    elif line.startswith("SwapFree:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            free_kb = float(parts[1])
                    if total_kb and free_kb:
                        break
        except OSError:
            return None
        if total_kb <= 0:
            return 0.0
        return max(0.0, (total_kb - free_kb) / 1024.0)

    @staticmethod
    def _read_sys_used_mb() -> float | None:
        """System-level used memory (MB), matching what most monitors
        ("htop", KSysGuard, GNOME) show as "Used".

        Computed as ``MemTotal - MemAvailable`` from ``/proc/meminfo``.
        ``MemAvailable`` is the kernel's own estimate of how much
        memory could be allocated to a new process without swapping —
        it already accounts for evictable page-cache + reclaimable
        slab + lazily-freed anonymous pages (MADV_FREE).  Subtracting
        from total gives the closest single-number match to what the
        user sees in their desktop's system monitor.

        Includes contributions from every process on the host, not
        just this one — which is the right metric for desktop-crash
        awareness (the desktop crashes when total ``MemAvailable``
        approaches zero, regardless of which process is consuming the
        pages).

        Returns ``None`` if /proc/meminfo isn't available (the cell then
        shows ``-``).
        """
        total_kb = 0.0
        avail_kb = 0.0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kb = float(parts[1])
                    elif line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            avail_kb = float(parts[1])
                    if total_kb and avail_kb:
                        break
        except OSError:
            return None
        if total_kb <= 0:
            return None
        return (total_kb - avail_kb) / 1024.0

    # Fixed widths of the phase-progress table.  A row reads
    #   "<label> <time> | <RSS cell> | <system cell> | <swap cell>"
    # with a single space before each "|" and 17 characters between two
    # bars (a 16-wide right-aligned cell plus the space before the next
    # bar), so the bars sit in the same columns on every row:
    #   Matrix built by polar-high 10000s (+10000) |     544 MB (-5) | ...
    # The label+time part is ``_LABEL_W`` + 1 + ``_TIME_CELL_W`` wide and
    # stretches only when a label is longer than ``_LABEL_W`` (one space
    # is always kept before the time) or a time passes 99999 s.
    _LABEL_W = 26      # the longest whitelisted label, "Matrix built by polar-high"
    _TIME_CELL_W = 15  # fits "99999s (+99999)"
    _MEM_CELL_W = 16   # fits "16.9 GB (-0.01)" plus the space after the bar

    _SIZE_UNITS: tuple[str, ...] = ("kB", "MB", "GB", "TB")

    @staticmethod
    def _fmt_sig3(value: float) -> str:
        """``value`` (>= 0) to 3 significant figures in at most 4 digits.

        Below 1000 the value keeps at most two decimals so it never
        passes 4 characters (``0.12``, ``9.1``, ``16.9``, ``544``);
        trailing zeros are dropped (``9.10`` -> ``9.1``, ``16.0`` ->
        ``16``).  From 1000 up it is an integer rounded to 3 significant
        figures (``1230``, ``10000``).
        """
        if value <= 0 or not math.isfinite(value):
            return "0"
        exp = math.floor(math.log10(value))
        rounded = round(value, 2 - exp)
        if rounded >= 1000:
            return f"{rounded:.0f}"
        exp = math.floor(math.log10(rounded)) if rounded > 0 else 0
        decimals = min(2, max(0, 2 - exp))
        text = f"{rounded:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @classmethod
    def _fmt_signed_sig3(cls, value: float) -> str:
        """``value`` with an explicit sign (``+12``, ``-0.01``); ``+0``
        when it rounds to zero."""
        text = cls._fmt_sig3(abs(value))
        if text == "0":
            return "+0"
        return ("+" if value >= 0 else "-") + text

    @classmethod
    def _pick_size_unit(cls, mb: float) -> tuple[float, str]:
        """Pick the kB/MB/GB/TB unit that keeps ``mb`` in 0-999 once
        rounded to 3 significant figures.  Returns ``(divisor_in_mb,
        label)`` — e.g. ``(1024.0, "GB")``."""
        if mb == 0:
            return 1.0, "MB"       # an empty swap reads "0 MB", not "0 kB"
        div = 1.0 / 1024.0
        for label in cls._SIZE_UNITS[:-1]:
            if float(cls._fmt_sig3(abs(mb) / div)) < 1000:
                return div, label
            div *= 1024.0
        return div, cls._SIZE_UNITS[-1]

    @classmethod
    def _fmt_mem_cell(cls, mb: float | None, delta_mb: float | None) -> str:
        """Render a memory cell as ``"<absolute> <unit> (<±delta>)"``
        right-aligned within ``_MEM_CELL_W``.  The delta is in the unit
        of the absolute, so the cell reads as one magnitude (e.g.
        ``"16.9 GB (-0.01)"``); a negligible delta reads ``+0``.  When
        ``mb`` is None, a single dash fills the cell.
        """
        if mb is None:
            return f"{'-':>{cls._MEM_CELL_W}}"
        div, label = cls._pick_size_unit(mb)
        val = f"{cls._fmt_sig3(mb / div)} {label}"
        if delta_mb is None:
            cell = f"{val} (-)"
        elif float(cls._fmt_sig3(abs(delta_mb) / div)) >= 1000:
            # A delta far larger than the absolute (swap 2 GB -> 0.5 MB) would not fit in the absolute's unit:
            # it gets its own (``512 kB (-2 GB)``), so the cell keeps its width.
            ddiv, dlabel = cls._pick_size_unit(delta_mb)
            cell = f"{val} ({cls._fmt_signed_sig3(delta_mb / ddiv)} {dlabel})"
            if len(cell) >= cls._MEM_CELL_W:      # still too wide: the delta as a whole number of its unit
                cell = f"{val} ({'+' if delta_mb >= 0 else '-'}{max(1, round(abs(delta_mb) / ddiv))} {dlabel})"
        else:
            cell = f"{val} ({cls._fmt_signed_sig3(delta_mb / div)})"
        return f"{cell:>{cls._MEM_CELL_W}}"

    @classmethod
    def _fmt_time_cell(cls, t_elapsed: float, t_section: float | None) -> str:
        """Render the time cell as ``"<elapsed>s (+<section>)"`` (seconds,
        3 significant figures) right-aligned within ``_TIME_CELL_W``.
        First-row ``t_section is None`` renders ``"<elapsed>s (-)"``.
        """
        elapsed = f"{cls._fmt_sig3(t_elapsed)}s"
        if t_section is None:
            cell = f"{elapsed} (-)"
        else:
            cell = f"{elapsed} ({cls._fmt_signed_sig3(t_section)})"
        return f"{cell:>{cls._TIME_CELL_W}}"

    @classmethod
    def _fmt_row(cls, label: str, time_cell: str, mem_cells: Sequence[str]) -> str:
        """Join a label, a time cell and memory cells into one table row.

        The label+time part keeps its fixed width (label left, time
        right) unless the two do not fit with one space between them.
        """
        width = cls._LABEL_W + 1 + cls._TIME_CELL_W
        lead = f"{label} {time_cell.lstrip()}"
        if len(lead) < width:
            lead = f"{label}{time_cell.lstrip():>{width - len(label)}}"
        return lead + "".join(f" |{cell}" for cell in mem_cells)

    def _emit_header(self) -> None:
        """Print the column-header line for the phase-progress table."""
        header = self._fmt_row(
            "", f"{'time':>{self._TIME_CELL_W}}",
            [f"{name:>{self._MEM_CELL_W}}" for name in
             ("RSS memory", "system memory", "system swap")],
        ).rstrip()
        try:
            print(header, flush=True)
        except OSError:
            pass

    def checkpoint(self, label: str, logger: logging.Logger,
                   user_label: str | None = None) -> None:
        """Record a phase checkpoint.

        ``label`` is the canonical machine-readable identifier persisted
        to the CSV (when full diagnostics is enabled).  ``user_label``
        (optional) is the human-friendly phrasing emitted to the log;
        when absent, ``label`` is used.

        Log lines always emit (RSS read from ``/proc`` is essentially
        free).  When full diagnostics is enabled (env-var
        ``FLEXTOOL_MEMORY_DIAGNOSTICS=1``) the ``traced_peak`` column
        and the CSV emission are populated by tracemalloc; otherwise
        the peak shows as ``-``.
        """
        peak_mb: float | None = None
        current_mb: float | None = None
        if self.enabled:
            import tracemalloc
            if not self._started:
                tracemalloc.start()
                self._started = True
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / (1024.0 * 1024.0)
            peak_mb = peak / (1024.0 * 1024.0)
        rss_mb = self._read_rss_mb()
        sys_mb = self._read_sys_used_mb()
        swap_mb = self._read_sys_swap_used_mb()
        t_elapsed = time.perf_counter() - self.t0
        # Section deltas relative to previous *emitted* checkpoint.  When
        # we suppress an emission we leave ``_t_prev`` /
        # ``_rss_prev_mb`` / ``_sys_prev_mb`` / ``_peak_prev_mb`` alone,
        # so the next emitted line shows the cumulative delta covering
        # all the suppressed activity in between — exactly what the
        # user wants to see in the compact mode.
        t_section = t_elapsed - (self._t_prev - self.t0)
        delta_rss = rss_mb - self._rss_prev_mb
        delta_sys = None if sys_mb is None else sys_mb - self._sys_prev_mb
        delta_swap = None if swap_mb is None else swap_mb - self._swap_prev_mb
        # CSV row — only when full diagnostics is enabled and a path was
        # configured.  Always written for every checkpoint (independent
        # of the log-line whitelist) so the CSV remains a complete
        # trace.
        if self.enabled and self._path is not None and peak_mb is not None:
            row = (
                str(label),
                f"{t_elapsed:.6f}",
                f"{current_mb:.3f}",
                f"{peak_mb:.3f}",
                f"{rss_mb:.3f}",
            )
            import csv as _csv
            try:
                with open(self._path, "a", encoding="utf-8", newline="") as f:
                    _csv.writer(f).writerow(row)
            except OSError:
                pass
        # FLEXTOOL_PYRAMID_PROFILE=1: emit a tab-separated stderr line
        # for emit_solve_time.* labels (polar-high precedent).  Reuses
        # the already-computed rss_mb / t_elapsed; zero cost when unset.
        if (os.environ.get("FLEXTOOL_PYRAMID_PROFILE") == "1"
                and label.startswith("emit_solve_time.")):
            delta_pyramid = rss_mb - self._pyramid_prev_rss_mb
            rss_gb = rss_mb / 1024.0
            delta_gb = delta_pyramid / 1024.0
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"[pyramid profile]\tphase={label}\t"
                    f"rss_gb={rss_gb:.2f}\tdelta_gb={delta_gb:+.2f}\t"
                    f"t_s={t_elapsed:.1f}\n"
                )
                _sys.stderr.flush()
            except OSError:
                pass
            self._pyramid_prev_rss_mb = rss_mb
        # Decide whether to emit the log line.  Regular mode shows only
        # the whitelisted phase labels; ``FLEXTOOL_MEMORY_VERBOSE=1``
        # restores the full per-checkpoint trace.
        verbose_mode = bool(os.environ.get("FLEXTOOL_MEMORY_VERBOSE"))
        emit = self.verbose and (
            verbose_mode or _is_whitelisted_mem_label(user_label or label)
        )
        if emit:
            display = user_label or label
            is_first = self._t_prev == self.t0
            # Emit the column header once, immediately before the first
            # data line.  Inline labels are dropped from data lines
            # (cleaner, narrower); the header is the legend.
            if not self._header_emitted:
                self._emit_header()
                self._header_emitted = True
            elif display == "Solver":
                # Blank line + header repeat above each Solver row so
                # the dominant solve phase visually separates from the
                # per-group prep block printed above it.
                try:
                    print("", flush=True)
                except OSError:
                    pass
                self._emit_header()
            # First-row convention: time delta renders "(-)" (no prior
            # checkpoint to subtract from); memory deltas equal their
            # absolute (prev=0), so e.g. "+230.1" appears alongside the
            # absolute "230.1 MB" — which is correct: the process has
            # consumed exactly that much since the recorder started.
            time_cell = self._fmt_time_cell(
                t_elapsed, None if is_first else t_section
            )
            rss_cell = self._fmt_mem_cell(rss_mb, delta_rss)
            sys_cell = self._fmt_mem_cell(sys_mb, delta_sys)
            swap_cell = self._fmt_mem_cell(swap_mb, delta_swap)
            line = self._fmt_row(
                display, time_cell, (rss_cell, sys_cell, swap_cell))
            try:
                print(line, flush=True)
            except OSError:
                pass
            # Advance prev-section bookkeeping ONLY on actual emission so
            # the next emitted line's delta covers all the suppressed
            # activity since the last visible checkpoint.
            self._t_prev = time.perf_counter()
            self._rss_prev_mb = rss_mb
            self._sys_prev_mb = sys_mb or 0.0
            self._swap_prev_mb = swap_mb or 0.0
            if peak_mb is not None:
                self._peak_prev_mb = peak_mb


class _NoopMemoryRecorder:
    """Zero-overhead drop-in when ``FLEXTOOL_MEMORY_DIAGNOSTICS`` is unset.

    Retained for callers that explicitly want a fully silent recorder
    (rare; debugging).  The default code path now uses
    :class:`_MemoryRecorder` with ``enabled=False`` instead — that mode
    still emits user-visible log lines (RSS + section time + Δrss)
    while skipping CSV emission and tracemalloc startup.
    """

    enabled = False

    def checkpoint(self, label: str, logger: logging.Logger,
                   user_label: str | None = None) -> None:  # noqa: D401, ARG002
        return None


# Module-level recorder reference.  ``run_orchestration`` constructs the
# per-run recorder and publishes it here so deeper-stack modules (e.g.
# :mod:`flextool.engine.preprocess.flex_data`'s ``_apply_db_overrides``) can
# emit phase progress in the unified ``[mem]`` format without each
# carrying a recorder kwarg.  Reset to ``None`` when the run completes
# so a subsequent run starts clean.
_PHASE_RECORDER: "_MemoryRecorder | None" = None


def set_phase_recorder(rec: "_MemoryRecorder | None") -> None:
    """Publish the current run's phase recorder so deeper callers can
    emit checkpoints without explicit plumbing.  Pass ``None`` to clear.
    """
    global _PHASE_RECORDER
    _PHASE_RECORDER = rec


def get_phase_recorder() -> "_MemoryRecorder | None":
    """Return the current run's phase recorder, or ``None`` when none
    is active (e.g. unit tests that bypass ``run_orchestration``).
    """
    return _PHASE_RECORDER


# ---------------------------------------------------------------------------
# Heap release (glibc malloc_trim)
# ---------------------------------------------------------------------------
#
# The polars/Rust allocator routes through glibc malloc, and glibc's main
# arena holds freed pages internally instead of returning them to the OS.
# After a heavy allocation+free cycle (``write_workdir_inputs``,
# ``load_flextool``, the broadcast cascade) we leak hundreds of MB to
# multiple GB of unmapped-but-untrimmed heap.  Direct measurement on
# H2_trade y2050 (2026-05-13):  RSS 3.8 GB → 2.25 GB after a single
# ``malloc_trim(0)`` call (1.6 GB / 41 % drop).  ``pa.default_memory_pool
# ().release_unused()`` and ``gc.collect()`` had zero effect — polars
# does not route through pyarrow's pool, so only the libc-level trim
# releases anything.
#
# The helper is a no-op on non-glibc systems (musl Alpine containers,
# macOS, Windows).  Safe to call freely; cost is ~10-50ms per call.
_libc_malloc_trim = None


def _try_malloc_trim() -> bool:
    """Call ``libc.so.6.malloc_trim(0)`` if available; return True on success.

    Cached lookup after the first call.  Failures (non-glibc systems,
    missing libc, etc.) are logged once at DEBUG level and the helper
    becomes a permanent no-op for the process lifetime.
    """
    global _libc_malloc_trim
    if _libc_malloc_trim is False:
        return False
    if _libc_malloc_trim is None:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # malloc_trim(size_t pad) -> int.  pad=0 means trim aggressively.
            _libc_malloc_trim = libc.malloc_trim
        except (OSError, AttributeError):
            _libc_malloc_trim = False
            return False
    try:
        _libc_malloc_trim(0)
        return True
    except Exception:  # noqa: BLE001
        _libc_malloc_trim = False
        return False


def _phase_prof(label: str) -> None:
    """Env-gated (FLEXTOOL_PHASE_PROFILE=1) epoch-stamped RSS print to stderr. No-op otherwise.
    Epoch matches flextool/_mem_sampler.py's `epoch=` field for 1:1 alignment with mem.log."""
    import os
    import sys
    import time
    if os.environ.get("FLEXTOOL_PHASE_PROFILE") != "1":
        return
    try:
        with open("/proc/self/status", encoding="utf-8") as _f:
            for _ln in _f:
                if _ln.startswith("VmRSS:"):
                    sys.stderr.write(f"[phase profile] epoch={time.time():.3f}\tstep={label}\trss_gb={int(_ln.split()[1])/(1024*1024):.3f}\n")
                    sys.stderr.flush()
                    break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cross-level retention audit (env-gated diagnostic + regression hook)
#
# A prior solve-level's ``Solution.highs`` (and ``flex_data_provider``) must be
# released BEFORE the next solve builds its FlexData + LP — otherwise the two
# levels' footprints coexist (storage + dispatch ≈ 2x peak; the DES 7/9
# near-OOM).  This records any prior step whose level is EXHAUSTED (no upcoming
# solve of that level) yet still holds a live ``solution.highs`` at the instant
# a new solve is about to build.  A non-empty record == the cross-level
# retention bug is present.  Gated by ``FLEXTOOL_LEVEL_RELEASE_AUDIT=1``;
# consumed by tests/engine/test_cross_level_highs_release.py.
# ---------------------------------------------------------------------------
_LEVEL_RELEASE_AUDIT: "list[dict]" = []


def _audit_prior_level_release(*, steps, step_level_keys, all_level_keys,
                               iter_idx, this_level, complete_solve_name):
    """Append a violation record for exhausted-level prior steps still holding
    a live ``solution.highs``.  No-op unless ``FLEXTOOL_LEVEL_RELEASE_AUDIT=1``."""
    if os.environ.get("FLEXTOOL_LEVEL_RELEASE_AUDIT") != "1":
        return
    upcoming = (
        set(all_level_keys[iter_idx + 1:])
        if (iter_idx is not None and all_level_keys) else set()
    )
    violators = []
    for _k, _step in (steps or {}).items():
        _lvl = step_level_keys.get(_k)
        sol = getattr(_step, "solution", None)
        if sol is None or getattr(sol, "highs", None) is None:
            continue
        if _lvl is not None and _lvl != this_level and _lvl not in upcoming:
            violators.append(_k)
    _LEVEL_RELEASE_AUDIT.append({
        "kind": "exhausted_entry",
        "solve": complete_solve_name,
        "iter_idx": iter_idx,
        "violators": violators,
    })


def _audit_cold_rebuild_release(*, steps, complete_solve_name, warm_problem_live=False):
    """At a COLD rebuild (warm_used False), no parked step's HiGHS is the
    reuse source, so any prior step still holding a live ``solution.highs``
    is a same-level stacking risk.  Record them (post-release should be
    empty).  ``warm_problem_live`` flags a previous ``WarmProblem`` still
    referenced by the cascade solver (its HiGHS instance would coexist with
    the new build); it is recorded as the violator ``"<warm_problem>"``.
    No-op unless ``FLEXTOOL_LEVEL_RELEASE_AUDIT=1``."""
    if os.environ.get("FLEXTOOL_LEVEL_RELEASE_AUDIT") != "1":
        return
    violators = [
        _k for _k, _step in (steps or {}).items()
        if getattr(getattr(_step, "solution", None), "highs", None) is not None
    ]
    if warm_problem_live:
        violators.append("<warm_problem>")
    _LEVEL_RELEASE_AUDIT.append({
        "kind": "cold_rebuild",
        "solve": complete_solve_name,
        "violators": violators,
    })
