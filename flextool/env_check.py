"""Native-environment self-check and auto-remediation.

FlexTool's solve path loads compiled extensions — chiefly ``polars`` and
``highspy`` (HiGHS) — that can crash *natively* (not as a catchable Python
exception) when the installed wheel is wrong for the machine.  The single
most common case is the default ``polars`` wheel on a CPU without the SIMD
instruction set it was built for: the process dies with a Windows access
violation (``0xC0000005`` → exit ``3221225477``) or illegal instruction
(``0xC000001D``), or a POSIX ``SIGILL``/``SIGSEGV``.  The cure is to swap
to the ``polars-lts-cpu`` build, which targets an older instruction
baseline.

A native fault terminates the process, so it cannot be caught with
``try/except`` in-process.  The only robust way to observe it is to run the
risky operation in a **child process** and inspect its exit code — which is
exactly what :func:`probe_polars` does.  :func:`swap_to_lts_cpu_steps`
returns the pip commands that fix the common case; callers run them (the
self-update flow does this unattended, the GUI asks for one click first).

This module deliberately imports **stdlib only** so it stays importable in
a broken environment.  It never imports ``polars`` in-process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import platform
import subprocess
import sys

# Loud banner shown right before an automatic re-install, so an unattended
# pip run never looks like silent tampering.
SWAP_HEADING = (
    "================================================================\n"
    "  Your computer needs a different 'polars' build to run FlexTool.\n"
    "  Re-installing the compatible version (polars-lts-cpu) now...\n"
    "  (press Ctrl-C to stop)\n"
    "================================================================"
)

# Probe program: import polars and run a real, SIMD-exercising operation.
# A bare ``import polars`` can succeed on an incompatible CPU and only fault
# later on the first vectorised op, so we force one here (sort + group-by +
# sum) before declaring the build usable.
_PROBE_CODE = (
    "import polars as pl\n"
    "df = pl.DataFrame({'a': list(range(1000)), 'b': list(range(1000))})\n"
    "out = (df.sort('a', descending=True)\n"
    "         .group_by('a').agg(pl.col('b').sum())\n"
    "         .select(pl.col('b').sum())).to_series()[0]\n"
    "assert out == 499500, out\n"
    "print('POLARS_PROBE_OK', pl.__version__)\n"
)

# Probe outcome statuses.
OK = "ok"                  # polars imported and computed correctly
NATIVE_FAULT = "native_fault"   # process killed by a native fault — swap-worthy
ERROR = "error"            # ordinary Python error (e.g. polars not installed)
TIMEOUT = "timeout"        # probe did not finish in time


@dataclasses.dataclass
class ProbeResult:
    """Outcome of running :func:`probe_polars` in a child process."""

    status: str
    returncode: int | None
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def is_native_fault(self) -> bool:
        return self.status == NATIVE_FAULT

    def summary(self) -> str:
        """One-line, copy-pasteable description of the outcome."""
        if self.status == OK:
            return "polars: OK"
        if self.status == NATIVE_FAULT:
            return (
                f"polars: NATIVE CRASH ({describe_fault(self.returncode)}) "
                f"— this build is incompatible with your CPU"
            )
        if self.status == TIMEOUT:
            return "polars: probe timed out"
        return f"polars: error (exit {self.returncode})"


def is_native_fault(returncode: int | None, plat: str | None = None) -> bool:
    """True if *returncode* indicates a native crash (segfault / illegal
    instruction / abort), as opposed to a normal Python exit.

    ``plat`` defaults to :data:`sys.platform`; it is a parameter so the
    classifier can be unit-tested for both platform families on one host.
    """
    if returncode is None or returncode == 0:
        return False
    plat = plat if plat is not None else sys.platform

    if plat.startswith("win"):
        # Windows exit codes are 32-bit DWORDs; NTSTATUS *fault* codes have
        # the top two bits set (0xC0000000 range): 0xC0000005 access
        # violation, 0xC000001D illegal instruction, 0xC00000FD stack
        # overflow, 0xC0000409 stack buffer overrun, etc.
        u = returncode & 0xFFFFFFFF
        return u >= 0xC0000000

    # POSIX: subprocess reports a signal-terminated child as ``-signal``.
    # Treat the fatal "the code itself is broken / corrupted memory" signals
    # as native faults; SIGILL=4, SIGABRT=6, SIGBUS=7, SIGFPE=8, SIGSEGV=11.
    if returncode < 0:
        return (-returncode) in (4, 6, 7, 8, 11)
    return False


def describe_fault(returncode: int | None, plat: str | None = None) -> str:
    """Human-readable name for a native-fault *returncode* (best effort)."""
    if returncode is None:
        return "no exit code"
    plat = plat if plat is not None else sys.platform
    if plat.startswith("win"):
        u = returncode & 0xFFFFFFFF
        names = {
            0xC0000005: "access violation (0xC0000005)",
            0xC000001D: "illegal instruction (0xC000001D)",
            0xC00000FD: "stack overflow (0xC00000FD)",
            0xC0000409: "stack buffer overrun (0xC0000409)",
            0xC0000094: "integer divide by zero (0xC0000094)",
        }
        return names.get(u, f"native fault (0x{u:08X})")
    if returncode < 0:
        sig = -returncode
        names = {
            4: "SIGILL (illegal instruction)",
            6: "SIGABRT (abort)",
            7: "SIGBUS (bus error)",
            8: "SIGFPE (floating-point exception)",
            11: "SIGSEGV (segmentation fault)",
        }
        return names.get(sig, f"signal {sig}")
    return f"exit code {returncode}"


def probe_polars(python: str | None = None, timeout: float = 90.0) -> ProbeResult:
    """Run polars in a child process and classify the result.

    Isolating the operation in a subprocess is what makes a native crash
    *observable* instead of fatal to us: we read the child's exit code.
    *python* defaults to the current interpreter (:data:`sys.executable`),
    which is also the interpreter the GUI uses to launch solves.
    """
    py = python or sys.executable
    try:
        completed = subprocess.run(
            [py, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(TIMEOUT, None)
    except OSError as exc:
        return ProbeResult(ERROR, None, stderr=str(exc))

    rc = completed.returncode
    if rc == 0 and "POLARS_PROBE_OK" in completed.stdout:
        return ProbeResult(OK, rc, completed.stdout.strip(), completed.stderr.strip())
    if is_native_fault(rc):
        return ProbeResult(
            NATIVE_FAULT, rc, completed.stdout.strip(), completed.stderr.strip()
        )
    return ProbeResult(ERROR, rc, completed.stdout.strip(), completed.stderr.strip())


def installed_polars() -> tuple[str | None, str | None]:
    """Return ``(distribution_name, version)`` for whichever polars build is
    installed, without importing polars.  Either the standard ``polars`` or
    the ``polars-lts-cpu`` distribution provides the ``polars`` import name.
    """
    for dist in ("polars", "polars-lts-cpu"):
        try:
            return dist, importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None, None


def env_fingerprint() -> str:
    """Short, stable hash of the things that decide whether the probe needs
    to re-run: the interpreter, its version, the machine/OS, and the
    installed polars build+version.  Swapping polars changes the
    fingerprint, so a fixed environment re-checks itself exactly once.
    """
    dist, ver = installed_polars()
    parts = [
        sys.executable,
        sys.version.split()[0],
        platform.machine(),
        platform.system(),
        f"{dist}=={ver}",
    ]
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def swap_to_lts_cpu_steps(python: str | None = None) -> list[list[str]]:
    """pip command steps that replace the standard polars with
    ``polars-lts-cpu``.  Both distributions own the ``polars`` import name,
    so the standard build must be uninstalled first — installing lts-cpu
    alone does not displace it.  Uninstalling a not-installed name is a
    no-op (pip warns, exits 0).
    """
    py = python or sys.executable
    return [
        [py, "-m", "pip", "uninstall", "-y", "polars", "polars-lts-cpu"],
        [py, "-m", "pip", "install", "polars-lts-cpu"],
    ]


def _cpu_hint() -> str:
    """Best-effort CPU description for the diagnostics block (stdlib only)."""
    proc = platform.processor() or "unknown"
    avx2 = ""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("flags"):
                        avx2 = "; AVX2" if " avx2 " in f" {line} " else "; no AVX2"
                        break
        except OSError:
            pass
    return f"{proc}{avx2}"


def diagnostics_report(probe: ProbeResult | None = None) -> str:
    """Multi-line, copy-pasteable environment report.  Used by the
    ``__main__`` entry point, the self-update flow, and the GUI startup
    check so a screenshot or paste is enough to triage remotely.
    """
    dist, ver = installed_polars()
    if probe is None:
        probe = probe_polars()
    lines = [
        "FlexTool environment check",
        f"  python      : {sys.executable}",
        f"  version     : {sys.version.split()[0]}",
        f"  platform    : {platform.platform()}",
        f"  machine     : {platform.machine()}",
        f"  cpu         : {_cpu_hint()}",
        f"  polars build: {dist} {ver}",
        f"  {probe.summary()}",
    ]
    if probe.stderr and not probe.ok:
        lines.append("  --- probe stderr ---")
        lines.extend("  " + ln for ln in probe.stderr.splitlines())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m flextool.env_check`` — print diagnostics, exit non-zero
    if polars cannot run here."""
    probe = probe_polars()
    print(diagnostics_report(probe))
    if probe.is_native_fault:
        print()
        print(
            "Fix: replace polars with the compatible build:\n"
            "  " + "  ".join(swap_to_lts_cpu_steps()[0]) + "\n"
            "  " + "  ".join(swap_to_lts_cpu_steps()[1])
        )
        return 1
    return 0 if probe.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
