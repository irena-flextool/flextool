"""Native-environment self-check and auto-remediation.

FlexTool's solve path loads compiled extensions — ``polars``, ``highspy``
(HiGHS, via ``polar_high``) — that can crash *natively* (not as a catchable
Python exception) when the installed wheel is wrong for the machine.  Two
real cases we have hit:

* The default ``polars`` wheel on a CPU without the SIMD instruction set it
  was built for: crashes on the first vectorised op.  Cure: the
  ``polars-lts-cpu`` build.
* ``highspy`` ``1.14.0`` crashes on *import* on older Windows (Server 2019 /
  Windows 10 1809-era), see ERGO-Code/HiGHS#2964 — a missing dependency in
  that wheel; works on Windows 11 / Linux.  Cure: pin ``highspy==1.13.1``.

Either way the symptom is a Windows access violation (``0xC0000005`` → exit
``3221225477``) or a POSIX ``SIGILL``/``SIGSEGV``.  A native fault
terminates the process, so it cannot be caught with ``try/except``; the only
robust way to observe it is to run the risky imports in a **child process**
and inspect its exit code — which is what :func:`probe_solver_stack` does.
It also pinpoints *which* extension died (via flushed progress markers), so
the right remedy can be applied.

This module imports **stdlib only** so it stays importable in a broken
environment, and it never imports ``polars``/``highspy`` in-process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import platform
import subprocess
import sys

# highspy 1.14.0 import-crashes on older Windows (HiGHS#2964); 1.13.1 is the
# last good release.  Used as the downgrade target.
HIGHSPY_GOOD_VERSION = "1.13.1"

# Probe program: import each native extension in turn and print a *flushed*
# marker after each one survives.  On a native crash the already-flushed
# markers remain in the captured pipe, so the last marker tells us how far
# we got — and therefore which component is the one that died.  polars is
# exercised with a real SIMD op (a bare import can pass on an incompatible
# CPU and only fault later); highspy and polar_high are constructed.
_PROBE_CODE = (
    "import sys\n"
    "def mark(m):\n"
    "    sys.stdout.write('PROBE:' + m + '\\n'); sys.stdout.flush()\n"
    "mark('begin')\n"
    "import polars as pl\n"
    "out = (pl.DataFrame({'a': list(range(1000)), 'b': list(range(1000))})\n"
    "         .sort('a', descending=True)\n"
    "         .group_by('a').agg(pl.col('b').sum())\n"
    "         .select(pl.col('b').sum())).to_series()[0]\n"
    "assert out == 499500, out\n"
    "mark('polars')\n"
    "import highspy\n"
    "highspy.Highs()\n"
    "mark('highspy')\n"
    "import polar_high\n"
    "polar_high.Problem()\n"
    "mark('polar_high')\n"
    "mark('all_ok')\n"
)

# Ordered probe checkpoints; the component that runs *after* a checkpoint is
# the suspect when the probe dies having last printed that checkpoint.
_CHECKPOINTS = ["begin", "polars", "highspy", "polar_high", "all_ok"]
_COMPONENT_AFTER = {
    "begin": "polars",
    "polars": "highspy",
    "highspy": "polar_high",
    "polar_high": None,
}

# Probe outcome statuses.
OK = "ok"                       # whole stack imported and computed correctly
NATIVE_FAULT = "native_fault"   # a process was killed by a native fault
ERROR = "error"                 # ordinary Python error (e.g. not installed)
TIMEOUT = "timeout"             # probe did not finish in time


@dataclasses.dataclass
class StackProbe:
    """Outcome of running :func:`probe_solver_stack` in a child process."""

    status: str
    returncode: int | None
    failed_component: str | None = None  # "polars" / "highspy" / "polar_high"
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
            return "solver stack: OK"
        if self.status == NATIVE_FAULT:
            comp = self.failed_component or "a solver library"
            return (
                f"solver stack: NATIVE CRASH in {comp} "
                f"({describe_fault(self.returncode)}) — this build is "
                f"incompatible with this computer"
            )
        if self.status == TIMEOUT:
            return "solver stack: probe timed out"
        return f"solver stack: error (exit {self.returncode})"


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


def _classify_failed_component(stdout: str) -> str | None:
    """Given the probe's (possibly truncated) stdout, return the component
    that was about to run when it died — the suspect for the crash."""
    seen = [c for c in _CHECKPOINTS if f"PROBE:{c}" in stdout]
    if not seen:
        return None
    return _COMPONENT_AFTER.get(seen[-1])


def probe_solver_stack(python: str | None = None, timeout: float = 120.0) -> StackProbe:
    """Import the native solver stack in a child process and classify the
    result, pinpointing which extension faulted.

    Isolating the imports in a subprocess is what makes a native crash
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
        return StackProbe(TIMEOUT, None)
    except OSError as exc:
        return StackProbe(ERROR, None, stderr=str(exc))

    rc = completed.returncode
    out, err = completed.stdout.strip(), completed.stderr.strip()
    if rc == 0 and "PROBE:all_ok" in completed.stdout:
        return StackProbe(OK, rc, None, out, err)
    if is_native_fault(rc):
        return StackProbe(
            NATIVE_FAULT, rc, _classify_failed_component(completed.stdout), out, err
        )
    return StackProbe(ERROR, rc, _classify_failed_component(completed.stdout), out, err)


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def installed_polars() -> tuple[str | None, str | None]:
    """Return ``(distribution_name, version)`` for whichever polars build is
    installed, without importing polars.  Either the standard ``polars`` or
    the ``polars-lts-cpu`` distribution provides the ``polars`` import name.
    """
    for dist in ("polars", "polars-lts-cpu"):
        ver = _dist_version(dist)
        if ver is not None:
            return dist, ver
    return None, None


def env_fingerprint() -> str:
    """Short, stable hash of the things that decide whether the probe needs
    to re-run: the interpreter, its version, the machine/OS, and the
    installed solver builds (polars + highspy).  Swapping or downgrading
    either build changes the fingerprint, so a fixed environment re-checks
    itself exactly once after any remediation.
    """
    pol_dist, pol_ver = installed_polars()
    parts = [
        sys.executable,
        sys.version.split()[0],
        platform.machine(),
        platform.system(),
        f"{pol_dist}=={pol_ver}",
        f"highspy=={_dist_version('highspy')}",
    ]
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def swap_to_lts_cpu_steps(python: str | None = None) -> list[list[str]]:
    """pip steps that replace the standard polars with ``polars-lts-cpu``.
    Both distributions own the ``polars`` import name, so the standard build
    must be uninstalled first — installing lts-cpu alone does not displace
    it.  Uninstalling a not-installed name is a no-op (pip warns, exits 0).
    """
    py = python or sys.executable
    return [
        [py, "-m", "pip", "uninstall", "-y", "polars", "polars-lts-cpu"],
        [py, "-m", "pip", "install", "polars-lts-cpu"],
    ]


def downgrade_highspy_steps(python: str | None = None) -> list[list[str]]:
    """pip step that pins highspy to the last release without the older-
    Windows import crash (HiGHS#2964)."""
    py = python or sys.executable
    return [[py, "-m", "pip", "install", f"highspy=={HIGHSPY_GOOD_VERSION}"]]


def _banner(message: str) -> str:
    return (
        "================================================================\n"
        f"  {message}\n"
        "  Re-installing the compatible version now...\n"
        "  (press Ctrl-C to stop)\n"
        "================================================================"
    )


# Per-component remediation.  ``steps`` is a callable(python) -> list of argv.
_REMEDIATIONS = {
    "polars": {
        "steps": swap_to_lts_cpu_steps,
        "banner": _banner(
            "Your computer needs a different 'polars' build to run FlexTool."
        ),
    },
    "highspy": {
        "steps": downgrade_highspy_steps,
        "banner": _banner(
            f"Your computer needs HiGHS solver 'highspy=={HIGHSPY_GOOD_VERSION}' "
            "to run FlexTool."
        ),
    },
}


def has_remediation(component: str | None) -> bool:
    return component in _REMEDIATIONS


def remediation_steps(component: str | None, python: str | None = None) -> list[list[str]] | None:
    spec = _REMEDIATIONS.get(component or "")
    return spec["steps"](python) if spec else None


def remediation_banner(component: str | None) -> str | None:
    spec = _REMEDIATIONS.get(component or "")
    return spec["banner"] if spec else None


# Message for the cases a package swap cannot fix (a polar_high crash, a
# swap that did not help, or a native crash outside the solver stack):
# genuinely environment-level.  The common Windows culprits are a venv
# built on Anaconda/conda (mixed native libraries), a missing Visual C++
# runtime, or running from a network/mapped drive under concurrency.
UNFIXABLE_HELP = (
    "This is an environment-level problem a package install cannot fix. A "
    "native crash like this usually means the Python environment itself is "
    "unstable — most often a venv built on Anaconda/conda (mixing native "
    "libraries), a missing Visual C++ runtime, or running from a network/"
    "mapped drive. Rebuild the venv from a clean python.org Python on a "
    "local disk (e.g. C:), not a network drive, and keep the project local too."
)


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


def diagnostics_report(probe: StackProbe | None = None) -> str:
    """Multi-line, copy-pasteable environment report.  Used by the
    ``__main__`` entry point, the self-update flow, and the GUI startup
    check so a screenshot or paste is enough to triage remotely.
    """
    pol_dist, pol_ver = installed_polars()
    if probe is None:
        probe = probe_solver_stack()
    lines = [
        "FlexTool environment check",
        f"  python      : {sys.executable}",
        f"  version     : {sys.version.split()[0]}",
        f"  platform    : {platform.platform()}",
        f"  machine     : {platform.machine()}",
        f"  cpu         : {_cpu_hint()}",
        f"  polars build: {pol_dist} {pol_ver}",
        f"  highspy     : {_dist_version('highspy')}",
        f"  {probe.summary()}",
    ]
    if probe.stderr and not probe.ok:
        lines.append("  --- probe stderr ---")
        lines.extend("  " + ln for ln in probe.stderr.splitlines())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m flextool.env_check`` — print diagnostics, exit non-zero
    if the solver stack cannot run here."""
    probe = probe_solver_stack()
    print(diagnostics_report(probe))
    if probe.is_native_fault and has_remediation(probe.failed_component):
        print("\nFix:")
        for step in remediation_steps(probe.failed_component):
            print("  " + " ".join(step))
        return 1
    if probe.is_native_fault:
        print("\n" + UNFIXABLE_HELP)
        return 1
    return 0 if probe.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
