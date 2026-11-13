"""Native-environment self-check and auto-remediation.

FlexTool's solve path loads compiled extensions — ``polars``, ``highspy``
(HiGHS, via ``polar_high``) — that can crash *natively* (not as a catchable
Python exception) when the installed wheel is wrong for the machine.  Two
real cases we have hit:

* The default ``polars`` runtime on a CPU without the instructions it was
  built for (AVX2, FMA, BMI2, MOVBE, ...; x86-64-v3): crashes on the first
  vectorised op.  Cure: add the ``polars-runtime-compat`` build (the
  ``polars[rtcompat]`` extra, x86-64-v2), which polars loads in preference
  to the default runtime.  :func:`detect_cpu_features` reads the CPU flags
  without importing polars, so the cure can be installed *before* the first
  crash, and a CPU below even x86-64-v2 gets a clear "unsupported" message.
* ``highspy`` ``1.14.0`` crashes on *import* on older Windows (Server 2019 /
  Windows 10 1809-era), see ERGO-Code/HiGHS#2964; works on Windows 11 /
  Linux.  highspy up to 1.14 uses the system's ``msvcp140.dll`` (an old one
  is the likely cause); from 1.15 the wheel ships its own.  FlexTool needs
  highspy 1.15.1 or newer, so the cure for an older highspy is to install
  one within FlexTool's requirement.

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
import importlib.util
import os
import platform
import re
import signal
import subprocess
import sys

# highspy from this release on ships its own C++ runtime (msvcp140) instead
# of importing the system's; older ones import-crash on older Windows
# (HiGHS#2964, 1.14.0).
HIGHSPY_OWN_CPP_RUNTIME = (1, 15)

# The Python children started here (the probe, the pip steps) keep the current folder off sys.path, so a
# folder named like a package in the workspace is never imported instead of it.  Spelled here, not imported from
# ``flextool._child_python``: this module is stdlib-only and is loaded without importing the flextool package.
_SAFE_PATH_FLAG = "-P"


def _safe_path_env() -> dict[str, str]:
    """This process's environment with ``PYTHONSAFEPATH=1`` (see ``flextool._child_python.child_env``)."""
    return {**os.environ, "PYTHONSAFEPATH": "1"}


def _pip_env() -> dict[str, str]:
    """This process's environment without ``PYTHONSAFEPATH``, for pip (see ``flextool._child_python.pip_env``)."""
    return {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}

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
    "sys.stdout.write('RUNTIME:' + str(getattr(sys.modules.get('polars._plr'), 'RUNTIME_REPR', '?')) + '\\n')\n"
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

    @property
    def polars_runtime(self) -> str | None:
        """The native runtime polars loaded in the probe (``"rt32"``,
        ``"rt64"``, ``"rtcompat"``), or ``None`` if it got no further."""
        match = re.search(r"^RUNTIME:(\S+)", self.stdout, re.MULTILINE)
        return match.group(1) if match else None

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


#: The POSIX signals that mean the code itself is broken or its memory is
#: corrupt (a native fault), by name, with what each means.
_FAULT_SIGNALS = {
    "SIGILL": "illegal instruction",
    "SIGABRT": "abort",
    "SIGBUS": "bus error",
    "SIGFPE": "floating-point exception",
    "SIGSEGV": "segmentation fault",
}


def signal_name(signum: int | None) -> str | None:
    """The name of signal *signum* (``"SIGBUS"``) in the running system's
    numbering, or ``None`` when it has no signal with that number.

    Signal numbers differ between systems: 10 is SIGBUS on macOS and SIGUSR1
    on Linux, 7 is SIGBUS on Linux and SIGEMT on macOS.  An exit code
    ``-signum`` comes from a child on this computer, so it is named with
    this computer's table (:class:`signal.Signals`).
    """
    try:
        return signal.Signals(signum).name
    except (TypeError, ValueError):
        return None


def is_native_fault(returncode: int | None, plat: str | None = None) -> bool:
    """True if *returncode* indicates a native crash (segfault / illegal
    instruction / abort), as opposed to a normal Python exit.

    ``plat`` defaults to :data:`sys.platform`; it is a parameter so the
    classifier can be unit-tested for both platform families on one host.
    A POSIX ``-signum`` is named in the running system's numbering
    (:func:`signal_name`).
    """
    if returncode is None or returncode == 0:
        return False
    plat = plat if plat is not None else sys.platform

    if plat.startswith("win"):
        # Windows exit codes are 32-bit DWORDs; NTSTATUS *fault* codes have
        # the top two bits set (0xC0000000 range): 0xC0000005 access
        # violation, 0xC000001D illegal instruction, 0xC00000FD stack
        # overflow, 0xC0000409 stack buffer overrun, etc.  0xC000013A
        # (STATUS_CONTROL_C_EXIT) is in that range but is a Ctrl-C, not a crash.
        u = returncode & 0xFFFFFFFF
        return u >= 0xC0000000 and u != 0xC000013A

    # POSIX: subprocess reports a signal-terminated child as ``-signal``.
    # The fatal "the code itself is broken / corrupted memory" signals are
    # native faults, by name: the numbers differ between systems.
    if returncode < 0:
        return signal_name(-returncode) in _FAULT_SIGNALS
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
        name = signal_name(-returncode)
        if name is None:
            return f"signal {-returncode}"
        meaning = _FAULT_SIGNALS.get(name)
        return f"{name} ({meaning})" if meaning else name
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
            [py, _SAFE_PATH_FLAG, "-c", _PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_safe_path_env(),
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
    the retired ``polars-lts-cpu`` distribution provides the ``polars``
    import name.  ``polars-lts-cpu`` wins when both are registered: it
    overwrote the standard build's files when it was installed over it.
    """
    for dist in ("polars-lts-cpu", "polars"):
        ver = _dist_version(dist)
        if ver is not None:
            return dist, ver
    return None, None


# polars >= 1.34 is a pure-Python ``polars`` package plus one or more native
# runtimes.  polars' loader (``polars/_plr.py``) tries ``compat``, then
# ``64``, then ``32``, and skips a runtime whose version differs from the
# ``polars`` package's.  ``POLARS_FORCE_PKG`` / ``POLARS_PREFER_PKG`` override
# the order.
_POLARS_RUNTIMES = ("compat", "64", "32")


def polars_runtimes() -> dict[str, str]:
    """Installed polars native runtimes, ``{"32": "1.40.1", "compat": ...}``
    (only the installed ones), read from package metadata."""
    found = {}
    for rt in _POLARS_RUNTIMES:
        ver = _dist_version(f"polars-runtime-{rt}")
        if ver is not None:
            found[rt] = ver
    return found


def default_runtime_installed() -> str | None:
    """The installed default runtime (``"64"`` before ``"32"``, polars' own
    order) whose version matches the ``polars`` package, or ``None``."""
    dist, ver = installed_polars()
    if dist != "polars" or ver is None:
        return None
    runtimes = polars_runtimes()
    for rt in ("64", "32"):
        if runtimes.get(rt) == ver:
            return rt
    return None


def prefer_default_runtime(verdict: CpuVerdict) -> str | None:
    """Make polars load its default runtime on a CPU that runs it, also when
    the compatible runtime is installed next to it (the Toolbox bundle ships
    both; ``flextool[compat]`` installs both).  polars loads the compatible
    runtime first whenever it is installed, which costs about 10-30% speed on
    an AVX2 CPU.  Sets ``POLARS_PREFER_PKG`` (read by ``polars/_plr.py`` at
    import; child processes inherit it) and returns the runtime, or ``None``
    when nothing is changed: the CPU is not confirmed :data:`CPU_FULL`, no
    compatible runtime is installed, no default runtime of the polars
    version is installed, or ``POLARS_PREFER_PKG`` is already set."""
    if verdict.status != CPU_FULL or os.environ.get("POLARS_PREFER_PKG") or "compat" not in polars_runtimes():
        return None
    rt = default_runtime_installed()
    if rt is not None:
        os.environ["POLARS_PREFER_PKG"] = rt
    return rt


def compat_runtime_ready() -> bool:
    """True if the standard ``polars`` is installed together with a
    ``polars-runtime-compat`` of the same version — the one polars loads."""
    dist, ver = installed_polars()
    return dist == "polars" and ver is not None and polars_runtimes().get("compat") == ver


def env_fingerprint() -> str:
    """Short, stable hash of the things that decide whether the probe needs
    to re-run: the interpreter, its version, the machine/OS, and the
    installed solver builds (polars and its runtimes, highspy).  Adding a
    runtime or downgrading highspy changes the fingerprint, so a fixed environment re-checks
    itself exactly once after any remediation.
    """
    pol_dist, pol_ver = installed_polars()
    parts = [
        sys.executable,
        sys.version.split()[0],
        platform.machine(),
        platform.system(),
        f"{pol_dist}=={pol_ver}",
        ",".join(f"{rt}=={v}" for rt, v in sorted(polars_runtimes().items())),
        f"highspy=={_dist_version('highspy')}",
    ]
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


# ── CPU features (read without importing polars) ───────────────────

# Instruction-set features, spelled as polars' ``BUILD_FEATURE_FLAGS`` spell
# them.  ``polars-runtime-compat`` needs x86-64-v2; the default runtimes
# (``polars-runtime-32`` / ``-64``) need x86-64-v3 plus MOVBE.  The installed
# runtimes' own ``build_feature_flags.py`` is read when present (see
# :func:`runtime_required_features`); these are the fallbacks, taken from the
# 1.40.1 and 2.1.0 wheels.
X86_64_V2_FEATURES = ("sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cmpxchg16b")
DEFAULT_RUNTIME_FEATURES = X86_64_V2_FEATURES + (
    "avx", "avx2", "fma", "bmi1", "bmi2", "lzcnt", "pclmulqdq", "movbe",
)

# Linux ``/proc/cpuinfo`` flag name -> polars feature name.
_LINUX_FLAGS = {
    "pni": "sse3", "ssse3": "ssse3", "sse4_1": "sse4.1", "sse4_2": "sse4.2",
    "popcnt": "popcnt", "cx16": "cmpxchg16b", "avx": "avx", "avx2": "avx2",
    "fma": "fma", "bmi1": "bmi1", "bmi2": "bmi2", "abm": "lzcnt",
    "pclmulqdq": "pclmulqdq", "movbe": "movbe",
}

# Windows ``IsProcessorFeaturePresent`` ids (winnt.h) -> polars feature name.
# There are no ids for POPCNT, FMA, BMI1/2, LZCNT, PCLMULQDQ or MOVBE, so
# this source can show a missing feature but never confirm the full set.
_WIN_PF_LEGACY = {13: "sse3", 14: "cmpxchg16b"}
_WIN_PF_NEWER = {36: "ssse3", 37: "sse4.1", 38: "sse4.2", 39: "avx", 40: "avx2"}

_X86_MACHINES = {"x86_64", "amd64", "x64", "i386", "i686", "x86"}

# Verdicts of :func:`classify_cpu`.
CPU_FULL = "full"                  # runs the default polars runtime
CPU_NEEDS_COMPAT = "needs_compat"  # x86-64-v2 but not v3: needs polars-runtime-compat
CPU_UNSUPPORTED = "unsupported"    # below x86-64-v2: no polars build runs
CPU_UNKNOWN = "unknown"            # x86, features could not be read (fully)
CPU_NOT_X86 = "not_x86"            # e.g. Apple silicon / ARM: not applicable


@dataclasses.dataclass
class CpuFeatures:
    """CPU instruction-set flags and where they were read from.  ``flags``
    holds only the features the source could report; a feature that is not
    a key is unknown, not missing."""

    source: str
    flags: dict[str, bool]


@dataclasses.dataclass
class CpuVerdict:
    """Outcome of :func:`classify_cpu`."""

    status: str
    source: str = ""
    missing_default: list[str] = dataclasses.field(default_factory=list)
    missing_compat: list[str] = dataclasses.field(default_factory=list)

    def describe(self) -> str:
        """Short text for the diagnostics block."""
        if self.status == CPU_FULL:
            return f"AVX2-class (x86-64-v3), runs the default polars build [{self.source}]"
        if self.status == CPU_NEEDS_COMPAT:
            return (
                f"no {', '.join(self.missing_default)}: needs the compatible polars build "
                f"(polars[rtcompat]) [{self.source}]"
            )
        if self.status == CPU_UNSUPPORTED:
            return f"no {', '.join(self.missing_compat)}: below x86-64-v2, not supported [{self.source}]"
        if self.status == CPU_NOT_X86:
            return "not x86 (no CPU feature check needed)"
        return f"features not fully known [{self.source or 'no source'}]"


def parse_proc_cpuinfo(text: str) -> CpuFeatures | None:
    """CPU features from the text of Linux ``/proc/cpuinfo``; ``None`` if it
    has no x86 ``flags`` line (e.g. an ARM machine lists ``Features``)."""
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "flags":
            present = set(value.split())
            return CpuFeatures(
                "/proc/cpuinfo", {name: flag in present for flag, name in _LINUX_FLAGS.items()}
            )
    return None


def windows_feature_flags(is_present) -> CpuFeatures:
    """CPU features from Windows ``IsProcessorFeaturePresent``.

    *is_present* is the API (``int -> bool``); a parameter so tests can mock
    it.  Windows releases that predate an id return False for it, so when
    none of the newer ids (SSSE3 ... AVX2) reads True they are treated as
    unknown: a pre-SSSE3 CPU and an OS that does not know the ids look alike.
    """
    flags = {name: bool(is_present(pf)) for pf, name in _WIN_PF_LEGACY.items()}
    newer = {name: bool(is_present(pf)) for pf, name in _WIN_PF_NEWER.items()}
    if any(newer.values()):
        flags.update(newer)
    return CpuFeatures("IsProcessorFeaturePresent", flags)


def _polars_cpuid_flags() -> CpuFeatures | None:
    """CPU features from the ``cpuid`` instruction, using polars' own
    stdlib-only checker (``polars/_cpu_check.py``, the code ``import polars``
    runs before it loads a runtime).  The file is loaded on its own, so the
    polars package and its native runtime are not imported."""
    try:
        spec = importlib.util.find_spec("polars")
        locations = list(spec.submodule_search_locations or []) if spec else []
        for loc in locations:
            path = os.path.join(loc, "_cpu_check.py")
            if not os.path.isfile(path):
                continue
            mod_spec = importlib.util.spec_from_file_location("_flextool_polars_cpu_check", path)
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)
            flags = module._read_cpu_flags()
            if flags:
                return CpuFeatures("cpuid", {k: bool(v) for k, v in flags.items()})
    except Exception:  # noqa: BLE001 - any failure means "source unavailable"
        return None
    return None


def merge_cpu_features(*sources: CpuFeatures) -> CpuFeatures:
    """One feature set from several sources: a feature is present only if
    every source that reports it says present, and missing if any says
    missing."""
    flags: dict[str, bool] = {}
    for features in sources:
        for name, present in features.flags.items():
            flags[name] = flags.get(name, True) and present
    return CpuFeatures("+".join(f.source for f in sources), flags)


def detect_cpu_features() -> CpuFeatures | None:
    """Read the CPU's instruction-set flags without importing polars.

    ``None`` if the machine is not x86 (nothing to check) or no source could
    be read.  Linux reads ``/proc/cpuinfo`` (which also reflects features the
    kernel disabled) and ``cpuid`` through polars' checker, and a feature
    counts as present only when both say so: under a user-mode emulator
    (``qemu-x86_64 -cpu Nehalem``) ``/proc/cpuinfo`` is the host's, while
    ``cpuid`` and the instructions that run are the emulated CPU's.
    Elsewhere ``cpuid``, and on Windows ``IsProcessorFeaturePresent`` when
    polars is not installed.
    """
    if platform.machine().lower() not in _X86_MACHINES:
        return None
    from_kernel = None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                from_kernel = parse_proc_cpuinfo(fh.read())
        except OSError:
            pass
    from_cpuid = _polars_cpuid_flags()
    if from_kernel is not None and from_cpuid is not None:
        return merge_cpu_features(from_kernel, from_cpuid)
    if from_kernel is not None or from_cpuid is not None:
        return from_kernel or from_cpuid
    if sys.platform.startswith("win"):
        try:
            import ctypes

            return windows_feature_flags(ctypes.windll.kernel32.IsProcessorFeaturePresent)
        except Exception:  # noqa: BLE001
            return None
    return None


def parse_build_feature_flags(text: str) -> tuple[str, ...]:
    """Feature names from a polars runtime's ``BUILD_FEATURE_FLAGS`` string
    (``"+sse3,+avx2,..."``), the same way ``polars/_cpu_check.py`` reads it."""
    flags = [f.strip().lstrip("+") for f in text.split(",") if not f.strip().startswith("-")]
    return tuple(f for f in flags if f and f != "ctr-static")


def runtime_required_features(runtime: str) -> tuple[str, ...] | None:
    """The CPU features an installed polars runtime (``"32"``, ``"64"``,
    ``"compat"``) was built for, read from its ``build_feature_flags.py``
    without importing it; ``None`` if the runtime is not installed."""
    try:
        spec = importlib.util.find_spec(f"_polars_runtime_{runtime}")
    except (ImportError, ValueError):
        return None
    for loc in (spec.submodule_search_locations or []) if spec else []:
        path = os.path.join(loc, "build_feature_flags.py")
        try:
            with open(path, encoding="utf-8") as fh:
                match = re.search(r'BUILD_FEATURE_FLAGS\s*=\s*"([^"]*)"', fh.read())
        except OSError:
            continue
        if match:
            return parse_build_feature_flags(match.group(1))
    return None


def classify_cpu(
    features: CpuFeatures | None,
    default_required: tuple[str, ...] = DEFAULT_RUNTIME_FEATURES,
    compat_required: tuple[str, ...] = X86_64_V2_FEATURES,
    is_x86: bool = True,
) -> CpuVerdict:
    """Decide which polars build this CPU can run.  Only features the source
    reports as absent count as missing; unknown ones never trigger an
    action (the child-process probe stays the backstop)."""
    if not is_x86:
        return CpuVerdict(CPU_NOT_X86)
    if features is None:
        return CpuVerdict(CPU_UNKNOWN)
    flags = features.flags
    missing_compat = [f for f in compat_required if flags.get(f) is False]
    missing_default = [f for f in default_required if flags.get(f) is False]
    if missing_compat:
        return CpuVerdict(CPU_UNSUPPORTED, features.source, missing_default, missing_compat)
    if missing_default:
        return CpuVerdict(CPU_NEEDS_COMPAT, features.source, missing_default)
    if all(flags.get(f) is True for f in default_required):
        return CpuVerdict(CPU_FULL, features.source)
    return CpuVerdict(CPU_UNKNOWN, features.source)


def cpu_verdict() -> CpuVerdict:
    """:func:`classify_cpu` for this machine, using the installed runtimes'
    own feature lists when available."""
    is_x86 = platform.machine().lower() in _X86_MACHINES
    default_req = runtime_required_features("64") or runtime_required_features("32")
    compat_req = runtime_required_features("compat")
    return classify_cpu(
        detect_cpu_features() if is_x86 else None,
        default_req or DEFAULT_RUNTIME_FEATURES,
        compat_req or X86_64_V2_FEATURES,
        is_x86=is_x86,
    )


# ── remediation steps ─────────────────────────────────────────────

# flextool's own polars / highspy requirements, used when the installed
# flextool's metadata cannot be read (e.g. running from a source checkout).
_POLARS_SPEC_FALLBACK = ">=1.40,<3"
_HIGHSPY_SPEC_FALLBACK = ">=1.15.1,<1.16"


def _flextool_requirement_spec(name: str, fallback: str) -> str:
    """The version specifier of flextool's requirement on *name* (e.g.
    ``">=1.40,<3"`` for ``polars``), from the installed flextool's metadata."""
    try:
        requirements = importlib.metadata.requires("flextool") or []
    except importlib.metadata.PackageNotFoundError:
        requirements = []
    for req in requirements:
        if "extra ==" in req:
            continue
        match = re.match(rf"^{re.escape(name)}\s*(?:\[[^\]]*\])?\s*([^;]*)", req.strip())
        if match and not req.strip().lower().startswith((f"{name}-", f"{name}_")):
            spec = match.group(1).strip().strip("()").replace(" ", "")
            if spec:
                return spec
    return fallback


def _flextool_polars_spec() -> str:
    return _flextool_requirement_spec("polars", _POLARS_SPEC_FALLBACK)


def polars_compat_steps(python: str | None = None) -> list[list[str]] | None:
    """pip steps that make polars load its compatible runtime
    (``polars-runtime-compat``, x86-64-v2), or ``None`` if it already does.

    * Standard ``polars`` X: ``pip install "polars[rtcompat]==X"`` adds the
      compat runtime of the same version next to the default one; polars
      prefers it.  Pinning to X keeps every other package as it is.
    * The retired ``polars-lts-cpu`` (stops at 1.33.1, below flextool's
      polars floor): uninstall it (and a ``polars`` it overwrote), then
      install ``polars[rtcompat]`` within flextool's requirement.
    """
    py = python or sys.executable
    pip = [py, _SAFE_PATH_FLAG, "-m", "pip"]
    if _dist_version("polars-lts-cpu") is not None:
        return [
            pip + ["uninstall", "-y", "polars-lts-cpu", "polars"],
            pip + ["install", f"polars[rtcompat]{_flextool_polars_spec()}"],
        ]
    ver = _dist_version("polars")
    if ver is None:
        return [pip + ["install", f"polars[rtcompat]{_flextool_polars_spec()}"]]
    if compat_runtime_ready():
        return None
    return [pip + ["install", f"polars[rtcompat]=={ver}"]]


def _polars_remedy(python: str | None = None) -> list[list[str]] | None:
    """The polars remedy, unless the CPU is below what any polars build
    needs (then nothing can be installed that runs)."""
    if cpu_verdict().status == CPU_UNSUPPORTED:
        return None
    return polars_compat_steps(python)


def proactive_polars_fix_needed() -> bool:
    """True if this CPU lacks a feature the default polars runtime needs and
    the compatible runtime is not in place yet — fix before polars crashes."""
    return cpu_verdict().status == CPU_NEEDS_COMPAT and polars_compat_steps() is not None


def _release(version: str | None) -> tuple[int, ...]:
    """The leading numeric release of a version string (``"1.15.1"`` ->
    ``(1, 15, 1)``)."""
    match = re.match(r"\d+(?:\.\d+)*", version or "")
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def highspy_upgrade_steps(python: str | None = None) -> list[list[str]] | None:
    """pip step that replaces a highspy older than 1.15 (it uses the
    system's ``msvcp140.dll``; 1.14.0 import-crashes on older Windows,
    HiGHS#2964) with one within flextool's requirement, which ships its own
    C++ runtime; ``None`` when highspy is 1.15 or newer already (or not
    installed: a crash is then not highspy's)."""
    installed = _release(_dist_version("highspy"))
    if not installed or installed >= HIGHSPY_OWN_CPP_RUNTIME:
        return None
    py = python or sys.executable
    return [[py, _SAFE_PATH_FLAG, "-m", "pip", "install", f"highspy{_flextool_requirement_spec('highspy', _HIGHSPY_SPEC_FALLBACK)}"]]


def _banner(message: str) -> str:
    return (
        "================================================================\n"
        f"  {message}\n"
        "  Re-installing the compatible version now...\n"
        "  (press Ctrl-C to stop)\n"
        "================================================================"
    )


# Per-component remediation.  ``steps`` is a callable(python) -> list of
# argv, or ``None`` when no install can help on this machine.
_REMEDIATIONS = {
    "polars": {
        "steps": _polars_remedy,
        "banner": _banner(
            "Your computer needs the compatible 'polars' build (polars[rtcompat]) to run FlexTool."
        ),
    },
    "highspy": {
        "steps": highspy_upgrade_steps,
        "banner": _banner(
            "Your computer needs a newer HiGHS solver (highspy 1.15 or later, which brings its own "
            "C++ runtime) to run FlexTool."
        ),
    },
}


def has_remediation(component: str | None) -> bool:
    """True if a package install can fix a crash of *component* here."""
    return remediation_steps(component) is not None


def remediation_steps(component: str | None, python: str | None = None) -> list[list[str]] | None:
    spec = _REMEDIATIONS.get(component or "")
    return spec["steps"](python) if spec else None


def remediation_banner(component: str | None) -> str | None:
    spec = _REMEDIATIONS.get(component or "")
    return spec["banner"] if spec else None


def proactive_banner(verdict: CpuVerdict | None = None) -> str:
    """Banner for installing the compatible polars build before a crash."""
    verdict = verdict or cpu_verdict()
    missing = ", ".join(verdict.missing_default) or "AVX2"
    return _banner(
        f"This computer's processor has no {missing}, which the default 'polars' build needs.\n"
        "  FlexTool installs the compatible build (polars[rtcompat], about 10-30% slower)."
    )


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


# For an ordinary ImportError "DLL load failed" (not a native crash): a
# compiled extension did not find a DLL, most often the Microsoft Visual C++
# runtime (msvcp140.dll) on a Windows without the redistributable.
DLL_LOAD_HELP = (
    "A library could not load a DLL it needs. On Windows this is most often the Microsoft Visual C++ "
    "2015-2022 runtime (msvcp140.dll): install the Visual C++ Redistributable (x64) from "
    "https://aka.ms/vs/17/release/vc_redist.x64.exe and try again."
)


def unsupported_cpu_help(verdict: CpuVerdict) -> str:
    """Message for a CPU below x86-64-v2, which no current build supports."""
    return (
        "This computer's processor cannot run FlexTool 4: it has no "
        f"{', '.join(verdict.missing_compat)}. FlexTool's libraries (polars, numpy, "
        "pyarrow and the Qt user interface) need at least an x86-64-v2 processor "
        "(SSE4.2 and POPCNT: Intel Core i3/i5/i7 from 2009 on, AMD FX or Ryzen; not "
        "Core 2 or Phenom II). No package install can fix this; please use a newer computer."
    )


def unfixable_help() -> str:
    """The message to show when no package install fixes a native crash:
    the unsupported-CPU text on a CPU below x86-64-v2, otherwise
    :data:`UNFIXABLE_HELP`."""
    verdict = cpu_verdict()
    if verdict.status == CPU_UNSUPPORTED:
        return unsupported_cpu_help(verdict)
    return UNFIXABLE_HELP


def _stderr(message: str) -> None:
    """Write to stderr if there is one (``pythonw.exe`` has none)."""
    if sys.stderr is not None:
        sys.stderr.write(message + "\n")
        sys.stderr.flush()


class CompatRuntimeRequired(ImportError):
    """``import flextool`` stopped: this processor needs the compatible
    polars build, which is not installed, and FlexTool did not install it
    (no terminal to ask in, or the user said no).  The message gives the
    reason and the command that installs it."""


def format_command(argv: list[str]) -> str:
    """*argv* as a command line to copy into this platform's shell."""
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    import shlex

    return shlex.join(argv)


def compat_runtime_reason(verdict: CpuVerdict | None = None) -> str:
    """Why this processor needs the compatible polars build."""
    verdict = verdict or cpu_verdict()
    missing = ", ".join(verdict.missing_default) or "AVX2"
    return (
        f"This computer's processor has no {missing}. The standard build of the 'polars' data library, "
        "which FlexTool uses, is compiled for processors with these instructions (x86-64-v3: most "
        "processors from 2013 on) and crashes on this one. The compatible build (polars[rtcompat], "
        "about 10-30% slower) runs on this processor."
    )


def compat_runtime_message(verdict: CpuVerdict | None = None, steps: list[list[str]] | None = None) -> str:
    """The reason plus the exact command(s) that install the compatible build."""
    steps = steps if steps is not None else polars_compat_steps()
    commands = "\n".join("  " + format_command(step) for step in steps or [])
    return f"{compat_runtime_reason(verdict)}\n\nTo install it, run:\n{commands}"


# How the import-time check behaves, by what started this Python process:
# ENTRY_ASK asks in the terminal (or stops with CompatRuntimeRequired when
# there is none); ENTRY_GUI (the desktop GUI) asks in the terminal when it
# has one, otherwise in a small tkinter yes/no dialog, and exits on no; the
# diagnostics command prints the fix itself, so the check only notes it
# (ENTRY_DEFER); "Update FlexTool" is a repair the user asked for, so it
# installs without asking (ENTRY_REPAIR).
ENTRY_ASK = "ask"
ENTRY_GUI = "gui"
ENTRY_DEFER = "defer"
ENTRY_REPAIR = "repair"
_ENTRY_MODULES = {
    "flextool.gui": ENTRY_GUI,
    "flextool.gui.__main__": ENTRY_GUI,
    "flextool.env_check": ENTRY_DEFER,
    "flextool.cli.cmd_update_flextool": ENTRY_REPAIR,
}
_ENTRY_SCRIPTS = {
    "flextool": ENTRY_GUI,
    "flextool-gui": ENTRY_GUI,
    "flextool-update": ENTRY_REPAIR,
    "update_flextool": ENTRY_REPAIR,
}


def _script_mode(path: str) -> str:
    name = re.split(r"[\\/]", path)[-1].lower()
    for suffix in (".exe", ".pyw", ".py", "-script"):
        name = name.removesuffix(suffix)
    return _ENTRY_SCRIPTS.get(name, ENTRY_ASK)


def import_entry_mode(orig_argv: list[str] | None = None, argv0: str | None = None) -> str:
    """:data:`ENTRY_ASK`, :data:`ENTRY_GUI`, :data:`ENTRY_DEFER` or :data:`ENTRY_REPAIR` for
    the command line that started this process.  Read from
    ``sys.orig_argv`` (while ``python -m pkg.mod`` imports ``pkg``,
    ``sys.argv[0]`` is still ``-m``), then from the script in
    ``sys.argv[0]`` (an installed launcher such as ``flextool-update.exe``)."""
    argv = list(sys.orig_argv if orig_argv is None else orig_argv)[1:]
    argv0 = (sys.argv[0] if sys.argv else "") if argv0 is None else argv0
    mode = ENTRY_ASK
    skip_next = False
    for i, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "-m":
            mode = _ENTRY_MODULES.get(argv[i + 1] if i + 1 < len(argv) else "", ENTRY_ASK)
            break
        if arg.startswith("-m"):
            mode = _ENTRY_MODULES.get(arg[2:], ENTRY_ASK)
            break
        if arg in ("-c", "-") or (arg.startswith("-c") and not arg.startswith("--")):
            break
        if arg in ("-W", "-X", "--check-hash-based-pycs"):
            skip_next = True
            continue
        if not arg.startswith("-"):
            mode = _script_mode(arg)
            break
    if mode == ENTRY_ASK and argv0 not in ("", "-m", "-c"):
        mode = _script_mode(argv0)
    return mode


def _has_terminal() -> bool:
    """True if stdin and stdout are both a terminal, so a question can be
    asked and answered (not a GUI subprocess, Toolbox tool, CI or pytest)."""
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def _run_steps(steps: list[list[str]], run) -> int | str:
    """Run pip *steps* in order, echoing each on stderr; 0, or the first
    failing exit code (or why it could not start)."""
    for step in steps:
        _stderr("$ " + format_command(step))
        try:
            returncode = run(step, env=_pip_env()).returncode
        except OSError as exc:
            returncode = f"not started ({exc})"
        if returncode != 0:
            return returncode
    importlib.invalidate_caches()
    return 0


_DIALOG_TITLE = "FlexTool needs the compatible polars build"


def _no_console_window() -> dict:
    """``subprocess`` options that keep pip from opening a console window
    when the GUI runs under ``pythonw.exe``."""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _open_dialog_root():
    """A hidden tkinter root for the import-time dialogs, or ``None`` when
    no window can be opened (tkinter missing, no display)."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        return root
    except Exception:  # noqa: BLE001  (TclError, ImportError, missing display)
        return None


def _dialog_notice(title: str, text: str) -> None:
    """Show *text* in an error dialog if a window can be opened (the GUI
    started without a terminal has no other way to say it)."""
    root = _open_dialog_root()
    if root is None:
        return
    try:
        from tkinter import messagebox

        messagebox.showerror(title, text, parent=root)
    finally:
        root.destroy()


def compat_install_dialog(message: str, steps: list[list[str]], run=subprocess.run) -> str:
    """Ask in a small tkinter dialog whether to install the compatible
    polars build and, on yes, run *steps* with a progress window.

    Used by the GUI when it starts without a terminal (a shortcut,
    ``pythonw.exe``), before polars is imported: tkinter is stdlib and
    imports no polars.  Returns ``"installed"``, ``"declined"``,
    ``"failed"`` or ``"unavailable"`` (no window could be opened)."""
    root = _open_dialog_root()
    if root is None:
        return "unavailable"
    import threading
    from tkinter import messagebox, ttk

    try:
        if not messagebox.askyesno(_DIALOG_TITLE, f"{message}\n\nRun it now?", icon="warning", parent=root):
            return "declined"
        outcome: dict = {}

        def _run_quietly(argv, **kwargs):
            done = run(argv, capture_output=True, text=True, **kwargs, **_no_console_window())
            outcome["output"] = (getattr(done, "stdout", "") or "") + (getattr(done, "stderr", "") or "")
            return done

        def _worker() -> None:
            outcome["returncode"] = _run_steps(steps, _run_quietly)

        root.title(_DIALOG_TITLE)
        ttk.Label(root, text="Installing the compatible polars build (polars[rtcompat])...", padding=12).pack()
        bar = ttk.Progressbar(root, mode="indeterminate", length=320)
        bar.pack(padx=12, pady=(0, 12))
        bar.start(15)
        root.deiconify()
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        def _poll() -> None:
            if thread.is_alive():
                root.after(100, _poll)
            else:
                root.quit()

        root.after(100, _poll)
        root.mainloop()
        thread.join()
        bar.stop()
        root.withdraw()
        returncode = outcome.get("returncode", "not run")
        if returncode != 0:
            tail = "\n".join(outcome.get("output", "").strip().splitlines()[-8:])
            messagebox.showerror(
                _DIALOG_TITLE,
                f"FlexTool could not install the compatible polars build (exit {returncode}).\n\n{tail}\n\n"
                f"{message}",
                parent=root,
            )
            return "failed"
        messagebox.showinfo(
            _DIALOG_TITLE, "The compatible polars build is installed. FlexTool starts now.", parent=root
        )
        return "installed"
    finally:
        root.destroy()


def ensure_polars_runtime_before_import(
    run=subprocess.run, ask=input, interactive=None, entry=None, dialog=None
) -> str:
    """Offer the compatible polars runtime before polars is first imported,
    on a CPU that lacks the default runtime's features.

    Called at the top of ``flextool/__init__.py``.  The default runtime can
    crash natively on such a CPU already while flextool's modules import, so
    the CPU flags are read here (stdlib only).  When the compatible runtime
    is needed and missing, what happens depends on :func:`import_entry_mode`:

    * ask (scripts, the CLI, Toolbox tools, ``python -c``): with a terminal,
      print the reason and the command and ask "Run it now? [y/N]"; only an
      explicit yes runs pip.  Without a terminal (*interactive* false: a GUI
      subprocess, a Toolbox tool, CI, pytest), or on no, or when pip fails,
      raise :class:`CompatRuntimeRequired` with the reason and the command.
      pip never runs without a yes.
    * gui (the ``flextool`` / ``flextool-gui`` launchers, ``python -m
      flextool.gui``): with a terminal, the same question; without one (a
      shortcut, ``pythonw.exe``), the reason and the command in a tkinter
      yes/no dialog (*dialog*, default :func:`compat_install_dialog`), with
      a progress window while pip runs.  On yes and a successful install the
      GUI starts (polars is not imported yet, so it loads the compatible
      runtime); on no, or when pip fails, the process exits
      (``SystemExit``) with the message.  When no window can be opened the
      reason is noted on stderr and the GUI's own startup check offers the
      install (``"deferred"``).
    * defer (``python -m flextool.env_check``): note the reason on stderr
      and return ``"diagnostics"``, on which ``flextool/__init__.py`` skips
      the rest of the package (it imports polars, whose default runtime
      would crash this process before the diagnostics print); the
      diagnostics command probes polars in a child process and prints the
      fix.  The same on a CPU below x86-64-v2 (its own message, no note).
    * repair (``flextool-update``): the user asked for a repair, so install
      without asking (``"failed"`` is reported and the update continues).

    On a CPU that runs the default runtime, the default runtime is selected
    with ``POLARS_PREFER_PKG`` when the compatible one is installed next to
    it (:func:`prefer_default_runtime`).

    Returns ``"skipped"``, ``"not_needed"``, ``"unsupported"``,
    ``"diagnostics"``, ``"deferred"``, ``"installed"`` or ``"failed"``.  *run*, *ask*,
    *interactive*, *entry* and *dialog* are parameters for tests.  Skipped
    when polars is already imported (a Toolbox console, a notebook), when
    ``POLARS_FORCE_PKG`` picks a runtime, or when ``FLEXTOOL_NO_ENV_CHECK``
    is set.
    """
    if (
        "polars" in sys.modules
        or os.environ.get("FLEXTOOL_NO_ENV_CHECK")
        or os.environ.get("POLARS_FORCE_PKG")
    ):
        return "skipped"
    verdict = cpu_verdict()
    if verdict.status == CPU_FULL:
        prefer_default_runtime(verdict)
        return "not_needed"
    if compat_runtime_ready():
        return "not_needed"
    entry = entry or import_entry_mode()
    terminal = _has_terminal() if interactive is None else interactive
    if verdict.status == CPU_UNSUPPORTED:
        if entry == ENTRY_DEFER:
            return "diagnostics"
        _stderr(unsupported_cpu_help(verdict))
        if entry == ENTRY_GUI and not terminal:
            _dialog_notice("FlexTool cannot run on this computer", unsupported_cpu_help(verdict))
        return "unsupported"
    steps = polars_compat_steps()
    if verdict.status != CPU_NEEDS_COMPAT or steps is None:
        return "not_needed"
    message = compat_runtime_message(verdict, steps)
    if entry == ENTRY_DEFER:
        _stderr(message)
        return "diagnostics"
    if entry == ENTRY_REPAIR:
        _stderr(proactive_banner(verdict))
        returncode = _run_steps(steps, run)
        if returncode != 0:
            _stderr(f"FlexTool could not install the compatible polars build (exit {returncode}). {message}")
            return "failed"
        return "installed"

    opt_out = (
        "Then start FlexTool again. (To skip this check, set FLEXTOOL_NO_ENV_CHECK=1; polars will "
        "then probably crash on this processor.)"
    )
    # The GUI is an application, not a library import: it exits with the
    # message instead of an ImportError traceback.
    stop = SystemExit if entry == ENTRY_GUI else CompatRuntimeRequired
    if entry == ENTRY_GUI and not terminal:
        result = (dialog or compat_install_dialog)(message, steps, run)
        if result == "installed":
            return "installed"
        if result == "unavailable":
            _stderr(message)
            return "deferred"
        if result == "failed":
            raise SystemExit(f"FlexTool could not install the compatible polars build. {message}\n\n{opt_out}")
        raise SystemExit(f"Not installed. {message}\n\n{opt_out}")
    if not terminal:
        raise CompatRuntimeRequired(f"{message}\n\n{opt_out}")
    print(message, flush=True)
    try:
        answer = ask("Run it now? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip().lower() not in ("y", "yes"):
        raise stop(f"Not installed. {message}\n\n{opt_out}")
    returncode = _run_steps(steps, run)
    if returncode != 0:
        raise stop(
            f"FlexTool could not install the compatible polars build (exit {returncode}). {message}\n\n{opt_out}"
        )
    return "installed"


def _cpu_hint(verdict: CpuVerdict | None = None) -> str:
    """CPU description for the diagnostics block (stdlib only)."""
    proc = platform.processor() or "unknown"
    verdict = verdict or cpu_verdict()
    return f"{proc}; {verdict.describe()}"


def _runtimes_text() -> str:
    runtimes = polars_runtimes()
    return ", ".join(f"{rt} {ver}" for rt, ver in runtimes.items()) or "none"


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
        f"  polars build: {pol_dist} {pol_ver} (runtimes: {_runtimes_text()}; loaded: {probe.polars_runtime or '-'})",
        f"  highspy     : {_dist_version('highspy')}",
        f"  {probe.summary()}",
    ]
    if probe.stderr and not probe.ok:
        lines.append("  --- probe stderr ---")
        lines.extend("  " + ln for ln in probe.stderr.splitlines())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """``python -m flextool.env_check`` — print diagnostics, exit non-zero
    if the solver stack cannot run here (or will crash once polars runs a
    vectorised operation: a CPU without the default runtime's features and
    no compatible runtime installed)."""
    verdict = cpu_verdict()
    if verdict.status == CPU_UNSUPPORTED:
        # The probe would only crash; say why instead.
        print(diagnostics_report(StackProbe(ERROR, None, stderr="not run: unsupported CPU")))
        print("\n" + unsupported_cpu_help(verdict))
        return 1
    probe = probe_solver_stack()
    print(diagnostics_report(probe))
    if proactive_polars_fix_needed():
        print("\nThis processor needs the compatible polars build. Fix:")
        for step in polars_compat_steps():
            print("  " + format_command(step))
        return 1
    if probe.is_native_fault and has_remediation(probe.failed_component):
        print("\nFix:")
        for step in remediation_steps(probe.failed_component):
            print("  " + format_command(step))
        return 1
    if probe.is_native_fault:
        print("\n" + unfixable_help())
        return 1
    if not probe.ok and "DLL load failed" in probe.stderr:
        print("\n" + DLL_LOAD_HELP)
    return 0 if probe.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
