"""Unit tests for flextool.env_check — the native solver-stack self-check.

Stdlib-only and fast: the native-fault classifier and the failed-component
classifier are exercised directly, and the live probe runs against the
current interpreter (which, in any environment able to run the test suite,
has a working solver stack).
"""

import enum
import json
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from flextool import env_check


# ── native-fault classification ──────────────────────────────────

@pytest.mark.parametrize(
    "rc",
    [
        3221225477,  # 0xC0000005 access violation — the reported crash
        3221225501,  # 0xC000001D illegal instruction (no-AVX2 CPU)
        3221226505,  # 0xC0000409 stack buffer overrun
        0xC00000FD,  # stack overflow
    ],
)
def test_windows_native_faults(rc):
    assert env_check.is_native_fault(rc, plat="win32") is True


@pytest.mark.parametrize("rc", [0, 1, 2, 3, 9009, 0xC000013A])  # incl. Ctrl-C (STATUS_CONTROL_C_EXIT)
def test_windows_normal_exits_are_not_faults(rc):
    assert env_check.is_native_fault(rc, plat="win32") is False


def _host_signal(name):
    """``-signum`` of signal *name* on this computer (the numbers differ between systems); skip where it has none."""
    signum = getattr(signal, name, None)
    if signum is None:
        pytest.skip(f"this computer has no {name}")
    return -int(signum)


@pytest.mark.parametrize("name", ["SIGSEGV", "SIGILL", "SIGBUS", "SIGABRT", "SIGFPE"])
def test_posix_native_faults(name):
    assert env_check.is_native_fault(_host_signal(name), plat="linux") is True


@pytest.mark.parametrize("rc", [0, 1, 2])
def test_posix_normal_exits_are_not_faults(rc):
    assert env_check.is_native_fault(rc, plat="linux") is False


@pytest.mark.parametrize("name", ["SIGTERM", "SIGINT", "SIGKILL", "SIGUSR1"])
def test_posix_stop_signals_are_not_faults(name):
    assert env_check.is_native_fault(_host_signal(name), plat="linux") is False


# Darwin's and Linux's numbers for the signals where they differ (and the ones around them).
_MACOS_SIGNALS = enum.IntEnum("Signals", {
    "SIGILL": 4, "SIGABRT": 6, "SIGEMT": 7, "SIGFPE": 8, "SIGKILL": 9, "SIGBUS": 10, "SIGSEGV": 11, "SIGUSR1": 30,
})
_LINUX_SIGNALS = enum.IntEnum("Signals", {
    "SIGILL": 4, "SIGABRT": 6, "SIGBUS": 7, "SIGFPE": 8, "SIGKILL": 9, "SIGUSR1": 10, "SIGSEGV": 11,
})


@pytest.mark.parametrize(
    ("table", "plat", "rc", "fault", "described"),
    [
        (_MACOS_SIGNALS, "darwin", -10, True, "SIGBUS (bus error)"),
        (_MACOS_SIGNALS, "darwin", -7, False, "SIGEMT"),
        (_MACOS_SIGNALS, "darwin", -11, True, "SIGSEGV (segmentation fault)"),
        (_MACOS_SIGNALS, "darwin", -30, False, "SIGUSR1"),
        (_LINUX_SIGNALS, "linux", -7, True, "SIGBUS (bus error)"),
        (_LINUX_SIGNALS, "linux", -10, False, "SIGUSR1"),
        (_LINUX_SIGNALS, "linux", -30, False, "signal 30"),
    ],
)
def test_posix_signals_are_named_in_the_running_systems_numbering(monkeypatch, table, plat, rc, fault, described):
    """Item 215: on macOS 10 is SIGBUS (Linux: SIGUSR1), so a macOS SIGBUS is a native fault, not a kill."""
    monkeypatch.setattr(signal, "Signals", table)
    assert env_check.is_native_fault(rc, plat=plat) is fault
    assert env_check.describe_fault(rc, plat=plat) == described
    assert env_check.signal_name(-rc) == (None if described.startswith("signal ") else described.split(" (")[0])


def test_none_and_zero_never_fault():
    assert env_check.is_native_fault(None) is False
    assert env_check.is_native_fault(0) is False


def test_describe_fault_names_known_codes():
    assert "access violation" in env_check.describe_fault(3221225477, plat="win32")
    assert "illegal instruction" in env_check.describe_fault(3221225501, plat="win32")
    assert "SIGSEGV" in env_check.describe_fault(-11, plat="linux")


# ── failed-component classification ───────────────────────────────

def test_classify_component_from_markers():
    f = env_check._classify_failed_component
    # Crashed after the last printed checkpoint -> the next component.
    assert f("PROBE:begin\n") == "polars"
    assert f("PROBE:begin\nPROBE:polars\n") == "highspy"
    assert f("PROBE:begin\nPROBE:polars\nPROBE:highspy\n") == "polar_high"
    # Reached the end -> nothing to blame.
    assert f("PROBE:begin\nPROBE:polars\nPROBE:highspy\nPROBE:polar_high\n") is None
    # No output at all -> unknown.
    assert f("") is None


def test_stack_probe_summary_names_component():
    probe = env_check.StackProbe(
        env_check.NATIVE_FAULT, 3221225477, failed_component="highspy"
    )
    s = probe.summary()
    assert "highspy" in s
    assert "NATIVE CRASH" in s
    assert probe.is_native_fault is True


# ── live probe against the current interpreter ────────────────────

def test_probe_solver_stack_ok_here():
    """The interpreter running the tests has a working solver stack."""
    result = env_check.probe_solver_stack()
    assert result.ok, result.summary() + "\n" + result.stderr
    assert result.is_native_fault is False
    assert result.failed_component is None
    assert result.summary() == "solver stack: OK"


# ── remediation steps + dispatch ──────────────────────────────────

def _fake_dists(monkeypatch, **versions):
    """Make env_check see exactly the given distributions (name -> version,
    with '_' for '-'), without touching the real environment."""
    table = {name.replace("_", "-"): ver for name, ver in versions.items()}
    monkeypatch.setattr(env_check, "_dist_version", lambda name: table.get(name))


def _fake_cpu(monkeypatch, status, missing_default=(), missing_compat=()):
    verdict = env_check.CpuVerdict(status, "test", list(missing_default), list(missing_compat))
    monkeypatch.setattr(env_check, "cpu_verdict", lambda: verdict)
    return verdict


def test_polars_heal_adds_compat_runtime_pinned_to_installed_version(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1")
    assert env_check.polars_compat_steps(python="/some/python") == [
        ["/some/python", "-P", "-m", "pip", "install", "polars[rtcompat]==1.40.1"],
    ]


def test_polars_heal_follows_an_upgraded_polars(monkeypatch):
    # A compat runtime of an older version is skipped by polars' loader.
    _fake_dists(monkeypatch, polars="2.1.0", polars_runtime_32="2.1.0", polars_runtime_compat="1.40.1")
    assert env_check.compat_runtime_ready() is False
    assert env_check.polars_compat_steps(python="/p") == [
        ["/p", "-P", "-m", "pip", "install", "polars[rtcompat]==2.1.0"],
    ]


def test_polars_heal_not_needed_when_compat_runtime_matches(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1", polars_runtime_compat="1.40.1")
    assert env_check.compat_runtime_ready() is True
    assert env_check.polars_compat_steps(python="/p") is None


def _requirement_parts(spec):
    return sorted(part.strip() for part in spec.split(","))


def test_polars_heal_replaces_retired_lts_cpu_within_flextool_floor(monkeypatch):
    # polars-lts-cpu stops at 1.33.1, below flextool's polars floor: remove
    # it (and a standard polars it overwrote) and install polars[rtcompat]
    # within flextool's own polars requirement.
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    core = [d for d in pyproject["project"]["dependencies"] if d.startswith("polars")]
    assert len(core) == 1
    _fake_dists(monkeypatch, polars_lts_cpu="1.33.1", polars_runtime_32="1.40.1")
    uninstall, install = env_check.polars_compat_steps(python="/p")
    assert uninstall == ["/p", "-P", "-m", "pip", "uninstall", "-y", "polars-lts-cpu", "polars"]
    assert install[:5] == ["/p", "-P", "-m", "pip", "install"]
    assert install[5].startswith("polars[rtcompat]")
    assert _requirement_parts(install[5][len("polars[rtcompat]"):]) == _requirement_parts(core[0][len("polars"):])


def test_compat_extra_matches_core_polars_requirement():
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    core = [d for d in pyproject["project"]["dependencies"] if d.startswith("polars")][0]
    (compat,) = pyproject["project"]["optional-dependencies"]["compat"]
    assert compat.startswith("polars[rtcompat]")
    assert _requirement_parts(compat[len("polars[rtcompat]"):]) == _requirement_parts(core[len("polars"):])


def _flextool_highspy_spec():
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    (req,) = [r for r in pyproject["project"]["dependencies"] if r.startswith("highspy")]
    return req[len("highspy"):].replace(" ", "")


@pytest.mark.parametrize("old", ["1.13.1", "1.14.0"])
def test_highspy_older_than_its_own_cpp_runtime_is_upgraded_within_flextool_requirement(monkeypatch, old):
    # highspy <= 1.14 imports the system msvcp140.dll (1.14.0 crashes on
    # Windows 10 1809 / Server 2019, HiGHS#2964; 1.13.1 was the old heal's pin).
    _fake_dists(monkeypatch, highspy=old)
    monkeypatch.setattr(env_check, "_flextool_requirement_spec", lambda name, fallback: _flextool_highspy_spec())
    (step,) = env_check.highspy_upgrade_steps(python="/some/python")
    assert step == ["/some/python", "-P", "-m", "pip", "install", "highspy" + _flextool_highspy_spec()]


@pytest.mark.parametrize("current", ["1.15.0", "1.15.1", "1.16.0", None])
def test_no_highspy_remedy_from_the_release_with_its_own_cpp_runtime(monkeypatch, current):
    _fake_dists(monkeypatch, highspy=current)
    assert env_check.highspy_upgrade_steps() is None
    assert env_check.has_remediation("highspy") is False


def test_flextool_requires_a_highspy_with_its_own_cpp_runtime():
    # The requirement's floor must exclude the releases the remedy replaces,
    # or "Update FlexTool" would leave them in place.
    spec = _flextool_highspy_spec()
    floor = next(part[2:] for part in spec.split(",") if part.startswith(">="))
    assert env_check._release(floor) >= env_check.HIGHSPY_OWN_CPP_RUNTIME
    # The source-checkout fallback must track the pyproject requirement.  The
    # weekly canary lifts the upper caps in pyproject.toml before running the
    # suite (.github/scripts/dependency_caps.py lift), so the lower bounds are
    # compared always and the full spec only while pyproject is still capped.
    fallback = env_check._HIGHSPY_SPEC_FALLBACK
    assert _lower_bounds(fallback) == _lower_bounds(spec)
    if any(part.startswith(("<", "==", "~=")) for part in spec.split(",")):
        assert fallback == spec


def _lower_bounds(spec: str) -> list[str]:
    return sorted(part for part in spec.split(",") if part.startswith(">"))


def test_default_python_is_current_interpreter(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1", highspy="1.14.0")
    assert env_check.polars_compat_steps()[0][0] == sys.executable
    assert env_check.highspy_upgrade_steps()[0][0] == sys.executable


def test_remediation_dispatch_by_component(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1", highspy="1.14.0")
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    assert env_check.has_remediation("polars") is True
    assert env_check.has_remediation("highspy") is True
    assert env_check.has_remediation("polar_high") is False
    assert env_check.has_remediation(None) is False

    assert env_check.remediation_steps("polars", "/p")[-1][-1] == "polars[rtcompat]==1.40.1"
    assert "highspy" in env_check.remediation_steps("highspy", "/p")[0][-1]
    assert env_check.remediation_steps("polar_high") is None

    assert "polars" in env_check.remediation_banner("polars")
    assert "highspy" in env_check.remediation_banner("highspy")
    assert env_check.remediation_banner("polar_high") is None


def test_no_polars_remedy_once_compat_runtime_is_in_place(monkeypatch):
    # A polars crash with the compat runtime already loaded is not a CPU
    # feature problem a package install can fix.
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1", polars_runtime_compat="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    assert env_check.has_remediation("polars") is False
    assert env_check.unfixable_help() == env_check.UNFIXABLE_HELP


def test_no_polars_remedy_below_x86_64_v2(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_UNSUPPORTED, ["avx2"], ["sse4.2", "popcnt"])
    assert env_check.has_remediation("polars") is False
    help_text = env_check.unfixable_help()
    assert "sse4.2, popcnt" in help_text
    assert "x86-64-v2" in help_text


@pytest.mark.parametrize(
    ("status", "dists", "expected"),
    [
        (env_check.CPU_NEEDS_COMPAT, {"polars": "1.40.1"}, True),
        (env_check.CPU_NEEDS_COMPAT, {"polars": "1.40.1", "polars_runtime_compat": "1.40.1"}, False),
        (env_check.CPU_FULL, {"polars": "1.40.1"}, False),
        (env_check.CPU_UNKNOWN, {"polars": "1.40.1"}, False),
        (env_check.CPU_NOT_X86, {"polars": "1.40.1"}, False),
        (env_check.CPU_UNSUPPORTED, {"polars": "1.40.1"}, False),
    ],
)
def test_proactive_polars_fix_decision(monkeypatch, status, dists, expected):
    _fake_dists(monkeypatch, **dists)
    _fake_cpu(monkeypatch, status, ["avx2"])
    assert env_check.proactive_polars_fix_needed() is expected


# ── CPU feature detection ─────────────────────────────────────────

# /proc/cpuinfo "flags" lines, abridged to the relevant flags plus a few
# neighbours, as the kernel prints them for these processors.
_CPUINFO_HASWELL = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 "
    "ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc pni pclmulqdq dtes64 monitor ssse3 fma cx16 "
    "sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm abm fsgsbase bmi1 avx2 smep "
    "bmi2 erms invpcid"
)
# Celeron N4020 (Gemini Lake): SSE4.2/POPCNT/MOVBE, no AVX at all.
_CPUINFO_GEMINI_LAKE = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 "
    "ss ht syscall nx pdpe1gb rdtscp lm constant_tsc pni pclmulqdq monitor ssse3 cx16 sse4_1 sse4_2 "
    "x2apic movbe popcnt aes rdrand lahf_lm 3dnowprefetch fsgsbase smep erms sha_ni"
)
# Core i5-2500 (Sandy Bridge): AVX but no AVX2/FMA/BMI/MOVBE.
_CPUINFO_SANDY_BRIDGE = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 "
    "ss ht syscall nx rdtscp lm constant_tsc pni pclmulqdq ssse3 cx16 sse4_1 sse4_2 x2apic popcnt aes "
    "xsave avx lahf_lm"
)
# Core 2 Duo E8400 (Penryn): SSE4.1 but no SSE4.2/POPCNT.
_CPUINFO_PENRYN = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 "
    "ss ht syscall nx lm constant_tsc pni dtes64 monitor ssse3 cx16 sse4_1 xsave lahf_lm"
)
# Phenom II X4 (K10): POPCNT/LZCNT (abm) and SSE4a, but no SSSE3/SSE4.1/SSE4.2.
_CPUINFO_PHENOM_II = (
    "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 "
    "ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm 3dnowext 3dnow constant_tsc pni monitor cx16 popcnt "
    "lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch"
)


def _cpuinfo(flags):
    return (
        "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: test cpu\n"
        f"flags\t\t: {flags}\nbugs\t\t: spectre_v1\n\nprocessor\t: 1\nflags\t\t: {flags}\n"
    )


@pytest.mark.parametrize(
    ("flags", "status", "missing_default", "missing_compat"),
    [
        (_CPUINFO_HASWELL, env_check.CPU_FULL, [], []),
        (_CPUINFO_GEMINI_LAKE, env_check.CPU_NEEDS_COMPAT, ["avx", "avx2", "fma", "bmi1", "bmi2", "lzcnt"], []),
        (_CPUINFO_SANDY_BRIDGE, env_check.CPU_NEEDS_COMPAT, ["avx2", "fma", "bmi1", "bmi2", "lzcnt", "movbe"], []),
        (_CPUINFO_PENRYN, env_check.CPU_UNSUPPORTED, None, ["sse4.2", "popcnt"]),
        (_CPUINFO_PHENOM_II, env_check.CPU_UNSUPPORTED, None, ["ssse3", "sse4.1", "sse4.2"]),
    ],
)
def test_classify_cpu_from_proc_cpuinfo(flags, status, missing_default, missing_compat):
    features = env_check.parse_proc_cpuinfo(_cpuinfo(flags))
    assert features.source == "/proc/cpuinfo"
    verdict = env_check.classify_cpu(features)
    assert verdict.status == status
    assert verdict.missing_compat == missing_compat
    if missing_default is not None:
        assert verdict.missing_default == missing_default


def test_proc_cpuinfo_without_x86_flags_is_not_parsed():
    arm = "processor\t: 0\nBogoMIPS\t: 48.00\nFeatures\t: fp asimd evtstrm aes pmull sha1 sha2 crc32\n"
    assert env_check.parse_proc_cpuinfo(arm) is None
    assert env_check.classify_cpu(None).status == env_check.CPU_UNKNOWN
    assert env_check.classify_cpu(None, is_x86=False).status == env_check.CPU_NOT_X86


def _cpuid_from(flags_line):
    """cpuid flags as polars' checker reports them, for a CPU whose kernel
    flags are *flags_line*."""
    return env_check.CpuFeatures("cpuid", dict(env_check.parse_proc_cpuinfo(_cpuinfo(flags_line)).flags))


def test_feature_missing_in_any_source_is_missing():
    # qemu-x86_64 -cpu SandyBridge on a Haswell host: /proc/cpuinfo is the
    # host's, cpuid (and the instructions that run) the emulated CPU's.
    host = env_check.parse_proc_cpuinfo(_cpuinfo(_CPUINFO_HASWELL))
    merged = env_check.merge_cpu_features(host, _cpuid_from(_CPUINFO_SANDY_BRIDGE))
    assert merged.source == "/proc/cpuinfo+cpuid"
    verdict = env_check.classify_cpu(merged)
    assert verdict.status == env_check.CPU_NEEDS_COMPAT
    assert verdict.missing_default == ["avx2", "fma", "bmi1", "bmi2", "lzcnt", "movbe"]
    # The other way round (a kernel that disabled AVX on an AVX2 CPU) too.
    reverse = env_check.merge_cpu_features(
        env_check.parse_proc_cpuinfo(_cpuinfo(_CPUINFO_SANDY_BRIDGE)), _cpuid_from(_CPUINFO_HASWELL)
    )
    assert env_check.classify_cpu(reverse).status == env_check.CPU_NEEDS_COMPAT
    # A feature only one source reports keeps that source's answer.
    partial = env_check.merge_cpu_features(
        env_check.CpuFeatures("a", {"avx2": True}), env_check.CpuFeatures("b", {"sse3": False})
    )
    assert partial.flags == {"avx2": True, "sse3": False}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/cpuinfo")
def test_linux_detection_sees_the_emulated_cpu(monkeypatch):
    import io

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/cpuinfo":
            return io.StringIO(_cpuinfo(_CPUINFO_HASWELL))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(env_check, "open", fake_open, raising=False)
    monkeypatch.setattr(env_check.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env_check, "_polars_cpuid_flags", lambda: _cpuid_from(_CPUINFO_GEMINI_LAKE))
    verdict = env_check.classify_cpu(env_check.detect_cpu_features())
    assert verdict.status == env_check.CPU_NEEDS_COMPAT
    assert "avx2" in verdict.missing_default


def test_linux_flag_mapping_covers_every_runtime_feature():
    assert set(env_check._LINUX_FLAGS.values()) == set(env_check.DEFAULT_RUNTIME_FEATURES)
    assert set(env_check.X86_64_V2_FEATURES) < set(env_check.DEFAULT_RUNTIME_FEATURES)


def _windows_api(present_ids):
    return lambda pf: pf in present_ids


def test_windows_api_jasper_lake_needs_compat():
    # SSE3, CMPXCHG16B, SSSE3, SSE4.1, SSE4.2 present; no AVX/AVX2.
    features = env_check.windows_feature_flags(_windows_api({13, 14, 36, 37, 38}))
    assert features.source == "IsProcessorFeaturePresent"
    verdict = env_check.classify_cpu(features)
    assert verdict.status == env_check.CPU_NEEDS_COMPAT
    assert verdict.missing_default == ["avx", "avx2"]


def test_windows_api_cannot_confirm_full_feature_set():
    # Every id True, yet FMA/BMI2/MOVBE have no id: never "full" from this
    # source alone (no false all-clear, no needless install either).
    features = env_check.windows_feature_flags(_windows_api({13, 14, 36, 37, 38, 39, 40}))
    assert env_check.classify_cpu(features).status == env_check.CPU_UNKNOWN


def test_windows_api_on_os_without_newer_ids_is_unknown_not_unsupported():
    # An OS that predates ids 36-40 returns False for all of them.
    features = env_check.windows_feature_flags(_windows_api({13, 14}))
    assert "sse4.2" not in features.flags
    assert env_check.classify_cpu(features).status == env_check.CPU_UNKNOWN


def test_windows_api_pre_sse42_cpu_is_unsupported():
    # Penryn on a current Windows: SSSE3 and SSE4.1 reported, SSE4.2 not.
    features = env_check.windows_feature_flags(_windows_api({13, 14, 36, 37}))
    verdict = env_check.classify_cpu(features)
    assert verdict.status == env_check.CPU_UNSUPPORTED
    assert verdict.missing_compat == ["sse4.2"]


def test_parse_build_feature_flags():
    text = "+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+cmpxchg16b,-avx512f,+ctr-static"
    assert env_check.parse_build_feature_flags(text) == env_check.X86_64_V2_FEATURES


def test_installed_runtime_feature_list_is_known():
    """Every feature the installed default runtime was built for is one the
    detectors can report (otherwise it would always read as unknown)."""
    required = env_check.runtime_required_features("32") or env_check.runtime_required_features("64")
    if required is None:
        pytest.skip("no default polars runtime installed")
    assert set(required) <= set(env_check.DEFAULT_RUNTIME_FEATURES)


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64")),
    reason="compares cpuid with /proc/cpuinfo on x86 Linux",
)
def test_cpuid_agrees_with_proc_cpuinfo():
    with open("/proc/cpuinfo", encoding="utf-8") as fh:
        from_kernel = env_check.parse_proc_cpuinfo(fh.read())
    from_cpuid = env_check._polars_cpuid_flags()
    assert from_cpuid is not None and from_cpuid.source == "cpuid"
    assert {k: from_cpuid.flags[k] for k in env_check.DEFAULT_RUNTIME_FEATURES} == from_kernel.flags


def test_cpu_detection_does_not_import_polars():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('ec', {env_check.__file__!r})\n"
        "ec = importlib.util.module_from_spec(spec); sys.modules['ec'] = ec; spec.loader.exec_module(ec)\n"
        "print(ec.cpu_verdict().status, ec.proactive_polars_fix_needed())\n"
        "assert 'polars' not in sys.modules and '_polars_runtime_32._polars_runtime' not in sys.modules\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


# ── heal before the first polars import ───────────────────────────

_REAL_RUN = subprocess.run


class _Ran:
    """Stands in for ``subprocess.run``: records pip calls and returns
    *returncode*; anything else (e.g. ``platform``'s ``uname -p``, which
    shares the patched module) runs for real."""

    def __init__(self, returncode=0):
        self.calls = []
        self.returncode = returncode

    def __call__(self, argv, *args, **kwargs):
        if list(argv[1:4]) != ["-P", "-m", "pip"]:
            return _REAL_RUN(argv, *args, **kwargs)
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, self.returncode)


def test_pip_does_not_inherit_the_safe_path_variable(monkeypatch):
    """A FlexTool child carries PYTHONSAFEPATH=1; pip must not (it runs setup scripts that import from their folder)."""
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    envs = []

    def run(argv, **kwargs):
        envs.append(kwargs.get("env"))
        return subprocess.CompletedProcess(argv, 0)

    assert env_check._run_steps([_COMPAT_STEP], run) == 0
    assert envs[0] is not None and "PYTHONSAFEPATH" not in envs[0] and envs[0]["PATH"] == os.environ["PATH"]


def _no_polars_imported(monkeypatch):
    for name in [m for m in sys.modules if m == "polars" or m.startswith("polars.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delenv("FLEXTOOL_NO_ENV_CHECK", raising=False)
    monkeypatch.delenv("POLARS_FORCE_PKG", raising=False)


def _needs_compat(monkeypatch, missing=("avx2", "fma")):
    _no_polars_imported(monkeypatch)
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1")
    return _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, list(missing))


_COMPAT_STEP = [sys.executable, "-P", "-m", "pip", "install", "polars[rtcompat]==1.40.1"]


class _Stream:
    """A stdin/stdout stand-in that is, or is not, a terminal."""

    def __init__(self, tty):
        self.tty = tty
        self.written = []

    def isatty(self):
        return self.tty

    def write(self, text):
        self.written.append(text)

    def flush(self):
        pass


def _answers(*replies):
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return replies[len(asked) - 1]

    return ask, asked


def test_import_guard_without_terminal_never_installs(monkeypatch):
    # A GUI subprocess, Toolbox tool, CI or pytest: stdin/stdout are not a
    # terminal.  The guard must stop with the reason and the command, and
    # must not run pip.
    _needs_compat(monkeypatch)
    monkeypatch.setattr(sys, "stdin", _Stream(tty=False))
    monkeypatch.setattr(sys, "stdout", _Stream(tty=False))
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "flextool.cli.cmd_run_flextool"])
    ran = _Ran()
    with pytest.raises(ImportError) as err:
        env_check.ensure_polars_runtime_before_import(run=ran)
    assert ran.calls == []
    text = str(err.value)
    assert "avx2, fma" in text
    assert env_check.format_command(_COMPAT_STEP) in text
    assert "FLEXTOOL_NO_ENV_CHECK" in text


def test_import_guard_terminal_yes_installs(monkeypatch, capsys):
    _needs_compat(monkeypatch)
    ask, asked = _answers("y")
    ran = _Ran()
    assert env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=True, entry="ask") == "installed"
    assert asked == ["Run it now? [y/N] "]
    assert ran.calls == [_COMPAT_STEP]
    out = capsys.readouterr().out
    assert "avx2, fma" in out and env_check.format_command(_COMPAT_STEP) in out


@pytest.mark.parametrize("reply", ["", "n", "no", "maybe", "Y es"])
def test_import_guard_terminal_without_explicit_yes_installs_nothing(monkeypatch, reply):
    _needs_compat(monkeypatch)
    ask, _asked = _answers(reply)
    ran = _Ran()
    with pytest.raises(env_check.CompatRuntimeRequired, match="Not installed"):
        env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=True, entry="ask")
    assert ran.calls == []


def test_import_guard_terminal_end_of_input_is_no(monkeypatch):
    _needs_compat(monkeypatch)

    def ask(_prompt):
        raise EOFError

    ran = _Ran()
    with pytest.raises(env_check.CompatRuntimeRequired):
        env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=True, entry="ask")
    assert ran.calls == []


def test_import_guard_terminal_yes_but_pip_fails(monkeypatch):
    _needs_compat(monkeypatch)
    ask, _asked = _answers("yes")
    with pytest.raises(env_check.CompatRuntimeRequired, match="exit 1"):
        env_check.ensure_polars_runtime_before_import(run=_Ran(returncode=1), ask=ask, interactive=True, entry="ask")


def test_import_guard_defers_to_diagnostics_without_asking(monkeypatch, capsys):
    _needs_compat(monkeypatch)
    ran = _Ran()
    ask = lambda _p: pytest.fail("asked")  # noqa: E731
    assert env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=True, entry="defer") == "diagnostics"
    assert ran.calls == []
    assert env_check.format_command(_COMPAT_STEP) in capsys.readouterr().err


def test_import_guard_update_repairs_without_asking(monkeypatch, capsys):
    _needs_compat(monkeypatch)
    ran = _Ran()
    ask = lambda _p: pytest.fail("asked")  # noqa: E731
    result = env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=False, entry="repair")
    assert result == "installed"
    assert ran.calls == [_COMPAT_STEP]
    assert "avx2, fma" in capsys.readouterr().err


def test_import_guard_update_reports_failed_install(monkeypatch, capsys):
    _needs_compat(monkeypatch)
    assert env_check.ensure_polars_runtime_before_import(run=_Ran(returncode=1), entry="repair") == "failed"
    assert "polars[rtcompat]==1.40.1" in capsys.readouterr().err


def test_gui_terminal_yes_installs_and_continues(monkeypatch, capsys):
    _needs_compat(monkeypatch)
    ask, asked = _answers("y")
    ran = _Ran()
    dialog = lambda *_a: pytest.fail("dialog opened in a terminal")  # noqa: E731
    result = env_check.ensure_polars_runtime_before_import(
        run=ran, ask=ask, interactive=True, entry="gui", dialog=dialog
    )
    assert result == "installed"
    assert asked == ["Run it now? [y/N] "]
    assert ran.calls == [_COMPAT_STEP]
    assert env_check.format_command(_COMPAT_STEP) in capsys.readouterr().out


@pytest.mark.parametrize("reply", ["", "n"])
def test_gui_terminal_no_exits_with_the_message(monkeypatch, reply):
    # The GUI is an application: it exits with the message, not an ImportError traceback.
    _needs_compat(monkeypatch)
    ask, _asked = _answers(reply)
    ran = _Ran()
    with pytest.raises(SystemExit) as err:
        env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=True, entry="gui")
    assert not isinstance(err.value, ImportError)
    assert ran.calls == []
    assert "Not installed" in str(err.value.code)
    assert env_check.format_command(_COMPAT_STEP) in str(err.value.code)


def test_gui_without_terminal_asks_in_a_dialog(monkeypatch):
    _needs_compat(monkeypatch)
    shown = []

    def dialog(message, steps, run):
        shown.append((message, steps))
        return "installed"

    ask = lambda _p: pytest.fail("asked in a terminal")  # noqa: E731
    result = env_check.ensure_polars_runtime_before_import(
        run=_Ran(), ask=ask, interactive=False, entry="gui", dialog=dialog
    )
    assert result == "installed"
    (message, steps), = shown
    assert steps == [_COMPAT_STEP]
    assert "avx2, fma" in message
    assert env_check.format_command(_COMPAT_STEP) in message


@pytest.mark.parametrize(("outcome", "text"), [("declined", "Not installed"), ("failed", "could not install")])
def test_gui_dialog_no_or_failure_exits(monkeypatch, outcome, text):
    _needs_compat(monkeypatch)
    with pytest.raises(SystemExit, match=text):
        env_check.ensure_polars_runtime_before_import(
            run=_Ran(), interactive=False, entry="gui", dialog=lambda *_a: outcome
        )


def test_gui_without_any_window_falls_back_to_the_startup_check(monkeypatch, capsys):
    # No terminal and no display: note it and let the GUI's own startup check offer the install.
    _needs_compat(monkeypatch)
    result = env_check.ensure_polars_runtime_before_import(
        run=_Ran(), interactive=False, entry="gui", dialog=lambda *_a: "unavailable"
    )
    assert result == "deferred"
    assert env_check.format_command(_COMPAT_STEP) in capsys.readouterr().err


# The GUI launcher, in a fresh interpreter: /proc/cpuinfo reads as an
# x86-64-v2 CPU without AVX2, pip is recorded instead of run, and the
# tkinter dialog / terminal answer are stubbed.  The question must be asked
# while no polars module is loaded.
_GUI_LAUNCH_DRIVER = r"""
import builtins, io, json, subprocess, sys
import tkinter
from tkinter import messagebox

mode, reply = sys.argv[1], sys.argv[2] == "yes"
record = {"pip": []}
_open = builtins.open
_FLAGS = "flags\t\t: fpu sse sse2 pni ssse3 sse4_1 sse4_2 popcnt cx16 pclmulqdq\n"

def fake_open(file, *args, **kwargs):
    if file == "/proc/cpuinfo":
        return io.StringIO(_FLAGS)
    return _open(file, *args, **kwargs)

builtins.open = fake_open
_run = subprocess.run

def fake_run(argv, *args, **kwargs):
    if list(argv[1:4]) == ["-P", "-m", "pip"]:
        record["pip"].append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")
    return _run(argv, *args, **kwargs)

subprocess.run = fake_run

def loaded():
    return sorted(m for m in sys.modules if m == "polars" or m.startswith(("polars.", "_polars")))

def answer(text):
    record["asked"] = text
    record["loaded_when_asked"] = loaded()
    return reply

class FakeTk:
    def withdraw(self): pass
    def destroy(self): pass

tkinter.Tk = FakeTk
messagebox.askyesno = lambda title, text, **kw: answer(text)

class Tty:
    def __init__(self, stream): self.stream = stream
    def isatty(self): return True
    def write(self, text): return self.stream.write(text)
    def flush(self): self.stream.flush()
    def readline(self): return ""

if mode == "tty":
    sys.stdin, sys.stdout = Tty(sys.stdin), Tty(sys.__stderr__)
    builtins.input = lambda prompt: answer(prompt) and "y" or "n"
sys.argv = ["/venv/bin/flextool-gui"]
try:
    import flextool.gui.__main__  # noqa: F401
    record["exit"] = None
except SystemExit as exc:
    record["exit"] = str(exc.code)
record["loaded_after"] = loaded()
sys.__stdout__.write("RESULT:" + json.dumps(record) + "\n")
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the CPU is faked through /proc/cpuinfo")
@pytest.mark.skipif(env_check.compat_runtime_ready(), reason="the compatible runtime is installed here")
@pytest.mark.parametrize(("mode", "reply"), [("dialog", "no"), ("tty", "no"), ("tty", "yes")])
def test_gui_launch_asks_before_polars_is_imported(tmp_path, mode, reply):
    driver = tmp_path / "launch_gui.py"
    driver.write_text(_GUI_LAUNCH_DRIVER, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("FLEXTOOL_NO_ENV_CHECK", "POLARS_FORCE_PKG")}
    done = subprocess.run(
        [sys.executable, str(driver), mode, reply],
        capture_output=True, text=True, timeout=240, env=env, stdin=subprocess.DEVNULL,
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT:")]
    assert lines, done.stdout + done.stderr
    record = json.loads(lines[-1].removeprefix("RESULT:"))
    assert "asked" in record, done.stderr
    assert record["loaded_when_asked"] == []
    if reply == "no":
        assert record["pip"] == []
        assert "Not installed" in record["exit"]
        assert record["loaded_after"] == []
    else:
        assert record["pip"] == [[sys.executable, "-P", "-m", "pip", "install", f"polars[rtcompat]=={env_check._dist_version('polars')}"]]
        assert record["exit"] is None
        assert "polars" in record["loaded_after"]  # the GUI went on importing after the install


# ``python -m flextool.env_check`` in a fresh interpreter on a CPU without
# AVX2 (faked /proc/cpuinfo) and no compatible runtime: importing the package
# must not load polars, whose default runtime would crash the process before
# the diagnostics print.
_DIAGNOSTICS_IMPORT_DRIVER = r"""
import builtins, io, json, sys
_open = builtins.open
_FLAGS = "flags\t\t: fpu sse sse2 pni ssse3 sse4_1 sse4_2 popcnt cx16 pclmulqdq\n"

def fake_open(file, *args, **kwargs):
    if file == "/proc/cpuinfo":
        return io.StringIO(_FLAGS)
    return _open(file, *args, **kwargs)

builtins.open = fake_open
sys.orig_argv = [sys.executable, "-m", "flextool.env_check"]
import flextool
loaded = sorted(m for m in sys.modules if m == "polars" or m.startswith(("polars.", "_polars")))
sys.__stdout__.write("RESULT:" + json.dumps(loaded) + "\n")
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the CPU is faked through /proc/cpuinfo")
@pytest.mark.skipif(env_check.compat_runtime_ready(), reason="the compatible runtime is installed here")
def test_diagnostics_import_does_not_load_polars(tmp_path):
    driver = tmp_path / "diagnostics_import.py"
    driver.write_text(_DIAGNOSTICS_IMPORT_DRIVER, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("FLEXTOOL_NO_ENV_CHECK", "POLARS_FORCE_PKG")}
    done = subprocess.run(
        [sys.executable, str(driver)], capture_output=True, text=True, timeout=240, env=env, stdin=subprocess.DEVNULL
    )
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT:")]
    assert lines, done.stdout + done.stderr
    assert json.loads(lines[-1].removeprefix("RESULT:")) == []
    assert "polars[rtcompat]" in done.stderr  # the note with the fix


@pytest.mark.parametrize(
    ("orig_argv", "argv0", "expected"),
    [
        (["python", "-m", "flextool.gui"], "-m", "gui"),
        (["python", "-X", "dev", "-m", "flextool.gui.__main__"], "-m", "gui"),
        (["pythonw.exe", "C:\\venv\\Scripts\\flextool.exe"], "C:\\venv\\Scripts\\flextool.exe", "gui"),
        (["python", "/venv/bin/flextool-gui"], "/venv/bin/flextool-gui", "gui"),
        (["python", "-m", "flextool.env_check"], "-m", "defer"),
        (["python", "-m", "flextool.cli.cmd_update_flextool"], "-m", "repair"),
        (["python", "update_flextool.py", "--skip-git"], "update_flextool.py", "repair"),
        (["python", "/venv/bin/flextool-update"], "/venv/bin/flextool-update", "repair"),
        (["python", "-m", "flextool.cli.cmd_run_flextool"], "-m", "ask"),
        (["python", "run_flextool.py"], "run_flextool.py", "ask"),
        (["python", "-c", "import flextool"], "-c", "ask"),
        (["python", "-m", "pytest", "tests"], "-m", "ask"),
        (["python"], "", "ask"),
    ],
)
def test_import_entry_mode(orig_argv, argv0, expected):
    assert env_check.import_entry_mode(orig_argv, argv0) == expected


@pytest.mark.parametrize(
    ("status", "dists", "env", "expected"),
    [
        (env_check.CPU_FULL, {"polars": "1.40.1"}, {}, "not_needed"),
        (env_check.CPU_UNKNOWN, {"polars": "1.40.1"}, {}, "not_needed"),
        (env_check.CPU_NEEDS_COMPAT, {"polars": "1.40.1", "polars_runtime_compat": "1.40.1"}, {}, "not_needed"),
        (env_check.CPU_NEEDS_COMPAT, {"polars": "1.40.1"}, {"FLEXTOOL_NO_ENV_CHECK": "1"}, "skipped"),
        (env_check.CPU_NEEDS_COMPAT, {"polars": "1.40.1"}, {"POLARS_FORCE_PKG": "32"}, "skipped"),
        (env_check.CPU_UNSUPPORTED, {"polars": "1.40.1"}, {}, "unsupported"),
    ],
)
def test_import_guard_leaves_environment_alone(monkeypatch, status, dists, env, expected):
    _no_polars_imported(monkeypatch)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _fake_dists(monkeypatch, **dists)
    _fake_cpu(monkeypatch, status, ["avx2"], ["sse4.2"] if status == env_check.CPU_UNSUPPORTED else [])
    ran = _Ran()
    ask = lambda _p: pytest.fail("asked")  # noqa: E731
    assert env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=False) == expected
    assert ran.calls == []


def test_import_guard_skips_when_polars_already_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", object())
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    ran = _Ran()
    assert env_check.ensure_polars_runtime_before_import(run=ran) == "skipped"
    assert ran.calls == []


# ── default runtime next to the compatible one ────────────────────

_BOTH_RUNTIMES = {"polars": "2.1.0", "polars_runtime_32": "2.1.0", "polars_runtime_compat": "2.1.0"}


@pytest.mark.parametrize(
    ("status", "dists", "preset", "expected"),
    [
        # AVX2 CPU with both runtimes (the Toolbox bundle, flextool[compat]): the default one.
        (env_check.CPU_FULL, _BOTH_RUNTIMES, None, "32"),
        (env_check.CPU_FULL, {**_BOTH_RUNTIMES, "polars_runtime_64": "2.1.0"}, None, "64"),
        # The user's own choice stays.
        (env_check.CPU_FULL, _BOTH_RUNTIMES, "compat", "compat"),
        # No compatible runtime: polars loads the default one anyway, nothing to set.
        (env_check.CPU_FULL, {"polars": "2.1.0", "polars_runtime_32": "2.1.0"}, None, None),
        # A default runtime of another version would be skipped by polars: leave the order alone.
        (env_check.CPU_FULL, {**_BOTH_RUNTIMES, "polars_runtime_32": "2.0.0"}, None, None),
        # Not confirmed to run the default runtime: polars keeps loading the compatible one.
        (env_check.CPU_NEEDS_COMPAT, _BOTH_RUNTIMES, None, None),
        (env_check.CPU_UNKNOWN, _BOTH_RUNTIMES, None, None),
    ],
)
def test_import_guard_selects_the_default_runtime_on_an_avx2_cpu(monkeypatch, status, dists, preset, expected):
    _no_polars_imported(monkeypatch)
    # setenv first, so the variable the guard sets is undone after the test.
    monkeypatch.setenv("POLARS_PREFER_PKG", preset or "")
    if preset is None:
        monkeypatch.delenv("POLARS_PREFER_PKG")
    _fake_dists(monkeypatch, **dists)
    _fake_cpu(monkeypatch, status, [] if status == env_check.CPU_FULL else ["avx2"])
    ran = _Ran()
    ask = lambda _p: pytest.fail("asked")  # noqa: E731
    assert env_check.ensure_polars_runtime_before_import(run=ran, ask=ask, interactive=False) == "not_needed"
    assert os.environ.get("POLARS_PREFER_PKG") == expected
    assert ran.calls == []


def test_runtime_selection_respects_the_opt_outs(monkeypatch):
    _no_polars_imported(monkeypatch)
    monkeypatch.setenv("POLARS_PREFER_PKG", "")
    monkeypatch.delenv("POLARS_PREFER_PKG")
    monkeypatch.setenv("FLEXTOOL_NO_ENV_CHECK", "1")
    _fake_dists(monkeypatch, **_BOTH_RUNTIMES)
    _fake_cpu(monkeypatch, env_check.CPU_FULL)
    assert env_check.ensure_polars_runtime_before_import(run=_Ran()) == "skipped"
    assert "POLARS_PREFER_PKG" not in os.environ


def test_probe_reports_the_loaded_runtime():
    probe = env_check.StackProbe(env_check.OK, 0, stdout="PROBE:begin\nPROBE:polars\nRUNTIME:rtcompat\nPROBE:all_ok")
    assert probe.polars_runtime == "rtcompat"
    assert env_check.StackProbe(env_check.NATIVE_FAULT, -4, stdout="PROBE:begin").polars_runtime is None


# ── unattended heal of "Update FlexTool" ──────────────────────────

def _ok_probe(*_args, **_kwargs):
    return env_check.StackProbe(env_check.OK, 0)


def test_update_heals_non_avx2_cpu_before_probing(monkeypatch, capsys):
    from flextool.update_flextool import self_update

    _fake_dists(monkeypatch, polars="2.1.0", polars_runtime_32="2.1.0")
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    order = []
    ran = _Ran()

    def run(argv, *a, **k):
        if list(argv[1:4]) == ["-P", "-m", "pip"]:
            order.append("pip")
        return ran(argv, *a, **k)

    def probe(*a, **k):
        order.append("probe")
        return _ok_probe()

    monkeypatch.setattr(self_update.subprocess, "run", run)
    monkeypatch.setattr(env_check, "probe_solver_stack", probe)
    self_update._ensure_compatible_solver_stack()
    assert ran.calls == [[sys.executable, "-P", "-m", "pip", "install", "polars[rtcompat]==2.1.0"]]
    assert order[0] == "pip" and "probe" in order


def test_update_on_unsupported_cpu_installs_nothing(monkeypatch, capsys):
    from flextool.update_flextool import self_update

    _fake_dists(monkeypatch, polars="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_UNSUPPORTED, ["avx2"], ["sse4.2", "popcnt"])
    ran = _Ran()
    monkeypatch.setattr(self_update.subprocess, "run", ran)
    monkeypatch.setattr(env_check, "probe_solver_stack", _ok_probe)
    self_update._ensure_compatible_solver_stack()
    assert ran.calls == []
    assert "x86-64-v2" in capsys.readouterr().out


def test_update_heals_polars_crash_with_compat_runtime(monkeypatch):
    from flextool.update_flextool import self_update

    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_UNKNOWN)
    probes = iter([
        env_check.StackProbe(env_check.NATIVE_FAULT, -4, failed_component="polars"),
        env_check.StackProbe(env_check.OK, 0),
    ])
    ran = _Ran()
    monkeypatch.setattr(self_update.subprocess, "run", ran)
    monkeypatch.setattr(env_check, "probe_solver_stack", lambda *a, **k: next(probes))
    self_update._ensure_compatible_solver_stack()
    assert ran.calls == [[sys.executable, "-P", "-m", "pip", "install", "polars[rtcompat]==1.40.1"]]


# ── fingerprint + installed-build introspection ───────────────────

def test_env_fingerprint_stable_and_short():
    fp1 = env_check.env_fingerprint()
    fp2 = env_check.env_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 16
    int(fp1, 16)  # hex


def test_installed_polars_reports_a_known_build():
    dist, ver = env_check.installed_polars()
    assert dist in ("polars", "polars-lts-cpu")
    assert ver  # non-empty version string


def test_diagnostics_report_mentions_key_fields():
    report = env_check.diagnostics_report()
    assert "python" in report
    assert "polars build" in report
    assert "highspy" in report
    assert "solver stack:" in report  # the probe summary line


def test_env_check_cli_runs_as_a_fresh_module():
    """``import flextool`` runs the CPU check before polars without
    registering ``flextool.env_check``, so ``python -m flextool.env_check``
    gets no runpy double-import warning."""
    done = subprocess.run(
        [sys.executable, "-W", "error::RuntimeWarning", "-c",
         "import sys, flextool; assert 'flextool.env_check' not in sys.modules, 'registered'"],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 0, done.stderr
