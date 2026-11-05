"""Unit tests for flextool.env_check — the native solver-stack self-check.

Stdlib-only and fast: the native-fault classifier and the failed-component
classifier are exercised directly, and the live probe runs against the
current interpreter (which, in any environment able to run the test suite,
has a working solver stack).
"""

import platform
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


@pytest.mark.parametrize("rc", [0, 1, 2, 3, 9009])
def test_windows_normal_exits_are_not_faults(rc):
    assert env_check.is_native_fault(rc, plat="win32") is False


@pytest.mark.parametrize("rc", [-11, -4, -7, -6, -8])  # SEGV ILL BUS ABRT FPE
def test_posix_native_faults(rc):
    assert env_check.is_native_fault(rc, plat="linux") is True


@pytest.mark.parametrize("rc", [0, 1, 2, -15, -2])  # incl. SIGTERM, SIGINT
def test_posix_non_faults(rc):
    assert env_check.is_native_fault(rc, plat="linux") is False


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
        ["/some/python", "-m", "pip", "install", "polars[rtcompat]==1.40.1"],
    ]


def test_polars_heal_follows_an_upgraded_polars(monkeypatch):
    # A compat runtime of an older version is skipped by polars' loader.
    _fake_dists(monkeypatch, polars="2.1.0", polars_runtime_32="2.1.0", polars_runtime_compat="1.40.1")
    assert env_check.compat_runtime_ready() is False
    assert env_check.polars_compat_steps(python="/p") == [
        ["/p", "-m", "pip", "install", "polars[rtcompat]==2.1.0"],
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

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    core = [d for d in pyproject["project"]["dependencies"] if d.startswith("polars")]
    assert len(core) == 1
    _fake_dists(monkeypatch, polars_lts_cpu="1.33.1", polars_runtime_32="1.40.1")
    uninstall, install = env_check.polars_compat_steps(python="/p")
    assert uninstall == ["/p", "-m", "pip", "uninstall", "-y", "polars-lts-cpu", "polars"]
    assert install[:4] == ["/p", "-m", "pip", "install"]
    assert install[4].startswith("polars[rtcompat]")
    assert _requirement_parts(install[4][len("polars[rtcompat]"):]) == _requirement_parts(core[0][len("polars"):])


def test_compat_extra_matches_core_polars_requirement():
    import tomllib

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    core = [d for d in pyproject["project"]["dependencies"] if d.startswith("polars")][0]
    (compat,) = pyproject["project"]["optional-dependencies"]["compat"]
    assert compat.startswith("polars[rtcompat]")
    assert _requirement_parts(compat[len("polars[rtcompat]"):]) == _requirement_parts(core[len("polars"):])


def test_highspy_downgrade_steps_shape():
    steps = env_check.downgrade_highspy_steps(python="/some/python")
    assert len(steps) == 1
    assert steps[0] == [
        "/some/python", "-m", "pip", "install",
        f"highspy=={env_check.HIGHSPY_GOOD_VERSION}",
    ]
    assert env_check.HIGHSPY_GOOD_VERSION == "1.13.1"


def test_default_python_is_current_interpreter(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1")
    assert env_check.polars_compat_steps()[0][0] == sys.executable
    assert env_check.downgrade_highspy_steps()[0][0] == sys.executable


def test_remediation_dispatch_by_component(monkeypatch):
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1")
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
        if list(argv[1:3]) != ["-m", "pip"]:
            return _REAL_RUN(argv, *args, **kwargs)
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, self.returncode)


@pytest.fixture(autouse=True)
def _private_install_lock(monkeypatch, tmp_path):
    lock = str(tmp_path / "compat.lock")
    monkeypatch.setattr(env_check, "_install_lock_path", lambda: lock)
    return lock


def _no_polars_imported(monkeypatch):
    for name in [m for m in sys.modules if m == "polars" or m.startswith("polars.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delenv("FLEXTOOL_NO_ENV_CHECK", raising=False)
    monkeypatch.delenv("POLARS_FORCE_PKG", raising=False)


def test_import_guard_installs_compat_runtime_on_non_avx2_cpu(monkeypatch, capsys):
    _no_polars_imported(monkeypatch)
    _fake_dists(monkeypatch, polars="1.40.1", polars_runtime_32="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2", "fma"])
    ran = _Ran()
    assert env_check.ensure_polars_runtime_before_import(run=ran) == "installed"
    assert ran.calls == [[sys.executable, "-m", "pip", "install", "polars[rtcompat]==1.40.1"]]
    assert "avx2, fma" in capsys.readouterr().err


def test_import_guard_reports_failed_install(monkeypatch, capsys):
    _no_polars_imported(monkeypatch)
    _fake_dists(monkeypatch, polars="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    assert env_check.ensure_polars_runtime_before_import(run=_Ran(returncode=1)) == "failed"
    assert "polars[rtcompat]==1.40.1" in capsys.readouterr().err


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
    assert env_check.ensure_polars_runtime_before_import(run=ran) == expected
    assert ran.calls == []


def test_import_guard_waits_for_a_concurrent_install(monkeypatch, _private_install_lock):
    # Another process holds the lock and finishes the install while this one
    # waits: no second pip run, and the lock is gone afterwards.
    import os

    _no_polars_imported(monkeypatch)
    dists = {"polars": "1.40.1"}
    monkeypatch.setattr(env_check, "_dist_version", lambda name: dists.get(name))
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    open(_private_install_lock, "w").close()
    slept = []

    def other_process_finishes(seconds):
        slept.append(seconds)
        dists["polars-runtime-compat"] = "1.40.1"
        os.remove(_private_install_lock)

    ran = _Ran()
    assert env_check.ensure_polars_runtime_before_import(run=ran, sleep=other_process_finishes) == "not_needed"
    assert slept == [1.0]
    assert ran.calls == []
    assert not os.path.exists(_private_install_lock)


def test_import_guard_takes_over_a_stale_lock(monkeypatch, _private_install_lock):
    import os
    import time

    _no_polars_imported(monkeypatch)
    _fake_dists(monkeypatch, polars="1.40.1")
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    open(_private_install_lock, "w").close()
    old = time.time() - 2 * env_check._INSTALL_LOCK_STALE_S
    os.utime(_private_install_lock, (old, old))
    ran = _Ran()
    assert env_check.ensure_polars_runtime_before_import(run=ran, sleep=lambda s: pytest.fail("waited")) == "installed"
    assert len(ran.calls) == 1
    assert not os.path.exists(_private_install_lock)


def test_import_guard_skips_when_polars_already_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "polars", object())
    _fake_cpu(monkeypatch, env_check.CPU_NEEDS_COMPAT, ["avx2"])
    ran = _Ran()
    assert env_check.ensure_polars_runtime_before_import(run=ran) == "skipped"
    assert ran.calls == []


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
        if list(argv[1:3]) == ["-m", "pip"]:
            order.append("pip")
        return ran(argv, *a, **k)

    def probe(*a, **k):
        order.append("probe")
        return _ok_probe()

    monkeypatch.setattr(self_update.subprocess, "run", run)
    monkeypatch.setattr(env_check, "probe_solver_stack", probe)
    self_update._ensure_compatible_solver_stack()
    assert ran.calls == [[sys.executable, "-m", "pip", "install", "polars[rtcompat]==2.1.0"]]
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
    assert ran.calls == [[sys.executable, "-m", "pip", "install", "polars[rtcompat]==1.40.1"]]


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
