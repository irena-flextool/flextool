"""Unit tests for flextool.env_check — the native solver-stack self-check.

Stdlib-only and fast: the native-fault classifier and the failed-component
classifier are exercised directly, and the live probe runs against the
current interpreter (which, in any environment able to run the test suite,
has a working solver stack).
"""

import sys

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

def test_polars_swap_steps_shape():
    steps = env_check.swap_to_lts_cpu_steps(python="/some/python")
    assert len(steps) == 2
    uninstall, install = steps
    # Standard polars must be removed first (it owns the 'polars' import name).
    assert uninstall[:5] == ["/some/python", "-m", "pip", "uninstall", "-y"]
    assert "polars" in uninstall
    assert install[:4] == ["/some/python", "-m", "pip", "install"]
    assert install[-1] == "polars-lts-cpu"


def test_highspy_downgrade_steps_shape():
    steps = env_check.downgrade_highspy_steps(python="/some/python")
    assert len(steps) == 1
    assert steps[0] == [
        "/some/python", "-m", "pip", "install",
        f"highspy=={env_check.HIGHSPY_GOOD_VERSION}",
    ]
    assert env_check.HIGHSPY_GOOD_VERSION == "1.13.1"


def test_default_python_is_current_interpreter():
    assert env_check.swap_to_lts_cpu_steps()[0][0] == sys.executable
    assert env_check.downgrade_highspy_steps()[0][0] == sys.executable


def test_remediation_dispatch_by_component():
    assert env_check.has_remediation("polars") is True
    assert env_check.has_remediation("highspy") is True
    assert env_check.has_remediation("polar_high") is False
    assert env_check.has_remediation(None) is False

    assert env_check.remediation_steps("polars", "/p")[-1][-1] == "polars-lts-cpu"
    assert "highspy" in env_check.remediation_steps("highspy", "/p")[0][-1]
    assert env_check.remediation_steps("polar_high") is None

    assert "polars" in env_check.remediation_banner("polars")
    assert "highspy" in env_check.remediation_banner("highspy")
    assert env_check.remediation_banner("polar_high") is None


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
