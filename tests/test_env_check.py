"""Unit tests for flextool.env_check — the native-environment self-check.

These are stdlib-only and fast: the native-fault classifier is exercised for
both platform families on one host, and the live probe runs against the
current interpreter (which, in any environment able to run the test suite,
has a working polars).
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


# ── live probe against the current interpreter ────────────────────

def test_probe_polars_ok_here():
    """The interpreter running the tests has a working polars."""
    result = env_check.probe_polars()
    assert result.ok, result.summary() + "\n" + result.stderr
    assert result.is_native_fault is False
    assert "polars: OK" == result.summary()


# ── swap steps ────────────────────────────────────────────────────

def test_swap_steps_shape():
    steps = env_check.swap_to_lts_cpu_steps(python="/some/python")
    assert len(steps) == 2
    uninstall, install = steps
    # Standard polars must be removed first (it owns the 'polars' import name).
    assert uninstall[:5] == ["/some/python", "-m", "pip", "uninstall", "-y"]
    assert "polars" in uninstall
    assert install[:4] == ["/some/python", "-m", "pip", "install"]
    assert install[-1] == "polars-lts-cpu"


def test_swap_steps_default_python_is_current():
    steps = env_check.swap_to_lts_cpu_steps()
    assert steps[0][0] == sys.executable


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
    assert "polars:" in report  # the probe summary line
