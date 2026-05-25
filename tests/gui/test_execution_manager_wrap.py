"""Tests for `_wrap_for_memory_cap` — the per-platform subprocess wrap shim.

Post-3751be70 (2026-05-07), OS-level memory caps were removed: Linux
keeps the `systemd-run --scope` wrapper purely for cgroup slice isolation
(`FLEXTOOL_SLICE`), and Windows / macOS are no-ops. `MemoryWatchdog`
(in-process polling) is the sole memory enforcer on all platforms.

These tests are hermetic: they mock `sys.platform`, `shutil.which`, the
slice-probe `subprocess.run`, and the `FLEXTOOL_SLICE` env var. No real
subprocess runs, so they are safe on Linux/Windows/macOS CI runners.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from flextool.gui import execution_manager as em


@pytest.fixture(autouse=True)
def _clear_slice_cache():
    em._slice_probe_cache.clear()
    yield
    em._slice_probe_cache.clear()


def _fake_slice_probe(loaded: bool):
    """Return a `subprocess.run` mock that reports a slice as loaded/not."""
    payload = "LoadState=loaded\n" if loaded else "LoadState=not-found\n"

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=payload, stderr="")
    return _run


class TestLinuxBranch:
    def test_no_systemd_run_returns_unwrapped(self):
        with patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value=None), \
             patch.dict(em.os.environ, {}, clear=False):
            em.os.environ.pop("FLEXTOOL_SLICE", None)
            cmd, extras, post = em._wrap_for_memory_cap(["python", "-c", "x"], 4.0)
        assert cmd == ["python", "-c", "x"]
        assert extras == {}
        assert post is None

    def test_no_slice_no_estimate(self):
        with patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.dict(em.os.environ, {}, clear=False):
            em.os.environ.pop("FLEXTOOL_SLICE", None)
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 0.0)
        assert cmd == [
            "systemd-run", "--user", "--scope", "--quiet",
            "--", "python", "x.py",
        ]
        assert extras == {} and post is None

    def test_slice_loaded_is_appended(self):
        with patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.object(em.subprocess, "run", side_effect=_fake_slice_probe(True)), \
             patch.dict(em.os.environ, {"FLEXTOOL_SLICE": "heavy.slice"}):
            cmd, _, _ = em._wrap_for_memory_cap(["python", "x.py"], 4.0)
        assert "--slice=heavy.slice" in cmd
        # slice flag must appear before the -- separator
        assert cmd.index("--slice=heavy.slice") < cmd.index("--")

    def test_slice_not_loaded_is_skipped(self, caplog):
        with caplog.at_level("WARNING", logger=em.logger.name), \
             patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.object(em.subprocess, "run", side_effect=_fake_slice_probe(False)), \
             patch.dict(em.os.environ, {"FLEXTOOL_SLICE": "missing.slice"}):
            cmd, _, _ = em._wrap_for_memory_cap(["python", "x.py"], 4.0)
        assert not any(arg.startswith("--slice=") for arg in cmd)
        assert any("missing.slice" in r.message for r in caplog.records)

    def test_slice_probe_is_cached(self):
        calls = []

        def _counting_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="LoadState=loaded\n", stderr="")

        with patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.object(em.subprocess, "run", side_effect=_counting_run), \
             patch.dict(em.os.environ, {"FLEXTOOL_SLICE": "heavy.slice"}):
            em._wrap_for_memory_cap(["a"], 1.0)
            em._wrap_for_memory_cap(["b"], 1.0)
            em._wrap_for_memory_cap(["c"], 1.0)
        assert len(calls) == 1, "slice probe should be cached per-process"

    def test_slice_probe_systemctl_missing_treated_as_not_loaded(self):
        with patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.object(em.subprocess, "run", side_effect=FileNotFoundError), \
             patch.dict(em.os.environ, {"FLEXTOOL_SLICE": "heavy.slice"}):
            cmd, _, _ = em._wrap_for_memory_cap(["python", "x.py"], 4.0)
        assert not any(arg.startswith("--slice=") for arg in cmd)


class TestUnknownPlatform:
    def test_returns_unwrapped(self):
        with patch.object(em.sys, "platform", "freebsd"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 8.0)
        assert cmd == ["python", "x.py"]
        assert extras == {} and post is None


class TestWindowsBranch:
    """Windows branch — no OS-level memory cap; MemoryWatchdog is the sole
    enforcer. The wrap call must pass argv through unchanged with no Popen
    extras and no post-spawn callable. Tests removed at WF-2 fix:
    `test_pywin32_missing_falls_back`, `test_post_spawn_assigns_to_job_with_cap`,
    `test_post_spawn_swallows_exceptions` exercised the pywin32 Job Object
    cap path retired in 3751be70 (2026-05-07)."""

    def test_zero_estimate_no_post_spawn(self):
        with patch.object(em.sys, "platform", "win32"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 0.0)
        assert cmd == ["python", "x.py"]
        assert extras == {} and post is None


class TestMacosBranch:
    """macOS branch — no OS-level memory cap; MemoryWatchdog is the sole
    enforcer. Tests removed at WF-2 fix: `test_sets_preexec_with_rlimit_as`
    and `test_preexec_swallows_setrlimit_failure` exercised the RLIMIT_AS
    preexec_fn path retired in 3751be70 (2026-05-07)."""

    def test_zero_estimate_no_preexec(self):
        with patch.object(em.sys, "platform", "darwin"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 0.0)
        assert cmd == ["python", "x.py"]
        assert extras == {} and post is None
