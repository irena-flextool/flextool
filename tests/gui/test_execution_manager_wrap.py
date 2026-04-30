"""Tests for `_wrap_for_memory_cap` — the per-job memory cap shim.

These tests are hermetic: they mock `sys.platform`, `shutil.which`, the
slice-probe `subprocess.run`, and the `FLEXTOOL_SLICE` env var. No real
subprocess runs, so they are safe on Linux/Windows/macOS CI runners.
"""

from __future__ import annotations

import subprocess
import sys
import types
from unittest.mock import MagicMock, patch

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

    def test_no_slice_with_estimate_sets_high_and_max(self):
        with patch.object(em.sys, "platform", "linux"), \
             patch.object(em.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.dict(em.os.environ, {}, clear=False):
            em.os.environ.pop("FLEXTOOL_SLICE", None)
            cmd, _, _ = em._wrap_for_memory_cap(["python", "x.py"], 10.0)
        # 10 GB → 10240 MB cap, MemoryHigh = 90% = 9216 MB
        assert "-p" in cmd
        assert "MemoryHigh=9216M" in cmd
        assert "MemoryMax=10240M" in cmd
        # MemoryMax must come AFTER its -p flag
        i = cmd.index("MemoryMax=10240M")
        assert cmd[i - 1] == "-p"
        # And the actual command follows the -- separator
        sep = cmd.index("--")
        assert cmd[sep + 1:] == ["python", "x.py"]

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
    """Windows branch — uses pywin32 Job Object. Tests run on any platform
    by injecting a fake `win32job` / `win32api` / `win32con` module trio."""

    @pytest.fixture
    def fake_win32(self, monkeypatch):
        """Install fake pywin32 modules in sys.modules; return the mocks."""
        fake_job = MagicMock(name="win32job")
        fake_job.JobObjectExtendedLimitInformation = 9
        fake_job.JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
        # CreateJobObject returns an opaque sentinel handle
        fake_job.CreateJobObject.return_value = "JOB_HANDLE"
        fake_job.QueryInformationJobObject.return_value = {
            "BasicLimitInformation": {"LimitFlags": 0},
            "JobMemoryLimit": 0,
        }
        fake_api = MagicMock(name="win32api")
        fake_api.OpenProcess.return_value = "PROC_HANDLE"
        fake_con = types.SimpleNamespace(
            PROCESS_SET_QUOTA=0x0100, PROCESS_TERMINATE=0x0001,
        )
        monkeypatch.setitem(sys.modules, "win32job", fake_job)
        monkeypatch.setitem(sys.modules, "win32api", fake_api)
        monkeypatch.setitem(sys.modules, "win32con", fake_con)
        return fake_job, fake_api, fake_con

    def test_zero_estimate_no_post_spawn(self, fake_win32):
        with patch.object(em.sys, "platform", "win32"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 0.0)
        assert cmd == ["python", "x.py"]
        assert extras == {} and post is None

    def test_pywin32_missing_falls_back(self, monkeypatch, caplog):
        # Force `import win32job` to fail
        monkeypatch.setitem(sys.modules, "win32job", None)
        with caplog.at_level("WARNING", logger=em.logger.name), \
             patch.object(em.sys, "platform", "win32"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 4.0)
        assert cmd == ["python", "x.py"]
        assert extras == {} and post is None
        assert any("pywin32" in r.message for r in caplog.records)

    def test_post_spawn_assigns_to_job_with_cap(self, fake_win32):
        fake_job, fake_api, _ = fake_win32
        with patch.object(em.sys, "platform", "win32"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 2.0)
        assert cmd == ["python", "x.py"]
        assert extras == {}
        assert callable(post)

        fake_proc = MagicMock(spec=subprocess.Popen)
        fake_proc.pid = 12345
        handle = post(fake_proc)

        assert handle == "JOB_HANDLE"
        fake_job.CreateJobObject.assert_called_once()
        # Limit info written with cap_bytes = 2 GiB and JOB_MEMORY flag
        set_call = fake_job.SetInformationJobObject.call_args
        info = set_call.args[2]
        assert info["JobMemoryLimit"] == 2 * (1024 ** 3)
        assert info["BasicLimitInformation"]["LimitFlags"] & 0x200
        # Process opened by pid and assigned to job
        fake_api.OpenProcess.assert_called_once_with(
            0x0100 | 0x0001, False, 12345,
        )
        fake_job.AssignProcessToJobObject.assert_called_once_with(
            "JOB_HANDLE", "PROC_HANDLE",
        )
        fake_api.CloseHandle.assert_called_once_with("PROC_HANDLE")

    def test_post_spawn_swallows_exceptions(self, fake_win32, caplog):
        fake_job, _, _ = fake_win32
        fake_job.AssignProcessToJobObject.side_effect = RuntimeError("boom")
        with caplog.at_level("ERROR", logger=em.logger.name), \
             patch.object(em.sys, "platform", "win32"):
            _, _, post = em._wrap_for_memory_cap(["python", "x.py"], 2.0)
            handle = post(MagicMock(pid=99))
        assert handle is None
        assert any("JobObject" in r.message for r in caplog.records)


class TestMacosBranch:
    def test_zero_estimate_no_preexec(self):
        with patch.object(em.sys, "platform", "darwin"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 0.0)
        assert cmd == ["python", "x.py"]
        assert extras == {} and post is None

    def test_sets_preexec_with_rlimit_as(self, monkeypatch):
        fake_resource = MagicMock(name="resource")
        fake_resource.RLIMIT_AS = 99
        monkeypatch.setitem(sys.modules, "resource", fake_resource)
        with patch.object(em.sys, "platform", "darwin"):
            cmd, extras, post = em._wrap_for_memory_cap(["python", "x.py"], 3.0)
        assert cmd == ["python", "x.py"]
        assert post is None
        assert "preexec_fn" in extras and callable(extras["preexec_fn"])

        extras["preexec_fn"]()
        cap_bytes = 3 * (1024 ** 3)
        fake_resource.setrlimit.assert_called_once_with(99, (cap_bytes, cap_bytes))

    def test_preexec_swallows_setrlimit_failure(self, monkeypatch):
        fake_resource = MagicMock(name="resource")
        fake_resource.RLIMIT_AS = 99
        fake_resource.setrlimit.side_effect = OSError("EPERM")
        monkeypatch.setitem(sys.modules, "resource", fake_resource)
        with patch.object(em.sys, "platform", "darwin"):
            _, extras, _ = em._wrap_for_memory_cap(["python", "x.py"], 3.0)
        # Must not raise — preexec failure leaves child running uncapped,
        # caught by MemoryWatchdog
        extras["preexec_fn"]()
