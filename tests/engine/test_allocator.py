"""polars' jemalloc decay config, with polars imported before or after flextool (``flextool/_allocator.py``).

The import-order tests each run in a fresh interpreter, because jemalloc reads its config once per process and
the outcome depends on what was imported first.  The guard tests run in-process with the mallctl call replaced,
so they never touch this process's allocator.
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import textwrap

import pytest

from flextool import _allocator

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="polars uses jemalloc in an ELF library on Linux")

_READ_DECAY = """
from flextool import _allocator as A
ctl = A.Mallctl.from_library(*A.runtime_library())
out = {k: ctl.read_ssize(k) for k in ("arenas.dirty_decay_ms", "arenas.muzzy_decay_ms",
                                       "arena.0.dirty_decay_ms", "arena.0.muzzy_decay_ms")}
print("DECAY " + json.dumps(out))
"""


def _run(body: str, env_conf: str | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != _allocator.ENV_VAR}
    if env_conf is not None:
        env[_allocator.ENV_VAR] = env_conf
    env.setdefault("POLARS_MAX_THREADS", "2")
    return subprocess.run([sys.executable, "-c", "import json, os, sys, time, gc\n" + textwrap.dedent(body)],
                          env=env, capture_output=True, text=True, timeout=timeout)


def _decay(proc: subprocess.CompletedProcess) -> dict:
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("DECAY "))
    return json.loads(line[len("DECAY "):])


def _polars_first(env_conf=None, between="") -> subprocess.CompletedProcess:
    return _run("import polars\n" + between + "\nimport flextool\n" + _READ_DECAY, env_conf)


def _flextool_first(env_conf=None) -> subprocess.CompletedProcess:
    return _run("import flextool\nimport polars\n" + _READ_DECAY, env_conf)


def test_polars_first_gets_the_same_decay_as_flextool_first():
    first = _polars_first()
    assert "FlexTool:" not in first.stderr
    after = _decay(first)
    assert after == _decay(_flextool_first())
    assert after["arena.0.muzzy_decay_ms"] == 0 and after["arenas.dirty_decay_ms"] == 1000


def test_conf_set_before_polars_wins_whichever_import_comes_first():
    conf = "muzzy_decay_ms:5"
    after = _decay(_polars_first(conf))
    assert after == _decay(_flextool_first(conf))
    assert after["arena.0.muzzy_decay_ms"] == 5


def test_conf_set_after_polars_import_is_applied_at_runtime():
    after = _decay(_polars_first(between=f"os.environ[{_allocator.ENV_VAR!r}] = 'dirty_decay_ms:2000,muzzy_decay_ms:7'"))
    assert after["arena.0.dirty_decay_ms"] == 2000 and after["arena.0.muzzy_decay_ms"] == 7
    assert after["arenas.dirty_decay_ms"] == 2000 and after["arenas.muzzy_decay_ms"] == 7


def test_pages_freed_before_flextool_is_imported_are_returned():
    # ~0.7 GB of frames is freed while polars' own config (muzzy -1) is active: the pages stay resident.  Importing
    # flextool switches the decay, and jemalloc's background thread then returns them.
    proc = _run("""
        import polars as pl
        def rss():
            return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
        base = rss()
        df = pl.DataFrame({"a": pl.int_range(0, 30_000_000, eager=True)}).with_columns(b=pl.col("a").cast(pl.Utf8))
        peak = rss()
        del df
        gc.collect()
        time.sleep(3)
        freed = rss()
        import flextool
        time.sleep(4)
        print("RSS " + json.dumps({"base": base, "peak": peak, "freed": freed, "after": rss()}))
    """)
    assert proc.returncode == 0, proc.stderr[-2000:]
    r = json.loads(next(ln for ln in proc.stdout.splitlines() if ln.startswith("RSS "))[4:])
    held = r["freed"] - r["base"]
    assert held > 0.5 * (r["peak"] - r["base"]), r
    assert r["after"] - r["base"] < 0.3 * held, r


# ─── guards (in-process; the mallctl call is replaced) ──────────────────────

def _polars_written_conf() -> str:
    """The value ``import polars`` writes when nothing was set before: the decay values jemalloc read at load."""
    ctl = _allocator.Mallctl.from_library(*_allocator.runtime_library())
    return f"dirty_decay_ms:{ctl.read_ssize('opt.dirty_decay_ms')},muzzy_decay_ms:{ctl.read_ssize('opt.muzzy_decay_ms')}"


@pytest.fixture
def polars_first_env(monkeypatch):
    """This process looks like a polars-first host with polars' own config only, and nothing is applied."""
    import polars  # noqa: F401
    monkeypatch.setenv(_allocator.ENV_VAR, _polars_written_conf())
    monkeypatch.setitem(_allocator._state, "warned", False)
    calls = []
    monkeypatch.setattr(_allocator, "apply_decay", lambda ctl, targets: calls.append(targets))
    return calls


def _no_cfunctype(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the mallctl function must not be built")
    monkeypatch.setattr(ctypes, "CFUNCTYPE", boom)


def test_unpatched_guard_setup_reaches_the_call(polars_first_env, capsys):
    _allocator.configure_polars_allocator()
    assert polars_first_env == [{"dirty_decay_ms": 1000, "muzzy_decay_ms": 0}]
    assert capsys.readouterr().err == ""


def test_absent_symbol_warns_and_calls_nothing(polars_first_env, monkeypatch, capsys):
    monkeypatch.setattr(_allocator, "MALLCTL_SYMBOL", "_rjem_no_such_symbol")
    _no_cfunctype(monkeypatch)
    _allocator.configure_polars_allocator()
    _allocator.configure_polars_allocator()
    assert polars_first_env == []
    err = capsys.readouterr().err
    assert err.count("FlexTool:") == 1 and "_rjem_no_such_symbol" in err and _allocator.DEFAULT_CONF in err


def test_code_bytes_mismatch_warns_and_calls_nothing(polars_first_env, monkeypatch, capsys):
    real = _allocator.read_memory
    monkeypatch.setattr(_allocator, "read_memory", lambda addr, n: bytes(b ^ 0xFF for b in real(addr, n)))
    _no_cfunctype(monkeypatch)
    _allocator.configure_polars_allocator()
    assert polars_first_env == []
    err = capsys.readouterr().err
    assert err.count("FlexTool:") == 1 and "differs from the library file" in err


def test_library_without_jemalloc_is_silent(polars_first_env, monkeypatch, capsys):
    monkeypatch.setattr(_allocator, "has_jemalloc", lambda path: False)
    _no_cfunctype(monkeypatch)
    _allocator.configure_polars_allocator()
    assert polars_first_env == [] and capsys.readouterr().err == ""


def test_windows_is_silent(polars_first_env, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    _allocator.configure_polars_allocator()
    assert polars_first_env == [] and capsys.readouterr().err == ""


def test_conf_set_before_polars_is_left_alone(polars_first_env, monkeypatch, capsys):
    monkeypatch.setenv(_allocator.ENV_VAR, _polars_written_conf() + ",background_thread:false")
    _allocator.configure_polars_allocator()
    assert polars_first_env == [] and capsys.readouterr().err == ""


def test_child_processes_inherit_flextool_decay(polars_first_env):
    _allocator.configure_polars_allocator()
    assert _allocator.decay_targets(os.environ[_allocator.ENV_VAR]) == {"dirty_decay_ms": 1000, "muzzy_decay_ms": 0}


def test_failing_mallctl_write_is_unreachable_and_uninitialised_arenas_are_skipped():
    def fake(rc_for):
        written = []

        def fn(name, oldp, oldlenp, newp, newlen):
            name = name.decode()
            if oldp is not None:  # reads: two arenas, decay values that differ from the targets
                oldp._obj.value = 2 if name == "arenas.narenas" else -1
                return 0
            written.append(name)
            return rc_for(name)
        return _allocator.Mallctl(fn), written

    ctl, written = fake(lambda name: 14 if name.startswith("arena.1.") else 0)
    _allocator.apply_decay(ctl, {"muzzy_decay_ms": 0})
    assert written == ["arenas.muzzy_decay_ms", "arena.0.muzzy_decay_ms", "arena.1.muzzy_decay_ms"]
    ctl, written = fake(lambda name: 1 if name == "arena.0.muzzy_decay_ms" else 0)
    with pytest.raises(_allocator.AllocatorUnreachable, match="returned 1"):
        _allocator.apply_decay(ctl, {"muzzy_decay_ms": 0})
