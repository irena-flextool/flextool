"""Behaviour of ``flextool.cli._console.run_tool`` under the two runtimes a
FlexTool Tool runs in: a standalone CLI process, and Spine Toolbox's *Basic
Console* (a persistent ``python -i`` REPL, ``sys.flags.interactive`` set).

The critical contract: under ``-i`` ``run_tool`` must NEVER call ``sys.exit()``
(that terminates the REPL and Toolbox reports a spurious "Kernel died");
it signals success by returning and failure by raising.  Standalone, it
preserves normal exit-code semantics.
"""
import sys
import types

import pytest

from flextool.cli import _console


def _force_interactive(monkeypatch, value):
    """Replace ``sys.flags`` with a stand-in exposing ``.interactive``."""
    monkeypatch.setattr(sys, "flags", types.SimpleNamespace(interactive=value))


# --- standalone CLI (interactive == 0): normal exit-code semantics ---

def test_standalone_returns_code_via_sys_exit(monkeypatch):
    _force_interactive(monkeypatch, 0)
    with pytest.raises(SystemExit) as exc:
        _console.run_tool(lambda: 3)
    assert exc.value.code == 3


def test_standalone_none_exits_zero(monkeypatch):
    _force_interactive(monkeypatch, 0)
    with pytest.raises(SystemExit) as exc:
        _console.run_tool(lambda: None)
    assert exc.value.code in (0, None)


def test_standalone_reraises_systemexit(monkeypatch):
    _force_interactive(monkeypatch, 0)

    def entry():
        sys.exit(7)

    with pytest.raises(SystemExit) as exc:
        _console.run_tool(entry)
    assert exc.value.code == 7


# --- Basic Console (interactive == 1): never sys.exit; raise on failure ---

def test_interactive_success_returns_quietly(monkeypatch):
    _force_interactive(monkeypatch, 1)
    # rc == 0 -> returns, no SystemExit (REPL stays alive).
    assert _console.run_tool(lambda: 0) == 0


def test_interactive_success_none_returns_quietly(monkeypatch):
    _force_interactive(monkeypatch, 1)
    assert _console.run_tool(lambda: None) is None


def test_interactive_midflow_sys_exit_zero_is_swallowed(monkeypatch):
    _force_interactive(monkeypatch, 1)

    def entry():
        sys.exit(0)  # success early-return inside the tool

    # Must NOT raise (and must NOT propagate SystemExit that would kill REPL).
    assert _console.run_tool(entry) == 0


def test_interactive_nonzero_return_raises_runtimeerror(monkeypatch):
    _force_interactive(monkeypatch, 1)
    with pytest.raises(RuntimeError, match="exit code 1"):
        _console.run_tool(lambda: 1)


def test_interactive_midflow_sys_exit_nonzero_raises_runtimeerror(monkeypatch):
    _force_interactive(monkeypatch, 1)

    def entry():
        sys.exit(2)

    with pytest.raises(RuntimeError, match="exit code 2"):
        _console.run_tool(entry)


def test_interactive_sys_exit_message_raises_runtimeerror(monkeypatch):
    _force_interactive(monkeypatch, 1)

    def entry():
        sys.exit("boom")

    with pytest.raises(RuntimeError, match="boom"):
        _console.run_tool(entry)


def test_interactive_lets_other_exceptions_propagate(monkeypatch):
    _force_interactive(monkeypatch, 1)

    def entry():
        raise ValueError("real bug")

    # A genuine exception must propagate unchanged (Toolbox marks Tool failed).
    with pytest.raises(ValueError, match="real bug"):
        _console.run_tool(entry)
