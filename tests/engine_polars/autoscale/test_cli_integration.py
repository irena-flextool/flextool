"""Integration tests for the ``--auto-scale`` CLI flag wiring.

Phase 1f's contract:

* CLI flag ``--auto-scale {on,off}`` (added in
  ``flextool/cli/cmd_run_flextool.py``) is the operator-facing surface.
* ``resolve_auto_scale_config(args)`` honours CLI > env > default-on
  precedence for the master switch.
* ``--user-bound-scale N`` continues to flow through into
  ``AutoScaleConfig.user_bound_scale`` so Layer 3's manual-override
  branch fires (see ``recommend_layer3``).

These tests exercise the resolver directly with a simple ``Namespace``
stand-in for the parsed ``argparse`` args — that's the same shape
``cmd_run_flextool`` produces and the same shape cascade-internal call
sites receive when args are threaded.
"""
from __future__ import annotations

from argparse import Namespace

import pytest

from flextool.engine_polars.autoscale import (
    AutoScaleConfig,
    resolve_auto_scale_config,
)


@pytest.fixture(autouse=True)
def _clear_autoscale_env(monkeypatch):
    """Wipe FLEXTOOL_AUTO_SCALE / FLEXTOOL_USER_BOUND_SCALE per test.

    Without this fixture, a stray env value from the developer's shell
    or a sibling test's monkeypatch leak would flip the master switch
    and mask the CLI-precedence behaviour we're asserting.
    """
    monkeypatch.delenv("FLEXTOOL_AUTO_SCALE", raising=False)
    monkeypatch.delenv("FLEXTOOL_USER_BOUND_SCALE", raising=False)


def test_auto_scale_default_on():
    """CLI unset (``args.auto_scale = None``) + env unset → enabled."""
    args = Namespace(auto_scale=None, user_bound_scale=None)
    cfg = resolve_auto_scale_config(args)
    assert isinstance(cfg, AutoScaleConfig)
    assert cfg.enabled is True


def test_auto_scale_cli_off():
    """CLI ``--auto-scale=off`` disables the autoscaler."""
    args = Namespace(auto_scale="off", user_bound_scale=None)
    cfg = resolve_auto_scale_config(args)
    assert cfg.enabled is False


def test_auto_scale_cli_on_overrides_env_off(monkeypatch):
    """CLI ``on`` wins over env ``off`` — operator intent is explicit."""
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "off")
    args = Namespace(auto_scale="on", user_bound_scale=None)
    cfg = resolve_auto_scale_config(args)
    assert cfg.enabled is True


def test_auto_scale_env_off(monkeypatch):
    """CLI unset + env ``off`` → disabled (env fallback)."""
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "off")
    args = Namespace(auto_scale=None, user_bound_scale=None)
    cfg = resolve_auto_scale_config(args)
    assert cfg.enabled is False


def test_user_bound_scale_manual_override_propagates():
    """``--user-bound-scale -8`` populates ``cfg.user_bound_scale = -8``.

    This is the contract Layer 3's manual-override branch consumes —
    see ``recommend_layer3`` — to emit
    ``reasoning="manual override user_bound_scale=-8"``.
    """
    args = Namespace(auto_scale=None, user_bound_scale=-8)
    cfg = resolve_auto_scale_config(args)
    assert cfg.user_bound_scale == -8


def test_auto_scale_cli_off_resolves_to_false_even_with_env_on(monkeypatch):
    """Tightens CLI > env: ``--auto-scale=off`` beats env ``on`` too.

    Symmetric with ``test_auto_scale_cli_on_overrides_env_off`` — a
    half-implemented precedence chain (e.g. one that only honoured CLI
    when the env said off) would slip past the on→off test, so we
    cover the other direction explicitly.
    """
    monkeypatch.setenv("FLEXTOOL_AUTO_SCALE", "on")
    args = Namespace(auto_scale="off", user_bound_scale=None)
    cfg = resolve_auto_scale_config(args)
    assert cfg.enabled is False
