"""Integration tests for the ``--scaling`` CLI flag wiring.

Phase R2's contract:

* CLI flag ``--scaling {off,solver_only,basic,full}`` (added in
  ``flextool/cli/cmd_run_flextool.py``) is the operator-facing surface.
* ``resolve_scaling_config(args)`` honours CLI > env > default-full
  precedence for the mode.
* ``--user-bound-scale N`` continues to flow through into
  ``ScalingConfig.user_bound_scale`` so Layer 3's manual-override
  branch fires (see ``recommend_scaling``).

These tests exercise the resolver directly with a simple ``Namespace``
stand-in for the parsed ``argparse`` args — that's the same shape
``cmd_run_flextool`` produces and the same shape cascade-internal call
sites receive when args are threaded.
"""
from __future__ import annotations

from argparse import Namespace

import pytest

from flextool.engine_polars.autoscale import (
    ScalingConfig,
    ScalingMode,
    resolve_scaling_config,
)


@pytest.fixture(autouse=True)
def _clear_scaling_env(monkeypatch):
    """Wipe FLEXTOOL_SCALING / FLEXTOOL_USER_BOUND_SCALE per test.

    Without this fixture, a stray env value from the developer's shell
    or a sibling test's monkeypatch leak would flip the mode and mask
    the CLI-precedence behaviour we're asserting.
    """
    monkeypatch.delenv("FLEXTOOL_SCALING", raising=False)
    monkeypatch.delenv("FLEXTOOL_USER_BOUND_SCALE", raising=False)


def test_scaling_default_full():
    """CLI unset (``args.scaling = None``) + env unset → FULL."""
    args = Namespace(scaling=None, user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert isinstance(cfg, ScalingConfig)
    assert cfg.mode is ScalingMode.FULL


def test_scaling_cli_off():
    """CLI ``--scaling=off`` selects OFF mode."""
    args = Namespace(scaling="off", user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.OFF


def test_scaling_cli_solver_only():
    """CLI ``--scaling=solver_only`` selects SOLVER_ONLY mode."""
    args = Namespace(scaling="solver_only", user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.SOLVER_ONLY


def test_scaling_cli_basic():
    """CLI ``--scaling=basic`` selects BASIC mode."""
    args = Namespace(scaling="basic", user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.BASIC


def test_scaling_cli_full():
    """CLI ``--scaling=full`` selects FULL mode."""
    args = Namespace(scaling="full", user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.FULL


def test_scaling_cli_full_overrides_env_off(monkeypatch):
    """CLI ``full`` wins over env ``off`` — operator intent is explicit."""
    monkeypatch.setenv("FLEXTOOL_SCALING", "off")
    args = Namespace(scaling="full", user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.FULL


def test_scaling_env_off(monkeypatch):
    """CLI unset + env ``off`` → OFF (env fallback)."""
    monkeypatch.setenv("FLEXTOOL_SCALING", "off")
    args = Namespace(scaling=None, user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.OFF


def test_scaling_env_solver_only(monkeypatch):
    """CLI unset + env ``solver_only`` → SOLVER_ONLY."""
    monkeypatch.setenv("FLEXTOOL_SCALING", "solver_only")
    args = Namespace(scaling=None, user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.SOLVER_ONLY


def test_user_bound_scale_manual_override_propagates():
    """``--user-bound-scale -8`` populates ``cfg.user_bound_scale = -8``.

    This is the contract Layer 3's manual-override branch consumes —
    see ``recommend_scaling`` — to emit
    ``reasoning="manual override user_bound_scale=-8"``.
    """
    args = Namespace(scaling=None, user_bound_scale=-8)
    cfg = resolve_scaling_config(args)
    assert cfg.user_bound_scale == -8


def test_scaling_cli_off_resolves_to_off_even_with_env_full(monkeypatch):
    """Tightens CLI > env: ``--scaling=off`` beats env ``full`` too.

    Symmetric with ``test_scaling_cli_full_overrides_env_off`` — a
    half-implemented precedence chain (e.g. one that only honoured CLI
    when the env said off) would slip past the full→off test, so we
    cover the other direction explicitly.
    """
    monkeypatch.setenv("FLEXTOOL_SCALING", "full")
    args = Namespace(scaling="off", user_bound_scale=None)
    cfg = resolve_scaling_config(args)
    assert cfg.mode is ScalingMode.OFF


def test_invalid_scaling_value_raises():
    """A typo in ``args.scaling`` raises rather than silently falling back."""
    args = Namespace(scaling="banana", user_bound_scale=None)
    with pytest.raises(ValueError):
        resolve_scaling_config(args)


# ---------------------------------------------------------------------------
# Orchestration glue — mode-driven solver-option assembly.
#
# The cascade builds its baseline HiGHS option dict via
# ``_baseline_highs_options`` and threads the scaling mode in so
# ``--scaling=off`` is the only mode that touches HiGHS' internal
# matrix equilibration (forces ``simplex_scale_strategy=0``).  The
# other three modes leave HiGHS' Curtis-Reid equilibration enabled.
# ---------------------------------------------------------------------------

def test_baseline_options_off_forces_simplex_scale_zero():
    """``--scaling=off`` must force ``simplex_scale_strategy=0``."""
    from flextool.engine_polars._orchestration import _baseline_highs_options

    opts = _baseline_highs_options(scaling_mode=ScalingMode.OFF)
    assert opts["simplex_scale_strategy"] == 0


@pytest.mark.parametrize(
    "mode", [ScalingMode.SOLVER_ONLY, ScalingMode.BASIC, ScalingMode.FULL],
)
def test_baseline_options_non_off_keeps_simplex_scale_advanced(mode):
    """All modes other than OFF leave HiGHS' equilibration on (=2)."""
    from flextool.engine_polars._orchestration import _baseline_highs_options
    from flextool.engine_polars._determinism import (
        SIMPLEX_SCALE_STRATEGY_ADVANCED,
    )

    opts = _baseline_highs_options(scaling_mode=mode)
    assert opts["simplex_scale_strategy"] == SIMPLEX_SCALE_STRATEGY_ADVANCED


def test_baseline_options_default_mode_keeps_simplex_scale_advanced():
    """No-mode-supplied call keeps the determinism-pinned advanced value."""
    from flextool.engine_polars._orchestration import _baseline_highs_options
    from flextool.engine_polars._determinism import (
        SIMPLEX_SCALE_STRATEGY_ADVANCED,
    )

    opts = _baseline_highs_options()
    assert opts["simplex_scale_strategy"] == SIMPLEX_SCALE_STRATEGY_ADVANCED


# ---------------------------------------------------------------------------
# Layer 3 precedence-respect — caller-set user_bound_scale survives.
#
# The cascade hands ``recommend_scaling`` the polar-high ``Problem``
# instance so the layer can inspect already-set options and skip its
# own recommendation for any axis the caller has pinned (highs.opt /
# explicit ``set_solver_options`` / CLI manual override).
# ---------------------------------------------------------------------------

def test_layer3_precedence_respects_externally_set_user_bound_scale():
    """When the Problem already carries ``user_bound_scale``, Layer 3 keeps it."""
    import math

    from polar_high import Problem

    from flextool.engine_polars.autoscale import (
        RangeReport,
        ScalingConfig,
        recommend_scaling,
    )

    pb = Problem()
    pb.set_solver_options({"user_bound_scale": 5})
    # Bound spread that would normally trigger Layer 3's geometric-escape
    # branch (max_b ≥ 1e+9, naive clamp crushes min): the test must show
    # the precedence-check beats both the auto path and the escape branch.
    ranges = RangeReport(
        matrix=(1e-2, 1e2),
        cost=(1.0, 1e2),
        bound=(1e-3, 1e12),
        rhs=(1e-3, 1e12),
        cross_group_max_ratio=math.nan,
        trigger=True,
    )
    plan = recommend_scaling(
        ranges, ScalingConfig(mode=ScalingMode.BASIC), problem=pb,
    )
    assert plan.user_bound_scale == 5
    assert plan.bound_skipped_external is True
    assert "external user_bound_scale=5" in plan.reasoning


def test_layer3_precedence_respects_externally_set_user_objective_scale():
    """When the Problem carries ``user_objective_scale``, the cost axis is skipped."""
    import math

    from polar_high import Problem

    from flextool.engine_polars.autoscale import (
        RangeReport,
        ScalingConfig,
        recommend_scaling,
    )

    pb = Problem()
    pb.set_solver_options({"user_objective_scale": -3})
    ranges = RangeReport(
        matrix=(1e-2, 1e2),
        # Worst |c| of 1e+7 would normally yield N_obj = -10; precedence
        # must override that with the caller's -3.
        cost=(1.0, 1e7),
        bound=(1e-2, 1e3),
        rhs=(1e-2, 1e3),
        cross_group_max_ratio=math.nan,
        trigger=True,
    )
    plan = recommend_scaling(
        ranges, ScalingConfig(mode=ScalingMode.BASIC), problem=pb,
    )
    assert plan.user_objective_scale == -3
    assert plan.objective_skipped_external is True
    assert "external user_objective_scale=-3" in plan.reasoning
