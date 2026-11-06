"""FlexTool-side configuration for the autoscaler.

After Phase R2 the autoscaler's policy primitives (``ScalingMode`` enum,
``ScalingConfig`` dataclass, ``mode_enables_layer1`` / ``mode_enables_layer3``
predicates) live in :mod:`polar_high.autoscale._config`.  This module
keeps the FlexTool-specific resolver — ``resolve_scaling_config`` — which
honours ``--scaling`` / ``FLEXTOOL_SCALING`` (the run's options) over a solve's ``solve.scaling``.

Layer 2 (semantic per-quantity scaling) stays in FlexTool because it
needs FlexTool's parameter taxonomy.  Layer 2 fires only when
``mode == ScalingMode.FULL``.

The ``--user-bound-scale N`` override still lives here too: when
set, it propagates onto ``ScalingConfig.user_bound_scale`` so Layer 3's
manual-override branch (in polar-high) fires.
:func:`resolve_user_bound_scale_override` clamps the same value for the
cascade's ``_baseline_highs_options`` wire-in.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from polar_high.autoscale import (
    ScalingConfig,
    ScalingMode,
)

if TYPE_CHECKING:
    from flextool.run_options import RunOptions


# Clamp range for the HiGHS ``user_bound_scale`` exponent (power of 2).  HiGHS
# itself accepts ``[-30, 30]`` (see ``polar_high.autoscale.USER_SCALE_CLAMP_*``);
# we cap at ``[-10, 0]`` for the DB-override coercion path because (a) positive
# scales are nearly always harmful — they expand already-large coefficients,
# and (b) ``2**-10`` ≈ ``1e-3`` is plenty to neutralise the LP shapes FlexTool
# produces.  Centralised here so both the legacy DB-override coercion and
# the cascade-level ``user_bound_scale_override`` wire-in use the same bounds.
USER_BOUND_SCALE_MIN: int = -10
USER_BOUND_SCALE_MAX: int = 0


def _coerce_user_bound_scale(raw: Any) -> Optional[int]:
    """Coerce a ``--user-bound-scale`` value to ``int | None``.

    Accepts already-typed ``int`` / ``None``, as well as decimal strings
    (the form ``cli_args`` typically carries them).  Non-numeric strings
    raise ``ValueError`` — the CLI surface should reject them; surfacing
    the error here is louder than silently falling back to ``None``.
    """
    if raw is None:
        return None
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        return int(s)
    raise ValueError(
        f"user_bound_scale must be int, decimal str, or None; got {raw!r}"
    )


def _coerce_cli_scaling(raw: Any) -> Optional[ScalingMode]:
    """Coerce an ``args.scaling`` value to ``ScalingMode | None``.

    Accepts the four enum-value strings (``"off"`` / ``"solver_only"`` /
    ``"basic"`` / ``"full"``), an already-typed ``ScalingMode``, and
    ``None`` (CLI unset → fall through to env / default).  Unrecognised
    values raise so a typo surfaces loudly rather than silently picking
    a different policy.
    """
    if raw is None:
        return None
    if isinstance(raw, ScalingMode):
        return raw
    if isinstance(raw, str):
        v = raw.strip().lower()
        if not v:
            return None
        for m in ScalingMode:
            if v == m.value:
                return m
        raise ValueError(
            f"scaling must be one of {[m.value for m in ScalingMode]!r}, "
            f"ScalingMode, or None; got {raw!r}"
        )
    raise ValueError(
        f"scaling must be a string, ScalingMode, or None; got {raw!r}"
    )


def resolve_scaling_config(
    cli_args: Any = None,
    *,
    run_options: "RunOptions | None" = None,
    db_scaling: Optional[str] = None,
) -> ScalingConfig:
    """Build a :class:`ScalingConfig` from the run's options and a solve's database value.

    Precedence for the mode is **flag > environment variable > database > default-full**
    (:mod:`flextool.run_options`):

    * ``run_options`` carries the flag (``--scaling``) and the environment
      variable (``FLEXTOOL_SCALING``) tiers; ``None`` reads the environment
      (:meth:`RunOptions.from_env`), the form of a library caller or test.
    * ``db_scaling`` is the active solve's ``solve.scaling``; the cascade
      passes it per sub-solve.  ``None`` = the solve does not set it.
    * Default ``ScalingMode.FULL`` — the full autoscaler is on.
    * ``cli_args.scaling`` (``"off"`` / ``"solver_only"`` / ``"basic"`` /
      ``"full"`` / ``ScalingMode`` / ``None``), when given and set, wins over
      all of them.

    ``user_bound_scale`` comes from ``--user-bound-scale`` /
    ``FLEXTOOL_USER_BOUND_SCALE`` (``run_options``), overridden by a set
    ``cli_args.user_bound_scale``.  ``cli_args.autoscale_report_yaml``
    overrides the default YAML report path; otherwise ``_orchestration``
    picks ``work_folder/solve_data/autoscale_<solve>.yaml``.

    Returns a frozen :class:`ScalingConfig` (from
    :mod:`polar_high.autoscale`).
    """
    from flextool.run_options import RunOptions

    options = run_options if run_options is not None else RunOptions.from_env()
    mode = _coerce_cli_scaling(options.with_db("scaling", db_scaling).value) or ScalingMode.FULL
    user_bound_scale_env = _coerce_user_bound_scale(options.value("user_bound_scale"))

    user_bound_scale = user_bound_scale_env
    report_yaml_path: Optional[Path] = None

    if cli_args is not None:
        cli_mode = _coerce_cli_scaling(getattr(cli_args, "scaling", None))
        if cli_mode is not None:
            mode = cli_mode
        cli_ubs = getattr(cli_args, "user_bound_scale", None)
        if cli_ubs is not None:
            user_bound_scale = _coerce_user_bound_scale(cli_ubs)
        cli_yaml = getattr(cli_args, "autoscale_report_yaml", None)
        if cli_yaml is not None:
            report_yaml_path = Path(cli_yaml)

    return ScalingConfig(
        mode=mode,
        user_bound_scale=user_bound_scale,
        report_yaml_path=report_yaml_path,
    )


def resolve_user_bound_scale_override(
    user_value: Any,
) -> Optional[int]:
    """Coerce a ``user_bound_scale`` value to a clamped int (or None) for the HiGHS baseline options.

    Accepts ``None``, integers, floats, or numeric strings and returns:

    * an integer in ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]`` when
      the user provided a non-zero, finite, parseable value;
    * ``None`` when the user provided no value, ``0``, or anything
      unparseable — caller should then fall back to the heuristic (Layer 3's
      automatic recommendation).

    Truncates non-integer floats toward zero (e.g. ``-3.7`` → ``-3``) rather
    than rounding, mirroring the conservative direction (smaller |N| is
    gentler scaling).
    """
    if user_value is None:
        return None
    try:
        as_float = float(user_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    n = int(as_float)  # truncates toward zero
    if n == 0:
        return None
    if n > USER_BOUND_SCALE_MAX:
        n = USER_BOUND_SCALE_MAX
    if n < USER_BOUND_SCALE_MIN:
        n = USER_BOUND_SCALE_MIN
    return n


__all__ = [
    "ScalingConfig",
    "ScalingMode",
    "USER_BOUND_SCALE_MAX",
    "USER_BOUND_SCALE_MIN",
    "resolve_scaling_config",
    "resolve_user_bound_scale_override",
]
