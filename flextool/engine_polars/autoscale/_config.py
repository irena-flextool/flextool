"""Configuration for the FlexTool autoscaler.

Layer 1 only needs ``enabled`` + ``threshold_decades`` (for detection).
Layer 2 / Layer 3 will read ``user_bound_scale`` (manual override) and
``report_yaml_path`` (where to write the audit YAML).  The dataclass is
forward-declared in one place to keep the public surface stable as the
later layers land.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# Default trigger threshold: the H2_trade regression handoff identified a
# coefficient spread of ~1e9 as the operational pain point; anything past
# nine decades is the conservative "definitely scale" line.  This matches
# the rule the operator ran by hand in the handoff.
_DEFAULT_THRESHOLD_DECADES = 9.0


# Clamp range for the HiGHS ``user_bound_scale`` exponent (power of 2).  HiGHS
# itself accepts ``[-30, 30]``; we cap at ``[-10, 0]`` because (a) positive
# scales are nearly always harmful — they expand already-large coefficients,
# and (b) ``2**-10`` ≈ ``1e-3`` is plenty to neutralise the LP shapes FlexTool
# produces.  Centralised here so both the legacy DB-override coercion and
# Layer 3's autoscaler use the same bounds.
USER_BOUND_SCALE_MIN: int = -10
USER_BOUND_SCALE_MAX: int = 0


@dataclass(frozen=True)
class AutoScaleConfig:
    """Caller-facing autoscaler configuration.

    Parameters
    ----------
    enabled:
        Master switch.  When ``False``, the autoscaler short-circuits
        before any layer runs.  Default ``True``.
    threshold_decades:
        Layer 1 raises ``trigger=True`` when any single-group max/min
        ratio, or the cross-group max/min ratio, exceeds
        ``10 ** threshold_decades``.  Default 9.0.
    user_bound_scale:
        Manual override for the Layer 3 ``user_bound_scale`` HiGHS option.
        When set, Layer 3 must skip its own recommendation and apply this
        integer verbatim.  Default ``None`` (let Layer 3 decide).
    report_yaml_path:
        Where to write the autoscaler's audit YAML.  ``None`` disables
        the report.  Coordinated with the existing scaling-report file
        location so the operator finds both in one place.
    """

    enabled: bool = True
    threshold_decades: float = _DEFAULT_THRESHOLD_DECADES
    user_bound_scale: Optional[int] = None
    report_yaml_path: Optional[Path] = None


def _coerce_env_bool(raw: Optional[str], *, default: bool) -> bool:
    """Parse ``FLEXTOOL_AUTO_SCALE``-style env flags.

    Accepts ``on`` / ``off`` (case-insensitive) per the spec.  Empty or
    unrecognised values fall through to ``default`` so a misconfigured env
    doesn't silently flip behaviour.
    """
    if raw is None:
        return default
    v = raw.strip().lower()
    if v == "on":
        return True
    if v == "off":
        return False
    return default


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


def _coerce_cli_auto_scale(raw: Any) -> Optional[bool]:
    """Coerce an ``args.auto_scale`` value to ``bool | None``.

    Accepts ``"on"`` / ``"off"`` strings (the CLI ``choices=[...]``
    surface), already-typed ``bool``, and ``None`` (CLI unset →
    fall through to env / default).  Unrecognised values raise so a
    typo in caller code surfaces loudly rather than silently flipping
    the master switch.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        v = raw.strip().lower()
        if v == "on":
            return True
        if v == "off":
            return False
        if v == "":
            return None
        raise ValueError(
            f"auto_scale must be 'on', 'off', bool, or None; got {raw!r}"
        )
    raise ValueError(
        f"auto_scale must be 'on', 'off', bool, or None; got {raw!r}"
    )


def resolve_auto_scale_config(cli_args: Any = None) -> AutoScaleConfig:
    """Build an :class:`AutoScaleConfig` from CLI args + environment.

    Precedence for the master switch is **CLI > env > default-on**:

    * ``cli_args.auto_scale`` (``"on"`` / ``"off"`` / ``bool`` / ``None``)
      wins when set.  The ``cmd_run_flextool`` CLI defines this via
      ``--auto-scale`` and, when set, also exports
      ``FLEXTOOL_AUTO_SCALE`` so subprocess / cascade-internal call
      sites that resolve config with ``cli_args=None`` still observe
      the operator's choice.
    * ``FLEXTOOL_AUTO_SCALE`` env var (``on`` / ``off``) provides the
      fallback for callers that don't have access to the parsed
      ``args`` (e.g. cascade-internal helpers, tests that pre-set the
      env).
    * Default ``True`` (autoscaler enabled).

    ``user_bound_scale`` and the optional YAML report path follow the
    same CLI-wins-over-env layering:

    * ``cli_args.user_bound_scale`` (int / decimal str / ``None``) wins.
      When set, Layer 3 forces this exponent and surfaces
      ``reasoning="manual override user_bound_scale=N"`` in its plan
      (see :func:`flextool.engine_polars.autoscale.recommend_layer3`).
    * ``FLEXTOOL_USER_BOUND_SCALE`` env var (already plumbed by
      ``--user-bound-scale`` on the CLI surface) is the fallback.
    * ``cli_args.autoscale_report_yaml`` overrides the default YAML
      report path; otherwise ``_orchestration`` picks
      ``work_folder/solve_data/autoscale_<solve>.yaml``.

    Returns a frozen :class:`AutoScaleConfig`.
    """
    enabled_default = _coerce_env_bool(
        os.environ.get("FLEXTOOL_AUTO_SCALE"), default=True,
    )
    user_bound_scale_env = _coerce_user_bound_scale(
        os.environ.get("FLEXTOOL_USER_BOUND_SCALE"),
    )

    enabled = enabled_default
    user_bound_scale = user_bound_scale_env
    report_yaml_path: Optional[Path] = None

    if cli_args is not None:
        cli_enabled = _coerce_cli_auto_scale(
            getattr(cli_args, "auto_scale", None),
        )
        if cli_enabled is not None:
            enabled = cli_enabled
        cli_ubs = getattr(cli_args, "user_bound_scale", None)
        if cli_ubs is not None:
            user_bound_scale = _coerce_user_bound_scale(cli_ubs)
        cli_yaml = getattr(cli_args, "autoscale_report_yaml", None)
        if cli_yaml is not None:
            report_yaml_path = Path(cli_yaml)

    return AutoScaleConfig(
        enabled=enabled,
        threshold_decades=_DEFAULT_THRESHOLD_DECADES,
        user_bound_scale=user_bound_scale,
        report_yaml_path=report_yaml_path,
    )


def resolve_user_bound_scale_override(
    user_value: Any,
) -> Optional[int]:
    """Coerce a raw DB ``user_bound_scale`` value to a clamped int (or None).

    Accepts ``None``, integers, floats, or numeric strings and returns:

    * an integer in ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]`` when
      the user provided a non-zero, finite, parseable value;
    * ``None`` when the user provided no value, ``0``, or anything
      unparseable — caller should then fall back to the heuristic (Layer 3's
      automatic recommendation, or polar-high's stream-time auto-pick).

    Truncates non-integer floats toward zero (e.g. ``-3.7`` → ``-3``) rather
    than rounding, mirroring the conservative direction (smaller |N| is
    gentler scaling).

    This helper exists alongside :func:`_coerce_user_bound_scale` because the
    two have different contracts: ``_coerce_user_bound_scale`` is for
    CLI / env values where a non-numeric string is a user error worth
    surfacing as ``ValueError``; this helper is for legacy DB values where a
    malformed entry should fall back to the heuristic rather than crash a
    cascade.
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
    "AutoScaleConfig",
    "USER_BOUND_SCALE_MAX",
    "USER_BOUND_SCALE_MIN",
    "resolve_auto_scale_config",
    "resolve_user_bound_scale_override",
]
