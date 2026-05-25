"""Layer 3 (HiGHS-native global scaling) — implementation.

Layer 3 runs *after* Layer 2 has rewritten the LP arrays in place and
*before* ``Problem.solve(...)``.  It sets three HiGHS options that
let the solver apply its own (power-of-two-exact, internally unscaled
on output) global magnitude shifts on top of the per-quantity rescale
that Layer 2 already did:

* ``user_objective_scale`` — exponent ``N_obj`` such that HiGHS sees
  cost coefficients multiplied by ``2 ** N_obj``.  Pulls the worst-cost
  magnitude into HiGHS' comfort zone ``|c| <= 1e+4``.
* ``user_bound_scale`` — exponent ``N_bnd`` such that HiGHS sees
  variable bounds AND row bounds (RHS) multiplied by ``2 ** N_bnd``.
  Pulls the worst-bound magnitude into HiGHS' comfort zone
  ``1e-4 <= |b| <= 1e+6``.
* ``simplex_scale_strategy`` — HiGHS' matrix equilibration knob; we
  pin it to ``2`` (ADVANCED equilibration) so the constraint-matrix
  spread that neither ``user_*_scale`` touches gets standard Curtis-Reid
  equilibration treatment.

Phase 0a established that ``highspy.Highs.getObjectiveBoundScaling``
is not exposed in the FlexTool-pinned HiGHS build, so the
recommendation is reproduced in Python from the post-Layer-2 range
report (the same arithmetic HiGHS uses internally to decide what to
print in its "Consider scaling …" warning).

Layer 3 is invisible to HiGHS' MPS writer (``writeModel`` exports the
*unscaled* problem — ``user_*_scale`` is a HiGHS-internal scale, not a
mutation of the LP arrays), so handing off the model to an external
solver continues to see the post-Layer-2 LP without an extra step.

The two-sided "geometric-centering escape" branch (carried over from
polar-high's ``_recommend_user_bound_scale``) only fires when the
worst-bound magnitude is severely above HiGHS' ceiling
(``>= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT`` ≈ 1e+9) AND the
naive recommendation would drive the small end below the floor.  On
H2_trade with Layer 2 in place the post-Layer-2 ``max(|b|)`` lands well
inside HiGHS' zone, so the escape branch is a safety net only — but we
keep it so a model that mis-buckets in Layer 2 still produces a sane
recommendation rather than a single-sided crush.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from ._config import AutoScaleConfig
from ._ranges import RangeReport


_logger = logging.getLogger(__name__)


# HiGHS' comfort zones (mirrored from polar-high's port of
# ``HighsSolve.cpp::suggestScaling``):
#
# * Bounds + RHS: ``[1e-4, 1e+6]``.
# * Cost: ``|c| <= 1e+4`` (HiGHS prints "Consider scaling the objective"
#   when the worst |c| exceeds 1e+4).
#
# Power-of-two exponents are computed to pull the worst-magnitude end
# into the comfort zone in a single shot.  We use the *same* formula
# HiGHS itself does (outer-rounded log2 → smallest-|N| value, i.e. the
# more conservative scaling), so Layer 3's output matches HiGHS' own
# recommendation byte-for-byte on simple cases.
_HIGHS_LARGE_BOUND = 1e6
_HIGHS_SMALL_BOUND = 1e-4
_HIGHS_LARGE_COST = 1e4

# Severe-overshoot trigger for the two-sided escape (D's branch).
# When ``max_b >= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT`` (~1e+9)
# the post-Layer-2 LP is so skewed that one-sided clamping crushes the
# small end below the HiGHS warning threshold.  Geometric-centering
# distributes the (unavoidable) violation symmetrically.
_SEVERE_LARGE_OVERSHOOT = 1e3

# HiGHS rejects ``user_*_scale`` outside ``[-30, 30]``.  Clamp defensively.
_USER_SCALE_CLAMP_LO = -30
_USER_SCALE_CLAMP_HI = 30

# HiGHS ``simplex_scale_strategy`` value: 2 = ADVANCED equilibration
# (Curtis–Reid).  Matches the existing
# ``flextool.engine_polars.scaling.SIMPLEX_SCALE_STRATEGY_ADVANCED``
# constant.
_SIMPLEX_SCALE_STRATEGY_DEFAULT = 2


@dataclass(frozen=True)
class Layer3Plan:
    """HiGHS-native scaling decisions for one solve.

    Attributes
    ----------
    user_objective_scale:
        Power-of-two exponent applied via the HiGHS option
        ``user_objective_scale`` (``cost coefficients × 2**N``).  ``0``
        means "no scaling needed" — within HiGHS' comfort zone already.
    user_bound_scale:
        Power-of-two exponent applied via ``user_bound_scale``
        (variable bounds AND row bounds × ``2**N``).  ``0`` is a no-op.
    simplex_scale_strategy:
        HiGHS' equilibration strategy (0..5).  Default ``2`` (ADVANCED).
    reasoning:
        One-line free-text string explaining where the values came
        from (auto / manual override / escape).  Surfaces in the YAML
        audit and the orchestration log.
    """

    user_objective_scale: int
    user_bound_scale: int
    simplex_scale_strategy: int
    reasoning: str


def _clamp(n: int) -> int:
    if n < _USER_SCALE_CLAMP_LO:
        return _USER_SCALE_CLAMP_LO
    if n > _USER_SCALE_CLAMP_HI:
        return _USER_SCALE_CLAMP_HI
    return n


def _recommend_objective_scale(cost_max: float) -> int:
    """Power-of-two exponent that pulls ``max(|c|)`` into ``|c| <= 1e+4``.

    Mirrors HiGHS' ``suggestScaling`` lambda for the objective: when
    ``cost_max <= _HIGHS_LARGE_COST`` we return 0 (no scaling); when
    ``cost_max > _HIGHS_LARGE_COST`` we return ``floor(log2(ratio))``
    so the resulting scaled max lies in ``[_HIGHS_LARGE_COST / 2,
    _HIGHS_LARGE_COST]``.  Outer rounding (``floor`` for ratio < 1)
    picks the smaller-|N| value — same conservative rule HiGHS uses.

    Cost vectors don't have a "min too small" branch in HiGHS'
    recommendation (HiGHS doesn't warn about excessively small costs
    the way it does for bounds), so we don't add one either.
    """
    if not math.isfinite(cost_max) or cost_max <= 0.0:
        return 0
    if cost_max <= _HIGHS_LARGE_COST:
        return 0
    ratio = _HIGHS_LARGE_COST / cost_max
    # ratio < 1 here by construction.
    return int(math.floor(math.log2(ratio)))


def _recommend_bound_scale(
    bound_max: float,
    bound_min: float,
    rhs_max: float,
    rhs_min: float,
) -> tuple[int, str]:
    """Power-of-two exponent that pulls ``max(|b|)`` into the HiGHS
    comfort zone, with D's geometric-centering escape on severe
    overshoots.

    Inputs are the four post-Layer-2 magnitudes (``±inf`` / ``nan`` →
    treated as 0 for the max comparison, ``+inf`` for the min).  Returns
    ``(n, reasoning_tag)`` where ``reasoning_tag`` is one of
    ``"in-zone"``, ``"clamp-large"``, ``"clamp-small"``, ``"escape"``,
    or ``"refuse"`` for the operator log.
    """
    max_b = max(_safe_float(bound_max, 0.0), _safe_float(rhs_max, 0.0))
    min_b = min(_safe_float(bound_min, math.inf), _safe_float(rhs_min, math.inf))

    if not math.isfinite(max_b) or max_b <= 0.0:
        return 0, "in-zone"

    if max_b > _HIGHS_LARGE_BOUND:
        ratio = _HIGHS_LARGE_BOUND / max_b
        tag = "clamp-large"
    elif max_b < _HIGHS_SMALL_BOUND:
        ratio = _HIGHS_SMALL_BOUND / max_b
        tag = "clamp-small"
    else:
        return 0, "in-zone"

    # Outer-rounded log2: floor when ratio<1 (scaling down), ceil when
    # ratio>1 (scaling up).  Same conservative rule HiGHS uses.
    if ratio < 1.0:
        dl = math.floor(math.log2(ratio))
    else:
        dl = math.ceil(math.log2(ratio))

    # Min-floor guard: when scaling down would drive the small end
    # below ``_HIGHS_SMALL_BOUND``, decide between refusing (the
    # Rivendell-safe behaviour) and the geometric-centering escape
    # (the H2_trade-safe behaviour).
    if dl < 0 and math.isfinite(min_b) and min_b > 0.0:
        scaled_min = min_b * (2.0 ** dl)
        if scaled_min < _HIGHS_SMALL_BOUND:
            if max_b >= _HIGHS_LARGE_BOUND * _SEVERE_LARGE_OVERSHOOT:
                # D's escape: geometric centering over the comfort zone.
                geo_range = math.sqrt(min_b * max_b)
                geo_band = math.sqrt(_HIGHS_SMALL_BOUND * _HIGHS_LARGE_BOUND)
                if (
                    math.isfinite(geo_range)
                    and geo_range > 0.0
                    and math.isfinite(geo_band)
                    and geo_band > 0.0
                ):
                    dl = int(round(math.log2(geo_band / geo_range)))
                    tag = "escape"
                else:
                    return 0, "refuse"
            else:
                # Moderate overshoot — Rivendell-shaped LPs land here
                # post-Layer-2 because their col bounds are tightly
                # clustered.  Refusing keeps N=0 and lets HiGHS' own
                # default scaling handle the row spread.
                return 0, "refuse"

    return int(dl), tag


def _safe_float(x: float, fallback: float) -> float:
    if x is None:
        return fallback
    if isinstance(x, float) and math.isnan(x):
        return fallback
    return float(x)


def recommend_layer3(
    post_layer2_ranges: RangeReport,
    config: AutoScaleConfig,
) -> Layer3Plan:
    """Compute the three HiGHS-native scaling values for one solve.

    Pulls ``max(|c|)`` from ``post_layer2_ranges.cost`` and
    ``max(|b|)``, ``min(|b|)`` from the union of
    ``post_layer2_ranges.bound`` and ``post_layer2_ranges.rhs``.

    Honors three branches in priority order:

    1. **Manual override** — when ``config.user_bound_scale`` is set,
       Layer 3 forces ``user_bound_scale = config.user_bound_scale``
       and disables the auto-recommendation for bounds.  The objective
       and simplex strategy still auto-recommend.
    2. **Severe-overshoot escape** — geometric-centering branch fires
       when the naive recommendation would crush the small end and
       ``max_b >= 1e+9``.  Carried forward from polar-high's
       ``_recommend_user_bound_scale``.
    3. **Default auto** — power-of-two exponents that pull the worst
       end into the HiGHS comfort zone.

    The resulting exponents are clamped to ``[-30, 30]`` (HiGHS' option
    bounds).
    """
    cost_lo, cost_hi = post_layer2_ranges.cost
    bound_lo, bound_hi = post_layer2_ranges.bound
    rhs_lo, rhs_hi = post_layer2_ranges.rhs

    n_obj = _clamp(_recommend_objective_scale(_safe_float(cost_hi, 0.0)))

    if config.user_bound_scale is not None:
        n_bnd = _clamp(int(config.user_bound_scale))
        reasoning = (
            f"manual override user_bound_scale={config.user_bound_scale}; "
            f"objective auto N_obj={n_obj}"
        )
    else:
        n_bnd_raw, bnd_tag = _recommend_bound_scale(
            bound_max=_safe_float(bound_hi, 0.0),
            bound_min=_safe_float(bound_lo, math.inf),
            rhs_max=_safe_float(rhs_hi, 0.0),
            rhs_min=_safe_float(rhs_lo, math.inf),
        )
        n_bnd = _clamp(n_bnd_raw)
        if n_obj == 0 and n_bnd == 0:
            reasoning = (
                f"in-zone (objective={bnd_tag if n_obj == 0 else 'large'}, "
                f"bound={bnd_tag})"
            )
        else:
            reasoning = (
                f"auto (N_obj={n_obj}, N_bnd={n_bnd}, bound_tag={bnd_tag})"
            )

    return Layer3Plan(
        user_objective_scale=n_obj,
        user_bound_scale=n_bnd,
        simplex_scale_strategy=_SIMPLEX_SCALE_STRATEGY_DEFAULT,
        reasoning=reasoning,
    )


def apply_layer3(problem: Any, plan: Layer3Plan) -> None:
    """Set the three HiGHS options on ``problem``.

    Uses polar-high's :meth:`Problem.set_solver_options` so the values
    survive both warm-LP and cold-LP code paths.  Existing options are
    preserved — we *merge* with whatever the caller already set on the
    Problem (this matters for the determinism pins).

    Setting ``user_bound_scale`` here also makes polar-high's
    ``auto_user_bound_scale=True`` path a no-op: that branch's gate is
    "caller has NOT set ``user_bound_scale``" (see
    ``polar_high.engine.Problem._solve_streaming``), so Layer 3's
    explicit assignment takes precedence cleanly.

    Skipping option names whose corresponding plan value is 0 keeps
    HiGHS' own defaults visible in logs; explicit 0 also sets fine
    (HiGHS treats it as a no-op), but emitting fewer options keeps the
    audit trail clean.
    """
    # Merge with whatever the caller (typically
    # ``recommended_highs_options``) already set.
    existing = dict(getattr(problem, "_solver_options", None) or {})
    if plan.user_objective_scale != 0:
        existing["user_objective_scale"] = int(plan.user_objective_scale)
    if plan.user_bound_scale != 0:
        existing["user_bound_scale"] = int(plan.user_bound_scale)
    # Always set the simplex strategy — the caller's existing options
    # already pin this to 2, but Layer 3 owns the value going forward
    # so the plan is the single source of truth for the report.
    existing["simplex_scale_strategy"] = int(plan.simplex_scale_strategy)
    problem.set_solver_options(existing)


__all__ = [
    "Layer3Plan",
    "apply_layer3",
    "recommend_layer3",
]
