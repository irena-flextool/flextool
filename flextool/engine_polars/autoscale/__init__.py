"""FlexTool LP autoscaler — public API surface.

After Phase R2 the four-range Layer 1 detector and the HiGHS-native
Layer 3 recommendation live in :mod:`polar_high.autoscale`.  FlexTool
re-exports them here so the rest of the engine_polars code (and any
external caller) sees a single stable surface.  Layer 2 (semantic
per-quantity scaling, gated on ``ScalingMode.FULL``) stays FlexTool-side
because it depends on FlexTool's parameter taxonomy.

Three layers, run around the polar-high ``Problem``:

* **Layer 1 (detect, polar-high):** read the four standard LP
  coefficient ranges (Matrix / Cost / Bound / RHS) and decide whether
  the model needs scaling at all.  Detection only — no LP modification.
* **Layer 2 (semantic per-type scaling, FlexTool):** apply column
  scalers chosen by ``QuantityType`` so each physical quantity (power,
  energy, price, …) lives on its own well-conditioned magnitude.  Fires
  only when ``mode == ScalingMode.FULL``.
* **Layer 3 (HiGHS-native + bound-scale escape, polar-high):** set
  ``user_bound_scale`` / ``user_objective_scale`` / ``simplex_scale_strategy``
  to neutralise residual range pressure inside HiGHS.  Honours
  precedence so a caller-set option survives.
"""
from polar_high.autoscale import (
    USER_SCALE_CLAMP_HI,
    USER_SCALE_CLAMP_LO,
    Layer3Plan,
    RangeReport,
    ScalingConfig,
    ScalingMode,
    apply_scaling,
    detect_ranges,
    get_explicit_option,
    has_explicit_option,
    mode_enables_layer1,
    mode_enables_layer3,
    ranges_from_arrays,
    ranges_from_streamed,
    recommend_scaling,
)

from ._config import (
    USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN,
    resolve_scaling_config,
    resolve_user_bound_scale_override,
)
from ._layer2 import (
    Layer2Plan,
    apply_layer2,
    bucket_coefficients,
    choose_scale_powers,
    unscale_solution,
)
from ._layer2_types import (
    CONSTRAINT_FAMILIES,
    CstrFamily,
    VARIABLE_FAMILIES,
    VarFamily,
    lookup_cstr,
    lookup_var,
    resolve_cstr_rhs_type,
)
from ._quantity_types import QuantityType, lookup, resolve_group_capacity_type
from ._report import format_console_summary, format_nonoptimal_hint, write_report

__all__ = [
    "CONSTRAINT_FAMILIES",
    "CstrFamily",
    "Layer2Plan",
    "Layer3Plan",
    "QuantityType",
    "RangeReport",
    "ScalingConfig",
    "ScalingMode",
    "USER_BOUND_SCALE_MAX",
    "USER_BOUND_SCALE_MIN",
    "USER_SCALE_CLAMP_HI",
    "USER_SCALE_CLAMP_LO",
    "VARIABLE_FAMILIES",
    "VarFamily",
    "apply_layer2",
    "apply_scaling",
    "bucket_coefficients",
    "choose_scale_powers",
    "detect_ranges",
    "format_console_summary",
    "format_nonoptimal_hint",
    "get_explicit_option",
    "has_explicit_option",
    "lookup",
    "lookup_cstr",
    "lookup_var",
    "mode_enables_layer1",
    "mode_enables_layer3",
    "ranges_from_arrays",
    "ranges_from_streamed",
    "recommend_scaling",
    "resolve_cstr_rhs_type",
    "resolve_group_capacity_type",
    "resolve_scaling_config",
    "resolve_user_bound_scale_override",
    "unscale_solution",
    "write_report",
]
