"""FlexTool LP autoscaler — public API surface.

The autoscaler runs three layers around the polar-high ``Problem``:

* **Layer 1 (detect):** read the four standard LP coefficient ranges
  (Matrix / Cost / Bound / RHS) and decide whether the model needs
  scaling at all.  Detection only — no LP modification.
* **Layer 2 (semantic per-type scaling):** apply column scalers chosen
  by ``QuantityType`` so each physical quantity (power, energy, price,
  …) lives on its own well-conditioned magnitude.  Future phase.
* **Layer 3 (HiGHS native + bound-scale escape):** set HiGHS'
  ``user_bound_scale`` to neutralise residual range pressure on
  variable and row bounds.  Future phase.

This module exposes only the names the rest of FlexTool needs to call.
The leading-underscore submodules hold the implementation; their public
names are re-exported here.
"""
from ._config import (
    AutoScaleConfig,
    USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN,
    resolve_auto_scale_config,
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
from ._layer3 import Layer3Plan, apply_layer3, recommend_layer3
from ._quantity_types import QuantityType, lookup, resolve_group_capacity_type
from ._ranges import (
    RangeReport,
    compute_ranges,
    ranges_from_arrays,
    ranges_from_streamed,
)
from ._report import format_console_summary, format_nonoptimal_hint, write_report

__all__ = [
    "AutoScaleConfig",
    "CONSTRAINT_FAMILIES",
    "CstrFamily",
    "Layer2Plan",
    "Layer3Plan",
    "QuantityType",
    "RangeReport",
    "USER_BOUND_SCALE_MAX",
    "USER_BOUND_SCALE_MIN",
    "VARIABLE_FAMILIES",
    "VarFamily",
    "apply_layer2",
    "apply_layer3",
    "bucket_coefficients",
    "choose_scale_powers",
    "compute_ranges",
    "format_console_summary",
    "format_nonoptimal_hint",
    "lookup",
    "lookup_cstr",
    "lookup_var",
    "ranges_from_arrays",
    "ranges_from_streamed",
    "recommend_layer3",
    "resolve_auto_scale_config",
    "resolve_cstr_rhs_type",
    "resolve_group_capacity_type",
    "resolve_user_bound_scale_override",
    "unscale_solution",
    "write_report",
]
