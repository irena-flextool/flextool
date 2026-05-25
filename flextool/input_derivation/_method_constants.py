"""Method-enum subsets — invariants of the FlexTool method taxonomy.

These are NOT user-editable; they're invariants of the FlexTool method
taxonomy. Lifetime / ct / startup / co2 method constants are defined in
their own preprocessing modules to keep the lookup tight.
"""
from __future__ import annotations


# LP variants of online (linear)
METHOD_LP: frozenset[str] = frozenset((
    "method_1way_1var_LP",
    "method_1way_nvar_LP",
))

# MIP variants (binary online)
METHOD_MIP: frozenset[str] = frozenset((
    "method_1way_1var_MIP",
    "method_1way_nvar_MIP",
    "method_2way_2var_MIP_exclude",
))

# direct (no efficiency conversion)
METHOD_DIRECT: frozenset[str] = frozenset((
    "method_1way_1var_off",
    "method_1way_1var_LP",
    "method_1way_1var_MIP",
    "method_2way_1var_off",
    "method_2way_2var_off",
    "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))

# indirect (efficiency-converting)
METHOD_INDIRECT: frozenset[str] = frozenset((
    "method_1way_nvar_off",
    "method_1way_nvar_LP",
    "method_1way_nvar_MIP",
    "method_2way_nvar_off",
))

# 2-way 1-variable (one v_flow shared by both directions)
METHOD_2WAY_1VAR: frozenset[str] = frozenset((
    "method_2way_1var_off",
))

# 2-way 2-variable (separate v_flow per direction)
METHOD_2WAY_2VAR: frozenset[str] = frozenset((
    "method_2way_2var_off",
    "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))

# 2-way nvar (n flow variables)
METHOD_2WAY_NVAR: frozenset[str] = frozenset((
    "method_2way_nvar_off",
))

# 1-way 1-variable
METHOD_1WAY_1VAR: frozenset[str] = frozenset((
    "method_1way_1var_off",
    "method_1way_1var_LP",
    "method_1way_1var_MIP",
))

# all 1-way methods (1var + nvar variants)
METHOD_1WAY: frozenset[str] = frozenset((
    "method_1way_1var_off", "method_1way_1var_LP", "method_1way_1var_MIP",
    "method_1way_nvar_off", "method_1way_nvar_LP", "method_1way_nvar_MIP",
))

# universe of ramp methods
RAMP_METHOD: frozenset[str] = frozenset(("ramp_limit", "ramp_cost", "both"))

RAMP_LIMIT_METHOD: frozenset[str] = frozenset(("ramp_limit", "both"))

RAMP_COST_METHOD: frozenset[str] = frozenset(("ramp_cost", "both"))
