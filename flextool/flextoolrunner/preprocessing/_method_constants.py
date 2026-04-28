"""Method-enum subsets — model invariants from flextool/flextool_base.dat.

These are NOT user-editable; they're invariants of the FlexTool method
taxonomy. Each constant below mirrors the corresponding ``set ... :=
... ;`` definition in flextool_base.dat:60-95. If those definitions
ever change, update both sites in lockstep.

Lifetime / ct / startup / co2 method constants are defined in their
own preprocessing modules to keep the lookup tight.
"""
from __future__ import annotations


# flextool_base.dat:86 — single-variable representations
METHOD_1VAR_PER_WAY: frozenset[str] = frozenset((
    "method_1way_1var_off",
    "method_1way_1var_LP",
    "method_1way_1var_MIP",
    "method_2way_2var_off",
    "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))

# flextool_base.dat:87 — LP variants of online (linear)
METHOD_LP: frozenset[str] = frozenset((
    "method_1way_1var_LP",
    "method_1way_nvar_LP",
))

# flextool_base.dat:88 — MIP variants (binary online)
METHOD_MIP: frozenset[str] = frozenset((
    "method_1way_1var_MIP",
    "method_1way_nvar_MIP",
    "method_2way_2var_MIP_exclude",
))

# flextool_base.dat:89-91 — direct (no efficiency conversion)
METHOD_DIRECT: frozenset[str] = frozenset((
    "method_1way_1var_off",
    "method_1way_1var_LP",
    "method_1way_1var_MIP",
    "method_2way_1var_off",
    "method_2way_2var_off",
    "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))

# flextool_base.dat:92-93 — indirect (efficiency-converting)
METHOD_INDIRECT: frozenset[str] = frozenset((
    "method_1way_nvar_off",
    "method_1way_nvar_LP",
    "method_1way_nvar_MIP",
    "method_2way_nvar_off",
))

# flextool_base.dat:81-82 — 2-way 2-variable (separate v_flow per direction)
METHOD_2WAY_2VAR: frozenset[str] = frozenset((
    "method_2way_2var_off",
    "method_2way_2var_exclude",
    "method_2way_2var_MIP_exclude",
))

# flextool_base.dat:83 — 2-way nvar (n flow variables)
METHOD_2WAY_NVAR: frozenset[str] = frozenset((
    "method_2way_nvar_off",
))

# flextool_base.dat:79 — 1-way 1-variable
METHOD_1WAY_1VAR: frozenset[str] = frozenset((
    "method_1way_1var_off",
    "method_1way_1var_LP",
    "method_1way_1var_MIP",
))
