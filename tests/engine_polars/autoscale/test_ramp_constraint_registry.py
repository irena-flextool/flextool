"""Regression: the ramp constraint family resolves through Layer-2
``lookup_cstr`` via prefix dispatch.

The ramp constraints are emitted with dynamic f-string names
(``model.py:2328`` — ``f"ramp_{side}_{dir_}_constraint"``), so the
static-literal grep in :mod:`test_registry_coverage` never sees them.
Their Layer-2 registration is the single ``"ramp"`` entry in
``CONSTRAINT_FAMILIES``, which only resolves through the documented
*prefix dispatch* in :func:`lookup_cstr` (``name.startswith(key + "_")``).

A real Cyprus solve (``y2024_2050_5week``) surfaced the gap: prefix
dispatch was documented in the module header and in
``test_registry_coverage`` but never implemented in ``lookup_cstr``, so
``ramp_sink_up_constraint`` raised ``KeyError`` and autoscale Layer 2
silently degraded the whole solve to an un-scaled LP.  This test pins
the contract so the dispatch can't regress and so a rename of the ramp
constraints can't drift away from the registry unnoticed.
"""
from __future__ import annotations

from flextool.engine_polars.autoscale._layer2_types import (
    lookup_cstr,
    resolve_cstr_rhs_type,
)
from flextool.engine_polars.autoscale._quantity_types import QuantityType


# The exact names model.py:2327-2333 emits — (side, dir_) over the
# {sink, source} × {up, down} grid.
RAMP_CONSTRAINT_NAMES = [
    f"ramp_{side}_{dir_}_constraint"
    for side in ("sink", "source")
    for dir_ in ("up", "down")
]


def test_ramp_constraint_names_resolve_to_dimensionless() -> None:
    """Every ramp constraint resolves via prefix dispatch to the
    DIMENSIONLESS family (its row coefficient, not the RHS, carries the
    physical units)."""
    for name in RAMP_CONSTRAINT_NAMES:
        fam = lookup_cstr(name)
        assert fam.rhs_type is QuantityType.DIMENSIONLESS, (
            f"{name!r} resolved to {fam!r}, expected DIMENSIONLESS"
        )
        assert fam.member_class_resolver is None, (
            f"{name!r} unexpectedly carries a member_class_resolver"
        )
        # resolve_cstr_rhs_type is the path Layer 2 actually calls.
        assert resolve_cstr_rhs_type(name) is QuantityType.DIMENSIONLESS


def test_prefix_dispatch_longest_match_wins() -> None:
    """A longer registry key takes precedence over a shorter prefix, so
    a specific registration is never shadowed by a generic one."""
    # maxToSink_online is a more specific registration than maxToSink;
    # the suffix-strip path resolves the _linear variant to it (not to
    # the shorter maxToSink prefix).
    assert lookup_cstr("maxToSink_online_linear") is lookup_cstr(
        "maxToSink_online"
    )


def test_unregistered_constraint_still_raises() -> None:
    """Prefix dispatch must not turn an unknown name into a silent
    match — Layer 2 refuses to default (feedback_no_shortcuts)."""
    import pytest

    with pytest.raises(KeyError):
        lookup_cstr("totally_unregistered_constraint_xyz")
