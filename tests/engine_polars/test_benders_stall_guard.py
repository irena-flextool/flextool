"""Benders stall-guard domain wrapper: worst-offender selection + the assembled
stall diagnostic's FlexTool vocabulary.

The generic tail-off detector (``polar_high.StallMonitor``) is unit-tested in
polar-high; here we pin the FlexTool-side pieces the plan calls for:

* ``_stall_worst_offenders`` — pure selection of the ROOT (dominant stand-alone
  cost) and SYMPTOM (worst penalty-flow) node groups from two ``{region: cost}``
  maps, à la ``test_benders_cut_tolerance``'s pure-function style, including the
  tie and zero-autarky edge cases.
* the assembled three-section stall message stays in FlexTool class vocabulary
  (the ``_FORBIDDEN`` regex — no trade/pipe/line/region literals) while
  interpolating the runtime node-group names.
* the env resolver ``FLEXTOOL_BENDERS_MAX_STALL`` mirrors the workers knob.
"""

from __future__ import annotations

import re

from flextool.engine_polars._benders import (
    _benders_failure_message,
    _resolve_benders_max_stall,
    _stall_worst_offenders,
    _STALL_WINDOW_DEFAULT,
)

# Same model-instance vocabulary ban as test_benders_failure_messages.
_FORBIDDEN = re.compile(r"\b(trade|pipe|pipeline|line|region|regions)\b", re.I)


# ---------------------------------------------------------------------------
# _stall_worst_offenders — pure selection
# ---------------------------------------------------------------------------
def test_root_is_dominant_autarky_and_symptom_is_worst_ratio():
    """KOR-like case: one node group dominates stand-alone cost (root) and is
    also the worst penalty-flow ratio (symptom) — they coincide."""
    autarky = {"decomp_A": 5.0e5, "decomp_B": 8.0e6, "decomp_KOR": 8.62e9}
    # At the stalled iter KOR is forced worst into penalty flow (2000x its own
    # stand-alone cost vs ~1000x for the others).
    region_costs = {
        "decomp_A": 5.0e8,   # 1000x
        "decomp_B": 8.0e9,   # 1000x
        "decomp_KOR": 1.72e13,  # ~2000x
    }
    root, autarky_ratio, symptom, symptom_ratio = _stall_worst_offenders(
        autarky, region_costs
    )
    assert root == "decomp_KOR"
    # 8.62e9 / 8.0e6 ≈ 1077.
    assert 1000 < autarky_ratio < 1100
    assert symptom == "decomp_KOR"
    assert symptom_ratio > 1900  # ~2000x its stand-alone cost


def test_root_and_symptom_can_differ():
    """Root (dominant stand-alone) and symptom (worst current ratio) need not
    coincide — the message names both when they diverge."""
    autarky = {"decomp_X": 1.0e10, "decomp_Y": 1.0e6}
    # X carries a modest recourse; Y is forced deep into penalty flow.
    region_costs = {"decomp_X": 2.0e10, "decomp_Y": 5.0e9}
    root, _, symptom, symptom_ratio = _stall_worst_offenders(autarky, region_costs)
    assert root == "decomp_X"  # dominant stand-alone
    assert symptom == "decomp_Y"  # 5e9 / 1e6 = 5000x >> X's 2x
    assert symptom_ratio > 4000


def test_tie_breaks_on_name_deterministically():
    """Equal autarky magnitudes break on name (sorted) so the pick is stable."""
    autarky = {"decomp_B": 1.0e9, "decomp_A": 1.0e9}
    region_costs = {"decomp_B": 1.0e9, "decomp_A": 1.0e9}
    root, autarky_ratio, symptom, _ = _stall_worst_offenders(autarky, region_costs)
    assert root == "decomp_A"  # name-sorted tie-break
    # Ratio uses the next-largest (equal) autarky ⇒ 1e9 / 1e9 = 1.0.
    assert abs(autarky_ratio - 1.0) < 1e-9
    assert symptom == "decomp_A"


def test_zero_autarky_region_does_not_divide_by_zero():
    """A near-zero-autarky node group forced into penalty flow must not blow up
    the ratio (the max(1, ·) denominator guard)."""
    autarky = {"decomp_Z": 0.0, "decomp_W": 5.0e6}
    region_costs = {"decomp_Z": 3.0e9, "decomp_W": 1.0e7}
    root, autarky_ratio, symptom, symptom_ratio = _stall_worst_offenders(
        autarky, region_costs
    )
    # W dominates stand-alone (Z is zero); ratio uses second-largest (0) ⇒
    # 5e6 / max(1,0) = 5e6.
    assert root == "decomp_W"
    assert autarky_ratio == 5.0e6
    # Z's symptom ratio is finite (3e9 / max(1,0) = 3e9), not inf/NaN.
    assert symptom == "decomp_Z"
    assert symptom_ratio == 3.0e9


def test_single_region_degrades_gracefully():
    autarky = {"decomp_only": 4.0e6}
    root, autarky_ratio, symptom, symptom_ratio = _stall_worst_offenders(
        autarky, {"decomp_only": 4.0e9}
    )
    assert root == "decomp_only"
    assert autarky_ratio == 4.0e6  # / max(1, second=0)
    assert symptom == "decomp_only"


def test_empty_inputs_degrade_gracefully():
    assert _stall_worst_offenders({}, {}) == ("", 1.0, "", 1.0)


# ---------------------------------------------------------------------------
# Assembled diagnostic — FlexTool vocabulary
# ---------------------------------------------------------------------------
def _assemble_stall_message(root, autarky_ratio, symptom, symptom_ratio, *,
                            iterations, k, gap, tol):
    """Reproduce the loop's stall-message assembly (kept in lock-step with the
    call site in ``_solve_benders_inner``) so we can vocab-check the runtime
    text without driving a full solve."""
    root_clause = (
        f"Node group {root!r} is the likely cause — its stand-alone "
        f"cost is already {autarky_ratio:.0f}x the next largest, i.e. it "
        f"cannot meet its own demand without imports."
    )
    symptom_clause = (
        ""
        if symptom == root
        else (
            f" At the stalled iteration node group {symptom!r} is the "
            f"one forced worst into penalty/slack flow "
            f"({symptom_ratio:.0f}x its stand-alone cost)."
        )
    )
    return _benders_failure_message(
        summary=(
            f"Benders stalled at iteration {iterations}: the best "
            f"feasible cost has not improved for {k} iterations and the "
            f"relative gap is stuck at ~{gap:.2f}, far from the {tol} "
            "tolerance."
        ),
        meaning=(
            "The master keeps proposing node-group coupling flows that "
            "force one or more node groups into large penalty/slack "
            f"flow (recourse ~{symptom_ratio:.0f}x their stand-alone "
            "cost), so no improving feasible solution is found and the "
            f"bound cannot close. {root_clause}{symptom_clause}"
        ),
        how_to_avoid=(
            f"First, give the import/boundary nodes of {root!r} a "
            "finite, moderate import price (penalty) a small multiple "
            "above the real marginal supply cost — an over-large penalty "
            "is what inflates the recourse and freezes the bound (any "
            "price above the true import cost gives the same optimum). "
            f"Then check {root!r} in isolation for missing local "
            "capacity or imports, and rescale any extreme "
            "coupling-connection cost or capacity magnitudes. Only raise "
            "the iteration limit if the gap is still slowly improving "
            "(it is not here). If it persists, please report it with the "
            "model."
        ),
    )


def test_stall_message_uses_flextool_vocabulary():
    """The assembled stall diagnostic (both root==symptom and root!=symptom
    shapes) must stay in FlexTool class vocabulary."""
    # Coinciding root/symptom.
    msg1 = _assemble_stall_message(
        "decomp_KOR", 1077.0, "decomp_KOR", 1000.0,
        iterations=8, k=8, gap=1.02, tol=1e-4,
    )
    # Diverging root/symptom (exercises the extra symptom clause).
    msg2 = _assemble_stall_message(
        "decomp_X", 10.0, "decomp_Y", 5000.0,
        iterations=8, k=8, gap=1.02, tol=1e-4,
    )
    for msg in (msg1, msg2):
        assert "What this means:" in msg
        assert "How to avoid it:" in msg
        assert "node group" in msg
        offenders = sorted({m.group(0).lower() for m in _FORBIDDEN.finditer(msg)})
        assert not offenders, f"model-instance vocab leaked: {offenders}\n{msg}"


def test_stall_message_names_both_when_root_and_symptom_differ():
    msg = _assemble_stall_message(
        "decomp_X", 10.0, "decomp_Y", 5000.0,
        iterations=8, k=8, gap=1.02, tol=1e-4,
    )
    assert "'decomp_X'" in msg  # root
    assert "'decomp_Y'" in msg  # symptom
    assert "forced worst" in msg


# ---------------------------------------------------------------------------
# Env resolver — mirrors the workers knob
# ---------------------------------------------------------------------------
def test_max_stall_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_BENDERS_MAX_STALL", raising=False)
    assert _resolve_benders_max_stall() == _STALL_WINDOW_DEFAULT


def test_max_stall_env_override(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_BENDERS_MAX_STALL", "20")
    assert _resolve_benders_max_stall() == 20


def test_max_stall_non_positive_ignored(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_BENDERS_MAX_STALL", "0")
    assert _resolve_benders_max_stall() == _STALL_WINDOW_DEFAULT
    monkeypatch.setenv("FLEXTOOL_BENDERS_MAX_STALL", "-5")
    assert _resolve_benders_max_stall() == _STALL_WINDOW_DEFAULT


def test_max_stall_non_integer_ignored_with_warning(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("FLEXTOOL_BENDERS_MAX_STALL", "not-a-number")
    with caplog.at_level(logging.WARNING):
        assert _resolve_benders_max_stall() == _STALL_WINDOW_DEFAULT
    assert any("non-integer" in r.message for r in caplog.records)
