"""Benders hard-failure diagnostics: plain-English formatting + a static guard
that the user-facing messages stay in FlexTool class vocabulary (node, group,
connection, flow) and never leak model-instance-specific terms.

The fail-safe recovery branches (LB monotonicity dip, LB-meets-UB sandwich) are
exercised end-to-end by the convergence gates in ``test_benders_phase2_loop`` /
``test_benders_phase3b_rp_loop`` (they must NOT trip on the well-conditioned
fixtures); here we pin the contract the hard-failure paths expose to the user.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from flextool.engine_polars._benders import _benders_failure_message

_BENDERS_SRC = Path(__file__).resolve().parents[2] / (
    "flextool/engine_polars/_benders.py"
)

# Model-instance vocabulary that must not appear in a user-facing diagnostic.
# The H2_trade model labels connections "trade"/"pipeline"/"line" and groups of
# nodes "regions"; the engine messages must use the FlexTool class names.
_FORBIDDEN = re.compile(r"\b(trade|pipe|pipeline|line|region|regions)\b", re.I)


def test_failure_message_has_three_plain_english_sections():
    msg = _benders_failure_message(
        summary="Something went wrong at iteration 3.",
        meaning="Here is what it means.",
        how_to_avoid="Here is how to avoid it.",
    )
    assert "Something went wrong at iteration 3." in msg
    assert "What this means:" in msg
    assert "How to avoid it:" in msg
    assert "Here is what it means." in msg
    assert "Here is how to avoid it." in msg
    # Summary leads; the two guidance sections follow in order.
    assert msg.index("What this means:") < msg.index("How to avoid it:")


def _failure_message_strings() -> list[tuple[str, int, str]]:
    """Every (kwarg, lineno, text) passed to a ``_benders_failure_message``
    call in the engine, with concatenated string-literal arguments joined."""
    tree = ast.parse(_BENDERS_SRC.read_text())
    out: list[tuple[str, int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "_benders_failure_message"
        ):
            for kw in node.keywords:
                text = "".join(
                    c.value
                    for c in ast.walk(kw.value)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)
                )
                out.append((kw.arg or "", kw.value.lineno, text))
    return out


def test_user_facing_diagnostics_use_flextool_vocabulary():
    fields = _failure_message_strings()
    # All real call sites (master/region not-optimal, LB dip, coupling
    # overshoot, LB>UB sandwich, cut-check gross, cut-check non-finite, STALL) —
    # each contributes 3 keyword fields (summary/meaning/how_to_avoid).  Bump
    # this bound when a call site is added so a *removed* message is still
    # caught here (the vocab walk below auto-covers any new site).
    assert len(fields) >= 8 * 3, f"too few diagnostic fields found: {len(fields)}"
    offenders = [
        (arg, lineno, sorted({m.group(0).lower() for m in _FORBIDDEN.finditer(t)}))
        for arg, lineno, t in fields
        if _FORBIDDEN.search(t)
    ]
    assert not offenders, (
        "model-instance vocabulary leaked into user-facing Benders "
        f"diagnostics: {offenders}"
    )
