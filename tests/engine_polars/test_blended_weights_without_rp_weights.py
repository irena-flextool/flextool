"""Phase 5 regression: strict per-solve check for blended-weights w/o RP weights.

Covers the early ``_assert_blended_weights_have_rp_weights`` precondition
added in ``flextool/engine_polars/_native_run_model.py`` (Phase 5 of the
``storage_binding_method`` single-valued cleanup).

The originating user failure (H2_trade.sqlite scenario ``y2050`` solve
``lt_rp`` on timeset ``ts_y2050``) used to surface as a deep
``FlexData loader: nodeState_rp is non-empty (15 node(s)) but the
tightly-coupled field 'rp_base_period_set' is missing or empty.`` error
buried inside ``input.py`` — opaque to a user who only knows about
scenarios + alternatives.  The new check fires at the solve boundary
with the full context (solve name, scenario-style timeset name,
offending nodes, two concrete fixes), and the late ``input.py``
invariant is demoted to a backstop.

Building a full SpineDB fixture that reproduces this end-to-end is
non-trivial (would require staging a v54 DB with a real timeset, a
solve, and the correct alternatives wiring), so this test is a
focused unit test of the strict-check helper.  It exercises exactly
the code path that runs inside ``_native_run_model``'s per-solve loop
just before the ``emit_empty_rp_data`` fallthrough — populating the
same Provider key with the same shape that the cascade does in
production, then calling the helper with the same kwargs the cascade
passes.

The test asserts that the four expected tokens (``node name``,
``timeset name``, ``representative_period_weights``, ``alternative``)
appear in the error message AND that the error originates from
``_native_run_model.py`` (NOT from ``input.py:2759-2779``).
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._native_run_model import (
    _assert_blended_weights_have_rp_weights,
    _nodes_with_blended_weights,
)
from flextool.engine_polars._solve_state import FlexToolConfigError


def _seed_storage_binding_method_provider(
    *, nodes_blended: list[str], nodes_other: list[tuple[str, str]] | None = None,
) -> FlexDataProvider:
    """Build a Provider that mirrors what the cascade seeds into
    ``input/node__storage_binding_method`` at the point where the
    strict check runs.

    ``nodes_blended`` are written with ``bind_using_blended_weights``;
    ``nodes_other`` is an optional list of ``(node, method)`` pairs for
    sibling nodes carrying a different (non-RP) method.
    """
    provider = FlexDataProvider()
    rows_n: list[str] = list(nodes_blended)
    rows_m: list[str] = ["bind_using_blended_weights"] * len(nodes_blended)
    for n, m in (nodes_other or []):
        rows_n.append(n)
        rows_m.append(m)
    frame = pl.DataFrame(
        {"node": rows_n, "storage_binding_method": rows_m},
        schema={"node": pl.Utf8, "storage_binding_method": pl.Utf8},
    )
    provider.put("input/node__storage_binding_method", frame)
    return provider


def _make_logger() -> logging.Logger:
    log = logging.getLogger("test_blended_weights_without_rp_weights")
    log.setLevel(logging.ERROR)
    return log


# ---------------------------------------------------------------------------
# Helper under test: _nodes_with_blended_weights.
# ---------------------------------------------------------------------------


def test_nodes_with_blended_weights_empty_provider(tmp_path: Path) -> None:
    """Missing input CSV => empty list (no false positives)."""
    provider = FlexDataProvider()
    assert _nodes_with_blended_weights(tmp_path / "input", provider) == []


def test_nodes_with_blended_weights_filters_by_method(tmp_path: Path) -> None:
    """Only ``bind_using_blended_weights`` rows are returned; sorted."""
    provider = _seed_storage_binding_method_provider(
        nodes_blended=["beta", "alpha"],
        nodes_other=[("gamma", "bind_within_solve"),
                     ("delta", "bind_within_timeset")],
    )
    out = _nodes_with_blended_weights(tmp_path / "input", provider)
    assert out == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Strict check: error contents + provenance.
# ---------------------------------------------------------------------------


def test_strict_check_no_blended_weights_is_noop(tmp_path: Path) -> None:
    """No blended-weights nodes => helper returns silently."""
    provider = _seed_storage_binding_method_provider(
        nodes_blended=[],
        nodes_other=[("ARG_H2", "bind_within_solve")],
    )
    # Must not raise — the empty-RP path is correct when no node uses RP.
    _assert_blended_weights_have_rp_weights(
        solve="lt_rp",
        complete_solve_name="lt_rp",
        roll_index=0,
        active_timeset_names=["ts_y2050"],
        rp_weights_keys=[],
        input_dir=tmp_path / "input",
        provider=provider,
        logger=_make_logger(),
    )


def test_strict_check_blended_weights_no_rp_weights_raises(
    tmp_path: Path,
) -> None:
    """The originating H2_trade failure mode — should error with all four
    expected tokens, naming the solve + timeset + offending node + fix."""
    # Reproduce the 15-node H2_trade shape: ARG_H2, AUS_H2, CHI_H2 + 12 more.
    blended_nodes = [
        "ARG_H2", "AUS_H2", "CHI_H2", "EGY_H2", "IND_H2",
        "KAZ_H2", "MAR_H2", "MEX_H2", "NAM_H2", "OMN_H2",
        "PER_H2", "SAU_H2", "TUR_H2", "USA_H2", "ZAF_H2",
    ]
    provider = _seed_storage_binding_method_provider(
        nodes_blended=blended_nodes,
    )
    with pytest.raises(FlexToolConfigError) as exc:
        _assert_blended_weights_have_rp_weights(
            solve="lt_rp",
            complete_solve_name="lt_rp",
            roll_index=0,
            active_timeset_names=["ts_y2050"],
            rp_weights_keys=[],
            input_dir=tmp_path / "input",
            provider=provider,
            logger=_make_logger(),
        )
    msg = str(exc.value)
    # Solve + timeset context surfaced.
    assert "lt_rp" in msg, f"solve name missing from error: {msg!r}"
    assert "ts_y2050" in msg, f"timeset name missing from error: {msg!r}"
    # At least one offending node named (the first 5 by definition).
    assert "ARG_H2" in msg, f"first offending node missing: {msg!r}"
    # 15 > 5 => the truncation suffix kicks in.
    assert "... 10 more" in msg, (
        f"node-truncation summary missing: {msg!r}"
    )
    # The two mandated tokens from the spec.
    assert "representative_period_weights" in msg, (
        f"spec token missing: {msg!r}"
    )
    assert "alternative" in msg.lower(), (
        f"spec token 'alternative' missing: {msg!r}"
    )
    # Provenance: error must originate in _native_run_model.py
    # (NOT the late input.py:2759-2779 backstop).  Walk the traceback
    # frames to verify.  ``exc.traceback`` is a pytest Traceback of
    # TracebackEntry objects; each has a ``.path`` attribute.
    frames = [str(entry.path) for entry in exc.traceback]
    assert any("_native_run_model.py" in f for f in frames), (
        f"strict check did not originate in _native_run_model.py: {frames}"
    )
    assert not any(
        ("engine_polars" in f and f.endswith("input.py")) for f in frames
    ), (
        f"strict check leaked through to input.py backstop: {frames}"
    )


def test_strict_check_truncation_below_five_nodes(tmp_path: Path) -> None:
    """With <=5 blended nodes the message lists them all, no '... N more'."""
    blended_nodes = ["bat_a", "bat_b", "bat_c"]
    provider = _seed_storage_binding_method_provider(
        nodes_blended=blended_nodes,
    )
    with pytest.raises(FlexToolConfigError) as exc:
        _assert_blended_weights_have_rp_weights(
            solve="my_solve",
            complete_solve_name="my_solve",
            roll_index=2,
            active_timeset_names=["ts_short"],
            rp_weights_keys=["ts_other"],
            input_dir=tmp_path / "input",
            provider=provider,
            logger=_make_logger(),
        )
    msg = str(exc.value)
    for n in blended_nodes:
        assert n in msg, f"node {n!r} missing from <=5 case: {msg!r}"
    assert "more" not in msg.split("(", 1)[1].split(")", 1)[0], (
        f"truncation marker appeared with <=5 nodes: {msg!r}"
    )
    # The "other timeset that DOES have RP weights" hint is surfaced —
    # helps the user see what an alternative SHOULD look like.
    assert "ts_other" in msg, (
        f"competing-timeset hint missing from message: {msg!r}"
    )
