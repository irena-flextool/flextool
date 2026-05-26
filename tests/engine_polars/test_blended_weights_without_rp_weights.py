"""Phase C regression: silent-degrade replaces the Phase 5 strict check.

Covers the
:func:`flextool.engine_polars._native_run_model._downgrade_rp_methods_for_non_rp_solve`
helper introduced in Phase C of the
``storage-binding-methods-restructure`` migration, plus the
``nodeBalance_eq`` not-yet-implemented guard in
:mod:`flextool.engine_polars.model` for the two methods whose
constraint implementations land in Phases D / E
(``bind_within_period_blended_weights`` and
``bind_forward_only_blended_weights``).

What changed vs. Phase 5
------------------------
The original Phase 5 strict check raised ``FlexToolConfigError`` when a
node carried ``bind_within_solve_blended_weights`` but the solve's
active timeset had no ``representative_period_weights``.  That check
was wrong: the same storage entity can legitimately participate in
BOTH an RP investment solve AND a chronological dispatch solve
back-to-back.  Phase C replaces the strict check with a silent
per-solve downgrade — the offending method is rewritten to its non-RP
equivalent in the per-solve provider only, leaving the on-disk DB and
upstream CSVs untouched.  A second line of defence sits in
``model.py``'s ``nodeBalance_eq`` for the two methods whose constraint
implementations are not yet wired.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._native_run_model import (
    _downgrade_rp_methods_for_non_rp_solve,
)
from flextool.engine_polars._solve_state import FlexToolConfigError


# ---------------------------------------------------------------------------
# Provider fixture helpers.
# ---------------------------------------------------------------------------


def _seed_storage_binding_method_provider(
    *, rows: list[tuple[str, str]],
) -> FlexDataProvider:
    """Seed ``input/node__storage_binding_method`` with the given rows.

    ``rows`` is a list of ``(node, method)`` pairs.  Mirrors the
    parent-qualified Provider key convention used by the rest of the
    cascade (``"input/<csv-stem>"``).
    """
    provider = FlexDataProvider()
    nodes = [r[0] for r in rows]
    methods = [r[1] for r in rows]
    frame = pl.DataFrame(
        {"node": nodes, "storage_binding_method": methods},
        schema={"node": pl.Utf8, "storage_binding_method": pl.Utf8},
    )
    provider.put("input/node__storage_binding_method", frame)
    return provider


def _make_logger() -> logging.Logger:
    log = logging.getLogger("test_blended_weights_silent_degrade")
    log.setLevel(logging.INFO)
    return log


def _get_method_for_node(provider: FlexDataProvider, node: str) -> str | None:
    """Read back the method for ``node`` from the provider — or ``None``."""
    if not provider.has("input/node__storage_binding_method"):
        return None
    df = provider.get("input/node__storage_binding_method")
    col = ("storage_binding_method"
           if "storage_binding_method" in df.columns else "method")
    hit = df.filter(pl.col("node") == node).select(col)
    if hit.height == 0:
        return None
    return hit.row(0)[0]


# ---------------------------------------------------------------------------
# Test 1 — downgrade fires when active timeset has no RP weights.
# ---------------------------------------------------------------------------


def test_blended_weights_node_in_non_rp_solve_downgrades_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A node carrying ``bind_within_solve_blended_weights`` in a solve
    whose active timeset has no ``representative_period_weights`` is
    silently rewritten to ``bind_within_solve``.  No exception; one
    info-level log line per (old, new) downgrade pair."""
    provider = _seed_storage_binding_method_provider(
        rows=[
            ("ARG_H2", "bind_within_solve_blended_weights"),
            ("USA_H2", "bind_within_solve_blended_weights"),
            ("CHI_H2", "bind_within_solve_blended_weights"),
            # A non-RP sibling node — should be untouched.
            ("baseload", "bind_within_solve"),
        ],
    )
    logger = _make_logger()
    caplog.set_level(logging.INFO,
                     logger="test_blended_weights_silent_degrade")
    # No exception expected.
    _downgrade_rp_methods_for_non_rp_solve(
        solve="lt_rp",
        complete_solve_name="lt_rp",
        roll_index=0,
        active_timeset_names=["ts_y2050"],
        rp_weights={},  # empty — no timeset has RP weights
        provider=provider,
        logger=logger,
    )
    # Every blended-weights node now reads as its non-RP equivalent.
    for n in ("ARG_H2", "USA_H2", "CHI_H2"):
        assert _get_method_for_node(provider, n) == "bind_within_solve", (
            f"{n} was not downgraded: still reads "
            f"{_get_method_for_node(provider, n)!r}"
        )
    # The non-blended node remained untouched.
    assert _get_method_for_node(provider, "baseload") == "bind_within_solve"
    # Exactly one log line for the single (old, new) pair that fired,
    # naming the count, the old name and the new name.
    info_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.INFO
    ]
    assert len(info_msgs) == 1, (
        f"expected one info-level downgrade line, got {info_msgs!r}"
    )
    msg = info_msgs[0]
    assert "3 node(s)" in msg, f"missing node count: {msg!r}"
    assert "bind_within_solve_blended_weights" in msg, msg
    assert "bind_within_solve" in msg, msg
    assert "ts_y2050" in msg, f"missing active-timeset name: {msg!r}"
    assert "lt_rp" in msg, f"missing solve name: {msg!r}"


# ---------------------------------------------------------------------------
# Test 2 — downgrade is a no-op when the active timeset HAS RP weights.
# ---------------------------------------------------------------------------


def test_blended_weights_node_in_rp_solve_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same node + method, but this time the solve's active timeset
    DOES have ``representative_period_weights``.  The downgrade helper
    must be a strict no-op — the node keeps its blended-weights method
    so the RP code path stays live."""
    provider = _seed_storage_binding_method_provider(
        rows=[("ARG_H2", "bind_within_solve_blended_weights")],
    )
    logger = _make_logger()
    caplog.set_level(logging.INFO,
                     logger="test_blended_weights_silent_degrade")
    # Active timeset has an entry in rp_weights → RP path is correct,
    # do nothing.
    _downgrade_rp_methods_for_non_rp_solve(
        solve="invest_rp",
        complete_solve_name="invest_rp",
        roll_index=0,
        active_timeset_names=["ts_y2050"],
        rp_weights={"ts_y2050": [("p1", "step_001", 1.0)]},
        provider=provider,
        logger=logger,
    )
    assert (
        _get_method_for_node(provider, "ARG_H2")
        == "bind_within_solve_blended_weights"
    ), "RP-active solve must NOT downgrade blended-weights nodes"
    # No info log line should have been emitted.
    info_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.INFO
    ]
    assert info_msgs == [], (
        f"expected no downgrade log line on RP-active solve, got {info_msgs!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — same node drives BOTH paths in sequence (the headline use case).
# ---------------------------------------------------------------------------


def test_multi_solve_same_node_drives_both_paths(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One node carries ``bind_within_solve_blended_weights``.  We
    simulate two consecutive per-solve provider scenarios: (a) an
    RP-active solve, (b) a non-RP dispatch solve.  The helper must:

    * leave the method untouched on the RP-active solve,
    * downgrade to ``bind_within_solve`` on the non-RP solve,

    so the same storage entity drives both paths without raising.
    Each per-solve provider is a fresh in-memory carrier — modifying
    the non-RP provider does NOT mutate the RP provider's frame.
    """
    logger = _make_logger()
    caplog.set_level(logging.INFO,
                     logger="test_blended_weights_silent_degrade")

    # (a) RP-active solve: expect no-op.
    rp_provider = _seed_storage_binding_method_provider(
        rows=[("BAT_001", "bind_within_solve_blended_weights")],
    )
    _downgrade_rp_methods_for_non_rp_solve(
        solve="invest_rp",
        complete_solve_name="invest_rp",
        roll_index=0,
        active_timeset_names=["ts_invest"],
        rp_weights={"ts_invest": [("p1", "step_001", 1.0)]},
        provider=rp_provider,
        logger=logger,
    )
    assert (
        _get_method_for_node(rp_provider, "BAT_001")
        == "bind_within_solve_blended_weights"
    )

    # (b) Non-RP dispatch solve, fresh provider for the same node.
    chrono_provider = _seed_storage_binding_method_provider(
        rows=[("BAT_001", "bind_within_solve_blended_weights")],
    )
    _downgrade_rp_methods_for_non_rp_solve(
        solve="dispatch",
        complete_solve_name="dispatch",
        roll_index=0,
        active_timeset_names=["ts_dispatch"],
        rp_weights={"ts_invest": [("p1", "step_001", 1.0)]},
        provider=chrono_provider,
        logger=logger,
    )
    assert (
        _get_method_for_node(chrono_provider, "BAT_001")
        == "bind_within_solve"
    ), "non-RP solve must downgrade blended-weights to bind_within_solve"

    # The RP provider's frame is independent — its method is still the
    # blended-weights variant (no cross-provider mutation).
    assert (
        _get_method_for_node(rp_provider, "BAT_001")
        == "bind_within_solve_blended_weights"
    )

    # Exactly one downgrade log line — from the non-RP solve only.
    info_msgs = [
        rec.getMessage() for rec in caplog.records
        if rec.levelno == logging.INFO
    ]
    assert len(info_msgs) == 1, (
        f"expected one downgrade line (from non-RP solve), got {info_msgs!r}"
    )
    assert "dispatch" in info_msgs[0]


# ---------------------------------------------------------------------------
# Test 4 — not-yet-implemented method blocked by the model.py guard.
# ---------------------------------------------------------------------------


def test_not_yet_implemented_method_blocked() -> None:
    """A node carrying ``bind_within_period_blended_weights`` in an
    RP-active solve must be rejected by the
    :func:`flextool.engine_polars.model.build_flextool` guard with a
    :class:`FlexToolConfigError` naming the method and the Phase D/E hint.

    Exercises the actual production guard in ``model.py``'s
    ``nodeBalance_eq`` block, not a re-implementation.  The minimum
    :class:`flextool.engine_polars.input.FlexData` shape needed to reach
    the guard is small: ALWAYS-required fields satisfied by empty
    polars frames, every feature-flag attribute left at ``None``, and
    the not-yet-implemented partition pre-loaded with a single offending
    node.
    """
    import flextool.engine_polars.model as _model
    from polar_high.engine import Problem

    class _StubData:
        # ALWAYS-required fields per ``model.ALWAYS`` — empty frames
        # satisfy ``_check`` (None would raise; empty does not).
        dt = pl.DataFrame({"d": [], "t": []},
                          schema={"d": pl.Utf8, "t": pl.Utf8})
        nodeBalance = pl.DataFrame({"n": []}, schema={"n": pl.Utf8})
        p_step_duration = pl.DataFrame({"d": [], "t": [], "value": []},
                                       schema={"d": pl.Utf8, "t": pl.Utf8,
                                               "value": pl.Float64})
        p_rp_cost_weight = pl.DataFrame({"d": [], "value": []},
                                        schema={"d": pl.Utf8,
                                                "value": pl.Float64})
        p_inflation_op = pl.DataFrame({"d": [], "value": []},
                                      schema={"d": pl.Utf8,
                                              "value": pl.Float64})
        p_period_share = pl.DataFrame({"d": [], "value": []},
                                      schema={"d": pl.Utf8,
                                              "value": pl.Float64})
        p_inflow = pl.DataFrame({"n": [], "d": [], "t": [], "value": []},
                                schema={"n": pl.Utf8, "d": pl.Utf8,
                                        "t": pl.Utf8, "value": pl.Float64})
        p_penalty_up = pl.DataFrame({"n": [], "d": [], "value": []},
                                    schema={"n": pl.Utf8, "d": pl.Utf8,
                                            "value": pl.Float64})
        p_penalty_down = pl.DataFrame({"n": [], "d": [], "value": []},
                                      schema={"n": pl.Utf8, "d": pl.Utf8,
                                              "value": pl.Float64})
        # Trip the guard with two nodes so the truncation branch is
        # exercised too (covers the 10-node + "more" tail at the same
        # site).
        storage_bind_within_period_blended_weights = pl.DataFrame(
            {"n": ["BAT_001", "BAT_002"]},
            schema={"n": pl.Utf8},
        )

        def __getattr__(self, name):
            # Every other FlexData attribute defaults to None — matches
            # the loader's "feature disabled" convention.
            return None

    m = Problem()
    with pytest.raises(_model.FlexToolConfigError) as exc:
        _model.build_flextool(m, _StubData())
    msg = str(exc.value)
    # Mandated spec tokens.
    assert "bind_within_period_blended_weights" in msg, msg
    assert ("not landed yet" in msg or "Phase D/E" in msg), msg
    assert "BAT_001" in msg, msg
