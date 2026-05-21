"""End-to-end test for the Phase 5 external override provider path.

Phase 5c of ``specs/provider_consolidation.md``.  Phase 5a wired the
``override/*`` Provider namespace + the ``read_handoff_frame`` override-
aware lookup; Phase 5b wired the orchestrator to invoke
``state.override_provider`` at iteration start and fan the returned
dict into the ``override/*`` keys.  This test demonstrates that the
override actually shadows the natural handoff value when preprocessing
builds the LP inputs — i.e., the full Phase 5 mechanism delivers
external data into the per-solve cascade without code changes to any
consumer.

Strategy
========

1. Run the 4-solve invest+lifetime-renew cascade once unmodified.
   Capture solve 1 (``y2020_5week``)'s post-solve ``handoff.realized_invest``
   — this is the natural ``prior_handoff`` solve 2+ would consume.

2. Run the cascade again with ``override_provider`` returning a SCALED
   copy of that frame (×1e3) keyed by ``K.HANDOFF_REALIZED_INVEST``.
   The override fires for every iteration after the first; on the
   first sub-solve it returns ``{}`` so behaviour matches baseline.

3. Assert that the override propagated all the way into the per-solve
   preprocessing output: solve 2's
   ``solve_data/p_entity_previously_invested_capacity`` Provider entry
   reflects the inflated values, NOT the natural handoff values.

The assertion target — ``p_entity_previously_invested_capacity`` — is
the CSV emitted by ``_emit_chain_params.emit_p_entity_existing_chain``
after reading the handoff via ``read_handoff_frame``.  This proves the
full Phase 5a + 5b chain: external dict → ``translate_overrides_to_provider``
→ ``override/realized_invest`` Provider key → ``read_handoff_frame``
shadow lookup → ``_load_realized_from_handoff`` → preprocessing emit
→ Provider's ``solve_data/p_entity_previously_invested_capacity``.

Why this assertion shape (vs. comparing objectives)
---------------------------------------------------

The objective is an indirect signal: an inflated
``p_entity_previously_invested_capacity`` only shifts the optimum if
the corresponding constraint is binding, which depends on the fixture's
LP topology.  The Provider entry assertion is the direct signal that
the override mechanism delivered the data — independent of whether the
LP happens to be sensitive to that particular parameter in this
fixture.  This is the alternative path explicitly called out in
``specs/provider_consolidation.md`` Phase 5c (probe the Provider state
rather than the LP objective).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars import _provider_keys as K

pytestmark = pytest.mark.solver


WORK = (
    Path(__file__).resolve().parent
    / "data"
    / "work_wind_battery_invest_lifetime_renew_4solve"
)
SCENARIO_NAME = "wind_battery_invest_lifetime_renew_4solve"
SECOND_SOLVE = "y2025_5week"

# Scale factor applied to the natural ``realized_invest`` to produce the
# override frame.  Large enough that the resulting
# ``p_entity_previously_invested_capacity`` is unambiguously distinct
# from the natural value but small enough not to overflow / produce
# nonsense.
OVERRIDE_SCALE = 1_000.0


def _read_prev_inv(step) -> dict[tuple[str, str], float]:
    """Read ``solve_data/p_entity_previously_invested_capacity`` from
    the OrchestrationStep's Provider into a ``(entity, period) → value``
    dict for easy comparison.
    """
    prov = step.flex_data_provider
    assert prov is not None, (
        f"step.flex_data_provider missing — keep_solutions=True is required "
        f"for the assertion path; without it earlier steps are slimmed."
    )
    # No K constant for this key yet (mechanical migration deferred per
    # specs/provider_consolidation.md "Incremental follow-ups not blocking
    # Phase 4"); use the literal until then.
    _key = "solve_data/p_entity_previously_invested_capacity"
    frame = prov.get(_key)
    assert frame is not None, (
        f"Provider missing {_key!r} — preprocessing should always emit "
        f"this key for cascade solves."
    )
    # ``_ed_value_frame`` emits ``(entity, period, value)`` with ``value``
    # as Utf8 (the legacy CSV-roundtrip contract); consumers cast on
    # read.
    out: dict[tuple[str, str], float] = {}
    for r in frame.iter_rows(named=True):
        out[(str(r["entity"]), str(r["period"]))] = float(r["value"])
    return out


def test_external_override_shadows_handoff_realized_invest(tmp_path) -> None:
    if not WORK.exists():
        pytest.skip(f"fixture {WORK} not present")
    db_path = WORK / "tests.sqlite"
    if not db_path.exists():
        pytest.skip(f"DB {db_path} not present")

    # ── Run 1: baseline (no override) ───────────────────────────────────
    baseline_work = tmp_path / "baseline"
    baseline_work.mkdir()
    baseline = run_chain_from_db(
        db_path,
        scenario_name=SCENARIO_NAME,
        work_folder=baseline_work,
        keep_solutions=True,
    )
    chain_order = list(baseline)
    assert chain_order == [
        "y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week",
    ], f"chain order changed unexpectedly: {chain_order}"

    # Solve 1's natural realized_invest carrier — the override base.
    solve1_natural = baseline["y2020_5week"].handoff.realized_invest
    assert solve1_natural is not None and solve1_natural.height > 0, (
        "fixture invariant: y2020_5week handoff.realized_invest should be "
        "populated; the override test depends on a non-empty prior carrier."
    )

    # Baseline solve-2 ``p_entity_previously_invested_capacity`` — the
    # expected DENOMINATOR for the override-effect comparison.
    baseline_prev_inv = _read_prev_inv(baseline[SECOND_SOLVE])
    # At least one (entity, p2025) row should be non-zero (battery /
    # wind_plant were realised in solve 1).  Without this the override
    # comparison is vacuous.
    baseline_nonzero = [
        ((e, d), v) for (e, d), v in baseline_prev_inv.items() if v > 0.0
    ]
    assert baseline_nonzero, (
        f"fixture invariant: {SECOND_SOLVE} should have at least one non-zero "
        f"p_entity_previously_invested_capacity row (battery + wind_plant "
        f"realised in solve 1); got all-zeros."
    )

    # ── Override frame: scale realized_invest values by OVERRIDE_SCALE ──
    scaled = solve1_natural.with_columns(
        (pl.col("value") * OVERRIDE_SCALE).alias("value")
    )

    # ── Override provider — captures invocations + returns the scaled
    # frame on iterations after the first.
    invocations: list[int] = []

    def _override() -> dict[str, pl.DataFrame]:
        call_idx = len(invocations)
        invocations.append(call_idx)
        if call_idx == 0:
            # First sub-solve has no prior carrier; skip the override.
            return {}
        return {K.HANDOFF_REALIZED_INVEST: scaled}

    # ── Run 2: with override ────────────────────────────────────────────
    override_work = tmp_path / "override"
    override_work.mkdir()
    overridden = run_chain_from_db(
        db_path,
        scenario_name=SCENARIO_NAME,
        work_folder=override_work,
        keep_solutions=True,
        override_provider=_override,
    )

    # ── Assertion 1: override callable was invoked per sub-solve ───────
    assert len(invocations) == 4, (
        f"override_provider should be invoked once per sub-solve iteration; "
        f"got {len(invocations)} invocations (expected 4 for the 4-solve "
        f"fixture)."
    )

    # ── Assertion 2: chain still completes optimally ───────────────────
    for solve_name, step in overridden.items():
        assert step.optimal, (
            f"{solve_name}: solve did not return optimal status — the override "
            f"inflated prior-invested capacity but should not have broken "
            f"feasibility (the LP can only build more, not less)."
        )

    # ── Assertion 3: override propagated to solve 2's preprocessing ────
    # ``p_entity_previously_invested_capacity`` is emitted by
    # ``_emit_chain_params.emit_p_entity_existing_chain``, which reads
    # ``realized_invest`` via ``read_handoff_frame`` — i.e. the Phase 5a
    # override-aware lookup.  Inflating the override by OVERRIDE_SCALE
    # should produce a proportionally inflated capacity row for the
    # entities the override touched.
    overridden_prev_inv = _read_prev_inv(overridden[SECOND_SOLVE])

    for (entity, period), baseline_v in baseline_nonzero:
        override_v = overridden_prev_inv.get((entity, period), 0.0)
        # Allow modest fp slack (we're scaling, no precision loss expected).
        expected = baseline_v * OVERRIDE_SCALE
        assert abs(override_v - expected) < 1e-6 * max(abs(expected), 1.0), (
            f"({entity}, {period}): override should have scaled "
            f"p_entity_previously_invested_capacity by ×{OVERRIDE_SCALE}; "
            f"baseline={baseline_v}, expected_override={expected}, "
            f"observed_override={override_v}.  Either the override did "
            f"not reach _emit_chain_params._load_realized_from_handoff "
            f"(Phase 5a/5b plumbing gap) or the scale was lost in "
            f"transit."
        )
