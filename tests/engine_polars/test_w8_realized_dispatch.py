"""W8.1 — realized operations from a SEPARATE deterministic dispatch solve.

This end-to-end test pins the W8 premise: in a genuinely stochastic
model, the reported realized OPERATIONS can come from a separate
*deterministic* realized-dispatch solve whose investments are FIXED —
via the existing invest→dispatch handoff — from the stochastic solve.
Establishing that realized ops have a home is what later makes W4 ("hedge
forecasts only") safe.  Whether the realized-dispatch pairing is
author-declared (as in this fixture) or engine-automatic is a deferred
W4-era decision.

No engine code changes: the fixture wires a stochastic→deterministic
two-solve chain and this test locks the behaviour in.  The machinery
exercised is identical to the proven deterministic ``lh2_invest →
lh2_dispatch`` precedent (``test_invest_dispatch_handoff.py``), only with
a genuinely stochastic first solve.

Fixture (``tests/fixtures/stoch_realized_dispatch.json``, built by
``tests/fixtures/build_stoch_realized_dispatch.py``):

  * ``stoch_rd`` — GENUINELY stochastic, ``solve_mode=single_solve``
    (fan-at-first), ``stochastic_invest_method`` at the DEFAULT ``none``
    (shared first-stage invest).  Three branches rlz/high/low with
    weights SUMMING TO 1.0 (0.25 / 0.375 / 0.375).  Invests a peaker to
    cover the per-step residual (60 MW).
  * ``stoch_rd_realized_dispatch`` — DETERMINISTIC (no
    ``stochastic_branches``), listed AFTER in ``model.solves``, covering
    the SAME realized period p2035, consuming the stochastic solve's
    committed 60 MW peaker as FIXED existing capacity through the
    handoff.  It builds no ``v_invest`` of its own.

Four assertions (see the per-test docstrings):
  (a) both solves reach a strict optimum;
  (b) the dispatch receives the committed capacity as FIXED existing
      capacity — it builds NO invest set / ``v_invest`` for the peaker,
      and its effective invested capacity equals the stochastic solve's
      ``v_invest``;
  (c) the reported realized operations are SOURCED FROM the dispatch
      solve — the stochastic solve's realized (period, step) window is
      emptied by last-wins, so the dispatch solve owns the realized
      output;
  (d) a hand-calc of the dispatch solve's objective.

Note on the W4 boundary (why the realized data is deterministic): both
solves share one scenario, so a parameter resolves to the same value for
both.  A branch-DISTINCT (3-deep branch×time_start×time Map) profile is
only foldable by a genuinely stochastic solve; the deterministic
dispatch solve drops it and wind collapses to a 0 upper_limit (verified
while authoring this fixture).  Letting the hedge forecasts diverge from
the realized data the dispatch reads is exactly the deferred W4 change,
so W8.1 keeps the realized operational data deterministic (readable by
both solves) and the forecast branches structurally genuine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_JSON = (
    FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_realized_dispatch.json"
)

SCENARIO = "w8_realized_dispatch"
STOCH_SOLVE = "stoch_rd"
DISPATCH_SOLVE = "stoch_rd_realized_dispatch"
REALIZED_STEPS = {"t0001", "t0002", "t0003"}

# Hand-calc constants (see module docstring + build script).
COMMITTED_PEAKER_MW = 60.0          # shared first-stage v_invest
# dispatch objective = peaker 60 MW · 10 CUR/MWh · 8760 h/yr.
HOURS_PER_YEAR = 8760.0
PEAKER_VAR_COST = 10.0
DISPATCH_OBJECTIVE = COMMITTED_PEAKER_MW * PEAKER_VAR_COST * HOURS_PER_YEAR


def _build_db(tmp_path: Path) -> str:
    """Materialise the committed JSON fixture into a fresh SQLite and
    return its url (build-from-JSON + migrate, per CLAUDE.md invariant 3;
    never read a checked-in .sqlite)."""
    if not FIXTURE_JSON.exists():
        pytest.skip(f"fixture JSON not present: {FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / "stoch_realized_dispatch.sqlite"
    url = json_to_db(FIXTURE_JSON, db_path)
    migrate_database(url)
    return url


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """Run the two-solve chain once under ``FLEXTOOL_AUTOSCALE_STRICT=1``
    with ``warm=False`` + ``keep_solutions=True`` (so per-step
    ``flex_data`` / ``solution`` survive) and return
    ``(steps, work_folder)``."""
    from flextool.engine_polars import run_chain_from_db

    tmp_path = tmp_path_factory.mktemp("w8_realized_dispatch")
    url = _build_db(tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            url, SCENARIO, work_folder=work_folder,
            warm=False, keep_solutions=True,
        )
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev

    assert steps, "run_chain_from_db produced no orchestration steps"
    return steps, work_folder


def _step(steps, needle):
    key = next((k for k in steps if needle in k), None)
    assert key is not None, f"no {needle!r} step in {list(steps)}"
    return steps[key]


def _as_frame(param):
    """Unwrap a FlexData Param (``.frame``) or passthrough a polars frame."""
    if param is None:
        return None
    return param.frame if hasattr(param, "frame") else param


# ---------------------------------------------------------------------------
# (a) both solves reach a strict optimum
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_both_solves_optimal(chain) -> None:
    """(a) The stochastic invest solve and the deterministic
    realized-dispatch solve both complete with a strict optimum."""
    steps, _wf = chain
    stoch = _step(steps, STOCH_SOLVE)
    dispatch = _step(steps, DISPATCH_SOLVE)

    assert stoch.solution is not None, "stochastic solve produced no solution"
    assert stoch.optimal, "stochastic solve not optimal"
    assert dispatch.solution is not None, "dispatch solve produced no solution"
    assert dispatch.optimal, "dispatch solve not optimal"


# ---------------------------------------------------------------------------
# (b) the dispatch receives the committed capacity as FIXED existing capacity
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_dispatch_capacity_is_fixed_from_stochastic_invest(chain) -> None:
    """(b) The dispatch solve builds NO invest for the peaker — its
    capacity is FIXED at the stochastic solve's committed ``v_invest``.

    Detection of "capacity fixed":
      * the stochastic solve DID build a process invest set + deposited a
        non-empty handoff with ``realized_invest[peaker] = 60``;
      * the dispatch solve's FlexData has ``pd_invest_set is None`` (it
        builds NO ``v_invest`` for the peaker) yet carries
        ``p_entity_invested[peaker] = 60`` — i.e. the 60 MW arrived purely
        as fixed existing capacity through the handoff overlay, not as a
        decision variable it could re-optimise;
      * the two values are equal (the dispatch's fixed capacity ==
        the stochastic solve's investment decision).
    """
    steps, _wf = chain
    stoch = _step(steps, STOCH_SOLVE)
    dispatch = _step(steps, DISPATCH_SOLVE)

    # The stochastic solve genuinely invested (precondition for a handoff).
    assert stoch.flex_data is not None
    assert stoch.flex_data.pd_invest_set is not None, (
        "stochastic solve built no process invest set"
    )
    pcol = stoch.flex_data.pd_invest_set.columns[0]
    assert "peaker" in set(
        stoch.flex_data.pd_invest_set[pcol].to_list()
    ), "peaker is not in the stochastic solve's invest set"

    assert stoch.handoff is not None and not stoch.handoff.is_empty(), (
        "stochastic solve deposited no handoff"
    )
    ri = stoch.handoff.realized_invest
    assert ri is not None and ri.height > 0
    ent_col, val_col = ri.columns[0], ri.columns[-1]
    committed = {
        r[ent_col]: float(r[val_col]) for r in ri.iter_rows(named=True)
    }
    assert committed.get("peaker") == pytest.approx(COMMITTED_PEAKER_MW), (
        f"stochastic solve committed {committed.get('peaker')} MW peaker, "
        f"expected {COMMITTED_PEAKER_MW}"
    )

    # The dispatch solve builds NO invest for the peaker ...
    assert dispatch.flex_data is not None
    assert dispatch.flex_data.pd_invest_set is None, (
        "dispatch solve built a process invest set — it should consume the "
        "stochastic solve's investment as fixed existing capacity, not "
        "re-optimise it"
    )
    disp_sol = dispatch.solution
    disp_vars = set(getattr(disp_sol, "_vars", {}).keys())
    assert not any("v_invest" in v for v in disp_vars), (
        f"dispatch solve built v_invest variables {sorted(disp_vars)}"
    )

    # ... yet its effective invested capacity equals the committed 60 MW.
    pei = _as_frame(dispatch.flex_data.p_entity_invested)
    assert pei is not None, (
        "dispatch FlexData has no p_entity_invested — the handoff overlay "
        "did not reach the dispatch solve"
    )
    pei_ent, pei_val = pei.columns[0], pei.columns[-1]
    dispatch_cap = {
        r[pei_ent]: float(r[pei_val]) for r in pei.iter_rows(named=True)
    }
    assert dispatch_cap.get("peaker") == pytest.approx(COMMITTED_PEAKER_MW), (
        f"dispatch peaker fixed capacity {dispatch_cap.get('peaker')} != "
        f"stochastic v_invest {COMMITTED_PEAKER_MW}"
    )
    assert dispatch_cap["peaker"] == pytest.approx(committed["peaker"]), (
        "dispatch fixed capacity must equal the stochastic solve's v_invest"
    )


# ---------------------------------------------------------------------------
# (c) reported realized operations are sourced from the dispatch solve
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_realized_operations_sourced_from_dispatch_solve(chain) -> None:
    """(c) The dispatch solve OWNS the reported realized operations.

    Both solves declare ``realized_periods=[p2035]`` over the same steps.
    The dispatch solve is listed AFTER the stochastic solve in
    ``model.solves``, so the engine's last-wins resolution
    (``_native_run_model`` reversed-order claim) TRUNCATES the stochastic
    solve's realized window and the dispatch solve owns it.

    Detection via each solve's per-solve realized window
    (``flex_data.realized_dispatch`` — the (period, step) pairs that solve
    persists as realized, which drive the multi-solve output union):
      * the stochastic solve — though it DECLARES p2035 realized — has an
        EMPTY realized window (None / 0 rows): emptied by last-wins;
      * the dispatch solve owns the full p2035 / {t0001,t0002,t0003}
        realized window.
    """
    steps, _wf = chain
    stoch = _step(steps, STOCH_SOLVE)
    dispatch = _step(steps, DISPATCH_SOLVE)

    stoch_rd = stoch.flex_data.realized_dispatch
    # Emptied by last-wins: the stoch solve DECLARES realized p2035 (run
    # alone it owns a (3,2) window), but the later-declared dispatch solve
    # claims those cells, so here it is None (or a frame with no p2035 rows).
    # The dispatch-side assertions below pin the complementary direction, so
    # a last-wins regression fails loudly either way (stoch keeps the window
    # → this assert fires; dispatch loses it → the disp_rd asserts fire).
    if stoch_rd is not None:
        remaining = stoch_rd.filter(pl.col("period") == "p2035")
        assert remaining.height == 0, (
            "stochastic solve still owns realized p2035 cells — last-wins "
            f"did not hand them to the dispatch solve:\n{stoch_rd}"
        )

    disp_rd = dispatch.flex_data.realized_dispatch
    assert disp_rd is not None and disp_rd.height > 0, (
        "dispatch solve has no realized window — it should OWN realized "
        "p2035 after last-wins"
    )
    owned = disp_rd.filter(pl.col("period") == "p2035")
    assert set(owned["step"].to_list()) == REALIZED_STEPS, (
        f"dispatch realized window {sorted(owned['step'].to_list())} != "
        f"{sorted(REALIZED_STEPS)}"
    )


# ---------------------------------------------------------------------------
# (d) hand-calc of the dispatch solve's objective
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_dispatch_objective_hand_calc(chain) -> None:
    """(d) Hand-calc the deterministic dispatch solve's objective.

    Realized (deterministic) data read by the dispatch solve:
      * demand = 100 MW/step, 3 steps of 1 h;
      * wind upper_limit profile 0.40 → wind cap 0.40 · 100 = 40 MW/step
        (cost 0), so wind supplies 40 MW/step;
      * peaker fills the residual = 100 − 40 = 60 MW/step at 10 CUR/MWh;
      * the peaker's fixed existing capacity (60 MW) is non-binding and
        carries NO annuity in the dispatch solve.

    The 3-step timeset represents a full year (each modelled hour stands
    for 8760/3 h), so the annual peaker generation is 60 MW · 8760 h and
    the dispatch objective is pure fuel:

        objective = 60 MW · 8760 h · 10 CUR/MWh = 5 256 000 CUR.

    The physical split (peaker 60 MW/step, wind 40 MW/step) is also
    asserted directly off the dispatch solution — it is scaling-free.
    """
    steps, wf = chain
    dispatch = _step(steps, DISPATCH_SOLVE)

    # Objective from the per-solve output (independent of solver internals).
    obj = float(
        pl.read_parquet(wf / "output_raw" / f"v_obj__{DISPATCH_SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(DISPATCH_OBJECTIVE, rel=1e-9), (
        f"dispatch objective {obj} != hand-calc {DISPATCH_OBJECTIVE}"
    )

    # Physical dispatch split (scaling-free): peaker 60 MW, wind 40 MW/step.
    flows = dispatch.solution.value("v_flow")
    peaker = flows.filter(pl.col("source") == "peaker")
    assert peaker.height == 3
    assert all(
        v == pytest.approx(60.0) for v in peaker["value"].to_list()
    ), f"peaker dispatch != 60 MW/step:\n{peaker}"
    # wind's v_flow is reported as a capacity fraction (0.40 of existing
    # 100 MW = 40 MW); the balance 40 + 60 = 100 MW = demand confirms it.
    wind = flows.filter(pl.col("source") == "wind")
    assert wind.height == 3
    assert all(
        v == pytest.approx(0.40) for v in wind["value"].to_list()
    ), f"wind dispatch != 0.40 capacity-fraction (40 MW):\n{wind}"
