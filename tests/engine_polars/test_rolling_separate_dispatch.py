"""OC1 rolling + stochastic separate_dispatch — interleaved UC/dispatch rolls
with a two-source handoff.

A ``dispatch_realization_method=separate_dispatch`` solve under
``solve_mode=rolling_window`` expands to the INTERLEAVED order
``[uc_roll_0, disp_roll_0, uc_roll_1, disp_roll_1, ...]`` where, per roll, a
stochastic UC solve (forecast fan) is followed by a deterministic
realized-dispatch sibling.  Two carrier flows are kept separate:

  * PIN: each ``disp_roll_k`` fixes its ``v_online`` to ``uc_roll_k``'s
    committed hedge (``online_fix_eq`` / ``fix_online``), consuming it from
    its sequential predecessor ``uc_roll_k``.
  * ROLL-FORWARD: the realized trajectory flows disp→disp — ``disp_roll_k``
    takes its roll-forward (``roll_end_online`` → ``p_roll_continue_online``,
    etc.) from ``disp_roll_{k-1}``, NOT from ``uc_roll_k`` (whose own
    roll-forward it ignores).  ``uc_roll_{k+1}`` in turn rolls forward on
    ``disp_roll_k`` via the linear chain.

This is pure orchestration + handoff routing — NO new var/cstr/param family
(the pin family ``online_fix_eq`` is already registered; R1/R2 add none), so
CLAUDE.md autoscale invariant #1 is not triggered.

Fixture (``rolling_separate_dispatch.json``, built from JSON per CLAUDE.md
invariant 3): 72 steps = 3 jump-24 rolls (horizon 48); a committable
``gas_plant`` (market-fed, min_load 0.4, startup, min_up/down 3) backs ``wind``
(branch-varying forecast profile) against demand; the ``rlz``/``upper``/
``mid``/``lower`` fan re-reveals at each roll boundary.  Operational only
(no invest; hydro reservoir drained to a commodity so the demo isolates the
commitment/dispatch behaviour).

Owner-confirmed semantics (``==`` pin): the realized dispatch is LOCKED to the
plan's committed hedge.  When the realized scenario is milder than the hedge,
min-load forces the committed units to over-produce; the excess spills via the
node down-penalty (down-slack).  That spill is EXPECTED and correct — a future
per-unit ``==``/``≤`` setting may relax it, not now.  The test therefore
asserts ZERO unserved load (up-slack) but ALLOWS down-slack (spill).

Hand-calc note: the per-roll committed hedge (``v_online`` = 8.0 / 4.795 /
4.641 at rolls 0/1/2) is the solver-determined tied forecast commitment of
each UC roll; it is pinned verbatim into that roll's dispatch sibling (the
robust assertion is the per-roll EQUALITY disp == uc, with the values pinned
as regression anchors).  The realized carry is exact: ``disp_roll_k`` consumes
``disp_roll_{k-1}``'s realized end commitment (roll 1 ← 8.0, roll 2 ← 4.795).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "rolling_separate_dispatch.json"

SCENARIO = "rolling_separate_dispatch"
UC = "1week_rolling_roll_{k}"
DISP = "1week_rolling__realized_dispatch_roll_{k}"
N_ROLLS = 3
GAS = "gas_plant"
REALIZED_PERIOD = "period1"            # the realized trunk the sibling owns

# Per-roll committed hedge (solver-determined tied forecast commitment),
# pinned verbatim into the dispatch sibling.  Regression anchors.
HEDGE = {0: 8.0, 1: 4.795181568, 2: 4.64102016}


def _build_db(tmp_path: Path) -> str:
    if not FIXTURE.exists():
        pytest.skip(f"fixture JSON not present: {FIXTURE}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{FIXTURE.stem}.sqlite"
    url = json_to_db(FIXTURE, db_path)
    migrate_database(url)
    return url


def _run_chain(tmp_path: Path):
    from flextool.engine_polars import run_chain_from_db

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


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return _run_chain(tmp_path_factory.mktemp("rolling_separate_dispatch"))


def _gas_online(step) -> pl.DataFrame:
    return step.solution.value("v_online_linear").filter(pl.col("p") == GAS)


@pytest.mark.solver
def test_interleaved_expansion_order(chain):
    """R1: the solve expands to the INTERLEAVED order
    ``[uc_0, disp_0, uc_1, disp_1, uc_2, disp_2]``."""
    steps, _ = chain
    expected = []
    for k in range(N_ROLLS):
        expected.append(UC.format(k=k))
        expected.append(DISP.format(k=k))
    assert list(steps) == expected


@pytest.mark.solver
def test_each_dispatch_roll_pinned_to_its_uc_commitment(chain):
    """R2 pin: each ``disp_roll_k`` v_online == ``uc_roll_k``'s committed
    hedge (per roll), and is non-degenerate (nonzero real capacity, not the
    weight-0 realized branch)."""
    steps, _ = chain
    for k in range(N_ROLLS):
        uc = _gas_online(steps[UC.format(k=k)])
        disp = _gas_online(steps[DISP.format(k=k)])
        # UC committed hedge = the tied forecast-branch commitment (exclude
        # the weight-0 realized trunk branch ``period1``).
        uc_hedge = uc.filter(pl.col("d") != REALIZED_PERIOD)["value"].max()
        disp_commit = disp.filter(
            pl.col("d") == REALIZED_PERIOD)["value"].max()
        assert disp_commit == pytest.approx(uc_hedge, rel=1e-6), (
            f"roll {k}: disp commit {disp_commit} != uc hedge {uc_hedge}"
        )
        assert disp_commit == pytest.approx(HEDGE[k], rel=1e-6)
        assert disp_commit > 0.0, f"roll {k}: dispatch commitment is zero"


@pytest.mark.solver
def test_realized_trajectory_carries_dispatch_to_dispatch(chain):
    """R2 roll-forward: ``disp_roll_k`` rolls forward on ``disp_roll_{k-1}``
    — its consumed ``p_roll_continue_online`` equals the PREVIOUS dispatch
    roll's realized end commitment (``roll_end_online``), nonzero.  (The
    first dispatch roll has no carry.)"""
    steps, _ = chain

    def _reo(k):   # realized end commitment handed off by disp_roll_k
        h = steps[DISP.format(k=k)].handoff.roll_end_online
        g = h.filter(pl.col("process") == GAS) if h is not None else None
        return float(g["value"][0]) if g is not None and g.height else 0.0

    def _prc(k):   # roll-forward commitment consumed by disp_roll_k
        p = steps[DISP.format(k=k)].flex_data.p_roll_continue_online
        g = p.frame.filter(pl.col("p") == GAS) if p is not None else None
        return float(g["value"][0]) if g is not None and g.height else None

    assert _prc(0) is None          # first roll: no prior dispatch
    for k in range(1, N_ROLLS):
        carried = _prc(k)
        prev_end = _reo(k - 1)
        assert carried is not None, f"disp roll {k} consumed no roll-forward"
        assert carried == pytest.approx(prev_end, rel=1e-6), (
            f"roll {k}: consumed {carried} != prev dispatch end {prev_end}"
        )
        assert carried > 0.0, (
            f"roll {k}: realized carry is zero (weight-0 garbage, not a real "
            f"dispatch trajectory)"
        )


@pytest.mark.solver
def test_all_rolls_optimal_and_realized_demand_met(chain):
    """All six solves are OPTIMAL, and the REALIZED DISPATCH rolls serve all
    demand — ZERO unserved load (no up-slack VOLL).

    Down-slack (spill) IS allowed on the dispatch rolls — the ``==`` hedge
    locks the committed units ON, and min-load over-production in a milder
    realized scenario spills via the node down-penalty (owner-confirmed
    expected behaviour).  The UC (planning) rolls are NOT checked for
    up-slack: their realized trunk branch carries operational weight 0 (the
    forecasts-only model), so it is intentionally undispatched — the realized
    demand is served by the dispatch sibling, which is exactly what these
    up-slack-free dispatch rolls prove."""
    steps, _ = chain
    for name, st in steps.items():
        assert st.optimal is True, f"{name} not optimal"
    for k in range(N_ROLLS):
        sol = steps[DISP.format(k=k)].solution
        if "vq_state_up" in sol._vars:
            up = sol.value("vq_state_up").filter(pl.col("value") > 1e-3)
            assert up.height == 0, (
                f"dispatch roll {k}: {up.height} unserved-load (up-slack) "
                f"rows — realized demand not met"
            )
