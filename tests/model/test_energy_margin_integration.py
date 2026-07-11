"""End-to-end gating test for the ``energy_margin`` demand-margin feature.

The feature (``flextool/engine_polars/_emit_energy_margin.py``, wired into
``_emit_solve_time.run`` right after ``emit_pdtNodeInflow``) multiplies a
node's ``pdtNodeInflow`` by its ``energy_margin`` factor, but ONLY:

* in the solve that carries investment periods
  (``bool(state.solve.invest_periods.get(solve_name))``); and
* on DEMAND rows — i.e. NEGATIVE net inflow (``value < 0``); FlexTool
  represents exogenous demand as negative inflow (``p_negative_inflow``
  is the exogenous-outflow term in the node balance, see
  ``_group_slack.py``).  Positive/zero (supply) rows are left untouched.

This module proves the gating end-to-end on the REAL cascade as a
BASELINE-vs-MARGIN diff — which pins the invest-only gate, the
positive-supply-untouched rule, AND the default byte-parity (every
non-target row is byte-identical) in one shot.

Fixture & target
----------------
``multi_fullYear_battery_nested_24h_invest_one_solve`` (the smallest
nested invest→dispatch chain; scenario in ``tests/fixtures/tests.json``).
Its invest sub-solve is ``invest_24h`` (``invest=True``); every
``storage_fullYear_6h`` / ``dispatch_fullYear_roll_roll_*`` sub-solve is a
dispatch solve (``invest=False``).  The demand node ``west`` carries an
all-negative ``pdtNodeInflow`` in the invest solve, so it is the natural
scaling target; ``battery`` / ``coal_market`` are 0.0 there.

Assertions are on the EMITTED ``pdtNodeInflow`` frame retained on each
sub-solve's :class:`FlexDataProvider` (``keep_solutions=True``), parsing
the repr-rendered value strings back to float.

The DB is built from the JSON fixture under ``tmp_path`` (never a
checked-in ``.sqlite``); the margin is set on ``west`` via the spinedb
``import_data`` API on the tmp DB (the JSON fixture is not edited).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

SCENARIO = "multi_fullYear_battery_nested_24h_invest_one_solve"
FIXTURE_JSON = _TESTS_DIR / "fixtures" / "tests.json"
TARGET = "west"
MARGIN = 1.5
INVEST_SOLVE = "invest_24h"
MARGIN_ALT = "west"  # active alternative that defines west's inflow
PDT_KEY = "solve_data/pdtNodeInflow"
KEYS = ("node", "period", "time")

# HiGHS-driving, multi-solve cascade → keep out of the fast inner loop.
pytestmark = [pytest.mark.solver, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Cascade drivers
# ---------------------------------------------------------------------------

def _build_db(tmp_path_factory, *, with_margin: bool) -> str:
    """Build a fresh SQLite from the JSON fixture; optionally set the
    ``energy_margin`` on ``west`` via the spinedb API (not the JSON)."""
    from flextool.update_flextool.db_migration import migrate_database

    root = tmp_path_factory.mktemp(
        "em_margin" if with_margin else "em_baseline"
    )
    db_path = root / "tests.sqlite"
    url = json_to_db(FIXTURE_JSON, db_path)
    migrate_database(url)

    if with_margin:
        from spinedb_api import DatabaseMapping, import_data

        with DatabaseMapping(url) as db:
            _count, errors = import_data(
                db,
                parameter_values=[
                    ("node", TARGET, "energy_margin_method", "inflow_multiplier",
                     MARGIN_ALT),
                    ("node", TARGET, "energy_margin", MARGIN, MARGIN_ALT),
                ],
            )
            assert not errors, f"import_data errors: {errors}"
            db.commit_session("set energy_margin on west")
    return url


def _run(url: str, work_folder: Path) -> dict:
    """Drive the full cascade, retaining per-step providers for the
    emitted-``pdtNodeInflow`` assertions."""
    from flextool.engine_polars import run_chain_from_db

    steps = run_chain_from_db(
        url, SCENARIO, work_folder=work_folder,
        csv_dump=True, keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for {SCENARIO!r}"
    return steps


def _inflow(step) -> pl.DataFrame:
    """The emitted ``pdtNodeInflow`` frame for a sub-solve, sorted on the
    entity/time keys so baseline and margin frames align row-for-row."""
    provider = step.flex_data_provider
    assert provider is not None, (
        "flex_data_provider unexpectedly None (keep_solutions=True should "
        "retain it)"
    )
    df = provider.get(PDT_KEY)
    assert df is not None and df.height > 0, (
        "provider has no non-empty 'solve_data/pdtNodeInflow'"
    )
    return df.sort(KEYS)


@pytest.fixture(scope="module")
def cascades(tmp_path_factory):
    """Run the baseline and margin cascades ONCE each for the module.

    Returns ``(baseline_steps, margin_steps)`` — each a
    ``complete_solve_name → OrchestrationStep`` dict with per-step
    providers retained.
    """
    base_url = _build_db(tmp_path_factory, with_margin=False)
    marg_url = _build_db(tmp_path_factory, with_margin=True)
    base_steps = _run(base_url, tmp_path_factory.mktemp("em_base_work"))
    marg_steps = _run(marg_url, tmp_path_factory.mktemp("em_marg_work"))
    # The two runs must share the same sub-solve chain (only value cells
    # of west in the invest solve may differ).
    assert list(base_steps) == list(marg_steps), (
        "baseline and margin cascades produced different sub-solve chains"
    )
    return base_steps, marg_steps


# ---------------------------------------------------------------------------
# (1) Invest solve: west demand rows scaled ×MARGIN; everything else intact
# ---------------------------------------------------------------------------

def test_invest_solve_scales_target_demand_only(cascades):
    """In ``invest_24h`` the target's NEGATIVE rows are baseline×MARGIN;
    zero rows are untouched; every other node is byte-identical."""
    base_steps, marg_steps = cascades
    base = _inflow(base_steps[INVEST_SOLVE])
    marg = _inflow(marg_steps[INVEST_SOLVE])

    assert base.select(KEYS).equals(marg.select(KEYS)), (
        "invest-solve pdtNodeInflow (node,period,time) keys diverged"
    )

    joined = base.join(
        marg.rename({"value": "value_m"}), on=list(KEYS), how="inner",
    ).with_columns(
        pl.col("value").cast(pl.Float64).alias("bf"),
        pl.col("value_m").cast(pl.Float64).alias("mf"),
    )
    assert joined.height == base.height, "join dropped rows"

    tgt = joined.filter(pl.col("node") == TARGET)
    assert tgt.height > 0, f"target node {TARGET!r} absent from invest solve"

    # Guard: the target genuinely carries demand (negative) rows — else
    # the gate would be vacuously satisfied.
    neg = tgt.filter(pl.col("bf") < 0.0)
    assert neg.height > 0, (
        f"{TARGET!r} has no negative (demand) inflow rows in the invest "
        f"solve — scaling gate would be vacuous"
    )
    # Negative demand rows scaled ×MARGIN (parse repr strings → float).
    prod = (neg["bf"] * MARGIN).to_list()
    got = neg["mf"].to_list()
    for b, g in zip(prod, got):
        assert abs(g - b) <= 1e-9 * abs(b), (
            f"invest {TARGET}: expected {b!r} (=baseline×{MARGIN}), got {g!r}"
        )

    # Any non-negative target rows (zero/positive supply) are untouched —
    # asserted as byte-identical value strings.
    nonneg = tgt.filter(pl.col("bf") >= 0.0)
    assert nonneg.filter(
        pl.col("value") != pl.col("value_m")
    ).height == 0, f"{TARGET!r} non-demand rows were altered"

    # Every OTHER node in the invest solve is byte-identical.
    others_base = base.filter(pl.col("node") != TARGET)
    others_marg = marg.filter(pl.col("node") != TARGET)
    assert others_base.equals(others_marg), (
        "non-target nodes changed in the invest solve — margin leaked"
    )


# ---------------------------------------------------------------------------
# (2) Dispatch solve: the target is byte-identical (margin did NOT fire)
# ---------------------------------------------------------------------------

def _dispatch_solve_names(steps) -> list[str]:
    return [k for k in steps if k != INVEST_SOLVE]


def test_dispatch_solve_target_byte_identical(cascades):
    """In a dispatch sub-solve the target's rows are byte-identical to
    baseline — the invest-only gate kept the margin out of dispatch."""
    base_steps, marg_steps = cascades
    dispatch = _dispatch_solve_names(base_steps)
    assert dispatch, "fixture produced no dispatch sub-solves"

    # Assert on a representative dispatch solve that actually carries the
    # target node, and confirm it holds for the first such solve.
    checked = None
    for name in dispatch:
        base = _inflow(base_steps[name])
        if base.filter(pl.col("node") == TARGET).height == 0:
            continue
        marg = _inflow(marg_steps[name])
        base_t = base.filter(pl.col("node") == TARGET)
        marg_t = marg.filter(pl.col("node") == TARGET)
        assert base_t.equals(marg_t), (
            f"dispatch solve {name!r}: target {TARGET!r} pdtNodeInflow "
            f"differs between baseline and margin — margin fired in a "
            f"dispatch solve (invest-only gate broken)"
        )
        checked = name
        break
    assert checked is not None, (
        f"no dispatch sub-solve carried target {TARGET!r} — cannot prove "
        f"the dispatch gate"
    )


# ---------------------------------------------------------------------------
# (3) Every OTHER (solve, node) pdtNodeInflow is byte-identical
# ---------------------------------------------------------------------------

def test_only_invest_target_changes(cascades):
    """Across ALL sub-solves, the ONLY difference between baseline and
    margin is the target's rows in the invest solve.  Every other
    (solve, node) pdtNodeInflow is byte-identical."""
    base_steps, marg_steps = cascades

    changed_solves: list[str] = []
    for name in base_steps:
        base = _inflow(base_steps[name])
        marg = _inflow(marg_steps[name])
        if name == INVEST_SOLVE:
            # Only the target may differ here.
            b_other = base.filter(pl.col("node") != TARGET)
            m_other = marg.filter(pl.col("node") != TARGET)
            assert b_other.equals(m_other), (
                f"invest solve: a non-target node changed ({name})"
            )
        else:
            if not base.equals(marg):
                changed_solves.append(name)

    assert not changed_solves, (
        f"margin leaked into non-invest solves: {changed_solves[:5]} "
        f"(count={len(changed_solves)})"
    )


# ---------------------------------------------------------------------------
# (4) AUTOSCALE_STRICT: the energy_margin PARAMETER_TYPES entries are
#     complete on a real solve (no silent revert to an un-scaled LP)
# ---------------------------------------------------------------------------

def test_autoscale_strict_completes_with_margin(
    tmp_path_factory, monkeypatch,
):
    """With ``FLEXTOOL_AUTOSCALE_STRICT=1`` a margin-carrying solve
    completes without the autoscale silent-revert — proving the
    ``('energy_margin','node')`` / ``('energy_margin_method','node')``
    PARAMETER_TYPES entries are registered (CLAUDE.md invariant #1).

    Under strict mode an autoscale-registry gap re-raises (see
    ``_orchestration._autoscale_layer2_for_solve``), so a clean,
    all-optimal run is the positive signal.
    """
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    url = _build_db(tmp_path_factory, with_margin=True)
    steps = _run(url, tmp_path_factory.mktemp("em_strict_work"))
    # Every sub-solve reached an optimal LP — the scaled path did not
    # silently revert nor raise on a registry gap.
    non_optimal = [k for k, s in steps.items() if not s.optimal]
    assert not non_optimal, (
        f"AUTOSCALE_STRICT: sub-solves not optimal: {non_optimal[:5]}"
    )
    # And the margin genuinely reached the invest solve's emitted frame.
    base = _inflow(steps[INVEST_SOLVE]).filter(pl.col("node") == TARGET)
    assert base.filter(
        pl.col("value").cast(pl.Float64) < 0.0
    ).height > 0, "invest solve carried no scaled target demand rows"
