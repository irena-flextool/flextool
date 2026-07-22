"""End-to-end gating test for the ADDITIVE ``energy_margin_adder`` feature.

Sibling to ``test_energy_margin_integration.py`` (the multiplier).  The
adder (``flextool/engine_polars/_emit_energy_margin_adder.py``, wired into
``_emit_solve_time.run`` right AFTER the multiplier) applies, ONLY in the
solve that carries investment periods and ONLY to nodes whose
``energy_margin_method == 'inflow_adder'``:

* an EXISTING ``pdtNodeInflow`` row of the node → ``value - adder``
  (adding demand ⇒ MORE-negative inflow, since FlexTool represents
  exogenous demand as NEGATIVE inflow); and
* a MISSING ``(node, d, t)`` row → a CREATED row with value ``-adder``.

Why a REAL cascade test (the emit unit test is not enough)
---------------------------------------------------------
The sibling MULTIPLIER once shipped with an INVERTED sign that passed
synthetic positive-demand unit tests and was only caught by a real
negative-demand run.  The adder emitter's synthetic parity tests live in
``tests/engine_polars/test_energy_margin_adder_emit.py``; this module
proves the SIGN and the ROW-CREATION on real HiGHS cascade solves, where a
sign inversion would REDUCE demand / lower cost instead of deepening it.

Fixture & targets
-----------------
Same fixture as the multiplier test:
``multi_fullYear_battery_nested_24h_invest_one_solve`` (the smallest nested
invest→dispatch chain; scenario in ``tests/fixtures/tests.json``).  Its
invest sub-solve is ``invest_24h`` (``invest=True``); every
``storage_fullYear_6h`` / ``dispatch_fullYear_roll_roll_*`` sub-solve is a
dispatch solve (``invest=False``).

* **Case 1 — deepen an existing negative-demand node.**  ``west`` carries
  an all-negative ``pdtNodeInflow`` in the invest solve (12 rows,
  ~ -16.5k … -29.3k MWh).  Setting ``inflow_adder`` + a modest adder on
  ``west`` must (a) deepen EVERY demand row by exactly the adder (correct
  sign), (b) raise the invest solve's objective — the build/serve response
  to more demand — while staying optimal, and (c) leave a DISPATCH
  sub-solve's ``west`` inflow byte-identical (the invest-only gate).

* **Case 2 — CREATE rows on a genuinely zero-inflow node.**  The
  row-creation path fires only for a node the inflow derive EXCLUDES from
  ``pdtNodeInflow`` entirely, which happens ONLY when the node's
  ``inflow_method == 'no_inflow'`` (default ``use_original`` still floors a
  balance node to a 0.0 row, so it is NOT a missing row).  No shipped
  fixture carries an explicit ``no_inflow`` node, so — using the SAME
  tmp-DB ``import_data`` reconfiguration the multiplier test uses to inject
  its margin (NO JSON edit, NO new scenario) — we mark the real balance
  node ``battery`` ``no_inflow``.  Adder-OFF, ``battery`` then has NO inflow
  row; adder-ON, the emitter must CREATE one ``-adder`` demand row per
  invest ``(d, t)``, the solve must stay optimal, and (since ``battery`` is
  a real ``nodeBalance`` node) the served extra demand must raise cost.

The DB is built from the JSON fixture under ``tmp_path`` (never a
checked-in ``.sqlite``); every parameter is set on the tmp DB via the
spinedb ``import_data`` API (the JSON fixture is not edited).
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
INVEST_SOLVE = "invest_24h"
ALT = "west"  # an alternative active in the invest scenario (see multiplier test)
PDT_KEY = "solve_data/pdtNodeInflow"
KEYS = ("node", "period", "time")

TARGET_C1 = "west"       # existing all-negative demand node → DEEPEN
ADDER_WEST = 2000.0      # modest, per-timestep MWh added to west
TARGET_C2 = "battery"    # real nodeBalance node, marked no_inflow → CREATE
ADDER_BATT = 500.0       # modest, per-timestep MWh created on battery

# HiGHS-driving, multi-solve cascade → keep out of the fast inner loop.
pytestmark = [pytest.mark.solver, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Cascade drivers
# ---------------------------------------------------------------------------

def _build_db(
    tmp_path_factory,
    tag: str,
    *,
    adder_node: str | None = None,
    adder_value: float | None = None,
    no_inflow_node: str | None = None,
) -> str:
    """Build a fresh SQLite from the JSON fixture; optionally mark a node
    ``no_inflow`` and/or set the additive ``inflow_adder`` margin on a node
    via the spinedb API (never editing the JSON)."""
    from flextool.update_flextool.db_migration import migrate_database

    root = tmp_path_factory.mktemp(tag)
    url = json_to_db(FIXTURE_JSON, root / "tests.sqlite")
    migrate_database(url)

    pvs: list[tuple] = []
    if no_inflow_node is not None:
        # Turn a real balance node into a genuinely zero-inflow node (its
        # inflow row is DROPPED from pdtNodeInflow) — the only real-model
        # precondition under which the adder's row-CREATION path fires.
        pvs.append(("node", no_inflow_node, "inflow_method", "no_inflow", ALT))
    if adder_node is not None:
        pvs.append(
            ("node", adder_node, "energy_margin_method", "inflow_adder", ALT)
        )
        pvs.append(
            ("node", adder_node, "energy_margin_adder", adder_value, ALT)
        )
    if pvs:
        from spinedb_api import DatabaseMapping, import_data

        with DatabaseMapping(url) as db:
            _count, errors = import_data(db, parameter_values=pvs)
            assert not errors, f"import_data errors: {errors}"
            db.commit_session(f"energy_margin_adder integration: {tag}")
    return url


# Period-Map adder for the end-to-end map case.  ``invest_24h`` carries
# invest periods p2020/p2025/p2030/p2035; a distinct per-period value proves
# the Map's period axis survives DB→ingestion→emit→solve at per-period
# granularity (a flattened/dropped axis would apply one value or none).
MAP_PERIOD_VALUES = {
    "p2020": 1000.0,
    "p2025": 1500.0,
    "p2030": 2000.0,
    "p2035": 2500.0,
}


def _build_db_period_map(tmp_path_factory, tag: str) -> str:
    """Build a fresh SQLite from the JSON fixture with a period (1d) Map
    ``energy_margin_adder`` (+ ``inflow_adder`` method) on ``TARGET_C1``.

    Exercises the full ingestion wiring: the ``["1d_map"]`` spec added in
    ``_specs.py`` routes the Map into ``input/pd_energy_margin_adder`` and
    the emitter broadcasts each period's value over that period's (d, t)."""
    from spinedb_api import DatabaseMapping, Map, import_data

    from flextool.update_flextool.db_migration import migrate_database

    root = tmp_path_factory.mktemp(tag)
    url = json_to_db(FIXTURE_JSON, root / "tests.sqlite")
    migrate_database(url)

    periods = list(MAP_PERIOD_VALUES)
    value = Map(
        periods, [MAP_PERIOD_VALUES[p] for p in periods], index_name="period",
    )
    with DatabaseMapping(url) as db:
        _count, errors = import_data(
            db,
            parameter_values=[
                ("node", TARGET_C1, "energy_margin_adder", value, ALT),
                ("node", TARGET_C1, "energy_margin_method", "inflow_adder", ALT),
            ],
        )
        assert not errors, f"import_data errors: {errors}"
        db.commit_session(f"energy_margin_adder period-map: {tag}")
    return url


def _run(url: str, work_folder: Path) -> dict:
    """Drive the full cascade, retaining per-step providers for the
    emitted-``pdtNodeInflow`` assertions and per-step objectives."""
    from flextool.engine_polars import run_chain_from_db

    steps = run_chain_from_db(
        url, SCENARIO, work_folder=work_folder,
        csv_dump=True, keep_solutions=True,
    )
    assert steps, f"run_chain_from_db returned no steps for {SCENARIO!r}"
    return steps


def _inflow(step) -> pl.DataFrame:
    """The emitted ``pdtNodeInflow`` frame for a sub-solve, sorted on the
    entity/time keys so two runs align row-for-row."""
    provider = step.flex_data_provider
    assert provider is not None, (
        "flex_data_provider unexpectedly None (keep_solutions=True should "
        "retain it)"
    )
    df = provider.get(PDT_KEY)
    assert df is not None, "provider has no 'solve_data/pdtNodeInflow'"
    return df.sort(KEYS)


# ---------------------------------------------------------------------------
# Fixtures — each cascade pair is consumed by exactly ONE test, so the
# module-scoped fixture runs its (2 × ~35s) cascades once even under xdist.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case1_cascades(tmp_path_factory):
    """Baseline vs. west-adder cascades (existing negative-demand node)."""
    base = _build_db(tmp_path_factory, "adder_c1_base")
    adder = _build_db(
        tmp_path_factory, "adder_c1_on",
        adder_node=TARGET_C1, adder_value=ADDER_WEST,
    )
    base_steps = _run(base, tmp_path_factory.mktemp("c1_base_work"))
    adder_steps = _run(adder, tmp_path_factory.mktemp("c1_on_work"))
    assert list(base_steps) == list(adder_steps), (
        "baseline and adder cascades produced different sub-solve chains"
    )
    return base_steps, adder_steps


@pytest.fixture(scope="module")
def period_map_cascades(tmp_path_factory):
    """Baseline vs. period-Map-adder cascades (existing negative-demand
    node, per-period Map value)."""
    base = _build_db(tmp_path_factory, "adder_map_base")
    adder = _build_db_period_map(tmp_path_factory, "adder_map_on")
    base_steps = _run(base, tmp_path_factory.mktemp("map_base_work"))
    adder_steps = _run(adder, tmp_path_factory.mktemp("map_on_work"))
    assert list(base_steps) == list(adder_steps), (
        "baseline and period-map cascades produced different sub-solve chains"
    )
    return base_steps, adder_steps


@pytest.fixture(scope="module")
def case2_cascades(tmp_path_factory):
    """Baseline vs. battery-adder cascades, with ``battery`` marked
    ``no_inflow`` in BOTH so adder-off it has no inflow row at all."""
    base = _build_db(
        tmp_path_factory, "adder_c2_base", no_inflow_node=TARGET_C2,
    )
    adder = _build_db(
        tmp_path_factory, "adder_c2_on",
        no_inflow_node=TARGET_C2, adder_node=TARGET_C2, adder_value=ADDER_BATT,
    )
    base_steps = _run(base, tmp_path_factory.mktemp("c2_base_work"))
    adder_steps = _run(adder, tmp_path_factory.mktemp("c2_on_work"))
    assert list(base_steps) == list(adder_steps), (
        "baseline and adder cascades produced different sub-solve chains"
    )
    return base_steps, adder_steps


# ---------------------------------------------------------------------------
# Case 1 — deepen an existing negative-demand node (sign + build + gate)
# ---------------------------------------------------------------------------

def test_case1_west_demand_deepened_builds_and_gated(case1_cascades):
    """In ``invest_24h`` the adder DEEPENS every ``west`` demand row by
    exactly the adder (correct sign), raises the objective (build/serve
    response) while staying optimal, leaves every other node byte-identical,
    and does NOT touch a dispatch sub-solve (invest-only gate)."""
    base_steps, adder_steps = case1_cascades
    base = _inflow(base_steps[INVEST_SOLVE])
    adder = _inflow(adder_steps[INVEST_SOLVE])

    # The two runs share the same (node, period, time) keys in the invest
    # solve (west has a full grid; the adder deepens in place, creates none).
    assert base.select(KEYS).equals(adder.select(KEYS)), (
        "invest-solve pdtNodeInflow (node,period,time) keys diverged"
    )
    joined = base.join(
        adder.rename({"value": "value_a"}), on=list(KEYS), how="inner",
    ).with_columns(
        pl.col("value").cast(pl.Float64).alias("bf"),
        pl.col("value_a").cast(pl.Float64).alias("af"),
    )
    assert joined.height == base.height, "join dropped rows"

    tgt = joined.filter(pl.col("node") == TARGET_C1)
    neg = tgt.filter(pl.col("bf") < 0.0)
    assert neg.height > 0, (
        f"{TARGET_C1!r} has no negative (demand) inflow rows in the invest "
        f"solve — the deepen assertion would be vacuous"
    )

    # (a) SIGN: every demand row is DEEPER (more negative) by exactly the
    #     adder.  ``af < bf`` catches the inverted sign (which would REDUCE
    #     demand); the exact-delta pins the magnitude.
    for b, a in zip(neg["bf"].to_list(), neg["af"].to_list()):
        assert a < b, (
            f"adder REDUCED demand ({a!r} !< {b!r}) — inverted sign: adding "
            f"demand must make inflow MORE negative"
        )
        assert abs(a - (b - ADDER_WEST)) <= 1e-6 * max(1.0, abs(b)), (
            f"west demand not deepened by exactly {ADDER_WEST}: "
            f"base={b!r} adder={a!r} (Δ={b - a!r})"
        )

    # Every OTHER node in the invest solve is byte-identical — the margin
    # did not leak (pdtNodeInflow is an input-derived RHS constant).
    ob = base.filter(pl.col("node") != TARGET_C1)
    oa = adder.filter(pl.col("node") != TARGET_C1)
    assert ob.equals(oa), "non-target nodes changed in the invest solve"

    # (b) BUILD response: strictly higher invest objective, both optimal.
    b_obj = base_steps[INVEST_SOLVE].obj
    a_obj = adder_steps[INVEST_SOLVE].obj
    assert base_steps[INVEST_SOLVE].optimal, "baseline invest solve not optimal"
    assert adder_steps[INVEST_SOLVE].optimal, "adder invest solve not optimal"
    assert b_obj is not None and a_obj is not None
    assert a_obj > b_obj + 1.0, (
        f"invest objective did not rise with the extra demand: "
        f"base={b_obj!r} adder={a_obj!r} — no build/serve response"
    )

    # (c) DISPATCH gate: the first dispatch sub-solve carrying west is
    #     byte-identical — the invest-only gate kept the adder out.
    checked = None
    for name in (k for k in base_steps if k != INVEST_SOLVE):
        db_t = _inflow(base_steps[name]).filter(pl.col("node") == TARGET_C1)
        if db_t.height == 0:
            continue
        da_t = _inflow(adder_steps[name]).filter(pl.col("node") == TARGET_C1)
        assert db_t.equals(da_t), (
            f"dispatch solve {name!r}: {TARGET_C1!r} pdtNodeInflow differs "
            f"between baseline and adder — the adder fired in a dispatch "
            f"solve (invest-only gate broken)"
        )
        checked = name
        break
    assert checked is not None, (
        f"no dispatch sub-solve carried {TARGET_C1!r} — cannot prove gate"
    )


# ---------------------------------------------------------------------------
# Period-Map case — a per-period Map adder round-trips end-to-end and
# deepens west's invest demand by that period's value (not a flat scalar)
# ---------------------------------------------------------------------------

def test_period_map_adder_deepens_per_period(period_map_cascades):
    """A period (1d) Map ``energy_margin_adder`` survives DB → ingestion →
    emit → solve: in ``invest_24h`` every ``west`` demand row in period
    ``pX`` is deepened by exactly that period's Map value (distinct per
    period), no other node is touched, and the objective rises.

    This is the end-to-end guard for the ingestion fix: a mangled Map (the
    pre-fix behaviour) would deepen by one flat value or nothing at all, so
    the DISTINCT per-period deltas below are the load-bearing assertion."""
    base_steps, adder_steps = period_map_cascades
    base = _inflow(base_steps[INVEST_SOLVE])
    adder = _inflow(adder_steps[INVEST_SOLVE])

    assert base.select(KEYS).equals(adder.select(KEYS)), (
        "invest-solve pdtNodeInflow (node,period,time) keys diverged"
    )
    joined = base.join(
        adder.rename({"value": "value_a"}), on=list(KEYS), how="inner",
    ).with_columns(
        pl.col("value").cast(pl.Float64).alias("bf"),
        pl.col("value_a").cast(pl.Float64).alias("af"),
    )
    tgt = joined.filter(pl.col("node") == TARGET_C1)
    assert tgt.height > 0, f"{TARGET_C1!r} carried no invest-solve inflow rows"

    # At least two DISTINCT periods must be present on the target so the
    # per-period distinction is actually exercised (not vacuously one value).
    tgt_periods = set(tgt["period"].to_list())
    covered = tgt_periods & set(MAP_PERIOD_VALUES)
    assert len(covered) >= 2, (
        f"{TARGET_C1!r} invest rows span < 2 mapped periods ({tgt_periods!r}); "
        f"the per-period Map distinction would be vacuous"
    )

    # Every target demand row is deepened by EXACTLY its own period's value.
    for row in tgt.iter_rows(named=True):
        period = row["period"]
        b, a = row["bf"], row["af"]
        assert period in MAP_PERIOD_VALUES, (
            f"unexpected invest period {period!r} not in the authored Map"
        )
        delta = MAP_PERIOD_VALUES[period]
        assert a < b, (
            f"period {period!r}: adder REDUCED demand ({a!r} !< {b!r}) — "
            f"inverted sign"
        )
        assert abs(a - (b - delta)) <= 1e-6 * max(1.0, abs(b)), (
            f"period {period!r}: west demand not deepened by that period's "
            f"Map value {delta}: base={b!r} adder={a!r} (Δ={b - a!r})"
        )

    # No other node changed (the Map RHS did not leak).
    ob = base.filter(pl.col("node") != TARGET_C1)
    oa = adder.filter(pl.col("node") != TARGET_C1)
    assert ob.equals(oa), "period-map adder leaked onto non-target nodes"

    # Build response: strictly higher invest objective, both optimal.
    assert base_steps[INVEST_SOLVE].optimal, "baseline invest solve not optimal"
    assert adder_steps[INVEST_SOLVE].optimal, "adder invest solve not optimal"
    b_obj = base_steps[INVEST_SOLVE].obj
    a_obj = adder_steps[INVEST_SOLVE].obj
    assert b_obj is not None and a_obj is not None
    assert a_obj > b_obj + 1.0, (
        f"invest objective did not rise with the per-period Map demand: "
        f"base={b_obj!r} adder={a_obj!r}"
    )


# ---------------------------------------------------------------------------
# Case 2 — CREATE demand rows on a genuinely zero-inflow node
# ---------------------------------------------------------------------------

def test_case2_creates_rows_on_zero_inflow_node(case2_cascades):
    """With ``battery`` marked ``no_inflow`` it has NO inflow row adder-off;
    adder-on the emitter CREATES one ``-adder`` demand row per invest
    ``(d, t)``, the solve stays optimal, the served demand raises cost, and
    no other node is touched."""
    base_steps, adder_steps = case2_cascades
    base = _inflow(base_steps[INVEST_SOLVE])
    adder = _inflow(adder_steps[INVEST_SOLVE])

    # Precondition: adder-off the zero-inflow node has NO inflow row at all
    # (the row-creation path is otherwise unreachable and the test vacuous).
    base_b = base.filter(pl.col("node") == TARGET_C2)
    assert base_b.height == 0, (
        f"{TARGET_C2!r} unexpectedly carried {base_b.height} inflow row(s) "
        f"adder-off — the row-creation precondition (no_inflow ⇒ no row) is "
        f"not met, so row-creation cannot be proven"
    )

    # Adder-on: one CREATED negative-demand row per invest (d, t).
    adder_b = adder.filter(pl.col("node") == TARGET_C2)
    assert adder_b.height > 0, (
        f"adder created NO rows on the zero-inflow node {TARGET_C2!r}"
    )
    grid = adder.select(["period", "time"]).unique().height
    assert adder_b.height == grid, (
        f"{TARGET_C2!r} got {adder_b.height} created rows; expected one per "
        f"invest (d, t) cell = {grid}"
    )
    vals = (
        adder_b.with_columns(pl.col("value").cast(pl.Float64))["value"].to_list()
    )
    for v in vals:
        assert v < 0.0, (
            f"created row is not negative demand ({v!r}) — inverted sign: "
            f"adding demand must create NEGATIVE inflow"
        )
        assert abs(v - (-ADDER_BATT)) <= 1e-6 * ADDER_BATT, (
            f"created value {v!r} != -adder ({-ADDER_BATT!r})"
        )

    # The created rows are the ONLY change: every other node is identical.
    ob = base.filter(pl.col("node") != TARGET_C2)
    oa = adder.filter(pl.col("node") != TARGET_C2)
    assert ob.equals(oa), (
        "adder leaked onto non-target nodes in the invest solve"
    )

    # Solve stays optimal and the extra served demand raises cost.
    assert base_steps[INVEST_SOLVE].optimal, "baseline invest solve not optimal"
    assert adder_steps[INVEST_SOLVE].optimal, "adder invest solve not optimal"
    b_obj = base_steps[INVEST_SOLVE].obj
    a_obj = adder_steps[INVEST_SOLVE].obj
    assert b_obj is not None and a_obj is not None
    assert a_obj > b_obj + 1.0, (
        f"invest objective did not rise with the created demand on "
        f"{TARGET_C2!r}: base={b_obj!r} adder={a_obj!r}"
    )


# ---------------------------------------------------------------------------
# AUTOSCALE_STRICT: the energy_margin_adder PARAMETER_TYPES entry is
# complete on a real solve (no silent revert to an un-scaled LP)
# ---------------------------------------------------------------------------

def test_autoscale_strict_registers_adder_param(tmp_path_factory, monkeypatch):
    """With ``FLEXTOOL_AUTOSCALE_STRICT=1`` an adder-carrying solve completes
    without the autoscale silent-revert — proving the
    ``('energy_margin_adder', 'node')`` PARAMETER_TYPES entry is registered
    (CLAUDE.md invariant #1).  Under strict mode a registry gap re-raises,
    so a clean all-optimal run with the adder present is the positive
    signal."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    url = _build_db(
        tmp_path_factory, "adder_strict",
        adder_node=TARGET_C1, adder_value=ADDER_WEST,
    )
    steps = _run(url, tmp_path_factory.mktemp("adder_strict_work"))
    non_optimal = [k for k, s in steps.items() if not s.optimal]
    assert not non_optimal, (
        f"AUTOSCALE_STRICT: sub-solves not optimal: {non_optimal[:5]}"
    )
    # And the adder genuinely reached the invest solve's emitted frame.
    tgt = (
        _inflow(steps[INVEST_SOLVE])
        .filter(pl.col("node") == TARGET_C1)
        .with_columns(pl.col("value").cast(pl.Float64))
    )
    assert tgt.filter(pl.col("value") < 0.0).height > 0, (
        "invest solve carried no deepened target demand rows under strict mode"
    )
