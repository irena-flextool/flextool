"""DB-direct solve / overlay regression tests.

This module hosts the survivors from the retired
``tests/engine/test_db_direct_parity.py``.  That file was Phase H
migration scaffolding whose bulk verified CSV-loaded FlexData equals
DB-direct-loaded FlexData (via ``_load_pair``); the migration is long
established and those CSV-vs-DB comparisons no longer pull their weight.

The tests salvaged here have ongoing value:

* ``test_db_direct_solve_parity_with_derived_a..g`` and
  ``test_db_direct_solve_parity_with_projections`` — end-to-end LP solve
  through the production loader call (``tests._support.load.load_scenario``:
  post-solve provider + DB reader) checked against the recorded
  ``v_obj__*.parquet`` objective.  Guards the DB-direct overlay
  cascade against silent value regressions.

* ``test_inmemory_reader_*`` — unit tests for the
  :class:`InMemoryReader` ``InputSource`` Protocol implementation.

* ``test_load_flextool_*`` — edge cases for the source-aware
  ``load_flextool`` entry point.

* ``test_resolved_default_landed`` — v56 None-default policy regression
  matrix: each engine-substituted default must surface as ``None`` via
  ``SpineDbReader.parameter_default`` (engine-side fallback is the
  source of truth).

* A handful of singletons (``test_p_entity_max_units_canonical_test_a_lot``,
  ``test_e_invest_total_method_filter_test_a_lot``,
  ``test_p_entity_max_units_invest_no_limit_y2020_2029_1x10y``,
  ``test_lifetime_no_investment_filters_ed_invest_set``,
  ``test_lifetime_choice_truncates_p_entity_all_existing``,
  ``test_multi_solve_handoff_p_entity_all_existing_chain``,
  ``test_dispatch_only_gate_blanks_invest_cascade``) that check the
  field of a rebuild from the post-solve provider against the FlexData
  the cascade itself built (``run.last_flex_data``) and pin literal
  expected values for invest-cascade / lifetime-cascade behaviour.

Each test runs its scenario's cascade through the session-cached
``scenario_run`` factory (``tests/conftest.py``), built from the
JSON-backed SpineDB fixtures.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine import InMemoryReader, load_flextool
from flextool.engine.preprocess.params import _direct as dp
from polar_high import Param
from tests._support.fields import field_diff
from tests._support.load import clone_provider, load_scenario


# Scenarios live across three JSON fixtures (see ``tests/conftest.py``
# session DB fixtures).  Map each referenced scenario to the
# ``db_fixture`` key ``scenario_run`` expects.  Everything else
# defaults to ``"main"`` (``tests/fixtures/tests.json``).
_DB_FIXTURE_BY_SCENARIO: dict[str, str] = {
    "lh2_three_region": "lh2",
    "2_day_stochastic_dispatch": "stochastic",
}


def _db_fixture_for(scenario: str) -> str:
    return _DB_FIXTURE_BY_SCENARIO.get(scenario, "main")



def _load_solve_of(run, parquet: Path):
    """The FlexData of the solve whose objective ``parquet`` (``v_obj__<solve>.parquet``) holds: the last step
    unless the run has a step of that name (a stochastic solve followed by its realization solve)."""
    name = parquet.stem.removeprefix("v_obj__")
    return load_scenario(run, step=name) if name in {s.solve_name for s in run.steps} else load_scenario(run)

# ---------------------------------------------------------------------------
# Helpers


def _rebuild(scenario_run, scenario: str):
    """``(run, rebuilt FlexData, cascade FlexData)`` for the scenario's last step.

    The rebuild is the production loader call on the run's post-solve provider (``load_scenario``); the
    cascade FlexData is the one the cascade itself built and solved (``run.last_flex_data``).
    """
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    assert run.last_flex_data is not None, f"{scenario}: the cascade kept no FlexData"
    return run, load_scenario(run), run.last_flex_data


def _assert_rebuild_equal(data, cascade, field: str, scenario: str) -> None:
    diff = field_diff(getattr(data, field), getattr(cascade, field))
    assert diff is None, f"{field}: rebuild != cascade on {scenario}: {diff}"


# ---------------------------------------------------------------------------
# Singletons — invest / lifetime cascade literal-value regression guards
#
# Each test checks the field of a rebuild from the post-solve provider against the cascade's own
# FlexData (these used to compare a CSV load with a DB-direct load of the same, DB-derived snapshot,
# which could no longer fail), plus the literal values the docstring pins.


def test_p_entity_max_units_canonical_test_a_lot(scenario_run):
    """`p_entity_max_units` mirrors flextool's
    ``entity_period_calc_params.py:1718-1761`` (max_capacity / unitsize
    per entity × period_in_use) on ``test_a_lot`` after the Γ.6.C rewrite.

    Pre-fix: DB-direct emitted only entities with explicit
    ``invest_max_period`` (≈3 rows), CSV path had 56.  Post-fix:
    full per-(entity, period) coverage including the
    ``existing + invest_no_limit`` blanket cap and unitsize cascade.
    """
    _, data, cascade = _rebuild(scenario_run, "test_a_lot")
    assert data.p_entity_max_units is not None
    _assert_rebuild_equal(data, cascade, "p_entity_max_units", "test_a_lot")


def test_e_invest_total_method_filter_test_a_lot(scenario_run):
    """`e_invest_total` filters ``entity__invest_method`` by the four
    INVEST_TOTAL enum values (``invest_total / invest_period_total /
    invest_retire_total / invest_retire_period_total``) — NOT by
    ``invest_max_total > 0`` as the pre-Γ.6.C projection did.  On
    ``test_a_lot`` every entity uses ``invest_no_limit``, so
    ``e_invest_total`` is empty.
    """
    _, data, cascade = _rebuild(scenario_run, "test_a_lot")
    _assert_rebuild_equal(data, cascade, "e_invest_total", "test_a_lot")


def test_p_entity_max_units_invest_no_limit_y2020_2029_1x10y(scenario_run):
    """Regression for the ``invest_no_limit`` blanket-cap branch.

    On ``y2020_2029_1x10y``:
      * ``coal_plant`` uses ``invest_method=invest_total`` →
        max_capacity = existing(100) + invest_max_total(700) = 800.
      * ``wind_plant`` uses ``invest_method=invest_no_limit`` →
        max_capacity = existing(1000) + p_unconstrained_flow_cap(1e6)
        = 1001000.
      * Both divided by their unitsize cascade values
        (100, 1000) yield 8.0 and 1001.0 — pinned in this test.
    """
    _, data, cascade = _rebuild(scenario_run, "y2020_2029_1x10y")
    assert data.p_entity_max_units is not None
    _assert_rebuild_equal(data, cascade, "p_entity_max_units", "y2020_2029_1x10y")
    # Pin exact values for the coal_plant / wind_plant rows.
    db_rows = {(e, d): v
                for e, d, v in data.p_entity_max_units.frame.iter_rows()}
    assert db_rows[("coal_plant", "p2020")] == pytest.approx(8.0)
    assert db_rows[("wind_plant", "p2020")] == pytest.approx(1001.0)


def test_lifetime_no_investment_filters_ed_invest_set(scenario_run):
    """Γ.6.D — ``lifetime_method=no_investment`` entities whose lifetime
    window has expired must be anti-joined out of ``ed_invest_set``.

    Regression for ``multi_year_wind_no_investment``: ``wind_plant``
    has ``lifetime=10`` and ``no_investment``; with periods at years
    [0, 5, 10, 15] we need ed_invest to drop ``(wind_plant, p2030)``
    and ``(wind_plant, p2035)`` (they move to
    ``ed_invest_forbidden_no_investment``).
    """
    _, data, cascade = _rebuild(scenario_run, "multi_year_wind_no_investment")
    for field in ("ed_invest_set", "ed_invest_forbidden_no_investment"):
        _assert_rebuild_equal(data, cascade, field, "multi_year_wind_no_investment")
    db_set = data.ed_invest_set.sort("e", "d")
    # Specifically, wind_plant must NOT appear at p2030 / p2035.
    forbidden_rows = db_set.filter(
        (pl.col("e") == "wind_plant")
        & pl.col("d").is_in(["p2030", "p2035"]))
    assert forbidden_rows.height == 0, (
        f"wind_plant unexpectedly present in ed_invest at expired "
        f"periods: {forbidden_rows}")
    # ed_invest_forbidden_no_investment should hold those rows.
    forb = data.ed_invest_forbidden_no_investment
    assert forb is not None and forb.height == 2
    forb_set = set(forb.iter_rows())
    assert ("wind_plant", "p2030") in forb_set
    assert ("wind_plant", "p2035") in forb_set


def test_lifetime_choice_truncates_p_entity_all_existing(scenario_run):
    """Γ.6.D — ``lifetime_method=reinvest_choice`` truncates
    ``p_entity_all_existing`` past the lifetime expiry.

    Regression for ``wind_battery_invest_lifetime_choice``:
    ``wind_plant`` has ``lifetime=5`` and ``reinvest_choice``; existing
    capacity is 1000 at p2020 (yr=0) but must be 0 at p2025 (yr=5)
    and beyond.  This cascades through ``p_state_existing_capacity``,
    ``p_process_existing_count``, ``p_flow_upper_existing``.
    """
    _, data, cascade = _rebuild(scenario_run, "wind_battery_invest_lifetime_choice")
    _assert_rebuild_equal(data, cascade, "p_entity_all_existing", "wind_battery_invest_lifetime_choice")
    db_pae = data.p_entity_all_existing.frame.sort("e", "d")
    # Spot-check wind_plant at expired periods.
    wp = db_pae.filter(pl.col("e") == "wind_plant").sort("d")
    wp_dict = {row["d"]: row["value"] for row in wp.iter_rows(named=True)}
    assert wp_dict.get("p2020", 0.0) > 0.0
    for d in ("p2025", "p2030", "p2035"):
        assert wp_dict.get(d, 0.0) == 0.0, (
            f"wind_plant at {d} expected 0, got {wp_dict.get(d)}")


def test_multi_solve_handoff_p_entity_all_existing_chain(scenario_run):
    """Γ.6.D — multi-solve handoff: the last sub-solve's
    ``p_entity_all_existing`` carries the chained existing capacity from
    prior solves (the post-solve provider's handoff frames), not the raw
    ``entity.existing`` (which represents only the first solve's
    pre-existing capacity).

    Regression for ``wind_battery_invest_lifetime_renew_4solve``
    sub-solve y2035_5week (the 4th in the chain): ``battery`` arrives
    at p2035 with cumulative capacity ≈ 850, ``wind_plant`` ≈ 2289.7
    — neither value is in ``entity.existing`` (raw values are 50 and
    1000 respectively).
    """
    scenario = "wind_battery_invest_lifetime_renew_4solve"
    _, data, cascade = _rebuild(scenario_run, scenario)
    _assert_rebuild_equal(data, cascade, "p_entity_all_existing", scenario)
    db_pae = data.p_entity_all_existing.frame.sort("e", "d")
    # The chained values must be read, not raw entity.existing.
    rows = {(r["e"], r["d"]): r["value"]
             for r in db_pae.iter_rows(named=True)}
    assert rows.get(("battery", "p2035"), 0) > 800, (
        "battery at p2035 should be the chained ~850, not raw 50")
    assert rows.get(("wind_plant", "p2035"), 0) > 2000, (
        "wind_plant at p2035 should be the chained ~2289, not raw 1000")


def test_dispatch_only_gate_blanks_invest_cascade(scenario_run):
    """When the active solve has no ``invest_periods`` (dispatch-only),
    `_load_invest` returns blank — every invest-cascade Param is None.
    The DB-direct path's Γ.3.C overlay must mirror that: clear the
    overlays emitted by Γ.1 / Γ.2 so dispatch-only solves don't
    accidentally activate invest constraints.

    ``5weeks_invest_fullYear_dispatch_coal_wind`` is the
    canonical regression: it carries an
    ``invest_max_total = 700`` for ``coal_plant`` in the DB, but the
    active ``y2020_fullYear_dispatch`` solve has empty
    ``invest_periods`` so ed_invest is empty.  The load must emit
    ``e_invest_max_total = None`` — not a 1-row frame.
    """
    _, data, cascade = _rebuild(scenario_run, "5weeks_invest_fullYear_dispatch_coal_wind")
    # Every invest cascade field is None on dispatch-only, in the rebuild and in the cascade's FlexData.
    for field in ("e_invest_total", "e_invest_max_total",
                   "e_invest_min_total", "p_entity_max_units",
                   "ed_invest_set"):
        assert getattr(data, field) is None, f"{field} rebuild unexpectedly non-None"
        assert getattr(cascade, field) is None, f"{field} cascade unexpectedly non-None"


# ---------------------------------------------------------------------------
# InMemoryReader unit tests — Protocol exercise without sqlite


def test_inmemory_reader_basic_lookup():
    src = InMemoryReader(
        entities={
            "node": pl.DataFrame({"name": ["n1", "n2", "n3"]}),
            "commodity__node": pl.DataFrame(
                {"commodity": ["coal"], "node": ["n1"]}),
        },
        parameters={
            ("node", "penalty_up"): pl.DataFrame(
                {"name": ["n1", "n2"], "value": [100.0, 200.0]}),
            ("node", "node_type"): pl.DataFrame(
                {"name": ["n1"], "value": ["balance"]}),
        },
        defaults={
            ("node", "penalty_up"): 10000.0,
            ("node", "node_type"): "balance",
        },
    )
    # entities lookup
    assert src.entities("node").height == 3
    assert src.entities("commodity__node").columns == ["commodity", "node"]
    # parameter lookup
    pu = src.parameter("node", "penalty_up")
    assert pu["value"].to_list() == [100.0, 200.0]
    # defaults
    assert src.parameter_default("node", "penalty_up") == 10000.0
    assert src.parameter_default("node", "node_type") == "balance"
    # absent default
    assert src.parameter_default("node", "inflow") is None
    # unknown class / param raise KeyError
    with pytest.raises(KeyError):
        src.entities("zzz_unknown")
    with pytest.raises(KeyError):
        src.parameter("node", "zzz_unknown")


def test_inmemory_reader_drives_direct_param_helper():
    """The Direct Param helpers compose against the InputSource
    Protocol — exercise one end-to-end without sqlite.
    """
    src = InMemoryReader(
        entities={
            "commodity": pl.DataFrame({"name": ["coal", "gas"]}),
        },
        parameters={
            ("commodity", "co2_content"): pl.DataFrame({
                "name": ["coal", "gas"],
                "value": [0.34, 0.20],
            }),
        },
    )
    p = dp.p_co2_content_from_source(src)
    assert isinstance(p, Param)
    assert p.dims == ("c",)
    assert p.frame.sort("c")["value"].to_list() == [0.34, 0.20]


def test_inmemory_reader_none_default_skip():
    """Empty parameter frame → helper returns None (no synthetic rows).

    Mirrors the §4.5 None-skip default policy.
    """
    src = InMemoryReader(
        entities={
            "commodity": pl.DataFrame({"name": ["coal"]}),
        },
        parameters={
            ("commodity", "co2_content"): pl.DataFrame(
                schema={"name": pl.Utf8, "value": pl.Float64},
            ),
        },
    )
    p = dp.p_co2_content_from_source(src)
    assert p is None


# ---------------------------------------------------------------------------
# Source-aware load_flextool — DB-direct override smoke tests


def test_load_flextool_db_reader_override(scenario_run):
    """``load_flextool(folder, provider=..., db_reader=reader)`` produces a
    FlexData where the first-wave Direct fields have been replaced by
    frames sourced from the SpineDB scenario.
    """
    db_data = load_scenario(scenario_run("test_a_lot"))
    # The DB-direct override should populate p_co2_content from the
    # commodity.co2_content parameter even when the CSV path sourced
    # it via _load_co2_price's gating.
    assert db_data.p_co2_content is not None
    co2_frame = db_data.p_co2_content.frame
    assert "c" in co2_frame.columns and "value" in co2_frame.columns
    assert co2_frame.filter(pl.col("c") == "coal")["value"][0] == pytest.approx(0.34)


def test_load_flextool_rejects_invalid_db_reader(scenario_run):
    """``db_reader`` arg must implement the InputSource Protocol."""
    run = scenario_run("coal")
    with pytest.raises(TypeError):
        load_flextool(run.load_folder, provider=clone_provider(run.last.provider), db_reader=object())


def test_load_flextool_with_db_reader_solves_correctly(scenario_run):
    """End-to-end smoke: loading the ``coal`` scenario through the
    production loader call (provider + DB reader) and solving still
    produces the recorded objective.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run("coal")
    work = run.work_folder
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"coal db-direct: polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# v56 None-default policy regression guard — schema defaults stay None
#
# Each (entity_class, parameter_name) below historically had a Python
# convention default applied by the engine (1.0 for *_unitsize, 0.0 for
# loss/rate/load/reservation rows).  The v56 audit (see
# _audit_reports/v56_default_audit.md and commit 10d9a21f) committed to
# a "None-default" policy: spinedb_schema.json carries
# ``default_value = None`` for every such row, and consumers fall back
# to the engine's hard-coded substitute at read time (see
# ``_scalar_default`` in _derived.py and the min_load comment in
# _projection.py:1101).  Regression here would mean a vendored
# JSON drifted *to* a non-None default — silently bypassing the
# engine-side fallback and breaking the contract that the schema only
# declares user-overridable defaults.

# (entity_class, parameter_name, expected_default)
RESOLVED_DEFAULTS = [
    ("unit", "virtual_unitsize", None),
    ("node", "virtual_unitsize", None),
    ("connection", "virtual_unitsize", None),
    ("unit", "min_load", None),
    ("node", "self_discharge_loss", None),
    ("model", "inflation_rate", None),
    ("reserve__upDown__group", "reservation", None),
]


@pytest.mark.parametrize("entity_class, param_name, expected",
                         RESOLVED_DEFAULTS,
                         ids=[f"{c}.{p}" for c, p, _ in RESOLVED_DEFAULTS])
def test_resolved_default_landed(entity_class, param_name, expected,
                                  scenario_run):
    """v56 None-default policy: each historical Python-convention default
    must surface as ``None`` through ``SpineDbReader.parameter_default``,
    confirming the engine-side substitution path is still the source of
    truth.  The fixture (``test_a_lot``) descends from ``tests.json``,
    one of the three JSON sources that participate in the policy.  A
    future JSON edit that bakes a default back into the schema breaks
    here rather than silently bypassing ``_scalar_default``'s fallback.
    """
    reader = scenario_run("test_a_lot").reader()
    actual = reader.parameter_default(entity_class, param_name)
    assert actual == expected, (
        f"{entity_class}.{param_name} default: expected {expected}, "
        f"got {actual!r} — v56 None-default policy regression"
    )


# ---------------------------------------------------------------------------
# Γ.3.A — DB-direct solve parity on representative fixtures


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_coal", "tests.sqlite", "coal",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_test_a_lot_but_not_multi_year", "tests.sqlite",
          "test_a_lot_but_not_multi_year",
          "v_obj__y2020_2day_dispatch.parquet"),
    ],
    ids=["work_coal", "work_test_a_lot_but_not_multi_year"],
)
def test_db_direct_solve_parity_with_derived_a(work, sqlite, scenario, parquet,
                                                  scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.A overlay.

    Adds the Derived foundational frames (dt / step_duration / period
    share / branch weights / inflow / profile / existing count) to the
    Γ.1 + Γ.2 overlay chain.  Reproduce flextool's recorded objective
    to rel < 1e-6 — same bar as the Γ.2 solve parity test, with the
    additional Derived path active.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        # Some fixtures have a different obj filename — pick any.
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.A derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_coal", "tests.sqlite", "coal",
          "v_obj__y2020_2day_dispatch.parquet"),
    ],
    ids=["work_coal"],
)
def test_db_direct_solve_parity_with_projections(work, sqlite, scenario, parquet,
                                                    scenario_run):
    """End-to-end: loading via the DB-direct override (Direct + Projection
    wave) and solving must reproduce flextool's recorded objective to
    rel < 1e-6.  Identical to ``test_load_flextool_with_db_reader_solves_correctly``
    but the implementation now exercises the additional Projection
    overlay path — this guards against regressions where the new
    overlay corrupts a feature gate (e.g. accidentally activating the
    reserve subsystem on a fixture that doesn't model reserves).
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(fixture / "output_raw" / parquet)["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.2 projections): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# Γ.3.B — DB-direct solve parity on broader fixtures


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_coal", "tests.sqlite", "coal",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_coal_chp", "tests.sqlite", "coal_chp",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_test_a_lot_but_not_multi_year", "tests.sqlite",
          "test_a_lot_but_not_multi_year",
          "v_obj__y2020_2day_dispatch.parquet"),
    ],
    ids=["work_coal", "work_coal_chp",
          "work_test_a_lot_but_not_multi_year"],
)
def test_db_direct_solve_parity_with_derived_b(work, sqlite, scenario, parquet,
                                                  scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.B overlay.

    Adds the topology + slope + flow_constraint_coef + varCost frames
    to the Γ.1 + Γ.2 + Γ.3.A overlay chain.  Reproduce flextool's
    recorded objective to rel < 1e-6 — same bar as Γ.3.A's solve parity
    test, with the additional process-topology Derived path active.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.B derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# Γ.3.C — DB-direct solve parity


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_coal", "tests.sqlite", "coal",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_coal_min_load_MIP_wind", "tests.sqlite",
          "coal_min_load_MIP_wind",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_capacity_margin", "tests.sqlite", "capacity_margin",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_wind_battery_invest", "tests.sqlite",
          "wind_battery_invest",
          "v_obj__y2020_5week.parquet"),
    ],
    ids=["work_coal", "work_coal_min_load_MIP_wind",
          "work_capacity_margin", "work_wind_battery_invest"],
)
def test_db_direct_solve_parity_with_derived_c(work, sqlite, scenario,
                                                  parquet, scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.C overlay.

    Builds on top of the Γ.3.B chain — adds invest/divest, online/UC,
    and group-slack overlays.  Reproduces flextool's recorded objective
    to rel < 1e-6 with the full Γ.1 + Γ.2 + Γ.3.A + Γ.3.B + Γ.3.C
    overlay applied.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.C derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# Γ.3.D — DB-direct solve parity


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_coal", "tests.sqlite", "coal",
          "v_obj__y2020_2day_dispatch.parquet"),
        ("work_wind_battery_invest", "tests.sqlite",
          "wind_battery_invest",
          "v_obj__y2020_5week.parquet"),
        ("work_lh2_three_region", "tests.sqlite", "lh2_three_region",
          "v_obj__y2030_one_week.parquet"),
        ("work_network_coal_wind_reserve", "tests.sqlite",
          "network_coal_wind_reserve",
          "v_obj__y2020_2day_dispatch.parquet"),
    ],
    ids=["work_coal", "work_wind_battery_invest",
          "work_lh2_three_region", "work_network_coal_wind_reserve"],
)
def test_db_direct_solve_parity_with_derived_d(work, sqlite, scenario,
                                                  parquet, scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.D overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A + Γ.3.B + Γ.3.C + Γ.3.D overlay applied.
    Γ.3.D adds: ``p_entity_all_existing`` (when simple), DC-PF reference
    angle, and the reserve relationship Projection.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.D derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# Γ.3.E — DB-direct solve parity on storage-relevant fixtures


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_lh2_three_region", "tests.sqlite", "lh2_three_region",
          "v_obj__y2030_one_week.parquet"),
        ("work_wind_battery_invest", "tests.sqlite",
          "wind_battery_invest",
          "v_obj__y2020_5week.parquet"),
        ("work_5weeks_battery_intraperiod_blocks", "tests.sqlite",
          "5weeks_battery_intraperiod_blocks",
          "v_obj__y2020_5week.parquet"),
        ("work_2day_stochastic_dispatch_full_storage", "tests.sqlite",
          "2_day_stochastic_dispatch",
          "v_obj__2day_dispatch.parquet"),
    ],
    ids=["work_lh2_three_region", "work_wind_battery_invest",
          "work_5weeks_battery_intraperiod_blocks",
          "work_2day_stochastic_dispatch_full_storage"],
)
def test_db_direct_solve_parity_with_derived_e(work, sqlite, scenario,
                                                  parquet, scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.E overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A/B/C/D/E overlay applied.  Γ.3.E adds the
    storage block algebra: dtttdt + period_block family + nodeStateBlock
    multi-resolution synthesis + arc-block weights + state caps + the
    storage-use-reference-value exclusion + rolling-handoff carriers.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.E derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# Γ.3.F — DB-direct solve parity on representative multi-year +
# lifetime fixtures


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_5weeks_invest_fullYear_dispatch_coal_wind", "tests.sqlite",
          "5weeks_invest_fullYear_dispatch_coal_wind",
          "v_obj__y2020_fullYear_dispatch.parquet"),
        ("work_wind_battery_invest_lifetime_renew", "tests.sqlite",
          "wind_battery_invest_lifetime_renew",
          "v_obj__y2020_2035_5week.parquet"),
    ],
    ids=["work_5weeks_invest_fullYear_dispatch_coal_wind",
          "work_wind_battery_invest_lifetime_renew"],
)
def test_db_direct_solve_parity_with_derived_f(work, sqlite, scenario,
                                                  parquet, scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.F overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A/B/C/D/E/F overlay applied.  Γ.3.F adds the
    lifetime cascade family (annuity + discounted-fixed-cost), the
    multi-year inflation cascade, and the rolling-handoff state read
    side.
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.F derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# Γ.3.G — residual-feature solve parity

DERIVED_G_SOLVE_FIXTURES: list[tuple[str, str, str, str]] = [
    ("work_commodity_ladder_annual", "tests.sqlite", "coal_ladder_annual",
      "v_obj__*.parquet"),
    ("work_commodity_ladder_cumulative", "tests.sqlite",
      "coal_ladder_cumulative", "v_obj__*.parquet"),
    ("work_delay_source_coef", "tests.sqlite", "water_pump_delayed",
      "v_obj__*.parquet"),
    ("work_network_coal_wind_reserve", "tests.sqlite",
      "network_coal_wind_reserve", "v_obj__*.parquet"),
    ("work_2day_stochastic_dispatch_full_storage", "tests.sqlite",
      "2_day_stochastic_dispatch", "v_obj__*.parquet"),
]


@pytest.mark.parametrize("work, sqlite, scenario, parquet",
                          DERIVED_G_SOLVE_FIXTURES,
                          ids=[f[0] for f in DERIVED_G_SOLVE_FIXTURES])
def test_db_direct_solve_parity_with_derived_g(work, sqlite, scenario,
                                                  parquet, scenario_run):
    """End-to-end DB-direct solve parity through Γ.3.G overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A-G overlay applied.  Γ.3.G adds the residual
    Derived Params (commodity ladder, reserves, delay, full multi-
    branch normalisation).
    """
    from polar_high import Problem
    from flextool.engine import build_flextool
    run = scenario_run(scenario, db_fixture=_db_fixture_for(scenario))
    fixture = run.work_folder
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    data = _load_solve_of(run, parquets[0])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.G derived): "
        f"polar_high={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )
