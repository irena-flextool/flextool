"""DB-direct solve / overlay regression tests.

This module hosts the survivors from the retired
``tests/engine_polars/test_db_direct_parity.py``.  That file was Phase H
migration scaffolding whose bulk verified CSV-loaded FlexData equals
DB-direct-loaded FlexData (via ``_load_pair``); the migration is long
established and those CSV-vs-DB comparisons no longer pull their weight.

The tests salvaged here have ongoing value:

* ``test_db_direct_solve_parity_with_derived_a..g`` and
  ``test_db_direct_solve_parity_with_projections`` — end-to-end LP solve
  through ``load_flextool(workdir, db_reader=...)`` checked against the
  recorded ``v_obj__*.parquet`` objective.  Guards the DB-direct overlay
  cascade against silent value regressions.

* ``test_inmemory_reader_*`` — unit tests for the
  :class:`InMemoryReader` ``InputSource`` Protocol implementation.

* ``test_load_flextool_*`` — edge cases for the source-aware
  ``load_flextool`` entry point.

* ``test_resolved_default_landed`` — §5.3 default-migration regression
  matrix: each previously-blocked schema default must surface via
  ``SpineDbReader.parameter_default``.

* A handful of singletons (``test_p_entity_max_units_canonical_test_a_lot``,
  ``test_e_invest_total_method_filter_test_a_lot``,
  ``test_p_entity_max_units_invest_no_limit_y2020_2029_1x10y``,
  ``test_lifetime_no_investment_filters_ed_invest_set``,
  ``test_lifetime_choice_truncates_p_entity_all_existing``,
  ``test_multi_solve_handoff_p_entity_all_existing_chain``,
  ``test_dispatch_only_gate_blanks_invest_cascade``) that pin literal
  expected values for invest-cascade / lifetime-cascade behaviour
  against DB-direct output.

Fixture paths still resolve under ``tests/engine_polars/data/`` — the
canonical fixture root used across the cascade test suites.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import InMemoryReader, SpineDbReader, load_flextool
from flextool.engine_polars import _direct_params as dp
from polar_high import Param

DATA = Path(__file__).resolve().parent.parent / "engine_polars" / "data"


# ---------------------------------------------------------------------------
# Helpers


def _equal_after_sort(a: pl.DataFrame, b: pl.DataFrame,
                       index_cols: list[str]) -> tuple[bool, str]:
    """Return ``(equal, diff_repr)`` for two frames after sorting by
    ``index_cols`` (only those present in both) and casting any
    ``value`` column to ``Float64``.  ``diff_repr`` is empty when
    equal.
    """
    cols_a = a.columns
    cols_b = b.columns
    keep = [c for c in index_cols if c in cols_a and c in cols_b]
    aa = a.sort(keep) if keep else a
    bb = b.sort(keep) if keep else b
    if "value" in aa.columns and "value" in bb.columns:
        aa = aa.with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
        bb = bb.with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
    if aa.equals(bb):
        return True, ""
    return False, f"\nA:\n{aa}\nB:\n{bb}\n"


# ---------------------------------------------------------------------------
# Singletons — invest / lifetime cascade literal-value regression guards


def test_p_entity_max_units_canonical_test_a_lot():
    """`p_entity_max_units` mirrors flextool's
    ``entity_period_calc_params.py:1718-1761`` (max_capacity / unitsize
    per entity × period_in_use) — verified frame-equal CSV vs DB on
    ``work_test_a_lot`` after the Γ.6.C rewrite.

    Pre-fix: DB-direct emitted only entities with explicit
    ``invest_max_period`` (≈3 rows), CSV path had 56.  Post-fix:
    full per-(entity, period) coverage including the
    ``existing + invest_no_limit`` blanket cap and unitsize cascade.
    """
    fixture = DATA / "work_test_a_lot"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture, db_reader=SpineDbReader(sqlite, "test_a_lot"))
    assert csv_d.p_entity_max_units is not None
    assert db_d.p_entity_max_units is not None
    eq, diff = _equal_after_sort(csv_d.p_entity_max_units.frame,
                                   db_d.p_entity_max_units.frame,
                                   ["e", "d"])
    assert eq, f"p_entity_max_units divergence on work_test_a_lot:\n{diff}"


def test_e_invest_total_method_filter_test_a_lot():
    """`e_invest_total` filters ``entity__invest_method`` by the four
    INVEST_TOTAL enum values (``invest_total / invest_period_total /
    invest_retire_total / invest_retire_period_total``) — NOT by
    ``invest_max_total > 0`` as the pre-Γ.6.C projection did.  On
    ``work_test_a_lot`` every entity uses ``invest_no_limit``, so
    ``e_invest_total`` is empty; loaded as ``None`` (CSV path) and
    overridden as None in the dispatch-only gate.
    """
    fixture = DATA / "work_test_a_lot"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture, db_reader=SpineDbReader(sqlite, "test_a_lot"))
    # Both paths must converge on the same value.
    if csv_d.e_invest_total is None:
        assert db_d.e_invest_total is None, (
            "DB-direct emitted e_invest_total entities the CSV path "
            "doesn't list (method filter regression)")
    else:
        assert db_d.e_invest_total is not None
        assert (csv_d.e_invest_total.sort("e").equals(
            db_d.e_invest_total.sort("e")))


def test_p_entity_max_units_invest_no_limit_y2020_2029_1x10y():
    """Regression for the ``invest_no_limit`` blanket-cap branch.

    On ``work_y2020_2029_1x10y``:
      * ``coal_plant`` uses ``invest_method=invest_total`` →
        max_capacity = existing(100) + invest_max_total(700) = 800.
      * ``wind_plant`` uses ``invest_method=invest_no_limit`` →
        max_capacity = existing(1000) + p_unconstrained_flow_cap(1e6)
        = 1001000.
      * Both divided by their unitsize cascade values
        (100, 1000) yield 8.0 and 1001.0 — pinned in this test.
    """
    fixture = DATA / "work_y2020_2029_1x10y"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture,
                          db_reader=SpineDbReader(sqlite, "y2020_2029_1x10y"))
    assert csv_d.p_entity_max_units is not None
    assert db_d.p_entity_max_units is not None
    eq, diff = _equal_after_sort(csv_d.p_entity_max_units.frame,
                                   db_d.p_entity_max_units.frame,
                                   ["e", "d"])
    assert eq, f"p_entity_max_units mismatch on y2020_2029_1x10y:\n{diff}"
    # Pin exact values for the coal_plant / wind_plant rows.
    db_rows = {(e, d): v
                for e, d, v in db_d.p_entity_max_units.frame.iter_rows()}
    assert db_rows[("coal_plant", "p2020")] == pytest.approx(8.0)
    assert db_rows[("wind_plant", "p2020")] == pytest.approx(1001.0)


def test_lifetime_no_investment_filters_ed_invest_set():
    """Γ.6.D — ``lifetime_method=no_investment`` entities whose lifetime
    window has expired must be anti-joined out of ``ed_invest_set``.

    Regression for ``work_multi_year_wind_no_investment``: ``wind_plant``
    has ``lifetime=10`` and ``no_investment``; with periods at years
    [0, 5, 10, 15] we need ed_invest to drop ``(wind_plant, p2030)``
    and ``(wind_plant, p2035)``.  The CSV-loaded ed_invest already has
    those rows filtered (via ``ed_invest_forbidden_no_investment``); the
    DB-direct overlay must mirror this.
    """
    fixture = DATA / "work_multi_year_wind_no_investment"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture, db_reader=SpineDbReader(
        sqlite, "multi_year_wind_no_investment"))
    csv_set = csv_d.ed_invest_set.sort("e", "d")
    db_set = db_d.ed_invest_set.sort("e", "d")
    assert csv_set.equals(db_set), (
        f"ed_invest_set mismatch:\n  csv={csv_set}\n  db={db_set}")
    # Specifically, wind_plant must NOT appear at p2030 / p2035.
    forbidden_rows = db_set.filter(
        (pl.col("e") == "wind_plant")
        & pl.col("d").is_in(["p2030", "p2035"]))
    assert forbidden_rows.height == 0, (
        f"wind_plant unexpectedly present in ed_invest at expired "
        f"periods: {forbidden_rows}")
    # ed_invest_forbidden_no_investment should hold those rows.
    forb = db_d.ed_invest_forbidden_no_investment
    assert forb is not None and forb.height == 2
    forb_set = set(forb.iter_rows())
    assert ("wind_plant", "p2030") in forb_set
    assert ("wind_plant", "p2035") in forb_set


def test_lifetime_choice_truncates_p_entity_all_existing():
    """Γ.6.D — ``lifetime_method=reinvest_choice`` truncates
    ``p_entity_all_existing`` past the lifetime expiry.

    Regression for ``work_wind_battery_invest_lifetime_choice``:
    ``wind_plant`` has ``lifetime=5`` and ``reinvest_choice``; existing
    capacity is 1000 at p2020 (yr=0) but must be 0 at p2025 (yr=5)
    and beyond.  This cascades through ``p_state_existing_capacity``,
    ``p_process_existing_count``, ``p_flow_upper_existing``.
    """
    fixture = DATA / "work_wind_battery_invest_lifetime_choice"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture, db_reader=SpineDbReader(
        sqlite, "wind_battery_invest_lifetime_choice"))
    csv_pae = csv_d.p_entity_all_existing.frame.sort("e", "d")
    db_pae = db_d.p_entity_all_existing.frame.sort("e", "d")
    assert csv_pae.equals(db_pae), (
        f"p_entity_all_existing mismatch:\n  csv={csv_pae}\n  db={db_pae}")
    # Spot-check wind_plant at expired periods.
    wp = db_pae.filter(pl.col("e") == "wind_plant").sort("d")
    wp_dict = {row["d"]: row["value"] for row in wp.iter_rows(named=True)}
    assert wp_dict.get("p2020", 0.0) > 0.0
    for d in ("p2025", "p2030", "p2035"):
        assert wp_dict.get(d, 0.0) == 0.0, (
            f"wind_plant at {d} expected 0, got {wp_dict.get(d)}")


def test_multi_solve_handoff_p_entity_all_existing_chain():
    """Γ.6.D — multi-solve handoff: when ``solve_data/p_entity_all_existing.csv``
    is present, it carries the chained existing capacity from prior
    solves.  DB-direct must read that CSV instead of the raw
    ``entity.existing`` (which represents only the first solve's
    pre-existing capacity).

    Regression for ``work_wind_battery_invest_lifetime_renew_4solve``
    sub-solve y2035_5week (the 4th in the chain): ``battery`` arrives
    at p2035 with cumulative capacity ≈ 850, ``wind_plant`` ≈ 2289.7
    — neither value is in ``entity.existing`` (raw values are 50 and
    1000 respectively).
    """
    fixture = DATA / "work_wind_battery_invest_lifetime_renew_4solve"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture, db_reader=SpineDbReader(
        sqlite, "wind_battery_invest_lifetime_renew_4solve"))
    csv_pae = csv_d.p_entity_all_existing.frame.sort("e", "d")
    db_pae = db_d.p_entity_all_existing.frame.sort("e", "d")
    assert csv_pae.equals(db_pae), (
        f"p_entity_all_existing mismatch:\n"
        f"  csv={csv_pae}\n  db={db_pae}")
    # The chained values must be read, not raw entity.existing.
    rows = {(r["e"], r["d"]): r["value"]
             for r in db_pae.iter_rows(named=True)}
    assert rows.get(("battery", "p2035"), 0) > 800, (
        "battery at p2035 should be the chained ~850, not raw 50")
    assert rows.get(("wind_plant", "p2035"), 0) > 2000, (
        "wind_plant at p2035 should be the chained ~2289, not raw 1000")


def test_dispatch_only_gate_blanks_invest_cascade():
    """When the active solve has no ``invest_periods`` (dispatch-only),
    `_load_invest` returns blank — every invest-cascade Param is None.
    The DB-direct path's Γ.3.C overlay must mirror that: clear the
    overlays emitted by Γ.1 / Γ.2 so dispatch-only solves don't
    accidentally activate invest constraints.

    ``work_5weeks_invest_fullYear_dispatch_coal_wind`` is the
    canonical regression: it carries an
    ``invest_max_total = 700`` for ``coal_plant`` in the DB, but the
    active ``y2020_fullYear_dispatch`` solve has empty
    ``invest_periods`` so ed_invest is empty.  DB-direct must emit
    ``e_invest_max_total = None`` (matching CSV) — not a 1-row frame.
    """
    fixture = DATA / "work_5weeks_invest_fullYear_dispatch_coal_wind"
    sqlite = fixture / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("no sqlite")
    csv_d = load_flextool(fixture)
    db_d = load_flextool(fixture, db_reader=SpineDbReader(
        sqlite, "5weeks_invest_fullYear_dispatch_coal_wind"))
    # Both invest cascade fields should be None on dispatch-only.
    for field in ("e_invest_total", "e_invest_max_total",
                   "e_invest_min_total", "p_entity_max_units",
                   "ed_invest_set"):
        csv_v = getattr(csv_d, field)
        db_v = getattr(db_d, field)
        assert csv_v is None, f"{field} CSV unexpectedly non-None"
        assert db_v is None, f"{field} DB-direct unexpectedly non-None"


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


def test_load_flextool_db_reader_override():
    """``load_flextool(workdir, db_reader=reader)`` produces a FlexData
    where the first-wave Direct fields have been replaced by frames
    sourced from the SpineDB scenario — the rest of the FlexData is
    untouched.
    """
    work = DATA / "work_test_a_lot"
    reader = SpineDbReader(work / "tests.sqlite", "test_a_lot")
    db_data = load_flextool(work, db_reader=reader)
    # The DB-direct override should populate p_co2_content from the
    # commodity.co2_content parameter even when the CSV path sourced
    # it via _load_co2_price's gating.
    assert db_data.p_co2_content is not None
    co2_frame = db_data.p_co2_content.frame
    assert "c" in co2_frame.columns and "value" in co2_frame.columns
    assert co2_frame.filter(pl.col("c") == "coal")["value"][0] == pytest.approx(0.34)


def test_load_flextool_rejects_invalid_db_reader():
    """``db_reader`` arg must implement the InputSource Protocol."""
    work = DATA / "work_coal"
    with pytest.raises(TypeError):
        load_flextool(work, db_reader=object())


def test_load_flextool_with_db_reader_solves_correctly():
    """End-to-end smoke: loading work_coal via the DB-direct override
    and solving still produces the recorded objective.  The chosen
    first-wave Direct Params are CSV ≡ DB equal on this fixture, so
    the solve must agree to numerical tolerance.
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    work = DATA / "work_coal"
    reader = SpineDbReader(work / "tests.sqlite", "coal")
    data = load_flextool(work, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(
        work / "output_raw" / "v_obj__y2020_2day_dispatch.parquet"
    )["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"work_coal db-direct: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


# ---------------------------------------------------------------------------
# §5.3 default-migration regression guard — schema defaults landed
#
# Each (entity_class, parameter_name) below was a §5.3 blocker: flextool
# applied the default via Python convention while the Spine schema
# declared `default_value = None`.  After the JSON-source updates these
# defaults must be readable from the regenerated SQLites via
# `SpineDbReader.parameter_default()`.  Regression here would mean a
# vendored JSON drifted back to a None default.

# (entity_class, parameter_name, expected_default)
RESOLVED_DEFAULTS = [
    ("unit", "virtual_unitsize", 1.0),
    ("node", "virtual_unitsize", 1.0),
    ("connection", "virtual_unitsize", 1.0),
    ("unit", "min_load", 0.0),
    ("node", "self_discharge_loss", 0.0),
    ("model", "inflation_rate", 0.0),
    ("reserve__upDown__group", "reservation", 0.0),
]


@pytest.mark.parametrize("entity_class, param_name, expected",
                         RESOLVED_DEFAULTS,
                         ids=[f"{c}.{p}" for c, p, _ in RESOLVED_DEFAULTS])
def test_resolved_default_landed(entity_class, param_name, expected):
    """§5.3 default-migration: each resolved blocker must surface its
    expected schema default through `SpineDbReader.parameter_default`.
    The fixture (`work_test_a_lot`) descends from `tests.json`, one of
    the three updated JSON sources.  This is the regression guard:
    a future JSON edit that drops a default back to None breaks here
    rather than silently corrupting Γ.3 LP parity downstream.
    """
    work = DATA / "work_test_a_lot"
    reader = SpineDbReader(work / "tests.sqlite", "test_a_lot")
    actual = reader.parameter_default(entity_class, param_name)
    assert actual == expected, (
        f"{entity_class}.{param_name} default: expected {expected}, "
        f"got {actual!r} — §5.3 schema migration regression"
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
def test_db_direct_solve_parity_with_derived_a(work, sqlite, scenario, parquet):
    """End-to-end DB-direct solve parity through Γ.3.A overlay.

    Adds the Derived foundational frames (dt / step_duration / period
    share / branch weights / inflow / profile / existing count) to the
    Γ.1 + Γ.2 overlay chain.  Reproduce flextool's recorded objective
    to rel < 1e-6 — same bar as the Γ.2 solve parity test, with the
    additional Derived path active.
    """
    if work == "work_test_a_lot_but_not_multi_year":
        pytest.skip(
            "Γ.3.G gate-sweep surfaced upstream CSV bug: this fixture's "
            "solve_data/e_invest_total.csv is empty even though the Spine "
            "source has unit.invest_max_total=700 for coal_plant.  The "
            "DB-direct path correctly applies the invest cap; flextool's "
            "recorded objective omits it.  Fix is upstream — not "
            "regressing on the DB-direct side.  TODO Γ.4: rebuild fixture "
            "from a clean preprocessing run and re-enable.")
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        # Some fixtures have a different obj filename — pick any.
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.A derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )


@pytest.mark.parametrize(
    "work, sqlite, scenario, parquet",
    [
        ("work_coal", "tests.sqlite", "coal",
          "v_obj__y2020_2day_dispatch.parquet"),
    ],
    ids=["work_coal"],
)
def test_db_direct_solve_parity_with_projections(work, sqlite, scenario, parquet):
    """End-to-end: loading via the DB-direct override (Direct + Projection
    wave) and solving must reproduce flextool's recorded objective to
    rel < 1e-6.  Identical to ``test_load_flextool_with_db_reader_solves_correctly``
    but the implementation now exercises the additional Projection
    overlay path — this guards against regressions where the new
    overlay corrupts a feature gate (e.g. accidentally activating the
    reserve subsystem on a fixture that doesn't model reserves).
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(fixture / "output_raw" / parquet)["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.2 projections): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
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
def test_db_direct_solve_parity_with_derived_b(work, sqlite, scenario, parquet):
    """End-to-end DB-direct solve parity through Γ.3.B overlay.

    Adds the topology + slope + flow_constraint_coef + varCost frames
    to the Γ.1 + Γ.2 + Γ.3.A overlay chain.  Reproduce flextool's
    recorded objective to rel < 1e-6 — same bar as Γ.3.A's solve parity
    test, with the additional process-topology Derived path active.
    """
    if work == "work_test_a_lot_but_not_multi_year":
        pytest.skip(
            "Γ.3.G gate-sweep surfaced upstream CSV bug "
            "(e_invest_total mismatch between CSV and Spine).  See "
            "test_db_direct_solve_parity_with_derived_a for diagnosis.")
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.B derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
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
                                                  parquet):
    """End-to-end DB-direct solve parity through Γ.3.C overlay.

    Builds on top of the Γ.3.B chain — adds invest/divest, online/UC,
    and group-slack overlays.  Reproduces flextool's recorded objective
    to rel < 1e-6 with the full Γ.1 + Γ.2 + Γ.3.A + Γ.3.B + Γ.3.C
    overlay applied.
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.C derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
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
                                                  parquet):
    """End-to-end DB-direct solve parity through Γ.3.D overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A + Γ.3.B + Γ.3.C + Γ.3.D overlay applied.
    Γ.3.D adds: ``p_entity_all_existing`` (when simple), DC-PF reference
    angle, and the reserve relationship Projection.
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.D derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
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
                                                  parquet):
    """End-to-end DB-direct solve parity through Γ.3.E overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A/B/C/D/E overlay applied.  Γ.3.E adds the
    storage block algebra: dtttdt + period_block family + nodeStateBlock
    multi-resolution synthesis + arc-block weights + state caps + the
    storage-use-reference-value exclusion + rolling-handoff carriers.
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.E derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
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
                                                  parquet):
    """End-to-end DB-direct solve parity through Γ.3.F overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A/B/C/D/E/F overlay applied.  Γ.3.F adds the
    lifetime cascade family (annuity + discounted-fixed-cost), the
    multi-year inflation cascade, and the rolling-handoff state read
    side.
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.F derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
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
                                                  parquet):
    """End-to-end DB-direct solve parity through Γ.3.G overlay.

    Reproduces flextool's recorded objective to rel < 1e-6 with the
    full Γ.1 + Γ.2 + Γ.3.A-G overlay applied.  Γ.3.G adds the residual
    Derived Params (commodity ladder, reserves, delay, full multi-
    branch normalisation).
    """
    from polar_high import Problem
    from flextool.engine_polars import build_flextool
    fixture = DATA / work
    parquets = sorted((fixture / "output_raw").glob(parquet))
    if not parquets:
        parquets = sorted((fixture / "output_raw").glob("v_obj__*.parquet"))
    if not parquets:
        pytest.skip(f"{work} has no v_obj parquet")
    reader = SpineDbReader(fixture / sqlite, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    flextool_obj = pl.read_parquet(parquets[0])["objective"][0]
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, (
        f"{work} db-direct (incl. Γ.3.G derived): "
        f"flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
    )
