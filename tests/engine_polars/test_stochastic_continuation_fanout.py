"""Multi-period stochastic continuation fan-out — integration tests.

Design: ``specs/step05_continuation_fanout_design.md`` §5.1-§5.3.

Fixture ``tests/fixtures/stoch_two_period.json`` (built from JSON via
``json_to_db`` — CLAUDE.md invariant 3): one 6-step timeline split into
periods ``p2035`` (t0001-t0003) and ``p2040`` (t0004-t0006), demand
100 / 110 MW, a free wind unit capped by a branch-varying profile
(stochastic group membership via ``group__unit`` only), and a
50 EUR/MWh thermal backstop.  Branches ``rlz`` (realized, weight 1.0)
and ``low`` (weight 3.0) are declared at ``p2035``/t0001, making
``p2040`` a CONTINUATION period.

Hand calculation (design §5.2, uniform annualization factor
8760/3 = 2920, weights 0.25 / 0.75):

    stochastic objective   = 2920 x 24 000 = 70 080 000
    deterministic control  = 2920 x 15 000 = 43 800 000

DEFECT PINS (fail on unfixed main): before the fix the continuation
``else`` branch of ``create_stochastic_periods`` wrote NO active /
realized / fix-storage time for periods after the branching period, so
``steps_in_use.csv`` had only the 6 ``p2035`` cohort rows (silent
horizon truncation, zero LP variables in p2040) — the 12-row
``steps_in_use`` assertion, the ``period_in_use ⊆ period_with_history``
invariant, the p2040 rows of ``realized_dispatch`` / flows, and the
objective values all pin it.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

# Hand-calc constants (design §5.2).
STOCH_OBJECTIVE = 70_080_000.0
DET_OBJECTIVE = 43_800_000.0
WIND_EXISTING = 100.0
THERMAL_EXISTING = 200.0

WIND_COL = "('wind', 'wind', 'city')"
THERMAL_COL = "('thermal', 'thermal', 'city')"

P1_STEPS = ["t0001", "t0002", "t0003"]
P2_STEPS = ["t0004", "t0005", "t0006"]

# The 12 (period, step) rows of design §4.
EXPECTED_STEPS_IN_USE = (
    [("p2035", t) for t in P1_STEPS]
    + [("p2035_low", t) for t in P1_STEPS]
    + [("p2040", t) for t in P2_STEPS]
    + [("p2040_low", t) for t in P2_STEPS]
)

# period__branch rows of design §4 (metadata fan incl. realized).
EXPECTED_PERIOD_BRANCH = [
    ("p2035", "p2035"),
    ("p2035", "p2035_rlz"),
    ("p2035", "p2035_low"),
    ("p2040", "p2040"),
    ("p2040", "p2040_rlz"),
    ("p2040", "p2040_low"),
]

EXPECTED_PD_BRANCH_WEIGHT = {
    "p2035": 0.25,
    "p2035_low": 0.75,
    "p2040": 0.25,
    "p2040_low": 0.75,
}

# step_previous.csv rows of design §4 (7-column CSV shape).
EXPECTED_STEP_PREVIOUS_FIRST_ROWS = {
    # (period, time) -> (previous, previous_within_timeset,
    #                    previous_period, previous_within_solve, jump)
    ("p2035", "t0001"): ("t0003", "t0003", "p2040_low", "t0006", -5),
    ("p2035_low", "t0001"): ("t0003", "t0003", "p2035_low", "t0003", -2),
    ("p2040", "t0004"): ("t0006", "t0006", "p2035", "t0003", 1),
    ("p2040_low", "t0004"): ("t0006", "t0006", "p2035_low", "t0003", 1),
}

# flex_data.dtttdt rows of design §4 (all 12; first-of-period rows
# carry the cross-period t_previous, unlike the CSV's within-period
# cyclic ``previous`` column — pre-existing divergence, design §3.3).
EXPECTED_DTTTDT = [
    ("p2035", "t0001", "t0006", "t0003", "p2040_low", "t0006"),
    ("p2035", "t0002", "t0001", "t0001", "p2035", "t0001"),
    ("p2035", "t0003", "t0002", "t0002", "p2035", "t0002"),
    ("p2035_low", "t0001", "t0003", "t0003", "p2035_low", "t0003"),
    ("p2035_low", "t0002", "t0001", "t0001", "p2035_low", "t0001"),
    ("p2035_low", "t0003", "t0002", "t0002", "p2035_low", "t0002"),
    ("p2040", "t0004", "t0003", "t0006", "p2035", "t0003"),
    ("p2040", "t0005", "t0004", "t0004", "p2040", "t0004"),
    ("p2040", "t0006", "t0005", "t0005", "p2040", "t0005"),
    ("p2040_low", "t0004", "t0003", "t0006", "p2035_low", "t0003"),
    ("p2040_low", "t0005", "t0004", "t0004", "p2040_low", "t0004"),
    ("p2040_low", "t0006", "t0005", "t0005", "p2040_low", "t0005"),
]


@pytest.fixture(scope="module")
def stoch_wf(scenario_workdir):
    """The ``stoch`` scenario workdir, cascade run under
    ``FLEXTOOL_AUTOSCALE_STRICT=1`` (design §5.3: one scenario run
    executes strict so an autoscale-registry gap would be loud)."""
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("stoch", db_fixture="stoch_two_period")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def horizon_wf(scenario_workdir):
    return scenario_workdir("stoch_horizon", db_fixture="stoch_two_period")


@pytest.fixture(scope="module")
def det_wf(scenario_workdir):
    return scenario_workdir("det", db_fixture="stoch_two_period")


def _objective(wf) -> float:
    return float(
        pl.read_parquet(wf / "output_raw" / "v_obj__stoch_2p.parquet")
        ["objective"][0]
    )


def _v_flow(wf) -> pl.DataFrame:
    return pl.read_parquet(wf / "output_raw" / "v_flow__stoch_2p.parquet")


def _csv_rows(wf, name: str) -> list[tuple]:
    return [tuple(r) for r in
            pl.read_csv(wf / "solve_data" / name).rows()]


# ---------------------------------------------------------------------------
# ``stoch`` scenario (output_horizon off — realized-only outputs)
# ---------------------------------------------------------------------------


def test_stoch_objective_hand_calc(stoch_wf) -> None:
    """§5.3-1: hand-calculated stochastic objective (DEFECT PIN — the
    truncated pre-fix LP had no p2040 cohort and a different optimum)."""
    assert _objective(stoch_wf) == pytest.approx(
        STOCH_OBJECTIVE, rel=1e-6
    )


def test_stoch_steps_in_use_full_horizon(stoch_wf) -> None:
    """§5.3-2 (DEFECT PIN): exactly the 12 (period, step) rows of §4 —
    pre-fix the continuation periods produced ZERO rows here."""
    siu = pl.read_csv(stoch_wf / "solve_data" / "steps_in_use.csv")
    rows = [(r[0], r[1]) for r in siu.rows()]
    assert rows == EXPECTED_STEPS_IN_USE


def test_stoch_period_bookkeeping_consistent(stoch_wf) -> None:
    """§5.3-2: period_in_use_set == the four cohort members, each of
    which appears in period_with_history (subset invariant), and the
    continuation branch period has year rows."""
    piu = pl.read_csv(stoch_wf / "solve_data" / "period_in_use_set.csv")
    assert piu["period"].to_list() == [
        "p2035", "p2035_low", "p2040", "p2040_low",
    ]
    pwh = pl.read_csv(stoch_wf / "solve_data" / "period_with_history.csv")
    pwh_periods = set(pwh["period"].to_list())
    for p in piu["period"].to_list():
        assert p in pwh_periods, (
            f"{p} has LP variables but no period_with_history row"
        )
    pyr = pl.read_csv(stoch_wf / "solve_data" / "p_years_represented.csv")
    assert pyr.filter(pl.col("period") == "p2040_low").height > 0, (
        "continuation branch period p2040_low has no year rows"
    )


def test_stoch_branch_weights(stoch_wf) -> None:
    """§5.3-3: per-cohort normalization 0.25 / 0.75, constant along
    each branch chain; pdt dense over the 12 (d, t) cells."""
    pdw = pl.read_csv(stoch_wf / "solve_data" / "pd_branch_weight.csv")
    got = {r[0]: float(r[1]) for r in pdw.rows()}
    assert set(got) == set(EXPECTED_PD_BRANCH_WEIGHT)
    for p, w in EXPECTED_PD_BRANCH_WEIGHT.items():
        assert got[p] == pytest.approx(w, abs=1e-12)
    pdtw = pl.read_csv(stoch_wf / "solve_data" / "pdt_branch_weight.csv")
    assert pdtw.height == 12
    for r in pdtw.rows(named=True):
        assert float(r["value"]) == pytest.approx(
            EXPECTED_PD_BRANCH_WEIGHT[r["period"]], abs=1e-12
        )
    sbw = pl.read_csv(stoch_wf / "solve_data" / "solve_branch_weight.csv")
    got_input = {r[0]: float(r[1]) for r in sbw.rows()}
    assert got_input == {
        "p2035": 1.0, "p2040": 1.0, "p2035_low": 3.0, "p2040_low": 3.0,
    }


def test_stoch_realized_gating_and_flows(stoch_wf) -> None:
    """§5.3-4 (DEFECT PIN for the p2040 rows): realized_dispatch is the
    6 real-period rows; committed flow outputs carry no branch labels;
    realized flows match the hand calc (wind 60/50, thermal 40/60 MW).
    """
    rd = _csv_rows(stoch_wf, "realized_dispatch.csv")
    assert rd == (
        [("p2035", t) for t in P1_STEPS] + [("p2040", t) for t in P2_STEPS]
    )
    flows = _v_flow(stoch_wf)
    assert sorted(set(flows["period"].to_list())) == ["p2035", "p2040"], (
        "branch periods leaked into the committed (realized-only) outputs"
    )
    assert flows.height == 6
    for r in flows.rows(named=True):
        wind_mw = r[WIND_COL] * WIND_EXISTING
        thermal_mw = r[THERMAL_COL] * THERMAL_EXISTING
        if r["period"] == "p2035":
            assert wind_mw == pytest.approx(60.0, abs=1e-6)
            assert thermal_mw == pytest.approx(40.0, abs=1e-6)
        else:
            assert wind_mw == pytest.approx(50.0, abs=1e-6)
            assert thermal_mw == pytest.approx(60.0, abs=1e-6)


def test_stoch_step_previous_continuation_linkage(stoch_wf) -> None:
    """§5.3-5 / §4 (DEFECT PIN for the p2040 rows): step_previous.csv
    links each continuation period to its own branch chain."""
    sp = pl.read_csv(stoch_wf / "solve_data" / "step_previous.csv")
    assert sp.height == 12
    first = {
        (r["period"], r["time"]): (
            r["previous"], r["previous_within_timeset"],
            r["previous_period"], r["previous_within_solve"], r["jump"],
        )
        for r in sp.rows(named=True)
        if (r["period"], r["time"]) in EXPECTED_STEP_PREVIOUS_FIRST_ROWS
    }
    assert first == EXPECTED_STEP_PREVIOUS_FIRST_ROWS


def test_stoch_model_side_dtttdt(stoch_wf) -> None:
    """§5.5 model-side (review blocker 2): the LIVE lag frame
    ``flex_data.dtttdt`` — not just the CSV — carries the §4 rows.
    The fixture has no storage/ramps, so the objective cannot catch a
    wrong lag frame; assert it directly."""
    from flextool.engine_polars import load_flextool

    data = load_flextool(stoch_wf)
    d = data.dtttdt
    assert d is not None
    rows = sorted(
        tuple(str(v) for v in r)
        for r in d.select(
            "d", "t", "t_previous", "t_previous_within_timeset",
            "d_previous", "t_previous_within_solve",
        ).rows()
    )
    assert rows == sorted(EXPECTED_DTTTDT)


def test_stoch_csv_dtttdt_parity_guard(stoch_wf) -> None:
    """§5.5 parity guard (scoped per design §3.3): dtttdt's
    ``d_previous`` / ``t_previous_within_solve`` equal
    step_previous.csv's ``previous_period`` / ``previous_within_solve``
    for ALL 12 rows, and full 6-column equality holds for the 8
    interior rows.  (Full-frame equality is impossible: ``t_previous``
    vs ``previous`` legitimately differ on first-of-period rows.)"""
    from flextool.engine_polars import load_flextool

    data = load_flextool(stoch_wf)
    dtttdt = {
        (str(r["d"]), str(r["t"])): r
        for r in data.dtttdt.rows(named=True)
    }
    sp = pl.read_csv(stoch_wf / "solve_data" / "step_previous.csv")
    assert sp.height == len(dtttdt) == 12
    first_steps = {("p2035", "t0001"), ("p2035_low", "t0001"),
                   ("p2040", "t0004"), ("p2040_low", "t0004")}
    for r in sp.rows(named=True):
        key = (r["period"], r["time"])
        m = dtttdt[key]
        assert str(m["d_previous"]) == r["previous_period"], key
        assert str(m["t_previous_within_solve"]) == (
            r["previous_within_solve"]
        ), key
        if key not in first_steps:
            assert str(m["t_previous"]) == r["previous"], key
            assert str(m["t_previous_within_timeset"]) == (
                r["previous_within_timeset"]
            ), key


def test_stoch_na_domain_and_period_branch(stoch_wf) -> None:
    """§5.3-6: dt_non_anticipativity == the 6 realized rows;
    period__branch.csv == the 6 metadata rows of §4 (DEFECT PIN for
    the (p2040, p2040) self-row and the p2040 fan rows)."""
    dtna = _csv_rows(stoch_wf, "dt_non_anticipativity_set.csv")
    assert dtna == (
        [("p2035", t) for t in P1_STEPS] + [("p2040", t) for t in P2_STEPS]
    )
    pb = _csv_rows(stoch_wf, "period__branch.csv")
    assert pb == EXPECTED_PERIOD_BRANCH


# ---------------------------------------------------------------------------
# ``stoch_horizon`` scenario (output_horizon on — branch rows visible)
# ---------------------------------------------------------------------------


def test_horizon_objective_identical(horizon_wf) -> None:
    """§5.3-7: output_horizon only widens the output gate, not the LP."""
    assert _objective(horizon_wf) == pytest.approx(
        STOCH_OBJECTIVE, rel=1e-6
    )


def test_horizon_branch_dispatch_reached_lp(horizon_wf) -> None:
    """§5.3-8 (DEFECT PIN): the continuation branch profile data
    actually constrained the solver — p2035_low / p2040_low rows with
    wind 20/10 MW and thermal 80/100 MW."""
    flows = _v_flow(horizon_wf)
    assert flows.height == 12
    periods = set(flows["period"].to_list())
    assert {"p2035_low", "p2040_low"} <= periods
    for r in flows.rows(named=True):
        wind_mw = r[WIND_COL] * WIND_EXISTING
        thermal_mw = r[THERMAL_COL] * THERMAL_EXISTING
        expected = {
            "p2035": (60.0, 40.0),
            "p2040": (50.0, 60.0),
            "p2035_low": (20.0, 80.0),
            "p2040_low": (10.0, 100.0),
        }[r["period"]]
        assert wind_mw == pytest.approx(expected[0], abs=1e-6), r
        assert thermal_mw == pytest.approx(expected[1], abs=1e-6), r


# ---------------------------------------------------------------------------
# ``det`` scenario (deterministic control)
# ---------------------------------------------------------------------------


def test_det_control(det_wf) -> None:
    """§5.3-9: deterministic control — objective 43 800 000, flows
    wind 60/50 / thermal 40/60, no branch periods anywhere."""
    assert _objective(det_wf) == pytest.approx(DET_OBJECTIVE, rel=1e-6)
    siu = pl.read_csv(det_wf / "solve_data" / "steps_in_use.csv")
    assert sorted(set(siu["period"].to_list())) == ["p2035", "p2040"]
    flows = _v_flow(det_wf)
    assert sorted(set(flows["period"].to_list())) == ["p2035", "p2040"]
    for r in flows.rows(named=True):
        wind_mw = r[WIND_COL] * WIND_EXISTING
        thermal_mw = r[THERMAL_COL] * THERMAL_EXISTING
        if r["period"] == "p2035":
            assert wind_mw == pytest.approx(60.0, abs=1e-6)
            assert thermal_mw == pytest.approx(40.0, abs=1e-6)
        else:
            assert wind_mw == pytest.approx(50.0, abs=1e-6)
            assert thermal_mw == pytest.approx(60.0, abs=1e-6)
