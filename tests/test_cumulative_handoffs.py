"""Tests for the rolling ladder accumulator handoff writer (Bug #1 fix).

Unit tests only — the writer's correctness hinges on three independent
pieces (uniform-split realized-MWh attribution, per-period horizon/
realized hours split, prior-accumulator carryover) so the tests exercise
each in isolation with a mocked HiGHS instance and hand-crafted on-disk
parameter CSVs.  The end-to-end validation (real rolling solve, within-
period rolling for both annual and cumulative) lives in
``test_commodity_ladder_rolling.py``.

Test matrix:
    * First-solve empty-prior path → writes header-only files, or adds
      this-roll contributions when v_trade is non-zero.
    * Uniform-split logic (v_trade = 100, 25 % realized → 25 MWh added).
    * Infinite tier (quantity >= 1e29) → no accumulator row written.
    * Non-ladder commodity (``price`` method) → no accumulator row.
    * Second solve with non-empty prior → prior + this-roll = new total.
    * Per-period sim-hours accumulator: zero on first solve, prior +
      this-roll realized hours on second.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from flextool.process_outputs.cumulative_handoffs import (
    _horizon_and_realized_hours,
    _load_commodity_unitsize,
    _load_finite_ladder_tiers,
    _load_ladder_commodities,
    _load_prior_cum_realized_mwh,
    _load_prior_cum_sim_hours,
    write_ladder_rolling_accumulators,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_highs(
    variable_names: list[str], col_values: list[float],
) -> MagicMock:
    h = MagicMock()
    h.allVariableNames.return_value = list(variable_names)
    sol = SimpleNamespace(col_value=list(col_values), col_dual=[], row_dual=[])
    h.getSolution.return_value = sol
    h.getLp.return_value = SimpleNamespace(row_names_=[])
    return h


def _make_workfolder(tmp_path: Path, *, first_solve: bool = True) -> Path:
    (tmp_path / "input").mkdir()
    (tmp_path / "solve_data").mkdir()
    (tmp_path / "input" / "p_model.csv").write_text(
        f"modelParam,p_model\nsolveFirst,{1 if first_solve else 0}\nsolveLast,1\n"
    )
    return tmp_path


def _write_standard_params(
    work: Path,
    *,
    price_methods: dict[str, str],
    ladder_rows: list[tuple[str, int, float, float]],
    unitsize: dict[str, float],
    realized_pairs: list[tuple[str, str]],
    horizon_pairs: list[tuple[str, str, float]],
) -> None:
    """Seed every CSV the writer loads.

    ``realized_pairs`` go into ``realized_dispatch.csv``; ``horizon_pairs``
    go into ``steps_in_use.csv`` as ``(period, step, step_duration)``.
    """
    with open(work / "input" / "p_commodity_price_method.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commodity", "p_commodity_price_method"])
        for c, m in price_methods.items():
            w.writerow([c, m])

    with open(work / "input" / "commodity_ladder.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commodity", "tier", "price", "quantity"])
        for c, tier, price, q in ladder_rows:
            w.writerow([c, tier, price, q])

    with open(work / "input" / "p_commodity_unitsize.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["commodity", "p_commodity_unitsize"])
        for c, us in unitsize.items():
            w.writerow([c, us])

    with open(work / "solve_data" / "realized_dispatch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "step"])
        for d, t in realized_pairs:
            w.writerow([d, t])

    with open(work / "solve_data" / "steps_in_use.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "step", "step_duration"])
        for d, t, dur in horizon_pairs:
            w.writerow([d, t, dur])


# ---------------------------------------------------------------------------
# Loader unit tests
# ---------------------------------------------------------------------------


def test_load_ladder_commodities_includes_annual_and_cumulative(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_standard_params(
        work,
        price_methods={
            "coal": "price_ladder_cumulative",
            "oil": "price_ladder_annual",
            "gas": "price",  # ignored
        },
        ladder_rows=[],
        unitsize={"coal": 1.0},
        realized_pairs=[],
        horizon_pairs=[],
    )
    assert _load_ladder_commodities(work) == {"coal", "oil"}


def test_load_finite_ladder_tiers_drops_infinite_and_non_ladder(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative", "gas": "price"},
        ladder_rows=[
            ("coal", 1, 10.0, 100.0),
            ("coal", 2, 50.0, 1e30),         # infinite tier dropped
            ("gas",  1, 5.0, 999.0),         # non-ladder method dropped
        ],
        unitsize={"coal": 1.0},
        realized_pairs=[],
        horizon_pairs=[],
    )
    assert _load_finite_ladder_tiers(work) == {("coal", 1): 100.0}


def test_load_commodity_unitsize_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    (work / "input" / "p_commodity_unitsize.csv").write_text(
        "commodity,p_commodity_unitsize\ncoal,2.5\noil,10.0\n"
    )
    assert _load_commodity_unitsize(work) == {"coal": 2.5, "oil": 10.0}


def test_load_prior_cum_realized_mwh_header_only(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "ladder_cum_realized_mwh.csv"
    path.write_text("commodity,tier,period,p_ladder_cum_realized_mwh\n")
    assert _load_prior_cum_realized_mwh(path) == {}


def test_load_prior_cum_realized_mwh_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "ladder_cum_realized_mwh.csv"
    path.write_text(
        "commodity,tier,period,p_ladder_cum_realized_mwh\n"
        "coal,1,p2020,42.5\ncoal,1,p2025,7.0\n"
    )
    assert _load_prior_cum_realized_mwh(path) == {
        ("coal", 1, "p2020"): 42.5,
        ("coal", 1, "p2025"): 7.0,
    }


def test_load_prior_cum_sim_hours_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "ladder_cum_sim_hours.csv"
    path.write_text(
        "period,p_ladder_cum_sim_hours\np2020,4.0\np2025,12.5\n"
    )
    assert _load_prior_cum_sim_hours(path) == {"p2020": 4.0, "p2025": 12.5}


# ---------------------------------------------------------------------------
# Horizon / realized-hours helper
# ---------------------------------------------------------------------------


def test_horizon_and_realized_hours_partition() -> None:
    step_duration = {
        ("p2020", "t0001"): 1.0,
        ("p2020", "t0002"): 1.0,
        ("p2020", "t0003"): 1.0,
        ("p2020", "t0004"): 1.0,
    }
    realized_set = {("p2020", "t0001"), ("p2020", "t0002")}
    horizon, realized = _horizon_and_realized_hours(step_duration, realized_set)
    assert horizon == {"p2020": 4.0}
    assert realized == {"p2020": 2.0}


def test_horizon_and_realized_hours_empty_realized() -> None:
    # Lookahead-only horizon: no realized hours in this roll.
    step_duration = {("p2020", "t0001"): 1.0}
    horizon, realized = _horizon_and_realized_hours(step_duration, set())
    assert horizon == {"p2020": 1.0}
    assert realized == {"p2020": 0.0}


# ---------------------------------------------------------------------------
# write_ladder_rolling_accumulators
# ---------------------------------------------------------------------------


def _read_mwh(path: Path) -> dict[tuple[str, int, str], float]:
    rows = list(csv.DictReader(open(path)))
    return {
        (r["commodity"], int(r["tier"]), r["period"]):
            float(r["p_ladder_cum_realized_mwh"])
        for r in rows
    }


def _read_hrs(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(open(path)))
    return {
        r["period"]: float(r["p_ladder_cum_sim_hours"]) for r in rows
    }


def test_first_solve_full_realized_horizon(tmp_path: Path) -> None:
    """Single-roll scenario where every horizon hour is realized.
    Expected: cum_realized_mwh = v_trade * unitsize (no split); cum_sim_hours
    = horizon hours."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[
            ("coal", 1, 10.0, 100.0),
            ("coal", 2, 50.0, 1e30),        # infinite — dropped
        ],
        unitsize={"coal": 2.0},
        realized_pairs=[("p2020", "t0001"), ("p2020", "t0002")],
        horizon_pairs=[("p2020", "t0001", 1.0), ("p2020", "t0002", 1.0)],
    )
    # v_trade[coal, west, p2020, 1] = 30; realized fraction = 2/2 = 1.
    # realized_mwh = 30 * unitsize 2 * 1.0 = 60.
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[30.0],
    )
    write_ladder_rolling_accumulators(h, solve_name="s1", work_folder=work)

    mwh = _read_mwh(work / "solve_data" / "ladder_cum_realized_mwh.csv")
    assert mwh == {("coal", 1, "p2020"): 60.0}

    hrs = _read_hrs(work / "solve_data" / "ladder_cum_sim_hours.csv")
    assert hrs == {"p2020": 2.0}


def test_uniform_split_partial_realized(tmp_path: Path) -> None:
    """v_trade = 100 MWh period-level, but only 25 % of horizon realized.
    Uniform-split assumption → 25 MWh accumulated this roll."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[("coal", 1, 10.0, 500.0)],
        unitsize={"coal": 1.0},
        # 1 of 4 hours realized → fraction = 0.25
        realized_pairs=[("p2020", "t0001")],
        horizon_pairs=[
            ("p2020", "t0001", 1.0),
            ("p2020", "t0002", 1.0),
            ("p2020", "t0003", 1.0),
            ("p2020", "t0004", 1.0),
        ],
    )
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[100.0],
    )
    write_ladder_rolling_accumulators(h, solve_name="s1", work_folder=work)

    mwh = _read_mwh(work / "solve_data" / "ladder_cum_realized_mwh.csv")
    assert mwh == pytest.approx({("coal", 1, "p2020"): 25.0})
    hrs = _read_hrs(work / "solve_data" / "ladder_cum_sim_hours.csv")
    assert hrs == {"p2020": 1.0}


def test_prior_accumulation_second_roll(tmp_path: Path) -> None:
    """Second solve: prior accumulators non-empty → prior + this-roll = new."""
    work = _make_workfolder(tmp_path, first_solve=False)
    # Seed prior roll's CSVs.
    (work / "solve_data" / "ladder_cum_realized_mwh.csv").write_text(
        "commodity,tier,period,p_ladder_cum_realized_mwh\ncoal,1,p2020,25\n"
    )
    (work / "solve_data" / "ladder_cum_sim_hours.csv").write_text(
        "period,p_ladder_cum_sim_hours\np2020,1\n"
    )
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[("coal", 1, 10.0, 500.0)],
        unitsize={"coal": 1.0},
        realized_pairs=[("p2020", "t0002")],
        horizon_pairs=[
            ("p2020", "t0002", 1.0),
            ("p2020", "t0003", 1.0),
        ],
    )
    # v_trade = 20 MWh period-level; realized fraction = 1/2 = 0.5.
    # this-roll contribution = 20 * 1 * 0.5 = 10.  updated = 25 + 10 = 35.
    # this-roll realized hours = 1.0.  updated cum_sim_hours = 1 + 1 = 2.
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[20.0],
    )
    write_ladder_rolling_accumulators(h, solve_name="s2", work_folder=work)

    mwh = _read_mwh(work / "solve_data" / "ladder_cum_realized_mwh.csv")
    assert mwh == pytest.approx({("coal", 1, "p2020"): 35.0})

    hrs = _read_hrs(work / "solve_data" / "ladder_cum_sim_hours.csv")
    assert hrs == pytest.approx({"p2020": 2.0})


def test_lookahead_only_period_not_accumulated(tmp_path: Path) -> None:
    """v_trade on a period with zero realized hours → nothing accumulated
    for that period (pure lookahead has no realized MWh).  The period's
    sim-hours accumulator also stays unchanged."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[("coal", 1, 10.0, 500.0)],
        unitsize={"coal": 1.0},
        # p2020 realized, p2025 is lookahead only (in horizon, no realize).
        realized_pairs=[("p2020", "t0001")],
        horizon_pairs=[
            ("p2020", "t0001", 1.0),
            ("p2025", "t0001", 1.0),
        ],
    )
    h = _fake_highs(
        variable_names=[
            "v_trade[coal,west,p2020,1]",
            "v_trade[coal,west,p2025,1]",
        ],
        col_values=[10.0, 10.0],
    )
    write_ladder_rolling_accumulators(h, solve_name="s1", work_folder=work)

    # Only p2020 contributes; p2025's 10 MWh is lookahead only.
    mwh = _read_mwh(work / "solve_data" / "ladder_cum_realized_mwh.csv")
    assert mwh == pytest.approx({("coal", 1, "p2020"): 10.0})
    hrs = _read_hrs(work / "solve_data" / "ladder_cum_sim_hours.csv")
    # p2025 present at 0 realized hours; writer still records 0.
    assert hrs == pytest.approx({"p2020": 1.0, "p2025": 0.0})


def test_no_ladder_commodities_writes_header_only(tmp_path: Path) -> None:
    """Every commodity uses scalar price → header-only CSVs."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"gas": "price"},
        ladder_rows=[],
        unitsize={"gas": 1.0},
        realized_pairs=[("p2020", "t0001")],
        horizon_pairs=[("p2020", "t0001", 1.0)],
    )
    h = _fake_highs(variable_names=[], col_values=[])
    write_ladder_rolling_accumulators(h, solve_name="s1", work_folder=work)

    mwh_text = (work / "solve_data" / "ladder_cum_realized_mwh.csv").read_text()
    hrs_text = (work / "solve_data" / "ladder_cum_sim_hours.csv").read_text()
    assert mwh_text.strip() == "commodity,tier,period,p_ladder_cum_realized_mwh"
    assert hrs_text.strip() == "period,p_ladder_cum_sim_hours"


def test_pools_v_trade_across_nodes_and_tiers(tmp_path: Path) -> None:
    """Ladder is commodity-level — MWh pools across all (c, n) pairs for
    each tier, with separate rows per tier."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[
            ("coal", 1, 10.0, 500.0),
            ("coal", 2, 20.0, 500.0),
        ],
        unitsize={"coal": 1.0},
        realized_pairs=[("p2020", "t0001")],
        horizon_pairs=[("p2020", "t0001", 1.0)],
    )
    # Tier 1: 5 at west + 7 at east = 12.  Tier 2: 3 at west only.
    h = _fake_highs(
        variable_names=[
            "v_trade[coal,west,p2020,1]",
            "v_trade[coal,east,p2020,1]",
            "v_trade[coal,west,p2020,2]",
        ],
        col_values=[5.0, 7.0, 3.0],
    )
    write_ladder_rolling_accumulators(h, solve_name="s1", work_folder=work)

    mwh = _read_mwh(work / "solve_data" / "ladder_cum_realized_mwh.csv")
    assert mwh == pytest.approx({
        ("coal", 1, "p2020"): 12.0,
        ("coal", 2, "p2020"): 3.0,
    })
