"""Tests for the rolling cumulative-quota handoff writer.

Unit tests only — the running-balance formula's correctness hinges on
three independent pieces (allotment ratio, realized consumption, prior
carryover) so the tests exercise each in isolation with a mocked HiGHS
instance and hand-crafted on-disk parameter CSVs.  The end-to-end
validation (real rolling solve, two-roll underspend + overspend
assertions) is step 4e in the project plan and lands in a separate
commit.

Test matrix:
    * Single-period solve where ``span_weight == total_weight`` →
      allotment equals the full cap, and ``remaining = cap − consumption``.
    * First solve with header-only prior CSV → ``prior_q == 0``.
    * Infinite tier (quantity >= 1e29) → silently dropped from output.
    * Commodity without ``price_ladder_cumulative`` price method →
      ignored even if it appears in ``commodity_ladder.csv``.
    * Second solve with non-empty prior CSV → prior_remaining flows
      into the new row.
    * Missing ``cumulative_weight_total.csv`` → header-only seed (no
      arithmetic possible, constraint stays inactive next roll).
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from flextool.process_outputs.cumulative_handoffs import (
    _load_commodity_unitsize,
    _load_cumulative_weight_total,
    _load_price_ladder_cumulative,
    _load_prior_cumulative_ladder_remaining,
    _span_weight,
    write_cumulative_ladder_remaining,
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
    years_map: dict[str, float],
    period_share: dict[str, float],
    total_weight: float | None,
    realized_periods: list[str],
) -> None:
    """Seed every CSV the writer loads.

    ``ladder_rows`` is a list of ``(commodity, tier, price, quantity)``
    tuples matching the ``commodity,tier,price,quantity`` layout written
    by ``input_writer._write_commodity_ladder``.
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

    with open(work / "solve_data" / "p_years_represented_d.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solve", "period", "value"])
        for d, yrs in years_map.items():
            w.writerow(["s1", d, yrs])

    with open(work / "solve_data" / "complete_period_share_of_year.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["solve", "period", "value"])
        for d, share in period_share.items():
            w.writerow(["s1", d, share])

    if total_weight is not None:
        with open(work / "solve_data" / "cumulative_weight_total.csv", "w", newline="") as f:
            f.write("total_weight\n")
            f.write(f"{total_weight}\n")

    with open(work / "solve_data" / "realized_dispatch.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "step"])
        for d in realized_periods:
            # Single fake step per period — the writer only looks at the
            # period projection of realized_dispatch.csv.
            w.writerow([d, "t0001"])


# ---------------------------------------------------------------------------
# Loader unit tests
# ---------------------------------------------------------------------------


def test_load_price_ladder_cumulative_filters_on_method(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative", "gas": "price"},
        ladder_rows=[
            ("coal", 1, 10.0, 100.0),
            ("coal", 2, 50.0, 1e30),         # infinite tier dropped
            ("gas",  1, 5.0, 999.0),         # non-cumulative method dropped
        ],
        unitsize={"coal": 1.0},
        years_map={"p2020": 1.0},
        period_share={"p2020": 1.0},
        total_weight=1.0,
        realized_periods=["p2020"],
    )
    assert _load_price_ladder_cumulative(work) == {("coal", 1): 100.0}


def test_load_commodity_unitsize_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    (work / "input" / "p_commodity_unitsize.csv").write_text(
        "commodity,p_commodity_unitsize\ncoal,2.5\noil,10.0\n"
    )
    assert _load_commodity_unitsize(work) == {"coal": 2.5, "oil": 10.0}


def test_load_cumulative_weight_total_missing_returns_none(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    assert _load_cumulative_weight_total(work) is None


def test_load_cumulative_weight_total_zero_returns_none(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    (work / "solve_data" / "cumulative_weight_total.csv").write_text(
        "total_weight\n0\n"
    )
    assert _load_cumulative_weight_total(work) is None


def test_load_cumulative_weight_total_positive(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    (work / "solve_data" / "cumulative_weight_total.csv").write_text(
        "total_weight\n12.5\n"
    )
    assert _load_cumulative_weight_total(work) == 12.5


def test_load_prior_cumulative_ladder_remaining_header_only(tmp_path: Path) -> None:
    """Header-only seed → empty prior dict."""
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "cumulative_ladder_remaining.csv"
    path.write_text("commodity,tier,p_cumulative_ladder_remaining\n")
    assert _load_prior_cumulative_ladder_remaining(path) == {}


def test_load_prior_cumulative_ladder_remaining_round_trip(tmp_path: Path) -> None:
    work = _make_workfolder(tmp_path)
    path = work / "solve_data" / "cumulative_ladder_remaining.csv"
    path.write_text(
        "commodity,tier,p_cumulative_ladder_remaining\ncoal,1,42.5\ncoal,2,-3.0\n"
    )
    assert _load_prior_cumulative_ladder_remaining(path) == {
        ("coal", 1): 42.5, ("coal", 2): -3.0,
    }


# ---------------------------------------------------------------------------
# Span-weight helper
# ---------------------------------------------------------------------------


def test_span_weight_sums_years_over_share_over_realized() -> None:
    realized = {"p2020", "p2025"}
    years = {"p2020": 2.0, "p2025": 5.0, "p2030": 10.0}  # p2030 not realized
    share = {"p2020": 1.0, "p2025": 0.5, "p2030": 1.0}
    # realized contributions: 2/1 + 5/0.5 = 2 + 10 = 12.
    assert _span_weight(realized, years, share) == 12.0


def test_span_weight_skips_zero_share() -> None:
    realized = {"p2020"}
    years = {"p2020": 1.0}
    share = {"p2020": 0.0}  # guard in _span_weight skips div-by-zero
    assert _span_weight(realized, years, share) == 0.0


# ---------------------------------------------------------------------------
# write_cumulative_ladder_remaining
# ---------------------------------------------------------------------------


def _read_output(path: Path) -> dict[tuple[str, int], float]:
    rows = list(csv.DictReader(open(path)))
    return {
        (r["commodity"], int(r["tier"])): float(r["p_cumulative_ladder_remaining"])
        for r in rows
    }


def test_first_solve_span_equals_total_weight(tmp_path: Path) -> None:
    """Single-period solve where the realized span covers the whole
    horizon → span_weight == total_weight → allotment == cap.
    Expected: remaining = 0 (prior) + cap * 1.0 - consumption."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[
            ("coal", 1, 10.0, 100.0),   # finite, binding tier
            ("coal", 2, 50.0, 1e30),    # infinite tail — dropped from output
        ],
        unitsize={"coal": 2.0},
        years_map={"p2020": 1.0},
        period_share={"p2020": 1.0},   # total_weight = 1 / 1 = 1.0
        total_weight=1.0,
        realized_periods=["p2020"],
    )

    # v_trade[coal, west, p2020, 1] = 30 — both in MWh / unitsize units.
    # consumption = 30 * unitsize 2 * years 1 / share 1 = 60.
    # allot = 100 * 1 / 1 = 100.
    # remaining = 0 + 100 - 60 = 40.
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[30.0],
    )
    write_cumulative_ladder_remaining(h, solve_name="s1", work_folder=work)
    out = _read_output(work / "solve_data" / "cumulative_ladder_remaining.csv")
    assert out == {("coal", 1): 40.0}


def test_prior_carryover_accumulates(tmp_path: Path) -> None:
    """Second solve: prior CSV non-empty → ``prior + allot - consumed``."""
    work = _make_workfolder(tmp_path, first_solve=False)
    # Prior roll left remaining = 25.  span/total = 0.5 this roll;
    # total_cap = 100.  v_trade sums to 10 (unitsize 1, yrs/share 1).
    # allot = 100 * 0.5 = 50.  remaining = 25 + 50 - 10 = 65.
    (work / "solve_data" / "cumulative_ladder_remaining.csv").write_text(
        "commodity,tier,p_cumulative_ladder_remaining\ncoal,1,25\n"
    )
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[("coal", 1, 10.0, 100.0)],
        unitsize={"coal": 1.0},
        years_map={"p2020": 1.0, "p2025": 1.0},
        period_share={"p2020": 1.0, "p2025": 1.0},
        total_weight=2.0,  # 1/1 + 1/1 across dt_complete's two periods
        realized_periods=["p2020"],  # this roll realizes one period → span 1.0
    )
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[10.0],
    )
    write_cumulative_ladder_remaining(h, solve_name="s2", work_folder=work)
    out = _read_output(work / "solve_data" / "cumulative_ladder_remaining.csv")
    assert out[("coal", 1)] == pytest.approx(65.0)


def test_overspend_yields_negative_remaining(tmp_path: Path) -> None:
    """Realized-span consumption > allotment → negative remaining
    (legal, forces the tier out in the next roll)."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[("coal", 1, 10.0, 50.0)],
        unitsize={"coal": 1.0},
        years_map={"p2020": 1.0, "p2025": 1.0},
        period_share={"p2020": 1.0, "p2025": 1.0},
        total_weight=2.0,
        realized_periods=["p2020"],
    )
    # Consumed 40 MWh in the realized span; allot = 50 * 0.5 = 25.
    # remaining = 0 + 25 - 40 = -15.
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[40.0],
    )
    write_cumulative_ladder_remaining(h, solve_name="s1", work_folder=work)
    out = _read_output(work / "solve_data" / "cumulative_ladder_remaining.csv")
    assert out[("coal", 1)] == pytest.approx(-15.0)


def test_no_cumulative_commodities_writes_header_only(tmp_path: Path) -> None:
    """Every commodity uses scalar price → header-only CSV, no rows."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"gas": "price"},
        ladder_rows=[],
        unitsize={"gas": 1.0},
        years_map={"p2020": 1.0},
        period_share={"p2020": 1.0},
        total_weight=1.0,
        realized_periods=["p2020"],
    )
    h = _fake_highs(variable_names=[], col_values=[])
    write_cumulative_ladder_remaining(h, solve_name="s1", work_folder=work)
    text = (work / "solve_data" / "cumulative_ladder_remaining.csv").read_text()
    assert text.strip() == "commodity,tier,p_cumulative_ladder_remaining"


def test_missing_total_weight_writes_header_only(tmp_path: Path) -> None:
    """First solve where cumulative_weight_total.csv was never written
    (e.g. a seed bug) → writer bails out with header-only CSV so the
    constraint stays inactive on the next roll."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[("coal", 1, 10.0, 100.0)],
        unitsize={"coal": 1.0},
        years_map={"p2020": 1.0},
        period_share={"p2020": 1.0},
        total_weight=None,  # file absent
        realized_periods=["p2020"],
    )
    h = _fake_highs(
        variable_names=["v_trade[coal,west,p2020,1]"],
        col_values=[30.0],
    )
    write_cumulative_ladder_remaining(h, solve_name="s1", work_folder=work)
    text = (work / "solve_data" / "cumulative_ladder_remaining.csv").read_text()
    assert text.strip() == "commodity,tier,p_cumulative_ladder_remaining"


def test_pools_v_trade_across_nodes_and_into_per_tier_sums(tmp_path: Path) -> None:
    """Ladder is commodity-level — consumption pools across all (c, n)
    pairs for each tier."""
    work = _make_workfolder(tmp_path, first_solve=True)
    _write_standard_params(
        work,
        price_methods={"coal": "price_ladder_cumulative"},
        ladder_rows=[
            ("coal", 1, 10.0, 100.0),
            ("coal", 2, 20.0, 50.0),
        ],
        unitsize={"coal": 1.0},
        years_map={"p2020": 1.0},
        period_share={"p2020": 1.0},
        total_weight=1.0,
        realized_periods=["p2020"],
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
    write_cumulative_ladder_remaining(h, solve_name="s1", work_folder=work)
    out = _read_output(work / "solve_data" / "cumulative_ladder_remaining.csv")
    # Tier 1: remaining = 100 - 12 = 88.  Tier 2: 50 - 3 = 47.
    assert out[("coal", 1)] == pytest.approx(88.0)
    assert out[("coal", 2)] == pytest.approx(47.0)
