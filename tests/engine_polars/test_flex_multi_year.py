"""flexpy ``multi_year`` parity — multi-solve fixture; the
``solve_data/`` snapshot in ``work_multi_year`` reflects the *last*
solve in the chain (``y2035_5week``).  After ``_cumulative_invest``
was wired in (merge step 3), comparing the y2035 solve closes within
1e-6.  Earlier solves (y2020, y2025, y2030) require multi-solve
handoff (``p_entity_previously_invested_capacity`` /
``p_entity_divested``) which is not yet replayed in flexpy."""
from pathlib import Path
import polars as pl
from polar_high import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver

WORK = Path(__file__).resolve().parent / "data" / "work_multi_year"


def _flextool_obj():
    # Last solve = y2035_5week (matches the solve_data/ snapshot).
    p = WORK / "output_raw" / "v_obj__y2035_5week.parquet"
    return pl.read_parquet(p)["objective"][0]


def test_multi_year_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
