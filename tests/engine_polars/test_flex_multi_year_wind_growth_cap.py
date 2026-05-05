"""flexpy ``multi_year_wind_growth_cap`` parity — the ``wind_growth_cap``
user-defined ``process_constraint_less_than`` requires the cumulative-
prior-invest contribution to its prebuilt-capacity LHS (mod:2885-2898):
``v_invest[p,d] - 0.1 * (existing[p,d] + Σ_{d_invest<d} v_invest[p,d_invest]) ≤ 0``.
The static-existing half was already wired; the cumulative-invest
variable summand was added in model.py via the lookback-Sum-over-Var
pattern (rename d→d_invest, join on edd_invest_lookback_set, Sum)."""
from pathlib import Path
import polars as pl
from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool
import pytest

pytestmark = pytest.mark.solver

WORK = Path(__file__).resolve().parent / "data" / "work_multi_year_wind_growth_cap"


def _flextool_obj():
    parq = list(WORK.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return pl.read_parquet(parq[0])["objective"][0]
    return pl.read_csv(WORK / "output_raw" / "v_obj.csv")["objective"][0]


def test_multi_year_wind_growth_cap_parity():
    data = load_flextool(WORK)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj()
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-6, f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}"
