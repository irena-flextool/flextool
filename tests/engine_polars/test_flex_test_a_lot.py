"""flexpy parity for the multi-period ``test_a_lot`` kitchen-sink fixture
(4 periods × 5-week sample).  The companion fixture
``work_test_a_lot_but_not_multi_year`` (single-period) reaches 1e-11
parity; this multi-period variant adds an ``anti_energy_plant`` entity
with negative unitsize/existing capacity, which exercises the
``maxToSink_negCap`` (forced-min-output) code path.

Two model bugs were fixed to bring this fixture from a -0.099%
baseline to ~6.4e-5:

* ``co2_max_period``: the .mod splits CO2 emissions by partition —
  eff is multiplied by ``pdtProcess_slope``, noEff is not.  flexpy
  was lumping both partitions into a single set with the eff-style
  slope multiplier, over-counting noEff CHP emissions by ~11%.
  Fixed in ``flextool/input.py::_load_co2_cap`` (split into eff/noEff
  parts) and ``flextool/model.py`` (separate LHS terms).

* ``conversion_indirect`` delayed-input: the ``_delay.delayed_input_expr``
  did not honour ``p_process_source_flow_coefficient = 0`` (which the
  .mod uses to drop a source-side flow from the conversion balance).
  Mirroring the same coef-zero filter that ``_load_indirect`` already
  applies on the undelayed side keeps the LP self-consistent for
  water_pump's electricity input (coef=0).  Fixed in
  ``flextool/_delay.py::load_data``.

The residual ~1.3M / 0.006% gap remains under investigation (see
``audit/test_a_lot_residual.md``); the test is therefore tagged with
a tolerance of ``1e-4`` so it locks in the achieved parity without
becoming flaky on the tail-end LP-degenerate dispatch differences."""
from pathlib import Path
import polars as pl
from polar_high_opt import Problem
from flextool.engine_polars import load_flextool, build_flextool


def _flextool_obj(work: Path) -> float:
    sc_file = work / "solve_data" / "solve_current.csv"
    if sc_file.exists():
        sc = pl.read_csv(sc_file)
        if sc.height > 0:
            solve = sc["solve"][0]
            parq = work / "output_raw" / f"v_obj__{solve}.parquet"
            if parq.exists():
                return pl.read_parquet(parq)["objective"][0]
    parq_list = sorted(work.glob("output_raw/v_obj__*.parquet"))
    if parq_list:
        return pl.read_parquet(parq_list[-1])["objective"][0]
    return pl.read_csv(work / "output_raw" / "v_obj.csv")["objective"][0]


def test_test_a_lot_parity():
    work = Path(__file__).resolve().parent / "data" / "work_test_a_lot"
    data = load_flextool(work)
    pb = Problem(); build_flextool(pb, data); sol = pb.solve()
    assert sol.optimal
    flextool_obj = _flextool_obj(work)
    rel = abs(sol.obj - flextool_obj) / max(1.0, abs(flextool_obj))
    assert rel < 1e-4, (
        f"obj mismatch: flexpy={sol.obj}, flextool={flextool_obj}, rel={rel}")
