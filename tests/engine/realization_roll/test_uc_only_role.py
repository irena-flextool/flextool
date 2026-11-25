"""T-d3a: solve (1) of a rolling ``separate_dispatch`` solve has the ``uc_only`` non-anticipativity role (design rev
7.1 §2.5, D3): across its forecasts it ties only the units online of the ``_fix_in_uc`` units.

Model: lp_snapshot's patch ``rsd_storage_reserve`` = ``rolling_separate_dispatch.json`` (3 rolls, jump 24 h, horizon
48 h; forecasts ``upper`` / ``lower`` / ``mid`` made at each roll's first step, so 2 forecast pairs of the reference
forecast) with ``hydro_reservoir`` a storage in the stochastic group, an upward reserve on ``demand_node`` offered by
the hydro, coal and gas units, and ``gas_plant`` ``linear`` (``coal_plant`` stays ``integer_fix_in_uc``).

Hand count of each solve (1) (the 24 committed jump steps are the tied steps):
* ``non_anticipativity_online_integer``: coal only, 1 unit x 2 pairs x 24 steps = 48 rows (as before D3).
* no ``non_anticipativity_online_linear`` (gas is not ``_fix_in_uc``; before: 48), ``_reserve`` (before: 3
  providers x 2 x 24 = 144), ``_storage_use`` (before: 1 node x 2 x 24 = 48), ``_storage_state_first`` (before:
  2 in rolls 0-1, where the 24 tied steps are fewer than the 48 state steps) or ``_rp_block_start``.
The realization rolls keep their pins of coal (``online_fix_eq_integer``, 24 per roll) and none of gas.

Free start: without ``hydro_reservoir``'s ``storage_start_end_method`` and ``storage_solve_horizon_method`` (the
fixture's ``fix_start`` and ``use_reference_value`` each give every forecast the same start already; absent =
``fix_nothing``, ``free``), roll 0 has no carried state, so its three forecasts would each pick their own
start level; with no ``non_anticipativity_storage_state_first`` row, ``storage_state_fan_start`` ties every other
forecast to the first sorted one (``period1_lower``): 2 rows, and the three start levels (v_state at t0001, the
first step of a bind_forward_only node has no state change) are equal.  Rolls 1-2 start from the carried level.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm

pytestmark = pytest.mark.solver

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "rolling_separate_dispatch.json"
SCENARIO = "rsd_storage_reserve"
UNTIED = ("non_anticipativity_online_linear", "non_anticipativity_reserve", "non_anticipativity_storage_use",
          "non_anticipativity_storage_state_first", "non_anticipativity_rp_block_start")


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _data(free_start: bool) -> dict:
    lps = scm.lp_snapshot()
    data = lps._patch_rsd_storage_reserve(json.loads(FIXTURE.read_text(encoding="utf-8")))
    if free_start:
        free = ("storage_start_end_method", "storage_solve_horizon_method")
        data["parameter_values"] = [r for r in data["parameter_values"]
                                    if not (r[:2] == ["node", "hydro_reservoir"] and r[2] in free)]
    return data


def _run(data: dict, folder: Path):
    path = folder / "rsd.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    built, steps = scm.run_capturing(scm.build_db(path, folder), SCENARIO, folder)
    lps = scm.lp_snapshot()
    uc = [lps.parse_mps(p) for s, p, _ in built if "__realized_dispatch" not in s]
    real = [lps.parse_mps(p) for s, p, _ in built if "__realized_dispatch" in s]
    return uc, real, steps


def _families(lp) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in lp.rows:
        fam = scm.lp_snapshot().family_of(row)
        out[fam] = out.get(fam, 0) + 1
    return out


@pytest.fixture(scope="module")
def tied_start(tmp_path_factory):
    return _run(_data(False), tmp_path_factory.mktemp("rsd_uc_only"))


@pytest.fixture(scope="module")
def free_start(tmp_path_factory):
    return _run(_data(True), tmp_path_factory.mktemp("rsd_uc_only_free"))


def test_solve_one_ties_only_fix_in_uc_units_online(tied_start):
    uc, real, steps = tied_start
    assert len(uc) == 3 and len(real) == 3
    assert all(s.solution.optimal for s in steps.values())
    for k, lp in enumerate(uc):
        fams = _families(lp)
        assert fams.get("non_anticipativity_online_integer") == 48, k
        assert not {f: fams[f] for f in UNTIED if f in fams}, k
        online = [r for r in lp.rows if r.startswith("non_anticipativity_online_integer")]
        assert all("coal_plant" in r for r in online), k
    for k, lp in enumerate(real):
        pins = [r for r in lp.rows if r.startswith("online_fix_eq")]
        assert len(pins) == 24 and all(r.startswith("online_fix_eq_integer[coal_plant,") for r in pins), k


def test_free_start_levels_tied_by_fan_start(free_start):
    uc, _real, steps = free_start
    fams = [_families(lp) for lp in uc]
    assert fams[0].get("storage_state_fan_start") == 2
    assert all("storage_state_fan_start" not in f for f in fams[1:])
    assert not any(f in fam for fam in fams for f in UNTIED)
    roll0 = next(s for name, s in steps.items() if "__realized_dispatch" not in name)
    state = (roll0.solution.value("v_state")
             .with_columns(pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t"))
             .filter((pl.col("n") == "hydro_reservoir") & (pl.col("t") == "t0001")))
    assert sorted(state["d"].to_list()) == ["period1_lower", "period1_mid", "period1_upper"]
    levels = state["value"].to_list()
    assert max(levels) - min(levels) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("uc_only, kept", [(False, {"uc", "lin", "res"}), (True, {"uc"})])
def test_coarse_processes_kept_fine_only_where_tied(uc_only, kept):
    """``flex_data.coarse_tie_straddles``: a coarse step (b1 = t1-t2) that the tie covers only in part (the tied step
    is t1) keeps the processes tied there at the solve's step.  Full role: every online process and reserve provider;
    ``uc_only``: only the ``_fix_in_uc`` unit ``uc`` (``lin`` is online but not ``_fix_in_uc``, ``res`` offers
    reserve; neither is tied)."""
    from types import SimpleNamespace

    from flextool.engine.preprocess.flex_data import coarse_tie_straddles

    def frame(**cols):
        return pl.DataFrame(cols)

    procs = ["uc", "lin", "res"]
    d = SimpleNamespace(
        na_online_fix_in_uc=uc_only,
        process_source_sink=frame(p=procs, source=procs, sink=["elec"] * 3),
        dt_non_anticipativity=frame(d=["p"], t=["t1"]),
        period_branch_full=frame(d=["p", "p", "p"], b=["p", "p_A", "p_B"]),
        period_in_use_set=frame(d=["p_A", "p_B"]),
        process_online=frame(p=["uc", "lin"]),
        process_online_fix_in_uc=frame(p=["uc"]),
        process_reserve_upDown_node_active=frame(p=["res"]),
        group_node=None, groupStochastic=None, nodeBalance=None)
    steps = frame(p=[p for p in procs for _ in range(4)],
                  d=["p_A", "p_A", "p_B", "p_B"] * 3,
                  b_first=["t1"] * 12,
                  t=["t1", "t2"] * 6)
    assert coarse_tie_straddles(d, steps) == kept
