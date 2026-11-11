"""A period-Map ``existing`` gives the same unit size and unit count as the equivalent constant, in every solve mode.

Without ``virtual_unitsize`` an entity is one unit of its ``existing`` capacity (the largest over the periods for a
period Map).  The per-solve ``solve_data/p_entity_unitsize.csv`` read ``existing`` from the constant table only, so
a period-Map ``existing`` got the 1000 MW fallback there.  A single solve rebuilt the unit size from the database
and hid it; the rolls of a rolling solve kept it, so a 17 MW unit became 17/1000 = 0.017 units and an integer
(binary start-up) unit could never be online: a ``min_instant_flow`` floor it has to meet made roll 1 infeasible.

Hand derivation: coal_plant has ``existing`` 17 MW (constant, or the Map ``{p2020: 17, p2025: 17}``) and no
``virtual_unitsize``, so it is one unit of 17 MW: ``p_unitsize`` = 17 and ``p_process_existing_count`` in p2020 =
17 / 17 = 1.  With availability 0.89 it gives at most 0.89 × 17 = 15.13 MW, which meets the 10 MW floor only when
its one unit is online.  Fixtures are ``tests.json`` plus one override alternative, built under the pytest tmp dir.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, RunCache, b64, uc, utf8

pytestmark = pytest.mark.solver

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


SOLVE = "y2020_2day_dispatch"
FLOOR_GROUP = "sync_floor"
FLOOR_ENTITIES = [["flowGroup", FLOOR_GROUP, None],
                  ["flowGroup__unit__node", [FLOOR_GROUP, PLANT, "west"], None]]
ROLLING = [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
           ["solve", SOLVE, "rolling_solve_horizon", b64(24.0, "float")],
           ["solve", SOLVE, "rolling_solve_jump", b64(12.0, "float")]]


def _existing(shape: str, p2020: float, p2025: float) -> list:
    if shape == "constant":
        assert p2020 == p2025
        return b64(p2020, "float")
    return b64({"index_type": "str", "index_name": "period", "data": [["p2020", p2020], ["p2025", p2025]]}, "map")


def _values(shape: str, mode: str, p2020: float = 17.0, p2025: float = 17.0, *, floor: bool = True) -> list:
    values = uc(PLANT, 0.0, "binary") + [
        ["unit", PLANT, "existing", _existing(shape, p2020, p2025)],
        ["unit", PLANT, "availability", b64(0.89, "float")],
    ]
    if floor:
        values.append(["flowGroup", FLOOR_GROUP, "min_instant_flow", b64(10.0, "float")])
    return values + (ROLLING if mode == "rolling" else [])


def _steps(tmp_path_factory, shape: str, mode: str) -> dict:
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=f"uc17_{shape}_{mode}", values=_values(shape, mode),
                                  base_chain=PLANT_CHAIN, entities=FLOOR_ENTITIES)
    return steps


def _plant(param) -> list[dict]:
    return utf8(param.frame).filter(pl.col("p") == PLANT).drop_nulls().to_dicts()


@pytest.mark.parametrize("mode", ["single", "rolling"])
@pytest.mark.parametrize("shape", ["constant", "map"])
def test_one_unit_of_17_mw(tmp_path_factory, shape, mode):
    steps = _steps(tmp_path_factory, shape, mode)
    assert len(steps) == (1 if mode == "single" else 4), list(steps)
    for name, step in steps.items():
        assert step.optimal, f"{name} is not optimal"
        fd = step.flex_data
        assert _plant(fd.p_unitsize) == [{"p": PLANT, "value": 17.0}], name
        assert _plant(fd.p_process_existing_count) == [{"p": PLANT, "d": "p2020", "value": 1.0}], name
        all_us = utf8(fd.p_all_entity_unitsize.frame).filter(pl.col("e") == PLANT)
        assert all_us["value"].to_list() == [17.0], name


@pytest.mark.parametrize("mode", ["single", "rolling"])
def test_map_and_constant_give_the_same_solution(tmp_path_factory, mode):
    const = _steps(tmp_path_factory, "constant", mode)
    as_map = _steps(tmp_path_factory, "map", mode)
    assert list(const) == list(as_map)
    for name in const:
        assert as_map[name].obj == pytest.approx(const[name].obj, rel=1e-12), name
        on_c = utf8(const[name].solution.value("v_online_integer")).sort("p", "d", "t")
        on_m = utf8(as_map[name].solution.value("v_online_integer")).sort("p", "d", "t")
        assert on_m.equals(on_c), name


def test_fractional_unit_count_warns_once_per_run(tmp_path_factory, capsys):
    """existing ``{p2020: 8.5, p2025: 17}`` without virtual_unitsize: one unit of 17 MW (the largest), of which
    8.5 / 17 = 0.5 exists in p2020.  Integer online cannot use half a unit: one warning for the whole rolling run
    (4 rolls), naming the unit, the count and the unit size."""
    capsys.readouterr()
    _RUNS.get(tmp_path_factory, alt="uc_half_rolling", values=_values("map", "rolling", 8.5, 17.0, floor=False),
              base_chain=PLANT_CHAIN)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "not a whole number of units" in ln]
    assert len(lines) == 1, lines
    assert f"unit '{PLANT}'" in lines[0]
    assert "0.5 in p2020 units of 17 MW" in lines[0]


def test_whole_unit_count_does_not_warn(tmp_path_factory, capsys):
    capsys.readouterr()
    _RUNS.get(tmp_path_factory, alt="uc_whole_rolling", values=_values("map", "rolling", 17.0, 8.5, floor=False),
              base_chain=PLANT_CHAIN)
    assert [ln for ln in capsys.readouterr().out.splitlines() if "not a whole number of units" in ln] == []


def test_emitted_unitsize_reads_constant_and_period_map_existing():
    """``solve_data/p_entity_unitsize.csv`` for units, connections and storage nodes, from the per-solve tables:
    ``virtual_unitsize`` when non-zero, else ``existing`` (a constant from ``p_<class>``, a period Map from
    ``pd_<class>`` reduced to its largest value), else 1000; 0 for an entity that is neither a process nor a node.
    Hand values: u_const 17; u_map max(0, 17) = 17 (a Map that starts at 0 still names the unit size); u_vu 5 (its
    virtual_unitsize wins over existing 40); u_none 1000; n_map max(300, 200) = 300; n_const 50; c_zero (a Map of
    zeros) 1000; extra 0."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.engine.preprocess.emit._dispatchers import _compute_p_entity_unitsize
    from flextool.engine.preprocess.emit._pdt_lookup import PdLookup

    prov = FlexDataProvider()
    pd_schema = ["entity", "param", "period", "value"]
    prov.put("input/p_process", pl.DataFrame(
        [("u_const", "existing", 17.0), ("u_vu", "existing", 40.0), ("u_vu", "virtual_unitsize", 5.0),
         ("u_none", "efficiency", 0.4)], schema=["process", "processParam", "p_process"], orient="row"))
    prov.put("input/pd_process", pl.DataFrame(
        [("u_map", "existing", "p2020", 0.0), ("u_map", "existing", "p2025", 17.0),
         ("c_zero", "existing", "p2020", 0.0), ("c_zero", "existing", "p2025", 0.0)],
        schema=pd_schema, orient="row"))
    prov.put("input/p_node", pl.DataFrame(
        [("n_const", "existing", 50.0)], schema=["node", "nodeParam", "p_node"], orient="row"))
    prov.put("input/pd_node", pl.DataFrame(
        [("n_map", "existing", "p2020", 300.0), ("n_map", "existing", "p2025", 200.0)],
        schema=pd_schema, orient="row"))
    lookups = {cls: PdLookup(pd_csv=Path(f"input/pd_{cls}.csv"), p_csv=Path(f"input/p_{cls}.csv"),
                             period_branch_csv=Path("solve_data/period__branch.csv"), provider=prov)
               for cls in ("process", "node")}
    entities = ["u_const", "u_map", "u_vu", "u_none", "n_map", "n_const", "c_zero", "extra"]
    inp = {"pp": lookups["process"], "pn": lookups["node"], "entities": entities,
           "process_set": frozenset({"u_const", "u_map", "u_vu", "u_none", "c_zero"}),
           "node_set": frozenset({"n_map", "n_const"})}

    out = _compute_p_entity_unitsize(inp)

    assert out.columns == ["entity", "value"]
    assert out.rows() == [("u_const", "17.0"), ("u_map", "17.0"), ("u_vu", "5.0"), ("u_none", "1000.0"),
                          ("n_map", "300.0"), ("n_const", "50.0"), ("c_zero", "1000.0"), ("extra", "0.0")]
