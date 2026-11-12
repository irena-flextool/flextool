"""A reserve on a group without nodes in the solve is left out, with a warning naming the group.

Before, a reserve group without ``group__node`` rows crashed the build (``FlexData() got multiple values for
keyword argument 'group_node'``, or the reserve feature finding ``group_node`` None).  No node of such a group
can hold its reserve and no flow drives it, so its requirement is left out: the solve equals the one without it.

Hand-built model (on ``tests/fixtures/tests.json``, alternative ``init`` only; never a checked-in ``.sqlite``):
node ``elec`` with a demand of 100 MW, commodity node ``fuel`` at 20 €/MWh, unit ``g1`` fuel -> elec (existing
200 MW, efficiency 0.5).  A timeseries reserve ``primary`` (up, 50 MW) on group ``spin``, which has no
``group__node`` row.  With ``main``: the same reserve also on group ``main`` = {elec}, held by ``g1``.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, utf8

pytestmark = pytest.mark.solver

SPIN = ["primary", "up", "spin"]
MAIN = ["primary", "up", "main"]


def _f(x: float) -> list:
    return b64(float(x), "float")


def _s(x: str) -> list:
    return b64(x, "str")


def _reserve(grp: list) -> list:
    return [["reserve__upDown__group", grp, "reserve_method", _s("timeseries_only")],
            ["reserve__upDown__group", grp, "reservation", _f(50.0)]]


def _model(*, spin: bool, main: bool) -> tuple[list, list]:
    entities = [["node", "elec", None], ["node", "fuel", None], ["commodity", "fuelc", None],
                ["commodity__node", ["fuelc", "fuel"], None],
                ["unit", "g1", None], ["unit__inputNode", ["g1", "fuel"], None],
                ["unit__outputNode", ["g1", "elec"], None]]
    values = [["timeset", "2day", "timeset_duration", b64({"index_type": "str", "data": [["t0001", 2.0]]}, "map")],
              ["node", "elec", "node_type", _s("balance")], ["node", "elec", "inflow", _f(-100.0)],
              ["node", "fuel", "node_type", _s("commodity")], ["commodity", "fuelc", "price", _f(20.0)],
              ["unit", "g1", "efficiency", _f(0.5)], ["unit", "g1", "existing", _f(200.0)]]
    if spin or main:
        entities.append(["reserve", "primary", None])
    if spin:
        entities += [["group", "spin", None], ["reserve__upDown__group", SPIN, None]]
        values += _reserve(SPIN)
    if main:
        rel = ["primary", "up", "g1", "elec"]
        entities += [["group", "main", None], ["group__node", ["main", "elec"], None],
                     ["reserve__upDown__group", MAIN, None], ["reserve__upDown__unit__node", rel, None]]
        values += _reserve(MAIN) + [["reserve__upDown__unit__node", rel, "is_enabled", _s("yes")],
                                    ["reserve__upDown__unit__node", rel, "max_share", _f(1.0)],
                                    ["reserve__upDown__unit__node", rel, "reliability", _f(1.0)]]
    return entities, values


def _run(tmp_path_factory, capfd, alt: str, *, spin: bool, main: bool):
    """Solve and write the csv outputs; returns (step, printed output)."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    entities, values = _model(spin=spin, main=main)
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=["init"], entities=entities)
    work = tmp_path_factory.mktemp(f"{alt}_w")
    capfd.readouterr()
    steps = run_chain_from_db(url, scenario_name=scen, work_folder=work, keep_solutions=True)
    assert len(steps) == 1 and all(s.optimal for s in steps.values()), list(steps)
    step = next(iter(steps.values()))
    write_outputs(scenario_name=scen, output_location=str(work), subdir=scen, write_methods=["csv"],
                  flex_data=step.flex_data, solution=step.solution, solve_name=step.solve_name,
                  flex_data_provider=getattr(step, "flex_data_provider", None),
                  raw_output_dir=str(work / "output_raw"))
    return step, capfd.readouterr().out


def _shortfall_groups(step) -> list[str]:
    try:
        vq = step.solution.value("vq_reserve")
    except KeyError:
        return []
    return sorted(utf8(vq).get_column("g").unique().to_list())


def test_only_reserve_group_without_nodes_is_left_out(tmp_path_factory, capfd) -> None:
    """spin alone: the reserve feature is off, the solve equals the model without a reserve."""
    step, out = _run(tmp_path_factory, capfd, "rgn_spin", spin=True, main=False)
    control, control_out = _run(tmp_path_factory, capfd, "rgn_control", spin=False, main=False)
    assert "reserve group(s) spin have no nodes" in out, out
    assert "have no nodes" not in control_out, control_out
    assert step.flex_data.reserve_upDown_group is None
    assert _shortfall_groups(step) == []
    assert float(step.obj) == pytest.approx(float(control.obj), rel=1e-9)


def test_reserve_group_without_nodes_next_to_one_with_nodes(tmp_path_factory, capfd) -> None:
    """spin and main: only spin is left out; main keeps its requirement, held by g1 (50 MW, no shortfall)."""
    step, out = _run(tmp_path_factory, capfd, "rgn_both", spin=True, main=True)
    main_only, main_out = _run(tmp_path_factory, capfd, "rgn_main", spin=False, main=True)
    assert "reserve group(s) spin have no nodes" in out, out
    assert "have no nodes" not in main_out, main_out
    assert sorted(utf8(step.flex_data.reserve_upDown_group).get_column("g").to_list()) == ["main"]
    assert _shortfall_groups(step) == ["main"]
    held = utf8(step.solution.value("v_reserve"))
    us = utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == "g1")["value"][0]
    assert (held["value"] * us).to_list() == pytest.approx([50.0, 50.0], rel=1e-7)
    assert float(step.obj) == pytest.approx(float(main_only.obj), rel=1e-9)
