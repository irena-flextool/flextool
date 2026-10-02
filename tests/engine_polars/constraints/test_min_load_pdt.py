"""End-to-end guard for the ``unit.min_load`` Map double-count fix.

Background (``specs/fix_map_reading.md`` §5): ``p_min_load`` used to be a
``(p,)`` Direct Param whose scalar helper
(``_direct_params._entity_scalar_explicit``) did ``.select(dim, "value")`` and
**dropped the Map index column**.  A period Map ``{p1: 0.4, p2: 0.3}`` then
unrolled to two ``(unit, value)`` rows for the same unit, and polar_high
**summed** the duplicate ``(p)`` keys in the ``minFlow_minload`` constraint
(``model.py``) — producing a floor of 0.7 in *every* period.

The fix routes ``p_min_load`` through the efficiency shape resolver
(``_derived_params.p_min_load_pdt_from_source`` → ``_eff_param_to_pdt``) so it
becomes a shape-correct ``(p, d, t)`` Param, exactly like ``p_slope`` /
``p_section`` already are.

These tests exercise the full chain: the producer reads the Map correctly,
and the real ``minFlow_minload`` constraint applies the correct per-period /
per-timestep floor.

Model geometry (per repro): a single ``online_integer`` unit ``u`` with
``unitsize = 100``, node demand 10 MW/step (= 0.1 per unitsize, well below the
floor) and *cheap* down-penalty.  With the floor binding and excess dumped,
``v_flow`` (per-unitsize) reads back the applied floor directly.
"""
from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool, InMemoryReader
from flextool.engine_polars import _derived_params as drv
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


# Two periods, two steps each — enough to show a period Map differ per period.
_PERIODS = ["d1", "d2"]
_STEPS = 2
_DT = pl.DataFrame(
    [{"d": d, "t": f"t{k:02d}"} for d in _PERIODS for k in range(1, _STEPS + 1)]
)


def _build_uc_2period(p_min_load_param: Param) -> FlexData:
    """Single ``online_integer`` unit over 2 periods × 2 steps, floor-binding.

    ``p_min_load_param`` is injected verbatim (the thing under test).  Demand
    is low and down-slack cheap, so ``v_flow`` reveals the floor.
    """
    dt = _DT
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",),
        pl.DataFrame({"d": _PERIODS, "value": [1.0] * len(_PERIODS)}))
    p_psh = Param(("d",),
        pl.DataFrame({"d": _PERIODS, "value": [1.0] * len(_PERIODS)}))

    nb = pl.DataFrame({"n": ["n"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(-10.0)).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    # Cheap down-penalty so the unit can dump the above-demand floor output.
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.01)).select("n", "d", "t", "value"))

    pss = pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u"], "source": ["FUEL_n"], "sink": ["n"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [100.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").with_columns(value=pl.lit(1.0)))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    process_online = pl.DataFrame({"p": ["u"]})
    process_minload = pl.DataFrame({"p": ["u"]})
    p_online_dt = pss_dt.select("p", "d", "t").unique()
    # Integer commitment → v_online ∈ {0, 1} so the floor binds crisply.
    process_online_integer = pl.DataFrame({"p": ["u"]})
    pdt_online_integer = p_online_dt.clone()

    p_startup_cost = Param(("p", "d"),
        pl.DataFrame({"p": ["u"] * len(_PERIODS), "d": _PERIODS,
                      "value": [5.0] * len(_PERIODS)}))
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["u"] * len(_PERIODS), "d": _PERIODS,
                      "value": [1.0] * len(_PERIODS)}))

    # Cyclic dtttdt within each period (2 steps each).
    rows = []
    for d in _PERIODS:
        rows += [
            {"d": d, "t": "t01", "t_previous": "t02",
             "t_previous_within_timeset": "t02", "d_previous": d,
             "t_previous_within_solve": "t02"},
            {"d": d, "t": "t02", "t_previous": "t01",
             "t_previous_within_timeset": "t01", "d_previous": d,
             "t_previous_within_solve": "t01"},
        ]
    dtttdt = pl.DataFrame(rows)

    return FlexData(
        dt=dt, p_step_duration=p_step, p_timestep_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        process_online=process_online,
        process_online_integer=process_online_integer,
        process_minload=process_minload,
        p_online_dt=p_online_dt,
        pdt_online_integer=pdt_online_integer,
        p_min_load=p_min_load_param,
        p_startup_cost=p_startup_cost,
        p_process_existing_count=p_process_existing_count,
        dtttdt=dtttdt,
    )


def _solve(data: FlexData) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


def _floor_per_period(sol) -> dict[str, float]:
    """Return the min v_flow per period — the binding floor."""
    vf = sol.value("v_flow").sort(["d", "t"])
    out: dict[str, float] = {}
    for d in _PERIODS:
        out[d] = float(vf.filter(pl.col("d") == d)["value"].min())
    return out


def _unit_reader(min_load_frame: pl.DataFrame) -> InMemoryReader:
    return InMemoryReader(
        entities={"unit": pl.DataFrame({"name": ["u"]})},
        parameters={("unit", "min_load"): min_load_frame},
    )


# ---------------------------------------------------------------------------
# Producer-level: the Map keeps per-(d, t) semantics (not summed).

def test_period_map_producer_not_summed():
    """A period Map resolves to per-period (p, d, t) values, not a sum."""
    src = _unit_reader(pl.DataFrame({
        "name":   ["u", "u"],
        "period": ["d1", "d2"],
        "value":  [0.4, 0.3],
    }))
    p = drv.p_min_load_pdt_from_source(src, _DT)
    assert isinstance(p, Param)
    assert p.dims == ("p", "d", "t")
    frame = p.frame.collect() if hasattr(p.frame, "collect") else p.frame
    by_period = {
        d: sorted(frame.filter(pl.col("d") == d)["value"].to_list())
        for d in _PERIODS
    }
    assert by_period["d1"] == pytest.approx([0.4, 0.4])
    assert by_period["d2"] == pytest.approx([0.3, 0.3])


def test_scalar_producer_broadcasts():
    src = _unit_reader(pl.DataFrame({"name": ["u"], "value": [0.4]}))
    p = drv.p_min_load_pdt_from_source(src, _DT)
    frame = p.frame.collect() if hasattr(p.frame, "collect") else p.frame
    assert frame["value"].to_list() == pytest.approx([0.4] * 4)


# ---------------------------------------------------------------------------
# End-to-end: the real minFlow_minload constraint applies the right floor.

def test_period_map_min_load_floor_per_period():
    """Period Map ``{d1: 0.4, d2: 0.3}`` → floor 0.4 in d1 and 0.3 in d2.

    Pre-fix the two Map rows summed to a 0.7 floor in *every* period.
    """
    src = _unit_reader(pl.DataFrame({
        "name":   ["u", "u"],
        "period": ["d1", "d2"],
        "value":  [0.4, 0.3],
    }))
    p_min_load = drv.p_min_load_pdt_from_source(src, _DT)
    _, sol = _solve(_build_uc_2period(p_min_load))
    assert sol.optimal
    floors = _floor_per_period(sol)
    assert floors["d1"] == pytest.approx(0.4, abs=1e-6)
    assert floors["d2"] == pytest.approx(0.3, abs=1e-6)
    # The bug would have given 0.7 in both.
    assert floors["d1"] != pytest.approx(0.7, abs=1e-6)
    assert floors["d2"] != pytest.approx(0.7, abs=1e-6)


def test_scalar_min_load_floor_unchanged():
    """Scalar min_load 0.4 → floor 0.4 in every period (byte-parity of
    behaviour with the old (p,) broadcast)."""
    src = _unit_reader(pl.DataFrame({"name": ["u"], "value": [0.4]}))
    p_min_load = drv.p_min_load_pdt_from_source(src, _DT)
    _, sol = _solve(_build_uc_2period(p_min_load))
    assert sol.optimal
    floors = _floor_per_period(sol)
    assert floors["d1"] == pytest.approx(0.4, abs=1e-6)
    assert floors["d2"] == pytest.approx(0.4, abs=1e-6)


def test_scalar_min_load_parity_with_direct_param():
    """A hand-built (p,) scalar Param (the old direct-param shape) and the
    new (p, d, t) producer give the identical LP floor — the scalar path is
    unchanged by the fix."""
    old_shape = Param(("p",), pl.DataFrame({"p": ["u"], "value": [0.4]}))
    _, sol_old = _solve(_build_uc_2period(old_shape))
    src = _unit_reader(pl.DataFrame({"name": ["u"], "value": [0.4]}))
    new_shape = drv.p_min_load_pdt_from_source(src, _DT)
    _, sol_new = _solve(_build_uc_2period(new_shape))
    assert sol_old.optimal and sol_new.optimal
    assert _floor_per_period(sol_old) == pytest.approx(
        _floor_per_period(sol_new))


def test_time_map_min_load_floor_per_timestep():
    """A time Map ``{t01: 0.4, t02: 0.3}`` → per-timestep floor, same in both
    periods (the time axis broadcasts across periods)."""
    # InMemoryReader passes the authored frame through verbatim; a time map
    # carries a ``t`` column (SpineDbReader renames ``time`` → ``t`` on read).
    src = _unit_reader(pl.DataFrame({
        "name":  ["u", "u"],
        "t":     ["t01", "t02"],
        "value": [0.4, 0.3],
    }))
    p_min_load = drv.p_min_load_pdt_from_source(src, _DT)
    assert p_min_load is not None
    _, sol = _solve(_build_uc_2period(p_min_load))
    assert sol.optimal
    vf = sol.value("v_flow").sort(["d", "t"])
    for d in _PERIODS:
        got = vf.filter(pl.col("d") == d).sort("t")["value"].to_list()
        assert got == pytest.approx([0.4, 0.3], abs=1e-6)
