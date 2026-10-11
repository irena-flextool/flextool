"""The reserve shortfall ``vq_reserve`` (a fraction in [0, 1]) is scaled
by the group's largest possible requirement — in the reserve balances, in
the objective penalty and in the output reconstruction alike:

* timeseries groups: ``reservation`` (unchanged);
* dynamic groups: Σ ``p_flow_upper · unitsize · increase_reserve_ratio``
  over the flows that drive the requirement;
* n-1 groups: Σ ``p_flow_upper · unitsize · large_failure_ratio`` over the
  failing flows (the same sum the V1 n-1 RHS uses).

Before, the multiplier was the group's ``reservation``.  A reserve's
relationships are active once ANY group of the same (reserve, up/down)
authors a non-zero reservation (``prundt``, mirroring the .mod); a dynamic
/ n-1 group of such a reserve that authors none of its own then had a
zero (or, without any reservation table, a 1 MW) shortfall and no penalty —
a hard requirement that the downward-reserve floors could make
infeasible.  The tests activate each reserve with a small timeseries
requirement on the sibling group ``west_north`` (nodes west, north).

Fixtures: ``tests.json`` (``coal`` 500 MW, ``wind``, ``reserve_n_1``)
plus override alternatives, built from JSON.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine_polars._unit_db import (
    COAL, PLANT, PLANT_CHAIN, b64, flow, reserve, solve, uc, utf8,
)

N1_CHAIN = PLANT_CHAIN + ["reserve_n_1"]   # coal serves all of west
N1 = ["n_1", "up", "electricity"]


def _scale(step) -> pl.DataFrame:
    from flextool.engine_polars._reserve import reserve_shortfall_scale

    sc = reserve_shortfall_scale(step.flex_data)
    assert sc is not None
    return utf8(sc.frame)


def _vq(step, r: str, g: str = "electricity") -> pl.Series:
    return (utf8(step.solution.value("vq_reserve"))
            .filter((pl.col("r") == r) & (pl.col("g") == g))
            .sort("d", "t")["value"])


def test_timeseries_scale_is_the_reservation(tmp_path_factory) -> None:
    """A timeseries-only model keeps ``vq_reserve × reservation`` (the very
    same Param, so the LP is byte-identical)."""
    step = solve(tmp_path_factory, alt="rs_ts",
                 base_chain=PLANT_CHAIN + ["wind", "reserve"], values=[])
    from flextool.engine_polars._reserve import reserve_shortfall_scale

    fd = step.flex_data
    assert reserve_shortfall_scale(fd) is fd.pdtReserve_upDown_group_reservation


def _activator(r: str, ud: str) -> tuple[list, list]:
    """A 1 MW timeseries requirement of reserve ``r`` on ``west_north`` —
    it activates the reserve's relationships."""
    grp = [r, ud, "west_north"]
    return ([["reserve__upDown__group", grp, None]],
            [["reserve__upDown__group", grp, "reserve_method",
              b64("timeseries_only", "str")],
             ["reserve__upDown__group", grp, "reservation", b64(1.0, "float")],
             ["reserve__upDown__group", grp, "penalty_reserve",
              b64(20000.0, "float")]])


def _n1(tpf, alt: str, penalty: float):
    ents, vals = _activator("n_1", "up")
    return solve(tpf, alt=alt, base_chain=N1_CHAIN, entities=ents,
                 values=vals + [
        # wind cannot cover the n-1 requirement: only coal's 1 % share
        ["reserve__upDown__unit__node", ["n_1", "up", "wind_plant", "west"],
         "max_share", b64(0.0, "float")],
        ["reserve__upDown__group", N1, "penalty_reserve",
         b64(penalty, "float")]])


def test_n_1_shortfall_covers_requirement_and_is_penalised(
        tmp_path_factory) -> None:
    """n-1 group without its own reservation: scale = 500 MW · 0.1 = 50 MW
    (it was 0: a hard requirement), the shortfall covers whatever coal's
    own 5 MW cannot, and it is priced — doubling the penalty raises the
    objective while the shortfall stays.  (The penalties are low enough
    that the LP does not curb coal output to avoid the requirement.)"""
    s1 = _n1(tmp_path_factory, "rs_n1_a", 100.0)
    s2 = _n1(tmp_path_factory, "rs_n1_b", 200.0)
    sc = _scale(s1).filter((pl.col("r") == "n_1")
                           & (pl.col("g") == "electricity"))
    assert sc["value"].to_list() == pytest.approx([50.0] * sc.height)
    assert sc.height == s1.flex_data.dt.height
    requirement = 0.1 * flow(s1, PLANT, COAL, "west")
    provided = reserve(s1, PLANT, "n_1", "west")
    short = _vq(s1, "n_1", "electricity") * 50.0
    assert ((provided + short - requirement) >= -1e-6).all()
    assert short.max() > 10.0
    assert (_vq(s1, "n_1", "electricity") <= 1.0 + 1e-9).all()
    assert _vq(s2, "n_1", "electricity").to_list() == pytest.approx(
        _vq(s1, "n_1", "electricity").to_list(), abs=1e-9)
    assert s2.solution.obj > s1.solution.obj + 1.0


def test_dynamic_shortfall_covers_requirement(tmp_path_factory) -> None:
    """Dynamic group (``increase_reserve_ratio`` 0.5 on wind): scale =
    wind's structural capacity · 0.5, and the shortfall covers the part of
    0.5 · wind output the coal plant cannot."""
    grp = ["dyn", "up", "electricity"]
    ents = [["reserve", "dyn", None],
            ["reserve__upDown__group", grp, None],
            ["reserve__upDown__unit__node", ["dyn", "up", "wind_plant",
                                              "west"], None],
            ["reserve__upDown__unit__node", ["dyn", "up", PLANT, "west"],
             None]]
    vals = [
        ["reserve__upDown__group", grp, "reserve_method",
         b64("dynamic_only", "str")],
        # a low penalty: wind is not curtailed to avoid the requirement
        ["reserve__upDown__group", grp, "penalty_reserve",
         b64(1.0, "float")],
        ["reserve__upDown__unit__node", ["dyn", "up", "wind_plant", "west"],
         "increase_reserve_ratio", b64(0.5, "float")],
        ["reserve__upDown__unit__node", ["dyn", "up", "wind_plant", "west"],
         "is_enabled", b64("yes", "str")],
        ["reserve__upDown__unit__node", ["dyn", "up", PLANT, "west"],
         "is_enabled", b64("yes", "str")],
        ["reserve__upDown__unit__node", ["dyn", "up", PLANT, "west"],
         "max_share", b64(0.01, "float")],
        ["reserve__upDown__unit__node", ["dyn", "up", PLANT, "west"],
         "reliability", b64(1.0, "float")],
    ]
    a_ents, a_vals = _activator("dyn", "up")
    step = solve(tmp_path_factory, alt="rs_dyn",
                 base_chain=PLANT_CHAIN + ["wind"], entities=ents + a_ents,
                 values=vals + a_vals)
    fd = step.flex_data
    wind_cap = (utf8(fd.p_flow_upper.frame)
                .filter(pl.col("p") == "wind_plant")["value"].max()
                * float(utf8(fd.p_unitsize.frame)
                        .filter(pl.col("p") == "wind_plant")["value"][0]))
    sc = _scale(step).filter((pl.col("r") == "dyn")
                             & (pl.col("g") == "electricity"))
    assert sc["value"].max() == pytest.approx(0.5 * wind_cap, rel=1e-9)
    wind = (utf8(step.solution.value("v_flow"))
            .filter(pl.col("p") == "wind_plant").sort("d", "t")["value"]
            * float(utf8(fd.p_unitsize.frame)
                    .filter(pl.col("p") == "wind_plant")["value"][0]))
    requirement = 0.5 * wind
    provided = reserve(step, PLANT, "dyn", "west")
    short = _vq(step, "dyn") * sc.sort("d", "t")["value"]
    assert ((provided + short - requirement) >= -1e-6).all()
    assert short.max() > 10.0


def test_n_1_down_requirement_with_min_load_provider(
        tmp_path_factory) -> None:
    """n-1 DOWN group (the consumer ``dr_increase_demand`` failing) served
    by an online coal plant that may not go below min load: the LP solves,
    and the shortfall (scale = 50 MW · 1.0) covers what the coal plant's
    headroom above min load cannot."""
    grp = ["n1d", "down", "electricity"]
    ents = [["reserve", "n1d", None],
            ["reserve__upDown__group", grp, None],
            ["reserve__upDown__unit__node",
             ["n1d", "down", "dr_increase_demand", "west"], None],
            ["reserve__upDown__unit__node", ["n1d", "down", PLANT, "west"],
             None]]
    rel = "reserve__upDown__unit__node"
    vals = uc(PLANT, 0.9) + [
        ["reserve__upDown__group", grp, "reserve_method",
         b64("large_failure_only", "str")],
        ["reserve__upDown__group", grp, "penalty_reserve",
         b64(20000.0, "float")],
        [rel, ["n1d", "down", "dr_increase_demand", "west"],
         "large_failure_ratio", b64(1.0, "float")],
        [rel, ["n1d", "down", "dr_increase_demand", "west"], "is_enabled",
         b64("yes", "str")],
        [rel, ["n1d", "down", PLANT, "west"], "is_enabled",
         b64("yes", "str")],
        [rel, ["n1d", "down", PLANT, "west"], "max_share", b64(1.0, "float")],
        [rel, ["n1d", "down", PLANT, "west"], "reliability",
         b64(1.0, "float")],
        # the consumer must consume: a fixed 40 MW demand on its input
        ["unit", "dr_increase_demand", "min_load", b64(0.8, "float")],
        ["unit", "dr_increase_demand", "startup_method",
         b64("linear", "str")],
    ]
    a_ents, a_vals = _activator("n1d", "down")
    step = solve(tmp_path_factory, alt="rs_n1down",
                 base_chain=PLANT_CHAIN + ["dr_increase_demand"],
                 entities=ents + a_ents, values=vals + a_vals)
    sc = _scale(step).filter((pl.col("r") == "n1d")
                             & (pl.col("g") == "electricity"))
    assert sc["value"].to_list() == pytest.approx([50.0] * sc.height)
    out = flow(step, PLANT, COAL, "west")
    down = reserve(step, PLANT, "n1d", "west")
    from tests.engine_polars._unit_db import online

    assert ((out - down - 0.9 * online(step, PLANT)) >= -1e-5).all()
    cons = flow(step, "dr_increase_demand", "west", "dr_increase_demand")
    short = _vq(step, "n1d") * 50.0
    assert ((down + short - cons) >= -1e-6).all()


def test_outputs_use_the_same_multiplier(tmp_path_factory) -> None:
    """``read_parameters`` exposes the multiplier the reserve shortfall in
    MW and its penalty cost are rebuilt with."""
    from flextool.process_outputs.read_parameters import read_parameters

    step = _n1(tmp_path_factory, "rs_out", 100.0)
    par = read_parameters(step.flex_data, step.solution)
    sc = par.reserve_upDown_group_shortfall_scale
    col = [c for c in sc.columns if c[0] == "n_1" and c[2] == "electricity"]
    assert col
    assert sc[col[0]].to_list() == pytest.approx([50.0] * len(sc))
