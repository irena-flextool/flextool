"""``reserve__upDown__group.penalty_reserve`` defaults to 5000.

The GMPL-era model defaulted the reserve shortfall penalty to 5000
(``reserveParam_defaults``) and the schema carried that default from v18;
v56 cleared it because the 4.x engine read explicit values only, so a
reserve group without an authored penalty had a FREE shortfall — its
requirement was effectively optional.  The engine now densifies the
penalty over every reserve group (authored value, else 5000) and v70
restores the schema default.

Fixtures: ``tests.json`` + override alternatives (built from JSON) adding
a timeseries reserve ``rx`` (up) on group ``electricity`` that the coal
plant cannot fully cover, so part of it is shortfall.
"""
from __future__ import annotations

from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Param

from tests.engine_polars._unit_db import PLANT, PLANT_CHAIN, b64, solve, utf8

GRP = ["rx", "up", "electricity"]


def _rx(penalty: float | None) -> tuple[list, list]:
    """A 1000 MW upward requirement offered only by the 500 MW coal plant;
    ``penalty`` None leaves ``penalty_reserve`` unauthored."""
    rel = "reserve__upDown__unit__node"
    unode = ["rx", "up", PLANT, "west"]
    entities = [["reserve", "rx", None],
                ["reserve__upDown__group", GRP, None],
                [rel, unode, None]]
    values = [
        ["reserve__upDown__group", GRP, "reserve_method",
         b64("timeseries_only", "str")],
        ["reserve__upDown__group", GRP, "reservation", b64(1000.0, "float")],
        [rel, unode, "is_enabled", b64("yes", "str")],
        [rel, unode, "max_share", b64(1.0, "float")],
        [rel, unode, "reliability", b64(1.0, "float")],
    ]
    if penalty is not None:
        values.append(["reserve__upDown__group", GRP, "penalty_reserve",
                       b64(penalty, "float")])
    return entities, values


def _solve(tpf, alt: str, penalty: float | None):
    ents, vals = _rx(penalty)
    return solve(tpf, alt=alt, base_chain=PLANT_CHAIN, entities=ents,
                 values=vals)


def _penalty(step) -> float:
    from flextool.engine_polars._reserve import reserve_penalty

    pen = utf8(reserve_penalty(step.flex_data).frame).filter(
        (pl.col("r") == "rx") & (pl.col("ud") == "up")
        & (pl.col("g") == "electricity"))
    assert pen.height == 1
    return float(pen["value"][0])


def _shortfall(step) -> pl.Series:
    return (utf8(step.solution.value("vq_reserve"))
            .filter(pl.col("r") == "rx")["value"])


def test_unauthored_penalty_defaults_to_5000(tmp_path_factory) -> None:
    """No authored ``penalty_reserve``: the group gets 5000 in the loaded
    Param and in the objective — the same optimum as authoring 5000, and
    dearer than a zero penalty (before, the two coincided: no term)."""
    from flextool.engine_polars._direct_params import PENALTY_RESERVE_DEFAULT

    assert PENALTY_RESERVE_DEFAULT == 5000.0
    unauth = _solve(tmp_path_factory, "rpd_none", None)
    loaded = utf8(unauth.flex_data.p_reserve_upDown_group_penalty_reserve
                  .frame).filter(pl.col("r") == "rx")
    assert loaded["value"].to_list() == [5000.0]
    assert _penalty(unauth) == 5000.0
    assert _shortfall(unauth).max() > 0.1      # the requirement is not met

    explicit = _solve(tmp_path_factory, "rpd_5000", 5000.0)
    assert unauth.solution.obj == pytest.approx(explicit.solution.obj,
                                                rel=1e-9)
    free = _solve(tmp_path_factory, "rpd_zero", 0.0)
    assert unauth.solution.obj > free.solution.obj + 1.0


def test_authored_penalty_wins(tmp_path_factory) -> None:
    """An authored value replaces the default."""
    low = _solve(tmp_path_factory, "rpd_low", 1.0)
    assert _penalty(low) == 1.0
    unauth = _solve(tmp_path_factory, "rpd_none2", None)
    assert low.solution.obj < unauth.solution.obj - 1.0


def test_reserve_penalty_densifies_partial_param() -> None:
    """``reserve_penalty`` left-joins the authored Param onto
    ``reserve_upDown_group``: authored rows keep their value, the others
    get the default, and a missing Param means the default everywhere."""
    from flextool.engine_polars._reserve import reserve_penalty

    rug = pl.DataFrame({"r": ["a", "b"], "ud": ["up", "down"],
                        "g": ["g1", "g1"]})
    authored = Param(("r", "ud", "g"), pl.DataFrame(
        {"r": ["a"], "ud": ["up"], "g": ["g1"], "value": [7.0]}))
    d = SimpleNamespace(reserve_upDown_group=rug,
                        p_reserve_upDown_group_penalty_reserve=authored)
    got = utf8(reserve_penalty(d).frame).sort("r")
    assert got["value"].to_list() == [7.0, 5000.0]
    d.p_reserve_upDown_group_penalty_reserve = None
    got = utf8(reserve_penalty(d).frame).sort("r")
    assert got["value"].to_list() == [5000.0, 5000.0]
    d.reserve_upDown_group = rug.clear()
    assert reserve_penalty(d) is None
