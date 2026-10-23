"""The n-1 (large_failure) reserve requirement is one row per failing process.

``reserveBalance_{up,down}_n_1_eq[r, ud, g, p_fail, d, t]``: the reserve held by every provider of the group
except the failing process ``p_fail`` (× unitsize × reliability), plus the shared shortfall, covers that
process's flow into (up) / out of (down) the group's nodes × unitsize × ``large_failure_ratio``.  So the
largest single failure sets the requirement, and a failing process cannot cover its own failure.  The
shortfall multiplier of the n-1 component is the largest failing process's bound (a max over failing
processes, each summed over its own flows).

Before, one row per ``(r, ud, g, d, t)`` summed the failure of all large-failure processes and counted the
reserve of every provider, the failing one included.

Fixtures: ``tests.json`` (``coal`` 500 MW, ``wind`` 1000 MW, ``reserve_n_1``: coal ``large_failure_ratio``
0.1) plus override alternatives, built from JSON.  Each reserve is activated by a small timeseries
requirement on the sibling group ``west_north``, as in ``test_reserve_shortfall_scale.py``.
"""
from __future__ import annotations

import collections
import types

import polars as pl
import pytest
from polar_high import Param

from tests.engine._unit_db import (
    COAL, PLANT, PLANT_CHAIN, b64, build, flow, reserve, solve, uc, unitsize, utf8,
)

N1 = ["n_1", "up", "electricity"]
REL = "reserve__upDown__unit__node"
WIND = "wind_plant"


def _activator(r: str, ud: str) -> tuple[list, list]:
    grp = [r, ud, "west_north"]
    return ([["reserve__upDown__group", grp, None]],
            [["reserve__upDown__group", grp, "reserve_method", b64("timeseries_only", "str")],
             ["reserve__upDown__group", grp, "reservation", b64(1.0, "float")],
             ["reserve__upDown__group", grp, "penalty_reserve", b64(20000.0, "float")]])


def _row_terms(pb, family: str) -> dict[str, dict[str, float]]:
    """``{row: {column: coefficient}}`` of the canonical rows of ``family``."""
    m = pb.canonicalise()
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    rows = {i: nm for i, nm in enumerate(m.row_names) if nm.startswith(family + "[")}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            ri = m.row_idx[k]
            if ri in rows:
                out[rows[ri]][m.col_names[c]] = m.val[k]
    return out


def _cols(terms: dict[str, float], prefix: str) -> dict[str, float]:
    return {c: v for c, v in terms.items() if c.startswith(prefix)}


def _vq(step, r: str, g: str = "electricity") -> pl.Series:
    return (utf8(step.solution.value("vq_reserve"))
            .filter((pl.col("r") == r) & (pl.col("g") == g))
            .sort("d", "t")["value"])


def _scale(step, r: str, g: str = "electricity") -> pl.Series:
    from flextool.engine.model._reserve import reserve_shortfall_scale

    sc = reserve_shortfall_scale(step.flex_data)
    assert sc is not None
    return utf8(sc.frame).filter((pl.col("r") == r) & (pl.col("g") == g)).sort("d", "t")["value"]


def _two_failures(tpf, alt: str):
    """Coal (500 MW) and wind (1000 MW) both fail with ratio 1 at west; both provide n_1 reserve (coal
    max_share 1, reliability 1; wind max_share 1, reliability 0.6)."""
    ents, vals = _activator("n_1", "up")
    return solve(tpf, alt=alt, base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"], entities=ents,
                 values=vals + [
                     [REL, ["n_1", "up", PLANT, "west"], "large_failure_ratio", b64(1.0, "float")],
                     [REL, ["n_1", "up", PLANT, "west"], "max_share", b64(1.0, "float")],
                     [REL, ["n_1", "up", WIND, "west"], "large_failure_ratio", b64(1.0, "float")],
                     ["reserve__upDown__group", N1, "penalty_reserve", b64(100.0, "float")]])


@pytest.fixture(scope="module")
def two_failures(tmp_path_factory):
    return _two_failures(tmp_path_factory, "n1pf_two")


def test_one_row_per_failing_process(two_failures) -> None:
    """Per (d, t) one row per failing process; each carries only its own flow and the other process's
    reserve."""
    step = two_failures
    rows = _row_terms(build(step.flex_data), "reserveBalance_up_n_1_eq")
    n_dt = step.flex_data.dt.height
    by_fail = collections.Counter(r.split(",")[3] for r in rows)
    assert by_fail == {PLANT: n_dt, WIND: n_dt}
    coefs = {PLANT: -unitsize(step, PLANT), WIND: -unitsize(step, WIND)}
    for row, terms in rows.items():
        assert row.startswith("reserveBalance_up_n_1_eq[n_1,up,electricity,"), row
        fail = row.split(",")[3]
        other = WIND if fail == PLANT else PLANT
        flows = _cols(terms, "v_flow[")
        assert list(flows.values()) == [pytest.approx(coefs[fail])], row
        assert all(c.startswith(f"v_flow[{fail},") for c in flows), row
        assert not _cols(terms, f"v_reserve[{fail},"), row
        assert len(_cols(terms, f"v_reserve[{other},n_1,up,west,")) == 1, row
        assert len(_cols(terms, "vq_reserve[n_1,up,electricity,")) == 1, row


def test_requirement_is_the_largest_failure(two_failures) -> None:
    """Every single failure is covered by the other process's reserve plus the shortfall, and the
    shortfall multiplier is the largest failure (wind's 1000 MW), not the sum (1500 MW)."""
    step = two_failures
    sc = _scale(step, "n_1")
    assert sc.to_list() == pytest.approx([1000.0] * step.flex_data.dt.height)
    short = _vq(step, "n_1") * sc
    coal_out = flow(step, PLANT, COAL, "west")
    wind_out = flow(step, WIND, WIND, "west")
    coal_res = reserve(step, PLANT, "n_1", "west")
    wind_res = reserve(step, WIND, "n_1", "west") * 0.6
    assert ((wind_res + short - coal_out) >= -1e-6).all()
    assert ((coal_res + short - wind_out) >= -1e-6).all()
    # The largest failure binds: the shortfall is no larger than the larger single deficit.
    deficit = pl.DataFrame({"coal": (coal_out - wind_res).clip(lower_bound=0.0),
                            "wind": (wind_out - coal_res).clip(lower_bound=0.0)}).max_horizontal()
    assert ((short - deficit) <= 1e-6).all()


def test_failing_unit_cannot_cover_itself(tmp_path_factory) -> None:
    """Coal is the only failing process (ratio 0.1) and the only usable provider (max_share 0.01, wind
    max_share 0): its own reserve does not count, so the whole requirement is shortfall.  Before, coal's
    own 5 MW covered part of its own failure."""
    ents, vals = _activator("n_1", "up")
    step = solve(tmp_path_factory, alt="n1pf_self", base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"],
                 entities=ents, values=vals + [
                     [REL, ["n_1", "up", WIND, "west"], "max_share", b64(0.0, "float")],
                     ["reserve__upDown__group", N1, "penalty_reserve", b64(100.0, "float")]])
    requirement = 0.1 * flow(step, PLANT, COAL, "west")
    short = _vq(step, "n_1") * 50.0
    assert requirement.max() > 10.0
    assert ((short - requirement) >= -1e-6).all()


def test_down_n_1_excludes_own_reserve(tmp_path_factory) -> None:
    """n-1 DOWN: the failing consumer ``dr_increase_demand`` also offers downward reserve; its row has
    the consumer's flow and coal's reserve, not the consumer's own reserve."""
    grp = ["n1d", "down", "electricity"]
    ents = [["reserve", "n1d", None], ["reserve__upDown__group", grp, None],
            [REL, ["n1d", "down", "dr_increase_demand", "west"], None],
            [REL, ["n1d", "down", PLANT, "west"], None]]
    vals = uc(PLANT, 0.5) + [
        ["reserve__upDown__group", grp, "reserve_method", b64("large_failure_only", "str")],
        ["reserve__upDown__group", grp, "penalty_reserve", b64(20000.0, "float")],
        [REL, ["n1d", "down", "dr_increase_demand", "west"], "large_failure_ratio", b64(1.0, "float")],
        [REL, ["n1d", "down", "dr_increase_demand", "west"], "is_enabled", b64("yes", "str")],
        [REL, ["n1d", "down", "dr_increase_demand", "west"], "max_share", b64(1.0, "float")],
        [REL, ["n1d", "down", "dr_increase_demand", "west"], "reliability", b64(1.0, "float")],
        [REL, ["n1d", "down", PLANT, "west"], "is_enabled", b64("yes", "str")],
        [REL, ["n1d", "down", PLANT, "west"], "max_share", b64(1.0, "float")],
        [REL, ["n1d", "down", PLANT, "west"], "reliability", b64(1.0, "float")],
        ["unit", "dr_increase_demand", "min_load", b64(0.8, "float")],
        ["unit", "dr_increase_demand", "startup_method", b64("linear", "str")],
    ]
    a_ents, a_vals = _activator("n1d", "down")
    step = solve(tmp_path_factory, alt="n1pf_down", base_chain=PLANT_CHAIN + ["dr_increase_demand"],
                 entities=ents + a_ents, values=vals + a_vals)
    rows = _row_terms(build(step.flex_data), "reserveBalance_down_n_1_eq")
    assert len(rows) == step.flex_data.dt.height
    for row, terms in rows.items():
        assert row.startswith("reserveBalance_down_n_1_eq[n1d,down,electricity,dr_increase_demand,"), row
        assert len(_cols(terms, "v_flow[dr_increase_demand,west,")) == 1, row
        assert not _cols(terms, "v_reserve[dr_increase_demand,"), row
        assert len(_cols(terms, f"v_reserve[{PLANT},n1d,down,west,")) == 1, row


def test_flow_requirement_bound_sums_each_process_before_the_max() -> None:
    """``_flow_requirement_bound(per_process_max=True)`` sums a failing process's own arcs into the group
    first and then takes the max over processes: ``a`` has two arcs (into n1 and n2, 100 · 0.5 each =
    100 MW), ``b`` one arc of 80 MW → 100 (a max over arcs would give 80, a sum over processes 180)."""
    from flextool.engine.model._reserve import _flow_requirement_bound

    dt = pl.DataFrame({"d": ["d1"], "t": ["t01"]})
    pss = pl.DataFrame({"p": ["a", "a", "b"], "source": ["a", "a", "b"], "sink": ["n1", "n2", "n1"]})
    d = types.SimpleNamespace(
        dt=dt,
        group_node=pl.DataFrame({"g": ["g", "g"], "n": ["n1", "n2"]}),
        process_source_sink=pss,
        p_flow_upper=Param(("p", "source", "sink", "d", "t"),
                           pss.join(dt, how="cross").with_columns(value=pl.lit(1.0))),
        p_unitsize=Param(("p",), pl.DataFrame({"p": ["a", "b"], "value": [100.0, 80.0]})),
    )
    ratio_set = pl.DataFrame({"p": ["a", "a", "b"], "r": ["r"] * 3, "ud": ["up"] * 3, "n": ["n1", "n2", "n1"]})
    ratio = Param(("p", "r", "ud", "n"), ratio_set.with_columns(value=pl.Series([0.5, 0.5, 1.0])))
    rug = pl.DataFrame({"r": ["r"], "ud": ["up"], "g": ["g"]})
    per_max = _flow_requirement_bound(d, ratio_set, ratio, rug, "sink", per_process_max=True).collect()
    assert per_max["value"].to_list() == pytest.approx([100.0])
    summed = _flow_requirement_bound(d, ratio_set, ratio, rug, "sink").collect()
    assert summed["value"].to_list() == pytest.approx([180.0])
