"""A float-residue cost is zeroed before autoscale centres the objective (Q-gate-A).

Regression for #140.  Before 7dc33ecd a flat ``min_load_efficiency`` curve (``min_load`` 0.3, no
``efficiency_at_min_load``, efficiency 0.9) got its section as the difference ``1/eta − slope``:

    slope   = (1/eta − min_load/eta) / (1 − min_load) = 1.1111111111111114
    section = 1/eta − slope                           = −2.2e-16     (exactly 0 algebraically)

The residue reached the objective as a ~1e-17 cost next to O(1) costs (a 16-decade cost range); autoscale
Layer 3 centred the objective with ``user_objective_scale`` 2**25 and HiGHS' dual simplex stopped with
"excessive dual values" on the rolling solves.  7dc33ecd fixed the source; independently of it, FlexTool now
sets polar-high's relative objective cutoff (``Problem.obj_coef_zero_rel_threshold`` =
``core.OBJ_COEF_ZERO_REL_THRESHOLD``), so such a residue becomes an exact 0 before any scaling.

The test re-injects the old residue into ``p_section`` (wrapping ``build_flextool``, as
``tools/lp_snapshot.py`` does) on the rolling vehicle of ``tests/model/test_online_without_min_load.py``
and shows: the residue reaches the objective as a sub-cutoff cost; with the cutoff it is zeroed, Layer 3 picks
the same objective scale as the clean model and every roll solves to the clean model's objective.  (With
``OBJ_COEF_ZERO_REL_THRESHOLD`` patched to 0 the same run stops on the first roll: "Dual simplex ratio test
failed due to excessive dual values", ``FlexToolSolveError`` — checked when the test was written.)
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.model.test_online_without_min_load import EFF, UNIT, _fixture, _mle, _online
from tests.engine._unit_db import make_db

VEH = "roll"
ETA = EFF[VEH]
MIN_LOAD = 0.3


def _old_section(eta: float, min_load: float) -> float:
    """The pre-7dc33ecd section of a flat curve (eta_min = eta): ``1/eta − slope``."""
    cr = 1.0 / eta
    slope = (cr - min_load / eta) / (1.0 - min_load)
    return cr - slope


RESIDUE = _old_section(ETA, MIN_LOAD)


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _wrap_build(monkeypatch, built: list, *, inject: bool) -> None:
    """Wrap ``build_flextool``: record each built Problem; ``inject``: the unit's exact-0 sections become the
    old residue first."""
    import flextool.engine.model.core as core

    original = getattr(core.build_flextool, "__wrapped_original__", core.build_flextool)

    def build_with_residue(m, d, *args, **kwargs):
        ps = d.p_section
        if inject and ps is not None:
            frame = ps.frame.with_columns(
                value=pl.when((pl.col("p").cast(pl.Utf8) == UNIT[VEH]) & (pl.col("value") == 0.0))
                .then(pl.lit(RESIDUE)).otherwise(pl.col("value")))
            d.p_section = type(ps)(ps.dims, frame, name=getattr(ps, "name", None))
        out = original(m, d, *args, **kwargs)
        built.append(m)
        return out

    build_with_residue.__wrapped_original__ = original
    monkeypatch.setattr(core, "build_flextool", build_with_residue)


def _run(tmp_path_factory, alt: str):
    from flextool.engine import run_chain_from_db

    values = _online(VEH) + _mle(UNIT[VEH], min_load=MIN_LOAD)
    fj, chain = _fixture(tmp_path_factory, VEH, strip_efficiency=None)
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=fj)
    steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)
    assert steps
    for name, step in steps.items():
        assert step.solution is not None, f"{name}: solve produced no solution"
    return steps


def _cost_range_and_scale(pb, rel: float) -> tuple[tuple[float, float], int]:
    from polar_high.autoscale import ScalingConfig, detect_ranges, recommend_scaling

    saved = pb.obj_coef_zero_rel_threshold
    pb.obj_coef_zero_rel_threshold = rel
    try:
        cfg = ScalingConfig()
        r = detect_ranges(pb, cfg)
        return r.cost, recommend_scaling(r, cfg).user_objective_scale
    finally:
        pb.obj_coef_zero_rel_threshold = saved


def test_residue_cost_is_zeroed_and_rolls_solve(tmp_path_factory, monkeypatch) -> None:
    import flextool.engine.model.core as core

    assert RESIDUE != 0.0 and abs(RESIDUE) < 1e-15  # the old float residue (−2.2e-16)
    built_clean: list = []
    _wrap_build(monkeypatch, built_clean, inject=False)
    clean = _run(tmp_path_factory, "qgA_clean")

    built: list = []
    _wrap_build(monkeypatch, built, inject=True)
    steps = _run(tmp_path_factory, "qgA_residue")
    assert built and len(built) == len(built_clean), "build_flextool was not called through the wrapper"

    # Every solve of the chain solves to the clean model's objective.
    assert list(steps) == list(clean)
    for name in steps:
        assert float(steps[name].solution.obj) == pytest.approx(float(clean[name].solution.obj), rel=1e-9), name

    pb = built[0]
    assert pb.obj_coef_zero_rel_threshold == core.OBJ_COEF_ZERO_REL_THRESHOLD == 1e-14
    # Without the cutoff the residue is in the objective: a sub-cutoff cost that drives the centring.
    (lo_off, hi_off), n_off = _cost_range_and_scale(pb, 0.0)
    assert lo_off < 1e-14 * hi_off
    assert n_off >= 20
    # With it the cost range is the clean model's and so is Layer 3's objective scale.
    (lo_on, hi_on), n_on = _cost_range_and_scale(pb, core.OBJ_COEF_ZERO_REL_THRESHOLD)
    assert lo_on >= 1e-14 * hi_on
    pb_clean = built_clean[0]
    assert _cost_range_and_scale(pb_clean, core.OBJ_COEF_ZERO_REL_THRESHOLD) == ((lo_on, hi_on), n_on)

    # The coefficient itself is zeroed in the LP handed to the solver: every v_online_linear column of the unit
    # carries the same cost as in the clean model (0 where the clean model has none).
    def online_costs(problem) -> list[float]:
        col_obj = problem.canonicalise().col_obj
        v = problem._vars["v_online_linear"]
        cols = v.frame.filter(pl.col("p").cast(pl.Utf8) == UNIT[VEH])["col_id"].to_list()
        assert cols
        return [float(col_obj[c]) for c in cols]

    assert online_costs(pb) == online_costs(pb_clean)
    pb.obj_coef_zero_rel_threshold = 0.0
    pb._matrix = None  # rebuild the canonical matrix without the cutoff
    raw = online_costs(pb)
    assert any(c != 0.0 and abs(c) < 1e-14 * hi_off for c in raw), raw
