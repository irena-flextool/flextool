"""A multi-line ``piecewise_linear`` unit in one region of a Benders split (design T9).

``lh2_three_region_invest`` (regions A / B / C, ``benders_regional``; the investment solve ``lh2_invest`` is
decomposed) with ``liquefier_A`` (h2_A -> lh2_A, region A; one input, one output) given the falling curve
{0.2: 0.90, 0.6: 0.87, 1.0: 0.80} (two lines plus the origin line; no online variables: the capacity form):

* the Benders run converges to the objective of the same solve run monolithically (rtol 1e-6);
* the region split carries the curve only into region A: its LP has the ``conversion_piecewise_coarse`` rows of
  liquefier_A (its nodes are in the 24-h ``new_stepduration`` group: a coarse process), the LPs of regions B and
  C have none (and no curve parameters).

Built from ``tests/fixtures/lh2_three_region.json`` (never a checked-in ``.sqlite``), with
``FLEXTOOL_AUTOSCALE_STRICT=1``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.engine._unit_db import b64, build, cstr_over, make_db, utf8

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
LH2_JSON = REPO / "tests" / "fixtures" / "lh2_three_region.json"
CHAIN = ["lh2_three_region", "lh2_three_region_invest"]
CURVE = b64({"index_type": "float", "data": [[0.2, 0.90], [0.6, 0.87], [1.0, 0.80]]}, "map")
VALUES = [["unit", "liquefier_A", "conversion_method", b64("piecewise_linear", "str")],
          ["unit", "liquefier_A", "piecewise_linear_efficiency", CURVE]]
INVEST = "lh2_invest"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    """``{"benders": steps, "monolithic": steps}``."""
    from flextool.engine import run_chain_from_db

    out = {}
    for label, extra in (("benders", []),
                         ("monolithic", [["solve", INVEST, "decomposition", b64("none", "str")]])):
        url, scen = make_db(tmp_path_factory, alt=f"pw_benders_{label}", values=VALUES + extra, base_chain=CHAIN,
                            fixture_json=LH2_JSON)
        out[label] = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"pw_benders_{label}_w"),
                                       keep_solutions=True)
    return out


def test_benders_equals_monolithic(runs) -> None:
    bend, mono = runs["benders"][INVEST], runs["monolithic"][INVEST]
    assert bend.is_benders is True and not getattr(mono, "is_benders", False)
    assert utf8(mono.flex_data.process_piecewise)["p"].to_list() == ["liquefier_A"]
    want = float(mono.solution.obj)
    assert float(bend.obj) == pytest.approx(want, rel=1e-6)


def test_only_region_A_has_the_curve(runs) -> None:
    from flextool.engine.solve._region_filter import split

    data = runs["monolithic"][INVEST].flex_data
    splits = {s.region: s for s in split(data, regions=["region_A", "region_B", "region_C"],
                                         benders_uncap_cross_region=True)}
    rows = {}
    for region, s in splits.items():
        pb = build(s.data)
        rows[region] = set()
        for fam in ("conversion_piecewise", "conversion_piecewise_coarse"):
            over = cstr_over(pb, fam)
            rows[region] |= set() if over is None else {(fam, p) for p in over["p"].to_list()}
        if region != "region_A":
            for f in ("process_piecewise", "process_piecewise_remap"):
                frame = getattr(s.data, f)
                assert frame is None or frame.height == 0, (region, f)
            for f in ("p_piecewise_slope", "p_piecewise_intercept"):
                prm = getattr(s.data, f)
                assert prm is None or prm.frame.height == 0, (region, f)
    # h2_A / lh2_A are in the 24-h new_stepduration group daily_group: liquefier_A is a coarse process, its rows
    # are one per line and coarse step (conversion_piecewise_coarse).
    assert rows == {"region_A": {("conversion_piecewise_coarse", "liquefier_A")}, "region_B": set(),
                    "region_C": set()}
