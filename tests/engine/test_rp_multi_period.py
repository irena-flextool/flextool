"""Multi-period representative periods: every period gets its own rep-period data and base chain (#108).

Each period of a solve whose timeset has ``representative_period_weights`` gets its own weights, representative
blocks and base-period chain, also when several periods share a timeset or a timeline (the same base-period
names).  The across-solve blended variants chain the periods in solve order.  Databases are built from
``tests/fixtures/lh2_three_region_rp_invest.json`` plus an override alternative (``tools/lp_snapshot.py``
overlays), never from a checked-in ``.sqlite``.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _rp_hourly_ref as H
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build

REPO = Path(__file__).resolve().parents[2]
RP_JSON = REPO / "tests" / "fixtures" / "lh2_three_region_rp_invest.json"
STOCH_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
SFX = "__lh2_three_region_rp_invest"
SOLVE = "lh2_rp_invest" + SFX
LH2 = ("lh2_A", "lh2_B", "lh2_C")
L = H.L


def _run(tmp_path_factory, patch, json_path: Path = RP_JSON, scenario: str = "custom") -> dict:
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = patch(json.loads(json_path.read_text()))
    work = tmp_path_factory.mktemp("rp_multi_period")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "w", keep_solutions=True)
    assert steps
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal
    return steps


def _methods(method: str | dict):
    methods = method if isinstance(method, dict) else {n: method for n in LH2}
    return [["node", n, "storage_binding_method", m] for n, m in methods.items()]


def _overlay(values, entities=()):
    def patch(data):
        return L._overlay(data, name="custom", base="lh2_three_region_rp_invest", values=values, entities=entities)
    return patch


def _multi_base(methods: dict):
    def patch(data):
        return L._rp_multi_base(data, "custom", methods)
    return patch


def _utf8(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns if c != "value"])


def _weights(fd, period: str) -> list[float]:
    """The distinct timestep weights of ``period``, in step order of first appearance."""
    tw = _utf8(fd.p_timestep_weight.frame).filter(pl.col("d") == period).sort("t")
    return [round(v, 9) for v in tw["value"].unique(maintain_order=True).to_list()]


def _total_cost(step) -> float:
    return float(step.solution.obj)


# ---------------------------------------------------------------------------
# Weights and blocks of every period
# ---------------------------------------------------------------------------


def test_every_period_gets_rp_weights_without_blended_nodes(tmp_path_factory):
    """Plain bind_within_solve nodes: y2040's representative steps keep their weights 1.1 / 0.9 (before, every
    period but the first fell back to 1.0)."""
    steps = _run(tmp_path_factory, _overlay(_methods("bind_within_solve")))
    fd = steps[SOLVE].flex_data
    assert _weights(fd, "y2030") == [1.4, 0.6]
    assert _weights(fd, "y2040") == [1.1, 0.9]


def test_within_solve_second_period_has_rp_blocks(tmp_path_factory):
    steps = _run(tmp_path_factory, _overlay(_methods("bind_within_solve_blended_weights")))
    fd = steps[SOLVE].flex_data
    assert sorted(_utf8(fd.rp_block_first).rows()) == [
        ("y2030", "t0001"), ("y2030", "t0025"), ("y2040", "t0049"), ("y2040", "t0073")]
    assert _weights(fd, "y2040") == [1.1, 0.9]
    # The lh2 nodes are in the fixture's 24 h group: the balance of the step is its coarse-step row.
    bal = row_terms(build(fd), "nodeBalanceCoarse_eq", "lh2_A,y2040,t0049")
    cols = bal["nodeBalanceCoarse_eq[lh2_A,y2040,t0049]"]
    assert "v_state_rp_start[lh2_A,y2040,t0049]" in cols
    assert "v_state[lh2_A,y2040,t0048]" not in cols


def test_shared_timeset_each_period(tmp_path_factory):
    """Both periods on one rp timeset: both get the weights 1.4 / 0.6 (before: y2030 doubled, y2040 none)."""
    patch = _overlay([["solve", SOLVE, "period_timeset",
                       {"type": "map", "index_type": "str", "index_name": "period", "rank": 1,
                        "data": [["y2030", "rp_y2030" + SFX], ["y2040", "rp_y2030" + SFX]]}]])
    fd = _run(tmp_path_factory, patch)[SOLVE].flex_data
    assert _weights(fd, "y2030") == [1.4, 0.6]
    assert _weights(fd, "y2040") == [1.4, 0.6]
    assert sorted(_utf8(fd.rp_block_first).rows()) == [
        ("y2030", "t0001"), ("y2030", "t0025"), ("y2040", "t0001"), ("y2040", "t0025")]


# ---------------------------------------------------------------------------
# Base-period chains (rp_multi_base: seven shared base names in both periods)
# ---------------------------------------------------------------------------


def _inner(name: str) -> list[str]:
    return name[name.index("[") + 1:-1].split(",")


def test_shared_base_names_one_chain_per_period(tmp_path_factory):
    steps = _run(tmp_path_factory, _multi_base({n: "bind_within_period_blended_weights" for n in LH2}))
    pb = build(steps[SOLVE].flex_data)
    names = pb.canonicalise().row_names
    assert len(names) == len(set(names)), "duplicate row names"
    bal = row_terms(pb, "rp_inter_period_balance")
    cyc = row_terms(pb, "rp_inter_period_cyclic")
    counts = collections.Counter((_inner(r)[0], _inner(r)[1]) for r in bal)
    assert counts == {(n, d): 6 for n in LH2 for d in ("y2030", "y2040")}
    assert collections.Counter((_inner(r)[0], _inner(r)[1]) for r in cyc) == {
        (n, d): 1 for n in LH2 for d in ("y2030", "y2040")}
    for row, cols in {**bal, **cyc}.items():
        n, d, _b, d_prev, _bp = _inner(row)
        assert d_prev == d
        for col in cols:
            if col.startswith(("v_state[", "v_state_rp_start[")):
                assert _inner(col)[1] == d, (row, col)


def test_across_solve_chain_links_periods(tmp_path_factory):
    methods = {"lh2_A": "bind_within_solve_blended_weights", "lh2_B": "bind_forward_only_blended_weights",
               "lh2_C": "bind_within_solve_blended_weights"}
    pb = build(_run(tmp_path_factory, _multi_base(methods))[SOLVE].flex_data)
    bal = row_terms(pb, "rp_inter_period_balance")
    cyc = row_terms(pb, "rp_inter_period_cyclic")
    for n in LH2:
        row = f"rp_inter_period_balance[{n},y2040,t0001,y2030,t0145]"
        assert row in bal
        assert bal[row][f"v_state_inter[{n},y2040,t0001]"] == pytest.approx(1.0)
        assert bal[row][f"v_state_inter[{n},y2030,t0145]"] == pytest.approx(-1.0)
    assert set(cyc) == {f"rp_inter_period_cyclic[{n},y2030,t0001,y2040,t0145]" for n in ("lh2_A", "lh2_C")}


def test_variant_order_on_two_periods(tmp_path_factory):
    """forward_only ≤ within_solve ≤ within_period (more freedom, lower cost), and the within_period cost."""
    cost = {}
    for key in ("fo", "ws", "wp"):
        steps = _run(tmp_path_factory, _multi_base({n: H.BLENDED[key] for n in LH2}))
        cost[key] = _total_cost(steps[SOLVE])
    assert cost["fo"] <= cost["ws"] * (1 + 1e-9)
    assert cost["ws"] <= cost["wp"] * (1 + 1e-9)
    # 16 367 662 856 before c15e1: FA-1 (exact self-discharge of the fixture's 1e-4/h batteries) +2.9e-6 →
    # 16 367 709 994; FA-3 (the bind_within_solve batteries' fix_start pin moved from the end of y2030, mid-horizon,
    # to the end of y2040, the state the first step starts from) +2.8 %: the batteries now end y2040 at the pin and
    # refill there, competing with y2040's H2 shortfall (the same mechanism as the RP Benders M_rp pin).
    assert cost["wp"] == pytest.approx(16_820_838_994.0, rel=1e-6)


def test_mixed_variants_in_one_solve(tmp_path_factory):
    methods = {"lh2_A": "bind_within_period_blended_weights", "lh2_B": "bind_within_period_blended_weights",
               "lh2_C": "bind_within_solve_blended_weights"}
    pb = build(_run(tmp_path_factory, _multi_base(methods))[SOLVE].flex_data)
    cyc = set(row_terms(pb, "rp_inter_period_cyclic"))
    assert "rp_inter_period_cyclic[lh2_A,y2040,t0001,y2040,t0145]" in cyc
    assert "rp_inter_period_cyclic[lh2_C,y2030,t0001,y2040,t0145]" in cyc
    assert "rp_inter_period_balance[lh2_C,y2040,t0001,y2030,t0145]" in set(row_terms(pb, "rp_inter_period_balance"))


# ---------------------------------------------------------------------------
# Periods without representative periods in an rp solve
# ---------------------------------------------------------------------------


def _y2040_plain_timeset(methods):
    weights = {f"t{i:04d}": (2.0 if i < 73 else 1.0) for i in range(49, 97)}
    return _overlay(
        [["timeset", "w2040", "timeline", "y2030_168h"],
         ["timeset", "w2040", "timeset_duration", L._pmap({"t0049": 48.0}, "time")],
         ["timeset", "w2040", "timeset_weights", L._pmap(weights, "time")],
         ["solve", SOLVE, "period_timeset",
          {"type": "map", "index_type": "str", "index_name": "period", "rank": 1,
           "data": [["y2030", "rp_y2030" + SFX], ["y2040", "w2040"]]}]] + _methods(methods),
        entities=[["timeset", "w2040"]])


def test_mixed_rp_and_non_rp_periods_with_blended_node_raises(tmp_path_factory):
    with pytest.raises(FlexToolConfigError, match="mixes periods with representative periods"):
        _run(tmp_path_factory, _y2040_plain_timeset("bind_within_period_blended_weights"))


def test_mixed_rp_and_non_rp_periods_keep_both_weights(tmp_path_factory):
    fd = _run(tmp_path_factory, _y2040_plain_timeset("bind_within_solve"))[SOLVE].flex_data
    assert _weights(fd, "y2030") == [1.4, 0.6]
    assert _weights(fd, "y2040") == [round(2.0 * 48 / 72, 9), round(48 / 72, 9)]


# ---------------------------------------------------------------------------
# Solves where representative periods are not defined
# ---------------------------------------------------------------------------


def test_stochastic_solve_with_rp_timeset_raises(tmp_path_factory):
    """The forecast branch periods have no timeset of their own, so they would get no rp data."""
    rpw = {"type": "map", "index_type": "str", "index_name": "base_period", "rank": 1,
           "data": [[b, {"type": "map", "index_type": "str", "index_name": "representative_period", "rank": 1,
                         "data": [["t0001", 1.0]]}] for b in ("t0001", "t0025")]}

    def patch(data):
        return L._overlay(data, name="custom", base="2_day_stochastic_dispatch", values=[
            ["timeset", "2day", "timeset_duration", L._pmap({"t0001": 24.0}, "time")],
            ["timeset", "2day", "representative_period_weights", rpw]])
    with pytest.raises(FlexToolConfigError, match="does not support this combination"):
        _run(tmp_path_factory, patch, STOCH_JSON)


def test_rolling_solve_with_blended_nodes_raises(tmp_path_factory):
    patch = _overlay([["solve", SOLVE, "solve_mode", "rolling_window"],
                      ["solve", SOLVE, "rolling_solve_horizon", 48.0],
                      ["solve", SOLVE, "rolling_solve_jump", 24.0]])
    with pytest.raises(FlexToolConfigError, match="rolling_window solve on representative periods"):
        _run(tmp_path_factory, patch)


# ---------------------------------------------------------------------------
# Semantic pin: an RP model whose base days are exact copies of their rep days matches the hourly model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("exp", ["e1", "e2"])
@pytest.mark.parametrize("variant", ["ws", "fo", "wp"])
def test_rp_matches_hourly_reference(tmp_path_factory, exp, variant):
    """Within 1 % of the hourly model of the same days (before: 3–100 % off in ws / fo).  within_period's
    hourly twin cycles each period with one 168 h bind_within_timeblock block."""
    fn = H.EXPERIMENTS[exp]
    rp = _total_cost(_run(tmp_path_factory, fn("rp", H.BLENDED[variant]))[SOLVE])
    if variant == "wp":
        hourly_patch = H.one_block(fn, "hourly", "bind_within_timeblock")
    else:
        hourly_patch = fn("hourly", H.PLAIN[variant])
    hourly = _total_cost(_run(tmp_path_factory, hourly_patch)[SOLVE])
    assert rp == pytest.approx(hourly, rel=1e-2), (exp, variant, rp, hourly)
