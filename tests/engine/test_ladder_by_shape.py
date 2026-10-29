"""Price ladders read by shape, not by column name (#169).

``commodity.price_ladder_annual`` admits a depth-2 Map (``tier -> {price, quantity}``, the same ladder
in every model period) or a depth-3 Map (``period -> tier -> {price, quantity}``); ``price_ladder_cumulative``
is depth-2 only.  The pre-c25 ``_ladder_split`` / ``commodity__tier_ann`` / ``commodity__tier_cum`` detected
the tier and period columns by LITERAL NAME (``"tier" in cols``, ``"period" in cols``).  Spine names a Map's
index column after its ``index_name``, defaulting to ``"x"`` (and ``"x_2"``, ``"x_3"`` for deeper, unset
levels) when the author leaves it unset -- so a depth-2 ladder (no ``period`` column at all) and any
``x``-labelled ladder (no ``tier`` column either) both failed the literal check, and a commodity authoring
depth 2 next to another authoring depth 3 had its rows scrambled (the reader discovers index columns from
the single deepest row and places every row's values there by position).

The fix reads each entity's raw label PATH (:func:`._param_shapes.resolve_facet_groups`, never the label
text) to group entities by depth, and takes the axes of each depth-pure group BY POSITION
(:func:`._param_shapes.facet_long_frame`).  Built through the full production load (``run_chain_from_db``),
per critique.md item 8 -- a bare ``SpineDbReader`` without the axis contract Enums does not exercise the same
code paths (``_discover_index_cols`` names columns differently once the contract Enum-casts and scenario-trims
``period``-named columns).

Each case is compared, with :func:`tests._support.fields.field_diff` (frame + dtype equality, not just row
content), against the LABELLED twin that already resolves correctly -- never against a number this code
itself produced.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from tests._support.fields import field_diff
from tests.db_utils import json_to_db

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCH_JSON = REPO / "tests" / "fixtures" / "stochastics.json"


def _lps():
    name = "_lp_snapshot_for_ladder_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _tier(price: float, quantity: float, index_name: str = "x") -> dict:
    return {"type": "map", "index_type": "str", "rank": 1, "index_name": index_name,
            "data": [["price", price], ["quantity", quantity]]}


def _flex_data_of(tmp_path_factory, data: dict, scenario: str, tag: str, *, step: int = 0):
    """Build ``data`` and run ``scenario`` cold through the production loader
    (``run_chain_from_db``); return the FlexData of step ``step`` (0 = first)."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    work = tmp_path_factory.mktemp(tag)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "w", warm=False, keep_solutions=True)
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
    return list(steps.values())[step].flex_data


# ---------------------------------------------------------------------------
# (a) depth-2 annual ladder vs. the labelled depth-3 twin (coal_ladder_annual)
# ---------------------------------------------------------------------------


def _patch_ladder_ann_depth2_exact(data: dict) -> dict:
    """``coal_ladder_annual``'s depth-3 ladder, re-authored as the depth-2 form with EXACTLY the same tier
    values (including tier 3's genuine ``Infinity`` quantity -- ``tools/lp_snapshot.py``'s own
    ``ladder_ann_depth2`` patch uses ``1e30`` instead, an approximation that is LP-equivalent but not
    byte-equal to the labelled twin's raw value, which this test compares directly).  Depth 2 is
    period-invariant, which matches the base fixture exactly because its depth-3 ladder repeats the same
    three tiers for every period."""
    ladder = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(20.0, 100000.0)], ["2", _tier(25.0, 100000.0)],
                       ["3", _tier(30.0, float("inf"))]]}
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == "price_ladder_annual"
                and v[4] == "ladder_ann_on")]
    return _lps()._overlay(data, name="ladder_ann_depth2_exact", base="coal_ladder_annual",
                           values=[["commodity", "coal", "price_ladder_annual", ladder]])


def test_annual_depth2_matches_labelled_depth3(tmp_path_factory):
    case_data = _patch_ladder_ann_depth2_exact(json.loads(TESTS_JSON.read_text()))
    twin_data = json.loads(TESTS_JSON.read_text())
    case = _flex_data_of(tmp_path_factory, case_data, "ladder_ann_depth2_exact", "lad_a_case")
    twin = _flex_data_of(tmp_path_factory, twin_data, "coal_ladder_annual", "lad_a_twin")

    for field in ("p_ladder_ann_price", "p_ladder_ann_quantity", "commodity__tier_ann"):
        assert field_diff(getattr(case, field), getattr(twin, field)) is None, field


# ---------------------------------------------------------------------------
# (b) cumulative ladder with ``x`` on both levels vs. the labelled twin
# ---------------------------------------------------------------------------


def _patch_ladder_cum_silent_exact(data: dict) -> dict:
    """``coal_ladder_cumulative`` with ``index_name="x"`` on both levels and EXACTLY the labelled twin's
    values (see ``_patch_ladder_ann_depth2_exact``'s docstring for why this doesn't reuse
    ``tools/lp_snapshot.py``'s own ``ladder_cum_silent`` patch)."""
    ladder = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(20.0, 1000000.0)], ["2", _tier(25.0, 1000000.0)],
                       ["3", _tier(30.0, float("inf"))]]}
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == "price_ladder_cumulative"
                and v[4] == "ladder_cum_on")]
    return _lps()._overlay(data, name="ladder_cum_silent_exact", base="coal_ladder_cumulative",
                           values=[["commodity", "coal", "price_ladder_cumulative", ladder]])


def test_cumulative_x_on_both_levels_matches_labelled(tmp_path_factory):
    case_data = _patch_ladder_cum_silent_exact(json.loads(TESTS_JSON.read_text()))
    twin_data = json.loads(TESTS_JSON.read_text())
    case = _flex_data_of(tmp_path_factory, case_data, "ladder_cum_silent_exact", "lad_b_case")
    twin = _flex_data_of(tmp_path_factory, twin_data, "coal_ladder_cumulative", "lad_b_twin")

    for field in ("p_ladder_cum_price", "p_ladder_cum_quantity", "commodity__tier_cum"):
        assert field_diff(getattr(case, field), getattr(twin, field)) is None, field


# ---------------------------------------------------------------------------
# (c) annual depth-3 ladder with ``x`` on every level vs. the labelled twin
# ---------------------------------------------------------------------------


def _patch_ladder_ann_all_x(data: dict) -> dict:
    """``coal_ladder_annual``'s depth-3 ladder, re-authored with ``index_name="x"`` on every level (period,
    tier AND facet) instead of just the facet.  Before the fix: ``[name, x, x_2, x_3, value]``, with no
    ``period`` or ``tier`` column at all, so the literal column-name check returned ``(None, None)``.
    The base fixture repeats the same three tiers for every one of its four periods (p2020, p2025, p2030,
    p2035), so this re-authoring must do the same to stay an exact match of the labelled twin."""
    tier_map = {"type": "map", "index_type": "str", "rank": 2, "index_name": "x",
                "data": [["1", _tier(20.0, 100000.0)], ["2", _tier(25.0, 100000.0)],
                         ["3", _tier(30.0, float("inf"))]]}
    ladder = {"index_type": "str", "rank": 3, "index_name": "x",
              "data": [[p, tier_map] for p in ("p2020", "p2025", "p2030", "p2035")]}
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == "price_ladder_annual"
                and v[4] == "ladder_ann_on")]
    return _lps()._overlay(data, name="ladder_ann_all_x", base="coal_ladder_annual",
                           values=[["commodity", "coal", "price_ladder_annual", ladder]])


def test_annual_depth3_all_x_matches_labelled(tmp_path_factory):
    case_data = _patch_ladder_ann_all_x(json.loads(TESTS_JSON.read_text()))
    twin_data = json.loads(TESTS_JSON.read_text())
    case = _flex_data_of(tmp_path_factory, case_data, "ladder_ann_all_x", "lad_c_case")
    twin = _flex_data_of(tmp_path_factory, twin_data, "coal_ladder_annual", "lad_c_twin")

    for field in ("p_ladder_ann_price", "p_ladder_ann_quantity", "commodity__tier_ann"):
        assert field_diff(getattr(case, field), getattr(twin, field)) is None, field


# ---------------------------------------------------------------------------
# (d) mixed depths: coal2 depth-2 next to coal's depth-3
# ---------------------------------------------------------------------------


def _patch_ladder_mixed_depths(data: dict) -> dict:
    """``coal`` keeps its depth-3 annual ladder; a new commodity ``coal2`` authors the depth-2 form with
    different tier values, on the same scenario.  Before the fix: the reader discovers index columns from
    coal's depth-3 row (the single deepest), and coal2's depth-2 values land in those columns by position
    -- its tier keys end up in the ``period`` slot."""
    data = dict(data)
    data["entities"] = list(data["entities"]) + [["commodity", "coal2", None]]
    data["entity_alternatives"] = list(data["entity_alternatives"]) + [
        ["commodity", ["coal2"], "ladder_ann_on", True]]
    ladder2 = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(50.0, 2.0)], ["2", _tier(60.0, float("inf"))]]}
    data["parameter_values"] = list(data["parameter_values"]) + [
        ["commodity", "coal2", "price_ladder_annual", _lps()._pack(ladder2), "ladder_ann_on"]]
    return data


def test_mixed_depths_each_commodity_keeps_its_own(tmp_path_factory):
    data = json.loads(TESTS_JSON.read_text())
    data = _lps()._overlay(data, name="ladder_mixed_depths", base="coal_ladder_annual", values=[])
    data = _patch_ladder_mixed_depths(data)
    fd = _flex_data_of(tmp_path_factory, data, "ladder_mixed_depths", "lad_d")

    # coal: depth-3, one row per authored tier (p2020 only -- the sole period realized by
    # ``coal_ladder_annual``'s non-rolling base solve, even though tests.json's ``ladder_ann_on``
    # authors the same three tiers for p2025/p2030/p2035 too for the rolling-chain debug cases).
    coal_rows = fd.p_ladder_ann_price.frame.filter(pl.col("c").cast(pl.Utf8) == "coal").sort("i")
    assert coal_rows["d"].to_list() == ["p2020", "p2020", "p2020"]
    assert coal_rows["value"].to_list() == [20.0, 25.0, 30.0]

    # coal2: depth-2, cross-joined with every one of the solve's own periods.
    coal2_rows = fd.p_ladder_ann_price.frame.filter(pl.col("c").cast(pl.Utf8) == "coal2").sort("i", "d")
    solve_periods = sorted(set(fd.dt["d"].cast(pl.Utf8).to_list()))
    assert sorted(set(coal2_rows["d"].cast(pl.Utf8).to_list())) == solve_periods
    assert coal2_rows.filter(pl.col("i") == "1")["value"].to_list() == [50.0] * len(solve_periods)
    assert coal2_rows.filter(pl.col("i") == "2")["value"].to_list() == [60.0] * len(solve_periods)

    tiers = fd.commodity__tier_ann.sort("c", "i")
    assert tiers.filter(pl.col("c").cast(pl.Utf8) == "coal")["i"].to_list() == ["1", "2", "3"]
    assert tiers.filter(pl.col("c").cast(pl.Utf8) == "coal2")["i"].to_list() == ["1", "2"]


# ---------------------------------------------------------------------------
# (e) annual depth-2 on a 2-period solve: one row per tier per solve period
# ---------------------------------------------------------------------------


def _patch_ladder_depth2_multi_year(data: dict) -> dict:
    """``multi_year``'s first solve (``y2020_5week``) has two active periods (p2020, p2025) sharing one
    timeset.  coal authors the depth-2 annual ladder in place of ``multi_year``'s own (none -- the base
    fixture authors no ladder on ``multi_year``'s chain, so this just adds one)."""
    ladder = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(20.0, 1.0)], ["2", _tier(30.0, float("inf"))]]}
    return _lps()._overlay(data, name="ladder_depth2_multi_year", base="multi_year", values=[
        ["commodity", "coal", "price_ladder_annual", ladder]])


def test_annual_depth2_one_row_per_tier_per_period(tmp_path_factory):
    data = _patch_ladder_depth2_multi_year(json.loads(TESTS_JSON.read_text()))
    fd = _flex_data_of(tmp_path_factory, data, "ladder_depth2_multi_year", "lad_e", step=0)

    periods = sorted(set(fd.dt["d"].cast(pl.Utf8).to_list()))
    assert len(periods) == 2, f"expected a 2-period first solve; got {periods}"
    rows = fd.p_ladder_ann_price.frame.filter(pl.col("c").cast(pl.Utf8) == "coal").sort("i", "d")
    assert sorted(set(rows["d"].cast(pl.Utf8).to_list())) == periods
    assert rows.filter(pl.col("i") == "1")["value"].to_list() == [20.0, 20.0]
    assert rows.filter(pl.col("i") == "2")["value"].to_list() == [30.0, 30.0]


# ---------------------------------------------------------------------------
# (f) stochastic solve: depth-2 cross-joins the solve's NON-branch periods only (critique.md item 7)
# ---------------------------------------------------------------------------


def test_stochastic_depth2_excludes_branch_periods(tmp_path_factory):
    """A depth-2 annual ladder on a stochastic solve must equal its depth-3 equivalent (authored on the
    anchor period only): depth-3 has no branch-period rows today (design.md O-2), and the emit expands
    depth 2 over the model's own periods, not the branches either -- so depth 2 must exclude them too, or
    it would be a form-dependent LP no fixture covers."""
    ladder2 = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(20.0, 1.0)], ["2", _tier(30.0, float("inf"))]]}
    ladder3 = {"index_type": "str", "rank": 3, "index_name": "period",
              "data": [["period1", {"type": "map", "index_type": "str", "rank": 2, "index_name": "x",
                                     "data": [["1", _tier(20.0, 1.0)], ["2", _tier(30.0, float("inf"))]]}]]}
    lps = _lps()
    case_data = lps._overlay(json.loads(STOCH_JSON.read_text()), name="ladder_stoch_case",
                             base="2_day_stochastic_dispatch", values=[
                                 ["commodity", "coal", "price_ladder_annual", ladder2]])
    twin_data = lps._overlay(json.loads(STOCH_JSON.read_text()), name="ladder_stoch_twin",
                             base="2_day_stochastic_dispatch", values=[
                                 ["commodity", "coal", "price_ladder_annual", ladder3]])
    case = _flex_data_of(tmp_path_factory, case_data, "ladder_stoch_case", "lad_f_case")
    twin = _flex_data_of(tmp_path_factory, twin_data, "ladder_stoch_twin", "lad_f_twin")

    periods = sorted(set(case.dt["d"].cast(pl.Utf8).to_list()))
    assert len(periods) > 1, "expected branch periods in dt (the anchor plus at least one branch)"
    for field in ("p_ladder_ann_price", "p_ladder_ann_quantity", "commodity__tier_ann"):
        assert field_diff(getattr(case, field), getattr(twin, field)) is None, field
    # Explicitly confirm the branch periods carry no row of their own (O-2, not fixed here).
    rows = case.p_ladder_ann_price.frame
    assert sorted(set(rows["d"].cast(pl.Utf8).to_list())) == ["period1"]
