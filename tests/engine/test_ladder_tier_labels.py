"""Price-ladder tier labels are text, kept as written; every tier needs a price.

A tier label names a tier: ``01``, ``1`` and ``a`` are three different tiers, and the LP buys the tiers by
price whatever their labels.  Before, the input derivation turned labels into integers (``01`` → ``1``) and
skipped labels that were not integers, while the price / quantity Params kept the label as written: with
tiers ``01/02/03`` the ``v_trade`` columns were on tiers ``1/2/3`` and the prices on ``01/02/03``, so the
commodity was bought at no cost; tiers ``a/b/c`` were dropped.  A tier with a quantity but no price was given
the price 0; it is now an error.

Each relabelled case is compared with the ``1/2/3`` twin: the same objective and the same ladder fields after
mapping the labels back.  Fixtures: ``tests.json`` scenarios ``coal_ladder_annual`` /
``coal_ladder_cumulative`` with the coal ladder re-authored in an override alternative.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests._support.fields import field_diff
from tests.db_utils import json_to_db
from tests.engine.test_ladder_by_shape import TESTS_JSON, _lps, _tier

#: (price, quantity) of the three tiers, cheapest first; the last tier is unlimited.
ANN_TIERS = [(20.0, 100000.0), (25.0, 100000.0), (30.0, float("inf"))]
CUM_TIERS = [(20.0, 1000000.0), (25.0, 1000000.0), (30.0, float("inf"))]

_LADDER_FIELDS = {
    "annual": ("p_ladder_ann_price", "p_ladder_ann_quantity", "commodity__tier_ann", "cndi_ladder_ann"),
    "cumulative": ("p_ladder_cum_price", "p_ladder_cum_quantity", "ci_ladder_cumulative", "cndi_ladder_cum"),
}
_BASE = {"annual": "coal_ladder_annual", "cumulative": "coal_ladder_cumulative"}
_ALT = {"annual": "ladder_ann_on", "cumulative": "ladder_cum_on"}


def _ladder(labels: list[str], tiers: list[tuple[float, float]]) -> dict:
    return {"index_type": "str", "rank": 2, "index_name": "x",
            "data": [[lbl, _tier(p, q)] for lbl, (p, q) in zip(labels, tiers)]}


def _data(kind: str, name: str, ladder: dict, extra: list = ()) -> dict:
    """``tests.json`` with coal's ``price_ladder_<kind>`` replaced by ``ladder`` in scenario ``name``."""
    data = json.loads(TESTS_JSON.read_text())
    param = f"price_ladder_{kind}"
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "commodity" and v[1] == "coal" and v[2] == param and v[4] == _ALT[kind])]
    return _lps()._overlay(data, name=name, base=_BASE[kind],
                           values=[["commodity", "coal", param, ladder], *extra])


def _run(tmp_path_factory, data: dict, scenario: str) -> dict:
    """Build ``data``, run ``scenario`` cold; return the steps."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    work = tmp_path_factory.mktemp(scenario)
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "w", warm=False, keep_solutions=True)
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
    return steps


def _relabelled(value, mapping: dict[str, str]):
    """``value`` (a Param or frame) with its tier column (``i`` / ``tier``) mapped through ``mapping``."""
    frame = value.frame if hasattr(value, "frame") else value
    col = "i" if "i" in frame.columns else "tier"
    out = frame.with_columns(pl.col(col).cast(pl.Utf8).replace_strict(mapping))
    if hasattr(value, "frame"):
        from polar_high import Param
        return Param(value.dims, out)
    return out


def _assert_same_ladder(case_steps: dict, twin_steps: dict, kind: str, mapping: dict[str, str]) -> None:
    assert len(case_steps) == len(twin_steps)
    for case, twin in zip(case_steps.values(), twin_steps.values()):
        assert case.solution.obj == pytest.approx(twin.solution.obj, rel=1e-9)
        for field in _LADDER_FIELDS[kind]:
            got = getattr(case.flex_data, field)
            assert got is not None, field
            assert field_diff(_relabelled(got, mapping), getattr(twin.flex_data, field)) is None, field


@pytest.mark.parametrize("kind", ["annual", "cumulative"])
@pytest.mark.parametrize("labels", [["01", "02", "03"], ["a", "b", "c"]], ids=["zero_padded", "letters"])
def test_tier_labels_are_kept_as_written(tmp_path_factory, kind, labels) -> None:
    """Tiers ``01/02/03`` (or ``a/b/c``) ≡ tiers ``1/2/3``.  Before: ``01`` labels made coal free (objective
    9.30e8 instead of 1.249e9 on the annual ladder), ``a/b/c`` were skipped."""
    tiers = ANN_TIERS if kind == "annual" else CUM_TIERS
    tag = f"{kind}_{labels[0]}"
    case = _run(tmp_path_factory, _data(kind, f"lbl_{tag}", _ladder(labels, tiers)), f"lbl_{tag}")
    twin = _run(tmp_path_factory, _data(kind, f"lbl_{tag}_twin", _ladder(["1", "2", "3"], tiers)),
                f"lbl_{tag}_twin")
    _assert_same_ladder(case, twin, kind, dict(zip(labels, ["1", "2", "3"])))


def test_one_and_zero_one_are_two_tiers(tmp_path_factory) -> None:
    """Tiers ``1`` and ``01`` are two tiers: ``{1, 01, 3}`` ≡ ``{1, 2, 3}``.  Before, both became tier 1."""
    case = _run(tmp_path_factory, _data("annual", "lbl_pair", _ladder(["1", "01", "3"], ANN_TIERS)), "lbl_pair")
    twin = _run(tmp_path_factory, _data("annual", "lbl_pair_twin", _ladder(["1", "2", "3"], ANN_TIERS)),
                "lbl_pair_twin")
    _assert_same_ladder(case, twin, "annual", {"1": "1", "01": "2", "3": "3"})


_ROLLING = [
    ["solve", "y2020_2day_dispatch", "solve_mode", "rolling_window"],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", 24.0],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump", 12.0],
]


def test_rolling_cumulative_ladder_carries_text_labels(tmp_path_factory) -> None:
    """A rolling solve with a cumulative ladder whose first tier binds (quantity 2 000 MWh) carries the bought
    MWh per tier label from roll to roll: tiers ``01/02/03`` ≡ ``1/2/3`` in every roll.  Before, the carry
    keys were integers and the labels did not match."""
    tiers = [(20.0, 2000.0), (25.0, 1000000.0), (30.0, float("inf"))]
    labels = ["01", "02", "03"]
    case = _run(tmp_path_factory, _data("cumulative", "lbl_roll", _ladder(labels, tiers), _ROLLING), "lbl_roll")
    twin = _run(tmp_path_factory, _data("cumulative", "lbl_roll_twin", _ladder(["1", "2", "3"], tiers), _ROLLING),
                "lbl_roll_twin")
    assert len(case) > 1
    _assert_same_ladder(case, twin, "cumulative", dict(zip(labels, ["1", "2", "3"])))
    realized = [s.flex_data.p_ladder_cum_realized_mwh for s in case.values()]
    assert any(r is not None for r in realized), "the later rolls carry the earlier rolls' purchases"


def test_csv_reload_keeps_text_labels(tmp_path_factory, test_solver_config_dir) -> None:
    """A dump of a ``01/02/03`` ladder reloads with the same tier labels in the ladder sets (the reload reads
    ``tier`` columns as text, so ``01`` does not become ``1``)."""
    from flextool.engine._csv_reload import copy_runner_meta, load_from_dump
    from flextool.update_flextool.db_migration import migrate_database
    from tests._support.load import run_scenario

    work = tmp_path_factory.mktemp("lbl_reload")
    (work / "f.json").write_text(json.dumps(_data("annual", "lbl_reload", _ladder(["01", "02", "03"], ANN_TIERS))))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    run = run_scenario(url, "lbl_reload", work / "work_lbl_reload", solver_config_dir=test_solver_config_dir)
    dump = work / "dump"
    run.last_flex_data.dump_csvs(dump, include_heavy=True)
    assert copy_runner_meta(run.work_folder, dump) > 0
    with pytest.warns(Warning):
        reloaded = load_from_dump(dump)
    for field in ("commodity__tier_ann", "cndi_ladder_ann"):
        got = getattr(reloaded, field)
        assert got is not None, field
        assert field_diff(got, getattr(run.last_flex_data, field)) is None, field


def test_tier_with_quantity_but_no_price_is_an_error(tmp_path_factory) -> None:
    """A tier with a quantity and no price stops the run, naming the commodity, the period and the tier."""
    ladder = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(20.0, 100000.0)],
                       ["2", {"type": "map", "index_type": "str", "rank": 1, "index_name": "x",
                              "data": [["quantity", 100000.0]]}],
                       ["3", _tier(30.0, float("inf"))]]}
    with pytest.raises(FlexToolConfigError, match=r"tier '2' of commodity 'coal' \(period p2020\) has a quantity "
                                                  r"but no price"):
        _run(tmp_path_factory, _data("annual", "lbl_noprice", ladder), "lbl_noprice")


def test_tier_without_quantity_is_unlimited(tmp_path_factory) -> None:
    """GUARD (the same before and after): a last tier with a price and no quantity ≡ the ``inf`` twin."""
    no_qty = {"type": "map", "index_type": "str", "rank": 1, "index_name": "x", "data": [["price", 30.0]]}
    ladder = {"index_type": "str", "rank": 2, "index_name": "x",
              "data": [["1", _tier(20.0, 100000.0)], ["2", _tier(25.0, 100000.0)], ["3", no_qty]]}
    case = _run(tmp_path_factory, _data("annual", "lbl_noqty", ladder), "lbl_noqty")
    twin = _run(tmp_path_factory, _data("annual", "lbl_noqty_twin", _ladder(["1", "2", "3"], ANN_TIERS)),
                "lbl_noqty_twin")
    for a, b in zip(case.values(), twin.values()):
        assert a.solution.obj == pytest.approx(b.solution.obj, rel=1e-9)


@pytest.mark.parametrize("labels,marked", [(["01", "02", "03"], True), (["1", "2", "3"], False)],
                         ids=["relabelled", "plain"])
def test_upgrade_marks_relabelled_ladders_and_runs_name_them(tmp_path_factory, capsys, labels, marked) -> None:
    """The database upgrade marks a database whose ladder has a tier label earlier releases read as another tier
    (``01``), not one with plain whole-number labels; with the marker, a run prints the notice naming coal."""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.update_flextool import db_migration as m

    tag = f"lbl_mark_{labels[0]}"
    work = tmp_path_factory.mktemp(tag)
    (work / "f.json").write_text(json.dumps(_data("annual", tag, _ladder(labels, ANN_TIERS))))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    m.migrate_database(url)
    with DatabaseMapping(url) as db:
        m._v77_ladder_tier_marker(db)
        names = {md["name"] for md in db.get_items("metadata")}
        if marked:
            db.commit_session("marker")
    assert (m.BEHAVIOUR_CHANGE_MARKER_PREFIX + "ladder_tier_labels_as_written" in names) is marked
    capsys.readouterr()
    run_chain_from_db(url, tag, work_folder=work / "w", warm=False)
    out = capsys.readouterr().out
    notices = [ln for ln in out.splitlines() if "tier labels are now kept as written" in ln]
    assert len(notices) == (1 if marked else 0)
    if marked:
        assert "Affected commodities: coal." in notices[0]
