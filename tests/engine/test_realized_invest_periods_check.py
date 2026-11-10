"""A realized investment period must be an investment period of the same solve (#319).

``SolveConfig.load_from_db`` stops with an error naming the solve and the periods; the fallback (no
``realized_invest_periods``) realizes the ``realized_periods`` that are also ``invest_periods``.  Databases are
built from the schema JSON (``tests/engine/_tiny_model_db.py``); nothing is solved.
"""
from __future__ import annotations

import pytest
import spinedb_api as api

from flextool.engine.input._solve_config import SolveConfig
from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _tiny_model_db as T


def _load(tmp_path, solves: dict) -> SolveConfig:
    return SolveConfig.load_from_db_url(T.build_db(tmp_path, solves), T.SCENARIO)


def _map2d(data: dict[str, list[str]]) -> api.Map:
    """A 2D map ``{solve suffix: {period: 'yes'}}`` (one duplicated solve per outer key)."""
    return api.Map(list(data), [api.Map(ps, ["yes"] * len(ps), index_name="period") for ps in data.values()],
                   index_name="solve")


def test_a_realized_invest_period_outside_invest_periods_stops(tmp_path) -> None:
    solves = {"inv": T.invest_rp_solve(realized_invest_periods=T.arr(["p1", "p2"]))}
    with pytest.raises(FlexToolConfigError) as err:
        _load(tmp_path, solves)
    msg = str(err.value)
    assert "solve 'inv'" in msg and "['p2']" in msg and "['p1']" in msg


def test_realized_invest_periods_without_invest_periods_stops(tmp_path) -> None:
    # A dispatch solve with a leftover realized_invest_periods.
    solves = {"disp": T.dispatch_solve(realized_invest_periods=T.arr(["p1"]))}
    with pytest.raises(FlexToolConfigError, match=r"solve 'disp'.*\['p1'\].*invest_periods \[\]"):
        _load(tmp_path, solves)


def test_a_subset_loads(tmp_path) -> None:
    cfg = _load(tmp_path, {"inv": T.invest_rp_solve(invest_periods=T.arr(["p1", "p2"]))})
    assert [t[1] for t in cfg.realized_invest_periods["inv"]] == ["p1"]


def test_a_2d_invest_map_with_a_1d_realized_invest_value_stops(tmp_path) -> None:
    """The 1D value stays on the parent solve, which the 2D-map duplicates replace: it would be ignored."""
    solves = {"inv": T.invest_rp_solve(invest_periods=_map2d({"a": ["p1"]}))}
    with pytest.raises(FlexToolConfigError, match="'inv'.*2D map.*same solve names"):
        _load(tmp_path, solves)


def test_matching_2d_maps_are_checked_per_duplicated_solve(tmp_path) -> None:
    ok = _load(tmp_path, {"inv": T.invest_rp_solve(invest_periods=_map2d({"a": ["p1"]}),
                                                   realized_invest_periods=_map2d({"a": ["p1"]}))})
    assert [t[1] for t in ok.realized_invest_periods["inv_a"]] == ["p1"]


def test_a_2d_realized_invest_entry_without_its_invest_entry_stops(tmp_path) -> None:
    solves = {"inv": T.invest_rp_solve(invest_periods=_map2d({"a": ["p1"]}),
                                       realized_invest_periods=_map2d({"a": ["p1"], "b": ["p1"]}))}
    with pytest.raises(FlexToolConfigError, match=r"solve 'inv_b'.*\['p1'\]"):
        _load(tmp_path, solves)
