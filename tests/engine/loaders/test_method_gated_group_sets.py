"""The CO2-cap and reserve group sets follow the method, on the DB path as in the solve_data seed (item 84).

* ``group_co2_max_period`` is the set of groups whose ``co2_method`` caps the emissions per period, the same
  groups the LP gets ``co2_max_period`` rows for.  Before, the DB projection took every group with a
  ``co2_max_period`` value, so a ``price`` group with a stray value was reported as CO2-limited (84-a).  Such
  a group gets an input warning (the value is ignored).
* ``reserve_upDown_group`` leaves out ``no_reserve`` groups.  Before, the DB projection kept them, so each got
  shortfall columns ``vq_reserve`` over every time step (84-b), and a model whose only reserve group was
  ``no_reserve`` switched the reserve feature on (84-c).
* A ``co2_method`` with a cap but without its limit value is an input error naming the group, the method and
  the parameter (84-d; docs/reference/groups.md).

Expected values are from specs/bugfix/v1/oracle.md (by hand from the user docs), not from the code.  Fixture:
``tests.json`` plus an override alternative, built from JSON.
"""
from __future__ import annotations

import logging

import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine._lp_rows import row_bounds
from tests.engine._unit_db import b64, build, cstr_over, make_db, run, utf8

CO2_CHAIN = ["init", "west", "coal", "co2_limit"]
PLANT_CHAIN = ["init", "west", "coal"]
RESERVE_CHAIN = ["init", "west", "coal", "wind", "network", "reserve"]
NO_RESERVE_CHAIN = ["init", "west", "coal", "wind", "network"]
RESERVE_UP = ("primary", "up", "electricity")
RESERVE_DOWN = ("primary", "down", "electricity")
CO2_EXTRA_ENTITIES = [["group", "co2_extra", None],
                      ["group__node", ["co2_extra", "coal_market"], None]]


def num(v: float) -> list:
    return b64(float(v), "float")


def text(v: str) -> list:
    return b64(v, "str")


def _run(tmp_path_factory, alt: str, values: list, chain: list[str], *, entities=None) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, entities=entities)
    return run(tmp_path_factory, url, scen)


def _last(steps: dict):
    return list(steps.values())[-1]


def _groups(fd, field: str) -> set[str] | None:
    f = getattr(fd, field)
    return None if f is None else set(utf8(f)["g"].to_list())


def _capped_groups(step) -> set[str]:
    over = cstr_over(build(step.flex_data), "co2_max_period")
    return set() if over is None else set(over["g"].to_list())


def _objs(steps: dict) -> list[float]:
    return [float(st.solution.obj) for st in steps.values()]


def _assert_same_objs(a: dict, b: dict) -> None:
    oa, ob = _objs(a), _objs(b)
    assert len(oa) == len(ob)
    for x, y in zip(oa, ob):
        assert x == pytest.approx(y, rel=1e-9)


def _mps(step, path) -> bytes:
    build(step.flex_data).write_mps(str(path), emit_names=True)
    return path.read_bytes()


def _cols(step) -> list[str]:
    return list(build(step.flex_data).canonicalise().col_names)


def _vq_reserve_keys(step) -> dict[tuple[str, str, str], int]:
    out: dict[tuple[str, str, str], int] = {}
    for name in _cols(step):
        if name.startswith("vq_reserve["):
            r, ud, g = name[len("vq_reserve["):-1].split(",")[:3]
            out[(r, ud, g)] = out.get((r, ud, g), 0) + 1
    return out


# ---------------------------------------------------------------------------
# 84-a — group_co2_max_period is the LP's capped groups


def test_co2_period_cap_set_is_the_lp_capped_groups(tmp_path_factory) -> None:
    st = _last(_run(tmp_path_factory, "i84a_plain", [], CO2_CHAIN))
    assert _capped_groups(st) == {"co2_limit"}
    assert _groups(st.flex_data, "group_co2_max_period") == {"co2_limit"}


def test_co2_max_period_on_a_price_group_is_ignored_with_a_warning(tmp_path_factory, caplog) -> None:
    """co2_extra: co2_method price, co2_price 10, co2_max_period 100 → not capped, not CO2-limited in the
    outputs, the same LP as without its co2_max_period, and one warning naming the group and the method."""
    priced = [["group", "co2_extra", "co2_method", text("price")],
              ["group", "co2_extra", "co2_price", num(10.0)]]
    with caplog.at_level(logging.WARNING):
        steps = _run(tmp_path_factory, "i84a_extra",
                     priced + [["group", "co2_extra", "co2_max_period", num(100.0)]],
                     CO2_CHAIN, entities=CO2_EXTRA_ENTITIES)
    warnings = [r.getMessage() for r in caplog.records
                if r.levelno == logging.WARNING and "co2_extra" in r.getMessage()]
    ref = _run(tmp_path_factory, "i84a_ref", priced, CO2_CHAIN, entities=CO2_EXTRA_ENTITIES)
    st, ref_st = _last(steps), _last(ref)
    assert _capped_groups(st) == {"co2_limit"}
    assert _groups(st.flex_data, "group_co2_max_period") == {"co2_limit"}
    assert _groups(st.flex_data, "group_co2_max_total") in (None, set())
    _assert_same_objs(steps, ref)
    tmp = tmp_path_factory.mktemp("i84a_mps")
    assert _mps(st, tmp / "a.mps") == _mps(ref_st, tmp / "b.mps")
    assert len(warnings) == 1, warnings
    assert "co2_max_period" in warnings[0] and "'price'" in warnings[0]


# ---------------------------------------------------------------------------
# 84-b / 84-c — reserve_upDown_group leaves out no_reserve


def _no_reserve(rug: tuple[str, str, str]) -> list:
    return [["reserve__upDown__group", list(rug), "reserve_method", text("no_reserve")],
            ["reserve__upDown__group", list(rug), "reservation", num(10.0)],
            ["reserve__upDown__group", list(rug), "penalty_reserve", num(20000.0)]]


def test_no_reserve_group_gets_no_shortfall_columns(tmp_path_factory) -> None:
    """(primary, down, electricity) no_reserve next to the timeseries_only (primary, up, electricity): only the
    latter has vq_reserve columns (48 steps), and the LP is that of the unpatched model."""
    steps = _run(tmp_path_factory, "i84b", _no_reserve(RESERVE_DOWN), RESERVE_CHAIN,
                 entities=[["reserve__upDown__group", list(RESERVE_DOWN), None]])
    ref = _run(tmp_path_factory, "i84b_ref", [], RESERVE_CHAIN)
    st, ref_st = _last(steps), _last(ref)
    assert _groups(st.flex_data, "reserve_upDown_group") == {"electricity"}
    assert set(map(tuple, utf8(st.flex_data.reserve_upDown_group).select("r", "ud", "g").rows())) == {RESERVE_UP}
    assert _vq_reserve_keys(st) == {RESERVE_UP: 48}
    _assert_same_objs(steps, ref)
    tmp = tmp_path_factory.mktemp("i84b_mps")
    assert _mps(st, tmp / "a.mps") == _mps(ref_st, tmp / "b.mps")


def test_only_no_reserve_groups_switch_reserves_off(tmp_path_factory) -> None:
    """The one reserve group set to no_reserve (reservation 10 kept): no reserve variable or row, the feature
    is off, and the optimum is that of the model without the reserve alternative."""
    steps = _run(tmp_path_factory, "i84c", _no_reserve(RESERVE_UP)[:1], RESERVE_CHAIN)
    st = _last(steps)
    assert st.flex_data.reserve_upDown_group is None
    pb = build(st.flex_data)
    m = pb.canonicalise()
    assert not [c for c in m.col_names if c.startswith(("vq_reserve[", "v_reserve["))]
    assert not [r for r in m.row_names if r.startswith("reserve")]
    assert row_bounds(pb, "reserveBalance_timeseries_eq") == {}
    _assert_same_objs(steps, _run(tmp_path_factory, "i84c_ref", [], NO_RESERVE_CHAIN))


# ---------------------------------------------------------------------------
# 84-d — a capping co2_method without its limit value is an input error


@pytest.mark.parametrize("method, given, missing", [
    ("period", [], "co2_max_period"),
    ("period_total", [["group", "co2_limit", "co2_max_total", num(1.0e7)]], "co2_max_period"),
    ("price_total", [["group", "co2_limit", "co2_price", num(10.0)]], "co2_max_total"),
])
def test_co2_cap_method_without_its_value_is_an_input_error(tmp_path_factory, method, given, missing) -> None:
    """co2_limit (group__node coal_market) with the method only, no co2_limit alternative: no limit value
    anywhere in the scenario."""
    with pytest.raises(FlexToolConfigError,
                       match=rf"group 'co2_limit', co2_method '{method}' needs parameter '{missing}'"):
        _run(tmp_path_factory, f"i84d_{method}",
             [["group", "co2_limit", "co2_method", text(method)], *given], PLANT_CHAIN)
