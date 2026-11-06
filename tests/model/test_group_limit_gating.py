"""Group limits follow their method / flag (#69).

The capacity margin, inertia and non-synchronous rows are built for

    groupCapacityMargin = groups with capacity_margin_method = manual
    groupInertia        = groups with has_inertia = yes
    groupNonSync        = groups with has_non_synchronous = yes

and nowhere else.  A flagged group without a limit reads the limit as 0 (capacity margin: capacity covers the peak
net load; non-synchronous: no non-synchronous infeed without the penalised slack; inertia: no requirement).

Before the fix the sets were derived from the limit values when no group had the method / flag: a scalar limit
(and, for inertia and non-synchronous, a period map) built its constraint without the method or flag on every
solve type, and a flagged group without a limit got no row when it was alone but a limit-0 row next to another
flagged group with a limit.

Vehicles (databases built from JSON, never a checked-in ``.sqlite``):
- ``single``: ``tests/fixtures/tests.json`` chain init / west / coal / wind / inertia / capacity_margin (one invest
  solve, p2020).  Groups ``capacity_margin`` (capacity margin) and ``electricity`` (inertia, non-synchronous).
- ``roll``: ``stochastics.json`` ``1_week_rolling_wind`` made deterministic (7 rolls), group ``elec_g`` ∋
  ``demand_node``.
- ``stoch``: ``stochastics.json`` ``2_day_stochastic_dispatch``, group ``elec_g``.
The method / flag values of the fixture are stripped from the JSON, so "not authored" is tested.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, build, cstr_over, make_db, utf8

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
TESTS_JSON = FIXTURES / "tests.json"
STOCHASTICS_JSON = FIXTURES / "stochastics.json"

ROWS = ("capacityMargin", "inertia_constraint", "non_sync_constraint")
SLACKS = ("vq_capacity_margin", "vq_inertia", "vq_non_synchronous")
FLAGS = (("capacity_margin_method", "manual"), ("has_inertia", "yes"), ("has_non_synchronous", "yes"))
LIMITS = ("capacity_margin", "inertia_limit", "non_synchronous_limit")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


# ---------------------------------------------------------------------------
# Helpers


def scalar(v: float) -> list:
    return b64(float(v), "float")


def pmap(values: dict) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def s(v: str) -> list:
    return b64(v, "str")


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _stripped_json(tmp_path_factory, src: Path, *, deterministic: bool, drop_limits: bool) -> Path:
    """``src`` without the group methods / flags (and, with ``drop_limits``, without the group limits).

    ``deterministic``: the ``stochastics.json`` rolling vehicle without branches (c8 / c10 pattern)."""
    data = json.loads(src.read_text())
    gone = {f for f, _ in FLAGS} | (set(LIMITS) if drop_limits else set())
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if not (pv[0] == "group" and pv[2] in gone)]
    if deterministic:
        data["parameter_values"] = [
            pv for pv in data["parameter_values"]
            if pv[2] not in ("stochastic_branches", "include_stochastics")
            and not (pv[0] == "profile" and pv[1] == "wind" and pv[4] in ("1week_rolling", "2day"))]
        data["entities"] += [["connection", "link", None],
                             ["connection__node__node", ["link", "gas_market", "coal_market"], None]]
        data["entity_alternatives"].append(["connection", ["link"], "init", True])
    out = tmp_path_factory.mktemp("gating_json") / src.name
    out.write_text(json.dumps(data))
    return out


# vehicle -> (fixture, scenario, deterministic, capacity-margin group, inertia / non-sync group, periods, limits)
VEHICLES = {
    "single": (TESTS_JSON, "y2020_2day_dispatch", False, "capacity_margin", "electricity", ["p2020"],
               (100.0, 200.0, 0.5)),
    "roll": (STOCHASTICS_JSON, "1_week_rolling_wind", True, "elec_g", "elec_g", ["period1"], (10.0, 200.0, 0.5)),
    "stoch": (STOCHASTICS_JSON, "2_day_stochastic_dispatch", False, "elec_g", "elec_g", ["period1"],
              (10.0, 200.0, 0.5)),
}
SINGLE_CHAIN = ["init", "west", "coal", "wind", "inertia", "capacity_margin"]
ELEC_G = [["group", "elec_g", None], ["group__node", ["elec_g", "demand_node"], None]]

_RUNS: dict = {}


def _run(tmp_path_factory, veh: str, values: list, *, drop_limits: bool = True, entities: list | None = None,
         warm: bool = True) -> dict:
    """Run vehicle ``veh`` with the fixture's group methods / flags (and limits) stripped plus ``values``."""
    from flextool.engine import run_chain_from_db

    key = json.dumps([veh, values, drop_limits, entities, warm])
    if key not in _RUNS:
        fixture, scenario, det, *_ = VEHICLES[veh]
        fj = _stripped_json(tmp_path_factory, fixture, deterministic=det, drop_limits=drop_limits)
        chain = SINGLE_CHAIN if veh == "single" else _chain(fixture, scenario)
        ents = list(entities or []) + (ELEC_G if veh != "single" else [])
        alt = f"c11a_{len(_RUNS)}"
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=fj,
                            entities=ents)
        steps = run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"),
                                  keep_solutions=True, warm=warm)
        assert steps, "run_chain_from_db produced no orchestration steps"
        for step in steps.values():
            assert step.solution is not None, "solve produced no solution"
        _RUNS[key] = steps
    return _RUNS[key]


def _limits(veh: str, *, as_map: bool = False) -> list:
    _, _, _, g_cm, g_in, periods, (cm, inertia, ns) = VEHICLES[veh]

    def v(x):
        return pmap({p: x for p in periods}) if as_map else scalar(x)
    return [["group", g_cm, "capacity_margin", v(cm)], ["group", g_in, "inertia_limit", v(inertia)],
            ["group", g_in, "non_synchronous_limit", v(ns)]]


def _flags(veh: str, *, on: bool = True) -> list:
    _, _, _, g_cm, g_in, *_ = VEHICLES[veh]
    vals = ("manual", "yes", "yes") if on else ("none", "no", "no")
    return [["group", g_cm, "capacity_margin_method", s(vals[0])], ["group", g_in, "has_inertia", s(vals[1])],
            ["group", g_in, "has_non_synchronous", s(vals[2])]]


def _rows(step) -> dict[str, int]:
    pb = build(step.flex_data)
    out = {}
    for name in ROWS:
        over = cstr_over(pb, name)
        out[name] = 0 if over is None else over.height
    return out


def _slack_count(step, name: str) -> int:
    try:
        v = step.solution.value(name)
    except KeyError:
        return 0
    return 0 if v is None else v.height


def _groups(step, field: str) -> list[str] | None:
    f = getattr(step.flex_data, field)
    return None if f is None else sorted(utf8(f)["g"].to_list())


def _objs(steps: dict) -> list[float]:
    return [float(st.solution.obj) for st in steps.values()]


def _assert_same_objs(a: dict, b: dict) -> None:
    oa, ob = _objs(a), _objs(b)
    assert len(oa) == len(ob)
    for x, y in zip(oa, ob):
        assert x == pytest.approx(y, rel=1e-9)


def _assert_nothing_built(steps: dict) -> None:
    for name, st in steps.items():
        assert _rows(st) == {n: 0 for n in ROWS}, name
        for v in SLACKS:
            assert _slack_count(st, v) == 0, (name, v)
        for field in ("groupCapacityMargin", "groupInertia", "groupNonSync"):
            assert _groups(st, field) is None, (name, field)


# ---------------------------------------------------------------------------
# Limit without method / flag: nothing is built


@pytest.mark.parametrize("veh", ["single", "roll", "stoch"])
def test_limits_without_flag_build_nothing(tmp_path_factory, veh) -> None:
    """Scalar limits without the method / flags give no rows, no slacks, and the model without limits.

    Before: 48 / 96 / 240 rows of each family on single / roll / stoch."""
    steps = _run(tmp_path_factory, veh, _limits(veh))
    _assert_nothing_built(steps)
    _assert_same_objs(steps, _run(tmp_path_factory, veh, []))


def test_limit_maps_without_flag_build_nothing(tmp_path_factory) -> None:
    """Period-map limits without the flags: no rows (before: inertia and non-sync 48 rows each)."""
    steps = _run(tmp_path_factory, "single", _limits("single", as_map=True))
    _assert_nothing_built(steps)
    _assert_same_objs(steps, _run(tmp_path_factory, "single", []))


def test_explicit_off_flags(tmp_path_factory) -> None:
    """``capacity_margin_method = none``, ``has_inertia = no``, ``has_non_synchronous = no`` with limits."""
    steps = _run(tmp_path_factory, "single", _limits("single") + _flags("single", on=False))
    _assert_nothing_built(steps)
    _assert_same_objs(steps, _run(tmp_path_factory, "single", []))


# ---------------------------------------------------------------------------
# Flags on


@pytest.mark.parametrize("veh", ["single", "roll", "stoch"])
def test_flag_scalar_equals_period_map(tmp_path_factory, veh) -> None:
    """With the flags, a scalar limit and the equivalent period map build the same rows and objective."""
    a = _run(tmp_path_factory, veh, _limits(veh) + _flags(veh))
    b = _run(tmp_path_factory, veh, _limits(veh, as_map=True) + _flags(veh))
    for sa, sb in zip(a.values(), b.values()):
        ra = _rows(sa)
        assert ra == _rows(sb)
        assert ra["inertia_constraint"] > 0 and ra["non_sync_constraint"] > 0
    _assert_same_objs(a, b)


def test_flag_without_limit(tmp_path_factory) -> None:
    """Flags on, no limit anywhere: the capacity margin reads margin 0 (capacity ≥ peak net load), the
    non-synchronous limit 0, and inertia has no requirement.  Before: no row at all (objective of the model
    without flags)."""
    steps = _run(tmp_path_factory, "single", _flags("single"))
    st = list(steps.values())[-1]
    assert _rows(st) == {"capacityMargin": 48, "inertia_constraint": 0, "non_sync_constraint": 48}
    assert _slack_count(st, "vq_capacity_margin") == 1
    assert _slack_count(st, "vq_non_synchronous") == 48
    unflagged = _objs(_run(tmp_path_factory, "single", []))[-1]
    assert float(st.solution.obj) > unflagged * (1 + 1e-6)


def test_only_flagged_group_counts(tmp_path_factory) -> None:
    """``capacity_margin`` has the method; ``electricity`` authors a margin without it and has inertia only."""
    steps = _run(tmp_path_factory, "single", _limits("single") + [
        ["group", "capacity_margin", "capacity_margin_method", s("manual")],
        ["group", "electricity", "capacity_margin", scalar(5.0)],
        ["group", "electricity", "has_inertia", s("yes")]])
    st = list(steps.values())[-1]
    assert _groups(st, "groupCapacityMargin") == ["capacity_margin"]
    assert _groups(st, "groupInertia") == ["electricity"]
    assert _groups(st, "groupNonSync") is None
    assert sorted(set(cstr_over(build(st.flex_data), "capacityMargin")["g"].to_list())) == ["capacity_margin"]
    assert _rows(st)["non_sync_constraint"] == 0


# ---------------------------------------------------------------------------
# Several inertia groups: a flagged group without a limit has no requirement


def test_flag_without_limit_next_to_flagged_group_with_limit(tmp_path_factory) -> None:
    """F = ``capacity_margin`` has_inertia without a limit, G = ``electricity`` has_inertia with 200: F has
    no requirement, so the model equals the one where only G is flagged."""
    g_only = [["group", "electricity", "has_inertia", s("yes")],
              ["group", "electricity", "inertia_limit", scalar(200.0)]]
    both = _run(tmp_path_factory, "single", g_only + [["group", "capacity_margin", "has_inertia", s("yes")]])
    ref = _run(tmp_path_factory, "single", g_only)
    st = list(both.values())[-1]
    assert _groups(st, "groupInertia") == ["capacity_margin", "electricity"]
    _assert_same_objs(both, ref)
    # G's rows and slack are those of the G-only model.
    rows = cstr_over(build(st.flex_data), "inertia_constraint")
    ref_rows = cstr_over(build(list(ref.values())[-1].flex_data), "inertia_constraint")
    assert rows.filter(pl.col("g") == "electricity").height == ref_rows.height == 48


def test_flag_without_limit_next_to_unflagged_group_with_limit(tmp_path_factory, caplog) -> None:
    """F = ``capacity_margin`` has_inertia without a limit, U = ``electricity`` has a limit but no flag: no
    requirement anywhere (no crash), and both warnings are logged."""
    with caplog.at_level(logging.WARNING):
        steps = _run(tmp_path_factory, "single", [["group", "capacity_margin", "has_inertia", s("yes")],
                                                  ["group", "electricity", "inertia_limit", scalar(200.0)]])
    st = list(steps.values())[-1]
    assert _groups(st, "groupInertia") == ["capacity_margin"]
    rows = cstr_over(build(st.flex_data), "inertia_constraint")
    assert rows is None or set(rows["g"].to_list()) == {"capacity_margin"}
    _assert_same_objs(steps, _run(tmp_path_factory, "single", []))
    text = caplog.text
    assert "Group 'electricity' has inertia_limit but has_inertia is not yes: the limit is ignored." in text
    assert "Group 'capacity_margin' has has_inertia = yes but no inertia_limit" in text


# ---------------------------------------------------------------------------
# Warnings


def test_limit_without_flag_warns(tmp_path_factory, caplog) -> None:
    """A limit without its method / flag is ignored, and a warning names the group."""
    with caplog.at_level(logging.WARNING):
        _run(tmp_path_factory, "single", _limits("single") + [["group", "heat", "inertia_limit", scalar(0.0)]],
             entities=None, warm=False)
    text = caplog.text
    assert "Group 'capacity_margin' has capacity_margin but capacity_margin_method is not manual" in text
    assert "Group 'electricity' has inertia_limit but has_inertia is not yes" in text
    assert "Group 'electricity' has non_synchronous_limit but has_non_synchronous is not yes" in text
    # An explicit 0 without the flag has no effect and is not reported.
    assert "Group 'heat'" not in text


def test_flag_without_limit_warns_only_when_absent(tmp_path_factory, caplog) -> None:
    """The method / flag without any limit value warns; with an explicit 0 it does not."""
    with caplog.at_level(logging.WARNING):
        _run(tmp_path_factory, "single", _flags("single"), warm=False)
    absent = caplog.text
    assert "Group 'capacity_margin' has capacity_margin_method = manual but no capacity_margin" in absent
    assert "Group 'electricity' has has_inertia = yes but no inertia_limit" in absent
    assert "Group 'electricity' has has_non_synchronous = yes but no non_synchronous_limit" in absent
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        steps = _run(tmp_path_factory, "single", _flags("single") + [
            ["group", "capacity_margin", "capacity_margin", scalar(0.0)],
            ["group", "electricity", "inertia_limit", scalar(0.0)],
            ["group", "electricity", "non_synchronous_limit", scalar(0.0)]], warm=False)
    explicit = caplog.text
    assert "but no capacity_margin" not in explicit
    assert "but no inertia_limit" not in explicit
    assert "but no non_synchronous_limit" not in explicit
    # An explicit 0 means the same as an absent limit.
    _assert_same_objs(steps, _run(tmp_path_factory, "single", _flags("single"), warm=False))


# ---------------------------------------------------------------------------
# Warm start


def test_rolling_flags_warm_equals_cold(tmp_path_factory) -> None:
    """Rolling dispatch with all three flags and limits: the warm-started rolls equal cold rebuilds."""
    vals = _limits("roll") + _flags("roll")
    _assert_same_objs(_run(tmp_path_factory, "roll", vals, warm=True), _run(tmp_path_factory, "roll", vals, warm=False))
