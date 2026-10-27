"""Per-period invest / retire limits: a constant means the same as the equivalent period map (#66).

``cumulative_max_capacity`` / ``cumulative_min_capacity`` are "constant or period" (docs/reference.md), and the
engine reads every ``{invest,retire}_{max,min}_period`` limit through the shape resolver, so a constant applies to
every period:

    limit[e, d] = authored[e, d]           (period map)
                | authored[e]              (constant)
                | authored[e, anchor(d)]   (recourse branch period without its own value)

#66: a constant ``invest_max_period`` or ``cumulative_max_capacity`` gave no ``p_entity_max_units`` row.  The
max-units producer detected the period axis by column name and skipped the constant, so the cap defaulted to 0,
``p_flow_upper`` (built from it) was 0 and the entity could not use any capacity.  The legacy CSV seed, which is
authoritative for synthetic sub-solves (nested multi-invest steps, rolling rolls), dropped a constant
``invest_max_period`` too, because the scalar ``p_process`` / ``p_node`` specs did not list it.

The criterion: the constant and the equivalent period map give identical ``FlexData`` (every DataFrame / Param
field) and objective, deterministic and under recourse, and the named rows are present (so both variants cannot be
empty together).

Every DB is built from a JSON fixture plus an override alternative (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, run, utf8

REPO = Path(__file__).resolve().parents[2]
RECOURSE_JSON = REPO / "tests" / "fixtures" / "stoch_two_period_invest.json"
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"

DET = ["init", "base"]
REC = ["init", "base", "stoch", "recourse"]
DET_PERIODS = ["p2035", "p2040"]
REC_PERIODS = DET_PERIODS + [f"{a}_{b}" for a in DET_PERIODS for b in ("high", "low")]
GROUP_ENTS = [["group", "ginv", None], ["group__unit", ["ginv", "peaker"], None]]


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def S(v: float) -> list:
    """Constant."""
    return b64(float(v), "float")


def M(v: float, periods=DET_PERIODS, index_name: str | None = "period") -> list:
    """Period map with the same value in every period (``index_name=None``: Spine's silent ``x`` index)."""
    m = {"type": "map", "index_type": "str", "rank": 1, "data": [[d, float(v)] for d in periods]}
    if index_name is not None:
        m["index_name"] = index_name
    return b64(m, "map")


def method(ent: str, m: str, cls: str = "unit") -> list:
    return [cls, ent, "invest_method", b64(m, "str")]


# ---------------------------------------------------------------------------
# Runner: one solve per distinct (fixture, chain, values, entities); reference runs are shared between tests.

_RUNS: dict = {}


def _solve(tmp_path_factory, values: list, chain: list[str], *, entities: list | None = None,
           fixture_json: Path = RECOURSE_JSON) -> dict:
    key = json.dumps([str(fixture_json), chain, values, entities])
    if key not in _RUNS:
        url, scen = make_db(tmp_path_factory, alt=f"c7_{len(_RUNS)}", values=values, base_chain=chain,
                            entities=entities, fixture_json=fixture_json)
        _RUNS[key] = run(tmp_path_factory, url, scen)
    return _RUNS[key]


def _frame(x) -> pl.DataFrame | None:
    f = getattr(x, "frame", x)
    if isinstance(f, pl.LazyFrame):
        f = f.collect()
    return f if isinstance(f, pl.DataFrame) else None


def _all_fields(fd) -> dict[str, pl.DataFrame]:
    """Every DataFrame / Param field of a ``FlexData``, Enum columns as Utf8, sorted by all columns."""
    out = {}
    for f in dataclasses.fields(fd):
        fr = _frame(getattr(fd, f.name, None))
        if fr is None:
            continue
        fr = fr.with_columns(pl.col(c).cast(pl.Utf8) for c in fr.columns
                             if isinstance(fr.schema[c], (pl.Enum, pl.Categorical)))
        out[f.name] = fr.sort(fr.columns) if fr.width else fr
    return out


def _rows(fd, field: str, col: str, ent: str) -> dict[str, float]:
    f = _frame(getattr(fd, field, None))
    if f is None:
        return {}
    f = utf8(f).filter(pl.col(col) == ent)
    return {d: v for d, v in f.select("d", "value").iter_rows()}


def _assert_same(test_steps: dict, ref_steps: dict) -> None:
    """Same objective and identical FlexData in every solve of the chain."""
    assert list(test_steps) == list(ref_steps)
    for name in test_steps:
        t, r = test_steps[name], ref_steps[name]
        assert t.solution.obj == pytest.approx(r.solution.obj, rel=1e-9), name
        a, b = _all_fields(t.flex_data), _all_fields(r.flex_data)
        assert set(a) == set(b), name
        diff = sorted(k for k in a if not a[k].equals(b[k]))
        assert not diff, f"{name}: fields differ between constant and map: {diff}"


def _last(steps: dict):
    return list(steps.values())[-1]


# ---------------------------------------------------------------------------
# p_entity_max_units (#66: these failed before the fix)


@pytest.mark.parametrize("param,inv_method", [("invest_max_period", "invest_period"),
                                              ("cumulative_max_capacity", "cumulative_limits")])
@pytest.mark.parametrize("chain", ["det", "rec"])
def test_constant_cap_reaches_p_entity_max_units(tmp_path_factory, param, inv_method, chain):
    base_chain, periods = (DET, DET_PERIODS) if chain == "det" else (REC, REC_PERIODS)
    const = _solve(tmp_path_factory, [method("peaker", inv_method), ["unit", "peaker", param, S(20)]], base_chain)
    ref = _solve(tmp_path_factory, [method("peaker", inv_method), ["unit", "peaker", param, M(20)]], base_chain)
    _assert_same(const, ref)
    # peaker: existing 0, unitsize 1 -> max units = the cap, in every period (branch periods included).
    assert _rows(_last(const).flex_data, "p_entity_max_units", "e", "peaker") == {d: 20.0 for d in periods}


def test_constant_next_to_map_in_one_class(tmp_path_factory):
    """A constant on one unit and a period map on another unit of the same class."""
    wind = [method("wind", "invest_period"), ["unit", "wind", "invest_cost", S(1.0)],
            ["unit", "wind", "lifetime", S(1.0)], ["unit", "wind", "discount_rate", S(0.05)],
            ["unit", "wind", "invest_max_period", M(5)]]
    common = [method("peaker", "invest_period")] + wind
    mixed = _solve(tmp_path_factory, common + [["unit", "peaker", "invest_max_period", S(20)]], DET)
    ref = _solve(tmp_path_factory, common + [["unit", "peaker", "invest_max_period", M(20)]], DET)
    _assert_same(mixed, ref)
    fd = _last(mixed).flex_data
    assert _rows(fd, "p_entity_max_units", "e", "peaker") == {d: 20.0 for d in DET_PERIODS}
    assert set(_rows(fd, "p_entity_max_units", "e", "wind")) == set(DET_PERIODS)


@pytest.mark.parametrize("param,inv_method", [("invest_max_period", "invest_period"),
                                              ("cumulative_max_capacity", "cumulative_limits")])
def test_silent_x_index_map(tmp_path_factory, param, inv_method):
    """A period map without ``index_name`` (Spine's silent ``x`` index) equals the ``period``-named map.

    This is how the shipped fixtures author ``cumulative_max_capacity``.
    """
    xmap = _solve(tmp_path_factory, [method("peaker", inv_method),
                                     ["unit", "peaker", param, M(20, index_name=None)]], DET)
    ref = _solve(tmp_path_factory, [method("peaker", inv_method), ["unit", "peaker", param, M(20)]], DET)
    _assert_same(xmap, ref)
    assert _rows(_last(xmap).flex_data, "p_entity_max_units", "e", "peaker") == {d: 20.0 for d in DET_PERIODS}


# ---------------------------------------------------------------------------
# The rest of the family (already shape-resolved; guards that a constant keeps meaning the map)

# (id, invest_method, shape-independent values, (class, entity, parameter, value), checked field, field entity
#  column, entity, group entities, group invest_method)
# A divest variable needs existing capacity and a non-zero salvage_value (_eea_predicate).
RETIRABLE = [["unit", "peaker", "existing", S(100)], ["unit", "peaker", "salvage_value", S(1.0)]]

FAMILY = [
    ("invest_min_period", "invest_period", [["unit", "peaker", "invest_max_period", M(50)]],
     ("unit", "peaker", "invest_min_period", 30), "ed_invest_min_period", "e", "peaker", None, None),
    ("retire_max_period", "retire_period", RETIRABLE + [["unit", "peaker", "fixed_cost", S(1000)]],
     ("unit", "peaker", "retire_max_period", 20), "ed_divest_max_period", "e", "peaker", None, None),
    ("retire_min_period", "retire_period", RETIRABLE + [["unit", "peaker", "retire_max_period", M(50)]],
     ("unit", "peaker", "retire_min_period", 10), "ed_divest_min_period", "e", "peaker", None, None),
    ("cumulative_min_capacity", "cumulative_limits",
     [["unit", "peaker", "cumulative_max_capacity", M(50)]],
     ("unit", "peaker", "cumulative_min_capacity", 30), "ed_cumulative_min_capacity", "e", "peaker", None, None),
    ("group_invest_max_period", "invest_period", [["unit", "peaker", "invest_max_period", M(100)]],
     ("group", "ginv", "invest_max_period", 20), "p_group_invest_max_period", "g", "ginv", GROUP_ENTS,
     "invest_period"),
    ("group_invest_min_period", "invest_period", [["unit", "peaker", "invest_max_period", M(100)]],
     ("group", "ginv", "invest_min_period", 30), "p_group_invest_min_period", "g", "ginv", GROUP_ENTS,
     "invest_period"),
]


@pytest.mark.parametrize("case", FAMILY, ids=[c[0] for c in FAMILY])
@pytest.mark.parametrize("chain", ["det", "rec"])
def test_family_constant_equals_map(tmp_path_factory, case, chain):
    _, inv_method, base, (cls, ent, param, v), field, col, fent, ents, gmethod = case
    base_chain, periods = (DET, DET_PERIODS) if chain == "det" else (REC, REC_PERIODS)
    common = [method("peaker", inv_method)] + base
    if gmethod:
        common.append(method("ginv", gmethod, cls="group"))
    const = _solve(tmp_path_factory, common + [[cls, ent, param, S(v)]], base_chain, entities=ents)
    ref = _solve(tmp_path_factory, common + [[cls, ent, param, M(v)]], base_chain, entities=ents)
    _assert_same(const, ref)
    rows = _rows(_last(const).flex_data, field, col, fent)
    assert rows and set(rows) <= set(periods) and all(x == float(v) for x in rows.values()), rows


# ---------------------------------------------------------------------------
# Synthetic sub-solves read the legacy CSV seed (nested multi-invest steps; rolling rolls take the same path)

NESTED_CHAIN = ["init", "west", "coal", "coal_invest", "wind", "battery", "5weeks", "invest_solveSequence_5weeks",
                "invest_allYears_5weeks", "invest_oneYear4Times_5weeks", "invest_twoYears4Times_5weeks"]
NESTED_PERIODS = ["p2020", "p2025", "p2030", "p2035"]


def test_constant_invest_max_period_on_synthetic_sub_solves(tmp_path_factory):
    """``invest_5weeks`` with nested ``invest_periods`` runs as the synthetic sub-solves ``invest_5weeks_<p>``.

    These skip the DB-derived max-units producer and keep the CSV seed, which must carry a constant
    ``invest_max_period`` (scalar ``p_process`` spec).  ``contains_solves`` is emptied to drop the full-year
    storage/rolling children, which this check does not need.
    """
    common = [method("coal_plant", "invest_period"), ["unit", "coal_plant", "invest_cost", S(1.0)],
              ["unit", "coal_plant", "fixed_cost", S(0.0)],
              ["solve", "invest_5weeks", "contains_solves",
               b64({"type": "array", "value_type": "str", "data": []}, "array")]]
    const = _solve(tmp_path_factory, common + [["unit", "coal_plant", "invest_max_period", S(300)]], NESTED_CHAIN,
                   fixture_json=TESTS_JSON)
    ref = _solve(tmp_path_factory, common + [["unit", "coal_plant", "invest_max_period", M(300, NESTED_PERIODS)]],
                 NESTED_CHAIN, fixture_json=TESTS_JSON)
    assert list(const) == [f"invest_5weeks_{p}" for p in NESTED_PERIODS]
    _assert_same(const, ref)
    for name, step in const.items():
        rows = _rows(step.flex_data, "p_entity_max_units", "e", "coal_plant")
        existing = _rows(step.flex_data, "p_entity_all_existing", "e", "coal_plant")
        assert rows, name
        # coal_plant: existing 500 MW (+ earlier steps' investment), unitsize 500 -> the 300 MW cap adds 0.6.
        for d, v in rows.items():
            assert v == pytest.approx(existing.get(d, 0.0) / 500.0 + 0.6, rel=1e-12), (name, d)


# ---------------------------------------------------------------------------
# apply_derived_c: a configuration error from the producer is not hidden behind the CSV seed


@pytest.mark.parametrize("exc", ["config", "other"])
def test_p_entity_max_units_config_error_propagates(tmp_path_factory, monkeypatch, exc):
    import flextool.engine.preprocess.params._derived as D
    from flextool.engine.input._solve_state import FlexToolConfigError

    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise (FlexToolConfigError("bad shape") if exc == "config" else RuntimeError("legacy path"))

    monkeypatch.setattr(D, "p_entity_max_units_from_source", boom)
    values = [method("peaker", "invest_period"), ["unit", "peaker", "invest_max_period", M(20)]]
    url, scen = make_db(tmp_path_factory, alt=f"c7_raise_{exc}", values=values, base_chain=DET,
                        fixture_json=RECOURSE_JSON)
    if exc == "config":
        with pytest.raises(FlexToolConfigError, match="bad shape"):
            run(tmp_path_factory, url, scen)
        assert calls
    else:
        steps = run(tmp_path_factory, url, scen)
        assert calls
        # The generic handler keeps the CSV seed (correct for a deterministic period map).
        assert _rows(_last(steps).flex_data, "p_entity_max_units", "e", "peaker") == {d: 20.0 for d in DET_PERIODS}


# ---------------------------------------------------------------------------
# #5, B-Q3 — group invest/retire limit SETS follow invest_method, not "any group with a
# positive value of the matching max parameter".


def test_group_min_total_only_group_is_no_longer_silently_dropped(tmp_path_factory):
    """Covers B-Q3: ``g_invest_total`` used to be "any group with a positive
    invest_max_total" (ignoring ``invest_method``), not method-gated.  Two groups share
    ``invest_method = invest_total``: ``ga`` sets ``invest_max_total`` (so the DB-direct override
    of ``g_invest_total`` is non-empty and wins over the CSV-emit seed, which already computes the
    correct, method-gated set — the override must agree with it); ``gb`` sets only
    ``invest_min_total``, no max at all.

    Before the fix, the override's ``_g_param_set(source, "invest_max_total")`` picked up only
    ``ga`` (the one with a positive value) — ``gb`` silently dropped out of ``g_invest_total``
    even though its ``invest_method`` qualifies, so its ``invest_min_total`` floor was silently
    never enforced (no ``minInvestGroup_entity_total_path_p`` row, no error, just a wrong
    optimum).  After the fix, ``gb`` is correctly in the set — and FA-12 (unaffected by
    this fix) then correctly refuses to leave its max ambiguous: once ``ga`` sets
    ``invest_max_total``, a total-method group in the set with no value of its own reads as 0
    (silently forbidding investment) unless given one explicitly, so the solve now raises asking
    for ``gb``'s own ``invest_max_total`` (or a non-total method) instead of silently dropping the
    floor.  Both are correct; the silent, wrong-answer failure mode is gone either way.
    """
    entities = [["group", "ga", None], ["group", "gb", None],
                ["group__unit", ["ga", "peaker"], None],
                ["group__unit", ["gb", "peaker"], None]]
    values = [
        method("peaker", "invest_period"),
        ["unit", "peaker", "invest_max_period", M(100)],
        method("ga", "invest_total", cls="group"),
        ["group", "ga", "invest_max_total", S(50.0)],
        method("gb", "invest_total", cls="group"),
        ["group", "gb", "invest_min_total", S(95.0)],
    ]
    from flextool.engine.input._solve_state import FlexToolConfigError

    url, scen = make_db(tmp_path_factory, alt="group_min_total_only", values=values, base_chain=DET,
                        entities=entities, fixture_json=RECOURSE_JSON)
    with pytest.raises(FlexToolConfigError, match="gb"):
        run(tmp_path_factory, url, scen)
