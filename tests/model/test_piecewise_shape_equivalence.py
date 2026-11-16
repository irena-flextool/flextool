"""A multi-line ``piecewise_linear`` curve given once equals the same curve given per period (design T2; CLAUDE.md:
a parameter with several shapes needs a constant-vs-equivalent-map test in single, rolling and stochastic solves).

``coal_plant`` (one input, one output: remapped, ``process_piecewise_remap``) gets the curve
{0.4: 0.30, 0.7: 0.40, 1.0: 0.42} (two lines, rising: online form) as a 1-level map, and as a 2-level map with
that curve in the solve's period (the stochastic solve's map has the anchor period ``period1`` only: every branch
period inherits it).  Both give the same lines per period and the same LP (every row, column, bound and
coefficient; the written MPS byte for byte) in every step of a single, a rolling and a stochastic solve.

Databases are built from JSON fixtures (``tests/fixtures/tests.json``, ``stochastics.json``), never a checked-in
``.sqlite``.  Every run has ``FLEXTOOL_AUTOSCALE_STRICT=1``.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine._unit_db import RunCache, b64, build, cstr_over, utf8

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCHASTICS_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
SOLVE = "y2020_2day_dispatch"
POINTS = [(0.4, 0.30), (0.7, 0.40), (1.0, 0.42)]
PLANT = ["init", "west", "coal", "coal_min_load"]
ROLLING = [["solve", SOLVE, "solve_mode", b64("rolling_window", "str")],
           ["solve", SOLVE, "rolling_solve_horizon", b64(24.0, "float")],
           ["solve", SOLVE, "rolling_solve_jump", b64(12.0, "float")]]
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _chain(fixture_json: Path, scenario: str) -> list[str]:
    rows = [r for r in json.loads(fixture_json.read_text())["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def _curve() -> list:
    return b64({"index_type": "float", "data": [list(p) for p in POINTS]}, "map")


def _curve_per_period(period: str) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [
        [period, {"type": "map", "index_type": "float", "data": [list(p) for p in POINTS]}]]}, "map")


def _values(value: list, startup: str | None) -> list:
    out = [["unit", "coal_plant", "conversion_method", b64("piecewise_linear", "str")],
           ["unit", "coal_plant", "piecewise_linear_efficiency", value],
           ["unit", "coal_plant", "min_load", b64(0.4, "float")]]
    return out + ([["unit", "coal_plant", "startup_method", b64(startup, "str")]] if startup else [])


def _lp(pb) -> dict:
    m = pb.canonicalise()
    rows: dict = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    cols: dict = {}
    for c in range(m.n_cols):
        name = m.col_names[c]
        cols[name] = (float(m.col_obj[c]), float(m.col_lb[c]), float(m.col_ub[c]))
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][name] = float(m.val[k])
    return {"rows": rows, "cols": cols}


def _lines(fd) -> pl.DataFrame:
    return (utf8(fd.p_piecewise_slope.frame).join(utf8(fd.p_piecewise_intercept.frame), on=["p", "seg", "d"])
            .sort("p", "d", "seg"))


def _same(tmp_path_factory, tmp_path, label: str, chain: list[str], period: str, extra: list, *,
          fixture_json: Path = TESTS_JSON, entities: list | None = None, startup: str | None = None) -> dict:
    kw = dict(base_chain=chain, entities=entities, fixture_json=fixture_json)
    ref, _, _ = _RUNS.get(tmp_path_factory, alt=f"{label}_const", values=_values(_curve(), startup) + extra, **kw)
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"{label}_map",
                            values=_values(_curve_per_period(period), startup) + extra, **kw)
    assert list(steps) == list(ref)
    for name in ref:
        fd, fd_ref = steps[name].flex_data, ref[name].flex_data
        assert utf8(fd.process_piecewise)["p"].to_list() == ["coal_plant"]
        assert _lines(fd).equals(_lines(fd_ref)), name
        pb, pb_ref = build(fd), build(fd_ref)
        assert cstr_over(pb, "conversion_piecewise").height > 0
        assert _lp(pb) == _lp(pb_ref), f"{label} {name}: the LP differs"
        safe = name.replace("/", "_")
        pb.write_mps(tmp_path / f"{safe}_a.mps")
        pb_ref.write_mps(tmp_path / f"{safe}_b.mps")
        assert (tmp_path / f"{safe}_a.mps").read_bytes() == (tmp_path / f"{safe}_b.mps").read_bytes(), name
        assert float(steps[name].solution.obj) == float(ref[name].solution.obj), name
    return steps


def test_multi_line_constant_equals_period_map_single(tmp_path_factory, tmp_path) -> None:
    _same(tmp_path_factory, tmp_path, "t2_single", PLANT, "p2020", [])


def test_multi_line_constant_equals_period_map_rolling(tmp_path_factory, tmp_path) -> None:
    steps = _same(tmp_path_factory, tmp_path, "t2_rolling", PLANT, "p2020", ROLLING)
    assert len([k for k in steps if "_roll_" in k]) >= 3, list(steps)


def test_multi_line_constant_equals_period_map_stochastic(tmp_path_factory, tmp_path) -> None:
    """The map has the anchor period1 only; every branch period of the solve takes its curve (the same lines)."""
    steps = _same(tmp_path_factory, tmp_path, "t2_stoch", _chain(STOCHASTICS_JSON, "2_day_stochastic_dispatch"),
                  "period1", [], fixture_json=STOCHASTICS_JSON, startup="linear")
    fd = [s for k, s in steps.items() if "__realized_dispatch" not in k][-1].flex_data
    periods = set(utf8(fd.dt)["d"].to_list())
    lines = _lines(fd)
    assert len(periods) > 1 and set(lines["d"].to_list()) == periods
    assert lines.group_by("seg").agg(pl.col("value").n_unique())["value"].to_list() == [1, 1]
