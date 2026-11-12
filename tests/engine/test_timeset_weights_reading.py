"""``timeset.timeset_weights`` reaches every period that uses the timeset, in every authored index form.

The per-step weight is normalised per period over the period's active steps::

    p_timestep_weight[d, t] = raw[t] · n_d / Σ_{t' ∈ T_d} raw[t']      (raw[t] = 0 for a step the map leaves out)

(docs/reference/entities_and_time.md ``timeset_weights``; ``_solve_writers._compute_timeset_cost_weight_rows`` and
``_derived.p_timestep_weight_from_source``).  Every assertion below recomputes that formula from the authored map
and the solve's own ``(d, t)`` steps, so a period that does not see its timeset's weights (value 1.0) fails.

* Stochastic branch periods are copies of their anchor (calendar) period: they take the anchor's timeset and so its
  weights — single solve, recourse (two fanned periods) and rolling (the rolls are synthetic solves, which take the
  weights from the per-solve writer frame).
* The map's ``index_name`` may be ``time`` (or ``t``) or Spine's silent ``x``: all give the same weights and LP.  Any
  other shape (a period map, a scalar) raises instead of being ignored.

Databases are built from the JSON fixtures under the pytest tmp dir (never a checked-in ``.sqlite``).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db

pytestmark = pytest.mark.solver

FIX = FLEXTOOL_ROOT / "tests" / "fixtures"
STOCH = FIX / "stochastics.json"
STOCH_2P = FIX / "stoch_two_period.json"
COAL = ["init", "west", "coal"]


def _chain(fixture: Path, scenario: str) -> list[str]:
    data = json.loads(fixture.read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == scenario]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    out = []
    while alt is not None:
        out.append(alt)
        alt = nxt.get(alt)
    return out


def _raw(n_steps: int) -> dict[str, float]:
    """Alternating 1, 3, 1, 3, ... from t0001 (odd steps 1, even steps 3)."""
    return {f"t{i:04d}": (1.0 if i % 2 else 3.0) for i in range(1, n_steps + 1)}


def _weights_value(raw: dict[str, float], index_name: str | None) -> list:
    value: dict = {"index_type": "str", "data": [[t, w] for t, w in raw.items()]}
    if index_name is not None:
        value["index_name"] = index_name
    return b64(value, "map")


def _db(tmp_path_factory, alt: str, fixture: Path, scenario: str, weights: dict[str, list]):
    values = [["timeset", ts, "timeset_weights", v] for ts, v in weights.items()]
    return make_db(tmp_path_factory, alt=alt, values=values, base_chain=_chain(fixture, scenario),
                   fixture_json=fixture)


def _run(url: str, scen: str, wf: Path):
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scen, work_folder=wf, keep_solutions=True)
    assert steps and all(s.solution is not None for s in steps.values())
    return steps


def _weights(step) -> dict[tuple[str, str], float]:
    f = step.flex_data.p_timestep_weight.frame
    return {(str(d), str(t)): float(v) for d, t, v in f.select("d", "t", "value").iter_rows()}


def _expected(dt: dict[tuple[str, str], float], raw: dict[str, float]) -> dict[tuple[str, str], float]:
    """The normalisation formula of the module docstring over each period's own steps."""
    by_d: dict[str, list[str]] = {}
    for d, t in dt:
        by_d.setdefault(d, []).append(t)
    out = {}
    for d, ts in by_d.items():
        total = sum(raw.get(t, 0.0) for t in ts)
        for t in ts:
            out[(d, t)] = raw.get(t, 0.0) * len(ts) / total
    return out


def _assert_weighted(steps, raw: dict[str, float], branch_periods: set[str]) -> None:
    seen: set[str] = set()
    for name, step in steps.items():
        got = _weights(step)
        exp = _expected(got, raw)
        for key, v in got.items():
            assert v == pytest.approx(exp[key], rel=1e-9, abs=1e-12), (name, key, v, exp[key])
        seen |= {d for d, _t in got}
    assert branch_periods <= seen, f"branch periods {sorted(branch_periods - seen)} were not solved"


@pytest.mark.parametrize("case", ["single", "recourse", "rolling"])
def test_branch_periods_take_anchor_timeset_weights(tmp_path_factory, case) -> None:
    """single: stochastics.json ``2_day_stochastic_dispatch`` (timeset 2day, t0001-t0048, period1 fanned to
    period1_upper / _mid / _lower / _realized).  raw = 1, 3, 1, 3, ...: n = 48, Σ = 24·1 + 24·3 = 96, so every
    branch period has 1·48/96 = 0.5 at odd and 3·48/96 = 1.5 at even steps — the weights of the anchor period1.
    recourse: stoch_two_period.json ``stoch`` (ts1 = t0001-t0003 for p2035, ts2 = t0004-t0006 for p2040, both
    fanned to _high / _low): p2035_*: Σ = 1+3+1 = 5 → 0.6, 1.8, 0.6; p2040_*: Σ = 3+1+3 = 7 → 9/7, 3/7, 9/7.
    rolling: stochastics.json ``1_week_rolling_wind`` (timeset 1week, 168 steps, horizon 96, jump 24): each roll's
    branch periods are normalised over the roll's own steps (even windows: 0.5 / 1.5).  Before: 1.0 on every
    branch period (the weights were keyed by the period label only)."""
    if case == "single":
        fixture, scen, ts_steps = STOCH, "2_day_stochastic_dispatch", {"2day": 48}
        branches = {"period1_upper", "period1_mid", "period1_lower"}
    elif case == "recourse":
        fixture, scen, ts_steps = STOCH_2P, "stoch", {"ts1": 6, "ts2": 6}
        branches = {"p2035_high", "p2035_low", "p2040_high", "p2040_low"}
    else:
        fixture, scen, ts_steps = STOCH, "1_week_rolling_wind", {"1week": 168}
        branches = {"period1_upper", "period1_mid", "period1_lower"}
    raw = _raw(max(ts_steps.values()))
    url, s = _db(tmp_path_factory, f"tw_branch_{case}", fixture, scen,
                 {ts: _weights_value(raw, None) for ts in ts_steps})
    steps = _run(url, s, tmp_path_factory.mktemp(f"tw_branch_{case}_w"))
    _assert_weighted(steps, raw, branches)
    if case == "single":
        got = _weights(next(iter(steps.values())))
        assert sorted({v for (d, _t), v in got.items() if d in branches}) == pytest.approx([0.5, 1.5])


@pytest.mark.parametrize("index_name", ["time", "t"])
def test_time_index_name_equals_silent_index(tmp_path_factory, index_name) -> None:
    """tests.json coal case (timeset 2day, 48 steps, deterministic) with timeset_weights 1, 3, 1, 3, ...: a map
    with ``index_name`` 'time' / 't' gives the same p_timestep_weight as the silent-``x`` map — 0.5 at odd and 1.5
    at even steps (n = 48, Σ = 96) — and the same objective.  Before: the 'time' map was ignored (1.0)."""
    raw = _raw(48)
    runs = {}
    for label in (None, index_name):
        alt = f"tw_{label or 'x'}"
        url, s = make_db(tmp_path_factory, alt=alt, base_chain=COAL,
                         values=[["timeset", "2day", "timeset_weights", _weights_value(raw, label)]])
        runs[label] = _run(url, s, tmp_path_factory.mktemp(f"{alt}_w"))
    w_x = _weights(next(iter(runs[None].values())))
    w_l = _weights(next(iter(runs[index_name].values())))
    assert w_l == w_x
    assert w_x == pytest.approx(_expected(w_x, raw))
    assert sorted(set(w_x.values())) == pytest.approx([0.5, 1.5])
    obj = [float(next(iter(r.values())).solution.obj) for r in runs.values()]
    assert obj[1] == pytest.approx(obj[0], rel=1e-12)


@pytest.mark.parametrize("value", [
    pytest.param(b64({"index_type": "str", "index_name": "period", "data": [["p2020", 1.0]]}, "map"), id="period_map"),
    pytest.param(b64(2.0, "float"), id="scalar"),
])
def test_invalid_timeset_weights_shape_raises(tmp_path_factory, value) -> None:
    """A period map or a scalar is not a per-timestep weight map: both readers (the solve writer's timeline
    config and the DB-direct derivation) raise instead of reading it as weights of no timestep."""
    from flextool.engine.input._param_shapes import resolve_param_shape
    from flextool.engine.input._solve_state import FlexToolConfigError
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.input._timeline import TimelineConfig

    url, s = make_db(tmp_path_factory, alt="tw_bad", base_chain=COAL,
                     values=[["timeset", "2day", "timeset_weights", value]])
    with pytest.raises(FlexToolConfigError, match="timeset_weights"):
        TimelineConfig.load_from_db_url(url, s)
    with pytest.raises(FlexToolConfigError, match="timeset_weights"):
        resolve_param_shape(SpineDbReader(url, s), "timeset", "timeset_weights")
