"""Stochastic values in branch periods after the fan's period (item 228).

A forecast is authored as a ``(branch, analysis_time, time)`` map.  A solve
fans once, at its first stochastic step (``solve_data/branch_time_start``).
Every fold reads, per value cell, the forecast authored at the branch
period's own first step, else the forecast authored at the fan start, so a
model that authors its forecasts only at the fan's analysis time reads them
in every later period too (before: 0 there).  In a roll of a rolling solve
a look-ahead period reads the roll's fan-start forecast only: a cell that
forecast does not cover, where a forecast made later (at the period's own
first step) has a value, is an input error -- the roll's forecast must
cover its whole horizon.  A pre-reveal trunk period never reads the fan
start.

Fixture (built from the schema JSON, never a checked-in sqlite): one node
``s`` (constant demand 1 per step unless the case sets a stochastic inflow),
one unit ``src`` (10 MW, upper-limit profile ``prof``), steps t0001..t(4·NP),
period ``p<k>`` = t(4k-3)..t(4k), one step per hour.  Hand values (from the
design, independent of the code): the authored series is ``0.1 · i`` at step
``t000i``; branch ``high`` multiplies it by 1, ``low`` by 0.5.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

import flextool
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.emit._period_params import (
    derive_pdtNodeInflow,
    derive_pdtNodeInflow_vectorized,
)
from flextool.engine.preprocess.emit._pdt_lookup import FanStarts, ts_tiers
from flextool.engine.preprocess.emit._reserve_sets import derive_pdtReserve_upDown_group
from flextool.engine.preprocess.emit._solve_writers import (
    branch_time_start_rows,
    derive_branch_time_start,
    project_branch_time_start_rows,
)

SCHEMA_JSON = Path(flextool.__file__).parent / "schemas" / "spinedb_schema.json"
_FACTOR = {"high": 1.0, "low": 0.5}
_counter = itertools.count()


def _m(d: dict, name: str) -> api.Map:
    return api.Map(list(d), list(d.values()), index_name=name)


def _steps(n: int) -> list[str]:
    return [f"t{i:04d}" for i in range(1, n + 1)]


def _series(n_steps: int, scale: float = 0.1, offset: int = 0) -> dict[str, float]:
    """``scale · i`` at step ``t000i`` for i = offset+1 .. n_steps."""
    return {t: round(scale * (i + 1), 10) for i, t in enumerate(_steps(n_steps)) if i >= offset}


def _stochastic(per_at: dict[str, dict[str, dict[str, float]]]) -> api.Map:
    """``{branch: {analysis_time: {time: value}}}`` -> the rank-3 Map."""
    return _m({b: _m({at: _m(ts, "time") for at, ts in ats.items()}, "analysis_time")
               for b, ats in per_at.items()}, "branch")


def _forecast(ats: dict[str, dict[str, float]], branches=("high", "low")) -> api.Map:
    """Every branch's forecast: the analysis-time series times the branch factor."""
    return _stochastic({b: {at: {t: v * _FACTOR[b] for t, v in ts.items()} for at, ts in ats.items()}
                        for b in branches})


def _run(tmp_path_factory, *, n_periods: int, fans: list[tuple[str, str, str, str, float]],
         profile=None, inflow=-1.0, unit_grouped: bool = True, node_grouped: bool = False,
         rolling: "tuple[float, float] | None" = None, extra_values=()):
    """Build the fixture DB and run it; return ``{solve name: OrchestrationStep}``.

    *fans* are ``stochastic_branches`` rows ``(period, branch, analysis_time, realized, weight)``.
    """
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp(f"fan{next(_counter)}")
    db = work / "t.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    url = f"sqlite:///{db.resolve().as_posix()}"
    steps = _steps(4 * n_periods)
    ent = [("model", "flextool"), ("timeline", "tl"), ("node", "s"), ("unit", "src"),
           ("unit__outputNode", ("src", "s")), ("profile", "prof"),
           ("unit__node__profile", ("src", "s", "prof")), ("group", "g"), ("solve", "sol")]
    if unit_grouped:
        ent.append(("group__unit", ("g", "src")))
    if node_grouped:
        ent.append(("group__node", ("g", "s")))
    ent += [("timeset", f"ts{k}") for k in range(1, n_periods + 1)]
    branches: dict = {}
    for period, branch, at, realized, weight in fans:
        branches.setdefault(period, {}).setdefault(branch, {})[at] = {realized: weight}
    branch_map = _m({p: _m({b: _m({at: _m(rw, "realized") for at, rw in ats.items()}, "analysis_time")
                            for b, ats in bs.items()}, "branch") for p, bs in branches.items()}, "period")
    values = [
        ("model", "flextool", "solves", api.Array(["sol"])),
        ("timeline", "tl", "timestep_duration", _m({t: 1.0 for t in steps}, "time")),
        ("node", "s", "node_type", "balance"), ("node", "s", "inflow", inflow),
        ("node", "s", "penalty_up", 1000.0), ("node", "s", "penalty_down", 1000.0),
        ("unit", "src", "existing", 10.0), ("unit", "src", "virtual_unitsize", 1.0),
        ("unit", "src", "efficiency", 1.0),
        ("unit__node__profile", ("src", "s", "prof"), "profile_method", "upper_limit"),
        ("profile", "prof", "profile", profile if profile is not None else _m(_series(len(steps)), "time")),
        ("group", "g", "include_stochastics", "yes"),
        ("solve", "sol", "solve_mode", "rolling_window" if rolling else "single_solve"),
        ("solve", "sol", "period_timeset", _m({f"p{k}": f"ts{k}" for k in range(1, n_periods + 1)}, "period")),
        ("solve", "sol", "realized_periods", api.Array([f"p{k}" for k in range(1, n_periods + 1)])),
        ("solve", "sol", "years_represented", _m({f"p{k}": 1.0 for k in range(1, n_periods + 1)}, "period")),
        ("solve", "sol", "stochastic_branches", branch_map),
        *extra_values,
    ]
    if rolling:
        values += [("solve", "sol", "rolling_solve_jump", rolling[0]),
                   ("solve", "sol", "rolling_solve_horizon", rolling[1])]
    for k in range(1, n_periods + 1):
        values += [("timeset", f"ts{k}", "timeline", "tl"),
                   ("timeset", f"ts{k}", "timeset_duration", _m({f"t{4 * k - 3:04d}": 4.0}, "time"))]
    with open_database_mapping(url) as m:
        _, err = api.import_data(
            m, alternatives=[("a", "")], scenarios=[("sc", False, "")],
            scenario_alternatives=[("sc", "a", None)], entities=ent,
            entity_alternatives=[("node", ("s",), "a", True), ("unit", ("src",), "a", True)],
            parameter_values=[(c, e, p, v, "a") for c, e, p, v in values])
        assert not err, err
        m.commit_session("fixture")
    return run_chain_from_db(url, "sc", work_folder=work / "w", keep_solutions=True)


def _cells(step, key: str) -> dict[tuple[str, str], float]:
    """``(period, time) -> value`` of a single-entity ``solve_data`` frame (pdtProfile / pdtNodeInflow)."""
    df = step.flex_data_provider.get(f"solve_data/{key}")
    assert df is not None, key
    return {(str(d), str(t)): float(v) for d, t, v in df.select(df.columns[1:4]).iter_rows()}


def _profile_value(step) -> dict[tuple[str, str], float]:
    """``(d, t) -> value`` of the model's profile parameter (``_profile_stochastic_lf`` path)."""
    v = step.flex_data.p_profile_value
    df = v.frame if hasattr(v, "frame") else v
    df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns if c != "value"])
    return {(r["d"], r["t"]): float(r["value"]) for r in df.iter_rows(named=True)}


def _period(cells: dict, d: str) -> list[float]:
    """The values of period *d* (``p<k>`` or ``p<k>_<branch>``) at its own steps t(4k-3)..t(4k)."""
    k = int(d.split("_")[0][1:])
    own = {f"t{i:04d}" for i in range(4 * k - 3, 4 * k + 1)}
    return [v for (dd, t), v in sorted(cells.items()) if dd == d and t in own]


def _bts(step) -> "dict[str, tuple[str, str]]":
    """``period -> (fan start step, read_first)`` of the solve's ``branch_time_start``."""
    df = step.flex_data_provider.get("solve_data/branch_time_start")
    return {d: (s, r) for d, s, r in df.select("period", "step", "read_first").iter_rows()}


def _two_forecasts(period: str, at: str, realized=None) -> list[tuple]:
    rows = [(period, "high", at, "no", 0.5), (period, "low", at, "no", 0.5)]
    if realized is not None:
        rows.append((period, realized, at, "yes", 0.0))
    return rows


# ── 1: forecast authored only at the fan's analysis time ───────────────
def test_later_branch_period_reads_fan_start_forecast(tmp_path_factory) -> None:
    """Fan p1/t0001, data at t0001 only: p2_high reads 0.5..0.8 and p2_low 0.25..0.40 (was 0); a grouped
    stochastic inflow (-0.01·i in both forecasts) sums to -0.26 over p2 in each forecast (was 0.0)."""
    n = 8
    infl = _stochastic({b: {"t0001": {t: -0.01 * (i + 1) for i, t in enumerate(_steps(n))}}
                        for b in ("high", "low")})
    steps = _run(tmp_path_factory, n_periods=2, fans=_two_forecasts("p1", "t0001"),
                 profile=_forecast({"t0001": _series(n)}), inflow=infl, node_grouped=True)
    s = steps["sol"]
    for cells in (_cells(s, "pdtProfile"), _profile_value(s)):
        assert _period(cells, "p2_high") == pytest.approx([0.5, 0.6, 0.7, 0.8])
        assert _period(cells, "p2_low") == pytest.approx([0.25, 0.30, 0.35, 0.40])
        assert _period(cells, "p1_high") == pytest.approx([0.1, 0.2, 0.3, 0.4])
    inflow = _cells(s, "pdtNodeInflow")
    assert sum(_period(inflow, "p2_high")) == pytest.approx(-0.26)
    assert sum(_period(inflow, "p2_low")) == pytest.approx(-0.26)
    flows = s.solution.value("v_flow").with_columns(pl.col("d").cast(pl.Utf8))
    assert flows.filter(pl.col("d") == "p2_high")["value"].sum() > 0.0
    assert _bts(s) == {d: ("t0001", "own") for d in ("p1", "p1_high", "p1_low", "p2", "p2_high", "p2_low")}


# ── 2: a later fan; the pre-reveal trunk never reads the fan start ─────
def test_mid_horizon_fan_reads_its_own_start(tmp_path_factory) -> None:
    """NP=3, fan p2/t0005; data at t0001 (0.1·i) and at t0005 (0.01·i).  p3_x reads the t0005 series at
    t0009..t0012 (high 0.09..0.12, low 0.045..0.06); p2_x its own t0005 series; the pre-reveal trunk p1
    is unchanged (the reference forecast high at its own t0001: 0.1..0.4) and has no fan-start row."""
    n = 12
    prof = _forecast({"t0001": _series(n), "t0005": _series(n, 0.01, offset=4)})
    steps = _run(tmp_path_factory, n_periods=3, fans=_two_forecasts("p2", "t0005"), profile=prof)
    s = steps["sol"]
    for cells in (_cells(s, "pdtProfile"), _profile_value(s)):
        assert _period(cells, "p1") == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert _period(cells, "p2_high") == pytest.approx([0.05, 0.06, 0.07, 0.08])
        assert _period(cells, "p3_high") == pytest.approx([0.09, 0.10, 0.11, 0.12])
        assert _period(cells, "p3_low") == pytest.approx([0.045, 0.05, 0.055, 0.06])
    bts = _bts(s)
    assert "p1" not in bts
    assert set(bts.values()) == {("t0005", "own")}


# ── 3: single solve — a forecast authored at the period's own first step wins ──
def test_own_first_step_overrides_fan_start(tmp_path_factory) -> None:
    """Single (non-rolling) solve.  Data at t0001 (0.1·i) plus a constant 0.9 at analysis time t0005 for
    t0005..t0008 in every branch: p2 reads 0.9 in every forecast (no double count with the fan start); the
    rolling look-ahead rule (test 4) does not apply, ``read_first`` is ``own``."""
    n = 8
    per = {b: {"t0001": {t: v * _FACTOR[b] for t, v in _series(n).items()},
               "t0005": {t: 0.9 for t in _steps(n)[4:]}} for b in ("high", "low")}
    steps = _run(tmp_path_factory, n_periods=2, fans=_two_forecasts("p1", "t0001"), profile=_stochastic(per))
    s = steps["sol"]
    for cells in (_cells(s, "pdtProfile"), _profile_value(s)):
        assert _period(cells, "p2_high") == pytest.approx([0.9] * 4)
        assert _period(cells, "p2_low") == pytest.approx([0.9] * 4)
        assert _period(cells, "p1_low") == pytest.approx([0.05, 0.10, 0.15, 0.20])
    assert {r for _s, r in _bts(s).values()} == {"own"}


# ── 4: rolling — a roll reads only its own fan-start forecast ──────────
def test_rolling_lookahead_reads_roll_fan_start(tmp_path_factory, monkeypatch) -> None:
    """Rolling, jump 8 h, horizon 12 h, NP=3: roll 1 = p1..p3 fanning at t0001, roll 2 = p3 fanning at
    t0009.  The t0001 forecast (0.1·i) covers t0001..t0012, the t0009 forecast (0.01·i) t0009..t0012.
    Roll 1's look-ahead p2 has no forecast at its own first step t0005, so it reads the roll's fan start
    t0001 (high 0.5..0.8, low 0.25..0.40); roll 2's p3 reads its own t0009 forecast (0.09..0.12).

    Roll 1's look-ahead p3 starts at t0009, roll 2's analysis time.  Roll 1 decides at t0001 and must not see
    the forecast made at t0009, so p3 reads roll 1's t0001 forecast first.  Hand derivation: the t0001 series
    is 0.1·i at step t000i, so at t0009..t0012 (i = 9..12) branch high (×1) reads 0.9, 1.0, 1.1, 1.2 and
    branch low (×0.5) 0.45, 0.50, 0.55, 0.60 -- not the t0009 series (0.09..0.12 / 0.045..0.06) that the
    single-solve rule (own first step first) would read.

    The recorder is set with ``raising=False`` and the values are checked before the recorded rows, so on code
    without ``emit_branch_time_start`` the test fails on the values (p2 read 0), not on the patch."""
    n = 12
    per = {b: {"t0001": {t: v * _FACTOR[b] for t, v in _series(n).items()},
               "t0009": {t: v * _FACTOR[b] for t, v in _series(n, 0.01, offset=8).items()}}
           for b in ("high", "low")}
    fans = [*_two_forecasts("p1", "t0001"), *_two_forecasts("p3", "t0009")]
    from flextool.engine.preprocess.emit import _solve_writers
    emitted: list[list[tuple[str, str]]] = []
    real_emit = getattr(_solve_writers, "emit_branch_time_start", None)

    def _record(rows, *, provider, fan_first=False):
        emitted.append((list(rows), fan_first))
        real_emit(rows, provider=provider, fan_first=fan_first)

    monkeypatch.setattr(_solve_writers, "emit_branch_time_start", _record, raising=False)
    steps = _run(tmp_path_factory, n_periods=3, fans=fans, profile=_stochastic(per), rolling=(8.0, 12.0))
    names = list(steps)
    assert len(names) == 2, names
    # The provider is shared by the rolls; the model parameters are per roll.
    c1, c2 = _profile_value(steps[names[0]]), _profile_value(steps[names[1]])
    assert _period(c1, "p1_high") == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert _period(c1, "p2_high") == pytest.approx([0.5, 0.6, 0.7, 0.8])
    assert _period(c1, "p2_low") == pytest.approx([0.25, 0.30, 0.35, 0.40])
    assert _period(c2, "p3_high") == pytest.approx([0.09, 0.10, 0.11, 0.12])
    # Roll 1's look-ahead p3 reads roll 1's fan-start (t0001) forecast, not roll 2's t0009 one.
    assert _period(c1, "p3_high") == pytest.approx([0.9, 1.0, 1.1, 1.2])
    assert _period(c1, "p3_low") == pytest.approx([0.45, 0.50, 0.55, 0.60])
    assert [(sorted({step for _d, step in rows}), ff) for rows, ff in emitted] == [(["t0001"], True),
                                                                                (["t0009"], True)]


# ── 4b: rolling — a roll's forecast must cover its look-ahead ──────────
def test_rolling_lookahead_not_covered_by_roll_forecast_is_an_error(tmp_path_factory) -> None:
    """Test 4's rolling solve, but the t0001 forecast covers only t0001..t0008: roll 1's look-ahead p3
    (t0009..t0012) has no value in roll 1's forecast, and only the t0009 forecast (made at roll 2's start)
    has values there.  Roll 1 must not read it: an input error naming the profile, roll 1 (``sol_roll_0``:
    rolls count from 0), the four steps of p3 in both forecast copies and the two analysis times -- raised
    while the roll's inputs are prepared, before it is solved (no result is returned)."""
    per = {b: {"t0001": {t: v * _FACTOR[b] for t, v in _series(8).items()},
               "t0009": {t: v * _FACTOR[b] for t, v in _series(12, 0.01, offset=8).items()}}
           for b in ("high", "low")}
    fans = [*_two_forecasts("p1", "t0001"), *_two_forecasts("p3", "t0009")]
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, n_periods=3, fans=fans, profile=_stochastic(per), rolling=(8.0, 12.0))
    assert str(err.value) == (
        "profile 'prof', parameter 'profile': roll 'sol_roll_0' of a rolling solve starts at time step 't0001' "
        "and reads the forecasts made then over its whole horizon, but those forecasts have no value at time "
        "steps 't0009', 't0010', 't0011', 't0012' (periods 'p3_high', 'p3_low').  Only the forecasts made "
        "later, at 't0009', have values there, and a roll must not use a forecast made after it starts.  Give "
        "the forecasts that start at 't0001' values for the whole horizon of the roll, look-ahead included.  "
        "Check which alternative of the scenario gives this value.")


# ── 4c: a later forecast cohort that no solve or roll fans out at (item 229) ──
_LATER_COHORT_HEAD = (
    "opens its stochastic fan at period 'p1' (analysis_time 't0001'). A solve or roll fans out only once, "
    "but its stochastic_branches also declare forecast branches later in its horizon: period 'p2' at "
    "analysis_time 't0005' (forecast branches ")


@pytest.mark.parametrize(("rolling", "fans", "expected"), [
    pytest.param(
        None,
        [*_two_forecasts("p1", "t0001"), ("p2", "x", "t0005", "no", 0.5), ("p2", "mid", "t0005", "no", 0.5)],
        "Solve 'sol' " + _LATER_COHORT_HEAD + "'mid', 'x'). They would be ignored without a message. The "
        "branches opened at 'p1' continue through every later period, and a forecast series given at a later "
        "period's first time step is read without these rows. Remove these rows, or put the later fan in a "
        "separate solve. Check stochastic_branches.",
        id="single"),
    pytest.param(
        (8.0, 12.0),
        [*_two_forecasts("p1", "t0001"), *_two_forecasts("p2", "t0005")],
        "Roll 'sol_roll_0' of rolling solve 'sol' " + _LATER_COHORT_HEAD + "'high', 'low'). They would be "
        "ignored without a message. Each roll fans out at the first reveal in its horizon, and no roll fans "
        "out at these. Remove these rows, or make a roll start there (rolling_solve_jump). Check "
        "stochastic_branches.",
        id="rolling"),
])
def test_later_forecast_cohort_no_solve_fans_at_is_an_error(tmp_path_factory, rolling, fans, expected) -> None:
    """A solve or roll fans out once, at its first reveal.  Single solve, NP=3: fan p1/t0001 (high/low) and
    forecasts x, mid declared at p2/t0005 -- before, they were dropped silently.  Rolling 8/12, NP=3: rolls
    start at t0001 and t0009; roll 0's horizon reaches p2/t0005, where no roll fans out.  Both stop before
    solving with a message naming the solve (roll), its fan point, the later cohort and its branches.  The
    valid case -- a later cohort that another roll fans out at -- is test 4 / 4b."""
    with pytest.raises(FlexToolConfigError) as err:
        _run(tmp_path_factory, n_periods=3, fans=fans, rolling=rolling)
    assert str(err.value) == expected


# ── 5: realized: yes — the realization sibling and ungrouped users ─────
def test_realized_branch_read_in_later_periods(tmp_path_factory) -> None:
    """Fan p1/t0001 with ``real`` (×0.8) designated realized: yes, data at t0001 only, unit ungrouped: the
    forecast copies p2_high / p2_low read the realized series at t0005..t0008 (0.40, 0.48, 0.56, 0.64), and
    so does the realization-dispatch sibling's p2."""
    n = 8
    per = {b: {"t0001": {t: v * f for t, v in _series(n).items()}}
           for b, f in (("high", 1.0), ("low", 0.5), ("real", 0.8))}
    steps = _run(tmp_path_factory, n_periods=2, fans=_two_forecasts("p1", "t0001", realized="real"),
                 profile=_stochastic(per), unit_grouped=False)
    real = [0.40, 0.48, 0.56, 0.64]
    parent = _cells(steps["sol"], "pdtProfile")
    assert _period(parent, "p2_high") == pytest.approx(real)
    assert _period(parent, "p2_low") == pytest.approx(real)
    sibling = steps["sol__realized_dispatch"]
    for cells in (_cells(sibling, "pdtProfile"), _profile_value(sibling)):
        assert _period(cells, "p2") == pytest.approx(real)
        assert _period(cells, "p1") == pytest.approx([0.08, 0.16, 0.24, 0.32])
    assert _bts(sibling) == {"p1": ("t0001", "own"), "p2": ("t0001", "own")}


# ── 6: ungrouped stochastic inflow in a forecasts-only fan ─────────────
def test_ungrouped_stochastic_inflow_without_realized_is_an_error(tmp_path_factory) -> None:
    """Node ``s`` is in no include_stochastics group, its inflow is stochastic and the fan has no
    realized: yes branch: a configuration error naming the node and inflow (was 0 even in p1)."""
    infl = _forecast({"t0001": {t: -0.5 for t in _steps(8)}})
    with pytest.raises(FlexToolConfigError, match=r"node 's', parameter 'inflow'.*include_stochastics") as err:
        _run(tmp_path_factory, n_periods=2, fans=_two_forecasts("p1", "t0001"), inflow=infl)
    message = str(err.value)
    assert "Add node 's' to a group that has include_stochastics" in message
    assert "reserv" not in message  # the hint fits the parameter: no reservation advice for an inflow


# ── coverage checks (own ∪ fan; inflow and reservation) ────────────────
def _fan_provider(pbt_name: str, pbt: pl.DataFrame, extra: "dict[str, pl.DataFrame]",
                  *, rolling: bool = False) -> FlexDataProvider:
    """A synthetic solve: fan p1/t1 into high/low, continuation p2 (first step t5); *rolling*: the solve is
    roll ``sol_roll_1`` of a rolling solve (``read_first`` = ``fan``)."""
    provider = FlexDataProvider()

    def put(key: str, df: pl.DataFrame) -> None:
        provider.put(key, df)

    put(f"solve_data/{pbt_name}", pbt)
    put("solve_data/first_timesteps", pl.DataFrame({
        "period": ["p1_high", "p1_low", "p2_high", "p2_low"], "step": ["t1", "t1", "t5", "t5"]}))
    put("solve_data/solve_branch__time_branch", pl.DataFrame({
        "period": ["p1_high", "p1_low", "p2_high", "p2_low"], "branch": ["high", "low", "high", "low"]}))
    put("solve_data/period__branch", pl.DataFrame({
        "period": ["p1", "p1", "p1", "p2", "p2", "p2"],
        "branch": ["p1", "p1_high", "p1_low", "p2", "p2_high", "p2_low"]}))
    put("solve_data/branch_time_start", pl.DataFrame({
        "period": ["p1", "p1_high", "p1_low", "p2", "p2_high", "p2_low"], "step": ["t1"] * 6,
        "read_first": ["fan" if rolling else "own"] * 6}))
    if rolling:
        put("solve_data/solve_current", pl.DataFrame({"solve": ["sol_roll_1"]}))
    put("solve_data/steps_in_use", pl.DataFrame({
        "period": ["p1_high", "p1_low", "p2_high", "p2_low"], "time": ["t1", "t1", "t5", "t5"]}))
    for k, v in extra.items():
        put(k, v)
    return provider


def test_inflow_partial_branch_coverage_is_an_error(tmp_path) -> None:
    """A stochastic inflow authoring only branch high of the fan is an input error, in both inflow twins."""
    pbt = pl.DataFrame({"node": ["s", "s"], "branch": ["high", "high"], "time_start": ["t1", "t1"],
                        "time": ["t1", "t5"], "value": [-1.0, -2.0]})
    provider = _fan_provider("pbt_node_inflow", pbt, {
        "input/node": pl.DataFrame({"node": ["s"]}),
        "input/group__node": pl.DataFrame({"group": ["g"], "node": ["s"]}),
        "input/groupIncludeStochastics": pl.DataFrame({"group": ["g"]}),
        "solve_data/node__inflow_method": pl.DataFrame({"node": ["s"], "method": ["use_original"]}),
    })
    for derive in (derive_pdtNodeInflow, derive_pdtNodeInflow_vectorized):
        with pytest.raises(FlexToolConfigError, match=r"node 's', parameter 'inflow'.*missing \['low'\]"):
            derive(tmp_path / "input", tmp_path / "solve_data", provider=provider)


def test_inflow_fan_start_fold_in_both_twins(tmp_path) -> None:
    """The fan-start fallback of the inflow fold, both twins: p2_high reads high's t1 forecast at t5 (-2.0),
    p2_low low's (-1.0)."""
    pbt = pl.DataFrame({"node": ["s"] * 4, "branch": ["high", "high", "low", "low"],
                        "time_start": ["t1"] * 4, "time": ["t1", "t5", "t1", "t5"],
                        "value": [-1.0, -2.0, -0.5, -1.0]})
    provider = _fan_provider("pbt_node_inflow", pbt, {
        "input/node": pl.DataFrame({"node": ["s"]}),
        "input/group__node": pl.DataFrame({"group": ["g"], "node": ["s"]}),
        "input/groupIncludeStochastics": pl.DataFrame({"group": ["g"]}),
        "solve_data/node__inflow_method": pl.DataFrame({"node": ["s"], "method": ["use_original"]}),
    })
    legacy = derive_pdtNodeInflow(tmp_path / "input", tmp_path / "solve_data", provider=provider)
    vec = derive_pdtNodeInflow_vectorized(tmp_path / "input", tmp_path / "solve_data", provider=provider)
    assert vec.equals(legacy)
    got = {(d, t): float(v) for _n, d, t, v in legacy.iter_rows()}
    assert got == {("p1_high", "t1"): -1.0, ("p1_low", "t1"): -0.5, ("p2_high", "t5"): -2.0,
                   ("p2_low", "t5"): -1.0}


def _reserve_provider(branches: list[str], values: list[float], *, time_starts: "list[str] | None" = None,
                      times: "list[str] | None" = None, rolling: bool = False) -> FlexDataProvider:
    n = len(branches)
    pbt = pl.DataFrame({"reserve": ["r"] * n, "upDown": ["up"] * n, "group": ["g"] * n,
                        "param": ["reservation"] * n, "branch": branches, "time_start": time_starts or ["t1"] * n,
                        "time": times or ["t5"] * n, "value": values})
    return _fan_provider("pbt_reserve__upDown__group", pbt, {
        "solve_data/reserve__upDown__group": pl.DataFrame({"reserve": ["r"], "upDown": ["up"], "group": ["g"]}),
        "input/groupIncludeStochastics": pl.DataFrame({"group": ["g"]}),
    }, rolling=rolling)


def test_reservation_partial_branch_coverage_is_an_error(tmp_path) -> None:
    """A stochastic reservation authoring only branch high of the fan is an input error."""
    provider = _reserve_provider(["high"], [3.0])
    with pytest.raises(FlexToolConfigError,
                       match=r"reserve 'r' \(up\) of group 'g', parameter 'reservation'.*missing \['low'\]"):
        derive_pdtReserve_upDown_group(tmp_path / "input", tmp_path / "solve_data", provider=provider)


def test_reservation_reads_fan_start(tmp_path) -> None:
    """A reservation forecast authored at the fan start t1 is read in p2 (own first step t5): 3.0 / 1.5."""
    provider = _reserve_provider(["high", "low"], [3.0, 1.5])
    df = derive_pdtReserve_upDown_group(tmp_path / "input", tmp_path / "solve_data", provider=provider)
    got = {(r[4], r[5]): float(r[6]) for r in df.iter_rows()}
    assert got[("p2_high", "t5")] == 3.0
    assert got[("p2_low", "t5")] == 1.5


# A reservation forecast made at the fan start t1 (for t1 only) and one made at p2's own first step t5 (for t5).
_LATER_ONLY = dict(branches=["high", "low", "high", "low"], values=[1.0, 0.5, 3.0, 1.5],
                   time_starts=["t1", "t1", "t5", "t5"], times=["t1", "t1", "t5", "t5"])


def test_reservation_roll_forecast_must_cover_lookahead(tmp_path) -> None:
    """A roll of a rolling solve: p2's step t5 has a value only in the forecast made at t5, later than the
    roll's start t1, so the roll may not read it -- an input error, worded for the user (hand-written)."""
    provider = _reserve_provider(**_LATER_ONLY, rolling=True)
    with pytest.raises(FlexToolConfigError) as err:
        derive_pdtReserve_upDown_group(tmp_path / "input", tmp_path / "solve_data", provider=provider)
    assert str(err.value) == (
        "reserve 'r' (up) of group 'g', parameter 'reservation': roll 'sol_roll_1' of a rolling solve starts at "
        "time step 't1' and reads the forecasts made then over its whole horizon, but those forecasts have no "
        "value at time step 't5' (periods 'p2_high', 'p2_low').  Only the forecasts made later, at 't5', have "
        "values there, and a roll must not use a forecast made after it starts.  Give the forecasts that start "
        "at 't1' values for the whole horizon of the roll, look-ahead included.  Check which alternative of the "
        "scenario gives this value.")


def test_reservation_single_solve_keeps_own_step_fallback(tmp_path) -> None:
    """The same data in a single solve is fine: p2 reads its own first step's forecast, 3.0 / 1.5."""
    df = derive_pdtReserve_upDown_group(tmp_path / "input", tmp_path / "solve_data",
                                        provider=_reserve_provider(**_LATER_ONLY))
    got = {(r[4], r[5]): float(r[6]) for r in df.iter_rows()}
    assert got[("p2_high", "t5")] == 3.0
    assert got[("p2_low", "t5")] == 1.5


def test_reservation_roll_forecast_covering_lookahead_is_read(tmp_path) -> None:
    """A roll whose forecast made at t1 also covers t5 (3.0 / 1.5) is silent and reads it there, not the t5
    forecast (30.0 / 15.0)."""
    provider = _reserve_provider(["high", "low", "high", "low"], [3.0, 1.5, 30.0, 15.0],
                                 time_starts=["t1", "t1", "t5", "t5"], times=["t5"] * 4, rolling=True)
    df = derive_pdtReserve_upDown_group(tmp_path / "input", tmp_path / "solve_data", provider=provider)
    got = {(r[4], r[5]): float(r[6]) for r in df.iter_rows()}
    assert got[("p2_high", "t5")] == 3.0
    assert got[("p2_low", "t5")] == 1.5


def test_ts_tiers_order() -> None:
    """Own first step first, then the fan start; a roll's look-ahead period (``fan_first``) the other way
    round; the fan's own period (own step = fan start) and a period label read one tier."""
    ts_for_d = {"p1_b": ["t1"], "p2_b": ["t5"], "p2": []}
    fan = {"p1_b": ["t1"], "p2_b": ["t1"], "p2": ["t1"]}
    assert ts_tiers("p2_b", ts_for_d, fan) == [["t5"], ["t1"]]
    assert ts_tiers("p2_b", ts_for_d, FanStarts(fan)) == [["t5"], ["t1"]]
    rolling = FanStarts(fan, fan_first=frozenset(fan))
    assert ts_tiers("p2_b", ts_for_d, rolling) == [["t1"], ["t5"]]
    assert ts_tiers("p1_b", ts_for_d, rolling) == [["t1"]]
    assert ts_tiers("p2", ts_for_d, rolling) == [[]]


# ── the frame itself ───────────────────────────────────────────────────
def test_branch_time_start_rows() -> None:
    """Rows for every anchor from the fan's period onwards and each copy; none before; none without a fan;
    the realization sibling keeps the real period names."""
    pb = [("p1", "p1"), ("p2", "p2"), ("p2", "p2_high"), ("p2", "p2_low"),
          ("p3", "p3"), ("p3", "p3_high"), ("p3", "p3_low")]
    rows = branch_time_start_rows(pb, ("p2", "t0005"))
    assert rows == [(d, "t0005") for d in ("p2", "p2_high", "p2_low", "p3", "p3_high", "p3_low")]
    assert branch_time_start_rows(pb, None) == []
    assert project_branch_time_start_rows(rows, pb) == [("p2", "t0005"), ("p3", "t0005")]
    frame = derive_branch_time_start(rows[:1], fan_first=True)
    assert frame.columns == ["period", "step", "read_first"]
    assert frame.rows() == [("p2", "t0005", "fan")]
    assert derive_branch_time_start(rows[:1]).rows() == [("p2", "t0005", "own")]
