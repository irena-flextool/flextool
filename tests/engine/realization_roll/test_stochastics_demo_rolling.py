"""The rolling dispatch example of ``docs/how_to/stochastic_examples.md``: a ``deterministic``, a ``within_solve`` and
a ``separate_dispatch`` scenario of the same rolling solve, and the same two methods with a perfect forecast.

Fixture ``tests/fixtures/stochastics.json``, the ``rolling_*`` scenarios (``build_stochastics_demo.py``, the model in
its docstring): three periods ``period1``-``period3`` of 72 one-hour steps (t0001-t0216); one rolling solve
``rolling_dispatch``, jump 24 h, horizon 48 h, so 9 rolls: roll k realizes the jump ``[s_k, s_k + 24)`` with s_k =
24k (step index; step t{s_k + 1}), looks ahead to ``s_k + 48`` (cut at the last step: the horizon of roll 8 ends
with its jump).  Rolls 2 and 5 look ahead into the next period.  Wind ``wind_cf``: the actual series ``actual``
(``realized: yes``, given at each roll start for its jump) and the forecasts ``high`` / ``low`` (weight 0.5 / 0.5)
made at each roll start for 48 h; in the ``perfect_forecast`` alternative one forecast ``exact`` (weight 1) equal to
the actual series.

What each method does (the ``dispatch_realization_method`` docs, ``docs/how_to/stochastics.md``):

* ``deterministic``: the forecasts are ignored; every roll reads the actual series as one timeline, jump and
  look-ahead (note W-0593).  One solve per roll.
* ``within_solve``: one solve per roll.  The jump is dispatched once, under the period's own name, on the actual
  series; from ``s_{k+1}`` on each forecast made at ``s_{k+1}`` looks ahead in its own branch period ``<period>_<f>``,
  continuing from the last jump step.  Nothing is tied across the forecasts (no non-anticipativity rows): the jump is
  one decision.  The last roll (no look-ahead) is the jump only, on the actual series.
* ``separate_dispatch``: two solves per roll, ``rolling_dispatch_roll_k`` then
  ``rolling_dispatch__realized_dispatch_roll_k``.  (1) fully stochastic from the roll's first step on the forecasts
  made at ``s_k``, tying across the forecasts only the units online of the ``_fix_in_uc`` unit (``coal_units``) over
  the jump; (2) the ``within_solve`` roll with ``coal_units``'s units online over the jump fixed to (1)'s.  The next
  roll continues from (2).

Cost of a cell (d, t), EUR per hour, from the fixture.  Fuel of a ``min_load_efficiency`` unit (efficiency eta1 at
full load, eta0 at min load L; ``docs/reference/units.md``): a line from (L x C, L x C / eta0) to (C, C / eta1) in
(output, input) with C the online capacity, so input = s x output + n x C, s = (1 / eta1 - L / eta0) / (1 - L),
n = 1 / eta1 - s: coal (0.4, 0.35, 0.4) s = 2.261905, n = 0.238095, at 12 EUR/MWh; peaker (0.35, 0.3, 0.2) s =
2.738095, n = 0.119048, at 28 EUR/MWh.  Start-ups 50 EUR/MW x 100 MW per coal unit, 5 EUR/MW of the 350 MW
peaker; the inverter 0.5 EUR/MWh in either direction.  ``v_flow`` and ``v_online`` are per unit size (wind 650,
coal 100, peaker 350, inverter 100 MW).  An objective is sum over cells of w(d, t) x 8760 / 72 x cost
(``years_represented`` 1, no discounting), w = 1 on a period's own cells, the forecast weight on a branch period's
cells.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm

pytestmark = pytest.mark.solver

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "stochastics.json"
SOLVE = "rolling_dispatch"
UC = "rolling_dispatch_roll_{k}"
DISP = "rolling_dispatch__realized_dispatch_roll_{k}"
JUMP, HORIZON = 24, 48
PERIODS = ("period1", "period2", "period3")
FORECASTS = ("high", "low")
WEIGHT = {"high": 0.5, "low": 0.5, "exact": 1.0}
#: 8760 h / the 72 h of a period (``years_represented`` 1).
FACTOR = 8760 / 72
UNITSIZE = {"wind_farm": 650.0, "coal_units": 100.0, "gas_peaker": 350.0, "battery_inverter": 100.0}
#: ``(full-load efficiency, efficiency at min load, min load, fuel price)`` of the min_load_efficiency units.
FUEL = {"coal_units": (0.4, 0.35, 0.4, 12.0), "gas_peaker": (0.35, 0.3, 0.2, 28.0)}
EUR_PER_MWH = {"battery_inverter": 0.5, "wind_farm": 0.0}       # other flows: variable cost per MWh
START_EUR = {"coal_units": 50.0 * 100.0, "gas_peaker": 5.0 * 350.0}     # per unit of v_startup
MIN_UPTIME = 8
SCENARIOS = ("rolling_deterministic", "rolling_within_solve", "rolling_separate_dispatch",
             "rolling_perfect_within_solve", "rolling_perfect_separate_dispatch")


# ---------------------------------------------------------------------------
# The fixture data, read from the JSON
# ---------------------------------------------------------------------------

def _unmap(value):
    if isinstance(value, dict) and "data" in value:
        return {k: _unmap(v) for k, v in value["data"]}
    return value


def _param(cls: str, name, par: str, alternative: str = "rolling_demo"):
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (value,) = [row[3] for row in raw["parameter_values"] if row[:3] == [cls, name, par] and row[4] == alternative]
    return _unmap(json.loads(base64.b64decode(value[0])))


WIND = _param("profile", "wind_cf", "profile")                       # {branch: {analysis time: {t: cf}}}
EXACT = _param("profile", "wind_cf", "profile", "perfect_forecast")
STEPS = list(_param("timeline", "hourly", "timestep_duration"))
N = len(STEPS)
N_ROLLS = -(-N // JUMP)


def period_of(i: int) -> str:
    return PERIODS[i // 72]


def actual(i: int) -> float:
    """R at step index *i*: the actual series given at the start of the jump that holds *i*."""
    return WIND["actual"][STEPS[i - i % JUMP]][STEPS[i]]


def made_at(branch: str, a: int, i: int, data=WIND) -> float:
    return data[branch][STEPS[a]][STEPS[i]]


def jump(k: int) -> range:
    return range(JUMP * k, min(JUMP * (k + 1), N))


def tail(k: int) -> range:
    return range(JUMP * (k + 1), min(JUMP * k + HORIZON, N))


def horizon(k: int) -> range:
    return range(JUMP * k, min(JUMP * k + HORIZON, N))


def cell(i: int, branch: str | None = None) -> tuple[str, str]:
    return (period_of(i) if branch is None else f"{period_of(i)}_{branch}", STEPS[i])


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``{scenario: (MPS of each build by solve name, steps, coded warning codes, output CSV folder)}``."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs
    from tests._support.coded_warnings import capture_coded

    folder = tmp_path_factory.mktemp("stochastics_demo_rolling")
    url = scm.build_db(FIXTURE, folder)
    out = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
        for scenario in SCENARIOS:
            work = folder / scenario
            work.mkdir()
            mp.chdir(work)
            with capture_coded() as got:
                built, steps = scm.run_capturing(url, scenario, work, csv_dump=True)
            last = next(reversed(steps.values()))
            write_outputs(
                scenario_name=scenario, output_location=str(work), subdir=scenario,
                output_config_path=str(scm.REPO / "flextool" / "schemas" / "default_plots.yaml"),
                write_methods=["csv"], fallback_output_location=str(work), raw_output_dir=str(work / "w" / "output_raw"),
                solution=last.solution, solve_name=last.solve_name,
                solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
                flex_data_provider=FlexDataProvider())
            assert len(built) == len(steps), scenario      # one build per solve, in run order
            mps = {name: path for name, (_s, path, _o) in zip(steps, built)}
            out[scenario] = (mps, steps, got, work / "output_csv" / scenario)
    return out


def _frame(x) -> pl.DataFrame:
    return x if isinstance(x, pl.DataFrame) else x.frame


def _str(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col(c).cast(pl.Utf8) for c in frame.columns if c != "value")


def _cells(step, var: str, **keys) -> dict[tuple[str, str], float]:
    """``{(d, t): value}`` of *var* in *step*'s solution, filtered on *keys*."""
    frame = _str(step.solution.value(var)).filter(*(pl.col(k) == v for k, v in keys.items()))
    return {(d, t): v for d, t, v in frame.select("d", "t", "value").rows()}


def _families(mps: Path) -> dict[str, list[str]]:
    lps = scm.lp_snapshot()
    out: dict[str, list[str]] = {}
    for row in lps.parse_mps(mps).rows:
        out.setdefault(lps.family_of(row), []).append(row)
    return out


def _solve_names(scenario: str) -> list[str]:
    if scenario.endswith("separate_dispatch"):
        return [name for k in range(N_ROLLS) for name in (UC.format(k=k), DISP.format(k=k))]
    return [UC.format(k=k) for k in range(N_ROLLS)]


def _realized_name(scenario: str, k: int) -> str:
    return (DISP if scenario.endswith("separate_dispatch") else UC).format(k=k)


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------

def test_solve_lists_and_status(runs):
    """deterministic and within_solve: one solve per roll (9); separate_dispatch: the stochastic solve and the
    realization solve of each roll, interleaved (18).  Every solve is optimal."""
    for scenario in SCENARIOS:
        _mps, steps, _codes, _csv = runs[scenario]
        assert list(steps) == _solve_names(scenario), scenario
        assert all(s.optimal for s in steps.values()), scenario


# ---------------------------------------------------------------------------
# What each solve reads (the wind capacity factor of each cell)
# ---------------------------------------------------------------------------

def _wind(step) -> dict[tuple[str, str], float]:
    """The wind capacity factor of each cell the solve has (``dt``)."""
    cells = set(_str(_frame(step.flex_data.dt)).select("d", "t").rows())
    frame = _str(_frame(step.flex_data.p_profile_value)).filter(pl.col("f") == "wind_cf")
    return {(d, t): v for d, t, v in frame.select("d", "t", "value").rows() if (d, t) in cells}


def _realization_roll_reads(k: int) -> dict[tuple[str, str], float]:
    """A realization roll: the jump under the period's own name on R, the tail of each forecast made at s_{k+1}."""
    want = {cell(i): actual(i) for i in jump(k)}
    for f in FORECASTS:
        want.update({cell(i, f): made_at(f, JUMP * (k + 1), i) for i in tail(k)})
    return want


def test_deterministic_rolls_read_the_realized_series_as_one_timeline(runs):
    """Every cell of every roll, look-ahead included, is the period's own and reads R (the series given at the start
    of the jump that holds the step: made after the roll's start for the look-ahead)."""
    _mps, steps, _codes, _csv = runs["rolling_deterministic"]
    for k in range(N_ROLLS):
        assert _wind(steps[UC.format(k=k)]) == {cell(i): actual(i) for i in horizon(k)}, k


def test_within_solve_realizes_the_jump_and_looks_ahead_with_the_forecasts_made_at_its_end(runs):
    _mps, steps, _codes, _csv = runs["rolling_within_solve"]
    for k in range(N_ROLLS):
        assert _wind(steps[UC.format(k=k)]) == _realization_roll_reads(k), k
    assert not tail(N_ROLLS - 1)          # the last roll is its jump only


def test_separate_dispatch_reads(runs):
    """Solve (1) of roll k: every cell of its horizon in each forecast's branch period, on the forecast made at s_k
    (also the look-ahead into the next period of rolls 2 and 5).  The realization solve reads as the within_solve
    roll."""
    _mps, steps, _codes, _csv = runs["rolling_separate_dispatch"]
    for k in range(N_ROLLS):
        want = {cell(i, f): made_at(f, JUMP * k, i) for f in FORECASTS for i in horizon(k)}
        assert _wind(steps[UC.format(k=k)]) == want, k
        assert _wind(steps[DISP.format(k=k)]) == _realization_roll_reads(k), k


# ---------------------------------------------------------------------------
# Non-anticipativity and the pins
# ---------------------------------------------------------------------------

def test_within_solve_jump_is_one_decision(runs):
    """Every variable of a within_solve roll has its jump cells only under the period's own name and its tail cells
    only under the forecasts' branch periods: one decision over the jump, the same for every forecast.  The LP has
    no non-anticipativity and no pin rows (nothing to tie)."""
    mps, steps, _codes, _csv = runs["rolling_within_solve"]
    for k in range(N_ROLLS):
        step = steps[UC.format(k=k)]
        allowed = {cell(i) for i in jump(k)} | {cell(i, f) for f in FORECASTS for i in tail(k)}
        for var in ("v_flow", "v_online_integer", "v_online_linear", "v_startup_integer", "v_state"):
            got = set(_str(step.solution.value(var)).select("d", "t").unique().rows())
            assert got == allowed, (k, var, sorted(got ^ allowed)[:4])
        fams = _families(mps[UC.format(k=k)])
        assert not [f for f in fams if f.startswith(("non_anticipativity", "online_fix_eq"))], (k, list(fams))


UNTIED = ("non_anticipativity_online_linear", "non_anticipativity_reserve", "non_anticipativity_storage_use",
          "non_anticipativity_storage_state_first", "non_anticipativity_rp_block_start")


def test_separate_dispatch_solve_one_ties_only_the_fix_in_uc_unit_and_the_realization_keeps_it(runs):
    """Solve (1) of each roll: ``non_anticipativity_online_integer`` on ``coal_units`` only, one row per jump step
    for the one forecast pair (high, low): 24 rows; no tie of the peaker's (linear) units online, of the reserves or
    of the battery.  coal_units's units online are equal in both forecasts over the jump.  The realization solve:
    24 pins ``online_fix_eq_integer[coal_units, ...]`` (the jump), and coal_units's units online over the jump are
    solve (1)'s."""
    mps, steps, _codes, _csv = runs["rolling_separate_dispatch"]
    for k in range(N_ROLLS):
        uc, disp = steps[UC.format(k=k)], steps[DISP.format(k=k)]
        fams = _families(mps[UC.format(k=k)])
        tied = fams.get("non_anticipativity_online_integer", [])
        assert len(tied) == len(jump(k)) and all(r.startswith("non_anticipativity_online_integer[coal_units,")
                                                 for r in tied), (k, tied[:3])
        assert not [f for f in UNTIED if f in fams], k
        coal = _cells(uc, "v_online_integer", p="coal_units")
        plan = {}
        for i in jump(k):
            high, low = coal[cell(i, "high")], coal[cell(i, "low")]
            assert high == pytest.approx(low, abs=1e-6), (k, STEPS[i], high, low)
            plan[cell(i)] = high
        pins = _families(mps[DISP.format(k=k)]).get("online_fix_eq_integer", [])
        assert len(pins) == len(jump(k)) and all(r.startswith("online_fix_eq_integer[coal_units,") for r in pins), k
        realized = _cells(disp, "v_online_integer", p="coal_units")
        assert {c: realized[c] for c in plan} == pytest.approx(plan, abs=1e-6), k


# ---------------------------------------------------------------------------
# Carry from roll to roll
# ---------------------------------------------------------------------------

def _first(frame, key: str, name: str, col: str = "value") -> float:
    rows = _frame(frame).filter(pl.col(key).cast(pl.Utf8) == name)
    assert rows.height == 1, rows
    return float(rows[col][0])


@pytest.mark.parametrize("scenario", ["rolling_deterministic", "rolling_within_solve", "rolling_separate_dispatch"])
def test_storage_and_units_online_carry_from_the_realized_jump(runs, scenario):
    """Roll k + 1 (with separate_dispatch: both of its solves) starts from the realized end of roll k's jump (its
    realization solve): the battery state and coal_units's units online at step s_{k+1} - 1 of the realized solve.
    The battery's state is per its 400 MWh size in ``v_state``."""
    _mps, steps, _codes, _csv = runs[scenario]
    for k in range(N_ROLLS - 1):
        prev = steps[_realized_name(scenario, k)]
        last = cell(JUMP * (k + 1) - 1)
        state = _cells(prev, "v_state", n="battery")[last] * 400.0
        online = _cells(prev, "v_online_integer", p="coal_units")[last]
        nxt = [steps[UC.format(k=k + 1)]] + ([steps[DISP.format(k=k + 1)]]
                                             if scenario == "rolling_separate_dispatch" else [])
        for step in nxt:
            fd = step.flex_data
            assert _first(fd.p_roll_continue_state, "n", "battery") == pytest.approx(state, abs=1e-6), (k, step.solve_name)
            assert _first(fd.p_roll_continue_online, "p", "coal_units") == pytest.approx(online, abs=1e-6), k


@pytest.mark.parametrize("scenario", ["rolling_within_solve", "rolling_separate_dispatch"])
def test_min_uptime_carry_from_the_realized_start_ups(runs, scenario):
    """The minimum up time (8 h) of the start-ups realized in the last hours of roll k's jump reaches the first
    steps of roll k + 1: carry[x] = sum of the realized start-ups at steps h of roll k with x - h < 8, for x =
    s_{k+1} .. s_{k+1} + 7, in every period of roll k + 1 that holds these steps (separate_dispatch solve (1): each
    forecast's branch period; a realization roll: the period itself, its tails start 24 h later).  At least one
    seam must carry a start-up, else the check is empty."""
    _mps, steps, _codes, _csv = runs[scenario]
    carried = 0
    for k in range(N_ROLLS - 1):
        su = {t: v for (d, t), v in _cells(steps[_realized_name(scenario, k)], "v_startup_integer",
                                           p="coal_units").items() if d == period_of(STEPS.index(t))}
        first = JUMP * (k + 1)
        expect = {}
        for x in range(first, first + MIN_UPTIME):
            v = sum(s for t, s in su.items() if x - STEPS.index(t) < MIN_UPTIME)
            if abs(v) > 1e-9:
                expect[STEPS[x]] = v
        names = [UC.format(k=k + 1)] + ([DISP.format(k=k + 1)] if scenario == "rolling_separate_dispatch" else [])
        for name in names:
            stochastic = scenario == "rolling_separate_dispatch" and name == UC.format(k=k + 1)
            periods = ([cell(first, f)[0] for f in FORECASTS] if stochastic else [period_of(first)])
            carry = steps[name].flex_data.uptime_carry
            got = {} if carry is None else {
                (d, t): v for d, t, v in _str(_frame(carry)).filter(pl.col("p") == "coal_units")
                .select("d", "t", "value").rows()}
            want = {(d, t): v for d in periods for t, v in expect.items()}
            assert got == pytest.approx(want, abs=1e-9), (name, got, want)
            carried += bool(want)
    assert carried > 0


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def _cost_by_cell(step) -> dict[tuple[str, str], float]:
    """EUR per hour of each cell of *step* from its solution (the rates in the module docstring)."""
    out: dict[tuple[str, str], float] = {}

    def add(c, v):
        out[c] = out.get(c, 0.0) + v

    online = {p: _cells(step, var, p=p) for p, var in (("coal_units", "v_online_integer"),
                                                       ("gas_peaker", "v_online_linear"))}
    flow = _str(step.solution.value("v_flow"))
    for p, d, t, v in flow.select("p", "d", "t", "value").rows():
        if p in FUEL:
            eta1, eta0, low, price = FUEL[p]
            s = (1 / eta1 - low / eta0) / (1 - low)
            add((d, t), price * UNITSIZE[p] * (s * v + (1 / eta1 - s) * online[p][(d, t)]))
        else:
            add((d, t), v * UNITSIZE[p] * EUR_PER_MWH[p])
    for var, p in (("v_startup_integer", "coal_units"), ("v_startup_linear", "gas_peaker")):
        for c, v in _cells(step, var, p=p).items():
            add(c, v * START_EUR[p])
    return out


def _weight(d: str) -> float:
    return WEIGHT[d.split("_", 1)[1]] if "_" in d else 1.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_objective_is_the_weighted_cost_of_its_cells(runs, scenario):
    """Each solve's objective = sum over its cells of w(d, t) x 8760 / 72 x cost(d, t), w = 1 on a period's own
    cells and the forecast weight on a branch period's (0.5, or 1 for the perfect forecast).  A cell's cost includes
    its unserved and excess energy at 3000 EUR/MWh: a low forecast late in a stochastic solve's horizon may leave a
    few MW unserved (demand up to 702 MW against 650 MW of coal and peaker, an almost calm forecast and a battery
    that must end the horizon at half full), but a realized cell never does."""
    _mps, steps, _codes, _csv = runs[scenario]
    for name, step in steps.items():
        k = int(name.rsplit("_", 1)[1])
        realized = {cell(i) for i in jump(k)} if name == _realized_name(scenario, k) else set()
        slack = {}
        for var in ("vq_state_up", "vq_state_down"):
            frame = _str(step.solution.value(var))
            assert frame.filter((pl.col("n") != "elec") & (pl.col("value").abs() > 1e-9)).height == 0, (name, var)
            for d, t, v in frame.filter(pl.col("n") == "elec").select("d", "t", "value").rows():
                slack[(d, t)] = slack.get((d, t), 0.0) + v
        assert all(abs(slack.get(c, 0.0)) < 1e-9 for c in realized), name
        costs = _cost_by_cell(step)
        want = sum(_weight(d) * FACTOR * (c + 3000.0 * slack.get((d, t), 0.0)) for (d, t), c in costs.items())
        assert step.solution.obj == pytest.approx(want, rel=1e-7), name


def _read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=0)


@pytest.mark.parametrize("scenario", ["rolling_deterministic", "rolling_within_solve", "rolling_separate_dispatch"])
def test_reported_costs_are_the_realized_jumps(runs, scenario):
    """The results report, per time step, the cost of the realized jump cell (from the realization solve with
    separate_dispatch), and per period the annualized sum of those cells (x 8760 / 72 / 1e6, M EUR)."""
    _mps, steps, _codes, csv = runs[scenario]
    want = {}
    for k in range(N_ROLLS):
        costs = _cost_by_cell(steps[_realized_name(scenario, k)])
        want.update({cell(i): costs[cell(i)] for i in jump(k)})
    dt = _read_csv(csv / "costs__dt.csv")
    cols = [c for c in dt.columns if c not in ("solve", "period", "time")]
    got = {(r["period"], r["time"]): sum(float(r[c]) for c in cols) for r in dt.iter_rows(named=True)}
    assert got.keys() == want.keys()
    assert got == pytest.approx(want, rel=1e-6, abs=1e-3)
    by_period = _read_csv(csv / "annualized_costs__d.csv")
    cols = [c for c in by_period.columns if c not in ("solve", "period")]
    for r in by_period.iter_rows(named=True):
        total = sum(v for (d, _t), v in want.items() if d == r["period"]) * FACTOR / 1e6
        assert sum(float(r[c]) for c in cols) == pytest.approx(total, rel=1e-6), r["period"]


# ---------------------------------------------------------------------------
# Perfect forecast: the stochastic methods reduce to the deterministic run
# ---------------------------------------------------------------------------

def test_perfect_forecast_is_the_actual_series():
    """The fixture's sanity case: the one forecast ``exact`` made at each roll start equals R over its 48 h."""
    for a in range(0, N, JUMP):
        for i in horizon(a // JUMP):
            assert made_at("exact", a, i, EXACT) == actual(i)


@pytest.mark.parametrize("scenario", ["rolling_perfect_within_solve", "rolling_perfect_separate_dispatch"])
def test_perfect_forecast_matches_the_deterministic_run(runs, scenario):
    """With one forecast of weight 1 equal to R, every solve of the roll optimises the deterministic roll's problem:
    the same objective per roll (solve (1) and the realization solve of separate_dispatch too), and the same
    realized battery state and coal units online over the jump."""
    _mps, det, _codes, _csv = runs["rolling_deterministic"]
    _mps, steps, _codes, _csv = runs[scenario]
    for k in range(N_ROLLS):
        ref = det[UC.format(k=k)]
        names = [UC.format(k=k)] + ([DISP.format(k=k)] if scenario.endswith("separate_dispatch") else [])
        for name in names:
            assert steps[name].solution.obj == pytest.approx(ref.solution.obj, rel=1e-7), name
        mine = steps[_realized_name(scenario, k)]
        for var, key in (("v_state", {"n": "battery"}), ("v_online_integer", {"p": "coal_units"})):
            a, b = _cells(mine, var, **key), _cells(ref, var, **key)
            jump_cells = [cell(i) for i in jump(k)]
            assert [a[c] for c in jump_cells] == pytest.approx([b[c] for c in jump_cells], abs=1e-6), (k, var)


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------

def test_notices(runs):
    """rolling_deterministic: W-0593 (the forecasts are not solved).  The other scenarios: no notice (a database
    made at the current data version, so no upgrade notice).  W-0553 (warm reuse off) comes from the test's
    ``warm=False``.  No other coded warning or note."""
    for scenario in SCENARIOS:
        _mps, _steps, got, _csv = runs[scenario]
        expected = {"W-0593"} if scenario == "rolling_deterministic" else set()
        assert set(got.codes()) == expected | {"W-0553"}, (scenario, got.codes())
