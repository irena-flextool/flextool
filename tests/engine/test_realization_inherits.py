"""What a rolling ``separate_dispatch`` realization keeps of its stochastic solve: the units online of the units
with ``startup_method`` ``linear_fix_in_uc`` or ``integer_fix_in_uc``, and the investments.

``linear_fix_in_uc`` / ``integer_fix_in_uc``: each realization roll pins the units online over its jump to the
stochastic roll's reference forecast (``online_fix_eq_linear`` / ``online_fix_eq_integer``); since 4.7.0 the
realization roll also looks ahead after the jump with the forecasts made at its end, where nothing is pinned.  ``linear`` / ``integer``: no pin; the realization decides the units online again
with the realized data, their minimum up and down times and the previous realization's history.  When a realization
solve keeps nothing (no ``_fix_in_uc`` unit online, no investments), note W-0588 says so, once.

Fixture: ``rolling_separate_dispatch.json`` as the reported two-stage case (one forecast ``mid`` + the ``realized:
yes`` branch at each roll start, 24-h jump, 48-h horizon, 10-h minimum up / down times; no investment), built from
JSON under the pytest tmp dir (``tests.engine.online.test_online_free_first_step``).  No rolling optimum is
hand-derived; the first realization roll starts free in both cases, so its re-decided objective cannot be above the
pinned one (the pinned commitment is feasible for the re-decided LP, which differs from the pinned LP only by the pin
rows).  Later rolls start from different histories and are not compared.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests._support.coded_warnings import capture_coded
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build, utf8
from tests.engine.online.test_online_free_first_step import ROLLS, _enc, _reporter, _run_rolling

pytestmark = pytest.mark.solver

REALIZATION = "1week_rolling__realized_dispatch"
UNITS = ("coal_plant", "gas_plant")


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``{startup_method: (steps, coded warnings, work folder)}``, each run once."""
    cache: dict = {}

    def get(startup: str):
        if startup not in cache:
            mp = pytest.MonkeyPatch()
            mp.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
            try:
                tmp = tmp_path_factory.mktemp(f"inherits_{startup}")
                with capture_coded() as got:
                    steps = _run_rolling(tmp, _reporter(startup))
            finally:
                mp.undo()
            cache[startup] = (steps, got, tmp / "work")
        return cache[startup]
    return get


def _no_online(data: dict) -> dict:
    """The reported case without units online: no startup_method and no minimum_time_method."""
    data = _reporter("linear")(data)
    for r in data["parameter_values"]:
        if r[0] == "unit" and r[2] == "startup_method":
            r[3] = _enc("no_startup", "str")
        elif r[0] == "unit" and r[2] == "minimum_time_method":
            r[3] = _enc("none", "str")
    return data


def _pairs(steps: dict) -> list:
    """``[(stochastic roll k, realization roll k)]``."""
    return [(steps[ROLLS[2 * k]], steps[ROLLS[2 * k + 1]]) for k in range(len(ROLLS) // 2)]


def _jump(step) -> set[tuple[str, str]]:
    """The realization roll's realized (jump) steps ``(period, step)``."""
    rd = utf8(step.flex_data.realized_dispatch)
    return set(zip(rd["period"], rd["step"]))


def _objective(work, solve: str) -> float:
    return float(pl.read_parquet(work / "output_raw" / f"v_obj__{solve}.parquet")["objective"][0])


def _kind(startup: str) -> str:
    return "linear" if startup.startswith("linear") else "integer"


@pytest.mark.parametrize("startup", ["linear", "integer"])
def test_plain_units_online_are_decided_again_in_the_realization(runs, startup) -> None:
    """``linear`` / ``integer``: every realization roll has no units-online pin (no p_fix_online, no online_fix_eq
    row) but its own minimum up time rows (2 units × its 48 steps: the 24 of the jump and the 24 of the forecast
    ``mid`` after it; the last roll, t0049-t0072, is its jump only: 2 × 24), all six solves are optimal, and W-0588
    is given once (entityless): nothing is pinned and the stochastic solve has no investments."""
    steps, got, _ = runs(startup)
    kind = _kind(startup)
    assert list(steps) == ROLLS and all(s.solution.optimal for s in steps.values())
    pairs = _pairs(steps)
    for k, (_, real) in enumerate(pairs):
        assert real.flex_data.p_fix_online is None
        pb = build(real.flex_data)
        assert not row_terms(pb, "online_fix_eq_linear") and not row_terms(pb, "online_fix_eq_integer")
        assert len(row_terms(pb, f"minimum_uptime_{kind}")) == (48 if k == len(pairs) - 1 else 96), k
    assert got.entityless("W-0588") and len(got.lines("W-0588")) == 1
    assert "'1week_rolling__realized_dispatch'" in got.lines("W-0588")[0] and "'1week_rolling'" in got.lines(
        "W-0588")[0]


@pytest.mark.parametrize("startup", ["linear_fix_in_uc", "integer_fix_in_uc"])
def test_fix_in_uc_units_online_are_pinned_over_the_jump(runs, startup) -> None:
    """``_fix_in_uc``: each realization roll pins both units at exactly the 24 steps of its jump (one
    online_fix_eq row per unit and step, none of the other class), each pin equals the stochastic roll's units
    online of the reference forecast ``<period>_mid`` at that step, the realization's units online equal the pins,
    and no note is given."""
    steps, got, _ = runs(startup)
    kind = _kind(startup)
    other = "integer" if kind == "linear" else "linear"
    assert all(s.solution.optimal for s in steps.values())
    for k, (stoch, real) in enumerate(_pairs(steps)):
        jump = _jump(real)
        assert len(jump) == 24, k
        pb = build(real.flex_data)
        rows = {tuple(r[r.index("[") + 1:-1].split(",")) for r in row_terms(pb, f"online_fix_eq_{kind}")}
        assert rows == {(p, d, t) for p in UNITS for d, t in jump}, k
        assert not row_terms(pb, f"online_fix_eq_{other}"), k
        pin = utf8(real.flex_data.p_fix_online.frame)
        ref = (utf8(stoch.solution.value(f"v_online_{kind}"))
               .with_columns(pl.col("d").str.replace(r"_mid$", ""))
               .select("p", "d", "t", pl.col("value").alias("ref")))
        joined = pin.join(ref, on=["p", "d", "t"], how="left")
        assert joined.height == 48 and joined["ref"].null_count() == 0, k
        assert joined["value"].to_list() == pytest.approx(joined["ref"].to_list(), abs=1e-9), k
        got_online = (utf8(real.solution.value(f"v_online_{kind}"))
                      .join(pin.select("p", "d", "t", pl.col("value").alias("pin")), on=["p", "d", "t"]))
        assert got_online["value"].to_list() == pytest.approx(got_online["pin"].to_list(), abs=1e-6), k
    assert "W-0588" not in got.codes()


@pytest.mark.parametrize("plain", ["linear", "integer"])
def test_first_realization_roll_redecided_is_not_dearer_than_pinned(runs, plain) -> None:
    """The first realization roll (a free start in both runs, the same realized data): the re-decided objective is
    at most the pinned one (module docstring).  The stochastic first rolls are the same LP, so their pins are the
    plain run's stochastic commitment."""
    _, _, plain_work = runs(plain)
    _, _, pinned_work = runs(f"{plain}_fix_in_uc")
    stoch, real = (_objective(plain_work, ROLLS[i]) for i in (0, 1))
    assert stoch == pytest.approx(_objective(pinned_work, ROLLS[0]), rel=1e-9)
    assert real <= _objective(pinned_work, ROLLS[1]) * (1 + 1e-9)


def test_a_realization_without_units_online_or_investment_is_noted(tmp_path_factory) -> None:
    """No units online (no startup_method, no minimum_time_method) and no investment: the realization keeps nothing
    of its stochastic solve, W-0588 says so once (entityless)."""
    with capture_coded() as got:
        steps = _run_rolling(tmp_path_factory.mktemp("inherits_nothing"), _no_online)
    assert all(s.solution.optimal for s in steps.values())
    assert got.entityless("W-0588") and len(got.lines("W-0588")) == 1
