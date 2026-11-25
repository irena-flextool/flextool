"""The ``separate_dispatch`` realization rolls of ``rolling_separate_dispatch.json`` (design rev 7.1 §8.3 T-b1, T-b3,
T-b4; rev 4 §7 / rev 2 §5 for the hand values).

Fixture: one period ``period1`` of 72 one-hour steps, a rolling ``separate_dispatch`` solve ``1week_rolling`` with jump
24 and horizon 48 (3 rolls; roll 2 holds t0049-72 only: horizon = jump), forecasts ``upper`` / ``mid`` / ``lower``
(weight 1 each, so 1/3) and ``realized`` made at t0001, t0025, t0049.  Demand 3000 MW (overlay: 4470), ``wind_plant``
4000 MW on the upper-limit profile ``wind``, hydro 1000 MW to the demand (a commodity reservoir), ``gas_plant`` 2 x 250
MW (fuel 6 / 0.9) and ``coal_plant`` 3 x 600 MW (fuel 16 / 0.4 = 40 EUR/MWh, min load 0.4, startup_cost 100 EUR/MW), both
``integer_fix_in_uc``.  Every DB is built from the JSON in tmp_path.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine import _solve_copy_models as scm

pack = scm.pack
pytestmark = pytest.mark.solver

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "rolling_separate_dispatch.json"
SCENARIO = "rolling_separate_dispatch"
FORECASTS = ("lower", "mid", "upper")


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _run(data: dict, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rsd.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return scm.run_capturing(scm.build_db(path, folder), SCENARIO, folder)


def _cf_caps(mps, unit: str = "wind_plant") -> dict[tuple[str, str], float]:
    """``{(d, t): capacity factor ceiling}`` of *unit* in an LP: ``profile_flow_upper_limit`` rhs / its flow coefficient
    (``v_flow`` is per unit size, the existing 4000 MW)."""
    lp = scm.lp_snapshot().parse_mps(mps)
    out = {}
    for row in lp.rows:
        if row.startswith(f"profile_flow_upper_limit[{unit},"):
            d, t = row[:-1].split(",")[-2:]
            coef = next(v for (r, c), v in lp.coefs.items() if r == row and c.startswith("v_flow["))
            out[(d, t)] = lp.rhs.get(row, 0.0) / coef
    return out


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return _run(json.loads(FIXTURE.read_text(encoding="utf-8")), tmp_path_factory.mktemp("rsd_base"))


def test_realization_roll_reads_the_forecasts_made_at_its_jump_end(base):
    """T-b3 (rev 4 §7) under rev 7.1 / T-b2's "R_k reads C_k*": realization roll 0 dispatches t0001-24 once under
    ``period1`` on ``realized`` (made at t0001: 0.500883085 at t0024, the fixture's value) and looks ahead from t0025
    with the forecasts made at t0025 (at t0025: lower 0.200301152, mid 0.444466533, upper 0.904231580), weights
    1/3 each; ``period1`` cells weigh 1.  Its cells: 24 jump steps + 3 x 24 tail steps."""
    built, steps = base
    real = [(s, p) for s, p, _ in built if "__realized_dispatch" in s]
    names = list(steps)
    r0 = steps[names[1]]
    assert "__realized_dispatch" in names[1]
    caps = _cf_caps(real[0][1])
    assert len(caps) == 24 + 3 * 24
    assert {d for d, _ in caps} == {"period1", *(f"period1_{f}" for f in FORECASTS)}
    assert sorted(t for d, t in caps if d == "period1") == [f"t{i:04d}" for i in range(1, 25)]
    assert caps[("period1", "t0024")] == pytest.approx(0.500883085, rel=1e-9)
    for f, cf in zip(FORECASTS, (0.200301152, 0.444466533, 0.904231580), strict=True):
        assert caps[(f"period1_{f}", "t0025")] == pytest.approx(cf, rel=1e-9), f
        assert sorted(t for d, t in caps if d == f"period1_{f}") == [f"t{i:04d}" for i in range(25, 49)]
    frame = r0.flex_data.pdt_branch_weight
    frame = frame if isinstance(frame, pl.DataFrame) else frame.frame
    weights = {str(d): v for d, v in frame.select("d", "value").unique().rows()}
    assert weights == pytest.approx({"period1": 1.0, **{f"period1_{f}": 1 / 3 for f in FORECASTS}}, rel=1e-12)


def test_last_realization_roll_is_its_jump_only(base):
    """T-b4: roll 2's horizon is its jump (t0049-72), so its realization roll is deterministic on ``realized`` (made
    at t0049): only ``period1`` cells, no forecast period, no non-anticipativity row, the coal and gas pins of solve
    (1) over the 24 steps (2 units x 24 = 48 ``online_fix_eq_integer`` rows)."""
    built, _steps = base
    last = [p for s, p, _ in built if "__realized_dispatch" in s][-1]
    lp = scm.lp_snapshot().parse_mps(last)
    caps = _cf_caps(last)
    assert sorted(caps) == [("period1", f"t{i:04d}") for i in range(49, 73)]
    assert caps[("period1", "t0049")] == pytest.approx(0.473682299, rel=1e-9)
    assert not [r for r in lp.rows if r.startswith("non_anticipativity")]
    assert not [c for c in lp.cols if any(f"period1_{f}" in c for f in FORECASTS)]
    pins = [r for r in lp.rows if r.startswith("online_fix_eq")]
    assert len(pins) == 48 and all(r.startswith("online_fix_eq_integer[") for r in pins)


def _myopia(startup: str) -> dict:
    """T-b1's overlay: demand 4470 MW; ``coal_plant`` re-decided by the realization (*startup* ``linear`` or ``integer``,
    not ``_fix_in_uc``), ``min_uptime`` 8 h, availability 0 at t0025-30 (a plain series: every branch); the realized
    wind made at t0001 is 0.9 at t0001-17 and 0.3 at t0018-24."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rows = []
    for r in data["parameter_values"]:
        if r[:2] == ["node", "demand_node"] and r[2] == "inflow":
            r = [*r[:3], pack(-4470.0, "float"), r[4]]
        elif r[:2] == ["unit", "coal_plant"] and r[2] in ("startup_method", "min_uptime"):
            continue
        elif r[:3] == ["profile", "wind", "profile"] and r[4] == "1week_rolling":
            value = json.loads(base64.b64decode(r[3][0]))
            realized = dict(value["data"])["realized"]
            for at, series in realized["data"]:
                if at == "t0001":
                    series["data"] = [[t, 0.9 if int(t[1:]) <= 17 else 0.3] for t, _ in series["data"]]
            r = [*r[:3], pack(value, "map"), r[4]]
        rows.append(r)
    availability = {"index_type": "str", "index_name": "time",
                    "data": [[f"t{i:04d}", 0.0 if 25 <= i <= 30 else 1.0] for i in range(1, 73)]}
    rows += [["unit", "coal_plant", "startup_method", pack(startup, "str"), "system"],
             ["unit", "coal_plant", "min_uptime", pack(8.0, "float"), "system"],
             ["unit", "coal_plant", "availability", pack(availability, "map"), "system"]]
    data["parameter_values"] = rows
    return data


@pytest.mark.parametrize(("startup", "online"), [("linear", 2.95), ("integer", 3.0)])
def test_realization_sees_the_outage_after_its_jump(tmp_path, startup, online):
    """T-b1 (the myopia, rev 2 §5), hand: at t0018-24 the realized demand needs 4470 - 4000 x 0.3 - 1000 (hydro) - 500
    (gas, both units pinned online by solve (1): its lower forecast is short in every jump hour) = 1770 MW of coal =
    2.95 units at full output (linear; integer: 3).  Coal is unavailable at t0025-30 and must stay online 8 h after a
    start, so the realization roll, which looks ahead into the outage, cannot start it after t0017; an earlier start
    only adds hours at min load (2.95 x 240 MW x 40 EUR/MWh per hour > nothing: at t0001-16 wind 3600 + hydro + gas
    cover the demand), and roll 0 starts free (no carried history), so coal is offline at t0001-16 and starts 2.95
    (3) units at t0017.  Before 4.7.0 the jump-only realization started coal at t0018 and solve (1) of roll 1
    inherited an up-time running into the outage (infeasible).  Every solve is optimal."""
    built, steps = _run(_myopia(startup), tmp_path)
    assert all(s.solution.optimal for s in steps.values())
    names = list(steps)
    assert len(names) == 6 and "__realized_dispatch" in names[1]
    r0 = steps[names[1]]
    var = "v_online_linear" if startup == "linear" else "v_online_integer"
    frame = r0.solution.value(var).with_columns(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t"))
    coal = {t: v for t, v in frame.filter((pl.col("p") == "coal_plant") & (pl.col("d") == "period1"))
            .select("t", "value").rows()}
    want = {f"t{i:04d}": (online if i >= 17 else 0.0) for i in range(1, 25)}
    assert coal == pytest.approx(want, abs=1e-6)
    starts = r0.solution.value("v_startup_linear" if startup == "linear" else "v_startup_integer").with_columns(
        pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t"))
    started = {t: v for t, v in starts.filter((pl.col("p") == "coal_plant") & (pl.col("d") == "period1"))
               .select("t", "value").rows() if abs(v) > 1e-9}
    assert started == pytest.approx({"t0017": online}, abs=1e-6)
