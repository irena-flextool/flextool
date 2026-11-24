"""OC1 rolling + stochastic separate_dispatch — interleaved UC/dispatch rolls
with a two-source handoff, each dispatch roll scoped to its realized jump.

A ``dispatch_realization_method=separate_dispatch`` solve under
``solve_mode=rolling_window`` expands to the INTERLEAVED order
``[uc_roll_0, disp_roll_0, uc_roll_1, disp_roll_1, ...]`` where, per roll, a
stochastic UC solve (forecast fan) is followed by a deterministic
realized-dispatch sibling.  Two carrier flows are kept separate:

  * PIN: each ``disp_roll_k`` fixes its ``v_online`` to ``uc_roll_k``'s
    committed hedge (``online_fix_eq`` / ``fix_online``), consuming it from
    its sequential predecessor ``uc_roll_k``.
  * ROLL-FORWARD: the realized trajectory flows disp→disp — ``disp_roll_k``
    takes its roll-forward (``roll_end_online`` → ``p_roll_continue_online``,
    etc.) from ``disp_roll_{k-1}``, NOT from ``uc_roll_k``.

**SCOPE (the invariant this test pins; 4.7.0).** Each ``disp_roll_k`` is the
realization roll of ``uc_roll_k``: it spans the UC roll's whole horizon, the
jump dispatched once under the period's own name (``period1``) on the
realized data, with the units online pinned to the UC commitment, and the
look-ahead after it (``period1_<forecast>``) read from the forecasts made at
the end of the jump, with their probabilities.  Before 4.7.0 the dispatch
solved only the jump, without look-ahead (solving the look-ahead on the
realized-only data then forced spurious unserved load).
``test_dispatch_rolls_scoped_to_jump`` asserts the jump cells are exactly the
``JUMP_STEPS`` (24) steps of the realized period and the roll spans its UC
roll's steps.  The jump's realized values (flows, fuel, spill, pins) are read
under ``period1``: only the pinned units online couple the jump and the
look-ahead here, so their hand values hold.

This is pure orchestration + handoff routing — NO new var/cstr/param family
(the pin family ``online_fix_eq`` is already registered), so CLAUDE.md
autoscale invariant #1 is not triggered.

Fixture (``rolling_separate_dispatch.json``, built from JSON per CLAUDE.md
invariant 3): 72 steps = 3 jump-24 rolls (horizon 48).  A committable INTEGER
fleet (MIP, ``startup_method: binary``): ``gas_plant`` (2 units × 250 MW,
efficiency 0.9, gas price 6 → fuel 6.67 per MWh out, min_load 0.4, min_up/down
3, startup_cost 50) and ``coal_plant`` (3 units × 600 MW, efficiency 0.4, coal
price 16 → fuel 40 per MWh out, min_load 0.4, min_up/down 4, startup_cost 100)
back ``wind`` (4000 MW, branch-varying forecast; freely curtailable) + ``hydro``
(2000 MW on the sum of its two outputs, split equally by ``hydro_divide``, so
1000 MW reach the demand; spilling to ``downriver`` costs 1 per MWh); the
``realized``/``upper``/``mid``/``lower`` fan re-reveals at each roll boundary.
Operational only (no invest).  Coal is the EXPENSIVE thermal unit here, not a
cheap baseload.

Overlay (``_retune``, applied by the ``chain`` fixture only; the shared JSON,
its other users, ``lp_snapshot`` and ``output_snapshot`` are unchanged): the
demand is 4470 MW, and the realized wind of each roll is three 8-hour blocks
(``REALIZED_WIND``, different per roll).  Why: with every unit at full output
the lower forecast has 4000·r + 1000 + 1800 + 500 MW against 4470, so it is
short by 1170 − 4000·r_lower MW in every jump hour (at least 14.04 / 143.90 /
12.57 MW in rolls 0 / 1 / 2, from the JSON's lower forecast).  The commitment
is tied across the forecasts over the jump, and taking any unit offline costs at
least 250 MW x 5000 x 1/3 of VOLL per hour, far above any saving, so the UC
commits 2 gas + 3 coal in every jump hour of every roll (asserted directly by
``test_uc_lower_forecast_is_short_in_every_jump_hour``).  In the realized
dispatch coal runs at its 720 MW floor, hydro at 1000 MW, wind in full, and gas
closes the gap, 2750 − 4000·r MW, strictly between its 200 MW floor and its
500 MW capacity: gas is the marginal unit, and the flows, fuel and objective
are unique and hand-derived from the constants below.

Known limit: the forced full-fleet commitment equals the existing unit count in
every roll, so a pin routed from the wrong roll or to the unit count is not
caught by these numbers.  ``test_dispatch_within_committed_unit_limits``
(per-step pin == UC plan) and the base-fixture users (``peak_chain``,
``test_online_free_first_step``, ``test_realization_inherits``, whose
commitments vary inside the rolls) cover that.

Owner-confirmed semantics (``==`` pin): the realized dispatch is LOCKED to the
plan's committed hedge (per-unit ``==``/``≤`` is a future setting).  The test
asserts ZERO unserved load (up-slack) in every realized-dispatch roll; the
``==`` lock may spill via the free curtailment / small water penalty.

NOTE on anchors: only the realized-DISPATCH objectives are anchored, as
formulas of the fixture constants (with ``v_online`` pinned the dispatch is a
pure LP with a unique optimum).  The UC-parent objectives are intentionally NOT
anchored: they are dominated by VOLL in the lower forecast and in the DISCARDED
look-ahead half, and the look-ahead commitment is not unique.  Everything else
is asserted as a structural invariant.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "rolling_separate_dispatch.json"

SCENARIO = "rolling_separate_dispatch"
UC = "1week_rolling_roll_{k}"
DISP = "1week_rolling__realized_dispatch_roll_{k}"
N_ROLLS = 3
GAS = "gas_plant"
COAL = "coal_plant"
REALIZED_PERIOD = "period1"            # the realized trunk the sibling owns

# Fixture data used for the hand-derived checks.  ``v_flow`` of a unit is in
# units of its ``virtual_unitsize`` (1.0 = one unit at full output).
UNIT = {   # p: (unitsize MW, efficiency, fuel price per MWh fuel, min_load)
    GAS: (250.0, 0.9, 6.0, 0.4),
    COAL: (600.0, 0.4, 16.0, 0.4),
}
N_UNITS = {GAS: 2, COAL: 3}            # unit.existing
SPILL_NODE = "downriver"               # the only node with a nonzero penalty_down
SPILL_PENALTY = 1.0                    # downriver.penalty_down
ANNUALISE = 8760.0 / 72.0              # the 72 h period scaled to a year

# Overlay data (``_retune``) and the fixture data the derivation uses.
DEMAND = 4470.0                        # demand_node inflow (overlay), MW
WIND_MW = 4000.0                       # wind_plant existing
HYDRO_TO_DEMAND = 2000.0 / 2           # existing on the sum of the two outputs, hydro_divide splits them equally
DOWNRIVER_DEMAND = 100.0               # downriver inflow -100
BLOCK_H = 8                            # realized wind: three 8-hour blocks per roll
REALIZED_WIND = {0: (0.60, 0.63, 0.57), 1: (0.58, 0.61, 0.595), 2: (0.625, 0.60, 0.59)}
COAL_FLOOR_MW = N_UNITS[COAL] * UNIT[COAL][0] * UNIT[COAL][3]   # 3 x 600 x 0.4 = 720


def _fuel_per_mwh(p: str) -> float:
    _, eff, price, _ = UNIT[p]
    return price / eff


# Gas closes the gap left by full wind, hydro at its maximum and coal at its floor
# (merit order wind 0 < hydro 1 < gas 6.67 < coal 40); per roll and 8-hour block,
# in units of the 250 MW unitsize.
GAS_FLOW = {k: tuple((DEMAND - WIND_MW * r - HYDRO_TO_DEMAND - COAL_FLOOR_MW) / UNIT[GAS][0] for r in rs)
            for k, rs in REALIZED_WIND.items()}
COAL_FLOW = COAL_FLOOR_MW / UNIT[COAL][0]                       # 1.2
DISP_FUEL = {k: sum(BLOCK_H * g * UNIT[GAS][0] * _fuel_per_mwh(GAS) for g in GAS_FLOW[k])
             + 24 * COAL_FLOOR_MW * _fuel_per_mwh(COAL) for k in GAS_FLOW}
SPILL = 24 * (HYDRO_TO_DEMAND - DOWNRIVER_DEMAND)               # 21 600 MWh per roll
DISP_OBJ = {k: ANNUALISE * (DISP_FUEL[k] + SPILL_PENALTY * SPILL) for k in DISP_FUEL}

# Rolling config from the fixture (``solve.jump`` / ``solve.horizon`` hours at
# 1 h/step): each roll optimises HORIZON steps, commits/realises JUMP steps.
JUMP_STEPS = 24
HORIZON_STEPS = 48

# Per-roll committed hedge (the tied forecast commitment, pinned verbatim into
# the dispatch sibling): the lower forecast is short in every jump hour, so every
# unit is committed.
HEDGE = {k: float(N_UNITS[GAS]) for k in range(N_ROLLS)}
COAL_HEDGE = {k: float(N_UNITS[COAL]) for k in range(N_ROLLS)}


def _build_db(tmp_path: Path) -> str:
    if not FIXTURE.exists():
        pytest.skip(f"fixture JSON not present: {FIXTURE}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{FIXTURE.stem}.sqlite"
    url = json_to_db(FIXTURE, db_path)
    migrate_database(url)
    return url


def _retune(url: str) -> None:
    """Overlay of this module: demand ``DEMAND`` and, in the stored wind profile Map, the ``realized`` branch of
    each analysis time replaced by ``REALIZED_WIND`` (three 8-hour blocks); the forecasts are kept as stored."""
    from spinedb_api import DatabaseMapping, import_data
    from spinedb_api.parameter_value import Map, from_database

    with DatabaseMapping(url) as dm:
        item = dm.get_parameter_value_item(
            entity_class_name="profile", entity_byname=("wind",),
            parameter_definition_name="profile", alternative_name="1week_rolling")
        stored = from_database(item["value"], item["type"])
        branches, values = list(stored.indexes), list(stored.values)
        realized = values[branches.index("realized")]
        per_roll = []
        for k, (analysis_time, inner) in enumerate(zip(realized.indexes, realized.values)):
            assert analysis_time == f"t{1 + JUMP_STEPS * k:04d}" and len(inner.indexes) == JUMP_STEPS
            per_roll.append(Map(list(inner.indexes),
                                [REALIZED_WIND[k][h // BLOCK_H] for h in range(JUMP_STEPS)],
                                index_name=inner.index_name))
        values[branches.index("realized")] = Map(list(realized.indexes), per_roll, index_name=realized.index_name)
        n, errors = import_data(dm, parameter_values=[
            ("node", "demand_node", "inflow", -DEMAND, "system"),
            ("profile", "wind", "profile", Map(branches, values, index_name=stored.index_name), "1week_rolling"),
        ])
        assert not errors and n == 2, (n, errors)
        dm.commit_session("rolling_separate_dispatch: gas-marginal overlay")


def _run_chain(tmp_path: Path):
    from flextool.engine import run_chain_from_db

    url = _build_db(tmp_path)
    _retune(url)
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            url, SCENARIO, work_folder=work_folder,
            warm=False, keep_solutions=True,
        )
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    assert steps, "run_chain_from_db produced no orchestration steps"
    return steps, work_folder


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return _run_chain(tmp_path_factory.mktemp("rolling_separate_dispatch"))


def _online(step, p: str) -> pl.DataFrame:
    """``v_online`` for process ``p`` — integer (binary UC) or linear."""
    sol = step.solution
    for vn in ("v_online_integer", "v_online_linear"):
        if vn in sol._vars:
            f = sol.value(vn).filter(pl.col("p") == p)
            if f.height:
                return f
    return pl.DataFrame(schema={"p": pl.Utf8, "d": pl.Utf8,
                                "t": pl.Utf8, "value": pl.Float64})


def _n_time_steps(step) -> int:
    """Distinct time steps the solve actually spans (branch-independent)."""
    return int(step.solution.value("v_flow")["t"].n_unique())


@pytest.mark.solver
def test_is_mip_binary_commitment(chain):
    """The committable fleet is BINARY — gas and coal both carry an integer
    ``v_online`` (MIP), the realistic UC formulation."""
    steps, _ = chain
    uc0 = steps[UC.format(k=0)].solution
    assert "v_online_integer" in uc0._vars, "expected a binary (MIP) UC"
    procs = set(uc0.value("v_online_integer")["p"].unique().to_list())
    assert GAS in procs and COAL in procs, (
        f"both gas and coal should be binary-committable; got {procs}"
    )


@pytest.mark.solver
def test_interleaved_expansion_order(chain):
    """R1: the solve expands to the INTERLEAVED order
    ``[uc_0, disp_0, uc_1, disp_1, uc_2, disp_2]``."""
    steps, _ = chain
    expected = []
    for k in range(N_ROLLS):
        expected.append(UC.format(k=k))
        expected.append(DISP.format(k=k))
    assert list(steps) == expected


@pytest.mark.solver
def test_dispatch_rolls_scoped_to_jump(chain):
    """SCOPE INVARIANT (4.7.0).  Each realization roll spans its UC roll's
    horizon (``HORIZON_STEPS`` = 48 steps for rolls 0 and 1; roll 2 is the
    truncated last 24 steps of the 72-step timeline): the jump t(24k+1) ..
    t(24k+24) under the realized period ``period1`` only, and from t(24k+25)
    on the forecasts ``period1_<f>`` made there (none in roll 2, whose horizon
    is its jump).  Before 4.7.0 the realization solved the jump only."""
    steps, _ = chain
    assert JUMP_STEPS < HORIZON_STEPS   # sanity: a look-ahead exists
    for k in range(N_ROLLS):
        uc_steps = _n_time_steps(steps[UC.format(k=k)])
        disp = steps[DISP.format(k=k)].flex_data.dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
        assert uc_steps == (HORIZON_STEPS if k < N_ROLLS - 1 else JUMP_STEPS), (k, uc_steps)
        assert _n_time_steps(steps[DISP.format(k=k)]) == uc_steps
        jump = {f"t{24 * k + h:04d}" for h in range(1, JUMP_STEPS + 1)}
        assert set(disp.filter(pl.col("d") == REALIZED_PERIOD)["t"].to_list()) == jump
        tails = disp.filter(pl.col("d") != REALIZED_PERIOD)
        if k == N_ROLLS - 1:
            assert tails.height == 0
        else:
            assert set(tails["d"].to_list()) == {f"{REALIZED_PERIOD}_{f}" for f in ("upper", "mid", "lower")}
            assert set(tails["t"].to_list()) == {f"t{24 * k + h:04d}" for h in range(25, 49)}


@pytest.mark.solver
def test_each_dispatch_roll_pinned_to_its_uc_commitment(chain):
    """R2 pin: each ``disp_roll_k`` gas and coal ``v_online`` == ``uc_roll_k``'s
    committed hedge (the tied forecast commitment, excluding the weight-0
    realized trunk branch), non-degenerate: every unit (``HEDGE`` /
    ``COAL_HEDGE``), as the lower forecast's shortfall forces."""
    steps, _ = chain
    for p, hedge in ((GAS, HEDGE), (COAL, COAL_HEDGE)):
        for k in range(N_ROLLS):
            uc = _online(steps[UC.format(k=k)], p)
            disp = _online(steps[DISP.format(k=k)], p)
            uc_hedge = uc.filter(pl.col("d") != REALIZED_PERIOD)["value"].max()
            disp_commit = disp.filter(
                pl.col("d") == REALIZED_PERIOD)["value"].max()
            assert disp_commit == pytest.approx(uc_hedge, rel=1e-6), (
                f"roll {k} {p}: disp commit {disp_commit} != uc hedge {uc_hedge}"
            )
            assert disp_commit == pytest.approx(hedge[k], rel=1e-6)
            assert disp.filter(pl.col("d") == REALIZED_PERIOD)["value"].min() == pytest.approx(hedge[k], rel=1e-6), (
                f"roll {k} {p}: not every unit online in every realized hour")


def _unit_flow(step, p: str) -> pl.DataFrame:
    """``v_flow`` of unit *p* summed per ``(d, t)`` (in units of its unitsize)."""
    return (step.solution.value("v_flow").filter(pl.col("p") == p)
            .group_by(["d", "t"]).agg(pl.col("value").sum().alias("flow")))


def _fuel_cost(step) -> float:
    """Fuel cost of the solve's realized jump (period ``period1``) from its flows: Σ flow x unitsize / efficiency x
    price."""
    total = 0.0
    for p, (size, eff, price, _) in UNIT.items():
        flows = _unit_flow(step, p).filter(pl.col("d") == REALIZED_PERIOD)
        total += float(flows["flow"].sum()) * size / eff * price
    return total


@pytest.mark.solver
def test_dispatch_within_committed_unit_limits(chain):
    """In every realized-dispatch roll and step, each committable unit's online
    count equals its UC roll's commitment at that step (in every forecast), and
    its output lies within [min_load x online, capacity x online].  A dispatch
    that ignored the pinned commitment, ran a unit with none online, or ran it
    below its floor or above the committed capacity fails here."""
    steps, _ = chain
    for k in range(N_ROLLS):
        disp = steps[DISP.format(k=k)]
        uc = steps[UC.format(k=k)]
        for p, (_, _, _, min_load) in UNIT.items():
            # The jump cells: the look-ahead is decided again per forecast (its units online are not pinned).
            on = _online(disp, p).filter(pl.col("d") == REALIZED_PERIOD).select(
                "d", "t", pl.col("value").alias("online"))
            assert on.height == JUMP_STEPS, f"roll {k} {p}: {on.height} online rows"
            assert set(on["d"].unique()) == {REALIZED_PERIOD}
            uc_on = _online(uc, p).filter(pl.col("d") != REALIZED_PERIOD)
            for forecast in uc_on["d"].unique().sort().to_list():
                plan = dict(uc_on.filter(pl.col("d") == forecast).select("t", "value").rows())
                for t, online in on.select("t", "online").rows():
                    assert online == pytest.approx(plan[t], abs=1e-6), (
                        f"roll {k} {p} {t}: online {online} != UC {forecast} {plan[t]}"
                    )
            j = on.join(_unit_flow(disp, p), on=["d", "t"], how="left").with_columns(
                pl.col("flow").fill_null(0.0))
            low = j.filter(pl.col("flow") < pl.col("online") * min_load - 1e-6)
            high = j.filter(pl.col("flow") > pl.col("online") + 1e-6)
            assert low.height == 0, f"roll {k} {p}: below the min-load floor:\n{low}"
            assert high.height == 0, f"roll {k} {p}: above the committed capacity:\n{high}"


@pytest.mark.solver
def test_dispatch_fuel_cost_hand_derived(chain):
    """The fuel cost of each realization roll's jump, computed from its flows,
    equals the closed form ``DISP_FUEL`` (gas closing the gap per 8-hour block,
    coal at its 720 MW floor): 747 200 / 750 400 / 744 000 for rolls 0 / 1 / 2."""
    steps, _ = chain
    # Closed forms (8 x Σ gas MW / 0.9 x 6 + 24 x 720 / 0.4 x 16), so a constant typo shows:
    # roll 0 Σ gas 350 + 230 + 470, roll 1 430 + 310 + 370, roll 2 250 + 350 + 390.
    for k, gas_mw in ((0, 1050.0), (1, 1110.0), (2, 990.0)):
        assert DISP_FUEL[k] == pytest.approx(8 * gas_mw / 0.9 * 6 + 691_200.0, rel=1e-12)
    assert [DISP_FUEL[k] for k in range(N_ROLLS)] == pytest.approx([747_200.0, 750_400.0, 744_000.0], rel=1e-12)
    for k in range(N_ROLLS):
        got = _fuel_cost(steps[DISP.format(k=k)])
        assert got == pytest.approx(DISP_FUEL[k], rel=1e-9), (
            f"roll {k}: fuel cost {got} != hand-derived {DISP_FUEL[k]}"
        )


@pytest.mark.solver
def test_coal_committed_and_used(chain):
    """Coal is committed in the realized dispatch: all three coal units, since
    the lower forecast is short in every jump hour even with every unit at full
    output (``test_uc_lower_forecast_is_short_in_every_jump_hour``)."""
    steps, _ = chain
    coal = _online(steps[DISP.format(k=0)], COAL)
    assert coal.height > 0, "coal has no online var"
    assert coal["value"].max() > 0.5, "coal is never committed in dispatch"


def _lower_forecast_wind(k: int) -> dict[str, float]:
    """The stored ``lower`` wind forecast of roll *k* over its jump, read from the fixture JSON (the overlay keeps
    it)."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    (value,) = [pv[3] for pv in raw["parameter_values"]
                if pv[:3] == ["profile", "wind", "profile"] and pv[4] == "1week_rolling"]
    branches = dict(json.loads(base64.b64decode(value[0]))["data"])
    fan = dict(dict(branches["lower"]["data"])[f"t{1 + JUMP_STEPS * k:04d}"]["data"])
    return {f"t{i:04d}": fan[f"t{i:04d}"] for i in range(1 + JUMP_STEPS * k, 1 + JUMP_STEPS * (k + 1))}


@pytest.mark.solver
def test_uc_lower_forecast_is_short_in_every_jump_hour(chain):
    """The premise of the forced commitment: in every UC roll and jump hour the lower forecast is short by exactly
    ``DEMAND − HYDRO_TO_DEMAND − 3·600 − 2·250 − 4000·r_lower(t)`` = 1170 − 4000·r_lower(t) MW, the shortfall
    with every unit at full output — positive in every hour (≥ 14.04 / 143.90 / 12.57 MW in rolls 0 / 1 / 2).
    Taking any unit offline would add its whole capacity to the lower forecast's VOLL, so the UC commits every
    unit over the jump.  Fails if the forecast fan or its hours are read wrongly."""
    steps, _ = chain
    full_fleet = DEMAND - HYDRO_TO_DEMAND - sum(N_UNITS[p] * UNIT[p][0] for p in UNIT)
    assert full_fleet == pytest.approx(1170.0)
    for k in range(N_ROLLS):
        up = steps[UC.format(k=k)].solution.value("vq_state_up").filter(pl.col("n").cast(pl.Utf8) == "demand_node")
        got = dict(up.filter(pl.col("d").cast(pl.Utf8) == f"{REALIZED_PERIOD}_lower")
                   .select(pl.col("t").cast(pl.Utf8), "value").rows())
        lower = _lower_forecast_wind(k)
        assert min(full_fleet - WIND_MW * r for r in lower.values()) > 0
        for t, r in lower.items():
            assert got.get(t, 0.0) == pytest.approx(full_fleet - WIND_MW * r, rel=1e-9), (
                f"UC roll {k} {t}: lower-forecast shortfall {got.get(t)} != {full_fleet - WIND_MW * r}")


@pytest.mark.solver
def test_gas_is_marginal_in_every_realized_hour(chain):
    """In every realized-dispatch hour gas closes the gap, ``GAS_FLOW`` = (DEMAND − 4000·r − 1000 − 720) / 250,
    strictly between its floor (2 units x 0.4) and its capacity (2 units), with a 0.1-unit margin each way; coal
    stays at its floor (1.2 = 720 / 600)."""
    steps, _ = chain
    floor, cap = N_UNITS[GAS] * UNIT[GAS][3], float(N_UNITS[GAS])
    for k in range(N_ROLLS):
        disp = steps[DISP.format(k=k)]
        gas = dict(_unit_flow(disp, GAS).select(pl.col("t").cast(pl.Utf8), "flow").rows())
        coal = dict(_unit_flow(disp, COAL).select(pl.col("t").cast(pl.Utf8), "flow").rows())
        for h in range(JUMP_STEPS):
            t = f"t{1 + JUMP_STEPS * k + h:04d}"
            want = GAS_FLOW[k][h // BLOCK_H]
            assert floor + 0.1 < want < cap - 0.1
            assert gas.get(t, 0.0) == pytest.approx(want, rel=1e-9), f"roll {k} {t}: gas {gas.get(t)} != {want}"
            assert coal.get(t, 0.0) == pytest.approx(COAL_FLOW, rel=1e-9), f"roll {k} {t}: coal {coal.get(t)}"


@pytest.mark.solver
def test_realized_trajectory_carries_dispatch_to_dispatch(chain):
    """R2 roll-forward: ``disp_roll_k`` rolls forward on ``disp_roll_{k-1}``
    — its consumed ``p_roll_continue_online`` equals the PREVIOUS dispatch
    roll's realized end commitment (``roll_end_online``), nonzero.  (The
    first dispatch roll has no carry.)"""
    steps, _ = chain

    def _reo(k):   # realized end commitment handed off by disp_roll_k
        h = steps[DISP.format(k=k)].handoff.roll_end_online
        g = h.filter(pl.col("process") == GAS) if h is not None else None
        return float(g["value"][0]) if g is not None and g.height else 0.0

    def _prc(k):   # roll-forward commitment consumed by disp_roll_k
        p = steps[DISP.format(k=k)].flex_data.p_roll_continue_online
        g = p.frame.filter(pl.col("p") == GAS) if p is not None else None
        return float(g["value"][0]) if g is not None and g.height else None

    assert _prc(0) is None          # first roll: no prior dispatch
    for k in range(1, N_ROLLS):
        carried = _prc(k)
        prev_end = _reo(k - 1)
        assert carried is not None, f"disp roll {k} consumed no roll-forward"
        assert carried == pytest.approx(prev_end, rel=1e-6), (
            f"roll {k}: consumed {carried} != prev dispatch end {prev_end}"
        )
        assert carried > 0.0, (
            f"roll {k}: realized carry is zero (weight-0 garbage, not a real "
            f"dispatch trajectory)"
        )


@pytest.mark.solver
def test_all_rolls_optimal_and_realized_demand_met(chain):
    """All six solves OPTIMAL, and every realization roll serves all realized
    demand of its jump — ZERO unserved load (no up-slack VOLL) on ``period1``.
    Its look-ahead reads the forecasts, whose lower one is short (as in the
    UC roll, ``test_uc_lower_forecast_is_short_in_every_jump_hour``)."""
    steps, _ = chain
    for name, st in steps.items():
        assert st.optimal is True, f"{name} not optimal"
    for k in range(N_ROLLS):
        sol = steps[DISP.format(k=k)].solution
        if "vq_state_up" in sol._vars:
            up = sol.value("vq_state_up").filter(
                (pl.col("value") > 1e-3) & (pl.col("d").cast(pl.Utf8) == REALIZED_PERIOD))
            assert up.height == 0, (
                f"dispatch roll {k}: {up.height} unserved-load (up-slack) "
                f"rows — realized demand not met"
            )


@pytest.mark.solver
def test_dispatch_per_solve_objectives(chain):
    """The jump's realized cost: ``DISP_OBJ`` = ``ANNUALISE x (DISP_FUEL +
    downriver spill x penalty)`` (no start-up and no unserved load in the
    jump), with the spill ``SPILL`` = 24 x (1000 − 100) MWh: 93 537 333.33 /
    93 926 666.67 / 93 148 000 for rolls 0 / 1 / 2.  The last realization
    roll is its jump only (deterministic LP, ``v_online`` pinned), so its
    objective is ``DISP_OBJ[2]``; rolls 0 and 1 add the look-ahead, dominated
    by VOLL in the lower forecast, and are not anchored (as the UC rolls, see
    module docstring); their jump spill is ``SPILL``."""
    steps, _ = chain
    assert [DISP_OBJ[k] for k in range(N_ROLLS)] == pytest.approx(
        [8760 / 72 * 768_800, 8760 / 72 * 772_000, 8760 / 72 * 765_600], rel=1e-12)
    for k in range(N_ROLLS):
        name = DISP.format(k=k)
        assert name in steps, f"missing solve {name}"
        sol = steps[name].solution
        if k == N_ROLLS - 1:
            assert sol.obj == pytest.approx(DISP_OBJ[k], rel=1e-9), (
                f"{name}: objective {sol.obj} != {DISP_OBJ[k]}"
            )
        spill = float(sol.value("vq_state_down")
                      .filter((pl.col("n") == SPILL_NODE) & (pl.col("d") == REALIZED_PERIOD))["value"].sum())
        assert spill == pytest.approx(SPILL, rel=1e-9), f"{name}: spill {spill} != {SPILL}"


def _ancestor_closure(link: pl.DataFrame) -> set[tuple[str, str, str, str]]:
    """``(d, t, d′, t′)`` with ``(d′, t′)`` the step itself or an ancestor of ``(d, t)`` along *link*."""
    prev = {(d, t): (dp, tp) for d, t, dp, tp in
            link.select(*(pl.col(c).cast(pl.Utf8) for c in ("d", "t", "d_previous", "t_previous"))).rows()}
    out = set()
    for start in list(prev) + [v for v in prev.values()]:
        cur = start
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            out.add((*start, *cur))
            cur = prev.get(cur)
    return out


def test_mintime_lookback_never_reaches_a_sibling_forecast(chain):
    """In every UC roll (a fan of forecasts that starts at the roll's first step), each min up/down look-back row
    pairs a step with itself or with an ancestor along the ramp predecessor links — never with a step of a
    sibling forecast — and the first step of each forecast looks back at nothing but itself.  The look-back used
    to walk the name-sorted step list, from ``period1_mid,t0001`` into the last steps of ``period1_lower``."""
    steps, _ = chain
    for k in range(N_ROLLS):
        fd = steps[UC.format(k=k)].flex_data
        closure = _ancestor_closure(fd.dt_ramp_link)
        dt = fd.dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
        firsts = set(dt.group_by("d", maintain_order=True).agg(pl.col("t").first()).rows())
        assert len(firsts) == 3, f"roll {k}: expected three forecasts, got {firsts}"
        for which in ("uptime_lookback", "downtime_lookback"):
            lb = getattr(fd, which)
            assert lb is not None and lb.height > 0, f"roll {k}: {which} not built"
            rows = set(lb.select(*(pl.col(c).cast(pl.Utf8) for c in ("d", "t", "d_back", "t_back"))).rows())
            outside = {r for r in rows if r[:2] != r[2:] and r not in closure}
            assert not outside, f"roll {k} {which}: rows outside the predecessor chain: {sorted(outside)[:6]}"
            for d, t in firsts:
                backs = {r[2:] for r in rows if r[:2] == (d, t)}
                assert backs == {(d, t)}, f"roll {k} {which}: first step ({d}, {t}) looks back at {backs}"


def test_mintime_lookback_of_each_forecast_equals_the_realized_dispatch_twin(chain):
    """Relational: over the realized jump, each forecast of a UC roll and the jump of the roll's realization roll
    (its ``period1`` cells) cover the same steps of the same timeline, both starting at the roll's first step.
    Their min up/down look-back rows are therefore the same once the forecast's period name is mapped to the
    realized period's.  (The look-ahead of the realization roll looks back into the jump: other rows.)"""
    steps, _ = chain
    cols = ("p", "d", "t", "d_back", "t_back")
    for k in range(N_ROLLS):
        disp_fd = steps[DISP.format(k=k)].flex_data
        uc_fd = steps[UC.format(k=k)].flex_data
        disp_t = set(disp_fd.dt.filter(pl.col("d").cast(pl.Utf8) == REALIZED_PERIOD)["t"].cast(pl.Utf8).to_list())
        for which in ("uptime_lookback", "downtime_lookback"):
            twin = {r for r in getattr(disp_fd, which).select(*(pl.col(c).cast(pl.Utf8) for c in cols)).rows()
                    if r[1] == REALIZED_PERIOD}
            assert twin, f"roll {k}: {which} of the realized dispatch is empty"
            assert {r[1] for r in twin} == {REALIZED_PERIOD}
            uc = getattr(uc_fd, which).select(*(pl.col(c).cast(pl.Utf8) for c in cols))
            for forecast in uc["d"].unique().sort().to_list():
                mine = {(p, REALIZED_PERIOD, t, REALIZED_PERIOD if db == forecast else db, tb)
                        for p, d, t, db, tb in uc.filter(pl.col("d") == forecast).rows() if t in disp_t}
                assert mine == twin, (
                    f"roll {k} {which} {forecast}: differs from the realized twin: "
                    f"extra {sorted(mine - twin)[:4]}, missing {sorted(twin - mine)[:4]}")


# Min up/down times of the fixture (``unit.min_uptime`` / ``min_downtime``, hours; 1 h steps).
MIN_TIME = {COAL: 4.0, GAS: 3.0}


@pytest.fixture(scope="module")
def peak_chain(tmp_path_factory):
    """The fixture with a demand of 300 MW except 7000 MW in the first four hours of each later day
    (t0024-t0027, t0048-t0051, t0072): each realized dispatch roll starts units at its last step, t0024 / t0048,
    whose minimum up times reach into the next roll."""
    from spinedb_api import DatabaseMapping, import_data
    from spinedb_api.parameter_value import Map
    from flextool.engine import run_chain_from_db

    tmp = tmp_path_factory.mktemp("rolling_separate_dispatch_peak")
    url = _build_db(tmp)
    steps_t = [f"t{i:04d}" for i in range(1, 73)]
    demand = [-7000.0 if i % 24 in (0, 1, 2, 3) and i > 3 else -300.0 for i in range(1, 73)]
    with DatabaseMapping(url) as dm:
        import_data(dm, parameter_values=[("node", "demand_node", "inflow",
                                           Map(steps_t, demand, index_name="time"), "system")])
        dm.commit_session("demand peak at the roll seams")
    work = tmp / "work"
    work.mkdir()
    return run_chain_from_db(url, SCENARIO, work_folder=work, warm=False, keep_solutions=True)


def _startups(step, p: str) -> dict[str, float]:
    """The realized start-ups of *p* (period ``period1``: the jump of a realization roll, not its look-ahead)."""
    sol = step.solution
    for vn in ("v_startup_integer", "v_startup_linear"):
        if vn in sol._vars:
            f = sol.value(vn).filter((pl.col("p").cast(pl.Utf8) == p) & (pl.col("d").cast(pl.Utf8) == REALIZED_PERIOD))
            if f.height:
                return dict(zip(f["t"].cast(pl.Utf8).to_list(), f["value"].to_list()))
    return {}


@pytest.mark.solver
def test_min_uptime_carry_reaches_every_forecast_of_the_next_roll(peak_chain):
    """Each roll ≥ 1 of the UC fan continues the previous realized dispatch at its first step, so every forecast's
    first steps carry the start-ups that realized dispatch made within the minimum up time: on 1 h steps,
    carry[x] = Σ v_startup[h] over the realized steps h of the previous dispatch roll with idx(x) − idx(h) < L
    (L = 4 h coal, 3 h gas) — the same as the realized dispatch twin of the roll.  The forecasts used to get no
    carry (the history is under ``period1``, the forecasts' steps are not)."""
    steps = peak_chain
    assert all(s.optimal for s in steps.values())
    carried = 0
    for k in range(1, N_ROLLS):
        prev_disp = steps[DISP.format(k=k - 1)]
        for p, L in MIN_TIME.items():
            su = _startups(prev_disp, p)
            first = int(steps[DISP.format(k=k)].flex_data.dt["t"].cast(pl.Utf8).sort()[0][1:])
            expect = {}
            for x in range(first, first + int(L)):
                v = sum(s for t, s in su.items() if x - int(t[1:]) < L)
                if abs(v) > 1e-9:
                    expect[f"t{x:04d}"] = v
            assert expect, f"roll {k} {p}: the witness starts no unit within the minimum up time of the seam"
            for name, owner in ((DISP.format(k=k), None), (UC.format(k=k), "forecasts")):
                uc = steps[name].flex_data.uptime_carry
                assert uc is not None, f"{name}: no uptime carry"
                uc = uc.filter(pl.col("p") == p)
                periods = sorted(uc["d"].unique().to_list())
                if owner is None:
                    assert periods == [REALIZED_PERIOD]
                else:
                    assert periods == sorted(steps[name].flex_data.dt["d"].cast(pl.Utf8).unique().to_list())
                for d in periods:
                    got = dict(uc.filter(pl.col("d") == d).select("t", "value").rows())
                    assert got.keys() == expect.keys(), (name, p, d, got, expect)
                    for t, v in expect.items():
                        assert got[t] == pytest.approx(v, abs=1e-9), (name, p, d, t)
                    carried += 1
    assert carried > 0
