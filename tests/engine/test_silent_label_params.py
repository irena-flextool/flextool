"""Silent-label parameter inputs build the same LP as their unambiguous twin.

A silent-label input is a Map with Toolbox's default index name ``x`` whose keys are periods of other solves only
(``{p2030: ...}`` on the p2020 solve).  The entity then has no value in this solve, so the parameter default
applies.  Each case is paired with a twin that writes the intended value in a form the DB reader resolves
unambiguously; the twins are written by hand from the user meaning, not derived from code output.

Items of specs/github_issues.md:

* 167: a silent-label period Map with no key in the solve resolves to nothing (``pen``).
* 168: one ambiguous silent-label entity drops the whole parameter for its class (``avail``, ``varcost``; the
  unconditional-write field ``self_discharge_loss``, ``sd``, is fixed by the per-entity fallback in
  ``flextool/engine/input/_param_shapes.py`` -- ``resolve_param_shape`` now drops only the entity with no key
  in the solve instead of the whole parameter).  ``sd_lab``, ``sd_const_vs_map``, ``roll_sd``,
  ``roll_const_vs_map``, ``stoch_eff`` and ``stoch_const_vs_map`` extend the fix's coverage to single, rolling
  and stochastic solves (T168-2).
* 169: the depth-2 annual price ladder (tier -> {price, quantity}) is not read DB-side (``ladder``).

Today the emit seed masks items 167 and 169 (conditional writes), so those pairs hold; the comparison is the
whole LP (every MPS of the chain, canonical up to row/column order) plus the objective.  Item 168's
unconditional-write field (``sd``) was wrong before the per-entity fallback and is now a guard.

Rolling / stochastic key choice (critique.md item 1): ``fullYear_roll``'s active solve actually visits all
four of its periods (p2020, p2025, p2030, p2035) across its 72 rolls (verified empirically -- a per-roll probe
of ``node.self_discharge_loss``'s ``period_filter`` sees every one of them), so a silent key of ``p2030`` is
*not* reliably out of vocabulary: some rolls' ``d`` holds it, and the whole-frame value-domain probe then
resolves the parameter directly, without ever reaching the new per-entity fallback (the same "guard, not a
fail-before" situation as case ``sd_lab``/(b) in ``test_param_shapes_silent_per_entity.py``).  ``roll_sd`` /
``roll_const_vs_map`` use ``p2050`` instead (confirmed absent from ``fullYear_roll``'s periods and from its
timeline's step names), which is out of vocabulary in *every* roll.  ``roll_p2030_guard`` keeps a ``p2030``
case as the guard critique.md asks for: it passes before (west has no storage, so its value never reaches the
LP either way) and checks, at the FlexData level, that west's resolved value is present only in the rolls
whose window actually holds p2030 (verified empirically: rolls 35-53 of 72).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import polars as pl
import pytest

from tests.engine.test_cyclic_fix_start import _lp_snapshot

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
STOCH_JSON = REPO / "tests" / "fixtures" / "stochastics.json"
_RTOL_LP = 1e-12
_RTOL_OBJ = 1e-9

# fullYear_roll's 4 periods; every one of them is seen in some roll's d (see module docstring).
_FULLYEAR_ROLL_PERIODS = ("p2020", "p2025", "p2030", "p2035")


def _pen_twin(data: dict) -> dict:
    """west penalty_up / penalty_down written as the schema default 10000 (node penalty_up / penalty_down
    default_value), which is what a Map with no key in the solve leaves the node with."""
    return _lp_snapshot()._overlay(data, name="pen_twin", base="base", values=[
        ["node", "west", "penalty_up", 10000.0], ["node", "west", "penalty_down", 10000.0]])


def _avail_twin(data: dict) -> dict:
    """Only wind_plant availability 0.5: coal_plant keeps the default availability."""
    return _lp_snapshot()._overlay(data, name="avail_twin", base="coal_wind_ev", values=[
        ["unit", "wind_plant", "availability", 0.5]])


def _varcost_twin(data: dict) -> dict:
    """Only (wind_plant, west) other_operational_cost 3.0: (coal_plant, west) keeps no operational cost."""
    return _lp_snapshot()._overlay(data, name="varcost_twin", base="coal_wind_ev", values=[
        ["unit__outputNode", ["wind_plant", "west"], "other_operational_cost", 3.0]])


def _sd_case(data: dict) -> dict:
    """west self_discharge_loss = {p2030: 0.1} with index ``x``; west has no storage, so this value must change
    nothing, and ev_battery keeps its own self_discharge_loss."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="sd_silent", base="coal_wind_ev", values=[
        ["node", "west", "self_discharge_loss", lps._pmap({"p2030": 0.1}, index_name="x")]])


def _no_patch(data: dict) -> dict:
    return data


# ---------------------------------------------------------------------------
# T168-2: sd_lab -- the silent case against its LABELLED twin (not just "no override at all")
# ---------------------------------------------------------------------------


def _sd_lab_twin(data: dict) -> dict:
    """Same west self_discharge_loss = {p2030: 0.1}, but ``index_name="period"`` (labelled, unambiguous): the
    reader classifies it as MAP_PERIOD directly, west's row is filtered out by the solve's own periods (p2030
    is not p2020), and the LP equals ``coal_wind_ev``.  Before the fix: silent and labelled diverge, because
    the silent form drops ev_battery's own loss too."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="sd_lab_twin", base="coal_wind_ev", values=[
        ["node", "west", "self_discharge_loss", lps._pmap({"p2030": 0.1}, index_name="period")]])


# ---------------------------------------------------------------------------
# T168-2: sd_const_vs_map -- ev_battery's own loss as a constant vs. an equivalent silent map
# ---------------------------------------------------------------------------


def _sd_const_vs_map_case(data: dict) -> dict:
    """ev_battery self_discharge_loss reasserted as the constant 0.0001 (its own authored value), plus west's
    silent no-key map (same as ``_sd_case``).  ev_battery contributes nothing to the whole-frame value-domain
    probe (a constant has no index column), so west's lone unmatched key leaves the probe ambiguous and only
    the per-entity fallback saves ev_battery's own value."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="sd_const_vs_map", base="coal_wind_ev", values=[
        ["node", "ev_battery", "self_discharge_loss", 0.0001],
        ["node", "west", "self_discharge_loss", lps._pmap({"p2030": 0.1}, index_name="x")]])


def _sd_const_vs_map_twin(data: dict) -> dict:
    """The same value as a silent map keyed by the solve's own active period (p2020): the whole-frame probe
    resolves this directly (p2020 is a genuine period), independently of the fix, so this twin is the
    reference value for ``_sd_const_vs_map_case``."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="sd_const_vs_map_twin", base="coal_wind_ev", values=[
        ["node", "ev_battery", "self_discharge_loss", lps._pmap({"p2020": 0.0001}, index_name="x")],
        ["node", "west", "self_discharge_loss", lps._pmap({"p2030": 0.1}, index_name="x")]])


# ---------------------------------------------------------------------------
# T168-2: roll_sd / roll_const_vs_map -- fullYear_roll (rolling), per critique.md item 1
# ---------------------------------------------------------------------------


def _roll_sd_case(data: dict) -> dict:
    """battery self_discharge_loss reasserted as the constant 0.001, plus west's silent no-key map keyed by
    p2050 (outside fullYear_roll's period AND timestep vocabulary in every roll).  battery contributes nothing
    to the whole-frame probe (constant), so west's key alone leaves it ambiguous in every roll; only the
    per-entity fallback saves battery's own value."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="roll_sd", base="fullYear_roll", values=[
        ["node", "battery", "self_discharge_loss", 0.001],
        ["node", "west", "self_discharge_loss", lps._pmap({"p2050": 0.1}, index_name="x")]])


def _roll_sd_twin(data: dict) -> dict:
    """The same battery constant, without west's map at all: the reference value for ``_roll_sd_case`` (west
    has no storage, so its resolved value never reaches the LP either way; this twin only has to be right
    about battery, which it trivially is)."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="roll_sd_twin", base="fullYear_roll", values=[
        ["node", "battery", "self_discharge_loss", 0.001]])


def _roll_const_vs_map_case(data: dict) -> dict:
    """Same patch as ``_roll_sd_case`` (battery constant 0.001 + west's p2050 map), registered under its own
    scenario name so it doesn't collide with ``roll_sd``'s."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="roll_const_vs_map", base="fullYear_roll", values=[
        ["node", "battery", "self_discharge_loss", 0.001],
        ["node", "west", "self_discharge_loss", lps._pmap({"p2050": 0.1}, index_name="x")]])


def _roll_const_vs_map_twin(data: dict) -> dict:
    """battery's loss as a silent map with one entry per fullYear_roll period (the genuine constant-vs-map
    equivalence: a constant applies in every period, so its equivalent map must too -- a single-period key
    would be right only in the rolls that hold that one period and silently zero in the others), plus west's
    SAME p2050 map.  battery's own period keys resolve the whole-frame probe directly in every roll
    (independently of the fix), so this twin is the reference value for ``_roll_const_vs_map_case``."""
    lps = _lp_snapshot()
    m = lps._pmap({p: 0.001 for p in _FULLYEAR_ROLL_PERIODS}, index_name="x")
    return lps._overlay(data, name="roll_const_vs_map_twin", base="fullYear_roll", values=[
        ["node", "battery", "self_discharge_loss", m],
        ["node", "west", "self_discharge_loss", lps._pmap({"p2050": 0.1}, index_name="x")]])


# ---------------------------------------------------------------------------
# T168-2 guard: roll_p2030_guard -- critique.md item 1's explicit per-roll presence guard
# ---------------------------------------------------------------------------


def _roll_p2030_guard_case(data: dict) -> dict:
    """west self_discharge_loss = {p2030: 0.1} with index ``x`` (silent) on fullYear_roll.  p2030 IS one of
    fullYear_roll's periods, so (unlike roll_sd/roll_const_vs_map) this does not reach the per-entity
    fallback: the whole-frame probe already resolves it via the union of keys.  Passes before and after; it
    guards that the per-roll resolution gives west's value in exactly the rolls whose window holds p2030."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="roll_p2030_guard", base="fullYear_roll", values=[
        ["node", "west", "self_discharge_loss", lps._pmap({"p2030": 0.1}, index_name="x")]])


def _roll_p2030_guard_twin(data: dict) -> dict:
    """Same map, labelled (``index_name="period"``): unambiguous in every roll, so this is the reference LP
    for ``_roll_p2030_guard_case``."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="roll_p2030_guard_twin", base="fullYear_roll", values=[
        ["node", "west", "self_discharge_loss", lps._pmap({"p2030": 0.1}, index_name="period")]])


# ---------------------------------------------------------------------------
# T168-2: stoch_eff / stoch_const_vs_map -- 2_day_stochastic_dispatch (stochastics.json)
# ---------------------------------------------------------------------------


def _stoch_eff_case(data: dict) -> dict:
    """coal_plant.efficiency = {p_none: 0.5} with index ``x`` (silent, no key in the solve); gas_plant,
    hydro_plant and wind_plant keep their authored scalars.  coal_plant contributes the only index column to
    the whole-frame probe and p_none matches neither a period nor a timestep, so before the fix EVERY unit's
    efficiency (not just coal_plant's) falls back to the schema default 1.0 -- verified: p_slope was 1.0 for
    all four units on the parent export, 63733518.8496 -> 74438732.9461 objective."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="stoch_eff", base="2_day_stochastic_dispatch", values=[
        ["unit", "coal_plant", "efficiency", lps._pmap({"p_none": 0.5}, index_name="x")]])


def _stoch_eff_twin(data: dict) -> dict:
    """coal_plant's efficiency row removed entirely (not just overridden back to its authored 0.9): the twin
    entity has no value at all in the solve, which is the actual contract the silent no-key case must match
    (critique.md item 2 -- "the scenario as is" keeps 0.9, which would make this twin red AFTER the fix)."""
    data = dict(data)
    data["parameter_values"] = [
        v for v in data["parameter_values"]
        if not (v[0] == "unit" and v[1] == "coal_plant" and v[2] == "efficiency" and v[4] == "system")]
    return _lp_snapshot()._overlay(data, name="stoch_eff_twin", base="2_day_stochastic_dispatch", values=[])


def _stoch_const_vs_map_case(data: dict) -> dict:
    """coal_plant.efficiency = {p_none: 0.5} (silent, no key -- same patch as ``_stoch_eff_case``, under its
    own scenario name); gas_plant keeps its authored constant 0.9.  Same ambiguity mechanism as
    ``_stoch_eff_case``: gas_plant's constant contributes nothing to the whole-frame probe, so coal_plant's
    lone unmatched key leaves it ambiguous and only the per-entity fallback saves gas_plant's (and hydro's
    and wind's) own value."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="stoch_const_vs_map", base="2_day_stochastic_dispatch", values=[
        ["unit", "coal_plant", "efficiency", lps._pmap({"p_none": 0.5}, index_name="x")]])


def _stoch_const_vs_map_twin(data: dict) -> dict:
    """gas_plant.efficiency as a silent map keyed by the solve's own anchor period (``period1``, not itself a
    branch -- :func:`_with_branch_periods` / ``inherit_active_branch_periods`` copies the row to every branch
    period), plus coal_plant's SAME no-key map.  gas_plant's own anchor key resolves the whole-frame probe
    directly (period1 is a genuine period, in ``dt`` on every verified stochastic fixture), independently of
    the fix, so this twin is the reference value for ``_stoch_const_vs_map_case``."""
    lps = _lp_snapshot()
    return lps._overlay(data, name="stoch_const_vs_map_twin", base="2_day_stochastic_dispatch", values=[
        ["unit", "gas_plant", "efficiency", lps._pmap({"period1": 0.9}, index_name="x")],
        ["unit", "coal_plant", "efficiency", lps._pmap({"p_none": 0.5}, index_name="x")]])


# (case patch, case scenario, twin patch, twin scenario, json path); a patch is a name in lp_snapshot.PATCHES
# (the S4-0 LP fixture of the same name) or a local function.
CASES = [
    pytest.param("pen_silent", "pen_silent", _pen_twin, "pen_twin", TESTS_JSON, id="pen-item167"),
    pytest.param("avail_silent", "avail_silent", _avail_twin, "avail_twin", TESTS_JSON, id="avail-item168"),
    pytest.param("varcost_silent", "varcost_silent", _varcost_twin, "varcost_twin", TESTS_JSON,
                 id="varcost-item168"),
    pytest.param("ladder_ann_depth2", "ladder_ann_depth2", _no_patch, "coal_ladder_annual", TESTS_JSON,
                 id="ladder-item169"),
    pytest.param(_sd_case, "sd_silent", _no_patch, "coal_wind_ev", TESTS_JSON, id="self_discharge-item168"),
    pytest.param(_sd_case, "sd_silent", _sd_lab_twin, "sd_lab_twin", TESTS_JSON, id="sd_lab"),
    pytest.param(_sd_const_vs_map_case, "sd_const_vs_map", _sd_const_vs_map_twin, "sd_const_vs_map_twin",
                 TESTS_JSON, id="sd_const_vs_map"),
    pytest.param(_roll_sd_case, "roll_sd", _roll_sd_twin, "roll_sd_twin", TESTS_JSON, id="roll_sd",
                 marks=pytest.mark.nightly_heavy),
    pytest.param(_roll_const_vs_map_case, "roll_const_vs_map", _roll_const_vs_map_twin,
                 "roll_const_vs_map_twin", TESTS_JSON, id="roll_const_vs_map",
                 marks=pytest.mark.nightly_heavy),
    pytest.param(_roll_p2030_guard_case, "roll_p2030_guard", _roll_p2030_guard_twin, "roll_p2030_guard_twin",
                 TESTS_JSON, id="roll_p2030_guard",
                 marks=pytest.mark.nightly_heavy),
    pytest.param(_stoch_eff_case, "stoch_eff", _stoch_eff_twin, "stoch_eff_twin", STOCH_JSON, id="stoch_eff"),
    pytest.param(_stoch_const_vs_map_case, "stoch_const_vs_map", _stoch_const_vs_map_twin,
                 "stoch_const_vs_map_twin", STOCH_JSON, id="stoch_const_vs_map"),
]


def _build_and_run(tmp_path_factory, patch, scenario: str,
                   json_path: Path = TESTS_JSON) -> tuple[list[tuple[str, Path, float]], list[float]]:
    """Build the DB from ``json_path`` + ``patch``, run the chain cold, and return every built Problem as
    (solve, MPS path, objective offset) in build order plus each step's objective."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    lps = _lp_snapshot()
    fn = lps.PATCHES[patch] if isinstance(patch, str) else patch
    work = tmp_path_factory.mktemp(scenario)
    (work / "f.json").write_text(json.dumps(fn(json.loads(json_path.read_text()))))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)

    built: list[tuple[str, Path, float]] = []
    original = core.build_flextool

    def capturing(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        solve = lps._solve_name_from_stack()
        path = work / f"{len(built):03d}_{lps._safe(solve)}.mps"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the objective offset is kept separately
            m.write_mps(path, emit_names=True)
        built.append((solve, path, float(m._obj_offset)))
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, "build_flextool", capturing)
        # keep_solutions=True: a rolling solve only keeps its LAST roll's solution otherwise, and every
        # step's objective is asserted below (T168-2's rolling cases have 72 steps).
        steps = run_chain_from_db(url, scenario, work_folder=work / "w", warm=False, keep_solutions=True)
    assert built, f"{scenario}: no Problem was built"
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
    return built, [float(s.solution.obj) for s in steps.values()]


@pytest.mark.parametrize("case_patch, case_scenario, twin_patch, twin_scenario, json_path", CASES)
def test_silent_label_case_builds_the_twin_lp(tmp_path_factory, case_patch, case_scenario, twin_patch,
                                              twin_scenario, json_path):
    lps = _lp_snapshot()
    case_lp, case_obj = _build_and_run(tmp_path_factory, case_patch, case_scenario, json_path)
    twin_lp, twin_obj = _build_and_run(tmp_path_factory, twin_patch, twin_scenario, json_path)

    assert [s for s, _, _ in case_lp] == [s for s, _, _ in twin_lp]
    diffs = {}
    for (solve, pa, off_a), (_, pb, off_b) in zip(case_lp, twin_lp):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), _RTOL_LP, off_a, off_b)
        if msgs:
            diffs[solve] = msgs
    assert not diffs, f"case LP differs from its twin: {diffs}"
    assert case_obj == pytest.approx(twin_obj, rel=_RTOL_OBJ)


# ---------------------------------------------------------------------------
# roll_p2030_guard's FlexData-level presence check (critique.md item 1)
# ---------------------------------------------------------------------------


def test_roll_p2030_present_only_in_the_rolls_holding_it(tmp_path_factory):
    """west's self_discharge_loss (keyed by p2030, silent) resolves to a value only in the rolls whose window
    holds p2030, and to nothing in every other roll -- the per-roll re-evaluation (critique.md item 6) must
    see a different drop set per roll, not a single answer memoised for the whole solve.  Verified
    empirically: rolls 35-53 of fullYear_roll's 72 hold p2030; the rest don't."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = _roll_p2030_guard_case(json.loads(TESTS_JSON.read_text()))
    work = tmp_path_factory.mktemp("roll_p2030_presence")
    (work / "f.json").write_text(json.dumps(data))
    url = json_to_db(work / "f.json", work / "f.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, "roll_p2030_guard", work_folder=work / "w", warm=False,
                              keep_solutions=True)

    with_p2030, without_p2030 = [], []
    for solve, step in steps.items():
        dt = step.flex_data.dt
        dt_df = dt.collect() if hasattr(dt, "collect") else dt
        holds_p2030 = "p2030" in dt_df["d"].cast(pl.Utf8).to_list()
        loss = step.flex_data.p_state_self_discharge
        loss_df = loss.frame if loss is not None else pl.DataFrame(schema={"n": pl.Utf8})
        loss_df = loss_df.collect() if hasattr(loss_df, "collect") else loss_df
        west_present = loss_df.filter(pl.col("n").cast(pl.Utf8) == "west").height > 0
        (with_p2030 if holds_p2030 else without_p2030).append(west_present)

    assert with_p2030, "no roll of fullYear_roll holds p2030 -- the scenario or the fixture changed"
    assert all(with_p2030), f"west should be present in every roll holding p2030: {with_p2030}"
    assert not any(without_p2030), f"west should be absent from every roll NOT holding p2030: {without_p2030}"
