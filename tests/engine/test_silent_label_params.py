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
  in the solve instead of the whole parameter).
* 169: the depth-2 annual price ladder (tier -> {price, quantity}) is not read DB-side (``ladder``).

Today the emit seed masks items 167 and 169 (conditional writes), so those pairs hold; the comparison is the
whole LP (every MPS of the chain, canonical up to row/column order) plus the objective.  Item 168's
unconditional-write field (``sd``) was wrong before the per-entity fallback and is now a guard.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from tests.engine.test_cyclic_fix_start import _lp_snapshot

REPO = Path(__file__).resolve().parents[2]
TESTS_JSON = REPO / "tests" / "fixtures" / "tests.json"
_RTOL_LP = 1e-12
_RTOL_OBJ = 1e-9


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


# (case patch, case scenario, twin patch, twin scenario); a patch is a name in lp_snapshot.PATCHES (the S4-0 LP
# fixture of the same name) or a local function.
CASES = [
    pytest.param("pen_silent", "pen_silent", _pen_twin, "pen_twin", id="pen-item167"),
    pytest.param("avail_silent", "avail_silent", _avail_twin, "avail_twin", id="avail-item168"),
    pytest.param("varcost_silent", "varcost_silent", _varcost_twin, "varcost_twin", id="varcost-item168"),
    pytest.param("ladder_ann_depth2", "ladder_ann_depth2", _no_patch, "coal_ladder_annual", id="ladder-item169"),
    pytest.param(_sd_case, "sd_silent", _no_patch, "coal_wind_ev", id="self_discharge-item168"),
]


def _build_and_run(tmp_path_factory, patch, scenario: str) -> tuple[list[tuple[str, Path, float]], list[float]]:
    """Build the DB from tests.json + ``patch``, run the chain cold, and return every built Problem as
    (solve, MPS path, objective offset) in build order plus each step's objective."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    lps = _lp_snapshot()
    fn = lps.PATCHES[patch] if isinstance(patch, str) else patch
    work = tmp_path_factory.mktemp(scenario)
    (work / "f.json").write_text(json.dumps(fn(json.loads(TESTS_JSON.read_text()))))
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
        steps = run_chain_from_db(url, scenario, work_folder=work / "w", warm=False)
    assert built, f"{scenario}: no Problem was built"
    assert steps and all(s.solution is not None and s.solution.optimal for s in steps.values())
    return built, [float(s.solution.obj) for s in steps.values()]


@pytest.mark.parametrize("case_patch, case_scenario, twin_patch, twin_scenario", CASES)
def test_silent_label_case_builds_the_twin_lp(tmp_path_factory, case_patch, case_scenario, twin_patch,
                                              twin_scenario):
    lps = _lp_snapshot()
    case_lp, case_obj = _build_and_run(tmp_path_factory, case_patch, case_scenario)
    twin_lp, twin_obj = _build_and_run(tmp_path_factory, twin_patch, twin_scenario)

    assert [s for s, _, _ in case_lp] == [s for s, _, _ in twin_lp]
    diffs = {}
    for (solve, pa, off_a), (_, pb, off_b) in zip(case_lp, twin_lp):
        msgs = lps.canonical_diff(lps.parse_mps(pa), lps.parse_mps(pb), _RTOL_LP, off_a, off_b)
        if msgs:
            diffs[solve] = msgs
    assert not diffs, f"case LP differs from its twin: {diffs}"
    assert case_obj == pytest.approx(twin_obj, rel=_RTOL_OBJ)
