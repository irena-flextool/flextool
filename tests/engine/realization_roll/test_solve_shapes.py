"""Solve shapes around the realization roll (design rev 7.1 §8.2 T-seam / T-w5, §8.3 T-sd2).

Model (``_spec``): T-w1's economics (``test_split_roll_lp._tw1``) on two periods of one timeline: node ``elec``
(demand 1 MW, in the stochastic group) with 1 MWh of lossless storage (fix_start 0, free end, bind_forward_only),
``gas`` 2 MW at 10 EUR/MWh in ``p1`` and 30 in ``p2``, ``peak`` 10 MW at 30, ``wind`` 1 MW on profile ``wp``: realized
``R`` 0 everywhere; forecasts ``A`` (wind 1) and ``B`` (wind 0) made at the first step of ``p2``, weights 0.5 / 0.5.
``years_represented`` 1 per period.  Storing 1 MWh at 10 in ``p1`` saves 30 under ``B`` only: 0.5 x 30 = 15 > 10.
"""
from __future__ import annotations

import pytest

from tests.engine import _solve_copy_models as scm
from tests.engine.realization_roll import test_split_roll_lp as srl

pack, pmap, nested = scm.pack, scm.pmap, scm.nested
pytestmark = pytest.mark.solver


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _fan(rows) -> list:
    """``stochastic_branches``: *rows* ``(period, branch, {analysis time: weight}, realized)``."""
    periods: dict = {}
    for p, b, times, yes in rows:
        periods.setdefault(p, []).append((b, nested([(at, nested([(yes, w)])) for at, w in times.items()])))
    return pack(pmap([(p, nested(branches)) for p, branches in periods.items()]), "map")


def _spec(periods: dict[str, tuple[str, ...]], *, mode: str, method: str, realized: list[str], fan: list,
          wind: dict, gas: dict, jump: int | None = None, horizon: int | None = None,
          solves: tuple[str, ...] = ("st",)) -> dict:
    """*periods* ``{period: its 1 h steps}`` (one timeset each), solve ``st`` (*mode*, *method*, *realized* periods,
    *fan*), profile ``wp`` = *wind* (``{branch: {analysis time: {step: value}}}``), gas cost *gas* ``{step: EUR}``."""
    from flextool.update_flextool.fixture_schema import schema_sections

    steps = [t for ts in periods.values() for t in ts]
    spec = schema_sections("packed")
    spec["alternatives"] = [["init", ""]]
    spec["scenarios"] = [["base", False, ""]]
    spec["scenario_alternatives"] = [["base", "init", None]]
    entities = [["node", "elec", None], ["unit", "gas", None], ["unit", "peak", None], ["unit", "wind", None],
                ["profile", "wp", None], ["unit__outputNode", ["gas", "elec"], None],
                ["unit__outputNode", ["peak", "elec"], None], ["unit__outputNode", ["wind", "elec"], None],
                ["group__unit", ["sg", "wind"], None], ["group__node", ["sg", "elec"], None],
                ["unit__node__profile", ["wind", "elec", "wp"], None]]
    spec["entities"] = [["model", "flextool", None], ["timeline", "tl", None], ["solve", "st", None],
                        ["group", "sg", None], *(["timeset", f"ts_{p}", None] for p in periods), *entities]
    spec["entity_alternatives"] = [[c, [n], "init", True] for c, n in (
        ("node", "elec"), ("unit", "gas"), ("unit", "peak"), ("unit", "wind"))]
    solve = [("solve_mode", pack(mode, "str")),
             ("period_timeset", pack(pmap([(p, f"ts_{p}") for p in periods], index_name="period"), "map")),
             ("realized_periods", scm.array(realized)),
             ("years_represented", pack(pmap([(p, 1.0) for p in periods], index_name="period"), "map")),
             ("dispatch_realization_method", pack(method, "str")),
             ("stochastic_branches", fan)]
    if jump is not None:
        solve += [("rolling_solve_jump", pack(float(jump), "float")),
                  ("rolling_solve_horizon", pack(float(horizon), "float"))]
    spec["parameter_values"] = [
        ["model", "flextool", "solves", scm.array(list(solves)), "init"],
        ["timeline", "tl", "timestep_duration", pack(pmap([(t, 1.0) for t in steps], index_name="time"), "map"),
         "init"],
        *(row for p, ts in periods.items() for row in (
            ["timeset", f"ts_{p}", "timeline", pack("tl", "str"), "init"],
            ["timeset", f"ts_{p}", "timeset_duration", pack(pmap([(ts[0], float(len(ts)))], index_name="time"),
                                                            "map"), "init"])),
        ["group", "sg", "include_stochastics", pack("yes", "str"), "init"],
        *(["solve", "st", name, value, "init"] for name, value in solve),
        *srl._node("elec", "storage", pack(-1.0, "float")),
        ["node", "elec", "existing", pack(1.0, "float"), "init"],
        ["node", "elec", "storage_start_end_method", pack("fix_start", "str"), "init"],
        ["node", "elec", "storage_state_start", pack(0.0, "float"), "init"],
        *srl._unit("gas", "elec", 2.0, gas), *srl._unit("peak", "elec", 10.0, 30.0),
        *srl._unit("wind", "elec", 1.0, 0.0),
        ["unit__node__profile", ["wind", "elec", "wp"], "profile_method", pack("upper_limit", "str"), "init"],
        ["profile", "wp", "profile", srl._series(wind), "init"]]
    return spec


# ---------------------------------------------------------------------------
# T-seam: the jump is one whole period; the forecasts start at the next period's first step.
# ---------------------------------------------------------------------------

SEAM_PERIODS = {"p1": ("t01", "t02"), "p2": ("t03", "t04")}
SEAM_WIND = {"R": {"t01": {t: 0.0 for t in ("t01", "t02", "t03", "t04")}, "t03": {"t03": 0.0, "t04": 0.0}},
             "A": {"t03": {"t03": 1.0, "t04": 1.0}}, "B": {"t03": {"t03": 0.0, "t04": 0.0}}}
SEAM_GAS = {"t01": 10.0, "t02": 10.0, "t03": 30.0, "t04": 30.0}
SEAM_ROWS = [("p2", "R", {"t03": 0.0}, "yes"), ("p2", "A", {"t03": 0.5}, "no"), ("p2", "B", {"t03": 0.5}, "no")]
#: The rolling solve reads the jump p1 on R by its realized rows (D7): R is designated at p1's start too.  The
#: single solve keeps today's cohort rule (a row at (p1, t01) would be a cohort without forecasts).
SEAM_FAN = _fan([("p1", "R", {"t01": 0.0}, "yes"), *SEAM_ROWS])
SEAM_SINGLE_FAN = _fan(SEAM_ROWS)


def test_seam_jump_of_a_whole_period(tmp_path):
    """T-seam.  Rolling ``within_solve``, jump 2 = p1 (t01-t02), horizon 4: R_0 = p1 on R + ``p2_A`` / ``p2_B``
    (t03-t04, the forecasts made at t03); R_1 = p2 on R (deterministic).  2 solves, no realization solve spawned.
    Factor 8760 / 2 = 4380 per period.  R_0: p1 needs 2 MWh at 10 and stores s <= 1 more (gas 2 MW); under B p2
    needs 2 MWh at 30 less s, under A nothing: 20 + 10s + 0.5 x 30 (2 - s) = 50 - 5s -> s = 1, 45 -> 197 100.  R_1
    starts from s = 1: 1 MWh of gas at 30 -> 131 400.  The hand-written single solve over p1 + p2 with the same
    mid-horizon fan at p2 (p1 realized on R, p2 the forecasts) is the same LP economics: its first build's
    objective is 197 100 too."""
    spec = _spec(SEAM_PERIODS, mode="rolling_window", method="within_solve", realized=["p1", "p2"], fan=SEAM_FAN,
                 wind=SEAM_WIND, gas=SEAM_GAS, jump=2, horizon=4)
    built, steps = scm.run_spec(spec, tmp_path / "rolling")
    assert len(built) == 2 and not any("realized_dispatch" in s for s, _, _ in built)
    assert srl._objectives(steps) == pytest.approx([197_100.0, 131_400.0], rel=1e-9)
    first = list(steps.values())[0]
    assert srl._value(first.solution.value("v_state"), n="elec", d="p1", t="t02") == pytest.approx(1.0, abs=1e-9)
    single = _spec(SEAM_PERIODS, mode="single_solve", method="within_solve", realized=["p1"], fan=SEAM_SINGLE_FAN,
                   wind=SEAM_WIND, gas=SEAM_GAS)
    _built, twin = scm.run_spec(single, tmp_path / "single")
    assert list(twin.values())[0].solution.obj == pytest.approx(197_100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# T-w5: an author-declared realization solve keeps two solves; no realization roll, no error.
# ---------------------------------------------------------------------------

def test_author_declared_realization_keeps_two_solves(tmp_path, monkeypatch):
    """T-w5.  T-w1's single period ``p1`` (t01-t02) in solve ``st`` (rolling ``within_solve``, jump 1, horizon 2,
    forecasts made at t01 and t02) followed in ``model.solves`` by ``real``: the same rolls, no forecasts, realizing p1
    (``_realization.configured_realization_solve``).  The §2.1 gate gives no solve a plan (no realization roll), no
    realization solve is spawned, both solves run their 2 rolls (4 builds), and no W3 error or W-0635 / W-0636
    notice is raised."""
    from flextool.engine.solve import _run_model
    from tests._support.coded_warnings import capture_coded

    wind = {"R": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}},
            "A": {"t01": {"t01": 0.0, "t02": 1.0}, "t02": {"t02": 1.0}},
            "B": {"t01": {"t01": 0.0, "t02": 0.0}, "t02": {"t02": 0.0}}}
    fan = _fan([("p1", "R", {"t01": 0.0, "t02": 0.0}, "yes"), ("p1", "A", {"t01": 0.5, "t02": 0.5}, "no"),
                ("p1", "B", {"t01": 0.5, "t02": 0.5}, "no")])
    spec = _spec({"p1": ("t01", "t02")}, mode="rolling_window", method="within_solve", realized=["p1"], fan=fan,
                 wind=wind, gas={"t01": 10.0, "t02": 30.0}, jump=1, horizon=2, solves=("st", "real"))
    spec["entities"].append(["solve", "real", None])
    spec["parameter_values"] += [
        ["solve", "real", name, value, "init"] for name, value in (
            ("solve_mode", pack("rolling_window", "str")),
            ("period_timeset", pack(pmap([("p1", "ts_p1")], index_name="period"), "map")),
            ("realized_periods", scm.array(["p1"])),
            ("years_represented", pack(pmap([("p1", 1.0)], index_name="period"), "map")),
            ("rolling_solve_jump", pack(1.0, "float")), ("rolling_solve_horizon", pack(2.0, "float")))]
    live = _run_model._roll_tail_plans
    plans: list = []

    def recording(solve_cfg, rolls, timelines):
        plans.append(live(solve_cfg, rolls, timelines))
        return plans[-1]

    monkeypatch.setattr(_run_model, "_roll_tail_plans", recording)
    with capture_coded() as got:
        built, steps = scm.run_spec(spec, tmp_path)
    assert plans and all(p == {} for p in plans), plans
    names = [s for s, _, _ in built]
    assert len(names) == 4 and not any("realized_dispatch" in s for s in names), names
    assert sorted({s.split("_roll")[0] for s in steps}) == ["real", "st"], list(steps)
    assert all(s.solution.optimal for s in steps.values())
    assert not got.names("W-0635") and not got.names("W-0636")


# ---------------------------------------------------------------------------
# T-sd2: a single separate_dispatch solve with a look-ahead period; only solve (1) changes in 4.7.0 (D3).
# ---------------------------------------------------------------------------

SD2_PERIODS = {"p1": ("t01",), "p2": ("t02",)}
SD2_WIND = {"R": {"t01": {"t01": 0.0, "t02": 0.0}},
            "A": {"t01": {"t01": 0.0, "t02": 1.0}}, "B": {"t01": {"t01": 0.0, "t02": 0.0}}}
SD2_FAN = _fan([("p1", "R", {"t01": 0.0}, "yes"), ("p1", "A", {"t01": 0.5}, "no"), ("p1", "B", {"t01": 0.5}, "no")])


def _sd2_run(folder, monkeypatch, *, today: bool):
    from flextool.engine.preprocess.flex_data import _load

    if today:
        # Before D3 solve (1) tied every non-anticipative family across its forecasts (no uc_only role).
        monkeypatch.setattr(_load, "_na_online_fix_in_uc", lambda sd, provider=None: False)
    spec = _spec(SD2_PERIODS, mode="single_solve", method="separate_dispatch", realized=["p1"], fan=SD2_FAN,
                 wind=SD2_WIND, gas={"t01": 10.0, "t02": 30.0})
    return scm.run_spec(spec, folder)


def test_single_separate_dispatch_changes_only_solve_one(tmp_path, monkeypatch):
    """T-sd2.  Single ``separate_dispatch``, p1 (t01) realized, p2 (t02) look-ahead, forecasts made at t01 (A wind 0
    at t01 and 1 at t02, B 0 at both; 0.5 / 0.5).  Factor 8760 per 1 h period.  Solve (1) has no ``_fix_in_uc`` unit,
    so under D3 nothing is tied: A stores nothing (10), B stores 1 MWh at 10 for t02 (20): 15 -> 131 400.  Before D3
    the storage use and state were tied over p1: both store 1 (0.5 x 30 > 10): 20 -> 175 200.  Nothing passes from
    solve (1) to the realization (no pins), so its LP is byte-identical (canonical diff empty), and its objective
    the same."""
    new_built, new_steps = _sd2_run(tmp_path / "new", monkeypatch, today=False)
    with pytest.MonkeyPatch.context() as mp:
        old_built, old_steps = _sd2_run(tmp_path / "old", mp, today=True)
    assert [s for s, _, _ in new_built] == [s for s, _, _ in old_built]
    assert len(new_built) == 2 and "realized_dispatch" in new_built[1][0]
    assert list(new_steps.values())[0].solution.obj == pytest.approx(131_400.0, rel=1e-9)
    assert list(old_steps.values())[0].solution.obj == pytest.approx(175_200.0, rel=1e-9)
    diff = scm.lp_differences(new_built, old_built)
    assert diff.get(new_built[1][0]) in (None, []), diff
    assert new_built[1][1].read_bytes() == old_built[1][1].read_bytes()
    assert diff.get(new_built[0][0]), diff
    assert list(new_steps.values())[1].solution.obj == pytest.approx(list(old_steps.values())[1].solution.obj,
                                                                     rel=1e-12)
