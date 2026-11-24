"""The label-dispatch sites of a realization roll R_k (design rev 7.1 §2.2 S1-S9, §2.7 S13), through the seam.

The realization roll is wired (``_run_model._roll_tail_plans``): a rolling ``within_solve`` solve gets no
realization solve, each of its rolls is a realization roll.  The split-roll tests here stop at the first build
and check the data the build reads (FlexData and ``solve_data``); ``test_split_roll_lp.py`` builds and solves split
rolls end to end.  Rolls whose horizon is their jump (``deterministic`` plans: no label with steps) build and solve
end to end and pass G1-G4.

Model (``test_deterministic_realization._spec``): one node ``elec`` (demand 100 MW), ``gen`` (10 EUR/MWh) and ``wind``
(100 MW, upper-limit profile ``wp``, stochastic), one period ``p1`` on 1 h steps t01..t06.  Forecasts ``high``
(weight 1, CF 0.9) and ``low`` (weight 3, CF 0.05); the realized branch ``real`` made at analysis time ``at`` is
REAL[t] + 0.01 * position of ``at`` (REAL = 0.3, 0.5, 0.2, 0.6, 0.4, 0.1), so a value tells which series was read.
Every expected value below is derived by hand from these rules.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from tests.engine import _solve_copy_models as scm
from tests.engine import test_deterministic_realization as tdr

STEPS = tdr.STEPS
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class _Stop(BaseException):
    """Ends the run at the first build (not an Exception, so no handler of the run takes it)."""


def _live_plan(monkeypatch) -> None:
    """The realization roll is wired (``_run_model._roll_tail_plans``): only the strict autoscale check is set."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _spec(*, real_at: dict, forecast_at: dict, rolling=(2.0, 4.0), start: str = "t01") -> dict:
    """The model with ``dispatch_realization_method`` ``within_solve``: ``real`` made at the analysis times of
    *real_at* and the forecasts at those of *forecast_at* (``{analysis time: steps it covers}``)."""
    pack, pmap, nested = scm.pack, scm.pmap, scm.nested
    spec = tdr._spec(stochastic=True, rolling=rolling, method="within_solve", start=start)
    branches = [(b, nested([(at, nested([(t, v) for t in steps])) for at, steps in forecast_at.items()]))
                for b, v in tdr.FORECAST.items()]
    branches.append(("real", nested([(at, nested([(t, tdr._real(at, t)) for t in steps]))
                                     for at, steps in real_at.items()])))
    rows = [("high", 1.0, "no", forecast_at), ("low", 3.0, "no", forecast_at), ("real", 0.0, "yes", real_at)]
    fan = pmap([("p1", nested([(b, nested([(at, nested([(yes, w)])) for at in times]))
                               for b, w, yes, times in rows]))])
    for row in spec["parameter_values"]:
        if row[:3] == ["profile", "wp", "profile"]:
            row[3] = pack(pmap(branches), "map")
        elif row[:3] == ["solve", "st", "stochastic_branches"]:
            row[3] = pack(fan, "map")
    return spec


def _first_build(spec: dict, folder: Path, monkeypatch):
    """Run *spec* with the live plan up to its first build: ``(FlexData, {solve_data frame name: frame})``."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    _live_plan(monkeypatch)
    puts: dict[int, dict[str, pl.DataFrame]] = {}
    put = FlexDataProvider.put

    def recording(self, name, frame, **kw):
        puts.setdefault(id(self), {})[str(name).removesuffix(".csv")] = frame
        return put(self, name, frame, **kw)

    seen: dict = {}

    def capture(m, d, *args, **kwargs):
        seen["d"] = d
        raise _Stop

    monkeypatch.setattr(FlexDataProvider, "put", recording)
    monkeypatch.setattr(core, "build_flextool", capture)
    folder.mkdir(parents=True, exist_ok=True)
    url = scm.build_db(scm.write_spec(spec, folder), folder)
    with pytest.raises(_Stop):
        run_chain_from_db(url, "base", work_folder=folder / "w", warm=False)
    d = seen["d"]
    cells = set(_rows(d.dt, "d", "t"))
    frames = next(f for f in puts.values()
                  if "solve_data/steps_in_use" in f and {r[:2] for r in _rows(f["solve_data/steps_in_use"])} == cells)
    return d, {k.removeprefix("solve_data/"): v for k, v in frames.items() if k.startswith("solve_data/")}


def _rows(frame, *cols) -> list[tuple]:
    if frame is None:
        return []
    cols = cols or tuple(frame.columns)
    return [tuple(str(v) for v in r) for r in frame.select(pl.col(c).cast(pl.Utf8) for c in cols).rows()]


def _values(d, field: str, *cols) -> dict:
    frame = getattr(d, field)
    frame = frame if isinstance(frame, pl.DataFrame) else frame.frame
    return {tuple(str(v) for v in r[:-1]): r[-1] for r in frame.select(*cols, "value").rows()}


# ---------------------------------------------------------------------------
# A split roll: roll 0 of jump 2 / horizon 4 (jump t01-t02, tail from t03).
# ---------------------------------------------------------------------------

#: The realized series and the forecasts made at t01, t03 and t05, each up to the end of the timeline.
OVERLAP = tdr.OVERLAP


def test_split_roll_reads_its_jump_on_the_realized_branch_and_its_tail_on_the_forecasts(tmp_path, monkeypatch):
    """T-r1, T-leaf, S1-S3, S5-S7.  Roll 0 realizes t01-t02 under the period's own name ``p1`` and looks ahead from
    t03 with the forecasts made there (``p1_high`` / ``p1_low`` hold t03-t04)."""
    d, sd = _first_build(_spec(real_at=OVERLAP, forecast_at=OVERLAP), tmp_path, monkeypatch)
    # S1: the label keeps its jump steps, the forecasts get the rest of the period.
    assert _rows(d.dt, "d", "t") == [("p1", "t01"), ("p1", "t02"), ("p1_high", "t03"), ("p1_high", "t04"),
                                     ("p1_low", "t03"), ("p1_low", "t04")]
    # S3: the label's cells are a cohort of one (weight 1); the forecasts 1 / (1 + 3) and 3 / (1 + 3).  The
    # period-level weight of the label stays 0 (its forecasts carry the period).
    assert _values(d, "pdt_branch_weight", "d", "t") == {
        ("p1", "t01"): 1.0, ("p1", "t02"): 1.0, ("p1_high", "t03"): 0.25, ("p1_high", "t04"): 0.25,
        ("p1_low", "t03"): 0.75, ("p1_low", "t04"): 0.75}
    assert _values(d, "pd_branch_weight", "d")[("p1",)] == 0.0
    # S7 (T-r1): the jump reads the realized series nearest to the roll start t01 (made at t01: REAL), the tails
    # the forecasts made at t03 (0.9 / 0.05).
    wind = _values(d, "p_profile_value", "f", "d", "t")
    assert [wind[("wp", "p1", t)] for t in ("t01", "t02")] == [0.3, 0.5]
    assert [wind[("wp", f, t)] for f in ("p1_high", "p1_low") for t in ("t03", "t04")] == [0.9, 0.9, 0.05, 0.05]
    # 3.8: the label reads the realized branch through one flat tier per analysis time (nearest to t01 first:
    # t01, t03, t05), the tails the forecasts made at t03.
    assert _rows(sd["branch_time_start"]) == [
        ("p1", "t01", "flat"), ("p1", "t03", "flat"), ("p1", "t05", "flat"),
        ("p1_high", "t03", "fan"), ("p1_low", "t03", "fan")]
    assert sorted(_rows(sd["solve_branch__time_branch"])) == [("p1", "real"), ("p1_high", "high"),
                                                               ("p1_low", "low")]
    # S2 (G4's link): each tail continues from the label's last jump step, one step on.
    links = {(r[0], r[1]): r[2:] for r in _rows(d.dtttdt, "d", "t", "d_previous", "t_previous_within_solve")}
    assert links[("p1_high", "t03")] == links[("p1_low", "t03")] == ("p1", "t02")
    assert ("p1_high", "t03", "p1", "t02") in _rows(d.dt_ramp_link, "d", "t", "d_previous", "t_previous")
    # S5: the label is the first period of the roll, its tails are no roots.
    assert _rows(sd["period_first_of_solve"]) == [("p1",)]
    # S6: the label's realized cells are its own: nothing is committed inside the fan or tied across forecasts.
    assert _rows(sd["committed_in_fan"]) == []
    assert _rows(sd["dt_non_anticipativity_set"]) == []
    assert _rows(d.realized_dispatch) == [("p1", "t01"), ("p1", "t02")]
    # T-leaf: each leaf is the label and its own forecast period; no anchor leaf.
    assert sorted(_rows(d.d_leaf, "d", "leaf")) == [("p1", "high"), ("p1", "low"), ("p1_high", "high"),
                                                    ("p1_low", "low")]


def test_split_roll_reads_a_realized_series_made_before_the_roll(tmp_path, monkeypatch):
    """T-r2, the D7 case: ``real`` is made once, at t01 (it covers t01..t06); the forecasts at t03 and t05.  The
    timeset starts at t03, so roll 0 realizes t03-t04 and looks ahead from t05.  Its jump reads the t01 series
    (REAL: 0.2 at t03, 0.6 at t04), its tails the forecasts made at t05."""
    forecasts = {at: STEPS[STEPS.index(at):] for at in ("t03", "t05")}
    d, sd = _first_build(_spec(real_at={"t01": STEPS}, forecast_at=forecasts, start="t03"), tmp_path, monkeypatch)
    wind = _values(d, "p_profile_value", "f", "d", "t")
    assert [wind[("wp", "p1", t)] for t in ("t03", "t04")] == [0.2, 0.6]
    assert [wind[("wp", f, t)] for f in ("p1_high", "p1_low") for t in ("t05", "t06")] == [0.9, 0.9, 0.05, 0.05]
    assert _rows(sd["branch_time_start"]) == [("p1", "t01", "flat"), ("p1_high", "t05", "fan"),
                                              ("p1_low", "t05", "fan")]
    assert ("p1", "real") in _rows(sd["solve_branch__time_branch"])


def test_split_roll_gap_in_the_realized_series(tmp_path, monkeypatch):
    """S7 coverage: the only realized series (made at t01) covers t01..t03, so the jump step t04 of roll 0 (t03-t04)
    has no realized value: the input error names the profile, the step and the analysis time."""
    forecasts = {at: STEPS[STEPS.index(at):] for at in ("t03", "t05")}
    spec = _spec(real_at={"t01": STEPS[:3]}, forecast_at=forecasts, start="t03")
    with pytest.raises(FlexToolConfigError, match=(
            r"profile 'wp', parameter 'profile': solve '\w+' has stochastic_branches and dispatch_realization_method "
            r"deterministic, or realizes a jump of a rolling within_solve or separate_dispatch solve, .*"
            r"No realized series covers time step 't04' \(period 'p1'\); the realized series start at analysis_time "
            r"'t01'")):
        _first_build(spec, tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Rolls without a tail (horizon = jump): every roll is a deterministic plan; the LP builds and passes G1-G4.
# ---------------------------------------------------------------------------

def test_rolls_without_tail_read_the_realized_branch_flattened(tmp_path_factory, monkeypatch):
    """W4 through the plan: jump 2 / horizon 2, the realized series and forecasts made at t01, t03, t05 (each to the
    end).  Roll k reads the series made at its own start (nearest): t01 for t01-t02, t03 for t03-t04, t05 for
    t05-t06, so roll k builds the LP of the model written without stochastic_branches with that series as a plain
    profile (the oracle: one plain run per series, roll k compared byte for byte)."""
    _live_plan(monkeypatch)
    case, _ = scm.run_spec(_spec(real_at=OVERLAP, forecast_at=OVERLAP, rolling=(2.0, 2.0)),
                           tmp_path_factory.mktemp("case"))
    assert [s for s, _, _ in case] == ["st", "st", "st"]
    for k, at in enumerate(("t01", "t03", "t05")):
        oracle, _ = scm.run_spec(tdr._spec(stochastic=False, rolling=(2.0, 2.0), plain=tdr._made_at(at)),
                                 tmp_path_factory.mktemp(f"oracle_{at}"))
        assert not scm.lp_differences([case[k]], [oracle[k]], rtol=0.0), at
        assert case[k][1].read_bytes() == oracle[k][1].read_bytes(), at


# ---------------------------------------------------------------------------
# S13: the separate_dispatch realization roll.
# ---------------------------------------------------------------------------

def _rsd_horizon_is_jump(folder: Path) -> Path:
    """``rolling_separate_dispatch.json`` with ``rolling_solve_horizon`` = its jump (24 h): each realization roll's
    plan is ``deterministic``, so it builds without a label with steps."""
    data = json.loads((FIXTURES / "rolling_separate_dispatch.json").read_text(encoding="utf-8"))
    for row in data["parameter_values"]:
        if row[2] == "rolling_solve_horizon":
            row[3] = scm.pack(24.0, "float")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "rsd.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_separate_dispatch_realization_roll_reads_its_plan(tmp_path_factory, monkeypatch):
    """T-s13 (§2.7 items 2-3): a realization roll of ``separate_dispatch`` reads its realized branch from its
    plan (3.8: ``realized`` through flat tiers nearest to the UC roll's start), not from the W4c projection of its
    UC roll.  ``realized`` is made at t0001, t0025 and t0049 and covers each roll's jump, so the LPs equal today's
    (the W4c projection reads the same series, made at the roll start)."""
    from flextool.engine.solve import _run_model

    frames: dict[str, list] = {}
    emit = _run_model.solve_writers.emit_branch_time_start

    def recording(rows, **kw):
        frames.setdefault("rows", []).append(list(rows))
        return emit(rows, **kw)

    path = _rsd_horizon_is_jump(tmp_path_factory.mktemp("rsd"))
    with monkeypatch.context() as unplanned:
        # Today's realization rolls: no plan (the W4c projection of the UC roll; horizon = jump, so the window is
        # the jump either way).
        unplanned.setattr(_run_model, "_roll_tail_plans", lambda *a: {})
        today, _ = scm.run_capturing(scm.build_db(path, tmp_path_factory.mktemp("today")),
                                     "rolling_separate_dispatch", tmp_path_factory.mktemp("today_run"))
    _live_plan(monkeypatch)
    monkeypatch.setattr(_run_model.solve_writers, "emit_branch_time_start", recording)
    planned, _ = scm.run_capturing(scm.build_db(path, tmp_path_factory.mktemp("planned")),
                                   "rolling_separate_dispatch", tmp_path_factory.mktemp("planned_run"))
    assert [s for s, _, _ in planned] == [s for s, _, _ in today]
    assert not scm.lp_differences(planned, today, rtol=0.0)
    # The realization rolls (every second sub-solve, after its UC roll) read flat tiers by distance from the UC
    # roll's start (24 h apart, the earlier one first on a tie).
    flat = [rows for rows in frames["rows"] if rows and rows[0][2:] == ("flat",)]
    assert flat == [
        [("period1", t, "flat") for t in ("t0001", "t0025", "t0049")],
        [("period1", t, "flat") for t in ("t0025", "t0001", "t0049")],
        [("period1", t, "flat") for t in ("t0049", "t0025", "t0001")]]


def test_label_first_step_starts_no_cohort():
    """S7: a split roll's label reads the realized branch flattened, so the analysis times of its own rows (its first
    step t01, the flat tiers) start no cohort: a value that only ``high`` authors at t01 is not a partial cohort;
    at the jump end t03, where the forecasts start, it is (``low`` is missing)."""
    from flextool.engine.preprocess.emit._pdt_lookup import FanStarts, _check_pbt_branch_coverage

    tb = {"p1": ["real"], "p1_high": ["high"], "p1_low": ["low"]}
    ts = {"p1": ["t01"], "p1_high": ["t03"], "p1_low": ["t03"]}
    fan = FanStarts({"p1": ["t01", "t03"], "p1_high": ["t03"], "p1_low": ["t03"]},
                    fan_first=frozenset({"p1_high", "p1_low"}), flat=frozenset({"p1"}))
    members = {"p1": ["p1_high", "p1_low"]}
    _check_pbt_branch_coverage({("wp", "profile", "high", "t01", "t01"): 0.9}, 1, members, tb, ts, fan,
                               kind="profile")
    with pytest.raises(FlexToolConfigError, match=r"start at time step 't03' have values for \['high'\] but are "
                                                  r"missing \['low'\]"):
        _check_pbt_branch_coverage({("wp", "profile", "high", "t03", "t03"): 0.9}, 1, members, tb, ts, fan,
                                   kind="profile")
