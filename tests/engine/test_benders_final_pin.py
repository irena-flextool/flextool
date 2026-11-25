"""Item 287 P1: the building blocks of the final solve of a Benders solve (``engine/solve/_benders_final.py``).

The final solve is the whole (undecomposed) model of a Benders solve with every investment, retirement and
life-extension decision fixed to the Benders incumbent (:func:`pin_invest_decisions`).  Oracles are relational:

* **T5 pin exact**: the monolithic LP of ``lh2_three_region_trade_invest`` (``decomposition none``) pinned with
  the monolith's **own** optimal decisions has the monolith's objective (rel 1e-9), and its MPS differs from the
  unpinned one only in the bounds of the pinned columns, each bounded to the monolith's value ± ``PIN_BAND``.  A
  missing column, a value for no column, two values for one column, a family the model lacks and values for a
  family without columns raise.
* **m3 band** (review): a value a fraction of the band below a binding investment floor (a row) is infeasible
  fixed exactly and solves to the monolith's objective with the band.  Run with the
  axis Enums off and on, and with a Utf8 value frame against the Enum-typed model.
* **T5b pin coverage**: on each Benders fixture shape, ``BendersResult.invest_solution_vars`` has a value for
  exactly the columns of each pinned family of the undecomposed model, and the model pinned with them solves to
  an objective between the monolith optimum and the monolith optimum plus the Benders gap (the incumbent is a
  feasible plan of the whole model; the Benders lower bound is valid).
* **T6 option**: ``--benders-final-solve`` > ``FLEXTOOL_BENDERS_FINAL_SOLVE`` > ``on``.

Databases are built from the JSON fixtures (``tests/fixtures/lh2_three_region*.json``).
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.solve._benders_final import (
    AGGREGATE_FAMILIES,
    PIN_BAND,
    PINNED_FAMILIES,
    check_final_objective,
    final_solve_on,
    pin_invest_decisions,
)
from flextool.run_options import RunOptions
from tests.engine._unit_db import axis_vocabulary, b64, build, make_db, utf8
from tests.engine.test_chain_costs import _period_map

REPO = Path(__file__).resolve().parents[2]
LH2_JSON = REPO / "tests" / "fixtures" / "lh2_three_region.json"
TRADE_JSON = REPO / "tests" / "fixtures" / "lh2_three_region_trade_invest.json"
TRADE_CHAIN = ["lh2_three_region", "lh2_three_region_trade_invest"]
INVEST_CHAIN = ["lh2_three_region", "lh2_three_region_invest"]
REGIONS = ["region_A", "region_B", "region_C"]


def _none(solve: str) -> list:
    return [["solve", solve, "decomposition", b64("none", "str")]]


def _run(tmp_path_factory, *, alt: str, values: list, base_chain: list[str], fixture_json: Path) -> dict:
    from flextool.engine import run_chain_from_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=base_chain, fixture_json=fixture_json)
    return run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"), keep_solutions=True)


def _solve(pb):
    sol = pb.solve()
    assert sol.optimal
    return sol


def _pinned_present(pb) -> list[str]:
    return [n for n in PINNED_FAMILIES if n in pb._vars and pb._vars[n].frame.height > 0]


# ---------------------------------------------------------------------------------------------------------------
# T5: the monolith pinned with its own decisions
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=["utf8", "enums"])
def mono(request, tmp_path_factory):
    """``(flex_data, own decision frames, objective)`` of the monolithic trade-invest solve, axis Enums off / on."""
    with pytest.MonkeyPatch.context() as mp:
        if request.param == "enums":
            mp.setenv("FLEXTOOL_AXIS_ENUMS", "1")
        else:
            mp.delenv("FLEXTOOL_AXIS_ENUMS", raising=False)
        steps = _run(tmp_path_factory, alt=f"p287_mono_{request.param}", values=_none("lh2_trade_invest"),
                     base_chain=TRADE_CHAIN, fixture_json=TRADE_JSON)
    step = steps["lh2_trade_invest"]
    assert not step.is_benders
    fd = step.flex_data
    assert (fd._axis_enums is not None) == (request.param == "enums")
    pb = build(fd)
    with axis_vocabulary(fd):
        sol = _solve(pb)
        frames = {n: sol.value(n) for n in _pinned_present(pb)}
    assert set(frames) == {"v_invest_p"}  # the fixture invests in the two pipes only
    # pipe_AB is built, pipe_BC is not (test_benders_invest_handoff): one nonzero and one zero value are pinned.
    built = {r["p"]: r["value"] for r in utf8(frames["v_invest_p"]).iter_rows(named=True)}
    assert built["pipe_AB"] > 1e-6 and abs(built["pipe_BC"]) <= 1e-9
    return fd, frames, float(sol.obj), request.param


def _pinned_problem(fd, frames):
    pb = build(fd)
    with axis_vocabulary(fd):
        pin_invest_decisions(pb, frames)
    return pb


def test_pin_with_own_decisions_keeps_the_objective(mono):
    fd, frames, obj, _ = mono
    pb = _pinned_problem(fd, frames)
    with axis_vocabulary(fd):
        sol = _solve(pb)
        got = utf8(sol.value("v_invest_p")).sort("p")
    assert float(sol.obj) == pytest.approx(obj, rel=1e-9)
    want = utf8(frames["v_invest_p"]).sort("p")
    for g, w in zip(got["value"].to_list(), want["value"].to_list()):
        assert abs(g - w) <= PIN_BAND * max(1.0, abs(w)) * (1 + 1e-9), (g, w)


def _mps_sections(path: Path) -> tuple[list[str], list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cut = lines.index("BOUNDS")
    return lines[:cut], [ln for ln in lines[cut + 1:] if ln != "ENDATA"]


def test_pin_changes_only_the_bounds_of_the_pinned_columns(mono, tmp_path):
    fd, frames, _, _ = mono
    free = build(fd)
    pinned = _pinned_problem(fd, frames)
    with axis_vocabulary(fd):
        free.write_mps(tmp_path / "free.mps")
        pinned.write_mps(tmp_path / "pinned.mps")
    free_head, free_bounds = _mps_sections(tmp_path / "free.mps")
    pin_head, pin_bounds = _mps_sections(tmp_path / "pinned.mps")
    assert pin_head == free_head  # rows, matrix, objective and right-hand sides
    prefixes = tuple(f"{n}[" for n in frames)

    def is_pinned(line: str) -> bool:
        return line.split()[2].startswith(prefixes)

    assert [ln for ln in free_bounds if not is_pinned(ln)] == [ln for ln in pin_bounds if not is_pinned(ln)]
    # Each pinned column's bounds (MPS: lower 0 unless an LO / FX line, upper +inf unless an UP / FX line).
    bounds: dict[str, list[float]] = {}
    for ln in pin_bounds:
        if is_pinned(ln):
            kind, _, name, value = ln.split()
            lo_up = bounds.setdefault(name, [0.0, math.inf])
            if kind in ("LO", "FX"):
                lo_up[0] = float(value)
            if kind in ("UP", "FX"):
                lo_up[1] = float(value)
    want = {f"v_invest_p[{r['p']},{r['d']}]": r["value"] for r in utf8(frames["v_invest_p"]).iter_rows(named=True)}
    assert bounds.keys() == want.keys()
    for name, value in want.items():
        band = PIN_BAND * max(1.0, abs(value))
        assert bounds[name] == pytest.approx([max(0.0, value - band), value + band], abs=1e-15), name


def test_pin_accepts_a_utf8_frame_against_the_model_axis(mono):
    """A value frame whose key columns are plain strings (a frame from another vocabulary) joins the model's
    columns through ``align_join_dtypes``: the same objective as the model-typed frame."""
    fd, frames, obj, _ = mono
    pb = _pinned_problem(fd, {n: utf8(f) for n, f in frames.items()})
    with axis_vocabulary(fd):
        assert float(_solve(pb).obj) == pytest.approx(obj, rel=1e-9)


def test_pin_clamps_a_value_just_outside_the_bounds(mono):
    """A Benders value a solver tolerance below the column's lower bound 0 is fixed at 0, not at the value (which
    would make the LP infeasible)."""
    fd, frames, _, _ = mono
    frame = utf8(frames["v_invest_p"]).with_columns(
        value=pl.when(pl.col("p") == "pipe_BC").then(-1e-9).otherwise(pl.col("value")))
    pb = _pinned_problem(fd, {"v_invest_p": frame})
    var = pb._vars["v_invest_p"]
    lower = dict(zip(utf8(var.frame)["p"].to_list(), var.lower))
    upper = dict(zip(utf8(var.frame)["p"].to_list(), var.upper))
    assert (lower["pipe_BC"], upper["pipe_BC"]) == (0.0, PIN_BAND)


def test_pin_raises_on_a_missing_column(mono):
    fd, frames, _, _ = mono
    short = utf8(frames["v_invest_p"]).filter(pl.col("p") != "pipe_BC")
    with pytest.raises(RuntimeError, match=r"no Benders value for v_invest_p of 'pipe_BC', d 'y2030'"):
        _pinned_problem(fd, {"v_invest_p": short})
    with pytest.raises(RuntimeError, match=r"no Benders value for v_invest_p of 'pipe_"):
        _pinned_problem(fd, {})


def test_pin_raises_on_a_value_for_no_column(mono):
    fd, frames, _, _ = mono
    extra = pl.concat([utf8(frames["v_invest_p"]),
                       pl.DataFrame({"p": ["pipe_XY"], "d": ["y2030"], "value": [1.0]})])
    with pytest.raises(RuntimeError, match=r"a Benders value for v_invest_p of 'pipe_XY', d 'y2030', which"):
        _pinned_problem(fd, {"v_invest_p": extra})


def test_pin_raises_on_two_values_for_one_column(mono):
    fd, frames, _, _ = mono
    twice = pl.concat([utf8(frames["v_invest_p"]), utf8(frames["v_invest_p"])])
    with pytest.raises(RuntimeError, match=r"two Benders values for v_invest_p"):
        _pinned_problem(fd, {"v_invest_p": twice})


def test_pin_raises_on_a_family_the_model_lacks(mono):
    fd, frames, _, _ = mono
    node = pl.DataFrame({"n": ["lh2_B"], "d": ["y2030"], "value": [1.0]})
    with pytest.raises(RuntimeError, match=r"values for 'v_invest_n', which the undecomposed model does not"):
        _pinned_problem(fd, {**frames, "v_invest_n": node})
    with pytest.raises(RuntimeError, match=r"'v_flow' is not an investment decision family"):
        _pinned_problem(fd, {**frames, "v_flow": node})
    # An aggregate family (an equality defines it) is not pinned and not an error.
    pinned = _pinned_problem(fd, {**frames, AGGREGATE_FAMILIES[0]: node})
    assert AGGREGATE_FAMILIES[0] not in pinned._vars


def test_pin_raises_on_values_for_a_family_without_columns():
    """A family the model has as a variable with no column, but for which the Benders solve has values: raised (not
    silently ignored)."""
    from types import SimpleNamespace

    empty = SimpleNamespace(frame=pl.DataFrame({"n": [], "d": []}, schema={"n": pl.Utf8, "d": pl.Utf8}),
                            dims=("n", "d"))
    pb = SimpleNamespace(_vars={"v_invest_n": empty})
    node = pl.DataFrame({"n": ["lh2_B"], "d": ["y2030"], "value": [1.0]})
    with pytest.raises(RuntimeError, match=r"1 values for v_invest_n, which has no column"):
        pin_invest_decisions(pb, {"v_invest_n": node})
    pin_invest_decisions(pb, {"v_invest_n": node.head(0)})  # no values: nothing to pin


# ---------------------------------------------------------------------------------------------------------------
# Review m3: a Benders value a solver tolerance outside a row bound of the whole model
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.solver
def test_pin_band_keeps_a_value_just_below_a_binding_floor_feasible(tmp_path_factory, monkeypatch):
    """wind_A must build at least 100 MW in y2030 (invest_min_period, a row) and builds exactly that.  Its value a
    fraction of the band below the floor (a solver-tolerance miss, as a Benders master can leave), fixed exactly,
    makes the LP infeasible (the row has no other column); :func:`pin_invest_decisions` bounds it to the
    :data:`PIN_BAND` band, so the final LP solves, to the monolith's objective."""
    from flextool.engine.solve import _benders_final

    steps = _run(tmp_path_factory, alt="p287_floor_band", values=INVEST_FLOOR + _none("lh2_invest"),
                 base_chain=INVEST_CHAIN, fixture_json=LH2_JSON)
    fd = steps["lh2_invest"].flex_data
    free = build(fd)
    with axis_vocabulary(fd):
        sol = _solve(free)
        mono_obj = float(sol.obj)
        frames = {n: utf8(sol.value(n)) for n in _pinned_present(free)}
    inv = frames["v_invest_p"]
    [wind] = inv.filter(pl.col("p") == "wind_A")["value"].to_list()
    assert wind > 0
    below = wind - 0.2 * PIN_BAND * max(1.0, wind)
    shifted = {**frames, "v_invest_p": inv.with_columns(
        value=pl.when(pl.col("p") == "wind_A").then(below).otherwise(pl.col("value")))}
    with axis_vocabulary(fd):
        with monkeypatch.context() as mp:
            mp.setattr(_benders_final, "PIN_BAND", 0.0)  # fixed exactly
            exact = build(fd)
            pin_invest_decisions(exact, shifted)
            assert not exact.solve().optimal  # the floor row cannot be met
        banded = build(fd)
        pin_invest_decisions(banded, shifted)
        got = _solve(banded)
    assert float(got.obj) == pytest.approx(mono_obj, rel=1e-6)


# ---------------------------------------------------------------------------------------------------------------
# T5b: the Benders decisions cover the pinned columns on each fixture shape
# ---------------------------------------------------------------------------------------------------------------

#: wind_A builds at least 100 MW in y2030 (test_benders_invest_minimum).
INVEST_FLOOR = [["unit", "wind_A", "existing", b64(0.0, "float")],
                ["unit", "wind_A", "invest_method", b64("invest_period", "str")],
                ["unit", "wind_A", "invest_cost", b64(1.0e7, "float")],
                ["unit", "wind_A", "invest_max_period", _period_map({"y2030": 1000.0})],
                ["unit", "wind_A", "invest_min_period", _period_map({"y2030": 100.0})]]
#: wind_A builds 100 MW and retires 40 MW of them in y2030 (test_cost_lines_reported): v_divest_built_p.
RETIRE_BUILT = [["unit", "wind_A", "existing", b64(0.0, "float")],
                ["unit", "wind_A", "lifetime_method", b64("reinvest_choice", "str")],
                ["unit", "wind_A", "invest_method", b64("invest_retire_period", "str")],
                ["unit", "wind_A", "net_salvage_value", b64(100_000.0, "float")],
                ["unit", "wind_A", "invest_max_period", _period_map({"y2030": 100.0})],
                ["unit", "wind_A", "retire_max_period", _period_map({"y2030": 40.0})]]
#: liquefier_A with a two-line efficiency curve, in region A only (test_benders_piecewise_region).
PIECEWISE = [["unit", "liquefier_A", "conversion_method", b64("piecewise_linear", "str")],
             ["unit", "liquefier_A", "piecewise_linear_efficiency",
              b64({"index_type": "float", "data": [[0.2, 0.90], [0.6, 0.87], [1.0, 0.80]]}, "map")]]

#: ``id: (fixture json, base chain, override values, the Benders solve, families the fixture must pin)``.
CHAIN_SHAPES = {
    "trade_invest": (TRADE_JSON, TRADE_CHAIN, [], "lh2_trade_invest", {"v_invest_p"}),
    "region_invest": (LH2_JSON, INVEST_CHAIN, [], "lh2_invest", {"v_invest_p"}),
    "invest_minimum": (LH2_JSON, INVEST_CHAIN, INVEST_FLOOR, "lh2_invest", {"v_invest_p"}),
    "retire_built": (LH2_JSON, INVEST_CHAIN, RETIRE_BUILT, "lh2_invest", {"v_invest_p", "v_divest_built_p"}),
    "piecewise_region": (LH2_JSON, INVEST_CHAIN, PIECEWISE, "lh2_invest", {"v_invest_p"}),
    # Existing capacity only (load-bearing trade over the existing pipes, test_benders_existing_capacity_trade):
    # no decision to pin.
    "existing_trade": (LH2_JSON, ["lh2_three_region"], [["solve", "lh2_week", "decomposition",
                                                         b64("benders", "str")]], "lh2_week", set()),
}


def _check_coverage_and_sandwich(fd, frames: dict, gap: float, expect: set[str]) -> None:
    """Every column of each pinned family of the undecomposed model has exactly one Benders value, and the model
    pinned with them solves to ``[M, M + (gap + 1e-6)·|M|]`` (M: the undecomposed optimum)."""
    free = build(fd)
    present = _pinned_present(free)
    assert set(present) >= expect, present
    for name in present:
        assert name in frames, name
        dims = list(free._vars[name].dims)
        model_keys = utf8(free._vars[name].frame.select(dims)).sort(dims)
        benders_keys = utf8(frames[name].select(dims)).sort(dims)
        assert benders_keys.equals(model_keys), name
    assert not {n for n in frames if n in PINNED_FAMILIES} - set(present)
    with axis_vocabulary(fd):
        mono_obj = float(_solve(free).obj)
    pinned = build(fd)
    with axis_vocabulary(fd):
        pin_invest_decisions(pinned, frames)
        obj = float(_solve(pinned).obj)
    assert obj >= mono_obj - 1e-9 * abs(mono_obj)
    assert obj - mono_obj <= (gap + 1e-6) * abs(mono_obj)


@pytest.mark.solver
@pytest.mark.parametrize("shape", sorted(CHAIN_SHAPES))
def test_benders_decisions_cover_the_pinned_columns(shape, tmp_path_factory, monkeypatch):
    # Final solve off: the step then keeps the Benders decisions (a SnapshotSolution of invest_solution_vars).
    monkeypatch.setenv("FLEXTOOL_BENDERS_FINAL_SOLVE", "off")
    fixture, chain, values, solve, expect = CHAIN_SHAPES[shape]
    steps = _run(tmp_path_factory, alt=f"p287_{shape}", values=values, base_chain=chain, fixture_json=fixture)
    step = steps[solve]
    assert step.is_benders and step.optimal  # converged
    frames = dict(step.solution._vars)  # the SnapshotSolution of BendersResult.invest_solution_vars
    _check_coverage_and_sandwich(step.flex_data, frames, float(step.benders_gap), expect)


@pytest.mark.solver
def test_benders_master_hosted_decisions_cover_the_pinned_columns(scenario_run):
    """Regions A and C only: the master hosts region B's nodes and invests in its storage node lh2_B
    (``v_invest_n``) and its unit liquefier_B (test_benders_master_hosted)."""
    from flextool.engine.solve._benders import solve_benders
    from tests._support.load import load_scenario

    data = load_scenario(scenario_run("lh2_three_region_trade_invest", db_fixture="lh2_master_hourly"))
    res = solve_benders(data, ["region_A", "region_C"], max_iters=20, tol=1e-4, scale_the_objective=1e-6)
    assert res.converged
    _check_coverage_and_sandwich(data, dict(res.invest_solution_vars), float(res.gap), {"v_invest_p", "v_invest_n"})


# ---------------------------------------------------------------------------------------------------------------
# check_final_objective and T6 (the option)
# ---------------------------------------------------------------------------------------------------------------


def test_check_final_objective():
    assert check_final_objective(100.0, 99.0, 101.0)
    assert check_final_objective(101.0 + 1e-5, 99.0, 101.0)  # within 1e-6 relative of 101
    assert not check_final_objective(101.01, 99.0, 101.0)
    assert not check_final_objective(98.9, 99.0, 101.0)
    assert check_final_objective(101.05, 99.0, 101.0, solver_tolerance=1e-3)
    assert check_final_objective(-5.0, -math.inf, 0.0)  # no lower bound yet
    assert not check_final_objective(math.nan, 99.0, 101.0)
    assert check_final_objective(-101.0, -101.0 - 1e-5, -99.0)


def test_final_solve_option_flag_over_env_over_default():
    from flextool.cli.cmd_run_flextool import build_parser

    parser = build_parser()
    unset = parser.parse_args(["sqlite:///x.sqlite"])
    flag_on = parser.parse_args(["sqlite:///x.sqlite", "--benders-final-solve", "on"])
    flag_off = parser.parse_args(["sqlite:///x.sqlite", "--benders-final-solve", "off"])
    env_off = {"FLEXTOOL_BENDERS_FINAL_SOLVE": "off"}
    assert final_solve_on(RunOptions.from_cli(unset, environ={})) is True
    assert final_solve_on(RunOptions.from_cli(unset, environ=env_off)) is False
    assert final_solve_on(RunOptions.from_cli(flag_on, environ=env_off)) is True
    assert final_solve_on(RunOptions.from_cli(flag_off, environ={})) is False
    with pytest.raises(SystemExit):
        parser.parse_args(["sqlite:///x.sqlite", "--benders-final-solve", "auto"])


def test_final_solve_option_of_a_library_call_reads_the_environment(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_BENDERS_FINAL_SOLVE", raising=False)
    assert final_solve_on() is True
    monkeypatch.setenv("FLEXTOOL_BENDERS_FINAL_SOLVE", "off")
    assert final_solve_on() is False
    monkeypatch.setenv("FLEXTOOL_BENDERS_FINAL_SOLVE", "maybe")  # malformed: one warning, the default
    assert final_solve_on() is True
