"""#234: the reserve requirement, provision and shortfall of a reserve group, per step and per period.

Vehicle: ``tests.json`` chain ``init, west, coal`` cut to two one-hour steps (``2day`` timeset_duration 2): node west
with a constant demand of 80 MW, coal_plant its only unit (existing 100 MW, so its up headroom is 20 MW; coal 20
CUR/MWh at efficiency 0.4: 50 CUR/MWh).  Upward reserve ``rsv`` on group ``g`` = {west}, coal_plant a provider
(max_share 1, reliability 1), penalty_reserve 500 CUR/MWh (10× the fuel cost); west's penalty_up 900 is above it, so
the LP never drops load to free reserve.  Built from JSON (never a checked-in ``.sqlite``).

Hand values (independent of the code):

(a) timeseries reservation 30, 50 MW at t0001, t0002: provided 20, 20; shortfall 10, 30; requirement 30, 50.  Per
    period with w ≡ 1: requirement_avg 40, provided_avg 20, shortfall_avg 20, shortfall_max 30.  With
    timeset_weights 3, 1 (normalised to w = 1.5, 0.5, Σw = 2): requirement_avg (1.5·30 + 0.5·50)/2 = 35,
    shortfall_avg (1.5·10 + 0.5·30)/2 = 15, provided_avg 20, requirement_max 50.
(b) large_failure_only, coal_plant large_failure_ratio 1, a second provider u2 (30 MW, efficiency 0.3, so 66.7
    CUR/MWh, dearer than coal_plant: it produces nothing): requirement_n_1 = 80 (coal_plant's flow), n_1_process
    coal_plant, counted = u2's reserve 30 (coal_plant's own does not count), shortfall 80 − 30 = 50.  Moving flow to
    u2 keeps the shortfall at 50 but costs more, so the optimum is unique.  coal_plant's own reserve is free, so
    ``provided`` is not asserted.
(c) dynamic_only, coal_plant increase_reserve_ratio 0.1: requirement_dynamic 0.1 · 80 = 8 ≤ the headroom 20, so the
    shortfall is 0 (provided can be anything in 8..20: not asserted).
In each case the provider rows sum to ``provided``.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import PLANT, PLANT_CHAIN, b64, make_db, time_map

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
REL = "reserve__upDown__unit__node"
RUG = ["rsv", "up", "g"]
U2 = "u2"
SOLVE = "y2020_2day_dispatch"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _map(rows: list) -> list:
    return b64({"index_type": "str", "data": rows}, "map")


def _case(method: str, provider_params: dict, *, weights: bool = False, extra_values: list = (),
          extra_entities: list = ()) -> tuple[list, list]:
    ents = [["reserve", "rsv", None], ["group", "g", None], ["group__node", ["g", "west"], None],
            ["reserve__upDown__group", RUG, None], [REL, ["rsv", "up", PLANT, "west"], None], *extra_entities]
    vals = [["timeset", "2day", "timeset_duration", _map([["t0001", 2.0]])],
            ["node", "west", "inflow", b64(-80.0, "float")],
            ["unit", PLANT, "existing", b64(100.0, "float")],
            ["reserve__upDown__group", RUG, "reserve_method", b64(method, "str")],
            ["reserve__upDown__group", RUG, "penalty_reserve", b64(500.0, "float")],
            [REL, ["rsv", "up", PLANT, "west"], "is_enabled", b64("yes", "str")],
            [REL, ["rsv", "up", PLANT, "west"], "max_share", b64(1.0, "float")],
            [REL, ["rsv", "up", PLANT, "west"], "reliability", b64(1.0, "float")],
            *([[REL, ["rsv", "up", PLANT, "west"], k, b64(v, "float")] for k, v in provider_params.items()]),
            *extra_values]
    if weights:
        vals.append(["timeset", "2day", "timeset_weights", _map([["t0001", 3.0], ["t0002", 1.0]])])
    return ents, vals


def _run(tmp_path_factory, alt: str, ents: list, vals: list) -> dict:
    """Solve the case, write its CSV outputs; ``{"work", "csv", "stdout", "step"}``."""
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    url, scen = make_db(tmp_path_factory, alt=alt, values=vals, base_chain=PLANT_CHAIN, entities=ents)
    work = tmp_path_factory.mktemp(f"{alt}_w")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        steps = run_chain_from_db(url, scen, work_folder=work, keep_solutions=True)
    last = next(reversed(steps.values()))
    write_outputs(scen, output_location=str(work / "out"), subdir=scen,
                  output_config_path=str(REPO / "flextool" / "schemas" / "default_plots.yaml"),
                  write_methods=["csv"], fallback_output_location=str(work), raw_output_dir=str(work / "output_raw"),
                  solution=last.solution, solve_name=last.solve_name, flex_data=last.flex_data,
                  flex_data_provider=last.flex_data_provider)
    return {"work": work, "csv": work / "out" / "output_csv" / scen, "stdout": out.getvalue(), "step": last}


def _raw(run: dict, prefix: str) -> pl.DataFrame:
    from flextool.engine.output._reserve_report import read_reserve_report_parts

    frame = read_reserve_report_parts(run["work"] / "output_raw", prefix)
    assert frame is not None, prefix
    return frame.sort("period", "time")


def _report(run: dict) -> pl.DataFrame:
    from flextool.engine.output._reserve_report import REPORT_PREFIX

    return _raw(run, REPORT_PREFIX).filter(pl.col("node_group") == "g")


def _providers_sum_to_provided(run: dict) -> None:
    from flextool.engine.output._reserve_report import PROVIDER_PREFIX

    prov = (_raw(run, PROVIDER_PREFIX).filter(pl.col("node_group") == "g")
            .group_by("period", "time").agg(pl.col("value").sum()).sort("period", "time"))
    rep = _report(run)
    assert prov["time"].to_list() == rep["time"].to_list()
    assert prov["value"].to_list() == pytest.approx(rep["provided"].to_list(), rel=1e-12, abs=1e-9)


def _csv(run: dict, name: str) -> pd.DataFrame:
    """A written CSV (header rows reserve, updown, node_group, quantity; the provider table reserve, updown,
    node_group, process, node) with its column levels and its solve / period / time index."""
    path = run["csv"] / f"{name}.csv"
    head = pd.read_csv(path, nrows=0).columns
    n_index = sum(1 for c in head if c in ("solve", "period", "time"))
    n_levels = 5 if "provider" in name else 4
    return pd.read_csv(path, header=list(range(n_levels)), index_col=list(range(n_index)))


@pytest.fixture(scope="module")
def timeseries(tmp_path_factory):
    ents, vals = _case("timeseries_only", {}, extra_values=[
        ["reserve__upDown__group", RUG, "reservation", time_map({"t0001": 30.0, "t0002": 50.0}, 0.0)]])
    return _run(tmp_path_factory, "r234a", ents, vals)


@pytest.fixture(scope="module")
def timeseries_weighted(tmp_path_factory):
    ents, vals = _case("timeseries_only", {}, weights=True, extra_values=[
        ["reserve__upDown__group", RUG, "reservation", time_map({"t0001": 30.0, "t0002": 50.0}, 0.0)]])
    return _run(tmp_path_factory, "r234aw", ents, vals)


def test_a_timeseries_requirement_per_step(timeseries) -> None:
    rep = _report(timeseries)
    assert rep["time"].to_list() == ["t0001", "t0002"]
    assert rep["requirement"].to_list() == pytest.approx([30.0, 50.0])
    assert rep["requirement_timeseries"].to_list() == pytest.approx([30.0, 50.0])
    assert rep["requirement_dynamic"].to_list() == [None, None]
    assert rep["requirement_n_1"].to_list() == [None, None]
    assert rep["binding_method"].to_list() == ["timeseries", "timeseries"]
    assert rep["provided"].to_list() == pytest.approx([20.0, 20.0], abs=1e-6)
    assert rep["counted"].to_list() == pytest.approx([20.0, 20.0], abs=1e-6)
    assert rep["shortfall"].to_list() == pytest.approx([10.0, 30.0], abs=1e-6)
    _providers_sum_to_provided(timeseries)
    assert "reserve requirement check" not in timeseries["stdout"]


def test_a_timeseries_tables(timeseries) -> None:
    g = _csv(timeseries, "reserve__upDown__group__dt")[("rsv", "up", "g")]
    assert list(g.columns) == ["requirement", "requirement_timeseries", "provided", "counted", "shortfall"]
    assert g["requirement"].tolist() == pytest.approx([30.0, 50.0])
    assert g["shortfall"].tolist() == pytest.approx([10.0, 30.0], abs=1e-6)
    d = _csv(timeseries, "reserve__upDown__group__d")[("rsv", "up", "g")]
    assert list(d.columns) == ["requirement_avg", "provided_avg", "shortfall_avg", "requirement_max", "shortfall_max"]
    row = d.iloc[0]
    assert row["requirement_avg"] == pytest.approx(40.0)
    assert row["provided_avg"] == pytest.approx(20.0, abs=1e-6)
    assert row["shortfall_avg"] == pytest.approx(20.0, abs=1e-6)
    assert row["requirement_max"] == pytest.approx(50.0)
    assert row["shortfall_max"] == pytest.approx(30.0, abs=1e-6)
    prov = _csv(timeseries, "reserve__group__provider__dt")
    assert prov[("rsv", "up", "g", PLANT, "west")].tolist() == pytest.approx([20.0, 20.0], abs=1e-6)


def test_a_period_averages_follow_the_timestep_weights(timeseries_weighted) -> None:
    rep = _report(timeseries_weighted)
    assert rep["shortfall"].to_list() == pytest.approx([10.0, 30.0], abs=1e-6)
    d = _csv(timeseries_weighted, "reserve__upDown__group__d")[("rsv", "up", "g")]
    row = d.iloc[0]
    assert row["requirement_avg"] == pytest.approx(35.0)
    assert row["shortfall_avg"] == pytest.approx(15.0, abs=1e-6)
    assert row["provided_avg"] == pytest.approx(20.0, abs=1e-6)
    assert row["requirement_max"] == pytest.approx(50.0)
    # Item 188: the provider's period average follows the same weights (20 MW at both steps: 20).
    avg = _csv(timeseries_weighted, "process__reserve__upDown__node__d_average")
    assert avg.loc[:, (PLANT, "rsv", "up", "west")].tolist() == pytest.approx([20.0], abs=1e-6)


def test_b_n_1_requirement_counts_the_others_reserve(tmp_path_factory) -> None:
    ents, vals = _case("large_failure_only", {"large_failure_ratio": 1.0},
                       extra_entities=[["unit", U2, None], ["unit__inputNode", [U2, "coal_market"], None],
                                       ["unit__outputNode", [U2, "west"], None], [REL, ["rsv", "up", U2, "west"], None]],
                       extra_values=[["unit", U2, "existing", b64(30.0, "float")],
                                     ["unit", U2, "efficiency", b64(0.3, "float")],
                                     [REL, ["rsv", "up", U2, "west"], "is_enabled", b64("yes", "str")],
                                     [REL, ["rsv", "up", U2, "west"], "max_share", b64(1.0, "float")],
                                     [REL, ["rsv", "up", U2, "west"], "reliability", b64(1.0, "float")]])
    run = _run(tmp_path_factory, "r234b", ents, vals)
    rep = _report(run)
    assert rep.height == 2
    assert rep["requirement_n_1"].to_list() == pytest.approx([80.0, 80.0], abs=1e-6)
    assert rep["n_1_process"].to_list() == [PLANT, PLANT]
    assert rep["binding_method"].to_list() == ["n_1", "n_1"]
    assert rep["binding_process"].to_list() == [PLANT, PLANT]
    assert rep["requirement"].to_list() == pytest.approx([80.0, 80.0], abs=1e-6)
    assert rep["counted"].to_list() == pytest.approx([30.0, 30.0], abs=1e-6)
    assert rep["shortfall"].to_list() == pytest.approx([50.0, 50.0], abs=1e-6)
    _providers_sum_to_provided(run)
    assert "reserve requirement check" not in run["stdout"]


def test_c_dynamic_requirement(tmp_path_factory) -> None:
    ents, vals = _case("dynamic_only", {"increase_reserve_ratio": 0.1})
    run = _run(tmp_path_factory, "r234c", ents, vals)
    rep = _report(run)
    assert rep["requirement_dynamic"].to_list() == pytest.approx([8.0, 8.0], abs=1e-6)
    assert rep["requirement"].to_list() == pytest.approx([8.0, 8.0], abs=1e-6)
    assert rep["binding_method"].to_list() == ["dynamic", "dynamic"]
    assert rep["shortfall"].to_list() == pytest.approx([0.0, 0.0], abs=1e-6)
    _providers_sum_to_provided(run)
    assert "reserve requirement check" not in run["stdout"]


# ---------------------------------------------------------------------------
# The evaluated rows against the LP's own rows (critique 7): for every reserveBalance row of the built LP, the
# right-hand side at the solution, b − Σ (the row's coefficients × values over every column but v_reserve,
# v_reserve_total and vq_reserve), is the report's need of that row.  One vehicle per term family: the input a unit
# draws (slope and the min-load section), the reverse flow of a one-variable connection, the group demand term,
# and coarse processes.

_FAMILIES = {"reserveBalance_timeseries_eq": "timeseries", "reserveBalance_dynamic_eq": "dynamic",
             "reserveBalance_up_n_1_eq": "n_1", "reserveBalance_down_n_1_eq": "n_1"}
_RESERVE_COLS = ("v_reserve[", "v_reserve_total[", "vq_reserve[")


def _lp_needs(step) -> dict[tuple, float]:
    """``{(method, r, ud, g, f, d, t): need}`` of the reserveBalance rows of the step's LP, rebuilt (the column ids
    of a rebuild are the solve's: the build is deterministic) and evaluated at the solution's column values."""
    import numpy as np

    from tests.engine._unit_db import build

    m = build(step.flex_data).canonicalise()
    x = np.asarray(step.solution.col_value, dtype=float)
    assert len(x) == m.n_cols
    flows = [i for i, nm in enumerate(m.col_names) if nm.startswith("v_flow[")]
    assert x[flows].sum() == pytest.approx(float(step.solution.value("v_flow")["value"].sum()), rel=1e-12)
    skip = np.array([nm.startswith(_RESERVE_COLS) for nm in m.col_names])
    col = np.repeat(np.arange(m.n_cols), np.diff(m.col_ptr))
    activity = np.bincount(m.row_idx, weights=np.where(skip[col], 0.0, m.val * x[col]), minlength=m.n_rows)
    out: dict[tuple, float] = {}
    for i, name in enumerate(m.row_names):
        family, _, inner = name.partition("[")
        method = _FAMILIES.get(family)
        if method is None:
            continue
        keys = inner[:-1].split(",")
        r, ud, g, *rest = keys
        f = rest[0] if method == "n_1" else None
        out[(method, r, ud, g, f, rest[-2], rest[-1])] = float(m.row_lb[i] - activity[i])
    return out


def _report_needs(step) -> dict[tuple, float]:
    from flextool.engine.model._reserve import reserve_requirement_rows

    rows = reserve_requirement_rows(step.flex_data, step.solution)
    assert rows is not None
    return {(m, r, ud, g, f, d, t): need for r, ud, g, m, f, d, t, need in rows.iter_rows()}


def _assert_needs_match(step, families: set[str]) -> None:
    lp, report = _lp_needs(step), _report_needs(step)
    assert set(report) == set(lp)
    assert {k[0] for k in lp} == families
    assert any(abs(v) > 1e-3 for v in lp.values())
    for key, need in lp.items():
        assert report[key] == pytest.approx(need, rel=1e-9, abs=1e-6), key


def test_rows_match_the_lp_drawn_input_and_section(tmp_path_factory) -> None:
    """``test_reserve_input_drawn`` with a min_load_efficiency e_boiler (west → heat, linear online), heat's
    penalty_up raised to 10 000 so that e_boiler runs: the dynamic row counts wind's output and e_boiler's input
    (v_flow · slope + v_online · section), the n-1-down rows the input of e_boiler and the consumption of
    dr_increase_demand."""
    from tests.engine._unit_db import solve
    from tests.engine.test_reserve_input_drawn import BASE, EB, EB_ENTS, _eb_values, _reserves

    ents, vals = _reserves()
    step = solve(tmp_path_factory, alt="r234_drawn", base_chain=BASE, entities=EB_ENTS + ents,
                 values=_eb_values("min_load_efficiency") + vals + [
                     ["unit", EB, "efficiency_at_min_load", b64(0.4, "float")],
                     ["unit", EB, "min_load", b64(0.3, "float")],
                     ["unit", EB, "startup_method", b64("linear", "str")],
                     ["node", "heat", "penalty_up", b64(10000.0, "float")]])
    assert step.solution.value("v_online_linear")["value"].max() > 1e-3
    _assert_needs_match(step, {"dynamic", "n_1"})


def test_rows_match_the_lp_reverse_flow(tmp_path_factory) -> None:
    """``test_connection_reserve``'s one-variable connection ll (lw ↔ le): a dynamic upward reserve at lw (ll's
    v_flow and v_flow_back × 0.5), an upward n-1 at lw and a downward n-1 at le (ll failing, ratio 1)."""
    from tests.engine.test_connection_reserve import _group, _reserve, _steps

    ents, vals = [*_group("gw", "lw"), *_group("ge", "le")], []
    for r, ud, g, n, method, extra in (("dyu", "up", "gw", "lw", "dynamic_only", {"increase_reserve_ratio": 0.5}),
                                       ("n1u", "up", "gw", "lw", "large_failure_only", {"large_failure_ratio": 1.0}),
                                       ("n1d", "down", "ge", "le", "large_failure_only", {"large_failure_ratio": 1.0})):
        e, v = _reserve(r, ud, g, [("ll", n)], method=method, extra=extra)
        ents += [x for x in e if x not in ents]
        vals += v
    step = _steps(tmp_path_factory, "r234_back", ents, vals)[-1]
    assert step.solution.value("v_flow_back")["value"].abs().max() > 1e-3
    _assert_needs_match(step, {"dynamic", "n_1"})


def test_rows_match_the_lp_group_demand_and_coarse(tmp_path_factory) -> None:
    """``test_coarse_online_reserves``'s hand-built model (gen_d coarse: one flow and one reserve per day into hd,
    demand 25 MW on day 1) with reserve_method ``all``: the reservation, the dynamic row (gen_d's daily flow ×
    0.5 on every hour plus the group demand 0.1 × 25 MW), and gen_d failing (ratio 1)."""
    from tests.engine._unit_db import run
    from tests.engine.test_coarse_online_reserves import _hand_res_model

    model = _hand_res_model(False)
    grp = ["r_up", "up", "hd_res"]
    rel = ["r_up", "up", "gen_d", "hd"]
    vals = [v for v in model["values"] if v[:3] != ["reserve__upDown__group", grp, "reserve_method"]]
    vals += [["reserve__upDown__group", grp, "reserve_method", b64("all", "str")],
             ["reserve__upDown__group", grp, "increase_reserve_ratio", b64(0.1, "float")],
             ["reserve__upDown__unit__node", rel, "increase_reserve_ratio", b64(0.5, "float")],
             ["reserve__upDown__unit__node", rel, "large_failure_ratio", b64(1.0, "float")]]
    url, scen = make_db(tmp_path_factory, alt="r234_coarse", values=vals, base_chain=model["base_chain"],
                        entities=model["entities"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    assert _has_coarse(step)
    _assert_needs_match(step, {"timeseries", "dynamic", "n_1"})
    dyn = {k[-1]: v for k, v in _report_needs(step).items() if k[0] == "dynamic"}
    # Day 1: gen_d's flow 25 MW · 0.5 + the demand 25 MW · 0.1 = 15 MW on every hour.
    assert dyn["t0001"] == pytest.approx(15.0, abs=1e-6)


def _has_coarse(step) -> bool:
    from flextool.engine.model._common import _has_rows

    return _has_rows(getattr(step.flex_data, "process_coarse_step", None))


def test_a_broken_evaluation_is_reported(timeseries, monkeypatch) -> None:
    """The check: a need 5 MW off the LP's (here: the evaluation made to miss 5 MW at t0002) leaves a deficit of 25
    MW against the LP's shortfall of 30 MW; the run-log line names the solve, the reserve, the group and the
    step.  Within the tolerance nothing is reported (the hand cases above)."""
    import flextool.engine.model._reserve as reserve
    from flextool.engine.output._reserve_report import check_lines

    step = timeseries["step"]
    rows = reserve.reserve_requirement_rows

    def short(d, sol):
        out = rows(d, sol)
        return out.with_columns(need=pl.when(pl.col("t") == "t0002").then(pl.col("need") - 5.0)
                                .otherwise(pl.col("need")))

    monkeypatch.setattr(reserve, "reserve_requirement_rows", short)
    _, _, violations = reserve.reserve_requirement_report(step.flex_data, step.solution)
    assert violations.filter(pl.col("node_group") == "g")["time"].to_list() == ["t0002"]
    lines = check_lines(violations.filter(pl.col("node_group") == "g"), SOLVE, near_optimal=False)
    assert lines == [f"reserve requirement check: solve {SOLVE}, reserve rsv up, group g, step (p2020, t0002): "
                     "deficit 25 MW, shortfall 30 MW"]


def test_steps_of_a_solve_without_reserves_are_empty_and_not_averaged(tmp_path_factory) -> None:
    """``test_reduce_detail_chains``: sA (t0001-t0002) and sB (t0003-t0004) share p2020; sB leaves reserves out
    (reduce_system_detail_method).  The requirement of 10 MW is met by g1 in sA; sB's steps are empty cells, and the
    period averages are over sA's steps: 10 MW, not (10 + 10 + 0 + 0) / 4 = 5 MW (the per-provider average of item
    188 too)."""
    from tests.engine.test_reduce_detail_chains import _run as run_chain, _write_outputs

    steps, wf, scen = run_chain(tmp_path_factory, "r234_reduced", {"sB": ("online", "reserves")})
    out = _write_outputs(steps, wf, scen, in_memory=False)
    run = {"csv": out}
    dt = _csv(run, "reserve__upDown__group__dt")[("rsv", "up", "electricity")]
    times = dt.index.get_level_values(-1).tolist()
    assert times == ["t0001", "t0002", "t0003", "t0004"]
    assert dt["requirement"].tolist()[:2] == pytest.approx([10.0, 10.0])
    assert dt["provided"].tolist()[:2] == pytest.approx([10.0, 10.0], abs=1e-6)
    assert dt.iloc[2:].isna().all().all()
    d = _csv(run, "reserve__upDown__group__d")[("rsv", "up", "electricity")].iloc[0]
    assert d["requirement_avg"] == pytest.approx(10.0)
    assert d["provided_avg"] == pytest.approx(10.0, abs=1e-6)
    assert d["shortfall_avg"] == pytest.approx(0.0, abs=1e-6)
    avg = _csv(run, "process__reserve__upDown__node__d_average")
    assert avg.loc[:, ("g1", "rsv", "up", "elec")].tolist() == pytest.approx([10.0], abs=1e-6)
