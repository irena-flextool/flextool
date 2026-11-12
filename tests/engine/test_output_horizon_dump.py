"""Raw horizon dump behind ``model.output_horizon`` (``flextool/engine/output/_horizon_dump.py``).

With the flag on, every sub-solve writes all its LP variables and constraints over all rows as raw solver values
to ``output_raw/horizon/<lp_name>/<seq>__<roll>.parquet``; ``write_outputs`` streams them into
``output_parquet/<scenario>/horizon/<lp_name>.parquet``.  Columns: the LP dims, ``solve``, ``roll``, ``realized``
and ``value`` + ``reduced_cost`` (variables) or ``dual`` (constraints); duals × 1/scale_the_objective, nothing else.

Fixtures are built from JSON (``stoch_two_period``, ``stoch_two_period_invest``, ``tests.json`` overlays and
``lh2_three_region``).  Hand values:

* ``stoch_horizon`` (forecasts-only {high 0.25, low 0.75} fan, no realized member): wind 60/50 MW (high) and 20/10
  MW (low) in p2035/p2040, thermal the rest of demand 100/110 MW; wind existing 100, thermal 200 (values are per
  unitsize).  Thermal (50 EUR/MWh) is marginal everywhere, so the city balance dual is
  50 × 2920 (annualization 8760/3) × weight = 36 500 (high) / 109 500 (low).
* ``recourse_horizon``: per-forecast peaker investment 60/30 (high) and 80/40 (low).
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, make_db

OUTPUT_CONFIG = FLEXTOOL_ROOT / "flextool" / "schemas" / "default_plots.yaml"
LH2_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"
FLAG_ON = ["model", "flexTool", "output_horizon", b64("yes", "str")]

pytestmark = pytest.mark.solver


def _run(url: str, scenario: str, work: Path) -> dict:
    from flextool.engine import run_chain_from_db

    steps = run_chain_from_db(url, scenario, work_folder=work, keep_solutions=True)
    assert steps, "run_chain_from_db produced no steps"
    return steps


def _write_outputs(steps: dict, work: Path, scenario: str, out: Path, subdir: str | None = None) -> Path:
    """``write_outputs`` as the CLI runs it (parquet only); returns the scenario's parquet folder."""
    from flextool.process_outputs.write_outputs import write_outputs

    last = next(reversed(list(steps.values())))
    write_outputs(
        scenario, output_location=str(out), subdir=subdir or scenario, output_config_path=str(OUTPUT_CONFIG),
        write_methods=["parquet"], raw_output_dir=str(work / "output_raw"), flex_data=last.flex_data,
        solution=last.solution, solve_name=last.solve_name, flex_data_provider=last.flex_data_provider,
    )
    return out / "output_parquet" / (subdir or scenario)


def _dump(parquet_dir: Path, name: str) -> pl.DataFrame:
    return pl.read_parquet(parquet_dir / "horizon" / f"{name}.parquet")


def _inv_scale(work: Path) -> float:
    """1 / the solve's effective scale_the_objective (``solve_data/scale_the_objective.csv``, written every run)."""
    return 1.0 / float(pl.read_csv(work / "solve_data" / "scale_the_objective.csv")["value"][0])


@pytest.fixture(scope="module")
def stoch_horizon(stoch_two_period_db_url, tmp_path_factory):
    work = tmp_path_factory.mktemp("hz_stoch") / "work"
    steps = _run(stoch_two_period_db_url, "stoch_horizon", work)
    return steps, work, _write_outputs(steps, work, "stoch_horizon", work.parent / "out")


@pytest.fixture(scope="module")
def recourse_horizon(stoch_two_period_invest_db_url, tmp_path_factory):
    work = tmp_path_factory.mktemp("hz_recourse") / "work"
    steps = _run(stoch_two_period_invest_db_url, "recourse_horizon", work)
    return steps, work, _write_outputs(steps, work, "recourse_horizon", work.parent / "out")


def test_fan_branch_dispatch_in_dump(stoch_horizon) -> None:
    """D1: every forecast-branch flow of the fan is in ``horizon/v_flow.parquet``, unrealized, with the hand
    values; the city balance duals are the weighted marginal thermal cost.  Before: no ``horizon/`` at all."""
    _steps, _work, pq = stoch_horizon
    flows = _dump(pq, "v_flow")
    assert flows.height == 24
    assert set(flows["d"]) == {"p2035_high", "p2040_high", "p2035_low", "p2040_low"}
    assert flows["realized"].to_list() == [False] * 24
    assert set(flows["solve"]) == {"stoch_2p"} and set(flows["roll"]) == {"stoch_2p"}
    expected = {"p2035_high": (60.0, 40.0), "p2040_high": (50.0, 60.0),
                "p2035_low": (20.0, 80.0), "p2040_low": (10.0, 100.0)}
    for r in flows.iter_rows(named=True):
        wind, thermal = expected[r["d"]]
        mw = r["value"] * (100.0 if r["p"] == "wind" else 200.0)
        assert mw == pytest.approx(wind if r["p"] == "wind" else thermal, abs=1e-6), r
    bal = _dump(pq, "nodeBalance_eq")
    assert bal.height == 12 and bal["realized"].to_list() == [False] * 12
    for r in bal.iter_rows(named=True):
        weight = 0.25 if r["d"].endswith("_high") else 0.75
        assert r["dual"] == pytest.approx(50.0 * 2920.0 * weight, rel=1e-9), r


def test_recourse_branch_invest_in_dump(recourse_horizon) -> None:
    """D2: the per-forecast investment decisions (high 60/30, low 80/40) are in ``horizon/v_invest_p.parquet``
    as unrealized rows; every realized row equals the normal investment output of its period."""
    _steps, work, pq = recourse_horizon
    inv = _dump(pq, "v_invest_p").filter(pl.col("p") == "peaker")
    got = {r["d"]: (round(r["value"], 6), r["realized"]) for r in inv.iter_rows(named=True)}
    for d, value in {"p2035_high": 60.0, "p2040_high": 30.0, "p2035_low": 80.0, "p2040_low": 40.0}.items():
        assert got[d] == (value, False), (d, got)
    # The realized investment periods are the weight-0 period labels p2035/p2040, which carry no investment
    # column: the normal output reports 0 there and the dump has no row (nothing realized to mark).
    normal = pl.read_parquet(work / "output_raw" / "v_invest__stoch_2p_inv.parquet")
    normal_by_d = dict(zip(normal["period"].to_list(), normal["peaker"].to_list()))
    assert normal_by_d["p2035"] == 0.0 and normal_by_d["p2040"] == 0.0
    assert not {"p2035", "p2040"} & set(inv["d"])
    for r in inv.filter(pl.col("realized")).iter_rows(named=True):
        assert r["value"] == pytest.approx(normal_by_d[r["d"]], abs=1e-9), r


def _payload(name: str) -> str:
    return name[name.index("[") + 1:-1] if "[" in name else ""


@pytest.mark.parametrize("case", ["stoch_horizon", "recourse_horizon"])
def test_dump_equals_solution(case, request) -> None:
    """D3: every family file holds exactly the family's LP columns / rows, keyed as the LP names them, with
    ``value`` = primal, ``reduced_cost`` = column dual / scale_the_objective and ``dual`` = row dual /
    scale_the_objective; the variable and constraint files together cover every LP column and row."""
    steps, work, pq = request.getfixturevalue(case)
    (step,) = steps.values()
    sol = step.solution
    inv = _inv_scale(work)
    files = {p.stem: p for p in (pq / "horizon").glob("*.parquet")}
    n_cols = 0
    for name, var in sol._vars.items():
        if var.frame.height == 0:
            assert name not in files
            continue
        part = pl.read_parquet(files.pop(name))
        ids = var.frame["col_id"].to_numpy()
        assert part.height == len(ids)
        np.testing.assert_array_equal(part["value"].to_numpy(), sol.col_value[ids])
        np.testing.assert_allclose(part["reduced_cost"].to_numpy(), sol.col_dual[ids] * inv, rtol=1e-12)
        keys = part.select(pl.concat_str(list(var.dims), separator=",")).to_series().to_list() if var.dims else [""]
        assert keys == [_payload(sol.col_names[i]) for i in ids]
        n_cols += len(ids)
    assert n_cols == len(sol.col_value)
    # Constraint rows: polar-high numbers them in family declaration order, which the row names spell out.
    row_family = [n[:n.index("[")] if "[" in n else n for n in sol.row_names]
    order = list(dict.fromkeys(row_family))
    assert set(files) == set(order)
    base = 0
    for name in order:
        part = pl.read_parquet(files[name])
        n = part.height
        assert row_family[base:base + n] == [name] * n
        dims = [c for c in part.columns if c not in ("solve", "roll", "realized", "dual")]
        keys = part.select(pl.concat_str(dims, separator=",")).to_series().to_list() if dims else [""] * n
        assert keys == [_payload(r) for r in sol.row_names[base:base + n]]
        np.testing.assert_allclose(part["dual"].to_numpy(), sol.row_dual[base:base + n] * inv, rtol=1e-12)
        base += n
    assert base == len(sol.row_dual)


# --------------------------------------------------------------------------------------------------------------
# Rolling: 48 h horizon, 12 h jump (``tests.json`` battery chain over ``fullYear``)
# --------------------------------------------------------------------------------------------------------------

BATTERY = ["init", "west", "wind", "coal", "battery", "fullYear", "dispatch_fullYear_roll"]
JUMP = 12


def _roll_values(flag: bool) -> list:
    pmap = {"index_type": "str", "index_name": "period", "data": [["p2020", "fullYear"]]}
    vals = [["solve", "dispatch_fullYear_roll", "period_timeset", b64(pmap, "map")],
            ["solve", "dispatch_fullYear_roll", "realized_periods",
             b64({"value_type": "str", "data": ["p2020"]}, "array")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(48.0, "float")],
            ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(float(JUMP), "float")]]
    return vals + ([FLAG_ON] if flag else [])


@pytest.fixture(scope="module")
def rolling_horizon(tmp_path_factory):
    url, scen = make_db(tmp_path_factory, alt="hz_roll", values=_roll_values(True), base_chain=BATTERY)
    work = tmp_path_factory.mktemp("hz_roll_w") / "work"
    steps = _run(url, scen, work)
    return steps, work, _write_outputs(steps, work, scen, work.parent / "out")


def test_rolling_dump_appends_rolls(rolling_horizon) -> None:
    """D4: one ``roll`` per sub-solve in manifest order; within a roll, the steps up to its jump are realized and
    the look-ahead steps are not; the realized flows equal that roll's ``output_raw/v_flow__<roll>`` values."""
    steps, work, pq = rolling_horizon
    manifest = (work / "output_raw" / "_solve_order.txt").read_text().split()
    assert len(manifest) >= 2 and manifest == list(steps)
    flows = _dump(pq, "v_flow")
    assert list(dict.fromkeys(flows["roll"].to_list())) == manifest
    for i, roll in enumerate(manifest):
        part = flows.filter(pl.col("roll") == roll).with_columns(
            pl.col("t").str.slice(1).cast(pl.Int64).alias("_k"))
        last_realized = JUMP * (i + 1)
        assert part.filter(pl.col("_k") > last_realized)["realized"].to_list() == \
            [False] * part.filter(pl.col("_k") > last_realized).height
        assert part.filter(pl.col("_k") > last_realized).height > 0 or i == len(manifest) - 1
        inside = part.filter(pl.col("_k") <= last_realized)
        assert inside.height > 0 and inside["realized"].all(), roll
        normal = pl.read_parquet(work / "output_raw" / f"v_flow__{roll}.parquet")
        for r in inside.iter_rows(named=True):
            col = str((r["p"], r["source"], r["sink"]))
            match = normal.filter((pl.col("period") == r["d"]) & (pl.col("time") == r["t"]))
            assert match.height == 1, (roll, r)
            assert match[col][0] == pytest.approx(r["value"], rel=1e-12, abs=1e-12), (roll, r)


def test_flag_off_has_no_dump_and_clears_stale(stoch_horizon, stoch_two_period_db_url, tmp_path_factory) -> None:
    """D5: without the flag nothing is dumped (no ``horizon/`` under ``output_raw`` or the parquet folder), and
    a flag-off run written to a location that holds an earlier flag-on dump removes it."""
    _steps, _work, pq_on = stoch_horizon
    out = tmp_path_factory.mktemp("hz_stale")
    shutil.copytree(pq_on, out / "output_parquet" / "stoch")
    assert (out / "output_parquet" / "stoch" / "horizon").is_dir()
    work = out / "work"
    steps = _run(stoch_two_period_db_url, "stoch", work)
    assert not (work / "output_raw" / "horizon").exists()
    pq = _write_outputs(steps, work, "stoch", out)
    assert not (pq / "horizon").exists()
    assert any(pq.glob("*.parquet"))


def _lh2_chain() -> list[str]:
    data = json.loads(LH2_FIXTURE_JSON.read_text())
    rows = [r for r in data["scenario_alternatives"] if r[0] == "lh2_three_region_invest"]
    nxt = {a: b for _, a, b in rows}
    alt = next(a for a in nxt if a not in set(nxt.values()))
    chain = []
    while alt is not None:
        chain.append(alt)
        alt = nxt.get(alt)
    return chain


def test_benders_solve_warns_and_is_not_dumped(tmp_path_factory, caplog, capfd) -> None:
    """D6: a Benders solve writes no outputs, so under the flag it gives one warning naming the solve (also
    printed to the console once per run: the runner's logger is pinned at ERROR) and writes no parts; the
    monolithic dispatch solve after it is dumped."""
    url, scen = make_db(tmp_path_factory, alt="hz_benders", values=[FLAG_ON], base_chain=_lh2_chain(),
                        fixture_json=LH2_FIXTURE_JSON)
    work = tmp_path_factory.mktemp("hz_benders_w") / "work"
    with caplog.at_level(logging.WARNING):
        steps = _run(url, scen, work)
    benders = [n for n, s in steps.items() if getattr(s, "is_benders", False)]
    assert benders, list(steps)
    others = [n for n in steps if n not in benders]
    assert others, list(steps)
    msgs = [r.getMessage() for r in caplog.records if "raw horizon dump is not available" in r.getMessage()]
    assert len(msgs) == len(benders) and all(any(b in m for m in msgs) for b in benders)
    assert capfd.readouterr().out.count("raw horizon dump is not available for Benders solves") == 1
    parts = list((work / "output_raw" / "horizon").glob("*/*.parquet"))
    rolls = {p.stem.split("__", 1)[1] for p in parts}
    assert rolls == set(others)
