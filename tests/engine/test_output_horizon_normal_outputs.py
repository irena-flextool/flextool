"""Normal outputs hold realized rows only, with or without ``model.output_horizon``.

``model.output_horizon`` writes a separate raw LP dump (``horizon/``, ``tests/engine/test_output_horizon_dump.py``);
it never changes the normal outputs.  A solve's forecast-branch periods and the look-ahead steps of a roll are not
realized, so no normal output frame holds them.

Absolute check: the periods the LP holds (``d`` of every variable of every sub-solve) minus the hand-derived
realized ones (``REALIZED``) never appear in a final ``output_parquet/<scenario>`` frame, and every ``(period,
time)`` row of a per-roll ``output_raw`` file (of a final frame) is a step that roll (the run) realizes.  Relational check: flag on and flag off give equal
``output_raw/`` files (the raw dump aside) and equal final frames.

Cases (fixtures built from JSON): ``stoch_two_period`` ``stoch`` / ``stoch_horizon`` (forecasts-only fan, no
realization solve), ``stoch_two_period_invest`` ``recourse`` / ``recourse_horizon`` (per-forecast investment) and
``stochastics`` ``1_week_rolling_wind`` (rolling, a forecast fan in each roll's look-ahead; the fixture enables the
flag, the flag-off twin drops that value).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tests.engine.test_output_horizon_dump import _run, _write_outputs
from tests.engine._unit_db import FLEXTOOL_ROOT

pytestmark = pytest.mark.solver

FIXTURES = FLEXTOOL_ROOT / "tests" / "fixtures"


def _db_without_flag(tmp_path_factory, fixture: str) -> str:
    """``fixture`` with its ``model.output_horizon`` value removed."""
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / fixture).read_text())
    before = len(data["parameter_values"])
    data["parameter_values"] = [pv for pv in data["parameter_values"]
                                if not (pv[0] == "model" and pv[2] == "output_horizon")]
    assert len(data["parameter_values"]) == before - 1
    work = tmp_path_factory.mktemp("db_noflag")
    path = work / fixture
    path.write_text(json.dumps(data))
    url = json_to_db(path, work / "db.sqlite")
    migrate_database(url)
    return url


def _case(url: str, scenario: str, tmp_path_factory, tag: str) -> dict:
    work = tmp_path_factory.mktemp(tag) / "work"
    steps = _run(url, scenario, work)
    return {"steps": steps, "work": work, "scenario": scenario,
            "pq": _write_outputs(steps, work, scenario, work.parent / "out")}


@pytest.fixture(scope="module")
def runs(stoch_two_period_db_url, stoch_two_period_invest_db_url, stochastic_db_url, tmp_path_factory) -> dict:
    out = {}
    for scen in ("stoch", "stoch_horizon"):
        out[scen] = _case(stoch_two_period_db_url, scen, tmp_path_factory, f"no_{scen}")
    for scen in ("recourse", "recourse_horizon"):
        out[scen] = _case(stoch_two_period_invest_db_url, scen, tmp_path_factory, f"no_{scen}")
    out["rolling_horizon"] = _case(stochastic_db_url, "1_week_rolling_wind", tmp_path_factory, "no_roll_on")
    out["rolling"] = _case(_db_without_flag(tmp_path_factory, "stochastics.json"), "1_week_rolling_wind",
                           tmp_path_factory, "no_roll_off")
    return out


def _window(k: int) -> set[tuple[str, str]]:
    """Steps realized by roll ``k`` of ``1week_rolling`` (jump 24 of 168 one-hour steps)."""
    return {("period1", f"t{24 * k + i:04d}") for i in range(1, 25)}


# Hand-derived realization of each case, from the fixture JSON:
# * ``stoch_2p`` / ``stoch_2p_inv``: the fan opens in the first period, so no step is realized (forecasts only, no
#   realization solve); the realized periods are the period labels ``p2035`` / ``p2040`` (no dispatch).
# * ``1week_rolling``: rolling_solve_horizon 96, rolling_solve_jump 24, one period ``period1`` of 168 steps and a
#   ``realized`` forecast designated at every jump start, so each roll ``k`` is a fan solve that realizes nothing
#   plus a realization solve ``1week_rolling__realized_dispatch_roll_k`` that realizes steps 24k+1 .. 24k+24.
_FAN = {"periods": {"p2035", "p2040"}, "windows": {}}
_ROLL = {"periods": {"period1"},
         "windows": {f"1week_rolling__realized_dispatch_roll_{k}": _window(k) for k in range(7)}}
REALIZED = {"stoch": _FAN, "stoch_horizon": _FAN, "recourse": _FAN, "recourse_horizon": _FAN,
            "rolling": _ROLL, "rolling_horizon": _ROLL}


def _unrealized_lp_periods(steps: dict, realized_periods: set[str]) -> set[str]:
    """The periods of the LP variables of every sub-solve that are not realized."""
    lp_d: set[str] = set()
    for step in steps.values():
        for name in step.solution._vars:
            values = step.solution.value(name)
            if "d" in values.columns:
                lp_d |= set(values["d"].cast(str))
    return lp_d - realized_periods


def _final_frames(pq: Path) -> dict[str, pl.DataFrame]:
    return {str(p.relative_to(pq)): pl.read_parquet(p) for p in sorted(pq.rglob("*.parquet"))
            if "horizon" not in p.relative_to(pq).parts}


@pytest.mark.parametrize("case", ["stoch", "stoch_horizon", "recourse", "recourse_horizon",
                                  "rolling", "rolling_horizon"])
def test_no_unrealized_rows_in_normal_outputs(runs, case) -> None:
    """No final frame names an unrealized LP period (a forecast-branch copy) and no ``(period, time)`` row of a
    per-roll ``output_raw`` file or of a final frame lies outside the steps realized by that roll / by the run.
    Before, with the flag on, the forecast branches' dispatch, per-forecast investment and each roll's
    look-ahead were in the normal outputs; flag off, a forecasts-only fan still reported its forecast-branch rows
    (costs, prices, capacities)."""
    run = runs[case]
    windows = REALIZED[case]["windows"]
    unrealized = _unrealized_lp_periods(run["steps"], REALIZED[case]["periods"])
    assert unrealized, "the case has no unrealized LP period: it checks nothing"
    # The per-solve dual files (``v_dual_*``) of a solve that realizes no step hold every step of it (the price
    # source of handoffs such as a nested parent's ``fix_price``); their unrealized rows are cut from the final
    # frames, checked below.
    for path in sorted((run["work"] / "output_raw").glob("*__*.parquet")):
        if path.name.startswith("v_dual_"):
            continue
        rolls = [name for name in run["steps"] if path.stem.endswith(f"__{name}")]
        assert rolls, path.name
        roll = max(rolls, key=len)
        df = pl.read_parquet(path)
        if {"period", "time"} <= set(df.columns) and df.height:
            rows = set(zip(df["period"].cast(str), df["time"].cast(str)))
            outside = rows - windows.get(roll, set())
            assert not outside, (path.name, sorted(outside)[:5])
    every_step = set().union(*windows.values()) if windows else set()
    frames = _final_frames(run["pq"])
    assert frames
    for name, df in frames.items():
        for col in df.columns:
            assert not any(f"'{p}'" in col or col == p for p in unrealized), (name, col)
            if df[col].dtype == pl.Utf8:
                hit = set(df[col].drop_nulls().unique()) & unrealized
                assert not hit, (name, col, sorted(hit))
        if {"period", "time"} <= set(df.columns) and df.height:
            rows = set(zip(df["period"].cast(str), df["time"].cast(str)))
            assert rows <= every_step, (name, sorted(rows - every_step)[:5])


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    """Columns in name order: the entity-column order of some per-roll parameter files (``entity_all_existing``)
    is not deterministic between runs, flag or not."""
    return df.select(sorted(df.columns))


def _canonical(df: pl.DataFrame) -> pl.DataFrame:
    """A per-roll set file (``__set_marker__`` column) as a sorted row set: the row order of some sets
    (``period_in_use``) is not deterministic between runs, flag or not.  Other files: :func:`_sorted`."""
    df = _sorted(df)
    if "__set_marker__" in df.columns:
        return df.drop("__set_marker__").sort(pl.all())
    return df


def _manifest_files(path: Path) -> list[dict]:
    """The file entries of ``output_raw/manifest.json`` without the run's location (``bundle_root`` and
    ``generated_at`` are run-specific); the raw dump's entries are left out."""
    files = json.loads(path.read_text())["files"]
    return [f for f in files if "horizon" not in Path(f.get("path", "")).parts]


def _normalised(df: pl.DataFrame, scenario: str) -> pl.DataFrame:
    return df.rename({c: "S" if c == scenario else c.replace(f"'{scenario}'", "'S'")
                      for c in df.columns if c == scenario or f"'{scenario}'" in c})


@pytest.mark.parametrize("off,on", [("stoch", "stoch_horizon"), ("recourse", "recourse_horizon"),
                                    ("rolling", "rolling_horizon")])
def test_normal_outputs_independent_of_flag(runs, off, on) -> None:
    """Flag on equals flag off: every ``output_raw/`` file except the raw dump and every final frame."""
    a, b = runs[off], runs[on]
    raw_a, raw_b = a["work"] / "output_raw", b["work"] / "output_raw"
    files_a = sorted(str(p.relative_to(raw_a)) for p in raw_a.rglob("*") if p.is_file()
                     and "horizon" not in p.relative_to(raw_a).parts)
    files_b = sorted(str(p.relative_to(raw_b)) for p in raw_b.rglob("*") if p.is_file()
                     and "horizon" not in p.relative_to(raw_b).parts)
    assert files_a == files_b
    for rel in files_a:
        if rel.endswith(".parquet"):
            assert _canonical(pl.read_parquet(raw_a / rel)).equals(_canonical(pl.read_parquet(raw_b / rel))), rel
        elif rel == "manifest.json":
            assert _manifest_files(raw_a / rel) == _manifest_files(raw_b / rel)
        else:
            assert (raw_a / rel).read_bytes() == (raw_b / rel).read_bytes(), rel
    fa, fb = _final_frames(a["pq"]), _final_frames(b["pq"])
    assert sorted(fa) == sorted(fb)
    for rel in fa:
        assert _sorted(_normalised(fa[rel], a["scenario"])).equals(_sorted(_normalised(fb[rel], b["scenario"]))), rel
