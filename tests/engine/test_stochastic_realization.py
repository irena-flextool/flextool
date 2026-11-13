"""The realization solve of a stochastic fan (owner decision "Stochastic realization model (2026-11-01)").

A stochastic solve optimises over forecasts only.  Its realized operations come from a realization solve that runs
right after it over the same committed window, with its investments fixed and its non-anticipativity-tied unit
commitment and reserves inherited (the value of the first sorted forecast, on the tied cells).  A ``realized: yes``
designation whose fan reaches the committed window makes the engine add that solve when none is configured.

Fixtures (built from JSON, CLAUDE.md invariant 3): ``stoch_t0_commit.json`` — a single solve over ``p2035`` (3 hourly
steps), branches ``rlz`` (realized: yes, weight 0), ``high`` / ``low`` (0.5 each) revealed at t0001; a committable
``peaker`` (2 units of 50 MW, min_load 0.5) backs ``wind`` (100 MW, CF rlz 0.5 / high 0.8 / low 0.0) against 100 MW
demand; it authors ``dispatch_realization_method = separate_dispatch``, which the variants here drop.
``w4b_reserve_dispatch.json`` — the same structure with an up-reserve of 10 MW on the peaker.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from tests.engine._unit_db import is_realization, utf8

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
T0_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_t0_commit.json"
RESERVE_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "w4b_reserve_dispatch.json"

PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"
ANCHOR = "p2035"
FIRST_FORECAST = "p2035_high"       # sorted forecasts of p2035: p2035_high < p2035_low
STEPS = ("t0001", "t0002", "t0003")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _enc(value) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode(), "str"]


def _variant(src: Path, tmp: Path, name: str, mutate) -> Path:
    data = json.loads(src.read_text(encoding="utf-8"))
    mutate(data)
    out = tmp / f"{name}.json"
    out.write_text(json.dumps(data), encoding="utf-8")
    return out


def _drop_separate_dispatch(data: dict) -> None:
    data["parameter_values"] = [r for r in data["parameter_values"] if r[2] != "dispatch_realization_method"]


def _set_profile(data: dict, values: dict) -> None:
    """Set the ``wprof`` CF per branch (constant over the steps)."""
    for r in data["parameter_values"]:
        if r[1] == "wprof" and r[2] == "profile":
            m = json.loads(base64.b64decode(r[3][0]).decode())
            for branch in m["data"]:
                for ts in branch[1]["data"]:
                    for cell in ts[1]["data"]:
                        cell[1] = values[branch[0]]
            r[3] = [base64.b64encode(json.dumps(m).encode()).decode(), r[3][1]]


def _hard_node(data: dict) -> None:
    data["parameter_values"].append(["node", "elec", "penalty_method", _enc("off"), "base"])


def _run(fixture_json: Path, tmp: Path, *, scenario: str = "stoch_t0_commit", capture_mps: bool = False):
    """Run the chain (FLEXTOOL_AUTOSCALE_STRICT=1, warm=False); return ``(steps, {solve: mps bytes})``."""
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core as model_core
    from flextool.update_flextool.db_migration import migrate_database

    url = json_to_db(fixture_json, tmp / f"{fixture_json.stem}.sqlite")
    migrate_database(url)
    work = tmp / "work"
    work.mkdir()
    mps: dict[str, bytes] = {}
    original = model_core.build_flextool

    def capturing(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        frame = sys._getframe(1)
        while frame is not None and not isinstance(frame.f_locals.get("complete_solve_name"), str):
            frame = frame.f_back
        name = frame.f_locals["complete_solve_name"] if frame is not None else "solve"
        path = tmp / f"{name}.mps"
        m.write_mps(path, emit_names=True)
        mps[name] = path.read_bytes()
        return result

    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    if capture_mps:
        model_core.build_flextool = capturing
    try:
        steps = run_chain_from_db(url, scenario, work_folder=work, warm=False, keep_solutions=True)
    finally:
        model_core.build_flextool = original
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    return steps, mps


def _value(steps, solve: str, var: str) -> pl.DataFrame:
    assert solve in steps, f"no {solve!r} step in {list(steps)}"
    sol = steps[solve].solution
    assert var in sol._vars, f"{var} not in {solve}: {sorted(sol._vars)}"
    return sol.value(var).with_columns(pl.col(c).cast(pl.Utf8) for c in sol._vars[var].dims)


def _at(frame: pl.DataFrame, **keys) -> float:
    row = frame.filter(pl.all_horizontal(*(pl.col(k) == v for k, v in keys.items())))
    assert row.height == 1, (keys, row)
    return float(row["value"][0])


@pytest.fixture(scope="module")
def designated(tmp_path_factory):
    """The designated single solve WITHOUT dispatch_realization_method (the engine adds the realization solve)."""
    tmp = tmp_path_factory.mktemp("realization_designated")
    return _run(_variant(T0_FIXTURE, tmp, "designated", _drop_separate_dispatch), tmp, capture_mps=True)


@pytest.fixture(scope="module")
def separate(tmp_path_factory):
    """The same solve WITH dispatch_realization_method = separate_dispatch."""
    tmp = tmp_path_factory.mktemp("realization_separate")
    return _run(T0_FIXTURE, tmp, capture_mps=True)


# ---------------------------------------------------------------------------
# auto-spawn
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_designation_adds_the_realization_solve(designated) -> None:
    """A realized: yes designation reaching the committed window adds ``<solve>__realized_dispatch`` right after
    the stochastic solve, though no dispatch_realization_method is authored."""
    steps, _ = designated
    assert list(steps) == [PARENT, SIBLING]


@pytest.mark.solver
def test_auto_spawned_realization_solve_equals_separate_dispatch(designated, separate) -> None:
    """The auto-spawned realization solve builds the same LP as the one separate_dispatch adds, and the stochastic
    solve's LP does not depend on which of the two added it."""
    _, mps_auto = designated
    _, mps_sep = separate
    assert set(mps_auto) == set(mps_sep) == {PARENT, SIBLING}
    assert mps_auto[SIBLING] == mps_sep[SIBLING]
    assert mps_auto[PARENT] == mps_sep[PARENT]


@pytest.mark.solver
def test_realization_online_is_first_sorted_forecast(designated) -> None:
    """The realization solve's units online equal the stochastic solve's first sorted forecast (``p2035_high``) on
    the tied cells (the committed window p2035 t0001-t0003), re-keyed to the anchor period."""
    steps, _ = designated
    parent = _value(steps, PARENT, "v_online_linear")
    sib = _value(steps, SIBLING, "v_online_linear")
    for t in STEPS:
        assert _at(sib, p="peaker", d=ANCHOR, t=t) == pytest.approx(
            _at(parent, p="peaker", d=FIRST_FORECAST, t=t), abs=1e-9)


@pytest.mark.solver
def test_realization_reserve_is_first_sorted_forecast(tmp_path_factory) -> None:
    """The realization solve's reserve is pinned (reserve_fix_eq) to the stochastic solve's first sorted forecast on
    the tied cells: v_reserve of the realization solve == v_reserve of ``p2035_high`` at every step."""
    tmp = tmp_path_factory.mktemp("realization_reserve")
    steps, mps = _run(RESERVE_FIXTURE, tmp, scenario="w8_realized_dispatch", capture_mps=True)
    assert b"reserve_fix_eq" in mps[SIBLING]
    parent = _value(steps, PARENT, "v_reserve")
    sib = _value(steps, SIBLING, "v_reserve")
    for t in STEPS:
        assert _at(sib, p="peaker", d=ANCHOR, t=t) == pytest.approx(
            _at(parent, p="peaker", d=FIRST_FORECAST, t=t), abs=1e-9)


# ---------------------------------------------------------------------------
# feasibility pre-check on a hard-balance node
# ---------------------------------------------------------------------------

def _hard_variant(cf: dict):
    def mutate(data: dict) -> None:
        _drop_separate_dispatch(data)
        _hard_node(data)
        _set_profile(data, cf)
    return mutate


@pytest.mark.solver
def test_hard_node_realized_shortfall_raises_before_solve(tmp_path_factory) -> None:
    """``elec`` has no balance slack.  Forecast CF high 0.8 / low 0.6: high needs 20 MW of peaker, so the min_load
    floor 0.5 · 50 · v_online ≤ 20 keeps v_online ≤ 0.8; low needs 40 MW, so v_online ≥ 0.8 — the tied commitment is
    0.8 units = 40 MW.  Realized CF 0.0: the realized need is 100 MWh per step against at most 40 (peaker) + 0 (wind)
    MWh, so the realization solve raises, naming the node and the first step, before it is solved."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    tmp = tmp_path_factory.mktemp("realization_hard_short")
    jf = _variant(T0_FIXTURE, tmp, "hard_short", _hard_variant({"rlz": 0.0, "high": 0.8, "low": 0.6}))
    with pytest.raises(FlexToolConfigError, match=r"cannot balance node 'elec'.*'p2035', step 't0001'"):
        _run(jf, tmp)


@pytest.mark.solver
def test_hard_node_realized_within_pin_solves(tmp_path_factory) -> None:
    """Same commitment (0.8 units = 40 MW); realized CF 0.7: wind ≤ 70 MW and the peaker 20..40 MW cover the 100 MW
    demand (peaker 30 MW), so the check passes and the realization solve reaches an optimum."""
    tmp = tmp_path_factory.mktemp("realization_hard_ok")
    jf = _variant(T0_FIXTURE, tmp, "hard_ok", _hard_variant({"rlz": 0.7, "high": 0.8, "low": 0.6}))
    steps, _ = _run(jf, tmp)
    assert SIBLING in steps and steps[SIBLING].solution is not None
    sib = _value(steps, SIBLING, "v_online_linear")
    assert _at(sib, p="peaker", d=ANCHOR, t="t0001") == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# registration rules (config level, no solve)
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self):
        self.registered: list[tuple[str, str]] = []

    def entities(self, _cls):
        return pl.DataFrame({"name": ["p"]})

    def register_derived_solve(self, parent, sibling, **_kw):
        self.registered.append((parent, sibling))


def _config(**over):
    from collections import defaultdict
    from flextool.engine.input._solve_config import SolveConfig
    cfg = SimpleNamespace(
        model_solve={"m": ["p"]}, contains_solves={}, solve_modes={"p": "single_solve"}, rolling_times={},
        stochastic_branches={"p": [["d1", "rlz", "t0001", "yes", 0.0], ["d1", "a", "t0001", "no", 1.0]]},
        realized_periods=defaultdict(list, {"p": [("d1", "d1")]}),
        timesets_used_by_solves={"p": [("d1", "ts")]}, solve_period_years_represented={},
        stochastic_invest_method={},
        dispatch_realization_method={}, fix_storage_periods={}, realized_dispatch_parent={},
        reduce_unit_detail_method={}, reduce_system_detail_method={},
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    for name in ("dispatch_realization_method_for", "stochastic_invest_method_for", "register_realization_solve",
                 "spawn_realized_dispatch_sibling", "realization_reader_values", "realization_window",
                 "reduce_detail_overridden", "_reduce_detail_for",
                 "reduce_unit_detail_for", "reduce_system_detail_for"):
        setattr(cfg, name, getattr(SolveConfig, name).__get__(cfg))
    return cfg


def test_configured_realization_solve_is_registered_not_spawned() -> None:
    """A deterministic solve right after the stochastic solve, same mode and rolls, realizing its committed fan
    period, is registered as its realization solve; no sibling is added."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(model_solve={"m": ["p", "q"]}, solve_modes={"p": "single_solve", "q": "single_solve"})
    cfg.realized_periods["q"] = [("d1", "d1")]
    reader = _Reader()
    _spawn_realized_dispatch_siblings(cfg, reader, logger=None)
    assert cfg.realized_dispatch_parent == {"q": "p"}
    assert reader.registered == []
    assert cfg.model_solve["m"] == ["p", "q"]


def test_configured_solve_with_a_lookahead_period_warns(capsys) -> None:
    """An author-declared realization solve of a single solve that also holds a period the stochastic solve does
    not realize dispatches it with the realized data, information the stochastic solve did not have: registered as
    authored, with a warning naming the period (item 255)."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(model_solve={"m": ["p", "q"]}, solve_modes={"p": "single_solve", "q": "single_solve"})
    cfg.timesets_used_by_solves["q"] = [("d1", "ts"), ("d2", "ts")]
    cfg.realized_periods["q"] = [("d1", "d1")]
    _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)
    assert cfg.realized_dispatch_parent == {"q": "p"}
    out = capsys.readouterr().out
    assert ("Warning: Realization solve 'q' of solve 'p' also dispatches 'd2', which 'p' does not realize: it "
            "dispatches that period with the realized data, information 'p' did not have when it decided.") in out


def test_configured_solve_missing_the_fan_period_is_not_registered() -> None:
    """A following deterministic solve that does not realize the fan's committed period is not a realization
    solve: the designation then adds the engine sibling right after the stochastic solve."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(model_solve={"m": ["p", "q"]}, solve_modes={"p": "single_solve", "q": "single_solve"})
    cfg.realized_periods["q"] = [("d2", "d2")]
    reader = _Reader()
    _spawn_realized_dispatch_siblings(cfg, reader, logger=None)
    assert cfg.realized_dispatch_parent == {"p__realized_dispatch": "p"}
    assert cfg.model_solve["m"] == ["p", "p__realized_dispatch", "q"]


def test_recourse_designation_warns_and_adds_nothing(caplog) -> None:
    """Under recourse there is no realized post-reveal investment to fix: a warning, no realization solve."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(stochastic_invest_method={"p": "recourse"})
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._realization"):
        _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)
    assert cfg.realized_dispatch_parent == {}
    assert any("recourse" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("nesting", ["parent", "child"])
def test_nested_designation_raises(nesting: str) -> None:
    """A designated solve that contains solves, or is nested in another solve, cannot get a realization solve."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    if nesting == "parent":
        cfg = _config(contains_solves={"p": ["c"]})
    else:
        cfg = _config(model_solve={"m": ["top"]}, contains_solves={"top": ["p"]})
    with pytest.raises(FlexToolConfigError, match="realized: yes designation"):
        _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)


def test_designation_outside_the_committed_window_adds_nothing() -> None:
    """A designated fan opened at a period after every realized period does not reach the committed window."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(stochastic_branches={"p": [["d2", "rlz", "t0001", "yes", 0.0], ["d2", "a", "t0001", "no", 1.0]]},
                  timesets_used_by_solves={"p": [("d1", "ts"), ("d2", "ts")]})
    _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)
    assert cfg.realized_dispatch_parent == {}


# ---------------------------------------------------------------------------
# A period label has no dispatch
# ---------------------------------------------------------------------------

FIXTURES = FLEXTOOL_ROOT / "tests" / "fixtures"
TWO_PERIOD = FIXTURES / "stoch_two_period.json"
RECOURSE = FIXTURES / "stoch_two_period_invest.json"
HEDGE = FIXTURES / "stoch_two_period_hedge.json"
STOCHASTICS = FIXTURES / "stochastics.json"


def _mps_rows(mps: bytes) -> set[str]:
    """Row names of an MPS file (the ROWS section)."""
    rows, section = set(), None
    for line in mps.decode().splitlines():
        if not line.startswith(" "):
            section = line.split()[0] if line.split() else section
            continue
        if section == "ROWS":
            rows.add(line.split()[1])
    return rows


def _mps_entries(mps: bytes) -> list[tuple[str, str]]:
    """``(column, row)`` entries of the COLUMNS section of an MPS file."""
    out, section = [], None
    for line in mps.decode().splitlines():
        if not line.startswith(" "):
            section = line.split()[0] if line.split() else section
            continue
        f = line.split()
        if section == "COLUMNS" and "MARKER" not in line:
            out += [(f[0], f[i]) for i in range(1, len(f) - 1, 2)]
    return out


def _mps_columns(mps: bytes) -> set[str]:
    return {c for c, _r in _mps_entries(mps)}


def _with_label_dt(names, label: str) -> list[str]:
    """Names indexed by the period *label* together with a time step (``[..,label,t0001..]``)."""
    import re
    pattern = re.compile(rf"[\[,]{re.escape(label)},t\d{{4}}[,\]]")
    return sorted(n for n in names if pattern.search(n))


def _branch_map_without(data: dict, solve: str, drop: str) -> None:
    """Remove branch *drop* from *solve*'s ``stochastic_branches``."""
    for r in data["parameter_values"]:
        if r[0] == "solve" and r[1] == solve and r[2] == "stochastic_branches":
            m = json.loads(base64.b64decode(r[3][0]).decode())
            for _period, period_map in m["data"]:
                period_map["data"] = [b for b in period_map["data"] if b[0] != drop]
            r[3] = [base64.b64encode(json.dumps(m).encode()).decode(), r[3][1]]


def _all_forecasts(data: dict, solve: str) -> None:
    """Flag every branch of *solve*'s ``stochastic_branches`` ``realized: no`` (a forecasts-only fan)."""
    for r in data["parameter_values"]:
        if r[0] == "solve" and r[1] == solve and r[2] == "stochastic_branches":
            m = json.loads(base64.b64decode(r[3][0]).decode())
            for _period, period_map in m["data"]:
                for _branch, branch_map in period_map["data"]:
                    for _step, leaf in branch_map["data"]:
                        leaf["data"] = [["no", leaf["data"][0][1]]]
            r[3] = [base64.b64encode(json.dumps(m).encode()).decode(), r[3][1]]


def _objective(steps, solve: str, tmp: Path) -> float:
    return float(pl.read_parquet(tmp / "work" / "output_raw" / f"v_obj__{solve}.parquet")["objective"][0])


@pytest.mark.solver
def test_forecasts_only_fan_has_no_label_dispatch(tmp_path_factory) -> None:
    """``stoch_two_period`` (forecasts-only {high, low} fan at p2035, continuing into p2040): the stochastic solve
    solves; no row or column of the LP is indexed by a period label with a time step; the objective is the hand
    calculation 2920 × 24 000 = 70 080 000 of ``test_stochastic_continuation_fanout`` (also the objective of the
    parent before #82, 75f73acf^, when the fixture still solved)."""
    tmp = tmp_path_factory.mktemp("label_two_period")
    steps, mps = _run(TWO_PERIOD, tmp, scenario="stoch", capture_mps=True)
    assert list(steps) == ["stoch_2p"]
    names = _mps_rows(mps["stoch_2p"]) | _mps_columns(mps["stoch_2p"])
    for label in ("p2035", "p2040"):
        assert _with_label_dt(names, label) == [], label
    assert _objective(steps, "stoch_2p", tmp) == pytest.approx(70_080_000.0, rel=1e-9)


@pytest.mark.solver
def test_designation_leaves_the_stochastic_lp_unchanged(tmp_path_factory, designated) -> None:
    """``stoch_t0_commit`` as a forecasts-only {high, low} fan and with the extra ``rlz`` (realized: yes, weight 0)
    designation: the stochastic solve's LP is byte-identical (the designation only feeds the realization solve)."""
    tmp = tmp_path_factory.mktemp("label_forecasts_only")

    def forecasts_only(data: dict) -> None:
        _drop_separate_dispatch(data)
        _branch_map_without(data, PARENT, "rlz")

    steps, mps = _run(_variant(T0_FIXTURE, tmp, "forecasts_only", forecasts_only), tmp, capture_mps=True)
    assert list(steps) == [PARENT]
    _, mps_designated = designated
    assert mps[PARENT] == mps_designated[PARENT]


@pytest.mark.solver
def test_recourse_label_has_no_investment(tmp_path_factory) -> None:
    """Under recourse a period label invests nothing here-and-now: ``stoch_two_period_invest`` has no ``v_invest``
    column at p2035 / p2040, only at the forecast branch periods, and the objective equals the parent before #82
    (75f73acf^, lp_snapshot ``stoch_recourse``: 0.196875 in objective units, here unscaled 196 875)."""
    tmp = tmp_path_factory.mktemp("label_recourse")
    steps, mps = _run(RECOURSE, tmp, scenario="recourse", capture_mps=True)
    cols = [c for c in _mps_columns(mps["stoch_2p_inv"]) if c.startswith("v_invest")]
    assert cols
    assert not [c for c in cols if ",p2035]" in c or ",p2040]" in c], sorted(cols)
    assert _objective(steps, "stoch_2p_inv", tmp) == pytest.approx(196_875.0, rel=1e-9)


@pytest.mark.solver
def test_recourse_shared_investment_ties_forecasts_to_each_other(tmp_path_factory) -> None:
    """``hedge_na_window`` (recourse, fan at p2035, shared_invest_periods p2035): the shared-investment window ties
    the forecasts in the forecast star of each window period, members − 1 rows per entity and window period, never a
    forecast to the period label (which has no investment column under recourse)."""
    tmp = tmp_path_factory.mktemp("label_recourse_na")
    steps, mps = _run(HEDGE, tmp, scenario="hedge_na_window", capture_mps=True)
    (solve,) = [s for s in steps if not is_realization(s)]
    rows = sorted(r for r in _mps_rows(mps[solve]) if r.startswith("non_anticipativity_invest"))
    assert rows
    fd = steps[solve].flex_data
    pairs = utf8(fd.pd_non_anticipativity)
    members = utf8(fd.period_branch_full).filter(pl.col("d") != pl.col("b"))
    labels = set(members["d"].to_list())
    assert not (set(pairs["d"].to_list()) | set(pairs["b"].to_list())) & labels
    in_use = set(utf8(fd.period_in_use_set)["d"].to_list())
    for d_ref, group in pairs.group_by("d"):
        anchor = members.filter(pl.col("b") == d_ref[0])["d"][0]
        forecasts = set(members.filter((pl.col("d") == anchor) & pl.col("b").is_in(list(in_use)))["b"].to_list())
        assert group.height == len(forecasts) - 1, (anchor, group)


def _rolling_hedge(data: dict) -> None:
    """``hedge_storage`` (fan at p2040 = t0004, storage ``resv`` with fix_start) as a rolling solve, horizon 4 h,
    jump 3 h: roll 0 (t0001-t0004) commits t0001-t0003 before the reveal, the fan only in its look-ahead; roll 1
    (t0004-t0006) opens the fan at its first step.  The fan is forecasts-only (high / low, no realized branch), so
    the wind profile is plain: 0.8 at p2035, 0.0 at p2040 (a forecasts-only fan has no series for roll 0's
    p2035 before the reveal, so a stochastic value there is an error, item 250)."""
    data["alternatives"].append(["roll", "rolling, fan in the look-ahead"])
    for sa in data["scenario_alternatives"]:
        if sa[0] == "hedge_storage" and sa[1] == "storage" and sa[2] is None:
            sa[2] = "roll"
    data["scenario_alternatives"].append(["hedge_storage", "roll", None])

    def enc(value, kind):
        return [base64.b64encode(json.dumps(value).encode()).decode(), kind]

    def nested(pairs):
        return {"type": "map", "index_type": "str", "data": [[k, v] for k, v in pairs]}
    forecasts = {"index_type": "str", "data": [["p2040", nested([
        ("high", nested([("t0004", nested([("no", 1.0)]))])), ("low", nested([("t0004", nested([("no", 3.0)]))]))])]]}
    profile = {"index_type": "str", "data": [[f"t{i:04d}", 0.8 if i <= 3 else 0.0] for i in range(1, 7)]}
    data["parameter_values"] += [
        ["solve", "stoch_2p_hedge", "stochastic_branches", enc(forecasts, "map"), "roll"],
        ["profile", "wprof", "profile", enc(profile, "map"), "roll"],
        ["solve", "stoch_2p_hedge", "solve_mode", enc("rolling_window", "str"), "roll"],
        ["solve", "stoch_2p_hedge", "rolling_solve_horizon", enc(4.0, "float"), "roll"],
        ["solve", "stoch_2p_hedge", "rolling_solve_jump", enc(3.0, "float"), "roll"],
        ["solve", "stoch_2p_hedge", "stochastic_invest_method", enc("none", "str"), "roll"]]


@pytest.mark.solver
def test_rolling_fan_in_the_look_ahead_carries_the_committed_state(tmp_path_factory) -> None:
    """A rolling fan only in the look-ahead needs no designation: roll 1 starts from roll 0's committed end state,
    v_state of the pre-reveal period p2035 at t0003 (its last committed step) × the state unitsize, and every forecast of
    roll 1's fan (opened at its first step) starts from that carry."""
    tmp = tmp_path_factory.mktemp("label_rolling_lookahead")
    steps, _ = _run(_variant(HEDGE, tmp, "rolling_hedge", _rolling_hedge), tmp, scenario="hedge_storage")
    rolls = list(steps.values())
    assert len(rolls) == 2 and not any(is_realization(k) for k in steps), list(steps)
    first, second = rolls
    v0 = utf8(first.solution.value("v_state")).filter(
        (pl.col("n") == "resv") & (pl.col("d") == "p2035") & (pl.col("t") == "t0003"))
    assert v0.height == 1
    su = utf8(first.flex_data.p_state_unitsize.frame).filter(pl.col("n") == "resv")["value"][0]
    carry = utf8(second.flex_data.p_roll_continue_state.frame).filter(pl.col("n") == "resv")["value"].to_list()
    assert carry == [pytest.approx(float(v0["value"][0]) * su, abs=1e-6)]
    starts = utf8(second.flex_data.nodeState_first_dt).filter(pl.col("n") == "resv")
    assert set(starts["d"].to_list()) == {"p2040_high", "p2040_low"}


def _rolling_hedge_online(data: dict) -> None:
    """:func:`_rolling_hedge` with unit commitment on ``base`` (startup_method linear, min_load 0.3)."""
    _rolling_hedge(data)
    data["parameter_values"] += [["unit", "base", "startup_method", _enc("linear"), "roll"],
                                 ["unit", "base", "min_load", [base64.b64encode(b"0.3").decode(), "float"], "roll"]]


@pytest.mark.solver
def test_rolling_fan_in_the_look_ahead_leaves_units_online_untied(tmp_path_factory) -> None:
    """Units online are tied only where committed.  Roll 0 commits t0001-t0003 of the pre-reveal period p2035 and
    has the fan (p2040 = t0004) only in its look-ahead: no step of the fan is committed, so its forecasts commit
    units freely there (no non_anticipativity_online row).  Roll 1 commits t0004-t0006 inside the fan: units online
    are tied on those 3 steps, reference forecast ``p2040_high`` to the one other forecast ``p2040_low``, for the
    one online unit ``base``: 3 rows."""
    from tests.engine._unit_db import build, cstr_over
    tmp = tmp_path_factory.mktemp("label_rolling_lookahead_online")
    steps, _ = _run(_variant(HEDGE, tmp, "rolling_hedge_online", _rolling_hedge_online), tmp,
                    scenario="hedge_storage")
    rolls = list(steps.values())
    assert len(rolls) == 2 and not any(is_realization(k) for k in steps), list(steps)
    first, second = (cstr_over(build(r.flex_data), "non_anticipativity_online_linear") for r in rolls)
    assert first is None
    assert second is not None
    assert sorted(second.select("p", "d", "b", "t").iter_rows()) == [
        ("base", "p2040_high", "p2040_low", t) for t in ("t0004", "t0005", "t0006")]


def _stochastics_variant(tmp: Path, name: str, mutate) -> Path:
    return _variant(STOCHASTICS, tmp, name, mutate)


@pytest.mark.solver
def test_committed_window_in_forecasts_only_fan_with_storage_raises(tmp_path_factory) -> None:
    """``1_week_rolling_wind`` with every branch a forecast: each roll commits its jump inside the fan, the next
    roll continues from it and the storage state is decided per forecast there, so the run stops before the first
    roll's LP and names the storage state."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    tmp = tmp_path_factory.mktemp("label_committed_storage")
    fixture = _stochastics_variant(tmp, "forecasts_only_rolling", lambda d: _all_forecasts(d, "1week_rolling"))
    with pytest.raises(FlexToolConfigError, match=r"1week_rolling_roll_0.*storage state"):
        _run(fixture, tmp, scenario="1_week_rolling_wind")


def _append_alternative(data: dict, scenario: str, alt: str, entities: list, values: list) -> None:
    """Append alternative *alt* (with *entities* and *values* ``[cls, name, param, value]``) to the end of
    *scenario*'s chain."""
    data["alternatives"].append([alt, ""])
    for sa in data["scenario_alternatives"]:
        if sa[0] == scenario and sa[2] is None:
            sa[2] = alt
    data["scenario_alternatives"].append([scenario, alt, None])
    for ent in entities:
        data["entities"].append(ent)
        if ent[0] in ("node", "unit", "connection"):
            data["entity_alternatives"].append([ent[0], [ent[1]], alt, True])
    data["parameter_values"] += [[*v, alt] for v in values]


def _period_map(values: dict) -> list:
    value = {"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}
    return [base64.b64encode(json.dumps(value).encode()).decode(), "map"]


def _annual_tiers(quantity: float) -> list:
    """``price_ladder_annual``: tier 1 at 5 up to *quantity* per year, tier 2 at 30 unlimited."""
    def tier(price, q):
        return {"type": "map", "index_type": "str", "rank": 1, "data": [["price", price], ["quantity", q]]}
    return [base64.b64encode(json.dumps(
        {"index_type": "str", "rank": 2, "index_name": "tier",
         "data": [["1", tier(5.0, quantity)], ["2", tier(30.0, float("inf"))]]}).encode()).decode(), "map"]


# ``1_week_rolling_wind`` (stochastics.json) with every branch a forecast and ``hydro_reservoir`` made a commodity
# node (the ``no_storage_override`` alternative), so no storage state is carried; each roll commits its 24 h jump
# inside the fan.
_CO2_PERIOD_CAP = ([["group", "co2_limit", None], ["group__node", ["co2_limit", "gas_market"], None]],
                   [["commodity", "gas", "co2_content", [base64.b64encode(b"0.2").decode(), "float"]],
                    ["group", "co2_limit", "co2_method", _enc("period")],
                    ["group", "co2_limit", "co2_max_period", _period_map({"period1": 2.0e6})]])
_ANNUAL_LADDER = ([], [["commodity", "coal", "price_method", _enc("price_ladder_annual")],
                       ["commodity", "coal", "price_ladder_annual", _annual_tiers(1.0e6)]])


def _forecasts_only_without_storage(extra: tuple):
    def mutate(data: dict) -> None:
        _all_forecasts(data, "1week_rolling")
        _append_alternative(data, "1_week_rolling_wind", "no_storage_override", [], [])
        _append_alternative(data, "1_week_rolling_wind", "committed_carry", *extra)
    return mutate


@pytest.mark.solver
@pytest.mark.parametrize(("family", "extra"), [("CO2 period cap", _CO2_PERIOD_CAP),
                                               ("annual commodity ladder", _ANNUAL_LADDER)])
def test_committed_window_in_forecasts_only_fan_with_period_carry_raises(tmp_path_factory, family: str,
                                                                         extra: tuple) -> None:
    """A CO2 period cap and a finite annual ladder tier continue from the period's realized emissions /
    purchases.  With every branch a forecast and no storage, each roll still commits its jump inside the fan and
    the next roll continues from it, but those emissions / purchases are decided per forecast there and no solve
    realizes them: the run stops before the first roll's LP and names the family."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    tmp = tmp_path_factory.mktemp("label_committed_period_carry")
    fixture = _stochastics_variant(tmp, "forecasts_only_no_storage", _forecasts_only_without_storage(extra))
    with pytest.raises(FlexToolConfigError, match=rf"1week_rolling_roll_0.*{family}") as raised:
        _run(fixture, tmp, scenario="1_week_rolling_wind")
    assert "storage state" not in str(raised.value)


def _period_carry_flex_data(cap_period: str) -> SimpleNamespace:
    """A FlexData stand-in with one fan (label ``p1``, forecasts ``p1_a`` / ``p1_b``), a finite annual ladder
    tier and a CO2 period cap, both on *cap_period*; every other family absent."""
    from polar_high import Param

    return SimpleNamespace(
        nodeState=None, dt_non_anticipativity=None, process_online=None, process_reserve_upDown_node_active=None,
        ci_ladder_cumulative=None, p_ladder_cum_quantity=None, group_co2_max_total=None, recourse_invest=False,
        period_branch_full=pl.DataFrame({"d": ["p1", "p1", "p2"], "b": ["p1_a", "p1_b", "p2"]}),
        cndi_ladder_ann=pl.DataFrame({"c": ["coal"], "n": ["coal_node"], "d": [cap_period], "i": ["1"]}),
        p_ladder_ann_quantity=Param(("c", "i", "d"), pl.DataFrame(
            {"c": ["coal"], "i": ["1"], "d": [cap_period], "value": [1.0e6]})),
        group_d_co2_capped=pl.DataFrame({"g": ["co2g"], "d": [cap_period]}))


@pytest.mark.parametrize(("cap_period", "families"), [
    ("p1", ["annual commodity ladder", "CO2 period cap"]),
    ("p1_a", ["annual commodity ladder", "CO2 period cap"]),
    ("p2", []),
])
def test_committed_window_period_carries_only_on_the_fan_period(cap_period: str, families: list) -> None:
    """An annual ladder tier or a CO2 period cap continues from its own period's realized carry, so a committed
    window inside the fan of p1 leaves it undecided only when the cap is on p1 (its label or a forecast of it); a
    cap on another period p2 of the chain continues from a carry the window does not touch."""
    from flextool.engine.solve._realization import _untied_families

    assert _untied_families(_period_carry_flex_data(cap_period), [("p1", "t1")]) == families


def _rolling_hedge_with_period_carry(data: dict) -> None:
    """:func:`_rolling_hedge` with 100 MW of ``base`` existing (a rolling solve has no investment) fed from a
    commodity node ``fuel`` (commodity ``fuel`` on a finite annual ladder, co2_content 0.5) inside a CO2
    period-capped group ``co2g``."""
    _rolling_hedge(data)
    _append_alternative(
        data, "hedge_storage", "period_carry",
        [["node", "fuel", None], ["commodity", "fuel", None], ["commodity__node", ["fuel", "fuel"], None],
         ["unit__inputNode", ["base", "fuel"], None], ["group", "co2g", None], ["group__node", ["co2g", "fuel"], None]],
        [["unit", "base", "existing", [base64.b64encode(b"100.0").decode(), "float"]],
         ["node", "fuel", "node_type", _enc("commodity")],
         ["commodity", "fuel", "co2_content", [base64.b64encode(b"0.5").decode(), "float"]],
         ["commodity", "fuel", "price_method", _enc("price_ladder_annual")],
         ["commodity", "fuel", "price_ladder_annual", _annual_tiers(1.0e6)],
         ["group", "co2g", "co2_method", _enc("period")],
         ["group", "co2g", "co2_max_period", _period_map({"p2035": 1.0e6, "p2040": 1.0e6})]])


@pytest.mark.solver
def test_rolling_fan_in_the_look_ahead_with_period_carry_solves(tmp_path_factory) -> None:
    """A fan only in the look-ahead commits nothing inside it: roll 0 commits t0001-t0003 of the pre-reveal
    period p2035, so an annual ladder tier and a CO2 period cap need no realization solve.  The run solves, and
    the carries roll 1 continues from (simulated hours, CO2 tonnes, ladder purchases) are keyed by the committed
    period p2035 only, the simulated hours being its 3 committed one-hour steps."""
    tmp = tmp_path_factory.mktemp("label_rolling_lookahead_period_carry")
    steps, _ = _run(_variant(HEDGE, tmp, "rolling_hedge_period_carry", _rolling_hedge_with_period_carry), tmp,
                    scenario="hedge_storage")
    rolls = list(steps.values())
    assert len(rolls) == 2 and not any(is_realization(k) for k in steps), list(steps)
    first, second = rolls
    assert second.flex_data.group_d_co2_capped is not None and second.flex_data.cndi_ladder_ann is not None
    handoff = first.handoff
    hours = handoff.cum_sim_hours
    assert dict(zip(hours["period"].cast(pl.Utf8), hours["p_ladder_cum_sim_hours"])) == {
        "p2035": pytest.approx(3.0)}
    for name in ("cumulative_co2", "cumulative_commodity"):
        carrier = getattr(handoff, name)
        assert carrier is not None and carrier.height > 0, name
        period_col = "period" if "period" in carrier.columns else "d"
        assert set(utf8(carrier)[period_col].to_list()) == {"p2035"}, carrier
    co2 = second.flex_data.p_co2_cum_realized_tonnes.frame
    assert set(utf8(co2).select("g", "d").iter_rows()) == {("co2g", "p2035")}, co2
    # the carried tonnes are the carried fuel purchases × co2_content 0.5, and both are non-zero
    mwh = float(handoff.cumulative_commodity["p_ladder_cum_realized_mwh"].sum())
    assert mwh > 0.0
    assert float(co2["value"][0]) == pytest.approx(0.5 * mwh, rel=1e-9)


def _rolling_online_only(data: dict) -> None:
    """``stoch_t0_commit`` as a rolling solve over a forecasts-only fan with unit commitment only: no designation,
    no ``separate_dispatch``, no investment; horizon 2 h, jump 1 h, so every roll commits its first step inside the
    fan.  Each roll start t0001-t0003 has an analysis time: ``high`` (wind CF 0.8) and ``low`` (0.0), 0.5 each."""
    drop = {"dispatch_realization_method", "invest_periods", "realized_invest_periods", "invest_method",
            "invest_cost", "discount_rate", "lifetime", "invest_max_total"}
    data["parameter_values"] = [r for r in data["parameter_values"] if r[2] not in drop]
    steps = ("t0001", "t0002", "t0003")

    def enc(value, kind):
        return [base64.b64encode(json.dumps(value).encode()).decode(), kind]

    def leaf(m):
        return {"type": "map", "index_type": "str", "data": m}
    cf = {"high": 0.8, "low": 0.0}
    for r in data["parameter_values"]:
        if r[2] == "solve_mode":
            r[3] = enc("rolling_window", "str")
        elif r[2] == "stochastic_branches":
            r[3] = enc({"index_type": "str", "data": [["p2035", leaf(
                [[b, leaf([[ts, leaf([["no", 0.5]])] for ts in steps])] for b in cf])]]}, "map")
        elif r[1] == "wprof" and r[2] == "profile":
            r[3] = enc({"index_type": "str", "data": [
                [b, leaf([[ts, leaf([[t, v] for t in steps if t >= ts])] for ts in steps])] for b, v in cf.items()]},
                "map")
    data["parameter_values"] += [["solve", "stoch_rd", "rolling_solve_horizon", enc(2.0, "float"), "init"],
                                 ["solve", "stoch_rd", "rolling_solve_jump", enc(1.0, "float"), "init"]]


@pytest.mark.solver
def test_committed_window_in_forecasts_only_fan_online_only_solves(tmp_path_factory) -> None:
    """A committed window inside a forecasts-only fan with only non-anticipativity-tied families (online): the
    run needs no realization solve, every roll solves, and the next roll's online carry is the first sorted
    forecast's commitment at the committed step, applied to every forecast at the next roll's first step.  Hand value: demand 100 MW, wind 80 (high) or 0 (low) MW, the
    peaker 2 × 50 MW with min_load 0.5; ``low`` needs both units, the tie gives ``high`` the same commitment, and
    a slack costs 10 000 per MWh against the 10 per MWh peaker, so v_online = 2 at each roll's first step."""
    tmp = tmp_path_factory.mktemp("label_committed_online")
    steps, _ = _run(_variant(T0_FIXTURE, tmp, "rolling_online_only", _rolling_online_only), tmp)
    rolls = list(steps.values())
    assert len(rolls) == 3 and not any(is_realization(k) for k in steps), list(steps)
    for prev, nxt in zip(rolls, rolls[1:]):
        online = utf8(prev.solution.value("v_online_linear")).filter(pl.col("p") == "peaker")
        first = online.filter(pl.col("d") == FIRST_FORECAST).sort("t")
        assert float(first["value"][0]) == pytest.approx(2.0, abs=1e-6)
        assert set(online.filter(pl.col("t") == first["t"][0])["value"].round(6).to_list()) == {2.0}
        carry = utf8(nxt.flex_data.p_roll_continue_online.frame).filter(pl.col("p") == "peaker")
        assert carry["value"].to_list() == [pytest.approx(float(first["value"][0]), abs=1e-6)]
        # ... and it reaches every forecast of the next roll's fan at its first step.
        takes = utf8(nxt.flex_data.dt_online_carry)
        assert set(takes["d"].to_list()) == {"p2035_high", "p2035_low"} and takes["t"].n_unique() == 1, takes


def _rolling_online_only_min_uptime(data: dict) -> None:
    """:func:`_rolling_online_only` with the peaker's minimum up and down time 2 h (1 h steps)."""
    _rolling_online_only(data)
    for r in data["parameter_values"]:
        if r[1] == "peaker" and r[2] in ("min_uptime", "min_downtime"):
            r[3] = [base64.b64encode(json.dumps(2.0).encode()).decode(), "float"]


@pytest.mark.solver
def test_committed_window_in_forecasts_only_fan_carries_the_min_uptime(tmp_path_factory) -> None:
    """The forecasts-only rolling fan with a 2 h minimum up time: each roll commits its first step inside the fan
    (the tied forecast ``p2035_high`` holds its values), so the next roll's forecasts both start at the step after
    it and owe the start-ups of that committed step — on 1 h steps the committed step is 1 h before the next roll's
    first step (1 < 2), the step before it 2 h (not < 2).  So carry = v_startup of ``p2035_high`` at the committed
    step, at the first step of every forecast of the next roll, nothing later.  The history used to be stored under
    the period label, where no forecast's look-back reached it."""
    tmp = tmp_path_factory.mktemp("label_committed_min_uptime")
    steps, _ = _run(_variant(T0_FIXTURE, tmp, "rolling_online_only_min_uptime", _rolling_online_only_min_uptime),
                    tmp)
    rolls = list(steps.values())
    assert len(rolls) == 3 and all(r.solution.optimal for r in rolls)
    carried = 0
    for prev, nxt in zip(rolls, rolls[1:]):
        committed = prev.handoff.roll_end_step["step"][0]
        su = utf8(prev.solution.value("v_startup_linear")).filter(
            (pl.col("p") == "peaker") & (pl.col("d") == FIRST_FORECAST) & (pl.col("t") == committed))
        expect = float(su["value"][0])
        first = utf8(nxt.flex_data.dt)["t"].sort()[0]
        uc = nxt.flex_data.uptime_carry
        got = {} if uc is None else {(d, t): v for _p, d, t, v in utf8(uc).rows()}
        want = {(d, first): expect for d in ("p2035_high", "p2035_low")} if abs(expect) > 1e-9 else {}
        assert got.keys() == want.keys(), (got, want)
        for key, v in want.items():
            assert got[key] == pytest.approx(v, abs=1e-9)
        carried += len(want)
    assert carried > 0, "the witness starts no unit at a committed step"


def _flowgroup_path(invest_method: str):
    import importlib.util
    name = "_lp_snapshot_for_realization"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FLEXTOOL_ROOT / "tools" / "lp_snapshot.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    lps = sys.modules[name]
    base = ["init", "base", "stoch", "ws", "na_window", "divest"] + (
        ["invest_none"] if invest_method == "none" else [])
    data = lps._overlay(json.loads(HEDGE.read_text(encoding="utf-8")), name="flowgroup_path", base=base,
                        entities=[["flowGroup", "base_flow"],
                                  ["flowGroup__unit__node", ["base_flow", "base", "elec"]]],
                        values=[["flowGroup", "base_flow", "min_cumulative_flow", 1.0]])
    return data



def _rolling_online_only_free(data: dict) -> None:
    """:func:`_rolling_online_only` with ``shared_operation_periods = []``: nothing is tied, units online included."""
    _rolling_online_only(data)
    data["parameter_values"].append(["solve", "stoch_rd", "shared_operation_periods",
                                     [base64.b64encode(json.dumps({"value_type": "str", "data": []}).encode()).decode(),
                                      "array"], "init"])


@pytest.mark.solver
def test_committed_window_in_forecasts_only_fan_with_untied_online_raises(tmp_path_factory) -> None:
    """The twin of the online-only case with ``shared_operation_periods = []``: units online are no longer tied on
    the committed step, so they are decided per forecast there, no solve realizes them and the next roll would
    continue from them: the run stops before the first roll's LP and names the units online."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    tmp = tmp_path_factory.mktemp("label_committed_online_free")
    with pytest.raises(FlexToolConfigError, match=r"stoch_rd_roll_0.*units online"):
        _run(_variant(T0_FIXTURE, tmp, "rolling_online_only_free", _rolling_online_only_free), tmp)


@pytest.mark.solver
@pytest.mark.parametrize("invest_method", ["recourse", "none"])
def test_branch_paths_hold_the_period_label(tmp_path_factory, invest_method: str) -> None:
    """A flowGroup minimum cumulative flow on the fan-at-p2035 hedge: the solve is feasible and no row is keyed by
    an ``__anchor`` path (the period labels have no dispatch).  Under ``none`` the label p2035 carries the
    here-and-now investment: every investment-total row holds that investment."""
    tmp = tmp_path_factory.mktemp(f"label_paths_{invest_method}")
    path = tmp / "flowgroup_path.json"
    path.write_text(json.dumps(_flowgroup_path(invest_method)), encoding="utf-8")
    steps, mps = _run(path, tmp, scenario="flowgroup_path", capture_mps=True)
    (solve,) = list(steps)
    assert steps[solve].solution.optimal
    rows = _mps_rows(mps[solve])
    assert not [r for r in rows if "__anchor" in r]
    assert [r for r in rows if r.startswith("minCumulative_flow")]
    totals = [r for r in rows if r.startswith("maxInvest_entity_total")]
    if invest_method == "none":
        assert totals
        holders = {r for c, r in _mps_entries(mps[solve]) if c == "v_invest_p[base,p2035]"}
        assert set(totals) <= holders, (totals, holders)
