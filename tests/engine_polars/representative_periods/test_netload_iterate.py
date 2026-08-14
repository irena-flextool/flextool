"""Tests for the net-load solve-iteration driver (Phase 4).

The loop-logic tests drive :func:`run_netload_iteration` with every I/O
boundary MONKEYPATCHED — the selection, the solve subprocess, the fail-closed
assessor, and the cap / cost readers — so no real solve is launched (mirroring
``tests/calibrate/test_loop.py``).  ``_select`` is stubbed as a DETERMINISTIC
function of the fed-back caps (exactly as the real net-load selection is a
deterministic function of its inputs), so the keep-best re-materialisation and
the convergence check exercise the real control flow.

The orphan-timeset test and the fail-closed guard test build a real temp DB
from the live schema (invariant #3 — never a checked-in ``.sqlite``) and stub
only the solve, so the real selection + DB-write path runs.

The reader unit tests write hand-built parquet with the repo's lean-parquet
writer and read it back.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import spinedb_api as api
from spinedb_api import Array, DatabaseMapping, Map, import_data

import flextool.representative_periods.netload_iterate as ni
from flextool.calibrate._readers import (
    read_total_system_cost,
    read_unit_capacity_total,
)
from flextool.lean_parquet import write_lean_parquet
from flextool.representative_periods.netload_iterate import (
    NetloadIterConfig,
    NetloadIterError,
    run_netload_iteration,
)
from flextool.update_flextool.initialize_database import initialize_database

_SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "flextool"
    / "schemas"
    / "spinedb_schema.json"
)


# --------------------------------------------------------------------------
# Loop-logic harness (fully monkeypatched)
# --------------------------------------------------------------------------
def _caps_key(caps):
    """Hashable identity of a solved-caps dict (or None)."""
    return None if caps is None else tuple(sorted(caps.items()))


class _SelectSpy:
    """Deterministic ``_select`` stub: rep-set is a function of the fed caps.

    Records every ``solved_caps`` it is called with (so tests assert the
    fed-back caps) and returns ``mapping[key(solved_caps)]`` — deterministic,
    so the keep-best re-materialisation reproduces the winning set exactly.
    """

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls: list = []

    def __call__(self, url, config, solved_caps):
        self.calls.append(None if solved_caps is None else dict(solved_caps))
        return self.mapping[_caps_key(solved_caps)]


class _SeqSpy:
    """Return successive scripted values from a list; records the call count."""

    def __init__(self, seq):
        self.seq = list(seq)
        self.i = -1

    def __call__(self, *args, **kwargs):
        self.i += 1
        return self.seq[self.i]

    @property
    def n_calls(self):
        return self.i + 1


def _install(
    monkeypatch,
    *,
    select_map,
    invest_caps,
    dispatch_costs,
    full_chain=("invest", "dispatch"),
):
    """Patch the driver's I/O boundaries and return the installed spies."""
    select = _SelectSpy(select_map)
    caps_reader = _SeqSpy(invest_caps)
    cost_reader = _SeqSpy(dispatch_costs)
    solves_override = _SeqSpy([None] * 64)  # no-op, count writes

    def fake_run_solve(url, scenario, *, work_dir, out_root, cache_dir):
        return SimpleNamespace(
            assess_dir=Path(out_root) / "assess",
            returncode=0,
            started_at=0.0,
        )

    def fake_assess_solve(assess_dir, *, exit_code, required_outputs, started_at):
        return SimpleNamespace(succeeded=True, reason="")

    monkeypatch.setattr(ni, "_guard_no_timed_energy_margin", lambda url, scen: None)
    monkeypatch.setattr(ni, "_full_chain_solves", lambda url, scen: list(full_chain))
    monkeypatch.setattr(ni, "_select", select)
    monkeypatch.setattr(ni, "_write_solves_override_alt", solves_override)
    monkeypatch.setattr(ni, "run_solve", fake_run_solve)
    monkeypatch.setattr(ni, "assess_solve", fake_assess_solve)
    monkeypatch.setattr(ni, "read_unit_capacity_total", caps_reader)
    monkeypatch.setattr(ni, "read_total_system_cost", cost_reader)
    return SimpleNamespace(
        select=select, caps=caps_reader, cost=cost_reader,
    )


def _config(tmp_path, *, iterations, keep_best=False):
    return NetloadIterConfig(
        n_rp=2,
        period_length=2,
        iterations=iterations,
        scenario="scenA",
        invest_solves=["invest"],
        dispatch_solves=None,
        work_dir=tmp_path / "work",
        out_root=tmp_path / "out",
        alternative_name="netload_2rp_2h",
        keep_best=keep_best,
    )


def test_bootstrap_only_no_bootstrap_dispatch(tmp_path, monkeypatch):
    """n=0 → one select, one invest-only solve, NO bootstrap dispatch, one
    final dispatch for output."""
    spies = _install(
        monkeypatch,
        select_map={None: ("A",)},
        invest_caps=[{"wind": 10.0}],
        dispatch_costs=[42.0],  # only the final dispatch reads a cost
    )
    result = run_netload_iteration("db.sqlite", _config(tmp_path, iterations=0))

    # Exactly one selection (the bootstrap), fed the demand-match default caps.
    assert spies.select.calls == [None]
    # One invest-only solve.
    assert result.iterations_run == 1
    # NO bootstrap dispatch: the single cost read is the FINAL dispatch.
    assert spies.cost.n_calls == 1
    assert result.stop_reason == "bootstrap_only"
    assert result.final_rep_starts == ("A",)
    assert result.final_cost == 42.0


def test_reselects_with_fed_back_caps(tmp_path, monkeypatch):
    """n=2 → re-selects between invest-only solves using fed-back caps; the
    bootstrap dispatch is skipped (keep_best off ⇒ only the final dispatch)."""
    spies = _install(
        monkeypatch,
        select_map={
            None: ("A",),
            _caps_key({"wind": 10.0}): ("B",),
            _caps_key({"wind": 11.0}): ("C",),
        },
        invest_caps=[{"wind": 10.0}, {"wind": 11.0}, {"wind": 12.0}],
        dispatch_costs=[7.0],  # final dispatch only
    )
    result = run_netload_iteration("db.sqlite", _config(tmp_path, iterations=2))

    # Selection fed None (bootstrap), then caps_0, then caps_1 — the fed-back
    # invested fleet of the PRIOR iteration each time.
    assert spies.select.calls == [None, {"wind": 10.0}, {"wind": 11.0}]
    assert result.iterations_run == 3
    # Only the final dispatch ran (no per-iteration dispatch without keep_best).
    assert spies.cost.n_calls == 1
    assert result.final_rep_starts == ("C",)
    assert result.stop_reason == "budget_exhausted"


def test_early_stop_on_unchanged_set(tmp_path, monkeypatch):
    """A re-selection reproducing the previous set stops the loop before n."""
    spies = _install(
        monkeypatch,
        select_map={
            None: ("A",),
            _caps_key({"wind": 10.0}): ("A",),  # unchanged → converge
        },
        invest_caps=[{"wind": 10.0}],  # only the bootstrap invest-solves
        dispatch_costs=[5.0],  # final dispatch only
    )
    result = run_netload_iteration("db.sqlite", _config(tmp_path, iterations=5))

    # Two selects (bootstrap + the matching re-selection), but only ONE invest
    # solve — the loop broke before solving the matched iteration.
    assert spies.select.calls == [None, {"wind": 10.0}]
    assert result.iterations_run == 1
    assert result.converged is True
    assert result.stop_reason == "converged"
    assert result.final_rep_starts == ("A",)


def test_keep_best_picks_cheaper_earlier_iter(tmp_path, monkeypatch):
    """keep_best dispatches mature iters and keeps the lowest-cost set, then
    re-materialises that earlier winner for the final dispatch."""
    spies = _install(
        monkeypatch,
        select_map={
            None: ("A",),
            _caps_key({"wind": 10.0}): ("B",),  # k=1 winner (cheaper)
            _caps_key({"wind": 11.0}): ("C",),  # k=2 (pricier)
        },
        invest_caps=[{"wind": 10.0}, {"wind": 11.0}, {"wind": 12.0}],
        # k=1 dispatch, k=2 dispatch, final dispatch (re-materialise does not
        # dispatch): the earlier iter (k=1) is cheaper.
        dispatch_costs=[100.0, 200.0, 150.0],
    )
    result = run_netload_iteration(
        "db.sqlite", _config(tmp_path, iterations=2, keep_best=True)
    )

    # Mature iters dispatched (k=1, k=2) + final = 3 cost reads; the BOOTSTRAP
    # (k=0) was NOT dispatched (would have made it 4).
    assert spies.cost.n_calls == 3
    assert result.best_cost == 100.0
    # The winner is the earlier, cheaper iteration's set.
    assert result.final_rep_starts == ("B",)
    # Re-materialised by re-selecting with the winner's INPUT caps (caps_0),
    # so the last select call fed those caps and reproduced ("B",).
    assert spies.select.calls[-1] == {"wind": 10.0}


# --------------------------------------------------------------------------
# Real-DB fixture (schema-built) for the orphan + guard tests
# --------------------------------------------------------------------------
_KEYS = [f"t{i:02d}" for i in range(1, 9)]


def _build_iter_db(
    db_path: Path,
    *,
    scenario_name: str = "base",
    energy_margin_adder=None,
    energy_margin_method: str | None = None,
) -> str:
    """A one-node invest+dispatch model with a solve, model.solves, timeline.

    Enough structure for the driver's real DB path: a co-located investable
    wind unit (net-load signal), a ``solve`` carrying ``period_timeset`` +
    ``invest_periods``, and a ``model`` whose ``solves`` names that solve.
    Optionally sets an ``energy_margin_adder`` / method on the node (for the
    fail-closed guard test).
    """
    initialize_database(str(_SCHEMA), str(db_path))
    url = f"sqlite:///{db_path}"

    demand = Map(_KEYS, [-10.0, -40.0, -80.0, -20.0, -15.0, -90.0, -30.0, -25.0])
    wind_pf = Map(_KEYS, [0.9, 0.2, 0.1, 0.8, 0.7, 0.05, 0.6, 0.5])
    timeline = Map(_KEYS, [1.0] * len(_KEYS))
    period_timeset = Map(["p2025"], ["baseline_timeset"])
    invest_periods = Array(["p2025"])

    param_values = [
        ["node", "n1", "inflow", demand, scenario_name],
        ["profile", "wind_profile", "profile", wind_pf, scenario_name],
        ["timeline", "main", "timestep_duration", timeline, scenario_name],
        ["unit", "wind", "existing", 1.0, scenario_name],
        ["unit", "wind", "invest_method", "invest_no_limit", scenario_name],
        ["solve", "invest_solve", "period_timeset", period_timeset, scenario_name],
        ["solve", "invest_solve", "invest_periods", invest_periods, scenario_name],
        ["model", "m1", "solves", Array(["invest_solve"]), scenario_name],
    ]
    if energy_margin_method is not None:
        param_values.append(
            ["node", "n1", "energy_margin_method", energy_margin_method,
             scenario_name]
        )
    if energy_margin_adder is not None:
        param_values.append(
            ["node", "n1", "energy_margin_adder", energy_margin_adder,
             scenario_name]
        )

    with DatabaseMapping(url) as db:
        _count, errors = import_data(
            db,
            alternatives=[[scenario_name, ""]],
            scenarios=[[scenario_name, True, ""]],
            scenario_alternatives=[[scenario_name, scenario_name]],
            entities=[
                ["node", ["n1"], None],
                ["profile", ["wind_profile"], None],
                ["unit", ["wind"], None],
                ["timeline", ["main"], None],
                ["solve", ["invest_solve"], None],
                ["model", ["m1"], None],
                ["unit__outputNode", ["wind", "n1"], None],
                ["unit__node__profile", ["wind", "n1", "wind_profile"], None],
            ],
            entity_alternatives=[
                ["node", ["n1"], scenario_name, True],
                ["profile", ["wind_profile"], scenario_name, True],
                ["unit", ["wind"], scenario_name, True],
                ["timeline", ["main"], scenario_name, True],
                ["solve", ["invest_solve"], scenario_name, True],
                ["model", ["m1"], scenario_name, True],
                ["unit__outputNode", ["wind", "n1"], scenario_name, True],
                ["unit__node__profile", ["wind", "n1", "wind_profile"],
                 scenario_name, True],
            ],
            parameter_values=param_values,
        )
        assert not errors, errors
        db.commit_session("netload iterate fixture")
    return url


def _count_timeset_alternatives(url: str) -> tuple[int, int]:
    """Return (n_timeset_entities, n_alts_carrying_timeset_duration)."""
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        timesets = {e["name"] for e in db.get_entity_items(entity_class_name="timeset")}
        alts = {
            pv["alternative_name"]
            for pv in db.find_parameter_values(
                entity_class_name="timeset",
                parameter_definition_name="timeset_duration",
            )
        }
    return len(timesets), len(alts)


def test_stable_alt_name_leaves_one_timeset(tmp_path, monkeypatch):
    """N iterations with the pinned stable alt leave exactly ONE net-load
    timeset alternative (the purge+rewrite replaces, never accumulates)."""
    url = _build_iter_db(tmp_path / "orphan.sqlite")

    # Real selection + DB write; only the solve and cap read are stubbed.
    def fake_run_solve(u, scenario, *, work_dir, out_root, cache_dir):
        return SimpleNamespace(
            assess_dir=Path(out_root) / "assess", returncode=0, started_at=0.0
        )

    def fake_assess_solve(assess_dir, *, exit_code, required_outputs, started_at):
        return SimpleNamespace(succeeded=True, reason="")

    monkeypatch.setattr(ni, "run_solve", fake_run_solve)
    monkeypatch.setattr(ni, "assess_solve", fake_assess_solve)
    monkeypatch.setattr(ni, "read_unit_capacity_total", lambda a: {"wind": 5.0})
    monkeypatch.setattr(ni, "read_total_system_cost", lambda a: 1.0)

    config = NetloadIterConfig(
        n_rp=2,
        period_length=2,
        iterations=2,
        scenario="base",
        invest_solves=["invest_solve"],
        dispatch_solves=None,
        work_dir=tmp_path / "work",
        out_root=tmp_path / "out",
        alternative_name="netload_2rp_2h",
    )
    run_netload_iteration(url, config)

    n_timesets, n_alts = _count_timeset_alternatives(url)
    assert n_timesets == 1, f"expected 1 timeset entity, got {n_timesets}"
    assert n_alts == 1, f"expected 1 timeset-bearing alternative, got {n_alts}"

    # And the single timeset carries exactly n_rp representative periods.
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        pvs = [
            api.from_database(pv["value"], pv["type"])
            for pv in db.find_parameter_values(
                entity_class_name="timeset",
                parameter_definition_name="timeset_duration",
            )
            if pv["alternative_name"] == "netload_2rp_2h"
        ]
    assert len(pvs) == 1
    assert len(list(pvs[0].indexes)) == 2


def test_fail_closed_on_timed_energy_margin(tmp_path):
    """An active per-cell (period→time) energy_margin_adder raises clearly."""
    adder = Map(
        ["p2025"],
        [Map(["t01", "t02"], [3.0, 4.0], index_name="time")],
        index_name="period",
    )
    url = _build_iter_db(
        tmp_path / "timed.sqlite",
        energy_margin_adder=adder,
        energy_margin_method="inflow_adder",
    )
    config = NetloadIterConfig(
        n_rp=2, period_length=2, iterations=1, scenario="base",
        invest_solves=["invest_solve"], dispatch_solves=None,
        work_dir=tmp_path / "work", out_root=tmp_path / "out",
        alternative_name="netload_2rp_2h",
    )
    with pytest.raises(NetloadIterError, match="timed per-cell energy_margin_adder"):
        run_netload_iteration(url, config)


def test_scalar_energy_margin_is_allowed(tmp_path):
    """A scalar (uniform) energy_margin_adder is grid-independent → no raise."""
    url = _build_iter_db(
        tmp_path / "scalar.sqlite",
        energy_margin_adder=12.5,
        energy_margin_method="inflow_adder",
    )
    # The guard alone must not raise on a scalar adder.
    ni._guard_no_timed_energy_margin(url, "base")


# --------------------------------------------------------------------------
# Reader unit tests (hand-built parquet)
# --------------------------------------------------------------------------
def test_read_unit_capacity_total_takes_mature_total(tmp_path):
    """read_unit_capacity_total returns the max ``total`` over periods per unit."""
    idx = pd.MultiIndex.from_tuples(
        [("wind", "p2025"), ("wind", "p2030"),
         ("solar", "p2025"), ("solar", "p2030")],
        names=["unit", "period"],
    )
    cols = pd.MultiIndex.from_tuples(
        [("scenA", "existing"), ("scenA", "invested"),
         ("scenA", "divested"), ("scenA", "total")],
        names=["scenario", "parameter"],
    )
    data = [
        [10.0, 90.0, 0.0, 100.0],   # wind p2025 total 100
        [10.0, 140.0, 0.0, 150.0],  # wind p2030 total 150 (mature)
        [5.0, 45.0, 0.0, 50.0],     # solar p2025 total 50
        [5.0, 45.0, 0.0, 50.0],     # solar p2030 total 50
    ]
    df = pd.DataFrame(data, index=idx, columns=cols)
    write_lean_parquet(df, tmp_path / "unit_capacity_ed_p.parquet")

    caps = read_unit_capacity_total(tmp_path)
    assert caps == {"solar": 50.0, "wind": 150.0}


def test_read_total_system_cost_sums_all_categories(tmp_path):
    """read_total_system_cost sums every signed category (penalty + revenue)."""
    series = pd.Series(
        {
            "unit investment & retirement": 10.0,
            "upward slack penalty": 5.0,     # unserved-energy penalty INCLUDED
            "commodity_sales": -3.0,         # revenue enters negative
        }
    )
    series.index.name = "category"
    df = pd.concat({"scenA": series}, axis=1, names=["scenario"])
    write_lean_parquet(df, tmp_path / "costs_discounted_p_.parquet")

    assert read_total_system_cost(tmp_path) == pytest.approx(12.0)
