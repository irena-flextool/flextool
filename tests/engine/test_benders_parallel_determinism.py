"""Benders (Option C) — parallel region recourse DETERMINISM gate.

The per-iteration region recourse pass solves the N independent region
subproblems across a thread pool (``solve_benders(..., workers=...)``, backed by
``polar_high.solve_indexed_parallel``).  The Benders trajectory depends on the
region ``col_dual`` cut slopes, so the parallel solves MUST produce IDENTICAL
``(cost_r, slopes)`` and therefore identical LB / UB / iteration-count / recovered
invest+trade to the sequential (``workers=1``) path.

This module is the HARD determinism gate: run ``solve_benders`` on the prototype
``lh2_three_region_trade_invest`` fixture with ``workers=1`` and with
``workers=3`` and assert the outcomes are EXACTLY (bit-for-bit) equal — not just
close.  A divergence means the parallelism is non-deterministic and must not
ship.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from polar_high import resolve_worker_count

from flextool.engine.solve._benders import _resolve_benders_workers, solve_benders
from flextool.run_options import RunOptions
from tests._support.load import load_scenario, save_provider

_REGIONS = ["region_A", "region_B", "region_C"]


@pytest.fixture(scope="module")
def ti_data(scenario_run):
    run = scenario_run(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )
    return load_scenario(run)


def _run(ti_data, workers):
    return solve_benders(
        ti_data,
        _REGIONS,
        max_iters=20,
        tol=1e-4,
        monolith_objective=None,
        workers=workers,
    )


def test_parallel_equals_sequential_exact(ti_data) -> None:
    seq = _run(ti_data, workers=1)
    par = _run(ti_data, workers=3)

    # Scalar trajectory: EXACT equality (bit-for-bit), not approximate.
    assert par.iterations == seq.iterations, (
        f"iteration count differs: parallel={par.iterations} "
        f"sequential={seq.iterations}"
    )
    assert par.converged == seq.converged
    assert par.total_objective == seq.total_objective, (
        f"objective differs: parallel={par.total_objective!r} "
        f"sequential={seq.total_objective!r}"
    )
    assert par.lower_bound == seq.lower_bound, (
        f"LB differs: parallel={par.lower_bound!r} sequential={seq.lower_bound!r}"
    )
    assert par.upper_bound == seq.upper_bound, (
        f"UB differs: parallel={par.upper_bound!r} sequential={seq.upper_bound!r}"
    )
    assert par.gap == seq.gap, (
        f"gap differs: parallel={par.gap!r} sequential={seq.gap!r}"
    )

    # Per-region recourse costs: EXACT.
    assert par.region_costs.keys() == seq.region_costs.keys()
    for r in seq.region_costs:
        assert par.region_costs[r] == seq.region_costs[r], (
            f"region cost[{r}] differs: parallel={par.region_costs[r]!r} "
            f"sequential={seq.region_costs[r]!r}"
        )

    # Recovered investment per connection: EXACT.
    assert par.invest.keys() == seq.invest.keys()
    for c in seq.invest:
        assert par.invest[c] == seq.invest[c], (
            f"invest[{c}] differs: parallel={par.invest[c]!r} "
            f"sequential={seq.invest[c]!r}"
        )

    # Recovered trade-flow frames: EXACT (same arcs, same per-cell values).
    assert par.trade_flow.keys() == seq.trade_flow.keys()
    for key in seq.trade_flow:
        assert par.trade_flow[key].equals(seq.trade_flow[key]), (
            f"trade_flow[{key}] differs between parallel and sequential"
        )

    # The whole-system invest handoff frames: EXACT.
    assert par.invest_solution_vars.keys() == seq.invest_solution_vars.keys()
    for name in seq.invest_solution_vars:
        assert par.invest_solution_vars[name].equals(
            seq.invest_solution_vars[name]
        ), f"invest_solution_vars[{name}] differs between parallel and sequential"


def test_malformed_workers_variable_warns_once(monkeypatch, caplog) -> None:
    """A malformed ``FLEXTOOL_BENDERS_WORKERS`` is reported once, by the run's option resolution; the Benders
    worker count then uses the resolved (automatic) value without reading the variable again."""
    monkeypatch.setenv("FLEXTOOL_BENDERS_WORKERS", "many")
    with caplog.at_level(logging.WARNING):
        opts = RunOptions.from_env()
        workers = _resolve_benders_workers(4, None, opts)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "FLEXTOOL_BENDERS_WORKERS" in warnings[0].getMessage()
    assert workers == resolve_worker_count(4, None)


def test_explicit_workers_beat_the_run_options() -> None:
    opts = RunOptions.from_env(environ={"FLEXTOOL_BENDERS_WORKERS": "3"})
    assert _resolve_benders_workers(4, None, opts) == resolve_worker_count(4, 3)
    assert _resolve_benders_workers(4, 1, opts) == 1


def test_env_override_resolves(ti_data, monkeypatch) -> None:
    """The machine-local ``FLEXTOOL_BENDERS_WORKERS`` override drives the same
    deterministic result as the explicit ``workers`` arg / the sequential path."""
    monkeypatch.setenv("FLEXTOOL_BENDERS_WORKERS", "2")
    env = _run(ti_data, workers=None)  # env wins over the auto default
    monkeypatch.delenv("FLEXTOOL_BENDERS_WORKERS")
    seq = _run(ti_data, workers=1)
    assert env.total_objective == seq.total_objective
    assert env.lower_bound == seq.lower_bound
    assert env.iterations == seq.iterations


# ---------------------------------------------------------------------------
# Cross-PROCESS determinism (#153): the same model must give the same region
# LPs and the same Benders trajectory in every Python process, whatever its
# PYTHONHASHSEED.  ``split`` widens the axis Enum vocabularies with the
# virtual half-flow tokens, and the Enum category order is the sort key of
# every LP domain frame; appended in ``set`` order, the region subproblem's
# column / row order (and so the simplex path, the last bits of the cut
# coefficients and, on a marginal master, kUnknown vs kOptimal) changed with
# the hash seed.  An in-process check cannot see this — one process has one
# seed — so each seed runs in its own interpreter.
#
# The child cannot receive the in-memory provider, so the parent saves the
# run's provider as parquet (``save_provider``) and the child repeats the
# production load on it (``load_saved``).  The child builds the axis Enums
# (``axis_enums=True``): #153 is an Enum category-order bug, so this probe is
# one of the tests of the Enum activation itself (production loads without
# Enums).
# ---------------------------------------------------------------------------

_HASHSEED_PROBE = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from polar_high import Problem
from flextool.engine import build_flextool
from flextool.engine.solve import _region_filter
from flextool.engine.solve._benders import solve_benders
from flextool.engine.input._axis_enums import set_global_axis_enums
from tests._support.load import load_saved

saved, db_url, scenario, load_folder = sys.argv[2:6]
data = load_saved(saved, db_url, scenario, load_folder, axis_enums=True)
assert data._axis_enums is not None, "the probe must run with axis Enums (#153)"
regions = ["region_A", "region_B", "region_C"]
if getattr(data, "_axis_enums", None) is not None:
    set_global_axis_enums(data._axis_enums)
lps = {}
for s in _region_filter.split(data, regions=regions, benders_uncap_cross_region=True):
    pb = Problem()
    build_flextool(pb, s.data)
    sol = pb.solve()
    lps[s.region] = hashlib.sha256(
        "\n".join(list(sol.col_names) + list(sol.row_names)).encode()).hexdigest()
out = {"lps": lps}
for master in ("hand", "flextool"):
    res = solve_benders(data, regions, max_iters=20, tol=1e-4, master=master)
    out[master] = {
        "iterations": res.iterations, "lb": repr(res.lower_bound), "ub": repr(res.upper_bound),
        "invest": {k: repr(v) for k, v in sorted(res.invest.items())},
    }
print("PROBE" + json.dumps(out))
"""


@pytest.fixture(scope="module")
def ti_retuned_run(scenario_run):
    return scenario_run(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest_retuned"
    )


def test_region_lps_and_trajectory_independent_of_hash_seed(ti_retuned_run, tmp_path) -> None:
    import json
    import os
    import subprocess
    import sys

    run = ti_retuned_run
    # ``load_saved`` passes no handoff: the run must be one step that consumed none.
    assert [s.prior_handoff for s in run.steps] == [None], [s.solve_name for s in run.steps]
    saved = tmp_path / "provider"
    save_provider(run.last.provider, saved)
    repo_root = Path(__file__).resolve().parents[2]
    outs = {}
    # Seeds 0 and 2 ordered the half-flow tokens differently before the fix
    # (seed 2 then also failed the hand master with kUnknown).
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        load_folder = tmp_path / f"load_seed{seed}"
        load_folder.mkdir()
        proc = subprocess.run(
            [sys.executable, "-c", _HASHSEED_PROBE, str(repo_root), str(saved), run.db_url,
             run.scenario, str(load_folder)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, (
            f"PYTHONHASHSEED={seed}: probe failed\n{proc.stderr[-4000:]}"
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE")][-1]
        outs[seed] = json.loads(line[len("PROBE"):])
    ref = outs["0"]
    for seed, got in outs.items():
        assert got["lps"] == ref["lps"], (
            f"region LP column/row order differs between PYTHONHASHSEED=0 and {seed}"
        )
        assert got == ref, f"Benders trajectory differs: seed 0 {ref} vs seed {seed} {got}"
