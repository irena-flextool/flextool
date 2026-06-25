"""Regression: non-unit stochastic branch weights must reach the objective.

The twin of ``test_rp_weight_applied.py`` for the *stochastic-branch*
weight path.  Where the RP test pins ``representative_period_weights``,
this one pins the per-branch probability weights that flow from
``solve.stochastic_branches`` through the derived cascade into the
objective.

Vehicle: the ``2_day_stochastic_dispatch`` scenario from
``tests/fixtures/stochastics.json`` (a 2-day / 48h model whose
``2day_dispatch`` solve declares four branches —
``realized`` / ``upper`` / ``lower`` / ``mid`` — over one base period).
Every committed stochastic fixture ships UNIT (1.0) branch weights, so
nothing exercised non-unit weights end-to-end into the objective.  This
test injects non-unit weights and asserts:

1. **Normalisation reaches the CSV / FlexData.**  Injecting input
   weights ``realized=1 / upper=2 / lower=3 / mid=4`` (sum 10)
   produces sibling-normalised ``pd_branch_weight`` /
   ``pdt_branch_weight`` of ``0.1 / 0.2 / 0.3 / 0.4`` — NOT a dense 1.0.
   The normalisation is done by
   ``apply_derived_g`` →
   ``_derived_branch.apply_branch_cluster`` (``pd_branch_weight_lf`` /
   ``pdt_branch_weight_lf``), which divides each branch's input weight
   by the sum across siblings sharing the same first-step; the
   per-(d, t) value is emitted to ``solve_data/pdt_branch_weight.csv``
   by ``_emit_period_calc.emit_branch_weights``.

2. **Objective-sensitivity (liveness).**  Two solves whose branch
   weights are a pure REDISTRIBUTION with the sum preserved
   (``1/2/3/4`` vs ``4/3/2/1``, both sum 10) yield DIFFERENT objective
   values.  Because the sum is held fixed the only thing that changes is
   which branch carries the larger probability — so a non-zero delta
   proves the normalised branch weights genuinely fold into the
   per-branch cost terms of the objective (``op_factor`` at
   ``model.py:3664`` for pdt and ``model.py:3963`` for pd).  Were the
   weights silently clobbered to 1.0 (or otherwise not reaching the
   objective) this delta would be exactly 0.0.

The full ``run_chain_from_db`` cascade (HiGHS solve included) is
required — it is the path that NORMALISES the weights, EMITS the folded
CSVs, and LOADS them back into FlexData for the objective.  We cannot
read a checked-in ``.sqlite``; instead we build the DB from the JSON
fixture under ``tmp_path`` and rewrite the ``stochastic_branches`` map's
leaf weights before importing (CLAUDE.md invariant 3).
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"
STOCHASTICS_JSON = TESTS_DIR / "fixtures" / "stochastics.json"

# ``json_to_db`` lives in ``tests/db_utils.py``; make it importable.
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


# Normalised branch weights expected once the cascade folds the injected
# input weights ``1 / 2 / 3 / 4`` (sum 10): each value is ``w_b / sum_b``.
_BASE_NORMALISED_WEIGHTS = {0.1, 0.2, 0.3, 0.4}

# Two sum-preserving (sum == 10) input-weight distributions over the
# four branches.  The redistribution keeps the denominator constant so
# the only thing that changes is which branch is the most probable.
_DIST_A = {"realized": 1.0, "upper": 2.0, "lower": 3.0, "mid": 4.0}
_DIST_B = {"realized": 4.0, "upper": 3.0, "lower": 2.0, "mid": 1.0}


def _rewrite_branch_weights(
    json_path: Path, weights: dict[str, float], db_path: Path
) -> str:
    """Build a stochastics DB whose ``2day_dispatch`` branch weights are
    rewritten to *weights* (a ``branch -> input_weight`` map).

    Reads the committed JSON fixture, rewrites the base64-encoded Spine
    Map's leaf values for the ``2day_dispatch`` solve's
    ``stochastic_branches`` parameter, dumps the modified JSON to a temp
    file, and imports it into a fresh SQLite DB.  Returns the
    ``sqlite:///`` URL.
    """
    from db_utils import json_to_db

    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(json_path.read_text())
    rewritten = False
    for row in data["parameter_values"]:
        if (
            row[0] == "solve"
            and row[1] == "2day_dispatch"
            and row[2] == "stochastic_branches"
        ):
            # Spine Map nesting: period -> branch -> timestep ->
            # {yes|no: weight}.  The leaf weight is the branch's input
            # probability weight.
            spine_map = json.loads(base64.b64decode(row[3][0]))
            for _period, period_map in spine_map["data"]:
                for branch, branch_map in period_map["data"]:
                    if branch not in weights:
                        raise AssertionError(
                            f"unexpected branch {branch!r}; the fixture's "
                            f"branch set changed — update this test"
                        )
                    for _tstep, leaf in branch_map["data"]:
                        yes_no = leaf["data"][0][0]
                        leaf["data"] = [[yes_no, weights[branch]]]
            row[3][0] = base64.b64encode(
                json.dumps(spine_map).encode()
            ).decode("ascii")
            rewritten = True
    assert rewritten, (
        "did not find the 2day_dispatch stochastic_branches parameter — "
        "the fixture layout changed"
    )

    tmp_json = db_path.with_suffix(".rewritten.json")
    tmp_json.write_text(json.dumps(data))
    url = json_to_db(tmp_json, db_path)
    migrate_database(url)
    return url


def _solve_branch_scenario(weights, tmp_path_factory, test_solver_config_dir):
    """Build a DB with *weights*, run the cascade, return (workdir, solution)."""
    from flextool.engine_polars import run_chain_from_db

    root = tmp_path_factory.mktemp("branch_weight_run")
    db_path = root / "stochastics.sqlite"
    url = _rewrite_branch_weights(STOCHASTICS_JSON, weights, db_path)

    workdir = root / "wf"
    workdir.mkdir()
    # ``run_chain_from_db`` writes solve_data/ relative to CWD.
    os.chdir(workdir)
    steps = run_chain_from_db(
        url,
        "2_day_stochastic_dispatch",
        work_folder=workdir,
        solver_config_dir=test_solver_config_dir,
        csv_dump=True,
        keep_solutions=True,
    )
    assert steps, "run_chain_from_db returned no steps"
    last_step = next(reversed(steps.values()))
    sol = last_step.solution
    assert sol is not None and sol.optimal, (
        "stochastic dispatch model failed to solve"
    )
    provider = getattr(last_step, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(workdir)
    return workdir, sol


def _csv_weight_set(workdir: Path, name: str) -> set[float]:
    path = workdir / "solve_data" / name
    assert path.exists(), f"cascade did not emit {path}"
    df = pl.read_csv(path)
    return {round(float(w), 9) for w in df.get_column("value").to_list()}


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, test_solver_config_dir):
    """Distribution A (1/2/3/4) — the base non-unit-weight solve."""
    return _solve_branch_scenario(
        _DIST_A, tmp_path_factory, test_solver_config_dir
    )


@pytest.fixture(scope="module")
def swapped_run(tmp_path_factory, test_solver_config_dir):
    """Distribution B (4/3/2/1) — same sum, redistributed probabilities."""
    return _solve_branch_scenario(
        _DIST_B, tmp_path_factory, test_solver_config_dir
    )


# --- Tests ------------------------------------------------------------------


def test_branch_weights_normalised_in_csv_and_flexdata(base_run):
    """NORMALISATION: injected 1/2/3/4 input weights fold to the
    sibling-normalised 0.1/0.2/0.3/0.4 in both emitted weight CSVs —
    NOT a dense 1.0 (which is what a non-stochastic / clobbered path
    would emit)."""
    workdir, _sol = base_run

    pd_w = _csv_weight_set(workdir, "pd_branch_weight.csv")
    pdt_w = _csv_weight_set(workdir, "pdt_branch_weight.csv")

    assert pd_w == _BASE_NORMALISED_WEIGHTS, (
        f"pd_branch_weight.csv weights {pd_w} != expected normalised "
        f"{_BASE_NORMALISED_WEIGHTS}"
    )
    assert pdt_w == _BASE_NORMALISED_WEIGHTS, (
        f"pdt_branch_weight.csv weights {pdt_w} != expected normalised "
        f"{_BASE_NORMALISED_WEIGHTS}"
    )
    # Sanity: genuinely non-unit (the whole point — a dense-1.0 emit
    # would mean the branch weights never reached the cascade).
    assert pd_w != {1.0}, "pd_branch_weight is all-1.0 — branch weights lost."
    assert pdt_w != {1.0}, "pdt_branch_weight is all-1.0 — branch weights lost."


def test_branch_weight_redistribution_moves_objective(base_run, swapped_run):
    """LIVENESS: redistributing the branch probabilities (sum preserved)
    MOVES the objective — proving the normalised branch weights fold into
    the per-branch cost terms of the objective.

    Distribution A (1/2/3/4) and B (4/3/2/1) have the SAME sum (10), so
    the sibling-normalisation denominator is identical; only which branch
    is most probable changes.  Were the weights clobbered to 1.0 (or not
    reaching the objective) the two solves would be identical and this
    delta would be exactly 0.0.
    """
    _workdir_a, sol_a = base_run
    _workdir_b, sol_b = swapped_run

    m_a = sol_a.obj
    m_b = sol_b.obj
    delta = abs(m_a - m_b)

    # Pre-regression this delta would be 0.0; the redistribution shifts
    # the expected-cost objective by ~1e7 here.  Guard with a generous
    # absolute floor far above any solver noise.
    assert delta > 1.0, (
        "branch-weight redistribution did not move the objective — "
        "stochastic branch weights are NOT reaching the objective. "
        f"M_A={m_a!r}, M_B={m_b!r}, delta={delta!r}"
    )
    # Both solves are finite, optimal and in the same broad band (the
    # redistribution reweights costs, it does not change the model size).
    assert 1e7 < m_a < 1e8, f"M_A out of band: {m_a!r}"
    assert 1e7 < m_b < 1e8, f"M_B out of band: {m_b!r}"
