"""Regression for INFLOW-1: synthetic (nested/rolling) sub-solves must
derive the inflow signed-split so the non_sync / capacity_margin demand
budget survives.

Background
----------
``load_flextool`` has a synthetic-solve EARLY-RETURN: when the active
solve name is NOT a Spine ``solve`` row (true for every nested/rolling
sub-solve, e.g. ``invest_5weeks_p2020`` or
``dispatch_fullYear_roll_roll_0``), it runs only a hand-patched subset
of the derive cascade and ``return``s — skipping ``apply_derived_c``,
which is the sole producer of

  * ``p_positive_inflow``        (clip-low at 0 of ``p_inflow``)
  * ``p_negative_inflow``        (clip-high at 0 of ``p_inflow``)
  * ``pdtNodeInflow_per_step``   (``p_inflow / p_step_duration``)

Without Fix A (INFLOW-1) those three fields stay ``None`` on every
synthetic sub-solve even though ``p_inflow`` itself is correctly loaded.
The downstream consumers silently degrade:

  * ``_add_non_sync`` gates its RHS exogenous-demand term (and its LHS
    positive-inflow term) on these being non-None — so the
    non_synchronous_limit constraint loses the whole demand budget,
    pinning VRE to 0 and letting synchronous (coal) generation backfill.
  * ``_add_cap_margin`` reads ``pdtNodeInflow_per_step`` for its RHS.

This test exercises the *real* synthetic ``load_flextool`` path (via a
nested multi-invest scenario whose per-sub-solve FlexData is loaded
through the early-return branch) and asserts:

  1. all three signed-split fields are now populated (Fix A),
  2. ``p_negative_inflow`` / ``p_positive_inflow`` are byte-identical to
     re-running the exact ``apply_derived_c`` helpers on ``p_inflow``
     (proves the derivation matches the non-synthetic path), and
  3. with a non_synchronous_limit group wired onto a node that carries
     negative inflow, ``_add_non_sync`` emits a ``non_sync_constraint``
     whose exogenous-demand RHS is non-trivial — i.e. the budget that
     the bug silently dropped is restored.

The scenario chosen (``multi_fullYear_battery_nested_multi_invest``) is
nested invest→storage→dispatch, so every step is a synthetic sub-solve.
No checked-in scenario activates non_synchronous_limit, so the group is
injected onto the loaded FlexData for the consumer assertion — the
*derivation* under test is exercised by the genuine load path.
"""
from __future__ import annotations

import shutil

import polars as pl
import pytest

pytestmark = pytest.mark.solver

SCENARIO = "multi_fullYear_battery_nested_multi_invest"


@pytest.fixture(scope="module")
def _synthetic_step_flex_data(test_db_url, test_solver_config_dir, tmp_path_factory):
    """Run the nested chain and return a synthetic sub-solve's FlexData
    that carries some negative (demand) inflow.

    Every step of a nested invest→dispatch chain is a synthetic sub-solve
    (the ``<base>_<anchor>`` names don't exist in Spine), so each step's
    ``flex_data`` was loaded through ``load_flextool``'s synthetic
    early-return branch — the exact code path INFLOW-1 fixes.
    """
    from flextool.engine_polars._orchestration import run_chain_from_db

    wf = tmp_path_factory.mktemp("inflow1") / "work"
    if wf.exists():
        shutil.rmtree(wf)
    wf.mkdir(parents=True)

    steps = run_chain_from_db(
        input_db_url=test_db_url,
        scenario_name=SCENARIO,
        work_folder=wf,
        solver_config_dir=test_solver_config_dir,
        csv_dump=False,
        keep_solutions=True,   # retain per-step flex_data on every step
    )
    assert steps, "nested chain produced no steps"

    # Select on p_inflow (loaded pre-early-return, so present with or
    # without Fix A) carrying negative (demand) inflow.  Selecting on the
    # DERIVED p_negative_inflow would make the test SKIP rather than FAIL
    # when the bug is present (the field would be None) — masking the
    # regression.  Keying on p_inflow guarantees the test reaches the
    # populated-field assertions on both the fixed and buggy paths.
    for name, step in steps.items():
        fd = getattr(step, "flex_data", None)
        if fd is None:
            continue
        pi = getattr(fd, "p_inflow", None)
        if pi is None or pi.frame.height == 0:
            continue
        if pi.frame.filter(pl.col("value") < 0).height > 0:
            return name, fd

    pytest.skip(
        "no synthetic step with negative p_inflow found in "
        f"{SCENARIO} (fixture changed?)"
    )


def test_synthetic_solve_derives_inflow_signed_split(_synthetic_step_flex_data):
    """Fix A: the three signed-split fields are populated on a synthetic
    sub-solve, and match the canonical ``apply_derived_c`` helpers.
    """
    from flextool.engine_polars import _derived_params as drv

    name, fd = _synthetic_step_flex_data

    # p_inflow is loaded pre-early-return; the split must now exist too.
    assert fd.p_inflow is not None and fd.p_inflow.frame.height > 0, (
        f"{name}: p_inflow should be loaded on the synthetic path"
    )
    assert fd.p_positive_inflow is not None, (
        f"{name}: p_positive_inflow is None — INFLOW-1 derivation missing"
    )
    assert fd.p_negative_inflow is not None, (
        f"{name}: p_negative_inflow is None — INFLOW-1 derivation missing"
    )
    assert fd.pdtNodeInflow_per_step is not None, (
        f"{name}: pdtNodeInflow_per_step is None — INFLOW-1 derivation missing"
    )

    # Byte-parity with the canonical (non-synthetic) producers: re-run the
    # SAME helpers apply_derived_c uses on the SAME inputs and compare.
    exp_pos = drv.p_positive_inflow_from_inflow(fd.p_inflow)
    exp_neg = drv.p_negative_inflow_from_inflow(fd.p_inflow)
    exp_per = drv.pdtNodeInflow_per_step_from_inflow(
        fd.p_inflow, fd.p_step_duration)

    assert exp_pos is not None and exp_neg is not None and exp_per is not None

    assert fd.p_positive_inflow.frame.sort("n", "d", "t").equals(
        exp_pos.frame.sort("n", "d", "t")
    ), f"{name}: p_positive_inflow diverges from apply_derived_c helper"
    assert fd.p_negative_inflow.frame.sort("n", "d", "t").equals(
        exp_neg.frame.sort("n", "d", "t")
    ), f"{name}: p_negative_inflow diverges from apply_derived_c helper"
    assert fd.pdtNodeInflow_per_step.frame.sort("n", "d", "t").equals(
        exp_per.frame.sort("n", "d", "t")
    ), f"{name}: pdtNodeInflow_per_step diverges from apply_derived_c helper"

    # And the negative split must equal the clip-high of p_inflow.
    pi = fd.p_inflow.frame
    expect_neg = (pi.with_columns(
        value=pl.when(pl.col("value") < 0)
              .then(pl.col("value"))
              .otherwise(0.0))
        .sort("n", "d", "t"))
    assert fd.p_negative_inflow.frame.sort("n", "d", "t").equals(expect_neg), (
        f"{name}: p_negative_inflow is not the negative clip of p_inflow"
    )


def test_synthetic_solve_non_sync_budget_restored(_synthetic_step_flex_data):
    """With the signed-split populated, ``_add_non_sync`` carries the
    exogenous-demand budget — the term the bug silently dropped.

    A non_synchronous_limit group is injected onto a node that has
    negative (demand) inflow.  The constraint must emit and its RHS
    exo_demand term must be non-trivial (non-zero).  Without Fix A,
    ``p_negative_inflow is None`` and the ``if p_neg is not None`` gate at
    ``_group_slack.py`` skips the exo_demand term entirely (budget lost).
    """
    from polar_high import Param, Problem
    from flextool.engine_polars import _group_slack as gs

    name, fd = _synthetic_step_flex_data

    # Without Fix A this field is None on the synthetic path; assert it
    # up front so the regression surfaces as a clear failure here too.
    assert fd.p_negative_inflow is not None, (
        f"{name}: p_negative_inflow is None — INFLOW-1 derivation missing; "
        f"the non_sync demand budget would be silently dropped"
    )

    # Pick a node carrying negative (demand) inflow.  Select via p_inflow
    # (always present) so the node is identifiable regardless of the fix.
    neg = fd.p_inflow.frame.filter(pl.col("value") < 0)
    assert neg.height > 0, f"{name}: expected some negative inflow"
    node = neg["n"][0]

    g = "ns_grp_inflow1"
    ds = fd.dt.select("d").unique()

    fd.group_node = pl.DataFrame({"g": [g], "n": [node]})
    fd.groupNonSync = pl.DataFrame({"g": [g]})
    fd.pdGroup_non_synchronous_limit = Param(
        ("g", "d"),
        ds.with_columns(g=pl.lit(g), value=pl.lit(0.8)).select("g", "d", "value"),
    )
    fd.p_inv_group_cap = Param(
        ("g", "d"),
        ds.with_columns(g=pl.lit(g), value=pl.lit(1.0)).select("g", "d", "value"),
    )

    pb = Problem()
    cstr_vars: dict = {}
    gs._add_non_sync(pb, fd, cstr_vars)

    assert "non_sync_constraint" in pb.cstr_names(), (
        f"{name}: non_sync_constraint was not emitted"
    )
    assert pb.cstr_row_count("non_sync_constraint") > 0

    # The actual exo_demand RHS term _add_non_sync builds
    # (_group_slack.py): -p_negative_inflow * inv_group_cap *
    # non_synchronous_limit, summed over n in group_node.  With the fix it
    # is non-trivial; with the bug p_negative_inflow is None and the whole
    # term is dropped.
    gn = fd.group_node.join(fd.groupNonSync, on="g", how="inner")
    exo = (
        gn.join(fd.p_negative_inflow.frame.rename({"value": "v"}),
                on="n", how="inner")
          .join(fd.p_inv_group_cap.frame.rename({"value": "iv"}),
                on=["g", "d"], how="inner")
          .join(fd.pdGroup_non_synchronous_limit.frame.rename({"value": "lim"}),
                on=["g", "d"], how="inner")
          .with_columns(value=-pl.col("v") * pl.col("iv") * pl.col("lim"))
          .group_by(["g", "d", "t"]).agg(pl.col("value").sum())
    )
    nonzero = exo.filter(pl.col("value") != 0.0)
    assert nonzero.height > 0, (
        f"{name}: exo_demand budget is all-zero — non_sync demand lost"
    )
    assert abs(exo["value"].max()) > 0.0
