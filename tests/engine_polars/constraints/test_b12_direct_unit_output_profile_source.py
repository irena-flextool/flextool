"""Regression: ``profile_flow_fixed`` on a DIRECT unit's OUTPUT-side
profile must key the constraint LHS on the unit's REAL input-node source.

The bug
-------
A DIRECT (``constant_efficiency``) unit with a real input node and an
output-node ``fixed`` profile (the Koeberg class) builds its real flow
arc as ``v_flow[unit, input_node, output_node]``.  But the per-roll
*projection* producer
:func:`flextool.engine_polars._projection_params._profile_method_arc`
hard-coded ``source = unit`` on its output-side branch, so the
``process_profile_fixed`` ``(p, source, sink, f)`` tuple came out as
``(unit, unit, output_node, f)``.  ``_add_profile_cstr`` then filters
``v_flow`` on ``(p, source, sink)`` — and there is no
``v_flow[unit, unit, output_node]`` for a direct unit — yielding an
empty LHS ``0 == rhs`` row → HiGHS presolve **Infeasible**.

This projection override WINS in the live cascade (it runs in
``apply_projection_params`` *after* ``_load_profiles`` reads the
correct CSV-union derivation), so the whole solve goes infeasible.

Source-less generators (``conversion_method = none``, no input node)
are unaffected: their real arc genuinely IS ``(p, p, sink)``, so the
``source = unit`` aliasing is accidentally correct.  The fix must stay
a no-op for them.

Fixtures
--------
Both cases are composed on the committed ``tests.json`` fixture (built
fresh from JSON per CLAUDE.md invariant 3 — no checked-in ``.sqlite``),
on the tiny ``init`` 2-day single-solve timeline.

* ``coal_plant`` — DIRECT ``constant_efficiency`` unit, input
  ``coal_market`` → output ``west``, ``existing = 500``, ``efficiency =
  0.4``.  We attach a ``fixed`` OUTPUT profile (constant value 0.6) on
  ``unit__node__profile = (coal_plant, west, coal_out_profile)``.

  Hand-calc (per the .mod ``profile_flow_fixed`` RHS =
  ``profile · existing_count · availability``):
      existing_count = existing / unitsize = 500 / 500 = 1
          (unitsize cascade = virtual_unitsize OR existing OR 1000;
           no virtual_unitsize authored ⇒ unitsize = existing = 500)
      availability    = 1.0  (no availability authored on coal_plant in
                              this scenario ⇒ schema default 1.0)
      profile         = 0.6  (constant)
      ⇒ v_flow[coal_plant, coal_market, west, *, t] = 0.6 · 1 · 1.0
                                                    = 0.6  for every t.
  Pre-fix: the ``(coal_plant, coal_plant, west)`` mis-key makes the LHS
  empty ⇒ ``0 == 0.6`` ⇒ solve is NOT optimal (infeasible).

* ``fusion_plant`` — source-less ``conversion_method = none`` generator,
  output ``west`` only (no input node, no pre-existing profile).  We
  attach a ``fixed`` profile (constant value 0.3) on
  ``(fusion_plant, west, fusion_fixed_profile)``.

  Hand-calc: existing_count = existing / unitsize.  ``fusion`` alt sets
  ``existing = 500`` ⇒ unitsize = 500 ⇒ existing_count = 1; availability
  default 1.0 ⇒ v_flow[fusion_plant, fusion_plant, west, *, t] = 0.3 ·
  1 · 1.0 = 0.3 for every t.  The real arc for a source-less unit IS
  ``(fusion_plant, fusion_plant, west)`` so the ``source = unit``
  aliasing is correct both before AND after the fix — this is the no-op
  parity guard.  (``wind_plant`` is avoided here because the ``wind``
  alt already attaches an ``upper_limit`` profile to its output arc,
  which would clash with a second ``fixed`` profile on the same arc.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

TEST_DIR = Path(__file__).resolve().parents[2]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

FIXTURES_DIR = TEST_DIR / "fixtures"


def _add_direct_output_profile_scenario(db_url: str) -> None:
    """Compose ``coal_fixed_out_profile``: the ``coal`` dispatch scenario
    plus a ``fixed`` OUTPUT profile on the DIRECT ``coal_plant`` and a
    ``fixed`` profile on the source-less ``fusion_plant``.
    """
    from spinedb_api import DatabaseMapping, import_data

    scenario = "coal_fixed_out_profile"
    alt = "coal_fixed_out_profile_alt"
    with DatabaseMapping(db_url) as db_map:
        _, errors = import_data(
            db_map,
            alternatives=[alt],
            entities=[
                ("profile", "coal_out_profile", None),
                ("profile", "fusion_fixed_profile", None),
                ("unit__node__profile",
                 ["coal_plant", "west", "coal_out_profile"], None),
                ("unit__node__profile",
                 ["fusion_plant", "west", "fusion_fixed_profile"], None),
            ],
            parameter_values=[
                # Constant profile values (a scalar broadcasts across t).
                ("profile", "coal_out_profile", "profile", 0.6, alt),
                ("profile", "fusion_fixed_profile", "profile", 0.3, alt),
                # fixed method on both arcs.
                ("unit__node__profile",
                 ["coal_plant", "west", "coal_out_profile"],
                 "profile_method", "fixed", alt),
                ("unit__node__profile",
                 ["fusion_plant", "west", "fusion_fixed_profile"],
                 "profile_method", "fixed", alt),
            ],
            scenarios=[(scenario, False, "")],
            scenario_alternatives=[
                (scenario, "init", "west"),
                (scenario, "west", "coal"),
                (scenario, "coal", "fusion"),
                (scenario, "fusion", alt),
                (scenario, alt, None),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add direct-unit output-profile scenario")


@pytest.fixture(scope="module")
def direct_profile_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    db_path = tmp_path_factory.mktemp("db_direct_prof") / "tests.sqlite"
    url = json_to_db(FIXTURES_DIR / "tests.json", db_path)
    migrate_database(url)
    _add_direct_output_profile_scenario(url)
    return url


def test_direct_unit_output_profile_keys_real_source(
    direct_profile_db_url: str,
    test_solver_config_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A DIRECT unit's fixed OUTPUT profile must pin its real input→output
    flow arc — not collapse to an infeasible ``0 == rhs`` row.

    Pre-fix: the projection override keys ``source = coal_plant`` →
    empty LHS → solve is infeasible (``last_step.solution.optimal`` False
    / no solution).  Post-fix: the real ``(coal_plant, coal_market, west)``
    arc is pinned to 0.6 each step, and the source-less wind arc to 0.3.
    """
    workdir = tmp_path_factory.mktemp("direct_prof_run")
    os.chdir(workdir)

    steps = run_chain_from_db(
        direct_profile_db_url,
        "coal_fixed_out_profile",
        work_folder=workdir,
        solver_config_dir=test_solver_config_dir,
        keep_solutions=True,
    )

    assert steps, "cascade produced no solve steps"
    last_step = next(reversed(steps.values()))
    # The crux: pre-fix the DIRECT-unit output-profile row is an empty-LHS
    # ``0 == 0.6`` → HiGHS presolve Infeasible → NOT optimal.
    assert last_step.solution is not None and last_step.solution.optimal, (
        "solve was not optimal — the DIRECT-unit output profile mis-keyed "
        "the source slot, producing an infeasible 0 == rhs row"
    )

    v_flow = last_step.solution.value("v_flow")
    assert v_flow is not None and v_flow.height > 0

    # ── DIRECT coal_plant: real arc (coal_plant, coal_market, west) ──
    coal = v_flow.filter(
        (pl.col("p") == "coal_plant")
        & (pl.col("source") == "coal_market")
        & (pl.col("sink") == "west")
    )
    assert coal.height > 0, (
        "no v_flow rows on the real (coal_plant, coal_market, west) arc — "
        "the constraint mis-keyed the source (regression)"
    )
    # The mis-keyed (coal_plant, coal_plant, west) arc must NOT exist.
    bogus = v_flow.filter(
        (pl.col("p") == "coal_plant")
        & (pl.col("source") == "coal_plant")
        & (pl.col("sink") == "west")
    )
    assert bogus.height == 0, (
        "a v_flow on the unit-aliased (coal_plant, coal_plant, west) arc "
        "exists — the source slot is still aliased to the unit name"
    )
    # Hand-calc: 0.6 · (500/500) · 1.0 = 0.6 every step.
    coal_vals = coal.get_column("value").to_list()
    assert all(v == pytest.approx(0.6, abs=1e-6) for v in coal_vals), (
        f"coal_plant fixed-output flow != 0.6 each step: {coal_vals[:8]}"
    )

    # ── Source-less fusion_plant: real arc (fusion_plant, fusion_plant, west) ──
    fusion = v_flow.filter(
        (pl.col("p") == "fusion_plant")
        & (pl.col("source") == "fusion_plant")
        & (pl.col("sink") == "west")
    )
    assert fusion.height > 0, "no v_flow on the source-less fusion arc"
    # Hand-calc: 0.3 · (500/500) · 1.0 = 0.3 every step (no-op parity guard).
    fusion_vals = fusion.get_column("value").to_list()
    assert all(v == pytest.approx(0.3, abs=1e-6) for v in fusion_vals), (
        f"fusion_plant fixed flow != 0.3 each step: {fusion_vals[:8]}"
    )
