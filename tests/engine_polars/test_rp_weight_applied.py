"""Regression: ``representative_period_weights`` must reach the objective.

Exercises the Benders Phase-3b ``lh2_three_region_rp_invest`` fixture — a
2-day (48h) three-region LH2 model with two representative periods per
FlexTool period carrying NON-UNIT ``representative_period_weights``.

The fixture ships three sibling scenarios over one base topology,
differing ONLY in their RP weight values (see
``tests/fixtures/regen_lh2_three_region.py`` ``_RP_WEIGHT_VARIANTS``):

* ``lh2_three_region_rp_invest``        — 0.7/0.3, 0.55/0.45 (base)
* ``lh2_three_region_rp_invest_swap``   — 0.3/0.7, 0.55/0.45 (reps swapped)
* ``lh2_three_region_rp_invest_uniform``— 0.5/0.5, 0.5/0.5 (w≡1)

These pin the RP-weight engine fix
(``_derived_params.p_timestep_weight_from_source`` returns ``None`` on the
RP-only branch so the folded ``solve_data/timestep_weight.csv`` value
survives the guarded call site at ``apply_derived_a``):

1. **Bug fixed (liveness).** Swapping the two y2030 reps' weights (sum
   preserved ⇒ ``period_share`` unchanged) MOVES the monolith objective.
   Pre-fix this delta was EXACTLY ``0.0`` (RP weights silently clobbered
   to 1.0); post-fix it must be clearly non-zero.
2. **FlexData matches CSV.** The loaded ``flex_data.p_timestep_weight``
   equals the emitted ``solve_data/timestep_weight.csv`` folded weights
   (``{1.4, 0.6, 1.1, 0.9}`` for the base case — NOT all-1.0).
3. **w≡1 byte-identity.** The uniform-weight scenario folds to a dense
   1.0 ``p_timestep_weight`` — byte-identical to the pre-fix clobber — so
   the fix is inert at unit weights.

See ``specs/benders_option_c.md`` "RP-weight bug — fix design".

NOTE: the cascade ``build_flextool`` + ``.solve()`` path is required (not
a direct hand-built FlexData) because it is the path that EMITS the
folded ``timestep_weight.csv`` and LOADS it back into FlexData — the
exact round-trip the fix repairs.
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Problem

from flextool.engine_polars import build_flextool, load_flextool


# Folded RP weights expected in ``timestep_weight.csv`` for the BASE
# scenario.  ``_compute_rp_frames`` normalises ``w_r = weight·n_rp/n_base``
# = ``weight·2`` (1 base period, 2 reps): y2030 0.7/0.3 → 1.4/0.6,
# y2040 0.55/0.45 → 1.1/0.9.
_BASE_FOLDED_WEIGHTS = {1.4, 0.6, 1.1, 0.9}


def _solve_monolith(workdir):
    """Load the cascade-emitted workdir and solve the monolith."""
    data = load_flextool(workdir)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    return data, sol


def _csv_weights(workdir) -> set[float]:
    path = workdir / "solve_data" / "timestep_weight.csv"
    assert path.exists(), f"cascade did not emit {path}"
    df = pl.read_csv(path)
    return set(
        round(float(w), 9)
        for w in df.get_column("weight").unique().to_list()
    )


def _flexdata_weights(data) -> set[float]:
    return set(
        round(float(v), 9)
        for v in data.p_timestep_weight.frame.get_column("value")
        .unique()
        .to_list()
    )


# --- Per-scenario workdirs (module-scoped, cascade-built once each) ---------


@pytest.fixture(scope="module")
def rp_base_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_rp_invest", db_fixture="lh2_rp_invest"
    )


@pytest.fixture(scope="module")
def rp_swap_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_rp_invest_swap", db_fixture="lh2_rp_invest"
    )


@pytest.fixture(scope="module")
def rp_uniform_workdir(scenario_workdir):
    return scenario_workdir(
        "lh2_three_region_rp_invest_uniform", db_fixture="lh2_rp_invest"
    )


# --- Tests ------------------------------------------------------------------


def test_rp_weights_reach_objective_swap_moves_M(
    rp_base_workdir, rp_swap_workdir
):
    """LIVENESS: swapping the two y2030 reps' weights MOVES the objective.

    Pre-fix the RP weights were clobbered to 1.0 so both solves were
    identical (delta == 0.0).  Post-fix the cost-asymmetric reps make the
    objective strictly depend on which rep carries the larger weight.
    """
    _, sol_base = _solve_monolith(rp_base_workdir)
    _, sol_swap = _solve_monolith(rp_swap_workdir)
    assert sol_base.optimal
    assert sol_swap.optimal

    m_base = sol_base.obj
    m_swap = sol_swap.obj
    delta = abs(m_swap - m_base)

    # Pre-fix delta was EXACTLY 0.0; post-fix it is on the order of 1e10.
    # Guard with a generous absolute floor far above any solver noise.
    assert delta > 1.0, (
        "RP-weight swap did not move the objective — representative_"
        f"period_weights are NOT reaching the objective. "
        f"M_base={m_base!r}, M_swap={m_swap!r}, delta={delta!r}"
    )
    # Pin the post-fix weighted M_rp magnitude (both finite, well-banded).
    assert 1e10 < m_base < 1e11, f"M_base out of band: {m_base!r}"
    assert 1e10 < m_swap < 1e11, f"M_swap out of band: {m_swap!r}"


def test_flexdata_matches_emitted_csv_folded_weights(rp_base_workdir):
    """FLEXDATA == CSV: the loaded p_timestep_weight equals the folded
    weights the cascade emitted to ``timestep_weight.csv`` — NOT 1.0."""
    data, sol = _solve_monolith(rp_base_workdir)
    assert sol.optimal

    csv_w = _csv_weights(rp_base_workdir)
    fd_w = _flexdata_weights(data)

    # The emitted CSV carries the folded RP weights, not a dense 1.0.
    assert csv_w == _BASE_FOLDED_WEIGHTS, (
        f"emitted timestep_weight.csv weights {csv_w} != expected folded "
        f"{_BASE_FOLDED_WEIGHTS}"
    )
    # FlexData must carry the SAME folded weights (single source of truth).
    assert fd_w == csv_w, (
        f"flex_data.p_timestep_weight {fd_w} != emitted CSV {csv_w}; "
        "the RP weights were clobbered (the bug)."
    )
    # Sanity: it is genuinely non-unit (the whole point).
    assert fd_w != {1.0}, "p_timestep_weight is all-1.0 — RP weights lost."


def test_uniform_rp_weights_byte_identical_to_prefix(rp_uniform_workdir):
    """W≡1 BYTE-IDENTITY: uniform RP weights (0.5/0.5) fold to a dense
    1.0 p_timestep_weight — byte-identical to the pre-fix clobber, so the
    fix is inert at unit weights."""
    data, sol = _solve_monolith(rp_uniform_workdir)
    assert sol.optimal

    csv_w = _csv_weights(rp_uniform_workdir)
    fd_w = _flexdata_weights(data)

    # Uniform 0.5/0.5 → folded w_r = 0.5·2 = 1.0 everywhere.
    assert csv_w == {1.0}, f"uniform fold should be all-1.0, got {csv_w}"
    assert fd_w == {1.0}, (
        f"uniform p_timestep_weight should be all-1.0 (byte-identical to "
        f"pre-fix), got {fd_w}"
    )
    # Objective is finite & banded (the uniform run is the byte-identity
    # baseline; its value is unchanged by the fix by construction since
    # the loaded CSV value already coalesces to the dense 1.0 default).
    assert 1e10 < sol.obj < 1e11, f"uniform M out of band: {sol.obj!r}"
