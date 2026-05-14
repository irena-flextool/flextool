"""Cascade-vs-seed parity diagnostic and step_duration > 1 guard.

The cascade in ``apply_derived_a/b/c/...`` overrides FlexData fields
that ``_load_*`` already seeded from ``solve_data/*.csv``.  On the
``lt_rp`` model with ``step_duration = 8760`` the ``p_inflow``
override produced a wrong LP (the cascade joined raw per-hour source
data against ``dt`` without scaling by ``step_duration``); the
override is now hard-disabled in ``apply_derived_a`` and the seed
survives — see ``specs/native_inflow_pathway_disabled.md``.

Other cascade overrides may suffer the same bug class.  This test
A/Bs the two paths per fixture:

* **seed**     — solve with ``_cascade_gate()`` armed so
  ``FlexData.__setattr__`` drops every cascade reassignment of a
  field that already has a seeded value.  Effectively "what the LP
  would see if every ``*_from_source`` helper returned None".
* **cascade**  — solve as production does, with the cascade free to
  override seeds.

Both are then compared to the pre-baked flextool reference objective
in ``output_raw/v_obj__<solve>.parquet`` (the canonical truth).

Three outcomes per fixture:

* **seed ≡ cascade ≡ flextool** — cascade overrides are no-ops or
  perfectly replicate the seed.  All good.
* **seed ≠ cascade, cascade ≡ flextool** — the cascade is doing
  necessary work; the CSV seed is stale.  This is the dominant
  pattern in the existing fixture corpus.  Recorded as ``xfail`` so
  the test stays green while we audit which seeds need refreshing.
* **seed ≡ flextool, cascade ≠ flextool** — the cascade is broken
  (the ``p_inflow`` lt_rp bug class).  Test fails; the diverging
  cascade override needs to be disabled or fixed.

Why both directions matter:

The ``p_inflow`` bug went undetected for a long time because nothing
compared the cascade against the seed on a fixture where the seed
was right.  Inversely, the existing fixtures pass CI because the
cascade is right where the seed is stale.  Catching either bug class
requires running both paths and comparing.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars.input import _cascade_gate


pytestmark = pytest.mark.solver

_DATA = Path(__file__).resolve().parent / "data"

# ``True`` flags the fixture as a known-stale-seed case (cascade is
# necessary for parity with flextool reference).  Move to ``False``
# once the seed CSV has been refreshed (or once we've established the
# cascade override is correct for an a-priori reason).
_FIXTURES: list[tuple[str, str, bool]] = [
    # fixture                                solve                       seed_stale
    ("work_commodity_ladder_cumulative",  "y2020_2day_dispatch",       False),
    ("work_commodity_ladder_annual",      "y2020_2day_dispatch",       False),
    ("work_coal",                          "y2020_2day_dispatch",       True),
    ("work_base",                          "y2020_2day_dispatch",       False),
    ("work_network_coal_wind",             "y2020_2day_dispatch",       True),
    ("work_wind_battery_invest",           "y2020_2day_dispatch",       False),
]


def _solve(workdir: Path, *, gated: bool) -> float:
    if gated:
        with _cascade_gate():
            data = load_flextool(workdir)
    else:
        data = load_flextool(workdir)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"solve not optimal for {workdir} (gated={gated})"
    return float(sol.obj)


@pytest.mark.parametrize(
    "fixture,solve,seed_stale",
    _FIXTURES,
    ids=[f for f, _, _ in _FIXTURES],
)
def test_cascade_matches_flextool_reference(
        fixture: str, solve: str, seed_stale: bool):
    """The cascade-path LP objective matches flextool's reference.

    Always asserts cascade ≡ flextool; this catches the ``p_inflow``
    lt_rp bug class (cascade override produces a wrong model).
    """
    workdir = _DATA / fixture
    if not workdir.exists():
        pytest.skip(f"fixture missing: {workdir}")
    ref_path = workdir / "output_raw" / f"v_obj__{solve}.parquet"
    if not ref_path.exists():
        pytest.skip(f"no flextool reference at {ref_path}")

    ref = float(pl.read_parquet(ref_path)["objective"][0])
    obj = _solve(workdir, gated=False)
    rel = abs(obj - ref) / max(1.0, abs(ref))
    assert rel < 1e-6, (
        f"cascade-path objective drifts from flextool reference on "
        f"{fixture}: cascade={obj:.6f}, flextool={ref:.6f}, "
        f"rel={rel:.3e}"
    )


@pytest.mark.parametrize(
    "fixture,solve,seed_stale",
    _FIXTURES,
    ids=[f for f, _, _ in _FIXTURES],
)
def test_seed_path_matches_flextool_reference(
        fixture: str, solve: str, seed_stale: bool):
    """The seed-only (cascade-disabled) LP objective matches flextool.

    Fixtures flagged ``seed_stale=True`` are ``xfail`` — the cascade
    is doing necessary work in those.  When you refresh the CSV
    seeds in such a fixture, flip the flag to False and watch the
    test go xpassed.
    """
    workdir = _DATA / fixture
    if not workdir.exists():
        pytest.skip(f"fixture missing: {workdir}")
    ref_path = workdir / "output_raw" / f"v_obj__{solve}.parquet"
    if not ref_path.exists():
        pytest.skip(f"no flextool reference at {ref_path}")

    ref = float(pl.read_parquet(ref_path)["objective"][0])
    obj = _solve(workdir, gated=True)
    rel = abs(obj - ref) / max(1.0, abs(ref))
    if seed_stale:
        if rel < 1e-6:
            pytest.fail(
                f"{fixture} marked seed_stale=True but seed path "
                f"matches flextool — flip the flag to False")
        pytest.xfail(
            f"seed CSV is stale on {fixture}; cascade override "
            f"provides correctness.  seed={obj:.6f}, flextool={ref:.6f}, "
            f"rel={rel:.3e}.  Refresh CSV seeds or migrate this "
            f"fixture's cascade dependency to an explicit field")
    assert rel < 1e-6, (
        f"seed-path objective drifts from flextool reference on "
        f"{fixture}: seed={obj:.6f}, flextool={ref:.6f}, rel={rel:.3e}"
    )


def _build_step_duration_fixture(src: Path, dst: Path, scale: float) -> None:
    """Clone ``src`` into ``dst`` with every step_duration scaled by
    ``scale`` and every per-step inflow scaled by ``scale`` to match.

    The transformation keeps the LP internally consistent: each
    timestep now covers ``scale`` hours and the demand booked in
    ``pdtNodeInflow.csv`` (MWh-per-step) is multiplied accordingly.
    Steady-state instantaneous power is unchanged; total demand
    across the horizon multiplies by ``scale``.

    Only the seed-side CSVs are modified — the upstream
    ``tests.sqlite`` (Spine DB) is copied verbatim.  Under the
    ``p_inflow`` cascade disable this means ``flex_data.p_inflow``
    carries the ×``scale`` values.  If a future regression re-enables
    the override and reads raw per-hour values from the DB, the seed
    survives only at 1× and the ``test_step_duration_3_…`` assertion
    below fails.
    """
    shutil.copytree(src, dst)
    sd = dst / "solve_data"

    # step_duration files — value is in the rightmost float column.
    for fname in (
        "steps_in_use.csv",
        "p_step_duration.csv",
        "block_step_duration.csv",
    ):
        path = sd / fname
        if not path.exists():
            continue
        df = pl.read_csv(path)
        # rightmost column is the duration value
        value_col = df.columns[-1]
        df = df.with_columns(
            (pl.col(value_col).cast(pl.Float64) * scale).alias(value_col)
        )
        df.write_csv(path)

    # pdtNodeInflow.csv — MWh-per-step scales with step_duration.
    inflow_path = sd / "pdtNodeInflow.csv"
    if inflow_path.exists():
        df = pl.read_csv(inflow_path)
        df = df.with_columns(
            (pl.col("value").cast(pl.Float64) * scale).alias("value")
        )
        df.write_csv(inflow_path)


def test_step_duration_3_keeps_seed_intact(tmp_path: Path):
    """``step_duration > 1`` regression guard.

    Build a copy of ``work_base`` with ``step_duration = 3`` and a
    consistently-scaled ``pdtNodeInflow.csv``.  Under the current
    code (``p_inflow`` override disabled), the seed survives and
    ``flex_data.p_inflow`` carries the ×3 scaled values verbatim.

    If a future change re-enables the cascade override path and the
    override is still ``step_duration``-unaware (the lt_rp bug
    class), it will silently replace the seed's ×3 values with the
    raw per-hour values from ``tests.sqlite`` — i.e. divide demand
    by 3.  The assertion below catches that.
    """
    src = _DATA / "work_base"
    if not src.exists():
        pytest.skip(f"fixture missing: {src}")
    workdir = tmp_path / "work_base_step3"
    _build_step_duration_fixture(src, workdir, scale=3.0)

    # The seed CSV after scaling
    scaled_seed = pl.read_csv(workdir / "solve_data" / "pdtNodeInflow.csv")
    scaled_sum = float(scaled_seed["value"].cast(pl.Float64).sum())
    assert scaled_sum != 0.0, "seed CSV must have non-zero rows for this test"

    # Load and confirm the seed survived end-to-end
    d = load_flextool(workdir)
    fr = d.p_inflow.frame if hasattr(d.p_inflow, "frame") else d.p_inflow
    loaded_sum = float(fr["value"].sum())

    rel = abs(loaded_sum - scaled_sum) / max(1.0, abs(scaled_sum))
    assert rel < 1e-9, (
        f"step_duration=3 seed did not survive into flex_data.p_inflow.  "
        f"Scaled seed sum={scaled_sum}, loaded sum={loaded_sum}, "
        f"rel={rel:.3e}.  Most likely cause: the native p_inflow "
        f"cascade override in apply_derived_a has been re-enabled — "
        f"see specs/native_inflow_pathway_disabled.md."
    )

    # Also confirm p_step_duration came through scaled
    sd = d.p_step_duration.frame
    sd_max = float(sd["value"].max())
    assert sd_max == pytest.approx(3.0), (
        f"p_step_duration scaling lost in load — got max={sd_max}"
    )


def test_step_duration_3_lp_objective_scales(tmp_path: Path):
    """End-to-end LP solve with ``step_duration = 3`` and ×3 demand
    produces an objective ~3× the unmodified baseline.

    Catches step_duration-unaware bugs anywhere in the LP build
    pipeline (not just ``p_inflow``).  If a cost coefficient,
    penalty, or flow upper bound silently ignores ``step_duration``,
    the modified LP's objective drifts from the 3× target.
    """
    src = _DATA / "work_base"
    if not src.exists():
        pytest.skip(f"fixture missing: {src}")

    baseline = _solve(src, gated=False)

    workdir = tmp_path / "work_base_step3"
    _build_step_duration_fixture(src, workdir, scale=3.0)
    modified = _solve(workdir, gated=False)

    ratio = modified / baseline
    # Tolerance is wide because the LP has free dispatch decisions —
    # aggregating into longer blocks can reshape the optimal flow
    # pattern enough to change the objective by O(10 %).  What we're
    # protecting against is order-of-magnitude divergence — the
    # p_inflow bug would push the ratio toward 1 (or below).
    assert 2.5 < ratio < 3.5, (
        f"step_duration=3 LP objective doesn't track the demand "
        f"scaling: baseline={baseline:.4e}, modified={modified:.4e}, "
        f"ratio={ratio:.3f}.  Expected ~3.0× (since demand scaled "
        f"3× and step_duration scaled 3×).  A ratio near 1 indicates "
        f"the LP is silently dropping the ×3 demand scaling — most "
        f"likely the p_inflow override is back."
    )


def test_inflow_seed_survives_under_cascade():
    """Regression guard for the ``p_inflow`` lt_rp bug.

    The native ``p_inflow`` override in ``apply_derived_a`` is
    hard-disabled (see ``specs/native_inflow_pathway_disabled.md``);
    this test confirms the disable holds by checking that loading
    a fixture without the gate still leaves ``flex_data.p_inflow``
    equal to the seed CSV's row count.  If a future patch re-enables
    the override and it diverges from the seed, this test catches it.
    """
    workdir = _DATA / "work_commodity_ladder_cumulative"
    if not workdir.exists():
        pytest.skip(f"fixture missing: {workdir}")
    seed_csv = workdir / "solve_data" / "pdtNodeInflow.csv"
    if not seed_csv.exists():
        pytest.skip(f"no pdtNodeInflow.csv on {workdir}")
    seed_rows = pl.read_csv(seed_csv)
    seed_sum = float(seed_rows["value"].cast(pl.Float64).sum())

    d = load_flextool(workdir)
    fr = d.p_inflow.frame if hasattr(d.p_inflow, "frame") else d.p_inflow
    loaded_sum = float(fr["value"].sum())

    rel = abs(loaded_sum - seed_sum) / max(1.0, abs(seed_sum) or 1.0)
    assert rel < 1e-9, (
        f"p_inflow diverged from seed CSV: seed_sum={seed_sum}, "
        f"loaded_sum={loaded_sum}, rel={rel:.3e}.  This means the "
        f"native cascade override re-entered apply_derived_a — see "
        f"specs/native_inflow_pathway_disabled.md")
