"""Cascade-vs-seed parity diagnostic.

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
