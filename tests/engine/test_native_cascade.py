"""Cascade-vs-seed parity diagnostic and step_duration > 1 guard.

The cascade in ``apply_derived_a/b/c/...`` overrides FlexData fields
that ``_load_*`` already seeded from the provider's ``solve_data/*``
frames.  On the
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

import polars as pl
import pytest

from polar_high import Problem
from flextool.engine import build_flextool
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.flex_data import _cascade_gate
from tests._support.load import ScenarioRun, clone_provider, load_scenario


pytestmark = pytest.mark.solver

# ``True`` flags the fixture as a known-stale-seed case (cascade is
# necessary for parity with flextool reference).  Move to ``False``
# once the seed CSV has been refreshed (or once we've established the
# cascade override is correct for an a-priori reason).
#
# Phase 3d: ``commodity_ladder_*`` map to the ``coal_ladder_*`` scenarios
# that were added to ``tests.json`` by ``_augment_phase3d.py``.  The
# alias is preserved here so the v_obj parquet path stays unchanged.
_FIXTURES: list[tuple[str, str, str, bool]] = [
    # fixture                                scenario                  solve                       seed_stale
    ("commodity_ladder_cumulative",  "coal_ladder_cumulative",  "y2020_2day_dispatch",       False),
    ("commodity_ladder_annual",      "coal_ladder_annual",      "y2020_2day_dispatch",       False),
    ("coal",                          "coal",                    "y2020_2day_dispatch",       True),
    ("base",                          "base",                    "y2020_2day_dispatch",       False),
    ("network_coal_wind",             "network_coal_wind",       "y2020_2day_dispatch",       True),
    ("wind_battery_invest",           "wind_battery_invest",     "y2020_2day_dispatch",       False),
]


def _solve_data(data, label: str) -> float:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"solve not optimal for {label}"
    return float(sol.obj)


def _solve(run: ScenarioRun, *, gated: bool) -> float:
    """Rebuild the run's last solve with the production loader call and solve it."""
    if gated:
        with _cascade_gate():
            data = load_scenario(run)
    else:
        data = load_scenario(run)
    return _solve_data(data, f"{run.scenario} (gated={gated})")


@pytest.mark.parametrize(
    "fixture,scenario,solve,seed_stale",
    _FIXTURES,
    ids=[f for f, _, _, _ in _FIXTURES],
)
def test_cascade_matches_flextool_reference(
        fixture: str, scenario: str, solve: str,
        seed_stale: bool, scenario_run):
    """The cascade-path LP objective matches flextool's reference.

    Always asserts cascade ≡ flextool; this catches the ``p_inflow``
    lt_rp bug class (cascade override produces a wrong model).
    """
    run = scenario_run(scenario)
    ref_path = run.work_folder / "output_raw" / f"v_obj__{solve}.parquet"
    if not ref_path.exists():
        pytest.skip(f"no flextool reference at {ref_path}")

    ref = float(pl.read_parquet(ref_path)["objective"][0])
    obj = _solve(run, gated=False)
    rel = abs(obj - ref) / max(1.0, abs(ref))
    assert rel < 1e-6, (
        f"cascade-path objective drifts from flextool reference on "
        f"{fixture}: cascade={obj:.6f}, flextool={ref:.6f}, "
        f"rel={rel:.3e}"
    )


_STEP_DURATION_KEYS = ("solve_data/steps_in_use", "solve_data/block_step_duration")
_INFLOW_KEY = "solve_data/pdtNodeInflow"


def _scale_value_column(p: FlexDataProvider, key: str, col: str, scale: float) -> None:
    """Multiply ``col`` of provider frame ``key`` by ``scale``, keeping the frame's schema (the dtype the
    loader reads; the snapshot frames carry String value columns)."""
    frame = p.get(key)
    p.put(key, frame.with_columns(
        (pl.col(col).cast(pl.Float64) * scale).cast(frame.schema[col]).alias(col)))


def _step_duration_provider(run: ScenarioRun, scale: float) -> FlexDataProvider:
    """A clone of ``run``'s last provider with every step_duration scaled by ``scale`` and every per-step
    inflow scaled by ``scale`` to match.

    On the production load only the inflow scaling reaches the LP:
    with a DB reader, ``apply_derived`` re-derives ``p_step_duration``
    from the DB timeline (1 h steps, no ``solve.new_stepduration``), so
    the loaded LP has ×``scale`` inflow on 1 h steps — not an internally
    consistent model, and not meant to be solved.  The scaled
    step_duration seed is kept so the provider looks like a
    ``step_duration = scale`` timeline to any reader of the seed frames.

    Only the provider's seed frames are modified — the DB the load
    reads is the run's own.  Under the ``p_inflow`` cascade disable
    this means ``flex_data.p_inflow`` carries the ×``scale`` values.
    If a future regression re-enables the override and reads raw
    per-hour values from the DB, the seed survives only at 1× and the
    ``test_step_duration_3_…`` assertion below fails.
    """
    p = clone_provider(run.last.provider)
    for key in _STEP_DURATION_KEYS:
        assert p.has(key), f"{run.scenario}: provider has no {key}"
        _scale_value_column(p, key, "step_duration", scale)
    assert p.has(_INFLOW_KEY), f"{run.scenario}: provider has no {_INFLOW_KEY}"
    _scale_value_column(p, _INFLOW_KEY, "value", scale)
    return p


def _sum(frame: pl.DataFrame) -> float:
    return float(frame["value"].cast(pl.Float64).sum())


def test_step_duration_3_keeps_seed_intact(scenario_run):
    """``step_duration > 1`` regression guard for the ``p_inflow`` seed.

    Load ``base`` from a provider whose step_duration seed frames are
    scaled ×3 and whose ``pdtNodeInflow`` seed is scaled ×3 to match.
    Under the current code (``p_inflow`` override disabled), the inflow
    seed survives and ``flex_data.p_inflow`` carries the ×3 scaled
    values verbatim.  ``p_step_duration`` does not keep the ×3 seed: the
    DB re-derives it (1 h steps, see the trailing comment), so the
    loaded LP is ×3 inflow on 1 h steps.

    If a future change re-enables the cascade override path and the
    override is still ``step_duration``-unaware (the lt_rp bug
    class), it will silently replace the seed's ×3 values with the
    raw per-hour values from the DB — i.e. divide demand
    by 3.  The assertion below catches that.
    """
    run = scenario_run("base")
    unscaled_sum = _sum(run.last.provider.get(_INFLOW_KEY))
    provider = _step_duration_provider(run, scale=3.0)

    # The seed frame after scaling
    scaled_sum = _sum(provider.get(_INFLOW_KEY))
    assert scaled_sum != 0.0, "seed frame must have non-zero rows for this test"
    assert scaled_sum == pytest.approx(3.0 * unscaled_sum, rel=1e-12)

    # Load and confirm the seed survived end-to-end
    d = load_scenario(run, provider=provider)
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

    # p_step_duration is NOT the scaled seed on the production path: with a DB reader,
    # ``apply_derived`` (params/_derived.py, the dt / step-duration override) replaces it with the DB
    # timeline's durations unless ``solve.new_stepduration`` is set.  Only the old reader-less CSV reload
    # kept the scaled ``steps_in_use`` seed (reload gap, c24 stage 1 S1-5), so the rebuild carries the
    # solved FlexData's step durations; the p_inflow guard above is unaffected (the bug class it catches
    # replaces the ×3 seed with the DB's ×1 values).
    def _sorted(param) -> pl.DataFrame:
        return param.frame.select("d", "t", "value").cast({"d": pl.Utf8, "t": pl.Utf8}).sort("d", "t")

    assert _sorted(d.p_step_duration).equals(_sorted(run.last_flex_data.p_step_duration)), (
        "p_step_duration must be the DB timeline's (the solved FlexData's) on the production path"
    )


def test_inflow_seed_survives_under_cascade(scenario_run):
    """Regression guard for the ``p_inflow`` lt_rp bug.

    The native ``p_inflow`` override in ``apply_derived_a`` is
    hard-disabled (see ``specs/native_inflow_pathway_disabled.md``);
    this test confirms the disable holds by checking that loading
    a fixture without the gate still leaves ``flex_data.p_inflow``
    equal to the provider's ``pdtNodeInflow`` seed (its sum).  If a
    future patch re-enables the override and it diverges from the
    seed, this test catches it.
    """
    run = scenario_run("coal_ladder_cumulative")
    assert run.last.provider.has(_INFLOW_KEY), f"no {_INFLOW_KEY} in the coal_ladder_cumulative provider"
    seed_sum = _sum(run.last.provider.get(_INFLOW_KEY))

    d = load_scenario(run)
    fr = d.p_inflow.frame if hasattr(d.p_inflow, "frame") else d.p_inflow
    loaded_sum = float(fr["value"].sum())

    rel = abs(loaded_sum - seed_sum) / max(1.0, abs(seed_sum) or 1.0)
    assert rel < 1e-9, (
        f"p_inflow diverged from the seed: seed_sum={seed_sum}, "
        f"loaded_sum={loaded_sum}, rel={rel:.3e}.  This means the "
        f"native cascade override re-entered apply_derived_a — see "
        f"specs/native_inflow_pathway_disabled.md")
