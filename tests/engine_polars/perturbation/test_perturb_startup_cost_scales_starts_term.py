"""Tier-6 perturbation: ``p_startup_cost`` is the per-(process, period)
multiplier on the unit-startup obj term — see
``audit/objective_audit.md`` §6.

The §6 startup-cost term is::

    + Σ_{(p, d, t) ∈ pdt_online_integer ∪ pdt_online_linear}
          v_startup * p_startup_cost * p_unitsize
              * p_rp_cost_weight * p_inflation_op / p_period_share

(no ``p_step_duration`` factor — startup is a discrete event, not a
duration-weighted flow.)

work_coal_min_load_MIP_wind exercises the integer branch
(``pdt_online_integer``): the wind variability forces coal cycling
on/off, giving non-zero ``v_startup_integer`` at the optimum.
Doubling ``p_startup_cost`` doubles the §6 obj coefficient on every
(p, d, t) startup tuple; because the cost change preserves the
integer optimum's startup pattern (the cheaper alternative is to
keep coal off and pay slack at 900 €/MWh, which is unchanged), the
LP-MIP optimum's v_startup_integer doesn't move and the obj delta
equals the baseline §6 term.

A failure here narrows the bug to: a missing ``p_startup_cost``
factor on the §6 obj term, or a missed factor in the chain
(unitsize, rp_cost_weight, inflation, /period_share, or an
inadvertent step_duration multiplier).

flextool counterpart:
``flextool/tests/perturbation/test_perturb_startup_cost_scales_starts_term.py``
(uses LP-relaxed ``coal_min_load_wind``; flexpy uses the MIP variant
for clean v_startup invariance under the perturbation).
"""

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import load_flextool

from tests.perturbation._harness import (
    scale_param,
    solve_full,
    assert_obj_changed_by,
)


WORK = (Path(__file__).resolve().parents[1]
        / "data" / "work_coal_min_load_MIP_wind")


@pytest.fixture(scope="module")
def mip_data():
    return load_flextool(WORK)


def _startup_term(d, sol) -> float:
    """Closed-form value of obj §6 (startup cost) on the baseline LP/MIP.

    Sums the integer and linear branches; one of the two is empty
    in any given fixture.
    """
    total = 0.0
    base_factor_cols = ["p_startup_cost", "p_unitsize",
                        "p_rp_cost_weight", "p_inflation_op",
                        "p_period_share"]
    del base_factor_cols  # not used directly; left here for documentation

    def _branch(idx_set, var_name):
        try:
            startup = sol.value(var_name).rename({"value": "vs"})
        except KeyError:
            return 0.0
        df = (
            idx_set
            .join(startup, on=["p", "d", "t"], how="inner")
            .join(d.p_startup_cost.frame.rename({"value": "sc"}),
                  on=["p", "d"])
            .join(d.p_unitsize.frame.rename({"value": "us"}), on="p")
            .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}),
                  on=["d", "t"])
            .join(d.p_inflation_op.frame.rename({"value": "infl"}),
                  on="d")
            .join(d.p_period_share.frame.rename({"value": "psh"}),
                  on="d")
        )
        df = df.with_columns(
            contrib=pl.col("vs") * pl.col("sc") * pl.col("us")
                    * pl.col("rpcw") * pl.col("infl") / pl.col("psh")
        )
        return float(df["contrib"].sum())

    if d.pdt_online_linear is not None and d.pdt_online_linear.height > 0:
        total += _branch(d.pdt_online_linear, "v_startup_linear")
    if d.pdt_online_integer is not None and d.pdt_online_integer.height > 0:
        total += _branch(d.pdt_online_integer, "v_startup_integer")
    return total


@pytest.mark.perturbation
def test_perturb_startup_cost_scales_starts_term(mip_data):
    if mip_data.p_startup_cost is None:
        pytest.skip("p_startup_cost not present in fixture")

    factor = 2.0

    pb_base, sol_base = solve_full(mip_data)
    base_obj = float(sol_base.obj)
    term_base = _startup_term(mip_data, sol_base)
    assert term_base > 0, (
        f"baseline starts component is {term_base!r}; perturbation "
        f"test needs a non-zero starts obj term — fixture choice issue")

    perturbed = scale_param(mip_data, "p_startup_cost", factor)
    pb_pert, sol_pert = solve_full(perturbed)
    perturbed_obj = float(sol_pert.obj)

    # Sanity: with integer online, doubling startup_cost preserves
    # the optimum's startup pattern in this fixture (slack at 900
    # €/MWh is the unchanged alternative).
    var = ("v_startup_integer"
           if (mip_data.pdt_online_integer is not None
               and mip_data.pdt_online_integer.height > 0)
           else "v_startup_linear")
    sb = sol_base.value(var).sort("p", "d", "t")
    sp = sol_pert.value(var).sort("p", "d", "t")
    assert (sb["value"].to_numpy()
            == pytest.approx(sp["value"].to_numpy(), abs=1e-7))

    expected_delta = (factor - 1.0) * term_base
    assert_obj_changed_by(base_obj, perturbed_obj, expected_delta)
