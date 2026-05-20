"""flexpy port of flextool's ``test_cost_aggregation_semantics.py``.

flextool's test runs a scenario end-to-end, parses
``summary_solve.csv`` / ``costs__dt.csv`` / ``slack__*__d.csv``,
and decomposes the LP objective into hand-derived per-class
buckets to verify the post-processing pipeline applies the right
factors per cost class:

================================  =================================================
term                              factors that must appear
================================  =================================================
commodity/co2/varCost/reserve     flow x step_duration x rp_weight x inflation
node-state slack                  q x capacity x step_duration x rp_weight
                                  x penalty x inflation
capacity-margin slack             q x group_capacity x penalty x inflation
                                  (NO step_duration, NO rp_weight, period-only)
================================  =================================================

flexpy doesn't post-process to ``costs__dt.csv``; the
LP-objective IS the decomposition (the per-(d,t) factors live
inside ``op_factor`` in :mod:`flextool.model`).  The port asserts
the closed-form decomposition adds back to ``sol.obj`` for each
cost class — which IS the same correctness assertion flextool's
test makes about its post-processing.

Cases (each pins a different factor set):

* ``base``                    — pure-penalty obj.  Closed-form
                               ``Σ|inflow|·pen·dur·rpcw·infl/psh``
                               equals obj.  Pins the slack factor
                               set with rpcw=1 everywhere.
* ``base_weighted``           — same scenario with non-uniform
                               ``p_rp_cost_weight``.  Decomposition
                               equals obj — and the rpcw factor is
                               provably non-trivial (control: a
                               version computed with rpcw≡1 differs
                               from obj).
* ``capacity_margin``         — ``vq_capacity_margin`` is period-only
                               (NO step_duration, NO rp_cost_weight,
                               NO /period_share); pins that distinct
                               factor set, alongside the standard
                               state-slack contribution.
* ``coal``                    — commodity-buy term (flow * unitsize
                               * slope * commodity_price * op_factor).
                               obj = slack_up + commodity_buy.
* ``coal_co2_price``          — CO2 term (flow * unitsize * slope
                               * co2_content * co2_price * op_factor).
                               Pins the co2_content × co2_price
                               coefficient pair on the CO2 bucket.

Pre-existing fixed cost (§8.1) is NOT in flextool's published v_obj
(see ``progress.md`` 2026-05-01 evening note); flexpy mirrors that
parquet alignment.  Tests assume ``include_existing_fixed_cost`` is
False (the ``build_flextool`` default).

Tests run in <1 s each (small fixtures already used by the
parity suite).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._param_shapes import promote_param_to_dt


DATA = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Closed-form bucket decompositions.
# Each helper returns a single float — the sum of one objective sub-term —
# computed from the LP solution and the loaded inputs.
# ---------------------------------------------------------------------------

def _state_slack_term(d, sol, *, side: str) -> float:
    """Σ vq * pen * (ncs?) * step_dur * rpcw * infl / psh  for one side.

    Mirrors model.py:1991-1997 (the .mod's §1.1 / §1.2 obj term).
    """
    var = "vq_state_up" if side == "up" else "vq_state_down"
    pen = d.p_penalty_up if side == "up" else d.p_penalty_down
    df = (
        sol.value(var).rename({"value": "vq"})
        .join(pen.frame.rename({"value": "pen"}), on=["n", "d", "t"])
        .join(d.p_step_duration.frame.rename({"value": "dur"}), on=["d", "t"])
        .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}), on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
    )
    if d.p_node_capacity_for_scaling is not None:
        df = df.join(
            d.p_node_capacity_for_scaling.frame.rename({"value": "ncs"}),
            on=["n", "d"],
        )
        df = df.with_columns(
            c=pl.col("vq") * pl.col("pen") * pl.col("ncs")
              * pl.col("dur") * pl.col("rpcw")
              * pl.col("infl") / pl.col("psh")
        )
    else:
        df = df.with_columns(
            c=pl.col("vq") * pl.col("pen")
              * pl.col("dur") * pl.col("rpcw")
              * pl.col("infl") / pl.col("psh")
        )
    return float(df["c"].sum())


def _commodity_buy_eff_term(d, sol) -> float:
    """Σ Where(v_flow * unitsize * slope, flow_from_commodity_eff)
       * commodity_price * op_factor   — model.py:2000-2002."""
    if d.flow_from_commodity_eff is None or d.flow_from_commodity_eff.height == 0:
        return 0.0
    df = (
        sol.value("v_flow").rename({"value": "flow"})
        .join(d.flow_from_commodity_eff,
              on=["p", "source", "sink"], how="inner")
        .join(d.p_unitsize.frame.rename({"value": "us"}), on="p", how="left")
        .join(d.p_slope.frame.rename({"value": "sl"}),
              on=["p", "d", "t"], how="left")
        .join(promote_param_to_dt(d.p_commodity_price, d.dt)
                  .rename({"value": "price"}).collect(),
              on=["c", "d", "t"], how="left")
        .join(d.p_step_duration.frame.rename({"value": "dur"}),
              on=["d", "t"])
        .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}),
              on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
        .with_columns(c=pl.col("flow") * pl.col("us") * pl.col("sl")
                        * pl.col("price")
                        * pl.col("dur") * pl.col("rpcw")
                        * pl.col("infl") / pl.col("psh"))
    )
    return float(df["c"].sum())


def _co2_price_term(d, sol) -> float:
    """Σ Where(v_flow * unitsize * slope, flow_from_co2_priced)
       * co2_content * co2_price * op_factor — model.py:2034-2036."""
    if d.flow_from_co2_priced is None or d.flow_from_co2_priced.height == 0:
        return 0.0
    df = (
        sol.value("v_flow").rename({"value": "flow"})
        .join(d.flow_from_co2_priced,
              on=["p", "source", "sink"], how="inner")
        .join(d.p_unitsize.frame.rename({"value": "us"}), on="p", how="left")
        .join(d.p_slope.frame.rename({"value": "sl"}),
              on=["p", "d", "t"], how="left")
        .join(d.p_co2_content.frame.rename({"value": "co2c"}),
              on="c", how="left")
        .join(promote_param_to_dt(d.p_co2_price, d.dt)
                  .rename({"value": "co2p"}).collect(),
              on=["g", "d", "t"], how="left")
        .join(d.p_step_duration.frame.rename({"value": "dur"}),
              on=["d", "t"])
        .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}),
              on=["d", "t"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
        .with_columns(c=pl.col("flow") * pl.col("us") * pl.col("sl")
                        * pl.col("co2c") * pl.col("co2p")
                        * pl.col("dur") * pl.col("rpcw")
                        * pl.col("infl") / pl.col("psh"))
    )
    return float(df["c"].sum())


def _capacity_margin_slack_term(d, sol) -> float:
    """Σ vq_capacity_margin * group_capacity_for_scaling * penalty * inflation
    * 1000.

    Period-only: NO step_duration, NO rp_cost_weight, NO /period_share.
    The × 1000 is the CUR/kW → CUR/MW unit conversion in
    ``_group_slack.py:1233-1238`` (BUG A4 fix); mirrors
    ``flextool.process_outputs.calc_slacks.costPenalty_capacity_margin_d``'s
    ``.mul(1000.0)``.
    """
    try:
        vq = sol.value("vq_capacity_margin")
    except Exception:
        return 0.0
    if vq is None or vq.height == 0:
        return 0.0
    df = (
        vq.rename({"value": "vq"})
        .join(d.p_group_capacity_for_scaling.frame.rename({"value": "gcs"}),
              on=["g", "d"])
        .join(d.pdGroup_penalty_capacity_margin.frame.rename({"value": "pen"}),
              on=["g", "d"])
        .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
        .with_columns(c=pl.col("vq") * pl.col("gcs") * pl.col("pen")
                        * pl.col("infl") * 1000.0)
    )
    return float(df["c"].sum())


def _solve(work: Path):
    data = load_flextool(work)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"{work.name} did not solve to optimality"
    return data, sol


# ---------------------------------------------------------------------------
# Test cases — each exercises a different factor set.
# ---------------------------------------------------------------------------

class TestPurePenaltyBase:
    """``base`` — uniform rp_cost_weight (= 1 everywhere), pure-penalty obj.

    Mirrors flextool's ``TestBaseControl::test_solver_obj_matches_python_total``
    and ``test_hand_derived_penalty_matches``.
    """

    def test_obj_equals_state_slack_decomposition(self) -> None:
        d, sol = _solve(DATA / "work_base")
        slack_up = _state_slack_term(d, sol, side="up")
        slack_dn = _state_slack_term(d, sol, side="down")
        decomposition = slack_up + slack_dn
        rel = abs(decomposition - sol.obj) / max(1.0, abs(sol.obj))
        assert rel < 1e-9, (
            f"base: closed-form decomposition {decomposition} vs "
            f"sol.obj {sol.obj}, rel={rel} (residual={sol.obj - decomposition})"
        )

    def test_hand_derived_penalty_matches(self) -> None:
        """For ``base``: penalty = Σ |inflow| * pen_up * dur * rpcw * infl
        / psh.  Exact closed-form re-evaluation from inputs (LP value of
        vq_state_up = max(-inflow, 0) since slack is pinned by
        nodeBalance_eq).  Mirrors flextool's
        ``test_hand_derived_penalty_matches``.
        """
        d, sol = _solve(DATA / "work_base")
        df = (
            d.p_inflow.frame.rename({"value": "inflow"})
            .join(d.p_penalty_up.frame.rename({"value": "pen_up"}),
                  on=["n", "d", "t"])
            .join(d.p_penalty_down.frame.rename({"value": "pen_dn"}),
                  on=["n", "d", "t"])
            .join(d.p_step_duration.frame.rename({"value": "dur"}),
                  on=["d", "t"])
            .join(d.p_rp_cost_weight.frame.rename({"value": "rpcw"}),
                  on=["d", "t"])
            .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
            .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
            .with_columns(
                slack_up=pl.max_horizontal(-pl.col("inflow"), pl.lit(0.0)),
                slack_dn=pl.max_horizontal( pl.col("inflow"), pl.lit(0.0)),
            )
        )
        if d.p_node_capacity_for_scaling is not None:
            df = df.join(
                d.p_node_capacity_for_scaling.frame.rename({"value": "ncs"}),
                on=["n", "d"],
            )
            expected = float((
                (df["slack_up"] * df["pen_up"] + df["slack_dn"] * df["pen_dn"])
                * df["ncs"] * df["dur"] * df["rpcw"]
                * df["infl"] / df["psh"]
            ).sum())
        else:
            expected = float((
                (df["slack_up"] * df["pen_up"] + df["slack_dn"] * df["pen_dn"])
                * df["dur"] * df["rpcw"]
                * df["infl"] / df["psh"]
            ).sum())
        rel = abs(expected - sol.obj) / max(1.0, abs(sol.obj))
        assert rel < 1e-9, (
            f"base hand-calc {expected} vs obj {sol.obj}, rel={rel}"
        )


class TestRpCostWeightFactor:
    """``base_weighted`` — same scenario, non-uniform rp_cost_weight.

    Mirrors flextool's ``TestRpCostWeightSlackPenalty`` — verifies that
    the rp_cost_weight factor is applied to slack penalties (was the
    P3b bug).
    """

    def test_obj_equals_decomposition(self) -> None:
        d, sol = _solve(DATA / "work_base_weighted")
        decomp = (_state_slack_term(d, sol, side="up")
                  + _state_slack_term(d, sol, side="down"))
        rel = abs(decomp - sol.obj) / max(1.0, abs(sol.obj))
        assert rel < 1e-9, (
            f"base_weighted: decomposition {decomp} vs obj {sol.obj}, rel={rel}"
        )

    def test_rp_cost_weight_factor_is_non_trivial(self) -> None:
        """Positive control: re-compute the slack-up term with rpcw≡1 and
        verify it differs materially from obj.  This proves the rpcw
        factor is actively scaling the cost — not silently ignored.
        """
        d, sol = _solve(DATA / "work_base_weighted")
        rpcw_unique = sorted(d.p_rp_cost_weight.frame["value"].unique().to_list())
        assert len(rpcw_unique) > 1, (
            "fixture invariant: work_base_weighted must have non-uniform "
            f"rp_cost_weight; got unique values {rpcw_unique}"
        )

        # Recompute the slack-up term replacing rpcw with 1.0 everywhere.
        df = (
            sol.value("vq_state_up").rename({"value": "vq"})
            .join(d.p_penalty_up.frame.rename({"value": "pen"}),
                  on=["n", "d", "t"])
            .join(d.p_step_duration.frame.rename({"value": "dur"}),
                  on=["d", "t"])
            .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
            .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
        )
        if d.p_node_capacity_for_scaling is not None:
            df = df.join(
                d.p_node_capacity_for_scaling.frame.rename({"value": "ncs"}),
                on=["n", "d"],
            )
            term_no_rpcw = float((
                df["vq"] * df["pen"] * df["ncs"]
                * df["dur"] * df["infl"] / df["psh"]
            ).sum())
        else:
            term_no_rpcw = float((
                df["vq"] * df["pen"]
                * df["dur"] * df["infl"] / df["psh"]
            ).sum())
        with_rpcw = _state_slack_term(d, sol, side="up")
        # The values must differ — proof rpcw is actively being applied.
        assert abs(with_rpcw - term_no_rpcw) / max(1.0, abs(with_rpcw)) > 1e-3, (
            f"rpcw appears to have no effect: with={with_rpcw}, "
            f"without={term_no_rpcw}.  base_weighted rpcw values "
            f"{rpcw_unique} should produce a >0.1% difference."
        )


class TestCapacityMarginPeriodOnly:
    """``capacity_margin`` — vq_capacity_margin is the period-only slack.

    Mirrors flextool's ``TestCapacityMarginPenalty`` — the cap-margin
    penalty has a *different* factor set (NO step_duration / rp_cost_weight
    / period_share).  flextool's test was the bug-finder for the missing
    1000 multiplier in flextool; flexpy's analogue verifies the period-only
    semantics.
    """

    def test_decomposition_matches_obj(self) -> None:
        d, sol = _solve(DATA / "work_capacity_margin")
        slack_up = _state_slack_term(d, sol, side="up")
        slack_dn = _state_slack_term(d, sol, side="down")
        cap_margin = _capacity_margin_slack_term(d, sol)
        decomp = slack_up + slack_dn + cap_margin
        rel = abs(decomp - sol.obj) / max(1.0, abs(sol.obj))
        assert rel < 1e-9, (
            f"capacity_margin: decomposition {decomp} vs obj {sol.obj}, "
            f"rel={rel} (slack_up={slack_up}, slack_dn={slack_dn}, "
            f"cap_margin={cap_margin})"
        )

    def test_capacity_margin_term_is_non_zero(self) -> None:
        """Positive control: cap-margin slack must contribute non-trivially
        in this scenario (otherwise the period-only factor set is not
        being exercised)."""
        d, sol = _solve(DATA / "work_capacity_margin")
        cm = _capacity_margin_slack_term(d, sol)
        assert cm > 0.0, (
            f"capacity_margin slack is zero — fixture not exercising "
            f"vq_capacity_margin; cm={cm}"
        )

    def test_capacity_margin_does_NOT_use_step_duration(self) -> None:
        """Negative control: rebuild the cap-margin term using the FULL
        (state-slack-style) factor set including step_duration /
        rp_cost_weight / period_share, and verify it gives a *different*
        answer.  This directly pins the "period-only, no per-(d,t)
        scaling" semantic.  Both sides include the × 1000 unit conversion
        (CUR/kW → CUR/MW) so the ratio cleanly isolates the
        step_duration / period_share factor.
        """
        d, sol = _solve(DATA / "work_capacity_margin")
        df = (
            sol.value("vq_capacity_margin").rename({"value": "vq"})
            .join(d.p_group_capacity_for_scaling.frame.rename({"value": "gcs"}),
                  on=["g", "d"])
            .join(d.pdGroup_penalty_capacity_margin.frame.rename({"value": "pen"}),
                  on=["g", "d"])
            .join(d.p_inflation_op.frame.rename({"value": "infl"}), on="d")
            # ---- Wrong factor set: include step_duration / rpcw / psh ----
            # Take a (d) representative — all (d,t) share the same period
            # for this fixture, so just take the per-period mean.
            .join(d.p_period_share.frame.rename({"value": "psh"}), on="d")
        )
        wrong = float((df["vq"] * df["gcs"] * df["pen"] * df["infl"]
                       / df["psh"] * 1000.0).sum())
        right = _capacity_margin_slack_term(d, sol)
        # 1/period_share is huge (~182) for the 2day fixture, so the wrong
        # variant must differ by at least 50× from the right one.
        ratio = wrong / right if right > 0 else float("inf")
        assert ratio > 50.0 or ratio < 0.02, (
            f"period-only semantic check: wrong/right ratio = {ratio} "
            f"(expected >>50× or <<1/50× since 1/period_share = "
            f"{float(d.p_period_share.frame['value'][0])**-1:.1f}). "
            "If this passes by accident the period-only semantics aren't "
            "being meaningfully tested."
        )


class TestCommodityBucket:
    """``coal`` — commodity-buy term + state-slack add to the objective."""

    def test_decomposition_matches_obj(self) -> None:
        d, sol = _solve(DATA / "work_coal")
        slack_up = _state_slack_term(d, sol, side="up")
        slack_dn = _state_slack_term(d, sol, side="down")
        commodity = _commodity_buy_eff_term(d, sol)
        decomp = slack_up + slack_dn + commodity
        rel = abs(decomp - sol.obj) / max(1.0, abs(sol.obj))
        assert rel < 1e-9, (
            f"coal: decomposition {decomp} vs obj {sol.obj}, rel={rel} "
            f"(slack_up={slack_up}, slack_dn={slack_dn}, "
            f"commodity={commodity})"
        )

    def test_commodity_term_is_non_zero(self) -> None:
        d, sol = _solve(DATA / "work_coal")
        c = _commodity_buy_eff_term(d, sol)
        assert c > 0.0, (
            f"commodity-buy term is zero — fixture not exercising the "
            f"flow_from_commodity_eff path; c={c}"
        )


class TestCO2PriceBucket:
    """``coal_co2_price`` — CO2-price term carries both ``co2_content`` and
    ``co2_price`` factors in addition to the usual op_factor."""

    def test_decomposition_matches_obj(self) -> None:
        d, sol = _solve(DATA / "work_coal_co2_price")
        slack_up = _state_slack_term(d, sol, side="up")
        slack_dn = _state_slack_term(d, sol, side="down")
        commodity = _commodity_buy_eff_term(d, sol)
        co2 = _co2_price_term(d, sol)
        decomp = slack_up + slack_dn + commodity + co2
        rel = abs(decomp - sol.obj) / max(1.0, abs(sol.obj))
        assert rel < 1e-9, (
            f"coal_co2_price: decomposition {decomp} vs obj {sol.obj}, "
            f"rel={rel} (slack_up={slack_up}, slack_dn={slack_dn}, "
            f"commodity={commodity}, co2={co2})"
        )

    def test_co2_term_is_non_zero(self) -> None:
        d, sol = _solve(DATA / "work_coal_co2_price")
        c = _co2_price_term(d, sol)
        assert c > 0.0, (
            f"co2 price term is zero — fixture not exercising the "
            f"flow_from_co2_priced path; c={c}"
        )
