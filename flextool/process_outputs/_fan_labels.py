"""The period labels of a solve's stochastic fans, and the label rows of its branch-keyed investment inputs.

When the investments branch (``invest_solve_branching_method`` other than ``operations_only``) the investment
inputs of the reported costs (the discounted investment and lifetime fixed-cost lumps, the yearly annuity, the
start of operation and the operating window) are keyed by the fan's branch periods.  The reported costs carry the fan's investment at its label (the
branch-weighted amount, ``v_invest_fan``), so they read these inputs at the label: a branch period takes its
label's years and period parameters, so the rows of a fan's branches are the same as the label's.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from flextool.engine.preprocess.flex_data import FlexData


def fan_branch_labels(flex_data: "FlexData") -> dict[str, str]:
    """``{branch period: its label}`` of every stochastic fan of the solve (empty for a deterministic solve)."""
    pbf = getattr(flex_data, "period_branch_full", None)
    if pbf is None or pbf.height == 0:
        return {}
    from flextool.engine.preprocess.params._branch import fan_period_labels

    pairs = [(str(d), str(b)) for d, b in zip(pbf["d"].to_list(), pbf["b"].to_list())]
    labels = fan_period_labels(set(pairs))
    return {b: d for d, b in pairs if d in labels and b != d}


def recourse_branch_labels(flex_data: "FlexData") -> dict[str, str]:
    """:func:`fan_branch_labels` of a solve that decides investments in fan branch periods (recourse: a branch
    period in ``ed_invest_set``).  Empty for a shared fan investment (decided at the label) and for a
    deterministic solve."""
    branch_of = fan_branch_labels(flex_data)
    invest = getattr(flex_data, "ed_invest_set", None)
    if not branch_of or invest is None or invest.height == 0:
        return {}
    decided = set(invest["d"].cast(pl.Utf8).to_list())
    return branch_of if decided & set(branch_of) else {}


def with_label_rows(frame: "pl.DataFrame | None", branch_of: dict[str, str], period_cols: tuple[str, ...],
                    ) -> "pl.DataFrame | None":
    """``frame`` with each row of a branch period of ``branch_of`` (in ``period_cols``) copied to its label: the
    label key takes the branch row (the first branch row of a key), replacing a row the label already has (under
    recourse the label decides nothing; its own rows are not those of a decision).  Unchanged without branch
    rows."""
    if frame is None or frame.height == 0 or not branch_of:
        return frame
    cols = [c for c in period_cols if c in frame.columns]
    if not cols:
        return frame
    branches = list(branch_of)
    has_branch = pl.any_horizontal([pl.col(c).cast(pl.Utf8).is_in(branches) for c in cols])
    mapped = frame.filter(has_branch)
    if mapped.height == 0:
        return frame
    mapped = mapped.with_columns(pl.col(c).cast(pl.Utf8).replace(branch_of).cast(frame.schema[c]) for c in cols)
    keys = [c for c in frame.columns if c != "value"]
    mapped = mapped.unique(subset=keys, keep="first", maintain_order=True)
    return pl.concat([frame.join(mapped, on=keys, how="anti"), mapped], how="vertical")


def at_fan_labels(frame: pl.DataFrame, flex_data: "FlexData", decision: str, reports: tuple[str, ...] = (),
                  value: str = "value") -> pl.DataFrame:
    """``frame`` (Utf8 period columns) with each row whose ``decision`` period is a forecast-branch period of a fan
    moved to the fan's label, its ``value`` times the branch's ``pd_branch_weight`` (the expected amount, as the
    objective weights it), and the branch periods of the ``reports`` columns moved to their labels; rows that land
    on the same key are summed.  Unchanged for a deterministic solve.  Under recourse a fan's decisions are per
    branch: this is how the reported costs carry them at the label."""
    branch_of = fan_branch_labels(flex_data)
    if not branch_of or frame.height == 0:
        return frame
    weights = getattr(flex_data, "pd_branch_weight", None)
    weight = {}
    if weights is not None:
        from flextool.engine.preprocess.params._cohorts import as_names
        weight = dict(as_names(weights.frame.select("d", "value"), ("d",)).iter_rows())
    branch = pl.col(decision).is_in(list(branch_of))
    factor = pl.col(decision).replace_strict({b: float(weight.get(b, 1.0)) for b in branch_of}, default=1.0,
                                             return_dtype=pl.Float64)
    moved = frame.with_columns(pl.when(branch).then(pl.col(value) * factor).otherwise(pl.col(value)).alias(value),
                               *(pl.col(c).replace(branch_of) for c in (decision, *reports)))
    keys = [c for c in frame.columns if c != value]
    return moved.group_by(keys, maintain_order=True).agg(pl.col(value).sum()).select(frame.columns)
