"""Storage helpers that several storage families read: the states an upper-level solve fixes, the start
decisions of the solve, the storage_roll_end rows and the storage notices."""

import polars as pl
from flextool.engine.model import _extend
from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.model._common import _has_rows
from flextool.engine.model.core._context import _Build


def _nested_nodes(b: _Build, *, with_price: bool) -> "pl.DataFrame | None":
    """``(n,)`` Utf8: the nodes an upper-level solve fixes at some step of this solve (_nested_fixed_end)."""
    steps = _nested_fixed_end(b, with_price=with_price)
    return None if steps is None else steps.select("n").unique()


def _start_decision_periods(b: _Build) -> dict[str, set[str]]:
    """``{node: {periods}}``: the periods in which this solve decides storage capacity of the node: an investment
    (nd_invest_set) or a life extension (``_extend.options``: the extended cohort starts to operate there).  A start
    in such a period is on capacity the solve decides, so it is not an endowment (fix_start_energy and
    node_in_service count the extended capacity too)."""
    d = b.d
    out: dict[str, set[str]] = {}
    frames = [d.nd_invest_set] if b.has_invest_n and _has_rows(d.nd_invest_set) else []
    extend = _extend.options(d, "n")
    if extend is not None:
        frames.append(extend)
    for frame in frames:
        for n, p in frame.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).unique().iter_rows():
            out.setdefault(n, set()).add(p)
    return out


def _invest_at_start_nodes(b: _Build, nodes: "pl.DataFrame | None", periods: list[str]) -> set[str]:
    """The ``nodes`` (n) whose capacity this solve decides (invests in or extends, :func:`_start_decision_periods`)
    during one of ``periods`` (the periods the chain or the solve starts in): their capacity at the start, so also
    a fix_start start, is a decision of the solve."""
    if nodes is None:
        return set()
    names = set(str(n) for n in nodes["n"].to_list())
    wanted = set(periods)
    return {n for n, ps in _start_decision_periods(b).items() if n in names and ps & wanted}


def _fix_quantity_target(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t, value)``: the storage state (MWh) an upper-level solve fixes at the last step of each last
    period (node_balance_fix_quantity_eq_lower); None when the solve has no such fix."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.n_fix_storage_quantity)
            and _has_rows(d.ndt_fix_storage_quantity)
            and _has_rows(d.dtt_timeline_matching)
            and d.period_branch is not None
            and d.period_last is not None
            and d.p_fix_storage_quantity is not None
            and d.p_state_unitsize is not None
            and d.nodeState_last_dt is not None):
        return None
    target = (d.n_fix_storage_quantity
              .join(d.p_fix_storage_quantity.frame.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"}),
                    on="n", how="inner")
              .join(d.period_branch, on="d_upper", how="inner")
              .join(d.dtt_timeline_matching, on=["d", "t_upper"], how="inner")
              .join(d.period_last, on="d", how="inner")
              .join(d.nodeState_last_dt, on=["n", "d", "t"], how="inner")
              # Fixed summation order: the joins above leave row order unspecified.
              .sort(["n", "d", "t", "d_upper", "t_upper"])
              .group_by(["n", "d", "t"], maintain_order=True)
              .agg(pl.col("value").sum())
              .select("n", "d", "t", "value"))
    target = _restated(d, target, "fix_quantity")
    return target if target.height > 0 else None


def _restated(d, target: pl.DataFrame, kind: str) -> pl.DataFrame:
    """``target`` (n, d, t, value) with the value of each node that ``storage_roll_end`` restates by ``kind``
    (_run_model._nested_storage._nested_end_rows: the upper level interpolated inside an upper step, the cumulative
    fix_usage cap)."""
    rows = _roll_end_rows(d, (kind,))
    if rows is None or target.height == 0:
        return target
    over = rows.select(pl.col("n").cast(pl.Utf8).alias("_n"), pl.col("value").alias("_v"))
    return (target.with_columns(_n=pl.col("n").cast(pl.Utf8))
            .join(over, on="_n", how="left")
            .with_columns(value=pl.coalesce("_v", "value"))
            .drop("_n", "_v"))


def _fix_usage_target(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t, value)``: the storage usage (MWh) an upper-level solve caps over each last period, at the period's
    last step (node_storage_usage_fix_le); None when the solve has no such cap."""
    d = b.d
    if not (b.has_proc
            and b.has_storage
            and _has_rows(d.n_fix_storage_usage)
            and _has_rows(d.ndt_fix_storage_usage)
            and _has_rows(d.dtt_timeline_matching)
            and d.period_branch is not None
            and d.period_last is not None
            and d.p_fix_storage_usage is not None
            and d.nodeState_last_dt is not None):
        return None
    target = (d.n_fix_storage_usage
              .join(d.p_fix_storage_usage.frame.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"}),
                    on="n", how="inner")
              .join(d.period_branch, on="d_upper", how="inner")
              .join(d.dtt_timeline_matching.select("d", "t_upper").unique(), on=["d", "t_upper"], how="inner")
              .join(d.period_last, on="d", how="inner")
              # On (n, d) only: every window of the period sums onto its last step.
              .join(d.nodeState_last_dt, on=["n", "d"], how="inner")
              # Fixed summation order: the joins above leave row order
              # hash-dependent, which changes the float sum in the last bit.
              .sort(["n", "d", "t", "d_upper", "t_upper"])
              .group_by(["n", "d", "t"], maintain_order=True)
              .agg(pl.col("value").sum())
              .select("n", "d", "t", "value"))
    target = _restated(d, target, "usage_cap")
    target = _with_period_caps(d, target)
    return target if target.height > 0 else None


def _with_period_caps(d, target: pl.DataFrame) -> pl.DataFrame:
    """``target`` (n, d, t, value) with the ``usage_cap_period`` rows of ``storage_roll_end``: the cumulative
    fix_usage cap of a period of a roll before its horizon's last period, at the period's last step in the roll
    (_run_model._nested_storage._nested_end_rows; a period already in ``target`` keeps its row)."""
    sre = getattr(d, "storage_roll_end", None)
    if not _has_rows(sre) or "period" not in sre.columns:
        return target
    extra = (sre.filter((pl.col("kind") == "usage_cap_period") & pl.col("period").is_not_null())
             .select(pl.col("n").cast(pl.Utf8), pl.col("period").cast(pl.Utf8).alias("d"), "value"))
    if extra.height == 0:
        return target
    fixed = d.n_fix_storage_usage.select(pl.col("n").cast(pl.Utf8)).unique()
    last = d.nodeState_last_dt.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8),
                                      pl.col("t").cast(pl.Utf8))
    extra = (extra.join(fixed, on="n", how="semi").join(last, on=["n", "d"], how="inner")
             .join(target.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)), on=["n", "d"], how="anti")
             .select("n", "d", "t", "value").sort("n", "d", "t"))
    if extra.height == 0:
        return target
    schema = target.schema
    extra = extra.with_columns([pl.col(c).cast(schema[c], strict=False) for c in ("n", "d", "t")]
                               + [pl.col("value").cast(schema["value"])])
    return pl.concat([target, extra], how="vertical")


def _fix_price_steps(b: _Build) -> "pl.DataFrame | None":
    """``(n, d, t)``: the last step of each last period that an upper-level solve prices for a nested fix_price (the
    storage reference price term values the state there); None when the solve has no such step."""
    d = b.d
    if not (b.has_storage
            and _has_rows(d.ndt_fix_storage_price)
            and _has_rows(d.dtt_timeline_matching)
            and d.period_branch is not None
            and d.period_last is not None
            and d.nodeState_last_dt is not None):
        return None
    steps = (d.ndt_fix_storage_price.pipe(rename_to_axis, {"d": "d_upper", "t": "t_upper"})
             .join(d.period_branch, on="d_upper", how="inner")
             .join(d.dtt_timeline_matching, on=["d", "t_upper"], how="inner")
             .join(d.period_last, on="d", how="inner")
             .join(d.nodeState_last_dt, on=["n", "d", "t"], how="inner")
             .select("n", "d", "t").unique())
    return steps if steps.height > 0 else None


def _nested_fixed_end(b: _Build, *, with_price: bool) -> "pl.DataFrame | None":
    """``(n, d, t)``, all Utf8: the steps at which an upper-level solve fixes a nested storage node, i.e. the end of a
    last period: node_balance_fix_quantity_eq_lower (fix_quantity), node_storage_usage_fix_le (fix_usage) and, with
    ``with_price``, the nested fix_price steps.  None when there are none."""
    parts = [_fix_quantity_target(b), _fix_usage_target(b)]
    if with_price:
        parts.append(_fix_price_steps(b))
    frames = [f.select(pl.col("n").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
              for f in parts if f is not None]
    if not frames:
        return None
    return pl.concat(frames).unique().sort("n", "d", "t")


def _without_steps(over: pl.DataFrame, steps: "pl.DataFrame | None") -> pl.DataFrame:
    """``over`` (n, d, t) without the rows in ``steps`` (all-Utf8 (n, d, t)), keeping ``over``'s dtypes and order."""
    if steps is None:
        return over
    utf = steps.rename({"n": "_n", "d": "_d", "t": "_t"})
    return (over.with_columns(_n=pl.col("n").cast(pl.Utf8), _d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(utf, on=["_n", "_d", "_t"], how="anti")
            .drop("_n", "_d", "_t"))


def _roll_end_rows(d, kinds) -> "pl.DataFrame | None":
    """``(n, value)`` of the storage_roll_end rows of the ``kinds``
    (_run_model._rolling_storage._anchor_cyclic_in_rolling); None
    when there are none."""
    sre = getattr(d, "storage_roll_end", None)
    if not _has_rows(sre):
        return None
    out = sre.filter(pl.col("kind").is_in(list(kinds))).select("n", "value")
    return out if out.height > 0 else None


def _notice_nodes(change_id: str, nodes, b: "_Build | None" = None) -> None:
    """The behaviour-change notice of ``change_id`` for the nodes of ``nodes`` (a frame with n, or names): one
    printed line per change and complete solve of ``b`` (how many nodes and the first; owner rule), the names in the
    log at debug level, however many rows of the build name the change."""
    if nodes is None:
        return
    names = nodes["n"].to_list() if isinstance(nodes, pl.DataFrame) else list(nodes)
    if names:
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice(change_id, "nodes", sorted(set(str(n) for n in names)),
                                  solve=(b.solve_name if b is not None else None) or "")


def _warn_reference_below_start(b: _Build, nodes: set) -> None:
    """Warn about the nodes whose use_reference_value end lies below their start share while the solve can invest
    in their storage in the period they start in (owner decision, Phase 5; before a configuration error): as for
    fix_start_end, the matching end stays, so storage built must return its start energy, which caps the
    investment.  One printed line per solve."""
    if not nodes:
        return
    from flextool import run_warnings
    run_warnings.warn_grouped("W-0538", sorted(nodes), solve=b.solve_name)
