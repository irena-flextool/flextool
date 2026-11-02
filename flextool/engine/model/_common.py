"""Small helpers shared by ``model/core.py`` and the model feature modules.

Frame utilities (row checks, Utf8 join keys, renamed variable views), the
arc classes of the capacity constraints, and the densified per-entity
coefficient factors.  Nothing here emits rows or columns.
"""

import polars as pl
from polar_high import Param
from polar_high.engine import Var

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.input._param_shapes import promote_param_to_dt
from flextool.engine.preprocess.params._branch import inherit_anchor_period


def _has_rows(frame) -> bool:
    """True when ``frame`` is populated and non-empty."""
    return frame is not None and frame.height > 0


def dispatch_periods(d) -> pl.DataFrame:
    """``(d,)``: the periods that have dispatch, i.e. the periods of ``dt``.

    ``period_in_use_set`` also holds the period labels of a stochastic fan (bookkeeping only: investment, years,
    lineage), which have no (d, t) rows; rows over operational sums take their periods from here.
    """
    return d.dt.select("d").unique(maintain_order=True)


def _var_view(v: Var, suffix: str, rename: dict, *, to_axis: bool = False) -> Var:
    """``v`` with renamed dims and the same columns.

    Used to index one variable twice in a term (e.g. ``d → d_invest`` to sum
    investments of earlier periods, ``t → last_step`` to pick a step).
    ``to_axis`` renames through :func:`rename_to_axis` (axis-typed column).
    """
    frame = (v.frame.pipe(rename_to_axis, rename) if to_axis
             else v.frame.rename(rename))
    return Var(name=v.name + suffix,
               dims=tuple(rename.get(x, x) for x in v.dims),
               frame=frame, lower=v.lower, upper=v.upper)


def _entity_rows(frame: pl.DataFrame, ent_col: str, members: pl.DataFrame,
                 axis: str) -> pl.DataFrame:
    """Rows of ``frame`` whose ``ent_col`` is in ``members[axis]``, with
    ``ent_col`` renamed to ``axis``.

    ``frame`` is keyed on the entity-union (``e``) or process (``p``) axis
    while ``members`` uses the process or node axis.  Their Enum
    vocabularies differ and polars cannot compare them, so ``members`` is
    cast to ``frame``'s dtype first (every member is in the larger
    vocabulary).
    """
    dtype = frame.schema[ent_col]
    keys = (members.select(pl.col(axis).cast(dtype, strict=False).alias(ent_col))
            .unique())
    return (frame.join(keys, on=ent_col, how="semi")
            .pipe(rename_to_axis, {ent_col: axis}))


def _entities_of_kind(d, e_set: pl.DataFrame, axis: str) -> pl.DataFrame:
    """Rows of ``e_set`` (keyed on e) whose entity is a process (axis p) or a storage node (axis n)."""
    members, col = (d.process_source_sink, "p") if axis == "p" else (d.nodeState, "n")
    if not _has_rows(members):
        return e_set.head(0)
    return e_set.filter(pl.col("e").is_in(members[col].unique()))


def chain_prior(d, kind: str) -> "pl.DataFrame | None":
    """``[e, value]``: capacity realized (``kind="invest"``) or retired (``kind="divest"``) by all
    earlier solves of a solve chain, per entity.  ``None`` when the carrier Param is absent (a
    single solve, or the first solve of a chain, with nothing yet realized/retired).

    ``invest`` is ``d.p_entity_invested`` — every investment of the chain, retired or not (Q10).
    ``divest`` is ``d.p_entity_divested`` — capacity retired at realized, non-branch periods of
    the chain only (the handoff carrier itself already excludes look-ahead and
    alternative-branch periods, see ``flex_data.build_handoff_from_solution``).
    """
    param = d.p_entity_invested if kind == "invest" else d.p_entity_divested
    return None if param is None else param.frame


def minus_prior(cap: pl.DataFrame, key: str, prior: "pl.DataFrame | None") -> pl.DataFrame:
    """``cap`` (indexed on ``key``, with a float ``"value"`` column) with
    ``value = max(0, value − prior[key])``, by a left join on ``key``.

    ``prior`` has two columns: an entity/group key (named ``"e"`` from :func:`chain_prior`, or
    ``"g"`` from :func:`group_prior`) and ``"value"``; it is renamed to ``key`` and its dtype cast
    to ``cap[key]``'s dtype (``strict=False``, as :func:`_entity_rows` does — the wider
    entity-union vocabulary casts down to the narrower axis one) before the join.  Rows of
    ``cap`` with no matching ``prior`` row are unchanged (prior = 0).  ``None`` / empty ``prior``
    is a no-op.  Vectorized (no ``map_elements``), per the vectorized-outputs rule.
    """
    if prior is None or prior.height == 0:
        return cap
    prior_key_col = next(c for c in prior.columns if c != "value")
    if prior_key_col != key:
        prior = prior.rename({prior_key_col: key})
    dtype = cap.schema[key]
    if prior.schema[key] != dtype:
        prior = prior.with_columns(pl.col(key).cast(dtype, strict=False))
    return (cap.join(prior, on=key, how="left", suffix="_prior")
            .with_columns(value=pl.col("value") - pl.col("value_prior").fill_null(0.0))
            .with_columns(value=pl.when(pl.col("value") < 0).then(0.0).otherwise(pl.col("value")))
            .drop("value_prior"))


def group_prior(group_entity: pl.DataFrame, prior: "pl.DataFrame | None", d, axis: str) -> pl.DataFrame:
    """``[g, value] = Σ_{e ∈ g, e of axis kind} prior[e]``.

    Members are every entity of ``group_entity`` that is a process (``axis="p"``) or storage node
    (``axis="n"``) — **not** restricted to entities with a variable in this solve: their earlier
    investment/divestment still counts towards the group's chain-cumulative total.
    Returns an empty ``[g, value]`` frame when ``prior`` is ``None``/empty or no member of
    the axis kind has a prior entry.
    """
    out_schema = {"g": group_entity.schema["g"], "value": pl.Float64}
    if prior is None or prior.height == 0:
        return pl.DataFrame(schema=out_schema)
    members = _entities_of_kind(d, group_entity, axis)
    if members.height == 0:
        return pl.DataFrame(schema=out_schema)
    dtype = members.schema["e"]
    if prior.schema["e"] != dtype:
        prior = prior.with_columns(pl.col("e").cast(dtype, strict=False))
    out = (members.join(prior, on="e", how="inner")
           .group_by("g").agg(pl.col("value").sum()))
    return out if out.height > 0 else pl.DataFrame(schema=out_schema)


def leaf_rows(terms: pl.DataFrame, key: str, var: Var, d_leaf: "pl.DataFrame | None"
             ) -> tuple[bool, pl.DataFrame, pl.DataFrame]:
    """``(is_path, rows, mask)`` for the ``Σ_d`` block shared by every ``*_total`` family.

    ``terms`` is the actual support of the Σ term — every row has a ``key`` column (``"p"``/
    ``"n"`` for an entity total, ``"g"`` for a group total) and a ``"d"`` column, already
    restricted to rows that have the variable (an entity caller intersects its ``over`` with
    ``var.frame.select(key, "d")``; a group caller additionally joins in the group's members —
    callers apply the #43 restriction this way before calling this).  With one scenario leaf (or
    no ``d_leaf``) returns ``(False, rows, terms)`` where ``rows = terms.select(key).unique()`` —
    the legacy shape: ``Sum(Where(... , terms), over=(...,"d"))`` sums every period/member
    ``terms`` has for each ``key``.

    With several leaves (recourse) returns ``(True, rows, mask)``: ``mask`` is ``terms`` joined to
    the scenario tree's ``(d, leaf)`` pairs (``d_leaf``), and ``rows = mask.select(key,
    "leaf").unique()``.  A ``(key, leaf)`` combination with no ``d`` (that key has no
    variable/member in that leaf) therefore gets no row at all, instead of a constants-only row
    that would be infeasible for a ``>=`` family and a no-op for ``<=``.  Leaves of one key with
    the same ``d`` support (the period labels' here-and-now investment is on every branch path)
    collapse to the first sorted leaf; when every key then has one row, the legacy shape is
    returned.
    """
    if d_leaf is None or getattr(d_leaf, "height", 0) == 0:
        return False, terms.select(key).unique(), terms
    d_dtype = var.frame.schema["d"]
    leaves_all = (d_leaf if d_leaf.schema["d"] == d_dtype
                  else d_leaf.with_columns(pl.col("d").cast(d_dtype, strict=False)))
    leaves = leaves_all.join(var.frame.select("d").unique(), on="d", how="semi")
    if leaves.height == 0 or leaves["leaf"].n_unique() <= 1:
        return False, terms.select(key).unique(), terms
    mask = terms.join(leaves_all, on="d", how="inner")
    # Leaves of one key with the SAME d-support give the same row (e.g. a here-and-now investment at a period
    # label, on every branch path): keep the first sorted leaf.  When that leaves one row per key, the rows are the
    # legacy single-path shape (each key's row sums its whole support).
    support = (mask.group_by(key, "leaf").agg(pl.col("d").cast(pl.Utf8).sort().alias("__support"))
               .sort(key, "leaf").unique(subset=[key, "__support"], keep="first", maintain_order=True))
    mask = mask.join(support.select(key, "leaf"), on=[key, "leaf"], how="semi")
    if support.group_by(key).len()["len"].max() <= 1:
        return False, terms.select(key).unique(), terms
    rows = mask.select(key, "leaf").unique()
    return True, rows, mask


def check_max_cap(over: pl.DataFrame, cap: "pl.DataFrame | None", key: str, has_period: bool,
                  *, param_name: str) -> None:
    """Raise ``FlexToolConfigError`` (FA-12) when some row of ``over`` — the rows that would get
    a ``<=`` cap row, already restricted to entities/groups that have the variable (the #43
    scope) — has no matching value in ``cap`` (the sparse Param supplying the
    RHS).

    A row with no value there does not mean "unlimited": once some OTHER entity/group sets
    ``param_name``, the RHS column exists, and a row absent from it reads as ``0`` — silently
    forbidding investment/retirement instead of leaving it unlimited.  A ``cap`` that is entirely
    ``None`` (nobody sets ``param_name`` at all) is unaffected — callers already skip the whole
    family in that case (Q2: the outcome stays "no rows / unlimited").

    ``over`` and ``cap`` are joined on ``key`` (plus ``"d"`` when ``has_period``).
    """
    if cap is None:
        return
    join_keys = [key, "d"] if has_period else [key]
    missing = (over.select(join_keys).unique()
               .join(cap.select(join_keys).unique(), on=join_keys, how="anti"))
    if missing.height == 0:
        return
    from flextool.engine.input._solve_state import FlexToolConfigError
    if has_period:
        by_key: dict[str, list[str]] = {}
        for row in missing.sort(join_keys).iter_rows(named=True):
            by_key.setdefault(str(row[key]), []).append(str(row["d"]))
        detail = "; ".join(f"{k} (period(s) {', '.join(v)})" for k, v in sorted(by_key.items()))
    else:
        detail = ", ".join(sorted({str(x) for x in missing[key].to_list()}))
    raise FlexToolConfigError(
        f"{detail}: an investment/retirement cap applies (another entity/group sets "
        f"{param_name}) but no {param_name} value is given here; a missing value reads as 0, "
        f"silently forbidding investment/retirement. Give {param_name} a value explicitly "
        f"(0 forbids it) or give the capping entities/groups their own invest/retire method so "
        f"this one is not swept in by a shared column."
    )


def _utf8_keys(lf: "pl.LazyFrame", keys) -> "pl.LazyFrame":
    """Add ``_<k>`` Utf8 copies of ``keys`` — join keys that bridge Enum
    vocabulary edges between frames built by different producers."""
    return lf.with_columns([pl.col(k).cast(pl.Utf8).alias(f"_{k}")
                            for k in keys])


def _indirect_sum_arcs(d) -> "pl.DataFrame | None":
    """Output arcs ``(p, source, sink)`` summed by ``maxOutputSum``.

    ``process_output_flows`` (``source == p``, ``p ∈ process_indirect``,
    ``conversion_flow_coeff ≠ 0`` — zero-coefficient outputs are removed
    from every capacity constraint) restricted to arcs that actually
    carry a ``v_flow`` column (``process_source_sink``) and to processes
    still in ``process_indirect`` — region-filtered / Benders-master
    FlexData narrow ``process_indirect`` and ``process_source_sink`` but
    not necessarily ``process_output_flows``.  ``None`` when empty.
    """
    out_arcs = getattr(d, "process_output_flows", None)
    indirect = getattr(d, "process_indirect", None)
    pss = getattr(d, "process_source_sink", None)
    if (out_arcs is None or out_arcs.height == 0 or indirect is None
            or indirect.height == 0 or pss is None or pss.height == 0):
        return None
    arcs = (out_arcs.select("p", "source", "sink")
            .join(indirect.select("p").unique(), on="p", how="semi")
            .join(pss.select("p", "source", "sink"),
                  on=["p", "source", "sink"], how="semi")
            .unique(maintain_order=True))
    return arcs if arcs.height > 0 else None


def _capped_arcs(d, arcs: "pl.DataFrame") -> "pl.DataFrame":
    """``arcs`` (``p, source, sink``) restricted to the CAPPED class of
    ``model._max_flow_rhs``: drops indirect-process arcs that are not output
    arcs (fuel / input arcs) and ``conversion_flow_coeff = 0`` (uncapped)
    edges."""
    keys = ["p", "source", "sink"]
    out = arcs
    indirect = getattr(d, "process_indirect", None)
    if indirect is not None and indirect.height > 0:
        out_arcs = getattr(d, "process_output_flows", None)
        indir_arcs = out.join(indirect.select("p").unique(), on="p",
                              how="semi")
        if out_arcs is not None and out_arcs.height > 0:
            indir_arcs = indir_arcs.join(out_arcs.select(keys), on=keys,
                                         how="anti")
        out = out.join(indir_arcs.select(keys), on=keys, how="anti")
    return _drop_uncapped(d, out)


def _drop_uncapped(d, arcs: "pl.DataFrame") -> "pl.DataFrame":
    """``arcs`` (any frame carrying ``p, source, sink``) without the
    ``conversion_flow_coeff = 0`` edges (``process_source_sink_uncapped``),
    which are outside every per-edge capacity / ramp / min-load
    constraint.  Columns and dtypes of ``arcs`` are preserved."""
    unc = getattr(d, "process_source_sink_uncapped", None)
    if unc is None or unc.height == 0 or arcs.height == 0:
        return arcs
    keys = ["p", "source", "sink"]
    return (_utf8_keys(arcs.lazy(), keys)
            .join(_utf8_keys(unc.lazy(), keys)
                  .select([f"_{k}" for k in keys]),
                  on=[f"_{k}" for k in keys], how="anti")
            .select(arcs.columns).collect())


def _indirect_input_cap(d, arcs: "pl.DataFrame") -> "pl.DataFrame":
    """``p_indirect_input_cap`` over ``arcs`` (indirect units' input arcs ``p, source, sink``) × dt, as a
    frame ``(p, source, sink, d, t, value)`` with the dtypes of ``arcs`` / ``d.dt``.

    The cap exists for every capped input arc (``model._max_flow_rhs`` raises without it).  The
    ``conversion_flow_coeff = 0`` inputs (``process_source_sink_uncapped``) have no cap and are outside every
    per-arc capacity row; callers drop them (``_drop_uncapped``) or treat them explicitly before calling.
    Raises when a cap cell is missing (the limit would be silently dropped or replaced by a guess).
    """
    keys = ["p", "source", "sink"]
    over = arcs.select(keys).unique(maintain_order=True).join(d.dt.select("d", "t"), how="cross")
    in_cap = getattr(d, "p_indirect_input_cap", None)
    if in_cap is None:
        if over.height == 0:
            return over.with_columns(pl.lit(None, pl.Float64).alias("value"))
        raise ValueError(
            "FlexData.p_indirect_input_cap is missing but indirect input arcs need it: "
            f"{arcs.select(keys).unique().head(10).to_dicts()}.")
    all_keys = keys + ["d", "t"]
    out = (_utf8_keys(over.lazy(), all_keys)
           .join(_utf8_keys(in_cap.frame.lazy(), all_keys)
                 .select([f"_{k}" for k in all_keys] + ["value"]),
                 on=[f"_{k}" for k in all_keys], how="left")
           .select(*all_keys, "value").collect())
    missing = out.filter(pl.col("value").is_null())
    if missing.height > 0:
        raise ValueError(
            "p_indirect_input_cap has no value for indirect input arc(s) "
            f"{missing.select(keys).unique().head(10).to_dicts()} — the input-arc limit would be dropped.")
    return out


def _coef_factor(pairs: "pl.DataFrame", coef: "Param", keys: list,
                 *, default: float = 1.0) -> "Param":
    """Densify a sparse per-``keys`` coefficient ``coef`` over the key-tuples
    present in ``pairs``, filling absent tuples with ``default``.

    Folding a sparse coefficient into a constraint via ``Param * Param`` (an
    inner join) would DROP any tuple missing from ``coef`` — collapsing the
    term to 0 instead of multiplying by the schema default.  This is the same
    hazard as :func:`_availability_factor`; here it covers static per-arc
    coefficients (e.g. ``capacity_min_coeff`` keyed ``(p, sink)``).  The join
    keys are compared as ``Utf8`` to bridge Enum-vocabulary edges between the
    consumer set and the coefficient Param.
    """
    base = pairs.select(keys).unique()
    base_lf = base.lazy().with_columns(
        [pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in keys])
    coef_lf = (coef.frame.lazy()
               .with_columns([pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in keys])
               .drop(keys))
    join_keys = [f"_{k}" for k in keys]
    merged = (base_lf
              .join(coef_lf, on=join_keys, how="left")
              .with_columns(pl.col("value").fill_null(default))
              .select(*keys, "value")
              .collect())
    return Param(tuple(keys), merged)


def _availability_factor(d, members: "pl.DataFrame", *,
                         axis: str = "p",
                         avail: "Param | None" = None) -> "Param":
    """``availability`` as a Param keyed ``(axis, d, t)`` that COVERS every
    entity in ``members`` — the DB-authored value where one exists, else the
    schema default ``1.0``.  ``axis`` is ``"p"`` for the process factor
    (``p_process_availability``, the default) or ``"n"`` for the node-state
    factor (pass ``avail=d.p_node_availability``).

    The availability Params are SPARSE: their ``*_from_source`` producers emit
    only DB-authored rows, expecting absent entities to fall back to the
    ``availability`` schema default (1.0).  But the LP folds the factor into
    constraint RHSs via ``Param * Param`` / ``Var * Param``, which are INNER
    joins (polar_high contract) — so an entity absent from the sparse Param would
    have its RHS row DROPPED rather than multiplied by 1.0, collapsing the
    bound to 0 and forcing the variable (``v_flow`` / ``v_state``) to 0 once
    SOME other entity authors an availability.  Densifying against the
    consumer's own entity set (left-join + ``fill_null(1.0)``) makes the
    multiply a multiplicative overlay.  Callers guard on ``avail is not None``
    (an all-1.0 factor is a no-op).
    """
    avail = avail if avail is not None else d.p_process_availability
    base = members.select(axis).unique().join(d.dt, how="cross")  # (axis, d, t)
    # availability may be authored scalar / 1d_map[period] / 1d_map[time] /
    # 2d_map, so promote to (axis, d, t) before the left-join.
    avail_lf = promote_param_to_dt(avail, d.dt)
    # The entity axis can carry a different Enum vocabulary on the consumer set
    # vs the availability Param (e.g. node_profile ``n`` vs p_node_availability
    # ``n``): compare on a Utf8 key so value-equal members match.  The
    # original-dtype axis is carried through from ``base`` for the downstream
    # multiply.
    base_lf = base.lazy().with_columns(pl.col(axis).cast(pl.Utf8).alias("_k"))
    avail_lf = (avail_lf.with_columns(pl.col(axis).cast(pl.Utf8).alias("_k"))
                        .drop(axis))
    merged = (base_lf
              .join(avail_lf, on=["_k", "d", "t"], how="left")
              .with_columns(pl.col("value").fill_null(1.0))
              .select(axis, "d", "t", "value")
              .collect())
    return Param((axis, "d", "t"), merged)


def _connection_procs(d) -> "pl.DataFrame | None":
    """``(p,)`` Utf8: the processes of ``process_source_sink`` that are not units (``process_unit``), i.e. the
    connections.  None when ``process_unit`` is None (hand-built data: the caller keeps the process-generic
    reserve coupling); an empty ``process_unit`` frame makes every process a connection."""
    unit = getattr(d, "process_unit", None)
    pss = getattr(d, "process_source_sink", None)
    if unit is None or pss is None:
        return None
    units = unit.select(pl.col("p").cast(pl.Utf8)).unique()
    return (pss.select(pl.col("p").cast(pl.Utf8)).unique(maintain_order=True)
            .join(units, on="p", how="anti"))


def _period_branch_pairs(d) -> "pl.DataFrame | None":
    """(anchor, br) pairs of every stochastic forecast branch and its anchor period; None when there are none.

    Built from ``period_branch_full`` (the full unfiltered ``period__branch.csv`` map, already on
    ``flex_data`` by model-build time — the same source ``_non_anticipativity._forecast_star`` reads for
    the forecast ties) rather than ``branch_anchor_pairs``, which needs a ``provider`` the model build
    does not have.  Self rows (``d == b``, trunks / deterministic periods) are dropped.
    """
    pbf = getattr(d, "period_branch_full", None)
    if not _has_rows(pbf):
        return None
    pairs = (pbf.select(pl.col("d").cast(pl.Utf8).alias("anchor"), pl.col("b").cast(pl.Utf8).alias("br"))
             .filter(pl.col("anchor") != pl.col("br")))
    return pairs if pairs.height > 0 else None


def branch_periods_inherit(d, frame: pl.DataFrame, group_keys: "tuple[str, ...]") -> pl.DataFrame:
    """``frame`` (period column ``d``) with its period labels' rows copied to their forecast branch periods.

    :func:`inherit_anchor_period` with :func:`_period_branch_pairs`: a forecast that has a row of its own keeps
    it; identity outside stochastic solves.  The period caps of a forecast read their period-keyed inputs (period
    share, f_d_k, the realized carry) through it, so every forecast continues from its period's realized history.
    """
    return inherit_anchor_period(frame, _period_branch_pairs(d), group_keys=group_keys)
