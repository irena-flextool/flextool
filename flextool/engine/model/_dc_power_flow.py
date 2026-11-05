"""DC power flow: linearised AC flow on the connections of a DC power-flow network.

Owns the variable v_angle and the rows dc_reference_angle_eq and dc_flow_eq: the flow on each DC arc equals
the connection's susceptance times the voltage-angle difference of its end nodes.  One node per connected
component (node_reference_angle, chosen in input derivation) has its angle pinned to 0.  The reverse flow
v_flow_back of the arcs, its capacity row and its node-balance terms belong to model/core.py, which shares them
with the other method_2way_1var_off arcs; :func:`nodeBalance_back_flow_terms` builds those balance terms.

Coarse steps (``new_stepduration`` groups): the nodes of a DC network component whose every line is a coarse
process and whose nodes share one set of coarse steps (``flex_data.coarse_angle_steps``) have one angle per coarse
step (an aliased level); every other angle stays per solve step.  dc_flow_eq follows the row rule — a coarse line
between two such nodes gets one row per coarse step (``dc_flow_eq_coarse``), any other line keeps a row per solve
step and reads a coarse flow at its constant value.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Var

from flextool.engine.input._axis_enums import cast_dim
from flextool.engine.model import _coarse_flows
from flextool.engine.model._common import _has_rows

if TYPE_CHECKING:
    from flextool.engine.model.core import _Build


# ---------------------------------------------------------------------------
# Feature detection


def has_feature(d) -> bool:
    """True when at least one node is in node_dc_power_flow and one connection in connection_dc_power_flow."""
    nd = getattr(d, "node_dc_power_flow", None)
    cd = getattr(d, "connection_dc_power_flow", None)
    if nd is None or cd is None:
        return False
    return nd.height > 0 and cd.height > 0


# ---------------------------------------------------------------------------
# Loader


def load_data(inp_dir: str | Path, *, provider=None) -> dict:
    """The DC power-flow sets and susceptance, as ``field_name -> DataFrame | Param | None``.

    Fields::

        node_dc_power_flow         (n,)
        connection_dc_power_flow   (p,)
        node_reference_angle       (n,)      angle pinned to 0
        p_connection_susceptance   Param (p,)
        process_source_toSink_dc   (p, source, sink), one orientation per arc

    Every value is None when the feature is inactive (header-only frames).  Frames are read through
    ``provider``; ``inp_dir`` is accepted for the loaders' uniform call.
    """
    blank = dict(
        node_dc_power_flow       = None,
        connection_dc_power_flow = None,
        node_reference_angle     = None,
        p_connection_susceptance = None,
        process_source_toSink_dc = None,
    )

    def _frame_for(key: str) -> pl.DataFrame | None:
        """The provider's frame for ``key``; None when missing or empty."""
        if provider is None or not provider.has(key):
            return None
        df = provider.get(key)
        if df is None or df.height == 0:
            return None
        return df

    def _project_single(df: pl.DataFrame | None, col_in: str, col_out: str) -> pl.DataFrame | None:
        """``df`` reduced to ``col_in`` renamed to ``col_out``."""
        if df is None:
            return None
        if col_in in df.columns and col_in != col_out:
            df = df.rename({col_in: col_out})
        return df.select(col_out)

    nd = _project_single(_frame_for("input/node_dc_power_flow"), "node", "n")
    cd = _project_single(_frame_for("input/connection_dc_power_flow"), "process", "p")
    rd = _project_single(_frame_for("input/node_reference_angle"), "node", "n")

    pcs_raw = _frame_for("input/p_connection_susceptance")
    pcs_param = None
    if pcs_raw is not None:
        df = pcs_raw.rename(
            {c: r for c, r in [
                ("process", "p"),
                ("p_connection_susceptance", "value"),
            ] if c in pcs_raw.columns}
        ).select("p", "value")
        # The provider may carry the value as text.
        if df["value"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("value").cast(pl.Float64))
        pcs_param = Param(("p",), df)

    if nd is None and cd is None and rd is None and pcs_param is None:
        return blank

    # One orientation per arc: process_source_sink lists a two-way connection in both directions, but
    # dc_flow_eq has one row per arc.
    sst = None
    if provider is not None and provider.has("solve_data/process_source_toSink"):
        sst = provider.get("solve_data/process_source_toSink")
        if sst is not None and sst.height > 0:
            sst = sst.rename({
                c: r for c, r in [
                    ("process", "p"),
                    ("source", "source"),
                    ("sink", "sink"),
                ] if c in sst.columns
            }).select("p", "source", "sink")
        else:
            sst = None

    return dict(
        node_dc_power_flow       = nd,
        connection_dc_power_flow = cd,
        node_reference_angle     = rd,
        p_connection_susceptance = pcs_param,
        process_source_toSink_dc = sst,
    )


# ---------------------------------------------------------------------------
# Arcs


# Bound of the non-reference angles: π truncated to 8 digits, the literal the angle bounds have always used
# (a change would shift the LP's bound values).
_PI_LITERAL = 3.14159265


def dc_arcs_frame(d) -> pl.DataFrame | None:
    """(p, source, sink) of the DC arcs; None when there are none.

    A DC arc is an arc of a connection_dc_power_flow connection whose both end nodes are in
    node_dc_power_flow, in one orientation (process_source_toSink_dc; process_source_sink, which has both
    orientations, when that set is absent).  model/core.py adds these arcs to the v_flow_back arcs.
    """
    if (d.process_source_sink is None
            or d.connection_dc_power_flow is None
            or d.connection_dc_power_flow.height == 0):
        return None
    # source / sink carry the entity-union Enum, node_dc_power_flow.n the node Enum: widen n to match.
    _dc_n_e = d.node_dc_power_flow.with_columns(cast_dim(pl.col("n"), None, "e"))["n"]
    arcs_src = (getattr(d, "process_source_toSink_dc", None)
                if getattr(d, "process_source_toSink_dc", None) is not None
                else d.process_source_sink)
    if arcs_src is not d.process_source_sink:
        for col in ("p", "source", "sink"):
            target_dtype = d.process_source_sink.schema[col]
            if arcs_src.schema[col] != target_dtype:
                arcs_src = arcs_src.with_columns(pl.col(col).cast(target_dtype, strict=False))
    dc_arcs = (arcs_src
        .join(d.connection_dc_power_flow, on="p", how="inner")
        .filter(pl.col("source").is_in(_dc_n_e))
        .filter(pl.col("sink").is_in(_dc_n_e)))
    return dc_arcs if dc_arcs.height > 0 else None


# ---------------------------------------------------------------------------
# Variables


def add_variables(b: _Build) -> None:
    """Declare v_angle when the DC power-flow network is active."""
    d = b.d
    if not has_feature(d) or d.dt is None or d.dt.height == 0:
        return
    angle_idx = d.node_dc_power_flow.join(d.dt, how="cross").select("n", "d", "t")
    if angle_idx.height == 0:
        return

    # ─── VAR. ─── v_angle ────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # −π ≤ v_angle[n,d,t] ≤ π   ∀ n ∈ node_dc_power_flow, (d,t) ∈ dt
    # Voltage angle of node n, radians (π = 3.14159265).  Variable bounds are
    # scalar, so the reference nodes are pinned to 0 by dc_reference_angle_eq.
    # ─── VAR. ─── v_angle (coarse-angle nodes) ───────────────────────────────
    # −π ≤ v_angle[n,d,b] ≤ π   ∀ n ∈ coarse-angle nodes, (d,b) the coarse steps of n (b = its first fine step)
    #   coarse-angle node = a node of a DC network component whose every line is a coarse process and whose
    #   nodes all have the same coarse steps (flex_data.coarse_angle_steps)
    # One angle per coarse step, held over its fine steps (a level: b.v_angle is indexed on every fine step,
    # _coarse_flows.alias).  Every node of a component with a DC line at the solve's step keeps an angle per solve
    # step, so the fine line keeps its per-step physics.
    b.angle_steps = _angle_steps(b)
    b.v_angle = _coarse_flows.alias(
        b, b.m.add_var("v_angle", ("n", "d", "t"), _coarse_flows.domain(b, angle_idx, b.angle_steps),
                       lower=-_PI_LITERAL, upper=_PI_LITERAL),
        angle_idx, b.angle_steps)


def _angle_steps(b: _Build) -> pl.DataFrame:
    """``(n, d, b_first, t)`` of the coarse-angle nodes (``flex_data.coarse_angle_steps``); empty without any."""
    from flextool.engine.preprocess.flex_data import coarse_angle_steps

    if not b.has_coarse_proc:
        return pl.DataFrame(schema={c: pl.Utf8 for c in ("n", "d", "b_first", "t")})
    return coarse_angle_steps(b.d)


# ---------------------------------------------------------------------------
# Constraints


def add_constraints(b: _Build) -> None:
    """dc_reference_angle_eq and dc_flow_eq, when the DC power-flow network is active."""
    if not b.has_proc or b.v_angle is None:
        return
    _eq_reference_angle(b)
    _eq_dc_flow(b)


def _eq_reference_angle(b: _Build) -> None:
    """dc_reference_angle_eq: the angle of each reference node is 0."""
    d = b.d
    if not _has_rows(d.node_reference_angle):
        return
    ref_idx = d.node_reference_angle.join(d.dt, how="cross").select("n", "d", "t")
    if ref_idx.height == 0:
        return

    # ─── EQ. ─── dc_reference_angle_eq ───────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # v_angle[n,d,t] = 0   ∀ n ∈ node_reference_angle, (d,t) ∈ dt
    #   (a coarse-angle node: (d,b) its coarse steps, b = the step's first fine step)
    # Fixes the angle reference of each connected DC network.
    if b.angle_steps is not None:
        ref_idx = _coarse_flows.domain(b, ref_idx, b.angle_steps)
    b.m.add_cstr(
        "dc_reference_angle_eq",
        over      = ref_idx,
        sense     = "==",
        lhs_terms = {"angle": Where(b.v_angle, ref_idx)},
        rhs_terms = {},
    )


def _eq_dc_flow(b: _Build) -> None:
    """dc_flow_eq: flow on each DC arc from the angle difference of its end nodes."""
    d, v_flow, v_angle = b.d, b.v_flow, b.v_angle
    if v_flow is None or d.p_unitsize is None or d.p_connection_susceptance is None:
        return
    dc_arcs = dc_arcs_frame(d)
    if dc_arcs is None:
        return

    # ─── EQ. ─── dc_flow_eq ──────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # (v_flow[p,s,k,d,t] − v_flow_back[p,s,k,d,t]) · unitsize[p]
    #   = susceptance[p] · (v_angle[s,d,t] − v_angle[k,d,t])
    #   ∀ (p,s,k) ∈ DC arcs (dc_arcs_frame), (d,t) ∈ dt
    # Linearised AC flow: the signed flow (MW) on a DC arc follows the angle difference of its end nodes.
    # v_flow_back is absent when the model has no reverse-flow variable.  The rows run over dt.
    # Row rule (coarse steps): a coarse line whose both end nodes are coarse-angle nodes gets one row per coarse
    # step, dc_flow_eq_coarse = the row's mean over the step (_coarse_flows.add_cstr; the susceptance is constant,
    # so it is the row itself on one flow and two angles per step).  Any other line keeps a row per solve step and
    # reads a coarse flow at its constant value (a coarse line in a component with a fine DC line holds the angle
    # difference of its per-step angles constant over the coarse step).
    over = dc_arcs.join(d.dt, how="cross").select("p", "source", "sink", "d", "t")
    flow_signed = Where(v_flow, dc_arcs)
    if b.v_flow_back is not None:
        flow_signed = flow_signed - b.v_flow_back
    # Views of v_angle (same columns) indexed by the arc's source and sink node.
    v_angle_src = Var(name=v_angle.name + "__as_source", dims=("source", "d", "t"),
                      frame=v_angle.frame.rename({"n": "source"}), lower=v_angle.lower, upper=v_angle.upper)
    v_angle_snk = Var(name=v_angle.name + "__as_sink", dims=("sink", "d", "t"),
                      frame=v_angle.frame.rename({"n": "sink"}), lower=v_angle.lower, upper=v_angle.upper)
    angle_diff = (Where(v_angle_src, dc_arcs.select("p", "source"))
                  - Where(v_angle_snk, dc_arcs.select("p", "sink"))) * d.p_connection_susceptance
    _coarse_flows.add_row_rule_cstr(
        b, "dc_flow_eq",
        over      = over,
        sense     = "==",
        lhs_terms = {"flow": flow_signed * d.p_unitsize},
        rhs_terms = {"angle_diff": angle_diff},
        partition = _dc_partition(b, dc_arcs, over),
    )


def _dc_partition(b: _Build, dc_arcs: pl.DataFrame, over: pl.DataFrame) -> "pl.DataFrame | None":
    """The coarse steps ``(p, source, sink, d, b_first, t, w)`` of the DC arcs whose line is a coarse process and
    whose both end nodes are coarse-angle nodes (:func:`_coarse_flows.row_partition`); None without any."""
    steps = b.angle_steps
    if steps is None or steps.height == 0:
        return None
    arcs = dc_arcs.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink"))).unique()
    members = pl.concat([
        arcs.with_columns(kind=pl.lit("p"), e=pl.col("p")),
        arcs.with_columns(kind=pl.lit("n"), e=pl.col("source")),
        arcs.with_columns(kind=pl.lit("n"), e=pl.col("sink")),
    ]).sort("p", "source", "sink", "kind", "e")
    return _coarse_flows.row_partition(b, members, over, node_steps=steps)


# ---------------------------------------------------------------------------
# Node-balance terms of v_flow_back (used by model/core.py)


def nodeBalance_back_flow_terms(arcs: pl.DataFrame | None, v_flow_back: Var | None, p_unitsize: Param,
                                p_step_duration: Param) -> dict:
    """The v_flow_back terms of nodeBalance_eq, open dims (n, d, t); {} without reverse flow.

    back_at_sink[n,d,t]   = − Σ_{(p,s,n) ∈ arcs} v_flow_back[p,s,n,d,t] · unitsize[p] · step_duration[d,t]
    back_at_source[n,d,t] = + Σ_{(p,n,k) ∈ arcs} v_flow_back[p,n,k,d,t] · unitsize[p] · step_duration[d,t]
    The signed flow of an arc is v_flow − v_flow_back, so the reverse flow leaves the sink node and arrives
    at the source node.  ``arcs`` = every arc with a reverse-flow variable (twoway1var_arcs).
    """
    if v_flow_back is None or arcs is None or arcs.height == 0:
        return {}
    sink_as_n = arcs.with_columns(n=pl.col("sink")).select("p", "source", "sink", "n")
    src_as_n = arcs.with_columns(n=pl.col("source")).select("p", "source", "sink", "n")
    return {
        "back_at_sink": -Sum(
            Where(v_flow_back * p_unitsize, sink_as_n) * p_step_duration,
            over=("p", "source", "sink"),
        ),
        "back_at_source": Sum(
            Where(v_flow_back * p_unitsize, src_as_n) * p_step_duration,
            over=("p", "source", "sink"),
        ),
    }
