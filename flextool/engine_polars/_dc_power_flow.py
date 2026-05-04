"""DC power flow physics — linearised AC transmission constraints.

Mirrors the .mod's ``dc_flow_eq`` family (flextool.mod:3236-3244)::

    s.t. dc_flow_eq {p in connection_dc_power_flow,
                     (p, source, sink) in process_source_toSink,
                     (p, 'sink', b_out) in process__side__block,
                     (b_out, d, t) in block__period__step
                     : source in node_dc_power_flow
                       && sink in node_dc_power_flow} :
      v_flow[p, source, sink, d, t] * p_entity_unitsize[p]
      =
      p_connection_susceptance[p] * (v_angle[source, d, t] - v_angle[sink, d, t])

In flextool's V1 DC PF scenarios all participating nodes sit on the
default (hourly) block, so the block-aware filter
(``(p, 'sink', b_out) in process__side__block`` × ``(b_out, d, t) in
block__period__step``) reduces to the plain ``(d, t)`` set.  flexpy's
emission follows that V1 simplification: we index dc_flow_eq over
``connection_dc_power_flow × process_source_toSink × dt`` filtered to
``source, sink in node_dc_power_flow``.  Block-aware extension is
out of scope until a fixture exercises non-default blocks for DC PF
nodes (none do as of v51).

Inputs
------
* ``input/node_dc_power_flow.csv``      single-column ``node``
* ``input/connection_dc_power_flow.csv`` single-column ``process``
* ``input/node_reference_angle.csv``    single-column ``node`` (angle pinned to 0)
* ``input/p_connection_susceptance.csv`` two-column ``process,p_connection_susceptance``

Reference-angle policy
----------------------
flextool's ``_write_dc_power_flow_data`` (flextool/flextoolrunner/
input_writer.py:1060-1140) handles reference selection upstream and
emits the ``node_reference_angle.csv`` file.  Per connected component
of the DC PF subnetwork, exactly one node carries angle = 0:

  * If the group has ``reference_node`` set explicitly, that node is used.
  * Otherwise BFS finds connected components (via
    ``connection__node__node`` adjacency) and picks the node with the
    largest ``existing`` capacity in each component.

flexpy reads the resulting CSV verbatim and pins
``v_angle[ref, d, t] = 0`` via the angle Var's per-row tight bound (the
.mod uses the same trick — ``p_angle_lower = p_angle_upper = 0`` for
ref nodes).  The non-reference angle bounds are ±π (the .mod literal
``3.14159265``); flexpy's scalar-bound Var declaration uses ±π as the
loose bound, then emits an explicit equality ``v_angle[ref] = 0`` for
reference nodes (since Var bounds are scalar in flexpy).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high_opt import Param, Where

from ._input_source import _read_csv_file

if TYPE_CHECKING:
    from polar_high_opt.engine import Var


# ---------------------------------------------------------------------------
# Feature detection

def has_feature(d) -> bool:
    """True iff DC power flow data is populated and non-empty.

    Activation requires both at least one node in ``node_dc_power_flow``
    AND at least one connection in ``connection_dc_power_flow``.  A
    fixture with header-only CSVs (most of the v51 fixtures) returns
    False.
    """
    nd = getattr(d, "node_dc_power_flow", None)
    cd = getattr(d, "connection_dc_power_flow", None)
    if nd is None or cd is None:
        return False
    return nd.height > 0 and cd.height > 0


# ---------------------------------------------------------------------------
# Data loading

def load_data(inp_dir: str | Path) -> dict:
    """Read DC power flow CSVs from ``input/``.

    Returns a dict with keys matching the FlexData field names:

        node_dc_power_flow         pl.DataFrame | None  # cols: (n,)
        connection_dc_power_flow   pl.DataFrame | None  # cols: (p,)
        node_reference_angle       pl.DataFrame | None  # cols: (n,)
        p_connection_susceptance   Param | None         # dims: (p,)

    All values are ``None`` (or empty) when the feature is inactive
    (header-only CSV files, which is what the flextool preprocessor
    writes for non-DC-PF scenarios).
    """
    inp = Path(inp_dir)

    blank = dict(
        node_dc_power_flow       = None,
        connection_dc_power_flow = None,
        node_reference_angle     = None,
        p_connection_susceptance = None,
    )

    def _read_singles(path: Path, col_in: str, col_out: str) -> pl.DataFrame | None:
        if not path.exists():
            return None
        df = _read_csv_file(path)
        if df.height == 0:
            return None
        if col_in in df.columns and col_in != col_out:
            df = df.rename({col_in: col_out})
        return df.select(col_out)

    nd = _read_singles(inp / "node_dc_power_flow.csv", "node", "n")
    cd = _read_singles(inp / "connection_dc_power_flow.csv", "process", "p")
    rd = _read_singles(inp / "node_reference_angle.csv", "node", "n")

    pcs_path = inp / "p_connection_susceptance.csv"
    pcs_param = None
    if pcs_path.exists():
        df = _read_csv_file(pcs_path)
        if df.height > 0:
            df = df.rename(
                {c: r for c, r in [
                    ("process", "p"),
                    ("p_connection_susceptance", "value"),
                ] if c in df.columns}
            ).select("p", "value")
            pcs_param = Param(("p",), df)

    if nd is None and cd is None and rd is None and pcs_param is None:
        return blank

    return dict(
        node_dc_power_flow       = nd,
        connection_dc_power_flow = cd,
        node_reference_angle     = rd,
        p_connection_susceptance = pcs_param,
    )


# ---------------------------------------------------------------------------
# Variable + constraint emission

# Same float literal flextool's preprocessing (preprocessing/dc_angle_bounds.py)
# uses for the non-reference upper / lower angle bound.  An 8-digit
# truncation of π — preserved here for parity with flextool's MPS bit
# pattern.  See flextool.mod:1680-1681.
_PI_LITERAL = 3.14159265


def add_variables(m, d) -> "dict[str, Var]":
    """Declare ``v_angle[n, d, t]`` and ``v_flow_back[p, source, sink, d, t]``.

    ``v_angle`` is indexed by (n, d, t) where n ∈ ``node_dc_power_flow``.
    Bounds are set to the loose ±π for non-reference nodes; reference-angle
    pin to 0 is enforced by ``dc_reference_angle_eq`` below (flexpy Vars
    have scalar bounds, so we can't pin per-row in the Var declaration).

    ``v_flow_back`` is indexed by (p, source, sink, d, t) for DC PF arcs
    (p ∈ connection_dc_power_flow, both endpoints in node_dc_power_flow).
    flextool's ``method_2way_1var_off`` for DC PF connections allows
    ``v_flow ∈ [-1, +1]`` — i.e., physical flow can run sink→source.
    flexpy's standard ``v_flow ≥ 0`` doesn't admit that, so we model the
    reverse direction with a non-negative auxiliary ``v_flow_back`` and
    treat ``v_flow_signed := v_flow - v_flow_back`` as the algebraic flow:

      * ``dc_flow_eq``: ``(v_flow - v_flow_back) * unitsize ==
                          susc * (angle[source] - angle[sink])``
      * ``nodeBalance``: ``v_flow_back`` adds to source-side balance and
        subtracts from sink-side balance (mirror of ``v_flow``).
      * ``maxToSink_back``: ``v_flow_back ≤ existing/unitsize`` (matching
        the .mod's ``v_flow ≥ -existing/unitsize`` lower bound).
    """
    if not has_feature(d):
        return {}
    if d.dt is None or d.dt.height == 0:
        return {}

    # v_angle's index is the cross of node_dc_power_flow × dt.
    angle_idx = d.node_dc_power_flow.join(d.dt, how="cross").select("n", "d", "t")
    if angle_idx.height == 0:
        return {}

    v_angle = m.add_var(
        "v_angle", ("n", "d", "t"), angle_idx,
        lower=-_PI_LITERAL, upper=_PI_LITERAL,
    )

    out: dict = {"v_angle": v_angle}

    # v_flow_back index: DC arcs × dt, where DC arc means
    # connection_dc_power_flow ∩ process_source_sink with both endpoints in
    # node_dc_power_flow.  Skip if no such arcs (defensive — feature flag
    # already checked).
    if (d.process_source_sink is not None
            and d.connection_dc_power_flow is not None
            and d.connection_dc_power_flow.height > 0):
        dc_arcs = (d.process_source_sink
            .join(d.connection_dc_power_flow, on="p", how="inner")
            .filter(pl.col("source").is_in(d.node_dc_power_flow["n"]))
            .filter(pl.col("sink").is_in(d.node_dc_power_flow["n"])))
        if dc_arcs.height > 0:
            back_idx = dc_arcs.join(d.dt, how="cross").select(
                "p", "source", "sink", "d", "t")
            v_flow_back = m.add_var(
                "v_flow_back",
                ("p", "source", "sink", "d", "t"),
                back_idx, lower=0.0,
            )
            out["v_flow_back"] = v_flow_back
            out["dc_arcs"] = dc_arcs

    return out


def add_constraints(m, d, vars: dict, *,
                    v_flow=None, p_unitsize=None,
                    p_flow_upper_existing=None) -> None:
    """Emit the dc_flow_eq + reference-angle pin + back-flow capacity bound.

    ``v_flow``: the model's ``v_flow[p, source, sink, d, t]`` Var.  Required
    when ``connection_dc_power_flow`` is non-empty — without flow the
    angle-only LP would be vacuous.

    ``p_unitsize``: the ``p_entity_unitsize`` Param indexed by (p,).
    Required for the same reason.

    ``p_flow_upper_existing``: ``existing/unitsize`` Param indexed by
    (p, source, sink, d).  Used to bound ``v_flow_back`` symmetrically
    with ``v_flow``'s maxToSink (so the line's |flow| ≤ capacity).
    """
    if not has_feature(d):
        return
    v_angle = vars.get("v_angle")
    if v_angle is None:
        return

    # ── 1. Reference-angle pin ───────────────────────────────────────────
    # v_angle[ref, d, t] == 0 for ref ∈ node_reference_angle.
    # The .mod preprocesses this as a tight Var bound (p_angle_lower =
    # p_angle_upper = 0 on those rows).  flexpy Var bounds are scalar,
    # so emit an explicit equality constraint instead — same algebra.
    if (d.node_reference_angle is not None
            and d.node_reference_angle.height > 0):
        ref_idx = (d.node_reference_angle
                   .join(d.dt, how="cross")
                   .select("n", "d", "t"))
        if ref_idx.height > 0:
            m.add_cstr(
                "dc_reference_angle_eq",
                over      = ref_idx,
                sense     = "==",
                lhs_terms = {"angle": Where(v_angle, ref_idx)},
                rhs_terms = {},
            )

    # ── 2. dc_flow_eq ────────────────────────────────────────────────────
    # (v_flow - v_flow_back) * unitsize[p]
    #   == susceptance[p] * (v_angle[source, d, t] - v_angle[sink, d, t])
    # Indexed over (p, source, sink, d, t) where p ∈
    # connection_dc_power_flow AND source, sink ∈ node_dc_power_flow.
    if v_flow is None or p_unitsize is None or d.p_connection_susceptance is None:
        return
    if d.process_source_sink is None:
        return

    dc_arcs = vars.get("dc_arcs")
    if dc_arcs is None or dc_arcs.height == 0:
        return

    over = dc_arcs.join(d.dt, how="cross").select("p", "source", "sink", "d", "t")

    # LHS: (v_flow - v_flow_back) * unitsize.  v_flow_back is the
    # non-negative reverse-direction auxiliary (see add_variables).  In
    # fixtures with no negative-flow demand the LP keeps v_flow_back at
    # zero and the term reduces to v_flow * unitsize.
    v_flow_back = vars.get("v_flow_back")
    flow_signed = Where(v_flow, dc_arcs)
    if v_flow_back is not None:
        flow_signed = flow_signed - v_flow_back
    lhs_flow = flow_signed * p_unitsize

    # RHS: susceptance[p] * (v_angle[source, d, t] - v_angle[sink, d, t]).
    # v_angle is indexed by (n, d, t).  We need it twice — once aliased as
    # ``source`` and once aliased as ``sink`` — so the join with the per-arc
    # ``over`` frame matches the right column on each side.  Build virtual
    # Vars sharing v_angle's column ids but with renamed dim columns.
    from polar_high_opt.engine import Var

    v_angle_src = Var(
        name=v_angle.name + "__as_source",
        dims=("source", "d", "t"),
        frame=v_angle.frame.rename({"n": "source"}),
        lower=v_angle.lower, upper=v_angle.upper,
    )
    v_angle_snk = Var(
        name=v_angle.name + "__as_sink",
        dims=("sink", "d", "t"),
        frame=v_angle.frame.rename({"n": "sink"}),
        lower=v_angle.lower, upper=v_angle.upper,
    )
    susc = d.p_connection_susceptance   # Param over (p,)
    rhs_angle_diff = (Where(v_angle_src, dc_arcs.select("p", "source"))
                      - Where(v_angle_snk, dc_arcs.select("p", "sink"))) * susc

    m.add_cstr(
        "dc_flow_eq",
        over      = over,
        sense     = "==",
        lhs_terms = {"flow": lhs_flow},
        rhs_terms = {"angle_diff": rhs_angle_diff},
    )

    # ── 3. maxToSink for v_flow_back ─────────────────────────────────────
    # Symmetric capacity bound on the reverse-flow auxiliary so that
    # |signed flow| ≤ capacity (matching the .mod's ``v_flow ∈ [-1, 1]``
    # range scaled by unitsize).  Without this the LP could pump
    # arbitrary amounts of "fake" reverse flow to manufacture angle
    # differences, which would still net to zero in nodeBalance (since
    # v_flow_back appears with opposite signs at source/sink) but would
    # leave the angle solution unbounded for non-binding angles.
    if v_flow_back is not None and p_flow_upper_existing is not None:
        m.add_cstr(
            "maxToSink_back",
            over      = over,
            sense     = "<=",
            lhs_terms = {"flow_back": v_flow_back},
            rhs_terms = {"upper": p_flow_upper_existing},
        )


def nodeBalance_back_flow_terms(d, vars: dict, p_unitsize, p_step_duration) -> dict:
    """Return the v_flow_back contribution to nodeBalance, keyed by name.

    The signed flow is ``v_flow - v_flow_back``.  ``v_flow`` is already
    plumbed into ``model.py``'s nb_terms via ``flow_to_n`` / source-side
    ``flow_from_nodeBalance_*`` sets.  The ``-v_flow_back`` half mirrors
    those terms with reversed signs:

      * sink-side gets ``-v_flow_back × unitsize`` (back flow LEAVES sink)
      * source-side gets ``+v_flow_back × unitsize`` (back flow ARRIVES at source)

    Returns ``{}`` when no DC PF back flow is active.
    """
    v_flow_back = vars.get("v_flow_back")
    dc_arcs = vars.get("dc_arcs")
    if v_flow_back is None or dc_arcs is None or dc_arcs.height == 0:
        return {}

    from polar_high_opt import Sum

    # Sink side: back flow leaves the sink — subtract from sink balance.
    # Build ``flow_from_n_back`` = (p, source, sink, n=sink): same shape
    # as flow_to_n but flips the sign in nodeBalance.  We use
    # `Where(... )` with the frame having an explicit ``n`` column to
    # collapse the (p, source, sink) dims into (n, d, t) via Sum.
    sink_as_n = dc_arcs.with_columns(n=pl.col("sink")).select(
        "p", "source", "sink", "n")
    src_as_n = dc_arcs.with_columns(n=pl.col("source")).select(
        "p", "source", "sink", "n")

    return {
        # back flow at sink: -v_flow_back × unitsize × step_duration
        "dc_back_at_sink": -Sum(
            Where(v_flow_back * p_unitsize, sink_as_n) * p_step_duration,
            over=("p", "source", "sink"),
        ),
        # back flow at source: +v_flow_back × unitsize × step_duration
        "dc_back_at_source": Sum(
            Where(v_flow_back * p_unitsize, src_as_n) * p_step_duration,
            over=("p", "source", "sink"),
        ),
    }
