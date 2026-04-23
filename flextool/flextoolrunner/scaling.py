"""Per-solve scaling analyzer for the LP-scaling project (Agent 8).

This module implements a lightweight, Python-only numerical diagnostic
that inspects a solve's input CSVs (the ones emitted by
:mod:`flextool.flextoolrunner.input_writer` plus the solve-local
CSVs under ``solve_data/``) and recommends two scaling levers:

* ``use_row_scaling`` — the Agent-5 opt-in flag (0/1) that activates
  unitsize-based row scaling via ``p_use_row_scaling.csv``.  The
  analyzer recommends ``"yes"`` whenever the ``log10`` spread of
  ``p_entity_unitsize`` across the entities present in this solve
  exceeds 3 decades.
* ``scale_the_objective`` — the global objective scalar.  The
  analyzer estimates the expected objective magnitude from
  VOM × typical flows and CAPEX × typical capacities, then rounds
  the reciprocal to a power of 10.  (Symmetry-preservation rule
  from the project design memo: every auto-computed scale must be
  a power of 10 so structurally-identical entities share scales
  and HiGHS ``mip_detect_symmetry`` stays effective.)  As of
  Agent 12, the recommendation is emitted per solve to
  ``solve_data/scale_the_objective.csv`` and picked up by
  ``flextool.mod``; the legacy hardcoded
  ``param scale_the_objective := 1E-6;`` line in
  ``flextool_base.dat`` has been removed.

Modes
-----

Analysis runs **always** (cheap — one CSV pass, ~milliseconds per
solve).  The recommendation is serialised to
``solve_data/scaling_analysis.json`` for Agent 10's user-facing
report.

Application of the recommendation is opt-in:

* Default: **recommend only**.  The analyzer writes the JSON; it
  does not alter ``p_use_row_scaling.csv`` nor the objective scalar.
* ``--auto-scale`` (or ``FLEXTOOL_AUTO_SCALE=1``): the recommendation
  IS applied — but only when the user has not explicitly set the
  corresponding DB parameter.  If the user's ``solve.use_row_scaling``
  is ``"yes"`` or ``"no"``, the analyzer logs "user override detected,
  not auto-applying" and respects the user's choice.

**Safety caveat**: flipping ``use_row_scaling`` from ``"no"`` to
``"yes"`` is only numerically safe once Agent 9's output un-scaling
path is in place.  Because Agent 9 follows Agent 8 in the sequential
plan, auto-apply is **off by default** and must be explicitly enabled.

Caching
-------

Analysis is cached per solve name (module-level dict).  Rolling
windows invoke the same solve name repeatedly; the second and
subsequent calls are served from cache with no CSV re-read.  The
cache is keyed on ``solve_name`` alone; the CSVs are expected to
be deterministic for a given solve name within a run.

Design constraints
------------------

* Stdlib only (``csv``, ``json``, ``math``, ``os``, ``pathlib``).
* Does not touch ``flextool.mod`` or ``flextool_base.dat``.
* Does not introduce new DB schema.
* Emits at most one summary log line per solve — the full report is
  Agent 10's responsibility.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional


AUTO_SCALE_ENV_VAR = "FLEXTOOL_AUTO_SCALE"
"""Environment-variable fallback for the ``--auto-scale`` CLI flag."""

FORCE_USER_BOUND_SCALE_ENV_VAR = "FLEXTOOL_FORCE_USER_BOUND_SCALE"
"""Test-hook env var for Agent 18c variable-bound scaling.  When set to
an integer ``N`` (positive or negative) the runtime unconditionally
applies ``user_bound_scale=N`` on the ``Highs`` instance, bypassing the
analyser's automatic decision and the ``--auto-scale`` gate.  Mirrors
Agent 9's :data:`FLEXTOOL_FORCE_ROW_SCALING`.  Unset (default) means
no override; empty string / non-integer / out-of-range values fall
back to the analyser's decision."""

DEFAULT_OBJECTIVE_SCALE = 1e-6
"""Matches the legacy hardcoded ``scale_the_objective`` (removed in
Agent 12; now the ``default`` clause on the ``param`` declaration in
``flextool.mod`` and the fallback emitted here).

Returned as the recommendation whenever the analyzer cannot estimate
the objective magnitude (empty/unreadable inputs, all-zero cost
parameters, etc.)."""

OBJECTIVE_SCALE_MIN = 1e-12
OBJECTIVE_SCALE_MAX = 1e0
"""Clamp range for the recommended ``scale_the_objective``.  Stays
within the numerically-sane region any LP solver can digest without
underflow/overflow in the objective row."""

UNITSIZE_SPREAD_THRESHOLD = 3.0
"""Decades (base 10) — when ``log10(max) - log10(min)`` across
nonzero ``p_entity_unitsize`` values exceeds this threshold, row
scaling is recommended."""

RHS_SPREAD_THRESHOLD = 6.0
"""Decades (base 10) — when the pooled ``log10`` spread across the
``node_inflow`` and ``node_annual_flow`` families (the main RHS
contributors in balance constraints) exceeds this threshold, row
scaling is recommended.  Added by Agent 18b after rivendell's S19
showed that uniform unitsizes with a wide RHS benefit measurably
from row scaling (17% fewer dual-simplex iterations, 38% less wall
time on the productive phase)."""

COST_SPREAD_THRESHOLD = 5.0
"""Decades (base 10) — when the pooled ``log10`` spread across the
``vom_and_op_costs``, ``capex_invest``, and ``node_penalty`` families
(the objective contributors) exceeds this threshold, row scaling is
recommended.  Added by Agent 18b alongside :data:`RHS_SPREAD_THRESHOLD`
to catch composite models whose cost coefficients span many decades
despite uniform unit sizes."""


RHS_FAMILIES: tuple[str, ...] = ("node_inflow", "node_annual_flow")
"""Families pooled for the RHS spread trigger."""

COST_FAMILIES: tuple[str, ...] = (
    "vom_and_op_costs",
    "capex_invest",
    "node_penalty",
)
"""Families pooled for the cost spread trigger."""


BOUND_SPREAD_THRESHOLD = 6.0
"""Decades (base 10) — when the variable-bound spread ``log10(abs_max) -
log10(abs_min)`` across finite, non-zero LP column bounds exceeds this
threshold, :func:`decide_user_bound_scale` recommends a non-zero
``user_bound_scale``.  Agent 18c addition.  Rivendell S19 shows
bound spread ~9 decades (``[2e-3, 1e+6]``) and HiGHS itself prints
"Consider setting the user_bound_scale option to -8"; the threshold
sits below that so the trigger fires for similar rivendell-shaped
models."""

USER_BOUND_SCALE_MIN = -10
USER_BOUND_SCALE_MAX = 0
"""Clamp range for the recommended integer ``user_bound_scale``.  N is
always non-positive — positive N would grow bounds, which is the
opposite direction from what wide-spread models need.

Agent 18e softened the clamp from ``-30`` to ``-10``: empirically (rivendell
S19 with ``--ipm --auto-scale``) the anchored-to-max heuristic picked
``N=-20`` which broke HiGHS' crossover from interior point to dual
simplex — bound values compressed by 2^-20 ≈ 1e-6 × fell below float64
practical precision, and the crossover simplex could no longer represent
them at their original precision.  Capping at ``-10`` (bounds compressed
by ~1000×) is enough to tame HiGHS' coefficient ratio reporting without
crushing the bottom end.  Combined with the geometric-midpoint formula
below, the realistic N for rivendell-shaped models is around ``-6`` to
``-10``."""

BOUND_ABS_MIN_EFFECTIVE_ZERO = 1e-30
"""Threshold below which a measured ``abs_min`` is considered
effectively zero (e.g. ``1e-300`` from a slack with no lower cap).  The
geometric midpoint formula needs a strictly-positive ``abs_min``; a
denormal-small ``abs_min`` would collapse ``log2(geo_mid)`` to ``-inf``.
When the measured value falls at or below this threshold we fall back
to the floor below.  Agent 18e addition."""

BOUND_ABS_MIN_FLOOR_RATIO = 1e-6
"""Fallback floor for the bound-range ``abs_min`` when the measured
minimum is missing or effectively zero.  We then use
``abs_max * BOUND_ABS_MIN_FLOOR_RATIO`` as the floor, i.e. treat the
range as no wider than 6 decades for the purpose of centering.  A
legitimate ``abs_min`` (e.g. ``2e-3`` from a real lower bound) is
preserved — only the effectively-zero case is floored.  Agent 18e
addition."""


# ---------------------------------------------------------------------------
# CLI / env-var resolution
# ---------------------------------------------------------------------------


def resolve_auto_scale(cli_flag: bool) -> bool:
    """True iff the CLI flag is set OR ``FLEXTOOL_AUTO_SCALE`` is truthy.

    Mirrors ``resolve_report_near_duplicates`` in :mod:`precision`.
    Truthy env-var values: ``1`` / ``true`` / ``yes`` / ``on``
    (case-insensitive).
    """
    if cli_flag:
        return True
    raw = os.environ.get(AUTO_SCALE_ENV_VAR, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FamilyStats:
    """Summary statistics for one parameter family.

    ``log10`` stats are computed over the absolute values of the
    non-zero entries.  Zeros are counted separately so downstream
    consumers know the denominator for sparsity diagnostics.
    """

    n_values: int
    n_zero: int
    n_nonzero: int
    log10_min: Optional[float] = None
    log10_max: Optional[float] = None
    log10_median: Optional[float] = None
    log10_p10: Optional[float] = None
    log10_p90: Optional[float] = None
    abs_min: Optional[float] = None
    abs_max: Optional[float] = None
    abs_median: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScaleTable:
    """Analyzer output — one per solve.

    Serialised to ``solve_data/scaling_analysis.json`` via
    :meth:`to_dict` / :func:`write_scaling_analysis_json`.

    ``scale_the_state`` is currently fixed at ``1.0``; the field is
    reserved for future analyser tuning.  Agent 12 centralised the
    global scalars in Python so the .mod's ``default`` clauses are
    only consulted when the CSVs are absent (e.g. AMPL invoked
    outside the Python harness).
    """

    solve_name: str
    use_row_scaling: Literal["yes", "no"]
    scale_the_objective: float
    family_ranges: dict[str, FamilyStats]
    unitsize_spread_log10: float
    rough_obj_estimate: float
    timestamp: str
    source_dir: str
    scale_the_state: float = 1.0
    rhs_spread_log10: float = 0.0
    """Pooled ``log10`` spread across the :data:`RHS_FAMILIES` families.
    0.0 when no RHS-family data is present.  Agent 18b addition."""
    cost_spread_log10: float = 0.0
    """Pooled ``log10`` spread across the :data:`COST_FAMILIES` families.
    0.0 when no cost-family data is present.  Agent 18b addition."""
    row_scaling_trigger: Literal["unitsize", "rhs", "cost", "none"] = "none"
    """Which trigger activated ``use_row_scaling``.  First-match wins with
    precedence ``unitsize`` > ``rhs`` > ``cost``.  ``"none"`` when no
    trigger fired (``use_row_scaling == "no"``).  Agent 18b addition."""
    bound_spread_log10: float = 0.0
    """Variable-bound ``log10(abs_max) - log10(abs_min)`` spread across
    the finite, non-zero LP column bounds.  Populated after HiGHS has
    loaded the model (see :func:`compute_bound_stats` /
    :func:`decide_user_bound_scale`); ``0.0`` when the bounds have not
    been inspected yet, or when none of the bounds are finite and
    non-zero.  Agent 18c addition."""
    user_bound_scale: int = 0
    """Integer ``N`` passed to HiGHS as ``user_bound_scale``; 0 means no
    bound scaling.  Populated by :func:`decide_user_bound_scale` after
    the LP is loaded.  Negative values shrink bounds internally (HiGHS
    un-scales on output, so the solution remains invariant).  Agent 18c
    addition."""
    bound_abs_min: Optional[float] = None
    """Smallest absolute value of a finite, non-zero LP column bound.
    ``None`` when bounds have not been inspected.  Agent 18c addition."""
    bound_abs_max: Optional[float] = None
    """Largest absolute value of a finite, non-zero LP column bound.
    ``None`` when bounds have not been inspected.  Agent 18c addition."""

    def to_dict(self) -> dict:
        d = asdict(self)
        # dataclass asdict already flattens FamilyStats via its own
        # asdict call on nested dataclasses.
        return d


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------


_scale_cache: dict[str, ScaleTable] = {}
"""Per-solve-name cache.  Rolling windows reuse the same entry."""


def clear_cache() -> None:
    """Reset the cache (primarily for tests)."""
    _scale_cache.clear()


# ---------------------------------------------------------------------------
# Low-level CSV scanning helpers
# ---------------------------------------------------------------------------


def _iter_numeric_cells(path: Path) -> list[float]:
    """Yield every finite float value found in any column of *path*.

    The CSVs emitted by FlexTool use a variety of layouts:

    * Long (``entity, param, value`` etc.) — most parameter files.
    * Per-timestep (``node, time, value``) — ``pt_*`` / ``pd_*`` files.
    * Wide (header = entity names; a single data row of values) —
      ``p_entity_unitsize.csv``.

    For the purpose of summary log-spread statistics, the only thing
    that matters is the multiset of finite numeric cells; column
    semantics don't affect the spread / median / quantile
    computations.  We therefore parse every cell and keep those that
    float() accepts as finite.
    """
    values: list[float] = []
    if not path.exists():
        return values
    try:
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                for cell in row:
                    s = cell.strip()
                    if not s:
                        continue
                    try:
                        v = float(s)
                    except ValueError:
                        continue
                    if math.isfinite(v):
                        values.append(v)
    except OSError:
        return values
    return values


def _family_stats(values: list[float]) -> FamilyStats:
    """Compute a :class:`FamilyStats` summary from a list of floats.

    Uses pure stdlib; does **not** import ``numpy`` / ``pandas``.
    """
    n = len(values)
    if n == 0:
        return FamilyStats(n_values=0, n_zero=0, n_nonzero=0)
    zeros = sum(1 for v in values if v == 0.0)
    nonzero = [v for v in values if v != 0.0]
    if not nonzero:
        return FamilyStats(n_values=n, n_zero=zeros, n_nonzero=0)
    abs_vals = sorted(abs(v) for v in nonzero)
    log10_vals = sorted(math.log10(v) for v in abs_vals)

    def _pct(sorted_list: list[float], q: float) -> float:
        # Linear interpolation percentile; matches numpy default.
        if not sorted_list:
            return math.nan
        if len(sorted_list) == 1:
            return sorted_list[0]
        k = q * (len(sorted_list) - 1)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return sorted_list[lo]
        return sorted_list[lo] + (sorted_list[hi] - sorted_list[lo]) * (k - lo)

    return FamilyStats(
        n_values=n,
        n_zero=zeros,
        n_nonzero=len(nonzero),
        log10_min=log10_vals[0],
        log10_max=log10_vals[-1],
        log10_median=_pct(log10_vals, 0.5),
        log10_p10=_pct(log10_vals, 0.10),
        log10_p90=_pct(log10_vals, 0.90),
        abs_min=abs_vals[0],
        abs_max=abs_vals[-1],
        abs_median=_pct(abs_vals, 0.5),
    )


# ---------------------------------------------------------------------------
# Parameter-family definitions
# ---------------------------------------------------------------------------


# A *family* is a named set of CSV files whose values we want to pool
# for summary statistics.  The CSV layout is irrelevant — we just want
# all their finite numeric cells together.
FAMILIES: dict[str, list[str]] = {
    "entity_unitsize": [
        "p_entity_unitsize.csv",
    ],
    "node_inflow": [
        "pt_node_inflow.csv",
        "pbt_node_inflow.csv",
        "pd_node_inflow.csv",
    ],
    "node_annual_flow": [
        "pd_node.csv",  # annual-flow-style node params live here when period-indexed
    ],
    "vom_and_op_costs": [
        "p_process.csv",
        "p_process_source.csv",
        "p_process_sink.csv",
        "pd_process.csv",
        "pd_process_source.csv",
        "pd_process_sink.csv",
        "pbt_process.csv",
        "pbt_process_source.csv",
        "pbt_process_sink.csv",
        "pt_process.csv",
        "pt_process_source.csv",
        "pt_process_sink.csv",
        "p_commodity.csv",
        "pd_commodity.csv",
        "pdt_commodity.csv",
    ],
    "capex_invest": [
        "p_entity_invest_cost.csv",
        "p_group.csv",
    ],
    "node_penalty": [
        # Slack penalties live on the node parameters.
        "p_node.csv",
        "pd_node.csv",
    ],
}
"""Which CSVs to pool into each family.  Missing files are silently
skipped (they simply contribute zero values)."""


def _scan_family(input_dir: Path, filenames: list[str]) -> list[float]:
    pooled: list[float] = []
    for name in filenames:
        pooled.extend(_iter_numeric_cells(input_dir / name))
    return pooled


# ---------------------------------------------------------------------------
# Unitsize-specific helper (needs bespoke parsing for the wide format)
# ---------------------------------------------------------------------------


def _read_entity_unitsizes(input_dir: Path) -> list[float]:
    """Return one unitsize value per entity found in the solve's inputs.

    Two sources, in priority order:

    1. ``p_entity_unitsize.csv`` — a two-row wide-format file written
       by ``flextool.mod``'s printf block after the first solve.
       Layout::

           entity,e1,e2,e3,...
           value,v1,v2,v3,...

    2. Fallback: derive the same quantity from the DB-written
       ``p_process.csv`` / ``p_node.csv`` / ``p_connection.csv``
       files using the mod's rule
       ``virtual_unitsize if set else existing if set else 1000``.
       This fallback runs whenever the printf output is absent —
       notably on the very first solve of a run.

    The resulting list has one value per entity encountered across
    both sources (duplicates de-duplicated by entity name, with the
    wide-format file winning when present).
    """
    wide = _read_entity_unitsizes_wide(input_dir / "p_entity_unitsize.csv")
    if wide:
        return list(wide.values())
    derived = _derive_entity_unitsizes_from_params(input_dir)
    return list(derived.values())


def _read_entity_unitsizes_wide(path: Path) -> dict[str, float]:
    """Parse the wide-format ``p_entity_unitsize.csv`` if present.

    Returns ``{entity_name: unitsize}`` (empty dict when the file
    does not exist or has an unexpected shape).
    """
    if not path.exists():
        return {}
    try:
        with path.open(newline="") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
    except OSError:
        return {}
    if len(rows) < 2:
        return {}
    header = rows[0]
    data = rows[1]
    out: dict[str, float] = {}
    # Column 0 is the "entity" / "value" label — skip it.
    for name, cell in zip(header[1:], data[1:]):
        s = str(cell).strip()
        if not s:
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        if math.isfinite(v):
            out[name] = v
    return out


def _derive_entity_unitsizes_from_params(input_dir: Path) -> dict[str, float]:
    """Compute per-entity unitsize from the DB-written parameter CSVs.

    Mirrors the ``p_entity_unitsize`` rule in ``flextool.mod``::

        unitsize(e) =
            virtual_unitsize if present and > 0
            else existing if present and > 0
            else 1000        (model-level default)

    Scans ``p_process.csv``, ``p_node.csv``, ``p_connection.csv``
    — each has the "long" layout (``<entity>, <param>, <value>``).
    The analyzer does NOT apply the default 1000 fallback: that
    sentinel is the same for every entity and would falsely flatten
    the spread.  Only entities with at least one explicit value
    contribute.
    """
    sources = [
        ("p_process.csv", "process"),
        ("p_node.csv", "node"),
        ("p_connection.csv", "connection"),
    ]
    virtual: dict[str, float] = {}
    existing: dict[str, float] = {}
    for filename, key in sources:
        path = input_dir / filename
        if not path.exists():
            continue
        try:
            with path.open(newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if header is None:
                    continue
                # Column 0 is the entity name, column 1 is the param,
                # column 2 is the value.  This is FlexTool's long
                # layout for scalar params on processes / nodes /
                # connections.
                for row in reader:
                    if len(row) < 3:
                        continue
                    entity = row[0].strip()
                    param = row[1].strip()
                    value_raw = row[2].strip()
                    if not entity or not value_raw:
                        continue
                    try:
                        v = float(value_raw)
                    except ValueError:
                        continue
                    if not math.isfinite(v) or v == 0.0:
                        continue
                    if param == "virtual_unitsize":
                        virtual[entity] = v
                    elif param == "existing":
                        existing[entity] = v
        except OSError:
            continue

    # virtual_unitsize wins; then fall back to existing.  We skip the
    # model-level default of 1000 so the spread reflects only the
    # user's actual data.
    out: dict[str, float] = {}
    for entity, v in virtual.items():
        out[entity] = v
    for entity, v in existing.items():
        out.setdefault(entity, v)
    return out


# ---------------------------------------------------------------------------
# Rough objective estimate
# ---------------------------------------------------------------------------


_COST_PARAM_NAMES: tuple[str, ...] = (
    # Process / connection cost parameters (recognised by name in the
    # long-format p_process.csv / p_connection.csv).
    "other_operational_cost",
    "startup_cost",
    "invest_cost",
    "fixed_cost",
    "salvage_value",
    # Node-level cost / penalty parameters.
    "penalty_up",
    "penalty_down",
    "storage_state_reference_price",
)


def _sum_cost_params(input_dir: Path) -> tuple[float, float]:
    """Walk the long-format parameter CSVs and split costs into
    ``(sum_vom, sum_capex)`` — both in the parameter's native unit.

    Classification follows the project's cost-family taxonomy:

    * CAPEX: ``invest_cost``, ``fixed_cost``, ``salvage_value``.
    * VOM / OPEX / penalties: everything else in
      :data:`_COST_PARAM_NAMES`.
    """
    capex_params = {"invest_cost", "fixed_cost", "salvage_value"}
    sum_vom = 0.0
    sum_capex = 0.0
    for filename in ("p_process.csv", "p_node.csv", "p_connection.csv"):
        path = input_dir / filename
        if not path.exists():
            continue
        try:
            with path.open(newline="") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                if header is None:
                    continue
                for row in reader:
                    if len(row) < 3:
                        continue
                    param = row[1].strip()
                    if param not in _COST_PARAM_NAMES:
                        continue
                    try:
                        v = float(row[2].strip())
                    except ValueError:
                        continue
                    if not math.isfinite(v) or v == 0.0:
                        continue
                    if param in capex_params:
                        sum_capex += abs(v)
                    else:
                        sum_vom += abs(v)
        except OSError:
            continue
    # Commodity price (p_commodity.csv, long-format) also contributes
    # to VOM-style operational cost.
    cpath = input_dir / "p_commodity.csv"
    if cpath.exists():
        try:
            with cpath.open(newline="") as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    if len(row) < 3:
                        continue
                    param = row[1].strip()
                    if param != "price":
                        continue
                    try:
                        v = float(row[2].strip())
                    except ValueError:
                        continue
                    if math.isfinite(v) and v != 0.0:
                        sum_vom += abs(v)
        except OSError:
            pass
    return sum_vom, sum_capex


def _estimate_rough_obj(
    family_values: dict[str, list[float]],
    input_dir: Path,
    unitsize_family: str = "entity_unitsize",
    inflow_family: str = "node_inflow",
) -> float:
    """Back-of-envelope total-cost magnitude.

    The goal is not accuracy — only the order of magnitude matters,
    because we round to the nearest power of 10 to preserve
    symmetry.  The formula::

        rough_obj ≈ (sum |VOM|) × typical_flow × n_timesteps
                  + (sum |CAPEX|) × typical_capacity

    * ``typical_flow`` ≈ median absolute inflow (MW).  Scaled by the
      number of non-zero inflow cells which is a proxy for
      period × timesteps.
    * ``typical_capacity`` ≈ median non-zero unitsize (MW).

    VOM and CAPEX are recognised by parameter name (see
    :data:`_COST_PARAM_NAMES` / :func:`_sum_cost_params`) rather
    than pooled from whole CSVs so unitsize / existing / lifetime
    cells don't contaminate the cost estimate.

    Returns ``0.0`` when inputs are insufficient for a meaningful
    estimate (caller falls back to the default scalar).
    """
    inflow_vals = family_values.get(inflow_family, [])
    unitsize_vals = family_values.get(unitsize_family, [])

    sum_vom, sum_capex = _sum_cost_params(input_dir)

    nonzero_inflows = [abs(v) for v in inflow_vals if v != 0.0]
    nonzero_unitsizes = [abs(v) for v in unitsize_vals if v != 0.0]

    typical_flow = 0.0
    timesteps_like = 0
    if nonzero_inflows:
        nonzero_inflows.sort()
        typical_flow = nonzero_inflows[len(nonzero_inflows) // 2]
        timesteps_like = len(nonzero_inflows)

    typical_cap = 0.0
    if nonzero_unitsizes:
        nonzero_unitsizes.sort()
        typical_cap = nonzero_unitsizes[len(nonzero_unitsizes) // 2]

    operational = sum_vom * typical_flow * max(1, timesteps_like)
    investment = sum_capex * typical_cap

    return operational + investment


def _pooled_spread_log10(
    family_stats: dict[str, FamilyStats],
    families: tuple[str, ...] | list[str],
) -> float:
    """Return ``log10(max_across_all) − log10(min_nonzero_across_all)``
    over the ``abs_min`` / ``abs_max`` of the named families.

    Families with no nonzero data (``abs_min`` / ``abs_max`` is
    ``None``) contribute nothing.  If no family has data at all, the
    spread is ``0.0``.  Agent 18b helper for the widened row-scaling
    decision.
    """
    abs_mins: list[float] = []
    abs_maxs: list[float] = []
    for name in families:
        stats = family_stats.get(name)
        if stats is None:
            continue
        if stats.abs_min is not None and stats.abs_min > 0.0:
            abs_mins.append(stats.abs_min)
        if stats.abs_max is not None and stats.abs_max > 0.0:
            abs_maxs.append(stats.abs_max)
    if not abs_mins or not abs_maxs:
        return 0.0
    overall_min = min(abs_mins)
    overall_max = max(abs_maxs)
    if overall_min <= 0.0 or overall_max <= 0.0:
        return 0.0
    try:
        return math.log10(overall_max) - math.log10(overall_min)
    except ValueError:
        return 0.0


def compute_bound_stats(
    col_lower: list[float] | tuple[float, ...],
    col_upper: list[float] | tuple[float, ...],
) -> tuple[Optional[float], Optional[float], float]:
    """Compute ``(abs_min, abs_max, spread_log10)`` from LP column bounds.

    Only finite, non-zero bounds contribute — ``inf`` / ``-inf`` are
    skipped (free variables), ``NaN`` is skipped (defensive), and ``0.0``
    is skipped because its absolute value is not a meaningful scale
    indicator.  Negative bounds contribute via their absolute value.

    Parameters
    ----------
    col_lower, col_upper:
        Sequences (highspy returns Python lists from ``getLp()``) of
        column lower / upper bounds, in the column order of the LP.

    Returns
    -------
    (abs_min, abs_max, spread_log10)
        ``abs_min`` / ``abs_max`` are the smallest and largest absolute
        values encountered, or ``None`` when no finite non-zero bound
        exists.  ``spread_log10`` = ``log10(abs_max) - log10(abs_min)``
        or ``0.0`` when ``abs_min`` or ``abs_max`` is absent.
    """
    abs_min: Optional[float] = None
    abs_max: Optional[float] = None

    def _consider(v: float) -> None:
        nonlocal abs_min, abs_max
        if not math.isfinite(v):
            return
        a = abs(v)
        if a == 0.0:
            return
        if abs_min is None or a < abs_min:
            abs_min = a
        if abs_max is None or a > abs_max:
            abs_max = a

    for v in col_lower:
        _consider(float(v))
    for v in col_upper:
        _consider(float(v))

    if abs_min is None or abs_max is None:
        return (abs_min, abs_max, 0.0)
    try:
        spread = math.log10(abs_max) - math.log10(abs_min)
    except ValueError:
        return (abs_min, abs_max, 0.0)
    return (abs_min, abs_max, spread)


def decide_user_bound_scale(
    bound_abs_max: Optional[float],
    bound_spread_log10: float,
    threshold: float = BOUND_SPREAD_THRESHOLD,
    bound_abs_min: Optional[float] = None,
) -> int:
    """Pick an integer ``user_bound_scale`` from measured bound statistics.

    Policy (Agent 18e, softened from Agent 18c's anchored-to-max):

    * If ``bound_spread_log10 <= threshold`` (default
      :data:`BOUND_SPREAD_THRESHOLD` = 6 decades) → return ``0`` (no
      scaling needed).
    * Otherwise choose::

          geo_mid = sqrt(abs_max * abs_min)
          N       = -round(log2(geo_mid))

      so that ``2^N * geo_mid ≈ 1``, centering the bound range around
      ``O(1)`` rather than collapsing its upper end (Agent 18c behaviour).
      This is much gentler on the bottom end of the range: a model with
      bounds in ``[2e-3, 1e+6]`` gets ``geo_mid ≈ 45`` → ``N = -5``
      instead of the former ``-round(log2(1e6)) = -20``.
    * Clamp to ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]`` = ``[-10, 0]``.

    When ``bound_abs_min`` is missing or effectively zero we fall back to
    :data:`BOUND_ABS_MIN_FLOOR_RATIO` × ``bound_abs_max`` (i.e. treat the
    range as no wider than 6 decades for the purpose of centering); this
    protects against ``log2(0) = -inf`` when a slack variable lacks a
    lower cap.

    Agent 18e rationale (see ``projects/rivendell/agent18d_tolerance_ipm/
    BENCHMARK_REPORT.md``): the old anchored-to-max heuristic picks
    ``N=-20`` on rivendell S19, which breaks HiGHS' crossover from IPM
    to dual simplex.  HiGHS' own hint for that same model is ``-8``; the
    geometric-midpoint centering lands within 2–3 of HiGHS' hint and the
    hard clamp at ``|N| ≤ 10`` prevents pathological compression even on
    more extreme models.

    Degenerate inputs (missing / non-positive / non-finite ``bound_abs_max``)
    return ``0``.  This is a scalar, solver-invariant internal scaling:
    HiGHS multiplies all column bounds by ``2^N``, solves, and un-scales
    outputs before returning them — so applying ``N != 0`` does NOT
    change the optimum.
    """
    if bound_spread_log10 <= threshold:
        return 0
    if bound_abs_max is None or bound_abs_max <= 0.0 or not math.isfinite(
        bound_abs_max
    ):
        return 0
    # Pick an effective abs_min for centering.  A non-positive or
    # effectively-zero measured min (happens when a slack with no lower
    # cap registers as ``~1e-300``) makes ``sqrt(abs_max * abs_min)``
    # numerically unusable; fall back to the 6-decade floor relative to
    # abs_max.  A legitimate small-but-nonzero abs_min (e.g. ``2e-3``
    # from a real lower bound) is preserved — only the effectively-zero
    # case is floored.
    if (
        bound_abs_min is None
        or not math.isfinite(bound_abs_min)
        or bound_abs_min <= BOUND_ABS_MIN_EFFECTIVE_ZERO
    ):
        effective_min = bound_abs_max * BOUND_ABS_MIN_FLOOR_RATIO
    else:
        effective_min = bound_abs_min
    try:
        geo_mid = math.sqrt(bound_abs_max * effective_min)
        if geo_mid <= 0.0 or not math.isfinite(geo_mid):
            return 0
        lg2 = math.log2(geo_mid)
    except ValueError:
        return 0
    n = -int(round(lg2))
    if n < USER_BOUND_SCALE_MIN:
        n = USER_BOUND_SCALE_MIN
    if n > USER_BOUND_SCALE_MAX:
        n = USER_BOUND_SCALE_MAX
    return n


def resolve_force_user_bound_scale() -> Optional[int]:
    """Parse :data:`FORCE_USER_BOUND_SCALE_ENV_VAR` into an int or None.

    Returns ``None`` when the env var is unset, empty, or non-integer.
    Out-of-range values are clamped to
    ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``.  This hook lets
    benchmarks and tests pin a specific N (rivendell's -8, for
    example) without editing code.
    """
    raw = os.environ.get(FORCE_USER_BOUND_SCALE_ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    if n < USER_BOUND_SCALE_MIN:
        n = USER_BOUND_SCALE_MIN
    if n > USER_BOUND_SCALE_MAX:
        n = USER_BOUND_SCALE_MAX
    return n


def apply_bound_scale_decision(
    solve_name: str,
    col_lower: list[float] | tuple[float, ...],
    col_upper: list[float] | tuple[float, ...],
    auto_scale: bool,
    user_opt_set: bool,
    logger: Optional[logging.Logger] = None,
) -> tuple[int, Optional[float], Optional[float], float, str]:
    """Orchestrate the full Agent-18c bound-scaling decision.

    Combines :func:`compute_bound_stats`, :func:`decide_user_bound_scale`
    and the opt-out / force-override rules into a single call so the
    solver-runner hook stays small.

    Returns
    -------
    (n, abs_min, abs_max, spread_log10, source)
        * ``n`` — integer to pass to ``h.setOptionValue('user_bound_scale', n)``.
          ``0`` means "leave alone".  When *user_opt_set* is True, this
          function always returns ``0`` regardless of the spread because
          the user's ``highs.opt`` has already set an explicit value
          which must not be overridden.
        * ``abs_min`` / ``abs_max`` / ``spread_log10`` — diagnostics for
          the cached :class:`ScaleTable` / the scaling report.
        * ``source`` — one of ``"user-opt"`` / ``"force-env"`` /
          ``"auto-scale"`` / ``"auto-scale-off"`` / ``"below-threshold"``.
          Explains which branch of the decision tree applied.
    """
    abs_min, abs_max, spread = compute_bound_stats(col_lower, col_upper)

    if user_opt_set:
        if logger is not None:
            logger.info(
                "[scaling] %s: user_bound_scale already set via highs.opt; "
                "leaving alone (bound range [%s, %s], spread=%.2f decades)",
                solve_name,
                _fmt_bound(abs_min),
                _fmt_bound(abs_max),
                spread,
            )
        return (0, abs_min, abs_max, spread, "user-opt")

    forced = resolve_force_user_bound_scale()
    if forced is not None:
        if logger is not None:
            logger.info(
                "[scaling] %s: user_bound_scale forced to %d via %s "
                "(bound range [%s, %s], spread=%.2f decades)",
                solve_name,
                forced,
                FORCE_USER_BOUND_SCALE_ENV_VAR,
                _fmt_bound(abs_min),
                _fmt_bound(abs_max),
                spread,
            )
        return (forced, abs_min, abs_max, spread, "force-env")

    if not auto_scale:
        return (0, abs_min, abs_max, spread, "auto-scale-off")

    n = decide_user_bound_scale(abs_max, spread, bound_abs_min=abs_min)
    if n == 0:
        return (0, abs_min, abs_max, spread, "below-threshold")
    if logger is not None:
        logger.info(
            "[scaling] %s: user_bound_scale set to %d "
            "(bound range was [%s, %s], spread=%.2f decades)",
            solve_name,
            n,
            _fmt_bound(abs_min),
            _fmt_bound(abs_max),
            spread,
        )
    return (n, abs_min, abs_max, spread, "auto-scale")


def _fmt_bound(v: Optional[float]) -> str:
    if v is None or not math.isfinite(v):
        return "n/a"
    return f"{v:.3g}"


def update_bound_scale_in_cache(
    solve_name: str,
    n: int,
    abs_min: Optional[float],
    abs_max: Optional[float],
    spread_log10: float,
) -> None:
    """Mutate the cached :class:`ScaleTable` for *solve_name* with the
    bound-scale decision.

    The LP bounds are only known post-load — after HiGHS has read the
    MPS — which is later than the analyser's CSV-only pass.  The
    per-solve cache is shared with the scaling-report renderer, so
    updating it here lets the report show the actually-applied values
    without plumbing a second object through the call chain.  No-op
    when no entry exists (e.g. analyser was never called for this
    solve, as in glpsol-only paths).
    """
    table = _scale_cache.get(solve_name)
    if table is None:
        return
    table.user_bound_scale = int(n)
    table.bound_abs_min = abs_min
    table.bound_abs_max = abs_max
    table.bound_spread_log10 = float(spread_log10)


def _recommend_scale_the_objective(rough_obj: float) -> float:
    """Map *rough_obj* to a power-of-10 scalar within the clamp range.

    Rule: ``scale = 10 ** -round(log10(rough_obj))``.  ``round``
    rounds-to-even in Python 3; the tie-breaking doesn't matter
    because the input is a rough estimate anyway.

    Clamped to ``[1e-12, 1e0]``.  Degenerate / unusable inputs fall
    back to :data:`DEFAULT_OBJECTIVE_SCALE`.
    """
    if not (math.isfinite(rough_obj)) or rough_obj <= 0.0:
        return DEFAULT_OBJECTIVE_SCALE
    try:
        lg = math.log10(rough_obj)
    except ValueError:
        return DEFAULT_OBJECTIVE_SCALE
    scale = 10.0 ** -round(lg)
    if scale < OBJECTIVE_SCALE_MIN:
        scale = OBJECTIVE_SCALE_MIN
    if scale > OBJECTIVE_SCALE_MAX:
        scale = OBJECTIVE_SCALE_MAX
    return scale


# ---------------------------------------------------------------------------
# The analyzer entry point
# ---------------------------------------------------------------------------


def analyze_solve(
    solve_name: str,
    input_dir: Path | str,
    logger: Optional[logging.Logger] = None,
) -> ScaleTable:
    """Analyse the inputs of *solve_name* and return a :class:`ScaleTable`.

    Parameters
    ----------
    solve_name:
        The solve name being analysed.  Used as the cache key.
    input_dir:
        Directory containing the solve's ``*.csv`` inputs.  Typically
        ``<work_folder>/input`` — the analyzer is permissive: missing
        files contribute zero values.
    logger:
        Optional logger for one-line summaries.  Silent when ``None``.

    Returns
    -------
    ScaleTable
        The per-solve analysis result.  Subsequent calls with the same
        *solve_name* return the cached table without re-reading the
        CSVs.
    """
    if solve_name in _scale_cache:
        return _scale_cache[solve_name]

    input_path = Path(input_dir)

    # ---- Scan every family ----
    # Unitsize gets its own parser because of the wide-format layout
    # AND because it may not exist yet on the very first solve — it
    # is emitted by flextool.mod's printf block, not the Python input
    # writer.  When the wide-format CSV is absent, we derive the
    # same quantity from the DB-sourced p_process / p_node / p_connection
    # CSVs (``virtual_unitsize`` → ``existing`` → default 1000).
    family_values: dict[str, list[float]] = {}
    for name, filenames in FAMILIES.items():
        if name == "entity_unitsize":
            family_values[name] = _read_entity_unitsizes(input_path)
        else:
            family_values[name] = _scan_family(input_path, filenames)

    family_stats = {name: _family_stats(vals) for name, vals in family_values.items()}

    # ---- Unitsize spread (log10) ----
    unitsize_stats = family_stats.get("entity_unitsize")
    if (
        unitsize_stats is not None
        and unitsize_stats.log10_max is not None
        and unitsize_stats.log10_min is not None
    ):
        spread = unitsize_stats.log10_max - unitsize_stats.log10_min
    else:
        spread = 0.0

    # ---- RHS spread (log10) — Agent 18b ----
    # Pool across the RHS-family stats.  "Pooled spread" = log10 of the
    # overall abs_max divided by the overall abs_min across the
    # listed families.  Families with no nonzero data contribute
    # nothing.
    rhs_spread = _pooled_spread_log10(family_stats, RHS_FAMILIES)

    # ---- Cost spread (log10) — Agent 18b ----
    cost_spread = _pooled_spread_log10(family_stats, COST_FAMILIES)

    # ---- Trigger decision (first match wins; precedence
    #      unitsize > rhs > cost) ----
    trigger: Literal["unitsize", "rhs", "cost", "none"]
    if spread > UNITSIZE_SPREAD_THRESHOLD:
        trigger = "unitsize"
    elif rhs_spread > RHS_SPREAD_THRESHOLD:
        trigger = "rhs"
    elif cost_spread > COST_SPREAD_THRESHOLD:
        trigger = "cost"
    else:
        trigger = "none"

    use_row_scaling: Literal["yes", "no"] = (
        "yes" if trigger != "none" else "no"
    )

    # ---- Objective scalar recommendation ----
    rough_obj = _estimate_rough_obj(family_values, input_path)
    scale_obj = _recommend_scale_the_objective(rough_obj)

    table = ScaleTable(
        solve_name=solve_name,
        use_row_scaling=use_row_scaling,
        scale_the_objective=scale_obj,
        family_ranges=family_stats,
        unitsize_spread_log10=spread,
        rough_obj_estimate=rough_obj,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_dir=str(input_path),
        rhs_spread_log10=rhs_spread,
        cost_spread_log10=cost_spread,
        row_scaling_trigger=trigger,
    )
    _scale_cache[solve_name] = table

    if logger is not None:
        logger.info(
            "[scaling] %s: unitsize_spread=%.2f rhs_spread=%.2f cost_spread=%.2f "
            "decades → use_row_scaling=%s (trigger=%s); "
            "rough_obj=%.3g → scale_the_objective=%g",
            solve_name,
            spread,
            rhs_spread,
            cost_spread,
            use_row_scaling,
            trigger,
            rough_obj,
            scale_obj,
        )
    return table


# ---------------------------------------------------------------------------
# JSON emission
# ---------------------------------------------------------------------------


def write_scaling_analysis_json(
    table: ScaleTable,
    solve_data_dir: Path | str,
    filename: str = "scaling_analysis.json",
) -> Path:
    """Serialise *table* under ``solve_data_dir / filename``.

    Format: nested JSON with one key per dataclass field; family
    stats are serialised via :meth:`FamilyStats.to_dict`.  Read by
    Agent 10 to build its user-facing report.
    """
    sd = Path(solve_data_dir)
    sd.mkdir(parents=True, exist_ok=True)
    payload = table.to_dict()
    # asdict already flattens nested dataclasses; json.dump handles
    # float("nan") only with allow_nan=True (the default) — we keep
    # the default to make debug-time round-tripping work even when
    # some families are empty.
    path = sd / filename
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# Auto-apply helper
# ---------------------------------------------------------------------------


def maybe_auto_apply_row_scaling(
    solve_name: str,
    table: ScaleTable,
    user_setting: Optional[str],
    auto_scale: bool,
    logger: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Decide whether the analyzer's ``use_row_scaling`` recommendation
    should override the caller's user-setting dict.

    Returns
    -------
    str or None
        * ``"yes"`` / ``"no"`` — caller should use this value instead
          of whatever was loaded from the DB.
        * ``None`` — caller should leave its DB value untouched.

    Decision tree
    -------------
    1. ``auto_scale=False`` → always ``None`` (recommend-only mode).
    2. ``auto_scale=True`` AND user_setting is ``"yes"`` / ``"no"`` →
       respect the user; log "user override detected".  Returns
       ``None``.
    3. ``auto_scale=True`` AND user_setting is missing / empty → apply
       the analyzer's recommendation (returns ``table.use_row_scaling``).
    """
    if not auto_scale:
        return None
    if user_setting is not None:
        s = str(user_setting).strip().lower()
        if s in ("yes", "no"):
            if logger is not None:
                logger.info(
                    "[scaling] %s: user override detected "
                    "(use_row_scaling=%r), not auto-applying (recommended=%s).",
                    solve_name,
                    user_setting,
                    table.use_row_scaling,
                )
            return None
    if logger is not None:
        logger.info(
            "[scaling] %s: auto-applying recommended use_row_scaling=%s",
            solve_name,
            table.use_row_scaling,
        )
    return table.use_row_scaling
