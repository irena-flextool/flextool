"""Phase F — typed registry of FlexTool output bundle files + manifest.

This module sits *atop* the existing output writers; it does NOT change
who writes what.  It provides:

* :class:`ParquetSpec` — a typed schema describing one emitted output.
* :data:`REGISTRY` — a ``key → ParquetSpec`` dict covering both
  ``output_raw/`` (per-solve LP variable parquets, plus 4 capacity CSVs
  carried by ``handoff_writers``) and ``output_processed/`` (the
  ``write_outputs`` per-table outputs governed by ``--write-methods``;
  ``output_processed`` here is the logical category covering
  ``output_parquet/``, ``output_csv/``, ``output_excel/`` and
  ``output_plots/`` produced by :mod:`flextool.process_outputs.write_outputs`).
* :func:`write_parquet` — typed write path.  Validates a frame's column
  set against its REGISTRY entry, then writes via polars
  ``DataFrame.write_parquet`` to ``work_folder / category / filename``.
  Existing writers can opt in by calling this; the existing writers are
  NOT migrated in this phase.
* :func:`write_manifest` — walks REGISTRY, checks file presence on disk,
  emits ``manifest.json`` at ``work_folder / "manifest.json"``.  Idempotent.
  Logs warnings (not errors) for mismatches: a registered file that
  doesn't exist (and isn't documented as conditional), or a parquet
  file present in ``output_raw/`` / ``output_processed/`` with no
  matching REGISTRY entry.

Per-solve raw variables are emitted as ``<name>__<solve>.parquet``
(one shard per cascade sub-solve).  The manifest enumerates each shard
as a separate file entry under the same logical key — discovery is by
filename-glob ``<filename>`` interpreted as a glob pattern when it
contains ``*``; otherwise as an exact basename.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

_logger = logging.getLogger(__name__)


# Logical category → on-disk subdirectory (relative to ``work_folder``).
# ``output_raw`` is the canonical home of the per-solve LP variable parquets +
# the 4 handoff-capacity CSVs.
# ``output_processed`` is the LOGICAL home for the user-facing outputs;
# the actual files live under ``<output_location>/output_{parquet,csv,excel,plots}/``
# (which may sit outside ``work_folder``).  When the processed bundle is
# co-located with ``work_folder`` (the default for tests / CLI without
# ``--output-location``), the manifest will find them via the symlink-free
# ``output_parquet`` lookup below.
CATEGORY_DIRS: dict[str, tuple[str, ...]] = {
    "raw": ("output_raw",),
    # Processed files appear under one of these subdirs depending on
    # ``--write-methods``.  The first match wins.
    "processed": ("output_parquet", "output_csv", "output_excel"),
}


@dataclass(frozen=True)
class ParquetSpec:
    """Typed schema for one emitted output file (or one glob of shards).

    Attributes
    ----------
    key
        Registry key — short, stable name for this output (e.g.
        ``"v_flow"``, ``"costs_dt_p"``).  Matches the file basename
        for processed outputs; matches the per-solve shard prefix for
        raw variables.
    category
        ``"raw"`` (mapped to ``output_raw/`` under the bundle root)
        or ``"processed"`` (mapped to ``output_processed/`` — the
        logical category covering the user-facing output formats).
        See :data:`CATEGORY_DIRS` for the on-disk directory mapping.
    filename
        Basename or glob pattern, relative to ``category`` directory.
        Glob patterns (containing ``*``) are matched against the
        directory listing at manifest-write time so per-solve shards
        like ``v_flow__<solve>.parquet`` enumerate cleanly.
    columns
        Canonical column list.  Validated by :func:`write_parquet`
        against the frame.  Empty tuple = "any columns" (skip
        validation; appropriate for wide formats whose column set is
        scenario-dependent).
    indices
        Subset of ``columns`` that form the row index in the canonical
        wide-format DataFrame.  Empty tuple if rows are positional /
        the file has no logical index columns.
    note
        Human-readable description.  When the file is conditional
        (depends on a feature being active — e.g. CO2 cap, DC PF, …),
        document the condition here so manifest readers can interpret a
        missing file.
    producer
        ``module.function`` path of the writer that emits this file.
    """

    key: str
    category: str
    filename: str
    columns: tuple[str, ...]
    indices: tuple[str, ...]
    note: str
    producer: str


# ---------------------------------------------------------------------------
# REGISTRY — populated by hand from a walk of the writers.
#
# Coverage status (Phase F first cut):
#   * raw / v_*.parquet — COMPLETE (all 30 emitted variables documented)
#   * raw / *.csv handoff capacity tables — COMPLETE (4 files)
#   * processed — REPRESENTATIVE.  ~95 unique table names exist; the
#     registry below covers the high-traffic ones.  Missing entries are
#     surfaced (as warnings, not errors) by ``write_manifest`` so the
#     coverage gap is visible without blocking the bundle write.
#
# Adding a new entry: pick a stable ``key``, set ``category``, set
# ``filename`` (glob if per-solve shards), document ``columns`` /
# ``indices`` / ``note`` / ``producer``.  Order doesn't matter; the
# registry is a dict.
# ---------------------------------------------------------------------------


def _v(name: str, col_names: tuple[str, ...], *, has_period: bool = True,
       has_time: bool = True, note: str = "") -> ParquetSpec:
    """Helper for the per-solve raw LP variable specs."""
    idx: list[str] = ["solve"]
    if has_period:
        idx.append("period")
    if has_time:
        idx.append("time")
    return ParquetSpec(
        key=name,
        category="raw",
        filename=f"{name}__*.parquet",
        columns=tuple(col_names),
        indices=tuple(idx),
        note=note or f"per-solve LP variable {name}; one parquet per cascade sub-solve.",
        producer="flextool.process_outputs.read_highs_solution.write_all_variables",
    )


REGISTRY: dict[str, ParquetSpec] = {}


# -- raw / v_*.parquet (per-solve LP decision/slack/dual variables) ----------
for _spec in (
    _v("v_flow",            ("process", "source", "sink")),
    _v("v_ramp",            ("process", "source", "sink")),
    _v("v_reserve",         ("process", "reserve", "updown", "node")),
    _v("v_state",           ("node",)),
    _v("v_online_linear",   ("process",)),
    _v("v_startup_linear",  ("process",)),
    _v("v_shutdown_linear", ("process",)),
    _v("v_online_integer",  ("process",)),
    _v("v_startup_integer", ("process",)),
    _v("v_shutdown_integer", ("process",)),
    _v("v_angle",           ("node",), note="DC power flow voltage angle; only emitted when DC PF active."),
    _v("vq_state_up",       ("node",)),
    _v("vq_state_down",     ("node",)),
    _v("vq_reserve",        ("reserve", "updown", "node_group")),
    _v("vq_inertia",        ("group",)),
    _v("vq_non_synchronous", ("group",)),
    _v("vq_state_up_group", ("group",)),
    _v("vq_capacity_margin", ("group",), has_time=False),
    _v("v_invest",          ("entity",), has_time=False),
    _v("v_divest",          ("entity",), has_time=False),
    _v("v_trade",           ("commodity", "node", "tier"), has_time=False,
       note="commodity-ladder trade; only emitted when commodity ladder active."),
    _v("v_dual_node_balance", ("node",)),
    _v("v_dual_reserve__upDown__group__period__t",
       ("reserve", "updown", "node_group")),
    _v("v_dual_invest_unit",       ("unit",), has_time=False),
    _v("v_dual_invest_connection", ("connection",), has_time=False),
    _v("v_dual_invest_node",       ("node",), has_time=False),
    _v("v_dual_maxInvest_period",       ("entity",), has_time=False,
       note="MaxInvest-per-period dual; only emitted when maxInvest_period set."),
    _v("v_dual_maxInvest_total",        ("entity",), has_time=False,
       note="MaxInvest-total dual; only emitted when maxInvest_total set."),
    _v("v_dual_maxCumulative",          ("entity",), has_time=False,
       note="Cumulative-cap dual; only emitted when maxCumulative set."),
    _v("v_dual_maxInvestGroup_period",     ("group",), has_time=False,
       note="MaxInvestGroup-period dual; only emitted when group invest cap set."),
    _v("v_dual_maxInvestGroup_total",      ("group",), has_time=False,
       note="MaxInvestGroup-total dual; only emitted when group invest cap set."),
    _v("v_dual_maxInvestGroup_cumulative", ("group",), has_time=False,
       note="MaxInvestGroup-cumulative dual; only emitted when group invest cap set."),
    _v("v_dual_co2_max_period",   ("group",), has_time=False,
       note="CO2 cap dual (per-period); only emitted when CO2 cap set."),
    _v("v_dual_co2_max_total",    ("group",), has_period=False, has_time=False,
       note="CO2 cap dual (total horizon); only emitted when total CO2 cap set."),
    _v("v_obj", ("objective",), has_period=False, has_time=False,
       note="Objective value (un-scaled); one row per solve."),
):
    REGISTRY[_spec.key] = _spec
del _spec


# -- raw / handoff capacity CSVs (legacy, kept as CSV for handoff-reader use) -
for _key, _filename, _first_col in (
    ("entity_all_capacity",  "entity_all_capacity.csv",   "entity"),
    ("unit_capacity__period",      "unit_capacity__period.csv",       "unit"),
    ("connection_capacity__period", "connection_capacity__period.csv", "connection"),
    ("node_capacity__period",      "node_capacity__period.csv",       "node"),
):
    REGISTRY[_key] = ParquetSpec(
        key=_key,
        category="raw",
        filename=_filename,
        columns=(_first_col, "solve", "period", "existing", "invested", "divested", "total"),
        indices=(_first_col, "solve", "period"),
        note=(
            "Capacity handoff CSV (NOT parquet) — appended across rolls "
            "by handoff_writers; consumed by downstream readers in CSV form."
        ),
        producer="flextool.process_outputs.handoff_writers._write_capacity_per_period",
    )
del _key, _filename, _first_col


# -- processed / output_parquet/<scenario>/*.parquet -------------------------
# These are emitted by ``write_outputs`` when ``parquet`` ∈ ``write_methods``.
# Per-scenario subdir; see ``write_outputs.py:670-678`` for path construction.
# Wide / long shapes carry MultiIndex column / row layouts that we don't
# enumerate in detail (``columns`` left as ``()`` — "any columns").

def _proc(key: str, *, note: str, indices: tuple[str, ...] = (),
          columns: tuple[str, ...] = (),
          producer: str) -> ParquetSpec:
    return ParquetSpec(
        key=key, category="processed",
        filename=f"{key}.parquet",
        columns=columns, indices=indices,
        note=note, producer=producer,
    )


# Cost-related outputs
for _key, _producer, _note in (
    ("discountFactors_d_p",        "out_costs.generic",        "Per-period discount factors (operations / investment)."),
    ("entity_annuity_d_p",         "out_costs.generic",        "Per-entity annuity rows (when invest entities present)."),
    ("costs_dt_p",                 "out_costs.cost_summaries", "Per-(period,time) cost decomposition."),
    ("annualized_costs_d_p",       "out_costs.cost_summaries", "Annualized cost summary by period."),
    ("costs_discounted_d_p",       "out_costs.cost_summaries", "Discounted cost summary by period."),
    ("costs_discounted_p_",        "out_costs.cost_summaries", "Discounted cost total over horizon."),
    ("CO2__",                      "out_costs.CO2",            "System-wide CO2 totals."),
    ("CO2_d_g",                    "out_costs.CO2",            "Per-(period,group) CO2 emissions."),
    ("process_co2_d_eee",          "out_costs.CO2",            "Per-process CO2 contribution; only when CO2-emitting commodities present."),
    ("co2_price_period_d_g",       "out_ancillary.co2_duals",  "CO2 price per period; only when CO2 cap active."),
    ("co2_price_total_d_g",        "out_ancillary.co2_duals",  "CO2 price (total horizon); only when total CO2 cap active."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Capacity outputs
for _key, _producer, _note in (
    ("unit_capacity_ed_p",        "out_capacity.unit_capacity",       "Per-(unit, period) capacity decomposition."),
    ("connection_capacity_ed_p",  "out_capacity.connection_capacity", "Per-(connection, period) capacity decomposition."),
    ("node_capacity_ed_p",        "out_capacity.node_capacity",       "Per-(node, period) capacity decomposition; only for nodeState nodes."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Node / dispatch outputs
for _key, _producer, _note in (
    ("node_dt_ep",                 "out_node.node_summary",            "Per-(period,time) node balance."),
    ("node_d_ep",                  "out_node.node_summary",            "Per-period aggregated node balance."),
    ("node_state_dt_e",            "out_node.node_additional_results", "Per-(period,time) node state."),
    ("node_inflow__dt",            "out_node.node_additional_results", "Per-(period,time) node inflow."),
    ("node_prices_dt_e",           "out_node.node_additional_results", "Per-(period,time) node price (dual_node_balance)."),
    ("node_slack_up_dt_e",         "out_ancillary.slack_variables",    "Per-(period,time) node up-slack."),
    ("node_slack_down_dt_e",       "out_ancillary.slack_variables",    "Per-(period,time) node down-slack."),
    ("node_slack_up_d_e",          "out_ancillary.slack_variables",    "Per-period node up-slack."),
    ("node_slack_down_d_e",        "out_ancillary.slack_variables",    "Per-period node down-slack."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Unit-flow outputs
for _key, _producer, _note in (
    ("unit_outputNode_dt_ee",      "out_flows.unit_outputNode",        "Per-(period,time) unit→node flow."),
    ("unit_outputNode_d_ee",       "out_flows.unit_outputNode",        "Per-period unit→node flow."),
    ("unit_inputNode_dt_ee",       "out_flows.unit_inputNode",         "Per-(period,time) node→unit flow."),
    ("unit_inputNode_d_ee",        "out_flows.unit_inputNode",         "Per-period node→unit flow."),
    ("unit_VRE_potential_outputNode_dt_ee",     "out_flows.unit_VRE_curtailment_and_potential", "VRE potential; only with VRE units."),
    ("unit_VRE_potential_outputNode_d_ee",      "out_flows.unit_VRE_curtailment_and_potential", "VRE potential per period; only with VRE units."),
    ("unit_curtailment_outputNode_dt_ee",       "out_flows.unit_VRE_curtailment_and_potential", "VRE curtailment; only with VRE units."),
    ("unit_curtailment_outputNode_d_ee",        "out_flows.unit_VRE_curtailment_and_potential", "VRE curtailment per period; only with VRE units."),
    ("unit_curtailment_share_outputNode_dt_ee", "out_flows.unit_VRE_curtailment_and_potential", "VRE curtailment share; only with VRE units."),
    ("unit_curtailment_share_outputNode_d_ee",  "out_flows.unit_VRE_curtailment_and_potential", "VRE curtailment share per period; only with VRE units."),
    ("unit_ramp_inputs_dt_ee",     "out_flows.unit_ramps",             "Per-(period,time) input-side unit ramp; only with ramp limits set."),
    ("unit_ramp_outputs_dt_ee",    "out_flows.unit_ramps",             "Per-(period,time) output-side unit ramp; only with ramp limits set."),
    ("unit_online_dt_e",           "out_flows.unit_online_and_startup", "Per-(period,time) unit online state; only with online tracking."),
    ("unit_online_average_d_e",    "out_flows.unit_online_and_startup", "Per-period unit online average; only with online tracking."),
    ("unit_startup_d_e",           "out_flows.unit_online_and_startup", "Per-period unit startup count; only with online tracking."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Connection-flow + DC PF outputs
for _key, _producer, _note in (
    ("connection_dt_eee",            "out_ancillary.connection",         "Per-(period,time) net connection flow."),
    ("connection_d_eee",             "out_ancillary.connection",         "Per-period net connection flow."),
    ("connection_leftward_dt_eee",   "out_ancillary.connection",         "Per-(period,time) leftward connection flow."),
    ("connection_leftward_d_eee",    "out_ancillary.connection",         "Per-period leftward connection flow."),
    ("connection_rightward_dt_eee",  "out_ancillary.connection",         "Per-(period,time) rightward connection flow."),
    ("connection_rightward_d_eee",   "out_ancillary.connection",         "Per-period rightward connection flow."),
    ("connection_losses_dt_eee",     "out_ancillary.connection",         "Per-(period,time) connection losses."),
    ("connection_losses_d_eee",      "out_ancillary.connection",         "Per-period connection losses."),
    ("connection_dc_power_flow",     "out_ancillary.dc_power_flow",      "DC power flow per connection; only with DC PF active."),
    ("node_dc_power_flow",           "out_ancillary.dc_power_flow",      "DC power flow per node; only with DC PF active."),
    ("dc_angle_dt_e",                "out_ancillary.dc_power_flow",      "DC voltage angle per (period,time); only with DC PF active."),
    ("dc_angle_diff_dt_e",           "out_ancillary.dc_power_flow",      "DC angle difference per branch; only with DC PF active."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Reserve / inertia / capacity-margin outputs
for _key, _producer, _note in (
    ("process_reserve_upDown_node_dt_eppe", "out_ancillary.reserves",   "Per-(period,time) reserve provision; only with reserves active."),
    ("process_reserve_average_d_eppe",      "out_ancillary.reserves",   "Per-period average reserve provision; only with reserves active."),
    ("reserve_prices_dt_ppg",               "out_ancillary.reserves",   "Per-(period,time) reserve price (dual); only with reserves active."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Investment-dual outputs
for _key, _producer, _note in (
    ("dual_invest_unit_d_e",             "out_ancillary.investment_duals", "Per-period investment dual for units."),
    ("dual_invest_connection_d_e",       "out_ancillary.investment_duals", "Per-period investment dual for connections."),
    ("dual_invest_node_d_e",             "out_ancillary.investment_duals", "Per-period investment dual for nodes."),
    ("dual_invest_effective_unit_d_e",        "out_ancillary.investment_duals", "Effective investment dual for units."),
    ("dual_invest_effective_connection_d_e",  "out_ancillary.investment_duals", "Effective investment dual for connections."),
    ("dual_invest_effective_node_d_e",        "out_ancillary.investment_duals", "Effective investment dual for nodes."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Group-level dispatch / VRE-share / inertia / slack outputs
for _key, _producer, _note in (
    ("nodeGroup_gdt_p",                  "out_group.nodeGroup_indicators",   "Per-(group, period, time) indicators."),
    ("nodeGroup_gd_p",                   "out_group.nodeGroup_indicators",   "Per-(group, period) indicators."),
    ("nodeGroup_VRE_share_dt_g",         "out_group.nodeGroup_VRE_share",    "Per-(period,time) VRE share by group."),
    ("nodeGroup_VRE_share_d_g",          "out_group.nodeGroup_VRE_share",    "Per-period VRE share by group."),
    ("nodeGroup_flows_dt_g",             "out_group.nodeGroup_flows",        "Per-(period,time) group net flow."),
    ("nodeGroup_flows_d_g",              "out_group.nodeGroup_flows",        "Per-period group net flow."),
    ("nodeGroup_flows_dt_gpe",           "out_group.nodeGroup_flows",        "Per-(period,time) group flow by participant."),
    ("nodeGroup_flows_d_gpe",            "out_group.nodeGroup_flows",        "Per-period group flow by participant."),
    ("nodeGroup_inertia_dt_g",           "out_ancillary.inertia_results",    "Per-(period,time) group inertia; only with inertia constraint active."),
    ("nodeGroup_inertia_largest_flow_dt_g",        "out_ancillary.inertia_results", "Largest-flow inertia decomposition; only with N-1 / dynamic reserves."),
    ("nodeGroup_unit_node_inertia_dt_gee",         "out_ancillary.inertia_results", "Per-unit inertia contribution; only with inertia constraint."),
    ("nodeGroup_slack_capacity_margin_d_g", "out_ancillary.slack_variables", "Per-period capacity-margin slack by group."),
    ("nodeGroup_slack_inertia_d_g",         "out_ancillary.slack_variables", "Per-period inertia slack by group."),
    ("nodeGroup_slack_inertia_dt_g",        "out_ancillary.slack_variables", "Per-(period,time) inertia slack."),
    ("nodeGroup_slack_nonsync_d_g",         "out_ancillary.slack_variables", "Per-period non-sync slack."),
    ("nodeGroup_slack_nonsync_dt_g",        "out_ancillary.slack_variables", "Per-(period,time) non-sync slack."),
    ("nodeGroup_slack_reserve_d_eeg",       "out_ancillary.slack_variables", "Per-period reserve slack."),
    ("nodeGroup_slack_reserve_dt_eeg",      "out_ancillary.slack_variables", "Per-(period,time) reserve slack."),
    ("nodeGroup_total_inflow",              "out_group.nodeGroup_total_inflow", "Total per-period node-group inflow."),
    ("flowGroup_gd_p",                      "out_flowgroup.flowGroup_indicators", "Per-(period) flow-group indicators; only with flow groups configured."),
    ("flowGroupIndicators",                 "out_ancillary.input_sets",     "Set of flow groups configured for indicator output."),
    ("nodeGroupIndicators",                 "out_ancillary.input_sets",     "Set of node groups configured for indicator output."),
    ("nodeGroupDispatch",                   "out_ancillary.input_sets",     "Set of node groups configured for dispatch output."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note

# Misc input-side sets propagated to processed outputs
for _key, _producer, _note in (
    ("years_represented__d",      "out_ancillary.input_sets", "Years represented per period (annualisation factor)."),
    ("group_node",                "out_ancillary.input_sets", "Group → node membership set."),
    ("group_process",             "out_ancillary.input_sets", "Group → process membership set."),
    ("group_process_node",        "out_ancillary.input_sets", "Group → (process, node) membership set."),
):
    REGISTRY[_key] = _proc(
        _key, note=_note, producer=f"flextool.process_outputs.{_producer}",
    )
del _key, _producer, _note


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_parquet(spec_key: str, frame, work_folder: Path | str) -> Path:
    """Validate ``frame`` against ``REGISTRY[spec_key]`` and write parquet.

    Parameters
    ----------
    spec_key
        Registry key.  Must be present in :data:`REGISTRY`.
    frame
        ``polars.DataFrame``-like object exposing ``.columns`` and
        ``.write_parquet(path)``.  We accept duck-typed input so the
        function works for both polars frames and any future writer
        adapter.
    work_folder
        Bundle root.  File lands at
        ``work_folder / spec.category / spec.filename``.  When the
        spec's ``filename`` is a glob (contains ``*``), we error: the
        typed write path is for single files only; per-solve shards are
        managed by their existing writers.

    Returns
    -------
    Path
        Destination path written.

    Raises
    ------
    KeyError
        ``spec_key`` not in REGISTRY.
    ValueError
        ``frame``'s columns don't match ``spec.columns`` (when
        ``spec.columns`` is non-empty).  Also raised when the spec's
        ``filename`` is a glob pattern.
    """
    if spec_key not in REGISTRY:
        raise KeyError(
            f"write_parquet: unknown spec key {spec_key!r}; add an entry "
            f"to REGISTRY in flextool/engine_polars/_parquet_bundle.py.",
        )
    spec = REGISTRY[spec_key]
    if "*" in spec.filename:
        raise ValueError(
            f"write_parquet: spec {spec_key!r} uses a glob filename "
            f"({spec.filename!r}); per-shard writes belong to their existing "
            f"writers (e.g. read_highs_solution.write_all_variables).  "
            f"Use this typed path only for single-file outputs.",
        )

    if spec.columns:
        # Tolerate polars (``.columns`` is a list[str]) and pandas
        # (``.columns`` is an Index).  Normalise to a tuple of strings.
        actual = tuple(str(c) for c in frame.columns)
        if set(actual) != set(spec.columns):
            raise ValueError(
                f"write_parquet: frame columns {actual!r} do not match "
                f"spec {spec_key!r} columns {spec.columns!r}.",
            )

    work_folder = Path(work_folder)
    # Use the FIRST configured directory for that category (canonical
    # write target).  ``write_manifest`` searches all configured dirs.
    subdirs = CATEGORY_DIRS.get(spec.category, (spec.category,))
    out_dir = work_folder / subdirs[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / spec.filename

    # Best-effort polars-style write; fall back to pandas-style.
    if hasattr(frame, "write_parquet"):
        frame.write_parquet(str(out_path))
    elif hasattr(frame, "to_parquet"):
        frame.to_parquet(str(out_path))
    else:
        raise TypeError(
            f"write_parquet: frame of type {type(frame).__name__} exposes "
            f"neither .write_parquet nor .to_parquet.",
        )
    return out_path


def _enumerate_files(work_folder: Path) -> list[dict]:
    """Walk REGISTRY, resolve each spec to zero-or-more on-disk files.

    Returns one entry per file present.  Glob specs (``filename``
    containing ``*``) expand to one entry per matching shard.  Non-glob
    specs always produce exactly one entry (``exists`` flags presence).
    """
    files: list[dict] = []
    for spec in REGISTRY.values():
        # Try every directory configured for this category.  For "raw"
        # there's only one (``output_raw``); for "processed" the file
        # may be under any of ``output_parquet`` / ``output_csv`` /
        # ``output_excel`` depending on ``--write-methods``.  We collect
        # all matches across configured dirs.
        subdirs = CATEGORY_DIRS.get(spec.category, (spec.category,))
        if "*" in spec.filename:
            shards: list[Path] = []
            for sub in subdirs:
                d = work_folder / sub
                if d.is_dir():
                    shards.extend(sorted(d.glob(spec.filename)))
            if not shards:
                # Document the absent variable so readers know to expect
                # zero shards (as opposed to "this spec doesn't exist").
                files.append(_file_entry(spec, subdirs[0], None, exists=False))
                continue
            for shard in shards:
                # Path relative to work_folder gives the consumer enough
                # context to find the file.
                rel_dir = shard.parent.relative_to(work_folder).as_posix()
                files.append(_file_entry(spec, rel_dir, shard, exists=True))
        else:
            found = None
            for sub in subdirs:
                candidate = work_folder / sub / spec.filename
                if candidate.is_file():
                    found = (sub, candidate)
                    break
            if found is not None:
                files.append(_file_entry(spec, found[0], found[1], exists=True))
            else:
                files.append(_file_entry(spec, subdirs[0], None, exists=False))
    return files


def _file_entry(spec: ParquetSpec, subdir: str,
                resolved: Path | None, *, exists: bool) -> dict:
    """Build one manifest ``files[]`` entry.

    ``subdir`` is the on-disk subdirectory (relative to the bundle root)
    where the file lives — e.g. ``"output_raw"`` or ``"output_parquet"``.
    """
    if resolved is None:
        rel_path = f"{subdir}/{spec.filename}"
        size = None
    else:
        rel_path = f"{subdir}/{resolved.name}"
        try:
            size = resolved.stat().st_size if exists else None
        except OSError:
            size = None
    return {
        "key": spec.key,
        "category": spec.category,
        "path": rel_path,
        "exists": exists,
        "size_bytes": size,
        "columns": list(spec.columns),
        "indices": list(spec.indices),
        "note": spec.note,
        "producer": spec.producer,
    }


def _registry_filenames(category: str) -> set[str]:
    """Set of REGISTRY ``filename`` basenames for a category (glob-aware)."""
    out: set[str] = set()
    for spec in REGISTRY.values():
        if spec.category != category:
            continue
        out.add(spec.filename)
    return out


def _is_known(name: str, registered: set[str]) -> bool:
    """Match a basename against either an exact filename or a glob pattern."""
    import fnmatch
    for pattern in registered:
        if "*" in pattern:
            if fnmatch.fnmatch(name, pattern):
                return True
        elif name == pattern:
            return True
    return False


def _warn_coverage_gaps(work_folder: Path, files: list[dict]) -> None:
    """Log warnings for:
       1. files in ``output_raw/`` with no matching REGISTRY entry,
       2. unconditional REGISTRY entries whose file is missing.

    Both are advisory — the manifest is still written.  ``output_processed``
    is intentionally NOT walked here: its layout depends on
    ``output_location`` + ``--write-methods``, which the bundle module
    doesn't see.
    """
    raw_dir = work_folder / "output_raw"
    if raw_dir.is_dir():
        registered = _registry_filenames("raw")
        for entry in os.listdir(raw_dir):
            if entry.startswith("."):
                continue
            full = raw_dir / entry
            if not full.is_file():
                continue
            if not _is_known(entry, registered):
                _logger.warning(
                    "manifest: file in output_raw/ with no REGISTRY entry: %s",
                    entry,
                )

    note_conditional = ("only when", "only with", "only for", "only emitted",
                        "conditional")
    for record in files:
        if record["exists"]:
            continue
        note = record["note"].lower()
        if any(tok in note for tok in note_conditional):
            continue
        # Skip warnings for processed/* — coverage there is incomplete by design.
        if record["category"] == "processed":
            continue
        # Skip when the entire raw dir is missing (run aborted before
        # any output was emitted) — every entry would warn redundantly.
        subdirs = CATEGORY_DIRS.get(record["category"], (record["category"],))
        if not any((work_folder / sub).is_dir() for sub in subdirs):
            continue
        _logger.warning(
            "manifest: registered file missing on disk: %s "
            "(REGISTRY key %r)", record["path"], record["key"],
        )


def write_manifest(work_folder: Path | str) -> Path:
    """Emit ``work_folder / manifest.json`` describing the output bundle.

    Idempotent — safe to call multiple times.  Each call regenerates
    the file from the current on-disk state.

    The manifest schema is::

        {
          "version": "1",
          "generated_at": "<iso8601 utc>",
          "bundle_root": "<absolute work_folder path>",
          "files": [
            {
              "key": "<registry key>",
              "category": "raw" | "processed",
              "path": "<category>/<basename>",
              "exists": true | false,
              "size_bytes": <int> | null,
              "columns": [...],
              "indices": [...],
              "note": "<human description>",
              "producer": "<module.function path>"
            },
            ...
          ]
        }

    Per the spec, the manifest only documents files that ARE present
    on disk (``exists: true``).  Missing files are still listed (with
    ``exists: false``) when their REGISTRY entry is unconditional, so
    a reader can distinguish "we didn't run that variable" from "we
    ran it and emitted a real file".  Conditional entries (those with
    ``"only when"`` / ``"only with"`` / ``"only emitted"`` /
    ``"only for"`` in ``note``) silently skip when absent.
    """
    work_folder = Path(work_folder)
    files = _enumerate_files(work_folder)
    _warn_coverage_gaps(work_folder, files)

    manifest = {
        "version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(work_folder.resolve()),
        "files": files,
    }

    manifest_path = work_folder / "manifest.json"
    work_folder.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    _logger.debug(
        "wrote manifest %s (%d entries; %d present on disk)",
        manifest_path, len(files), sum(1 for f in files if f["exists"]),
    )
    return manifest_path


__all__ = ["ParquetSpec", "REGISTRY", "write_parquet", "write_manifest"]
