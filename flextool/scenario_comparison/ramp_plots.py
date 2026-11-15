"""Ramp envelope plots: the envelope over time and the ramp histogram (#25).

The tables are ``node_ramp_envelope_dt_ep`` / ``nodeGroup_ramp_envelope_dt_gp`` (``(period, time)`` rows,
``(entity, band)`` columns per scenario; row t is the transition t → t⁺, MW), ``timestep_hours_dt`` (hours a year
of each step, the histogram weight) and ``transition_hours_dt`` ((Δt + Δt⁺)/2 of each row's transition, h).

* Envelope (:func:`build_ramp_envelope_figure`): the up bands stacked above 0, the down bands below (one colour
  per band, :func:`ramp_band_colors`), the net-load ramp as a black line, the demand ramp as a grey line; empty
  rows (no single next step) and timeline breaks are gaps.
* Histogram (:func:`build_ramp_histogram_figure`), per entity and period: the demand rate and the net-load rate
  x = line / ((Δt + Δt⁺)/2) [MW/h] of every compared scenario as step histograms weighted by the hours a year
  [h/year], on one bin set per plot (:func:`ramp_histogram_bins`, spec §2.11); vertical lines at the smallest
  up room and the smallest down room over the shown rows of every scenario, per transition divided by the same
  (Δt + Δt⁺)/2 (:func:`room_lines`).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from flextool import units_of_measure as uom
from flextool.plot_outputs.format_helpers import (
    has_timeline_breaks,
    insert_timeline_breaks,
    write_plot_data_csv,
)
from flextool.process_outputs.calc_ramp_envelope import DOWN_BANDS, UP_BANDS
from flextool.scenario_comparison.constants import DEFAULT_RAMP_BAND_COLORS
from flextool.safe_names import safe_file_component

logger = logging.getLogger(__name__)

#: The band keys of the colours (``categories.ramp_envelope`` of plot_settings.yaml), in stacking order.
RAMP_BAND_KEYS = ("units", "storage", "vre", "connections")
#: Legend label of each band key.
RAMP_BAND_LABELS = {"units": "Units", "storage": "Storage", "vre": "VRE", "connections": "Connections"}
#: The two histogram series: (band of the envelope table, legend label).
RATE_SERIES = (("ramp_net_load", "net load"), ("ramp_demand", "demand"))
#: The colour of the net-load and the demand line of the envelope plot.
NET_LOAD_COLOR = "black"
DEMAND_COLOR = "dimgray"
#: The y-axis note of the envelope plot (row t is the change from t toward the next step).
ENVELOPE_AXIS_NOTE = "change toward the next step"
#: Default number of bins across [−M, M] the bin width is rounded up from (spec §2.11: w ≥ 2M/40).
_BIN_DIVISOR = 40.0
_REL_TOL = 1e-9


# ── Data ─────────────────────────────────────────────────────────────────────────────────────────────────────


def entity_kind_table(kind: str) -> str:
    """The envelope table of an entity kind (``"node"`` / ``"group"``)."""
    return "node_ramp_envelope_dt_ep" if kind == "node" else "nodeGroup_ramp_envelope_dt_gp"


def scenario_frame(results, name: str, scenario: str) -> pd.DataFrame | None:
    """The frame *name* of *scenario* (its own rows, scenario level dropped), ``None`` when absent."""
    from flextool.scenario_comparison.dispatch_data import _scenario_rows

    df = getattr(results, name, None)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex) and "scenario" in df.columns.names:
        if scenario not in df.columns.get_level_values("scenario"):
            return None
        return _scenario_rows(df, scenario, results.own_rows(name, scenario))
    return df


def entity_envelope(results, kind: str, entity: str, scenario: str) -> pd.DataFrame | None:
    """The envelope of one node / group in one scenario: ``(period, time)`` rows, the band columns; ``None``
    when the scenario has no envelope for it."""
    df = scenario_frame(results, entity_kind_table(kind), scenario)
    if df is None or entity not in df.columns.get_level_values(0):
        return None
    return df.xs(entity, axis=1, level=0)


def transition_hours(results, scenario: str, index: pd.Index) -> pd.Series:
    """(Δt + Δt⁺)/2 of each row of *index* (h); NaN where unknown (results written before the table existed)."""
    df = scenario_frame(results, "transition_hours_dt", scenario)
    if df is None or df.shape[1] == 0:
        return pd.Series(np.nan, index=index)
    return df.iloc[:, 0].reindex(index)


def timestep_hours(results, scenario: str, index: pd.Index) -> pd.Series:
    """The hours a year of each row of *index* (the histogram weight); NaN where unknown."""
    df = scenario_frame(results, "timestep_hours_dt", scenario)
    if df is None or df.shape[1] == 0:
        return pd.Series(np.nan, index=index)
    return df.iloc[:, 0].reindex(index)


def ramp_rates(envelope: pd.DataFrame, transition: pd.Series) -> pd.DataFrame:
    """The net-load and demand rates of each row, line / ((Δt + Δt⁺)/2) [MW/h] (columns ``ramp_net_load``,
    ``ramp_demand``); empty where the row is empty."""
    lines = envelope.reindex(columns=[band for band, _ in RATE_SERIES])
    return lines.div(transition.reindex(envelope.index), axis=0)


# ── Histogram ────────────────────────────────────────────────────────────────────────────────────────────────


def _bin_width(m: float) -> float:
    """The smallest 1·10ᵏ, 2·10ᵏ or 5·10ᵏ that is ≥ 2M/40."""
    target = 2.0 * m / _BIN_DIVISOR
    k = math.floor(math.log10(target))
    candidates = np.array([mult * 10.0 ** e for e in (k - 1, k, k + 1) for mult in (1.0, 2.0, 5.0)])
    return float(candidates[candidates >= target * (1.0 - _REL_TOL)].min())


def ramp_histogram_bins(series) -> np.ndarray | None:
    """The bin edges of one histogram plot (spec §2.11), shared by every series in *series* (iterables of rates;
    NaN dropped).

    M = max |x| over all of them; w = the smallest 1·10ᵏ, 2·10ᵏ or 5·10ᵏ ≥ 2M/40; bins k = −K…K, bin k =
    [(k − ½)·w, (k + ½)·w) with K = ⌈M/w − ½⌉, so one bin is centred on 0 and the bins span at least [−M, M]; the
    last bin is closed on both sides (:func:`ramp_histogram`).  M = 0: one bin [−½, ½].  No values: ``None``."""
    parts = [np.asarray(s, dtype=float).ravel() for s in series]
    values = np.concatenate(parts) if parts else np.zeros(0)
    values = values[~np.isnan(values)]
    if not len(values):
        return None
    m = float(np.abs(values).max())
    if m == 0.0:
        return np.array([-0.5, 0.5])
    w = _bin_width(m)
    n_half = max(0, math.ceil(m / w - 0.5 - _REL_TOL))
    return (np.arange(-n_half, n_half + 2) - 0.5) * w


def ramp_histogram(values, weights, edges: np.ndarray) -> np.ndarray:
    """Σ weight of the values in each bin of *edges* (bins closed on the left, open on the right, the last bin
    closed on both sides); rows with a NaN value or weight are dropped."""
    x = np.asarray(values, dtype=float).ravel()
    wgt = np.asarray(weights, dtype=float).ravel()
    keep = ~np.isnan(x) & ~np.isnan(wgt)
    n_bins = len(edges) - 1
    at = np.clip(np.searchsorted(edges, x[keep], side="right") - 1, 0, n_bins - 1)
    return np.bincount(at, weights=wgt[keep], minlength=n_bins).astype(float)


def room_lines(envelopes, transitions) -> tuple[float, float]:
    """``(up, down)``: the smallest up room (Σ up bands) and the smallest down room (Σ down bands, negative) over
    the filled rows of every envelope, each row divided by its (Δt + Δt⁺)/2 [MW/h]; NaN without rows."""
    ups, downs = [], []
    for env, trans in zip(envelopes, transitions):
        if env is None or env.empty:
            continue
        filled = env.reindex(columns=list(UP_BANDS) + list(DOWN_BANDS)).notna().any(axis=1)
        div = trans.reindex(env.index)
        up = env.reindex(columns=list(UP_BANDS)).sum(axis=1, min_count=1).div(div)
        down = env.reindex(columns=list(DOWN_BANDS)).sum(axis=1, min_count=1).div(div)
        ups.append(up[filled].to_numpy(dtype=float))
        downs.append(down[filled].to_numpy(dtype=float))
    up_all = np.concatenate(ups) if ups else np.zeros(0)
    down_all = np.concatenate(downs) if downs else np.zeros(0)
    up_all, down_all = up_all[~np.isnan(up_all)], down_all[~np.isnan(down_all)]
    return (float(up_all.min()) if len(up_all) else math.nan,
            float(down_all.max()) if len(down_all) else math.nan)


def histogram_data(envelopes: dict, transitions: dict, weights: dict) -> dict:
    """The histograms of one entity, per period: ``{period: {"edges", "hours" {(scenario, series): array},
    "room" (up, down)}}``.  *envelopes* / *transitions* / *weights* are keyed by scenario (one entity); a period
    without any rate gets no entry."""
    periods: list = []
    for env in envelopes.values():
        if env is not None:
            for d in env.index.get_level_values("period"):
                if d not in periods:
                    periods.append(d)
    out = {}
    for period in periods:
        rates, envs, trans, wgts = {}, [], [], {}
        for scenario, env in envelopes.items():
            if env is None:
                continue
            sel = env.index.get_level_values("period") == period
            if not sel.any():
                continue
            env_d = env.loc[sel]
            trans_d = transitions[scenario].reindex(env_d.index)
            rates[scenario] = ramp_rates(env_d, trans_d)
            wgts[scenario] = weights[scenario].reindex(env_d.index)
            envs.append(env_d)
            trans.append(trans_d)
        edges = ramp_histogram_bins([r[band] for r in rates.values() for band, _ in RATE_SERIES])
        if edges is None:
            continue
        hours = {(scenario, band): ramp_histogram(rate[band], wgts[scenario], edges)
                 for scenario, rate in rates.items() for band, _ in RATE_SERIES}
        out[period] = {"edges": edges, "hours": hours, "room": room_lines(envs, trans)}
    return out


# ── Colours and labels ───────────────────────────────────────────────────────────────────────────────────────


def ramp_band_colors(template: dict | None) -> dict:
    """The colour of each band key: ``categories.ramp_envelope`` of the plot-settings *template*, else
    :data:`~flextool.scenario_comparison.constants.DEFAULT_RAMP_BAND_COLORS`."""
    from flextool.plot_outputs.color_template import resolve_label_color

    out = {}
    for key in RAMP_BAND_KEYS:
        color = resolve_label_color(key, template or {}, category="ramp_envelope")
        out[key] = color if color is not None else DEFAULT_RAMP_BAND_COLORS[key]
    return out


def ramp_rate_unit(ctx, *, group: str | None = None, node: str | None = None) -> str:
    """The rate unit of the node or the group's members (``MW`` for MWh nodes; mixed for a group over nodes of
    different units)."""
    if ctx is None or ctx.is_default:
        return uom.fill("{R}", uom.DEFAULT_A)
    nodes = (node,) if node is not None else tuple(ctx.group_nodes.get(str(group), ()))
    return uom.fill("{R}", uom.units_of(nodes, ctx.node_units))


def _per_hour(unit: str) -> str:
    return unit if unit == uom.MIXED_UOM else f"{unit}/h"


# ── Figures ──────────────────────────────────────────────────────────────────────────────────────────────────


def envelope_plot_frame(envelope: pd.DataFrame, timeline: tuple[int, int]) -> pd.DataFrame:
    """The drawn data of one envelope plot: the rows of the window, the band columns and the two lines."""
    return envelope.reindex(columns=[*UP_BANDS, *DOWN_BANDS, "ramp_net_load", "ramp_demand"]).iloc[
        timeline[0]:timeline[1]]


def build_ramp_envelope_figure(
    envelope: pd.DataFrame,
    title: str,
    unit: str = "MW",
    colors: dict | None = None,
    timeline: tuple[int, int] = (0, 168),
    break_times: pd.MultiIndex | None = None,
    keep_plot_data: bool = False,
) -> Figure | None:
    """The envelope of one entity in one scenario over the rows ``timeline`` (positions); ``None`` when the
    window has no filled row.  With *keep_plot_data* the drawn frame is kept for
    :func:`~flextool.plot_outputs.format_helpers.write_plot_data_csv`."""
    from flextool.scenario_comparison.dispatch_plots import _set_step_ticks

    colors = colors or ramp_band_colors(None)
    data = envelope_plot_frame(envelope, timeline)
    if data.empty or data.isna().all().all():
        return None
    plot = insert_timeline_breaks(data, break_times) if has_timeline_breaks(break_times) else data
    fig = Figure(figsize=(12.0, 4.0))
    ax = fig.add_axes([0.08, 0.2, 0.74, 0.7])
    _draw_envelope_steps(ax, plot, colors)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.5)
    _set_step_ticks(ax, plot.index)
    ax.set_ylabel(f"{unit}, {ENVELOPE_AXIS_NOTE}")
    ax.set_title(title)
    ax.set_xlim(0, len(plot))
    handles, labels = ax.get_legend_handles_labels()
    lines = [(h, lbl) for h, lbl in zip(handles, labels) if lbl in ("Net load ramp", "Demand ramp")]
    areas = [(h, lbl) for h, lbl in zip(handles, labels) if lbl not in ("Net load ramp", "Demand ramp")]
    ordered = lines + areas[::-1]
    ax.legend([h for h, _ in ordered], [lbl for _, lbl in ordered], bbox_to_anchor=(1.02, 1), loc="upper left")
    if keep_plot_data:
        fig.flextool_plot_batch = [(None, data)]
    return fig


def _draw_envelope_steps(ax, plot: pd.DataFrame, colors: dict) -> None:
    """Draw the envelope rows as steps: row i (the transition from step i to the next) spans x ∈ [i, i + 1).  The
    up bands stack above 0 and the down bands below it in band order (units nearest the axis), one colour per
    band; the net-load and the demand ramp are lines.  An empty row (no single next step, a timeline-break gap
    row) is a gap."""
    n = len(plot)
    x = np.arange(n + 1)
    drawn = plot[[*UP_BANDS, *DOWN_BANDS]].notna().any(axis=1).to_numpy()
    # A row's step runs to x = i + 1: the point after a drawn row is in its region too.
    where = np.append(drawn, False) | np.append(False, drawn)

    def ext(values: np.ndarray) -> np.ndarray:
        return np.append(values, values[-1] if n else 0.0)

    for bands, legend in ((UP_BANDS, True), (DOWN_BANDS, False)):
        base = np.zeros(n)
        for band in bands:
            key = band.rsplit("_", 1)[0]
            top = base + plot[band].fillna(0.0).to_numpy(dtype=float)
            ax.fill_between(x, ext(base), ext(top), where=where, step="post", color=colors[key], linewidth=0,
                            label=RAMP_BAND_LABELS[key] if legend else None)
            base = top
    for band, color, width, label in (("ramp_net_load", NET_LOAD_COLOR, 1.5, "Net load ramp"),
                                      ("ramp_demand", DEMAND_COLOR, 1.0, "Demand ramp")):
        ax.step(x, ext(plot[band].to_numpy(dtype=float)), where="post", color=color, linewidth=width, label=label)


def histogram_plot_frame(period_data: dict) -> pd.DataFrame:
    """The drawn data of one histogram subplot: one row per bin (``bin_from``, ``bin_to``), the hours a year of
    each ``scenario | series`` and the two room lines."""
    edges = period_data["edges"]
    index = pd.MultiIndex.from_arrays([edges[:-1], edges[1:]], names=["bin_from", "bin_to"])
    labels = dict(RATE_SERIES)
    frame = pd.DataFrame({f"{scenario} | {labels[band]} [h/year]": hours
                          for (scenario, band), hours in period_data["hours"].items()}, index=index)
    up, down = period_data["room"]
    frame["smallest up room"] = up
    frame["smallest down room"] = down
    return frame


def build_ramp_histogram_figure(
    data: dict,
    title: str,
    unit: str = "MW",
    scenario_colors: dict | None = None,
    keep_plot_data: bool = False,
) -> Figure | None:
    """The ramp histograms of one entity (:func:`histogram_data`): one subplot per period, a step histogram per
    scenario and series (net load solid, demand dashed), the room lines dotted.  ``None`` without data."""
    if not data:
        return None
    import matplotlib.pyplot as plt

    periods = list(data)
    scenarios: list = []
    for period_data in data.values():
        for scenario, _band in period_data["hours"]:
            if scenario not in scenarios:
                scenarios.append(scenario)
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {s: (scenario_colors or {}).get(s) or palette[i % 10] for i, s in enumerate(scenarios)}
    fig = Figure(figsize=(10.0, 3.2 * len(periods) + 0.8))
    axes = fig.subplots(len(periods), 1, squeeze=False)[:, 0]
    rate_unit = _per_hour(unit)
    for ax, period in zip(axes, periods):
        period_data = data[period]
        edges = period_data["edges"]
        for scenario in scenarios:
            for band, label in RATE_SERIES:
                hours = period_data["hours"].get((scenario, band))
                if hours is None:
                    continue
                name = f"{scenario}: {label}" if len(scenarios) > 1 else label
                ax.stairs(hours, edges, color=colors[scenario], linewidth=1.5,
                          linestyle="solid" if band == "ramp_net_load" else "dashed", label=name)
        up, down = period_data["room"]
        if not math.isnan(up):
            ax.axvline(up, color="gray", linestyle=":", linewidth=1.2, label=f"smallest up room {up:.4g}")
        if not math.isnan(down):
            ax.axvline(down, color="gray", linestyle="-.", linewidth=1.2, label=f"smallest down room {down:.4g}")
        ax.set_title(f"{title} — {period}" if len(periods) > 1 else title)
        ax.set_xlabel(f"Ramp rate toward the next step ({rate_unit})")
        ax.set_ylabel("h/year")
        ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    if keep_plot_data:
        fig.flextool_plot_batch = [(str(period), histogram_plot_frame(data[period])) for period in periods]
    return fig


def _save(fig: Figure, path: Path, write_plot_data: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=150)
    if write_plot_data:
        write_plot_data_csv(fig, path)


def create_ramp_plots(
    results,
    plot_dir: str | Path,
    scenarios: list[str],
    groups: list[str],
    nodes: list[str],
    plot_rows: list[int] | tuple[int, int] | None = None,
    break_times: pd.MultiIndex | dict | None = None,
    template: dict | None = None,
    write_plot_data: bool = False,
    unit_ctx: dict | None = None,
) -> list[Path]:
    """The comparison run's ramp plots: per dispatch group in *groups* and per node in *nodes* (the caller lists
    nodes only with the all-nodes option), an envelope PNG per scenario (``ramp_envelope_<kind>_<entity>_
    <scenario>.png``, the rows of *plot_rows*) and one histogram PNG over all scenarios
    (``ramp_frequency_<kind>_<entity>.png``); with *write_plot_data* each also as a CSV.  Returns the paths."""
    plot_dir = Path(plot_dir)
    timeline = ((int(plot_rows[0]), int(plot_rows[1]) + 1) if plot_rows and len(plot_rows) >= 2 else (0, 168))
    colors = ramp_band_colors(template)
    from flextool.plot_outputs.color_template import resolve_label_color

    scenario_colors = {s: resolve_label_color(s, template or {}, scenario=True) for s in scenarios}
    written: list[Path] = []
    for kind, entities, file_kind in (("group", groups, "nodeGroup"), ("node", nodes, "node")):
        for entity in entities:
            envelopes, trans, weights = {}, {}, {}
            for scenario in scenarios:
                env = entity_envelope(results, kind, entity, scenario)
                if env is None:
                    continue
                envelopes[scenario] = env
                trans[scenario] = transition_hours(results, scenario, env.index)
                weights[scenario] = timestep_hours(results, scenario, env.index)
                ctx = (unit_ctx or {}).get(scenario)
                unit = ramp_rate_unit(ctx, **{kind: entity})
                breaks = break_times.get(scenario) if isinstance(break_times, dict) else break_times
                fig = build_ramp_envelope_figure(env, f"{entity} - {scenario}: ramp envelope", unit=unit,
                                                 colors=colors, timeline=timeline, break_times=breaks,
                                                 keep_plot_data=write_plot_data)
                if fig is not None:
                    path = plot_dir / safe_file_component(f"ramp_envelope_{file_kind}_{entity}_{scenario}.png")
                    _save(fig, path, write_plot_data)
                    written.append(path)
            if not envelopes:
                continue
            ctx = (unit_ctx or {}).get(next(iter(envelopes)))
            fig = build_ramp_histogram_figure(histogram_data(envelopes, trans, weights),
                                              f"{entity}: ramp frequency", unit=ramp_rate_unit(ctx, **{kind: entity}),
                                              scenario_colors=scenario_colors, keep_plot_data=write_plot_data)
            if fig is not None:
                path = plot_dir / safe_file_component(f"ramp_frequency_{file_kind}_{entity}.png")
                _save(fig, path, write_plot_data)
                written.append(path)
    return written
