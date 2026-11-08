"""Input checks for the node inflow scaling methods.

``inflow_method`` ``scale_to_annual_flow``, ``scale_in_proportion`` and ``scale_to_annual_and_peak_flow`` rescale
the node's ``inflow`` per period (docs/how_to.md "How to set the demand in a node"; docs/reference.md node ``inflow_method``):

* ``scale_to_annual_flow``  — ``inflow · annual_flow / (|Σ inflow| / period share of year)``;
* ``scale_in_proportion``   — ``inflow · annual_flow / (|Σ inflow| / timeline duration in years)``;
* ``scale_to_annual_and_peak_flow`` — ``(peak/peak_old)·(1+c)·inflow − peak·c``, ``c`` set so the annual sum
  matches ``annual_flow``.

The scaling emitter (:mod:`._inflow_scaling`) drops every (node, period) whose ingredient is missing or zero, and
the node's inflow then silently becomes 0 in that period (no demand, objective near 0).  Worse, a
``scale_to_annual_and_peak_flow`` node without an ``inflow`` profile has ``peak_old = 0``: the slope term vanishes,
the profile's direction defaults to "supply" (the sign of an all-zero series is taken as ``+``), and the remaining
constant ``−peak·c`` equals ``+annual_flow / 8760`` whatever the sign of ``peak_inflow`` — a demand node becomes a
constant supply.  A ``peak_inflow`` of the opposite sign to the profile, or a negative ``annual_flow``, flips the
direction the same way.

:func:`check_inflow_scaling_inputs` runs per solve, per balance node with a scaling method and per period in use,
before any scaling frame is emitted.  Period values are read through :class:`~._pdt_lookup.PdLookup` (the same
period resolution as ``pdNode``, stochastic branch periods included).  Errors (``FlexToolConfigError``, all problems
of the solve in one message) for a result that is not what the method promises:

* ``annual_flow`` not given for the period, or negative;
* ``peak_inflow`` not given or 0 for the period (``scale_to_annual_and_peak_flow`` with ``annual_flow`` > 0);
* no ``inflow`` profile, or one that sums to 0 over the period where the method divides by that sum;
* ``peak_inflow`` with the opposite sign of the inflow profile's peak.

Warnings (printed once per node and run) where the result is well defined but likely unintended:

* ``annual_flow`` above ``|peak_inflow|`` times the time steps of a year (``8760 · S / H``: ``S`` the time steps of
  the period's complete solve, weighted, counted per original step on a solve with ``new_stepduration``; ``H`` their
  hours): no profile with that peak reaches the annual flow, so the scaled inflow exceeds ``peak_inflow`` at its
  off-peak steps;
* the scaled inflow changes direction (demand to supply or back) at steps where the original profile did not.

An explicit ``annual_flow`` of 0 is accepted (the inflow is 0 in that period).  A node whose inflow comes from
stochastic branch data (``pbt_node_inflow``) is not checked against its deterministic profile.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine._behaviour_changes import warn_once
from flextool.engine.input._constants import HOURS_PER_YEAR
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.emit._base_steps import read_new_step_base_steps
from flextool.engine.preprocess.emit._pdt_lookup import PdLookup

_ANNUAL = "scale_to_annual_flow"
_PROPORTION = "scale_in_proportion"
_PEAK = "scale_to_annual_and_peak_flow"
_SCALING_METHODS = (_ANNUAL, _PROPORTION, _PEAK)
_BALANCE_TYPES = ("balance", "storage", "balance_within_period")


def _frame(provider, path: Path) -> pl.DataFrame | None:
    df = provider.get(_provider_key(path))
    return df if df is not None and df.height > 0 else None


def _col_strings(df: pl.DataFrame | None, idx: int) -> list[str]:
    if df is None:
        return []
    return [str(v) for v in df.get_column(df.columns[idx]).to_list() if v is not None and str(v) != ""]


def _keyed_float(df: pl.DataFrame | None, n_keys: int, names: list[str]) -> pl.DataFrame:
    """``df``'s first ``n_keys`` columns as Utf8 ``names`` plus a Float64 ``value`` (unparsable values dropped)."""
    schema = {**{n: pl.Utf8 for n in names}, "value": pl.Float64}
    if df is None:
        return pl.DataFrame(schema=schema)
    cols = df.columns
    return (df.select([pl.col(cols[i]).cast(pl.Utf8).alias(names[i]) for i in range(n_keys)]
                      + [pl.col(cols[n_keys]).cast(pl.Utf8).cast(pl.Float64, strict=False).alias("value")])
            .drop_nulls())


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def check_inflow_scaling_inputs(input_dir: Path, solve_data_dir: Path, *, provider,
                                frames: dict[str, pl.DataFrame]) -> None:
    """Raise / warn for scaling inputs that the inflow scaling cannot honour (see the module docstring).

    ``frames`` are the scaling frames of this solve (``ptNode_inflow.csv``, ``new_old_slope.csv``,
    ``new_old_section.csv``) from :func:`._inflow_scaling._compute_inflow_scaling_frames_vectorized`.
    """
    from flextool.engine.preprocess.emit._mid_sets import derive_node_effective_type

    method_df = _frame(provider, solve_data_dir / "node__inflow_method.csv")
    if method_df is None:
        return
    pairs = [(str(n), str(m)) for n, m in method_df.select(method_df.columns[:2]).iter_rows()
             if m is not None and str(m) in _SCALING_METHODS]
    if not pairs:
        return
    node_types = derive_node_effective_type(input_dir, provider=provider)
    balance = set(node_types.filter(pl.col("type").cast(pl.Utf8).is_in(_BALANCE_TYPES))
                  .get_column("node").cast(pl.Utf8).to_list())
    pairs = [(n, m) for n, m in pairs if n in balance]
    if not pairs:
        return

    periods = _col_strings(_frame(provider, solve_data_dir / "period_in_use_set.csv"), 0)
    times = _col_strings(_frame(provider, solve_data_dir / "time.csv"), 0)
    pn = PdLookup(pd_csv=input_dir / "pd_node.csv", p_csv=input_dir / "p_node.csv",
                  period_branch_csv=solve_data_dir / "period__branch.csv", provider=provider)
    pd_node = {(r[0], r[1], r[2]): r[3] for r in _keyed_float(
        _frame(provider, solve_data_dir / "pdNode.csv"), 3, ["node", "param", "period"]).iter_rows()}

    def _period_value(n: str, param: str, d: str) -> float | None:
        """``pdNode[n, param, d]`` (the value the scaling uses), or None when the input gives none for *d* (a 0
        in ``pdNode`` is the fallback of an unset parameter or of a period map without *d*)."""
        v = pd_node.get((n, param, d), 0.0)
        return v if v != 0 or pn.has(n, param, d) else None

    stochastic_nodes = set(_col_strings(_frame(provider, input_dir / "pbt_node_inflow.csv"), 0))
    explicit_nodes = set(_col_strings(_frame(provider, solve_data_dir / "pt_node_inflow.csv"), 0))
    full_timeline = {r[0]: r[1] for r in _keyed_float(
        _frame(provider, solve_data_dir / "node_full_timeline_flow.csv"), 1, ["node"]).iter_rows()}

    scaled_nodes = sorted({n for n, _m in pairs})
    pti = (_keyed_float(frames.get("ptNode_inflow.csv"), 2, ["node", "time"])
           .filter(pl.col("node").is_in(scaled_nodes)))
    weights = _keyed_float(_frame(provider, solve_data_dir / "timestep_weight.csv"), 2, ["period", "time"])
    dt_complete = _keyed_float(None, 2, ["period", "time"]).drop("value")
    dtc = _frame(provider, solve_data_dir / "steps_complete_solve.csv")
    if dtc is not None:
        dt_complete = dtc.select([pl.col(dtc.columns[0]).cast(pl.Utf8).alias("period"),
                                  pl.col(dtc.columns[1]).cast(pl.Utf8).alias("time")])

    def _weighted(steps: pl.DataFrame) -> pl.DataFrame:
        """Σ_t inflow[n, t] · w[d, t] over the (period, time) ``steps`` (the sums the methods divide by)."""
        return (pti.join(steps, on="time", how="inner")
                .join(weights.rename({"value": "w"}), on=["period", "time"], how="left")
                .group_by(["node", "period"])
                .agg((pl.col("value") * pl.col("w").fill_null(1.0)).sum().alias("s")))

    period_frame = pl.DataFrame({"period": periods}, schema={"period": pl.Utf8})
    times_frame = pl.DataFrame({"time": times}, schema={"time": pl.Utf8})
    complete_sum = {(r[0], r[1]): r[2] for r in _weighted(dt_complete).iter_rows()}
    time_sum = {(r[0], r[1]): r[2] for r in _weighted(period_frame.join(times_frame, how="cross")).iter_rows()}
    series = {n: pti.filter(pl.col("node") == n).get_column("value").to_list() for n in scaled_nodes}
    # Original time steps per solve step (new_stepduration); empty → 1.  The peak is per original step: a coarsened
    # solve reads it from the original series (input/pt_node_inflow, else the p_node constant).
    n_of, _absorbed = read_new_step_base_steps(solve_data_dir, provider=provider)
    if n_of:
        base = _keyed_float(_frame(provider, input_dir / "pt_node_inflow.csv"), 2, ["node", "time"])
        explicit_nodes = set(base.get_column("node").to_list())
        base_of = {(r[0], r[1]): r[2] for r in base.iter_rows()}
        const = {r[0]: r[2] for r in _keyed_float(_frame(provider, input_dir / "p_node.csv"), 2, ["node", "param"])
                 .filter(pl.col("param") == "inflow").iter_rows()}
        series = {n: [base_of.get((n, t), const.get(n, 0.0)) for t in times] for n in scaled_nodes}
    hours = {r[0]: r[1] for r in _keyed_float(
        _frame(provider, solve_data_dir / "complete_hours_in_period.csv"), 1, ["period"]).iter_rows()}
    share = {r[0]: r[1] for r in _keyed_float(
        _frame(provider, solve_data_dir / "complete_period_share_of_year_calc.csv"), 1, ["period"]).iter_rows()}
    w_of = {(r[0], r[1]): r[2] for r in weights.iter_rows()}

    def _old_peak(n: str) -> tuple[float, float]:
        """``(sign, peak)`` of the inflow profile as the peak scaling reads them."""
        vals = series.get(n) or [0.0]
        if n in explicit_nodes:
            hi, lo = max(vals), min(vals)
            sign = 1.0 if abs(hi) >= abs(lo) else -1.0
        else:
            hi = lo = vals[0]
            sign = 1.0 if vals[0] >= 0 else -1.0
        return sign, hi if sign >= 0 else lo

    slope = {(r[0], r[1]): r[2] for r in _keyed_float(frames.get("new_old_slope.csv"), 2, ["n", "d"]).iter_rows()}
    section = {(r[0], r[1]): r[2] for r in _keyed_float(frames.get("new_old_section.csv"), 2, ["n", "d"]).iter_rows()}
    steps_of = {}
    for d, t in dt_complete.iter_rows():
        steps_of.setdefault(d, []).append(t)
    pti_of = {(r[0], r[1]): r[2] for r in pti.iter_rows()}

    def _steps_a_year(d: str) -> float:
        """``8760 · S_d / H_d``: time steps a year as the peak scaling counts them (8760 on 1 h steps)."""
        s_d = sum(n_of.get(t, 1.0) * w_of.get((d, t), 1.0) for t in steps_of.get(d, ()))
        h_d = hours.get(d, share.get(d, 0.0) * HOURS_PER_YEAR)
        return HOURS_PER_YEAR * (s_d / h_d if h_d != 0 else 1.0)

    no_inflow_fix = ("give the node an inflow profile (negative values for a demand, positive for a supply), or "
                     "set inflow_method to no_inflow")
    errors: list[str] = []
    for n, m in pairs:
        for d in periods:
            where = f"node '{n}', inflow_method '{m}', period '{d}'"
            af = _period_value(n, "annual_flow", d)
            if af is None:
                errors.append(f"{where}: annual_flow is not given for this period; give annual_flow (a constant or "
                              "a map over periods that includes this period), or set inflow_method to use_original")
                continue
            if af < 0:
                errors.append(f"{where}: annual_flow is negative ({_fmt(af)}); annual_flow is the annual energy as a "
                              "positive number, the sign of inflow sets the direction (negative = demand)")
                continue
            if af == 0:
                continue
            peak = None
            if m == _PEAK:
                peak = _period_value(n, "peak_inflow", d)
                if not peak:
                    errors.append(f"{where}: peak_inflow is not given (or is 0) for this period; give peak_inflow "
                                  "with the sign of the inflow (negative for a demand)")
                    continue
            if n in stochastic_nodes:
                continue
            if m == _ANNUAL:
                if full_timeline.get(n, 0.0) == 0 and complete_sum.get((n, d), 0.0) == 0:
                    errors.append(f"{where}: the inflow sums to 0 over the period's timesteps (no inflow profile?), "
                                  f"so it cannot be scaled to annual_flow {_fmt(af)}; {no_inflow_fix}")
            elif m == _PROPORTION:
                if time_sum.get((n, d), 0.0) == 0:
                    errors.append(f"{where}: the inflow sums to 0 over the solve's timesteps (no inflow profile?), "
                                  f"so it cannot be scaled in proportion to annual_flow {_fmt(af)}; {no_inflow_fix}")
            else:
                sign, old_peak = _old_peak(n)
                if old_peak == 0:
                    errors.append(f"{where}: the node has no inflow profile (all inflow is 0), so there is no shape "
                                  f"to scale to peak_inflow {_fmt(peak)} and annual_flow {_fmt(af)}; "
                                  f"{no_inflow_fix}")
                    continue
                if (peak > 0) != (sign > 0):
                    errors.append(f"{where}: peak_inflow {_fmt(peak)} has the opposite sign of the inflow profile's "
                                  f"peak {_fmt(old_peak)}; the scaling would turn "
                                  f"{'demand into supply' if sign < 0 else 'supply into demand'}. Give peak_inflow "
                                  "with the sign of the inflow (negative for a demand)")
                    continue
                _warn_peak(n, d, af, peak, slope.get((n, d)), section.get((n, d)),
                           [(pti_of.get((n, t), 0.0), n_of.get(t, 1.0)) for t in steps_of.get(d, ())],
                           _steps_a_year(d))
    if errors:
        raise FlexToolConfigError(
            "Inflow scaling cannot be applied:\n  - " + "\n  - ".join(errors)
            + "\nSee docs/how_to.md 'How to set the demand in a node' for what each inflow_method needs.")


def _warn_peak(n: str, d: str, af: float, peak: float, slope: float | None, section: float | None,
               old: list[tuple[float, float]], steps_a_year: float) -> None:
    """Warnings of ``scale_to_annual_and_peak_flow`` for a well-defined but likely unintended result.

    ``old`` holds ``(inflow, original steps)`` of each solve step; ``steps_a_year`` is ``8760 · S / H``.
    """
    reach = abs(peak) * steps_a_year
    if af > reach:
        warn_once(("inflow_scaling_peak_reach", n),
                  f"Warning: node '{n}' (scale_to_annual_and_peak_flow, period '{d}'): annual_flow {_fmt(af)} is "
                  f"above |peak_inflow| x {_fmt(steps_a_year)} time steps a year = {_fmt(reach)}; no profile with "
                  f"peak {_fmt(peak)} reaches it, so the scaled inflow exceeds peak_inflow at its off-peak steps. "
                  "Raise peak_inflow or lower annual_flow.")
    if slope is None or section is None:
        return
    s = 1.0 if peak > 0 else -1.0
    tol = 1e-9 * abs(peak)
    flipped = [slope * v - section * k for v, k in old if s * v >= 0 and s * (slope * v - section * k) < -tol]
    if flipped:
        worst = min(flipped, key=lambda x: s * x)
        warn_once(("inflow_scaling_peak_flip", n),
                  f"Warning: node '{n}' (scale_to_annual_and_peak_flow, period '{d}'): the scaled inflow changes "
                  f"direction at {len(flipped)} timestep(s) (down to {_fmt(worst)} against peak_inflow "
                  f"{_fmt(peak)}): annual_flow {_fmt(af)} is too small for this peak and profile shape, so the "
                  f"low-{'demand' if s < 0 else 'supply'} steps turn into {'supply' if s < 0 else 'demand'}. Raise "
                  "annual_flow or lower |peak_inflow|.")
