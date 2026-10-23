"""Delayed processes: the input of a unit arrives at its conversion balance some steps later.

Canonical use is a river chain, where water released upstream reaches the downstream plant after a known
travel time.  For a delayed unit, conversion_indirect (model.py) takes the inputs out of its undelayed
sum and adds the term of :func:`delayed_input_expr` instead: the source-side flows at earlier steps,
weighted by the unit's delay profile.  The module declares no variables and emits no rows or objective
terms of its own; it loads the delay tables and checks that they line up.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from polar_high import Param, Sum, Where
from polar_high.engine import Expr, Var

from flextool.engine._axis_enums import cast_dim
from flextool.engine.preprocess.params._derived import process_delayed__duration_from_weight
from flextool.engine.preprocess._provider_io import _provider_key


def _provider_get(provider, path: Path) -> pl.DataFrame | None:
    """The provider's frame for ``path``, or None when the provider is missing or lacks it."""
    if provider is None:
        return None
    key = _provider_key(path)
    if not provider.has(key):
        return None
    return provider.get(key)


# ---------------------------------------------------------------------------
# Feature detection


def has_feature(d) -> bool:
    """True when ``d`` carries delayed processes."""
    pd_set = getattr(d, "process_delayed", None)
    return pd_set is not None and pd_set.height > 0


# ---------------------------------------------------------------------------
# Loader


def load_data(inp_dir: str | Path, sd_dir: str | Path, *, provider=None) -> dict:
    """The delay tables, as ``field_name -> DataFrame | Param | None``; every value None when no process is delayed.

    Fields::

        process_delayed               (p,)
        process_delayed__duration     (p, td)
        process_source_delayed        (p, source)
        process_source_undelayed      (p, source)
        process_source_sink_delayed   (p, source, sink)
        process_source_sink_undelayed (p, source, sink)
        dtt__delay_duration           (d, t_source, t_sink, td)
        p_process_delay_weight        Param (p, td)

    ``inp_dir`` / ``sd_dir`` are the input and solve_data directories; files are read through ``provider``.
    ``td`` (delay duration, hours) is Float64 in every table.
    """
    sd = Path(sd_dir)

    blank = dict(
        process_delayed              = None,
        process_delayed__duration    = None,
        process_source_delayed       = None,
        process_source_undelayed     = None,
        process_source_sink_delayed  = None,
        process_source_sink_undelayed = None,
        dtt__delay_duration          = None,
        p_process_delay_weight       = None,
    )

    pd_path = sd / "process_delayed.csv"
    pd_df = _provider_get(provider, pd_path)
    if pd_df is None or pd_df.height == 0:
        # The file is written header-only when no process is delayed.
        return blank

    pd_df = pd_df.rename({"process": "p"}) if "process" in pd_df.columns else pd_df

    def _read_pse(name: str) -> pl.DataFrame | None:
        """(p, source) rows of solve_data/<name>.csv."""
        p = sd / f"{name}.csv"
        df = _provider_get(provider, p)
        if df is None or df.height == 0:
            return None
        if "process" in df.columns:
            df = df.rename({"process": "p"})
        return df.select("p", "source")

    pse_delayed   = _read_pse("process_source_delayed")
    pse_undelayed = _read_pse("process_source_undelayed")

    def _read_psse(name: str) -> pl.DataFrame | None:
        """(p, source, sink) rows of solve_data/<name>.csv."""
        p = sd / f"{name}.csv"
        df = _provider_get(provider, p)
        if df is None or df.height == 0:
            return None
        if "process" in df.columns:
            df = df.rename({"process": "p"})
        return df.select("p", "source", "sink")

    psse_delayed   = _read_psse("process_source_sink_delayed")
    psse_undelayed = _read_psse("process_source_sink_undelayed")

    # Inputs with source conversion_flow_coeff = 0 do not enter the conversion balance; drop them from the
    # delayed sets as the undelayed input set drops them, so they are not counted.
    inp = Path(inp_dir)
    src_path = inp / "p_process_source_conversion_flow_coeff.csv"
    srcdf = _provider_get(provider, src_path)
    if srcdf is not None and srcdf.height > 0 and "p_process_source_conversion_flow_coeff" in srcdf.columns:
        zero_src = (srcdf
            .rename({"process": "p",
                     "p_process_source_conversion_flow_coeff": "coef"})
            .with_columns(pl.col("coef").cast(pl.Float64, strict=False))
            .filter(pl.col("coef") == 0.0)
            .select("p", "source"))
        if zero_src.height > 0:
            if pse_delayed is not None:
                pse_delayed = pse_delayed.join(
                    zero_src, on=["p", "source"], how="anti")
            if psse_delayed is not None:
                psse_delayed = psse_delayed.join(
                    zero_src, on=["p", "source"], how="anti")

    # Time-shift table (d, t_source, t_sink, td).  The text column may be written as "1" or "1.0", so td is
    # cast to Float64 like on the DB path.
    dtt_path = sd / "dtt__delay_duration.csv"
    dtt_df = None
    raw = _provider_get(provider, dtt_path)
    if raw is not None and raw.height > 0:
        rename_map = {}
        if "period" in raw.columns:
            rename_map["period"] = "d"
        if "time_source" in raw.columns:
            rename_map["time_source"] = "t_source"
        if "time_sink" in raw.columns:
            rename_map["time_sink"] = "t_sink"
        if "delay_duration" in raw.columns:
            rename_map["delay_duration"] = "td"
        dtt_df = (raw.rename(rename_map)
                  .select("d", "t_source", "t_sink",
                          pl.col("td").cast(pl.Float64)))

    pw_path = sd / "p_process_delay_weight.csv"
    pw_param = None
    raw = _provider_get(provider, pw_path)
    if raw is not None and raw.height > 0:
        rename_map = {}
        if "process" in raw.columns:
            rename_map["process"] = "p"
        if "delay_duration" in raw.columns:
            rename_map["delay_duration"] = "td"
        pw_long = raw.rename(rename_map).select(
            "p", pl.col("td").cast(pl.Float64),
            pl.col("value").cast(pl.Float64))
        pw_param = Param(("p", "td"), pw_long)
    # The duration set comes from the weight keys (the same derivation as on the DB path), so its td labels
    # match dtt__delay_duration and p_process_delay_weight exactly.
    pdd_df = process_delayed__duration_from_weight(pw_param)

    return dict(
        process_delayed              = pd_df.select("p"),
        process_delayed__duration    = pdd_df,
        process_source_delayed       = pse_delayed,
        process_source_undelayed     = pse_undelayed,
        process_source_sink_delayed  = psse_delayed,
        process_source_sink_undelayed = psse_undelayed,
        dtt__delay_duration          = dtt_df,
        p_process_delay_weight       = pw_param,
    )


# ---------------------------------------------------------------------------
# Consistency check

class DelayDataError(RuntimeError):
    """The delay tables of a delayed unit do not line up.

    An internal wiring error, raised instead of building a conversion
    equation without the unit's input (which forces its output to zero
    without any message).
    """


def _p_utf8(df: pl.DataFrame, cols: tuple[str, ...]) -> pl.DataFrame:
    return df.select(pl.col(c).cast(pl.Utf8) for c in cols).unique()


def check_delay_tables(d) -> None:
    """Raise :class:`DelayDataError` unless every delayed indirect unit can
    get its delayed input term.

    ``conversion_indirect`` takes a delayed unit's inputs out of the
    undelayed sum (``process_delayed``), so the delayed term must cover
    them.  For every unit in ``process_delayed`` that is indirect and has
    inputs (``process_input_flows``):

    * it has delayed input arcs (``process_source_sink_delayed``);
    * it has ``process_delayed__duration`` rows;
    * ``td`` has the same dtype in ``process_delayed__duration``,
      ``dtt__delay_duration`` and ``p_process_delay_weight``;
    * every ``(p, td)`` has ``dtt__delay_duration`` rows and a weight.

    Delayed connections are not indirect and are not checked here (no
    delayed term applies to them).
    """
    if not has_feature(d):
        return
    indirect = getattr(d, "process_indirect", None)
    inputs = getattr(d, "process_input_flows", None)
    if indirect is None or indirect.height == 0 or inputs is None:
        return
    units = (_p_utf8(d.process_delayed, ("p",))
             .join(_p_utf8(indirect, ("p",)), on="p", how="semi")
             .join(_p_utf8(inputs, ("p",)), on="p", how="semi"))
    if units.height == 0:
        return
    names = sorted(units["p"].to_list())

    def _fail(msg: str, which: list[str]) -> None:
        raise DelayDataError(
            f"delayed unit(s) {', '.join(which)}: {msg}. Their inputs are "
            "excluded from the undelayed conversion, so the unit would "
            "be forced to zero output.")

    psse = getattr(d, "process_source_sink_delayed", None)
    if psse is None or psse.height == 0:
        _fail("no delayed input arcs (process_source_sink_delayed)", names)
    arcs = (psse.select(pl.col(c).cast(pl.Utf8) for c in ("p", "sink"))
            .filter(pl.col("sink") == pl.col("p")).select("p").unique())
    missing = units.join(arcs, on="p", how="anti")
    if missing.height > 0:
        _fail("no delayed input arcs (process_source_sink_delayed)",
              sorted(missing["p"].to_list()))

    pdd = getattr(d, "process_delayed__duration", None)
    dtt = getattr(d, "dtt__delay_duration", None)
    pw = getattr(d, "p_process_delay_weight", None)
    if pdd is None or pdd.height == 0:
        _fail("no delay durations (process_delayed__duration)", names)
    missing = units.join(_p_utf8(pdd, ("p",)), on="p", how="anti")
    if missing.height > 0:
        _fail("no delay durations (process_delayed__duration)",
              sorted(missing["p"].to_list()))
    if dtt is None or dtt.height == 0:
        _fail("no time-shift rows (dtt__delay_duration)", names)
    if pw is None or pw.frame.height == 0:
        _fail("no delay weights (p_process_delay_weight)", names)
    dtypes = {"process_delayed__duration": pdd.schema["td"],
              "dtt__delay_duration": dtt.schema["td"],
              "p_process_delay_weight": pw.frame.schema["td"]}
    if len(set(dtypes.values())) != 1:
        _fail("delay duration (td) dtypes differ: "
              + ", ".join(f"{k} {v}" for k, v in dtypes.items()), names)
    used = (pdd.with_columns(pl.col("p").cast(pl.Utf8))
            .join(units, on="p", how="semi"))
    no_shift = used.join(dtt.select("td").unique(), on="td", how="anti")
    if no_shift.height > 0:
        _fail("delay duration(s) without time-shift rows "
              "(dtt__delay_duration): "
              + ", ".join(f"{p} {td!r}" for p, td in no_shift.iter_rows()),
              sorted(set(no_shift["p"].to_list())))
    no_weight = used.join(
        pw.frame.select(pl.col("p").cast(pl.Utf8), "td"),
        on=["p", "td"], how="anti")
    if no_weight.height > 0:
        _fail("delay duration(s) without a weight "
              "(p_process_delay_weight): "
              + ", ".join(f"{p} {td!r}" for p, td in no_weight.iter_rows()),
              sorted(set(no_weight["p"].to_list())))


# ---------------------------------------------------------------------------
# Delayed input term of conversion_indirect


def delayed_input_expr(d, v_flow: Var) -> Expr | None:
    """The delayed input of each delayed indirect unit, MWh per step, open dims (p, d, t); None without delays.

    I_delayed[p,d,t] = Σ_{s : (p,s,p) ∈ process_source_sink_delayed}
                       Σ_{(d,t_s,t,δ) ∈ dtt__delay_duration, (p,δ) ∈ process_delayed__duration}
                           v_flow[p,s,p,d,t_s] · unitsize[p] · delay_weight[p,δ] · conv_in[p,s]
      t = sink-side step of conversion_indirect, t_s = source-side step δ hours earlier
      conv_in = source conversion_flow_coeff (default 1)
    conversion_indirect adds this term on its input side in place of the delayed unit's undelayed inputs.
    Raises :class:`DelayDataError` when a delayed indirect unit's tables do not line up.
    """
    if not has_feature(d):
        return None
    check_delay_tables(d)
    if (d.dtt__delay_duration is None or d.p_process_delay_weight is None
            or d.process_source_delayed is None or d.process_delayed__duration is None
            or d.p_unitsize is None):
        return None
    psse_delayed = d.process_source_sink_delayed
    if psse_delayed is None or psse_delayed.height == 0:
        return None
    # Delayed input arcs feed the unit's own balance (sink == p); sink carries the entity-union Enum, so p is
    # widened to it for the comparison.
    indirect_inputs_delayed = psse_delayed.filter(
        pl.col("sink") == cast_dim(pl.col("p"), None, "e")
    ).select("p", "source", "sink")
    if indirect_inputs_delayed.height == 0:
        return None

    # (p, source, sink, td, d, t_source, t): each delayed input arc × its delay durations × the time shifts.
    pst_td = indirect_inputs_delayed.join(d.process_delayed__duration, on="p", how="inner")
    if pst_td.height == 0:
        return None
    full_map = pst_td.join(d.dtt__delay_duration, on="td", how="inner")
    if full_map.height == 0:
        return None
    full_map = full_map.rename({"t_sink": "t"})

    # A view of v_flow (same columns) indexed by the source-side step t_source, so that the Where on
    # full_map attaches the sink-side t and td as new open dims.
    v_flow_at_source = Var(
        name=v_flow.name + "__at_t_source",
        dims=("p", "source", "sink", "d", "t_source"),
        frame=v_flow.frame.rename({"t": "t_source"}),
        lower=v_flow.lower, upper=v_flow.upper, integer=v_flow.integer,
    )
    expr = Where(v_flow_at_source, full_map) * d.p_unitsize * d.p_process_delay_weight
    # Without the Param every coefficient is 1 and the factor is skipped.
    if getattr(d, "p_process_source_conversion_flow_coeff", None) is not None:
        expr = expr * d.p_process_source_conversion_flow_coeff
    return Sum(expr, over=("source", "sink", "td", "t_source"))
