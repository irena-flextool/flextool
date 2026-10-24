"""``_direct.apply_varcost_params``: the variable-cost index sets and ``p_pssdt_varCost`` (#74).

    pssdt_varCost_eff_unit_source = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_eff, src[p,s,d,t] ≠ 0}
    pssdt_varCost_eff_unit_sink   = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_eff, snk[p,k,d,t] ≠ 0}
    pssdt_varCost_eff_connection  = {(p,s,k,d,t) : (p,s,k) ∈ process_source_sink_eff, proc[p,d,t] ≠ 0}
    p_pssdt_varCost[p,s,k,d,t]    = src + snk + proc   ∀ (p,s,k) ∈ process_source_sink, (d,t) ∈ dt; ≠ 0 only
    pssdt_varCost_noEff           = keys of p_pssdt_varCost on process_source_sink_noEff

Every shape the per-side Params can take (constant, period, time, period × time) gives the same cells; before the
fix the index sets were emit seeds, which have no period × time tier.  No DB: a ``SimpleNamespace`` FlexData.
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import polars as pl
import pytest

from polar_high import Param

from flextool.engine.preprocess.params._direct import apply_varcost_params

DT = [("d1", "t1"), ("d1", "t2"), ("d2", "t1"), ("d2", "t2")]
KEYS = ["p", "source", "sink", "d", "t"]
FIELDS = ("pssdt_varCost_noEff", "pssdt_varCost_eff_unit_source", "pssdt_varCost_eff_unit_sink",
          "pssdt_varCost_eff_connection", "p_pssdt_varCost")

# (p, source, sink) -> eff / noEff
ARCS = {
    ("u1", "a", "b"): "eff",
    ("c1", "a", "b"): "eff",
    ("c1", "b", "a"): "eff",
    ("n1", "a", "b"): "noEff",
    ("n2", "a", "b"): "noEff",
}
# Authored cell values (0 = no cost on that cell).
SRC = {("u1", "a"): {("d1", "t1"): 2.0, ("d1", "t2"): 0.0, ("d2", "t1"): 2.5, ("d2", "t2"): 2.5},
       ("n1", "a"): {dt: 1.0 for dt in DT},
       ("n2", "a"): {dt: 1.0 for dt in DT}}
SNK = {("u1", "b"): {("d1", "t1"): 0.0, ("d1", "t2"): 3.0, ("d2", "t1"): 3.0, ("d2", "t2"): 0.0},
       ("n1", "b"): {dt: 2.0 for dt in DT},
       ("n2", "b"): {dt: -1.0 for dt in DT}}       # n2: src + snk = 0 → no cost row
PROC = {("c1",): {dt: 0.5 for dt in DT}}


def _frame(table: dict, keys: list[str]) -> pl.DataFrame:
    rows = [(*k, d, t, v) for k, cells in table.items() for (d, t), v in cells.items()]
    return pl.DataFrame(rows, schema=[*keys, "d", "t", "value"], orient="row")


def _shaped(table: dict, keys: list[str], shape: str) -> Param:
    """The per-side Param in one authored shape.  Every shape must carry the same cells, so the tables use
    values that each shape can express where it is tested (``_constant_tables``)."""
    full = _frame(table, keys)
    if shape == "pt":
        return Param((*keys, "d", "t"), full)
    if shape == "p":
        return Param((*keys, "d"), full.unique([*keys, "d"], keep="first").select(*keys, "d", "value"))
    if shape == "t":
        return Param((*keys, "t"), full.unique([*keys, "t"], keep="first").select(*keys, "t", "value"))
    return Param(tuple(keys), full.unique(keys, keep="first").select(*keys, "value"))


def _constant_tables():
    """Tables a constant can express (one value per entity)."""
    def const(table):
        return {k: {dt: next(iter(cells.values())) for dt in DT} for k, cells in table.items()}
    return const(SRC), const(SNK), const(PROC)


def _fd(src, snk, proc, *, pss_dtypes=None) -> SimpleNamespace:
    arcs = pl.DataFrame(list(ARCS), schema=["p", "source", "sink"], orient="row")
    eff = arcs.filter(pl.Series([v == "eff" for v in ARCS.values()]))
    noeff = arcs.filter(pl.Series([v == "noEff" for v in ARCS.values()]))
    dt = pl.DataFrame(DT, schema=["d", "t"], orient="row")
    if pss_dtypes is not None:
        arcs, eff, noeff = (f.with_columns(pl.col(c).cast(pss_dtypes) for c in ("p", "source", "sink"))
                            for f in (arcs, eff, noeff))
    return SimpleNamespace(dt=dt, process_source_sink=arcs, process_source_sink_eff=eff,
                           process_source_sink_noEff=noeff,
                           p_pdt_varCost_source=src, p_pdt_varCost_sink=snk, p_pdt_varCost_process=proc)


def _expected(src_t: dict, snk_t: dict, proc_t: dict) -> dict:
    out = {f: set() for f in FIELDS[:4]}
    cost: dict = {}
    for (p, s, k), kind in ARCS.items():
        for d, t in DT:
            vs = src_t.get((p, s), {}).get((d, t), 0.0)
            vk = snk_t.get((p, k), {}).get((d, t), 0.0)
            vp = proc_t.get((p,), {}).get((d, t), 0.0)
            cell = (p, s, k, d, t)
            if kind == "eff":
                if vs != 0.0:
                    out["pssdt_varCost_eff_unit_source"].add(cell)
                if vk != 0.0:
                    out["pssdt_varCost_eff_unit_sink"].add(cell)
                if vp != 0.0:
                    out["pssdt_varCost_eff_connection"].add(cell)
            if vs + vk + vp != 0.0:
                cost[cell] = vs + vk + vp
                if kind == "noEff":
                    out["pssdt_varCost_noEff"].add(cell)
    return {**out, "p_pssdt_varCost": cost}


def _actual(fd) -> dict:
    out: dict = {}
    for f in FIELDS:
        x = getattr(fd, f)
        if x is None:
            out[f] = {} if f == "p_pssdt_varCost" else set()
            continue
        frame = x.frame if f == "p_pssdt_varCost" else x
        assert isinstance(frame, pl.DataFrame), f"{f}: expected an eager frame"
        frame = frame.with_columns(pl.col(c).cast(pl.Utf8) for c in KEYS)
        assert frame.columns[:5] == KEYS, f
        assert not frame.select(KEYS).is_duplicated().any(), f"{f}: duplicate cells"
        assert frame.select(KEYS).equals(frame.select(KEYS).sort(KEYS)), f"{f}: not sorted by its keys"
        if f == "p_pssdt_varCost":
            assert x.dims == tuple(KEYS)
            out[f] = {tuple(r[:5]): r[5] for r in frame.select(*KEYS, "value").iter_rows()}
        else:
            out[f] = {tuple(r) for r in frame.select(KEYS).iter_rows()}
    return out


def test_period_time_shapes_cell_by_cell() -> None:
    """Period × time per-side Params (the shape #74 lost): index sets and costs cell by cell; zero cells leave no
    row, and a noEff arc whose sides cancel has no cost row."""
    fd = _fd(_shaped(SRC, ["p", "source"], "pt"), _shaped(SNK, ["p", "sink"], "pt"), _shaped(PROC, ["p"], "pt"))
    apply_varcost_params(fd)
    got, want = _actual(fd), _expected(SRC, SNK, PROC)
    assert got == want
    assert ("n2", "a", "b", "d1", "t1") not in got["p_pssdt_varCost"]
    assert got["p_pssdt_varCost"][("n1", "a", "b", "d1", "t1")] == 3.0


@pytest.mark.parametrize("shape", ["scalar", "p", "t", "pt"])
def test_every_shape_gives_the_same_cells(shape: str) -> None:
    """A constant authored as scalar, period, time or period × time gives the same index sets and costs."""
    src_t, snk_t, proc_t = _constant_tables()
    fd = _fd(_shaped(src_t, ["p", "source"], shape), _shaped(snk_t, ["p", "sink"], shape),
             _shaped(proc_t, ["p"], shape))
    apply_varcost_params(fd)
    assert _actual(fd) == _expected(src_t, snk_t, proc_t)


@pytest.mark.parametrize("pss_enum", [True, False], ids=["enum_arcs_utf8_params", "utf8_arcs_enum_params"])
def test_mismatched_key_dtypes(pss_enum: bool) -> None:
    """Arc sets on an Enum entity-union axis with Utf8 Param keys, and the reverse: the joins compare values, the
    sets carry the arc sets' dtypes."""
    names = sorted({x for k in ARCS for x in k} | {"x_unused"})
    enum = pl.Enum(names)
    src, snk, proc = (_shaped(SRC, ["p", "source"], "pt"), _shaped(SNK, ["p", "sink"], "pt"),
                      _shaped(PROC, ["p"], "pt"))
    if not pss_enum:
        src, snk, proc = (Param(par.dims, par.frame.with_columns(pl.col(c).cast(enum) for c in par.dims
                                                                   if c in ("p", "source", "sink")))
                          for par in (src, snk, proc))
    fd = _fd(src, snk, proc, pss_dtypes=enum if pss_enum else None)
    apply_varcost_params(fd)
    assert _actual(fd) == _expected(SRC, SNK, PROC)
    want_dtype = enum if pss_enum else pl.Utf8
    for f in FIELDS:
        frame = getattr(fd, f).frame if f == "p_pssdt_varCost" else getattr(fd, f)
        for c in ("p", "source", "sink"):
            assert frame.schema[c] == want_dtype, (f, c, frame.schema[c])


def test_no_cost_params_give_none() -> None:
    """All per-side Params None → every set and ``p_pssdt_varCost`` None (stale seeds are cleared too)."""
    fd = _fd(None, None, None)
    for f in FIELDS:
        setattr(fd, f, "stale seed")
    apply_varcost_params(fd)
    assert all(getattr(fd, f) is None for f in FIELDS)


def test_zero_only_costs_give_none() -> None:
    zero = {k: {dt: 0.0 for dt in DT} for k in SRC}
    fd = _fd(_shaped(zero, ["p", "source"], "pt"), None, None)
    apply_varcost_params(fd)
    assert all(getattr(fd, f) is None for f in FIELDS)


def test_dt_restricts_cells() -> None:
    """Cells outside the solve's (d, t) pairs (a broader authored map) leave no row."""
    wide = {k: {**cells, ("d3", "t1"): 9.0} for k, cells in SRC.items()}
    fd = _fd(_shaped(wide, ["p", "source"], "pt"), None, None)
    apply_varcost_params(fd)
    cells = set(itertools.chain.from_iterable(
        {c[3:] for c in _actual(fd)[f]} for f in ("pssdt_varCost_eff_unit_source", "pssdt_varCost_noEff")))
    assert ("d3", "t1") not in cells
    assert cells <= set(DT)
