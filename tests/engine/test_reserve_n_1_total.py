"""The n-1 reserve rows use the group's total reserve ``v_reserve_total``.

Each ``reserveBalance_{up,down}_n_1_eq[r, ud, g, p_fail, d, t]`` row is
``v_reserve_total − own reserve of p_fail + vq_term ≥ failure of p_fail``, and
``reserve_total_upper[r, ud, g, d, t]`` bounds the total by the reserve held by all providers of the group
(``v_reserve_total − reserve_sum ≤ 0``).  ``v_reserve_total`` has no cost and enters only ``≥`` requirement rows,
so eliminating it gives back the per-provider form (one row per failing process with every other provider's
reserve): the reformulation is exact and the number of nonzeros grows with providers plus failing processes
instead of their product.  The timeseries and dynamic rows of the same groups use the total in place of
the providers' reserve (exact by the same argument).

The tests work on the unscaled canonical matrix of ``build(...)`` (no autoscale side vectors), where the
coefficients are exactly ``unitsize · reliability`` and the failing process's own coefficient cancels to 0.0 in
the elimination.

Fixtures: ``tests.json`` plus override alternatives, built from JSON (``coal`` 500 MW, ``wind`` 1000 MW,
``coal_chp`` 1200 MW, ``reserve_n_1``).
"""
from __future__ import annotations

import collections
import types

import numpy as np
import polars as pl
import pytest

from tests.engine._unit_db import (
    CHP, CHP_CHAIN, PLANT, PLANT_CHAIN, b64, build, make_db, run, solve, utf8,
)
from tests.engine.test_reserve_method_partitions import _db_values as _partition_values
from tests.engine.test_reserve_n_1_per_failure import _activator, _down_n_1, _two_failures

REL = "reserve__upDown__unit__node"
WIND = "wind_plant"
N1 = ["n_1", "up", "electricity"]
N1_FAMILIES = ("reserveBalance_up_n_1_eq", "reserveBalance_down_n_1_eq")
TS_DYN_FAMILIES = ("reserveBalance_timeseries_eq", "reserveBalance_dynamic_eq")
RESERVE_FAMILIES = N1_FAMILIES + TS_DYN_FAMILIES
TOTAL_ROW = "reserve_total_upper"
TOTAL_COL = "v_reserve_total"
N1_KEY = ("n_1", "up", "electricity")


# ---------------------------------------------------------------------------
# Fixtures


def _one_failure(tpf, alt: str):
    """Coal fails (ratio 0.1); coal (max_share 0.01) and wind (max_share 0.05, reliability 0.6) provide."""
    ents, vals = _activator("n_1", "up")
    return solve(tpf, alt=alt, base_chain=PLANT_CHAIN + ["wind", "reserve_n_1"], entities=ents,
                 values=vals + [[REL, ["n_1", "up", WIND, "west"], "max_share", b64(0.05, "float")],
                                ["reserve__upDown__group", N1, "penalty_reserve", b64(100.0, "float")]])


def _several_failures(tpf, alt: str):
    """Coal, wind and the CHP fail (ratio 1); the CHP fails into two group nodes (west and heat, with heat
    added to the electricity group) and provides reserve at both."""
    ents, vals = _activator("n_1", "up")
    ents = ents + [["group__node", ["electricity", "heat"], None],
                   [REL, ["n_1", "up", CHP, "west"], None], [REL, ["n_1", "up", CHP, "heat"], None]]
    for node in ("west", "heat"):
        vals = vals + [[REL, ["n_1", "up", CHP, node], "is_enabled", b64("yes", "str")],
                       [REL, ["n_1", "up", CHP, node], "large_failure_ratio", b64(1.0, "float")],
                       [REL, ["n_1", "up", CHP, node], "max_share", b64(0.2, "float")],
                       [REL, ["n_1", "up", CHP, node], "reliability", b64(0.9, "float")]]
    vals = vals + [[REL, ["n_1", "up", PLANT, "west"], "large_failure_ratio", b64(1.0, "float")],
                   [REL, ["n_1", "up", PLANT, "west"], "max_share", b64(0.3, "float")],
                   [REL, ["n_1", "up", WIND, "west"], "large_failure_ratio", b64(1.0, "float")],
                   [REL, ["n_1", "up", WIND, "west"], "max_share", b64(0.1, "float")],
                   ["reserve__upDown__group", N1, "penalty_reserve", b64(100.0, "float")]]
    return solve(tpf, alt=alt, base_chain=CHP_CHAIN + ["coal", "wind", "reserve_n_1"], entities=ents,
                 values=vals)


# Larger reservations than in test_reserve_method_partitions.py, so the timeseries rows of the combined
# groups compete with coal's energy output and carry a price.
PARTITION_RESERVATIONS = {"tlf": 200.0, "allm": 150.0}


def _partitions(tpf, alt: str):
    """The reserve-method model of ``test_reserve_method_partitions.py``: groups combining timeseries,
    dynamic and n-1 requirements (``tlf``, ``dl``, ``allm``) next to single-method groups, with the
    reservations of :data:`PARTITION_RESERVATIONS`."""
    ents, vals = _partition_values()
    vals = [[cls, ent, par, b64(PARTITION_RESERVATIONS[ent[0]], "float")]
            if par == "reservation" and ent[0] in PARTITION_RESERVATIONS else [cls, ent, par, val]
            for cls, ent, par, val in vals]
    url, scenario = make_db(tpf, alt=alt, values=vals, base_chain=PLANT_CHAIN + ["wind"], entities=ents)
    return list(run(tpf, url, scenario).values())[-1]


def _down(tpf, alt: str):
    """``_down_n_1`` with a consumption credit of 80 per MWh, so the failing consumer runs and coal's
    downward reserve covers its loss."""
    return _down_n_1(tpf, alt, extra=[["unit__inputNode", ["dr_increase_demand", "west"],
                                       "other_operational_cost", b64(-80.0, "float")]])


FIXTURES = {
    "one": _one_failure,
    "two": _two_failures,
    "several": _several_failures,
    "down": _down,
}


@pytest.fixture(scope="module")
def steps(tmp_path_factory):
    return {name: fn(tmp_path_factory, f"n1tot_{name}") for name, fn in FIXTURES.items()}


@pytest.fixture(scope="module")
def partitions(tmp_path_factory):
    return _partitions(tmp_path_factory, "n1tot_partitions")


# ---------------------------------------------------------------------------
# Canonical LP helpers


def _lp(step):
    """The unscaled canonical matrix of the step's model as a dict LP: ``rows[i] = {col: coef}``."""
    m = build(step.flex_data).canonicalise()
    rows: list[dict[int, float]] = [{} for _ in range(m.n_rows)]
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[int(m.row_idx[k])][c] = rows[int(m.row_idx[k])].get(c, 0.0) + float(m.val[k])
    return types.SimpleNamespace(
        m=m, rows=rows, row_lb=np.asarray(m.row_lb, dtype=float), row_ub=np.asarray(m.row_ub, dtype=float),
        row_names=list(m.row_names), col_names=list(m.col_names),
        row_of={nm: i for i, nm in enumerate(m.row_names)}, col_of={nm: i for i, nm in enumerate(m.col_names)})


def _key(name: str) -> str:
    """``family[a,b,…]`` → ``[a,b,…]``."""
    return name[name.index("["):]


def _family_rows(lp, families) -> list[int]:
    return [i for i, nm in enumerate(lp.row_names) if nm.split("[", 1)[0] in families]


def _eliminate_total(lp, families):
    """Substitute ``v_reserve_total`` out of the rows of ``families`` through its ``reserve_total_upper`` row.

    Each row with a ``v_reserve_total[k]`` coefficient ``a`` gets the defining row ``reserve_total_upper[k]``
    (coefficient ``c`` of the total) added with factor ``−a / c``; the total then cancels.  Returns
    ``{row index: {col: coef}}`` of the substituted rows, explicit zeros kept.
    """
    out: dict[int, dict[int, float]] = {}
    for i in _family_rows(lp, families):
        row = dict(lp.rows[i])
        totals = [c for c in row if lp.col_names[c].startswith(TOTAL_COL + "[")]
        for c_tot in totals:
            j = lp.row_of[TOTAL_ROW + _key(lp.col_names[c_tot])]
            assert lp.row_ub[j] == 0.0 and lp.row_lb[j] == -np.inf, lp.row_names[j]
            factor = -(row[c_tot] / lp.rows[j][c_tot])
            for c, v in lp.rows[j].items():
                row[c] = row.get(c, 0.0) + factor * v
            assert row.pop(c_tot) == 0.0
        out[i] = row
    return out


def _per_provider_lp(lp, families):
    """The LP with every ``v_reserve_total`` substituted out of the rows of ``families`` and the total's
    columns and defining rows removed: the per-provider form."""
    subst = _eliminate_total(lp, families)
    keep_cols = [c for c, nm in enumerate(lp.col_names) if not nm.startswith(TOTAL_COL + "[")]
    keep_rows = [i for i, nm in enumerate(lp.row_names) if not nm.startswith(TOTAL_ROW + "[")]
    new_col = {c: k for k, c in enumerate(keep_cols)}
    rows = []
    for i in keep_rows:
        row = subst.get(i, lp.rows[i])
        assert not [c for c in row if c not in new_col and row[c] != 0.0], lp.row_names[i]
        rows.append({new_col[c]: v for c, v in row.items() if c in new_col and v != 0.0})
    return types.SimpleNamespace(rows=rows, keep_cols=keep_cols, keep_rows=keep_rows,
                                 row_lb=lp.row_lb[keep_rows], row_ub=lp.row_ub[keep_rows],
                                 row_names=[lp.row_names[i] for i in keep_rows])


def _highs(rows, row_lb, row_ub, col_idx, m):
    """Solve the LP of ``rows`` over the canonical columns ``col_idx`` of ``m`` with HiGHS."""
    import highspy

    h = highspy.Highs()
    for opt, val in (("output_flag", False), ("random_seed", 0), ("solver", "simplex"),
                     ("primal_feasibility_tolerance", 1e-9), ("dual_feasibility_tolerance", 1e-9)):
        h.setOptionValue(opt, val)
    lp = highspy.HighsLp()
    lp.num_col_ = len(col_idx)
    lp.num_row_ = len(rows)
    lp.col_cost_ = np.asarray(m.col_obj, dtype=float)[col_idx]
    lp.col_lower_ = np.asarray(m.col_lb, dtype=float)[col_idx]
    lp.col_upper_ = np.asarray(m.col_ub, dtype=float)[col_idx]
    lp.row_lower_ = np.asarray(row_lb, dtype=float)
    lp.row_upper_ = np.asarray(row_ub, dtype=float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    starts, index, value = [0], [], []
    for row in rows:
        for c in sorted(row):
            index.append(c)
            value.append(row[c])
        starts.append(len(index))
    lp.a_matrix_.start_ = np.asarray(starts, dtype=np.int32)
    lp.a_matrix_.index_ = np.asarray(index, dtype=np.int32)
    lp.a_matrix_.value_ = np.asarray(value, dtype=float)
    integrality = np.asarray(m.col_int, dtype=bool)[col_idx]
    if integrality.any():
        lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous
                           for v in integrality]
    h.passModel(lp)
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    sol = h.getSolution()
    return types.SimpleNamespace(obj=h.getInfo().objective_function_value, x=np.asarray(sol.col_value),
                                 activity=np.asarray(sol.row_value), dual=np.asarray(sol.row_dual))


def _solve_both(step, families):
    """Solve the canonical LP and its per-provider form; return ``(lp, pp, new, old)``."""
    lp = _lp(step)
    pp = _per_provider_lp(lp, families)
    new = _highs(lp.rows, lp.row_lb, lp.row_ub, list(range(lp.m.n_cols)), lp.m)
    old = _highs(pp.rows, pp.row_lb, pp.row_ub, pp.keep_cols, lp.m)
    return lp, pp, new, old


# ---------------------------------------------------------------------------
# Expected per-provider rows, from flex_data


def _frame(fd, field: str) -> pl.DataFrame:
    return utf8(getattr(fd, field))


def _collect(frame) -> pl.DataFrame:
    return utf8(frame.collect() if isinstance(frame, pl.LazyFrame) else frame)


def _param(fd, field: str, keys: tuple[str, ...]) -> dict[tuple, float]:
    return {tuple(r[:-1]): r[-1] for r in _collect(getattr(fd, field).frame).select(*keys, "value").iter_rows()}


def _n_1_data(fd):
    """Providers (with unitsize · reliability) and failing arcs (with −unitsize · ratio) per group."""
    us = _param(fd, "p_unitsize", ("p",))
    rel = _param(fd, "p_process_reserve_upDown_node_reliability", ("p", "r", "ud", "n"))
    ratio = _param(fd, "p_process_reserve_upDown_node_large_failure_ratio_value", ("p", "r", "ud", "n"))
    gn = _frame(fd, "group_node").select("g", "n")
    rug = _frame(fd, "reserve_upDown_group").select("r", "ud", "g")
    providers: dict[tuple, dict[tuple, float]] = collections.defaultdict(dict)
    prov = (_frame(fd, "process_reserve_upDown_node_active").select("p", "r", "ud", "n")
            .join(gn, on="n").join(rug, on=["r", "ud", "g"]))
    for p, r, ud, n, g in prov.select("p", "r", "ud", "n", "g").iter_rows():
        providers[(r, ud, g)][(p, n)] = us[(p,)] * rel[(p, r, ud, n)]
    fails: dict[tuple, dict[str, dict[tuple, float]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
    n1 = _frame(fd, "reserve_upDown_group_method_n_1").select("r", "ud", "g")
    lfr = (_frame(fd, "process_reserve_upDown_node_large_failure_ratio").select("p", "r", "ud", "n")
           .join(gn, on="n").join(n1, on=["r", "ud", "g"]))
    pss = _frame(fd, "process_source_sink").select("p", "source", "sink")
    for p, r, ud, n, g in lfr.select("p", "r", "ud", "n", "g").iter_rows():
        side = "sink" if ud == "up" else "source"
        for _p, s, k in pss.filter((pl.col("p") == p) & (pl.col(side) == n)).iter_rows():
            fails[(r, ud, g)][p][(s, k)] = -(us[(p,)] * ratio[(p, r, ud, n)])
    from flextool.engine.model._reserve import reserve_shortfall_scale

    scale = {tuple(r[:-1]): r[-1]
             for r in _collect(reserve_shortfall_scale(fd).frame).select("r", "ud", "g", "d", "t", "value").iter_rows()}
    return providers, fails, scale


def _cols_with(terms: dict[str, float], prefix: str) -> dict[str, float]:
    return {c: v for c, v in terms.items() if c.startswith(prefix)}


def _expected_row(data, r, ud, g, f, d, t) -> dict[str, float]:
    providers, fails, scale = data
    row = {f"v_reserve[{p},{r},{ud},{n},{d},{t}]": v for (p, n), v in providers[(r, ud, g)].items() if p != f}
    row |= {f"v_flow[{f},{s},{k},{d},{t}]": v for (s, k), v in fails[(r, ud, g)][f].items()}
    if (r, ud, g, d, t) in scale:
        row[f"vq_reserve[{r},{ud},{g},{d},{t}]"] = scale[(r, ud, g, d, t)]
    return row


# ---------------------------------------------------------------------------
# Tests


@pytest.mark.parametrize("name", list(FIXTURES))
def test_n_1_rows_equal_per_provider_form_after_substitution(steps, name) -> None:
    """Substituting ``v_reserve_total`` through ``reserve_total_upper`` turns every n-1 row into the
    per-provider row: every provider other than the failing process at unitsize · reliability, the failing
    process's flows at −unitsize · ratio and the shortfall at its scale.  The failing process's own
    coefficient cancels to exactly 0.0."""
    step = steps[name]
    lp = _lp(step)
    data = _n_1_data(step.flex_data)
    subst = _eliminate_total(lp, N1_FAMILIES)
    assert subst, "no n-1 rows"
    n_cancelled = 0
    for i, row in subst.items():
        r, ud, g, f, d, t = _key(lp.row_names[i])[1:-1].split(",")
        cancelled = {lp.col_names[c] for c, v in row.items() if v == 0.0}
        assert all(c.startswith(f"v_reserve[{f},") for c in cancelled), lp.row_names[i]
        n_cancelled += len(cancelled)
        got = {lp.col_names[c]: v for c, v in row.items() if v != 0.0}
        expected = _expected_row(data, r, ud, g, f, d, t)
        assert got.keys() == expected.keys(), lp.row_names[i]
        for col, v in expected.items():
            assert got[col] == pytest.approx(v, rel=1e-12, abs=0.0), (lp.row_names[i], col)
    assert n_cancelled > 0, "no failing process is also a provider"


def _check_non_vacuous(lp, new) -> None:
    """Some n-1 row binds and, in a binding row, a provider other than the failing process holds reserve."""
    binding = [i for i in _family_rows(lp, N1_FAMILIES)
               if new.activity[i] - lp.row_lb[i] <= 1e-6 * max(1.0, abs(lp.row_lb[i]))]
    assert binding, "no n-1 row binds"
    helped = []
    for i in binding:
        r, ud, g, f, d, t = _key(lp.row_names[i])[1:-1].split(",")
        j = lp.row_of.get(f"{TOTAL_ROW}[{r},{ud},{g},{d},{t}]")
        if j is not None:
            helped += [lp.col_names[c] for c in lp.rows[j] if new.x[c] > 1e-6
                       and lp.col_names[c].startswith("v_reserve[")
                       and not lp.col_names[c].startswith(f"v_reserve[{f},")]
    assert helped, "no provider other than the failing process holds reserve in a binding n-1 row"


@pytest.mark.parametrize("name", list(FIXTURES))
def test_objective_equals_per_provider_form(steps, name) -> None:
    """The canonical LP and its per-provider form (total substituted out, its columns and rows removed)
    have the same optimum; the new solution restricted to the shared columns is feasible in the
    per-provider LP; and the canonical optimum is the production solve's objective."""
    step = steps[name]
    lp, pp, new, old = _solve_both(step, RESERVE_FAMILIES)
    _check_non_vacuous(lp, new)
    assert new.obj == pytest.approx(old.obj, rel=1e-9)
    assert new.obj == pytest.approx(step.solution.obj, rel=1e-9)
    x = new.x[pp.keep_cols]
    for k, row in enumerate(pp.rows):
        act = sum(v * x[c] for c, v in row.items())
        tol = 1e-7 * max(1.0, abs(act))
        assert act >= pp.row_lb[k] - tol and act <= pp.row_ub[k] + tol, pp.row_names[k]


@pytest.mark.parametrize("name", ["two", "several"])
def test_structure_scales_with_providers_plus_failures(steps, name) -> None:
    """Per (r, ud, g, d, t): one v_reserve_total column; reserve_total_upper has the total and every
    provider pair (P + 1 nonzeros); each n-1 row has the total, the failing process's own reserve pairs,
    its failing arcs and the shortfall — no other provider's reserve."""
    step = steps[name]
    lp = _lp(step)
    providers, fails, scale = _n_1_data(step.flex_data)
    totals = collections.Counter(_key(nm) for nm in lp.col_names if nm.startswith(TOTAL_COL + "["))
    assert totals and set(totals.values()) == {1}
    n1_rows: dict[str, list[int]] = collections.defaultdict(list)
    for i in _family_rows(lp, N1_FAMILIES):
        r, ud, g, f, d, t = _key(lp.row_names[i])[1:-1].split(",")
        n1_rows[f"[{r},{ud},{g},{d},{t}]"].append(i)
        assert not [lp.col_names[c] for c in lp.rows[i]
                    if lp.col_names[c].startswith("v_reserve[") and not lp.col_names[c].startswith(f"v_reserve[{f},")]
    assert set(n1_rows) == set(totals)
    for k, rows in n1_rows.items():
        r, ud, g, d, t = k[1:-1].split(",")
        own = {f: sum(1 for (p, _n) in providers[(r, ud, g)] if p == f) for f in fails[(r, ud, g)]}
        expected = (len(providers[(r, ud, g)]) + 1) + sum(
            1 + own[f] + len(fails[(r, ud, g)][f]) + ((r, ud, g, d, t) in scale) for f in fails[(r, ud, g)])
        nnz = len(lp.rows[lp.row_of[TOTAL_ROW + k]]) + sum(len(lp.rows[i]) for i in rows)
        assert nnz == expected, k
    if name == "several":
        own_chp = sum(1 for (p, _n) in providers[N1_KEY] if p == CHP)
        assert own_chp == 2 and len(fails[N1_KEY][CHP]) == 2


def test_v_reserve_total_registered() -> None:
    """The autoscale registry types the new column and row as POWER."""
    from flextool.engine.autoscale._layer2_types import QuantityType, lookup_cstr, lookup_var

    assert lookup_var(TOTAL_COL).column_type is QuantityType.POWER
    assert lookup_cstr(TOTAL_ROW).rhs_type is QuantityType.POWER


def test_timeseries_duals_equal_per_provider_form(partitions) -> None:
    """On the combined-method model the reported reserve price (the reserveBalance_timeseries_eq duals)
    is the same in the canonical LP and in its per-provider form (the total substituted out of the n-1,
    timeseries and dynamic rows)."""
    lp, pp, new, old = _solve_both(partitions, RESERVE_FAMILIES)
    assert new.obj == pytest.approx(old.obj, rel=1e-9)
    assert new.obj == pytest.approx(partitions.solution.obj, rel=1e-9)
    ts = [k for k, nm in enumerate(pp.row_names) if nm.startswith("reserveBalance_timeseries_eq[")]
    assert ts
    new_dual = new.dual[[pp.keep_rows[k] for k in ts]]
    old_dual = old.dual[ts]
    assert np.abs(old_dual).max() > 0.0
    np.testing.assert_allclose(new_dual, old_dual, rtol=1e-7, atol=1e-9)


def test_timeseries_and_dynamic_rows_use_the_total(partitions) -> None:
    """The timeseries and dynamic rows of the groups that also have n-1 rows (tlf, dl, allm) carry
    v_reserve_total instead of the providers' reserve; the other groups (ts, comb) keep reserve_sum.
    Substituting the total gives back reserve_sum: every provider at unitsize · reliability."""
    lp = _lp(partitions)
    providers, _fails, _scale = _n_1_data(partitions.flex_data)
    n_1_groups = {r for (r, _ud, _g) in providers if (r, _ud, _g) in _fails}
    assert n_1_groups == {"tlf", "dl", "allm"}
    subst = _eliminate_total(lp, TS_DYN_FAMILIES)
    seen = collections.Counter()
    for i, row in subst.items():
        name = lp.row_names[i]
        r, ud, g, d, t = _key(name)[1:-1].split(",")
        cols = {lp.col_names[c]: v for c, v in lp.rows[i].items()}
        totals = _cols_with(cols, TOTAL_COL + "[")
        reserves = _cols_with(cols, "v_reserve[")
        if r in n_1_groups:
            assert totals == {f"{TOTAL_COL}[{r},{ud},{g},{d},{t}]": 1.0} and not reserves, name
            seen[name.split("[")[0]] += 1
        else:
            assert not totals and reserves, name
        got = _cols_with({lp.col_names[c]: v for c, v in row.items() if v != 0.0}, "v_reserve[")
        expected = {f"v_reserve[{p},{r},{ud},{n},{d},{t}]": v for (p, n), v in providers[(r, ud, g)].items()}
        assert got.keys() == expected.keys(), name
        for col, v in expected.items():
            assert got[col] == pytest.approx(v, rel=1e-12, abs=0.0), (name, col)
    n_dt = partitions.flex_data.dt.height
    assert seen == {"reserveBalance_timeseries_eq": 2 * n_dt, "reserveBalance_dynamic_eq": 2 * n_dt}
