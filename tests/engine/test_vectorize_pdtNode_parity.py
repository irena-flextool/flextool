"""Parity gate for the vectorized ``pdtNode`` derive.

FINDING (verified, see the extra-stochastic test below): the design
brief expected the pdtNode fold to fire on real data via a non-empty
``pbt_node.csv`` in ``stochastics_pbt_inflow`` / ``branch2_parent_period``.
It does NOT.  ``_specs.py`` writes ``pbt_node.csv`` (the ``PdtLookup``
input) ONLY from node params authored as a ``3d_map`` among
``{penalty_down, self_discharge_loss, availability,
storage_state_reference_value}``; the ``inflow`` 3d_map in those fixtures
routes to ``pbt_node_inflow.csv`` (the inflow-scaling path), a DIFFERENT
family.  No shipped fixture authors a pbt_node-trigger param as a
3d_map, so ``pbt_node.csv`` is EMPTY for the pdtNode cascade on every
real fixture — exactly like pbt_process (build-log).  The SYNTHETIC test
is therefore the real fold coverage; the real-fixture tests gate the
full vectorized derive (time-first pt/pd, p, def1, class-default, 0.0)
against the legacy scalar cascade on real node domains.

Tier policy (design §1): assert strict ``df_vec.equals(df_legacy)``
(Tier A) first.  If a real multi-term node fold produces last-ULP float
drift (a ``group_by().sum()`` pairwise reduction vs the legacy
sequential ``total += v``), the cell is byte-different only by
``|Δ| ≤ ~1e-12``; that legitimately demotes the fixture to **Tier B**
(parse both ``value`` columns to float, key cols ``.equals``, values
``rtol ≤ 1e-12``).  Any NON-ULP difference is a real bug — those are NOT
tolerated; the test fails hard.

The cascade exercised (``PdtLookup`` 9-branch, time-first):
fold → ``pt`` → ``pd`` → ``p`` → def1{availability}→1.0 →
class-default(param) → 0.0.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess.emit._pdt_params import (
    derive_pdtNode,
    derive_pdtNode_vectorized,
)
from tests.engine._fan_fold_fixture import FAN_CASES

_KEY_COLS = ["node", "param", "period", "time"]


def _provider_from_workdir(workdir: Path):
    """Reconstruct a Provider from every CSV in input/ + solve_data/.

    Dual-registers each frame under ``"<parent>/<stem>"`` AND the bare
    ``"<stem>"`` key so both ``_provider_key``-qualified lookups and any
    bare lookup resolve (design §6 / S6 — glob, do not under-register).
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()
    for parent in ("input", "solve_data"):
        d = workdir / parent
        if not d.is_dir():
            continue
        for csv_path in sorted(d.glob("*.csv")):
            try:
                df = pl.read_csv(csv_path)
            except Exception:
                continue
            stem = csv_path.stem
            provider.put(f"{parent}/{stem}", df)
            provider.put(stem, df)
    return provider


def _assert_parity(df_legacy: pl.DataFrame, df_vec: pl.DataFrame,
                   label: str) -> str:
    """Tier-A strict ``.equals``; demote to Tier B only on pure ULP drift.

    Returns ``"A"`` if byte-parity held, ``"B"`` if a legitimate
    last-ULP float-only demotion was applied (max ``|Δ| ≤ ~1e-12``).
    Raises on any structural mismatch or a non-ULP value difference
    (a real bug must STOP the gate, not be loosened away).
    """
    assert df_vec.columns == df_legacy.columns, (
        f"{label}: column mismatch legacy={df_legacy.columns} "
        f"vec={df_vec.columns}")
    assert df_vec.shape == df_legacy.shape, (
        f"{label}: shape mismatch legacy={df_legacy.shape} "
        f"vec={df_vec.shape}")

    if df_vec.equals(df_legacy):
        return "A"

    # Key columns MUST match byte-for-byte (row order + identity); only
    # the value column may drift, and only by last-ULP float noise.
    assert df_vec.select(_KEY_COLS).equals(df_legacy.select(_KEY_COLS)), (
        f"{label}: KEY columns differ — this is a structural bug, NOT a "
        f"float-ULP demotion. STOP.")

    leg_v = df_legacy["value"].cast(pl.Float64)
    vec_v = df_vec["value"].cast(pl.Float64)
    diff = (leg_v - vec_v).abs()
    max_abs = diff.max() or 0.0
    # Tolerance parity: |Δ| ≤ ~1e-12 (Tier B). Anything larger is a real
    # bug — fail loudly with the offending rows.
    if max_abs > 1e-12:
        bad = df_legacy.with_columns(
            df_vec["value"].alias("vec_value"),
            diff.alias("abs_diff"),
        ).filter(pl.col("abs_diff") > 1e-12)
        raise AssertionError(
            f"{label}: value drift {max_abs:.3e} exceeds Tier-B tolerance "
            f"1e-12 — NOT a last-ULP demotion, a real bug:\n{bad}")
    return "B"


# --- Standard two (Tier A expected) ----------------------------------------

_STD_CASES = [
    ("fullYear", "main", False),
    ("2_day_stochastic_dispatch", "stochastic", True),
]


@pytest.mark.parametrize("scenario,db_fixture,expect_rows", _STD_CASES)
def test_vectorized_pdtNode_matches_legacy_standard(
    scenario, db_fixture, expect_rows, scenario_workdir,
):
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    df_legacy = derive_pdtNode(inp, sdd, provider=p)
    df_vec = derive_pdtNode_vectorized(inp, sdd, provider=p)

    if expect_rows:
        assert df_legacy.height > 0, (
            f"{scenario}: legacy pdtNode produced 0 rows — gate vacuous")

    tier = _assert_parity(df_legacy, df_vec, scenario)
    # The standard fixtures ship empty pbt_node ⇒ no multi-term fold ⇒
    # Tier A must hold.
    assert tier == "A", (
        f"{scenario}: unexpected Tier-B demotion on a fixture with no "
        f"node fold (empty pbt_node) — investigate")


# --- Extra stochastic-fixture parity (pbt_node IS empty on real data) -------
#
# IMPORTANT FINDING (verified): the ``pbt_node.csv`` input that
# ``PdtLookup`` reads for pdtNode is populated by ``_specs.py`` ONLY from
# node params authored as a ``3d_map`` among
# ``{penalty_down, self_discharge_loss, availability,
# storage_state_reference_value}``.  The ``inflow`` 3d_map in the
# ``stochastics_pbt_inflow`` / ``branch2_parent_period`` fixtures routes
# to a DIFFERENT file — ``pbt_node_inflow.csv`` — consumed by the inflow-
# scaling path (design §5 ``pdtNodeInflow`` / ``inflow scaling``), NOT by
# pdtNode.  No shipped fixture authors a pbt_node-trigger param as a
# 3d_map, so ``pbt_node.csv`` is EMPTY for the pdtNode cascade on every
# real fixture — exactly like pbt_process / pbt_process_{source,sink}
# (build-log "Real fixtures ship EMPTY pbt_process").  The pdtNode fold
# therefore does NOT fire on real data; the SYNTHETIC test below is the
# real fold coverage.  These cases still gate the FULL vectorized derive
# (time-first pt/pd, p, def1, class-default, 0.0) against the legacy
# scalar cascade on real-shaped node domains through both stochastic
# fixtures — non-vacuous via the height check on the node domain.

_EXTRA_STOCH_FIXTURES = ["stochastics_pbt_inflow", "branch2_parent_period"]


@pytest.mark.parametrize("db_fixture", _EXTRA_STOCH_FIXTURES)
def test_vectorized_pdtNode_extra_stochastic_fixtures(
    db_fixture, scenario_workdir,
):
    work = scenario_workdir("2_day_stochastic_dispatch", db_fixture=db_fixture)
    p = _provider_from_workdir(work)
    inp = work / "input"
    sdd = work / "solve_data"

    # Document the empty-pbt_node reality (the pdtNode fold cannot fire).
    pbt = p.get("solve_data/pbt_node")
    pbt_h = 0 if pbt is None else pbt.height
    assert pbt_h == 0, (
        f"{db_fixture}: pbt_node unexpectedly NON-empty ({pbt_h} rows) — "
        f"a fixture now authors a pbt_node-trigger param as 3d_map; the "
        f"pdtNode fold can fire on real data and this test should add a "
        f"non-vacuous fold assertion + re-check the tier")

    df_legacy = derive_pdtNode(inp, sdd, provider=p)
    df_vec = derive_pdtNode_vectorized(inp, sdd, provider=p)
    # Non-vacuous: the node domain on these stochastic fixtures is
    # non-empty, so the full cascade (incl. class-default + def1) runs.
    assert df_legacy.height > 0, (
        f"{db_fixture}: legacy pdtNode produced 0 rows — gate vacuous")

    tier = _assert_parity(df_legacy, df_vec, db_fixture)
    # No fold ⇒ no multi-term sum ⇒ Tier A must hold.
    assert tier == "A", (
        f"{db_fixture}: unexpected Tier-B demotion with empty pbt_node "
        f"(no fold) — investigate")
    print(f"\n[pdtNode parity] {db_fixture}: Tier {tier} "
          f"(pbt_node empty — fold not exercised on real data)")


# --- Synthetic fold-coverage: class-default + def1 + time_first ordering ----

def test_vectorized_pdtNode_synthetic_fold_and_cascade(tmp_path):
    """Drive the vectorized derive vs the legacy scalar cascade on a
    synthetic fixture that exercises EVERY differentiating branch of
    ``pdtNode``:

    * **Stoch fold (branch 1):** ``riverA`` (stochastic) under ``d1``
      sums two branches ``b1``+``b2`` of its ``inflow`` pbt → 1+2 = 3.0.
    * **Stoch fall-through (S2/§12.6):** ``riverA`` is stochastic but has
      no branch under ``dchild`` (branch-1 miss) → falls through to the
      branch-2 parent fold → 2.0.
    * **Multi-parent multiplicity (S2):** ``riverB`` (non-stoch) has a
      single ``(b1, s1, t1)`` pbt value shared by two parents
      ``pA``/``pB`` of ``dchild`` → summed ONCE PER PARENT = 20.0.
    * **time-first (branch 3 before 4):** ``lake.storage`` has BOTH a
      ``pt`` value (time-first → 7.0) and a ``pd`` value (5.0) for the
      same cell → time-first must pick ``pt`` = 7.0.
    * **``pd`` only (branch 4):** ``lake.cost`` has only a ``pd`` value
      → 9.0.
    * **``p`` (branch 5):** ``lake.eff`` has only a scalar ``p`` value
      → 0.5.
    * **def1 (branch 6):** ``lake.availability`` (∈ NODE_PARAM_DEF1)
      with no pt/pd/p → 1.0.
    * **class-default (branch 7):** ``lake.penalty`` has no pt/pd/p, is
      NOT in def1, but has a class-default → 3.14.
    * **literal 0.0 (branch 8):** ``lake.other`` resolves nowhere → 0.0.

    The legacy ``derive_pdtNode`` (scalar ``PdtLookup.get`` loop) is the
    oracle; ``derive_pdtNode_vectorized`` must be byte-identical (Tier A;
    these are small integer-valued sums, no ULP drift).
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    # pbt_node: (node, param, branch, time_start, time, value)
    put("solve_data", "pbt_node", pl.DataFrame({
        "node": ["riverA", "riverA", "riverB"],
        "param": ["inflow", "inflow", "inflow"],
        "branch": ["b1", "b2", "b1"],
        "time_start": ["s1", "s1", "s1"],
        "time": ["t1", "t1", "t1"],
        "value": [1.0, 2.0, 10.0],
    }))
    # pd_node: (node, param, period, value)
    put("input", "pd_node", pl.DataFrame({
        "node": ["lake", "lake"],
        "param": ["storage", "cost"],
        "period": ["d1", "d1"],
        "value": [5.0, 9.0],
    }))
    # pt_node: (node, param, time, value) — time-first override on storage.
    put("solve_data", "pt_node", pl.DataFrame({
        "node": ["lake"],
        "param": ["storage"],
        "time": ["t1"],
        "value": [7.0],
    }))
    # p_node: (node, param, value)
    put("input", "p_node", pl.DataFrame({
        "node": ["lake"],
        "param": ["eff"],
        "value": [0.5],
    }))
    # default_values: class-default lookup for "node" class. The reader
    # filters to the node class and yields {param: value}.
    put("input", "default_values", pl.DataFrame({
        "class": ["node", "node"],
        "parameter": ["penalty", "availability"],
        "value": [3.14, 99.0],
    }))
    # first_timesteps: (period, time_start) → ts_for_d.
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "dchild"], "time_start": ["s1", "s1"]}))
    # Every period is a fan period whose fan start is its own first step (its tiers stay [own]); without
    # these rows each period would lie before the reveal, where only the solve's start is read (item 250).
    put("solve_data", "branch_time_start", pl.DataFrame({
        "period": ["d1", "dchild"], "step": ["s1", "s1"], "read_first": ["own", "own"]}))
    # solve_branch__time_branch: (period, branch) → tb_for_d. d1 has 2
    # branches; parents pA/pB each have b1; dchild has NONE (forces the
    # stoch fall-through for riverA).
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1", "d1", "pA", "pB"],
        "branch": ["b1", "b2", "b1", "b1"]}))
    # period__branch: read with key_col=1 → pe_for_d[child] = [parent].
    put("solve_data", "period__branch", pl.DataFrame({
        "parent": ["pA", "pB"], "child": ["dchild", "dchild"]}))
    # group__node + groupIncludeStochastics → riverA is stochastic.
    put("input", "group__node", pl.DataFrame({
        "group": ["g"], "node": ["riverA"]}))
    put("input", "groupIncludeStochastics", pl.DataFrame({"group": ["g"]}))
    # Domain (node, param) — entity-major order preserved.
    put("solve_data", "node__TimeParam_in_use", pl.DataFrame({
        "node": ["riverA", "riverB", "lake", "lake", "lake", "lake",
                 "lake", "lake"],
        "param": ["inflow", "inflow", "storage", "cost", "eff",
                  "availability", "penalty", "other"],
    }))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "dchild"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"

    df_legacy = derive_pdtNode(inp, sdd, provider=provider)
    df_vec = derive_pdtNode_vectorized(inp, sdd, provider=provider)

    # Oracle sanity: every differentiating branch landed where expected.
    # (node, param, period) -> value-string (time is constant "t1").
    leg = {(r[0], r[1], r[2]): r[4] for r in df_legacy.iter_rows()}
    assert leg[("riverA", "inflow", "d1")] == repr(3.0), leg     # stoch fold
    assert leg[("riverA", "inflow", "dchild")] == repr(2.0), leg  # fall-thru
    assert leg[("riverB", "inflow", "dchild")] == repr(20.0), leg  # multi-par
    assert leg[("lake", "storage", "d1")] == repr(7.0), leg      # time-first
    assert leg[("lake", "cost", "d1")] == repr(9.0), leg         # pd only
    assert leg[("lake", "eff", "d1")] == repr(0.5), leg          # p scalar
    assert leg[("lake", "availability", "d1")] == repr(1.0), leg  # def1
    assert leg[("lake", "penalty", "d1")] == repr(3.14), leg     # class-def
    assert leg[("lake", "other", "d1")] == repr(0.0), leg        # literal 0

    assert df_vec.equals(df_legacy), (
        f"synthetic pdtNode cascade: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}")


def test_vectorized_pdtNode_branch_period_inherits_anchor_pd(tmp_path):
    """A stochastic branch period without a ``pd`` row of its own takes its
    anchor period's ``pd`` row (#55), in both twins; time-first still holds:

    * ``city.penalty_up``: ``pd`` only for the anchor ``d1`` (8000) →
      ``d1_b`` gets 8000, not the class default 5000 (non-vacuous).
    * ``lake.storage``: ``pt`` 7.0 and anchor ``pd`` 5.0 → ``pt`` wins in
      ``d1_b`` as in ``d1`` (time-first, the inherited ``pd`` is below it).
    * ``lake.cost``: ``d1_b`` authors its own ``pd`` (2.0) → it wins.
    * ``lake.flow``: a genuine period x time (``pdt``) row for the anchor on
      top of a ``pd`` row AND a ``pt`` row of the same param — the pdt tier
      outranks both on the anchor, and ``d1_b`` (no pdt row of its own)
      inherits the anchor's pdt row rather than falling through to ``pt``.
    """
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    put("solve_data", "pbt_node", pl.DataFrame(schema={
        "node": pl.Utf8, "param": pl.Utf8, "branch": pl.Utf8,
        "time_start": pl.Utf8, "time": pl.Utf8, "value": pl.Float64}))
    put("input", "pd_node", pl.DataFrame({
        "node": ["city", "lake", "lake", "lake", "lake"],
        "param": ["penalty_up", "storage", "cost", "cost", "flow"],
        "period": ["d1", "d1", "d1", "d1_b", "d1"],
        "value": [8000.0, 5.0, 9.0, 2.0, 10.0],
    }))
    put("solve_data", "pt_node", pl.DataFrame({
        "node": ["lake", "lake"], "param": ["storage", "flow"], "time": ["t1", "t1"],
        "value": [7.0, 20.0]}))
    put("solve_data", "pdt_node", pl.DataFrame({
        "node": ["lake"], "param": ["flow"], "period": ["d1"], "time": ["t1"],
        "value": [30.0]}))
    put("input", "p_node", pl.DataFrame(schema={
        "node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("input", "default_values", pl.DataFrame({
        "class": ["node"], "parameter": ["penalty_up"], "value": [5000.0]}))
    put("solve_data", "first_timesteps", pl.DataFrame({
        "period": ["d1", "d1_b"], "time_start": ["s1", "s1"]}))
    # Every period is a fan period whose fan start is its own first step (its tiers stay [own]); without
    # these rows each period would lie before the reveal, where only the solve's start is read (item 250).
    put("solve_data", "branch_time_start", pl.DataFrame({
        "period": ["d1", "d1_b"], "step": ["s1", "s1"], "read_first": ["own", "own"]}))
    put("solve_data", "solve_branch__time_branch", pl.DataFrame({
        "period": ["d1"], "branch": ["b1"]}))
    put("solve_data", "period__branch", pl.DataFrame({
        "period": ["d1", "d1"], "branch": ["d1", "d1_b"]}))
    put("input", "group__node", pl.DataFrame(schema={
        "group": pl.Utf8, "node": pl.Utf8}))
    put("input", "groupIncludeStochastics", pl.DataFrame(schema={
        "group": pl.Utf8}))
    put("solve_data", "node__TimeParam_in_use", pl.DataFrame({
        "node": ["city", "lake", "lake", "lake"],
        "param": ["penalty_up", "storage", "cost", "flow"]}))
    put("solve_data", "steps_in_use", pl.DataFrame({
        "period": ["d1", "d1_b"], "time": ["t1", "t1"]}))

    inp = tmp_path / "input"
    sdd = tmp_path / "solve_data"
    df_legacy = derive_pdtNode(inp, sdd, provider=provider)
    df_vec = derive_pdtNode_vectorized(inp, sdd, provider=provider)

    leg = {(r[0], r[1], r[2]): r[4] for r in df_legacy.iter_rows()}
    assert leg[("city", "penalty_up", "d1_b")] == repr(8000.0), leg
    assert leg[("lake", "storage", "d1")] == leg[("lake", "storage", "d1_b")] == repr(7.0), leg
    assert leg[("lake", "cost", "d1")] == repr(9.0), leg
    assert leg[("lake", "cost", "d1_b")] == repr(2.0), leg
    assert leg[("lake", "flow", "d1")] == repr(30.0), "pdt must outrank pt and pd on the anchor"
    assert leg[("lake", "flow", "d1_b")] == repr(30.0), "d1_b must inherit the anchor's pdt row, not fall through"
    assert df_vec.equals(df_legacy), (
        f"branch-period pd: vectorized != legacy.\n"
        f"legacy:\n{df_legacy}\nvec:\n{df_vec}")


@pytest.mark.parametrize("case", FAN_CASES)
def test_vectorized_pdtNode_fan_start_fallback(case, tmp_path):
    """Items 228 / 255: a non-empty ``fan_for_d`` (``branch_time_start``) -- each fold cell reads the forecast at
    the fan start (``read_first`` = ``fan``, as every solve writes it; a cell it does not cover that a later
    forecast does is an input error in both twins), or at its period's own first step first in a frame that says
    ``own`` -- byte-identical in both twins,
    values hand-derived in
    :mod:`tests.engine._fan_fold_fixture`."""
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from tests.engine._fan_fold_fixture import (
        STEPS, expected_fold, fan_case, pbt_frame, put_fan_scaffold, uncovered_error)

    provider = FlexDataProvider()

    def put(parent: str, stem: str, df: pl.DataFrame) -> None:
        provider.put(f"{parent}/{stem}", df)
        provider.put(stem, df)

    rolling, rows = fan_case(case)
    put_fan_scaffold(put, rolling=rolling)
    put("solve_data", "pbt_node", pbt_frame({"node": "entity", "param": "cap"}, rows=rows))
    put("input", "pd_node", pl.DataFrame(schema={
        "node": pl.Utf8, "param": pl.Utf8, "period": pl.Utf8, "value": pl.Float64}))
    put("solve_data", "pt_node", pl.DataFrame(schema={
        "node": pl.Utf8, "param": pl.Utf8, "time": pl.Utf8, "value": pl.Float64}))
    put("input", "p_node", pl.DataFrame(schema={
        "node": pl.Utf8, "param": pl.Utf8, "value": pl.Float64}))
    put("input", "default_values", pl.DataFrame(schema={
        "class": pl.Utf8, "parameter": pl.Utf8, "value": pl.Float64}))
    put("input", "group__node", pl.DataFrame({"group": ["g"], "node": ["X"]}))
    put("solve_data", "node__TimeParam_in_use", pl.DataFrame({
        "node": ["X", "Y"], "param": ["cap", "cap"]}))

    inp, sdd = tmp_path / "input", tmp_path / "solve_data"
    if case == "uncovered":
        for derive in (derive_pdtNode, derive_pdtNode_vectorized):
            with pytest.raises(FlexToolConfigError, match=uncovered_error("node 'Y'", "cap")):
                derive(inp, sdd, provider=provider)
        return
    df_legacy = derive_pdtNode(inp, sdd, provider=provider)
    df_vec = derive_pdtNode_vectorized(inp, sdd, provider=provider)
    leg = {(r[0], r[2], r[3]): r[4] for r in df_legacy.iter_rows()}
    for e in ("X", "Y"):
        for d, t in STEPS:
            assert leg[(e, d, t)] == repr(expected_fold(rolling).get((e, d, t), 0.0)), (e, d, t, leg)
    assert df_vec.equals(df_legacy), f"fan fallback: vectorized != legacy.\nlegacy:\n{df_legacy}\nvec:\n{df_vec}"
