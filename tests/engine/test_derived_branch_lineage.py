"""Slice A — scenario-lineage frames: unit + integration tests.

Design: ``specs/sliceA_lineage_frames_design.md`` (r3) §4/§5, tests
T1-T12.  The two frames under test — ``dd_same_scenario (d, d_other)``
and ``pd_non_anticipativity (d, b)`` — are pure bookkeeping consumed
by NOTHING at HEAD (recourse plan §6b Slice A; no behavior change).

T1-T8 are solver-free: they seed a :class:`FlexDataProvider` with
in-memory frames (``period__branch`` cols ``period,branch``;
``period_in_use_set`` col ``period``; ``solve_branch__time_branch``
cols ``period,branch``) and call the builders with a temp workdir.
T9 pins the region-filter pass-through.  T10-T12 are integration
assertions on the 4.0.3 multi-period stochastic fixture
(``tests/fixtures/stoch_two_period.json``), including the first
stochastic dump round-trip (T12 — the lineage rebuild inputs survive a
``dump_csvs``; the dump → reload half, with the run-metadata copy, is in
``tests/csv_reload/test_derived_branch_lineage.py``).
"""
from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from flextool.engine.preprocess.params._branch import (
    dd_same_scenario_annuity_df,
    dd_same_scenario_df,
    pd_non_anticipativity_df,
)
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from tests._support.load import load_scenario


# ---------------------------------------------------------------------------
# Harness — provider seeding (design §5: keys per ``_provider_key`` =
# ``"solve_data/<stem>"``; builders require a non-None workdir to
# consult the provider).
# ---------------------------------------------------------------------------


def _mk_provider(pb=None, piu=None, sbtb=None) -> FlexDataProvider:
    prov = FlexDataProvider()
    if pb is not None:
        prov.put("solve_data/period__branch",
                 pl.DataFrame({"period": [r[0] for r in pb],
                               "branch": [r[1] for r in pb]}))
    if piu is not None:
        prov.put("solve_data/period_in_use_set",
                 pl.DataFrame({"period": list(piu)}))
    if sbtb is not None:
        prov.put("solve_data/solve_branch__time_branch",
                 pl.DataFrame({"period": [r[0] for r in sbtb],
                               "branch": [r[1] for r in sbtb]}))
    return prov


def _build(tmp_path, pb=None, piu=None, sbtb=None, branch_start=None):
    prov = _mk_provider(pb=pb, piu=piu, sbtb=sbtb)
    dd = dd_same_scenario_df(tmp_path, provider=prov)
    pd_na = pd_non_anticipativity_df(tmp_path, provider=prov,
                                     branch_start=branch_start)
    return dd, pd_na


def _rows(df: pl.DataFrame) -> list[tuple[str, str]]:
    return [(str(a), str(b)) for a, b in df.rows()]


# ---------------------------------------------------------------------------
# §4.1 fixture-shaped inputs (multi-period continuation fan-out)
# ---------------------------------------------------------------------------

PB_41 = [
    ("p2035", "p2035"),
    ("p2035", "p2035_rlz"),
    ("p2035", "p2035_low"),
    ("p2040", "p2040"),
    ("p2040", "p2040_rlz"),
    ("p2040", "p2040_low"),
]
PIU_41 = ["p2035", "p2035_low", "p2040", "p2040_low"]
SBTB_41 = [
    ("p2035_low", "low"),
    ("p2040_low", "low"),
    ("p2035", "rlz"),
    ("p2040", "rlz"),
]

# §4.1 — exactly 8 rows, in this order (Utf8-key sort).
DD_41 = [
    ("p2035", "p2035"),
    ("p2035", "p2040"),
    ("p2035_low", "p2035_low"),
    ("p2035_low", "p2040_low"),
    ("p2040", "p2035"),
    ("p2040", "p2040"),
    ("p2040_low", "p2035_low"),
    ("p2040_low", "p2040_low"),
]

# The §4.1 table for the REAL re-grounded ``stoch_two_period`` fixture
# (R3): a genuine {high, low} forecast fan (n_real = 0), so both forecast
# scenario paths are active in period_in_use — the realized/reference
# trunk path (p2035 ↔ p2040) PLUS the high path PLUS the low path, 12 rows
# (vs the hand-crafted 8-row ``DD_41`` whose PB_41 keeps ``_rlz`` as a
# metadata-only, not-in-use member).
DD_41_STOCH = [
    ("p2035", "p2035"),
    ("p2035", "p2040"),
    ("p2035_high", "p2035_high"),
    ("p2035_high", "p2040_high"),
    ("p2035_low", "p2035_low"),
    ("p2035_low", "p2040_low"),
    ("p2040", "p2035"),
    ("p2040", "p2040"),
    ("p2040_high", "p2035_high"),
    ("p2040_high", "p2040_high"),
    ("p2040_low", "p2035_low"),
    ("p2040_low", "p2040_low"),
]

# §4.3 — deterministic multi-period all-pairs.
DD_43 = [
    ("p2035", "p2035"),
    ("p2035", "p2040"),
    ("p2040", "p2035"),
    ("p2040", "p2040"),
]


def test_t1_multi_period_fan(tmp_path):
    """T1 (§4.1): the 8-row table verbatim, including row ORDER;
    pd_non_anticipativity empty with schema (d, b)."""
    dd, pd_na = _build(tmp_path, pb=PB_41, piu=PIU_41, sbtb=SBTB_41)
    assert dd.columns == ["d", "d_other"]
    assert _rows(dd) == DD_41
    assert pd_na.columns == ["d", "b"]
    assert pd_na.height == 0


def test_t2_single_period_fan(tmp_path):
    """T2 (§4.2): 2day single-period shape — 4 singleton leaves →
    exactly the 4 self-rows."""
    pb = [
        ("period1", "period1"),
        ("period1", "period1_realized"),
        ("period1", "period1_upper"),
        ("period1", "period1_lower"),
        ("period1", "period1_mid"),
    ]
    piu = ["period1", "period1_upper", "period1_lower", "period1_mid"]
    sbtb = [
        ("period1_upper", "upper"),
        ("period1_lower", "lower"),
        ("period1_mid", "mid"),
        ("period1", "realized"),
    ]
    dd, pd_na = _build(tmp_path, pb=pb, piu=piu, sbtb=sbtb)
    assert _rows(dd) == [
        ("period1", "period1"),
        ("period1_lower", "period1_lower"),
        ("period1_mid", "period1_mid"),
        ("period1_upper", "period1_upper"),
    ]
    assert pd_na.height == 0


def test_t3_deterministic_multi_period(tmp_path):
    """T3 (§4.3): PB self-rows only, SBTB empty → single trunk leaf →
    all-pairs (4 rows)."""
    pb = [("p2035", "p2035"), ("p2040", "p2040")]
    dd, pd_na = _build(tmp_path, pb=pb, piu=["p2035", "p2040"])
    assert _rows(dd) == DD_43
    assert pd_na.height == 0


def test_t4_rolling_realized_only(tmp_path):
    """T4 (§4.4): rolling per-roll shape — metadata realized fan member
    (∉ PIU) excluded → single self-row."""
    pb = [("period1", "period1"), ("period1", "period1_realized")]
    sbtb = [("period1", "realized")]
    dd, pd_na = _build(tmp_path, pb=pb, piu=["period1"], sbtb=sbtb)
    assert _rows(dd) == [("period1", "period1")]
    assert pd_na.height == 0


def test_t5_degenerate_empty(tmp_path):
    """T5 (§4.5): empty PIU → both frames empty, typed schemas, no
    raise."""
    dd, pd_na = _build(tmp_path)
    assert dd.columns == ["d", "d_other"]
    assert dd.height == 0
    assert pd_na.columns == ["d", "b"]
    assert pd_na.height == 0


def test_t5b_no_period_branch_single_trunk(tmp_path):
    """T5b (deterministic fallback, design §3.1 impl note): PB silent
    entirely (fixture-only loads without per-solve scaffolding) but
    PIU populated → single trunk leaf over PIU (all-pairs), no raise —
    mirrors ``pd_branch_weight_lf``'s ``pb.height == 0`` branch."""
    dd, pd_na = _build(tmp_path, piu=["p2035", "p2040"])
    assert _rows(dd) == DD_43
    assert pd_na.height == 0


def test_t6_failure_semantics_missing_sbtb(tmp_path):
    """T6 (§3.1.1): PB fan rows with PIU synthetic members but no SBTB
    frame → ValueError from BOTH builders (mirrors the
    ``dtttdt_from_source`` failure semantics — a silent
    treat-as-deterministic fallback would mis-classify branch periods
    onto the trunk)."""
    prov = _mk_provider(pb=PB_41, piu=PIU_41, sbtb=None)
    with pytest.raises(ValueError, match="solve_branch__time_branch"):
        dd_same_scenario_df(tmp_path, provider=prov)
    with pytest.raises(ValueError, match="solve_branch__time_branch"):
        pd_non_anticipativity_df(tmp_path, provider=prov)


def test_t6b_failure_semantics_orphan_synthetic(tmp_path):
    """T6b: a PIU member with neither a self-row nor a fan row in a
    populated PB → ValueError (inconsistent artefacts)."""
    pb = [("p2035", "p2035")]
    prov = _mk_provider(pb=pb, piu=["p2035", "p2040_low"],
                        sbtb=[("p2040_low", "low")])
    with pytest.raises(ValueError, match="no period__branch row"):
        dd_same_scenario_df(tmp_path, provider=prov)


# ---------------------------------------------------------------------------
# T7 — future mid-horizon shapes (§4.6; not constructible end-to-end
# at HEAD — the branch fan always starts at the solve's first step).
# Pins the parameterized ``branch_start`` rule TODAY so Slice E
# inherits a tested rule instead of dead code.
# ---------------------------------------------------------------------------

# Shape A — reveal at p2, NO pre-reveal copies.
PB_A = [
    ("p1", "p1"), ("p2", "p2"), ("p3", "p3"),
    ("p2", "p2_low"), ("p3", "p3_low"),
]
PIU_A = ["p1", "p2", "p3", "p2_low", "p3_low"]
SBTB_A = [
    ("p2_low", "low"), ("p3_low", "low"),
    ("p1", "rlz"), ("p2", "rlz"), ("p3", "rlz"),
]

# §4.6 shape A — 17 unique rows (9 + 9 − shared (p1, p1)).
DD_A = [
    ("p1", "p1"), ("p1", "p2"), ("p1", "p2_low"),
    ("p1", "p3"), ("p1", "p3_low"),
    ("p2", "p1"), ("p2", "p2"), ("p2", "p3"),
    ("p2_low", "p1"), ("p2_low", "p2_low"), ("p2_low", "p3_low"),
    ("p3", "p1"), ("p3", "p2"), ("p3", "p3"),
    ("p3_low", "p1"), ("p3_low", "p2_low"), ("p3_low", "p3_low"),
]

# Shape B — reveal at p2 WITH fanned pre-reveal copy p1_low.
PB_B = PB_A + [("p1", "p1_low")]
PIU_B = ["p1", "p1_low", "p2", "p3", "p2_low", "p3_low"]
SBTB_B = SBTB_A + [("p1_low", "low")]

# §4.6 shape B — 18 rows (R×R 9 ∪ L×L 9, disjoint).
_R_B = ["p1", "p2", "p3"]
_L_B = ["p1_low", "p2_low", "p3_low"]
DD_B = sorted([(x, y) for x in _R_B for y in _R_B]
              + [(x, y) for x in _L_B for y in _L_B])


def _branch_start_low_p2() -> pl.DataFrame:
    return pl.DataFrame({"time_branch": ["low"], "d_start": ["p2"]})


def test_t7_shape_a_pre_reveal_anchor(tmp_path):
    """T7/A (§4.6): the pre-reveal anchor p1 joins L(low); 17-row
    table; (p1, p2_low) present, (p2, p2_low) ABSENT; pd-NA empty
    under both the fallback and branch_start=[(low, p2)]."""
    dd, pd_na = _build(tmp_path, pb=PB_A, piu=PIU_A, sbtb=SBTB_A)
    rows = _rows(dd)
    assert rows == DD_A
    assert ("p1", "p2_low") in rows and ("p2_low", "p1") in rows
    assert ("p2", "p2_low") not in rows
    assert pd_na.height == 0
    _, pd_na_bs = _build(tmp_path, pb=PB_A, piu=PIU_A, sbtb=SBTB_A,
                         branch_start=_branch_start_low_p2())
    assert pd_na_bs.height == 0


def test_t7_shape_b_authority_rule(tmp_path):
    """T7/B (§4.6, F6 authority rule): the fanned pre-reveal copy
    p1_low displaces anchor p1 on L(low) — 18-row table with
    (p1, p2_low) and (p1, p1_low) ABSENT from dd (the trunk↔branch
    coupling at period 1 is expressed by pd_non_anticipativity, not
    leaf membership); dd is branch_start-independent; pd-NA empty
    under the min-anchor fallback (the documented under-detection) and
    exactly [(p1, p1_low)] under branch_start=[(low, p2)]."""
    dd, pd_na = _build(tmp_path, pb=PB_B, piu=PIU_B, sbtb=SBTB_B)
    rows = _rows(dd)
    assert rows == DD_B
    assert len(rows) == 18
    assert ("p1", "p2_low") not in rows
    assert ("p1", "p1_low") not in rows
    # pd-NA: fallback under-detects by design (§2.3) — empty.
    assert pd_na.height == 0
    dd_bs, pd_na_bs = _build(tmp_path, pb=PB_B, piu=PIU_B, sbtb=SBTB_B,
                             branch_start=_branch_start_low_p2())
    # dd content is branch_start-independent (the builder derives the
    # leaf rule's k from the min-anchor; the table above was
    # hand-derived under the TRUE reveal k — identical by §4.6).
    assert _rows(dd_bs) == rows
    assert _rows(pd_na_bs) == [("p1", "p1_low")]


def test_t7_symmetry_property(tmp_path):
    """Symmetry as a tested property (design §2.2): for every shape,
    (x, y) ∈ dd ⇔ (y, x) ∈ dd, and every PIU member has a self-row."""
    for pb, piu, sbtb in (
        (PB_41, PIU_41, SBTB_41),
        (PB_A, PIU_A, SBTB_A),
        (PB_B, PIU_B, SBTB_B),
    ):
        dd, _ = _build(tmp_path, pb=pb, piu=piu, sbtb=sbtb)
        pairs = set(_rows(dd))
        assert {(y, x) for x, y in pairs} == pairs
        for m in piu:
            assert (m, m) in pairs


def test_t8_metadata_exclusion(tmp_path):
    """T8: metadata-only fan members (realized-named ``_rlz`` and
    zero-weight members absent from PIU) appear in NEITHER frame —
    focused assert for failure localisation (keep-diagnostic-tests)."""
    pb = PB_41 + [("p2035", "p2035_zero"), ("p2040", "p2040_zero")]
    dd, pd_na = _build(tmp_path, pb=pb, piu=PIU_41, sbtb=SBTB_41)
    tokens = ({v for r in _rows(dd) for v in r}
              | {v for r in _rows(pd_na) for v in r})
    assert not [t for t in tokens if t.endswith(("_rlz", "_zero"))]
    assert _rows(dd) == DD_41


# ---------------------------------------------------------------------------
# T9 — region-filter pass-through (F8)
# ---------------------------------------------------------------------------


def _minimal_flex_data():
    from polar_high import Param

    from flextool.engine.preprocess.flex_data import FlexData

    dt = pl.DataFrame({"d": ["p2035"], "t": ["t0001"]})
    val = dt.with_columns(value=pl.lit(1.0))
    n_val = pl.DataFrame({"n": ["city"], "d": ["p2035"], "t": ["t0001"],
                          "value": [1.0]})
    d_val = pl.DataFrame({"d": ["p2035"], "value": [1.0]})
    return FlexData(
        dt=dt,
        p_step_duration=Param(("d", "t"), val),
        p_timestep_weight=Param(("d", "t"), val),
        p_inflation_op=Param(("d",), d_val),
        p_period_share=Param(("d",), d_val),
        nodeBalance=pl.DataFrame({"n": ["city"]}),
        p_inflow=Param(("n", "d", "t"), n_val),
        p_penalty_up=Param(("n", "d", "t"), n_val),
        p_penalty_down=Param(("n", "d", "t"), n_val),
    )


def test_t9_region_filter_pass_through(tmp_path):
    """T9 (F8): ``_drop_master_rows`` with non-empty node/proc/cn/group
    drop sets leaves both lineage frames row-identical — the generic
    field sweep filters only frames carrying n/p/e/cn/g columns, and
    the lineage frames carry only d/d_other/b."""
    from flextool.engine.solve._region_filter import _drop_master_rows
    from flextool.engine.preprocess.flex_data import FlexData

    dd, pd_na = _build(tmp_path, pb=PB_41, piu=PIU_41, sbtb=SBTB_41)
    rd = _minimal_flex_data()
    rd.dd_same_scenario = dd.clone()
    rd.pd_non_anticipativity = pd_na.clone()
    out = _drop_master_rows(
        rd,
        master_nodes={"city"},
        master_procs={"thermal"},
        master_cns=frozenset({"cn1"}),
        master_groups=frozenset({"g1"}),
    )
    assert out.dd_same_scenario.equals(dd)
    assert out.pd_non_anticipativity.equals(pd_na)
    # Secondary (trivially cheap): both fields are declared dataclass
    # fields — they survive the splitter's ``dataclasses.replace``
    # shallow copies by stdlib contract.
    names = {f.name for f in dataclasses.fields(FlexData)}
    assert {"dd_same_scenario", "pd_non_anticipativity"} <= names


# ---------------------------------------------------------------------------
# T10-T12 — integration on the 4.0.3 multi-period stochastic fixture
# (same fixture machinery as test_stochastic_continuation_fanout.py).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stoch_run(scenario_run):
    return scenario_run("stoch", db_fixture="stoch_two_period")


@pytest.fixture(scope="module")
def stoch_wf(stoch_run):
    """The cascade's work folder (T12 copies its metadata into the dump)."""
    return stoch_run.work_folder


@pytest.fixture(scope="module")
def det_run(scenario_run):
    return scenario_run("det", db_fixture="stoch_two_period")


@pytest.fixture(scope="module")
def stoch_data(stoch_run):
    """The last sub-solve's FlexData, loaded as production loads it (axis
    Enums OFF: dim columns are Utf8)."""
    return load_scenario(stoch_run)


@pytest.fixture(scope="module")
def stoch_enum_data(stoch_run):
    """The same load with the axis-Enum vocabulary activated — only for
    T11, which pins the Enum mechanism itself (production builds no Enums)."""
    return load_scenario(stoch_run, axis_enums=True)


def test_t10_integration_stoch(stoch_data):
    """T10: end-to-end ``load_flextool`` on the stoch workdir — the
    §4.1 table for the genuine {high, low} fan (12 rows: realized/reference
    trunk + both forecast paths), pd-NA empty (pins provider seeding +
    apply_derived_g wiring on the production (Utf8) load; T11 covers the
    same frames under enum activation, where the §3.1.1 null-guard
    engages)."""
    dd = stoch_data.dd_same_scenario
    assert dd is not None
    assert _rows(dd) == DD_41_STOCH
    pd_na = stoch_data.pd_non_anticipativity
    assert pd_na is not None
    assert pd_na.columns == ["d", "b"]
    assert pd_na.height == 0


def test_t10_integration_det(det_run):
    """T10 (det control): the §4.3 all-pairs table."""
    data = load_scenario(det_run)
    assert _rows(data.dd_same_scenario) == DD_43
    assert data.pd_non_anticipativity.height == 0


def test_t11_enum_hygiene(stoch_enum_data):
    """T11: under ``load_flextool`` activation the d/d_other columns
    carry the d-axis Enum (same dtype as ``period_in_use_set``'s d),
    ``b`` carries the branch Enum (same as ``period_branch_full``'s
    b), with zero nulls — pins the 4.0.4 continuation-token vocabulary
    splice against regression.  Loads with ``axis_enums=True``:
    production builds no axis Enums, so this pins the Enum mechanism
    itself."""
    stoch_data = stoch_enum_data
    dd = stoch_data.dd_same_scenario
    d_dtype = stoch_data.period_in_use_set.schema["d"]
    assert isinstance(d_dtype, pl.Enum)
    assert dd.schema["d"] == d_dtype
    assert dd.schema["d_other"] == d_dtype
    assert dd["d"].null_count() == 0
    assert dd["d_other"].null_count() == 0
    cats = d_dtype.categories.to_list()
    assert {"p2035_low", "p2040_low"} <= set(cats), (
        "continuation branch-period tokens missing from the d "
        "vocabulary (4.0.4 splice regression)"
    )
    pd_na = stoch_data.pd_non_anticipativity
    assert pd_na.schema["d"] == d_dtype
    b_dtype = stoch_data.period_branch_full.schema["b"]
    assert isinstance(b_dtype, pl.Enum)
    assert pd_na.schema["b"] == b_dtype


def test_t12_stoch_dump_keeps_lineage_inputs(stoch_run, stoch_wf, tmp_path):
    """T12 writer half (F2, r3): dump the cascade's stochastic FlexData and
    assert the lineage frames REBUILD identically from the CSVs on disk
    (no §3.1.1 inconsistency raise).  ``period_in_use_set`` is a FlexData
    field, so ``dump_csvs`` writes it; ``period__branch`` and
    ``solve_branch__time_branch`` are runner metadata (not FlexData
    fields), read here from the run's work folder.  First stochastic dump
    round-trip in the suite — existing round-trip tests cover
    deterministic fixtures only.

    The builders run directly against the CSVs.  The reload half (the
    run-metadata copy into the dump with ``copy_runner_meta`` and a
    ``load_from_dump`` of the dumped tree) is in
    ``tests/csv_reload/test_derived_branch_lineage.py`` (c24 S1-8)."""
    fd = stoch_run.last_flex_data
    out_dir = tmp_path / "dumped"
    fd.dump_csvs(out_dir)
    # F2 substance: the builders, fed the dumped FlexData field plus the
    # run's metadata CSVs, rebuild both frames identically — i.e. the
    # rebuild input the FlexData carries survived the dump.
    prov = FlexDataProvider()
    for stem, folder in (("period__branch", stoch_wf),
                         ("period_in_use_set", out_dir),
                         ("solve_branch__time_branch", stoch_wf)):
        p = folder / "solve_data" / f"{stem}.csv"
        assert p.exists(), f"rebuild input {stem}.csv missing from {folder}"
        prov.put(f"solve_data/{stem}", pl.read_csv(p))
    dd_rebuilt = dd_same_scenario_df(out_dir, provider=prov)
    pd_rebuilt = pd_non_anticipativity_df(out_dir, provider=prov)
    assert dd_rebuilt.columns == ["d", "d_other"]
    assert _rows(dd_rebuilt) == _rows(fd.dd_same_scenario)
    assert pd_rebuilt.columns == ["d", "b"]
    assert _rows(pd_rebuilt) == _rows(fd.pd_non_anticipativity)


# ---------------------------------------------------------------------------
# Annuity-lineage dedup (Slice E design §4.1 vs §8.1): the NPV ANNUITY /
# fixed-cost window frame ``dd_same_scenario_annuity`` collapses branch
# copies of the SAME calendar slot to one representative per anchor ``d``,
# so a shared pre-reveal trunk period is charged its fixed-cost annuity
# ONCE per calendar year — not once per surviving branch (the latent NPV
# over-count).  These are direct, solver-free frame-content pins; the
# 173,250 objective gate covers the dedup only INDIRECTLY, so a per-frame
# regression here localises the failure (keep-diagnostic-tests).
# ---------------------------------------------------------------------------

# Shared-trunk 2-period: reveal at p2040 — p2035 is a shared pre-reveal
# anchor (NO p2035_low branch copy); only p2040 fans to p2040_low.
PB_ST2 = [
    ("p2035", "p2035"),
    ("p2040", "p2040"),
    ("p2040", "p2040_rlz"),
    ("p2040", "p2040_low"),
]
PIU_ST2 = ["p2035", "p2040", "p2040_low"]
SBTB_ST2 = [
    ("p2040_low", "low"),
    ("p2035", "rlz"),
    ("p2040", "rlz"),
]

# Shared-trunk 3-period: reveal at p2040, low branch p2040_low + p2045_low
# (p2035 the shared pre-reveal anchor).
PB_ST3 = [
    ("p2035", "p2035"),
    ("p2040", "p2040"),
    ("p2045", "p2045"),
    ("p2040", "p2040_low"),
    ("p2045", "p2045_low"),
]
PIU_ST3 = ["p2035", "p2040", "p2045", "p2040_low", "p2045_low"]
SBTB_ST3 = [
    ("p2040_low", "low"), ("p2045_low", "low"),
    ("p2035", "rlz"), ("p2040", "rlz"), ("p2045", "rlz"),
]


def _cap_and_annuity(tmp_path, pb=None, piu=None, sbtb=None):
    """Build BOTH the capacity frame (``dd_same_scenario``) and the
    annuity frame from ONE seeded provider (provider.get is
    non-destructive) — the pair under comparison for the dedup pins."""
    prov = _mk_provider(pb=pb, piu=piu, sbtb=sbtb)
    cap = dd_same_scenario_df(tmp_path, provider=prov)
    ann = dd_same_scenario_annuity_df(tmp_path, provider=prov)
    return cap, ann


def test_annuity_shared_trunk_2period_dedup(tmp_path):
    """Shared-trunk 2-period (reveal at p2040): the trunk p2035 gets the
    DEDUPED annuity forward reachability {p2035, p2040} — NOT
    {p2035, p2040, p2040_low}.  The branch copy p2040_low is collapsed to
    its calendar anchor p2040 for the annuity frame, so the shared
    first-stage capacity is charged 2 period-annuities (one per physical
    period), never 3."""
    cap, ann = _cap_and_annuity(tmp_path, pb=PB_ST2, piu=PIU_ST2,
                                sbtb=SBTB_ST2)
    assert ann.columns == ["d", "d_other"]
    assert [r for r in _rows(ann) if r[0] == "p2035"] == [
        ("p2035", "p2035"), ("p2035", "p2040")]
    # The capacity frame KEEPS the branch copy (the shared first-stage
    # capacity must be alive in the low scenario's dispatch).
    assert ("p2035", "p2040_low") in _rows(cap)
    assert ("p2035", "p2040_low") not in _rows(ann)


def test_annuity_shared_trunk_3period_window(tmp_path):
    """Shared-trunk 3-period (reveal at p2040, low branch p2040_low +
    p2045_low) — the general-case fix: trunk p2035's annuity reachability
    is the window-3 calendar set {p2035, p2040, p2045}, NOT the window-5
    set that would carry both branch copies p2040_low / p2045_low."""
    _, ann = _cap_and_annuity(tmp_path, pb=PB_ST3, piu=PIU_ST3,
                              sbtb=SBTB_ST3)
    assert [r for r in _rows(ann) if r[0] == "p2035"] == [
        ("p2035", "p2035"), ("p2035", "p2040"), ("p2035", "p2045")]


def test_annuity_branch_window_not_collapsed(tmp_path):
    """A genuine recourse window is NOT wrongly collapsed: for the branch
    anchor p2040_low the annuity frame EQUALS the capacity frame on its
    leaf.  The real-named calendar siblings p2040 / p2045 live on the
    ``__anchor`` leaf, unreachable from p2040_low's branch leaf — so
    there is nothing to dedup its own window against."""
    cap, ann = _cap_and_annuity(tmp_path, pb=PB_ST3, piu=PIU_ST3,
                                sbtb=SBTB_ST3)
    cap_low = [r for r in _rows(cap) if r[0] == "p2040_low"]
    ann_low = [r for r in _rows(ann) if r[0] == "p2040_low"]
    assert ann_low == cap_low
    assert ann_low == [
        ("p2040_low", "p2035"),
        ("p2040_low", "p2040_low"),
        ("p2040_low", "p2045_low")]


def test_annuity_noop_parity(tmp_path):
    """No-op parity — the "true no-op for existing models" property: for a
    fan-at-first-step shape (no shared trunk) AND a deterministic shape,
    the annuity frame is BYTE-IDENTICAL to ``dd_same_scenario``.  Every
    leaf's members carry distinct calendar anchors, so the dedup finds
    nothing to collapse."""
    # Fan-at-first-step (§4.1 fixture shape) — each branch owns its own
    # first-step copy, so no anchor is a shared pre-reveal trunk.
    cap, ann = _cap_and_annuity(tmp_path, pb=PB_41, piu=PIU_41,
                                sbtb=SBTB_41)
    assert ann.equals(cap)
    assert _rows(ann) == DD_41
    # Deterministic multi-period (§4.3) — single trunk leaf, all-pairs.
    cap_d, ann_d = _cap_and_annuity(
        tmp_path, pb=[("p2035", "p2035"), ("p2040", "p2040")],
        piu=["p2035", "p2040"])
    assert ann_d.equals(cap_d)
    assert _rows(ann_d) == DD_43
