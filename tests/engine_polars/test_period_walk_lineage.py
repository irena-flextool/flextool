"""Slice B — pre-aggregation lineage support in the shared period walker.

Design: ``specs/sliceB_walker_lineage_design.md`` (r2), honoring the
Slice A gating contract (``specs/sliceA_lineage_frames_design.md``
§2.2/§2.5 as corrected by erratum E1).

Two layers (design §9):

* **Capability harness (W1-W5)** — fully solver-free, hand-built
  frames + a minimal stub source.  These pin the filter semantics
  (set-shape W1, PRE-aggregation W2, unfiltered W3, empty-frame W4/W4b,
  guard cases W5a/b/c) exactly against the design's expected tables.
* **Stochastic fixture pins (W6, W7, P1, P2, P4)** — *solver-required
  fixture, solver-free assertions*: ``scenario_workdir`` materialises
  the workdir by running the full cascade (incl. HiGHS); the tests then
  make pure-frame assertions against it — no golden depends on them.
* **Proof-obligation tests (P1-P4)** — the three named NPV obligations
  from Slice A §2.2 consequence 2, exercised via
  :func:`check_recourse_npv_preconditions` (hand-seeded providers for
  P3, live workdir for P1/P2/P4).

Every production call passes ``lineage=None``; this slice is inert
plumbing.  The filter is exercised here only.
"""
from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import _derived_npv as npv
from flextool.engine_polars._derived_branch import (
    check_recourse_npv_preconditions,
)
from flextool.engine_polars._derived_existing import (
    edd_history_lf,
    edd_invest_lookback_set_lf,
    edd_invest_set_lf,
)
from flextool.engine_polars._derived_params import (
    _p_years_d_lf,
    _period_in_use_set,
    _periodAll_from_source,
    _read_active_solve,
    _read_period_with_history,
    edd_divest_active_from_source,
)
from flextool.engine_polars._derived_walks import (
    WindowMethod,
    period_walk_iterator,
)
from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._input_source import seed_provider_from_dir


# ---------------------------------------------------------------------------
# §9.1 capability harness — stub source + hand-built frames
# ---------------------------------------------------------------------------

VOCAB = ["h2020", "p2035", "p2035_low", "p2040", "p2040_low"]
# ``period_in_use`` for the capability tests: h2020 is the HISTORY
# anchor and is DELIBERATELY not in PIU (design §9.1).
PIU = ["p2035", "p2035_low", "p2040", "p2040_low"]


class _StubSource:
    """Answers ``parameter(entity_class, name)`` probes from a dict;
    ``KeyError`` for everything else (matches ``_try_param`` /
    ``_solve_inflation_scalars`` fall-throughs)."""

    def __init__(self, params: dict[tuple[str, str], pl.DataFrame]):
        self._params = params

    def parameter(self, entity_class: str, name: str) -> pl.DataFrame:
        try:
            return self._params[(entity_class, name)]
        except KeyError:
            raise KeyError((entity_class, name)) from None

    def parameter_default(self, entity_class: str, name: str):
        raise KeyError((entity_class, name))


def _years_from_start_df() -> pl.DataFrame:
    # Hits arm ``_p_years_d_lf`` step 2 (``solve.years_from_start``) —
    # deterministic, no cumulative-sum mechanics.  h2020 → 0, the two
    # p2035 siblings → 15, the two p2040 siblings → 20.
    return pl.DataFrame({
        "name": ["s"] * 5,
        "period": VOCAB,
        "value": [0.0, 15.0, 15.0, 20.0, 20.0],
    })


def _years_represented_df() -> pl.DataFrame:
    # Deliberately UNEQUAL sibling year counts so every wrong d_all
    # subset is numerically distinguishable (design §9.1 W2).  This
    # intentionally violates NPV obligation (iii) — pure walker
    # mechanics, no obligation check runs here.
    return pl.DataFrame({
        "name": ["s"] * 4,
        "period": ["p2035", "p2035_low", "p2040", "p2040_low"],
        "value": [1.0, 1.0, 2.0, 3.0],
    })


def _stub_set() -> _StubSource:
    return _StubSource({("solve", "years_from_start"): _years_from_start_df()})


def _stub_factor() -> _StubSource:
    return _StubSource({
        ("solve", "years_from_start"): _years_from_start_df(),
        ("solve", "years_represented"): _years_represented_df(),
    })


def _ed_lf(d_dtype: pl.DataType | None = None) -> pl.LazyFrame:
    """Anchor frame ``[(e1, h2020), (e1, p2035), (e1, p2035_low)]``."""
    ds = ["h2020", "p2035", "p2035_low"]
    d_col = (pl.Series("d", ds, dtype=d_dtype) if d_dtype is not None
             else pl.Series("d", ds))
    return pl.DataFrame({"e": ["e1", "e1", "e1"]}).with_columns(
        d_col).lazy()


def _life_lf(d_dtype: pl.DataType | None = None) -> pl.LazyFrame:
    ds = ["h2020", "p2035", "p2035_low"]
    d_col = (pl.Series("d", ds, dtype=d_dtype) if d_dtype is not None
             else pl.Series("d", ds))
    return pl.DataFrame({"e": ["e1", "e1", "e1"],
                         "life": [30.0, 30.0, 30.0]}).with_columns(
        d_col).lazy()


# The Slice A §4.1 8-row ``dd_same_scenario`` table for the stoch
# fixture (leaf-pairs: same-leaf continuations + self rows).
LINEAGE_ROWS = [
    ("p2035", "p2035"), ("p2035", "p2040"),
    ("p2035_low", "p2035_low"), ("p2035_low", "p2040_low"),
    ("p2040", "p2035"), ("p2040", "p2040"),
    ("p2040_low", "p2035_low"), ("p2040_low", "p2040_low"),
]


def _lineage(d_dtype: pl.DataType | None = None,
             rows: list[tuple[str, str]] | None = None) -> pl.DataFrame:
    rows = LINEAGE_ROWS if rows is None else rows
    d = [r[0] for r in rows]
    do = [r[1] for r in rows]
    if d_dtype is not None:
        return pl.DataFrame({
            "d": pl.Series("d", d, dtype=d_dtype),
            "d_other": pl.Series("d_other", do, dtype=d_dtype),
        })
    return pl.DataFrame({"d": d, "d_other": do})


def _walk_set(source, lineage=None):
    return (period_walk_iterator(
                source, "s", _ed_lf(), PIU, VOCAB,
                window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
                life_lf=_life_lf(), factor_side=None, lineage=lineage)
            .collect().sort("e", "d", "d_all"))


def _walk_factor(source, lineage=None):
    return (period_walk_iterator(
                source, "s", _ed_lf(), PIU, VOCAB,
                window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
                life_lf=_life_lf(), factor_side="inv", lineage=lineage)
            .collect().sort("e", "d"))


def _rows(df: pl.DataFrame, cols) -> list[tuple]:
    return [tuple(r) for r in df.select(cols).rows()]


# --- W1 — set-shape capability (the REQUIRED test) ---------------------

W1_EXPECTED = [
    ("e1", "h2020", "p2035"),      # history anchor — passes uncond.
    ("e1", "h2020", "p2035_low"),  # history anchor — passes uncond.
    ("e1", "h2020", "p2040"),      # history anchor — passes uncond.
    ("e1", "h2020", "p2040_low"),  # history anchor — passes uncond.
    ("e1", "p2035", "p2035"),      # in lineage (self)
    ("e1", "p2035", "p2040"),      # in lineage (same leaf)
    ("e1", "p2035_low", "p2035_low"),  # in lineage (self)
    ("e1", "p2035_low", "p2040_low"),  # in lineage (same leaf)
]

# The four both-ends-PIU cross-leaf pairs the filter must REMOVE.
W1_ABSENT = [
    ("e1", "p2035", "p2035_low"),
    ("e1", "p2035", "p2040_low"),
    ("e1", "p2035_low", "p2035"),
    ("e1", "p2035_low", "p2040"),
]


def test_w1_set_shape_capability() -> None:
    """W1 (§9.1): ``factor_side=None`` with the 8-row lineage → exactly
    the 8 rows.  Contains BOTH mandated demonstrations: a
    history-anchored row passing through unconditionally AND
    both-ends-PIU cross-leaf pairs being removed."""
    out = _walk_set(_stub_set(), lineage=_lineage())
    got = _rows(out, ["e", "d", "d_all"])
    assert got == W1_EXPECTED
    for row in W1_ABSENT:
        assert row not in got, f"cross-leaf pair {row} should be removed"


# --- W2 — factor-side capability (proves PRE-aggregation) --------------

W2_EXPECTED = [
    ("e1", "h2020", 7.0),      # 1 + 1 + 2 + 3 (unfiltered — history)
    ("e1", "p2035", 3.0),      # 1 (p2035) + 2 (p2040)
    ("e1", "p2035_low", 4.0),  # 1 (p2035_low) + 3 (p2040_low)
]


def test_w2_factor_pre_aggregation() -> None:
    """W2 (§9.1): ``factor_side="inv"`` with the 8-row lineage → factor
    sums 7.0 / 3.0 / 4.0.  A post-aggregation filter cannot produce
    3.0/4.0 from the unfiltered 7.0 — this is the numerical proof that
    the injection is PRE-aggregation."""
    out = _walk_factor(_stub_factor(), lineage=_lineage())
    got = _rows(out, ["e", "d", "factor"])
    assert got == W2_EXPECTED


def test_w2_factor_none_control() -> None:
    """W2 control: ``lineage=None`` → every anchor sums the full 7.0."""
    out = _walk_factor(_stub_factor(), lineage=None)
    got = _rows(out, ["e", "d", "factor"])
    assert got == [
        ("e1", "h2020", 7.0),
        ("e1", "p2035", 7.0),
        ("e1", "p2035_low", 7.0),
    ]


# --- W3 — lineage=None unfiltered pin ----------------------------------

W3_EXPECTED = [
    ("e1", "h2020", "p2035"),
    ("e1", "h2020", "p2035_low"),
    ("e1", "h2020", "p2040"),
    ("e1", "h2020", "p2040_low"),
    ("e1", "p2035", "p2035"),
    ("e1", "p2035", "p2035_low"),
    ("e1", "p2035", "p2040"),
    ("e1", "p2035", "p2040_low"),
    ("e1", "p2035_low", "p2035"),
    ("e1", "p2035_low", "p2035_low"),
    ("e1", "p2035_low", "p2040"),
    ("e1", "p2035_low", "p2040_low"),
]


def test_w3_none_unfiltered() -> None:
    """W3 (§9.1): W1's inputs with ``lineage=None`` → the full 12-row
    table (fine-grained walker-semantics pin, golden-independent)."""
    out = _walk_set(_stub_set(), lineage=None)
    assert _rows(out, ["e", "d", "d_all"]) == W3_EXPECTED


# --- W4 — empty-frame contract edge (set shape) ------------------------


def test_w4_empty_frame_set() -> None:
    """W4 (§9.1): a 0-row typed lineage frame → exactly the 4 h2020
    rows (every both-ends-PIU pair removed; history passes)."""
    empty = pl.DataFrame(schema={"d": pl.Utf8, "d_other": pl.Utf8})
    out = _walk_set(_stub_set(), lineage=empty)
    assert _rows(out, ["e", "d", "d_all"]) == [
        ("e1", "h2020", "p2035"),
        ("e1", "h2020", "p2035_low"),
        ("e1", "h2020", "p2040"),
        ("e1", "h2020", "p2040_low"),
    ]


# --- W4b — empty-frame contract edge (factor shape) --------------------


def test_w4b_empty_frame_factor() -> None:
    """W4b (§9.1 F9): a 0-row lineage frame, ``factor_side="inv"`` →
    exactly ONE output row ``(e1, h2020, 7.0)``.  The p2035 / p2035_low
    anchors are ABSENT from the aggregate (all their pairs removed
    before ``group_by``, so no group exists) — NOT present with factor
    0.0.  The NPV consumers re-join with ``fill_null(0.0)``, so an
    absent anchor becomes a zero factor downstream."""
    empty = pl.DataFrame(schema={"d": pl.Utf8, "d_other": pl.Utf8})
    out = _walk_factor(_stub_factor(), lineage=empty)
    assert _rows(out, ["e", "d", "factor"]) == [("e1", "h2020", 7.0)]


# --- W5 — guard cases --------------------------------------------------


def test_w5a_duplicate_pair_set() -> None:
    """W5a (§9.1): a lineage with a DUPLICATED pair → output identical
    to W1 (the mandatory ``.unique()`` guard prevents join fan-out)."""
    dup = _lineage(rows=LINEAGE_ROWS + [("p2035", "p2035")])
    out = _walk_set(_stub_set(), lineage=dup)
    assert _rows(out, ["e", "d", "d_all"]) == W1_EXPECTED


def test_w5a_duplicate_pair_factor() -> None:
    """W5a factor leg: a duplicated pair must not double-count the
    factor sums (the lift-dict-not-raw-CSV lesson)."""
    dup = _lineage(rows=LINEAGE_ROWS + [("p2035", "p2040")])
    out = _walk_factor(_stub_factor(), lineage=dup)
    assert _rows(out, ["e", "d", "factor"]) == W2_EXPECTED


def test_w5b_uncastable_token_raises() -> None:
    """W5b (§9.1): a lineage token NOT castable into the Enum-typed
    ``ed_lf.d`` vocabulary → ``ValueError`` naming the token (run with
    a hand-Enum-typed anchor so the non-strict cast is non-trivial;
    a Utf8 walk dtype could never null)."""
    enum_d = pl.Enum(VOCAB)
    bad = pl.DataFrame({
        "d": pl.Series("d", ["p2035", "p9999"], dtype=pl.Utf8),
        "d_other": pl.Series("d_other", ["p2035", "p2040"], dtype=pl.Utf8),
    })
    with pytest.raises(ValueError, match="p9999"):
        period_walk_iterator(
            _stub_set(), "s", _ed_lf(enum_d), PIU, VOCAB,
            window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
            life_lf=_life_lf(enum_d), factor_side=None,
            lineage=bad).collect()


def test_w5c_enum_x_enum_missing_token_raises() -> None:
    """W5c (§9.1 F6): lineage columns typed as a DIFFERENT Enum than
    the walk's Enum-typed ``d``, containing one token absent from the
    walk vocabulary → ``ValueError`` from ``_assert_lineage_castable``
    (raise, never a silent null-drop).

    polars 1.40.1 empirical behavior this guards: a NON-STRICT cast
    between two different Enum vocabularies NULLS every missing token
    instead of raising — so it is the guard that raises, not polars."""
    enum_d = pl.Enum(VOCAB)
    other_enum = pl.Enum(["p2035", "p2035_low", "p2040", "p2040_low", "zzz"])
    bad = pl.DataFrame({
        "d": pl.Series("d", ["p2035", "zzz"], dtype=other_enum),
        "d_other": pl.Series("d_other", ["p2035", "p2040"], dtype=other_enum),
    })
    with pytest.raises(ValueError, match="zzz"):
        period_walk_iterator(
            _stub_set(), "s", _ed_lf(enum_d), PIU, VOCAB,
            window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
            life_lf=_life_lf(enum_d), factor_side=None,
            lineage=bad).collect()


def test_w5c_enum_x_enum_all_present_identical() -> None:
    """W5c (§9.1 F6): Enum×Enum with all tokens present in both
    vocabularies → the non-strict re-cast is value-preserving, output
    identical to W1."""
    enum_d = pl.Enum(VOCAB)
    lin_enum = pl.Enum(["p2035", "p2035_low", "p2040", "p2040_low"])
    out = (period_walk_iterator(
                _stub_set(), "s", _ed_lf(enum_d), PIU, VOCAB,
                window_method=WindowMethod.BOUNDED_INCLUSIVE_LOOKBACK,
                life_lf=_life_lf(enum_d), factor_side=None,
                lineage=_lineage(lin_enum))
           .collect().sort("e", "d", "d_all"))
    # walk-side ``d`` is Enum-typed, so compare against str values.
    got = [(str(e), str(d), str(da)) for e, d, da in
           out.select(["e", "d", "d_all"]).rows()]
    assert got == W1_EXPECTED


# ---------------------------------------------------------------------------
# §9.2 / §9.3 stochastic fixture pins — solver-required fixture,
# solver-free assertions.
# ---------------------------------------------------------------------------

WIND = "wind"
STOCH_SOLVE = "stoch_2p"


@pytest.fixture(scope="module")
def stoch_wf(scenario_workdir):
    """The ``stoch`` scenario workdir (cascade incl. HiGHS), built
    under ``FLEXTOOL_AUTOSCALE_STRICT=1`` — cached per (scenario,
    db_fixture) by the session-scoped factory."""
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("stoch", db_fixture="stoch_two_period")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def det_wf(scenario_workdir):
    """A deterministic workdir on the same fixture (no fan rows)."""
    return scenario_workdir("det", db_fixture="stoch_two_period")


def _reader(wf: Path):
    from flextool.engine_polars import SpineDbReader
    return SpineDbReader(wf / "tests.sqlite", "stoch")


def _det_reader(wf: Path):
    from flextool.engine_polars import SpineDbReader
    return SpineDbReader(wf / "tests.sqlite", "det")


def _provider(wf: Path) -> FlexDataProvider:
    provider = FlexDataProvider()
    if (wf / "input").exists():
        seed_provider_from_dir(provider, wf / "input", "input")
    if (wf / "solve_data").exists():
        seed_provider_from_dir(provider, wf / "solve_data", "solve_data")
    return provider


def _period_lists(wf: Path, reader, provider):
    active = _read_active_solve(wf, provider=provider)
    piu = _period_in_use_set(reader, active, wf, provider=provider)
    pwh = _read_period_with_history(wf, provider=provider) or list(piu)
    universe = _periodAll_from_source(reader, active, workdir=wf,
                                      provider=provider)
    return active, piu, pwh, universe


# --- W6 — the (e, p_k, p_j_b) structural pin AND §6.4 today-bug witness -


def test_w6_edd_invest_set_today_bug_witness(stoch_wf) -> None:
    """W6 (§9.2): call ``edd_invest_set_lf`` with hand ed_invest
    anchors ``[(wind, p2035), (wind, p2040)]`` on the real stoch
    cascade inputs.  wind's ``lifetime_method`` defaults to
    ``reinvest_automatic`` → UNBOUNDED_FORWARD; walker years via the
    §6.4 source fallback (``{p2035: 0.0, p2040: 1.0}``, synthetic
    members → 0.0).

    Predicted 5 rows AND ``(wind, p2040, p2040_low)`` ABSENT — the
    latter asserted as the TODAY-BUG WITNESS (erratum E1): under
    correct (byte-copied) years that pair would be present, and
    today's walker overwrite of the CSV seed drops it, so the absence
    IS the live first-stage-visibility break for continuation anchors.
    Slice D's year threading makes the pair appear — re-point the pin
    then."""
    wf = stoch_wf
    reader = _reader(wf)
    provider = _provider(wf)
    _, piu, pwh, _ = _period_lists(wf, reader, provider)
    ed_invest_lf = pl.LazyFrame({
        "e": [WIND, WIND], "d": ["p2035", "p2040"]})
    out = (edd_invest_set_lf(reader, STOCH_SOLVE, ed_invest_lf,
                             pwh, piu, wf)
           .collect())
    got = {(str(e), str(di), str(d)) for e, di, d in
           out.select(["e", "d_invest", "d"]).rows()}
    expected = {
        (WIND, "p2035", "p2035"),
        (WIND, "p2035", "p2035_low"),  # the surviving year-0 channel
        (WIND, "p2035", "p2040"),
        (WIND, "p2035", "p2040_low"),  # passes at fill-year 0 (§6.4)
        (WIND, "p2040", "p2040"),
    }
    assert got == expected
    # Erratum E1 today-bug witness: under correct byte-copied years
    # this pair WOULD be present; today's walker overwrite drops it.
    # Slice D's year-threading prerequisite resolves it — re-point here.
    assert (WIND, "p2040", "p2040_low") not in got


# --- W7 — non-vacuity + repeat-determinism pins ------------------------


def test_w7_edd_families_nonvacuous_deterministic(stoch_wf) -> None:
    """W7 (§9.2): the edd/divest legs are non-vacuous on the stoch
    workdir and repeat-deterministic (the None-path guarantees are
    structural — default arg + untouched code path — witnessed by
    W3/W6; this pins non-vacuity + determinism, not parity)."""
    wf = stoch_wf
    reader = _reader(wf)
    provider = _provider(wf)
    active, piu, pwh, _ = _period_lists(wf, reader, provider)

    # Walker join row order is not relied upon by any consumer
    # (design §4.2), so sort before ``.equals`` — the pin is
    # content-determinism, not row-stream identity.
    def _sorted(df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(df.columns)

    hist1 = edd_history_lf(reader, STOCH_SOLVE, pwh, piu, wf).collect()
    hist2 = edd_history_lf(reader, STOCH_SOLVE, pwh, piu, wf).collect()
    assert hist1.height > 0
    assert _sorted(hist1).equals(_sorted(hist2))

    ed_invest_lf = pl.LazyFrame({
        "e": [WIND, WIND], "d": ["p2035", "p2040"]})
    inv1 = edd_invest_set_lf(reader, STOCH_SOLVE, ed_invest_lf,
                             pwh, piu, wf).collect()
    inv2 = edd_invest_set_lf(reader, STOCH_SOLVE, ed_invest_lf,
                             pwh, piu, wf).collect()
    assert inv1.height > 0
    assert _sorted(inv1).equals(_sorted(inv2))

    lb1 = edd_invest_lookback_set_lf(
        reader, STOCH_SOLVE, ed_invest_lf, piu, wf).collect()
    lb2 = edd_invest_lookback_set_lf(
        reader, STOCH_SOLVE, ed_invest_lf, piu, wf).collect()
    assert _sorted(lb1).equals(_sorted(lb2))  # may be empty, but stable

    # #18 divest pair-former with a hand pd_divest frame.
    pd_divest = pl.DataFrame({"p": [WIND, WIND], "d": ["p2035", "p2040"]})
    dv1 = edd_divest_active_from_source(reader, STOCH_SOLVE, pd_divest)
    dv2 = edd_divest_active_from_source(reader, STOCH_SOLVE, pd_divest)
    assert dv1.height > 0
    assert _sorted(dv1).equals(_sorted(dv2))


class _InvestMethodOverlay:
    """Overlay source: delegates ``entities`` / ``parameter`` /
    everything else to the wrapped reader, but injects
    ``("unit", "invest_method")`` rows so an otherwise-vacuous invest
    grid becomes non-empty (design §9.2 W7 F5)."""

    def __init__(self, reader, rows: pl.DataFrame):
        self._reader = reader
        self._rows = rows

    def parameter(self, entity_class: str, name: str) -> pl.DataFrame:
        if (entity_class, name) == ("unit", "invest_method"):
            return self._rows
        return self._reader.parameter(entity_class, name)

    def __getattr__(self, attr):
        return getattr(self._reader, attr)


def test_w7_npv_variants_vacuity_and_overlay(stoch_wf) -> None:
    """W7 (§9.2): the three INVEST-METHOD-GATED NPV ``_lf`` variants are
    VACUOUS against the raw fixture (no ``entity__invest_method`` rows →
    empty ``_entity_invest_set`` anchor grids); asserted empty AS the
    vacuity pin.  ONE leg (``npv_invest_discounted_lf``) is made
    non-vacuous via an invest-method overlay stub — asserted non-empty
    and repeat-deterministic.

    DEVIATION from design §9.2 (recorded, design §implementation-notes):
    the design claimed all FOUR NPV legs are vacuous, but
    ``lifetime_fixed_cost_invest_lf`` is anchored on ``all_entities ×
    period_with_history`` (NOT ``_entity_invest_set``), so it is
    non-vacuous whenever the fixture has entities.  It is pinned here
    as non-empty + repeat-deterministic instead — a strictly stronger
    non-vacuity witness that also exercises the lineage-threaded ops
    walk on live data.  The None-path guarantee is unaffected."""
    wf = stoch_wf
    reader = _reader(wf)
    provider = _provider(wf)
    active, piu, pwh, universe = _period_lists(wf, reader, provider)
    period_invest = ["p2035"]

    def _sorted(df: pl.DataFrame) -> pl.DataFrame:
        return df.sort(df.columns)

    # The three invest/divest-method-gated legs — vacuity pin.
    assert npv.npv_invest_discounted_lf(
        reader, STOCH_SOLVE, period_invest, piu, universe
        ).collect().height == 0
    assert npv.npv_divest_discounted_lf(
        reader, STOCH_SOLVE, period_invest, piu, universe
        ).collect().height == 0
    assert npv.lifetime_fixed_cost_divest_lf(
        reader, STOCH_SOLVE, period_invest, piu, universe
        ).collect().height == 0

    # lifetime_fixed_cost_invest — all-entities anchored → non-vacuous;
    # pin non-empty + repeat-deterministic (see DEVIATION above).
    lfc1 = npv.lifetime_fixed_cost_invest_lf(
        reader, STOCH_SOLVE, pwh, piu, universe).collect()
    lfc2 = npv.lifetime_fixed_cost_invest_lf(
        reader, STOCH_SOLVE, pwh, piu, universe).collect()
    assert lfc1.height > 0
    assert _sorted(lfc1).equals(_sorted(lfc2))

    # Overlay makes the invest leg non-vacuous (wind → invest_period).
    overlay_rows = pl.DataFrame({"name": [WIND], "value": ["invest_period"]})
    overlay = _InvestMethodOverlay(reader, overlay_rows)
    ov1 = npv.npv_invest_discounted_lf(
        overlay, STOCH_SOLVE, period_invest, piu, universe).collect()
    ov2 = npv.npv_invest_discounted_lf(
        overlay, STOCH_SOLVE, period_invest, piu, universe).collect()
    assert ov1.height > 0  # rows exist (value 0.0 without invest_cost)
    assert _sorted(ov1).equals(_sorted(ov2))


# --- P1 / P2 — obligation checker on live workdirs ---------------------


def test_p1_preconditions_clean_stoch(stoch_wf) -> None:
    """P1 (§9.3): ``check_recourse_npv_preconditions`` → ``[]`` on the
    stoch workdir (weights 0.25/0.75 constant per chain; sum-to-1 per
    cohort; CSV year rows byte-equal)."""
    wf = stoch_wf
    reader = _reader(wf)
    provider = _provider(wf)
    active = _read_active_solve(wf, provider=provider)
    violations = check_recourse_npv_preconditions(
        wf, reader, active, provider=provider)
    assert violations == []


def test_p2_preconditions_clean_deterministic(det_wf) -> None:
    """P2 (§9.3): ``[]`` on a deterministic workdir (no fan rows →
    vacuously clean)."""
    wf = det_wf
    reader = _det_reader(wf)
    provider = _provider(wf)
    active = _read_active_solve(wf, provider=provider)
    violations = check_recourse_npv_preconditions(
        wf, reader, active, provider=provider)
    assert violations == []


# --- P3 — hand-built obligation violations (fully solver-free) ---------

_PB = [("p2035", "p2035"), ("p2035", "p2035_rlz"), ("p2035", "p2035_low"),
       ("p2040", "p2040"), ("p2040", "p2040_rlz"), ("p2040", "p2040_low")]
_PIU = ["p2035", "p2035_low", "p2040", "p2040_low"]
_SBTB = [("p2035_low", "low"), ("p2040_low", "low"),
         ("p2035", "rlz"), ("p2040", "rlz")]
_BW = [("p2035", 1.0), ("p2035_rlz", 1.0), ("p2035_low", 3.0),
       ("p2040", 1.0), ("p2040_rlz", 1.0), ("p2040_low", 3.0)]
_FT = [("p2035", "t0001"), ("p2035_low", "t0001"),
       ("p2040", "t0004"), ("p2040_low", "t0004")]
_PYD = [("p2035", 0.0), ("p2035_low", 0.0), ("p2040", 1.0), ("p2040_low", 1.0)]
_PYR = [("p2035", "0", 1.0), ("p2035_low", "0", 1.0),
        ("p2040", "1", 1.0), ("p2040_low", "1", 1.0)]


def _mk_provider(pb=None, piu=None, sbtb=None, bw=None, ft=None,
                 pyd=None, pyr=None) -> FlexDataProvider:
    """Slice A ``_mk_provider`` idiom, extended with the branch-weight
    cascade inputs and the two year CSVs the obligation checker reads."""
    p = FlexDataProvider()
    if pb is not None:
        p.put("solve_data/period__branch", pl.DataFrame(
            {"period": [r[0] for r in pb], "branch": [r[1] for r in pb]}))
    if piu is not None:
        p.put("solve_data/period_in_use_set",
              pl.DataFrame({"period": list(piu)}))
    if sbtb is not None:
        p.put("solve_data/solve_branch__time_branch", pl.DataFrame(
            {"period": [r[0] for r in sbtb], "branch": [r[1] for r in sbtb]}))
    if bw is not None:
        p.put("solve_data/solve_branch_weight", pl.DataFrame(
            {"branch": [r[0] for r in bw],
             "p_branch_weight_input": [r[1] for r in bw]}))
    if ft is not None:
        p.put("solve_data/first_timesteps", pl.DataFrame(
            {"period": [r[0] for r in ft], "time": [r[1] for r in ft]}))
    if pyd is not None:
        p.put("solve_data/p_years_d", pl.DataFrame(
            {"period": [r[0] for r in pyd], "value": [r[1] for r in pyd]}))
    if pyr is not None:
        p.put("solve_data/p_years_represented", pl.DataFrame(
            {"period": [r[0] for r in pyr],
             "years_from_solve": [r[1] for r in pyr],
             "p_years_represented": [r[2] for r in pyr]}))
    return p


def test_p3a_unequal_chain_weights(tmp_path) -> None:
    """P3a (§9.3): unequal ``pd_branch_weight`` along a chain → message
    (i)."""
    sbtb = [("p2035", "low"), ("p2035_low", "low"),
            ("p2040", "rlz"), ("p2040_low", "rlz")]
    v = check_recourse_npv_preconditions(
        tmp_path, provider=_mk_provider(
            pb=_PB, piu=_PIU, sbtb=sbtb, bw=_BW, ft=_FT,
            pyd=_PYD, pyr=_PYR))
    assert any("obligation (i)" in m for m in v), v


def test_p3b_cohort_sum_violation(tmp_path) -> None:
    """P3b (§9.3): a cohort whose weights do not sum to 1 → message
    (ii).  Distinct first-timesteps make each branch its own
    normalization group, so the cohort weights sum > 1."""
    ft = [("p2035", "t0001"), ("p2035_low", "t0002"),
          ("p2040", "t0004"), ("p2040_low", "t0005")]
    sbtb = [("p2035", "a"), ("p2035_low", "b"),
            ("p2040", "c"), ("p2040_low", "d")]
    v = check_recourse_npv_preconditions(
        tmp_path, provider=_mk_provider(
            pb=_PB, piu=_PIU, sbtb=sbtb, bw=_BW, ft=ft,
            pyd=_PYD, pyr=_PYR))
    assert any("obligation (ii)" in m for m in v), v


def test_p3c_diverging_years_d(tmp_path) -> None:
    """P3c (§9.3): a fan member with a diverging ``p_years_d`` value →
    message (iii)."""
    pyd_bad = [("p2035", 0.0), ("p2035_low", 99.0),
               ("p2040", 1.0), ("p2040_low", 1.0)]
    v = check_recourse_npv_preconditions(
        tmp_path, provider=_mk_provider(
            pb=_PB, piu=_PIU, sbtb=_SBTB, bw=_BW, ft=_FT,
            pyd=pyd_bad, pyr=_PYR))
    assert any("obligation (iii)" in m for m in v), v


def test_p3d_missing_years_represented(tmp_path) -> None:
    """P3d (§9.3): a fan member missing a ``p_years_represented`` row →
    message (iii)."""
    pyr_missing = [("p2035", "0", 1.0),
                   ("p2040", "1", 1.0), ("p2040_low", "1", 1.0)]
    v = check_recourse_npv_preconditions(
        tmp_path, provider=_mk_provider(
            pb=_PB, piu=_PIU, sbtb=_SBTB, bw=_BW, ft=_FT,
            pyd=_PYD, pyr=pyr_missing))
    assert any("obligation (iii)" in m for m in v), v


# --- P4 — diagnostic regression pin of §6.4 (erratum E1 witness) -------


def test_p4_walker_year_source_blindness(stoch_wf) -> None:
    """P4 (§9.3, keep-diagnostic-tests): on the stoch workdir,
    ``_p_years_d_lf(reader, active, wf)`` (provider-LESS, exactly as
    the walker calls it) yields NO rows for the synthetic branch
    members ``p2035_low`` / ``p2040_low``, WHILE
    ``solve_data/p_years_d.csv`` carries them byte-equal to their
    anchors.  Together with W6's absence assert this is the erratum E1
    witness pair — the concrete numeric mechanism behind plan §8.3."""
    wf = stoch_wf
    reader = _reader(wf)
    # Provider-less call — the workdir-CSV arm is provider-gated, so
    # this is the SOURCE-fallback resolution the walker actually uses.
    pyd = _p_years_d_lf(reader, STOCH_SOLVE, wf).collect()
    walker_periods = {str(d) for d in pyd["d"].to_list()}
    assert "p2035_low" not in walker_periods
    assert "p2040_low" not in walker_periods

    # The emitted CSV carries the byte-copied fan-member year rows.
    csv = pl.read_csv(wf / "solve_data" / "p_years_d.csv",
                      infer_schema_length=0)
    val_col = "value" if "value" in csv.columns else csv.columns[-1]
    years = {str(p): float(v) for p, v in
             zip(csv["period"].to_list(), csv[val_col].to_list())}
    assert "p2035_low" in years and "p2040_low" in years
    assert years["p2035_low"] == years["p2035"]
    assert years["p2040_low"] == years["p2040"]
