"""Slice D — Phase D0 recourse-active plumbing (inert).

Design: ``specs/sliceD_recourse_invest_design.md`` §2 / §18 D0.
Covers the flag-emit → read round-trip and the two-part activation
predicate (flag CSV == 'recourse' AND branch invest axis present).  All
solver-free: the CSV read and the capability conjunct are pure.
"""
from __future__ import annotations

import polars as pl

from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from flextool.engine.preprocess.params._derived import (
    _has_branch_invest_axis,
    _recourse_invest_active,
)
from flextool.engine.preprocess.emit._solve_writers import (
    emit_stochastic_invest_method,
)
from tests._support.provider import provider_from_csv_dirs


CSV = "solve_data/stochastic_invest_method.csv"


def _write(workdir, text: str) -> None:
    p = workdir / "solve_data"
    p.mkdir(parents=True, exist_ok=True)
    (p / "stochastic_invest_method.csv").write_text(text)


def _seeded_provider(workdir):
    """Provider holding the CSVs of ``workdir/solve_data`` — the cascade
    reads the flag exclusively from the sub-solve's Provider (never disk),
    where the run-model emit puts it (see
    ``test_emit_read_roundtrip_provider``).
    """
    sd = workdir / "solve_data"
    return provider_from_csv_dirs(solve_data_dir=sd if sd.is_dir() else None)


# ---------------------------------------------------------------------------
# Reader — provider-only read + absent-default
# ---------------------------------------------------------------------------


def test_recourse_active_true(tmp_path):
    _write(tmp_path, "method\nrecourse\n")
    assert _recourse_invest_active(
        tmp_path, provider=_seeded_provider(tmp_path)) is True


def test_recourse_active_none_flag_false(tmp_path):
    _write(tmp_path, "method\nnone\n")
    assert _recourse_invest_active(
        tmp_path, provider=_seeded_provider(tmp_path)) is False


def test_recourse_active_absent_csv_false(tmp_path):
    # No CSV emitted — the byte-parity-safe default (Provider carries no key).
    assert _recourse_invest_active(
        tmp_path, provider=_seeded_provider(tmp_path)) is False


def test_recourse_active_reads_provider_not_disk(tmp_path):
    # The flag CSV on disk, an empty Provider → no key → default False (never touches disk).
    _write(tmp_path, "method\nrecourse\n")
    assert _recourse_invest_active(tmp_path, provider=FlexDataProvider()) is False


def test_recourse_active_workdir_none_false():
    assert _recourse_invest_active(None, provider=FlexDataProvider()) is False


# ---------------------------------------------------------------------------
# Emit round-trip through a real Provider (mirrors the run-model emit)
# ---------------------------------------------------------------------------


def test_emit_read_roundtrip_provider(tmp_path):
    prov = FlexDataProvider()
    emit_stochastic_invest_method(
        "recourse", str(tmp_path / CSV), provider=prov)
    assert _recourse_invest_active(tmp_path, provider=prov) is True

    prov2 = FlexDataProvider()
    emit_stochastic_invest_method(
        "none", str(tmp_path / CSV), provider=prov2)
    assert _recourse_invest_active(tmp_path, provider=prov2) is False


# ---------------------------------------------------------------------------
# Capability conjunct — branch invest axis present
# ---------------------------------------------------------------------------


class _Data:
    def __init__(self, pbf, piu):
        self.period_branch_full = pbf
        self.period_in_use_set = piu


def _pbf(pairs):
    d_cats = sorted({p[0] for p in pairs})
    b_cats = sorted({p[1] for p in pairs if p[1] is not None})
    return pl.DataFrame(
        {"d": [p[0] for p in pairs], "b": [p[1] for p in pairs]}
    ).with_columns(
        pl.col("d").cast(pl.Enum(d_cats)),
        pl.col("b").cast(pl.Enum(b_cats)),
    )


def _piu(periods):
    return pl.DataFrame({"d": list(periods)}).with_columns(
        pl.col("d").cast(pl.Enum(sorted(set(periods))))
    )


def test_has_branch_invest_axis_true():
    pbf = _pbf([("p2035", "p2035"), ("p2035", "p2035_low")])
    piu = _piu(["p2035", "p2035_low"])
    assert _has_branch_invest_axis(_Data(pbf, piu)) is True


def test_has_branch_invest_axis_deterministic_false():
    pbf = _pbf([("p2035", None), ("p2040", None)])
    piu = _piu(["p2035", "p2040"])
    assert _has_branch_invest_axis(_Data(pbf, piu)) is False


def test_has_branch_invest_axis_none_false():
    assert _has_branch_invest_axis(None) is False
