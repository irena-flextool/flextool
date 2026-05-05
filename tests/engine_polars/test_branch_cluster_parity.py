"""Δ.8 — Cluster D (multi-branch / stochastic propagation) parity tests.

Per-fixture parity check: lazy port in
:mod:`flextool.engine_polars._derived_branch` vs. the canonical
preprocessed CSVs in:

* ``solve_data/pd_branch_weight.csv``
* ``solve_data/pdt_branch_weight.csv``
* ``solve_data/dt_non_anticipativity_set.csv``
* ``solve_data/period__branch.csv`` (the FULL ``(d, b)`` map)
* ``solve_data/period_in_use_set.csv``

The CSV is the parity oracle — any divergence between the lazy port
and the CSV surfaces as a per-fixture failure.

The test reuses the fixture-discovery pattern from
:mod:`test_existing_chain_cluster_parity` and the hand-cooked
:class:`InMemoryReader` pattern from earlier cluster tests.

Fields covered (per
``audit/native_data_path_design_derived_clusters.md``):

* ``pd_branch_weight`` — per-period branch probability.
* ``pdt_branch_weight`` — per-(d, t) branch probability (sparse, prior
  to the dt-dense coalesce in :mod:`._derived_branch.apply_branch_cluster`).
* ``dt_non_anticipativity`` — realised-dispatch ∪ fix-storage timesteps.
* ``period_branch_full`` — anchor → sibling map.
* ``period_in_use_set`` — periods active in the active solve.

R-O6 invariant: see :mod:`test_stochastic_parity` for the
realized-only-invest verification (branches do NOT enter
``invest_periods``).  This module verifies the *operational* side of
cluster D — the dispatch-class probability weights and the
non-anticipativity gate.
"""
from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from flextool.engine_polars._derived_branch import (
    apply_branch_cluster,
    dt_non_anticipativity_df,
    dt_non_anticipativity_lf,
    pd_branch_weight_lf,
    pd_branch_weight_param,
    pdt_branch_weight_lf,
    pdt_branch_weight_param,
    period_branch_full_df,
    period_branch_pairs_lf,
    period_in_use_set_df,
    period_in_use_set_lf,
)
from flextool.engine_polars._input_source import _read_csv_file


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# Fixture discovery
# ---------------------------------------------------------------------------


_DIRNAME_TO_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_2day_stochastic_dispatch_no_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _discover_fixtures() -> list[tuple[str, Path, str]]:
    """Return ``[(work_dirname, solve_data_path, scenario_name), ...]``
    covering every fixture with at least one ``solve_data`` directory
    that has a ``pdt_branch_weight.csv`` (the cluster D oracle).
    """
    out: list[tuple[str, Path, str]] = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("work_"):
            continue
        sqlite = d / "tests.sqlite"
        if not sqlite.exists():
            continue
        # Find solve_data dirs with cluster D oracle present.
        sd_candidates = [
            sd for sd in d.glob("solve_data*")
            if sd.is_dir() and (sd / "pdt_branch_weight.csv").exists()
        ]
        if not sd_candidates:
            continue
        # Pick scenario.
        if d.name in _DIRNAME_TO_SCENARIO_OVERRIDES:
            target = _DIRNAME_TO_SCENARIO_OVERRIDES[d.name]
            try:
                with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                    found = any(
                        s.name == target for s in db.query(db.scenario_sq).all()
                    )
            except Exception:
                found = False
            if found:
                for sd in sd_candidates:
                    out.append((d.name, sd, target))
                continue
        scen_target = d.name.removeprefix("work_")
        try:
            with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                scenarios = sorted(
                    s.name for s in db.query(db.scenario_sq).all()
                )
        except Exception:
            continue
        candidates = [scen_target]
        import re
        candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", scen_target))
        candidates.append(re.sub(r"(\d+)_([a-z])", r"\1\2", scen_target))
        if scen_target.endswith("_full_storage"):
            base = scen_target[: -len("_full_storage")]
            candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base))
            candidates.append(base)
        if scen_target.endswith("_no_storage"):
            base = scen_target[: -len("_no_storage")]
            candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base))
            candidates.append(base)
        chosen: str | None = None
        for cand in candidates:
            if cand in scenarios:
                chosen = cand
                break
        if chosen is not None:
            for sd in sd_candidates:
                out.append((d.name, sd, chosen))
        elif scenarios:
            for sd in sd_candidates:
                out.append((d.name, sd, scenarios[0]))
    return out


PARITY_CASES = _discover_fixtures()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _workdir_of(sd: Path, *, tmp_root: Path | None = None) -> Path:
    """Return a workdir whose ``solve_data`` directory is *sd*.

    For canonical ``solve_data`` dirs, returns ``sd.parent`` directly.
    For non-canonical (``solve_data_*``) dirs, builds a tmp workdir
    with ``solve_data`` as a symlink to *sd*.
    """
    if sd.name == "solve_data":
        return sd.parent
    if tmp_root is None:
        return sd.parent  # caller will skip
    wd = tmp_root / "wd"
    wd.mkdir(parents=True, exist_ok=True)
    link = wd / "solve_data"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(sd)
    return wd


def _csv_pd_branch_weight(sd: Path) -> pl.DataFrame:
    """Read the canonical ``pd_branch_weight.csv``."""
    p = sd / "pd_branch_weight.csv"
    df = _read_csv_file(p)
    return (df.rename({"period": "d"})
              .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
              .select("d", "value")
              .sort("d"))


def _csv_pdt_branch_weight(sd: Path) -> pl.DataFrame:
    """Read the canonical ``pdt_branch_weight.csv``."""
    p = sd / "pdt_branch_weight.csv"
    df = _read_csv_file(p)
    return (df.rename({"period": "d", "time": "t"})
              .with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
              .select("d", "t", "value")
              .sort("d", "t"))


def _csv_dt_non_anticipativity(sd: Path) -> pl.DataFrame:
    """Read the canonical ``dt_non_anticipativity_set.csv``."""
    p = sd / "dt_non_anticipativity_set.csv"
    if not p.exists():
        return pl.DataFrame(schema={"d": pl.Utf8, "t": pl.Utf8})
    df = _read_csv_file(p)
    if df.height == 0:
        return pl.DataFrame(schema={"d": pl.Utf8, "t": pl.Utf8})
    return (df.rename({"period": "d", "time": "t"})
              .select("d", "t").unique().sort("d", "t"))


def _csv_period_branch_full(sd: Path) -> pl.DataFrame:
    p = sd / "period__branch.csv"
    df = _read_csv_file(p)
    return (df.rename({"period": "d", "branch": "b"})
              .select("d", "b").unique().sort("d", "b"))


def _csv_period_in_use_set(sd: Path) -> pl.DataFrame:
    p = sd / "period_in_use_set.csv"
    df = _read_csv_file(p)
    col = df.columns[0]
    return df.select(pl.col(col).alias("d")).unique().sort("d")


def _frames_equal(a: pl.DataFrame, b: pl.DataFrame, *,
                   tol: float = 1e-9) -> tuple[bool, str]:
    """Compare two frames after sorting by every column (numeric tol on
    'value'-named float column).  Returns (equal, diagnostic).
    """
    if set(a.columns) != set(b.columns):
        return False, f"columns differ: {a.columns} vs {b.columns}"
    cols = sorted(a.columns)
    a = a.select(*cols).sort(*cols)
    b = b.select(*cols).sort(*cols)
    if a.height != b.height:
        return False, f"heights differ: {a.height} vs {b.height}"
    for c in cols:
        if a[c].dtype.is_numeric() and b[c].dtype.is_numeric():
            la = a[c].cast(pl.Float64, strict=False)
            lb = b[c].cast(pl.Float64, strict=False)
            diff = (la - lb).abs().max()
            if diff is not None and diff > tol:
                return False, f"col {c!r} max-diff = {diff} > {tol}"
        else:
            la = a[c].cast(pl.Utf8, strict=False)
            lb = b[c].cast(pl.Utf8, strict=False)
            if not la.equals(lb):
                # Find first diff
                for i, (xa, xb) in enumerate(zip(la.to_list(), lb.to_list())):
                    if xa != xb:
                        return False, f"col {c!r} differ at row {i}: {xa!r} vs {xb!r}"
                return False, f"col {c!r} differ"
    return True, "ok"


# ---------------------------------------------------------------------------
# Per-fixture parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "work_name,sd,scenario",
    PARITY_CASES,
    ids=[f"{w}::{sd.name}@{s}" for w, sd, s in PARITY_CASES],
)
def test_pd_branch_weight_lazy_vs_csv(
    work_name: str, sd: Path, scenario: str, tmp_path: Path,
) -> None:
    """``pd_branch_weight_lf`` matches the canonical
    ``solve_data/pd_branch_weight.csv``."""
    wd = _workdir_of(sd, tmp_root=tmp_path)
    expected = _csv_pd_branch_weight(sd)
    actual = pd_branch_weight_lf(wd).collect().sort("d")
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work_name}::{sd.name}@{scenario} pd_branch_weight differs: {diag}\n"
        f"expected (head):\n{expected.head(10)}\n"
        f"actual   (head):\n{actual.head(10)}"
    )


@pytest.mark.parametrize(
    "work_name,sd,scenario",
    PARITY_CASES,
    ids=[f"{w}::{sd.name}@{s}" for w, sd, s in PARITY_CASES],
)
def test_pdt_branch_weight_lazy_vs_csv(
    work_name: str, sd: Path, scenario: str, tmp_path: Path,
) -> None:
    """``pdt_branch_weight_lf`` matches the canonical
    ``solve_data/pdt_branch_weight.csv``."""
    wd = _workdir_of(sd, tmp_root=tmp_path)
    expected = _csv_pdt_branch_weight(sd)
    # Build a minimal dt frame that matches steps_in_use.csv (the
    # flextool reference's implicit dt domain).
    siu = _read_csv_file(sd / "steps_in_use.csv")
    dt = (siu.rename({"period": "d", "step": "t"})
              .select("d", "t").unique())
    actual = pdt_branch_weight_lf(wd, dt=dt).collect().sort("d", "t")
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work_name}::{sd.name}@{scenario} pdt_branch_weight differs: {diag}\n"
        f"expected (head):\n{expected.head(10)}\n"
        f"actual   (head):\n{actual.head(10)}"
    )


@pytest.mark.parametrize(
    "work_name,sd,scenario",
    PARITY_CASES,
    ids=[f"{w}::{sd.name}@{s}" for w, sd, s in PARITY_CASES],
)
def test_dt_non_anticipativity_lazy_vs_csv(
    work_name: str, sd: Path, scenario: str, tmp_path: Path,
) -> None:
    """``dt_non_anticipativity_lf`` matches the canonical
    ``solve_data/dt_non_anticipativity_set.csv``."""
    wd = _workdir_of(sd, tmp_root=tmp_path)
    expected = _csv_dt_non_anticipativity(sd)
    actual = dt_non_anticipativity_lf(wd).collect().sort("d", "t")
    if expected.height == 0 and actual.height == 0:
        return
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work_name}::{sd.name}@{scenario} dt_non_anticipativity differs: {diag}\n"
        f"expected (head):\n{expected.head(10)}\n"
        f"actual   (head):\n{actual.head(10)}"
    )


@pytest.mark.parametrize(
    "work_name,sd,scenario",
    PARITY_CASES,
    ids=[f"{w}::{sd.name}@{s}" for w, sd, s in PARITY_CASES],
)
def test_period_branch_full_lazy_vs_csv(
    work_name: str, sd: Path, scenario: str, tmp_path: Path,
) -> None:
    """``period_branch_pairs_lf`` matches the canonical
    ``solve_data/period__branch.csv``."""
    wd = _workdir_of(sd, tmp_root=tmp_path)
    expected = _csv_period_branch_full(sd)
    actual = period_branch_pairs_lf(wd).collect().sort("d", "b")
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work_name}::{sd.name}@{scenario} period_branch_full differs: {diag}"
    )


@pytest.mark.parametrize(
    "work_name,sd,scenario",
    PARITY_CASES,
    ids=[f"{w}::{sd.name}@{s}" for w, sd, s in PARITY_CASES],
)
def test_period_in_use_set_lazy_vs_csv(
    work_name: str, sd: Path, scenario: str, tmp_path: Path,
) -> None:
    """``period_in_use_set_lf`` matches the canonical
    ``solve_data/period_in_use_set.csv``."""
    wd = _workdir_of(sd, tmp_root=tmp_path)
    expected = _csv_period_in_use_set(sd)
    actual = period_in_use_set_lf(wd).collect().sort("d")
    eq, diag = _frames_equal(expected, actual)
    assert eq, (
        f"{work_name}::{sd.name}@{scenario} period_in_use_set differs: {diag}"
    )


# ---------------------------------------------------------------------------
# Hand-cooked 2-period 3-branch test
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + "\n".join(rows) + ("\n" if rows else ""))


def test_handcooked_3branch_weights(tmp_path: Path) -> None:
    """3-branch scenario with weights [0.4, 0.4, 0.2], realized=branch_1.

    The hand-cooked fixture mirrors ``test_stochastic_parity``'s
    3-branch scenario but exercises *only* the operational side of
    cluster D — the per-(d, t) probability normalisation.

    Setup:

    * 2 periods (p2020, p2025).  p2020 branches at first step into
      branch_1 (realized, 0.4), branch_2 (0.4), branch_3 (0.2).
    * period__branch lists every (anchor, sibling) including the
      self-branch (p2020, p2020) per flextool's convention.
    * solve_branch_weight assigns the input weights.

    Expected ``pd_branch_weight``:

      pd[p2020]            = 0.4 / (0.4 + 0.4 + 0.4 + 0.2) = 0.4 / 1.4
      pd[p2020_branch_1]   = 0.4 / 1.4
      pd[p2020_branch_2]   = 0.4 / 1.4
      pd[p2020_branch_3]   = 0.2 / 1.4
    """
    wd = tmp_path / "work_test"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)

    # period__branch.csv — anchor → sibling map.
    _write_csv(
        sd / "period__branch.csv",
        "period,branch",
        [
            "p2020,p2020",
            "p2020,p2020_branch_1",
            "p2020,p2020_branch_2",
            "p2020,p2020_branch_3",
        ],
    )

    # solve_branch_weight.csv — per-branch input weight.
    _write_csv(
        sd / "solve_branch_weight.csv",
        "branch,p_branch_weight_input",
        [
            "p2020,0.4",
            "p2020_branch_1,0.4",
            "p2020_branch_2,0.4",
            "p2020_branch_3,0.2",
        ],
    )

    # first_timesteps.csv — every "period" gets the same first step
    # (which is what triggers the sibling grouping).
    _write_csv(
        sd / "first_timesteps.csv",
        "period,step",
        [
            "p2020,t0001",
            "p2020_branch_1,t0001",
            "p2020_branch_2,t0001",
            "p2020_branch_3,t0001",
        ],
    )

    # period_in_use_set.csv — output domain.
    _write_csv(
        sd / "period_in_use_set.csv",
        "period",
        ["p2020", "p2020_branch_1", "p2020_branch_2", "p2020_branch_3"],
    )

    # steps_in_use.csv — single timestep × every branch period.
    _write_csv(
        sd / "steps_in_use.csv",
        "period,step,step_duration",
        [
            "p2020,t0001,1.0",
            "p2020_branch_1,t0001,1.0",
            "p2020_branch_2,t0001,1.0",
            "p2020_branch_3,t0001,1.0",
        ],
    )

    pd_bw = pd_branch_weight_lf(wd).collect().sort("d")
    expected_pd = pl.DataFrame({
        "d": ["p2020", "p2020_branch_1", "p2020_branch_2", "p2020_branch_3"],
        "value": [0.4 / 1.4, 0.4 / 1.4, 0.4 / 1.4, 0.2 / 1.4],
    }).sort("d")
    eq, diag = _frames_equal(expected_pd, pd_bw, tol=1e-12)
    assert eq, f"pd_branch_weight diverged: {diag}\nactual:\n{pd_bw}"

    # pdt — every (d, t) gets the same w(d) / 1.4 because every branch
    # shares the same single timestep.
    pdt_bw = pdt_branch_weight_lf(wd).collect().sort("d", "t")
    assert pdt_bw.height == 4
    val_for = {row["d"]: row["value"] for row in pdt_bw.iter_rows(named=True)}
    assert abs(val_for["p2020"] - 0.4 / 1.4) < 1e-12
    assert abs(val_for["p2020_branch_1"] - 0.4 / 1.4) < 1e-12
    assert abs(val_for["p2020_branch_2"] - 0.4 / 1.4) < 1e-12
    assert abs(val_for["p2020_branch_3"] - 0.2 / 1.4) < 1e-12


def test_handcooked_deterministic_default(tmp_path: Path) -> None:
    """Without ``period__branch.csv``, all weights default to 1.0."""
    wd = tmp_path / "work_det"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)

    _write_csv(
        sd / "period_in_use_set.csv",
        "period",
        ["p2020", "p2025"],
    )
    _write_csv(
        sd / "steps_in_use.csv",
        "period,step,step_duration",
        ["p2020,t0001,1.0", "p2025,t0001,1.0"],
    )

    pd_bw = pd_branch_weight_lf(wd).collect().sort("d")
    assert pd_bw["value"].to_list() == [1.0, 1.0]
    pdt_bw = pdt_branch_weight_lf(wd).collect().sort("d", "t")
    assert pdt_bw["value"].to_list() == [1.0, 1.0]


def test_handcooked_dt_non_anticipativity(tmp_path: Path) -> None:
    """``dt_non_anticipativity = realized_dispatch ∪ fix_storage_timesteps``."""
    wd = tmp_path / "work_dtna"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)

    _write_csv(
        sd / "realized_dispatch.csv",
        "period,step",
        ["p2020,t0001", "p2020,t0002"],
    )
    _write_csv(
        sd / "fix_storage_timesteps.csv",
        "period,step",
        ["p2020,t0002", "p2020,t0003"],  # overlapping t0002
    )

    out = dt_non_anticipativity_lf(wd).collect().sort("d", "t")
    expected = pl.DataFrame({
        "d": ["p2020", "p2020", "p2020"],
        "t": ["t0001", "t0002", "t0003"],
    })
    eq, diag = _frames_equal(expected, out)
    assert eq, f"dt_non_anticipativity diverged: {diag}\nactual:\n{out}"


def test_handcooked_dt_non_anticipativity_empty(tmp_path: Path) -> None:
    """Empty workdir yields empty (d, t) frame — non-anticipativity off."""
    wd = tmp_path / "work_empty"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)

    out = dt_non_anticipativity_lf(wd).collect()
    assert out.height == 0
    assert sorted(out.columns) == ["d", "t"]

    # Public API returns None for empty.
    assert dt_non_anticipativity_df(wd) is None
    assert period_branch_full_df(wd) is None


def test_handcooked_period_in_use_falls_back_to_source(tmp_path: Path) -> None:
    """Without ``period_in_use_set.csv``, fall back to source-derived
    realized + invest periods."""
    wd = tmp_path / "work_fallback"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)
    # No period_in_use_set.csv on disk.

    # Build a minimal InputSource that exposes
    # ``solve.realized_periods`` for active solve "S".
    class _FakeSource:
        def parameter(self, ec: str, par: str) -> pl.DataFrame:
            if ec == "solve" and par == "realized_periods":
                return pl.DataFrame({
                    "name": ["S", "S"],
                    "value": ["p2020", "p2025"],
                })
            if ec == "solve" and par == "invest_periods":
                return pl.DataFrame({
                    "name": ["S"],
                    "value": ["p2030"],
                })
            raise KeyError(par)

    src = _FakeSource()
    out = period_in_use_set_lf(wd, src, "S").collect().sort("d")
    assert sorted(out["d"].to_list()) == ["p2020", "p2025", "p2030"]


# ---------------------------------------------------------------------------
# Multi-period stochastic stress test
# ---------------------------------------------------------------------------


def test_handcooked_multiperiod_stochastic(tmp_path: Path) -> None:
    """Multi-period stochastic stress: 2 periods × 3 branches each.

    Period p1 has all 3 branches; period p2 carries the same 3
    branches forward (continuation-after-branching pattern).  Every
    branch period has 2 timesteps.

    Expected ``pdt`` denominator at every (d, t):
      sum w[b] for b ∈ {p1, p1_a, p1_b, p2, p2_a, p2_b} restricted to
      same-parent + same-t criterion.

    Per the algorithm, the denominator at (d, t) iterates ``(d2, b) ∈
    period__branch`` and counts those where ``(b, t) ∈ dt`` AND
    ``(d2, d) ∈ period__branch``.

    For (d=p1, t=t01): branches with t01 are {p1, p1_a, p1_b}.  Their
    parents ARE p1 → all three count.  Denom = 1.0 + 0.4 + 0.6 = 2.0.
    Value = w[p1] / 2.0 = 1.0 / 2.0 = 0.5.

    For (d=p1_a, t=t01): same denominator (siblings sharing parent p1
    that are present at t01).  Value = w[p1_a] / 2.0 = 0.4 / 2.0 = 0.2.
    """
    wd = tmp_path / "work_multi"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)

    # period__branch: each period has 3 self+branch entries.
    _write_csv(
        sd / "period__branch.csv",
        "period,branch",
        [
            "p1,p1",
            "p1,p1_a",
            "p1,p1_b",
            "p2,p2",
            "p2,p2_a",
            "p2,p2_b",
        ],
    )

    # solve_branch_weight: anchor=1.0, branch_a=0.4, branch_b=0.6.
    _write_csv(
        sd / "solve_branch_weight.csv",
        "branch,p_branch_weight_input",
        [
            "p1,1.0",
            "p1_a,0.4",
            "p1_b,0.6",
            "p2,1.0",
            "p2_a,0.4",
            "p2_b,0.6",
        ],
    )

    _write_csv(
        sd / "first_timesteps.csv",
        "period,step",
        [
            "p1,t01",
            "p1_a,t01",
            "p1_b,t01",
            "p2,t11",
            "p2_a,t11",
            "p2_b,t11",
        ],
    )

    _write_csv(
        sd / "period_in_use_set.csv",
        "period",
        ["p1", "p1_a", "p1_b", "p2", "p2_a", "p2_b"],
    )

    # steps_in_use: 2 timesteps per branch period.
    rows: list[str] = []
    for p in ["p1", "p1_a", "p1_b"]:
        for t in ["t01", "t02"]:
            rows.append(f"{p},{t},1.0")
    for p in ["p2", "p2_a", "p2_b"]:
        for t in ["t11", "t12"]:
            rows.append(f"{p},{t},1.0")
    _write_csv(sd / "steps_in_use.csv", "period,step,step_duration", rows)

    # Validate pd_branch_weight.
    pd_bw = pd_branch_weight_lf(wd).collect().sort("d")
    val_for = {r["d"]: r["value"] for r in pd_bw.iter_rows(named=True)}
    # p1 cluster denom = 1.0 + 0.4 + 0.6 = 2.0
    assert abs(val_for["p1"] - 1.0 / 2.0) < 1e-12
    assert abs(val_for["p1_a"] - 0.4 / 2.0) < 1e-12
    assert abs(val_for["p1_b"] - 0.6 / 2.0) < 1e-12
    # p2 cluster denom = same.
    assert abs(val_for["p2"] - 1.0 / 2.0) < 1e-12
    assert abs(val_for["p2_a"] - 0.4 / 2.0) < 1e-12
    assert abs(val_for["p2_b"] - 0.6 / 2.0) < 1e-12

    # Validate pdt_branch_weight at one (d, t).
    pdt_bw = pdt_branch_weight_lf(wd).collect().sort("d", "t")
    pdt_for = {(r["d"], r["t"]): r["value"]
               for r in pdt_bw.iter_rows(named=True)}
    # At (p1, t01): siblings present at t01 are {p1, p1_a, p1_b};
    # parents = p1 in all three → denom = 2.0.
    assert abs(pdt_for[("p1", "t01")] - 1.0 / 2.0) < 1e-12
    # At (p1_a, t02): siblings present at t02 are {p1, p1_a, p1_b}.
    assert abs(pdt_for[("p1_a", "t02")] - 0.4 / 2.0) < 1e-12

    # Cross-cluster: (p1, t11) should NOT exist in pdt because t11
    # only appears in p2-cluster entries (different parent).  Verify
    # by checking the denominator-zero fallback to 1.0 IS applied
    # (dense semantics) since (p1, t11) is NOT a row in
    # steps_in_use.csv at all.
    assert ("p1", "t11") not in pdt_for


# ---------------------------------------------------------------------------
# apply_branch_cluster integration — verify it mutates flex_data
# ---------------------------------------------------------------------------


def test_apply_branch_cluster_mutates_flex_data(tmp_path: Path) -> None:
    """Integration: ``apply_branch_cluster`` populates the relevant
    ``flex_data`` fields via setattr.
    """
    wd = tmp_path / "work_int"
    sd = wd / "solve_data"
    sd.mkdir(parents=True)
    _write_csv(
        sd / "period__branch.csv",
        "period,branch",
        ["p1,p1", "p1,p1_a"],
    )
    _write_csv(
        sd / "solve_branch_weight.csv",
        "branch,p_branch_weight_input",
        ["p1,1.0", "p1_a,0.5"],
    )
    _write_csv(
        sd / "first_timesteps.csv",
        "period,step",
        ["p1,t01", "p1_a,t01"],
    )
    _write_csv(
        sd / "period_in_use_set.csv",
        "period",
        ["p1", "p1_a"],
    )
    _write_csv(
        sd / "steps_in_use.csv",
        "period,step,step_duration",
        ["p1,t01,1.0", "p1_a,t01,1.0"],
    )

    class _Bag:
        pass
    flex_data = _Bag()
    flex_data.dt = pl.DataFrame({"d": ["p1", "p1_a"], "t": ["t01", "t01"]})
    apply_branch_cluster(flex_data, source=None, workdir=wd, active_solve=None)

    assert getattr(flex_data, "period_branch_full") is not None
    assert getattr(flex_data, "period_in_use_set") is not None
    assert getattr(flex_data, "pd_branch_weight") is not None
    assert getattr(flex_data, "pdt_branch_weight") is not None


# ---------------------------------------------------------------------------
# R-O6: branches stay realised-only for invest
# ---------------------------------------------------------------------------


def test_branch_periods_not_in_invest_periods_2day_stochastic() -> None:
    """R-O6: in the canonical 2_day_stochastic_dispatch fixture, the
    branch periods (period1_upper / period1_lower / period1_mid) do
    NOT appear in ``invest_periods``.  This is the R-O6 invariant
    enforced by the cluster D port.

    The actual flexpy enforcement happens in
    :class:`StochasticSolver` (Γ.8.C) — branches go into
    active_time_lists, NOT into invest_periods.  This test verifies
    that the cluster D helpers don't accidentally re-introduce
    branches into the invest set via a derivation.
    """
    sqlite = (DATA / "work_2day_stochastic_dispatch_full_storage" /
              "tests.sqlite")
    if not sqlite.exists():
        pytest.skip("stochastic fixture missing")

    from spinedb_api.filters.scenario_filter import (
        apply_scenario_filter_to_subqueries,
    )
    from flextool.engine_polars._solve_config import SolveConfig
    log = logging.getLogger("test_R-O6")
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        apply_scenario_filter_to_subqueries(db, "2_day_stochastic_dispatch")
        s = SolveConfig.load_from_db(db, log)

    # invest_periods is solve → list[(p_from, p_in)]
    for solve, periods in s.invest_periods.items():
        for p_from, p_in in periods:
            # Branch periods carry an extra suffix like ``_upper`` or
            # ``_lower``.  The canonical fixture branches all start
            # with ``period1`` and end with a branch tag; the
            # un-branched name is just ``period1``.
            assert p_in == p_from or "_" not in p_in, (
                f"R-O6 violation: invest_periods[{solve}] contains "
                f"branched period {p_in!r}"
            )
