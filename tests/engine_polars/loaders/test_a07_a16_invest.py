"""Surface A.7 (invest/divest capacity expansion) and A.16 (cumulative
invest / group-build constraints) loader tests.

A.7 covers ``flextool.engine_polars.input._load_invest`` — the helper
that reads ``solve_data/ed_invest.csv`` etc. and assembles the dict of
invest-set frames + per-(e, d) cost params consumed by the override
chain and the variable emitter.

A.16 covers ``flextool.engine_polars.input._load_cumulative_invest``
(the cascade-of-15+ optional CSVs that drive cumulative-capacity and
group-invest constraints) and a single direct call into
``flextool.engine_polars._cumulative_invest._emit_group_invest_cumulative``
to pin the all-three-required-or-no-emit guard.

Both helper families are pure: they take ``Path`` arguments + a
db_reader stub and return a dict.  Hand-built minimal solve_data
folders isolate one transformation per test (mirroring the A.6
storage-handoff test pattern in ``test_a06_a20_storage_handoff.py``).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine_polars.input import (
    _load_invest,
    _load_cumulative_invest,
)
from flextool.engine_polars._cumulative_invest import (
    _emit_group_invest_cumulative,
    has_feature,
)


# --- helpers --------------------------------------------------------

def _empty_dt() -> pl.DataFrame:
    return pl.DataFrame({"d": [], "t": []}, schema={"d": pl.Utf8, "t": pl.Utf8})


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _make_invest_workdir(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (inp, sd, dt) ready for ``_load_invest`` calls."""
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    return inp, sd, _empty_dt()


# ====================================================================
# A.7 — _load_invest
# ====================================================================

def test_divest_only_no_invest_bypasses_blank_shortcut(tmp_path: Path):
    """Covers A7-ed_divest_only_no_invest.

    Empty ``ed_invest.csv`` + populated ``ed_divest.csv`` (1 row).
    The blank-shortcut gate is ``ed_inv.height==0 AND ed_div.height==0``;
    populating divest alone must keep the helper on the live path so
    ``ed_divest_set`` is populated and ``ed_invest_set`` stays None.
    """
    inp, sd, dt = _make_invest_workdir(tmp_path)
    _write(sd / "ed_invest.csv", "entity,period\n")
    _write(sd / "ed_divest.csv", "entity,period\ne1,p1\n")

    out = _load_invest(sd, dt, inp, None, db_reader=None)
    # Hand-calc: ed_inv.height=0, ed_div.height=1 -> NOT blank;
    # ed_invest_set _hnz(0)=None, ed_divest_set _hnz(1)=frame{(e1,p1)}.
    assert out["ed_invest_set"] is None
    assert out["ed_divest_set"] is not None
    assert out["ed_divest_set"].height == 1
    assert out["ed_divest_set"].row(0) == ("e1", "p1")


def test_period_set_independent_from_max_period_param(tmp_path: Path):
    """Covers A7-period_set_independent_from_max_period_param.

    Populate ``ed_invest_period.csv`` (set membership) WITHOUT
    ``ed_invest_max_period.csv``.  The set-frame seed and the cap-param
    read are decoupled — set populated, cap None — so the constraint
    can index by set membership and let the cap default via the
    override chain.
    """
    inp, sd, dt = _make_invest_workdir(tmp_path)
    # Need ed_invest non-empty so the helper doesn't take the blank
    # shortcut (which would null out every key including period_set).
    _write(sd / "ed_invest.csv", "entity,period\ne1,p1\n")
    _write(sd / "ed_invest_period.csv", "entity,period\ne1,p1\n")
    # Intentionally NO ed_invest_max_period.csv

    out = _load_invest(sd, dt, inp, None, db_reader=None)
    # Hand-calc: period_set populated by _seed_period_set -> height=1;
    # max_period CSV absent -> _read_e_d returns None.
    assert out["ed_invest_period_set"] is not None
    assert out["ed_invest_period_set"].height == 1
    assert out["ed_invest_max_period"] is None


def test_skip_set_seeds_when_synthetic_solve(tmp_path: Path):
    """Covers A7-skip_set_seeds_when_synthetic_solve.

    Build a workdir whose ``solve_current.csv`` names a synthetic
    ``<base>_<anchor>`` solve (``mysolve_p2020``) NOT in Spine, but
    Spine carries the base ``mysolve`` row and ``solve.invest_periods``
    lists ``p2020``.  ``_load_invest`` must:
      * skip the 8 set-seed reads (set fields all None even though the
        CSVs ARE present on disk),
      * still read the cost-param seeds (``ed_lifetime_fixed_cost``).
    """
    inp, sd, dt = _make_invest_workdir(tmp_path)
    _write(sd / "solve_current.csv", "solve\nmysolve_p2020\n")
    # Set CSVs that should be IGNORED on the synthetic-skip path.
    _write(sd / "ed_invest.csv", "entity,period\ne1,p1\n")
    _write(sd / "ed_divest.csv", "entity,period\ne1,p1\n")
    # Cost CSV that MUST still be read.
    _write(sd / "ed_lifetime_fixed_cost.csv",
           "entity,period,value\ne1,p1,100.0\n")

    class _Reader:
        def entities(self, ec):
            if ec == "solve":
                return pl.DataFrame({"name": ["mysolve"]})
            raise KeyError(ec)
        def parameter(self, ec, par):
            if ec == "solve" and par == "invest_periods":
                # Array-shaped: name + value (period).  _solve_periods
                # filter sees no row for active 'mysolve_p2020' -> falls
                # to synthetic resolver -> base='mysolve', anchor='p2020'
                # -> Array path returns [anchor] when present in base.
                return pl.DataFrame({"name": ["mysolve"], "value": ["p2020"]})
            raise KeyError(par)

    out = _load_invest(sd, dt, inp, None, db_reader=_Reader())
    # Hand-calc: synthetic gate fires -> skip_set_seeds=True;
    # invest_periods=['p2020'] -> NOT blank -> set fields stay None,
    # cost-param seed loaded -> ed_lifetime_fixed_cost Param at (e1,p1)=100.
    for set_field in ("ed_invest_set", "ed_divest_set",
                      "pd_invest_set", "pd_divest_set",
                      "nd_invest_set", "nd_divest_set", "edd_invest_set"):
        assert out[set_field] is None, (
            f"{set_field} should be None on synthetic-skip path")
    assert out["ed_lifetime_fixed_cost"] is not None
    fc = out["ed_lifetime_fixed_cost"].frame
    assert fc.height == 1
    assert fc["value"][0] == pytest.approx(100.0, rel=1e-7)


# ====================================================================
# A.16 — _load_cumulative_invest + _emit_group_invest_cumulative
# ====================================================================

def test_group_entity_solve_data_precedence_and_input_fallback(tmp_path: Path):
    """Covers A16-group_entity_solve_data_precedence + A16-group_entity_input_fallback.

    Two passes share the same minimal cumulative-set CSV:
      (1) BOTH ``solve_data/group_entity.csv`` (g1,e1) AND
          ``input/group__entity.csv`` (g1,e2) — solve_data wins, e1 only.
      (2) DROP the solve_data file — input fallback fires, e2 only.
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    _write(sd / "group_entity.csv",   "group,entity\ng1,e1\n")
    _write(inp / "group__entity.csv", "group,entity\ng1,e2\n")

    out1 = _load_cumulative_invest(inp, sd, _empty_dt())
    # Hand-calc: cascade prefers solve_data/ -> ge = {(g1, e1)}.
    ge1 = out1["group_entity"]
    assert ge1 is not None and ge1.height == 1
    assert ge1.sort("g", "e").row(0) == ("g1", "e1")

    # Drop solve_data file -> input fallback wins.
    (sd / "group_entity.csv").unlink()
    out2 = _load_cumulative_invest(inp, sd, _empty_dt())
    # Hand-calc: solve_data absent -> next cand input/ exists -> ge = {(g1, e2)}.
    ge2 = out2["group_entity"]
    assert ge2 is not None and ge2.height == 1
    assert ge2.sort("g", "e").row(0) == ("g1", "e2")


def test_g_invest_cumulative_alone_does_not_emit_and_max_capacity_is_none(
        tmp_path: Path):
    """Covers A16-g_invest_cumulative_set_alone_does_not_emit
    + A16-cumulative_max_capacity_field_is_none_in_loader.

    Populate ONLY ``g_invest_cumulative.csv`` (gate field).  Loader
    must:
      * set ``g_invest_cumulative`` non-empty (so ``has_feature(d)``
        returns True for the gate),
      * leave ``ed_cumulative_max_capacity = None`` per the Δ.12-drop
        contract (override-only field; never seeded by the loader).
    Then call ``_emit_group_invest_cumulative`` with a stub data object
    where ``group_entity`` and ``edd_invest_set`` are None — the
    all-three-required guard must early-return without emitting any
    constraint (record m.add_cstr calls and assert empty).
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    _write(sd / "g_invest_cumulative.csv", "group\ng1\n")

    out = _load_cumulative_invest(inp, sd, _empty_dt())
    # Hand-calc: g_invest_cumulative read as {(g1,)}; ed_cumulative_max_capacity
    # explicitly assigned None per the Δ.12-drop contract.
    assert out["g_invest_cumulative"] is not None
    assert out["g_invest_cumulative"].height == 1
    assert "ed_cumulative_max_capacity" in out
    assert out["ed_cumulative_max_capacity"] is None

    # Build a stub FlexData with the gate field populated.  has_feature
    # consumes only attribute access via getattr.
    d_stub = SimpleNamespace(
        e_invest_min_total=None, e_divest_min_total=None,
        ed_invest_min_period=None, ed_divest_min_period=None,
        ed_invest_forbidden_no_investment=None,
        ed_invest_cumulative=None,
        ed_cumulative_max_capacity=None, ed_cumulative_min_capacity=None,
        gd_invest_period=None, gd_divest_period=None,
        g_invest_total=None, g_divest_total=None,
        g_invest_cumulative=out["g_invest_cumulative"],
        group_entity=None,  # the missing piece for the emit-guard
        p_group_max_cumulative_flow=None, p_group_min_cumulative_flow=None,
        pd_max_cumulative_flow=None, pd_min_cumulative_flow=None,
        gdt_maxInstantFlow=None, gdt_minInstantFlow=None,
        group_process_node=None,
        edd_invest_set=None,
        p_group_invest_max_cumulative=None,
    )
    # Hand-calc: g_invest_cumulative non-empty -> has_feature returns True.
    assert has_feature(d_stub) is True

    class _Recorder:
        def __init__(self): self.calls = []
        def add_cstr(self, *a, **kw): self.calls.append((a, kw))
    m = _Recorder()
    _emit_group_invest_cumulative(m, d_stub, vars={}, sense="<=")
    # Hand-calc: group_entity is None -> early return; no add_cstr call.
    assert m.calls == []


def test_min_invest_total_seed_param(tmp_path: Path):
    """Covers A16-min_invest_total_seed_param.

    ``e_invest_min_total.csv`` with one row entity=e1,value=3.0.
    ``_read_e_seed`` casts value to Float64, fills nulls with 0.0, and
    wraps as ``Param(("e",), ...)``.  Drives the lower-bound RHS in
    ``_emit_invest_total_minmax(kind='invest', sense='>=')``.
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    _write(sd / "e_invest_min_total.csv", "entity,value\ne1,3.0\n")

    out = _load_cumulative_invest(inp, sd, _empty_dt())
    # Hand-calc: 1 row -> Param(("e",), {(e1,): 3.0}).
    p = out["e_invest_min_total"]
    assert p is not None
    fr = p.frame.sort("e")
    assert fr.height == 1
    assert fr.row(0)[0] == "e1"
    assert fr["value"][0] == pytest.approx(3.0, rel=1e-7)
