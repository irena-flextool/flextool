"""Tier-9 synthetic driver: storage-backed reserve energy adequacy
(``reserve_duration``, issue #322).

Exercises ``_reserve._add_storage_reserve_constraints`` end to end via
``build_flextool``.  Every case runs under ``FLEXTOOL_AUTOSCALE_STRICT=1``
and additionally re-solves through ``apply_layer2`` so the two new
constraint families are proven complete in the autoscale registry
(CLAUDE.md invariant #1) — a missing entry makes ``apply_layer2`` raise.

See ``tests/engine_polars/fixtures/flex_toy_reserve_storage.py`` for the
construction and closed-form derivations.
"""
from __future__ import annotations

import pytest

from polar_high import Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.autoscale import (
    apply_layer2, ScalingConfig, unscale_solution,
)

from flex_toy_reserve_storage import data

_OPTS = {"random_seed": 42, "parallel": "off"}
_UP = "reserve_storage_up_floor"
_DN = "reserve_storage_down_headroom"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    """Make missing autoscale registry entries loud for every case."""
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _solve(d):
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options=_OPTS)
    assert sol.optimal
    return pb, sol


def _solve_scaled(d):
    """Build, apply Layer-2 autoscale (STRICT — raises on an unregistered
    family), solve, unscale.  Returns the (bit-exact) unscaled solution."""
    pb = Problem()
    build_flextool(pb, d)
    plan = apply_layer2(pb, ScalingConfig())
    sol = pb.solve(options=_OPTS)
    unscale_solution(sol, plan)
    assert sol.optimal
    return pb, sol


def _vals(sol, name):
    return sol.value(name)["value"].to_list()


# ── Case 1: UP floor binds; raising duration tightens reserve ────────────

def test_up_floor_binds_and_tightens_with_duration():
    # duration = 0.5 h: energy floor caps v_reserve at cap/(unitsize·dur)
    #   = 20 / (100·0.5) = 0.4; reserveBalance ⇒ vq = (50-40)/50 = 0.2.
    pb, sol = _solve(data(direction="up", duration=0.5))
    assert _UP in pb.cstr_names()
    assert _DN not in pb.cstr_names()
    vr_half = _vals(sol, "v_reserve")
    vq_half = _vals(sol, "vq_reserve")
    assert vr_half == pytest.approx([0.4, 0.4])
    assert vq_half == pytest.approx([0.2, 0.2])

    # duration = 1.0 h halves the energy allowance: v_reserve → 0.2,
    # vq → 0.6.  Reserve tightens, slack (and objective) rise.
    pb2, sol2 = _solve(data(direction="up", duration=1.0))
    vr_full = _vals(sol2, "v_reserve")
    vq_full = _vals(sol2, "vq_reserve")
    assert vr_full == pytest.approx([0.2, 0.2])
    assert vq_full == pytest.approx([0.6, 0.6])
    assert vr_full[0] < vr_half[0]           # reserve tightened
    assert vq_full[0] > vq_half[0]           # slack rose
    assert sol2.obj > sol.obj                # objective rose


def test_up_floor_autoscale_strict_roundtrip():
    """Layer-2 (STRICT) must not raise on the new ENERGY family, and the
    unscaled objective is bit-exact."""
    _, raw = _solve(data(direction="up", duration=0.5))
    _, scaled = _solve_scaled(data(direction="up", duration=0.5))
    assert scaled.obj == pytest.approx(raw.obj, rel=1e-9)


# ── Case 2: DOWN headroom + efficiency (÷slope ⇒ ×η) ─────────────────────

def test_down_headroom_binds_with_efficiency():
    # eta = 0.9 ⇒ E_dn = v_reserve·100·dur·(1/slope) = v_reserve·100·0.5·0.9.
    # LP drives v_state → 0 (max headroom): committed down MW ≤
    #   cap·unitsize / (0.5·0.9) = 20 / 0.45 = 44.44 ⇒ v_reserve = 0.4444,
    # vq = (50-44.44)/50 = 0.1111.
    pb, sol = _solve(data(direction="down", eta=0.9, duration=0.5))
    assert _DN in pb.cstr_names()
    assert _UP not in pb.cstr_names()
    assert _vals(sol, "v_reserve") == pytest.approx([0.4 / 0.9, 0.4 / 0.9])
    assert _vals(sol, "vq_reserve") == pytest.approx([1.0 / 9.0, 1.0 / 9.0])
    # Autoscale STRICT roundtrip (down family registered).
    _, scaled = _solve_scaled(data(direction="down", eta=0.9, duration=0.5))
    assert scaled.obj == pytest.approx(sol.obj, rel=1e-9)


def test_down_headroom_tighter_efficiency_costs_more():
    """Lower efficiency stores less per charge-MW ⇒ same headroom backs
    LESS reserve is FALSE here; instead ×η means lower η ⇒ smaller E_dn ⇒
    MORE reserve fits.  Verify the monotonic direction against η=1."""
    _, lossless = _solve(data(direction="down", eta=1.0, duration=0.5))
    _, lossy = _solve(data(direction="down", eta=0.9, duration=0.5))
    # eta=1: v_reserve ≤ 20/(100·0.5·1) = 0.4.  eta=0.9: ≤ 0.4/0.9 ≈ 0.444.
    assert _vals(lossy, "v_reserve")[0] > _vals(lossless, "v_reserve")[0]


# ── Case 3: connection provider (pss / pruna are class-agnostic) ─────────

def test_connection_provider_up_floor_binds():
    """The engine keys purely on the process id — a connection-backed
    provider couples to storage identically to a unit-backed one."""
    pb, sol = _solve(data(direction="up", duration=0.5, provider="conn_bat_elec"))
    assert _UP in pb.cstr_names()
    assert _vals(sol, "v_reserve") == pytest.approx([0.4, 0.4])
    assert _vals(sol, "vq_reserve") == pytest.approx([0.2, 0.2])


# ── Case 4: no-op guard — absent reserve_duration ⇒ no coupling ──────────

def test_no_op_without_duration():
    """Omitting reserve_duration emits NEITHER coupling constraint and
    leaves the LP/objective at the pre-feature value (§9 no-op proof)."""
    pb, sol = _solve(data(direction="up", duration=None))
    names = pb.cstr_names()
    assert _UP not in names
    assert _DN not in names
    # With no energy floor, reserve is limited only by the maxFlow coupling
    # (≤ 1.0) and fully covers the 50 MW reservation ⇒ vq = 0, obj = 0.
    assert sol.obj == pytest.approx(0.0, abs=1e-9)
    assert _vals(sol, "vq_reserve") == pytest.approx([0.0, 0.0])
    # Byte-identical proof: the feature-on build differs ONLY by the added
    # up-floor row; strip the duration and that row (and its objective
    # impact) is gone.
    pb_on, _ = _solve(data(direction="up", duration=0.5))
    assert set(pb.cstr_names()) | {_UP} == set(pb_on.cstr_names())


# ── Case 5: feasibility guard — a too-short battery stays feasible ───────

def test_feasibility_with_long_duration():
    """A multi-hour duration on a small battery must NOT make the model
    infeasible: reserve is simply reduced and vq_reserve absorbs the
    shortfall at the existing penalty (§3 — no new slack variable)."""
    # duration = 10 h ⇒ v_reserve ≤ 20/(100·10) = 0.02; vq = (50-2)/50 = 0.96.
    pb, sol = _solve(data(direction="up", duration=10.0))
    assert sol.optimal
    assert _vals(sol, "v_reserve") == pytest.approx([0.02, 0.02])
    vq = _vals(sol, "vq_reserve")
    assert vq == pytest.approx([0.96, 0.96])
    assert all(v < 1.0 for v in vq)          # slack never saturates its cap
    # Autoscale STRICT stays clean under the extreme coefficient range.
    _, scaled = _solve_scaled(data(direction="up", duration=10.0))
    assert scaled.obj == pytest.approx(sol.obj, rel=1e-9)
