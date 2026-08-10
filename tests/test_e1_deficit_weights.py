"""Tests for E1 — deficit-space representative-period weighting.

Covers the phased build of ``specs/rp_weighting_e1_implementation_plan.md``:

* **Phase 0** — the byte-parity fallback (flag OFF ⇒ ``W`` identical to today's
  shape fit) and the require-``--region-groups`` guard.
* **Phase 1** — the ``build_region_deficit_matrix`` assembler: the signed
  extensive net-deficit formula, interior conservation vs the shape fit, and
  extensivity.
* **Phase 2** — cross-process determinism, the degenerate-region fallback, and
  the shared annualization-weight conservation invariant.

DB-backed tests build a schema-complete database from the FlexTool JSON schema
via ``initialize_database`` (CLAUDE.md invariant #3 — never read a checked-in
``.sqlite``). The formula / conservation / extensivity tests are pure-numpy and
need no database.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest
import spinedb_api as api
from spinedb_api import DatabaseMapping, Map, import_data

from flextool._resources import package_data_path
from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.force_include import (
    build_region_deficit_matrix,
)
from flextool.representative_periods.preprocess import (
    _build_clustering_matrix,
    preprocess_representative_periods,
)
from flextool.representative_periods.weights import compute_weight_matrix
from flextool.update_flextool import initialize_database

PERIOD_LENGTH = 24
N_PERIODS = 6
N_RP = 2
SCENARIO = "deficit_scen"
ALT = "deficit_base"
REGIONS = ["region_a", "region_b"]


# ---------------------------------------------------------------------------
# Synthetic horizon
# ---------------------------------------------------------------------------

def _timestep_keys() -> list[str]:
    return [f"t{i:04d}" for i in range(N_PERIODS * PERIOD_LENGTH)]


def _wind(region: str) -> list[float]:
    """Per-hour VRE availability for a region over the whole horizon.

    Two distinct square-wave shape families plus a sustained trough in the last
    period, offset between the two regions so their deficit rows differ.
    """
    values: list[float] = []
    for p in range(N_PERIODS):
        for h in range(PERIOD_LENGTH):
            if p == N_PERIODS - 1:
                values.append(0.05)  # sustained trough (drought)
            elif region == "region_a":
                values.append(1.0 if h < PERIOD_LENGTH // 2 else 0.0)
            else:
                values.append(0.0 if h < PERIOD_LENGTH // 2 else 1.0)
    return values


def _synthetic_series() -> tuple[
    dict[str, list[tuple[str, float]]],
    dict[str, list[tuple[str, float]]],
    dict[str, float],
    list[str],
]:
    """Plain-dict series matching the DB fixture (for the shape-fit baseline)."""
    keys = _timestep_keys()
    profiles = {
        "wind_a": list(zip(keys, _wind("region_a"))),
        "wind_b": list(zip(keys, _wind("region_b"))),
    }
    inflows: dict[str, list[tuple[str, float]]] = {}
    demand_scalars = {"demand_a": -1000.0, "demand_b": -500.0}
    return profiles, inflows, demand_scalars, keys


def _build_deficit_db(db_path: str, *, attach_profiles: bool = True) -> str:
    """Schema-complete DB with two region node-groups.

    ``region_a`` owns node ``demand_a`` (scalar demand 1000) + profile
    ``wind_a``; ``region_b`` owns ``demand_b`` (500) + ``wind_b``. Profiles are
    attached to their region's node via ``unit__node__profile`` when
    ``attach_profiles`` (the signal-present fixture); with ``attach_profiles``
    False no profile maps to any region, so every region's deficit row collapses
    to zero — the degenerate-fallback fixture.
    """
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"

    keys = _timestep_keys()
    timeline_map = Map(keys, [1.0] * len(keys))
    wind_a_map = Map(keys, _wind("region_a"))
    wind_b_map = Map(keys, _wind("region_b"))
    period_timeset_map = Map(["y2050"], ["placeholder_ts"])

    entities = [
        ("timeline", "tl", None),
        ("profile", "wind_a", None),
        ("profile", "wind_b", None),
        ("node", "demand_a", None),
        ("node", "demand_b", None),
        ("unit", "u_a", None),
        ("unit", "u_b", None),
        ("group", "region_a", None),
        ("group", "region_b", None),
        ("group__node", ("region_a", "demand_a"), None),
        ("group__node", ("region_b", "demand_b"), None),
        ("solve", "s1", None),
    ]
    if attach_profiles:
        entities += [
            ("unit__node__profile", ("u_a", "demand_a", "wind_a"), None),
            ("unit__node__profile", ("u_b", "demand_b", "wind_b"), None),
        ]

    with DatabaseMapping(url) as db:
        count, errors = import_data(
            db,
            alternatives=[(ALT, "deficit fixture")],
            scenarios=[(SCENARIO, True, "deficit scenario")],
            scenario_alternatives=[(SCENARIO, ALT)],
            entities=entities,
            parameter_values=[
                ("timeline", "tl", "timestep_duration", timeline_map, ALT),
                ("profile", "wind_a", "profile", wind_a_map, ALT),
                ("profile", "wind_b", "profile", wind_b_map, ALT),
                ("node", "demand_a", "inflow", -1000.0, ALT),
                ("node", "demand_b", "inflow", -500.0, ALT),
                ("solve", "s1", "period_timeset", period_timeset_map, ALT),
            ],
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("deficit fixture")
    return url


def _read_timeset_map(url: str, timeset_name: str, param: str):
    """Return the deserialised ``Map`` value for a timeset parameter."""
    with DatabaseMapping(url) as db:
        db.fetch_all("parameter_value")
        matches = [
            pv
            for pv in db.find_parameter_values(
                entity_class_name="timeset", parameter_definition_name=param
            )
            if pv["entity_name"] == timeset_name
        ]
    assert len(matches) == 1, (
        f"expected exactly one '{param}' on timeset '{timeset_name}', "
        f"got {len(matches)}"
    )
    pv = matches[0]
    return api.from_database(pv["value"], pv["type"])


def _densify_weights(
    weights_map: Map, base_start_keys: list[str], rep_start_keys: list[str]
) -> np.ndarray:
    """Rebuild the dense ``(n_base, n_rp)`` weight matrix from the sparse Map."""
    b_idx = {k: i for i, k in enumerate(base_start_keys)}
    r_idx = {k: i for i, k in enumerate(rep_start_keys)}
    W = np.zeros((len(base_start_keys), len(rep_start_keys)))
    for b_key, inner in zip(weights_map.indexes, weights_map.values):
        i = b_idx[str(b_key)]
        for r_key, w in zip(inner.indexes, inner.values):
            W[i, r_idx[str(r_key)]] = float(w)
    return W


def _expected_shape_fit() -> tuple[np.ndarray, list[int], list[str], list[str]]:
    """Reconstruct today's shape-fit ``W`` and rep indices from the series.

    ``W`` is invariant to feature-row order, so building ``C`` from the
    in-memory series (rather than the DB read order) yields the same matrix.
    """
    profiles, inflows, _demand, keys = _synthetic_series()
    C, n_base = _build_clustering_matrix(profiles, inflows, keys, PERIOD_LENGTH)
    rep_indices = sorted(greedy_convex_hull_clustering(C, N_RP))
    W = compute_weight_matrix(C, rep_indices)
    base_start_keys = [keys[d * PERIOD_LENGTH] for d in range(n_base)]
    rep_start_keys = [keys[i * PERIOD_LENGTH] for i in rep_indices]
    return W, rep_indices, base_start_keys, rep_start_keys


# ---------------------------------------------------------------------------
# Phase 0 — byte-parity fallback + guard
# ---------------------------------------------------------------------------

class TestPhase0Parity:
    def test_default_byte_parity(self, tmp_path):
        """Flag OFF ⇒ emitted ``W`` byte-identical to today's shape fit.

        This is the primary regression guard (plan §4). ``timestep_weight.csv``
        parity follows for free because it is a pure function of ``W``
        (its column sums), so pinning ``W`` pins the annualization weight too.
        """
        url = _build_deficit_db(str(tmp_path / "parity.sqlite"))
        W_exp, _rep, base_keys, rep_keys = _expected_shape_fit()

        # Region groups are PRESENT but the flag is OFF: the region maps must
        # not perturb the default path.
        name = preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=N_RP,
            period_length=PERIOD_LENGTH,
            region_groups=REGIONS,
            deficit_weights=False,
        )
        w_map = _read_timeset_map(url, name, "representative_period_weights")
        W_emit = _densify_weights(w_map, base_keys, rep_keys)

        # The emitted map drops weights <= 1e-10, so sparsify the expected W the
        # same way before the exact comparison.
        W_exp_sparse = np.where(W_exp > 1e-10, W_exp, 0.0)
        assert np.array_equal(W_emit, W_exp_sparse)

    def test_deficit_requires_region_groups(self, tmp_path):
        """``deficit_weights=True`` without region groups raises ``ValueError``."""
        url = _build_deficit_db(str(tmp_path / "guard.sqlite"))
        with pytest.raises(ValueError, match="requires --region-groups"):
            preprocess_representative_periods(
                url,
                SCENARIO,
                n_rp=N_RP,
                period_length=PERIOD_LENGTH,
                region_groups=None,
                deficit_weights=True,
            )


# ---------------------------------------------------------------------------
# Phase 1 — deficit-feature assembler
# ---------------------------------------------------------------------------

class TestAssemblerFormula:
    def test_build_region_deficit_matrix_formula(self):
        """Exact ``d[r,b]`` on hand-built inputs.

        Region A (2 profiles, D_r=10, one time-varying inflow node) and region B
        (1 profile, D_r=5, no inflow) over 2 periods of length 2. Checks the sign
        convention (a supply inflow lowers net load, a demand / negative inflow
        raises it), ``D_r`` counted once, the per-hour profile-axis mean, and the
        reshape/time-integral.
        """
        keys = ["t0", "t1", "t2", "t3"]
        period_length, n_base = 2, 2

        profiles = {
            "pa1": list(zip(keys, [0.0, 1.0, 0.5, 0.5])),
            "pa2": list(zip(keys, [0.5, 0.5, 0.5, 0.5])),
            "pb": list(zip(keys, [1.0, 0.0, 1.0, 1.0])),
        }
        inflows = {"ia": list(zip(keys, [2.0, -3.0, 0.0, 1.0]))}
        demand_scalars: dict[str, float] = {}
        region_profiles = {"region_a": ["pa1", "pa2"], "region_b": ["pb"]}
        region_demand = {"region_a": 10.0, "region_b": 5.0}
        region_nodes = {"region_a": ["ia"], "region_b": []}

        D = build_region_deficit_matrix(
            profiles,
            inflows,
            demand_scalars,
            keys,
            period_length,
            n_base,
            region_profiles=region_profiles,
            region_demand=region_demand,
            region_nodes=region_nodes,
        )

        # Region A: mean avail = [0.25, 0.75, 0.5, 0.5]; vg = 10*(1-mean) =
        # [7.5, 2.5, 5, 5]; nl = vg - inflow = [5.5, 5.5, 5, 4]; integral = [11, 9].
        # Region B: vg = 5*(1-pb) = [0, 5, 0, 0]; integral = [5, 0].
        expected = np.array([[11.0, 9.0], [5.0, 0.0]])
        assert D is not None
        assert D.shape == (2, 2)
        assert np.allclose(D, expected)

    def test_deficit_matrix_extensive(self):
        """Doubling a period's hourly deficit doubles its column integral.

        A raw (extensive) integral scales linearly; a min–max-normalised feature
        would not. A pure-inflow region (no profile, D_r=0) isolates the scaling.
        """
        keys = ["t0", "t1", "t2", "t3"]
        period_length, n_base = 2, 2
        region_profiles = {"region_x": []}
        region_demand = {"region_x": 0.0}
        region_nodes = {"region_x": ["nx"]}

        base_inflow = [1.0, -2.0, 3.0, -4.0]
        D1 = build_region_deficit_matrix(
            {},
            {"nx": list(zip(keys, base_inflow))},
            {},
            keys,
            period_length,
            n_base,
            region_profiles=region_profiles,
            region_demand=region_demand,
            region_nodes=region_nodes,
        )
        D2 = build_region_deficit_matrix(
            {},
            {"nx": list(zip(keys, [2.0 * v for v in base_inflow]))},
            {},
            keys,
            period_length,
            n_base,
            region_profiles=region_profiles,
            region_demand=region_demand,
            region_nodes=region_nodes,
        )
        assert D1 is not None and D2 is not None
        assert np.allclose(D2, 2.0 * D1)

    def test_deficit_conservation_interior(self):
        """Deficit-fit reconstructs interior droughts deeper than the shape fit.

        Two regions with disjoint droughts (A weeks {1,2}, B weeks {4,5}) plus a
        mild rep. In *shape* space the interior drought targets look like the
        mild rep, so the shape fit puts their weight on the low-deficit rep and
        under-reconstructs the drought; the deficit fit puts weight on the
        drought rep and reconstructs it.
        """
        # Deficit matrix D (2 regions x 6 weeks). Reps: 0 (mild), 2 (A peak),
        # 5 (B peak). Targets: week 1 (A drought), week 4 (B drought).
        D = np.array(
            [
                [0.0, 6.0, 10.0, 0.0, 0.0, 0.0],   # region A
                [0.0, 0.0, 0.0, 0.0, 6.0, 10.0],   # region B
            ]
        )
        # Shape matrix C (3 features x 6 weeks). Drought TARGET weeks (1, 4)
        # share the mild rep's shape; only the drought PEAK reps (2, 5) are
        # shape-distinct.
        v0, v2, v5 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
        C = np.array([v0, v0, v2, v0, v0, v5]).T
        reps = [0, 2, 5]

        W_shape = compute_weight_matrix(C, reps)
        W_def = compute_weight_matrix(D, reps)

        D_reps = D[:, reps]  # (2, 3)

        # (a) Interior conservation: the deficit fit reproduces D[:, b] for the
        # interior drought targets.
        for b in (1, 4):
            recon = D_reps @ W_def[b, :]
            assert np.allclose(recon, D[:, b], atol=1e-6)

        # (b) Cumulative drawdown over each drought run is strictly deeper under
        # the deficit fit than the shape fit.
        def region_run_depth(W, region, weeks):
            recon = D_reps[region, :] @ W[weeks, :].T  # per-week reconstruction
            return float(recon.sum())

        a_def = region_run_depth(W_def, 0, [1, 2])
        a_shape = region_run_depth(W_shape, 0, [1, 2])
        b_def = region_run_depth(W_def, 1, [4, 5])
        b_shape = region_run_depth(W_shape, 1, [4, 5])
        assert a_def > a_shape + 1e-6, (a_def, a_shape)
        assert b_def > b_shape + 1e-6, (b_def, b_shape)


# ---------------------------------------------------------------------------
# Phase 2 — determinism / degenerate fallback / annualization conservation
# ---------------------------------------------------------------------------

_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    from flextool.representative_periods.clustering import (
        greedy_convex_hull_clustering,
    )
    from flextool.representative_periods.force_include import (
        build_region_deficit_matrix,
    )
    from flextool.representative_periods.weights import compute_weight_matrix

    n_base, period_length = 8, 3
    keys = [f"t{i:04d}" for i in range(n_base * period_length)]

    # Deterministic, RNG-free per-hour series for two regions.
    def wave(offset):
        return [
            0.5 + 0.4 * np.sin((i + offset) * 0.7)
            for i in range(n_base * period_length)
        ]

    profiles = {
        "wa": list(zip(keys, wave(0))),
        "wb": list(zip(keys, wave(3))),
    }
    inflows = {"na": list(zip(keys, wave(1)))}
    demand_scalars = {}
    region_profiles = {"ra": ["wa"], "rb": ["wb"]}
    region_demand = {"ra": 100.0, "rb": 50.0}
    region_nodes = {"ra": ["na"], "rb": []}

    D = build_region_deficit_matrix(
        profiles, inflows, demand_scalars, keys, period_length, n_base,
        region_profiles=region_profiles, region_demand=region_demand,
        region_nodes=region_nodes,
    )
    rep = sorted(greedy_convex_hull_clustering(D, 3))
    W = compute_weight_matrix(D, rep)
    np.save(sys.argv[1], W)
    """
)


class TestPhase2:
    def test_deficit_weights_cross_process(self, tmp_path):
        """Assembler + fit reproduce ``W`` bit-for-bit across two processes.

        Two separate Python processes under different ``PYTHONHASHSEED`` values
        (cross-process determinism, per CLAUDE.md memory) must produce an
        identical ``W`` — the reduction is deterministic (sorted region order,
        RNG-free fit).
        """
        script = tmp_path / "cross.py"
        script.write_text(_CROSS_PROCESS_SCRIPT)
        out_a = tmp_path / "wa.npy"
        out_b = tmp_path / "wb.npy"

        import os

        env_a = dict(os.environ, PYTHONHASHSEED="0")
        env_b = dict(os.environ, PYTHONHASHSEED="1")
        subprocess.run(
            [sys.executable, str(script), str(out_a)], check=True, env=env_a
        )
        subprocess.run(
            [sys.executable, str(script), str(out_b)], check=True, env=env_b
        )
        W_a = np.load(out_a)
        W_b = np.load(out_b)
        assert np.array_equal(W_a, W_b)

    def test_deficit_degenerate_fallback(self, tmp_path):
        """No-signal regions ⇒ ``W`` falls back to the shape fit.

        With no profile attached to any region node, every region's deficit row
        is identically zero, the assembler returns ``None``, and the fit reverts
        to the shape matrix — so ``W`` equals the byte-parity shape fit.
        """
        url = _build_deficit_db(
            str(tmp_path / "degenerate.sqlite"), attach_profiles=False
        )
        W_exp, _rep, base_keys, rep_keys = _expected_shape_fit()

        name = preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=N_RP,
            period_length=PERIOD_LENGTH,
            region_groups=REGIONS,
            deficit_weights=True,
        )
        w_map = _read_timeset_map(url, name, "representative_period_weights")
        W_emit = _densify_weights(w_map, base_keys, rep_keys)
        W_exp_sparse = np.where(W_exp > 1e-10, W_exp, 0.0)
        assert np.array_equal(W_emit, W_exp_sparse)

    def test_annualization_total_conserved(self, tmp_path):
        """Under E1 the shared annualization weight is conserved and non-negative.

        The annualization weight is ``w_r = (Σ_b W[b,r]) · n_rp/n_base`` (the
        column sums of ``W``, scaled). Because every row of ``W`` sums to 1,
        ``Σ_r w_r == n_rp`` for any row-stochastic ``W`` — a fit-invariant that
        bounds MAJOR #1's distribution shift. The per-rep concentration
        ``max_r w_r`` is recorded against the shape fit (it may move; only the
        total is pinned).
        """
        url = _build_deficit_db(str(tmp_path / "annual.sqlite"))
        W_exp, _rep, base_keys, rep_keys = _expected_shape_fit()

        name = preprocess_representative_periods(
            url,
            SCENARIO,
            n_rp=N_RP,
            period_length=PERIOD_LENGTH,
            region_groups=REGIONS,
            deficit_weights=True,
        )
        w_map = _read_timeset_map(url, name, "representative_period_weights")
        W_emit = _densify_weights(w_map, base_keys, rep_keys)

        n_base = W_emit.shape[0]
        n_rp = W_emit.shape[1]
        w_r = W_emit.sum(axis=0) * (n_rp / n_base)

        assert abs(float(w_r.sum()) - n_rp) < 1e-9
        assert float(w_r.min()) >= -1e-12
        assert float(w_r.max()) <= n_rp + 1e-9

        # Record the concentration shift vs the shape fit (informational — only
        # the total is a hard invariant).
        w_r_shape = np.where(W_exp > 1e-10, W_exp, 0.0).sum(axis=0) * (
            n_rp / n_base
        )
        assert float(w_r_shape.sum()) == pytest.approx(n_rp, abs=1e-9)
