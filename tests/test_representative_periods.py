"""Tests for representative period clustering, weights, and FlexTool integration.

Unit tests for the clustering and weight algorithms, plus functional tests
that verify the full pipeline: build model → cluster → solve with RP weights.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.weights import (
    compute_nearest_assignment,
    compute_weight_matrix,
    fit_convex_weights,
    project_onto_simplex,
)


# ---------------------------------------------------------------------------
# Unit tests: simplex projection
# ---------------------------------------------------------------------------

class TestSimplexProjection:
    def test_already_on_simplex(self):
        v = np.array([0.3, 0.5, 0.2])
        w = project_onto_simplex(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= 0)
        assert np.allclose(w, v)

    def test_negative_values(self):
        v = np.array([-1.0, 2.0, 0.5])
        w = project_onto_simplex(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= -1e-10)

    def test_all_equal(self):
        v = np.array([1.0, 1.0, 1.0])
        w = project_onto_simplex(v)
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w, [1/3, 1/3, 1/3])

    def test_single_element(self):
        w = project_onto_simplex(np.array([5.0]))
        assert np.allclose(w, [1.0])


# ---------------------------------------------------------------------------
# Unit tests: convex weight fitting
# ---------------------------------------------------------------------------

class TestConvexWeights:
    def test_point_on_vertex(self):
        """A point that IS a representative should get weight 1 on itself."""
        R = np.array([[1, 0], [0, 1]], dtype=float)  # 2 RPs, 2 features
        c = np.array([1.0, 0.0])  # exactly RP 0
        w = fit_convex_weights(R, c)
        assert np.allclose(w.sum(), 1.0)
        assert w[0] > 0.99

    def test_midpoint(self):
        """A point at the midpoint of two RPs should get ~equal weights."""
        R = np.array([[0, 1], [0, 1]], dtype=float)
        c = np.array([0.5, 0.5])
        w = fit_convex_weights(R, c)
        assert np.allclose(w.sum(), 1.0)
        assert np.allclose(w, [0.5, 0.5], atol=1e-3)

    def test_weights_nonnegative(self):
        rng = np.random.default_rng(42)
        R = rng.random((20, 5))
        c = rng.random(20)
        w = fit_convex_weights(R, c)
        assert np.allclose(w.sum(), 1.0)
        assert np.all(w >= -1e-8)


# ---------------------------------------------------------------------------
# Unit tests: greedy clustering
# ---------------------------------------------------------------------------

class TestGreedyClustering:
    def test_deterministic(self):
        """Same input always gives same output."""
        C = np.random.default_rng(42).random((10, 20))
        r1 = greedy_convex_hull_clustering(C, 5)
        r2 = greedy_convex_hull_clustering(C, 5)
        assert r1 == r2

    def test_selects_correct_count(self):
        C = np.random.default_rng(42).random((10, 20))
        r = greedy_convex_hull_clustering(C, 7)
        assert len(r) == 7
        assert len(set(r)) == 7  # all unique

    def test_all_periods_requested(self):
        C = np.random.default_rng(42).random((10, 5))
        r = greedy_convex_hull_clustering(C, 5)
        assert sorted(r) == [0, 1, 2, 3, 4]

    def test_more_than_available(self):
        C = np.random.default_rng(42).random((10, 3))
        r = greedy_convex_hull_clustering(C, 10)
        assert sorted(r) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Unit tests: weight matrix
# ---------------------------------------------------------------------------

class TestWeightMatrix:
    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(42)
        C = rng.random((20, 15))
        rep = greedy_convex_hull_clustering(C, 5)
        W = compute_weight_matrix(C, rep)
        assert W.shape == (15, 5)
        for d in range(15):
            assert abs(W[d, :].sum() - 1.0) < 1e-6
            assert np.all(W[d, :] >= -1e-8)

    def test_representatives_have_high_self_weight(self):
        rng = np.random.default_rng(42)
        C = rng.random((20, 15))
        rep = greedy_convex_hull_clustering(C, 5)
        W = compute_weight_matrix(C, rep)
        for r_idx, r_col in enumerate(rep):
            assert W[r_col, r_idx] > 0.9


# ---------------------------------------------------------------------------
# Unit tests: single-nearest assignment
# ---------------------------------------------------------------------------

class TestNearestAssignment:
    def test_rows_are_one_hot(self):
        rng = np.random.default_rng(42)
        C = rng.random((20, 15))
        rep = greedy_convex_hull_clustering(C, 5)
        W = compute_nearest_assignment(C, rep)
        assert W.shape == (15, 5)
        for d in range(15):
            assert W[d, :].sum() == 1.0
            assert np.count_nonzero(W[d, :]) == 1
            assert np.all((W[d, :] == 0.0) | (W[d, :] == 1.0))

    def test_representative_maps_to_itself(self):
        rng = np.random.default_rng(7)
        C = rng.random((20, 15))
        rep = greedy_convex_hull_clustering(C, 5)
        W = compute_nearest_assignment(C, rep)
        for r_idx, r_col in enumerate(rep):
            # A base period that IS a rep has distance 0 to itself → self weight.
            assert W[r_col, r_idx] == 1.0

    def test_hand_constructed_argmin(self):
        # 3 features x 5 base periods; reps are base columns 0, 2, 4.
        C = np.array(
            [
                [0.0, 1.0, 10.0, 11.0, 20.0],
                [0.0, 1.0, 10.0, 11.0, 20.0],
                [0.0, 1.0, 10.0, 11.0, 20.0],
            ]
        )
        rep = [0, 2, 4]
        W = compute_nearest_assignment(C, rep)
        # Base 0 -> rep 0 (self); base 1 (=1) nearest to rep 0 (=0);
        # base 2 -> rep 1 (self, col 2); base 3 (=11) nearest to rep 1 (=10);
        # base 4 -> rep 2 (self, col 4).
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        assert np.array_equal(W, expected)

    def test_deterministic_repeated_calls(self):
        rng = np.random.default_rng(123)
        C = rng.random((12, 30))
        rep = greedy_convex_hull_clustering(C, 6)
        W1 = compute_nearest_assignment(C, rep)
        W2 = compute_nearest_assignment(C, rep)
        assert np.array_equal(W1, W2)

    def test_exact_tie_breaks_to_lowest_rep(self):
        # Base column 1 is exactly equidistant from reps at cols 0 and 2.
        C = np.array([[0.0, 1.0, 2.0]])
        rep = [0, 2]
        W = compute_nearest_assignment(C, rep)
        # Tie → lowest rep index (r=0, i.e. rep_indices[0]=col 0).
        assert np.array_equal(W[1, :], np.array([1.0, 0.0]))

    # -- k-nearest inverse-distance weighting ------------------------------

    def test_k1_byte_identical_to_default(self):
        rng = np.random.default_rng(2024)
        C = rng.random((20, 15))
        rep = greedy_convex_hull_clustering(C, 5)
        W_default = compute_nearest_assignment(C, rep)
        W_k1 = compute_nearest_assignment(C, rep, k=1)
        assert np.array_equal(W_default, W_k1)

    def test_k2_hand_computation(self):
        # Single feature; reps at cols 0 (value 0), 1 (value 10), 2 (value 30).
        C = np.array([[0.0, 10.0, 30.0, 4.0]])
        rep = [0, 1, 2]
        W = compute_nearest_assignment(C, rep, k=2)
        # Base 3 (value 4): distances to reps = [4, 6, 26]. Two nearest are
        # rep 0 (d=4) and rep 1 (d=6). Inverse-distance weights:
        # 1/4 and 1/6 → normalized: (1/4)/(1/4+1/6), (1/6)/(1/4+1/6).
        inv = np.array([1.0 / 4.0, 1.0 / 6.0])
        norm = inv / inv.sum()
        expected_row = np.array([norm[0], norm[1], 0.0])
        assert np.allclose(W[3, :], expected_row)
        # Exactly 2 nonzero entries; row sums to 1.
        assert np.count_nonzero(W[3, :]) == 2
        assert W[3, :].sum() == pytest.approx(1.0)

    def test_k2_two_nonzero_and_row_sums(self):
        rng = np.random.default_rng(99)
        C = rng.random((10, 12))
        rep = greedy_convex_hull_clustering(C, 5)
        W = compute_nearest_assignment(C, rep, k=2)
        for d in range(C.shape[1]):
            nz = np.count_nonzero(W[d, :])
            # Reps coincide with themselves → 1 nonzero; others → 2.
            assert nz in (1, 2)
            assert W[d, :].sum() == pytest.approx(1.0)
        # A base period that IS a rep gets a clean one-hot (zero-distance guard).
        for r_idx, r_col in enumerate(rep):
            assert W[r_col, r_idx] == 1.0
            assert np.count_nonzero(W[r_col, :]) == 1

    def test_k2_zero_distance_guard_one_hot(self):
        # Base col 3 equals rep col 1 exactly → one-hot, no NaN/inf even at k=2.
        C = np.array(
            [
                [0.0, 10.0, 30.0, 10.0],
                [0.0, 10.0, 30.0, 10.0],
            ]
        )
        rep = [0, 1, 2]
        W = compute_nearest_assignment(C, rep, k=2)
        assert np.all(np.isfinite(W))
        assert np.array_equal(W[3, :], np.array([0.0, 1.0, 0.0]))
        assert np.count_nonzero(W[3, :]) == 1

    def test_k2_deterministic_repeated_calls(self):
        rng = np.random.default_rng(321)
        C = rng.random((12, 30))
        rep = greedy_convex_hull_clustering(C, 6)
        W1 = compute_nearest_assignment(C, rep, k=2)
        W2 = compute_nearest_assignment(C, rep, k=2)
        assert np.array_equal(W1, W2)

    def test_k2_tie_breaks_to_lowest_rep(self):
        # Base col 1 (value 1) is equidistant from reps at cols 0 (0) and 2 (2),
        # and rep at col 3 (value 5) is farther. With k=2 the two nearest are
        # the tied pair → equal split; STABLE sort keeps them ordered by rep
        # index so both are selected deterministically.
        C = np.array([[0.0, 1.0, 2.0, 5.0]])
        rep = [0, 2, 3]
        W = compute_nearest_assignment(C, rep, k=2)
        # Reps 0 and 1 (cols 0, 2) are the two nearest, both at distance 1 →
        # equal 0.5/0.5 split; rep 2 (col 3) is zero.
        assert np.allclose(W[1, :], np.array([0.5, 0.5, 0.0]))

    def test_k_greater_than_nrp_clamps(self):
        C = np.array([[0.0, 1.0, 2.0]])
        rep = [0, 2]
        # k=5 > n_rp=2 → clamp to 2; must not error and row sums to 1.
        W = compute_nearest_assignment(C, rep, k=5)
        assert W.shape == (3, 2)
        for d in range(3):
            assert W[d, :].sum() == pytest.approx(1.0)

    def test_k_zero_raises(self):
        C = np.array([[0.0, 1.0, 2.0]])
        rep = [0, 2]
        with pytest.raises(ValueError):
            compute_nearest_assignment(C, rep, k=0)


# ---------------------------------------------------------------------------
# Unit test: _read_time_series scalar handling
# ---------------------------------------------------------------------------

class TestReadTimeSeriesScalarFiltering:
    """Regression: scalar inflows must be dropped, not crash the pipeline.

    ``params_to_dict`` returns a scalar float as ``str(value)`` (a plain
    string), but ``_build_clustering_matrix`` iterates each entry's
    value as ``for k, v in ts_data:`` — iterating a string yields
    single-character substrings, which then fail to unpack into
    ``(k, v)`` with ``not enough values to unpack (expected 2, got 1)``.
    Real-world databases routinely set ``node.inflow`` to a constant
    scalar; without this filter, RP preprocessing crashes on the first
    such node.
    """

    def test_drops_scalar_inflows_keeps_map_inflows(self, tmp_path):
        from spinedb_api import DatabaseMapping, Map, import_data
        from flextool.representative_periods.preprocess import (
            _read_time_series,
        )

        db_path = tmp_path / "scalars.sqlite"
        url = f"sqlite:///{db_path}"

        with DatabaseMapping(url, create=True) as db:
            inflow_map = Map(["t1", "t2", "t3"], [1.0, 2.0, 3.0])
            _, errors = import_data(
                db,
                entity_classes=[("node", ()), ("profile", ())],
                parameter_definitions=[
                    ("node", "inflow", None, None,
                     "Negative for demand, positive for supply."),
                    ("profile", "profile", None, None, "Profile time series."),
                ],
                entities=[
                    ("node", "scalar_demand_node", None),
                    ("node", "timeseries_inflow_node", None),
                    ("profile", "wind", None),
                ],
                parameter_values=[
                    ("node", "scalar_demand_node", "inflow", -1500.0),
                    ("node", "timeseries_inflow_node", "inflow", inflow_map),
                    ("profile", "wind", "profile", inflow_map),
                ],
            )
            if errors:
                raise RuntimeError(f"import errors: {errors[:5]}")
            db.commit_session("scalar-inflow regression fixture")

        with DatabaseMapping(url) as db:
            db.fetch_all("parameter_value")
            profiles, inflows, demand_scalars = _read_time_series(db)

        # The Map-valued entries survive as list-of-tuple time series.
        assert "timeseries_inflow_node" in inflows, (
            f"Map-valued inflow dropped unexpectedly. Got: {inflows!r}"
        )
        assert isinstance(inflows["timeseries_inflow_node"], list)
        assert "wind" in profiles and isinstance(profiles["wind"], list)

        # The scalar inflow is filtered out (would otherwise crash the
        # downstream `for k, v in ts_data:` unpack at preprocess.py:109).
        assert "scalar_demand_node" not in inflows, (
            f"Scalar inflow leaked through; would crash _build_clustering_matrix. "
            f"Got: {inflows!r}"
        )

        # The scalar inflow is instead collected as a demand level (coerced to
        # float) for force-include's per-node demand weighting.
        assert demand_scalars.get("scalar_demand_node") == -1500.0, (
            f"Scalar demand level not collected. Got: {demand_scalars!r}"
        )
        assert "timeseries_inflow_node" not in demand_scalars


# ---------------------------------------------------------------------------
# Functional tests: full pipeline
# ---------------------------------------------------------------------------

def _build_test_db(db_path: str, yaml_path: str, seed: int = 42) -> None:
    """Initialize DB, build model, return."""
    from flextool._resources import package_data_path
    from flextool.update_flextool import initialize_database
    from flextool.model_builder.build_model import build_model
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    build_model(yaml_path, db_url=f"sqlite:///{db_path}", seed=seed)


def _run_flextool(db_path: str, out_path: str, scenario: str) -> float:
    """Run FlexTool and return total_cost."""
    work_cwd = Path(db_path).parent
    result = subprocess.run(
        [sys.executable, "run_flextool.py",
         f"sqlite:///{db_path}", f"sqlite:///{out_path}",
         "--scenario-name", scenario,
         "--work-folder", str(work_cwd)],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if result.returncode != 0:
        pytest.fail(f"FlexTool failed:\n{result.stdout}\n{result.stderr}")
    # Parse total_cost from output
    for line in result.stdout.split("\n"):
        if "total_cost.val" in line:
            return float(line.split("=")[1].strip())
    pytest.fail(f"Could not find total_cost in output:\n{result.stdout}")


def _setup_rp_scenario(db_url: str, n_rp: int, period_length: int, storage_nodes: list[str]) -> None:
    """Run RP pre-processing and create a scenario that uses the RP timeset.

    The pre-processing now writes both the timeset entity AND the solve's
    period_timeset override in the hull_* alternative. We only need to add
    the storage_binding_method overrides and wire up the scenario.
    """
    from spinedb_api import DatabaseMapping, import_data
    from flextool.representative_periods.preprocess import preprocess_representative_periods

    preprocess_representative_periods(db_url, "generated_scenario", n_rp, period_length)
    hull_alt = f"hull_{n_rp}rp_{period_length}h"

    with DatabaseMapping(db_url) as db:
        rp_alt = "rp_override"
        param_values = []
        for node_name in storage_nodes:
            param_values.append(
                ("node", node_name, "storage_binding_method",
                 "bind_within_solve_blended_weights", rp_alt)
            )

        num, log = import_data(db,
            alternatives=[(rp_alt, "RP overrides")],
            scenarios=[("rp_scenario", True, "RP test")],
            scenario_alternatives=[
                ("rp_scenario", "generated"),
                ("rp_scenario", hull_alt),
                ("rp_scenario", rp_alt),
            ],
            parameter_values=param_values,
            entity_alternatives=[
                ("node", node_name, rp_alt, True) for node_name in storage_nodes
            ],
        )
        db.commit_session("RP scenario setup")


@pytest.fixture
def small_yaml(tmp_path):
    """Create a small YAML spec for testing."""
    yaml_content = """\
profile_lengths: 168

node_locations:
  island:
    number: 3
    distribution:
      name: pert
      min: 50
      mode: 120
      max: 300

grids:
  elec:
    node_group: island
    demand:
      source: demand_elec
      distribution:
        name: pert
        min: 500
        mode: 1200
        max: 2500
      nodes_with_demand: 3
    connections_per_node:
      distribution:
        name: pert_integer
        min: 1
        mode: 2
        max: 3
    cost-per-kw-per-km: 20
    loss-per-100km: 0.01

fuels:
  natgas:
    price: 10

profiles:
  wind:
    auto_correlate:
      pattern_length_average: 72
      pattern_length_st_dev: 48
      distribution:
        name: pert
        min: 0.05
        mode: 0.4
        max: 0.7
  demand_elec:
    auto_correlate:
      pattern_length_average: 24
      pattern_length_st_dev: 0
      distribution:
        name: pert
        min: -1.0
        mode: -0.8
        max: -0.6

storages:
  elec:
    battery:
      storage:
        invest_cost: 300
      connection:
        invest_cost: 0
        efficiency: 0.93
      fix_storage_to_connection_ratio: 4

technologies:
  natgas:
    elec:
      open_cycle:
        conversion_method: constant_efficiency
        efficiency: 0.35
        invest_cost: 700
  wind:
    elec:
      wind_pp:
        conversion_method: constant_efficiency
        efficiency: 1.0
        invest_cost: 800
"""
    yaml_file = tmp_path / "test_model.yaml"
    yaml_file.write_text(yaml_content)
    return str(yaml_file)


@pytest.mark.slow
class TestRPAllRepresented:
    """When ALL periods are selected as representatives, the result should
    match the full model very closely."""

    def test_all_represented_matches_full(self, small_yaml, tmp_path):
        db_full = str(tmp_path / "full.sqlite")
        db_rp = str(tmp_path / "rp.sqlite")
        out_full = str(tmp_path / "out_full.sqlite")
        out_rp = str(tmp_path / "out_rp.sqlite")

        # Build two identical models
        _build_test_db(db_full, small_yaml, seed=42)
        _build_test_db(db_rp, small_yaml, seed=42)

        # Full model: 168h = 7 days of 24h
        cost_full = _run_flextool(db_full, out_full, "generated_scenario")

        # RP model: ALL 7 days selected
        storage_nodes = [
            "battery_Mercury_elec", "battery_Venus_elec", "battery_Earth_elec"
        ]
        _setup_rp_scenario(f"sqlite:///{db_rp}", n_rp=7, period_length=24,
                          storage_nodes=storage_nodes)
        cost_rp = _run_flextool(db_rp, out_rp, "rp_scenario")

        # Should be very close (within 1%)
        rel_diff = abs(cost_rp - cost_full) / abs(cost_full)
        assert rel_diff < 0.01, (
            f"All-represented RP cost {cost_rp} differs from full {cost_full} "
            f"by {rel_diff:.2%}"
        )


@pytest.mark.slow
class TestRPHalfRepresented:
    """With about half the periods represented, results should be reasonably
    close to the full model."""

    def test_half_represented_close(self, small_yaml, tmp_path):
        db_full = str(tmp_path / "full.sqlite")
        db_rp = str(tmp_path / "rp.sqlite")
        out_full = str(tmp_path / "out_full.sqlite")
        out_rp = str(tmp_path / "out_rp.sqlite")

        _build_test_db(db_full, small_yaml, seed=42)
        _build_test_db(db_rp, small_yaml, seed=42)

        cost_full = _run_flextool(db_full, out_full, "generated_scenario")

        # RP model: 4 of 7 days
        storage_nodes = [
            "battery_Mercury_elec", "battery_Venus_elec", "battery_Earth_elec"
        ]
        _setup_rp_scenario(f"sqlite:///{db_rp}", n_rp=4, period_length=24,
                          storage_nodes=storage_nodes)
        cost_rp = _run_flextool(db_rp, out_rp, "rp_scenario")

        # Should be within 20%
        rel_diff = abs(cost_rp - cost_full) / abs(cost_full)
        assert rel_diff < 0.20, (
            f"Half-represented RP cost {cost_rp} differs from full {cost_full} "
            f"by {rel_diff:.2%}"
        )
