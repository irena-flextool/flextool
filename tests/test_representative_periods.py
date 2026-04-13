"""Tests for representative period clustering, weights, and FlexTool integration.

Unit tests for the clustering and weight algorithms, plus functional tests
that verify the full pipeline: build model → cluster → solve with RP weights.
"""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from flextool.representative_periods.clustering import greedy_convex_hull_clustering
from flextool.representative_periods.weights import (
    compute_weight_matrix,
    distance_to_hull,
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
# Functional tests: full pipeline
# ---------------------------------------------------------------------------

def _build_test_db(db_path: str, yaml_path: str, seed: int = 42) -> None:
    """Initialize DB, build model, return."""
    from flextool.update_flextool import initialize_database
    from flextool.model_builder.build_model import build_model
    initialize_database(
        "version/flextool_template_master.json", db_path
    )
    build_model(yaml_path, db_url=f"sqlite:///{db_path}", seed=seed)


def _run_flextool(db_path: str, out_path: str, scenario: str) -> float:
    """Run FlexTool and return total_cost."""
    result = subprocess.run(
        [sys.executable, "run_flextool.py",
         f"sqlite:///{db_path}", f"sqlite:///{out_path}",
         "--scenario-name", scenario],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"FlexTool failed:\n{result.stdout}\n{result.stderr}")
    # Parse total_cost from output
    for line in result.stdout.split("\n"):
        if "total_cost.val" in line:
            return float(line.split("=")[1].strip())
    pytest.fail(f"Could not find total_cost in output:\n{result.stdout}")


def _setup_rp_scenario(db_url: str, n_rp: int, period_length: int, storage_nodes: list[str]) -> None:
    """Run RP pre-processing and create a scenario that uses the RP timeset."""
    from spinedb_api import DatabaseMapping, import_data, Map
    from flextool.representative_periods.preprocess import preprocess_representative_periods

    timeset_name = preprocess_representative_periods(db_url, "generated_scenario", n_rp, period_length)

    with DatabaseMapping(db_url) as db:
        rp_alt = "rp_override"
        param_values = [
            ("solve", "generated_solve", "period_timeset",
             Map(["p2025"], [timeset_name]), rp_alt),
        ]
        for node_name in storage_nodes:
            param_values.append(
                ("node", node_name, "storage_binding_method",
                 "bind_using_rp_weights", rp_alt)
            )

        num, log = import_data(db,
            alternatives=[(rp_alt, "RP overrides")],
            scenarios=[("rp_scenario", True, "RP test")],
            scenario_alternatives=[
                ("rp_scenario", "generated"),
                ("rp_scenario", f"hull_{n_rp}rp_{period_length}h"),
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
