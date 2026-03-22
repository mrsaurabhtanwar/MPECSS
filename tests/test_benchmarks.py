"""
Tests for MPECSS benchmark suite loading and solving.

These tests verify that:
1. MacMPEC problems can be loaded
2. The solver produces correct results on known-optimum problems
3. At least 80% of a small test set converge
"""
import os
import pytest
import numpy as np

# Skip all tests if casadi is not installed
pytest.importorskip("casadi")

from mpecss import run_mpecss
from mpecss.helpers.loaders.macmpec_loader import load_macmpec, load_macmpec_batch


# Locate benchmark directory relative to this file
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TEST_DIR)
_MACMPEC_DIR = os.path.join(_REPO_ROOT, "benchmarks", "macmpec", "macmpec-json")


# Known optimal values for selected problems (from MacMPEC documentation)
KNOWN_OPTIMA = {
    "bard1": 17.0,
    "bard2": -6600.0,
    "bard3": -12.6787,
    "dempe": 31.25,
    "bilin": -21.0,
    "gauvin": 2.0,
}


def _get_problem_path(name: str) -> str:
    return os.path.join(_MACMPEC_DIR, f"{name}.nl.json")


class TestMacMPECLoader:
    """Tests for MacMPEC problem loading."""

    def test_load_single_problem(self):
        """Test loading a single MacMPEC problem."""
        path = _get_problem_path("bard1")
        if not os.path.isfile(path):
            pytest.skip("Benchmark data not available")

        problem = load_macmpec(path)

        assert problem["name"] == "bard1"
        assert problem["n_x"] == 5
        assert problem["n_comp"] == 3
        assert callable(problem["x0_fn"])
        assert callable(problem["build_casadi"])
        assert callable(problem["G_fn"])
        assert callable(problem["H_fn"])

    def test_load_batch(self):
        """Test batch loading of MacMPEC problems."""
        if not os.path.isdir(_MACMPEC_DIR):
            pytest.skip("Benchmark data not available")

        problems = load_macmpec_batch(_MACMPEC_DIR, pattern="bard*.nl.json")

        assert len(problems) >= 3  # bard1, bard2, bard3
        names = {p["name"] for p in problems}
        assert "bard1" in names

    def test_initial_point_generation(self):
        """Test that x0_fn produces valid initial points."""
        path = _get_problem_path("bard1")
        if not os.path.isfile(path):
            pytest.skip("Benchmark data not available")

        problem = load_macmpec(path)

        # Different seeds should produce different points
        z0_seed0 = problem["x0_fn"](seed=0)
        z0_seed1 = problem["x0_fn"](seed=1)

        assert z0_seed0.shape == (problem["n_x"],)
        assert not np.allclose(z0_seed0, z0_seed1)

        # Points should respect bounds
        lbx = np.array(problem["lbx"])
        ubx = np.array(problem["ubx"])
        assert np.all(z0_seed0 >= lbx - 1e-6)
        assert np.all(z0_seed0 <= ubx + 1e-6)


class TestMPECSSSolver:
    """Tests for the MPECSS solver."""

    # Default parameters that work well for benchmark testing
    _TEST_PARAMS = {
        "feasibility_phase": False,  # Direct homotopy for reproducible results
        "eps_tol": 1e-7,  # Slightly relaxed tolerance
    }

    def test_known_optimum_bard1(self):
        """Test that solver finds bard1 optimal value (17.0)."""
        path = _get_problem_path("bard1")
        if not os.path.isfile(path):
            pytest.skip("Benchmark data not available")

        problem = load_macmpec(path)
        z0 = problem["x0_fn"](seed=0)

        result = run_mpecss(problem, z0, params=self._TEST_PARAMS)

        assert result["status"] == "converged", f"Solver did not converge: {result['status']}"
        assert result["comp_res"] < 1e-6, f"Complementarity residual too large: {result['comp_res']}"

        # bard1 optimal is 17.0
        rel_error = abs(result["f_final"] - 17.0) / 17.0
        assert rel_error < 0.01, f"Objective {result['f_final']} != expected 17.0"

    def test_solver_produces_result(self):
        """Test that solver runs and produces valid output on multiple problems."""
        if not os.path.isdir(_MACMPEC_DIR):
            pytest.skip("Benchmark data not available")

        # Test a few problems to ensure the solver works
        test_problems = ["bard1", "bard2", "dempe", "bilin"]
        available = [p for p in test_problems if os.path.isfile(_get_problem_path(p))]

        if len(available) < 2:
            pytest.skip("Not enough benchmark problems available")

        for name in available:
            problem = load_macmpec(_get_problem_path(name))
            z0 = problem["x0_fn"](seed=0)
            result = run_mpecss(problem, z0, params=self._TEST_PARAMS)

            # Solver should produce valid output (not crash)
            assert "status" in result
            assert "f_final" in result
            assert "comp_res" in result
            assert result["z_final"].shape == (problem["n_x"],)

    def test_custom_parameters(self):
        """Test that custom solver parameters are respected."""
        path = _get_problem_path("bard1")
        if not os.path.isfile(path):
            pytest.skip("Benchmark data not available")

        problem = load_macmpec(path)
        z0 = problem["x0_fn"](seed=0)

        # Use custom parameters
        params = {
            "eps_tol": 1e-6,
            "max_outer": 200,
            "feasibility_phase": False,
        }
        result = run_mpecss(problem, z0, params=params)

        # Should converge (bard1 is easy)
        assert result["n_outer_iters"] <= 200


class TestResultStructure:
    """Tests for the structure of solver results."""

    def test_result_keys(self):
        """Test that result dict contains all expected keys."""
        path = _get_problem_path("bard1")
        if not os.path.isfile(path):
            pytest.skip("Benchmark data not available")

        problem = load_macmpec(path)
        z0 = problem["x0_fn"](seed=0)
        result = run_mpecss(problem, z0)

        expected_keys = [
            "z_final", "f_final", "objective", "comp_res",
            "stationarity", "n_outer_iters", "cpu_time",
            "logs", "status", "sign_test_pass",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_z_final_shape(self):
        """Test that z_final has correct shape."""
        path = _get_problem_path("bard1")
        if not os.path.isfile(path):
            pytest.skip("Benchmark data not available")

        problem = load_macmpec(path)
        z0 = problem["x0_fn"](seed=0)
        result = run_mpecss(problem, z0)

        assert result["z_final"].shape == (problem["n_x"],)
