"""
Example: Customizing MPECSS solver parameters.

This script shows how to tune solver behavior for different problem types.
"""
import os
from mpecss import run_mpecss
from mpecss.phase_2 import DEFAULT_PARAMS
from mpecss.helpers.loaders.macmpec_loader import load_macmpec


def main():
    # Locate benchmark data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    problem_path = os.path.join(
        repo_root, "benchmarks", "macmpec", "macmpec-json", "bard1.nl.json"
    )

    if not os.path.isfile(problem_path):
        raise FileNotFoundError(f"Benchmark file not found: {problem_path}")

    problem = load_macmpec(problem_path)
    z0 = problem["x0_fn"](seed=0)

    # Print default parameters
    print("Default Parameters:")
    for key, value in DEFAULT_PARAMS.items():
        print(f"  {key}: {value}")

    # Custom parameters for faster convergence (less precision)
    fast_params = {
        "eps_tol": 1e-6,           # Relaxed tolerance (default: 1e-8)
        "max_outer": 100,          # Fewer iterations (default: 3000)
        "feasibility_phase": False,  # Skip Phase I for small problems
    }

    print("\n--- Solving with fast parameters ---")
    result_fast = run_mpecss(problem, z0, params=fast_params)
    print(f"Status: {result_fast['status']}")
    print(f"Objective: {result_fast['f_final']:.6e}")
    print(f"Iterations: {result_fast['n_outer_iters']}")
    print(f"CPU time: {result_fast['cpu_time']:.3f} s")

    # Custom parameters for higher precision
    precise_params = {
        "eps_tol": 1e-10,          # Stricter tolerance
        "tau": 1e-8,               # Stricter sign test tolerance
        "adaptive_t": True,        # Adaptive t-update (default)
        "feasibility_phase": False,
    }

    print("\n--- Solving with precise parameters ---")
    result_precise = run_mpecss(problem, z0, params=precise_params)
    print(f"Status: {result_precise['status']}")
    print(f"Objective: {result_precise['f_final']:.6e}")
    print(f"Iterations: {result_precise['n_outer_iters']}")
    print(f"CPU time: {result_precise['cpu_time']:.3f} s")


if __name__ == "__main__":
    main()
