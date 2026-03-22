"""
Minimal MPECSS example: Load and solve a MacMPEC problem.

This script demonstrates the recommended workflow:
1. Load a problem from the MacMPEC benchmark suite
2. Generate an initial point
3. Run the MPECSS solver
4. Inspect the results
"""
import os
from mpecss import run_mpecss
from mpecss.helpers.loaders.macmpec_loader import load_macmpec


def main():
    # 1. Locate the benchmark data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    benchmark_dir = os.path.join(repo_root, "benchmarks", "macmpec", "macmpec-json")

    # 2. Load a simple problem (bard1 is a classic 5-variable bilevel problem)
    problem_path = os.path.join(benchmark_dir, "bard1.nl.json")
    if not os.path.isfile(problem_path):
        raise FileNotFoundError(
            f"Benchmark file not found: {problem_path}\n"
            "Make sure you're running from the MPECSS repository root."
        )
    problem = load_macmpec(problem_path)

    print(f"Problem: {problem['name']}")
    print(f"  Variables:         {problem['n_x']}")
    print(f"  Complementarities: {problem['n_comp']}")
    print(f"  Constraints:       {problem['n_con']}")

    # 3. Generate an initial point (seed=0 for reproducibility)
    z0 = problem["x0_fn"](seed=0)

    # 4. Solve
    #    For this simple problem, we disable the feasibility phase to go
    #    directly to the homotopy solver. In practice, Phase I helps find
    #    feasible starting points for larger problems.
    print("\nSolving...")
    params = {
        "feasibility_phase": False,
        "eps_tol": 1e-7,  # Slightly relaxed from default 1e-8
    }
    result = run_mpecss(problem, z0, params=params)

    # 5. Print results
    print(f"\nStatus:               {result['status']}")
    print(f"Objective:            {result['f_final']:.6e}")
    print(f"Complementarity res:  {result['comp_res']:.2e}")
    print(f"Outer iterations:     {result['n_outer_iters']}")
    print(f"CPU time:             {result['cpu_time']:.3f} s")

    # The optimal objective for bard1 is 17.0
    if result['status'] == 'converged':
        print("\nSUCCESS: Problem solved to optimality.")
    else:
        print(f"\nWARNING: Solver did not converge ({result['status']})")


if __name__ == "__main__":
    main()
