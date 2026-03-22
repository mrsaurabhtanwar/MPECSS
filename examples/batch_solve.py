"""
Example: Batch solve multiple MacMPEC problems.

This script demonstrates solving multiple problems in a loop and collecting
results for analysis.
"""
import os
from mpecss import run_mpecss
from mpecss.helpers.loaders.macmpec_loader import load_macmpec_batch


def main():
    # Locate benchmark data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    benchmark_dir = os.path.join(repo_root, "benchmarks", "macmpec", "macmpec-json")

    if not os.path.isdir(benchmark_dir):
        raise FileNotFoundError(f"Benchmark directory not found: {benchmark_dir}")

    # Load a subset of problems (bard* family)
    problems = load_macmpec_batch(benchmark_dir, pattern="bard*.nl.json")
    print(f"Loaded {len(problems)} problems")

    # Solver parameters
    params = {
        "feasibility_phase": False,
        "eps_tol": 1e-7,
        "max_outer": 200,
    }

    # Solve all problems
    results = []
    print("\n" + "-" * 60)
    print(f"{'Problem':<15} {'Status':<12} {'Objective':>12} {'Comp Res':>10} {'Time':>8}")
    print("-" * 60)

    for problem in problems:
        z0 = problem["x0_fn"](seed=0)
        result = run_mpecss(problem, z0, params=params)

        results.append({
            "name": problem["name"],
            "status": result["status"],
            "objective": result["f_final"],
            "comp_res": result["comp_res"],
            "cpu_time": result["cpu_time"],
        })

        print(
            f"{problem['name']:<15} "
            f"{result['status']:<12} "
            f"{result['f_final']:>12.4e} "
            f"{result['comp_res']:>10.2e} "
            f"{result['cpu_time']:>7.2f}s"
        )

    print("-" * 60)

    # Summary statistics
    n_converged = sum(1 for r in results if r["status"] == "converged")
    total_time = sum(r["cpu_time"] for r in results)
    print(f"\nConverged: {n_converged}/{len(results)}")
    print(f"Total time: {total_time:.2f} s")


if __name__ == "__main__":
    main()
