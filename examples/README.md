# MPECSS Examples

This folder contains simple scripts to help you get started with the MPECSS solver.

## Available Examples

| File | Description |
| :--- | :--- |
| **`solve_simple.py`** | The best place to start. Loads one problem and solves it. |
| **`custom_params.py`** | Shows how to change solver settings like accuracy or speed. |
| **`batch_solve.py`** | Demonstrates how to solve a list of problems in a single run. |

## Quick Start

To run an example, open your terminal in the project root and type:

```bash
python examples/solve_simple.py
```

## Understanding the Solver Output

When you run `run_mpecss(problem)`, it returns a dictionary with these important pieces of information:

- **`z_final`**: The final answer (the values the solver found).
- **`f_final`**: The final "score" or cost (smaller is usually better).
- **`status`**: Tells you if it worked (`"converged"`) or failed (`"solver_fail"`).
- **`stationarity`**: The quality of the solution (`"S"` for perfect, `"B"` for reliable, or `"FAIL"`).
- **`cpu_time`**: How many seconds the solver took to find the answer.

## How to Load Your Own Problems

MPECSS makes it easy to load problems from different researchers:

```python
from mpecss.helpers.loaders.macmpec_loader import load_macmpec
from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib
from mpecss.helpers.loaders.nosbench_loader import load_nosbench

# Example: Load a MacMPEC problem
problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")
```

---
For more details on the math behind the solver, check the [Full README](../README.md).
