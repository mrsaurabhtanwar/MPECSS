# MPECSS Quick Start Guide

Get up and running with MPECSS in less than 5 minutes.

## 1. Fast Setup (1 minute)

The easiest way to use MPECSS is via `pip`. Open your terminal and run:

```bash
# Create a folder for your work
mkdir my-mpec-work && cd my-mpec-work

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install MPECSS
pip install mpecss
```

## 2. Preflight Check (30 seconds)

Make sure your computer has everything it needs to run the math:

```bash
mpecss-preflight
```

This will check your Python version, memory, and solver availability.

## 3. Solve Your First Problem (1 minute)

Create a file named `solve_one.py` and paste this code:

```python
from mpecss.helpers.loaders.macmpec_loader import load_macmpec
from mpecss.phase_2.mpecss import run_mpecss

# 1. Load a benchmark problem
# (Assumes you have extracted benchmarks.zip in your folder)
problem = load_macmpec("benchmarks/macmpec/macmpec-json/dempe.nl.json")

# 2. Start the solver
result = run_mpecss(problem)

# 3. View the result
print(f"Status: {result['status']}")
print(f"Objective Value: {result['f_final']:.4f}")
```

## 4. Run Benchmarks (2 minutes)

If you have extracted the `benchmarks.zip` data, you can run hundreds of tests with a single command:

```bash
# Run 191 MacMPEC problems using 4 cores
mpecss-macmpec --workers 4
```

Check your results in the `results/` folder!

## 5. What do the results mean?

When the solver finishes, it will give you one of these statuses:

- **S-stationary** ✅: Found the "Gold Standard" solution.
- **B-stationary** ✅: Found a mathematically solid solution.
- **Failed** ❌: The problem was extremely difficult; try a different starting point.

## Troubleshooting

- **`CasADi not found`**: Ensure your virtual environment is active.
- **`Memory Error`**: If your computer slows down, run benchmarks with fewer workers (e.g., `--workers 1`).

---
For more details, see the [Full README](README.md).
