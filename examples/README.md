# MPECSS Examples

This directory contains example scripts demonstrating MPECSS usage patterns.

## Examples

| File | Description |
|------|-------------|
| `solve_simple.py` | Minimal example: load and solve a single MacMPEC problem |
| `custom_params.py` | How to customize solver parameters |
| `batch_solve.py` | Solve multiple problems in a loop |

## Quick Start

```bash
# From the repository root:
python examples/solve_simple.py
```

## Requirements

- Python 3.10+
- CasADi 3.6+
- MPECSS installed (`pip install -e .`)

## Problem Loaders

MPECSS provides loaders for three benchmark suites:

```python
from mpecss.helpers.loaders.macmpec_loader import load_macmpec
from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib
from mpecss.helpers.loaders.nosbench_loader import load_nosbench

# Load a MacMPEC problem
problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")

# Load an MPECLib problem
problem = load_mpeclib("benchmarks/mpeclib-json/scholtes_ex1.json")
```

## Result Structure

The solver returns a dictionary with:

```python
result = {
    "z_final":       np.ndarray,  # Solution vector
    "f_final":       float,       # Objective value
    "comp_res":      float,       # Complementarity residual
    "status":        str,         # "converged", "max_iter", or "solver_fail"
    "n_outer_iters": int,         # Number of homotopy iterations
    "cpu_time":      float,       # Total solve time (seconds)
    "logs":          list,        # Per-iteration logs
}
```
