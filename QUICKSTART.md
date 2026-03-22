# Quick Start

Get MPECSS running in 5 minutes.

## 1. Install

```bash
pip install casadi numpy
pip install -e .
```

## 2. Verify Installation

```bash
python -c "from mpecss import run_mpecss; print('OK')"
```

## 3. Run Your First Solve

```python
from mpecss import run_mpecss
from mpecss.helpers.loaders.macmpec_loader import load_macmpec

# Load a benchmark problem
problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")

# Generate initial point
z0 = problem["x0_fn"](seed=0)

# Solve
result = run_mpecss(problem, z0)

print(f"Status: {result['status']}")
print(f"Objective: {result['f_final']:.6f}")
```

Expected output:
```
Status: converged
Objective: 17.000000
```

## 4. Run Tests

```bash
pytest tests/ -v
```

## 5. Next Steps

- See `examples/` for more usage patterns
- Read the main [README.md](README.md) for algorithm details
- Check `mpecss/phase_2/mpecss.py` for solver parameters

## Troubleshooting

**ImportError: No module named 'casadi'**
```bash
pip install casadi
```

**FileNotFoundError: Benchmark file not found**
```bash
# Run from the repository root directory
cd /path/to/MPECSS
python examples/solve_simple.py
```

**Solver returns status='max_iter'**
- Try increasing `max_outer` parameter
- Or use a different initial point with `seed=1, 2, ...`
