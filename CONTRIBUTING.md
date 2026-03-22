# Contributing to MPECSS

Thank you for your interest in contributing to MPECSS!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/mpecss.git
   cd mpecss
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Keep lines under 100 characters
- Use descriptive variable names

## Testing

- Add tests for new functionality in `tests/`
- Ensure all tests pass before submitting a PR
- For solver changes, verify benchmark convergence rates

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add or update tests as needed
4. Update documentation if applicable
5. Submit a PR with a clear description

## Reporting Issues

- Check existing issues first
- Include Python version, OS, and CasADi version
- Provide a minimal reproducible example
- Include the full error traceback

## Project Structure

```
mpecss/
  __init__.py          # Main entry point (run_mpecss)
  phase_1/             # Feasibility NLP
  phase_2/             # Main solver loop
  phase_3/             # Post-processing (BNLP polish, B-stationarity)
  helpers/             # Solver wrappers, utilities, loaders
benchmarks/            # MacMPEC, MPECLib, NOSBENCH data
examples/              # Usage examples
tests/                 # Test suite
```

## Questions?

Open an issue or reach out to the maintainers.
