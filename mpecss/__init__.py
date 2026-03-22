"""
MPECSS: Smoothed Regularization with Multiplier Check for MPCCs.

This package implements the MPECSS algorithm using CasADi + IPOPT,
with B-stationarity certification (LPEC), feasibility phase (Phase I),
and cascading restoration strategies.

Subpackages:
    phase_1/        - Phase I: feasibility NLP
    phase_2/        - Phase II: main solver loop, multistart, acceleration, restoration
    phase_3/        - BNLP polishing, LPEC refinement, B-stationarity checks
    benchmarks/     - MacMPEC / NOSBENCH loaders, known optima
    helpers/        - Solver wrapper, utilities, B-stationarity, turbo mode
"""
__version__ = '1.0.1'

from mpecss.phase_2.mpecss import run_mpecss  # noqa: F401
