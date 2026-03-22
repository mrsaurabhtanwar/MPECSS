"""
Main entry point for the MPECSS solver stack.
Re-exports from solver_cache and solver_ipopt to keep
backward-compatible import paths for the rest of the codebase.

Constraint layout (fixed order produced by problems.build_casadi):
    g = [ g_orig(x)      (n_orig_con)   original constraints
          G(x) + delta   (n_comp)       shifted complementarity G
          H(x) + delta   (n_comp)       shifted complementarity H
          G(x)*H(x) - t  (n_comp)       relaxed complementarity product ]
"""

import casadi as ca
from mpecss.helpers.solver_cache import clear_solver_cache, build_problem
from mpecss.helpers.solver_ipopt import (
    solve_smooth_subproblem,
    is_solver_success,
    solve_with_solver_fallback,
    DEFAULT_IPOPT_OPTS,
)

def build_universal_nlp_solver(name, n_x, nlp, ipopt_opts=None):
    """
    Universal open-source NLP builder for all MPEC-SS phases.
    Builds IPOPT + MUMPS unless otherwise specified.
    """
    if ipopt_opts is None:
        ipopt_opts = dict(DEFAULT_IPOPT_OPTS)
        
    # Default IPOPT formulation
    return ca.nlpsol(name, 'ipopt', nlp, {'ipopt': ipopt_opts, 'print_time': False, 'error_on_fail': False})
