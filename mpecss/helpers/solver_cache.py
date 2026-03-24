"""
The "Memory Bank": Saving time by remembering solvers.

Building a math solver can be slow. This module acts as a 
filing cabinet — it saves the "blueprint" (template) of a 
problem so we don't have to rebuild it every single time 
the difficulty (t_k) changes.
"""

import math
import sys
import logging
from typing import Dict, Any
import casadi as ca

logger = logging.getLogger('mpecss.solver.cache')

# Module-level caches (populated at runtime)
_TEMPLATE_CACHE: Dict[str, Any] = {}
_SOLVER_CACHE: Dict[str, Any] = {}
_INFO_CACHE: Dict[str, Any] = {}
_PARAMETRIC_CACHE: Dict[str, Any] = {}


def clear_solver_cache():
    """Clear all caches and run GC. Call between problems to free memory."""
    _TEMPLATE_CACHE.clear()
    _SOLVER_CACHE.clear()
    _INFO_CACHE.clear()
    _PARAMETRIC_CACHE.clear()
    import gc
    gc.collect()


def _evict_problem_from_cache(prob_name):
    """Remove all concrete and parametric solver entries for prob_name."""
    # Remove entries matching this problem from all caches
    for cache in (_SOLVER_CACHE, _PARAMETRIC_CACHE):
        keys_to_remove = [k for k in cache if k.startswith(f'{prob_name}|')]
        for k in keys_to_remove:
            del cache[k]


def _get_template(problem, smoothing='product'):
    """
    Step 1: "The Master Blueprint."

    This is where we build the core math structure of the problem one 
    time. We leave "placeholders" for the difficulty level (t) and 
    shift (delta), so we can reuse this same blueprint for the whole 
    homotopy process.
    """
    prob_name = problem.get('name', 'unknown')
    n_x = problem.get('n_x', 0)
    ckey = f'{prob_name}|{smoothing}'
    if ckey not in _TEMPLATE_CACHE:
        _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
        t_sym = _sym('t_param')
        d_sym = _sym('d_param')
        info_sym = problem['build_casadi'](t_sym, d_sym, smoothing=smoothing)
        info_sym['t_sym'] = t_sym
        info_sym['d_sym'] = d_sym
        _TEMPLATE_CACHE[ckey] = (t_sym, d_sym, info_sym)
    return _TEMPLATE_CACHE[ckey]


def build_problem(problem, t_k, delta_k, smoothing='product'):
    """Return the info dict (bounds + CasADi expressions) for problem.
    
    NOTE: This function previously cached by (prob_name, smoothing) ignoring t_k/delta_k,
    which was semantically incorrect. The cache has been removed since _get_template()
    already handles the expensive symbolic compilation. Rebuilding the info dict is
    cheap (just bound arrays and references to the template).
    """
    # Always rebuild - the template handles the expensive symbolic compilation
    return problem['build_casadi'](t_k, delta_k, smoothing=smoothing)


def _t_round(t):
    """Round t/delta to 4 significant figures for stable cache keys."""
    if t == 0:
        return 0
    mag = math.floor(math.log10(abs(t)))
    return round(t, -mag + 3)


def _tol_bucket(tol):
    """Round IPOPT tol to the nearest power of 10 for cache key stability."""
    if tol <= 0:
        return 1e-08
    exp = math.floor(math.log10(tol + sys.float_info.min))
    return 10 ** exp


def _cache_key(problem_name, n_x, tol_bucket):
    """Composite cache key (retained for external callers)."""
    return f'{problem_name}|{n_x}|{tol_bucket}'
