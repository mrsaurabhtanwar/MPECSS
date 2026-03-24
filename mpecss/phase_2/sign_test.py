"""
Stationarity Testing: Measuring the "Quality" of our Answer.

How do we know if we found the best possible answer? We use a series 
of mathematical tests called "Stationarity Checks."

This module does the "heavy lifting" for these checks:
1. It identifies "Biactive" points (where the problem is most complex).
2. It checks the "multipliers" (the forces acting on the solution)
   to see if they all point in the right directions (the "Sign Test").
"""

from typing import Any, Dict, Tuple, cast
import numpy as np
from mpecss.helpers.loaders.mpeclib_loader import biactive_indices, complementarity_residual
from mpecss.helpers.utils import extract_multipliers, multiplier_sign_test


def evaluate_iteration_stationarity(z_k, lam_g, problem, problem_info, n_comp, t_k, sta_tol, tau, biactive_tol_floor=1e-8):
    """
    Step-by-Step Quality Check:
    1. Extract the "Forces" (multipliers) acting on the solution.
    2. Find the "Intersections" (biactive indices) where the problem is sharp.
    3. Run the "Sign Test" to see if we reached the Gold Standard (S-stationarity).
    """
    # Auto-compute sta_tol if not provided
    if sta_tol is None:
        sta_tol = max(biactive_tol_floor, np.sqrt(t_k))
    
    # Extract multipliers in MPCC convention
    lambda_G, lambda_H, lambda_comp = cast(
        Tuple[np.ndarray, np.ndarray, np.ndarray],
        extract_multipliers(lam_g, n_comp, problem_info)
    )
    
    # Detect biactive indices where both G ≈ 0 and H ≈ 0
    biactive_idx = biactive_indices(z_k, problem, sta_tol)
    
    # Compute complementarity residual
    comp_res = complementarity_residual(z_k, problem)
    
    # Run the sign test
    sign_pass, sign_reason = cast(
        Tuple[bool, str],
        multiplier_sign_test(lambda_G, lambda_H, lambda_comp, biactive_idx, tau=tau)
    )
    
    return {
        'lambda_G': lambda_G,
        'lambda_H': lambda_H,
        'lambda_comp': lambda_comp,
        'sta_tol': sta_tol,
        'biactive_idx': biactive_idx,
        'n_biactive': len(biactive_idx),
        'comp_res': comp_res,
        'sign_pass': sign_pass,
        'sign_reason': sign_reason,
    }


def evaluate_restoration_sign(z_candidate, problem, lambda_G, lambda_H, lambda_comp, sta_tol, tau):
    """
    Re-evaluate sign test for a restoration candidate.
    """
    # Get new biactive indices at the candidate point
    biactive_idx = biactive_indices(z_candidate, problem, sta_tol)
    
    # Run sign test with the provided multipliers
    sign_pass, sign_reason = cast(
        Tuple[bool, str],
        multiplier_sign_test(lambda_G, lambda_H, lambda_comp, biactive_idx, tau=tau)
    )
    
    return {
        'biactive_idx': biactive_idx,
        'n_biactive': len(biactive_idx),
        'sign_pass': sign_pass,
        'sign_reason': sign_reason,
    }
