"""
The Toolbox: Helpful tools for logging and math.

This module is like a "utility belt" for the solver. It contains:
1. IterationLog: A "Flight Recorder" that tracks every move the solver makes.
2. extract_multipliers: A tool to "harvest" the mathematical forces acting 
   on our solution so we can check its quality.
3. multiplier_sign_test: A "Quality Seal" check to see if we reached our goal.
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import casadi as ca




@dataclass
class IterationLog:
    """
    The Flight Recorder.

    Every time the solver takes a step, we record everything: 
    where we are, how much progress we made, and whether we 
    ran into any trouble. This is what you see in the .csv 
    reports later!
    """
    iteration: int = 0
    t_k: float = 0.0
    delta_k: float = 0.0
    comp_res: float = float('inf')
    kkt_res: float = float('inf')
    objective: float = float('inf')
    sign_test: str = 'N/A'
    sign_test_reason: str = ''
    restoration_used: str = 'none'
    solver_status: str = ''
    cpu_time: float = 0.0
    n_biactive: int = 0
    t_update_regime: str = ''
    nlp_iter_count: int = 0
    solver_type: str = ''
    warmstart_type: str = 'none'
    sta_tol: float = 0.0
    improvement_ratio: float = 0.0
    restoration_trigger_reason: str = 'none'
    restoration_success: bool = False
    biactive_indices_str: str = ''
    stagnation_count: int = 0
    tracking_count: int = 0
    is_in_tracking_regime: bool = False
    solver_fallback_occurred: bool = False
    consecutive_solver_failures: int = 0
    best_comp_res_so_far: float = float('inf')
    best_iter_achieved: int = -1
    ipopt_tol_used: float = 1e-06
    lambda_G_min: float = 0.0
    lambda_G_max: float = 0.0
    lambda_H_min: float = 0.0
    lambda_H_max: float = 0.0
    z_k: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_G: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_H: Optional[np.ndarray] = field(default=None, repr=False)
    lambda_comp: Optional[np.ndarray] = field(default=None, repr=False)

    def to_row(self) -> dict:
        """Return a CSV-exportable dict (large array fields excluded)."""
        d = asdict(self)
        for key in ('z_k', 'lambda_G', 'lambda_H', 'lambda_comp'):
            d.pop(key, None)
        return d


def extract_multipliers(lam_g, n_comp, problem_info):
    """
    Harvesting the Forces (Multipliers).

    When a computer solves an optimization problem, it doesn't just
    find a point; it also finds the "stresses" or "forces" acting at 
    that point. We slice these forces out of the solver's output so 
    we can use them for our quality checks.
    """
    lam_g = np.asarray(lam_g).flatten()
    n_orig_con  = problem_info.get('n_orig_con', 0)
    n_bounded_G = problem_info.get('n_bounded_G', n_comp)   # default: NCP
    n_ubH       = problem_info.get('n_ubH', 0)

    # Use explicit offsets if available (populated by updated build_casadi)
    if 'off_G_lb' in problem_info and 'off_H_lb' in problem_info and 'off_comp' in problem_info:
        off_G_lb = problem_info['off_G_lb']
        off_H_lb = problem_info['off_H_lb']
        off_comp = problem_info['off_comp']
    else:
        # Legacy fallback: assume standard NCP layout
        off_G_lb = n_orig_con
        off_H_lb = off_G_lb + n_comp
        off_comp = off_H_lb + n_comp

    # lambda_G: multipliers for G >= 0 lower-bound constraints
    # For box-MCP (all G free), n_bounded_G == 0 → lambda_G is all zeros
    if n_bounded_G > 0:
        lambda_G = -lam_g[off_G_lb : off_G_lb + n_bounded_G]
        # Pad to n_comp if needed (bounded subset only)
        if len(lambda_G) < n_comp:
            full_lG = np.zeros(n_comp)
            bounded_idx = problem_info.get('_bounded_G_idx', list(range(n_bounded_G)))
            for k, i in enumerate(bounded_idx):
                if k < len(lambda_G):
                    full_lG[i] = lambda_G[k]
            lambda_G = full_lG
    else:
        lambda_G = np.zeros(n_comp)

    # lambda_H: multipliers for H >= 0 lower-bound constraints (always n_comp)
    lambda_H = -lam_g[off_H_lb : off_H_lb + n_comp]

    # lambda_comp: multipliers for the complementarity product (lower pair)
    lambda_comp = -lam_g[off_comp : off_comp + n_comp]

    return lambda_G, lambda_H, lambda_comp




def multiplier_sign_test(lambda_G, lambda_H, lambda_comp, biactive_idx, tau=1e-6):
    """
    S-stationarity sign check at biactive indices.
    Requires lambda_G[i], lambda_H[i], lambda_comp[i] >= -tau for all i in
    biactive_idx (where both G_i ≈ 0 and H_i ≈ 0).

    Returns
    -------
    passed : bool
    reason : str — empty if passed, diagnostic string if failed
    """
    if len(biactive_idx) == 0:
        return (True, 'no_biactive')
    
    reasons = []
    for i in biactive_idx:
        if lambda_G[i] < -tau:
            reasons.append(f'lam_G[{i}]={lambda_G[i]:.2e}<-{tau:.2e}')
        if lambda_H[i] < -tau:
            reasons.append(f'lam_H[{i}]={lambda_H[i]:.2e}<-{tau:.2e}')
        if lambda_comp[i] < -tau:
            reasons.append(f'lam_comp[{i}]={lambda_comp[i]:.2e}<-{tau:.2e}')
    
    if not reasons:
        return (True, 'PASS')
    
    return (False, 'FAIL: ' + '; '.join(reasons))


def export_csv(logs: List[IterationLog], filepath: str):
    """Export iteration logs to CSV. Creates the output directory if needed."""
    import pandas as pd
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    df = pd.DataFrame([log.to_row() for log in logs])
    df.to_csv(filepath, index=False)


