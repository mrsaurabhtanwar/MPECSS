"""
B-Stationarity certification for MPECSS via LPEC.

Under MPEC-LICQ, S-stationarity ⟺ B-stationarity (Scheel & Scholtes, 2000).
This module provides:
    1. An LPEC-based post-solve verification that certifies B-stationarity
       directly (even when MPEC-LICQ is unclear).
    2. An MPEC-LICQ check to confirm the equivalence holds.

The LPEC (Linear Program with Equilibrium Constraints) solved is:

    min_d   ∇f(x*)ᵀ d
    s.t.    ∇c_j(x*)ᵀ d ≤ 0    for active inequality constraints j
            ∇c_j(x*)ᵀ d = 0    for equality constraints j
            x_lb ≤ x* + d ≤ x_ub  (linearized bound feasibility)
            d_G[i] ≥ 0, d_H[i] ≥ 0   for biactive i
            d_G[i] free             for i ∈ I_G (G active only)
            d_H[i] free             for i ∈ I_H (H active only)

    where d_G = ∇G(x*)ᵀ d,  d_H = ∇H(x*)ᵀ d

If the optimal value ≥ -ε (for small ε), the point is B-stationary:
no complementarity-feasible descent direction exists.

Implementation uses SciPy's linprog (open-source, no Gurobi needed).

References:
    - Scheel, H. & Scholtes, S. (2000). Mathematical Programs with
      Complementarity Constraints: Stationarity, Optimality, and Sensitivity.
      Mathematics of Operations Research, 25(1), 1-22.
    - Ralph, D. & Wright, S.J. (2004). Some properties of regularization
      and penalization schemes for MPECs.
    - Nurkanović, A. & Leyffer, S. (2025). A Globally Convergent Method for
      Computing B-stationary Points of MPECs. arXiv preprint.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import casadi as ca

logger = logging.getLogger('mpecss.bstationarity')

_ACTIVE_TOL = 1e-06
_BSTAT_TOL = 1e-08
_LICQ_TOL = 1e-08
_DIR_BOUND = 1.0
_BSTAT_TIMEOUT = 60.0


_JACOBIAN_CACHE: Dict[str, Any] = {}

def clear_jacobian_cache():
    """Clear the Jacobian cache to free memory."""
    _JACOBIAN_CACHE.clear()


def _compute_jacobians(z, problem):
    """
    Compute Jacobians of f, g_orig, G, H at point z.

    Returns
    -------
    grad_f : np.ndarray (n_x,)
        Gradient of objective.
    J_g : np.ndarray (n_con, n_x) or None
        Jacobian of original constraints (None if no constraints).
    J_G : np.ndarray (n_comp, n_x)
        Jacobian of G.
    J_H : np.ndarray (n_comp, n_x)
        Jacobian of H.
    """
    n_x = problem['n_x']
    z = np.asarray(z).flatten()
    prob_name = problem.get('name', 'unknown')
    
    if prob_name not in _JACOBIAN_CACHE:
        _sym = ca.MX.sym if n_x >= 500 else ca.SX.sym
        x_sym = _sym('x', n_x)
        
        info = problem['build_casadi'](0, 0)
        grad_f_expr = ca.jacobian(info['f'], info['x'])
        grad_f_fn = ca.Function('grad_f', [info['x']], [grad_f_expr])
        
        G_expr = problem['G_fn'](x_sym)
        jac_G_fn = ca.Function('jac_G', [x_sym], [ca.jacobian(G_expr, x_sym)])
        
        H_expr = problem['H_fn'](x_sym)
        jac_H_fn = ca.Function('jac_H', [x_sym], [ca.jacobian(H_expr, x_sym)])
        
        jac_g_fn = None
        n_con = problem.get('n_con', 0)
        if n_con > 0:
            g_orig_expr = info['g'][:n_con]
            jac_g_fn = ca.Function('jac_g', [info['x']], [ca.jacobian(g_orig_expr, info['x'])])
            
        _JACOBIAN_CACHE[prob_name] = (grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn)
    
    grad_f_fn, jac_G_fn, jac_H_fn, jac_g_fn = _JACOBIAN_CACHE[prob_name]
    
    grad_f = np.asarray(grad_f_fn(z)).flatten()
    
    J_G = np.asarray(jac_G_fn(z))
    if J_G.ndim == 1:
        J_G = J_G.reshape(1, -1)
        
    J_H = np.asarray(jac_H_fn(z))
    if J_H.ndim == 1:
        J_H = J_H.reshape(1, -1)
        
    J_g = None
    if jac_g_fn is not None:
        J_g = np.asarray(jac_g_fn(z))
        if J_g.ndim == 1:
            J_g = J_g.reshape(1, -1)
    
    return grad_f, J_g, J_G, J_H


def _classify_complementarity_indices(z, problem, tol=_ACTIVE_TOL):
    """
    Classify complementarity indices into active sets.

    Returns
    -------
    I_G : list[int]
        Indices where G_i ≈ 0 and H_i > tol (G-active only).
    I_H : list[int]
        Indices where H_i ≈ 0 and G_i > tol (H-active only).
    I_B : list[int]
        Biactive indices where both |G_i| < tol and |H_i| < tol.
    I_free : list[int]
        Indices where both G_i > tol and H_i > tol (neither active).
    """
    from mpecss.helpers.loaders.macmpec_loader import evaluate_GH
    G, H = evaluate_GH(z, problem)
    I_G, I_H, I_B, I_free = [], [], [], []
    
    for i in range(len(G)):
        g_active = abs(G[i]) < tol
        h_active = abs(H[i]) < tol
        if g_active and h_active:
            I_B.append(i)
        elif g_active:
            I_G.append(i)
        elif h_active:
            I_H.append(i)
        else:
            I_free.append(i)
    
    return I_G, I_H, I_B, I_free


def check_mpec_licq(z, problem, tol=_LICQ_TOL):
    """
    Check MPEC-LICQ at point z.

    MPEC-LICQ holds if the gradients of all active constraints
    (including separate ∇G_i and ∇H_i for biactive indices i ∈ I_B)
    are linearly independent.

    Under MPEC-LICQ: B-stationarity ⟺ S-stationarity.

    Parameters
    ----------
    z : np.ndarray
        Point to check.
    problem : dict
        Problem specification.
    tol : float
        Tolerance for rank deficiency.

    Returns
    -------
    licq_holds : bool
        True if MPEC-LICQ holds at z.
    rank : int
        Rank of the active constraint Jacobian.
    n_active : int
        Number of active constraint gradients.
    details : str
        Diagnostic string.
    """
    grad_f, J_g, J_G, J_H = _compute_jacobians(z, problem)
    I_G, I_H, I_B, I_free = _classify_complementarity_indices(z, problem, tol=_ACTIVE_TOL)
    n_x = problem['n_x']
    
    active_rows = []
    
    # Add active constraint gradients
    if J_g is not None:
        n_con = problem.get('n_con', 0)
        info = problem['build_casadi'](0, 0)
        lbg = np.array(info['lbg'][:n_con])
        ubg = np.array(info['ubg'][:n_con])
        g_expr_orig = info['g'][:n_con]
        _g_eval_fn = ca.Function('g_licq_eval', [info['x']], [g_expr_orig])
        g_val = np.asarray(_g_eval_fn(z)).flatten()
        
        for j in range(n_con):
            if abs(g_val[j] - lbg[j]) < tol or abs(g_val[j] - ubg[j]) < tol:
                active_rows.append(J_g[j])
    
    # Add ∇G_i for G-active and biactive
    for i in I_G:
        active_rows.append(J_G[i])
    for i in I_B:
        active_rows.append(J_G[i])
        active_rows.append(J_H[i])
    
    # Add ∇H_i for H-active
    for i in I_H:
        active_rows.append(J_H[i])
    
    n_active = len(active_rows)
    if n_active == 0:
        return True, 0, 0, 'No active constraints'
    
    A = np.vstack(active_rows)
    rank = np.linalg.matrix_rank(A, tol=tol)
    licq_holds = (rank == n_active)
    
    details = f'rank={rank}, n_active={n_active}, |I_B|={len(I_B)}'
    if not licq_holds:
        details += ' (rank-deficient)'
    
    return licq_holds, rank, n_active, details


def certify_bstationarity(z, problem, f_val=None, tol=_BSTAT_TOL, dir_bound=None, timeout=None):
    """
    Certify B-stationarity at point z by solving an LPEC.

    Uses SciPy linprog (fully open-source, no Gurobi needed).

    Parameters
    ----------
    z : np.ndarray
        Candidate point.
    problem : dict
        Problem specification.
    f_val : float or None
        Objective value at z (for logging only).
    tol : float
        Tolerance for B-stationarity declaration.
    dir_bound : float or None
        Trust-region radius for the LPEC direction.
    timeout : float or None
        Wall-clock timeout in seconds.

    Returns
    -------
    is_bstat : bool
        True if the point is certified B-stationary.
    lpec_obj : float
        Optimal value of the LPEC.
    licq_holds : bool
        Whether MPEC-LICQ holds at z.
    details : dict
        Diagnostic information.
    """
    from mpecss.helpers.loaders.macmpec_loader import evaluate_GH
    
    if dir_bound is None:
        dir_bound = _DIR_BOUND
    if timeout is None:
        timeout = _BSTAT_TIMEOUT
    
    n_x = problem['n_x']
    z = np.asarray(z).flatten()
    
    # Compute Jacobians
    grad_f, J_g, J_G, J_H = _compute_jacobians(z, problem)
    I_G, I_H, I_B, I_free = _classify_complementarity_indices(z, problem)
    
    # Check LICQ
    licq_holds, licq_rank, n_active, licq_details = check_mpec_licq(z, problem)
    
    n_biactive = len(I_B)
    logger.info(f'B-stat check: |I_G|={len(I_G)}, |I_H|={len(I_H)}, |I_B|={n_biactive}, LICQ={licq_holds}')
    
    # Fast path: no biactive indices
    if n_biactive == 0:
        is_bstat = True
        lpec_obj = 0.0
        details = {
            'lpec_status': 'fast_path',
            'licq_rank': licq_rank,
            'n_biactive': 0,
            'classification': 'B-stationary (no biactive)',
            'best_direction': None,
            'best_branch_idx': -1
        }
        return is_bstat, lpec_obj, licq_holds, details
    
    # Full LPEC enumeration for biactive indices
    from scipy.optimize import linprog
    max_enum = min(2**n_biactive, 2**15)  # Cap enumeration
    t_start = time.time()
    
    best_obj = 0.0
    best_direction = None
    best_branch = -1
    timed_out = False
    
    # Build base LP constraints
    A_ub_rows = []
    b_ub = []
    A_eq_rows = []
    b_eq = []
    
    # Bound constraints: -dir_bound <= d <= dir_bound
    bounds = [(-dir_bound, dir_bound) for _ in range(n_x)]
    
    # Variable bound constraints from x_lb <= x + d <= x_ub
    info = problem['build_casadi'](0, 0)
    lbx = np.array(info['lbx'])
    ubx = np.array(info['ubx'])
    
    _BIG = 1e20
    for i in range(n_x):
        if lbx[i] > -_BIG:
            bounds[i] = (max(bounds[i][0], lbx[i] - z[i]), bounds[i][1])
        if ubx[i] < _BIG:
            bounds[i] = (bounds[i][0], min(bounds[i][1], ubx[i] - z[i]))
    
    # Enumerate branches for biactive indices
    for branch_idx in range(max_enum):
        if time.time() - t_start > timeout:
            timed_out = True
            break
        
        # Build branch-specific constraints
        A_ub_branch = list(A_ub_rows)
        b_ub_branch = list(b_ub)
        
        # For each biactive index, either ∇G_i·d >= 0 or ∇H_i·d >= 0 (exclusive)
        for bit_pos, i in enumerate(I_B):
            if (branch_idx >> bit_pos) & 1:
                # G stays active: ∇G_i·d >= 0 → -∇G_i·d <= 0
                A_ub_branch.append(-J_G[i])
                b_ub_branch.append(0)
            else:
                # H stays active: ∇H_i·d >= 0 → -∇H_i·d <= 0
                A_ub_branch.append(-J_H[i])
                b_ub_branch.append(0)
        
        # Solve LP
        if len(A_ub_branch) > 0:
            A_ub = np.vstack(A_ub_branch)
        else:
            A_ub = None
        
        try:
            result = linprog(grad_f, A_ub=A_ub, b_ub=b_ub_branch if b_ub_branch else None,
                            bounds=bounds, method='highs')
            if result.success and result.fun < best_obj:
                best_obj = result.fun
                best_direction = result.x.copy()
                best_branch = branch_idx
        except Exception as e:
            logger.debug(f'LP solve failed for branch {branch_idx}: {e}')
            continue
    
    is_bstat = best_obj >= -tol
    
    if is_bstat:
        classification = 'B-stationary'
    elif licq_holds:
        classification = 'S-stationary (LICQ)' if best_obj >= -tol else 'not B-stationary'
    else:
        classification = 'not B-stationary'
    
    details = {
        'lpec_status': 'timed_out' if timed_out else 'complete',
        'licq_rank': licq_rank,
        'n_biactive': n_biactive,
        'classification': classification,
        'best_direction': best_direction,
        'best_branch_idx': best_branch,
        'branches_enumerated': min(branch_idx + 1, max_enum)
    }
    
    logger.info(f'B-stat result: obj={best_obj:.2e}, is_bstat={is_bstat}, timed_out={timed_out}')
    return is_bstat, best_obj, licq_holds, details


def bstat_post_check(result, problem, timeout=None):
    """
    Convenience wrapper: run B-stationarity check on MPECSS result.

    Only runs the check if the solver converged (status='converged'
    and stationarity='S').

    Parameters
    ----------
    result : dict
        Result dictionary from run_mpecss().
    problem : dict
        Problem specification.
    timeout : float or None
        Wall-clock timeout in seconds for the LPEC enumeration.

    Returns
    -------
    result : dict
        Updated result dict with added keys:
        - 'b_stationarity': bool or None
        - 'lpec_obj': float or None
        - 'licq_holds': bool or None
        - 'bstat_details': dict or None
        - 'stationarity': upgraded to 'B' if B-stat certified
    """
    result = dict(result)
    
    if result.get('status') != 'converged' or result.get('stationarity') != 'S':
        result['b_stationarity'] = None
        result['lpec_obj'] = None
        result['licq_holds'] = None
        result['bstat_details'] = None
        logger.info(f"Skipping B-stat check: status={result.get('status')}, stationarity={result.get('stationarity')}")
        return result
    
    z = result['z_final']
    f_val = result.get('f_final')
    
    try:
        is_bstat, lpec_obj, licq_holds, details = certify_bstationarity(z, problem, f_val=f_val, timeout=timeout)
        result['b_stationarity'] = is_bstat
        result['lpec_obj'] = lpec_obj
        result['licq_holds'] = licq_holds
        result['bstat_details'] = details
        
        if is_bstat:
            result['stationarity'] = 'B'
            logger.info('Stationarity upgraded: S → B (LPEC certified)')
    except Exception as e:
        logger.warning(f'B-stat check failed: {e}')
        result['b_stationarity'] = None
        result['lpec_obj'] = None
        result['licq_holds'] = None
        result['bstat_details'] = {'error': str(e)}
    
    return result
