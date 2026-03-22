"""
Restoration heuristics for MPECSS.

Three restoration strategies for function-based G(x)/H(x):
    1. random_perturb: Jacobian-guided perturbation of biactive variables
    2. quadratic_regularizer: penalty term pulling biactive G, H apart
    3. directional_escape: move along sign-based direction using Jacobians

Each function returns a new initial point z_new to be used for warm-starting
the next subproblem solve (with the same t_k).
"""

import logging
import time
from typing import List, Optional, Dict, Any
import numpy as np
import casadi as ca
from mpecss.helpers.solver_wrapper import solve_smooth_subproblem, is_solver_success

logger = logging.getLogger('mpecss.restoration')

_PERTURB_SCALE_LO = 0.5
_PERTURB_SCALE_HI = 2.0
_GRAD_ZERO_TOL = 1e-12
_GAMMA_INCREASE = 2.0
_BIG = 1e+20
_DENOM_REG = 1e-12

# Module-level cache for Jacobians (cleared between problems via clear_jacobian_cache)
_JACOBIAN_CACHE: dict = {}


def clear_jacobian_cache():
    """Clear the Jacobian cache to free memory."""
    _JACOBIAN_CACHE.clear()


def _get_jacobians(problem):
    """
    Get or compute Jacobian CasADi Functions for G and H.

    Caches the result in a module-level cache keyed by problem name.

    Parameters
    ----------
    problem : dict
        Problem specification with 'G_fn', 'H_fn', 'n_x'.

    Returns
    -------
    jac_G_fn : ca.Function
        Jacobian of G w.r.t. x: (n_comp, n_x) matrix.
    jac_H_fn : ca.Function
        Jacobian of H w.r.t. x: (n_comp, n_x) matrix.
    """
    prob_name = problem.get('name', 'unknown')
    if prob_name in _JACOBIAN_CACHE:
        return _JACOBIAN_CACHE[prob_name]
    
    n_x = problem['n_x']
    _sym = ca.MX if n_x > 500 else ca.SX
    x_sym = _sym.sym('x_jac', n_x)
    
    G_expr = problem['G_fn'](x_sym)
    H_expr = problem['H_fn'](x_sym)
    
    jac_G_fn = ca.Function('jac_G', [x_sym], [ca.jacobian(G_expr, x_sym)])
    jac_H_fn = ca.Function('jac_H', [x_sym], [ca.jacobian(H_expr, x_sym)])
    
    _JACOBIAN_CACHE[prob_name] = (jac_G_fn, jac_H_fn)
    
    return jac_G_fn, jac_H_fn


def random_perturb(z, biactive_idx, problem, eps=0.01, seed=None):
    """
    Jacobian-guided perturbation of biactive variables.

    For each biactive index i, computes the gradient of G_i or H_i w.r.t. x,
    then perturbs z in that gradient direction to push one function positive.

    Parameters
    ----------
    z : np.ndarray
        Current iterate (full variable vector).
    biactive_idx : list[int]
        Indices of biactive complementarity pairs.
    problem : dict
        Problem specification with 'G_fn', 'H_fn', 'n_x'.
    eps : float
        Perturbation magnitude.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    z_new : np.ndarray
        Perturbed iterate.
    """
    rng = np.random.RandomState(seed)
    z_new = np.copy(z)
    
    if len(biactive_idx) == 0:
        logger.debug('random_perturb: no biactive indices, returning copy')
        return z_new
    
    # Get bounds for clipping
    lbx = np.asarray(problem.get('lbx', np.full(z.shape, -_BIG))).flatten()
    ubx = np.asarray(problem.get('ubx', np.full(z.shape, _BIG))).flatten()
    
    try:
        jac_G_fn, jac_H_fn = _get_jacobians(problem)
        JG = np.asarray(jac_G_fn(z))
        JH = np.asarray(jac_H_fn(z))
    except Exception as e:
        logger.warning(f'random_perturb: Jacobian evaluation failed: {e}, using random perturbation')
        # Fall back to simple random perturbation
        direction = rng.uniform(-eps, eps, size=z.shape)
        z_new = z + direction
        z_new = np.clip(z_new, np.where(np.isfinite(lbx), lbx, -1e10), 
                                np.where(np.isfinite(ubx), ubx, 1e10))
        return z_new
    
    for i in biactive_idx:
        # Randomly choose to push G or H positive
        if rng.rand() < 0.5:
            grad = JG[i] if JG.ndim == 2 else JG.flatten()
        else:
            grad = JH[i] if JH.ndim == 2 else JH.flatten()
        
        # Check for NaN in gradient
        if not np.all(np.isfinite(grad)):
            logger.debug(f'random_perturb: NaN in gradient at index {i}, using random direction')
            direction = rng.uniform(_PERTURB_SCALE_LO, _PERTURB_SCALE_HI, size=z.shape)
        else:
            norm = np.linalg.norm(grad)
            if norm < _GRAD_ZERO_TOL:
                # Zero gradient: perturb randomly
                direction = rng.uniform(_PERTURB_SCALE_LO, _PERTURB_SCALE_HI, size=z.shape)
            else:
                direction = grad / norm
        
        z_new += eps * direction
    
    # Clip to bounds (use finite values only)
    clip_lb = np.where(np.isfinite(lbx), lbx, -1e10)
    clip_ub = np.where(np.isfinite(ubx), ubx, 1e10)
    z_new = np.clip(z_new, clip_lb, clip_ub)
    
    # Final NaN check
    if not np.all(np.isfinite(z_new)):
        logger.warning('random_perturb: NaN/Inf in result, returning original z')
        return np.copy(z)
    
    logger.info(f'random_perturb: perturbed {len(biactive_idx)} biactive indices with eps={eps}')
    return z_new


def quadratic_regularizer(z, t_k, delta_k, problem, biactive_idx, gamma=1.0, solver_opts=None, max_tries=3):
    """
    Quadratic regularisation that adds a penalty pulling G and H apart.

    For biactive indices, adds:
        gamma * sum_i (G_i(x) - H_i(x))^2 / (|G_i(x)| + |H_i(x)| + eps)

    to the objective, then re-solves with the same t_k.

    Parameters
    ----------
    z : np.ndarray
        Current iterate.
    t_k : float
        Current smoothing parameter.
    delta_k : float
        Current shift.
    problem : dict
        Problem specification.
    biactive_idx : list[int]
        Biactive indices.
    gamma : float
        Regularization weight.
    solver_opts : dict or None
        IPOPT options.
    max_tries : int
        Max attempts with increasing gamma.

    Returns
    -------
    result : dict or None
        Solution dict if successful, None otherwise.
    """
    if len(biactive_idx) == 0:
        logger.info('quadratic_regularizer: no biactive indices, skipping')
        return None
    
    G_fn = problem['G_fn']
    H_fn = problem['H_fn']
    
    for attempt in range(max_tries):
        try:
            info = problem['build_casadi'](t_k, delta_k, smoothing='product')
            x_sym = info['x']
            f_orig = info['f']
            n_comp = info['n_comp']
            
            G_expr = G_fn(x_sym)
            H_expr = H_fn(x_sym)
            
            # Build regularization term
            reg_term = 0
            for i in biactive_idx:
                Gi = G_expr[i] if hasattr(G_expr, '__getitem__') else G_expr
                Hi = H_expr[i] if hasattr(H_expr, '__getitem__') else H_expr
                diff = Gi - Hi
                denom = ca.fabs(Gi) + ca.fabs(Hi) + _DENOM_REG
                reg_term += diff**2 / denom
            
            f_reg = f_orig + gamma * reg_term
            
            nlp = {'x': x_sym, 'f': f_reg, 'g': info['g']}
            
            from mpecss.helpers.solver_wrapper import DEFAULT_IPOPT_OPTS
            opts = dict(DEFAULT_IPOPT_OPTS)
            if solver_opts:
                opts.update(solver_opts)
            
            casadi_opts = {'ipopt': opts, 'print_time': False, 'error_on_fail': False}
            solver = ca.nlpsol('reg_solver', 'ipopt', nlp, casadi_opts)
            
            t_start = time.perf_counter()
            sol = solver({"x0": z, "lbg": info['lbg'], "ubg": info['ubg'], "lbx": info['lbx'], "ubx": info['ubx']})
            cpu_time = time.perf_counter() - t_start
            
            z_new = np.asarray(sol['x']).flatten()
            stats = solver.stats()
            status = stats.get('return_status', 'unknown')
            
            if is_solver_success(status):
                logger.info(f'quadratic_regularizer: attempt {attempt+1} succeeded, gamma={gamma}')
                return {
                    'z_k': z_new,
                    'lam_g': np.asarray(sol['lam_g']).flatten(),
                    'lam_x': np.asarray(sol['lam_x']).flatten(),
                    'f_val': float(sol['f']),
                    'status': status,
                    'cpu_time': cpu_time,
                    'g_val': np.asarray(sol['g']).flatten(),
                    'problem_info': info,
                }
            else:
                logger.warning(f'quadratic_regularizer: attempt {attempt+1} failed with status={status}')
        
        except Exception as e:
            logger.warning(f'quadratic_regularizer: attempt {attempt+1} exception: {e}')
            if 'Invalid_Number_Detected' in str(e):
                return {'status': 'Invalid_Number_Detected'}
        
        gamma *= _GAMMA_INCREASE
    
    return None


def directional_escape(z, lambda_G, lambda_H, biactive_idx, problem, step_size=0.1, max_tries=3):
    """
    Directional escape using sign-based direction computed via Jacobians.

    For biactive index i, computes:
        d_i = sign(lambda_G_i - lambda_H_i)
    Then moves z along the gradient of G_i (if d_i > 0) or H_i (if d_i < 0)
    to push the iterate toward a vertex of the complementarity constraint.

    Parameters
    ----------
    z : np.ndarray
        Current iterate.
    lambda_G : np.ndarray
        Multipliers for G >= -delta constraints.
    lambda_H : np.ndarray
        Multipliers for H >= -delta constraints.
    biactive_idx : list[int]
        Biactive indices.
    problem : dict
        Problem specification.
    step_size : float
        Initial step size.
    max_tries : int
        Number of step-size halvings to try.

    Returns
    -------
    z_new : np.ndarray
        Escaped iterate.
    """
    z_new = np.copy(z)
    
    if len(biactive_idx) == 0:
        logger.info('directional_escape: no biactive indices, skipping')
        return z_new
    
    # Get bounds for clipping
    lbx = np.asarray(problem.get('lbx', np.full(z.shape, -_BIG))).flatten()
    ubx = np.asarray(problem.get('ubx', np.full(z.shape, _BIG))).flatten()
    
    try:
        jac_G_fn, jac_H_fn = _get_jacobians(problem)
        JG = np.asarray(jac_G_fn(z))
        JH = np.asarray(jac_H_fn(z))
    except Exception as e:
        logger.warning(f'directional_escape: Jacobian evaluation failed: {e}')
        return z_new
    
    for attempt in range(max_tries):
        z_trial = np.copy(z)
        
        for i in biactive_idx:
            d_i = np.sign(lambda_G[i] - lambda_H[i])
            if d_i == 0:
                d_i = 1.0
            
            if d_i > 0:
                grad = JG[i] if JG.ndim == 2 else JG.flatten()
            else:
                grad = JH[i] if JH.ndim == 2 else JH.flatten()
            
            # Check for NaN in gradient
            if not np.all(np.isfinite(grad)):
                logger.debug(f'directional_escape: NaN in gradient at index {i}, skipping')
                continue
                
            norm = np.linalg.norm(grad)
            if norm > _GRAD_ZERO_TOL:
                z_trial += step_size * (grad / norm)
        
        logger.info(f'directional_escape: attempt {attempt+1}, step_size={step_size:.4e}')
        z_new = z_trial
        step_size *= 0.5
    
    # Clip to bounds (use finite values only)
    clip_lb = np.where(np.isfinite(lbx), lbx, -1e10)
    clip_ub = np.where(np.isfinite(ubx), ubx, 1e10)
    z_new = np.clip(z_new, clip_lb, clip_ub)
    
    # Final NaN check
    if not np.all(np.isfinite(z_new)):
        logger.warning('directional_escape: NaN/Inf in result, returning original z')
        return np.copy(z)
    
    return z_new


def run_restoration(z, t_k, delta_k, problem, biactive_idx, lambda_G, lambda_H, lambda_comp,
                    strategy='random_perturb', params=None, solver_opts=None, seed=None):
    """
    Run a restoration heuristic and optionally re-solve.

    Parameters
    ----------
    z : np.ndarray
        Current iterate.
    t_k, delta_k : float
        Smoothing parameters.
    problem : dict
        Problem specification.
    biactive_idx : list[int]
        Biactive indices.
    lambda_G, lambda_H, lambda_comp : np.ndarray
        Current multipliers.
    strategy : str
        One of 'random_perturb', 'quadratic_regularizer', 'directional_escape', 'cascade'.
    params : dict or None
        Strategy-specific parameters.
    solver_opts : dict or None
        IPOPT options.
    seed : int or None
        Random seed.

    Returns
    -------
    z_new : np.ndarray
        New iterate.
    sol : dict or None
        Solution dict from re-solve (if applicable).
    """
    params = params or {}
    _smoothing = params.get('smoothing', 'product')
    
    if strategy == 'cascade':
        # Try strategies in order: random_perturb -> directional_escape -> quadratic_regularizer
        strategies_to_try = ['random_perturb', 'directional_escape', 'quadratic_regularizer']
        for strat in strategies_to_try:
            try:
                result = run_restoration(z, t_k, delta_k, problem, biactive_idx, 
                                       lambda_G, lambda_H, lambda_comp,
                                       strategy=strat, params=params, 
                                       solver_opts=solver_opts, seed=seed)
                if result is not None:
                    return result
            except Exception:
                continue
        # If all failed, return None
        return None
    
    elif strategy == 'random_perturb':
        eps = params.get('perturb_eps', 0.01)
        z_new = random_perturb(z, biactive_idx, problem, eps=eps, seed=seed)
        sol = solve_smooth_subproblem(z_new, t_k, delta_k, problem, 
                                      solver_opts=solver_opts, smoothing=_smoothing)
        # Return dict with z_k for consistency with mpecss.py expectations
        sol['z_k'] = sol.get('z_k', z_new)
        return sol
    
    elif strategy == 'quadratic_regularizer':
        gamma = params.get('gamma', 1.0)
        max_tries = params.get('max_tries', 3)
        sol = quadratic_regularizer(z, t_k, delta_k, problem, biactive_idx,
                                    gamma=gamma, solver_opts=solver_opts, max_tries=max_tries)
        if sol is None:
            sol = {'status': 'quadratic_regularizer(failed)', 'z_k': z}
        return sol
    
    elif strategy == 'directional_escape':
        step_size = params.get('step_size', 0.1)
        max_tries = params.get('max_tries', 3)
        z_new = directional_escape(z, lambda_G, lambda_H, biactive_idx, problem,
                                   step_size=step_size, max_tries=max_tries)
        sol = solve_smooth_subproblem(z_new, t_k, delta_k, problem,
                                      solver_opts=solver_opts, smoothing=_smoothing)
        # Return dict with z_k for consistency with mpecss.py expectations
        sol['z_k'] = sol.get('z_k', z_new)
        return sol
    
    else:
        raise ValueError(f'Unknown restoration strategy: {strategy}')



