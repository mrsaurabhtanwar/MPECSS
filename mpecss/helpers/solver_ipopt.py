"""
IPOPT interior-point solver functions and automatic fallback wrappers.
Builds CasADi NLP instances, runs IPOPT, and handles solver fallbacks.
"""

import time
import logging
import numpy as np
import casadi as ca
from mpecss.helpers.solver_cache import (
    _TEMPLATE_CACHE,
    _SOLVER_CACHE,
    _PARAMETRIC_CACHE,
    _LEAKED_SOLVERS,
    _get_template,
    _tol_bucket,
    _t_round,
    _evict_problem_from_cache,
)

logger = logging.getLogger('mpecss.solver.ipopt')

DEFAULT_IPOPT_OPTS = {
    'tol': 1e-12,
    'acceptable_tol': 1e-09,
    'print_level': 0,
    'max_iter': 3000,
    'mu_strategy': 'adaptive',
    'mu_oracle': 'quality-function',
    'linear_solver': 'mumps'
}

_WARM_START_EXTRA_OPTS = {
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-09,
    'warm_start_bound_frac': 1e-09,
    'warm_start_slack_bound_frac': 1e-09,
    'warm_start_slack_bound_push': 1e-09,
    'warm_start_mult_bound_push': 1e-09
}

_FALLBACK_TRIGGER_STATUSES = {
    'Invalid_Option',
    'Restoration_Failed',
    'Invalid_Number_Detected',
    'Error_In_Step_Computation',
    'Infeasible_Problem_Detected',
    'Maximum_Iterations_Exceeded',
    'Diverging_Iterates',          # ex9.1.x, design-cent-4: IPOPT diverges
}

_SOLVER_FALLBACK_CHAIN = [
    {'mehrotra_algorithm': 'yes'},
    {'tol': 1e-08, 'acceptable_tol': 1e-06},
    # Diverging-specific: gradient-based scaling + relaxed bounds help
    # when the unscaled problem causes IPOPT to take huge steps
    {
        'nlp_scaling_method': 'gradient-based',
        'bound_relax_factor': 1e-8,
        'tol': 1e-06,
        'acceptable_tol': 1e-04,
    },
]


def _get_concrete_solver(problem, t_k, delta_k, solver_opts, warm_start, smoothing='product'):
    """
    Get or create a CasADi NLP solver for a given (t_k, delta_k) pair.

    SX path  (n_x < 500):  substitutes concrete (t, d) values into the
                            symbolic template — cheap to rebuild.
    MX path  (n_x >= 500): compiles a parametric solver with p=[t, d] once,
                            then reuses it for all homotopy iterations.
    """
    prob_name = problem.get('name', 'unknown')
    n_x = problem['n_x']
    opts = dict(DEFAULT_IPOPT_OPTS)
    if solver_opts:
        opts.update(solver_opts)
    if warm_start:
        for k, v in _WARM_START_EXTRA_OPTS.items():
            opts.setdefault(k, v)
    
    # Auto-select linear solver if not specified
    if not solver_opts or 'linear_solver' not in solver_opts:
        try:
            from mpecss.helpers.solver_acceleration import select_linear_solver_oss
            auto_ls = select_linear_solver_oss(n_x)
            if auto_ls != opts.get('linear_solver'):
                logger.debug(f'Auto-selecting linear solver: {auto_ls} for n_x={n_x}')
                opts['linear_solver'] = auto_ls
        except ImportError:
            pass
    
    tol_b = _tol_bucket(opts.get('tol', 1e-08))
    ws_flag = 'ws' if warm_start else 'cs'
    use_mx = n_x >= 500
    
    if use_mx:
        opt_key = f"{opts.get('linear_solver', 'mumps')}|{tol_b}|{ws_flag}"
        pkey = f'{prob_name}|{opt_key}|{smoothing}'
        if pkey not in _PARAMETRIC_CACHE:
            t_sym, d_sym, info_sym = _get_template(problem, smoothing)
            p_sym = ca.vertcat(t_sym, d_sym)
            nlp = {
                'x': info_sym['x'],
                'f': info_sym['f'],
                'g': info_sym['g'],
                'p': p_sym
            }
            opts_copy = dict(opts)
            opts_copy['tol'] = tol_b
            opts_copy.setdefault('print_level', 0)
            casadi_opts = {
                'ipopt': opts_copy,
                'print_time': False,
                'error_on_fail': False,
                # ── Reverse-mode AD for large MX problems ────────────────────
                # CasADi's default forward-mode uses fwd64_g (64 simultaneous
                # seed vectors) which needs O(n_x * 64) memory for the Jacobian.
                # Reverse-mode uses one sweep per output (O(n_g) memory), which
                # is far cheaper when n_x >> n_g — typical for large MPECs.
                # ad_weight = 1.0  → always prefer reverse-mode
                # ad_weight_sp = 1.0 → use reverse-mode for sparsity detection
                'ad_weight': 1.0,
                'ad_weight_sp': 1.0,
            }
            logger.info(f"Compiling parametric MX solver for {prob_name} (n_x={n_x}, ls={opts.get('linear_solver', 'mumps')}, AD=reverse)...")
            t0 = time.perf_counter()
            solver = ca.nlpsol('solver', 'ipopt', nlp, casadi_opts)
            logger.info(f'Parametric MX solver compiled in {time.perf_counter() - t0:.1f}s')
            _PARAMETRIC_CACHE[pkey] = solver
        return _PARAMETRIC_CACHE[pkey]
    
    # SX path: concrete solver for each (t, delta) pair
    t_r = _t_round(t_k)
    d_r = _t_round(delta_k)
    ckey = f'{prob_name}|{t_r}|{d_r}|{tol_b}|{n_x}|{ws_flag}|{smoothing}'
    if ckey in _SOLVER_CACHE:
        return _SOLVER_CACHE[ckey]
    
    t_sym, d_sym, info_sym = _get_template(problem, smoothing)
    f_concrete = ca.substitute([info_sym['f']], [t_sym, d_sym], [t_k, delta_k])[0]
    g_concrete = ca.substitute([info_sym['g']], [t_sym, d_sym], [t_k, delta_k])[0]
    nlp = {
        'x': info_sym['x'],
        'f': f_concrete,
        'g': g_concrete
    }
    opts_copy = dict(opts)
    opts_copy['tol'] = tol_b
    opts_copy.setdefault('print_level', 0)
    casadi_opts = {
        'ipopt': opts_copy,
        'print_time': False,
        'error_on_fail': False
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp, casadi_opts)
    _SOLVER_CACHE[ckey] = solver
    logger.debug(f'Compiled SX solver for {prob_name} (t={t_r}, d={d_r})')
    return solver


def solve_smooth_subproblem(z0, t_k, delta_k, problem, solver_opts=None,
                             lam_g0=None, lam_x0=None, smoothing='product'):
    """
    Solve one smoothed NLP subproblem via CasADi + IPOPT.

    Parameters
    ----------
    z0 : np.ndarray
        Warm-start primal point (length n_x).
    t_k : float
        Smoothing parameter.
    delta_k : float
        Regularisation shift.
    problem : dict
        Problem specification.
    solver_opts : dict or None
        Override IPOPT options.
    lam_g0 : np.ndarray or None
        Warm-start constraint multipliers.
    lam_x0 : np.ndarray or None
        Warm-start bound multipliers.
    smoothing : str
        'product' or 'fb' (Fischer-Burmeister).

    Returns
    -------
    dict
        Keys: z_k, lam_g, lam_x, f_val, status, cpu_time, g_val,
              problem_info, iter_count.
    """
    t_sym, d_sym, info = _get_template(problem, smoothing)
    x_sym = info['x']
    g_sym = info['g']
    n_x = x_sym.shape[0]
    n_g = g_sym.shape[0]
    
    do_warm_start = lam_g0 is not None
    solver = _get_concrete_solver(problem, t_k, delta_k, solver_opts, warm_start=do_warm_start, smoothing=smoothing)
    
    z0 = np.asarray(z0).flatten()
    if z0.shape[0] != n_x:
        raise ValueError(f'z0 has length {z0.shape[0]}, expected {n_x}')
    
    # Check for NaN/Inf in initial point and fix if needed
    if not np.all(np.isfinite(z0)):
        logger.warning(f'NaN/Inf detected in z0, replacing with bounded values')
        lbx_arr = np.asarray(info['lbx']).flatten()
        ubx_arr = np.asarray(info['ubx']).flatten()
        for i in range(len(z0)):
            if not np.isfinite(z0[i]):
                lb = lbx_arr[i] if np.isfinite(lbx_arr[i]) else -1.0
                ub = ubx_arr[i] if np.isfinite(ubx_arr[i]) else 1.0
                z0[i] = (lb + ub) / 2 if np.isfinite(lb + ub) else 0.0
    
    sol_args = {
        'x0': ca.DM(z0),
        'lbg': ca.DM(info['lbg']),
        'ubg': ca.DM(info['ubg']),
        'lbx': ca.DM(info['lbx']),
        'ubx': ca.DM(info['ubx'])
    }
    if n_x >= 500:
        sol_args['p'] = ca.DM([t_k, delta_k])
    
    # Warm-start multipliers
    if lam_g0 is not None:
        sol_args['lam_g0'] = ca.DM(lam_g0)
    if lam_x0 is not None:
        sol_args['lam_x0'] = ca.DM(lam_x0)
    
    t0 = time.perf_counter()
    try:
        sol = solver.call(sol_args)
        cpu_time = time.perf_counter() - t0
        
        stats = solver.stats()
        status = stats.get('return_status', 'unknown')
        iter_count = stats.get('iter_count', -1)
        
        z_k = np.asarray(sol['x']).flatten()
        lam_g = np.asarray(sol['lam_g']).flatten()
        lam_x = np.asarray(sol['lam_x']).flatten()
        f_val = float(sol['f'])
        g_val = np.asarray(sol['g']).flatten()
    except Exception as e:
        err_msg = str(e)
        if "exception set" in err_msg or "KeyboardInterrupt" in err_msg:
            logger.warning('Solver interrupted (Timeout/OOM likely)')
        else:
            logger.warning(f'Solver exception: {e}')
        cpu_time = time.perf_counter() - t0
        z_k, lam_g, lam_x, f_val, g_val = _zero_fallback(z0, n_x, n_g)
        status = 'Exception'
        iter_count = -1
    
    return {
        'z_k': z_k,
        'lam_g': lam_g,
        'lam_x': lam_x,
        'f_val': f_val,
        'status': status,
        'cpu_time': cpu_time,
        'g_val': g_val,
        'problem_info': info,
        'iter_count': iter_count
    }


def is_solver_success(status):
    """Return True if IPOPT status indicates a successful solve."""
    return status in frozenset({'Solve_Succeeded', 'Solved_To_Acceptable_Level', 'Search_Direction_Becomes_Too_Small'})


def solve_with_solver_fallback(z0, t_k, delta_k, problem, solver_opts=None,
                                lam_g0=None, lam_x0=None, smoothing='product'):
    """
    Solve a smoothed NLP with automatic open-source solver fallback.

    Tries the primary solver first. On failure (status in
    _FALLBACK_TRIGGER_STATUSES) retries with alternatives from
    _SOLVER_FALLBACK_CHAIN (Mehrotra, relaxed tolerances).
    As a last resort, retries with the Fischer-Burmeister reformulation.
    Parameters and return value are identical to solve_smooth_subproblem().
    """
    n_x = problem.get('n_x', 0)
    
    # Always use IPOPT for robustness
    sol = solve_smooth_subproblem(z0, t_k, delta_k, problem, solver_opts=solver_opts, lam_g0=lam_g0, lam_x0=lam_x0, smoothing=smoothing)
        
    if is_solver_success(sol['status']):
        return sol
    
    # Fallback to IPOPT directly on failure
    if not is_solver_success(sol['status']):
        sol = solve_smooth_subproblem(z0, t_k, delta_k, problem, solver_opts=solver_opts, lam_g0=lam_g0, lam_x0=lam_x0, smoothing=smoothing)
        if is_solver_success(sol['status']):
            return sol

    if sol['status'] not in _FALLBACK_TRIGGER_STATUSES:
        return sol

    
    primary_status = sol['status']
    best_sol = sol
    
    if solver_opts is None:
        solver_opts = {}
    primary_ls = solver_opts.get('linear_solver', DEFAULT_IPOPT_OPTS.get('linear_solver', 'mumps'))
    
    for i, fallback_opts in enumerate(_SOLVER_FALLBACK_CHAIN):
        fb_opts = dict(solver_opts)
        fb_opts.update(fallback_opts)
        ls_name = fallback_opts.get('linear_solver', primary_ls)
        if ls_name == primary_ls and 'tol' not in fallback_opts and 'mehrotra_algorithm' not in fallback_opts:
            logger.debug(f"Skipping fallback [{i + 1}]: same solver '{ls_name}'")
            continue
        _evict_problem_from_cache(problem.get('name', 'unknown'))
        fb_sol = solve_smooth_subproblem(z0, t_k, delta_k, problem, solver_opts=fb_opts, lam_g0=lam_g0, lam_x0=lam_x0, smoothing=smoothing)
        if is_solver_success(fb_sol['status']):
            logger.info(f'Fallback [{i + 1}] succeeded: {ls_name} (primary: {primary_status})')
            return fb_sol
        if fb_sol.get('f_val', float('inf')) < best_sol.get('f_val', float('inf')):
            best_sol = fb_sol
    
    if smoothing == 'product':
        logger.info('All linear solver fallbacks failed. Trying Fischer-Burmeister...')
        fb_sol = solve_smooth_subproblem(z0, t_k, delta_k, problem, solver_opts=solver_opts, lam_g0=None, lam_x0=None, smoothing='fb')
        if is_solver_success(fb_sol['status']):
            logger.info('Fischer-Burmeister fallback succeeded.')
            return fb_sol
        if fb_sol.get('f_val', float('inf')) < best_sol.get('f_val', float('inf')):
            best_sol = fb_sol
    
    logger.debug(f'All fallbacks exhausted (primary: {primary_status})')
    return best_sol


def _zero_fallback(z0, n_x, n_g):
    """Return zero-filled fallback arrays for failed solves."""
    return (z0.copy(), np.zeros(n_g), np.zeros(n_x), float('inf'), np.zeros(n_g))


def _oom_result(z0, n_x, n_g, info):
    """Return a structured OOM result dict."""
    z_k, lam_g, lam_x, f_val, g_val = _zero_fallback(z0, n_x, n_g)
    return {
        'z_k': z_k,
        'lam_g': lam_g,
        'lam_x': lam_x,
        'f_val': f_val,
        'status': 'OOM',
        'cpu_time': 0.0,
        'g_val': g_val,
        'problem_info': info,
        'iter_count': -1
    }
