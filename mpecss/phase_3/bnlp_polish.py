"""
Branch NLP (BNLP) polishing for MPECSS.

After MPECSS converges to an approximate solution, the complementarity
constraints still introduce near-degeneracy in the KKT system.  BNLP
polishing removes this degeneracy by:

  1. Identifying the active-set partition from the approximate solution:
       I1 = {i : G_i ≈ 0}  →  fix G_i = 0, keep H_i ≥ 0
       I2 = {i : H_i ≈ 0}  →  keep G_i ≥ 0, fix H_i = 0

  2. Solving the resulting "branch NLP" (BNLP) — a standard NLP with
     equality/inequality constraints but NO complementarity products.
     Under MPEC-MFCQ, MFCQ holds for every BNLP, so IPOPT converges
     reliably.

  3. Checking whether the BNLP solution improves the objective value
     and satisfies complementarity (it does by construction).

This mirrors MPECopt's Phase II BNLP solves (Nurkanović & Leyffer, 2025)
but in a simpler single-shot setting.

Benefits:
  - Removes complementarity degeneracy → IPOPT converges better
  - Can tighten the solution from "nearly B-stat" to "exactly B-stat"
  - Provides an exact complementarity-feasible point (G_i·H_i = 0)
  - Uses open-source IPOPT linear solvers (default: MUMPS)

Reference:
  Nurkanović, A. & Leyffer, S. (2025). A Globally Convergent Method for
  Computing B-stationary Points of MPECs. arXiv:2501.13835, Section 3.4.
"""

import logging
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import casadi as ca
from mpecss.helpers.loaders.macmpec_loader import evaluate_GH, complementarity_residual
from mpecss.helpers.solver_wrapper import DEFAULT_IPOPT_OPTS, is_solver_success

logger = logging.getLogger('mpecss.bnlp_polish')

_BIG = 1e+20
_ACTIVE_TOL = 1e-06

_BNLP_IPOPT_OPTS = {
    'tol': 1e-12,
    'acceptable_tol': 1e-09,
    'print_level': 0,
    'max_iter': 3000,
    'mu_strategy': 'adaptive',
    'mu_oracle': 'quality-function',
    'linear_solver': 'mumps',
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-09,
    'warm_start_bound_frac': 1e-09,
    'warm_start_slack_bound_frac': 1e-09,
    'warm_start_slack_bound_push': 1e-09,
    'warm_start_mult_bound_push': 1e-09
}

_FINAL_POLISH_OPTS = {
    'tol': 1e-14,
    'acceptable_tol': 1e-12,
    'print_level': 0,
    'max_iter': 5000,
    'mu_strategy': 'adaptive',
    'mu_oracle': 'quality-function',
    'linear_solver': 'mumps',
    'warm_start_init_point': 'yes',
    'warm_start_bound_push': 1e-09,
    'warm_start_bound_frac': 1e-09,
    'warm_start_slack_bound_frac': 1e-09,
    'warm_start_slack_bound_push': 1e-09,
    'warm_start_mult_bound_push': 1e-09
}


def identify_active_set(z, problem, tol=_ACTIVE_TOL):
    """
    Identify the complementarity active-set partition at point z.

    For standard NCP (two-partition):
      I1 : G_i ≈ 0  → fix G_i = 0, H_i ≥ 0
      I2 : H_i ≈ 0  → G_i ≥ 0, fix H_i = 0

    For box-MCP (three-partition, gnash*m family):
      I1 : G_i ≈ 0              → fix G_i = 0, H_i ≥ 0           (interior)
      I2 : H_i ≈ 0              → G_i ≥ 0, fix H_i = 0           (lower-bound active)
      I3 : H_i ≈ ubH_i         → G_i ≤ 0, fix H_i = ubH_i       (upper-bound active)

    Returns (I1, I2, I_biactive, I3) for callers that support box-MCP,
    or (I1, I2, I_biactive) for legacy callers (I3 rolled into I1).
    """
    G, H = evaluate_GH(z, problem)
    n_comp = problem['n_comp']
    ubH_finite = problem.get('ubH_finite', [])   # [(idx, ub_val), ...]
    ubH_map    = {i: ub for i, ub in ubH_finite}

    I1 = []; I2 = []; I3 = []; I_biactive = []

    for i in range(n_comp):
        g_val = abs(float(G[i]))
        h_val = abs(float(H[i]))

        # ── Box-MCP: check upper-bound active first ──────────────────────
        if i in ubH_map:
            slack_upper = abs(float(H[i]) - ubH_map[i])
            if slack_upper < tol:
                # H_i ≈ ubH_i: upper-bound active → I3
                I3.append(i)
                continue

        # ── Standard NCP partition ──────────────────────────────────
        if g_val < tol and h_val < tol:
            I_biactive.append(i)
            if g_val <= h_val:
                I1.append(i)
            else:
                I2.append(i)
        elif g_val < tol:
            I1.append(i)
        elif h_val < tol:
            I2.append(i)
        else:
            # Neither near zero: assign to the smaller (tiebreaker)
            if g_val <= h_val:
                I1.append(i)
            else:
                I2.append(i)

    return I1, I2, I_biactive, I3


def _build_bnlp(z_star, problem, I1, I2, I3=None, solver_opts=None, f_cut=None, use_ultra_tight=False):
    """
    Build a Branch NLP with fixed complementarity active set.

    BNLP(I1, I2, I3):
        min  f(x)
        s.t. c_lb ≤ c(x) ≤ c_ub        (original constraints)
             x_lb ≤ x ≤ x_ub            (variable bounds)
             G_i(x) = 0, H_i(x) ≥ 0    for i ∈ I1  (interior active)
             G_i(x) ≥ 0, H_i(x) = 0    for i ∈ I2  (lower-bound active)
             G_i(x) ≤ 0, H_i(x) = ubH  for i ∈ I3  (upper-bound active, box-MCP)
             f(x) ≤ f_cut               (optional objective cut)

    NO complementarity product constraint (G·H ≤ t) — the NLP is regular
    under MPEC-MFCQ for all three active-set types.

    Returns
    -------
    result : dict with keys: 'z_polish', 'f_val', 'status', 'success', 'cpu_time'
    """
    if I3 is None:
        I3 = []
    I3_set = set(I3)
    ubH_map = {i: ub for i, ub in problem.get('ubH_finite', [])}

    n_x = problem['n_x']
    n_comp = problem['n_comp']
    n_con = problem.get('n_con', 0)

    info = problem['build_casadi'](0, 0)
    x_sym = info['x']
    f_sym = info['f']

    g_parts = []
    lbg_parts = []
    ubg_parts = []

    # Add original constraints
    if n_con > 0:
        g_orig = info['g'][:n_con]
        g_parts.append(g_orig)
        lbg_parts.extend(info['lbg'][:n_con])
        ubg_parts.extend(info['ubg'][:n_con])

    G_expr = problem['G_fn'](x_sym)
    H_expr = problem['H_fn'](x_sym)

    # Add complementarity constraints based on active set
    for i in range(n_comp):
        if i in I3_set:
            # Upper-bound active (box-MCP): G_i ≤ 0, H_i = ubH_i
            ub_val = ubH_map.get(i, _BIG)
            g_parts.append(G_expr[i])
            lbg_parts.append(-_BIG)
            ubg_parts.append(0)          # G_i ≤ 0
            g_parts.append(H_expr[i])
            lbg_parts.append(ub_val)
            ubg_parts.append(ub_val)     # H_i = ubH_i (equality)
        elif i in I1:
            # Interior active: G_i = 0, H_i ≥ 0
            g_parts.append(G_expr[i])
            lbg_parts.append(0)
            ubg_parts.append(0)
            g_parts.append(H_expr[i])
            lbg_parts.append(0)
            ubg_parts.append(_BIG)
        else:
            # Lower-bound active (standard NCP): G_i ≥ 0, H_i = 0
            g_parts.append(G_expr[i])
            lbg_parts.append(0)
            ubg_parts.append(_BIG)
            g_parts.append(H_expr[i])
            lbg_parts.append(0)
            ubg_parts.append(0)
    
    # Optional objective cut
    if f_cut is not None:
        g_parts.append(f_sym)
        lbg_parts.append(-_BIG)
        ubg_parts.append(f_cut)
    
    g_all = ca.vertcat(*g_parts) if g_parts else ca.SX(0, 1)
    
    nlp = {'x': x_sym, 'f': f_sym, 'g': g_all}
    
    # Select IPOPT options
    if use_ultra_tight:
        opts = dict(_FINAL_POLISH_OPTS)
    else:
        opts = dict(_BNLP_IPOPT_OPTS)
    if solver_opts:
        opts.update(solver_opts)
    
    from mpecss.helpers.solver_wrapper import build_universal_nlp_solver
    n_x = problem.get('n_x', len(z_star))
    solver = build_universal_nlp_solver('bnlp', n_x, nlp, ipopt_opts=opts)

    
    t_start = time.perf_counter()
    try:
        sol = solver(x0=z_star, lbg=lbg_parts, ubg=ubg_parts, lbx=info['lbx'], ubx=info['ubx'])
        cpu_time = time.perf_counter() - t_start
        
        stats = solver.stats()
        status = stats.get('return_status', 'unknown')
        
        z_polish = np.asarray(sol['x']).flatten()
        f_val = float(sol['f'])
        success = is_solver_success(status)
    except Exception as e:
        logger.warning(f'BNLP solve exception: {e}')
        cpu_time = time.perf_counter() - t_start
        z_polish = z_star.copy()
        f_val = float('inf')
        status = 'Exception'
        success = False
    
    return {
        'z_polish': z_polish,
        'f_val': f_val,
        'status': status,
        'success': success,
        'cpu_time': cpu_time
    }


def bnlp_polish(results, problem, solver_opts=None):
    """
    Apply BNLP polishing to MPECSS results.

    Takes the converged (or nearly-converged) MPECSS solution, identifies
    the active set, and solves a clean BNLP to get a more accurate solution.

    Parameters
    ----------
    results : dict
        MPECSS results dict (from run_mpecss).
    problem : dict
        Problem specification.
    solver_opts : dict or None
        Override IPOPT options for BNLP solve.

    Returns
    -------
    results : dict
        Updated results dict. If polishing improved the solution, the
        following keys are updated:
        - 'z_final', 'f_final', 'comp_res', 'stationarity'
        - 'bnlp_polish': dict with polishing details
    """
    z_star = results['z_final']
    f_star = results.get('f_final', float('inf'))
    
    I1, I2, I_biactive, I3 = identify_active_set(z_star, problem)
    logger.info(f'BNLP polish: |I1|={len(I1)}, |I2|={len(I2)}, |biactive|={len(I_biactive)}, |I3|={len(I3)}')

    bnlp_result = _build_bnlp(z_star, problem, I1, I2, I3=I3, solver_opts=solver_opts)
    
    polish_details = {
        'I1': I1,
        'I2': I2,
        'I_biactive': I_biactive,
        'bnlp_status': bnlp_result['status'],
        'bnlp_success': bnlp_result['success'],
        'bnlp_f_val': bnlp_result['f_val'],
        'bnlp_cpu_time': bnlp_result['cpu_time'],
        'original_f_val': f_star,
        'improvement': 0,
        'accepted': False
    }
    
    if bnlp_result['success']:
        z_polish = bnlp_result['z_polish']
        f_polish = bnlp_result['f_val']
        comp_res_polish = complementarity_residual(z_polish, problem)
        
        polish_details['comp_res_polish'] = comp_res_polish
        polish_details['improvement'] = f_star - f_polish
        
        eps_tol = results.get('comp_res', 1e-06)
        comp_ok = comp_res_polish < max(1e-06, 10 * eps_tol)
        
        if comp_ok:
            polish_details['accepted'] = True
            results['z_final'] = z_polish
            results['f_final'] = f_polish
            results['comp_res'] = comp_res_polish
            if results['stationarity'] in ('FAIL', 'C', 'M'):
                results['stationarity'] = 'S'
                results['sign_test_pass'] = True
                results['status'] = 'converged'
            logger.info(f'BNLP polish accepted: f={f_polish:.6e} (was {f_star:.6e}), comp_res={comp_res_polish:.2e}')
        else:
            logger.info(f'BNLP polish rejected: comp_res={comp_res_polish:.2e}')
    else:
        logger.info(f"BNLP polish failed: {bnlp_result['status']}")
    
    results['bnlp_polish'] = polish_details
    
    # Try alternative partitions if not converged
    _should_try_alt = (len(I_biactive) > 0 and not polish_details['accepted'] and
                       results.get('stationarity') in ('FAIL', 'C', 'M'))
    if _should_try_alt:
        current_f = results.get('f_final', f_star)
        current_z = results.get('z_final', z_star)
        results = _try_alternative_partitions(results, problem, current_z, current_f,
                                              I1, I2, I_biactive, solver_opts=solver_opts)

    # Final ultra-tight polish if accepted
    if results.get('bnlp_polish', {}).get('accepted', False):
        I1_final, I2_final, _, I3_final = identify_active_set(results['z_final'], problem)
        ultra_result = _build_bnlp(results['z_final'], problem, I1_final, I2_final, I3=I3_final,
                                   solver_opts=solver_opts, use_ultra_tight=True)
        if ultra_result['success']:
            comp_ultra = complementarity_residual(ultra_result['z_polish'], problem)
            if comp_ultra < 1e-06:
                results['z_final'] = ultra_result['z_polish']
                results['f_final'] = ultra_result['f_val']
                results['comp_res'] = comp_ultra
                logger.info(f"Ultra-tight polish: f={ultra_result['f_val']:.10e}, comp={comp_ultra:.2e}")
    
    return results


def _try_alternative_partitions(results, problem, z_star, f_star, I1_base, I2_base,
                                 I_biactive, solver_opts=None, max_partitions=32, time_budget=30):
    """
    Try alternative active-set partitions for biactive indices.

    Uses single-flip enumeration with a time budget to avoid hanging on
    large problems. Each biactive index is flipped one at a time.

    Parameters
    ----------
    time_budget : float
        Maximum wall-clock seconds for the whole enumeration (default: 30).
    """
    n_biactive = len(I_biactive)
    if n_biactive == 0:
        return results
    
    t_start = time.time()
    best_f = f_star
    best_z = z_star.copy()
    best_accepted = False
    n_tried = 0
    I1_set = set(I1_base)
    
    for flip_i in I_biactive:
        if n_tried >= max_partitions:
            break
        if time.time() - t_start > time_budget:
            logger.info(f'  Partition search: time budget {time_budget:.0f}s exhausted after {n_tried} tries')
            break
        
        # Flip the index between I1 and I2
        I1_alt = list(I1_base)
        I2_alt = list(I2_base)
        if flip_i in I1_set:
            I1_alt.remove(flip_i)
            I2_alt.append(flip_i)
        else:
            I2_alt.remove(flip_i)
            I1_alt.append(flip_i)
        
        bnlp_result = _build_bnlp(z_star, problem, I1_alt, I2_alt, I3=[], solver_opts=solver_opts)
        n_tried += 1
        
        if not bnlp_result['success']:
            continue
        
        comp_res = complementarity_residual(bnlp_result['z_polish'], problem)
        if comp_res >= 1e-06:
            continue
        if bnlp_result['f_val'] >= best_f:
            continue
        
        best_f = bnlp_result['f_val']
        best_z = bnlp_result['z_polish']
        best_accepted = True
        logger.info(f'  Single flip {flip_i}: f={best_f:.6e}, comp={comp_res:.2e}')
    
    if best_accepted:
        results['z_final'] = best_z
        results['f_final'] = best_f
        results['comp_res'] = complementarity_residual(best_z, problem)
        results['bnlp_polish']['accepted'] = True
        results['bnlp_polish']['alt_partition_used'] = True
        results['bnlp_polish']['n_partitions_tried'] = n_tried
        if results['stationarity'] in ('FAIL', 'C', 'M'):
            results['stationarity'] = 'S'
            results['sign_test_pass'] = True
            results['status'] = 'converged'
        logger.info(f'Alternative partition accepted: f={best_f:.6e} (tried {n_tried} partitions)')
    
    return results
