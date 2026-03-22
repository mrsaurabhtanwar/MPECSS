"""
Utility functions for MPECSS: logging setup, KKT/complementarity metrics,
multiplier extraction, and CSV export/import.
"""

import logging
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import casadi as ca


def setup_logger(name='mpecss', level=logging.INFO, logfile=None) -> logging.Logger:
    """Configure and return a named logger, optionally mirroring to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
        if logfile:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    
    return logger


@dataclass
class IterationLog:
    """Per-outer-iteration data container (one row per homotopy step)."""
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
    Slice lam_g into (lambda_G, lambda_H, lambda_comp) using the constraint
    layout produced by build_casadi.

    Standard NCP layout (n_ubH == 0):
        [n_orig_con | n_bounded_G | n_comp (H>=0) | n_comp (comp)]

    Box-MCP layout (n_ubH > 0, e.g. gnash*m):
        [n_orig_con | 0 | n_comp (H>=0) | n_ubH (H<=ubH) | n_ubH (upper comp) | n_comp (lower comp)]

    If build_casadi exposes 'off_G_lb', 'off_H_lb', 'off_comp' offsets the
    exact positions are used directly.  Otherwise falls back to the legacy
    NCP heuristic so existing callers without the new fields still work.

    Sign convention: lambda_G/H = -lam_g_casadi (CasADi lam <= 0 at active
    lower bound → MPCC convention is non-negative). lambda_comp keeps CasADi sign.

    Returns
    -------
    lambda_G, lambda_H, lambda_comp : np.ndarray (n_comp,) each
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


def compute_kkt_residual(x_val, f_sym, g_sym, x_sym, lam_g, lam_x, p_sym=None, p_val=None) -> float:
    """
    Compute ||∇_x L||_inf where L = f + lam_g^T g (+ lam_x^T x).

    Parameters
    ----------
    x_val : np.ndarray
    f_sym, g_sym, x_sym : ca.SX symbolic expressions
    lam_g : np.ndarray
    lam_x : np.ndarray or None
    p_sym : ca.SX or None — symbolic parameters (e.g. [t_sym, d_sym])
    p_val : list or None — concrete parameter values

    Returns
    -------
    float
    """
    x_val = np.asarray(x_val).flatten()
    lam_g = np.asarray(lam_g).flatten()
    
    grad_f = ca.jacobian(f_sym, x_sym).T
    J_g = ca.jacobian(g_sym, x_sym)
    L_grad_expr = grad_f + ca.mtimes(J_g.T, ca.DM(lam_g))
    
    if lam_x is not None:
        lam_x = np.asarray(lam_x).flatten()
        L_grad_expr = L_grad_expr + ca.DM(lam_x)
    
    inputs = [x_sym]
    if p_sym is not None:
        inputs.append(p_sym)
    
    grad_L_fn = ca.Function('grad_L', inputs, [L_grad_expr])
    
    if p_val is not None:
        grad_L_val = grad_L_fn(x_val, p_val)
    else:
        grad_L_val = grad_L_fn(x_val)
    
    return float(np.max(np.abs(np.asarray(grad_L_val).flatten())))


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


def export_run_summary(results: List[dict], filepath: str):
    """Export aggregated run summary dicts to CSV."""
    import pandas as pd
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)


def load_csv(filepath: str):
    """Load a CSV results file into a DataFrame."""
    import pandas as pd
    return pd.read_csv(filepath)


def aggregate_summary(df):
    """
    Compute mean/median/std for numeric columns and stationarity type counts.

    Returns
    -------
    pd.Series
    """
    import pandas as pd
    
    numeric = df.select_dtypes(include=[np.number])
    stats = {}
    
    for col in numeric.columns:
        stats[col + '_mean'] = numeric[col].mean()
        stats[col + '_median'] = numeric[col].median()
        stats[col + '_std'] = numeric[col].std()
    
    if 'stationarity' in df.columns:
        for stype in ('B', 'S', 'M', 'C', 'W', 'FAIL'):
            stats['stat_' + stype + '_count'] = int((df['stationarity'] == stype).sum())
    
    return pd.Series(stats)
