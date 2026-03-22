"""
Adaptive smoothing-parameter update policy for Phase II.

This module centralizes the four-track t-update logic so `mpecss.py` remains
an orchestrator.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

_TRACKING_UPPER = 5.0
_TRACKING_LOWER = 0.2
_TRACKING_MIN_COUNT = 2
_TRACKING_MIN_ITER = 4
_STAGNATION_THRESHOLD = 0.05


def compute_next_t(p, t_k, kappa, comp_res, prev_comp_res, stagnation_count, tracking_count, 
                   n_biactive, k, adaptive_t, stagnation_window, logs):
    """
    Compute next smoothing parameter t_{k+1} and regime label.

    Returns
    -------
    t_next, stagnation_count, tracking_count, regime
    """
    # Compute improvement ratio
    denom = max(prev_comp_res, np.finfo(float).tiny)
    improvement = (prev_comp_res - comp_res) / denom
    
    # Get steering mode
    steering = p.get('steering', 'adaptive')
    
    # Handle non-adaptive modes
    if steering == 'linf':
        regime = 'linf'
        t_new = kappa * t_k
        return t_new, stagnation_count, tracking_count, regime
    
    if np.isnan(comp_res) or np.isnan(prev_comp_res):
        regime = 'linf_fallback'
        t_new = kappa * t_k
        return t_new, stagnation_count, tracking_count, regime
    
    # Check for stagnation
    if stagnation_window is None:
        stagnation_window = 10
    
    if improvement < _STAGNATION_THRESHOLD:
        stagnation_count += 1
    else:
        stagnation_count = 0
    
    # Check if biactive set is stable over recent iterations
    biactive_stable = False
    if len(logs) >= 3:
        biactive_stable = all(
            logs[i].n_biactive == n_biactive 
            for i in range(max(0, len(logs) - 3), len(logs))
        )
    
    # Tracking mode: activated when biactive set is stable and improving
    tracking = False
    if adaptive_t and biactive_stable and k >= _TRACKING_MIN_ITER:
        if _TRACKING_LOWER <= improvement <= _TRACKING_UPPER:
            tracking_count += 1
            if tracking_count >= _TRACKING_MIN_COUNT:
                tracking = True
        else:
            tracking_count = max(0, tracking_count - 1)
    
    # Determine update regime
    if tracking and improvement > 0.5:
        regime = 'superlinear'
        t_new = kappa * kappa * t_k
    elif tracking and improvement > 0.1:
        regime = 'fast'
        t_new = min(kappa, 0.5) * t_k
    elif stagnation_count >= 4:
        regime = 'adaptive_jump'
        t_new = 2 * kappa * t_k  # larger step to escape stagnation
        stagnation_count = 0
    elif stagnation_count >= 2:
        regime = 'post_stagnation_fast'
        t_new = min(kappa, 0.5) * t_k
    else:
        regime = 'slow'
        t_new = kappa * t_k
    
    return t_new, stagnation_count, tracking_count, regime
