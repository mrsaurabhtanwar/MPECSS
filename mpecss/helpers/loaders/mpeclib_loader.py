"""
The "Problem Translator" (MPECLib): Turning libraries into math.

MPECLib is another big library of complementarity problems. 
This module reads the specialized JSON files and prepares 
them for the MPECSS solver.
"""
import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

logger = logging.getLogger('mpecss.problems')

_BIG = 1e20
_X0_PERTURBATION = 0.01

# FIX #7: was annotated as `str`; correct type is List[Tuple[str, str]]
_FAMILY_PATTERNS: List[Tuple[str, str]] = [
    ('^aampec', 'aampec'),
    ('^bard', 'bard'),
    ('^bartruss', 'bartruss'),
    ('^dempe', 'dempe'),
    ('^desilva', 'desilva'),
    ('^ex9_', 'ex9'),
    ('^finda', 'find_a'),
    ('^findb', 'find_b'),
    ('^findc', 'find_c'),
    ('^fjq', 'fjq'),
    ('^frictionalblock', 'frictional_block'),
    ('^gauvin', 'gauvin'),
    ('^hq', 'hq'),
    ('^kehoe', 'kehoe'),
    ('^kojshin', 'kojshin'),
    ('^mss', 'mss'),
    ('^nappi', 'nappi'),
    ('^outrata3', 'outrata'),
    ('^oz', 'oz'),
    ('^qvi', 'qvi'),
    ('^three', 'three'),
    ('^tinloi', 'tinloi'),
    ('^tinque', 'tinque'),
]


def _detect_family(problem_name: str) -> str:
    name_lower = problem_name.lower()
    for pattern, family in _FAMILY_PATTERNS:
        if re.match(pattern, name_lower):
            return family
    return 'mpeclib'


def _sanitize_bound(value: float, default: float) -> float:
    if value is None:
        return default
    value = float(value)
    if value < -1e19:
        return -_BIG
    if value > 1e19:
        return _BIG
    return value


def _sanitize_bounds(values: list, default: float) -> list:
    return [_sanitize_bound(v, default) for v in values]


def _as_list(value, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def _load_original_constraints(data: dict):
    g_fn = data.get('g_fun')
    if g_fn is None:
        return None, [], [], 0
    g_fn = ca.Function.deserialize(g_fn)
    raw_lbg = _as_list(data.get('lbg'), [-_BIG])
    raw_ubg = _as_list(data.get('ubg'), [_BIG])
    n_con = len(raw_lbg)
    lbg = _sanitize_bounds(raw_lbg, -_BIG)
    ubg = _sanitize_bounds(raw_ubg, _BIG)
    return g_fn, lbg, ubg, n_con


def _load_complementarity_bounds(data: dict):
    raw_lbG = data.get('lbG')
    if isinstance(raw_lbG, (int, float)):
        lbG = [raw_lbG]
    else:
        lbG = list(raw_lbG) if raw_lbG else [0.0]
    ubG = _sanitize_bounds(data.get('ubG', [_BIG]), _BIG)
    lbH = _sanitize_bounds(data.get('lbH', [0.0]), 0.0)
    ubH = _sanitize_bounds(data.get('ubH', [_BIG]), _BIG)
    return lbG, ubG, lbH, ubH


def _tighten_linear_bounds(problem_name, n_x, n_comp, w0, lbx, G_fn, H_fn, lbG_eff, lbH_eff):
    """Try to infer tighter variable lower bounds from simple linear rows in G/H."""
    if n_x > 2000:
        return 0
    tightened = 0
    x_sym = ca.MX.sym('xbt', n_x)

    for label, fn, eff_lb in [('H', H_fn, lbH_eff), ('G', G_fn, lbG_eff)]:
        expr = fn(x_sym)
        jac_expr = ca.jacobian(expr, x_sym)
        jac_fn = ca.Function('J_' + label, [x_sym], [jac_expr])
        jac_at_w0 = np.asarray(ca.DM(jac_fn(w0))).flatten()
        val_at_w0 = np.asarray(ca.DM(fn(w0))).flatten()

        for j in range(n_comp):
            row = jac_at_w0[j * n_x:(j + 1) * n_x] if jac_at_w0.ndim == 1 else jac_at_w0[j]
            nz_idx = np.where(np.abs(row) > 1e-10)[0]
            if len(nz_idx) != 1:
                continue
            i = int(nz_idx[0])
            w_pert = w0.copy()
            w_pert[i] += 1.0
            jac_pert = np.asarray(ca.DM(jac_fn(w_pert))).flatten()
            if np.abs(jac_pert[j * n_x + i] - row[i]) > 1e-8:
                continue
            const_term = val_at_w0[j] - row[i] * w0[i]
            new_lb = (eff_lb[j] - const_term) / row[i] if row[i] > 0 else -1e10
            if new_lb > lbx[i] + 1e-8:
                logger.info('%s: tightened lbx[%d] to %.4g (from %s[%d] >= %.4g)',
                            problem_name, i, new_lb, label, j, eff_lb[j])
                lbx[i] = new_lb
                tightened += 1
    return tightened


def load_mpeclib(filepath: str) -> Dict[str, Any]:
    """
    Load one MPECLib problem from a .nl.json file.

    Returns a problem dict compatible with all MPECSS phases:
      name, n_x, n_comp, n_con, n_p, family
      x0_fn(seed) -> np.ndarray
      build_casadi(t_k, delta_k, smoothing) -> NLP subproblem dict
      G_fn, H_fn
      lbx, ubx
      lbG_eff, lbH_eff
      _source_path
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('MPECLib benchmark file not found: ' + filepath)

    with open(filepath) as f:
        data = json.load(f)

    problem_name = os.path.basename(filepath).replace('.nl.json', '')

    lbx = _sanitize_bounds(data.get('lbw', []), -_BIG)
    raw_ubx = data.get('ubw') or data.get('Ubw')
    ubx = _sanitize_bounds(raw_ubx, _BIG)
    w0 = np.array(data.get('w0', []), dtype=float)
    n_x = len(lbx)

    f_fn = ca.Function.deserialize(data.get('f_fun'))
    G_fn = ca.Function.deserialize(data.get('G_fun'))
    H_fn = ca.Function.deserialize(data.get('H_fun'))

    g_fn, lbg_orig, ubg_orig, n_con = _load_original_constraints(data)
    lbG_raw, ubG_raw, lbH_raw, ubH_raw = _load_complementarity_bounds(data)

    n_comp = len(lbG_raw)
    if n_comp == 0:
        raise ValueError('Problem ' + problem_name + ' has no complementarity pairs')

    lbG_eff = _sanitize_bounds(lbG_raw, 0.0)
    lbH_eff = _sanitize_bounds(lbH_raw, 0.0)
    ubG_fin = _sanitize_bounds(ubG_raw, _BIG)
    ubH_fin = _sanitize_bounds(ubH_raw, _BIG)

    has_nonstandard = max(lbH_eff) > 0.0 or any(v < _BIG for v in ubG_fin) or any(v < _BIG for v in ubH_fin)
    if has_nonstandard:
        logger.info('%s: non-standard comp bounds lbH_eff=%s, ubG_fin=%s, ubH_fin=%s',
                    problem_name, lbH_eff, ubG_fin, ubH_fin)

    n_tightened = _tighten_linear_bounds(problem_name, n_x, n_comp, w0, lbx, G_fn, H_fn, lbG_eff, lbH_eff)
    logger.debug('Loaded %s: n_x=%d, n_comp=%d, n_con=%d%s', problem_name, n_x, n_comp, n_con,
                 ', tightened ' + str(n_tightened) + ' bounds' if n_tightened else '')

    family = _detect_family(problem_name)

    def x0_fn(seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        x0 = w0.copy()
        x0 += rng.uniform(-_X0_PERTURBATION, _X0_PERTURBATION, size=x0.shape)
        lb = np.array(lbx)
        ub = np.array(ubx)
        clip_lb = np.where(lb > -_BIG, lb, -np.inf)
        clip_ub = np.where(ub < _BIG, ub, np.inf)
        return np.clip(x0, clip_lb + 1e-8, clip_ub - 1e-8)

    def build_casadi(t_k: float, delta_k: float, smoothing: str = 'product') -> Dict[str, Any]:
        symbol_type = ca.MX if n_x > 500 else ca.SX
        x = symbol_type.sym('x', n_x)
        f = f_fn(x)

        G_expr = G_fn(x)
        H_expr = H_fn(x)
        G_shifted = G_expr - ca.DM(lbG_eff)
        H_shifted = H_expr - ca.DM(lbH_eff)

        g_parts = []
        lbg_parts = []
        ubg_parts = []

        if g_fn is not None:
            g_parts.append(g_fn(x))
            lbg_parts.extend(lbg_orig)
            ubg_parts.extend(ubg_orig)

        g_parts.append(G_shifted + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)

        g_parts.append(H_shifted + delta_k)
        lbg_parts.extend([0.0] * n_comp)
        ubg_parts.extend([_BIG] * n_comp)

        if smoothing == 'fb':
            comp = ca.sqrt(G_shifted**2 + H_shifted**2) - G_shifted - H_shifted - t_k
            g_parts.append(comp)
            lbg_parts.extend([-_BIG] * n_comp)
            ubg_parts.extend([0.0] * n_comp)
        else:
            for i in range(n_comp):
                g_parts.append(G_shifted[i] * H_shifted[i] - t_k)
                lbg_parts.append(-_BIG)
                ubg_parts.append(0.0)

        g = ca.vertcat(*g_parts)

        # Layout offsets for extract_multipliers (standard NCP only for mpeclib):
        #   [n_orig_con | n_comp (G-lb) | n_comp (H-lb) | n_comp (comp)]
        off_G_lb = n_con
        off_H_lb = off_G_lb + n_comp
        off_comp = off_H_lb + n_comp

        return {
            'x': x, 'f': f, 'g': g,
            'lbg': lbg_parts, 'ubg': ubg_parts,
            'lbx': lbx, 'ubx': ubx,
            'n_comp': n_comp, 'n_orig_con': n_con,
            # Layout offsets:
            'n_bounded_G': n_comp,
            'n_ubH':       0,
            'off_G_lb':    off_G_lb,
            'off_H_lb':    off_H_lb,
            'off_comp':    off_comp,
        }

    return {
        'name': problem_name,
        'n_x': n_x,
        'n_comp': n_comp,
        'n_con': n_con,
        'n_p': 0,
        'family': family,
        'x0_fn': x0_fn,
        'build_casadi': build_casadi,
        'G_fn': G_fn,
        'H_fn': H_fn,
        'lbx': lbx,
        'ubx': ubx,
        'lbG_eff': lbG_eff,
        'lbH_eff': lbH_eff,
        '_source_path': os.path.abspath(filepath),
    }


def load_mpeclib_batch(directory: str, pattern: str = '*.json') -> List[Dict[str, Any]]:
    """Load all MPECLib problems from a directory."""
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    problems = []
    for file_path in files:
        try:
            problems.append(load_mpeclib(file_path))
        except Exception as exc:
            logger.warning('Failed to load %s: %s', file_path, exc)
    logger.info('Loaded %d MPECLib problems from %s', len(problems), directory)
    return problems


def get_mpeclib_problem(name: str, mpeclib_dir: str = None) -> Dict[str, Any]:
    """
    Get one MPECLib problem by full path or by benchmark name.

    Examples
    --------
    get_mpeclib_problem("bard1")
    get_mpeclib_problem("bard1.nl.json")
    get_mpeclib_problem("/abs/path/to/bard1.nl.json")
    """
    if os.path.isfile(name):
        return load_mpeclib(name)

    if mpeclib_dir is None:
        mpeclib_dir = os.path.join('benchmarks', 'mpeclib', 'mpeclib-json')

    candidate = name if name.endswith('.nl.json') else name + '.nl.json'
    file_path = os.path.join(mpeclib_dir, candidate)

    if os.path.isfile(file_path):
        return load_mpeclib(file_path)

    raise FileNotFoundError(
        f"Problem '{name}' not found in '{mpeclib_dir}'. Tried: {file_path}"
    )


def evaluate_GH(x: np.ndarray, problem: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate G(x) and H(x) for a loaded MPECLib problem."""
    G = np.asarray(problem['G_fn'](x)).flatten()
    H = np.asarray(problem['H_fn'](x)).flatten()
    return G, H


def complementarity_residual(x: np.ndarray, problem: Dict[str, Any]) -> float:
    """Compute complementarity residual, box-MCP-aware.

    Standard NCP:
        max_i  min( |G_shifted_i|, |H_shifted_i| )

    Box-MCP (finite ubH_i, e.g. gnash*m via macmpec_loader):
        max_i  min( |G_shifted_i * H_shifted_i|,
                    |(-G_shifted_i) * (ubH_i - H_shifted_i)| )

    Correctly returns 0 at upper-bound active solutions (H_i = ubH_i, G_i <= 0).
    """
    G, H = evaluate_GH(x, problem)
    if len(G) == 0:
        return 0.0

    lbG_eff = np.array(problem.get('lbG_eff', np.zeros(len(G))), dtype=float)
    lbH_eff = np.array(problem.get('lbH_eff', np.zeros(len(H))), dtype=float)

    G_shifted = G - lbG_eff
    H_shifted = H - lbH_eff

    ubH_finite = problem.get('ubH_finite', [])   # [(idx, ub_val), ...]
    if not ubH_finite:
        # Standard NCP — original formula unchanged
        return float(np.max(np.minimum(np.abs(G_shifted), np.abs(H_shifted))))

    # Box-MCP: use product formula with min of both complementarity pairs
    residuals = np.minimum(np.abs(G_shifted), np.abs(H_shifted)).copy()
    for i, ub in ubH_finite:
        lower = abs(float(G_shifted[i]) * float(H_shifted[i]))
        upper = abs((-float(G_shifted[i])) * (ub - float(H_shifted[i])))
        residuals[i] = min(lower, upper)
    return float(np.max(residuals))


def biactive_indices(x: np.ndarray, problem: Dict[str, Any], tol: float = 1e-6) -> List[int]:
    """Return indices where both |G_i| < tol and |H_i| < tol."""
    G, H = evaluate_GH(x, problem)
    mask = (np.abs(G) < tol) & (np.abs(H) < tol)
    return list(np.where(mask)[0])
