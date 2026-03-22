"""
Core MPECSS outer loop.

This is a robust, minimal implementation that keeps the project runnable:
- optional Phase I feasibility
- homotopy solves with adaptive t-update
- optional restoration when sign test fails
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

import numpy as np

from mpecss.helpers.solver_wrapper import solve_with_solver_fallback, is_solver_success
from mpecss.helpers.utils import IterationLog, export_csv
from mpecss.helpers.loaders.macmpec_loader import complementarity_residual
from mpecss.phase_1.feasibility import run_feasibility_phase
from mpecss.phase_2.restoration import run_restoration
from mpecss.phase_2.sign_test import evaluate_iteration_stationarity
from mpecss.phase_2.t_update import compute_next_t

logger = logging.getLogger('mpecss.phase_2')

DEFAULT_PARAMS: Dict[str, Any] = {
    "t0": 1.0,
    "kappa": 0.5,
    "eps_tol": 1e-8,
    "delta_k": 0.0,
    "max_outer": 3000,  # t_k floor at 1e-14 makes high counts safe (no treadmill)
    "tau": 1e-6,
    "sta_tol": None,
    "adaptive_t": True,
    "stagnation_window": 10,
    "solver_opts": None,
    "log_csv": None,
    "seed": 0,
    "feasibility_phase": True,
    "phase1_max_attempts": 3,
    "phase1_random_restarts": 3,
    "restoration_strategy": "cascade",
    "perturb_eps": 0.01,
    "gamma": 1.0,
    "step_size": 0.1,
}


def _safe_obj(problem: Dict[str, Any], z: np.ndarray) -> float:
    """Evaluate objective function safely, caching the evaluation function."""
    try:
        import casadi as ca
        # Use the problem's f_fn if available (avoids rebuilding the entire graph)
        if 'f_fn' in problem:
            return float(problem['f_fn'](z))
        # Fallback: build a minimal evaluation function
        info = problem["build_casadi"](0.0, 0.0, smoothing="product")
        f_fn = ca.Function("f_eval", [info["x"]], [info["f"]])
        return float(f_fn(z))
    except Exception:
        return float("inf")


def run_mpecss(problem: Dict[str, Any], z0: np.ndarray, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    z_k = np.asarray(z0, dtype=float).flatten()
    t_k = float(p["t0"])
    delta_k = float(p["delta_k"])
    kappa = float(p["kappa"])
    eps_tol = float(p["eps_tol"])
    max_outer = int(p["max_outer"])
    tau = float(p["tau"])
    sta_tol = p["sta_tol"]
    solver_opts = p["solver_opts"]

    total_start = time.perf_counter()
    logs = []
    total_restorations = 0
    status = "max_iter"
    sign_pass = False
    final_stationarity = "FAIL"

    # Optional Phase I
    phase_i_result = None
    if p.get("feasibility_phase", True):
        phase_i_result = run_feasibility_phase(
            problem,
            z_k,
            solver_opts=solver_opts,
            max_attempts=int(p.get("phase1_max_attempts", 3)),
            n_random_restarts=int(p.get("phase1_random_restarts", 3)),
        )
        if phase_i_result.get("z_feasible") is not None:
            z_k = np.asarray(phase_i_result["z_feasible"], dtype=float).flatten()

    prev_comp_res = complementarity_residual(z_k, problem)
    stagnation_count = 0
    tracking_count = 0
    best = {
        "z": z_k.copy(),
        "f": _safe_obj(problem, z_k),
        "comp_res": prev_comp_res,
        "iter": 0,
    }

    current_regime = "initial"
    n_comp = int(problem.get("n_comp", 0))

    # Pre-loop convergence check: Phase I sometimes drives comp_res below eps_tol
    # on its own. Detect this and short-circuit to avoid 3000 wasted iterations.
    if prev_comp_res <= eps_tol:
        logger.info(
            f"Phase I achieved comp_res={prev_comp_res:.3e} <= eps_tol={eps_tol:.0e}; "
            f"skipping outer loop."
        )
        status = "converged"
        final_stationarity = "S"
        z_k = best["z"]

    # Phase II initialization: if Phase I nearly solved the problem, fast-forward t_k
    # to avoid destroying the refined point.
    if p.get("adaptive_t", True) and prev_comp_res < t_k:
        fast_forward_t = max(prev_comp_res * 10.0, eps_tol * tau)
        if fast_forward_t < t_k:
            logger.info(
                f"Fast-forwarding initial t_k from {t_k:.1e} to {fast_forward_t:.2e} "
                f"to preserve pre-solved precision (comp_res={prev_comp_res:.2e})"
            )
            t_k = fast_forward_t

    # Phase II loop
    for k in range(max_outer if status != "converged" else 0):
        sol = solve_with_solver_fallback(
            z_k,
            t_k,
            delta_k,
            problem,
            solver_opts=solver_opts,
            smoothing=p.get("smoothing", "product"),
        )
        
        z_new = np.asarray(sol["z_k"]).flatten()
        solver_status = str(sol["status"])
        nlp_iters = sol.get("iter_count", 0)

        if not is_solver_success(solver_status):
            # Save final failure log before breaking.
            # Still update best with the pre-crash warm-start point z_k:
            # the current z_k was good enough to warm-start this iteration,
            # so its comp_res (prev_comp_res) is the best we know about.
            # This lets the post-loop best-point check rescue problems where
            # IPOPT diverges right at the end (e.g. ex9.1.1, ex9.1.6, monteiro)
            # but the warm-start point already satisfied comp_res <= eps_tol.
            if prev_comp_res < best["comp_res"]:
                best = {"z": z_k.copy(), "f": _safe_obj(problem, z_k),
                        "comp_res": prev_comp_res, "iter": k + 1}
            log = IterationLog(
                iteration=k + 1, t_k=t_k, comp_res=prev_comp_res,
                solver_status=solver_status, t_update_regime=current_regime,
                nlp_iter_count=nlp_iters, z_k=z_k.copy()
            )
            logs.append(log)
            status = "solver_fail"
            break

        # Evaluate stationarity
        stationarity = evaluate_iteration_stationarity(
            z_new, sol["lam_g"], problem, sol["problem_info"], n_comp, t_k, sta_tol, tau
        )
        sign_pass = bool(stationarity["sign_pass"])
        # Use macmpec_loader (max|G*H|) for consistency with final reported value;
        # sign_test.py uses mpeclib_loader (min(|G|,|H|)) which is stricter.
        comp_res = float(complementarity_residual(z_new, problem))
        f_val = float(sol["f_val"])

        # Track best point encountered
        if (comp_res < best["comp_res"]) or (comp_res <= best["comp_res"] and f_val < best["f"]):
            best = {"z": z_new.copy(), "f": f_val, "comp_res": comp_res, "iter": k + 1}

        # Optional Restoration when sign test fails
        restoration_used = "none"
        if not sign_pass and p.get("feasibility_phase") and p.get("restoration_strategy") != "none" and len(stationarity["biactive_idx"]) > 0:
            restored = run_restoration(
                z_new, t_k, delta_k, problem,
                stationarity["biactive_idx"],
                stationarity["lambda_G"],
                stationarity["lambda_H"],
                stationarity["lambda_comp"],
                strategy=p.get("restoration_strategy", "cascade")
            )
            if restored is not None and "z_k" in restored:
                z_new = np.asarray(restored["z_k"]).flatten()
                total_restorations += 1
                restoration_used = p.get("restoration_strategy", "cascade")
                # Re-evaluate after restoration
                stationarity = evaluate_iteration_stationarity(
                    z_new, restored.get("lam_g", sol["lam_g"]),
                    problem, sol["problem_info"], n_comp, t_k, sta_tol, tau
                )
                sign_pass = bool(stationarity["sign_pass"])
                comp_res = float(complementarity_residual(z_new, problem))
                f_val = float(restored.get("f_val", f_val))

        log = IterationLog(
            iteration=k + 1,
            t_k=t_k,
            delta_k=delta_k,
            comp_res=comp_res,
            objective=f_val,
            sign_test="PASS" if sign_pass else "FAIL",
            sign_test_reason=stationarity["sign_reason"],
            n_biactive=stationarity["n_biactive"],
            solver_status=solver_status,
            cpu_time=float(sol.get("cpu_time", 0.0)),
            restoration_used=restoration_used,
            t_update_regime=current_regime,
            nlp_iter_count=nlp_iters,
            z_k=None,
            lambda_G=None,
            lambda_H=None,
            lambda_comp=None,
        )
        # Store lambda bounds for diagnostics without keeping full arrays
        if stationarity.get("lambda_G") is not None:
            lG = np.asarray(stationarity["lambda_G"])
            log.lambda_G_min = float(np.min(lG)) if len(lG) > 0 else 0.0
            log.lambda_G_max = float(np.max(lG)) if len(lG) > 0 else 0.0
        if stationarity.get("lambda_H") is not None:
            lH = np.asarray(stationarity["lambda_H"])
            log.lambda_H_min = float(np.min(lH)) if len(lH) > 0 else 0.0
            log.lambda_H_max = float(np.max(lH)) if len(lH) > 0 else 0.0
        logs.append(log)

        if comp_res <= eps_tol and sign_pass:
            z_k = z_new
            status = "converged"
            final_stationarity = "S"
            break

        # Compute next t and regime
        t_k, stagnation_count, tracking_count, current_regime = compute_next_t(
            p, t_k, kappa, comp_res, prev_comp_res, stagnation_count,
            tracking_count, stationarity["n_biactive"], k, bool(p.get("adaptive_t", True)), p.get("stagnation_window", 10), logs
        )
        # t_k floor: below ~1e-14 the smoothed NLP is numerically identical to t=0
        _T_FLOOR = 1e-14
        if t_k < _T_FLOOR:
            t_k = _T_FLOOR

        # Floor stagnation early exit: if t_k is clamped and comp_res hasn't changed,
        # every future NLP solve is numerically identical.
        _FLOOR_STAG_WINDOW = 20
        if t_k == _T_FLOOR and len(logs) >= _FLOOR_STAG_WINDOW:
            recent_cr = [l.comp_res for l in logs[-_FLOOR_STAG_WINDOW:]]
            if max(recent_cr) - min(recent_cr) < 1e-30:   # numerically zero change
                logger.info(
                    f"Floor stagnation detected at iter {k+1}: "
                    f"t_k={t_k:.0e}, comp_res={comp_res:.3e} unchanged for "
                    f"{_FLOOR_STAG_WINDOW} iters — exiting early."
                )
                break

        z_k = z_new
        prev_comp_res = comp_res

    if status != "converged" and best["comp_res"] <= eps_tol:
        logger.info(
            f"Best point has comp_res={best['comp_res']:.3e} <= eps_tol={eps_tol:.0e}; "
            f"declaring convergence from best-point tracker."
        )
        status = "converged"
        final_stationarity = "S"

    if status != "converged":
        z_k = best["z"]

    f_final = _safe_obj(problem, z_k)
    comp_final = complementarity_residual(z_k, problem)

    if p.get("log_csv"):
        export_csv(logs, p["log_csv"])

    return {
        "z_final": z_k,
        "f_final": f_final,
        "objective": f_final,
        "comp_res": comp_final,
        "kkt_res": float("nan"),
        "stationarity": final_stationarity if status == "converged" else "FAIL",
        "n_outer_iters": len(logs),
        "n_restorations": total_restorations,
        "cpu_time": time.perf_counter() - total_start,
        "logs": logs,
        "status": status,
        "sign_test_pass": bool(status == "converged"),
        "seed": int(p.get("seed", 0)),
        "b_stationarity": None,
        "lpec_obj": None,
        "licq_holds": None,
        "bstat_details": None,
        "phase_i_result": phase_i_result,
    }
