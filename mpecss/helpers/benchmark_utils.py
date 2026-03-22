import os
import gc
import time
import queue as _queue_module
import logging
import argparse
import signal
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import pandas as pd
import sys
import platform
import subprocess
import json

try:
    import psutil
except ImportError:
    psutil = None


from mpecss.phase_2.mpecss import run_mpecss, DEFAULT_PARAMS
from mpecss.helpers.utils import IterationLog, export_csv

# Phase III imports
from mpecss.phase_3.bnlp_polish import bnlp_polish
from mpecss.phase_3.lpec_refine import lpec_refinement_loop
from mpecss.phase_3.bstationarity import bstat_post_check

logger = logging.getLogger("mpecss.benchmark")

# Define the full set of columns matching the official CSV
OFFICIAL_COLUMNS = [
    "benchmark_suite", "problem_file", "run_timestamp", "seed", "wall_timeout_cfg", "problem_name",
    "n_x", "n_comp", "n_con", "n_p", "family", "problem_size_mode",
    "cfg_t0", "cfg_kappa", "cfg_eps_tol", "cfg_delta_policy", "cfg_delta_k", "cfg_delta_factor",
    "cfg_delta0", "cfg_kappa_delta", "cfg_tau", "cfg_sta_tol", "cfg_max_outer", "cfg_max_restoration",
    "cfg_restoration_strategy", "cfg_perturb_eps", "cfg_gamma", "cfg_step_size", "cfg_smoothing",
    "cfg_adaptive_t", "cfg_steering", "cfg_stagnation_window", "cfg_adaptive_ipopt_tol",
    "cfg_feasibility_phase", "cfg_bstat_check", "cfg_lpec_refine", "cfg_fb_auto_retry",
    "cfg_solver_fallback", "cfg_skip_redundant_postsolve", "cfg_early_stag_window",
    "cfg_early_stag_threshold", "cfg_early_stag_floor", "cfg_k1_max_nlp_calls", "cfg_max_stag_recoveries",
    "status", "stationarity", "f_final", "comp_res", "kkt_res", "sign_test_pass", "b_stationarity",
    "lpec_obj", "licq_holds", "n_outer_iters", "n_restorations", "cpu_time_total",
    "smoothing_actually_used", "fb_auto_retry_triggered",
    "phase_i_ran", "phase_i_success", "phase_i_cpu_time", "phase_i_ipopt_iter_count", "phase_i_n_attempts",
    "phase_i_initial_comp_res", "phase_i_final_comp_res", "phase_i_residual_improvement_pct",
    "phase_i_best_obj_regime", "phase_i_attempt_0_comp_res", "phase_i_attempt_1_comp_res",
    "phase_i_attempt_2_comp_res", "phase_i_n_restarts_attempted", "phase_i_n_restarts_rejected",
    "phase_i_best_restart_idx", "phase_i_multistart_improved", "phase_i_displacement_from_z0",
    "phase_i_unbounded_dims_count", "phase_i_interior_push_frac", "phase_i_feasibility_achieved",
    "phase_i_near_feasibility", "phase_i_skipped_large",
    "bootstrap_time", "bootstrap_iters", "final_t_k", "n_biactive_final", "n_sign_test_fails",
    "total_nlp_iters", "tracking_count_final", "stagnation_count_final", "last_feasible_t",
    "infeasibility_hits", "max_consecutive_fails_reached",
    "regime_superlinear_count", "regime_fast_count", "regime_slow_count", "regime_adaptive_jump_count",
    "regime_post_stagnation_count", "regime_linf_count", "regime_linf_fallback_count",
    "restoration_random_perturb_count", "restoration_directional_escape_count",
    "restoration_quadratic_reg_count", "restoration_qr_failed_count",
    "solver_ipopt_iters"
]

# Snapshot prefixes
for pfx in ["iter1_", "last_iter_", "best_"]:
    OFFICIAL_COLUMNS += [
        pfx + "t_k", pfx + "delta_k", pfx + "comp_res", pfx + "kkt_res", pfx + "objective",
        pfx + "sign_test", pfx + "solver_status", pfx + "n_biactive", pfx + "nlp_iters",
        pfx + "solver_type", pfx + "warmstart", pfx + "t_update_regime", pfx + "cpu_time",
        pfx + "sta_tol", pfx + "improvement_ratio", pfx + "stagnation_count", pfx + "tracking_count",
        pfx + "is_tracking", pfx + "solver_fallback", pfx + "consec_fails", pfx + "best_comp_so_far",
        pfx + "best_iter_achieved", pfx + "ipopt_tol_used", pfx + "restoration_used",
        pfx + "restoration_trigger", pfx + "restoration_success", pfx + "biactive_indices",
        pfx + "lambda_G_min", pfx + "lambda_G_max", pfx + "lambda_H_min", pfx + "lambda_H_max"
    ]

OFFICIAL_COLUMNS += ["best_iter_number"]
OFFICIAL_COLUMNS += ["lambda_G_min_final", "lambda_G_max_final", "lambda_H_min_final", "lambda_H_max_final"]

# Phase III columns
OFFICIAL_COLUMNS += [
    "bnlp_ran", "bnlp_accepted", "bnlp_status", "bnlp_success", "bnlp_f_val", "bnlp_original_f_val",
    "bnlp_improvement", "bnlp_comp_res_polish", "bnlp_cpu_time", "bnlp_I1_size", "bnlp_I2_size",
    "bnlp_biactive_size", "bnlp_alt_partition_used", "bnlp_n_partitions_tried", "bnlp_phase_time",
    "bnlp_ultra_tight_ran", "bnlp_active_set_frac",
    "lpec_refine_ran", "lpec_refine_bstat_found", "lpec_refine_n_outer", "lpec_refine_n_inner_total",
    "lpec_refine_n_bnlps", "lpec_refine_n_lpecs", "lpec_refine_improvement", "lpec_refine_cpu_time",
    "lpec_phase_time",
    "bstat_cert_ran", "bstat_lpec_status", "bstat_classification", "bstat_lpec_obj", "bstat_n_biactive",
    "bstat_n_active_G", "bstat_n_active_H", "bstat_licq_rank", "bstat_licq_holds", "bstat_licq_details",
    "bstat_n_branches_total", "bstat_n_branches_explored", "bstat_n_feasible_branches", "bstat_timed_out",
    "bstat_elapsed_s", "bstat_used_relaxation", "bstat_trivial_no_biactive"
]

OFFICIAL_COLUMNS += ["time_phase_i", "time_bootstrap", "time_phase_ii", "time_bnlp", "time_lpec", "time_total", "error_msg"]


def _classify_problem_size(n_x: int) -> str:
    """
    Derive problem_size_mode from the number of decision variables.

    Thresholds match the size distribution documented in benchmarks/mpeclib/README.md:
      small  : n_x < 50
      medium : 50 ≤ n_x < 500
      large  : n_x ≥ 500
    """
    if n_x < 50:
        return "small"
    if n_x < 500:
        return "medium"
    return "large"


def map_iteration_to_snapshot(log: IterationLog, prefix: str) -> Dict[str, Any]:
    return {
        prefix + "t_k": log.t_k,
        prefix + "delta_k": log.delta_k,
        prefix + "comp_res": log.comp_res,
        prefix + "kkt_res": log.kkt_res,
        prefix + "objective": log.objective,
        prefix + "sign_test": log.sign_test,
        prefix + "solver_status": log.solver_status,
        prefix + "n_biactive": log.n_biactive,
        prefix + "nlp_iters": log.nlp_iter_count,
        prefix + "solver_type": log.solver_type,
        prefix + "warmstart": log.warmstart_type,
        prefix + "t_update_regime": log.t_update_regime,
        prefix + "cpu_time": log.cpu_time,
        prefix + "sta_tol": log.sta_tol,
        prefix + "improvement_ratio": log.improvement_ratio,
        prefix + "stagnation_count": log.stagnation_count,
        prefix + "tracking_count": log.tracking_count,
        prefix + "is_tracking": log.is_in_tracking_regime,
        prefix + "solver_fallback": log.solver_fallback_occurred,
        prefix + "consec_fails": log.consecutive_solver_failures,
        prefix + "best_comp_so_far": log.best_comp_res_so_far,
        prefix + "best_iter_achieved": log.best_iter_achieved,
        prefix + "ipopt_tol_used": log.ipopt_tol_used,
        prefix + "restoration_used": log.restoration_used,
        prefix + "restoration_trigger": log.restoration_trigger_reason,
        prefix + "restoration_success": log.restoration_success,
        prefix + "biactive_indices": log.biactive_indices_str,
        prefix + "lambda_G_min": log.lambda_G_min,
        prefix + "lambda_G_max": log.lambda_G_max,
        prefix + "lambda_H_min": log.lambda_H_min,
        prefix + "lambda_H_max": log.lambda_H_max,
    }


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Wall clock timeout exceeded")


def run_single_problem_internal(
    loader_fn: Callable[[str], Dict[str, Any]],
    problem_path: str,
    seed: int,
    tag: str,
    results_dir: str,
    save_logs: bool,
    dataset_tag: str,
    wall_timeout: Optional[float] = None,
):
    """Core logic to run a single problem and return the wide data row."""
    import gc
    
    # Clear all caches at start - critical for multiprocessing isolation
    from mpecss.helpers.solver_cache import clear_solver_cache
    from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
    from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
    clear_solver_cache()
    clear_restoration_jac()
    clear_bstat_jac()
    gc.collect()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_total = time.perf_counter()

    try:
        problem = loader_fn(problem_path)
    except Exception as e:
        logger.error(f"Failed to load {problem_path}: {e}")
        return {"problem_file": os.path.basename(problem_path), "error_msg": f"Load error: {e}", "status": "load_failed"}

    params = {
        "seed": seed,
        "max_outer": 3000,  # 3000 outer iters; t_k floor prevents the iteration treadmill
        "adaptive_t": True,
        # Inner NLP solver tolerance MUST be strictly tighter than outer eps_tol (1e-8)
        # to guarantee the solver pushes the point across the boundary.
        "solver_opts": {"max_iter": 5000, "tol": 1e-9},
        "feasibility_phase": True,
    }

    # NOTE: Signal-based timeout (SIGALRM) removed. It was unreliable:
    # 1. SIGALRM doesn't exist on Windows
    # 2. signal.alarm() only interrupts Python code, NOT C++ code (CasADi/IPOPT)
    # 3. If IPOPT gets stuck in matrix factorization, the signal never fires
    # The process-based timeout in _run_parallel_isolated() handles this correctly
    # by monitoring wall-clock time and calling .terminate()/.kill() on the worker.

    # Phase I & II
    res = None
    time_phase_ii = 0.0
    time_bnlp = 0.0
    time_lpec = 0.0
    try:
        z0 = problem["x0_fn"](seed)
        start_phase_ii = time.perf_counter()
        res = run_mpecss(problem, z0, params)
        # Compute time_phase_ii correctly (wall time minus Phase I cpu time)
        _phase_i_cpu = res.get("phase_i_result", {}).get("cpu_time", 0.0) or 0.0
        time_phase_ii = max(0.0, (time.perf_counter() - start_phase_ii) - _phase_i_cpu)

        # Phase III
        time_bnlp_start = time.perf_counter()
        res = bnlp_polish(res, problem)
        time_bnlp = time.perf_counter() - time_bnlp_start

        time_lpec_start = time.perf_counter()
        res = lpec_refinement_loop(res, problem)
        res = bstat_post_check(res, problem)
        time_lpec = time.perf_counter() - time_lpec_start

    except MemoryError as e:
        logger.error(f"OOM for {os.path.basename(problem_path)}: {e}")
        # Aggressive cleanup on OOM
        clear_solver_cache()
        clear_restoration_jac()
        clear_bstat_jac()
        gc.collect()
        return {"problem_file": os.path.basename(problem_path), "error_msg": f"MemoryError: {e}", "status": "oom"}
    except Exception as e:
        # Classify CasADi std::bad_alloc and mmap failures as oom, not crashed
        err_str = str(e)
        _OOM_SIGNALS = (
            "bad_alloc",
            "std::bad_alloc",
            "failed to map segment",
            "cannot allocate memory",
            "out of memory",
        )
        if any(sig in err_str.lower() for sig in _OOM_SIGNALS):
            logger.error(f"OOM (CasADi/system) for {os.path.basename(problem_path)}: {err_str[:200]}")
            clear_solver_cache()
            clear_restoration_jac()
            clear_bstat_jac()
            gc.collect()
            return {"problem_file": os.path.basename(problem_path), "error_msg": f"OOM: {err_str[:300]}", "status": "oom"}
        logger.error(f"Solver error for {os.path.basename(problem_path)}: {err_str[:300]}")
        clear_solver_cache()
        clear_restoration_jac()
        clear_bstat_jac()
        gc.collect()
        return {"problem_file": os.path.basename(problem_path), "error_msg": f"Solver error: {err_str[:300]}", "status": "crashed"}

    total_time = time.perf_counter() - start_total

    # Construct the wide row (initialized with None for pandas NaN handling)
    row = {col: None for col in OFFICIAL_COLUMNS}

    # ── Global Info ────────────────────────────────────────────────────────────
    row["benchmark_suite"] = dataset_tag
    row["problem_file"]    = os.path.basename(problem_path)
    row["run_timestamp"]   = timestamp
    row["seed"]            = seed
    row["wall_timeout_cfg"] = wall_timeout        # FIX #8a: was always None
    row["problem_name"]    = problem.get("name", "unknown")
    n_x = problem.get("n_x", 0)
    row["n_x"]             = n_x
    row["n_comp"]          = problem.get("n_comp", 0)
    row["n_con"]           = problem.get("n_con", 0)
    row["n_p"]             = problem.get("n_p", 0)
    row["family"]          = problem.get("family", "")
    row["problem_size_mode"] = _classify_problem_size(n_x)  # FIX #8b: was always None

    # ── Config ─────────────────────────────────────────────────────────────────
    for k, v in DEFAULT_PARAMS.items():
        row[f"cfg_{k}"] = v
    for k, v in params.items():
        if k != "solver_opts":
            row[f"cfg_{k}"] = v

    # ── Core Results ───────────────────────────────────────────────────────────
    row["status"]                = res.get("status")
    row["stationarity"]          = res.get("stationarity")
    row["f_final"]               = res.get("f_final")
    row["comp_res"]              = res.get("comp_res")
    row["kkt_res"]               = res.get("kkt_res")
    row["sign_test_pass"]        = res.get("sign_test_pass")
    row["b_stationarity"]        = res.get("b_stationarity")
    row["lpec_obj"]              = res.get("lpec_obj")
    row["licq_holds"]            = res.get("licq_holds")
    row["n_outer_iters"]         = res.get("n_outer_iters")
    row["n_restorations"]        = res.get("n_restorations")
    row["cpu_time_total"]        = total_time
    row["smoothing_actually_used"]  = res.get("smoothing_actually_used")
    row["fb_auto_retry_triggered"]  = res.get("fb_auto_retry_triggered")

    # ── Phase I ────────────────────────────────────────────────────────────────
    p1 = res.get("phase_i_result", {})
    if p1:
        row["phase_i_ran"] = True
        for k in [
            "success", "cpu_time", "ipopt_iter_count", "n_attempts",
            "initial_comp_res", "final_comp_res", "residual_improvement_pct",
            "best_obj_regime", "attempt_0_comp_res", "attempt_1_comp_res",
            "attempt_2_comp_res", "n_restarts_attempted", "n_restarts_rejected",
            "best_restart_idx", "multistart_improved", "displacement_from_z0",
            "unbounded_dims_count", "interior_push_frac",
            "feasibility_achieved", "near_feasibility",
        ]:
            row[f"phase_i_{k}"] = p1.get(k)
        row["time_phase_i"]          = p1.get("cpu_time", 0)
        row["phase_i_skipped_large"] = (p1.get("solver_status") == "skipped_large")

    # ── Per-iteration logs ─────────────────────────────────────────────────────
    logs = res.get("logs", [])
    if logs:
        regimes = [l.t_update_regime for l in logs]
        row["regime_superlinear_count"]        = regimes.count("superlinear")
        row["regime_fast_count"]               = regimes.count("fast")
        row["regime_slow_count"]               = regimes.count("slow")
        row["regime_adaptive_jump_count"]      = regimes.count("adaptive_jump")
        row["regime_post_stagnation_count"]    = regimes.count("post_stagnation")
        row["regime_linf_count"]               = regimes.count("linf")
        row["regime_linf_fallback_count"]      = regimes.count("linf_fallback")
        row["total_nlp_iters"]                 = sum(l.nlp_iter_count for l in logs)
        row["final_t_k"]                       = logs[-1].t_k
        row["n_biactive_final"]                = logs[-1].n_biactive
        row["n_sign_test_fails"]               = sum(1 for l in logs if l.sign_test == "FAIL")
        row["tracking_count_final"]            = logs[-1].tracking_count
        row["stagnation_count_final"]          = logs[-1].stagnation_count

        # Snapshots
        row.update(map_iteration_to_snapshot(logs[0],  "iter1_"))
        row.update(map_iteration_to_snapshot(logs[-1], "last_iter_"))
        best_log = min(logs, key=lambda l: l.comp_res)
        row.update(map_iteration_to_snapshot(best_log, "best_"))
        row["best_iter_number"]      = best_log.iteration

        # Final multiplier bounds
        row["lambda_G_min_final"]    = logs[-1].lambda_G_min
        row["lambda_G_max_final"]    = logs[-1].lambda_G_max
        row["lambda_H_min_final"]    = logs[-1].lambda_H_min
        row["lambda_H_max_final"]    = logs[-1].lambda_H_max

    # Solver-level scalar fields surfaced by run_mpecss
    for k in [
        "bootstrap_time", "bootstrap_iters", "last_feasible_t",
        "infeasibility_hits", "max_consecutive_fails_reached",
        "restoration_random_perturb_count", "restoration_directional_escape_count",
        "restoration_quadratic_reg_count", "restoration_qr_failed_count",
        "solver_ipopt_iters",
    ]:
        row[k] = res.get(k)

    # ── BNLP Polish ────────────────────────────────────────────────────────────
    bnlp = res.get("bnlp_polish", {})
    if bnlp:
        row["bnlp_ran"] = True
        for k in [
            "accepted",           # → bnlp_accepted
            "status",             # → bnlp_status
            "success",            # → bnlp_success
            "f_val",              # → bnlp_f_val
            "original_f_val",     # → bnlp_original_f_val
            "improvement",        # → bnlp_improvement
            "comp_res_polish",    # → bnlp_comp_res_polish
            "cpu_time",           # → bnlp_cpu_time
            "alt_partition_used", # → bnlp_alt_partition_used
            "n_partitions_tried", # → bnlp_n_partitions_tried
            "ultra_tight_ran",    # → bnlp_ultra_tight_ran
            "active_set_frac",    # → bnlp_active_set_frac
        ]:
            row[f"bnlp_{k}"] = bnlp.get(k)
        row["bnlp_I1_size"]      = len(bnlp.get("I1", []))
        row["bnlp_I2_size"]      = len(bnlp.get("I2", []))
        row["bnlp_biactive_size"] = len(bnlp.get("I_biactive", []))
        row["time_bnlp"]         = time_bnlp
        row["bnlp_phase_time"]   = time_bnlp   # FIX #8c: was always None

    # ── LPEC Refinement ────────────────────────────────────────────────────────
    lpec = res.get("lpec_refine", {})
    if lpec:
        row["lpec_refine_ran"] = True
        for k in ["bstat_found", "n_outer", "n_inner_total", "n_bnlps", "n_lpecs", "improvement", "cpu_time"]:
            row[f"lpec_refine_{k}"] = lpec.get(k)
        row["time_lpec"]         = time_lpec
        row["lpec_phase_time"]   = time_lpec   # FIX #8d: was always None

    # ── B-stationarity Certificate ─────────────────────────────────────────────
    bstat = res.get("bstat_details", {})
    if bstat:
        row["bstat_cert_ran"] = True
        for k in [
            "lpec_status", "classification", "lpec_obj", "n_biactive",
            "n_active_G", "n_active_H", "licq_rank", "licq_holds", "licq_details",
            "n_branches_total", "n_branches_explored", "n_feasible_branches",
            "timed_out", "elapsed_s", "used_relaxation", "trivial_no_biactive",
        ]:
            row[f"bstat_{k}"] = bstat.get(k)

    # ── Time Breakdown ─────────────────────────────────────────────────────────
    row["time_phase_ii"]  = time_phase_ii
    row["time_bootstrap"] = res.get("bootstrap_time")   # FIX #8e: was always None
    row["time_total"]     = total_time

    # ── Optional per-iteration log export ──────────────────────────────────────
    if save_logs:
        log_dir = os.path.join(results_dir, "iteration_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{row['problem_name']}_{tag}_{timestamp}.csv")
        export_csv(logs, log_path)

    # ── CRITICAL: Post-problem memory cleanup ──────────────────────────────────
    # Clear all caches and force garbage collection to prevent memory accumulation
    # across problems. Without this, CasADi solver objects, symbolic graphs, and
    # numpy arrays accumulate, eventually causing OOM on memory-constrained systems.
    from mpecss.helpers.solver_cache import clear_solver_cache
    from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
    from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
    clear_solver_cache()
    clear_restoration_jac()
    clear_bstat_jac()
    # Delete large result data that's no longer needed
    if 'logs' in res:
        res['logs'] = None  # Free the logs list (can be large)
    gc.collect()

    return row


def run_benchmark_main(loader_fn: Callable[[str], Dict[str, Any]], dataset_tag: str, default_path: str):
    """Entry point for the three main benchmark runner scripts."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description=f"Parallel {dataset_tag} Benchmark Runner")
    parser.add_argument("--tag",          type=str,   default="Official")
    parser.add_argument("--problem",      type=str,   help="Problem name or substring filter")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--workers",      type=int,   default=1,
                        help="Number of parallel workers. Each worker runs one problem at a time (default: 1).")
    parser.add_argument("--timeout",      type=float, default=3600.0,
                        help="Per-problem wall-clock timeout in seconds (default: 3600). "
                             "Set 0 to disable.")
    parser.add_argument("--mem-limit-gb", type=float, default=None,
                        help="Soft per-worker RAM cap in GB (Linux/WSL only).")
    parser.add_argument("--save-logs",    action="store_true", help="Save detailed per-iteration CSV logs")
    parser.add_argument("--sort-by-size", action="store_true", help="Sort problems by file size (small -> large)")
    parser.add_argument("--shuffle",      action="store_true", help="Shuffle problems randomly to distribute RAM load evenly")
    parser.add_argument("--path",         type=str,   default=default_path,

                        help="Path to benchmark JSON directory")
    parser.add_argument("--resume",       type=str,   help="Path to existing CSV results to resume from")
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, ignore past OOM/timeout/crash results and re-run them")
    args = parser.parse_args()

    # Normalise timeout: treat 0 as None (no limit)
    if args.timeout is not None and args.timeout <= 0:
        args.timeout = None

    # Prevent internal thread oversubscription inside worker processes
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    results_dir = os.path.abspath("results")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.isdir(args.path):
        logger.error(f"Benchmark path not found: {args.path}")
        return

    problem_files = [f for f in os.listdir(args.path) if f.endswith(".json")]
    if args.sort_by_size:
        # Sort problems by file size (ascending) so small problems run first.
        # This gives the user early feedback and makes it easy to see which
        # size class is running.
        problem_files.sort(key=lambda f: os.path.getsize(os.path.join(args.path, f)))
        logger.info("Problem execution order: Sorted by size (small -> large).")
    elif getattr(args, 'shuffle', False):
        import random
        random.seed(args.seed)
        random.shuffle(problem_files)
        logger.info(f"Problem execution order: Shuffled randomly (seed={args.seed}) to distribute RAM load.")
    else:
        # Default to alphabetical sorting for consistency
        problem_files.sort()
        logger.info("Problem execution order: Alphabetical.")


    all_results: List[Dict[str, Any]] = []
    if args.resume:
        if not os.path.isfile(args.resume):
            logger.error(f"Resume file not found: {args.resume}")
            return
        
        try:
            df_old = pd.read_csv(args.resume)
            if getattr(args, 'retry_failed', False):
                # Filter out previous failures so they get re-run
                failed_mask = df_old['status'].isin(['oom', 'timeout', 'crashed', 'Exception', 'load_failed'])
                df_success = df_old[~failed_mask]
                all_results = df_success.to_dict('records')
                done_files = set(df_success['problem_file'].tolist())
            else:
                all_results = df_old.to_dict('records')
                done_files = set(df_old['problem_file'].tolist())
                
            count_before = len(problem_files)
            problem_files = [f for f in problem_files if f not in done_files]
            logger.info(f"Resuming from {args.resume}: skipped {count_before - len(problem_files)} already completed problems.")
        except Exception as e:
            logger.error(f"Failed to read resume file {args.resume}: {e}")
            return

    if args.problem:
        problem_files = [f for f in problem_files if args.problem in f]

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(results_dir, f"{dataset_tag}_full_{args.tag}_{timestamp}.csv")
    if args.resume:
        # Use a name that indicates it's a continuation
        summary_path = os.path.join(results_dir, f"{dataset_tag}_full_{args.tag}_{timestamp}_resumed.csv")

    if psutil:
        vm = psutil.virtual_memory()
        avail_gb = vm.available / 1024**3
        total_gb = vm.total / 1024**3
        logger.info(
            f"System memory: {avail_gb:.1f} GB available / {total_gb:.1f} GB total"
        )

    logger.info(
        f"Starting {dataset_tag} benchmark: {len(problem_files)} problem(s), "
        f"{args.workers} worker(s), timeout={args.timeout}s."
    )
    logger.info(f"Results will be written to: {summary_path}")

    # Always use the isolated process runner, even with workers=1.
    # The serial path relied on signal.alarm for timeouts, which is
    # Unix-only and silently does nothing on Windows, causing infinite
    # stalls.  The isolated runner enforces wall-clock deadlines via
    # Process.is_alive() checks, which work on all platforms.
    all_results = _run_parallel_isolated(
        problem_files, loader_fn, args, results_dir, dataset_tag, summary_path
    )

    _write_run_env(results_dir, timestamp, dataset_tag, args)
    logger.info(f"Benchmark complete. Results: {summary_path}")


def _write_run_env(results_dir: str, timestamp: str, dataset_tag: str, args) -> None:
    """
    Write a machine-readable JSON snapshot of every setting that could affect
    reproducibility: package versions, Python version, OS info, CLI args,
    thread env vars, and hardware info.  One file per benchmark run.
    """
    env = {
        "run_timestamp":  timestamp,
        "dataset_tag":    dataset_tag,
        "cli_args": {
            "tag":          args.tag,
            "seed":         args.seed,
            "workers":      args.workers,
            "timeout_s":    args.timeout,
            "mem_limit_gb": getattr(args, "mem_limit_gb", None),
            "path":         args.path,
            "save_logs":    args.save_logs,
        },
        "env_vars": {
            k: os.environ.get(k, "not set")
            for k in [
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            ]
        },
        "python": {
            "version":      platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable":   sys.executable,
        },
        "platform": {
            "system":   platform.system(),
            "release":  platform.release(),
            "machine":  platform.machine(),
            "node":     platform.node(),
        },
        "packages": {},
        "hardware": {},
    }

    # Collect installed package versions
    for pkg in ["casadi", "numpy", "pandas", "scipy", "psutil", "matplotlib"]:
        try:
            import importlib.metadata
            env["packages"][pkg] = importlib.metadata.version(pkg)
        except Exception:
            env["packages"][pkg] = "unknown"

    # Hardware info (best-effort)
    try:
        import psutil
        vm = psutil.virtual_memory()
        env["hardware"]["ram_total_gb"]     = round(vm.total / 1024**3, 2)
        env["hardware"]["ram_available_gb"] = round(vm.available / 1024**3, 2)
        env["hardware"]["cpu_logical"]      = psutil.cpu_count(logical=True)
        env["hardware"]["cpu_physical"]     = psutil.cpu_count(logical=False)
    except Exception:
        pass

    # CPU model (Linux/WSL)
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    env["hardware"]["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    # Git commit (for exact code reproducibility)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        env["git_commit"] = commit
    except Exception:
        env["git_commit"] = "unknown"

    env_path = os.path.join(
        results_dir, f"{dataset_tag}_run_env_{args.tag}_{timestamp}.json"
    )
    try:
        with open(env_path, "w") as f:
            json.dump(env, f, indent=2)
        logger.info(f"Run environment snapshot: {env_path}")
    except Exception as e:
        logger.warning(f"Could not write run environment snapshot: {e}")


def _worker_process(problem_file, loader_fn, args_path, seed, tag, results_dir,
                    save_logs, dataset_tag, timeout, mem_limit_gb, result_queue):
    """
    Worker function that runs in an isolated spawned process.

    Catches BaseException (including MemoryError and CasADi std::bad_alloc)
    so that even Python-level OOMs produce a structured result row rather
    than silently dying.  No memory cap is imposed: the OS manages memory
    freely across all workers, and the monitor loop in the parent detects
    any hard kernel OOM-kill via exit code and records it as status=oom.
    """
    # Set thread environment variables BEFORE any imports that might use them.
    # This ensures CasADi/NumPy/OpenBLAS don't spawn extra threads.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # ── Run solver ────────────────────────────────────────────────────────
    res = None
    try:
        res = run_single_problem_internal(
            loader_fn, os.path.join(args_path, problem_file),
            seed, tag, results_dir, save_logs, dataset_tag, timeout
        )
    except MemoryError:
        res = {
            "problem_file": problem_file,
            "status": "oom",
            "error_msg": "MemoryError: worker exceeded memory limit",
        }
    except BaseException as e:   # includes KeyboardInterrupt, SystemExit, etc.
        res = {
            "problem_file": problem_file,
            "status": "crashed",
            "error_msg": f"Worker error: {type(e).__name__}: {e}",
        }
    finally:
        # Always run cleanup, even on exception, to minimize memory footprint
        # before sending result (queue.put may need some memory headroom)
        try:
            from mpecss.helpers.solver_cache import clear_solver_cache
            from mpecss.phase_2.restoration import clear_jacobian_cache as clear_restoration_jac
            from mpecss.phase_3.bstationarity import clear_jacobian_cache as clear_bstat_jac
            clear_solver_cache()
            clear_restoration_jac()
            clear_bstat_jac()
            gc.collect()
        except Exception:
            pass  # Don't let cleanup failures mask the actual error

    # ── Send result (best-effort; may fail if we are already OOM) ─────────
    try:
        result_queue.put((problem_file, res))
    except Exception as qe:
        # Last-ditch: shrink the payload to just the key fields and retry.
        try:
            slim = {
                "problem_file": res.get("problem_file", problem_file),
                "status":       res.get("status", "crashed"),
                "error_msg":    str(res.get("error_msg", ""))[:200],
            }
            result_queue.put((problem_file, slim))
        except Exception:
            pass  # If this also fails the monitor loop will detect exit code != 0


def _run_parallel_isolated(problem_files, loader_fn, args, results_dir, dataset_tag, summary_path):
    """
    Run problems in parallel using isolated Process objects and a sliding window.

    Each problem gets its own fresh process so that an OOM-kill or crash
    of one worker cannot corrupt or block the collection of results from
    the other workers. The collection loop polls the queue frequently and
    monitors each child's wall-clock time individually, enforcing timeouts.

    Results are checkpointed to the CSV after every single problem
    completes, so progress is never lost even if a later problem hangs.
    """
    mp_context = multiprocessing.get_context('spawn')
    manager = mp_context.Manager()
    all_results = []
    completed = 0
    total = len(problem_files)
    benchmark_start = time.time()

    remaining = list(problem_files)
    active_procs = {}  # problem_file -> (Process, start_time)
    result_queue = manager.Queue()

    timeout_grace = 120
    timeout_per_problem = (args.timeout + timeout_grace) if args.timeout else None

    while remaining or active_procs:
        # 1. Fill open slots
        while len(active_procs) < args.workers and remaining:
            f = remaining.pop(0)
            
            p = mp_context.Process(
                target=_worker_process,
                args=(f, loader_fn, args.path, args.seed, args.tag,
                      results_dir, args.save_logs, dataset_tag, args.timeout,
                      getattr(args, "mem_limit_gb", None), result_queue),
            )
            p.start()
            active_procs[f] = (p, time.time())

        # 2. Consume all immediately available results from the queue
        while True:
            try:
                problem_file, res = result_queue.get(timeout=0.2)
                if problem_file in active_procs:
                    dp, _ = active_procs.pop(problem_file)
                    dp.join(timeout=1.0)
                completed += 1
                elapsed = time.time() - benchmark_start
                prob_time = res.get('cpu_time_total', res.get('time_total', '?'))
                if isinstance(prob_time, (int, float)):
                    prob_time = f"{prob_time:.1f}s"
                size_tag = res.get('problem_size_mode', '?')
                logger.info(
                    f"[{completed}/{total}] "
                    f"{res.get('problem_file', problem_file)} — "
                    f"{res.get('status')} | "
                    f"size={size_tag} | prob_time={prob_time} | "
                    f"elapsed={elapsed:.0f}s"
                )
                all_results.append(res)
                _save_csv(all_results, summary_path)
            except _queue_module.Empty:
                break
            except Exception as exc:
                logger.debug(f"Queue read error: {exc}")
                break

        # 3. Check for timeouts and dead processes
        for f in list(active_procs.keys()):
            p, start_time = active_procs[f]

            # Check for timeout
            if timeout_per_problem and time.time() - start_time > timeout_per_problem:
                logger.error(f"[{completed + 1}/{total}] {f} — wall-clock deadline exceeded, terminating")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                p.join()
                completed += 1
                timeout_res = {
                    "problem_file": f,
                    "status": "timeout",
                    "error_msg": "Wall-clock deadline exceeded (force killed)",
                }
                all_results.append(timeout_res)
                _save_csv(all_results, summary_path)
                del active_procs[f]
                continue

            # Check if process died
            if not p.is_alive():
                # Process is dead, but a result might still be in the pipe buffer.
                # Give it a tiny window to flush.
                try:
                    problem_file, res = result_queue.get(timeout=0.2)
                    if problem_file in active_procs:
                        dp, _ = active_procs.pop(problem_file)
                        dp.join(timeout=1.0)
                    completed += 1
                    elapsed = time.time() - benchmark_start
                    prob_time = res.get('cpu_time_total', res.get('time_total', '?'))
                    if isinstance(prob_time, (int, float)):
                        prob_time = f"{prob_time:.1f}s"
                    logger.info(
                        f"[{completed}/{total}] "
                        f"{res.get('problem_file', problem_file)} — "
                        f"{res.get('status')} | "
                        f"prob_time={prob_time} | "
                        f"elapsed={elapsed:.0f}s"
                    )
                    all_results.append(res)
                    _save_csv(all_results, summary_path)
                except _queue_module.Empty:
                    pass
                except Exception:
                    pass

                # If f still in active_procs, it truly yielded no result.
                if f in active_procs:
                    exit_code = p.exitcode
                    completed += 1
                    if exit_code == 0:
                        logger.error(f"[{completed}/{total}] {f} — process exited cleanly but sent no result")
                        crash_status = "crashed"
                        crash_msg = "Worker exited without sending result"
                    elif exit_code in (-9, 137, 9):
                        logger.error(f"[{completed}/{total}] {f} — OOM-killed by the kernel (exit={exit_code}).")
                        crash_status = "oom"
                        crash_msg = f"OOM kill (exit {exit_code})"
                    elif exit_code in (-11, 139, 11):
                        logger.error(f"[{completed}/{total}] {f} — segmentation fault (exit={exit_code})")
                        crash_status = "crashed"
                        crash_msg = f"Segfault (exit {exit_code})"
                    else:
                        logger.error(f"[{completed}/{total}] {f} — process killed (exit={exit_code})")
                        crash_status = "crashed"
                        crash_msg = f"Process terminated with exit code {exit_code}"

                    crash_res = {
                        "problem_file": f,
                        "status": crash_status,
                        "error_msg": crash_msg,
                    }
                    all_results.append(crash_res)
                    _save_csv(all_results, summary_path)
                    del active_procs[f]
                    p.join()

    # Empty any final messages just in case
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Exception:
            break

    return all_results


def _save_csv(results: List[Dict[str, Any]], path: str) -> None:
    """Write the current results list to a CSV, keeping only OFFICIAL_COLUMNS in order."""
    df   = pd.DataFrame(results)
    cols = [c for c in OFFICIAL_COLUMNS if c in df.columns]
    df[cols].to_csv(path, index=False)
