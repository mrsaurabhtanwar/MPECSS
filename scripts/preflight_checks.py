#!/usr/bin/env python3
"""
Pre-flight checks for MPECSS benchmark runs.

Verifies:
  1. Python environment and virtualenv status
  2. All required dependencies (CasADi, IPOPT, MUMPS, etc.)
  3. WSL environment detection
  4. Disk space availability
  5. Existing results and potential conflicts
  6. System memory and CPU cores
  7. Problem data integrity (JSON loading)
"""

import os
import sys
import json
import platform
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')
logger = logging.getLogger('preflight')


def check_python_env():
    """Check Python version and environment."""
    logger.info("=" * 70)
    logger.info("1. PYTHON ENVIRONMENT")
    logger.info("=" * 70)
    
    version = platform.python_version()
    impl = platform.python_implementation()
    executable = sys.executable
    prefix = sys.prefix
    
    logger.info(f"  Python version  : {version} ({impl})")
    logger.info(f"  Executable      : {executable}")
    logger.info(f"  Installation dir: {prefix}")
    
    # Check if in virtualenv
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    logger.info(f"  In virtualenv?  : {'✓ YES' if in_venv else '⚠ NO (recommend using venv)'}")
    
    if not in_venv:
        logger.warning("Running outside virtualenv may cause package conflicts.")
    
    return True


def check_dependencies():
    """Check all required packages."""
    logger.info("\n" + "=" * 70)
    logger.info("2. DEPENDENCIES")
    logger.info("=" * 70)
    
    required = {
        'casadi': 'CasADi (nonlinear solver framework)',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas (results export)',
        'psutil': 'psutil (system monitoring)',
    }
    
    all_ok = True
    for pkg, desc in required.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            logger.info(f"  ✓ {pkg:12} v{version:20} ({desc})")
        except ImportError:
            logger.error(f"  ✗ {pkg:12} NOT FOUND ({desc})")
            all_ok = False
    
    # Special check: CasADi + IPOPT (test by actually creating a solver)
    try:
        import casadi as ca
        x = ca.SX.sym('x', 2)
        nlp = {'x': x, 'f': ca.sumsqr(x), 'g': ca.vertcat(x[0] + x[1])}
        solver = ca.nlpsol('ipopt_test', 'ipopt', nlp, {'ipopt': {'print_level': 0}})
        logger.info(f"  ✓ IPOPT backend available in CasADi")
    except Exception as e:
        logger.error(f"  ✗ IPOPT backend NOT available in CasADi: {e}")
        all_ok = False
    
    # Check MUMPS linear solver
    try:
        import casadi as ca
        x = ca.SX.sym('x', 2)
        nlp = {'x': x, 'f': ca.sumsqr(x), 'g': ca.vertcat(x[0] + x[1])}
        solver = ca.nlpsol('mumps_test', 'ipopt', nlp, {'ipopt': {'linear_solver': 'mumps', 'print_level': 0}})
        logger.info(f"  ✓ MUMPS linear solver available")
    except Exception as e:
        logger.warning(f"  ⚠ MUMPS may not be available (fallback to default): {e}")
    
    # Check qpOASES for small problem acceleration (optional, open-source LGPL)
    try:
        import casadi as ca
        # QP structure: h=Hessian sparsity, a=constraint Jacobian sparsity
        # Note: linear cost 'g' is passed at solve time, not in structure
        h = ca.DM.eye(2)
        a = ca.DM.ones(1, 2)  # One constraint
        qp = {'h': h.sparsity(), 'a': a.sparsity()}
        solver = ca.conic('qpoases_test', 'qpoases', qp, {'printLevel': 'none'})
        # Test solve
        sol = solver(h=h, g=ca.DM.zeros(2), a=a, lba=-1, uba=1, lbx=-10, ubx=10)
        logger.info(f"  ✓ qpOASES available (SQP acceleration for small problems ≤400 vars)")
    except Exception as e:
        logger.info(f"  ℹ qpOASES not available - using IPOPT for all problems (optional)")
        logger.info(f"    Reason: {str(e).split(':')[-1].strip() if ':' in str(e) else str(e)[:80]}")
        logger.info(f"    See QPOASES_INSTALLATION.md for installation instructions")
    
    return all_ok


def check_wsl():
    """Detect WSL environment."""
    logger.info("\n" + "=" * 70)
    logger.info("3. WSL DETECTION")
    logger.info("=" * 70)
    
    system = platform.system()
    logger.info(f"  Operating System: {system}")
    
    is_wsl = False
    if system == 'Linux':
        try:
            with open('/proc/version', 'r') as f:
                content = f.read().lower()
                is_wsl = 'wsl' in content or 'microsoft' in content
        except:
            pass
    
    logger.info(f"  WSL environment?: {'✓ YES' if is_wsl else 'Windows/Native Linux/Other'}")
    
    if system == 'Windows':
        logger.info("  ℹ Running on Windows. For official runs, WSL2 is recommended.")
    
    return is_wsl


def check_disk_space():
    """Check available disk space."""
    logger.info("\n" + "=" * 70)
    logger.info("4. DISK SPACE")
    logger.info("=" * 70)
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    try:
        stat = shutil.disk_usage(str(project_root))
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        pct_free = (stat.free / stat.total) * 100
        
        logger.info(f"  Project root    : {project_root}")
        logger.info(f"  Free space      : {free_gb:.1f} GB ({pct_free:.1f}% of {total_gb:.1f} GB)")
        
        # Recommend 10 GB for safe results storage
        if free_gb < 10:
            logger.warning(f"  ⚠ Less than 10 GB free. Results may fail to save.")
            return False
        else:
            logger.info(f"  ✓ Sufficient disk space")
            return True
    except Exception as e:
        logger.warning(f"  Could not check disk space: {e}")
        return True


def check_results_dir():
    """Check existing results and potential conflicts."""
    logger.info("\n" + "=" * 70)
    logger.info("5. EXISTING RESULTS")
    logger.info("=" * 70)
    
    results_dir = Path('results')
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
        logger.info(f"  Created results directory: {results_dir}")
    
    # Count existing results
    mpeclib_csvs = list(results_dir.glob('mpeclib_full_*.csv'))
    macmpec_csvs = list(results_dir.glob('macmpec_full_*.csv'))
    nosbench_csvs = list(results_dir.glob('nosbench_full_*.csv'))
    
    logger.info(f"  MPECLib CSVs    : {len(mpeclib_csvs)} file(s)")
    logger.info(f"  MacMPEC CSVs    : {len(macmpec_csvs)} file(s)")
    logger.info(f"  NOSBench CSVs   : {len(nosbench_csvs)} file(s)")
    
    if mpeclib_csvs:
        latest_mpec = sorted(mpeclib_csvs)[-1]
        logger.info(f"    Latest MPECLib: {latest_mpec.name}")
    
    if macmpec_csvs:
        latest_mac = sorted(macmpec_csvs)[-1]
        logger.info(f"    Latest MacMPEC: {latest_mac.name}")
    
    if nosbench_csvs:
        latest_nos = sorted(nosbench_csvs)[-1]
        logger.info(f"    Latest NOSBench: {latest_nos.name}")
    
    return True


def check_system_resources():
    """Check available system memory and CPU cores."""
    logger.info("\n" + "=" * 70)
    logger.info("6. SYSTEM RESOURCES")
    logger.info("=" * 70)
    
    try:
        import psutil
        
        # RAM
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024**3)
        avail_gb = vm.available / (1024**3)
        pct_used = vm.percent
        
        logger.info(f"  RAM             : {avail_gb:.1f} GB available / {total_gb:.1f} GB total ({pct_used:.1f}% used)")
        
        # CPU
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        
        logger.info(f"  Logical cores   : {logical}")
        logger.info(f"  Physical cores  : {physical}")
        
        # Recommendation
        recommended_workers = min(physical, max(1, int(total_gb / 2)))
        logger.info(f"  Recommended workers: {recommended_workers} (for ~2GB per worker)")
        
        if total_gb < 4:
            logger.warning(f"  ⚠ Less than 4 GB RAM. Use --workers 1 for safety.")
        
        return True
    except ImportError:
        logger.warning("  psutil not available — skipping resource checks")
        return True


def check_problem_data():
    """Check that problem JSON files can be loaded."""
    logger.info("\n" + "=" * 70)
    logger.info("7. PROBLEM DATA INTEGRITY")
    logger.info("=" * 70)
    
    issues = []
    
    # Check MPECLib
    mpeclib_dir = Path('benchmarks/mpeclib/macmpec-json')
    if mpeclib_dir.exists():
        json_files = list(mpeclib_dir.glob('*.json'))
        logger.info(f"  MPECLib JSON files: {len(json_files)}")
        
        # Try to load first and last
        if json_files:
            for test_file in [json_files[0], json_files[-1]]:
                try:
                    with open(test_file) as f:
                        data = json.load(f)
                        # Required keys that should exist in JSON (n_x, n_comp, n_con are computed by loader)
                        required_keys = {'name', 'f_fun', 'G_fun', 'H_fun'}
                        if not required_keys.issubset(data.keys()):
                            issues.append(f"  ✗ {test_file.name}: missing keys {required_keys - set(data.keys())}")
                except Exception as e:
                    issues.append(f"  ✗ {test_file.name}: {e}")
    else:
        logger.warning(f"  MPECLib directory not found: {mpeclib_dir}")
    
    # Check MacMPEC
    macmpec_dir = Path('benchmarks/macmpec/macmpec-json')
    if macmpec_dir.exists():
        json_files = list(macmpec_dir.glob('*.json'))
        logger.info(f"  MacMPEC JSON files: {len(json_files)}")
        
        if json_files:
            for test_file in [json_files[0], json_files[-1]]:
                try:
                    with open(test_file) as f:
                        data = json.load(f)
                        # Required keys that should exist in JSON (n_x, n_comp, n_con are computed by loader)
                        required_keys = {'name', 'f_fun', 'G_fun', 'H_fun'}
                        if not required_keys.issubset(data.keys()):
                            issues.append(f"  ✗ {test_file.name}: missing keys {required_keys - set(data.keys())}")
                except Exception as e:
                    issues.append(f"  ✗ {test_file.name}: {e}")
    else:
        logger.warning(f"  MacMPEC directory not found: {macmpec_dir}")
    
    # Check NOSBench
    nosbench_dir = Path('benchmarks/nosbench/nosbench-json')
    if nosbench_dir.exists():
        json_files = list(nosbench_dir.glob('*.json'))
        logger.info(f"  NOSBench JSON files: {len(json_files)}")
        
        if json_files:
            for test_file in [json_files[0], json_files[-1]]:
                try:
                    with open(test_file) as f:
                        data = json.load(f)
                        # NOSBench uses 'w' (serialized CasADi function) format
                        required_keys = {'w', 'w0', 'lbw', 'ubw'}
                        if not required_keys.issubset(data.keys()):
                            issues.append(f"  ✗ {test_file.name}: missing keys {required_keys - set(data.keys())}")
                except Exception as e:
                    issues.append(f"  ✗ {test_file.name}: {e}")
    else:
        logger.warning(f"  NOSBench directory not found: {nosbench_dir}")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        return False
    else:
        logger.info("  ✓ Problem data loads cleanly")
        return True


def main():
    """Run all pre-flight checks."""
    logger.info("\n")
    logger.info("╔" + "=" * 68 + "╗")
    logger.info("║" + " " * 68 + "║")
    logger.info("║" + "  MPEC-SS BENCHMARK PRE-FLIGHT CHECKS".center(68) + "║")
    logger.info("║" + " " * 68 + "║")
    logger.info("╚" + "=" * 68 + "╝")
    
    checks = [
        ("Python environment", check_python_env),
        ("Dependencies", check_dependencies),
        ("WSL environment", check_wsl),
        ("Disk space", check_disk_space),
        ("Results directory", check_results_dir),
        ("System resources", check_system_resources),
        ("Problem data", check_problem_data),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            logger.error(f"Check '{name}' failed with exception: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    all_ok = all(results.values())
    for name, status in results.items():
        icon = "✓" if status else "✗"
        logger.info(f"  {icon} {name}")
    
    logger.info("\n" + "=" * 70)
    if all_ok:
        logger.info("✓ All pre-flight checks passed. Ready to benchmark!")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("✗ Some checks failed. Please fix issues before benchmarking.")
        logger.info("=" * 70)
        return 1


if __name__ == '__main__':
    sys.exit(main())
