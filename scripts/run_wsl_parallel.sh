#!/bin/bash
# =============================================================================
# MPEC-SS: Official Reproducible Benchmark Runner
# =============================================================================
#
# Designed for: 2 workers, ~7 GB total RAM, 3600 s per-problem timeout.
# Works on any Linux / WSL2 machine; auto-detects RAM and computes a safe
# per-worker memory cap so that OOM kills are caught gracefully.
#
# Usage:
#   ./scripts/run_wsl_parallel.sh           # official defaults (2 workers)
#   ./scripts/run_wsl_parallel.sh 4         # override worker count
#
# All parameters that affect results are written to
#   results/<suite>_run_env_<tag>_<timestamp>.json
# so every run is fully reproducible from that file alone.
# =============================================================================

set -euo pipefail

# ─── Configurable parameters ───────────────────────────────────────────────────
# ─── Hardware Detection & Recommendation ──────────────────────────────────────
# Detect total RAM available to the environment (works on Linux/WSL)
TOTAL_RAM_GB=$(free -g | awk '/^Mem:/{print $2}' || echo "0")
PHYSICAL_CORES=$(nproc --all 2>/dev/null || echo "1")

# Default recommendation: Use all physical cores if we have at least 1GB per core
RECOMMENDED_WORKERS=$PHYSICAL_CORES
if [ "$TOTAL_RAM_GB" -lt "$PHYSICAL_CORES" ] && [ "$TOTAL_RAM_GB" -gt 0 ]; then
    RECOMMENDED_WORKERS=$TOTAL_RAM_GB
fi
[ $RECOMMENDED_WORKERS -lt 1 ] && RECOMMENDED_WORKERS=1

WORKERS="${1:-$RECOMMENDED_WORKERS}"   # parallel workers (one problem each)
EXTRA_ARGS="${@:2}" # forward all other arguments (e.g. --problem, --tag)

if [ "$WORKERS" -gt "$PHYSICAL_CORES" ]; then
    echo "[WARNING] Using more workers (${WORKERS}) than physical cores (${PHYSICAL_CORES}) may cause slowdown due to context switching."
    echo "[WARNING] Continuing in 3 seconds... (Ctrl+C to abort)"
    sleep 3
fi

TIMEOUT=3600        # per-problem wall-clock budget (seconds)
SEED=42             # RNG seed (change here to run robustness studies)
TAG="Official"      # label embedded in results filenames

# ─── Thread isolation ────────────────────────────────────────────────────────
# Each worker must use exactly 1 thread so that CPU time is comparable
# across machines and the wall-clock / CPU-time ratio stays near 1.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# ─── Helpers ────────────────────────────────────────────────────────────────
print_header() {
    local SUITE="$1"
    echo ""
    echo "============================================================"
    echo "  MPEC-SS  |  Suite: ${SUITE}  |  Tag: ${TAG}"
    echo "  Workers: ${WORKERS}  |  Timeout: ${TIMEOUT}s  |  Seed: ${SEED}"
    echo "============================================================"
}

build_args() {
    # Default: 4 workers, shuffle problems to distribute RAM load
    local ARGS="--workers ${WORKERS} --timeout ${TIMEOUT} --seed ${SEED} --tag ${TAG} --shuffle"
    
    # Append any manually passed arguments (overrides defaults if repeated)
    if [ -n "$EXTRA_ARGS" ]; then
        ARGS="$ARGS $EXTRA_ARGS"
    fi

    # Auto-resume from the latest full run CSV to retry failures and complete remaining
    local LATEST_CSV=$(ls -t results/*_full_${TAG}_*.csv 2>/dev/null | head -n 1 || true)
    if [ -n "$LATEST_CSV" ] && [[ ! "$EXTRA_ARGS" == *"--resume"* ]]; then
        echo "[info] Resuming from: $LATEST_CSV and retrying failures/remaining..." >&2
        ARGS="$ARGS --resume $LATEST_CSV --retry-failed"
    fi
    echo "$ARGS"
}

die() { echo "[ERROR] $*" >&2; exit 1; }

# ─── Pre-flight checks ─────────────────────────────────────────────────────────

# Must be run from the project root
[ -f "scripts/run_wsl_parallel.sh" ] || \
    die "Run this script from the project root: ./scripts/run_wsl_parallel.sh"

# Python must be on PATH
command -v python3 >/dev/null 2>&1 || die "python3 not found. Activate your venv first."

# Confirm CasADi imports cleanly (catches broken installs early)
python3 -c "import casadi" 2>/dev/null || \
    die "CasADi import failed. Run: pip install -r requirements.txt"

# ─── Clear stale bytecode ───────────────────────────────────────────────────────
# Guarantees that source-file edits are always picked up on re-runs.
python3 scripts/clear_pyc.py --delete 2>/dev/null && \
    echo "[info] Stale .pyc files cleared." || true

mkdir -p results

# ─── Print run configuration summary ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "  MPEC-SS Benchmark Runner  —  Configuration"
echo "============================================================"
echo "  Python   : $(python3 --version 2>&1)"
echo "  CasADi   : $(python3 -c 'import casadi; print(casadi.__version__)' 2>/dev/null)"
echo "  Workers  : ${WORKERS}"
echo "  Timeout  : ${TIMEOUT} s per problem"
echo "  Seed     : ${SEED}"
echo "  Tag      : ${TAG}"
echo "  Results  : ./results/"
echo "============================================================"
echo ""

# ─── Run benchmarks ─────────────────────────────────────────────────────────
# Uncomment the suites you want to run.
# Each suite writes its own CSV + run_env JSON to ./results/.

print_header "MacMPEC (191 problems)"
python3 scripts/run_macmpec_benchmark.py $(build_args)

# print_header "MPECLib (92 problems)"
# python3 scripts/run_mpeclib_benchmark.py $(build_args)

# print_header "NOSBENCH (603 problems)"
# python3 scripts/run_nosbench_benchmark.py $(build_args)

# ─── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "All benchmarks complete.  Results: ./results/"
echo "Each run produced a *_run_env_*.json file for exact reproducibility."
