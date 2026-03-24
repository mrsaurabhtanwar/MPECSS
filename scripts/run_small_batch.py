#!/usr/bin/env python3
"""
Small-batch benchmark runner for integrity checking.

Runs 5 MPECLib and 10 MacMPEC problems to detect unusual activity
before committing to full benchmark runs.

Usage:
    python scripts/run_small_batch.py
    python scripts/run_small_batch.py --workers 2
    python scripts/run_small_batch.py --tag "test-run"
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('small_batch')


def main():
    parser = argparse.ArgumentParser(
        description='Run small-batch benchmarks (5 MPECLib, 10 MacMPEC) for integrity testing.'
    )
    parser.add_argument('--workers', type=int, default=2, help='Number of parallel workers')
    parser.add_argument('--tag', type=str, default='SmallBatch', help='Tag for results CSV')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout per problem (seconds)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Verify we're in project root
    if not Path('scripts/run_wsl_parallel.sh').exists():
        logger.error("Must run from project root directory")
        return 1
    
    logger.info("=" * 70)
    logger.info("SMALL-BATCH BENCHMARK RUNNER")
    logger.info("=" * 70)
    logger.info(f"Tag       : {args.tag}")
    logger.info(f"Workers   : {args.workers}")
    logger.info(f"Timeout   : {args.timeout}s per problem")
    logger.info(f"Seed      : {args.seed}")
    logger.info("=" * 70)
    
    try:
        from mpecss.helpers.loaders.macmpec_loader import load_macmpec
        from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib
        from mpecss.helpers.benchmark_utils import run_benchmark_main
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure mpecss is installed and you're in the right virtualenv")
        return 1
    
    os.makedirs('results', exist_ok=True)
    
    # Run MPECLib small batch (first 5 problems)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: MPECLib Small Batch (5 problems)")
    logger.info("=" * 70)
    
    mpeclib_path = Path("benchmarks/mpeclib/macmpec-json")
    if mpeclib_path.exists():
        try:
            # Get first 5 problems
            json_files = sorted([f.name for f in mpeclib_path.glob("*.json")])[:5]
            logger.info(f"Running: {', '.join(json_files[:3])}...")
            
            logger.info("Starting MPECLib run...")
            
            # Reconstruct sys.argv for run_benchmark_main
            original_argv = sys.argv.copy()
            sys.argv = [
                'run_small_batch.py',
                '--path', str(mpeclib_path),
                '--workers', str(args.workers),
                '--timeout', str(args.timeout),
                '--seed', str(args.seed),
                '--tag', f"{args.tag}_mpeclib",
                '--num-problems', '5'
            ]
            
            run_benchmark_main(
                loader_fn=load_mpeclib,
                dataset_tag='mpeclib_batch',
                default_path=str(mpeclib_path),
            )
            
            sys.argv = original_argv
        except Exception as e:
            logger.error(f"MPECLib batch failed: {e}")
            return 1
    else:
        logger.warning(f"MPECLib data not found at {mpeclib_path}")
    
    # Run MacMPEC small batch (first 10 problems)
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: MacMPEC Small Batch (10 problems)")
    logger.info("=" * 70)
    
    macmpec_path = Path("benchmarks/macmpec/macmpec-json")
    if macmpec_path.exists():
        try:
            # Get first 10 problems
            json_files = sorted([f.name for f in macmpec_path.glob("*.json")])[:10]
            logger.info(f"Running: {', '.join(json_files[:3])}...")
            
            logger.info("Starting MacMPEC run...")
            
            # Reconstruct sys.argv for run_benchmark_main
            original_argv = sys.argv.copy()
            sys.argv = [
                'run_small_batch.py',
                '--path', str(macmpec_path),
                '--workers', str(args.workers),
                '--timeout', str(args.timeout),
                '--seed', str(args.seed),
                '--tag', f"{args.tag}_macmpec",
                '--num-problems', '10'
            ]
            
            run_benchmark_main(
                loader_fn=load_macmpec,
                dataset_tag='macmpec_batch',
                default_path=str(macmpec_path),
            )
            
            sys.argv = original_argv
        except Exception as e:
            logger.error(f"MacMPEC batch failed: {e}")
            return 1
    else:
        logger.warning(f"MacMPEC data not found at {macmpec_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Small-batch runs complete!")
    logger.info("Check results/ directory for CSV and JSON files.")
    logger.info("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
