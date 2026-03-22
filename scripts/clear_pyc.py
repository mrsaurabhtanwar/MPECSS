#!/usr/bin/env python3
"""
clear_pyc.py — Remove all stale Python bytecode caches under the project root.

Run this once after pulling changes or editing source files to guarantee
Python loads the updated .py files rather than old .pyc snapshots.

Usage:
    python scripts/clear_pyc.py            # dry-run (shows what would be removed)
    python scripts/clear_pyc.py --delete   # actually deletes the files
"""
import os
import sys
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DELETE = "--delete" in sys.argv


def scan(root):
    removed_files = 0
    removed_dirs  = 0

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden dirs and the venv
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".")
            and d not in ("venv", ".venv", "node_modules")
        ]

        # Remove individual .pyc files (Python 2 style, in same dir as .py)
        for fname in filenames:
            if fname.endswith(".pyc") or fname.endswith(".pyo"):
                fpath = os.path.join(dirpath, fname)
                if DELETE:
                    os.remove(fpath)
                    print(f"  deleted  {fpath}")
                else:
                    print(f"  [dry-run] would delete  {fpath}")
                removed_files += 1

        # Remove entire __pycache__ directories (Python 3 style)
        if "__pycache__" in dirnames:
            cache_dir = os.path.join(dirpath, "__pycache__")
            if DELETE:
                shutil.rmtree(cache_dir)
                print(f"  deleted  {cache_dir}/")
            else:
                print(f"  [dry-run] would delete  {cache_dir}/")
            dirnames.remove("__pycache__")
            removed_dirs += 1

    return removed_files, removed_dirs


print(f"Scanning: {ROOT}")
print(f"Mode    : {'DELETE' if DELETE else 'dry-run (pass --delete to actually remove)'}")
print()

n_files, n_dirs = scan(ROOT)

print()
print(f"Found {n_files} .pyc/.pyo file(s) and {n_dirs} __pycache__ dir(s).")
if not DELETE:
    print("Re-run with  --delete  to remove them.")
else:
    print("Done. Python will recompile from source on next import.")
