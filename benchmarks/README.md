# Benchmarks

This folder contains three benchmark suites of Mathematical Programs with Complementarity Constraints (MPCCs), all serialised in formats compatible with [CasADi](https://web.casadi.org/). Together they cover 886 distinct MPCC problems ranging from small academic test cases to large-scale nonsmooth optimal control problems.

> **Note:** The problem data files (`.json`, `.nl.json`, `.gms`, `.mat`) are **not tracked by Git** due to their size. The full dataset is available as **`benchmarks.zip`** — download and extract it into this folder before running any experiments.

---

## Directory Structure

```
benchmarks/
├── README.md                   ← this file
│
├── macmpec/
│   ├── README.md               ← dataset description, contributors, citation
│   └── macmpec-json/           ← 191 problems as CasADi JSON (.nl.json)
│
├── mpeclib/
│   ├── README.md               ← dataset description, contributors, citation
│   ├── mpeclib-gms/            ← 92 original GAMS source files (.gms)
│   └── mpeclib-json/           ← 92 problems as CasADi JSON (.nl.json)
│
└── nosbench/
    ├── README.md               ← dataset description, naming convention, citation
    ├── nosbench-json/          ← 603 problems as CasADi JSON (.json)
    └── nosbench-mat/           ← 1576 MATLAB .mat files (organised by level)
        ├── generators/
        ├── level1/
        ├── level2/
        ├── level3/
        └── level4/
```

---

## Benchmark Suites at a Glance

| Suite      | Problems | Format(s)            | Problem Type                        | Source |
|------------|----------|----------------------|-------------------------------------|--------|
| MacMPEC    | 191      | CasADi JSON          | Academic, bilevel, traffic, design  | Argonne / SYSCOP Freiburg |
| MPECLib    | 92       | CasADi JSON + GAMS   | Structural, friction, traffic, game | GAMS World |
| NOSBENCH   | 603      | CasADi JSON + MATLAB | Nonsmooth OCP time-discretisations  | SYSCOP Freiburg |
| **Total**  | **886**  |                      |                                     | |

---

## File Formats

### CasADi JSON (`.nl.json` / `.json`)

All three suites provide problems as JSON files containing CasADi-serialised functions. Each file encodes a complete MPCC in the form:

```
minimize    f(w, p)
subject to  lbw ≤ w ≤ ubw
            lbg ≤ g(w, p) ≤ ubg
            0 ≤ G(w, p) ⊥ H(w, p) ≥ 0
```

| JSON key   | Contents                                                              |
|------------|-----------------------------------------------------------------------|
| `w`        | Serialised decision variable vector                                   |
| `lbw`      | Lower bounds on `w`                                                   |
| `ubw`      | Upper bounds on `w`                                                   |
| `w0`       | Initial guess for `w`                                                 |
| `f_fun`    | Serialised objective function `f(w, p)` (MacMPEC/MPECLib)            |
| `augmented_objective_fun` | Serialised objective `f(w, p)` (NOSBENCH primary key; `objective_fun` is fallback) |
| `g_fun`    | Serialised constraint function `g(w, p)`                             |
| `lbg`      | Lower bounds on `g`                                                   |
| `ubg`      | Upper bounds on `g`                                                   |
| `G_fun`    | Serialised complementarity function `G(w, p)`                        |
| `H_fun`    | Serialised complementarity function `H(w, p)`                        |
| `p`        | Serialised parameter vector (NOSBENCH only)                          |
| `p0`       | Default parameter values (NOSBENCH only — **not** `p_val`)           |

Functions are serialised with `casadi.Function.serialize()` and can be loaded with `casadi.Function.deserialize(s)`.

### MATLAB `.mat` (NOSBENCH only)

The `nosbench-mat/` subdirectory contains MATLAB `.mat` files for use with [NOSNOC](https://github.com/nosnoc/nosnoc) v0.5.0 (MATLAB). Problems are organised into four difficulty levels:

| Subfolder    | Description                                                     |
|--------------|-----------------------------------------------------------------|
| `generators/` | NOSNOC generator scripts for each OCP                          |
| `level1/`    | Smallest problems — 440 files (simple OCPs, small n_x)         |
| `level2/`    | Medium problems — 448 files (medium N_stages/n_x)              |
| `level3/`    | Large problems — 384 files (large N_stages/n_x)                |
| `level4/`    | Largest problems — 304 files (largest N_stages/n_x)            |

> **Note:** Levels represent problem size tiers (ordered by number of variables/discretisation steps), not OCP source type. All four source types (Filippov, CLS, time-freezing, hysteresis) appear across multiple levels.

---

## Loading Problems in Python

```python
# MacMPEC
from mpecss.helpers.loaders.macmpec_loader import load_macmpec

problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")

# MPECLib
from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib

problem = load_mpeclib("benchmarks/mpeclib/mpeclib-json/bard1.nl.json")

# NOSBENCH
from mpecss.helpers.loaders.nosbench_loader import load_nosbench

problem = load_nosbench("benchmarks/nosbench/nosbench-json/<problem>.json")
```

All loaders return a unified problem dict — see `mpecss/helpers/loaders/` for the full API.

> **Note:** The MPECLib GAMS-to-JSON conversion script (`scripts/convert_mpeclib.py`) is required to
> regenerate the `.nl.json` files from the raw `.gms` sources. The pre-converted files are included in
> `benchmarks.zip` so the script is only needed if you modify or extend the source problems.

---

## Individual Suite Documentation

Each subdirectory contains its own `README.md` with:
- Problem formulation and dataset statistics
- Problem classification and naming conventions
- Full contributor and citation information
- JSON schema details

---

## Contributors & Credits

| Suite    | Original Source             | CasADi Conversion            |
|----------|-----------------------------|------------------------------|
| MacMPEC  | Sven Leyffer (ANL)          | Nurkanović, Pozharskiy, Diehl (Univ. Freiburg) |
| MPECLib  | Steven Dirkse (GAMS Corp.)  | Anton Pozharskiy (Univ. Freiburg) |
| NOSBENCH | Nurkanović, Pozharskiy, Diehl (Univ. Freiburg) | — (native CasADi format) |

---

## References

```bibtex
@techreport{Leyffer2000MacMPEC,
  author      = {Leyffer, Sven},
  title       = {{MacMPEC}: {AMPL} collection of {MPEC}s},
  institution = {Argonne National Laboratory},
  year        = {2000}
}

@misc{Dirkse2004MPECLib,
  author = {Dirkse, Steven},
  title  = {{MPECLib}: a collection of mathematical programs with equilibrium constraints},
  year   = {2004},
  url    = {https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib}
}

@article{NurkanovicPozharskiyDiehl2024,
  author  = {Nurkanovi\'{c}, Armin and Pozharskiy, Anton and Diehl, Moritz},
  title   = {Solving Mathematical Programs with Complementarity Constraints Arising in Nonsmooth Optimal Control},
  journal = {Vietnam Journal of Mathematics},
  volume  = {53},
  pages   = {659--697},
  year    = {2025},
  doi     = {10.1007/s10013-024-00704-z}
}
```
