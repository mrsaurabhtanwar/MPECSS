# NOSBENCH

`NOSBENCH` is a benchmark suite of Mathematical Programs with Complementarity Constraints (MPCCs) arising from time-discretisation of nonsmooth optimal control problems (OCPs).

- **GitHub repository**: https://github.com/nosnoc/nosbench

> **Note:** The `nosbench-json/` and `nosbench-mat/` folders are not tracked by Git. The 603 `.json` problem files and MATLAB `.mat` files are available in **`benchmarks.zip`** — extract it into the `benchmarks/` folder to populate these directories.

Each problem has the parameterised MPCC form:

```
minimize    f(w, p)
subject to  lbw ≤ w ≤ ubw
            lbg ≤ g(w, p) ≤ ubg
            0 ≤ G(w, p) ⊥ H(w, p) ≥ 0
```

Problems are provided in two formats:
- **CasADi JSON** (`.json`) — load with Python/CasADi
- **MATLAB `.mat`** — compatible with [`nosnoc`](https://github.com/nosnoc/nosnoc) v0.5.0
  > ⚠️ The `.mat` files were generated with NOSNOC v0.5.0. Compatibility with later releases (v0.6.0+) is not guaranteed.

---

## Original Contributors

**Armin Nurkanović**, **Anton Pozharskiy**, **Moritz Diehl** — University of Freiburg / Systems Control and Optimization Laboratory (SYSCOP).

NOSBENCH was generated using the [NOSNOC](https://github.com/nosnoc/nosnoc) software for nonsmooth optimal control (Nurkanović & Diehl, 2022). Each MPCC is obtained by applying a finite-element discretisation (FESD — Finite Elements with Switch Detection) to one of 33 nonsmooth OCPs. The benchmark generation code was primarily implemented by **Anton Pozharskiy** (GitHub: [@apozharski](https://github.com/apozharski)), who is the sole code contributor to the [nosbench](https://github.com/nosnoc/nosbench) repository.

NOSBENCH is the first benchmark suite specifically designed for MPCCs from optimal control, and fills the gap left by MacMPEC/MPECLib (which contain mostly small academic problems).

---

## Dataset Statistics

| Property              | Value                                      |
|-----------------------|--------------------------------------------|
| Total problems        | 603                                        |
| Underlying OCPs       | 33                                         |
| Variable range        | n_x up to 14,622 *(per Pozharskiy 2023)*   |
| Median variables      | n_x ≈ 1,771 *(per Pozharskiy 2023)*        |
| Original format       | NOSNOC/FESD time-discretisation            |
| Stored formats        | CasADi JSON (`.json`) · MATLAB (`.mat`)    |

---

## Benchmark Subsets

| Subset        | Size | Description                                             |
|---------------|------|---------------------------------------------------------|
| NOSBENCH-S    | 100  | Simple Filippov + simple CLS/time-freezing problems     |
| NOSBENCH-RS   | 32   | Representative small (easiest to hardest)               |
| NOSBENCH-RL   | 167  | Representative large (main benchmark)                   |
| NOSBENCH-F    | 603  | Full benchmark                                          |

Subset definitions are sourced from Pozharskiy (2023) and listed in [`results/nosbench_subsets.txt`](../../results/nosbench_subsets.txt).

---

## File Naming Convention

Filenames encode the full problem configuration. The naming is defined by `generate_problem_name.m` in the NOSBENCH repository:

```
name = join([model_name, init_cond_idx, N_stages, N_finite_elements,
             n_s, irk_scheme, dcs_mode, cross_comp_mode,
             source_type, lift_complementarities], '_')
```

Example:

```
SCHUMI_001_050_002_4_RIIA_STEP_3_FIL_0.json
│       │   │   │   │  │    │    │  │   └─ lift_complementarities (0 or 1)
│       │   │   │   │  │    │    │  └───── source_type (FIL/ELC/IEC/HYS)
│       │   │   │   │  │    │    └──────── cross_comp_mode (1, 3, 4, or 7)
│       │   │   │   │  │    └───────────── dcs_mode (STEP/STEWART/CLS)
│       │   │   │   │  └────────────────── irk_scheme (RIIA/GL/ERK)
│       │   │   │   └───────────────────── n_s — RK stages per finite element
│       │   │   └────────────────────────── N_finite_elements per stage (%03d)
│       │   └────────────────────────────── N_stages — control intervals (%03d)
│       └────────────────────────────────── init_cond_idx — initial condition index (%03d)
└────────────────────────────────────────── OCP family name (model_name)
```

**Field details**:

| # | Field                    | Format   | Description                                          |
|---|--------------------------|----------|------------------------------------------------------|
| 1 | `model_name`             | string   | OCP family (e.g., SCHUMI, OSCILL, BOUNCING_BALL)     |
| 2 | `init_cond_idx`          | `%03d`   | Initial condition scenario index                     |
| 3 | `N_stages`               | `%03d`   | Number of control stages (time intervals)            |
| 4 | `N_finite_elements`      | `%03d`   | Finite elements per control stage                    |
| 5 | `n_s`                    | `%d`     | Runge-Kutta stages per finite element                |
| 6 | `irk_scheme`             | string   | IRK scheme: `RIIA` (Radau IIA), `GL` (Gauss-Legendre), `ERK` (Explicit RK) |
| 7 | `dcs_mode`               | string   | Discretisation of complementarity: `STEP`, `STEWART`, `CLS` |
| 8 | `cross_comp_mode`        | `%d`     | Cross-complementarity mode (1, 3, 4, or 7)           |
| 9 | `source_type`            | string   | `FIL` (Filippov), `ELC` (elastic contact), `IEC` (inelastic contact), `HYS` (hysteresis) |
| 10| `lift_complementarities` | `0`/`1`  | Whether complementarity variables are lifted         |

> **JSON subset note:** The 603 problems in `nosbench-json/` use only `cross_comp_mode ∈ {3, 4, 7}` and `lift_complementarities = 0`. Values `1` (for `cross_comp_mode`) and `1` (for `lift_complementarities`) appear only in the `.mat` files under `nosbench-mat/`.

---

## JSON File Format

Each `.json` file contains a CasADi-serialised parameterised MPCC with the following fields:

| Field                          | Description                                                     |
|--------------------------------|-----------------------------------------------------------------|
| `w`                            | Serialised decision variable vector                             |
| `lbw`                          | Lower bounds on `w`                                             |
| `ubw`                          | Upper bounds on `w`                                             |
| `w0`                           | Initial guess for `w`                                           |
| `p`                            | Serialised parameter vector                                     |
| `p0`                           | Default parameter values (use this key — **not** `p_val`)       |
| `g_fun(w,p)`                   | Serialised general constraint function `g`                      |
| `lbg`                          | Lower bounds on `g(w,p)`                                        |
| `ubg`                          | Upper bounds on `g(w,p)`                                        |
| `G_fun(w,p)`                   | Serialised complementarity function `G`                         |
| `H_fun(w,p)`                   | Serialised complementarity function `H`                         |
| `augmented_objective_fun(w,p)` | Serialised objective `f(w,p)` — **primary key**                 |
| `objective_fun(w,p)`           | Serialised objective `f(w,p)` — fallback key if primary is absent |

Functions are serialised using CasADi `Function.serialize()` and can be loaded with `casadi.Function.deserialize(s)`.

---

## Loading in Python

```python
# FIX: correct import path (module is mpecss.helpers.loaders, not mpecss)
from mpecss.helpers.loaders.nosbench_loader import load_nosbench

problem = load_nosbench("benchmarks/nosbench/nosbench-json/SCHUMI_001_050_002_4_RIIA_STEP_3_FIL_0.json")
# Returns a problem dict with CasADi Function objects, numpy arrays for bounds/initial guess
```

---

## Literature

Primary reference:

> [Solving mathematical programs with complementarity constraints arising in nonsmooth optimal control](https://link.springer.com/article/10.1007/s10013-024-00704-z) \
> A. Nurkanović, A. Pozharskiy, M. Diehl \
> Vietnam J. Math. (2024).

```bibtex
@article{Nurkanovic2024,
  title     = {Solving mathematical programs with complementarity constraints arising in nonsmooth optimal control},
  author    = {Nurkanovi{\'c}, Armin and Pozharskiy, Anton and Diehl, Moritz},
  journal   = {Vietnam Journal of Mathematics},
  volume    = {53},
  pages     = {659--697},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/s10013-024-00704-z}
}
```

NOSBENCH generation methodology:

> [Evaluating Methods for Solving Mathematical Programs With Complementarity Constraints Arising From Nonsmooth Optimal Control](https://publications.syscop.de/Pozharskiy2023.pdf) \
> A. Pozharskiy \
> Master's Thesis, Albert-Ludwigs-University Freiburg, 2023.

```bibtex
@mastersthesis{Pozharskiy2023,
  year   = {2023},
  school = {Albert-Ludwigs-University Freiburg},
  author = {Anton Pozharskiy},
  title  = {Evaluating Methods for Solving Mathematical Programs With Complementarity Constraints Arising From Nonsmooth Optimal Control.}
}
```

Underlying NOSNOC software:

```bibtex
@article{NurkanovicDiehl2022NOSNOC,
  author  = {Nurkanovi{\'c}, Armin and Diehl, Moritz},
  title   = {{NOSNOC}: A Software Package for Numerical Optimal Control of Nonsmooth Systems},
  journal = {IEEE Control Systems Letters},
  volume  = {6},
  pages   = {3110--3115},
  year    = {2022},
  doi     = {10.1109/LCSYS.2022.3181800}
}
```
