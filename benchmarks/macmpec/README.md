
# MacMPEC

`MacMPEC` is the most widely-used benchmark suite of Mathematical Programs with Equilibrium Constraints (MPECs/MPCCs). It was created by Sven Leyffer and is documented on his personal wiki at Argonne National Laboratory.

- **Original source (wiki)**: https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC
- **Original AMPL files**: http://www.mcs.anl.gov/~leyffer/macmpec/MacMPEC.tar.gz
- **CasADi JSON download**: https://cloud.syscop.de/s/rBnTMocFoLcNLWG

> **Note:** The `macmpec-json/` folder is not tracked by Git. The 191 `.nl.json` problem files are available in **`benchmarks.zip`** — extract it into the `benchmarks/` folder to populate this directory.

---

## Contributors

### Original Collection

**Sven Leyffer** — Argonne National Laboratory (ANL), Mathematics and Computer Science Division.

MacMPEC was curated and made publicly available by Leyffer as a standard reference collection for MPEC algorithm development and benchmarking. Problems were originally written in [AMPL](http://www.ampl.com/) (`.mod` / `.dat` files). Many problems were contributed or adapted from the broader optimization community (Bard, Outrata, Dempe, Scholtes, Ralph, and others). All major MPEC papers since 2001 use MacMPEC as a primary benchmark.

### CasADi JSON Conversion

**Armin Nurkanović, Anton Pozharskiy, and Moritz Diehl** — Systems Control and Optimization Laboratory (SYSCOP), Department of Microsystems Engineering (IMTEK), University of Freiburg.

The AMPL files were converted to CasADi JSON format as part of their [NOSBENCH paper](https://doi.org/10.1007/s10013-024-00680-2) (Nurkanović et al., *Vietnam Journal of Mathematics*, 2024). The conversion pipeline first generated `.nl` files from the AMPL `.mod`/`.dat` sources, then read them in using a modified version of CasADi to produce MPCCs in the CasADi JSON format described in Section 4.1 of the paper. The resulting files are hosted on the SYSCOP cloud. The original paper reports extracting 95 of 106 problems; the current collection contains 191 `.nl.json` files (reflecting later expansions of the problem set). Further details on the benchmarking methodology are in Pozharskiy's master's thesis (Albert-Ludwigs-Universität Freiburg, 2023).

---

## Problem Formulation

Each problem in MacMPEC has the form:

```
minimize    f(w)
subject to  lbw ≤ w ≤ ubw
            lbg ≤ g(w) ≤ ubg
            0 ≤ G(w) ⊥ H(w) ≥ 0
```

where `⊥` denotes complementarity: `G_i(w) · H_i(w) = 0` for all `i`.

---

## Dataset Statistics

| Property              | Value                         |
|-----------------------|-------------------------------|
| Total problems        | 191                           |
| Variable range        | n_x ∈ [2, 3436]               |
| Comp. pair range      | n_comp ∈ [1, 1500]            |
| Median variables      | n_x ≈ 50                      |
| Original format       | AMPL (`.mod` / `.dat`)        |
| Stored format         | CasADi JSON (`.nl.json`)      |

Notable large problems: `bem-milanc30-s` (n_x = 3436), `siouxfls` (n_x = 2402), `incid-set*-32` (n_x ≈ 1989), `qpec-200-*` (n_x ≈ 220).

> **Note:** The collection has grown over time. Fletcher & Leyffer (2004) used ~130 problems; Hoheisel et al. (2013) used 137 problems; the current version has 191 problems.

---

## Problem Classification

Each problem on the MacMPEC wiki carries a classification code of the form `XYR-ZW-TYPE-n-m-p`, where:

| Position | Values | Meaning |
|----------|--------|---------|
| X | L/Q/O/S | Objective type: Linear, Quadratic, general, Sum-of-squares |
| Y | L/Q/U/B/O | Constraint type: Linear, Quadratic, Unconstrained, Bounded, general |
| R | R | Regularity marker |
| Z | A/M | AMPL model source: single-file (A) or model+data (M) |
| W | Y/N | Has separate data file |
| TYPE | NLP/NCP/LCP/MCP | Complementarity formulation type |
| n, m, p | integers | Number of variables, constraints, complementarity pairs |

---

## Problem Sources

Problems in MacMPEC originate from diverse application domains:

- **Bilevel programming** — `bard*`, `bilevel*`, `dempe`, `ex9.*`, `outrata31`–`outrata34`
- **Design optimisation** — `design-cent-*`, `bar-truss-3`
- **Nash equilibrium / game theory** — `nash*`, `gnash*`, `stackelberg1`
- **Contact mechanics** — `bem-milanc30-s`, `jr*`
- **Traffic assignment** — `siouxfls`, `siouxfls1`, `incid-set-*`, `tap-*`, `taxmcp`, `water-FL`, `water-net`
- **Portfolio optimisation** — `portfl-i-*`, `portfl1`–`portfl6`
- **Discrete optimisation relaxations** — `hs044-i`, `flp*`
- **Packing problems** — `pack-comp*`, `pack-rig*`
- **Quadratic programs with comp. constraints** — `qpec-100-*`, `qpec-200-*`, `qpec1`, `qpec2`
- **Piecewise / least-squares** — `liswet1-*`, `sl1`, `bilin`
- **Academic test problems** — `kth*`, `gauvin`, `hakonsen`, `monteiro`, `monteiroB`, `desilva`, `df1`, `scholtes*`, `ralph*`, `ralphmod`, `scale*`

> **Note on portfl-i-\*:** Files portfl-i-1, portfl-i-2, portfl-i-3, portfl-i-4, portfl-i-6 are present; portfl-i-5 is absent from the collection.

---

## JSON File Format

Each `.nl.json` file contains a CasADi-serialised NLP with the following fields:

| Field     | Description                                                                                         |
|-----------|-----------------------------------------------------------------------------------------------------|
| `name`    | Name of the original AMPL `.nl` file (e.g. `"bard1.nl"`). The loader derives the clean problem name from the filepath instead and does not read this field. |
| `f_fun`   | Serialised objective function `f(w)`                                                                |
| `w`       | Serialised CasADi symbolic variable vector (stored but not read by the loader)                      |
| `lbw`     | Lower bounds on `w`                                                                                 |
| `ubw`     | Upper bounds on `w`                                                                                 |
| `w0`      | Initial guess for `w`                                                                               |
| `g_fun`   | Serialised general constraint function `g(w)` (absent when there are no general constraints)        |
| `lbg`     | Lower bounds on `g(w)` (empty list `[]` when `g_fun` is absent)                                     |
| `ubg`     | Upper bounds on `g(w)` (empty list `[]` when `g_fun` is absent)                                     |
| `G_fun`   | Serialised complementarity function `G(w)`                                                          |
| `H_fun`   | Serialised complementarity function `H(w)`                                                          |
| `lbG`     | Lower bound per comp. pair on `G(w)`. **Stored as `-Infinity`** (G is unconstrained in the file); the loader sanitizes `-Infinity` → `0.0`, giving the effective constraint `G(w) ≥ 0`. When n_comp = 1 this may be a scalar rather than a list. |
| `ubG`     | Upper bound per comp. pair on `G(w)`. Stored as `Infinity` (no upper bound). |
| `lbH`     | Lower bound per comp. pair on `H(w)`. Stored as `0` (H ≥ 0 enforced). When n_comp = 1 this may be a scalar. |
| `ubH`     | Upper bound per comp. pair on `H(w)`. Stored as `Infinity`. |

Functions are serialised using `casadi.Function.serialize()` and can be loaded with `casadi.Function.deserialize(s)`.

### Fields read by the loader

`macmpec_loader.py` reads: `lbw`, `ubw`, `w0`, `f_fun`, `g_fun`, `lbg`, `ubg`, `G_fun`, `H_fun`, `lbG`, `lbH`.  
It does **not** read `name` or `w` — the problem name is derived from the filename, and the symbolic variable `w` is reconstructed from scratch.

---

## Loading in Python

```python
from mpecss.helpers.loaders.macmpec_loader import load_macmpec

problem = load_macmpec("benchmarks/macmpec/macmpec-json/bard1.nl.json")
# Returns a problem dict with CasADi Function objects and numpy arrays
```

---

## Literature

If you use MacMPEC, please cite:

```bibtex
@techreport{Leyffer2000MacMPEC,
  author      = {Leyffer, Sven},
  title       = {{MacMPEC}: {AMPL} collection of {MPEC}s},
  institution = {Argonne National Laboratory},
  year        = {2000},
  note        = {Available at \url{https://wiki.mcs.anl.gov/leyffer/index.php/MacMPEC}}
}
```

Key papers that benchmark with MacMPEC:

```bibtex
@article{FletcherLeyffer2004,
  author  = {Fletcher, Roger and Leyffer, Sven},
  title   = {Solving mathematical programs with complementarity constraints as nonlinear programs},
  journal = {Optimization Methods and Software},
  volume  = {19},
  number  = {1},
  pages   = {15--40},
  year    = {2004},
  doi     = {10.1080/10556780410001654241}
}

@article{HoheiselKanzowSchwartz2013,
  author  = {Hoheisel, Tim and Kanzow, Christian and Schwartz, Alexandra},
  title   = {Theoretical and numerical comparison of relaxation methods for mathematical programs with complementarity constraints},
  journal = {Mathematical Programming},
  volume  = {137},
  number  = {1--2},
  pages   = {257--288},
  year    = {2013},
  doi     = {10.1007/s10107-011-0488-5}
}
```

CasADi JSON conversion (NOSBENCH paper):

```bibtex
@article{NurkanovicPozharskiyDiehl2024,
  author  = {Nurkanovi\'{c}, Armin and Pozharskiy, Anton and Diehl, Moritz},
  title   = {Solving Mathematical Programs with Complementarity Constraints Arising in Nonsmooth Optimal Control},
  journal = {Vietnam Journal of Mathematics},
  year    = {2024},
  doi     = {10.1007/s10013-024-00680-2}
}
```
