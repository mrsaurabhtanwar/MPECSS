# MPECLib

`MPECLib` is a benchmark collection of Mathematical Programs with Equilibrium Constraints (MPECs/MPCCs). It is part of the [GAMS World](https://github.com/GAMS-dev/gamsworld) model library initiative.

- **Original source (GAMS World)**: https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib

> **Note:** The `mpeclib-json/` and `mpeclib-gms/` folders are not tracked by Git. The 92 `.nl.json` and `.gms` problem files are available in **`benchmarks.zip`** — extract it into the `benchmarks/` folder to populate these directories.

---

## Original Contributors

**Steven Dirkse** — GAMS Development Corporation (GitHub: [@sdirkse67](https://github.com/sdirkse67)).

MPECLib was created as part of the GAMS World Performance Libraries, curated by **Michael Bussieck** ([@mbussieck](https://github.com/mbussieck)) at GAMS Development Corporation. It provides problems in GAMS scalar format and complements the MacMPEC collection by including larger-scale structural, friction, and traffic problems. It was used as a benchmark by Baumrucker, Renfro & Biegler (2008) for chemical engineering MPEC testing.

---

## Problem Formulation

Each problem has the standard MPEC form:

```
minimize    f(w)
subject to  lbw ≤ w ≤ ubw
            lbg ≤ g(w) ≤ ubg
            0 ≤ G(w) ⊥ H(w) ≥ 0
```

---

## Dataset Statistics

| Property              | Value                              |
|-----------------------|------------------------------------|
| Total problems        | 92                                 |
| Problem families      | 21                                 |
| Variable range        | n_x ∈ [3, ~5671]                   |
| Comp. pair range      | n_comp ∈ [1, ~2800]                |
| Original format       | GAMS scalar (`.gms`)               |
| Stored format         | CasADi JSON (`.nl.json`)           |

---

## Problem Families

| Family                  | Count | n_x Range     | Description                                    |
|-------------------------|-------|---------------|------------------------------------------------|
| `aampec` (1–6)          | 6     | ~72           | Academic AMPEC test problems                   |
| `bard` (1–3)            | 3     | 5–15          | Bilevel programming (Bard)                     |
| `bartruss3` (0–5)       | 6     | ~36           | Bar truss structural design (3 bars)            |
| `dempe` / `dempe2`      | 2     | 4–5           | Bilevel (Dempe)                                |
| `desilva`               | 1     | 7             | Traffic equilibrium (deSilva)                  |
| `ex9_1` (1m–4m)         | 4     | ~9            | Small academic examples                        |
| `find-a` (10–35/l/s/t)  | 12    | ~50–350       | FIND problems, variant A                       |
| `find-b` (10–35/l/s/t)  | 12    | ~50–350       | FIND problems, variant B                       |
| `find-c` (10–35/l/s/t)  | 12    | ~50–350       | FIND problems, variant C                       |
| `fjq1`                  | 1     | 8             | Academic                                       |
| `frictionalblock` (1–6) | 6     | ~200–5671     | Friction contact mechanics (largest problems)  |
| `gauvin` / `hq1` / `mss`| 3     | 3–6           | Small academic                                 |
| `kehoe` (1–3)           | 3     | ~11           | Economic equilibrium (Kehoe)                   |
| `kojshin` (3–4)         | 2     | 5             | Game theory (Kojima–Shindo)                    |
| `nappi` (a–d)           | 4     | ~113          | Traffic equilibrium (Nappi)                    |
| `outrata` (31–34)       | 4     | ~6            | Bilevel (Outrata)                              |
| `oz3`                   | 1     | 7             | Academic                                       |
| `qvi` / `three`         | 2     | 3–5           | Quasi-variational inequality / academic        |
| `tinloi`                | 1     | ~100          | Structural (Tin-Loi)                           |
| `tinque` (variants)     | 6     | ~500–1000     | Structural elastoplastic analysis              |
| `tollmpec`              | 1     | ~2400         | Large traffic equilibrium (largest problem)    |

**Size distribution**: small (n_x < 50) ≈ 35 problems · medium (50–500) ≈ 40 · large (≥ 500) ≈ 17 problems.

---

## File Structure

```
mpeclib-gms/        # Original GAMS scalar source files (.gms) — 92 files
mpeclib-json/       # CasADi-serialised NLP files (.nl.json) — 92 files
```

The `.gms` files were converted to `.nl.json` using the `scripts/convert_mpeclib.py` script, which parses GAMS variable/equation/complementarity declarations and builds CasADi symbolic expressions.

---

## JSON File Format

Each `.nl.json` file uses a CasADi-serialised format. The MPECLib schema **differs from the MacMPEC schema** in several ways:

- There is **no `name` field** (the loader derives the problem name from the filename).
- Extra fields `"p"` and `"p0"` are always present (serialised parameter vector and its initial value; always empty for MPECLib problems: `"p0": []`).
- `"g_fun"` is always present, stored as `null` when there are no general constraints (rather than being absent).
- `"lbg"` and `"ubg"` are always present, stored as `[]` when there are no constraints.
- Infinite bounds are stored as `1e+20` / `-1e+20` (not as `Infinity`).
- `"lbG"` and `"lbH"` are always proper lists, never scalars — even when n_comp = 1.
- `"lbG"` may contain `0.0` (standard) or `-1e+20` (unconstrained) per pair. `"ubH"` may be finite.

| Field      | Description                                                              |
|------------|--------------------------------------------------------------------------|
| `f_fun`    | Serialised objective `f(w)`                                              |
| `w`        | Serialised CasADi symbolic variable vector (stored but not read by the loader) |
| `lbw`      | Lower bounds on `w`                                                      |
| `ubw`      | Upper bounds on `w`                                                      |
| `w0`       | Initial guess for `w`                                                    |
| `p`        | Serialised parameter vector (always empty; not read by the loader)       |
| `p0`       | Initial parameter values — always `[]` for MPECLib problems              |
| `f_fun`    | Serialised objective `f(w)`                                              |
| `g_fun`    | Serialised general constraint function `g(w)`, or `null` when none       |
| `lbg`      | Lower bounds on `g(w)` — `[]` when `g_fun` is null                      |
| `ubg`      | Upper bounds on `g(w)` — `[]` when `g_fun` is null                      |
| `G_fun`    | Serialised complementarity function `G(w)`                               |
| `H_fun`    | Serialised complementarity function `H(w)`                               |
| `lbG`      | Lower bounds per comp. pair on `G(w)` — `0.0` (standard) or `-1e+20` (unconstrained) |
| `ubG`      | Upper bounds per comp. pair on `G(w)` — typically `1e+20`               |
| `lbH`      | Lower bounds per comp. pair on `H(w)` — typically `0.0`, may be `-1e+20` or other finite values |
| `ubH`      | Upper bounds per comp. pair on `H(w)` — typically `1e+20`, may be finite |

### Fields read by the loader

`mpeclib_loader.py` reads: `lbw`, `ubw`, `w0`, `f_fun`, `G_fun`, `H_fun`, `g_fun`, `lbg`, `ubg`, `lbG`, `ubG`, `lbH`, `ubH`.  
It does **not** read `w`, `p`, or `p0`, and derives the problem name from the filepath rather than from any JSON field.

---

## Loading in Python

```python
from mpecss.helpers.loaders.mpeclib_loader import load_mpeclib

problem = load_mpeclib("benchmarks/mpeclib/mpeclib-json/frictionalblock_1.nl.json")
# Returns a problem dict with CasADi Function objects and numpy arrays
```

The MPECLib loader (`mpeclib_loader.py`) is **distinct from the MacMPEC loader** and handles MPECLib-specific features including non-standard complementarity bounds, family detection across 23 pattern groups, and a bound-tightening pass (`_tighten_linear_bounds`) that infers tighter variable lower bounds from simple linear rows in G and H.

---

## Literature

If you use MPECLib, please cite:

```bibtex
@techreport{Dirkse2004MPECLib,
  author      = {Dirkse, Steven},
  title       = {{MPECLib}: A collection of mathematical programs with equilibrium constraints},
  institution = {GAMS Development Corporation},
  year        = {2004},
  note        = {Available at \url{https://github.com/GAMS-dev/gamsworld/tree/master/MPECLib}}
}
```

Key papers that benchmark with MPECLib:

```bibtex
@article{BaumruckerRenfroBiegler2008,
  author  = {Baumrucker, Brian T. and Renfro, John G. and Biegler, Lorenz T.},
  title   = {{MPEC} problem formulations and solution strategies with chemical engineering applications},
  journal = {Computers \& Chemical Engineering},
  volume  = {32},
  number  = {12},
  pages   = {2903--2913},
  year    = {2008},
  doi     = {10.1016/j.compchemeng.2008.02.010}
}

@article{BaumruckerBiegler2009,
  author  = {Baumrucker, Brian T. and Biegler, Lorenz T.},
  title   = {{MPEC} strategies for optimization of a class of hybrid dynamic systems},
  journal = {Journal of Process Control},
  volume  = {19},
  number  = {8},
  pages   = {1248--1256},
  year    = {2009},
  doi     = {10.1016/j.jprocont.2009.02.010}
}
```
