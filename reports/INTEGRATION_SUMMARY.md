# Integration summary — every headline number and where it comes from

Branch `integration/vaughan-moo`. Every number below was computed in this repo
by the command or file named next to it. Nothing is quoted from the brief,
from Vedant's notes, or from an earlier report without being recomputed here.

## Gates

| Gate | Result | Source |
|---|---|---|
| **G1** emulator vs Vedant's exact solver, 100 random designs | **PASS** — mean rel. error 7.0e-9 (θ), 7.6e-9 (lengths); max 3.9e-8; 0 feasibility disagreements | `reports/gates/G1.json` |
| **G2** synthetic rows never test data; all feasible | **PASS** — splits train/val only; 1015/1015 feasible (emulator), 24/24 sampled feasible and identical (exact) | `tests/test_synthetic_v2.py` |
| **G3** Pareto designs re-run through his exact solver | **PASS** — 13 designs per study (knee included), 0 infeasible, worst rel. error 5.3e-8 (Shin), 7.0e-8 (CEERI) vs the 2 % limit | `reports/moo/*/verify.csv` |

## Physics

| Quantity | Value | Source |
|---|---|---|
| Langmuir–Blodgett cone constant | **14.665** µperv (full sphere 29.33; Vaughan prints 14.67) | `src/physics/lb.py` (CODATA) |
| Space-charge spread constant | 0.174090 (Vaughan 0.17409) | `src/physics/lb.py` |
| Old Tier B v1 labels vs exact solver | 3.22 % MRE (1000 rows) — used 29.33 (D10) | computed in session; `tests/test_physics_layer.py` asserts the constant |
| Exact solver θ_T vs the paper's iterative column | 0.55 % MRE (30 guns) | computed from `dataset_master.csv` |
| CEERI klystron, his final solver vs published θ | 28.797° vs 28.5652° (0.81 %); kf = 0.979 | `tests/test_physics_layer.py::test_ceeri_klystron_through_both` |
| Emulator speed | 100 designs ≈ 0.3 s; his solver 3–5 s each | `reports/gates/G1.json` |

## θ against measured guns — the honest table

7 held-out real guns (Panahi's asterisked cases), evaluated **once** after every
choice was frozen (`python -m src.ann.stage2`, `reports/synthetic_roles.json`).

| Model | Test MRE (7 guns) | Bootstrap 90 % | Other estimate |
|---|---|---|---|
| **M1** — ANN on 23 real guns (**headline**) | **7.93 %** | 4.7 – 11.4 % | restart median 8.33 % (`BENCHMARK.md`) |
| ANN-E (on synthetic v2) **vs the solver** | 0.08 % | — | judged against the solver only |
| ANN-E + Stage-2 correction | 5.40 % | 3.0 – 8.0 % | correction LOO on 16 in-envelope train guns 7.03 % |
| Calibrated closure (exact solver + ridge on P, ln r_w, ln C) | 2.88 % | 1.3 – 5.0 % | **LOO, all 30 guns: 5.51 %**; LOO 23 train: 6.84 % |
| — same, scale-free (P, ln C only; **post-hoc**) | 7.82 % | — | LOO 23 train: 8.35 % |
| Exact solver, uncalibrated | 10.37 % | 7.1 – 13.4 % | all 30 guns: 10.49 % |
| Paper's iterative column | 10.14 % | — | all 30 guns: 10.15 % |
| Panahi et al. ANN (as printed) | 3.74 % | — | not reproducible (`BENCHMARK.md`) |
| Yang 2006 modified non-iterative (paper's column) | 7.93 % | — | same as M1 to two decimals |

**Why M1 stays the headline although the calibrated closure scored lower.**
The calibrated closure's advantage comes entirely from ln r_w, and θ cannot
depend on r_w (a length scale; Vaughan's θ depends on P and C). The scale-free
version is no better than M1. See D11.

**What the design tool uses for "a measured gun would show":** θ_meas ≈
**1.046 × θ_T** (scalar k, scale-free), band **±13.8 %** (90th-percentile LOO
error on the 23 training guns). Source: `reports/synthetic_roles.json`.

## Optimisation

NSGA-II, population 100 × 500 generations = **50 000 evaluations per study**,
backend `lb-closure`, seed 1. Sources: `reports/moo/<study>/run.json`,
`knee.json`, `verify.csv`, `pareto.csv`, `egun_deck_<knee>.txt`.

| | Shin 35 GHz EIK | CEERI 8 MW klystron |
|---|---|---|
| Fixed current | 0.5 A | 120 A |
| Objectives (min) | J_c, θ_T, \|r_w − 0.2255\| | J_c, V, \|r_w − 9.8\| (throw 77.9 mm ± 10 %) |
| Wall time, 50 000 evaluations | 164 s | 189 s |
| Pareto designs | 100, all deck-constructible | 100, all deck-constructible |
| Knee (V, r_w, J_c) | 13.96 kV, 0.2405 mm, 4.14 A/cm² | 110.7 kV, 10.26 mm, 3.06 A/cm² |
| Knee θ_T / θ_raw | 16.43° / 14.83° | 45.32° / 40.05° |
| Knee R_c, z_ac, throw | 7.66, 4.65, 13.18 mm | 54.88, 29.96, 81.22 mm |
| Knee gap field V/z_ac | 3.0 kV/mm | 3.7 kV/mm |
| If built and measured (k × θ_T ± 13.8 %) | 17.2° (14.8 – 19.6°) | 47.4° (40.8 – 54.0°) |
| θ_T sensitivity to ±5 % V | +4.0 % / −3.7 % | +4.6 % / −4.1 % |
| θ_T sensitivity to ±5 % r_w, J_c | ∓1.0 %, ∓0.5 % | ∓2.2 %, ∓1.1 % |
| Published design for comparison | 14 kV, 0.2255 mm, 5 A/cm²: θ_T 16.29°, feasible | 150 kV, 9.8 mm, 5 A/cm²: θ_T 32.16°, feasible |

**What the fronts say.** Shin: the published 5 A/cm² can drop to ~4.1 A/cm²
(longer cathode life) for a 6.7 % larger waist at essentially the same angle.
CEERI: the voltage/loading objectives push θ_T up until Vedant's 67.5°
focus-electrode construction stops closing (~45°); D12's deck constraint is
the active limit, which is why the whole front sits at θ_T 43.7–45.3°. Before
that constraint was added, 0 of 100 CEERI Pareto designs could be turned into a
deck.

## What this does NOT claim

- **It does not beat the paper's 3.74 %, and does not reproduce it.** Against
  measured guns every honest model here sits at 7–10 %; the exact classical
  solver at 10.5 %. The one lower number (2.88 %) is explained above and is not
  claimed.
- **The emulator is not "more accurate than Vaughan".** It reproduces Vaughan
  (via Vedant) to ~1e-8. Being exactly Vaughan is all it claims; Vaughan itself
  is ~10 % from measured guns.
- **ANN-E's 0.08 % is agreement with a solver, not with reality.**
- **Pareto fronts are optima of the classical model**, not of real guns. A
  measured gun built to a knee design should show θ within the ±13.8 % band,
  and the band itself comes from 23 guns.
- **Decks are EGUN inputs, not simulation results.** No trajectory or field
  solve was run. The focus-electrode angle (67.5°) and default geometry
  parameters are Vedant's starting values, which a simulation refines.
- **Sharma et al. (2001) aperture synthesis is not implemented.** Designs with
  θ_raw > 56.44° get no deck and are kept off the fronts.
- **The "already built integration" (94 tests) was never seen.** The file of
  that name was a text note; this branch was built from scratch.
- **Vedant has not yet given permission** for his solver to be vendored.
