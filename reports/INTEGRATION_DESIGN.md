# Integration design — Vedant's exact synthesis × our surrogate × NSGA-II

**Spec in → Pareto front of gun designs → each verified by Vedant's exact solver →
EGUN deck out.**

Branch `integration/vaughan-moo`. Built from scratch: the "already built
integration" archive described in the brief was not among the files supplied
(`docs/context/README.md` records what arrived and under which names).

## 1. The pieces and who owns them

| Layer | File | Role |
|---|---|---|
| Vedant's solver | `src/physics/vendor/pierce_gun_design_fixed.py` | **The truth.** Vaughan 1981 + Yang 2006 + Liao 2003 + Greenwood 2016 kf, 30-digit mpmath, 3–5 s per design. Imported, never edited. |
| Adapter | `src/physics/vaughan_adapter.py` | Units (mm ↔ inches), `synthesize`, parallel `synthesize_many`, `write_deck` (his code, his `ask()` replaced at runtime by his own defaults), `deck_constructible`, `verify`. |
| LB constants | `src/physics/lb.py` | Cone constant 14.665 and spread constant 0.17409 derived from CODATA; Eq. 5/6/10 polynomials. |
| kf table | `src/physics/kf.py`, `models/kf_table.json` | His `relativistic_factor` on a 49 × 41 (ln V, ln Rc/Ra) grid, 4-point Lagrange. Max error 1.3e-7. |
| Emulator | `src/physics/closure.py` (`LBClosure`) | His algorithm in vectorised float64: same 400-point window, same 300-point scan, same root choice, same kf iteration. 100 designs in ~0.3 s. |
| Synthetic v2 | `src/data/synth_v2.py` | 1015 rows labelled by his exact solver (D11). |
| ANN-E + Stage 2 | `src/ann/stage2.py` | Surrogate of the solver; solver→measurement calibration; the one-time 7-gun evaluation. |
| Optimiser | `src/moo/problem.py`, `run.py` | NSGA-II (pymoo) over his four inputs (V, r_w, J_c at fixed I). |
| Gate G3 | `src/moo/verify.py` | Pareto designs re-run through his exact solver. |
| Deck | `src/moo/deck.py` | His EGUN deck for any design, re-synthesised exactly first. |
| Browser | `web/js/closure.js`, `web/js/design.js` | The emulator and his deck construction ported to JS; Design tab. |
| API | `api/app.py` | `/pareto/{study}`, `/optimize` (FastAPI). |

## 2. θ_raw and θ_T are never mixed

| | θ_raw (`theta`) | θ_T (`thetaT`) |
|---|---|---|
| Defined by | Eq. (10) = Eq. (14), the analysis | sin θ_raw = 0.905 sin θ_T (Sec. III-F) |
| Geometry from it | R_c, R_a, z_ac, z_a, throw, aperture | R_cT = 0.905 R_c: the cathode **as built** |
| 56.44° limit (Eq. 20) | **applies here** | — |
| Compared with Table 2 / measured guns | — | **yes** (Table 2's iterative column is θ_T) |
| ANN target | — | yes |

The Design tab's drawing uses θ_raw lengths and draws the cathode with R_cT,
exactly as his `compute_gun_points_mm` does.

## 3. Gates

| Gate | Definition | Where | Result |
|---|---|---|---|
| **G1** | Emulator vs his solver on ≥ 80 random designs (P 0.2–3.7, r_w 0.1–10 mm, C 5–300, V 1–300 kV): θ mean rel. error ≤ 0.05 %, lengths ≤ 0.1 %, feasibility identical | `python -m src.physics.gate_g1`, `tests/test_physics_layer.py` | `reports/gates/G1.json` |
| **G2** | No synthetic row in any test split; every synthetic row feasible for his solver | `tests/test_synthetic_v2.py` | all rows via emulator, 24 sampled via exact solver |
| **G3** | ≥ 10 Pareto designs per study (knee included) re-run exactly; non-zero exit if any is infeasible or > 2 % off | `python -m src.moo.verify` | `reports/moo/*/verify.csv` |

## 4. The optimisation problem

Variables: V, r_w, J_c at the study's fixed I (his four inputs). Objectives,
selectable per study: J_c, θ_T, V, z_ac, throw, |r_w − target|, |θ_T − target|.
Constraints: solver-feasible; C and P ranges; θ_T ≤ 66.5°; mean gap field
V/z_ac ≤ E_max; Eq. 5 inversion converged (−α < 7.9); anode aperture defined
(θ_raw ≤ 56.44°); **deck-constructible by his electrode construction** (D12);
optional throw target; optional ANN-E envelope. NSGA-II, population 100,
500 generations = 50 000 evaluations, backend `lb-closure`. Knee: minimum
distance to the ideal point after normalising each objective to [0, 1] over the
front. Studies: `src/moo/studies/shin_eik.json`, `ceeri_klystron.json`,
`template.json`.

The exact solver is never the search backend: at 3–5 s per design, 50 000
evaluations is two to three days. It is the verifier.

## 5. Defects found in the vendored solver (reported, not fixed — his code)

1. **Eq. 5 inversion stops converging at −α ≈ 7.94** (R_c/R_a ≈ 15). `y_from_a`
   was capped at 20 Newton iterations ("was 60") for speed; from −α ≈ 7.94 the
   Eq. 6 starting guess is poor and 20 iterations are not enough (200 are).
   Above that his θ is wrong. The emulator reproduces the behaviour (its job is
   to match him) and flags `alpha_ok`; synthetic data and the optimiser stay
   below 7.9. Fix for Vedant: restore 60+ iterations or add a bracketed fallback.
2. **CEERI "within 0.51 %" predates the relativistic factor.** With kf = 0.979 at
   150 kV his final code gives θ = 28.797° against the published 28.5652°
   (0.81 %), z_ac 29.79 mm against 30.1 mm (1.0 %).
3. His code comments say "~1.8 s per design"; measured here, 3–5 s.

## 6. What is NOT implemented

- **Sharma, Sinha & Joshi (2001)** anode-aperture synthesis. Above θ_raw = 56.44°
  Vaughan's Eq. 20 has no real root; we refuse those designs for decks and keep
  them off the front (aperture constraint), rather than invent an aperture.
- No field solve. Decks are inputs to EGUN/TRAK, not their outputs.
- Measured-angle calibration beyond Table 2's ranges is extrapolation, and the
  tool says so.
