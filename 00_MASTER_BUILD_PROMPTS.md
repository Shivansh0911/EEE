# MASTER BUILD — Multi-Objective Optimization of an Electron Optical System
## Every prompt needed, in order, start to finish

**Stack: Netlify (static frontend) + Render (FastAPI backend). Not Streamlit.**

---

## How to use this document

Each section `P0` … `P10` is one self-contained prompt. Open a Claude Code session
in the project folder and paste them **one at a time, in order**. Do not paste the
whole file at once — each phase has a gate, and the gates exist to stop bad work
propagating.

Before you start, read **§A (status)** and **§B (architecture)**. §B in particular
contains the one decision that makes or breaks the live demo.

At the end: **§C** is the viva/explanation briefing, **§D** is the deployment
runbook, **§E** is the standing rules every prompt inherits.

---

# §A — What already exists (do not redo)

The repo already contains verified work. Prompts below build on it.

```
electron-gun-moo/
├── data/
│   ├── raw/
│   │   ├── ann_testing_original.csv         # your original 15-row sheet, untouched
│   │   ├── panahi2025.pdf                   # the reference paper
│   │   └── literature/panahi2025_table2.csv # 30 rows, extracted from the PDF
│   ├── processed/dataset_literature.csv     # Tier A, 30 rows, 22 columns
│   └── SOURCES.md                           # provenance ledger
├── reports/
│   ├── DECISIONS.md                         # judgement log (D1–D6 + open items)
│   └── VALIDATION_table2.md                 # physics gate result: FAILED, with evidence
└── src/
    ├── data/build_tier_a.py                 # rebuilds Tier A from verbatim PDF text
    └── physics/probe_closure.py             # the two-equation physics probe
```

**Established facts — carry these forward, do not re-derive:**

1. **Tier A = 30 rows.** Extracted programmatically from Table 2 of the PDF. The
   asterisked test cases give a 23/7 split, matching the paper's own prose — an
   independent confirmation the extraction is correct.
2. **Provenance is exact.** Paper states: cases 1, 2, 9, 12, 25, 29 from refs
   [14,18,19]; all others from ref [19]. Recorded per row.
3. **Your sheet has one error.** Case 11: θ = `37.4`, should be `37.04`.
4. **`Convergence angle` is not an angle.** It is the convergence *ratio* C
   (cathode area / waist area), dimensionless. Renamed `C`.
4b. **Scope is PENCIL BEAMS ONLY** — solid, axially symmetric, circular
   cross-section. All 30 rows carry `beam_type = "pencil"`. Sheet-beam,
   annular/MIG, multi-beam and RF-photocathode guns are out of scope and must
   never enter the dataset; the spherical Langmuir–Blodgett formulation and
   equations (1)–(7) assume circular symmetry. See P2.
5. **Langmuir–Blodgett is solved and verified.** The ODE
   `3αα'' + α'² + 3αα' = 1`, `α(0)=0`, `α'(0)=1`, derived from Poisson's equation,
   reproduces the classical series to ~10⁻⁹. **Reuse this. It is correct.**
6. **The physics gate FAILED.** Best closure: MAE 1.66° / MRE 5.75 % against
   required 0.5° / 1.5 %. Residuals are structured: `R(err, lnC) = +0.794`;
   mean error −4.29° for C < 8 versus −0.35° for C ≥ 8. Fitted aperture-lens
   coefficient ≈ 0.62 ≈ 2/3, exactly twice the thin-lens 1/3.
7. **Tier B has NOT been generated.** Zero synthetic rows exist. This is correct
   and deliberate — see the gate argument in §E.

---

# §B — Architecture (read before P7)

## B.1 The decision that saves the demo

Render's free tier **sleeps after 15 minutes of inactivity** and takes **~50 seconds**
to wake. If your whole app depends on that backend, then the moment you open it in
front of your supervisor after it has been idle, you get a spinner for a minute.

**So the prediction path must not touch the backend at all.**

The trained network is 26 numbers and three matrix multiplications. Equations
(1)–(7) are arithmetic. All of it runs in JavaScript in under a millisecond. So:

| Path | Where it runs | Latency | Works if backend is asleep? |
|---|---|---|---|
| **Predict** (θ, geometry, gun drawing, analytical comparison) | Browser, pure JS | < 1 ms | **Yes** |
| **Dataset browser** | Browser, static CSV/JSON | instant | **Yes** |
| **Model / figures / about** | Browser, static PNGs | instant | **Yes** |
| **Optimize** (NSGA-II, ~50 000 evals) | Render backend | 5–30 s | No — needs wake |

This means **the entire site is useful and instant with the backend dead**, and
Render is called only for the one genuinely heavy operation. It also means the
site keeps working forever even if you stop paying attention to the backend.

Do not architect this as "frontend calls API for everything." That is the obvious
design and it is the wrong one here.

## B.2 Repo layout

```
electron-gun-moo/
├── data/              # as §A, plus processed/dataset_master.csv later
├── src/               # python: physics, data, ann, moo, export
├── models/            # trained weights + scaler, as .json
├── tests/
├── reports/
├── notebooks/
├── api/               # FastAPI app  -> deployed to Render
│   ├── main.py
│   ├── requirements.txt
│   └── render.yaml
└── web/               # static site   -> deployed to Netlify
    ├── index.html
    ├── src/           # JS: ann.js, physics.js, ui/
    ├── public/models/ # weights JSON copied here at build
    ├── package.json
    └── netlify.toml
```

## B.3 Contract between the two

The frontend must never guess the network shape. `src/export/weights_to_json.py`
emits a single file consumed by both Python and JS:

```json
{
  "schema_version": 1,
  "architecture": [3, 3, 2, 2],
  "activation": "tanh",
  "output_activation": "tanh",
  "weights": { "W1": [[...]], "b1": [...], "W2": [[...]], "b2": [...],
               "W3": [[...]], "b3": [...] },
  "scaler": { "x_min": [...], "x_max": [...], "y_min": [...], "y_max": [...] },
  "targets": ["theta_cone_deg", "theta_conv_deg"],
  "training_envelope": { "P": [0.29, 3.67], "rw": [0.51, 4.42], "C": [5.12, 306.3] },
  "metrics": { "train_mre": 0.0, "test_mre": 0.0, "test_rmse": 0.0 },
  "provenance": { "model_id": "M3", "trained_at": "...", "n_train": 0, "n_test": 0,
                  "tier_a_rows": 0, "tier_b_rows": 0 }
}
```

`tests/test_js_python_parity.py` must assert the JS forward pass and the Python
forward pass agree to < 1e-9 on 100 random inputs (run the JS under Node). If the
browser and the paper disagree, you need to know immediately, not during the demo.

---

# P0 — Bootstrap

> **Prompt to paste:**
>
> Set up the repository skeleton for this project. The folders `data/`, `reports/`
> and `src/data`, `src/physics` already exist with verified content — **do not
> modify or regenerate anything already in them**. Read `data/SOURCES.md` and
> `reports/DECISIONS.md` first to understand what has been established.
>
> Create the remaining structure from §B.2 of the master build document:
> `src/ann/`, `src/moo/`, `src/export/`, `models/`, `tests/`, `notebooks/`,
> `api/`, `web/`.
>
> Create `requirements.txt` (numpy, scipy, pandas, matplotlib, pymoo, pytest,
> pdfplumber) and `api/requirements.txt` (fastapi, uvicorn, numpy, scipy, pymoo,
> pydantic). Pin versions.
>
> Write `README.md` describing the project, the three-tier dataset concept, the
> Netlify+Render split and why prediction is client-side (§B.1), and how to
> reproduce everything from scratch.
>
> Initialise git, add a sensible `.gitignore` (include `__pycache__`, `.venv`,
> `node_modules`, `web/dist`), and make the first commit.

---

# P1 — Physics engine and the acceptance gate ⚠️ GATE

This is the most important prompt in the document. Everything downstream depends
on it, and it is currently the blocking item.

> **Prompt to paste:**
>
> Build the physics engine in `src/physics/`. Read `reports/VALIDATION_table2.md`
> first — substantial work has already been done and one half of it is verified.
>
> **Reuse as-is (already verified, do not re-derive):** the Langmuir–Blodgett
> solver in `src/physics/probe_closure.py`. It integrates
> `3αα'' + α'² + 3αα' = 1` with `α(0)=0, α'(0)=1`, derived from Poisson's equation
> in spherical coordinates, and reproduces the classical converging series
> `α = γ + 0.3γ² + 0.075γ³ + 0.0143182γ⁴ + …` to ~1e-9. Promote it into a clean
> module `src/physics/langmuir_blodgett.py` with that series check as a unit test.
>
> **Implement `src/physics/pierce_geometry.py`** — equations (1)–(7) from the
> paper as pure functions, forward and inverse where well-posed. Perveance in
> µperv internally. Every docstring cites its equation number.
>
> ```
> (1) r_c = sqrt(I/(π·J_c))      (2) r_c = sqrt(C)·r_w      (3) P = I/V^(3/2)
> (4) R_c = r_c/sin θ            (5) R_a = R_c·e^(−γ)       (6) r_b(z_a) = r_c·e^(−γ)
> (7) z_ac = R_c − R_a           γ = ln(R_c/R_a)
> ```
>
> **The blocking problem.** The beam-optics closure is not accurate enough. The
> current model is:
> ```
> (I)  P[µperv] = 29.33·(1 − cos θ)/α(γ)²
> (II) sin θ·(1 − c·α'(γ)/α(γ)) = sqrt(2K·(½lnC − γ)),   K = 1.5156e-2·P
> ```
> Equation (I) and the drift integral in (II) are exact. The aperture-lens term
> `(1 − c·α'/α)` is the approximation that fails. Thin-lens Davisson–Calbick gives
> `c = 1/3`; the empirical optimum is `c ≈ 0.62 ≈ 2/3`. Residuals correlate with
> `ln C` at R = +0.794 — the model is unbiased (−0.35°) for C ≥ 8 and breaks
> (−4.29°) for C < 8, i.e. exactly the large-aperture geometries where a thin lens
> is worst.
>
> **Your task is to fix the aperture term properly.** In priority order:
>
> 1. **Get the primary source.** Vaughan, *IEEE TED* 28(1) (1981) 37–41; or
>    Tiwary & Basu, *IEEE TED* 34(5) (1987) 1218–1222; or Yang, Jia & Zhu,
>    *IEEE TED* 53(11) (2006) 2849–2852. Try WebFetch on DOI resolvers, Sci-Hub
>    mirrors if the user has indicated that is acceptable, ResearchGate, author
>    institutional pages, and any university proxy the user provides. If the user
>    can supply a PDF, ask for it — **this is the single fastest unblock**.
> 2. **If you obtain a paper, implement its formulation exactly.** Do not blend it
>    with the current approximation.
> 3. **If you cannot**, implement a *physically motivated* thick-lens or
>    finite-aperture correction — not an arbitrary polynomial — and state clearly
>    in the report which terms are derived and which are empirical.
>
> **Also implement** `tiwary_noniterative.py` and `yang_modified.py` if and only if
> you obtain the formulations. Do not guess coefficients. If a formula stays
> uncertain after searching, stop and say so — shipping a plausible-looking wrong
> equation is the one outcome worse than shipping nothing.
>
> **The gate.** Write `tests/test_physics_reproduces_table2.py` asserting that each
> implemented method reproduces its column in `data/processed/dataset_literature.csv`
> (`theta_iterative_deg`, `theta_noniterative_deg`, `theta_modified_noniter_deg`)
> to **MAE ≤ 0.5° or MRE ≤ 1.5 %** over all 30 cases. Update
> `reports/VALIDATION_table2.md` with a per-case table and aggregates.
>
> **If the gate fails, STOP. Do not proceed to P3. Report the residual structure.**
> The reason is arithmetic and is not negotiable: Tier B exists to train a
> surrogate that beats the paper's 3.74 % test MRE, and a generator carrying more
> error than that cannot produce data which gets a model there.
>
> If the gate cannot be passed after genuine effort, report back with the fallback
> options: (a) restrict the envelope to C ≥ 8 where the model is already unbiased,
> (b) calibrate against CST/IBSimu simulation, (c) proceed with Tier A only.

---

# P2 — Tier A expansion (literature)

Independent of P1. Can be run in parallel or while waiting on a paper.

> **Prompt to paste:**
>
> Expand the Tier A literature dataset beyond the 30 rows currently in
> `data/processed/dataset_literature.csv`. Read `data/SOURCES.md` first — it
> records what has been extracted and what has not been retrieved.
>
> **Set expectations honestly before you start.** There are not 1000 real
> experimental Pierce-gun cases in the literature. The reference paper used 30 and
> took them from essentially every well-documented practical gun in the field. A
> thorough sweep realistically yields **60–150 unique cases**, with heavy
> duplication between papers. Report the honest count. Never pad Tier A.
>
> Work through, in priority order: Frost/Purl/Johnson *Proc. IRE* 50(8) 1962;
> Tiwary & Basu 1987; Yang/Jia/Zhu 2006; Vaughan 1981; Sharma/Sinha/Joshi *IEEE
> TED* 48(2) 2001; Gharaati & Mardani *IEEE TED* 68(1) 2020; Hoseinzade et al.
> *Chin. Phys. C* 40(5) 2016; Iqbal et al. *NIM-A* 1014 2021; Ou et al. *IEEE TMTT*
> 70(1) 2021; Zhang & Cross *IEEE TED* 68(6) 2021; Alabdullah *Optik* 268 2022.
> Then search systematically — arXiv, open-access ScienceDirect, NIM-A, IPAC/LINAC
> proceedings — for Pierce/convergent gun design tables carrying perveance,
> convergence ratio, waist radius and cone angle. Useful queries:
> `convergent Pierce gun solid beam design table perveance convergence ratio`,
> `axially symmetric thermionic gun cathode spherical radius beam waist`.
>
> ### ⚠️ PENCIL BEAMS ONLY — this filter is mandatory
>
> This project is scoped to **pencil beams**: solid, axially symmetric, circular
> cross-section. That is what the 30 existing rows are, and what the physics
> assumes everywhere — spherical-cap cathode with a single `R_c`, `C` as a ratio of
> two circular areas, the **spherical** Langmuir–Blodgett solution, and equations
> (1)–(7), every one of which is written for circular symmetry.
>
> A sheet beam (ribbon/elliptical) or an annular beam (hollow, from a magnetron
> injection gun) has a perveance and something you can call a convergence ratio, so
> **its numbers look perfectly valid in the CSV** — and are silently wrong, because
> the geometry differs. Such a row would teach the network a relationship that does
> not hold for pencil beams, and the damage stays invisible until a prediction is
> unexplainably off.
>
> **INCLUDE** only guns producing a solid round axially symmetric beam. Signals:
> "pencil beam", "solid beam", "circular cross-section", "axially symmetric",
> Pierce/convergent gun with a spherical cathode cap and one convergence ratio.
>
> **EXCLUDE**, logging the reason:
>
> | Beam type | Signals | Why |
> |---|---|---|
> | **Sheet / ribbon** | "sheet beam", "ribbon", "elliptical", "rectangular cross-section", two transverse dimensions | Cathode is not a spherical cap; C is not one area ratio; spherical L–B does not apply |
> | **Annular / hollow** | "magnetron injection gun", "MIG", "hollow beam", "annular", gyrotron guns | Inner and outer radii; different space-charge geometry |
> | **Multi-beam** | "multiple beam", "MBK", beamlets | Per-beamlet geometry differs; C ambiguous |
> | **RF / photocathode** | "photocathode RF gun", "laser-driven" | Not Pierce synthesis; not space-charge-limited thermionic |
>
> Borderline (slightly elliptical treated as round): include as
> `beam_type = pencil_approx` and explain in `notes`. Never silently round to
> `pencil`.
>
> **Schema addition — required:**
> ```
> beam_type   pencil | pencil_approx | sheet | annular | multibeam | rf_photocathode | unknown
> ```
> - The 30 existing rows are already backfilled as `pencil`.
> - **Training/validation/test use `beam_type == "pencil"` only** — an explicit
>   filter in the loader, not a convention someone must remember.
> - `unknown` may sit in the CSV but is **excluded from training**. Inability to
>   determine beam type is not a reason to assume pencil.
> - Add `tests/test_no_nonpencil_in_training.py` asserting no non-pencil row
>   reaches a training split. Cheap test, catches an expensive silent mistake.
>
> **Report counts split by beam type**, e.g. "Tier A: 47 rows from 6 papers — 41
> pencil, 4 sheet (excluded), 2 unknown (excluded); 41 usable." Never a single
> headline number that quietly includes excluded rows.
>
> Expect this to **reduce** yield toward the lower end of 60–150, since much modern
> gun work is sheet-beam or multi-beam for THz and high-power devices. That is the
> correct trade: a smaller physically homogeneous dataset beats a larger one whose
> rows obey different geometry.
>
> **Extraction rules:**
> - Record a row **only if you can see the actual printed numbers.** Paywalled and
>   not visible → log under "unretrieved" in `SOURCES.md` and move on.
> - Every row carries paper key, full citation, DOI, table/figure number, and the
>   case number as printed in that source.
> - Deduplicate by physics, not by string. Same gun in two papers → one row,
>   both citations, `also_cited_in` populated.
> - Unit-normalize on entry (µperv, mm, degrees); keep `orig_units`.
> - Derive a column only where the derivation is exact (e.g. `r_c` from eq. 2) and
>   list it in `derived_fields`. **Never derive θ and call it experimental.**
> - If a value is digitised from a figure rather than read from a table, say so in
>   `notes` with an error estimate.
>
> Keep the existing 22-column schema. Write one CSV per source paper under
> `data/raw/literature/`, then `src/data/build_master.py` to merge into
> `data/processed/dataset_literature.csv`. Update `SOURCES.md` with a section per
> paper including the ones you could not get. Report: "Tier A now N rows from M
> papers; K papers unretrievable."

---

# P3 — Tier B synthetic generation ⚠️ GATED ON P1

> **Prompt to paste:**
>
> **First, confirm the P1 gate passed.** Run `pytest tests/test_physics_reproduces_table2.py`.
> If it fails, stop and report — do not generate synthetic data from an
> unvalidated generator.
>
> Build `src/data/synth_generator.py`.
>
> - **Envelope**, anchored to the paper's stated validity range:
>   `P ∈ [0.30, 3.70] µperv`, `r_w ∈ [0.50, 4.50] mm`, `C ∈ [5, 320]` sampled
>   **log-uniformly** (the real data spans two decades). Reject any sample whose
>   θ falls outside `[9°, 72°]` — the paper's trained range slightly widened.
>   If P1 passed only for C ≥ 8, use that as the lower bound and document it.
> - **Sampling:** Latin Hypercube or Sobol, not a uniform grid. Include the real
>   (P, r_w, C) triples as anchor points.
> - **Labelling:** primary target from the validated iterative method; also store
>   the other methods' values as auxiliary columns.
> - **Volume:** `--n` flag, default 1000, scalable to 10 000.
> - **Tagging:** every row `tier=B_synthetic`, `source_key=vaughan1981_synthesis`,
>   citation stating plainly that it was generated by this project's
>   implementation and validated against Panahi et al. Table 2.
>
> **Reality-gap check.** Produce `reports/figures/tier_overlap.png` — pairwise
> scatter of Tier A and Tier B in (P, r_w, C) space plus a 3D view — and report a
> coverage statistic. The synthetic cloud must *contain* the real cloud. If real
> points sit outside it, widen the envelope and say so.
>
> Merge into `data/processed/dataset_master.csv` with a `split` column. Add a
> Tier B section to `SOURCES.md` in plain language: what it is (a verified physics
> model evaluated at new points — standard surrogate modelling) and what it is not
> (experimental data, or anything extracted from a paper).

---

# P4 — ANN: refactor, extend to multi-output

> **Prompt to paste:**
>
> Port the Colab notebook's logic into `src/ann/` as a proper package, keeping the
> same conceptual steps so the notebook and package stay recognisably the same
> algorithm. **Keep it from-scratch NumPy. Do not introduce Keras or PyTorch** —
> the hand-implemented network and optimizer are the pedagogical point of this
> project.
>
> **Architecture:** `3 → 3 → 2 → n_out`, `tansig` throughout, matching the paper's
> Table 1. `n_out` is a **config value, default 2**; targets are a **config list**.
> Parameter count at n_out=2 is `(3·3+3) + (3·2+2) + (2·2+2) = 26`.
>
> **⚠️ Open question — do not resolve this yourself.** See D3 in
> `reports/DECISIONS.md`. The task says "predict the half-beam convergence angle",
> but the column so named holds the convergence *ratio*, and the paper uses
> "convergence half angle" as a synonym for θ elsewhere. Three readings are live:
> (A) synonym for θ — nothing to add; (B) the beam-envelope convergence half-angle
> at the anode, which equation (II) of the physics model computes directly;
> (C) a second geometric output such as z_ac or R_a. Implement so all three are
> satisfiable by changing config, default to **B**, document the formula used, and
> **flag it for the supervisor to confirm**.
>
> **Fixes to the existing implementation** — each is a deliberate improvement and
> a talking point:
>
> | # | Current | Change to | Why |
> |---|---|---|---|
> | 1 | Finite-difference Jacobian, 23 extra forward passes/epoch | **Analytic Jacobian** by backprop of each residual | Exact, far cheaper, no ε to tune |
> | 2 | — | Keep FD as `tests/test_jacobian.py`, assert max abs diff < 1e-6 | Proves the derivation |
> | 3 | Jacobian recomputed on rejected steps | Recompute **only after an accepted step**; retry rejections with larger µ in an inner loop | Textbook LM, large speedup |
> | 4 | `np.random.seed(100)`, one run | **Nguyen–Widrow init** + `--restarts` (default 10), keep best by validation loss | Kills the seed lottery |
> | 5 | 12 train / 3 test, no validation | **70/15/15** stratified on θ, early stopping on val MSE (patience ~200) | 26 parameters on 12 points was hopeless |
> | 6 | `tansig` output caps predictions at training range | Keep `tansig` for fidelity but **warn** when normalised output saturates beyond ±0.98; expose `--output-activation {tansig,linear}` | The paper's own test case 3 (50.28° true → 45.03° predicted) is this failure |
> | 7 | Scaler fit on train only — already correct | Keep; persist to `models/scaler.json` | Inference must reuse train min/max |
> | 8 | No convergence criteria | Stop on µ > 1e10, ‖grad‖ < 1e-10, ΔMSE < 1e-12, or `max_epochs` (default 5000) | Matches paper Table 1 |
>
> **Multi-output residuals:** for N samples and n_out outputs the residual vector
> is length `N·n_out` (row-major) and `J` is `(N·n_out) × 26`. Normal equations
> unchanged: `(JᵀJ + µI)δ = −Jᵀe`. If the two targets have different natural
> magnitudes, apply per-output residual weights, exposed as config — otherwise the
> larger-range output dominates the loss.
>
> Modules: `scaling.py`, `network.py`, `jacobian.py`, `lm.py`, `metrics.py`
> (MRE/MSE/RMSE/Pearson R per the paper's eqs 8–11), `train.py` (CLI).

---

# P5 — Train, benchmark, reproduce the paper's figures

> **Prompt to paste:**
>
> Train three models and tabulate them side by side in `reports/BENCHMARK.md`.
>
> | Model | Trained on | Tested on |
> |---|---|---|
> | **M1** replication | the paper's 23 training cases | the paper's 7 test cases (3, 6, 12, 15, 17, 26, 29) |
> | **M2** literature only | all Tier A | held-out Tier A |
> | **M3** literature + synthetic | Tier A + Tier B | **held-out Tier A only** |
>
> **M3 must be tested on real guns only.** Testing a surrogate on its own
> generator's output proves nothing. The headline claim is M1 vs M3 on the *same
> seven real test cases*: does physics-informed augmentation improve accuracy on
> real hardware? Target: beat the paper's test MRE of **3.74 %** (train 2.18 %,
> RMSE 0.95/2.09, Pearson R 0.997/0.990).
>
> M1 is also your correctness check on the whole pipeline — if M1 cannot roughly
> reproduce the paper's numbers on the paper's own split, something is wrong in
> the training code, not in the data.
>
> Regenerate the paper's figures for direct comparison, into `reports/figures/`:
> `fig3_regression_train.png` / `_test.png` (predicted vs experimental with y=x),
> `fig4_mse_vs_epochs.png` (train + validation, log y), `fig5_method_comparison.png`
> (ANN vs iterative vs non-iterative vs modified vs actual across the 30 real
> cases), `fig6_percentage_error.png`.
>
> Export the chosen model with `src/export/weights_to_json.py` to
> `models/model_m3.json` in exactly the schema in §B.3 of the master build
> document — including `training_envelope`, `metrics` and `provenance`.
>
> **If any test MRE comes out below 1 % or Pearson R above 0.999 on real held-out
> data, do not celebrate — suspect leakage.** Check specifically for duplicate
> guns appearing in both Tier A and the Tier B anchor points, and for the scaler
> having seen test data.

---

# P6 — NSGA-II multi-objective optimizer

This is the actual project title. Everything before it was infrastructure.

> **Prompt to paste:**
>
> Build `src/moo/` using `pymoo` (NSGA-II).
>
> - **Design variables:** P, r_w, C. No new parameters.
> - **Evaluator:** the trained ANN gives θ in microseconds; equations (1)–(7) then
>   give the full geometry. This is the whole reason the surrogate exists —
>   NSGA-II needs ~50 000 evaluations and the iterative method is too slow inside
>   that loop.
> - **Objectives** (selectable in config and in the UI):
>   minimise `|r_w − r_w_target|` (beam quality); minimise `J_c` (cathode current
>   density → cathode lifetime); minimise `z_ac` (compactness); minimise
>   `|θ − θ_target|` or keep θ inside a manufacturable band.
> - **Constraints:** θ inside the ANN's trained envelope (hard — refuse to
>   extrapolate), C > 1, positive geometry, `r_b(z_a) <` aperture radius.
> - **Output:** Pareto front as CSV plus the full synthesized geometry
>   (r_c, R_c, R_a, z_ac, γ) for each point.
>
> **Sanity gate — do not skip.** Re-evaluate the top 10 Pareto solutions with the
> *exact* iterative method rather than the surrogate, and report surrogate error on
> them. If it exceeds ~2 %, say so prominently. A surrogate that is accurate on
> average but wrong at the optimum is the classic failure mode of
> surrogate-assisted optimization, and the optimizer will actively seek out the
> regions where the surrogate is over-optimistic.

---

# P7 — Backend API (FastAPI → Render)

> **Prompt to paste:**
>
> Build `api/` as a FastAPI service for deployment on Render's free tier.
>
> **Read §B.1 of the master build document first.** Render free tier sleeps after
> 15 minutes and takes ~50 s to wake. Prediction therefore does **not** live here —
> it runs client-side in JS. This backend exists for NSGA-II only.
>
> **Endpoints:**
> - `GET  /health` → `{status, model_id, uptime}`. Cheap, used to wake the service
>   and to drive the frontend's "waking up" state.
> - `GET  /model` → the `models/model_m3.json` contract, so the frontend can fetch
>   weights rather than hard-coding them.
> - `POST /optimize` → body: objectives, targets, bounds, population, generations.
>   Returns the Pareto front plus synthesized geometry per point.
> - `POST /verify` → body: a list of (P, r_w, C). Runs the **exact** iterative
>   physics, not the surrogate. This is what backs the Pareto sanity gate in the UI.
> - `POST /predict` → same as the client-side path. Exists only as a reference
>   implementation for the parity test; the frontend does not call it in normal use.
>
> **Engineering:**
> - Load model and physics tables once at startup, not per request.
> - Cap `population × generations` so a request cannot run unboundedly; return
>   `422` with a clear message if exceeded.
> - Long optimizations: either stream progress (SSE) or return a job id with
>   `GET /optimize/{id}`. Prefer streaming — it makes the UI feel alive.
> - CORS restricted to the Netlify domain plus `localhost` for development.
> - Structured JSON errors: `{error, detail, hint}`. Never leak a stack trace.
> - `api/render.yaml` for one-click deploy; document the `PYTHON_VERSION` env var.
> - `tests/test_api.py` with FastAPI `TestClient` covering every endpoint plus
>   malformed input.

---

# P8 — Frontend (static site → Netlify)

> **Prompt to paste:**
>
> Build `web/` as a static site for Netlify. Vite + vanilla JS or React — your
> choice, but **no server-side rendering and no backend dependency for anything
> except the Optimize tab**.
>
> **Read §B.1 and §B.3 of the master build document first.**
>
> **Port to JavaScript, in `web/src/`:**
> - `ann.js` — forward pass for the `3→3→2→n_out` tanh network, reading
>   `models/model_m3.json`. About 30 lines. Plus normalize/denormalize using the
>   persisted scaler.
> - `physics.js` — equations (1)–(7) for the geometry synthesis.
>
> **`tests/test_js_python_parity.py` must assert the JS and Python forward passes
> agree to < 1e-9 on 100 random inputs**, running the JS under Node. Non-negotiable:
> if the browser and the paper disagree you need to know now, not during the demo.
>
> **Tabs:**
>
> 1. **Predict** — three inputs (P, r_w, C) → θ and the second output, the full
>    synthesized geometry, and a **scale cross-section drawing of the gun** (SVG,
>    mirroring the paper's Fig. 1) that updates live. Show the analytical methods'
>    predictions alongside the ANN's. All client-side, instant.
>    - **Extrapolation guard:** if an input falls outside `training_envelope`, show
>      an amber banner naming the variable and by how much. Never silently return a
>      confident number.
>    - **Saturation warning:** if the normalised output exceeds ±0.98, say so — the
>      tanh output layer cannot extrapolate beyond the training target range.
> 2. **Optimize** — the only tab that calls Render. Target specs in → Pareto front
>    plotted interactively; click a point → geometry table, gun drawing, download
>    CSV. Show the exact-physics verification alongside the surrogate values.
>    - **Handle the cold start explicitly.** On mount, ping `/health`. If it does
>      not answer immediately, show "Waking the optimizer — up to a minute on the
>      free tier" with a progress indicator. Do not show a bare spinner and do not
>      show an error. This is the single most likely thing to go wrong live.
> 3. **Dataset** — browse `dataset_master.csv` with filters on tier and source.
>    Every row shows its citation and a clickable DOI. **This tab is the answer to
>    "where did your data come from"** — make it good. Show tier counts prominently
>    and state plainly which rows are real and which are physics-generated.
> 4. **Model** — training curves, regression plots, the M1/M2/M3 benchmark table,
>    parameter count, an architecture diagram.
> 5. **About** — plain-language method explanation, citation of Panahi et al.,
>    licence note (paper is CC BY-NC), and a clear statement of what is synthetic.
>
> **Engineering:** responsive, works on a phone (your supervisor may open it on
> one). Dark mode. `netlify.toml` with build command and publish dir. Weights JSON
> copied into `web/public/models/` at build so the site is self-contained. Backend
> URL from an environment variable, never hard-coded.

---

# P9 — Deploy

> **Prompt to paste:**
>
> Deploy both halves and write `reports/DEPLOYMENT.md` documenting every step so
> it can be repeated.
>
> **Render (backend):** create a Web Service from the repo, root directory `api/`,
> build `pip install -r requirements.txt`, start
> `uvicorn main:app --host 0.0.0.0 --port $PORT`. Free instance type. Set
> `PYTHON_VERSION`. Note the assigned URL.
>
> **Netlify (frontend):** create a site from the repo, base directory `web/`,
> build `npm run build`, publish `web/dist`. Set `VITE_API_URL` to the Render URL.
> Note the assigned URL.
>
> **Then verify, in a fresh incognito window:**
> - Predict tab works **with the backend asleep** — this is the architecture's
>   whole point. If it does not, the JS port is wrong.
> - Optimize tab shows the waking state, then completes.
> - Dataset tab loads and DOI links resolve.
> - The site works on a phone.
>
> **Cold-start mitigation:** add a GitHub Action or cron-job.org ping to `/health`
> every 10 minutes during the week of your review. Document that this is a
> demo-day measure, not a permanent fix, and that it exists because of the free
> tier's sleep policy.
>
> Put both URLs at the top of `README.md`.

---

# P10 — Teaching notebook and final report

> **Prompt to paste:**
>
> Build `notebooks/01_walkthrough.ipynb` — the presentation artefact. Same cell
> order as the original Colab so it is recognisably an evolution of the existing
> work, but each stage preceded by a markdown cell explaining *what* and *why*:
> normalization → forward pass → residuals → Jacobian → LM update → µ adaptation →
> convergence → denormalization → inference.
>
> Include as live cells:
> - The **FD-vs-analytic Jacobian check**. This is the most convincing cell in the
>   notebook — it proves the hand-derived gradient is right.
> - A **worked LM step on a 2-parameter toy problem**, so the matrix algebra is
>   visible rather than asserted.
> - The **Langmuir–Blodgett ODE vs the classical series** agreement to 1e-9.
>
> Then write `reports/FINAL_REPORT.md` pulling everything together: problem,
> method, dataset construction and its honest tier structure, the physics
> validation (including the failure if it was not resolved), ANN architecture and
> training, the M1/M2/M3 benchmark, the optimizer and its Pareto front, the
> deployed tool, limitations, and future work. Use `reports/DECISIONS.md` as the
> methodology section's backbone.
>
> Be honest about every limitation. A report that states its own weak points is
> far stronger under questioning than one that has to have them extracted.

---

# §C — Explaining this (the viva briefing)

## C.1 Why a neural network at all — the honest answer

"Because it's accurate" is a weak answer and invites the obvious follow-up. The
real answer is **speed inside an optimization loop**.

NSGA-II with population 100 over 500 generations is **50 000 evaluations of θ**.
The iterative method takes milliseconds per call — fine once, fatal 50 000 times.
The trained network is 26 numbers and three matrix multiplications: microseconds.
It turns the optimization from a batch job into something responsive enough to sit
behind a web page.

The network's role has a precise name: a **surrogate model** — a cheap,
differentiable stand-in for an expensive evaluator. Surrogate-assisted optimization
is standard engineering practice. That framing is far stronger than "we used AI."

## C.2 The ML mechanism, layer by layer

**MLP, 3 → 3 → 2 → 2.** Feed-forward, no recurrence. Inputs P, r_w, C; outputs θ
and the second target. Each neuron computes `z = Σwᵢxᵢ + b` then `h = tanh(z)`.
Weights say how much each incoming signal matters; the bias shifts the neuron's
threshold, letting it respond at inputs other than zero.

**Why `tansig`.** Two reasons. Non-linearity is the entire point — stack linear
layers and they collapse into one matrix, leaving you with an expensive linear
regression, and θ(P, r_w, C) is strongly non-linear at high convergence. And its
range matches the data, which is scaled to [−1, 1].

*State the limitation before you are asked:* a tanh output with min-max scaled
targets **structurally cannot predict outside the training range**. The paper's own
test case 3 shows it — true 50.28°, predicted 45.03°, its largest test error.

**Normalization, and the rule that matters.** P ~ 0.3–3.7, r_w ~ 0.5–4.5, C ~ 5–306.
Unscaled, C's magnitude swamps the others and the network effectively sees one
input. We map to [−1, 1] with `x_norm = 2(x − x_min)/(x_max − x_min) − 1`, and
**x_min/x_max come from the training set only**, reused unchanged for validation,
test and live inference. Fitting the scaler on everything leaks test information
into training. Your original Colab code already did this correctly — worth saying,
it is a common mistake.

**Levenberg–Marquardt.** With 26 parameters this is a non-linear least-squares
problem. Gradient descent is robust but slow and zig-zags in narrow valleys.
Gauss–Newton approximates the error surface as a quadratic bowl and jumps to its
minimum — fast near a solution, divergent far from one. LM blends them via damping
µ:

```
(JᵀJ + µI)δ = −Jᵀe
```

Large µ → `µI` dominates → small step along `−Jᵀe` → gradient descent, safe.
Small µ → reduces to Gauss–Newton → big well-aimed jumps, fast. And µ adapts
itself: step improved MSE → accept, **µ ÷ 10** (be bolder); step made it worse →
reject, **µ × 10** (be cautious), retry. No learning rate to tune. Converges in
tens of iterations where gradient descent needs thousands. This is MATLAB's
`trainlm` and why the paper used it.

**It does not scale** — `JᵀJ` is 26×26 here but would be millions-squared for a
deep network, which is exactly why modern deep learning uses Adam/SGD. Knowing
*when* LM is right, and when it is not, is the mark of understanding it.

**The Jacobian.** One row per residual, one column per parameter. `J[i][j]` answers:
if I nudge parameter j, how much does residual i change? A local sensitivity map.
Shape `(N·n_out) × 26`. The original code used finite differences — correct but 26
extra forward passes per iteration, and accuracy depends on choosing ε well (too
big is crude, too small and floating-point cancellation destroys it). We use an
**analytic Jacobian by backprop**, propagating each residual's derivative separately
instead of summing into one gradient — and we kept the FD version **as a unit test**
asserting agreement to 1e-6. That is the standard way to prove a hand-derived
gradient, and a good thing to be able to show on demand.

**Overfitting guards.** 26 parameters against 12 original training points was
hopeless. Fixes: far more data; 70/15/15 with early stopping on validation; ten
Nguyen–Widrow restarts keeping the best (a single seed means your result is partly
luck); and test error reported on **real guns only**.

## C.3 The dataset — the part needing most care

**The constraint we could not engineer around.** ~1000 real experimental Pierce-gun
cases **do not exist in the literature**. The paper used 30, drawn from essentially
every well-documented gun in the field. A thorough sweep yields 60–150.

We treated that as a finding, not a failure, and tiered the data:

| Tier | What | Label |
|---|---|---|
| **A** | Real guns from published tables | `A_literature` |
| **B** | Generated by our verified implementation of the classical synthesis | `B_synthetic` |
| **C** | Beam-optics simulation, if available | `C_simulation` |

**Why Tier B is legitimate, not fabrication.** Table 2 publishes not only the
experimental θ but what all three classical methods predict, for all 30 cases —
a published benchmark. We implement the methods, run them on the 30 published
input triples, and require our output to reproduce their columns. Once that
passes, the generator is demonstrably the same physics as the literature, and
sampling it is **evaluating a verified model at new points** — which is exactly
what surrogate modelling means. Anchored to reality by the Tier A rows in training.

**Current status: that gate has not passed.** Best MAE 1.66° / MRE 5.75 % against
a required 0.5° / 1.5 %. So **no synthetic data has been generated.** If asked why
the dataset is 30 rows rather than 1000, that is the answer, and it is a better
answer than 1000 rows of unvalidated numbers.

## C.4 Anticipated questions

**"Why not just use the iterative method directly?"** For a single design you
should. The ANN earns its place inside the optimization loop at ~50 000
evaluations. We also verify the final Pareto designs with the exact method — the
surrogate proposes, the physics confirms.

**"Isn't 26 parameters too small to be a real neural network?"** Deliberately
small. Limited real data, and a larger network would overfit. The paper tested
one, two and three hidden layers and selected this. Universal approximation needs
enough hidden units for the function's complexity, not depth — and a smooth
three-variable function is not complex.

**"Why LM instead of Adam?"** LM is second-order, using curvature via `JᵀJ`, not
just the gradient. On small networks that means convergence in tens of iterations
with nothing to tune. It does not scale because `JᵀJ` grows as the square of the
parameter count.

**"How do you know it isn't memorising?"** Held-out data never seen in training or
scaler fitting; early stopping on a separate validation split; ten restarts to show
the result is not seed-dependent; and an explicit leakage check between Tier A and
the Tier B anchor points.

**"What if an input is outside your training range?"** The model refuses to answer
confidently — the UI shows which variable is out of range and by how much, and the
optimizer constrains its search to the validated envelope. A tanh output cannot
extrapolate beyond the training target range by construction, and we would rather
say that than return a plausible wrong number.

**"Where did your data come from?"** The Dataset tab and `SOURCES.md`. Every row
carries tier, citation and DOI. 30 rows are real published guns, extracted
programmatically from the PDF — and we found a transcription error in our own
original sheet doing it. No synthetic rows exist yet, because the generator has
not passed its validation gate.

**"What's the contribution beyond the paper?"** Three things: multi-output
prediction; a physics-based data generator with an honest validation gate, plus
the finding that the naive closure is insufficient and *why* (residuals correlate
with convergence ratio at R = 0.79, localising the failure to the aperture lens);
and an NSGA-II optimizer with a deployed interface, turning a prediction result
into a usable design tool.

**"Why did the physics validation fail?"** Because we checked. The Langmuir–Blodgett
half is exact to 1e-9. The aperture-lens term is a thin-lens approximation that
breaks for low-convergence, large-aperture geometries — and the residual structure
says so precisely. The fitted coefficient lands at 0.62 ≈ 2/3, exactly twice the
thin-lens 1/3, which points at a specific missing term rather than general error.
We need one of three paywalled IEEE papers to close it.

## C.5 Glossary

| Term | Meaning here |
|---|---|
| **Perveance P** | `I/V^(3/2)`; characterises space-charge-limited flow. In µperv |
| **Convergence ratio C** | Cathode area ÷ waist area. Dimensionless. **Not an angle** |
| **Half-beam cone angle θ** | Beam edge angle at the cathode; fixes the whole geometry |
| **Langmuir–Blodgett α** | Solution of spherical space-charge flow; sets how much current a spherical diode passes |
| **MLP** | Multi-layer perceptron; feed-forward, no recurrence |
| **tansig** | tanh activation, range (−1, 1) |
| **Residual** | predicted − actual, one sample one output |
| **Jacobian J** | ∂residual_i/∂parameter_j; local sensitivity map |
| **Levenberg–Marquardt** | Second-order least-squares optimizer interpolating gradient descent and Gauss–Newton via damping µ |
| **Damping µ** | Large → cautious; small → aggressive |
| **Surrogate model** | Cheap stand-in for an expensive evaluator inside an optimization loop |
| **MRE / RMSE** | The paper's error metrics (its eqs 8–10) |
| **Pearson R** | Predicted-vs-experimental correlation; paper reports 0.997 / 0.990 |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II |
| **Pareto front** | Designs where no objective improves without another worsening |

---

# §D — Deployment quick reference

| | Netlify | Render |
|---|---|---|
| Hosts | `web/` static site | `api/` FastAPI |
| Base dir | `web/` | `api/` |
| Build | `npm run build` | `pip install -r requirements.txt` |
| Publish / start | `web/dist` | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Env var | `VITE_API_URL` → Render URL | `PYTHON_VERSION` |
| Free-tier catch | none meaningful | **sleeps after 15 min, ~50 s cold start** |

**The cold start is the one thing likely to embarrass you live.** Mitigations, in
order: prediction is client-side so the demo works regardless; the Optimize tab
shows an explicit "waking up" state rather than a bare spinner; and a `/health`
ping every 10 minutes during review week.

---

# §E — Standing rules (every prompt inherits these)

1. **Never invent a datapoint or a citation.** Cannot verify it → leave it blank
   and log it. 400 verified rows beat 1000 with 600 fabricated, and fabrication is
   discoverable in five minutes.
2. **Tier B rows are never described as experimental or literature-derived** — not
   in the CSV, the app, the report, or conversation.
3. **P1's gate blocks P3.** If the physics does not reproduce Table 2, no synthetic
   data is generated. The reason is arithmetic: a generator carrying 5.75 % error
   cannot train a surrogate to beat 3.74 %. The network would faithfully learn the
   generator's bias, and 1000 such rows would make the model *worse* on real guns
   while making the dataset look more impressive.
4. **Do not fit your way past a validation gate.** Sweeping one physically
   meaningful coefficient and reporting where it lands is a diagnostic. Adding
   free parameters until residuals vanish turns validation into curve-fitting and
   destroys the argument that makes Tier B legitimate.
5. **The D3 output-definition question is the supervisor's to answer**, not
   Claude's. Default to reading B, flag it, move on.
6. **Report honest counts at every phase.** "Tier A yielded N rows from M papers;
   K unretrievable." No rounding up.
7. Every physics function docstring cites its source paper and equation number.
8. Keep `reports/DECISIONS.md` current — it becomes the methodology section.
9. Commit after each phase with a descriptive message.
10. **If a result looks too good, suspect leakage** before celebrating — duplicate
    guns across splits, or a scaler that has seen test data.
11. **Pencil beams only.** Nothing with `beam_type != "pencil"` ever reaches a
    training split. Sheet and annular guns have a perveance and a convergence
    ratio too, so wrong rows look right — the filter is the only thing catching it.

---

# §F — Definition of done

- [ ] `pytest` green: Table-2 physics benchmark, Jacobian check, JS/Python parity, API
- [ ] `dataset_master.csv`, every row traceable to a source
- [ ] `SOURCES.md` complete, including unretrieved papers
- [ ] M1/M2/M3 trained, benchmark table filled, M3 tested on real guns only
- [ ] M3 test MRE reported against the paper's 3.74 %
- [ ] Paper figures 3–6 reproduced
- [ ] NSGA-II Pareto front, verified against exact physics at the optima
- [ ] Netlify site live; **Predict tab works with the backend asleep**
- [ ] Render API live; Optimize tab handles cold start gracefully
- [ ] `DECISIONS.md`, `FINAL_REPORT.md`, `DEPLOYMENT.md` written
- [ ] Teaching notebook runs top to bottom
- [ ] Both URLs at the top of `README.md`
