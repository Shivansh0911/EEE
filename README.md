# Multi-Objective Optimization of an Electron Optical System

ANN surrogate for Pierce electron gun synthesis, plus an NSGA-II optimizer and a
deployed web tool. Extends Panahi et al., *Results in Physics* **73** (2025) 108256.

**Scope: pencil beams only** — solid, axially symmetric, circular cross-section.

| | URL |
|---|---|
| Web tool (Netlify) | *not deployed yet — drag `web/` onto <https://app.netlify.com/drop>, then paste the URL here. See `reports/DEPLOYMENT.md`.* |
| API (Render) | *not built — prediction is client-side; the API is only needed for NSGA-II (P6)* |

**To see it now without deploying anything: open `web/index.html` in a browser.**
It is a static page with no build step and no network calls.

---

## What this is

A Pierce gun is synthesised from three numbers a designer actually cares about:
the perveance `P`, the beam waist radius `r_w`, and the convergence ratio `C`.
Turning those into a geometry requires the half-beam cone angle θ, and the
classical route to θ is an iterative solution of the spherical space-charge flow
equations. Panahi et al. showed a small neural network can replace that iteration.

This project does three things with that result:

1. **Reproduces it** — same network, same data, benchmarked against the paper's
   reported 3.74 % test MRE.
2. **Extends it** to a second output (see D3 in `reports/DECISIONS.md` — the
   definition is an open question for the supervisor, and the code is written so
   the answer changes one config line, not the architecture).
3. **Optimises over it** — the trained surrogate is cheap enough to evaluate
   ~50 000 times inside NSGA-II, which is the entire reason for having a
   surrogate rather than just running the iteration.

---

## Start here

| File | What it is |
|---|---|
| **`00_MASTER_BUILD_PROMPTS.md`** | **The whole build.** Prompts P0–P10, plus §C the viva briefing, §D deployment, §E standing rules. |
| `data/SOURCES.md` | Provenance ledger — where every row came from, including papers we could not get |
| `reports/DECISIONS.md` | Every judgement call with reasoning (D1–D7). Becomes the methodology section. |
| `reports/VALIDATION_table2.md` | The physics validation result and why Tier B is blocked |
| `reports/BENCHMARK.md` | M1 against the paper's numbers, and the evidence for why they differ |
| `reports/DEPLOYMENT.md` | How to put the site live — drag and drop, about a minute |
| `web/` | The static site. Open `web/index.html`. |
| `data/processed/dataset_literature.csv` | Tier A — 30 verified rows, 23 columns |
| `data/processed/dataset_master.csv` | The training dataset: Tier A plus derived targets |

Open Claude Code in this folder and work through P0–P10 in order. The gates exist
to stop bad work propagating.

---

## The three-tier dataset

The dataset is stratified by *where a row came from*, and every row carries its
tier in a column. The point is that the tiers are never silently mixed: a claim
about model accuracy on real guns has to be testable on real guns alone.

| Tier | Origin | Count | Used for |
|---|---|---|---|
| **A — literature** | Printed design tables in published papers, extracted programmatically from the PDF | **30** | Training **and** the headline test metric. The test split is real guns only. |
| **B — synthetic** | This project's physics engine, evaluated at new points, restricted to `C ≥ 8` where it was validated at MAE 0.85° / MRE 2.80 % | **1000** | Training only. Never a test row, never described as measured data. |
| **C — simulation** | Full field-solver runs (CST / EGUN class) | **0 — not attempted** | Training only, same rules as B. |

**Tier B is gated, and the gate is only partly passed.** Over all 30 published
cases the physics engine misses at MAE 1.66° / MRE 5.75 %, against a required
0.5° / 1.5 % — that gate is **still failed**, and a test asserts it in the
failing direction. But the failure is confined to the seven lowest-convergence
guns: restricted to `C ≥ 8` the same model reaches **MAE 0.85° / MRE 2.80 %**
(leave-one-out 0.92° / 2.89 %).

That number is what matters, because the original objection was arithmetic
rather than procedural: a generator carrying 5.75 % of its own error cannot
train a surrogate to beat the paper's 3.74 %. At 2.80 % it can. So Tier B is
generated **over `C ≥ 8` only**, the low-convergence corner is explicitly
excluded and has no synthetic support, and every synthetic row carries the
generator's accuracy and envelope in its own columns. See D4 and D9 in
`reports/DECISIONS.md`.

**1030 rows: 30 real guns, 1000 physics-model outputs.** Nobody should read this
dataset and come away thinking there are a thousand measurements.

**And it did not help.** M3, trained on those 1000 synthetic rows, scores
**9.77 %** on the seven real held-out guns against **7.93 %** for M1 trained on
real data alone. The generator reproduces the classical *iterative method* to
2.80 %, but that method is itself 9.71 % from measurement — so against the
quantity the network must predict, the generator carries 10.48 %, and M3
inherited the bias. D4's original arithmetic objection turns out to have been
right; the 2.80 %-versus-3.74 % comparison that lifted it compared errors
measured against different references. See `reports/BENCHMARK.md` §11 and the
outcome block on D9. **M1 remains the headline model.**

---

## Architecture — why prediction does not touch the backend

**Netlify** (static frontend) + **Render** (FastAPI backend).

Render's free tier sleeps after 15 minutes of inactivity and takes roughly 50
seconds to wake. An app whose every action calls that backend shows a spinner for
a minute the first time anyone opens it. So the prediction path is built to not
call it at all.

The trained network is 26 numbers and three matrix multiplications, and equations
(1)–(7) are arithmetic. All of it runs in the browser in well under a millisecond.

| Path | Where it runs | Latency | Works with the backend asleep? |
|---|---|---|---|
| Predict — θ, geometry, gun drawing, analytical comparison | Browser, pure JS | < 1 ms | **Yes** |
| Dataset browser | Browser, static CSV/JSON | instant | **Yes** |
| Model / figures / about | Browser, static assets | instant | **Yes** |
| Optimize — NSGA-II, ~50 000 evaluations | Render backend | 5–30 s | No — needs a wake |

The frontend never hardcodes the network shape. `src/export/weights_to_json.py`
emits one JSON file — architecture, weights, scaler, targets, training envelope,
metrics, provenance — and both the Python and JS forward passes read it.
`tests/test_js_python_parity.py` asserts the two agree to < 1e-9 on 100 random
inputs, running the JS under Node. If the browser and the paper ever disagree,
that test fails before a demo does.

---

## Repository layout

```
electron-gun-moo/
├── data/
│   ├── raw/          # the paper PDF, the original sheet, extracted Table 2
│   ├── processed/    # dataset_literature.csv (Tier A); dataset_master.csv later
│   └── SOURCES.md    # provenance ledger
├── src/
│   ├── physics/      # Langmuir-Blodgett solver + the closure probe
│   ├── data/         # dataset builders
│   ├── ann/          # network, training, benchmarking
│   ├── moo/          # NSGA-II problem definition
│   └── export/       # weights -> JSON for the browser
├── models/           # trained weights + scaler, as .json (committed)
├── tests/            # physics benchmark, Jacobian, JS/Python parity, API
├── reports/          # DECISIONS.md, VALIDATION_table2.md, figures/
├── notebooks/        # teaching notebook
├── api/              # FastAPI app  -> Render
└── web/              # static site  -> Netlify
```

---

## Current status

**Done and verified**

- Tier A dataset: 30 rows, extracted programmatically from the paper's Table 2.
  The asterisked test cases give a 23/7 split matching the paper's own prose —
  an independent check the extraction is right.
- Provenance exact: cases 1, 2, 9, 12, 25, 29 from refs [14,18,19], rest from [19].
- Langmuir–Blodgett solver: ODE `3αα'' + α'² + 3αα' = 1`, derived from Poisson's
  equation, reproduces the classical series to ~1e-9. **Reusable as-is.**
- ANN package (`src/ann/`): from-scratch NumPy, 3-3-2-2 tansig, 26 parameters,
  Levenberg–Marquardt with an **analytic Jacobian** verified against finite
  differences to < 1e-6. Nguyen–Widrow restarts, per-output residual weights,
  saturation warnings, train-only scaler. 46 tests pass.
- M1 (θ) and M1-multi (θ + `Rc_mm`) trained on the paper's own 23/7 split,
  figures 3–7 reproduced, both exported for the browser.
- C is log-scaled before normalisation (D8) — ln C is the coordinate the physics
  is written in. This halved the test error and collapsed the restart spread.

**Found along the way**

- `ANN testing - Sheet1.csv` case 11: θ = `37.4` should be `37.04`.
- The column named `Convergence angle` is not an angle — it is the convergence
  *ratio* C. This matters for the "predict the convergence angle" task (D3).

**Blocked**

- **Physics gate FAILED** — MAE 1.66° / MRE 5.75 % against a required 0.5° / 1.5 %.
  Residuals are structured (`R(err, lnC) = +0.794`; −0.35° for C ≥ 8 versus −4.29°
  for C < 8), localising the failure to the aperture-lens term. Fitted coefficient
  0.62 ≈ 2/3, exactly twice the thin-lens 1/3.
- **Tier B not generated. Zero synthetic rows.** Deliberate — see the tier table
  above and D4.
- **M2 and M3 not trained.** M3 needs Tier B; M2 exists only as M3's control, so
  training it alone yields a number with nothing to compare it to.
- **M1 does not reproduce the paper's test accuracy** — 8.33 % median test MRE
  (IQR 7.1–9.9 % over 30 restarts) against the paper's 3.74 %. The dataset and
  metric code are verified correct: the paper's own reported numbers recompute
  exactly from our Tier A columns. We can say the published figure is not
  robustly reproducible from the method as stated; we cannot say why, because
  the paper reports no seed, no variance and no selection rule. Full evidence
  in `reports/BENCHMARK.md`.
- **Unblock:** a clean PDF of Vaughan 1981, Tiwary & Basu 1987, or Yang 2006. All
  paywalled at IEEE; a university library login is the fastest route.

**Open question for the supervisor** — what "half-beam convergence angle" means
(D3). Code is written so the answer changes one config line.

---

## Reproduce from scratch

Requires Python 3.12 and, for the frontend and the JS/Python parity test, Node 20.

```bash
git clone <this repo>
cd electron-gun-moo

python -m venv .venv
.venv\Scripts\activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Rebuild the artefacts, in dependency order:

```bash
python src/data/build_tier_a.py      # 30-row Tier A set from the verbatim PDF text
python src/data/build_master.py      # adds the derived targets -> dataset_master.csv
python src/physics/probe_closure.py  # rerun the physics gate sweep (expects FAIL)
python -m src.ann.train --model m1       --export models/model_m1.json
python -m src.ann.train --model m1-multi --export models/model_m1_multi.json
python -m src.ann.experiments        # the supporting measurements in BENCHMARK.md
python -m src.ann.figures            # paper figures 3-6, plus fig 7
pytest                               # Jacobian, scaler, data contracts, export
```

`build_tier_a.py` reads a verbatim text block committed inside the script rather
than the PDF, so it reproduces byte-identically without pdfplumber installed.
That is the point of D1: the block can be diffed against the PDF by hand.

The later phases (train, optimise, export, deploy) add their own entry points and
are documented here as they land. Run the API locally with:

```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

and the frontend with `npm install && npm run dev` inside `web/`.

---

## Reference

R. Panahi, S.A.H. Feghhi, Sh. Sanaye Hajari, M. Khorsandi, "Determination of the
Half-Beam cone angle for Pierce electron gun design using an artificial neural
network", *Results in Physics* **73** (2025) 108256.
[10.1016/j.rinp.2025.108256](https://doi.org/10.1016/j.rinp.2025.108256) — CC BY-NC.
