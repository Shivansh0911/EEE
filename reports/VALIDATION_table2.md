# Physics engine validation against Panahi et al. Table 2

**Status: FAILED the acceptance gate. Tier B synthetic generation is therefore
blocked and has NOT been run.**

Date: 2026-09-18

---

## What was tested

The build spec requires that, before any synthetic data is generated, our own
implementation of the classical synthesis physics must reproduce the published
`θ_Iterative` column of Table 2 across all 30 cases, to within

> MAE ≤ 0.5° **or** MRE ≤ 1.5 %

The rationale is in the spec: a generator that cannot reproduce the published
benchmark produces rows that look like physics but are not, and that is worse
than having no synthetic data.

---

## The model that was implemented

Two equations in two unknowns — θ (half cone angle) and γ = ln(R_c/R_a).

**(I) Space-charge-limited spherical flow (Langmuir–Blodgett)**

```
P[µperv] = 29.33 · (1 − cos θ) / α(γ)²
```

The constant is `(16π/9)·ε₀·√(2e/m)·10⁶`. Rather than truncate the classical
power series — which converges poorly at the γ ≈ 1.5–2.5 values these guns need —
α is obtained by integrating the Langmuir–Blodgett ODE, derived here directly
from Poisson's equation in spherical coordinates with `u = ln(r/r_c)`:

```
3αα'' + α'² + 3αα' = 1,     α(0) = 0,  α'(0) = 1
```

**(II) Drift from anode aperture to beam waist**

For `r'' = K/r` with `r' = 0` at the waist, one integration gives exactly

```
r'² = 2K · ln(r/r_w),        K = 1.5156×10⁻² · P[µperv]
```

and since `r_a = √C · r_w · e^(−γ)`, the anode term is `ln(r_a/r_w) = ½lnC − γ`.
The slope leaving the anode is the conical convergence `sin θ` weakened by the
aperture lens:

```
sin θ · (1 − c · α'(γ)/α(γ)) = √(2K(½lnC − γ))
```

`c` is the aperture-lens coefficient. A thin-lens Davisson–Calbick treatment of
the space-charge-limited spherical diode gives `c = 1/3`.

---

## Result 1 — the Langmuir–Blodgett leg is exact ✅

The ODE integration reproduces the classical converging series
`α = γ + 0.3γ² + 0.075γ³ + 0.0143182γ⁴ + …` to near machine precision:

| γ | ODE | classical series | difference |
|---|---|---|---|
| 0.05 | 0.05075947 | 0.05075946 | +6.8×10⁻¹⁰ |
| 0.10 | 0.10307645 | 0.10307643 | +2.2×10⁻⁸ |
| 0.20 | 0.21262362 | 0.21262291 | +7.1×10⁻⁷ |

This half of the physics is verified and can be used as-is.

---

## Result 2 — the beam-optics closure does NOT pass ❌

Sweeping the aperture-lens coefficient `c` against the 30 published values:

| c | cases solved | MAE (deg) | MRE (%) |
|---|---|---|---|
| 0.00 | 30 | 18.68 | 63.50 |
| 0.33 | 30 | ~11.2 | ~37.5 |
| 0.55 | 30 | 3.98 | 13.57 |
| 0.60 | 30 | 2.06 | 7.31 |
| **0.62** | **30** | **1.66** | **5.75** |
| 0.65 | 28 | 1.67 | 5.71 |
| 0.70 | 24 | 3.78 | 11.24 |

Best achievable: **MAE 1.66°, MRE 5.75 %** — roughly 3× outside the gate.

Two things are worth noting:

1. The thin-lens prediction `c = 1/3` is badly wrong; the empirical optimum sits
   at **c ≈ 0.62**, i.e. very close to **2/3**, exactly twice the thin-lens value.
   A clean factor-of-two discrepancy usually means a definite missing term, not
   accumulated slop.
2. Beyond c ≈ 0.65 the solver starts losing roots entirely, so the minimum is
   genuinely a minimum and not an artefact of the search.

---

## Result 3 — the residuals are structured, and the failure is localized

This is the most useful part of the result. The errors are not noise:

```
Pearson R(error, ln C)       = +0.794
Pearson R(error, perveance)  = −0.570

mean error, C <  8   (n =  7):  −4.29°
mean error, C ≥  8   (n = 23):  −0.35°
```

The model is **essentially unbiased over 23 of the 30 cases** and fails almost
entirely on the seven lowest-convergence guns (C = 5.12 … 7.27, cases 4, 7, 10,
18, 19, 27 and one other), where it under-predicts θ by 4–7°.

Those are precisely the large-aperture, low-convergence geometries where a thin
aperture lens and a paraxial drift are the weakest approximations. So the missing
term is identifiable rather than mysterious — most likely a finite-thickness
aperture lens correction, and possibly the "true modification"
`sin θ / sin θ_T = 0.905` noted under Table 2, whose exact point of application
in Vaughan's procedure is not recoverable from the Panahi paper alone.

---

## Why we did not simply fit our way to a pass

We could add correction terms until the 30 residuals vanish. We did not, for two
reasons.

**It would stop being physics.** With 30 reference points, a few free parameters
tuned to those same 30 points is curve-fitting. It would then not be legitimate to
call the generated rows "a verified physics model evaluated at new points," which
is the entire argument that makes Tier B defensible rather than fabricated.

**More decisively — the arithmetic does not work.** The purpose of Tier B is to
train a surrogate that beats the paper's test MRE of **3.74 %**. A generator whose
own error against the benchmark is **5.75 %** cannot do that. Every synthetic row
would carry more error than the target the model is supposed to hit, and the ANN
would faithfully learn the generator's bias. Adding 1000 such rows would make the
model *worse* on real guns while making the dataset look impressive.

So the gate did its job.

---

## What is needed to unblock Tier B

The exact aperture-lens / beam-spread treatment from one of:

- Vaughan J.R.M., "Synthesis of the Pierce gun", *IEEE TED* 28(1) (1981) 37–41
- Tiwary U., Basu B., *IEEE TED* 34(5) (1987) 1218–1222
- Yang C., Jia B., Zhu Z., *IEEE TED* 53(11) (2006) 2849–2852

All three are paywalled at IEEE. An accessible secondary copy of Tiwary & Basu was
located but the automated text extraction returned visibly garbled equations
(inconsistent symbols, a series in a variable stated to range to 20 but expanded
only to fifth order) and was not trusted. **A clean PDF of any one of these three
papers is the single blocking item.** With the aperture term pinned down, the rest
of the engine is already written and verified.

Two other routes, if the papers cannot be obtained:

- **Beam-optics simulation** (CST, IBSimu, EGUN) to generate Tier C directly and
  calibrate the aperture term against simulation rather than against the 30 points.
- **Restrict the operating envelope to C ≥ 8**, where the current model is already
  unbiased to −0.35°, and generate Tier B only there — explicitly documenting that
  the low-convergence corner is excluded. This is viable but narrows the design
  space the optimizer can explore, so it is a fallback rather than a plan.

---

## Reproducing this

```
python3 src/data/build_tier_a.py        # rebuild the 30-row literature set
python3 src/physics/probe_closure.py    # rerun the sweep above
```
