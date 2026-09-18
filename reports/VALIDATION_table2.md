# Physics engine validation against Panahi et al. Table 2

**Status: FAILED the acceptance gate over the full envelope.** The strict gate
is still not met and that has not changed.

**Superseded in part.** Tier B generation is no longer blocked outright: it is
restricted to `C ≥ 8`, where the same model reaches MAE 0.85° / MRE 2.80 %. See
the addendum at the foot of this document and D9 in `DECISIONS.md`. Everything
between here and the addendum describes the full-envelope result and stands
unchanged.

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

---

# ADDENDUM — restricted-envelope validation (C ≥ 8)

**Date: 2026-09-18. This addendum does not overturn the result above. The strict
gate is still failed.** What it establishes is that the failure is confined to a
corner of the design space, and that outside that corner the model is accurate
enough to generate training data. See D9 in `DECISIONS.md`.

## The measurement

Aperture-lens coefficient `c = 0.62` throughout, exactly as swept above.

| Case | P | r_w | C | θ_iter published | θ ours | error | in envelope |
|---|---|---|---|---|---|---|---|
| 2 | 2.20 | 0.51 | 306.30 | 57.26 | 59.58 | +2.32 | yes |
| 1 | 3.00 | 0.62 | 207.60 | 69.21 | 70.16 | +0.95 | yes |
| 12 | 1.20 | 0.76 | 111.00 | 35.99 | 36.99 | +1.00 | yes |
| 17 | 0.74 | 1.23 | 66.10 | 26.21 | 26.79 | +0.58 | yes |
| 11 | 1.32 | 1.41 | 50.30 | 34.60 | 34.99 | +0.39 | yes |
| 23 | 0.32 | 1.46 | 46.91 | 16.36 | 16.63 | +0.27 | yes |
| 26 | 0.29 | 1.46 | 46.91 | 15.56 | 15.82 | +0.26 | yes |
| 9 | 1.90 | 1.35 | 35.55 | 40.46 | 40.28 | -0.18 | yes |
| 22 | 0.53 | 1.82 | 30.19 | 20.03 | 20.07 | +0.04 | yes |
| 21 | 0.77 | 1.83 | 29.86 | 24.26 | 24.26 | +0.00 | yes |
| 14 | 1.34 | 1.95 | 26.30 | 31.92 | 31.63 | -0.29 | yes |
| 15 | 1.01 | 2.00 | 25.00 | 27.25 | 27.03 | -0.22 | yes |
| 24 | 0.30 | 2.01 | 24.75 | 14.57 | 14.52 | -0.05 | yes |
| 20 | 0.79 | 2.03 | 24.27 | 23.86 | 23.68 | -0.18 | yes |
| 5 | 2.29 | 2.35 | 18.11 | 40.27 | 38.92 | -1.35 | yes |
| 3 | 2.50 | 2.49 | 16.13 | 41.40 | 39.68 | -1.72 | yes |
| 6 | 2.27 | 2.72 | 13.52 | 37.85 | 36.01 | -1.84 | yes |
| 13 | 1.52 | 2.81 | 12.66 | 30.14 | 28.69 | -1.45 | yes |
| 16 | 1.32 | 2.93 | 11.65 | 27.49 | 26.07 | -1.42 | yes |
| 25 | 0.55 | 0.94 | 11.41 | 17.42 | 16.60 | -0.82 | yes |
| 29 | 0.40 | 0.84 | 10.61 | 14.59 | 13.84 | -0.75 | yes |
| 8 | 2.33 | 3.23 | 9.59 | 35.63 | 32.94 | -2.69 | yes |
| 30 | 0.38 | 0.77 | 9.31 | 13.84 | 12.96 | -0.88 | yes |
| 19 | 1.07 | 3.71 | 7.27 | 22.12 | 19.94 | -2.18 | **EXCLUDED** |
| 18 | 1.29 | 4.05 | 6.10 | 23.28 | 20.19 | -3.09 | **EXCLUDED** |
| 27 | 0.79 | 4.13 | 5.86 | 17.95 | 15.42 | -2.53 | **EXCLUDED** |
| 28 | 0.33 | 4.23 | 5.59 | 11.40 | 9.68 | -1.72 | **EXCLUDED** |
| 7 | 2.93 | 4.26 | 5.51 | 34.86 | 28.88 | -5.98 | **EXCLUDED** |
| 4 | 3.67 | 4.33 | 5.33 | 39.04 | 31.72 | -7.32 | **EXCLUDED** |
| 10 | 3.27 | 4.42 | 5.12 | 36.31 | 29.06 | -7.25 | **EXCLUDED** |

### Aggregates

| Subset | n | MAE (deg) | MRE (%) | mean error (deg) | max abs error (deg) |
|---|---|---|---|---|---|
| All 30 | 30 | 1.66 | 5.75 | -1.27 | 7.32 |
| C >= 8 | 23 | 0.85 | 2.80 | -0.35 | 2.69 |
| C < 8 (excluded) | 7 | 4.29 | 15.45 | -4.29 | 7.32 |
| C >= 12 | 18 | 0.73 | 1.97 | -0.08 | 2.32 |

## What this does and does not show

**Passes:** the restricted gate for the generator — `MAE ≤ 1.0°` **and**
`MRE ≤ 3.0 %` over C ≥ 8. Asserted by
`tests/test_physics_reproduces_table2.py::test_restricted_envelope_meets_the_generator_gate`.

**Still fails:** the original strict gate of `MAE ≤ 0.5°` or `MRE ≤ 1.5 %` over
all 30 cases, at 1.66° / 5.75 %. That is asserted too, in the failing direction,
by `test_strict_gate_over_all_30_cases_still_fails` — so a green test suite
cannot be mistaken for a passed gate.

**The decisive number** is 2.80 % against the paper's own test MRE of 3.74 %.
D4 blocked Tier B on an arithmetic argument: a generator carrying 5.75 % of its
own error cannot train a surrogate to beat 3.74 %, because the network learns the
generator's bias faithfully. Over C ≥ 8 the generator's error is *below* the
target, so that argument no longer applies. It still applies everywhere else,
which is why the envelope is restricted rather than removed.

**The excluded corner is large and it is named.** Seven of the thirty published
guns, all with C < 8, sit outside the envelope. There the model under-predicts θ
by a mean of −4.29° (MRE 15.45 %, worst case −7.25° on case 10). Those are the
large-aperture, low-convergence geometries where a thin aperture lens and a
paraxial drift are weakest. No synthetic row is generated there, and the coverage
gap is plotted in `reports/figures/tier_overlap.png` rather than left implied.

## The in-sample caveat, and what it is worth

`c = 0.62` was chosen by sweeping against these same 30 published cases. So
0.85° / 2.80 % is an **in-sample** figure and is optimistic by an unknown amount.
Rather than assume the amount is small, it was measured.

**Leave-one-out refit.** For each of the 23 restricted cases, `c` was refitted
from scratch on the other 22 and then used to predict the held-out case:

| | MAE (deg) | MRE (%) | mean error | max abs error |
|---|---|---|---|---|
| in-sample (c = 0.62 fixed) | 0.85 | 2.80 | −0.35 | 2.69 |
| **leave-one-out, c refitted per fold** | **0.92** | **2.89** | −0.29 | 2.92 |

The optimism is about 0.07° — one free parameter against 23 points cannot overfit
much, and now that is demonstrated rather than asserted. The out-of-sample figure
is still inside the gate and still below 3.74 %. The refitted `c` never moved
outside 0.620–0.625.

Asserted by `test_leave_one_out_refit_of_c_is_still_within_the_gate` (marked
`slow`).

## What has not changed

- The aperture-lens physics is still wrong; we have an empirical coefficient
  near 2/3 where thin-lens theory says 1/3, and no derivation for the factor of
  two. **No additional free parameters were introduced** to close the gap (D5).
- A clean PDF of Vaughan 1981, Tiwary & Basu 1987 or Yang 2006 remains the item
  that would fix the physics properly and open the full envelope.
- Tier B rows are physics, not measurements, and are never to be described as
  experimental — in the CSV, the app, the report, or conversation.

## Reproducing this addendum

```
python src/data/build_tier_a.py
python src/data/build_master.py
pytest tests/test_physics_reproduces_table2.py -q          # fast checks
pytest tests/test_physics_reproduces_table2.py -q -m slow  # leave-one-out
```
