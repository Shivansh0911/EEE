# Benchmark — M1 against Panahi et al.

Written for a reader who wants to know whether this pipeline works, and is
entitled to be told where it does not.

**Headline.** Our θ test MRE is **8.33 %** (median over 30 restarts; range
2.99–115.75 %, interquartile 7.14–9.91 %) against the paper's reported
**3.74 %**. We do not reproduce the paper's figure, and section 4 sets out what
we can and cannot conclude from that.

Reproduce with:

```bash
python src/data/build_master.py
python -m src.ann.train --model m1       --export models/model_m1.json
python -m src.ann.train --model m1-multi --export models/model_m1_multi.json
python -m src.ann.experiments
python -m src.ann.figures
```

---

## 0. What changed since the previous benchmark

**C is now log-scaled before normalisation (D8).** An earlier revision of this
document rejected that as test-set fitting. That reasoning was wrong: the paper
states only that the data were "normalized" and never specifies a per-input
transform, so linear scaling of C was *our* assumption rather than the paper's
method — there was nothing to match. ln C is the coordinate the physics is
written in (equation (II) contains `0.5*ln(C)` explicitly), and the supporting
evidence — `R(err, lnC) = +0.794` in `VALIDATION_table2.md` — comes from the
*physics* residuals, computed before any network existed and with no
involvement of the seven test guns. Under linear scaling 25 of the 30 rows fall
in the bottom 15 % of C's range, the median row at normalised position 0.04.
The full argument, including the caveat that the improvement was observed before
the change was adopted, is D8 in `DECISIONS.md`.

**Two findings from the previous revision did not survive it.** Both were
artifacts of the degenerate scaling, and both are corrected in sections 6 and 7
below rather than quietly dropped:

| Previous claim | Status now |
|---|---|
| "The second output costs θ accuracy" (18.89 % vs 13.36 %) | **Withdrawn.** Under ln C the two models are within each other's restart spread, and which one leads depends on the epoch protocol. |
| "Down-weighting Rc improves θ" (8.94 % at 1.0/0.0) | **Withdrawn.** Under ln C the default 1.0/1.0 is the best of the four settings tested. |

---

## 1. The models

| Model | Targets | Params | Trained on | Tested on | Status |
|---|---|---|---|---|---|
| **M1** — headline | θ | 23 | the paper's 23 training cases | the paper's 7 test cases | **trained** |
| **M1-multi** — the multi-output deliverable | θ, `Rc_mm` | 26 | same | same | **trained** |
| **M2** literature only | — | — | all Tier A | held-out Tier A | **not trained** |
| **M3** literature + synthetic | — | — | Tier A + Tier B | held-out Tier A only | **not trained — blocked** |

M1 is the headline because θ is the quantity the paper predicts and the only one
with an external number to compare against. M1-multi carries the second output
that was the actual deliverable; it is reported beside M1 rather than folded into
it, so the cost of multi-task learning is visible rather than assumed.

**M3 is blocked and M2 is pointless without it.** M3 needs Tier B, and Tier B has
zero rows because the physics gate failed at MAE 1.66° / MRE 5.75 % against a
required 0.5° / 1.5 % (`VALIDATION_table2.md`, D4). M2's only job is to be M3's
control. `--model m3` refuses with that explanation rather than silently doing
something else. The claim this project was built to test — M1 versus M3 on the
same seven real guns — **cannot be made** until a clean PDF of Vaughan 1981,
Tiwary & Basu 1987 or Yang 2006 unblocks the physics engine.

---

## 2. Results

All figures in physical units, metric definitions per the paper's eqs (8)–(11).

### The distribution, which is the honest headline

30 Nguyen–Widrow restarts, each with the full protocol (5-fold CV inside the 23
training rows to set its epoch budget; test rows never consulted):

| Model | Test MRE θ — median | IQR | full range | restarts beating 3.74 % |
|---|---|---|---|---|
| **M1** | **8.33 %** | 7.14 – 9.91 % | 2.99 – 115.75 % | 1 of 30 |
| **M1-multi** | **9.12 %** | 7.18 – 13.88 % | 5.68 – 23.94 % | 0 of 30 |
| Panahi et al. | 3.74 %, single value | not reported | not reported | — |

See `reports/figures/fig7_restart_histogram.png`. M1's range includes one
divergent restart at 115.75 %; the interquartile range is the useful summary.

### The selected models

The shipped model is one concrete restart, chosen by cross-validation on the
training rows alone:

| | Train MRE % | Test MRE % | Train RMSE | Test RMSE | Train R | Test R |
|---|---|---|---|---|---|---|
| **Panahi et al.** | 2.18 | **3.74** | 0.95° | 2.09° | 0.997 | 0.990 |
| **M1** (θ) | 6.18 | **7.93** | 2.01° | 3.81° | 0.9906 | 0.9683 |
| **M1-multi** (θ) | 6.67 | **8.77** | 1.85° | 3.26° | 0.9919 | 0.9706 |
| **M1-multi** (`Rc_mm`) | 7.73 | 8.34 | 1.55 mm | 1.88 mm | 0.9847 | 0.9750 |

M1: seed 101, 5 epochs (its own CV folds' median stopping epoch). Out-of-fold
CV performance on the training rows, which is the estimate that used no test
data at all: **MRE 9.75 %, RMSE 3.61°, R 0.9732** — close to the test figure,
which is the encouraging part. The model generalises about as well to the seven
held-out guns as it does within its own training set.

The paper's row is not quoted from its prose. It is recomputed from the
`theta_ann_panahi_deg` column of our own Tier A extraction and comes out at
train MRE 2.1870 %, test 3.7437 %, RMSE 0.959/2.093, R 0.997/0.990 — matching
the printed values to the digit. **The dataset, the split and the metric code
are therefore verified.** Whatever separates us from 3.74 % is not there.

`Rc_mm` is the placeholder second output from D3. It is a deterministic function
of the inputs and θ, so its 8.34 % is **not** independent evidence and must not
be presented as a second physical prediction.

---

## 3. Where the test error is

M1, per test gun:

| Case | P | r_w | C | θ true | θ M1 | error |
|---|---|---|---|---|---|---|
| **3** | 2.50 | 2.49 | 16.13 | 50.28 | 41.21 | **−9.07** |
| 6 | 2.27 | 2.72 | 13.52 | 42.16 | 39.50 | −2.66 |
| 12 | 1.20 | 0.76 | 111.00 | 35.70 | 37.94 | +2.24 |
| 15 | 1.01 | 2.00 | 25.00 | 30.00 | 30.08 | +0.08 |
| 17 | 0.74 | 1.23 | 66.10 | 29.04 | 30.36 | +1.32 |
| 26 | 0.29 | 1.46 | 46.91 | 17.00 | 19.12 | +2.12 |
| 29 | 0.40 | 0.84 | 10.61 | 12.70 | 13.67 | +0.97 |

**Case 12 is fixed.** Under linear C it was the disaster of the previous
benchmark — 35.70° true against 65.91° predicted, +30.21°. It sits at C = 111,
in the stretch of C that linear scaling compressed to nothing, and under ln C
the error is +2.24°. That is the clearest single piece of evidence for D8, and
it is a case the model was not trained on.

**Case 3 is now the worst, and it is the same case the paper struggles with.**
Panahi et al. predict 45.03° against 50.28° — their largest test error too. It is
the highest-θ gun in the test set, and a tanh output over min-max scaled targets
compresses hardest near the edge of the training range. This looks like a
structural limit of the architecture rather than something specific to our run.

---

## 4. The reproducibility finding

**What we observe.** Across 30 restarts of the same code, data and protocol,
test MRE on the paper's seven test guns spans 2.99 % to 115.75 %, with an
interquartile range of 7.14–9.91 % and a median of 8.33 %. One restart in thirty
lands below the paper's 3.74 %.

**Whether we can pick the good ones in advance.** Not reliably. The
cross-validation score computed inside the 23 training rows ranks restarts only
weakly:

| Model | Pearson R | Spearman R | Pearson, excluding the top 5 % tail |
|---|---|---|---|
| M1 | +0.991 | +0.498 | +0.380 |
| M1-multi | +0.284 | +0.289 | +0.404 |

M1's Pearson +0.991 is an artifact of the single divergent restart and should
not be read as a strong relationship — the rank correlation of +0.498 is the
honest figure, and with 23 rows the CV estimate is itself noisy enough that the
ranking it produces is only somewhat better than arbitrary.

**What we conclude, and what we do not.**

We conclude that **the published 3.74 % is not robustly reproducible from the
method as stated**. A reader following the paper's description — this
architecture, this split, this optimizer, this data — should expect a result
somewhere in a wide distribution, and 3.74 % sits in the best few per cent of
that distribution rather than at its centre.

We do not conclude anything about how the paper's number was obtained. The paper
reports a single value without stating a seed, an initialisation scheme, a
number of restarts, a selection rule, or a variance — so there is no way for us
to tell from the outside whether their result was a typical run of a procedure
that differs from ours in some detail we cannot see, or one draw of a lottery
like the one we observe. Both are consistent with what is published. **The gap
may well be a detail of their method that the paper does not record rather than
anything about the result itself**, and several such details would be sufficient:
a different per-input transform, a different initialisation, a different stopping
rule, or a selection over runs.

The substantive point is not about this paper. It is that **on 23 training rows,
a single test-set number on seven points does not identify a model**, whoever
reports it. Reporting a median and a range costs nothing and would have made the
comparison decidable. That is the practice we adopt here.

---

## 5. Overfitting — the known limitation, stated up front

**23 parameters against 23 training rows** (26 against 23 for M1-multi) is
over-parameterised: there are about as many free parameters as data points, and
the network can interpolate the training set exactly given enough epochs.

`fig4_mse_vs_epochs.png` shows it. Training MSE falls monotonically while the
cross-validation curve bottoms early and then climbs. The selected M1 runs **5
epochs** — that is not a typo, it is the median stopping epoch of its own CV
folds, and it is how little training this much capacity can absorb before it
starts memorising. Train MRE 6.18 % against test 7.93 % is the shape you want:
the model is not fitting the training set much better than the test set, because
it is being stopped before it can.

Mitigations in place: early stopping via a CV-derived epoch budget; ten restarts
rather than one seed; test metrics on real guns only; the `beam_type == "pencil"`
filter. **None of these create information.** The real fix is more real guns or a
validated Tier B generator, and both are blocked on the same thing.

---

## 6. Do the residual weights matter?

Median test MRE on θ over 30 restarts at a fixed 200-epoch budget:

| Residual weights (θ / Rc) | median | min | max |
|---|---|---|---|
| **1.0 / 1.0 — default** | **7.78 %** | 5.44 | 26.42 |
| 4.0 / 1.0 — θ favoured | 9.06 % | 6.55 | 61.23 |
| 1.0 / 4.0 — Rc favoured | 19.08 % | 5.62 | 184.38 |
| 1.0 / 0.0 — Rc switched off | 8.42 % | 3.40 | 38.36 |

The previous revision found 1.0/0.0 best by a wide margin and concluded that the
second output was competing for capacity. **Under ln C that reverses**: the equal
default is now the best of the four, and switching the second output off is
slightly worse. Given the spread in each row these medians are not far apart, and
the fair reading is that with a well-conditioned input the weighting stops
mattering much — which is what the original argument for 1.0/1.0 predicted, and
what the previous result contradicted only because the inputs were badly scaled.

Heavily over-weighting either output is clearly bad, and that much is robust.

## 7. What does the second output cost?

| | Params | Train MRE θ | Test MRE θ (median of 30) |
|---|---|---|---|
| n_out = 1, fixed 200-epoch budget | 23 | 1.96 % | 9.87 % |
| n_out = 2, fixed 200-epoch budget | 26 | 3.34 % | **7.78 %** |
| M1, full CV protocol | 23 | 6.18 % | **8.33 %** |
| M1-multi, full CV protocol | 26 | 6.67 % | 9.12 % |

**The honest answer is that we can no longer measure a cost.** The two protocols
disagree about which model leads, and both differences are small against
interquartile ranges of 7.1–9.9 % and 7.2–13.9 %. The previous revision's finding
— "multi-task regression degrades accuracy at 23 training rows" — was measured
under linear C and does not survive D8. It should not be repeated.

What can still be said, and does not depend on this measurement, is that
`Rc_mm` carries **no independent information**: it is `sqrt(C)·r_w/sin θ`, a
deterministic function of the inputs and the first output (D3). The multi-output
machinery is real and tested; this particular second target cannot teach the
network anything new, and a good score on it is not evidence.

## 8. Linear C against ln C

Median test MRE on θ, 30 restarts, fixed 200-epoch budget:

| | median | min |
|---|---|---|
| n_out = 1, linear C (pre-D8) | 13.36 % | 3.10 |
| n_out = 1, ln C (adopted) | **9.87 %** | 3.59 |
| n_out = 2, linear C (pre-D8) | 18.89 % | 7.12 |
| n_out = 2, ln C (adopted) | **7.78 %** | 5.44 |

The medians improve, but the more informative change is in the spread: under the
full protocol the M1 interquartile range is 7.14–9.91 %, where the pre-D8
equivalent ranged 8.10–41.60 % over ten restarts. A model that has stopped
depending so heavily on its initialisation is better posed, not merely better
scoring.

---

## 9. What is verified

Not affected by anything above:

- **Analytic Jacobian** agrees with central finite differences to < 1e-6 across
  20 random parameter draws, n_out ∈ {1, 2, 3}, and both output activations
  (`tests/test_jacobian.py`).
- **JS/Python parity** to < 1e-9 on 100 random inputs, the JS run under Node
  (`tests/test_js_python_parity.py`), including the ln C transform.
- **Paper metrics reproduce exactly** from our extracted Table 2 columns.
- **Scaler and transform** fitted on training rows only, persisted, tested.
- **No gun appears in both splits** (`assert_no_test_leakage`).
- **Pencil-beam filter** holds, including against a planted intruder row.
- **Tier B is still empty**, asserted by test.
- **Langmuir–Blodgett solver** reproduces the classical series to ~1e-9.

---

## 10. Summary for the supervisor

1. Log-scaling C (D8) roughly halved the test error and cut the restart spread
   by much more. The previous benchmark's worst failure, test case 12 at +30°,
   is now +2.2°.
2. M1 sits at **8.33 % median test MRE** (IQR 7.1–9.9 %) against the paper's
   3.74 %. We do not match it.
3. We can say the published figure is not robustly reproducible from the stated
   method. We cannot say why, because the paper reports no seed, no variance and
   no selection rule — the difference may be a method detail it does not record.
4. Two conclusions from the previous revision were withdrawn, both artifacts of
   the old scaling. They are listed in section 0 rather than deleted.
5. M2 and M3 remain blocked on Tier B → the physics gate → one of three
   paywalled papers. D3 remains open.
