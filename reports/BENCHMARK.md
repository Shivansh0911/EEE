# Benchmark — M1 against Panahi et al.

Written for a reader who wants to know whether this pipeline works, and is
entitled to be told when it does not.

**Headline: M1 does not reproduce the paper's test accuracy.** Our test MRE on
θ is **17.27 %** against the paper's **3.74 %**. The rest of this document is
the evidence about why, and the reason we are not tuning until the number
improves.

Reproduce with:

```bash
python src/data/build_master.py
python -m src.ann.train --model m1 --export models/model_m1.json
python -m src.ann.figures
python -m src.ann.experiments
```

---

## 1. The models

| Model | Trained on | Tested on | Status |
|---|---|---|---|
| **M1** replication | the paper's 23 training cases | the paper's 7 test cases (3, 6, 12, 15, 17, 26, 29) | **trained** |
| **M2** literature only | all Tier A | held-out Tier A | **not trained** |
| **M3** literature + synthetic | Tier A + Tier B | held-out Tier A only | **not trained — blocked** |

**M3 is blocked and M2 is pointless without it.** M3 needs Tier B, and Tier B
has zero rows because the physics gate failed at MAE 1.66° / MRE 5.75 % against
a required 0.5° / 1.5 % (`VALIDATION_table2.md`, D4). M2's only job is to be
M3's control — the "does physics-informed augmentation help?" comparison — so
training it alone would produce a number with nothing to compare it to.
`python -m src.ann.train --model m3` refuses with that explanation rather than
silently doing something else.

The headline claim this project was built to test — M1 versus M3 on the same
seven real guns — **cannot be made**, and will not be until a clean PDF of
Vaughan 1981, Tiwary & Basu 1987 or Yang 2006 unblocks the physics engine.

---

## 2. M1 against the paper

Both rows are the same 23/7 split, the same seven test guns, and the same
metric definitions (eqs 8–11), in physical units.

| | Train MRE % | Test MRE % | Train RMSE ° | Test RMSE ° | Train R | Test R |
|---|---|---|---|---|---|---|
| **Panahi et al.** | 2.18 | **3.74** | 0.95 | 2.09 | 0.997 | 0.990 |
| **M1, this work** | 4.31 | **17.27** | 1.33 | 11.93 | 0.9955 | 0.7133 |

The paper's row is not quoted from its prose — it is recomputed from the
`theta_ann_panahi_deg` column of our own Tier A extraction, and comes out at
train MRE 2.1870 %, test MRE 3.7437 %, RMSE 0.959 / 2.093, R 0.997 / 0.990.
Those match the paper's printed values to the digit. **That agreement is the
first piece of evidence: the dataset, the split and the metric code are all
correct.** Whatever is wrong is not there.

Full M1 metrics, both outputs:

| Split | Target | n | MRE % | MSE | RMSE | Pearson R | Max abs err |
|---|---|---|---|---|---|---|---|
| train | `theta_cone_deg` | 23 | 4.31 | 1.7804 | 1.3343 | 0.9955 | 3.4913 |
| train | `Rc_mm` | 23 | 3.12 | 0.6046 | 0.7776 | 0.9960 | 1.8875 |
| test | `theta_cone_deg` | 7 | 17.27 | 142.3297 | 11.9302 | 0.7133 | 30.2095 |
| test | `Rc_mm` | 7 | 8.33 | 3.2766 | 1.8102 | 0.9690 | 3.7692 |
| CV, out-of-fold (train rows) | `theta_cone_deg` | 23 | 10.41 | 14.4880 | 3.8063 | 0.9677 | 12.7450 |
| CV, out-of-fold (train rows) | `Rc_mm` | 23 | 8.10 | 5.7906 | 2.4064 | 0.9700 | 9.0028 |

Selected seed 100 of 10, epoch budget 72 (median stopping epoch of its own CV
folds), 72 accepted and 70 rejected LM steps.

`Rc_mm` is the placeholder second output from D3. It is a deterministic function
of the inputs and θ, so its 8.33 % is **not** independent evidence of anything
and must not be reported as a second physical prediction.

---

## 3. Where the test error is

| Case | P | r_w | C | θ true | θ M1 | error |
|---|---|---|---|---|---|---|
| 3 | 2.50 | 2.49 | 16.13 | 50.28 | 42.16 | −8.12 |
| 6 | 2.27 | 2.72 | 13.52 | 42.16 | 40.26 | −1.90 |
| **12** | **1.20** | **0.76** | **111.0** | **35.70** | **65.91** | **+30.21** |
| 15 | 1.01 | 2.00 | 25.00 | 30.00 | 30.17 | +0.17 |
| 17 | 0.74 | 1.23 | 66.10 | 29.04 | 25.30 | −3.74 |
| 26 | 0.29 | 1.46 | 46.91 | 17.00 | 17.29 | +0.29 |
| 29 | 0.40 | 0.84 | 10.61 | 12.70 | 12.76 | +0.06 |

**Case 12 is most of the error.** Drop it and test MRE falls from 17.27 % to
**6.04 %**. Four of the seven cases are predicted to better than 0.3°.

Case 12 sits at C = 111 with a low perveance. Of the 23 training rows, 21 have
C ≤ 51 and the other two are C = 207.6 and C = 306.3 — both high-θ guns (70.0°
and 57.3°). So the interval C ∈ [51, 306] is spanned by two training points,
both of which say "high C means high θ", and case 12 falls in the middle of that
gap while being a 35.7° gun. The network extrapolates the only trend it was
shown and returns 65.91°. The paper's own network got 36.09° here.

This is a data-coverage failure, not obviously a code failure.

---

## 4. Why we are not tuning until 3.74 % appears

Ten Nguyen–Widrow restarts, all scored post-hoc on the test set purely to show
the spread (this table was never used to choose anything):

| | min | median | max |
|---|---|---|---|
| test MRE % across 10 restarts (M1 config) | 8.10 | 16.63 | 41.60 |
| test MRE % across 30 restarts, n_out = 1 | 3.10 | 13.36 | — |

**The paper's 3.74 % lies inside our restart distribution, near its good tail.**
Some initialisations of this exact code land there. The question is whether any
honest procedure can *pick* those initialisations in advance, and the answer
measured here is no:

| Selection signal | Correlation with test MRE |
|---|---|
| 5-fold CV score within the 23 training rows, n_out = 1 | R = **+0.328** |
| 5-fold CV score within the 23 training rows, n_out = 2 | R = **+0.156** |

With 23 rows the cross-validation estimate is itself so noisy that it barely
ranks the restarts. Which restart "wins" is close to arbitrary — M1's selected
seed 100 scored best on CV and landed at 17.27 % on test, while other seeds
scoring worse on CV did better on test.

So a single test-MRE number on this split is not a measurement of the model. It
is a measurement of the model plus a draw from a wide lottery, and **the spread
is the honest result**. Reporting 3.74 % by trying seeds until one produced it
would be selecting on the test set: those seven guns are the entire reported
result, and touching them for any decision destroys it.

This is the same argument as D5, applied to initialisation instead of to a
physics coefficient.

---

## 5. Overfitting — the known limitation, stated up front

**26 parameters against 23 training rows is over-parameterised.** There are more
free parameters than data points. The network can, and does, interpolate the
training set exactly: run it to 5000 epochs with n_out = 1 and training MRE
reaches **0.00 %** while test MRE stays near 17 %. That is memorisation, not
learning.

`fig4_mse_vs_epochs.png` shows it directly. Training MSE falls monotonically;
the cross-validation curve bottoms out around epoch 50–93 and then climbs by
roughly a factor of four before plateauing. Everything after that minimum is the
network fitting noise.

What is done about it, and what it does not fix:

- Early stopping, via an epoch budget taken from the CV folds' own stopping
  epochs — this is why M1 runs 72 epochs rather than 5000.
- Ten restarts rather than one seed.
- Test metrics reported on real guns only.
- The `beam_type == "pencil"` filter, so no structurally different gun inflates
  the row count.

None of these create information. The real fix is more real guns or a validated
Tier B generator, and both are blocked on the same thing.

---

## 6. Do the residual weights matter?

They were expected to be a no-op: both targets are min-max scaled to [−1, 1]
before residuals are formed, so neither can dominate by magnitude. **That
expectation is wrong, and the reason is worth stating.** Median test MRE on θ
over 30 restarts, 200-epoch budget:

| Residual weights (θ / Rc) | Test MRE % median | min | max |
|---|---|---|---|
| 1.0 / 1.0 — default | 18.89 | 7.12 | 61.23 |
| 4.0 / 1.0 — θ favoured | 12.98 | 5.05 | 61.23 |
| 1.0 / 4.0 — Rc favoured | 28.80 | 6.55 | 176.95 |
| 1.0 / 0.0 — Rc switched off | 8.94 | 2.93 | 30.59 |

Equal scale does not mean equal claim on capacity. The two outputs share the
same 26 parameters and the same two-neuron second hidden layer, so weighting is
not just about magnitude — it decides how much of a very small network each
target gets. Given the spread in each row these medians carry real uncertainty,
but the ordering is consistent and the direction is clear.

**M1 ships at 1.0 / 1.0 anyway.** Choosing 1.0 / 0.0 because it scores best on
these seven test guns would be the same test-set selection the previous section
refuses. If the supervisor's answer to D3 keeps a second output, this table is
the argument for weighting it deliberately rather than by default.

## 7. What the second output costs

Median over 30 restarts, 200-epoch budget:

| | Parameters | Train MRE % median | Test MRE % median |
|---|---|---|---|
| n_out = 1, θ only | 23 | 2.80 | 13.36 |
| n_out = 2, θ + Rc_mm | 26 | 3.38 | 18.89 |

Carrying `Rc_mm` makes θ worse. That is the expected result and not an argument
against multi-output in general — it is an argument against *this* second
output, which by construction contains no information the first does not
(D3). If D3 resolves to reading A, `targets=["theta_cone_deg"]` recovers the
first row.

## 8. One diagnostic, deliberately not adopted

C spans 5.12 to 306.3, but 21 of 23 training rows sit below 51. After linear
min-max scaling almost every row lands near −0.9 and the feature loses
resolution exactly where case 12 falls. The one physically motivated
alternative is ln C — the physics enters through ln(R_c/R_a), and
`VALIDATION_table2.md` already found the closure error correlating with ln C at
R = +0.794.

| | Test MRE % median | min |
|---|---|---|
| n_out = 1, linear C (as shipped) | 13.36 | 3.10 |
| n_out = 1, ln C | 9.87 | 3.59 |
| n_out = 2, linear C (as shipped) | 18.89 | 7.12 |
| n_out = 2, ln C | 7.78 | 5.44 |

**Not adopted.** M1's job is to replicate the paper on the paper's own terms,
and the paper scales C linearly. Swapping preprocessing because it improves the
number on the seven reported test guns is fitting to the test set. Per D5,
sweeping one physically meaningful choice and reporting where it lands is a
diagnostic; adopting the winner is not. It is recorded here as a lead worth
following with a proper held-out set, which this dataset does not have.

---

## 9. What is actually verified

The failure above is specific, and these are not affected by it:

- **Analytic Jacobian** agrees with central finite differences to < 1e-6 across
  20 random parameter draws, n_out ∈ {1, 2, 3}, and both output activations
  (`tests/test_jacobian.py`).
- **LM optimizer** converges: it drives training MRE to 0.00 % on 23 rows with
  23 parameters, which is the correct behaviour for an over-parameterised fit.
- **Paper metrics reproduce exactly** from our extracted Table 2 columns.
- **Scaler** is fitted on training rows only and persisted, verified by test.
- **No gun appears in both splits** (`assert_no_test_leakage`).
- **Pencil-beam filter** holds, including against a planted intruder row.
- **Tier B is still empty**, asserted by test.

46 tests pass (`pytest`).

---

## 10. Summary for the supervisor

1. The pipeline is correct where it can be checked, and the paper's own numbers
   fall out of our data exactly.
2. M1 nonetheless reports 17.27 % test MRE against the paper's 3.74 %.
3. The dominant cause is that 30 rows cannot support 26 parameters: test MRE
   varies from 8 % to 42 % across initialisations, and no honest selection
   signal available within 23 rows can pick the good ones (R ≈ +0.16 to +0.33).
4. One test gun, case 12, contributes most of the error, and it sits in a gap in
   the training coverage of C.
5. M2 and M3 remain blocked on Tier B, which is blocked on the physics gate,
   which is blocked on one of three paywalled papers.

The open questions are unchanged: D3 (what the second output should be) and D4
(a clean PDF of Vaughan 1981, Tiwary & Basu 1987, or Yang 2006).
